# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import shutil

import numpy as np
import pandas as pd
import pytest

from mostlyai.engine.domain import ModelEncodingType, ModelStateStrategy, DifferentialPrivacyConfig
from mostlyai.engine import analyze, encode, split
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._common import read_json
from mostlyai.engine._encoding_types.tabular.categorical import (
    CATEGORICAL_NULL_TOKEN,
    CATEGORICAL_UNKNOWN_TOKEN,
)
from mostlyai.engine._tabular.generation import RareCategoryReplacementMethod, generate
from mostlyai.engine._tabular.training import train
from .conftest import MockData


def test_sequential_with_context(tmp_path_factory):
    n_samples = 2_000
    workspace_dir = tmp_path_factory.mktemp("workspace")
    mock_data = MockData(n_samples=n_samples)
    mock_data.add_index_column("customer_id")
    mock_data.add_categorical_column(
        name="country",
        probabilities={"AUT": 0.4, "USA": 0.6},
        rare_categories=["XXX"],
    )
    ctx_primary_key = "customer_id"
    ctx_encoding_types = {"country": ModelEncodingType.tabular_categorical}
    ctx_data = mock_data.df.copy()
    # the last 10 customers will have zero-length sequences
    mock_data.df = mock_data.df.iloc[:-10]

    # mock sequences of dates
    mock_data.add_date_column("purchase_date", "2000-01-01", "2025-12-31")
    mock_data.add_sequential_column(
        name="seq",
        seq_len_quantiles={0.0: 1, 0.1: 1, 0.2: 1, 0.3: 2, 0.4: 2, 0.5: 3, 0.6: 4, 0.7: 5, 0.8: 5, 0.9: 5, 1.0: 5},
    )
    mock_data.df["purchase_date"] = mock_data.df.apply(
        lambda x: x["purchase_date"] + pd.to_timedelta(int(x["seq"]) ** 2, unit="d"), axis=1
    )
    # set some dates to NaN
    mock_data.df.loc[mock_data.df.sample(n=100).index, "purchase_date"] = np.nan

    # customer from a certain country will only purchase products from that country
    mock_data.add_categorical_column(
        name="__product_aut",
        probabilities={"Energy drink": 0.5, "Spritzer": 0.3, "Sturm": 0.2},
        rare_categories=[f"X{i}" for i in range(100)],
    )
    mock_data.add_categorical_column(
        name="__product_usa",
        probabilities={"Coke": 0.5, "Cocktail": 0.3, "Iced tea": 0.2},
        rare_categories=[f"Y{i}" for i in range(100)],
    )
    mock_data.df.loc[mock_data.df["country"] == "AUT", "product"] = mock_data.df[mock_data.df["country"] == "AUT"][
        "__product_aut"
    ]
    mock_data.df.loc[mock_data.df["country"] == "USA", "product"] = mock_data.df[mock_data.df["country"] == "USA"][
        "__product_usa"
    ]
    mock_data.df.drop(columns=["__product_aut", "__product_usa"], inplace=True)

    mock_data.add_index_column("id")
    tgt_encoding_types = {
        "product": ModelEncodingType.tabular_categorical,
        "purchase_date": ModelEncodingType.tabular_datetime_relative,
    }
    tgt_primary_key = "id"
    tgt_context_key = "customer_id"
    tgt_data = mock_data.df[[tgt_primary_key, tgt_context_key] + list(tgt_encoding_types.keys())]
    split(
        tgt_data=tgt_data,
        tgt_primary_key=tgt_primary_key,
        tgt_context_key=tgt_context_key,
        tgt_encoding_types=tgt_encoding_types,
        ctx_data=ctx_data,
        ctx_primary_key=ctx_primary_key,
        ctx_encoding_types=ctx_encoding_types,
        workspace_dir=workspace_dir,
        n_partitions=10,
    )

    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    train(max_epochs=10, workspace_dir=workspace_dir)
    generate(
        workspace_dir=workspace_dir,
        rare_category_replacement_method=RareCategoryReplacementMethod.sample,
    )
    # check number of records within stats
    stats = read_json(workspace_dir / "ModelStore" / "tgt-stats" / "stats.json")
    assert stats["no_of_training_records"] + stats["no_of_validation_records"] == n_samples

    tgt_data_path = workspace_dir / "OriginalData" / "tgt-data"
    ctx_data_path = workspace_dir / "OriginalData" / "ctx-data"
    syn_data_path = workspace_dir / "SyntheticData"

    df_ctx = pd.read_parquet(ctx_data_path)
    tgt = pd.read_parquet(tgt_data_path).assign(__exists=True)
    tgt = df_ctx.merge(tgt, on=tgt_context_key, how="left")
    syn = pd.read_parquet(syn_data_path).assign(__exists=True)
    syn = df_ctx.merge(syn, on=tgt_context_key, how="left")

    # check sequence lengths
    tgt_seq_lens = tgt.groupby(tgt_context_key)["__exists"].count()
    syn_seq_lens = syn.groupby(tgt_context_key)["__exists"].count()
    assert np.abs(syn_seq_lens.mean() / tgt_seq_lens.mean() - 1) < 0.07
    # check for occurrence of zero-sequence lengths
    assert np.abs((syn_seq_lens == 0).mean() - (tgt_seq_lens == 0).mean()) < 0.01

    # check that rare category is not showing up in generated data
    assert CATEGORICAL_UNKNOWN_TOKEN not in syn["product"].unique()

    # check that special NULL category does not show up in generated data
    assert CATEGORICAL_NULL_TOKEN not in syn["product"].unique()

    # check coherence of `product`
    syn_avg_cat_per_customer = syn.groupby(tgt_context_key)["product"].nunique().mean()
    tgt_avg_cat_per_customer = tgt.groupby(tgt_context_key)["product"].nunique().mean()
    assert np.abs(syn_avg_cat_per_customer / tgt_avg_cat_per_customer - 1) < 0.1
    # ITT
    assert stats["columns"]["purchase_date"]["has_nan"] is True
    assert stats["columns"]["purchase_date"]["has_neg"] is False
    assert stats["columns"]["purchase_date"]["has_time"] is False
    assert pd.api.types.is_datetime64_any_dtype(syn["purchase_date"])
    # check that all ITTs are non-zero
    syn_date_itts = syn.groupby(tgt_context_key)["purchase_date"].diff().apply(lambda x: x.days)
    tgt_date_itts = tgt.groupby(tgt_context_key)["purchase_date"].diff().apply(lambda x: x.days)
    assert np.min(syn_date_itts) >= 0
    assert np.min(tgt_date_itts) >= 0
    # check median ITT is realistic
    assert np.abs(np.nanmedian(tgt_date_itts) - np.nanmedian(syn_date_itts)) < 0.1
    # check occurrence of missing dates
    assert np.abs(syn["purchase_date"].isna().mean() - tgt["purchase_date"].isna().mean()) < 0.01


def test_sequential_without_context(tmp_path_factory):
    n_samples = 2_000
    workspace_dir = tmp_path_factory.mktemp("workspace")
    mock_data = MockData(n_samples=n_samples)
    mock_data.add_index_column("id")
    ctx_data = mock_data.df.copy()
    mock_data.add_sequential_column(
        name="seq",
        seq_len_quantiles={0.0: 1, 0.1: 1, 0.2: 1, 0.3: 2, 0.4: 2, 0.5: 3, 0.6: 4, 0.7: 6, 0.8: 8, 0.9: 11, 1.0: 15},
    )
    tgt_data = mock_data.df
    # test single categorical column
    split(
        tgt_data=tgt_data,
        tgt_context_key="id",
        tgt_encoding_types={"seq": ModelEncodingType.tabular_categorical},
        workspace_dir=workspace_dir,
    )
    tgt_data_path = workspace_dir / "OriginalData" / "tgt-data"
    syn_data_path = workspace_dir / "SyntheticData"
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    train(max_epochs=10, workspace_dir=workspace_dir)
    generate(ctx_data=ctx_data, workspace_dir=workspace_dir)
    tgt = pd.read_parquet(tgt_data_path)
    syn = pd.read_parquet(syn_data_path)
    assert "id" in syn.columns and "seq" in syn.columns
    assert syn["id"].nunique() == n_samples
    syn_seq_lens = syn.groupby("id").size()
    tgt_seq_lens = tgt.groupby("id").size()
    assert abs(1 - syn_seq_lens.mean() / tgt_seq_lens.mean()) < 0.1

    # test zero columns
    shutil.rmtree(workspace_dir / "SyntheticData", ignore_errors=True)
    split(
        tgt_data=tgt_data[["id"]],
        tgt_context_key="id",
        tgt_encoding_types={},
        workspace_dir=workspace_dir,
    )
    tgt_data_path = workspace_dir / "OriginalData" / "tgt-data"
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    train(max_epochs=10, workspace_dir=workspace_dir)
    generate(ctx_data=ctx_data, workspace_dir=workspace_dir)
    syn_data_path = workspace_dir / "SyntheticData"
    tgt = pd.read_parquet(tgt_data_path)
    syn = pd.read_parquet(syn_data_path)
    assert "id" in syn.columns
    assert syn["id"].nunique() == n_samples
    syn_seq_lens = syn.groupby("id").size()
    tgt_seq_lens = tgt.groupby("id").size()
    assert abs(1 - syn_seq_lens.mean() / tgt_seq_lens.mean()) < 0.1


def test_sequential_zero_length_subjects_only(tmp_path):
    # prepare data
    # each subject has no sequences
    ctx = pd.DataFrame({"id": list(range(100)), "flat_numerics": list(range(100))})
    ws_dir = tmp_path / "ws"

    split(
        tgt_data=pd.DataFrame({"cid": [], "seq_numerics": []}),
        tgt_context_key="cid",
        ctx_data=ctx,
        ctx_primary_key="id",
        ctx_encoding_types={"flat_numerics": ModelEncodingType.tabular_numeric_digit},
        workspace_dir=ws_dir,
    )
    analyze(workspace_dir=ws_dir)
    encode(workspace_dir=ws_dir)
    train(max_epochs=1, model="MOSTLY_AI/Small", workspace_dir=ws_dir)
    generate(sample_size=2_000, workspace_dir=ws_dir)
    syn_data_path = ws_dir / "SyntheticData"

    # assert empty outcome
    df = pd.read_parquet(syn_data_path)
    assert df.empty
    assert df.columns.tolist() == ["cid", "seq_numerics"]


def test_many_partitions_one_empty(tmp_path):
    # data must be big enough to trigger partitioning
    key = "col_0"
    ctx = pd.DataFrame({f"col_{n}": list(range(1_000)) for n in range(3)})
    # only one tgt partition is not empty
    tgt = pd.DataFrame(
        {
            key: [0],
            "cat": ["a"],
            "char": ["a"],
            "num": [1],
            "dt": [pd.to_datetime("2020-01-01")],
        }
    )
    tgt_encoding_types = {
        "cat": ModelEncodingType.tabular_categorical,
        "char": ModelEncodingType.tabular_character,
        "num": ModelEncodingType.tabular_numeric_digit,
        "dt": ModelEncodingType.tabular_datetime,
    }
    ctx_encoding_types = {f"col_{n}": ModelEncodingType.tabular_numeric_digit for n in range(1, 3)}

    split(
        tgt_data=tgt,
        tgt_context_key=key,
        tgt_encoding_types=tgt_encoding_types,
        ctx_data=ctx,
        ctx_primary_key=key,
        ctx_encoding_types=ctx_encoding_types,
        workspace_dir=tmp_path,
        n_partitions=3,
    )
    analyze(workspace_dir=tmp_path)
    encode(workspace_dir=tmp_path)
    train(max_epochs=1, model="MOSTLY_AI/Small", workspace_dir=tmp_path)
    generate(sample_size=2_000, workspace_dir=tmp_path)
    syn_data_path = tmp_path / "SyntheticData"

    df = pd.read_parquet(syn_data_path)
    assert df.columns.tolist() == ["col_0", "cat", "char", "num", "dt"]


def test_sequential_max_sequence_window(tmp_path):
    # create dummy data with fixed A/B/A/B pattern
    df = pd.DataFrame({"id": np.repeat(list(range(1000)), 99), "cat": ["A", "B"] * int(99 * 1000 / 2)})
    split(tgt_data=df, tgt_context_key="id", workspace_dir=tmp_path)
    analyze(workspace_dir=tmp_path)
    encode(workspace_dir=tmp_path)
    train(max_epochs=20, max_sequence_window=3, batch_size=64, workspace_dir=tmp_path)
    generate(workspace_dir=tmp_path)
    syn = pd.read_parquet(tmp_path / "SyntheticData")
    # check whether we learned the A/B/A/B pattern
    acc = (syn.groupby("id").cat.shift() != syn.cat).mean()
    assert acc > 0.9


def test_sequential_gen_ctx_and_size(tmp_path):
    workspace_dir = tmp_path / "ws"
    key_column = "id"
    ctx = pd.DataFrame({key_column: list(range(500))})
    tgt = pd.DataFrame({key_column: np.repeat(list(range(500)), 2), "col": ["abcd"] * 1000})

    split(
        tgt_data=tgt,
        tgt_context_key=key_column,
        tgt_encoding_types={"col": ModelEncodingType.tabular_categorical},
        ctx_data=ctx,
        ctx_primary_key=key_column,
        ctx_encoding_types={},
        workspace_dir=workspace_dir,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)

    # ensure context big enough to trigger more than 1 batch generation
    ctx_data = pd.DataFrame({key_column: list(range(10_000))})

    test_cases = {
        "0 < sample_size <= len(ctx_data)": {
            "sample_size_in": len(ctx_data) - len(ctx_data) // 2,
            "sample_size_out": len(ctx_data) - len(ctx_data) // 2,
        },
        "len(ctx_data) < sample_size": {
            "sample_size_in": len(ctx_data) + len(ctx_data) // 2,
            "sample_size_out": len(ctx_data),
        },
    }

    for test_case in test_cases.values():
        generate(
            ctx_data=ctx_data,
            sample_size=test_case["sample_size_in"],
            workspace_dir=workspace_dir,
        )
        syn_data_path = workspace_dir / "SyntheticData"
        syn = pd.read_parquet(syn_data_path)
        # assert that syn has first "sample_size_out" ids of gen_ctx
        assert set(ctx_data.head(test_case["sample_size_out"])["id"]) == set(syn["id"])


def test_multiple_batches(tmp_path):
    # this test is to ensure that we can generate quality data in multiple batches
    workspace_dir = tmp_path / "ws"
    key = "id"
    trn_sample_size = 1_000
    ctx = pd.DataFrame({key: list(range(trn_sample_size))})
    tgt = []
    for k in ctx[key]:
        start = np.random.randint(3, 7)
        length = np.random.randint(3, 7)
        tgt.append(pd.DataFrame({key: [k] * length, "seq": list(range(start, start + length))}))
    tgt = pd.concat(tgt, axis=0)

    split(
        tgt_data=tgt,
        tgt_context_key=key,
        tgt_encoding_types={
            "seq": ModelEncodingType.tabular_numeric_digit,
        },
        ctx_data=ctx,
        ctx_primary_key=key,
        ctx_encoding_types={},
        workspace_dir=workspace_dir,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    train(max_epochs=5, workspace_dir=workspace_dir)
    generate(
        batch_size=trn_sample_size // 3,
        workspace_dir=workspace_dir,
    )

    # check that sequences are increasing by one
    syn_data_path = workspace_dir / "SyntheticData"
    syn = pd.read_parquet(syn_data_path)
    syn_diff = syn.groupby(key)["seq"].diff().mean()
    assert 0.8 < syn_diff < 1.2


class TestExtremeSequenceLengths:
    @pytest.mark.parametrize(
        "length_counts,expected_shape",
        [
            # SequentialModel generating zero-length sequences
            ({0: 100}, (0, 2)),
            # SequentialModel generating zero-length sequences (because of extreme sequence length protection)
            ({0: 100} | {i: 1 for i in range(1, 6)}, (0, 2)),
            # FlatModel (because of extreme sequence length protection)
            ({0: 10}, (10, 2)),
            ({i: 1 for i in range(10)}, (10, 2)),
            ({0: 5, 1: 100, 2: 5}, (110, 2)),
        ],
    )
    def test_extreme_sequence_lengths(self, tmp_path, length_counts, expected_shape):
        ctx = []
        tgt = []
        pk = 0
        for length, count in length_counts.items():
            for _ in range(count):
                ctx.append([pk, "ctx"])
                tgt.extend([[pk, "tgt"]] * length)
                pk += 1
        ctx = pd.DataFrame(ctx, columns=["id", "ctx"])
        tgt = pd.DataFrame(tgt, columns=["id", "tgt"])
        workspace_dir = tmp_path / "ws"
        split(
            tgt_data=tgt,
            tgt_context_key="id",
            ctx_data=ctx,
            ctx_primary_key="id",
            workspace_dir=workspace_dir,
        )
        analyze(workspace_dir=workspace_dir)
        encode(workspace_dir=workspace_dir)
        train(
            max_epochs=1,
            model="MOSTLY_AI/Small",
            workspace_dir=workspace_dir,
        )
        generate(workspace_dir=workspace_dir)
        syn_data_path = workspace_dir / "SyntheticData"
        syn = pd.read_parquet(syn_data_path)
        assert syn.shape == expected_shape


def test_itt_constant(tmp_path):
    # create temp paths
    workspace_dir = tmp_path / "ws"
    # generate training data with constant ITT of 1sec
    no_of_subjects = 30
    seq_len = 6
    start_dates = pd.to_datetime("2020-01-01 13:00:00") + pd.to_timedelta(range(no_of_subjects), unit="m")
    tgt = pd.DataFrame(
        {
            "id": np.repeat(range(no_of_subjects), seq_len),
            "ts": np.repeat(start_dates, seq_len),
        }
    )
    tgt["ts"] = tgt["ts"] + pd.to_timedelta(np.tile(np.array(range(seq_len)), no_of_subjects), unit="s")
    ctx = tgt[["id"]].drop_duplicates()

    split(
        tgt_data=tgt,
        tgt_context_key="id",
        tgt_encoding_types={"ts": ModelEncodingType.tabular_datetime_relative},
        workspace_dir=workspace_dir,
        ctx_data=ctx,
        ctx_primary_key="id",
        ctx_encoding_types={},
    )
    tgt_data_path = workspace_dir / "OriginalData" / "tgt-data"
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    train(workspace_dir=workspace_dir)
    generate(workspace_dir=workspace_dir)
    syn_data_path = workspace_dir / "SyntheticData"
    tgt = pd.read_parquet(tgt_data_path)
    syn = pd.read_parquet(syn_data_path)
    assert syn.shape == tgt.shape
    assert (syn.groupby("id").ts.diff().dt.total_seconds().dropna() == 1).mean() == 1


class TestSmokeModelSizes:
    @pytest.fixture(scope="class")
    def before_training(self, tmp_path_factory):
        workspace_dir = tmp_path_factory.mktemp("ws")
        cats = ["a", "b", "c"]
        fltctx = pd.DataFrame({"id": list(range(100)), "flt": np.random.choice(cats, 100)})
        seqctx = (
            pd.DataFrame(
                {
                    "id": np.repeat(list(range(100)), 2),
                    "seq": np.random.choice(cats, 100 * 2),
                }
            )
            .groupby("id")
            .agg(list)
        )
        ctx = fltctx.merge(seqctx, on="id")
        tgt = pd.DataFrame(
            {
                "id": np.repeat(list(range(100)), 2),
                "seq": np.random.choice(cats, 100 * 2),
            }
        )
        split(
            tgt_data=tgt,
            tgt_context_key="id",
            workspace_dir=workspace_dir,
            ctx_data=ctx,
            ctx_primary_key="id",
        )
        analyze(workspace_dir=workspace_dir)
        encode(workspace_dir=workspace_dir)
        return workspace_dir

    @pytest.mark.parametrize("model_id", ["MOSTLY_AI/Small", "MOSTLY_AI/Medium", "MOSTLY_AI/Large"])
    def test_smoke_model_sizes(self, before_training, model_id):
        # this is a smoke test to ensure that we don't crush on different model sizes
        train(model=model_id, max_epochs=1, workspace_dir=before_training)
        generate(workspace_dir=before_training)


class TestTabularTrainingStrategy:
    @pytest.fixture(scope="class")
    def workspace_before_training(self, tmp_path_factory):
        workspace_dir = tmp_path_factory.mktemp("ws")
        cats = ["a", "b", "c"]
        fltctx = pd.DataFrame({"id": list(range(100)), "flt": np.random.choice(cats, 100)})
        seqctx = (
            pd.DataFrame(
                {
                    "id": np.repeat(list(range(100)), 2),
                    "seq": np.random.choice(cats, 100 * 2),
                }
            )
            .groupby("id")
            .agg(list)
        )
        ctx = fltctx.merge(seqctx, on="id")
        tgt = pd.DataFrame(
            {
                "id": np.repeat(list(range(100)), 2),
                "seq": np.random.choice(cats, 100 * 2),
            }
        )
        split(
            tgt_data=tgt,
            tgt_context_key="id",
            workspace_dir=workspace_dir,
            ctx_data=ctx,
            ctx_primary_key="id",
        )
        return workspace_dir

    @pytest.mark.parametrize(
        "differential_privacy",
        [
            None,  # DP is disabled
            DifferentialPrivacyConfig(),  # DP is enabled with default parameters
        ],
    )
    def test_training_strategy(self, workspace_before_training, differential_privacy):
        model_id = "MOSTLY_AI/Small"
        workspace = Workspace(workspace_before_training)
        analyze(workspace_dir=workspace_before_training, differential_privacy=differential_privacy)
        encode(workspace_dir=workspace_before_training)
        train(
            workspace_dir=workspace_before_training,
            model=model_id,
            max_epochs=1,
            differential_privacy=differential_privacy,
            model_state_strategy=ModelStateStrategy.reset,
        )
        progress_reset = pd.read_csv(workspace.model_progress_messages_path)

        train(
            workspace_dir=workspace_before_training,
            model=model_id,
            max_epochs=1,
            differential_privacy=differential_privacy,
            model_state_strategy=ModelStateStrategy.reuse,
        )
        progress_reuse = pd.read_csv(workspace.model_progress_messages_path)
        assert not progress_reuse["epoch"].duplicated().any()
        # progress should be different but with the same shape
        assert progress_reset.shape == progress_reuse.shape
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(progress_reset, progress_reuse)

        train(
            workspace_dir=workspace_before_training,
            model=model_id,
            max_epochs=2,
            differential_privacy=differential_privacy,
            model_state_strategy=ModelStateStrategy.resume,
        )
        progress_resume = pd.read_csv(workspace.model_progress_messages_path)
        assert not progress_resume["epoch"].duplicated().any()
        # training resumed from epoch 1 and only appended a new line for epoch 2
        # so the progress should be identical except for the last row
        pd.testing.assert_frame_equal(progress_reuse, progress_resume.iloc[:-1])

        # in case the checkpoint doesn't exist, it should still work but change to reset strategy
        shutil.rmtree(workspace_before_training / "ModelStore" / "model-data")
        train(
            workspace_dir=workspace_before_training,
            model=model_id,
            max_epochs=1,
            differential_privacy=differential_privacy,
            model_state_strategy=ModelStateStrategy.resume,
        )
        progress_resume_without_checkpoint = pd.read_csv(workspace.model_progress_messages_path)
        # it's actually a fresh training, so the progress will look different
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(progress_resume.iloc[:2], progress_resume_without_checkpoint.iloc[:2])
