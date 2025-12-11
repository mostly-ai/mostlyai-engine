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
import time

import numpy as np
import pandas as pd
import pytest
import torch

from mostlyai import engine
from mostlyai.engine import analyze, encode, generate, split
from mostlyai.engine._encoding_types.tabular.categorical import CATEGORICAL_UNKNOWN_TOKEN
from mostlyai.engine._encoding_types.tabular.lat_long import split_str_to_latlong
from mostlyai.engine._workspace import Workspace
from mostlyai.engine.domain import ModelEncodingType, RareCategoryReplacementMethod

from .conftest import MockData, train_from_workspace


@pytest.fixture(scope="module")
def input_data():
    n_train = 10_000
    n_samples = 10_000 * 2  # the second half is for TestTabularFlatWithContext
    mock_data = MockData(n_samples=n_samples)
    mock_data.add_index_column("id")
    mock_data.add_numeric_column(
        name="amount",
        quantiles={0.0: 17, 0.1: 22, 0.2: 26, 0.3: 30, 0.4: 33, 0.5: 37, 0.6: 41, 0.7: 45, 0.8: 50, 0.9: 58, 1.0: 90},
        dtype="int32",
    )
    # categorical column without rare values
    mock_data.add_categorical_column(
        name="product_type",
        probabilities={"A": 0.4, "B": 0.4, "C": 0.2},
    )
    # product_type_id has one-to-one mapping with product_type
    mock_data.df["product_type_id"] = mock_data.df["product_type"].map(ord)
    # categorical column with rare values
    mock_data.add_categorical_column(
        name="producer",
        probabilities={"A": 0.5, "B": 0.25, "C": 0.15, "D": 0.05, "E": 0.05},
        rare_categories=[f"Rare {i}" for i in range(20)],
    )

    # mock some random datetime values and replace year with {2025 - amount}
    mock_data.add_datetime_column(name="datetime", start_date="2025-01-01", end_date="2025-12-31")
    mock_data.df["datetime"] = mock_data.df.apply(
        lambda x: x["datetime"].replace(year=2025 - x["amount"]),
        axis=1,
    )
    mock_data.df["date"] = pd.to_datetime(mock_data.df["datetime"].dt.date)
    # set some datetime values to NaN
    mock_data.df.loc[mock_data.df.sample(n=int(0.05 * n_samples)).index, "datetime"] = np.nan

    # price correlates to amount and product_type
    product_type_multiplier = np.where(mock_data.df["product_type"] == "A", 0.7, 1.0)
    mock_data.df["price"] = (1000 / mock_data.df["amount"]) * product_type_multiplier
    mock_data.df["price_category"] = mock_data.df["price"].apply(lambda x: "Cheap" if x < 15 else "Expensive")
    # geo data with some invalid longitudes (which will be considered as NaN)
    mock_data.add_lat_long_column(name="geo", lat_limit=(22.0, 25.2), long_limit=(118.3, 121.7))
    mock_data.df.loc[mock_data.df.sample(n=int(0.05 * n_samples)).index, "geo"] = "25.0, XXX"
    return mock_data.df[:n_train], mock_data.df[n_train:].reset_index(drop=True)


class TestTabularFlatWithoutContext:
    @pytest.fixture(autouse=True)
    def cleanup(self, workspace_after_training):
        # make sure the synthetic data folder is clean before each test starts
        shutil.rmtree(workspace_after_training / "SyntheticData", ignore_errors=True)
        yield

    @pytest.fixture(scope="class")
    def workspace_after_training(self, input_data, tmp_path_factory):
        df = input_data[0]
        workspace_dir = tmp_path_factory.mktemp("workspace")
        tgt_encoding_types = {
            "amount": ModelEncodingType.tabular_numeric_auto,
            "product_type": ModelEncodingType.tabular_categorical,
            "producer": ModelEncodingType.tabular_categorical,
            "price": ModelEncodingType.tabular_numeric_auto,
            "price_category": ModelEncodingType.tabular_categorical,
        }
        split(
            tgt_data=df[["id"] + list(tgt_encoding_types.keys())],
            tgt_primary_key="id",
            tgt_encoding_types=tgt_encoding_types,
            n_partitions=10,
            workspace_dir=workspace_dir,
        )
        analyze(workspace_dir=workspace_dir)
        encode(workspace_dir=workspace_dir)
        train_from_workspace(workspace_dir, max_epochs=10, enable_flexible_generation=True)
        return workspace_dir

    def test_standard_generation(self, workspace_after_training):
        workspace_dir = workspace_after_training
        generate(workspace_dir=workspace_dir)
        orig = pd.read_parquet(workspace_dir / "OriginalData" / "tgt-data")
        syn = pd.read_parquet(workspace_dir / "SyntheticData")
        assert set(syn.columns) == set(orig.columns)
        assert syn.shape == orig.shape

    @pytest.mark.parametrize(
        "temperature, top_p, expected_match_ratio",
        [
            (10, 1.0, (0.5, 0.7)),  # creative sampling
            (0.5, 0.9, (0.9, 1.0)),  # conservative sampling
        ],
    )
    def test_sampling_temp_and_top_p(self, workspace_after_training, temperature, top_p, expected_match_ratio):
        workspace_dir = workspace_after_training
        generate(sampling_temperature=temperature, sampling_top_p=top_p, workspace_dir=workspace_dir)
        syn = pd.read_parquet(workspace_dir / "SyntheticData")
        match = (syn.loc[syn["price"] > 15, "price_category"] == "Expensive").mean()
        assert expected_match_ratio[0] <= match <= expected_match_ratio[1]

    def test_seed(self, workspace_after_training):
        workspace_dir = workspace_after_training
        seed_data = pd.DataFrame(
            {
                "id": [f"seed_{i:04d}" for i in range(5_000)],
                "product_type": ["A"] * 2500 + ["B"] * 2500,
                "amount": [10.0, 20.0, 30.0, 40.0, 50.0] * 1000,
            }
        )
        generate(seed_data=seed_data, workspace_dir=workspace_dir)
        syn = pd.read_parquet(workspace_dir / "SyntheticData")
        # the content of seed should remain in the synthetic data
        pd.testing.assert_frame_equal(seed_data, syn[["id", "product_type", "amount"]], check_dtype=False)
        syn_a = syn[syn["product_type"] == "A"].reset_index(drop=True)
        syn_b = syn[syn["product_type"] == "B"].reset_index(drop=True)
        # given the fixed amount values, the price of product A should be ~70% of the price of product B
        assert abs(1 - syn_b["price"] * 0.7 / syn_a["price"]).median() < 0.05

    def test_seed_special_cases(self, workspace_after_training):
        workspace_dir = workspace_after_training
        # case #1: empty seed
        seed_data = pd.DataFrame(columns=["product_type", "amount"])
        generate(seed_data=seed_data, workspace_dir=workspace_dir)
        syn = pd.read_parquet(workspace_dir / "SyntheticData")
        assert syn.empty

        # case #2: mismatch dtype + nonexistent column
        seed_data = pd.DataFrame(
            {
                "product_type": pd.Series(["A", "B", "C"], dtype="string"),
                "amount": pd.Series([20, 30, 40], dtype="string"),  # numeric column with string seeds
                "nonexistent_column": [404, 404, 404],
            }
        )
        generate(seed_data=seed_data, workspace_dir=workspace_dir)
        syn = pd.read_parquet(workspace_dir / "SyntheticData")
        # ensure columns unseen during training are ignored
        assert "nonexistent_column" not in syn.columns
        seed_data["amount"] = seed_data["amount"].astype("Int64")
        pd.testing.assert_frame_equal(seed_data[["product_type", "amount"]], syn[["product_type", "amount"]])

    def test_rebalancing(self, workspace_after_training):
        workspace_dir = workspace_after_training
        generate(
            rebalancing={"column": "producer", "probabilities": {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}},
            workspace_dir=workspace_dir,
        )
        syn = pd.read_parquet(workspace_dir / "SyntheticData")
        # the frequency of the producer should be close to the specified probabilities
        assert (syn["producer"].value_counts(normalize=True) - 0.2).mean() < 0.005

    @pytest.mark.parametrize(
        "target_column, sensitive_columns",
        [
            ("price_category", ["product_type"]),  # binary target, single sensitive column
            ("producer", ["product_type"]),  # multi-class target, single sensitive column
            ("producer", ["price_category", "product_type"]),  # multi-class target, multiple sensitive columns
        ],
    )
    def test_fairness(self, workspace_after_training, target_column, sensitive_columns):
        workspace_dir = workspace_after_training
        generate(
            fairness={"target_column": target_column, "sensitive_columns": sensitive_columns},
            workspace_dir=workspace_dir,
        )
        syn = pd.read_parquet(workspace_dir / "SyntheticData")
        # get distribution of target column for each group
        group_stats = syn.groupby(sensitive_columns)[target_column].value_counts(normalize=True)
        # use the first group as reference, check if other groups have similar distributions
        first_group, *other_groups = list(group_stats.index.droplevel(-1).unique())
        for group in other_groups:
            print(f"{first_group} vs. {group}: {abs(group_stats[first_group] - group_stats[group]).mean()}")
            assert abs(group_stats[first_group] - group_stats[group]).mean() < 0.05


def test_imputation(input_data, tmp_path_factory):
    df = input_data[0]
    workspace_dir = tmp_path_factory.mktemp("workspace-imputation")
    tgt_encoding_types = {
        "datetime": ModelEncodingType.tabular_datetime,
        "geo": ModelEncodingType.tabular_lat_long,
    }
    split(
        tgt_data=df[["id"] + list(tgt_encoding_types.keys())],
        tgt_primary_key="id",
        tgt_encoding_types=tgt_encoding_types,
        workspace_dir=workspace_dir,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    train_from_workspace(workspace_dir, max_epochs=1, enable_flexible_generation=True)
    generate(
        sample_size=1_000,
        rare_category_replacement_method=RareCategoryReplacementMethod.constant,
        imputation={"columns": list(tgt_encoding_types.keys())},
        workspace_dir=workspace_dir,
    )
    syn = pd.read_parquet(workspace_dir / "SyntheticData")
    for k in tgt_encoding_types.keys():
        assert syn[k].isna().mean() == 0


def test_zero_column(input_data, tmp_path_factory):
    df = input_data[0]
    workspace_dir = tmp_path_factory.mktemp("workspace-zero-column")
    split(
        tgt_data=df[["id"]].iloc[:1_000],
        tgt_primary_key="id",
        workspace_dir=workspace_dir,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    train_from_workspace(workspace_dir, max_epochs=1)
    generate(workspace_dir=workspace_dir)
    syn = pd.read_parquet(workspace_dir / "SyntheticData")
    assert syn.shape == (1_000, 1)
    assert "id" in syn.columns


@pytest.mark.flaky(reruns=3)
def test_seed_imputation(input_data, tmp_path_factory):
    """test that imputation strictly preserves correlations via conditional generation"""
    # constants
    workspace_dir = tmp_path_factory.mktemp("workspace-imputation-correlation")
    n_seed_samples = 20
    price_threshold = 15  # price < 15 -> "Cheap", price >= 15 -> "Expensive"
    null_probability = 0.5  # probability that a seed value is NULL (to be imputed)
    max_epochs = 10
    max_violation_rate = 0.05  # allow up to 5% of checks to fail for model approximation
    price_relative_tolerance = 0.05  # allow 5% relative error for price correlation
    imputed_cols = ["amount", "price", "price_category"]

    # train on data with strong price -> price_category correlation
    df_train = input_data[0]
    df_holdout = input_data[1]

    tgt_encoding_types = {
        "amount": ModelEncodingType.tabular_numeric_auto,
        "product_type": ModelEncodingType.tabular_categorical,
        "price": ModelEncodingType.tabular_numeric_auto,
        "price_category": ModelEncodingType.tabular_categorical,
    }

    split(
        tgt_data=df_train[["id"] + list(tgt_encoding_types.keys())],
        tgt_primary_key="id",
        tgt_encoding_types=tgt_encoding_types,
        workspace_dir=workspace_dir,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    train_from_workspace(workspace_dir, max_epochs=max_epochs, enable_flexible_generation=True)

    # create seed data from random samples in holdout set with random nulls for imputation testing
    # use holdout data (not seen during training) to avoid memorization
    seed_data = (
        df_holdout[["amount", "product_type", "price", "price_category"]]
        .sample(n=n_seed_samples)
        .reset_index(drop=True)
        .copy()
    )
    seed_data["amount"] = seed_data["amount"].astype("int32")

    # randomly set some values to None for imputation testing
    for col in imputed_cols:
        null_mask = np.random.rand(n_seed_samples) < null_probability
        seed_data.loc[null_mask, col] = None
    # randomly set one Nan to product_type column
    seed_data.loc[np.random.randint(0, n_seed_samples), "product_type"] = None

    generate(
        seed_data=seed_data,
        imputation={"columns": imputed_cols},
        workspace_dir=workspace_dir,
    )

    syn = pd.read_parquet(workspace_dir / "SyntheticData")
    assert len(syn) == n_seed_samples

    # verify seeded values are preserved (except for nulls in imputed columns)
    for idx in range(len(syn)):
        for col in seed_data.columns:
            seed_val = seed_data.loc[idx, col]
            if pd.notna(seed_val):
                syn_val = syn.loc[idx, col]
                assert pd.notna(syn_val), f"row {idx}, col '{col}': seed value {seed_val} not preserved (got NaN)"
                if isinstance(seed_val, str):
                    assert syn_val == seed_val, (
                        f"row {idx}, col '{col}': seed value {seed_val} not preserved, got {syn_val}"
                    )
                else:
                    # amount is integer, use exact equality
                    assert seed_val == syn_val, (
                        f"row {idx}, col '{col}': seed value {seed_val} not preserved, got {syn_val}"
                    )

    # verify correlations preserved in imputed values
    violations = []
    total_checks = 0

    for idx in range(len(syn)):
        amount = syn.loc[idx, "amount"]
        product_type = syn.loc[idx, "product_type"]
        price = syn.loc[idx, "price"]
        price_cat = syn.loc[idx, "price_category"]

        # ensure no NaN values after imputation
        if pd.isna(price):
            violations.append(f"row {idx}: price should not be NaN after imputation")
            continue
        if pd.isna(price_cat):
            violations.append(f"row {idx}: price_category should not be NaN after imputation")
            continue

        # skip correlation checks if amount is NaN (can't compute expected values)
        if pd.isna(product_type):
            continue

        # check amount <-> product_type <-> price correlation
        total_checks += 1
        product_type_multiplier = 0.7 if product_type == "A" else 1.0
        expected_price = (1000 / amount) * product_type_multiplier
        price_relative_error = abs(price - expected_price) / expected_price
        if price_relative_error > price_relative_tolerance:
            violations.append(
                f"row {idx}: amount={amount}, product_type={product_type} "
                f"should yield price≈{expected_price:.2f}, got {price:.2f} "
                f"(relative error: {price_relative_error:.1%})"
            )

        # check price <-> price_category correlation: price < 15 <-> "Cheap", else "Expensive"
        total_checks += 1
        expected_cat = "Cheap" if price < price_threshold else "Expensive"
        if price_cat != expected_cat:
            violations.append(f"row {idx}: price={price:.2f} should yield '{expected_cat}', got '{price_cat}'")

    assert len(violations) / max(total_checks, 1) <= max_violation_rate, (
        f"Violations ({len(violations)}/{total_checks}):\n" + "\n".join(violations)
    )


def test_value_protection_disabled(input_data, tmp_path_factory):
    def encode_train_generate(workspace_dir):
        ws = Workspace(workspace_dir)
        ws.model_progress_messages_path.unlink(missing_ok=True)
        encode(workspace_dir=workspace_dir)
        train_from_workspace(workspace_dir, model="MOSTLY_AI/Small", max_epochs=1)
        generate(workspace_dir=workspace_dir)
        syn = pd.read_parquet(workspace_dir / "SyntheticData")
        return syn

    df = input_data[0]
    workspace_dir = tmp_path_factory.mktemp("workspace-no-val-protection")
    split(
        tgt_data=df[["id"]].iloc[:1000],
        tgt_encoding_types={"id": ModelEncodingType.tabular_categorical},
        workspace_dir=workspace_dir,
    )
    # test that values protection is enabled by default
    analyze(workspace_dir=workspace_dir)
    syn = encode_train_generate(workspace_dir)
    assert syn["id"].unique().tolist() == [CATEGORICAL_UNKNOWN_TOKEN]

    # test that values protection is disabled when value_protection is set to False
    analyze(value_protection=False, workspace_dir=workspace_dir)
    syn = encode_train_generate(workspace_dir)
    assert len(syn["id"].unique()) > 1
    assert CATEGORICAL_UNKNOWN_TOKEN not in syn["id"]


@pytest.mark.parametrize("n_rows", [0, 1])
@pytest.mark.parametrize("n_cols", ["zero", "pk", "all"])
def test_emptish_flat(tmp_path_factory, input_data, n_rows, n_cols):
    if n_cols == "zero" and n_rows > 0:
        return  # skip impossible combinations
    workspace_dir = tmp_path_factory.mktemp("workspace-emptish-flat")
    pk_col = "id" if n_cols == "pk" else None
    tgt_encoding_types = {
        "product_type": ModelEncodingType.tabular_categorical,
        "amount": ModelEncodingType.tabular_numeric_digit,
        "date": ModelEncodingType.tabular_datetime,
    }
    cols = {
        "zero": [],
        "pk": [pk_col],
        "all": list(tgt_encoding_types.keys()),
    }[n_cols]
    tgt_encoding_types = {k: v for k, v in tgt_encoding_types.items() if k in cols}
    tgt = input_data[0][cols].iloc[:n_rows]
    split(
        tgt_primary_key=pk_col,
        tgt_data=tgt,
        tgt_encoding_types=tgt_encoding_types,
        workspace_dir=workspace_dir,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    train_from_workspace(workspace_dir, max_epochs=1)
    generate(workspace_dir=workspace_dir)
    syn_data_path = workspace_dir / "SyntheticData"
    syn = pd.read_parquet(syn_data_path)
    assert syn.shape == (n_rows, len(cols))


def test_max_training_time(input_data, tmp_path_factory):
    df = input_data[0][["id", "amount"]]
    workspace_dir = tmp_path_factory.mktemp("workspace-max-training-time")
    split(tgt_data=df, tgt_primary_key="id", workspace_dir=workspace_dir)
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    # measure training time vs max training time
    start_time = time.time()
    max_training_time_secs = 2
    train_from_workspace(
        workspace_dir,
        max_training_time=max_training_time_secs / 60,
        batch_size=8,
    )
    elapsed_time = time.time() - start_time
    # overhead time for saving model and computing val loss at the end of a very short (<1 epoch) training
    training_time_overhead = 6
    assert max_training_time_secs < elapsed_time < max_training_time_secs + training_time_overhead
    # check that model weights are saved
    workspace = Workspace(workspace_dir)
    assert workspace.model_tabular_weights_path.exists()
    # check that max_epochs is respected as well if both are provided
    start_time = time.time()
    train_from_workspace(workspace_dir, max_epochs=2, max_training_time=10)
    elapsed_time = time.time() - start_time
    assert elapsed_time < 10


def test_reproducibility(input_data, tmp_path_factory, cleanup_joblib_pool):
    df = input_data[0]
    ws_1 = tmp_path_factory.mktemp("ws_1")
    ws_2 = tmp_path_factory.mktemp("ws_2")

    def run_with_fixed_seed(ws):
        engine.set_random_state(42)
        split(tgt_data=df, workspace_dir=ws, tgt_primary_key="id")
        analyze(workspace_dir=ws)
        encode(workspace_dir=ws)
        train_from_workspace(ws, max_epochs=1)
        generate(workspace_dir=ws, sample_size=100)
        engine.set_random_state(None)

    run_with_fixed_seed(ws_1)
    run_with_fixed_seed(ws_2)

    def extract_artifacts(ws):
        return {
            "original_data": pd.read_parquet(ws / "OriginalData" / "tgt-data"),
            "stats": pd.read_json(ws / "ModelStore" / "tgt-stats" / "stats.json"),
            "encoded_data": pd.read_parquet(ws / "OriginalData" / "encoded-data"),
            "model_weights": torch.load(f=ws / "ModelStore" / "model-data" / "model-weights.pt", weights_only=True),
            "synthetic_data": pd.read_parquet(ws / "SyntheticData"),
        }

    ws_1_artifacts = extract_artifacts(ws_1)
    ws_2_artifacts = extract_artifacts(ws_2)

    assert ws_1_artifacts["original_data"].equals(ws_2_artifacts["original_data"]), (
        "reproducibility of split step failed"
    )
    assert ws_1_artifacts["stats"].equals(ws_2_artifacts["stats"]), "reproducibility of analyze step failed"
    assert ws_1_artifacts["encoded_data"].equals(ws_2_artifacts["encoded_data"]), (
        "reproducibility of encode step failed"
    )
    for key in ws_1_artifacts["model_weights"].keys():
        assert torch.equal(ws_1_artifacts["model_weights"][key], ws_2_artifacts["model_weights"][key]), (
            "reproducibility of train step failed"
        )
    assert ws_1_artifacts["synthetic_data"].equals(ws_2_artifacts["synthetic_data"]), (
        "reproducibility of generate step failed"
    )


def test_seed_generation_for_pk_only_flat_table(tmp_path_factory):
    ws = tmp_path_factory.mktemp("ws")
    tgt_primary_key = "id"
    df = pd.DataFrame({tgt_primary_key: range(1_000)})
    split(tgt_data=df, workspace_dir=ws, tgt_primary_key=tgt_primary_key)
    analyze(workspace_dir=ws)
    encode(workspace_dir=ws)
    train_from_workspace(ws, max_epochs=1)
    seed_data = pd.DataFrame({tgt_primary_key: [f"seed_{i:04d}" for i in range(100)]})
    generate(workspace_dir=ws, seed_data=seed_data)
    syn = pd.read_parquet(ws / "SyntheticData")
    pd.testing.assert_frame_equal(seed_data, syn)


class TestTabularFlatWithContext:
    @pytest.fixture(autouse=True)
    def cleanup(self, workspace_after_training):
        yield
        # clear generated data after each test
        shutil.rmtree(workspace_after_training / "SyntheticData", ignore_errors=True)

    @pytest.fixture(scope="class")
    def ctx_encoding_types(self):
        return {
            "date": ModelEncodingType.tabular_datetime,
            "product_type": ModelEncodingType.tabular_categorical,
        }

    @pytest.fixture(scope="class")
    def tgt_encoding_types(self):
        return {
            "datetime": ModelEncodingType.tabular_datetime,
            "product_type_id": ModelEncodingType.tabular_numeric_auto,
            "geo": ModelEncodingType.tabular_lat_long,
        }

    @pytest.fixture(scope="class")
    def workspace_after_training(self, tmp_path_factory, input_data, ctx_encoding_types, tgt_encoding_types):
        df = input_data[0]
        workspace_dir = tmp_path_factory.mktemp("workspace")
        ctx_df = df[["id"] + list(ctx_encoding_types.keys())]
        tgt_df = df[["id"] + list(tgt_encoding_types.keys())]
        # one context_id has seq_len=0, and one context_id has seq_len=2
        # due to privacy-protected sequence lengths, we can end up having to process sequential data as flat data
        tgt_df.loc[0, "id"] = "1"

        split(
            tgt_data=tgt_df,
            tgt_context_key="id",
            tgt_encoding_types=tgt_encoding_types,
            ctx_data=ctx_df,
            ctx_primary_key="id",
            ctx_encoding_types=ctx_encoding_types,
            n_partitions=4,
            workspace_dir=workspace_dir,
        )
        analyze(workspace_dir=workspace_dir)
        encode(workspace_dir=workspace_dir)
        train_from_workspace(workspace_dir, max_epochs=10, enable_flexible_generation=True)
        return workspace_dir

    def test_standard_generation(self, workspace_after_training, input_data, ctx_encoding_types, tgt_encoding_types):
        orig_tgt = input_data[0][["id"] + list(tgt_encoding_types.keys())]
        syn_ctx = input_data[1][["id"] + list(ctx_encoding_types.keys())]
        workspace_dir = workspace_after_training
        generate(ctx_data=syn_ctx, workspace_dir=workspace_dir)
        syn_tgt = pd.read_parquet(workspace_dir / "SyntheticData")

        match_producer = (syn_tgt.loc[syn_ctx["product_type"] == "A", "product_type_id"] == ord("A")).mean()
        assert match_producer > 0.95

        # quantiles of datetime should not be too far apart
        syn_time, orig_time = syn_tgt["datetime"], orig_tgt["datetime"]
        assert abs((syn_time.quantile(0.1) - orig_time.quantile(0.1)).days) < 365 * 3
        assert abs((syn_time.quantile(0.5) - orig_time.quantile(0.5)).days) < 365
        assert abs((syn_time.quantile(0.9) - orig_time.quantile(0.9)).days) < 365 * 3

        # check consistency between date and datetime
        assert (syn_ctx["date"].dt.year == syn_tgt["datetime"].dt.year).mean() > 0.9
        assert (syn_ctx["date"].dt.month == syn_tgt["datetime"].dt.month).mean() > 0.9
        assert (syn_ctx["date"].dt.day == syn_tgt["datetime"].dt.day).mean() > 0.9

        orig_latlong = split_str_to_latlong(orig_tgt["geo"])
        orig_latlong_mean = orig_latlong.dropna().astype(float).mean()
        syn_latlong = split_str_to_latlong(syn_tgt["geo"])
        syn_latlong_mean = syn_latlong.dropna().astype(float).mean()
        tgt_geo_na_ratio = (orig_latlong.isna().any(axis=1)).mean()
        syn_geo_na_ratio = (syn_latlong.isna().any(axis=1)).mean()
        assert abs(tgt_geo_na_ratio - syn_geo_na_ratio) < 0.02
        assert np.allclose(orig_latlong_mean.values, syn_latlong_mean.values, atol=0.1)
