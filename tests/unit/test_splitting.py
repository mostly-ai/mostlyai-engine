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

import os

import pandas as pd
import pytest

from mostlyai.engine._common import read_json
from mostlyai.engine.domain import ModelEncodingType, ModelType
from mostlyai.engine.splitting import split

FIXTURES_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"))


@pytest.mark.parametrize("has_pk", [True, False])
def test_split_tgt(tmp_path, has_pk):
    tgt_df = pd.DataFrame(
        {
            "id": list(range(100)),
            "int": [10, 20] * 50,
            "cat": ["A", "B"] * 50,
        }
    )
    tgt_types = {
        "int": ModelEncodingType.tabular_numeric_binned,
    }
    split(
        tgt_data=tgt_df,
        tgt_encoding_types=tgt_types,
        tgt_primary_key="id" if has_pk else None,
        workspace_dir=tmp_path,
    )
    tgt_data_path = tmp_path / "OriginalData" / "tgt-data"
    tgt_meta_path = tmp_path / "OriginalData" / "tgt-meta"
    ctx_data_path = tmp_path / "OriginalData" / "ctx-data"
    ctx_meta_path = tmp_path / "OriginalData" / "ctx-meta"
    assert tgt_data_path.exists()
    assert tgt_meta_path.exists()
    assert not ctx_data_path.exists()
    assert not ctx_meta_path.exists()
    # check tgt-data
    out_df = pd.read_parquet(tgt_data_path)
    assert len(out_df) == len(tgt_df)
    assert set(out_df.columns) == set(tgt_df.columns)
    # check tgt-meta
    out_types = read_json(tgt_meta_path / "encoding-types.json")
    out_keys = read_json(tgt_meta_path / "keys.json")
    assert out_types["int"] == tgt_types["int"]
    assert out_types["cat"] == ModelEncodingType.tabular_categorical
    if has_pk:
        assert "id" not in out_types
        assert out_keys == {"primary_key": "id"}
    else:
        assert "id" in out_types
        assert out_keys == {}


def test_split_tgt_context_key(tmp_path):
    tgt_df = pd.DataFrame(
        {
            "fk": list(range(10)) * 10,
            "int": [10, 20] * 50,
            "cat": ["A", "B"] * 50,
        }
    )
    split(
        tgt_data=tgt_df,
        tgt_context_key="fk",
        workspace_dir=tmp_path,
    )
    tgt_data_path = tmp_path / "OriginalData" / "tgt-data"
    tgt_meta_path = tmp_path / "OriginalData" / "tgt-meta"
    ctx_data_path = tmp_path / "OriginalData" / "ctx-data"
    ctx_meta_path = tmp_path / "OriginalData" / "ctx-meta"
    assert tgt_data_path.exists()
    assert tgt_meta_path.exists()
    assert ctx_data_path.exists()
    assert ctx_meta_path.exists()
    # check tgt-data
    out_df = pd.read_parquet(tgt_data_path)
    assert len(out_df) == len(tgt_df)
    assert set(out_df.columns) == set(tgt_df.columns)
    # check ctx-data
    out_ctx_df = pd.read_parquet(ctx_data_path)
    assert len(out_ctx_df) == 10


def test_split_tgt_and_ctx(tmp_path):
    tgt_df = pd.DataFrame(
        {
            "id": list(range(100)),
            "fk": list(range(10)) * 10,
            "int": [10, 20] * 50,
            "cat": ["A", "B"] * 50,
        }
    )
    ctx_df = pd.DataFrame(
        {
            "pk": list(range(10)),
            "int": [10, 20] * 5,
            "cat": ["A", "B"] * 5,
        }
    )
    split(
        tgt_data=tgt_df,
        ctx_data=ctx_df,
        tgt_primary_key="id",
        tgt_context_key="fk",
        ctx_primary_key="pk",
        n_partitions=3,
        workspace_dir=tmp_path,
    )
    tgt_data_path = tmp_path / "OriginalData" / "tgt-data"
    tgt_meta_path = tmp_path / "OriginalData" / "tgt-meta"
    ctx_data_path = tmp_path / "OriginalData" / "ctx-data"
    ctx_meta_path = tmp_path / "OriginalData" / "ctx-meta"
    assert tgt_data_path.exists()
    assert tgt_meta_path.exists()
    assert ctx_data_path.exists()
    assert ctx_meta_path.exists()
    # check single partition
    out_tgt_df2 = pd.read_parquet(tgt_data_path / "part.000002-trn.parquet")
    out_ctx_df2 = pd.read_parquet(ctx_data_path / "part.000002-trn.parquet")
    assert set(out_ctx_df2["pk"]) == set(out_tgt_df2["fk"])
    # check partitions
    tgt_pqt_paths = list(tgt_data_path.glob("*.parquet"))
    ctx_pqt_paths = list(ctx_data_path.glob("*.parquet"))
    assert len(ctx_pqt_paths) == 6  # 3 partitions * (train + val) = 6
    assert len(tgt_pqt_paths) == 6
    # check complete data
    out_tgt_df = pd.read_parquet(tgt_data_path)
    out_ctx_df = pd.read_parquet(ctx_data_path)
    assert len(out_tgt_df) == len(tgt_df)
    assert len(out_ctx_df) == len(ctx_df)
    assert set(out_ctx_df["pk"]) == set(out_tgt_df["fk"])
    # check meta
    out_tgt_types = read_json(tgt_meta_path / "encoding-types.json")
    out_tgt_keys = read_json(tgt_meta_path / "keys.json")
    out_ctx_types = read_json(ctx_meta_path / "encoding-types.json")
    out_ctx_keys = read_json(ctx_meta_path / "keys.json")
    assert out_ctx_keys == {"primary_key": "pk"}
    assert out_tgt_keys == {"primary_key": "id", "context_key": "fk"}
    assert out_tgt_types == {
        "int": ModelEncodingType.tabular_numeric_auto.value,
        "cat": ModelEncodingType.tabular_categorical.value,
    }
    assert out_ctx_types == {
        "int": ModelEncodingType.tabular_numeric_auto.value,
        "cat": ModelEncodingType.tabular_categorical.value,
    }


def test_split_model_type_no_encoding_types(tmp_path):
    tgt_df = pd.DataFrame(
        {
            "int": [10, 20] * 50,
            "cat": ["A", "B"] * 50,
        }
    )
    # test TABULAR
    split(
        tgt_data=tgt_df,
        model_type=ModelType.tabular,
        workspace_dir=tmp_path,
    )
    out_types = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
    assert out_types == {
        "int": ModelEncodingType.tabular_numeric_auto.value,
        "cat": ModelEncodingType.tabular_categorical.value,
    }
    # test LANGUAGE
    split(
        tgt_data=tgt_df,
        model_type=ModelType.language,
        workspace_dir=tmp_path,
    )
    out_types = read_json(tmp_path / "OriginalData" / "tgt-meta" / "encoding-types.json")
    assert out_types == {
        "int": ModelEncodingType.language_numeric.value,
        "cat": ModelEncodingType.language_categorical.value,
    }


def test_split_mixed_tgt_raises(tmp_path):
    tgt_df = pd.DataFrame(
        {
            "int": [10, 20] * 50,
            "cat": ["A", "B"] * 50,
        }
    )
    with pytest.raises(Exception):
        split(
            tgt_data=tgt_df,
            tgt_encoding_types={
                "int": ModelEncodingType.tabular_numeric_binned,
                "cat": ModelEncodingType.language_text,
            },
        )
    with pytest.raises(Exception):
        split(
            tgt_data=tgt_df,
            tgt_encoding_types={
                "int": ModelEncodingType.tabular_numeric_binned,
                "cat": ModelEncodingType.tabular_categorical,
            },
            model_type=ModelType.language,
        )


@pytest.mark.parametrize("trn_val_split", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("tgt_pk", [None, "id"])
def test_trn_val_split(tmp_path, trn_val_split, tgt_pk):
    n = 10_000
    tgt_df = pd.DataFrame({"id": list(range(n))})
    split(
        tgt_data=tgt_df,
        tgt_primary_key=tgt_pk,
        trn_val_split=trn_val_split,
        workspace_dir=tmp_path,
    )
    trn_tgt_data = pd.read_parquet(list((tmp_path / "OriginalData" / "tgt-data").glob("*trn*")))
    val_tgt_data = pd.read_parquet(list((tmp_path / "OriginalData" / "tgt-data").glob("*val*")))
    assert len(trn_tgt_data) == pytest.approx(int(n * trn_val_split), rel=0.2)
    assert len(val_tgt_data) == pytest.approx(int(n * (1 - trn_val_split)), rel=0.2)
