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

import numpy as np
import pandas as pd

from mostlyai.engine._common import ANALYZE_MIN_MAX_TOP_N, read_json, write_json
from mostlyai.engine.analysis import (
    _analyze_col,
    _analyze_partition,
    _analyze_reduce_seq_len,
    _analyze_seq_len,
)
from mostlyai.engine.domain import ModelEncodingType


def test_analyze_cnt(tmp_path):
    events = pd.DataFrame({"context_key": [1, 1, 1, 2, 3, 4], "x": [1, 1, 1, 1, 1, 1]})
    no_of_records = events["context_key"].nunique()

    events.to_parquet(tmp_path / "part.000000-trn.parquet")
    stats_path = tmp_path / "stats"
    stats_path.mkdir()
    _analyze_partition(
        tmp_path / "part.000000-trn.parquet",
        stats_path,
        tgt_context_key="context_key",
        tgt_encoding_types={"x": ModelEncodingType.tabular_numeric_digit},
    )
    stats = read_json(stats_path / "part.000000-trn.json")
    assert stats["no_of_training_records"] == no_of_records
    assert stats["no_of_validation_records"] == 0


def test_analyze_seq_len(tmp_path):
    tgt_context_keys = pd.Series(np.repeat(range(21), range(21)), name="account_id")
    partition_stats = _analyze_seq_len(tgt_context_keys=tgt_context_keys, ctx_primary_keys=tgt_context_keys)
    write_json(partition_stats, tmp_path / "stats1.json")
    partition_stats = read_json(tmp_path / "stats1.json")
    global_stats = _analyze_reduce_seq_len([partition_stats])
    write_json(global_stats, tmp_path / "stats.json")
    global_stats = read_json(tmp_path / "stats.json")
    assert global_stats["max"] >= 12 and global_stats["max"] <= 15
    assert isinstance(global_stats["max"], int)
    global_stats = _analyze_reduce_seq_len([partition_stats for i in range(9)])
    assert global_stats["max"] == 20

    tgt_context_keys = pd.Series(np.repeat(range(21), range(21)), name="account_id")
    partition_stats = _analyze_seq_len(
        tgt_context_keys=tgt_context_keys,
        ctx_primary_keys=pd.concat([tgt_context_keys, pd.Series([100])]),
    )
    global_stats = _analyze_reduce_seq_len([partition_stats], value_protection=False)
    assert global_stats["min"] == 0
    assert global_stats["max"] == 20


def test_analyze_root_key(tmp_path):
    tgt_context_keys = pd.Series(np.repeat(range(40), 2), name="tgt_context_key")
    tgt_values = pd.Series(list(range(80)), name="tgt_values")
    tgt = pd.concat([tgt_context_keys, tgt_values], axis=1)

    ctx_root_keys = pd.Series(np.repeat(range(20), 2), name="ctx_root_keys")
    ctx_primary_keys = pd.Series(list(range(40)), name="ctx_primary_key")
    ctx_values = pd.Series(list(range(40)), name="ctx_values")
    ctx = pd.concat([ctx_root_keys, ctx_primary_keys, ctx_values], axis=1)

    tgt_partition_path, ctx_partition_path = (
        tmp_path / "tgt.000000-trn.parquet",
        tmp_path / "ctx.000000-trn.parquet",
    )
    tgt.to_parquet(tgt_partition_path), ctx.to_parquet(ctx_partition_path)

    tgt_stats_path, ctx_stats_path = tmp_path / "tgt_stats", tmp_path / "ctx_stats"
    tgt_stats_path.mkdir(), ctx_stats_path.mkdir()

    # root key column is in tgt table
    _analyze_partition(
        tgt_partition_file=tgt_partition_path,
        tgt_stats_path=tgt_stats_path,
        tgt_encoding_types={tgt_values.name: ModelEncodingType.tabular_numeric_digit},
        tgt_context_key=tgt_context_keys.name,
        ctx_partition_file=ctx_partition_path,
        ctx_stats_path=ctx_stats_path,
        ctx_encoding_types={ctx_values.name: ModelEncodingType.tabular_numeric_digit},
        ctx_primary_key=ctx_primary_keys.name,
        ctx_root_key=ctx_root_keys.name,
    )
    ctx_stats = read_json(ctx_stats_path / "part.000000-trn.json")
    assert ctx_stats["columns"][ctx_values.name]["max_n"] == list(range(40))[::-2][:ANALYZE_MIN_MAX_TOP_N]
    assert ctx_stats["columns"][ctx_values.name]["min_n"] == list(range(40))[::2][:ANALYZE_MIN_MAX_TOP_N]

    # root key column is in ctx table
    _analyze_partition(
        tgt_partition_file=tgt_partition_path,
        tgt_stats_path=tgt_stats_path,
        tgt_encoding_types={tgt_values.name: ModelEncodingType.tabular_numeric_digit},
        tgt_context_key=tgt_context_keys.name,
        ctx_partition_file=ctx_partition_path,
        ctx_stats_path=ctx_stats_path,
        ctx_encoding_types={ctx_values.name: ModelEncodingType.tabular_numeric_digit},
        ctx_primary_key=ctx_primary_keys.name,
        ctx_root_key=ctx_root_keys.name,
    )
    tgt_stats = read_json(tgt_stats_path / "part.000000-trn.json")
    assert tgt_stats["columns"][tgt_values.name]["max_n"] == list(range(80))[::-1][:ANALYZE_MIN_MAX_TOP_N]
    assert tgt_stats["columns"][tgt_values.name]["min_n"] == list(range(80))[::1][:ANALYZE_MIN_MAX_TOP_N]
    ctx_stats = read_json(ctx_stats_path / "part.000000-trn.json")
    assert ctx_stats["columns"][ctx_values.name]["max_n"] == list(range(40))[::-2][:ANALYZE_MIN_MAX_TOP_N]
    assert ctx_stats["columns"][ctx_values.name]["min_n"] == list(range(40))[::2][:ANALYZE_MIN_MAX_TOP_N]


class TestAnalyzeCol:
    def test_empty_values(self):
        values = pd.Series([], name="values")
        root_keys = pd.Series([], name="root_keys")
        stats = _analyze_col(values=values, encoding_type=ModelEncodingType.tabular_categorical, root_keys=root_keys)
        assert stats == {"encoding_type": ModelEncodingType.tabular_categorical.value}

    def test_flat_values(self):
        values = pd.Series([1, 2, 3], name="values")
        root_keys = pd.Series([1, 2, 3], name="root_keys")
        stats = _analyze_col(
            values=values, encoding_type=ModelEncodingType.tabular_categorical.value, root_keys=root_keys
        )
        assert stats == {
            "encoding_type": ModelEncodingType.tabular_categorical.value,
            "cnt_values": {"1": 1, "2": 1, "3": 1},
            "has_nan": False,
        }

    def test_sequential_values(self):
        values = pd.Series([[1, 2, 3], [], [3], [], [2]], name="values")
        root_keys = pd.Series([1, 2, 3, 4, 5], name="root_keys")
        stats = _analyze_col(values=values, encoding_type=ModelEncodingType.tabular_categorical, root_keys=root_keys)
        assert stats == {
            "encoding_type": ModelEncodingType.tabular_categorical.value,
            "cnt_values": {"1": 1, "2": 2, "3": 2},
            "has_nan": False,
            "seq_len": {"cnt_lengths": {0: 2, 1: 2, 3: 1}},
        }


def test_synthetic_integer_roots_preserve_counts():
    for values in (
        pd.Series(["a", "a", None, "b"], name="x"),
        pd.Series([["a", "a"], ["b"], [], ["a", None]], name="x"),
    ):
        expected = _analyze_col(
            values,
            ModelEncodingType.tabular_categorical,
            root_keys=pd.Series([str(i) for i in range(len(values))], name="root_keys"),
        )
        assert _analyze_col(values, ModelEncodingType.tabular_categorical) == expected


def test_threaded_analysis_shares_roots_without_reseeding(tmp_path, monkeypatch):
    from joblib import parallel_config
    from pandas.testing import assert_frame_equal

    from mostlyai.engine import analysis

    frame = pd.DataFrame({"id": [1, 1, 2, 3], "x": ["a", "a", "b", None], "y": ["c", "d", "c", "d"]})
    original = frame.copy(deep=True)
    partition = tmp_path / "part.000000-trn.parquet"
    frame.to_parquet(partition)
    stats_path = tmp_path / "stats"
    stats_path.mkdir()
    columns = {"x": ModelEncodingType.tabular_categorical, "y": ModelEncodingType.tabular_categorical}
    _analyze_partition(partition, stats_path, columns, tgt_context_key="id")
    expected = read_json(stats_path / "part.000000-trn.json")
    monkeypatch.setattr(analysis.pd, "read_parquet", lambda *args, **kwargs: frame)
    roots = []
    analyze_col = analysis._analyze_col

    def capture_roots(*args, **kwargs):
        roots.append(kwargs["root_keys"])
        return analyze_col(*args, **kwargs)

    monkeypatch.setattr(analysis, "_analyze_col", capture_roots)
    monkeypatch.setattr(analysis, "set_random_state", lambda **kwargs: (_ for _ in ()).throw(AssertionError("reseed")))
    with parallel_config(backend="threading"):
        _analyze_partition(partition, stats_path, columns, tgt_context_key="id", n_jobs=2)
    assert read_json(stats_path / "part.000000-trn.json") == expected
    assert roots[0] is roots[1]
    assert roots[0].dtype == np.dtype("int64")
    assert_frame_equal(frame, original)
