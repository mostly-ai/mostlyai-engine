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
import pytest

from mostlyai import engine
from mostlyai.engine._workspace import Workspace
from mostlyai.engine.domain import ModelEncodingType


@pytest.mark.parametrize("sequential", [False, True])
def test_empty_generation_context(tmp_path, sequential):
    context = pd.DataFrame({"pid": range(40), "history": [[1, 2]] * 40})
    ids = np.repeat(context.pid, 2 if sequential else 1)
    target = pd.DataFrame({"cid": range(len(ids)), "pid": ids, "value": ["a", "b"] * (len(ids) // 2)})
    target["amount"] = np.arange(len(ids), dtype=float) / 2
    target["date"] = pd.date_range("2020-01-01", periods=len(ids))
    engine.split(
        tgt_data=target,
        tgt_primary_key="cid",
        tgt_context_key="pid",
        ctx_data=context,
        ctx_primary_key="pid",
        ctx_encoding_types={"history": ModelEncodingType.tabular_numeric_discrete},
        workspace_dir=tmp_path,
    )
    engine.analyze(workspace_dir=tmp_path)
    engine.encode(workspace_dir=tmp_path)
    engine.train(max_epochs=1, model="MOSTLY_AI/Small", workspace_dir=tmp_path)
    assert Workspace(tmp_path).tgt_stats.read()["is_sequential"] == sequential

    engine.generate(ctx_data=context.head(2), workspace_dir=tmp_path)
    assert not pd.read_parquet(tmp_path / "SyntheticData").empty

    for kwargs in ({"ctx_data": context.iloc[:0]}, {"ctx_data": context, "sample_size": 0}):
        engine.generate(workspace_dir=tmp_path, **kwargs)
        parts = list((tmp_path / "SyntheticData").glob("*.parquet"))
        assert len(parts) == 1
        result = pd.read_parquet(parts[0])
        assert result.empty
        assert set(result.columns) == set(target.columns)
        assert result["pid"].dtype == context.pid.dtype
        assert pd.api.types.is_string_dtype(result["value"].dtype)
        assert pd.api.types.is_numeric_dtype(result["amount"].dtype)
        assert pd.api.types.is_datetime64_any_dtype(result["date"].dtype)


def test_zero_samples_without_context(tmp_path):
    target = pd.DataFrame({"id": range(40), "value": ["a", "b"] * 20})
    engine.split(tgt_data=target, tgt_primary_key="id", workspace_dir=tmp_path)
    engine.analyze(workspace_dir=tmp_path)
    engine.encode(workspace_dir=tmp_path)
    engine.train(max_epochs=1, model="MOSTLY_AI/Small", workspace_dir=tmp_path)
    for kwargs in ({"sample_size": 0}, {"seed_data": target[["value"]].iloc[:0]}):
        engine.generate(workspace_dir=tmp_path, **kwargs)
        result = pd.read_parquet(tmp_path / "SyntheticData")
        assert result.empty
        assert set(result.columns) == set(target.columns)
