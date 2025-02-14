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

import pandas as pd

from mostlyai.engine._common import read_json, write_json
from mostlyai.engine._encoding_types.tabular.itt import (
    analyze_itt,
    analyze_reduce_itt,
    decode_itt,
    encode_itt,
)


def test_itt_date(tmp_path):
    values = pd.to_datetime(
        pd.Series(
            [None, "1978-05-24", "1976-06-22", "1992-12-24", None],
            name="date",
            dtype="datetime64[us]",
        )
    )
    context_keys = pd.Series(["a", "a", "a", "b", "c"], name="__context_key")
    root_keys = context_keys.copy()
    root_keys.name = "__root_key"
    stats1 = analyze_itt(values=values, root_keys=root_keys, context_keys=context_keys)
    write_json(stats1, tmp_path / "stats1.json")
    stats1 = read_json(tmp_path / "stats1.json")
    stats = analyze_reduce_itt([stats1], value_protection=False)
    write_json(stats, tmp_path / "stats.json")
    stats = read_json(tmp_path / "stats.json")
    df_encoded = encode_itt(values=values, stats=stats, context_keys=context_keys)
    df_decoded = decode_itt(df_encoded=df_encoded, stats=stats, context_keys=context_keys)
    assert values.equals(df_decoded)


def test_itt_datetime(tmp_path):
    values = pd.to_datetime(
        pd.Series(
            [
                None,
                "1978-05-24 12:23:43",
                "1976-06-22 17:32:00",
                "1992-12-24 01:32:59",
                None,
            ],
            name="date",
            dtype="datetime64[us]",
        )
    )
    context_keys = pd.Series(["a", "a", "a", "b", "c"], name="__context_key")
    root_keys = context_keys.copy()
    root_keys.name = "__root_key"
    stats1 = analyze_itt(values=values, root_keys=root_keys, context_keys=context_keys)
    stats = analyze_reduce_itt([stats1], value_protection=False)
    df_encoded = encode_itt(values=values, stats=stats, context_keys=context_keys)
    df_decoded = decode_itt(df_encoded=df_encoded, stats=stats, context_keys=context_keys)
    assert values.equals(df_decoded)


def test_itt_nones_only(tmp_path):
    values = pd.to_datetime(pd.Series([None, None, None], name="value", dtype="datetime64[us]"))
    context_keys = pd.Series(["a", "a", "b"], name="id")
    root_keys = pd.Series(["a", "a", "b"], name="rid")
    stats = analyze_reduce_itt([analyze_itt(values, root_keys, context_keys)], value_protection=False)
    df_encoded = encode_itt(values, stats, context_keys)
    df_decoded = decode_itt(df_encoded, stats, context_keys)
    assert all(df_decoded.isna())


def test_itt_empty(tmp_path):
    values = pd.Series([], name="value")
    root_keys = pd.Series([], name="rid")
    context_keys = pd.Series([], name="id")
    partition_stats = analyze_itt(values, root_keys, context_keys)
    stats = analyze_reduce_itt([partition_stats])
    df_encoded = encode_itt(values, stats, context_keys)
    df_decoded = decode_itt(df_encoded, stats, context_keys)
    min_max_values = {
        "itt_day": 0,
        "itt_hour": 0,
        "itt_minute": 0,
        "itt_second": 0,
        "itt_week": 0,
        "start_day": 1,
        "start_hour": 0,
        "start_minute": 0,
        "start_month": 1,
        "start_second": 0,
        "start_year": 2022,
    }
    assert partition_stats == {
        "has_nan": False,
        "has_neg": False,
        "max11": [],
        "max_values": min_max_values,
        "min11": [],
        "min_values": min_max_values,
    }
    assert stats == {
        "cardinalities": {
            "itt_day": 1,
            "itt_week": 1,
            "start_day": 1,
            "start_month": 1,
            "start_year": 1,
        },
        "has_nan": False,
        "has_neg": False,
        "has_time": False,
        "max5": [],
        "max_values": min_max_values,
        "min5": [],
        "min_values": min_max_values,
    }
    assert df_encoded.empty, df_encoded.columns.tolist() == (True, [])
    assert df_decoded.empty, df_encoded.columns.tolist() == (True, [])


def test_itt_1to1(tmp_path):
    values = pd.to_datetime(
        pd.Series(
            [None, "1978-05-24", "1976-06-22", "1992-12-24", None],
            name="date",
            dtype="datetime64[us]",
        )
    )
    context_keys = pd.Series(["a", "b", "c", "d", "e"], name="__context_key")
    root_keys = context_keys.copy()
    root_keys.name = "__root_key"
    stats1 = analyze_itt(values=values, root_keys=root_keys, context_keys=context_keys)
    write_json(stats1, tmp_path / "stats1.json")
    stats1 = read_json(tmp_path / "stats1.json")
    stats = analyze_reduce_itt([stats1], value_protection=False)
    write_json(stats, tmp_path / "stats.json")
    stats = read_json(tmp_path / "stats.json")
    df_encoded = encode_itt(values=values, stats=stats, context_keys=context_keys)
    df_decoded = decode_itt(df_encoded=df_encoded, stats=stats, context_keys=context_keys)
    assert values.equals(df_decoded)


def test_itt_with_prev_steps(tmp_path):
    values = pd.to_datetime(
        pd.Series(
            ["1978-05-24", "1976-06-22", "1976-06-23", "1976-06-24"],
            name="date",
            dtype="datetime64[us]",
        )
    )
    context_keys = pd.Series(["a", "b", "b", "b"], name="__context_key")
    root_keys = context_keys.copy()
    root_keys.name = "__root_key"
    stats1 = analyze_itt(values=values, root_keys=root_keys, context_keys=context_keys)
    write_json(stats1, tmp_path / "stats1.json")
    stats1 = read_json(tmp_path / "stats1.json")
    stats = analyze_reduce_itt([stats1], value_protection=False)
    write_json(stats, tmp_path / "stats.json")
    stats = read_json(tmp_path / "stats.json")
    df_encoded = encode_itt(values=values, stats=stats, context_keys=context_keys)
    prev_steps = {
        "prev_dts": pd.DataFrame(
            {
                "__CONTEXT_KEYS": ["a", "b"],
                "__STARTS": pd.to_datetime(pd.Series(["1978-05-23", "1976-06-21"], dtype="datetime64[us]")),
            }
        )
    }
    df_decoded = decode_itt(df_encoded=df_encoded, stats=stats, context_keys=context_keys, prev_steps=prev_steps)
    assert values.equals(df_decoded)
