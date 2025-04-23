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
from mostlyai.engine._encoding_types.tabular.datetime import (
    analyze_datetime,
    analyze_reduce_datetime,
    decode_datetime,
    encode_datetime,
    split_sub_columns_datetime,
)


def test_datetime(tmp_path):
    s1 = pd.Series(
        [
            "1910-01-01",
            "",
            "1930-01-31",
            "1940-02-12",
            "",
            "1971-09-01",
            "1983-05-19",
            "1998-05-24",
        ],
        name="birth_date",
    )
    i1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], name="id")
    s2 = pd.Series(
        [
            "1912-01-01",
            "",
            "1932-01-31",
            "1942-02-12",
            "",
            "1972-09-01",
            "1984-05-19",
            "1994-05-24",
        ],
        name="birth_date",
    )
    i2 = pd.Series([11, 12, 13, 14, 15, 16, 17], name="id")

    stats1 = analyze_datetime(s1, i1)
    stats2 = analyze_datetime(s2, i2)
    write_json(stats1, tmp_path / "stats1.json")
    write_json(stats2, tmp_path / "stats2.json")

    stats = analyze_reduce_datetime([stats1, stats2], value_protection=False)
    write_json(stats, tmp_path / "stats.json")

    stats = read_json(tmp_path / "stats.json")
    df_encoded = encode_datetime(s1, stats)
    df_decoded = decode_datetime(df_encoded, stats)
    assert pd.to_datetime(s1).astype("datetime64[ns]").equals(df_decoded)


def test_datetime_empty(tmp_path):
    values = pd.to_datetime(pd.Series([pd.NaT, pd.NaT, pd.NaT], name="value")).astype("datetime64[ns]")
    root_keys = pd.Series(range(len(values)), name="id")
    stats = analyze_reduce_datetime([analyze_datetime(values, root_keys)], value_protection=False)
    df_encoded = encode_datetime(values, stats)
    df_decoded = decode_datetime(df_encoded, stats)
    assert values.equals(df_decoded)
    assert all(df_decoded.isna())

    values = pd.to_datetime(pd.Series(["2020-05-24", pd.NaT, pd.NaT], name="value"))
    df_encoded = encode_datetime(values, stats)
    df_decoded = decode_datetime(df_encoded, stats)
    assert all(df_decoded.isna())

    # no values at all
    values = pd.to_datetime(pd.Series([], name="value"))
    root_keys = pd.Series(range(len(values)), name="id")
    partition_stats = analyze_datetime(values, root_keys)
    stats = analyze_reduce_datetime([partition_stats])
    df_encoded = encode_datetime(values, stats)
    df_decoded = decode_datetime(df_encoded, stats)
    min_max_values = {
        "day": 1,
        "hour": 0,
        "minute": 0,
        "month": 1,
        "ms_E0": 0,
        "ms_E1": 0,
        "ms_E2": 0,
        "second": 0,
        "year": 2022,
    }
    assert partition_stats == {
        "has_nan": False,
        "max11": [],
        "max_values": min_max_values,
        "min11": [],
        "min_values": min_max_values,
    }
    assert stats == {
        "cardinalities": {"day": 1, "month": 1, "year": 1},
        "has_ms": False,
        "has_nan": False,
        "has_time": False,
        "max": None,
        "max_values": min_max_values,
        "min": None,
        "min_values": min_max_values,
    }
    assert df_encoded.empty, df_encoded.columns.tolist() == (True, [])
    assert df_decoded.empty, df_encoded.columns.tolist() == (True, [])


def test_datetime_noempties(tmp_path):
    values = pd.to_datetime(pd.Series(["2020-05-24", "2021-05-24", "2022-05-24"], name="value"))
    root_keys = pd.Series(range(len(values)), name="id")
    stats = analyze_reduce_datetime([analyze_datetime(values, root_keys)], value_protection=False)
    values = pd.to_datetime(pd.Series([pd.NaT, pd.NaT, pd.NaT], name="value"))
    df_encoded = encode_datetime(values, stats)
    df_decoded = decode_datetime(df_encoded, stats)
    assert all(df_decoded.notna())


def test_datetime_min_max_overlapping():
    root_keys = pd.Series(list(range(21)), name="id")
    values = pd.Series([pd.to_datetime(f"01-01-{2000 + y}") for y in range(21)], name="value")
    stats = analyze_reduce_datetime([analyze_datetime(values, root_keys)])
    for pos, card in stats["cardinalities"].items():
        assert card > 0


def test_split_sub_columns_datetime():
    values = pd.Series([pd.to_datetime("2020-01-01"), pd.NaT], name="dt", index=[1, 1])
    df = split_sub_columns_datetime(values)
    cols = [
        "nan",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "ms_E2",
        "ms_E1",
        "ms_E0",
    ]
    vals = [
        [0, 2020, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    pd.testing.assert_frame_equal(df, pd.DataFrame(vals, columns=cols))
