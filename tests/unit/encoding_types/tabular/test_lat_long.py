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

from mostlyai.engine._common import read_json, write_json
from mostlyai.engine._encoding_types.tabular.lat_long import (
    GEOPOSITION_PRECISION,
    analyze_latlong,
    analyze_reduce_latlong,
    decode_latlong,
    encode_latlong,
    split_str_to_latlong,
)


def stringify_latlong_arrays(arr_lat, arr_long):
    return [f"{lat}, {long}" for lat, long in zip(arr_lat, arr_long)]


def _geo_grid(size, start_lat, start_long, end_lat, end_long, rep=1):
    """
    Create a geo grid from (start_lat, start_long) to (end_lat, end_long)
    with a linear distribution, having a repetition of rep per each coordinate.
    """
    start_lat = int(start_lat * GEOPOSITION_PRECISION)
    start_long = int(start_long * GEOPOSITION_PRECISION)
    end_lat = int(end_lat * GEOPOSITION_PRECISION)
    end_long = int(end_long * GEOPOSITION_PRECISION)
    dim_len = np.sqrt(size).astype(int)
    steps = (np.arange(dim_len) / rep).astype(int)
    step_factor = steps / np.max(steps)
    lats = start_lat + ((end_lat - start_lat) * step_factor).astype(int)
    longs = start_long + ((end_long - start_long) * step_factor).astype(int)
    grid_lat, grid_long = np.meshgrid(lats / GEOPOSITION_PRECISION, longs / GEOPOSITION_PRECISION)
    return stringify_latlong_arrays(np.hstack(grid_lat), np.hstack(grid_long))


def _random_latlong_array(lat_min=-90, lat_max=90, long_min=-90, long_max=90, cnt=50):
    arr_lat = (
        np.random.randint(
            low=lat_min * GEOPOSITION_PRECISION,
            high=lat_max * GEOPOSITION_PRECISION,
            size=cnt,
        )
        / GEOPOSITION_PRECISION
    )
    arr_long = (
        np.random.randint(
            low=long_min * GEOPOSITION_PRECISION,
            high=long_max * GEOPOSITION_PRECISION,
            size=cnt,
        )
        / GEOPOSITION_PRECISION
    )
    return stringify_latlong_arrays(arr_lat, arr_long)


def _is_latlong_nearly_equal(latlong1, latlong2):
    df_1 = split_str_to_latlong(latlong1)
    df_1_na_cnt = sum(df_1.isna().any(axis=1) > 0)
    df_1.fillna(0, inplace=True)
    df_2 = split_str_to_latlong(latlong2)
    df_2_na_cnt = sum(df_2.isna().any(axis=1) > 0)
    df_2.fillna(0, inplace=True)
    # relax the precision by one decimal vs the geoposition precision
    return (
        np.allclose(df_1[0].values, df_2[0].values, atol=1 / GEOPOSITION_PRECISION)
        and np.allclose(df_1[1].values, df_2[1].values, atol=1 / GEOPOSITION_PRECISION)
        and df_1_na_cnt == df_2_na_cnt
    )


def test_latlong_grid(tmp_path):
    n = 10_000
    m = 10

    # Test two geo-grids of n coordinates each
    # each root_id m times in a row
    grid_root_keys1 = pd.Series((np.arange(n) / m).astype(int), name="user_id")
    # grid with multiple repeating coordinates
    grid_values1 = pd.Series(_geo_grid(n, 33, 33, 35, 37, 3))
    grid_stats1 = analyze_latlong(grid_values1, grid_root_keys1)
    write_json(grid_stats1, tmp_path / "grid_stats1.json")
    grid_stats1 = read_json(tmp_path / "grid_stats1.json")
    grid_root_keys2 = pd.Series(np.arange(n) + n, name="user_id")
    # slightly shifted grid1, 3 times less repetitions
    grid_values2 = pd.Series(_geo_grid(n, 33.001, 33.001, 35.001, 37.001))
    grid_stats2 = analyze_latlong(grid_values2, grid_root_keys2)
    write_json(grid_stats2, tmp_path / "grid_stats2.json")
    grid_stats2 = read_json(tmp_path / "grid_stats2.json")
    grid_reduced_stats = analyze_reduce_latlong([grid_stats1, grid_stats2])
    encoded_grid1 = encode_latlong(grid_values1, grid_reduced_stats)
    encoded_quads = [col for col in encoded_grid1.columns if col.startswith("QUAD")]
    assert encoded_quads == [
        "QUAD12",
        "QUAD14",
    ]  # at least, for n = 10k - due to filtering out noise
    decoded_grid1 = decode_latlong(encoded_grid1, grid_reduced_stats)
    assert _is_latlong_nearly_equal(grid_values1, decoded_grid1)


def test_random_latlong_with_na(tmp_path):
    # Test random coordinates within a latlong range + NAs
    latlong_range = dict(lat_min=15, lat_max=16, long_min=75, long_max=76)
    root_keys = pd.Series(np.arange(100), name="user_id")
    values1 = pd.Series(np.array(_random_latlong_array(**latlong_range) + ["ERR"] * 30 + [pd.NA] * 20))
    values2 = pd.Series(np.array(_random_latlong_array(**latlong_range, cnt=100)))
    stats1 = analyze_latlong(values1, root_keys)
    write_json(stats1, tmp_path / "stats1.json")
    stats1 = read_json(tmp_path / "stats1.json")
    stats2 = analyze_latlong(values2, root_keys)
    write_json(stats2, tmp_path / "stats2.json")
    stats2 = read_json(tmp_path / "stats2.json")
    stats = analyze_reduce_latlong([stats1, stats2])
    encoded1 = encode_latlong(values1, stats)
    encoded_quads = [col for col in encoded1.columns if col.startswith("QUAD")]
    assert not encoded_quads  # dataset is too small to have any categorical quads
    decoded1 = decode_latlong(encoded1, stats)
    assert _is_latlong_nearly_equal(values1, decoded1)
    encoded2 = encode_latlong(values2, stats)
    decoded2 = decode_latlong(encoded2, stats)
    assert _is_latlong_nearly_equal(values2, decoded2)


def test_latlong_na(tmp_path):
    root_keys = pd.Series(np.arange(100), name="user_id")
    values1 = pd.Series(np.array(["ERR"] * 30 + [pd.NA] * 20 + [",,,,"] * 30 + ["A,B,C,"] * 20))
    stats1 = analyze_latlong(values1, root_keys)
    write_json(stats1, tmp_path / "stats1.json")
    stats1 = read_json(tmp_path / "stats1.json")
    stats = analyze_reduce_latlong([stats1])
    encoded1 = encode_latlong(values1, stats)
    encoded_quads = [col for col in encoded1.columns if col.startswith("QUAD")]
    assert not encoded_quads  # dataset is too small to have any categorical quads
    decoded1 = decode_latlong(encoded1, stats)
    assert _is_latlong_nearly_equal(values1, decoded1)

    # no values at all
    values = pd.Series(pd.Series([], name="value"))
    root_keys = pd.Series(range(len(values)), name="id")
    partition_stats = analyze_latlong(values, root_keys)
    stats = analyze_reduce_latlong([partition_stats])
    df_encoded = encode_latlong(values, stats)
    df_decoded = decode_latlong(df_encoded, stats)
    characters = {f"P{idx}": c for idx, c in enumerate(["+", "+"] + ["A"] * 28)}
    assert partition_stats == {
        "characters": {pos: [c] for pos, c in characters.items()},
        "has_nan": False,
        "quad_codes": {f"QUAD{num}": {} for num in range(12, 27, 2)},
    }
    assert stats == {
        "cardinalities": {pos: 1 for pos in characters.keys()},
        "has_nan": False,
        "quad_codes": {},
        "quadtile_characters": {
            "codes": {pos: {c: 0} for pos, c in characters.items()},
            "has_nan": False,
            "max_string_length": 30,
        },
    }
    assert df_encoded.empty, df_encoded.columns.tolist() == (True, [])
    assert df_decoded.empty, df_encoded.columns.tolist() == (True, [])
