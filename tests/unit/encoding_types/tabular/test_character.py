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
from mostlyai.engine._encoding_types.tabular.character import (
    MAX_LENGTH_CHARS,
    analyze_character,
    analyze_reduce_character,
    decode_character,
    encode_character,
)


def test_character(tmp_path):
    # create sequence of common strings, with some of those being overly long
    vals = np.repeat(["word", "sentence", "_".join("too_long") * 10], 100)
    # inject canaries, to then check whether those tokens are suppressed
    canary = "§§§"
    no_of_canaries = 3
    values1 = pd.Series([canary] * no_of_canaries + list(vals), name="chars")
    ids1 = pd.Series(np.arange(len(values1)), name="subject_id")
    # create sequence of common strings, with some of those missing
    values2 = pd.Series([pd.NA, "random_word", pd.NA] * 100, name="chars")
    ids2 = pd.Series(np.arange(len(values2)), name="subject_id")
    unseen_values = pd.Series(["a_sentence", "new_word"], name="chars")

    stats1 = analyze_character(values1, ids1)
    stats2 = analyze_character(values2, ids2)
    assert stats1["max_string_length"] == MAX_LENGTH_CHARS
    assert len(stats1["characters"]) == MAX_LENGTH_CHARS
    assert stats2["max_string_length"] == values2.str.len().max()
    assert len(stats2["characters"]) == values2.str.len().max()
    write_json(stats1, tmp_path / "stats1.json")
    write_json(stats2, tmp_path / "stats2.json")

    stats1 = read_json(tmp_path / "stats1.json")
    stats2 = read_json(tmp_path / "stats2.json")
    stats = analyze_reduce_character([stats1, stats2])
    assert len(stats["codes"]) == MAX_LENGTH_CHARS
    # check that those rare characters don't occur in any vocabulary set
    for p in stats["codes"]:
        assert "§" not in stats["codes"][p]
    write_json(stats, tmp_path / "stats.json")

    stats = read_json(tmp_path / "stats.json")
    encoded1 = encode_character(values1, stats)
    decoded1 = decode_character(encoded1, stats)
    assert decoded1[no_of_canaries:].equals(values1[no_of_canaries:].str.slice(stop=MAX_LENGTH_CHARS))
    encoded2 = encode_character(values2, stats)
    decoded2 = decode_character(encoded2, stats)
    assert decoded2.equals(values2.str.slice(stop=MAX_LENGTH_CHARS))

    unseen_encoded = encode_character(unseen_values, stats)
    assert all(unseen_encoded.drop("nan", axis=1).values.flatten() >= 0)


def test_character_empty():
    values = pd.Series([None, None, None], name="value")
    ids = pd.Series(np.arange(len(values)), name="subject_id")
    stats = analyze_reduce_character([analyze_character(values, ids)])
    df_encoded = encode_character(values, stats)
    df_decoded = decode_character(df_encoded, stats)
    assert all(df_decoded.isna())

    values = pd.Series(["hello", None, None], name="value")
    df_encoded = encode_character(values, stats)
    df_decoded = decode_character(df_encoded, stats)
    assert all(df_decoded.isna())

    # no values at all
    values = pd.Series([], name="value")
    ids = pd.Series(np.arange(len(values)), name="subject_id")
    partition_stats = analyze_character(values, ids)
    stats = analyze_reduce_character([partition_stats])
    df_encoded = encode_character(values, stats)
    df_decoded = decode_character(df_encoded, stats)
    assert partition_stats == {
        "characters": {},
        "has_nan": False,
        "max_string_length": 0,
    }
    assert stats == {
        "cardinalities": {},
        "codes": {},
        "has_nan": False,
        "max_string_length": 0,
    }
    assert df_encoded.empty, df_encoded.columns.tolist() == (True, [])
    assert df_decoded.empty, df_encoded.columns.tolist() == (True, [])


def test_character_noempties():
    values = pd.Series(["hello", "world", "!"], name="value")
    ids = pd.Series(np.arange(len(values)), name="subject_id")
    stats = analyze_reduce_character([analyze_character(values, ids)])
    values = pd.Series([None, None, None], name="value")
    df_encoded = encode_character(values, stats)
    df_decoded = decode_character(df_encoded, stats)
    assert df_decoded.size == values.size
