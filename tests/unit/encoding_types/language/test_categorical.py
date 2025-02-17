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
import pytest
import numpy as np

from mostlyai.engine._encoding_types.language.categorical import (
    CATEGORICAL_UNKNOWN_TOKEN,
    decode_language_categorical,
    analyze_language_categorical,
    analyze_reduce_language_categorical,
    encode_language_categorical,
)


class TestLanguageCategoricalAnalyze:
    def test_3_frequent_and_1_rare_values(self):
        values = pd.Series(np.repeat(["secret", "male", "female", pd.NA], 100), name="gender")
        ids = pd.Series(
            np.concatenate([np.repeat(0, 100), range(100), range(100, 200), range(200, 300)]),
            name="subject_id",
        )
        stats = analyze_language_categorical(values, ids)
        assert stats == {
            "cnt_values": {"female": 100, "male": 100, "secret": 1},
            "has_nan": True,
        }


class TestLanguageCategoricalAnalyzeReduce:
    @pytest.fixture
    def stats_list(self):
        stats1 = {
            "cnt_values": {"secret1": 1, "male": 100},
            "has_nan": True,
        }
        stats2 = {
            "cnt_values": {"secret2": 1, "male": 100, "female": 100},
            "has_nan": False,
        }
        return stats1, stats2

    def test_with_value_protection(self, stats_list):
        stats1, stats2 = stats_list
        stats = analyze_reduce_language_categorical([stats1, stats2], value_protection=True)
        assert stats == {
            "categories": [CATEGORICAL_UNKNOWN_TOKEN, None, "female", "male"],
            "no_of_rare_categories": 2,
        }


class TestLanguageCategoricalEncode:
    def test_2_frequent_and_1_rare_and_1_null_values(self):
        values = pd.Series(np.repeat(["secret", "male", "female", pd.NA], 100), name="gender")
        stats = {
            "categories": [CATEGORICAL_UNKNOWN_TOKEN, None, "female", "male"],
            "no_of_rare_categories": 1,
        }
        expected = pd.Series(
            np.repeat([CATEGORICAL_UNKNOWN_TOKEN, "male", "female", pd.NA], 100), name="gender", dtype="string"
        )
        encoded = encode_language_categorical(values, stats)
        pd.testing.assert_series_equal(encoded, expected)


class TestLanguageCategoricalDecode:
    @pytest.fixture
    def col_stats(self):
        return {"categories": [CATEGORICAL_UNKNOWN_TOKEN, None, "apple", "banana", "cherry"]}

    @pytest.fixture
    def sample_values(self):
        return pd.Series(["apple", "durian", "banana", None, "cherry", "fig", None])

    def test_language_categorical_decode(self, sample_values, col_stats):
        decoded = decode_language_categorical(sample_values, col_stats)
        expected = pd.Series(
            ["apple", CATEGORICAL_UNKNOWN_TOKEN, "banana", None, "cherry", CATEGORICAL_UNKNOWN_TOKEN, None],
            dtype=decoded.dtype,
        )
        pd.testing.assert_series_equal(decoded, expected)
