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

from mostlyai.engine._encoding_types.tabular.categorical import (
    CATEGORICAL_NULL_TOKEN,
    CATEGORICAL_SUB_COL_SUFFIX,
    CATEGORICAL_UNKNOWN_TOKEN,
    analyze_categorical,
    analyze_reduce_categorical,
    decode_categorical,
    encode_categorical,
    safe_categorical_escape,
    safe_categorical_unescape,
)


class TestCategoricalAnalyze:
    def test_2_frequent_and_1_rare_values(self):
        values = pd.Series(np.repeat(["secret", "male", "female"], 100), name="gender")
        ids = pd.Series(
            np.concatenate([np.repeat(0, 100), range(100), range(100, 200)]),
            name="subject_id",
        )
        stats = analyze_categorical(values, ids)
        assert stats == {
            "cnt_values": {"female": 100, "male": 100, "secret": 1},
            "has_nan": False,
        }

    def test_null_only_values(self):
        values = pd.Series([pd.NA, pd.NA, np.nan] * 100, name="gender")
        ids = pd.Series([1, 2, 3] * 100, name="subject_id")
        stats = analyze_categorical(values, ids)
        assert stats == {"cnt_values": {}, "has_nan": True}

    def test_mixed_frequent_and_rare_and_null_values(self):
        values = pd.Series(np.repeat(["secret", "male", pd.NA], 100), name="value")
        ids = pd.Series(
            np.concatenate([np.repeat(0, 100), range(100), range(100, 200)]),
            name="subject_id",
        )
        stats = analyze_categorical(values, ids)
        assert stats == {
            "cnt_values": {"male": 100, "secret": 1},
            "has_nan": True,
        }

    def test_no_values_at_all(self):
        values = pd.Series([], name="value")
        ids = pd.Series([], name="subject_id")
        stats = analyze_categorical(values, ids)
        assert stats == {"cnt_values": {}, "has_nan": False}


class TestCategoricalAnalyzeReduce:
    @pytest.fixture
    def stats_list(self):
        stats1 = {
            "cnt_values": {"secret1": 1, "male": 100},
            "has_nan": False,
        }
        stats2 = {
            "cnt_values": {"secret2": 1, "male": 100, "female": 100},
            "has_nan": False,
        }
        return stats1, stats2

    def test_without_value_protection(self, stats_list):
        stats1, stats2 = stats_list
        stats = analyze_reduce_categorical([stats1, stats2], value_protection=False)
        assert stats == {
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 5},
            "codes": {
                CATEGORICAL_UNKNOWN_TOKEN: 0,
                "female": 1,
                "male": 2,
                "secret1": 3,
                "secret2": 4,
            },
            "no_of_rare_categories": 0,
        }

    def test_with_value_protection(self, stats_list):
        stats1, stats2 = stats_list
        stats = analyze_reduce_categorical([stats1, stats2], value_protection=True)
        assert stats == {
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 3},
            "codes": {CATEGORICAL_UNKNOWN_TOKEN: 0, "female": 1, "male": 2},
            "no_of_rare_categories": 2,
        }

    def test_with_has_nan_and_values_protection(self, stats_list):
        stats1, stats2 = stats_list
        stats1["has_nan"] = True
        stats = analyze_reduce_categorical([stats1, stats2], value_protection=True)
        assert stats == {
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 4},
            "codes": {
                CATEGORICAL_NULL_TOKEN: 1,
                CATEGORICAL_UNKNOWN_TOKEN: 0,
                "female": 2,
                "male": 3,
            },
            "no_of_rare_categories": 2,
        }


class TestCategoricalEncode:
    def test_frequent_only_values(self):
        values = pd.Series(np.repeat(["male", "female"], 100), name="gender")
        stats = {
            "no_of_rare_categories": 0,
            "codes": {CATEGORICAL_UNKNOWN_TOKEN: 0, "female": 1, "male": 2},
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 3},
        }
        expected = pd.DataFrame({"cat": np.repeat([2, 1], 100)})
        encoded = encode_categorical(values, stats)
        pd.testing.assert_frame_equal(encoded, expected, check_dtype=False)

    def test_2_frequent_and_1_rare_and_1_null_values(self):
        values = pd.Series(np.repeat(["secret", "male", "female", pd.NA], 100), name="gender")
        stats = {
            "no_of_rare_categories": 1,
            "codes": {
                CATEGORICAL_UNKNOWN_TOKEN: 0,
                CATEGORICAL_NULL_TOKEN: 1,
                "female": 2,
                "male": 3,
            },
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 4},
        }
        expected = pd.DataFrame({"cat": np.repeat([0, 3, 2, 1], 100)})
        encoded = encode_categorical(values, stats)
        pd.testing.assert_frame_equal(encoded, expected, check_dtype=False)

    def test_rare_values_only(self):
        values = pd.Series(["secret1", "secret2", "secret3"], name="value")
        stats = {
            "no_of_rare_categories": 3,
            "codes": {CATEGORICAL_UNKNOWN_TOKEN: 0},
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 1},
        }
        expected = pd.DataFrame({"cat": [0, 0, 0]})
        encoded = encode_categorical(values, stats)
        pd.testing.assert_frame_equal(encoded, expected, check_dtype=False)

    def test_null_values_only(self):
        values = pd.Series([pd.NA, pd.NA, pd.NA], name="value")
        stats = {
            "no_of_rare_categories": 0,
            "codes": {CATEGORICAL_UNKNOWN_TOKEN: 0, CATEGORICAL_NULL_TOKEN: 1},
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 2},
        }
        expected = pd.DataFrame({"cat": [1, 1, 1]})
        encoded = encode_categorical(values, stats)
        pd.testing.assert_frame_equal(encoded, expected, check_dtype=False)

    def test_empty(self):
        values = pd.Series([], name="value")
        stats = {
            "no_of_rare_categories": 0,
            "codes": {CATEGORICAL_UNKNOWN_TOKEN: 0},
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 1},
        }
        expected = pd.DataFrame({"cat": []})
        encoded = encode_categorical(values, stats)
        pd.testing.assert_frame_equal(encoded, expected, check_dtype=False)


class TestCategoricalDecode:
    def test_frequent_only_values(self):
        encoded = pd.DataFrame({"cat": np.repeat([1, 2], 100)})
        stats = {
            "no_of_rare_categories": 0,
            "codes": {CATEGORICAL_UNKNOWN_TOKEN: 0, "female": 1, "male": 2},
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 3},
        }
        expected = pd.Series(np.repeat(["female", "male"], 100))
        decoded = decode_categorical(encoded, stats)
        pd.testing.assert_series_equal(decoded, expected, check_dtype=False)

    def test_2_frequent_and_1_rare_and_1_null_values(self):
        encoded = pd.DataFrame({"cat": np.repeat([0, 3, 2, 1], 100)})
        stats = {
            "no_of_rare_categories": 1,
            "codes": {
                CATEGORICAL_UNKNOWN_TOKEN: 0,
                CATEGORICAL_NULL_TOKEN: 1,
                "female": 2,
                "male": 3,
            },
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 4},
        }
        expected = pd.Series(np.repeat(["_RARE_", "male", "female", pd.NA], 100))
        decoded = decode_categorical(encoded, stats)
        pd.testing.assert_series_equal(decoded, expected, check_dtype=False)

    def test_rare_values_only(self):
        encoded = pd.DataFrame({"cat": [0, 0, 0]})
        stats = {
            "no_of_rare_categories": 3,
            "codes": {CATEGORICAL_UNKNOWN_TOKEN: 0},
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 1},
        }
        expected = pd.Series(["_RARE_", "_RARE_", "_RARE_"])
        decoded = decode_categorical(encoded, stats)
        pd.testing.assert_series_equal(decoded, expected, check_dtype=False)

    def test_null_values_only(self):
        encoded = pd.DataFrame({"cat": [1, 1, 1]})
        stats = {
            "no_of_rare_categories": 0,
            "codes": {CATEGORICAL_UNKNOWN_TOKEN: 0, CATEGORICAL_NULL_TOKEN: 1},
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 2},
        }
        expected = pd.Series([pd.NA, pd.NA, pd.NA])
        decoded = decode_categorical(encoded, stats)
        pd.testing.assert_series_equal(decoded, expected, check_dtype=False)

    def test_empty(self):
        encoded = pd.DataFrame({"cat": []})
        stats = {
            "no_of_rare_categories": 0,
            "codes": {CATEGORICAL_UNKNOWN_TOKEN: 0},
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: 1},
        }
        expected = pd.Series([])
        decoded = decode_categorical(encoded, stats)
        pd.testing.assert_series_equal(decoded, expected, check_dtype=False)


class TestCategoricalEscape:
    @pytest.fixture
    def edgy_values(self):
        yield [
            "",
            pd.NA,
            "1",
            "_RARE_",
            "<<NULL>>",
            "\x01123",
            "\x01",
            "\x01_RARE_",
            "\x01\x01<<NULL>>",
        ]

    def test_escape_unescape(self, edgy_values):
        values = pd.Series(edgy_values, name="value")
        escaped = safe_categorical_escape(values.copy())
        unescaped = safe_categorical_unescape(escaped)
        assert unescaped.equals(values)

    @pytest.mark.parametrize("value_protection", [True, False])
    def test_analyze_reduce_encode_decode(self, edgy_values, value_protection):
        frequent_values = ["<<NULL>>", "_RARE_"]
        values_1 = pd.Series(edgy_values, name="value")
        values_2 = pd.Series(frequent_values * 10, name="value")
        stats_1 = analyze_categorical(values_1.copy(), pd.Series(range(len(values_1)), name="id"))
        stats_2 = analyze_categorical(values_2.copy(), pd.Series(range(len(values_2)), name="id"))
        stats = analyze_reduce_categorical([stats_1, stats_2], value_protection)
        enc_1 = encode_categorical(values_1.copy(), stats)
        enc_2 = encode_categorical(values_2.copy(), stats)
        dec_1 = decode_categorical(enc_1, stats)
        dec_2 = decode_categorical(enc_2, stats)
        if value_protection:
            rare_values = list(set(edgy_values) - set(frequent_values + [pd.NA]))
            protected_values_1 = values_1.replace(rare_values, ["_RARE_"] * len(rare_values))
            pd.testing.assert_series_equal(protected_values_1, dec_1, check_dtype=False, check_names=False)
        else:
            pd.testing.assert_series_equal(values_1, dec_1, check_dtype=False, check_names=False)
        pd.testing.assert_series_equal(values_2, dec_2, check_dtype=False, check_names=False)
