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

from mostlyai.engine.domain import ModelEncodingType
from mostlyai.engine._encoding_types.tabular.numeric import (
    NUMERIC_BINNED_MAX_QUANTILES,
    NUMERIC_BINNED_MAX_TOKEN,
    NUMERIC_BINNED_MIN_TOKEN,
    NUMERIC_BINNED_SUB_COL_SUFFIX,
    NUMERIC_BINNED_UNKNOWN_TOKEN,
    NUMERIC_DISCRETE_UNKNOWN_TOKEN,
    analyze_numeric,
    analyze_reduce_numeric,
    decode_numeric,
    encode_numeric,
    split_sub_columns_digit,
)


def _ints(string: str) -> list[int]:
    return [int(c) for c in string]


def _digit_cols(start: int, end: int) -> list[str]:
    return ["nan", "neg"] + [f"E{idx}" for idx in range(start, end - 1, -1)]


def _digit_to_int(string: str) -> dict[str, int]:
    return {f"E{18 - idx}": int(c) for idx, c in enumerate(string)}


class TestSplitSubColumnsDigit:
    def test_max_min_specified(self):
        values = pd.Series([21047147.89, -910635.287793, pd.NA], dtype="Float64")
        actual = split_sub_columns_digit(values, max_decimal=7, min_decimal=-6)
        expected = pd.DataFrame(
            [
                # _ints(
                #     "__"        null position, sign position
                #   + "________" digits after the comma
                #   + "______"   digits before the comma
                # )
                _ints("00" + "21047147" + "890000"),
                _ints("01" + "00910635" + "287793"),
                _ints("10" + "00000000" + "000000"),
            ],
            columns=_digit_cols(7, -6),
        )
        pd.testing.assert_frame_equal(actual, expected)

    def test_default_max_min(self):
        values = pd.Series([0.2997, 0.1546, 71.46, 4.1, 364210.16, 0.00999999977648])
        actual = split_sub_columns_digit(values)
        expected = pd.DataFrame(
            [
                # _ints(
                #     "__"                  null position, sign position
                #   + "___________________" digits after the comma
                #   + "________"            digits before the comma
                # )
                _ints("00" + "0000000000000000000" + "29970000"),
                _ints("00" + "0000000000000000000" + "15460000"),
                _ints("00" + "0000000000000000071" + "46000000"),
                _ints("00" + "0000000000000000004" + "10000000"),
                _ints("00" + "0000000000000364210" + "16000000"),
                _ints("00" + "0000000000000000000" + "00999999"),
            ],
            columns=_digit_cols(18, -8),
        )
        pd.testing.assert_frame_equal(actual, expected)


class TestDigitAnalyze:
    def test_positive_integers_and_fractions(self):
        fractions = pd.Series(np.repeat(np.linspace(0, 0.9999, 10), 10))
        integers = pd.Series(np.repeat(np.linspace(1, 10, 10), 10))
        values = pd.concat([fractions, integers]).reset_index(drop=True).rename("vals")
        values = values.round(4)
        ids = pd.Series(range(len(values)), name="subject_id")
        stats = analyze_numeric(values, ids)
        assert stats["has_nan"] is False
        assert stats["has_neg"] is False
        assert stats["min_digits"] == _digit_to_int("0000000000000000000" + "00000000")  # E18...E0 + E-1...E-8
        assert stats["max_digits"] == _digit_to_int("0000000000000000019" + "99990000")  # E18...E0 + E-1...E-8
        assert stats["min11"] == [0.0] * 10 + [0.1111]
        assert stats["max11"] == [10.0] * 10 + [9.0]

    def test_negative_integers_and_fractions(self):
        fractions = pd.Series(np.repeat(np.linspace(-0.9999, 0, 10), 10))
        integers = pd.Series(np.repeat(np.linspace(-1, -10, 10), 10))
        values = pd.concat([fractions, integers]).reset_index(drop=True).rename("vals")
        values = values.round(4)
        ids = pd.Series(range(len(values)), name="subject_id")
        stats = analyze_numeric(values, ids)
        assert stats["has_nan"] is False
        assert stats["has_neg"] is True
        assert stats["min_digits"] == _digit_to_int("0000000000000000000" + "00000000")  # E18...E0 + E-1...E-8
        assert stats["max_digits"] == _digit_to_int("0000000000000000019" + "99990000")  # E18...E0 + E-1...E-8
        assert stats["min11"] == [-10.0] * 10 + [-9.0]
        assert stats["max11"] == [0.0] * 10 + [-0.1111]

    def test_integers_and_nulls(self):
        values = pd.Series([1, 2, 3, None, pd.NA], name="vals")
        ids = pd.Series(range(len(values)), name="subject_id")
        stats = analyze_numeric(values, ids)
        assert stats["has_nan"] is True
        assert stats["has_neg"] is False
        assert stats["min_digits"] == _digit_to_int("0000000000000000001" + "00000000")  # E18...E0 + E-1...E-8
        assert stats["max_digits"] == _digit_to_int("0000000000000000003" + "00000000")  # E18...E0 + E-1...E-8
        assert stats["min11"] == [1.0, 2.0, 3.0]
        assert stats["max11"] == [3.0, 2.0, 1.0]

    def test_nulls_only(self):
        values = pd.Series([None, np.nan, pd.NA], name="vals")
        ids = pd.Series(range(len(values)), name="subject_id")
        stats = analyze_numeric(values, ids)
        assert stats["has_nan"] is True
        assert stats["has_neg"] is False
        assert stats["min_digits"] == _digit_to_int("0000000000000000000" + "00000000")  # E18...E0 + E-1...E-8
        assert stats["max_digits"] == _digit_to_int("0000000000000000000" + "00000000")  # E18...E0 + E-1...E-8
        assert stats["min11"] == []
        assert stats["max11"] == []

    def test_empty(self):
        values = pd.Series([], name="vals")
        ids = pd.Series(range(len(values)), name="subject_id")
        stats = analyze_numeric(values, ids)
        assert stats["has_nan"] is False
        assert stats["has_neg"] is False
        assert stats["min_digits"] == _digit_to_int("0000000000000000000" + "00000000")  # E18...E0 + E-1...E-8
        assert stats["max_digits"] == _digit_to_int("0000000000000000000" + "00000000")  # E18...E0 + E-1...E-8
        assert stats["min11"] == []
        assert stats["max11"] == []

    def test_min_max_int_value(self):
        min_int64_val = np.iinfo(np.int64).min
        max_int64_val = np.iinfo(np.int64).max
        values = pd.Series(np.repeat([min_int64_val, max_int64_val], 100), name="vals")
        ids = pd.Series(range(len(values)), name="subject_id")
        stats = analyze_numeric(values, ids)
        assert stats["has_nan"] is False
        assert stats["has_neg"] is True
        assert stats["min_digits"] == _digit_to_int("9223372036854776000" + "00000000")  # E18...E0 + E-1...E-8
        assert stats["max_digits"] == _digit_to_int("9223372036854776000" + "00000000")  # E18...E0 + E-1...E-8
        assert stats["min11"] == [-9.223372036854776e18] * 11
        assert stats["max11"] == [+9.223372036854776e18] * 11
        assert stats["cnt_values"] == {
            -9223372036854775808: 100,
            9223372036854775807: 100,
        }

    def test_precision_higher_than_limit(self):
        values = pd.Series([0.111111112222] * 50 + [0.999999998888] * 50, name="vals")
        ids = pd.Series(range(len(values)), name="subject_id")
        stats = analyze_numeric(values, ids)
        assert stats["min_digits"] == _digit_to_int("0000000000000000000" + "11111111")
        assert stats["max_digits"] == _digit_to_int("0000000000000000000" + "99999999")
        assert stats["min11"] == [0.111111112222] * 11
        assert stats["max11"] == [0.999999998888] * 11

    def test_decimals_above_limit(self):
        values = pd.Series([9e30] * 50 + [1e30] * 50, name="vals")
        ids = pd.Series(range(len(values)), name="subject_id")
        stats = analyze_numeric(values, ids)
        assert stats["min_digits"] == _digit_to_int("0000000000000000000" + "00000000")
        assert stats["max_digits"] == _digit_to_int("0000000000000000000" + "00000000")
        assert stats["min11"] == [1e30] * 11
        assert stats["max11"] == [9e30] * 11

    def test_min11_max11_overlapping(self):
        values = pd.Series(list(range(11)), name="vals")
        ids = pd.Series(range(len(values)), name="subject_id")
        stats = analyze_numeric(values, ids)
        assert stats["min11"] == list(np.linspace(0, 10, 11))
        assert stats["max11"] == list(np.linspace(10, 0, 11))

    def test_rare_digits(self):
        values = pd.Series(np.repeat([0.1, 0.2, 0.3, 0.4, 0.5], 10), name="vals")
        ids = pd.Series([0] * 40 + list(range(1, 11)), name="subject_id")
        stats = analyze_numeric(values, ids)
        assert stats["min11"] == [0.1] + [0.5] * 10
        assert stats["max11"] == [0.5] * 10 + [0.4]


class TestDigitAnalyzeReduce:
    @staticmethod
    def stats_template(
        min_digits=None,
        max_digits=None,
        min11=None,
        max11=None,
        has_nan=False,
        has_neg=False,
        cnt_values=None,
    ):
        if min_digits is None:
            min_digits = _digit_to_int("0000000000000000000" + "00000000")
        if max_digits is None:
            max_digits = _digit_to_int("0000000000000000999" + "00000000")
        if min11 is None:
            min11 = np.linspace(0, 10, 11)
        if max11 is None:
            max11 = np.linspace(999, 989, 11)
        return {
            "has_nan": has_nan,
            "has_neg": has_neg,
            "min_digits": min_digits,
            "max_digits": max_digits,
            "min11": min11,
            "max11": max11,
            "cnt_values": cnt_values,
        }

    @pytest.fixture
    def stats_positives(self):
        return self.stats_template(
            min_digits=_digit_to_int("0000000000000000000" + "00000000"),
            max_digits=_digit_to_int("0000000000000000039" + "90000000"),
            min11=np.linspace(0, 11, 11),
            max11=np.linspace(30, 19, 11),
        )

    @pytest.fixture
    def stats_negatives(self):
        return self.stats_template(
            min_digits=_digit_to_int("0000000000000000010" + "00000000"),
            max_digits=_digit_to_int("0000000000000000099" + "90000000"),
            min11=np.linspace(-90, -79, 11),
            max11=np.linspace(-10, -21, 11),
            has_neg=True,
        )

    @pytest.fixture
    def stats_nulls(self, stats_positives):
        return stats_positives | {"has_nan": True}

    def test_positives_only(self, stats_positives):
        result = analyze_reduce_numeric([stats_positives] * 2, encoding_type=ModelEncodingType.tabular_numeric_digit)
        assert result["cardinalities"] == {"E1": 4, "E0": 10, "E-1": 10}
        assert result["has_nan"] is False
        assert result["has_neg"] is False
        assert result["min_decimal"] == -1
        assert result["max_decimal"] == 1
        assert result["min_digits"] == _digit_to_int("0000000000000000000" + "00000000")
        assert result["max_digits"] == _digit_to_int("0000000000000000039" + "90000000")
        assert result["min5"] == [2.2, 3.3000000000000003, 3.3000000000000003, 4.4, 4.4]
        assert result["max5"] == [27.8, 26.7, 26.7, 25.6, 25.6]

    def test_negatives_only(self, stats_negatives):
        result = analyze_reduce_numeric([stats_negatives] * 2, encoding_type=ModelEncodingType.tabular_numeric_digit)
        assert result["cardinalities"] == {"E-1": 10, "E0": 10, "E1": 9, "neg": 2}
        assert result["has_nan"] is False
        assert result["has_neg"] is True
        assert result["min_decimal"] == -1
        assert result["max_decimal"] == 1
        assert result["min_digits"] == _digit_to_int("0000000000000000010" + "00000000")
        assert result["max_digits"] == _digit_to_int("0000000000000000099" + "90000000")
        assert result["min5"] == [-87.8, -86.7, -86.7, -85.6, -85.6]
        assert result["max5"] == [-12.2, -13.3, -13.3, -14.4, -14.4]

    def test_positives_and_negatives(self, stats_positives, stats_negatives):
        result = analyze_reduce_numeric(
            [stats_positives, stats_negatives], encoding_type=ModelEncodingType.tabular_numeric_digit
        )
        assert result["cardinalities"] == {"E-1": 10, "E0": 10, "E1": 10, "neg": 2}
        assert result["has_nan"] is False
        assert result["has_neg"] is True
        assert result["min_decimal"] == -1
        assert result["max_decimal"] == 1
        assert result["min_digits"] == _digit_to_int("0000000000000000000" + "00000000")
        assert result["max_digits"] == _digit_to_int("0000000000000000099" + "90000000")
        assert result["min5"] == [-84.5, -83.4, -82.3, -81.2, -80.1]
        assert result["max5"] == [24.5, 23.4, 22.299999999999997, 21.2, 20.1]

    def test_positives_and_nulls(self, stats_positives, stats_nulls):
        result = analyze_reduce_numeric(
            [stats_positives, stats_nulls], encoding_type=ModelEncodingType.tabular_numeric_digit
        )
        assert result["cardinalities"] == {"E1": 4, "E0": 10, "E-1": 10, "nan": 2}
        assert result["has_nan"] is True
        assert result["has_neg"] is False
        assert result["min_decimal"] == -1
        assert result["max_decimal"] == 1
        assert result["min_digits"] == _digit_to_int("0000000000000000000" + "00000000")
        assert result["max_digits"] == _digit_to_int("0000000000000000039" + "90000000")

    def test_value_protection_off(self, stats_positives):
        result = analyze_reduce_numeric(
            [stats_positives], value_protection=False, encoding_type=ModelEncodingType.tabular_numeric_digit
        )
        assert result["cardinalities"] == {"E1": 4, "E0": 10, "E-1": 10}
        assert result["has_nan"] is False
        assert result["has_neg"] is False
        assert result["min_decimal"] == -1
        assert result["max_decimal"] == 1
        assert result["min_digits"] == _digit_to_int("0000000000000000000" + "00000000")
        assert result["max_digits"] == _digit_to_int("0000000000000000039" + "90000000")
        # most extreme values are included
        assert result["min5"] == [
            0.0,
            1.1,
            2.2,
            3.3000000000000003,
            4.4,
        ]
        # most extreme values are included
        assert result["max5"] == [30.0, 28.9, 27.8, 26.7, 25.6]


class TestDigitEncode:
    @pytest.fixture
    def stats(self):
        return {
            "encoding_type": ModelEncodingType.tabular_numeric_digit.value,
            "cardinalities": {"E1": 10, "E0": 10, "E-1": 10, "nan": 2, "neg": 2},
            "has_neg": True,
            "has_nan": True,
            "min_digits": _digit_to_int("0000000000000000000" + "00000000"),
            "max_digits": _digit_to_int("0000000000000000099" + "90000000"),
            "max_decimal": 1,
            "min_decimal": -1,
            "min5": [-99.0] * 5,
            "max5": [+99.9] * 5,
        }

    def test_known_positives_negatives_nulls(self, stats):
        values = pd.Series(np.repeat([10, -20, 0.1, -0.2, pd.NA], 100), name="vals")
        expected = pd.DataFrame(
            [[0, 0, 1, 0, 0]] * 100  # 10
            + [[0, 1, 2, 0, 0]] * 100  # -20
            + [[0, 0, 0, 0, 1]] * 100  # 0.1
            + [[0, 1, 0, 0, 2]] * 100  # -0.2
            + [[1, 0, 0, 0, 0]] * 100,  # None
            columns=["nan", "neg", "E1", "E0", "E-1"],
        )
        encoded = encode_numeric(values, stats)
        pd.testing.assert_frame_equal(encoded, expected)

    def test_unknown_nulls_and_negatives(self, stats):
        stats["has_neg"] = False
        stats["has_nan"] = False
        values = pd.Series(np.repeat([10, -20, pd.NA], 100), name="vals")
        expected = pd.DataFrame(
            [[1, 0, 0]] * 100  # 10
            + [[2, 0, 0]] * 100  # -20 -> 20
            + [[0, 0, 0]] * 100,  # None -> 0
            columns=["E1", "E0", "E-1"],
        )
        encoded = encode_numeric(values, stats)
        pd.testing.assert_frame_equal(encoded, expected)

    def test_values_outside_of_bounds(self, stats):
        values = pd.Series(np.repeat([999, 0.999], 100), name="vals")
        expected = pd.DataFrame(
            [[0, 0, 9, 9, 9]] * 100  # 999 -> 99.9
            + [[0, 0, 0, 0, 9]] * 100,  # 0.999 -> 0.9
            columns=["nan", "neg", "E1", "E0", "E-1"],
        )
        encoded = encode_numeric(values, stats)
        pd.testing.assert_frame_equal(encoded, expected)

    def test_empty(self, stats):
        values = pd.Series([], name="vals")
        expected = pd.DataFrame(columns=["nan", "neg", "E1", "E0", "E-1"])
        encoded = encode_numeric(values, stats)
        pd.testing.assert_frame_equal(encoded, expected, check_index_type=False, check_dtype=False)

    def test_extra_long_and_high_precision(self):
        stats = {
            "encoding_type": ModelEncodingType.tabular_numeric_digit.value,
            "cardinalities": {f"E{18 - idx}": 10 for idx in range(27)} | {"neg": 2},
            "has_neg": True,
            "has_nan": False,
            "min_digits": _digit_to_int("0000000000000000000" + "00000000"),
            "max_digits": _digit_to_int("9999999999999999999" + "99999999"),
            "max_decimal": 18,
            "min_decimal": -8,
            "min5": [-9999999999999999999.99999999] * 5,
            "max5": [+9999999999999999999.99999999] * 5,
        }
        min_int64_val = np.iinfo(np.int64).min
        max_int64_val = np.iinfo(np.int64).max
        values = pd.Series(
            [
                123456789987654321123456789.987654321123456789,
                min_int64_val,
                max_int64_val,
                123456789.123456,
            ],
            name="vals",
        )
        expected = pd.DataFrame(
            [[0] * 28]  # 123456789987654321123456789.987654321123456789 -> 0.0
            + [_ints("1922337203685477600000000000")]  # properly encoded
            + [_ints("0922337203685477600000000000")]  # properly encoded
            + [_ints("0000000000012345678912345600")],  # properly encoded
            columns=["neg"] + [f"E{18 - idx}" for idx in range(27)],
        )
        encoded = encode_numeric(values, stats)
        pd.testing.assert_frame_equal(encoded, expected, check_index_type=False, check_dtype=False)


class TestDigitDecode:
    @pytest.fixture
    def stats(self):
        return {
            "encoding_type": ModelEncodingType.tabular_numeric_digit.value,
            "cardinalities": {"E1": 10, "E0": 10, "E-1": 10, "nan": 2, "neg": 2},
            "has_neg": True,
            "has_nan": True,
            "min_digits": _digit_to_int("0000000000000000000" + "00000000"),
            "max_digits": _digit_to_int("0000000000000000099" + "90000000"),
            "max_decimal": 1,
            "min_decimal": -1,
            "min5": [-90.0] * 5,
            "max5": [+90.0] * 5,
        }

    def test_known_positives_negatives_nulls(self, stats):
        encoded = pd.DataFrame(
            [[0, 0, 1, 0, 0]] * 100  # 10
            + [[0, 1, 2, 0, 0]] * 100  # -20
            + [[0, 0, 0, 0, 1]] * 100  # 0.1
            + [[0, 1, 0, 0, 2]] * 100  # -0.2
            + [[1, 0, 0, 0, 0]] * 100,  # None
            columns=["nan", "neg", "E1", "E0", "E-1"],
        )
        expected = pd.Series(np.repeat([10, -20, 0.1, -0.2, pd.NA], 100))
        decoded = decode_numeric(encoded, stats)
        pd.testing.assert_series_equal(decoded, expected, check_dtype=False)

    def test_empty(self, stats):
        encoded = pd.DataFrame(columns=["nan", "neg", "E1", "E0", "E-1"])
        expected = pd.Series([])
        encoded = decode_numeric(encoded, stats)
        pd.testing.assert_series_equal(encoded, expected, check_dtype=False)

    def test_extra_long_and_high_precision(self):
        stats = {
            "encoding_type": ModelEncodingType.tabular_numeric_digit.value,
            "cardinalities": {f"E{18 - idx}": 10 for idx in range(27)} | {"neg": 2},
            "has_neg": True,
            "has_nan": False,
            "min_digits": _digit_to_int("0000000000000000000" + "00000000"),
            "max_digits": _digit_to_int("9999999999999999999" + "99999999"),
            "max_decimal": 18,
            "min_decimal": -8,
            "min5": [-9999999999999999999.99999999] * 5,
            "max5": [+9999999999999999999.99999999] * 5,
        }
        min_int64_val = np.iinfo(np.int64).min
        max_int64_val = np.iinfo(np.int64).max
        expected = pd.Series(
            [
                min_int64_val,
                max_int64_val,
                123456789.123456,
            ],
        )
        encoded = pd.DataFrame(
            [_ints("1922337203685477600000000000")]  # min_int64_val
            + [_ints("0922337203685477600000000000")]  # max_int64_val
            + [_ints("0000000000012345678912345600")],  # just long and high precision
            columns=["neg"] + [f"E{18 - idx}" for idx in range(27)],
        )
        decoded = decode_numeric(encoded, stats)
        pd.testing.assert_series_equal(decoded, expected, check_dtype=False)

    def test_never_less_than_min_and_more_than_max(self, stats):
        expected = pd.Series([-90.0, +90.0])
        encoded = pd.DataFrame(
            [_ints("1000000000000000099900000000")]  # -99.9
            + [_ints("0000000000000000099900000000")],  # +99.9
            columns=["neg"] + [f"E{18 - idx}" for idx in range(27)],
        )
        decoded = decode_numeric(encoded, stats)
        pd.testing.assert_series_equal(decoded, expected, check_dtype=False)


class TestNumericBinned:
    def test_analyze(self):
        values1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], name="x")
        rkeys1 = pd.Series(range(len(values1)), name="id")
        stats1 = analyze_numeric(values1, rkeys1, encoding_type=ModelEncodingType.tabular_numeric_binned)
        values2 = pd.Series([0] * 100, name="x")
        rkeys2 = pd.Series(range(len(values2)), name="id")
        stats2 = analyze_numeric(values2, rkeys2, encoding_type=ModelEncodingType.tabular_numeric_binned)
        stats = analyze_reduce_numeric(
            [stats1, stats2], value_protection=False, encoding_type=ModelEncodingType.tabular_numeric_binned
        )
        assert stats["bins"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert NUMERIC_BINNED_UNKNOWN_TOKEN in stats["codes"]
        assert NUMERIC_BINNED_MIN_TOKEN in stats["codes"]
        assert NUMERIC_BINNED_MAX_TOKEN not in stats["codes"]
        # test case where all values are value protected
        values1 = pd.Series([1, 2, 3, 4, 5], name="x")
        rkeys1 = pd.Series(range(len(values1)), name="id")
        stats1 = analyze_numeric(values1, rkeys1, encoding_type=ModelEncodingType.tabular_numeric_binned)
        stats = analyze_reduce_numeric(
            [stats1], value_protection=True, encoding_type=ModelEncodingType.tabular_numeric_binned
        )
        assert stats["bins"] == [0]
        assert stats["cardinalities"][NUMERIC_BINNED_SUB_COL_SUFFIX] == 1
        assert NUMERIC_BINNED_UNKNOWN_TOKEN in stats["codes"]

    def test_encode_decode(self):
        stats = {
            "encoding_type": ModelEncodingType.tabular_numeric_binned.value,
            "cardinalities": {"bin": 11},
            "codes": {NUMERIC_BINNED_UNKNOWN_TOKEN: 0, NUMERIC_BINNED_MIN_TOKEN: 1},
            "min_decimal": 0,
            "bins": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        }
        # NA is mapped to UNKNOWN, because it didn't appear in the training data
        assert all(encode_numeric(pd.Series([pd.NA]), stats).bin == [0])
        # 0 is mapped to MIN
        assert all(encode_numeric(pd.Series([0]), stats).bin == [1])
        assert all(decode_numeric(pd.DataFrame({"bin": [1]}), stats) == [0])
        # -1 is mapped to first bin, as it's less than MIN
        assert all(encode_numeric(pd.Series([-1]), stats).bin == [2])
        assert all(decode_numeric(pd.DataFrame({"bin": [2]}), stats) == [0])
        # 1.5 is mapped to second bin
        assert all(encode_numeric(pd.Series([1.5]), stats).bin == [3])
        assert all(decode_numeric(pd.DataFrame({"bin": [3]}), stats) == [1])
        # 15 is mapped to last bin as it's more than MAX
        assert all(encode_numeric(pd.Series([15]), stats).bin == [stats["cardinalities"]["bin"] - 1])
        assert all(decode_numeric(pd.DataFrame({"bin": [10]}), stats) == [8])

    class TestQuantiles:
        def test_large_float_column(self):
            # generate the data using a normal distribution
            data = np.random.normal(0, 1, 20000)
            # trim the digits just to force numbers to repeat
            data = np.array([round(x, 3) for x in data])
            data = pd.Series(data, name="col")
            keys = pd.Series(np.arange(len(data)), name="id")
            stats = analyze_numeric(data, keys, encoding_type=ModelEncodingType.tabular_numeric_binned)

            quantiles = stats["quantiles"]
            assert len(quantiles) == NUMERIC_BINNED_MAX_QUANTILES

            # check the types of the quantiles
            assert all(isinstance(x, (float, np.floating)) for x in data), "Not all values are floats"

            # 68% of the quantiles should be within 1 std dev of the mean
            assert len([q for q in quantiles if -1 <= q <= 1]) > 650

            # 95% of the quantiles should be within 2 std dev of the mean
            assert len([q for q in quantiles if -2 <= q <= 2]) > 930

        def test_large_integer_column(self):
            # generate the data using a normal distribution
            data = np.random.normal(0, 1000, 20000)
            # round the numbers to be integers
            data = np.array([round(x) for x in data])

            data = pd.Series(data, name="col")
            keys = pd.Series(np.arange(len(data)), name="id")
            stats = analyze_numeric(data, keys, encoding_type=ModelEncodingType.tabular_numeric_binned)

            quantiles = stats["quantiles"]
            assert len(quantiles) == NUMERIC_BINNED_MAX_QUANTILES

            # check the types of the quantiles
            assert all(isinstance(x, (int, np.integer)) for x in data), "Not all values are integers"

            # 68% of the quantiles should be within 1 std dev of the mean
            assert len([q for q in quantiles if -1000 <= q <= 1000]) > 650

            # 95% of the quantiles should be within 2 std dev of the mean
            assert len([q for q in quantiles if -2000 <= q <= 2000]) > 930


class TestNumericDiscrete:
    def test_analyze(self):
        values1 = pd.Series([1, 2, 3, 4], name="x")
        rkeys1 = pd.Series(range(len(values1)), name="id")
        stats1 = analyze_numeric(values1, rkeys1, encoding_type=ModelEncodingType.tabular_numeric_discrete)
        values2 = pd.Series([0] * 100, name="x")
        rkeys2 = pd.Series(range(len(values2)), name="id")
        stats2 = analyze_numeric(values2, rkeys2, encoding_type=ModelEncodingType.tabular_numeric_discrete)
        stats = analyze_reduce_numeric(
            [stats1, stats2], value_protection=False, encoding_type=ModelEncodingType.tabular_numeric_discrete
        )
        assert NUMERIC_DISCRETE_UNKNOWN_TOKEN in stats["codes"]
        assert all([v in stats["codes"].keys() for v in values1])

    def test_encode_decode(self):
        stats = {
            "encoding_type": ModelEncodingType.tabular_numeric_discrete.value,
            "cardinalities": {"cat": 6},
            "codes": {"_RARE_": 0, 1: 1, 2: 2, 3: 3, 4: 4, 0: 5},
            "min_decimal": 0,
        }
        # NA is mapped to UNKNOWN, because it didn't appear in the training data
        assert all(encode_numeric(pd.Series([pd.NA]), stats).cat == [0])
        assert all(encode_numeric(pd.Series([-1]), stats).cat == [0])
        assert all(encode_numeric(pd.Series([0]), stats).cat == stats["codes"][0])
        assert all(decode_numeric(pd.DataFrame({"cat": [stats["codes"][0]]}), stats) == [0])
        assert all(decode_numeric(pd.DataFrame({"cat": [stats["codes"][1]]}), stats) == [1])

    def test_encode_decode_edge_case(self):
        stats = {
            "encoding_type": ModelEncodingType.tabular_numeric_discrete.value,
            "cardinalities": {"cat": 1},
            "codes": {"_RARE_": 0},
            "min_decimal": 0,
        }
        encoded_df = encode_numeric(pd.Series([pd.NA, 1, 2, 3, 4, 5]), stats)
        decoded_df = decode_numeric(encoded_df, stats)
        assert len(encoded_df) == len(decoded_df)
        assert pd.isna(decoded_df).all()


class TestEdgeCases:
    def test_digit_min_max_decimal_bug(self):
        root_keys = pd.Series([1, 2, 3, 4, 5], name="key")
        values = pd.Series([500000, 600000, 700000, np.nan, np.nan], name="dig")

        stats1 = analyze_numeric(values, root_keys, encoding_type=ModelEncodingType.tabular_numeric_digit)
        stats = analyze_reduce_numeric([stats1])
        encoded_df = encode_numeric(values, stats)
        decoded_ser = decode_numeric(encoded_df, stats)

        assert len(encoded_df) == len(decoded_ser)
        assert decoded_ser.isna().all()
