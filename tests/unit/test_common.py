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

from logging import Logger
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from mostlyai.engine.domain import ModelEncodingType
from mostlyai.engine._common import (
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
    CtxSequenceLengthError,
    _dp_approx_bounds,
    apply_encoding_type_dtypes,
    dp_non_rare,
    dp_quantiles,
    find_distinct_bins,
    get_argn_name,
    get_columns_from_cardinalities,
    get_ctx_sequence_length,
    get_max_data_points_per_sample,
    get_sub_columns_from_cardinalities,
    get_sub_columns_nested,
    get_sub_columns_nested_from_cardinalities,
    handle_with_nested_lists,
    is_sequential,
    safe_convert_datetime,
    safe_convert_numeric,
    safe_convert_string,
    skip_if_error,
)


class TestSafeConvertDatetime:
    @pytest.mark.parametrize(
        "input,expect",
        [
            (
                [],
                [],
            ),
            (
                ["1978-05-24", "1976-06-22 12:43:34"],
                ["1978-05-24 00:00:00", "1976-06-22 12:43:34"],
            ),
            (
                ["", "xx", "3/1/2022", "2/28/2022"],
                [pd.NaT, pd.NaT, "2022-03-01", "2022-02-28"],
            ),
            (
                ["", "1800-01-01", "3/1/2022", "2/28/2022"],
                [pd.NaT, "1800-01-01", "2022-01-03", "2022-02-28"],
            ),
            (
                ["2021-10-05 16:06:38.995523400", "1996-04-27T20:36:11.420Z"],
                [
                    "2021-10-05 16:06:38.995523",
                    "1996-04-27 20:36:11.420000",
                ],
            ),
        ],
    )
    def test_plain_inputs(self, input, expect):
        result = safe_convert_datetime(pd.Series(input))
        assert_series_equal(result, pd.Series(expect, dtype="datetime64[us]"))

    @pytest.mark.parametrize(
        "input,expect",
        [
            (
                ["1978-05-24", "1976-06-22 12:43:34"],
                ["1978-05-24", "1976-06-22"],
            ),
        ],
    )
    def test_plain_inputs_date_only(self, input, expect):
        result = safe_convert_datetime(pd.Series(input), date_only=True)
        assert_series_equal(result, pd.Series(expect, dtype="datetime64[us]"))

    def test_nested_lists(self):
        input = [
            ["1978-05-24", "1976-06-22 12:43:34"],
            [],
            ["2022-01-02 12:31"],
        ]
        expect = pd.Series(
            [
                np.array(
                    ["1978-05-24 00:00:00", "1976-06-22 12:43:34"],
                    dtype="datetime64[us]",
                ),
                np.array([], dtype="datetime64[us]"),
                np.array(["2022-01-02 12:31:00"], dtype="datetime64[us]"),
            ]
        )

        result = safe_convert_datetime(pd.Series(input))
        assert_series_equal(result, expect)


def test_safe_convert_datetime(tmp_path):
    values = pd.Series(["", "xx", "1/3/2022", "28/2/2022"])
    values_date = safe_convert_datetime(values)
    assert all(values_date.isna() == [True, True, False, False])
    assert all(values_date[2:] == ["2022-03-01", "2022-02-28"])


class TestSafeConvertDigit:
    @pytest.mark.parametrize(
        "input,expect",
        [
            (
                [],
                pd.Series([], dtype="int64"),
            ),
            (
                [1, 2, 3],
                [1, 2, 3],
            ),
            (
                ["-35.3", "100.0", "003", " .348 ", "-0", "N/A"],
                [-35.3, 100, 3, 0.348, 0, np.nan],
            ),
            (
                ["tax: 1.23 EURO", "tax: -1", "+.23 EURO", "#23", "E"],
                [1.23, -1, 0.23, 23, np.nan],
            ),
            (
                ["-35.3", "100.0", "003", " .348 ", "-0", "N/A"],
                [-35.3, 100, 3, 0.348, 0, np.nan],
            ),
            (["x4y2", "4+2", "1.2.3"], [4, 4, 1.2]),
            (
                pd.Series([True, False, None], dtype="boolean"),
                pd.Series([1, 0, np.nan], dtype="Int8"),
            ),
            (
                ["1.2e+3", "1e3", ".1e3", "0.1e-1"],
                [1200, 1000, 100, 0.01],
            ),
            (["1.2E+3", "1E03", ".1E3", "0.1E-1"], [1200, 1000, 100, 0.01]),
        ],
    )
    def test_flatten_inputs(self, input, expect):
        result = safe_convert_numeric(pd.Series(input))
        assert_series_equal(result, pd.Series(expect))

    def test_nested_lists(self):
        input = [
            [100, "003", "n/a"],
            [],
            ["-0"],
            ["1.30 EURO", 0.32],
            [],
        ]
        expect = pd.Series(
            [
                np.array([100, 3, np.nan], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([0], dtype=np.float64),
                np.array([1.3, 0.32], dtype=np.float64),
                np.array([], dtype=np.float64),
            ]
        )

        result = safe_convert_numeric(pd.Series(input))
        assert_series_equal(result, expect)


class TestSafeConvertString:
    @pytest.mark.parametrize(
        "input,expect",
        [
            (
                [],
                [],
            ),
            (
                ["a", "as%", ""],
                ["a", "as%", ""],
            ),
            (
                [1, 23],
                ["1", "23"],
            ),
            (
                [1, 23.1],
                ["1.0", "23.1"],
            ),
        ],
    )
    def test_flatten_inputs(self, input, expect):
        result = safe_convert_string(pd.Series(input))
        assert_series_equal(result, pd.Series(expect, dtype="string"))

    def test_nested_lists(self):
        input = [
            [100, "003", "n/a"],
            [],
            ["abc %^&", 0.32],
            [],
        ]
        expect = pd.Series(
            [
                np.array(["100", "003", "n/a"], dtype=str),
                np.array([], dtype=str),
                np.array(["abc %^&", "0.32"], dtype=str),
                np.array([], dtype=str),
            ]
        )

        result = safe_convert_string(pd.Series(input))
        assert_series_equal(result, expect)


def test_apply_encoding_type_dtypes():
    # no data loss
    df = pd.DataFrame(
        {
            "ints": ["1", None],
            "floats": ["1.4", None],
            "texts": ["a", None],
            "datetimes": ["2020-01-01", None],
        }
    ).convert_dtypes()
    expected_df = pd.DataFrame(
        {
            "ints": [1, None],
            "floats": [1.4, None],
            "texts": ["a", None],
            "datetimes": pd.Series([pd.to_datetime("2020-01-01"), None], dtype="datetime64[us]"),
        }
    ).convert_dtypes()
    df = apply_encoding_type_dtypes(
        df=df,
        encoding_types={
            "ints": ModelEncodingType.tabular_numeric_digit,
            "floats": ModelEncodingType.tabular_numeric_digit,
            "texts": ModelEncodingType.tabular_categorical,
            "datetimes": ModelEncodingType.tabular_datetime,
        },
    )
    assert_frame_equal(df, expected_df)

    # some data loss
    df = pd.DataFrame(
        {
            "ints": ["1", "abc"],
            "floats": ["1.4", "2.0"],
            "texts": ["a", "b"],
            "datetimes": ["2020-01-01", "date"],
        }
    ).convert_dtypes()
    expected_df = pd.DataFrame(
        {
            "ints": [1, None],
            "floats": [1.4, 2.0],
            "texts": ["a", "b"],
            "datetimes": pd.Series([pd.to_datetime("2020-01-01"), None], dtype="datetime64[us]"),
        }
    ).convert_dtypes()
    df = apply_encoding_type_dtypes(
        df=df,
        encoding_types={
            "ints": ModelEncodingType.tabular_numeric_digit,
            "floats": ModelEncodingType.tabular_numeric_digit,
            "texts": ModelEncodingType.tabular_categorical,
            "datetimes": ModelEncodingType.tabular_datetime,
        },
    )
    assert_frame_equal(df, expected_df)

    # unspecified encoding types are kept unmodified
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "digits": ["1.4", "2.0"],
        }
    ).convert_dtypes()
    expected_df = pd.DataFrame(
        {
            "id": [1, 2],
            "digits": [1.4, 2.0],
        }
    ).convert_dtypes()
    df = apply_encoding_type_dtypes(
        df=df,
        encoding_types={
            "digits": ModelEncodingType.tabular_numeric_digit,
        },
    )
    assert_frame_equal(df, expected_df)


def test_skip_if_error():
    @skip_if_error
    def div_zero():
        return 1 / 0

    @skip_if_error
    def successful_run():
        return True

    with patch.object(Logger, "warning") as log_warning_mock:
        div_zero()
        assert (
            log_warning_mock.call_args[0][0]
            == "test_skip_if_error.<locals>.div_zero failed with <class 'ZeroDivisionError'>: division by zero"
        )
        log_warning_mock.reset_mock()
        res = successful_run()
        assert log_warning_mock.call_count == 0
        assert res


@pytest.mark.parametrize(
    "series,expected_bool",
    [
        (pd.Series([]), False),
        (pd.Series([[]]), True),
        (pd.Series([1, 2, 3]), False),
        (pd.Series(["abc"]), False),
        (pd.Series([[1], [2, 3]]), True),
        (pd.Series([1, [2, 3], 4]), True),
    ],
)
def test_is_sequential(series, expected_bool):
    assert is_sequential(series) == expected_bool


class TestHandleWithNestedLists:
    @handle_with_nested_lists
    def sum_one(self, values: pd.Series):
        return values + 1

    def test_func_called_with_flatten_series(self):
        input = pd.Series(np.zeros(4))
        result = self.sum_one(input)

        expected = pd.Series(np.ones(4))
        assert_series_equal(result, expected)

    def test_func_called_with_nested_lists(self):
        input = pd.Series([[0], [0, 0, 0], [0]])
        result = self.sum_one(input)

        expected = pd.Series([[1], [1, 1, 1], [1]])
        assert_series_equal(result, expected)

    def test_func_called_with_nested_lists_and_duplicates_in_index(self):
        input = pd.Series([[0], [0], [0]], index=[0, 0, 0])
        result = self.sum_one(input)

        expected = pd.Series([[1], [1], [1]], index=[0, 0, 0])
        assert_series_equal(result, expected)


def test_get_argn_name():
    assert get_argn_name("tgt", "t0", "c0", "cat") == "tgt:t0/c0__cat"
    assert get_argn_name("tgt", "t0", "c0", "") == "tgt:t0/c0__"
    assert get_argn_name("tgt", "t0", "c0") == "tgt:t0/c0"
    assert get_argn_name("tgt", "t0", "") == "tgt:t0/"
    assert get_argn_name("tgt", "t0") == "tgt:t0"
    assert get_argn_name("tgt", "") == "tgt:"
    assert get_argn_name("tgt") == "tgt"
    assert get_argn_name("tgt", None, "c0", "cat") == "tgt:/c0__cat"
    assert get_argn_name("tgt", None, None, "sidx") == "tgt:/__sidx"


def test_get_sub_columns_from_cardinalities():
    cards = {"c0__E1": 10, "c0__E0": 10, "c1__value": 2}
    assert get_sub_columns_from_cardinalities(cards) == [
        "c0__E1",
        "c0__E0",
        "c1__value",
    ]


def test_columns_from_cardinalities():
    cards = {"c0__E1": 10, "c0__E0": 10, "c1__value": 2}
    assert get_columns_from_cardinalities(cards) == ["c0", "c1"]


def test_get_max_data_points_per_sample():
    # flat stats
    stats = {
        "columns": {
            "age": {
                "cardinalities": {"E1": 7, "E0": 10},
            },
        },
        "seq_len": {
            "min": 1,
            "median": 1,
            "max": 1,
        },
    }
    assert get_max_data_points_per_sample(stats) == 2
    # sequential stats
    stats = {
        "columns": {
            "age": {
                "cardinalities": {"E1": 7, "E0": 10},
            },
        },
        "seq_len": {
            "min": 1,
            "median": 3,
            "max": 15,
        },
    }
    assert get_max_data_points_per_sample(stats) == 2 * 15
    # flat stats with sequential columns
    stats = {
        "columns": {
            "age": {
                "cardinalities": {"E1": 7, "E0": 10},
            },
            "description": {
                "cardinalities": {"tokens": 44},
                "seq_len": {"min": 3, "max": 10, "median": 5},
            },
        },
        "seq_len": {
            "min": 1,
            "median": 1,
            "max": 1,
        },
    }
    assert get_max_data_points_per_sample(stats) == 2 + 1 * 10


class TestGetSubColumnsNestedFromCardinalities:
    def test_groupby_processor(self):
        cards = {
            "ctxseq:t1/c0__E1": 10,
            "ctxseq:t1/c0__E0": 10,
            "ctxflt:t0/c1__value": 2,
        }
        assert get_sub_columns_nested_from_cardinalities(cards, groupby="processor") == {
            "ctxseq": ["ctxseq:t1/c0__E1", "ctxseq:t1/c0__E0"],
            "ctxflt": ["ctxflt:t0/c1__value"],
        }

    def test_groupby_tables(self):
        cards = {
            "ctxseq:t1/c0__E1": 10,
            "ctxseq:t1/c0__E0": 10,
            "ctxseq:t2/c1__value": 2,
        }
        assert get_sub_columns_nested_from_cardinalities(cards, groupby="tables") == {
            "ctxseq:t1": ["ctxseq:t1/c0__E1", "ctxseq:t1/c0__E0"],
            "ctxseq:t2": ["ctxseq:t2/c1__value"],
        }

    def test_groupby_columns(self):
        cards = {"tgt:t0/c0__E1": 10, "tgt:t0/c0__E0": 10, "tgt:t0/c1__value": 2}
        assert get_sub_columns_nested_from_cardinalities(cards, groupby="columns") == {
            "tgt:t0/c0": ["tgt:t0/c0__E1", "tgt:t0/c0__E0"],
            "tgt:t0/c1": ["tgt:t0/c1__value"],
        }


class TestGetCtxSequenceLength:
    def test_get_by_key(self):
        stats = {
            "columns": {
                "table0.col0": {
                    ARGN_PROCESSOR: "ctxflt",
                    ARGN_TABLE: "t0",
                    ARGN_COLUMN: "c0",
                },
                "table1.col0": {
                    ARGN_PROCESSOR: "ctxflt",
                    ARGN_TABLE: "t1",
                    ARGN_COLUMN: "c1",
                },
                "table1.col1": {
                    ARGN_PROCESSOR: "ctxseq",
                    "seq_len": {
                        "deciles": [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3],
                        "min": 1,
                        "max": 3,
                        "median": 2,
                    },
                    ARGN_TABLE: "t1",
                    ARGN_COLUMN: "c2",
                },
            }
        }
        assert get_ctx_sequence_length(stats, key="min") == {"ctxseq:t1": 1}
        assert get_ctx_sequence_length(stats, key="max") == {"ctxseq:t1": 3}
        assert get_ctx_sequence_length(stats, key="median") == {"ctxseq:t1": 2}

    def test_raise_exception_when_cols_not_converge(self):
        stats = {
            "columns": {
                "table0.col0": {
                    ARGN_PROCESSOR: "ctxflt",
                    ARGN_TABLE: "t0",
                    ARGN_COLUMN: "c0",
                },
                "table1.col0": {
                    ARGN_PROCESSOR: "ctxseq",
                    "seq_len": {
                        "any-key": 2,
                    },
                    ARGN_TABLE: "t1",
                    ARGN_COLUMN: "c1",
                },
                "table1.col1": {
                    ARGN_PROCESSOR: "ctxseq",
                    "seq_len": {
                        "any-key": 9,
                    },
                    ARGN_TABLE: "t1",
                    ARGN_COLUMN: "c2",
                },
            }
        }

        with pytest.raises(CtxSequenceLengthError):
            assert get_ctx_sequence_length(stats, key="any-key")


@pytest.mark.parametrize(
    "columns",
    (
        [
            "ctxseq:t0/c0__cat",
            "ctxseq:t0/c1__cat",
            "ctxflt:t1/c2__cat",
            "ctxseq:t2/c3__cat",
            "tgt:t0/c0__cat",
            "tgt:t0/c1__cat",
            "tgt:t0/c2__cat",
            "tgt:t0/c11__cat",
        ],
    ),
)
@pytest.mark.parametrize(
    "groupby, expected",
    [
        (
            "processor",
            {
                "ctxflt": ["ctxflt:t1/c2__cat"],
                "ctxseq": [
                    "ctxseq:t0/c0__cat",
                    "ctxseq:t0/c1__cat",
                    "ctxseq:t2/c3__cat",
                ],
                "tgt": [
                    "tgt:t0/c0__cat",
                    "tgt:t0/c1__cat",
                    "tgt:t0/c2__cat",
                    "tgt:t0/c11__cat",
                ],
            },
        ),
        (
            "tables",
            {
                "ctxflt:t1": ["ctxflt:t1/c2__cat"],
                "ctxseq:t0": ["ctxseq:t0/c0__cat", "ctxseq:t0/c1__cat"],
                "ctxseq:t2": ["ctxseq:t2/c3__cat"],
                "tgt:t0": [
                    "tgt:t0/c0__cat",
                    "tgt:t0/c1__cat",
                    "tgt:t0/c2__cat",
                    "tgt:t0/c11__cat",
                ],
            },
        ),
        (
            "columns",
            {
                "ctxflt:t1/c2": ["ctxflt:t1/c2__cat"],
                "ctxseq:t0/c0": ["ctxseq:t0/c0__cat"],
                "ctxseq:t0/c1": ["ctxseq:t0/c1__cat"],
                "ctxseq:t2/c3": ["ctxseq:t2/c3__cat"],
                "tgt:t0/c0": ["tgt:t0/c0__cat"],
                "tgt:t0/c1": ["tgt:t0/c1__cat"],
                "tgt:t0/c2": ["tgt:t0/c2__cat"],
                "tgt:t0/c11": ["tgt:t0/c11__cat"],
            },
        ),
    ],
)
def test_get_sub_columns_nested(columns, groupby, expected):
    assert get_sub_columns_nested(columns, groupby) == expected


def test_find_distinct_bins():
    # test continuous
    x = list(np.random.uniform(0, 1, 100))
    bins = find_distinct_bins(x, 10)
    assert all([b in x for b in bins])
    assert len(bins) == 11
    assert bins[0] == min(x)
    assert bins[-1] == max(x)
    buckets = pd.cut(x, bins, include_lowest=True).codes
    _, cnts = np.unique(buckets, return_counts=True)
    assert list(cnts) == [10] * 10
    # test continuous with dominant single value
    x += [0.5] * 100
    bins = find_distinct_bins(x, 10)
    assert len(bins) == 11
    assert 0.5 in bins
    # test few values
    x = np.repeat(np.random.uniform(0, 1, 10), 5)
    bins = find_distinct_bins(x, 10)
    assert all([b in x for b in bins])
    assert bins == list(sorted(set(x)))
    # test case where we exhaust search
    x = list(np.random.uniform(0, 1, 100))
    x += [0.5] * 2_000
    bins = find_distinct_bins(x, 10, n_max=20)
    assert len(bins) > 0


def test_dp_quantiles():
    epsilon = 1.0
    q = [0.05, 0.95]

    # given large enough sample size and epsilon, dp_quantiles should be reasonably close to the true quantiles
    values = np.random.lognormal(0, 1, 10_000) - 0.5  # right-skewed distribution with some negative values
    q5_dp, q95_dp = dp_quantiles(values, q, epsilon)
    assert abs(values[values < q5_dp].shape[0] / values.shape[0] - 0.05) < 0.005
    assert abs(values[values > q95_dp].shape[0] / values.shape[0] - 0.05) < 0.005

    # edge case: uniform distribution of the same value
    values = np.random.uniform(1, 1, 10_000)
    q5_dp, q95_dp = dp_quantiles(values, q, epsilon)
    assert q5_dp <= 1 <= q95_dp

    # small sample size
    # it should fall back to the unbounded quantiles method and not fail
    values = np.random.normal(0, 10, 100)
    lower, upper = _dp_approx_bounds(values, epsilon / (len(q) + 1))
    assert lower is None and upper is None
    q5_dp, q95_dp = dp_quantiles(values, q, epsilon)


def test_dp_non_rare():
    value_counts = {i: i for i in range(1, 101)}
    epsilon = 1.0
    selected, non_rare_ratio = dp_non_rare(value_counts, epsilon, threshold=10)
    # given epsilon=1.0, the noise added to the count should be within the range [-5, 5]
    # so in the worst case, we will have at least 4 and at most at most 14 rare categories
    assert len(selected) >= 86 and len(selected) <= 96
    assert non_rare_ratio >= 0.98 and non_rare_ratio <= 1.0
