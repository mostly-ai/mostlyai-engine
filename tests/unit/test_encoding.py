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

import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mostlyai.engine._common import SDEC_SUB_COLUMN_PREFIX, SIDX_SUB_COLUMN_PREFIX, SLEN_SUB_COLUMN_PREFIX
from mostlyai.engine._language.encoding import format_df
from mostlyai.engine._tabular.encoding import (
    _encode_col,
    _enrich_slen_sidx_sdec,
    flatten_frame,
    pad_horizontally,
)
from mostlyai.engine.domain import ModelEncodingType


def test_flatten_frame():
    df = pd.DataFrame(
        {
            "key": [1, 1, 2],
            "product": [3, 2, 9],
            "is_paid": [0, 1, 1],
        }
    )
    expected_df = pd.DataFrame(
        {
            "key": [1, 2],
            "product": [[3, 2], [9]],
            "is_paid": [[0, 1], [1]],
        }
    )
    assert_frame_equal(flatten_frame(df, "key"), expected_df)


def test_enrich_slen_sidx_sdec():
    df = pd.DataFrame(
        {
            "key": [1, 1, 2],
            "product": [3, 2, 9],
            "is_paid": [0, 1, 1],
        }
    )
    expected_df = pd.DataFrame(
        {
            f"{SLEN_SUB_COLUMN_PREFIX}cat": [2, 2, 1],
            f"{SIDX_SUB_COLUMN_PREFIX}cat": [0, 1, 0],
            f"{SDEC_SUB_COLUMN_PREFIX}cat": [0, 5, 0],
            "key": [1, 1, 2],
            "product": [3, 2, 9],
            "is_paid": [0, 1, 1],
        }
    )
    assert_frame_equal(_enrich_slen_sidx_sdec(df, context_key="key", max_seq_len=1), expected_df)


def test_pad_horizontally():
    df = pd.DataFrame(
        {
            "key": [1, 2],
            "product": [[3, 2], []],
            "is_paid": [[], []],
        }
    )
    right_padded = pad_horizontally(df.copy(), padding_value=0, right=True)
    left_padded = pad_horizontally(df.copy(), padding_value=0, right=False)
    assert_frame_equal(
        right_padded,
        pd.DataFrame(
            {
                "key": [1, 2],
                "product": [[3, 2], [0]],
                "is_paid": [[0], [0]],
            }
        ),
    )
    assert_frame_equal(
        left_padded,
        pd.DataFrame(
            {
                "key": [1, 2],
                "product": [[3, 2], [0]],
                "is_paid": [[0], [0]],
            }
        ),
    )


class TestEncodeCol:
    def test_empty_values(self):
        values = pd.Series([], name="values")
        stats = {
            "encoding_type": ModelEncodingType.tabular_categorical.value,
            "no_of_rare_categories": 3,
            "codes": {"_RARE_": 0},
            "cardinalities": {"cat": 1},
        }
        df = _encode_col(values=values, column_stats=stats)
        pd.testing.assert_frame_equal(df, pd.DataFrame({"cat": []}, dtype="int8"))

    def test_flat_values(self):
        values = pd.Series([1, 2, 3], name="values")
        stats = {
            "encoding_type": ModelEncodingType.tabular_categorical.value,
            "no_of_rare_categories": 3,
            "codes": {"_RARE_": 0},
            "cardinalities": {"cat": 1},
        }
        df = _encode_col(values=values, column_stats=stats)
        pd.testing.assert_frame_equal(df, pd.DataFrame({"cat": [0, 0, 0]}, dtype="int8"))

    def test_sequential_values(self):
        values = pd.Series(
            [np.array([1, 2]), np.array([]), np.array([3]), np.array([])],
            name="values",
        )
        stats = {
            "encoding_type": ModelEncodingType.tabular_categorical.value,
            "no_of_rare_categories": 3,
            "codes": {"_RARE_": 0},
            "cardinalities": {"cat": 1},
        }
        df = _encode_col(values=values, column_stats=stats)
        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame({"cat": [[0, 0], [], [0], []]}),
        )

    def test_long_sequential_values(self):
        values = pd.Series(
            [np.array([1] * 20), np.array([]), np.array([3] * 3), np.array([4] * 10)],
            name="values",
        )
        stats = {
            "encoding_type": ModelEncodingType.tabular_categorical.value,
            "no_of_rare_categories": 3,
            "codes": {"_RARE_": 0},
            "cardinalities": {"cat": 1},
            "seq_len": {"max": 5},
        }
        df = _encode_col(values=values, column_stats=stats)
        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame({"cat": [[0] * 5, [], [0] * 3, [0] * 5]}),
        )


class TestLanguageEncode:
    @pytest.fixture(scope="class")
    def ctx_stats(self):
        return {
            "columns": {
                "table0::col_obj": {},
                "table1::col_int": {},
                "table1::col_float": {},
                "table1::col_bool": {},
                "table2::col_date": {},
                "table3::col_datetime": {},
            }
        }

    @pytest.fixture(scope="class")
    def tgt_stats(self):
        return {"columns": {"table3::col_str": {}}}

    @pytest.fixture(scope="class")
    def ctx_df(self):
        n_rows = 10
        n_reps = int(n_rows / 2)
        df = pd.DataFrame(
            {
                "table0::actual_primary_key": range(n_rows),
                "table0::col_obj": pd.Series([np.nan, "qwertyuiop"] * n_reps, dtype="object"),
                "table1::col_int": pd.Series([np.nan, -1_234_567_890] * n_reps, dtype="Int64"),
                "table1::col_float": pd.Series([np.nan, -1.23456789] * n_reps, dtype="Float64"),
                "table1::col_bool": pd.Series([np.nan, False] * n_reps, dtype="bool"),
                "table2::col_date": pd.to_datetime([np.nan, "2030-01-01"] * n_reps),
                "table3::col_datetime": pd.to_datetime([np.nan, "2030-01-01 19:59:59.1234"] * n_reps),
                "table3::__primary_key": pd.Series(list(range(n_rows))),
            }
        )
        return df

    @pytest.fixture(scope="class")
    def tgt_df(self):
        n_rows = 10
        n_reps = int(n_rows / 2)
        df = pd.DataFrame(
            {
                "table3::col_str": pd.Series([np.nan, "日本国"] * n_reps, dtype="string"),
                "table3::__primary_key": pd.Series(list(range(n_rows))),
            }
        )
        return df

    def test_format_df(self, ctx_df, tgt_df, ctx_stats, tgt_stats):
        formatted_ctx_df = format_df(ctx_df, is_target=False, stats=ctx_stats)
        formatted_tgt_df = format_df(tgt_df, is_target=True, stats=tgt_stats)

        ctx = formatted_ctx_df.iloc[0]
        tgt = formatted_tgt_df.iloc[0]
        # make sure the string starts with a space to avoid inconsistent tokenization results
        assert ctx.startswith(" ")
        assert tgt.startswith(" ")

        ctx_dict = json.loads(ctx)
        tgt_dict = json.loads(tgt)
        assert formatted_ctx_df.shape[0] == ctx_df.shape[0]
        assert "col_obj" in ctx_dict["table0"].keys()
        assert "actual_primary_key" not in ctx_dict["table0"].keys()
        assert "__primary_key" not in ctx_dict["table3"].keys()
        assert formatted_tgt_df.shape[0] == tgt_df.shape[0]
        assert "col_str" in tgt_dict.keys()
        assert "__primary_key" not in tgt_dict.keys()
