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

from mostlyai.engine._common import ANALYZE_N_MIN_MAX
from mostlyai.engine._encoding_types.language.numeric import (
    analyze_language_numeric,
    analyze_reduce_language_numeric,
    decode_language_numeric,
    encode_language_numeric,
)
from mostlyai.engine.domain import ModelEncodingType


class TestLanguageNumericAnalyze:
    def test_analyze_language_numeric(self):
        values = pd.Series([0, 1, 2, 3, 4, 5] * ANALYZE_N_MIN_MAX, name="value")
        ids = pd.Series(range(len(values)), name="id")
        stats = analyze_language_numeric(values, ids)
        assert stats["has_nan"] is False
        assert stats["max_n"] == [5] * ANALYZE_N_MIN_MAX
        assert stats["min_n"] == [0] * ANALYZE_N_MIN_MAX


class TestLanguageNumericAnalyzeReduce:
    def test_analyze_reduce_language_numeric(self):
        stats1 = {
            "has_nan": False,
            "max_n": [5] * ANALYZE_N_MIN_MAX,
            "min_n": [0] * ANALYZE_N_MIN_MAX,
            "max_scale": 0,
        }
        stats2 = {
            "has_nan": True,
            "max_n": [10] * ANALYZE_N_MIN_MAX,
            "min_n": [6] * ANALYZE_N_MIN_MAX,
            "max_scale": 1,
        }
        reduced = analyze_reduce_language_numeric([stats1, stats2])
        assert reduced["has_nan"] is True
        assert reduced["max"] == 10
        assert reduced["min"] == 0
        assert reduced["max_scale"] == 1


class TestLanguageNumericEncode:
    def test_encode_language_numeric(self):
        values = pd.Series([-1, 0, 1, 2, 3, 4, 5, 6], name="value")
        stats = {
            "has_nan": False,
            "max": 5,
            "min": 0,
            "max_scale": 0,
        }
        encoded = encode_language_numeric(values, stats)
        assert encoded.dtype == "Int64"
        assert encoded.isna().sum() == 0
        assert encoded.iloc[0] == 0
        assert encoded.iloc[1] == 0
        assert encoded.iloc[2] == 1
        assert encoded.iloc[3] == 2
        assert encoded.iloc[4] == 3
        assert encoded.iloc[5] == 4
        assert encoded.iloc[6] == 5
        assert encoded.iloc[7] == 5


class TestLanguageNumericDecode:
    @pytest.fixture
    def int_stats(self):
        return {
            "encoding_type": ModelEncodingType.language_numeric,
            "has_nan": False,
            "max": 91,
            "max_scale": 0,
            "min": 17,
        }

    @pytest.fixture
    def float_stats(self):
        return {
            "encoding_type": ModelEncodingType.language_numeric,
            "has_nan": False,
            "max": 91.12,
            "max_scale": 2,
            "min": 17.0,
        }

    @pytest.fixture
    def sample_values(self):
        return pd.Series(["25.3541", "99.99", "-312.0", "61", None, "35.10091", "-1.223"])

    @pytest.mark.parametrize(
        "stats_name, expected_dtype",
        [
            ("int_stats", "Int64"),
            ("float_stats", float),
        ],
    )
    def test_decode_language_numeric(self, sample_values, request, stats_name, expected_dtype):
        stats = request.getfixturevalue(stats_name)
        decoded = decode_language_numeric(sample_values, stats)
        assert decoded.dtype == expected_dtype
        non_null = decoded.dropna()  # we don't enforce compatability with "has_nan"
        round_digits = stats["max_scale"]
        for v in non_null:
            assert np.isclose(v, round(v, round_digits), atol=1e-8)
        assert all(non_null <= stats["max"])
        assert all(non_null >= stats["min"])
