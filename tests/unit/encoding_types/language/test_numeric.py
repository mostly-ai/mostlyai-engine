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

from mostlyai.engine._encoding_types.language.numeric import decode_numeric
from mostlyai.engine.domain import ModelEncodingType


class TestNumericDecode:
    @pytest.fixture
    def int_stats(self):
        return {
            "encoding_type": ModelEncodingType.language_numeric,
            "has_nan": False,
            "max5": [91] * 5,
            "max_scale": 0,
            "min5": [17] * 5,
        }

    @pytest.fixture
    def float_stats(self):
        return {
            "encoding_type": ModelEncodingType.language_numeric,
            "has_nan": False,
            "max5": [91.12] * 5,
            "max_scale": 2,
            "min5": [17.0] * 5,
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
    def test_decode_numeric(self, sample_values, request, stats_name, expected_dtype):
        stats = request.getfixturevalue(stats_name)
        decoded = decode_numeric(sample_values, stats)
        assert decoded.dtype == expected_dtype
        non_null = decoded.dropna()  # we don't enforce compatability with "has_nan"
        max_val = stats["max5"][0]
        min_val = stats["min5"][0]
        round_digits = stats["max_scale"]
        for v in non_null:
            assert np.isclose(v, round(v, round_digits), atol=1e-8)
        assert all(non_null <= max_val)
        assert all(non_null >= min_val)
