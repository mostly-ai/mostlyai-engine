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

from mostlyai.engine._encoding_types.language.datetime import decode_datetime
from mostlyai.engine.domain import ModelEncodingType


class TestDatetimeDecode:
    @pytest.fixture
    def datetime_stats(self):
        return {
            "encoding_type": ModelEncodingType.language_datetime,
            "has_nan": True,
            "min5": ["2000-01-01"] * 5,
            "max5": ["2024-12-31"] * 5,
        }

    @pytest.fixture
    def no_clip_stats(self):
        return {
            "encoding_type": ModelEncodingType.language_datetime,
            "has_nan": True,
            "min5": ["1900-01-01"] * 5,
            "max5": ["2100-01-01"] * 5,
        }

    @pytest.fixture
    def sample_dates(self):
        return pd.Series(
            [
                "2021-05-20 14:30:00",  # valid datetime with time
                "2020-02-30",  # Feb 30 is invalid; should be clamped to Feb 29, 2020
                "1999-12-31",  # below the min bound -> will be clipped upward
                "2025-01-01",  # above the max bound -> will be clipped downward
                "abcd",  # invalid date string -> becomes NaT
                "",  # empty string -> becomes NaT
                "_INVALID_",  # marked as invalid -> becomes NaT
                "2010-10-10",  # valid date without explicit time (defaults to 00:00:00)
            ]
        )

    def test_datetime_dtype_bounds_and_invalids(self, sample_dates, datetime_stats):
        decoded = decode_datetime(sample_dates, datetime_stats)
        assert decoded.dtype == "datetime64[ns]"
        non_null = decoded.dropna()
        min_bound = pd.to_datetime(datetime_stats["min5"][0])
        max_bound = pd.to_datetime(datetime_stats["max5"][0])
        for dt in non_null:
            assert dt >= min_bound
            assert dt <= max_bound
        assert all(pd.isna(decoded.iloc[4:7]))

    def test_date_day_clamping(self, no_clip_stats):
        s = pd.Series(["2021-04-31"])
        decoded = decode_datetime(s, no_clip_stats)
        expected = pd.Timestamp("2021-04-30 00:00:00")
        assert decoded.iloc[0] == expected

    def test_time_extraction(self, no_clip_stats):
        s = pd.Series(["2021-07-15T23:59:59.123"])
        decoded = decode_datetime(s, no_clip_stats)
        expected = pd.Timestamp("2021-07-15 23:59:59.123")
        assert decoded.iloc[0] == expected
