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

from mostlyai.engine._encoding_types.language.categorical import decode_categorical


class TestCategoricalDecode:
    @pytest.fixture
    def col_stats(self):
        return {"categories": ["apple", "banana", "cherry"]}

    @pytest.fixture
    def sample_values(self):
        return pd.Series(["apple", "durian", "banana", "elderberry", "cherry", "fig", None])

    def test_decode_categorical(self, sample_values, col_stats):
        decoded = decode_categorical(sample_values, col_stats)
        expected = pd.Series(["apple", None, "banana", None, "cherry", None, None], dtype=decoded.dtype)
        pd.testing.assert_series_equal(decoded, expected)
