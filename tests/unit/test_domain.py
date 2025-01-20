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

import pytest
from pydantic import ValidationError
from mostlyai.engine.domain import RebalancingConfig


def test_rebalancing_config_valid():
    config = RebalancingConfig(column="test_column", probabilities={"A": 0.3, "B": 0.5})
    assert config.column == "test_column"
    assert config.probabilities == {"A": 0.3, "B": 0.5}


def test_rebalancing_config_invalid_probabilities_values_out_of_range():
    with pytest.raises(ValidationError):
        RebalancingConfig(column="test_column", probabilities={"A": -0.5, "B": 1.5})


def test_rebalancing_config_invalid_probabilities_values_sum():
    with pytest.raises(ValidationError):
        RebalancingConfig(column="test_column", probabilities={"A": 0.3, "B": 0.8})
