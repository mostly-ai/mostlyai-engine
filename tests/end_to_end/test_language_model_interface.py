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

"""
Functional tests for LanguageModel interface.
"""

import pandas as pd
import pytest

from mostlyai.engine._language.interface import LanguageModel
from mostlyai.engine.domain import ModelEncodingType


@pytest.fixture
def simple_language_data():
    """Create minimal language data for testing."""
    data = pd.DataFrame(
        {
            "category": ["business", "tech", "sports", "business", "tech"] * 10,
            "headline": [
                "Company announces new product",
                "Tech innovation changes industry",
                "Team wins championship",
                "Market analysis shows growth",
                "AI breakthrough announced",
            ]
            * 10,
            "date": pd.date_range("2024-01-01", periods=50, freq="D"),
        }
    )
    return data


class TestLanguageModelBasic:
    """Test basic LanguageModel functionality: fit and unconditional sampling."""

    def test_fit_and_unconditional_sample(self, simple_language_data, tmp_path_factory):
        """Test fit() and unconditional sample()."""
        data = simple_language_data

        lm = LanguageModel(
            model="MOSTLY_AI/LSTMFromScratch-3m",
            tgt_encoding_types={
                "category": ModelEncodingType.language_categorical.value,
                "headline": ModelEncodingType.language_text.value,
                "date": ModelEncodingType.language_datetime.value,
            },
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )

        # Fit the model
        lm.fit(X=data)
        assert lm._fitted is True

        # Generate unconditional samples
        syn_data = lm.sample(
            n_samples=10,
            sampling_temperature=0.5,
        )

        # Verify output shape and columns
        assert syn_data.shape[0] == 10
        assert set(syn_data.columns) == set(data.columns)
        assert all(col in syn_data.columns for col in data.columns)
        # Verify text columns are strings
        assert syn_data["headline"].dtype == "string" or str(syn_data["headline"].dtype).startswith("string")


class TestLanguageModelConditional:
    """Test conditional sampling with seed data."""

    @pytest.fixture
    def fitted_model(self, simple_language_data, tmp_path_factory):
        """Create a fitted model for reuse in tests."""
        data = simple_language_data
        lm = LanguageModel(
            model="MOSTLY_AI/LSTMFromScratch-3m",
            tgt_encoding_types={
                "category": ModelEncodingType.language_categorical.value,
                "headline": ModelEncodingType.language_text.value,
                "date": ModelEncodingType.language_datetime.value,
            },
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )
        lm.fit(X=data)
        return lm

    def test_conditional_sample(self, fitted_model, simple_language_data):
        """Test conditional sampling with seed_data."""
        lm = fitted_model

        # Prepare seed data
        seed_data = pd.DataFrame(
            {
                "category": ["business", "tech"],
            }
        )

        # Generate conditional samples
        syn_data = lm.sample(
            seed_data=seed_data,
            sampling_temperature=0.5,
        )

        # Verify seeded columns are preserved
        assert len(syn_data) == 2
        assert all(syn_data["category"] == seed_data["category"])
        # Verify all columns are present
        assert set(syn_data.columns) == set(simple_language_data.columns)
