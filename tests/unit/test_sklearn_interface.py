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

from mostlyai.engine.sklearn_interface import LanguageModel


def test_language_model_basic_fit_sample():
    """Test basic fit and sample functionality of LanguageModel."""
    # Create small text DataFrame
    text_data = [
        "Hello world",
        "Good morning",
        "How are you",
        "Nice day today",
        "The weather is great",
        "I love sunny days",
        "Python is fun",
        "Machine learning rocks",
        "Data science is cool",
        "Artificial intelligence",
    ]
    df = pd.DataFrame({"text": text_data})
    
    # Create and fit model with fast settings
    model = LanguageModel(
        model="MOSTLY_AI/LSTMFromScratch-3m",
        max_epochs=1,
        max_training_time=0.5,
        verbose=0,
    )
    
    # Fit the model
    model.fit(df)
    
    # Assert fitted attributes exist
    assert hasattr(model, "_fitted")
    assert model._fitted is True
    assert hasattr(model, "n_features_in_")
    assert model.n_features_in_ == 1
    assert hasattr(model, "feature_names_in_")
    assert list(model.feature_names_in_) == ["text"]
    assert hasattr(model, "workspace_path_")
    assert model.workspace_path_ is not None
    
    # Sample synthetic data
    synthetic = model.sample(n_samples=5, seed=42)
    
    # Assert sample returns DataFrame with correct shape and columns
    assert isinstance(synthetic, pd.DataFrame)
    assert synthetic.shape[0] == 5
    assert "text" in synthetic.columns


def test_language_model_not_fitted_error():
    """Test that calling sample() on unfitted model raises error."""
    # Create unfitted model
    model = LanguageModel(verbose=0)
    
    # Assert that calling sample() raises ValueError
    with pytest.raises(ValueError, match="Model must be fitted"):
        model.sample(n_samples=5)
