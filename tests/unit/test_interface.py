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
Unit tests for sklearn interface with sequential data support.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mostlyai.engine import (
    LanguageModel,
    TabularARGN,
    TabularARGNClassifier,
    TabularARGNImputer,
    TabularARGNRegressor,
)


@pytest.fixture
def flat_data():
    """Create simple flat data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feature1": np.random.uniform(0, 100, 30),
            "feature2": np.random.choice(["A", "B", "C"], 30),
            "feature3": np.random.randint(0, 10, 30),
        }
    )


@pytest.fixture
def sequential_data():
    """Create simple sequential data for testing."""
    np.random.seed(42)
    n_customers = 8
    n_transactions = 30
    customer_ids = np.random.choice(range(1, n_customers + 1), size=n_transactions)

    df = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "amount": np.random.uniform(10, 500, n_transactions),
            "category": np.random.choice(["A", "B", "C"], n_transactions),
        }
    )
    return df.sort_values("customer_id").reset_index(drop=True)


@pytest.fixture
def context_and_target_data():
    """Create context and target data for two-table sequential testing."""
    np.random.seed(42)
    n_customers = 8

    ctx_data = pd.DataFrame(
        {
            "customer_id": range(1, n_customers + 1),
            "age": np.random.randint(18, 70, n_customers),
            "country": np.random.choice(["USA", "UK"], n_customers),
        }
    )

    n_transactions = 30
    customer_ids = np.random.choice(ctx_data["customer_id"].values, size=n_transactions)

    tgt_data = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "amount": np.random.uniform(10, 500, n_transactions),
            "category": np.random.choice(["A", "B", "C"], n_transactions),
        }
    )
    tgt_data = tgt_data.sort_values("customer_id").reset_index(drop=True)

    return ctx_data, tgt_data


def test_tabular_argn_flat_data(flat_data):
    """Test TabularARGN with flat data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TabularARGN(max_epochs=1, workspace_dir=tmp_dir, verbose=0)

        # Fit model
        model.fit(flat_data)

        # Check fitted attributes
        assert model._fitted
        assert model.n_features_in_ == 3
        assert len(model.feature_names_in_) == 3

        # Generate samples
        synthetic_data = model.sample(n_samples=10)

        # Check output
        assert isinstance(synthetic_data, pd.DataFrame)
        assert len(synthetic_data) == 10
        assert set(synthetic_data.columns) == set(flat_data.columns)


def test_tabular_argn_sequential_data(sequential_data):
    """Test TabularARGN with sequential data using tgt_context_key."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TabularARGN(tgt_context_key="customer_id", max_epochs=1, workspace_dir=tmp_dir, verbose=0)

        # Fit model
        model.fit(sequential_data)

        # Check fitted attributes
        assert model._fitted
        assert model.n_features_in_ == 3

        # Generate samples
        n_customers = sequential_data["customer_id"].nunique()
        synthetic_data = model.sample(n_samples=n_customers)

        # Check output
        assert isinstance(synthetic_data, pd.DataFrame)
        assert "customer_id" in synthetic_data.columns
        assert synthetic_data["customer_id"].nunique() <= n_customers


def test_tabular_argn_two_table_sequential(context_and_target_data):
    """Test TabularARGN with context and target data."""
    ctx_data, tgt_data = context_and_target_data

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TabularARGN(
            ctx_data=ctx_data,
            ctx_primary_key="customer_id",
            tgt_context_key="customer_id",
            max_epochs=1,
            workspace_dir=tmp_dir,
            verbose=0,
        )

        # Fit model
        model.fit(tgt_data)

        # Check fitted attributes
        assert model._fitted

        # Generate samples with same context
        synthetic_data = model.sample(n_samples=len(ctx_data), ctx_data=ctx_data)

        # Check output
        assert isinstance(synthetic_data, pd.DataFrame)
        assert "customer_id" in synthetic_data.columns

        # Verify all synthetic customer_ids exist in context
        synthetic_ids = set(synthetic_data["customer_id"].unique())
        context_ids = set(ctx_data["customer_id"].unique())
        assert synthetic_ids.issubset(context_ids)


def test_tabular_argn_parameters_stored():
    """Test that sequential parameters are properly stored in the model."""
    model = TabularARGN(
        tgt_context_key="test_key",
        tgt_primary_key="id",
        ctx_primary_key="ctx_id",
        max_epochs=5,
    )

    # Check parameters are stored
    params = model.get_params()
    assert params["tgt_context_key"] == "test_key"
    assert params["tgt_primary_key"] == "id"
    assert params["ctx_primary_key"] == "ctx_id"
    assert params["max_epochs"] == 5


def test_tabular_argn_not_fitted_error():
    """Test that appropriate error is raised when using unfitted model."""
    # Create a model without fitting
    model = TabularARGN()

    with pytest.raises(ValueError, match="must be fitted"):
        model.sample(n_samples=10)


def test_tabular_argn_workspace_persistence():
    """Test that workspace directory is properly managed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir) / "test_workspace"

        model = TabularARGN(workspace_dir=workspace_dir, max_epochs=1, verbose=0)

        # Create minimal data
        data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": ["a", "b", "c", "d", "e"]})

        model.fit(data)

        # Check workspace exists
        assert workspace_dir.exists()
        assert model.workspace_path_ == str(workspace_dir)


def test_tabular_argn_log_prob(flat_data):
    """Test log_prob method."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TabularARGN(max_epochs=1, workspace_dir=tmp_dir, verbose=0)

        model.fit(flat_data)

        # Compute log probabilities
        test_data = flat_data.head(10)
        log_probs = model.log_prob(test_data)

        # Check output
        assert isinstance(log_probs, np.ndarray)
        assert len(log_probs) == 10
        assert np.issubdtype(log_probs.dtype, np.floating)


def test_tabular_argn_classifier():
    """Test TabularARGNClassifier predict and score."""
    np.random.seed(42)

    # Create classification data
    data = pd.DataFrame(
        {
            "feature1": np.random.uniform(0, 100, 30),
            "feature2": np.random.choice(["A", "B"], 30),
            "target": np.random.choice(["Class1", "Class2"], 30),
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TabularARGNClassifier(target="target", n_draws=2, max_epochs=1, workspace_dir=tmp_dir, verbose=0)

        model.fit(data)

        # Test predict
        X_test = data[["feature1", "feature2"]].head(5)
        y_pred = model.predict(X_test)

        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == 5

        # Test predict_proba
        y_proba = model.predict_proba(X_test)

        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape[0] == 5
        assert y_proba.shape[1] >= 1  # At least one class
        assert np.allclose(y_proba.sum(axis=1), 1.0)  # Probabilities sum to 1

        # Test score
        y_true = data["target"].head(5).values
        score = model.score(X_test, y_true)

        assert isinstance(score, float)
        assert 0 <= score <= 1


def test_tabular_argn_regressor():
    """Test TabularARGNRegressor predict and score."""
    np.random.seed(42)

    # Create regression data
    data = pd.DataFrame(
        {
            "feature1": np.random.uniform(0, 100, 30),
            "feature2": np.random.choice(["A", "B"], 30),
            "target": np.random.uniform(0, 50, 30),
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TabularARGNRegressor(target="target", n_draws=2, max_epochs=1, workspace_dir=tmp_dir, verbose=0)

        model.fit(data)

        # Test predict
        X_test = data[["feature1", "feature2"]].head(5)
        y_pred = model.predict(X_test)

        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == 5

        # Test score
        y_true = data["target"].head(5).values
        score = model.score(X_test, y_true)

        assert isinstance(score, float)


def test_tabular_argn_imputer():
    """Test TabularARGNImputer fit and transform."""
    np.random.seed(42)

    # Create data
    data = pd.DataFrame(
        {
            "feature1": np.random.uniform(0, 100, 20),
            "feature2": np.random.choice(["A", "B", "C"], 20),
        }
    )

    # Create test data
    test_data = data.head(8).copy()

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TabularARGNImputer(max_epochs=1, workspace_dir=tmp_dir, verbose=0)

        # Test fit and transform
        model.fit(data)
        imputed_data = model.transform(test_data)

        assert isinstance(imputed_data, pd.DataFrame)
        assert imputed_data.shape == test_data.shape
        assert set(imputed_data.columns) == set(test_data.columns)


def test_language_model_basic():
    """Test LanguageModel with basic fit and sample."""
    np.random.seed(42)

    # Create simple text data
    data = pd.DataFrame(
        {
            "category": np.random.choice(["A", "B", "C"], 20),
            "text": [f"Sample text {i}" for i in range(20)],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = LanguageModel(
            tgt_encoding_types={
                "category": "LANGUAGE_CATEGORICAL",
                "text": "LANGUAGE_TEXT",
            },
            max_epochs=1,
            workspace_dir=tmp_dir,
            verbose=0,
        )

        # Fit model
        model.fit(data)

        # Check fitted attributes
        assert model._fitted
        assert model.n_features_in_ == 2
        assert len(model.feature_names_in_) == 2

        # Generate samples
        synthetic_data = model.sample(n_samples=10)

        # Check output
        assert isinstance(synthetic_data, pd.DataFrame)
        assert len(synthetic_data) == 10
        assert "category" in synthetic_data.columns
        assert "text" in synthetic_data.columns


def test_language_model_not_fitted_error():
    """Test that appropriate error is raised when using unfitted model."""
    # Create a model without fitting
    model = LanguageModel()

    with pytest.raises(ValueError, match="must be fitted"):
        model.sample(n_samples=10)


def test_language_model_workspace_persistence():
    """Test that workspace directory is properly managed for LanguageModel."""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "text": [f"Sample {i}" for i in range(10)],
        }
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir) / "test_language_workspace"

        model = LanguageModel(
            tgt_encoding_types={"text": "LANGUAGE_TEXT"},
            max_epochs=1,
            workspace_dir=workspace_dir,
            verbose=0,
        )

        model.fit(data)

        # Check workspace exists
        assert workspace_dir.exists()
        assert model.workspace_path_ == str(workspace_dir)
