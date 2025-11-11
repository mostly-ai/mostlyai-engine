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
Functional tests for TabularARGN interface.
"""

import numpy as np
import pandas as pd
import pytest

from mostlyai.engine._tabular.interface import TabularARGN

from .conftest import MockData


@pytest.fixture
def simple_tabular_data():
    """Create minimal tabular data for testing."""
    mock_data = MockData(n_samples=100)
    mock_data.add_numeric_column(
        name="age",
        quantiles={0.0: 18, 0.5: 35, 1.0: 65},
        dtype="int32",
    )
    mock_data.add_categorical_column(
        name="category",
        probabilities={"A": 0.5, "B": 0.3, "C": 0.2},
    )
    mock_data.add_numeric_column(
        name="score",
        quantiles={0.0: 0, 0.5: 50, 1.0: 100},
        dtype="float32",
    )
    return mock_data.df


class TestTabularARGNBasic:
    """Test basic TabularARGN functionality: fit and unconditional sampling."""

    def test_fit_and_unconditional_sample(self, simple_tabular_data, tmp_path_factory):
        """Test fit() and unconditional sample()."""
        data = simple_tabular_data
        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )

        # Fit the model
        argn.fit(X=data)
        assert argn._fitted is True

        # Generate unconditional samples
        syn_data = argn.sample(n_samples=10)

        # Verify output shape and columns
        assert syn_data.shape[0] == 10
        assert set(syn_data.columns) == set(data.columns)
        assert all(col in syn_data.columns for col in data.columns)


class TestTabularARGNConditional:
    """Test conditional sampling with seed data."""

    @pytest.fixture
    def fitted_model(self, simple_tabular_data, tmp_path_factory):
        """Create a fitted model for reuse in tests."""
        data = simple_tabular_data
        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )
        argn.fit(X=data)
        return argn

    def test_conditional_sample(self, fitted_model, simple_tabular_data):
        """Test conditional sampling with seed_data."""
        argn = fitted_model

        # Prepare seed data with partial columns
        seed_data = pd.DataFrame(
            {
                "age": [25, 50],
                "category": ["A", "B"],
            }
        )

        # Generate conditional samples
        syn_data = argn.sample(seed_data=seed_data)

        # Verify seeded columns are preserved
        assert len(syn_data) == 2
        assert all(syn_data["age"] == seed_data["age"])
        assert all(syn_data["category"] == seed_data["category"])
        # Verify all columns are present
        assert set(syn_data.columns) == set(simple_tabular_data.columns)


class TestTabularARGNImputation:
    """Test imputation functionality."""

    def test_impute(self, simple_tabular_data, tmp_path_factory):
        """Test impute() method."""
        data = simple_tabular_data.copy()

        # Create data with missing values
        data_with_missings = data.head(20).copy()
        data_with_missings.loc[0:9, "age"] = pd.NA
        data_with_missings.loc[0:4, "category"] = pd.NA

        # Fit model
        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )
        argn.fit(X=data)

        # Impute missing values
        imputed_data = argn.impute(data_with_missings)

        # Verify missing values are filled
        assert imputed_data.shape == data_with_missings.shape
        assert imputed_data["age"].isna().sum() == 0
        assert imputed_data["category"].isna().sum() == 0
        assert set(imputed_data.columns) == set(data_with_missings.columns)


class TestTabularARGNClassification:
    """Test classification: predict and predict_proba."""

    @pytest.fixture
    def classification_data(self):
        """Create data with categorical target."""
        mock_data = MockData(n_samples=100)
        mock_data.add_numeric_column(
            name="feature1",
            quantiles={0.0: 0, 0.5: 50, 1.0: 100},
            dtype="float32",
        )
        mock_data.add_categorical_column(
            name="feature2",
            probabilities={"X": 0.5, "Y": 0.5},
        )
        # Create target column correlated with features
        mock_data.df["target"] = mock_data.df["feature2"].map({"X": "class1", "Y": "class2"})
        return mock_data.df

    def test_predict_classification(self, classification_data, tmp_path_factory):
        """Test predict() for classification."""
        data = classification_data
        X = data[["feature1", "feature2"]]
        y = data["target"]

        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )
        argn.fit(X=X, y=y)

        # Predict on test data
        test_X = X.head(10)
        predictions = argn.predict(test_X, target="target", n_draws=5, agg_fn="mode")

        # Verify predictions
        assert len(predictions) == 10
        assert all(pred in ["class1", "class2"] for pred in predictions)

    def test_predict_proba(self, classification_data, tmp_path_factory):
        """Test predict_proba() for classification."""
        data = classification_data
        X = data[["feature1", "feature2"]]
        y = data["target"]

        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )
        argn.fit(X=X, y=y)

        # Predict probabilities on test data
        test_X = X.head(10)
        proba = argn.predict_proba(test_X, target="target", n_draws=5)

        # Verify probabilities
        assert proba.shape[0] == 10
        assert proba.shape[1] >= 2  # At least 2 classes
        # Verify probabilities sum to 1.0 for each sample
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)


class TestTabularARGNRegression:
    """Test regression: predict numeric target."""

    @pytest.fixture
    def regression_data(self):
        """Create data with numeric target."""
        mock_data = MockData(n_samples=100)
        mock_data.add_numeric_column(
            name="feature1",
            quantiles={0.0: 0, 0.5: 50, 1.0: 100},
            dtype="float32",
        )
        mock_data.add_categorical_column(
            name="feature2",
            probabilities={"X": 0.5, "Y": 0.5},
        )
        # Create numeric target correlated with feature1
        mock_data.df["target"] = mock_data.df["feature1"] * 2 + 10
        return mock_data.df

    def test_predict_regression(self, regression_data, tmp_path_factory):
        """Test predict() for regression."""
        data = regression_data
        X = data[["feature1", "feature2"]]
        y = data["target"]

        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )
        argn.fit(X=X, y=y)

        # Predict on test data
        test_X = X.head(10)
        predictions = argn.predict(test_X, target="target", n_draws=5, agg_fn="mean")

        # Verify predictions are numeric
        assert len(predictions) == 10
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)
        assert all(not np.isnan(pred) for pred in predictions)


class TestTabularARGNSequentialWithContext:
    """Test sequential data with context."""

    @pytest.fixture
    def sequential_data_with_context(self):
        """Create sequential data with context."""
        # Context data (flat)
        ctx_data = pd.DataFrame(
            {
                "id": ["ctx1", "ctx2", "ctx3"],
                "ctx_feature": ["A", "B", "A"],
            }
        )

        # Target data (sequential)
        tgt_data = pd.DataFrame(
            {
                "ctx_id": ["ctx1", "ctx1", "ctx2", "ctx2", "ctx2", "ctx3"],
                "value": [10, 20, 15, 25, 30, 12],
                "label": ["x", "y", "x", "x", "y", "x"],
            }
        )

        return tgt_data, ctx_data

    def test_sequential_with_context(self, sequential_data_with_context, tmp_path_factory):
        """Test fit() and sample() with sequential data and context."""
        tgt_data, ctx_data = sequential_data_with_context

        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            tgt_context_key="ctx_id",
            ctx_primary_key="id",
            ctx_data=ctx_data,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )

        # Fit the model
        argn.fit(X=tgt_data)
        assert argn._fitted is True

        # Generate samples with custom context
        custom_ctx = pd.DataFrame(
            {
                "id": ["ctx1", "ctx2"],
                "ctx_feature": ["A", "B"],
            }
        )
        syn_data = argn.sample(ctx_data=custom_ctx)

        # Verify output
        assert "ctx_id" in syn_data.columns
        assert "value" in syn_data.columns
        assert "label" in syn_data.columns


class TestTabularARGNSequentialWithoutContext:
    """Test sequential data without context."""

    @pytest.fixture
    def sequential_data_no_context(self):
        """Create sequential data without context."""
        # Target data (sequential) with context key but no separate context table
        tgt_data = pd.DataFrame(
            {
                "ctx_id": ["ctx1", "ctx1", "ctx2", "ctx2", "ctx2", "ctx3"],
                "value": [10, 20, 15, 25, 30, 12],
                "label": ["x", "y", "x", "x", "y", "x"],
            }
        )
        return tgt_data

    def test_sequential_without_context(self, sequential_data_no_context, tmp_path_factory):
        """Test fit() and sample() with sequential data but no context."""
        tgt_data = sequential_data_no_context

        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            tgt_context_key="ctx_id",
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )

        # Fit the model
        argn.fit(X=tgt_data)
        assert argn._fitted is True

        # Generate samples - for sequential data, n_samples generates that many sequences
        syn_data = argn.sample(n_samples=2)

        # Verify output
        assert "ctx_id" in syn_data.columns
        assert "value" in syn_data.columns
        assert "label" in syn_data.columns
        # For sequential data, we get sequences, so we should have at least some rows
        assert len(syn_data) > 0
        # Verify we have the expected number of unique context IDs (sequences)
        assert syn_data["ctx_id"].nunique() == 2


class TestTabularARGNFlatWithContext:
    """Test flat data with context."""

    @pytest.fixture
    def flat_data_with_context(self):
        """Create flat data with context."""
        # Context data
        ctx_data = pd.DataFrame(
            {
                "id": ["ctx1", "ctx2", "ctx3", "ctx4", "ctx5"],
                "ctx_feature": ["A", "B", "A", "B", "A"],
            }
        )

        # Target data (flat, linked to context)
        tgt_data = pd.DataFrame(
            {
                "ctx_id": ["ctx1", "ctx2", "ctx3", "ctx4", "ctx5"],
                "value": [10, 20, 15, 25, 30],
                "label": ["x", "y", "x", "x", "y"],
            }
        )

        return tgt_data, ctx_data

    def test_flat_with_context(self, flat_data_with_context, tmp_path_factory):
        """Test fit() and sample() with flat data and context."""
        tgt_data, ctx_data = flat_data_with_context

        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            tgt_context_key="ctx_id",
            ctx_primary_key="id",
            ctx_data=ctx_data,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )

        # Fit the model
        argn.fit(X=tgt_data)
        assert argn._fitted is True

        # Generate samples with custom context
        custom_ctx = pd.DataFrame(
            {
                "id": ["ctx1", "ctx2"],
                "ctx_feature": ["A", "B"],
            }
        )
        syn_data = argn.sample(ctx_data=custom_ctx)

        # Verify output
        assert len(syn_data) == 2
        assert "ctx_id" in syn_data.columns
        assert "value" in syn_data.columns
        assert "label" in syn_data.columns
