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

from mostlyai.engine import TabularARGN
from mostlyai.engine._workspace import Workspace

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
        model_cfg = Workspace(argn.workspace_dir).model_configs.read()
        assert "loss_weights" in model_cfg
        assert model_cfg["loss_weights"].get("target") == 1.0
        assert model_cfg["loss_weights"].get("feature1") == 0.0
        assert model_cfg["loss_weights"].get("feature2") == 0.0

        # Test single target prediction
        test_X = X.head(10)
        predictions = argn.predict(test_X, target="target", n_draws=5, agg_fn="mode")

        # Verify predictions - now returns DataFrame
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == 10
        assert "target" in predictions.columns
        assert all(pred in ["class1", "class2"] for pred in predictions["target"])

        # Multi-target prediction is locked to the fitted y targets; this should raise
        with pytest.raises(ValueError, match="(?i)target.*match.*fitted"):
            argn.predict(test_X, target=["target", "feature2"], n_draws=5, agg_fn="mode")

        # Fit with multi-output y and ensure multi-target prediction works when it matches fitted targets
        X2 = data[["feature1"]]
        y2 = data[["target", "feature2"]]
        argn2 = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace_multi_y"),
        )
        argn2.fit(X=X2, y=y2)
        model_cfg2 = Workspace(argn2.workspace_dir).model_configs.read()
        assert model_cfg2["loss_weights"].get("target") == 1.0
        assert model_cfg2["loss_weights"].get("feature2") == 1.0
        assert model_cfg2["loss_weights"].get("feature1") == 0.0
        test_X2 = X2.head(10)
        multi_predictions = argn2.predict(test_X2, target=["target", "feature2"], n_draws=3, agg_fn="mode")
        assert isinstance(multi_predictions, pd.DataFrame)
        assert len(multi_predictions) == 10
        assert set(multi_predictions.columns) == {"target", "feature2"}

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
        proba = argn.predict_proba(test_X, target="target")

        # Verify probabilities
        assert proba.shape[0] == 10
        assert proba.shape[1] >= 2  # At least 2 classes
        # Verify probabilities sum to 1.0 for each sample
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    @pytest.fixture
    def multi_target_data(self):
        """
        Create comprehensive dataset with multiple categorical and numeric targets.

        This fixture is reused across multiple test scenarios to avoid data duplication.
        Includes both categorical (size, color, material) and numeric (count, age) targets.
        """
        n_samples = 200

        # Create correlated categorical features
        size = np.random.choice(["small", "large"], n_samples)
        color = []
        material = []
        for s in size:
            if s == "small":
                color.append(np.random.choice(["red", "blue"], p=[0.7, 0.3]))
            else:
                color.append(np.random.choice(["red", "blue"], p=[0.3, 0.7]))

        for c in color:
            if c == "red":
                material.append(np.random.choice(["wood", "metal"], p=[0.6, 0.4]))
            else:
                material.append(np.random.choice(["wood", "metal"], p=[0.4, 0.6]))

        # Create numeric targets
        count = np.random.choice([1, 2, 3, 4, 5], n_samples)  # Discrete numeric (few values)
        # Create continuous-like age data that will be binned (many distinct values)
        age = np.random.uniform(18, 65, n_samples)  # Continuous values → will be binned

        return pd.DataFrame(
            {
                "feature1": np.random.randn(n_samples),
                "feature2": np.random.choice(["A", "B"], n_samples),
                "size": size,
                "color": color,
                "material": material,
                "count": count,
                "age": age,
            }
        )

    @pytest.mark.parametrize(
        "target_columns,expected_min_columns,test_n_samples",
        [
            (["size", "color"], 4, 5),  # 2 categorical: 2 sizes × 2 colors = 4
            (["size", "color", "material"], 8, 3),  # 3 categorical: 2×2×2 = 8
            (["count"], 5, 5),  # Single numeric discrete (5 possible values)
            (["age"], 3, 5),  # Single numeric binned (continuous → bins)
            (["size", "count"], 10, 3),  # Mixed: categorical + numeric discrete
            (["size", "count", "age"], 30, 3),  # All 3 types: categorical + discrete + binned
        ],
    )
    def test_predict_proba_multi_target(
        self, multi_target_data, tmp_path_factory, target_columns, expected_min_columns, test_n_samples
    ):
        """Test predict_proba() with various target combinations (categorical, numeric discrete, binned, mixed)."""
        # Train model with explicit encoding types to ensure proper testing
        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
            tgt_encoding_types={
                "feature1": "TABULAR_NUMERIC_AUTO",
                "feature2": "TABULAR_CATEGORICAL",
                "size": "TABULAR_CATEGORICAL",
                "color": "TABULAR_CATEGORICAL",
                "material": "TABULAR_CATEGORICAL",
                "count": "TABULAR_NUMERIC_DISCRETE",  # Force discrete
                "age": "TABULAR_NUMERIC_BINNED",  # Force binned
            },
        )
        argn.fit(multi_target_data)

        # Test multi-target predict_proba
        test_X = multi_target_data[["feature1", "feature2"]].head(test_n_samples)
        proba = argn.predict_proba(test_X, target=target_columns)

        # Verify DataFrame structure (single or multi-target)
        if len(target_columns) == 1:
            # Single target: regular DataFrame
            assert isinstance(proba, pd.DataFrame)
            assert proba.shape[0] == test_n_samples
        else:
            # Multi-target: MultiIndex DataFrame
            assert isinstance(proba.columns, pd.MultiIndex)
            assert proba.columns.names == target_columns
            assert proba.shape[0] == test_n_samples
            assert len(proba.columns.levels) == len(target_columns)

        # Verify we have at least the expected minimum combinations
        # May have more due to _RARE_ or other encoded categories
        assert proba.shape[1] >= expected_min_columns

        # Verify probabilities sum to 1.0 for each sample
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

        # Verify all probabilities are non-negative
        assert (proba >= 0).all().all()

        # Verify expected values are present
        for col_name in target_columns:
            if len(target_columns) == 1:
                # Single target: check column names directly
                col_values = set(proba.columns)
            else:
                # Multi-target: check level values
                level_idx = target_columns.index(col_name)
                col_values = set(proba.columns.get_level_values(level_idx).unique())

            # Verify main categories/values are present
            if col_name == "size":
                assert {"small", "large"}.issubset(col_values)
            elif col_name == "color":
                assert {"red", "blue"}.issubset(col_values)
            elif col_name == "material":
                assert {"wood", "metal"}.issubset(col_values)
            elif col_name == "count":
                # Numeric discrete values (may include bins or exact values)
                assert len(col_values) >= 3  # At least some discrete values present
            elif col_name == "age":
                # Numeric binned values (may be bin labels or ranges)
                assert len(col_values) >= 3  # At least some bins present

    def test_predict_proba_wrong_column_order_raises(self, classification_data, tmp_path_factory):
        """Test predict_proba raises error with different column order when flexible generation is disabled."""
        data = classification_data
        X = data[["feature1", "feature2"]]
        y = data["target"]

        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            enable_flexible_generation=False,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )
        argn.fit(X=X, y=y)

        # Reorder columns in test data
        test_X = X.head(10)[["feature2", "feature1"]]

        with pytest.raises(ValueError, match="(?i)column order.*does not match"):
            argn.predict_proba(test_X, target="target")


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
        model_cfg = Workspace(argn.workspace_dir).model_configs.read()
        assert "loss_weights" in model_cfg
        assert model_cfg["loss_weights"].get("target") == 1.0
        assert model_cfg["loss_weights"].get("feature1") == 0.0
        assert model_cfg["loss_weights"].get("feature2") == 0.0

        # Test single target prediction
        test_X = X.head(10)
        predictions = argn.predict(test_X, target="target", n_draws=5, agg_fn="mean")

        # Verify predictions - now returns DataFrame
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == 10
        assert "target" in predictions.columns
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions["target"])
        assert all(not np.isnan(pred) for pred in predictions["target"])

        # Multi-target prediction is locked to the fitted y targets; this should raise
        with pytest.raises(ValueError, match="(?i)target.*match.*fitted"):
            argn.predict(test_X, target=["target", "feature1"], n_draws=5, agg_fn="mean")

        # Fit with multi-output y and ensure multi-target prediction works when it matches fitted targets
        X2 = data[["feature2"]]
        y2 = data[["target", "feature1"]]
        argn2 = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace_multi_y"),
        )
        argn2.fit(X=X2, y=y2)
        model_cfg2 = Workspace(argn2.workspace_dir).model_configs.read()
        assert model_cfg2["loss_weights"].get("target") == 1.0
        assert model_cfg2["loss_weights"].get("feature1") == 1.0
        assert model_cfg2["loss_weights"].get("feature2") == 0.0
        test_X2 = X2.head(10)
        multi_predictions = argn2.predict(test_X2, target=["target", "feature1"], n_draws=3, agg_fn="mean")
        assert isinstance(multi_predictions, pd.DataFrame)
        assert len(multi_predictions) == 10
        assert set(multi_predictions.columns) == {"target", "feature1"}


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


class TestTabularARGNLogProb:
    """Test log_prob() for computing log likelihood of observations."""

    @pytest.fixture
    def log_prob_data(self):
        """Create data with multiple column types for log_prob testing."""
        n_samples = 100
        return pd.DataFrame(
            {
                "feature": np.random.randn(n_samples),
                "color": np.random.choice(["red", "blue", "green"], n_samples),
                "size": np.random.choice(["small", "large"], n_samples),
            }
        )

    @pytest.fixture
    def fitted_log_prob_model(self, log_prob_data, tmp_path_factory):
        """Create a fitted model for log_prob tests."""
        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )
        argn.fit(X=log_prob_data)
        return argn

    @pytest.fixture
    def fitted_log_prob_model_no_flex(self, log_prob_data, tmp_path_factory):
        """Create a fitted model with flexible generation disabled."""
        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            enable_flexible_generation=False,
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )
        argn.fit(X=log_prob_data)
        return argn

    def test_log_prob(self, fitted_log_prob_model, log_prob_data):
        """Test log_prob computes log probability for observations."""
        test_data = log_prob_data.head(10)
        log_probs = fitted_log_prob_model.log_prob(test_data)

        assert isinstance(log_probs, np.ndarray)
        assert len(log_probs) == 10
        assert (log_probs <= 0).all()
        assert np.isfinite(log_probs).any()

    def test_log_prob_values_differ_by_observation(self, fitted_log_prob_model, log_prob_data):
        """Test that different observations get different log probabilities."""
        test_data = log_prob_data.head(10)
        log_probs = fitted_log_prob_model.log_prob(test_data)

        # Different observations should generally have different log probs
        assert len(np.unique(log_probs)) > 1

    def test_log_prob_wrong_column_order_raises(self, fitted_log_prob_model_no_flex, log_prob_data):
        """Test log_prob raises error with different column order when flexible generation is disabled."""
        test_data = log_prob_data.head(10)[["size", "color", "feature"]]  # Reordered

        with pytest.raises(ValueError, match="(?i)column order.*does not match"):
            fitted_log_prob_model_no_flex.log_prob(test_data)

    def test_log_prob_sequential(self, tmp_path_factory):
        """Test log_prob works with sequential models."""
        tgt_data = pd.DataFrame(
            {
                "ctx_id": ["ctx1", "ctx1", "ctx1", "ctx2", "ctx2", "ctx3", "ctx3", "ctx3", "ctx3"],
                "value": [10, 20, 30, 15, 25, 12, 22, 32, 42],
                "label": ["x", "y", "x", "x", "y", "x", "y", "x", "y"],
            }
        )

        argn = TabularARGN(
            model="MOSTLY_AI/Small",
            max_epochs=1,
            verbose=0,
            tgt_context_key="ctx_id",
            workspace_dir=tmp_path_factory.mktemp("workspace"),
        )
        argn.fit(X=tgt_data)

        log_probs = argn.log_prob(tgt_data)

        assert isinstance(log_probs, np.ndarray)
        assert len(log_probs) == len(tgt_data)
        assert (log_probs <= 0).all()
        assert np.isfinite(log_probs).any()
