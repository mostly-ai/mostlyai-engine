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
Sklearn-compatible estimators that wrap the MOSTLY AI engine for supervised learning.
"""

import logging
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Import from submodules directly to avoid circular imports
from mostlyai.engine.analysis import analyze
from mostlyai.engine.domain import ModelType
from mostlyai.engine.encoding import encode
from mostlyai.engine.generation import generate
from mostlyai.engine.random_state import set_random_state
from mostlyai.engine.splitting import split
from mostlyai.engine.training import train

_LOG = logging.getLogger(__name__)


class MostlyEstimator:
    """
    Base class for sklearn-compatible estimators using MOSTLY AI engine.

    This estimator trains a generative model on tabular data with the target
    column positioned last, and uses it for prediction by generating synthetic
    samples conditioned on the features.

    Args:
        workspace_dir: Directory for storing model artifacts. If None, uses a temporary directory
            that is cleaned up automatically. Default is None.
        max_training_time: Maximum training time in minutes. Default is 10.
        max_epochs: Maximum number of training epochs. Default is 100.
        random_state: Random seed for reproducibility. Default is None.
        verbose: Verbosity level. 0 = silent, 1 = progress messages. Default is 0.
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        max_training_time: float = 10.0,
        max_epochs: float = 100.0,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        # Validate parameters
        if max_training_time is not None and max_training_time <= 0:
            raise ValueError(f"max_training_time must be positive, got {max_training_time}")
        if max_epochs is not None and max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")

        self.workspace_dir = workspace_dir
        self.max_training_time = max_training_time
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.verbose = verbose

        # Internal state
        self._temp_dir = None
        self._workspace_path = None
        self._feature_names_in = None
        self._target_name = None
        self._is_fitted = False

    def _get_workspace_path(self) -> Path:
        """Get or create the workspace path."""
        if self._workspace_path is not None:
            return self._workspace_path

        if self.workspace_dir is not None:
            self._workspace_path = Path(self.workspace_dir)
            self._workspace_path.mkdir(parents=True, exist_ok=True)
        else:
            # Create temporary directory
            self._temp_dir = tempfile.TemporaryDirectory()
            self._workspace_path = Path(self._temp_dir.name)

        return self._workspace_path

    def _cleanup_workspace(self):
        """Clean up temporary workspace if needed."""
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception as e:
                warnings.warn(f"Failed to cleanup temporary workspace: {e}")
            finally:
                self._temp_dir = None
                self._workspace_path = None

    def __del__(self):
        """Cleanup on deletion."""
        self._cleanup_workspace()

    def _validate_input(self, X, y=None):
        """Validate input data and convert to appropriate format."""
        # Convert to pandas if needed
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
            n_features = X.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X = pd.DataFrame(X, columns=feature_names)
        elif not isinstance(X, pd.DataFrame):
            try:
                X = pd.DataFrame(X)
            except Exception as e:
                raise ValueError(f"Cannot convert X to DataFrame: {e}")

        # Check for empty data
        if len(X) == 0:
            raise ValueError("X cannot be empty")

        if y is not None:
            if isinstance(y, (pd.Series, pd.DataFrame)):
                y = y.values if isinstance(y, pd.Series) else y.values.ravel()
            y = np.asarray(y)

            # Check length match
            if len(X) != len(y):
                raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")

            # Check for NaN in target
            if np.any(pd.isna(y)):
                raise ValueError("y contains NaN values. Please remove or impute them.")

        return X, y

    def fit(self, X, y):
        """
        Fit the estimator on training data.

        Args:
            X: Training features.
            y: Target values.

        Returns:
            Fitted estimator.
        """
        # Validate and convert input
        X, y = self._validate_input(X, y)

        # Store feature information
        self._feature_names_in = X.columns.tolist()
        self.n_features_in_ = len(self._feature_names_in)

        # Create target column name (ensure no conflict with feature names)
        self._target_name = "target"
        counter = 0
        while self._target_name in self._feature_names_in:
            self._target_name = f"target_{counter}"
            counter += 1

        # Combine features and target with target at the end
        train_data = X.copy()
        train_data[self._target_name] = y

        # Get workspace path
        workspace = self._get_workspace_path()
        if self.verbose > 0:
            print(f"Using workspace: {workspace}")

        # Set random state if provided
        if self.random_state is not None:
            set_random_state(self.random_state)

        # Execute the engine pipeline
        if self.verbose > 0:
            print("Starting engine pipeline: split -> analyze -> encode -> train")

        # Split data
        if self.verbose > 0:
            print("Step 1/4: Splitting data...")
        split(
            workspace_dir=workspace,
            tgt_data=train_data,
            model_type=ModelType.tabular,
        )

        # Analyze
        if self.verbose > 0:
            print("Step 2/4: Analyzing data...")
        analyze(workspace_dir=workspace)

        # Encode
        if self.verbose > 0:
            print("Step 3/4: Encoding data...")
        encode(workspace_dir=workspace)

        # Train with flexible generation disabled (fixed column order)
        if self.verbose > 0:
            print("Step 4/4: Training model...")
        train(
            workspace_dir=workspace,
            max_training_time=self.max_training_time,
            max_epochs=self.max_epochs,
            enable_flexible_generation=False,
        )

        self._is_fitted = True
        if self.verbose > 0:
            print("✓ Training complete!")

        return self

    def _check_is_fitted(self):
        """Check if the estimator is fitted."""
        if not self._is_fitted:
            raise RuntimeError("This estimator instance is not fitted yet. Call 'fit' first.")

    def _predict_base(self, X, n_samples: int = 1):
        """
        Base prediction method that generates synthetic samples.

        Args:
            X: Features to predict on.
            n_samples: Number of synthetic samples to generate per input row.

        Returns:
            Generated data including the target column.
        """
        self._check_is_fitted()

        # Validate and convert input
        X, _ = self._validate_input(X)

        # Ensure feature names match
        if X.columns.tolist() != self._feature_names_in:
            # Try to reorder if all features are present
            if set(X.columns) == set(self._feature_names_in):
                X = X[self._feature_names_in]
            else:
                raise ValueError(f"Feature mismatch. Expected {self._feature_names_in}, got {X.columns.tolist()}")

        # Create seed data with features only (excluding target)
        if n_samples > 1:
            # Replicate each row n_samples times consecutively
            # Result: [row1, row1, ..., row2, row2, ..., row3, row3, ...]
            seed_data = X.loc[X.index.repeat(n_samples)].reset_index(drop=True)
        else:
            seed_data = X.copy()

        workspace = self._get_workspace_path()

        # Generate synthetic data conditioned on features
        generate(
            workspace_dir=workspace,
            seed_data=seed_data,
        )

        # Load generated data
        synthetic_data = pd.read_parquet(workspace / "SyntheticData")

        return synthetic_data

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep: If True, will return the parameters for this estimator and
                contained subobjects that are estimators.

        Returns:
            Parameter names mapped to their values.
        """
        return {
            "workspace_dir": self.workspace_dir,
            "max_training_time": self.max_training_time,
            "max_epochs": self.max_epochs,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }

    def set_params(self, **params) -> "MostlyEstimator":
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            Estimator instance.
        """
        valid_params = self.get_params(deep=False)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self.__class__.__name__}. "
                    f"Valid parameters are: {sorted(valid_params.keys())}"
                )
            setattr(self, key, value)
        return self


class TabularARGNClassifier(MostlyEstimator):
    """
    Classifier using MOSTLY AI engine for prediction.

    This classifier trains a generative model on tabular data and uses it for
    prediction by generating synthetic samples conditioned on the features.
    For prediction, it generates n_samples per input and returns the mode.
    For probability estimation, it uses the distribution of the n_samples.

    Args:
        workspace_dir: Directory for storing model artifacts. If None, uses a temporary directory
            that is cleaned up automatically. Default is None.
        max_training_time: Maximum training time in minutes. Default is 10.
        random_state: Random seed for reproducibility. Default is None.
        verbose: Verbosity level. 0 = silent, 1 = progress messages. Default is 0.

    Attributes:
        classes_: The classes seen during fit.
        n_features_in_: Number of features seen during fit.
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        max_training_time: float = 10.0,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(
            workspace_dir=workspace_dir,
            max_training_time=max_training_time,
            random_state=random_state,
            verbose=verbose,
        )
        self.classes_ = None

    def fit(self, X, y):
        """
        Fit the classifier on training data.

        Args:
            X: Training features.
            y: Target class labels.

        Returns:
            Fitted classifier.
        """
        # Validate input
        X, y = self._validate_input(X, y)

        # Store unique classes
        self.classes_ = np.unique(y)

        # Call parent fit
        super().fit(X, y)

        return self

    def predict(self, X, n_samples: int = 1):
        """
        Predict class labels for samples in X.

        Generates n_samples per input and returns the mode (most common class).

        Args:
            X: Samples to predict.
            n_samples: Number of samples to generate per input for prediction.
                Higher values give more stable predictions but are slower.

        Returns:
            Predicted class labels.
        """
        self._check_is_fitted()

        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        X, _ = self._validate_input(X)
        n_test_samples = len(X)

        # Generate multiple samples per input
        synthetic_data = self._predict_base(X, n_samples=n_samples)

        # Extract predictions and reshape to (n_test_samples, n_samples)
        predictions = synthetic_data[self._target_name].values
        predictions = predictions.reshape(n_test_samples, n_samples)

        # Calculate mode for each test sample using pandas mode
        # (scipy.stats.mode doesn't work with non-numeric arrays)
        y_pred = np.array(
            [pd.Series(row).mode().iloc[0] if len(pd.Series(row).mode()) > 0 else row[0] for row in predictions]
        )

        return y_pred

    def predict_proba(self, X, n_samples: int = 1):
        """
        Predict class probabilities for samples in X.

        Generates n_samples per input and estimates probabilities
        based on the distribution of generated classes.

        Args:
            X: Samples to predict.
            n_samples: Number of samples to generate per input for probability estimation.
                Higher values give more accurate probability estimates but are slower.

        Returns:
            Class probabilities for each sample.
        """
        self._check_is_fitted()

        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        X, _ = self._validate_input(X)
        n_test_samples = len(X)

        # Generate multiple samples per input
        synthetic_data = self._predict_base(X, n_samples=n_samples)

        # Extract predictions and reshape to (n_test_samples, n_samples)
        predictions = synthetic_data[self._target_name].values

        # Convert to numpy array if it's a pandas extension array
        if hasattr(predictions, "to_numpy"):
            predictions = predictions.to_numpy()

        predictions = predictions.reshape(n_test_samples, n_samples)

        # Calculate probabilities for each class
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_test_samples, n_classes))

        for i, class_label in enumerate(self.classes_):
            # Count occurrences of this class for each test sample
            probabilities[:, i] = (predictions == class_label).mean(axis=1)

        return probabilities


class TabularARGNRegressor(MostlyEstimator):
    """
    Regressor using MOSTLY AI engine for prediction.

    This regressor trains a generative model on tabular data and uses it for
    prediction by generating synthetic samples conditioned on the features.
    For prediction, it generates n_samples per input and returns the mean.

    Args:
        workspace_dir: Directory for storing model artifacts. If None, uses a temporary directory
            that is cleaned up automatically. Default is None.
        max_training_time: Maximum training time in minutes. Default is 10.
        random_state: Random seed for reproducibility. Default is None.
        verbose: Verbosity level. 0 = silent, 1 = progress messages. Default is 0.

    Attributes:
        n_features_in_: Number of features seen during fit.
    """

    def __init__(
        self,
        workspace_dir: str | Path | None = None,
        max_training_time: float = 10.0,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(
            workspace_dir=workspace_dir,
            max_training_time=max_training_time,
            random_state=random_state,
            verbose=verbose,
        )

    def predict(self, X, n_samples: int = 1):
        """
        Predict target values for samples in X.

        Generates n_samples per input and returns the mean estimate.

        Args:
            X: Samples to predict.
            n_samples: Number of samples to generate per input for prediction.
                Higher values give more stable predictions but are slower.

        Returns:
            Predicted target values.
        """
        self._check_is_fitted()

        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")

        X, _ = self._validate_input(X)
        n_test_samples = len(X)

        # Generate multiple samples per input
        synthetic_data = self._predict_base(X, n_samples=n_samples)

        # Extract predictions and reshape to (n_test_samples, n_samples)
        predictions = synthetic_data[self._target_name].values
        predictions = predictions.reshape(n_test_samples, n_samples)

        # Calculate mean for each test sample
        y_pred = predictions.mean(axis=1)

        return y_pred
