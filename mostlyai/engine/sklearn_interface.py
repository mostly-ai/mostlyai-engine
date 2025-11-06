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
Scikit-learn compatible interface for MOSTLY AI TabularARGN models.

This module provides sklearn-compatible estimators that wrap the MOSTLY AI engine
for classification, regression, imputation, and density estimation tasks.
"""

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score

from mostlyai.engine._workspace import Workspace, resolve_model_type
from mostlyai.engine.analysis import analyze
from mostlyai.engine.domain import ImputationConfig, ModelType
from mostlyai.engine.encoding import encode
from mostlyai.engine.generation import generate
from mostlyai.engine.logging import disable_logging, init_logging
from mostlyai.engine.splitting import split
from mostlyai.engine.training import train

_LOG = logging.getLogger(__name__)


def _ensure_dataframe(X: Any, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert array-like to DataFrame with column names."""
    if isinstance(X, pd.DataFrame):
        return X
    elif isinstance(X, np.ndarray):
        if columns is None:
            columns = [f"col_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=columns)
    elif hasattr(X, "__array__"):
        arr = np.asarray(X)
        if columns is None:
            columns = [f"col_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=columns)
    else:
        raise ValueError(f"Unsupported data type: {type(X)}")


def _load_synthetic_data(workspace_dir: str | Path) -> pd.DataFrame:
    """Load generated parquet files from SyntheticData directory."""
    workspace = Workspace(workspace_dir)
    synthetic_files = workspace.generated_data.fetch_all()
    if not synthetic_files:
        raise ValueError(f"No synthetic data found in {workspace_dir}/SyntheticData")
    dfs = [pd.read_parquet(f) for f in synthetic_files]
    return pd.concat(dfs, ignore_index=True)


def _determine_model_type_from_name(model_name: str | None) -> ModelType:
    """Determine if a model is TABULAR or LANGUAGE based on model name."""
    if model_name is None:
        # Default to tabular if no model specified
        return ModelType.tabular

    # Check if it's a tabular model (MOSTLY_AI/Small, MOSTLY_AI/Medium, MOSTLY_AI/Large)
    if "MOSTLY_AI/" in model_name and any(size in model_name for size in ["Small", "Medium", "Large"]):
        return ModelType.tabular

    # Check if it's the LSTM language model
    if "LSTMFromScratch" in model_name:
        return ModelType.language

    # If it looks like a HuggingFace model path (e.g., "gpt2", "bert-base", etc.)
    # or doesn't match tabular patterns, assume it's a language model
    if "/" in model_name or model_name.startswith("MOSTLY_AI/LSTM"):
        return ModelType.language

    # Default to tabular for unrecognized patterns
    return ModelType.tabular


class TabularARGN(BaseEstimator):
    """
    Base class for MOSTLY AI TabularARGN models with sklearn interface.

    This class wraps the MOSTLY AI engine to provide a scikit-learn compatible
    interface for training generative models on tabular or language data.

    Args:
        model: The identifier of the tabular model to train. Defaults to MOSTLY_AI/Medium.
        max_training_time: Maximum training time in minutes. Defaults to 14400 (10 days).
        max_epochs: Maximum number of training epochs. Defaults to 100.
        batch_size: Per-device batch size for training and validation. If None, determined automatically.
        gradient_accumulation_steps: Number of steps to accumulate gradients. If None, determined automatically.
        enable_flexible_generation: Whether to enable flexible order generation. Defaults to True.
        max_sequence_window: Maximum sequence window for tabular sequential models. Only applicable for tabular models.
        differential_privacy: Configuration for differential privacy training. If None, DP is disabled.
        device: Device to run training on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        workspace_dir: Directory path for workspace. If None, a temporary directory will be created.
        random_state: Random seed for reproducibility.
        verbose: Verbosity level. 0 = silent, 1 = progress messages.
        update_progress: Callback function to report training progress.
    """

    def __init__(
        self,
        model: str | None = None,
        max_training_time: float | None = 14400.0,
        max_epochs: float | None = 100.0,
        batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,
        enable_flexible_generation: bool = True,
        max_sequence_window: int | None = None,
        differential_privacy: dict | None = None,
        device: torch.device | str | None = None,
        workspace_dir: str | Path | None = None,
        random_state: int | None = None,
        verbose: int = 0,
        update_progress: Callable | None = None,
    ):
        self.model = model
        self.max_training_time = max_training_time
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_flexible_generation = enable_flexible_generation
        self.max_sequence_window = max_sequence_window
        self.differential_privacy = differential_privacy
        self.device = device
        self.workspace_dir = workspace_dir
        self.random_state = random_state
        self.verbose = verbose
        self.update_progress = update_progress

        self._fitted = False
        self._temp_dir = None
        self._workspace_path = None
        self._model_type = None
        self._feature_names = None

        # Initialize or disable logging based on verbose setting
        if self.verbose > 0:
            init_logging()
        else:
            disable_logging()

    def _get_workspace_dir(self) -> Path:
        """Get or create workspace directory."""
        # If workspace_path was set from a base TabularARGN, use it
        if self._workspace_path is not None:
            return self._workspace_path

        # If workspace_dir parameter was provided, use it
        if self.workspace_dir is not None:
            self._workspace_path = Path(self.workspace_dir)
            return self._workspace_path

        # Otherwise create a temp directory
        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="mostlyai_sklearn_")
            self._workspace_path = Path(self._temp_dir.name)

        return self._workspace_path

    def _set_random_state(self):
        """Set random state for reproducibility."""
        if self.random_state is not None:
            from mostlyai.engine import set_random_state

            set_random_state(self.random_state)

    def fit(self, X, y=None):
        """
        Fit the TabularARGN model on training data.

        This method wraps the MOSTLY AI engine's split(), analyze(), encode(), and train() pipeline.

        Args:
            X: Training data. Can be array-like or pd.DataFrame of shape (n_samples, n_features).
            y: Target values. If provided, will be included as a column in training data.

        Returns:
            self: Returns self.
        """
        self._set_random_state()

        # Convert to DataFrame
        X_df = _ensure_dataframe(X)
        self._feature_names = list(X_df.columns)

        # Add target column if provided
        if y is not None:
            y_array = np.asarray(y)
            if hasattr(self, "_target_column"):
                X_df[self._target_column] = y_array
            else:
                X_df["target"] = y_array

        # Get workspace directory
        workspace_dir = self._get_workspace_dir()

        if self.verbose > 0:
            _LOG.info(f"Fitting TabularARGN model on data with shape {X_df.shape}")
            _LOG.info(f"Using workspace directory: {workspace_dir}")

        # Determine model type from model name
        self._model_type = _determine_model_type_from_name(self.model)

        # Split data
        split(
            tgt_data=X_df,
            workspace_dir=workspace_dir,
            update_progress=self.update_progress if self.verbose > 0 else None,
        )

        # Analyze data
        analyze(
            workspace_dir=workspace_dir,
            update_progress=self.update_progress if self.verbose > 0 else None,
        )

        # Encode data
        encode(
            workspace_dir=workspace_dir,
            update_progress=self.update_progress if self.verbose > 0 else None,
        )

        # Train model
        train(
            model=self.model,
            max_training_time=self.max_training_time,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            enable_flexible_generation=self.enable_flexible_generation,
            max_sequence_window=self.max_sequence_window,
            differential_privacy=self.differential_privacy,
            device=self.device,
            workspace_dir=workspace_dir,
            update_progress=self.update_progress if self.verbose > 0 else None,
        )

        # Update model type from workspace after training
        self._model_type = resolve_model_type(workspace_dir)
        self._fitted = True

        # Add sklearn-compatible fitted attributes (ending with underscore)
        # These signal to sklearn's HTML repr that the model is fitted
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = np.array(self._feature_names)

        if self.verbose > 0:
            _LOG.info(f"Model training complete. Model type: {self._model_type}")

        return self

    def sample(self, n_samples: int | None = 1, seed: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic samples from the fitted model.

        Args:
            n_samples: Number of samples to generate for unconditional generation. If both n_samples and seed are None, generates same number as training data.
            seed: Seed data to condition generation on fixed columns. If provided, performs conditional generation.
            **kwargs: Additional arguments passed to generate() function.

        Returns:
            Generated synthetic samples as pd.DataFrame.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling. Call fit() first.")

        workspace_dir = self._get_workspace_dir()

        if self.verbose > 0:
            if seed is not None:
                _LOG.info(f"Generating samples with seed data of shape {seed.shape}")
            else:
                _LOG.info(f"Generating {n_samples} unconditional samples")

        # Generate synthetic data
        generate(
            seed_data=seed,
            sample_size=n_samples,
            device=self.device,
            workspace_dir=workspace_dir,
            update_progress=self.update_progress if self.verbose > 0 else None,
            **kwargs,
        )

        # Load and return synthetic data
        synthetic_data = _load_synthetic_data(workspace_dir)

        if self.verbose > 0:
            _LOG.info(f"Generated synthetic data with shape {synthetic_data.shape}")

        return synthetic_data

    def log_prob(self, X) -> np.ndarray:
        """
        Compute the log-likelihood of each sample under the model.

        This method estimates the log-probability (log-likelihood) of data samples
        under the fitted generative model. Higher values indicate samples that are
        more likely under the learned distribution.

        Note: This method only supports TABULAR models, not LANGUAGE models.

        Args:
            X: Data samples to score. Can be array-like or pd.DataFrame of shape (n_samples, n_features).

        Returns:
            Log-likelihood of each sample as np.ndarray of shape (n_samples,).
            More positive values indicate higher likelihood under the model.

        Raises:
            ValueError: If the model is not fitted, or if the model type is LANGUAGE.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before computing log probabilities. Call fit() first.")

        if self._model_type == ModelType.language:
            raise ValueError("log_prob() does not support LANGUAGE models, only TABULAR models")

        from mostlyai.engine.log_prob import log_prob

        X_df = _ensure_dataframe(X, columns=self._feature_names)
        workspace_dir = self._get_workspace_dir()

        return log_prob(
            tgt_data=X_df,
            workspace_dir=workspace_dir,
            device=self.device,
        )

    def __del__(self):
        """Clean up temporary directory if created."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass


class TabularARGNClassifier(TabularARGN):
    """
    TabularARGN classifier with sklearn interface.

    This classifier trains a generative model on the full dataset and uses it
    to predict target classes by conditioning on input features.

    Args:
        X: Training data or a fitted TabularARGN instance.
        target: Name of the target column to predict.
        **kwargs: All other arguments are passed to TabularARGN base class.
            See TabularARGN docstring for available parameters.
    """

    def __init__(
        self,
        X=None,
        target: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Store parameters as attributes for sklearn compatibility
        self.X = X
        self.target = target

        # Internal attributes
        self._base_argn = None
        self._target_column = target
        self._X_init = X

        # If X is a fitted TabularARGN, use it as base
        if isinstance(X, TabularARGN):
            if not X._fitted:
                raise ValueError("Provided TabularARGN instance must be fitted")
            if target is None:
                raise ValueError("target parameter must be specified when using a fitted TabularARGN instance")
            self._base_argn = X
            self._fitted = True
            self._workspace_path = X._workspace_path
            self._model_type = X._model_type
            self._feature_names = X._feature_names
            # Copy sklearn-compatible fitted attributes
            if hasattr(X, "n_features_in_"):
                self.n_features_in_ = X.n_features_in_
            if hasattr(X, "feature_names_in_"):
                self.feature_names_in_ = X.feature_names_in_

    def fit(self, X=None, y=None):
        """
        Fit the classifier.

        If X was provided during initialization and is array-like, trains the model.
        If X was a fitted TabularARGN, this is a no-op.

        Args:
            X: Training data. If None, uses X from initialization.
            y: Target values. If None and target column is in X, uses that.

        Returns:
            self: Returns self.
        """
        if self._base_argn is not None:
            # Already fitted via base TabularARGN
            return self

        # Use X from init if not provided
        if X is None:
            X = self._X_init

        if X is None:
            raise ValueError("X must be provided either during initialization or fit()")

        # Infer target column name if not specified and y is provided
        if self._target_column is None and y is not None:
            if isinstance(y, pd.Series) and y.name is not None:
                # Use Series name
                self._target_column = y.name
            elif isinstance(y, pd.DataFrame) and len(y.columns) == 1:
                # Use single DataFrame column name
                self._target_column = y.columns[0]
            else:
                # Fall back to default name
                self._target_column = "target"

        # Call parent fit which trains on full X (including target)
        return super().fit(X, y=y)

    def predict(
        self,
        X,
        n_draws: int = 10,
        agg_fn: Callable = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Input samples.
            n_draws: Number of draws to generate for each sample. Defaults to 10.
            agg_fn: Aggregation function to combine predictions across draws. Defaults to mode (most common value).
            **kwargs: Additional arguments passed to sample() method.

        Returns:
            Predicted class labels as np.ndarray.
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction. Call fit() first.")

        if self._target_column is None:
            raise ValueError("Target column must be specified for prediction.")

        if agg_fn is None:
            # Default to mode (most common value)
            # Use a custom function that works with categorical data
            def mode_fn(x):
                values, counts = np.unique(x, return_counts=True)
                return values[np.argmax(counts)]

            agg_fn = mode_fn

        X_df = _ensure_dataframe(X, columns=self._feature_names)

        # Generate predictions across multiple draws
        all_predictions = []
        for _ in range(n_draws):
            samples = self.sample(seed=X_df, **kwargs)
            if self._target_column in samples.columns:
                all_predictions.append(samples[self._target_column].values)
            else:
                raise ValueError(f"Target column '{self._target_column}' not found in generated samples")

        # Stack predictions and aggregate
        predictions_array = np.column_stack(all_predictions)
        y_pred = np.apply_along_axis(agg_fn, 1, predictions_array)

        return y_pred

    def predict_proba(
        self,
        X,
        n_draws: int = 10,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Input samples.
            n_draws: Number of draws to generate for each sample. Defaults to 10.
            **kwargs: Additional arguments passed to sample() method.

        Returns:
            Predicted class probabilities as np.ndarray of shape (n_samples, n_classes).
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction. Call fit() first.")

        if self._target_column is None:
            raise ValueError("Target column must be specified for prediction.")

        X_df = _ensure_dataframe(X, columns=self._feature_names)

        # Generate predictions across multiple draws
        all_predictions = []
        for _ in range(n_draws):
            samples = self.sample(seed=X_df, **kwargs)
            if self._target_column in samples.columns:
                all_predictions.append(samples[self._target_column].values)
            else:
                raise ValueError(f"Target column '{self._target_column}' not found in generated samples")

        # Stack predictions
        predictions_array = np.column_stack(all_predictions)

        # Get unique classes and compute probabilities
        classes = np.unique(predictions_array)
        n_samples = predictions_array.shape[0]
        n_classes = len(classes)

        # Compute probability for each class
        proba = np.zeros((n_samples, n_classes))
        for i, cls in enumerate(classes):
            proba[:, i] = np.mean(predictions_array == cls, axis=1)

        self.classes_ = classes
        return proba

    def score(self, X, y, **kwargs) -> float:
        """
        Return the accuracy score on the given test data and labels.

        Args:
            X: Test samples.
            y: True labels for X.
            **kwargs: Additional arguments passed to predict() method.

        Returns:
            Accuracy score as float.
        """
        y_pred = self.predict(X, **kwargs)
        return accuracy_score(y, y_pred)


class TabularARGNRegressor(TabularARGN):
    """
    TabularARGN regressor with sklearn interface.

    This regressor trains a generative model on the full dataset and uses it
    to predict continuous target values by conditioning on input features.

    Args:
        X: Training data or a fitted TabularARGN instance.
        target: Name of the target column to predict.
        **kwargs: All other arguments are passed to TabularARGN base class.
            See TabularARGN docstring for available parameters.
    """

    def __init__(
        self,
        X=None,
        target: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Store parameters as attributes for sklearn compatibility
        self.X = X
        self.target = target

        # Internal attributes
        self._base_argn = None
        self._target_column = target
        self._X_init = X

        # If X is a fitted TabularARGN, use it as base
        if isinstance(X, TabularARGN):
            if not X._fitted:
                raise ValueError("Provided TabularARGN instance must be fitted")
            if target is None:
                raise ValueError("target parameter must be specified when using a fitted TabularARGN instance")
            self._base_argn = X
            self._fitted = True
            self._workspace_path = X._workspace_path
            self._model_type = X._model_type
            self._feature_names = X._feature_names
            # Copy sklearn-compatible fitted attributes
            if hasattr(X, "n_features_in_"):
                self.n_features_in_ = X.n_features_in_
            if hasattr(X, "feature_names_in_"):
                self.feature_names_in_ = X.feature_names_in_

    def fit(self, X=None, y=None):
        """
        Fit the regressor.

        If X was provided during initialization and is array-like, trains the model.
        If X was a fitted TabularARGN, this is a no-op.

        Args:
            X: Training data. If None, uses X from initialization.
            y: Target values. If None and target column is in X, uses that.

        Returns:
            self: Returns self.
        """
        if self._base_argn is not None:
            # Already fitted via base TabularARGN
            return self

        # Use X from init if not provided
        if X is None:
            X = self._X_init

        if X is None:
            raise ValueError("X must be provided either during initialization or fit()")

        # Infer target column name if not specified and y is provided
        if self._target_column is None and y is not None:
            if isinstance(y, pd.Series) and y.name is not None:
                # Use Series name
                self._target_column = y.name
            elif isinstance(y, pd.DataFrame) and len(y.columns) == 1:
                # Use single DataFrame column name
                self._target_column = y.columns[0]
            else:
                # Fall back to default name
                self._target_column = "target"

        # Call parent fit which trains on full X (including target)
        return super().fit(X, y=y)

    def predict(
        self,
        X,
        n_draws: int = 10,
        agg_fn: Callable = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict continuous target values for samples in X.

        Args:
            X: Input samples.
            n_draws: Number of draws to generate for each sample. Defaults to 10.
            agg_fn: Aggregation function to combine predictions across draws. Defaults to np.mean.
            **kwargs: Additional arguments passed to sample() method.

        Returns:
            Predicted target values as np.ndarray.
        """
        if not self._fitted:
            raise ValueError("Regressor must be fitted before prediction. Call fit() first.")

        if self._target_column is None:
            raise ValueError("Target column must be specified for prediction.")

        if agg_fn is None:
            agg_fn = np.mean

        X_df = _ensure_dataframe(X, columns=self._feature_names)

        # Generate predictions across multiple draws
        all_predictions = []
        for _ in range(n_draws):
            samples = self.sample(seed=X_df, **kwargs)
            if self._target_column in samples.columns:
                all_predictions.append(samples[self._target_column].values)
            else:
                raise ValueError(f"Target column '{self._target_column}' not found in generated samples")

        # Stack predictions and aggregate
        predictions_array = np.column_stack(all_predictions)
        y_pred = np.apply_along_axis(agg_fn, 1, predictions_array)

        return y_pred

    def score(self, X, y, **kwargs) -> float:
        """
        Return the R² score on the given test data and labels.

        Args:
            X: Test samples.
            y: True target values for X.
            **kwargs: Additional arguments passed to predict() method.

        Returns:
            R² score as float.
        """
        y_pred = self.predict(X, **kwargs)
        return r2_score(y, y_pred)


class TabularARGNImputer(TabularARGN):
    """
    TabularARGN imputer with sklearn interface.

    This imputer trains a generative model and uses it to impute missing values in the data.

    Args:
        X: Training data or a fitted TabularARGN instance.
        **kwargs: All other arguments are passed to TabularARGN base class.
            See TabularARGN docstring for available parameters.
    """

    def __init__(
        self,
        X=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Store parameters as attributes for sklearn compatibility
        self.X = X

        # Internal attributes
        self._base_argn = None
        self._X_init = X

        # If X is a fitted TabularARGN, use it as base
        if isinstance(X, TabularARGN):
            if not X._fitted:
                raise ValueError("Provided TabularARGN instance must be fitted")
            self._base_argn = X
            self._fitted = True
            self._workspace_path = X._workspace_path
            self._model_type = X._model_type
            self._feature_names = X._feature_names
            # Copy sklearn-compatible fitted attributes
            if hasattr(X, "n_features_in_"):
                self.n_features_in_ = X.n_features_in_
            if hasattr(X, "feature_names_in_"):
                self.feature_names_in_ = X.feature_names_in_

    def fit(self, X=None, y=None):
        """
        Fit the imputer.

        If X was provided during initialization and is array-like, trains the model.
        If X was a fitted TabularARGN, this is a no-op.

        Args:
            X: Training data. If None, uses X from initialization.
            y: Not used, present for API consistency.

        Returns:
            self: Returns self.
        """
        if self._base_argn is not None:
            # Already fitted via base TabularARGN
            return self

        # Use X from init if not provided
        if X is None:
            X = self._X_init

        if X is None:
            raise ValueError("X must be provided either during initialization or fit()")

        # Call parent fit
        return super().fit(X, y=None)

    def transform(self, X, **kwargs) -> pd.DataFrame:
        """
        Impute missing values in X.

        Args:
            X: Data with missing values to impute.
            **kwargs: Additional arguments passed to generate() function.

        Returns:
            Data with imputed values as pd.DataFrame.
        """
        if not self._fitted:
            raise ValueError("Imputer must be fitted before transform. Call fit() first.")

        X_df = _ensure_dataframe(X, columns=self._feature_names)

        # Identify columns with missing values
        columns_with_missing = X_df.columns[X_df.isnull().any()].tolist()

        if not columns_with_missing:
            if self.verbose > 0:
                _LOG.info("No missing values found in data")
            return X_df

        if self.verbose > 0:
            _LOG.info(f"Imputing missing values in columns: {columns_with_missing}")

        # Use imputation config to specify which columns to impute
        imputation_config = ImputationConfig(columns=columns_with_missing)

        # Generate imputed data
        workspace_dir = self._get_workspace_dir()
        generate(
            seed_data=X_df,
            sample_size=len(X_df),
            imputation=imputation_config,
            device=self.device,
            workspace_dir=workspace_dir,
            update_progress=self.update_progress if self.verbose > 0 else None,
            **kwargs,
        )

        # Load imputed data
        X_imputed = _load_synthetic_data(workspace_dir)

        return X_imputed

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the imputer and transform X.

        Args:
            X: Training data with missing values.
            y: Not used, present for API consistency.
            **kwargs: Additional arguments passed to transform() method.

        Returns:
            Data with imputed values as pd.DataFrame.
        """
        self.fit(X, y)
        return self.transform(X, **kwargs)
