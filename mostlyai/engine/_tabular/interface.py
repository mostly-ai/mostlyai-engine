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
Scikit-learn compatible interface for MOSTLY AI tabular models.

This module provides sklearn-compatible estimators that wrap the MOSTLY AI engine
for sampling, classification, regression, imputation, and density estimation tasks.
"""

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score

from mostlyai.engine._common import ensure_dataframe, load_generated_data
from mostlyai.engine.analysis import analyze
from mostlyai.engine.domain import (
    DifferentialPrivacyConfig,
    FairnessConfig,
    ImputationConfig,
    ModelType,
    RareCategoryReplacementMethod,
    RebalancingConfig,
)
from mostlyai.engine.encoding import encode
from mostlyai.engine.generation import generate
from mostlyai.engine.logging import disable_logging, init_logging
from mostlyai.engine.splitting import split
from mostlyai.engine.training import train

_LOG = logging.getLogger(__name__)


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
        max_sequence_window: Maximum sequence window for tabular sequential models.
        value_protection: Whether to enable value protection for rare values. Defaults to True.
        differential_privacy: Configuration for differential privacy training. If None, DP is disabled.
        tgt_context_key: Context key column name in the target data for sequential models.
        tgt_primary_key: Primary key column name in the target data.
        tgt_encoding_types: Dictionary mapping target column names to encoding types. If None, types are inferred.
        ctx_data: DataFrame containing the context data for two-table sequential models.
        ctx_primary_key: Primary key column name in the context data.
        ctx_encoding_types: Dictionary mapping context column names to encoding types. If None, types are inferred.
        device: Device to run training on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        workspace_dir: Directory path for workspace. If None, a temporary directory will be created.
        random_state: Random seed for reproducibility.
        verbose: Verbosity level. 0 = silent, 1 = progress messages.
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
        value_protection: bool = True,
        differential_privacy: DifferentialPrivacyConfig | dict | None = None,
        tgt_context_key: str | None = None,
        tgt_primary_key: str | None = None,
        tgt_encoding_types: dict[str, str] | None = None,
        ctx_data: pd.DataFrame | None = None,
        ctx_primary_key: str | None = None,
        ctx_encoding_types: dict[str, str] | None = None,
        device: torch.device | str | None = None,
        workspace_dir: str | Path | None = None,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.model = model
        self.max_training_time = max_training_time
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_flexible_generation = enable_flexible_generation
        self.max_sequence_window = max_sequence_window
        self.value_protection = value_protection
        self.differential_privacy = differential_privacy
        self.tgt_context_key = tgt_context_key
        self.tgt_primary_key = tgt_primary_key
        self.tgt_encoding_types = tgt_encoding_types
        self.ctx_data = ctx_data
        self.ctx_primary_key = ctx_primary_key
        self.ctx_encoding_types = ctx_encoding_types
        self.device = device
        self.workspace_dir = workspace_dir
        self.random_state = random_state
        self.verbose = verbose

        self._fitted = False
        self._temp_dir = None
        self._workspace_path = None
        self._feature_names = None
        self._target_column = None

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
            self._temp_dir = tempfile.TemporaryDirectory(prefix="mostlyai_")
            self._workspace_path = Path(self._temp_dir.name)
            # Update the parameter so it shows in get_params()
            self.workspace_dir = str(self._workspace_path)

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
        X_df = ensure_dataframe(X)
        self._feature_names = list(X_df.columns)

        # Add target column if provided
        if y is not None:
            y_array = np.asarray(y)
            # Infer target column name if not already set
            if hasattr(self, "_target_column"):
                if self._target_column is None:
                    # Infer from y
                    if isinstance(y, pd.Series) and y.name is not None:
                        # Use Series name
                        self._target_column = y.name
                    elif isinstance(y, pd.DataFrame) and len(y.columns) == 1:
                        # Use single DataFrame column name
                        self._target_column = y.columns[0]
                    else:
                        # Fall back to default name
                        self._target_column = "target"
                X_df[self._target_column] = y_array
            else:
                X_df["target"] = y_array

        # Get workspace directory
        workspace_dir = self._get_workspace_dir()

        # Convert ctx_data to DataFrame if provided
        ctx_data_df = None
        if self.ctx_data is not None:
            ctx_data_df = ensure_dataframe(self.ctx_data)

        # Split data
        split(
            tgt_data=X_df,
            ctx_data=ctx_data_df,
            tgt_primary_key=self.tgt_primary_key,
            ctx_primary_key=self.ctx_primary_key,
            tgt_context_key=self.tgt_context_key,
            tgt_encoding_types=self.tgt_encoding_types,
            ctx_encoding_types=self.ctx_encoding_types,
            model_type=ModelType.tabular,
            workspace_dir=workspace_dir,
        )

        # Analyze data
        analyze(
            value_protection=self.value_protection,
            differential_privacy=self.differential_privacy,
            workspace_dir=workspace_dir,
        )

        # Encode data
        encode(
            workspace_dir=workspace_dir,
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
        )

        self._fitted = True

        # Add sklearn-compatible fitted attributes (ending with underscore)
        # These signal to sklearn's HTML repr that the model is fitted
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = np.array(self._feature_names)
        self.workspace_path_ = str(workspace_dir)

        return self

    def __del__(self):
        """Clean up temporary directory if created."""
        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass


class TabularARGNSampler(BaseEstimator):
    """
    Sampler for generating synthetic data from a fitted TabularARGN model.

    This class provides a scikit-learn compatible interface for synthetic data generation
    with sampling parameters configured at initialization.

    Args:
        X: A fitted TabularARGN instance or training data to fit a new model.
        batch_size: Batch size for generation. If None, determined automatically.
        sampling_temperature: Sampling temperature. Higher values increase randomness. Defaults to 1.0.
        sampling_top_p: Nucleus sampling probability threshold. Defaults to 1.0.
        device: Device to run generation on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        rare_category_replacement_method: Method for handling rare categories.
        rebalancing: Configuration for rebalancing column distributions.
        imputation: Configuration for imputing missing values.
        fairness: Configuration for fairness constraints.
        **kwargs: Additional arguments passed to TabularARGN if X is not a fitted model.
    """

    def __init__(
        self,
        X=None,
        batch_size: int | None = None,
        sampling_temperature: float = 1.0,
        sampling_top_p: float = 1.0,
        device: str | None = None,
        rare_category_replacement_method: RareCategoryReplacementMethod | str = RareCategoryReplacementMethod.constant,
        rebalancing: RebalancingConfig | dict | None = None,
        imputation: ImputationConfig | dict | None = None,
        fairness: FairnessConfig | dict | None = None,
        **kwargs,
    ):
        # Store all parameters for sklearn compatibility
        self.X = X
        self.batch_size = batch_size
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p
        self.device = device
        self.rare_category_replacement_method = rare_category_replacement_method
        self.rebalancing = rebalancing
        self.imputation = imputation
        self.fairness = fairness

        # Internal attributes
        self._base_argn = None
        self._fitted = False
        self._X_init = X
        self._kwargs = kwargs

        # If X is a fitted TabularARGN, use it as base
        if isinstance(X, TabularARGN):
            if not X._fitted:
                raise ValueError("Provided TabularARGN instance must be fitted")
            self._base_argn = X
            self._fitted = True
            # Copy sklearn-compatible fitted attributes
            if hasattr(X, "n_features_in_"):
                self.n_features_in_ = X.n_features_in_
            if hasattr(X, "feature_names_in_"):
                self.feature_names_in_ = X.feature_names_in_

    def fit(self, X=None, y=None):
        """
        Fit the sampler.

        If X was provided during initialization and is a fitted TabularARGN, this is a no-op.
        Otherwise, trains a new TabularARGN model.

        Args:
            X: Training data. If None, uses X from initialization.
            y: Target values (optional).

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

        # Create and fit a new TabularARGN
        self._base_argn = TabularARGN(**self._kwargs)
        self._base_argn.fit(X, y=y)
        self._fitted = True

        # Copy sklearn-compatible fitted attributes
        if hasattr(self._base_argn, "n_features_in_"):
            self.n_features_in_ = self._base_argn.n_features_in_
        if hasattr(self._base_argn, "feature_names_in_"):
            self.feature_names_in_ = self._base_argn.feature_names_in_

        return self

    def sample(
        self,
        n_samples: int | None = None,
        seed_data: pd.DataFrame | None = None,
        ctx_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic samples from the fitted model.

        Args:
            n_samples: Number of samples to generate. If None and ctx_data is provided, infers from ctx_data length.
                      If None and no ctx_data, defaults to 1.
            seed_data: Seed data to condition generation on fixed columns.
            ctx_data: Context data for generation. If None, uses the context data from training.

        Returns:
            Generated synthetic samples as pd.DataFrame.
        """
        if not self._fitted:
            raise ValueError("Sampler must be fitted before sampling. Call fit() first.")

        workspace_dir = self._base_argn._get_workspace_dir()

        # Determine if ctx_data was explicitly provided (vs using training default)
        ctx_data_explicit = ctx_data is not None

        # Use ctx_data from training if not provided
        if ctx_data is None:
            ctx_data = self._base_argn.ctx_data

        # Convert ctx_data to DataFrame if provided
        ctx_data_df = None
        if ctx_data is not None:
            ctx_data_df = ensure_dataframe(ctx_data)

            # Infer n_samples from ctx_data if it was explicitly provided and n_samples not specified
            if ctx_data_explicit and n_samples is None:
                n_samples = len(ctx_data_df)

            # For sequential models: if ctx_data was not explicitly provided and n_samples is specified,
            # take a random sample of the training ctx_data
            if not ctx_data_explicit and n_samples is not None and self._base_argn.tgt_context_key is not None:
                if len(ctx_data_df) > n_samples:
                    ctx_data_df = ctx_data_df.sample(n=n_samples, random_state=self._base_argn.random_state)

        # Default n_samples to 1 if still None
        if n_samples is None:
            n_samples = 1

        # Generate synthetic data using configured parameters
        generate(
            ctx_data=ctx_data_df,
            seed_data=seed_data,
            sample_size=n_samples,
            batch_size=self.batch_size,
            sampling_temperature=self.sampling_temperature,
            sampling_top_p=self.sampling_top_p,
            device=self.device,
            rare_category_replacement_method=self.rare_category_replacement_method,
            rebalancing=self.rebalancing,
            imputation=self.imputation,
            fairness=self.fairness,
            workspace_dir=workspace_dir,
        )

        # Load and return synthetic data
        synthetic_data = load_generated_data(workspace_dir)

        return synthetic_data


class TabularARGNDensity(BaseEstimator):
    """
    Density estimator for computing log-likelihoods using a fitted TabularARGN model.

    This class provides a scikit-learn compatible interface for density estimation
    following sklearn conventions (uses .score_samples() method).

    Args:
        X: A fitted TabularARGN instance or training data to fit a new model.
        batch_size: Batch size for computation. If None, determined automatically.
        device: Device to run computation on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        **kwargs: Additional arguments passed to TabularARGN if X is not a fitted model.
    """

    def __init__(
        self,
        X=None,
        batch_size: int | None = None,
        device: str | None = None,
        **kwargs,
    ):
        self.X = X
        self.batch_size = batch_size
        self.device = device

        # Internal attributes
        self._base_argn = None
        self._fitted = False
        self._X_init = X
        self._kwargs = kwargs

        # If X is a fitted TabularARGN, use it as base
        if isinstance(X, TabularARGN):
            if not X._fitted:
                raise ValueError("Provided TabularARGN instance must be fitted")
            self._base_argn = X
            self._fitted = True
            # Copy sklearn-compatible fitted attributes
            if hasattr(X, "n_features_in_"):
                self.n_features_in_ = X.n_features_in_
            if hasattr(X, "feature_names_in_"):
                self.feature_names_in_ = X.feature_names_in_

    def fit(self, X=None, y=None):
        """
        Fit the density estimator.

        If X was provided during initialization and is a fitted TabularARGN, this is a no-op.
        Otherwise, trains a new TabularARGN model.

        Args:
            X: Training data. If None, uses X from initialization.
            y: Target values (optional).

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

        # Create and fit a new TabularARGN
        self._base_argn = TabularARGN(**self._kwargs)
        self._base_argn.fit(X, y=y)
        self._fitted = True

        # Copy sklearn-compatible fitted attributes
        if hasattr(self._base_argn, "n_features_in_"):
            self.n_features_in_ = self._base_argn.n_features_in_
        if hasattr(self._base_argn, "feature_names_in_"):
            self.feature_names_in_ = self._base_argn.feature_names_in_

        return self

    def score_samples(self, X, ctx_data: pd.DataFrame | None = None) -> np.ndarray:
        """
        Compute the log-likelihood of each sample under the model.

        This method estimates the log-probability (log-likelihood) of data samples
        under the fitted generative model. Higher values indicate samples that are
        more likely under the learned distribution.

        Args:
            X: Data samples to score. Can be array-like or pd.DataFrame of shape (n_samples, n_features).
            ctx_data: Optional context data for models trained with context.

        Returns:
            Log-likelihood of each sample as np.ndarray of shape (n_samples,).
            More positive values indicate higher likelihood under the model.
        """
        from mostlyai.engine.log_prob import log_prob

        X_df = ensure_dataframe(X, columns=self._base_argn._feature_names)

        # Convert ctx_data to DataFrame if provided
        ctx_data_df = None
        if ctx_data is not None:
            ctx_data_df = ensure_dataframe(ctx_data)

        workspace_dir = self._base_argn._get_workspace_dir()

        return log_prob(
            tgt_data=X_df,
            ctx_data=ctx_data_df,
            workspace_dir=workspace_dir,
            device=self.device or self._base_argn.device,
        )


class TabularARGNClassifier(TabularARGN):
    """
    TabularARGN classifier with sklearn interface.

    This classifier trains a generative model on the full dataset and uses it
    to predict target classes by conditioning on input features.

    Args:
        X: Training data or a fitted TabularARGN instance.
        target: Name of the target column to predict.
        n_draws: Number of draws to generate for each sample during prediction. Defaults to 10.
        agg_fn: Aggregation function to combine predictions across draws. Defaults to mode (most common value).
        **kwargs: All other arguments are passed to TabularARGN base class.
            See TabularARGN docstring for available parameters.
    """

    def __init__(
        self,
        X=None,
        target: str | None = None,
        n_draws: int = 10,
        agg_fn: Callable = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Store parameters as attributes for sklearn compatibility
        self.X = X
        self.target = target
        self.n_draws = n_draws
        self.agg_fn = agg_fn

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
            self._feature_names = X._feature_names
            # Copy sklearn-compatible fitted attributes
            if hasattr(X, "n_features_in_"):
                self.n_features_in_ = X.n_features_in_
            if hasattr(X, "feature_names_in_"):
                self.feature_names_in_ = X.feature_names_in_
            if hasattr(X, "workspace_path_"):
                self.workspace_path_ = X.workspace_path_

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

        # Call parent fit which trains on full X (including target)
        return super().fit(X, y=y)

    def predict(
        self,
        X,
        ctx_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Input samples.
            ctx_data: Context data for generation. If None, uses the context data from training.
            **kwargs: Additional arguments passed to sample() method.

        Returns:
            Predicted class labels as np.ndarray.
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction. Call fit() first.")

        if self._target_column is None:
            raise ValueError("Target column must be specified for prediction.")

        agg_fn = self.agg_fn
        if agg_fn is None:
            # Default to mode (most common value)
            def mode_fn(x):
                values, counts = np.unique(x, return_counts=True)
                return values[np.argmax(counts)]

            agg_fn = mode_fn

        X_df = ensure_dataframe(X, columns=self._feature_names)

        # Exclude target column from seed if present
        if self._target_column in X_df.columns:
            X_df = X_df.drop(columns=[self._target_column])

        # Create a sampler to generate predictions
        sampler = TabularARGNSampler(self._base_argn, **kwargs)

        # Generate predictions across multiple draws
        all_predictions = []
        for _ in range(self.n_draws):
            samples = sampler.sample(seed_data=X_df, ctx_data=ctx_data)
            all_predictions.append(samples[self._target_column].values)

        # Stack predictions and aggregate
        predictions_array = np.column_stack(all_predictions)
        y_pred = np.apply_along_axis(agg_fn, 1, predictions_array)

        return y_pred

    def predict_proba(
        self,
        X,
        ctx_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Input samples.
            ctx_data: Context data for generation. If None, uses the context data from training.
            **kwargs: Additional arguments passed to sample() method.

        Returns:
            Predicted class probabilities as np.ndarray of shape (n_samples, n_classes).
        """
        if not self._fitted:
            raise ValueError("Classifier must be fitted before prediction. Call fit() first.")

        if self._target_column is None:
            raise ValueError("Target column must be specified for prediction.")

        X_df = ensure_dataframe(X, columns=self._feature_names)

        # Exclude target column from seed if present
        if self._target_column in X_df.columns:
            X_df = X_df.drop(columns=[self._target_column])

        # Create a sampler to generate predictions
        sampler = TabularARGNSampler(self._base_argn, **kwargs)

        # Generate predictions across multiple draws
        all_predictions = []
        for _ in range(self.n_draws):
            samples = sampler.sample(seed_data=X_df, ctx_data=ctx_data)
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
        n_draws: Number of draws to generate for each sample during prediction. Defaults to 10.
        agg_fn: Aggregation function to combine predictions across draws. Defaults to np.mean.
        **kwargs: All other arguments are passed to TabularARGN base class.
            See TabularARGN docstring for available parameters.
    """

    def __init__(
        self,
        X=None,
        target: str | None = None,
        n_draws: int = 10,
        agg_fn: Callable = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Store parameters as attributes for sklearn compatibility
        self.X = X
        self.target = target
        self.n_draws = n_draws
        self.agg_fn = agg_fn

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
            self._feature_names = X._feature_names
            # Copy sklearn-compatible fitted attributes
            if hasattr(X, "n_features_in_"):
                self.n_features_in_ = X.n_features_in_
            if hasattr(X, "feature_names_in_"):
                self.feature_names_in_ = X.feature_names_in_
            if hasattr(X, "workspace_path_"):
                self.workspace_path_ = X.workspace_path_

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

        # Call parent fit which trains on full X (including target)
        return super().fit(X, y=y)

    def predict(
        self,
        X,
        ctx_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict continuous target values for samples in X.

        Args:
            X: Input samples.
            ctx_data: Context data for generation. If None, uses the context data from training.
            **kwargs: Additional arguments passed to sample() method.

        Returns:
            Predicted target values as np.ndarray.
        """
        if not self._fitted:
            raise ValueError("Regressor must be fitted before prediction. Call fit() first.")

        if self._target_column is None:
            raise ValueError("Target column must be specified for prediction.")

        agg_fn = self.agg_fn
        if agg_fn is None:
            agg_fn = np.mean

        X_df = ensure_dataframe(X, columns=self._feature_names)

        # Exclude target column from seed if present
        if self._target_column in X_df.columns:
            X_df = X_df.drop(columns=[self._target_column])

        # Create a sampler to generate predictions
        sampler = TabularARGNSampler(self._base_argn, **kwargs)

        # Generate predictions across multiple draws
        all_predictions = []
        for _ in range(self.n_draws):
            samples = sampler.sample(seed_data=X_df, ctx_data=ctx_data)
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
            self._feature_names = X._feature_names
            # Copy sklearn-compatible fitted attributes
            if hasattr(X, "n_features_in_"):
                self.n_features_in_ = X.n_features_in_
            if hasattr(X, "feature_names_in_"):
                self.feature_names_in_ = X.feature_names_in_
            if hasattr(X, "workspace_path_"):
                self.workspace_path_ = X.workspace_path_

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

    def transform(self, X, ctx_data: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
        """
        Impute missing values in X.

        Args:
            X: Data with missing values to impute.
            ctx_data: Context data for generation. If None, uses the context data from training.
            **kwargs: Additional arguments passed to generate() function.

        Returns:
            Data with imputed values as pd.DataFrame.
        """
        if not self._fitted:
            raise ValueError("Imputer must be fitted before transform. Call fit() first.")

        X_df = ensure_dataframe(X, columns=self._feature_names)

        # Use ctx_data from training if not provided
        if ctx_data is None:
            ctx_data = self.ctx_data

        # Convert ctx_data to DataFrame if provided
        ctx_data_df = None
        if ctx_data is not None:
            ctx_data_df = ensure_dataframe(ctx_data)

        # Use imputation config to specify which columns to impute
        imputation_config = ImputationConfig(columns=X_df.columns.tolist())

        # Create a sampler with imputation config
        sampler = TabularARGNSampler(self._base_argn, imputation=imputation_config, **kwargs)

        # Generate imputed data
        X_imputed = sampler.sample(n_samples=len(X_df), seed_data=X_df, ctx_data=ctx_data_df)

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
