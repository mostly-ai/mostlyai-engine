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
Scikit-learn compatible interface for the MOSTLY AI TabularARGN Models.

This module provides a public interface for the MOSTLY AI TabularARGN Models
for sampling, classification, regression, imputation, and density estimation tasks.
"""

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from mostlyai.engine._common import (
    ensure_dataframe,
    list_fn,
    load_generated_data,
    mean_fn,
    median_fn,
    mode_fn,
)
from mostlyai.engine._encoding_types.tabular.categorical import (
    encode_categorical,
)
from mostlyai.engine._encoding_types.tabular.numeric import (
    encode_numeric,
    NUMERIC_BINNED_UNKNOWN_TOKEN,
    NUMERIC_BINNED_MIN_TOKEN,
    NUMERIC_BINNED_MAX_TOKEN,
)
from mostlyai.engine._workspace import Workspace
from mostlyai.engine.analysis import analyze
from mostlyai.engine.domain import (
    DifferentialPrivacyConfig,
    FairnessConfig,
    ImputationConfig,
    ModelEncodingType,
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
    Scikit-learn compatible class TabularARGN.

    This class wraps the MOSTLY AI TabularARGN Models to provide a public
    interface for training generative models on tabular data.

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
            if not hasattr(self, "_target_column") or self._target_column is None:
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
        self.n_features_in_ = len(self._feature_names)
        self.feature_names_in_ = np.array(self._feature_names)
        self.workspace_path_ = str(workspace_dir)

        return self

    def close(self):
        """Explicitly clean up temporary directory if created."""
        if getattr(self, "_temp_dir", None) is not None:
            try:
                self._temp_dir.cleanup()
                self._temp_dir = None  # Mark as cleaned up
            except Exception:
                pass

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up resources."""
        self.close()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Fallback cleanup if context manager wasn't used."""
        self.close()

    def sample(
        self,
        n_samples: int | None = None,
        seed_data: pd.DataFrame | None = None,
        ctx_data: pd.DataFrame | None = None,
        batch_size: int | None = None,
        sampling_temperature: float = 1.0,
        sampling_top_p: float = 1.0,
        device: str | None = None,
        rare_category_replacement_method: RareCategoryReplacementMethod | str = RareCategoryReplacementMethod.constant,
        rebalancing: RebalancingConfig | dict | None = None,
        imputation: ImputationConfig | dict | None = None,
        fairness: FairnessConfig | dict | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic samples from the fitted model.

        Args:
            n_samples: Number of samples to generate. If None and ctx_data is provided, infers from ctx_data length.
                      If None and no ctx_data, defaults to 1.
            seed_data: Seed data to condition generation on fixed columns.
            ctx_data: Context data for generation. If None, uses the context data from training.
            batch_size: Batch size for generation. If None, determined automatically.
            sampling_temperature: Sampling temperature. Higher values increase randomness. Defaults to 1.0.
            sampling_top_p: Nucleus sampling probability threshold. Defaults to 1.0.
            device: Device to run generation on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
            rare_category_replacement_method: Method for handling rare categories.
            rebalancing: Configuration for rebalancing column distributions.
            imputation: Configuration for imputing missing values.
            fairness: Configuration for fairness constraints.

        Returns:
            Generated synthetic samples as pd.DataFrame.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling. Call fit() first.")

        workspace_dir = self._get_workspace_dir()

        # Determine if ctx_data was explicitly provided (vs using training default)
        ctx_data_explicit = ctx_data is not None

        # Use ctx_data from training if not provided
        if ctx_data is None:
            ctx_data = self.ctx_data

        # Convert ctx_data to DataFrame if provided
        ctx_data_df = None
        if ctx_data is not None:
            ctx_data_df = ensure_dataframe(ctx_data)

            # Infer n_samples from ctx_data if it was explicitly provided and n_samples not specified
            if ctx_data_explicit and n_samples is None:
                n_samples = len(ctx_data_df)

            # For sequential models: if ctx_data was not explicitly provided and n_samples is specified,
            # take a random sample of the training ctx_data
            if not ctx_data_explicit and n_samples is not None and self.tgt_context_key is not None:
                if len(ctx_data_df) > n_samples:
                    ctx_data_df = ctx_data_df.sample(n=n_samples, random_state=self.random_state)

        # Default n_samples to 1 if still None
        if n_samples is None:
            n_samples = 1

        # Generate synthetic data using configured parameters
        generate(
            ctx_data=ctx_data_df,
            seed_data=seed_data,
            sample_size=None if seed_data is not None else n_samples,
            batch_size=batch_size,
            sampling_temperature=sampling_temperature,
            sampling_top_p=sampling_top_p,
            device=device or self.device,
            rare_category_replacement_method=rare_category_replacement_method,
            rebalancing=rebalancing,
            imputation=imputation,
            fairness=fairness,
            workspace_dir=workspace_dir,
        )

        # Load and return synthetic data
        synthetic_data = load_generated_data(workspace_dir)
        if self.tgt_context_key is not None:
            synthetic_data = synthetic_data.sort_values(by=self.tgt_context_key).reset_index(drop=True)
        else:
            synthetic_data = synthetic_data.reset_index(drop=True)

        return synthetic_data

    def impute(
        self,
        X,
        ctx_data: pd.DataFrame | None = None,
        n_draws: int = 1,
        agg_fn: Literal["mode", "mean", "median", "list"] | Callable[[np.ndarray], Any] = "mode",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Impute missing values in X.

        Args:
            X: Data with missing values to impute.
            ctx_data: Context data for generation. If None, uses the context data from training.
            n_draws: Number of draws to generate for each row during imputation. Defaults to 1.
            agg_fn: Aggregation method to combine imputed values across draws. Options:
                - "mode" (default): Most common value across draws.
                - "mean": Average across draws.
                - "median": Median across draws.
                - "list": Return all draws as arrays.
                - Callable: A function that takes a 1D numpy array and returns a scalar or array.
            **kwargs: Additional arguments passed to sample() method.

        Returns:
            Data with imputed values as pd.DataFrame.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before imputation. Call fit() first.")

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

        # If n_draws == 1, use simple imputation (no aggregation needed)
        if n_draws == 1:
            X_imputed = self.sample(
                n_samples=len(X_df), seed_data=X_df, ctx_data=ctx_data_df, imputation=imputation_config, **kwargs
            )
        else:
            # Resolve aggregation function
            if callable(agg_fn):
                agg_fn_func = agg_fn
                is_list_agg = False
            else:
                # Map aggregation method to function
                agg_fn_map = {"mode": mode_fn, "mean": mean_fn, "median": median_fn, "list": list_fn}
                agg_fn_func = agg_fn_map[agg_fn]
                is_list_agg = agg_fn == "list"

            # Generate multiple imputed datasets
            all_imputations = []
            for _ in range(n_draws):
                imputed = self.sample(
                    n_samples=len(X_df), seed_data=X_df, ctx_data=ctx_data_df, imputation=imputation_config, **kwargs
                )
                all_imputations.append(imputed)

            # Aggregate across imputations for each column
            X_imputed = X_df.copy()
            for col in X_df.columns:
                # Stack the column values from all imputations
                col_values = np.column_stack([imp[col].values for imp in all_imputations])
                # Apply aggregation function row-wise
                if is_list_agg:
                    # For list aggregation, store arrays directly
                    X_imputed[col] = [agg_fn_func(row) for row in col_values]
                else:
                    X_imputed[col] = np.apply_along_axis(agg_fn_func, 1, col_values)

        return X_imputed

    def _get_column_stats(self) -> dict:
        """Get column statistics from workspace."""
        workspace_dir = self._get_workspace_dir()
        workspace = Workspace(workspace_dir)
        stats = workspace.tgt_stats.read()
        return stats.get("columns", {})

    def _get_target_encoding_type(self, target_column: str | None = None) -> str | None:
        """Get the encoding type of the target column from workspace stats."""
        if not self._fitted:
            return None

        target_col = target_column or self._target_column
        if target_col is None:
            return None

        columns = self._get_column_stats()
        if target_col in columns:
            return columns[target_col].get("encoding_type")
        return None

    def _is_numeric_or_datetime_encoding(self, encoding_type: str | None) -> bool:
        """Check if encoding type is numeric or datetime."""
        if encoding_type is None:
            return False
        encoding_type_upper = encoding_type.upper()
        return "NUMERIC" in encoding_type_upper or encoding_type_upper in [
            "TABULAR_DATETIME",
            "TABULAR_DATETIME_RELATIVE",
            "LANGUAGE_DATETIME",
        ]

    def _resolve_agg_fn(
        self,
        agg_fn: Literal["auto", "mode", "mean", "median", "list"] | Callable[[np.ndarray], Any] | None = None,
        target_column: str | None = None,
    ) -> Callable:
        """
        Resolve aggregation function based on agg_fn parameter and target column encoding type.

        Args:
            agg_fn: Aggregation method. If "auto", uses mean for numeric/datetime columns, mode for others.
                   Can also be a callable function that takes a 1D numpy array and returns a scalar or array.
            target_column: Target column name for encoding type lookup when agg_fn is "auto".

        Returns:
            Aggregation function to use.
        """
        if agg_fn is None:
            agg_fn = "auto"

        # If agg_fn is already a callable, return it directly
        if callable(agg_fn):
            return agg_fn

        if agg_fn == "auto":
            encoding_type = self._get_target_encoding_type(target_column)
            if self._is_numeric_or_datetime_encoding(encoding_type):
                return mean_fn
            else:
                return mode_fn
        elif agg_fn == "mean":
            return mean_fn
        elif agg_fn == "median":
            return median_fn
        elif agg_fn == "mode":
            return mode_fn
        elif agg_fn == "list":
            return list_fn
        else:
            raise ValueError(
                f"Invalid agg_fn: {agg_fn}. Must be one of 'auto', 'mean', 'median', 'mode', 'list', or a callable function."
            )

    def predict(
        self,
        X,
        target: str | None = None,
        ctx_data: pd.DataFrame | None = None,
        n_draws: int = 1,
        agg_fn: Literal["auto", "mode", "mean", "median", "list"] | Callable[[np.ndarray], Any] = "auto",
        **kwargs,
    ) -> np.ndarray:
        """
        Predict target values for samples in X.

        This method generates synthetic samples conditioned on the input features and aggregates
        the target column predictions across multiple draws.

        Args:
            X: Input samples. Can be array-like or pd.DataFrame of shape (n_samples, n_features).
            target: Name of the target column to predict. If None, uses the target column from fit().
            ctx_data: Context data for generation. If None, uses the context data from training.
            n_draws: Number of draws to generate for each sample during prediction. Defaults to 1.
            agg_fn: Aggregation method to combine predictions across draws. Options:
                - "auto" (default): Automatically selects aggregation based on target column encoding type.
                  Uses "mean" for numeric and datetime columns, "mode" for others.
                - "mean": Average across draws (suitable for numeric/datetime targets).
                - "median": Median across draws (suitable for numeric/datetime targets).
                - "mode": Most common value across draws (suitable for categorical targets).
                - "list": Return all draws as arrays (returns shape (n_samples, n_draws) instead of (n_samples,)).
                - Callable: A function that takes a 1D numpy array and returns a scalar or array.
            **kwargs: Additional arguments passed to sample() method.

        Returns:
            Predicted target values as np.ndarray of shape (n_samples,) or (n_samples, n_draws) if agg_fn="list".
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Determine target column
        target_column = target or self._target_column
        if target_column is None:
            raise ValueError(
                "Target column must be specified for prediction. Provide 'target' parameter or fit with y."
            )

        # Resolve aggregation function
        agg_fn_func = self._resolve_agg_fn(agg_fn, target_column)

        X_df = ensure_dataframe(X, columns=self._feature_names)

        # Exclude target column from seed if present
        if target_column in X_df.columns:
            X_df = X_df.drop(columns=[target_column])

        # Generate predictions across multiple draws
        all_predictions = []
        for _ in range(n_draws):
            samples = self.sample(seed_data=X_df, ctx_data=ctx_data, **kwargs)
            if target_column in samples.columns:
                all_predictions.append(samples[target_column].values)
            else:
                raise ValueError(f"Target column '{target_column}' not found in generated samples")

        # Stack predictions and aggregate
        predictions_array = np.column_stack(all_predictions)

        # Check if we should return all draws (list aggregation)
        is_list_agg = (agg_fn == "list") if not callable(agg_fn) else False

        if is_list_agg:
            # For list aggregation, return all draws as a 2D array
            return predictions_array
        else:
            y_pred = np.apply_along_axis(agg_fn_func, 1, predictions_array)
            return y_pred

    def predict_proba(
        self,
        X,
        target: str | None = None,
        ctx_data: pd.DataFrame | None = None,
        n_draws: int = 1,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predict class probabilities for samples in X.

        This method generates synthetic samples conditioned on the input features and computes
        the probability of each class based on the frequency across multiple draws.

        Supported encoding types:
        - tabular_categorical: Multi-class categorical variables
        - tabular_numeric_discrete: Discrete numeric variables
        - tabular_numeric_binned: Binned numeric variables

        Args:
            X: Input samples. Can be array-like or pd.DataFrame of shape (n_samples, n_features).
            target: Name of the target column to predict. If None, uses the target column from fit().
            ctx_data: Context data for generation. If None, uses the context data from training.
            n_draws: Number of draws to generate for each sample during prediction. Defaults to 1.
            **kwargs: Additional arguments passed to sample() method.

        Returns:
            pd.DataFrame with shape (n_samples, n_classes) where columns are named by class labels.
            Column names are derived from encoding stats:
            - Binned: Special tokens (e.g., "<<NULL>>"), min value, bin ranges (e.g., "<10000", ">=50000"), and max value (e.g., "0", ..., "99999")
            - Categorical: All category names including special tokens (e.g., "_RARE_", "<<NULL>>", "male", "female")
            - Discrete: All numeric values including special tokens (e.g., "_RARE_", "<<NULL>>", "1", "2", "3")
            Each row contains probability distribution that sums to 1.0.

        Raises:
            ValueError: If target column has unsupported encoding type.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        # Determine target column
        target_column = target or self._target_column
        if target_column is None:
            raise ValueError(
                "Target column must be specified for prediction. Provide 'target' parameter or fit with y."
            )

        # Get target column stats and validate encoding type
        columns = self._get_column_stats()
        if target_column not in columns:
            raise ValueError(f"Target column '{target_column}' not found in model statistics")

        target_stats = columns[target_column]
        encoding_type = target_stats.get("encoding_type")
        if not encoding_type:
            raise ValueError(f"Target column '{target_column}' has no encoding type")

        X_df = ensure_dataframe(X, columns=self._feature_names)

        # Exclude target column from seed if present
        if target_column in X_df.columns:
            X_df = X_df.drop(columns=[target_column])

        # Repeat X_df n_draws times for batch sampling
        n_samples = len(X_df)
        X_df_repeated = pd.concat([X_df] * n_draws, ignore_index=True)

        # Generate all samples in a single batch
        samples = self.sample(seed_data=X_df_repeated, ctx_data=ctx_data, **kwargs)
        if target_column not in samples.columns:
            raise ValueError(f"Target column '{target_column}' not found in generated samples")

        # Re-encode the sampled values using the same encoding logic as the generator
        sampled_values = samples[target_column]

        if encoding_type == ModelEncodingType.tabular_numeric_binned:
            encoded_df = encode_numeric(sampled_values, target_stats)
            codes = encoded_df["bin"].values
        elif encoding_type == ModelEncodingType.tabular_numeric_discrete:
            encoded_df = encode_numeric(sampled_values, target_stats)
            codes = encoded_df["cat"].values
        elif encoding_type == ModelEncodingType.tabular_categorical:
            encoded_df = encode_categorical(sampled_values, target_stats)
            codes = encoded_df["cat"].values
        else:
            raise ValueError(
                f"Target column '{target_column}' has unsupported encoding type '{encoding_type}'. "
                f"Only categorical, discrete numeric, and binned numeric columns are supported."
            )

        # Reshape codes to (n_samples, n_draws)
        codes_array = codes.reshape(n_samples, n_draws)

        # Build class labels with their corresponding code values
        codes = target_stats["codes"]
        class_labels = []
        class_code_values = []

        # For binned: replace <<MIN>>/<<MAX>> tokens with actual bin values and reorder
        if encoding_type == ModelEncodingType.tabular_numeric_binned:
            bins = target_stats["bins"]
            min_code_value = None
            max_code_value = None

            # First add special tokens (except <<UNK>>, <<MIN>>, <<MAX>>)
            for code_name, code_value in codes.items():
                if code_name == NUMERIC_BINNED_UNKNOWN_TOKEN:
                    continue  # Skip <<UNK>>
                elif code_name == NUMERIC_BINNED_MIN_TOKEN:
                    min_code_value = code_value  # Save for later
                elif code_name == NUMERIC_BINNED_MAX_TOKEN:
                    max_code_value = code_value  # Save for later
                else:
                    class_labels.append(code_name)
                    class_code_values.append(code_value)

            # Add MIN value if present
            if min_code_value is not None:
                class_labels.append(str(bins[0]))
                class_code_values.append(min_code_value)

            # Add bin range labels
            codes_bin_offset = len(codes)
            for i in range(len(bins) - 1):
                lower_bound = bins[i]
                upper_bound = bins[i + 1]
                label = f">={lower_bound}" if i == len(bins) - 2 else f"<{upper_bound}"
                class_labels.append(label)
                class_code_values.append(i + codes_bin_offset)

            # Add MAX value at the end if present
            if max_code_value is not None:
                class_labels.append(str(bins[-1]))
                class_code_values.append(max_code_value)
        else:
            # For discrete and categorical: use code names as-is
            class_labels = list(codes.keys())
            class_code_values = list(codes.values())

        # Calculate probabilities for all classes at once
        proba_list = [np.mean(codes_array == code_value, axis=1) for code_value in class_code_values]
        proba = np.column_stack(proba_list)

        return pd.DataFrame(proba, columns=class_labels)
