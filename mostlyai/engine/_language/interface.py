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
Scikit-learn compatible interface for the MOSTLY AI Language Models.

This module provides sklearn-compatible estimators that wrap the MOSTLY AI engine
for training language models and generating synthetic text data.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator

from mostlyai.engine._common import ensure_dataframe, load_generated_data
from mostlyai.engine.analysis import analyze
from mostlyai.engine.domain import (
    DifferentialPrivacyConfig,
    ModelType,
    RareCategoryReplacementMethod,
)
from mostlyai.engine.encoding import encode
from mostlyai.engine.generation import generate
from mostlyai.engine.logging import disable_logging, init_logging
from mostlyai.engine.splitting import split
from mostlyai.engine.training import train

_LOG = logging.getLogger(__name__)


class LanguageModel(BaseEstimator):
    """
    Scikit-learn compatible class LanguageModel.

    This class wraps the MOSTLY AI Language Models to provide a public
    interface for training generative models on text data.

    Args:
        tgt_encoding_types: Dictionary mapping column names to encoding types.
            Example: {'category': 'LANGUAGE_CATEGORICAL', 'headline': 'LANGUAGE_TEXT'}
        model: The identifier of the language model to train. Defaults to MOSTLY_AI/LSTMFromScratch-3m.
        max_training_time: Maximum training time in minutes. Defaults to 14400 (10 days).
        max_epochs: Maximum number of training epochs. Defaults to 100.
        batch_size: Per-device batch size for training and validation. If None, determined automatically.
        gradient_accumulation_steps: Number of steps to accumulate gradients. If None, determined automatically.
        enable_flexible_generation: Whether to enable flexible order generation. Defaults to True.
        value_protection: Whether to enable value protection for rare values. Defaults to True.
        differential_privacy: Configuration for differential privacy training. If None, DP is disabled.
        tgt_context_key: Context key column name in the target data for sequential models.
        tgt_primary_key: Primary key column name in the target data.
        ctx_data: DataFrame containing the context data for two-table sequential models.
        ctx_primary_key: Primary key column name in the context data.
        ctx_encoding_types: Dictionary mapping context column names to encoding types.
        device: Device to run training on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        workspace_dir: Directory path for workspace. If None, a temporary directory will be created.
        random_state: Random seed for reproducibility.
        verbose: Verbosity level. 0 = silent, 1 = progress messages.
    """

    def __init__(
        self,
        tgt_encoding_types: dict[str, str] | None = None,
        model: str | None = None,
        max_training_time: float | None = 14400.0,
        max_epochs: float | None = 100.0,
        batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,
        enable_flexible_generation: bool = True,
        value_protection: bool = True,
        differential_privacy: DifferentialPrivacyConfig | dict | None = None,
        tgt_context_key: str | None = None,
        tgt_primary_key: str | None = None,
        ctx_data: pd.DataFrame | None = None,
        ctx_primary_key: str | None = None,
        ctx_encoding_types: dict[str, str] | None = None,
        device: torch.device | str | None = None,
        workspace_dir: str | Path | None = None,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.tgt_encoding_types = tgt_encoding_types
        self.model = model
        self.max_training_time = max_training_time
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.enable_flexible_generation = enable_flexible_generation
        self.value_protection = value_protection
        self.differential_privacy = differential_privacy
        self.tgt_context_key = tgt_context_key
        self.tgt_primary_key = tgt_primary_key
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

        # Initialize or disable logging based on verbose setting
        if self.verbose > 0:
            init_logging()
        else:
            disable_logging()

    def _get_workspace_dir(self) -> Path:
        """Get or create workspace directory."""
        if self._workspace_path is not None:
            return self._workspace_path

        if self.workspace_dir is not None:
            self._workspace_path = Path(self.workspace_dir)
            return self._workspace_path

        if self._temp_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="mostlyai_language_")
            self._workspace_path = Path(self._temp_dir.name)
            self.workspace_dir = str(self._workspace_path)

        return self._workspace_path

    def _set_random_state(self):
        """Set random state for reproducibility."""
        if self.random_state is not None:
            from mostlyai.engine import set_random_state

            set_random_state(self.random_state)

    def fit(self, X):
        """
        Fit the LanguageModel on training data.

        This method wraps the MOSTLY AI engine's split(), analyze(), encode(), and train() pipeline
        for language models.

        Args:
            X: Training data as pd.DataFrame of shape (n_samples, n_features).

        Returns:
            self: Returns self.
        """
        self._set_random_state()

        # Convert to DataFrame
        X_df = ensure_dataframe(X)
        self._feature_names = list(X_df.columns)

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
            model_type=ModelType.language,
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
        model_id = self.model or "MOSTLY_AI/LSTMFromScratch-3m"
        train(
            model=model_id,
            max_training_time=self.max_training_time,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            enable_flexible_generation=self.enable_flexible_generation,
            differential_privacy=self.differential_privacy,
            device=self.device,
            workspace_dir=workspace_dir,
        )

        self._fitted = True

        # Add sklearn-compatible fitted attributes
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

        Returns:
            Generated synthetic samples as pd.DataFrame.
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling. Call fit() first.")

        workspace_dir = self._get_workspace_dir()

        # Determine if ctx_data was explicitly provided
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

        # Default n_samples to 1 if still None and no seed_data
        if n_samples is None and seed_data is None:
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
            workspace_dir=workspace_dir,
        )

        # Load and return synthetic data
        synthetic_data = load_generated_data(workspace_dir)

        return synthetic_data
