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
Sklearn-compatible interface for MOSTLY AI engine models.
"""

import tempfile
from pathlib import Path

import pandas as pd
import torch

try:
    from sklearn.base import BaseEstimator
except ImportError:
    raise ImportError(
        "scikit-learn is required for the sklearn interface. "
        "Install it with: pip install scikit-learn"
    )

from mostlyai.engine.analysis import analyze
from mostlyai.engine.domain import DifferentialPrivacyConfig, ModelType
from mostlyai.engine.encoding import encode
from mostlyai.engine.generation import generate
from mostlyai.engine.logging import init_logging
from mostlyai.engine.splitting import split
from mostlyai.engine.training import train


class LanguageModel(BaseEstimator):
    """
    Language model for generating synthetic text data using pre-trained LLMs or LSTM models.
    
    This class provides a sklearn-compatible interface for training and generating
    synthetic language/text data. It uses language models (either pre-trained LLMs 
    or LSTM trained from scratch) to learn patterns in text data and generate 
    synthetic samples.
    
    Note: Unlike tabular models, language models do not support:
    - log_prob() method for computing log probabilities
    - Imputation, fairness, and rebalancing during generation
    
    Parameters
    ----------
    model : str, optional
        The identifier of the model to use. Defaults to "MOSTLY_AI/LSTMFromScratch-3m".
        Can be a pre-trained language model or an LSTM model.
        
    max_training_time : float, optional
        Maximum training time in minutes. Defaults to 14400.0 (10 days).
        
    max_epochs : float, optional
        Maximum number of training epochs. Defaults to 100.0.
        
    batch_size : int, optional
        Per-device batch size for training and validation. If None, determined automatically.
        
    gradient_accumulation_steps : int, optional
        Number of steps to accumulate gradients. If None, determined automatically.
        
    enable_flexible_generation : bool, default=True
        Whether to enable flexible order generation.
        
    value_protection : bool, optional
        Whether to enable value protection during training.
        
    differential_privacy : DifferentialPrivacyConfig or dict, optional
        Configuration for differential privacy training. If None, DP is disabled.
        
    tgt_context_key : str, optional
        Context key column name in the target data.
        
    tgt_primary_key : str, optional
        Primary key column name in the target data.
        
    ctx_data : pd.DataFrame, optional
        Context data for two-table scenarios.
        
    ctx_primary_key : str, optional
        Primary key column name in the context data.
        
    device : torch.device or str, optional
        Device to run training/generation on ('cuda' or 'cpu'). 
        Defaults to 'cuda' if available, else 'cpu'.
        
    workspace_dir : str or Path, optional
        Directory path for workspace. If None, a temporary directory will be created.
        Training outputs are stored in ModelStore subdirectory.
        
    random_state : int, optional
        Random seed for reproducibility.
        
    verbose : int, default=1
        Verbosity level. 0 for silent, 1 for progress messages.
        
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
        
    feature_names_in_ : ndarray
        Names of features seen during fit.
        
    workspace_path_ : Path
        Path to the workspace directory used for this model.
        
    Examples
    --------
    >>> from mostlyai.engine.sklearn_interface import LanguageModel
    >>> import pandas as pd
    >>> 
    >>> # Create sample text data
    >>> df = pd.DataFrame({
    ...     'text': ['Hello world', 'Good morning', 'How are you', 'Nice day']
    ... })
    >>> 
    >>> # Train model
    >>> model = LanguageModel(model='MOSTLY_AI/LSTMFromScratch-3m', max_epochs=10)
    >>> model.fit(df)
    >>> 
    >>> # Generate synthetic data
    >>> synthetic = model.sample(n_samples=10, seed=42)
    """
    
    def __init__(
        self,
        model: str = "MOSTLY_AI/LSTMFromScratch-3m",
        max_training_time: float | None = 14400.0,
        max_epochs: float | None = 100.0,
        batch_size: int | None = None,
        gradient_accumulation_steps: int | None = None,
        enable_flexible_generation: bool = True,
        value_protection: bool | None = None,
        differential_privacy: DifferentialPrivacyConfig | dict | None = None,
        tgt_context_key: str | None = None,
        tgt_primary_key: str | None = None,
        ctx_data: pd.DataFrame | None = None,
        ctx_primary_key: str | None = None,
        device: torch.device | str | None = None,
        workspace_dir: str | Path | None = None,
        random_state: int | None = None,
        verbose: int = 1,
    ):
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
        self.device = device
        self.workspace_dir = workspace_dir
        self.random_state = random_state
        self.verbose = verbose
        
        self._fitted = False
        self._temp_dir = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the language model on the provided data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data containing text/language columns.
            
        y : Ignored
            Not used, present for sklearn API compatibility.
            
        Returns
        -------
        self : LanguageModel
            Fitted estimator.
        """
        # Initialize logging
        if self.verbose > 0:
            init_logging(level="INFO")
        else:
            init_logging(level="WARNING")
        
        # Setup workspace
        if self.workspace_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            workspace_path = Path(self._temp_dir.name)
        else:
            workspace_path = Path(self.workspace_dir)
        
        self.workspace_path_ = workspace_path
        
        # Store feature information
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.to_numpy()
        
        # Split data with model_type=ModelType.language
        split(
            tgt_data=X,
            ctx_data=self.ctx_data,
            tgt_primary_key=self.tgt_primary_key,
            ctx_primary_key=self.ctx_primary_key,
            tgt_context_key=self.tgt_context_key,
            model_type=ModelType.language,
            workspace_dir=workspace_path,
        )
        
        # Analyze data
        analyze(
            workspace_dir=workspace_path,
            differential_privacy=self.differential_privacy,
        )
        
        # Encode data
        encode(workspace_dir=workspace_path)
        
        # Train the model
        train(
            model=self.model,
            max_training_time=self.max_training_time,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            enable_flexible_generation=self.enable_flexible_generation,
            differential_privacy=self.differential_privacy,
            device=self.device,
            workspace_dir=workspace_path,
        )
        
        self._fitted = True
        return self
    
    def sample(
        self,
        n_samples: int | None = None,
        seed: int | None = None,
        ctx_data: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate synthetic samples from the fitted model.
        
        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. If None, generates the same number
            as in the training data.
            
        seed : int, optional
            Random seed for generation. If None, uses random_state from init.
            
        ctx_data : pd.DataFrame, optional
            Context data for generation. If None, uses ctx_data from init.
            
        **kwargs : dict
            Additional parameters for generation:
            - sampling_temperature : float, default=1.0
            - sampling_top_p : float, default=1.0
            - batch_size : int, optional
            
        Returns
        -------
        pd.DataFrame
            Generated synthetic data.
            
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if not self._fitted:
            raise ValueError(
                "Model must be fitted before calling sample(). "
                "Call fit(X) first."
            )
        
        # Use provided ctx_data or fall back to the one from init
        ctx_data = ctx_data if ctx_data is not None else self.ctx_data
        
        # Extract generation parameters from kwargs
        sampling_temperature = kwargs.get("sampling_temperature", 1.0)
        sampling_top_p = kwargs.get("sampling_top_p", 1.0)
        batch_size = kwargs.get("batch_size", self.batch_size)
        
        # Set seed if provided
        if seed is not None:
            from mostlyai.engine.random_state import set_random_state
            set_random_state(seed)
        elif self.random_state is not None:
            from mostlyai.engine.random_state import set_random_state
            set_random_state(self.random_state)
        
        # Generate synthetic data
        generate(
            ctx_data=ctx_data,
            sample_size=n_samples,
            batch_size=batch_size,
            sampling_temperature=sampling_temperature,
            sampling_top_p=sampling_top_p,
            device=self.device,
            workspace_dir=self.workspace_path_,
        )
        
        # Load and return generated data
        from mostlyai.engine._workspace import Workspace
        
        ws = Workspace(self.workspace_path_)
        synthetic_files = ws.generated_data.fetch_multiple()
        
        if not synthetic_files:
            raise RuntimeError("No synthetic data files were generated")
        
        # Read all generated parquet files
        dfs = [pd.read_parquet(f) for f in synthetic_files]
        synthetic_df = pd.concat(dfs, ignore_index=True)
        
        return synthetic_df
    
    def __del__(self):
        """Clean up temporary directory if created."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
