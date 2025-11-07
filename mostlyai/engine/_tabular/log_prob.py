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

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mostlyai.engine._tabular.common import load_model
from mostlyai.engine._tabular.encoding import encode_df
from mostlyai.engine._tabular.training import _calculate_sample_losses

_LOG = logging.getLogger(__name__)


def log_prob(
    tgt_data: pd.DataFrame,
    *,
    ctx_data: pd.DataFrame | None = None,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
) -> np.ndarray:
    """
    Compute log-probability (log-likelihood) of samples under the trained flat tabular model.

    Args:
        tgt_data: Target data samples to score.
        ctx_data: Optional context data for models trained with context. Must have
            the same number of rows as tgt_data.
        device: Device to run computation on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        workspace_dir: Directory path for workspace containing trained model.

    Returns:
        Log-probability of each sample as np.ndarray of shape (n_samples,).
        More positive values indicate higher likelihood under the model.

    Raises:
        ValueError: If model has not been trained yet, if model is sequential,
            if ctx_data is provided when model has no context, or if ctx_data
            has different number of rows than tgt_data.
    """
    _LOG.info("LOG_PROB_TABULAR started")

    # Set device
    device = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    _LOG.info(f"{device=}")

    # Load model and metadata
    model, metadata = load_model(
        workspace_dir=workspace_dir,
        device=device,
    )

    # Extract metadata
    tgt_stats = metadata["tgt_stats"]
    ctx_stats = metadata["ctx_stats"]
    ctx_cardinalities = metadata["ctx_cardinalities"]
    is_sequential = metadata["is_sequential"]
    has_context = metadata["has_context"]

    # Check if model is sequential and raise error
    if is_sequential:
        raise ValueError(
            "This log_prob function only supports flat models. The loaded model is sequential, which is not supported."
        )

    # Validate ctx_data usage
    if ctx_data is not None and not has_context:
        raise ValueError(
            "ctx_data was provided but the model was not trained with context data. "
            "Remove ctx_data parameter or train a model with context."
        )

    if has_context and ctx_data is None and ctx_cardinalities:
        # Model was trained with context, but no context data provided
        # This is allowed - we'll use empty context (all zeros)
        _LOG.warning(
            "Model was trained with context data but ctx_data was not provided. "
            "Computing log-probabilities without context information."
        )

    # Get keys
    ctx_primary_key = ctx_stats.get("keys", {}).get("primary_key")

    # Encode target data
    tgt_data_encoded, _, _ = encode_df(df=tgt_data, stats=tgt_stats, tgt_context_key=None)

    # Encode context data if provided
    if ctx_data is not None:
        ctx_data_encoded, ctx_primary_key_encoded, _ = encode_df(
            df=ctx_data, stats=ctx_stats, ctx_primary_key=ctx_primary_key
        )
    else:
        ctx_data_encoded = pd.DataFrame()
        ctx_primary_key_encoded = None

    # Validate alignment between tgt_data and ctx_data
    if ctx_data is not None and len(tgt_data_encoded) != len(ctx_data_encoded):
        raise ValueError(
            f"tgt_data and ctx_data must have the same number of rows. "
            f"Got {len(tgt_data_encoded)} target rows and {len(ctx_data_encoded)} context rows."
        )

    # Convert to tensors and move to device
    data_dict = {}

    # Process target columns
    for col in tgt_data_encoded.columns:
        values = tgt_data_encoded[col].values
        # Flat target data
        if values.dtype == object:
            values = pd.to_numeric(values, errors="coerce").fillna(0).astype(np.int64)
        else:
            values = values.astype(np.int64)
        tensor = torch.from_numpy(values).to(device).unsqueeze(-1)
        data_dict[col] = tensor

    # Process context columns if they exist
    if not ctx_data_encoded.empty:
        # Filter out key columns from context data
        ctx_data_cols = [col for col in ctx_data_encoded.columns if col != ctx_primary_key_encoded]

        for col in ctx_data_cols:
            values = ctx_data_encoded[col].values
            # Flat context columns
            if values.dtype == object:
                values = pd.to_numeric(values, errors="coerce").fillna(0).astype(np.int64)
            else:
                values = values.astype(np.int64)
            tensor = torch.from_numpy(values).to(device).unsqueeze(-1)
            data_dict[col] = tensor

    # Calculate sample losses (negative log-likelihood)
    with torch.no_grad():
        losses = _calculate_sample_losses(model, data_dict)

    # Convert to numpy and negate (to get log-likelihood instead of loss)
    log_likelihood = -losses.cpu().numpy()

    _LOG.info(f"LOG_PROB_TABULAR completed for {len(tgt_data)} samples")
    return log_likelihood
