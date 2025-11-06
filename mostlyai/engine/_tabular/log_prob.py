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
import warnings
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mostlyai.engine._common import (
    CTXFLT,
    CTXSEQ,
    DEFAULT_HAS_RIDX,
    DEFAULT_HAS_SDEC,
    DEFAULT_HAS_SLEN,
    RIDX_SUB_COLUMN_PREFIX,
    SDEC_SUB_COLUMN_PREFIX,
    SIDX_SUB_COLUMN_PREFIX,
    SLEN_SUB_COLUMN_PREFIX,
    encode_positional_column,
    get_cardinalities,
    get_columns_from_cardinalities,
    get_sequence_length_stats,
)
from mostlyai.engine._tabular.argn import FlatModel, SequentialModel
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._tabular.encoding import encode_df
from mostlyai.engine._tabular.training import _calculate_sample_losses
from mostlyai.engine._workspace import Workspace, ensure_workspace_dir

_LOG = logging.getLogger(__name__)


def log_prob(
    tgt_data: pd.DataFrame,
    *,
    ctx_data: pd.DataFrame | None = None,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
) -> np.ndarray:
    """
    Compute log-probability (log-likelihood) of samples under the trained tabular model.

    Args:
        tgt_data: Target data samples to score.
        ctx_data: Optional context data for models trained with context. If the model was trained with
            context data, this should be provided with the same structure. For flat models, must have
            the same number of rows as tgt_data. For sequential models, should contain the context
            records that link to tgt_data via the context key.
        device: Device to run computation on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        workspace_dir: Directory path for workspace containing trained model.

    Returns:
        Log-probability of each sample as np.ndarray of shape (n_samples,).
        More positive values indicate higher likelihood under the model.

    Raises:
        ValueError: If model has not been trained yet, or if ctx_data is provided when model has no context.
    """
    _LOG.info("LOG_PROB_TABULAR started")
    workspace = Workspace(ensure_workspace_dir(workspace_dir))

    # Check if all required files and keys exist before proceeding
    if not workspace.tgt_stats.path.exists() or not workspace.model_configs.path.exists():
        raise ValueError("Model statistics or config missing. Train the model first.")

    # Load stats and config after all checks
    tgt_stats = workspace.tgt_stats.read()
    model_configs = workspace.model_configs.read()

    # Prepare sequential info
    tgt_stats.setdefault("is_sequential", False)
    is_sequential = tgt_stats["is_sequential"]
    seq_len_stats = get_sequence_length_stats(tgt_stats)

    # Load context stats
    has_context = workspace.ctx_stats_path.exists()
    if has_context:
        ctx_stats = workspace.ctx_stats.read()
    else:
        ctx_stats = {"columns": {}, "is_sequential": False}

    tgt_cardinalities = get_cardinalities(tgt_stats)
    ctx_cardinalities = get_cardinalities(ctx_stats)

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
    tgt_context_key = tgt_stats.get("keys", {}).get("context_key")
    ctx_primary_key = ctx_stats.get("keys", {}).get("primary_key")

    # Get column order from configs (for backwards compatibility)
    trn_column_order = model_configs.get("trn_column_order")
    if trn_column_order is None and not model_configs.get("enable_flexible_generation", True):
        # Fixed column order based on cardinalities
        trn_column_order = get_columns_from_cardinalities(tgt_cardinalities)

    # Set device
    device = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    _LOG.info(f"{device=}")

    # Get model size
    model_units = model_configs.get("model_units")
    if model_units is None:
        raise ValueError("Model units not found in model config")

    # Create model
    if is_sequential:
        tgt_seq_len_median = model_configs.get("tgt_seq_len_median", seq_len_stats.get("median", 1))
        tgt_seq_len_max = model_configs.get("tgt_seq_len_max", seq_len_stats.get("max", 1))
        ctx_seq_len_median = model_configs.get("ctx_seq_len_median", {})

        model = SequentialModel(
            tgt_cardinalities=tgt_cardinalities,
            tgt_seq_len_median=tgt_seq_len_median,
            tgt_seq_len_max=tgt_seq_len_max,
            ctx_cardinalities=ctx_cardinalities,
            ctxseq_len_median=ctx_seq_len_median,
            model_size=model_units,
            column_order=trn_column_order,
            device=device,
        )
    else:
        ctx_seq_len_median = model_configs.get("ctx_seq_len_median", {})

        model = FlatModel(
            tgt_cardinalities=tgt_cardinalities,
            ctx_cardinalities=ctx_cardinalities,
            ctxseq_len_median=ctx_seq_len_median,
            model_size=model_units,
            column_order=trn_column_order,
            device=device,
        )

    # Load trained weights
    if workspace.model_tabular_weights_path.exists():
        load_model_weights(
            model=model,
            path=workspace.model_tabular_weights_path,
            device=device,
        )
    else:
        warnings.warn("Model weights not found; scores will be from untrained model")

    model.to(device)
    model.eval()

    # Determine which positional columns are needed for sequential models
    has_slen = has_ridx = has_sdec = None
    if is_sequential:
        if isinstance(model_units, dict):
            has_slen = any(SLEN_SUB_COLUMN_PREFIX in k for k in model_units.keys())
            has_ridx = any(RIDX_SUB_COLUMN_PREFIX in k for k in model_units.keys())
            has_sdec = any(SDEC_SUB_COLUMN_PREFIX in k for k in model_units.keys())
        else:
            has_slen, has_ridx, has_sdec = DEFAULT_HAS_SLEN, DEFAULT_HAS_RIDX, DEFAULT_HAS_SDEC

    # For sequential target data, group by context key to create sequences
    if is_sequential and tgt_context_key and tgt_context_key in tgt_data.columns:
        # Group by context key and aggregate each column into lists
        tgt_data_grouped = tgt_data.groupby(tgt_context_key, sort=False).agg(list).reset_index()
        tgt_data_to_encode = tgt_data_grouped
    else:
        tgt_data_to_encode = tgt_data

    # Encode target data
    tgt_data_encoded, _, tgt_context_key_encoded = encode_df(
        df=tgt_data_to_encode, stats=tgt_stats, tgt_context_key=tgt_context_key
    )

    # For sequential target data, add positional columns (SIDX, SLEN, RIDX, SDEC)
    if is_sequential:
        # Get sequence lengths for each record (look at first data column to determine length)
        first_data_col = [col for col in tgt_data_encoded.columns if col != tgt_context_key_encoded][0]
        seq_lengths = tgt_data_encoded[first_data_col].apply(
            lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 1
        )
        seq_len_max = seq_len_stats["max"]

        # Generate SIDX (sequence index): [0, 1, 2, ..., seq_len-1]
        sidx_lists = []
        for seq_len in seq_lengths:
            sidx_encoded = encode_positional_column(
                pd.Series(range(seq_len)), max_seq_len=seq_len_max, prefix=SIDX_SUB_COLUMN_PREFIX
            )
            sidx_lists.append({col: sidx_encoded[col].tolist() for col in sidx_encoded.columns})
        for col in sidx_lists[0].keys():
            tgt_data_encoded[col] = [sidx[col] for sidx in sidx_lists]

        # Generate SLEN (sequence length): [seq_len-1, seq_len-1, ..., seq_len-1]
        # Note: SLEN is the last valid index (0-based), so it's seq_len - 1
        if has_slen:
            slen_lists = []
            for seq_len in seq_lengths:
                slen_encoded = encode_positional_column(
                    pd.Series([seq_len - 1] * seq_len), max_seq_len=seq_len_max, prefix=SLEN_SUB_COLUMN_PREFIX
                )
                slen_lists.append({col: slen_encoded[col].tolist() for col in slen_encoded.columns})
            for col in slen_lists[0].keys():
                tgt_data_encoded[col] = [slen[col] for slen in slen_lists]

        # Generate RIDX (reverse index): [seq_len-1, seq_len-2, ..., 1, 0]
        if has_ridx:
            ridx_lists = []
            for seq_len in seq_lengths:
                ridx_encoded = encode_positional_column(
                    pd.Series(range(seq_len - 1, -1, -1)), max_seq_len=seq_len_max, prefix=RIDX_SUB_COLUMN_PREFIX
                )
                ridx_lists.append({col: ridx_encoded[col].tolist() for col in ridx_encoded.columns})
            for col in ridx_lists[0].keys():
                tgt_data_encoded[col] = [ridx[col] for ridx in ridx_lists]

        # Generate SDEC (sequence index decile): [0, 0, ..., 9]
        if has_sdec:
            sdec_col = f"{SDEC_SUB_COLUMN_PREFIX}cat"
            tgt_data_encoded[sdec_col] = [
                [min(9, 10 * i // seq_len) for i in range(seq_len)] for seq_len in seq_lengths
            ]

    # Encode context data if provided
    if ctx_data is not None:
        ctx_data_encoded, ctx_primary_key_encoded, _ = encode_df(
            df=ctx_data, stats=ctx_stats, ctx_primary_key=ctx_primary_key
        )
    else:
        ctx_data_encoded = pd.DataFrame()
        ctx_primary_key_encoded = None

    # Validate alignment between tgt_data and ctx_data
    if ctx_data is not None:
        if not is_sequential and len(tgt_data_encoded) != len(ctx_data_encoded):
            raise ValueError(
                f"For flat models, tgt_data and ctx_data must have the same number of rows. "
                f"Got {len(tgt_data_encoded)} target rows and {len(ctx_data_encoded)} context rows."
            )
        if is_sequential and tgt_context_key and ctx_primary_key:
            # Check that all tgt_context_keys exist in ctx_primary_keys
            tgt_keys = set(tgt_data[tgt_context_key].unique()) if tgt_context_key in tgt_data.columns else set()
            ctx_keys = set(ctx_data[ctx_primary_key].unique()) if ctx_primary_key in ctx_data.columns else set()
            missing_keys = tgt_keys - ctx_keys
            if missing_keys:
                raise ValueError(
                    f"Target data references context keys that are not present in ctx_data: "
                    f"{list(missing_keys)[:5]}{'...' if len(missing_keys) > 5 else ''}"
                )

    # Convert to tensors and move to device
    data_dict = {}

    # Filter out key columns from target data
    tgt_data_cols = [col for col in tgt_data_encoded.columns if col != tgt_context_key_encoded]

    # Process target columns
    for col in tgt_data_cols:
        values = tgt_data_encoded[col].values
        # Check if values are lists (sequential data) by looking at the first non-null value
        if len(values) > 0 and isinstance(values[0], (list, np.ndarray)):
            # Sequential target data - pad to longest sequence in batch
            tensor = torch.tensor(
                np.array(list(zip_longest(*values, fillvalue=0))).T,
                dtype=torch.int64,
                device=device,
            ).unsqueeze(-1)
        else:
            # Flat target data or scalar values
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
            if col.startswith(CTXSEQ):
                # Sequential context columns - convert to nested tensor
                tensor = torch.nested.as_nested_tensor(
                    [torch.tensor(row, dtype=torch.int64, device=device) for row in values],
                    dtype=torch.int64,
                    device=device,
                ).unsqueeze(-1)
            elif col.startswith(CTXFLT):
                # Flat context columns
                if values.dtype == object:
                    values = pd.to_numeric(values, errors="coerce").fillna(0).astype(np.int64)
                else:
                    values = values.astype(np.int64)
                tensor = torch.from_numpy(values).to(device).unsqueeze(-1)
            else:
                # Handle any other columns (shouldn't happen, but be defensive)
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
