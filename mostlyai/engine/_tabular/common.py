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

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import torch

from mostlyai.engine._common import (
    DEFAULT_HAS_RIDX,
    DEFAULT_HAS_SDEC,
    DEFAULT_HAS_SLEN,
    RIDX_SUB_COLUMN_PREFIX,
    SDEC_SUB_COLUMN_PREFIX,
    SLEN_SUB_COLUMN_PREFIX,
    get_cardinalities,
    get_columns_from_cardinalities,
    get_sequence_length_stats,
)
from mostlyai.engine._tabular.argn import FlatModel, SequentialModel
from mostlyai.engine._workspace import Workspace, ensure_workspace_dir

_LOG = logging.getLogger(__name__)

DPLSTM_SUFFIXES: tuple = ("ih.weight", "ih.bias", "hh.weight", "hh.bias")


def load_model_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    t0 = time.time()
    incompatible_keys = model.load_state_dict(torch.load(f=path, map_location=device, weights_only=True), strict=False)
    missing_keys = incompatible_keys.missing_keys
    unexpected_keys = incompatible_keys.unexpected_keys
    # for DP-trained models, we expect extra keys from the DPLSTM layers (which is fine to ignore because we use standard LSTM layers during generation)
    # but if there're any other missing or unexpected keys, an error should be raised
    if len(missing_keys) > 0 or any(not k.endswith(DPLSTM_SUFFIXES) for k in unexpected_keys):
        raise RuntimeError(
            f"failed to load model weights due to incompatibility: {missing_keys = }, {unexpected_keys = }"
        )
    _LOG.info(f"loaded model weights in {time.time() - t0:.2f}s")


def load_model(
    workspace_dir: str | Path,
    device: torch.device,
    column_order: list[str] | None = None,
) -> tuple[FlatModel | SequentialModel, dict]:
    """
    Load a trained tabular model from workspace.

    Args:
        workspace_dir: Directory path for workspace containing trained model.
        device: Device to load model on ('cuda' or 'cpu').
        column_order: Optional column order for model. If None, uses training order.

    Returns:
        Tuple of (model, metadata_dict) where metadata_dict contains:
            - tgt_stats: Target data statistics
            - ctx_stats: Context data statistics
            - model_configs: Model configuration
            - tgt_cardinalities: Target cardinalities
            - ctx_cardinalities: Context cardinalities
            - is_sequential: Whether model is sequential
            - seq_len_stats: Sequence length statistics
            - has_context: Whether model has context data
            - has_slen, has_ridx, has_sdec: Positional column flags (if sequential)
    """
    _LOG.info("Loading tabular model from workspace")
    workspace = Workspace(ensure_workspace_dir(workspace_dir))

    # Check if all required files exist before proceeding
    if not workspace.tgt_stats.path.exists() or not workspace.model_configs.path.exists():
        raise ValueError("Model statistics or config missing. Train the model first.")

    # Load stats and config
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

    # Get cardinalities
    has_slen = has_ridx = has_sdec = None
    if is_sequential:
        model_units = model_configs.get("model_units")
        if isinstance(model_units, dict):
            has_slen = any(SLEN_SUB_COLUMN_PREFIX in k for k in model_units.keys())
            has_ridx = any(RIDX_SUB_COLUMN_PREFIX in k for k in model_units.keys())
            has_sdec = any(SDEC_SUB_COLUMN_PREFIX in k for k in model_units.keys())
        else:
            has_slen, has_ridx, has_sdec = DEFAULT_HAS_SLEN, DEFAULT_HAS_RIDX, DEFAULT_HAS_SDEC

    tgt_cardinalities = get_cardinalities(tgt_stats, has_slen, has_ridx, has_sdec)
    ctx_cardinalities = get_cardinalities(ctx_stats)

    # Get column order from configs (for backwards compatibility)
    if column_order is None:
        trn_column_order = model_configs.get("trn_column_order")
        if trn_column_order is None and not model_configs.get("enable_flexible_generation", True):
            # Fixed column order based on cardinalities
            trn_column_order = get_columns_from_cardinalities(tgt_cardinalities)
        column_order = trn_column_order

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
            column_order=column_order,
            device=device,
        )
    else:
        ctx_seq_len_median = model_configs.get("ctx_seq_len_median", {})

        model = FlatModel(
            tgt_cardinalities=tgt_cardinalities,
            ctx_cardinalities=ctx_cardinalities,
            ctxseq_len_median=ctx_seq_len_median,
            model_size=model_units,
            column_order=column_order,
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
        warnings.warn("Model weights not found; using untrained model")

    model.to(device)
    model.eval()

    # Prepare metadata dict
    metadata = {
        "tgt_stats": tgt_stats,
        "ctx_stats": ctx_stats,
        "model_configs": model_configs,
        "tgt_cardinalities": tgt_cardinalities,
        "ctx_cardinalities": ctx_cardinalities,
        "is_sequential": is_sequential,
        "seq_len_stats": seq_len_stats,
        "has_context": has_context,
    }

    if is_sequential:
        metadata["has_slen"] = has_slen
        metadata["has_ridx"] = has_ridx
        metadata["has_sdec"] = has_sdec

    return model, metadata
