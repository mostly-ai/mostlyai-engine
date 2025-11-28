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
import time
from pathlib import Path

import pandas as pd
import torch

from mostlyai.engine._common import (
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
    CTXFLT,
    CTXSEQ,
    get_argn_name,
)
from mostlyai.engine._encoding_types.tabular.categorical import (
    CATEGORICAL_SUB_COL_SUFFIX,
    CATEGORICAL_UNKNOWN_TOKEN,
)
from mostlyai.engine._encoding_types.tabular.numeric import (
    NUMERIC_BINNED_SUB_COL_SUFFIX,
    NUMERIC_BINNED_UNKNOWN_TOKEN,
    NUMERIC_DISCRETE_SUB_COL_SUFFIX,
    NUMERIC_DISCRETE_UNKNOWN_TOKEN,
)
from mostlyai.engine._tabular.encoding import encode_df, pad_ctx_sequences
from mostlyai.engine.domain import ModelEncodingType, RareCategoryReplacementMethod

_LOG = logging.getLogger(__name__)

# Type alias for fixed probabilities
CodeProbabilities = dict[int, float]

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


def load_model_artifacts(workspace):
    """
    Load model configurations and statistics from workspace.

    Returns:
        Tuple of (model_config, tgt_stats, ctx_stats, is_sequential)
    """
    model_config = workspace.model_configs.read()
    tgt_stats = workspace.tgt_stats.read()
    ctx_stats = workspace.ctx_stats.read()
    is_sequential = tgt_stats["is_sequential"]
    return model_config, tgt_stats, ctx_stats, is_sequential


def resolve_device(device: torch.device | str | None) -> torch.device:
    """
    Resolve device to use for inference.

    Args:
        device: Device specification ('cuda', 'cpu', or None for auto-detect)

    Returns:
        torch.device instance
    """
    if device is None:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device)


def create_and_load_model(
    workspace,
    is_sequential: bool,
    tgt_cardinalities: dict,
    ctx_cardinalities: dict,
    model_units,
    ctx_seq_len_median: int | None,
    column_order: list[str],
    device: torch.device,
    seq_len_median: int | None = None,
    seq_len_max: int | None = None,
):
    """
    Create model, load weights, and prepare for inference.

    Args:
        workspace: Workspace containing model weights
        is_sequential: Whether to create SequentialModel or FlatModel
        tgt_cardinalities: Target column cardinalities
        ctx_cardinalities: Context column cardinalities
        model_units: Model size configuration
        ctx_seq_len_median: Median context sequence length
        column_order: Order of columns for generation
        device: Device to load model on
        seq_len_median: Median sequence length (for sequential models)
        seq_len_max: Maximum sequence length (for sequential models)

    Returns:
        Initialized model ready for inference
    """
    from mostlyai.engine._tabular.argn import FlatModel, SequentialModel, get_no_of_model_parameters

    _LOG.info("Creating generative model")

    if is_sequential:
        model = SequentialModel(
            tgt_cardinalities=tgt_cardinalities,
            tgt_seq_len_median=seq_len_median,
            tgt_seq_len_max=seq_len_max,
            ctx_cardinalities=ctx_cardinalities,
            ctxseq_len_median=ctx_seq_len_median,
            model_size=model_units,
            column_order=column_order,
            device=device,
        )
    else:
        model = FlatModel(
            tgt_cardinalities=tgt_cardinalities,
            ctx_cardinalities=ctx_cardinalities,
            ctxseq_len_median=ctx_seq_len_median,
            model_size=model_units,
            column_order=column_order,
            device=device,
        )

    no_of_model_params = get_no_of_model_parameters(model)
    _LOG.info(f"{no_of_model_params=}")

    if workspace.model_tabular_weights_path.exists():
        load_model_weights(
            model=model,
            path=workspace.model_tabular_weights_path,
            device=device,
        )
    else:
        _LOG.warning("Model weights not found; using untrained model")

    model.to(device)
    model.eval()

    return model


def prepare_context_inputs(
    ctx_data: pd.DataFrame,
    ctx_stats: dict,
    device: torch.device | str,
    ctx_primary_key: str | None = None,
) -> tuple[dict[str, torch.Tensor], pd.DataFrame, str | None]:
    """
    Encode context data and prepare tensor inputs for model forward pass.

    Handles both flat context (CTXFLT) and sequential context (CTXSEQ).

    Args:
        ctx_data: Context DataFrame to encode
        ctx_stats: Context statistics from training
        device: Device for tensor placement
        ctx_primary_key: Optional primary key column for context

    Returns:
        Tuple of (context_tensors, encoded_dataframe, encoded_primary_key):
        - context_tensors: Dict of CTXFLT/* and CTXSEQ/* tensors for model.context_compressor()
        - encoded_dataframe: Encoded context DataFrame (for extracting keys if needed)
        - encoded_primary_key: Name of encoded primary key column (None if not provided)
    """

    # Encode context data
    ctx_encoded, ctx_primary_key_encoded, _ = encode_df(df=ctx_data, stats=ctx_stats, ctx_primary_key=ctx_primary_key)

    # Pad empty sequences (required for model)
    ctx_encoded = pad_ctx_sequences(ctx_encoded)

    # Build flat context inputs (CTXFLT/*)
    ctxflt_inputs = {
        col: torch.unsqueeze(
            torch.as_tensor(ctx_encoded[col].to_numpy(), device=device).type(torch.int),
            dim=-1,
        )
        for col in ctx_encoded.columns
        if col.startswith(CTXFLT)
    }

    # Build sequential context inputs (CTXSEQ/*)
    ctxseq_inputs = {
        col: torch.unsqueeze(
            torch.nested.as_nested_tensor(
                [torch.as_tensor(t, device=device).type(torch.int) for t in ctx_encoded[col]],
                device=device,
            ),
            dim=-1,
        )
        for col in ctx_encoded.columns
        if col.startswith(CTXSEQ)
    }

    # Merge and return with encoded dataframe
    return (ctxflt_inputs | ctxseq_inputs), ctx_encoded, ctx_primary_key_encoded


def check_column_order(
    gen_column_order: list[str],
    trn_column_order: list[str],
) -> None:
    """
    Check if column order matches training order.

    Args:
        gen_column_order: Column order for the current operation
        trn_column_order: Column order from training

    Raises:
        ValueError: If column order doesn't match training order
    """
    if gen_column_order != trn_column_order:
        raise ValueError(
            "Column order does not match training order. "
            "A change in column order is only permitted for models that were trained with `enable_flexible_generation=True`."
        )


def fix_rare_token_probs(
    stats: dict,
    rare_category_replacement_method: RareCategoryReplacementMethod | None = None,
) -> dict[str, dict[str, CodeProbabilities]]:
    """
    Create fixed probabilities to suppress rare tokens.

    Args:
        stats: Target statistics dict
        rare_category_replacement_method: How to handle rare categories

    Returns:
        Dict of column -> sub_column -> code -> probability
    """
    # suppress rare token for categorical when no_of_rare_categories == 0
    mask = {
        col: {CATEGORICAL_SUB_COL_SUFFIX: {col_stats["codes"][CATEGORICAL_UNKNOWN_TOKEN]: 0.0}}
        for col, col_stats in stats["columns"].items()
        if col_stats["encoding_type"] == ModelEncodingType.tabular_categorical
        if "codes" in col_stats
        if col_stats.get("no_of_rare_categories", 0) == 0
    }
    # suppress rare token for categorical if RareCategoryReplacementMethod is sample
    if rare_category_replacement_method == RareCategoryReplacementMethod.sample:
        mask |= {
            col: {CATEGORICAL_SUB_COL_SUFFIX: {col_stats["codes"][CATEGORICAL_UNKNOWN_TOKEN]: 0.0}}
            for col, col_stats in stats["columns"].items()
            if col_stats["encoding_type"] == ModelEncodingType.tabular_categorical
            if "codes" in col_stats
        }
    # always suppress rare token for numeric_binned
    mask |= {
        col: {NUMERIC_BINNED_SUB_COL_SUFFIX: {col_stats["codes"][NUMERIC_BINNED_UNKNOWN_TOKEN]: 0.0}}
        for col, col_stats in stats["columns"].items()
        if col_stats["encoding_type"] == ModelEncodingType.tabular_numeric_binned
        if "codes" in col_stats
    }
    # always suppress rare token for numeric_discrete
    mask |= {
        col: {NUMERIC_DISCRETE_SUB_COL_SUFFIX: {col_stats["codes"][NUMERIC_DISCRETE_UNKNOWN_TOKEN]: 0.0}}
        for col, col_stats in stats["columns"].items()
        if col_stats["encoding_type"] == ModelEncodingType.tabular_numeric_discrete
        if "codes" in col_stats
    }
    return mask


def translate_fixed_probs(
    fixed_probs: dict[str, dict[str, CodeProbabilities]], stats: dict
) -> dict[str, CodeProbabilities]:
    """
    Translate fixed probs to ARGN naming conventions.

    Args:
        fixed_probs: Dict of column -> sub_column -> code -> probability
        stats: Target statistics dict

    Returns:
        Dict of ARGN sub_column name -> code -> probability
    """
    mask = {
        get_argn_name(
            argn_processor=stats["columns"][col][ARGN_PROCESSOR],
            argn_table=stats["columns"][col][ARGN_TABLE],
            argn_column=stats["columns"][col][ARGN_COLUMN],
            argn_sub_column=sub_col,
        ): sub_col_mask
        for col, col_mask in fixed_probs.items()
        for sub_col, sub_col_mask in col_mask.items()
    }
    return mask
