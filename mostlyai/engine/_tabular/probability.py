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
Probability computation for trained models.

This module provides functions for computing probability distributions
from trained generative models, including support for single and multi-target
joint probability distributions.
"""

import itertools
import logging

import numpy as np
import pandas as pd
import torch

from mostlyai.engine._common import (
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
    get_argn_name,
    get_cardinalities,
    get_columns_from_cardinalities,
)
from mostlyai.engine._encoding_types.tabular.numeric import (
    NUMERIC_BINNED_MAX_TOKEN,
    NUMERIC_BINNED_MIN_TOKEN,
    NUMERIC_BINNED_UNKNOWN_TOKEN,
)
from mostlyai.engine._tabular.argn import ModelSize
from mostlyai.engine._tabular.common import (
    check_column_order,
    create_and_load_model,
    fix_rare_token_probs,
    get_argn_column_names,
    load_model_artifacts,
    prepare_context_inputs,
    resolve_device,
    translate_fixed_probs,
)
from mostlyai.engine._tabular.encoding import encode_df
from mostlyai.engine._tabular.training import _calculate_sample_losses
from mostlyai.engine._workspace import Workspace
from mostlyai.engine.domain import ModelEncodingType, RareCategoryReplacementMethod

_LOG = logging.getLogger(__name__)


def _initialize_model(
    *,
    workspace: Workspace,
    rare_category_replacement_method: RareCategoryReplacementMethod | str | None = None,
    device: torch.device | str | None = None,
    allow_sequential: bool = True,
) -> tuple[torch.nn.Module, dict, dict, dict, list[str], dict, torch.device, bool]:
    """
    Initialize model and artifacts for probability computation.

    Args:
        workspace: Workspace containing model and stats
        rare_category_replacement_method: How to handle rare categories (None to skip fixed_probs)
        device: Device for computation
        allow_sequential: Whether to allow sequential models

    Returns:
        Tuple of (model, tgt_stats, ctx_stats, tgt_cardinalities, all_columns, fixed_probs, device, enable_flexible_generation)

    Raises:
        ValueError: If model is sequential and allow_sequential is False
    """
    # Load model artifacts
    model_config, tgt_stats, ctx_stats, is_sequential = load_model_artifacts(workspace)

    # Check model type
    if is_sequential and not allow_sequential:
        raise ValueError("Sequential models are not supported for this operation")

    # Get cardinalities and config
    tgt_cardinalities = get_cardinalities(tgt_stats)
    ctx_cardinalities = get_cardinalities(ctx_stats)
    enable_flexible_generation = model_config.get("enable_flexible_generation", True)

    # Resolve device
    device = resolve_device(device)
    _LOG.info(f"Using device: {device}")

    # Get all columns in training order
    all_columns = get_columns_from_cardinalities(tgt_cardinalities)

    # Create model
    model_units = model_config.get("model_units") or ModelSize.M
    ctx_seq_len_median = ctx_stats.get("sequence_len_median")

    model = create_and_load_model(
        workspace=workspace,
        is_sequential=is_sequential,
        tgt_cardinalities=tgt_cardinalities,
        ctx_cardinalities=ctx_cardinalities,
        model_units=model_units,
        ctx_seq_len_median=ctx_seq_len_median,
        column_order=all_columns,
        device=device,
    )

    # Prepare fixed_probs to suppress rare tokens (if replacement method provided)
    if rare_category_replacement_method is not None:
        _LOG.info(f"{rare_category_replacement_method=}")
        rare_token_fixed_probs = fix_rare_token_probs(tgt_stats, rare_category_replacement_method)
        fixed_probs = translate_fixed_probs(
            fixed_probs=rare_token_fixed_probs,
            stats=tgt_stats,
        )
    else:
        fixed_probs = {}

    return model, tgt_stats, ctx_stats, tgt_cardinalities, all_columns, fixed_probs, device, enable_flexible_generation


def _get_column_metadata(target_column: str, target_stats: dict) -> list[dict]:
    """
    Get metadata for each column (code value and label) in display order.

    For binned columns: filters <<UNK>>, reorders MIN/MAX, creates bin labels.

    Args:
        target_column: Name of target column
        target_stats: Target column statistics

    Returns:
        List of dicts with 'code_value' and 'label' keys, in display order
    """
    encoding_type = ModelEncodingType(target_stats["encoding_type"])
    codes = target_stats["codes"]

    if encoding_type in (
        ModelEncodingType.tabular_categorical,
        ModelEncodingType.tabular_numeric_discrete,
    ):
        # Keep all codes as-is
        return [{"code_value": code_value, "label": code_name} for code_name, code_value in codes.items()]

    elif encoding_type == ModelEncodingType.tabular_numeric_binned:
        bins = target_stats["bins"]
        metadata = []

        min_code_value = None
        max_code_value = None

        # First: special tokens (except <<UNK>>, <<MIN>>, <<MAX>>)
        for code_name, code_value in codes.items():
            if code_name == NUMERIC_BINNED_UNKNOWN_TOKEN:
                continue  # Skip <<UNK>>
            elif code_name == NUMERIC_BINNED_MIN_TOKEN:
                min_code_value = code_value  # Save for later
            elif code_name == NUMERIC_BINNED_MAX_TOKEN:
                max_code_value = code_value  # Save for later
            else:
                metadata.append({"code_value": code_value, "label": code_name})

        # Add MIN value if present
        if min_code_value is not None:
            metadata.append({"code_value": min_code_value, "label": str(bins[0])})

        # Add bin range labels
        codes_bin_offset = len(codes)
        for i in range(len(bins) - 1):
            lower_bound = bins[i]
            upper_bound = bins[i + 1]
            label = f">={lower_bound}" if i == len(bins) - 2 else f"<{upper_bound}"
            metadata.append({"code_value": i + codes_bin_offset, "label": label})

        # Add MAX value at the end if present
        if max_code_value is not None:
            metadata.append({"code_value": max_code_value, "label": str(bins[-1])})

        return metadata

    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")


def _create_probability_dataframe(
    probs_array: np.ndarray,
    target_column: str,
    target_stats: dict,
) -> pd.DataFrame:
    """
    Create probability DataFrame from numpy array with filtered and formatted columns.

    For binned columns:
    - Filters out <<UNK>> token
    - Replaces <<MIN>>/<<MAX>> tokens with actual bin boundary values
    - Reorders columns: special tokens, MIN, bins, MAX

    Args:
        probs_array: Probability array of shape (n_samples, cardinality)
        target_column: Name of target column
        target_stats: Target column statistics

    Returns:
        DataFrame with filtered and labeled probability columns
    """
    metadata = _get_column_metadata(target_column, target_stats)
    selected_cols = [item["code_value"] for item in metadata]
    labels = [item["label"] for item in metadata]
    selected_probs = probs_array[:, selected_cols]
    return pd.DataFrame(selected_probs, columns=labels)


def _get_possible_values(target_column: str, tgt_stats: dict) -> list[int]:
    """
    Get all possible encoded values for a target column.

    Returns values in the same order as DataFrame columns (filters out <<UNK>> for binned).

    Args:
        target_column: Name of the target column
        tgt_stats: Target statistics dictionary

    Returns:
        List of integer codes matching DataFrame column order
    """
    target_stats = tgt_stats["columns"][target_column]
    metadata = _get_column_metadata(target_column, target_stats)
    return [item["code_value"] for item in metadata]


def _generate_marginal_probs(
    model,
    seed_encoded: pd.DataFrame,
    target_column: str,
    tgt_stats: dict,
    seed_columns: list[str],
    device: torch.device,
    n_samples: int,
    ctx_data: pd.DataFrame | None = None,
    ctx_stats: dict | None = None,
    fixed_probs: dict | None = None,
) -> pd.DataFrame:
    """
    Generate P(target | seed_features, context).

    Args:
        model: The generative model
        seed_encoded: Encoded seed features (may include previous targets)
        target_column: Target column to predict (original column name)
        tgt_stats: Target statistics
        seed_columns: Seed column names in original format, in correct order
        device: Device for computation
        n_samples: Number of samples to generate probabilities for
        ctx_data: Optional context data
        ctx_stats: Optional context statistics (required if ctx_data provided)
        fixed_probs: Optional fixed probabilities for rare token handling

    Returns:
        DataFrame of shape (n_samples, cardinality) with probabilities and column names
    """
    target_stats = tgt_stats["columns"][target_column]

    # Build fixed_values dict from seed_encoded
    seed_batch_dict = {}

    # Add all columns from seed_encoded
    for sub_col in seed_encoded.columns:
        seed_batch_dict[sub_col] = torch.tensor(seed_encoded[sub_col].values, dtype=torch.long, device=device)

    # Get target sub-columns
    target_sub_cols = [
        get_argn_name(
            argn_processor=target_stats[ARGN_PROCESSOR],
            argn_table=target_stats[ARGN_TABLE],
            argn_column=target_stats[ARGN_COLUMN],
            argn_sub_column=sub_col,
        )
        for sub_col in target_stats["cardinalities"].keys()
    ]

    # Convert column names to ARGN format for model
    seed_columns_argn = get_argn_column_names(tgt_stats["columns"], seed_columns)
    target_argn_name = get_argn_column_names(tgt_stats["columns"], [target_column])[0]

    # Determine column order: seed columns + target (in training order)
    gen_column_order = seed_columns_argn + [target_argn_name]

    # Prepare context inputs if provided
    if ctx_data is not None and ctx_stats is not None:
        x, _, _ = prepare_context_inputs(ctx_data=ctx_data, ctx_stats=ctx_stats, device=device)
    else:
        x = torch.zeros((n_samples, 0), dtype=torch.long, device=device)

    # Forward pass
    _, probs_dct = model(
        x,
        mode="probs",
        batch_size=n_samples,
        fixed_probs=fixed_probs or {},
        fixed_values=seed_batch_dict,
        column_order=gen_column_order,
    )

    probs_array = probs_dct[target_sub_cols[0]].cpu().numpy()

    # Create DataFrame with filtered and formatted columns
    return _create_probability_dataframe(probs_array, target_column, target_stats)


@torch.no_grad()
def predict_proba(
    *,
    workspace: Workspace,
    seed_data: pd.DataFrame,
    target_columns: list[str],
    ctx_data: pd.DataFrame | None = None,
    rare_category_replacement_method: RareCategoryReplacementMethod | str = RareCategoryReplacementMethod.constant,
    device: torch.device | str | None = None,
) -> pd.DataFrame:
    """
    Compute probability distributions for target column(s).

    This function generates raw model probabilities (no temperature/top_p transformations)
    for one or more target columns conditioned on seed_data (and optionally ctx_data).
    Returns results in-memory without workspace I/O.

    Args:
        workspace: Workspace object containing model and stats
        seed_data: Feature columns to condition on (fixed_values) - should NOT include target
        target_columns: List of target column names (single or multiple) to predict probabilities for
        ctx_data: Optional separate context data (for models with context)
        device: Device to run inference on ('cuda' or 'cpu'). Defaults to 'cuda' if available.

    Returns:
        - Single target: DataFrame with shape (n_samples, n_categories) with exploded columns
          Column names: category/bin/value labels (e.g., "male", "female" or "<10000", ">=10000")
        - Multiple targets: MultiIndex DataFrame with joint probabilities
          P(col1, col2, ...) = P(col1) * P(col2|col1) * P(col3|col1,col2) * ...
          Columns: MultiIndex.from_product of all category combinations

    Raises:
        ValueError: If target column not found, unsupported encoding type, or sequential model
    """
    _LOG.info(f"PREDICT_PROBA started for targets: {target_columns}")

    # Initialize model
    model, tgt_stats, ctx_stats, tgt_cardinalities, all_columns, fixed_probs, device, enable_flexible_generation = (
        _initialize_model(
            workspace=workspace,
            rare_category_replacement_method=rare_category_replacement_method,
            device=device,
            allow_sequential=False,
        )
    )

    # Get seed column names (needed for _generate_marginal_probs)
    seed_columns = list(seed_data.columns)

    # Note: Column order validation is now handled in the interface layer (TabularARGN.predict_proba)
    # to centralize validation logic with sample() method

    # Encode seed data (features to condition on) - common for both single and multi-target
    # seed_data should NOT include any target columns
    seed_encoded, _, _ = encode_df(
        df=seed_data,
        stats=tgt_stats,
    )
    n_samples = len(seed_data)

    _LOG.info(f"Computing joint probabilities for {len(target_columns)} targets")

    # Warn about exponential complexity for multiple targets
    if len(target_columns) > 1:
        # Compute total number of probability values
        total_cardinality = 1
        for target_col in target_columns:
            col_stats = tgt_stats["columns"][target_col]
            target_cardinality = sum(col_stats["cardinalities"].values())
            total_cardinality *= target_cardinality

        if total_cardinality > 100:
            _LOG.warning(
                f"Computing joint probabilities for {len(target_columns)} targets "
                f"results in {total_cardinality:,} total probability values per sample. "
                f"Computation complexity grows exponentially with the number of targets. "
                f"Consider computing probabilities for targets separately if this takes too long."
            )

    # Initialize with first target: P(col1)
    first_target_df = _generate_marginal_probs(
        model=model,
        seed_encoded=seed_encoded,
        target_column=target_columns[0],
        tgt_stats=tgt_stats,
        seed_columns=seed_columns,
        device=device,
        n_samples=n_samples,
        ctx_data=ctx_data,
        ctx_stats=ctx_stats,
        fixed_probs=fixed_probs,
    )
    _LOG.info(f"Generated P({target_columns[0]}) with shape {first_target_df.shape}")

    # For single target, return the DataFrame directly
    if len(target_columns) == 1:
        _LOG.info(f"PREDICT_PROBA finished: returned probabilities for {len(first_target_df)} samples")
        return first_target_df

    # For multi-target: extract values and column names for joint probability computation
    joint_probs = first_target_df.values
    all_possible_values = [_get_possible_values(target_columns[0], tgt_stats)]
    all_column_names = [list(first_target_df.columns)]

    # Create extended_seed once - will grow incrementally as we process each target
    extended_seed = seed_encoded.copy()

    # Iteratively add each subsequent target
    for target_idx in range(1, len(target_columns)):
        target_col = target_columns[target_idx]
        _LOG.info(f"Processing target {target_idx + 1}/{len(target_columns)}: {target_col}")

        # Get possible values for current target
        current_possible_values = _get_possible_values(target_col, tgt_stats)
        current_card = len(current_possible_values)

        # Get all combinations of previous targets
        # prev_combos has shape matching the columns in joint_probs
        prev_combos = list(itertools.product(*all_possible_values))
        num_prev_combos = len(prev_combos)

        _LOG.info(
            f"Computing P({target_col}|previous) for {num_prev_combos} combinations "
            f"× {current_card} values = {num_prev_combos * current_card} total"
        )

        # Allocate array for new joint probabilities
        new_joint_probs = np.zeros((n_samples, num_prev_combos * current_card))

        # Build DataFrames for each combo with actual values, then concatenate
        combo_dfs = []
        for combo_idx, prev_combo in enumerate(prev_combos):
            # Build data dict starting with columns from extended_seed
            data = {col: extended_seed[col].values for col in extended_seed.columns}

            # Add previous target columns with actual values
            for i in range(target_idx):
                prev_target_col = target_columns[i]
                encoded_val = prev_combo[i]
                prev_target_stats = tgt_stats["columns"][prev_target_col]
                for sub_col_key in prev_target_stats["cardinalities"].keys():
                    full_sub_col_name = get_argn_name(
                        argn_processor=prev_target_stats[ARGN_PROCESSOR],
                        argn_table=prev_target_stats[ARGN_TABLE],
                        argn_column=prev_target_stats[ARGN_COLUMN],
                        argn_sub_column=sub_col_key,
                    )
                    data[full_sub_col_name] = encoded_val

            # Create DataFrame with explicit row count
            df = pd.DataFrame(data, index=range(n_samples))
            combo_dfs.append(df)

        # Concatenate all combo DataFrames into single batch
        batched_seed = pd.concat(combo_dfs, ignore_index=True)
        # batched_seed shape: (n_samples * num_combos, features)

        # Replicate ctx_data to match batch size if provided
        batched_ctx_data = None
        if ctx_data is not None:
            batched_ctx_data = pd.concat([ctx_data] * num_prev_combos, ignore_index=True)

        # Compute extended seed_columns including previous targets
        extended_seed_columns = seed_columns + target_columns[:target_idx]

        # Single batched forward pass for all combinations
        all_conditional_df = _generate_marginal_probs(
            model=model,
            seed_encoded=batched_seed,
            target_column=target_col,
            tgt_stats=tgt_stats,
            seed_columns=extended_seed_columns,
            device=device,
            n_samples=n_samples * num_prev_combos,
            ctx_data=batched_ctx_data,
            ctx_stats=ctx_stats,
            fixed_probs=fixed_probs,
        )  # DataFrame: (n_samples * num_combos, current_card)

        # Extract probabilities for each combo and compute joint probabilities
        all_conditional_probs = all_conditional_df.values
        for combo_idx in range(num_prev_combos):
            start_idx = combo_idx * n_samples
            end_idx = start_idx + n_samples
            conditional_probs = all_conditional_probs[start_idx:end_idx, :]  # (n_samples, current_card)

            # Multiply: P(prev_combo) * P(current | prev_combo)
            for val_idx in range(current_card):
                new_col_idx = combo_idx * current_card + val_idx
                new_joint_probs[:, new_col_idx] = joint_probs[:, combo_idx] * conditional_probs[:, val_idx]

        # Update for next iteration
        joint_probs = new_joint_probs
        all_possible_values.append(current_possible_values)
        all_column_names.append(list(all_conditional_df.columns))

        _LOG.info(f"Joint probabilities now have shape {joint_probs.shape}")

    # Create MultiIndex DataFrame for multi-target results
    multi_index = pd.MultiIndex.from_product(all_column_names, names=target_columns)
    probs_df = pd.DataFrame(joint_probs, columns=multi_index)

    _LOG.info(f"PREDICT_PROBA finished: returned probabilities for {len(probs_df)} samples")
    return probs_df


@torch.no_grad()
def log_prob(
    *,
    workspace: Workspace,
    data: pd.DataFrame,
    ctx_data: pd.DataFrame | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """
    Compute log probability of full observations.

    This function computes P(observation | model) - the likelihood of the observation
    under the trained model. For autoregressive models:

    log P(x1, x2, ..., xn) = log P(x1) + log P(x2|x1) + ... + log P(xn|x1,...,xn-1)

    Each term is the probability the model assigns to the actual observed value.

    Supports all encoding types including multi-sub-column encodings like numeric_digit,
    and both flat and sequential models.

    Args:
        workspace: Workspace object containing model and stats
        data: DataFrame with ALL columns containing observed values
        ctx_data: Optional context data (for models with context)
        device: Device to run inference on ('cuda' or 'cpu'). Defaults to 'cuda' if available.

    Returns:
        np.ndarray of shape (n_samples,) with log probability per row.
        Values are <= 0 (log probabilities).
    """
    _LOG.info("LOG_PROB started")

    # Initialize model
    model, tgt_stats, ctx_stats, _, all_columns, _, device, enable_flexible_generation = _initialize_model(
        workspace=workspace,
        device=device,
    )

    # Check column order of input data when flexible generation is disabled
    if not enable_flexible_generation:
        check_column_order(list(data.columns), all_columns)

    # Encode full data to get observed codes for all columns
    full_encoded, _, _ = encode_df(df=data, stats=tgt_stats)

    n_samples = len(data)
    _LOG.info(f"Computing log probabilities for {n_samples} samples")

    # Build batch dict with ALL encoded values as tensors
    batch_dict: dict[str, torch.Tensor] = {}
    for col in full_encoded.columns:
        batch_dict[col] = torch.tensor(full_encoded[col].values, dtype=torch.long, device=device).unsqueeze(-1)

    # Add context data if provided
    if ctx_data is not None and ctx_stats:
        ctx_inputs, _, _ = prepare_context_inputs(ctx_data=ctx_data, ctx_stats=ctx_stats, device=device)
        batch_dict.update(ctx_inputs)

    # Use the training loss calculation directly - handles both flat and sequential models
    losses = _calculate_sample_losses(model, batch_dict)

    # Negate loss to get log probability (loss = -log_prob)
    log_probs = -losses.cpu().numpy()

    _LOG.info(f"LOG_PROB finished: computed log probs for {n_samples} samples")
    return log_probs
