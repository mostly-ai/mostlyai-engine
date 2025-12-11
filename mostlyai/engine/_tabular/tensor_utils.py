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
Tensor preparation utilities for training optimization.

These functions convert encoded data directly to GPU tensors, replacing the
per-batch transformations in BatchCollator._convert_to_tensors.

The key optimization is moving tensor preparation out of the training loop,
allowing pre-computation or streaming from Ray Data pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from mostlyai.engine._common import CTXFLT, CTXSEQ, TGT, get_cardinalities, get_ctx_sequence_length, get_sequence_length_stats

if TYPE_CHECKING:
    from mostlyai.engine._tabular.training import ModelConfig
    from mostlyai.engine._workspace import Workspace


def prepare_flat_batch(
    batch: dict[str, np.ndarray | list],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Convert a batch of encoded data to tensors for flat (non-sequential) training.

    This replaces BatchCollator._convert_to_tensors for the non-sequential case.
    All values are encoded integers that get converted to int64 tensors with
    shape (batch_size, 1) for embedding lookup.

    Args:
        batch: Dict with column names as keys and encoded integer arrays as values.
               Column names should follow naming convention:
               - "tgt:..." for target columns
               - "ctxflt/..." for flat context columns
        device: Target device for tensors ("cpu", "cuda", or torch.device)

    Returns:
        Dict of tensors ready for model.forward()

    Example:
        >>> batch = {"tgt:col1": [1, 2, 3], "ctxflt/col2": [4, 5, 6]}
        >>> tensors = prepare_flat_batch(batch, device="cuda")
        >>> tensors["tgt:col1"].shape
        torch.Size([3, 1])
    """
    if isinstance(device, str):
        device = torch.device(device)

    tensors = {}
    for col, values in batch.items():
        if col.startswith(TGT) or col.startswith(CTXFLT):
            arr = np.asarray(values, dtype=np.int64)
            tensors[col] = torch.tensor(arr, dtype=torch.int64, device=device).unsqueeze(-1)

    return tensors


def prepare_sequential_batch(
    batch: dict[str, list[list[int]] | np.ndarray | list],
    max_seq_len: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Convert a batch of sequential encoded data to padded tensors.

    This replaces BatchCollator._convert_to_tensors for the sequential case.
    Target columns contain variable-length sequences that get padded to uniform
    length. Context columns (ctxflt) remain flat.

    Args:
        batch: Dict with column names as keys:
               - "tgt:..." columns contain lists of variable-length sequences
               - "ctxflt/..." columns contain flat integer values
               - "ctxseq/..." columns contain nested sequences (variable-length context)
        max_seq_len: Maximum sequence length for padding. If None, uses the
                    longest sequence in the batch.
        device: Target device for tensors

    Returns:
        Dict of tensors ready for model.forward():
        - Target tensors have shape (batch_size, seq_len, 1)
        - Flat context tensors have shape (batch_size, 1)
        - Sequential context uses nested tensors

    Example:
        >>> batch = {
        ...     "tgt:col1": [[1, 2, 3], [4, 5]],
        ...     "ctxflt/col2": [10, 20]
        ... }
        >>> tensors = prepare_sequential_batch(batch, device="cuda")
        >>> tensors["tgt:col1"].shape  # Padded to longest
        torch.Size([2, 3, 1])
    """
    if isinstance(device, str):
        device = torch.device(device)

    tensors = {}
    for col, values in batch.items():
        if col.startswith(TGT):
            # Target columns: variable-length sequences, need padding
            sequences = values
            seq_lens = [len(s) for s in sequences]
            pad_len = max_seq_len if max_seq_len is not None else max(seq_lens)

            # Pre-allocate padded array (more efficient than zip_longest)
            padded = np.zeros((len(sequences), pad_len), dtype=np.int64)
            for i, seq in enumerate(sequences):
                seq_len = min(len(seq), pad_len)
                padded[i, :seq_len] = seq[:seq_len]

            tensors[col] = torch.tensor(padded, dtype=torch.int64, device=device).unsqueeze(-1)

        elif col.startswith(CTXFLT):
            # Flat context: same as non-sequential
            arr = np.asarray(values, dtype=np.int64)
            tensors[col] = torch.tensor(arr, dtype=torch.int64, device=device).unsqueeze(-1)

        elif col.startswith(CTXSEQ):
            # Sequential context: use nested tensors for variable-length
            row_tensors = [
                torch.tensor(seq, dtype=torch.int64, device=device)
                for seq in values
            ]
            nested = torch.nested.as_nested_tensor(
                row_tensors, dtype=torch.int64, device=device
            )
            tensors[col] = nested.unsqueeze(-1)

    return tensors


def slice_sequences(
    batch: dict[str, torch.Tensor],
    max_window: int,
    strategy: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Slice sequences to a maximum window size for sequential training.

    This is the vectorized replacement for BatchCollator._slice_sequences,
    applied to already-tensorized data.

    The slicing strategy matches BatchCollator behavior:
    - 30% of batches: start of sequence (learn sequence beginnings)
    - 10% of batches: end of sequence (learn sequence endings)
    - 60% of batches: random window position (learn middle patterns)

    Args:
        batch: Dict of tensors with target columns having shape (batch, seq_len, 1)
        max_window: Maximum sequence length after slicing
        strategy: Override slicing strategy. One of:
                 - None (default): use random strategy distribution
                 - "start": always take from start of sequence
                 - "end": always take from end of sequence
                 - "random": always take random window

    Returns:
        Dict of tensors with target sequences sliced to max_window length.
        Non-target columns are passed through unchanged.

    Note:
        The max_window is increased by 1 internally to account for the one-step
        padding used in sequence training, matching BatchCollator behavior.
    """
    # Match BatchCollator: add 1 to max_window for padding step
    max_window = max_window + 1

    # Find target columns
    tgt_cols = [k for k in batch.keys() if k.startswith(TGT)]
    if not tgt_cols:
        return batch

    # Get sequence length from first target column
    first_col = batch[tgt_cols[0]]  # Shape: (batch, seq_len, 1)
    batch_size, seq_len, _ = first_col.shape

    # No slicing needed if sequences fit within window
    if seq_len <= max_window:
        return batch

    # Determine strategy for this batch
    if strategy is None:
        flip = np.random.random()
        if flip < 0.3:
            strategy = "start"
        elif flip < 0.4:
            strategy = "end"
        else:
            strategy = "random"

    # Compute start indices based on strategy
    if strategy == "start":
        # Take from start of sequence
        starts = torch.zeros(batch_size, dtype=torch.long, device=first_col.device)
    elif strategy == "end":
        # Take from end of sequence
        starts = torch.full(
            (batch_size,), seq_len - max_window, dtype=torch.long, device=first_col.device
        )
    else:  # random
        # Random window position
        max_start = seq_len - max_window
        starts = torch.randint(
            0, max_start + 1, (batch_size,), dtype=torch.long, device=first_col.device
        )

    # Create index tensor for gather: (batch, window, 1)
    window_indices = torch.arange(max_window, device=first_col.device).unsqueeze(0)  # (1, window)
    indices = starts.unsqueeze(1) + window_indices  # (batch, window)
    indices = indices.unsqueeze(-1)  # (batch, window, 1)

    # Slice all target columns using gather
    result = {}
    for col, tensor in batch.items():
        if col.startswith(TGT):
            result[col] = torch.gather(tensor, dim=1, index=indices)
        else:
            result[col] = tensor

    return result


def slice_sequences_variable_length(
    batch: dict[str, torch.Tensor],
    seq_lens: torch.Tensor | np.ndarray | list[int],
    max_window: int,
    strategy: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Slice sequences with variable actual lengths (accounting for padding).

    This is closer to the original BatchCollator._slice_sequences behavior,
    which respects actual sequence lengths when computing valid window ranges.

    Args:
        batch: Dict of tensors with target columns having shape (batch, seq_len, 1)
        seq_lens: Actual sequence lengths per sample (before padding)
        max_window: Maximum sequence length after slicing
        strategy: Override slicing strategy (see slice_sequences)

    Returns:
        Dict of tensors with target sequences sliced to max_window length.
    """
    # Match BatchCollator: add 1 to max_window for padding step
    max_window = max_window + 1

    tgt_cols = [k for k in batch.keys() if k.startswith(TGT)]
    if not tgt_cols:
        return batch

    first_col = batch[tgt_cols[0]]
    batch_size, padded_len, _ = first_col.shape
    device = first_col.device

    if not isinstance(seq_lens, torch.Tensor):
        seq_lens = torch.tensor(seq_lens, dtype=torch.long, device=device)

    # No slicing needed if all sequences fit within window
    if padded_len <= max_window:
        return batch

    # Determine strategy
    if strategy is None:
        flip = np.random.random()
        if flip < 0.3:
            strategy = "start"
        elif flip < 0.4:
            strategy = "end"
        else:
            strategy = "random"

    # Compute per-sample window starts
    if strategy == "start":
        starts = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif strategy == "end":
        # End of actual sequence, not padded length
        starts = torch.clamp(seq_lens - max_window, min=0)
    else:  # random
        # Random window within actual sequence
        max_starts = torch.clamp(seq_lens - max_window, min=0)
        # For short sequences that fit, start is always 0
        starts = torch.where(
            seq_lens <= max_window,
            torch.zeros_like(max_starts),
            torch.randint_like(max_starts, 0, 1) * max_starts  # Placeholder for per-element randint
        )
        # Per-element random: need to loop or use a different approach
        # For efficiency, use vectorized uniform random
        rand = torch.rand(batch_size, device=device)
        starts = (rand * (max_starts.float() + 1)).long()
        starts = torch.clamp(starts, max=max_starts)

    # Compute end indices respecting actual sequence lengths
    ends = torch.minimum(starts + max_window, seq_lens)

    # Determine actual window size (may vary per sample)
    window_sizes = ends - starts
    actual_window = window_sizes.max().item()

    # Build gather indices: (batch, actual_window, 1)
    window_indices = torch.arange(actual_window, device=device).unsqueeze(0)  # (1, window)
    indices = starts.unsqueeze(1) + window_indices  # (batch, window)

    # Clamp indices to valid range (for samples with shorter windows)
    indices = torch.clamp(indices, max=padded_len - 1)
    indices = indices.unsqueeze(-1)  # (batch, window, 1)

    # Slice target columns
    result = {}
    for col, tensor in batch.items():
        if col.startswith(TGT):
            result[col] = torch.gather(tensor, dim=1, index=indices)
        else:
            result[col] = tensor

    return result


def build_model_config(
    tgt_stats: dict,
    ctx_stats: dict | None = None,
    empirical_probs: dict[str, list[float]] | None = None,
) -> "ModelConfig":
    """
    Build a ModelConfig dict from analyze() statistics.

    This is the primary way to construct ModelConfig for tensor interface training
    when you have the stats dicts directly (e.g., from Ray Data pipelines).

    Args:
        tgt_stats: Target statistics dict from engine.analyze() / workspace.tgt_stats.read()
        ctx_stats: Optional context statistics dict (for models with context)
        empirical_probs: Optional pre-computed empirical probabilities for predictor
                        initialization. If not provided, training will skip the
                        optimization (slight convergence penalty but still works).

    Returns:
        ModelConfig dict ready for train() function

    Example:
        >>> # After engine.analyze()
        >>> workspace = Workspace(workspace_dir)
        >>> tgt_stats = workspace.tgt_stats.read()
        >>> config = build_model_config(tgt_stats)
        >>> train(workspace_dir, train_tensors=tensors, model_config=config, ...)

        >>> # With Ray Data (stats collected during distributed processing)
        >>> config = build_model_config(tgt_stats, ctx_stats)
        >>> train(workspace_dir, train_tensors=ray_tensor_iter, model_config=config, ...)
    """
    from mostlyai.engine._tabular.training import ModelConfig

    ctx_stats = ctx_stats or {}
    is_sequential = tgt_stats.get("is_sequential", False)

    config: ModelConfig = {
        "tgt_cardinalities": get_cardinalities(tgt_stats),
        "ctx_cardinalities": get_cardinalities(ctx_stats) if ctx_stats else {},
        "is_sequential": is_sequential,
        "trn_cnt": tgt_stats["no_of_training_records"],
        "val_cnt": tgt_stats["no_of_validation_records"],
    }

    if is_sequential:
        seq_stats = get_sequence_length_stats(tgt_stats)
        config["tgt_seq_len_median"] = seq_stats.get("median", 1)
        config["tgt_seq_len_max"] = seq_stats.get("max", 1)

    if ctx_stats and ctx_stats.get("is_sequential", False):
        # ctx_seq_len_median is a dict mapping table names to median lengths
        config["ctx_seq_len_median"] = get_ctx_sequence_length(ctx_stats, key="median")

    if empirical_probs is not None:
        config["empirical_probs"] = empirical_probs

    return config


def build_model_config_from_workspace(
    workspace_dir: Path | str,
    empirical_probs: dict[str, list[float]] | None = None,
) -> "ModelConfig":
    """
    Build a ModelConfig from a workspace directory after analyze().

    This is a convenience wrapper that reads stats from the workspace.
    Use this when you have a standard engine workspace.

    Args:
        workspace_dir: Path to workspace directory (after engine.analyze())
        empirical_probs: Optional pre-computed empirical probabilities

    Returns:
        ModelConfig dict ready for train() function

    Example:
        >>> engine.split(workspace_dir=ws, ...)
        >>> engine.analyze(workspace_dir=ws)
        >>> engine.encode(workspace_dir=ws)
        >>>
        >>> config = build_model_config_from_workspace(ws)
        >>> tensors = load_my_tensors(ws)  # Your tensor loading logic
        >>> train(ws, train_tensors=tensors, model_config=config, ...)
    """
    from mostlyai.engine._workspace import Workspace

    workspace = Workspace(Path(workspace_dir))
    tgt_stats = workspace.tgt_stats.read()
    ctx_stats = workspace.ctx_stats.read() if workspace.ctx_stats.path.exists() else None

    return build_model_config(tgt_stats, ctx_stats, empirical_probs)


def encode_batch(
    df: "pd.DataFrame",
    tgt_stats: dict,
    tgt_context_key: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Encode a DataFrame batch for use with Ray Data map_batches().

    This function encodes raw data using stats from analyze() and returns
    a dict of numpy arrays suitable for Ray Data's map_batches() return type.

    Use this with Ray Data to create a distributed encoding pipeline:
    - map_batches() with this function produces encoded numpy arrays
    - iter_batches() on the training side converts to tensors

    Args:
        df: Raw (unencoded) DataFrame batch
        tgt_stats: Target statistics from engine.analyze()
        tgt_context_key: Target context key column name (for sequential data)

    Returns:
        Dict of numpy arrays (compatible with Ray Data map_batches return type)

    Example with Ray Data:
        >>> import ray
        >>> from functools import partial
        >>> from mostlyai.engine import encode_batch, prepare_flat_batch
        >>>
        >>> # After analyze(), get stats
        >>> tgt_stats = workspace.tgt_stats.read()
        >>>
        >>> # Create encoding function with stats bound
        >>> encode_fn = partial(encode_batch, tgt_stats=tgt_stats)
        >>>
        >>> # Create Ray Dataset and encode
        >>> ds = ray.data.read_parquet("data/")
        >>> encoded_ds = ds.map_batches(encode_fn, batch_format="pandas")
        >>>
        >>> # On training side, iterate and convert to tensors
        >>> def tensor_iter(dataset):
        ...     for batch in dataset.iter_batches(batch_format="numpy"):
        ...         yield prepare_flat_batch(batch, device="cuda")
        >>>
        >>> train(workspace_dir, train_tensors=tensor_iter(encoded_ds), ...)
    """
    import pandas as pd
    from mostlyai.engine._tabular.encoding import encode_df

    # Encode the DataFrame
    encoded_df, _, _ = encode_df(
        df,
        tgt_stats,
        ctx_primary_key=None,
        tgt_context_key=tgt_context_key,
        n_jobs=1,  # Single-threaded in Ray workers
    )

    # Convert to dict of numpy arrays (Ray Data compatible)
    return {col: encoded_df[col].to_numpy() for col in encoded_df.columns}


def load_tensors_from_workspace(
    workspace_dir: Path | str,
    device: torch.device | str = "cpu",
    batch_size: int = 1024,
    max_sequence_window: int | None = None,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], "ModelConfig"]:
    """
    Load training and validation tensors from an encoded workspace.

    This is a convenience function for testing and simple workflows that
    loads encoded parquet data from a workspace and creates tensor batches.
    For production with large datasets, use Ray Data pipelines instead.

    Args:
        workspace_dir: Path to workspace directory (after engine.encode())
        device: Device for tensors ("cpu" or "cuda")
        batch_size: Batch size for tensor batches
        max_sequence_window: Optional window size for slicing sequences

    Returns:
        Tuple of (train_batches, val_batches, model_config) where batches are lists

    Example:
        >>> # After split/analyze/encode
        >>> trn, val, config = load_tensors_from_workspace(workspace_dir, device="cuda")
        >>> train(workspace_dir, train_tensors=iter(trn), val_tensors=iter(val), model_config=config)
    """
    import pyarrow.parquet as pq
    import pandas as pd
    from mostlyai.engine._workspace import Workspace

    workspace = Workspace(Path(workspace_dir))
    tgt_stats = workspace.tgt_stats.read()
    is_sequential = tgt_stats.get("is_sequential", False)

    # Build model config
    config = build_model_config_from_workspace(workspace_dir)

    def load_parquet_files(parquet_paths: list[Path]) -> pd.DataFrame:
        if not parquet_paths:
            return pd.DataFrame()
        tables = [pq.read_table(p) for p in parquet_paths]
        return pd.concat([t.to_pandas() for t in tables], ignore_index=True)

    def create_batches(df: pd.DataFrame) -> list[dict[str, torch.Tensor]]:
        """Create tensor batches from a DataFrame."""
        if df.empty:
            return []

        n_samples = len(df)
        batches = []

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_df = df.iloc[start_idx:end_idx]

            # Convert to columnar format
            batch_dict = {col: batch_df[col].tolist() for col in batch_df.columns}

            # Convert to tensors
            if is_sequential:
                tensors = prepare_sequential_batch(batch_dict, device=device)
                if max_sequence_window:
                    tensors = slice_sequences(tensors, max_sequence_window)
            else:
                tensors = prepare_flat_batch(batch_dict, device=device)

            batches.append(tensors)

        return batches

    # Load data from parquet files
    trn_df = load_parquet_files(workspace.encoded_data_trn.fetch_all())
    val_df = load_parquet_files(workspace.encoded_data_val.fetch_all())

    return create_batches(trn_df), create_batches(val_df), config
