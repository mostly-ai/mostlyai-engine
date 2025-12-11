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

from collections.abc import Callable, Iterator
from pathlib import Path

import torch

from mostlyai.engine._common import ProgressCallback
from mostlyai.engine._tabular.training import ModelConfig
from mostlyai.engine.domain import ModelStateStrategy


def train(
    *,
    # Required: tensor data and configuration
    train_tensors: Iterator[dict[str, torch.Tensor]],
    val_tensors: Iterator[dict[str, torch.Tensor]],
    model_config: ModelConfig,
    # Required: output location
    workspace_dir: str | Path,
    # Training parameters
    model: str = "MOSTLY_AI/Medium",
    max_training_time: float = 14400.0,
    max_epochs: float = 100.0,
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    enable_flexible_generation: bool = True,
    model_state_strategy: ModelStateStrategy | str = ModelStateStrategy.reset,
    device: torch.device | str | None = None,
    update_progress: ProgressCallback | None = None,
    upload_model_data_callback: Callable | None = None,
) -> None:
    """
    Train a TabularARGN model from pre-computed tensor batches.

    This function accepts pre-computed tensor batches directly, eliminating the
    CPU overhead of per-batch data transformation. Use with Ray Data pipelines
    for distributed encoding and streaming tensor delivery.

    Args:
        train_tensors: Iterator yielding training batches as dict[str, torch.Tensor].
                       Use prepare_flat_batch() or prepare_sequential_batch() to create.
        val_tensors: Iterator yielding validation batches as dict[str, torch.Tensor]
        model_config: Model configuration from build_model_config() or build_model_config_from_workspace()
        workspace_dir: Directory for model weights and progress tracking
        model: Model size ("MOSTLY_AI/Small", "MOSTLY_AI/Medium", "MOSTLY_AI/Large")
        max_training_time: Maximum training time in minutes (default: 14400 = 10 days)
        max_epochs: Maximum number of epochs (default: 100)
        batch_size: Batch size for heuristics (batches are pre-sized by caller)
        gradient_accumulation_steps: Gradient accumulation steps
        enable_flexible_generation: Enable flexible column order generation
        model_state_strategy: How to handle existing model state (reset/resume/reuse)
        device: Device for training (auto-detected if None)
        update_progress: Progress callback
        upload_model_data_callback: Callback for model data upload

    Example:
        >>> from mostlyai import engine
        >>>
        >>> # After split/analyze/encode
        >>> trn, val, config = engine.load_tensors_from_workspace(workspace_dir, device="cuda")
        >>> engine.train(
        ...     train_tensors=iter(trn),
        ...     val_tensors=iter(val),
        ...     model_config=config,
        ...     workspace_dir=workspace_dir,
        ...     max_epochs=10,
        ... )

    Example with Ray Data:
        >>> # Create tensor iterator from Ray Dataset
        >>> def tensor_iter(dataset):
        ...     for batch in dataset.iter_batches(batch_format="numpy"):
        ...         yield engine.prepare_flat_batch(batch, device="cuda")
        >>>
        >>> config = engine.build_model_config(tgt_stats)
        >>> engine.train(
        ...     train_tensors=tensor_iter(train_ds),
        ...     val_tensors=tensor_iter(val_ds),
        ...     model_config=config,
        ...     workspace_dir="/path/to/output",
        ... )
    """
    from mostlyai.engine._tabular.training import train as train_tabular

    train_tabular(
        train_tensors=train_tensors,
        val_tensors=val_tensors,
        model_config=model_config,
        workspace_dir=workspace_dir,
        model=model,
        max_training_time=max_training_time,
        max_epochs=max_epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        enable_flexible_generation=enable_flexible_generation,
        model_state_strategy=model_state_strategy,
        device=device,
        update_progress=update_progress,
        upload_model_data_callback=upload_model_data_callback,
    )
