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

import inspect
from pathlib import Path
from collections.abc import Callable

import torch

from mostlyai.engine._common import ProgressCallback
from mostlyai.engine.domain import ModelType, ModelStateStrategy, DifferentialPrivacyConfig
from mostlyai.engine._workspace import resolve_model_type


def train(
    *,
    model: str | None = None,
    max_training_time: float | None = 14400.0,  # 10 days
    max_epochs: float | None = 100.0,  # 100 epochs
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    enable_flexible_generation: bool = True,
    max_sequence_window: int | None = None,
    value_protection: bool = True,
    differential_privacy: DifferentialPrivacyConfig | dict | None = None,
    model_state_strategy: ModelStateStrategy = ModelStateStrategy.reset,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
    upload_model_data_callback: Callable | None = None,  # TODO
) -> None:
    """
    Trains a model with optional early stopping and differential privacy.

    Creates the following folder structure within the `workspace_dir`:

    - `ModelStore`: Trained model checkpoints and logs.

    Args:
        model: The identifier of the model to train. If tabular, defaults to MOSTLY_AI/Medium. If language, defaults to MOSTLY_AI/LSTMFromScratch-3m.
        max_training_time: Maximum training time in minutes. If None, defaults to 10 days.
        max_epochs: Maximum number of training epochs. If None, defaults to 100 epochs.
        batch_size: Per-device batch size for training and validation. If None, determined automatically.
        gradient_accumulation_steps: Number of steps to accumulate gradients. If None, determined automatically.
        enable_flexible_generation: Whether to enable flexible order generation. Defaults to True.
        max_sequence_window: Maximum sequence window for tabular sequential models. Only applicable for tabular models.
        differential_privacy: Configuration for differential privacy training. If None, DP is disabled.
        model_state_strategy: Strategy for handling existing model state (reset/resume/reuse).
        device: Device to run training on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        workspace_dir: Directory path for workspace. Training outputs are stored in ModelStore subdirectory.
        update_progress: Callback function to report training progress.
        upload_model_data_callback: Callback function to upload model data during training.
    """
    model_type = resolve_model_type(workspace_dir)
    if model_type == ModelType.tabular:
        from mostlyai.engine._tabular.training import train as train_tabular

        args = inspect.signature(train_tabular).parameters
        train_tabular(
            model=model if model else args["model"].default,
            workspace_dir=workspace_dir,
            max_training_time=max_training_time if max_training_time else args["max_training_time"].default,
            max_epochs=max_epochs if max_epochs else args["max_epochs"].default,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            enable_flexible_generation=enable_flexible_generation,
            value_protection=value_protection,
            differential_privacy=differential_privacy,
            update_progress=update_progress,
            upload_model_data_callback=upload_model_data_callback,
            model_state_strategy=model_state_strategy,
            device=device,
            max_sequence_window=max_sequence_window if max_sequence_window else args["max_sequence_window"].default,
        )
    else:
        from mostlyai.engine._language.training import train as train_language

        if max_sequence_window is not None:
            raise ValueError("max_sequence_window is not supported for language models")

        args = inspect.signature(train_language).parameters
        train_language(
            model=model if model else args["model"].default,
            workspace_dir=workspace_dir,
            max_training_time=max_training_time if max_training_time else args["max_training_time"].default,
            max_epochs=max_epochs if max_epochs else args["max_epochs"].default,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            enable_flexible_generation=enable_flexible_generation,
            value_protection=value_protection,
            differential_privacy=differential_privacy,
            update_progress=update_progress,
            upload_model_data_callback=upload_model_data_callback,
            model_state_strategy=model_state_strategy,
            device=device,
        )
