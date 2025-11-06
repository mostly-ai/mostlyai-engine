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

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mostlyai.engine._workspace import resolve_model_type
from mostlyai.engine.domain import ModelType


def log_prob(
    tgt_data: pd.DataFrame,
    *,
    ctx_data: pd.DataFrame | None = None,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
) -> np.ndarray:
    """
    Compute log-probability (log-likelihood) of samples under the trained model.

    Only supports TABULAR models for now. Raises ValueError for LANGUAGE models.

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
        ValueError: If the model type is not tabular (language models are not supported).
    """
    model_type = resolve_model_type(workspace_dir)
    if model_type == ModelType.tabular:
        from mostlyai.engine._tabular.log_prob import log_prob as log_prob_tabular

        return log_prob_tabular(
            tgt_data=tgt_data,
            ctx_data=ctx_data,
            device=device,
            workspace_dir=workspace_dir,
        )
    else:
        raise ValueError("log_prob() is only supported for tabular models, not language models")
