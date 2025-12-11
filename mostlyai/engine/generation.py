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

import pandas as pd

from mostlyai.engine._common import ProgressCallback
from mostlyai.engine.domain import (
    FairnessConfig,
    ImputationConfig,
    RareCategoryReplacementMethod,
    RebalancingConfig,
)


def generate(
    *,
    ctx_data: pd.DataFrame | None = None,
    seed_data: pd.DataFrame | None = None,
    sample_size: int | None = None,
    batch_size: int | None = None,
    sampling_temperature: float = 1.0,
    sampling_top_p: float = 1.0,
    device: str | None = None,
    rare_category_replacement_method: RareCategoryReplacementMethod | str = RareCategoryReplacementMethod.constant,
    rebalancing: RebalancingConfig | dict | None = None,
    imputation: ImputationConfig | dict | None = None,
    fairness: FairnessConfig | dict | None = None,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
) -> None:
    """
    Generates synthetic data from a trained model.

    Creates the following folder structure within the `workspace_dir`:

    - `SyntheticData`: Generated synthetic data, stored as parquet files.

    Args:
        ctx_data: Context data to be used for generation.
        seed_data: Seed data to condition generation on fixed target columns.
        sample_size: Number of samples to generate. Defaults to number of original samples.
        batch_size: Batch size for generation. If None, determined automatically.
        sampling_temperature: Sampling temperature. Higher values increase randomness.
        sampling_top_p: Nucleus sampling probability threshold.
        device: Device to run generation on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        rare_category_replacement_method: Method for handling rare categories.
        rebalancing: Configuration for rebalancing column distributions.
        imputation: List of columns to impute missing values.
        fairness: Configuration for fairness constraints.
        workspace_dir: Directory path for workspace.
        update_progress: Callback for progress updates.
    """
    from mostlyai.engine._tabular.generation import generate as generate_tabular

    return generate_tabular(
        ctx_data=ctx_data,
        seed_data=seed_data,
        sample_size=sample_size,
        batch_size=batch_size,
        sampling_temperature=sampling_temperature,
        sampling_top_p=sampling_top_p,
        rare_category_replacement_method=rare_category_replacement_method,
        rebalancing=rebalancing,
        imputation=imputation,
        fairness=fairness,
        device=device,
        workspace_dir=workspace_dir,
        update_progress=update_progress,
    )
