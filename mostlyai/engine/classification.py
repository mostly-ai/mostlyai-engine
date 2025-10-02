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

from pathlib import Path

import pandas as pd

from mostlyai.engine._workspace import resolve_model_type
from mostlyai.engine.domain import ModelType


def classify(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    workspace_dir: str | Path,
    output: pd.DataFrame | None = None,
) -> pd.DataFrame:
    model_type = resolve_model_type(workspace_dir)
    if model_type != ModelType.tabular:
        raise NotImplementedError("classify is currently supported for TABULAR models only")
    from mostlyai.engine._tabular.classification import classify as classify_tabular

    return classify_tabular(
        data=data,
        features=features,
        target=target,
        workspace_dir=workspace_dir,
        output=output,
    )
