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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from mostlyai.engine._workspace import Workspace

_LOG = logging.getLogger(__name__)


def apply_predictor_initialization(
    *,
    argn: Any,
    workspace: Workspace,
    tgt_cardinalities: dict[str, int],
    alpha: float = 1.0,
) -> None:
    """
    Initialize predictor layers with Xavier weights and empirical log-probability biases.

    The empirical distribution is computed from encoded training and validation splits,
    with Laplace smoothing of strength `alpha`.

    Args:
        argn: A tabular ARGN model instance with `predictors.predictors` mapping sub-columns to layers.
        workspace: Workspace providing access to encoded train/val parquet parts.
        tgt_cardinalities: Mapping from target sub-column name to its cardinality.
        alpha: Additive smoothing strength for Laplace smoothing (default: 1.0).
    """
    try:
        counts_map: dict[str, np.ndarray] = {
            sub_col: np.zeros(int(k), dtype=np.float64) for sub_col, k in tgt_cardinalities.items()
        }

        parts: list[Path] = []
        try:
            parts.extend(workspace.encoded_data_trn.fetch_all())
        except Exception:
            pass
        try:
            parts.extend(workspace.encoded_data_val.fetch_all())
        except Exception:
            pass

        for part_path in parts:
            try:
                df_part = pd.read_parquet(part_path)
            except Exception:
                continue
            for sub_col in tgt_cardinalities.keys():
                if sub_col in df_part.columns:
                    vc = df_part[sub_col].value_counts().to_dict()
                    arr = counts_map[sub_col]
                    for idx, cnt in vc.items():
                        if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < arr.shape[0]:
                            arr[int(idx)] += float(cnt)

        for sub_col, layer in getattr(getattr(argn, "predictors", {}), "predictors", {}).items():
            with torch.no_grad():
                # Xavier initialize weights
                nn.init.xavier_uniform_(layer.weight)
                # Empirical log-prob bias with Laplace smoothing
                counts = counts_map.get(sub_col)
                if counts is not None and counts.shape[0] == layer.bias.shape[0]:
                    k = counts.shape[0]
                    denom = counts.sum() + alpha * k
                    probs = (counts + alpha) / max(float(denom), 1e-12)
                    layer.bias.copy_(
                        torch.as_tensor(
                            np.log(np.clip(probs, 1e-12, None)),
                            dtype=layer.bias.dtype,
                            device=layer.bias.device,
                        )
                    )
    except Exception as e:
        _LOG.warning(f"failed to apply weight/bias init: {e}")
