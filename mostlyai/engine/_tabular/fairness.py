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
from functools import partial
from typing import Any, TypedDict
from collections.abc import Callable

import pandas as pd
import numpy as np

from itertools import product
import torch

from mostlyai.engine._common import (
    get_argn_name,
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
)
from mostlyai.engine._encoding_types.tabular.categorical import CATEGORICAL_SUB_COL_SUFFIX, CATEGORICAL_UNKNOWN_TOKEN
from mostlyai.engine.domain import ModelEncodingType, FairnessConfig

_LOG = logging.getLogger(__name__)


class FairnessTransforms(TypedDict):
    target_sub_col: str
    sensitive_sub_cols: list[str]
    transforms: dict[int, dict[tuple[int], callable]]


def _get_sensitive_groups(
    column_stats: dict[str, Any],
    sensitive_cols: list[str],
    sensitive_sub_cols: list[str],
) -> pd.DataFrame:
    """
    Create a DataFrame with all possible combinations of (encoded) values for the sensitive columns.
    """
    category_map = {}
    for sensitive_col in sensitive_cols:
        category_map[sensitive_col] = list(column_stats[sensitive_col]["codes"].values())
        # when there are no rare categories, we can exclude _RARE_ to reduce the number of combinations
        if column_stats[sensitive_col]["no_of_rare_categories"] == 0:
            category_map[sensitive_col].remove(column_stats[sensitive_col]["codes"][CATEGORICAL_UNKNOWN_TOKEN])
    groups_df = pd.DataFrame(list(product(*category_map.values())), columns=sensitive_sub_cols)
    _LOG.info(f"fairness: {groups_df.shape[0]} sensitive groups")
    return groups_df


def _estimate_target_distribution(
    forward_fn: Callable,
    device: str | torch.device,
    target_sub_col: str,
    sample_size: int,
    n_quantiles: int = 101,
    seed_values: dict[str, int] | None = None,
) -> np.ndarray:
    """
    Estimate the (conditional or unconditional) empirical  distribution of the target column by fetching model probabilities for
     that column, and then computing the quantiles of that empirical distribution.

    :return: A numpy array of shape (n_quantiles, target_cardinality).
    """
    if seed_values is None:
        fixed_values = None
        _LOG.info(f"fairness: sample {sample_size:,} unconditionally")
    else:
        fixed_values = {
            col: torch.as_tensor([values] * sample_size, device=device).type(torch.int)
            for col, values in seed_values.items()
        }
        _LOG.info(f"fairness: sample {sample_size:,} for sensitive group {seed_values}")

    # fetch probs for target column
    _, probs = forward_fn(
        x={},
        mode="gen",
        batch_size=sample_size,
        fixed_values=fixed_values,
        return_probs=[target_sub_col],
    )
    # convert to numpy
    probs = probs[target_sub_col].detach().cpu().numpy()
    # calculate quantiles, and wrap with 0 a
    quantiles = np.concatenate(
        [
            np.zeros((1, probs.shape[1])),
            np.quantile(probs, np.linspace(0, 1, n_quantiles), axis=0),
            np.ones((1, probs.shape[1])),
        ]
    )
    return quantiles


def get_fairness_transforms(
    forward_fn: Callable,
    device: str | torch.device,
    fairness: FairnessConfig,
    tgt_stats: dict[str, Any],
) -> FairnessTransforms:
    """
    Generate the transforms for the target column conditioned on the sensitive columns.
    """
    _LOG.info(
        f"fairness: create transforms for target `{fairness.target_column}` across "
        f"sensitive columns: {''.join(fairness.sensitive_columns)}"
    )

    # validate the fairness configuration
    if fairness.sensitive_columns is None or len(fairness.sensitive_columns) == 0:
        raise ValueError("The sensitive columns must be specified in the fairness configuration")
    if fairness.target_column is None:
        raise ValueError("The target column must be specified in the fairness configuration")
    if fairness.target_column in fairness.sensitive_columns:
        raise ValueError("The target column cannot be a sensitive column")
    cols = [fairness.target_column] + fairness.sensitive_columns
    for col in cols:
        if col not in tgt_stats["columns"]:
            raise ValueError(f"Invalid fairness column: `{col}` not found in the training data")
        if tgt_stats["columns"][col]["encoding_type"] != ModelEncodingType.tabular_categorical:
            raise ValueError(f"Invalid fairness column: `{col}` is not encoded categorically")

    # estimate the marginal target distribution
    target_sub_col = get_argn_name(
        argn_processor=tgt_stats["columns"][fairness.target_column][ARGN_PROCESSOR],
        argn_table=tgt_stats["columns"][fairness.target_column][ARGN_TABLE],
        argn_column=tgt_stats["columns"][fairness.target_column][ARGN_COLUMN],
        argn_sub_column=CATEGORICAL_SUB_COL_SUFFIX,
    )
    marginal_target_quantiles: np.ndarray = _estimate_target_distribution(
        forward_fn=forward_fn,
        device=device,
        target_sub_col=target_sub_col,
        sample_size=15_000,
        n_quantiles=101,
        seed_values=None,
    )

    # estimate the conditional target distribution for each combination of sensitive values
    sensitive_sub_cols = [
        get_argn_name(
            argn_processor=tgt_stats["columns"][col][ARGN_PROCESSOR],
            argn_table=tgt_stats["columns"][col][ARGN_TABLE],
            argn_column=tgt_stats["columns"][col][ARGN_COLUMN],
            argn_sub_column=CATEGORICAL_SUB_COL_SUFFIX,
        )
        for col in fairness.sensitive_columns
    ]
    sensitive_groups_df = _get_sensitive_groups(
        column_stats=tgt_stats["columns"],
        sensitive_cols=fairness.sensitive_columns,
        sensitive_sub_cols=sensitive_sub_cols,
    )
    conditional_target_quantiles: dict[tuple[int], np.ndarray] = {}
    for _, sensitive_group in sensitive_groups_df.iterrows():
        seed_values = sensitive_group.to_dict()
        conditional_target_quantiles[tuple(seed_values.values())] = _estimate_target_distribution(
            forward_fn=forward_fn,
            device=device,
            target_sub_col=target_sub_col,
            sample_size=5_000,
            n_quantiles=101,
            seed_values=seed_values,
        )

    # calculate the transforms, one for each target value and sensitive group
    transforms: dict[int, dict[tuple[int], callable]] = {}
    for i in range(marginal_target_quantiles.shape[1]):
        transforms[i] = {}
        for sensitive_group, target_quantiles in conditional_target_quantiles.items():
            transforms[i][sensitive_group] = partial(
                torch_interp,
                xp=torch.as_tensor(target_quantiles[:, i], device=device),
                fp=torch.as_tensor(marginal_target_quantiles[:, i], device=device),
            )

    _LOG.info("fairness: created transforms")
    return {
        "target_sub_col": target_sub_col,
        "sensitive_sub_cols": sensitive_sub_cols,
        "transforms": transforms,
    }


def apply_fairness_transforms(
    sub_col: str, probs: torch.Tensor, outputs: dict[str, torch.Tensor], fairness_transforms: FairnessTransforms
) -> torch.Tensor:
    """
    Apply fairness transforms to the generated probabilities of the target column.

    :param sub_col: The current sub column.
    :param probs: The probabilities for the current sub column.
    :param outputs: The sampled values for the sensitive columns.
    :param fairness_transforms: The transform callables to apply.
    :return: The transformed probabilities for the target column.
    """
    # only transform the target column
    device = probs.device
    if sub_col == fairness_transforms["target_sub_col"]:
        sensitive_values = torch.stack([outputs[col] for col in fairness_transforms["sensitive_sub_cols"]]).t()
        for i, i_transforms in fairness_transforms["transforms"].items():
            for group, transform in i_transforms.items():
                probs[:, i] = torch.where(
                    torch.all(
                        sensitive_values == torch.as_tensor(group, device=device).tile((probs.shape[0], 1)),
                        dim=1,
                    ),
                    transform(probs[:, i]),
                    probs[:, i],
                )
        # normalize transformed probabilities to make sure they sum to 1
        probs /= probs.sum(dim=1, keepdim=True)
    return probs


def torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """
    Torch implementation of numpy.interp.
    """
    # find the indices of the bins to which each value in x belongs
    indices = torch.searchsorted(xp, x, side="right")
    indices = torch.clamp(indices, 1, len(xp) - 1)
    # compute the slopes of the segments
    x_lo = xp[indices - 1]
    x_hi = xp[indices]
    y_lo = fp[indices - 1]
    y_hi = fp[indices]
    # compute the interpolated values
    eps = 1e-10
    slopes = (y_hi - y_lo) / (x_hi - x_lo + eps)
    interpolated_values = y_lo + slopes * (x - x_lo)
    return interpolated_values
