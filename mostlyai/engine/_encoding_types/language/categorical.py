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
Categorical encoding for language models.
"""

import pandas as pd

from mostlyai.engine._common import STRING, safe_convert_string
from mostlyai.engine._encoding_types.tabular.categorical import analyze_categorical, analyze_reduce_categorical

CATEGORICAL_UNKNOWN_TOKEN = "_RARE_"


def analyze_language_categorical(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    return analyze_categorical(values, root_keys, _, safe_escape=False)


def analyze_reduce_language_categorical(
    stats_list: list[dict],
    value_protection: bool = True,
    value_protection_epsilon: float | None = None,
) -> dict:
    stats = analyze_reduce_categorical(stats_list, value_protection, value_protection_epsilon)
    stats["categories"] = list(stats["codes"].keys())
    if any([j["has_nan"] for j in stats_list]):
        # when has_nan, tabular stats are like [CATEGORICAL_UNKNOWN_TOKEN, CATEGORICAL_NULL_TOKEN, ...]
        # and we need to replace CATEGORICAL_NULL_TOKEN with None for language
        stats["categories"][1] = None
    # drop tabular stats
    stats.pop("codes")
    stats.pop("cardinalities")
    return stats


def encode_language_categorical(values: pd.Series, stats: dict) -> pd.Series:
    values = safe_convert_string(values)
    values = values.copy()
    known_categories = stats["categories"]
    mask = ~values.isin(known_categories)
    if None in known_categories:
        mask &= ~pd.isna(values)
    values[mask] = CATEGORICAL_UNKNOWN_TOKEN
    return values


def decode_language_categorical(x: pd.Series, col_stats: dict[str, str]) -> pd.Series:
    x = x.astype(STRING)
    allowed_categories = col_stats.get("categories", [])
    return x.where(x.isin(allowed_categories), other=None)
