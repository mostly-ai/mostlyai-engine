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

import numpy as np
import pandas as pd

from mostlyai.engine._common import safe_convert_string

CATEGORICAL_UNKNOWN_TOKEN = "_RARE_"


def analyze_language_categorical(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    values = safe_convert_string(values)
    # count distinct root_keys per categorical value for rare-category protection
    df = pd.concat([root_keys, values], axis=1)
    cnt_values = df.groupby(values.name)[root_keys.name].nunique().to_dict()
    stats = {"has_nan": sum(values.isna()) > 0, "cnt_values": cnt_values}
    return stats


def analyze_reduce_language_categorical(stats_list: list[dict], value_protection: bool = True) -> dict:
    # sum up all counts for each categorical value
    cnt_values: dict[str, int] = {}
    for item in stats_list:
        for value, count in item["cnt_values"].items():
            cnt_values[value] = cnt_values.get(value, 0) + count
    # create alphabetically sorted list of non-rare categories
    known_categories = [k for k in sorted(cnt_values.keys())]
    if value_protection:
        # stochastic threshold for rare categories
        rare_min = 5 + int(3 * np.random.uniform())
    else:
        rare_min = 0
    categories = [k for k in known_categories if cnt_values[k] >= rare_min]
    no_of_rare_categories = len(known_categories) - len(categories)
    # add None to categories, if any are present
    if any([j["has_nan"] for j in stats_list]):
        categories = [None] + categories
    # add special token for UNKNOWN categories at first position
    if no_of_rare_categories > 0:
        categories = [CATEGORICAL_UNKNOWN_TOKEN] + categories
    stats = {"no_of_rare_categories": no_of_rare_categories, "categories": categories}
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
