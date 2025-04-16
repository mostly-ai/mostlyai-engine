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
Categorical encoding maps each categorical value to its own integer code.
"""

import numpy as np
import pandas as pd

from mostlyai.engine._common import safe_convert_string, dp_non_rare

CATEGORICAL_UNKNOWN_TOKEN = "_RARE_"
CATEGORICAL_NULL_TOKEN = "<<NULL>>"
CATEGORICAL_SUB_COL_SUFFIX = "cat"
CATEGORICAL_ESCAPE_CHAR = "\x01"


def safe_categorical_escape(values: pd.Series) -> pd.Series:
    """Inplace escaping of categorical values"""
    reserved_tokens = (CATEGORICAL_UNKNOWN_TOKEN, CATEGORICAL_NULL_TOKEN)
    reserved_tokens_replacement_map = {t: CATEGORICAL_ESCAPE_CHAR + t for t in reserved_tokens}
    # first, prefix values starting with escape char with another escape char
    mask = values.str.startswith(CATEGORICAL_ESCAPE_CHAR, na=False)
    values.loc[mask] = values.loc[mask].str.slice_replace(stop=1, repl=CATEGORICAL_ESCAPE_CHAR * 2)
    # second, add escape char to all reserved tokens
    values = values.replace(reserved_tokens_replacement_map)
    return values


def safe_categorical_unescape(values: pd.Series) -> pd.Series:
    """Inplace un-escaping of categorical values"""
    # de-prefix all values starting with escape char by removing just the first one
    mask = values.str.startswith(CATEGORICAL_ESCAPE_CHAR, na=False)
    values.loc[mask] = values.loc[mask].str[1:]
    return values


def analyze_categorical(
    values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None, *, safe_escape: bool = True
) -> dict:
    # ensure a safe representation of values: 1. string dtype; 2. escape reserved tokens
    values = safe_convert_string(values)
    if safe_escape:
        values = safe_categorical_escape(values)
    # count distinct root_keys per categorical value for rare-category protection
    df = pd.concat([root_keys, values], axis=1)
    cnt_values = df.groupby(values.name)[root_keys.name].nunique().to_dict()
    stats = {"has_nan": sum(values.isna()) > 0, "cnt_values": cnt_values}
    return stats


def analyze_reduce_categorical(
    stats_list: list[dict],
    value_protection: bool = True,
    value_protection_epsilon: float | None = None,
    value_protection_delta: float | None = None,
) -> dict:
    assert (value_protection_epsilon is not None) is (value_protection_delta is not None), (
        "value_protection_epsilon and value_protection_delta must be both set or both unset"
    )
    # sum up all counts for each categorical value
    cnt_values: dict[str, int] = {}
    for item in stats_list:
        for value, count in item["cnt_values"].items():
            cnt_values[value] = cnt_values.get(value, 0) + count
    # create alphabetically sorted list of non-rare categories
    known_categories = [k for k in sorted(cnt_values.keys())]
    if value_protection:
        if value_protection_delta is not None and value_protection_epsilon is not None:
            categories = dp_non_rare(cnt_values, value_protection_epsilon, value_protection_delta, threshold=5)
        else:
            # stochastic threshold for rare categories
            rare_min = 5 + int(3 * np.random.uniform())
            categories = [k for k in known_categories if cnt_values[k] >= rare_min]
    else:
        categories = known_categories
    no_of_rare_categories = len(known_categories) - len(categories)
    # add special token for MISSING categories, if any are present
    if any([j["has_nan"] for j in stats_list]):
        categories = [CATEGORICAL_NULL_TOKEN] + categories
    # add special token for UNKNOWN categories at first position
    categories = [CATEGORICAL_UNKNOWN_TOKEN] + categories
    stats = {
        "no_of_rare_categories": no_of_rare_categories,
        "codes": {categories[i]: i for i in range(len(categories))},
        "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: len(categories)},
    }
    return stats


def encode_categorical(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.DataFrame:
    # ensure a safe representation of values: 1. string dtype; 2. escape reserved tokens
    values = safe_categorical_escape(safe_convert_string(values))
    known_categories = [str(k) for k in stats["codes"].keys()]
    values = values.copy()
    if CATEGORICAL_NULL_TOKEN in known_categories:
        values[values.isna()] = CATEGORICAL_NULL_TOKEN
    values[~values.isin(known_categories)] = CATEGORICAL_UNKNOWN_TOKEN

    # map categories to their corresponding codes
    codes = pd.Series(
        pd.Categorical(values, categories=known_categories).codes,
        name=CATEGORICAL_SUB_COL_SUFFIX,
        index=values.index,
    )
    return codes.to_frame()


def decode_categorical(df_encoded: pd.DataFrame, stats: dict) -> pd.Series:
    categories = stats["codes"].keys()
    values = pd.Series(
        pd.Categorical.from_codes(df_encoded[CATEGORICAL_SUB_COL_SUFFIX], categories=categories),
        dtype="string",
    )
    values[values == CATEGORICAL_NULL_TOKEN] = pd.NA
    # convert escaped values to their original representation
    values = safe_categorical_unescape(values)
    return values
