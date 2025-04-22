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
Character encoding splits any value into its characters, and encodes each position then separately as a categorical.
"""

import numpy as np
import pandas as pd

from mostlyai.engine._common import dp_non_rare, get_stochastic_rare_threshold, safe_convert_string

UNKNOWN_TOKEN = "\0"
MAX_LENGTH_CHARS = 50


def analyze_character(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    values = safe_convert_string(values)
    df_split = split_sub_columns_character(values)
    has_nan = sum(df_split["nan"]) > 0
    # count distinct root_keys per token for each character position
    df = pd.concat([root_keys, df_split], axis=1)
    characters = {
        sub_col: df.groupby(sub_col)[root_keys.name].nunique().to_dict()
        for sub_col in df_split.columns
        if sub_col.startswith("P")
    }
    stats = {
        "max_string_length": len(characters),
        "has_nan": has_nan,
        "characters": characters,
    }
    return stats


def analyze_reduce_character(
    stats_list: list[dict],
    value_protection: bool = True,
    value_protection_epsilon: float | None = None,
    value_protection_delta: float | None = None,
) -> dict:
    # gather maximum string length across partitions
    max_string_length = max(stats["max_string_length"] for stats in stats_list)
    positions = [f"P{idx}" for idx in range(max_string_length)]
    # gather codes for each position
    codes: dict[str, dict[str, int]] = {pos: {} for pos in positions}
    for pos in positions:
        cnt_values: dict[str, int] = {}
        # sum up all counts for each token
        for item in stats_list:
            for value, count in item["characters"].get(pos, {}).items():
                cnt_values[value] = cnt_values.get(value, 0) + count
        cnt_values = dict(sorted(cnt_values.items()))
        known_categories = list(cnt_values.keys())
        if value_protection:
            if value_protection_delta is not None and value_protection_epsilon is not None:
                categories = dp_non_rare(cnt_values, value_protection_epsilon, value_protection_delta, threshold=5)
            else:
                rare_min = get_stochastic_rare_threshold(min_threshold=5)
                categories = [k for k in known_categories if cnt_values[k] >= rare_min]
        else:
            categories = known_categories
        # add special token for UNKNOWN at first position
        categories = [UNKNOWN_TOKEN] + [c for c in categories if c != UNKNOWN_TOKEN]
        # assign codes for each token
        codes[pos] = {categories[i]: i for i in range(len(categories))}
    # determine cardinalities
    cardinalities = {}
    has_nan = any([s["has_nan"] for s in stats_list])
    if has_nan:
        cardinalities["nan"] = 2  # binary
    for sub_col, sub_col_codes in codes.items():
        cardinalities[sub_col] = len(sub_col_codes)
    stats = {
        "has_nan": has_nan,
        "max_string_length": max_string_length,
        "codes": codes,
        "cardinalities": cardinalities,
    }
    return stats


def encode_character(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.DataFrame:
    values = safe_convert_string(values)
    max_string_length = stats["max_string_length"]
    df_split = split_sub_columns_character(values, max_string_length)
    if not stats["has_nan"]:
        df_split.drop(["nan"], axis=1, inplace=True)
    for idx in range(max_string_length):
        sub_col = f"P{idx}"
        np_codes = np.array(pd.Categorical(df_split[sub_col], categories=stats["codes"][sub_col]).codes)
        np.place(np_codes, np_codes == -1, 0)
        df_split[sub_col] = np_codes
    return df_split


def split_sub_columns_character(
    values: pd.Series,
    max_string_length: int | None = None,
) -> pd.DataFrame:
    if not pd.api.types.is_string_dtype(values):
        raise ValueError("expected to be string")
    is_na = pd.Series(values.isna().astype("int"), name="nan").to_frame()
    values = values.fillna("")
    # trim strings to a maximum length
    values = values.str.slice(stop=MAX_LENGTH_CHARS)
    # pad strings to string_length
    if max_string_length is None:
        max_string_length = values.str.len().max()
        max_string_length = (
            int(max_string_length)  # type: ignore
            if np.isscalar(max_string_length) and not np.isnan(max_string_length)
            else 0
        )
    else:
        values = values.str.slice(stop=max_string_length)
    # explode to wide dataframe
    padded_values = values.str.ljust(max_string_length, UNKNOWN_TOKEN)
    chars_df = padded_values.str.split("", expand=True)
    if not chars_df.empty:
        chars_df = chars_df.drop([0, max_string_length + 1], axis=1)
        chars_df.columns = [f"P{idx}" for idx in range(max_string_length)]
    else:  # chars_df.empty is True
        # even though the input is empty, we still need to return a dataframe with the correct columns
        chars_df = pd.DataFrame(columns=[f"P{idx}" for idx in range(max_string_length)])
    df = pd.concat([is_na, chars_df], axis=1)
    return df


def decode_character(df_encoded: pd.DataFrame, stats: dict) -> pd.Series:
    if len(stats["codes"].keys()) > 0:
        df_decoded = pd.DataFrame(
            {
                sub_col: pd.Series(
                    pd.Categorical.from_codes(df_encoded[sub_col], categories=stats["codes"][sub_col]),
                    dtype="string",
                )
                for sub_col in stats["codes"].keys()
            },
        )
        values = df_decoded.apply(lambda item: "".join(item), axis=1, result_type="reduce").astype(
            str
        )  # necessary to keep string dtype for empty df_decoded
        # remove unknown tokens and strip trailing whitespaces
        values = values.apply(lambda item: item.replace(UNKNOWN_TOKEN, "")).str.rstrip()
    else:
        # handle de-generate case, where no tokens were stored
        values = pd.Series(pd.NA).repeat(df_encoded.shape[0])
    if stats["has_nan"]:
        values[df_encoded["nan"] == 1] = pd.NA
    return values
