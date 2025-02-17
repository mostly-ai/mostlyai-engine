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
import numpy as np
import pandas as pd

from mostlyai.engine._common import safe_convert_numeric
from mostlyai.engine._encoding_types.tabular.numeric import _type_safe_numeric_series
from mostlyai.engine.domain import ModelEncodingType


def analyze_language_numeric(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    values = safe_convert_numeric(values)

    # determine lowest/highest values by root ID, and return top 11
    df = pd.concat([root_keys, values], axis=1)
    min_values = df.groupby(root_keys.name)[values.name].min().dropna()
    min11 = min_values.sort_values(ascending=True).head(11).tolist()
    max_values = df.groupby(root_keys.name)[values.name].max().dropna()
    max11 = max_values.sort_values(ascending=False).head(11).tolist()

    # determine if there are any NaN values
    has_nan = bool(values.isna().any())

    # determine max scale
    def count_scale(num: float) -> int:
        # represent number as fixed point string, remove trailing zeros and decimal point
        num = format(num, "f").rstrip("0").rstrip(".")
        if "." in num:
            # in case of decimal, return number of digits after decimal point
            return len(num.split(".")[1])
        # in case of integer, return 0
        return 0

    max_scale = int(values.apply(count_scale).max())

    stats = {
        "has_nan": has_nan,
        "max_scale": max_scale,
        "min11": min11,
        "max11": max11,
    }
    return stats


def analyze_reduce_language_numeric(stats_list: list[dict], value_protection: bool = True) -> dict:
    # check for occurrence of NaN values
    has_nan = any([j["has_nan"] for j in stats_list])

    # determine max scale
    max_scale = max([j["max_scale"] for j in stats_list])

    # determine min / max 5 values to map too low / too high values to
    min11 = sorted([v for min11 in [j["min11"] for j in stats_list] for v in min11], reverse=False)[:11]
    max11 = sorted([v for max11 in [j["max11"] for j in stats_list] for v in max11], reverse=True)[:11]
    if value_protection:
        # extreme value protection - discard lowest/highest 5 values
        if len(min11) < 11 or len(max11) < 11:
            # less than 11 subjects with non-NULL values; we need to protect all
            min5 = []
            max5 = []
        else:
            min5 = min11[5:10]  # drop 1 to 5th lowest; keep 6th to 10th lowest
            max5 = max11[5:10]  # drop 1 to 5th highest; keep 6th to 10th highest
    else:
        min5 = min11[0:5]
        max5 = max11[0:5]

    stats = {
        "encoding_type": ModelEncodingType.language_numeric.value,
        "has_nan": has_nan,
        "max_scale": max_scale,
        "min5": min5,
        "max5": max5,
    }

    return stats


def encode_language_numeric(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.DataFrame:
    values = safe_convert_numeric(values)
    # try to convert to int, if possible
    dtype = "Int64" if stats["max_scale"] == 0 else "Float64"
    if dtype == "Int64":
        values = values.round()
    try:
        values = values.astype(dtype)
    except TypeError:
        if dtype == "Int64":  # if couldn't safely convert to int, stick to float
            dtype = "Float64"
            values = values.astype(dtype)
    # reset index, as `values.mask` can throw errors for misaligned indices
    values.reset_index(drop=True, inplace=True)
    # replace extreme values with randomly sampled 5-th to 10-th largest/smallest values
    min5 = _type_safe_numeric_series(stats["min5"] or [0], dtype)
    max5 = _type_safe_numeric_series(stats["max5"] or [0], dtype)
    values.mask(
        values < min5[0],
        min5.sample(n=len(values), replace=True, ignore_index=True),
        inplace=True,
    )
    values.mask(
        values > max5[0],
        max5.sample(n=len(values), replace=True, ignore_index=True),
        inplace=True,
    )
    return values


def _clip_numeric(x: pd.Series, min5: list, max5: list) -> pd.Series:
    x_numeric = pd.to_numeric(x, errors="coerce")
    min_arr = np.array(min5, dtype=x_numeric.dtype)
    max_arr = np.array(max5, dtype=x_numeric.dtype)
    n = len(x_numeric)
    random_mins = np.random.choice(min_arr, size=n)
    random_maxs = np.random.choice(max_arr, size=n)
    clipped = np.minimum(np.maximum(x_numeric.to_numpy(), random_mins), random_maxs)
    return pd.Series(clipped, index=x.index)


def decode_numeric(x: pd.Series, col_stats: dict[str, str]) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.round(col_stats["max_scale"])
    x = _clip_numeric(x, col_stats["min5"], col_stats["max5"])
    dtype = "Int64" if col_stats["max_scale"] == 0 else float
    return x.astype(dtype)
