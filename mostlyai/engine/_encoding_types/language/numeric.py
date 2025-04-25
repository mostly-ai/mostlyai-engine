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

from mostlyai.engine._common import (
    ANALYZE_MIN_MAX_TOP_N,
    ANALYZE_REDUCE_MIN_MAX_N,
    dp_quantiles,
    get_stochastic_rare_threshold,
    safe_convert_numeric,
)
from mostlyai.engine._encoding_types.tabular.numeric import _type_safe_numeric_series
from mostlyai.engine.domain import ModelEncodingType


def analyze_language_numeric(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    values = safe_convert_numeric(values)

    # determine lowest/highest values by root ID, and return top ANALYZE_MIN_MAX_TOP_N
    df = pd.concat([root_keys, values], axis=1)
    min_values = df.groupby(root_keys.name)[values.name].min().dropna()
    min_n = min_values.sort_values(ascending=True).head(ANALYZE_MIN_MAX_TOP_N).tolist()
    max_values = df.groupby(root_keys.name)[values.name].max().dropna()
    max_n = max_values.sort_values(ascending=False).head(ANALYZE_MIN_MAX_TOP_N).tolist()

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
        "min_n": min_n,
        "max_n": max_n,
    }
    return stats


def analyze_reduce_language_numeric(
    stats_list: list[dict],
    value_protection: bool = True,
    value_protection_epsilon: float | None = None,
    value_protection_delta: float | None = None,
) -> dict:
    # check for occurrence of NaN values
    has_nan = any([j["has_nan"] for j in stats_list])

    # determine max scale
    max_scale = max([j["max_scale"] for j in stats_list])

    # determine min / max 5 values to map too low / too high values to
    reduced_min_n = sorted([v for min_n in [j["min_n"] for j in stats_list] for v in min_n], reverse=False)
    reduced_max_n = sorted([v for max_n in [j["max_n"] for j in stats_list] for v in max_n], reverse=True)
    if value_protection:
        if len(reduced_min_n) < ANALYZE_REDUCE_MIN_MAX_N or len(reduced_max_n) < ANALYZE_REDUCE_MIN_MAX_N:
            # protect all values if there are less than ANALYZE_REDUCE_MIN_MAX_N values
            reduced_min = None
            reduced_max = None
        else:
            if value_protection_delta is not None and value_protection_epsilon is not None:
                values = sorted(reduced_min_n + reduced_max_n)
                quantiles = [0.01, 0.99] if len(values) >= 10_000 else [0.05, 0.95]
                reduced_min, reduced_max = dp_quantiles(
                    values, quantiles, value_protection_epsilon, value_protection_delta
                )
                if max_scale == 0:
                    reduced_min = int(reduced_min)
                    reduced_max = int(reduced_max)
            else:
                reduced_min = reduced_min_n[get_stochastic_rare_threshold(min_threshold=5)]
                reduced_max = reduced_max_n[get_stochastic_rare_threshold(min_threshold=5)]
    else:
        reduced_min = reduced_min_n[0] if len(reduced_min_n) > 0 else None
        reduced_max = reduced_max_n[0] if len(reduced_max_n) > 0 else None

    stats = {
        "encoding_type": ModelEncodingType.language_numeric.value,
        "has_nan": has_nan,
        "max_scale": max_scale,
        "min": reduced_min,
        "max": reduced_max,
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
    if stats["min"] is not None:
        reduced_min = _type_safe_numeric_series([stats["min"]], dtype).iloc[0]
        values.loc[values < reduced_min] = reduced_min
    if stats["max"] is not None:
        reduced_max = _type_safe_numeric_series([stats["max"]], dtype).iloc[0]
        values.loc[values > reduced_max] = reduced_max
    return values


def decode_language_numeric(x: pd.Series, stats: dict[str, str]) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.round(stats["max_scale"])
    if stats["min"] is not None:
        reduced_min = np.dtype(x.dtype).type(stats["min"])
        x.loc[x < reduced_min] = reduced_min
    if stats["max"] is not None:
        reduced_max = np.dtype(x.dtype).type(stats["max"])
        x.loc[x > reduced_max] = reduced_max
    dtype = "Int64" if stats["max_scale"] == 0 else float
    return x.astype(dtype)
