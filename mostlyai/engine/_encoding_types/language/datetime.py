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
import calendar

import numpy as np
import pandas as pd

from mostlyai.engine._common import (
    ANALYZE_MIN_MAX_TOP_N,
    ANALYZE_REDUCE_MIN_MAX_N,
    dp_quantiles,
    get_stochastic_rare_threshold,
    safe_convert_datetime,
)


def analyze_language_datetime(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    values = safe_convert_datetime(values)
    df = pd.concat([root_keys, values], axis=1)
    # determine lowest/highest values by root ID, and return Top 10
    min_dates = df.groupby(root_keys.name)[values.name].min().dropna()
    min_n = min_dates.sort_values(ascending=True).head(ANALYZE_MIN_MAX_TOP_N).astype(str).tolist()
    max_dates = df.groupby(root_keys.name)[values.name].max().dropna()
    max_n = max_dates.sort_values(ascending=False).head(ANALYZE_MIN_MAX_TOP_N).astype(str).tolist()
    # determine if there are any NaN values
    has_nan = bool(values.isna().any())
    # return stats
    stats = {
        "has_nan": has_nan,
        "min_n": min_n,
        "max_n": max_n,
    }
    return stats


def analyze_reduce_language_datetime(
    stats_list: list[dict],
    value_protection: bool = True,
    value_protection_epsilon: float | None = None,
) -> dict:
    # check if there are missing values
    has_nan = any([j["has_nan"] for j in stats_list])
    # determine min / max 5 values to map too low / too high values to
    reduced_min_n = sorted([v for min_n in [j["min_n"] for j in stats_list] for v in min_n], reverse=False)
    reduced_max_n = sorted([v for max_n in [j["max_n"] for j in stats_list] for v in max_n], reverse=True)
    if value_protection:
        if len(reduced_min_n) < ANALYZE_REDUCE_MIN_MAX_N or len(reduced_max_n) < ANALYZE_REDUCE_MIN_MAX_N:
            # protect all values if there are less than ANALYZE_REDUCE_MIN_MAX_N values
            reduced_min = None
            reduced_max = None
        else:
            if value_protection_epsilon is not None:
                values = reduced_min_n + reduced_max_n
                # convert to int64 unix timestamp so that we can apply dp_quantiles
                if any(len(v) > 10 for v in values):
                    dt_format = "%Y-%m-%d %H:%M:%S"
                else:
                    dt_format = "%Y-%m-%d"
                values = pd.to_datetime(values).astype("int64")
                quantiles = [0.01, 0.99] if len(values) >= 10_000 else [0.05, 0.95]
                reduced_min, reduced_max = dp_quantiles(values, quantiles, value_protection_epsilon)
                # convert back to the original string format
                reduced_min = pd.to_datetime(int(reduced_min)).strftime(dt_format)
                reduced_max = pd.to_datetime(int(reduced_max)).strftime(dt_format)
            else:
                reduced_min = str(reduced_min_n[get_stochastic_rare_threshold(min_threshold=5)])
                reduced_max = str(reduced_max_n[get_stochastic_rare_threshold(min_threshold=5)])
    else:
        reduced_min = str(reduced_min_n[0]) if len(reduced_min_n) > 0 else None
        reduced_max = str(reduced_max_n[0]) if len(reduced_max_n) > 0 else None
    stats = {
        "has_nan": has_nan,
        "min": reduced_min,
        "max": reduced_max,
    }
    return stats


def _clip_datetime(values: pd.Series, stats: dict) -> pd.Series:
    if stats["min"] is not None:
        reduced_min = np.datetime64(stats["min"], "ns")
        values.loc[values < reduced_min] = reduced_min
    if stats["max"] is not None:
        reduced_max = np.datetime64(stats["max"], "ns")
        values.loc[values > reduced_max] = reduced_max
    return values


def encode_language_datetime(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.Series:
    # convert
    values = safe_convert_datetime(values)
    values = values.copy()
    # reset index, as `values.mask` can throw errors for misaligned indices
    values.reset_index(drop=True, inplace=True)
    # replace extreme values with min/max
    values = _clip_datetime(values, stats)
    return values


def decode_language_datetime(x: pd.Series, stats: dict[str, str]) -> pd.Series:
    x = x.where(~x.isin(["", "_INVALID_"]), np.nan)

    valid_mask = (
        x.str.len().ge(10)
        & x.str.slice(0, 4).str.isdigit()
        & x.str.slice(5, 7).str.isdigit()
        & x.str.slice(8, 10).str.isdigit()
    )
    if valid_mask.sum() > 0:  # expected "YYYY-MM-DD" prefix
        # handle the date portion, ensuring validity
        years = x[valid_mask].str.slice(0, 4).astype(int)
        months = x[valid_mask].str.slice(5, 7).astype(int)
        days = x[valid_mask].str.slice(8, 10).astype(int)

        # clamp days according to maximum possible day of the month of a given year
        last_days = np.array([calendar.monthrange(y, m)[1] for y, m in zip(years, months)])
        clamped_days = np.minimum(days, last_days)

        # rebuild the date portion
        new_date = (
            years.astype(str).str.zfill(4)
            + "-"
            + months.astype(str).str.zfill(2)
            + "-"
            + pd.Series(clamped_days, index=years.index).astype(str).str.zfill(2)
        )

        # handle the time portion, ensuring validity
        remainder = x[valid_mask].str.slice(10)

        time_regex = r"^[ T]?(\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
        valid_time = remainder.str.extract(time_regex, expand=False)
        valid_time = valid_time.fillna("00:00:00")
        valid_time = " " + valid_time

        new_date = new_date + valid_time
        x.loc[valid_mask] = new_date

    x = pd.to_datetime(x, errors="coerce")
    x = _clip_datetime(x, stats)
    return x.astype("datetime64[ns]")
