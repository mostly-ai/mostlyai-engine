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

from mostlyai.engine._common import safe_convert_datetime


def analyze_language_datetime(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    values = safe_convert_datetime(values)
    df = pd.concat([root_keys, values], axis=1)
    # determine lowest/highest values by root ID, and return Top 10
    min_dates = df.groupby(root_keys.name)[values.name].min().dropna()
    min11 = min_dates.sort_values(ascending=True).head(11).astype(str).tolist()
    max_dates = df.groupby(root_keys.name)[values.name].max().dropna()
    max11 = max_dates.sort_values(ascending=False).head(11).astype(str).tolist()
    # determine if there are any NaN values
    has_nan = bool(values.isna().any())
    # return stats
    stats = {
        "has_nan": has_nan,
        "min11": min11,
        "max11": max11,
    }
    return stats


def analyze_reduce_language_datetime(stats_list: list[dict], value_protection: bool = True) -> dict:
    # check if there are missing values
    has_nan = any([j["has_nan"] for j in stats_list])
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
            min5 = [str(v) for v in min11[5:10]]  # drop 1 to 5th lowest; keep 6th to 10th lowest
            max5 = [str(v) for v in max11[5:10]]  # drop 1 to 5th highest; keep 6th to 10th highest
    else:
        min5 = min11[0:4]
        max5 = max11[0:4]
    stats = {
        "has_nan": has_nan,
        "min5": min5,
        "max5": max5,
    }
    return stats


def encode_language_datetime(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.Series:
    # convert
    values = safe_convert_datetime(values)
    values = values.copy()
    # reset index, as `values.mask` can throw errors for misaligned indices
    values.reset_index(drop=True, inplace=True)
    # replace extreme values with randomly sampled 5-th to 10-th largest/smallest values
    min5 = stats["min5"] if len(stats["min5"]) > 0 else [0]
    max5 = stats["max5"] if len(stats["max5"]) > 0 else [0]
    min5 = pd.Series(min5, dtype=values.dtype)
    max5 = pd.Series(max5, dtype=values.dtype)
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


def _clip_datetime(x: pd.Series, min5: list, max5: list) -> pd.Series:
    x_dt = pd.to_datetime(x, errors="coerce")
    min_arr = pd.to_datetime(min5).to_numpy(dtype="datetime64[ns]")
    max_arr = pd.to_datetime(max5).to_numpy(dtype="datetime64[ns]")
    n = len(x_dt)
    random_mins = np.random.choice(min_arr, size=n)
    random_maxs = np.random.choice(max_arr, size=n)
    clipped = np.minimum(np.maximum(x_dt.to_numpy(dtype="datetime64[ns]"), random_mins), random_maxs)
    return pd.Series(clipped, index=x.index)


def decode_datetime(x: pd.Series, col_stats: dict[str, str]) -> pd.Series:
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
    x = _clip_datetime(x, col_stats["min5"], col_stats["max5"])
    return x.astype("datetime64[ns]")
