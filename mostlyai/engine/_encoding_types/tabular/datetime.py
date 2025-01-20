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
Datetime encoding splits a datetime into its parts, and encodes each part separately. For each part its
minimum and maximum values are determined. Any part is then encoded as `x - min_value`, resulting in integers
ranging from 0 to `max_value - min_value`. Thus, the corresponding cardinality is `max_value + 1 - min_value`.
"""

import calendar

import numpy as np
import pandas as pd
from dateutil import parser  # type: ignore

from mostlyai.engine._common import safe_convert_datetime
from mostlyai.engine._dtypes import is_date_dtype, is_timestamp_dtype

DATETIME_PARTS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "ms_E2",
    "ms_E1",
    "ms_E0",
]


def analyze_datetime(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    values = safe_convert_datetime(values)
    df = pd.concat([root_keys, values], axis=1)
    # determine lowest/highest values by root ID, and return Top 10
    min_dates = df.groupby(root_keys.name)[values.name].min().dropna()
    min11 = min_dates.sort_values(ascending=True).head(11).astype(str).tolist()
    max_dates = df.groupby(root_keys.name)[values.name].max().dropna()
    max11 = max_dates.sort_values(ascending=False).head(11).astype(str).tolist()
    # split into datetime parts
    df_split = split_sub_columns_datetime(values)
    is_not_nan = df_split["nan"] == 0
    has_nan = any(df_split["nan"] == 1)
    # extract min/max value for each part to determine valid value range
    if any(is_not_nan):
        min_values = {k: int(df_split[k][is_not_nan].min()) for k in DATETIME_PARTS}
        max_values = {k: int(df_split[k][is_not_nan].max()) for k in DATETIME_PARTS}
    else:
        def_values = {"year": 2022, "month": 1, "day": 1}
        min_values = {k: 0 for k in DATETIME_PARTS} | def_values
        max_values = {k: 0 for k in DATETIME_PARTS} | def_values
    # return stats
    stats = {
        "has_nan": has_nan,
        "min_values": min_values,
        "max_values": max_values,
        "min11": min11,
        "max11": max11,
    }
    return stats


def analyze_reduce_datetime(stats_list: list[dict], value_protection: bool = True) -> dict:
    # check if there are missing values
    has_nan = any([j["has_nan"] for j in stats_list])
    # determine min/max values for each part
    keys = stats_list[0]["min_values"].keys()
    min_values = {k: min([j["min_values"][k] for j in stats_list]) for k in keys}
    max_values = {k: max([j["max_values"][k] for j in stats_list]) for k in keys}
    # check if any record has non-zero timestamp information
    has_time = max_values["hour"] > 0 or max_values["minute"] > 0 or max_values["second"] > 0
    has_ms = has_time and (max_values["ms_E2"] > 0 or max_values["ms_E1"] > 0 or max_values["ms_E0"] > 0)
    # determine min / max 5 values to map too low / too high values to
    min11 = sorted([v for min11 in [j["min11"] for j in stats_list] for v in min11], reverse=False)[:11]
    max11 = sorted([v for max11 in [j["max11"] for j in stats_list] for v in max11], reverse=True)[:11]
    if value_protection:
        # extreme value protection - discard lowest/highest 5 values
        if len(min11) < 11 or len(max11) < 11:
            # less than 11 subjects with non-NULL values; we need to protect all
            min5 = []
            max5 = []
            has_time = False
            has_ms = False
        else:
            min5 = [str(v) for v in min11[5:10]]  # drop 1 to 5th lowest; keep 6th to 10th lowest
            max5 = [str(v) for v in max11[5:10]]  # drop 1 to 5th highest; keep 6th to 10th highest
            # update min/max year based on first four letters of protected min/max dates
            max_values["year"] = int(max5[0][0:4])
            min_values["year"] = int(min5[0][0:4])
    else:
        min5 = min11[0:4]
        max5 = max11[0:4]
    # determine cardinalities
    cardinalities = {}
    if has_nan:
        cardinalities["nan"] = 2  # binary
    cardinalities["year"] = max_values["year"] + 1 - min_values["year"]
    cardinalities["month"] = max_values["month"] + 1 - min_values["month"]
    cardinalities["day"] = max_values["day"] + 1 - min_values["day"]
    if has_time:
        cardinalities["hour"] = max_values["hour"] + 1 - min_values["hour"]
        cardinalities["minute"] = max_values["minute"] + 1 - min_values["minute"]
        cardinalities["second"] = max_values["second"] + 1 - min_values["second"]
    if has_ms:
        cardinalities["ms_E2"] = max_values["ms_E2"] + 1 - min_values["ms_E2"]
        cardinalities["ms_E1"] = max_values["ms_E1"] + 1 - min_values["ms_E1"]
        cardinalities["ms_E0"] = max_values["ms_E0"] + 1 - min_values["ms_E0"]
    stats = {
        "cardinalities": cardinalities,
        "has_nan": has_nan,
        "has_time": has_time,
        "has_ms": has_ms,
        "min_values": min_values,
        "max_values": max_values,
        "min5": min5,
        "max5": max5,
    }
    return stats


def encode_datetime(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.DataFrame:
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
    # split to sub_columns
    df = split_sub_columns_datetime(values)
    is_not_nan = df["nan"] == 0
    # encode values so that each datetime part ranges from 0 to `max_value-min_value`
    for key in DATETIME_PARTS:
        # subtract minimum value
        df[key] = df[key].where(~is_not_nan, df[key] - stats["min_values"][key])
        # ensure that any value is mapped onto valid value range
        df[key] = np.minimum(df[key], stats["max_values"][key] - stats["min_values"][key])
        df[key] = np.maximum(df[key], 0)
    if not stats["has_nan"]:
        df.drop(["nan"], inplace=True, axis=1)
    if not stats["has_time"]:
        df.drop(["hour", "minute", "second"], inplace=True, axis=1)
    if not stats["has_ms"]:
        df.drop(["ms_E2", "ms_E1", "ms_E0"], inplace=True, axis=1)
    return df


def split_sub_columns_datetime(values: pd.Series) -> pd.DataFrame:
    if not is_date_dtype(values) and not is_timestamp_dtype(values):
        raise ValueError("expected to be datetime")
    values = values.astype("datetime64[us]")

    # fast datetime part extraction taken from https://stackoverflow.com/a/56260054
    dt = values.to_numpy()
    parts = np.empty(dt.shape + (7,), dtype="u4")
    year, month, day, hour, minute, second = (dt.astype(f"M8[{x}]") for x in "YMDhms")
    parts[:, 0] = year + 1970  # Gregorian Year
    parts[:, 1] = (month - year) + 1  # month
    parts[:, 2] = (day - month) + 1  # dat
    parts[:, 3] = (dt - day).astype("m8[h]")  # hour
    parts[:, 4] = (dt - hour).astype("m8[m]")  # minute
    parts[:, 5] = (dt - minute).astype("m8[s]")  # second
    parts[:, 6] = (dt - second).astype("m8[us]")  # microsecond
    # create pd.DataFrame with datetime parts
    sub_columns = {
        "nan": values.reset_index(drop=True).isna(),
        "year": pd.Series(parts[:, 0]),
        "month": pd.Series(parts[:, 1]),
        "day": pd.Series(parts[:, 2]),
        "hour": pd.Series(parts[:, 3]),
        "minute": pd.Series(parts[:, 4]),
        "second": pd.Series(parts[:, 5]),
        "ms_E2": pd.Series(np.floor(parts[:, 6] / 100_000) % 10),
        "ms_E1": pd.Series(np.floor((parts[:, 6] / 10_000) % 10)),
        "ms_E0": pd.Series(np.floor((parts[:, 6] / 1_000) % 10)),
    }
    df = pd.DataFrame(sub_columns)
    df = df.fillna(0)
    df = df.astype("int")
    return df


def decode_datetime(df_encoded: pd.DataFrame, stats: dict):
    # decode y/m/d components
    y = df_encoded["year"] + stats["min_values"]["year"]
    m = df_encoded["month"] + stats["min_values"]["month"]
    d = df_encoded["day"] + stats["min_values"]["day"]
    # fix invalid dates by setting these to last day of month
    is_leap = y.apply(lambda x: calendar.isleap(x))
    d[is_leap & (m == 2) & (d > 29)] = 29
    d[~is_leap & (m == 2) & (d > 28)] = 28
    d[((m == 4) | (m == 6) | (m == 9) | (m == 11)) & (d > 30)] = 30
    # concatenate to datetime string
    y = y.astype(str)
    m = m.astype(str).str.zfill(2)
    d = d.astype(str).str.zfill(2)
    dt_format = "%Y-%m-%d"
    values = y + "-" + m + "-" + d
    if stats["has_time"]:
        hh = (df_encoded["hour"] + stats["min_values"]["hour"]).astype(str).str.zfill(2)
        mm = (df_encoded["minute"] + stats["min_values"]["minute"]).astype(str).str.zfill(2)
        ss = (df_encoded["second"] + stats["min_values"]["second"]).astype(str).str.zfill(2)
        dt_format += " %H:%M:%S"
        values = values + " " + hh + ":" + mm + ":" + ss
    if stats["has_ms"]:
        ms2 = df_encoded["ms_E2"] + stats["min_values"]["ms_E2"]
        ms1 = df_encoded["ms_E1"] + stats["min_values"]["ms_E1"]
        ms0 = df_encoded["ms_E0"] + stats["min_values"]["ms_E0"]
        ms = (100 * ms2 + 10 * ms1 + ms0).astype(str).str.zfill(3)
        dt_format += ".%f"
        values = values + "." + ms
    if "nan" in df_encoded.columns:
        values[df_encoded["nan"] == 1] = pd.NA
    # replace extreme values with randomly sampled 5-th to 10-th largest/smallest values
    if len(stats["min5"]) > 0 and len(stats["max5"]) > 0:
        # format datetime with accordance to the expected unified format when reading from stats
        min5 = [parser.parse(i).strftime(dt_format) for i in stats["min5"]]
        max5 = [parser.parse(i).strftime(dt_format) for i in stats["max5"]]
        is_too_low = values.notna() & (values < min5[0])
        is_too_high = values.notna() & (values > max5[0])
        values.loc[is_too_low] = np.random.choice(min5, size=sum(is_too_low))
        values.loc[is_too_high] = np.random.choice(max5, size=sum(is_too_high))
    elif "nan" in df_encoded.columns:
        # set all values to NaN if no valid values were present
        values[df_encoded["nan"] == 0] = pd.NA
    # convert from string to datetime
    values = pd.to_datetime(values)
    if not stats["has_time"]:
        values = pd.to_datetime(values.dt.date)
    return values
