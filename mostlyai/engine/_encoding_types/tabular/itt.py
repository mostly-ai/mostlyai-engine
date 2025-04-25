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
ITT encoding is only applicable for sequential data, with a context_key being present. For each context_key, the
datetime values are transformed into ITTs, and the first non-missing date is determined. That start_date is then split
into its datetime parts, and the ITTs are split into its own separate components. For each part its
minimum and maximum values are determined. Any part is then encoded as `x - min_value`, resulting in integers
ranging from 0 to `max_value - min_value`. Thus, the corresponding cardinality is `max_value + 1 - min_value`.
"""

import calendar
from datetime import timedelta

import numpy as np
import pandas as pd

from mostlyai.engine._common import (
    ANALYZE_MIN_MAX_TOP_N,
    ANALYZE_REDUCE_MIN_MAX_N,
    dp_quantiles,
    get_stochastic_rare_threshold,
    safe_convert_datetime,
)
from mostlyai.engine._dtypes import is_date_dtype, is_timestamp_dtype
from mostlyai.engine._encoding_types.tabular.datetime import split_sub_columns_datetime


def analyze_itt(
    values: pd.Series,
    root_keys: pd.Series,
    context_keys: pd.Series,
) -> dict:
    values = safe_convert_datetime(values)
    df = pd.concat([root_keys, context_keys, values], axis=1)
    # calculate min/max values for start dates
    start_dates = df.dropna().groupby(root_keys.name)[values.name].nth(0)
    min_n = start_dates.sort_values(ascending=True).head(ANALYZE_MIN_MAX_TOP_N).astype(str).tolist()
    start_dates = df.dropna().groupby(root_keys.name)[values.name].nth(0)
    max_n = start_dates.sort_values(ascending=False).head(ANALYZE_MIN_MAX_TOP_N).astype(str).tolist()
    # split into datetime/ITT parts
    df_split = split_sub_columns_itt(values, context_keys)
    is_not_nan = df_split["nan"] == 0
    has_nan = any(df_split["nan"] == 1)
    has_neg = sum(df_split["itt_neg"]) > 0
    # extract min/max value for each part to determine valid value range;
    # for ITT parts we need to discard first entry per sequence as these have dummy `0` inserted
    keys = [k for k in df_split if k not in ["nan", "itt_neg"]]
    keys_start = [k for k in keys if k.startswith("start_")]
    keys_itt = [k for k in keys if k.startswith("itt_")]
    if any(is_not_nan):
        # extract min/max for start_date parts
        min_values_start = {k: int(df_split[k][is_not_nan].min()) for k in keys_start}
        max_values_start = {k: int(df_split[k][is_not_nan].max()) for k in keys_start}
        # extract min/max for ITT parts
        df_split = pd.concat([context_keys, df_split], axis=1)
        df_split = df_split.loc[df_split.groupby(context_keys.name).cumcount() > 0, :]
        if not df_split.empty:
            min_values_itt = {k: int(df_split[k][is_not_nan].min()) for k in keys_itt}
            max_values_itt = {k: int(df_split[k][is_not_nan].max()) for k in keys_itt}
        else:
            min_values_itt = {k: 0 for k in keys_itt}
            max_values_itt = {k: 0 for k in keys_itt}
        # merge together
        min_values = min_values_start | min_values_itt
        max_values = max_values_start | max_values_itt
    else:
        def_values = {"start_year": 2022, "start_month": 1, "start_day": 1}
        min_values = {k: 0 for k in keys} | def_values
        max_values = {k: 0 for k in keys} | def_values
    # return stats
    stats = {
        "has_nan": has_nan,
        "has_neg": has_neg,
        "min_values": min_values,
        "max_values": max_values,
        "min_n": min_n,
        "max_n": max_n,
    }
    return stats


def analyze_reduce_itt(
    stats_list: list[dict],
    value_protection: bool = True,
    value_protection_epsilon: float | None = None,
    value_protection_delta: float | None = None,
) -> dict:
    # check if there are missing values
    has_nan = any([j["has_nan"] for j in stats_list])
    # check if there are negative values
    has_neg = any([j["has_neg"] for j in stats_list])
    # determine min/max values for each part
    keys = stats_list[0]["min_values"].keys()
    min_values = {k: min([j["min_values"][k] for j in stats_list]) for k in keys}
    max_values = {k: max([j["max_values"][k] for j in stats_list]) for k in keys}
    # check if any record has non-zero timestamp information
    has_time = max_values["start_hour"] > 0 or max_values["start_minute"] > 0 or max_values["start_second"] > 0
    # determine min / max 5 values to map too low / too high values to
    reduced_min_n = sorted([v for min_n in [j["min_n"] for j in stats_list] for v in min_n], reverse=False)
    reduced_max_n = sorted([v for max_n in [j["max_n"] for j in stats_list] for v in max_n], reverse=True)
    if value_protection:
        if len(reduced_min_n) < ANALYZE_REDUCE_MIN_MAX_N or len(reduced_max_n) < ANALYZE_REDUCE_MIN_MAX_N:
            # protect all values if there are less than ANALYZE_REDUCE_MIN_MAX_N values
            reduced_min = None
            reduced_max = None
            has_time = False
        else:
            if value_protection_delta is not None and value_protection_epsilon is not None:
                values = sorted(reduced_min_n + reduced_max_n)
                quantiles = [0.01, 0.99] if len(values) >= 10_000 else [0.05, 0.95]
                reduced_min, reduced_max = dp_quantiles(
                    values, quantiles, value_protection_epsilon, value_protection_delta
                )
                reduced_min = str(reduced_min)
                reduced_max = str(reduced_max)
            else:
                reduced_min = str(reduced_min_n[get_stochastic_rare_threshold(min_threshold=5)])
                reduced_max = str(reduced_max_n[get_stochastic_rare_threshold(min_threshold=5)])
                # update min/max year based on first four letters of protected min/max dates
                max_values["start_year"] = int(reduced_max[0:4])
                min_values["start_year"] = int(reduced_min[0:4])
    else:
        reduced_min = str(reduced_min_n[0]) if len(reduced_min_n) > 0 else None
        reduced_max = str(reduced_max_n[0]) if len(reduced_max_n) > 0 else None

    # determine cardinalities
    cardinalities = {}
    if has_nan:
        cardinalities["nan"] = 2  # binary
    # start date
    cardinalities["start_year"] = max_values["start_year"] + 1 - min_values["start_year"]
    cardinalities["start_month"] = max_values["start_month"] + 1 - min_values["start_month"]
    cardinalities["start_day"] = max_values["start_day"] + 1 - min_values["start_day"]
    if has_time:
        cardinalities["start_hour"] = max_values["start_hour"] + 1 - min_values["start_hour"]
        cardinalities["start_minute"] = max_values["start_minute"] + 1 - min_values["start_minute"]
        cardinalities["start_second"] = max_values["start_second"] + 1 - min_values["start_second"]
    # ITT
    if has_neg:
        cardinalities["itt_neg"] = 2
    cardinalities["itt_week"] = max_values["itt_week"] + 1 - min_values["itt_week"]
    cardinalities["itt_day"] = max_values["itt_day"] + 1 - min_values["itt_day"]
    if has_time:
        cardinalities["itt_hour"] = max_values["itt_hour"] + 1 - min_values["itt_hour"]
        cardinalities["itt_minute"] = max_values["itt_minute"] + 1 - min_values["itt_minute"]
        cardinalities["itt_second"] = max_values["itt_second"] + 1 - min_values["itt_second"]

    stats = {
        "cardinalities": cardinalities,
        "has_nan": has_nan,
        "has_neg": has_neg,
        "has_time": has_time,
        "min_values": min_values,
        "max_values": max_values,
        "min": reduced_min,
        "max": reduced_max,
    }
    return stats


def encode_itt(
    values: pd.Series,
    stats: dict,
    context_keys: pd.Series,
) -> pd.DataFrame:
    # convert
    values = safe_convert_datetime(values)
    # split to sub_columns
    df = split_sub_columns_itt(values, context_keys)
    # encode values so that each datetime part ranges from 0 to `max_value-min_value`
    start_parts = [k for k in df.columns if k.startswith("start_")]
    itt_parts = [k for k in df.columns if k.startswith("itt_") if k != "itt_neg"]
    for key in start_parts + itt_parts:
        # subtract minimum value
        df[key] = df[key] - stats["min_values"][key]
        # ensure that any value is mapped onto valid value range
        df[key] = np.minimum(df[key], stats["max_values"][key] - stats["min_values"][key])
        df[key] = np.maximum(df[key], 0)
    # remove unused columns
    if not stats["has_nan"]:
        df.drop(["nan"], inplace=True, axis=1)
    if not stats["has_time"]:
        df.drop(["start_hour", "start_minute", "start_second"], inplace=True, axis=1)
        df.drop(["itt_hour", "itt_minute", "itt_second"], inplace=True, axis=1)
    if not stats["has_neg"]:
        df.drop(["itt_neg"], inplace=True, axis=1)
    return df


def split_sub_columns_itt(
    values: pd.Series,
    context_keys: pd.Series,
) -> pd.DataFrame:
    if not is_date_dtype(values) and not is_timestamp_dtype(values):
        raise ValueError("expected to be datetime")
    values = values.astype("datetime64[us]")

    sub_columns = {
        "nan": values.isna(),
    }

    # fill NAs with valid dates; use minimum date as fallback, if a group has no dates at all;
    df = pd.concat([values, context_keys], axis=1)
    df[values.name] = df.groupby(df[context_keys.name])[values.name].ffill().bfill().fillna(values.dropna().min())

    # first convert itt column to datetime delta in [ms] stored as float64
    if not df.empty:
        itt_in_ms = df.groupby(context_keys.name)[values.name].diff().astype("timedelta64[ms]").fillna(timedelta(0))
    else:  # df.empty is True
        itt_in_ms = pd.Series(name=values.name, dtype="float64")

    itt_sub_columns = {
        "itt_neg": itt_in_ms < timedelta(0),
    }

    # convert positive number the negative raw differences to get the right differences
    itt_in_ms = np.abs(itt_in_ms)

    # calculate the ITT parts
    s_in_min = 60
    s_in_hour = 60 * 60
    s_in_day = 60 * 60 * 24
    s_in_week = 60 * 60 * 24 * 7
    itt_sub_columns["itt_week"] = (itt_in_ms / 1000) / s_in_week
    itt_sub_columns["itt_day"] = ((itt_in_ms / 1000) % s_in_week) / s_in_day
    itt_sub_columns["itt_hour"] = ((itt_in_ms / 1000) % s_in_day) / s_in_hour
    itt_sub_columns["itt_minute"] = ((itt_in_ms / 1000) % s_in_hour) / s_in_min
    itt_sub_columns["itt_second"] = (itt_in_ms / 1000) % s_in_min

    # copy the first non-null date as `start` to all events within the sequence
    if any(df[values.name].notna()):
        df[values.name] = df.groupby(df[context_keys.name])[values.name].transform(
            lambda series: series.loc[series.first_valid_index()]
        )

    # extract datetime parts from start_date
    date_sub_columns_df_split = split_sub_columns_datetime(df[values.name])
    date_sub_columns = {
        "start_year": date_sub_columns_df_split["year"],
        "start_month": date_sub_columns_df_split["month"],
        "start_day": date_sub_columns_df_split["day"],
        "start_hour": date_sub_columns_df_split["hour"],
        "start_minute": date_sub_columns_df_split["minute"],
        "start_second": date_sub_columns_df_split["second"],
    }

    df = pd.DataFrame(sub_columns | date_sub_columns | itt_sub_columns)
    df = df.fillna(0)
    df = df.astype("int")
    return df


def decode_itt(
    df_encoded: pd.DataFrame,
    stats: dict,
    context_keys: pd.Series,
    prev_steps: dict | None = None,
):
    min_values = stats["min_values"]
    prev_dts = (prev_steps or {}).get("prev_dts")

    ## decode `start` parts
    def decode_initial_starts():
        # decode y/m/d components
        y = df_encoded["start_year"] + min_values["start_year"]
        m = df_encoded["start_month"] + min_values["start_month"]
        d = df_encoded["start_day"] + min_values["start_day"]
        # fix invalid dates by setting these to last day of month
        is_leap = y.apply(lambda x: calendar.isleap(x))
        d[is_leap & (m == 2) & (d > 29)] = 29
        d[~is_leap & (m == 2) & (d > 28)] = 28
        d[((m == 4) | (m == 6) | (m == 9) | (m == 11)) & (d > 30)] = 30
        # concatenate to datetime string
        y = y.astype(str)
        m = m.astype(str).str.zfill(2)
        d = d.astype(str).str.zfill(2)
        starts = y + "-" + m + "-" + d
        if stats["has_time"]:
            hh = (df_encoded["start_hour"] + min_values["start_hour"]).astype(str).str.zfill(2)
            mm = (df_encoded["start_minute"] + min_values["start_minute"]).astype(str).str.zfill(2)
            ss = (df_encoded["start_second"] + min_values["start_second"]).astype(str).str.zfill(2)
            starts = starts + " " + hh + ":" + mm + ":" + ss
        # propagate the first non-null date per group to all events within that group
        starts.name = "__STARTS"
        starts = starts.loc[
            pd.concat([context_keys, starts], axis=1)
            .groupby(context_keys.name)[starts.name]
            .transform(pd.Series.first_valid_index)
        ]
        starts = starts.reset_index(drop=True)
        starts = safe_convert_datetime(starts)
        # clip start values to privacy-safe value range;
        # note, that final date values may fall out of original date range, as we prioritize retaining ITT properties
        if stats["min"] is not None and stats["max"] is not None:
            starts.loc[starts > stats["max"]] = stats["max"]
            starts.loc[starts < stats["min"]] = stats["min"]
        return starts

    def continue_starts():
        # select start for each context key
        starts = pd.merge(context_keys, prev_dts, left_on=context_keys.name, right_on="__CONTEXT_KEYS", how="left")
        starts = starts.reset_index(drop=True)
        starts = starts["__STARTS"]
        return starts

    starts = decode_initial_starts() if prev_dts is None else continue_starts()

    ## decode `itt` parts
    def decode_itts():
        itts = pd.Series(np.repeat(0, df_encoded.shape[0]), dtype="int")
        itts = itts + (df_encoded["itt_week"] + min_values["itt_week"]) * 60 * 60 * 24 * 7
        itts = itts + (df_encoded["itt_day"] + min_values["itt_day"]) * 60 * 60 * 24
        if stats["has_time"]:
            itts = itts + (df_encoded["itt_hour"] + min_values["itt_hour"]) * 60 * 60
            itts = itts + (df_encoded["itt_minute"] + min_values["itt_minute"]) * 60
            itts = itts + (df_encoded["itt_second"] + min_values["itt_second"])
        if stats["has_neg"]:
            itts = itts * pd.Series(np.where(df_encoded["itt_neg"] == 1, -1, 1))
        itts.name = "__ITTS"
        return itts

    itts = decode_itts()
    itts_df = pd.concat([context_keys, itts], axis=1)
    if prev_dts is None:
        # we cumulate ITTs for each context key, to then add these to start_date;
        # by definition, the first event of each sequence represents start_date, thus we set ITT = 0
        itts_df["__ITTS"] = itts_df["__ITTS"].mask(itts_df.groupby(context_keys.name).cumcount() == 0, 0)
    itts = itts_df.groupby(context_keys.name)["__ITTS"].cumsum()

    # add cumulative ITTs to start_dates
    values = starts.astype(np.int64) // 10**3 + itts * 10**3
    values = np.clip(
        a=values,
        a_min=pd.Timestamp.min.value // 10**6 + 1,
        a_max=pd.Timestamp.max.value // 10**6,
    )
    values = pd.to_datetime(values, unit="ms")

    # convert to datetime
    values = safe_convert_datetime(values, date_only=not stats["has_time"])

    # keep track of last decoded values for next iteration
    if prev_steps is not None:
        prev_steps["prev_dts"] = (
            pd.concat(
                [context_keys.rename("__CONTEXT_KEYS"), values.rename("__STARTS")],
                axis=1,
            )
            .groupby("__CONTEXT_KEYS")
            .last()
            .reset_index()
        )

    # add nan values
    if "nan" in df_encoded:
        values[df_encoded["nan"] == 1] = pd.NA
    return values
