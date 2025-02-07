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

import pandas as pd

from mostlyai.engine._common import safe_convert_datetime
from mostlyai.engine._encoding_types.tabular.datetime import split_sub_columns_datetime, DATETIME_PARTS


def analyze_language_datetime(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
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


def analyze_reduce_language_datetime(stats_list: list[dict], value_protection: bool = True) -> dict:
    # check if there are missing values
    has_nan = any([j["has_nan"] for j in stats_list])
    # determine min/max values for each part
    keys = stats_list[0]["min_values"].keys()
    min_values = {k: min([j["min_values"][k] for j in stats_list]) for k in keys}
    max_values = {k: max([j["max_values"][k] for j in stats_list]) for k in keys}
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
            # update min/max year based on first four letters of protected min/max dates
            max_values["year"] = int(max5[0][0:4])
            min_values["year"] = int(min5[0][0:4])
    else:
        min5 = min11[0:4]
        max5 = max11[0:4]
    stats = {
        "has_nan": has_nan,
        "min5": min5,
        "max5": max5,
    }
    return stats
