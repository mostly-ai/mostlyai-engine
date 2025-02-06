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
from mostlyai.engine._encoding_types.tabular.numeric import split_sub_columns_digit, NUMERIC_DIGIT_MAX_DECIMAL
from mostlyai.engine.domain import ModelEncodingType


def analyze_language_numeric(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    values = safe_convert_numeric(values)

    # determine lowest/highest values by root ID, and return Top 10
    df = pd.concat([root_keys, values], axis=1)
    min_values = df.groupby(root_keys.name)[values.name].min().dropna()
    min11 = min_values.sort_values(ascending=True).head(11).astype("float").tolist()
    max_values = df.groupby(root_keys.name)[values.name].max().dropna()
    max11 = max_values.sort_values(ascending=False).head(11).astype("float").tolist()

    # split values into digits; used for digit numeric encoding, plus to determine precision
    df_split = split_sub_columns_digit(values)
    is_not_nan = df_split["nan"] == 0
    has_nan = sum(df_split["nan"]) > 0
    has_neg = sum(df_split["neg"]) > 0

    # extract min/max digit for each position to determine valid value range for digit encoding
    if any(is_not_nan):
        min_digits = {k: int(df_split[k][is_not_nan].min()) for k in df_split if k.startswith("E")}
        max_digits = {k: int(df_split[k][is_not_nan].max()) for k in df_split if k.startswith("E")}
    else:
        min_digits = {k: 0 for k in df_split if k.startswith("E")}
        max_digits = {k: 0 for k in df_split if k.startswith("E")}

    # return stats
    stats = {
        "has_nan": has_nan,
        "has_neg": has_neg,
        "min_digits": min_digits,
        "max_digits": max_digits,
        "min11": min11,
        "max11": max11,
    }
    return stats


def analyze_reduce_language_numeric(stats_list: list[dict], value_protection: bool = True) -> dict:
    # check for occurrence of NaN values
    has_nan = any([j["has_nan"] for j in stats_list])
    # check if there are negative values
    has_neg = any([j["has_neg"] for j in stats_list])

    # determine precision to apply rounding of sampled values during generation
    keys = stats_list[0]["max_digits"].keys()
    min_digits = {k: min([j["min_digits"][k] for j in stats_list]) for k in keys}
    max_digits = {k: max([j["max_digits"][k] for j in stats_list]) for k in keys}
    non_zero_prec = [k for k in keys if max_digits[k] > 0 and k.startswith("E")]
    min_decimal = min([int(k[1:]) for k in non_zero_prec]) if len(non_zero_prec) > 0 else 0

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

    if len(min5) > 0 or len(max5) > 0:
        max_abs = np.max(np.abs(np.array([min5[0], max5[0]])))
        max_decimal = int(np.floor(np.log10(max_abs))) if max_abs >= 10 else 0
    else:
        max_decimal = 0
    # don't allow more digits than the capped value for it
    decimal_cap = [d[1:] for d in keys][0]
    decimal_cap = int(decimal_cap) if decimal_cap.isnumeric() else NUMERIC_DIGIT_MAX_DECIMAL
    max_decimal = min(max(min_decimal, max_decimal), decimal_cap)

    stats = {
        "encoding_type": ModelEncodingType.language_numeric_digit.value,
        "has_nan": has_nan,
        "has_neg": has_neg,
        "min_digits": min_digits,
        "max_digits": max_digits,
        "max_decimal": max_decimal,
        "min_decimal": min_decimal,
        "min5": min5,
        "max5": max5,
    }

    return stats
