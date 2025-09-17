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
Numeric encoding can be done in 3 distinct ways: numeric_discrete, numeric_binned or numeric_digit. Discrete treats
the numeric values basically as a categorical, that doesn't generate RARE tokens. Binned assigns each value to one of
the intervals within the valid value range, and then randomly draws from it during decoding. And digit splits the values
into separate sub columns, one for each digit position. In contrast to digit, discrete and binned use only one sub
column, and with that is significantly less compute intensive.

The encoding type numeric_auto allows the selection of the numeric encoding type to be based on a heuristic after the
analysis phase is done.
"""

import typing

import numpy as np
import pandas as pd

from mostlyai.engine._common import (
    ANALYZE_MIN_MAX_TOP_N,
    ANALYZE_REDUCE_MIN_MAX_N,
    compute_log_histogram,
    dp_approx_bounds,
    dp_non_rare,
    find_distinct_bins,
    get_stochastic_rare_threshold,
    safe_convert_numeric,
)
from mostlyai.engine._dtypes import is_float_dtype, is_integer_dtype
from mostlyai.engine._encoding_types.tabular.categorical import (
    CATEGORICAL_NULL_TOKEN,
    CATEGORICAL_SUB_COL_SUFFIX,
    CATEGORICAL_UNKNOWN_TOKEN,
    decode_categorical,
    encode_categorical,
)
from mostlyai.engine.domain import ModelEncodingType

# do not use numeric_discrete if we have too many distinct values
NUMERIC_AUTO_DISCRETE_MAX_VALUES = 100
# do not use numeric_discrete if we have too many rare values
NUMERIC_AUTO_DISCRETE_MIN_NON_RARE = 0.999
# do not use numeric digit if we have too many digit positions
NUMERIC_AUTO_DIGIT_MAX_DIGITS = 3

# max number of bins to use for binning
NUMERIC_BINNED_MAX_BINS = 100
# max number of quantiles to be calculated for each partition
NUMERIC_BINNED_MAX_QUANTILES = 1001
# suffix for numeric binned sub-column
NUMERIC_BINNED_SUB_COL_SUFFIX = "bin"
# special tokens for numeric binned encoding
NUMERIC_BINNED_UNKNOWN_TOKEN = "<<UNK>>"
NUMERIC_BINNED_NULL_TOKEN = "<<NULL>>"
NUMERIC_BINNED_MIN_TOKEN = "<<MIN>>"
NUMERIC_BINNED_MAX_TOKEN = "<<MAX>>"

# suffix for numeric discrete sub-column
NUMERIC_DISCRETE_SUB_COL_SUFFIX = CATEGORICAL_SUB_COL_SUFFIX
# special tokens for numeric discrete encoding
NUMERIC_DISCRETE_UNKNOWN_TOKEN = CATEGORICAL_UNKNOWN_TOKEN
NUMERIC_DISCRETE_NULL_TOKEN = CATEGORICAL_NULL_TOKEN

# maximum and minimum precision that is being considered
NUMERIC_DIGIT_MAX_DECIMAL = 18
NUMERIC_DIGIT_MIN_DECIMAL = -8


def _type_safe_numeric_series(numeric_array: np.ndarray | list, pd_dtype: str) -> pd.Series:
    # make a safe conversion using numpy's astype as an intermediary
    # and then pandas type to match values to pd_dtype
    np_dtype = int if pd_dtype == "Int64" else float
    i_min = np.iinfo(int).min
    i_max = np.iinfo(int).max

    def _clip_int(vals):
        # clip the array values to keep within representable boundaries in signed int representation
        return [min(max(int(v), i_min), i_max) for v in vals]

    if isinstance(numeric_array, list):
        if np_dtype is int:
            numeric_array = _clip_int(numeric_array)
        numeric_array = np.array(numeric_array)

    elif np_dtype is int:
        try:
            numeric_array.astype(int, casting="safe")
        except TypeError:
            # if it cannot be casted safely (e.g. without integer overflow)
            numeric_array = np.array(_clip_int(numeric_array))

    return pd.Series(np.array([v for v in numeric_array]).astype(np_dtype), dtype=pd_dtype)


def split_sub_columns_digit(
    values: pd.Series,
    max_decimal=NUMERIC_DIGIT_MAX_DECIMAL,
    min_decimal=NUMERIC_DIGIT_MIN_DECIMAL,
) -> pd.DataFrame:
    if not is_integer_dtype(values) and not is_float_dtype(values):
        raise ValueError("expected to be numeric")

    columns = [f"E{i}" for i in np.arange(max_decimal, min_decimal - 1, -1)]
    if values.isna().all():
        # handle special case when all values are nan
        df = pd.DataFrame({c: [0] * len(values) for c in columns})
    else:
        # convert to float64 as `np.format_float_positional` doesn't support Float64
        values = values.astype("float64")
        # rely on `np.format_float_positional` to determine string representation of absolute values
        values_str = (
            values.abs()
            .apply(lambda x: np.format_float_positional(x, unique=True, pad_left=50, pad_right=20, precision=20))
            # convert to string[pyarrow] for faster processing
            .astype("string[pyarrow]")
            # replace nan with pd.NA for faster processing
            .replace("nan", pd.NA)
        )
        values_str = values_str.str.replace(" ", "0")
        values_str = values_str.str.replace(".", "", n=1, regex=False)
        values_str = values_str.str[(49 - max_decimal) : (49 - min_decimal + 1)]
        df = values_str.str.split("", n=max_decimal - min_decimal + 2, expand=True)
        df = df.drop(columns=[0, max_decimal - min_decimal + 2])
        df = df.fillna("0")
        df.columns = columns
    df.insert(0, "nan", values.isna())
    df.insert(1, "neg", (~values.isna()) & (values < 0))
    df = df.astype("int")
    return df


def analyze_numeric(
    values: pd.Series,
    root_keys: pd.Series,
    _: pd.Series | None = None,
    encoding_type: ModelEncodingType = ModelEncodingType.tabular_numeric_auto,
) -> dict:
    values = safe_convert_numeric(values)
    non_na_values = values.dropna()
    cnt_unique_values = non_na_values.nunique()

    # compute log histogram for DP bounds
    log_hist = compute_log_histogram(non_na_values)

    # determine sufficient quantiles; used for binned numeric encoding
    if (
        encoding_type in [ModelEncodingType.tabular_numeric_binned, ModelEncodingType.tabular_numeric_auto]
        and len(non_na_values) > 0
    ):
        quantiles = np.quantile(
            values.dropna(),
            np.linspace(0, 1, NUMERIC_BINNED_MAX_QUANTILES),
            method="closest_observation",
        ).tolist()
    else:
        quantiles = None

    # for each unique value count distinct root_keys; used for discrete numeric encoding
    if encoding_type == ModelEncodingType.tabular_numeric_discrete or (
        encoding_type == ModelEncodingType.tabular_numeric_auto and cnt_unique_values < NUMERIC_AUTO_DISCRETE_MAX_VALUES
    ):
        df = pd.concat([root_keys, values], axis=1)
        cnt_values = df.groupby(values.name)[root_keys.name].nunique().to_dict()
    else:
        # do not count values, if there are too many
        cnt_values = None

    # determine lowest/highest values by root ID, and return top ANALYZE_MIN_MAX_TOP_N
    df = pd.concat([root_keys, values], axis=1)
    min_values = df.groupby(root_keys.name)[values.name].min().dropna()
    min_n = min_values.sort_values(ascending=True).head(ANALYZE_MIN_MAX_TOP_N).astype("float").tolist()
    max_values = df.groupby(root_keys.name)[values.name].max().dropna()
    max_n = max_values.sort_values(ascending=False).head(ANALYZE_MIN_MAX_TOP_N).astype("float").tolist()

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
        "min_n": min_n,
        "max_n": max_n,
        "cnt_values": cnt_values,
        "quantiles": quantiles,
        "log_hist": log_hist,
    }
    return stats


def analyze_reduce_numeric(
    stats_list: list[dict],
    value_protection: bool = True,
    value_protection_epsilon: float | None = None,
    encoding_type: ModelEncodingType | None = ModelEncodingType.tabular_numeric_auto,
) -> dict:
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

    reduced_min_n = sorted([v for min_n in [j["min_n"] for j in stats_list] for v in min_n], reverse=False)
    reduced_max_n = sorted([v for max_n in [j["max_n"] for j in stats_list] for v in max_n], reverse=True)
    if value_protection:
        if len(reduced_min_n) < ANALYZE_REDUCE_MIN_MAX_N or len(reduced_max_n) < ANALYZE_REDUCE_MIN_MAX_N:
            # protect all values if there are less than ANALYZE_REDUCE_MIN_MAX_N values
            reduced_min = None
            reduced_max = None
        else:
            if value_protection_epsilon is not None:
                # Sum up log histograms bin-wise from all partitions
                log_hist = [sum(bin) for bin in zip(*[j["log_hist"] for j in stats_list])]
                reduced_min, reduced_max = dp_approx_bounds(log_hist, value_protection_epsilon)
            else:
                reduced_min = reduced_min_n[get_stochastic_rare_threshold(min_threshold=5)]
                reduced_max = reduced_max_n[get_stochastic_rare_threshold(min_threshold=5)]
    else:
        reduced_min = reduced_min_n[0] if len(reduced_min_n) > 0 else None
        reduced_max = reduced_max_n[0] if len(reduced_max_n) > 0 else None

    if reduced_min is not None or reduced_max is not None:
        max_abs = np.max(np.abs(np.array([reduced_min, reduced_max])))
        max_decimal = int(np.floor(np.log10(max_abs))) if max_abs >= 10 else 0
    else:
        max_decimal = 0
    # don't allow more digits than the capped value for it
    decimal_cap = [d[1:] for d in keys][0]
    decimal_cap = int(decimal_cap) if decimal_cap.isnumeric() else NUMERIC_DIGIT_MAX_DECIMAL
    max_decimal = min(max(min_decimal, max_decimal), decimal_cap)

    # sum up cnt_values (if they exist for all partitions)
    has_cnt_values = all([j["cnt_values"] for j in stats_list])
    if has_cnt_values:
        # sum up all counts for each categorical value
        cnt_values: dict[str, int] = {}
        for item in stats_list:
            for value, count in item["cnt_values"].items():
                cnt_values[value] = cnt_values.get(value, 0) + count
        cnt_total = sum(cnt_values.values())
        # apply rare value protection
        if value_protection:
            if value_protection_epsilon is not None:
                categories, non_rare_ratio = dp_non_rare(cnt_values, value_protection_epsilon, threshold=5)
            else:
                rare_min = get_stochastic_rare_threshold(min_threshold=5)
                cnt_values = {c: v for c, v in cnt_values.items() if v >= rare_min}
                categories = list(cnt_values.keys())
                non_rare_ratio = sum(cnt_values.values()) / cnt_total
        else:
            categories = list(cnt_values.keys())
            non_rare_ratio = 1.0
    else:
        categories = []
        non_rare_ratio = 1.0

    # auto heuristic
    if encoding_type == ModelEncodingType.tabular_numeric_auto:
        if non_rare_ratio > NUMERIC_AUTO_DISCRETE_MIN_NON_RARE:
            encoding_type = ModelEncodingType.tabular_numeric_discrete
        elif len(non_zero_prec) <= NUMERIC_AUTO_DIGIT_MAX_DIGITS:
            encoding_type = ModelEncodingType.tabular_numeric_digit
        else:
            encoding_type = ModelEncodingType.tabular_numeric_binned

    if encoding_type == ModelEncodingType.tabular_numeric_discrete:
        # add NULL token if NaN values exist
        if has_nan:
            categories = [NUMERIC_DISCRETE_NULL_TOKEN] + categories
        # add unknown/rare token
        categories = [NUMERIC_DISCRETE_UNKNOWN_TOKEN] + categories
        stats = {
            "encoding_type": ModelEncodingType.tabular_numeric_discrete.value,
            "cardinalities": {CATEGORICAL_SUB_COL_SUFFIX: len(categories)},
            "codes": {categories[i]: i for i in range(len(categories))},
            "min_decimal": min_decimal,
        }

    elif encoding_type == ModelEncodingType.tabular_numeric_digit:
        cardinalities = {}
        if has_nan:
            cardinalities["nan"] = 2  # binary
        if has_neg:
            cardinalities["neg"] = 2  # binary
        for d in np.arange(max_decimal, min_decimal - 1, -1):
            # each digit will be encoded to `[0, max_digit-min_digit]`, thus the cardinality is `max_digit+1-min_digit`;
            cardinalities[f"E{d}"] = max_digits[f"E{d}"] + 1 - min_digits[f"E{d}"]
        stats = {
            "encoding_type": ModelEncodingType.tabular_numeric_digit.value,
            "cardinalities": cardinalities,
            "has_nan": has_nan,
            "has_neg": has_neg,
            "min_digits": min_digits,
            "max_digits": max_digits,
            "max_decimal": max_decimal,
            "min_decimal": min_decimal,
            "min": reduced_min,
            "max": reduced_max,
        }

    elif encoding_type == ModelEncodingType.tabular_numeric_binned:
        if reduced_min is None or reduced_max is None:
            # handle edge case where all values are privacy protected
            bins = [0]
            min_decimal = 0
        else:
            if value_protection_epsilon is None:
                quantiles = np.concatenate([j["quantiles"] for j in stats_list if j["quantiles"]])
                quantiles = list(np.clip(quantiles, reduced_min, reduced_max))
                bins = find_distinct_bins(quantiles, NUMERIC_BINNED_MAX_BINS)
            else:
                # use linear interpolation between the value-protected min and max
                # to avoid spending privacy budget on calculating all those quantiles
                bins = list(np.linspace(reduced_min, reduced_max, NUMERIC_BINNED_MAX_BINS + 1))
        # add unknown/rare token
        categories = [NUMERIC_BINNED_UNKNOWN_TOKEN]
        # add NULL token if NaN values exist
        if has_nan:
            categories += [NUMERIC_BINNED_NULL_TOKEN]

        # add min/max tokens if min/max are not rare
        # FIXME: what should the new behavior be?
        if reduced_min is not None:
            categories += [NUMERIC_BINNED_MIN_TOKEN]
        if reduced_max is not None:
            categories += [NUMERIC_BINNED_MAX_TOKEN]
        stats = {
            "encoding_type": ModelEncodingType.tabular_numeric_binned.value,
            "cardinalities": {NUMERIC_BINNED_SUB_COL_SUFFIX: len(categories) + len(bins) - 1},
            "codes": {categories[i]: i for i in range(len(categories))},
            "min_decimal": min_decimal,
            "bins": bins,
        }
    else:
        raise ValueError(f"Unknown encoding type {encoding_type}")
    return stats


def encode_numeric(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.DataFrame:
    if stats["encoding_type"] == ModelEncodingType.tabular_numeric_discrete:
        df = _encode_numeric_discrete(values, stats)
    elif stats["encoding_type"] == ModelEncodingType.tabular_numeric_digit:
        df = _encode_numeric_digit(values, stats)
    elif stats["encoding_type"] == ModelEncodingType.tabular_numeric_binned:
        df = _encode_numeric_binned(values, stats)
    else:
        raise ValueError(f"Unknown encoding type {stats['encoding_type']}")
    return df


def _encode_numeric_discrete(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.DataFrame:
    values = safe_convert_numeric(values)
    df = encode_categorical(values, stats)
    return df


def _encode_numeric_digit(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.DataFrame:
    values = safe_convert_numeric(values)
    # try to convert to int, if possible
    dtype = "Int64" if stats["min_decimal"] == 0 else "Float64"
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
    # replace extreme values with min/max
    if stats["min"] is not None:
        reduced_min = _type_safe_numeric_series([stats["min"]], dtype).iloc[0]
        values.loc[values < reduced_min] = reduced_min
    if stats["max"] is not None:
        reduced_max = _type_safe_numeric_series([stats["max"]], dtype).iloc[0]
        values.loc[values > reduced_max] = reduced_max
    # split to sub_columns
    df = split_sub_columns_digit(values, stats["max_decimal"], stats["min_decimal"])
    is_not_nan = df["nan"] == 0
    # normalize values to `[0, max_digit-min_digit]`
    for d in np.arange(stats["max_decimal"], stats["min_decimal"] - 1, -1):
        key = f"E{d}"
        # subtract minimum value
        df[key] = df[key].where(~is_not_nan, df[key] - stats["min_digits"][key])
        # ensure that any value is mapped onto valid value range
        df[key] = np.minimum(df[key], stats["max_digits"][key] - stats["min_digits"][key])
        df[key] = np.maximum(df[key], 0)

    # ensure that encoded digits are mapped onto valid value range
    for d in np.arange(stats["max_decimal"], stats["min_decimal"] - 1, -1):
        df[f"E{d}"] = np.minimum(df[f"E{d}"], stats["max_digits"][f"E{d}"])
    if not stats["has_nan"]:
        df.drop("nan", inplace=True, axis=1)
    if not stats["has_neg"]:
        df.drop("neg", inplace=True, axis=1)
    return df


def _encode_numeric_binned(values: pd.Series, stats: dict, _: pd.Series | None = None) -> pd.DataFrame:
    values = safe_convert_numeric(values)
    bins = stats["bins"].copy()
    min_value = bins[0]
    max_value = bins[-1]
    # we expand first bin edge to -inf, and last bin edge to +inf, to ensure that any value that is too low or too
    # high will be mapped to first or last bin, respectively
    bins[0] = -np.inf
    bins[-1] = np.inf
    codes = pd.Series(
        pd.cut(values, bins=bins, right=False).cat.codes,
        name=CATEGORICAL_SUB_COL_SUFFIX,
        index=values.index,
    )
    # offset codes by number of special tokens
    codes_bin_offset = len(stats["codes"])
    codes = codes + codes_bin_offset
    # explicitly encode NA values
    if NUMERIC_BINNED_NULL_TOKEN in stats["codes"]:
        codes.mask(values.isna(), stats["codes"][NUMERIC_BINNED_NULL_TOKEN], inplace=True)
    else:
        codes.mask(values.isna(), stats["codes"][NUMERIC_BINNED_UNKNOWN_TOKEN], inplace=True)
    # explicitly encode min/max values
    if NUMERIC_BINNED_MIN_TOKEN in stats["codes"]:
        codes.mask(values == min_value, stats["codes"][NUMERIC_BINNED_MIN_TOKEN], inplace=True)
    if NUMERIC_BINNED_MAX_TOKEN in stats["codes"]:
        codes.mask(values == max_value, stats["codes"][NUMERIC_BINNED_MAX_TOKEN], inplace=True)
    df = pd.DataFrame({NUMERIC_BINNED_SUB_COL_SUFFIX: codes})
    return df


def decode_numeric(df_encoded: pd.DataFrame, stats: dict, _: pd.Series | None = None) -> pd.Series:
    if stats["encoding_type"] == ModelEncodingType.tabular_numeric_discrete:
        values = _decode_numeric_discrete(df_encoded, stats)
    elif stats["encoding_type"] == ModelEncodingType.tabular_numeric_digit:
        values = _decode_numeric_digit(df_encoded, stats)
    elif stats["encoding_type"] == ModelEncodingType.tabular_numeric_binned:
        values = _decode_numeric_binned(df_encoded, stats)
    else:
        raise ValueError(f"Unknown encoding type {stats['encoding_type']}")
    return values


def _decode_numeric_discrete(df_encoded: pd.DataFrame, stats: dict) -> pd.Series:
    # determine output dtype
    dtype = "Float64" if stats["min_decimal"] < 0 else "Int64"

    values = decode_categorical(df_encoded, stats)
    # replace any unknown values; either with NAs or with random values
    cnt_unk = (values == NUMERIC_DISCRETE_UNKNOWN_TOKEN).sum()
    if cnt_unk > 0:
        if NUMERIC_DISCRETE_NULL_TOKEN in stats["codes"]:
            values.mask(values == NUMERIC_DISCRETE_UNKNOWN_TOKEN, pd.NA, inplace=True)
        else:
            valid_values = pd.Series(
                [
                    str(k)
                    for k in stats["codes"].keys()
                    if k not in [NUMERIC_DISCRETE_UNKNOWN_TOKEN, NUMERIC_DISCRETE_NULL_TOKEN]
                ]
            )
            if valid_values.empty:  # for the edge-case of "all values are rare"
                valid_values = pd.Series([pd.NA])
            values.mask(
                values == NUMERIC_DISCRETE_UNKNOWN_TOKEN,
                valid_values.sample(cnt_unk, replace=True, ignore_index=True),
                inplace=True,
            )
    values = safe_convert_numeric(values)
    values = values.astype(dtype)
    return values


def _decode_numeric_binned(df_encoded: pd.DataFrame, stats: dict) -> pd.Series:
    # determine output dtype
    dtype = "Float64" if stats["min_decimal"] < 0 else "Int64"

    codes = df_encoded[NUMERIC_BINNED_SUB_COL_SUFFIX]
    bins = stats["bins"]
    # create empty series with N/As of same length as of df_encoded
    values = pd.Series(pd.NA, index=df_encoded.index, dtype="Float64")
    # map special tokens
    if NUMERIC_BINNED_NULL_TOKEN in stats["codes"]:
        values.mask(codes == stats["codes"][NUMERIC_BINNED_NULL_TOKEN], pd.NA, inplace=True)
    if NUMERIC_BINNED_MIN_TOKEN in stats["codes"]:
        values.mask(codes == stats["codes"][NUMERIC_BINNED_MIN_TOKEN], bins[0], inplace=True)
    if NUMERIC_BINNED_MAX_TOKEN in stats["codes"]:
        values.mask(codes == stats["codes"][NUMERIC_BINNED_MAX_TOKEN], bins[-1], inplace=True)
    # map bin tokens to bin intervals
    for i in range(len(bins) - 1):
        # find all rows that have the current bin code
        codes_bin_offset = len(stats["codes"])
        idx = codes[codes == i + codes_bin_offset].index
        # randomly sample from bin interval
        rnd_draws = np.random.uniform(bins[i], bins[i + 1], len(idx))
        # truncate to min_decimal precision
        scaler = 10 ** -stats["min_decimal"]
        rnd_draws = np.floor(rnd_draws * scaler) / scaler
        # replace values with random draws
        values.loc[idx] = rnd_draws

    values = values.astype(dtype)
    return values


@typing.no_type_check
def _decode_numeric_digit(df_encoded: pd.DataFrame, stats: dict) -> pd.Series:
    max_decimal = stats["max_decimal"]
    min_decimal = stats["min_decimal"]
    # sum up all digits positions
    values = [
        (df_encoded[f"E{d}"] + stats["min_digits"][f"E{d}"]).to_numpy("uint64") * 10 ** int(d)
        for d in np.arange(max_decimal, min_decimal - 1, -1)
    ]
    values = sum(values)
    # convert to float if necessary
    dtype = "Float64" if min_decimal < 0 else "Int64"
    values = _type_safe_numeric_series(values, dtype)
    if "nan" in df_encoded.columns:
        values[df_encoded["nan"] == 1] = pd.NA
    if "neg" in df_encoded.columns:
        values[df_encoded["neg"] == 1] = -1 * values[df_encoded["neg"] == 1]
    # replace extreme values with min/max
    if stats["min"] is not None and stats["max"] is not None:
        is_too_low = values.notna() & (values < stats["min"])
        is_too_high = values.notna() & (values > stats["max"])
        values.loc[is_too_low] = _type_safe_numeric_series(np.ones(sum(is_too_low)) * stats["min"], dtype).values
        values.loc[is_too_high] = _type_safe_numeric_series(np.ones(sum(is_too_high)) * stats["max"], dtype).values
    elif "nan" in df_encoded.columns:
        # set all values to NaN if no valid values were present
        values[df_encoded["nan"] == 0] = pd.NA
    # round to min_decimal precision
    values = np.round(values, -min_decimal)
    return values
