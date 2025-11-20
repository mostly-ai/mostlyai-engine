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

import inspect
import itertools
import json
import logging
import platform
import time
from collections.abc import Callable, Iterable
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Literal,
    NamedTuple,
    Protocol,
)

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mostlyai.engine._dtypes import is_boolean_dtype, is_float_dtype, is_integer_dtype
from mostlyai.engine.domain import ModelEncodingType

_LOG = logging.getLogger(__name__)

_LOG.info(f"running on Python ({platform.python_version()})")

TGT = "tgt"
CTXFLT = "ctxflt"
CTXSEQ = "ctxseq"
ARGN_PROCESSOR = "argn_processor"
ARGN_TABLE = "argn_table"
ARGN_COLUMN = "argn_column"
PREFIX_TABLE = ":"
PREFIX_COLUMN = "/"
PREFIX_SUB_COLUMN = "__"
SIDX_RIDX_DIGIT_ENCODING_THRESHOLD = 100
POSITIONAL_COLUMN = f"{TGT}{PREFIX_TABLE}{PREFIX_COLUMN}"
SIDX_SUB_COLUMN_PREFIX = f"{POSITIONAL_COLUMN}{PREFIX_SUB_COLUMN}sidx_"  # sequence index
RIDX_SUB_COLUMN_PREFIX = f"{POSITIONAL_COLUMN}{PREFIX_SUB_COLUMN}ridx_"  # reverse index
SLEN_SUB_COLUMN_PREFIX = f"{POSITIONAL_COLUMN}{PREFIX_SUB_COLUMN}slen_"  # sequence length
SDEC_SUB_COLUMN_PREFIX = f"{POSITIONAL_COLUMN}{PREFIX_SUB_COLUMN}sdec_"  # sequence index decile
TABLE_COLUMN_INFIX = "::"  # this should be consistent as in mostly-data and mostlyai-qa

# the latest version of the model uses SIDX/SLEN/RIDX positional column
DEFAULT_HAS_SLEN = True
DEFAULT_HAS_RIDX = True
DEFAULT_HAS_SDEC = False

ANALYZE_MIN_MAX_TOP_N = 1000  # the number of min/max values to be kept from each partition

# the minimal number of min/max values to trigger the reduction; if less, the min/max will be reduced to None
# this should be at least greater than the non-DP stochastic threshold for rare value protection (5 + noise)
ANALYZE_REDUCE_MIN_MAX_N = 20

TEMPORARY_PRIMARY_KEY = "__primary_key"

STRING = "string[pyarrow]"  # This utilizes pyarrow's large string type since pandas 2.2

# considering pandas timestamp boundaries ('1677-09-21 00:12:43.145224193' < x < '2262-04-11 23:47:16.854775807')
_MIN_DATE = np.datetime64("1700-01-01")
_MAX_DATE = np.datetime64("2250-01-01")

SubColumnsNested = dict[str, list[str]]


class ProgressCallback(Protocol):
    def __call__(
        self,
        total: int | None = None,
        completed: int | None = None,
        advance: int | None = None,
        message: dict | None = None,
        **kwargs,
    ) -> dict | None: ...


class ProgressCallbackWrapper:
    def _add_to_progress_history(self, message: dict) -> None:
        # convert message to DataFrame; drop all-NA columns to avoid pandas 2.x warning for concat
        message_df = pd.DataFrame([message]).dropna(axis=1, how="all")
        # append to history of progress messages
        if self._progress_messages is None:
            self._progress_messages = message_df
        else:
            self._progress_messages = pd.concat([self._progress_messages, message_df], ignore_index=True)
        if self._progress_messages_path is not None:
            self._progress_messages.to_csv(self._progress_messages_path, index=False)

    def update(
        self,
        total: int | None = None,
        completed: int | None = None,
        advance: int | None = None,
        message: dict | BaseModel | None = None,
        **kwargs,
    ) -> dict | None:
        if isinstance(message, BaseModel):
            message = message.model_dump(mode="json")
        if message is not None:
            _LOG.info(message)
            self._add_to_progress_history(message)
        return self._update_progress(total=total, completed=completed, advance=advance, message=message, **kwargs)

    def get_last_progress_message(self) -> dict | None:
        if self._progress_messages is not None:
            return self._progress_messages.iloc[-1].to_dict()

    def reset_progress_messages(self):
        if self._progress_messages is not None:
            self._progress_messages = None
        if self._progress_messages_path and self._progress_messages_path.exists():
            self._progress_messages_path.unlink()

    def __init__(
        self, update_progress: ProgressCallback | None = None, progress_messages_path: Path | None = None, **kwargs
    ):
        self._update_progress = update_progress if update_progress is not None else (lambda *args, **kwargs: None)
        self._progress_messages_path = progress_messages_path
        if self._progress_messages_path and self._progress_messages_path.exists():
            self._progress_messages = pd.read_csv(self._progress_messages_path)
        else:
            self._progress_messages = None

    def __enter__(self):
        self._update_progress(completed=0, total=1)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self._update_progress(completed=1, total=1)


class SubColumnLookup(NamedTuple):
    col_name: str
    col_idx: int  # column index within a list of columns
    sub_col_idx: int  # index within the column it belongs to
    sub_col_cum: int  # cumulative index within a list of columns
    sub_col_offset: int  # offset of the first sub-column in the scope of the column


def get_class_labels_from_stats(column_stats: dict) -> list[str]:
    """
    Get class labels for a column from its encoding statistics.

    Args:
        column_stats: Dictionary containing encoding statistics for a column,
                     including 'encoding_type', 'codes', and optionally 'bins'.

    Returns:
        List of class labels in order. Label format depends on encoding type:
        - Binned: "<{value}" format (e.g., "<10000", "<20000", "<50000")
        - Categorical: category names (e.g., "male", "female"), "_RARE_" becomes "rare"
        - Discrete: discrete values (e.g., "0", "1", "2"), "_RARE_" becomes "rare"
        Special tokens like "<<NULL>>" are trimmed and lowercased (e.g., "null").

    Raises:
        ValueError: If encoding type is not supported for probability output.
    """
    encoding_type = ModelEncodingType(column_stats["encoding_type"])

    if encoding_type == ModelEncodingType.tabular_numeric_binned:
        # Get bin boundaries and construct labels
        bins = column_stats["bins"]  # e.g., [0, 10000, 20000, 50000, inf]
        codes = column_stats["codes"]  # e.g., {"<<UNK>>": 0, "<<NULL>>": 1, "<<MIN>>": 2, "<<MAX>>": 3}
        labels = []

        # Handle special codes first (trim << >> and lowercase)
        for code_name in codes.keys():
            if code_name.startswith("<<") and code_name.endswith(">>"):
                labels.append(code_name[2:-2].lower())

        # Add regular bin labels using upper bounds (bins are [a, b) intervals)
        for i in range(len(bins) - 1):
            labels.append(f"<{bins[i + 1]}")

        return labels

    elif encoding_type == ModelEncodingType.tabular_categorical:
        # Get category names from codes
        codes = column_stats["codes"]  # e.g., {"_RARE_": 0, "<<NULL>>": 1, "male": 2, "female": 3}
        labels = []
        for category in codes.keys():
            if category == "_RARE_":
                labels.append("rare")
            elif category.startswith("<<") and category.endswith(">>"):
                # Trim << >> and lowercase for special tokens
                labels.append(category[2:-2].lower())
            else:
                labels.append(category)
        return labels

    elif encoding_type == ModelEncodingType.tabular_numeric_discrete:
        # Get discrete values from codes
        codes = column_stats["codes"]  # e.g., {"_RARE_": 0, "<<NULL>>": 1, "0": 2, "1": 3, "2": 4}
        labels = []
        for value in codes.keys():
            if value == "_RARE_":
                labels.append("rare")
            elif value.startswith("<<") and value.endswith(">>"):
                # Trim << >> and lowercase for special tokens
                labels.append(value[2:-2].lower())
            else:
                labels.append(value)
        return labels

    else:
        raise ValueError(f"Unsupported encoding type for probabilities: {encoding_type}")


def cast_numpy_keys_to_python(data: Any) -> dict:
    if not isinstance(data, dict):
        return data

    new_data = {}
    for key, value in data.items():
        if isinstance(key, (np.int64, np.int32)):
            new_key = int(key)
        else:
            new_key = key
        new_data[new_key] = cast_numpy_keys_to_python(value)

    return new_data


def write_json(data: dict, fn: Path) -> None:
    data = cast_numpy_keys_to_python(data)
    fn.parent.mkdir(parents=True, exist_ok=True)
    with open(fn, "w", encoding="utf-8") as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


def read_json(path: Path, default: dict | None = None, raises: bool | None = None) -> dict:
    """
    Reads JSON.

    :param path: path to json
    :param default: default used in case path does not exist
    :param raises: if True, raises exception if path does not exist,
        otherwise returns default
    :return: dict representation of JSON
    """

    if default is None:
        default = {}
    if not path.exists():
        if raises:
            raise RuntimeError(f"File [{path}] does not exist")
        else:
            return default
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def is_a_list(x) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, str)


def is_sequential(series: pd.Series) -> bool:
    return not series.empty and series.apply(is_a_list).any()


def handle_with_nested_lists(func: Callable, param_reference: str = "values"):
    @wraps(func)
    def wrapper(*args, **kwargs):
        signature = inspect.signature(func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        series = bound_args.arguments.get(param_reference)

        if series is not None and is_sequential(series):

            def func_on_exploded_series(series):
                is_empty = series.apply(lambda x: isinstance(x, Iterable) and len(x) == 0)
                bound_args.arguments[param_reference] = series.explode()

                result = func(*bound_args.args, **bound_args.kwargs)

                result = result.groupby(level=0).apply(np.array)
                result[is_empty] = result[is_empty].apply(lambda x: np.array([], dtype=x.dtype))
                return result

            index, series = series.index, series.reset_index(drop=True)
            result = func_on_exploded_series(series).set_axis(index)
            return result
        else:
            return func(*args, **kwargs)

    return wrapper


@handle_with_nested_lists
def safe_convert_numeric(values: pd.Series, nullable_dtypes: bool = False) -> pd.Series:
    if is_boolean_dtype(values):
        # convert booleans to integer -> True=1, False=0
        values = values.astype("Int8")
    elif not is_integer_dtype(values) and not is_float_dtype(values):
        # convert other non-numerics to string, and extract valid numeric sub-string
        valid_num = r"(-?[0-9]*[\.]?[0-9]+(?:[eE][+\-]?\d+)?)"
        values = values.astype(str).str.extract(valid_num, expand=False)
    values = pd.to_numeric(values, errors="coerce")
    if nullable_dtypes:
        values = values.convert_dtypes()
    return values


@handle_with_nested_lists
def safe_convert_datetime(values: pd.Series, date_only: bool = False) -> pd.Series:
    # turn null[pyarrow] into string, can be removed once the following line is fixed in pandas:
    # pd.Series([pd.NA], dtype="null[pyarrow]").mask([True], pd.NA)
    # see https://github.com/pandas-dev/pandas/issues/58696 for tracking the fix of this bug
    if values.dtype == "null[pyarrow]":
        values = values.astype("string")
    # Convert any pd.Series to datetime via `pd.to_datetime`.
    values_parsed_flexible = pd.to_datetime(
        values,
        errors="coerce",  # silently map invalid dates to NA
        utc=True,
        format="mixed",
        dayfirst=False,  # assume 1/3/2020 is Jan 3
    )
    values = values.mask(values_parsed_flexible.isna(), pd.NA)
    values_parsed_fixed = pd.to_datetime(
        values,
        errors="coerce",  # silently map invalid dates to NA
        utc=True,
        dayfirst=False,  # assume 1/3/2020 is Jan 3
    )
    # check whether firstday=True yields less non-NA, and if so, switch to using that flag
    if values_parsed_fixed.isna().sum() > values.isna().sum():
        values_parsed_fixed_dayfirst = pd.to_datetime(
            values,
            errors="coerce",  # silently map invalid dates to NA
            utc=True,
            format="mixed",
            dayfirst=True,  # assume 1/3/2020 is Mar 1
        )
        if values_parsed_fixed_dayfirst.isna().sum() < values_parsed_fixed.isna().sum():
            values_parsed_fixed = values_parsed_fixed_dayfirst
    # combine results of consistent and flexible datetime parsing, with the former having precedence
    values = values_parsed_fixed.fillna(values_parsed_flexible)
    if date_only:
        values = pd.to_datetime(values.dt.date)
    values = values.dt.tz_localize(None)
    # We need to downcast from `datetime64[ns]` to `datetime64[us]`
    # otherwise `pd.to_parquet` crashes for overly long precisions.
    # See https://stackoverflow.com/a/56795049
    values = values.astype("datetime64[us]")
    return values


@handle_with_nested_lists
def safe_convert_string(values: pd.Series) -> pd.Series:
    values = values.astype("string")
    return values


def get_argn_name(
    argn_processor: str,
    argn_table: str | None = None,
    argn_column: str | None = None,
    argn_sub_column: str | None = None,
) -> str:
    name = [
        argn_processor,
        PREFIX_TABLE if any(c is not None for c in [argn_table, argn_column, argn_sub_column]) else "",
        argn_table if argn_table is not None else "",
        PREFIX_COLUMN if any(c is not None for c in [argn_column, argn_sub_column]) else "",
        argn_column if argn_column is not None else "",
        PREFIX_SUB_COLUMN if argn_sub_column is not None else "",
        argn_sub_column if argn_sub_column is not None else "",
    ]
    return "".join(name)


def get_cardinalities(
    stats: dict, has_slen: bool | None = None, has_ridx: bool | None = None, has_sdec: bool | None = None
) -> dict[str, int]:
    # the latest version of the model uses SIDX/SLEN/RIDX positional column (applies to sequential model only)
    cardinalities: dict[str, int] = {}

    if stats.get("is_sequential", False):
        max_seq_len = get_sequence_length_stats(stats)["max"]
        cardinalities |= get_positional_cardinalities(max_seq_len, has_slen, has_ridx, has_sdec)

    for i, column in enumerate(stats.get("columns", [])):
        column_stats = stats["columns"][column]
        if "cardinalities" not in column_stats:
            continue
        sub_columns = {
            get_argn_name(
                argn_processor=column_stats[ARGN_PROCESSOR],
                argn_table=column_stats[ARGN_TABLE],
                argn_column=column_stats[ARGN_COLUMN],
                argn_sub_column=k,
            ): v
            for k, v in column_stats["cardinalities"].items()
        }
        cardinalities = cardinalities | sub_columns
    return cardinalities


def get_sub_columns_from_cardinalities(cardinalities: dict[str, int]) -> list[str]:
    # eg. {'c0__E1': 10, 'c0__E0': 10, 'c1__value': 2} -> ['c0__E1', 'c0__E0', 'c1__value']
    sub_columns = list(cardinalities.keys())
    return sub_columns


def get_columns_from_cardinalities(cardinalities: dict[str, int]) -> list[str]:
    # eg. {'c0__E1': 10, 'c0__E0': 10, 'c1__value': 2} -> ['c0', 'c1']
    sub_columns = get_sub_columns_from_cardinalities(cardinalities)
    columns = [col for col, _ in itertools.groupby(sub_columns, lambda x: x.split(PREFIX_SUB_COLUMN)[0])]
    return columns


def get_sub_columns_nested(
    sub_columns: list[str], groupby: Literal["processor", "tables", "columns"]
) -> dict[str, list[str]]:
    splitby = {
        "processor": PREFIX_TABLE,
        "tables": PREFIX_COLUMN,
        "columns": PREFIX_SUB_COLUMN,
    }[groupby]
    out: dict[str, list[str]] = dict()
    for sub_col in sub_columns:
        key = sub_col.split(splitby)[0]
        out[key] = out.get(key, []) + [sub_col]
    return out


def get_sub_columns_nested_from_cardinalities(
    cardinalities: dict[str, int], groupby: Literal["processor", "tables", "columns"]
) -> SubColumnsNested:
    # eg. {'c0__E1': 10, 'c0__E0': 10, 'c1__value': 2} -> {'c0': ['c0__E1', 'c0__E0'], 'c1': ['c1__value']}
    sub_columns = get_sub_columns_from_cardinalities(cardinalities)
    return get_sub_columns_nested(sub_columns, groupby)


def get_sub_columns_lookup(
    sub_columns_nested: SubColumnsNested,
) -> dict[str, SubColumnLookup]:
    """
    Create a convenient reverse lookup for each of the sub-columns
    :param sub_columns_nested: must be grouped-by "columns"
    :return: dict of sub_col -> SubColumnLookup items
    """
    sub_cols_lookup = {}
    sub_col_cum_i = 0
    for col_i, (name, sub_cols) in enumerate(sub_columns_nested.items()):
        sub_col_offset = sub_col_cum_i
        for sub_col_i, sub_col in enumerate(sub_cols):
            sub_cols_lookup[sub_col] = SubColumnLookup(
                col_name=name,
                col_idx=col_i,
                sub_col_idx=sub_col_i,
                sub_col_cum=sub_col_cum_i,
                sub_col_offset=sub_col_offset,
            )
            sub_col_cum_i += 1
    return sub_cols_lookup


def get_ctx_sequence_length(ctx_stats: dict, key: str) -> dict[str, int]:
    """
    Get the stats of sequence lengths from the first column_stats of each context table
    """
    ctxseq_stats: dict[str, int] = {}

    for column_stats in ctx_stats.get("columns", {}).values():
        if "seq_len" in column_stats:
            table = get_argn_name(
                argn_processor=column_stats[ARGN_PROCESSOR],
                argn_table=column_stats[ARGN_TABLE],
            )
            if table not in ctxseq_stats:
                ctxseq_stats[table] = column_stats["seq_len"][key]

    return ctxseq_stats


def get_max_data_points_per_sample(stats: dict) -> int:
    """Return the maximum number of data points per sample. Either for target or for context"""
    data_points = 0
    seq_len_max = stats["seq_len"]["max"] if "seq_len" in stats else 1
    for info in stats.get("columns", {}).values():
        col_seq_len_max = info["seq_len"]["max"] if "seq_len" in info else 1
        no_sub_cols = len(info["cardinalities"]) if "cardinalities" in info else 1
        data_points += col_seq_len_max * no_sub_cols * seq_len_max
    return data_points


def get_sequence_length_stats(stats: dict) -> dict:
    if stats["is_sequential"]:
        stats = {
            "min": stats["seq_len"]["min"],
            "median": stats["seq_len"]["median"],
            "max": stats["seq_len"]["max"],
        }
    else:
        stats = {
            "min": 1,
            "median": 1,
            "max": 1,
        }
    return stats


def find_distinct_bins(x: list[Any], n: int, n_max: int = 1_000) -> list[Any]:
    """
    Find distinct bins so that `pd.cut(x, bins, include_lowest=True)` returns `n` distinct buckets with similar
    number of values. For that we compute quantiles, and increase the number of quantiles until we get `n` distinct
    values. If we have less distinct values than `n`, we return the distinct values.
    """
    # return immediately if we have less distinct values than `n`
    if len(x) <= n or len(set(x)) <= n:
        return list(sorted(set(x)))
    no_of_quantiles = n
    # increase quantiles until we have found `n` distinct bins
    while no_of_quantiles <= n_max:
        # calculate quantiles
        qs = np.quantile(x, np.linspace(0, 1, no_of_quantiles + 1), method="closest_observation")
        no_of_distinct_quantiles = len(set(qs))
        # return if we have found `n` distinct quantiles
        if no_of_distinct_quantiles >= n + 1:
            bins = list(sorted(set(qs)))
            if len(bins) > n + 1:
                # handle edge case where we have more than `n` + 1 bins; this can happen if we have a eg 100 bins for
                # no_of_quantiles=200, but then 102 bins for no_of_quantiles=201.
                bins = bins[: (n // 2) + 1] + bins[-(n // 2) :]
            return bins
        # we need to increase at least by number of missing quantiles to acchieve `n` distinct quantiles
        no_of_quantiles += 1 + max(0, n - no_of_distinct_quantiles)
    # in case we fail to find `n` distinct bins before `n_max` we return largest set of bins
    return list(sorted(set(qs)))


def apply_encoding_type_dtypes(df: pd.DataFrame, encoding_types: dict[str, ModelEncodingType]) -> pd.DataFrame:
    return df.apply(lambda x: _get_type_converter(encoding_types[x.name])(x) if x.name in encoding_types else x)


def _get_type_converter(
    encoding_type: ModelEncodingType | None,
) -> Callable[[pd.Series], pd.Series]:
    if encoding_type in (ModelEncodingType.tabular_categorical, ModelEncodingType.tabular_lat_long):
        return safe_convert_string
    elif encoding_type in (
        ModelEncodingType.tabular_numeric_auto,
        ModelEncodingType.tabular_numeric_digit,
        ModelEncodingType.tabular_numeric_binned,
        ModelEncodingType.tabular_numeric_discrete,
    ):
        return lambda values: safe_convert_numeric(values, nullable_dtypes=True)
    elif encoding_type in (ModelEncodingType.tabular_datetime, ModelEncodingType.tabular_datetime_relative):
        return safe_convert_datetime
    else:
        return safe_convert_string


def skip_if_error(func: Callable) -> Callable:
    """
    Decorator that executes the wrapped function, and gracefully absorbs any exceptions
    in a case of a failure and logs the exception, accordingly.
    """

    @wraps(func)
    def skip_if_error_wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _LOG.warning(f"{func.__qualname__} failed with {type(e)}: {e}")

    return skip_if_error_wrapper


def encode_positional_column(vals: pd.Series, max_seq_len: int, prefix: str = "") -> pd.DataFrame:
    assert is_integer_dtype(vals)
    if max_seq_len < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
        # encode positional column as numeric_discrete
        df = pd.DataFrame({f"{prefix}cat": vals})
    else:
        # encode positional column as numeric_digit
        n_digits = len(str(max_seq_len))
        df = pd.DataFrame(vals.astype(str).str.pad(width=n_digits, fillchar="0").apply(list).tolist()).astype(int)
        df.columns = [f"{prefix}E{i}" for i in range(n_digits - 1, -1, -1)]
    return df


def decode_positional_column(df_encoded: pd.DataFrame, max_seq_len: int, prefix: str = "") -> pd.Series:
    if max_seq_len < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
        # decode positional column as numeric_discrete
        vals = df_encoded[f"{prefix}cat"]
    else:
        # decode positional column as numeric_digit
        n_digits = len(str(max_seq_len))
        vals = sum([df_encoded[f"{prefix}E{d}"] * 10 ** int(d) for d in list(range(n_digits))])
    return vals


def get_positional_cardinalities(
    max_seq_len: int, has_slen: bool | None = None, has_ridx: bool | None = None, has_sdec: bool | None = None
) -> dict[str, int]:
    has_slen = has_slen if has_slen is not None else DEFAULT_HAS_SLEN
    has_ridx = has_ridx if has_ridx is not None else DEFAULT_HAS_RIDX
    has_sdec = has_sdec if has_sdec is not None else DEFAULT_HAS_SDEC

    if max_seq_len < SIDX_RIDX_DIGIT_ENCODING_THRESHOLD:
        # encode positional columns as numeric_discrete
        sidx_cardinalities = {f"{SIDX_SUB_COLUMN_PREFIX}cat": max_seq_len + 1}
        slen_cardinalities = {f"{SLEN_SUB_COLUMN_PREFIX}cat": max_seq_len + 1}
        ridx_cardinalities = {f"{RIDX_SUB_COLUMN_PREFIX}cat": max_seq_len + 1}
    else:
        # encode positional columns as numeric_digit
        digits = [int(digit) for digit in str(max_seq_len)]
        sidx_cardinalities = {}
        slen_cardinalities = {}
        ridx_cardinalities = {}
        for idx, digit in enumerate(digits):
            # cap cardinality of the most significant position
            # less significant positions allow any digit
            card = digit + 1 if idx == 0 else 10
            e_idx = len(digits) - idx - 1
            sidx_cardinalities[f"{SIDX_SUB_COLUMN_PREFIX}E{e_idx}"] = card
            ridx_cardinalities[f"{RIDX_SUB_COLUMN_PREFIX}E{e_idx}"] = card
            slen_cardinalities[f"{SLEN_SUB_COLUMN_PREFIX}E{e_idx}"] = card
    sdec_cardinalities = {f"{SDEC_SUB_COLUMN_PREFIX}cat": 10}
    match has_slen, has_ridx, has_sdec:
        case True, True, False:
            # SIDX/SLEN/RIDX model
            return sidx_cardinalities | slen_cardinalities | ridx_cardinalities
        case True, False, True:
            # SLEN/SIDX/SDEC model
            return slen_cardinalities | sidx_cardinalities | sdec_cardinalities
        case True, False, False:
            # SLEN/SIDX model
            return slen_cardinalities | sidx_cardinalities
        case _:
            raise ValueError(f"Invalid positional encoding: {has_slen=}, {has_ridx=}, {has_sdec=}")


def persist_data_part(df: pd.DataFrame, output_path: Path, infix: str):
    t0 = time.time()
    part_fn = f"part.{infix}.parquet"
    # ensure df.shape[0] is persisted when no columns are generated by keeping index
    df.to_parquet(output_path / part_fn, index=True)
    _LOG.info(f"persisted {df.shape} to `{part_fn}` in {time.time() - t0:.2f}s")


class FixedSizeSampleBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.current_size = 0
        self.n_clears = 0

    def add(self, tup: tuple):
        assert not self.is_full()
        assert len(tup) > 0 and isinstance(tup[0], Iterable)
        n_samples = len(tup[0])  # assume first element holds samples
        self.current_size += n_samples
        self.buffer.append(tup)

    def is_full(self):
        return self.current_size >= self.capacity

    def is_empty(self):
        return len(self.buffer) == 0

    def clear(self):
        self.buffer = []
        self.current_size = 0
        self.n_clears += 1


def _get_log_histogram_bin_bounds(idx: int, bins: int = 64) -> tuple[float, float]:
    """
    Compute the lower and upper boundaries for a logarithmically-spaced histogram bin.

    Creates symmetric logarithmic bins around zero that efficiently represent values
    across many orders of magnitude. With bins=64, creates 128 total bins covering
    negative powers of 2, the range [-1, 1], and positive powers of 2.

    Modified from OpenDP's SmartNoise SDK (MIT License)
    Source: https://github.com/opendp/smartnoise-sdk/blob/main/sql/snsql/sql/_mechanisms/approx_bounds.py

    Args:
        idx: The bin index (0 to bins*2-1)
        bins: Number of bins per side (default 64, creating 128 total bins)

    Returns:
        Tuple of (lower_edge, upper_edge) for the bin
    """
    if idx == bins:
        return (0.0, 1.0)
    elif idx > bins:
        return (2.0 ** (idx - bins - 1), 2.0 ** (idx - bins))
    elif idx == bins - 1:
        return (-1.0, -0.0)
    else:
        return (-1 * 2.0 ** np.abs(bins - idx - 1), -1 * 2.0 ** np.abs(bins - idx - 2))


def compute_log_histogram(values: np.ndarray, bins: int = 64) -> list[int]:
    """
    Compute a histogram using logarithmically-spaced bins for efficient distribution analysis.

    This creates a histogram that can efficiently represent values spanning many orders of
    magnitude (e.g., 0.001 to 1,000,000) using relatively few bins. The bins are symmetric
    around zero with exponentially increasing widths away from zero.

    Modified from OpenDP's SmartNoise SDK (MIT License)
    Source: https://github.com/opendp/smartnoise-sdk/blob/main/sql/snsql/sql/_mechanisms/approx_bounds.py

    Args:
        values: Array of numeric values to histogram
        bins: Number of bins per side (default 64, creating 128 total bins)

    Returns:
        List of counts for each bin. Invalid values (NaN, inf) are filtered out.
    """
    # filter out invalid values
    values = np.array(values, dtype=np.float64)
    values = values[~np.isinf(values) & ~np.isnan(values)]

    # generate all bin edges efficiently
    edge_list = [_get_log_histogram_bin_bounds(idx, bins) for idx in range(bins * 2)]
    bin_edges = np.array([lower for lower, _ in edge_list] + [edge_list[-1][1]])

    # clip values to be within the bin edges to ensure all values are counted
    min_val = bin_edges[0]
    max_val = bin_edges[-1]
    values = np.clip(values, min_val, max_val)

    # use numpy's histogram for efficient binning (O(n log bins) vs O(n * bins))
    hist, _ = np.histogram(values, bins=bin_edges)

    return hist.tolist()


def dp_approx_bounds(hist: list[int], epsilon: float) -> tuple[float | None, float | None]:
    """
    Estimate the minimum and maximum values using a differentially private histogram.

    Uses Laplace noise on histogram bin counts, then finds the lowest and highest bins
    that exceed a threshold (based on failure probability). Returns None if insufficient
    data or privacy budget makes reliable estimation impossible.

    Reference: https://desfontain.es/thesis/Usability.html#usability-u-ding-
    Modified from OpenDP's SmartNoise SDK (MIT License)
    Source: https://github.com/opendp/smartnoise-sdk/blob/main/sql/snsql/sql/_mechanisms/approx_bounds.py

    Args:
        hist: A list of log histogram counts (typically from compute_log_histogram).
        epsilon: The privacy budget to spend estimating the bounds.

    Returns:
        Tuple of (min, max) estimates, or (None, None) if bounds cannot be reliably estimated.
    """

    n_bins = len(hist)

    noise = np.random.laplace(loc=0.0, scale=1 / epsilon, size=n_bins)
    hist = [val + lap_noise for val, lap_noise in zip(hist, noise)]

    failure_prob = 10e-9
    highest_failure_prob = 1 / (n_bins * 2)

    exceeds = []
    while len(exceeds) < 1 and failure_prob <= highest_failure_prob:
        p = 1 - failure_prob
        K = -np.log(2 - 2 * p ** (1 / (n_bins - 1))) / epsilon
        exceeds = [idx for idx, v in enumerate(hist) if v > K]
        failure_prob *= 10

    if len(exceeds) == 0:
        return (None, None)

    lower_bin, upper_bin = min(exceeds), max(exceeds)
    lower, _ = _get_log_histogram_bin_bounds(lower_bin)
    _, upper = _get_log_histogram_bin_bounds(upper_bin)
    return (float(lower), float(upper))


def _dp_bounded_quantiles(
    values: np.ndarray, quantiles: list[float], epsilon: float, lower: float, upper: float
) -> list[float]:
    """
    Estimate quantiles with differential privacy using the Smith (2011) smooth sensitivity method.

    Assumes values are bounded within [lower, upper]. Uses exponential mechanism to sample
    quantile estimates with noise proportional to local sensitivity. Privacy budget is split
    evenly across all requested quantiles. Results are post-processed to ensure monotonicity.

    Reference: http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf

    Args:
        values: A 1D array of numeric values.
        quantiles: List of quantile probabilities to estimate (e.g., [0.05, 0.5, 0.95]).
        epsilon: Privacy budget (split evenly across quantiles).
        lower: Lower bound for clipping values.
        upper: Upper bound for clipping values.

    Returns:
        List of differentially private quantile estimates (monotonically ordered).
    """

    _LOG.info(f"compute DP bounded quantiles within [{lower}, {upper}]")
    results = []
    eps_part = epsilon / len(quantiles)
    k = len(values)
    values = np.clip(values, lower, upper)
    values = np.sort(values)
    for q in quantiles:
        Z = np.concatenate(([lower], values, [upper]))
        Z -= lower  # shift right to be 0 bounded
        y = np.exp(-eps_part * np.abs(np.arange(len(Z) - 1) - q * k)) * (Z[1:] - Z[:-1])
        y_sum = y.sum()
        p = y / y_sum if y_sum > 0 else np.ones(len(y)) / len(y)  # use uniform distribution if y_sum is zero
        idx = np.random.choice(range(k + 1), 1, False, p)[0]
        v = np.random.uniform(Z[idx], Z[idx + 1])
        results.append(v + lower)

    # ensure monotonicity of results with respect to quantiles
    sorted_indices = [t[0] for t in sorted(enumerate(quantiles), key=lambda x: x[1])]
    sorted_results = sorted(results)
    results = [sorted_results[sorted_indices.index(i)] for i in range(len(quantiles))]

    return results


def dp_quantiles(values: list | np.ndarray, quantiles: list[float], epsilon: float) -> list[float]:
    """
    Estimate quantiles with differential privacy using a two-phase approach.

    Phase 1: Estimate data bounds using dp_approx_bounds on a log histogram.
    Phase 2: Estimate quantiles within those bounds using _dp_bounded_quantiles.

    Privacy budget is split as epsilon/(m+1) for bounds and m*epsilon/(m+1) for m quantiles.
    Returns None values if bounds cannot be reliably estimated (insufficient data/privacy budget).

    Args:
        values: A list or array of numeric values.
        quantiles: List of quantile probabilities to estimate (e.g., [0.05, 0.95]).
        epsilon: Total privacy budget to allocate.

    Returns:
        List of differentially private quantile estimates, or list of None if estimation fails.
    """
    values = np.array(values)

    # split epsilon in (m + 1) parts for m quantiles and 1 for the bounds
    m = len(quantiles)
    eps_bounds = epsilon / (m + 1)
    eps_quantiles = epsilon - eps_bounds

    # get the bounds
    # for too small values of epsilon and/or sample size this can return None
    hist = compute_log_histogram(values)
    lower, upper = dp_approx_bounds(hist, eps_bounds)

    if lower is None or upper is None:
        return [None] * len(quantiles)
    return _dp_bounded_quantiles(values=values, quantiles=quantiles, epsilon=eps_quantiles, lower=lower, upper=upper)


def dp_non_rare(value_counts: dict[str, int], epsilon: float, threshold: int = 5) -> tuple[list[str], float]:
    """
    Select non-rare categories (count >= threshold) with differential privacy.

    Uses the Laplace vector mechanism: adds independent Laplace(1/ε) noise to each count,
    then selects categories where noisy_count >= threshold. Also computes the non-rare ratio
    (fraction of total counts in selected categories).

    Provides ε-differential privacy via the Laplace mechanism with L1 sensitivity = 1.

    Args:
        value_counts: Mapping from category name to its count.
        epsilon: Privacy budget.
        threshold: Minimum count threshold for non-rare categories (default: 5).

    Returns:
        Tuple of (selected_categories, non_rare_ratio), both with ε-DP guarantees.
    """

    # 1. Add independent Laplace(1/ε) noise to each count (vector Laplace mechanism)
    # Note: sensitivity of the count vector is 1 in L1 norm
    noise = np.random.laplace(loc=0.0, scale=1 / epsilon, size=len(value_counts))
    noisy_counts = np.clip(np.array(list(value_counts.values())) + noise, 0, None).astype(int)
    for i, cat in enumerate(value_counts):
        value_counts[cat] = noisy_counts[i]
    # NOTE: total_counts can be 0 in the edge case when the column only has null values
    total_counts = sum(value_counts.values())

    # 2. Collect all categories whose noisy count >= threshold
    selected = {cat: nc for cat, nc in value_counts.items() if nc >= threshold}

    # 3. Compute the non-rare ratio
    noisy_total_counts = sum(selected.values())
    non_rare_ratio = noisy_total_counts / total_counts if total_counts > 0 else 0

    return list(selected.keys()), non_rare_ratio


def get_stochastic_rare_threshold(min_threshold: int = 5, noise_multiplier: float = 3) -> int:
    """
    Generate a randomized threshold for rare category detection.

    Adds uniform random noise to the base threshold to prevent adversaries from
    exploiting knowledge of exact threshold values. The threshold is sampled from
    [min_threshold, min_threshold + noise_multiplier).

    Args:
        min_threshold: Base threshold value (default: 5).
        noise_multiplier: Maximum noise to add (default: 3).

    Returns:
        Integer threshold in range [min_threshold, min_threshold + noise_multiplier).
    """
    return min_threshold + int(noise_multiplier * np.random.uniform())


def get_empirical_probs_for_predictor_init(
    first_encoded_part: Path, tgt_cardinalities: dict[str, int], is_sequential: bool, alpha: float = 1.0
) -> dict[str, np.ndarray]:
    """
    Calculate empirical probabilities of each sub column from the first partition of encoded data.
    The probabilities will be used for predictor layer initialization.

    Args:
        first_encoded_part: Path to the first partition of encoded data.
        tgt_cardinalities: Mapping from column name to its cardinality.
        is_sequential: Whether the model is sequential.
        alpha: Laplace smoothing parameter. If smaller or equal to 0, no smoothing is applied.

    Returns:
        dict[str, np.ndarray]: Mapping from sub column name to its empirical probabilities.
    """
    df_part = pd.read_parquet(first_encoded_part)
    # for sequential models, we will use the empirical probs of the first time step for weight initialization
    if is_sequential:
        for sub_col in df_part.columns:
            df_part[sub_col] = df_part[sub_col].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
    # check which columns have a separate NaN sub column
    has_nan_map = {
        col: f"{col}{PREFIX_SUB_COLUMN}nan" in tgt_cardinalities
        for col in get_columns_from_cardinalities(tgt_cardinalities)
    }
    probs_map: dict[str, np.ndarray] = {}
    for sub_col, cardinality in tgt_cardinalities.items():
        col, _ = sub_col.split(PREFIX_SUB_COLUMN)
        nan_sub_col = f"{col}{PREFIX_SUB_COLUMN}nan"
        if has_nan_map[col] is True and sub_col != nan_sub_col and (df_part[nan_sub_col] == 0).sum() > 0:
            # exclude NaN rows from the calculation if
            # - this column has a separate NaN sub column
            # - the NaN sub column has at least one non-NaN row
            # - this sub column is not the NaN sub column
            df_part_sub_col = df_part.loc[df_part[nan_sub_col] == 0, sub_col]
        else:
            df_part_sub_col = df_part[sub_col]
        # calculate empirical probabilities
        vc = df_part_sub_col.value_counts()
        if vc.empty:
            # fallback to uniform distribution
            probs_map[sub_col] = np.full(cardinality, 1.0 / cardinality)
        else:
            counts = np.zeros(cardinality)
            for idx, count in vc.items():
                counts[int(idx)] = float(count)
            # apply Laplace smoothing
            alpha = max(0.0, alpha)
            total = counts.sum() + alpha * len(counts)
            probs_map[sub_col] = (counts + alpha) / max(total, 1e-12)
            probs_map[sub_col] = np.clip(probs_map[sub_col], a_min=1e-12, a_max=None)
    return probs_map


def impute_from_non_nan_distribution(values: pd.Series, column_stats: dict) -> tuple[pd.Series, pd.Series]:
    """
    Impute NaNs with values from the empirical distributions of non-NaN rows.
    This is helpful especially in the low-data regime to avoid bias towards strong artificial patterns.
    It is applied when encoding columns with the following encoding types:
    - TABULAR_NUMERIC_DIGIT
    - TABULAR_DATETIME
    - TABULAR_LAT_LONG
    - TABULAR_CHARACTER

    Args:
        values: Series of values before encoding.
        column_stats: Column statistics.

    Returns:
        tuple[pd.Series, pd.Series]: The series with imputed values and the mask of NaNs.
    """
    nan_mask = values.isna()
    vc = values.value_counts(normalize=True)
    if vc.empty:
        return values, nan_mask.astype(int)
    probs = vc.values
    categories = vc.index
    # NOTE: an alternative will be to use the largest remainder method
    values[nan_mask] = np.random.choice(categories, size=nan_mask.sum(), p=probs)
    return values, nan_mask.astype(int)


def ensure_dataframe(X: Any, columns: list[str] | None = None) -> pd.DataFrame:
    """Convert array-like to DataFrame with column names."""
    if isinstance(X, pd.DataFrame):
        return X
    elif isinstance(X, np.ndarray):
        if columns is None:
            columns = [f"col_{i}" for i in range(X.shape[1])]
        return pd.DataFrame(X, columns=columns)
    elif hasattr(X, "__array__"):
        arr = np.asarray(X)
        if columns is None:
            columns = [f"col_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=columns)
    else:
        raise ValueError(f"Unsupported data type: {type(X)}")


def mode_fn(x: np.ndarray) -> Any:
    """
    Calculate the mode (most common value) of an array.

    Handles NaN values by excluding them from the calculation.
    If all values are NaN, returns np.nan.

    Args:
        x: Array of values.

    Returns:
        The most common value in the array, or np.nan if all values are NaN.
    """
    # Handle both numeric and categorical data
    if pd.isna(x).all():
        return np.nan
    x_notna = x[~pd.isna(x)]
    values, counts = np.unique(x_notna, return_counts=True)
    return values[np.argmax(counts)]


def mean_fn(x: np.ndarray) -> float:
    """
    Calculate the mean (average) of an array.

    Handles NaN values by excluding them from the calculation.
    If all values are NaN, returns np.nan.

    Args:
        x: Array of numeric values.

    Returns:
        The mean of the array, or np.nan if all values are NaN.
    """
    if pd.isna(x).all():
        return np.nan
    x_notna = x[~pd.isna(x)]
    return float(np.mean(x_notna))


def median_fn(x: np.ndarray) -> float:
    """
    Calculate the median (middle value) of an array.

    Handles NaN values by excluding them from the calculation.
    If all values are NaN, returns np.nan.

    Args:
        x: Array of numeric values.

    Returns:
        The median of the array, or np.nan if all values are NaN.
    """
    if pd.isna(x).all():
        return np.nan
    x_notna = x[~pd.isna(x)]
    return float(np.median(x_notna))


def list_fn(x: np.ndarray) -> np.ndarray:
    """
    Return the array as-is without aggregation.

    This function preserves all values in the array, useful when you want
    to keep all draws instead of aggregating them.

    Args:
        x: Array of values.

    Returns:
        The array as a numpy array.
    """
    return np.array(x)


def load_generated_data(workspace_dir: str | Path) -> pd.DataFrame:
    """
    Load generated synthetic data from workspace directory.

    Args:
        workspace_dir: Path to the workspace directory.

    Returns:
        DataFrame containing the generated synthetic data.
    """
    workspace_dir = Path(workspace_dir)
    synthetic_data_path = workspace_dir / "SyntheticData"

    # Read all parquet files from SyntheticData directory
    parquet_files = sorted(synthetic_data_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {synthetic_data_path}")

    # Read and concatenate all parquet files
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)
