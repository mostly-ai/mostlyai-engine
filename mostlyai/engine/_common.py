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
SLEN_SIDX_SDEC_COLUMN = f"{TGT}{PREFIX_TABLE}{PREFIX_COLUMN}"
SLEN_SIDX_DIGIT_ENCODING_THRESHOLD = 100
SLEN_SUB_COLUMN_PREFIX = f"{SLEN_SIDX_SDEC_COLUMN}{PREFIX_SUB_COLUMN}slen_"  # sequence length
SIDX_SUB_COLUMN_PREFIX = f"{SLEN_SIDX_SDEC_COLUMN}{PREFIX_SUB_COLUMN}sidx_"  # sequence index
SDEC_SUB_COLUMN_PREFIX = f"{SLEN_SIDX_SDEC_COLUMN}{PREFIX_SUB_COLUMN}sdec_"  # sequence index decile
STOP_SUB_COLUMN_PREFIX = f"{SLEN_SIDX_SDEC_COLUMN}{PREFIX_SUB_COLUMN}stop_"  # sequence stop token
TABLE_COLUMN_INFIX = "::"  # this should be consistent as in mostly-data and mostlyai-qa

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


def get_cardinalities(stats: dict) -> dict[str, int]:
    cardinalities: dict[str, int] = {}
    if stats.get("is_sequential", False):
        max_seq_len = get_sequence_length_stats(stats)["max"]
        cardinalities |= get_sidx_slen_stop_cardinalities(max_seq_len)

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


class CtxSequenceLengthError(Exception):
    """Error raised when the cols of the same table do not have the same stats value"""


def get_ctx_sequence_length(ctx_stats: dict, key: str) -> dict[str, int]:
    seq_stats: dict[str, int] = {}

    for column_stats in ctx_stats.get("columns", {}).values():
        if "seq_len" in column_stats:
            table = get_argn_name(
                argn_processor=column_stats[ARGN_PROCESSOR],
                argn_table=column_stats[ARGN_TABLE],
            )
            cur_value = seq_stats.get(table)
            if cur_value and cur_value != column_stats["seq_len"][key]:
                raise CtxSequenceLengthError()
            seq_stats[table] = column_stats["seq_len"][key]

    return seq_stats


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


def encode_slen_sidx_sdec(vals: pd.Series, max_seq_len: int, prefix: str = "") -> pd.DataFrame:
    assert is_integer_dtype(vals)
    if max_seq_len < SLEN_SIDX_DIGIT_ENCODING_THRESHOLD or prefix == SDEC_SUB_COLUMN_PREFIX:
        # encode slen and sidx as numeric_discrete
        df = pd.DataFrame({f"{prefix}cat": vals})
    else:
        # encode as numeric_digit
        n_digits = len(str(max_seq_len))
        df = pd.DataFrame(vals.astype(str).str.pad(width=n_digits, fillchar="0").apply(list).tolist()).astype(int)
        df.columns = [f"{prefix}E{i}" for i in range(n_digits - 1, -1, -1)]
    return df


def decode_slen_sidx_sdec(df_encoded: pd.DataFrame, max_seq_len: int, prefix: str = "") -> pd.Series:
    if max_seq_len < SLEN_SIDX_DIGIT_ENCODING_THRESHOLD or prefix == SDEC_SUB_COLUMN_PREFIX:
        # decode slen and sidx as numeric_discrete
        vals = df_encoded[f"{prefix}cat"]
    else:
        # decode slen and sidx as numeric_digit
        n_digits = len(str(max_seq_len))
        vals = sum([df_encoded[f"{prefix}E{d}"] * 10 ** int(d) for d in list(range(n_digits))])
    return vals


def get_sidx_slen_stop_cardinalities(max_seq_len) -> dict[str, int]:
    if max_seq_len < SLEN_SIDX_DIGIT_ENCODING_THRESHOLD:
        # encode slen and sidx as numeric_discrete
        slen_cardinalities = {f"{SLEN_SUB_COLUMN_PREFIX}cat": max_seq_len + 1}
        sidx_cardinalities = {f"{SIDX_SUB_COLUMN_PREFIX}cat": max_seq_len + 1}
    else:
        # encode slen and sidx as numeric_digit
        digits = [int(digit) for digit in str(max_seq_len)]
        slen_cardinalities = {}
        sidx_cardinalities = {}
        for idx, digit in enumerate(digits):
            # cap cardinality of the most significant position
            # less significant positions allow any digit
            card = digit + 1 if idx == 0 else 10
            e_idx = len(digits) - idx - 1
            slen_cardinalities[f"{SLEN_SUB_COLUMN_PREFIX}E{e_idx}"] = card
            sidx_cardinalities[f"{SIDX_SUB_COLUMN_PREFIX}E{e_idx}"] = card
    stop_cardinalities = {f"{STOP_SUB_COLUMN_PREFIX}cat": 2}
    # order is important: sidx first, then slen, then stop,as the latter should be informed by the former
    return sidx_cardinalities | slen_cardinalities | stop_cardinalities


def trim_sequences(syn: pd.DataFrame, tgt_context_key: str, seq_len_min: int, seq_len_max: int):
    if syn.empty:
        return syn

    # use SIDX and SLEN to determine sequence length
    syn[SIDX_SUB_COLUMN_PREFIX] = decode_slen_sidx_sdec(syn, seq_len_max, prefix=SIDX_SUB_COLUMN_PREFIX)
    syn[SLEN_SUB_COLUMN_PREFIX] = decode_slen_sidx_sdec(syn, seq_len_max, prefix=SLEN_SUB_COLUMN_PREFIX)
    # ensure that seq_len_min is respected
    syn[SLEN_SUB_COLUMN_PREFIX] = np.maximum(seq_len_min, syn[SLEN_SUB_COLUMN_PREFIX])
    syn = syn[syn[SIDX_SUB_COLUMN_PREFIX] < syn[SLEN_SUB_COLUMN_PREFIX]].reset_index(drop=True)
    # discarded padded context rows, ie where context key has been set to None
    syn = syn.dropna(subset=[tgt_context_key])
    # discard SLEN and SIDX columns
    syn.drop(
        [c for c in syn.columns if c.startswith(SLEN_SIDX_SDEC_COLUMN)],
        axis=1,
        inplace=True,
    )
    syn.reset_index(drop=True, inplace=True)
    return syn


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


def _get_log_histogram_edges(idx: int, bins: int = 64) -> tuple[float, float]:
    """
    Modified from OpenDP's SmartNoise SDK (MIT License)
    Source: https://github.com/opendp/smartnoise-sdk/blob/main/sql/snsql/sql/_mechanisms/approx_bounds.py
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
    Modified from OpenDP's SmartNoise SDK (MIT License)
    Source: https://github.com/opendp/smartnoise-sdk/blob/main/sql/snsql/sql/_mechanisms/approx_bounds.py
    """
    hist = [0.0] * bins * 2

    values = np.array(values, dtype=np.float64)
    values = values[values != np.inf]
    values = values[values != -np.inf]
    values = values[~np.isnan(values)]
    edge_list = [_get_log_histogram_edges(idx) for idx in range(len(hist))]
    min_val = min([lower for lower, _ in edge_list])
    max_val = max([upper for _, upper in edge_list]) - 1
    values = np.clip(values, min_val, max_val)

    # compute histograms
    for v in values:
        bin = None
        for idx, (lower, upper) in enumerate(edge_list):
            if lower <= v < upper:
                bin = idx
                break
        if bin is None:
            bin = idx
        hist[bin] += 1

        # for testing
        lower, upper = _get_log_histogram_edges(bin)
    return hist


def dp_approx_bounds(hist: list[int], epsilon: float) -> tuple[float | None, float | None]:
    """
    Modified from OpenDP's SmartNoise SDK (MIT License)
    Source: https://github.com/opendp/smartnoise-sdk/blob/main/sql/snsql/sql/_mechanisms/approx_bounds.py

    Estimate the minimium and maximum values of a list of values.
    from: https://desfontain.es/thesis/Usability.html#usability-u-ding-

    Args:
        hist (list[int]): A list of log histogram counts.
        epsilon (float): The privacy budget to spend estimating the bounds.

    Returns:
        tuple[float | None, float | None]: A tuple of the estimated minimum and maximum values.
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
    lower, _ = _get_log_histogram_edges(lower_bin)
    _, upper = _get_log_histogram_edges(upper_bin)
    return (float(lower), float(upper))


def _dp_bounded_quantiles(
    values: np.ndarray, quantiles: list[float], epsilon: float, lower: float, upper: float
) -> list[float]:
    """
    Estimate the quantile.
    from: http://cs-people.bu.edu/ads22/pubs/2011/stoc194-smith.pdf

    Args:
        values (np.ndarray): A 1D array of numeric values.
        quantiles (list[float]): List of probabilities of the quantiles to estimate.
        epsilon (float): Privacy budget.
        lower (float): A bounding parameter. The quantile will be estimated only for values greater than or equal to this bound.
        upper (float): A bounding parameter. The quantile will be estimated only for values less than or equal to this bound.

    Returns:
        list[float]: The estimated quantile.
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


# NOTE: the unbounded method is not used in the current implementation
# def _dp_unbounded_quantiles(
#     values: np.ndarray, quantiles: list[float], epsilon: float, beta: float = 1.01
# ) -> tuple[list[float], float]:
#     """
#     Fully unbounded differentially private quantile estimation using two AboveThreshold calls
#     with Exponential noise (one-sided Laplace).

#     Implements Algorithm 4 from Durfee (2023):
#       1) AboveThreshold on positives: T1 = q*n, f_i = |{x_j + 1 < beta^i}|
#       2) AboveThreshold on negatives: T2 = (1-q)*n, f_i = |{x_j - 1 > -beta^i}|
#       3) If first halts at k>0: return  beta^k - 1
#       4) If second halts at k>0: return -beta^k + 1
#       5) Otherwise return 0

#     Args:
#         values (np.ndarray): A 1D array of numeric values.
#         quantiles (list[float]): List of probabilities of the quantiles to estimate.
#         epsilon (float): Privacy budget.
#         beta (float): Multiplicative step size (default 1.01). Section 6.4 from Durfee (2023) suggests the range [1.01, 1.001] and 1.01 for general use, especially for more significant
#         decreases in epsilon or in the data size.

#     Returns:
#         list[float]: Differentially private estimates of the quantiles.
#     """

#     def above_threshold(
#         values: np.ndarray, q: float, eps: float, beta: float, is_positive_side: bool
#     ) -> tuple[int, float]:
#         n = len(values)
#         eps1 = eps2 = eps / 2.0
#         T = q * n if is_positive_side else (1 - q) * n
#         noisy_T = T + np.random.exponential(scale=1 / eps1)
#         i = 0
#         while True:
#             candidate = beta**i - 1 if is_positive_side else -(beta**i - 1)
#             f_i = (values < candidate).sum() if is_positive_side else (values > candidate).sum()
#             noisy_f_i = f_i + np.random.exponential(scale=1 / eps2)
#             if noisy_f_i >= noisy_T:
#                 return i, candidate
#             i += 1

#     _LOG.info("compute DP unbounded quantiles")
#     # Split epsilon across quantiles and the two AboveThreshold calls per quantile
#     eps_pass = epsilon / len(quantiles) / 2.0

#     results = []
#     for q in quantiles:
#         # 1) Positive-side AboveThreshold
#         k, candidate = above_threshold(values, q, eps_pass, beta, is_positive_side=True)
#         if k > 0:
#             results.append(candidate)
#         else:
#             # 2) Continue with negative-side AboveThreshold only if the first one did not halt at k > 0
#             k, candidate = above_threshold(values, q, eps_pass, beta, is_positive_side=False)
#             if k > 0:
#                 results.append(candidate)
#             else:
#                 # 3) Return 0 if both AboveThreshold calls did not halt at k > 0
#                 results.append(0.0)

#     # ensure monotonicity of results with respect to quantiles
#     sorted_indices = [t[0] for t in sorted(enumerate(quantiles), key=lambda x: x[1])]
#     sorted_results = sorted(results)
#     results = [sorted_results[sorted_indices.index(i)] for i in range(len(quantiles))]

#     # NOTE: consider returning the actual epsilon spent in the future, so that the unused budget can be used for training later
#     return results


def dp_quantiles(values: list | np.ndarray, quantiles: list[float], epsilon: float) -> list[float]:
    """
    Differentially private quantile estimation.
    First, estimate the bounds of the values, then use the bounds to estimate the quantiles.
    If the bounds are not available, estimate the quantiles using the unbounded method.

    Args:
        values (list | np.ndarray): A list of numeric values.
        quantiles (list[float]): List of probabilities of the quantiles to estimate.
        epsilon (float): Privacy budget.

    Returns:
        list[float]: The estimated quantiles.
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
    Differentially private selection of all categories whose true count >= threshold,
    via the Laplace vector mechanism + post-processing.

    Args:
        value_counts (dict): Mapping from category to its count.
        epsilon (float): Privacy budget.
        threshold (int): Threshold for non-rare values.

    Returns:
        list[str]: Categories whose noisy counts are above the threshold (DP guarantee: ε-DP).
        float: Non-rare ratio (DP guarantee: ε-DP).
    """

    # 1. Add independent Laplace(1/ε) noise to each count (vector Laplace mechanism)
    # Note: sensitivity of the count vector is 1 in L1 norm
    noise = np.random.laplace(loc=0.0, scale=1 / epsilon, size=len(value_counts))
    noisy_counts = np.clip(np.array(list(value_counts.values())) + noise, 0, None).astype(int)
    for i, cat in enumerate(value_counts):
        value_counts[cat] = noisy_counts[i]
    total_counts = sum(value_counts.values())

    # 2. Collect all categories whose noisy count >= threshold
    selected = {cat: nc for cat, nc in value_counts.items() if nc >= threshold}

    # 3. Compute the non-rare ratio
    noisy_total_counts = sum(selected.values())
    non_rare_ratio = noisy_total_counts / total_counts

    return list(selected.keys()), non_rare_ratio


def get_stochastic_rare_threshold(min_threshold: int = 5, noise_multiplier: float = 3) -> int:
    return min_threshold + int(noise_multiplier * np.random.uniform())
