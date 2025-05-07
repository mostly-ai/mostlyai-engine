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
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Literal,
    NamedTuple,
    Protocol,
)
from collections.abc import Callable, Iterable

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
TABLE_COLUMN_INFIX = "::"  # this should be consistent as in mostly-data and mostlyai-qa

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
        cardinalities |= get_slen_sidx_sdec_cardinalities(max_seq_len)

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
            "deciles": stats["seq_len"]["deciles"],
        }
    else:
        stats = {
            "min": 1,
            "median": 1,
            "max": 1,
            "deciles": [1 for i in range(11)],
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


def get_slen_sidx_sdec_cardinalities(max_seq_len) -> dict[str, int]:
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
    # order is important: slen first, then sidx, as the former has highest priority
    sdec_cardinalities = {f"{SDEC_SUB_COLUMN_PREFIX}cat": 10}
    return slen_cardinalities | sidx_cardinalities | sdec_cardinalities


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


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
