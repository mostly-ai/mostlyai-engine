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

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed, parallel_config

from mostlyai.engine._common import (
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
    RIDX_SUB_COLUMN_PREFIX,
    SIDX_SUB_COLUMN_PREFIX,
    TGT,
    ProgressCallback,
    ProgressCallbackWrapper,
    encode_sidx_ridx,
    get_argn_name,
    get_sequence_length_stats,
    is_a_list,
    is_sequential,
)
from mostlyai.engine._encoding_types.tabular.categorical import encode_categorical
from mostlyai.engine._encoding_types.tabular.character import encode_character
from mostlyai.engine._encoding_types.tabular.datetime import encode_datetime
from mostlyai.engine._encoding_types.tabular.itt import encode_itt
from mostlyai.engine._encoding_types.tabular.lat_long import encode_latlong
from mostlyai.engine._encoding_types.tabular.numeric import encode_numeric
from mostlyai.engine._workspace import Workspace, ensure_workspace_dir, reset_dir
from mostlyai.engine.domain import ModelEncodingType
from mostlyai.engine.random_state import set_random_state

_LOG = logging.getLogger(__name__)


def encode(
    workspace_dir: str | Path | None = None,
    update_progress: ProgressCallback | None = None,
) -> None:
    _LOG.info("ENCODE_TABULAR started")
    t0 = time.time()
    with ProgressCallbackWrapper(update_progress) as progress:
        # build paths based on workspace dir
        workspace_dir = ensure_workspace_dir(workspace_dir)
        workspace = Workspace(workspace_dir)
        reset_dir(workspace.encoded_data_val.path)
        reset_dir(workspace.encoded_data_trn.path)
        reset_dir(workspace.encoded_data_path)

        has_context = workspace.ctx_data_path.exists()

        tgt_pqt_partitions = workspace.tgt_data.fetch_all()
        if has_context:
            ctx_pqt_partitions = workspace.ctx_data.fetch_all()
            if len(tgt_pqt_partitions) != len(ctx_pqt_partitions):
                raise RuntimeError("partition files for tgt and ctx do not match")
        else:
            ctx_pqt_partitions = []
        tgt_stats = workspace.tgt_stats.read()
        ctx_stats = workspace.ctx_stats.read()

        for i in range(len(tgt_pqt_partitions)):
            _encode_partition(
                tgt_partition_file=tgt_pqt_partitions[i],
                tgt_stats=tgt_stats,
                output_path=workspace.encoded_data_path,
                ctx_partition_file=ctx_pqt_partitions[i] if has_context else None,
                ctx_stats=ctx_stats if has_context else None,
                n_jobs=min(16, max(1, cpu_count() - 1)),
            )
            progress.update(completed=i, total=len(tgt_pqt_partitions) + 1)
    _LOG.info(f"ENCODE_TABULAR finished in {time.time() - t0:.2f}s")


def _encode_partition(
    *,
    tgt_partition_file: Path,
    tgt_stats: dict,
    output_path: Path,
    ctx_partition_file: Path | None = None,
    ctx_stats: dict | None = None,
    n_jobs: int = 1,
) -> None:
    seq_len_stats = get_sequence_length_stats(tgt_stats)
    is_sequential = tgt_stats["is_sequential"]

    tgt_context_key = tgt_stats.get("keys", {}).get("context_key")
    ctx_primary_key = ctx_stats.get("keys", {}).get("primary_key") if ctx_stats else None

    # encode target data
    df = pd.read_parquet(tgt_partition_file)
    df, _, tgt_context_key = encode_df(
        df,
        tgt_stats,
        ctx_primary_key=None,
        tgt_context_key=tgt_context_key,
        n_jobs=n_jobs,
    )

    has_context = ctx_partition_file is not None and tgt_context_key and ctx_primary_key
    ctx_stats = ctx_stats or {}
    if has_context:
        # check consistency of partitioned file names
        if ctx_partition_file and tgt_partition_file.name != ctx_partition_file.name:
            raise RuntimeError(f"mismatch: {tgt_partition_file}!={ctx_partition_file}")
        # encode context data
        df_ctx = pd.read_parquet(ctx_partition_file)
        df_ctx, ctx_primary_key, _ = encode_df(
            df_ctx,
            ctx_stats,
            ctx_primary_key=ctx_primary_key,
            tgt_context_key=None,
            n_jobs=n_jobs,
        )
        # pad each list with one extra item
        df_ctx = pad_ctx_sequences(df_ctx)

    if is_sequential:
        assert isinstance(tgt_context_key, str)
        # trim sequences to (privacy-protected) max_len
        max_len = seq_len_stats["max"]
        df = df[df.groupby(tgt_context_key).cumcount() < max_len].reset_index(drop=True)
        # pad each list with one extra item
        df = pad_tgt_sequences(df, context_key=tgt_context_key)
        # add empty records for IDs, that are present in context, but not in target; i.e., for zero-sequence records
        if has_context:
            zero_seq_ids = list(set(df_ctx[ctx_primary_key]) - set(df[tgt_context_key]))
            df_miss = pd.DataFrame({tgt_context_key: zero_seq_ids})
            df_pads = pd.DataFrame({c: [0] for c in df.columns if c != tgt_context_key})
            df_miss = df_miss.merge(df_pads, how="cross")
            df = pd.concat([df, df_miss], axis=0).reset_index(drop=True)
        # enrich with sequence lengths and sequence indexes
        df = _enrich_sidx_ridx(df, tgt_context_key, max_len)
        # flatten to list columns
        df = flatten_frame(df, tgt_context_key)
    elif has_context:
        # add 0-rows for IDs, that are present in context, but not in target; i.e., for zero-sequence records
        zero_seq_ids = list(set(df_ctx[ctx_primary_key]) - set(df[tgt_context_key]))
        df_miss = pd.DataFrame({tgt_context_key: zero_seq_ids})
        df_pads = pd.DataFrame({c: [0] for c in df.columns if c != tgt_context_key})
        df_miss = df_miss.merge(df_pads, how="cross")
        df = pd.concat([df, df_miss], axis=0).reset_index(drop=True)
        # ensure that max 1 item is retained per context_id for flat mode
        df = df[df.groupby(tgt_context_key).cumcount() < 1]

    # merge context with target
    if has_context:
        df = df_ctx.merge(df, left_on=ctx_primary_key, right_on=tgt_context_key, how="inner")

    # drop all key columns
    keys = [tgt_context_key, ctx_primary_key]
    df.drop(columns=keys, inplace=True, errors="ignore")

    # return, if nothing to write or if no target columns are present
    tgt_columns = [c for c in df.columns if c.startswith(TGT)]
    if df.empty or len(tgt_columns) == 0:
        return

    # shuffle and persist to disk as parquet files
    df = df.sample(frac=1)
    df.to_parquet(output_path / tgt_partition_file.name, engine="pyarrow", index=False)
    _LOG.info(f"encoded partition {tgt_partition_file.name} {df.shape}")


def encode_df(
    df: pd.DataFrame,
    stats: dict,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """
    Encodes a given table represented by a DataFrame object. The result will be delivered
    as DataFrame, as well.

    :param df: input data to be encoded
    :param stats: stats for each of the columns
    :param ctx_primary_key: context primary key
    :param tgt_context_key: target context key
    :return: encoded data and keys following columns' naming conventions
    """

    if ctx_primary_key and ctx_primary_key not in df:
        raise ValueError(f"primary key `{ctx_primary_key}` not present")
    if tgt_context_key and tgt_context_key not in df:
        raise ValueError(f"context key `{tgt_context_key}` not present")
    if ctx_primary_key and df[[ctx_primary_key]].duplicated().any():
        raise ValueError("Duplicate primary keys in table")

    df_columns = []

    # key columns
    context_keys = df[tgt_context_key] if tgt_context_key in df.columns else None
    if ctx_primary_key is not None:
        unique_primary_key = f"flt/primary_key.{ctx_primary_key}"
        df_columns.append(df[ctx_primary_key].reset_index(drop=True).to_frame(unique_primary_key))
        ctx_primary_key = unique_primary_key
    if tgt_context_key is not None:
        unique_context_key = f"context_key.{tgt_context_key}"
        df_columns.append(df[tgt_context_key].reset_index(drop=True).to_frame(unique_context_key))
        tgt_context_key = unique_context_key

    # data columns
    stats = stats or {"columns": {}}
    delayed_encodes = []
    for column in stats["columns"].keys():
        if column not in df:
            # skip encoding for columns that are not present in the data
            continue
        column_stats = stats["columns"][column]
        if not set(column_stats.keys()) - {"encoding_type"}:
            # all partitions are empty; skip encoding
            continue
        # encode (empty and non-empty) column
        delayed_encodes.append(
            delayed(_encode_col)(
                values=df[column],
                column_stats=column_stats,
                context_keys=context_keys,
            )
        )
    if delayed_encodes:
        with parallel_config("loky", n_jobs=n_jobs):
            df_columns.extend(Parallel()(delayed_encodes))

    df = pd.concat(df_columns, axis=1) if df_columns else pd.DataFrame()
    return df, ctx_primary_key, tgt_context_key


def _encode_col(
    values: pd.Series,
    column_stats: dict,
    context_keys: pd.Series | None = None,
) -> pd.DataFrame:
    set_random_state(worker=True)
    is_sequential_column = is_sequential(values)
    if is_sequential_column:
        # explode nested columns and encode the same way as flat columns
        non_empties = values.apply(lambda v: len(v) if is_a_list(v) else 1) > 0
        # generate serial context_keys, if context_keys are not provided
        context_keys = (
            context_keys
            if context_keys is not None
            else pd.Series(range(len(values)), index=values.index).rename("__ckey")
        )
        # explode non-empty values and context_keys in sync, remember sequence keys, reset index afterwards
        df = pd.concat([values[non_empties], context_keys[non_empties]], axis=1)
        df = df.explode(values.name)
        # trim sequences to max_seq_len
        max_seq_len = column_stats.get("seq_len", {}).get("max")
        if max_seq_len:
            df = df[df.groupby(context_keys.name).cumcount() < max_seq_len]
        sequence_keys = df.index
        df = df.reset_index(drop=True)
        values, context_keys = df[values.name], df[context_keys.name]

    df = _encode_flat_col(
        encoding_type=column_stats["encoding_type"],
        values=values,
        column_stats=column_stats,
        context_keys=context_keys,
    )

    if is_sequential_column:
        # flatten encoded, non-empty sequences
        encoded_sequences = flatten_frame(df.assign(__sequence_key=sequence_keys), "__sequence_key").set_index(
            "__sequence_key"
        )
        # inject empty sequences back into their original positions
        empty_sequences = pd.DataFrame(index=non_empties.index[~non_empties], columns=encoded_sequences.columns).map(
            lambda _: []
        )
        df = pd.concat([encoded_sequences, empty_sequences]).sort_index()

    df.reset_index(drop=True, inplace=True)

    if column_stats.get(ARGN_TABLE) is not None:
        # assign sanitized, unique column names for usage in ARGN
        df.columns = [
            get_argn_name(
                argn_processor=column_stats[ARGN_PROCESSOR],
                argn_table=column_stats[ARGN_TABLE],
                argn_column=column_stats[ARGN_COLUMN],
                argn_sub_column=sub_column,
            )
            for sub_column in df.columns
        ]

    return df


def _encode_flat_col(
    encoding_type: ModelEncodingType,
    values: pd.Series,
    column_stats: dict,
    context_keys: pd.Series | None,
) -> pd.DataFrame:
    if encoding_type == ModelEncodingType.tabular_categorical:
        df = encode_categorical(values, column_stats, context_keys)
    elif encoding_type in [
        ModelEncodingType.tabular_numeric_discrete,
        ModelEncodingType.tabular_numeric_binned,
        ModelEncodingType.tabular_numeric_digit,
    ]:
        df = encode_numeric(values, column_stats, context_keys)
    elif encoding_type == ModelEncodingType.tabular_datetime:
        df = encode_datetime(values, column_stats, context_keys)
    elif encoding_type == ModelEncodingType.tabular_datetime_relative:
        df = encode_itt(values, column_stats, context_keys)
    elif encoding_type == ModelEncodingType.tabular_character:
        df = encode_character(values, column_stats, context_keys)
    elif encoding_type == ModelEncodingType.tabular_lat_long:
        df = encode_latlong(values, column_stats, context_keys)
    else:
        raise RuntimeError(f"unknown encoding_type `{encoding_type}`")
    return df


def flatten_frame(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """
    Flattens non-key columns. See also `explode_frame` for reverse.

    This is an optimized flattening of data to rows by a given foreign key column. It is equivalent to
    `return data.groupby(foreign_key_column).agg(list).reset_index(level=0)`, but was benchmarked being
     significantly faster

    :example:
        this method converts
           key  product  is_paid
             1        3        0
             1        2        1
             2        9        1
        to
           key  product  is_paid
             1   [3, 2]   [0, 1]
             2      [9]      [1]
    """
    orig_columns_order = df.columns
    sorted_data = (
        df.reset_index(drop=True).rename_axis("__flatten_index").sort_values(by=[group_key, "__flatten_index"])
    )  # pre-group rows by FK, and preserve the order of their appearance by the index

    keys = sorted_data.pop(group_key).values.T
    values = sorted_data.values.T
    column_names = sorted_data.columns.values
    unique_keys, index = np.unique(keys, True)  # cutoff indexes for the groups (by FK)
    grouped_values = [
        np.split(value, index[1:]) for value in values
    ]  # a list of lists of arrays: column -> rows as a list -> grouped array of values

    flattened_data = {group_key: unique_keys}
    flattened_data.update(
        {
            column_name: [list(group) for group in column_values]
            for column_name, column_values in zip(column_names, grouped_values)
        }
    )

    flattened_data = (
        pd.DataFrame(flattened_data).reindex(columns=orig_columns_order)
        if len(unique_keys) > 0
        else pd.DataFrame(columns=orig_columns_order)
    )
    return flattened_data


def _enrich_sidx_ridx(df: pd.DataFrame, context_key: str, max_seq_len: int) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    sidx = df.groupby(context_key).cumcount(ascending=True)  # sequence index
    ridx = df.groupby(context_key).cumcount(ascending=False)  # sequence remainder
    sidx = encode_sidx_ridx(sidx, max_seq_len=max_seq_len, prefix=SIDX_SUB_COLUMN_PREFIX)
    ridx = encode_sidx_ridx(ridx, max_seq_len=max_seq_len, prefix=RIDX_SUB_COLUMN_PREFIX)
    df = pd.concat([sidx, ridx, df], axis=1)
    return df


def pad_tgt_sequences(df: pd.DataFrame, context_key: str, padding_value: int = 0) -> pd.DataFrame:
    """
    Pad one extra row to for each subject in the target data frame.

    Args:
        df: Exploded (unflattened) target data frame. Each event is a row.
        context_key: Context key.
        padding_value: Value to pad with for columns other than context_key.

    Returns:
        Padded target data frame.
    """

    def pad_row(x):
        return pd.concat(
            [
                x,
                pd.DataFrame(
                    {col: [padding_value] for col in x.columns if col not in [context_key]}
                    | {context_key: [x.iloc[0][context_key]]}
                ),
            ],
            axis=0,
        )

    return df.groupby(context_key)[df.columns].apply(pad_row).reset_index(drop=True)


def pad_ctx_sequences(df: pd.DataFrame, padding_value: int = 0) -> pd.DataFrame:
    """
    Pad one extra item to the context sequences with a given padding value.

    Args:
        df: Flattened context data frame.
        padding_value: Value to pad with.

    Returns:
        Padded context data frame.
    """
    if df.shape[0] == 0:
        return df
    list_cols = [c for c in df.columns if is_a_list(df.loc[0, c])]
    for col in list_cols:
        df[col] = df[col].apply(lambda x: x + [padding_value])
    return df
