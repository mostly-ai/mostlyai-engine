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
Provides analysis functionality of the engine
"""

import logging
import time
from pathlib import Path
from typing import Any, Literal
from collections.abc import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config, cpu_count

from mostlyai.engine._common import (
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
    CTXFLT,
    CTXSEQ,
    TGT,
    is_a_list,
    is_sequential,
    read_json,
    set_random_state,
    write_json,
    TABLE_COLUMN_INFIX,
    ProgressCallback,
    ProgressCallbackWrapper,
)
from mostlyai.engine._encoding_types.language.datetime import (
    analyze_reduce_language_datetime,
    analyze_language_datetime,
)
from mostlyai.engine._encoding_types.language.numeric import analyze_language_numeric, analyze_reduce_language_numeric
from mostlyai.engine._encoding_types.tabular.categorical import (
    analyze_categorical,
    analyze_reduce_categorical,
)
from mostlyai.engine._encoding_types.tabular.character import (
    analyze_character,
    analyze_reduce_character,
)
from mostlyai.engine._encoding_types.tabular.datetime import (
    analyze_datetime,
    analyze_reduce_datetime,
)
from mostlyai.engine._encoding_types.tabular.itt import analyze_itt, analyze_reduce_itt
from mostlyai.engine._encoding_types.tabular.lat_long import (
    analyze_latlong,
    analyze_reduce_latlong,
)
from mostlyai.engine._encoding_types.tabular.numeric import (
    analyze_numeric,
    analyze_reduce_numeric,
)
from mostlyai.engine._encoding_types.language.text import (
    analyze_text,
    analyze_reduce_text,
)
from mostlyai.engine._encoding_types.language.categorical import (
    analyze_language_categorical,
    analyze_reduce_language_categorical,
)
from mostlyai.engine.domain import ModelEncodingType

from mostlyai.engine._workspace import (
    PathDesc,
    Workspace,
    ensure_workspace_dir,
    reset_dir,
)

_LOG = logging.getLogger(__name__)

_VALUE_PROTECTION_ENCODING_TYPES = (
    ModelEncodingType.tabular_categorical,
    ModelEncodingType.tabular_numeric_digit,
    ModelEncodingType.tabular_numeric_discrete,
    ModelEncodingType.tabular_numeric_binned,
    ModelEncodingType.tabular_datetime,
    ModelEncodingType.tabular_datetime_relative,
    ModelEncodingType.language_categorical,
    ModelEncodingType.language_numeric,
    ModelEncodingType.language_datetime,
)


def analyze(
    *,
    value_protection: bool = True,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
    random_state: int | None = None,
) -> None:
    """
    Generates (privacy-safe) column-level statistics of the original data, that has been `split` into the workspace.
    This information is required for encoding the original as well as for decoding the generating data.

    Creates the following folder structure within the `workspace_dir`:

    - `ModelStore/tgt-stats/stats.json`: Column-level statistics for target data
    - `ModelStore/ctx-stats/stats.json`: Column-level statistics for context data (if context is provided).

    Args:
        value_protection: Whether to enable value protection for rare values.
        workspace_dir: Path to workspace directory containing partitioned data.
        update_progress: Optional callback to update progress during analysis.
    """

    _LOG.info("ANALYZE started")
    t0 = time.time()
    set_random_state(random_state)
    with ProgressCallbackWrapper(update_progress) as progress:
        # build paths based on workspace dir
        workspace_dir = ensure_workspace_dir(workspace_dir)
        workspace = Workspace(workspace_dir)

        tgt_keys = workspace.tgt_keys.read()
        tgt_context_key = tgt_keys.get("context_key")
        ctx_keys = workspace.ctx_keys.read()
        ctx_primary_key = ctx_keys.get("primary_key")
        ctx_root_key = ctx_keys.get("root_key")

        has_context = workspace.ctx_data_path.exists()

        reset_dir(workspace.tgt_stats_path)
        if has_context:
            reset_dir(workspace.ctx_stats_path)

        tgt_pqt_partitions = workspace.tgt_data.fetch_all()
        if has_context:
            ctx_pqt_partitions = workspace.ctx_data.fetch_all()
            if len(tgt_pqt_partitions) != len(ctx_pqt_partitions):
                raise RuntimeError("partition files for tgt and ctx do not match")
        else:
            ctx_pqt_partitions = []

        _LOG.info(f"analyzing {len(tgt_pqt_partitions)} partitions in parallel")
        tgt_encoding_types = workspace.tgt_encoding_types.read()
        ctx_encoding_types = workspace.ctx_encoding_types.read()

        for i in range(len(tgt_pqt_partitions)):
            _analyze_partition(
                tgt_partition_file=tgt_pqt_partitions[i],
                tgt_stats_path=workspace.tgt_stats_path,
                tgt_encoding_types=tgt_encoding_types,
                tgt_context_key=tgt_context_key,
                ctx_partition_file=ctx_pqt_partitions[i] if has_context else None,
                ctx_stats_path=workspace.ctx_stats_path if has_context else None,
                ctx_encoding_types=ctx_encoding_types,
                ctx_primary_key=ctx_primary_key if has_context else None,
                ctx_root_key=ctx_root_key,
                n_jobs=min(16, max(1, cpu_count() - 1)),
            )
            progress.update(completed=i, total=len(tgt_pqt_partitions) + 1)

        # combine partition statistics
        _LOG.info("combine partition statistics")
        _analyze_reduce(
            all_stats=workspace.tgt_all_stats,
            out_stats=workspace.tgt_stats,
            keys=tgt_keys,
            mode="tgt",
            value_protection=value_protection,
        )
        if has_context:
            _analyze_reduce(
                all_stats=workspace.ctx_all_stats,
                out_stats=workspace.ctx_stats,
                keys=ctx_keys,
                mode="ctx",
                value_protection=True,  # always protect context values
            )

        # clean up partition-wise stats files, as they contain non-protected values
        for file in workspace.tgt_all_stats.fetch_all():
            file.unlink()
        for file in workspace.ctx_all_stats.fetch_all():
            file.unlink()
    _LOG.info(f"ANALYZE finished in {time.time() - t0:.2f}s")


def _analyze_partition(
    tgt_partition_file: Path,
    tgt_stats_path: Path,
    tgt_encoding_types: dict[str, ModelEncodingType],
    tgt_context_key: str | None = None,
    ctx_partition_file: Path | None = None,
    ctx_stats_path: Path | None = None,
    ctx_encoding_types: dict[str, ModelEncodingType] | None = None,
    ctx_primary_key: str | None = None,
    ctx_root_key: str | None = None,
    n_jobs: int = 1,
) -> None:
    """
    Calculates partial statistics about a single partition.

    If context exist, target and context partitions are analyzed jointly,
    thus single run can produce one or two partial statistics files.
    """

    has_context = ctx_partition_file is not None

    # read partitioned parquet file into memory
    tgt_df = pd.read_parquet(tgt_partition_file)
    partition_id = tgt_partition_file.name.split(".")[1]

    # get tgt context keys
    tgt_context_keys = (tgt_df[tgt_context_key] if tgt_context_key else pd.Series(range(tgt_df.shape[0]))).rename(
        "__ckey"
    )

    # get ctx primary keys
    if has_context:
        ctx_primary_keys = pd.read_parquet(ctx_partition_file, columns=[ctx_primary_key])[ctx_primary_key]
    else:
        ctx_primary_keys = tgt_context_keys.drop_duplicates()

    if ctx_root_key:
        ctx_root_keys = pd.read_parquet(ctx_partition_file, columns=[ctx_root_key])[ctx_root_key].rename("__rkey")
    else:
        ctx_root_keys = ctx_primary_keys.rename("__rkey")

    # analyze all target columns
    with parallel_config("loky", n_jobs=n_jobs):
        results = Parallel()(
            delayed(_analyze_col)(
                values=tgt_df[column],
                encoding_type=encoding_type,
                context_keys=tgt_context_keys,
            )
            for column, encoding_type in tgt_encoding_types.items()
        )
        tgt_column_stats = {column: stats for column, stats in zip(tgt_encoding_types.keys(), results)}

    # collect target sequence length stats
    tgt_seq_len = _analyze_seq_len(
        tgt_context_keys=tgt_context_keys,
        ctx_primary_keys=ctx_primary_keys,
    )

    # persist tgt stats
    tgt_stats_file = tgt_stats_path / f"part.{partition_id}.json"
    if "val" in partition_id:
        tgt_stats = {"no_of_training_records": 0, "no_of_validation_records": ctx_primary_keys.size}
    elif "trn" in partition_id:
        tgt_stats = {"no_of_training_records": ctx_primary_keys.size, "no_of_validation_records": 0}
    else:
        raise RuntimeError("partition file name must include 'trn' or 'val'")
    tgt_stats |= {
        "seq_len": tgt_seq_len,
        "columns": tgt_column_stats,
    }
    write_json(tgt_stats, tgt_stats_file)
    _LOG.info(f"analyzed target partition {partition_id} {tgt_df.shape}")

    if has_context:
        assert isinstance(ctx_partition_file, Path) and ctx_partition_file.exists()
        ctx_df = pd.read_parquet(ctx_partition_file)
        ctx_partition_id = ctx_partition_file.name.split(".")[1]
        if partition_id != ctx_partition_id:
            raise RuntimeError("partition files for tgt and ctx do not match")

        # analyze all context columns
        assert isinstance(ctx_encoding_types, dict)
        with parallel_config("loky", n_jobs=n_jobs):
            results = Parallel()(
                delayed(_analyze_col)(
                    values=ctx_df[column],
                    encoding_type=encoding_type,
                    root_keys=ctx_root_keys,
                )
                for column, encoding_type in ctx_encoding_types.items()
            )
            ctx_column_stats = {column: stats for column, stats in zip(ctx_encoding_types.keys(), results)}

        # persist context stats
        assert isinstance(ctx_stats_path, Path) and ctx_stats_path.exists()
        ctx_stats_file = ctx_stats_path / f"part.{partition_id}.json"
        ctx_stats = {
            "columns": ctx_column_stats,
        }
        write_json(ctx_stats, ctx_stats_file)
        _LOG.info(f"analyzed context partition {partition_id} {ctx_df.shape}")


def _analyze_reduce(
    all_stats: PathDesc,
    out_stats: PathDesc,
    keys: dict[str, str],
    mode: Literal["tgt", "ctx"],
    value_protection: bool = True,
) -> None:
    """
    Reduces partial statistics.

    Regardless of the provided argument 'mode', the function sequentially
    iterates over columns and for each it reduces partial column
    statistics. Those reduction procedures are column encoding type
    dependent and are defined in separate submodules.
    The important point is that rare / extreme value protection is applied during this step.

    If target partial statistics are reduced, some additional stats are
    recorded such as training / validation records number, sequence lengths
    summary and others.
    """

    stats_files = all_stats.fetch_all()
    stats_list = [read_json(file) for file in stats_files]
    stats: dict[str, Any] = {"columns": {}}

    encoding_types = {
        column: column_stats.get("encoding_type") for column, column_stats in stats_list[0]["columns"].items()
    }

    for column in encoding_types:
        encoding_type = encoding_types[column]
        column_stats_list = [item["columns"][column] for item in stats_list]
        column_stats_list = [
            column_stats
            for column_stats in column_stats_list
            if set(column_stats.keys()) - {"encoding_type"}  # skip empty partitions
        ]
        # all partitions are empty
        if not column_stats_list:
            # express that as {"encoding_type": ...} in stats
            stats["columns"][column] = {"encoding_type": encoding_type}
            continue

        if encoding_type == ModelEncodingType.tabular_categorical:
            stats_col = analyze_reduce_categorical(
                stats_list=column_stats_list,
                value_protection=value_protection,
            )
        elif encoding_type in [
            ModelEncodingType.tabular_numeric_auto,
            ModelEncodingType.tabular_numeric_digit,
            ModelEncodingType.tabular_numeric_discrete,
            ModelEncodingType.tabular_numeric_binned,
        ]:
            stats_col = analyze_reduce_numeric(
                stats_list=column_stats_list,
                value_protection=value_protection,
                encoding_type=encoding_type,
            )
        elif encoding_type == ModelEncodingType.tabular_datetime:
            stats_col = analyze_reduce_datetime(
                stats_list=column_stats_list,
                value_protection=value_protection,
            )
        elif encoding_type == ModelEncodingType.tabular_datetime_relative:
            stats_col = analyze_reduce_itt(
                stats_list=column_stats_list,
                value_protection=value_protection,
            )
        elif encoding_type == ModelEncodingType.tabular_character:
            stats_col = analyze_reduce_character(
                stats_list=column_stats_list,
                value_protection=value_protection,
            )
        elif encoding_type == ModelEncodingType.tabular_lat_long:
            stats_col = analyze_reduce_latlong(
                stats_list=column_stats_list,
            )
        elif encoding_type == ModelEncodingType.language_text:
            stats_col = analyze_reduce_text(stats_list=column_stats_list)
        elif encoding_type == ModelEncodingType.language_categorical:
            stats_col = analyze_reduce_text(stats_list=column_stats_list) | analyze_reduce_language_categorical(
                stats_list=column_stats_list,
                value_protection=value_protection,
            )
        elif encoding_type == ModelEncodingType.language_numeric:
            stats_col = analyze_reduce_text(stats_list=column_stats_list) | analyze_reduce_language_numeric(
                stats_list=column_stats_list,
                value_protection=value_protection,
            )
        elif encoding_type == ModelEncodingType.language_datetime:
            stats_col = analyze_reduce_text(stats_list=column_stats_list) | analyze_reduce_language_datetime(
                stats_list=column_stats_list,
                value_protection=value_protection,
            )
        else:
            raise RuntimeError(f"unknown encoding type {encoding_type}")

        # store encoding type, if it's not present yet
        stats_col = {"encoding_type": encoding_type} | stats_col
        # store flag indicating whether value protection was applied
        if encoding_type in _VALUE_PROTECTION_ENCODING_TYPES:
            stats_col = {"value_protection": value_protection} | stats_col

        is_flat_column = "seq_len" not in column_stats_list[0]
        if not is_flat_column:
            stats_col["seq_len"] = _analyze_reduce_seq_len([column_stats_list[0]["seq_len"]])

        is_language_column = encoding_type in (
            ModelEncodingType.language_text,
            ModelEncodingType.language_categorical,
            ModelEncodingType.language_numeric,
            ModelEncodingType.language_datetime,
        )

        if not is_language_column:
            # build mapping of original column name to ARGN table and column identifiers
            def get_table(qualified_column_name: str) -> str:
                # column names are assumed to be <table>::<column>
                return qualified_column_name.split(TABLE_COLUMN_INFIX)[0]

            def get_unique_tables(qualified_column_names: Iterable[str]) -> list[str]:
                duplicated_tables = [get_table(c) for c in qualified_column_names]
                return list(dict.fromkeys(duplicated_tables))

            unique_tables = get_unique_tables(encoding_types.keys())
            argn_identifiers: dict[str, tuple[str, str]] = {
                c: (f"t{unique_tables.index(get_table(qualified_column_name=c))}", f"c{idx}")
                for idx, c in enumerate(encoding_types.keys())
            }

            def get_argn_processor(mode, is_flat) -> str:
                if mode == "tgt":
                    return TGT
                else:  # mode == "ctx"
                    return CTXFLT if is_flat else CTXSEQ

            stats_col[ARGN_PROCESSOR] = get_argn_processor(mode, is_flat="seq_len" not in column_stats_list[0])
            (
                stats_col[ARGN_TABLE],
                stats_col[ARGN_COLUMN],
            ) = argn_identifiers[column]

        _LOG.info(
            f"analyzed column `{column}`: {stats_col['encoding_type']} "
            + (
                f"nchar_max={stats_col['nchar_max']} nchar_avg={stats_col['nchar_avg']}"
                if is_language_column
                else f"{stats_col['cardinalities']}"
            )
        )
        stats["columns"][column] = stats_col

    if mode == "ctx":
        # log ctxseq sequence length statistics
        deciles: dict[str, list[int]] = {}
        for column in stats["columns"]:
            if "seq_len" in stats["columns"][column]:  # ctxseq column
                table = get_table(column)
                if table not in deciles:  # first column in ctxseq table
                    deciles[table] = stats["columns"][column]["seq_len"]["deciles"]
        _LOG.info(f"ctxseq sequence length deciles: {deciles}")

    if mode == "tgt":
        # gather number of records and split into trn/val
        trn_cnt = sum(item["no_of_training_records"] for item in stats_list)
        val_cnt = sum(item["no_of_validation_records"] for item in stats_list)
        stats["no_of_training_records"] = trn_cnt
        stats["no_of_validation_records"] = val_cnt
        _LOG.info(f"analyzed {trn_cnt + val_cnt:,} records: {trn_cnt:,} training / {val_cnt:,} validation")
        # gather sequence length statistics
        stats["seq_len"] = _analyze_reduce_seq_len(
            stats_list=[item["seq_len"] for item in stats_list],
            value_protection=True,  # always protect sequence lengths
        )
        seq_len_min = stats["seq_len"]["min"]
        seq_len_max = stats["seq_len"]["max"]
        deciles = stats["seq_len"]["deciles"]
        _LOG.info(f"tgt sequence length deciles: {deciles}")
        # check whether data is sequential or not
        stats["is_sequential"] = seq_len_min != 1 or seq_len_max != 1
        _LOG.info(f"is_sequential: {stats['is_sequential']}")

    stats["keys"] = keys

    # persist statistics
    _LOG.info(f"write statistics to `{out_stats.path}`")
    out_stats.write(stats)


def _analyze_col(
    values: pd.Series,
    encoding_type: ModelEncodingType,
    root_keys: pd.Series | None = None,
    context_keys: pd.Series | None = None,
) -> dict:
    set_random_state()  # inherit random state from environment

    stats: dict = {"encoding_type": encoding_type}

    if values.empty:
        # empty partition columns are expressed as {"encoding_type": ...} in partial stats
        return stats

    if root_keys is None:
        root_keys = pd.Series([str(i) for i in range(len(values))], name="root_keys")

    if is_sequential(values):
        # analyze sequential column
        non_empties = values.apply(lambda v: len(v) if is_a_list(v) else 1) > 0
        # generate serial context_keys, if context_keys are not provided
        context_keys = context_keys if context_keys is not None else pd.Series(range(len(values))).rename("__ckey")
        # explode non-empty values and keys in sync, reset index afterwards
        df = pd.concat(
            [values[non_empties], root_keys[non_empties], context_keys[non_empties]],
            axis=1,
        )
        df = df.explode(values.name).reset_index(drop=True)
        # analyze sequence lengths
        cnt_lengths = _analyze_seq_len(df[root_keys.name], root_keys)
        stats |= _analyze_flat_col(encoding_type, df[values.name], df[root_keys.name], df[context_keys.name]) | {
            "seq_len": cnt_lengths
        }
    else:
        # analyze flat column
        stats |= _analyze_flat_col(encoding_type, values, root_keys, context_keys)

    return stats


def _analyze_flat_col(
    encoding_type: ModelEncodingType,
    values: pd.Series,
    root_keys: pd.Series,
    context_keys: pd.Series | None,
) -> dict:
    if encoding_type == ModelEncodingType.tabular_categorical:
        stats = analyze_categorical(values, root_keys, context_keys)
    elif encoding_type in [
        ModelEncodingType.tabular_numeric_auto,
        ModelEncodingType.tabular_numeric_digit,
        ModelEncodingType.tabular_numeric_discrete,
        ModelEncodingType.tabular_numeric_binned,
    ]:
        stats = analyze_numeric(values, root_keys, context_keys, encoding_type)
    elif encoding_type == ModelEncodingType.tabular_datetime:
        stats = analyze_datetime(values, root_keys, context_keys)
    elif encoding_type == ModelEncodingType.tabular_datetime_relative:
        stats = analyze_itt(values, root_keys, context_keys)
    elif encoding_type == ModelEncodingType.tabular_character:
        stats = analyze_character(values, root_keys, context_keys)
    elif encoding_type == ModelEncodingType.tabular_lat_long:
        stats = analyze_latlong(values, root_keys, context_keys)
    elif encoding_type == ModelEncodingType.language_text:
        stats = analyze_text(values, root_keys, context_keys)
    elif encoding_type == ModelEncodingType.language_categorical:
        stats = analyze_text(values, root_keys, context_keys) | analyze_language_categorical(
            values, root_keys, context_keys
        )
    elif encoding_type == ModelEncodingType.language_numeric:
        stats = analyze_text(values, root_keys, context_keys) | analyze_language_numeric(
            values, root_keys, context_keys
        )
    elif encoding_type == ModelEncodingType.language_datetime:
        stats = analyze_text(values, root_keys, context_keys) | analyze_language_datetime(
            values, root_keys, context_keys
        )
    else:
        raise RuntimeError(f"unknown encoding type: `{encoding_type}` for `{values.name}`")
    return stats


# SEQUENCE LENGTH


def _analyze_seq_len(
    tgt_context_keys: pd.Series,
    ctx_primary_keys: pd.Series,
) -> dict[str, Any]:
    # add extra mask record for each unique ctx_primary_key
    ctx_primary_keys = ctx_primary_keys.drop_duplicates()
    df_keys = pd.concat([tgt_context_keys, ctx_primary_keys]).to_frame()
    extra_rows = 1
    # count records per key
    values = df_keys.groupby(df_keys.columns[0]).size() - extra_rows
    # count records per sequence length
    cnt_lengths = values.value_counts().to_dict()
    stats = {"cnt_lengths": cnt_lengths}
    return stats


def _analyze_reduce_seq_len(stats_list: list[dict], value_protection: bool = True) -> dict:
    # gather sequence length counts
    cnt_lengths: dict[str, int] = {}
    for item in stats_list:
        for value, count in item["cnt_lengths"].items():
            cnt_lengths[value] = cnt_lengths.get(value, 0) + count
    # explode counts to np.array to gather statistics
    lengths = (
        np.sort(np.concatenate([np.repeat(int(k), v) for k, v in cnt_lengths.items()], axis=0))
        if len(cnt_lengths) > 0
        else np.empty(0)
    )
    if value_protection:
        # extreme value protection - discard lowest/highest 5 values
        if len(lengths) <= 10:
            # less or equal to 10 subjects; we need to protect all
            lengths = np.repeat(1, 10)
        else:
            lengths = lengths[5:-5]
    stats = {
        # calculate min/max for GENERATE
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        # calculate median for LSTM heuristic
        "median": int(np.median(lengths)),
        # calculate deciles of sequence lengths for bucket_by_seq_length
        "deciles": [int(v) for v in np.quantile(lengths, q=np.arange(0, 1.1, 0.1), method="inverted_cdf")]
        if len(lengths) > 0
        else [],
        "value_protection": value_protection,
    }
    return stats
