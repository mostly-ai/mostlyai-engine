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

import json
import logging
import time
import warnings
from pathlib import Path

import pandas as pd
from pandas.core.dtypes.common import is_datetime64_any_dtype
from tokenizers.pre_tokenizers import ByteLevel

from mostlyai.engine._common import is_sequential, ProgressCallback, ProgressCallbackWrapper, TABLE_COLUMN_INFIX
from mostlyai.engine._workspace import ensure_workspace_dir, Workspace, reset_dir

_LOG = logging.getLogger(__name__)


def format_df(df: pd.DataFrame, columns: list[str], is_target: bool = False) -> pd.DataFrame:
    df = df[columns].copy()
    # Some columns (e.g., SCP columns) may contain np.ndarray, which are not JSON serializable
    # We need to drop them before converting the DataFrame to JSON
    sequential_columns = [col for col in df.columns if is_sequential(df[col])]
    df = df.drop(columns=sequential_columns)
    _LOG.info(f"Formatting {'target' if is_target else 'context'} columns {df.columns.tolist()} to JSON")
    # convert date format to ISO so that it's JSON serializable
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df.apply(lambda x: row_to_json(x, is_target=is_target), axis=1)


def fallback_serializer(obj):
    warnings.warn(f"{type(obj)} is not JSON serializable. Converting it to str(obj) instead.")
    return str(obj)


def row_to_json(row: pd.Series, is_target: bool = False) -> str:
    row_dict = row.to_dict()
    nested_dict = {}
    for key, value in row_dict.items():
        tokens = key.split(TABLE_COLUMN_INFIX, maxsplit=1)
        if len(tokens) == 1:
            table_name = "tgt"  # assign a default table name for the target table
            column_name = tokens[0]  # columns from the target table
        else:  # len(tokens) == 2
            table_name, column_name = tokens  # prefixed columns from the context tables
        if table_name not in nested_dict:
            nested_dict[table_name] = {}
        nested_dict[table_name][column_name] = value
    if is_target:
        # assume there is at most one item in nested_dict when is_target=True
        nested_dict = next(iter(nested_dict.values()), {})
    # The leading space is required to avoid inconsistent tokenization results during training and generation
    # This ensures that '{' of the context and target json are tokenized into the same thing
    return " " + json.dumps(
        nested_dict,
        ensure_ascii=False,
        separators=None,
        indent=None,
        default=fallback_serializer,
    )


def encode_df(
    ctx_df: pd.DataFrame,
    ctx_columns: list[str],
    tgt_df: pd.DataFrame | None = None,
    tgt_columns: list[str] | None = None,
) -> pd.DataFrame:
    assert (tgt_df is None) == (tgt_columns is None), "tgt_df and tgt_columns must be both None or both not None"
    df = pd.DataFrame()
    df["ctx"] = format_df(ctx_df, columns=ctx_columns, is_target=False)
    if tgt_df is not None and tgt_columns is not None:
        df["tgt"] = format_df(tgt_df, columns=tgt_columns, is_target=True)

    # log the bounds of n_tokens in this partition
    content = df["ctx"] + df["tgt"] if "tgt" in df.columns else df["ctx"]
    n_chars_stats = content.apply(lambda x: len(x)).describe(percentiles=[0.5]).rename("#chars")
    pretokenizer = ByteLevel(add_prefix_space=True, use_regex=True)
    n_pretokens_stats = (
        content.apply(lambda x: len(pretokenizer.pre_tokenize_str(x))).describe(percentiles=[0.5]).rename("#pretokens")
    )
    stats = pd.concat([n_pretokens_stats, n_chars_stats], axis=1).loc[["min", "50%", "max"], :]
    _LOG.info(f"token statistics of this partition: \n{stats}")
    return df


def _encode_partition(
    *,
    tgt_partition_file: Path,
    tgt_stats: dict,
    output_path: Path,
    ctx_partition_file: Path | None = None,
    ctx_stats: dict | None = None,
) -> None:
    tgt_df = pd.read_parquet(tgt_partition_file)
    tgt_columns = list(tgt_stats.get("columns", {}).keys())
    if ctx_partition_file:
        ctx_df = pd.read_parquet(ctx_partition_file)
        ctx_columns = list(ctx_stats.get("columns", {}).keys())
    else:
        # create on-the-fly context
        ctx_df = pd.DataFrame(index=range(len(tgt_df)))
        ctx_columns = []
    df = encode_df(
        ctx_df=ctx_df,
        ctx_columns=ctx_columns,
        tgt_df=tgt_df,
        tgt_columns=tgt_columns,
    )
    # shuffle and persist to disk as parquet files
    df = df.sample(frac=1)
    df.to_parquet(output_path / tgt_partition_file.name, engine="pyarrow", index=False)
    _LOG.info(f"encoded partition {tgt_partition_file.name} {df.shape}")


def encode(workspace_dir: str | Path | None = None, update_progress: ProgressCallback | None = None) -> None:
    _LOG.info("ENCODE_LANGUAGE started")
    t0 = time.time()
    with ProgressCallbackWrapper(update_progress) as progress:
        workspace_dir = ensure_workspace_dir(workspace_dir)
        workspace = Workspace(workspace_dir)
        reset_dir(workspace.encoded_data_val.path)
        reset_dir(workspace.encoded_data_trn.path)
        reset_dir(workspace.encoded_data_path)
        has_context = workspace.ctx_data_path.exists()

        tgt_stats = workspace.tgt_stats.read()
        tgt_pqt_partitions = workspace.tgt_data.fetch_all()

        if has_context:
            ctx_stats = workspace.ctx_stats.read()
            ctx_pqt_partitions = workspace.ctx_data.fetch_all()
            if len(tgt_pqt_partitions) != len(ctx_pqt_partitions):
                raise RuntimeError("partition files for tgt and ctx do not match")
        else:
            ctx_stats = None
            ctx_pqt_partitions = None

        for i in range(len(tgt_pqt_partitions)):
            _encode_partition(
                tgt_partition_file=tgt_pqt_partitions[i],
                tgt_stats=tgt_stats,
                output_path=workspace.encoded_data_path,
                ctx_partition_file=ctx_pqt_partitions[i] if has_context else None,
                ctx_stats=ctx_stats if has_context else None,
            )
            progress.update(completed=i, total=len(tgt_pqt_partitions) + 1)
    _LOG.info(f"ENCODE_LANGUAGE finished in {time.time() - t0:.2f}s")
