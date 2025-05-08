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
Split original data for training and validation.
"""

import time
import warnings
from pathlib import Path

from mostlyai.engine._common import ProgressCallbackWrapper, ProgressCallback, set_random_state
from mostlyai.engine._dtypes import (
    is_date_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_timestamp_dtype,
)

import numpy as np
import logging

import pandas as pd

from mostlyai.engine.domain import ModelType, ModelEncodingType
from mostlyai.engine._workspace import (
    PathDesc,
    Workspace,
    ensure_workspace_dir,
    reset_dir,
)

warnings.simplefilter(action="ignore", category=UserWarning)

_LOG = logging.getLogger(__name__)


def _get_default_tabular_encoding_type(x: pd.Series) -> ModelEncodingType:
    if is_integer_dtype(x) or is_float_dtype(x):
        return ModelEncodingType.tabular_numeric_auto
    elif is_date_dtype(x) or is_timestamp_dtype(x):
        return ModelEncodingType.tabular_datetime
    else:
        return ModelEncodingType.tabular_categorical


def _get_default_language_encoding_type(x: pd.Series) -> ModelEncodingType:
    if is_integer_dtype(x) or is_float_dtype(x):
        return ModelEncodingType.language_numeric
    elif is_date_dtype(x) or is_timestamp_dtype(x):
        return ModelEncodingType.language_datetime
    else:
        return ModelEncodingType.language_categorical


def split(
    tgt_data: pd.DataFrame,
    *,
    ctx_data: pd.DataFrame | None = None,
    tgt_primary_key: str | None = None,
    ctx_primary_key: str | None = None,
    tgt_context_key: str | None = None,
    model_type: str | ModelType | None = None,
    tgt_encoding_types: dict[str, str | ModelEncodingType] | None = None,
    ctx_encoding_types: dict[str, str | ModelEncodingType] | None = None,
    n_partitions: int = 1,
    trn_val_split: float = 0.9,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
    random_state: int | None = None,
) -> None:
    """
    Splits the provided original data into training and validation sets, and stores these as partitioned Parquet files.
    This is a simplified version of `mostlyai-data`, tailored towards single- and two-table use cases, while requiring
    all data to be passed as DataFrames in memory.

    Creates the following folder structure within the `workspace_dir`:

      - `OriginalData/tgt-data`: Partitioned target data files.
      - `OriginalData/tgt-meta`: Metadata files for target data.
      - `OriginalData/ctx-data`: Partitioned context data files (if context is provided).
      - `OriginalData/ctx-meta`: Metadata files for context data (if context is provided).

    Args:
        tgt_data: DataFrame containing the target data.
        ctx_data: DataFrame containing the context data.
        tgt_primary_key: Primary key column name in the target data.
        ctx_primary_key: Primary key column name in the context data.
        tgt_context_key: Context key column name in the target data.
        model_type: Model type for the target data. If not provided, it will be inferred from the encoding types, or set to TABULAR by default.
        tgt_encoding_types: Encoding types for columns in the target data (excluding key columns).
        ctx_encoding_types: Encoding types for columns in the context data (excluding key columns).
        n_partitions: Number of partitions to split the data into.
        trn_val_split: Fraction of data to use for training, with the remaining data used for validation.
        workspace_dir: Path to the workspace directory where files will be created.
        update_progress: A custom progress callback.
        random_state: Seed for the random number generators.
    """
    _LOG.info("SPLIT started")
    t0 = time.time()
    set_random_state(random_state)
    with ProgressCallbackWrapper(update_progress) as progress:
        # validate input
        if tgt_primary_key and tgt_primary_key not in tgt_data:
            raise Exception("tgt_primary_key not found in tgt_data")
        if tgt_primary_key and tgt_data[tgt_primary_key].duplicated().any():
            raise Exception("tgt_primary_key must be unique")
        if ctx_data is not None:
            if tgt_context_key is None:
                raise Exception("tgt_context_key must be provided")
            if tgt_context_key not in tgt_data:
                raise Exception("tgt_context_key not found in tgt_data")
            if ctx_primary_key is None:
                raise Exception("ctx_primary_key must be provided")
            if ctx_primary_key not in ctx_data:
                raise Exception("ctx_primary_key not found in ctx_data")
            if ctx_data[ctx_primary_key].duplicated().any():
                raise Exception("ctx_primary_key must be unique")
            if not tgt_data[tgt_context_key].isin(ctx_data[ctx_primary_key]).all():
                raise Exception("tgt_primary_keys must match ctx_primary_keys")

        # prepare data
        if ctx_data is None and tgt_context_key is not None:
            ctx_primary_key = ctx_primary_key or tgt_context_key
            ctx_data = pd.DataFrame({ctx_primary_key: tgt_data[tgt_context_key].drop_duplicates()})

        # prepare workspace directory
        workspace_dir = ensure_workspace_dir(workspace_dir)
        ws = Workspace(workspace_dir)
        reset_dir(ws.tgt_data_path)
        reset_dir(ws.tgt_meta_path)
        if ctx_data is not None:
            reset_dir(ws.ctx_data_path)
            reset_dir(ws.ctx_meta_path)

        # set model_type if not provided
        if not model_type and not tgt_encoding_types:
            # use TABULAR as default
            model_type = ModelType.tabular
        elif not model_type and tgt_encoding_types:
            # consider ModelType of the first encoding type
            first_enc_type = next(iter(tgt_encoding_types.values()))
            model_type = ModelType.tabular if first_enc_type.startswith(ModelType.tabular.value) else ModelType.language
        model_type = ModelType(model_type).value

        # populate model_encoding_types with defaults if not specified
        tgt_encoding_types = tgt_encoding_types or {}
        tgt_cols = [c for c in tgt_data if c != tgt_primary_key and c != tgt_context_key]
        for col in tgt_cols:
            if col not in tgt_encoding_types:
                if model_type == ModelType.language:
                    tgt_encoding_types[col] = _get_default_language_encoding_type(tgt_data[col]).value
                else:
                    tgt_encoding_types[col] = _get_default_tabular_encoding_type(tgt_data[col]).value
            else:
                tgt_encoding_types[col] = ModelEncodingType(tgt_encoding_types[col]).value
        ctx_encoding_types = ctx_encoding_types or {}
        ctx_cols = [c for c in ctx_data if c != ctx_primary_key] if ctx_data is not None else []
        for col in ctx_cols:
            if col not in ctx_encoding_types:
                ctx_encoding_types[col] = _get_default_tabular_encoding_type(ctx_data[col]).value
            else:
                ctx_encoding_types[col] = ModelEncodingType(ctx_encoding_types[col]).value
        _LOG.info(f"{model_type=}")
        _LOG.info(f"{tgt_encoding_types=}")
        if ctx_encoding_types:
            _LOG.info(f"{ctx_encoding_types=}")

        # check if tgt_encoding_types are consistent with model_type
        for col, enc_type in tgt_encoding_types.items():
            if not enc_type.startswith(model_type):
                raise Exception("Mismatch between model_type and model_encoding_types")

        # split into trn/val
        tmp_key = "__key"
        if ctx_data is None and tgt_context_key is None:
            # assign temporary key to enable consistent splitting logic; we drop that key later on again
            tgt_data = tgt_data.copy()
            tgt_data[tmp_key] = range(len(tgt_data))
            tgt_context_key = tmp_key
        # gather all keys
        if ctx_data is None:
            keys = tgt_data[tgt_context_key].drop_duplicates()
        else:
            keys = ctx_data[ctx_primary_key]
        # shuffle keys
        keys = keys.sample(frac=1)
        # split randomly into trn and val
        assert 0 < trn_val_split < 1, f"invalid trn_val_split: {trn_val_split}"
        trn_cnt = round(trn_val_split * len(keys))
        trn_keys = keys[:trn_cnt]
        val_keys = keys[trn_cnt:]

        def save_partition(
            df: pd.DataFrame, path: Path, key: str, sel_trn_keys: np.ndarray, sel_val_keys: np.ndarray, idx: int
        ):
            trn_df = df[df[key].isin(sel_trn_keys)].drop(columns=[tmp_key], errors="ignore")
            val_df = df[df[key].isin(sel_val_keys)].drop(columns=[tmp_key], errors="ignore")
            trn_df.to_parquet(path / f"part.{idx:06d}-trn.parquet", engine="pyarrow", index=False)
            val_df.to_parquet(path / f"part.{idx:06d}-val.parquet", engine="pyarrow", index=False)

        # save tgt-data
        for i in range(n_partitions):
            save_partition(
                df=tgt_data,
                path=ws.tgt_data_path,
                key=tgt_context_key,
                sel_trn_keys=trn_keys[i::n_partitions],
                sel_val_keys=val_keys[i::n_partitions],
                idx=i,
            )
            progress.update(completed=i, total=2 * n_partitions)
        if tgt_context_key == tmp_key:
            tgt_context_key = None
        # save tgt-meta
        _save_meta_encoding_types(tgt_encoding_types, ws.tgt_encoding_types)
        _save_meta_keys(primary_key=tgt_primary_key, context_key=tgt_context_key, path=ws.tgt_keys)

        if ctx_data is not None:
            # save ctx-data
            for i in range(n_partitions):
                save_partition(
                    df=ctx_data,
                    path=ws.ctx_data_path,
                    key=ctx_primary_key,
                    sel_trn_keys=trn_keys[i::n_partitions],
                    sel_val_keys=val_keys[i::n_partitions],
                    idx=i,
                )
                progress.update(completed=n_partitions + i, total=2 * n_partitions)
            # save ctx-meta
            _save_meta_keys(primary_key=ctx_primary_key, path=ws.ctx_keys)
            _save_meta_encoding_types(ctx_encoding_types, path=ws.ctx_encoding_types)

    _LOG.info(f"SPLIT finished in {time.time() - t0:.2f}s")


def _save_meta_keys(
    path: PathDesc,
    primary_key: str | None = None,
    context_key: str | None = None,
):
    keys = {}
    if primary_key is not None:
        keys["primary_key"] = primary_key
    if context_key is not None:
        keys["context_key"] = context_key
    path.write(keys)


def _save_meta_encoding_types(encoding_types: dict[str, ModelEncodingType], path: PathDesc):
    for col, enc_type in encoding_types.items():
        try:
            ModelEncodingType(enc_type)
        except ValueError:
            raise ValueError(f"unknown encoding type: {enc_type} on column {col}")
    path.write(encoding_types)
