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
import random
import time
import uuid
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from mostlyai.engine._common import (
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
    CTXFLT,
    CTXSEQ,
    DEFAULT_HAS_RIDX,
    DEFAULT_HAS_SDEC,
    DEFAULT_HAS_SLEN,
    POSITIONAL_COLUMN,
    RIDX_SUB_COLUMN_PREFIX,
    SDEC_SUB_COLUMN_PREFIX,
    SIDX_SUB_COLUMN_PREFIX,
    SLEN_SUB_COLUMN_PREFIX,
    FixedSizeSampleBuffer,
    ProgressCallback,
    ProgressCallbackWrapper,
    apply_encoding_type_dtypes,
    decode_positional_column,
    encode_positional_column,
    get_argn_name,
    get_cardinalities,
    get_columns_from_cardinalities,
    get_ctx_sequence_length,
    get_sequence_length_stats,
    get_sub_columns_from_cardinalities,
    get_sub_columns_nested_from_cardinalities,
    persist_data_part,
)
from mostlyai.engine._encoding_types.tabular.categorical import (
    CATEGORICAL_NULL_TOKEN,
    CATEGORICAL_SUB_COL_SUFFIX,
    CATEGORICAL_UNKNOWN_TOKEN,
    decode_categorical,
)
from mostlyai.engine._encoding_types.tabular.character import decode_character
from mostlyai.engine._encoding_types.tabular.datetime import decode_datetime
from mostlyai.engine._encoding_types.tabular.itt import decode_itt
from mostlyai.engine._encoding_types.tabular.lat_long import decode_latlong
from mostlyai.engine._encoding_types.tabular.numeric import (
    NUMERIC_BINNED_NULL_TOKEN,
    NUMERIC_BINNED_SUB_COL_SUFFIX,
    NUMERIC_BINNED_UNKNOWN_TOKEN,
    NUMERIC_DISCRETE_NULL_TOKEN,
    NUMERIC_DISCRETE_SUB_COL_SUFFIX,
    NUMERIC_DISCRETE_UNKNOWN_TOKEN,
    decode_numeric,
)
from mostlyai.engine._memory import get_available_ram_for_heuristics, get_available_vram_for_heuristics
from mostlyai.engine._tabular.argn import (
    FlatModel,
    ModelSize,
    SequentialModel,
    get_no_of_model_parameters,
)
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._tabular.encoding import encode_df, pad_ctx_sequences
from mostlyai.engine._tabular.fairness import FairnessTransforms, get_fairness_transforms
from mostlyai.engine._workspace import Workspace, ensure_workspace_dir, reset_dir
from mostlyai.engine.domain import (
    FairnessConfig,
    ImputationConfig,
    ModelEncodingType,
    RareCategoryReplacementMethod,
    RebalancingConfig,
)

_LOG = logging.getLogger(__name__)

DUMMY_CONTEXT_KEY = "__dummy_context_key"


CodeProbabilities = dict[int, float]  # CategoryProbabilities after encoding, e.g. {0: 0.3, 1: 0.4}


def _resolve_gen_column_order(
    column_stats: dict,
    cardinalities: dict,
    rebalancing: RebalancingConfig | None = None,
    imputation: ImputationConfig | None = None,
    seed_data: pd.DataFrame | None = None,
    fairness: FairnessConfig | None = None,
):
    column_order = get_columns_from_cardinalities(cardinalities)

    # Reorder columns in the following order:
    # 0. Positional column
    # 1. Seed data columns
    # 2. Rebalancing column
    # 3. Fairness sensitive columns (which are not imputation columns)
    # 4. Fairness sensitive columns (which are imputation columns as well)
    # 5. The rest of the columns
    # 6. Imputation columns (which are not fairness sensitive columns)
    # 7. Fairness target column

    if imputation:
        # imputed columns should be at the end in the generation model
        imputation_argn = [
            get_argn_name(
                argn_processor=column_stats[col][ARGN_PROCESSOR],
                argn_table=column_stats[col][ARGN_TABLE],
                argn_column=column_stats[col][ARGN_COLUMN],
            )
            for col in imputation.columns
            if col in column_stats
        ]
        column_order = [c for c in column_order if c not in imputation_argn] + imputation_argn
    else:
        imputation_argn = []

    if fairness:
        # bring sensitive columns to the front and target column to the back
        sensitive_columns_argn = [
            get_argn_name(
                argn_processor=column_stats[col][ARGN_PROCESSOR],
                argn_table=column_stats[col][ARGN_TABLE],
                argn_column=column_stats[col][ARGN_COLUMN],
            )
            for col in fairness.sensitive_columns
            if col in column_stats
        ]
        # imputed sensitive columns should be after other usual sensitive columns
        sensitive_columns_argn = [c for c in sensitive_columns_argn if c not in imputation_argn] + [
            c for c in sensitive_columns_argn if c in imputation_argn
        ]

        target_column_argn = get_argn_name(
            argn_processor=column_stats[fairness.target_column][ARGN_PROCESSOR],
            argn_table=column_stats[fairness.target_column][ARGN_TABLE],
            argn_column=column_stats[fairness.target_column][ARGN_COLUMN],
        )
        column_order = (
            sensitive_columns_argn
            + [c for c in column_order if c not in sensitive_columns_argn + [target_column_argn]]
            + [target_column_argn]
        )

    if rebalancing:
        # rebalance column should be at the beginning in the generation model
        # rebalancing has higher priority than imputation
        if rebalancing.column in column_stats:
            rebalance_column_argn = get_argn_name(
                argn_processor=column_stats[rebalancing.column][ARGN_PROCESSOR],
                argn_table=column_stats[rebalancing.column][ARGN_TABLE],
                argn_column=column_stats[rebalancing.column][ARGN_COLUMN],
            )
            column_order = [rebalance_column_argn] + [c for c in column_order if c != rebalance_column_argn]

    if seed_data is not None:
        # seed_data columns should be at the beginning in the generation model
        # seed_data has higher priority than rebalancing and imputation
        seed_columns_argn = [
            get_argn_name(
                argn_processor=column_stats[col][ARGN_PROCESSOR],
                argn_table=column_stats[col][ARGN_TABLE],
                argn_column=column_stats[col][ARGN_COLUMN],
            )
            for col in seed_data.columns
            if col in column_stats
        ]
        column_order = seed_columns_argn + [c for c in column_order if c not in seed_columns_argn]

    if POSITIONAL_COLUMN in column_order:
        # positional column needs to be the first one in the generation model
        column_order = [POSITIONAL_COLUMN] + [c for c in column_order if c != POSITIONAL_COLUMN]

    return column_order


def _generate_primary_keys(size: int, type: Literal["uuid", "int"] = "uuid") -> pd.Series:
    if type == "uuid":
        # generate watermarked 36-chars UUIDv4s
        # e.g. mostly2b-d87c-4825-884f-611b309c3c55
        return pd.Series(
            [f"mostly{str(uuid.UUID(int=random.getrandbits(128), version=4))[6:]}" for _ in range(size)], dtype="string"
        )
    else:
        return pd.Series(range(size), dtype="int")


def _batch_df(df: pd.DataFrame, no_of_batches: int) -> pd.DataFrame:
    rows_per_batch = len(df) / no_of_batches
    running_total = pd.Series(range(len(df))) / rows_per_batch
    df = df.assign(__BATCH=running_total.astype(int) + 1)
    return df


def _regroup_partial_sequences_by_length(
    ctx_data: pd.DataFrame, seed_data: pd.DataFrame, ctx_primary_key: str, tgt_context_key: str
) -> tuple[pd.DataFrame, int]:
    # add temporary __PARTIAL_SEQ_LEN column to ctx_data
    partial_seq_lens = seed_data.groupby(tgt_context_key).size().rename("__PARTIAL_SEQ_LEN")
    ctx_data = ctx_data.assign(__PARTIAL_SEQ_LEN=ctx_data[ctx_primary_key].map(partial_seq_lens).fillna(0).astype(int))

    # regroup batches so that partial sequences of equal length are together
    new_batches = []
    for _, old_batch_df in ctx_data.groupby("__BATCH", sort=False):
        for _, new_batch_df in old_batch_df.groupby("__PARTIAL_SEQ_LEN", sort=False):
            new_batch_df = new_batch_df.assign(__BATCH=len(new_batches) + 1)
            new_batches.append(new_batch_df)

    # rebuild ctx_data; drop temporary __PARTIAL_SEQ_LEN column
    ctx_data = pd.concat(new_batches, axis=0).drop(columns=["__PARTIAL_SEQ_LEN"]).reset_index(drop=True)

    return ctx_data, len(new_batches)


def _flat_null_pattern(group: pd.DataFrame, relevant_columns: list[str]) -> tuple:
    """returns tuple of bools: True if column is fully NULL"""
    return tuple(group[col].isna().all() for col in relevant_columns)


def _trailing_null_pattern(group: pd.DataFrame, relevant_columns: list[str]) -> tuple:
    """returns tuple of bools: True if column has trailing NULLs"""
    pattern = []
    for col in relevant_columns:
        col_values = group[col].reset_index(drop=True)
        non_null_mask = col_values.notna()
        if non_null_mask.any():
            last_non_null_idx = non_null_mask[::-1].idxmax()
            has_trailing_nulls = last_non_null_idx < len(col_values) - 1
        else:
            has_trailing_nulls = True
        pattern.append(has_trailing_nulls)
    return tuple(pattern)


def _regroup_by_pattern(
    ctx_data: pd.DataFrame,
    seed_data: pd.DataFrame,
    ctx_primary_key: str,
    imputation_columns: list[str],
    pattern_fn: Callable[[pd.DataFrame, list[str]], tuple],
    *,
    groupby_key: str | None = None,
    use_vectorized_regroup: bool = True,
) -> tuple[pd.DataFrame, int]:
    """regroup batches so that rows/sequences with the same NULL pattern are together

    Args:
        pattern_fn: function that computes NULL pattern for a grouped dataframe
        groupby_key: key to group seed_data by (defaults to ctx_primary_key)
        use_vectorized_regroup: if True, use fast categorical factorization;
                                if False, use nested loop approach
    """
    # only consider columns that are BOTH in imputation_columns AND in seed_data
    relevant_columns = [col for col in imputation_columns if col in seed_data.columns]

    # early exit: no relevant columns
    if not relevant_columns or seed_data[relevant_columns].isna().all().all():
        return ctx_data, ctx_data["__BATCH"].nunique()

    # compute NULL pattern for each group
    groupby_key = groupby_key or ctx_primary_key
    seed_data_grouped = seed_data.groupby(groupby_key, sort=False)
    null_patterns = seed_data_grouped.apply(
        lambda group: pattern_fn(group, relevant_columns),
        include_groups=False,
    ).rename("__NULL_PATTERN")

    # early exit: all NULL patterns are the same
    if null_patterns.nunique() == 1:
        return ctx_data, ctx_data["__BATCH"].nunique()

    # add __NULL_PATTERN to ctx_data
    ctx_data = ctx_data.assign(__NULL_PATTERN=ctx_data[ctx_primary_key].map(null_patterns))

    # regroup batches
    if use_vectorized_regroup:
        # vectorized approach for flat data
        ctx_data = ctx_data.assign(
            __COMPOSITE_KEY=ctx_data["__BATCH"].astype(str) + "_" + ctx_data["__NULL_PATTERN"].astype(str)
        )
        composite_cat = pd.Categorical(ctx_data["__COMPOSITE_KEY"], categories=ctx_data["__COMPOSITE_KEY"].unique())
        ctx_data = ctx_data.assign(__BATCH=pd.factorize(composite_cat)[0] + 1)
        num_batches = ctx_data["__BATCH"].max()
        ctx_data = ctx_data.drop(columns=["__NULL_PATTERN", "__COMPOSITE_KEY"]).reset_index(drop=True)
    else:
        # nested loop approach for sequential data
        new_batches = []
        for _, old_batch_df in ctx_data.groupby("__BATCH", sort=False):
            for _, new_batch_df in old_batch_df.groupby("__NULL_PATTERN", sort=False):
                new_batch_df = new_batch_df.assign(__BATCH=len(new_batches) + 1)
                new_batches.append(new_batch_df)
        ctx_data = pd.concat(new_batches, axis=0).drop(columns=["__NULL_PATTERN"]).reset_index(drop=True)
        num_batches = len(new_batches)

    return ctx_data, num_batches


def _regroup_by_null_pattern(
    ctx_data: pd.DataFrame,
    seed_data: pd.DataFrame,
    ctx_primary_key: str,
    tgt_context_key: str,
    imputation_columns: list[str],
    is_sequential: bool,
) -> tuple[pd.DataFrame, int]:
    """regroup batches by NULL pattern (flat) or trailing NULL pattern (sequential)"""
    pattern_fn = _trailing_null_pattern if is_sequential else _flat_null_pattern
    groupby_key = tgt_context_key if is_sequential else None
    use_vectorized_regroup = not is_sequential

    return _regroup_by_pattern(
        ctx_data,
        seed_data,
        ctx_primary_key,
        imputation_columns,
        pattern_fn=pattern_fn,
        groupby_key=groupby_key,
        use_vectorized_regroup=use_vectorized_regroup,
    )


def _reshape_pt_to_pandas(
    data: list[torch.Tensor], sub_cols: list[str], keys: list[pd.Series], key_name: str
) -> pd.DataFrame:
    # len(data)=seq_len, data[0].shape=(1<=x<=batch_size, n_sub_cols, 1)
    # len(keys)=seq_len, keys[0].shape=(1<=x<=batch_size,)
    # len(sub_cols)=n_sub_cols
    assert len(data) == len(keys)
    for step_data, step_keys in zip(data, keys):
        assert step_data.shape[0] == step_keys.shape[0]
    seq_len = len(data)
    if seq_len == 0:
        return pd.DataFrame(columns=[key_name] + sub_cols)
    # transform from list[torch.Tensor] to pd.DataFrame, by concatenating sequence steps
    # df.shape=(sum(1<=x<=batch_size), n_sub_cols)
    df = pd.concat(
        [
            pd.DataFrame(
                step_tensor.squeeze(-1).detach().cpu().numpy(),
                columns=sub_cols,
                dtype="int32",
            )
            for step_tensor in data
        ],
        axis=0,
    ).reset_index(drop=True)
    # transform keys from list[pd.Series] to pd.Series, by concatenating sequence steps
    # keys.shape=(sum(1<=x<=batch_size),)
    keys = pd.concat(keys, axis=0).rename(key_name).reset_index(drop=True)
    return pd.concat([keys, df], axis=1)


def _drop_fully_null_imputed_columns(seed_batch: pd.DataFrame, imputation_columns: list[str]) -> pd.DataFrame:
    """drop columns from seed_batch that are in imputation_columns and are fully NULL

    this allows the model to freely generate these columns rather than conditioning on NULL values.
    """
    if not imputation_columns or seed_batch.empty:
        return seed_batch

    fully_null_cols = [col for col in imputation_columns if col in seed_batch.columns and seed_batch[col].isna().all()]
    return seed_batch.drop(columns=fully_null_cols) if fully_null_cols else seed_batch


def _post_process_decoding(
    syn: pd.DataFrame,
    tgt_primary_key: str | None = None,
) -> pd.DataFrame:
    # sort by dummy context key to restore original order (if exists)
    if DUMMY_CONTEXT_KEY in syn:
        syn = syn.sort_values(DUMMY_CONTEXT_KEY).reset_index(drop=True)
        syn = syn.drop(columns=DUMMY_CONTEXT_KEY)

    # generate primary keys, if they are not present
    if tgt_primary_key and tgt_primary_key not in syn:
        syn[tgt_primary_key] = _generate_primary_keys(len(syn), type="uuid")

    # reset index to ensure sequential indices for consistent test assertions
    syn = syn.reset_index(drop=True)

    return syn


##################
### HEURISTICS ###
##################


def _generation_batch_size_heuristic(mem_available_gb: float, ctx_stats: dict, tgt_stats: dict, device: torch.device):
    tgt_cardinalities = get_cardinalities(tgt_stats)
    ctx_cardinalities = get_cardinalities(ctx_stats)
    ctxflt_cardinalities = {k: v for k, v in ctx_cardinalities.items() if k.startswith(CTXFLT)}
    ctxseq_cardinalities = {k: v for k, v in ctx_cardinalities.items() if k.startswith(CTXSEQ)}
    ctxseq_max_lengths = get_ctx_sequence_length(ctx_stats, key="max")
    ctxseq_table_sub_columns = get_sub_columns_nested_from_cardinalities(ctxseq_cardinalities, groupby="tables")

    one_hot_unit_bytes = 4
    mem_available_bytes = mem_available_gb * 1_000 * 1_000 * 1_000

    ctxflt_one_hot = sum(ctxflt_cardinalities.values())
    ctxflt_one_hot_bytes = ctxflt_one_hot * one_hot_unit_bytes

    ctxseq_one_hots = []
    for table, sub_columns in ctxseq_table_sub_columns.items():
        one_hot = sum([ctxseq_cardinalities[sub_column] for sub_column in sub_columns]) * ctxseq_max_lengths[table]
        ctxseq_one_hots.append(one_hot)
    ctxseq_one_hot = sum(ctxseq_one_hots)
    ctxseq_one_hot_bytes = ctxseq_one_hot * one_hot_unit_bytes

    tgt_one_hot = sum(tgt_cardinalities.values())
    tgt_one_hot_bytes = tgt_one_hot * one_hot_unit_bytes

    sample_bytes = ctxflt_one_hot_bytes + ctxseq_one_hot_bytes + tgt_one_hot_bytes
    sample_kb = sample_bytes / 1_000
    scaling_factor = 0.1

    batch_size = int((mem_available_bytes // max(1, sample_bytes)) * scaling_factor)
    device_max = 10_000 if device.type == "cuda" else 100_000
    batch_size = int(np.clip(batch_size, a_min=100, a_max=device_max))

    _LOG.info(
        f"batch_size heuristic: {batch_size:,} ({mem_available_gb=:.1f}GB, {sample_kb=:.1f}KB, {scaling_factor=:.1f})"
    )

    return batch_size


#########################
### PROGRAMMABLE DATA ###
#########################


def _fix_rare_token_probs(
    stats: dict,
    rare_category_replacement_method: RareCategoryReplacementMethod | None = None,
) -> dict[str, dict[str, CodeProbabilities]]:
    # suppress rare token for categorical when no_of_rare_categories == 0
    mask = {
        col: {CATEGORICAL_SUB_COL_SUFFIX: {col_stats["codes"][CATEGORICAL_UNKNOWN_TOKEN]: 0.0}}
        for col, col_stats in stats["columns"].items()
        if col_stats["encoding_type"] == ModelEncodingType.tabular_categorical
        if "codes" in col_stats
        if col_stats.get("no_of_rare_categories", 0) == 0
    }
    # suppress rare token for categorical if RareCategoryReplacementMethod is sample
    if rare_category_replacement_method == RareCategoryReplacementMethod.sample:
        mask |= {
            col: {CATEGORICAL_SUB_COL_SUFFIX: {col_stats["codes"][CATEGORICAL_UNKNOWN_TOKEN]: 0.0}}
            for col, col_stats in stats["columns"].items()
            if col_stats["encoding_type"] == ModelEncodingType.tabular_categorical
            if "codes" in col_stats
        }
    # always suppress rare token for numeric_binned
    mask |= {
        col: {NUMERIC_BINNED_SUB_COL_SUFFIX: {col_stats["codes"][NUMERIC_BINNED_UNKNOWN_TOKEN]: 0.0}}
        for col, col_stats in stats["columns"].items()
        if col_stats["encoding_type"] == ModelEncodingType.tabular_numeric_binned
        if "codes" in col_stats
    }
    # always suppress rare token for numeric_discrete
    mask |= {
        col: {NUMERIC_DISCRETE_SUB_COL_SUFFIX: {col_stats["codes"][NUMERIC_DISCRETE_UNKNOWN_TOKEN]: 0.0}}
        for col, col_stats in stats["columns"].items()
        if col_stats["encoding_type"] == ModelEncodingType.tabular_numeric_discrete
        if "codes" in col_stats
    }
    return mask


def _fix_imputation_probs(
    stats: dict,
    imputation: ImputationConfig | None = None,
) -> dict[str, dict[str, CodeProbabilities]]:
    imputation = imputation.columns if imputation is not None else []
    _LOG.info(f"imputation: {imputation}")
    fixed_probs: dict[str, dict[str, CodeProbabilities]] = {}
    for col in imputation:
        if col not in stats["columns"]:
            _LOG.info(f"imputed [{col}] not found in stats")
            continue
        col_stats = stats["columns"][col]
        encoding_type = col_stats["encoding_type"]
        # null_name will be either None, "na" or "nan"
        null_subcol = next(iter([k[4:] for k in col_stats.keys() if k in ["has_na", "has_nan"]]), None)
        if null_subcol is not None and col_stats[f"has_{null_subcol}"]:
            # column has separate null sub column and there are some nulls
            code_null = 1
            col_fixed_probs = {col: {null_subcol: {code_null: 0.0}}}
        elif encoding_type in [
            ModelEncodingType.tabular_categorical,
            ModelEncodingType.tabular_numeric_discrete,
            ModelEncodingType.tabular_numeric_binned,
        ]:
            # column is categorical-like and has single sub column
            sub_column = {
                ModelEncodingType.tabular_categorical: CATEGORICAL_SUB_COL_SUFFIX,
                ModelEncodingType.tabular_numeric_discrete: NUMERIC_DISCRETE_SUB_COL_SUFFIX,
                ModelEncodingType.tabular_numeric_binned: NUMERIC_BINNED_SUB_COL_SUFFIX,
            }[encoding_type]
            code_probs = {
                ModelEncodingType.tabular_categorical: {
                    CATEGORICAL_NULL_TOKEN: 0.0,
                    CATEGORICAL_UNKNOWN_TOKEN: 0.0,
                },
                ModelEncodingType.tabular_numeric_discrete: {NUMERIC_DISCRETE_NULL_TOKEN: 0.0},
                ModelEncodingType.tabular_numeric_binned: {NUMERIC_BINNED_NULL_TOKEN: 0.0},
            }[encoding_type]
            # map and filter out codes that did not occur (happens when there are no nulls)
            col_fixed_probs = {
                col: {
                    sub_column: {
                        col_stats["codes"][category]: probs
                        for category, probs in code_probs.items()
                        if category in col_stats["codes"]
                    }
                }
            }
        else:
            col_fixed_probs = {}
        fixed_probs |= col_fixed_probs
    return fixed_probs


def _fix_rebalancing_probs(
    stats: dict,
    rebalancing: RebalancingConfig | None = None,
) -> dict[str, dict[str, CodeProbabilities]]:
    column, probabilities = (rebalancing.column, rebalancing.probabilities) if rebalancing else (None, {})
    _LOG.info(f"rebalance_column: {column}")
    _LOG.info(f"rebalance_probabilities: {probabilities}")
    mask = {}
    if (
        column is not None
        and column in stats["columns"]
        and stats["columns"][column]["encoding_type"] == ModelEncodingType.tabular_categorical
        and "codes" in stats["columns"][column]
    ):
        col_codes = stats["columns"][column]["codes"]
        code_probabilities = {
            col_codes[category]: max(0.0, prob) for category, prob in probabilities.items() if category in col_codes
        }
        # normalize probabilities if they sum up to more than 1.0
        total_share = sum(code_probabilities.values())
        if total_share > 1.0:
            code_probabilities = {code: share / total_share for code, share in code_probabilities.items()}
        if code_probabilities:
            mask = {column: {CATEGORICAL_SUB_COL_SUFFIX: code_probabilities}}

    return mask


def _translate_fixed_probs(
    fixed_probs: dict[str, dict[str, CodeProbabilities]], stats: dict
) -> dict[str, CodeProbabilities]:
    # translate fixed probs to ARGN conventions
    mask = {
        get_argn_name(
            argn_processor=stats["columns"][col][ARGN_PROCESSOR],
            argn_table=stats["columns"][col][ARGN_TABLE],
            argn_column=stats["columns"][col][ARGN_COLUMN],
            argn_sub_column=sub_col,
        ): sub_col_mask
        for col, col_mask in fixed_probs.items()
        for sub_col, sub_col_mask in col_mask.items()
    }
    return mask


def _deepmerge(*dictionaries: dict, merged: dict | None = None) -> dict:
    merged = merged or {}
    for dictionary in dictionaries:
        for key, value in dictionary.items():
            if isinstance(value, dict):
                if key not in merged:
                    merged[key] = {}
                merged[key] |= _deepmerge(value, merged[key])
            else:
                merged[key] = value
    return merged


##################
###   DECODE   ###
##################


def _decode_df(
    df_encoded: pd.DataFrame,
    stats: dict,
    context_key: str | None = None,
    prev_steps: dict | None = None,
) -> pd.DataFrame:
    columns = []
    if context_key and context_key in df_encoded.columns:
        columns.append(df_encoded[context_key])
    for column, column_stats in stats["columns"].items():
        if column_stats.keys() == {"encoding_type"}:
            # training data was empty
            values = pd.Series(data=[], name=column, dtype="object")
            columns.append(values)
            continue
        sub_columns = [
            get_argn_name(
                argn_processor=column_stats[ARGN_PROCESSOR],
                argn_table=column_stats[ARGN_TABLE],
                argn_column=column_stats[ARGN_COLUMN],
                argn_sub_column=sub_col,
            )
            for sub_col in column_stats["cardinalities"].keys()
        ]
        # fetch column-specific sub_columns from data
        df_encoded_col = df_encoded[sub_columns]
        # remove column prefixes before decoding
        df_encoded_col.columns = [
            ocol.replace(
                # replace conventional column name without sub_column part
                get_argn_name(
                    argn_processor=column_stats[ARGN_PROCESSOR],
                    argn_table=column_stats[ARGN_TABLE],
                    argn_column=column_stats[ARGN_COLUMN],
                    argn_sub_column="",
                ),
                "",
            )
            for ocol in df_encoded_col.columns
        ]
        # handle column prev_steps
        prev_steps_col = None
        if prev_steps is not None:
            prev_steps[column] = prev_steps.get(column, {})
            prev_steps_col = prev_steps[column]
        # decode encoded sub_columns into single decoded column
        values = _decode_col(
            df_encoded=df_encoded_col,
            stats=column_stats,
            context_keys=df_encoded[context_key] if context_key in df_encoded.columns else None,
            prev_steps=prev_steps_col,
        )
        values.name = column
        columns.append(values)
    return pd.concat(columns, axis=1)


def _decode_col(
    df_encoded: pd.DataFrame,
    stats: dict,
    context_keys: pd.Series | None = None,
    prev_steps: dict | None = None,
) -> pd.Series:
    if df_encoded.empty:
        return pd.Series()

    encoding_type = stats["encoding_type"]

    if encoding_type == ModelEncodingType.tabular_categorical:
        values = decode_categorical(df_encoded=df_encoded, stats=stats)
    elif encoding_type in [
        ModelEncodingType.tabular_numeric_auto,
        ModelEncodingType.tabular_numeric_discrete,
        ModelEncodingType.tabular_numeric_binned,
        ModelEncodingType.tabular_numeric_digit,
    ]:
        values = decode_numeric(df_encoded=df_encoded, stats=stats)
    elif encoding_type == ModelEncodingType.tabular_datetime:
        values = decode_datetime(df_encoded=df_encoded, stats=stats)
    elif encoding_type == ModelEncodingType.tabular_datetime_relative:
        values = decode_itt(
            df_encoded=df_encoded,
            stats=stats,
            context_keys=context_keys,
            prev_steps=prev_steps,
        )
    elif encoding_type == ModelEncodingType.tabular_character:
        values = decode_character(df_encoded=df_encoded, stats=stats)
    elif encoding_type == ModelEncodingType.tabular_lat_long:
        values = decode_latlong(df_encoded=df_encoded, stats=stats)
    return values


def decode_buffered_samples(
    buffer: FixedSizeSampleBuffer,
    tgt_stats: dict,
    tgt_sub_columns: list[str],
    tgt_primary_key: str,
    tgt_context_key: str,
    decode_prev_steps: dict | None = None,
    impute_columns: list[str] | None = None,
) -> pd.DataFrame:
    is_sequential = tgt_stats["is_sequential"]
    seq_len_stats = get_sequence_length_stats(tgt_stats)
    seq_len_max = seq_len_stats["max"]

    assert not buffer.is_empty() or seq_len_max == 0

    if is_sequential:
        data, keys, seed_data = zip(*buffer.buffer) if buffer.buffer else ([], [], [])
        df_syn = _reshape_pt_to_pandas(
            data=data,
            sub_cols=tgt_sub_columns,
            keys=keys,
            key_name=tgt_context_key,
        )
        df_syn = df_syn.drop(
            columns=[c for c in df_syn.columns if c.startswith(POSITIONAL_COLUMN)],
            axis=1,
        ).reset_index(drop=True)
    else:
        data, seed_data = zip(*buffer.buffer)
        df_syn = pd.concat(data, axis=0).reset_index(drop=True)

    # decode generated data
    _LOG.info(f"decode generated data {df_syn.shape}")
    df_syn = _decode_df(
        df_encoded=df_syn,
        stats=tgt_stats,
        context_key=tgt_context_key,
        prev_steps=decode_prev_steps,
    )

    # preserve all seed values
    df_seed = pd.concat(seed_data, axis=0).reset_index(drop=True) if seed_data else pd.DataFrame()

    if not df_seed.empty:
        seed_columns = [col for col in df_seed.columns]
        if is_sequential:
            # overwrite first steps of each sequence in synthetic data with values from seed data
            impute_columns = impute_columns or []
            df_syn["__SEQ_IDX"] = df_syn.groupby(tgt_context_key).cumcount()
            df_seed["__SEQ_IDX"] = df_seed.groupby(tgt_context_key).cumcount()
            # df_overwrite is a dataframe with the same shape as df_syn, but with the seed values for the first steps of each sequence
            df_overwrite = pd.merge(
                df_syn[[tgt_context_key, "__SEQ_IDX"]].copy(),
                df_seed,
                on=[tgt_context_key, "__SEQ_IDX"],
                how="left",
                indicator="__INDICATOR",
            )
            # project df_overwrite onto df_syn
            seed_rows = df_overwrite["__INDICATOR"] == "both"
            # overwrite columns based on imputation logic
            for col in seed_columns:
                if col in [tgt_context_key, "__SEQ_IDX"]:
                    continue  # skip the key columns
                if col not in impute_columns:
                    # non-impute columns: override all values
                    df_syn.loc[seed_rows, col] = df_overwrite.loc[seed_rows, col]
                else:
                    # impute columns: override only non-NULL seed values
                    mask = seed_rows & df_overwrite[col].notna()
                    df_syn.loc[mask, col] = df_overwrite.loc[mask, col]
            df_syn.drop(columns=["__SEQ_IDX"], inplace=True)
            df_seed.drop(columns=["__SEQ_IDX"], inplace=True)
        else:
            # for flat data, overwrite seed columns using merge to handle reordered rows
            # for non-impute columns: override all values
            # for impute columns: override only non-NULL seed values (let model impute NULL values)
            impute_columns = impute_columns or []

            # use merge on context key to properly align seed values with synthetic data
            df_overwrite = pd.merge(
                df_syn[[tgt_context_key]].copy(),
                df_seed,
                on=tgt_context_key,
                how="left",
                suffixes=("", "_seed"),
            )

            # overwrite columns based on imputation logic
            for col in seed_columns:
                if col == tgt_context_key:
                    continue  # skip the key column itself
                seed_col_name = col if col in df_overwrite.columns else f"{col}_seed"
                if seed_col_name in df_overwrite.columns:
                    if col not in impute_columns:
                        # non-impute columns: override all values
                        df_syn[col] = df_overwrite[seed_col_name]
                    else:
                        # impute columns: override only non-NULL seed values
                        mask = df_overwrite[seed_col_name].notna()
                        df_syn.loc[mask, col] = df_overwrite.loc[mask, seed_col_name]

    # postprocess generated data
    _LOG.info(f"post-process generated data {df_syn.shape}")
    df_syn = _post_process_decoding(
        df_syn,
        tgt_primary_key=tgt_primary_key,
    )
    return df_syn


##################
### GENERATION ###
##################


def _load_trained_model(
    *,
    workspace_dir: str | Path,
    device: torch.device | str | None = None,
    column_order: list[str] | None = None,
    allow_sequential: bool = True,
    require_weights: bool = True,
) -> tuple[FlatModel | SequentialModel, dict, dict, dict, torch.device]:
    """
    Load a trained model from workspace with all necessary stats.

    This helper function consolidates common model loading logic used by both
    generate() and log_prob() functions.

    Args:
        workspace_dir: Directory path for workspace containing the trained model.
        device: Device to run on ('cuda' or 'cpu'). Defaults to 'cuda' if available, else 'cpu'.
        column_order: Column order for model. If None, uses default training order.
        allow_sequential: Whether to allow sequential models. If False, raises error for sequential models.
        require_weights: If True, raises error when weights missing. If False, logs warning and continues.

    Returns:
        Tuple of (model, tgt_stats, ctx_stats, model_configs, device)

    Raises:
        ValueError: If allow_sequential=False and model is sequential.
        ValueError: If require_weights=True and model weights not found.
    """
    # Build paths based on workspace dir
    workspace_dir = ensure_workspace_dir(workspace_dir)
    workspace = Workspace(workspace_dir)

    # Read stats and model config
    tgt_stats = workspace.tgt_stats.read()
    ctx_stats = workspace.ctx_stats.read()
    model_configs = workspace.model_configs.read()

    is_sequential = tgt_stats["is_sequential"]

    # Validate model type if sequential not allowed
    if not allow_sequential and is_sequential:
        raise ValueError(
            "This operation is only supported for flat (non-sequential) models. "
            "Sequential models were trained with tgt_context_key or tgt_primary_key."
        )

    # Get model configuration
    model_units = model_configs.get("model_units") or ModelSize.M

    # Handle different approaches to sequence modeling (backwards compatibility)
    has_slen = has_ridx = has_sdec = None
    if is_sequential:
        if isinstance(model_units, dict):
            has_slen = any(SLEN_SUB_COLUMN_PREFIX in k for k in model_units.keys())
            has_ridx = any(RIDX_SUB_COLUMN_PREFIX in k for k in model_units.keys())
            has_sdec = any(SDEC_SUB_COLUMN_PREFIX in k for k in model_units.keys())
        else:
            has_slen, has_ridx, has_sdec = DEFAULT_HAS_SLEN, DEFAULT_HAS_RIDX, DEFAULT_HAS_SDEC

    # Get cardinalities
    tgt_cardinalities = get_cardinalities(tgt_stats, has_slen, has_ridx, has_sdec)
    ctx_cardinalities = get_cardinalities(ctx_stats)
    ctx_seq_len_median = get_ctx_sequence_length(ctx_stats, key="median")

    # Resolve device
    device = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    _LOG.info(f"{device=}")

    # Create model
    if is_sequential:
        seq_len_stats = get_sequence_length_stats(tgt_stats)
        seq_len_median = seq_len_stats["median"]
        seq_len_max = seq_len_stats["max"]

        model = SequentialModel(
            tgt_cardinalities=tgt_cardinalities,
            tgt_seq_len_median=seq_len_median,
            tgt_seq_len_max=seq_len_max,
            ctx_cardinalities=ctx_cardinalities,
            ctxseq_len_median=ctx_seq_len_median,
            model_size=model_units,
            column_order=column_order,
            device=device,
        )
    else:
        model = FlatModel(
            tgt_cardinalities=tgt_cardinalities,
            ctx_cardinalities=ctx_cardinalities,
            ctxseq_len_median=ctx_seq_len_median,
            model_size=model_units,
            column_order=column_order,
            device=device,
        )

    no_of_model_params = get_no_of_model_parameters(model)
    _LOG.info(f"{no_of_model_params=}")

    # Load model weights
    if workspace.model_tabular_weights_path.exists():
        load_model_weights(
            model=model,
            path=workspace.model_tabular_weights_path,
            device=device,
        )
    else:
        if require_weights:
            raise ValueError("Model weights not found. Please train the model first.")
        else:
            _LOG.warning("model weights not found; generating data with an untrained model")

    model.to(device)
    model.eval()

    return model, tgt_stats, ctx_stats, model_configs, device


def _prepare_context_data(
    *,
    workspace: Workspace,
    ctx_stats: dict,
    tgt_stats: dict,
    ctx_data: pd.DataFrame | None = None,
    sample_size: int | None = None,
    seed_data: pd.DataFrame | None = None,
    validate_size: bool = False,
    data_size: int | None = None,
) -> tuple[pd.DataFrame, str, str, bool]:
    """
    Prepare context data for generation or log_prob evaluation.

    Handles workspace setup, context existence check, context data loading/creation,
    and dummy context creation when needed.

    Args:
        workspace: Workspace instance.
        ctx_stats: Context statistics dictionary.
        tgt_stats: Target statistics dictionary.
        ctx_data: Optional context data DataFrame.
        sample_size: Optional sample size for generation.
        seed_data: Optional seed data DataFrame.
        validate_size: If True, validate that context has enough rows for data_size.
        data_size: Size of data to validate against (used when validate_size=True).

    Returns:
        Tuple of (ctx_data, ctx_primary_key, tgt_context_key, has_context)
    """
    has_context = workspace.ctx_stats.path.exists()

    if has_context:
        if ctx_data is None:
            # Re-use context from training if no new context provided
            ctx_data = pd.read_parquet(workspace.ctx_data_path)
        _LOG.info(f"Using context data {ctx_data.shape}")

        ctx_data = ctx_data.reset_index(drop=True)

        if validate_size and data_size is not None:
            # Validate we have enough context rows (for log_prob)
            if len(ctx_data) < data_size:
                raise ValueError(
                    f"Context data has {len(ctx_data)} rows but data has {data_size} rows. "
                    f"Context data must have at least as many rows as data."
                )
            # Take first data_size rows of context
            ctx_data = ctx_data.head(data_size)
        elif sample_size is not None:
            # For generation, take first sample_size rows
            sample_size = min(sample_size, len(ctx_data))
            ctx_data = ctx_data.head(sample_size)
        # else: sample_size is None, will use full context (determined by caller)

        # Validate context data columns
        ctx_column_stats = list(ctx_stats["columns"].keys())
        missing_columns = [c for c in ctx_column_stats if c not in ctx_data.columns]
        if len(missing_columns) > 0:
            raise ValueError(f"missing columns in provided context data: {', '.join(missing_columns[:5])}")

        ctx_primary_key = ctx_stats.get("keys", {}).get("primary_key")
        tgt_context_key = tgt_stats.get("keys", {}).get("context_key")
    else:
        # Create on-the-fly context
        tgt_context_key = tgt_stats.get("keys", {}).get("context_key")
        ctx_primary_key = tgt_context_key or DUMMY_CONTEXT_KEY
        tgt_context_key = ctx_primary_key

        if sample_size is None:
            if seed_data is None:
                trn_sample_size = tgt_stats["no_of_training_records"] + tgt_stats["no_of_validation_records"]
                sample_size = trn_sample_size
            else:
                sample_size = len(seed_data)
        elif validate_size and data_size is not None:
            sample_size = data_size

        ctx_primary_keys = _generate_primary_keys(sample_size, type="int")
        ctx_primary_keys.rename(ctx_primary_key, inplace=True)
        ctx_data = ctx_primary_keys.to_frame()

    return ctx_data, ctx_primary_key, tgt_context_key, has_context


def _prepare_encoding_types(
    ctx_stats: dict,
    tgt_stats: dict,
    has_context: bool,
) -> tuple[dict, dict]:
    """
    Resolve encoding types for context and target data.

    Args:
        ctx_stats: Context statistics dictionary.
        tgt_stats: Target statistics dictionary.
        has_context: Whether context exists.

    Returns:
        Tuple of (ctx_encoding_types, tgt_encoding_types)
    """
    ctx_encoding_types = (
        {c_name: c_data["encoding_type"] for c_name, c_data in ctx_stats["columns"].items()} if has_context else {}
    )
    tgt_encoding_types = {c_name: c_data["encoding_type"] for c_name, c_data in tgt_stats["columns"].items()}

    return ctx_encoding_types, tgt_encoding_types


def _encode_context_and_target(
    *,
    ctx_data: pd.DataFrame,
    tgt_data: pd.DataFrame,
    ctx_stats: dict,
    tgt_stats: dict,
    ctx_primary_key: str,
    tgt_context_key: str | None = None,
) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """
    Encode context and target data.

    Args:
        ctx_data: Context data DataFrame.
        tgt_data: Target data DataFrame.
        ctx_stats: Context statistics dictionary.
        tgt_stats: Target statistics dictionary.
        ctx_primary_key: Context primary key column name.
        tgt_context_key: Optional target context key column name.

    Returns:
        Tuple of (ctx_data_encoded, ctx_primary_key_encoded, tgt_data_encoded)
    """
    # Encode context data
    _LOG.info(f"Encoding context data {ctx_data.shape}")
    ctx_data_encoded, ctx_primary_key_encoded, _ = encode_df(
        df=ctx_data, stats=ctx_stats, ctx_primary_key=ctx_primary_key
    )
    # Pad left context sequences to ensure non-empty sequences
    ctx_data_encoded = pad_ctx_sequences(ctx_data_encoded)

    # Encode target data
    _LOG.info(f"Encoding target data {tgt_data.shape}")
    tgt_data_encoded, _, _ = encode_df(df=tgt_data, stats=tgt_stats, tgt_context_key=tgt_context_key)

    return ctx_data_encoded, ctx_primary_key_encoded, tgt_data_encoded


def _prepare_input_tensors(
    *,
    ctx_data_encoded: pd.DataFrame,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Prepare input tensors from encoded context data.

    Converts encoded context data to PyTorch tensors for model input.

    Args:
        ctx_data_encoded: Encoded context data DataFrame.
        device: Device to place tensors on.

    Returns:
        Dictionary of input tensors (ctxflt_inputs | ctxseq_inputs)
    """
    ctxflt_inputs = {
        col: torch.unsqueeze(
            torch.as_tensor(ctx_data_encoded[col].to_numpy(), device=device).type(torch.int),
            dim=-1,
        )
        for col in ctx_data_encoded.columns
        if col.startswith(CTXFLT)
    }
    ctxseq_inputs = {
        col: torch.unsqueeze(
            torch.nested.as_nested_tensor(
                [torch.as_tensor(t, device=device).type(torch.int) for t in ctx_data_encoded[col]],
                device=device,
            ),
            dim=-1,
        )
        for col in ctx_data_encoded.columns
        if col.startswith(CTXSEQ)
    }
    return ctxflt_inputs | ctxseq_inputs


@torch.no_grad()
def generate(
    *,
    ctx_data: pd.DataFrame | None = None,
    seed_data: pd.DataFrame | None = None,
    sample_size: int | None = None,
    batch_size: int | None = None,
    rare_category_replacement_method: RareCategoryReplacementMethod | str = RareCategoryReplacementMethod.constant,
    sampling_temperature: float = 1.0,
    sampling_top_p: float = 1.0,
    rebalancing: RebalancingConfig | dict | None = None,
    imputation: ImputationConfig | dict | None = None,
    fairness: FairnessConfig | dict | None = None,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
) -> None:
    _LOG.info("GENERATE_TABULAR started")
    t0 = time.time()
    with ProgressCallbackWrapper(update_progress) as progress:
        # build paths based on workspace dir
        workspace_dir = ensure_workspace_dir(workspace_dir)
        workspace = Workspace(workspace_dir)
        output_path = workspace.generated_data_path
        reset_dir(output_path)

        # Load stats first (needed for gen_column_order computation)
        model_configs = workspace.model_configs.read()
        tgt_stats = workspace.tgt_stats.read()
        ctx_stats = workspace.ctx_stats.read()
        is_sequential = tgt_stats["is_sequential"]
        _LOG.info(f"{is_sequential=}")
        has_context = workspace.ctx_stats.path.exists()
        _LOG.info(f"{has_context=}")

        # read model config
        model_units = model_configs.get("model_units") or ModelSize.M
        _LOG.debug(f"{model_units=}")
        enable_flexible_generation = model_configs.get("enable_flexible_generation", True)
        _LOG.info(f"{enable_flexible_generation=}")

        # handle different approaches to sequence modeling (backwards compatibility)
        has_slen = has_ridx = has_sdec = None
        if is_sequential:
            if isinstance(model_units, dict):
                has_slen = any(SLEN_SUB_COLUMN_PREFIX in k for k in model_units.keys())
                has_ridx = any(RIDX_SUB_COLUMN_PREFIX in k for k in model_units.keys())
                has_sdec = any(SDEC_SUB_COLUMN_PREFIX in k for k in model_units.keys())
            else:
                has_slen, has_ridx, has_sdec = DEFAULT_HAS_SLEN, DEFAULT_HAS_RIDX, DEFAULT_HAS_SDEC

        tgt_cardinalities = get_cardinalities(tgt_stats, has_slen, has_ridx, has_sdec)
        ctx_cardinalities = get_cardinalities(ctx_stats)
        tgt_sub_columns = get_sub_columns_from_cardinalities(tgt_cardinalities)
        ctx_sub_columns = get_sub_columns_from_cardinalities(ctx_cardinalities)
        _LOG.info(f"{len(tgt_sub_columns)=}")
        _LOG.info(f"{len(ctx_sub_columns)=}")

        # resolve device
        device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
        _LOG.info(f"{device=}")

        tgt_primary_key = tgt_stats.get("keys", {}).get("primary_key")
        tgt_context_key = tgt_stats.get("keys", {}).get("context_key")
        ctx_primary_key = ctx_stats.get("keys", {}).get("primary_key")
        _LOG.info(f"{tgt_primary_key=}, {tgt_context_key=}, {ctx_primary_key=}")

        if rebalancing and isinstance(rebalancing, dict):
            rebalancing = RebalancingConfig(**rebalancing)
        if imputation and isinstance(imputation, dict):
            imputation = ImputationConfig(**imputation)
        if fairness and isinstance(fairness, dict):
            fairness = FairnessConfig(**fairness)
        _LOG.info(f"imputation: {imputation}")
        _LOG.info(f"rebalancing: {rebalancing}")
        _LOG.info(f"fairness: {fairness}")
        _LOG.info(f"seed_data: {list(seed_data.columns) if isinstance(seed_data, pd.DataFrame) else None}")
        gen_column_order = _resolve_gen_column_order(
            column_stats=tgt_stats["columns"],
            cardinalities=tgt_cardinalities,
            rebalancing=rebalancing,
            imputation=imputation,
            seed_data=seed_data,
            fairness=fairness,
        )
        _LOG.info(f"{gen_column_order=}")
        trn_column_order = get_columns_from_cardinalities(tgt_cardinalities)
        _LOG.info(f"{trn_column_order=}")

        if not enable_flexible_generation:
            # check if resolved column order is the same as the one from training
            if gen_column_order != trn_column_order:
                raise ValueError(
                    "The column order for generation does not match the column order from training, due to seed, rebalancing, fairness or imputation configs. "
                    "A change in column order is only permitted for models that were trained with `enable_flexible_generation=True`."
                )

        _LOG.info(f"{rare_category_replacement_method=}")
        rare_token_fixed_probs = _fix_rare_token_probs(tgt_stats, rare_category_replacement_method)
        imputation_fixed_probs = _fix_imputation_probs(tgt_stats, imputation)
        rebalancing_fixed_probs = _fix_rebalancing_probs(tgt_stats, rebalancing)
        fixed_probs = _translate_fixed_probs(
            fixed_probs=_deepmerge(
                rare_token_fixed_probs,
                imputation_fixed_probs,
                rebalancing_fixed_probs,
            ),
            stats=tgt_stats,
        )
        _LOG.info(f"{sampling_temperature=}, {sampling_top_p=}")

        # Prepare context data using helper
        ctx_data, ctx_primary_key, tgt_context_key, has_context = _prepare_context_data(
            workspace=workspace,
            ctx_stats=ctx_stats,
            tgt_stats=tgt_stats,
            ctx_data=ctx_data,
            sample_size=sample_size,
            seed_data=seed_data,
        )
        # Update sample_size from context data length
        if sample_size is None:
            sample_size = len(ctx_data)
        else:
            sample_size = min(sample_size, len(ctx_data))

        if seed_data is None:
            # create on-the-fly seed data
            seed_data = pd.DataFrame(columns=[tgt_context_key])

        if not is_sequential:
            # link seed data to dummy context for flat data generation
            seed_data = seed_data.assign(**{tgt_context_key: ctx_data[ctx_primary_key].values})

        # sequence lengths
        seq_len_stats = get_sequence_length_stats(tgt_stats)
        seq_len_min = seq_len_stats["min"]
        seq_len_max = seq_len_stats["max"]

        # validate sequential seed_data has tgt_context_key
        if is_sequential and tgt_context_key not in seed_data.columns:
            raise ValueError(
                f"Seed data must contain tgt_context_key column `{tgt_context_key}` for sequential generation"
            )

        # trim sequences in seed_data to seq_len_max for sequential generation
        seed_data_grouped = seed_data.groupby(tgt_context_key, group_keys=False)
        if is_sequential and (seed_seq_len_max := seed_data_grouped.size().max()) > seq_len_max:
            _LOG.warning(f"truncating seed sequences: max allowed = `{seq_len_max}`, found = `{seed_seq_len_max}`")
            seed_data = seed_data_grouped.apply(lambda x: x.iloc[:seq_len_max]).reset_index(drop=True)

        # ensure valid columns in seed_data
        tgt_columns = (
            list(tgt_stats["columns"].keys()) + [tgt_context_key] + ([tgt_primary_key] if tgt_primary_key else [])
        )
        seed_data = seed_data[[c for c in tgt_columns if c in seed_data.columns]]

        # determine batch_size for generation
        if batch_size is None:
            cpu_mem_available_gb = get_available_ram_for_heuristics() / 1024**3
            gpu_mem_available_gb = get_available_vram_for_heuristics() / 1024**3
            batch_size = _generation_batch_size_heuristic(
                mem_available_gb=cpu_mem_available_gb if device.type == "cpu" else gpu_mem_available_gb,
                ctx_stats=ctx_stats,
                tgt_stats=tgt_stats,
                device=device,
            )
        if batch_size < sample_size:
            no_of_batches = int(np.ceil(sample_size / batch_size))
        else:
            no_of_batches = 1
            batch_size = min(batch_size, sample_size)
        _LOG.info(f"{sample_size=}")
        _LOG.info(f"{list(seed_data.columns)=}")
        _LOG.info(f"{batch_size=}")
        _LOG.info(f"{no_of_batches=}")

        # init progress with total_count; +1 for the final decoding step
        progress.update(completed=0, total=no_of_batches * (seq_len_max + 1))

        # Load model using helper (with computed gen_column_order)
        _LOG.info("create generative model")
        model, tgt_stats, ctx_stats, model_configs, device = _load_trained_model(
            workspace_dir=workspace_dir,
            device=device,
            column_order=gen_column_order,
            allow_sequential=True,
            require_weights=False,  # Allow untrained model for generation
        )

        # calculate fairness transforms only once before batch generation
        fairness_transforms: FairnessTransforms | None = None
        if fairness and isinstance(model, FlatModel):
            fairness_transforms: FairnessTransforms = get_fairness_transforms(
                fairness=fairness,
                tgt_stats=tgt_stats,
                forward_fn=partial(
                    model.forward,
                    fixed_probs=fixed_probs,
                    temperature=sampling_temperature,
                    top_p=sampling_top_p,
                ),
                device=device,
            )

        # resolve encoding types for dtypes harmonisation
        ctx_encoding_types, tgt_encoding_types = _prepare_encoding_types(
            ctx_stats=ctx_stats,
            tgt_stats=tgt_stats,
            has_context=has_context,
        )
        seed_encoding_types = {
            c_name: tgt_encoding_types[c_name] for c_name in tgt_encoding_types if c_name in seed_data.columns
        }

        # add __BATCH to ctx_data
        ctx_data = _batch_df(ctx_data, no_of_batches)

        # update __BATCH to ensure that, partial sequences of the same length are grouped within the same batch
        if is_sequential:
            ctx_data, no_of_batches = _regroup_partial_sequences_by_length(
                ctx_data, seed_data, ctx_primary_key, tgt_context_key
            )

        # regroup by NULL pattern if imputation is enabled
        if imputation and seed_data is not None and len(seed_data) > 0:
            ctx_data, no_of_batches = _regroup_by_null_pattern(
                ctx_data, seed_data, ctx_primary_key, tgt_context_key, imputation.columns, is_sequential
            )

        # keep at most 500k samples in memory before decoding and writing to disk
        buffer = FixedSizeSampleBuffer(capacity=500_000)

        decode_prev_steps = None

        _LOG.info(f"generate {no_of_batches} batches")
        for batch in range(1, no_of_batches + 1):
            ctx_batch = ctx_data[ctx_data["__BATCH"] == batch].drop(columns="__BATCH")
            ctx_batch = apply_encoding_type_dtypes(ctx_batch, ctx_encoding_types)
            batch_size = len(ctx_batch)

            seed_batch = seed_data[seed_data[tgt_context_key].isin(ctx_batch[ctx_primary_key])]
            # drop fully-NULL imputation columns from seed_batch to allow conditional generation
            if imputation:
                seed_batch = _drop_fully_null_imputed_columns(seed_batch, imputation.columns)
            seed_batch = apply_encoding_type_dtypes(seed_batch, seed_encoding_types)

            if ctx_primary_key not in ctx_batch.columns:
                ctx_batch[ctx_primary_key] = pd.Series(
                    data=_generate_primary_keys(len(ctx_batch), type="int").values,
                    index=ctx_batch.index,
                )

            # align ctx_batch and seed_batch by their respective keys
            ctx_batch = ctx_batch.sort_values(ctx_primary_key).reset_index(drop=True)
            seed_batch = seed_batch.sort_values(tgt_context_key).reset_index(drop=True)

            # encode ctx_batch
            _LOG.info(f"encode context {ctx_batch.shape}")
            ctx_batch_encoded, ctx_primary_key_encoded, _ = encode_df(
                df=ctx_batch, stats=ctx_stats, ctx_primary_key=ctx_primary_key
            )
            # pad left context sequences to ensure non-empty sequences
            ctx_batch_encoded = pad_ctx_sequences(ctx_batch_encoded)
            ctx_keys = ctx_batch_encoded[ctx_primary_key_encoded]
            ctx_keys.rename(tgt_context_key, inplace=True)

            # encode seed_batch
            _LOG.info(f"encode seed data values {seed_batch.shape}")
            seed_batch_encoded, _, seed_context_key_encoded = encode_df(
                df=seed_batch, stats=tgt_stats, tgt_context_key=tgt_context_key
            )
            seed_batch_grouped = seed_batch.groupby(tgt_context_key, sort=False)
            seed_batch_encoded_grouped = seed_batch_encoded.groupby(seed_context_key_encoded, sort=False)
            # it is assumed that all seeded sequences have the same length
            n_seed_steps = max(list(seed_batch_encoded_grouped.size()), default=0)

            # sample data from generative model
            _LOG.info(f"sample data from model with context {ctx_batch.shape}")
            if not tgt_sub_columns:
                # there are no columns to sample, emit warning and continue to batch decoding; this case can only happen for flat tables
                _LOG.warning("no target columns to sample")
                syn = ctx_keys.to_frame().reset_index(drop=True)
                buffer.add((syn, seed_batch))
            elif isinstance(model, SequentialModel):
                # Prepare context input tensors using helper
                context_inputs = _prepare_input_tensors(
                    ctx_data_encoded=ctx_batch_encoded,
                    device=model.device,
                )
                seq_steps = model.tgt_seq_len_max
                history = None
                history_state = None
                # process context just once for all sequence steps
                context = model.context_compressor(context_inputs)
                # loop over sequence steps, and pass forward history to keep model state-less
                out_df: pd.DataFrame | None = None
                decode_prev_steps = {}
                # continue sequences until they reach their predicted length
                step_ctx_keys = ctx_keys
                step_size = batch_size
                for seq_step in range(seq_steps):
                    # exit early if nothing more to sample
                    if step_size == 0:
                        break

                    # get seed data for current step
                    seed_step = seed_batch_grouped.nth(seq_step) if seq_step < n_seed_steps else pd.DataFrame()

                    # drop NULL imputation columns from seed_step to allow conditional generation
                    if imputation and len(seed_step) > 0:
                        seed_step = _drop_fully_null_imputed_columns(seed_step, imputation.columns)

                    # encode seed_step (after dropping NULL imputation columns)
                    if len(seed_step) > 0:
                        seed_step_encoded, _, _ = encode_df(
                            df=seed_step, stats=tgt_stats, tgt_context_key=tgt_context_key
                        )
                    else:
                        seed_step_encoded = pd.DataFrame()

                    # fix SIDX by incrementing ourselves instead of sampling
                    sidx = pd.Series([seq_step] * step_size)
                    sidx_df = encode_positional_column(sidx, max_seq_len=seq_steps, prefix=SIDX_SUB_COLUMN_PREFIX)
                    sidx_vals = {
                        c: torch.unsqueeze(
                            torch.as_tensor(sidx_df[c].to_numpy(), device=model.device).type(torch.int),
                            dim=-1,
                        )
                        for c in sidx_df
                    }

                    # fix SLEN by propagating sampled SLEN from first step
                    slen_vals = {}
                    if has_slen and seq_step > 0:
                        slen = out_df[SLEN_SUB_COLUMN_PREFIX]
                        slen = encode_positional_column(slen, max_seq_len=seq_len_max, prefix=SLEN_SUB_COLUMN_PREFIX)
                        slen_vals = {
                            col: torch.unsqueeze(
                                torch.as_tensor(slen[col], dtype=torch.int64, device=model.device),
                                dim=-1,
                            )
                            for col in slen
                        }

                    # fix RIDX by propagating sampled RIDX from first step after seeded part of sequence
                    ridx_vals = {}
                    if has_ridx and seq_step > n_seed_steps:
                        ridx = (out_df[RIDX_SUB_COLUMN_PREFIX] - 1).clip(lower=0)
                        ridx = encode_positional_column(ridx, max_seq_len=seq_len_max, prefix=RIDX_SUB_COLUMN_PREFIX)
                        ridx_vals = {
                            col: torch.unsqueeze(
                                torch.as_tensor(ridx[col], dtype=torch.int64, device=model.device),
                                dim=-1,
                            )
                            for col in ridx
                        }

                    # fix SDEC according to SIDX and SLEN
                    sdec_vals = {}
                    if has_sdec:
                        if seq_step > 0:
                            slen = out_df[SLEN_SUB_COLUMN_PREFIX]
                            sdec = (
                                (10 * sidx / slen.clip(lower=1)).clip(upper=9).astype(int)
                            )  # sequence index decile; clip as during GENERATE SIDX can become larger than SLEN
                        else:
                            sdec = pd.Series([0] * step_size)  # initial sequence index decile
                        sdec_vals = {
                            f"{SDEC_SUB_COLUMN_PREFIX}cat": torch.unsqueeze(
                                torch.as_tensor(sdec.to_numpy(), device=model.device).type(torch.int), dim=-1
                            )
                        }

                    # fix seeded columns
                    seed_vals = {}
                    if len(seed_step_encoded) > 0:
                        seed_vals = {
                            col: torch.unsqueeze(
                                torch.as_tensor(seed_step_encoded[col].to_numpy(), device=model.device).type(torch.int),
                                dim=-1,
                            )
                            for col in seed_step_encoded.columns
                            if col in tgt_sub_columns
                        }

                    fixed_values = sidx_vals | slen_vals | ridx_vals | sdec_vals | seed_vals
                    column_order = _resolve_gen_column_order(
                        column_stats=tgt_stats["columns"],
                        cardinalities=tgt_cardinalities,
                        rebalancing=rebalancing,
                        imputation=imputation,
                        seed_data=seed_step,
                        fairness=fairness,
                    )
                    out_dct, history, history_state = model(
                        x=None,  # not used in generation forward pass
                        mode="gen",
                        batch_size=step_size,
                        fixed_probs=fixed_probs,
                        fixed_values=fixed_values,
                        temperature=sampling_temperature,
                        top_p=sampling_top_p,
                        history=history,
                        history_state=history_state,
                        context=context,
                        column_order=column_order,
                    )

                    # transform output dict to tensor for memory efficiency
                    out_pt = torch.stack(list(out_dct.values()), dim=0).transpose(0, 1)
                    # reshape tensor to pandas
                    out_df = _reshape_pt_to_pandas(
                        data=[out_pt],
                        sub_cols=tgt_sub_columns,
                        keys=[step_ctx_keys],
                        key_name=tgt_context_key,
                    )
                    # decode positional columns
                    out_df[SIDX_SUB_COLUMN_PREFIX] = decode_positional_column(
                        out_df, seq_len_max, prefix=SIDX_SUB_COLUMN_PREFIX
                    )
                    if has_slen:
                        out_df[SLEN_SUB_COLUMN_PREFIX] = decode_positional_column(
                            out_df, seq_len_max, prefix=SLEN_SUB_COLUMN_PREFIX
                        )
                        out_df[SLEN_SUB_COLUMN_PREFIX] = out_df[SLEN_SUB_COLUMN_PREFIX].clip(lower=seq_len_min)
                    if has_ridx:
                        out_df[RIDX_SUB_COLUMN_PREFIX] = decode_positional_column(
                            out_df, seq_len_max, prefix=RIDX_SUB_COLUMN_PREFIX
                        )
                        out_df[RIDX_SUB_COLUMN_PREFIX] = out_df[RIDX_SUB_COLUMN_PREFIX].clip(
                            lower=seq_len_min - seq_step, upper=seq_len_max
                        )
                    # calculate include step mask (True: include current step, False: exclude current step)
                    if RIDX_SUB_COLUMN_PREFIX in out_df.columns:
                        include_mask = out_df[RIDX_SUB_COLUMN_PREFIX] > 0
                    else:
                        # fall back to calculating the mask based on SLEN column (backwards compatibility)
                        include_mask = out_df[SIDX_SUB_COLUMN_PREFIX] < out_df[SLEN_SUB_COLUMN_PREFIX]
                    include_mask = include_mask | (out_df[SIDX_SUB_COLUMN_PREFIX] < n_seed_steps)
                    next_step_size = include_mask.sum()
                    # filter next iteration inputs only when threshold is passed
                    # or there is no more data to sample on next iteration
                    if step_size > next_step_size or next_step_size == 0:
                        _LOG.info(f"step_size: {step_size} -> {next_step_size}")
                        step_size = next_step_size
                        step_ctx_keys = step_ctx_keys[include_mask].reset_index(drop=True)
                        out_df = out_df[include_mask].reset_index(drop=True)
                        out_pt = out_pt[include_mask, ...]
                        # filter context, if it is a sequential context then filter the list of contexts
                        context = [
                            c[include_mask, ...]
                            if isinstance(c, torch.Tensor)
                            else [sub_c[include_mask, ...] for sub_c in c]
                            for c in context
                        ]
                        history = history[include_mask, ...]
                        history_state = tuple(h[:, include_mask, ...] for h in history_state)
                    # filter seed_step to match step_ctx_keys (always, not just when filtering above)
                    if len(seed_step) > 0:
                        seed_step = seed_step[seed_step[tgt_context_key].isin(step_ctx_keys)].reset_index(drop=True)
                    # accumulate outputs in memory
                    buffer.add((out_pt, step_ctx_keys, seed_step))
                    # increment progress by 1 for each step
                    progress.update(advance=1)
                    # conditionally decode on step processing end
                    if buffer.is_full():
                        syn = decode_buffered_samples(
                            buffer=buffer,
                            tgt_stats=tgt_stats,
                            tgt_sub_columns=tgt_sub_columns,
                            tgt_primary_key=tgt_primary_key,
                            tgt_context_key=tgt_context_key,
                            decode_prev_steps=decode_prev_steps,
                            impute_columns=imputation.columns if imputation else None,
                        )
                        persist_data_part(syn, output_path, f"{buffer.n_clears:06}.{0:06}")
                        buffer.clear()
            else:  # isinstance(model, FlatModel)
                # Prepare context input tensors using helper
                x = _prepare_input_tensors(
                    ctx_data_encoded=ctx_batch_encoded,
                    device=model.device,
                )
                fixed_values = {
                    col: torch.as_tensor(seed_batch_encoded[col].to_numpy(), device=model.device).type(torch.int)
                    for col in seed_batch_encoded.columns
                    if col in tgt_sub_columns
                }

                column_order = _resolve_gen_column_order(
                    column_stats=tgt_stats["columns"],
                    cardinalities=tgt_cardinalities,
                    rebalancing=rebalancing,
                    imputation=imputation,
                    seed_data=seed_batch,
                    fairness=fairness,
                )
                out_dct, _ = model(
                    x,
                    mode="gen",
                    batch_size=batch_size,
                    fixed_probs=fixed_probs,
                    fixed_values=fixed_values,
                    temperature=sampling_temperature,
                    top_p=sampling_top_p,
                    fairness_transforms=fairness_transforms,
                    column_order=column_order,
                )

                syn = pd.concat(
                    [ctx_keys]
                    + [
                        pd.Series(out_dct[sub_col].detach().cpu().numpy(), dtype="int32", name=sub_col)
                        for sub_col in tgt_cardinalities.keys()
                    ],
                    axis=1,
                )
                syn.reset_index(drop=True, inplace=True)
                buffer.add((syn, seed_batch))

            # send number of processed batches / steps
            progress.update(completed=batch * (seq_len_max + 1) - 1)

            # conditionally decode on batch processing end
            if buffer.is_full():
                syn = decode_buffered_samples(
                    buffer=buffer,
                    tgt_stats=tgt_stats,
                    tgt_sub_columns=tgt_sub_columns,
                    tgt_primary_key=tgt_primary_key,
                    tgt_context_key=tgt_context_key,
                    decode_prev_steps=decode_prev_steps,
                    impute_columns=imputation.columns if imputation else None,
                )
                persist_data_part(syn, output_path, f"{buffer.n_clears:06}.{0:06}")
                buffer.clear()

            progress.update(completed=batch * (seq_len_max + 1))

        # decode before exit if buffer is not empty or seq_len_max is 0
        if not buffer.is_empty() or seq_len_max == 0:
            syn = decode_buffered_samples(
                buffer=buffer,
                tgt_stats=tgt_stats,
                tgt_sub_columns=tgt_sub_columns,
                tgt_primary_key=tgt_primary_key,
                tgt_context_key=tgt_context_key,
                decode_prev_steps=decode_prev_steps,
                impute_columns=imputation.columns if imputation else None,
            )
            persist_data_part(syn, output_path, f"{buffer.n_clears:06}.{0:06}")
            buffer.clear()
    _LOG.info(f"GENERATE_TABULAR finished in {time.time() - t0:.2f}s")


@torch.no_grad()
def log_prob(
    *,
    data: pd.DataFrame,
    ctx_data: pd.DataFrame | None = None,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
) -> np.ndarray:
    """
    Compute log probabilities for tabular data.

    This function evaluates the likelihood of each sample under the trained model,
    returning per-sample log probabilities. Lower (more negative) values indicate
    samples that are less likely according to the model, which can be useful for
    anomaly detection or model evaluation.

    For sequential models, data should be grouped by context key. The function
    returns log probabilities per sequence (not per step).

    Args:
        data: Input data to evaluate. For flat models: DataFrame of shape (n_samples, n_features).
              For sequential models: DataFrame with rows grouped by context key.
        ctx_data: Optional context data for models trained with context.
        device: Device to run computation on ('cuda' or 'cpu').
               Defaults to 'cuda' if available, else 'cpu'.
        workspace_dir: Directory path for workspace containing the trained model.

    Returns:
        Array of log probabilities. For flat models: shape (n_samples,).
        For sequential models: shape (n_sequences,), one value per sequence.
        Higher (less negative) values indicate higher likelihood under the model.
    """
    _LOG.info("LOG_PROB started")
    t0 = time.time()

    _LOG.info(f"Computing log probabilities for {len(data)} samples")

    # Load model and stats using helper (supports both flat and sequential models)
    model, tgt_stats, ctx_stats, model_configs, device = _load_trained_model(
        workspace_dir=workspace_dir,
        device=device,
        column_order=None,  # Use default training order
        allow_sequential=True,  # Support both flat and sequential models
        require_weights=True,  # Weights required for log_prob
    )

    is_sequential = tgt_stats["is_sequential"]
    _LOG.info(f"{is_sequential=}")

    # Get cardinalities and sub columns
    tgt_cardinalities = get_cardinalities(tgt_stats)
    tgt_sub_columns = get_sub_columns_from_cardinalities(tgt_cardinalities)

    # Get keys
    tgt_context_key = tgt_stats.get("keys", {}).get("context_key")
    ctx_primary_key = ctx_stats.get("keys", {}).get("primary_key")

    # Build paths based on workspace dir
    workspace_dir = ensure_workspace_dir(workspace_dir)
    workspace = Workspace(workspace_dir)

    # For sequential models, validate that context key is present
    if is_sequential:
        if tgt_context_key is None:
            raise ValueError(
                "Sequential models require tgt_context_key in data. "
                "Please ensure your data includes the context key column."
            )
        if tgt_context_key not in data.columns:
            raise ValueError(
                f"Sequential data must contain tgt_context_key column `{tgt_context_key}`. "
                f"Found columns: {list(data.columns)}"
            )

    # Prepare context data using helper
    # For sequential models, validate_size should check sequences, not individual rows
    ctx_data, ctx_primary_key, tgt_context_key, has_context = _prepare_context_data(
        workspace=workspace,
        ctx_stats=ctx_stats,
        tgt_stats=tgt_stats,
        ctx_data=ctx_data,
        validate_size=not is_sequential,  # Only validate size for flat models
        data_size=len(data) if not is_sequential else None,
    )

    # Link data to context for flat data
    if not is_sequential and tgt_context_key and tgt_context_key not in data.columns:
        data = data.copy()
        data[tgt_context_key] = ctx_data[ctx_primary_key].values

    # Resolve encoding types using helper
    ctx_encoding_types, tgt_encoding_types = _prepare_encoding_types(
        ctx_stats=ctx_stats,
        tgt_stats=tgt_stats,
        has_context=has_context,
    )

    # Apply encoding types for dtype harmonization
    ctx_data = apply_encoding_type_dtypes(ctx_data, ctx_encoding_types)
    data = apply_encoding_type_dtypes(data, tgt_encoding_types)

    # Encode context and target data using helper
    ctx_data_encoded, ctx_primary_key_encoded, data_encoded = _encode_context_and_target(
        ctx_data=ctx_data,
        tgt_data=data,
        ctx_stats=ctx_stats,
        tgt_stats=tgt_stats,
        ctx_primary_key=ctx_primary_key,
        tgt_context_key=tgt_context_key,
    )

    # Prepare input tensors using helper
    x = _prepare_input_tensors(
        ctx_data_encoded=ctx_data_encoded,
        device=device,
    )

    # Prepare target tensors - handle sequential vs flat differently
    if is_sequential:
        # For sequential models, data_encoded has list columns that need to be converted to tensors
        # The shape should be (batch, seq_len, 1) for each column
        data_tensors = {}
        for col in tgt_sub_columns:
            if col in data_encoded.columns:
                # Convert list column to tensor: (batch_size, seq_len, 1)
                col_data = data_encoded[col]
                if isinstance(col_data.iloc[0], list):
                    # List column - convert to tensor
                    max_len = max(len(lst) if isinstance(lst, list) else 1 for lst in col_data)
                    tensor_list = []
                    for lst in col_data:
                        if isinstance(lst, list):
                            padded = lst + [0] * (max_len - len(lst))
                        else:
                            padded = [lst] + [0] * (max_len - 1)
                        tensor_list.append(padded)
                    data_tensors[col] = torch.as_tensor(tensor_list, device=device).type(torch.int).unsqueeze(-1)
                else:
                    # Scalar column - treat as single-step sequences
                    data_tensors[col] = (
                        torch.as_tensor(col_data.to_numpy(), device=device).type(torch.int).unsqueeze(-1).unsqueeze(-1)
                    )
            else:
                # Column not present - create zero tensor
                batch_size = len(data_encoded)
                # Determine seq_len from other columns if available
                if len(data_encoded) > 0 and len(data_encoded.columns) > 0:
                    first_col = data_encoded.iloc[:, 0]
                    seq_len = (
                        max(len(lst) if isinstance(lst, list) else 1 for lst in first_col) if len(first_col) > 0 else 1
                    )
                else:
                    seq_len = 1
                data_tensors[col] = torch.zeros((batch_size, seq_len, 1), device=device, dtype=torch.int)
    else:
        # For flat models, standard tensor preparation
        data_tensors = {
            col: torch.as_tensor(data_encoded[col].to_numpy(), device=device).type(torch.int).unsqueeze(-1)
            for col in tgt_sub_columns
        }

    # Forward pass
    _LOG.info("Computing forward pass")
    output, _ = model(x | data_tensors, mode="trn")

    # Compute per-sample losses - handle sequential vs flat differently
    criterion = nn.CrossEntropyLoss(reduction="none")
    tgt_cols = list(tgt_cardinalities.keys())

    if isinstance(model, SequentialModel):
        # Sequential model loss calculation (similar to training)
        sidx_cols = {k for k in data_tensors if k.startswith(SIDX_SUB_COLUMN_PREFIX)}
        slen_cols = {k for k in data_tensors if k.startswith(SLEN_SUB_COLUMN_PREFIX)}
        ridx_cols = [k for k in data_tensors if k.startswith(RIDX_SUB_COLUMN_PREFIX)]

        # mask for data columns
        if ridx_cols:
            data_mask = torch.zeros_like(data_tensors[ridx_cols[0]], dtype=torch.int64)
            for ridx_col in ridx_cols:
                data_mask |= data_tensors[ridx_col] != 0  # mask loss for padded rows, which have RIDX=0
            data_mask = data_mask.squeeze(-1)
        else:
            # Fallback: use first data column to determine mask shape
            first_data_col = next((k for k in data_tensors if k not in sidx_cols and k not in slen_cols), None)
            if first_data_col:
                data_mask = torch.ones_like(data_tensors[first_data_col].squeeze(-1), dtype=torch.int64)
            else:
                # No data columns found - create default mask
                batch_size = len(data_encoded) if len(data_encoded) > 0 else 1
                seq_len = data_tensors[next(iter(data_tensors))].shape[1] if data_tensors else 1
                data_mask = torch.ones((batch_size, seq_len), dtype=torch.int64, device=device)
        # mask for slen columns; only first step is unmasked
        slen_mask = torch.zeros_like(data_mask)
        slen_mask[:, 0] = 1
        # mask for ridx columns: this takes the sequence padding into account to learn the stopping with ridx=0
        ridx_mask = torch.nn.functional.pad(data_mask, (1, 0), value=1)[:, :-1]
        # mask for sidx columns
        sidx_mask = torch.zeros_like(data_mask)

        # calculate per column losses
        losses_by_column = []
        for col in tgt_cols:
            if col in sidx_cols:
                mask = sidx_mask
            elif col in slen_cols:
                mask = slen_mask
            elif col in ridx_cols:
                mask = ridx_mask
            else:
                mask = data_mask
            column_loss = criterion(output[col].transpose(1, 2), data_tensors[col].squeeze(2))
            masked_loss = torch.sum(column_loss * mask, dim=1) / torch.clamp(torch.sum(mask, dim=1), min=1)
            losses_by_column.append(masked_loss)
    else:
        # Flat model loss calculation
        losses_by_column = [criterion(output[col], data_tensors[col].squeeze(1)) for col in tgt_cols]

    # sum up column level losses to get overall losses at sample level
    losses = torch.sum(torch.stack(losses_by_column, dim=0), dim=0)

    # Convert losses to log probabilities (negative loss)
    log_probs = -losses.detach().cpu().numpy()

    _LOG.info(f"LOG_PROB finished in {time.time() - t0:.2f}s")
    return log_probs
