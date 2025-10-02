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

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from mostlyai.engine._common import (
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
    CTXFLT,
    CTXSEQ,
    get_argn_name,
    get_cardinalities,
    get_sub_columns_from_cardinalities,
)
from mostlyai.engine._tabular.argn import FlatModel
from mostlyai.engine._tabular.encoding import encode_df
from mostlyai.engine._tabular.generation import _resolve_gen_column_order
from mostlyai.engine._workspace import Workspace, ensure_workspace_dir


@torch.no_grad()
def classify(
    data: pd.DataFrame,
    features: list[str],
    target: str,
    workspace_dir: str | Path,
    output: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute predictive probabilities for a target column conditioned on feature columns.

    - Orders columns as [features, target, rest]
    - Uses model forward to obtain probability distribution for target without sampling
    - Stops logically after target (no use of further sampled values)

    Returns a DataFrame with feature columns and `proba_{category}` columns for the target.
    """
    workspace_dir = ensure_workspace_dir(workspace_dir)
    workspace = Workspace(workspace_dir)

    # Read stats and configs
    tgt_stats = workspace.tgt_stats.read()
    ctx_stats = workspace.ctx_stats.read()
    model_configs = workspace.model_configs.read()

    # Validate inputs
    missing = [c for c in features + [target] if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    # Resolve cardinalities and sub-columns
    tgt_cardinalities = get_cardinalities(tgt_stats)
    ctx_cardinalities = get_cardinalities(ctx_stats)
    tgt_sub_columns = get_sub_columns_from_cardinalities(tgt_cardinalities)

    # Determine desired column order: [features, target, rest]
    column_stats: dict[str, Any] = tgt_stats["columns"]
    feat_argn = [
        get_argn_name(
            argn_processor=column_stats[col][ARGN_PROCESSOR],
            argn_table=column_stats[col][ARGN_TABLE],
            argn_column=column_stats[col][ARGN_COLUMN],
        )
        for col in features
        if col in column_stats
    ]
    if target not in column_stats:
        raise ValueError(f"Target column `{target}` not present in workspace stats")
    target_argn = get_argn_name(
        argn_processor=column_stats[target][ARGN_PROCESSOR],
        argn_table=column_stats[target][ARGN_TABLE],
        argn_column=column_stats[target][ARGN_COLUMN],
    )

    default_order = _resolve_gen_column_order(
        column_stats=tgt_stats["columns"],
        cardinalities=tgt_cardinalities,
    )
    rest = [c for c in default_order if c not in feat_argn + [target_argn]]
    classify_order = feat_argn + [target_argn] + rest

    # Instantiate model
    model_units = model_configs.get("model_units")
    is_sequential = tgt_stats.get("is_sequential", False)
    if is_sequential:
        raise NotImplementedError("classify is only implemented for flat tabular models")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: FlatModel = FlatModel(
        tgt_cardinalities=tgt_cardinalities,
        ctx_cardinalities=ctx_cardinalities,
        ctxseq_len_median={},
        model_size=model_units,
        column_order=classify_order,
        device=device,
    )

    # Load weights if present
    from mostlyai.engine._tabular.training import load_model_weights

    if workspace.model_tabular_weights_path.exists():
        load_model_weights(model=model, path=workspace.model_tabular_weights_path, device=device)
    else:
        raise FileNotFoundError("Trained model weights not found in workspace; train the model first")
    model.to(device)
    model.eval()

    # Prepare (possibly empty) context and encode
    ctx_df = pd.DataFrame(index=data.index)
    ctx_encoded, _, _ = encode_df(df=ctx_df, stats=ctx_stats)
    ctxflt_inputs = {
        col: torch.unsqueeze(
            torch.as_tensor(ctx_encoded[col].to_numpy(), device=device).type(torch.int),
            dim=-1,
        )
        for col in ctx_encoded.columns
        if col.startswith(CTXFLT)
    }
    ctxseq_inputs = {
        col: torch.unsqueeze(
            torch.nested.as_nested_tensor(
                [torch.as_tensor(t, device=device).type(torch.int) for t in ctx_encoded[col]],
                device=device,
            ),
            dim=-1,
        )
        for col in ctx_encoded.columns
        if col.startswith(CTXSEQ)
    }
    x = ctxflt_inputs | ctxseq_inputs

    # Encode feature seed and pass as fixed_values
    seed_df = data[features].copy()
    seed_encoded, _, _ = encode_df(df=seed_df, stats=tgt_stats)
    fixed_values = {
        col: torch.as_tensor(seed_encoded[col].to_numpy(), device=device).type(torch.int)
        for col in seed_encoded.columns
        if col in tgt_sub_columns
    }

    # Identify target's first sub-column
    target_subs = [sc for sc in tgt_sub_columns if sc.startswith(target_argn)]
    if not target_subs:
        raise RuntimeError("Failed to resolve target sub-columns")
    target_first_sub = target_subs[0]

    # Forward to get target probabilities
    _, probs_dct = model(
        x,
        mode="gen",
        batch_size=len(seed_df),
        fixed_probs={},
        fixed_values=fixed_values,
        temperature=1.0,
        top_p=1.0,
        return_probs=[target_first_sub],
        fairness_transforms=None,
    )
    probs = probs_dct.get(target_first_sub)
    if probs is None:
        raise RuntimeError("Failed to obtain probabilities for target column")

    # Build human-readable probability column names if available
    proba_np = probs.detach().cpu().numpy()
    codes = tgt_stats["columns"].get(target, {}).get("codes") or {}
    inv_codes = {v: k for k, v in codes.items()} if isinstance(codes, dict) else {}
    cols = [f"proba_{inv_codes.get(i, i)}" for i in range(proba_np.shape[1])]
    proba_df = pd.DataFrame(proba_np, columns=cols, index=seed_df.index)

    return pd.concat([data[features].reset_index(drop=True), proba_df.reset_index(drop=True)], axis=1)
