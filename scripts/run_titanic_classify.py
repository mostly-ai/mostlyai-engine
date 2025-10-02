#!/usr/bin/env python3
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

import sys
from pathlib import Path

import pandas as pd

from mostlyai import engine
from mostlyai.engine.domain import ModelType

ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT / "data" / "titanic.csv",
    ROOT / "data" / "titanic_full.csv",
]


def _load_titanic_df() -> pd.DataFrame:
    for path in DATA_CANDIDATES:
        if path.exists():
            return pd.read_csv(path)
    raise SystemExit(f"Missing dataset. Tried: {', '.join(str(p) for p in DATA_CANDIDATES)}")


def main() -> None:
    df = _load_titanic_df()
    if "survived" not in df.columns:
        raise SystemExit("Expected target column 'survived' not found in dataset")

    features = [c for c in df.columns if c != "survived"]
    ws = ROOT / "ws-titanic-classify"

    engine.init_logging()

    # Train generator
    print(f"\n=== Training workspace: {ws} ===")
    engine.split(workspace_dir=ws, tgt_data=df, model_type=ModelType.tabular)
    engine.analyze(workspace_dir=ws, value_protection=False)
    engine.encode(workspace_dir=ws)
    engine.train(workspace_dir=ws, enable_flexible_generation=True)

    # Generate synthetic data (optional step in pipeline)
    engine.generate(workspace_dir=ws, sample_size=len(df))

    # Classify using trained generator
    print("\n=== Classifying target 'survived' ===")
    proba_df = engine.classify(data=df, features=features, target="survived", workspace_dir=ws)
    out_path = ws / "SyntheticData" / "classification_proba.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proba_df.to_csv(out_path, index=False)
    print(f"Saved classification probabilities to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
