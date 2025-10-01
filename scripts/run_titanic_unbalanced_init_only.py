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

from pathlib import Path

import pandas as pd

from mostlyai import engine
from mostlyai.engine._tabular.training import train as train_tabular
from mostlyai.engine.domain import ModelType

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanic_unbalanced.csv"


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    scenarios = [
        (ROOT / "ws-initonly-titanicT", True),
        (ROOT / "ws-initonly-titanicF", False),
    ]

    engine.init_logging()

    for ws, init in scenarios:
        print(f"\n=== {ws.name} init={init} ===")
        engine.split(workspace_dir=ws, tgt_data=df, model_type=ModelType.tabular)
        engine.analyze(workspace_dir=ws, value_protection=False)
        engine.encode(workspace_dir=ws)
        train_tabular(workspace_dir=ws, enable_flexible_generation=True, weight_initialization=init)
        engine.generate(workspace_dir=ws, sample_size=len(df))

    print("\nDone.")


if __name__ == "__main__":
    main()
