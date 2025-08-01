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

"""
Create Base Model for Fine-tuning
=================================

This script creates a base model that can be used for fine-tuning with business rules.
It runs the full training pipeline to create a trained model.
"""

from pathlib import Path

import pandas as pd

from mostlyai import engine


def main():
    print("🚀 Creating Base Model for Fine-tuning")
    print("=" * 50)

    # Set up workspace and default logging
    ws = Path("ws-tabular-flat-base")  # Separate workspace for base model
    engine.init_logging()

    print(f"📁 Workspace directory: {ws}")

    # Load original data
    print("📥 Loading census dataset...")
    url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
    trn_df = pd.read_csv(f"{url}/census.csv.gz")

    print(f"📊 Dataset shape: {trn_df.shape}")
    print("📊 Workclass distribution:")
    print(trn_df["workclass"].value_counts(normalize=True))

    # Execute the engine steps to create a base model
    print("\n🔧 Step 1: Splitting data...")
    engine.split(
        workspace_dir=ws,
        tgt_data=trn_df,
        model_type="TABULAR",
    )

    print("📊 Step 2: Analyzing data...")
    engine.analyze(workspace_dir=ws)

    print("🔢 Step 3: Encoding data...")
    engine.encode(workspace_dir=ws)

    print("🤖 Step 4: Training base model...")
    engine.train(
        workspace_dir=ws,
        # No epoch limit - let it train fully
        max_training_time=3600.0,  # 1 hour max
    )

    print("🎲 Step 5: Generating sample data...")
    # Generate synthetic data with base model
    engine.generate(workspace_dir=ws)

    # Load and display synthetic data
    print("\n📊 Loading synthetic data...")
    synthetic_df = pd.read_parquet(ws / "SyntheticData")

    print(f"🎯 Synthetic data shape: {synthetic_df.shape}")
    print("📊 Workclass distribution in synthetic data:")
    print(synthetic_df["workclass"].value_counts(normalize=True))

    # Save base model synthetic data with different name
    base_synthetic_path = ws / "SyntheticData-BaseModel"
    synthetic_df.to_parquet(base_synthetic_path)
    print(f"💾 Base model synthetic data saved to: {base_synthetic_path}")

    print("\n✅ Base model created successfully!")
    print(f"📁 Base model workspace: {ws}")
    print("📁 Next step: Run fine-tuning with business rules")


if __name__ == "__main__":
    main()
