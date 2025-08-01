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
Census Dataset Example with Business Rules Loss - Fine-tuning
============================================================

This script performs fine-tuning on an already trained model using business rules loss adjustment.
It loads an existing model and trains for one epoch with custom loss for target columns.
"""

from pathlib import Path

import pandas as pd

from mostlyai import engine


def main():
    print("🚀 Starting Census Dataset Fine-tuning with Business Rules")
    print("=" * 50)

    # Set up workspace and default logging
    base_ws = Path("ws-tabular-flat-base")  # Base model workspace
    finetuned_ws = Path("ws-tabular-flat-finetuned")  # Fine-tuned model workspace
    engine.init_logging()

    print(f"📁 Base workspace directory: {base_ws}")
    print(f"📁 Fine-tuned workspace directory: {finetuned_ws}")

    # Load original data
    print("📥 Loading census dataset...")
    url = "https://github.com/mostly-ai/public-demo-data/raw/refs/heads/dev/census"
    trn_df = pd.read_csv(f"{url}/census.csv.gz")

    print(f"📊 Dataset shape: {trn_df.shape}")
    print("📊 Workclass distribution:")
    print(trn_df["workclass"].value_counts(normalize=True))

    # Import business rules configuration
    from business_rules_config import (
        get_lambda_weight,
        get_predefined_distribution,
        get_target_columns,
        is_lora_style_enabled,
    )

    target_columns = get_target_columns()
    predefined_distributions = {col: get_predefined_distribution(col) for col in target_columns}
    lambda_weights = {col: get_lambda_weight(col) for col in target_columns}

    print(f"🎯 Target columns: {target_columns}")
    print(f"📊 Predefined distributions: {predefined_distributions}")
    print(f"⚖️ Lambda weights: {lambda_weights}")

    # Check if we have an existing base model
    base_model_weights_path = base_ws / "ModelStore" / "model-data" / "model-weights.pt"

    if not base_model_weights_path.exists():
        print("❌ No existing base model found!")
        print("Please run create_base_model.py first to create a base model.")
        print(f"Expected path: {base_model_weights_path}")
        return

    print(f"✅ Found existing base model weights at: {base_model_weights_path}")

    # Check for additional model files
    optimizer_path = base_ws / "ModelStore" / "model-data" / "optimizer.pt"
    lr_scheduler_path = base_ws / "ModelStore" / "model-data" / "lr-scheduler.pt"
    progress_path = base_ws / "ModelStore" / "model-data" / "progress-messages.csv"

    print("📋 Checking base model files:")
    print(f"  - Model weights: {'✅' if base_model_weights_path.exists() else '❌'}")
    print(f"  - Optimizer state: {'✅' if optimizer_path.exists() else '❌'}")
    print(f"  - LR scheduler state: {'✅' if lr_scheduler_path.exists() else '❌'}")
    print(f"  - Progress messages: {'✅' if progress_path.exists() else '❌'}")

    # Copy base model workspace to fine-tuned workspace
    print("\n📋 Copying base model to fine-tuned workspace...")
    import shutil

    if finetuned_ws.exists():
        shutil.rmtree(finetuned_ws)
    shutil.copytree(base_ws, finetuned_ws)
    print(f"✅ Base model copied to: {finetuned_ws}")

    # Perform fine-tuning with business rules
    print("\n🔧 Starting fine-tuning with business rules...")
    print("=" * 50)

    try:
        # Fine-tune the model with business rules for 10 epochs
        engine.train(
            # Use fine-tuned workspace
            workspace_dir=str(finetuned_ws),
            # Fine-tuning parameters
            max_epochs=1,  # 10 epochs for fine-tuning
            max_training_time=3600.0,  # 1 hour max
            # Business rules parameters
            target_columns=target_columns,
            predefined_distributions=predefined_distributions,
            lambda_weights=lambda_weights,
            enable_lora_style=is_lora_style_enabled(),
            # Keep existing model settings
            enable_flexible_generation=True,
            # Model state strategy for fine-tuning
            model_state_strategy="REUSE",  # Load existing model weights but start fresh
        )

        # Generate some samples to verify the business rules
        print("\n🎲 Generating sample data to verify business rules...")

        target_column_positions = {"age": -1}  # Generate synthetic data to verify business rules
        engine.generate(
            workspace_dir=str(finetuned_ws),
            # Position target columns at specific positions in generation order
            # For age column, position it at the end (last position)
            target_column_positions=target_column_positions,  # -1 means last position
        )
        print(f"🎯 Target column positions: {target_column_positions}")
        # Load and compare synthetic data
        print("\n📊 Loading synthetic data for verification...")
        synthetic_df = pd.read_parquet(finetuned_ws / "SyntheticData")

        print(f"🎯 Synthetic data shape: {synthetic_df.shape}")

        # Save fine-tuned model synthetic data with different name
        finetuned_synthetic_path = finetuned_ws / "SyntheticData-FineTuned"
        synthetic_df.to_parquet(finetuned_synthetic_path)
        print(f"💾 Fine-tuned model synthetic data saved to: {finetuned_synthetic_path}")

        # Compare distributions for target columns
        print("\n📊 Distribution Comparison:")
        for col in target_columns:
            if col in trn_df.columns and col in synthetic_df.columns:
                print(f"\n{col} distribution:")
                print("Original:", trn_df[col].value_counts(normalize=True).head())
                print("Synthetic:", synthetic_df[col].value_counts(normalize=True).head())
                print("Predefined:", predefined_distributions.get(col, {}))

        # Show the new distribution after fine-tuning
        print("\n📊 Checking results...")
        if (finetuned_ws / "ModelStore" / "model-data" / "model-weights.pt").exists():
            print("✅ Fine-tuned model saved successfully")
        else:
            print("❌ Fine-tuned model not found")

        print("\n✅ Fine-tuning with business rules completed successfully!")
        print(f"📁 Results saved in: {finetuned_ws}")

    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        raise


if __name__ == "__main__":
    main()
