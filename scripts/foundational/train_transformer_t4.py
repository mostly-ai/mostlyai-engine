#!/usr/bin/env python
"""
Train the Foundational Tabular Transformer on the T4 dataset.

Usage:
    python scripts/train_transformer_t4.py --num_steps 1000 --batch_size 64

Requirements:
    pip install datasets
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset

from mostlyai.engine._tabular.transformer import create_model
from mostlyai.engine._tabular.transformer_training import (
    pretrain_mcm_streaming,
    save_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
_LOG = logging.getLogger(__name__)


def t4_table_iterator(max_tables: int | None = None):
    """
    Iterator that yields DataFrames from the T4 dataset.

    Args:
        max_tables: Maximum number of tables to yield (None = unlimited)

    Yields:
        pandas DataFrames with categorical columns only
    """
    _LOG.info("Loading T4 dataset (streaming mode)...")

    # Load dataset in streaming mode
    dataset = load_dataset(
        "mlfoundations/t4-full",
        split="train",
        streaming=True,
    )

    tables_yielded = 0
    for example in dataset:
        # Each example should be a table - convert to DataFrame
        # The exact structure depends on how T4 stores tables
        # Let's handle different possible formats

        if isinstance(example, dict):
            # Try to convert dict to DataFrame
            try:
                # T4 stores tables as dicts with column names as keys
                df = pd.DataFrame(example)
            except Exception as e:
                _LOG.debug(f"Skipping table: {e}")
                continue
        elif isinstance(example, pd.DataFrame):
            df = example
        else:
            _LOG.debug(f"Unknown example type: {type(example)}")
            continue

        # Filter to categorical columns only
        cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        # Skip if too few categorical columns
        if len(cat_cols) < 2:
            continue

        # Keep only categorical columns
        df = df[cat_cols]

        # Skip empty tables
        if len(df) == 0:
            continue

        yield df
        tables_yielded += 1

        if tables_yielded % 100 == 0:
            _LOG.info(f"Yielded {tables_yielded} tables")

        if max_tables is not None and tables_yielded >= max_tables:
            _LOG.info(f"Reached max_tables limit: {max_tables}")
            break


def main():
    parser = argparse.ArgumentParser(description="Train transformer on T4 dataset")
    parser.add_argument("--num_steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--max_tables", type=int, default=None, help="Max tables to use (for testing)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_every", type=int, default=100, help="Log every N steps")
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Setup
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _LOG.info(f"Using device: {device}")

    # Create model
    _LOG.info(f"Creating {args.model_size} model...")
    model = create_model(args.model_size)
    num_params = sum(p.numel() for p in model.parameters())
    _LOG.info(f"Model has {num_params:,} parameters")

    # Create table iterator
    table_iter = t4_table_iterator(max_tables=args.max_tables)

    # Pre-train
    _LOG.info(f"Starting pre-training for {args.num_steps} steps...")
    model = pretrain_mcm_streaming(
        model=model,
        table_iterator=table_iter,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
    )

    # Save final model
    final_path = checkpoint_dir / "model_final.pt"
    save_model(model, final_path)
    _LOG.info(f"Training complete! Model saved to {final_path}")


if __name__ == "__main__":
    main()
