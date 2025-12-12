#!/usr/bin/env python
"""
Train the Foundational Tabular Transformer on the T4 dataset.

Usage:
    python scripts/foundational/train_transformer_t4.py --num_steps 1000 --batch_size 64

Requirements:
    pip install huggingface_hub pyarrow
"""

import argparse
import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfFileSystem

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

    Each parquet file in T4 is a separate table with its own schema.
    We load them individually using HfFileSystem to avoid schema conflicts.

    Args:
        max_tables: Maximum number of tables to yield (None = unlimited)

    Yields:
        pandas DataFrames with categorical columns only
    """
    _LOG.info("Connecting to T4 dataset via HfFileSystem...")

    fs = HfFileSystem()
    base_path = "datasets/mlfoundations/t4-full"

    # List all chunk zip files
    try:
        all_files = fs.ls(base_path, detail=False)
        chunk_zips = sorted([f for f in all_files if f.endswith(".zip")])
        _LOG.info(f"Found {len(chunk_zips)} chunk zip files")
    except Exception as e:
        _LOG.error(f"Failed to list dataset: {e}")
        raise

    tables_yielded = 0
    tables_skipped = 0

    for chunk_path in chunk_zips:
        chunk_name = chunk_path.split("/")[-1]
        _LOG.info(f"Processing {chunk_name}...")

        try:
            # Download and open the zip file
            with fs.open(chunk_path, "rb") as f:
                zip_data = f.read()

            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                parquet_files = [n for n in zf.namelist() if n.endswith(".parquet")]
                _LOG.info(f"  Found {len(parquet_files)} parquet files in {chunk_name}")

                for pq_name in parquet_files:
                    try:
                        # Read parquet file from zip
                        with zf.open(pq_name) as pq_file:
                            pq_data = pq_file.read()
                            table = pq.read_table(io.BytesIO(pq_data))
                            df = table.to_pandas()

                        # Filter to categorical columns only (object/string dtype)
                        cat_cols = df.select_dtypes(
                            include=["object", "string", "category"]
                        ).columns.tolist()

                        # Skip if too few categorical columns
                        if len(cat_cols) < 2:
                            tables_skipped += 1
                            continue

                        # Keep only categorical columns
                        df = df[cat_cols]

                        # Skip empty or tiny tables
                        if len(df) < 10:
                            tables_skipped += 1
                            continue

                        yield df
                        tables_yielded += 1

                        if tables_yielded % 100 == 0:
                            _LOG.info(
                                f"  Progress: yielded {tables_yielded} tables, skipped {tables_skipped}"
                            )

                        if max_tables is not None and tables_yielded >= max_tables:
                            _LOG.info(f"Reached max_tables limit: {max_tables}")
                            return

                    except Exception as e:
                        _LOG.debug(f"  Failed to read {pq_name}: {e}")
                        tables_skipped += 1
                        continue

        except Exception as e:
            _LOG.warning(f"Failed to process {chunk_name}: {e}")
            continue

    _LOG.info(f"Finished: yielded {tables_yielded} tables, skipped {tables_skipped}")


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
