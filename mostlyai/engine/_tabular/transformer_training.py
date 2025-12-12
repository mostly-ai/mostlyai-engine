"""
Training utilities for the Foundational Tabular Transformer.

Provides pre-training (MCM) and classification functions.
"""

import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .transformer import (
    FoundationalTabularTransformer,
    TabularTransformerConfig,
    create_model,
)
from .transformer_data import (
    MultiTableDataset,
    TabularDataset,
    TabularTokenizer,
    apply_masking,
    collate_fn,
    mask_target_column,
)

_LOG = logging.getLogger(__name__)


def pretrain_mcm(
    model: FoundationalTabularTransformer,
    tables: list[pd.DataFrame],
    num_epochs: int = 10,
    batch_size: int = 256,
    mask_prob: float = 0.15,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    max_columns: int = 64,
    device: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    log_every: int = 100,
) -> FoundationalTabularTransformer:
    """
    Pre-train model with Masked Cell Modeling (MCM) on multiple tables.

    Args:
        model: The transformer model to train
        tables: List of DataFrames to train on
        num_epochs: Number of training epochs
        batch_size: Batch size
        mask_prob: Probability of masking each cell
        learning_rate: Peak learning rate
        weight_decay: AdamW weight decay
        warmup_steps: Linear warmup steps
        max_columns: Maximum columns per table
        device: Device to train on (defaults to CUDA if available)
        checkpoint_dir: Directory to save checkpoints
        log_every: Log metrics every N steps

    Returns:
        Trained model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.train()

    # Create tokenizer and dataset
    tokenizer = TabularTokenizer(vocab_size=model.config.hash_vocab_size)
    dataset = MultiTableDataset(tables, tokenizer, max_columns=max_columns)

    if len(dataset) == 0:
        raise ValueError("No valid samples found in tables. Ensure tables have categorical columns.")

    _LOG.info(f"Pre-training on {len(dataset)} samples from {len(tables)} tables")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Keep simple for now
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(dataloader) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    _LOG.info(f"Training config: {num_epochs} epochs, {len(dataloader)} batches/epoch, {total_steps} total steps")
    _LOG.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Device: {device}")

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)
        for batch in pbar:
            value_ids = batch["value_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Apply masking
            masked_ids, labels = apply_masking(
                value_ids,
                attention_mask,
                mask_prob=mask_prob,
                vocab_size=model.config.hash_vocab_size,
            )

            # Skip batch if no tokens were masked (would cause NaN loss)
            num_masked = (labels != -100).sum()
            if num_masked == 0:
                continue

            # Forward pass
            optimizer.zero_grad()
            output = model(masked_ids, attention_mask, labels=labels)
            loss = output["loss"]

            # Skip if loss is NaN (shouldn't happen now but safety check)
            if torch.isnan(loss):
                _LOG.warning("NaN loss detected, skipping batch")
                continue

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Update progress bar
            avg_loss = epoch_loss / num_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        # End of epoch logging
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        _LOG.info(f"Epoch {epoch + 1}/{num_epochs} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "config": model.config,
                },
                checkpoint_path,
            )
            _LOG.info(f"Saved checkpoint to {checkpoint_path}")

    return model


def pretrain_mcm_streaming(
    model: FoundationalTabularTransformer,
    table_iterator: Iterator[pd.DataFrame],
    num_steps: int = 10000,
    batch_size: int = 256,
    mask_prob: float = 0.15,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_columns: int = 64,
    device: Optional[str] = None,
    checkpoint_dir: Optional[Path] = None,
    log_every: int = 100,
    checkpoint_every: int = 1000,
) -> FoundationalTabularTransformer:
    """
    Pre-train model with MCM using a streaming table iterator.

    Designed for large datasets like T4 that don't fit in memory.

    Args:
        model: The transformer model to train
        table_iterator: Iterator yielding DataFrames (e.g., from HuggingFace streaming)
        num_steps: Total training steps
        batch_size: Batch size
        mask_prob: Probability of masking each cell
        learning_rate: Learning rate
        weight_decay: AdamW weight decay
        max_columns: Maximum columns per table
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_every: Log metrics every N steps
        checkpoint_every: Save checkpoint every N steps

    Returns:
        Trained model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.train()

    tokenizer = TabularTokenizer(vocab_size=model.config.hash_vocab_size)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps)

    # Buffer for accumulating samples
    sample_buffer = []
    running_loss = 0.0
    steps_since_log = 0

    _LOG.info(f"Starting streaming pre-training for {num_steps} steps")
    _LOG.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Device: {device}")

    pbar = tqdm(range(num_steps), desc="Pre-training", leave=True)
    for step in pbar:
        # Fill buffer if needed
        while len(sample_buffer) < batch_size:
            try:
                table = next(table_iterator)
            except StopIteration:
                _LOG.warning("Table iterator exhausted")
                break

            # Filter to categorical columns
            cat_cols = table.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            if len(cat_cols) < 2:
                continue

            cat_cols = cat_cols[:max_columns]

            # Tokenize and add to buffer
            value_ids, attention_mask, _ = tokenizer.tokenize_dataframe(table, cat_cols, max_columns)
            for i in range(len(value_ids)):
                sample_buffer.append((value_ids[i], attention_mask[i]))

        if len(sample_buffer) < batch_size:
            _LOG.warning(f"Buffer has only {len(sample_buffer)} samples, ending training")
            break

        # Sample a batch
        batch_samples = sample_buffer[:batch_size]
        sample_buffer = sample_buffer[batch_size:]

        # Prepare batch tensors
        value_ids_list = [s[0] for s in batch_samples]
        attention_masks_list = [s[1] for s in batch_samples]

        # Pad to same length
        max_len = max(v.size(0) for v in value_ids_list)
        value_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

        for i, (ids, mask) in enumerate(zip(value_ids_list, attention_masks_list)):
            length = ids.size(0)
            value_ids[i, :length] = ids
            attention_mask[i, :length] = mask

        value_ids = value_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Apply masking
        masked_ids, labels = apply_masking(
            value_ids, attention_mask, mask_prob=mask_prob, vocab_size=model.config.hash_vocab_size
        )

        # Skip batch if no tokens were masked
        num_masked = (labels != -100).sum()
        if num_masked == 0:
            continue

        # Forward pass
        optimizer.zero_grad()
        output = model(masked_ids, attention_mask, labels=labels)
        loss = output["loss"]

        # Skip if loss is NaN
        if torch.isnan(loss):
            _LOG.warning("NaN loss detected, skipping batch")
            continue

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        steps_since_log += 1

        # Update progress bar
        if steps_since_log > 0:
            avg_loss = running_loss / steps_since_log
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}", "buf": len(sample_buffer)})

        # Reset running stats periodically
        if (step + 1) % log_every == 0:
            running_loss = 0.0
            steps_since_log = 0

        # Checkpoint
        if checkpoint_dir is not None and (step + 1) % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step + 1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model.config,
                },
                checkpoint_path,
            )
            _LOG.info(f"Saved checkpoint to {checkpoint_path}")

    return model


def classify(
    model: FoundationalTabularTransformer,
    df: pd.DataFrame,
    target_column: str,
    target_classes: list[str],
    columns: Optional[list[str]] = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Classify rows by masking target column and predicting.

    Args:
        model: Trained transformer model
        df: DataFrame with features (target column can have any values)
        target_column: Name of column to predict
        target_classes: List of possible class values
        columns: Feature columns to use (defaults to all categorical)
        device: Device for inference

    Returns:
        DataFrame with probability columns for each class
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    tokenizer = TabularTokenizer(vocab_size=model.config.hash_vocab_size)

    # Determine columns
    if columns is None:
        columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # Ensure target column is in columns
    if target_column not in columns:
        columns = columns + [target_column]

    columns = columns[: model.config.max_columns]

    # Find target column index
    target_idx = columns.index(target_column)

    # Pre-compute hash IDs for target classes
    class_hash_ids = {cls: tokenizer.hash_value(str(cls), target_column) for cls in target_classes}

    # Tokenize data
    value_ids, attention_mask, _ = tokenizer.tokenize_dataframe(df, columns, model.config.max_columns)
    value_ids = value_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Mask target column
    masked_ids, _, target_positions = mask_target_column(value_ids, attention_mask, target_idx)

    # Get predictions
    with torch.no_grad():
        probs = model.predict_masked(masked_ids, attention_mask, target_positions)

    # Extract probabilities for target classes
    probs_np = probs.cpu().numpy()
    result_data = {}
    for cls in target_classes:
        hash_id = class_hash_ids[cls]
        result_data[f"prob_{cls}"] = probs_np[:, hash_id]

    return pd.DataFrame(result_data)


def evaluate_classification(
    model: FoundationalTabularTransformer,
    df: pd.DataFrame,
    target_column: str,
    columns: Optional[list[str]] = None,
    device: Optional[str] = None,
) -> dict[str, float]:
    """
    Evaluate classification accuracy on a labeled dataset.

    Args:
        model: Trained transformer model
        df: DataFrame with features and true labels
        target_column: Name of target column
        columns: Feature columns to use
        device: Device for inference

    Returns:
        Dictionary with accuracy metrics
    """
    # Get unique classes from data
    target_classes = df[target_column].dropna().unique().tolist()

    # Get predictions
    probs_df = classify(model, df, target_column, target_classes, columns, device)

    # Get predicted class (highest probability)
    prob_cols = [f"prob_{cls}" for cls in target_classes]
    predicted_idx = probs_df[prob_cols].values.argmax(axis=1)
    predicted_classes = [target_classes[i] for i in predicted_idx]

    # Calculate accuracy
    true_labels = df[target_column].tolist()
    correct = sum(p == t for p, t in zip(predicted_classes, true_labels))
    accuracy = correct / len(true_labels)

    return {
        "accuracy": accuracy,
        "num_samples": len(true_labels),
        "num_classes": len(target_classes),
    }


def save_model(model: FoundationalTabularTransformer, path: Path):
    """Save model to disk."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,
        },
        path,
    )
    _LOG.info(f"Model saved to {path}")


def load_model(path: Path, device: Optional[str] = None) -> FoundationalTabularTransformer:
    """Load model from disk."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]

    model = FoundationalTabularTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    _LOG.info(f"Model loaded from {path}")
    return model
