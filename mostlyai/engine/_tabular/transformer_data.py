"""
Data processing utilities for the Foundational Tabular Transformer.

Provides tokenization (feature hashing), dataset wrappers, and masking utilities
for Masked Cell Modeling (MCM) pre-training.
"""

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from .transformer import MASK_TOKEN_ID, PAD_TOKEN_ID, SPECIAL_TOKENS_COUNT


class TabularTokenizer:
    """
    Tokenizer for tabular data using feature hashing.

    Converts categorical values to integer IDs by hashing the combination
    of column name and value. This enables handling arbitrary vocabularies
    without pre-building a vocabulary.
    """

    def __init__(self, vocab_size: int = 32768):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Size of hash vocabulary (excluding special tokens)
        """
        self.vocab_size = vocab_size

    def hash_value(self, value: str, column_name: str) -> int:
        """
        Hash a categorical value to an integer ID.

        Args:
            value: The categorical value (will be converted to string)
            column_name: Name of the column

        Returns:
            Hash ID in range [SPECIAL_TOKENS_COUNT, vocab_size + SPECIAL_TOKENS_COUNT)
        """
        combined = f"{column_name}:{value}"
        hash_int = hash(combined)
        return (hash_int % self.vocab_size) + SPECIAL_TOKENS_COUNT

    def tokenize_row(self, row: dict, columns: list[str]) -> list[int]:
        """
        Tokenize a single row.

        Args:
            row: Dictionary mapping column names to values
            columns: Ordered list of column names to include

        Returns:
            List of hashed value IDs
        """
        return [self.hash_value(str(row[col]), col) for col in columns]

    def tokenize_dataframe(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        max_columns: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Tokenize an entire DataFrame.

        Args:
            df: pandas DataFrame to tokenize
            columns: Column names to use (defaults to all columns)
            max_columns: Maximum number of columns to include

        Returns:
            Tuple of:
            - value_ids: (num_rows, num_cols) tensor of hashed IDs
            - attention_mask: (num_rows, num_cols) tensor of True for real values
            - columns: List of column names used
        """
        if columns is None:
            columns = df.columns.tolist()

        # Truncate to max columns
        columns = columns[:max_columns]

        # Tokenize all rows
        value_ids = []
        for _, row in df.iterrows():
            row_ids = self.tokenize_row(row.to_dict(), columns)
            value_ids.append(row_ids)

        value_ids = torch.tensor(value_ids, dtype=torch.long)
        attention_mask = torch.ones_like(value_ids, dtype=torch.bool)

        return value_ids, attention_mask, columns

    def pad_batch(
        self,
        value_ids_list: list[torch.Tensor],
        attention_masks_list: list[torch.Tensor],
        max_length: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a batch of tokenized rows to the same length.

        Args:
            value_ids_list: List of (seq_len,) tensors
            attention_masks_list: List of (seq_len,) tensors
            max_length: Pad to this length (defaults to max in batch)

        Returns:
            Tuple of:
            - value_ids: (batch_size, max_length) padded tensor
            - attention_mask: (batch_size, max_length) mask tensor
        """
        if max_length is None:
            max_length = max(v.size(0) for v in value_ids_list)

        batch_size = len(value_ids_list)
        padded_ids = torch.full((batch_size, max_length), PAD_TOKEN_ID, dtype=torch.long)
        padded_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

        for i, (ids, mask) in enumerate(zip(value_ids_list, attention_masks_list)):
            length = ids.size(0)
            padded_ids[i, :length] = ids
            padded_mask[i, :length] = mask

        return padded_ids, padded_mask


class TabularDataset(Dataset):
    """
    PyTorch Dataset wrapper for tabular data.

    Stores pre-tokenized data for efficient batch loading during training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TabularTokenizer,
        columns: Optional[list[str]] = None,
        max_columns: int = 64,
    ):
        """
        Initialize dataset.

        Args:
            df: pandas DataFrame with categorical data
            tokenizer: TabularTokenizer instance
            columns: Column names to use (defaults to categorical columns)
            max_columns: Maximum number of columns
        """
        self.tokenizer = tokenizer

        # Select categorical columns if not specified
        if columns is None:
            columns = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        self.columns = columns[:max_columns]

        # Tokenize entire dataframe
        self.value_ids, self.attention_mask, self.columns = tokenizer.tokenize_dataframe(
            df, self.columns, max_columns
        )

    def __len__(self) -> int:
        return len(self.value_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "value_ids": self.value_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


class MultiTableDataset(Dataset):
    """
    Dataset that samples rows from multiple tables.

    For pre-training on diverse tabular data (e.g., T4 dataset).
    Each table may have different columns.
    """

    def __init__(
        self,
        tables: list[pd.DataFrame],
        tokenizer: TabularTokenizer,
        max_columns: int = 64,
        rows_per_table: Optional[int] = None,
    ):
        """
        Initialize multi-table dataset.

        Args:
            tables: List of DataFrames
            tokenizer: TabularTokenizer instance
            max_columns: Maximum columns per table
            rows_per_table: Max rows to sample per table (None = all)
        """
        self.tokenizer = tokenizer
        self.max_columns = max_columns
        self.samples = []  # List of (value_ids, attention_mask)

        for df in tables:
            # Select categorical columns
            cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            if len(cat_cols) < 2:
                continue  # Skip tables with too few categorical columns

            cat_cols = cat_cols[:max_columns]

            # Sample rows if needed
            if rows_per_table is not None and len(df) > rows_per_table:
                df = df.sample(n=rows_per_table, random_state=42)

            # Tokenize
            value_ids, attention_mask, _ = tokenizer.tokenize_dataframe(df, cat_cols, max_columns)

            # Add each row as a sample
            for i in range(len(value_ids)):
                self.samples.append((value_ids[i], attention_mask[i]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        value_ids, attention_mask = self.samples[idx]
        return {
            "value_ids": value_ids,
            "attention_mask": attention_mask,
        }


def apply_masking(
    value_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob: float = 0.15,
    random_prob: float = 0.1,
    keep_prob: float = 0.1,
    vocab_size: int = 32768,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply BERT-style masking to value IDs for MCM pre-training.

    Of the selected positions:
    - 80% are replaced with [MASK]
    - 10% are replaced with a random token
    - 10% are kept as original

    Args:
        value_ids: (batch_size, seq_len) original value IDs
        attention_mask: (batch_size, seq_len) True for real tokens
        mask_prob: Probability of selecting a position for masking
        random_prob: Of selected positions, probability of random replacement
        keep_prob: Of selected positions, probability of keeping original
        vocab_size: Size of vocabulary for random replacement

    Returns:
        Tuple of:
        - masked_ids: (batch_size, seq_len) with masking applied
        - labels: (batch_size, seq_len) original IDs at masked positions, -100 elsewhere
    """
    device = value_ids.device
    batch_size, seq_len = value_ids.shape

    # Only mask positions where attention_mask is True (real tokens)
    can_mask = attention_mask.clone()

    # Randomly select positions to mask
    mask_probs = torch.full_like(value_ids, mask_prob, dtype=torch.float)
    mask_probs = mask_probs * can_mask.float()
    selected = torch.bernoulli(mask_probs).bool()

    # Create labels: original values at selected positions, -100 elsewhere
    labels = value_ids.clone()
    labels[~selected] = -100

    # Create masked input
    masked_ids = value_ids.clone()

    # Determine which selected positions get which treatment
    # 80% -> [MASK], 10% -> random, 10% -> keep original
    rand_vals = torch.rand(batch_size, seq_len, device=device)

    # [MASK] replacement (80% of selected)
    mask_replace = selected & (rand_vals < (1 - random_prob - keep_prob))
    masked_ids[mask_replace] = MASK_TOKEN_ID

    # Random replacement (10% of selected)
    random_replace = selected & (rand_vals >= (1 - random_prob - keep_prob)) & (rand_vals < (1 - keep_prob))
    random_tokens = torch.randint(
        SPECIAL_TOKENS_COUNT,
        vocab_size + SPECIAL_TOKENS_COUNT,
        (batch_size, seq_len),
        device=device,
    )
    masked_ids[random_replace] = random_tokens[random_replace]

    # Keep original (10% of selected) - already in masked_ids

    return masked_ids, labels


def mask_target_column(
    value_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_column_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mask a specific column for classification inference.

    Args:
        value_ids: (batch_size, seq_len) original value IDs
        attention_mask: (batch_size, seq_len) attention mask
        target_column_idx: Index of column to mask

    Returns:
        Tuple of:
        - masked_ids: (batch_size, seq_len) with target column masked
        - labels: (batch_size, seq_len) original IDs at target position, -100 elsewhere
        - target_positions: (batch_size,) indices of target column
    """
    batch_size = value_ids.size(0)

    # Create masked input
    masked_ids = value_ids.clone()
    masked_ids[:, target_column_idx] = MASK_TOKEN_ID

    # Create labels
    labels = torch.full_like(value_ids, -100)
    labels[:, target_column_idx] = value_ids[:, target_column_idx]

    # Target positions
    target_positions = torch.full((batch_size,), target_column_idx, dtype=torch.long, device=value_ids.device)

    return masked_ids, labels, target_positions


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads sequences to the same length within a batch.

    Args:
        batch: List of samples from TabularDataset

    Returns:
        Batched and padded tensors
    """
    value_ids_list = [item["value_ids"] for item in batch]
    attention_masks_list = [item["attention_mask"] for item in batch]

    # Find max length in batch
    max_length = max(v.size(0) for v in value_ids_list)

    # Pad
    batch_size = len(batch)
    padded_ids = torch.full((batch_size, max_length), PAD_TOKEN_ID, dtype=torch.long)
    padded_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for i, (ids, mask) in enumerate(zip(value_ids_list, attention_masks_list)):
        length = ids.size(0)
        padded_ids[i, :length] = ids
        padded_mask[i, :length] = mask

    return {
        "value_ids": padded_ids,
        "attention_mask": padded_mask,
    }
