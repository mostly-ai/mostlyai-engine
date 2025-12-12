"""
Foundational Tabular Transformer for categorical data.

A unified architecture for pre-training (MCM) and classification on tabular data.
Uses feature hashing for universal value representation and no positional encoding
(permutation-invariant over columns).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Special token IDs
PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
SPECIAL_TOKENS_COUNT = 2  # PAD and MASK


@dataclass
class TabularTransformerConfig:
    """Configuration for FoundationalTabularTransformer."""

    hash_vocab_size: int = 32768
    max_columns: int = 64
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    intermediate_size: int = 512
    dropout: float = 0.1
    attention_dropout: float = 0.1


class TabularEmbeddings(nn.Module):
    """
    Embedding layer for tabular data.

    Maps hashed value IDs to dense vectors. No positional encoding -
    the model is permutation-invariant over columns.
    """

    def __init__(self, config: TabularTransformerConfig):
        super().__init__()
        self.config = config

        # Value embeddings: hash_vocab_size + special tokens (PAD, MASK)
        total_vocab_size = config.hash_vocab_size + SPECIAL_TOKENS_COUNT
        self.value_embeddings = nn.Embedding(
            num_embeddings=total_vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=PAD_TOKEN_ID,
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, value_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed value IDs.

        Args:
            value_ids: (batch_size, seq_len) tensor of hashed value IDs

        Returns:
            embeddings: (batch_size, seq_len, hidden_size)
        """
        embeddings = self.value_embeddings(value_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class FoundationalTabularTransformer(nn.Module):
    """
    Foundational transformer for tabular data.

    Unified architecture for:
    - Pre-training: Masked Cell Modeling (MCM) - mask random cells, predict original
    - Classification: Mask target column, predict at that position

    Key design choices:
    - Feature hashing: Values are hashed with column name to create unique IDs
    - No positional encoding: Columns are treated as an unordered set
    - No pooling: Predictions are made directly at masked positions
    """

    def __init__(self, config: TabularTransformerConfig):
        super().__init__()
        self.config = config

        # Embeddings (no positional encoding)
        self.embeddings = TabularEmbeddings(config)

        # Transformer encoder with Pre-LN (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Output projection to vocabulary
        total_vocab_size = config.hash_vocab_size + SPECIAL_TOKENS_COUNT
        self.output_projection = nn.Linear(config.hidden_size, total_vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def forward(
        self,
        value_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            value_ids: (batch_size, seq_len) hashed value IDs (with MASK tokens for prediction)
            attention_mask: (batch_size, seq_len) True for real tokens, False for padding
            labels: (batch_size, seq_len) original value IDs for loss computation
                    Use -100 for positions to ignore in loss

        Returns:
            dict with:
            - logits: (batch_size, seq_len, vocab_size) predictions at each position
            - loss: scalar loss (if labels provided)
            - hidden_states: (batch_size, seq_len, hidden_size) encoder outputs
        """
        # Embed inputs
        hidden_states = self.embeddings(value_ids)

        # Create attention mask for transformer (True = masked/ignored)
        # PyTorch transformer expects True for positions to IGNORE
        src_key_padding_mask = ~attention_mask

        # Encode
        hidden_states = self.encoder(
            hidden_states,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_projection(hidden_states)

        output = {
            "logits": logits,
            "hidden_states": hidden_states,
        }

        # Compute loss if labels provided
        if labels is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss

        return output

    def predict_masked(
        self,
        value_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get predictions at masked positions.

        Args:
            value_ids: (batch_size, seq_len) with MASK tokens at positions to predict
            attention_mask: (batch_size, seq_len)
            masked_positions: (batch_size,) indices of masked positions to predict

        Returns:
            probs: (batch_size, vocab_size) probability distribution at masked positions
        """
        output = self.forward(value_ids, attention_mask)
        logits = output["logits"]

        # Gather logits at masked positions
        batch_size = logits.size(0)
        batch_indices = torch.arange(batch_size, device=logits.device)
        masked_logits = logits[batch_indices, masked_positions]  # (batch_size, vocab_size)

        # Convert to probabilities
        probs = F.softmax(masked_logits, dim=-1)
        return probs

    @staticmethod
    def hash_value(value: str, column_name: str, vocab_size: int = 32768) -> int:
        """
        Hash a categorical value to an integer ID.

        Combines column name and value to create unique hashes even for
        same values in different columns.

        Args:
            value: The categorical value (converted to string)
            column_name: Name of the column
            vocab_size: Size of hash vocabulary

        Returns:
            Hash ID in range [SPECIAL_TOKENS_COUNT, vocab_size + SPECIAL_TOKENS_COUNT)
        """
        combined = f"{column_name}:{value}"
        # Use Python's built-in hash (for simplicity; can switch to mmh3 for production)
        hash_int = hash(combined)
        # Map to vocabulary range, offset by special tokens
        return (hash_int % vocab_size) + SPECIAL_TOKENS_COUNT


# Model size presets
MODEL_CONFIGS = {
    "small": TabularTransformerConfig(
        hash_vocab_size=32768,
        max_columns=64,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        intermediate_size=512,
    ),
    "medium": TabularTransformerConfig(
        hash_vocab_size=65536,
        max_columns=128,
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        intermediate_size=1024,
    ),
    "large": TabularTransformerConfig(
        hash_vocab_size=131072,
        max_columns=256,
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        intermediate_size=2048,
    ),
}


def create_model(size: str = "small") -> FoundationalTabularTransformer:
    """
    Create a model with preset configuration.

    Args:
        size: One of "small", "medium", "large"

    Returns:
        Initialized FoundationalTabularTransformer
    """
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {size}. Choose from {list(MODEL_CONFIGS.keys())}")
    config = MODEL_CONFIGS[size]
    return FoundationalTabularTransformer(config)
