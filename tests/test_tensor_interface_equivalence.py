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
Tests for tensor interface training.

These tests verify that training via the tensor interface produces
valid models that can be used for generation.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mostlyai import engine
from mostlyai.engine._common import get_cardinalities
from mostlyai.engine._tabular.tensor_utils import prepare_flat_batch, prepare_sequential_batch
from mostlyai.engine._tabular.training import train as train_tabular, ModelConfig
from mostlyai.engine._workspace import Workspace
from mostlyai.engine.domain import ModelEncodingType, ModelType


def generate_flat_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic flat (cross-sectional) data."""
    np.random.seed(seed)
    return pd.DataFrame({
        "cat_a": np.random.choice(["X", "Y", "Z"], n_rows),
        "cat_b": np.random.choice(["A", "B", "C", "D"], n_rows),
        "num_a": np.random.randn(n_rows) * 10 + 50,
        "num_b": np.random.exponential(5, n_rows),
    })


def generate_sequential_data(n_entities: int, avg_seq_len: int = 10, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic sequential (longitudinal) data."""
    np.random.seed(seed)
    rows = []
    for entity_id in range(n_entities):
        seq_len = max(1, int(np.random.exponential(avg_seq_len)))
        for step in range(seq_len):
            rows.append({
                "entity_id": entity_id,
                "cat_a": np.random.choice(["X", "Y", "Z"]),
                "num_a": np.random.randn() * 10 + 50,
                "step": step,
            })
    return pd.DataFrame(rows)


def extract_model_config_from_workspace(workspace: Workspace, is_sequential: bool) -> ModelConfig:
    """Extract ModelConfig from workspace stats files."""
    tgt_stats = workspace.tgt_stats.read()
    ctx_stats = workspace.ctx_stats.read() if workspace.ctx_stats.path.exists() else {}

    config: ModelConfig = {
        "tgt_cardinalities": get_cardinalities(tgt_stats),
        "ctx_cardinalities": get_cardinalities(ctx_stats) if ctx_stats else {},
        "is_sequential": is_sequential,
        "trn_cnt": tgt_stats["no_of_training_records"],
        "val_cnt": tgt_stats["no_of_validation_records"],
    }

    if is_sequential:
        seq_stats = tgt_stats.get("sequence_length_stats", {})
        config["tgt_seq_len_median"] = seq_stats.get("median", 1)
        config["tgt_seq_len_max"] = seq_stats.get("max", 1)

    return config


def create_tensor_iterator_from_parquet(
    parquet_paths: list[Path],
    is_sequential: bool,
    device: str = "cpu",
):
    """
    Create a tensor iterator from parquet files (mimics what Ray Data would do).

    This loads the encoded parquet files and converts them to tensors.
    """
    import pyarrow.parquet as pq

    # Load all parquet files
    tables = [pq.read_table(p) for p in parquet_paths]
    df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)

    # Convert to columnar format expected by tensor_utils
    batch_dict = {col: df[col].tolist() for col in df.columns}

    # Convert to tensors
    if is_sequential:
        tensors = prepare_sequential_batch(batch_dict, device=device)
    else:
        tensors = prepare_flat_batch(batch_dict, device=device)

    # Return as single-batch iterator (for simplicity in testing)
    # In production, this would be batched appropriately
    return iter([tensors])


class TestFlatDataTraining:
    """Test tensor interface training for flat (cross-sectional) data."""

    @pytest.fixture
    def flat_workspace(self, tmp_path):
        """Create a workspace with flat data."""
        workspace_dir = tmp_path / "flat_ws"
        df = generate_flat_data(1000)

        engine.split(
            workspace_dir=workspace_dir,
            tgt_data=df,
            tgt_encoding_types={
                "cat_a": ModelEncodingType.tabular_categorical,
                "cat_b": ModelEncodingType.tabular_categorical,
                "num_a": ModelEncodingType.tabular_numeric_auto,
                "num_b": ModelEncodingType.tabular_numeric_auto,
            },
            model_type=ModelType.tabular,
        )
        engine.analyze(workspace_dir=workspace_dir, value_protection=False)
        engine.encode(workspace_dir=workspace_dir)

        return workspace_dir

    def test_tensor_format_produces_valid_batches(self, flat_workspace):
        """Verify tensor format from tensor_utils produces valid batches for training."""
        workspace = Workspace(flat_workspace)

        # Load encoded data
        parquet_paths = workspace.encoded_data_trn.fetch_all()
        import pyarrow.parquet as pq
        df = pd.concat([pq.read_table(p).to_pandas() for p in parquet_paths])

        # Get batch via tensor_utils
        batch_dict = {col: df[col].head(32).tolist() for col in df.columns}
        tensor_batch = prepare_flat_batch(batch_dict, device="cpu")

        # Verify batch structure - keys are like "tgt:t0/c0__cat"
        tgt_keys = [k for k in tensor_batch if k.startswith("tgt:")]
        assert len(tgt_keys) > 0, "Expected at least one tgt column"
        for key in tgt_keys:
            assert tensor_batch[key].dim() == 2  # [batch_size, 1]
            assert tensor_batch[key].shape[0] == 32  # batch size
            assert tensor_batch[key].dtype == torch.long

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_training_completes_successfully(self, flat_workspace):
        """
        Train with tensor interface and verify training completes.
        """
        workspace = Workspace(flat_workspace)
        device = "cuda"

        # Extract model config
        model_config = extract_model_config_from_workspace(workspace, is_sequential=False)

        # Create tensor iterators
        trn_tensors = create_tensor_iterator_from_parquet(
            workspace.encoded_data_trn.fetch_all(),
            is_sequential=False,
            device=device,
        )
        val_tensors = create_tensor_iterator_from_parquet(
            workspace.encoded_data_val.fetch_all(),
            is_sequential=False,
            device=device,
        )

        # Train with tensor interface
        train_tabular(
            workspace_dir=flat_workspace,
            train_tensors=trn_tensors,
            val_tensors=val_tensors,
            model_config=model_config,
            max_epochs=3,
            max_training_time=60,
            device=device,
            enable_flexible_generation=False,
        )

        # Verify training completed (weights were saved)
        assert (flat_workspace / "ModelStore" / "model-data" / "model-weights.pt").exists()


class TestSequentialDataTraining:
    """Test tensor interface training for sequential (longitudinal) data."""

    @pytest.fixture
    def sequential_workspace(self, tmp_path):
        """Create a workspace with sequential data."""
        workspace_dir = tmp_path / "seq_ws"
        df = generate_sequential_data(100, avg_seq_len=8)

        engine.split(
            workspace_dir=workspace_dir,
            tgt_data=df,
            tgt_context_key="entity_id",
            tgt_encoding_types={
                "cat_a": ModelEncodingType.tabular_categorical,
                "num_a": ModelEncodingType.tabular_numeric_auto,
                "step": ModelEncodingType.tabular_numeric_discrete,
            },
            model_type=ModelType.tabular,
        )
        engine.analyze(workspace_dir=workspace_dir, value_protection=False)
        engine.encode(workspace_dir=workspace_dir)

        return workspace_dir

    def test_sequential_tensor_format_produces_valid_batches(self, sequential_workspace):
        """Verify tensor format for sequential data produces valid batches."""
        workspace = Workspace(sequential_workspace)

        # Load encoded data
        parquet_paths = workspace.encoded_data_trn.fetch_all()
        import pyarrow.parquet as pq
        df = pd.concat([pq.read_table(p).to_pandas() for p in parquet_paths])

        # Get batch via tensor_utils
        batch_dict = {col: df[col].head(8).tolist() for col in df.columns}
        tensor_batch = prepare_sequential_batch(batch_dict, device="cpu")

        # Verify batch structure - keys are like "tgt:t0/c0__cat"
        tgt_keys = [k for k in tensor_batch if k.startswith("tgt:")]
        assert len(tgt_keys) > 0, "Expected at least one tgt column"
        for key in tgt_keys:
            assert tensor_batch[key].dim() == 3  # [batch_size, seq_len, 1]
            assert tensor_batch[key].dtype == torch.long


class TestModelConfigExtraction:
    """Test ModelConfig extraction from workspace."""

    def test_flat_config_extraction(self, tmp_path):
        """Test config extraction for flat data."""
        workspace_dir = tmp_path / "config_test"
        df = generate_flat_data(500)

        engine.split(
            workspace_dir=workspace_dir,
            tgt_data=df,
            tgt_encoding_types={
                "cat_a": ModelEncodingType.tabular_categorical,
                "cat_b": ModelEncodingType.tabular_categorical,
                "num_a": ModelEncodingType.tabular_numeric_auto,
                "num_b": ModelEncodingType.tabular_numeric_auto,
            },
            model_type=ModelType.tabular,
        )
        engine.analyze(workspace_dir=workspace_dir, value_protection=False)

        workspace = Workspace(workspace_dir)
        config = extract_model_config_from_workspace(workspace, is_sequential=False)

        assert "tgt_cardinalities" in config
        assert len(config["tgt_cardinalities"]) > 0
        assert config["is_sequential"] is False
        assert config["trn_cnt"] > 0
        assert config["val_cnt"] > 0
