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

"""Tests for tensor_utils - tensor conversion functions."""

import numpy as np
import pandas as pd
import pytest
import torch

from mostlyai.engine._tabular.tensor_utils import (
    prepare_flat_batch,
    prepare_sequential_batch,
    slice_sequences,
)


class TestPrepareFlatBatch:
    """Test prepare_flat_batch for non-sequential data."""

    def test_basic_flat_batch(self):
        """Test basic flat batch conversion."""
        batch = {
            "tgt:col1": [1, 2, 3, 4],
            "tgt:col2": [10, 20, 30, 40],
            "ctxflt/ctx1": [100, 200, 300, 400],
        }
        tensors = prepare_flat_batch(batch, device="cpu")

        assert set(tensors.keys()) == {"tgt:col1", "tgt:col2", "ctxflt/ctx1"}
        assert tensors["tgt:col1"].shape == (4, 1)
        assert tensors["tgt:col1"].dtype == torch.int64
        assert tensors["tgt:col1"].tolist() == [[1], [2], [3], [4]]

    def test_multiple_columns(self):
        """Test with multiple tgt and ctx columns."""
        samples = [
            {"tgt:a": 1, "tgt:b": 10, "ctxflt/c": 100},
            {"tgt:a": 2, "tgt:b": 20, "ctxflt/c": 200},
            {"tgt:a": 3, "tgt:b": 30, "ctxflt/c": 300},
        ]

        # Convert to columnar format
        columnar = {
            "tgt:a": [s["tgt:a"] for s in samples],
            "tgt:b": [s["tgt:b"] for s in samples],
            "ctxflt/c": [s["ctxflt/c"] for s in samples],
        }
        tensor_result = prepare_flat_batch(columnar, device="cpu")

        # Verify structure
        assert set(tensor_result.keys()) == {"tgt:a", "tgt:b", "ctxflt/c"}
        assert tensor_result["tgt:a"].shape == (3, 1)
        assert tensor_result["tgt:b"].tolist() == [[10], [20], [30]]
        assert tensor_result["ctxflt/c"].tolist() == [[100], [200], [300]]

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        batch = {
            "tgt:col1": np.array([1, 2, 3]),
            "ctxflt/col2": np.array([4, 5, 6]),
        }
        tensors = prepare_flat_batch(batch)
        assert tensors["tgt:col1"].shape == (3, 1)
        assert tensors["ctxflt/col2"].tolist() == [[4], [5], [6]]


class TestPrepareSequentialBatch:
    """Test prepare_sequential_batch for sequential data."""

    def test_basic_sequential_batch(self):
        """Test basic sequential batch conversion with padding."""
        batch = {
            "tgt:col1": [[1, 2, 3], [4, 5], [6]],  # Variable length sequences
            "ctxflt/ctx1": [100, 200, 300],  # Flat context
        }
        tensors = prepare_sequential_batch(batch, device="cpu")

        # Target should be padded to longest sequence (3)
        assert tensors["tgt:col1"].shape == (3, 3, 1)
        assert tensors["tgt:col1"].dtype == torch.int64

        # First sequence: [1, 2, 3]
        assert tensors["tgt:col1"][0, :, 0].tolist() == [1, 2, 3]
        # Second sequence: [4, 5, 0] (padded)
        assert tensors["tgt:col1"][1, :, 0].tolist() == [4, 5, 0]
        # Third sequence: [6, 0, 0] (padded)
        assert tensors["tgt:col1"][2, :, 0].tolist() == [6, 0, 0]

        # Context should be flat
        assert tensors["ctxflt/ctx1"].shape == (3, 1)

    def test_sequential_with_multiple_columns(self):
        """Test sequential conversion with multiple columns."""
        samples = [
            {"tgt:a": [1, 2, 3], "tgt:b": [10, 20, 30], "ctxflt/c": 100},
            {"tgt:a": [4, 5], "tgt:b": [40, 50], "ctxflt/c": 200},
            {"tgt:a": [6, 7, 8, 9], "tgt:b": [60, 70, 80, 90], "ctxflt/c": 300},
        ]

        # Convert to columnar format
        columnar = {
            "tgt:a": [s["tgt:a"] for s in samples],
            "tgt:b": [s["tgt:b"] for s in samples],
            "ctxflt/c": [s["ctxflt/c"] for s in samples],
        }
        tensor_result = prepare_sequential_batch(columnar, device="cpu")

        # Verify shapes - max seq length is 4
        assert set(tensor_result.keys()) == {"tgt:a", "tgt:b", "ctxflt/c"}
        assert tensor_result["tgt:a"].shape == (3, 4, 1)
        assert tensor_result["tgt:b"].shape == (3, 4, 1)
        assert tensor_result["ctxflt/c"].shape == (3, 1)

        # Verify padding
        assert tensor_result["tgt:a"][0, :, 0].tolist() == [1, 2, 3, 0]  # padded
        assert tensor_result["tgt:a"][1, :, 0].tolist() == [4, 5, 0, 0]  # padded
        assert tensor_result["tgt:a"][2, :, 0].tolist() == [6, 7, 8, 9]  # full length

    def test_max_seq_len_truncation(self):
        """Test explicit max_seq_len parameter truncates sequences."""
        batch = {
            "tgt:col1": [[1, 2, 3, 4, 5], [6, 7]],
        }
        tensors = prepare_sequential_batch(batch, max_seq_len=3, device="cpu")

        # Should be truncated to 3
        assert tensors["tgt:col1"].shape == (2, 3, 1)
        assert tensors["tgt:col1"][0, :, 0].tolist() == [1, 2, 3]
        assert tensors["tgt:col1"][1, :, 0].tolist() == [6, 7, 0]

    def test_ctxseq_nested_tensor(self):
        """Test sequential context produces nested tensors."""
        batch = {
            "tgt:col1": [[1, 2], [3, 4]],
            "ctxseq/ctx1": [[10, 20, 30], [40]],  # Variable length context
        }
        tensors = prepare_sequential_batch(batch, device="cpu")

        # ctxseq should be nested tensor
        assert tensors["ctxseq/ctx1"].is_nested
        # Nested tensors don't support len() directly, use unbind to check elements
        unbound = tensors["ctxseq/ctx1"].unbind()
        assert len(unbound) == 2


class TestSliceSequences:
    """Test slice_sequences function."""

    def test_no_slicing_when_short(self):
        """Test that short sequences pass through unchanged."""
        batch = {
            "tgt:col1": torch.tensor([[[1], [2], [3]], [[4], [5], [6]]]),  # (2, 3, 1)
            "ctxflt/ctx1": torch.tensor([[100], [200]]),  # (2, 1)
        }
        result = slice_sequences(batch, max_window=10)

        # Should be unchanged
        assert torch.equal(result["tgt:col1"], batch["tgt:col1"])
        assert torch.equal(result["ctxflt/ctx1"], batch["ctxflt/ctx1"])

    def test_slicing_with_start_strategy(self):
        """Test start strategy takes beginning of sequence."""
        # Sequence of length 10, window of 3 (+1 for padding = 4)
        seq = torch.arange(10).unsqueeze(0).unsqueeze(-1)  # (1, 10, 1)
        batch = {"tgt:col1": seq}

        result = slice_sequences(batch, max_window=3, strategy="start")

        # Should take first 4 elements (max_window + 1)
        assert result["tgt:col1"].shape == (1, 4, 1)
        assert result["tgt:col1"][0, :, 0].tolist() == [0, 1, 2, 3]

    def test_slicing_with_end_strategy(self):
        """Test end strategy takes end of sequence."""
        seq = torch.arange(10).unsqueeze(0).unsqueeze(-1)  # (1, 10, 1)
        batch = {"tgt:col1": seq}

        result = slice_sequences(batch, max_window=3, strategy="end")

        # Should take last 4 elements
        assert result["tgt:col1"].shape == (1, 4, 1)
        assert result["tgt:col1"][0, :, 0].tolist() == [6, 7, 8, 9]

    def test_slicing_preserves_context(self):
        """Test that context columns are not sliced."""
        batch = {
            "tgt:col1": torch.arange(10).unsqueeze(0).unsqueeze(-1),
            "ctxflt/ctx1": torch.tensor([[999]]),
        }
        result = slice_sequences(batch, max_window=3, strategy="start")

        # Context should be unchanged
        assert torch.equal(result["ctxflt/ctx1"], batch["ctxflt/ctx1"])

    def test_random_strategy_produces_valid_windows(self):
        """Test random strategy produces windows within valid range."""
        seq = torch.arange(20).unsqueeze(0).unsqueeze(-1)  # (1, 20, 1)
        batch = {"tgt:col1": seq}

        # Run multiple times to test randomness
        for _ in range(10):
            result = slice_sequences(batch, max_window=5, strategy="random")

            # Window should be 6 (5 + 1 for padding)
            assert result["tgt:col1"].shape == (1, 6, 1)

            # Values should be consecutive
            values = result["tgt:col1"][0, :, 0].tolist()
            assert values == list(range(values[0], values[0] + 6))

            # Window should be within original sequence
            assert values[0] >= 0
            assert values[-1] <= 19


class TestDevicePlacement:
    """Test device placement works correctly."""

    def test_cpu_device(self):
        """Test CPU device placement."""
        batch = {"tgt:col1": [1, 2, 3]}
        tensors = prepare_flat_batch(batch, device="cpu")
        assert tensors["tgt:col1"].device == torch.device("cpu")

    def test_string_device(self):
        """Test string device specification."""
        batch = {"tgt:col1": [1, 2, 3]}
        tensors = prepare_flat_batch(batch, device="cpu")
        assert tensors["tgt:col1"].device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test CUDA device placement."""
        batch = {"tgt:col1": [1, 2, 3]}
        tensors = prepare_flat_batch(batch, device="cuda")
        assert tensors["tgt:col1"].device.type == "cuda"


class TestBuildModelConfig:
    """Test build_model_config functions."""

    def test_build_model_config_flat(self):
        """Test building config for flat data."""
        from mostlyai.engine._tabular.tensor_utils import build_model_config

        tgt_stats = {
            "is_sequential": False,
            "no_of_training_records": 1000,
            "no_of_validation_records": 200,
            "columns": {
                "col1": {"cardinality": {"cat": 10}},
                "col2": {"cardinality": {"bin": 50}},
            },
        }

        config = build_model_config(tgt_stats)

        assert config["is_sequential"] is False
        assert config["trn_cnt"] == 1000
        assert config["val_cnt"] == 200
        assert "tgt_cardinalities" in config
        assert config["ctx_cardinalities"] == {}

    def test_build_model_config_sequential(self):
        """Test building config for sequential data."""
        from mostlyai.engine._tabular.tensor_utils import build_model_config

        tgt_stats = {
            "is_sequential": True,
            "no_of_training_records": 500,
            "no_of_validation_records": 100,
            "seq_len": {
                "min": 1,
                "median": 15,
                "max": 100,
            },
            "columns": {},
        }

        config = build_model_config(tgt_stats)

        assert config["is_sequential"] is True
        assert config["tgt_seq_len_median"] == 15
        assert config["tgt_seq_len_max"] == 100

    def test_build_model_config_with_context(self):
        """Test building config with context stats."""
        from mostlyai.engine._tabular.tensor_utils import build_model_config

        tgt_stats = {
            "is_sequential": False,
            "no_of_training_records": 800,
            "no_of_validation_records": 200,
            "columns": {},
        }
        # Context stats format matches what get_cardinalities expects
        ctx_stats = {
            "columns": {},  # Real stats would have columns with proper cardinalities structure
        }

        config = build_model_config(tgt_stats, ctx_stats)

        # Just verify the structure is correct - real integration tests verify values
        assert "ctx_cardinalities" in config
        assert isinstance(config["ctx_cardinalities"], dict)

    def test_build_model_config_with_empirical_probs(self):
        """Test building config with pre-computed empirical probs."""
        from mostlyai.engine._tabular.tensor_utils import build_model_config

        tgt_stats = {
            "is_sequential": False,
            "no_of_training_records": 100,
            "no_of_validation_records": 20,
            "columns": {},
        }
        probs = {"tgt:col1:cat": [0.3, 0.5, 0.2]}

        config = build_model_config(tgt_stats, empirical_probs=probs)

        assert config["empirical_probs"] == probs

    def test_build_model_config_from_workspace(self, tmp_path):
        """Test building config from a real workspace after analyze()."""
        from mostlyai.engine._tabular.tensor_utils import build_model_config_from_workspace
        from mostlyai import engine
        from mostlyai.engine.domain import ModelEncodingType, ModelType

        # Create a small workspace
        workspace_dir = tmp_path / "ws"
        df = pd.DataFrame({
            "cat_col": np.random.choice(["A", "B", "C"], 100),
            "num_col": np.random.randn(100) * 10,
        })

        engine.split(
            workspace_dir=workspace_dir,
            tgt_data=df,
            tgt_encoding_types={
                "cat_col": ModelEncodingType.tabular_categorical,
                "num_col": ModelEncodingType.tabular_numeric_auto,
            },
            model_type=ModelType.tabular,
        )
        engine.analyze(workspace_dir=workspace_dir, value_protection=False)

        # Build config from workspace
        config = build_model_config_from_workspace(workspace_dir)

        # Verify essential fields
        assert config["is_sequential"] is False
        assert config["trn_cnt"] > 0
        assert config["val_cnt"] > 0
        assert "tgt_cardinalities" in config
        assert len(config["tgt_cardinalities"]) > 0


class TestEncodeBatch:
    """Test encode_batch function for Ray Data integration."""

    def test_encode_batch_flat(self, tmp_path):
        """Test encoding a flat batch returns dict of numpy arrays."""
        from mostlyai.engine._tabular.tensor_utils import encode_batch
        from mostlyai import engine
        from mostlyai.engine.domain import ModelEncodingType, ModelType
        from mostlyai.engine._workspace import Workspace

        # Create workspace with stats
        workspace_dir = tmp_path / "ws"
        df = pd.DataFrame({
            "cat_col": np.random.choice(["A", "B", "C"], 100),
            "num_col": np.random.randn(100) * 10,
        })

        engine.split(
            workspace_dir=workspace_dir,
            tgt_data=df,
            tgt_encoding_types={
                "cat_col": ModelEncodingType.tabular_categorical,
                "num_col": ModelEncodingType.tabular_numeric_auto,
            },
            model_type=ModelType.tabular,
        )
        engine.analyze(workspace_dir=workspace_dir, value_protection=False)

        # Get stats
        workspace = Workspace(workspace_dir)
        tgt_stats = workspace.tgt_stats.read()

        # Encode a batch
        batch_df = pd.DataFrame({
            "cat_col": ["A", "B", "C", "A"],
            "num_col": [1.0, 2.0, 3.0, 4.0],
        })
        result = encode_batch(batch_df, tgt_stats)

        # Verify result is dict of numpy arrays
        assert isinstance(result, dict)
        assert len(result) > 0
        for key, val in result.items():
            assert isinstance(val, np.ndarray), f"Expected numpy array for {key}"

    def test_encode_batch_to_tensor_pipeline(self, tmp_path):
        """Test full pipeline: encode_batch -> prepare_flat_batch."""
        from mostlyai.engine._tabular.tensor_utils import encode_batch, prepare_flat_batch
        from mostlyai import engine
        from mostlyai.engine.domain import ModelEncodingType, ModelType
        from mostlyai.engine._workspace import Workspace

        # Create workspace with stats
        workspace_dir = tmp_path / "ws"
        df = pd.DataFrame({
            "cat_col": np.random.choice(["A", "B", "C"], 100),
            "num_col": np.random.randn(100) * 10,
        })

        engine.split(
            workspace_dir=workspace_dir,
            tgt_data=df,
            tgt_encoding_types={
                "cat_col": ModelEncodingType.tabular_categorical,
                "num_col": ModelEncodingType.tabular_numeric_auto,
            },
            model_type=ModelType.tabular,
        )
        engine.analyze(workspace_dir=workspace_dir, value_protection=False)

        workspace = Workspace(workspace_dir)
        tgt_stats = workspace.tgt_stats.read()

        # Encode batch
        batch_df = pd.DataFrame({
            "cat_col": ["A", "B", "C"],
            "num_col": [1.0, 2.0, 3.0],
        })
        encoded = encode_batch(batch_df, tgt_stats)

        # Convert to tensors
        tensors = prepare_flat_batch(encoded, device="cpu")

        # Verify tensors
        assert isinstance(tensors, dict)
        for key, val in tensors.items():
            assert isinstance(val, torch.Tensor)
            assert val.shape[0] == 3  # batch size
