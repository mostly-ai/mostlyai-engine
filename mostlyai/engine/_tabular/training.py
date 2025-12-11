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

import logging
import time
import warnings
from collections.abc import Callable, Iterator
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from opacus import GradSampleModule
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

from mostlyai.engine._common import (
    CTXFLT,
    CTXSEQ,
    RIDX_SUB_COLUMN_PREFIX,
    SIDX_SUB_COLUMN_PREFIX,
    SLEN_SUB_COLUMN_PREFIX,
    ProgressCallback,
    ProgressCallbackWrapper,
    get_columns_from_cardinalities,
    get_sub_columns_from_cardinalities,
    get_sub_columns_nested_from_cardinalities,
)
from mostlyai.engine._memory import get_available_ram_for_heuristics
from mostlyai.engine._tabular.argn import (
    FlatModel,
    ModelSize,
    SequentialModel,
    get_model_units,
    get_no_of_model_parameters,
)
from mostlyai.engine._tabular.common import load_model_weights
from mostlyai.engine._training_utils import (
    EarlyStopper,
    ModelCheckpoint,
    ProgressMessage,
    check_early_training_exit,
    gpu_memory_cleanup,
)
from mostlyai.engine._workspace import Workspace, ensure_workspace_dir
from mostlyai.engine.domain import ModelStateStrategy

_LOG = logging.getLogger(__name__)


class ModelConfig(TypedDict, total=False):
    """
    Configuration for training with tensor interface.

    When using train_tensors/val_tensors, this config provides the model
    parameters that would normally be extracted from workspace stats.
    """

    tgt_cardinalities: dict[str, int]
    """Target column cardinalities: {"tgt:col1": 10, "tgt:col2": 100, ...}"""

    ctx_cardinalities: dict[str, int]
    """Context column cardinalities: {"ctxflt/col1": 5, ...}"""

    is_sequential: bool
    """Whether the data contains sequences (longitudinal) or is flat (cross-sectional)"""

    trn_cnt: int
    """Number of training samples"""

    val_cnt: int
    """Number of validation samples"""

    tgt_seq_len_median: int
    """Median sequence length for sequential data (required if is_sequential=True)"""

    tgt_seq_len_max: int
    """Maximum sequence length for sequential data (required if is_sequential=True)"""

    ctx_seq_len_median: int
    """Median context sequence length (optional)"""

    empirical_probs: dict[str, list[float]] | None
    """Empirical probabilities for predictor initialization (optional, improves convergence)"""


##################
### HEURISTICS ###
##################


def _physical_batch_size_heuristic(
    mem_available_gb: float,
    no_of_records: int,
    no_tgt_data_points: int,
    no_ctx_data_points: int,
    no_of_model_params: int,
) -> int:
    """
    Calculate the physical batch size.

    Args:
        mem_available_gb (float): Available memory in GB.
        no_of_records (int): Number of records in the training dataset.
        no_tgt_data_points (int): Number of target data points per sample.
        no_ctx_data_points (int): Number of context data points per sample.
        no_of_model_params (int): Number of model parameters.

    Returns:
        Batch size (int)
    """
    data_points = no_tgt_data_points + no_ctx_data_points
    min_batch_size = 8
    # scale batch_size corresponding to available memory
    if mem_available_gb >= 32:
        mem_scale = 2.0
    elif mem_available_gb >= 8:
        mem_scale = 1.0
    else:
        mem_scale = 0.5
    # set max_batch_size corresponding to available memory, model params and data points
    if no_of_model_params > 1_000_000_000 or data_points > 100_000:
        max_batch_size = int(8 * mem_scale)
    elif no_of_model_params > 100_000_000 or data_points > 10_000:
        max_batch_size = int(32 * mem_scale)
    elif no_of_model_params > 10_000_000 or data_points > 1_000:
        max_batch_size = int(128 * mem_scale)
    elif no_of_model_params > 1_000_000 or data_points > 100:
        max_batch_size = int(512 * mem_scale)
    else:
        max_batch_size = int(2048 * mem_scale)
    # ensure a minimum number of batches to avoid excessive padding
    min_batches = 64
    batch_size = 2 ** int(np.log2(no_of_records / min_batches)) if no_of_records > 0 else min_batch_size
    return int(np.clip(a=batch_size, a_min=min_batch_size, a_max=max_batch_size))


def _learn_rate_heuristic(batch_size: int) -> float:
    learn_rate = np.round(0.001 * np.sqrt(batch_size / 32), 5)
    return learn_rate


####################
### DATA LOADERS ###
####################


class _TensorIteratorWrapper:
    """
    Wrapper for tensor iterators to provide DataLoader-like interface.

    Caches batches on first pass for fast replay on subsequent epochs.
    This avoids repeated data loading overhead and keeps GPU fed continuously.
    """

    def __init__(self, tensor_iterator: Iterator[dict[str, torch.Tensor]]):
        self._source = tensor_iterator
        self._cache: list[dict[str, torch.Tensor]] | None = None

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        if self._cache is not None:
            return iter(self._cache)
        return self._caching_iterator()

    def _caching_iterator(self) -> Iterator[dict[str, torch.Tensor]]:
        cache = []
        for batch in self._source:
            cache.append(batch)
            yield batch
        self._cache = cache

    def __len__(self) -> int:
        if self._cache is not None:
            return len(self._cache)
        raise TypeError("Length unknown until first iteration completes")


#####################
### TRAINING LOOP ###
#####################


class TabularModelCheckpoint(ModelCheckpoint):
    def _save_model_weights(self, model: torch.nn.Module):
        if isinstance(model, GradSampleModule):
            state_dict = model._module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, self.workspace.model_tabular_weights_path)

    def _clear_model_weights(self) -> None:
        self.workspace.model_tabular_weights_path.unlink(missing_ok=True)

    def model_weights_path_exists(self) -> bool:
        return self.workspace.model_tabular_weights_path.exists()


def _calculate_sample_losses(
    model: FlatModel | SequentialModel | GradSampleModule, data: dict[str, torch.Tensor]
) -> torch.Tensor:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook*")
        output, _ = model(data, mode="trn")
    criterion = nn.CrossEntropyLoss(reduction="none")

    tgt_cols = (
        list(model.tgt_cardinalities.keys())
        if not isinstance(model, GradSampleModule)
        else model._module.tgt_cardinalities.keys()
    )
    if isinstance(model, SequentialModel) or (
        isinstance(model, GradSampleModule) and isinstance(model._module, SequentialModel)
    ):
        sidx_cols = {k for k in data if k.startswith(SIDX_SUB_COLUMN_PREFIX)}
        slen_cols = {k for k in data if k.startswith(SLEN_SUB_COLUMN_PREFIX)}
        ridx_cols = [k for k in data if k.startswith(RIDX_SUB_COLUMN_PREFIX)]

        # mask for data columns
        data_mask = torch.zeros_like(data[ridx_cols[0]], dtype=torch.int64)
        for ridx_col in ridx_cols:
            data_mask |= data[ridx_col] != 0  # mask loss for padded rows, which have RIDX=0
        data_mask = data_mask.squeeze(-1)
        # mask for slen columns; only first step is unmasked
        slen_mask = torch.zeros_like(data_mask)
        slen_mask[:, 0] = 1
        # mask for ridx columns: this takes the sequence padding into account to learn the stopping with ridx=0
        ridx_mask = torch.nn.functional.pad(data_mask, (1, 0), value=1)[:, :-1]
        # mask for sidx columns
        sidx_mask = torch.zeros_like(data_mask)

        # calculate per column losses
        losses_by_column = []
        for col in tgt_cols:
            if col in sidx_cols:
                mask = sidx_mask
            elif col in slen_cols:
                mask = slen_mask
            elif col in ridx_cols:
                mask = ridx_mask
            else:
                mask = data_mask
            column_loss = criterion(output[col].transpose(1, 2), data[col].squeeze(2))
            masked_loss = torch.sum(column_loss * mask, dim=1) / torch.clamp(torch.sum(mask), min=1)
            losses_by_column.append(masked_loss)
    else:
        losses_by_column = [criterion(output[col], data[col].squeeze(1)) for col in tgt_cols]
    # sum up column level losses to get overall losses at sample level
    losses = torch.sum(torch.stack(losses_by_column, dim=0), dim=0)
    return losses


# gradient tracking is not needed for validation steps, disable it to save memory
@torch.no_grad()
def _calculate_val_loss(
    model: FlatModel | SequentialModel,
    val_dataloader: "_TensorIteratorWrapper",
) -> float:
    val_sample_losses: list[torch.Tensor] = []
    model.eval()
    for step_data in val_dataloader:
        step_losses = _calculate_sample_losses(model, step_data)
        val_sample_losses.extend(step_losses.detach())
    model.train()
    val_sample_losses: torch.Tensor = torch.stack(val_sample_losses, dim=0)
    val_loss_avg = torch.mean(val_sample_losses).item()
    return val_loss_avg


def _calculate_average_trn_loss(trn_sample_losses: list[torch.Tensor], n: int | None = None) -> float | None:
    if len(trn_sample_losses) == 0:
        return None
    trn_losses_latest = torch.stack(trn_sample_losses, dim=0)
    if n is not None:
        trn_losses_latest = trn_losses_latest[-n:]
    trn_loss = torch.mean(trn_losses_latest).item()
    return trn_loss


################
### TRAINING ###
################


@gpu_memory_cleanup
def train(
    *,
    # Required: tensor data and configuration
    train_tensors: Iterator[dict[str, torch.Tensor]],
    val_tensors: Iterator[dict[str, torch.Tensor]],
    model_config: ModelConfig,
    # Required: output location (for model weights and progress)
    workspace_dir: str | Path,
    # Training parameters
    model: str = "MOSTLY_AI/Medium",
    max_training_time: float = 14400.0,  # 10 days
    max_epochs: float = 100.0,  # 100 epochs
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    enable_flexible_generation: bool = True,
    upload_model_data_callback: Callable | None = None,
    model_state_strategy: ModelStateStrategy | str = ModelStateStrategy.reset,
    device: torch.device | str | None = None,
    update_progress: ProgressCallback | None = None,
):
    """
    Train a TabularARGN model from pre-computed tensor batches.

    This function accepts pre-computed tensor batches directly, eliminating the
    CPU overhead of per-batch data transformation. Use with Ray Data pipelines
    for distributed encoding and streaming tensor delivery.

    Args:
        train_tensors: Iterator yielding training batches as dict[str, torch.Tensor]
        val_tensors: Iterator yielding validation batches as dict[str, torch.Tensor]
        model_config: Model configuration from build_model_config()
        workspace_dir: Directory for model weights and progress tracking
        model: Model size ("MOSTLY_AI/Small", "MOSTLY_AI/Medium", "MOSTLY_AI/Large")
        max_training_time: Maximum training time in minutes (default: 14400 = 10 days)
        max_epochs: Maximum number of epochs (default: 100)
        batch_size: Batch size for heuristics (batches are pre-sized by caller)
        gradient_accumulation_steps: Gradient accumulation steps
        enable_flexible_generation: Enable flexible column order generation
        upload_model_data_callback: Callback for model data upload
        model_state_strategy: How to handle existing model state
        device: Device for training (auto-detected if None)
        update_progress: Progress callback

    Example:
        >>> from mostlyai.engine import train, build_model_config, prepare_flat_batch
        >>>
        >>> # Build config from stats
        >>> config = build_model_config(tgt_stats)
        >>>
        >>> # Create tensor iterator (e.g., from Ray Data)
        >>> def tensor_iter():
        ...     for batch in encoded_dataset.iter_batches():
        ...         yield prepare_flat_batch(batch, device="cuda")
        >>>
        >>> train(
        ...     train_tensors=tensor_iter(),
        ...     val_tensors=val_tensor_iter(),
        ...     model_config=config,
        ...     workspace_dir="/path/to/output",
        ... )
    """
    _LOG.info("TRAIN_TABULAR started")
    t0 = time.time()

    workspace_dir = ensure_workspace_dir(workspace_dir)
    workspace = Workspace(workspace_dir)
    with ProgressCallbackWrapper(
        update_progress, progress_messages_path=workspace.model_progress_messages_path
    ) as progress:
        _LOG.info(f"numpy={version('numpy')}")
        _LOG.info(f"torch={version('torch')}")
        device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
        _LOG.info(f"{device=}")
        torch.set_default_dtype(torch.float32)

        # Extract configuration from model_config
        tgt_cardinalities = model_config["tgt_cardinalities"]
        ctx_cardinalities = model_config.get("ctx_cardinalities", {})
        is_sequential = model_config["is_sequential"]
        trn_cnt = model_config["trn_cnt"]
        val_cnt = model_config["val_cnt"]
        tgt_seq_len_median = model_config.get("tgt_seq_len_median", 1)
        tgt_seq_len_max = model_config.get("tgt_seq_len_max", 1)
        ctx_seq_len_median = model_config.get("ctx_seq_len_median", 1)
        empirical_probs_for_predictor_init = model_config.get("empirical_probs")

        _LOG.info(f"{is_sequential=}")
        tgt_sub_columns = get_sub_columns_from_cardinalities(tgt_cardinalities)
        ctx_nested_sub_columns = get_sub_columns_nested_from_cardinalities(ctx_cardinalities, "processor")
        ctxflt_sub_columns = ctx_nested_sub_columns.get(CTXFLT, [])
        ctxseq_sub_columns = ctx_nested_sub_columns.get(CTXSEQ, [])

        # set defaults
        max_training_time = max(0.0, max_training_time) * 60  # convert to seconds
        _LOG.info(f"{max_training_time=}s")
        max_epochs = max(0.0, max_epochs)
        _LOG.info(f"{max_epochs=}")
        model_sizes = {
            "MOSTLY_AI/Small": ModelSize.S,
            "MOSTLY_AI/Medium": ModelSize.M,
            "MOSTLY_AI/Large": ModelSize.L,
        }
        if model not in model_sizes:
            raise ValueError(f"model {model} not supported")
        model_size = model_sizes[model]
        _LOG.info(f"{model_size=}")
        _LOG.info(f"{enable_flexible_generation=}")
        _LOG.info(f"{model_state_strategy=}")

        # initialize callbacks
        upload_model_data_callback = upload_model_data_callback or (lambda *args, **kwargs: None)

        # early exit if there is not enough data to train the model
        # in such scenario, training model is not created
        # and weights are not stored, so generation must be resilient to that
        if check_early_training_exit(workspace=workspace, trn_cnt=trn_cnt, val_cnt=val_cnt):
            _LOG.warning("not enough data to train model; skipping training")
            return

        # determine column order for training
        if enable_flexible_generation:
            # random column order for each batch
            trn_column_order = None
        else:
            # fixed column order based on cardinalities
            trn_column_order = get_columns_from_cardinalities(tgt_cardinalities)

        # the line below fixes issue with growing epoch time for later epochs
        # https://discuss.pytorch.org/t/training-time-gets-slower-and-slower-on-cpu/145483
        torch.set_flush_denormal(True)

        _LOG.info("create training model")
        model_checkpoint = TabularModelCheckpoint(workspace=workspace)
        argn: SequentialModel | FlatModel
        model_kwargs = {
            "tgt_cardinalities": tgt_cardinalities,
            "ctx_cardinalities": ctx_cardinalities,
            "ctxseq_len_median": ctx_seq_len_median,
            "model_size": model_size,
            "column_order": trn_column_order,
            "device": device,
            "with_dp": False,  # DP training not supported with tensor interface
            "empirical_probs_for_predictor_init": empirical_probs_for_predictor_init,
        }
        if is_sequential:
            argn = SequentialModel(
                **model_kwargs,
                tgt_seq_len_median=tgt_seq_len_median,
                tgt_seq_len_max=tgt_seq_len_max,
            )
        else:
            argn = FlatModel(**model_kwargs)
        _LOG.info(f"model class: {argn.__class__.__name__}")

        if isinstance(model_state_strategy, str):
            model_state_strategy = ModelStateStrategy(model_state_strategy)
        if not model_checkpoint.model_weights_path_exists():
            _LOG.info(f"model weights not found; change strategy from {model_state_strategy} to RESET")
            model_state_strategy = ModelStateStrategy.reset
        _LOG.info(f"{model_state_strategy=}")
        if model_state_strategy in [ModelStateStrategy.resume, ModelStateStrategy.reuse]:
            _LOG.info("load existing model weights")
            torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])
            load_model_weights(model=argn, path=workspace.model_tabular_weights_path, device=device)
        else:  # ModelStateStrategy.reset
            _LOG.info("remove existing checkpoint files")
            model_checkpoint.clear_checkpoint()

        # check how to handle existing progress state
        last_progress_message = progress.get_last_progress_message()
        if last_progress_message and model_state_strategy == ModelStateStrategy.resume:
            epoch = last_progress_message.get("epoch", 0.0)
            steps = last_progress_message.get("steps", 0)
            samples = last_progress_message.get("samples", 0)
            initial_lr = last_progress_message.get("learn_rate", None)
            total_time_init = last_progress_message.get("total_time", 0.0)
        else:
            epoch = 0.0
            steps = 0
            samples = 0
            initial_lr = None
            total_time_init = 0.0
            progress.reset_progress_messages()
        _LOG.info(f"start training progress from {epoch=}, {steps=}")

        argn.to(device)
        no_of_model_params = get_no_of_model_parameters(argn)
        _LOG.info(f"{no_of_model_params=}")

        # persist model configs
        model_units = get_model_units(argn)
        model_configs = {
            "model_id": model,
            "model_units": model_units,
            "enable_flexible_generation": enable_flexible_generation,
        }
        workspace.model_configs.write(model_configs)

        # heuristics for batch_size and for initial learn_rate
        # With tensor interface, batches are pre-sized by the caller
        mem_available_gb = get_available_ram_for_heuristics() / 1024**3
        no_tgt_data_points = len(tgt_cardinalities) * (tgt_seq_len_median if is_sequential else 1)
        no_ctx_data_points = len(ctx_cardinalities)
        if batch_size is None:
            batch_size = _physical_batch_size_heuristic(
                mem_available_gb=mem_available_gb,
                no_of_records=trn_cnt,
                no_tgt_data_points=no_tgt_data_points,
                no_ctx_data_points=no_ctx_data_points,
                no_of_model_params=no_of_model_params,
            )
        if gradient_accumulation_steps is None:
            # for TABULAR the batch size is typically large, so we use step=1 as default
            gradient_accumulation_steps = 1

        # setup params for input pipeline
        batch_size = max(1, min(batch_size, trn_cnt))
        gradient_accumulation_steps = max(1, min(gradient_accumulation_steps, trn_cnt // batch_size))
        trn_batch_size = batch_size * gradient_accumulation_steps
        trn_steps = max(1, trn_cnt // trn_batch_size)
        val_batch_size = max(1, min(batch_size, val_cnt))
        val_steps = max(1, val_cnt // val_batch_size)

        if initial_lr is None:
            initial_lr = _learn_rate_heuristic(trn_batch_size)
        if is_sequential:
            # reduce val_batch_size to reduce padding for validation batches,
            # which speeds up compute, plus it results in a more stable val_loss
            val_batch_size = val_batch_size // 2

        # Create data iterators from provided tensor iterators
        _LOG.info("Using tensor iterators for training data")
        trn_dataloader = _TensorIteratorWrapper(train_tensors)
        val_dataloader = _TensorIteratorWrapper(val_tensors)

        _LOG.info(f"{trn_cnt=}, {val_cnt=}")
        _LOG.info(f"{len(tgt_sub_columns)=}, {len(ctxflt_sub_columns)=}, {len(ctxseq_sub_columns)=}")
        if len(tgt_cardinalities) > 0:
            tgt_cardinalities_deciles = list(
                np.quantile(
                    list(tgt_cardinalities.values()),
                    np.arange(0, 1.1, 0.1),
                    method="lower",
                )
            )
            _LOG.info(f"{tgt_cardinalities_deciles=}")
        if len(ctx_cardinalities) > 0:
            ctx_cardinalities_deciles = list(
                np.quantile(
                    list(ctx_cardinalities.values()),
                    np.arange(0, 1.1, 0.1),
                    method="lower",
                )
            )
            _LOG.info(f"{ctx_cardinalities_deciles=}")
        _LOG.info(f"{trn_batch_size=}, {val_batch_size=}")
        _LOG.info(f"{trn_steps=}, {val_steps=}")
        _LOG.info(f"{batch_size=}, {gradient_accumulation_steps=}, {initial_lr=}")

        early_stopper = EarlyStopper(val_loss_patience=4)
        optimizer = torch.optim.AdamW(params=argn.parameters(), lr=initial_lr)
        lr_scheduler: LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=2,
            min_lr=0.1 * initial_lr,
            # threshold=0,  # if we prefer to completely mimic the behavior of previous implementation
        )
        if (
            model_state_strategy == ModelStateStrategy.resume
            and model_checkpoint.optimizer_and_lr_scheduler_paths_exist()
        ):
            # restore the full states of optimizer and lr_scheduler when possible
            # otherwise, only the learning rate from the last progress message will be restored
            _LOG.info("restore optimizer and LR scheduler states")
            optimizer.load_state_dict(
                torch.load(workspace.model_optimizer_path, map_location=device, weights_only=True)
            )
            lr_scheduler.load_state_dict(
                torch.load(workspace.model_lr_scheduler_path, map_location=device, weights_only=True)
            )

        if device.type == "cuda":
            # this can help accelerate GPU compute
            torch.backends.cudnn.benchmark = True

        progress_message = None
        start_trn_time = time.time()
        last_msg_time = time.time()
        trn_data_iter = iter(trn_dataloader)
        trn_sample_losses: list[torch.Tensor] = []
        do_stop = False
        current_lr = initial_lr
        # infinite loop over training steps, until we decide to stop
        # either because of max_epochs, max_training_time or early_stopping
        while not do_stop:
            is_checkpoint = 0
            steps += 1
            epoch = steps / trn_steps

            stop_accumulating_grads = False
            accumulated_steps = 0
            optimizer.zero_grad(set_to_none=True)
            while not stop_accumulating_grads:
                # fetch next training (micro)batch
                try:
                    step_data = next(trn_data_iter)
                except StopIteration:
                    trn_data_iter = iter(trn_dataloader)
                    step_data = next(trn_data_iter)
                # forward pass + calculate sample losses
                step_losses = _calculate_sample_losses(argn, step_data)
                # FIXME in sequential case, this is an approximation, it should be divided by total sum of masks in the
                #  entire batch to get the average loss per sample. Less importantly the final sample may be smaller
                #  than the batch size in both flat and sequential case.
                # calculate total step loss
                step_loss = torch.mean(step_losses) / gradient_accumulation_steps
                # backward pass
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook*")
                    step_loss.backward()
                accumulated_steps += 1
                samples += step_losses.shape[0]
                if accumulated_steps % gradient_accumulation_steps == 0:
                    # update parameters with accumulated gradients
                    optimizer.step()
                    stop_accumulating_grads = True
                # detach losses from the graph
                step_losses = step_losses.detach()
                trn_sample_losses.extend(step_losses)

            current_lr = optimizer.param_groups[0][
                "lr"
            ]  # currently assume that we have the same lr for all param groups

            # only the scheduling for ReduceLROnPlateau is postponed until the metric becomes available
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step()

            # do validation
            do_validation = on_epoch_end = epoch.is_integer()
            if do_validation:
                # calculate val loss and trn loss
                val_loss = _calculate_val_loss(model=argn, val_dataloader=val_dataloader)
                # handle scenario where model training ran into numeric instability
                if pd.isna(val_loss):
                    _LOG.warning("validation loss is not available - reset model weights to last checkpoint")
                    load_model_weights(
                        model=argn,
                        path=workspace.model_tabular_weights_path,
                        device=device,
                    )
                trn_loss = _calculate_average_trn_loss(trn_sample_losses)
                # save model weights with the best validation loss
                is_checkpoint = model_checkpoint.save_checkpoint_if_best(
                    val_loss=val_loss,
                    model=argn,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    dp_accountant=None,
                )
                # gather message for progress with checkpoint info
                progress_message = ProgressMessage(
                    epoch=epoch,
                    is_checkpoint=is_checkpoint,
                    steps=steps,
                    samples=samples,
                    trn_loss=trn_loss,
                    val_loss=val_loss,
                    total_time=total_time_init + time.time() - start_trn_time,
                    learn_rate=current_lr,
                    dp_eps=None,
                    dp_delta=None,
                )
                # check for early stopping
                do_stop = early_stopper(val_loss=val_loss)
                # scheduling for ReduceLROnPlateau
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(metrics=val_loss)

            # log progress, either by time or by steps, whatever is shorter
            elapsed_training_time = time.time() - start_trn_time
            estimated_time_for_max_epochs = (max_epochs * trn_steps) * (elapsed_training_time / steps)
            if max_training_time < estimated_time_for_max_epochs:
                # use seconds for measuring progress against max_training_time
                progress_total_count = max_training_time
                progress_processed = elapsed_training_time
            else:
                # use steps for measuring progress against max_epochs
                progress_total_count = max_epochs * trn_steps
                progress_processed = steps
            # send a progress message at least every X minutes
            last_msg_interval = 5 * 60
            last_msg_elapsed = time.time() - last_msg_time
            if progress_message is None and (last_msg_elapsed > last_msg_interval or steps == 1):
                # running mean loss of the most recent training samples
                running_trn_loss = _calculate_average_trn_loss(trn_sample_losses, n=val_steps * val_batch_size)
                progress_message = ProgressMessage(
                    epoch=epoch,
                    is_checkpoint=is_checkpoint,
                    steps=steps,
                    samples=samples,
                    trn_loss=running_trn_loss,
                    val_loss=None,
                    total_time=total_time_init + time.time() - start_trn_time,
                    learn_rate=current_lr,
                    dp_eps=None,
                    dp_delta=None,
                )
            if progress_message:
                last_msg_time = time.time()
            # send progress update
            res = progress.update(
                completed=int(progress_processed),
                total=int(progress_total_count),
                message=progress_message,
            )
            if do_validation:
                upload_model_data_callback()
            progress_message = None
            if (res or {}).get("stopExecution", False):
                _LOG.info("received STOP EXECUTION signal")
                do_stop = True

            if on_epoch_end:
                trn_sample_losses = []

            # check for max_epochs
            if epoch > max_epochs:
                do_stop = True

            # check for max_training_time
            total_training_time = total_time_init + time.time() - start_trn_time
            if total_training_time > max_training_time:
                do_stop = True

        # no checkpoint is saved yet because the training stopped before the first epoch ended
        if not model_checkpoint.has_saved_once():
            _LOG.info("saving model weights, as none were saved so far")
            model_checkpoint.save_checkpoint(
                model=argn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                dp_accountant=privacy_engine.accountant if with_dp else None,
            )
            if total_training_time > max_training_time:
                _LOG.info("skip validation loss calculation due to time-capped early stopping")
                val_loss = None
            else:
                _LOG.info("calculate validation loss")
                val_loss = _calculate_val_loss(model=argn, val_dataloader=val_dataloader)
            dp_total_epsilon = (
                privacy_engine.get_epsilon(dp_total_delta) + dp_value_protection_epsilon if with_dp else None
            )
            # send a final message to inform how far we've progressed
            trn_loss = _calculate_average_trn_loss(trn_sample_losses)
            progress_message = ProgressMessage(
                epoch=epoch,
                is_checkpoint=1,
                steps=steps,
                samples=samples,
                trn_loss=trn_loss,
                val_loss=val_loss,
                total_time=total_training_time,
                learn_rate=current_lr,
                dp_eps=dp_total_epsilon,
                dp_delta=dp_total_delta,
            )
            progress.update(completed=steps, total=steps, message=progress_message)
            # ensure everything gets uploaded
            upload_model_data_callback()

    _LOG.info(f"TRAIN_TABULAR finished in {time.time() - t0:.2f}s")
