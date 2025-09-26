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

import abc
import functools
import gc
import logging
import time

import pandas as pd
import torch
from opacus.accountants import IAccountant
from pydantic import BaseModel, Field, field_validator

from mostlyai.engine._workspace import Workspace

_LOG = logging.getLogger(__name__)


class ProgressMessage(BaseModel, extra="allow"):
    epoch: float | None = Field(None, description="Current epoch number")
    is_checkpoint: bool | int | None = Field(0, description="Whether this progress is a checkpoint")
    steps: int | None = Field(None, description="Number of processed steps")
    samples: int | None = Field(None, description="Number of processed samples")
    trn_loss: float | None = Field(None, description="Training loss")
    val_loss: float | None = Field(None, description="Validation loss")
    total_time: float | None = Field(None, description="Elapsed total time (s)")
    learn_rate: float | None = Field(None, description="Learning rate")
    dp_eps: float | None = Field(None, description="Differential privacy epsilon")
    dp_delta: float | None = Field(None, description="Differential privacy delta")

    @field_validator("epoch", "trn_loss", "val_loss", "learn_rate", "total_time", "dp_eps", "dp_delta")
    @classmethod
    def round_float(cls, v, info) -> float:
        field_decimal_places = {
            "epoch": 2,
            "trn_loss": 4,
            "val_loss": 4,
            "learn_rate": 6,
            "total_time": 1,
            "dp_eps": 2,
            "dp_delta": 8,
        }
        if isinstance(v, float) and info.field_name in field_decimal_places:
            return round(v, field_decimal_places[info.field_name])
        return v

    @field_validator("is_checkpoint")
    @classmethod
    def cast_to_int(cls, v) -> int:
        return int(v)


class EarlyStopper:
    """
    Stop training when val_loss stopped improving for a while
    """

    def __init__(self, val_loss_patience: int) -> None:
        self.val_loss_patience = val_loss_patience
        self.best_loss = float("inf")
        self.val_loss_cnt = 0

    def __call__(self, val_loss: float) -> bool:
        do_stop = False
        # check val_loss
        if not pd.isna(val_loss) and val_loss < self.best_loss:
            # remember best val_loss
            self.best_loss = val_loss
            # reset counter
            self.val_loss_cnt = 0
        else:
            self.val_loss_cnt += 1
            if self.val_loss_cnt > self.val_loss_patience:
                _LOG.info("early stopping: val_loss stopped improving")
                do_stop = True
        return do_stop


class ModelCheckpoint(abc.ABC):
    """
    Save model weights for best model.
    """

    def __init__(self, workspace: Workspace, initial_best_val_loss: float = float("inf")) -> None:
        self.workspace = workspace
        self.best_val_loss = initial_best_val_loss
        self.last_save_time = time.time()
        self.save_count = 0

    def optimizer_and_lr_scheduler_paths_exist(self) -> bool:
        return self.workspace.model_optimizer_path.exists() and self.workspace.model_lr_scheduler_path.exists()

    @abc.abstractmethod
    def model_weights_path_exists(self) -> None:
        pass

    def clear_checkpoint(self):
        self.workspace.model_optimizer_path.unlink(missing_ok=True)
        self.workspace.model_lr_scheduler_path.unlink(missing_ok=True)
        self._clear_model_weights()

    def save_checkpoint_if_best(
        self,
        val_loss: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        dp_accountant: IAccountant | None = None,
    ) -> bool:
        # save model weights if validation loss has improved
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(model, optimizer, lr_scheduler, dp_accountant)
            return True
        else:
            return False

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        dp_accountant: IAccountant | None = None,
    ) -> None:
        if optimizer is not None and lr_scheduler is not None:
            torch.save(optimizer.state_dict(), self.workspace.model_optimizer_path)
            torch.save(lr_scheduler.state_dict(), self.workspace.model_lr_scheduler_path)
        if dp_accountant is not None:
            torch.save(dp_accountant.state_dict(), self.workspace.model_dp_accountant_path)
        self._save_model_weights(model)
        self.last_save_time = time.time()
        self.save_count += 1

    def has_saved_once(self) -> bool:
        return self.save_count > 0

    @abc.abstractmethod
    def _save_model_weights(self, model: torch.nn.Module) -> None:
        pass

    @abc.abstractmethod
    def _clear_model_weights(self) -> None:
        pass


def check_early_training_exit(workspace: Workspace, trn_cnt: int, val_cnt: int) -> bool:
    trn_files = workspace.encoded_data_trn.fetch_all()
    val_files = workspace.encoded_data_val.fetch_all()
    return any((len(trn_files) == 0, len(val_files) == 0, trn_cnt == 0, val_cnt == 0))


def gpu_memory_cleanup(func):
    """Decorator to clean up GPU memory after function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            for _ in range(5):
                gc.collect()
            torch.cuda.empty_cache()

    return wrapper
