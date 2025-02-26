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

import json
import logging
import math
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from collections.abc import Callable
import warnings

from importlib.metadata import version
import pandas as pd
import numpy as np
import torch
from opacus.grad_sample import register_grad_sampler
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LRScheduler

from opacus import PrivacyEngine, GradSampleModule
from opacus.accountants import PRVAccountant, RDPAccountant, GaussianAccountant
from opacus.utils.batch_memory_manager import wrap_data_loader

from torch.utils.data import DataLoader

from mostlyai.engine._common import (
    ProgressCallback,
    ProgressCallbackWrapper,
    TABLE_COLUMN_INFIX,
)
from mostlyai.engine._language.common import (
    is_bf16_supported,
    load_base_model_and_config,
    estimate_max_tokens,
    MAX_LENGTH,
)
from mostlyai.engine._language.encoding import row_to_json
from mostlyai.engine._language.tokenizer_utils import (
    train_tokenizer,
    MostlyDataCollatorForLanguageModeling,
    tokenize_fn,
)
from mostlyai.engine._training_utils import (
    check_early_training_exit,
    EarlyStopper,
    ModelCheckpoint,
    ProgressMessage,
)
from mostlyai.engine.domain import ModelStateStrategy, DifferentialPrivacyConfig
from mostlyai.engine._workspace import Workspace, ensure_workspace_dir
from datasets import load_dataset, DatasetDict, Dataset, disable_progress_bar
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
)
from mostlyai.engine._language.lstm import LSTMFromScratchLMHeadModel, LSTMFromScratchConfig
from peft import LoraConfig, PeftModel

# TODO: multi-gpu
from mostlyai.engine._common import ddp_setup
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group, all_reduce
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.data_loader import DPDataLoader

_LOG = logging.getLogger(__name__)


#####################
### TRAINING LOOP ###
#####################


def _training_batch_size_heuristic(no_of_records: int, no_of_model_params: int, max_tokens: int) -> tuple[int, int]:
    """
    Calculate the physical batch size and gradient accumulation steps.

    Args:
        no_of_records (int): Number of records in the training dataset.
        no_of_model_params (int): Number of model parameters.
        max_tokens (int): Maximum number of tokens that are in the training dataset.

    Returns:
        Tuple[int, int]: A tuple containing:
            - Batch size (int)
            - Gradient accumulation steps (int)
    """

    if no_of_model_params < 10_000_000:
        batch_size = 32
    elif no_of_model_params < 2_000_000_000:
        batch_size = 16 if max_tokens < 100 else 8
    else:
        batch_size = 8 if max_tokens < 100 else 4
    gradient_accumulation_steps = 2
    max_batch_size = no_of_records // gradient_accumulation_steps
    batch_size = int(np.clip(a=batch_size, a_min=1, a_max=max_batch_size))
    return batch_size, gradient_accumulation_steps


def _learn_rate_heuristic(no_of_model_params: int) -> float:
    if no_of_model_params < 10_000_000:
        learn_rate = 4e-4
    else:
        learn_rate = 2e-5
    return learn_rate


@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample_full_precision(
    layer: nn.Linear, activations: list[torch.Tensor], backprops: torch.Tensor
) -> dict[nn.Parameter, torch.Tensor]:
    """
    Overwrite the default backward hook for linear layer implemented in
    https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/linear.py#L29-L48

    The difference is that this ensures activations and backprops are upcasted to float32 before the computation.
    """
    activations = activations[0]
    ret = {}
    if layer.weight.requires_grad:
        gs = torch.einsum("n...i,n...j->nij", backprops.float(), activations.float())
        ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = torch.einsum("n...k->nk", backprops.float())
    return ret


class LanguageModelCheckpoint(ModelCheckpoint):
    def _save_model_weights(self, model: PreTrainedModel | GradSampleModule | DDP) -> None:
        if isinstance(model, GradSampleModule):
            # TODO: multi-gpu
            # should be found more elegant way
            if isinstance(model._module, DPDDP):
                # LSTMFromScratchLMHeadModel with DPLSTM layers can only be saved without safe serialization
                # the weights will be saved as *.bin instead of .safetensors
                safe_serialization = model._module.module.config.model_type != LSTMFromScratchConfig.model_type
                model._module.module.save_pretrained(self.workspace.model_path, safe_serialization=safe_serialization)
            else:
                # LSTMFromScratchLMHeadModel with DPLSTM layers can only be saved without safe serialization
                # the weights will be saved as *.bin instead of .safetensors
                safe_serialization = model._module.config.model_type != LSTMFromScratchConfig.model_type
                model._module.save_pretrained(self.workspace.model_path, safe_serialization=safe_serialization)
        # TODO: multi-gpu
        elif isinstance(model,DDP):
            model.module.save_pretrained(self.workspace.model_path)
        else:
            model.save_pretrained(self.workspace.model_path)

    def _clear_model_weights(self) -> None:
        patterns = ["*.safetensors", "*.bin", "*.json"]
        files = [f for p in patterns for f in self.workspace.model_path.glob(p)]
        for f in files:
            f.unlink(missing_ok=True)

    def model_weights_path_exists(self) -> bool:
        return any(self.workspace.model_path.glob("*.safetensors")) or any(self.workspace.model_path.glob("*.bin"))


def _calculate_per_label_losses(
    model: PreTrainedModel | GradSampleModule | DDP | DPDDP, step_data: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(input_ids=step_data["input_ids"], attention_mask=step_data["attention_mask"])
    logits = outputs.logits

    labels = step_data["labels"]
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    # TODO: multi-gpu
    if isinstance(model, GradSampleModule):
        # TODO: multi-gpu
        # should be found more elegant way
        if isinstance(model._module, DPDDP):
            vocab_size = model._module.module.config.vocab_size
        else:
            vocab_size = model._module.config.vocab_size
    elif isinstance(model,DDP):
        vocab_size = model.module.config.vocab_size
    else:
        vocab_size = model.config.vocab_size

    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Ensure tensors are on the same device
    shift_labels = shift_labels.to(shift_logits.device)
    loss_fct = CrossEntropyLoss(reduction="sum")
    loss = loss_fct(shift_logits, shift_labels)
    labels_ignored = torch.sum(shift_labels == -100)
    num_labels = shift_labels.numel() - labels_ignored
    return loss, num_labels


@torch.no_grad()
def _calculate_val_loss(model: PreTrainedModel | GradSampleModule | DDP, val_dataloader: DataLoader) -> float:
    # TODO: multi-gpu
    # should be found more elegant way
    if isinstance(model, GradSampleModule):
        if isinstance(model._module, DPDDP):
            device = model._module.module.device
        else:
            device = model._module.device
    elif isinstance(model,DDP):
        device = model.module.device
    else:
        device = model.device
    total_loss = torch.tensor(0, dtype=torch.float32, device=device)
    total_num_labels = torch.tensor(0, dtype=torch.long, device=device)
    model.eval()
    for step_data in val_dataloader:
        step_data = {k: v.to(device) for k, v in step_data.items()}
        loss, num_labels = _calculate_per_label_losses(model, step_data)
        total_loss += loss
        total_num_labels += num_labels
    model.train()
    val_loss_avg = total_loss / total_num_labels
    return val_loss_avg.item()

# TODO: multi-gpu
def train(
    *,
    model: str = "MOSTLY_AI/LSTMFromScratch-3m",
    max_training_time: float = 14400.0,  # 10 days
    max_epochs: float = 100.0,  # 100 epochs
    batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
    enable_flexible_generation: bool = True,
    differential_privacy: DifferentialPrivacyConfig | dict | None = None,
    upload_model_data_callback: Callable | None = None,
    model_state_strategy: ModelStateStrategy | str = ModelStateStrategy.reset,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
):
    device = (
        torch.device(device)
        if device is not None
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    if device == torch.device("cuda"):
        gpu_world_size = torch.cuda.device_count()
    else:
        gpu_world_size = None

    # in general, we could keep mp.spawn() for single GPU amd CPU training as well
    # but for now, we keep the old code path for single GPU and CPU training
    if gpu_world_size is not None and gpu_world_size  > 1:
        mp.spawn(_train, args=(gpu_world_size,
                               model,
                               max_training_time,
                               max_epochs,
                               batch_size,
                               gradient_accumulation_steps,
                               enable_flexible_generation,
                               differential_privacy,
                               upload_model_data_callback,
                               model_state_strategy,
                               workspace_dir,
                               update_progress,
                               ),
                 nprocs=gpu_world_size,
                 join=True,
                 )
    else:
        _train(rank=None,
               gpu_world_size=gpu_world_size,
               model=model,
               max_training_time=max_training_time,
               max_epochs=max_epochs,
               batch_size=batch_size,
               gradient_accumulation_steps=gradient_accumulation_steps,
               enable_flexible_generation=enable_flexible_generation,
               differential_privacy=differential_privacy,
               upload_model_data_callback=upload_model_data_callback,
               model_state_strategy=model_state_strategy,
               workspace_dir=workspace_dir,
               update_progress=update_progress,
               )

# TODO: multi-gpu
def _train(
    rank: int | None, # TODO: should be fixed to None for cpu
    gpu_world_size: int | None, # TODO: should be fixed to None for cpu
    model: str,
    max_training_time: float,
    max_epochs: float,
    batch_size: int | None,
    gradient_accumulation_steps: int | None,
    enable_flexible_generation: bool,
    differential_privacy: DifferentialPrivacyConfig | dict | None,
    upload_model_data_callback: Callable | None,
    model_state_strategy: ModelStateStrategy | str,
    workspace_dir: str | Path,
    update_progress: ProgressCallback | None,
):
    _LOG.info("TRAIN_LANGUAGE started")
    t0_ = time.time()
    workspace_dir = ensure_workspace_dir(workspace_dir)
    workspace = Workspace(workspace_dir)

    with ProgressCallbackWrapper(
        update_progress, progress_messages_path=workspace.model_progress_messages_path
    ) as progress:
        _LOG.info(f"numpy={version('numpy')}, pandas={version('pandas')}")
        _LOG.info(f"torch={version('torch')}, opacus={version('opacus')}")
        _LOG.info(f"transformers={version('transformers')}, peft={version('peft')}")
        # TODO: multi-gpu
        # we could use only the last branch of the if-else for single GPU and CPU training too
        # and provide rank=None and gpu_world_size=None as default values, but for now, we keep the old code path
        if rank is None and gpu_world_size is None:
            device = torch.device("cpu")
        elif rank is None and gpu_world_size == 1:
            device = torch.device("cuda")
        else:
            ddp_setup(rank, gpu_world_size)
            device = torch.device("cuda", rank)

        _LOG.info(f"{device=}")
        bf16_supported = is_bf16_supported(device)
        _LOG.info(f"{bf16_supported=}")
        use_mixed_precision = bf16_supported and model != LSTMFromScratchConfig.model_id
        _LOG.info(f"{use_mixed_precision=}")

        tgt_stats = workspace.tgt_stats.read()
        trn_cnt = tgt_stats["no_of_training_records"]
        val_cnt = tgt_stats["no_of_validation_records"]

        # set defaults
        model_id = model or LSTMFromScratchConfig.model_id
        _LOG.info(f"{model_id=}")
        _LOG.info(f"{enable_flexible_generation=}")
        max_training_time = max(0.0, max_training_time * 60)  # convert to seconds
        _LOG.info(f"{max_training_time=}s")
        max_epochs = max(0.0, max_epochs)
        max_epochs_cap = math.ceil((trn_cnt + val_cnt) / 50)
        if max_epochs_cap < max_epochs:
            _LOG.info(f"{max_epochs=} -> max_epochs={max_epochs_cap} due to small sample size")
            max_epochs = max_epochs_cap
        else:
            _LOG.info(f"{max_epochs=}")
        with_dp = differential_privacy is not None
        _LOG.info(f"{with_dp=}")
        _LOG.info(f"{model_state_strategy=}")

        # initialize callbacks
        upload_model_data_callback = upload_model_data_callback or (lambda *args, **kwargs: None)

        # the line below fixes issue with growing epoch time for later epochs
        # https://discuss.pytorch.org/t/training-time-gets-slower-and-slower-on-cpu/145483
        torch.set_flush_denormal(True)

        # load raw encoded data
        if check_early_training_exit(workspace=workspace, trn_cnt=trn_cnt, val_cnt=val_cnt):
            empty_ds = Dataset.from_dict({"ctx": [], "tgt": []})
            raw_dataset = DatasetDict({"train": empty_ds, "validation": empty_ds})
        else:
            data_files = {
                "train": [str(f) for f in workspace.encoded_data_trn.fetch_all()],
                "validation": [str(f) for f in workspace.encoded_data_val.fetch_all()],
            }
            disable_progress_bar()
            raw_dataset = load_dataset("parquet", data_files=data_files)

        def shuffle_tgt_columns(x):
            x_tgt = pd.DataFrame([json.loads(x.pop("tgt"))])  # convert to DataFrame
            x_tgt = x_tgt.sample(frac=1, axis=1)  # shuffle columns
            x_tgt = row_to_json(
                x_tgt.add_prefix("tgt" + TABLE_COLUMN_INFIX).squeeze(axis=0), is_target=True
            )  # convert back to JSON
            return x | {"tgt": x_tgt}

        # shuffle target columns if flexible generation is enabled
        anyorder_dataset = raw_dataset.map(shuffle_tgt_columns) if enable_flexible_generation else raw_dataset

        def concat_prompt_and_response(x):
            return {"content": "".join(x.values())}

        # concatenate prompt and response to form the content
        content_dataset = anyorder_dataset.map(
            concat_prompt_and_response, remove_columns=anyorder_dataset["train"].column_names
        )

        tokenizer_args = {
            "padding_side": "right",
            "truncation_side": "right",
            # these must be False at initialization, as we manually add them later in tokenize_fn
            "add_bos_token": False,
            "add_eos_token": False,
            "legacy": True,
        }

        _LOG.info("create training model")
        model_checkpoint = LanguageModelCheckpoint(workspace=workspace)
        # TODO: multi-gpu
        model: PreTrainedModel | PeftModel | DPDDP | DDP

        # check how to handle existing model weights
        if isinstance(model_state_strategy, str):
            model_state_strategy = ModelStateStrategy(model_state_strategy)
        if not model_checkpoint.model_weights_path_exists():
            _LOG.info(f"model weights not found; change strategy from {model_state_strategy} to RESET")
            model_state_strategy = ModelStateStrategy.reset
        _LOG.info(f"{model_state_strategy=}")
        if model_state_strategy in [ModelStateStrategy.resume, ModelStateStrategy.reuse]:
            _LOG.info("load existing model weights")
            torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, np.dtypes.Float64DType])
            resume_from_last_checkpoint = True
            model_id_or_path = workspace.model_path
        else:  # ModelStateStrategy.reset
            _LOG.info("clear existing checkpoint files")
            model_checkpoint.clear_checkpoint()
            resume_from_last_checkpoint = False
            model_id_or_path = model

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

        t0 = time.time()
        if model == LSTMFromScratchConfig.model_id:
            if resume_from_last_checkpoint:
                tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, **tokenizer_args)
                model, _ = load_base_model_and_config(
                    model_id_or_path, device=device, is_peft_adapter=False, is_training=True
                )
            else:
                # fresh initialization of the custom tokenizer and LSTM model
                tokenizer_train_iter = (
                    content_dataset["train"][i : i + 1_000]["content"]
                    for i in range(0, len(content_dataset["train"]), 1_000)
                )
                # train a custom tokenizer and convert it to a LlamaTokenizerFast object
                tokenizer = train_tokenizer(tokenizer_train_iter, tokenizer_kwargs=tokenizer_args, tgt_stats=tgt_stats)
                model_config = LSTMFromScratchConfig(vocab_size=len(tokenizer), with_dp=with_dp)
                model = LSTMFromScratchLMHeadModel(model_config).to(device)
        else:
            model, model_config = load_base_model_and_config(
                model_id_or_path,
                device=device,
                is_peft_adapter=resume_from_last_checkpoint,
                is_training=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, **tokenizer_args)
            if tokenizer.eos_token is None:
                if getattr(model_config, "eos_token_id", None) is not None:
                    tokenizer.eos_token_id = model_config.eos_token_id
            if tokenizer.bos_token is None:
                if getattr(model_config, "bos_token_id", None) is not None:
                    tokenizer.bos_token_id = model_config.bos_token_id
                else:
                    tokenizer.bos_token = tokenizer.eos_token
                    _LOG.warning("bos token not found, setting eos token as bos token")
            if getattr(tokenizer, "pad_token", None) is None:
                if getattr(tokenizer, "unk_token", None) is not None:
                    # warning: unk token can be valid output, although very unlikely for proper tokenizers
                    tokenizer.pad_token = tokenizer.unk_token
                else:
                    _LOG.warning(
                        "pad_token not found and unk token not available as fallback, setting eos token as pad token -- this can result in eos being masked."
                    )
                    tokenizer.pad_token = tokenizer.eos_token
            if resume_from_last_checkpoint:
                model = PeftModel.from_pretrained(model, model_id_or_path, is_trainable=True)
            else:
                peft_config = LoraConfig(
                    lora_alpha=32,  # 2x rank
                    lora_dropout=0.05,
                    r=16,
                    target_modules="all-linear",
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                model.add_adapter(peft_config)

        _LOG.info(f"model loading time: {time.time() - t0:.2f}s")

        # TODO: multi-gpu
        # set up distributed training if there are multiple devices available
        if gpu_world_size is not None and gpu_world_size  > 1:
            if with_dp:
                model = DPDDP(model)
            else:
                model = DDP(model)

        model.train()
        # TODO: multi-gpu
        no_of_model_params = model.num_parameters() if hasattr(model, "num_parameters") else model.module.num_parameters()
        _LOG.info(f"{no_of_model_params=}")
        # TODO: multi-gpu
        no_of_trainable_model_params = model.num_parameters(only_trainable=True) if hasattr(model, "num_parameters") else model.module.num_parameters(only_trainable=True)
        _LOG.info(f"{no_of_trainable_model_params=}")

        _LOG.info(f"{tokenizer=}")
        # TODO: multi-gpu
        # save the tokenizer only on the main process
        if rank is None or rank == 0:
            tokenizer.save_pretrained(workspace.model_path)

        tokenized_datasets = content_dataset.map(
            partial(
                tokenize_fn,
                tokenizer=tokenizer,
                text_key="content",
                add_bos_token=True,
                add_eos_token=True,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
            ),
            batched=True,
            remove_columns=content_dataset["train"].column_names,
        )
        data_collator = MostlyDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        max_tokens_estimate = estimate_max_tokens(tgt_stats)
        default_batch_size, default_gradient_accumulation_steps = _training_batch_size_heuristic(
            no_of_records=trn_cnt, no_of_model_params=no_of_model_params, max_tokens=max_tokens_estimate
        )
        if batch_size is None:
            batch_size = default_batch_size
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = default_gradient_accumulation_steps

        # setup params for input pipeline
        batch_size = max(1, min(batch_size, trn_cnt))
        gradient_accumulation_steps = max(1, min(gradient_accumulation_steps, trn_cnt // batch_size))
        trn_batch_size = batch_size * gradient_accumulation_steps
        trn_steps = max(1, trn_cnt // trn_batch_size)
        val_batch_size = max(1, min(batch_size, val_cnt))
        val_steps = max(1, val_cnt // val_batch_size)

        if initial_lr is None:
            initial_lr = _learn_rate_heuristic(no_of_model_params)

        # TODO: multi-gpu
        # save pretrained model at early exit, if there is only one device, or it is the master device
        if rank is None or rank == 0:
            # early exit if there is not enough data to train the model
            if len(tokenized_datasets["train"]) == 0 or len(tokenized_datasets["validation"]) == 0:
                _LOG.warning("not enough data to train model; skipping training")
                model.save_pretrained(workspace.model_path) if hasattr(model, "save_pretrained") else model.module.save_pretrained(workspace.model_path)
                return

        # TODO: multi-gpu
        # THIS PART REQUIRES MORE ATTENTION
        if gpu_world_size is not None and gpu_world_size  > 1 and not with_dp:
            # if it distributed training, we need to set up the samplers and data loaders accordingly
            # also we should adjust steps
            # also if it is with DP then the Opacus DP Data Loader will be used later in the code wrapping the vanilla trn_dataloader
            # shuffling should not be set here, it is handled by .set_epoch(int(epoch)) below  (see DistributedSampler docs)
            trn_sampler = DistributedSampler(tokenized_datasets["train"])
            val_sampler = DistributedSampler(tokenized_datasets["validation"], shuffle=False)

            _LOG.info(f"Total {trn_cnt=}, total {val_cnt=}")

            # TODO: This should be checked, if number of samples from trn_sampler and val_sampler should be used instead!?
            trn_cnt = trn_cnt // gpu_world_size
            val_cnt = val_cnt // gpu_world_size
            trn_steps = max(1, trn_steps // gpu_world_size)
            val_steps = max(1, val_steps // gpu_world_size)

            # https://discuss.pytorch.org/t/how-to-choose-num-worker-when-using-ddp/140978
            # num_workers <= cpu_count / GPU_count if dataloader is CPU intensive,
            # if the dataloader is IO intensive, it might be larger and set to 2 * cpu_count / GPU_count
            num_workers = mp.cpu_count() // gpu_world_size
        else:
            trn_sampler = None
            val_sampler = None
            num_workers = 0


        trn_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=(trn_sampler is None),
            sampler=trn_sampler,
            # either DP logical batch size or grad accumulation physical batch size
            batch_size=trn_batch_size if with_dp else batch_size,
            collate_fn=data_collator,
            pin_memory=(device.type == "cuda"),
            # num_workers=num_workers, # did not see any improvement
        )
        val_dataloader = DataLoader(
            tokenized_datasets["validation"],
            shuffle=False,
            sampler=val_sampler,
            batch_size=val_batch_size,
            collate_fn=data_collator,
            pin_memory=(device.type == "cuda"),
            # num_workers=num_workers, # did not see any improvement
        )

        _LOG.info(f"{trn_cnt=}, {val_cnt=}")
        _LOG.info(f"{trn_batch_size=}, {val_batch_size=}")
        _LOG.info(f"{trn_steps=}, {val_steps=}")
        _LOG.info(f"{batch_size=}, {gradient_accumulation_steps=}, {initial_lr=}")

        early_stopper = EarlyStopper(val_loss_patience=4)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=initial_lr)
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

        if with_dp:
            if isinstance(differential_privacy, DifferentialPrivacyConfig):
                dp_config = differential_privacy.model_dump()
            else:
                dp_config = DifferentialPrivacyConfig(**differential_privacy).model_dump()
            dp_max_epsilon = dp_config.get("max_epsilon") or float("inf")
            dp_delta = dp_config.get("delta", 1e-5)
            # the implementation of PRV accountant seems to have numerical and memory issues for small noise multiplier
            # therefore, we choose RDP instead as it is more stable and provides comparable privacy guarantees
            dp_accountant = "rdp"  # hard-coded for now
            _LOG.info(f"{dp_config=}, {dp_accountant=}")
            privacy_engine = PrivacyEngine(accountant=dp_accountant)
            if model_state_strategy == ModelStateStrategy.resume and workspace.model_dp_accountant_path.exists():
                _LOG.info("restore DP accountant state")
                torch.serialization.add_safe_globals([getattr, PRVAccountant, RDPAccountant, GaussianAccountant])
                privacy_engine.accountant.load_state_dict(
                    torch.load(workspace.model_dp_accountant_path, map_location=device, weights_only=True),
                )

            # Opacus will return the modified objects
            # - model: wrapped in GradSampleModule and contains additional hooks for computing per-sample gradients
            # - optimizer: wrapped in DPOptimizer and will do different operations during virtual steps and logical steps
            # - dataloader: the dataloader with batch_sampler=UniformWithReplacementSampler (for Poisson sampling)
            model, optimizer, trn_dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=trn_dataloader,
                noise_multiplier=dp_config.get("noise_multiplier"),
                max_grad_norm=dp_config.get("max_grad_norm"),
                poisson_sampling=True,
            )
            # TODO: multi-gpu
            # This part is not clear: how does it work with batch_sampler=BatchSplittingSampler from wrap_data_loader()?
            if gpu_world_size is not None and gpu_world_size  > 1:
                # Opacus DP Data Loader is used for distributed training with DP
                trn_dataloader = DPDataLoader.from_data_loader(trn_dataloader, distributed=True)
            # this further wraps the dataloader with batch_sampler=BatchSplittingSampler to achieve gradient accumulation
            # it will split the sampled logical batches into smaller sub-batches with batch_size
            trn_dataloader = wrap_data_loader(
                data_loader=trn_dataloader, max_batch_size=batch_size, optimizer=optimizer
            )
        else:
            privacy_engine = None
            dp_config, dp_delta, dp_accountant = None, None, None

        progress_message = None
        start_trn_time = time.time()
        last_msg_time = time.time()
        # TODO: multi-gpu
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        # For multi GPU it should be run BEFORE creating DataLoader iterator
        # when it is with DP and distributed trn_sampler is not used expliccitly, it is handled by DPDataLoader?
        if gpu_world_size is not None and gpu_world_size  > 1:
            if trn_sampler is not None:
                trn_sampler.set_epoch(int(epoch))
            else:
                # for DPDDP we don't specify Sampler explicitly
                trn_dataloader.batch_sampler.epoch = int(epoch)

        trn_data_iter = iter(trn_dataloader)
        do_stop = False

        #TODO: multi-gpu
        # it is a flag for multi GPU to stop the training
        multi_gpu_do_stop = torch.tensor(False, dtype=torch.bool, device=device)

        current_lr = initial_lr
        forward_ctx_mgr = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_mixed_precision else nullcontext()
        )
        # infinite loop over training steps, until we decide to stop
        # either because of max_epochs, max_training_time or early_stopping
        while not do_stop:
            is_checkpoint = 0
            steps += 1
            epoch = steps / trn_steps

            stop_accumulating_grads = False
            accumulated_steps = 0
            if not with_dp:
                optimizer.zero_grad(set_to_none=True)
            while not stop_accumulating_grads:
                # fetch next training (micro)batch
                try:
                    step_data = next(trn_data_iter)
                except StopIteration:
                    # TODO: multi-gpu
                    # Assume that this happens only at the end of the epoch
                    # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
                    # For multi GPU, it should be run BEFORE creating DataLoader iterator
                    # FIXME: if it continues training then it should be compared not with epoch 0
                    if gpu_world_size is not None and gpu_world_size  > 1:
                        if trn_sampler is not None:
                            trn_sampler.set_epoch(int(epoch))
                        else:
                            # for DPDDP we don't specify Sampler explicitly
                            trn_dataloader.batch_sampler.epoch = int(epoch)
                    trn_data_iter = iter(trn_dataloader)
                    step_data = next(trn_data_iter)
                step_data = {k: v.to(device) for k, v in step_data.items()}
                if with_dp:
                    # opacus handles the gradient accumulation internally
                    optimizer.zero_grad(set_to_none=True)
                with warnings.catch_warnings():
                    # remove this ctx mgr and filter when https://github.com/pytorch/pytorch/issues/130659 is fixed
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cpu.amp.autocast.*")
                    warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-full backward hook*")
                    # forward pass + calculate sample losses
                    with forward_ctx_mgr:
                        outputs = model(**step_data)
                    # FIXME approximation, should be divided by total sum of number of tokens in the batch
                    #  as in _calculate_per_label_losses, also the final sample may be smaller than the batch size.
                    step_loss = outputs.loss / (1 if with_dp else gradient_accumulation_steps)
                    step_loss.backward()
                accumulated_steps += 1
                # explicitly count the number of processed samples as the actual batch size can vary when DP is on
                samples += step_data["input_ids"].shape[0]
                if with_dp:
                    # for DP training, the optimizer will do different operations during virtual steps and logical steps
                    # - virtual step: clip and accumulate gradients
                    # - logical step: clip and accumulate gradients, add noises to gradients and update parameters
                    optimizer.step()
                    # if step was not skipped, it was a logical step, and we can stop accumulating gradients
                    stop_accumulating_grads = not optimizer._is_last_step_skipped
                elif accumulated_steps % gradient_accumulation_steps == 0:
                    # update parameters with accumulated gradients
                    optimizer.step()
                    stop_accumulating_grads = True
            current_lr = optimizer.param_groups[0][
                "lr"
            ]  # currently assume that we have the same lr for all param groups
            # only the scheduling for ReduceLROnPlateau is postponed until the metric becomes available
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step()

            # do validation
            do_validation = epoch.is_integer()
            if do_validation:
                # calculate val loss
                with forward_ctx_mgr:
                    val_loss = _calculate_val_loss(model=model, val_dataloader=val_dataloader)
                dp_epsilon = privacy_engine.get_epsilon(dp_delta) if with_dp else None
                has_exceeded_dp_max_epsilon = dp_epsilon > dp_max_epsilon if with_dp else False
                # save model weights with the best validation loss (and that hasn't exceeded DP max epsilon)
                # TODO: multi-gpu
                # save model weights for best model if there is only one device, or it is the master device
                if not has_exceeded_dp_max_epsilon and (rank is None or rank == 0):
                    is_checkpoint = model_checkpoint.save_checkpoint_if_best(
                        val_loss=val_loss,
                        model=model,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        dp_accountant=privacy_engine.accountant if with_dp else None,
                    )
                else:
                    _LOG.info("early stopping: current DP epsilon has exceeded max epsilon")
                # gather message for progress with checkpoint info
                progress_message = ProgressMessage(
                    epoch=epoch,
                    is_checkpoint=is_checkpoint,
                    steps=steps,
                    samples=samples,
                    trn_loss=None,
                    val_loss=val_loss,
                    total_time=total_time_init + time.time() - start_trn_time,
                    learn_rate=current_lr,
                    dp_eps=dp_epsilon,
                    dp_delta=dp_delta,
                )
                # check for early stopping
                do_stop = early_stopper(val_loss=val_loss) or has_exceeded_dp_max_epsilon
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
                dp_epsilon = privacy_engine.get_epsilon(dp_delta) if with_dp else None
                progress_message = ProgressMessage(
                    epoch=epoch,
                    is_checkpoint=is_checkpoint,
                    steps=steps,
                    samples=samples,
                    trn_loss=None,
                    val_loss=None,
                    total_time=total_time_init + time.time() - start_trn_time,
                    learn_rate=current_lr,
                    dp_eps=dp_epsilon,
                    dp_delta=dp_delta,
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

            # check for max_epochs
            if epoch > max_epochs:
                do_stop = True

            # check for max_training_time
            total_training_time = total_time_init + time.time() - start_trn_time
            if total_training_time > max_training_time:
                do_stop = True

            # TODO: multi-gpu
            # If early stopping happens for one device, then the other will wait the synchronization
            # resulting in a timeout, so we need to send a signal to stop the other devices
            if gpu_world_size is not None and gpu_world_size  > 1:
                if do_stop:
                    multi_gpu_do_stop = torch.tensor(True, dtype=torch.bool, device=device)

                all_reduce(multi_gpu_do_stop, async_op=False)

                # we check a signal from other devices to stop the training
                if multi_gpu_do_stop.item():
                    _LOG.info(f"{device}: received STOP EXECUTION signal from another device")
                    do_stop = True

        # no checkpoint is saved yet because the training stopped before the first epoch ended
        if not model_checkpoint.has_saved_once():
            # TODO: multi-gpu
            #  save weights if there is only one device, or it is the master device
            if rank is None or rank == 0:
                _LOG.info("saving model weights, as none were saved so far")
                model_checkpoint.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    dp_accountant=privacy_engine.accountant if with_dp else None,
                )
            if total_training_time > max_training_time:
                _LOG.info("skip validation loss calculation due to time-capped early stopping")
                val_loss = None
            else:
                _LOG.info("calculate validation loss")
                with forward_ctx_mgr:
                    val_loss = _calculate_val_loss(model=model, val_dataloader=val_dataloader)
            dp_epsilon = privacy_engine.get_epsilon(dp_delta) if with_dp else None
            # send a final message to inform how far we've progressed
            progress_message = ProgressMessage(
                epoch=epoch,
                is_checkpoint=1,
                steps=steps,
                samples=samples,
                trn_loss=None,
                val_loss=val_loss,
                total_time=total_training_time,
                learn_rate=current_lr,
                dp_eps=dp_epsilon,
                dp_delta=dp_delta,
            )
            progress.update(
                completed=steps,
                total=steps,
                message=progress_message,
            )
            # ensure everything gets uploaded
            upload_model_data_callback()
            # TODO: multi-gpu
            # if there are multiple devices, we need to clean up the process group
    if gpu_world_size is not None and gpu_world_size  > 1:
        destroy_process_group()
    _LOG.info(f"TRAIN_LANGUAGE finished in {time.time() - t0_:.2f}s")
