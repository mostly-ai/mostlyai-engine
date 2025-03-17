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

import importlib
import logging
from typing import Any

from peft import PeftConfig, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel, PretrainedConfig
from mostlyai.engine._language.lstm import LSTMFromScratchConfig
from pathlib import Path

import torch

_LOG = logging.getLogger(__name__)

MAX_LENGTH = 10_000


def is_bf16_supported(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    compute_capability = torch.cuda.get_device_capability(device)
    return compute_capability[0] >= 8


def get_attention_implementation(config: PretrainedConfig) -> str | None:
    model_cls = AutoModelForCausalLM._model_mapping[type(config)]
    attn_implementation = None
    if getattr(model_cls, "_supports_sdpa", False):
        attn_implementation = "sdpa"
    return attn_implementation


def estimate_max_tokens(tgt_stats: dict[str, Any]) -> int:
    estimated_nchar = (
        # accommodate leading space, curly brackets and eos
        10
        # each column is roughly like '"' + col + '": "' + value + '", '
        + sum([(1 + len(col) + 4 + stats["nchar_max"] + 3) for col, stats in tgt_stats["columns"].items()])
    )
    estimated_tokens = estimated_nchar / 2  # ~2 chars per tokens
    max_tokens = int(estimated_tokens * 1.4)  # add some safety buffer
    return max_tokens


def load_base_model_and_config(
    model_id_or_path: str | Path, device: torch.device, is_peft_adapter: bool, is_training: bool
) -> tuple[PreTrainedModel, PretrainedConfig]:
    model_id_or_path = str(model_id_or_path)
    if is_peft_adapter:
        # get the base model name from adapter_config.json
        peft_config = PeftConfig.from_pretrained(model_id_or_path)
        model_id_or_path = peft_config.base_model_name_or_path
        config = AutoConfig.from_pretrained(model_id_or_path)
    else:
        config = AutoConfig.from_pretrained(model_id_or_path)
        if config.model_type == LSTMFromScratchConfig.model_id:
            # make sure that we use standard LSTM layers during inference for the model trained with DP
            # (see https://opacus.ai/api/dp_rnn.html#opacus.layers.dp_rnn.DPLSTM for more details)
            if not is_training:
                config.with_dp = False
            return AutoModelForCausalLM.from_pretrained(model_id_or_path, config=config, device_map=device), config

    # Load pretrained base model
    use_cache = not is_training  # KV cache is not needed during training
    is_gpu_training = is_training and device.type == "cuda" and importlib.util.find_spec("bitsandbytes") is not None
    bf16_supported = is_bf16_supported(device)
    if bf16_supported:
        attn_implementation = get_attention_implementation(config)
        torch_dtype = torch.bfloat16
    else:
        attn_implementation = None
        torch_dtype = torch.float32
    if is_gpu_training:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    else:
        quantization_config = None
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        use_cache=use_cache,
        device_map=device,
        quantization_config=quantization_config,
    )
    if quantization_config:
        # convert all non-kbit layers to float32
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    if is_gpu_training and model.supports_gradient_checkpointing:
        # pay 50% time penalty for _large_ memory savings
        _LOG.info("enable gradient checkpointing")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model, config
