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

from __future__ import annotations

import os

os.environ["VLLM_USE_V1"] = "1"


import time
from os import PathLike

import torch
from peft import PeftConfig
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs.data import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import GuidedDecodingParams

from mostlyai.engine._language.common import is_bf16_supported
from mostlyai.engine._language.engine.base import EngineMetrics, LanguageEngine
from mostlyai.engine._language.tokenizer_utils import tokenize_fn


def get_dynamic_gpu_memory_utilization(utilization_ratio: float = 0.9) -> float:
    """
    Calculate dynamic GPU memory utilization based on available memory.

    Args:
        utilization_ratio: Fraction of available GPU memory to use (default: 0.9)

    Returns:
        GPU memory utilization as a fraction.
    """
    if not torch.cuda.is_available():
        return utilization_ratio  # fallback for non-GPU environments

    try:
        # Get free and total memory from CUDA
        free_memory, total_memory = torch.cuda.mem_get_info()

        # Use specified ratio of free memory
        target_memory = free_memory * utilization_ratio
        utilization = target_memory / total_memory

        # Ensure utilization is within reasonable bounds (0.1 to 0.95)
        return max(0.1, min(0.95, utilization))

    except Exception:
        # Fallback to provided ratio if anything goes wrong
        return utilization_ratio


class VLLMEngine(LanguageEngine):
    def __init__(
        self, model_path: PathLike | str, device: torch.device, max_new_tokens: int, tokenizer_max_length: int
    ):
        self.device = device
        self.tokenizer_max_length = tokenizer_max_length
        self.max_new_tokens = max_new_tokens

        peft_config = PeftConfig.from_pretrained(model_path)
        base_config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)

        model_path = str(model_path)
        self._lora_request = LoRARequest("adapter", 1, model_path)
        # Get max model length from config (different models use different attribute names)
        config_max_model_len = getattr(
            base_config,
            "max_position_embeddings",
            getattr(base_config, "n_positions", getattr(base_config, "max_sequence_length", 2048)),
        )

        self.llm = LLM(
            model=peft_config.base_model_name_or_path,
            tokenizer=model_path,
            max_model_len=min(config_max_model_len, self.tokenizer_max_length + max_new_tokens),
            enable_lora=True,
            dtype=torch.bfloat16 if is_bf16_supported(device) else torch.float16,
            # enforce_eager=True,  # results in big slowdown, but is needed when running pytest locally
            swap_space=0,
            disable_log_stats=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=get_dynamic_gpu_memory_utilization(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            truncation_side="left",
            legacy=True,
            # these must be False at initialization, as we manually add them later in tokenize_fn
            add_bos_token=False,
            add_eos_token=False,
        )
        self._prepared_schemas = None

    def get_default_batch_size(self) -> int:
        return 192

    def supports_json_enforcing(self) -> bool:
        return True

    def generate(
        self, text: list[str], sampling_temperature: float, sampling_top_p: float
    ) -> tuple[list[int], EngineMetrics]:
        tokenize_kwargs = dict(
            tokenizer=self.tokenizer,
            return_tensors=None,
            add_bos_token=True,
            add_eos_token=False,
            padding=False,
            truncation=True,
            max_length=self.tokenizer_max_length,  # truncates input
        )
        t_tokenize = time.time()
        inputs = tokenize_fn(text=text, **tokenize_kwargs)
        tokenize_time = time.time() - t_tokenize

        actual_batch_size = len(inputs["input_ids"])

        # Create sampling params with guided decoding if schemas are prepared
        effective_schemas = self._prepared_schemas

        sampling_params = []
        for i in range(actual_batch_size):
            guided_decoding = None
            if effective_schemas and i < len(effective_schemas):
                # Convert Pydantic model to JSON schema for guided decoding
                schema_dict = effective_schemas[i].model_json_schema()
                guided_decoding = GuidedDecodingParams(json=schema_dict)

            sampling_params.append(
                SamplingParams(
                    max_tokens=self.max_new_tokens,
                    temperature=sampling_temperature,
                    top_p=sampling_top_p,
                    guided_decoding=guided_decoding,
                )
            )
        t_generate = time.time()
        outputs = self.llm.generate(
            prompts=[TokensPrompt(prompt_token_ids=token_ids) for token_ids in inputs["input_ids"]],
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=self._lora_request,
        )
        generate_time = time.time() - t_generate
        metrics = EngineMetrics(tokenize_time=tokenize_time, generate_time=generate_time)
        return [r.outputs[0].token_ids for r in outputs], metrics

    def cleanup(self):
        del self.llm
        cleanup_dist_env_and_memory()

    def update_json_constraints(self, schemas: list[BaseModel] | None) -> None:
        """Update JSON schema constraints for the next generation call."""
        self._prepared_schemas = list(schemas) if schemas else None

    def can_reuse_schemas(self) -> bool:
        """VLLMEngine can handle variable batch sizes since it creates sampling params per sample."""
        return True
