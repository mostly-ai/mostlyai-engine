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

# Per-Request Logits Processors are not supported in V1 (https://docs.vllm.ai/en/latest/getting_started/v1_user_guide.html#feature-model)
# Global Logits Processors will be new mechanism to use, but it is not yet released
# Here is PR for it: https://github.com/vllm-project/vllm/pull/13360
os.environ["VLLM_USE_V1"] = "0"

import contextlib
import gc
import time
from collections.abc import Generator
from os import PathLike

import torch
import xgrammar as xgr
from peft import PeftConfig
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import _get_and_verify_max_len
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

from mostlyai.engine._language.common import is_bf16_supported
from mostlyai.engine._language.engine.base import EngineMetrics, LanguageEngine
from mostlyai.engine._language.tokenizer_utils import tokenize_fn
from mostlyai.engine._language.xgrammar_utils import create_compiled_grammars


def cleanup_dist_env_and_memory():
    """Copy from current main of vllm replace by import when possible"""
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if not current_platform.is_cpu():
        torch.cuda.empty_cache()


class XGrammarLogitsProcessor:
    """
    Inspired by [XGrammarLogitsProcessor](https://github.com/vllm-project/vllm/blob/a43aa183dc0cb2639044c15d272e0ce1941392b0/vllm/model_executor/guided_decoding/xgrammar_decoding.py#L280).
    VLLM's XGrammarLogitsProcessor can be reused.
    """

    def __init__(self, compiled_grammar: xgr.CompiledGrammar):
        self.compiled_grammar = compiled_grammar
        self.tokenizer_info = compiled_grammar.tokenizer_info
        self.batch_size = 1

        self.matchers: list[xgr.GrammarMatcher] | None = None
        self.token_bitmask: torch.Tensor | None = None
        self.prefilled = False

    def __call__(self, input_ids: tuple[int], scores: torch.Tensor) -> torch.Tensor:
        # lazily initialize GrammarMatchers and bitmask
        if self.matchers is None:
            self.matchers = [xgr.GrammarMatcher(self.compiled_grammar) for _ in range(self.batch_size)]
            self.token_bitmask = xgr.allocate_token_bitmask(self.batch_size, self.tokenizer_info.vocab_size)

        if not self.prefilled:
            # have not sampled a token yet
            self.prefilled = True
        else:
            for i, matcher in enumerate(self.matchers):
                if not matcher.is_terminated():
                    sampled_token = input_ids[-1]
                    assert self.matchers[i].accept_token(sampled_token)

        for i, matcher in enumerate(self.matchers):
            if not matcher.is_terminated():
                matcher.fill_next_token_bitmask(self.token_bitmask, i)

        # token_bitmask is a CPU tensor for use with accept_token and
        # fill_next_token_bitmask so we move it to the device of scores
        device_type = scores.device.type
        dtype = scores.dtype
        if device_type != "cuda":
            # xgrammar on cpu only supports float32 scores
            # see: https://github.com/mlc-ai/xgrammar/blob/c1b64920cad24f44f235778c1c00bb52d57da01a/python/xgrammar/kernels/apply_token_bitmask_inplace_cpu.py#L22
            scores = scores.to("cpu").float().unsqueeze(0)

        # Note: In this method, if the tensors have different dimensions
        # on CPU device fails, but on GPU it runs without error. Hence the
        # unsqueeze above for scores, to match the token bitmask shape
        xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device, non_blocking=True))
        if device_type != "cuda":
            scores = scores.to(dtype).to(device_type).squeeze()
        return scores

    def clone(self) -> XGrammarLogitsProcessor:
        """Create a new instance with shared compiled grammar but separate state"""
        new_processor = XGrammarLogitsProcessor(self.compiled_grammar)

        # create fresh matchers for the new sequence
        if self.matchers is not None:
            new_processor.matchers = [xgr.GrammarMatcher(self.compiled_grammar) for _ in range(self.batch_size)]

        # create a new token bitmask with the same size
        if self.token_bitmask is not None:
            new_processor.token_bitmask = self.token_bitmask

        new_processor.batch_size = self.batch_size

        # reset prefilled state for new sequence
        new_processor.prefilled = False

        return new_processor


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
        config_max_model_len = _get_and_verify_max_len(
            base_config, max_model_len=None, disable_sliding_window=None, sliding_window_len=None
        )
        self.llm = LLM(
            model=peft_config.base_model_name_or_path,
            tokenizer=model_path,
            device=device.type,
            max_model_len=min(config_max_model_len, self.tokenizer_max_length + max_new_tokens),
            enable_lora=True,
            dtype=torch.bfloat16 if is_bf16_supported(device) else torch.float16,
            # enforce_eager=True,  # results in big slowdown, but is needed when running pytest locally
            swap_space=0,
            disable_log_stats=True,
            tensor_parallel_size=torch.cuda.device_count(),
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
        self._logits_processors = None

    def get_default_batch_size(self) -> int:
        return 192

    def supports_json_enforcing(self) -> bool:
        return True

    def initialize_logits_processors(self, schemas: Generator[BaseModel]):
        compiled_grammars = create_compiled_grammars(
            schemas=schemas,
            tokenizer=self.llm.get_tokenizer(),
            vocab_size=self.llm.llm_engine.get_model_config().get_vocab_size(),
            is_peft_adapter=True,
        )
        self._logits_processors = [XGrammarLogitsProcessor(compiled_grammar) for compiled_grammar in compiled_grammars]

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
        sampling_params = [
            SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=sampling_temperature,
                top_p=sampling_top_p,
                logits_processors=[lp],
            )
            for lp in self._logits_processors[:actual_batch_size]
        ]
        t_generate = time.time()
        outputs = self.llm.generate(
            prompts=None,
            prompt_token_ids=inputs["input_ids"],
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
