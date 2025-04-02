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

import contextlib
import gc
import json
import time
from dataclasses import dataclass, field
from os import PathLike

import torch
import xgrammar as xgr
from peft import PeftConfig
from pydantic import BaseModel
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from vllm.config import _get_and_verify_max_len
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.xgrammar_decoding import GrammarCompilerCache, GrammarConfig
from vllm.platforms import current_platform
from vllm.sampling_params import GuidedDecodingParams
from xgrammar.testing import _json_schema_to_ebnf

from mostlyai.engine._language.common import is_bf16_supported
from mostlyai.engine._language.engine.base import EngineMetrics, LanguageEngine
from mostlyai.engine._language.tokenizer_utils import tokenize_fn
from mostlyai.engine._language.xgrammar_utils import adapt_grammar


def cleanup_dist_env_and_memory():
    """Copy from current main of vllm replace by import when possible"""
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if not current_platform.is_cpu():
        torch.cuda.empty_cache()


def create_formatter_logits_processors(llm: LLM, schemas: list[BaseModel]) -> list[XGrammarLogitsProcessor]:
    logits_processors = []
    tokenizer = llm.get_tokenizer()
    model_config = llm.llm_engine.get_model_config()
    for schema in schemas:
        grammar = adapt_grammar(_json_schema_to_ebnf(json.dumps(schema.model_json_schema())))
        guided_decoding_params = GuidedDecodingParams(grammar=grammar)
        grammar_config = GrammarConfig.from_guided_params(
            guided_params=guided_decoding_params, model_config=model_config, tokenizer=tokenizer, max_threads=8
        )
        tokenizer_info = GrammarConfig.tokenizer_info(grammar_config.tokenizer_data)
        logits_processor = XGrammarLogitsProcessor(config=grammar_config, tokenizer_info=tokenizer_info)
        logits_processors.append(logits_processor)
    return logits_processors


@dataclass
class XGrammarLogitsProcessor:
    # TODO: trim down further
    """Adapted from vllm.model_executor.guided_decoding.xgrammar_decoding.XGrammarLogitsProcessor"""

    config: GrammarConfig
    tokenizer_info: xgr.TokenizerInfo

    ctx: xgr.CompiledGrammar | None = None
    token_bitmask: torch.Tensor = None  # type: ignore[assignment]
    matchers: list[xgr.GrammarMatcher] = field(default_factory=list)
    batch_size: int = field(default=1)
    prefilled: bool = field(default=False)

    def _ensure_ctx(self):
        """Lazily initialize the processor in the worker process"""
        if self.ctx is None:
            compiler = GrammarCompilerCache.get_compiler(self.config)
            if self.config.json_str is not None:
                any_whitespace = self.config.any_whitespace
                self.ctx = compiler.compile_json_schema(self.config.json_str, any_whitespace=any_whitespace)
            elif self.config.grammar_str is not None:
                self.ctx = compiler.compile_grammar(self.config.grammar_str)
            elif self.config.json_object:
                self.ctx = compiler.compile_builtin_json_grammar()
            else:
                raise ValueError("Invalid configuration for xgrammar logits processor")

    def __call__(self, input_ids: list[int], scores: torch.Tensor) -> torch.Tensor:
        if self.ctx is None:
            self._ensure_ctx()

        if len(self.matchers) == 0:
            self.matchers = [xgr.GrammarMatcher(self.ctx) for _ in range(self.batch_size)]
            self.token_bitmask = xgr.allocate_token_bitmask(self.batch_size, self.tokenizer_info.vocab_size)

        if not self.prefilled:
            # Have not sampled a token yet
            self.prefilled = True
        else:
            for i, matcher in enumerate(self.matchers):
                if not matcher.is_terminated():
                    sampled_token = input_ids[-1]
                    assert self.matchers[i].accept_token(sampled_token)

        for i, matcher in enumerate(self.matchers):
            if not matcher.is_terminated():
                # @ubospica: ideally, fill_next_token_bitmask should be
                # parallelized with model decoding
                # See https://github.com/vllm-project/vllm/pull/10785/files#r1864278303
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
        """Create a new instance with shared compiled grammar
        but separate state"""
        new_processor = XGrammarLogitsProcessor(self.config, self.tokenizer_info)

        # Share the compiled grammar context (immutable after compilation)
        new_processor.ctx = self.ctx

        # Create fresh matchers for the new sequence
        if self.ctx is not None:
            new_processor.matchers = [xgr.GrammarMatcher(self.ctx) for _ in range(self.batch_size)]

        # Create a new token bitmask with the same size
        if hasattr(self, "token_bitmask") and self.token_bitmask is not None:
            new_processor.token_bitmask = self.token_bitmask

        # Copy simple attributes
        new_processor.batch_size = self.batch_size
        # Reset prefilled state for new sequence
        new_processor.prefilled = False
        return new_processor


class MaskInvalidIndicesLogitsProcessor:
    """
    Certain models have output size greater than their vocabulary size.
    This logits processor masks the output indices that do not correspond
    to a token id.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):  # : PretrainedTokenizer
        self.mask: torch.Tensor | None = None
        self.valid_token_ids = torch.tensor(list(tokenizer.vocab.values()))

    def __call__(self, input_ids: list[int], scores: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            ninf = float("-inf")
            self.mask = torch.full_like(scores, ninf)
            self.mask[..., self.valid_token_ids] = 0
        scores = scores + self.mask
        return scores


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
        )
        self._base_logits_processors = [MaskInvalidIndicesLogitsProcessor(self.llm.get_tokenizer())]
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

    def initialize_logits_processors(self, schemas: list[BaseModel]):
        self._logits_processors = create_formatter_logits_processors(schemas=schemas, llm=self.llm)

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
        if self._logits_processors is not None:
            sampling_params = [
                SamplingParams(
                    max_tokens=self.max_new_tokens,
                    temperature=sampling_temperature,
                    top_p=sampling_top_p,
                    logits_processors=[lp, *self._base_logits_processors],
                )
                for lp in self._logits_processors[:actual_batch_size]
            ]
        else:
            sampling_params = SamplingParams(
                max_tokens=self.max_new_tokens,
                temperature=sampling_temperature,
                top_p=sampling_top_p,
                logits_processors=self._base_logits_processors,
            )
        t_generate = time.time()
        outputs = self.llm.generate(
            prompts=None,
            prompt_token_ids=inputs["input_ids"],
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=self._lora_request,
        )
        generate_time = time.time() - t_generate
        # TODO: do we need this?
        if self._logits_processors is not None:
            self._logits_processors = [lp.clone() for lp in self._logits_processors]
        metrics = EngineMetrics(tokenize_time=tokenize_time, generate_time=generate_time)
        return [r.outputs[0].token_ids for r in outputs], metrics

    def cleanup(self):
        del self.llm
        cleanup_dist_env_and_memory()
