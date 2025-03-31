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

import contextlib
from os import PathLike
import time
import typing

from mostlyai.engine._language.common import is_bf16_supported
from mostlyai.engine._language.engine.base import EngineMetrics, LanguageEngine
import torch
from formatron.config import EngineGenerationConfig
from formatron.formatter import FormatterBuilder

from peft import PeftConfig

from transformers import AutoTokenizer, AutoConfig, PreTrainedTokenizerBase
from mostlyai.engine._language.formatron_utils import monkey_patch_formatron
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.config import _get_and_verify_max_len
from mostlyai.engine._language.tokenizer_utils import tokenize_fn
from formatron.integrations.vllm import create_engine_vocabulary, FormattersLogitsProcessor
from vllm.distributed import destroy_model_parallel, destroy_distributed_environment

import gc
from vllm.platforms import current_platform


def cleanup_dist_env_and_memory():
    """Copy from current main of vllm replace by import when possible"""
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if not current_platform.is_cpu():
        torch.cuda.empty_cache()


def create_vllm_logits_processors(
    llm: LLM,
    formatter_builders: typing.Sequence[FormatterBuilder | None],
    configs: typing.Sequence[EngineGenerationConfig] = None,
    vocab_processors: list[typing.Callable] | None = None,
) -> list[FormattersLogitsProcessor]:
    """
    Create list of formatter logits processors without creating engine vocabulary several times.
    """
    tokenizer = llm.get_tokenizer()
    vocab = create_engine_vocabulary(tokenizer, vocab_processors)
    formatters = [
        i.build(vocab, lambda tokens: tokenizer.decode(tokens)) if i is not None else None for i in formatter_builders
    ]
    return [FormattersLogitsProcessor([formatter], tokenizer.eos_token_id, configs) for formatter in formatters]


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
        self, model_path: PathLike | str, device: torch.device, max_new_tokens: int, tokenizer_max_length: int, dev=None
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
        monkey_patch_formatron()
        self._dev = dev

    def get_default_batch_size(self) -> int:
        return 192

    def supports_json_enforcing(self) -> bool:
        return True

    def initialize_logits_processors(
        self,
        formatter_builders: list[FormatterBuilder],
        vocab_processors: list[typing.Callable] | None = None,
        dev=None,
    ):
        self._logits_processors = create_vllm_logits_processors(
            llm=self.llm, formatter_builders=formatter_builders, configs=None, vocab_processors=vocab_processors
        )

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
        if self._logits_processors is not None:
            for lp in self._logits_processors:
                lp.reset()
        metrics = EngineMetrics(tokenize_time=tokenize_time, generate_time=generate_time)
        return [r.outputs[0].token_ids for r in outputs], metrics

    def cleanup(self):
        del self.llm
        cleanup_dist_env_and_memory()
