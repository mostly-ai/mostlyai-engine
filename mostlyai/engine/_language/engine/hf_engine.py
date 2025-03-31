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

from os import PathLike
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel
import torch
from formatron.integrations.transformers import create_formatter_logits_processor_list
from peft import PeftModel

from transformers import AutoTokenizer
from mostlyai.engine._language.common import load_base_model_and_config
from mostlyai.engine._language.formatron_utils import monkey_patch_formatron
from mostlyai.engine._language.tokenizer_utils import tokenize_fn

from mostlyai.engine._language.engine.base import EngineMetrics, LanguageEngine
from formatron.formatter import FormatterBuilder
import xgrammar as xgr
import transformers

from mostlyai.engine._language.xgrammar_utils import XGrammarHFLogitsProcessor


def create_hf_logits_processors(
    schemas: list[BaseModel], tokenizer: AutoTokenizer
) -> list[transformers.LogitsProcessor]:
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=tokenizer.vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled_grammars = [grammar_compiler.compile_json_schema(schema) for schema in schemas]
    logits_processor = XGrammarHFLogitsProcessor(compiled_grammars)
    return [logits_processor]


class HuggingFaceEngine(LanguageEngine):
    def __init__(
        self, model_path: PathLike | str, device: torch.device, max_new_tokens: int, tokenizer_max_length: int, dev=None
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer_max_length = tokenizer_max_length

        is_peft_adapter = (Path(model_path) / "adapter_config.json").exists()
        model_path = str(model_path)
        self._model, _ = load_base_model_and_config(
            model_path, device=device, is_peft_adapter=is_peft_adapter, is_training=False
        )
        if is_peft_adapter:
            self._model = PeftModel.from_pretrained(self._model, model_path, is_trainable=False)
            self._model = self._model.merge_and_unload()
            self._default_batch_size = 64
        else:
            # only the LSTM model does not have an adapter
            self._default_batch_size = 128

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            truncation_side="left",
            legacy=True,
            # these must be False at initialization, as we manually add them later in tokenize_fn
            add_bos_token=False,
            add_eos_token=False,
        )

        # we can't enforce JSON output if LSTM tokenizer training was skipped
        is_trained_lstm_tokenizer = not is_peft_adapter and self.tokenizer.vocab_size > len(
            self.tokenizer.special_tokens_map
        )
        self._json_enforcing_possible = is_peft_adapter or is_trained_lstm_tokenizer
        if self.supports_json_enforcing():
            monkey_patch_formatron()
        self._logits_processors = None
        self._dev = dev

    def get_default_batch_size(self) -> int:
        return self._default_batch_size

    def supports_json_enforcing(self) -> bool:
        return self._json_enforcing_possible

    def initialize_logits_processors(
        self,
        formatter_builders: list[FormatterBuilder],
        vocab_processors: list[Callable] | None = None,
        dev=None,
    ):
        if self._dev["use_xgrammar"]:
            self._logits_processors = create_hf_logits_processors(schemas=dev["schemas"], tokenizer=self.tokenizer)
        else:
            self._logits_processors = create_formatter_logits_processor_list(
                tokenizer=self.tokenizer, formatter_builders=formatter_builders, vocab_processors=vocab_processors
            )

    def generate(
        self, text: list[str], sampling_temperature: float, sampling_top_p: float
    ) -> tuple[list[int], EngineMetrics]:
        do_sample = sampling_temperature > 0.0

        tokenize_kwargs = dict(
            tokenizer=self.tokenizer,
            return_tensors="pt",
            add_bos_token=True,
            add_eos_token=False,
            padding=True,
            truncation=True,
            max_length=self.tokenizer_max_length,  # truncates input
        )
        t_tokenize = time.time()
        inputs = tokenize_fn(text=text, **tokenize_kwargs).to(self.device)
        tokenize_time = time.time() - t_tokenize

        generate_kwargs = dict(
            do_sample=do_sample,
            max_new_tokens=self.max_new_tokens,
            temperature=sampling_temperature if do_sample else None,
            top_p=sampling_top_p if do_sample else None,
            bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,  # must be eos or will get ValueError from formatron
            eos_token_id=self.tokenizer.eos_token_id,
        )

        if self._logits_processors is not None and not self._dev["use_xgrammar"]:
            # number of formatters must match the batch size, batch size is always reduced so this is fine
            actual_batch_size = len(inputs["input_ids"])
            self._logits_processors[0]._formatters = self._logits_processors[0]._formatters[:actual_batch_size]

        t_generate = time.time()
        outputs = self._model.generate(**inputs, **generate_kwargs, logits_processor=self._logits_processors)
        generate_time = time.time() - t_generate

        if self._logits_processors and not self._dev["use_xgrammar"]:
            self._logits_processors[0].reset()

        _, input_length = inputs["input_ids"].shape
        # truncate the prompt from the outputs
        outputs = outputs[:, input_length:]
        metrics = EngineMetrics(tokenize_time=tokenize_time, generate_time=generate_time)
        return outputs.detach().cpu().tolist(), metrics

    def cleanup(self):
        pass
