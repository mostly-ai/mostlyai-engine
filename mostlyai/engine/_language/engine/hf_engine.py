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
import time
from os import PathLike
from pathlib import Path
from collections.abc import Generator

import torch
import transformers
import xgrammar as xgr
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoTokenizer
from xgrammar.testing import _json_schema_to_ebnf

from mostlyai.engine._language.common import load_base_model_and_config
from mostlyai.engine._language.engine.base import EngineMetrics, LanguageEngine
from mostlyai.engine._language.tokenizer_utils import tokenize_fn
from mostlyai.engine._language.xgrammar_utils import prepend_grammar_root_with_space


def get_tokenizer_info_for_lstm(tokenizer: AutoTokenizer, vocab_size: int):
    # trimmed down version of xgr.TokenizerInfo.from_huggingface
    # the original function sets vocab_type to VocabType.RAW,
    # but LSTM tokenizer needs VocabType.BYTE_FALLBACK, because of the usage of metaspace ("▁")
    encoded_vocab = [""] * vocab_size
    for token, idx in tokenizer.get_vocab().items():
        if idx < vocab_size:
            encoded_vocab[idx] = token
    tokenizer_info = xgr.TokenizerInfo(
        encoded_vocab,
        vocab_type=xgr.VocabType.BYTE_FALLBACK,
        vocab_size=vocab_size,
        stop_token_ids=[tokenizer.eos_token_id],
        add_prefix_space=True,
    )
    return tokenizer_info


def create_formatter_logits_processors(
    schemas: Generator[BaseModel, None, None], tokenizer: AutoTokenizer, is_peft_adapter: bool, vocab_size: int
) -> list[transformers.LogitsProcessor]:
    # in general, there might be misalignment between the model's and tokenizer's vocab_size
    # the former is expected by XGrammar
    make_tokenizer_info = xgr.TokenizerInfo.from_huggingface if is_peft_adapter else get_tokenizer_info_for_lstm
    tokenizer_info = make_tokenizer_info(tokenizer, vocab_size=vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    schemas = (json.dumps(schema.model_json_schema()) for schema in schemas)
    grammars = (_json_schema_to_ebnf(schema) for schema in schemas)
    grammars = (prepend_grammar_root_with_space(grammar) for grammar in grammars)
    compiled_grammars = (grammar_compiler.compile_grammar(grammar) for grammar in grammars)
    logits_processors = [XGrammarLogitsProcessor(list(compiled_grammars))]
    return logits_processors


class XGrammarLogitsProcessor(transformers.LogitsProcessor):
    """
    Inspired by [LogitsProcessor](https://github.com/mlc-ai/xgrammar/blob/414473e7c029d0d9e2dfbeacb48afa946d0e3419/python/xgrammar/contrib/hf.py#L14).
    HuggingFace's XGrammarLogitsProcessor cannot be reused. Logits processors must be initialized for each call to generate().
    """

    def __init__(self, compiled_grammars: list[xgr.CompiledGrammar]):
        self.compiled_grammars = compiled_grammars
        self.vocab_size = self.compiled_grammars[0].tokenizer_info.vocab_size
        self.batch_size = len(compiled_grammars)

        self.matchers: list[xgr.GrammarMatcher] = []
        self.token_bitmask = None
        self.prefilled = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # lazily initialize GrammarMatchers and bitmask
        if len(self.matchers) == 0:
            self.matchers = [xgr.GrammarMatcher(self.compiled_grammars[i]) for i in range(self.batch_size)]
            self.token_bitmask = xgr.allocate_token_bitmask(self.batch_size, self.vocab_size)

        if input_ids.shape[0] != self.batch_size:
            raise RuntimeError(
                "Expect input_ids.shape[0] to be XGrammarLogitsProcessor.batch_size. "
                + f"Got {input_ids.shape[0]} for the former, and {self.batch_size} for the latter."
            )

        if not self.prefilled:
            # have not sampled a token yet
            self.prefilled = True
        else:
            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    sampled_token = input_ids[i][-1]
                    assert self.matchers[i].accept_token(sampled_token)

        for i in range(self.batch_size):
            if not self.matchers[i].is_terminated():
                self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)

        device_type = scores.device.type
        if device_type != "cuda":
            scores = scores.to("cpu")
        xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device))
        if device_type != "cuda":
            scores = scores.to(device_type)
        return scores


class HuggingFaceEngine(LanguageEngine):
    def __init__(
        self, model_path: PathLike | str, device: torch.device, max_new_tokens: int, tokenizer_max_length: int
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer_max_length = tokenizer_max_length
        self.is_peft_adapter = (Path(model_path) / "adapter_config.json").exists()

        model_path = str(model_path)
        self._model, self._model_config = load_base_model_and_config(
            model_path, device=device, is_peft_adapter=self.is_peft_adapter, is_training=False
        )
        if self.is_peft_adapter:
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
        is_trained_lstm_tokenizer = not self.is_peft_adapter and self.tokenizer.vocab_size > len(
            self.tokenizer.special_tokens_map
        )
        self._json_enforcing_possible = self.is_peft_adapter or is_trained_lstm_tokenizer
        self._logits_processors = None

    def get_default_batch_size(self) -> int:
        return self._default_batch_size

    def supports_json_enforcing(self) -> bool:
        return self._json_enforcing_possible

    def initialize_logits_processors(self, schemas: Generator[BaseModel, None, None]):
        self._logits_processors = create_formatter_logits_processors(
            schemas=schemas,
            tokenizer=self.tokenizer,
            is_peft_adapter=self.is_peft_adapter,
            vocab_size=self._model_config.vocab_size,
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

        t_generate = time.time()
        outputs = self._model.generate(**inputs, **generate_kwargs, logits_processor=self._logits_processors)
        generate_time = time.time() - t_generate

        _, input_length = inputs["input_ids"].shape
        # truncate the prompt from the outputs
        outputs = outputs[:, input_length:]
        metrics = EngineMetrics(tokenize_time=tokenize_time, generate_time=generate_time)
        return outputs.detach().cpu().tolist(), metrics

    def cleanup(self):
        pass
