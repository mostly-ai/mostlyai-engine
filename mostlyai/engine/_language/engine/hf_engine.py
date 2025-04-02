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
from mostlyai.engine._language.xgrammar_utils import adapt_grammar


class XGrammarHFLogitsProcessor(transformers.LogitsProcessor):
    """
    LogitsProcessor for processing logits in transformers' generate() method.

    Example usage
    -------------
        .. code:: python

            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            config = AutoConfig.from_pretrained(model_name)
            # This can be larger than tokenizer.vocab_size due to paddings
            full_vocab_size = config.vocab_size
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)

            grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
            compiled_grammar = grammar_compiler.compile_builtin_json_grammar()
            xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
            model.generate(prompt, logits_processor=[xgr_logits_processor])

        For an end-to-end example, see folder `examples/hf_transformers/`.

    Notes
    -----
        - Note that this LogitsProcessor can only be used once. For each `generate()` call,
            instantiate a new one.
        - Note that this implementation may contain extra overhead.
    """

    def __init__(self, compiled_grammar: xgr.CompiledGrammar | list[xgr.CompiledGrammar]):
        """Initialize the LogitsProcessor.

        Parameters
        ----------
        compiled_grammar : xgr.CompiledGrammar | List[xgr.CompiledGrammar]
            One or more grammars compiled according to the given grammar and the model's tokenizer_info.
        """
        self.matchers: list[xgr.GrammarMatcher] = []
        self.compiled_grammars = compiled_grammar if isinstance(compiled_grammar, list) else [compiled_grammar]
        self.full_vocab_size = self.compiled_grammars[0].tokenizer_info.vocab_size
        self.token_bitmask = None
        self.prefilled = False
        self.batch_size = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Accept token sampled in the last iteration, fill in bitmask, and apply bitmask to logits.

        Returns:
            scores: Logits modified with bitmask.
        """
        # Lazily initialize GrammarMatchers and bitmask
        if len(self.matchers) == 0:
            self.batch_size = input_ids.shape[0]
            self.compiled_grammars = (
                self.compiled_grammars if len(self.compiled_grammars) > 1 else self.compiled_grammars * self.batch_size
            )
            assert len(self.compiled_grammars) == self.batch_size, (
                "The number of compiled grammars must be equal to the batch size."
            )
            self.matchers = [xgr.GrammarMatcher(self.compiled_grammars[i]) for i in range(self.batch_size)]
            self.token_bitmask = xgr.allocate_token_bitmask(self.batch_size, self.full_vocab_size)

        if input_ids.shape[0] != self.batch_size:
            raise RuntimeError(
                "Expect input_ids.shape[0] to be LogitsProcessor.batch_size."
                + f"Got {input_ids.shape[0]} for the former, and {self.batch_size} for the latter."
            )

        if not self.prefilled:
            # Have not sampled a token yet
            self.prefilled = True
        else:
            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    sampled_token = input_ids[i][-1]
                    assert self.matchers[i].accept_token(sampled_token)

        for i in range(self.batch_size):
            if not self.matchers[i].is_terminated():
                self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)

        # We only support masking logits on CUDA or CPU
        device_type = scores.device.type
        if device_type != "cuda":
            scores = scores.to("cpu")
        xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device))
        if device_type != "cuda":
            scores = scores.to(device_type)

        # NOTE: Cannot reset here because __call__ is not invoked when stop token
        # is sampled. This is why each `generate()` call needs to instantiate an
        # LogitsProcessor

        return scores


def get_tokenizer_info_for_lstm(tokenizer: AutoTokenizer, vocab_size: int):
    # trimmed down version of xgr.TokenizerInfo.from_huggingface
    # the original function sets vocab_type to VocabType.RAW,
    # but LSTM tokenizer needs VocabType.BYTE_FALLBACK, because the usage of metaspace ("▁")
    vocab_dict = tokenizer.get_vocab()
    stop_token_ids = [tokenizer.eos_token_id]
    vocab_type = xgr.VocabType.BYTE_FALLBACK
    add_prefix_space = True  # TODO: what should this be?
    encoded_vocab = [""] * vocab_size
    for token, idx in vocab_dict.items():
        if idx < vocab_size:
            encoded_vocab[idx] = token
    tokenizer_info = xgr.TokenizerInfo(
        encoded_vocab,
        vocab_type=vocab_type,
        vocab_size=vocab_size,
        stop_token_ids=stop_token_ids,
        add_prefix_space=add_prefix_space,
    )
    return tokenizer_info


def create_formatter_logits_processors(
    schemas: list[BaseModel], tokenizer: AutoTokenizer, is_peft_adapter: bool
) -> list[transformers.LogitsProcessor]:
    # TODO: take vocab_size from model's config
    vocab_size = tokenizer.vocab_size
    if is_peft_adapter:
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    else:
        tokenizer_info = get_tokenizer_info_for_lstm(tokenizer, vocab_size=vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    grammars = (_json_schema_to_ebnf(json.dumps(schema.model_json_schema())) for schema in schemas)
    grammars = (adapt_grammar(grammar) for grammar in grammars)
    compiled_grammars = [grammar_compiler.compile_grammar(grammar) for grammar in grammars]
    logits_processor = XGrammarHFLogitsProcessor(compiled_grammars)
    return [logits_processor]


class HuggingFaceEngine(LanguageEngine):
    def __init__(
        self, model_path: PathLike | str, device: torch.device, max_new_tokens: int, tokenizer_max_length: int
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer_max_length = tokenizer_max_length
        self.is_peft_adapter = (Path(model_path) / "adapter_config.json").exists()

        model_path = str(model_path)
        self._model, _ = load_base_model_and_config(
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

    def initialize_logits_processors(self, schemas: list[BaseModel]):
        self._logits_processors = create_formatter_logits_processors(
            schemas=schemas, tokenizer=self.tokenizer, is_peft_adapter=self.is_peft_adapter
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
