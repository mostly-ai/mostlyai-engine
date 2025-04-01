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
from dataclasses import dataclass, field
import time
import pandas as pd
from pydantic import BaseModel, Field, SkipValidation, create_model
import torch
from typing import Literal

from mostlyai.engine._encoding_types.language.categorical import CATEGORICAL_UNKNOWN_TOKEN
from mostlyai.engine.domain import ModelEncodingType, RareCategoryReplacementMethod
import xgrammar as xgr
import transformers
from vllm.model_executor.guided_decoding.xgrammar_decoding import GrammarConfig, GrammarCompilerCache


def get_schemas(
    *,
    seed_df: pd.DataFrame | None = None,
    size: int | None = None,
    stats: dict,
    rare_category_replacement_method: RareCategoryReplacementMethod,
) -> list[BaseModel]:
    assert (seed_df is not None) ^ (size is not None), "exactly one of seed_df or size must be provided"
    schemas = []
    if seed_df is None:
        seed_df = pd.DataFrame(index=range(size))
    unseeded_fields = [c for c in list(stats["columns"].keys()) if c not in seed_df.columns.to_list()]
    field_types = {
        t: [col for col, col_stats in stats["columns"].items() if col_stats["encoding_type"] == t]
        for t in ModelEncodingType
    }
    categorical_fields = field_types.get(ModelEncodingType.language_categorical, [])
    numeric_fields = field_types.get(ModelEncodingType.language_numeric, [])
    datetime_fields = field_types.get(ModelEncodingType.language_datetime, [])
    cache = {}
    for _, seed_row in seed_df.iterrows():
        cache_key = hash(tuple(sorted([(field_name, str(seed_value)) for field_name, seed_value in seed_row.items()])))
        if cache_key in cache:
            schemas.append(cache[cache_key])
            continue
        model_dict = {}
        if not seed_row.empty:
            model_dict |= {field_name: (Literal[seed_value], ...) for field_name, seed_value in seed_row.items()}  # type: ignore[valid-type]
        for field_name in unseeded_fields:
            if field_name in categorical_fields:
                categories = stats["columns"][field_name]["categories"]
                if rare_category_replacement_method == RareCategoryReplacementMethod.sample and len(categories) > 1:
                    categories = [c for c in categories if c != CATEGORICAL_UNKNOWN_TOKEN]
                model_dict[field_name] = (Literal[tuple(categories)], ...)  # type: ignore[valid-type]
            elif field_name in numeric_fields:
                max_scale = stats["columns"][field_name]["max_scale"]
                min_min5 = min(stats["columns"][field_name]["min5"])
                max_max5 = max(stats["columns"][field_name]["max5"])
                if max_scale == 0:
                    model_dict[field_name] = (SkipValidation[int], Field(ge=min_min5, le=max_max5))
                else:
                    model_dict[field_name] = (
                        SkipValidation[float],
                        Field(ge=min_min5, le=max_max5, decimal_places=max_scale),
                    )
            elif field_name in datetime_fields:
                model_dict[field_name] = (
                    SkipValidation[str],
                    Field(
                        pattern=r"""(19\\d{2}|20\\d{2})-(0[1-9]|1[0-2])-(0[1-9]|1[0-9]|2[0-9]|3[0-1])T([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])"""
                    ),
                )
            else:
                model_dict[field_name] = (str, ...)
        schema = create_model("TargetModel", **model_dict)
        cache[cache_key] = schema
        schemas.append(schema)
    return schemas


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
    

@dataclass
class XGrammarVLLMLogitsProcessor:
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
                self.ctx = compiler\
                    .compile_json_schema(self.config.json_str,
                                         any_whitespace=any_whitespace)
            elif self.config.grammar_str is not None:
                self.ctx = compiler.compile_grammar(self.config.grammar_str)
            elif self.config.json_object:
                self.ctx = compiler.compile_builtin_json_grammar()
            else:
                raise ValueError(
                    "Invalid configuration for xgrammar logits processor")

    def __call__(self, input_ids: list[int],
                 scores: torch.Tensor) -> torch.Tensor:
        if self.ctx is None:
            self._ensure_ctx()

        if len(self.matchers) == 0:
            self.matchers = [
                xgr.GrammarMatcher(self.ctx) for _ in range(self.batch_size)
            ]
            self.token_bitmask = xgr.allocate_token_bitmask(
                self.batch_size, self.tokenizer_info.vocab_size)

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
        xgr.apply_token_bitmask_inplace(
            scores, self.token_bitmask.to(scores.device, non_blocking=True))
        if device_type != "cuda":
            scores = scores.to(dtype).to(device_type).squeeze()
        return scores

    def clone(self) -> XGrammarVLLMLogitsProcessor:
        """Create a new instance with shared compiled grammar
          but separate state"""
        new_processor = XGrammarVLLMLogitsProcessor(self.config, self.tokenizer_info)

        # Share the compiled grammar context (immutable after compilation)
        new_processor.ctx = self.ctx

        # Create fresh matchers for the new sequence
        if self.ctx is not None:
            new_processor.matchers = [
                xgr.GrammarMatcher(self.ctx) for _ in range(self.batch_size)
            ]

        # Create a new token bitmask with the same size
        if hasattr(self, 'token_bitmask') and self.token_bitmask is not None:
            new_processor.token_bitmask = self.token_bitmask

        # Copy simple attributes
        new_processor.batch_size = self.batch_size
        # Reset prefilled state for new sequence
        new_processor.prefilled = False
        return new_processor
