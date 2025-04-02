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
import pandas as pd
from pydantic import BaseModel, Field, SkipValidation, create_model
import torch
from typing import Literal

from mostlyai.engine._encoding_types.language.categorical import CATEGORICAL_UNKNOWN_TOKEN
from mostlyai.engine.domain import ModelEncodingType, RareCategoryReplacementMethod
import xgrammar as xgr
import transformers
from transformers import PreTrainedTokenizerBase


JSON_NULL = "null"


def ensure_seed_can_be_tokenized(sample_seed: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
    def transform(x: str | None) -> str:
        if pd.isna(x):
            null = tokenizer.decode(tokenizer.encode(JSON_NULL), skip_special_tokens=True)
            # xgrammar needs to be able to express JSON_NULL with available vocabulary
            # if that's the case, harmonize null-like values to None (e.g. pd.NA would cause xgrammar to fail)
            # otherwise, fallback to empty string
            return None if null == JSON_NULL else ""
        # skip tokens unseen during training
        return tokenizer.decode(tokenizer.encode(x), skip_special_tokens=True)

    return sample_seed.astype("string[pyarrow]").map(transform)


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
