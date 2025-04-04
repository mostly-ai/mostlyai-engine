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

import json
from collections.abc import Generator
from typing import Literal

import pandas as pd
import xgrammar as xgr
from pydantic import BaseModel, Field, SkipValidation, create_model
from transformers import PreTrainedTokenizerBase
from xgrammar.testing import _json_schema_to_ebnf

from mostlyai.engine._encoding_types.language.categorical import CATEGORICAL_UNKNOWN_TOKEN
from mostlyai.engine.domain import ModelEncodingType, RareCategoryReplacementMethod

JSON_NULL = "null"


def prepend_grammar_root_with_space(grammar: str) -> str:
    # XGrammar always starts with "{" when enforcing JSON Schema
    # training is done on strings like ` {...} {...}`
    # later, generation is done with prompt like ` {...}`
    # so the first output token must be a space
    start_of_grammar = 'root ::= "{"'
    start_of_grammar_with_space = 'root ::= " {"'
    assert start_of_grammar in grammar
    return grammar.replace(start_of_grammar, start_of_grammar_with_space)


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
) -> Generator[BaseModel]:
    assert (seed_df is not None) ^ (size is not None), "exactly one of seed_df or size must be provided"
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
            yield cache[cache_key]
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
                        pattern=r"""(19\d{2}|20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|1[0-9]|2[0-9]|3[0-1])T([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])"""
                    ),
                )
            else:
                model_dict[field_name] = (str, ...)
        schema = create_model("TargetModel", **model_dict)
        cache[cache_key] = schema
        yield schema


def _get_tokenizer_info_for_lstm(tokenizer: PreTrainedTokenizerBase, vocab_size: int):
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


def create_compiled_grammars(
    schemas: Generator[BaseModel], tokenizer: PreTrainedTokenizerBase, vocab_size: int, is_peft_adapter: bool
) -> Generator[xgr.CompiledGrammar]:
    # in general, there might be misalignment between the model's and tokenizer's vocab_size
    # the former is expected by XGrammar
    make_tokenizer_info = xgr.TokenizerInfo.from_huggingface if is_peft_adapter else _get_tokenizer_info_for_lstm
    tokenizer_info = make_tokenizer_info(tokenizer, vocab_size=vocab_size)
    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
    schemas = (json.dumps(schema.model_json_schema()) for schema in schemas)
    grammars = (_json_schema_to_ebnf(schema) for schema in schemas)
    grammars = (prepend_grammar_root_with_space(grammar) for grammar in grammars)
    compiled_grammars = (grammar_compiler.compile_grammar(grammar) for grammar in grammars)
    return compiled_grammars
