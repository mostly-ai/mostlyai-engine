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

import typing

import pandas as pd
from formatron.schemas.pydantic import ClassSchema
from json import JSONDecodeError
from pydantic import ValidationError
from formatron.formatter import FormatterBuilder
from typing import Literal
from formatron.formats import json
from pydantic import create_model
from transformers import PreTrainedTokenizerBase

JSON_NULL = "null"


def prepare_seed_for_formatron(sample_seed: pd.DataFrame, tokenizer: PreTrainedTokenizerBase) -> pd.DataFrame:
    def transform(x: str | None) -> str:
        if pd.isna(x):
            null = tokenizer.decode(tokenizer.encode(JSON_NULL), skip_special_tokens=True)
            # formatron needs to be able to express JSON_NULL with available vocabulary
            # if that's the case, harmonize null-like values to None (e.g. pd.NA would cause formatron to fail)
            # otherwise, fallback to empty string
            return None if null == JSON_NULL else ""
        # skip tokens unseen during training
        return tokenizer.decode(tokenizer.encode(x), skip_special_tokens=True)

    return sample_seed.astype("string[pyarrow]").map(transform)


def monkey_patch_formatron():
    # alter the Grammar of formatron's json schema
    FORMATRON_WHITESPACE_MAX_REPETITIONS = 10
    SPACE_NONTERMINAL = f"[ \t\n\r]{{0,{FORMATRON_WHITESPACE_MAX_REPETITIONS}}}"

    json.GRAMMAR_HEADER = rf"""integer ::= #"-?(0|[1-9]\\d*)";
    number ::= #"-?(0|[1-9]\\d*)(\\.\\d+)?([eE][+-]?\\d+)?";
    string ::= #'"([^\\\\"\u0000-\u001f]|\\\\["\\\\bfnrt/]|\\\\u[0-9A-Fa-f]{{4}})*"';
    boolean ::= "true"|"false";
    null ::= "null";
    array ::= array_begin (json_value (comma json_value)*)? array_end;
    object ::= object_begin (string colon json_value (comma string colon json_value)*)? object_end;
    json_value ::= number|string|boolean|null|array|object;
    comma ::= #"{SPACE_NONTERMINAL},{SPACE_NONTERMINAL}";
    colon ::= #"{SPACE_NONTERMINAL}:{SPACE_NONTERMINAL}";
    object_begin ::= #" \\{{{SPACE_NONTERMINAL}";
    object_end ::= #"{SPACE_NONTERMINAL}\\}}";
    array_begin ::= #"\\[{SPACE_NONTERMINAL}";
    array_end ::= #"{SPACE_NONTERMINAL}\\]";
    """


def get_formatter_builders(
    *, seed_df: pd.DataFrame | None = None, size: int | None = None, stats: dict
) -> list[FormatterBuilder]:
    assert (seed_df is not None) ^ (size is not None), "exactly one of seed_df or size must be provided"
    formatter_builders = []
    if seed_df is None:
        seed_df = pd.DataFrame(index=range(size))
    unseeded_fields = [c for c in list(stats["columns"].keys()) if c not in seed_df.columns.to_list()]
    categorical_fields = [
        column
        for column, column_stats in stats["columns"].items()
        if column_stats["encoding_type"] == "LANGUAGE_CATEGORICAL"
    ]
    numeric_fields = [
        column
        for column, column_stats in stats["columns"].items()
        if column_stats["encoding_type"] == "LANGUAGE_NUMERIC"
    ]
    for _, seed_row in seed_df.iterrows():
        formatter_builder = FormatterBuilder()
        model_dict = {}
        if not seed_row.empty:
            model_dict |= {field_name: (Literal[seed_value], ...) for field_name, seed_value in seed_row.items()}
        for field_name in unseeded_fields:
            if field_name in categorical_fields:
                model_dict[field_name] = (
                    Literal[tuple(cat for cat in stats["columns"][field_name]["codes"].keys())],
                    ...,
                )
            elif field_name in numeric_fields:
                max_decimals = stats["columns"][field_name]["max_decimals"]
                if max_decimals == 0:
                    model_dict[field_name] = (int, ...)
                else:
                    model_dict[field_name] = (float, ...)
            else:
                model_dict[field_name] = (str, ...)
        schema = create_model("TargetModel", **model_dict, __base__=MostlyClassSchema)
        formatter_builder.append_str(f"{formatter_builder.json(schema, capture_name=None)}")
        formatter_builders.append(formatter_builder)
    return formatter_builders


def get_vocab_processors(is_peft_adapter: bool) -> list[typing.Callable] | None:
    if not is_peft_adapter:

        def update_vocab_lstm(token_to_char: dict[bytes, bytes]):
            """
            Maps special tokens ("▁", "␊") back to their original representation (" ", "\n")
            (used in LSTM tokenizer)
            """
            token_to_char["\u2581".encode()] = b" "  # "▁" -> " "
            token_to_char["\u240a".encode()] = b"\n"  # "␊" -> "\n"

        return [update_vocab_lstm]
    return None


class MostlyClassSchema(ClassSchema):
    @classmethod
    def from_json(cls, _json: str) -> "MostlyClassSchema":
        """
        Create a MostlyClassSchema from a JSON string.
        """
        try:
            return cls.model_validate_json(_json)
        except ValidationError as e:
            for error in e.errors():
                if error["type"] == "json_invalid":
                    raise JSONDecodeError(
                        f"Caught pydantic ValidationError {e}, reraising as JSONDecodeError", _json, 0
                    )
            raise e
