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


import datetime
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
from mostlyai.engine._language.temp_formatron import JsonExtractor
import collections
from formatron.schemas.schema import Schema

from mostlyai.engine.domain import ModelEncodingType

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


class MostlyFormatterBuilder(FormatterBuilder):
    def __init__(self):
        super().__init__()

    def json(self, schema: type[Schema] | collections.abc.Sequence, *, capture_name: str = None) -> JsonExtractor:
        """
        Create a JSON extractor. Check out the JsonExtractor docs for more details.

        Args:
            schema: The schema for extraction.
            capture_name: The capture name of the extractor, or `None` if the extractor does not capture.
        Returns:
            The JSON extractor.
        """

        def to_json(_json: str):
            local_schema = schema
            origin = typing.get_origin(local_schema)
            if origin is not None:
                local_schema = origin
            if isinstance(local_schema, type) and issubclass(local_schema, Schema):
                try:
                    return local_schema.from_json(_json)
                except JSONDecodeError:  # make ChoiceExtractor work appropriately
                    return None
            else:
                try:
                    return json.loads(_json)
                except JSONDecodeError:
                    return None

        return self._add_extractor(
            "json", lambda nonterminal: JsonExtractor(nonterminal, capture_name, schema, to_json)
        )


def get_formatter_builders(
    *, seed_df: pd.DataFrame | None = None, size: int | None = None, stats: dict
) -> list[FormatterBuilder]:
    assert (seed_df is not None) ^ (size is not None), "exactly one of seed_df or size must be provided"
    formatter_builders = []
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
    for _, seed_row in seed_df.iterrows():
        formatter_builder = MostlyFormatterBuilder()
        model_dict = {}
        if not seed_row.empty:
            model_dict |= {field_name: (Literal[seed_value], ...) for field_name, seed_value in seed_row.items()}  # type: ignore[valid-type]
        for field_name in unseeded_fields:
            if field_name in categorical_fields:
                model_dict[field_name] = (
                    Literal[tuple(stats["columns"][field_name]["categories"])],  # type: ignore[valid-type]
                    ...,
                )
            elif field_name in numeric_fields:
                max_scale = stats["columns"][field_name]["max_scale"]
                if max_scale == 0:
                    model_dict[field_name] = (int, ...)
                else:
                    model_dict[field_name] = (float, ...)
            elif field_name in datetime_fields:
                # model_dict[field_name] = (str, Field(pattern=r"19\\d{2}|20\\d{2}-0[1-9]|1[0-2]-0[1-9]|1[0-9]|2[0-9]|3[0-1]")) - might be able to make this work, but it fails
                model_dict[field_name] = (datetime.datetime, ...)
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
