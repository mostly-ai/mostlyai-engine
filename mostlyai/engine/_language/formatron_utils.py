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
from pydantic import Field, SkipValidation, ValidationError
from formatron.formatter import FormatterBuilder
from formatron import schemas
from formatron.formats import json
from typing import Literal
from pydantic import create_model
from transformers import PreTrainedTokenizerBase
from mostlyai.engine._encoding_types.language.categorical import CATEGORICAL_UNKNOWN_TOKEN
from mostlyai.engine.domain import ModelEncodingType, RareCategoryReplacementMethod

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


def get_formatter_builders(
    *,
    seed_df: pd.DataFrame | None = None,
    size: int | None = None,
    stats: dict,
    rare_category_replacement_method: RareCategoryReplacementMethod,
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
        formatter_builder = FormatterBuilder()
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


# copy formatron: direct copy from formatron
def _string_metadata(current: type, nonterminal: str):
    min_length = current.metadata.get("min_length")
    max_length = current.metadata.get("max_length")
    pattern = current.metadata.get("pattern")
    substring_of = current.metadata.get("substring_of")
    if pattern:
        assert not (min_length or max_length or substring_of), (
            "pattern is mutually exclusive with min_length, max_length and substring_of"
        )
    if substring_of:
        assert not (min_length or max_length or pattern), (
            "substring_of is mutually exclusive with min_length, max_length and pattern"
        )
    repetition_map = {
        (True, False): f"{{{min_length},}}",
        (False, True): f"{{0,{max_length}}}",
        (True, True): f"{{{min_length},{max_length}}}",
    }
    repetition = repetition_map.get((min_length is not None, max_length is not None))
    if repetition is not None:
        return (
            rf"""{nonterminal} ::= #'"([^\\\\"\u0000-\u001f]|\\\\["\\\\bfnrt/]|\\\\u[0-9A-Fa-f]{{4}}){repetition}"';
""",
            [],
        )
    if pattern is not None:
        pattern = pattern.replace("'", "\\'")
        return f"""{nonterminal} ::= #'"{pattern}"';\n""", []
    if substring_of is not None:
        return f"""{nonterminal} ::= '"' #substrs{repr(substring_of)} '"';\n""", []


# completely altered vs formatron
def _number_metadata(current: type, nonterminal: str):
    # For now only constrains number of digits and whether it is negative
    gt = current.metadata.get("gt")
    ge = current.metadata.get("ge")
    lt = current.metadata.get("lt")
    le = current.metadata.get("le")
    if lt is not None or gt is not None:
        raise NotImplementedError("gt and lt are not supported for number metadata")
    if le < ge:
        raise ValueError("le must be greater than or equal to ge")

    pattern_parts = []
    if issubclass(current.type, float):
        le, le_frac = str(le).split(".")
        ge, ge_frac = str(ge).split(".")
        le, le_frac = int(le), int(le_frac)
        ge, ge_frac = int(ge), int(ge_frac)
        decimal_places = current.metadata.get("decimal_places")

    if ge is not None and le is not None:
        if ge < 0 and le < 0:
            pattern_parts.append("-")
            min_num = abs(le)
            max_num = abs(ge)
            max_digits = len(str(max_num))
            min_digits = len(str(min_num))
            pattern_parts.append(rf"([1-9][0-9]{{{min_digits - 1},{max_digits - 1}}})")
        elif ge > 0:
            min_num = ge
            max_num = le
            max_digits = len(str(max_num))
            min_digits = len(str(min_num))
            pattern_parts.append(rf"([1-9][0-9]{{{min_digits - 1},{max_digits - 1}}})")
        else:
            if ge < 0:
                pattern_parts.append("-?")
            max_digits = max(len(str(abs(ge))), len(str(abs(le))))
            pattern_parts.append(rf"(0|[1-9][0-9]{{0,{max_digits - 1}}})")

    if issubclass(current.type, float):
        # FIXME: currently is not constrained
        pattern_parts.append(rf"(\\.[0-9]{{0,{decimal_places}}})?")

    pattern = "".join(pattern_parts)
    return f"""{nonterminal} ::= #"{pattern}";\n""", []


# copy formatron: removed sequence metadata since unnecessary and altered number_metadata to use ours
def _metadata(current: type, nonterminal: str):
    if isinstance(current, schemas.schema.TypeWithMetadata):
        original = typing.get_origin(current.type)
        if original is None:
            original = current.type
        if not current.metadata:
            return "", [(current.type, nonterminal)]
        if isinstance(current.type, type) and issubclass(current.type, str):
            return _string_metadata(current, nonterminal)
        elif isinstance(current.type, type) and issubclass(current.type, (int, float)):
            return _number_metadata(current, nonterminal)
    return None


def monkey_patch_formatron():
    FORMATRON_WHITESPACE_MAX_REPETITIONS = 10
    SPACE_NONTERMINAL = f"[ \t\n\r]{{0,{FORMATRON_WHITESPACE_MAX_REPETITIONS}}}"

    # Copy from formatron, altered to have limited whitespace repetitions and datetime format
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

    def alter_type_to_nonterminals_metadata_inplace(type_to_nonterminals: list[typing.Callable]):
        metadata_idx = [idx for idx, fn in enumerate(type_to_nonterminals) if fn.__name__ == "metadata"]
        if len(metadata_idx) == 1:
            type_to_nonterminals[metadata_idx[0]] = _metadata

    alter_type_to_nonterminals_metadata_inplace(json._type_to_nonterminals)
