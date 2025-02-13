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

from formatron import schemas
from formatron.formats import json


# direct copy from formatron
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


# completely altered
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


# removed sequence metadata since unnecessary and altered number_metadata to use ours
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
