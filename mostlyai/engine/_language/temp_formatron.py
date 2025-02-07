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

"""
The module defines the `JsonExtractor` class, which is used to extract data from a string in JSON format.
"""

import collections
import datetime
import typing

from formatron import extractor, schemas
from formatron.formats.json import _type_to_nonterminals

__all__ = ["JsonExtractor"]


FORMATRON_WHITESPACE_MAX_REPETITIONS = 10
SPACE_NONTERMINAL = f"[ \t\n\r]{{0,{FORMATRON_WHITESPACE_MAX_REPETITIONS}}}"

GRAMMAR_HEADER = rf"""integer ::= #"-?(0|[1-9]\\d*)";
number ::= #"-?(0|[1-9]\\d*)(\\.\\d+)?([eE][+-]?\\d+)?";
string ::= #'"([^\\\\"\u0000-\u001f]|\\\\["\\\\bfnrt/]|\\\\u[0-9A-Fa-f]{{4}})*"';
boolean ::= "true"|"false";
null ::= "null";
array ::= array_begin (json_value (comma json_value)*)? array_end;
object ::= object_begin (string colon json_value (comma string colon json_value)*)? object_end;
json_value ::= number|string|boolean|null|array|object;
datetime ::= #'"(19\\d{{2}}|20\\d{{2}})-(0[1-9]|1[0-2])-(0[1-9]|1[0-9]|2[0-9]|3[0-1]) ([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])"';
comma ::= #"{SPACE_NONTERMINAL},{SPACE_NONTERMINAL}";
colon ::= #"{SPACE_NONTERMINAL}:{SPACE_NONTERMINAL}";
object_begin ::= #" \\{{{SPACE_NONTERMINAL}";
object_end ::= #"{SPACE_NONTERMINAL}\\}}";
array_begin ::= #"\\[{SPACE_NONTERMINAL}";
array_end ::= #"{SPACE_NONTERMINAL}\\]";
"""

_type_id_to_nonterminal = {
    id(int): "integer",
    id(float): "number",
    id(str): "string",
    id(bool): "boolean",
    id(type(None)): "null",
    id(list): "array",
    id(dict): "object",
    id(typing.Any): "json_value",
    id(datetime.datetime): "datetime",
}


def _generate_kbnf_grammar(schema: schemas.schema.Schema | collections.abc.Sequence, start_nonterminal: str) -> str:
    """
    Generate a KBNF grammar string from a schema for JSON format.

    Args:
        schema: The schema to generate a grammar for.
        start_nonterminal: The start nonterminal of the grammar. Default is "start".

    Returns:
        The generated KBNF grammar string.
    """
    result = [GRAMMAR_HEADER]
    nonterminals = set()
    stack = [(schema, start_nonterminal)]
    while stack:
        (current, nonterminal) = stack.pop()
        type_id = id(current)
        if type_id in _type_id_to_nonterminal:
            line = f"{nonterminal} ::= {_type_id_to_nonterminal[type_id]};\n"
            result.append(line)
            continue
        _type_id_to_nonterminal[type_id] = nonterminal
        for i in _type_to_nonterminals:
            value = i(current, nonterminal)
            if value is not None:
                line, to_stack = value
                result.append(line)
                stack.extend(to_stack)
                nonterminals.add(nonterminal)
                break
        else:
            raise TypeError(f"{current} from {nonterminal} is not supported in json_generators!")
    return "".join(result)


class JsonExtractor(extractor.NonterminalExtractor):
    """
    An extractor that loads json data to an object from a string.
    """

    def __init__(
        self,
        nonterminal: str,
        capture_name: str | None,
        schema: schemas.schema.Schema | collections.abc.Sequence,
        to_object: typing.Callable[[str], schemas.schema.Schema],
    ):
        """
        Create a json extractor from a given schema or a list of supported types.

        Currently, the following data types are supported:

        - bool
        - int
          - positive int
          - negative int
          - nonnegative int
          - nonpositive int
        - float
          - positive float
          - negative float
          - nonnegative float
          - nonpositive float
        - str
          - optionally with min_length, max_length and pattern constraints
            - length is measured in UTF-8 character number after json parsing
            - *Warning*: too large difference between min_length and max_length can lead to enormous memory consumption!
            - pattern is mutually exclusive with min_length and max_length
            - pattern will be compiled to a regular expression so all caveats of regular expressions apply
            - pattern currently is automatically anchored at both ends
            - the generated json could be invalid if the pattern allows invalid content between the json string's quotes.
              - for example, `pattern=".*"` will allow '\"' to appear in the json string which is forbidden by JSON standard.
          - also supports substring_of constraint which constrains the string to be a substring of a given string
            - the generated json could be invalid if the given string contains invalid content when put into the json string's quotes.
              - for example, `substring_of="abc\""` will allow '\"' to appear in the json string which is forbidden by JSON standard.
        - NoneType
        - typing.Any
        - Subclasses of collections.abc.Mapping[str,T] and typing.Mapping[str,T] where T is a supported type,
        - Subclasses of collections.abc.Sequence[T] and typing.Sequence[T] where T is a supported type.
          - optionally with `minItems`, `maxItems`, `prefixItems` constraints
          - *Warning*: too large difference between minItems and maxItems can lead to very slow performance!
          - *Warning*: By json schema definition, prefixItems by default allows additional items and missing items in the prefixItems, which may not be the desired behavior and can lead to very slow performance if prefixItems is long!
        - tuple[T1,T2,...] where T1,T2,... are supported types. The order, type and number of elements will be preserved.
        - typing.Literal[x1,x2,...] where x1, x2, ... are instances of int, string, bool or NoneType, or another typing.Literal[y1,y2,...]
        - typing.Union[T1,T2,...] where T1,T2,... are supported types.
        - schemas.Schema where all its fields' data types are supported. Recursive schema definitions are supported as well.
          - *Warning*: while not required field is supported, they can lead to very slow performance and/or enormous memory consumption if there are too many of them!
        - Custom types registered via register_type_nonterminal()

        Args:
            nonterminal: The nonterminal representing the extractor.
            capture_name: The capture name of the extractor, or `None` if the extractor does not capture.
            schema: The schema.
            to_object: A callable to convert the extracted string to a schema instance.
        """
        super().__init__(nonterminal, capture_name)
        self._to_object = to_object
        self._rule_str = _generate_kbnf_grammar(
            schema, self.nonterminal
        )  # FIXME, probably just monkey patch this instead

    def extract(self, input_str: str) -> tuple[str, schemas.schema.Schema] | None:
        """
        Extract a schema instance from a string.

        Args:
            input_str: The input string to extract from.

        Returns:
            A tuple of the remaining string and the extracted schema instance, or `None` if extraction failed.
        """

        # Ensure the input string starts with '{' or '[' after stripping leading whitespace
        input_str = input_str.lstrip()
        if not input_str.startswith(("{", "[")):
            return None

        # Variables to track the balance of brackets and the position in the string
        bracket_count = 0
        position = 0
        in_string = False
        escape_next = False
        start_char = input_str[0]
        end_char = "}" if start_char == "{" else "]"

        # Iterate over the string to find where the JSON object or array ends
        for char in input_str:
            if not in_string:
                if char == start_char:
                    bracket_count += 1
                elif char == end_char:
                    bracket_count -= 1
                elif char == '"':
                    in_string = True
            else:
                if char == '"' and not escape_next:
                    in_string = False
                elif char == "\\":
                    escape_next = not escape_next
                else:
                    escape_next = False

            # Move to the next character
            position += 1

            # If brackets are balanced and we're not in a string, stop processing
            if bracket_count == 0 and not in_string:
                break
        else:
            return None
        # The position now points to the character after the last '}', so we slice to position
        json_str = input_str[:position]
        remaining_str = input_str[position:]
        # Return the unparsed remainder of the string and the decoded JSON object
        return remaining_str, self._to_object(json_str)

    @property
    def kbnf_definition(self):
        return self._rule_str
