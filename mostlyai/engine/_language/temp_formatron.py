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
The module defines the `MostlyJsonExtractor` class, which is used to extract data from a string in JSON format.
"""

import collections
import datetime
import typing

from formatron import schemas
from formatron.formats.json import _type_to_nonterminals, JsonExtractor

__all__ = ["MostlyJsonExtractor"]


FORMATRON_WHITESPACE_MAX_REPETITIONS = 10
SPACE_NONTERMINAL = f"[ \t\n\r]{{0,{FORMATRON_WHITESPACE_MAX_REPETITIONS}}}"

# Copy from formatron, altered to have limited whitespace repetitions and datetime format
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

# FIXME add grammar constraint of integer and number


# Copy from formatron except `datetime`
def _generate_kbnf_grammar(schema: schemas.schema.Schema | collections.abc.Sequence, start_nonterminal: str) -> str:
    """
    Generate a KBNF grammar string from a schema for JSON format.

    Args:
        schema: The schema to generate a grammar for.
        start_nonterminal: The start nonterminal of the grammar. Default is "start".

    Returns:
        The generated KBNF grammar string.
    """
    type_id_to_nonterminal = {
        id(int): "integer",
        id(float): "number",
        id(str): "string",
        id(bool): "boolean",
        id(type(None)): "null",
        id(list): "array",
        id(dict): "object",
        id(typing.Any): "json_value",
        id(datetime.datetime): "datetime",  # altered
    }
    result = [GRAMMAR_HEADER]
    nonterminals = set()
    stack = [(schema, start_nonterminal)]
    while stack:
        (current, nonterminal) = stack.pop()
        type_id = id(current)
        if type_id in type_id_to_nonterminal:
            line = f"{nonterminal} ::= {type_id_to_nonterminal[type_id]};\n"
            result.append(line)
            continue
        type_id_to_nonterminal[type_id] = nonterminal
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


class MostlyJsonExtractor(JsonExtractor):
    """
    Same as the parent class from formatron
    except that it uses `_generate_kbnf_grammar` from this file to construct self._rule_str
    """
    def __init__(
            self,
            nonterminal: str,
            capture_name: str | None,
            schema: schemas.schema.Schema | collections.abc.Sequence,
            to_object: typing.Callable[[str], schemas.schema.Schema],
    ):
        super(JsonExtractor, self).__init__(nonterminal, capture_name)
        self._to_object = to_object
        self._rule_str = _generate_kbnf_grammar(schema, self.nonterminal)
