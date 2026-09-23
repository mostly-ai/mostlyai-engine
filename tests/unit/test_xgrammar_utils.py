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

import numpy as np
import pandas as pd
from pydantic import BaseModel
from xgrammar.testing import _json_schema_to_ebnf

from mostlyai.engine._language.xgrammar_utils import create_schemas, prepend_grammar_root_with_space
from mostlyai.engine.domain import ModelEncodingType, RareCategoryReplacementMethod


def test_create_schemas_normalizes_seed_nans_to_json_null():
    stats = {
        "columns": {
            "country": {
                "encoding_type": ModelEncodingType.language_categorical,
                "categories": ["USA", "Poland"],
            }
        }
    }
    seed_df = pd.DataFrame({"country": [np.nan, pd.NA, None]})

    schemas = list(
        create_schemas(
            seed_df=seed_df,
            stats=stats,
            rare_category_replacement_method=RareCategoryReplacementMethod.constant,
        )
    )

    for schema in schemas:
        schema_json = json.dumps(schema.model_json_schema(), allow_nan=False)
        parsed = json.loads(schema_json)
        assert parsed["properties"]["country"]["const"] is None


def test_prepend_grammar_root_with_space_legacy_ebnf():
    grammar = 'root ::= "{"\n'
    assert prepend_grammar_root_with_space(grammar) == 'root ::= " {"\n'


def test_prepend_grammar_root_with_space_current_xgrammar():
    class Target(BaseModel):
        bio: str

    updated = prepend_grammar_root_with_space(_json_schema_to_ebnf(Target))
    assert 'root ::= ((" {"' in updated
    assert 'root ::= (("{"' not in updated
