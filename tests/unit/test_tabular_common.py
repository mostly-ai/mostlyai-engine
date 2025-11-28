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

import pytest

from mostlyai.engine._common import (
    ARGN_COLUMN,
    ARGN_PROCESSOR,
    ARGN_TABLE,
)
from mostlyai.engine._encoding_types.tabular.categorical import (
    CATEGORICAL_SUB_COL_SUFFIX,
    CATEGORICAL_UNKNOWN_TOKEN,
)
from mostlyai.engine._encoding_types.tabular.numeric import (
    NUMERIC_BINNED_SUB_COL_SUFFIX,
    NUMERIC_BINNED_UNKNOWN_TOKEN,
    NUMERIC_DISCRETE_SUB_COL_SUFFIX,
    NUMERIC_DISCRETE_UNKNOWN_TOKEN,
)
from mostlyai.engine._tabular.common import (
    fix_rare_token_probs,
    translate_fixed_probs,
)
from mostlyai.engine.domain import ModelEncodingType, RareCategoryReplacementMethod


class TestFixRareTokenProbs:
    @pytest.mark.parametrize(
        "encoding_type",
        [
            ModelEncodingType.tabular_numeric_binned,
            ModelEncodingType.tabular_numeric_discrete,
            ModelEncodingType.tabular_numeric_digit,
        ],
    )
    def test_numerics(self, encoding_type):
        subcol, code = {
            ModelEncodingType.tabular_numeric_binned: (NUMERIC_BINNED_SUB_COL_SUFFIX, 1),
            ModelEncodingType.tabular_numeric_discrete: (NUMERIC_DISCRETE_SUB_COL_SUFFIX, 0),
            ModelEncodingType.tabular_numeric_digit: (None, None),
        }[encoding_type]

        def get_stats() -> dict:
            return {
                "columns": {
                    "column": {
                        "encoding_type": encoding_type,
                        "codes": {
                            NUMERIC_DISCRETE_UNKNOWN_TOKEN: 0,
                            NUMERIC_BINNED_UNKNOWN_TOKEN: 1,
                        },
                    },
                }
            }

        stats = get_stats()
        fixed_probs = fix_rare_token_probs(stats)
        expected = {"column": {subcol: {code: 0.0}}} if subcol else {}
        assert fixed_probs == expected

    @pytest.mark.parametrize(
        "no_of_rare_categories,rare_category_replacement_method,do_fix",
        [
            (0, None, True),
            (1, None, False),
            (1, RareCategoryReplacementMethod.sample, True),
        ],
    )
    def test_categoricals(self, no_of_rare_categories, rare_category_replacement_method, do_fix):
        def get_stats() -> dict:
            return {
                "columns": {
                    "column": {
                        "encoding_type": ModelEncodingType.tabular_categorical.value,
                        "no_of_rare_categories": no_of_rare_categories,
                        "codes": {CATEGORICAL_UNKNOWN_TOKEN: 0},
                    },
                }
            }

        fixed_probs = fix_rare_token_probs(
            stats=get_stats(),
            rare_category_replacement_method=rare_category_replacement_method,
        )
        expected = {"column": {CATEGORICAL_SUB_COL_SUFFIX: {0: 0.0}}} if do_fix else {}
        assert fixed_probs == expected


class TestTranslateFixedProbs:
    def test(self):
        fixed_probs = {"column": {"cat": {0: 0.0}}}
        stats = {
            "columns": {
                "column": {
                    ARGN_PROCESSOR: "tgt",
                    ARGN_TABLE: "t0",
                    ARGN_COLUMN: "c0",
                }
            }
        }
        fixed_probs_model = translate_fixed_probs(fixed_probs, stats)
        assert fixed_probs_model == {"tgt:t0/c0__cat": {0: 0.0}}
