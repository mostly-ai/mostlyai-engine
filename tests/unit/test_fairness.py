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

from mostlyai.engine.domain import ModelEncodingType
from mostlyai.engine._common import get_argn_name, ARGN_PROCESSOR, ARGN_TABLE, ARGN_COLUMN
from mostlyai.engine._encoding_types.tabular.categorical import CATEGORICAL_SUB_COL_SUFFIX
from mostlyai.engine._tabular.fairness import _get_sensitive_groups


@pytest.fixture(scope="module")
def tgt_stats():
    return {
        "columns": {
            "c0_cat": {
                "encoding_type": ModelEncodingType.tabular_categorical.value,
                "argn_processor": "tgt",
                "argn_table": "t0",
                "argn_column": "c0",
                "no_of_rare_categories": 0,
                "codes": {"_RARE_": 0, **{f"cat_{i}": i + 1 for i in range(2)}},
            },
            "c1_cat": {
                "encoding_type": ModelEncodingType.tabular_categorical.value,
                "argn_processor": "tgt",
                "argn_table": "t0",
                "argn_column": "c1",
                "no_of_rare_categories": 0,
                "codes": {"_RARE_": 0, **{f"cat_{i}": i + 1 for i in range(3)}},
            },
            "c2_cat": {
                "encoding_type": ModelEncodingType.tabular_categorical.value,
                "argn_processor": "tgt",
                "argn_table": "t0",
                "argn_column": "c2",
                "no_of_rare_categories": 1,
                "codes": {"_RARE_": 0, **{f"cat_{i}": i + 1 for i in range(5)}},
            },
            "c3_num": {
                "encoding_type": ModelEncodingType.tabular_numeric_auto.value,
                "argn_processor": "tgt",
                "argn_table": "t0",
                "argn_column": "c3",
            },
            "c4_cat": {
                "encoding_type": ModelEncodingType.tabular_categorical.value,
                "argn_processor": "tgt",
                "argn_table": "t0",
                "argn_column": "c4",
                "no_of_rare_categories": 0,
                "codes": {"_RARE_": 0, **{f"cat_{i}": i + 1 for i in range(7)}},
            },
        }
    }


@pytest.mark.parametrize(
    "target_column, sensitive_columns, expected_n_rows",
    [
        ("c0_cat", ["c1_cat", "c2_cat"], 18),  # 3 * (5+1)
        ("c0_cat", ["c1_cat", "c4_cat"], 21),  # 3 * 7
    ],
)
def test_get_sensitive_category_groups(tgt_stats, target_column, sensitive_columns, expected_n_rows):
    column_stats = tgt_stats["columns"]
    sensitive_sub_cols = [
        get_argn_name(
            argn_processor=tgt_stats["columns"][col][ARGN_PROCESSOR],
            argn_table=tgt_stats["columns"][col][ARGN_TABLE],
            argn_column=tgt_stats["columns"][col][ARGN_COLUMN],
            argn_sub_column=CATEGORICAL_SUB_COL_SUFFIX,
        )
        for col in sensitive_columns
    ]
    groups_df = _get_sensitive_groups(column_stats, sensitive_columns, sensitive_sub_cols)
    assert groups_df.shape[0] == expected_n_rows
