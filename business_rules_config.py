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
Business Rules Configuration for Loss Adjustment
==============================================

This module defines business rules for adjusting loss calculation during training.
It supports multiple target columns with different predefined distributions and lambda weights.
"""

# Business rules configuration for multiple target columns
BUSINESS_RULES_CONFIG = {
    "target_columns": {
        # "workclass": {
        #     "predefined_distribution": {
        #         0: 0.01, # _RARE_ - keep low
        #         1: 0.05, # ? - decrease from 0.057 to 0.05
        #         2: 0.05, # Federal-gov - increase from 0.029 to 0.05
        #         3: 0.1,  # Local-gov - increase from 0.064 to 0.1
        #         4: 0.01, # Never-worked - increase from 0.0002 to 0.01
        #         5: 0.15, # Private - decrease from 0.69 to 0.15
        #         6: 0.05, # Self-emp-inc - increase from 0.035 to 0.05
        #         7: 0.5,  # Self-emp-not-inc - increase from 0.079 to 0.5
        #         8: 0.08, # State-gov - increase from 0.041 to 0.08
        #         9: 0.01, # Without-pay - increase from 0.0004 to 0.01
        #     },
        #     "lambda_weight": 0.5,  # Weight for predefined distribution
        # },
        # "income": {
        #     "predefined_distribution": {
        #         0: 0.0,  # _RARE_ - keep at 0 since there are no rare categories
        #         1: 0.5,  # <=50K - target 50%
        #         2: 0.5,  # >50K - target 50%
        #     },
        #     "lambda_weight": 0.5,  # Weight for predefined distribution
        # },
        "age": {
            "predefined_distribution": {
                0: 0.0,
                1: 0.01369863,
                2: 0.01369863,
                3: 0.01369863,
                4: 0.01369863,
                5: 0.01369863,
                6: 0.01369863,
                7: 0.01369863,
                8: 0.01369863,
                9: 0.01369863,
                10: 0.01369863,
                11: 0.01369863,
                12: 0.01369863,
                13: 0.01369863,
                14: 0.01369863,
                15: 0.01369863,
                16: 0.01369863,
                17: 0.01369863,
                18: 0.01369863,
                19: 0.01369863,
                20: 0.01369863,
                21: 0.01369863,
                22: 0.01369863,
                23: 0.01369863,
                24: 0.01369863,
                25: 0.01369863,
                26: 0.01369863,
                27: 0.01369863,
                28: 0.01369863,
                29: 0.01369863,
                30: 0.01369863,
                31: 0.01369863,
                32: 0.01369863,
                33: 0.01369863,
                34: 0.01369863,
                35: 0.01369863,
                36: 0.01369863,
                37: 0.01369863,
                38: 0.01369863,
                39: 0.01369863,
                40: 0.01369863,
                41: 0.01369863,
                42: 0.01369863,
                43: 0.01369863,
                44: 0.01369863,
                45: 0.01369863,
                46: 0.01369863,
                47: 0.01369863,
                48: 0.01369863,
                49: 0.01369863,
                50: 0.01369863,
                51: 0.01369863,
                52: 0.01369863,
                53: 0.01369863,
                54: 0.01369863,
                55: 0.01369863,
                56: 0.01369863,
                57: 0.01369863,
                58: 0.01369863,
                59: 0.01369863,
                60: 0.01369863,
                61: 0.01369863,
                62: 0.01369863,
                63: 0.01369863,
                64: 0.01369863,
                65: 0.01369863,
                66: 0.01369863,
                67: 0.01369863,
                68: 0.01369863,
                69: 0.05479452,
                70: 0.02739726,
            },
            "lambda_weight": 0,
        }
    },
    "enable_lora_style": True,  # Only update weights related to target columns
    "zero_mask_target_columns": True,  # Use zero masking for target columns
}


def get_target_columns() -> list[str]:
    """Get list of target column names."""
    return list(BUSINESS_RULES_CONFIG["target_columns"].keys())


def get_predefined_distribution(column: str) -> dict[int, float] | None:
    """Get predefined distribution for a specific column."""
    return BUSINESS_RULES_CONFIG["target_columns"].get(column, {}).get("predefined_distribution")


def get_lambda_weight(column: str) -> float:
    """Get lambda weight for a specific column."""
    return BUSINESS_RULES_CONFIG["target_columns"].get(column, {}).get("lambda_weight", 0.5)


def is_target_column(column: str) -> bool:
    """Check if a column is a target column for business rules."""
    return column in BUSINESS_RULES_CONFIG["target_columns"]


def is_lora_style_enabled() -> bool:
    """Check if LORA-style fine-tuning is enabled."""
    return BUSINESS_RULES_CONFIG.get("enable_lora_style", False)


def is_zero_mask_enabled() -> bool:
    """Check if zero masking for target columns is enabled."""
    return BUSINESS_RULES_CONFIG.get("zero_mask_target_columns", False)
