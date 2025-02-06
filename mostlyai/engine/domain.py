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

from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, field_validator


class CustomBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)


class ModelType(str, Enum):
    """
    The type of model.

    - `TABULAR`: A generative AI model tailored towards tabular data, trained from scratch.
    - `LANGUAGE`: A generative AI model build upon a (pre-trained) language model.
    """

    tabular = "TABULAR"
    language = "LANGUAGE"


class ModelEncodingType(str, Enum):
    """
    The encoding type used for model training and data generation.

    - `AUTO`: Model chooses among available encoding types based on the column's data type.
    - `TABULAR_CATEGORICAL`: Model samples from existing (non-rare) categories.
    - `TABULAR_NUMERIC_AUTO`: Model chooses among 3 numeric encoding types based on the values.
    - `TABULAR_NUMERIC_DISCRETE`: Model samples from existing discrete numerical values.
    - `TABULAR_NUMERIC_BINNED`: Model samples from binned buckets, to then sample randomly within a bucket.
    - `TABULAR_NUMERIC_DIGIT`: Model samples each digit of a numerical value.
    - `TABULAR_CHARACTER`: Model samples each character of a string value.
    - `TABULAR_DATETIME`: Model samples each part of a datetime value.
    - `TABULAR_DATETIME_RELATIVE`: Model samples the relative difference between datetimes within a sequence.
    - `TABULAR_LAT_LONG`: Model samples a latitude-longitude column. The format is "latitude,longitude".
    - `LANGUAGE_TEXT`: Model will train a distinct LANGUAGE model for this column, to then generate free text.
    """

    auto = "AUTO"
    tabular_categorical = "TABULAR_CATEGORICAL"
    tabular_numeric_auto = "TABULAR_NUMERIC_AUTO"
    tabular_numeric_discrete = "TABULAR_NUMERIC_DISCRETE"
    tabular_numeric_binned = "TABULAR_NUMERIC_BINNED"
    tabular_numeric_digit = "TABULAR_NUMERIC_DIGIT"
    tabular_character = "TABULAR_CHARACTER"
    tabular_datetime = "TABULAR_DATETIME"
    tabular_datetime_relative = "TABULAR_DATETIME_RELATIVE"
    tabular_lat_long = "TABULAR_LAT_LONG"
    language_text = "LANGUAGE_TEXT"
    language_categorical = "LANGUAGE_CATEGORICAL"
    language_numeric = "LANGUAGE_NUMERIC"
    # language_datetime = "LANGUAGE_DATETIME"


class ModelStateStrategy(str, Enum):
    """
    The strategy of how any existing model states and training progress are to be handled.

    - `RESET`: Start training from scratch. Overwrite any existing model states and training progress.
    - `REUSE`: Reuse any existing model states, but start progress from scratch. Used for fine-tuning existing models.
    - `RESUME`: Reuse any existing model states and progress. Used for continuing an aborted training.
    """

    reset = "RESET"
    reuse = "REUSE"
    resume = "RESUME"


class RareCategoryReplacementMethod(str, Enum):
    """
    Specifies how rare categories will be sampled.
    Only applicable if value protection has been enabled.

    - `CONSTANT`: Replace rare categories by a constant `_RARE_` token.
    - `SAMPLE`: Replace rare categories by a sample from non-rare categories.
    """

    constant = "CONSTANT"
    sample = "SAMPLE"


class RebalancingConfig(CustomBaseModel):
    """
    Configure rebalancing.
    """

    column: str = Field(..., description="The name of the column to be rebalanced.")
    probabilities: dict[str, float] = Field(
        ...,
        description="The target distribution of samples values. The keys are the categorical values, and the values "
        "are the probabilities.",
    )

    @field_validator("probabilities", mode="after")
    def validate_probabilities(cls, v):
        if not all(0 <= v <= 1 for v in v.values()):
            raise ValueError("the probabilities must be between 0 and 1")
        if not sum(v.values()) <= 1:
            raise ValueError("the sum of probabilities must be less than or equal to 1")
        return v


class ImputationConfig(CustomBaseModel):
    """
    Configure imputation. Imputed columns will suppress the sampling of NULL values.
    """

    columns: list[str] = Field(..., description="The names of the columns to be imputed.")


class FairnessConfig(CustomBaseModel):
    """
    Configure a fairness objective for the table.

    The generated synthetic data will maintain robust statistical parity between the target column and
    the specified sensitive columns. All these columns must be categorical.
    """

    target_column: str = Field(..., alias="targetColumn")
    sensitive_columns: list[str] = Field(..., alias="sensitiveColumns")


class DifferentialPrivacyConfig(CustomBaseModel):
    """
    The differential privacy configuration for training the model.
    If not provided, then no differential privacy will be applied.
    """

    max_epsilon: float | None = Field(
        default=None,
        alias="maxEpsilon",
        description="Specifies the maximum allowable epsilon value. If the training process exceeds this threshold, "
        "it will be terminated early. Only model checkpoints with epsilon values below this limit will be "
        "retained. If not provided, the training will proceed without early termination based on "
        "epsilon constraints.",
        ge=0.0,
        le=10000.0,
    )
    noise_multiplier: float = Field(
        default=1.5,
        alias="noiseMultiplier",
        description="The ratio of the standard deviation of the Gaussian noise to the L2-sensitivity of the function "
        "to which the noise is added (How much noise to add).\n",
        ge=0.0,
        le=10000.0,
    )
    max_grad_norm: float = Field(
        default=1.0,
        alias="maxGradNorm",
        description="The maximum norm of the per-sample gradients for training the model with differential privacy.\n",
        ge=0.0,
        le=10000.0,
    )
    delta: float = Field(
        default=1e-5,
        description="The delta value for differential privacy. It is the probability of the privacy guarantee not "
        "holding. The smaller the delta, the more confident you can be that the privacy guarantee holds.",
        ge=0.0,
        le=1.0,
    )
