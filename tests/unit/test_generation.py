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

import pandas as pd
import pytest
import torch

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
from mostlyai.engine._tabular.generation import (
    RareCategoryReplacementMethod,
    RebalancingConfig,
    _batch_df,
    _deepmerge,
    _fix_imputation_probs,
    _fix_rare_token_probs,
    _fix_rebalancing_probs,
    _reshape_pt_to_pandas,
    _resolve_gen_column_order,
    _translate_fixed_probs,
)
from mostlyai.engine.domain import FairnessConfig, ImputationConfig, ModelEncodingType


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
        fixed_probs = _fix_rare_token_probs(stats)
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

        fixed_probs = _fix_rare_token_probs(
            stats=get_stats(),
            rare_category_replacement_method=rare_category_replacement_method,
        )
        expected = {"column": {CATEGORICAL_SUB_COL_SUFFIX: {0: 0.0}}} if do_fix else {}
        assert fixed_probs == expected


class TestFixImputationProbs:
    def get_stats(self):
        return {
            "columns": {
                "cat": {
                    "codes": {"_RARE_": 0, "<<NULL>>": 1, "<=50K": 2, ">50K": 3},
                    "encoding_type": ModelEncodingType.tabular_categorical.value,
                },
                "dt": {
                    "has_nan": True,
                    "encoding_type": ModelEncodingType.tabular_datetime.value,
                },
                "ll": {
                    "has_nan": True,
                    "encoding_type": ModelEncodingType.tabular_lat_long.value,
                },
                "num_digit": {
                    "has_nan": True,
                    "encoding_type": ModelEncodingType.tabular_numeric_digit.value,
                },
                "num_discrete": {
                    "codes": {"_RARE_": 0, "<<NULL>>": 1, "17.0": 2},
                    "encoding_type": ModelEncodingType.tabular_numeric_discrete.value,
                },
                "num_binned": {
                    "codes": {"<<UNK>>": 0, "<<NULL>>": 1, "<<MIN>>": 2, "<<MAX>>": 3},
                    "encoding_type": ModelEncodingType.tabular_numeric_binned.value,
                },
                "text": {
                    "has_na": True,
                    "encoding_type": ModelEncodingType.language_text.value,
                },
                "no_nulls": {
                    "has_nan": False,
                    "encoding_type": ModelEncodingType.tabular_numeric_digit.value,
                },
            }
        }

    def test_non_existing_columns(self):
        fixed_probs = _fix_imputation_probs(
            stats=self.get_stats(), imputation=ImputationConfig(columns=["missing_column"])
        )
        assert fixed_probs == {}

    def test_impute_all_encoding_types(self):
        imputation = ImputationConfig(
            columns=[
                "cat",
                "dt",
                "ll",
                "num_digit",
                "num_discrete",
                "num_binned",
                "text",
            ]
        )
        fixed_probs = _fix_imputation_probs(stats=self.get_stats(), imputation=imputation)
        assert fixed_probs == {
            "cat": {CATEGORICAL_SUB_COL_SUFFIX: {0: 0.0, 1: 0.0}},
            "dt": {"nan": {1: 0.0}},
            "ll": {"nan": {1: 0.0}},
            "num_digit": {"nan": {1: 0.0}},
            "num_discrete": {NUMERIC_DISCRETE_SUB_COL_SUFFIX: {1: 0.0}},
            "num_binned": {NUMERIC_BINNED_SUB_COL_SUFFIX: {1: 0.0}},
            "text": {"na": {1: 0.0}},
        }

    def test_impute_column_with_no_nulls(self):
        fixed_probs = _fix_imputation_probs(stats=self.get_stats(), imputation=ImputationConfig(columns=["no_nulls"]))
        assert fixed_probs == {}


class TestFixRebalancingProbs:
    def get_stats(self) -> dict:
        return {
            "columns": {
                "income": {
                    "encoding_type": ModelEncodingType.tabular_categorical.value,
                    "codes": {"<=50K": 1, ">50K": 2},
                }
            }
        }

    def test_non_existing_columns(self):
        fixed_probs = _fix_rebalancing_probs(
            stats=self.get_stats(), rebalancing=RebalancingConfig(column="income!", probabilities={"<=50K": 0.3})
        )
        assert fixed_probs == {}

    def test_non_existing_category(self):
        fixed_probs = _fix_rebalancing_probs(
            stats=self.get_stats(), rebalancing=RebalancingConfig(column="income", probabilities={">100K": 0.3})
        )
        assert fixed_probs == {}


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
        fixed_probs_model = _translate_fixed_probs(fixed_probs, stats)
        assert fixed_probs_model == {"tgt:t0/c0__cat": {0: 0.0}}


class TestDeepmerge:
    def test(self):
        fixed_probs_1 = {
            "column_1": {
                "cat_1": 0.1,
                "cat_2": 0.2,
            },
            "column_2": {"cat_1": 0.3},
        }
        fixed_probs_2 = {
            "column_1": {"cat_1": 0.4, "cat_3": 0.5},
            "column_3": {"cat_1": 0.6},
        }
        fixed_probs = _deepmerge(fixed_probs_1, fixed_probs_2)
        assert fixed_probs == {
            "column_1": {"cat_1": 0.1, "cat_2": 0.2, "cat_3": 0.5},
            "column_2": {"cat_1": 0.3},
            "column_3": {"cat_1": 0.6},
        }


def test_batch_df():
    df = pd.Series(range(109)).to_frame()
    no_of_batches = 10
    df = _batch_df(df, no_of_batches)
    # assert same behaviour as previous logic:
    # dask.dataframe.from_pandas(df, npartitions=no_of_batches)
    assert list(df["__BATCH"].value_counts()) == [11] * 9 + [10]


class TestReshapeToPandas:
    def test_reshape_empty(self):
        gen_steps = []
        sub_cols = ["a", "b"]
        keys = []
        key_name = "id"
        df = _reshape_pt_to_pandas(data=gen_steps, sub_cols=sub_cols, keys=keys, key_name=key_name)
        assert list(df.columns) == [key_name] + sub_cols
        assert df.empty

    def test_reshape_sequential(self):
        gen_steps = [
            torch.tensor([[[11.0], [12.0]], [[13.0], [14.0]]]),
            torch.tensor([[[21.0], [22.0]], [[23.0], [24.0]]]),
            torch.tensor([[[32.0], [34.0]]]),
        ]
        sub_cols = ["a", "b"]
        keys = [
            pd.Series(["u0", "u1"]),
            pd.Series(["u0", "u1"]),
            pd.Series(["u0"]),
        ]
        key_name = "id"
        df = _reshape_pt_to_pandas(data=gen_steps, sub_cols=sub_cols, keys=keys, key_name=key_name)
        assert list(df.columns) == [key_name] + sub_cols
        assert df.shape == (
            sum([step.shape[0] for step in gen_steps]),
            len(sub_cols) + 1,
        )
        assert pd.api.types.is_integer_dtype(df["a"])


class TestResolveGenColumnOrder:
    @pytest.fixture
    def stats_and_cards(self):
        col_stats = {
            "age": {
                ARGN_PROCESSOR: "tgt",
                ARGN_TABLE: "t0",
                ARGN_COLUMN: "c0",
            },
            "sex": {
                ARGN_PROCESSOR: "tgt",
                ARGN_TABLE: "t1",
                ARGN_COLUMN: "c1",
            },
            "income": {
                ARGN_PROCESSOR: "tgt",
                ARGN_TABLE: "t2",
                ARGN_COLUMN: "c2",
            },
        }
        flat_cardinalities = {
            "tgt:t0/c0__E0": 10,
            "tgt:t0/c0__E1": 9,
            "tgt:t1/c1__cat": 3,
            "tgt:t2/c2__cat": 3,
        }
        seq_cardinalities = flat_cardinalities | {
            "tgt:/__sidx__E0": 8,
            "tgt:/__slen__E0": 9,
        }
        return col_stats, flat_cardinalities, seq_cardinalities

    @pytest.fixture
    def stats_and_cards_for_fairness(self):
        col_stats = {
            "c0": {
                ARGN_PROCESSOR: "tgt",
                ARGN_TABLE: "t0",
                ARGN_COLUMN: "c0",
            },
            "c1": {
                ARGN_PROCESSOR: "tgt",
                ARGN_TABLE: "t0",
                ARGN_COLUMN: "c1",
            },
            "c2": {
                ARGN_PROCESSOR: "tgt",
                ARGN_TABLE: "t0",
                ARGN_COLUMN: "c2",
            },
            "c3": {
                ARGN_PROCESSOR: "tgt",
                ARGN_TABLE: "t0",
                ARGN_COLUMN: "c3",
            },
        }
        cardinalities = {
            "tgt:t0/c0__cat": 10,
            "tgt:t0/c1__cat": 4,
            "tgt:t0/c2__cat": 3,
            "tgt:t0/c3__cat": 7,
        }
        return col_stats, cardinalities

    @pytest.mark.parametrize("model_type", ["flat", "sequential"])
    def test_flexible_order_model(self, model_type, stats_and_cards):
        col_stats, flat_cardinalities, seq_cardinalities = stats_and_cards
        cardinalities = {"flat": flat_cardinalities, "sequential": seq_cardinalities}[model_type]
        gen_column_order = _resolve_gen_column_order(
            column_stats=col_stats,
            cardinalities=cardinalities,
        )
        flat_expected = ["tgt:t0/c0", "tgt:t1/c1", "tgt:t2/c2"]
        seq_expected = ["tgt:/"] + flat_expected  # SLEN/SIDX column is always first for sequential model
        assert gen_column_order == {"flat": flat_expected, "sequential": seq_expected}[model_type]

    @pytest.mark.parametrize("model_type", ["flat", "sequential"])
    @pytest.mark.parametrize("rebalance_column", ["sex", "I don't exist"])
    def test_rebalancing_only(self, model_type, rebalance_column, stats_and_cards):
        col_stats, flat_cardinalities, seq_cardinalities = stats_and_cards
        cardinalities = {"flat": flat_cardinalities, "sequential": seq_cardinalities}[model_type]
        rebalancing = RebalancingConfig(column=rebalance_column, probabilities={})
        gen_column_order = _resolve_gen_column_order(
            column_stats=col_stats,
            cardinalities=cardinalities,
            rebalancing=rebalancing,
        )
        flat_expected = ["tgt:t1/c1", "tgt:t0/c0", "tgt:t2/c2"]
        if rebalance_column == "I don't exist":
            flat_expected = sorted(flat_expected)
        seq_expected = ["tgt:/"] + flat_expected
        assert gen_column_order == {"flat": flat_expected, "sequential": seq_expected}[model_type]

    @pytest.mark.parametrize("model_type", ["flat", "sequential"])
    def test_imputation_only(self, model_type, stats_and_cards):
        col_stats, flat_cardinalities, seq_cardinalities = stats_and_cards
        cardinalities = {"flat": flat_cardinalities, "sequential": seq_cardinalities}[model_type]
        imputation = ImputationConfig(columns=["age", "sex", "I don't exist"])
        gen_column_order = _resolve_gen_column_order(
            column_stats=col_stats,
            cardinalities=cardinalities,
            imputation=imputation,
        )
        flat_expected = ["tgt:t2/c2", "tgt:t0/c0", "tgt:t1/c1"]
        seq_expected = ["tgt:/"] + flat_expected
        assert gen_column_order == {"flat": flat_expected, "sequential": seq_expected}[model_type]

    @pytest.mark.parametrize("model_type", ["flat", "sequential"])
    def test_imputation_and_rebalancing(self, model_type, stats_and_cards):
        col_stats, flat_cardinalities, seq_cardinalities = stats_and_cards
        cardinalities = {"flat": flat_cardinalities, "sequential": seq_cardinalities}[model_type]
        imputation = ImputationConfig(columns=["age", "sex"])
        rebalancing = RebalancingConfig(column="sex", probabilities={})
        gen_column_order = _resolve_gen_column_order(
            column_stats=col_stats,
            cardinalities=cardinalities,
            imputation=imputation,
            rebalancing=rebalancing,
        )
        flat_expected = [
            "tgt:t1/c1",
            "tgt:t2/c2",
            "tgt:t0/c0",
        ]  # rebalancing takes priority before imputation
        seq_expected = ["tgt:/"] + flat_expected
        assert gen_column_order == {"flat": flat_expected, "sequential": seq_expected}[model_type]

    def test_seed_data(self, stats_and_cards):
        # sample seed only applies to flat model
        col_stats, flat_cardinalities, _ = stats_and_cards
        seed_data = pd.DataFrame(columns=["sex", "age", "I_don't_exist"])
        gen_column_order = _resolve_gen_column_order(
            column_stats=col_stats,
            cardinalities=flat_cardinalities,
            seed_data=seed_data,
        )
        assert gen_column_order == ["tgt:t1/c1", "tgt:t0/c0", "tgt:t2/c2"]

    @pytest.mark.parametrize(
        "seed_cols, rebalancing_col, fairness_sensitive_cols, fairness_target_col, imputation_cols, expected_order",
        [
            # seed + fairness
            (["c1"], None, ["c3"], "c2", None, ["c1", "c3", "c0", "c2"]),
            # seed on a fairness sensitive column
            (["c1"], None, ["c2", "c1"], "c0", None, ["c1", "c2", "c3", "c0"]),
            # rebalancing + fairness
            (None, "c1", ["c3"], "c2", None, ["c1", "c3", "c0", "c2"]),
            # rebalancing on a fairness sensitive column
            (None, "c1", ["c2, c1"], "c0", None, ["c1", "c2", "c3", "c0"]),
            # rebalancing and imputation on the same column
            (None, "c2", None, None, ["c2"], ["c2", "c0", "c1", "c3"]),
            # imputation + fairness
            (None, None, ["c3"], "c0", ["c2"], ["c3", "c1", "c2", "c0"]),
            # imputation on a fairness sensitive column
            (None, None, ["c2", "c3"], "c0", ["c2"], ["c3", "c2", "c1", "c0"]),
            # imputation on the fairness target column
            (None, None, ["c3", "c2"], "c0", ["c0", "c1"], ["c3", "c2", "c1", "c0"]),
        ],
    )
    def test_combinations(
        self,
        stats_and_cards_for_fairness,
        seed_cols,
        rebalancing_col,
        fairness_sensitive_cols,
        fairness_target_col,
        imputation_cols,
        expected_order,
    ):
        col_stats, cardinalities = stats_and_cards_for_fairness

        seed = pd.DataFrame(columns=seed_cols) if seed_cols else None
        rebalancing = (
            RebalancingConfig(column=rebalancing_col, probabilities={"cat_1": 0.5}) if rebalancing_col else None
        )
        if fairness_sensitive_cols and fairness_target_col:
            fairness = FairnessConfig(
                sensitive_columns=fairness_sensitive_cols,
                target_column=fairness_target_col,
            )
        else:
            fairness = None
        if imputation_cols:
            imputation = ImputationConfig(columns=imputation_cols)
        else:
            imputation = None

        gen_column_order = _resolve_gen_column_order(
            column_stats=col_stats,
            cardinalities=cardinalities,
            seed_data=seed,
            rebalancing=rebalancing,
            fairness=fairness,
            imputation=imputation,
        )
        gen_column_order = [col.split("/")[-1] for col in gen_column_order]
        assert gen_column_order == expected_order
