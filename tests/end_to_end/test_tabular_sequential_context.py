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

import numpy as np
import pandas as pd
import pytest

from mostlyai.engine.domain import ModelEncodingType
from mostlyai.engine import analyze, encode, split
from mostlyai.engine._tabular.generation import generate
from mostlyai.engine._tabular.training import train


class TestThreeTableSetup:
    @pytest.fixture(scope="class")
    def three_table_setup(self):
        num_records = 1000
        # n/5 seqs [4,3,2,1]; n/5 seqs [3,2,1]; n/5 seqs[2,1]; n/5 seqs [1]; n/5 seqs []
        sequences = (
            [list(range(4, 0, -1)) for _ in range(num_records // 5)]
            + [list(range(3, 0, -1)) for _ in range(num_records // 5)]
            + [list(range(2, 0, -1)) for _ in range(num_records // 5)]
            + [list(range(1, 0, -1)) for _ in range(num_records // 5)]
            + [[] for _ in range(num_records // 5)]
        )

        # shuffle sequences
        np.random.shuffle(sequences)

        df_parent = pd.DataFrame({"id": range(num_records)})
        df_ctx_child = pd.DataFrame({"id": range(num_records), "events": sequences})

        df_ctx = df_parent.merge(df_ctx_child, left_on="id", right_on="id")

        df_tgt = (
            pd.DataFrame({"id": range(num_records), "events": sequences})
            .explode("events")
            .dropna(subset=["events"])
            .reset_index(drop=True)
        )

        yield df_ctx, df_tgt

    @pytest.fixture(scope="class")
    def scp_train(self, tmp_path_factory, three_table_setup):
        df_ctx, df_tgt = three_table_setup
        workspace_dir = tmp_path_factory.mktemp("ws")
        split(
            tgt_data=df_tgt,
            tgt_context_key="id",
            tgt_encoding_types={
                "events": ModelEncodingType.tabular_categorical,
            },
            ctx_data=df_ctx,
            ctx_primary_key="id",
            ctx_encoding_types={
                "events": ModelEncodingType.tabular_categorical,
            },
            workspace_dir=workspace_dir,
        )

        analyze(workspace_dir=workspace_dir)
        encode(workspace_dir=workspace_dir)
        train(max_epochs=50, workspace_dir=workspace_dir)

        generate(
            ctx_data=df_ctx,
            workspace_dir=workspace_dir,
        )
        syn_data_path = workspace_dir / "SyntheticData"
        df_syn = pd.read_parquet(syn_data_path)
        yield df_ctx, df_syn

    def test_amount_of_seqlen_match(self, scp_train):
        df_ctx, df_syn = scp_train
        df_ctx = df_ctx.set_index("id")
        df_syn = df_syn.set_index("id")
        df = pd.DataFrame()
        df["events_ctx"] = df_ctx["events"].apply(len)
        df["events_syn"] = df_syn["events"].groupby("id").apply(len)
        df["events_syn"] = df["events_syn"].fillna(0)

        total_matches = (df.events_ctx == df.events_syn).sum()
        total_records = df.shape[0]
        assert (total_matches / total_records) > 0.75

    def test_match_events(self, scp_train):
        df_ctx, df_syn = scp_train

        # ctx has exactly the same events as tgt (df_syn should match them)
        df_ctx["events"] = df_ctx["events"].apply(list)
        df_ctx.rename(columns={"events": "events_ctx"}, inplace=True)
        df_syn["events"] = df_syn["events"].astype("int")
        df_syn.rename(columns={"events": "events_syn"}, inplace=True)
        df_syn = df_syn.groupby("id").agg(list).reset_index()
        df_syn = df_syn.merge(df_ctx, on="id").astype(str)
        acc = (df_syn["events_ctx"] == df_syn["events_syn"]).mean()
        assert acc > 0.7


class TestFourTableSetup:
    @pytest.fixture(scope="class")
    def four_table_setup(self):
        num_records = 1000
        # scp1: n/2 [1,2,3] [1,2,3]; n/2 [4,5,6] [4,5,6]
        # scp2: n/2 [0,0,0] [0,0,0]; n/2 [6,6,6] [6,6,6]
        # tgt: row-wise sum of scp1 and scp2
        t1_sequences = [np.array([1, 2, 3]) for _ in range(num_records // 2)] + [
            np.array([4, 5, 6]) for _ in range(num_records // 2)
        ]
        t2_sequences = [np.array([0, 0, 0]) for _ in range(num_records // 2)] + [
            np.array([6, 6, 6]) for _ in range(num_records // 2)
        ]

        # shuffle sequences
        np.random.shuffle(t1_sequences)
        np.random.shuffle(t2_sequences)

        df_parent = pd.DataFrame({"flat::id": range(num_records)})
        df_scp1 = pd.DataFrame(
            {
                "scp1::id": range(num_records),
                "scp1::col1": t1_sequences,
                "scp1::col2": t1_sequences,
            }
        )
        df_scp2 = pd.DataFrame(
            {
                "scp2::id": range(num_records),
                "scp2::col1": t2_sequences,
                "scp2::col2": t2_sequences,
            }
        )

        df_ctx = df_parent.merge(df_scp1, left_on="flat::id", right_on="scp1::id").merge(
            df_scp2, left_on="flat::id", right_on="scp2::id"
        )

        df_tgt = (
            pd.DataFrame(
                {
                    "id": range(num_records),
                    "sums": df_ctx.apply(
                        lambda row: sum(
                            row[c]
                            for c in [
                                "scp1::col1",
                                "scp1::col2",
                                "scp2::col1",
                                "scp2::col2",
                            ]
                        ),
                        axis=1,
                    ),
                }
            )
            .explode("sums")
            .reset_index(drop=True)
        )

        yield df_ctx, df_tgt

    @pytest.fixture(scope="class")
    def scp_train(self, tmp_path_factory, four_table_setup):
        df_ctx, df_tgt = four_table_setup
        workspace_dir = tmp_path_factory.mktemp("ws")
        split(
            tgt_data=df_tgt,
            tgt_context_key="id",
            tgt_encoding_types={
                "sums": ModelEncodingType.tabular_categorical,
            },
            ctx_data=df_ctx,
            ctx_primary_key="flat::id",
            ctx_encoding_types={
                "scp1::col1": ModelEncodingType.tabular_categorical,
                "scp1::col2": ModelEncodingType.tabular_categorical,
                "scp2::col1": ModelEncodingType.tabular_categorical,
                "scp2::col2": ModelEncodingType.tabular_categorical,
            },
            workspace_dir=workspace_dir,
        )

        analyze(workspace_dir=workspace_dir)
        encode(workspace_dir=workspace_dir)
        train(max_epochs=10, workspace_dir=workspace_dir)

        generate(
            ctx_data=df_ctx,
            workspace_dir=workspace_dir,
        )
        syn_data_path = workspace_dir / "SyntheticData"
        df_syn = pd.read_parquet(syn_data_path)

        yield df_ctx, df_syn, df_tgt

    def test_match_sums(self, scp_train):
        df_ctx, df_syn, df_tgt = scp_train

        expected_sums = df_tgt.sums
        syn = df_syn.sort_values(by="id", kind="stable").reset_index(drop=True)
        syn_sums = syn.sums.astype("int64")

        # generated sums should match with a margin
        acc = (syn_sums == expected_sums).mean()
        assert acc > 0.8


class TestFlatWithSequentialContext:
    @pytest.fixture(scope="class")
    def flat_seq_setup(self):
        num_records = 1000
        sequences = [list(range(np.random.randint(1, 5))) for _ in range(num_records)]
        np.random.shuffle(sequences)
        df_ctx = pd.DataFrame({"id": range(num_records), "events": sequences})
        df_tgt = pd.DataFrame(
            {
                "id": range(num_records),
                "value": [sum(seq) for seq in sequences],
                "category": ["high" if sum(seq) > 2 else "low" for seq in sequences],
            }
        )

        yield df_ctx, df_tgt

    @pytest.fixture(scope="class")
    def scp_train(self, tmp_path_factory, flat_seq_setup):
        df_ctx, df_tgt = flat_seq_setup
        workspace_dir = tmp_path_factory.mktemp("ws")

        split(
            tgt_data=df_tgt,
            tgt_context_key="id",
            tgt_encoding_types={
                "value": ModelEncodingType.tabular_numeric_auto,
                "category": ModelEncodingType.tabular_categorical,
            },
            ctx_data=df_ctx,
            ctx_primary_key="id",
            ctx_encoding_types={
                "events": ModelEncodingType.tabular_categorical,
            },
            workspace_dir=workspace_dir,
        )

        analyze(workspace_dir=workspace_dir)
        encode(workspace_dir=workspace_dir)
        train(max_epochs=10, workspace_dir=workspace_dir)

        generate(
            ctx_data=df_ctx,
            workspace_dir=workspace_dir,
        )
        syn_data_path = workspace_dir / "SyntheticData"
        df_syn = pd.read_parquet(syn_data_path)

        yield df_ctx, df_syn, df_tgt

    def test_category_accuracy(self, scp_train):
        df_ctx, df_syn, df_tgt = scp_train

        # test if synthetic categories match expected categories based on sequence sums
        expected_categories = df_ctx["events"].apply(lambda x: "high" if sum(x) > 2 else "low")
        syn_categories = df_syn["category"]
        accuracy = (expected_categories == syn_categories).mean()
        assert accuracy > 0.7
