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

import shutil
from itertools import chain

import pandas as pd
import pytest
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from mostlyai.engine import split
from mostlyai.engine._workspace import Workspace
from mostlyai.engine._language.generation import generate
from mostlyai.engine._language.tokenizer_utils import tokenize_fn
from mostlyai.engine._language.encoding import encode
from mostlyai.engine.analysis import analyze
from mostlyai.engine._common import TEMPORARY_PRIMARY_KEY
from mostlyai.engine._encoding_types.language.categorical import CATEGORICAL_UNKNOWN_TOKEN
from mostlyai.engine._language.lstm import LSTMFromScratchConfig
from mostlyai.engine._language.tokenizer_utils import MostlyDataCollatorForLanguageModeling
from mostlyai.engine._language.training import (
    train,
    _gpu_estimate_max_batch_size,
    _physical_batch_size_heuristic,
    _gradient_accumulation_steps_heuristic,
)
from mostlyai.engine.domain import (
    ModelEncodingType,
    ModelStateStrategy,
    DifferentialPrivacyConfig,
    RareCategoryReplacementMethod,
)


def prepare_encoded_dataset(data: pd.DataFrame, workspace_dir, tgt_encoding_types, ctx_encoding_types=None):
    tbl_pk = TEMPORARY_PRIMARY_KEY
    data[tbl_pk] = list(range(data.shape[0]))
    ctx_columns = [tbl_pk, *[key for key in ctx_encoding_types.keys()]] if ctx_encoding_types else [tbl_pk]
    tgt_columns = [tbl_pk, *[key for key in tgt_encoding_types.keys()]]
    ctx_df = data[ctx_columns]
    tgt_df = data[tgt_columns]
    split(
        tgt_data=tgt_df,
        tgt_context_key=tbl_pk,
        tgt_encoding_types=tgt_encoding_types,
        ctx_data=ctx_df,
        ctx_primary_key=tbl_pk,
        ctx_encoding_types=ctx_encoding_types,
        workspace_dir=workspace_dir,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)


@pytest.fixture(scope="session")
def encoded_text_dataset(tmp_path_factory):
    workspace_dir = tmp_path_factory.mktemp("ws")
    no_of_records = 20
    data = pd.DataFrame(
        {
            "gender": ["m", "f", "x", pd.NA] * int(no_of_records / 4),
            "bio": chain(*[[f"Joe {i}", f"Anna {i}", pd.NA, pd.NA] for i in range(int(no_of_records / 4))]),
        }
    )
    ctx_encoding_types = {"gender": ModelEncodingType.tabular_categorical.value}
    tgt_encoding_types = {"bio": ModelEncodingType.language_text.value}
    prepare_encoded_dataset(data, workspace_dir, tgt_encoding_types, ctx_encoding_types)
    return workspace_dir


@pytest.fixture(scope="session")
def single_record_text_dataset(tmp_path_factory):
    workspace_dir = tmp_path_factory.mktemp("ws-single-record")
    data = pd.DataFrame({"gender": ["m"], "bio": ["Joe"]})
    ctx_encoding_types = {"gender": ModelEncodingType.tabular_categorical.value}
    tgt_encoding_types = {"bio": ModelEncodingType.language_text.value}
    prepare_encoded_dataset(data, workspace_dir, tgt_encoding_types, ctx_encoding_types)
    return workspace_dir


@pytest.fixture(scope="session")
def null_only_text_dataset(tmp_path_factory):
    workspace_dir = tmp_path_factory.mktemp("ws-null-only")
    data = pd.DataFrame({"nulls": [pd.NA] * 10})
    tgt_encoding_types = {"nulls": ModelEncodingType.language_text.value}
    prepare_encoded_dataset(data, workspace_dir, tgt_encoding_types)
    return workspace_dir


@pytest.fixture(scope="session")
def tgt_only_text_dataset(tmp_path_factory):
    workspace_dir = tmp_path_factory.mktemp("ws-tgt-only")
    tgt_df = pd.DataFrame({"bio": ["Joe"] * 20})
    split(
        tgt_data=tgt_df,
        workspace_dir=workspace_dir,
        model_type="LANGUAGE",
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    return workspace_dir


def test_tgt_only(tgt_only_text_dataset):
    workspace_dir = tgt_only_text_dataset
    train(workspace_dir=workspace_dir, model=LSTMFromScratchConfig.model_id)
    generate(workspace_dir=workspace_dir, sample_size=10)

    syn_data_path = workspace_dir / "SyntheticData"
    syn = pd.read_parquet(syn_data_path)
    assert len(syn) == 10
    assert set(syn.columns) == {"bio"}
    assert str(syn["bio"].dtype).startswith("string")


@pytest.mark.parametrize(
    ("model_name", "sampling_temperature"),
    [
        ("amd/AMD-Llama-135m", 1.0),
        ("HuggingFaceTB/SmolLM-135M", 1.0),
        ("HuggingFaceTB/SmolLM-135M", 0.0),
        (LSTMFromScratchConfig.model_id, 1.0),
    ],
)
def test_language_with_context(encoded_text_dataset, model_name, sampling_temperature):
    workspace_dir = encoded_text_dataset
    ctx_data = pd.read_parquet(workspace_dir / "OriginalData" / "ctx-data")
    train(workspace_dir=workspace_dir, model=model_name)
    generate(
        workspace_dir=workspace_dir,
        ctx_data=ctx_data,
        sampling_temperature=sampling_temperature,
    )

    syn_data = pd.read_parquet(workspace_dir / "SyntheticData")
    assert len(syn_data) == 20
    assert set(syn_data.columns) == {"bio", "__primary_key"}
    assert str(syn_data["bio"].dtype).startswith("string")


@pytest.mark.parametrize(
    ("model_name", "dp_max_epsilon"),
    [
        ("amd/AMD-Llama-135m", None),
        (LSTMFromScratchConfig.model_id, 40),
    ],
)
def test_language_with_dp(encoded_text_dataset, model_name, dp_max_epsilon):
    workspace_dir = encoded_text_dataset
    ctx_data = pd.read_parquet(workspace_dir / "OriginalData" / "ctx-data")
    differential_privacy = DifferentialPrivacyConfig(
        max_epsilon=dp_max_epsilon,
        noise_multiplier=0.2,
    )
    train(workspace_dir=workspace_dir, model=model_name, differential_privacy=differential_privacy, batch_size=16)
    generate(
        workspace_dir=workspace_dir,
        ctx_data=ctx_data,
    )

    if dp_max_epsilon is not None:
        progress_messages = pd.read_csv(workspace_dir / "ModelStore" / "model-data" / "progress-messages.csv")
        assert progress_messages[progress_messages["is_checkpoint"] == 1].iloc[-1]["dp_eps"] <= dp_max_epsilon
    syn_data = pd.read_parquet(workspace_dir / "SyntheticData")
    assert len(syn_data) == 20
    assert set(syn_data.columns) == {"bio", "__primary_key"}
    assert str(syn_data["bio"].dtype).startswith("string")


@pytest.mark.parametrize(
    "model_name",
    [
        "amd/AMD-Llama-135m",
        LSTMFromScratchConfig.model_id,
    ],
)
def test_training_strategy(encoded_text_dataset, model_name):
    workspace_dir = encoded_text_dataset
    workspace = Workspace(workspace_dir)
    train(workspace_dir=workspace_dir, model=model_name, max_epochs=1, model_state_strategy=ModelStateStrategy.reset)
    progress_reset = pd.read_csv(workspace.model_progress_messages_path)

    train(workspace_dir=workspace_dir, model=model_name, max_epochs=1, model_state_strategy=ModelStateStrategy.reuse)
    progress_reuse = pd.read_csv(workspace.model_progress_messages_path)
    assert not progress_reuse["epoch"].duplicated().any()
    # progress should be different but with the same shape
    assert progress_reset.shape == progress_reuse.shape
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(progress_reset, progress_reuse)

    train(workspace_dir=workspace_dir, model=model_name, max_epochs=2, model_state_strategy=ModelStateStrategy.resume)
    progress_resume = pd.read_csv(workspace.model_progress_messages_path)
    assert not progress_resume["epoch"].duplicated().any()
    # training resumed from epoch 1 and only appended a new line for epoch 2
    # so the progress should be identical except for the last row
    pd.testing.assert_frame_equal(progress_reuse, progress_resume.iloc[:-1])

    # in case the checkpoint doesn't exist, it should still work but change to reset strategy
    shutil.rmtree(workspace_dir / "ModelStore" / "model-data")
    train(workspace_dir=workspace_dir, model=model_name, max_epochs=1, model_state_strategy=ModelStateStrategy.resume)
    progress_resume_without_checkpoint = pd.read_csv(workspace.model_progress_messages_path)
    # it's actually a fresh training, so the progress will look different
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(progress_resume.iloc[:2], progress_resume_without_checkpoint.iloc[:2])


class TestConditionalGeneration:
    def test_lstm(self, tmp_path_factory):
        workspace_dir = tmp_path_factory.mktemp("ws")
        no_of_records = 2000
        data = pd.DataFrame(
            {
                "gender": ["m", "f"] * int(no_of_records / 2),
                "country": ["USA"] * int(no_of_records),
                "bio": ["Joe", "Anna"] * int(no_of_records / 2),
            }
        )
        seed_size = 100
        # re-balance towards the females, test non-existing column and token, test nulls
        sample_seed = pd.DataFrame(
            {
                "gender": ["f"] * (seed_size - 5) + ["Żf", "国", None, pd.NA, np.nan],
                "country": ["USA"] * (seed_size - 5) + ["USA USA", "Poland", None, pd.NA, np.nan],
                "nonexisting": ["x"] * seed_size,
            }
        )
        tgt_encoding_types = {
            "gender": ModelEncodingType.language_text.value,
            "country": ModelEncodingType.language_categorical.value,
            "bio": ModelEncodingType.language_text.value,
        }
        prepare_encoded_dataset(data, workspace_dir, tgt_encoding_types)
        ctx_data = pd.read_parquet(workspace_dir / "OriginalData" / "ctx-data")
        train(workspace_dir=workspace_dir, model=LSTMFromScratchConfig.model_id, max_training_time=0.5, batch_size=32)
        generate(
            workspace_dir=workspace_dir,
            ctx_data=ctx_data,
            seed_data=sample_seed,
        )

        syn_data = pd.read_parquet(workspace_dir / "SyntheticData")
        assert len(syn_data) == seed_size  # seed dictates the sample size
        assert all(syn_data["gender"][:95] == "f")  # test for regular tokens
        assert syn_data["gender"][95] == "f"  # unknown token is skipped, known token remains
        assert all(syn_data["gender"][96:] == "")  # nulls are skipped, if tokenizer can't express them
        assert all(syn_data["country"][:95] == "USA")  # test for regular categories
        assert syn_data["country"][95] == "USA USA"  # unseen category should persist, if tokenizer can express it
        assert (
            syn_data["country"][96] == "oand"
        )  # some unseen categories can be expressed only partially with the tokenizer
        assert all(syn_data["country"][97:] == "")  # nulls are skipped, if tokenizer can't express them
        n_annas = syn_data["bio"].str.startswith("Anna").sum()
        assert n_annas / len(syn_data) > 0.6  # seed re-balances towards females, thus Anna should be more frequent
        assert set(syn_data.columns) == {
            "__primary_key",
            "gender",
            "country",
            "bio",
        }
        assert str(syn_data["gender"].dtype).startswith("string")
        assert str(syn_data["country"].dtype).startswith("string")
        assert str(syn_data["bio"].dtype).startswith("string")

    def test_hf_model(self, tmp_path_factory):
        # this is a smoke test for the HF model, mainly to ensure seeded values are preserved in the final output
        # for more comprehensive test that also covers some quality checks, see test_lstm
        workspace_dir = tmp_path_factory.mktemp("ws")
        no_of_records = 200
        data = pd.DataFrame({"country": ["USA", "Portugal", "Austria", "Poland"] * int(no_of_records / 4)})
        tgt_encoding_types = {"country": ModelEncodingType.language_categorical.value}
        seed_data = pd.DataFrame(
            {
                "country": ["Greece", "Italy", "Spain", pd.NA],
            }
        )
        prepare_encoded_dataset(data, workspace_dir, tgt_encoding_types)
        ctx_data = pd.read_parquet(workspace_dir / "OriginalData" / "ctx-data")
        train(workspace_dir=workspace_dir, model="HuggingFaceTB/SmolLM-135M")
        generate(
            workspace_dir=workspace_dir,
            seed_data=seed_data,
            ctx_data=ctx_data,
        )
        syn_data = pd.read_parquet(workspace_dir / "SyntheticData")
        assert len(syn_data) == len(seed_data)
        pd.testing.assert_series_equal(syn_data["country"], seed_data["country"], check_dtype=False)


@pytest.mark.parametrize(
    "model_name",
    [
        "HuggingFaceTB/SmolLM-135M",
        LSTMFromScratchConfig.model_id,
    ],
)
def test_language_skipping_train(single_record_text_dataset, model_name):
    workspace_dir = single_record_text_dataset
    ctx_data = pd.read_parquet(workspace_dir / "OriginalData" / "ctx-data")
    train(workspace_dir=workspace_dir, model=model_name)
    generate(workspace_dir=workspace_dir, ctx_data=ctx_data)

    syn_data = pd.read_parquet(workspace_dir / "SyntheticData")
    assert len(syn_data) == 1
    assert set(syn_data.columns) == {"bio", "__primary_key"}
    assert str(syn_data["bio"].dtype).startswith("string")


def test_empty_generation_context(encoded_text_dataset):
    model_name = LSTMFromScratchConfig.model_id
    workspace_dir = encoded_text_dataset
    ctx_data_path = workspace_dir / "OriginalData" / "ctx-data"
    ctx_df = pd.read_parquet(ctx_data_path)
    empty_ctx_df = pd.DataFrame(columns=ctx_df.columns).astype(ctx_df.dtypes)
    train(workspace_dir=workspace_dir, model=model_name)
    generate(workspace_dir=workspace_dir, ctx_data=empty_ctx_df)
    syn_data = pd.read_parquet(workspace_dir / "SyntheticData")
    assert len(syn_data) == 0
    assert set(syn_data.columns) == {"bio", "__primary_key"}


def test_null_only_text_training(null_only_text_dataset):
    workspace_dir = null_only_text_dataset
    model_name = LSTMFromScratchConfig.model_id
    train(workspace_dir=workspace_dir, model=model_name)
    generate(workspace_dir=workspace_dir, ctx_data=None)

    syn_data = pd.read_parquet(workspace_dir / "SyntheticData")
    assert len(syn_data) == 10
    assert set(syn_data.columns) == {"nulls", "__primary_key"}
    assert str(syn_data["nulls"].dtype).startswith("string")


class TestTokenizerAndDataCollator:
    @staticmethod
    def get_tokenizer(model_name):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="right",
            truncation_side="right",
            legacy=True,
            add_bos_token=False,
            add_eos_token=False,
        )
        if getattr(tokenizer, "pad_token", None) is None:
            if getattr(tokenizer, "unk_token", None) is not None:
                tokenizer.pad_token = tokenizer.unk_token
            else:
                tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
        return tokenizer

    @pytest.mark.parametrize(
        "model_name",
        [
            "amd/AMD-Llama-135m",  # LllmaTokenizerFast
            "HuggingFaceTB/SmolLM-135M",  # GPT2TokenizerFast
            "Qwen/Qwen2-0.5B",  # Qwen2TokenizerFast
        ],
    )
    def test_tokenize_fn(self, model_name):
        tokenizer = self.get_tokenizer(model_name)
        batch = ["Hello", "Hello, World!"]
        tokenized_data = tokenize_fn(
            text=batch,
            tokenizer=tokenizer,
            padding=True,
            truncation=True,
            add_bos_token=True,
            add_eos_token=True,
        )
        assert None not in tokenized_data["input_ids"][0]
        assert tokenized_data["input_ids"][0][0] == tokenizer.bos_token_id
        assert tokenized_data["input_ids"][0][-1] == tokenizer.pad_token_id
        assert tokenized_data["input_ids"][1][-1] == tokenizer.eos_token_id

    @pytest.fixture(scope="class")
    def response(self):
        return ' {"word2":"world"}'

    @pytest.fixture(scope="class")
    def prompt_with_response(self, response):
        return ' {"word1":"Hello","word2":null}' + response

    @pytest.mark.parametrize(
        "model_name",
        [
            "amd/AMD-Llama-135m",  # LllmaTokenizerFast
            "HuggingFaceTB/SmolLM-135M",  # GPT2TokenizerFast
            "Qwen/Qwen2-0.5B",  # Qwen2TokenizerFast
        ],
    )
    def test_data_collator_for_language_modeling(self, model_name, response, prompt_with_response):
        tokenizer = self.get_tokenizer(model_name)
        tokenized_data = tokenize_fn(
            text=[prompt_with_response],
            tokenizer=tokenizer,
            padding="max_length",
            max_length=30,
            truncation=True,
            add_bos_token=True,
            add_eos_token=True,
        )
        # Note: tokenizers which have the identical pad token and bos/eos tokens (e.g., SmolLM and Qwen2)
        # will fail if we use transformers.DataCollatorForLanguageModeling here
        # That's why we have to implement a custom version
        data_collator = MostlyDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        collated_data = data_collator([tokenized_data])
        label = collated_data["labels"][collated_data["labels"] != -100]
        expected_ids = tokenizer.encode(tokenizer.bos_token + prompt_with_response + tokenizer.eos_token)
        assert label.tolist() == expected_ids


def test_special_character_column_name(tmp_path_factory):
    workspace_dir = tmp_path_factory.mktemp("ws-special-char")
    data = pd.DataFrame(
        {
            "name.lastname.middlename": ["Joe Smith F.", "Anna Brown M."],
            "country-": ["USA", "UK"],
            "hello国": ["world", "world"],
        }
    )
    tgt_encoding_types = {
        "name.lastname.middlename": ModelEncodingType.language_text.value,
        "country-": ModelEncodingType.language_text.value,
        "hello国": ModelEncodingType.language_text.value,
    }
    prepare_encoded_dataset(data, workspace_dir, tgt_encoding_types)
    train(workspace_dir=workspace_dir, model="HuggingFaceTB/SmolLM-135M")
    generate(workspace_dir=workspace_dir, sample_size=50)

    syn_data = pd.read_parquet(workspace_dir / "SyntheticData")
    assert len(syn_data) == 2
    assert set(syn_data.columns) == set([TEMPORARY_PRIMARY_KEY] + list(tgt_encoding_types.keys()))


@pytest.fixture(scope="session")
def encoded_numeric_categorical_datetime_dataset(tmp_path_factory):
    workspace_dir = tmp_path_factory.mktemp("ws")
    no_of_records = 40
    data = pd.DataFrame(
        {
            "gender": ["m", "f", "x", pd.NA] * int(no_of_records / 4),
            "age": [20, 30, 40, 50] * int(no_of_records / 4),
            "date": [
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-02"),
                pd.Timestamp("2023-01-03"),
                pd.Timestamp("2025-01-04"),
            ]
            * int(no_of_records / 4),
        }
    )
    rare_df = pd.DataFrame(
        {
            "gender": [f"rare{i + 1}" for i in range(20)],
            "age": list(range(10, 20)) + list(range(51, 61)),
            "date": (
                [pd.Timestamp("2019-01-01") + pd.Timedelta(days=i) for i in range(10)]
                + [pd.Timestamp("2026-01-01") + pd.Timedelta(days=i) for i in range(10)]
            ),
        }
    )
    data = pd.concat([data, rare_df], ignore_index=True)
    tgt_encoding_types = {
        "age": ModelEncodingType.language_numeric.value,
        "gender": ModelEncodingType.language_categorical.value,
        "date": ModelEncodingType.language_datetime.value,
    }
    split(
        tgt_data=data,
        workspace_dir=workspace_dir,
        model_type="LANGUAGE",
        tgt_encoding_types=tgt_encoding_types,
    )
    analyze(workspace_dir=workspace_dir)
    encode(workspace_dir=workspace_dir)
    return workspace_dir


@pytest.mark.parametrize(
    ("model_name"),
    [
        LSTMFromScratchConfig.model_id,
        "amd/AMD-Llama-135m",
    ],
)
def test_categorical_numeric_datetime(encoded_numeric_categorical_datetime_dataset, model_name):
    workspace_dir = encoded_numeric_categorical_datetime_dataset
    train(workspace_dir=workspace_dir, model=model_name)
    generate(
        workspace_dir=workspace_dir,
        sample_size=40,
        rare_category_replacement_method=RareCategoryReplacementMethod.sample,
    )

    syn_data_path = workspace_dir / "SyntheticData"
    syn = pd.read_parquet(syn_data_path)
    assert len(syn) == 40
    assert set(syn.columns) == {"age", "gender", "date"}

    assert syn["age"].dtype == "Int64"
    # test extreme value protection
    assert syn["age"].min() >= 15
    assert syn["age"].max() <= 55

    assert syn["gender"].dtype == "string"
    # test rare category protection
    assert "rare" not in syn["gender"].values
    assert CATEGORICAL_UNKNOWN_TOKEN not in syn["gender"].values
    assert syn["gender"].nunique(dropna=False) <= 4

    assert syn["date"].dtype == "datetime64[ns]"
    # test extreme value protection
    dates = syn["date"].dropna()
    if not dates.empty:
        assert dates.min() >= pd.Timestamp("2019-01-06")
        assert dates.max() <= pd.Timestamp("2026-01-05")


def test_gpu_estimate_max_batch_size():
    if not torch.cuda.is_available():
        pytest.skip("Skipping gpu only test")

    model = AutoModelForCausalLM.from_pretrained("amd/AMD-Llama-135m")
    model.to("cuda")
    initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    _gpu_estimate_max_batch_size(model, "cuda", 1024, 16)  # on an A10G, will return 8
    # verify parameters unchanged
    for name, param in model.named_parameters():
        assert torch.equal(param, initial_params[name])
        assert param.grad is None


def test_batch_size_and_gradient_accumulation_heuristics():
    gpu_device = torch.device("cuda")
    bs = _physical_batch_size_heuristic(
        no_of_records=1000, no_of_model_params=5_000_000, max_tokens=50, device=gpu_device
    )
    assert bs == 64
    steps = _gradient_accumulation_steps_heuristic(
        batch_size=64,
        no_of_records=1000,
    )
    assert steps == 1

    steps = _gradient_accumulation_steps_heuristic(
        batch_size=32,
        no_of_records=6000,
    )
    assert steps == 2

    # very small dataset
    bs = _physical_batch_size_heuristic(
        no_of_records=10, no_of_model_params=5_000_000, max_tokens=50, device=gpu_device
    )
    assert bs == 1
    steps = _gradient_accumulation_steps_heuristic(
        batch_size=32,
        no_of_records=10,
    )
    assert steps == 1
