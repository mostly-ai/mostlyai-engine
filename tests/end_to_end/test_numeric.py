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
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mostlyai.engine.domain import ModelEncodingType
from mostlyai.engine import analyze, encode
from mostlyai.engine._common import write_json
from mostlyai.engine._tabular.generation import generate
from mostlyai.engine._tabular.training import train


@pytest.fixture
def sum_df():
    df = pd.DataFrame(
        {
            "n": np.random.randint(0, 98, size=1000) * 100,
            "m": np.random.randint(0, 98, size=1000),
        }
    )

    # Create column 'o' as sum of 'n' and 'm'
    df["o"] = df["n"] + df["m"]

    yield df


@pytest.fixture
def product_df():
    # Generate 1000 uniformly distributed random prices between 0 and 300
    prices = np.random.uniform(0, 300, 1000).astype(int)

    # Replace the decimal part with one of the following: 00, 05, 50, 95, 99.
    decimals = np.random.choice([0, 0.05, 0.5, 0.95, 0.99], 1000)
    prices = prices.astype(int) + decimals

    # Create a DataFrame
    df = pd.DataFrame(
        {
            "price": prices,
        }
    )

    yield df


def prepare_ws(tmp_path: Path, df: pd.DataFrame, keys: dict, encoding_types: dict) -> Path:
    workspace_dir = tmp_path / "ws"
    shutil.rmtree(workspace_dir, ignore_errors=True)  # cleanup
    tgt_meta_path = workspace_dir / "OriginalData" / "tgt-meta"
    tgt_data_path = workspace_dir / "OriginalData" / "tgt-data"
    for path in [
        workspace_dir,
        tgt_meta_path,
        tgt_data_path,
    ]:
        path.mkdir(exist_ok=True, parents=True)

    df.to_parquet(tgt_data_path / "part.000000-trn.parquet")
    write_json(keys, tgt_meta_path / "keys.json")
    write_json(encoding_types, tgt_meta_path / "encoding-types.json")

    return workspace_dir


def synthetize(ws_dir: Path) -> pd.DataFrame:
    analyze(workspace_dir=ws_dir)
    encode(workspace_dir=ws_dir)
    train(max_epochs=5, workspace_dir=ws_dir)
    generate(workspace_dir=ws_dir)
    syn_data_path = ws_dir / "SyntheticData"
    syn = pd.read_parquet(syn_data_path)

    return syn


def compare_numeric_encodings(
    tmp_path,
    df,
    numeric_cols,
    first=ModelEncodingType.tabular_numeric_auto,
    second=ModelEncodingType.tabular_numeric_digit,
):
    syn = []
    for numeric_encoding in [first, second]:
        ws = prepare_ws(
            tmp_path=tmp_path,
            df=df,
            keys={},
            encoding_types={k: numeric_encoding.value for k in numeric_cols},
        )
        syn.append(synthetize(ws))

    return syn[0], syn[1]


def test_numeric_sum_quality(tmp_path, sum_df):
    sum_syn_auto, sum_syn_digit = compare_numeric_encodings(tmp_path=tmp_path, df=sum_df, numeric_cols=["n", "m", "o"])

    assert sum_syn_auto.shape == sum_syn_digit.shape

    def calculate_sum_square_errors(df: pd.DataFrame, expected: str, actual: str):
        # Calculate the squares of the % errors
        squared_error = np.square((df[actual] - df[expected]) / df[actual])
        return np.sum(squared_error)

    sum_syn_auto["expected"] = sum_syn_auto["n"] + sum_syn_auto["m"]
    sum_syn_auto_errors = calculate_sum_square_errors(df=sum_syn_auto, expected="expected", actual="o")
    sum_syn_digit["expected"] = sum_syn_digit["n"] + sum_syn_digit["m"]
    sum_syn_digit_errors = calculate_sum_square_errors(df=sum_syn_digit, expected="expected", actual="o")

    # ensure the quality is reasonable
    assert sum_syn_auto_errors / sum_syn_digit_errors < 10


def test_numeric_price_quality(tmp_path, product_df):
    prod_syn_auto, prod_syn_digit = compare_numeric_encodings(tmp_path=tmp_path, df=product_df, numeric_cols=["price"])

    assert prod_syn_auto.shape == prod_syn_digit.shape

    def similar_quantiles(ser_first, ser_second, threshold=0.05) -> bool:
        quantiles = [0.25, 0.5, 0.75, 1]
        q_first = ser_first.quantile(quantiles)
        q_second = ser_second.quantile(quantiles)
        return bool(all(np.abs((q_first - q_second) / ((q_first + q_second) / 2)) <= threshold))

    assert similar_quantiles(product_df, prod_syn_auto)
    assert similar_quantiles(product_df, prod_syn_digit)
