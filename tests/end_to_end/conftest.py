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

import random

import numpy as np
import pandas as pd
import pytest

from mostlyai.engine._common import STRING


@pytest.fixture
def set_random_seed():
    random.seed(0)
    np.random.seed(0)


class MockData:
    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.df = pd.DataFrame(index=range(self.n_samples))

    def add_index_column(self, name: str):
        values = pd.DataFrame({name: range(len(self.df))}).astype(STRING)
        self.df = pd.concat([self.df, values], axis=1)

    def add_categorical_column(
        self, name: str, probabilities: dict[str, float], rare_categories: list[str] | None = None
    ):
        values = np.random.choice(
            list(probabilities.keys()),
            size=len(self.df),
            p=list(probabilities.values()),
        )
        self.df = pd.concat([self.df, pd.DataFrame({name: values})], axis=1)
        if rare_categories:
            self.df.loc[np.random.choice(self.df.index, len(rare_categories), replace=False), name] = rare_categories

    def add_numeric_column(self, name: str, quantiles: dict[float, float], dtype: str = "float32"):
        uniform_samples = np.random.rand(len(self.df))
        values = np.interp(uniform_samples, list(quantiles.keys()), list(quantiles.values())).astype(dtype)
        self.df = pd.concat([self.df, pd.DataFrame({name: values})], axis=1)

    def add_datetime_column(self, name: str, start_date: str, end_date: str, freq: str = "s"):
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        values = np.random.choice(date_range, len(self.df), replace=True)
        self.df = pd.concat([self.df, pd.DataFrame({name: values})], axis=1)

    def add_date_column(self, name: str, start_date: str, end_date: str):
        self.add_datetime_column(name, start_date, end_date, freq="D")

    def add_lat_long_column(self, name: str, lat_limit: tuple[float, float], long_limit: tuple[float, float]):
        latitude = np.random.uniform(lat_limit[0], lat_limit[1], len(self.df))
        longitude = np.random.uniform(long_limit[0], long_limit[1], len(self.df))
        values = [f"{lat:.4f}, {long:.4f}" for lat, long in zip(latitude, longitude)]
        self.df = pd.concat([self.df, pd.DataFrame({name: values})], axis=1)

    def add_sequential_column(self, name: str, seq_len_quantiles: dict[float, float]):
        self.add_numeric_column("seq_len", seq_len_quantiles, dtype="int32")
        # if seq_len is 3, it will populate a sequence ["0", "1", "2"] and then explode the list to 3 rows
        self.df[name] = self.df["seq_len"].apply(lambda x: [str(i) for i in range(x)])
        self.df = self.df.explode(name).drop(columns="seq_len").reset_index(drop=True)
