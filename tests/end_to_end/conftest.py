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


import inspect
import re
import textwrap

import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor

# Workaround for vLLM / Triton kernel warmup issue (https://github.com/vllm-project/vllm/issues/49920)
# In Triton 3.7.x, `re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE)` in `triton/runtime/jit.py:JITCallable.__init__`
# assumes `def` is at the start of a line. When vLLM >= 0.28.0 warms up multi-decorated kernels
# (e.g. `minimax_m3/common/ops/index_topk.py` wrapped with `@triton.heuristics`), the regex fails,
# raising `AttributeError: 'NoneType' object has no attribute 'start'`.
# TODO: Remove once upstream vLLM/Triton safely handles multi-decorated kernel source inspection.
try:
    import triton.runtime.jit

    _orig_jit_callable_init = triton.runtime.jit.JITCallable.__init__

    def _patched_jit_callable_init(self, fn):
        try:
            _orig_jit_callable_init(self, fn)
        except AttributeError as e:
            if "'NoneType' object has no attribute 'start'" in str(e):
                self.fn = fn
                self.signature = inspect.signature(fn)
                self.raw_src, self.starting_line_number = inspect.getsourcelines(fn)
                self._fn_name = triton.runtime.jit.get_full_name(fn)
                self._hash_lock = triton.runtime.jit.threading.RLock()
                src = textwrap.dedent("".join(self.raw_src))
                m = re.search(r"(?:^|\n)\s*def\s+[\w_]+\s*\(", src)
                if m:
                    def_pos = src.find("def", m.start())
                    src = src[def_pos:]
                else:
                    lines = src.splitlines()
                    def_lines = [i for i, line in enumerate(lines) if "def " in line]
                    if def_lines:
                        src = "\n".join(lines[def_lines[0] :])
                self._src = src
                self.hash = None
            else:
                raise

    triton.runtime.jit.JITCallable.__init__ = _patched_jit_callable_init
except Exception:  # noqa: BLE001, S110
    pass

from mostlyai.engine._common import STRING


@pytest.fixture()
def cleanup_joblib_pool():
    # make sure the test is using a fresh joblib pool
    get_reusable_executor().shutdown(wait=True)
    yield
    get_reusable_executor().shutdown(wait=True)


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
