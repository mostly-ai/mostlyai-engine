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

from mostlyai.engine._common import STRING, safe_convert_string


def analyze_text(values: pd.Series, root_keys: pd.Series, _: pd.Series | None = None) -> dict:
    # ideally, we should ensure that values are converted to string in a consistent way across analyze/encode/qa steps
    values = safe_convert_string(values)
    nchars = values.map(str).str.len()
    stats = {"nchar_max": int(nchars.max()), "nchar_sum": int(nchars.sum()), "count": len(values)}
    return stats


def analyze_reduce_text(
    stats_list: list[dict],
    value_protection: bool = True,
    value_protection_epsilon: float | None = None,
) -> dict:
    nchar_max = 0
    nchar_sum = 0
    count = 0
    for stats in stats_list:
        nchar_max = max(stats["nchar_max"], nchar_max)
        nchar_sum += stats["nchar_sum"]
        count += stats["count"]

    stats = {
        "nchar_avg": round(nchar_sum / count, 1),
        "nchar_max": nchar_max,
    }
    return stats


def decode_text(x: pd.Series, col_stats: dict[str, str]) -> pd.Series:
    return x.astype(STRING)
