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

import logging
import os
import re

import psutil
import torch

_LOG = logging.getLogger(__name__)


def get_available_vram_for_heuristics() -> int:
    if not torch.cuda.is_available():
        return 0
    free, total = torch.cuda.mem_get_info()
    return total


def get_available_ram_for_heuristics() -> int:
    mem_limit = extract_memory_from_string(os.getenv("MOSTLY_ENGINE_AVAILABLE_RAM_FOR_HEURISTICS", default=None))
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available
    return mem_limit


def extract_memory_from_string(memory_str: str | None = None) -> int | None:
    """
    Extract the memory in bytes from a string.

    :param memory_str: The memory string to extract the memory from.
    :return: The memory in bytes.
    """
    if not memory_str:
        return None

    # Conversion factors, considering metric (decimal) vs. binary (IEC) units
    units = {
        "": 1,
        "b": 1,
        "k": 1024,
        "m": 1024**2,
        "g": 1024**3,
        "t": 1024**4,
    }
    match = re.match(r"(\d+(?:\.\d+)?)[ ]?([a-z]?)", memory_str.strip().lower())
    if not match:
        return None

    value, unit = match.groups()
    value = float(value)

    # Convert to bytes
    if unit in units:
        return int(value * units[unit])
    else:
        return None
