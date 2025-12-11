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
import warnings

from mostlyai.engine._language.interface import LanguageModel
from mostlyai.engine._tabular.interface import TabularARGN
from mostlyai.engine._tabular.tensor_utils import (
    build_model_config,
    build_model_config_from_workspace,
    encode_batch,
    prepare_flat_batch,
    prepare_sequential_batch,
    slice_sequences,
)
from mostlyai.engine._tabular.training import ModelConfig
from mostlyai.engine.analysis import analyze
from mostlyai.engine.encoding import encode
from mostlyai.engine.generation import generate
from mostlyai.engine.logging import init_logging
from mostlyai.engine.random_state import set_random_state
from mostlyai.engine.splitting import split
from mostlyai.engine.training import train

__all__ = [
    "split",
    "analyze",
    "encode",
    "train",
    "generate",
    "init_logging",
    "set_random_state",
    "TabularARGN",
    "LanguageModel",
    # Tensor interface for training optimization
    "ModelConfig",
    "build_model_config",
    "build_model_config_from_workspace",
    "encode_batch",
    "prepare_flat_batch",
    "prepare_sequential_batch",
    "slice_sequences",
]
__version__ = "2.3.3"

# suppress specific warning related to os.fork() in multi-threaded processes
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*multi-threaded.*fork.*")
