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

from mostlyai.engine.analysis import analyze
from mostlyai.engine.encoding import encode
from mostlyai.engine.generation import generate
from mostlyai.engine.logging import init_logging
from mostlyai.engine.random_state import set_random_state
from mostlyai.engine.splitting import split
from mostlyai.engine.training import train

__all__ = ["split", "analyze", "encode", "train", "generate", "init_logging", "set_random_state"]
__version__ = "1.5.4"

# suppress specific warning related to os.fork() in multi-threaded processes
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*multi-threaded.*fork.*")
