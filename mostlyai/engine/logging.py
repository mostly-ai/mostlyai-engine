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

import sys
import logging
from pathlib import Path

_LOG = logging.getLogger(__name__.rsplit(".", 1)[0])  # get the logger with the root module name (mostlyai.engine)


def init_logging(log_file: str | Path | None = None) -> None:
    """
    Initialize the logging configuration. Either log to stdout or to a file.

    Args:
        log_file: The path to the log file. If not provided, logs will be printed to stdout.
    """

    if log_file:
        # log to file
        log_file = Path(log_file).absolute()
        if log_file.exists() and log_file.is_dir():
            log_file = log_file / "mostlyai-engine.log"
        else:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, mode="a")
    else:
        # log to stdout
        handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-7s: %(message)s"))
    handler.setLevel(logging.INFO)

    if not _LOG.hasHandlers():
        _LOG.addHandler(handler)
        _LOG.setLevel(logging.INFO)
        _LOG.propagate = False
