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

_default_handler = logging.StreamHandler(stream=sys.stdout)
_default_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)-7s: %(message)s"))
_default_handler.setLevel(logging.INFO)

_LOG = logging.getLogger(__name__.rsplit(".", 1)[0])


def setup_logging() -> None:
    _LOG.addHandler(_default_handler)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def unset_logging() -> None:
    _LOG.removeHandler(_default_handler)
    _LOG.setLevel(logging.NOTSET)
    _LOG.propagate = True
