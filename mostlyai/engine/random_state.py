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
import random
import struct

import numpy as np
import torch

_LOG = logging.getLogger(__name__)


def set_random_state(random_state: int | None = None, worker: bool = False):
    def get_random_int_from_os() -> int:
        # 32-bit, cryptographically secure random int from os
        return int(struct.unpack("I", os.urandom(4))[0])

    if worker:  # worker process
        if "MOSTLYAI_ENGINE_SEED" in os.environ:
            random_state = int(os.environ["MOSTLYAI_ENGINE_SEED"])
        else:
            # don't set seed for worker process if not set in main process
            return
    else:  # main process
        if random_state is not None:
            _LOG.info(f"Global random_state set to `{random_state}`")

        if random_state is None:
            random_state = get_random_int_from_os()

        os.environ["MOSTLYAI_ENGINE_SEED"] = str(random_state)

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
