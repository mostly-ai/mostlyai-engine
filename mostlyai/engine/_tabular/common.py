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
import time
import torch
from pathlib import Path

_LOG = logging.getLogger(__name__)


def load_model_weights(model: torch.nn.Module, path: Path, device: torch.device):
    try:
        t00 = time.time()
        model.load_state_dict(torch.load(f=path, map_location=device, weights_only=True))
        _LOG.info(f"loaded model weights in {time.time() - t00:.2f}s")
    except Exception as e:
        _LOG.warning(f"failed to load model weights: {e}")
