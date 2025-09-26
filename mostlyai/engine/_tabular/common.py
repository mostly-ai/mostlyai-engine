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
from pathlib import Path

import torch

_LOG = logging.getLogger(__name__)

DPLSTM_SUFFIXES: tuple = ("ih.weight", "ih.bias", "hh.weight", "hh.bias")


def load_model_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    t0 = time.time()
    incompatible_keys = model.load_state_dict(torch.load(f=path, map_location=device, weights_only=True), strict=False)
    missing_keys = incompatible_keys.missing_keys
    unexpected_keys = incompatible_keys.unexpected_keys
    # for DP-trained models, we expect extra keys from the DPLSTM layers (which is fine to ignore because we use standard LSTM layers during generation)
    # but if there're any other missing or unexpected keys, an error should be raised
    if len(missing_keys) > 0 or any(not k.endswith(DPLSTM_SUFFIXES) for k in unexpected_keys):
        raise RuntimeError(
            f"failed to load model weights due to incompatibility: {missing_keys = }, {unexpected_keys = }"
        )
    _LOG.info(f"loaded model weights in {time.time() - t0:.2f}s")
