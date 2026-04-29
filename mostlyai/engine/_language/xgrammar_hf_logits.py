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

"""xgrammar Hugging Face LogitsProcessor with fixes for our dependency stack."""

from __future__ import annotations

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as _UpstreamLogitsProcessor


class LogitsProcessor(_UpstreamLogitsProcessor):
    """
    Same as ``xgrammar.contrib.hf.LogitsProcessor``, but passes a Python ``int`` to
    ``GrammarMatcher.accept_token``. Upstream uses ``input_ids[i][-1]`` (a 0-dim tensor);
    newer PyTorch / Transformers stacks leave that as ``torch.Tensor``, while xgrammar's
    TVM FFI expects ``int``.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if len(self.matchers) == 0:
            self.batch_size = input_ids.shape[0]
            self.compiled_grammars = (
                self.compiled_grammars if len(self.compiled_grammars) > 1 else self.compiled_grammars * self.batch_size
            )
            assert len(self.compiled_grammars) == self.batch_size, (
                "The number of compiled grammars must be equal to the batch size."
            )
            self.matchers = [xgr.GrammarMatcher(self.compiled_grammars[i]) for i in range(self.batch_size)]
            self.token_bitmask = xgr.allocate_token_bitmask(self.batch_size, self.full_vocab_size)

        if input_ids.shape[0] != self.batch_size:
            raise RuntimeError(
                "Expect input_ids.shape[0] to be LogitsProcessor.batch_size."
                + f"Got {input_ids.shape[0]} for the former, and {self.batch_size} for the latter."
            )

        if not self.prefilled:
            self.prefilled = True
        else:
            for i in range(self.batch_size):
                if not self.matchers[i].is_terminated():
                    sampled_token = int(input_ids[i, -1].item())
                    assert self.matchers[i].accept_token(sampled_token)

        for i in range(self.batch_size):
            if not self.matchers[i].is_terminated():
                self.matchers[i].fill_next_token_bitmask(self.token_bitmask, i)

        device_type = scores.device.type
        if device_type != "cuda":
            scores = scores.to("cpu")
        xgr.apply_token_bitmask_inplace(scores, self.token_bitmask.to(scores.device))
        if device_type != "cuda":
            scores = scores.to(device_type)

        return scores
