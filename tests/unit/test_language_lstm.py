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

import pytest
import torch

from mostlyai.engine._language.lstm import LSTMFromScratchConfig, LSTMFromScratchLMHeadModel


@pytest.mark.parametrize("with_dp", [False, True])
@pytest.mark.parametrize("mask_kind", ["omitted", "all_ones", "left_padded"])
def test_generate_with_optional_attention_mask(with_dp, mask_kind):
    torch.manual_seed(42)
    model = LSTMFromScratchLMHeadModel(
        LSTMFromScratchConfig(
            vocab_size=8,
            embedding_size=8,
            hidden_size=8,
            dropout=0.0,
            with_dp=with_dp,
            pad_token_id=0,
            bos_token_id=1,
        )
    ).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [1, 3, 4, 5]])
    if mask_kind == "left_padded":
        input_ids[0, :2] = 0
    attention_mask = (input_ids != 0).long()
    kwargs = {} if mask_kind == "omitted" else {"attention_mask": attention_mask}
    generated = model.generate(input_ids, max_new_tokens=3, do_sample=False, **kwargs)
    assert generated.shape == (2, 7)
    for row, mask, output in zip(input_ids, attention_mask, generated):
        unpadded = row[mask.bool()].unsqueeze(0)
        expected = model.generate(unpadded, max_new_tokens=3, do_sample=False)
        torch.testing.assert_close(output[-3:], expected[0, -3:])
