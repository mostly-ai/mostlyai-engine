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

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

_LOG = logging.getLogger(__name__)


class LSTMFromScratchConfig(PretrainedConfig):
    model_type = model_id = "MOSTLY_AI/LSTMFromScratch-3m"

    def __init__(
        self,
        vocab_size: int | None = None,
        embedding_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.25,
        with_dp: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.with_dp = with_dp
        super().__init__(**kwargs)


class LSTMFromScratchLMHeadModel(PreTrainedModel, GenerationMixin):
    config_class = LSTMFromScratchConfig

    def __init__(self, config: LSTMFromScratchConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_size)
        self.dropout = nn.Dropout(self.config.dropout)
        if self.config.with_dp:
            from opacus.layers import DPLSTM

            lstm_cls = DPLSTM
        else:
            lstm_cls = nn.LSTM
        self.lstm = lstm_cls(
            input_size=self.config.embedding_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

        # this will be filled by left_to_right_padding() during the generation
        self.pad_token_id = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutput:
        lengths = attention_mask.sum(dim=1)
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        # (DP)LSTM layers without pack_padded_sequence/pad_packed_sequence
        lstm_outputs, _ = self.lstm(embeddings)

        logits = self.lm_head(lstm_outputs)

        loss = None
        if labels is not None:
            labels = labels[:, 1:].contiguous()
            shifted_prediction_scores = logits[:, :-1, :].contiguous()
            loss = self.loss_fn(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            # overwrite the logit of the last time step with the logit of the actual last token
            # so that Hugging Face Transformers' generate() will sample on the right probabilities
            logits[:, -1, :] = torch.stack([logits[i, length - 1, :] for i, length in enumerate(lengths)])
        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        This function is mandatory so that the model is able to use the Hugging Face `.generate()` method.
        Since `.generate()` works with left-padded sequences but the model is trained with right-padded sequences,
        we need to convert the padding side here to make it work properly.
        """
        lengths = attention_mask.sum(dim=1)
        return {
            "input_ids": self.left_to_right_padding(input_ids, lengths),
            "attention_mask": attention_mask,
        }

    def left_to_right_padding(self, left_padded_tensors: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_length = left_padded_tensors.size()
        indices = torch.nonzero(lengths < max_length)
        if len(indices) == 0:
            # none of the samples are padded, so we can just return them as they are
            return left_padded_tensors
        else:
            if self.pad_token_id is None:
                # get the pad token id from the first padded sample
                self.pad_token_id = left_padded_tensors[indices[0], -1].item()
            right_padded_tensors = torch.full_like(left_padded_tensors, self.pad_token_id)
            for i in range(batch_size):
                right_padded_tensors[i, : lengths[i]] = left_padded_tensors[i, max_length - lengths[i] :]
            return right_padded_tensors


def register_mostly_lstm_model():
    # register the model so that we can load it with `AutoModelForCausalLM.from_pretrained()` later
    AutoConfig.register(LSTMFromScratchConfig.model_id, LSTMFromScratchConfig)
    AutoModel.register(LSTMFromScratchConfig, LSTMFromScratchLMHeadModel)
    AutoModelForCausalLM.register(LSTMFromScratchConfig, LSTMFromScratchLMHeadModel)
