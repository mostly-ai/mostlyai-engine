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

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Iterator

from transformers import DataCollatorForLanguageModeling, BatchEncoding, PreTrainedTokenizerFast, LlamaTokenizerFast
from transformers.data.data_collator import pad_without_fast_tokenizer_warning, _torch_collate_batch

from mostlyai.engine.domain import ModelEncodingType


#################
### TOKENIZER ###
#################


def train_tokenizer(
    training_iterator: Iterator | list | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
    tgt_stats: dict[str, Any] | None = None,
):
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE
    from tokenizers.normalizers import Replace
    from tokenizers.pre_tokenizers import Metaspace, Punctuation, Sequence, Split
    from tokenizers.trainers import BpeTrainer

    special_tokens = {
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    }
    SPECIAL_TOKENS = list(special_tokens.values())
    NEW_LINE_VALUE = "\n"
    NEW_LINE_SYMBOL = "\u240a"  # https://www.fileformat.info/info/unicode/char/240a/index.htm
    MIN_FREQ_MERGE = 20
    VOCAB_SIZE = 5000

    # add initial alphabet for numeric and datetime columns if needed
    has_numeric_columns = any(
        col_stats["encoding_type"] == ModelEncodingType.language_numeric for col_stats in tgt_stats["columns"].values()
    )
    has_datetime_columns = any(
        col_stats["encoding_type"] == ModelEncodingType.language_datetime for col_stats in tgt_stats["columns"].values()
    )
    initial_alphabet = set()
    if has_numeric_columns:
        # FIXME: maybe the set can be more fine-grained based on max_scale in stats
        initial_alphabet |= {str(i) for i in range(10)} | {".", "-", "+", "e", "E"}
    if has_datetime_columns:
        initial_alphabet |= {str(i) for i in range(10)} | {".", "-", ":", "T", "Z"}
    initial_alphabet = list(initial_alphabet)

    # Builds a BPE raw_tokenizer, and optionally trains it based on provided text
    training_iterator = training_iterator or []  # allow easy training skip
    raw_tokenizer = Tokenizer(BPE(unk_token=special_tokens["unk_token"]))
    trainer = BpeTrainer(
        initial_alphabet=initial_alphabet,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=MIN_FREQ_MERGE,
        vocab_size=VOCAB_SIZE,
        show_progress=False,
    )
    raw_tokenizer.normalizer = Replace(NEW_LINE_VALUE, NEW_LINE_SYMBOL)
    raw_tokenizer.pre_tokenizer = Sequence(
        [
            Metaspace(),
            Split(pattern=NEW_LINE_SYMBOL, behavior="isolated"),
            Punctuation(),
        ]
    )
    raw_tokenizer.decoder = decoders.Sequence(
        [
            decoders.Metaspace(),
            decoders.Replace(NEW_LINE_SYMBOL, NEW_LINE_VALUE),
        ]
    )
    raw_tokenizer.train_from_iterator(iterator=training_iterator, trainer=trainer)
    tokenizer = LlamaTokenizerFast(tokenizer_object=raw_tokenizer, **special_tokens, **tokenizer_kwargs)
    return tokenizer


def tokenize_fn(
    text: dict[str, str] | dict[str, list[str]] | list[str],
    tokenizer: PreTrainedTokenizerFast,
    text_key: str | None = None,
    return_tensors: str | None = None,
    padding: bool | str = True,
    truncation: bool = True,
    add_bos_token: bool = True,
    add_eos_token: bool = True,
    max_length: int = 1024,
) -> BatchEncoding:
    if text_key:
        text = text[text_key]
    # make sure the tokenizer is configured as expected
    if getattr(tokenizer, "add_bos_token", False) or getattr(tokenizer, "add_eos_token", False):
        raise RuntimeError("Tokenizer must be configured as add_bos_token=False and add_eos_token=False")
    if tokenizer.bos_token is None or tokenizer.eos_token is None:
        raise RuntimeError("Tokenizer must have bos_token and eos_token set")
    prefix = tokenizer.bos_token if add_bos_token else ""
    suffix = tokenizer.eos_token if add_eos_token else ""
    # NOTE: here we add bos/eos tokens before truncation and padding,
    # which means that they may be truncated for long sequences
    if isinstance(text, str):
        text = f"{prefix}{text}{suffix}"
    else:
        for i, t in enumerate(text):
            text[i] = f"{prefix}{t}{suffix}"
    tokenized_content = tokenizer(
        text,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
    )
    return tokenized_content


#####################
### DATA COLLATOR ###
#####################


@dataclass
class MostlyDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        """
        A variation of the original `DataCollatorForLanguageModeling.torch_call` method.

        This method can mask tokens based on the attention mask, so that bos and eos tokens will not be masked
        even if they are identical to pad token.
        If attention mask is not provided, it will fall back to masking pad tokens.
        """
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=None
            )
        else:
            batch = {"input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=None)}

        labels = batch["input_ids"].clone()
        attention_mask = batch.get("attention_mask", None)
        if attention_mask is not None:
            labels[(attention_mask == 0)] = -100
        else:
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
