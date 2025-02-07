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

import calendar
import contextlib
import datetime
import json
import os

import json_repair
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import constants as hf_constants
from transformers import (
    PreTrainedTokenizerBase,
)


from mostlyai.engine._common import (
    persist_data_part,
    FixedSizeSampleBuffer,
    STRING,
    ProgressCallback,
    ProgressCallbackWrapper,
)
from mostlyai.engine._language.common import estimate_max_tokens, MAX_LENGTH
from mostlyai.engine._language.encoding import encode_df
from mostlyai.engine._workspace import ensure_workspace_dir, Workspace, reset_dir
from mostlyai.engine._language.formatron_utils import (
    get_formatter_builders,
    prepare_seed_for_formatron,
    get_vocab_processors,
)
from mostlyai.engine.domain import ModelEncodingType

INVALID_VALUE = "_INVALID_"  # when JSON parsing fails, the values of target columns will be set to this
DUMMY_CONTEXT_KEY = "__dummy_context_key"
_LOG = logging.getLogger(__name__)


def decode_buffered_samples(
    buffer: FixedSizeSampleBuffer,
    tokenizer: PreTrainedTokenizerBase,
    tgt_stats: dict[str, str],
    tgt_context_key: str,
    max_new_tokens: int,
):
    t0 = time.time()

    def parse_json(x, columns: list[str]):
        try:
            parsed_x = json_repair.loads(x)
            if not isinstance(parsed_x, dict):
                raise ValueError("parsed_x has to be a dictionary")
        except (json.decoder.JSONDecodeError, ValueError):
            parsed_x = {}
        return [parsed_x.get(c, INVALID_VALUE) for c in columns]

    ctx_keys = []
    tgt_seed = []
    output_texts = []
    num_samples_max_length_limit = 0
    for outputs_ids, keys_df, seed_df in buffer.buffer:
        try:
            num_tokens_by_row = [sum(token != tokenizer.eos_token_id for token in row) for row in outputs_ids]
            num_samples_max_length_limit += sum(1 for tokens in num_tokens_by_row if tokens >= max_new_tokens)
        except AttributeError:
            num_samples_max_length_limit = float("-inf")
        outputs_text = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        output_texts.extend(outputs_text)
        ctx_keys.append(keys_df)
        tgt_seed.append(seed_df)
    _LOG.info(f"{num_samples_max_length_limit=}")
    ctx_keys = pd.concat(ctx_keys, axis=0).reset_index(drop=True).rename(tgt_context_key)
    tgt_seed = pd.concat(tgt_seed, axis=0).reset_index(drop=True)
    # The model works with un-prefixed column names, but we need to recover prefixed column names for the final output
    tgt_data = pd.DataFrame(
        [parse_json(text, tgt_stats["columns"].keys()) for text in output_texts],
        columns=tgt_stats["columns"].keys(),
        index=ctx_keys.index,
        dtype="string",
    )
    # make sure invalid/incomplete unicode chars are replaced with the replacement char � (U+FFFD)
    tgt_data = tgt_data.map(
        lambda x: x.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace") if not pd.isna(x) else x
    )
    # overwrite generated columns with the seeded values
    tgt_data.update(tgt_seed)

    # prepend the context keys to the data (if not dummy context)
    if ctx_keys.name != DUMMY_CONTEXT_KEY:
        tgt_data = pd.concat([ctx_keys, tgt_data], axis=1)
    invalid_percentage = ((tgt_data[tgt_stats["columns"].keys()] == INVALID_VALUE).sum() / len(tgt_data) * 100.0).map(
        "{:.2f}%".format
    )

    for col in tgt_stats["columns"].keys():
        col_stats = tgt_stats["columns"][col]
        if col_stats["encoding_type"] == ModelEncodingType.language_numeric:
            tgt_data[col] = _decode_numeric(tgt_data[col], col_stats)
        elif col_stats["encoding_type"] == ModelEncodingType.language_datetime:
            tgt_data[col] = _decode_datetime(tgt_data[col], col_stats)
        else:
            tgt_data[col] = _decode_string(tgt_data[col], col_stats)

    _LOG.info(f"percentage of invalid values: {invalid_percentage.to_dict()}")
    _LOG.info(f"decoded {tgt_data.shape} from {len(buffer.buffer)} batches in {time.time() - t0:.2f}s")
    return tgt_data


def _decode_string(x: pd.Series, col_stats: dict[str, str]) -> pd.Series:
    return x.astype(STRING)


def _decode_numeric(x: pd.Series, col_stats: dict[str, str]) -> pd.Series:
    # FIXME revisit for invalid values  -- sample from values / nan / or other
    x[(x == "") | (x == "_INVALID_")] = np.nan
    if col_stats["max_scale"] == 0:
        return x.astype("Int64")
    return x.astype(float)


def coerce_datetime(text: str) -> str:
    """
    Ensure that the text is a valid date or datetime string.
    """
    if text == "" or text == "_INVALID_":
        return text
    # FIXME copy paste from datallm, see if should be cleaned up

    # extract year, month, and day from the ISO formatted text
    y, m, d = int(text[:4]), int(text[5:7]), int(text[8:10])
    # set to last day of month, in case of too large day value
    last_day = calendar.monthrange(y, m)[1]
    d = min(d, last_day)
    dt_str = f"{y:04d}-{m:02d}-{d:02d}" + text[10:]
    # convert to date and back to check for valid date
    try:
        dt_str = datetime.datetime.fromisoformat(dt_str).isoformat().replace("T", " ")
    except ValueError:
        dt_str = text  # FIXME revisit, if e.g. a cutoff date, then we just return the original text
    # trim to original length
    dt_str = dt_str[: len(text)]
    return dt_str


def _decode_datetime(x: pd.Series, col_stats: dict[str, str]) -> pd.Series:
    print(x)
    # FIXME revisit for invalid values -- sample from values / nan / or other
    x = x.map(coerce_datetime)
    return pd.to_datetime(x, errors="coerce")


def generate(
    *,
    ctx_data: pd.DataFrame | None = None,
    seed_data: pd.DataFrame | None = None,
    sample_size: int | None = None,
    batch_size: int | None = None,
    sampling_temperature: float = 1.0,
    sampling_top_p: float = 1.0,
    device: torch.device | str | None = None,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
):
    _LOG.info("GENERATE_LANGUAGE started")
    t0_ = time.time()
    os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
    os.environ["VLLM_NO_DEPRECATION_WARNING"] = "1"

    @contextlib.contextmanager
    def tqdm_disabled():
        tqdm_disable = os.getenv("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        try:
            yield
        finally:
            os.environ["TQDM_DISABLE"] = tqdm_disable if tqdm_disable is not None else ""

    with ProgressCallbackWrapper(update_progress) as progress, tqdm_disabled():
        device = (
            torch.device(device)
            if device is not None
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )
        _LOG.info(f"{device=}")
        _LOG.info(f"{sampling_temperature=}, {sampling_top_p=}")

        workspace_dir = ensure_workspace_dir(workspace_dir)
        workspace = Workspace(workspace_dir)
        output_path = workspace.generated_data_path
        reset_dir(output_path)
        tgt_stats = workspace.tgt_stats.read()
        tgt_text_columns = list(tgt_stats["columns"].keys())
        tgt_context_key = tgt_stats["keys"].get("context_key")
        has_context = workspace.ctx_stats.path.exists()

        # resolve potential conflict between sample_seed and sample_size
        if seed_data is not None:
            assert sample_size is None, "either sample_seed or sample_size can be provided, not both"
            sample_size = len(seed_data)

        if has_context:
            ctx_stats = workspace.ctx_stats.read()
            ctx_columns = list(ctx_stats["columns"].keys())
            ctx_primary_key = ctx_stats["keys"].get("primary_key")

            # ensure ctx_data exists
            if ctx_data is None:
                if workspace.ctx_data_path.exists():
                    # attempt to re-use context from training, if no new context provided
                    ctx_data = pd.read_parquet(workspace.ctx_data_path)
                else:
                    # build dummy context; fallback to using training data size as sample_size
                    trn_sample_size = tgt_stats["no_of_training_records"] + tgt_stats["no_of_validation_records"]
                    ctx_data = pd.DataFrame({ctx_primary_key: list(range(sample_size or trn_sample_size))})
            _LOG.info(f"{ctx_data.shape=}")
            ctx_data = ctx_data.reset_index(drop=True)
            ctx_data_len = len(ctx_data)
            if sample_size is not None and sample_size < ctx_data_len:
                # take first `sample_size` rows of context
                ctx_data = ctx_data.head(sample_size)
                _LOG.info(f"dropped {ctx_data_len - len(ctx_data)} rows from context data")
            _LOG.info(f"{ctx_data.shape=}")

            # update sample_size based on fetched context data
            sample_size = len(ctx_data)
            _LOG.info(f"{sample_size=}")
        else:
            # create on-the-fly context
            if sample_size is None:
                trn_sample_size = tgt_stats["no_of_training_records"] + tgt_stats["no_of_validation_records"]
                sample_size = trn_sample_size if sample_size is None else sample_size
            ctx_columns = []
            ctx_primary_key = tgt_context_key = DUMMY_CONTEXT_KEY
            ctx_data = pd.DataFrame({ctx_primary_key: range(sample_size)})

        # ensure sample_seed exists; ensure valid columns
        if seed_data is None:
            # build dummy seed
            seed_data = pd.DataFrame(index=list(range(sample_size)))
        seed_data = seed_data[[c for c in tgt_text_columns if c in seed_data.columns]]
        _LOG.info(f"{seed_data.shape=}")

        # sanity check: at this point sample seed and context data should have the same number of rows
        assert len(seed_data) == len(ctx_data)

        # early exit in case generation context is empty
        if sample_size == 0:
            _LOG.info("terminating generation early as no context data provided")
            empty_out_df = pd.DataFrame(columns=[tgt_context_key] + tgt_text_columns, dtype="string")
            persist_data_part(empty_out_df, output_path, f"{0:06}.{0:06}")
            return

        # encode context data
        encoded_ctx_data = encode_df(ctx_df=ctx_data, ctx_columns=ctx_columns)

        # estimate max new tokens based on char length of original data; consider JSON overhead
        max_new_tokens = estimate_max_tokens(tgt_stats)
        _LOG.info(f"{max_new_tokens=}")

        t0 = time.time()
        hf_constants.HF_HUB_OFFLINE = (
            False  # needed for gated hf models that are not sharded, otherwise GatedRepoError in vLLM
        )
        # set the default env var so that we don't pass it explicitly to vLLM
        os.environ["HF_TOKEN"] = os.getenv("MOSTLY_HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN", "")

        is_peft_adapter = (workspace.model_path / "adapter_config.json").exists()
        if is_peft_adapter and device.type == "cuda":
            from mostlyai.engine._language.engine.vllm_engine import VLLMEngine

            engine = VLLMEngine(workspace.model_path, device, max_new_tokens, MAX_LENGTH)
        else:
            from mostlyai.engine._language.engine.hf_engine import HuggingFaceEngine

            engine = HuggingFaceEngine(workspace.model_path, device, max_new_tokens, MAX_LENGTH)
        _LOG.info(f"inference engine: {engine.__class__.__name__}")

        batch_size = batch_size or engine.get_default_batch_size()
        _LOG.info(f"model loading time: {time.time() - t0:.2f}s")

        if batch_size > sample_size:
            batch_size = sample_size
        _LOG.info(f"{batch_size=}")

        # prepare seed data for clean consumption by formatron
        seed_data = prepare_seed_for_formatron(seed_data, engine.tokenizer)
        seeded_tgt_columns = seed_data.columns.to_list()

        total_tokenize_fn_time = 0
        total_logits_processor_build_time = 0
        total_generate_fn_time = 0

        enforce_json_output = engine.supports_json_enforcing()
        _LOG.info(f"{enforce_json_output=}")
        formatron_vocab_processors = get_vocab_processors(is_peft_adapter)

        if enforce_json_output and len(seeded_tgt_columns) == 0:
            t0 = time.time()
            formatter_builders = get_formatter_builders(size=batch_size, stats=tgt_stats)
            engine.initialize_logits_processors(formatter_builders, formatron_vocab_processors)
            total_logits_processor_build_time += time.time() - t0

        # keep at most 500k samples in memory before decoding and writing to disk
        buffer = FixedSizeSampleBuffer(capacity=500_000)

        progress.update(completed=0, total=sample_size)
        samples_processed = 0
        while samples_processed < sample_size:
            encoded_ctx_batch = encoded_ctx_data.iloc[samples_processed : samples_processed + batch_size]
            sample_seed_batch = seed_data.iloc[samples_processed : samples_processed + batch_size]
            ctx_batch = ctx_data.iloc[samples_processed : samples_processed + batch_size]
            ctx_keys = ctx_batch[ctx_primary_key]

            if enforce_json_output and len(seeded_tgt_columns) > 0:
                t0 = time.time()
                # some columns are seeded, so we need to create a new logits processor for each batch
                formatter_builders = get_formatter_builders(seed_df=sample_seed_batch, stats=tgt_stats)
                engine.initialize_logits_processors(formatter_builders, formatron_vocab_processors)
                total_logits_processor_build_time += time.time() - t0

            outputs, metrics = engine.generate(
                encoded_ctx_batch["ctx"].tolist(),
                sampling_temperature=sampling_temperature,
                sampling_top_p=sampling_top_p,
            )
            total_tokenize_fn_time += metrics.tokenize_time
            total_generate_fn_time += metrics.generate_time

            buffer.add((outputs, ctx_keys, sample_seed_batch))
            if buffer.is_full():
                decoded_data = decode_buffered_samples(
                    buffer, engine.tokenizer, tgt_stats, tgt_context_key, max_new_tokens
                )
                persist_data_part(
                    decoded_data,
                    output_path,
                    f"{buffer.n_clears:06}.{0:06}",
                )
                buffer.clear()
            progress.update(advance=len(ctx_batch))
            samples_processed += len(ctx_batch)

        if not buffer.is_empty():
            decoded_data = decode_buffered_samples(buffer, engine.tokenizer, tgt_stats, tgt_context_key, max_new_tokens)
            persist_data_part(
                decoded_data,
                output_path,
                f"{buffer.n_clears:06}.{0:06}",
            )
            buffer.clear()
        _LOG.info(f"{total_tokenize_fn_time=:.2f}s")
        _LOG.info(f"{total_logits_processor_build_time=:.2f}s")
        _LOG.info(f"{total_generate_fn_time=:.2f}s")
        engine.cleanup()
    _LOG.info(f"GENERATE_LANGUAGE finished in {time.time() - t0_:.2f}s")
