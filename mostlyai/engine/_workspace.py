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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Callable

from mostlyai.engine._common import read_json, write_json
from mostlyai.engine.domain import ModelType

_LOG = logging.getLogger(__name__)


@dataclass
class PathDesc:
    root_dir: Path | None = None
    parts: list[str] | None = None
    is_multiple: bool = False
    fetch_multiple: Callable | None = None
    read_handler: Callable | None = None
    write_handler: Callable | None = None

    def __post_init__(self):
        all_parts = (list(self.root_dir.parts) if isinstance(self.root_dir, Path) else []) + (
            self.parts if isinstance(self.parts, list) else []
        )
        self.path = Path(*all_parts)

    def read(self):
        if not self.read_handler:
            raise RuntimeError(f"No read handler was defined for {self.path}")
        return self.read_handler(self.path)

    def write(self, data: Any):
        if not self.write_handler:
            raise RuntimeError(f"No write handler was defined for {self.path}")
        return self.write_handler(data, self.path)

    def fetch_all(self):
        if not self.fetch_multiple:
            raise RuntimeError(f"No fetch multiple handler was defined for {self.path}")
        return self.fetch_multiple(self.path)


@dataclass
class JsonPathDesc(PathDesc):
    read_handler: Callable | None = read_json
    write_handler: Callable | None = write_json


def fetch_sorted_glob(pattern: str):
    def fetch_func(path: Path):
        return sorted(list(path.glob(pattern)))

    return fetch_func


class Workspace:
    def __init__(
        self,
        ws_path: str | Path,
    ):
        self._ws_path: Path = Path(ws_path)
        self._make_workspace_objects_and_paths()

    def _make_workspace_objects_and_paths(self):
        fetch_part_parquets = fetch_sorted_glob("part.*.parquet")
        fetch_part_trn_parquets = fetch_sorted_glob("part.*-trn.parquet")
        fetch_part_val_parquets = fetch_sorted_glob("part.*-val.parquet")
        fetch_part_jsons = fetch_sorted_glob("part.*.json")

        def make_path_desc(path_desc_cls: type | None = None, **kwargs):
            path_desc_cls = path_desc_cls or PathDesc
            return path_desc_cls(root_dir=self._ws_path, **kwargs)

        def make_json_path_desc(**kwargs):
            return make_path_desc(path_desc_cls=JsonPathDesc, **kwargs)

        def make_stats_json_path_desc(**kwargs):
            return make_path_desc(path_desc_cls=JsonPathDesc, **kwargs)

        # split-related
        tgt_data = ["OriginalData", "tgt-data"]
        self.tgt_data_path = self._ws_path / Path(*tgt_data)
        self.tgt_data = make_path_desc(
            parts=tgt_data,
            fetch_multiple=fetch_part_parquets,
        )
        self.tgt_trn_data = make_path_desc(
            parts=tgt_data,
            fetch_multiple=fetch_part_trn_parquets,
        )
        self.tgt_val_data = make_path_desc(
            parts=tgt_data,
            fetch_multiple=fetch_part_val_parquets,
        )
        tgt_meta = ["OriginalData", "tgt-meta"]
        self.tgt_meta_path = self._ws_path / Path(*tgt_meta)
        self.tgt_encoding_types = make_json_path_desc(parts=tgt_meta + ["encoding-types.json"])
        self.tgt_keys = make_json_path_desc(parts=tgt_meta + ["keys.json"])
        ctx_data = ["OriginalData", "ctx-data"]
        self.ctx_data_path = self._ws_path / Path(*ctx_data)
        self.ctx_data = make_path_desc(
            parts=ctx_data,
            fetch_multiple=fetch_part_parquets,
        )
        self.ctx_trn_data = make_path_desc(
            parts=ctx_data,
            fetch_multiple=fetch_part_trn_parquets,
        )
        self.ctx_val_data = make_path_desc(
            parts=ctx_data,
            fetch_multiple=fetch_part_val_parquets,
        )
        ctx_meta = ["OriginalData", "ctx-meta"]
        self.ctx_meta_path = self._ws_path / Path(*ctx_meta)
        self.ctx_encoding_types = make_json_path_desc(parts=ctx_meta + ["encoding-types.json"])
        self.ctx_keys = make_json_path_desc(parts=ctx_meta + ["keys.json"])

        # analyze-related
        tgt_stats = ["ModelStore", "tgt-stats"]
        self.tgt_stats_path = Path(self._ws_path) / Path(*tgt_stats)
        self.tgt_all_stats = make_path_desc(
            parts=tgt_stats,
            fetch_multiple=fetch_part_jsons,
        )
        self.tgt_stats = make_stats_json_path_desc(parts=tgt_stats + ["stats.json"])
        ctx_stats = ["ModelStore", "ctx-stats"]
        self.ctx_stats_path = Path(self._ws_path) / Path(*ctx_stats)
        self.ctx_all_stats = make_path_desc(
            parts=ctx_stats,
            fetch_multiple=fetch_part_jsons,
        )
        self.ctx_stats = make_stats_json_path_desc(parts=ctx_stats + ["stats.json"])

        # Encode-related
        encoded_data = ["OriginalData", "encoded-data"]
        self.encoded_data_path = Path(self._ws_path) / Path(*encoded_data)
        self.encoded_data_val = make_path_desc(parts=encoded_data, fetch_multiple=fetch_part_val_parquets)
        self.encoded_data_trn = make_path_desc(parts=encoded_data, fetch_multiple=fetch_part_trn_parquets)

        # Train-related
        model_data = ["ModelStore", "model-data"]
        self.model_path: Path = Path(self._ws_path) / Path(*model_data)
        self.model_optimizer_path: Path = self.model_path / "optimizer.pt"
        self.model_lr_scheduler_path: Path = self.model_path / "lr-scheduler.pt"
        self.model_dp_accountant_path: Path = self.model_path / "dp-accountant.pt"
        self.model_tabular_weights_path: Path = self.model_path / "model-weights.pt"
        self.model_progress_messages_path: Path = self.model_path / "progress-messages.csv"
        self.model_configs = make_json_path_desc(parts=model_data + ["model-configs.json"])

        # Generate-related
        generated_data = ["SyntheticData"]
        self.generated_data_path = Path(self._ws_path) / Path(*generated_data)
        self.generated_data = make_path_desc(parts=generated_data, fetch_multiple=fetch_part_parquets)


def ensure_workspace_dir(workspace_dir: str | Path) -> Path:
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(exist_ok=True, parents=True)
    return workspace_dir


def reset_dir(path: Path) -> None:
    """Create directory if not exists. Otherwise remove all files from it"""
    if os.path.exists(path):
        _LOG.info(f"clean `{path}`")
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                os.remove(os.path.join(path, file))
    else:
        _LOG.info(f"create `{path}`")
        os.makedirs(path)


def resolve_model_type(workspace_dir: str | Path) -> ModelType:
    """
    Determine the model type based on encoding types in the workspace target metadata.
    """
    workspace_dir = ensure_workspace_dir(workspace_dir)
    workspace = Workspace(workspace_dir)
    stats = workspace.tgt_stats.read()
    columns = stats.get("columns", {})
    if len(columns) == 0:
        return ModelType.tabular
    # fetch encoding type from the first column
    encoding_type = next(iter(columns.values())).get("encoding_type")
    if encoding_type.startswith(ModelType.tabular):
        model_type = ModelType.tabular
    elif encoding_type.startswith(ModelType.language):
        model_type = ModelType.language
    else:
        raise ValueError(
            f"Unknown encoding type, valid encoding types start with TABULAR_ or LANGUAGE_: {encoding_type}"
        )
    return model_type
