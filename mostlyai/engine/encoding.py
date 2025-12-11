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

from pathlib import Path

from mostlyai.engine._common import ProgressCallback


def encode(
    *,
    workspace_dir: str | Path = "engine-ws",
    update_progress: ProgressCallback | None = None,
) -> None:
    """
    Encodes data in the workspace that has already been split and analyzed.

    Creates the following folder structure within the `workspace_dir`:

    - `OriginalData/encoded-data`: Encoded data for training, stored as parquet files.

    Args:
        workspace_dir: Directory path for workspace.
        update_progress: Callback for progress updates.
    """
    from mostlyai.engine._tabular.encoding import encode as encode_tabular

    return encode_tabular(workspace_dir=workspace_dir, update_progress=update_progress)
