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

import os
from pathlib import Path
from unittest import mock


from mostlyai.engine._common import read_json, write_json
from mostlyai.engine._workspace import Workspace

FIXTURES_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"))


class TestWorkspace:
    def test_workspace_all_objects(self):
        ws_path = Path(FIXTURES_PATH) / "workspace" / "all"
        ws = Workspace(ws_path)

        # Split-related
        assert ws.tgt_data_path == ws_path / "OriginalData" / "tgt-data"
        tgt_data_file_names = [i.name for i in ws.tgt_data.fetch_all()]
        assert tgt_data_file_names == ["part.000000-trn.parquet", "part.000000-val.parquet"]
        assert isinstance(ws.tgt_encoding_types.read(), dict)
        assert isinstance(ws.tgt_keys.read(), dict)

        assert ws.ctx_data_path == ws_path / "OriginalData" / "ctx-data"
        ctx_data_file_names = [i.name for i in ws.ctx_data.fetch_all()]
        assert ctx_data_file_names == ["part.000000-trn.parquet", "part.000000-val.parquet"]
        assert isinstance(ws.ctx_encoding_types.read(), dict)
        assert isinstance(ws.ctx_keys.read(), dict)

        # Analyze-related
        assert ws.tgt_stats_path == Path(ws_path) / "ModelStore" / "tgt-stats"
        tgt_all_stats_file_names = [i.name for i in ws.tgt_all_stats.fetch_all()]
        assert tgt_all_stats_file_names == ["part.000000-trn.json", "part.000000-val.json"]
        assert isinstance(ws.tgt_stats.read(), dict)
        assert ws.ctx_stats_path == Path(ws_path) / "ModelStore" / "ctx-stats"
        ctx_all_stats_file_names = [i.name for i in ws.ctx_all_stats.fetch_all()]
        assert ctx_all_stats_file_names == ["part.000000-trn.json", "part.000000-val.json"]
        assert isinstance(ws.tgt_stats.read(), dict)

        # Encode-related
        assert ws.encoded_data_path == Path(ws_path) / "OriginalData" / "encoded-data"
        assert len(ws.encoded_data_val.fetch_all()) == 1
        assert len(ws.encoded_data_trn.fetch_all()) == 1

        # Train-related
        assert ws.model_path == Path(ws_path) / "ModelStore" / "model-data"
        assert ws.model_tabular_weights_path.exists()
        assert isinstance(ws.model_configs.read(), dict)

        # Generate-related
        assert ws.generated_data_path == Path(ws_path) / "SyntheticData"
        generated_data_file_names = [i.name for i in ws.generated_data.fetch_all()]
        assert generated_data_file_names == ["part.000001.parquet", "part.000002.parquet"]

    def test_read_write_json(self):
        ws_path = Path(FIXTURES_PATH) / "workspace" / "some"
        ws = Workspace(ws_path)

        assert ws.tgt_keys.read_handler == read_json
        ws.tgt_keys.read() == {"context_key": "__primary_key"}
        assert ws.tgt_keys.write_handler == write_json
        with mock.patch.object(ws.tgt_keys, "write_handler") as write_mock:
            new_key_data = {"new_key": "test_key"}
            ws.tgt_keys.write(new_key_data)
            assert write_mock.call_args[0] == (
                new_key_data,
                ws_path / "OriginalData" / "tgt-meta" / "keys.json",
            )
