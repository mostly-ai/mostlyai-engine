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

from mostlyai.engine._memory import extract_memory_from_string


def test_extract_memory_from_string():
    assert extract_memory_from_string("3.2GB") == int(3.2 * 1024**3)
    assert extract_memory_from_string("3.2Gi") == int(3.2 * 1024**3)
    assert extract_memory_from_string(" 3 g ") == 3 * 1024**3
    assert extract_memory_from_string("0.23GB") == int(0.23 * 1024**3)
    assert extract_memory_from_string("32804 gb") == 32804 * 1024**3
    assert extract_memory_from_string("4B") == 4
    assert extract_memory_from_string("4") == 4
    assert extract_memory_from_string("") is None
    assert extract_memory_from_string() is None
