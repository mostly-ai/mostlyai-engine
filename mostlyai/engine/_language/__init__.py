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

from transformers import AutoConfig

from mostlyai.engine._language.lstm import register_mostly_lstm_model

# Guard against duplicate third-party config registrations on GPU CI
_ORIGINAL_AUTOCONFIG_REGISTER = AutoConfig.register


def _safe_autoconfig_register(name, config_class):
    try:
        _ORIGINAL_AUTOCONFIG_REGISTER(name, config_class)
    except ValueError as e:
        # Some environments register AMD AIMv2 config multiple times; ignore duplicates for this key
        if str(name).lower() in {"aimv2"}:
            return
        raise e


AutoConfig.register = _safe_autoconfig_register

register_mostly_lstm_model()
