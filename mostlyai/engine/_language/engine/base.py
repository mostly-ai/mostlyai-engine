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

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class EngineMetrics:
    tokenize_time: float
    generate_time: float


class LanguageEngine(ABC):
    @abstractmethod
    def generate(
        self, text: list[str], sampling_temperature: float, sampling_top_p: float
    ) -> tuple[list[int], EngineMetrics]:
        pass

    @abstractmethod
    def get_default_batch_size(self) -> int:
        pass

    @abstractmethod
    def supports_json_enforcing(self) -> bool:
        pass

    @abstractmethod
    def cleanup(self):
        pass

    def update_json_constraints(self, schemas: list[BaseModel] | None) -> None:
        """Update JSON schema constraints for the next generation call.

        Args:
            schemas: Schema constraints to apply to the next generate() call.
                    None to clear any existing constraints.

        Default implementation does nothing.
        Engines that support JSON constraints should override this method.
        """
        pass

    def can_reuse_schemas(self) -> bool:
        """Whether the engine can reuse JSON schema constraints across batches with different sizes.

        Returns:
            True if the engine can handle variable batch sizes with reused schema constraints,
            False if schema constraints need to be recreated for each batch.
        """
        return True
