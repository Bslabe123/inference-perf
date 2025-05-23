# Copyright 2025 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from abc import ABC, abstractmethod
from typing import Any, List

from inference_perf.report import ReportFile


class MetricsCollectorReporter(ABC):
    """
    Responsible for collecting information for and generating reports for a particicular report subtype
    (i.e. request lifecycle metrics, model server metrics, accelerator metrics, etc)
    """

    @abstractmethod
    async def reports(self, report_config: Any) -> List[ReportFile]:
        raise NotImplementedError
