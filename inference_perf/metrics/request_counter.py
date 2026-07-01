# Copyright 2026 The Kubernetes Authors.
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
from collections import defaultdict
from typing import Any, Iterable, List

from inference_perf.apis import RequestLifecycleMetric

UNKNOWN_MODEL = "unknown"


class RequestSentCounter:
    """In-process counter for the total number of requests sent to the model server.

    A request is considered "sent" once its lifecycle completes and produces a
    RequestLifecycleMetric, which happens for both successful and failed requests
    (errors are captured rather than dropped). The counter keeps a running total
    plus a breakdown by (stage, model, status) so per-stage and per-model send
    volumes are distinguishable. It is a plain in-process counter surfaced in the
    run report and logs, not a scraped Prometheus endpoint.
    """

    def __init__(self) -> None:
        self.total: int = 0
        # keyed by (stage_id, model, status) -> count
        self._breakdown: "defaultdict[tuple[Any, str, str], int]" = defaultdict(int)

    def observe(self, metric: RequestLifecycleMetric) -> None:
        """Record a single sent request."""
        status = "failure" if metric.error is not None else "success"
        model = metric.model or UNKNOWN_MODEL
        self.total += 1
        self._breakdown[(metric.stage_id, model, status)] += 1

    @classmethod
    def from_metrics(cls, metrics: Iterable[RequestLifecycleMetric]) -> "RequestSentCounter":
        counter = cls()
        for metric in metrics:
            counter.observe(metric)
        return counter

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the counter for reports."""
        try:
            items = sorted(
                self._breakdown.items(),
                key=lambda item: (item[0][0] if item[0][0] is not None else -1, item[0][1], item[0][2]),
            )
        except TypeError:
            # Keys are normally (int|None, str, str); if a caller passes exotic
            # (e.g. mock) values that are not comparable, fall back to insertion order
            # rather than failing to serialize the report.
            items = list(self._breakdown.items())
        by_stage_model_status: List[dict[str, Any]] = [
            {
                "stage_id": stage_id,
                "model": model,
                "status": status,
                "count": count,
            }
            for (stage_id, model, status), count in items
        ]
        return {
            "total": self.total,
            "by_stage_model_status": by_stage_model_status,
        }
