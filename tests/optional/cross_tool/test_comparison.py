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
"""Live cross-tool comparison: run several benchmark tools against one server.

This is the cross-tool tier under tests/optional; the other is the single-tool
live tier (../single_tool/). Every case under cross_tool/cases is a comparison
case: a case dir with a sibling ``tools/`` holding one self-contained manifest
per tool (see ../harness/comparison.py). Because the tier is the directory, this
module globs only its own subtree.

It reuses the same ``cluster_for_case`` fixture as the single-tool tier, so
cluster matching, the slot semaphore, and namespace lifecycle are all shared;
only the per-case body differs (deploy once, run every tool, record side by side).

This tier records results without asserting the tools agree. Look at
output/report.md from a real run, then add a tolerance assertion once the
expected spread between tools is known.

    pytest tests/optional -m live -k comparison --kubeconfigs=/path/to/kubeconfig
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import comparison
from harness.runner import Cluster

# cross_tool/cases/<case>/vllm.yaml. Every case here is a comparison case, so no
# tools/-dir filter is needed (the single-tool cases live under ../single_tool/).
COMPARISON_MANIFESTS = sorted(Path(__file__).parent.glob("cases/*/vllm.yaml"))


@pytest.mark.live
@pytest.mark.parametrize(
    "cluster_for_case",
    COMPARISON_MANIFESTS,
    ids=[m.parent.name for m in COMPARISON_MANIFESTS],
    indirect=True,
)
def test_comparison_case(cluster_for_case: Cluster, image: str) -> None:
    case_dir = cluster_for_case.manifest_path.parent
    results = comparison.run_comparison(
        cluster_for_case.kubeconfig, cluster_for_case.namespace, case_dir, image
    )
    # Side-by-side tier: assert only that every tool ran and was parsed into
    # metrics. Tolerance (do the tools agree?) is deliberately left for later,
    # once a real report shows the expected spread.
    unparsed = [tool for tool, metrics in results.items() if metrics is None]
    assert not unparsed, f"tools produced no parseable metrics: {unparsed} (see output/<tool>.log)"
