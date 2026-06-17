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
"""Single-tool live tier: run inference-perf against each case and assert success.

This is one of two live tiers under tests/optional; the other is the cross-tool
comparison tier (../cross_tool/). The tier is the directory, not an implicit
flag: every case under single_tool/<suite>/cases is driven here, so this module
globs only its own subtree and needs no filter against the comparison cases.

Each case directory holds a vllm.yaml plus a config.yml. Cases are discovered by
glob, so the harness is not multimodal-specific and adding a suite is pure data:
drop single_tool/<suite>/cases/<case>/{vllm.yaml,config.yml} and it runs, with no
Python change. Everything else (nodeSelector inference, live-node cluster
matching, the slot semaphore, namespace lifecycle, report extraction) is shared.

The cluster_for_case fixture (see ../conftest.py) infers each case's nodeSelector
from its manifest, skips when no supplied --kubeconfigs cluster has a matching
live node, and otherwise serializes contending cases onto the available nodes.

    pytest tests/optional -m live --kubeconfigs=/path/to/kubeconfig
    pytest tests/optional -m live -k text --kubeconfigs=...   # one suite by name
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import runner
from harness.runner import Cluster

# single_tool/<suite>/cases/<case>/vllm.yaml. The suite name is part of the case
# id below so `-k <suite>` still selects a single suite. No tools/-dir filter is
# needed: comparison cases live under ../cross_tool/, a different subtree.
CASE_MANIFESTS = sorted(Path(__file__).parent.glob("*/cases/*/vllm.yaml"))


@pytest.mark.live
@pytest.mark.parametrize(
    "cluster_for_case",
    CASE_MANIFESTS,
    ids=[f"{m.parents[2].name}/{m.parent.name}" for m in CASE_MANIFESTS],
    indirect=True,
)
def test_live_case(cluster_for_case: Cluster, image: str) -> None:
    case_dir = cluster_for_case.manifest_path.parent
    runner.run_case(cluster_for_case.kubeconfig, cluster_for_case.namespace, case_dir, image)
