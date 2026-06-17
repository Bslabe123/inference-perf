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
"""Run one end-to-end case against a cluster.

Owns deploy, run, verification, and artifact extraction for a single case. The
namespace lifecycle and concurrency control live in the fixture (conftest.py);
this module assumes the namespace already exists and does not delete it.

Nothing here is multimodal-specific: a case is a directory holding a server
manifest (``vllm.yaml``) and an inference-perf ``config.yml``. The Deployment to
wait on is read from the manifest, so any suite can bring its own server.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness import requirements
from harness.parsers import extract_block


@dataclass
class Cluster:
    """A matched cluster plus the namespace and manifest for one case."""

    kubeconfig: str | None
    namespace: str
    manifest_path: Path


DEFAULT_IMAGE = "quay.io/inference-perf/inference-perf:latest"
_TEMPLATE_IMAGE = "quay.io/inference-perf/inference-perf:latest"
_CONFIG_MAP = "inference-perf-config"
_JOB = "inference-perf"
_JOB_TIMEOUT = "600s"
_ROLLOUT_TIMEOUT = "900s"
# deploy/manifests.yaml relative to the repo root (this file is 3 levels deep
# under tests/optional/harness/).
_MANIFEST_TEMPLATE = Path(__file__).resolve().parents[3] / "deploy" / "manifests.yaml"


def _kubectl(
    kubeconfig: str | None,
    namespace: str | None,
    *args: str,
    stdin: str | None = None,
    capture: bool = False,
    check: bool = True,
) -> str:
    cmd = ["kubectl", *args]
    if namespace:
        cmd[1:1] = ["-n", namespace]
    if kubeconfig:
        cmd[1:1] = ["--kubeconfig", kubeconfig]
    result = subprocess.run(cmd, input=stdin, capture_output=capture, text=True, check=check)
    return result.stdout if capture else ""


def _copy_hf_secret(kubeconfig: str | None, namespace: str) -> None:
    """Copy hf-secret from the default namespace into the case namespace."""
    raw = _kubectl(
        kubeconfig,
        "default",
        "get",
        "secret",
        "hf-secret",
        "-o",
        "json",
        capture=True,
        check=False,
    )
    if not raw.strip():
        # Mirrors the bash warning: gated models will fail to start without it.
        print("Warning: hf-secret not found in 'default' namespace.")
        return
    secret = json.loads(raw)
    meta = secret.get("metadata", {})
    for field in ("namespace", "resourceVersion", "uid", "creationTimestamp"):
        meta.pop(field, None)
    secret["metadata"] = meta
    _kubectl(kubeconfig, namespace, "apply", "-f", "-", stdin=json.dumps(secret))


def deploy_server(
    kubeconfig: str | None,
    namespace: str,
    case_dir: Path,
    deployments: list[str] | None = None,
) -> None:
    """Bring up the model server for a case and block until it has rolled out.

    Tool-agnostic: applies ``vllm.yaml`` and waits on each Deployment it declares
    (inferred from the manifest unless the caller names them). Shared by the
    single-tool runner and the cross-tool comparison runner, which both need the
    same server up before driving any load. Copies hf-secret first so gated
    models can start.
    """
    case_dir = Path(case_dir)
    _copy_hf_secret(kubeconfig, namespace)

    vllm = case_dir / "vllm.yaml"
    if vllm.is_file():
        _kubectl(kubeconfig, namespace, "apply", "-f", str(vllm))
    if deployments is not None:
        rollout_targets = deployments
    else:
        rollout_targets = requirements.deployment_names(vllm) if vllm.is_file() else []
    for deployment in rollout_targets:
        _kubectl(
            kubeconfig,
            namespace,
            "rollout",
            "status",
            f"deployment/{deployment}",
            f"--timeout={_ROLLOUT_TIMEOUT}",
        )


def run_job(
    kubeconfig: str | None,
    namespace: str,
    manifest: str,
    job_name: str,
    timeout: str = _JOB_TIMEOUT,
) -> str:
    """Apply one Job manifest, wait for it to complete, and return its pod logs.

    ``manifest`` is rendered YAML (already substituted), applied via stdin so the
    caller controls templating. Deletes any prior Job of the same name first so a
    re-run is clean. Lets ``kubectl wait`` raise on timeout/failure so the test
    fails loudly rather than reading partial logs.
    """
    _kubectl(kubeconfig, namespace, "delete", "job", job_name, "--ignore-not-found")
    _kubectl(kubeconfig, namespace, "apply", "-f", "-", stdin=manifest)
    _kubectl(
        kubeconfig,
        namespace,
        "wait",
        "--for=condition=complete",
        f"job/{job_name}",
        f"--timeout={timeout}",
    )
    return _kubectl(kubeconfig, namespace, "logs", f"job/{job_name}", capture=True)


def run_case(
    kubeconfig: str | None,
    namespace: str,
    case_dir: Path,
    image: str = DEFAULT_IMAGE,
    deployments: list[str] | None = None,
) -> None:
    """Deploy the model server, run inference-perf, and assert all requests succeeded.

    The Deployment(s) to wait on default to whatever ``vllm.yaml`` declares (read
    from the manifest, not hardcoded); pass ``deployments`` to override. Raises
    AssertionError on zero successes or any failures; lets subprocess errors
    (rollout/job timeouts) propagate so the test fails loudly rather than hanging.
    """
    case_dir = Path(case_dir)

    # Step 1: model server.
    deploy_server(kubeconfig, namespace, case_dir, deployments)

    # Step 2: config.
    cm = _kubectl(
        kubeconfig,
        namespace,
        "create",
        "configmap",
        _CONFIG_MAP,
        f"--from-file={case_dir / 'config.yml'}",
        "--dry-run=client",
        "-o",
        "yaml",
        capture=True,
    )
    _kubectl(kubeconfig, namespace, "apply", "-f", "-", stdin=cm)

    # Step 3: inference-perf job (case override, else templated default).
    case_manifest = case_dir / "manifest.yaml"
    if case_manifest.is_file():
        manifest = case_manifest.read_text()
    else:
        manifest = _MANIFEST_TEMPLATE.read_text().replace(_TEMPLATE_IMAGE, image)
    logs = run_job(kubeconfig, namespace, manifest, _JOB)

    # Step 4: extract, persist, verify.
    output_dir = case_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = extract_block(logs, "=== START_SUMMARY ===", "=== END_SUMMARY ===")
    stage_0 = extract_block(logs, "=== START_STAGE_0 ===", "=== END_STAGE_0 ===")
    (output_dir / "summary_lifecycle_metrics.json").write_text(summary)
    (output_dir / "stage_0_lifecycle_metrics.json").write_text(stage_0)

    assert summary, "no summary block found in job logs"
    metrics = json.loads(summary)
    successes = metrics.get("successes", {}).get("count", 0)
    failures = metrics.get("failures", {}).get("count", 0)
    assert successes, f"zero successful requests for case {case_dir.name}"
    assert not failures, f"case {case_dir.name} had {failures} failed requests"
