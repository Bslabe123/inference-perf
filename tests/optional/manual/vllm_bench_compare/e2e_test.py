#!/usr/bin/env python3
"""Compare inference-perf vs ``vllm bench serve`` head-to-head.

For each case under ``cases/<name>/``, two namespaces are created in parallel::

    <prefix>-<case>-ip   inference-perf + its own vLLM
    <prefix>-<case>-vb   vllm bench serve + its own vLLM

Both vLLMs come up cold from the same ``vllm.yaml`` so neither tool benefits
from warm KV caches or compilation artifacts left by a prior run. Once both
servers are Ready, the two client Jobs run concurrently; results are scraped
from pod logs, written to ``cases/<name>/output/``, and a side-by-side
comparison is rendered by ``render_comparison.py``.

This is a manual, cluster-dependent test. It talks to whatever cluster your
current kubeconfig (or in-cluster service account) points at, via the
official Kubernetes Python client. Install the extra first::

    pip install -e ".[vllm-bench-compare]"   # or: pip install kubernetes

Then run, e.g.::

    python3 tests/optional/manual/vllm_bench_compare/e2e_test.py -k qwen3_coder_480b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# render_comparison lives next to this script; make sure it is importable
# regardless of the caller's working directory.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import render_comparison  # noqa: E402

CASES_DIR = SCRIPT_DIR / "cases"

DEFAULT_NAMESPACE_PREFIX = "vllm-bench-compare"
DEFAULT_IP_IMAGE = "quay.io/inference-perf/inference-perf:latest"
DEFAULT_ROLLOUT_TIMEOUT = "30m"
DEFAULT_JOB_TIMEOUT = "60m"

IP_CONFIGMAP = "inference-perf-config"
IP_JOB = "inference-perf"
VB_JOB = "vllm-bench"

# How often we poll the API server while waiting on rollouts / Jobs.
POLL_INTERVAL_S = 10
# How long to wait for a namespace to finish terminating before giving up.
NAMESPACE_DELETE_TIMEOUT_S = 600


# --------------------------------------------------------------------------- #
# Kubernetes access helpers
# --------------------------------------------------------------------------- #
@dataclass
class Cluster:
    """A bundle of typed API clients sharing one ApiClient connection pool."""

    core: client.CoreV1Api
    apps: client.AppsV1Api
    batch: client.BatchV1Api


def load_kube_config() -> None:
    """Load kubeconfig, falling back to in-cluster config when running in a pod."""
    try:
        config.load_kube_config()
    except Exception:
        config.load_incluster_config()


def make_cluster() -> Cluster:
    """Build a Cluster with its own ApiClient.

    Parallel branches each get their own Cluster so we never share a single
    ApiClient across threads.
    """
    api = client.ApiClient()
    return Cluster(
        core=client.CoreV1Api(api),
        apps=client.AppsV1Api(api),
        batch=client.BatchV1Api(api),
    )


def parse_duration(value: str) -> int:
    """Parse a kubectl-style duration like ``30m`` / ``90s`` / ``1h`` into seconds."""
    match = re.fullmatch(r"(\d+)([smh]?)", value.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"invalid duration: {value!r}")
    amount = int(match.group(1))
    unit = match.group(2) or "s"
    return amount * {"s": 1, "m": 60, "h": 3600}[unit]


# --------------------------------------------------------------------------- #
# Manifest application
# --------------------------------------------------------------------------- #
def apply_doc(cl: Cluster, ns: str, doc: dict[str, Any]) -> None:
    """Create one manifest doc, replacing it in place if it already exists.

    Fresh namespaces are the normal path, so create succeeds outright; the
    replace fallback only matters when re-running inside an existing namespace.
    """
    kind = doc["kind"]
    name = doc["metadata"]["name"]

    create: Callable[[], Any]
    replace: Optional[Callable[[], Any]]
    if kind == "Deployment":
        create = lambda: cl.apps.create_namespaced_deployment(ns, doc)  # noqa: E731
        replace = lambda: cl.apps.replace_namespaced_deployment(name, ns, doc)  # noqa: E731
    elif kind == "Service":
        create = lambda: cl.core.create_namespaced_service(ns, doc)  # noqa: E731
        replace = None  # ClusterIP/resourceVersion make replace fiddly; leave existing
    elif kind == "ConfigMap":
        create = lambda: cl.core.create_namespaced_config_map(ns, doc)  # noqa: E731
        replace = lambda: cl.core.replace_namespaced_config_map(name, ns, doc)  # noqa: E731
    elif kind == "Job":
        # Callers delete the Job first, so a plain create is expected to win.
        create = lambda: cl.batch.create_namespaced_job(ns, doc)  # noqa: E731
        replace = None
    else:
        raise ValueError(f"unsupported manifest kind: {kind}")

    try:
        create()
    except ApiException as exc:
        if exc.status != 409:
            raise
        if replace is not None:
            replace()


# --------------------------------------------------------------------------- #
# Namespace + secret lifecycle
# --------------------------------------------------------------------------- #
def copy_hf_secret(cl: Cluster, ns: str) -> None:
    """Copy the ``hf-secret`` from ``default`` into ``ns`` so gated models load."""
    try:
        src = cl.core.read_namespaced_secret("hf-secret", "default")
    except ApiException as exc:
        if exc.status == 404:
            print(
                "Warning: hf-secret not found in 'default' namespace; gated models will fail to load.",
                file=sys.stderr,
            )
            return
        raise

    body = client.V1Secret(
        metadata=client.V1ObjectMeta(name="hf-secret"),
        data=src.data,
        string_data=src.string_data,
        type=src.type,
    )
    try:
        cl.core.create_namespaced_secret(ns, body)
    except ApiException as exc:
        if exc.status != 409:
            raise
        cl.core.replace_namespaced_secret("hf-secret", ns, body)


def ensure_namespace(cl: Cluster, ns: str) -> None:
    try:
        cl.core.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=ns)))
    except ApiException as exc:
        if exc.status != 409:
            raise
    copy_hf_secret(cl, ns)


def cleanup_leftover_namespaces(cl: Cluster, prefix: str) -> None:
    leftover = [n.metadata.name for n in cl.core.list_namespace().items if n.metadata.name.startswith(f"{prefix}-")]
    if not leftover:
        return
    print(f"Deleting leftover namespaces: {' '.join(leftover)}")
    for ns in leftover:
        delete_namespace(cl, ns, wait=True)


def delete_namespace(cl: Cluster, ns: str, wait: bool) -> None:
    try:
        cl.core.delete_namespace(ns)
    except ApiException as exc:
        if exc.status == 404:
            return
        raise
    if not wait:
        return
    deadline = time.time() + NAMESPACE_DELETE_TIMEOUT_S
    while time.time() < deadline:
        try:
            cl.core.read_namespace(ns)
        except ApiException as exc:
            if exc.status == 404:
                return
            raise
        time.sleep(POLL_INTERVAL_S)
    print(f"Warning: namespace {ns} still terminating after timeout", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Waiting on workloads
# --------------------------------------------------------------------------- #
def bring_up_vllm(cl: Cluster, ns: str, vllm_yaml: Path, rollout_timeout_s: int) -> None:
    for doc in yaml.safe_load_all(vllm_yaml.read_text()):
        if doc:
            apply_doc(cl, ns, doc)
    wait_deployment_ready(cl, ns, "vllm-server", rollout_timeout_s)


def wait_deployment_ready(cl: Cluster, ns: str, name: str, timeout_s: int) -> None:
    """Block until the Deployment is fully rolled out, like ``kubectl rollout status``."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        dep = cl.apps.read_namespaced_deployment_status(name, ns)
        desired = dep.spec.replicas if dep.spec.replicas is not None else 1
        st = dep.status
        observed_ok = (st.observed_generation or 0) >= (dep.metadata.generation or 0)
        if (
            observed_ok
            and (st.updated_replicas or 0) >= desired
            and (st.available_replicas or 0) >= desired
            and (st.replicas or 0) == desired
        ):
            return
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"deployment/{name} in {ns} not ready within {timeout_s}s")


def delete_job(cl: Cluster, ns: str, name: str) -> None:
    """Delete a Job (and its pods) and block until it is gone."""
    try:
        cl.batch.delete_namespaced_job(name, ns, propagation_policy="Background")
    except ApiException as exc:
        if exc.status != 404:
            raise
        return
    while True:
        try:
            cl.batch.read_namespaced_job(name, ns)
        except ApiException as exc:
            if exc.status == 404:
                return
            raise
        time.sleep(2)


def wait_job_complete(cl: Cluster, ns: str, name: str, timeout_s: int) -> None:
    """Block until the Job reports Complete, raising if it Fails or times out."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        job = cl.batch.read_namespaced_job_status(name, ns)
        for cond in job.status.conditions or []:
            if cond.type == "Complete" and cond.status == "True":
                return
            if cond.type == "Failed" and cond.status == "True":
                raise RuntimeError(f"job/{name} in {ns} failed: {cond.message}")
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"job/{name} in {ns} did not complete within {timeout_s}s")


# --------------------------------------------------------------------------- #
# Client runs
# --------------------------------------------------------------------------- #
def run_inference_perf(
    cl: Cluster,
    ns: str,
    config_file: Path,
    template_manifest: Path,
    ip_image: str,
    job_timeout_s: int,
) -> None:
    config_map = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=IP_CONFIGMAP),
        data={"config.yml": config_file.read_text()},
    )
    try:
        cl.core.create_namespaced_config_map(ns, config_map)
    except ApiException as exc:
        if exc.status != 409:
            raise
        cl.core.replace_namespaced_config_map(IP_CONFIGMAP, ns, config_map)

    delete_job(cl, ns, IP_JOB)
    # Mirror the shell's sed: swap the default image for the requested one.
    manifest_text = template_manifest.read_text().replace(DEFAULT_IP_IMAGE, ip_image)
    for doc in yaml.safe_load_all(manifest_text):
        if doc:
            apply_doc(cl, ns, doc)

    wait_job_complete(cl, ns, IP_JOB, job_timeout_s)


def run_vllm_bench(cl: Cluster, ns: str, job_yaml: Path, job_timeout_s: int) -> None:
    delete_job(cl, ns, VB_JOB)
    for doc in yaml.safe_load_all(job_yaml.read_text()):
        if doc:
            apply_doc(cl, ns, doc)
    wait_job_complete(cl, ns, VB_JOB, job_timeout_s)


# --------------------------------------------------------------------------- #
# Result extraction
# --------------------------------------------------------------------------- #
def job_pod_logs(cl: Cluster, ns: str, app_label: str) -> str:
    """Return logs from the most recent pod of the Job labelled ``app=<app_label>``."""
    pods = cl.core.list_namespaced_pod(ns, label_selector=f"app={app_label}").items
    if not pods:
        raise RuntimeError(f"no pods found for app={app_label} in {ns}")
    pods.sort(key=lambda p: p.metadata.creation_timestamp or 0)
    return str(cl.core.read_namespaced_pod_log(pods[-1].metadata.name, ns))


def extract_block(text: str, start_marker: str, end_marker: str) -> str:
    """Return the lines strictly between the start and end marker lines."""
    out: list[str] = []
    capturing = False
    for line in text.splitlines():
        if start_marker in line:
            capturing = True
            continue
        if end_marker in line:
            capturing = False
            continue
        if capturing:
            out.append(line)
    return "\n".join(out)


def extract_inference_perf_results(cl: Cluster, ns: str, out_dir: Path) -> None:
    logs = job_pod_logs(cl, ns, IP_JOB)
    block = extract_block(logs, "=== START_SUMMARY ===", "=== END_SUMMARY ===")
    if not block.strip():
        raise RuntimeError(f"empty inference-perf summary extracted from {ns}")
    (out_dir / "inference_perf.json").write_text(block + "\n")


def extract_vllm_bench_results(cl: Cluster, ns: str, out_dir: Path) -> None:
    logs = job_pod_logs(cl, ns, VB_JOB)
    block = extract_block(logs, "=== START_VLLM_BENCH ===", "=== END_VLLM_BENCH ===")
    if not block.strip():
        raise RuntimeError(f"empty vllm bench result extracted from {ns}")
    (out_dir / "vllm_bench.json").write_text(block + "\n")


# --------------------------------------------------------------------------- #
# Case orchestration
# --------------------------------------------------------------------------- #
class CaseResult:
    PASSED = 0
    FAILED = 1
    SKIPPED = 2


def run_parallel(*calls: Callable[[], None]) -> bool:
    """Run callables concurrently; return True only if all succeed.

    Every callable is awaited before returning so no work is left dangling,
    matching the shell's ``wait pid1 pid2`` semantics.
    """
    ok = True
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = [pool.submit(fn) for fn in calls]
        for fut in futures:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"Error: {exc}", file=sys.stderr)
                ok = False
    return ok


def run_case(
    case_dir: Path,
    case_name: str,
    ns_ip: str,
    ns_vb: str,
    template_manifest: Path,
    ip_image: str,
    rollout_timeout_s: int,
    job_timeout_s: int,
) -> int:
    vllm_yaml = case_dir / "vllm.yaml"
    ip_config = case_dir / "inference_perf_config.yml"
    vb_job = case_dir / "vllm_bench_job.yaml"
    out_dir = case_dir / "output"

    for required in (vllm_yaml, ip_config, vb_job):
        if not required.is_file():
            print(f"Skipping {case_name}: missing {required}")
            return CaseResult.SKIPPED

    out_dir.mkdir(parents=True, exist_ok=True)

    # Separate clusters per branch keep each thread on its own ApiClient.
    cl_ip = make_cluster()
    cl_vb = make_cluster()

    print(f"=== [{case_name}] Ensuring namespaces {ns_ip} and {ns_vb} ===")
    if not run_parallel(
        lambda: ensure_namespace(cl_ip, ns_ip),
        lambda: ensure_namespace(cl_vb, ns_vb),
    ):
        return CaseResult.FAILED

    print(f"=== [{case_name}] Bringing up fresh vLLM in both namespaces (parallel) ===")
    if not run_parallel(
        lambda: bring_up_vllm(cl_ip, ns_ip, vllm_yaml, rollout_timeout_s),
        lambda: bring_up_vllm(cl_vb, ns_vb, vllm_yaml, rollout_timeout_s),
    ):
        print("Error: vLLM rollout failed in one or both namespaces", file=sys.stderr)
        return CaseResult.FAILED

    print(f"=== [{case_name}] Running inference-perf and vllm bench (parallel) ===")
    client_ok = run_parallel(
        lambda: run_inference_perf(cl_ip, ns_ip, ip_config, template_manifest, ip_image, job_timeout_s),
        lambda: run_vllm_bench(cl_vb, ns_vb, vb_job, job_timeout_s),
    )

    # Extract whatever results we can even if a Job failed; logs may still help.
    print(f"=== [{case_name}] Extracting results ===")
    extractors: tuple[Callable[[], None], ...] = (
        lambda: extract_inference_perf_results(cl_ip, ns_ip, out_dir),
        lambda: extract_vllm_bench_results(cl_vb, ns_vb, out_dir),
    )
    for extract in extractors:
        try:
            extract()
        except Exception as exc:  # noqa: BLE001
            print(f"Error: {exc}", file=sys.stderr)
            client_ok = False

    if not client_ok:
        print(f"Error: one or both client Jobs failed for {case_name}", file=sys.stderr)
        return CaseResult.FAILED

    print(f"=== [{case_name}] Rendering comparison ===")
    ip = json.loads((out_dir / "inference_perf.json").read_text())
    vb = json.loads((out_dir / "vllm_bench.json").read_text())
    table = render_comparison.render(render_comparison.build_rows(ip, vb))
    sys.stdout.write(table)
    (out_dir / "comparison.txt").write_text(table)

    print(f"Results for {case_name} written to {out_dir}/")
    return CaseResult.PASSED


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def find_template_manifest() -> Path:
    """Locate deploy/manifests.yaml relative to this script."""
    for candidate in (
        SCRIPT_DIR / "../../../../deploy/manifests.yaml",
        SCRIPT_DIR / "../../../deploy/manifests.yaml",
    ):
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError("deploy/manifests.yaml template not found")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every case under cases/ (or just one with --case). For each "
            "case, bring up a fresh vLLM in two sibling namespaces and run "
            "inference-perf in one and `vllm bench serve` in the other against "
            "the same workload config."
        )
    )
    parser.add_argument(
        "-n",
        "--namespace-prefix",
        default=DEFAULT_NAMESPACE_PREFIX,
        help=f"Namespace prefix (default: {DEFAULT_NAMESPACE_PREFIX})",
    )
    parser.add_argument(
        "-i",
        "--ip-image",
        default=DEFAULT_IP_IMAGE,
        help=f"inference-perf image (default: {DEFAULT_IP_IMAGE})",
    )
    parser.add_argument(
        "-t",
        "--rollout-timeout",
        type=parse_duration,
        default=parse_duration(DEFAULT_ROLLOUT_TIMEOUT),
        help=f"vLLM rollout timeout per namespace (default: {DEFAULT_ROLLOUT_TIMEOUT})",
    )
    parser.add_argument(
        "-j",
        "--job-timeout",
        type=parse_duration,
        default=parse_duration(DEFAULT_JOB_TIMEOUT),
        help=f"Client Job timeout (default: {DEFAULT_JOB_TIMEOUT})",
    )
    parser.add_argument(
        "-k",
        "--case",
        dest="only_case",
        default="",
        help="Only run a single case by name",
    )
    parser.add_argument(
        "-c",
        "--cleanup",
        action="store_true",
        help="Delete namespaces after the run completes",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if not CASES_DIR.is_dir():
        print(f"Error: cases directory not found at {CASES_DIR}", file=sys.stderr)
        return 1

    template_manifest = find_template_manifest()

    load_kube_config()
    cl = make_cluster()

    print("=== Checking for leftover namespaces from previous runs ===")
    cleanup_leftover_namespaces(cl, args.namespace_prefix)

    passed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    run_namespaces: list[str] = []

    for case_dir in sorted(p for p in CASES_DIR.iterdir() if p.is_dir()):
        case_name = case_dir.name
        if args.only_case and case_name != args.only_case:
            continue

        ns_suffix = case_name.replace("_", "-")
        ns_ip = f"{args.namespace_prefix}-{ns_suffix}-ip"
        ns_vb = f"{args.namespace_prefix}-{ns_suffix}-vb"
        run_namespaces += [ns_ip, ns_vb]

        print()
        print("########################################")
        print(f"### Case: {case_name}")
        print(f"###   inference-perf ns: {ns_ip}")
        print(f"###   vllm bench    ns: {ns_vb}")
        print("########################################")

        rc = run_case(
            case_dir,
            case_name,
            ns_ip,
            ns_vb,
            template_manifest,
            args.ip_image,
            args.rollout_timeout,
            args.job_timeout,
        )
        if rc == CaseResult.PASSED:
            passed.append(case_name)
        elif rc == CaseResult.SKIPPED:
            skipped.append(case_name)
        else:
            failed.append(case_name)

    print()
    print("########################################")
    print("### Summary")
    print("########################################")
    print(f"Passed:  {len(passed)} {' '.join(passed)}")
    print(f"Failed:  {len(failed)} {' '.join(failed)}")
    print(f"Skipped: {len(skipped)} {' '.join(skipped)}")

    if args.cleanup and run_namespaces:
        print()
        print("=== Cleaning up namespaces ===")
        for ns in run_namespaces:
            delete_namespace(cl, ns, wait=False)

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
