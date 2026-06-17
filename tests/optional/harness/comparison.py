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
"""Run several benchmark tools against one server and record results side by side.

A comparison case is a case directory (``vllm.yaml`` + a ``tools/`` subdir) where
each ``tools/<name>.yaml`` is a self-contained manifest that runs ``<name>`` as a
Job and prints its result for parsers.py to read. The server is deployed once;
the tools then run **sequentially** so each measures the server to itself (a
concurrent run would have the tools contend and pollute each other's latencies).

This is the side-by-side tier: it records what every tool reported and asserts
only that each tool ran and produced output. It deliberately does NOT assert the
tools agree, because the point is to look at a real run first and decide what a
fair tolerance is. Raw logs are always persisted per tool, so even a tool whose
parser does not yet match its output is fully recoverable from output/<name>.log.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from harness import parsers, requirements
from harness.parsers import Metrics
from harness.runner import DEFAULT_IMAGE, _TEMPLATE_IMAGE, deploy_server, run_job


def run_comparison(
    kubeconfig: str | None,
    namespace: str,
    case_dir: Path,
    image: str = DEFAULT_IMAGE,
) -> dict[str, Metrics | None]:
    """Deploy the server once, run each tool in tools/, and write a side-by-side report.

    Returns the per-tool parsed Metrics (None where a tool's parser found nothing)
    so a caller can assert on it. Writes output/<tool>.log (raw), output/report.json
    (machine-readable), and output/report.md (the comparison table) under case_dir.
    Raises AssertionError if a case has no tools or a tool produced empty logs; lets
    Job timeouts propagate from run_job.
    """
    case_dir = Path(case_dir)
    tool_manifests = sorted((case_dir / "tools").glob("*.yaml"))
    assert tool_manifests, f"comparison case {case_dir.name} has no tools/*.yaml"

    deploy_server(kubeconfig, namespace, case_dir)

    output_dir = case_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Metrics | None] = {}
    for manifest_path in tool_manifests:
        tool = manifest_path.stem
        # Substitute the inference-perf image placeholder so --image applies to the
        # inference-perf tool; other tools pin their own image and are untouched.
        manifest = manifest_path.read_text().replace(_TEMPLATE_IMAGE, image)
        job = requirements.job_names(manifest_path)
        assert job, f"tool manifest {manifest_path.name} declares no Job"

        logs = run_job(kubeconfig, namespace, manifest, job[0])
        (output_dir / f"{tool}.log").write_text(logs)
        assert logs.strip(), f"tool {tool} produced no logs"

        results[tool] = parsers.parse(tool, logs)

    report = _build_report(results)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2))
    table = render_table(results)
    (output_dir / "report.md").write_text(table)
    # Echo the table so it lands in pytest output / CI logs without opening a file.
    print(f"\nCross-tool comparison for {case_dir.name}:\n{table}")

    return results


def _build_report(results: dict[str, Metrics | None]) -> dict[str, object]:
    """JSON-friendly dump: every tool, with None preserved as null for unparsed."""
    return {tool: (asdict(m) if m is not None else None) for tool, m in results.items()}


# (column label, Metrics attribute, formatter). Scalars and the mean of each
# latency distribution; the full distributions live in report.json.
def _fmt_num(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _fmt_mean(stat: object) -> str:
    return _fmt_num(stat.get("mean") if isinstance(stat, dict) else None)


_ROWS: list[tuple[str, str, Callable[[object], str]]] = [
    ("completed", "completed", _fmt_num),
    ("failed", "failed", _fmt_num),
    ("req/s", "request_throughput", _fmt_num),
    ("out tok/s", "output_token_throughput", _fmt_num),
    ("total tok/s", "total_token_throughput", _fmt_num),
    ("TTFT mean (ms)", "ttft_ms", _fmt_mean),
    ("TPOT mean (ms)", "tpot_ms", _fmt_mean),
    ("ITL mean (ms)", "itl_ms", _fmt_mean),
    ("E2E mean (ms)", "e2e_latency_ms", _fmt_mean),
]


def render_table(results: dict[str, Metrics | None]) -> str:
    """A markdown table: one column per tool, one row per normalized metric.

    Unparsed tools (Metrics is None) get a column of blanks but still appear, so
    the report shows which tools ran without producing parseable metrics.
    """
    tools = list(results.keys())
    header = ["metric", *tools]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for label, attr, fmt in _ROWS:
        cells = [label]
        for tool in tools:
            metrics = results[tool]
            value = getattr(metrics, attr) if metrics is not None else None
            cells.append(fmt(value) if metrics is not None else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# Re-exported for callers/tests that want the dataclass field list without
# reaching into parsers; keeps the comparison surface in one module.
METRIC_FIELDS = [f.name for f in dataclasses.fields(Metrics)]
