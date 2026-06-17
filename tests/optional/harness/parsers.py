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
"""Normalize each benchmark tool's output into one comparable Metrics shape.

Every cross-tool case runs N tools against the *same* server, but each tool
reports in its own format and units. A parser's job is to pull a tool's logs
into the common ``Metrics`` below so the side-by-side report compares like with
like. Normalized units are fixed here once: **latencies in milliseconds,
throughput per second**. inference-perf reports latency in seconds, so its
parser multiplies; the vLLM script already reports milliseconds.

The contract with a tool's Job (see comparison.py): everything a parser needs
is printed to stdout between two markers it controls. inference-perf already
emits ``=== START_SUMMARY ===`` / ``=== END_SUMMARY ===``; every other tool's
Job wraps its machine-readable result in ``=== RESULT_START ===`` /
``=== RESULT_END ===`` so the parser never has to scrape the tool's own chatter.

Parsing is best-effort by design: a parser returns ``None`` if the expected
block is missing or malformed rather than raising, because the side-by-side
tier is meant to *show* whatever each tool produced (including nothing) so a
human can decide tolerances later. The runner always persists raw logs
regardless, so a None here loses nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

# A latency distribution, normalized to milliseconds. Keys are filled best-effort;
# a tool that does not report a given percentile simply omits it.
Stat = dict[str, float]

RESULT_START = "=== RESULT_START ==="
RESULT_END = "=== RESULT_END ==="


@dataclass
class Metrics:
    """One tool's results, normalized for side-by-side comparison.

    Latencies are milliseconds; throughput is per second. Any field a tool does
    not report stays None (scalars) or empty (Stat dicts) so the report can show
    a blank cell rather than a fabricated zero.
    """

    tool: str
    completed: int | None = None
    failed: int | None = None
    request_throughput: float | None = None  # requests/sec
    output_token_throughput: float | None = None  # output tokens/sec
    total_token_throughput: float | None = None  # (input+output) tokens/sec
    ttft_ms: Stat = field(default_factory=dict)  # time to first token
    tpot_ms: Stat = field(default_factory=dict)  # time per output token
    itl_ms: Stat = field(default_factory=dict)  # inter-token latency
    e2e_latency_ms: Stat = field(default_factory=dict)  # end-to-end request latency


def extract_block(text: str, start: str, end: str) -> str:
    """Return the lines strictly between the first ``start`` and the next ``end``.

    Empty string if either marker is absent. Shared with the single-tool runner,
    which uses inference-perf's summary markers.
    """
    lines = text.splitlines()
    try:
        i = lines.index(start)
        j = lines.index(end, i + 1)
    except ValueError:
        return ""
    return "\n".join(lines[i + 1 : j])


def _result_json(logs: str) -> dict[str, Any] | None:
    """Parse the JSON a tool printed between the RESULT markers, or None."""
    block = extract_block(logs, RESULT_START, RESULT_END).strip()
    if not block:
        return None
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _stat_seconds_to_ms(raw: Any) -> Stat:
    """inference-perf latency block {mean, median, p90, p99} in seconds -> ms."""
    if not isinstance(raw, dict):
        return {}
    out: Stat = {}
    for key in ("mean", "median", "p90", "p99"):
        val = raw.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val) * 1000.0
    return out


def _stat_ms(raw: dict[str, Any], *, mean: str, median: str, p99: str) -> Stat:
    """Pull named ms fields out of a flat dict (vLLM-style result JSON)."""
    out: Stat = {}
    for label, src in (("mean", mean), ("median", median), ("p99", p99)):
        val = raw.get(src)
        if isinstance(val, (int, float)):
            out[label] = float(val)
    return out


def parse_inference_perf(logs: str) -> Metrics | None:
    """inference-perf request-lifecycle summary (START_SUMMARY/END_SUMMARY).

    Schema mirrors ResponsesSummary in inference_perf/reportgen/base.py: latency
    under successes.latency (seconds), throughput under successes.throughput.
    """
    block = extract_block(logs, "=== START_SUMMARY ===", "=== END_SUMMARY ===").strip()
    if not block:
        return None
    try:
        summary = json.loads(block)
    except json.JSONDecodeError:
        return None
    successes = summary.get("successes", {}) if isinstance(summary, dict) else {}
    latency = successes.get("latency", {}) if isinstance(successes, dict) else {}
    throughput = successes.get("throughput", {}) if isinstance(successes, dict) else {}
    failures = summary.get("failures", {}) if isinstance(summary, dict) else {}
    return Metrics(
        tool="inference-perf",
        completed=successes.get("count"),
        failed=failures.get("count"),
        request_throughput=throughput.get("requests_per_sec"),
        output_token_throughput=throughput.get("output_tokens_per_sec"),
        total_token_throughput=throughput.get("total_tokens_per_sec"),
        ttft_ms=_stat_seconds_to_ms(latency.get("time_to_first_token")),
        tpot_ms=_stat_seconds_to_ms(latency.get("time_per_output_token")),
        itl_ms=_stat_seconds_to_ms(latency.get("inter_token_latency")),
        e2e_latency_ms=_stat_seconds_to_ms(latency.get("request_latency")),
    )


def parse_vllm_benchmark_serving(logs: str) -> Metrics | None:
    """vLLM benchmark_serving.py --save-result JSON (already in ms / per-sec).

    Field names are vLLM's documented result keys: request_throughput,
    output_throughput, total_token_throughput, and {mean,median,p99}_{ttft,tpot,
    itl,e2el}_ms. The Job cats the saved JSON between the RESULT markers.
    """
    data = _result_json(logs)
    if data is None:
        return None
    return Metrics(
        tool="vllm-benchmark-serving",
        completed=data.get("completed"),
        failed=None,  # the script reports completed only; failures surface as a short count
        request_throughput=data.get("request_throughput"),
        output_token_throughput=data.get("output_throughput"),
        total_token_throughput=data.get("total_token_throughput"),
        ttft_ms=_stat_ms(data, mean="mean_ttft_ms", median="median_ttft_ms", p99="p99_ttft_ms"),
        tpot_ms=_stat_ms(data, mean="mean_tpot_ms", median="median_tpot_ms", p99="p99_tpot_ms"),
        itl_ms=_stat_ms(data, mean="mean_itl_ms", median="median_itl_ms", p99="p99_itl_ms"),
        e2e_latency_ms=_stat_ms(data, mean="mean_e2el_ms", median="median_e2el_ms", p99="p99_e2el_ms"),
    )


# tool name (matches the tools/<name>.yaml stem) -> parser. Adding a tool is:
# (1) drop tools/<name>.yaml that prints its result between the RESULT markers,
# (2) register a parser here. No other harness change.
PARSERS: dict[str, Callable[[str], Metrics | None]] = {
    "inference-perf": parse_inference_perf,
    "vllm-benchmark-serving": parse_vllm_benchmark_serving,
}


def parse(tool: str, logs: str) -> Metrics | None:
    """Dispatch to the registered parser for ``tool``; None if unregistered."""
    parser = PARSERS.get(tool)
    return parser(logs) if parser else None
