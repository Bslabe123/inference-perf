#!/usr/bin/env python3
"""Render a side-by-side comparison of inference-perf vs `vllm bench serve`.

Both raw JSONs stay on disk untouched so future re-comparisons (different
metric mappings, different precision, etc.) can re-run without re-benchmarking.
This script is intentionally read-only on its inputs.

inference-perf reports latencies in seconds and throughput as raw per-second
counts. vllm bench reports latencies in milliseconds. We normalize everything
to ms / per-second here so the two columns are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Row:
    label: str
    ip_value: Optional[float]
    vb_value: Optional[float]
    unit: str

    def format_value(self, v: Optional[float]) -> str:
        if v is None:
            return "n/a"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:.3f}"

    def delta_pct(self) -> Optional[float]:
        if self.ip_value is None or self.vb_value is None or self.vb_value == 0:
            return None
        return (self.ip_value - self.vb_value) / self.vb_value * 100.0


def _get(d: dict, *path: str) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _to_ms(seconds: Optional[float]) -> Optional[float]:
    return None if seconds is None else seconds * 1000.0


def build_rows(ip: dict, vb: dict) -> list[Row]:
    """Map equivalent fields from each tool's report into a shared row layout."""
    rows: list[Row] = []

    # Throughput.
    rows.append(Row(
        "Request throughput (req/s)",
        _get(ip, "successes", "throughput", "requests_per_sec"),
        vb.get("request_throughput"),
        "req/s",
    ))
    rows.append(Row(
        "Output throughput (tok/s)",
        _get(ip, "successes", "throughput", "output_tokens_per_sec"),
        vb.get("output_throughput"),
        "tok/s",
    ))
    rows.append(Row(
        "Total throughput (tok/s)",
        _get(ip, "successes", "throughput", "total_tokens_per_sec"),
        vb.get("total_token_throughput"),
        "tok/s",
    ))

    # Latencies. inference-perf is in seconds, vllm bench in ms.
    def lat_row(
        label: str,
        ip_path: tuple[str, ...],
        vb_key: str,
        unit: str = "ms",
    ) -> Row:
        ip_seconds = _get(ip, *ip_path)
        ip_ms = _to_ms(ip_seconds if isinstance(ip_seconds, (int, float)) else None)
        vb_ms = vb.get(vb_key)
        if not isinstance(vb_ms, (int, float)):
            vb_ms = None
        return Row(label, ip_ms, vb_ms, unit)

    # TTFT.
    rows.append(lat_row(
        "TTFT mean (ms)",
        ("successes", "latency", "time_to_first_token", "mean"),
        "mean_ttft_ms",
    ))
    rows.append(lat_row(
        "TTFT p50 (ms)",
        ("successes", "latency", "time_to_first_token", "median"),
        "median_ttft_ms",
    ))
    rows.append(lat_row(
        "TTFT p90 (ms)",
        ("successes", "latency", "time_to_first_token", "p90"),
        "p90_ttft_ms",
    ))
    rows.append(lat_row(
        "TTFT p99 (ms)",
        ("successes", "latency", "time_to_first_token", "p99"),
        "p99_ttft_ms",
    ))

    # TPOT.
    rows.append(lat_row(
        "TPOT mean (ms)",
        ("successes", "latency", "time_per_output_token", "mean"),
        "mean_tpot_ms",
    ))
    rows.append(lat_row(
        "TPOT p50 (ms)",
        ("successes", "latency", "time_per_output_token", "median"),
        "median_tpot_ms",
    ))
    rows.append(lat_row(
        "TPOT p99 (ms)",
        ("successes", "latency", "time_per_output_token", "p99"),
        "p99_tpot_ms",
    ))

    # ITL.
    rows.append(lat_row(
        "ITL mean (ms)",
        ("successes", "latency", "inter_token_latency", "mean"),
        "mean_itl_ms",
    ))
    rows.append(lat_row(
        "ITL p99 (ms)",
        ("successes", "latency", "inter_token_latency", "p99"),
        "p99_itl_ms",
    ))

    # End-to-end latency.
    rows.append(lat_row(
        "E2E latency mean (ms)",
        ("successes", "latency", "request_latency", "mean"),
        "mean_e2el_ms",
    ))
    rows.append(lat_row(
        "E2E latency p50 (ms)",
        ("successes", "latency", "request_latency", "median"),
        "median_e2el_ms",
    ))
    rows.append(lat_row(
        "E2E latency p99 (ms)",
        ("successes", "latency", "request_latency", "p99"),
        "p99_e2el_ms",
    ))

    # Counts and run duration.
    rows.append(Row(
        "Completed requests",
        _get(ip, "successes", "count"),
        vb.get("completed"),
        "",
    ))
    rows.append(Row(
        "Failed requests",
        _get(ip, "failures", "count"),
        None,
        "",
    ))
    rows.append(Row(
        "Benchmark duration (s)",
        ip.get("benchmark_time_seconds"),
        vb.get("duration"),
        "s",
    ))

    return rows


def render(rows: list[Row]) -> str:
    headers = ("Metric", "inference-perf", "vllm bench", "delta vs vb")
    widths = [
        max(len(headers[0]), max(len(r.label) for r in rows)),
        max(len(headers[1]), max(len(r.format_value(r.ip_value)) for r in rows)),
        max(len(headers[2]), max(len(r.format_value(r.vb_value)) for r in rows)),
        len(headers[3]),
    ]
    widths[3] = max(widths[3], 12)

    def fmt_row(cells: tuple[str, str, str, str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), fmt_row(tuple("-" * w for w in widths))]
    for r in rows:
        delta = r.delta_pct()
        delta_str = "n/a" if delta is None else f"{delta:+.2f}%"
        lines.append(fmt_row((
            r.label,
            r.format_value(r.ip_value),
            r.format_value(r.vb_value),
            delta_str,
        )))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-perf", type=Path, required=True)
    parser.add_argument("--vllm-bench", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional path to write the rendered table to.")
    args = parser.parse_args()

    ip = json.loads(args.inference_perf.read_text())
    vb = json.loads(args.vllm_bench.read_text())

    rows = build_rows(ip, vb)
    table = render(rows)

    sys.stdout.write(table)
    if args.output is not None:
        args.output.write_text(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
