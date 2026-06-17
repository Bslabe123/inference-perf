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
"""Pure unit tests for cross-tool result parsing and the side-by-side table.

Like test_requirements.py these are NOT marked ``live``: parsing a tool's logs
into the common Metrics shape is pure and runs in default CI with no cluster.
The matching half (does the tool's real output match these field names?) is what
a live run verifies; this pins the normalization (units, field mapping, graceful
None on a missing block) so a refactor cannot silently change it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "optional"))

from harness import comparison, parsers  # noqa: E402

_IP_SUMMARY = """
=== START_SUMMARY ===
{"benchmark_time_seconds": 30.0,
 "successes": {"count": 28,
   "throughput": {"requests_per_sec": 0.93, "output_tokens_per_sec": 70.1, "total_tokens_per_sec": 600.0},
   "latency": {"time_to_first_token": {"mean": 0.12, "median": 0.11, "p90": 0.2, "p99": 0.3},
               "time_per_output_token": {"mean": 0.02},
               "inter_token_latency": {"mean": 0.019},
               "request_latency": {"mean": 1.5, "median": 1.4, "p99": 2.2}}},
 "failures": {"count": 0}}
=== END_SUMMARY ===
"""

_VLLM_RESULT = """
some vllm chatter
=== RESULT_START ===
{"completed": 30, "request_throughput": 0.95, "output_throughput": 71.0, "total_token_throughput": 610.0,
 "mean_ttft_ms": 130.0, "median_ttft_ms": 120.0, "p99_ttft_ms": 310.0,
 "mean_tpot_ms": 21.0, "median_tpot_ms": 20.0, "p99_tpot_ms": 35.0,
 "mean_itl_ms": 20.0, "median_itl_ms": 19.0, "p99_itl_ms": 33.0,
 "mean_e2el_ms": 1520.0, "median_e2el_ms": 1410.0, "p99_e2el_ms": 2250.0}
=== RESULT_END ===
"""


def test_inference_perf_latency_normalized_seconds_to_ms() -> None:
    m = parsers.parse("inference-perf", _IP_SUMMARY)
    assert m is not None
    assert m.completed == 28 and m.failed == 0
    assert m.request_throughput == 0.93
    # 0.12s -> 120ms, 2.2s -> 2200ms.
    assert m.ttft_ms == {"mean": 120.0, "median": 110.0, "p90": 200.0, "p99": 300.0}
    assert m.e2e_latency_ms["p99"] == 2200.0


def test_vllm_result_parsed_in_ms_as_is() -> None:
    m = parsers.parse("vllm-benchmark-serving", _VLLM_RESULT)
    assert m is not None
    assert m.completed == 30
    assert m.output_token_throughput == 71.0
    assert m.ttft_ms == {"mean": 130.0, "median": 120.0, "p99": 310.0}


def test_missing_block_returns_none_not_raises() -> None:
    assert parsers.parse("inference-perf", "no markers here") is None
    assert parsers.parse("vllm-benchmark-serving", "no markers here") is None


def test_malformed_json_returns_none() -> None:
    bad = "=== RESULT_START ===\n{not json}\n=== RESULT_END ==="
    assert parsers.parse("vllm-benchmark-serving", bad) is None


def test_unregistered_tool_returns_none() -> None:
    assert parsers.parse("guidellm", _VLLM_RESULT) is None


def test_render_table_keeps_a_column_for_unparsed_tools() -> None:
    results = {
        "inference-perf": parsers.parse("inference-perf", _IP_SUMMARY),
        "vllm-benchmark-serving": parsers.parse("vllm-benchmark-serving", _VLLM_RESULT),
        "guidellm": None,
    }
    table = comparison.render_table(results)
    # Every tool is a column even when its metrics are None...
    assert "| metric | inference-perf | vllm-benchmark-serving | guidellm |" in table
    # ...and a parsed mean lands in the body (0.12s -> 120.00ms).
    assert "TTFT mean (ms) | 120.00 | 130.00 |  |" in table


def test_build_report_preserves_null_for_unparsed() -> None:
    report = comparison._build_report({"guidellm": None})
    assert report == {"guidellm": None}
