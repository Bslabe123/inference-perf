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
"""Reporter behavior: what surfaces in non-TTY logs.

These tests pin the LogReporter contract: a start line, a finish line, and
graceful no-ops for counters not on the profile (so a caller using a newer
profile against an older reporter can't crash the run).
"""

import logging

import pytest

from inference_perf.observability import (
    LOCAL_INDEX_PREP,
    LogReporter,
    NullReporter,
    REMOTE_DOWNLOAD_PREP,
)


def test_null_reporter_is_noop(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    r = NullReporter()
    t = r.task(LOCAL_INDEX_PREP, "indexing")
    t.advance("files_indexed", 5.0)
    t.set_total("files_indexed", 100.0)
    t.finish()
    assert caplog.records == []


def test_log_reporter_emits_start_and_finish(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    r = LogReporter()
    t = r.task(LOCAL_INDEX_PREP, "indexing")
    t.finish("12 rows")
    messages = [rec.getMessage() for rec in caplog.records]
    # Start line names the profile, finish line includes the trailing message.
    assert any("indexing" in m and "start" in m for m in messages)
    assert any("indexing" in m and "12 rows" in m and "done" in m for m in messages)


def test_log_reporter_unknown_counter_is_silent_noop(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    r = LogReporter()
    t = r.task(LOCAL_INDEX_PREP, "indexing")
    t.advance("not_a_counter", 5.0)  # silently ignored
    t.set_total("not_a_counter", 100.0)
    t.finish()
    # No crash; start + finish are the only mandatory lines.
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("start" in m for m in messages)
    assert any("done" in m for m in messages)


def test_log_reporter_finish_includes_progress_summary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    r = LogReporter()
    t = r.task(REMOTE_DOWNLOAD_PREP, "downloading")
    t.set_total("bytes_total", 100_000_000.0)
    t.advance("bytes_downloaded", 50_000_000.0)
    t.advance("files_completed", 1.0)
    t.finish()
    finish_line = next(rec.getMessage() for rec in caplog.records if "done" in rec.getMessage())
    # Bytes are formatted human-readable, not raw.
    assert "MB" in finish_line or "GB" in finish_line
