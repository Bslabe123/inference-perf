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
"""Progress reporters consumed by ``BaseGenerator.prepare``.

Three implementations:

- :class:`NullReporter` for tests and headless workers; emits nothing.
- :class:`LogReporter` for non-TTY environments (cluster job logs); emits
  structured INFO lines on start, periodic increments, and finish.
- :class:`RichReporter` for TTY; renders ``rich`` progress bars on top of the
  shared console set up in :mod:`inference_perf.logger`, then emits the same
  INFO lines on finish so saved transcripts still describe the run.

Reporters expose tasks: a datagen opens a task tied to a
:class:`ProgressProfile`, increments named counters as work happens, and
finishes the task. The reporter formats based on the profile (bytes/sec when
the profile is NETWORK-bound, items/sec when CPU-bound).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Protocol

from .progress_profile import BoundBy, ProgressProfile, Unit

logger = logging.getLogger(__name__)

_MIN_LOG_INTERVAL_S = 5.0


class Task(Protocol):
    """Handle to one in-flight progress surface.

    Counters are addressed by the names declared on the profile; updates to
    undeclared names are ignored so a caller using a newer profile against an
    older reporter doesn't crash.
    """

    def advance(self, counter: str, by: float = 1.0) -> None: ...

    def set_total(self, counter: str, total: float) -> None: ...

    def finish(self, message: Optional[str] = None) -> None: ...


class ProgressReporter(Protocol):
    """Reporter handed to ``BaseGenerator.prepare``."""

    def task(self, profile: ProgressProfile, description: str) -> Task: ...


# --------------------- Null ---------------------


class _NullTask:
    def advance(self, counter: str, by: float = 1.0) -> None:
        return

    def set_total(self, counter: str, total: float) -> None:
        return

    def finish(self, message: Optional[str] = None) -> None:
        return


class NullReporter:
    """Reporter that does nothing. Default for tests and worker processes."""

    def task(self, profile: ProgressProfile, description: str) -> Task:
        return _NullTask()


# --------------------- Log (non-TTY) ---------------------


class _LogTask:
    """Throttled INFO-line progress task for non-TTY logs."""

    def __init__(self, profile: ProgressProfile, description: str) -> None:
        self._profile = profile
        self._description = description
        self._counters: Dict[str, float] = {c.name: 0.0 for c in profile.counters}
        self._totals: Dict[str, Optional[float]] = {c.name: None for c in profile.counters}
        self._started_at = time.monotonic()
        self._last_log_at = 0.0
        logger.info("[%s] start (%s)", description, profile.name)

    def advance(self, counter: str, by: float = 1.0) -> None:
        if counter not in self._counters:
            return
        self._counters[counter] += by
        now = time.monotonic()
        if now - self._last_log_at >= _MIN_LOG_INTERVAL_S:
            self._last_log_at = now
            logger.info("[%s] %s", self._description, self._format_progress())

    def set_total(self, counter: str, total: float) -> None:
        if counter not in self._totals:
            return
        self._totals[counter] = total

    def finish(self, message: Optional[str] = None) -> None:
        elapsed = time.monotonic() - self._started_at
        tail = f", {message}" if message else ""
        logger.info("[%s] done in %.1fs%s (%s)", self._description, elapsed, tail, self._format_progress())

    def _format_progress(self) -> str:
        parts = []
        for c in self._profile.counters:
            value = self._counters[c.name]
            total = self._totals[c.name]
            parts.append(_format_counter(c.label, value, total, c.unit))
        elapsed = max(time.monotonic() - self._started_at, 1e-6)
        if self._profile.rate_counter and self._profile.rate_counter in self._counters:
            value = self._counters[self._profile.rate_counter]
            rate = value / elapsed
            unit = next(c.unit for c in self._profile.counters if c.name == self._profile.rate_counter)
            parts.append(_format_rate(rate, unit, self._profile.bound_by))
        return ", ".join(parts)


class LogReporter:
    """Reporter that emits INFO log lines on start, periodically, and finish."""

    def task(self, profile: ProgressProfile, description: str) -> Task:
        return _LogTask(profile, description)


# --------------------- Rich (TTY) ---------------------


class _RichTask:
    """Wraps a Rich Progress task plus an INFO finish line for transcripts."""

    def __init__(
        self,
        profile: ProgressProfile,
        description: str,
        progress: object,
        task_id: object,
    ) -> None:
        self._profile = profile
        self._description = description
        self._progress = progress
        self._task_id = task_id
        self._counters: Dict[str, float] = {c.name: 0.0 for c in profile.counters}
        self._totals: Dict[str, Optional[float]] = {c.name: None for c in profile.counters}
        self._started_at = time.monotonic()

    def advance(self, counter: str, by: float = 1.0) -> None:
        if counter not in self._counters:
            return
        self._counters[counter] += by
        if counter == self._profile.rate_counter:
            self._progress.update(self._task_id, advance=by)  # type: ignore[attr-defined]

    def set_total(self, counter: str, total: float) -> None:
        if counter not in self._totals:
            return
        self._totals[counter] = total
        if counter == self._profile.rate_counter:
            self._progress.update(self._task_id, total=total)  # type: ignore[attr-defined]

    def finish(self, message: Optional[str] = None) -> None:
        self._progress.update(self._task_id, completed=self._progress.tasks[self._task_id].total)  # type: ignore[attr-defined]
        elapsed = time.monotonic() - self._started_at
        tail = f", {message}" if message else ""
        logger.info("[%s] done in %.1fs%s", self._description, elapsed, tail)


class RichReporter:
    """Reporter that renders Rich progress bars on the shared console.

    Falls back to LogReporter behavior on finish so transcripts capture the
    completion line even if the terminal scrolled the bar off-screen.
    """

    def __init__(self, console: object) -> None:
        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,  # type: ignore[arg-type]
            transient=False,
        )
        self._progress.start()

    def task(self, profile: ProgressProfile, description: str) -> Task:
        task_id = self._progress.add_task(description, total=None)
        return _RichTask(profile, description, self._progress, task_id)

    def stop(self) -> None:
        self._progress.stop()


# --------------------- Factory ---------------------


def make_reporter() -> ProgressReporter:
    """Return the reporter appropriate for the current process.

    TTY → :class:`RichReporter` against the shared console.
    Non-TTY → :class:`LogReporter`.
    """
    from inference_perf.logger import get_console

    console = get_console()
    if console is None:
        return LogReporter()
    return RichReporter(console)


# --------------------- Formatting helpers ---------------------


def _format_counter(label: str, value: float, total: Optional[float], unit: Unit) -> str:
    value_str = _format_value(value, unit)
    if total is not None:
        total_str = _format_value(total, unit)
        return f"{label} {value_str}/{total_str}"
    return f"{label} {value_str}"


def _format_value(value: float, unit: Unit) -> str:
    if unit == Unit.BYTES:
        return _humanize_bytes(value)
    if unit == Unit.MS:
        return f"{value:.0f}ms"
    if unit == Unit.SECONDS:
        return f"{value:.1f}s"
    return f"{int(value)}"


def _format_rate(rate: float, unit: Unit, bound_by: BoundBy) -> str:
    if unit == Unit.BYTES or bound_by == BoundBy.NETWORK:
        return f"{_humanize_bytes(rate)}/s"
    return f"{rate:.1f}/s"


def _humanize_bytes(n: float) -> str:
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or suffix == "TB":
            return f"{n:.1f}{suffix}" if suffix != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n:.1f}TB"
