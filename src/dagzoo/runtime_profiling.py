"""Internal runtime profiling helpers for maintainer benchmarks."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(slots=True)
class RuntimeProfileCollector:
    """Collect floating-point counters for one profiled runtime scope."""

    _metrics: dict[str, float] = field(default_factory=dict)

    def accumulate(self, key: str, value: float) -> None:
        self._metrics[str(key)] = float(self._metrics.get(str(key), 0.0)) + float(value)

    def snapshot(self) -> dict[str, float]:
        return dict(self._metrics)


_CURRENT_RUNTIME_PROFILE: ContextVar[RuntimeProfileCollector | None] = ContextVar(
    "dagzoo_runtime_profile",
    default=None,
)


def current_runtime_profile() -> RuntimeProfileCollector | None:
    """Return the active runtime profiler, if one is enabled."""

    return _CURRENT_RUNTIME_PROFILE.get()


def record_runtime_profile_metric(key: str, value: float) -> None:
    """Accumulate one metric on the active runtime profile."""

    collector = current_runtime_profile()
    if collector is None:
        return
    collector.accumulate(str(key), float(value))


@contextmanager
def runtime_profile_scope(*, enabled: bool) -> Iterator[RuntimeProfileCollector | None]:
    """Install a runtime profiler for the lifetime of one benchmark scope."""

    if not enabled:
        yield None
        return
    collector = RuntimeProfileCollector()
    token = _CURRENT_RUNTIME_PROFILE.set(collector)
    try:
        yield collector
    finally:
        _CURRENT_RUNTIME_PROFILE.reset(token)
