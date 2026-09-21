"""Thread-compatible monotonic event-rate and latency statistics."""

from collections import deque

import numpy as np


class EventRate:
    def __init__(self, window_s=2.0):
        self.window_s = float(window_s)
        if not np.isfinite(self.window_s) or self.window_s <= 0.0:
            raise ValueError("rate window must be positive")
        self._events = deque()

    def mark(self, timestamp):
        value = float(timestamp)
        if not np.isfinite(value):
            raise ValueError("event timestamp must be finite")
        self._events.append(value)
        self._trim(value)

    def _trim(self, now):
        cutoff = float(now) - self.window_s
        while len(self._events) > 1 and self._events[0] < cutoff:
            self._events.popleft()

    def rate(self, now, stale_after_s=None):
        value = float(now)
        self._trim(value)
        if not self._events:
            return 0.0
        if stale_after_s is not None and value - self._events[-1] > float(stale_after_s):
            return 0.0
        if len(self._events) < 2:
            return 0.0
        duration = self._events[-1] - self._events[0]
        return 0.0 if duration <= 0.0 else (len(self._events) - 1) / duration


def distribution(values):
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": int(len(data)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
    }
