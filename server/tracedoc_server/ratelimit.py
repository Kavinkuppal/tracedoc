"""Basic in-memory rate limiting, keyed by install ID, never by IP address."""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """At most `per_install` requests per install ID and `overall` requests in total,
    each within a sliding `window_s`."""

    def __init__(
        self, per_install: int = 60, overall: int = 6_000, window_s: float = 3_600
    ) -> None:
        self.per_install = per_install
        self.overall = overall
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._all: deque[float] = deque()
        self._lock = threading.Lock()

    def _trim(self, q: deque[float], now: float) -> None:
        while q and now - q[0] > self.window_s:
            q.popleft()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._hits.setdefault(key, deque())
            self._trim(q, now)
            self._trim(self._all, now)
            if len(q) >= self.per_install or len(self._all) >= self.overall:
                return False
            q.append(now)
            self._all.append(now)
            return True
