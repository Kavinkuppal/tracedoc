"""Config-driven redaction, applied to trace content before anything else sees it.

Two passes: `learn` walks a trace and records every string held under a
sensitive key, then `__call__` replaces those keys' values *and* scrubs the
learned values wherever else they appear (a tool echoing a secret argument
into its output, a model repeating it). Values shorter than
`MIN_LEARNED_LENGTH` are only redacted under their key, to avoid scrubbing
common short strings everywhere.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .config import RedactConfig

MIN_LEARNED_LENGTH = 4


class Redactor:
    def __init__(self, config: RedactConfig) -> None:
        self.fields = [f.lower() for f in config.fields]
        self.patterns = [re.compile(p) for p in config.patterns]
        self.replacement = config.replacement
        self._learned: set[str] = set()
        self._learned_pattern: re.Pattern[str] | None = None

    def _is_sensitive_key(self, key: str) -> bool:
        key = key.lower()
        return any(key == f or key.endswith(("_" + f, "-" + f)) for f in self.fields)

    def learn(self, value: Any, *, sensitive: bool = False) -> None:
        """Record string values that sit under sensitive keys anywhere in `value`."""
        if isinstance(value, str):
            if sensitive and len(value) >= MIN_LEARNED_LENGTH:
                self._learned.add(value)
                self._learned_pattern = None
        elif isinstance(value, Mapping):
            for k, v in value.items():
                self.learn(v, sensitive=sensitive or self._is_sensitive_key(str(k)))
        elif isinstance(value, (list, tuple)):
            for v in value:
                self.learn(v, sensitive=sensitive)

    def text(self, value: str) -> str:
        if self._learned:
            if self._learned_pattern is None:
                ordered = sorted(self._learned, key=len, reverse=True)
                self._learned_pattern = re.compile("|".join(map(re.escape, ordered)))
            value = self._learned_pattern.sub(self.replacement, value)
        for pattern in self.patterns:
            value = pattern.sub(self.replacement, value)
        return value

    def __call__(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return {
                k: self.replacement if self._is_sensitive_key(str(k)) else self(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self(v) for v in value]
        return value
