"""A file cache of model answers, so the same traces always give the same report.

Every answer is stored under a hash of everything that shaped it: the system prompt and the
prompt (which carry the trace window, and the check's instructions with their version), the
output schema, the model, and its settings (effort, temperature, token limit). Running again on
the same traces reuses the stored answers instead of asking the model again, so a report never
changes because a model answered differently the second time.

One JSON file per answer, under <cache dir>/<first two hash characters>/<hash>.json. Files are
written atomically, so an interrupted run never leaves a half-written answer. An unreadable or
outdated file counts as a miss, and the model is asked again.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from platformdirs import user_cache_dir
from pydantic import BaseModel

from ..__about__ import NAME
from .client import LLMClient, LLMResult

T = TypeVar("T", bound=BaseModel)

CACHE_FORMAT = 1
"""Bump when the meaning of a stored answer changes; older entries are then ignored."""


def default_cache_dir() -> Path:
    return Path(user_cache_dir(NAME, appauthor=False)) / "judge"


def cache_key(
    system: str,
    prompt: str,
    schema: type[BaseModel],
    *,
    model: str,
    max_tokens: int,
    effort: str | None,
    temperature: float | None,
) -> str:
    request = {
        "format": CACHE_FORMAT,
        "system": system,
        "prompt": prompt,
        "schema": schema.__name__,
        "schema_json": schema.model_json_schema(),
        "model": model,
        "max_tokens": max_tokens,
        "effort": effort,
        "temperature": temperature,
    }
    blob = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class CachingClient:
    """Wraps a client: answers from the cache when it can, and stores every new answer."""

    def __init__(self, client: LLMClient, directory: Path) -> None:
        self.client = client
        self.directory = directory
        self.stats = CacheStats()
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def lookup(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        model: str,
        max_tokens: int = 2_000,
        effort: str | None = None,
        temperature: float | None = None,
    ) -> LLMResult[T] | None:
        key = cache_key(
            system,
            prompt,
            schema,
            model=model,
            max_tokens=max_tokens,
            effort=effort,
            temperature=temperature,
        )
        try:
            record: dict[str, Any] = json.loads(self._path(key).read_text("utf-8"))
            value = schema.model_validate(record["value"])
            result = LLMResult(
                value=value,
                model=str(record["model"]),
                tokens_in=int(record["tokens_in"]),
                tokens_out=int(record["tokens_out"]),
                cached=True,
            )
        except (OSError, ValueError, KeyError, TypeError):  # missing, unreadable or outdated
            with self._lock:
                self.stats.misses += 1
            return None
        with self._lock:
            self.stats.hits += 1
        return result

    def complete_json(
        self,
        system: str,
        prompt: str,
        schema: type[T],
        *,
        model: str,
        max_tokens: int = 2_000,
        effort: str | None = None,
        temperature: float | None = None,
    ) -> LLMResult[T]:
        settings: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "effort": effort,
            "temperature": temperature,
        }
        hit = self.lookup(system, prompt, schema, **settings)
        if hit is not None:
            return hit
        result = self.client.complete_json(system, prompt, schema, **settings)
        self._store(cache_key(system, prompt, schema, **settings), result)
        return result

    def _store(self, key: str, result: LLMResult[Any]) -> None:
        record = {
            "format": CACHE_FORMAT,
            "model": result.model,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "value": result.value.model_dump(mode="json"),
        }
        path = self._path(key)
        tmp: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record, f, sort_keys=True, ensure_ascii=False)
            os.replace(tmp, path)
            tmp = None
        except OSError:
            pass  # a cache that can't be written only costs a repeat call next time
        finally:
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)
