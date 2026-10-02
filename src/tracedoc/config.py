"""Configuration: defaults, loading from TOML, and where config lives.

Precedence (later wins): built-in defaults, the user config file
(`<user config dir>/config.toml`), then an explicit `--config` path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from platformdirs import user_config_dir
from pydantic import BaseModel, Field

from .__about__ import NAME

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

DEFAULT_REDACT_FIELDS = [
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
]

DEFAULT_REDACT_PATTERNS = [
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email addresses
    r"\bsk-[A-Za-z0-9_-]{16,}",  # OpenAI / Anthropic style secret keys
    r"\b(?:ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{20,}",  # GitHub tokens
    r"\bAKIA[0-9A-Z]{16}\b",  # AWS access key IDs
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]+=*",  # bearer tokens
    r"\b(?:\d[ -]?){13,16}\b",  # card-like digit runs
]


class RedactConfig(BaseModel):
    fields: list[str] = Field(default_factory=lambda: list(DEFAULT_REDACT_FIELDS))
    """Keys whose values are replaced wholesale, matched case-insensitively."""
    patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_REDACT_PATTERNS))
    """Regexes replaced inside any string value."""
    replacement: str = "[REDACTED]"


class CaptureConfig(BaseModel):
    max_chars: int = 20_000
    """Longest string kept per value in captured traces; longer strings are truncated."""


class Config(BaseModel):
    redact: RedactConfig = Field(default_factory=RedactConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    preview_chars: int = 500
    thresholds: dict[str, Any] = Field(default_factory=dict)


def config_dir() -> Path:
    return Path(user_config_dir(NAME))


def user_config_path() -> Path:
    return config_dir() / "config.toml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None, *, use_user_config: bool = True) -> Config:
    data: dict[str, Any] = {}
    sources = [user_config_path()] if use_user_config else []
    if path is not None:
        sources.append(path)
    for source in sources:
        if source.is_file():
            with source.open("rb") as f:
                data = _deep_merge(data, tomllib.load(f))
    return Config.model_validate(data)
