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
from .schema import Severity

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


class Thresholds(BaseModel):
    """Every rule-check threshold, with defaults. Override any of them in config.toml."""

    # design.loop
    loop_min_repeats: int = 3
    loop_near_duplicate_similarity: float = 0.9
    loop_max_cycle_length: int = 4
    # design.no_stop
    step_limit: int = 25
    """LangGraph's default recursion limit."""
    # tools.unused_output
    unused_output_max_overlap: float = 0.2
    unused_output_min_tokens: int = 4
    # cost.token_hotspot
    token_hotspot_share: float = 0.6
    """Also must be at least 1.5x an even split between the run's nodes."""
    token_hotspot_min_tokens: int = 5_000
    context_growth_factor: float = 4.0
    context_growth_min_steps: int = 4
    context_growth_min_tokens: int = 2_000
    """Ignore growth unless the last call's input reaches this many tokens."""
    # cost.latency_outlier
    latency_p95_multiplier: float = 2.0
    latency_absolute_s: float = 60.0
    latency_min_s: float = 1.0
    """Relative outliers shorter than this are ignored as noise."""
    latency_min_samples: int = 5
    # signals
    efficient_path_ratio: float = 0.5
    efficient_path_min_runs: int = 5
    clean_handoff_min_overlap: float = 0.5


class ChecksConfig(BaseModel):
    disabled: list[str] = Field(default_factory=list)
    severity: dict[str, Severity] = Field(default_factory=dict)
    """Per-check severity overrides, e.g. {"cost.latency_outlier": "medium"}."""


class Config(BaseModel):
    redact: RedactConfig = Field(default_factory=RedactConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    preview_chars: int = 500
    thresholds: Thresholds = Field(default_factory=Thresholds)
    checks: ChecksConfig = Field(default_factory=ChecksConfig)


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
