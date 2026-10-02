"""Configuration: defaults, loading from TOML, and where config lives.

Precedence (later wins): built-in defaults, the user config file
(`<user config dir>/config.toml`), then an explicit `--config` path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from platformdirs import user_config_dir
from pydantic import BaseModel, ConfigDict, Field

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


class LLMConfig(BaseModel):
    """The developer's own model, used for judging and classification fallbacks.

    Keys are read only from environment variables, never from config files.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic", "openai", "openai_compatible"] = "anthropic"
    # TODO(founders): confirm default triage and confirmation models.
    triage_model: str = "claude-haiku-4-5"
    confirm_model: str = "claude-opus-5"
    confirm_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = "low"
    """Anthropic effort for the confirmation model; ignored by other providers."""
    base_url: str | None = None
    """Endpoint for openai_compatible providers (for example a local server)."""
    api_key_env: str | None = None
    """Name of the environment variable holding the key (never the key itself). Defaults to
    ANTHROPIC_API_KEY / OPENAI_API_KEY; openai_compatible sends no key unless this is set."""
    max_retries: int = 3
    timeout_s: float = 120.0


class JudgeConfig(BaseModel):
    window_steps: int = 3
    """Steps on each side of a candidate that the judge sees."""
    max_chars_per_field: int = 1_500
    sample_runs: int = 12
    """Runs sampled (in addition to rule flags and handoffs) for run-level judge checks."""
    max_handoffs_per_run: int = 4
    max_windows_per_signature: int = 5
    """Across the batch, judge at most this many windows per rule-candidate signature
    (check, node, key) and per handoff pair. The rest stay unconfirmed."""
    seed: int = 0
    concurrency: int = 4
    triage_min_confidence: float = 0.3
    confirm_min_confidence: float = 0.5
    min_precision: float = 0.8
    """Judge checks below this measured precision are disabled unless explicitly enabled."""
    enabled: list[str] = Field(default_factory=list)
    """Judge checks to run even if unmeasured or below min_precision."""
    include_unvalidated: bool = False
    estimate_confirm_rate: float = 0.3
    """Share of triaged (window, check) pairs expected to go to confirmation, for estimates."""
    estimate_candidate_confirm_rate: float = 0.6
    """Same, for rule candidates (which are likelier to be real)."""
    estimate_confirm_output_tokens: int = 800
    """Expected output per confirmation call, including thinking. A guess until measured."""


class Price(BaseModel):
    input: float
    """USD per million input tokens."""
    output: float
    """USD per million output tokens."""


# Anthropic list prices (USD per million tokens), checked 2026-10-02. Prices change: override or
# extend under [prices."model-name"] in config.toml. Unknown models get no cost estimate.
DEFAULT_PRICES: dict[str, Price] = {
    "claude-haiku-4-5": Price(input=1.0, output=5.0),
    "claude-sonnet-5": Price(input=2.0, output=10.0),
    "claude-sonnet-4-6": Price(input=3.0, output=15.0),
    "claude-opus-5": Price(input=5.0, output=25.0),
    "claude-opus-4-8": Price(input=5.0, output=25.0),
}


class Config(BaseModel):
    redact: RedactConfig = Field(default_factory=RedactConfig)
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    preview_chars: int = 500
    thresholds: Thresholds = Field(default_factory=Thresholds)
    checks: ChecksConfig = Field(default_factory=ChecksConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    prices: dict[str, Price] = Field(default_factory=lambda: dict(DEFAULT_PRICES))


class ConfigError(ValueError):
    """A config file problem the user can fix; shown as a one-line message."""


def config_dir() -> Path:
    """macOS: ~/Library/Application Support/tracedoc; Linux: $XDG_CONFIG_HOME/tracedoc
    (default ~/.config/tracedoc); Windows: %LOCALAPPDATA%\\tracedoc."""
    return Path(user_config_dir(NAME, appauthor=False))


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
    # Start from the default price table so user entries add to it rather than replace it.
    data: dict[str, Any] = {"prices": {k: v.model_dump() for k, v in DEFAULT_PRICES.items()}}
    sources = [user_config_path()] if use_user_config else []
    if path is not None:
        sources.append(path)
    for source in sources:
        if source.is_file():
            with source.open("rb") as f:
                loaded = tomllib.load(f)
            if "api_key" in loaded.get("llm", {}):
                raise ConfigError(
                    f"{source}: [llm] api_key is not supported; keys come only from environment "
                    "variables (set api_key_env to the variable's name)"
                )
            data = _deep_merge(data, loaded)
    return Config.model_validate(data)
