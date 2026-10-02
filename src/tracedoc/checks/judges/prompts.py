"""Load versioned judge prompt files and decide which judge checks are enabled.

Prompt files live in `prompts/`: `<check_id>.v<N>.toml` for judge checks and
`confirm.<check_id>.v<N>.toml` for confirming rule candidates. The highest version wins.

`precision.json` (written by `evals/run_evals.py`) records measured precision per check and
prompt version. A judge check is disabled by default unless it was measured at or above
`judge.min_precision` for its current prompt version, or it is explicitly enabled.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ...config import Config
from ...schema import Group, Severity

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

PROMPTS_DIR = Path(__file__).parent / "prompts"
PRECISION_FILE = Path(__file__).parent / "precision.json"
_NAME = re.compile(r"^(?P<confirm>confirm\.)?(?P<id>.+)\.v(?P<version>\d+)\.toml$")


@dataclass(frozen=True)
class JudgePrompt:
    id: str
    version: int
    title: str
    group: Group
    default_severity: Severity
    scope: tuple[str, ...]
    confirm: str
    triage: str = ""
    confirms: str | None = None
    """For rule-candidate confirmations: the rule check whose candidates this confirms."""


def _parse(path: Path) -> JudgePrompt:
    with path.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)
    data["scope"] = tuple(data["scope"])
    return JudgePrompt(**data)


@cache
def _load(directory: Path) -> tuple[dict[str, JudgePrompt], dict[str, JudgePrompt]]:
    judges: dict[str, JudgePrompt] = {}
    confirmations: dict[str, JudgePrompt] = {}
    for path in sorted(directory.glob("*.toml")):
        match = _NAME.match(path.name)
        if not match:
            raise ValueError(f"unexpected prompt file name: {path.name}")
        prompt = _parse(path)
        if prompt.version != int(match["version"]) or prompt.id != match["id"]:
            raise ValueError(f"{path.name}: id/version inside the file do not match its name")
        target = confirmations if match["confirm"] else judges
        key = prompt.confirms or prompt.id if match["confirm"] else prompt.id
        if key not in target or target[key].version < prompt.version:
            target[key] = prompt
    return judges, confirmations


def judge_prompts(directory: Path = PROMPTS_DIR) -> dict[str, JudgePrompt]:
    return _load(directory)[0]


def confirmation_prompts(directory: Path = PROMPTS_DIR) -> dict[str, JudgePrompt]:
    return _load(directory)[1]


def load_precision(path: Path = PRECISION_FILE) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def enabled_judges(
    config: Config, precision: dict[str, dict[str, Any]] | None = None
) -> tuple[dict[str, JudgePrompt], dict[str, str]]:
    """Return (enabled judge prompts, {check_id: reason} for the disabled ones)."""
    precision = load_precision() if precision is None else precision
    enabled: dict[str, JudgePrompt] = {}
    disabled: dict[str, str] = {}
    for check_id, prompt in judge_prompts().items():
        measured = precision.get(check_id)
        if check_id in config.checks.disabled:
            disabled[check_id] = "disabled in config"
        elif check_id in config.judge.enabled or config.judge.include_unvalidated:
            enabled[check_id] = prompt
        elif (
            measured is None
            or measured.get("mock")
            or measured.get("prompt_version") != prompt.version
        ):
            disabled[check_id] = f"no precision measured for prompt v{prompt.version}"
        elif measured["precision"] < config.judge.min_precision:
            disabled[check_id] = (
                f"precision {measured['precision']:.0%} is below {config.judge.min_precision:.0%}"
            )
        else:
            enabled[check_id] = prompt
    return enabled, disabled
