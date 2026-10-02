"""The data-sharing payload: a strict allow-list. Nothing else can be expressed in it.

Every field is a fixed enum, a number, a validated version string, a UUID4, or a model name
that passes `public_model_name`. Unknown fields are rejected (`extra="forbid"`), on both the
client and the server, which validates contributions against these same models.

Never in the payload: prompts, inputs, outputs, tool arguments or results, code, node or tool
names as written, file paths, keys, or anything identifying the developer or their users.
`tests/test_share.py` feeds traces with unique markers in every content field and checks that
none of them reaches the serialized payload.
"""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..__about__ import __version__
from ..schema import Finding, Role, Run, RunStatus, Severity, TaskType

SCHEMA_VERSION = 1

_VERSION = re.compile(r"^\d{1,4}(\.\d{1,6}){0,3}([.+-]?[a-z0-9]{1,12}){0,2}$")
# Public model IDs only: a known family, then tokens that are version-like ("4o", "8b",
# "20250101", "3.5") or from a fixed vocabulary. Anything else, including fine-tuned and custom
# IDs that can carry org or project names ("ft:gpt-4o:acme:..."), becomes "custom".
_FAMILIES = frozenset(
    (
        "claude",
        "gpt",
        "chatgpt",
        "o1",
        "o3",
        "o4",
        "gemini",
        "llama",
        "mistral",
        "mixtral",
        "codestral",
        "command",
        "qwen",
        "deepseek",
        "phi",
        "gemma",
    )
)
_VOCAB = frozenset(
    (
        "opus",
        "sonnet",
        "haiku",
        "fable",
        "mythos",
        "instant",
        "mini",
        "nano",
        "turbo",
        "pro",
        "flash",
        "lite",
        "ultra",
        "preview",
        "latest",
        "chat",
        "instruct",
        "it",
        "vision",
        "large",
        "medium",
        "small",
        "tiny",
        "base",
        "coder",
        "code",
        "reasoner",
        "r",
        "v",
        "r1",
        "v3",
        "omni",
        "audio",
        "realtime",
        "search",
        "exp",
        "experimental",
    )
)
_VERSIONISH = re.compile(r"^(\d+[a-z]?|\d+(\.\d+)+[a-z]?|[a-z]\d+(\.\d+)?)$")
PLACEHOLDER_MODELS = frozenset({"custom", "unknown"})


def public_model_name(model: str | None) -> str:
    if not model:
        return "unknown"
    name = model.strip().lower()
    for prefix in ("us.anthropic.", "eu.anthropic.", "anthropic.", "openai/", "anthropic/"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    if len(name) > 48 or not re.fullmatch(r"[a-z0-9.-]+", name):
        return "custom"
    tokens = name.split("-")
    if tokens[0] not in _FAMILIES and not any(tokens[0].startswith(f) for f in _FAMILIES):
        return "custom"
    head = tokens[0]
    family = head if head in _FAMILIES else next(f for f in _FAMILIES if head.startswith(f))
    rest_of_head = head[len(family) :]
    if rest_of_head and not _VERSIONISH.match(rest_of_head.lstrip(".")):
        return "custom"
    if all(t in _VOCAB or _VERSIONISH.match(t) for t in tokens[1:]):
        return name
    return "custom"


def known_check_ids() -> frozenset[str]:
    from ..checks import rules  # noqa: F401  (registers rule checks)
    from ..checks.base import REGISTRY
    from ..checks.judges.prompts import confirmation_prompts, judge_prompts

    return frozenset(REGISTRY) | frozenset(judge_prompts()) | frozenset(confirmation_prompts())


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SharedFinding(_Strict):
    check_id: str
    severity: Severity
    role: Role | None

    @field_validator("check_id")
    @classmethod
    def _known_check(cls, v: str) -> str:
        if v not in known_check_ids():
            raise ValueError("unknown check id")
        return v


class GraphShape(_Strict):
    agent_count: int = Field(ge=0, le=10_000)
    tool_count: int = Field(ge=0, le=10_000)
    max_depth: int = Field(ge=0, le=1_000)


class ModelsForRole(_Strict):
    role: Role
    models: list[str] = Field(max_length=20)

    @field_validator("models")
    @classmethod
    def _public(cls, v: list[str]) -> list[str]:
        if any(m not in PLACEHOLDER_MODELS and public_model_name(m) != m for m in v):
            raise ValueError("model names must be public model IDs, 'custom', or 'unknown'")
        return v


class SharedRun(_Strict):
    task_type: TaskType
    framework: Literal["langgraph", "langchain", "unknown"]
    framework_version: str | None
    models_by_role: list[ModelsForRole] = Field(max_length=10)
    graph: GraphShape
    status: RunStatus
    step_count: int = Field(ge=0)
    tokens_in: int = Field(ge=0)
    tokens_out: int = Field(ge=0)
    cost: float | None = Field(default=None, ge=0)
    duration_s: float | None = Field(default=None, ge=0)
    findings: list[SharedFinding] = Field(max_length=1_000)

    @field_validator("framework_version")
    @classmethod
    def _version(cls, v: str | None) -> str | None:
        if v is not None and not _VERSION.match(v):
            raise ValueError("not a version string")
        return v


class SharedVote(_Strict):
    check_id: str
    vote: Literal["up", "down"]

    @field_validator("check_id")
    @classmethod
    def _known_check(cls, v: str) -> str:
        if v not in known_check_ids():
            raise ValueError("unknown check id")
        return v


class Contribution(_Strict):
    schema_version: Literal[1]
    install_id: UUID
    tool_version: str
    runs: list[SharedRun] = Field(max_length=10_000)
    feedback: list[SharedVote] = Field(default_factory=list, max_length=10_000)

    @field_validator("install_id")
    @classmethod
    def _uuid4(cls, v: UUID) -> UUID:
        if v.version != 4:
            raise ValueError("install_id must be a random UUID4")
        return v

    @field_validator("tool_version")
    @classmethod
    def _tool_version(cls, v: str) -> str:
        if not _VERSION.match(v):
            raise ValueError("not a version string")
        return v


# ---- building ---------------------------------------------------------------------------------


def _depth(run: Run) -> int:
    parent = {s.id: s.parent_id for s in run.steps}
    best = 0
    for s in run.steps:
        d, cur, seen = 1, s.parent_id, set()
        while cur is not None and cur in parent and cur not in seen:
            seen.add(cur)
            d += 1
            cur = parent[cur]
        best = max(best, d)
    return best


def shared_run(run: Run, findings: list[Finding]) -> SharedRun:
    by_role: dict[str, set[str]] = {}
    for s in run.steps_of("llm"):
        by_role.setdefault(s.role or "other", set()).add(public_model_name(s.model))
    agents = {s.node for s in run.steps_of("node") if s.node and s.role != "tool_executor"}
    tools = {s.name for s in run.steps_of("tool")}
    framework = run.framework if run.framework in ("langgraph", "langchain") else "unknown"
    version = (
        run.framework_version
        if run.framework_version and _VERSION.match(run.framework_version)
        else None
    )
    known = known_check_ids()
    return SharedRun(
        task_type=run.task_type,
        framework=framework,  # type: ignore[arg-type]
        framework_version=version,
        models_by_role=[
            ModelsForRole(role=role, models=sorted(models))  # type: ignore[arg-type]
            for role, models in sorted(by_role.items())
        ],
        graph=GraphShape(agent_count=len(agents), tool_count=len(tools), max_depth=_depth(run)),
        status=run.status,
        step_count=run.totals.step_count,
        tokens_in=run.totals.tokens_in,
        tokens_out=run.totals.tokens_out,
        cost=round(run.totals.cost, 6) if run.totals.cost is not None else None,
        duration_s=round(run.totals.duration_s, 3) if run.totals.duration_s is not None else None,
        findings=[
            SharedFinding(check_id=f.check_id, severity=f.severity, role=f.role)
            for f in findings
            if f.run_id == run.id and f.check_id in known
        ],
    )


def build_contribution(
    install_id: UUID,
    runs: list[Run],
    findings: list[Finding],
    votes: list[tuple[str, str]] | None = None,
) -> Contribution:
    known = known_check_ids()
    return Contribution(
        schema_version=SCHEMA_VERSION,
        install_id=install_id,
        tool_version=__version__,
        runs=[shared_run(r, findings) for r in runs],
        feedback=[
            SharedVote(check_id=c, vote=v)  # type: ignore[arg-type]
            for c, v in votes or []
            if c in known and v in ("up", "down")
        ],
    )


def serialize(contribution: Contribution) -> bytes:
    """The exact bytes that are previewed, logged, and uploaded."""
    return contribution.model_dump_json(indent=2).encode("utf-8")
