"""Core data model. Every trace source normalizes into these."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

Source = Literal["capture", "langsmith", "otel"]
RunStatus = Literal["completed", "no_answer", "error", "recursion_limit", "interrupted"]
"""no_answer: the run finished, but its final reply had no text, so it did not complete."""
StepKind = Literal["llm", "tool", "node", "handoff"]
Role = Literal["planner", "researcher", "writer", "critic", "supervisor", "tool_executor", "other"]
TaskType = Literal[
    "research", "coding", "data_extraction", "customer_support", "planning", "writing", "other"
]
Group = Literal["design", "agents", "verification", "tools_cost"]
Method = Literal["rule", "judge"]
Severity = Literal["high", "medium", "low"]

ROLES: tuple[Role, ...] = (
    "planner",
    "researcher",
    "writer",
    "critic",
    "supervisor",
    "tool_executor",
    "other",
)
TASK_TYPES: tuple[TaskType, ...] = (
    "research",
    "coding",
    "data_extraction",
    "customer_support",
    "planning",
    "writing",
    "other",
)


class Step(BaseModel):
    id: str
    parent_id: str | None = None
    run_id: str
    kind: StepKind
    name: str
    role: Role | None = None
    node: str | None = None
    """The agent this step belongs to: the outermost LangGraph node, named as in the code.
    Steps of a shared tool node are credited to the agent that asked for the tool."""
    graph_node: str | None = None
    """The LangGraph node path the step ran in, outermost first, e.g. "research_agent/tools"."""
    graph_step: int | None = None
    """LangGraph superstep number, when known."""
    started_at: datetime | None = None
    ended_at: datetime | None = None
    input: Any = None
    output: Any = None
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost: float | None = None
    error: str | None = None
    handoff_from: str | None = None
    """For handoff steps: the sending node name."""
    handoff_to: str | None = None
    """For handoff steps: the receiving node name."""

    @property
    def duration_s(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


class Totals(BaseModel):
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float | None = None
    duration_s: float | None = None
    step_count: int = 0
    """Number of graph node executions (LangGraph supersteps' tasks), excluding handoffs."""


class Run(BaseModel):
    id: str
    source: Source
    started_at: datetime | None = None
    ended_at: datetime | None = None
    status: RunStatus
    error: str | None = None
    input_preview: str = ""
    output_preview: str = ""
    final_answer: str = ""
    """Text of the run's final reply ("" when there was none or it had no text)."""
    task_type: TaskType = "other"
    steps: list[Step] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    framework: str | None = None
    framework_version: str | None = None

    def steps_of(self, kind: StepKind) -> list[Step]:
        return [s for s in self.steps if s.kind == kind]


class Impact(BaseModel):
    """Tokens, cost, and latency attributable to a finding (the waste it caused)."""

    tokens: int = 0
    cost: float = 0.0
    latency_s: float = 0.0


class Finding(BaseModel):
    check_id: str
    group: Group
    method: Method
    severity: Severity
    run_id: str
    key: str = ""
    """Short signature key, e.g. the tool name; groups findings across runs."""
    evidence_step_ids: list[str] = Field(default_factory=list)
    evidence_excerpts: list[str] = Field(default_factory=list)
    explanation: str
    suggested_fix: str = ""
    confidence: float = Field(default=1.0, ge=0, le=1)
    node: str | None = None
    role: Role | None = None
    needs_confirmation: bool = False
    """A rule candidate that a judge should confirm before it is trusted."""
    impact: Impact = Field(default_factory=Impact)
    facts: dict[str, str] = Field(default_factory=dict)
    """Values that filled the catalog sentences of a rule finding (agent, tool, counts)."""
    evidence_quote: str = ""
    """One short quote from the trace (AI judge findings: written by the model and verified)."""
    change_type: str | None = None
    """For AI judge findings: prompt instruction, tool, graph edge, or setting."""
    confirmed_by_judge: bool = False
    """A rule candidate the AI judge confirmed (its confidence is then the judge's)."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def id(self) -> str:
        """Stable ID, so feedback on a finding survives re-running the analysis."""
        anchor = self.evidence_step_ids[0] if self.evidence_step_ids else ""
        raw = "|".join([self.check_id, self.run_id, self.key, self.node or "", anchor])
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


class FindingGroup(BaseModel):
    """Findings across runs that share a signature (check, node or role, short key)."""

    check_id: str
    node: str | None = None
    role: Role | None = None
    key: str = ""
    runs_affected: int
    frequency: float
    """Share of analyzed runs affected, 0 to 1."""
    impact: Impact = Field(default_factory=Impact)
    """Tokens, cost and time of the runs this problem appeared in (each run counted once)."""
    representative: Finding
    findings: list[Finding] = Field(default_factory=list)


class Report(BaseModel):
    scorecard: dict[str, Any] = Field(default_factory=dict)
    groups: list[FindingGroup] = Field(default_factory=list)
    signals: list[Finding] = Field(default_factory=list)
    node_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    distributions: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
