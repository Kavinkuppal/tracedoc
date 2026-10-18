"""The Check interface, the registry, and the entry point that runs rule checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar

from ..catalog import check_in_place_template, check_templates, check_title, fill
from ..config import Config
from ..schema import Finding, Group, Method, Run, Severity
from .safety import safe_fix
from .util import agent_instructions, instruction_in_place

REGISTRY: dict[str, type[Check]] = {}

C = TypeVar("C", bound="type[Check]")


class Check(ABC):
    id: ClassVar[str]
    group: ClassVar[Group]
    method: ClassVar[Method]
    default_severity: ClassVar[Severity]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.t = config.thresholds

    @abstractmethod
    def run(self, runs: list[Run]) -> list[Finding]: ...

    @classmethod
    def catalog_title(cls) -> str:
        return check_title(cls.id)

    def finding(
        self,
        run: Run,
        *,
        facts: dict[str, Any],
        variant: str | None = None,
        **fields: object,
    ) -> Finding:
        """A Finding with this check's identity and default severity. Its sentences come from
        the reviewed catalog (catalog.toml), filled with `facts`; code never writes them."""
        fields.setdefault("severity", self.default_severity)
        what, change = check_templates(self.id, variant)
        values = {k: v for k, v in facts.items()}
        values.setdefault("agent", fields.get("node") or "the agent")
        fix = fill(change, **values)
        in_place = check_in_place_template(self.id, variant)
        node = fields.get("node")
        if in_place and instruction_in_place(fix, agent_instructions(run, node)):  # type: ignore[arg-type]
            fix = fill(in_place, **values)  # don't suggest what the agent's prompt already says
        fix = safe_fix(fix, str(values["agent"]))  # never act for the user without asking
        return Finding(
            check_id=self.id,
            group=self.group,
            method=self.method,
            run_id=run.id,
            explanation=fill(what, **values),
            suggested_fix=fix,
            facts={k: str(v) for k, v in values.items()},
            **fields,  # type: ignore[arg-type]
        )


def register(cls: C) -> C:
    if cls.id in REGISTRY:
        raise ValueError(f"duplicate check id {cls.id}")
    REGISTRY[cls.id] = cls
    return cls


def enabled_checks(config: Config, method: Method) -> list[Check]:
    from . import rules  # noqa: F401  (importing registers the rule checks)

    return [
        cls(config)
        for check_id, cls in sorted(REGISTRY.items())
        if cls.method == method and check_id not in config.checks.disabled
    ]


def run_rule_checks(runs: list[Run], config: Config | None = None) -> list[Finding]:
    config = config or Config()
    findings: list[Finding] = []
    for check in enabled_checks(config, "rule"):
        findings.extend(check.run(runs))
    overrides = config.checks.severity
    for f in findings:
        if f.check_id in overrides:
            f.severity = overrides[f.check_id]
    return findings
