"""The Check interface, the registry, and the entry point that runs rule checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypeVar

from ..config import Config
from ..schema import Finding, Group, Method, Run, Severity

REGISTRY: dict[str, type[Check]] = {}

C = TypeVar("C", bound="type[Check]")


class Check(ABC):
    id: ClassVar[str]
    group: ClassVar[Group]
    method: ClassVar[Method]
    default_severity: ClassVar[Severity]
    title: ClassVar[str]

    def __init__(self, config: Config) -> None:
        self.config = config
        self.t = config.thresholds

    @abstractmethod
    def run(self, runs: list[Run]) -> list[Finding]: ...

    def finding(self, run: Run, **fields: object) -> Finding:
        """A Finding pre-filled with this check's identity and default severity."""
        fields.setdefault("severity", self.default_severity)
        return Finding(
            check_id=self.id,
            group=self.group,
            method=self.method,
            run_id=run.id,
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
