"""framework.*: known framework pitfalls, one rule check per entry in checks/pitfalls.py."""

from __future__ import annotations

from ...catalog import fill
from ...schema import Finding, Run
from ..base import Check, register
from ..pitfalls import PITFALLS, Pitfall
from ..util import excerpt


def _check_for(pitfall: Pitfall) -> type[Check]:
    class PitfallCheck(Check):
        id = pitfall.id
        group = "design"
        method = "rule"
        default_severity = pitfall.severity

        def run(self, runs: list[Run]) -> list[Finding]:
            findings = []
            for run in runs:
                found = pitfall.detect(run)
                if found is None:
                    continue
                excerpts = [f"Request: {excerpt(run.input_preview, 120)}"]
                if pitfall.evidence:
                    excerpts.append(fill(pitfall.evidence, **found.facts))
                findings.append(
                    self.finding(
                        run,
                        key=found.step.node or "",
                        node=found.step.node,
                        role=found.step.role,
                        evidence_step_ids=[found.step.id],
                        evidence_excerpts=excerpts,
                        facts=found.facts,
                    )
                )
            return findings

    PitfallCheck.__name__ = PitfallCheck.__qualname__ = f"Pitfall_{pitfall.id.replace('.', '_')}"
    return register(PitfallCheck)


PITFALL_CHECKS = [_check_for(p) for p in PITFALLS]
