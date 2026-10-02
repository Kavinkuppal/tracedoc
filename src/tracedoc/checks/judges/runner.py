"""Two-stage judging: plan windows, triage with a cheap model, confirm with a strong one.

Candidates come from three places:
  - rule findings marked `needs_confirmation` (confirmed directly; the rule was the triage),
  - every handoff (capped per run, distinct sender/receiver pairs first),
  - a seeded random sample of runs (a window at the end of each run).

Triage asks one cheap call per window about every judge check in scope for that window.
Each (window, check) the triage flags goes to one confirmation call with that check's prompt.
"""

from __future__ import annotations

import random
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import BaseModel

from ...config import Config
from ...llm.client import (
    ConfirmModelUnavailableError,
    KeyRejectedError,
    LLMClient,
    LLMError,
    ModelUnavailableError,
)
from ...llm.cost import BudgetExceededError, Spend, estimate_tokens, price_of
from ...schema import Finding, Run
from ..util import excerpt
from .prompts import JudgePrompt, confirmation_prompts
from .windows import Window, run_end_window, window_around

SYSTEM = """You review traces of LLM agent runs to diagnose specific failure modes.
You see a window of steps from one run, labeled S1, S2, and so on, plus the user's task and the
agents' system prompts. Judge only from what is shown. Be conservative: flag a problem only
when the steps contain concrete evidence for it, and cite the step labels that show it.
Confidence is a number from 0 to 1."""

TRIAGE_MAX_TOKENS = 1_000
CONFIRM_MAX_TOKENS = 8_000
TRIAGE_OUT_PER_CHECK = 40


class TriageItem(BaseModel):
    check_id: str
    flagged: bool
    confidence: float


class TriageResult(BaseModel):
    items: list[TriageItem]


class Verdict(BaseModel):
    verdict: bool
    confidence: float
    evidence_step_ids: list[str]
    explanation: str
    suggested_fix: str


@dataclass
class WindowTask:
    window: Window
    checks: list[JudgePrompt] = field(default_factory=list)
    """Checks to triage on this window."""
    confirmation: JudgePrompt | None = None
    """For rule candidates: the confirmation prompt (also the only check triaged)."""


@dataclass
class Plan:
    tasks: list[WindowTask]

    @property
    def triage_pairs(self) -> int:
        return sum(len(t.checks) for t in self.tasks)

    @property
    def candidates(self) -> int:
        return sum(1 for t in self.tasks if t.confirmation is not None)


@dataclass
class Estimate:
    triage_calls: int
    confirm_calls_expected: int
    confirm_calls_max: int
    tokens_in: int
    tokens_out: int
    cost: float | None
    cost_max: float | None
    unpriced_models: list[str]


@dataclass
class JudgeOutcome:
    findings: list[Finding]
    """Judge findings plus rule findings, with candidates confirmed or dropped."""
    calls: int = 0
    errors: list[str] = field(default_factory=list)
    stopped_for_budget: str | None = None


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _triage_prompt(window_text: str, checks: list[JudgePrompt]) -> str:
    questions = "\n".join(f"- {c.id}: {c.triage}" for c in checks)
    return (
        f"{window_text}\n\nFor each check below, answer whether this window plausibly shows the "
        f"problem (flagged true/false) and your confidence. Answer every check.\n{questions}"
    )


def _confirm_prompt(window_text: str, prompt: JudgePrompt, rule_explanation: str | None) -> str:
    context = f"\nRULE FINDING TO CONFIRM: {rule_explanation}\n" if rule_explanation else ""
    return (
        f"{window_text}\n{context}\nCHECK: {prompt.title} ({prompt.id})\n{prompt.confirm.strip()}"
        "\n\nReturn verdict=true only if the problem is present. Cite evidence by step label "
        "(for example S3). If verdict is true, give a one-paragraph explanation and a concrete, "
        "actionable suggested_fix for the developer of this agent."
    )


class Judge:
    def __init__(
        self,
        config: Config,
        judges: dict[str, JudgePrompt],
        client: LLMClient | None = None,
        spend: Spend | None = None,
    ) -> None:
        self.config = config
        self.judges = judges
        self.confirmations = confirmation_prompts()
        self.client = client
        self.spend = spend or Spend(prices=config.prices)
        self._lock = threading.Lock()
        self._reserved = 0.0

    # ---- planning ------------------------------------------------------------

    def plan(self, runs: list[Run], findings: list[Finding]) -> Plan:
        jc = self.config.judge
        k = jc.window_steps
        by_id = {r.id: r for r in runs}
        tasks: list[WindowTask] = []
        per_signature: Counter[tuple[str, ...]] = Counter()

        def take(*sig: str) -> bool:
            per_signature[sig] += 1
            return per_signature[sig] <= jc.max_windows_per_signature

        for f in sorted(findings, key=lambda f: -f.confidence):
            prompt = self.confirmations.get(f.check_id)
            if (
                f.needs_confirmation
                and prompt is not None
                and f.run_id in by_id
                and take(f.check_id, f.node or "", f.key)
            ):
                window = window_around(by_id[f.run_id], f.evidence_step_ids, k, "candidate")
                window.candidate = f
                tasks.append(WindowTask(window=window, checks=[prompt], confirmation=prompt))

        handoff_checks = [p for p in self.judges.values() if "handoff" in p.scope]
        run_checks = [p for p in self.judges.values() if "run" in p.scope]
        if handoff_checks:
            for run in runs:
                handoffs = run.steps_of("handoff")
                seen: set[tuple[str | None, str | None]] = set()
                distinct = []
                for h in handoffs:
                    if (h.handoff_from, h.handoff_to) not in seen:
                        seen.add((h.handoff_from, h.handoff_to))
                        distinct.append(h)
                rest = [h for h in handoffs if h not in distinct]
                for h in (distinct + rest)[: jc.max_handoffs_per_run]:
                    if not take("handoff", str(h.handoff_from), str(h.handoff_to)):
                        continue
                    window = window_around(run, [h.id], k, "handoff")
                    tasks.append(WindowTask(window=window, checks=handoff_checks))
        if run_checks:
            rng = random.Random(jc.seed)
            sample = sorted(runs, key=lambda r: r.id)
            rng.shuffle(sample)
            for run in sample[: jc.sample_runs]:
                if any(s.kind != "node" for s in run.steps):
                    tasks.append(WindowTask(window=run_end_window(run, k), checks=run_checks))
        return Plan(tasks=tasks)

    def estimate(self, plan: Plan) -> Estimate:
        """Tokens per window x number of windows x price. Triage always runs; a configured share
        of triaged pairs is expected to reach confirmation (the max assumes all of them do)."""
        llm, jc = self.config.llm, self.config.judge
        sys_tokens = estimate_tokens(SYSTEM)
        triage_calls = triage_in = triage_out = 0
        candidate_in: list[int] = []
        pair_in: list[int] = []
        for task in plan.tasks:
            if not task.checks:
                continue
            base = sys_tokens + estimate_tokens(task.window.render(jc.max_chars_per_field))
            triage_calls += 1
            triage_in += base + sum(estimate_tokens(c.triage) for c in task.checks)
            triage_out += TRIAGE_OUT_PER_CHECK * len(task.checks)
            pairs = [base + estimate_tokens(c.confirm) for c in task.checks]
            if task.confirmation is not None:
                candidate_in += pairs
            else:
                pair_in += pairs
        cand_rate, rate = jc.estimate_candidate_confirm_rate, jc.estimate_confirm_rate
        expected_calls = round(cand_rate * len(candidate_in)) + round(rate * len(pair_in))
        confirm_in = round(cand_rate * sum(candidate_in)) + round(rate * sum(pair_in))
        confirm_in_max = sum(candidate_in) + sum(pair_in)
        max_calls = len(candidate_in) + len(pair_in)

        triage_cost = price_of(self.config.prices, llm.triage_model, triage_in, triage_out)
        confirm_cost = price_of(
            self.config.prices,
            llm.confirm_model,
            confirm_in,
            expected_calls * self.config.judge.estimate_confirm_output_tokens,
        )
        confirm_cost_max = price_of(
            self.config.prices,
            llm.confirm_model,
            confirm_in_max,
            max_calls * self.config.judge.estimate_confirm_output_tokens,
        )
        unpriced = []
        if triage_calls and triage_cost is None:
            unpriced.append(llm.triage_model)
        if max_calls and confirm_cost is None:
            unpriced.append(llm.confirm_model)
        priced = not unpriced
        return Estimate(
            triage_calls=triage_calls,
            confirm_calls_expected=expected_calls,
            confirm_calls_max=max_calls,
            tokens_in=triage_in + confirm_in,
            tokens_out=triage_out
            + expected_calls * self.config.judge.estimate_confirm_output_tokens,
            cost=(triage_cost or 0.0) + (confirm_cost or 0.0) if priced else None,
            cost_max=(triage_cost or 0.0) + (confirm_cost_max or 0.0) if priced else None,
            unpriced_models=unpriced,
        )

    # ---- execution -----------------------------------------------------------

    def _call(self, model: str, prompt: str, schema: type, max_tokens: int, effort: str | None):  # noqa: ANN202
        assert self.client is not None
        projected_in = estimate_tokens(SYSTEM) + estimate_tokens(prompt)
        projected_out = (
            TRIAGE_OUT_PER_CHECK * 8
            if schema is TriageResult
            else self.config.judge.estimate_confirm_output_tokens
        )
        resolve = getattr(self.client, "resolve", None)
        priced_as = resolve(model) if resolve else model  # a fallback model costs its own price
        with self._lock:
            self.spend.check(priced_as, projected_in, projected_out, reserved=self._reserved)
            reserve = price_of(self.config.prices, priced_as, projected_in, projected_out) or 0.0
            self._reserved += reserve
        try:
            result = self.client.complete_json(
                SYSTEM, prompt, schema, model=model, max_tokens=max_tokens, effort=effort
            )
        finally:
            with self._lock:
                self._reserved -= reserve
        with self._lock:
            self.spend.record(result.model, result.tokens_in, result.tokens_out)
        return result.value

    def judge_text(
        self,
        text: str,
        checks: list[JudgePrompt],
        *,
        rule_explanation: str | None = None,
        stop: threading.Event | None = None,
    ) -> dict[str, Verdict | None]:
        """Triage `text` on all `checks`, then confirm each flagged one.

        Returns {check_id: Verdict} for confirmed calls and {check_id: None} for checks the
        triage dismissed. Checks skipped because `stop` was set are absent.
        """
        llm, jc = self.config.llm, self.config.judge
        triage = self._call(
            llm.triage_model, _triage_prompt(text, checks), TriageResult, TRIAGE_MAX_TOKENS, None
        )
        flagged = {
            i.check_id
            for i in triage.items
            if i.flagged and _clamp(i.confidence) >= jc.triage_min_confidence
        }
        results: dict[str, Verdict | None] = {}
        for prompt in checks:
            if prompt.id not in flagged:
                results[prompt.id] = None
                continue
            if stop is not None and stop.is_set():
                continue
            results[prompt.id] = self._call(
                llm.confirm_model,
                _confirm_prompt(text, prompt, rule_explanation),
                Verdict,
                CONFIRM_MAX_TOKENS,
                llm.confirm_effort,
            )
        return results

    def run(
        self,
        plan: Plan,
        findings: list[Finding],
        progress: Callable[[int, int], None] | None = None,
    ) -> JudgeOutcome:
        jc = self.config.judge
        outcome = JudgeOutcome(findings=[])
        verdicts: dict[str, Verdict | None] = {}  # candidate finding id -> verdict
        judged: list[Finding] = []
        stop = threading.Event()
        fatal: list[Exception] = []
        done = [0]

        def guarded(fn: Callable[[], None]) -> None:
            if stop.is_set():
                return
            try:
                fn()
            except BudgetExceededError as e:
                outcome.stopped_for_budget = str(e)
                stop.set()
            except (KeyRejectedError, ConfirmModelUnavailableError, ModelUnavailableError) as e:
                fatal.append(e)
                stop.set()
            except LLMError as e:
                with self._lock:
                    outcome.errors.append(str(e))
            finally:
                with self._lock:
                    done[0] += 1
                    if progress:
                        progress(done[0], len(plan.tasks))

        def handle(task: WindowTask) -> None:
            cand = task.window.candidate
            results = self.judge_text(
                task.window.render(jc.max_chars_per_field),
                task.checks,
                rule_explanation=cand.explanation if cand else None,
                stop=stop,
            )
            with self._lock:
                for prompt in task.checks:
                    if prompt.id not in results:
                        continue  # not reached (budget stop)
                    v = results[prompt.id]
                    if cand is not None:
                        verdicts[cand.id] = v
                    elif v and v.verdict and _clamp(v.confidence) >= jc.confirm_min_confidence:
                        judged.append(self._finding(task.window, prompt, v))

        with ThreadPoolExecutor(max_workers=max(1, jc.concurrency)) as pool:
            list(pool.map(lambda t: guarded(lambda: handle(t)), plan.tasks))
        if fatal:
            raise fatal[0]
        outcome.calls = self.spend.calls

        # Confirm or drop rule candidates. Candidates never judged (no prompt, budget, error)
        # stay as unconfirmed.
        for f in findings:
            if f.id not in verdicts:
                outcome.findings.append(f)
                continue
            v = verdicts[f.id]
            if v is None or not v.verdict or _clamp(v.confidence) < jc.confirm_min_confidence:
                continue
            outcome.findings.append(
                f.model_copy(
                    update={
                        "needs_confirmation": False,
                        "confidence": round(_clamp(v.confidence), 2),
                        "explanation": f"{f.explanation} {v.explanation}".strip(),
                        "suggested_fix": v.suggested_fix or f.suggested_fix,
                    }
                )
            )
        outcome.findings.extend(_dedupe(judged))
        return outcome

    def _finding(self, window: Window, prompt: JudgePrompt, v: Verdict) -> Finding:
        evidence = [s for s in (window.step_for(lbl) for lbl in v.evidence_step_ids) if s]
        anchor = evidence[0] if evidence else window.step_for("S1")
        handoff = next(
            (s for s in window.steps if s.id in window.focus and s.kind == "handoff"), None
        )
        key = (
            f"{handoff.handoff_from}->{handoff.handoff_to}"
            if handoff
            else (anchor.node if anchor else "") or ""
        )
        return Finding(
            check_id=prompt.id,
            group=prompt.group,
            method="judge",
            severity=prompt.default_severity,
            run_id=window.run.id,
            key=key,
            evidence_step_ids=[s.id for s in evidence],
            evidence_excerpts=[
                excerpt(s.output if s.kind != "handoff" else s.input, 160) for s in evidence[:3]
            ],
            explanation=v.explanation,
            suggested_fix=v.suggested_fix,
            confidence=round(_clamp(v.confidence), 2),
            node=(handoff.handoff_to if handoff else anchor.node if anchor else None),
            role=(handoff.role if handoff else anchor.role if anchor else None),
        )


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """The same check can fire on overlapping windows of one run; keep the most confident."""
    best: dict[tuple[str, str, str], Finding] = {}
    for f in findings:
        k = (f.check_id, f.run_id, f.key)
        if k not in best or f.confidence > best[k].confidence:
            best[k] = f
    return list(best.values())
