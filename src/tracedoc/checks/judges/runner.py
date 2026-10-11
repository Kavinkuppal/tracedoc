"""Two-stage judging: plan windows, triage with a cheap model, confirm with a strong one.

Candidates come from three places:
  - rule findings marked `needs_confirmation` (confirmed directly; the rule was the triage),
  - every handoff (capped per run, distinct sender/receiver pairs first),
  - a seeded random sample of runs (a window at the end of each run).

Triage asks one cheap call per window about every judge check in scope for that window.
Each (window, check) the triage flags goes to one confirmation call with that check's prompt.

A confirmed judge finding carries three short texts written by the model (what happened, what
to change, one quote from the window), checked against the limits in slots.py. Everything else
about it (title, explanation, why it matters) comes from the catalog. A confirmed rule
candidate keeps the rule's catalog text; the judge only confirms it and sets its confidence.

Every call is answered from the judge cache when possible (see llm/cache.py), so the same traces
give the same findings. Cached answers cost nothing now and skip the budget check.
"""

from __future__ import annotations

import random
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ...catalog import limit
from ...config import Config
from ...llm.client import (
    ConfirmModelUnavailableError,
    KeyRejectedError,
    LLMClient,
    LLMError,
    ModelUnavailableError,
)
from ...llm.cost import BudgetExceededError, Spend, estimate_tokens, price_of
from ...schema import Finding, Run, Step
from ..util import excerpt
from .prompts import JudgePrompt, confirmation_prompts
from .slots import fitted, slot_problems
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


ChangeType = Literal["prompt instruction", "tool", "graph edge", "setting"]


class Verdict(BaseModel):
    """The answer to a judge check. The three text fields are the only text a model writes
    into a report."""

    verdict: bool
    confidence: float
    evidence_step_ids: list[str]
    what_happened: str = Field(
        description="What went wrong in this run: specific, with names and numbers from the "
        f"steps. At most {limit('what_happened')} characters. Empty if verdict is false."
    )
    what_to_change: str = Field(
        description="One concrete change to the developer's system that would prevent this: "
        "a prompt instruction (quote the words to add), a tool, a graph edge, or a setting. "
        f"At most {limit('what_to_change')} characters. Empty if verdict is false."
    )
    change_type: ChangeType = Field(description="Which kind of change what_to_change is.")
    evidence_quote: str = Field(
        description="One quote copied exactly, character for character, from the steps shown, "
        f"that shows the problem. At most {limit('evidence_quote')} characters. Empty if "
        "verdict is false."
    )


class Confirmation(BaseModel):
    """The answer when confirming a rule candidate: the rule's catalog text stays as it is."""

    verdict: bool
    confidence: float
    evidence_step_ids: list[str]


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
    unreviewed_candidates: dict[str, int] = field(default_factory=dict)
    """Rule candidates per check the judge will not see because of the per-pattern cap."""
    unreviewed_handoffs: int = 0
    """Handoffs left out by the per-run or per-agent-pair cap."""
    unsampled_runs: int = 0
    """Runs left out of the sample for whole-run checks."""

    def unreviewed_summary(self, config: Config) -> str | None:
        """One sentence on what the judge caps left out, or None if nothing was left out."""
        jc = config.judge
        parts = []
        n = sum(self.unreviewed_candidates.values())
        if n:
            parts.append(
                f"{n} rule finding(s) were not reviewed (at most {jc.max_windows_per_signature} "
                "per problem pattern)"
            )
        if self.unreviewed_handoffs:
            parts.append(
                f"{self.unreviewed_handoffs} handoff(s) were not reviewed (at most "
                f"{jc.max_handoffs_per_run} per run and {jc.max_windows_per_signature} per "
                "pair of agents)"
            )
        if self.unsampled_runs:
            parts.append(
                f"{self.unsampled_runs} run(s) were not sampled for whole-run checks (sample "
                f"of {jc.sample_runs})"
            )
        return ("AI judge caps: " + "; ".join(parts) + ".") if parts else None

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
    reused: int = 0
    """Answers taken from the judge cache instead of a model call."""
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
    ask = (
        "\n\nReturn verdict=true only if the problem is present. Cite evidence by step label "
        "(for example S3)."
    )
    if not prompt.confirms:
        ask += (
            " If verdict is true, also write: what_happened (what went wrong, specific, with "
            f"names and numbers; at most {limit('what_happened')} characters), what_to_change "
            "(one concrete change to the developer's system: a prompt instruction, quoting the "
            "words to add, or a change to a tool, a graph edge or a setting; name the agent or "
            f"tool; at most {limit('what_to_change')} characters), change_type, and "
            "evidence_quote (one quote copied exactly from the steps above; at most "
            f"{limit('evidence_quote')} characters). Don't suggest a change the steps show is "
            "already in place, such as an instruction the agent's prompt already has, or a "
            "cheaper model when the agent already uses the cheapest one."
        )
    return (
        f"{window_text}\n{context}\nCHECK: {prompt.id} (prompt v{prompt.version})\n"
        f"{prompt.confirm.strip()}{ask}"
    )


def _retry_prompt(prompt: str, answer: Verdict, problems: list[str]) -> str:
    """The same question again, with the previous answer and what was wrong with it."""
    return (
        f"{prompt}\n\nYOUR PREVIOUS ANSWER:\n{answer.model_dump_json()}\n\nIt broke these "
        "rules:\n" + "\n".join(f"- {p}" for p in problems) + "\nReturn a corrected answer."
    )


def _step_text(step: Step) -> Any:
    return step.input if step.kind == "handoff" else (step.error or step.output)


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

        unreviewed: Counter[str] = Counter()
        unreviewed_handoffs = 0
        unsampled = 0

        def take(*sig: str) -> bool:
            per_signature[sig] += 1
            return per_signature[sig] <= jc.max_windows_per_signature

        for f in sorted(findings, key=lambda f: -f.confidence):
            prompt = self.confirmations.get(f.check_id)
            if not (f.needs_confirmation and prompt is not None and f.run_id in by_id):
                continue
            if not take(f.check_id, f.node or "", f.key):
                unreviewed[f.check_id] += 1
                continue
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
                ordered = distinct + rest
                unreviewed_handoffs += max(0, len(ordered) - jc.max_handoffs_per_run)
                for h in ordered[: jc.max_handoffs_per_run]:
                    if not take("handoff", str(h.handoff_from), str(h.handoff_to)):
                        unreviewed_handoffs += 1
                        continue
                    window = window_around(run, [h.id], k, "handoff")
                    tasks.append(WindowTask(window=window, checks=handoff_checks))
        if run_checks:
            rng = random.Random(jc.seed)
            sample = sorted(runs, key=lambda r: r.id)
            rng.shuffle(sample)
            judgeable = [r for r in sample if any(s.kind != "node" for s in r.steps)]
            unsampled = max(0, len(judgeable) - jc.sample_runs)
            for run in judgeable[: jc.sample_runs]:
                tasks.append(WindowTask(window=run_end_window(run, k), checks=run_checks))
        return Plan(
            tasks=tasks,
            unreviewed_candidates=dict(sorted(unreviewed.items())),
            unreviewed_handoffs=unreviewed_handoffs,
            unsampled_runs=unsampled,
        )

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
        settings: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "effort": effort,
            "temperature": self.config.llm.temperature,
        }
        lookup = getattr(self.client, "lookup", None)
        hit = lookup(SYSTEM, prompt, schema, **settings) if lookup else None
        if hit is not None:  # reused: free now, so no budget check
            with self._lock:
                self.spend.record_result(hit)
            return hit.value
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
            result = self.client.complete_json(SYSTEM, prompt, schema, **settings)
        finally:
            with self._lock:
                self._reserved -= reserve
        with self._lock:
            self.spend.record_result(result)
        return result.value

    def _confirm(
        self, text: str, prompt: JudgePrompt, rule_explanation: str | None
    ) -> Verdict | Confirmation:
        """One confirmation call. A judge finding whose texts break a limit is asked again once;
        then its texts are cut to fit (see slots.py)."""
        llm = self.config.llm
        question = _confirm_prompt(text, prompt, rule_explanation)
        if prompt.confirms:
            return self._call(
                llm.confirm_model, question, Confirmation, CONFIRM_MAX_TOKENS, llm.confirm_effort
            )
        answer: Verdict = self._call(
            llm.confirm_model, question, Verdict, CONFIRM_MAX_TOKENS, llm.confirm_effort
        )
        problems = slot_problems(answer, text) if answer.verdict else []
        if problems:
            answer = self._call(
                llm.confirm_model,
                _retry_prompt(question, answer, problems),
                Verdict,
                CONFIRM_MAX_TOKENS,
                llm.confirm_effort,
            )
        if not answer.verdict:
            return answer
        return answer.model_copy(update=fitted(answer, text))

    def judge_text(
        self,
        text: str,
        checks: list[JudgePrompt],
        *,
        rule_explanation: str | None = None,
        stop: threading.Event | None = None,
    ) -> dict[str, Verdict | Confirmation | None]:
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
        results: dict[str, Verdict | Confirmation | None] = {}
        for prompt in checks:
            if prompt.id not in flagged:
                results[prompt.id] = None
                continue
            if stop is not None and stop.is_set():
                continue
            results[prompt.id] = self._confirm(text, prompt, rule_explanation)
        return results

    def run(
        self,
        plan: Plan,
        findings: list[Finding],
        progress: Callable[[int, int], None] | None = None,
    ) -> JudgeOutcome:
        jc = self.config.judge
        outcome = JudgeOutcome(findings=[])
        verdicts: dict[str, Verdict | Confirmation | None] = {}  # candidate id -> answer
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
                    elif (
                        isinstance(v, Verdict)
                        and v.verdict
                        and _clamp(v.confidence) >= jc.confirm_min_confidence
                    ):
                        judged.append(self._finding(task.window, prompt, v))

        with ThreadPoolExecutor(max_workers=max(1, jc.concurrency)) as pool:
            list(pool.map(lambda t: guarded(lambda: handle(t)), plan.tasks))
        if fatal:
            raise fatal[0]
        outcome.calls = self.spend.calls
        outcome.reused = self.spend.reused_calls

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
                        "confirmed_by_judge": True,
                        "confidence": round(_clamp(v.confidence), 2),
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
            evidence_excerpts=[excerpt(_step_text(s), 160) for s in evidence[:3]],
            explanation=v.what_happened,
            suggested_fix=v.what_to_change,
            change_type=v.change_type,
            # A quote the model didn't copy exactly is replaced by the start of the cited step.
            evidence_quote=v.evidence_quote
            or (excerpt(_step_text(evidence[0]), limit("evidence_quote")) if evidence else ""),
            confidence=round(_clamp(v.confidence), 2),
            node=(handoff.handoff_to if handoff else anchor.node if anchor else None),
            role=(handoff.role if handoff else anchor.role if anchor else None),
        )


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """The same check can fire on overlapping windows of one run; keep the most confident.
    Ties are broken by content, never by which call finished first."""
    best: dict[tuple[str, str, str], Finding] = {}
    for f in sorted(findings, key=lambda f: (-f.confidence, f.evidence_step_ids, f.id)):
        best.setdefault((f.check_id, f.run_id, f.key), f)
    return [best[k] for k in sorted(best)]
