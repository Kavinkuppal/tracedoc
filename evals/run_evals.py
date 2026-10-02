"""Measure precision and recall of each judge check on its hand-labeled examples.

    python evals/run_evals.py                 # print results (spends money: uses your model key)
    python evals/run_evals.py --write         # also record results in precision.json
    python evals/run_evals.py --check agents.ignored_output
    python evals/run_evals.py --mock          # no model calls: exercises the harness (CI)

Labels live in evals/labels/<check_id>.jsonl (judge checks) and
evals/labels/confirm.<check_id>.jsonl (confirmations of rule candidates). Each line:
{"id", "label": true|false, "note", "window": <rendered window text>, "rule_explanation"?}

A judge check whose measured precision is below judge.min_precision (default 80%) stays
disabled by default; `--include-unvalidated` or `judge.enabled` in config turns it on.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracedoc.checks.judges.prompts import (
    PRECISION_FILE,
    JudgePrompt,
    confirmation_prompts,
    judge_prompts,
)
from tracedoc.checks.judges.runner import Judge, TriageItem, TriageResult, Verdict
from tracedoc.config import Config, load_config
from tracedoc.llm.client import LLMClient, LLMResult, make_client
from tracedoc.llm.cost import Spend

LABELS = Path(__file__).parent / "labels"


@dataclass
class Score:
    key: str
    prompt_version: int
    n: int
    tp: int
    fp: int
    fn: int
    tn: int
    errors: int

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None


def load_examples(key: str) -> list[dict[str, Any]]:
    path = LABELS / f"{key}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checks_to_run(only: str | None) -> dict[str, JudgePrompt]:
    """{label-file key: prompt}. Confirmation keys are prefixed with 'confirm.'."""
    checks: dict[str, JudgePrompt] = dict(judge_prompts())
    checks.update({f"confirm.{k}": p for k, p in confirmation_prompts().items()})
    if only:
        checks = {k: p for k, p in checks.items() if only in (k, p.id)}
    return checks


class OracleLLM:
    """For --mock: answers from the labels, so the harness can run without a model."""

    def __init__(self, labels: dict[str, bool]) -> None:
        self.labels = labels  # window text -> label

    def complete_json(self, system: str, prompt: str, schema: type, **kwargs: Any) -> LLMResult:
        label = next((v for w, v in self.labels.items() if prompt.startswith(w)), False)
        if schema is TriageResult:
            ids = [
                line[2:].split(":", 1)[0]
                for line in prompt.split("For each check below")[-1].splitlines()
                if line.startswith("- ")
            ]
            value: Any = TriageResult(
                items=[TriageItem(check_id=i, flagged=label, confidence=0.9) for i in ids]
            )
        else:
            value = Verdict(
                verdict=label,
                confidence=0.9,
                evidence_step_ids=["S1"],
                explanation="oracle",
                suggested_fix="oracle",
            )
        return LLMResult(value=value, model=kwargs["model"], tokens_in=0, tokens_out=0)


def evaluate(key: str, prompt: JudgePrompt, examples: list[dict[str, Any]], judge: Judge) -> Score:
    tp = fp = fn = tn = errors = 0
    threshold = judge.config.judge.confirm_min_confidence
    for ex in examples:
        try:
            result = judge.judge_text(
                ex["window"], [prompt], rule_explanation=ex.get("rule_explanation")
            ).get(prompt.id)
        except Exception as e:
            print(f"  {ex['id']}: error {e}", file=sys.stderr)
            errors += 1
            continue
        predicted = bool(result and result.verdict and result.confidence >= threshold)
        if predicted and ex["label"]:
            tp += 1
        elif predicted:
            fp += 1
        elif ex["label"]:
            fn += 1
        else:
            tn += 1
    return Score(key, prompt.version, len(examples), tp, fp, fn, tn, errors)


def _pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{x:5.0%}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", help="Only this check id (or label-file key).")
    parser.add_argument("--write", action="store_true", help="Record results in precision.json.")
    parser.add_argument("--mock", action="store_true", help="No model calls (harness test).")
    parser.add_argument("--config", type=Path, help="Extra config TOML.")
    parser.add_argument("--precision-file", type=Path, default=PRECISION_FILE)
    args = parser.parse_args(argv)

    config: Config = load_config(args.config)
    checks = checks_to_run(args.check)
    examples = {k: load_examples(k) for k in checks}
    missing = [k for k, ex in examples.items() if not ex]
    if missing:
        print(f"No labeled examples for: {', '.join(missing)}", file=sys.stderr)

    client: LLMClient
    if args.mock:
        client = OracleLLM({ex["window"]: ex["label"] for exs in examples.values() for ex in exs})
    else:
        client = make_client(config.llm)
    spend = Spend(prices=config.prices)
    judge = Judge(config, judges={}, client=client, spend=spend)

    scores = []
    print(f"{'check':45} {'n':>3} {'prec':>5} {'recall':>6}  status")
    for key, prompt in checks.items():
        if not examples[key]:
            continue
        score = evaluate(key, prompt, examples[key], judge)
        scores.append(score)
        ok = score.precision is not None and score.precision >= config.judge.min_precision
        status = "enabled" if ok else "disabled by default"
        if key.startswith("confirm."):
            status = "confirmation (always on)"
        print(
            f"{key:45} {score.n:>3} {_pct(score.precision)} {_pct(score.recall):>6}  {status}"
            + (f"  ({score.errors} errors)" if score.errors else "")
        )
    if not args.mock:
        print(f"\nCost: ${spend.cost:.2f} over {spend.calls} calls")

    if args.write:
        existing = (
            json.loads(args.precision_file.read_text()) if args.precision_file.is_file() else {}
        )
        now = datetime.now(timezone.utc).isoformat()
        for s in scores:
            existing[s.key] = {
                **asdict(s),
                "precision": s.precision if s.precision is not None else 0.0,
                "recall": s.recall,
                "triage_model": config.llm.triage_model,
                "confirm_model": config.llm.confirm_model,
                "mock": args.mock,
                "measured_at": now,
            }
        args.precision_file.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.precision_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
