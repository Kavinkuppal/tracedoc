# tracedoc

Reads traces from LangGraph agents and produces a diagnosis report: what went right, what went wrong, and the three changes most likely to improve results. Runs locally with your own model key.

Status: pre-alpha. See `SPEC.md` for the build plan.

## Capture traces

```python
from tracedoc import TraceCapture

app.invoke(inputs, config={"callbacks": [TraceCapture("./runs")]})
```

Or import from LangSmith: an export file (`--source langsmith PATH`) or a project (`--source langsmith --project NAME`, needs `pip install tracedoc[langsmith]`).

## Analyze

```
tracedoc estimate ./runs              # what judging would cost; no model calls
tracedoc analyze ./runs --max-cost 2  # asks before spending; writes tracedoc-report/report.json
tracedoc analyze ./runs --no-judge    # rule checks only, no model calls
```

Rule checks run without a model. Judge checks use your key (`ANTHROPIC_API_KEY` by default) in two stages: a cheap triage model, then a stronger confirmation model on what triage flags.

Judge checks start disabled until their precision is measured on the labeled examples in `evals/labels/`:

```
python evals/run_evals.py --write     # spends money; records precision per check
```

Checks at or above 80% precision are then enabled. `--include-unvalidated` runs all of them anyway.

## Configuration

`config.toml` in the user config directory (`tracedoc` under your platform's config dir), or `--config FILE`:

```toml
[llm]
provider = "anthropic"            # anthropic | openai | openai_compatible
triage_model = "claude-haiku-4-5"
confirm_model = "claude-opus-5"
# base_url = "http://localhost:11434/v1"   # for openai_compatible

[prices."my-model"]               # USD per million tokens; extends the built-in table
input = 0.5
output = 1.5

[thresholds]
loop_min_repeats = 3

[redact]
fields = ["api_key", "password", "customer_name"]
patterns = ['ACME-\d+']
```

Nothing leaves your machine except calls to the model you configure.

## Development

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
python -m tests.fixtures.generate_traces   # regenerate fixture traces
pytest -m live --live -s                   # real-model acceptance test (spends money)
```
