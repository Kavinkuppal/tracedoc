# tracedoc

Reads traces from LangGraph agents and produces a diagnosis report: what went right, what went wrong, and the three changes most likely to improve results. Runs locally with your own model key.

Status: pre-alpha. See `SPEC.md` for the build plan.

## Development

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```
