# Contributing to tracedoc

Thanks for helping. Bug reports, false findings and pull requests are all welcome.

## Set up

You need Python 3.10 or later and git.

```bash
git clone https://github.com/Kavinkuppal/tracedoc.git
cd tracedoc
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run the tests

These are the checks CI runs on every push and pull request:

```bash
ruff check .
ruff format --check .
pytest
python evals/run_evals.py --mock
```

- The tests never use the network: every model call is faked (`tests/fakes.py`), and the
  fixture traces in `tests/fixtures/traces/` come from small scripted agents in
  `tests/fixtures/agents/`. A test that needs the network is a bug.
- `python -m tests.fixtures.generate_traces` regenerates the fixture traces from those agents.
- The HTML report has browser tests:
  `pip install -e ".[dev,browser-tests]"`, `python -m playwright install chromium`, then
  `pytest -m browser --browser tests/test_browser.py`.
- `pytest -m live --live` calls a real model with your own key (`ANTHROPIC_API_KEY`) and
  spends a little money. Nothing else in the suite does.
- The data-sharing server in `server/` has its own tests, which need Postgres (`initdb` and
  `pg_ctl` on your PATH): `cd server && pip install -e .. -e ".[dev]" && pytest`.

## Where things live

- `src/tracedoc/checks/rules/`: the rule checks, which need no model.
- `src/tracedoc/checks/judges/`: the AI judge, with one versioned prompt file per check in
  `prompts/`.
- `src/tracedoc/checks/pitfalls.py`: known framework pitfalls, one entry each, with the
  documentation each fix cites.
- `src/tracedoc/catalog.toml`: every sentence a report shows. Report text is written and
  reviewed there, not in code.
- `src/tracedoc/report/`: the PDF reports, `report.json`, and the optional HTML report.
- `evals/`: labeled examples for each AI judge check, and `run_evals.py`, which measures a
  check's precision on them.

## Changing what tracedoc finds

- A new or changed AI judge prompt gets a new version number and labeled examples in
  `evals/labels/`. It stays off by default until its precision is measured:
  `python evals/run_evals.py --check <check id> --write` (this calls a real model with your
  key).
- Every suggested fix must pass the safety rule in `src/tracedoc/checks/safety.py`: no fix
  may tell an agent to book, buy, pay, send, cancel or delete anything without asking the
  user first. The tests check every fix template in the catalog.
- If you build a test case from a real trace, rewrite it with made-up names and values first.

## Submitting changes

1. Fork the repository and create a branch.
2. Make small, focused commits with clear messages.
3. Run the checks above.
4. Open a pull request that says what changed and why. CI must pass before it is merged.

Never put API keys, real trace data or personal information in an issue, a pull request, a
fixture or a test. Report security problems privately, as described in
[SECURITY.md](SECURITY.md).

By contributing, you agree that your contribution is licensed under the project's MIT license
([LICENSE](LICENSE)).
