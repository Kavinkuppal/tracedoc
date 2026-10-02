# tracedoc

tracedoc reads traces from your LangGraph agents and writes a diagnosis report: what went wrong (loops, runs that never finish, failing tools, ignored tool results, slow or expensive steps), what went right, and the three fixes most likely to help. It runs on your machine. The optional model-based checks use **your own** model API key, so their cost is billed to your account.

> Status: pre-release. `tracedoc` is a working name.

## Requirements

- Python 3.10 or newer
- A LangGraph (or LangChain) app to trace

## Install

tracedoc is not on PyPI yet. Install from a clone of this repository:

```bash
git clone <this repository's URL>
cd tracedoc
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install .                      # add LangSmith import support: pip install ".[langsmith]"
```

## Set up your API key

tracedoc never ships with, defaults to, or falls back on anyone else's key. It reads keys **only from environment variables**:

| You use | Set |
| --- | --- |
| Anthropic (default) | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| LangSmith import (`--project`) only | `LANGSMITH_API_KEY` |

macOS / Linux:

```bash
export ANTHROPIC_API_KEY="your-key"      # or OPENAI_API_KEY
export LANGSMITH_API_KEY="your-key"      # only for importing from a LangSmith project
```

Windows (PowerShell):

```powershell
$env:ANTHROPIC_API_KEY = "your-key"      # this session only
setx ANTHROPIC_API_KEY "your-key"        # future sessions
```

To use OpenAI instead of Anthropic, set the provider and models in your config file (see [Configuration](#configuration)):

```toml
[llm]
provider = "openai"
triage_model = "your-cheap-model"
confirm_model = "your-stronger-model"
```

Then add their prices under `[prices]`. Only Anthropic models come with built-in prices.

**Local models.** Any OpenAI-compatible endpoint works, for example Ollama or vLLM:

```toml
[llm]
provider = "openai_compatible"
base_url = "http://localhost:11434/v1"
triage_model = "llama3.1:8b"
confirm_model = "llama3.1:70b"
# api_key_env = "MY_LOCAL_KEY"   # only if your endpoint needs a key: the NAME of an env variable

[prices."llama3.1:8b"]            # USD per million tokens; 0 for local models
input = 0
output = 0
[prices."llama3.1:70b"]
input = 0
output = 0
```

A local endpoint receives no key unless you set `api_key_env`. tracedoc never sends your `OPENAI_API_KEY` there.

**Costs.** Analysis with a model is billed to **your** provider account. Run `tracedoc estimate` first. `tracedoc analyze` shows the estimate and asks before spending anything, and `--max-cost` stops judging before spend passes a limit.

**Never commit keys to a repository.** Keep them in environment variables or your OS keychain, not in code, config files, or `.env` files that get committed. tracedoc refuses a key placed in its config file.

## Capture runs

Add the callback to any LangGraph or LangChain invocation. It writes one JSONL file per run:

```python
from tracedoc import TraceCapture

app.invoke(inputs, config={"callbacks": [TraceCapture("./runs")]})
```

It records chain, model, and tool events with LangGraph node names, step numbers, and token usage, to files on your disk. Long strings are cut at 20,000 characters.

Capture files hold your agents' raw inputs and outputs: redaction happens when tracedoc analyzes them, not when it captures. Treat them like logs, and keep them out of version control.

## Import from LangSmith

From a LangSmith project (needs `pip install ".[langsmith]"` and `LANGSMITH_API_KEY`):

```bash
tracedoc analyze --source langsmith --project my-project
```

This pulls up to 100 traces from the project, with all of their child runs. From an exported file, no LangSmith key is needed:

```bash
tracedoc analyze runs.json --source langsmith
```

The file can be a JSON list of runs, `{"runs": [...]}`, or JSONL with one run per line. It must include every run of each trace, not only the root runs.

## Run an analysis

```bash
tracedoc estimate ./runs              # what analysis would cost; no model calls, no key needed
tracedoc analyze ./runs               # shows the estimate, asks, then analyzes
tracedoc analyze ./runs --max-cost 2  # stop judging before $2 of spend
tracedoc analyze ./runs --no-judge    # rule checks only: no model, no key, no cost
```

Reports go to `./tracedoc-report/` (change it with `--out`):

- `report.html` is a single file that works offline. Open it in any browser. Filter by task type, node, and severity; click any finding's evidence to open it in the run replay; and vote on whether findings were useful.
- `report.md` holds the scorecard and top three fixes.
- `report.json` holds the full results, for scripts and CI.

Reports include excerpts of your traces (after redaction), so share them only with people who may see that data.

**What the analysis does.** Rule checks always run, with no model: repeated steps, runs that never finish, tool errors and blind retries, ignored tool output, no verification before finishing, token hotspots, and slow steps. With a key, a cheap triage model and a stronger confirmation model (default `claude-haiku-4-5-20251001` and `claude-sonnet-5-5`) confirm or dismiss the uncertain rule findings. They see only a few steps around each finding.

Eight further judge checks (ignored instructions, lost context, incomplete handoffs, ignored agent output, reasoning/action mismatch, guessing on ambiguous input, task drift, finishing early) stay **off** until their precision is measured on labeled examples. None have been measured yet. Turn them on anyway with `--include-unvalidated`, or measure them from a clone of this repository with `python evals/run_evals.py --write`, which uses your key.

**Feedback on findings.** Each finding in the HTML report has thumbs up/down buttons. The page shows the command that records your vote, or exports all of your votes to a file:

```bash
tracedoc feedback <finding-id> up
tracedoc feedback --import tracedoc-feedback.json
```

Votes are stored locally (`tracedoc paths` shows where).

## What leaves your machine

- **Model calls, to the provider you configured, with your key.** Only when you run `analyze` without `--no-judge`. They contain short windows of trace content: the user's request, the agents' system prompts, and a few steps around each finding. They can also contain an input preview of up to 500 characters and node names, used to classify runs heuristics can't place. Redaction (below) runs before anything is sent.
- **LangSmith requests**, only when you import with `--project`. These download your runs; no trace content is uploaded.
- **Nothing else** by default. There are no analytics or telemetry, and captures and reports stay on your disk.

### Sharing anonymized statistics (opt-in, not yet enabled)

tracedoc has an opt-in way to share anonymized statistics that would help improve its checks. **It is not yet enabled in this version**: its terms are awaiting legal review, it is switched off, and no destination server is configured. Until it is enabled, tracedoc never asks and never sends anything.

When it is enabled, it works like this:

- **Nothing is sent without your consent.** After your first report, tracedoc shows a plain-language summary of the terms and asks once. Pressing Enter means no. It asks again only if the terms change.
- **What would be sent:** for each run, the task type, the framework and its version, the public model names used per role (custom or fine-tuned model names become `custom`), counts describing the agent graph, the run's status, step count, tokens, cost and duration, and the IDs, severities and roles of the checks that fired. Your thumbs up/down votes are sent as check IDs only. All of this is tied to a random install ID, not to you.
- **What is never sent:** prompts, inputs, outputs, tool arguments or results, code, node or tool names, file paths, API keys, or anything identifying you or your users. A test feeds traces with unique markers in every one of those fields and checks that none of them reaches the payload.
- **You stay in control:**

```bash
tracedoc share --preview   # print the exact bytes that would be sent for your last analysis
tracedoc share --off       # stop sharing (revoke consent)
tracedoc share --on        # opt in (once sharing is enabled)
tracedoc share --delete    # delete everything shared from this install (once sharing is enabled)
```

`share --preview` works today, so you can see exactly what would be sent. Every upload is logged locally with its payload; `tracedoc paths` shows where.

**Redaction.** Before anything else reads trace content, tracedoc replaces the values of sensitive keys (`api_key`, `password`, `token`, `authorization`, and more) and anything matching common secret patterns (emails, `sk-...` keys, GitHub tokens, AWS key IDs, bearer tokens, card-like numbers). It also scrubs those values wherever else they reappear in the trace. Add your own:

```toml
[redact]
fields = ["api_key", "password", "token", "secret", "authorization", "customer_name"]
patterns = ['ACME-\d+']
```

Setting `fields` or `patterns` replaces the default list, so include the defaults you want to keep.

## Configuration

Settings come from built-in defaults, then your user config file, then any file passed with `--config`. Run `tracedoc paths` to see the location on your machine:

| OS | Config file |
| --- | --- |
| macOS | `~/Library/Application Support/tracedoc/config.toml` |
| Linux | `~/.config/tracedoc/config.toml` (or `$XDG_CONFIG_HOME/tracedoc/config.toml`) |
| Windows | `%LOCALAPPDATA%\tracedoc\config.toml` |

The config file never holds API keys. It can name an environment variable (`api_key_env`), but a key value in the file is rejected.

```toml
[llm]
provider = "anthropic"             # anthropic | openai | openai_compatible
triage_model = "claude-haiku-4-5-20251001"
confirm_model = "claude-sonnet-5-5"

[prices."my-model"]                # USD per million tokens; adds to the built-in table
input = 0.5
output = 1.5

[thresholds]                       # every rule threshold is configurable
loop_min_repeats = 3
step_limit = 25

[checks]
disabled = ["cost.latency_outlier"]
```

## Troubleshooting

**`Error: ANTHROPIC_API_KEY is not set.`** (or `OPENAI_API_KEY`) You ran `analyze` without a key for the configured provider. (`tracedoc estimate` needs no key.) Set it as shown in [Set up your API key](#set-up-your-api-key), or run `tracedoc analyze ./runs --no-judge` for rule checks only.

**`Error: The provider rejected ANTHROPIC_API_KEY (HTTP 401).`** The key is wrong, expired, or lacks access. Analysis stops as soon as a call is rejected, without writing a report.

**`Error: LANGSMITH_API_KEY is not set.`** Importing with `--project` needs a LangSmith key. Alternatively, export the runs from LangSmith and pass the file.

**`Error: ... [llm] api_key is not supported`** Remove the key from the config file and set the environment variable instead.

**`Estimated cost: unknown (model not in price table)`** Add the model under `[prices."model-name"]`. `--max-cost` also needs prices.

**`Judge checks disabled (8)`** This is expected until precision is measured; see [Run an analysis](#run-an-analysis).

**`Error: data sharing is not enabled in this version`** Sharing is not switched on yet (see [What leaves your machine](#what-leaves-your-machine)). `tracedoc share --preview` and `--off` still work.

**`Error: no traces found in ./runs`** Point tracedoc at the folder `TraceCapture` wrote to (`*.jsonl` files), or use `--source langsmith` for LangSmith exports.

## License

TODO: the license (MIT or Apache 2.0) has not been decided yet. Until it is, no license is granted. Third-party components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); reports include Apache ECharts under the Apache License 2.0.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
python -m tests.fixtures.generate_traces          # regenerate fixture traces
UPDATE_SNAPSHOTS=1 pytest tests/test_report.py    # accept report-data snapshot changes
pytest -m browser --browser                       # report in Chromium/Firefox/WebKit (pip install ".[browser-tests]")
pytest -m live --live -s                          # real-model acceptance test; uses your key and spends money
python evals/run_evals.py --mock                  # eval harness without a model
```
