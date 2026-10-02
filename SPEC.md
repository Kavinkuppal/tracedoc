# tracedoc: build spec for the MVP

You are building an open-source Python tool that reads traces from LangGraph agents and produces a diagnosis report: what went right, what went wrong, and the three changes most likely to improve results. It runs locally, uses the developer's own model key, and optionally shares anonymized statistics with explicit consent.

`tracedoc` is a working name. Keep the package name in one place so it is easy to rename.

Work through the phases in order. Finish each phase's acceptance criteria, with tests passing, before starting the next. Ask before making architectural decisions not covered here.

---

## Ground rules

- **Verify APIs before using them.** LangChain, LangGraph, LangSmith, Anthropic, and OpenAI SDKs change often. Check the installed version's docs or source before writing integration code. Don't rely on memory for method names or event signatures.
- **No network in tests.** Mock every model call. Use LangChain's fake chat models to generate deterministic fixture traces.
- **Nothing leaves the machine** except (a) calls to the model the developer configured and (b) opt-in statistics after consent. Treat any other network call as a bug.
- **Small, reviewable commits**, one logical change each, with a clear message.
- **Type hints everywhere.** Run `ruff` and `pytest` before every commit.
- **Prefer boring choices.** Standard library and well-known packages over clever abstractions.

## Stack

- Python 3.10+, packaged with `pyproject.toml` (hatchling), installable with `pip install tracedoc`
- CLI: Typer
- Schema and validation: Pydantic v2
- Model clients: `anthropic` and `openai` SDKs; local models through any OpenAI-compatible endpoint
- Report: Jinja2 templates producing one self-contained HTML file, with Apache ECharts vendored into the package (check its license and keep the notice). ECharts covers bar, line, heatmap, graph, and box plot in one library.
- Config and state: `platformdirs` for the user config directory, TOML for config files
- Tests: pytest. Lint: ruff.
- Data-sharing server (Phase 5): FastAPI and Postgres, in a separate `server/` directory with its own `pyproject.toml`

## Repository layout

```
tracedoc/
  pyproject.toml
  README.md
  LICENSE                      # TODO: MIT or Apache 2.0, founders to decide
  src/tracedoc/
    cli.py                     # Typer app: analyze, compare, estimate, share
    config.py                  # config loading, defaults, thresholds, price table
    capture/
      callback.py              # TraceCapture: LangChain callback handler -> JSONL
    ingest/
      jsonl.py                 # read our own capture format
      langsmith.py             # import from a LangSmith project or export file
      otel.py                  # P1: OpenTelemetry GenAI spans
    schema.py                  # Run, Step, Finding, Group, Report models
    normalize.py               # all sources -> schema; redaction; role mapping; task typing
    redact.py
    classify/
      task_type.py             # fixed task-type list
      roles.py                 # fixed node-role list
    checks/
      base.py                  # Check interface and registry
      rules/                   # rule-based checks, no model calls
      judges/                  # judge checks and their prompt files
      signals.py               # good-behavior signals
    llm/
      client.py                # provider-agnostic wrapper, structured JSON output
      cost.py                  # token estimates and price table
    analyze.py                 # orchestrates rules -> judges -> grouping -> ranking
    group.py
    compare.py                 # P1
    report/
      html.py
      markdown.py
      json_out.py
      templates/
      vendor/echarts.min.js
    share/
      consent.py
      payload.py
      upload.py
  tests/
    fixtures/
      agents/                  # small LangGraph agents with deliberate bugs
      traces/                  # generated JSONL traces checked into the repo
    ...
  evals/
    labels/                    # hand-labeled examples per judge check
    run_evals.py               # measures precision per check
  server/                      # Phase 5
```

---

## Core data model

Define these in `schema.py` as Pydantic models. Every source normalizes into them.

**Run:** `id`, `source` (capture, langsmith, otel), `started_at`, `ended_at`, `status` (completed, error, recursion_limit, interrupted), `input_preview`, `output_preview`, `task_type`, `steps: list[Step]`, `totals` (tokens in and out, cost, duration, step count), `framework`, `framework_version`.

**Step:** `id`, `parent_id`, `run_id`, `kind` (llm, tool, node, handoff), `name` (as written), `role` (mapped from the fixed list), `node` (LangGraph node name if known), `started_at`, `ended_at`, `input`, `output`, `model`, `tokens_in`, `tokens_out`, `cost`, `error`.

**Finding:** `check_id`, `group` (design, agents, verification, tools_cost), `method` (rule or judge), `severity` (high, medium, low), `run_id`, `evidence_step_ids`, `evidence_excerpts` (short), `explanation`, `suggested_fix`, `confidence` (0 to 1), `node`, `role`.

**Group:** findings across runs that share a signature (`check_id`, node or role, and a short key such as the tool name). Holds `runs_affected`, `frequency`, `impact` (tokens, cost, latency attributed), and a representative finding.

**Report:** scorecard, ranked groups, signals, per-node stats, distributions, and metadata (tool version, analysis model, analysis cost).

Handoffs: infer a `handoff` step whenever control passes from one agent node to another. Record what the sender's output contained and what the receiver's input contained.

---

## Phase 0: scaffold and fixtures

1. Create the repo layout, `pyproject.toml`, ruff config, pytest config, and a GitHub Actions workflow that runs lint and tests.
2. Build five small LangGraph agents in `tests/fixtures/agents/`, each with one deliberate bug, using fake chat models so they run without API keys and produce the same trace every time:
   - a researcher that repeats the same search
   - a two-agent pipeline where the writer drops the analyst's caveat
   - a supervisor that never routes to the end and hits the recursion limit
   - an agent whose tool fails and gets retried with the same arguments
   - an agent that calls a tool and never uses the result
3. Add one healthy agent that should produce no high-severity findings.

**Done when:** `pytest` passes in CI, and each fixture agent runs in under a second without network access.

## Phase 1: capture, import, normalize

1. `TraceCapture(path)`: a LangChain callback handler that records chain, chat model, LLM, and tool start, end, and error events, with run IDs and parent run IDs, to one JSONL file per run. Capture LangGraph node names and step numbers from the callback metadata (verify the current metadata keys in the installed LangGraph version). Capture token usage from the response's usage metadata.
2. Generate fixture traces from Phase 0 agents and commit them to `tests/fixtures/traces/`.
3. LangSmith import: pull runs for a project (or read an exported file) using the LangSmith SDK, including all child runs of each trace.
4. `normalize.py`: convert both sources to the schema, infer handoffs, compute totals.
5. Redaction: config-driven field names and regex patterns, applied before any data reaches a model.
6. Role mapping: map node names to the fixed list (planner, researcher, writer, critic, supervisor, tool executor, other) using name heuristics first, with optional model fallback.
7. Task typing: classify each run into the fixed list (research, coding, data extraction, customer support, planning, writing, other). Use heuristics first, then the developer's configured model on a truncated input preview. This runs on the developer's side; nothing goes to our server.

**Done when:** every fixture trace normalizes correctly with tests asserting step counts, parent links, handoffs, tokens, and statuses; redaction tests prove configured fields never appear downstream.

## Phase 2: rule checks

Implement a `Check` interface (`id`, `group`, `method`, `default_severity`, `run(runs) -> list[Finding]`) and a registry. All thresholds live in config with sensible defaults.

| Check ID | What it detects | Default heuristic |
| --- | --- | --- |
| `design.loop` | Repeated steps | Same tool with identical normalized arguments 3+ times in a run, or a node cycle repeated without state change. Near-duplicates (high string similarity) are passed to the judge. |
| `design.no_stop` | Never reaches a clear end | Run status is recursion_limit, or step count hits the configured limit |
| `tools.errors_retries` | Tool failures | Tool error events; retries with identical arguments after an error |
| `tools.unused_output` | Tool output ignored | Little or no overlap between a tool's output and any later model input. Candidates go to the judge. |
| `verification.unchecked_result` | No checking before finishing | Final output produced with no critic or verifier step after the last tool output. Candidates go to the judge. |
| `cost.token_hotspot` | Token concentration and context growth | A node's share of tokens above threshold; input tokens per step growing steeply across a run |
| `cost.latency_outlier` | Slow steps | Step latency far above the batch's p95 for that node, or above an absolute threshold |

Good-behavior signals in `signals.py`: `good.error_recovery` (a tool error followed by a different, successful approach), `good.verified` (a verification step before the end), `good.efficient_path` (steps well below the batch median for the same task type), `good.clean_handoff` (handoffs carrying the sender's key fields).

**Done when:** each buggy fixture triggers its intended check, the healthy fixture triggers no high-severity rule findings, and rules on 100 runs finish in under 10 seconds.

## Phase 3: judge checks, grouping, cost

1. `llm/client.py`: one interface over Anthropic, OpenAI, and OpenAI-compatible endpoints, returning validated JSON matching a Pydantic model. Retries with backoff. Reads the key from the environment or config.
2. Judge checks, each with a versioned prompt file and labeled examples:

| Check ID | What it detects |
| --- | --- |
| `design.ignored_instructions` | Ignores task or role instructions |
| `design.lost_context` | Loses earlier context or decisions |
| `agents.incomplete_handoff` | Handoff missing information the next agent needed |
| `agents.ignored_output` | Ignores another agent's output |
| `agents.reason_action_mismatch` | Reasons toward one action, takes another |
| `agents.no_clarification` | Guesses instead of asking on ambiguous input |
| `agents.task_drift` | Drifts off the assigned task |
| `verification.premature_finish` | Finishes before the task is done |

   Plus judge confirmation for the rule candidates flagged in Phase 2.
3. Windows, not whole traces: a judge sees only the steps around a candidate (configurable, default a few steps either side) plus the relevant system prompt. Candidates come from rule flags, every handoff, and a capped random sample of runs.
4. Two-stage judging: a cheap triage model answers yes or no with confidence; a stronger model confirms and returns `{verdict, confidence, evidence_step_ids, explanation, suggested_fix}`. Model names come from config. TODO: founders pick defaults.
5. `llm/cost.py`: estimate tokens per window times number of windows times price. Keep the price table in config, user-overridable, because prices change. `estimate` prints the number before any spend; `analyze` asks to proceed and enforces `--max-cost`.
6. Grouping and ranking: group findings by signature; rank by severity, then frequency, then impact. Pick the top three for the report.
7. `evals/run_evals.py`: for each judge check, run it on its labeled examples and report precision and recall. A check below 80% precision is disabled by default, with a flag to turn it on.

**Done when:** the eval harness runs in CI with mocked responses; a real run against the fixtures (with a key) finds the intended issues; analyzing 50 fixture runs costs under $2 at configured prices.

## Phase 4: the report

One self-contained HTML file that works offline: data embedded as JSON in a script tag, ECharts and all CSS inlined, filters in plain JavaScript. Target under 5 MB for 100 runs. Every chart filters by task type, node, and severity.

| # | Section | Visual |
| --- | --- | --- |
| 1 | Scorecard | Tiles: completion rate, median steps, cost, latency, findings by severity |
| 2 | Top three fixes | Cards with evidence links and the suggested fix |
| 3 | Failure frequency | Bar chart of checks sorted by runs affected, colored by severity |
| 4 | Failures by task type | Heatmap: task type by check |
| 5 | How runs end | Stacked bar of run status per task type |
| 6 | Agent flow map | Graph of nodes and edges; edge width shows traffic, node color shows failure rate |
| 7 | Handoffs | Matrix: sending agent by receiving agent, with counts and failure share |
| 8 | Cost by node | Bar chart of token and cost share per node |
| 9 | Context growth | Line chart of input tokens per step, median and worst runs |
| 10 | Steps per run | Histogram of step counts |
| 11 | Latency by node | Box plot per node |
| 12 | Run replay | Timeline of one run's steps, flagged steps highlighted, expandable inputs and outputs |
| 13 | What went well | Signals with evidence |
| 14 | All findings | Searchable, filterable table |

Clicking any finding opens its evidence in the run replay. Each finding has thumbs up and down buttons; since a static file can't write to disk, show a short command to record the vote (for example `tracedoc feedback <finding-id> up`), or provide an export button that downloads a feedback file the CLI can import.

Also output:
- `report.md`: scorecard, top three fixes, and a link to the HTML file
- `report.json`: the full Report model, for scripting and CI

**Done when:** a report generated from the fixtures renders correctly offline in Chrome, Firefox, and Safari; every section has a snapshot test on its data; the Markdown and JSON outputs validate.

## Phase 5: consent and data sharing

**Client side**

1. After the first successful report, show a plain-language summary of the data-sharing terms and ask yes or no. Enter means no. Store the choice, the terms version, and a timestamp in the user config directory. Ask again whenever the terms version changes.
2. Generate a random install ID (UUID4) on first run. It must not be derived from anything identifying.
3. `tracedoc share --preview` prints the exact payload that would be sent. `tracedoc share --off` revokes. `tracedoc share --delete` requests deletion of everything tied to the install ID.
4. Log every upload locally with its payload.

**Payload: allowed fields only**

Per run: schema version, install ID, tool version, task type, framework and version, model per role, graph shape (agent count, tool count, max depth), run status, step count, tokens, cost, duration, and findings as (check ID, severity, role). Per batch: feedback votes on findings, and compare-mode deltas when present.

**Never in the payload:** prompts, inputs, outputs, tool arguments or results, code, node or tool names as written, file paths, keys, or anything identifying the developer or their users.

Enforce this with tests, not just review: build the payload with a strict allow-list model, and add a test that feeds traces containing unique marker strings in every content field and asserts none of them appear anywhere in the serialized payload.

**Server side** (`server/`)

- `POST /v1/contributions` accepts versioned payloads and validates them against the same allow-list schema
- `DELETE /v1/installs/{install_id}` removes all of that install's data
- Postgres storage, basic rate limiting, HTTPS only
- Do not log or store IP addresses; configure the web server and app logging accordingly

**Blocked on:** legal review of the terms and privacy policy. Build the flow behind a feature flag; ship it on only after the terms are approved.

**Done when:** the marker-string test passes, preview output matches what is sent byte for byte, revoke and delete work end to end against a local server.

## Phase 6: P1 and P2 items, after the above ships

- **P1: compare mode.** `tracedoc compare ./before ./after` matches groups by signature and reports improved, regressed, new, and resolved, with metric deltas, plus a 15th report section of paired bars.
- **P1: OpenTelemetry ingestion.** Read GenAI semantic-convention spans into the same schema.
- **P2: GitHub Action** that runs compare on a pull request and posts the summary as a comment.

---

## CLI reference

```
tracedoc analyze PATH [--source capture|langsmith] [--project NAME] [--model ...] [--max-cost USD] [--out DIR]
tracedoc estimate PATH
tracedoc compare BEFORE AFTER [--out DIR]
tracedoc share [--preview | --on | --off | --delete]
tracedoc feedback FINDING_ID up|down
```

Capture in the developer's app:

```python
from tracedoc import TraceCapture

app.invoke(inputs, config={"callbacks": [TraceCapture("./runs")]})
```

## Launch checklist

- [ ] All P0 phases done, CI green
- [ ] Rule checks plus at least five judge checks at 80%+ precision on labeled sets
- [ ] Install to first report under 10 minutes on a clean machine, timed
- [ ] README with a 60-second quickstart and a screenshot of the report
- [ ] Data terms approved by a lawyer before the sharing flag is turned on
- [ ] License file and third-party notices (including ECharts)

## Open decisions for the founders

- Final product name
- License: MIT or Apache 2.0
- Default triage and confirmation models
- Whether local models are supported at launch or right after
