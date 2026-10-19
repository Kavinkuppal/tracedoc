# Third-party notices

tracedoc bundles two third-party components. Everything else it uses is installed by pip as a
dependency and keeps its own license.

## Bundled

### Inter (fonts)

`src/tracedoc/report/fonts/Inter-Regular.ttf` and `Inter-SemiBold.ttf` are the static
TrueType fonts of Inter, version 4.1 (`src/tracedoc/report/fonts/INTER_VERSION`), from
https://github.com/rsms/inter. Copyright (c) 2016 The Inter Project Authors. They are embedded
(subset) in every PDF report.

- License: SIL Open Font License 1.1, full text in `src/tracedoc/report/fonts/Inter-OFL.txt`.

### Apache ECharts (JavaScript)

`src/tracedoc/report/vendor/echarts.min.js` is Apache ECharts, version 6.1.0
(`src/tracedoc/report/vendor/ECHARTS_VERSION`), inlined into the optional HTML report
(`tracedoc analyze --html`).

- License: Apache License 2.0, full text in `src/tracedoc/report/vendor/ECHARTS_LICENSE.txt`.
- Notice (`src/tracedoc/report/vendor/ECHARTS_NOTICE.txt`): Apache ECharts, Copyright
  2017-2026 The Apache Software Foundation. This product includes software developed at The
  Apache Software Foundation (https://www.apache.org/).
- ECharts includes parts of d3.js under the BSD 3-Clause license:
  `src/tracedoc/report/vendor/ECHARTS_LICENSE-d3.txt`.

HTML reports credit Apache ECharts and its license in their footer.

## In the repository, not in the package

### LangChain's supervisor example (demo traces)

`examples/langgraph-supervisor/` holds traces of, and reports on, the Quickstart example of
langgraph-supervisor-py (https://github.com/langchain-ai/langgraph-supervisor-py). The traces
contain the example's prompts and the output of its fake tools. Copyright (c) 2025 LangChain,
Inc.

- License: MIT, full text in `examples/langgraph-supervisor/LICENSE-langgraph-supervisor-py.txt`.

## Dependencies (installed by pip, not bundled)

| Package | Used for | License |
|---|---|---|
| typer | the command line | MIT |
| pydantic | the data model and config validation | MIT |
| jinja2 | the HTML report template | BSD |
| platformdirs | where config and cache files live | MIT |
| anthropic | Anthropic models for the AI judge | MIT |
| openai | OpenAI and OpenAI-compatible models for the AI judge | Apache 2.0 |
| langchain-core | `TraceCapture`, the LangChain callback that records traces | MIT |
| reportlab | the PDF reports | BSD |
| tomli (Python 3.10 only) | reading TOML config | MIT |
| langsmith (optional extra) | reading traces from LangSmith | MIT |

Licenses as declared in each package's metadata; the package's own license file is
authoritative.
