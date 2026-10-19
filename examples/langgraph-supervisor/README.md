# Demo: LangChain's supervisor example

Twenty traces of LangChain's own multi-agent supervisor example, and the report tracedoc wrote
about them.

## The system

This is the Quickstart example from the README of
[langchain-ai/langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py)
(commit `88859b3`), used under its MIT license (Copyright (c) 2025 LangChain, Inc.; full text
in [LICENSE-langgraph-supervisor-py.txt](LICENSE-langgraph-supervisor-py.txt)). A supervisor
passes each request to one of two agents:

- `research_expert` looks things up with a `web_search` tool;
- `math_expert` calculates with `add` and `multiply` tools.

The tools are the example's own fakes: `web_search` always returns the same short list of
company headcounts, whatever it is asked. The only change from LangChain's code is the model:
Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) instead of `gpt-4o`. It was run on 20 requests,
and each run was recorded with tracedoc's `TraceCapture`.

## What tracedoc found

In 16 of the 20 runs the supervisor's final answer was blank, even though an expert had
already answered: correctly, or saying plainly when the search didn't have the figure. tracedoc traced this to one setting in langgraph-supervisor:
with `include_agent_name` left at its default, the library marks which agent wrote each message
only in the message's `name` attribute, and, as the library's documentation says, only OpenAI's
API passes that attribute to the model. On Claude, the supervisor read the expert's answer as
if it had written it itself, and ended with nothing to add.

The fix tracedoc suggests is one argument:

```python
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt="...",
    include_agent_name="inline",  # each agent's name goes into the message text
)
```

Running the same 20 requests again with that change took blank answers from 16 of 20 to 0 of
20.

The report also shows that `research_expert` did the arithmetic itself in 9 runs, although its
instructions say "Do not do any math".

## The files

- [`summary.pdf`](summary.pdf): the report. Page 1 says what to fix; page 2 shows where and how
  often.
- [`details.pdf`](details.pdf): the evidence for every finding, an index of every run, and the
  catalog of checks.
- [`traces/`](traces/): the 20 traces, in tracedoc's capture format.

## Run it yourself

With no API key, the rule checks alone find the `include_agent_name` problem in all 20 runs:

```bash
tracedoc analyze examples/langgraph-supervisor/traces --no-judge --out supervisor-demo
```

The research expert's math was found by the AI judge, which needs your own model key (see the
main [README](../../README.md)). These reports were made with it, using Claude Haiku 4.5 for
triage and Claude Sonnet 5.5 for confirmation, for $1.13. Model answers vary a little from run
to run, so your findings and cost can differ slightly.
