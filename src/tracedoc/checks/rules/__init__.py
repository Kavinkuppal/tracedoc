"""Rule-based checks. Importing this package registers them."""

from . import (
    empty_answer,
    latency,
    loop,
    no_stop,
    run_error,
    token_hotspot,
    tool_errors,
    unchecked_result,
    unused_output,
)

__all__ = [
    "empty_answer",
    "latency",
    "loop",
    "no_stop",
    "run_error",
    "token_hotspot",
    "tool_errors",
    "unchecked_result",
    "unused_output",
]
