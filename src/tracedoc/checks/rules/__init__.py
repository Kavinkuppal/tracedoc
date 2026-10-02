"""Rule-based checks. Importing this package registers them."""

from . import latency, loop, no_stop, token_hotspot, tool_errors, unchecked_result, unused_output

__all__ = [
    "latency",
    "loop",
    "no_stop",
    "token_hotspot",
    "tool_errors",
    "unchecked_result",
    "unused_output",
]
