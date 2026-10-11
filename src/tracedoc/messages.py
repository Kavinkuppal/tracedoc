"""Turn the many shapes of chat messages and payloads into plain JSON.

Traces arrive with messages as live LangChain objects (capture), as
`dumpd` constructor dicts (LangSmith), as `{"type", "data"}` dicts (older
exports), or as OpenAI-style `{"role", "content"}` dicts. Everything
downstream sees one shape:

    {"type": "human"|"ai"|"system"|"tool"|..., "content": ..., "name"?: str,
     "tool_calls"?: [{"name", "args", "id"}], "tool_call_id"?: str, "status"?: str}
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any
from uuid import UUID

from langchain_core.messages import BaseMessage

_ROLE_TO_TYPE = {"user": "human", "assistant": "ai", "system": "system", "tool": "tool"}
_CLASS_TO_TYPE = {
    "HumanMessage": "human",
    "AIMessage": "ai",
    "AIMessageChunk": "ai",
    "SystemMessage": "system",
    "ToolMessage": "tool",
    "FunctionMessage": "function",
    "ChatMessage": "chat",
}


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... [truncated {len(text) - max_chars} chars]"


def _clean_tool_calls(calls: Any) -> list[dict[str, Any]]:
    cleaned = []
    for call in calls or []:
        if not isinstance(call, Mapping):
            continue
        args = call.get("args")
        if args is None and isinstance(call.get("function"), Mapping):  # OpenAI style
            fn = call["function"]
            call = {"name": fn.get("name"), "id": call.get("id")}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = fn.get("arguments")
        cleaned.append({"name": call.get("name"), "args": args, "id": call.get("id")})
    return cleaned


def _from_fields(msg_type: str, fields: Mapping[str, Any], max_chars: int | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "type": msg_type,
        "content": to_jsonable(fields.get("content", ""), max_chars),
    }
    if fields.get("name"):
        out["name"] = fields["name"]
    calls = _clean_tool_calls(fields.get("tool_calls"))
    if calls:
        out["tool_calls"] = to_jsonable(calls, max_chars)
    if fields.get("tool_call_id"):
        out["tool_call_id"] = fields["tool_call_id"]
    if fields.get("status") and fields.get("status") != "success":
        out["status"] = fields["status"]
    return out


def as_message(obj: Any, max_chars: int | None = None) -> dict[str, Any] | None:
    """Return the canonical message dict for `obj`, or None if it is not a message."""
    if isinstance(obj, BaseMessage):
        return _from_fields(obj.type, obj.__dict__, max_chars)
    if not isinstance(obj, Mapping):
        return None
    # langchain dumpd: {"lc": 1, "type": "constructor", "id": [..., "AIMessage"], "kwargs": {...}}
    if obj.get("lc") == 1 and obj.get("type") == "constructor" and isinstance(obj.get("id"), list):
        msg_type = _CLASS_TO_TYPE.get(obj["id"][-1])
        if msg_type is None:
            return None
        return _from_fields(msg_type, obj.get("kwargs") or {}, max_chars)
    # message_to_dict: {"type": "ai", "data": {...}}
    if isinstance(obj.get("data"), Mapping) and obj.get("type") in _CLASS_TO_TYPE.values():
        return _from_fields(obj["type"], obj["data"], max_chars)
    # already canonical, or LangChain-style {"type": "ai", "content": ...}
    if obj.get("type") in _CLASS_TO_TYPE.values() and "content" in obj:
        return _from_fields(obj["type"], obj, max_chars)
    # OpenAI style {"role": "user", "content": ...}
    if obj.get("role") in _ROLE_TO_TYPE and "content" in obj:
        return _from_fields(_ROLE_TO_TYPE[obj["role"]], obj, max_chars)
    return None


def to_jsonable(obj: Any, max_chars: int | None = None) -> Any:
    """Convert any payload to JSON-safe data, normalizing messages along the way."""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return _truncate(obj, max_chars)
    message = as_message(obj, max_chars)
    if message is not None:
        return message
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v, max_chars) for v in obj]
    if isinstance(obj, (UUID, Decimal)):
        return str(obj) if isinstance(obj, UUID) else float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return to_jsonable(obj.value, max_chars)
    if isinstance(obj, BaseException):
        return f"{type(obj).__name__}: {obj}"
    if isinstance(obj, bytes):
        return f"<{len(obj)} bytes>"
    if hasattr(obj, "model_dump"):
        try:
            return to_jsonable(obj.model_dump(), max_chars)
        except Exception:
            pass
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj), max_chars)
    return _truncate(repr(obj), max_chars)


def message_text(message: Mapping[str, Any]) -> str:
    """Plain text of a canonical message: content plus any tool calls."""
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, Mapping):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        text = " ".join(p for p in parts if p)
    else:
        text = str(content or "")
    for call in message.get("tool_calls") or []:
        text += f" [tool_call {call.get('name')}({json.dumps(call.get('args'), default=str)})]"
    return text.strip()


def reply_text(output: Any) -> str:
    """The words of a model reply, without its tool calls ("" for a tool-call-only reply)."""
    if isinstance(output, Mapping) and "content" in output:
        return message_text({"content": output.get("content")})
    return output.strip() if isinstance(output, str) else ""


def payload_text(payload: Any) -> str:
    """A readable string for any normalized payload, for previews and overlap checks."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping) and "type" in payload and "content" in payload:
        return message_text(payload)
    return json.dumps(payload, default=str, ensure_ascii=False)
