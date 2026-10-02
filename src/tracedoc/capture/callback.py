"""TraceCapture: a LangChain callback handler that writes one JSONL file per run.

Usage:

    app.invoke(inputs, config={"callbacks": [TraceCapture("./runs")]})

Line 1 of each file is a `trace_start` header. Every following line is one
callback event with `run_id`, `parent_run_id`, a UTC timestamp, and event data.
Events are flushed as they happen, so a crashed run still leaves a usable file.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import IO, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ..__about__ import __version__
from ..messages import to_jsonable

FORMAT = "tracedoc-capture"
FORMAT_VERSION = 1
DEFAULT_MAX_CHARS = 20_000

_KEPT_METADATA_PREFIXES = ("langgraph_", "ls_")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str | None:
    if kwargs.get("name"):
        return str(kwargs["name"])
    if serialized:
        if serialized.get("name"):
            return str(serialized["name"])
        if isinstance(serialized.get("id"), list) and serialized["id"]:
            return str(serialized["id"][-1])
    return None


def _metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {k: to_jsonable(v) for k, v in metadata.items() if k.startswith(_KEPT_METADATA_PREFIXES)}


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


class TraceCapture(BaseCallbackHandler):
    """Record chain, chat model, LLM, and tool events to `<path>/<run_id>.jsonl`."""

    raise_error = False

    def __init__(self, path: str | Path, *, max_chars: int = DEFAULT_MAX_CHARS) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_chars = max_chars
        self._lock = threading.Lock()
        self._root_of: dict[UUID, UUID] = {}
        self._files: dict[UUID, IO[str]] = {}
        self.files_written: list[Path] = []

    # ---- bookkeeping -------------------------------------------------------

    def _open_root(self, run_id: UUID) -> None:
        file_path = self.path / f"{run_id}.jsonl"
        handle = file_path.open("w", encoding="utf-8")
        self._files[run_id] = handle
        self.files_written.append(file_path)
        header = {
            "event": "trace_start",
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "ts": _now(),
            "run_id": str(run_id),
            "tracedoc_version": __version__,
            "framework": "langgraph" if _package_version("langgraph") else "langchain",
            "framework_version": _package_version("langgraph"),
            "langchain_core_version": _package_version("langchain-core"),
        }
        handle.write(json.dumps(header) + "\n")

    def _write(
        self,
        event: str,
        run_id: UUID,
        parent_run_id: UUID | None,
        data: dict[str, Any],
        *,
        start: bool = False,
        name: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if start:
                if parent_run_id is not None and parent_run_id in self._root_of:
                    self._root_of[run_id] = self._root_of[parent_run_id]
                else:
                    self._root_of[run_id] = run_id
                    self._open_root(run_id)
            root = self._root_of.get(run_id)
            if root is None or root not in self._files:
                return
            record: dict[str, Any] = {
                "event": event,
                "ts": _now(),
                "run_id": str(run_id),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
            }
            if name is not None:
                record["name"] = name
            if tags:
                record["tags"] = list(tags)
            if metadata:
                record["metadata"] = _metadata(metadata)
            record["data"] = to_jsonable(data, self.max_chars)
            handle = self._files[root]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if root == run_id and event.endswith(("_end", "_error")):
                handle.close()
                del self._files[root]
                self._root_of = {k: v for k, v in self._root_of.items() if v != root}

    # ---- chains ------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._write(
            "chain_start",
            run_id,
            parent_run_id,
            {"inputs": inputs},
            start=True,
            name=_name(serialized, kwargs),
            tags=tags,
            metadata=metadata,
        )

    def on_chain_end(
        self, outputs: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        self._write("chain_end", run_id, parent_run_id, {"outputs": outputs})

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = {"error": _error_text(error), "error_type": type(error).__name__}
        self._write("chain_error", run_id, parent_run_id, data)

    # ---- models ------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        params = kwargs.get("invocation_params") or {}
        model = (
            params.get("model")
            or params.get("model_name")
            or params.get("model_id")
            or (metadata or {}).get("ls_model_name")
        )
        data = {"messages": messages[0] if len(messages) == 1 else messages, "model": model}
        self._write(
            "chat_model_start",
            run_id,
            parent_run_id,
            data,
            start=True,
            name=_name(serialized, kwargs),
            tags=tags,
            metadata=metadata,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        params = kwargs.get("invocation_params") or {}
        model = (
            params.get("model") or params.get("model_name") or (metadata or {}).get("ls_model_name")
        )
        self._write(
            "llm_start",
            run_id,
            parent_run_id,
            {"prompts": prompts, "model": model},
            start=True,
            name=_name(serialized, kwargs),
            tags=tags,
            metadata=metadata,
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        output: Any = None
        usage: dict[str, int] | None = None
        model: str | None = None
        generation = response.generations[0][0] if response.generations else None
        if isinstance(generation, ChatGeneration):
            message = generation.message
            output = message
            if getattr(message, "usage_metadata", None):
                um = message.usage_metadata
                usage = {"input_tokens": um["input_tokens"], "output_tokens": um["output_tokens"]}
            meta = message.response_metadata or {}
            model = meta.get("model_name") or meta.get("model")
        elif generation is not None:
            output = generation.text
        llm_output = response.llm_output or {}
        if usage is None and isinstance(llm_output.get("token_usage"), dict):
            tu = llm_output["token_usage"]
            usage = {
                "input_tokens": int(tu.get("prompt_tokens", 0)),
                "output_tokens": int(tu.get("completion_tokens", 0)),
            }
        model = model or llm_output.get("model_name") or llm_output.get("model")
        self._write(
            "llm_end", run_id, parent_run_id, {"output": output, "usage": usage, "model": model}
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = {"error": _error_text(error), "error_type": type(error).__name__}
        self._write("llm_error", run_id, parent_run_id, data)

    # ---- tools -------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        data = {
            "input": inputs if inputs is not None else input_str,
            "tool_call_id": kwargs.get("tool_call_id"),
        }
        self._write(
            "tool_start",
            run_id,
            parent_run_id,
            data,
            start=True,
            name=_name(serialized, kwargs),
            tags=tags,
            metadata=metadata,
        )

    def on_tool_end(
        self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        if isinstance(output, ToolMessage):
            output = output.content
        self._write("tool_end", run_id, parent_run_id, {"output": output})

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        data = {"error": _error_text(error), "error_type": type(error).__name__}
        self._write("tool_error", run_id, parent_run_id, data)

    def close(self) -> None:
        """Close any files left open (for example, runs that never ended)."""
        with self._lock:
            for handle in self._files.values():
                handle.close()
            self._files.clear()
            self._root_of.clear()
