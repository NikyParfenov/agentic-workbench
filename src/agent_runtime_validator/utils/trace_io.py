"""Serialization helpers for ExecutionTrace.

Pydantic provides the underlying JSON encoding; these functions add a
file-path API and predictable pretty-print defaults so callers do not
need to know about ``model_dump_json`` / ``model_validate_json``.
"""

from pathlib import Path

from ..schema.trace import ExecutionTrace


def trace_to_json(trace: ExecutionTrace, indent: int = 2) -> str:
    """Serialize *trace* to a JSON string."""
    return trace.model_dump_json(indent=indent)


def trace_from_json(json_str: str) -> ExecutionTrace:
    """Deserialize an ``ExecutionTrace`` from a JSON string."""
    return ExecutionTrace.model_validate_json(json_str)


def save_trace(trace: ExecutionTrace, path: str | Path) -> None:
    """Write *trace* as JSON to *path*, creating parent directories if needed."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(trace_to_json(trace), encoding="utf-8")


def load_trace(path: str | Path) -> ExecutionTrace:
    """Load an ``ExecutionTrace`` from a JSON file previously saved by :func:`save_trace`."""
    return trace_from_json(Path(path).read_text(encoding="utf-8"))
