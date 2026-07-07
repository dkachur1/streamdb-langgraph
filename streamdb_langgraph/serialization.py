"""Serialization helpers for AG-UI payload fields."""

from __future__ import annotations

from pydantic_core import to_json


def to_ag_ui_json(value: object) -> str:
    """Serialize structured AG-UI payload fields as JSON.

    ``serialize_unknown=True`` is a deliberate, lossy fallback: anything
    pydantic can't serialize is rendered as its ``str()`` repr instead of
    raising. A serialization error here would tear down the run stream, and a
    best-effort representation for display is always preferable to that.
    """
    return to_json(value, serialize_unknown=True).decode()


def to_ag_ui_content(value: object) -> str:
    """Preserve text content; JSON-encode structured content."""
    if isinstance(value, str):
        return value
    return to_ag_ui_json(value)


def to_ag_ui_tool_name(value: object) -> str:
    """AG-UI tool names must be non-empty for stable frontend rendering."""
    name = str(value).strip() if value is not None else ""
    return name or "tool"
