"""Codex and OpenAI Agents event shapes.

Both emit events rather than messages. Codex wraps each in
`{"type": "item.completed", "item": {...}}`, unwrapped by `_common._iter_indexed_blocks`, and its
skill loads arrive as `command_execution` blocks (a shell read of a `SKILL.md` path) rather than as
a reserved skill tool, so recognizing them is the extractor's job. OpenAI Agents exposes a
`load_skill` tool, whose call rides the `{"name", "args"}` recognizer in `gemini`.

What is left here is the result envelope the two share.
"""

from __future__ import annotations

from typing import Any

from ._common import ResultMatch


def _event_result(block: dict[str, Any]) -> ResultMatch | None:
    """A `tool_response` / `function_response` event, as OpenAI Agents and Codex emit."""
    if block.get("type") in {"tool_response", "function_response"}:
        return block, block.get("tool_use_id") or block.get("id")
    return None
