"""The order the harness recognizers are tried in, and the common blocks they produce.

Each harness module knows one wire format and nothing about the others. This module is the only
place that knows they compete: a block is offered to each recognizer in turn and the first match
wins.

**The order is behavior, not style.** Shapes overlap, so a block can satisfy more than one
recognizer and the first one reached decides how it is read:

- `strands._bedrock_*` before `claude._anthropic_*`: a harness can wrap a `toolUse` and also tag the
  block `type: "tool_use"`, and the wrapper carries the identifier the flat block lacks.
- `gemini._args_call` last among the calls: `{"name", "args"}` is the loosest shape here, and any
  block with a string name and a dict of arguments matches it, including blocks a more specific
  recognizer would have read correctly.
- `openhands._openhands_call` after the rest: its blocks are `kind`-tagged and cannot collide, so
  its position is free, but keeping it last leaves the loose recognizers' relative order intact.

Adding a harness means adding a module and one entry here. Put it above `_args_call` unless its
blocks are tagged in a way nothing else matches.
"""

from __future__ import annotations

from typing import Any

from .._normalize import _body_from_result, _load_refused, _refusal_message
from . import claude, codex, gemini, openhands, strands
from ._common import ToolCallBlock, ToolResultBlock

_CALL_ADAPTERS = (
    strands._bedrock_call,
    gemini._gemini_call,
    strands._typed_call,
    claude._anthropic_call,
    gemini._named_tool_call,
    gemini._args_call,
    openhands._openhands_call,
)

_RESULT_ADAPTERS = (
    strands._bedrock_result,
    gemini._gemini_result,
    strands._typed_result,
    claude._anthropic_result,
    codex._event_result,
    openhands._openhands_result,
)


def _tool_call(block: dict[str, Any]) -> ToolCallBlock | None:
    """The tool call this block carries, or None if it is not one."""
    for adapter in _CALL_ADAPTERS:
        matched = adapter(block)
        if matched is not None:
            call_id, name, arguments = matched
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return None
            return ToolCallBlock(str(call_id) if call_id is not None else None, name, arguments)
    return None


def _tool_result(block: dict[str, Any]) -> ToolResultBlock | None:
    """The tool result this block carries, or None if it is not one."""
    for adapter in _RESULT_ADAPTERS:
        matched = adapter(block)
        if matched is not None:
            raw_result, result_id = matched
            refused = _load_refused(raw_result)
            return ToolResultBlock(
                call_id=str(result_id) if result_id is not None else None,
                refused=refused,
                body=_body_from_result(raw_result),
                error=_refusal_message(raw_result) if refused else None,
            )
    return None
