"""Per-harness block shapes.

Each harness writes a tool call and a tool result its own way. These functions know only
those shapes: given one content block, they say whether it is a call or a result and pull out
the identifier, the name and the arguments or payload. Deciding what a call *means* (a skill
load, a file read, a refusal) is the extractor's job, not theirs.

`_tool_call` and `_tool_result` try their recognizers in a fixed order, most specific first,
and the first one that matches wins. The order matters where shapes overlap: Bedrock's
`toolUse`/`toolResult` wrappers are checked before the flatter `type`-tagged blocks, since a
harness can set both.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from ._normalize import _as_dict, _body_from_result, _load_refused, _refusal_message


class ToolCallBlock(NamedTuple):
    """A tool call recovered from a trajectory, harness-independent."""

    call_id: str | None
    name: str
    arguments: dict[str, Any]


class ToolResultBlock(NamedTuple):
    """A tool result recovered from a trajectory, harness-independent."""

    call_id: str | None
    refused: bool
    body: str | None
    error: str | None


def _looks_like_block(value: dict[str, Any]) -> bool:
    return (
        "toolUse" in value
        or "toolResult" in value
        or value.get("type") in {"tool_use", "tool_result", "text", "command_execution"}
        or "tool_name" in value
        or ("name" in value and "args" in value)
        or str(value.get("kind", "")).startswith("InvokeSkill")
    )


def _iter_indexed_blocks(messages: list[Any]) -> list[tuple[int, str | None, dict[str, Any]]]:
    """Flatten raw, Claude stream, and Codex event wrappers into content blocks."""
    blocks: list[tuple[int, str | None, dict[str, Any]]] = []
    for index, item in enumerate(messages):
        outer = _as_dict(item)
        if outer is None:
            continue

        if outer.get("type") == "item.completed" and (codex_item := _as_dict(outer.get("item"))):
            blocks.append((index, None, codex_item))
            continue
        if outer.get("type") in {"tool_response", "function_response"} or str(outer.get("kind", "")).startswith(
            "InvokeSkill"
        ):
            blocks.append((index, None, outer))
            continue

        message = _as_dict(outer.get("message")) or outer
        role = message.get("role") or outer.get("role")
        content = message.get("content")
        if isinstance(content, list):
            blocks.extend(
                (index, str(role) if role else None, block)
                for content_item in content
                if (block := _as_dict(content_item)) is not None
            )
        elif (block := _as_dict(content)) is not None:
            blocks.append((index, str(role) if role else None, block))
        elif isinstance(content, str):
            blocks.append((index, str(role) if role else None, {"type": "text", "text": content}))
        elif _looks_like_block(message):
            blocks.append((index, str(role) if role else None, message))
    return blocks


# ---- Tool results -----------------------------------------------------------
#
# Each recognizer returns (raw payload, call id) for the shape it knows, else None.


def _bedrock_result(block: dict[str, Any]) -> tuple[Any, Any] | None:
    """Bedrock / Strands native: `{"toolResult": {"toolUseId", "content"}}`."""
    if isinstance(block.get("toolResult"), dict):
        return block["toolResult"], block["toolResult"].get("toolUseId")
    return None


def _gemini_result(block: dict[str, Any]) -> tuple[Any, Any] | None:
    """Gemini CLI / Google ADK content parts: payload nests under `response`."""
    raw = block.get("functionResponse") or block.get("function_response")
    if isinstance(raw, dict):
        return raw, raw.get("id")
    return None


def _typed_result(block: dict[str, Any]) -> tuple[Any, Any] | None:
    """A strands_evals `ToolResultContent`, dumped to a dict."""
    if block.get("content_type") == "tool_result":
        return block, block.get("tool_call_id")
    return None


def _anthropic_result(block: dict[str, Any]) -> tuple[Any, Any] | None:
    """Anthropic-style content block: `{"type": "tool_result", "tool_use_id"}`."""
    if block.get("type") == "tool_result":
        return block, block.get("tool_use_id") or block.get("id")
    return None


def _event_result(block: dict[str, Any]) -> tuple[Any, Any] | None:
    """A `tool_response` / `function_response` event, as OpenAI Agents and Codex emit."""
    if block.get("type") in {"tool_response", "function_response"}:
        return block, block.get("tool_use_id") or block.get("id")
    return None


def _openhands_result(block: dict[str, Any]) -> tuple[Any, Any] | None:
    """OpenHands: `{"kind": "InvokeSkillObservation"}`."""
    if block.get("kind") == "InvokeSkillObservation":
        return block, block.get("tool_call_id") or block.get("id")
    return None


_RESULT_ADAPTERS = (
    _bedrock_result,
    _gemini_result,
    _typed_result,
    _anthropic_result,
    _event_result,
    _openhands_result,
)


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


# ---- Tool calls -------------------------------------------------------------
#
# Each recognizer returns (call id, name, arguments) for the shape it knows, else None.
# `name` and `arguments` are returned unvalidated; `_tool_call` drops the block if either
# is not the expected type.


def _bedrock_call(block: dict[str, Any]) -> tuple[Any, Any, Any] | None:
    """Bedrock / Strands native: `{"toolUse": {"toolUseId", "name", "input"}}`."""
    raw = block.get("toolUse")
    if isinstance(raw, dict):
        return raw.get("toolUseId"), raw.get("name"), raw.get("input")
    return None


def _gemini_call(block: dict[str, Any]) -> tuple[Any, Any, Any] | None:
    """Gemini CLI / Google ADK content parts: `{"functionCall": {"id", "name", "args"}}`."""
    raw = block.get("functionCall") or block.get("function_call")
    if isinstance(raw, dict):
        return raw.get("id"), raw.get("name"), raw.get("args")
    return None


def _typed_call(block: dict[str, Any]) -> tuple[Any, Any, Any] | None:
    """A strands_evals `ToolCallContent`, dumped to a dict."""
    if block.get("content_type") == "tool_use":
        return block.get("tool_call_id"), block.get("name"), block.get("arguments")
    return None


def _anthropic_call(block: dict[str, Any]) -> tuple[Any, Any, Any] | None:
    """Anthropic-style content block: `{"type": "tool_use", "name", "input"}`."""
    if block.get("type") == "tool_use":
        return block.get("id") or block.get("tool_use_id"), block.get("name"), block.get("input")
    return None


def _named_tool_call(block: dict[str, Any]) -> tuple[Any, Any, Any] | None:
    """Gemini CLI stream events: `{"tool_name", "parameters"}`."""
    if block.get("tool_name"):
        return block.get("id"), block.get("tool_name"), block.get("parameters")
    return None


def _args_call(block: dict[str, Any]) -> tuple[Any, Any, Any] | None:
    """A bare `{"name", "args"}` call, as Google ADK and Codex event items emit."""
    if isinstance(block.get("name"), str) and isinstance(block.get("args"), dict):
        return block.get("id"), block.get("name"), block.get("args")
    return None


def _openhands_call(block: dict[str, Any]) -> tuple[Any, Any, Any] | None:
    """OpenHands: `{"kind": "InvokeSkillAction", "name"}`, with the skill name as the action name."""
    if block.get("kind") == "InvokeSkillAction":
        return block.get("tool_call_id") or block.get("id"), "invoke_skill", {"name": block.get("name")}
    return None


_CALL_ADAPTERS = (
    _bedrock_call,
    _gemini_call,
    _typed_call,
    _anthropic_call,
    _named_tool_call,
    _args_call,
    _openhands_call,
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
