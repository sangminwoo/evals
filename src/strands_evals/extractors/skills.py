"""Skill trajectory parsing helpers.

`parse_available_skills` recovers the skills exposed to the agent (name plus
description) from the harness-injected `<available_skills>` block, and
`extract_selected_skills` recovers the skills the agent loaded, in invocation
order, each with its `SKILL.md` body when the trajectory carried it. Both accept
a `Session` or a raw message list.

Skills are not first-class in the trace schema the way tools are: there is no
`AgentInvocationSpan.available_skills` field, and a skill invocation surfaces as
an ordinary tool call. A load is detected either by a reserved skill-tool name and
its skill-name argument, or by a read of a known `SKILL.md` path.
"""

from __future__ import annotations

import html
import json
import logging
import re
from typing import Any, NamedTuple

import yaml
from pydantic import BaseModel
from strands import Skill

from ..types.trace import (
    AgentInvocationSpan,
    Session,
    ToolExecutionSpan,
)

logger = logging.getLogger(__name__)

# Cap on serialized trajectory size in judge prompts, ~150k tokens at 4 chars/token.
_MAX_TRAJECTORY_CHARS = 600_000


class AvailableSkill(NamedTuple):
    """A skill exposed to the agent at runtime."""

    name: str
    description: str


class InvokedSkill(NamedTuple):
    """A skill the agent actually loaded during the run."""

    name: str
    body: str | None  # SKILL.md text if captured from the trajectory, else None


# Reserved skill-tool name -> the input-argument key that holds the skill name.
# From the design-doc B.1 table (verified against real runs).
_HARNESS_TOOLS: dict[str, str] = {
    "skills": "skill_name",  # Strands AgentSkills plugin
    "Skill": "skill",  # Claude Code / Claude Agent SDK
    "load_skill": "skill_name",  # OpenAI Agents SDK, Google ADK
    "activate_skill": "name",  # Gemini CLI
    "invoke_skill": "name",  # OpenHands
}

# The available-skills block the harness injects into the system prompt.
_AVAILABLE_BLOCK = re.compile(r"<available_skills>(.*?)</available_skills>", re.DOTALL | re.IGNORECASE)
_SKILL_ENTRY = re.compile(r"<skill>(.*?)</skill>", re.DOTALL | re.IGNORECASE)
_NAME_TAG = re.compile(r"<name>(.*?)</name>", re.DOTALL | re.IGNORECASE)
_DESC_TAG = re.compile(r"<description>(.*?)</description>", re.DOTALL | re.IGNORECASE)
_AVAILABLE_MARKDOWN = re.compile(
    r"^### Available skills\s*$\n(?P<body>.*?)(?=^###\s|\Z)",
    re.DOTALL | re.IGNORECASE | re.MULTILINE,
)
_AVAILABLE_MARKDOWN_ENTRY = re.compile(
    r"^\s*-\s+(?P<name>[^:\n]+):\s*(?P<description>.*?)\s*$",
    re.MULTILINE,
)
_FILE_LOCATOR = re.compile(r"\s+\(file:\s*.+?\)\s*$", re.IGNORECASE)
_SKILL_PATH = re.compile(
    r'"([^"\n]*SKILL\.md)"|\'([^\'\n]*SKILL\.md)\'|([^\s"\'=;|<>]*SKILL\.md)',
    re.IGNORECASE,
)
_READ_COMMAND = re.compile(r"\b(?:cat|sed|head|tail|bat|less|type|Get-Content)\b", re.IGNORECASE)
_READ_TOOL_NAMES = {
    "read",
    "read_file",
    "file_read",
    "filesystem_read",
    "read_text_file",
}
_SHELL_TOOL_NAMES = {"bash", "shell", "terminal", "command", "execute_command", "run_command"}
_DISCOVERY_TOOL_NAMES = {"list_skills", "search_skills"}
_FAILED_STATUSES = {"error", "errored", "fail", "failed", "failure", "cancelled", "canceled"}
_ACKNOWLEDGEMENT = re.compile(
    # A load acknowledgement is not the skill body. Harnesses word this either way round
    # ("Launching skill: x", or Gemini CLI's "Skill activated. Resources loaded from ..."),
    # and mistaking one for the body would have the judge score a status line as instructions.
    # The optional group is the skill name some harnesses interpose, e.g. the Strands
    # AgentSkills plugin's "Skill 'x' activated (no instructions available).".
    # Matched without DOTALL so that a body whose first line is an acknowledgement is kept:
    # only a result that is nothing but the status line is discarded.
    r"^(?:(?:Launching|Loading|Activating|Loaded|Activated)\s+skill"
    r"|skill\s+(?:'[^']*'|\"[^\"]*\"|[\w.-]+)?\s*(?:activated|loaded|launched))"
    r"\b(?:\s*[:.]?\s*.*)?$",
    re.IGNORECASE,
)


def _parse_available_block(text: str) -> list[AvailableSkill]:
    """Parse an XML or Markdown available-skills section from prompt text.

    Skills missing a name are skipped. A missing description yields an empty
    description rather than dropping the skill.
    """
    block_match = _AVAILABLE_BLOCK.search(text or "")
    if block_match:
        out: list[AvailableSkill] = []
        for entry in _SKILL_ENTRY.finditer(block_match.group(1)):
            body = entry.group(1)
            name_m = _NAME_TAG.search(body)
            if not name_m:
                continue
            desc_m = _DESC_TAG.search(body)
            out.append(
                AvailableSkill(
                    name=html.unescape(name_m.group(1).strip()),
                    description=html.unescape(desc_m.group(1).strip()) if desc_m else "",
                )
            )
        return out

    markdown_match = _AVAILABLE_MARKDOWN.search(text or "")
    if not markdown_match:
        return []
    return [
        AvailableSkill(
            name=match.group("name").strip(),
            description=_FILE_LOCATOR.sub("", match.group("description")).strip(),
        )
        for match in _AVAILABLE_MARKDOWN_ENTRY.finditer(markdown_match.group("body"))
    ]


def _skill_name_from_args(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Read the skill name from a reserved tool's input arguments."""
    key = _HARNESS_TOOLS.get(tool_name)
    if key is None and tool_name.casefold().endswith("_load_skill"):
        key = "skill_name"  # Google ADK permits a tool_name_prefix.
    if key is None:
        return None
    value = arguments.get(key)
    return str(value) if value else None


# ---- Session path -----------------------------------------------------------


def _available_from_session(session: Session) -> list[AvailableSkill]:
    """Recover available skills from the first AgentInvocationSpan.system_prompt that has the block."""
    for trace in session.traces:
        for span in trace.spans:
            if isinstance(span, AgentInvocationSpan) and span.system_prompt:
                skills = _parse_available_block(span.system_prompt)
                if skills:
                    return skills
    return []


def _selected_from_session(session: Session) -> list[InvokedSkill]:
    """Recover invoked skills from ToolExecutionSpans with a reserved skill-tool name.

    The skill body is taken from the tool result content when present. (Some
    harnesses put the body elsewhere, e.g. Claude Code's following message; those
    are handled by their raw-list adapters and are follow-ups for the Session path.)
    """
    out: list[InvokedSkill] = []
    for trace in session.traces:
        for span in trace.spans:
            if not isinstance(span, ToolExecutionSpan):
                continue
            if span.tool_result.error:
                continue
            skill_name = _skill_name_from_args(span.tool_call.name, span.tool_call.arguments)
            body = _body_from_result(span.tool_result.content)
            if skill_name is not None:
                out.append(InvokedSkill(name=skill_name, body=body))
                continue

            read_path = _skill_read_path(span.tool_call.name, span.tool_call.arguments)
            if read_path and body:
                out.append(InvokedSkill(name=_skill_name_from_body(body, read_path), body=body))
    return _deduplicate_invocations(out)


# ---- Raw message-list path --------------------------------------------------
#
# Strands' native in-memory message shape: assistant messages carry
# {"toolUse": {"name", "toolUseId", "input": {...}}} blocks, and the following
# user message carries {"toolResult": {"toolUseId", "content": [...]}}. We also
# accept already-parsed strands_evals message objects (UserMessage/AssistantMessage).


def _as_dict(value: Any) -> dict[str, Any] | None:
    """Normalize raw dictionaries and strands_evals Pydantic trace objects."""
    if isinstance(value, dict):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return None


def _content_text(content: Any) -> str:
    """Flatten common text/content wrappers into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_content_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    item = _as_dict(content)
    if item is not None:
        for key in (
            "instructions",
            "llmContent",
            "content",
            "output",
            "aggregated_output",
            "text",
            # Google ADK nests its tool payload under `response`, with plain tool output
            # under `result`; both must be traversed to reach the skill body or catalog.
            "response",
            "result",
        ):
            if key in item:
                text = _content_text(item[key])
                if text:
                    return text
    return ""


def _result_failed(result: Any) -> bool:
    value = _as_dict(result)
    if value is None:
        return False
    status = str(value.get("status", "")).casefold()
    exit_code = value.get("exit_code")
    failed = (
        status in _FAILED_STATUSES
        or value.get("is_error") is True
        or value.get("error") not in (None, "", False)
        or exit_code not in (None, 0, "0")
    )
    if failed:
        return True
    return any(
        _result_failed(value[key])
        for key in ("response", "result")
        if key in value and _as_dict(value[key]) is not None
    )


def _body_from_result(result: Any) -> str | None:
    """Return actual skill instructions, excluding errors and load acknowledgements."""
    if _result_failed(result):
        return None
    text = _content_text(result).strip()
    if not text or _ACKNOWLEDGEMENT.fullmatch(text):
        return None

    # Some tool integrations JSON-encode their structured result.
    if text.startswith(("{", "[")):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if decoded != result:
                return _body_from_result(decoded)
    return text


def _skill_name_from_path(path: str) -> str:
    normalized = path.replace("\\", "/").rstrip("/")
    parts = normalized.split("/")
    return parts[-2] if len(parts) >= 2 else normalized


def _canonical_skill_key(name: str) -> str:
    """Fold the naming variants that refer to one skill.

    An agent may read the same `SKILL.md` more than once, and a partial read (a paged `sed`
    window that misses the frontmatter) falls back to the directory name while a full read
    reports the frontmatter name. A directory named `pdf_processing` holding a skill whose
    frontmatter says `pdf-processing` is the same skill, so the two separators fold together.
    `.` is left alone: it is legal in a skill name, so folding it would merge `data.clean`
    and `data-clean`, which are two different skills.
    """
    return name.casefold().replace("_", "-")


def _skill_name_from_body(body: str, path: str) -> str:
    """Prefer the runtime-visible frontmatter name over a directory alias.

    `Skill.from_content` parses the `SKILL.md` YAML frontmatter. Bodies are
    read from arbitrary on-disk files, so malformed frontmatter is expected;
    `yaml` raises `YAMLError` (not a `ValueError`) on bad structure. Both
    fall back to the directory-derived name rather than aborting extraction.
    """
    try:
        return Skill.from_content(body).name
    except (ValueError, yaml.YAMLError):
        return _skill_name_from_path(path)


def _skill_path_from_text(value: str) -> str | None:
    match = _SKILL_PATH.search(value)
    if not match:
        return None
    return next(group for group in match.groups() if group is not None)


def _skill_read_path(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Return a SKILL.md path only for recognizable file-read operations."""
    lowered = tool_name.casefold()
    if lowered in _READ_TOOL_NAMES or any(lowered.endswith(f".{name}") for name in _READ_TOOL_NAMES):
        for key in ("path", "file_path", "filename"):
            value = arguments.get(key)
            if isinstance(value, str) and (path := _skill_path_from_text(value)):
                return path

    if lowered in _SHELL_TOOL_NAMES:
        command = arguments.get("command") or arguments.get("cmd")
        if isinstance(command, str) and _READ_COMMAND.search(command):
            return _skill_path_from_text(command)
    return None


def _deduplicate_invocations(invocations: list[InvokedSkill]) -> list[InvokedSkill]:
    """Keep first-invocation order and the fullest body recovered per skill."""
    out: list[InvokedSkill] = []
    index_by_key: dict[str, int] = {}
    for invocation in invocations:
        key = _canonical_skill_key(invocation.name)
        index = index_by_key.get(key)
        if index is None:
            index_by_key[key] = len(out)
            out.append(invocation)
            continue
        # Prefer the fullest body, and with it the name that body declares. A later read only
        # wins when it contains what was already recovered, which is what a re-read of the
        # same file looks like: a paged window is contained in the whole file. Unrelated
        # output that happened to be attributed to this skill is not, so it cannot displace
        # a real body just by being longer.
        kept = out[index].body or ""
        candidate = invocation.body or ""
        if len(candidate) > len(kept) and kept in candidate:
            out[index] = invocation
    return out


def _structured_available_skills(message: Any) -> list[AvailableSkill]:
    """Recover structured catalogs, including discovery-tool response wrappers."""
    pending = [message]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))

        if isinstance(candidate, str) and candidate.lstrip().startswith(("{", "[")):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(candidate, list):
            pending.extend(candidate)
            continue

        value = _as_dict(candidate)
        if value is None:
            continue
        skills = value.get("skills")
        if isinstance(skills, list):
            out: list[AvailableSkill] = []
            for skill in skills:
                if isinstance(skill, str):
                    out.append(AvailableSkill(skill, ""))
                elif isinstance(skill, dict) and skill.get("name"):
                    out.append(AvailableSkill(str(skill["name"]), str(skill.get("description", ""))))
            if out:
                return out
        pending.extend(
            value[key]
            for key in (
                "response",
                "result",
                "output",
                "content",
                "toolResult",
                "functionResponse",
                "function_response",
                "toolResponse",
            )
            if key in value
        )
    return []


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


def _iter_raw_blocks(messages: list[Any]) -> list[dict[str, Any]]:
    """Yield normalized content blocks from raw or typed messages."""
    return [block for _, _, block in _iter_indexed_blocks(messages)]


def _text_candidates(value: Any) -> list[str]:
    """Collect prompt/result text recursively without stringifying opaque objects."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _text_candidates(item)]
    item = _as_dict(value)
    if item is None:
        return []
    texts: list[str] = []
    for key in (
        "system_prompt",
        # Result wrappers, so a discovery tool's catalog is reachable: harnesses nest the
        # payload one level down (Bedrock `toolResult`, Gemini/ADK `functionResponse`).
        "toolResult",
        "functionResponse",
        "function_response",
        "toolResponse",
        "content",
        "text",
        "output",
        "aggregated_output",
        "llmContent",
        "instructions",
        "message",
        "response",
        "result",
    ):
        if key in item:
            texts.extend(_text_candidates(item[key]))
    return texts


def _discovery_tool_name(block: dict[str, Any]) -> str | None:
    for candidate in (
        block.get("name"),
        block.get("tool_name"),
        block.get("functionResponse"),
        block.get("function_response"),
        block.get("toolResponse"),
    ):
        if isinstance(candidate, str):
            return candidate
        value = _as_dict(candidate)
        if value is not None and isinstance(value.get("name"), str):
            return value["name"]
    return None


def _result_id(block: dict[str, Any]) -> str | None:
    candidates: list[Any] = [
        block.get("tool_call_id"),
        block.get("tool_use_id"),
        block.get("id"),
    ]
    for key in ("toolResult", "functionResponse", "function_response", "toolResponse"):
        value = _as_dict(block.get(key))
        if value is not None:
            candidates.extend(
                (
                    value.get("toolUseId"),
                    value.get("tool_call_id"),
                    value.get("tool_use_id"),
                    value.get("id"),
                )
            )
    return next((str(candidate) for candidate in candidates if candidate is not None), None)


def _is_discovery_tool_name(name: str) -> bool:
    lowered = name.casefold()
    return lowered in _DISCOVERY_TOOL_NAMES or any(
        lowered.endswith(f"_{discovery_name}") for discovery_name in _DISCOVERY_TOOL_NAMES
    )


def _available_from_list(messages: list[Any]) -> list[AvailableSkill]:
    """Parse trusted system catalogs and skill-discovery tool results."""
    indexed_blocks = _iter_indexed_blocks(messages)
    discovery_ids = {
        call_id
        for _, _, block in indexed_blocks
        if (call := _tool_call(block)) is not None
        and _is_discovery_tool_name(call[1])
        and (call_id := call[0]) is not None
    }

    for msg in messages:
        outer = _as_dict(msg)
        if outer is None:
            continue
        message = _as_dict(outer.get("message")) or outer
        role = str(message.get("role") or outer.get("role") or "").casefold()
        is_system = role in {"system", "developer"} or str(outer.get("type", "")).casefold() == "system"
        if is_system:
            structured = _structured_available_skills(msg)
            if structured:
                return structured
            for text in _text_candidates(msg):
                skills = _parse_available_block(text)
                if skills:
                    return skills
        elif "system_prompt" in outer:
            for text in _text_candidates(outer["system_prompt"]):
                skills = _parse_available_block(text)
                if skills:
                    return skills

    for _, _, block in indexed_blocks:
        tool_name = _discovery_tool_name(block)
        is_discovery_result = (isinstance(tool_name, str) and _is_discovery_tool_name(tool_name)) or _result_id(
            block
        ) in discovery_ids
        if not is_discovery_result:
            continue
        structured = _structured_available_skills(block)
        if structured:
            return structured
        for text in _text_candidates(block):
            skills = _parse_available_block(text)
            if skills:
                return skills
    return []


def _tool_result(block: dict[str, Any]) -> tuple[str | None, bool, str | None] | None:
    raw_result: Any = None
    result_id: Any = None
    if isinstance(block.get("toolResult"), dict):
        raw_result = block["toolResult"]
        result_id = raw_result.get("toolUseId")
    elif isinstance(block.get("functionResponse") or block.get("function_response"), dict):
        # Gemini / Google ADK content parts: payload nests under `response`.
        raw_result = block.get("functionResponse") or block.get("function_response")
        result_id = raw_result.get("id")
    elif block.get("content_type") == "tool_result":
        raw_result = block
        result_id = block.get("tool_call_id")
    elif block.get("type") == "tool_result":
        raw_result = block
        result_id = block.get("tool_use_id") or block.get("id")
    elif block.get("type") in {"tool_response", "function_response"}:
        raw_result = block
        result_id = block.get("tool_use_id") or block.get("id")
    elif block.get("kind") == "InvokeSkillObservation":
        raw_result = block
        result_id = block.get("tool_call_id") or block.get("id")
    if raw_result is None:
        return None
    return (
        str(result_id) if result_id is not None else None,
        _result_failed(raw_result),
        _body_from_result(raw_result),
    )


def _tool_call(block: dict[str, Any]) -> tuple[str | None, str, dict[str, Any]] | None:
    raw_call: Any = None
    call_id: Any = None
    if isinstance(block.get("toolUse"), dict):
        raw_call = block["toolUse"]
        call_id = raw_call.get("toolUseId")
        name = raw_call.get("name")
        arguments = raw_call.get("input")
    elif isinstance(block.get("functionCall") or block.get("function_call"), dict):
        # Gemini / Google ADK content parts.
        raw_call = block.get("functionCall") or block.get("function_call")
        call_id = raw_call.get("id")
        name = raw_call.get("name")
        arguments = raw_call.get("args")
    elif block.get("content_type") == "tool_use":
        call_id = block.get("tool_call_id")
        name = block.get("name")
        arguments = block.get("arguments")
    elif block.get("type") == "tool_use":
        call_id = block.get("id") or block.get("tool_use_id")
        name = block.get("name")
        arguments = block.get("input")
    elif block.get("tool_name"):
        call_id = block.get("id")
        name = block.get("tool_name")
        arguments = block.get("parameters")
    elif isinstance(block.get("name"), str) and isinstance(block.get("args"), dict):
        call_id = block.get("id")
        name = block.get("name")
        arguments = block.get("args")
    elif block.get("kind") == "InvokeSkillAction":
        call_id = block.get("tool_call_id") or block.get("id")
        name = "invoke_skill"
        arguments = {"name": block.get("name")}
    else:
        return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return (str(call_id) if call_id is not None else None, name, arguments)


def _claude_body_after(
    indexed_blocks: list[tuple[int, str | None, dict[str, Any]]],
    call_index: int,
) -> str | None:
    """Find Claude Code's injected skill body after its launch acknowledgement."""
    for index, role, block in indexed_blocks:
        if index <= call_index or role not in (None, "user"):
            continue
        text = _content_text(block.get("text") if block.get("type") == "text" else block)
        if text.lstrip().startswith("Base directory for this skill:"):
            return text
    return None


def _selected_from_list(messages: list[Any]) -> list[InvokedSkill]:
    """Recover invoked skills from raw or typed message lists.

    Matches assistant `toolUse` blocks with a reserved skill-tool name, then
    pairs each with the `toolResult` block (by toolUseId) that carries the body.
    Typed `ToolCallContent` / `ToolResultContent` blocks use the equivalent
    `content_type` and `tool_call_id` fields.
    """
    indexed_blocks = _iter_indexed_blocks(messages)
    results_by_id: dict[str, tuple[bool, str | None]] = {}
    unkeyed_results: list[tuple[int, bool, str | None]] = []
    for result_index, _, block in indexed_blocks:
        parsed_result = _tool_result(block)
        if parsed_result is None:
            continue
        if parsed_result[0] is not None:
            results_by_id[parsed_result[0]] = (parsed_result[1], parsed_result[2])
        else:
            unkeyed_results.append((result_index, parsed_result[1], parsed_result[2]))

    out: list[InvokedSkill] = []
    used_unkeyed_results: set[int] = set()
    for message_index, _, block in indexed_blocks:
        if block.get("type") == "command_execution":
            command = block.get("command")
            body = _body_from_result(block)
            if (
                isinstance(command, str)
                and _READ_COMMAND.search(command)
                and (path := _skill_path_from_text(command))
                and body
            ):
                out.append(InvokedSkill(_skill_name_from_body(body, path), body))
            continue

        call = _tool_call(block)
        if call is None:
            continue
        call_id, tool_name, arguments = call
        matched_result = results_by_id.get(call_id) if call_id is not None else None
        if matched_result is None and call_id is None:
            unkeyed_match = next(
                (
                    (index, failed, body)
                    for index, failed, body in unkeyed_results
                    if index > message_index and index not in used_unkeyed_results
                ),
                None,
            )
            if unkeyed_match is not None:
                used_unkeyed_results.add(unkeyed_match[0])
                matched_result = (unkeyed_match[1], unkeyed_match[2])

        skill_name = _skill_name_from_args(tool_name, arguments)
        if skill_name is not None:
            if matched_result is None or matched_result[0]:
                continue
            body = matched_result[1]
            if tool_name == "Skill" and body is None:
                body = _claude_body_after(indexed_blocks, message_index)
            out.append(InvokedSkill(skill_name, body))
            continue

        read_path = _skill_read_path(tool_name, arguments)
        if read_path and matched_result is not None and not matched_result[0] and matched_result[1]:
            out.append(InvokedSkill(_skill_name_from_body(matched_result[1], read_path), matched_result[1]))
    return _deduplicate_invocations(out)


# ---- Public API -------------------------------------------------------------


def serialize_trajectory(trajectory: Session | list[Any] | None, max_chars: int = _MAX_TRAJECTORY_CHARS) -> str:
    """Serialize a trajectory into stable JSON, for use in judge prompts.

    Truncates the middle of long runs: a real trajectory can reach millions of tokens
    (one read of a large artifact is enough), which overflows any judge context window.
    The head and tail are kept because skills are loaded early and the outcome lands late.
    Pass `max_chars=0` to disable.
    """
    if trajectory is None:
        return "(no trajectory)"
    if isinstance(trajectory, Session):
        value: Any = trajectory.model_dump(mode="json")
    else:
        value = [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in trajectory]
    text = json.dumps(value, indent=2, default=str)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return f"{text[:keep]}\n\n... [{len(text) - 2 * keep} characters omitted] ...\n\n{text[-keep:]}"


def parse_available_skills(trajectory: Session | list[Any] | str | None) -> list[AvailableSkill]:
    """Return the skills exposed to the agent (name + description).

    Accepts a `Session`, a raw message list, or a bare prompt string (e.g. a
    harness's system prompt, which is where the block is injected but which some
    session mappers store separately from the message list). Returns [] when no
    `<available_skills>` block is found.
    """
    if isinstance(trajectory, Session):
        return _available_from_session(trajectory)
    if isinstance(trajectory, str):
        return _parse_available_block(trajectory)
    if isinstance(trajectory, list):
        return _available_from_list(trajectory)
    if trajectory is not None:
        logger.debug("type=<%s> | unsupported trajectory type for available skills", type(trajectory).__name__)
    return []


def extract_selected_skills(trajectory: Session | list[Any] | None) -> list[InvokedSkill]:
    """Return the skills the agent loaded, in invocation order.

    Accepts a `Session` or a raw message list. Each `InvokedSkill` carries the
    `SKILL.md` body when the trajectory surfaced it, else `None`.
    """
    if isinstance(trajectory, Session):
        return _selected_from_session(trajectory)
    if isinstance(trajectory, list):
        return _selected_from_list(trajectory)
    if trajectory is not None:
        logger.debug("type=<%s> | unsupported trajectory type for invoked skills", type(trajectory).__name__)
    return []
