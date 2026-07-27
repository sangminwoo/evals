"""Unit tests for the skill parsing helpers (parse_available_skills, extract_selected_skills)."""

from datetime import datetime

import pytest

from strands_evals.extractors import (
    AvailableSkill,
    extract_selected_skills,
    parse_available_skills,
    serialize_trajectory,
)
from strands_evals.types.trace import (
    AgentInvocationSpan,
    AssistantMessage,
    Session,
    SpanInfo,
    TextContent,
    ToolCall,
    ToolCallContent,
    ToolExecutionSpan,
    ToolResult,
    ToolResultContent,
    Trace,
    UserMessage,
)

from . import skill_fixtures as fx

# ---- raw message-list path --------------------------------------------------


def test_available_skills_from_strands_list():
    skills = parse_available_skills(fx.STRANDS_MESSAGES)
    assert skills == [
        AvailableSkill("pdf-processing", fx.PDF_DESCRIPTION),
        AvailableSkill("spreadsheet-analysis", "Analyze, edit, or generate spreadsheets."),
    ]


def test_available_skills_unescapes_xml_entities():
    prompt = (
        "<available_skills><skill><name>research&amp;review</name>"
        "<description>Compare A &lt; B &amp; report.</description></skill></available_skills>"
    )

    assert parse_available_skills(prompt) == [
        AvailableSkill("research&review", "Compare A < B & report."),
    ]


def test_selected_skills_from_strands_list_with_body():
    invoked = extract_selected_skills(fx.STRANDS_MESSAGES)
    assert len(invoked) == 1
    assert invoked[0].name == "pdf-processing"
    assert invoked[0].body is not None and "PDF Processing Skill" in invoked[0].body


@pytest.mark.parametrize(
    "messages,expected_name,expect_body_substr",
    [
        (fx.STRANDS_MESSAGES, "pdf-processing", "PDF Processing Skill"),
        (fx.CLAUDE_CODE_MESSAGES, "pdf-processing", "PDF Processing Skill"),
        (fx.CODEX_MESSAGES, "pdf-processing", "PDF Processing Skill"),
        (fx.OPENAI_AGENTS_MESSAGES, "pdf-processing", "PDF Processing Skill"),
        (fx.GEMINI_MESSAGES, "pdf-processing", "<instructions>"),
        (fx.GEMINI_STREAM_MESSAGES, "pdf-processing", "<instructions>"),
        (fx.GOOGLE_ADK_MESSAGES, "pdf-processing", "PDF Processing Skill"),
        (fx.OPENHANDS_MESSAGES, "pdf-processing", "PDF Processing Skill"),
    ],
)
def test_selected_skills_cross_harness(messages, expected_name, expect_body_substr):
    invoked = extract_selected_skills(messages)
    assert len(invoked) == 1
    assert invoked[0].name == expected_name
    assert invoked[0].body is not None and expect_body_substr in invoked[0].body


def test_near_miss_is_not_an_invocation():
    # A skill name mentioned in prose is not an invocation, and here the file_read
    # of the SKILL.md path is rejected specifically because its result carried NO
    # skill body. A SKILL.md read whose result DOES return a body is a valid
    # filesystem-skill load (Codex / OpenAI Agents); see
    # test_session_skill_file_read_with_body. Do not remove the file-read branch.
    assert extract_selected_skills(fx.NEAR_MISS_MESSAGES) == []
    # The available block is still recoverable from the same messages.
    assert [s.name for s in parse_available_skills(fx.NEAR_MISS_MESSAGES)] == [
        "pdf-processing",
        "spreadsheet-analysis",
    ]


def test_empty_messages():
    assert parse_available_skills(fx.EMPTY_MESSAGES) == []
    assert extract_selected_skills(fx.EMPTY_MESSAGES) == []


def test_multiple_invocations_in_order():
    invoked = extract_selected_skills(fx.MULTI_INVOKE_MESSAGES)
    assert [i.name for i in invoked] == ["pdf-processing", "spreadsheet-analysis"]
    assert all(i.body for i in invoked)


def test_available_skills_from_markdown_section():
    assert parse_available_skills(fx.CODEX_MESSAGES) == [AvailableSkill("pdf-processing", "Use this skill for PDFs.")]


def test_available_skills_from_claude_init_event():
    messages = [{"type": "system", "subtype": "init", "skills": ["pdf-processing", "deep-research"]}]
    assert parse_available_skills(messages) == [
        AvailableSkill("pdf-processing", ""),
        AvailableSkill("deep-research", ""),
    ]


def test_available_skills_from_nested_discovery_response():
    messages = [
        {
            "type": "tool_response",
            "name": "list_skills",
            "response": {
                "skills": [
                    {"name": "pdf-processing", "description": "Read PDFs."},
                    {"name": "deep-research", "description": "Research topics."},
                ]
            },
        }
    ]

    assert parse_available_skills(messages) == [
        AvailableSkill("pdf-processing", "Read PDFs."),
        AvailableSkill("deep-research", "Research topics."),
    ]


def test_available_skills_from_correlated_discovery_result():
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "discovery-1",
                        "name": "search_skills",
                        "input": {"query": "PDF"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "discovery-1",
                        "content": [
                            {
                                "skills": [
                                    {
                                        "name": "pdf-processing",
                                        "description": "Read PDFs.",
                                    }
                                ]
                            }
                        ],
                    }
                }
            ],
        },
    ]

    assert parse_available_skills(messages) == [
        AvailableSkill("pdf-processing", "Read PDFs."),
    ]


def test_user_skills_json_is_not_treated_as_available_catalog():
    messages = [
        {
            "role": "user",
            "content": [{"text": 'Analyze this payload: {"skills": ["not-available"]}'}],
        }
    ]

    assert parse_available_skills(messages) == []


def test_failed_load_is_not_an_invocation():
    assert extract_selected_skills(fx.FAILED_LOAD_MESSAGES) == []


def test_nested_failed_load_is_not_an_invocation():
    messages = [
        {
            "name": "load_skill",
            "args": {"skill_name": "pdf-processing"},
            "id": "load-1",
        },
        {
            "type": "tool_response",
            "id": "load-1",
            "response": {"status": "error", "error": "skill not found"},
        },
    ]

    assert extract_selected_skills(messages) == []


def test_string_zero_exit_code_is_successful():
    messages = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "cat /skills/pdf-processing/SKILL.md",
                "status": "completed",
                "exit_code": "0",
                "aggregated_output": fx.SKILL_BODY,
            },
        }
    ]

    assert extract_selected_skills(messages) == [("pdf-processing", fx.SKILL_BODY)]


def test_successful_load_without_body_preserves_invocation():
    assert extract_selected_skills(fx.BODY_MISSING_MESSAGES) == [("pdf-processing", None)]


def test_duplicate_loads_are_coalesced():
    invoked = extract_selected_skills(fx.DUPLICATE_LOAD_MESSAGES)
    assert invoked == [("pdf-processing", fx.SKILL_BODY)]


def test_selected_skills_from_typed_messages():
    messages = [
        AssistantMessage(
            content=[
                TextContent(text="Loading the PDF skill"),
                ToolCallContent(
                    name="skills",
                    arguments={"skill_name": "pdf-processing"},
                    tool_call_id="typed-1",
                ),
            ]
        ),
        UserMessage(
            content=[
                ToolResultContent(
                    content=fx.SKILL_BODY,
                    tool_call_id="typed-1",
                )
            ]
        ),
    ]

    invoked = extract_selected_skills(messages)

    assert invoked == [("pdf-processing", fx.SKILL_BODY)]


def test_unsupported_trajectory_type_returns_empty():
    assert parse_available_skills(None) == []
    assert extract_selected_skills(None) == []
    assert parse_available_skills("not a trajectory") == []


def test_parse_available_from_bare_system_prompt_string():
    # Some session mappers store the system prompt separately from the message list;
    # parse_available_skills accepts the bare prompt string too.
    skills = parse_available_skills(fx.AVAILABLE_BLOCK)
    assert [s.name for s in skills] == ["pdf-processing", "spreadsheet-analysis"]


# ---- Session path -----------------------------------------------------------


def _span_info() -> SpanInfo:
    return SpanInfo(session_id="s", start_time=datetime(2026, 7, 14), end_time=datetime(2026, 7, 14))


def _session(spans) -> Session:
    return Session(session_id="s", traces=[Trace(trace_id="t", session_id="s", spans=spans)])


def test_session_available_from_system_prompt():
    agent_span = AgentInvocationSpan(
        span_info=_span_info(),
        user_prompt="do pdf",
        agent_response="done",
        available_tools=[],
        system_prompt=fx.AVAILABLE_BLOCK,
    )
    skills = parse_available_skills(_session([agent_span]))
    assert [s.name for s in skills] == ["pdf-processing", "spreadsheet-analysis"]


def test_session_selected_from_tool_execution_span():
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(name="skills", arguments={"skill_name": "pdf-processing"}, tool_call_id="tu-1"),
        tool_result=ToolResult(content=fx.SKILL_BODY),
    )
    invoked = extract_selected_skills(_session([tool_span]))
    assert len(invoked) == 1
    assert invoked[0].name == "pdf-processing"
    assert "PDF Processing Skill" in invoked[0].body


def test_session_non_skill_tool_ignored():
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(name="calculator", arguments={"expression": "2+2"}, tool_call_id="c-1"),
        tool_result=ToolResult(content="4"),
    )
    assert extract_selected_skills(_session([tool_span])) == []


def test_session_failed_skill_load_ignored():
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(name="skills", arguments={"skill_name": "pdf-processing"}, tool_call_id="tu-1"),
        tool_result=ToolResult(content="skill not found", error="error"),
    )
    assert extract_selected_skills(_session([tool_span])) == []


def test_session_skill_file_read_with_body():
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(
            name="read_file",
            arguments={"path": "/skills/pdf-processing/SKILL.md"},
            tool_call_id="read-1",
        ),
        tool_result=ToolResult(content=fx.SKILL_BODY),
    )
    assert extract_selected_skills(_session([tool_span])) == [("pdf-processing", fx.SKILL_BODY)]


def test_session_skill_file_read_uses_frontmatter_name():
    body = "---\nname: canonical-skill\ndescription: Test skill.\n---\n# Steps\n1. Test."
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(
            name="read_file",
            arguments={"path": "/skills/directory-alias/SKILL.md"},
            tool_call_id="read-1",
        ),
        tool_result=ToolResult(content=body),
    )

    assert extract_selected_skills(_session([tool_span])) == [("canonical-skill", body)]


def test_opaque_load_and_alias_path_read_are_coalesced():
    body = "---\nname: canonical-skill\ndescription: Test skill.\n---\n# Steps\n1. Test."
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "load-1",
                        "name": "load_skill",
                        "input": {"skill_name": "canonical-skill"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "load-1", "content": [{"text": "Loaded skill"}]}}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "read-1",
                        "name": "read_file",
                        "input": {"path": "/skills/directory-alias/SKILL.md"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "read-1", "content": [{"text": body}]}}],
        },
    ]

    assert extract_selected_skills(messages) == [("canonical-skill", body)]


def test_command_execution_skill_read_uses_frontmatter_name():
    body = "---\nname: canonical-skill\ndescription: Test skill.\n---\n# Steps\n1. Test."
    messages = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "cat /skills/directory-alias/SKILL.md",
                "status": "completed",
                "exit_code": 0,
                "aggregated_output": body,
            },
        }
    ]

    assert extract_selected_skills(messages) == [("canonical-skill", body)]


def test_malformed_frontmatter_body_falls_back_to_path_name():
    # A SKILL.md whose frontmatter is not parseable YAML must not abort the whole
    # extraction; the name degrades to the directory alias and the body is kept.
    body = "---\nname: [unclosed\ndescription: broken\n---\n# Steps\n1. Test."
    messages = [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "cat /skills/directory-alias/SKILL.md",
                "status": "completed",
                "exit_code": 0,
                "aggregated_output": body,
            },
        }
    ]

    assert extract_selected_skills(messages) == [("directory-alias", body)]


def test_unkeyed_results_are_not_reused_across_skill_calls():
    messages = [
        {"tool_name": "activate_skill", "parameters": {"name": "first"}},
        {"type": "tool_result", "status": "success", "llmContent": "first body"},
        {"tool_name": "activate_skill", "parameters": {"name": "second"}},
        {"type": "tool_result", "status": "success", "llmContent": "second body"},
    ]

    assert extract_selected_skills(messages) == [
        ("first", "first body"),
        ("second", "second body"),
    ]


def test_session_available_absent_when_no_system_prompt():
    # Mapped sessions may drop the system prompt; then the block is not recoverable
    # from the Session (would fall back to a raw message list in practice).
    agent_span = AgentInvocationSpan(
        span_info=_span_info(),
        user_prompt="do pdf",
        agent_response="done",
        available_tools=[],
        system_prompt=None,
    )
    assert parse_available_skills(_session([agent_span])) == []


def test_serialize_trajectory_truncates_oversized_runs():
    """Real runs can exceed any judge context window, so the middle is dropped."""
    huge = [{"role": "user", "content": [{"text": "x" * 900_000}]}]

    serialized = serialize_trajectory(huge)

    assert len(serialized) < 900_000
    assert "characters omitted" in serialized


def test_serialize_trajectory_leaves_normal_runs_intact():
    small = [{"role": "user", "content": [{"text": "do pdf"}]}]

    assert "omitted" not in serialize_trajectory(small)
