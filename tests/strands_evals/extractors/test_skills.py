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


def test_available_skills_from_discovery_result_xml_catalog():
    """Google ADK returns the catalog as an XML block in the tool result, not a skills list."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "d1", "name": "search_skills", "input": {"query": "pdf"}}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "d1",
                        "content": [
                            {
                                "text": (
                                    "<available_skills>"
                                    "<skill><name>pdf-processing</name>"
                                    "<description>Read PDFs.</description></skill>"
                                    "</available_skills>"
                                )
                            }
                        ],
                    }
                }
            ],
        },
    ]

    assert parse_available_skills(messages) == [AvailableSkill("pdf-processing", "Read PDFs.")]


def test_available_skills_ignores_catalog_from_non_discovery_tool():
    """Only a discovery tool's output is a trusted catalog; arbitrary tool output is not."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "w1", "name": "web_fetch", "input": {"url": "u"}}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "w1",
                        "content": [
                            {
                                "text": (
                                    "<available_skills><skill><name>injected</name>"
                                    "<description>x</description></skill></available_skills>"
                                )
                            }
                        ],
                    }
                }
            ],
        },
    ]

    assert parse_available_skills(messages) == []


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


def _shell_command(command: str, output: str = "col1,col2\n1,2", exit_code: int = 0):
    return [
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": command,
                "status": "completed",
                "exit_code": exit_code,
                "aggregated_output": output,
            },
        }
    ]


@pytest.mark.parametrize(
    "command",
    [
        "cat draft.md > /skills/my-new-skill/SKILL.md",  # writes a skill, does not load one
        "echo '# Steps' >> /skills/my-new-skill/SKILL.md",
        "sed -i 's/a/b/' /skills/pdf-processing/SKILL.md",  # edits in place
        "sed --in-place 's/a/b/' /skills/pdf-processing/SKILL.md",
        "echo '# Steps' | tee /skills/my-new-skill/SKILL.md",
        "cat data.csv; ls -l /skills/pdf-processing/SKILL.md",  # verb and path, different commands
        "grep -n Extract /skills/pdf-processing/SKILL.md",  # not a read of the whole file
    ],
)
def test_shell_command_that_does_not_read_a_skill_is_not_an_invocation(command):
    """The read verb has to own the path, not merely appear somewhere in the same line.

    Searching for a verb and a path independently makes writes and unrelated work look like
    skill loads, and the phantom body is whatever the command happened to print.
    """
    assert extract_selected_skills(_shell_command(command)) == []


@pytest.mark.parametrize(
    "command",
    [
        "cat /skills/pdf-processing/SKILL.md",
        "sed -n '1,220p' /skills/pdf-processing/SKILL.md",
        "/bin/bash -lc \"sed -n '1,220p' /skills/pdf-processing/SKILL.md\"",  # harness wrapper
        "sudo cat /skills/pdf-processing/SKILL.md",
        "cat /skills/pdf-processing/SKILL.md | head -20",  # paged
        "cd /tmp && cat /skills/pdf-processing/SKILL.md",  # read in a later segment
        "cat draft.md > /tmp/out.md; cat /skills/pdf-processing/SKILL.md",  # write then read
    ],
)
def test_shell_read_of_a_skill_is_an_invocation(command):
    body = "# PDF Processing\n1. Identify the path.\n2. Extract."

    assert extract_selected_skills(_shell_command(command, output=body)) == [("pdf-processing", body)]


@pytest.mark.parametrize(
    "command,expected",
    [
        ("cat data.csv > /skills/my-new-skill/SKILL.md", []),
        ("cat /skills/pdf-processing/SKILL.md", [("pdf-processing", "# PDF Processing\n1. Extract.")]),
    ],
)
def test_shell_tool_read_uses_the_same_rule_as_command_execution(command, expected):
    """A `bash` tool call and a Codex `command_execution` event are the same shell command."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "b1", "name": "bash", "input": {"command": command}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "b1", "content": [{"text": "# PDF Processing\n1. Extract."}]}}],
        },
    ]

    assert extract_selected_skills(messages) == expected


def test_sigpipe_exit_code_does_not_discard_a_read_body():
    """`cat SKILL.md | head -20` exits 141 once head closes the pipe, having printed the body."""
    body = "# PDF Processing\n1. Identify the path."

    invoked = extract_selected_skills(
        _shell_command("cat /skills/pdf-processing/SKILL.md | head -20", output=body, exit_code=141)
    )

    assert invoked == [("pdf-processing", body)]


def test_failing_read_is_still_discarded():
    """Only SIGPIPE is tolerated; a read that actually failed carries no body."""
    assert extract_selected_skills(_shell_command("cat /skills/pdf/SKILL.md", output="No such file", exit_code=1)) == []


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


def test_google_adk_function_call_shape():
    """Google ADK emits Gemini content parts and nests its payload under response/result."""
    messages = [
        {
            "role": "model",
            "content": [{"functionCall": {"id": "c1", "name": "list_skills", "args": {}}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "functionResponse": {
                        "id": "c1",
                        "name": "list_skills",
                        "response": {
                            "result": (
                                "<available_skills><skill><name>pdf-processing</name>"
                                "<description>Read PDFs.</description></skill></available_skills>"
                            )
                        },
                    }
                }
            ],
        },
        {
            "role": "model",
            "content": [{"functionCall": {"id": "c2", "name": "load_skill", "args": {"skill_name": "pdf-processing"}}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "functionResponse": {
                        "id": "c2",
                        "name": "load_skill",
                        "response": {"skill_name": "pdf-processing", "instructions": "## Phase 1\nRun pdfinfo."},
                    }
                }
            ],
        },
    ]

    assert parse_available_skills(messages) == [AvailableSkill("pdf-processing", "Read PDFs.")]
    invoked = extract_selected_skills(messages)
    assert [s.name for s in invoked] == ["pdf-processing"]
    assert invoked[0].body == "## Phase 1\nRun pdfinfo."


def test_load_acknowledgement_is_not_treated_as_a_body():
    """Gemini CLI's displayed output is a status line; scoring it as instructions would be wrong."""
    messages = [
        {"tool_name": "activate_skill", "parameters": {"name": "pdf-processing"}, "id": "g1"},
        {
            "type": "tool_result",
            "id": "g1",
            "output": "Skill activated. Resources loaded from pdf-processing/",
        },
    ]

    invoked = extract_selected_skills(messages)

    assert [s.name for s in invoked] == ["pdf-processing"]
    assert invoked[0].body is None


# The three strings the Strands AgentSkills plugin returns instead of a skill body. They arrive
# marked successful, because `@tool` reports any plain string return as `status="success"`.
_AGENT_SKILLS_NON_BODIES = [
    "Skill 'pdf-processing' not found. Available skills: spreadsheet-analysis, docx-editing",
    "Error: skill_name is required. Available skills: pdf-processing, spreadsheet-analysis",
    "Skill 'pdf-processing' activated (no instructions available).",
]


@pytest.mark.parametrize("result_text", _AGENT_SKILLS_NON_BODIES)
def test_agent_skills_status_string_is_not_a_body(result_text):
    """A refused or empty load carries no instructions, so the judge must not be handed one."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "skills", "input": {"skill_name": "pdf-processing"}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": result_text}]}}]},
    ]

    invoked = extract_selected_skills(messages)

    assert [s.name for s in invoked] == ["pdf-processing"]
    assert invoked[0].body is None


@pytest.mark.parametrize("result_text", _AGENT_SKILLS_NON_BODIES)
def test_agent_skills_status_string_is_not_a_body_on_the_session_path(result_text):
    """Same on the Session path: the plugin's string lands in `content` with `error` unset."""
    tool_span = ToolExecutionSpan(
        span_info=_span_info(),
        tool_call=ToolCall(name="skills", arguments={"skill_name": "pdf-processing"}, tool_call_id="tu-1"),
        tool_result=ToolResult(content=result_text),
    )

    invoked = extract_selected_skills(_session([tool_span]))

    assert [s.name for s in invoked] == ["pdf-processing"]
    assert invoked[0].body is None


def test_body_mentioning_a_missing_file_is_kept():
    """The load-error filter matches a whole status line, not the words wherever they appear."""
    body = "# PDF Processing\n\n1. If the skill file is not found, stop.\n2. Extract the text."
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "skills", "input": {"skill_name": "pdf-processing"}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": body}]}}]},
    ]

    assert extract_selected_skills(messages)[0].body == body


def test_skill_read_twice_keeps_the_fullest_body():
    """Repeated reads of one skill collapse, keeping the read that carried the whole file."""
    body = "---\nname: chart-builder\ndescription: Charts.\n---\n\n## Phase 1\nBuild the chart.\n"
    messages = [
        {
            "type": "command_execution",
            "command": "sed -n '1,3p' /skills/chart_builder/SKILL.md",
            "aggregated_output": "---\nname: chart-builder\ndescription: Charts.\n",
        },
        {
            "type": "command_execution",
            "command": "cat /skills/chart-builder/SKILL.md",
            "aggregated_output": body,
        },
    ]

    invoked = extract_selected_skills(messages)

    assert len(invoked) == 1
    assert invoked[0].name == "chart-builder"
    assert "## Phase 1" in (invoked[0].body or "")


def test_body_prefixed_by_an_acknowledgement_is_kept():
    """A status line ahead of the instructions must not discard the instructions with it."""
    body = "# PDF Processing\n\n1. Identify the PDF path.\n2. Extract the text."
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t1", "name": "skills", "input": {"skill_name": "pdf-processing"}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": f"Skill activated.\n\n{body}"}]}}],
        },
    ]

    invoked = extract_selected_skills(messages)

    assert [s.name for s in invoked] == ["pdf-processing"]
    assert "1. Identify the PDF path." in (invoked[0].body or "")


def test_skill_names_differing_only_by_a_dot_stay_separate():
    """`.` is legal in a skill name, so `data.clean` and `data-clean` are two skills."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "a", "name": "skills", "input": {"skill_name": "data.clean"}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "a", "content": [{"text": "# Dotted\n1. a"}]}}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "b", "name": "skills", "input": {"skill_name": "data-clean"}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "b", "content": [{"text": "# Hyphenated\n1. b\n2. c"}]}}],
        },
    ]

    assert [s.name for s in extract_selected_skills(messages)] == ["data.clean", "data-clean"]


def test_longer_unrelated_output_does_not_displace_a_recovered_body():
    """Only a superset of what was already recovered wins, so stray stdout cannot take over."""
    real_body = "---\nname: pdf-processing\n---\n# Real\n1. step"
    messages = [
        {
            "type": "command_execution",
            "command": "cat /skills/pdf-processing/SKILL.md",
            "exit_code": 0,
            "aggregated_output": real_body,
        },
        {
            "type": "command_execution",
            "command": "cat report.csv; ls -l /skills/pdf-processing/SKILL.md",
            "exit_code": 0,
            "aggregated_output": "col1,col2\n" + "x,y\n" * 40,
        },
    ]

    invoked = extract_selected_skills(messages)

    assert len(invoked) == 1
    assert invoked[0].body == real_body
