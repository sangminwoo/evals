import pytest

from strands_evals.evaluators.deterministic import SkillInvoked
from strands_evals.types.evaluation import EvaluationData


def _case(invoked_skill: str | None):
    messages = []
    if invoked_skill:
        messages.append(
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "t", "name": "skills", "input": {"skill_name": invoked_skill}}}],
            }
        )
        messages.append(
            {"role": "user", "content": [{"toolResult": {"toolUseId": "t", "content": [{"text": "# body"}]}}]}
        )
    return EvaluationData(input="do pdf", actual_output="done", actual_trajectory=messages)


def test_skill_invoked_present():
    result = SkillInvoked(skill_name="pdf-processing").evaluate(_case("pdf-processing"))
    assert len(result) == 1
    assert result[0].score == 1.0
    assert result[0].test_pass is True


def test_skill_invoked_absent():
    result = SkillInvoked(skill_name="pdf-processing").evaluate(_case("other-skill"))
    assert result[0].score == 0.0
    assert result[0].test_pass is False


def test_skill_invoked_no_skill():
    result = SkillInvoked(skill_name="pdf-processing").evaluate(_case(None))
    assert result[0].score == 0.0
    assert result[0].test_pass is False


def test_refused_load_does_not_count_as_invoked():
    """The agent asked for the skill and the harness refused, so it was never used."""
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t", "name": "skills", "input": {"skill_name": "pdf-processing"}}}],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "t", "status": "error", "content": [{"text": "skill not found"}]}}
            ],
        },
    ]
    case = EvaluationData(input="do pdf", actual_output="done", actual_trajectory=messages)

    result = SkillInvoked(skill_name="pdf-processing").evaluate(case)

    assert result[0].score == 0.0
    assert result[0].test_pass is False
    # Distinguished from never reaching for the skill, which needs a different fix.
    assert result[0].reason == "skill 'pdf-processing' was requested but the load failed"


def test_skill_invoked_no_trajectory():
    case = EvaluationData(input="x", actual_output="y", actual_trajectory=None)
    result = SkillInvoked(skill_name="pdf-processing").evaluate(case)
    assert result[0].score == 0.0
    assert "no trajectory" in result[0].reason


@pytest.mark.asyncio
async def test_skill_invoked_async():
    result = await SkillInvoked(skill_name="pdf-processing").evaluate_async(_case("pdf-processing"))
    assert result[0].score == 1.0
