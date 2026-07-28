from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ..extractors.skills import extract_selected_skills, parse_available_skills, serialize_trajectory
from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from .evaluator import Evaluator
from .prompt_templates.skill_selection_accuracy import get_template


class SkillSelectionScore(str, Enum):
    """Binary skill selection appropriateness ratings."""

    YES = "Yes"
    NO = "No"


class SkillSelectionRating(BaseModel):
    """Structured output for skill selection accuracy evaluation."""

    reasoning: str = Field(description="Step by step reasoning to derive the final score")
    score: SkillSelectionScore = Field(description="Score should be one of 'Yes' or 'No'")


class SkillSelectionAccuracyEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates whether each skill the agent invoked was an appropriate selection.

    Returns one `EvaluationOutput` per invoked skill, or a single row judging the abstention
    when no skill was invoked.
    """

    _score_mapping = {
        SkillSelectionScore.YES: 1.0,
        SkillSelectionScore.NO: 0.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: Model | str | None = None,
        system_prompt: str | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model

    def _available_str(self, evaluation_case: EvaluationData[InputT, OutputT]) -> str:
        available = parse_available_skills(evaluation_case.actual_trajectory)
        return "\n".join(f"- {s.name}: {s.description}" for s in available) if available else "(none listed)"

    def _case_context(self, evaluation_case: EvaluationData[InputT, OutputT]) -> tuple[str, str]:
        """The two halves of the prompt that do not depend on which decision is being judged.

        Built once per case: the skill catalog and the serialized trajectory are the same for
        every invoked skill, and serializing a long trajectory once per skill is wasted work.
        """
        head = f"## Task\n{evaluation_case.input}\n\n## Available skills\n{self._available_str(evaluation_case)}\n\n"
        tail = (
            f"## Agent trajectory\n{serialize_trajectory(evaluation_case.actual_trajectory)}\n\n"
            f"## Agent's final response\n{evaluation_case.actual_output}"
        )
        return head, tail

    @staticmethod
    def _prompt_for(context: tuple[str, str], focus_skill: str | None) -> str:
        """Prompt judging one focal decision: invoking `focus_skill`, or abstaining if None."""
        head, tail = context
        if focus_skill is not None:
            decision = f"## Decision under evaluation\nThe agent invoked the skill: {focus_skill}\n"
        else:
            decision = "## Decision under evaluation\nThe agent invoked no skill (abstained).\n"
        return f"{head}{decision}\n{tail}"

    def _build_prompt(
        self,
        evaluation_case: EvaluationData[InputT, OutputT],
        focus_skill: str | None,
    ) -> str:
        """Prompt judging one focal decision: invoking `focus_skill`, or abstaining if None."""
        return self._prompt_for(self._case_context(evaluation_case), focus_skill)

    def _rating_to_output(self, rating: SkillSelectionRating, label: str) -> EvaluationOutput:
        normalized_score = self._score_mapping[rating.score]
        return EvaluationOutput(
            score=normalized_score,
            test_pass=normalized_score == 1.0,
            reason=rating.reasoning,
            label=label,
        )

    def _new_judge(self) -> Agent:
        """A fresh judge per decision.

        Each skill is judged independently, so reusing one `Agent` across the loop would both
        carry the previous verdicts into the next prompt as conversation history and resend the
        whole trajectory on top of it, growing every request.
        """
        return Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)

    def _judge(self, prompt: str) -> SkillSelectionRating:
        result = self._new_judge()(prompt, structured_output_model=SkillSelectionRating)
        return cast(SkillSelectionRating, result.structured_output)

    async def _judge_async(self, prompt: str) -> SkillSelectionRating:
        result = await self._new_judge().invoke_async(prompt, structured_output_model=SkillSelectionRating)
        return cast(SkillSelectionRating, result.structured_output)

    @staticmethod
    def _missing_trajectory_row() -> EvaluationOutput:
        """A missing trajectory is absent data, not a correct abstention, so it is not scored."""
        return EvaluationOutput(score=0.0, test_pass=False, reason="no trajectory provided", label="not_applicable")

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        if evaluation_case.actual_trajectory is None:
            return [self._missing_trajectory_row()]
        invoked = extract_selected_skills(evaluation_case.actual_trajectory)
        context = self._case_context(evaluation_case)
        if not invoked:
            rating = self._judge(self._prompt_for(context, None))
            return [self._rating_to_output(rating, label="abstained")]
        results = []
        for skill in invoked:
            rating = self._judge(self._prompt_for(context, skill.name))
            results.append(self._rating_to_output(rating, label=skill.name))
        return results

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        if evaluation_case.actual_trajectory is None:
            return [self._missing_trajectory_row()]
        invoked = extract_selected_skills(evaluation_case.actual_trajectory)
        context = self._case_context(evaluation_case)
        if not invoked:
            rating = await self._judge_async(self._prompt_for(context, None))
            return [self._rating_to_output(rating, label="abstained")]
        results = []
        for skill in invoked:
            rating = await self._judge_async(self._prompt_for(context, skill.name))
            results.append(self._rating_to_output(rating, label=skill.name))
        return results
