from enum import Enum
from typing import cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ..extractors.skills import InvokedSkill, extract_selected_skills, parse_available_skills, serialize_trajectory
from ..types.evaluation import NOT_APPLICABLE, EvaluationData, EvaluationOutput, InputT, OutputT
from .evaluator import Evaluator
from .prompt_templates.skill_selection_accuracy import get_template

_ABSTAINED = "abstained"


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
        # A case with nothing to select from contributes a placeholder 0.0 row; averaging it in
        # would report a run that had no decision to make as a failed one.
        self.aggregator = self._aggregate_dropping_na

    def _available_str(self, evaluation_case: EvaluationData[InputT, OutputT]) -> str:
        available = parse_available_skills(evaluation_case.actual_trajectory)
        return "\n".join(f"- {s.name}: {s.description}" for s in available) if available else "(none listed)"

    def _has_catalog(self, evaluation_case: EvaluationData[InputT, OutputT]) -> bool:
        return bool(parse_available_skills(evaluation_case.actual_trajectory))

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
    def _prompt_for(context: tuple[str, str], focus_skill: InvokedSkill | None) -> str:
        """Prompt judging one focal decision: invoking `focus_skill`, or abstaining if None."""
        head, tail = context
        if focus_skill is None:
            decision = "## Decision under evaluation\nThe agent invoked no skill (abstained).\n"
        else:
            decision = f"## Decision under evaluation\nThe agent invoked the skill: {focus_skill.name}\n"
            if focus_skill.status == "failed":
                # What is being judged is the choice, not the outcome. Without this the judge sees
                # an error in the trajectory and marks a correct selection wrong for failing.
                decision += (
                    "The harness refused the load, so the agent never received the skill. "
                    "Judge whether asking for this skill was the right choice, not whether it worked.\n"
                )
        return f"{head}{decision}\n{tail}"

    def _build_prompt(
        self,
        evaluation_case: EvaluationData[InputT, OutputT],
        focus_skill: InvokedSkill | None,
    ) -> str:
        """Prompt judging one focal decision: invoking `focus_skill`, or abstaining if None."""
        return self._prompt_for(self._case_context(evaluation_case), focus_skill)

    def _rating_to_output(self, rating: SkillSelectionRating, decision: str) -> EvaluationOutput:
        """One row for one decision.

        `label` carries the judge's rating, as every other judge in the framework does, so a
        consumer reading labels across evaluators sees verdicts rather than a mix of verdicts and
        skill names. Which decision the row is about is named in `reason` instead, since a case
        with several invoked skills produces several rows.
        """
        normalized_score = self._score_mapping[rating.score]
        return EvaluationOutput(
            score=normalized_score,
            test_pass=normalized_score == 1.0,
            reason=f"{decision}: {rating.reasoning}",
            label=rating.score.value,
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
    def _not_applicable_row(reason: str, test_pass: bool) -> EvaluationOutput:
        return EvaluationOutput(score=0.0, test_pass=test_pass, reason=reason, label=NOT_APPLICABLE)

    @classmethod
    def _missing_trajectory_row(cls) -> EvaluationOutput:
        """A missing trajectory is absent data, not a correct abstention, so it is not scored."""
        return cls._not_applicable_row("no trajectory provided", test_pass=False)

    @classmethod
    def _no_catalog_row(cls) -> EvaluationOutput:
        """No advertised skills and no invocation means there was no selection decision to make.

        Judging it either way is wrong: a "Yes" credits the agent for declining an offer it never
        received, and a "No" penalizes it for the same. This also covers the case where skills were
        on offer but the trajectory did not carry the catalog, where the right verdict is unknowable
        rather than favorable.
        """
        return cls._not_applicable_row("no skills were available to select from", test_pass=True)

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        if evaluation_case.actual_trajectory is None:
            return [self._missing_trajectory_row()]
        invoked = extract_selected_skills(evaluation_case.actual_trajectory)
        if not invoked and not self._has_catalog(evaluation_case):
            return [self._no_catalog_row()]
        context = self._case_context(evaluation_case)
        if not invoked:
            rating = self._judge(self._prompt_for(context, None))
            return [self._rating_to_output(rating, decision=_ABSTAINED)]
        results = []
        for skill in invoked:
            rating = self._judge(self._prompt_for(context, skill))
            results.append(self._rating_to_output(rating, decision=skill.name))
        return results

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        if evaluation_case.actual_trajectory is None:
            return [self._missing_trajectory_row()]
        invoked = extract_selected_skills(evaluation_case.actual_trajectory)
        if not invoked and not self._has_catalog(evaluation_case):
            return [self._no_catalog_row()]
        context = self._case_context(evaluation_case)
        if not invoked:
            rating = await self._judge_async(self._prompt_for(context, None))
            return [self._rating_to_output(rating, decision=_ABSTAINED)]
        results = []
        for skill in invoked:
            rating = await self._judge_async(self._prompt_for(context, skill))
            results.append(self._rating_to_output(rating, decision=skill.name))
        return results
