from ...extractors.skills import extract_selected_skills
from ...types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..evaluator import Evaluator


class SkillInvoked(Evaluator[InputT, OutputT]):
    """Checks if a specific skill was invoked in the trajectory."""

    def __init__(self, skill_name: str, name: str | None = None):
        super().__init__(name=name)
        self.skill_name = skill_name

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        trajectory = evaluation_case.actual_trajectory
        if trajectory is None:
            return [EvaluationOutput(score=0.0, test_pass=False, reason="no trajectory provided")]

        selected = extract_selected_skills(trajectory)
        # A refused load does not count as invoked: the agent never received the skill, so an
        # assertion that it was used is false. It is reported separately from no attempt at all,
        # since the two call for different fixes (a broken harness, or a prompt that never
        # surfaced the skill).
        found = self.skill_name in {s.name for s in selected if s.status == "loaded"}
        refused = self.skill_name in {s.name for s in selected if s.status == "failed"}
        if found:
            reason = f"skill '{self.skill_name}' was invoked"
        elif refused:
            reason = f"skill '{self.skill_name}' was requested but the load failed"
        else:
            reason = f"skill '{self.skill_name}' was not invoked"
        return [
            EvaluationOutput(
                score=1.0 if found else 0.0,
                test_pass=found,
                reason=reason,
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)
