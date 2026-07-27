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

        invoked_names = {s.name for s in extract_selected_skills(trajectory)}
        found = self.skill_name in invoked_names
        return [
            EvaluationOutput(
                score=1.0 if found else 0.0,
                test_pass=found,
                reason=f"skill '{self.skill_name}' {'was invoked' if found else 'was not invoked'}",
            )
        ]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        return self.evaluate(evaluation_case)
