"""Multimodal output evaluator using MLLM-as-a-Judge."""

import logging
import warnings
from typing import cast

from strands import Agent
from strands.models.model import Model
from strands.types.content import ContentBlock

from ..types.evaluation import EvaluationData, EvaluationOutput, InputT, OutputT
from ..types.multimodal import MultimodalInput
from .output_evaluator import OutputEvaluator
from .prompt_templates.multimodal_case_prompt_template import compose_multimodal_test_prompt
from .prompt_templates.multimodal_judge_system_prompt import MLLM_JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class MultimodalOutputEvaluator(OutputEvaluator[InputT, OutputT]):
    """MLLM-as-a-Judge evaluator for multimodal tasks.

    Extends OutputEvaluator to handle multimodal inputs containing media (images,
    documents, etc.) and text. Supports both MLLM-as-a-Judge (with media) and
    LLM-as-a-Judge (text-only) modes, as well as reference-based and reference-free
    evaluation.

    When ``expected_output`` is provided in the evaluation case, a reference comparison
    suffix is automatically appended to the rubric to enable reference-based evaluation.

    Attributes:
        rubric: Evaluation criteria (e.g., correctness, faithfulness rubric)
        model: Model instance or model ID string for the MLLM judge
        include_media: Whether to include the media in the judge prompt (MLLM vs LLM mode)
        include_inputs: Whether to include the original instruction in evaluation
        system_prompt: System prompt for the MLLM judge
        reference_suffix: Text appended to rubric when expected_output is provided
    """

    DEFAULT_REFERENCE_SUFFIX = """

REFERENCE COMPARISON:
- Compare the response against the Oracle reference answer above.
- The reference is the gold standard. Use discrepancies as evidence for your judgment."""

    def __init__(
        self,
        rubric: str,
        model: Model | str | None = None,
        include_media: bool = True,
        include_inputs: bool = True,
        system_prompt: str | None = None,
        reference_suffix: str | None = None,
    ):
        super().__init__(
            rubric=rubric,
            model=model,
            system_prompt=system_prompt if system_prompt is not None else MLLM_JUDGE_SYSTEM_PROMPT,
            include_inputs=include_inputs,
        )
        self.include_media = include_media
        self.reference_suffix = reference_suffix if reference_suffix is not None else self.DEFAULT_REFERENCE_SUFFIX

    def _select_rubric(self, evaluation_case: EvaluationData[InputT, OutputT]) -> str:
        """Select the appropriate rubric based on whether a reference output is available.

        When expected_output is present, appends the reference comparison suffix to the rubric.
        """
        if evaluation_case.expected_output is not None:
            return self.rubric + self.reference_suffix
        return self.rubric

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Evaluate a multimodal test case.

        Automatically appends reference comparison suffix to rubric when
        ``expected_output`` is provided, otherwise uses the rubric as-is.

        Args:
            evaluation_case: Test case with multimodal input and expected/actual outputs.

        Returns:
            List containing a single EvaluationOutput with score, pass/fail, and reasoning.
        """
        if self.include_media and isinstance(evaluation_case.input, MultimodalInput) and not evaluation_case.input.media:
            warnings.warn(
                "include_media=True but no media found in input. Falling back to text-only evaluation.",
                UserWarning,
                stacklevel=2,
            )

        effective_rubric = self._select_rubric(evaluation_case)

        evaluation_prompt = compose_multimodal_test_prompt(
            evaluation_case=evaluation_case,
            rubric=effective_rubric,
            include_inputs=self.include_inputs,
            include_media=self.include_media,
        )

        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        prompt: str | list[ContentBlock] = cast(str | list[ContentBlock], evaluation_prompt)
        result = evaluator_agent(prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        """Evaluate a multimodal test case asynchronously.

        Automatically appends reference comparison suffix to rubric when
        ``expected_output`` is provided, otherwise uses the rubric as-is.

        Args:
            evaluation_case: Test case with multimodal input and expected/actual outputs.

        Returns:
            List containing a single EvaluationOutput with score, pass/fail, and reasoning.
        """
        if self.include_media and isinstance(evaluation_case.input, MultimodalInput) and not evaluation_case.input.media:
            warnings.warn(
                "include_media=True but no media found in input. Falling back to text-only evaluation.",
                UserWarning,
                stacklevel=2,
            )

        effective_rubric = self._select_rubric(evaluation_case)

        evaluation_prompt = compose_multimodal_test_prompt(
            evaluation_case=evaluation_case,
            rubric=effective_rubric,
            include_inputs=self.include_inputs,
            include_media=self.include_media,
        )

        evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
        prompt: str | list[ContentBlock] = cast(str | list[ContentBlock], evaluation_prompt)
        result = await evaluator_agent.invoke_async(prompt, structured_output_model=EvaluationOutput)
        return [cast(EvaluationOutput, result.structured_output)]
