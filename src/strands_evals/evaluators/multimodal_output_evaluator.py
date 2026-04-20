"""Multimodal output evaluator using MLLM-as-a-Judge."""

import logging
import warnings

from strands.models.model import Model

from ..types.evaluation import EvaluationData, InputT, OutputT
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

    def _build_prompt(self, evaluation_case: EvaluationData[InputT, OutputT]) -> str | list:
        """Build the evaluation prompt for a multimodal test case.

        Composes a prompt that may include media content blocks (for MLLM-as-a-Judge)
        or text only (for LLM-as-a-Judge), and appends reference comparison suffix
        when expected_output is available.

        Args:
            evaluation_case: The test case with multimodal input and expected/actual outputs.

        Returns:
            Either a text prompt string or a list of content blocks with media.
        """
        if (
            self.include_media
            and isinstance(evaluation_case.input, MultimodalInput)
            and not evaluation_case.input.media
        ):
            warnings.warn(
                "include_media=True but no media found in input. Falling back to text-only evaluation.",
                UserWarning,
                stacklevel=3,
            )

        effective_rubric = self._select_rubric(evaluation_case)

        return compose_multimodal_test_prompt(
            evaluation_case=evaluation_case,
            rubric=effective_rubric,
            include_inputs=self.include_inputs,
            include_media=self.include_media,
        )
