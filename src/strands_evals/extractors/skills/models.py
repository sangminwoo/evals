"""The types the skill extractors return."""

from __future__ import annotations

from typing import Literal, NamedTuple


class AvailableSkill(NamedTuple):
    """A skill exposed to the agent at runtime."""

    name: str
    description: str


class InvokedSkill(NamedTuple):
    """A skill the agent selected during the run, whether or not the load succeeded."""

    name: str
    body: str | None  # SKILL.md text if captured from the trajectory, else None
    # "failed" means the harness refused the load (unknown skill, sandbox error). Kept rather than
    # dropped because a refused load and no attempt at all are different runs: the agent that asked
    # for the right skill and was refused made a correct selection, and reporting it as an
    # abstention credits or blames the wrong decision.
    status: Literal["loaded", "failed"] = "loaded"
    # The harness's own refusal message, on a failed load. Which refusal it was decides what to
    # fix: "Skill 'pdf-procesing' not found. Available skills: pdf-processing" is a misspelled
    # name in the agent's call, while "Available skills: (none)" is a harness that mounted no
    # skills at all. Collapsing both into "the load failed" hides that difference from whoever
    # reads the result.
    error: str | None = None
