"""Held-out validation gate for skill acceptance."""

from __future__ import annotations

from dataclasses import dataclass

from skillopt.core.skill import SkillDocument
from skillopt.harness.base import HarnessAdapter


@dataclass
class GateDecision:
    accepted: bool
    score: float
    previous_score: float
    reason: str

    @property
    def improvement(self) -> float:
        return self.score - self.previous_score


class ValidationGate:
    """Accept candidate skills only when selection score strictly improves."""

    def __init__(self, harness: HarnessAdapter, strict: bool = True) -> None:
        self.harness = harness
        self.strict = strict

    def evaluate(self, skill: SkillDocument, selection_tasks) -> float:
        return self.harness.evaluate_batch(selection_tasks, skill)

    def decide(
        self,
        candidate: SkillDocument,
        selection_tasks,
        current_score: float,
    ) -> GateDecision:
        score = self.evaluate(candidate, selection_tasks)

        if self.strict:
            accepted = score > current_score
        else:
            accepted = score >= current_score

        if accepted:
            reason = f"accepted: {current_score:.4f} → {score:.4f}"
        else:
            reason = f"rejected: {current_score:.4f} → {score:.4f} (no strict improvement)"

        return GateDecision(
            accepted=accepted,
            score=score,
            previous_score=current_score,
            reason=reason,
        )
