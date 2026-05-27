"""Direct chat harness — single-turn QA with exact-match scoring."""

from __future__ import annotations

import re

from skillopt.core.skill import SkillDocument
from skillopt.core.trajectory import Task, Trajectory
from skillopt.harness.base import HarnessAdapter
from skillopt.llm.client import LLMClient


from skillopt.scoring.registry import get_scorer


class DirectChatHarness(HarnessAdapter):
    """Execute tasks via a single chat completion with skill prepended."""

    def __init__(self, target_client: LLMClient) -> None:
        self.target_client = target_client

    def run(self, task: Task, skill: SkillDocument) -> Trajectory:
        system = skill.to_system_prompt()
        user = task.input

        try:
            response = self.target_client.complete(system, user)
            answer = response.content.strip()
            scorer_name = task.metadata.get("scorer", "exact")
            scorer = get_scorer(scorer_name)
            score = scorer(answer, task.expected, task.metadata)
            return Trajectory(
                task_id=task.id,
                skill_hash=skill.hash,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": answer},
                ],
                final_answer=answer,
                score=score,
                success=score >= 1.0,
            )
        except Exception as e:
            return Trajectory(
                task_id=task.id,
                skill_hash=skill.hash,
                final_answer="",
                score=0.0,
                success=False,
                error=str(e),
            )

    def evaluate_batch(self, tasks: list[Task], skill: SkillDocument) -> float:
        if not tasks:
            return 0.0
        trajectories = self.run_batch(tasks, skill)
        return sum(t.score for t in trajectories) / len(trajectories)

    @staticmethod
    def _score(answer: str, expected: str | None) -> float:
        if expected is None:
            return 1.0 if answer else 0.0

        normalized_answer = _normalize(answer)
        normalized_expected = _normalize(expected)

        if normalized_answer == normalized_expected:
            return 1.0
        if normalized_expected in normalized_answer:
            return 0.8
        return 0.0


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)
