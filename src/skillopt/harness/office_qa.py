"""OfficeQA-style harness with oracle document context."""

from __future__ import annotations

import re

from skillopt.core.skill import SkillDocument
from skillopt.core.trajectory import Task, Trajectory
from skillopt.harness.base import HarnessAdapter
from skillopt.harness.direct_chat import DirectChatHarness, _normalize
from skillopt.llm.client import LLMClient


class OfficeQAHarness(HarnessAdapter):
    """Multi-field document QA with strict answer formatting."""

    harness_name = "office_qa"

    def __init__(self, target_client: LLMClient) -> None:
        self.target_client = target_client

    def run(self, task: Task, skill: SkillDocument) -> Trajectory:
        doc = task.metadata.get("document", "")
        user = (
            f"Task ID: {task.id}\n\n"
            f"## Document\n{doc}\n\n"
            f"## Question\n{task.input}\n\n"
            "Reply with ONLY the requested value. No labels, no explanation."
        )
        system = skill.to_system_prompt()

        try:
            response = self.target_client.complete(system, user)
            answer = response.content.strip()
            score = self._score(answer, task.expected, task.metadata)
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
                raw_trace=f"office_qa task={task.id} answer={answer!r} expected={task.expected!r}",
            )
        except Exception as e:
            return Trajectory(
                task_id=task.id,
                skill_hash=skill.hash,
                score=0.0,
                success=False,
                error=str(e),
            )

    def evaluate_batch(self, tasks: list[Task], skill: SkillDocument) -> float:
        if not tasks:
            return 0.0
        trajs = self.run_batch(tasks, skill)
        return sum(t.score for t in trajs) / len(trajs)

    @staticmethod
    def _score(answer: str, expected: str | None, metadata: dict) -> float:
        if expected is None:
            return 1.0 if answer.strip() else 0.0

        answer_clean = answer.strip()
        # Strip common label prefixes
        answer_clean = re.sub(r"^(answer|result|value)\s*[:=]\s*", "", answer_clean, flags=re.I)

        fmt = metadata.get("format", "exact")
        if fmt == "number":
            nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", answer_clean.replace(",", ""))
            if nums and nums[0].replace(",", "") == str(expected).replace(",", ""):
                return 1.0
            return 0.0
        if fmt == "date":
            return 1.0 if _normalize(answer_clean) == _normalize(expected) else 0.0

        return DirectChatHarness._score(answer_clean, expected)
