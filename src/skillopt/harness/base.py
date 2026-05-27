"""Harness adapter interface and direct-chat implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from skillopt.core.skill import SkillDocument
from skillopt.core.trajectory import Task, Trajectory


class HarnessAdapter(ABC):
    """Bridge between SkillOpt and task execution environments."""

    @abstractmethod
    def run(self, task: Task, skill: SkillDocument) -> Trajectory: ...

    @abstractmethod
    def evaluate_batch(self, tasks: list[Task], skill: SkillDocument) -> float: ...

    def run_batch(
        self, tasks: list[Task], skill: SkillDocument
    ) -> list[Trajectory]:
        return [self.run(task, skill) for task in tasks]
