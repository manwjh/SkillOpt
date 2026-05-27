"""Learning-rate schedules for textual edit budgets."""

from __future__ import annotations

import math
from enum import Enum


class ScheduleType(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    AUTONOMOUS = "autonomous"


class EditBudgetScheduler:
    """Controls max edits per step (textual learning rate)."""

    def __init__(
        self,
        initial: int = 4,
        minimum: int = 2,
        schedule: ScheduleType = ScheduleType.COSINE,
        total_steps: int = 16,
    ) -> None:
        self.initial = initial
        self.minimum = minimum
        self.schedule = schedule
        self.total_steps = max(total_steps, 1)

    def get_budget(self, step: int, recent_accept_rate: float | None = None) -> int:
        if self.schedule == ScheduleType.CONSTANT:
            return self.initial

        if self.schedule == ScheduleType.AUTONOMOUS:
            # Shrink edits when acceptance rate is low; expand when improving
            if recent_accept_rate is None:
                return self.initial
            if recent_accept_rate < 0.2:
                return self.minimum
            if recent_accept_rate > 0.6:
                return self.initial
            mid = (self.initial + self.minimum) // 2
            return max(self.minimum, mid)

        progress = min(step / self.total_steps, 1.0)

        if self.schedule == ScheduleType.LINEAR:
            value = self.initial - (self.initial - self.minimum) * progress
        else:  # cosine
            value = self.minimum + 0.5 * (self.initial - self.minimum) * (
                1 + math.cos(math.pi * progress)
            )

        return max(self.minimum, round(value))
