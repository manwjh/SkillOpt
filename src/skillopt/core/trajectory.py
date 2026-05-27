from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Task:
    """A single evaluation task."""

    id: str
    input: str
    expected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Execution trace from running a task with a skill."""

    task_id: str
    skill_hash: str
    messages: list[dict[str, str]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    score: float = 0.0
    success: bool = False
    error: str | None = None
    raw_trace: str = ""

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILURE"
        parts = [
            f"[{status}] task={self.task_id} score={self.score:.2f}",
            f"answer={self.final_answer!r}",
        ]
        if self.error:
            parts.append(f"error={self.error}")
        if self.tool_calls:
            parts.append(f"tool_calls={self.tool_calls[:8]}")
        if self.raw_trace:
            parts.append(f"execution_trace:\n{self.raw_trace[:1200]}")
        return "\n".join(parts)
