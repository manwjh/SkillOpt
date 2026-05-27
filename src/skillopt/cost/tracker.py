"""Cost tracking — tokens per point of gain."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostTracker:
    target_tokens: int = 0
    optimizer_tokens: int = 0
    steps: int = 0

    def add_target(self, tokens: int) -> None:
        self.target_tokens += tokens

    def add_optimizer(self, tokens: int) -> None:
        self.optimizer_tokens += tokens

    @property
    def total_tokens(self) -> int:
        return self.target_tokens + self.optimizer_tokens

    def cost_per_point(self, score_gain: float) -> float | None:
        if score_gain <= 0:
            return None
        return self.total_tokens / score_gain

    def to_dict(self) -> dict:
        return {
            "target_tokens": self.target_tokens,
            "optimizer_tokens": self.optimizer_tokens,
            "total_tokens": self.total_tokens,
            "steps": self.steps,
        }


class TrackingLLMClient:
    """Wrapper that tracks token usage."""

    def __init__(self, inner, tracker: CostTracker, role: str) -> None:
        self.inner = inner
        self.tracker = tracker
        self.role = role

    def complete(self, system: str, user: str):
        response = self.inner.complete(system, user)
        if self.role == "target":
            self.tracker.add_target(response.tokens_used)
        else:
            self.tracker.add_optimizer(response.tokens_used)
        return response
