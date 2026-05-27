from __future__ import annotations

from dataclasses import dataclass, field

from skillopt.core.edit import Edit
from skillopt.core.skill import SkillDocument


@dataclass
class RejectedEdit:
    edits: list[Edit]
    score_before: float
    score_after: float
    reason: str = "validation gate rejected"

    @property
    def score_delta(self) -> float:
        return self.score_after - self.score_before


@dataclass
class OptimizerState:
    """Full optimizer state for the training loop."""

    current_skill: SkillDocument
    best_skill: SkillDocument | None = None
    best_selection_score: float = 0.0
    current_selection_score: float = 0.0
    epoch: int = 0
    step: int = 0
    rejected_buffer: list[RejectedEdit] = field(default_factory=list)
    meta_skill: str = ""
    seen_hashes: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.best_skill is None:
            self.best_skill = self.current_skill.snapshot()
        self.seen_hashes.add(self.current_skill.hash)

    def accept_candidate(
        self, candidate: SkillDocument, selection_score: float
    ) -> bool:
        if selection_score <= self.current_selection_score:
            return False

        self.current_skill = candidate
        self.current_selection_score = selection_score
        self.seen_hashes.add(candidate.hash)

        if selection_score > self.best_selection_score:
            self.best_skill = candidate.snapshot()
            self.best_selection_score = selection_score
        return True

    def record_rejection(
        self, edits: list[Edit], score_before: float, score_after: float
    ) -> None:
        self.rejected_buffer.append(
            RejectedEdit(
                edits=edits,
                score_before=score_before,
                score_after=score_after,
            )
        )

    def rejected_summary(self) -> str:
        if not self.rejected_buffer:
            return ""
        lines = ["Previously rejected edits:"]
        for r in self.rejected_buffer[-10:]:
            edit_desc = "; ".join(str(e) for e in r.edits[:3])
            lines.append(
                f"  - {edit_desc} (score {r.score_before:.3f} → {r.score_after:.3f})"
            )
        return "\n".join(lines)

    def export_best(self, path: str) -> None:
        self.best_skill.save(path)
