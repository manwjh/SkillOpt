"""Epoch-wise slow update — dual rollout comparison (paper §3.6)."""

from __future__ import annotations

from dataclasses import dataclass, field

from skillopt.core.skill import SkillDocument
from skillopt.core.trajectory import Task, Trajectory
from skillopt.harness.base import HarnessAdapter


@dataclass
class TrajectoryPair:
    task_id: str
    prev: Trajectory
    curr: Trajectory

    @property
    def category(self) -> str:
        prev_ok = self.prev.success
        curr_ok = self.curr.success
        if not prev_ok and curr_ok:
            return "improvement"
        if prev_ok and not curr_ok:
            return "regression"
        if not prev_ok and not curr_ok:
            return "persistent_failure"
        return "stable_success"


@dataclass
class SlowUpdateEvidence:
    pairs: list[TrajectoryPair] = field(default_factory=list)

    def by_category(self) -> dict[str, list[TrajectoryPair]]:
        groups: dict[str, list[TrajectoryPair]] = {
            "improvement": [],
            "regression": [],
            "persistent_failure": [],
            "stable_success": [],
        }
        for pair in self.pairs:
            groups[pair.category].append(pair)
        return groups

    def summary(self) -> str:
        groups = self.by_category()
        lines = ["Cross-epoch trajectory comparison:"]
        for cat, items in groups.items():
            lines.append(f"\n## {cat} ({len(items)})")
            for p in items[:5]:
                lines.append(
                    f"  task={p.task_id} prev_score={p.prev.score:.2f} "
                    f"curr_score={p.curr.score:.2f}"
                )
                if p.prev.final_answer or p.curr.final_answer:
                    lines.append(
                        f"    prev_answer={p.prev.final_answer!r} "
                        f"curr_answer={p.curr.final_answer!r}"
                    )
        return "\n".join(lines)


def collect_slow_update_evidence(
    harness: HarnessAdapter,
    tasks: list[Task],
    prev_skill: SkillDocument,
    curr_skill: SkillDocument,
    sample_size: int,
) -> SlowUpdateEvidence:
    """Re-run the same training items under previous and current skill."""
    sample = tasks[:sample_size]
    prev_trajs = {t.task_id: t for t in harness.run_batch(sample, prev_skill)}
    curr_trajs = {t.task_id: t for t in harness.run_batch(sample, curr_skill)}

    pairs: list[TrajectoryPair] = []
    for task in sample:
        pairs.append(
            TrajectoryPair(
                task_id=task.id,
                prev=prev_trajs[task.id],
                curr=curr_trajs[task.id],
            )
        )
    return SlowUpdateEvidence(pairs=pairs)
