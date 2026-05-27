"""Main SkillOpt optimization loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from skillopt.core.edit import Edit, EditEngine, EditResult
from skillopt.core.skill import SkillDocument
from skillopt.core.state import OptimizerState
from skillopt.core.trajectory import Task, Trajectory
from skillopt.gate.validation import ValidationGate
from skillopt.harness.base import HarnessAdapter
from skillopt.cost.tracker import CostTracker, TrackingLLMClient
from skillopt.llm.client import LLMClient
from skillopt.optimizer.reflection import ReflectionEngine
from skillopt.optimizer.scheduler import EditBudgetScheduler, ScheduleType
from skillopt.optimizer.slow_update import collect_slow_update_evidence


class EditMode(str, Enum):
    PATCH = "patch"
    REWRITE = "rewrite"


@dataclass
class OptimizationConfig:
    epochs: int = 4
    rollout_batch_size: int = 8
    rollout_accumulation_steps: int = 1
    reflection_minibatch_size: int = 4
    reflection_workers: int = 4
    merge_workers: int = 4
    reflection_refinement_rounds: int = 3
    merge_batch_size: int = 8
    learning_rate: int = 4
    learning_rate_min: int = 2
    schedule: ScheduleType = ScheduleType.COSINE
    edit_mode: EditMode = EditMode.PATCH
    slow_update_samples: int = 5
    enable_rejected_buffer: bool = True
    enable_slow_update: bool = True
    enable_meta_skill: bool = True
    output_dir: str = "artifacts"


@dataclass
class StepReport:
    epoch: int
    step: int
    budget: int
    proposed_edits: int
    applied_edits: int
    gate_accepted: bool
    selection_score: float
    best_score: float
    reason: str


@dataclass
class OptimizationResult:
    initial_score: float
    final_score: float
    best_score: float
    total_steps: int
    accepted_edits: int
    reports: list[StepReport] = field(default_factory=list)
    best_skill_path: str = ""
    cost: dict = field(default_factory=dict)


class SkillOptLoop:
    """Orchestrates the full skill optimization pipeline."""

    def __init__(
        self,
        harness: HarnessAdapter,
        optimizer_client: LLMClient,
        config: OptimizationConfig,
        workspace_root: str | None = None,
    ) -> None:
        self.harness = harness
        self.config = config
        self.cost_tracker = CostTracker()
        tracked_optimizer = TrackingLLMClient(optimizer_client, self.cost_tracker, "optimizer")
        ws_root = workspace_root or getattr(harness, "workspace_root", None)
        self.reflection = ReflectionEngine(
            tracked_optimizer,
            config.reflection_minibatch_size,
            config.reflection_workers,
            config.reflection_refinement_rounds,
            config.merge_batch_size,
            workspace_root=str(ws_root) if ws_root else None,
            merge_workers=config.merge_workers,
        )
        self.edit_engine = EditEngine()
        self.gate = ValidationGate(harness)
        if hasattr(harness, "target_client"):
            harness.target_client = TrackingLLMClient(
                harness.target_client, self.cost_tracker, "target"
            )
        total_steps = config.epochs * max(1, len(range(0, 100, config.rollout_batch_size)))
        self.scheduler = EditBudgetScheduler(
            initial=config.learning_rate,
            minimum=config.learning_rate_min,
            schedule=config.schedule,
            total_steps=max(total_steps, config.epochs * 4),
        )
        self._accept_history: list[bool] = []
        self._meta_accepted: list[str] = []
        self._meta_rejected: list[str] = []
        self._meta_persistent: list[str] = []

    def run(
        self,
        initial_skill: SkillDocument,
        train_tasks: list[Task],
        selection_tasks: list[Task],
        test_tasks: list[Task] | None = None,
    ) -> OptimizationResult:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        state = OptimizerState(current_skill=initial_skill.snapshot())
        state.best_skill = initial_skill.snapshot()

        baseline = self.gate.evaluate(state.current_skill, selection_tasks)
        state.current_selection_score = baseline
        state.best_selection_score = baseline

        reports: list[StepReport] = []
        accepted_edits = 0
        prev_epoch_skill = state.current_skill.snapshot()
        global_step = 0
        accumulated: list[Trajectory] = []
        accum_batches = 0

        for epoch in range(self.config.epochs):
            state.epoch = epoch
            if self.config.enable_rejected_buffer:
                state.rejected_buffer.clear()

            for batch_start in range(0, len(train_tasks), self.config.rollout_batch_size):
                batch = train_tasks[batch_start : batch_start + self.config.rollout_batch_size]
                if not batch:
                    break

                trajectories = self.harness.run_batch(batch, state.current_skill)
                accumulated.extend(trajectories)
                accum_batches += 1

                is_last_batch = batch_start + len(batch) >= len(train_tasks)
                need_more = (
                    accum_batches < self.config.rollout_accumulation_steps and not is_last_batch
                )
                if need_more:
                    continue

                reflect_batch = accumulated
                accumulated = []
                accum_batches = 0

                rejected_summary = (
                    state.rejected_summary() if self.config.enable_rejected_buffer else ""
                )
                meta = state.meta_skill if self.config.enable_meta_skill else ""

                proposed = self.reflection.reflect(
                    reflect_batch, state.current_skill, rejected_summary, meta
                )

                accept_rate = (
                    sum(self._accept_history[-10:]) / len(self._accept_history[-10:])
                    if self._accept_history
                    else None
                )
                budget = self.scheduler.get_budget(global_step, accept_rate)
                result, candidate_content = self._apply_candidate_edits(
                    state.current_skill, proposed, budget
                )
                candidate = SkillDocument(content=candidate_content)

                score_before = state.current_selection_score
                decision = self.gate.decide(candidate, selection_tasks, score_before)

                self._write_edit_report(
                    output_dir,
                    global_step,
                    epoch,
                    proposed,
                    result,
                    decision.accepted,
                    decision.reason,
                )

                if decision.accepted:
                    candidate.version = state.current_skill.version + 1
                    state.accept_candidate(candidate, decision.score)
                    accepted_edits += len(result.applied)
                    self._update_meta(state, result.applied, accepted=True)
                    self._accept_history.append(True)
                else:
                    if self.config.enable_rejected_buffer:
                        state.record_rejection(
                            result.applied, score_before, decision.score
                        )
                    self._update_meta(state, result.applied, accepted=False)
                    self._track_persistent_failures(reflect_batch)
                    self._accept_history.append(False)

                reports.append(
                    StepReport(
                        epoch=epoch,
                        step=global_step,
                        budget=budget,
                        proposed_edits=len(proposed),
                        applied_edits=len(result.applied),
                        gate_accepted=decision.accepted,
                        selection_score=decision.score,
                        best_score=state.best_selection_score,
                        reason=decision.reason,
                    )
                )

                self._save_checkpoint(output_dir, state, reports)
                global_step += 1
                state.step = global_step

            if self.config.enable_slow_update and epoch > 0:
                self._run_slow_update(
                    state, prev_epoch_skill, train_tasks, selection_tasks
                )

            prev_epoch_skill = state.current_skill.snapshot()

        best_path = str(output_dir / "best_skill.md")
        state.export_best(best_path)

        test_score = 0.0
        if test_tasks:
            test_score = self.harness.evaluate_batch(test_tasks, state.best_skill)

        result = OptimizationResult(
            initial_score=baseline,
            final_score=state.current_selection_score,
            best_score=state.best_selection_score,
            total_steps=global_step,
            accepted_edits=accepted_edits,
            reports=reports,
            best_skill_path=best_path,
            cost=self.cost_tracker.to_dict(),
        )

        score_gain = state.best_selection_score - baseline
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "initial_selection_score": baseline,
            "final_selection_score": state.current_selection_score,
            "best_selection_score": state.best_selection_score,
            "test_score": test_score,
            "total_steps": global_step,
            "accepted_edits": accepted_edits,
            "best_skill_tokens": state.best_skill.token_estimate,
            "cost": self.cost_tracker.to_dict(),
            "cost_per_point": self.cost_tracker.cost_per_point(score_gain),
            "edit_mode": self.config.edit_mode.value,
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return result

    def _apply_candidate_edits(
        self,
        skill: SkillDocument,
        proposed: list[Edit],
        budget: int,
    ) -> tuple[EditResult, str]:
        if self.config.edit_mode == EditMode.REWRITE and proposed:
            selected = sorted(proposed, key=lambda e: e.priority, reverse=True)[:budget]
            rewritten = self.reflection.rewrite_skill(skill, selected)
            return EditResult(applied=selected, new_content=rewritten), rewritten

        result = self.edit_engine.apply_edits(skill, proposed, budget)
        return result, result.new_content

    def _run_slow_update(
        self,
        state: OptimizerState,
        prev_skill: SkillDocument,
        train_tasks: list[Task],
        selection_tasks: list[Task],
    ) -> None:
        evidence = collect_slow_update_evidence(
            self.harness,
            train_tasks,
            prev_skill,
            state.current_skill,
            self.config.slow_update_samples,
        )
        groups = evidence.by_category()
        for pair in groups.get("persistent_failure", []):
            self._meta_persistent.append(f"task={pair.task_id} still failing")

        guidance = self.reflection.slow_update(prev_skill, state.current_skill, evidence)
        candidate = state.current_skill.snapshot()
        candidate.set_slow_update(guidance)

        decision = self.gate.decide(
            candidate, selection_tasks, state.current_selection_score
        )
        if decision.accepted:
            state.accept_candidate(candidate, decision.score)

    def _update_meta(self, state: OptimizerState, edits: list[Edit], accepted: bool) -> None:
        if not self.config.enable_meta_skill:
            return
        for e in edits[:3]:
            snippet = f"{e.rationale}: {e.content[:80]}"
            if accepted:
                self._meta_accepted.append(snippet)
            else:
                self._meta_rejected.append(snippet)

        if len(self._meta_accepted) + len(self._meta_rejected) >= 3:
            state.meta_skill = self.reflection.build_meta_skill(
                self._meta_accepted,
                self._meta_rejected,
                self._meta_persistent,
            )

    def _track_persistent_failures(self, trajectories: list[Trajectory]) -> None:
        for t in trajectories:
            if not t.success:
                self._meta_persistent.append(f"task={t.task_id}: {t.final_answer[:60]}")

    @staticmethod
    def _write_edit_report(
        output_dir: Path,
        step: int,
        epoch: int,
        proposed: list[Edit],
        result: EditResult,
        accepted: bool,
        reason: str,
    ) -> None:
        applied_contents = {e.content.strip() for e in result.applied}
        report = {
            "step": step,
            "epoch": epoch,
            "gate_accepted": accepted,
            "reason": reason,
            "edits": [
                {
                    "action": e.action.value,
                    "content": e.content,
                    "target": e.target,
                    "rationale": e.rationale,
                    "priority": e.priority,
                    "status": "applied" if e.content.strip() in applied_contents else "skipped",
                }
                for e in proposed
            ],
            "applied_count": len(result.applied),
            "skipped_count": len(result.skipped),
        }
        path = output_dir / f"edit_apply_report_step{step:03d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Also maintain latest aggregate
        agg_path = output_dir / "edit_apply_report.json"
        history = []
        if agg_path.exists():
            with open(agg_path, encoding="utf-8") as f:
                history = json.load(f)
        history.append(report)
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _save_checkpoint(
        output_dir: Path, state: OptimizerState, reports: list[StepReport]
    ) -> None:
        state.current_skill.save(str(output_dir / "current_skill.md"))
        log = [
            {
                "epoch": r.epoch,
                "step": r.step,
                "budget": r.budget,
                "proposed": r.proposed_edits,
                "applied": r.applied_edits,
                "accepted": r.gate_accepted,
                "score": r.selection_score,
                "best": r.best_score,
                "reason": r.reason,
            }
            for r in reports
        ]
        with open(output_dir / "optimization_log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
