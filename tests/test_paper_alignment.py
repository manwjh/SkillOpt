"""Tests for paper-aligned optimizer components."""

from skillopt.core.skill import SkillDocument
from skillopt.core.trajectory import Task, Trajectory
from skillopt.harness.direct_chat import DirectChatHarness
from skillopt.llm.client import MockLLMClient
from skillopt.optimizer.reflection import ReflectionEngine
from skillopt.optimizer.scheduler import EditBudgetScheduler, ScheduleType
from skillopt.optimizer.slow_update import TrajectoryPair, collect_slow_update_evidence
from skillopt.scoring.registry import exact_match, spreadsheet_scorer


class _FlipHarness:
    """Runs worse with empty skill, better with enriched skill."""

    def run(self, task, skill):
        ok = "capital" in skill.content.lower() or "arithmetic" in skill.content.lower()
        if task.expected == "Paris":
            score = 1.0 if ok else 0.0
            answer = "Paris" if ok else "Lyon"
        else:
            score = 1.0 if ok else 0.0
            answer = "4" if ok else "5"
        return Trajectory(
            task_id=task.id,
            skill_hash=skill.hash,
            final_answer=answer,
            score=score,
            success=score >= 1.0,
        )

    def run_batch(self, tasks, skill):
        return [self.run(t, skill) for t in tasks]

    def evaluate_batch(self, tasks, skill):
        trajs = self.run_batch(tasks, skill)
        return sum(t.score for t in trajs) / len(trajs)


def test_trajectory_pair_categories():
    assert TrajectoryPair(
        "t1",
        Trajectory(task_id="t1", skill_hash="a", success=False, score=0.0),
        Trajectory(task_id="t1", skill_hash="b", success=True, score=1.0),
    ).category == "improvement"

    assert TrajectoryPair(
        "t2",
        Trajectory(task_id="t2", skill_hash="a", success=True, score=1.0),
        Trajectory(task_id="t2", skill_hash="b", success=False, score=0.0),
    ).category == "regression"


def test_collect_slow_update_evidence_dual_rollout():
    harness = _FlipHarness()
    tasks = [
        Task(id="t1", input="capital?", expected="Paris"),
        Task(id="t2", input="2+2?", expected="4"),
    ]
    weak = SkillDocument(content="Answer.")
    strong = SkillDocument(content="For geography use capital. For arithmetic compute.")

    evidence = collect_slow_update_evidence(harness, tasks, weak, strong, sample_size=2)
    groups = evidence.by_category()

    assert len(evidence.pairs) == 2
    assert len(groups["improvement"]) >= 1


def test_autonomous_scheduler():
    sched = EditBudgetScheduler(
        initial=4, minimum=2, schedule=ScheduleType.AUTONOMOUS, total_steps=10
    )
    assert sched.get_budget(0, recent_accept_rate=0.0) == 2
    assert sched.get_budget(0, recent_accept_rate=0.8) == 4


def test_spreadsheet_scorer():
    assert spreadsheet_scorer(
        "use static evaluated materialized computed value", "static_values", {}
    ) == 1.0
    assert spreadsheet_scorer("hello", "static_values", {}) == 0.0


def test_reflection_hierarchical_merge_with_mock():
    engine = ReflectionEngine(MockLLMClient(), minibatch_size=1, workers=1, refinement_rounds=1)
    trajs = [
        Trajectory(task_id="f1", skill_hash="x", success=False, score=0.0, final_answer="Lyon"),
    ]
    edits = engine.reflect(trajs, SkillDocument(content="initial"))
    assert len(edits) >= 1


def test_scorer_in_harness():
    harness = DirectChatHarness(MockLLMClient())
    task = Task(
        id="s1",
        input="Fill static values",
        expected="static_values",
        metadata={"scorer": "spreadsheet"},
    )
    skill = SkillDocument(content="Inspect workbook and write static evaluated values.")
    traj = harness.run(task, skill)
    assert traj.score >= 0.5
