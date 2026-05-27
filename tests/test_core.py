"""Unit tests for SkillOpt core components."""

from skillopt.core.edit import Edit, EditAction, EditEngine
from skillopt.core.skill import SkillDocument
from skillopt.core.state import OptimizerState
from skillopt.core.trajectory import Task, Trajectory
from skillopt.harness.direct_chat import DirectChatHarness, _normalize
from skillopt.llm.client import MockLLMClient, parse_edits_from_response
from skillopt.optimizer.scheduler import EditBudgetScheduler, ScheduleType


def test_skill_document_hash_and_slow_update():
    skill = SkillDocument(content="Hello world")
    assert skill.token_estimate == 2
    h1 = skill.hash
    skill.commit("Hello world v2")
    assert skill.hash != h1
    skill.set_slow_update("Keep it concise")
    assert "<!-- slow-update -->" in skill.content
    assert "Keep it concise" in skill.get_slow_update_content()


def test_edit_engine_add_and_replace():
    skill = SkillDocument(content="Rule one.\nRule two.")
    engine = EditEngine()
    edits = [
        Edit(action=EditAction.ADD, content="Rule three.", priority=1.0),
    ]
    result = engine.apply_edits(skill, edits, budget=2)
    assert "Rule three." in result.new_content
    assert len(result.applied) == 1


def test_edit_budget_scheduler_cosine():
    sched = EditBudgetScheduler(initial=4, minimum=2, schedule=ScheduleType.COSINE, total_steps=10)
    assert sched.get_budget(0) == 4
    assert sched.get_budget(10) == 2
    assert sched.get_budget(5) >= 2


def test_validation_gate_logic():
    from skillopt.gate.validation import ValidationGate, GateDecision

    decision = GateDecision(accepted=True, score=0.8, previous_score=0.5, reason="ok")
    assert abs(decision.improvement - 0.3) < 1e-9


def test_optimizer_state_accept_and_reject():
    skill = SkillDocument(content="initial")
    state = OptimizerState(current_skill=skill.snapshot())

    better = SkillDocument(content="improved")
    assert state.accept_candidate(better, 0.8) is True
    assert state.best_selection_score == 0.8

    worse = SkillDocument(content="worse")
    state.record_rejection([], 0.8, 0.3)
    assert len(state.rejected_buffer) == 1


def test_direct_chat_harness_with_mock():
    client = MockLLMClient()
    harness = DirectChatHarness(client)
    skill = SkillDocument(content="Answer questions accurately.")
    task = Task(id="t1", input="What is the capital of France?", expected="Paris")
    traj = harness.run(task, skill)
    assert traj.task_id == "t1"
    assert isinstance(traj.score, float)


def test_normalize_scoring():
    assert _normalize("Paris!") == _normalize("paris")


def test_parse_edits_from_json():
    text = '{"edits": [{"action": "add", "content": "Be precise", "priority": 0.9}]}'
    edits = parse_edits_from_response(text)
    assert len(edits) == 1
    assert edits[0].action == EditAction.ADD
