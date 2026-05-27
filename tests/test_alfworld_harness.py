"""Tests for ALFWorld text-world harness."""

import pytest

from skillopt.config import load_tasks
from skillopt.core.skill import SkillDocument
from skillopt.harness.alfworld import ALFWorldHarness, apply_action, goal_satisfied, WorldState
from skillopt.llm.client import MockLLMClient


@pytest.fixture
def alf_tasks():
    return load_tasks("benchmarks/alfworld/tasks.yaml")


def test_apply_action_pickup():
    state = WorldState(location="kitchen", objects={"apple": "kitchen"})
    rooms = {"kitchen", "living room"}
    state, _ = apply_action(state, "pick up apple", rooms)
    assert "apple" in state.inventory


def test_goal_satisfied():
    state = WorldState(objects={"apple": "countertop"}, flags={"apple_heated": True})
    goal = {"objects": {"apple": "countertop"}, "flags": {"apple_heated": True}}
    assert goal_satisfied(state, goal)


def test_alfworld_weak_skill(alf_tasks):
    harness = ALFWorldHarness(MockLLMClient())
    weak = SkillDocument(content="Complete household tasks.")
    traj = harness.run(alf_tasks[0], weak)
    assert traj.score < 1.0


def test_alfworld_strong_skill(alf_tasks):
    harness = ALFWorldHarness(MockLLMClient())
    strong = SkillDocument(
        content=(
            "Plan step-by-step with go to, pick up, heat, put, and clean actions "
            "until the goal is satisfied."
        )
    )
    traj = harness.run(alf_tasks[0], strong)
    assert traj.score == 1.0
    assert traj.success
