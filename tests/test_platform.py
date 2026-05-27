"""Extended tests for Phase 2/3 features."""

from pathlib import Path

import pytest

from skillopt.config import SkillOptConfig, create_client, load_config
from skillopt.core.skill import SkillDocument
from skillopt.cost.tracker import CostTracker
from skillopt.harness.direct_chat import DirectChatHarness
from skillopt.harness.factory import create_harness
from skillopt.harness.workspace import CodexHarness, WorkspaceHarness
from skillopt.library.catalog import SkillLibrary
from skillopt.llm.client import MockLLMClient
from skillopt.optimizer.reflection import ReflectionEngine
from skillopt.optimizer.scheduler import EditBudgetScheduler
from skillopt.runner import run_ab_compare, run_evaluation, run_transfer


@pytest.fixture
def demo_skill(tmp_path):
    skill = SkillDocument(content="Answer accurately.")
    path = tmp_path / "skill.md"
    skill.save(str(path))
    return path


@pytest.fixture
def demo_dataset(tmp_path):
    content = """
tasks:
  - id: t1
    input: "What is the capital of France?"
    expected: "Paris"
  - id: t2
    input: "What is 2 + 2?"
    expected: "4"
"""
    path = tmp_path / "tasks.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_create_harness_direct_chat():
    cfg = SkillOptConfig(harness="direct_chat")
    harness = create_harness(cfg, MockLLMClient())
    assert harness.__class__.__name__ == "DirectChatHarness"


def test_create_harness_codex():
    cfg = SkillOptConfig(harness="codex")
    harness = create_harness(cfg, MockLLMClient())
    assert isinstance(harness, CodexHarness)


def test_workspace_harness_creates_skill_md(tmp_path):
    from skillopt.core.trajectory import Task

    harness = WorkspaceHarness(MockLLMClient(), workspace_root=str(tmp_path / "ws"))
    skill = SkillDocument(content="Test skill rule.")
    task = Task(id="t1", input="What is the capital of France?", expected="Paris")
    traj = harness.run(task, skill)
    assert (tmp_path / "ws" / "t1" / "SKILL.md").exists()
    assert traj.task_id == "t1"


def test_cost_tracker():
    tracker = CostTracker()
    tracker.add_target(100)
    tracker.add_optimizer(200)
    assert tracker.total_tokens == 300
    assert tracker.cost_per_point(0.5) == 600


def test_reflection_parallel():
    from skillopt.core.trajectory import Trajectory

    engine = ReflectionEngine(MockLLMClient(), minibatch_size=1, workers=4)
    trajs = [
        Trajectory(task_id="f1", skill_hash="abc", success=False, score=0.0),
        Trajectory(task_id="f2", skill_hash="abc", success=False, score=0.0),
    ]
    edits = engine.reflect(trajs, SkillDocument(content="initial"))
    assert len(edits) >= 1


def test_skill_library(tmp_path, demo_skill):
    lib = SkillLibrary(tmp_path / "lib")
    entry = lib.add(demo_skill, name="Test", domain="qa", score=0.9)
    assert lib.get(entry.id) is not None
    assert len(lib.list(domain="qa")) == 1

    reviewed = lib.review(entry.id, "published", "tester")
    assert reviewed.status == "published"


def test_run_evaluation(demo_skill, demo_dataset):
    result = run_evaluation(demo_skill, demo_dataset)
    assert "score" in result
    assert result["tasks"] == 2


def test_run_transfer(demo_skill, demo_dataset):
    result = run_transfer(demo_skill, demo_dataset)
    assert "lift" in result


def test_run_ab_compare(demo_skill, demo_dataset, tmp_path):
    skill_b = tmp_path / "skill_b.md"
    SkillDocument(content="For geography questions, always state the official capital city name only.\nFor arithmetic, compute step-by-step and state only the numeric result.").save(str(skill_b))
    result = run_ab_compare(demo_skill, skill_b, demo_dataset)
    assert result["winner"] in ("A", "B", "tie")


def test_load_benchmark_config():
    config_path = Path("benchmarks/spreadsheet/profiles/mock-spreadsheet.yaml")
    if config_path.exists():
        cfg = load_config(config_path)
        assert cfg.harness == "spreadsheet"
        assert cfg.optimization.epochs == 3
        assert cfg.models.target == "mock"


def test_config_extends_merge():
    root = Path("benchmarks/spreadsheet/profiles")
    mock = load_config(root / "mock-spreadsheet.yaml")
    official_kimi_code = load_config(root / "official-kimi-code-smoke.yaml")
    full = load_config(root / "official-kimi-code-full.yaml")

    assert mock.harness == "spreadsheet"
    assert official_kimi_code.harness == "kimi_code"
    assert official_kimi_code.dataset.limit == 4
    assert full.dataset.limit is None
    assert full.output_dir == "artifacts_spreadsheetbench_full"
    assert full.harness_config.cli.extra_args == ["--no-thinking", "--max-steps-per-turn", "40"]


def test_create_client_mock():
    from skillopt.config import ModelConfig

    client = create_client("mock", ModelConfig())
    assert isinstance(client, MockLLMClient)
