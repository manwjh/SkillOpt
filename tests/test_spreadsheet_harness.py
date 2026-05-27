"""Tests for SpreadsheetHarness with openpyxl runtime."""

import pytest

from skillopt.config import load_tasks
from skillopt.core.skill import SkillDocument
from skillopt.harness.spreadsheet import SpreadsheetHarness
from skillopt.harness.spreadsheet_runtime import require_openpyxl, verify_cells, create_workbook
from skillopt.llm.client import MockLLMClient


@pytest.fixture
def ss_tasks():
    path = "benchmarks/spreadsheet/tasks.yaml"
    return load_tasks(path)


def test_openpyxl_workbook_verify(tmp_path):
    require_openpyxl()
    wb = tmp_path / "t.xlsx"
    create_workbook(wb, [["A", "B"], [1, 10], [2, 20]])
    from skillopt.harness.spreadsheet_runtime import apply_writes

    apply_writes(wb, [{"cell": "B3", "value": 30}])
    score, details = verify_cells(wb, {"B3": 30})
    assert score == 1.0


def test_spreadsheet_harness_weak_skill(ss_tasks):
    require_openpyxl()
    harness = SpreadsheetHarness(MockLLMClient())
    weak = SkillDocument(content="Use spreadsheet tools.")
    traj = harness.run(ss_tasks[0], weak)
    assert traj.score < 1.0
    assert "harness: spreadsheet" in traj.raw_trace


def test_spreadsheet_harness_strong_skill(ss_tasks):
    require_openpyxl()
    harness = SpreadsheetHarness(MockLLMClient())
    strong = SkillDocument(
        content=(
            "Inspect workbook structure from the preview. "
            "Write static numeric evaluated values to target cells only."
        )
    )
    traj = harness.run(ss_tasks[0], strong)
    assert traj.score == 1.0
    assert traj.success
    assert len(traj.tool_calls) >= 1
