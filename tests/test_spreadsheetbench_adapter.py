"""Tests for SpreadsheetBench dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillopt.benchmarks.spreadsheetbench import (
    export_tasks_yaml,
    find_test_cases,
    load_spreadsheetbench,
    load_spreadsheetbench_record,
    parse_answer_positions,
)
from skillopt.harness.spreadsheet_runtime import (
    create_workbook,
    require_openpyxl,
    verify_against_answer_workbook,
)


@pytest.fixture
def sb_root(tmp_path: Path) -> Path:
    require_openpyxl()
    record = {
        "id": "demo-1",
        "instruction": "Write the sum of B2:B3 into B4 as a static value.",
        "spreadsheet_path": "spreadsheet/demo-1",
        "instruction_type": "Cell-Level Manipulation",
        "answer_position": "B4",
    }
    sheet_dir = tmp_path / "spreadsheet" / "demo-1"
    sheet_dir.mkdir(parents=True)
    inp = sheet_dir / "1_demo-1_input.xlsx"
    ans = sheet_dir / "1_demo-1_answer.xlsx"
    create_workbook(inp, [["A", "B"], ["x", 10], ["y", 20], ["sum", ""]])
    create_workbook(ans, [["A", "B"], ["x", 10], ["y", 20], ["sum", 30]])

    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return tmp_path


def test_parse_answer_positions():
    assert parse_answer_positions("B4") == ["B4"]
    assert parse_answer_positions("B4, C5") == ["B4", "C5"]
    assert parse_answer_positions("'Sheet1'!B4") == ["B4"]


def test_load_spreadsheetbench_record(sb_root: Path):
    record = json.loads((sb_root / "dataset.jsonl").read_text().strip())
    task = load_spreadsheetbench_record(record, sb_root)
    assert task.id == "demo-1-tc1"
    assert "sum" in task.input.lower() or "B2" in task.input
    assert task.metadata["answer_position"] == ["B4"]


def test_verify_against_answer_workbook(sb_root: Path):
    record = json.loads((sb_root / "dataset.jsonl").read_text().strip())
    task = load_spreadsheetbench_record(record, sb_root)
    out = sb_root / "out.xlsx"
    create_workbook(out, [["A", "B"], ["x", 10], ["y", 20], ["sum", 30]])
    score, _ = verify_against_answer_workbook(
        out, Path(task.metadata["answer_workbook"]), task.metadata["answer_position"]
    )
    assert score == 1.0


def test_load_and_export(sb_root: Path, tmp_path: Path):
    tasks = load_spreadsheetbench(sb_root, manifest="dataset.jsonl")
    assert len(tasks) == 1
    out = tmp_path / "exported.yaml"
    export_tasks_yaml(tasks, out)
    assert out.exists()
    assert "demo-1-tc1" in out.read_text()


def test_find_test_cases(sb_root: Path):
    cases = find_test_cases(sb_root / "spreadsheet" / "demo-1")
    assert len(cases) == 1
    assert cases[0]["input"].name.endswith("_input.xlsx")
