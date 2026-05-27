"""Load official SpreadsheetBench JSON/JSONL into SkillOpt Task objects."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterator

from skillopt.core.trajectory import Task


def parse_answer_positions(raw: str) -> list[str]:
    """Parse answer_position field (e.g. 'B5', 'B5,C6', \"'Sheet1'!B5\")."""
    if not raw or not str(raw).strip():
        return []
    text = str(raw).strip()
    parts = re.split(r"[,;\s]+", text)
    cells: list[str] = []
    for part in parts:
        part = part.strip().strip("'\"")
        if "!" in part:
            part = part.split("!", 1)[1]
        part = part.replace("$", "")
        if re.fullmatch(r"[A-Za-z]+\d+", part):
            cells.append(part.upper())
    return cells


def find_test_cases(spreadsheet_dir: Path) -> list[dict[str, Path]]:
    """Return sorted test cases with input/answer xlsx paths."""
    if not spreadsheet_dir.is_dir():
        return []

    inputs = sorted(spreadsheet_dir.glob("*_input.xlsx"))
    cases: list[dict[str, Path]] = []
    for inp in inputs:
        answer = inp.name.replace("_input.xlsx", "_answer.xlsx")
        ans_path = inp.parent / answer
        if not ans_path.exists():
            continue
        prefix = inp.name.replace("_input.xlsx", "")
        cases.append({"prefix": prefix, "input": inp, "answer": ans_path})
    return cases


def load_spreadsheetbench_record(
    record: dict[str, Any],
    data_root: Path,
    *,
    test_case_index: int = 0,
) -> Task:
    """Map one SpreadsheetBench JSON record to a SkillOpt Task."""
    task_id = str(record["id"])
    instruction = str(record.get("instruction", "")).strip()
    rel_path = str(record.get("spreadsheet_path", "")).strip()
    spreadsheet_dir = (data_root / rel_path).resolve()
    cases = find_test_cases(spreadsheet_dir)
    if not cases:
        raise FileNotFoundError(f"No test cases under {spreadsheet_dir}")

    idx = min(max(0, test_case_index), len(cases) - 1)
    case = cases[idx]
    answer_cells = parse_answer_positions(str(record.get("answer_position", "")))

    return Task(
        id=f"{task_id}-tc{idx + 1}",
        input=instruction,
        metadata={
            "format": "spreadsheetbench",
            "instruction_type": record.get("instruction_type", ""),
            "source_id": task_id,
            "test_case_prefix": case["prefix"],
            "workbook_template": str(case["input"]),
            "answer_workbook": str(case["answer"]),
            "answer_position": answer_cells,
            "spreadsheet_dir": str(spreadsheet_dir),
        },
    )


def iter_spreadsheetbench_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from .json (array) or .jsonl."""
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict) and "tasks" in data:
        yield from data["tasks"]
    else:
        yield data


def load_spreadsheetbench(
    data_root: str | Path,
    manifest: str | Path | None = None,
    *,
    limit: int | None = None,
    expand_test_cases: bool = False,
) -> list[Task]:
    """Load SpreadsheetBench tasks from data_root + manifest file."""
    root = Path(data_root).resolve()
    if manifest is None:
        for candidate in (
            root / "dataset.jsonl",
            root / "dataset.json",
            root / "data" / "sample_data_200.jsonl",
            root / "sample_data_200.jsonl",
            root / "all_data_912.json",
            root.parent / "sample_data_200" / "dataset.json",
        ):
            if candidate.exists():
                manifest = candidate
                break
    if manifest is None:
        raise FileNotFoundError(
            f"No SpreadsheetBench manifest found under {root}. "
            "Download sample_data_200.tar.gz from github.com/RUCKBReasoning/SpreadsheetBench"
        )

    manifest_path = Path(manifest)
    if not manifest_path.is_absolute():
        manifest_path = (root / manifest_path).resolve()

    tasks: list[Task] = []
    for record in iter_spreadsheetbench_records(manifest_path):
        rel = str(record.get("spreadsheet_path", ""))
        sheet_dir = root / rel
        cases = find_test_cases(sheet_dir)
        if expand_test_cases and cases:
            for i in range(len(cases)):
                tasks.append(load_spreadsheetbench_record(record, root, test_case_index=i))
        else:
            tasks.append(load_spreadsheetbench_record(record, root, test_case_index=0))
        if limit and len(tasks) >= limit:
            break
    return tasks


def materialize_workbook(task: Task, dest: Path) -> Path:
    """Copy input workbook template into workspace as task.xlsx."""
    template = task.metadata.get("workbook_template")
    if not template:
        raise FileNotFoundError(f"Task {task.id} has no workbook_template")
    src = Path(template)
    if not src.exists():
        raise FileNotFoundError(f"Workbook template missing: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest


def export_tasks_yaml(tasks: list[Task], path: Path) -> None:
    """Export loaded tasks to SkillOpt YAML for offline use."""
    import yaml

    rows = []
    for t in tasks:
        rows.append(
            {
                "id": t.id,
                "input": t.input,
                "metadata": t.metadata,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"tasks": rows}, f, allow_unicode=True, sort_keys=False)
