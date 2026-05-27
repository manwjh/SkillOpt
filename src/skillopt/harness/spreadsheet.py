"""SpreadsheetBench-style harness with openpyxl runtime verification."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from skillopt.core.skill import SkillDocument
from skillopt.core.trajectory import Task, Trajectory
from skillopt.harness.base import HarnessAdapter
from skillopt.harness.spreadsheet_runtime import (
    apply_writes,
    create_workbook,
    parse_writes_json,
    verify_against_answer_workbook,
    verify_cells,
    workbook_preview,
)
from skillopt.benchmarks.spreadsheetbench import materialize_workbook
from skillopt.llm.client import LLMClient

SPREADSHEET_AGENT_SYSTEM = """You are a spreadsheet automation agent.
Given a skill, task, and workbook preview, output ONLY valid JSON:
{
  "writes": [
    {"cell": "B5", "value": 45}
  ]
}

Rules:
- Write evaluated STATIC values (numbers/strings), not formulas
- Inspect the workbook structure before deciding cells
- Fill all cells required by the task
- Do not include markdown or explanation outside JSON"""


CODEX_TRACE_SUMMARY = "codex_trace_summary.txt"
WORKBOOK_NAME = "task.xlsx"


class SpreadsheetHarness(HarnessAdapter):
    """LLM proposes cell writes; harness applies via openpyxl and verifies."""

    harness_name = "spreadsheet"

    def __init__(self, target_client: LLMClient, workspace_root: str | None = None) -> None:
        self.target_client = target_client
        self.workspace_root = Path(workspace_root) if workspace_root else None

    def run(self, task: Task, skill: SkillDocument) -> Trajectory:
        workspace = self._prepare_workspace(task, skill)
        trace_steps: list[str] = []
        tool_calls: list[dict] = []

        try:
            wb_path = workspace / WORKBOOK_NAME
            meta = task.metadata
            if meta.get("workbook_template"):
                materialize_workbook(task, wb_path)
            elif meta.get("sheet_data"):
                create_workbook(wb_path, meta["sheet_data"], meta.get("sheet_name", "Sheet1"))
            elif not wb_path.exists():
                raise FileNotFoundError(
                    f"Task {task.id} needs sheet_data or workbook_template metadata"
                )

            preview = workbook_preview(wb_path)
            trace_steps.append(f"loaded {WORKBOOK_NAME}")
            trace_steps.append(f"preview:\n{preview[:400]}")

            user = (
                f"Task ID: {task.id}\n\n"
                f"## Skill\n{skill.content}\n\n"
                f"## Task\n{task.input}\n\n"
                f"## Workbook Preview\n{preview}\n\n"
                "Respond with JSON writes only."
            )
            response = self.target_client.complete(SPREADSHEET_AGENT_SYSTEM, user)
            trace_steps.append(f"llm_response={response.content[:300]}")

            try:
                writes = parse_writes_json(response.content)
            except (json.JSONDecodeError, ValueError) as e:
                summary = self._write_trace(workspace, task, trace_steps, 0.0, False, str(e))
                return Trajectory(
                    task_id=task.id,
                    skill_hash=skill.hash,
                    final_answer=response.content[:500],
                    score=0.0,
                    success=False,
                    error=str(e),
                    raw_trace=summary,
                    tool_calls=tool_calls,
                )

            write_log = apply_writes(wb_path, writes)
            trace_steps.extend(write_log)
            for w in writes:
                tool_calls.append({"type": "cell_write", **w})

            expected = meta.get("expected_cells", {})
            answer_wb = meta.get("answer_workbook")
            answer_cells = meta.get("answer_position") or list(expected.keys())
            if answer_wb and answer_cells:
                score, details = verify_against_answer_workbook(
                    wb_path, Path(answer_wb), answer_cells
                )
            elif expected:
                score, details = verify_cells(wb_path, expected)
            else:
                score, details = 0.0, {"error": "no verification metadata"}
            trace_steps.append(f"verification={json.dumps(details)}")
            success = score >= 1.0

            summary = self._write_trace(workspace, task, trace_steps, score, success)
            return Trajectory(
                task_id=task.id,
                skill_hash=skill.hash,
                messages=[
                    {"role": "system", "content": SPREADSHEET_AGENT_SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": response.content},
                ],
                tool_calls=tool_calls,
                final_answer=json.dumps({"writes": writes, "score": score}),
                score=score,
                success=success,
                raw_trace=summary,
            )
        except Exception as e:
            summary = self._write_trace(workspace, task, trace_steps, 0.0, False, str(e))
            return Trajectory(
                task_id=task.id,
                skill_hash=skill.hash,
                score=0.0,
                success=False,
                error=str(e),
                raw_trace=summary,
            )
        finally:
            if self.workspace_root is None and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

    def evaluate_batch(self, tasks: list[Task], skill: SkillDocument) -> float:
        if not tasks:
            return 0.0
        trajs = self.run_batch(tasks, skill)
        return sum(t.score for t in trajs) / len(trajs)

    def _prepare_workspace(self, task: Task, skill: SkillDocument) -> Path:
        if self.workspace_root:
            workspace = self.workspace_root / task.id
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            workspace = Path(tempfile.mkdtemp(prefix=f"skillopt-ss-{task.id}-"))

        (workspace / "SKILL.md").write_text(skill.content, encoding="utf-8")
        (workspace / "task.md").write_text(task.input, encoding="utf-8")
        return workspace

    @staticmethod
    def _write_trace(
        workspace: Path,
        task: Task,
        steps: list[str],
        score: float,
        success: bool,
        error: str | None = None,
    ) -> str:
        lines = [
            f"task_id: {task.id}",
            f"harness: spreadsheet",
            f"score: {score:.3f}",
            f"success: {success}",
        ]
        if error:
            lines.append(f"error: {error}")
        lines.append("steps:")
        lines.extend(f"- {s}" for s in steps)
        summary = "\n".join(lines)
        (workspace / CODEX_TRACE_SUMMARY).write_text(summary, encoding="utf-8")
        (workspace / "execution_trace.json").write_text(
            json.dumps({"steps": steps, "score": score, "success": success}, indent=2),
            encoding="utf-8",
        )
        return summary
