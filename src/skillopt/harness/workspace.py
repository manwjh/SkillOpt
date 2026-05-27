"""Workspace-based harness for Codex / Claude Code style execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from skillopt.config import CLIHarnessConfig
from skillopt.core.skill import SkillDocument
from skillopt.core.trajectory import Task, Trajectory
from skillopt.harness.base import HarnessAdapter
from skillopt.harness.direct_chat import DirectChatHarness
from skillopt.llm.client import LLMClient

SKILL_FILENAME = "SKILL.md"
TASK_FILENAME = "task.md"
WORKBOOK_NAME = "task.xlsx"
TRACE_FILENAME = "execution_trace.json"
CODEX_TRACE_SUMMARY = "codex_trace_summary.txt"


class WorkspaceHarness(HarnessAdapter):
    """Run tasks in an isolated workspace with SKILL.md injected."""

    harness_name: str = "workspace"
    default_cli: list[str] = []

    def __init__(
        self,
        target_client: LLMClient,
        workspace_root: str | None = None,
        cli_command: list[str] | None = None,
        cli_config: CLIHarnessConfig | None = None,
    ) -> None:
        self.target_client = target_client
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.cli_config = cli_config or CLIHarnessConfig()
        self.cli_command = cli_command or self.cli_config.command or self.default_cli
        self.cli_timeout = self.cli_config.timeout
        self.require_cli = self.cli_config.require_cli
        self._fallback = DirectChatHarness(target_client)

    def run(self, task: Task, skill: SkillDocument) -> Trajectory:
        workspace = self._prepare_workspace(task, skill)
        try:
            cli_bin = self._cli_binary()
            if cli_bin:
                if self._cli_available(cli_bin):
                    return self._run_via_cli(task, skill, workspace)
                if self.require_cli:
                    summary = self._write_trace(
                        workspace,
                        task,
                        steps=[f"CLI not found: {cli_bin}"],
                        score=0.0,
                        success=False,
                        error=f"CLI required but not on PATH: {cli_bin}",
                    )
                    return Trajectory(
                        task_id=task.id,
                        skill_hash=skill.hash,
                        score=0.0,
                        success=False,
                        error=f"CLI not found: {cli_bin}",
                        raw_trace=summary,
                    )
            return self._run_via_fallback(task, skill, workspace)
        finally:
            if self.workspace_root is None and workspace.exists():
                shutil.rmtree(workspace, ignore_errors=True)

    def evaluate_batch(self, tasks: list[Task], skill: SkillDocument) -> float:
        if not tasks:
            return 0.0
        trajectories = self.run_batch(tasks, skill)
        return sum(t.score for t in trajectories) / len(trajectories)

    def _prepare_workspace(self, task: Task, skill: SkillDocument) -> Path:
        if self.workspace_root:
            workspace = self.workspace_root / task.id
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            workspace = Path(tempfile.mkdtemp(prefix=f"skillopt-{task.id}-"))

        (workspace / SKILL_FILENAME).write_text(skill.content, encoding="utf-8")
        (workspace / TASK_FILENAME).write_text(task.input, encoding="utf-8")

        meta = task.metadata
        if meta.get("workbook_template"):
            from skillopt.benchmarks.spreadsheetbench import materialize_workbook

            materialize_workbook(task, workspace / WORKBOOK_NAME)
        elif meta.get("sheet_data"):
            from skillopt.harness.spreadsheet_runtime import create_workbook

            create_workbook(
                workspace / WORKBOOK_NAME,
                meta["sheet_data"],
                meta.get("sheet_name", "Sheet1"),
            )

        if meta.get("files"):
            for name, content in meta["files"].items():
                (workspace / name).write_text(str(content), encoding="utf-8")

        prompt = self._build_prompt(skill, task, workspace)
        (workspace / self.cli_config.prompt_file).write_text(prompt, encoding="utf-8")
        if self.harness_name == "kimi_code":
            (workspace / "AGENTS.md").write_text(skill.content, encoding="utf-8")
        return workspace

    def _build_prompt(self, skill: SkillDocument, task: Task, workspace: Path) -> str:
        lines = [
            "You are an autonomous coding agent working in a local workspace.",
            f"Workspace: {workspace}",
            "",
            f"Read {SKILL_FILENAME} for skill instructions.",
            f"Read {TASK_FILENAME} for the task specification.",
        ]
        if (workspace / WORKBOOK_NAME).exists():
            lines.append(
                f"Modify {WORKBOOK_NAME} as required. Write static evaluated values, not formulas."
            )
        if self.harness_name == "kimi_code":
            lines.append("Read AGENTS.md for agent instructions (same as SKILL.md).")
        lines.extend(["", "## Task", task.input, "", "Complete the task now."])
        return "\n".join(lines)

    def _run_via_cli(
        self, task: Task, skill: SkillDocument, workspace: Path
    ) -> Trajectory:
        prompt_path = workspace / self.cli_config.prompt_file
        prompt = prompt_path.read_text(encoding="utf-8")
        cmd = self._build_cli_command(workspace, prompt, skill)
        trace_steps = [f"command={' '.join(cmd[:6])}..."]
        stdin_prompt = prompt if getattr(self, "prompt_via_stdin", False) else None
        if stdin_prompt is None and cmd and cmd[-1] != prompt:
            cmd = [*cmd, prompt]

        try:
            proc = subprocess.run(
                cmd,
                input=stdin_prompt,
                capture_output=True,
                text=True,
                timeout=self.cli_timeout,
                cwd=str(workspace),
                env=self._subprocess_env(),
            )
            answer = proc.stdout.strip() or proc.stderr.strip()
            trace = {
                "command": cmd,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
            trace_steps.append(f"returncode={proc.returncode}")
            trace_steps.append(f"stdout={proc.stdout[:400]}")
        except subprocess.TimeoutExpired:
            answer = ""
            trace = {"error": "timeout", "command": cmd}
            trace_steps.append("error=timeout")
            proc = None

        score, verify_details = self._score_task(task, workspace, answer)
        trace["verification"] = verify_details
        trace_steps.append(f"verification={json.dumps(verify_details)[:500]}")

        (workspace / TRACE_FILENAME).write_text(json.dumps(trace, indent=2), encoding="utf-8")
        summary = self._write_trace(
            workspace,
            task,
            trace_steps,
            score,
            score >= 1.0,
            harness=self.harness_name,
        )
        return Trajectory(
            task_id=task.id,
            skill_hash=skill.hash,
            final_answer=answer,
            score=score,
            success=score >= 1.0,
            raw_trace=summary,
        )

    def _build_cli_command(
        self, workspace: Path, prompt: str, skill: SkillDocument
    ) -> list[str]:
        if self.cli_config.command:
            return [*self.cli_config.command, *self.cli_config.extra_args, prompt]
        return [*self.default_cli, *self.cli_config.extra_args, prompt]

    def _run_via_fallback(
        self, task: Task, skill: SkillDocument, workspace: Path
    ) -> Trajectory:
        enriched_skill = SkillDocument(
            content=(
                f"{skill.content}\n\n"
                f"## Workspace Context\n"
                f"Working directory: {workspace}\n"
                f"Task file: {TASK_FILENAME}\n"
                f"Read {TASK_FILENAME} and any attached files before answering."
            )
        )
        traj = self._fallback.run(task, enriched_skill)
        trace = {
            "mode": "fallback",
            "harness": self.harness_name,
            "workspace": str(workspace),
            "messages": traj.messages,
        }
        summary = self._write_trace(
            workspace,
            task,
            [
                "mode=fallback",
                f"answer={traj.final_answer!r}",
                f"trace={json.dumps(trace)[:800]}",
            ],
            traj.score,
            traj.success,
            traj.error,
            harness=self.harness_name,
        )
        (workspace / TRACE_FILENAME).write_text(json.dumps(trace, indent=2), encoding="utf-8")
        traj.raw_trace = summary
        traj.skill_hash = skill.hash
        return traj

    def _score_task(
        self, task: Task, workspace: Path, stdout_answer: str
    ) -> tuple[float, dict]:
        meta = task.metadata
        wb_path = workspace / WORKBOOK_NAME

        if wb_path.exists():
            from skillopt.harness.spreadsheet_runtime import (
                verify_against_answer_workbook,
                verify_cells,
            )

            answer_wb = meta.get("answer_workbook")
            answer_cells = meta.get("answer_position") or list(meta.get("expected_cells", {}).keys())
            if answer_wb and answer_cells:
                score, details = verify_against_answer_workbook(
                    wb_path, Path(answer_wb), answer_cells
                )
                return score, {"mode": "answer_workbook", "details": details}
            expected = meta.get("expected_cells", {})
            if expected:
                score, details = verify_cells(wb_path, expected)
                return score, {"mode": "expected_cells", "details": details}

        from skillopt.harness.direct_chat import DirectChatHarness

        score = DirectChatHarness._score(stdout_answer, task.expected)
        return score, {"mode": "stdout_match", "answer": stdout_answer}

    @staticmethod
    def _write_trace(
        workspace: Path,
        task: Task,
        steps: list[str],
        score: float,
        success: bool,
        error: str | None = None,
        harness: str = "workspace",
    ) -> str:
        lines = [
            f"task_id: {task.id}",
            f"harness: {harness}",
            f"score: {score:.3f}",
            f"success: {success}",
        ]
        if error:
            lines.append(f"error: {error}")
        lines.append("steps:")
        lines.extend(f"- {s}" for s in steps)
        summary = "\n".join(lines)
        (workspace / CODEX_TRACE_SUMMARY).write_text(summary, encoding="utf-8")
        return summary

    @staticmethod
    def _cli_available(command: str) -> bool:
        return shutil.which(command) is not None

    def _cli_binary(self) -> str | None:
        if self.cli_config.command:
            return self.cli_config.command[0]
        if self.default_cli:
            return self.default_cli[0]
        return None

    def _subprocess_env(self) -> dict[str, str]:
        """Pass Kimi Code CLI credentials from .env (see Kimi env var docs)."""
        env = os.environ.copy()
        if self.harness_name != "kimi_code":
            return env
        from skillopt.config import load_dotenv

        load_dotenv()
        for src, dst in (
            ("KIMI_API_KEY", "KIMI_API_KEY"),
            ("KIMI_BASE_URL", "KIMI_BASE_URL"),
            ("KIMI_MODEL_NAME", "KIMI_MODEL_NAME"),
        ):
            val = os.environ.get(src)
            if val:
                env[dst] = val
        env.setdefault("KIMI_BASE_URL", "https://api.kimi.com/coding/v1")
        env.setdefault("KIMI_MODEL_NAME", "kimi-k2.6")
        # Ensure kimi binary on PATH (~/.local/bin after install.sh)
        local_bin = str(Path.home() / ".local" / "bin")
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"
        return env


class CodexHarness(WorkspaceHarness):
    harness_name = "codex"
    default_cli = ["codex", "exec", "--full-auto"]

    def _build_cli_command(
        self, workspace: Path, prompt: str, skill: SkillDocument
    ) -> list[str]:
        base = self.cli_config.command or ["codex", "exec", "--full-auto"]
        return [
            *base,
            "-C",
            str(workspace.resolve()),
            *self.cli_config.extra_args,
            prompt,
        ]


class ClaudeCodeHarness(WorkspaceHarness):
    harness_name = "claude_code"
    default_cli = ["claude", "--print"]
    prompt_via_stdin = True

    def _build_cli_command(
        self, workspace: Path, prompt: str, skill: SkillDocument
    ) -> list[str]:
        base = self.cli_config.command or ["claude", "--print"]
        cmd = list(base)
        if "--print" not in cmd and "-p" not in cmd:
            cmd.append("--print")
        cmd.extend(
            [
                "--permission-mode",
                self.cli_config.permission_mode,
                "--add-dir",
                str(workspace.resolve()),
            ]
        )
        cmd.extend(self.cli_config.extra_args)
        return cmd


class KimiCodeHarness(WorkspaceHarness):
    """Kimi Code CLI agent — https://www.kimi.com/code/docs/kimi-code-cli/getting-started.html"""

    harness_name = "kimi_code"
    default_cli = ["kimi", "--print", "--yolo", "--quiet"]

    def _build_cli_command(
        self, workspace: Path, prompt: str, skill: SkillDocument
    ) -> list[str]:
        base = self.cli_config.command or ["kimi", "--print", "--yolo", "--quiet"]
        cmd = list(base)
        if "--print" not in cmd:
            cmd.append("--print")
        if not any(x in cmd for x in ("--yolo", "-y", "--yes", "--auto-approve")):
            cmd.append("--yolo")
        if "-w" not in cmd and "--work-dir" not in cmd:
            cmd.extend(["-w", str(workspace.resolve())])
        cmd.extend(self.cli_config.extra_args)
        if "-p" not in cmd and "--prompt" not in cmd:
            cmd.extend(["-p", prompt])
        return cmd
