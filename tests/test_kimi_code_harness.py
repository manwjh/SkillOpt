"""Tests for Kimi Code CLI harness wiring."""

from skillopt.harness.workspace import KimiCodeHarness


def test_kimi_code_cli_command():
    from pathlib import Path
    from skillopt.core.skill import SkillDocument

    h = KimiCodeHarness(target_client=None)  # type: ignore[arg-type]
    ws = Path("/tmp/ws")
    cmd = h._build_cli_command(ws, "do task", SkillDocument(content="skill"))
    assert cmd[0] == "kimi"
    assert "--print" in cmd
    assert "--yolo" in cmd
    assert "-w" in cmd
    assert cmd[cmd.index("-w") + 1] == str(ws.resolve())
    assert "-p" in cmd
    assert cmd[-1] == "do task"
