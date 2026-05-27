"""Harness factory — create adapter from config."""

from __future__ import annotations

from skillopt.config import SkillOptConfig
from skillopt.harness.base import HarnessAdapter
from skillopt.harness.alfworld import ALFWorldHarness
from skillopt.harness.direct_chat import DirectChatHarness
from skillopt.harness.office_qa import OfficeQAHarness
from skillopt.harness.spreadsheet import SpreadsheetHarness
from skillopt.harness.workspace import ClaudeCodeHarness, CodexHarness, KimiCodeHarness
from skillopt.llm.client import LLMClient


def create_harness(config: SkillOptConfig, target_client: LLMClient) -> HarnessAdapter:
    harness_type = getattr(config, "harness", "direct_chat")
    workspace_root = getattr(config, "workspace_root", None) or config.harness_config.workspace_root
    cli_config = config.harness_config.cli

    if harness_type == "spreadsheet":
        return SpreadsheetHarness(target_client, workspace_root=workspace_root)
    if harness_type == "office_qa":
        return OfficeQAHarness(target_client)
    if harness_type == "alfworld":
        return ALFWorldHarness(target_client)
    if harness_type == "codex":
        return CodexHarness(
            target_client,
            workspace_root=workspace_root,
            cli_config=cli_config,
        )
    if harness_type == "claude_code":
        return ClaudeCodeHarness(
            target_client,
            workspace_root=workspace_root,
            cli_config=cli_config,
        )
    if harness_type == "kimi_code":
        return KimiCodeHarness(
            target_client,
            workspace_root=workspace_root,
            cli_config=cli_config,
        )
    return DirectChatHarness(target_client)
