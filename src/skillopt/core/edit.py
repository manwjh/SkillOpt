from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from skillopt.core.skill import SkillDocument


class EditAction(str, Enum):
    ADD = "add"
    INSERT_AFTER = "insert_after"
    DELETE = "delete"
    REPLACE = "replace"


@dataclass
class Edit:
    """A structured skill edit proposal."""

    action: EditAction
    content: str
    target: str = ""
    rationale: str = ""
    priority: float = 0.0
    source: str = ""

    def __str__(self) -> str:
        return f"[{self.action.value}] {self.rationale}: {self.content[:80]}"


@dataclass
class EditResult:
    applied: list[Edit] = field(default_factory=list)
    skipped: list[Edit] = field(default_factory=list)
    new_content: str = ""


class EditEngine:
    """Apply bounded add/delete/replace edits to a skill document."""

    def apply_edits(
        self,
        skill: SkillDocument,
        edits: list[Edit],
        budget: int,
    ) -> EditResult:
        ranked = sorted(edits, key=lambda e: e.priority, reverse=True)
        selected = ranked[:budget]

        content = skill.get_editable_content()
        applied: list[Edit] = []
        skipped: list[Edit] = []

        for edit in selected:
            try:
                content = self._apply_one(content, edit)
                applied.append(edit)
            except ValueError:
                skipped.append(edit)

        slow = skill.get_slow_update_content()
        final = content
        if slow:
            from skillopt.core.skill import SLOW_UPDATE_END, SLOW_UPDATE_START

            final = (
                f"{content}\n\n{SLOW_UPDATE_START}\n{slow}\n{SLOW_UPDATE_END}"
            )

        return EditResult(applied=applied, skipped=skipped, new_content=final)

    def _apply_one(self, content: str, edit: Edit) -> str:
        if edit.action in (EditAction.ADD, EditAction.INSERT_AFTER):
            if edit.action == EditAction.INSERT_AFTER:
                if not edit.target or edit.target not in content:
                    raise ValueError(f"insert_after anchor not found: {edit.target}")
                return content.replace(edit.target, f"{edit.target}\n{edit.content}")
            if edit.target:
                return content.replace(edit.target, f"{edit.target}\n{edit.content}")
            return content.rstrip() + f"\n\n{edit.content}"

        if edit.action == EditAction.DELETE:
            if edit.target and edit.target in content:
                return content.replace(edit.target, "")
            if edit.content and edit.content in content:
                return content.replace(edit.content, "")
            raise ValueError(f"Delete target not found: {edit.target or edit.content}")

        if edit.action == EditAction.REPLACE:
            if edit.target and edit.target in content:
                return content.replace(edit.target, edit.content)
            raise ValueError(f"Replace target not found: {edit.target}")

        raise ValueError(f"Unknown action: {edit.action}")

    @staticmethod
    def parse_rule_lines(text: str) -> list[Edit]:
        """Parse simple rule lines from optimizer output (fallback parser)."""
        edits: list[Edit] = []
        for i, line in enumerate(text.strip().split("\n")):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = re.sub(r"^[-*\d.]+\s*", "", line)
            edits.append(
                Edit(
                    action=EditAction.ADD,
                    content=line,
                    rationale="parsed rule",
                    priority=1.0 - i * 0.01,
                )
            )
        return edits
