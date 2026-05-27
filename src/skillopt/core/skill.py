from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

SLOW_UPDATE_START = "<!-- slow-update -->"
SLOW_UPDATE_END = "<!-- /slow-update -->"


@dataclass
class SkillDocument:
    """Natural-language skill policy injected into agent context."""

    content: str
    version: int = 0
    history: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> SkillDocument:
        with open(path, encoding="utf-8") as f:
            return cls(content=f.read())

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.content)

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:12]

    @property
    def token_estimate(self) -> int:
        return len(self.content.split())

    def snapshot(self) -> SkillDocument:
        return SkillDocument(
            content=self.content,
            version=self.version,
            history=list(self.history),
        )

    def commit(self, new_content: str) -> None:
        self.history.append(self.content)
        self.content = new_content
        self.version += 1

    def get_editable_content(self) -> str:
        """Content outside the protected slow-update region."""
        pattern = re.compile(
            re.escape(SLOW_UPDATE_START) + r".*?" + re.escape(SLOW_UPDATE_END),
            re.DOTALL,
        )
        return pattern.sub("", self.content).strip()

    def get_slow_update_content(self) -> str:
        pattern = re.compile(
            re.escape(SLOW_UPDATE_START) + r"(.*?)" + re.escape(SLOW_UPDATE_END),
            re.DOTALL,
        )
        match = pattern.search(self.content)
        return match.group(1).strip() if match else ""

    def set_slow_update(self, guidance: str) -> None:
        block = f"{SLOW_UPDATE_START}\n{guidance}\n{SLOW_UPDATE_END}"
        if SLOW_UPDATE_START in self.content:
            pattern = re.compile(
                re.escape(SLOW_UPDATE_START) + r".*?" + re.escape(SLOW_UPDATE_END),
                re.DOTALL,
            )
            self.content = pattern.sub(block, self.content)
        else:
            self.content = self.content.rstrip() + f"\n\n{block}\n"

    def to_system_prompt(self) -> str:
        return (
            "You are an AI agent. Follow the skill instructions below precisely.\n\n"
            f"## Skill Instructions\n\n{self.content}"
        )
