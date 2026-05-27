"""Skill library — catalog, store, and share optimized skills."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from skillopt.core.skill import SkillDocument


@dataclass
class SkillEntry:
    id: str
    name: str
    domain: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    skill_path: str = ""
    benchmark: str = ""
    harness: str = "direct_chat"
    model: str = "mock"
    score: float = 0.0
    tokens: int = 0
    created_at: str = ""
    version: int = 1
    status: str = "draft"  # draft | reviewed | published
    reviewer: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class SkillLibrary:
    """File-based skill library with catalog index."""

    def __init__(self, root: str | Path = "skill_library") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog_path = self.root / "catalog.json"
        self.skills_dir = self.root / "skills"
        self.skills_dir.mkdir(exist_ok=True)
        if not self.catalog_path.exists():
            self._save_catalog([])

    def list(self, domain: str | None = None, status: str | None = None) -> list[SkillEntry]:
        entries = [SkillEntry(**e) for e in self._load_catalog()]
        if domain:
            entries = [e for e in entries if e.domain == domain]
        if status:
            entries = [e for e in entries if e.status == status]
        return entries

    def get(self, skill_id: str) -> SkillEntry | None:
        for entry in self.list():
            if entry.id == skill_id:
                return entry
        return None

    def add(
        self,
        skill_path: Path,
        name: str,
        domain: str,
        description: str = "",
        tags: list[str] | None = None,
        benchmark: str = "",
        harness: str = "direct_chat",
        model: str = "mock",
        score: float = 0.0,
        status: str = "draft",
    ) -> SkillEntry:
        skill = SkillDocument.from_file(skill_path)
        skill_id = f"{domain}-{name}".lower().replace(" ", "-")
        dest = self.skills_dir / f"{skill_id}.md"
        shutil.copy2(skill_path, dest)

        entry = SkillEntry(
            id=skill_id,
            name=name,
            domain=domain,
            description=description,
            tags=tags or [],
            skill_path=str(dest),
            benchmark=benchmark,
            harness=harness,
            model=model,
            score=score,
            tokens=skill.token_estimate,
            created_at=datetime.now(timezone.utc).isoformat(),
            status=status,
        )

        catalog = self._load_catalog()
        catalog = [e for e in catalog if e["id"] != skill_id]
        catalog.append(entry.to_dict())
        self._save_catalog(catalog)
        return entry

    def export(self, skill_id: str, dest: Path) -> Path:
        entry = self.get(skill_id)
        if not entry:
            raise KeyError(f"Skill not found: {skill_id}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.skill_path, dest)
        return dest

    def review(self, skill_id: str, status: str = "reviewed", reviewer: str = "") -> SkillEntry:
        catalog = self._load_catalog()
        for e in catalog:
            if e["id"] == skill_id:
                e["status"] = status
                if reviewer:
                    e["reviewer"] = reviewer
                e["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                self._save_catalog(catalog)
                return SkillEntry(**e)
        raise KeyError(f"Skill not found: {skill_id}")

    def domains(self) -> list[str]:
        return sorted({e.domain for e in self.list()})

    def _load_catalog(self) -> list[dict]:
        with open(self.catalog_path, encoding="utf-8") as f:
            return json.load(f)

    def _save_catalog(self, catalog: list[dict]) -> None:
        with open(self.catalog_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
