"""Configuration loading and dataset utilities."""

from __future__ import annotations

import random
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from skillopt.core.skill import SkillDocument
from skillopt.core.trajectory import Task
from skillopt.optimizer.loop import OptimizationConfig
from skillopt.optimizer.scheduler import ScheduleType


class DatasetConfig(BaseModel):
    train_ratio: float = 0.4
    selection_ratio: float = 0.1
    test_ratio: float = 0.5
    seed: int = 42
    format: str = "yaml"  # yaml | spreadsheetbench
    data_root: str | None = None
    manifest: str | None = None
    limit: int | None = None


class CLIHarnessConfig(BaseModel):
    """Optional CLI settings for codex / claude_code harnesses."""

    command: list[str] = Field(default_factory=list)
    timeout: int = 300
    require_cli: bool = False
    prompt_file: str = "prompt.txt"
    permission_mode: str = "bypassPermissions"
    extra_args: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    target: str = "mock"
    optimizer: str = "mock"
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-20250514"
    azure_deployment: str = ""
    azure_endpoint: str = ""
    kimi_model: str = "kimi-k2.6"
    kimi_base_url: str = "https://api.kimi.com/coding/v1"
    kimi_disable_thinking: bool = True
    kimi_coding_agent: bool = True


class HarnessConfig(BaseModel):
    type: str = "direct_chat"  # direct_chat | codex | claude_code
    workspace_root: str | None = None
    cli: CLIHarnessConfig = Field(default_factory=CLIHarnessConfig)


class BaselinesConfig(BaseModel):
    """External baseline methods (paper Appendix C style)."""

    methods: list[str] = Field(default_factory=list)
    textgrad_steps: int = 3
    gepa_generations: int = 2
    gepa_population: int = 4
    evoskill_generations: int = 2


class SkillOptConfig(BaseModel):
    name: str = "skillopt-run"
    skill_path: str = "skill.md"
    dataset_path: str = "tasks.json"
    output_dir: str = "artifacts"
    harness: str = "direct_chat"
    workspace_root: str | None = None
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    harness_config: HarnessConfig = Field(default_factory=HarnessConfig)
    baselines: BaselinesConfig = Field(default_factory=BaselinesConfig)


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ (without overwriting)."""
    import os

    if path is not None:
        candidates = [Path(path)]
    else:
        here = Path(__file__).resolve()
        project_root = here.parents[2]  # src/skillopt/config.py -> repo root
        candidates = [
            Path.cwd() / ".env",
            project_root / ".env",
        ]

    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {path}")

    extends = raw.pop("extends", None)
    if not extends:
        return raw

    if isinstance(extends, str):
        extends = [extends]

    merged: dict = {}
    for ext in extends:
        ext_path = (path.parent / ext).resolve()
        if not ext_path.is_file():
            raise FileNotFoundError(f"Config extends missing file: {ext_path}")
        merged = _deep_merge(merged, _load_raw_config(ext_path))
    return _deep_merge(merged, raw)


def config_base_dir(config_path: str | Path) -> Path:
    """Resolve preset root for relative paths in skill/dataset/output fields."""
    path = Path(config_path).resolve()
    if path.parent.name == "profiles":
        return path.parent.parent
    return path.parent


def load_config(path: str | Path) -> SkillOptConfig:
    load_dotenv()
    raw = _load_raw_config(Path(path).resolve())
    return SkillOptConfig.model_validate(raw)


def load_tasks(path: str | Path, dataset: DatasetConfig | None = None) -> list[Task]:
    if dataset and dataset.format == "spreadsheetbench":
        from skillopt.benchmarks.spreadsheetbench import load_spreadsheetbench

        if not dataset.data_root:
            raise ValueError("dataset.data_root required for format=spreadsheetbench")
        return load_spreadsheetbench(
            dataset.data_root,
            manifest=dataset.manifest,
            limit=dataset.limit,
        )

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return [Task(**item) for item in raw["tasks"]]


def load_tasks_for_config(config: SkillOptConfig, base_dir: Path | None = None) -> list[Task]:
    if config.dataset.format == "spreadsheetbench":
        ds = config.dataset.model_copy()
        if ds.data_root:
            root = Path(ds.data_root)
            if not root.is_absolute():
                root = (base_dir or Path.cwd()) / root
            ds.data_root = str(root.resolve())
        return load_tasks("", dataset=ds)
    root = base_dir or Path.cwd()
    return load_tasks(root / config.dataset_path)


def split_tasks(
    tasks: list[Task],
    train_ratio: float = 0.4,
    selection_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[Task], list[Task], list[Task]]:
    rng = random.Random(seed)
    shuffled = list(tasks)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = max(1, int(n * train_ratio))
    sel_end = train_end + max(1, int(n * selection_ratio))

    return (
        shuffled[:train_end],
        shuffled[train_end:sel_end],
        shuffled[sel_end:] or shuffled[-1:],
    )


def create_client(provider: str, config: ModelConfig, role: str = "target") -> "LLMClient":
    from skillopt.llm.client import (
        AnthropicLLMClient,
        AzureOpenAILLMClient,
        KimiLLMClient,
        LLMClient,
        MockLLMClient,
        OpenAILLMClient,
    )

    if provider == "openai":
        return OpenAILLMClient(model=config.openai_model)
    if provider == "kimi":
        return KimiLLMClient(
            model=config.kimi_model,
            base_url=config.kimi_base_url or None,
            disable_thinking=config.kimi_disable_thinking,
            coding_agent=config.kimi_coding_agent,
        )
    if provider == "anthropic":
        return AnthropicLLMClient(model=config.anthropic_model)
    if provider == "azure":
        return AzureOpenAILLMClient(
            deployment=config.azure_deployment,
            endpoint=config.azure_endpoint or None,
        )
    if provider == "mock":
        return MockLLMClient()
    raise ValueError(f"Unknown provider: {provider}")


def create_clients(config: SkillOptConfig):
    target = create_client(config.models.target, config.models, "target")
    optimizer = create_client(config.models.optimizer, config.models, "optimizer")
    return target, optimizer
