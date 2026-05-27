"""Shared runner for CLI, API, and CI — single entry for optimize/evaluate/transfer."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from skillopt.config import SkillOptConfig, config_base_dir, create_clients, load_config, load_tasks, load_tasks_for_config, split_tasks
from skillopt.core.skill import SkillDocument
from skillopt.harness.factory import create_harness
from skillopt.optimizer.loop import OptimizationResult, SkillOptLoop


def run_optimization(config_path: Path) -> OptimizationResult:
    config = load_config(config_path)
    base_dir = config_base_dir(config_path)
    config.optimization.output_dir = str(base_dir / config.output_dir)

    skill = SkillDocument.from_file(base_dir / config.skill_path)
    tasks = load_tasks_for_config(config, base_dir)
    train, selection, test = split_tasks(
        tasks,
        config.dataset.train_ratio,
        config.dataset.selection_ratio,
        config.dataset.seed,
    )

    target_client, optimizer_client = create_clients(config)
    harness = create_harness(config, target_client)
    loop = SkillOptLoop(
        harness,
        optimizer_client,
        config.optimization,
        workspace_root=config.workspace_root or config.harness_config.workspace_root,
    )

    return loop.run(skill, train, selection, test)


def run_evaluation(
    skill_path: Path,
    dataset_path: Path,
    harness_type: str = "direct_chat",
    model: str = "mock",
) -> dict:
    from skillopt.config import ModelConfig, SkillOptConfig

    skill = SkillDocument.from_file(skill_path)
    tasks = load_tasks(dataset_path)
    cfg = SkillOptConfig(
        models=ModelConfig(target=model),
        harness=harness_type,  # type: ignore[arg-type]
    )
    target_client, _ = create_clients(cfg)
    harness = create_harness(cfg, target_client)
    score = harness.evaluate_batch(tasks, skill)
    return {"score": score, "tasks": len(tasks), "skill_tokens": skill.token_estimate}


def run_transfer(
    skill_path: Path,
    dataset_path: Path,
    harness_type: str = "direct_chat",
    model: str = "mock",
    baseline_skill_path: Path | None = None,
) -> dict:
    skill = SkillDocument.from_file(skill_path)
    tasks = load_tasks(dataset_path)

    from skillopt.config import ModelConfig, SkillOptConfig

    cfg = SkillOptConfig(
        models=ModelConfig(target=model),
        harness=harness_type,  # type: ignore[arg-type]
    )
    target_client, _ = create_clients(cfg)
    harness = create_harness(cfg, target_client)

    transferred = harness.evaluate_batch(tasks, skill)

    baseline = 0.0
    if baseline_skill_path:
        baseline_skill = SkillDocument.from_file(baseline_skill_path)
        baseline = harness.evaluate_batch(tasks, baseline_skill)
    else:
        empty = SkillDocument(content="Answer accurately.")
        baseline = harness.evaluate_batch(tasks, empty)

    return {
        "baseline_score": baseline,
        "transferred_score": transferred,
        "lift": transferred - baseline,
        "harness": harness_type,
        "model": model,
    }


def run_ab_compare(
    skill_a_path: Path,
    skill_b_path: Path,
    dataset_path: Path,
    harness_type: str = "direct_chat",
    model: str = "mock",
) -> dict:
    skill_a = SkillDocument.from_file(skill_a_path)
    skill_b = SkillDocument.from_file(skill_b_path)
    tasks = load_tasks(dataset_path)

    from skillopt.config import ModelConfig, SkillOptConfig

    cfg = SkillOptConfig(
        models=ModelConfig(target=model),
        harness=harness_type,  # type: ignore[arg-type]
    )
    target_client, _ = create_clients(cfg)
    harness = create_harness(cfg, target_client)

    score_a = harness.evaluate_batch(tasks, skill_a)
    score_b = harness.evaluate_batch(tasks, skill_b)
    winner = "A" if score_a > score_b else "B" if score_b > score_a else "tie"

    return {
        "skill_a": {"path": str(skill_a_path), "score": score_a, "tokens": skill_a.token_estimate},
        "skill_b": {"path": str(skill_b_path), "score": score_b, "tokens": skill_b.token_estimate},
        "winner": winner,
        "delta": abs(score_a - score_b),
    }


def load_run_summary(artifacts_dir: Path) -> dict | None:
    summary_path = artifacts_dir / "summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def load_run_log(artifacts_dir: Path) -> list[dict]:
    log_path = artifacts_dir / "optimization_log.json"
    if not log_path.exists():
        return []
    with open(log_path, encoding="utf-8") as f:
        return json.load(f)


def result_to_dict(result: OptimizationResult) -> dict:
    return {
        "initial_score": result.initial_score,
        "final_score": result.final_score,
        "best_score": result.best_score,
        "total_steps": result.total_steps,
        "accepted_edits": result.accepted_edits,
        "best_skill_path": result.best_skill_path,
        "reports": [asdict(r) for r in result.reports],
    }
