"""Run baseline comparisons (no-skill / initial / optimized / external methods)."""

from __future__ import annotations

from pathlib import Path

from skillopt.config import SkillOptConfig, create_clients, load_config, load_tasks_for_config, split_tasks
from skillopt.core.skill import SkillDocument
from skillopt.harness.factory import create_harness


def run_baselines(config_path: Path, *, include_external: bool = False) -> dict:
    config = load_config(config_path)
    base_dir = config_path.parent.resolve()

    skill_path = base_dir / config.skill_path
    best_path = base_dir / config.output_dir / "best_skill.md"

    tasks = load_tasks_for_config(config, base_dir)
    _, _, test_tasks = split_tasks(
        tasks,
        config.dataset.train_ratio,
        config.dataset.selection_ratio,
        config.dataset.seed,
    )

    target_client, _ = create_clients(config)
    harness = create_harness(config, target_client)

    empty = SkillDocument(content="Answer the question.")
    initial = SkillDocument.from_file(skill_path)

    results = {
        "harness": config.harness,
        "target": config.models.target,
        "benchmark": config.name,
        "test_tasks": len(test_tasks),
        "baselines": {},
    }

    results["baselines"]["no_skill"] = _eval(harness, test_tasks, empty)
    results["baselines"]["initial_skill"] = _eval(harness, test_tasks, initial)

    if best_path.exists():
        optimized = SkillDocument.from_file(best_path)
        results["baselines"]["skillopt"] = _eval(harness, test_tasks, optimized)
        results["baselines"]["skillopt_tokens"] = optimized.token_estimate

    for name, path in (
        ("textgrad", base_dir / config.output_dir / "baseline_textgrad.md"),
        ("gepa", base_dir / config.output_dir / "baseline_gepa.md"),
        ("evoskill", base_dir / config.output_dir / "baseline_evoskill.md"),
    ):
        if path.exists():
            skill = SkillDocument.from_file(path)
            results["baselines"][name] = _eval(harness, test_tasks, skill)

    base = results["baselines"]["no_skill"]["score"]
    for name in list(results["baselines"].keys()):
        if name.endswith("_tokens"):
            continue
        if name in results["baselines"] and isinstance(results["baselines"][name], dict):
            results["baselines"][name]["lift"] = results["baselines"][name]["score"] - base

    if include_external or config.baselines.methods:
        from skillopt.benchmarks.external import run_external_baselines

        ext = run_external_baselines(config_path)
        results["external"] = ext.get("external", {})
        for name, data in results["external"].items():
            skill = SkillDocument.from_file(data["skill_path"])
            results["baselines"][name] = _eval(harness, test_tasks, skill)
            results["baselines"][name]["lift"] = results["baselines"][name]["score"] - base

    return results


def _eval(harness, tasks, skill: SkillDocument) -> dict:
    score = harness.evaluate_batch(tasks, skill)
    return {"score": score, "tokens": skill.token_estimate}
