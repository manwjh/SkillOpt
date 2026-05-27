"""Full SpreadsheetBench evaluation runner with incremental report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from skillopt.config import SkillOptConfig, config_base_dir, create_clients, load_config, load_tasks_for_config, split_tasks
from skillopt.core.skill import SkillDocument
from skillopt.harness.factory import create_harness


def run_spreadsheetbench_eval(
    config_path: Path,
    *,
    split: str = "test",
    skill_path: Path | None = None,
    resume: bool = True,
    report_suffix: str = "",
) -> dict:
    """Evaluate skill on SpreadsheetBench tasks; write incremental JSON report."""
    config = load_config(config_path)
    base_dir = config_base_dir(config_path)
    config.workspace_root = str(base_dir / config.workspace_root) if config.workspace_root else None

    tasks = load_tasks_for_config(config, base_dir)
    train, selection, test = split_tasks(
        tasks,
        config.dataset.train_ratio,
        config.dataset.selection_ratio,
        config.dataset.seed,
    )

    if split == "all":
        eval_tasks = tasks
    elif split == "train":
        eval_tasks = train
    elif split == "selection":
        eval_tasks = selection
    else:
        eval_tasks = test

    skill_file = skill_path or (base_dir / config.skill_path)
    skill = SkillDocument.from_file(skill_file)

    out_dir = base_dir / config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"spreadsheetbench_{split}_report{report_suffix}.json"
    report_path = out_dir / report_name

    completed: dict[str, dict] = {}
    if resume and report_path.exists():
        prev = json.loads(report_path.read_text(encoding="utf-8"))
        for row in prev.get("results", []):
            completed[row["task_id"]] = row

    target_client, _ = create_clients(config)
    harness = create_harness(config, target_client)

    results: list[dict] = []
    for i, task in enumerate(eval_tasks):
        if task.id in completed:
            results.append(completed[task.id])
            continue
        traj = harness.run(task, skill)
        row = {
            "task_id": task.id,
            "score": traj.score,
            "success": traj.success,
            "error": traj.error,
            "final_answer": (traj.final_answer or "")[:500],
        }
        results.append(row)
        completed[task.id] = row

        partial = _build_summary(config, split, skill_file, eval_tasks, results)
        report_path.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = _build_summary(config, split, skill_file, eval_tasks, results)
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["report_path"] = str(report_path)
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _build_summary(
    config: SkillOptConfig,
    split: str,
    skill_path: Path,
    tasks: list,
    results: list[dict],
) -> dict:
    n = len(results)
    score = sum(r["score"] for r in results) / n if n else 0.0
    success = sum(1 for r in results if r.get("success"))
    return {
        "benchmark": config.name,
        "harness": config.harness,
        "target": config.models.target,
        "split": split,
        "tasks_total": len(tasks),
        "tasks_evaluated": n,
        "skill_path": str(skill_path),
        "average_score": score,
        "success_rate": success / n if n else 0.0,
        "success_count": success,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
