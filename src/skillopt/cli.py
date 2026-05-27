"""SkillOpt CLI entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from skillopt.config import load_dotenv
from skillopt.library.catalog import SkillLibrary
from skillopt.runner import (
    run_ab_compare,
    run_evaluation,
    run_optimization,
    run_transfer,
)

load_dotenv()

app = typer.Typer(name="skillopt", help="Text-space optimizer for agent skills")
library_app = typer.Typer(help="Skill library management")
app.add_typer(library_app, name="library")

console = Console()


@app.command()
def optimize(
    config_path: Path = typer.Argument(..., help="Path to YAML config file"),
) -> None:
    """Run skill optimization from a config file."""
    console.print(f"[bold blue]SkillOpt[/] — optimizing from {config_path}")
    result = run_optimization(config_path)

    table = Table(title="Optimization Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Initial selection score", f"{result.initial_score:.2%}")
    table.add_row("Best selection score", f"{result.best_score:.2%}")
    table.add_row("Accepted edits", str(result.accepted_edits))
    table.add_row("Total steps", str(result.total_steps))
    table.add_row("Total tokens", str(result.cost.get("total_tokens", 0)))
    if result.cost.get("total_tokens"):
        gain = result.best_score - result.initial_score
        if gain > 0:
            cpp = result.cost["total_tokens"] / gain
            table.add_row("Cost / point", f"{cpp:,.0f} tokens")
    table.add_row("Best skill", result.best_skill_path)
    console.print(table)
    console.print(f"\n[green]Done![/] Export: {result.best_skill_path}")


@app.command()
def baselines(
    config_path: Path = typer.Argument(..., help="Benchmark config YAML"),
    external: bool = typer.Option(False, "--external", help="Run TextGrad/GEPA/EvoSkill baselines"),
) -> None:
    """Compare no-skill / initial-skill / SkillOpt (+ optional external baselines)."""
    from skillopt.benchmarks.baselines import run_baselines

    results = run_baselines(config_path, include_external=external)
    table = Table(title=f"Baselines — {results['benchmark']}")
    table.add_column("Method", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Tokens", style="yellow")
    table.add_column("Lift", style="magenta")

    for name, data in results["baselines"].items():
        if name.endswith("_tokens"):
            continue
        lift = data.get("lift")
        table.add_row(
            name,
            f"{data['score']:.2%}",
            str(data.get("tokens", "—")),
            f"{lift:+.2%}" if lift is not None else "—",
        )
    console.print(table)
    console.print(f"Harness: {results['harness']} | Target: {results['target']} | Test: {results['test_tasks']} tasks")


@app.command()
def evaluate(
    skill_path: Path = typer.Argument(..., help="Path to skill markdown file"),
    dataset_path: Path = typer.Argument(..., help="Path to tasks YAML"),
    model: str = typer.Option("mock", help="Target model: mock, openai, anthropic, azure, kimi"),
    harness: str = typer.Option(
        "direct_chat",
        help="Harness: direct_chat, spreadsheet, office_qa, alfworld, codex, claude_code, kimi_code",
    )
) -> None:
    """Evaluate a skill on a task dataset without optimization."""
    data = run_evaluation(skill_path, dataset_path, harness, model)
    console.print(f"Average score: [green]{data['score']:.2%}[/] on {data['tasks']} tasks")


@app.command()
def transfer(
    skill_path: Path = typer.Argument(..., help="Skill to transfer"),
    dataset_path: Path = typer.Argument(..., help="Target dataset"),
    model: str = typer.Option("mock", help="Target model"),
    harness: str = typer.Option("direct_chat", help="Target harness"),
) -> None:
    """Test cross-model/harness/benchmark skill transfer."""
    data = run_transfer(skill_path, dataset_path, harness, model)
    table = Table(title="Transfer Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Baseline (no skill)", f"{data['baseline_score']:.2%}")
    table.add_row("Transferred skill", f"{data['transferred_score']:.2%}")
    table.add_row("Lift", f"{data['lift']:+.2%}")
    table.add_row("Harness", data["harness"])
    console.print(table)


@app.command()
def compare(
    skill_a: Path = typer.Argument(..., help="Skill A path"),
    skill_b: Path = typer.Argument(..., help="Skill B path"),
    dataset_path: Path = typer.Argument(..., help="Dataset path"),
    model: str = typer.Option("mock"),
    harness: str = typer.Option("direct_chat"),
) -> None:
    """A/B compare two skills on the same dataset."""
    data = run_ab_compare(skill_a, skill_b, dataset_path, harness, model)
    table = Table(title="A/B Comparison")
    table.add_column("Skill", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Tokens", style="yellow")
    table.add_row("A", f"{data['skill_a']['score']:.2%}", str(data["skill_a"]["tokens"]))
    table.add_row("B", f"{data['skill_b']['score']:.2%}", str(data["skill_b"]["tokens"]))
    table.add_row("Winner", data["winner"], f"Δ={data['delta']:.2%}")
    console.print(table)


@app.command()
def regression(
    skill_path: Path = typer.Argument(..., help="Skill to regression-test"),
    dataset_path: Path = typer.Argument(..., help="Dataset path"),
    min_score: float = typer.Option(0.8, help="Minimum passing score (0-1)"),
    model: str = typer.Option("mock"),
    harness: str = typer.Option("direct_chat"),
) -> None:
    """CI regression test — exit 1 if score below threshold."""
    data = run_evaluation(skill_path, dataset_path, harness, model)
    score = data["score"]
    console.print(f"Regression score: {score:.2%} (threshold: {min_score:.2%})")
    if score < min_score:
        console.print("[red]REGRESSION FAILED[/]")
        raise typer.Exit(1)
    console.print("[green]REGRESSION PASSED[/]")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8080, help="Bind port"),
) -> None:
    """Start Web Console and REST API server."""
    from skillopt.api.server import serve as _serve

    console.print(f"[bold blue]SkillOpt Console[/] → http://{host}:{port}")
    _serve(host=host, port=port)


@app.command("import-spreadsheetbench")
def import_spreadsheetbench(
    data_root: Path = typer.Argument(..., help="SpreadsheetBench data root (with spreadsheet/ + jsonl)"),
    output: Path = typer.Option(Path("tasks.yaml"), help="Output YAML path"),
    manifest: str | None = typer.Option(None, help="Manifest json/jsonl relative to data_root"),
    limit: int | None = typer.Option(None, help="Max tasks to export"),
) -> None:
    """Convert official SpreadsheetBench data to SkillOpt tasks YAML."""
    from skillopt.benchmarks.spreadsheetbench import export_tasks_yaml, load_spreadsheetbench

    tasks = load_spreadsheetbench(data_root, manifest=manifest, limit=limit)
    export_tasks_yaml(tasks, output)
    console.print(f"[green]Exported {len(tasks)} tasks → {output}[/]")


@app.command("spreadsheetbench")
def spreadsheetbench_cmd(
    config_path: Path = typer.Argument(..., help="SpreadsheetBench config YAML"),
    split: str = typer.Option("test", help="Task split: test | train | selection | all"),
    skill_path: Path | None = typer.Option(None, help="Override skill file for evaluation"),
    no_resume: bool = typer.Option(False, help="Ignore partial report and re-run all tasks"),
    report_suffix: str = typer.Option("", help="Report filename suffix, e.g. _best"),
) -> None:
    """Run full SpreadsheetBench evaluation with incremental JSON report."""
    from skillopt.benchmarks.spreadsheetbench_run import run_spreadsheetbench_eval

    summary = run_spreadsheetbench_eval(
        config_path,
        split=split,
        skill_path=skill_path,
        resume=not no_resume,
        report_suffix=report_suffix,
    )
    table = Table(title=f"SpreadsheetBench ({summary['split']})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Harness", summary["harness"])
    table.add_row("Tasks", f"{summary['tasks_evaluated']}/{summary['tasks_total']}")
    table.add_row("Average score", f"{summary['average_score']:.2%}")
    table.add_row("Success rate", f"{summary['success_rate']:.2%}")
    console.print(table)
    console.print(f"[dim]Report: {summary.get('report_path', '')}[/]")


@app.command("run-external-baselines")
def run_external_baselines_cmd(
    config_path: Path = typer.Argument(..., help="Benchmark config with baselines.methods"),
) -> None:
    """Run TextGrad / GEPA / EvoSkill baseline optimizers."""
    from skillopt.benchmarks.external import run_external_baselines

    results = run_external_baselines(config_path)
    table = Table(title="External Baselines")
    table.add_column("Method", style="cyan")
    table.add_column("Score", style="green")
    table.add_column("Skill", style="yellow")
    for name, data in results.get("external", {}).items():
        table.add_row(name, f"{data['score']:.2%}", data["skill_path"])
    console.print(table)


@app.command()
def benchmarks() -> None:
    """List available benchmark presets."""
    benchmarks_dir = Path("benchmarks")
    if not benchmarks_dir.exists():
        console.print("[yellow]No benchmarks directory found[/]")
        raise typer.Exit(0)

    table = Table(title="Benchmark Presets")
    table.add_column("ID", style="cyan")
    table.add_column("Config", style="green")
    for preset in sorted(benchmarks_dir.iterdir()):
        if not preset.is_dir():
            continue
        profiles_dir = preset / "profiles"
        if profiles_dir.is_dir():
            for profile in sorted(profiles_dir.glob("*.yaml")):
                table.add_row(f"{preset.name}/{profile.stem}", str(profile))
            continue
        config = preset / "config.yaml"
        if config.is_file():
            table.add_row(preset.name, str(config))
    console.print(table)


@library_app.command("list")
def library_list(
    domain: str | None = typer.Option(None),
    library_root: Path = typer.Option(Path("skill_library"), help="Library root"),
) -> None:
    """List skills in the library."""
    lib = SkillLibrary(library_root)
    entries = lib.list(domain=domain)
    if not entries:
        console.print("[yellow]Library is empty[/]")
        return
    table = Table(title="Skill Library")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Domain")
    table.add_column("Score")
    table.add_column("Status")
    for e in entries:
        table.add_row(e.id, e.name, e.domain, f"{e.score:.0%}", e.status)
    console.print(table)


@library_app.command("add")
def library_add(
    skill_path: Path = typer.Argument(...),
    name: str = typer.Argument(...),
    domain: str = typer.Argument(...),
    description: str = typer.Option(""),
    benchmark: str = typer.Option(""),
    score: float = typer.Option(0.0),
    library_root: Path = typer.Option(Path("skill_library")),
) -> None:
    """Add a skill to the library."""
    lib = SkillLibrary(library_root)
    entry = lib.add(skill_path, name=name, domain=domain, description=description, benchmark=benchmark, score=score)
    console.print(f"[green]Added[/] {entry.id} → {entry.skill_path}")


@library_app.command("review")
def library_review(
    skill_id: str = typer.Argument(...),
    status: str = typer.Option("reviewed"),
    reviewer: str = typer.Option(""),
    library_root: Path = typer.Option(Path("skill_library")),
) -> None:
    """Review/approve a skill in the library."""
    lib = SkillLibrary(library_root)
    entry = lib.review(skill_id, status, reviewer)
    console.print(f"[green]Reviewed[/] {entry.id} → {entry.status}")


@library_app.command("export")
def library_export(
    skill_id: str = typer.Argument(...),
    dest: Path = typer.Argument(...),
    library_root: Path = typer.Option(Path("skill_library")),
) -> None:
    """Export a skill from the library."""
    lib = SkillLibrary(library_root)
    path = lib.export(skill_id, dest)
    console.print(f"[green]Exported[/] → {path}")


if __name__ == "__main__":
    app()
