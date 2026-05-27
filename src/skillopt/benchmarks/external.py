"""External baseline optimizers (TextGrad / GEPA / EvoSkill — simplified paper reproductions)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from skillopt.config import SkillOptConfig, create_clients, load_config, load_tasks_for_config, split_tasks
from skillopt.core.edit import Edit, EditAction, EditEngine
from skillopt.core.skill import SkillDocument
from skillopt.harness.factory import create_harness
from skillopt.llm.client import LLMClient, parse_edits_from_response


TEXTGRAD_SYSTEM = """You are a TextGrad-style prompt optimizer.
Given a skill document and failure feedback, propose an improved skill.
Output JSON: {"edits": [{"action": "add", "content": "...", "rationale": "...", "priority": 0.9}]}
Focus on textual gradient signals from failures — general rules only."""

GEPA_MUTATE_SYSTEM = """You mutate a skill document for evolutionary search (GEPA-style).
Output JSON with one diverse edit: {"edits": [{"action": "add|replace", "content": "...", "target": "...", "priority": 0.7}]}
Explore alternative procedural strategies."""

EVOSKILL_MUTATE_SYSTEM = """You are an EvoSkill-style harness-aware skill mutator.
Given rollout failures, propose skill mutations that help the agent on similar tasks.
Output JSON edits only. Prefer concise procedural rules."""


def run_textgrad(
    config: SkillOptConfig,
    base_dir: Path,
    train_tasks,
    test_tasks,
    initial_skill: SkillDocument,
    steps: int = 3,
) -> dict:
    target_client, optimizer_client = create_clients(config)
    harness = create_harness(config, target_client)
    engine = EditEngine()
    skill = initial_skill.snapshot()
    best_skill = skill.snapshot()
    best_score = harness.evaluate_batch(test_tasks, skill)

    for step in range(steps):
        trajs = [harness.run(t, skill) for t in train_tasks]
        failures = [t for t in trajs if not t.success]
        feedback = "\n".join(t.summary() for t in failures[:8]) or "No failures — refine for robustness."
        user = f"Current skill:\n{skill.content}\n\nFailure feedback:\n{feedback}\n\nStep {step + 1}/{steps}"
        response = optimizer_client.complete(TEXTGRAD_SYSTEM, user)
        edits = parse_edits_from_response(response.content)
        if not edits:
            continue
        result = engine.apply_edits(skill, edits, budget=1)
        if not result.applied:
            continue
        candidate = SkillDocument(content=result.new_content)
        score = harness.evaluate_batch(test_tasks, candidate)
        if score >= best_score:
            best_score = score
            best_skill = candidate.snapshot()
            skill = candidate

    out_path = base_dir / config.output_dir / "baseline_textgrad.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_skill.save(str(out_path))
    return {"score": best_score, "skill_path": str(out_path), "tokens": best_skill.token_estimate}


def run_gepa(
    config: SkillOptConfig,
    base_dir: Path,
    train_tasks,
    test_tasks,
    initial_skill: SkillDocument,
    generations: int = 2,
    population: int = 4,
) -> dict:
    target_client, optimizer_client = create_clients(config)
    harness = create_harness(config, target_client)
    engine = EditEngine()
    rng = random.Random(config.dataset.seed)

    def evaluate(skill: SkillDocument) -> float:
        return harness.evaluate_batch(test_tasks, skill)

    population_skills = [initial_skill.snapshot()]
    while len(population_skills) < population:
        population_skills.append(initial_skill.snapshot())

    best = max(population_skills, key=evaluate)
    best_score = evaluate(best)

    for _ in range(generations):
        scored: list[tuple[float, SkillDocument]] = []
        for skill in population_skills:
            scored.append((evaluate(skill), skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [s for _, s in scored[: max(1, population // 2)]]
        best = elites[0]
        best_score = scored[0][0]

        next_gen = elites[:]
        while len(next_gen) < population:
            parent = rng.choice(elites)
            user = f"Parent skill:\n{parent.content}\n\nMutate for better held-out performance."
            response = optimizer_client.complete(GEPA_MUTATE_SYSTEM, user)
            edits = parse_edits_from_response(response.content)
            if not edits:
                next_gen.append(parent.snapshot())
                continue
            result = engine.apply_edits(parent, edits, budget=1)
            child = SkillDocument(content=result.new_content or parent.content)
            next_gen.append(child)
        population_skills = next_gen

    out_path = base_dir / config.output_dir / "baseline_gepa.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.save(str(out_path))
    return {"score": best_score, "skill_path": str(out_path), "tokens": best.token_estimate}


def run_evoskill(
    config: SkillOptConfig,
    base_dir: Path,
    train_tasks,
    test_tasks,
    initial_skill: SkillDocument,
    generations: int = 2,
) -> dict:
    target_client, optimizer_client = create_clients(config)
    harness = create_harness(config, target_client)
    engine = EditEngine()
    skill = initial_skill.snapshot()
    best = skill.snapshot()
    best_score = harness.evaluate_batch(test_tasks, skill)

    for gen in range(generations):
        trajs = [harness.run(t, skill) for t in train_tasks]
        context = "\n---\n".join(t.summary() for t in trajs[:6])
        user = (
            f"Current skill:\n{skill.content}\n\n"
            f"Harness rollouts (generation {gen + 1}):\n{context}"
        )
        response = optimizer_client.complete(EVOSKILL_MUTATE_SYSTEM, user)
        edits = parse_edits_from_response(response.content)
        if not edits:
            continue
        result = engine.apply_edits(skill, edits, budget=2)
        candidate = SkillDocument(content=result.new_content)
        score = harness.evaluate_batch(test_tasks, candidate)
        if score >= best_score:
            best_score = score
            best = candidate.snapshot()
        skill = candidate

    out_path = base_dir / config.output_dir / "baseline_evoskill.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best.save(str(out_path))
    return {"score": best_score, "skill_path": str(out_path), "tokens": best.token_estimate}


def run_external_baselines(config_path: Path) -> dict:
    config = load_config(config_path)
    base_dir = config_path.parent.resolve()
    skill_path = base_dir / config.skill_path

    tasks = load_tasks_for_config(config, base_dir)
    train, _, test_tasks = split_tasks(
        tasks,
        config.dataset.train_ratio,
        config.dataset.selection_ratio,
        config.dataset.seed,
    )
    initial = SkillDocument.from_file(skill_path)
    methods = [m.lower() for m in config.baselines.methods]

    results: dict = {"benchmark": config.name, "external": {}}
    if "textgrad" in methods:
        results["external"]["textgrad"] = run_textgrad(
            config,
            base_dir,
            train,
            test_tasks,
            initial,
            steps=config.baselines.textgrad_steps,
        )
    if "gepa" in methods:
        results["external"]["gepa"] = run_gepa(
            config,
            base_dir,
            train,
            test_tasks,
            initial,
            generations=config.baselines.gepa_generations,
            population=config.baselines.gepa_population,
        )
    if "evoskill" in methods:
        results["external"]["evoskill"] = run_evoskill(
            config,
            base_dir,
            train,
            test_tasks,
            initial,
            generations=config.baselines.evoskill_generations,
        )

    out = base_dir / config.output_dir / "external_baselines.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
