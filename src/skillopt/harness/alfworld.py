"""ALFWorld-style embodied harness (text-world simulator for SkillOpt)."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from skillopt.core.skill import SkillDocument
from skillopt.core.trajectory import Task, Trajectory
from skillopt.harness.base import HarnessAdapter
from skillopt.llm.client import LLMClient

ALFWORLD_SYSTEM = """You are an embodied household agent in a text world.
Given the scene and goal, output ONLY valid JSON:
{"actions": ["go to kitchen", "pick up apple", "heat apple with microwave", "put apple on countertop"]}

Rules:
- Use short imperative actions from the allowed verb set
- Complete the goal in as few steps as possible
- Do not include markdown outside JSON"""


@dataclass
class WorldState:
    location: str = "living room"
    inventory: list[str] = field(default_factory=list)
    objects: dict[str, str] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "inventory": list(self.inventory),
            "objects": dict(self.objects),
            "flags": dict(self.flags),
        }


def apply_action(state: WorldState, action: str, rooms: set[str]) -> tuple[WorldState, str]:
    """Apply one text action; return new state and log line."""
    s = deepcopy(state)
    a = action.lower().strip()
    log = f"action={action!r}"

    if m := re.search(r"go to (.+)", a):
        dest = m.group(1).strip()
        if dest in rooms or dest.replace("_", " ") in rooms:
            s.location = dest.replace("_", " ")
        return s, log + f" -> location={s.location}"

    if m := re.search(r"pick up (.+)", a):
        obj = m.group(1).strip()
        loc = s.objects.get(obj, "")
        if loc == s.location or loc == "here":
            s.inventory.append(obj)
            s.objects[obj] = "inventory"
        return s, log + f" -> inventory={s.inventory}"

    if m := re.search(r"heat (.+) with (.+)", a):
        obj, device = m.group(1).strip(), m.group(2).strip()
        if obj in s.inventory and device in s.objects:
            s.flags[f"{obj}_heated"] = True
        return s, log + f" -> flags={s.flags}"

    if m := re.search(r"put (.+) on (.+)", a):
        obj, surface = m.group(1).strip(), m.group(2).strip()
        if obj in s.inventory:
            s.inventory.remove(obj)
            s.objects[obj] = surface
        return s, log + f" -> objects={s.objects}"

    if m := re.search(r"open (.+)", a):
        device = m.group(1).strip()
        s.flags[f"{device}_open"] = True
        return s, log + f" -> flags={s.flags}"

    if m := re.search(r"clean (.+)", a):
        obj = m.group(1).strip()
        if obj in s.inventory or s.objects.get(obj) == s.location:
            s.flags[f"{obj}_clean"] = True
        return s, log + f" -> flags={s.flags}"

    return s, log + " (no effect)"


def goal_satisfied(state: WorldState, goal: dict[str, Any]) -> bool:
    for obj, req in goal.get("objects", {}).items():
        if state.objects.get(obj) != req:
            return False
    for flag, val in goal.get("flags", {}).items():
        if state.flags.get(flag) != val:
            return False
    for item in goal.get("inventory", []):
        if item not in state.inventory:
            return False
    if "location" in goal and state.location != goal["location"]:
        return False
    return True


def parse_actions_json(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    if isinstance(data, dict) and "actions" in data:
        return [str(a) for a in data["actions"]]
    if isinstance(data, list):
        return [str(a) for a in data]
    raise ValueError("Expected JSON with 'actions' array")


class ALFWorldHarness(HarnessAdapter):
    """Text-world ALFWorld-style tasks with deterministic goal checking."""

    harness_name = "alfworld"

    def __init__(self, target_client: LLMClient) -> None:
        self.target_client = target_client

    def run(self, task: Task, skill: SkillDocument) -> Trajectory:
        meta = task.metadata
        rooms = set(meta.get("rooms", ["kitchen", "living room", "bedroom"]))
        initial = meta.get("initial_state", {})
        state = WorldState(
            location=initial.get("location", "living room"),
            inventory=list(initial.get("inventory", [])),
            objects=dict(initial.get("objects", {})),
            flags=dict(initial.get("flags", {})),
        )
        goal = meta.get("goal", {})
        trace_steps = [f"initial={json.dumps(state.snapshot())}"]

        scene = (
            f"Task ID: {task.id}\n\n"
            f"## Skill\n{skill.content}\n\n"
            f"## Scene\nLocation: {state.location}\n"
            f"Inventory: {state.inventory}\n"
            f"Objects: {json.dumps(state.objects)}\n\n"
            f"## Goal\n{task.input}\n\n"
            "Respond with JSON actions only."
        )

        try:
            response = self.target_client.complete(ALFWORLD_SYSTEM, scene)
            trace_steps.append(f"llm={response.content[:300]}")
            actions = parse_actions_json(response.content)
            for action in actions[: meta.get("max_steps", 12)]:
                state, step_log = apply_action(state, action, rooms)
                trace_steps.append(step_log)

            success = goal_satisfied(state, goal)
            score = 1.0 if success else 0.0
            summary = "\n".join(
                [
                    f"task_id: {task.id}",
                    "harness: alfworld",
                    f"score: {score:.3f}",
                    f"success: {success}",
                    "steps:",
                    *[f"- {s}" for s in trace_steps],
                    f"final_state={json.dumps(state.snapshot())}",
                ]
            )
            return Trajectory(
                task_id=task.id,
                skill_hash=skill.hash,
                messages=[
                    {"role": "system", "content": ALFWORLD_SYSTEM},
                    {"role": "user", "content": scene},
                    {"role": "assistant", "content": response.content},
                ],
                tool_calls=[{"type": "action", "action": a} for a in actions],
                final_answer=json.dumps({"actions": actions, "final": state.snapshot()}),
                score=score,
                success=success,
                raw_trace=summary,
            )
        except Exception as e:
            return Trajectory(
                task_id=task.id,
                skill_hash=skill.hash,
                score=0.0,
                success=False,
                error=str(e),
                raw_trace=f"alfworld error: {e}",
            )

    def evaluate_batch(self, tasks: list[Task], skill: SkillDocument) -> float:
        if not tasks:
            return 0.0
        trajs = self.run_batch(tasks, skill)
        return sum(t.score for t in trajs) / len(trajs)
