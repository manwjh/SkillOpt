"""LLM client abstraction — supports OpenAI and mock for demo."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from skillopt.core.edit import Edit, EditAction


@dataclass
class LLMResponse:
    content: str
    tokens_used: int = 0


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse: ...


# Known answers for demo QA tasks
_SPREADSHEET_MOCK_WRITES: dict[str, list[dict]] = {
    "ss-sum": [{"cell": "B5", "value": 45}],
    "ss-vlookup": [{"cell": "D2", "value": 85}],
    "ss-static-col": [{"cell": "C2", "value": 10}, {"cell": "C3", "value": 20}, {"cell": "C4", "value": 30}],
    "ss-max": [{"cell": "B6", "value": 27}],
    "ss-count": [{"cell": "B7", "value": 3}],
    "ss-label": [{"cell": "C2", "value": "Report_DONE"}],
}

_OFFICE_MOCK_ANSWERS: dict[str, str] = {
    "oq-revenue": "1250000",
    "oq-headcount": "847",
    "oq-date": "2026-12-31",
    "oq-margin": "23.5",
    "oq-region": "APAC",
    "oq-sku": "1520",
}

_OFFICE_MOCK_WRONG: dict[str, str] = {
    "oq-revenue": "980000",
    "oq-headcount": "800",
    "oq-date": "2026-01-01",
    "oq-margin": "20.0",
    "oq-region": "NA",
    "oq-sku": "890",
}

_ALFWORLD_MOCK_ACTIONS: dict[str, list[str]] = {
    "alf-heat-apple": [
        "go to kitchen",
        "pick up apple",
        "heat apple with microwave",
        "put apple on countertop",
    ],
    "alf-clean-mug": [
        "go to kitchen",
        "pick up mug",
        "clean mug",
        "put mug on table",
    ],
    "alf-lamp": [
        "go to bedroom",
        "pick up lamp",
        "go to living room",
        "put lamp on table",
    ],
    "alf-book": [
        "go to bedroom",
        "pick up book",
        "go to living room",
        "put book on shelf",
    ],
    "alf-egg": [
        "go to kitchen",
        "pick up egg",
        "heat egg with microwave",
        "put egg on plate",
    ],
    "alf-remote": [
        "go to living room",
        "pick up remote",
        "put remote on table",
    ],
}

_DEMO_ANSWERS: dict[str, str] = {
    "capital of france": "Paris",
    "capital of japan": "Tokyo",
    "capital of germany": "Berlin",
    "capital of italy": "Rome",
    "capital of spain": "Madrid",
    "2 + 2": "4",
    "7 * 8": "56",
    "100 - 37": "63",
    "15 + 27": "42",
    "9 * 9": "81",
}


class MockLLMClient(LLMClient):
    """Deterministic mock — simulates weak baseline that improves with skill rules."""

    def __init__(self) -> None:
        self.call_count = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self.call_count += 1
        system_lower = system.lower()

        if any(
            k in system_lower
            for k in (
                "skill optimizer",
                "failure analyst",
                "success analyst",
                "merge skill",
                "merge failure",
                "merge_failure",
                "merge success",
                "merge_success",
                "merge final",
                "merge_final",
                "rank skill",
                "textgrad",
                "gepa",
                "evoskill",
            )
        ):
            return self._optimizer_response(system, user)

        if "rewrite a skill" in system_lower or "rewrite skill" in system_lower:
            return LLMResponse(
                content=(
                    "Answer accurately.\n\n"
                    "For geography, state official capital only.\n"
                    "For arithmetic, give numeric result only."
                ),
                tokens_used=80,
            )

        if "meta guidance" in system_lower or "cross-epoch" in user.lower():
            return LLMResponse(
                content="Maintain answer-format discipline and verify before responding.",
                tokens_used=50,
            )

        if "longitudinal guidance" in user.lower():
            return LLMResponse(
                content="- Verify answer format before responding.\n- Prefer concise procedural rules.",
                tokens_used=50,
            )

        if "spreadsheet automation agent" in system_lower:
            return self._spreadsheet_agent_response(system, user)

        if "## document" in user.lower():
            return self._office_qa_response(system, user)

        if "embodied household agent" in system_lower:
            return self._alfworld_response(system, user)

        return self._target_response(system, user)

    def _spreadsheet_agent_response(self, system: str, user: str) -> LLMResponse:
        skill_block = self._skill_block(user).lower()
        skilled = any(
            k in skill_block
            for k in ("inspect", "structure", "evaluated values", "openpyxl", "pandas", "static numeric")
        )
        for task_id, writes in _SPREADSHEET_MOCK_WRITES.items():
            if f"task id: {task_id}" in user.lower():
                if skilled:
                    return LLMResponse(
                        content=json.dumps({"writes": writes}),
                        tokens_used=40,
                    )
                wrong: list[dict] = []
                for w in writes:
                    val = w.get("value")
                    if isinstance(val, (int, float)):
                        wrong.append({**w, "value": 0})
                    elif isinstance(val, str):
                        wrong.append({**w, "value": "WRONG"})
                    else:
                        wrong.append(w)
                return LLMResponse(content=json.dumps({"writes": wrong}), tokens_used=40)
        return LLMResponse(content='{"writes": []}', tokens_used=10)

    @staticmethod
    def _skill_block(user: str) -> str:
        lower = user.lower()
        start = lower.find("## skill")
        if start < 0:
            return user
        end = len(user)
        for marker in ("## task", "## scene", "## goal", "## document", "## workbook"):
            pos = lower.find(marker, start + 1)
            if pos >= 0:
                end = min(end, pos)
        return user[start:end]

    def _office_qa_response(self, system: str, user: str) -> LLMResponse:
        skill = system.lower()
        strict = any(
            k in skill
            for k in ("number only", "no label", "oracle", "rounded", "format", "document")
        )
        for task_id, answer in _OFFICE_MOCK_ANSWERS.items():
            if f"task id: {task_id}" in user.lower():
                if strict:
                    return LLMResponse(content=answer, tokens_used=20)
                wrong = _OFFICE_MOCK_WRONG.get(task_id, "unknown")
                return LLMResponse(content=f"Based on the document: {wrong}", tokens_used=20)
        if "q3 revenue" in user.lower():
            return LLMResponse(
                content="1250000" if strict else "Based on the document: 980000",
                tokens_used=20,
            )
        if "headcount" in user.lower():
            return LLMResponse(content="847" if strict else "Based on the document: 800", tokens_used=20)
        if "contract end date" in user.lower():
            return LLMResponse(content="2026-12-31" if strict else "Based on the document: 2026-01-01", tokens_used=20)
        if "profit margin" in user.lower():
            return LLMResponse(content="23.5" if strict else "Based on the document: 20.0", tokens_used=20)
        if "highest sales" in user.lower() or "region" in user.lower():
            return LLMResponse(content="APAC" if strict else "Based on the document: NA", tokens_used=20)
        if "sku-42" in user.lower():
            return LLMResponse(content="1520" if strict else "Based on the document: 890", tokens_used=20)
        return LLMResponse(content="unknown", tokens_used=10)

    def _alfworld_response(self, system: str, user: str) -> LLMResponse:
        skill_block = self._skill_block(user).lower()
        skilled = any(
            k in skill_block
            for k in ("go to", "pick up", "heat", "put", "clean", "inspect", "step-by-step", "procedure")
        )
        for task_id, actions in _ALFWORLD_MOCK_ACTIONS.items():
            if f"task id: {task_id}" in user.lower():
                if skilled:
                    return LLMResponse(
                        content=json.dumps({"actions": actions}),
                        tokens_used=35,
                    )
                return LLMResponse(
                    content=json.dumps({"actions": actions[:1]}),
                    tokens_used=20,
                )
        return LLMResponse(content='{"actions": ["look"]}', tokens_used=10)

    def _target_response(self, system: str, user: str) -> LLMResponse:
        question = user.lower()
        skill = system.lower()

        for pattern, answer in _DEMO_ANSWERS.items():
            if pattern in question:
                is_geo = "capital" in pattern
                is_math = not is_geo

                geo_ok = "capital" in skill or "geography" in skill or "official capital" in skill
                math_ok = "arithmetic" in skill or "numeric result" in skill or "step-by-step" in skill
                format_ok = "verify" in skill or "format" in skill or "precise" in skill

                if is_geo and not geo_ok:
                    return LLMResponse(content="I think it might be Lyon.", tokens_used=20)
                if is_math and not math_ok:
                    return LLMResponse(content="approximately 5", tokens_used=20)
                if not format_ok and self.call_count < 3:
                    return LLMResponse(content=f"The answer is {answer}.", tokens_used=20)

                return LLMResponse(content=answer, tokens_used=20)

        if any(k in skill for k in ("static", "workbook", "spreadsheet", "evaluated")):
            if any(k in question for k in ("vlookup", "static", "formula", "sheet", "range", "sum")):
                return LLMResponse(
                    content="Inspect workbook structure, then write static evaluated values across the full target range.",
                    tokens_used=25,
                )

        return LLMResponse(content="I don't know.", tokens_used=10)

    def _optimizer_response(self, system: str, user: str) -> LLMResponse:
        system_lower = system.lower()
        if (
            "merge skill" in system_lower
            or "rank skill" in system_lower
            or "merge_failure" in system_lower
            or "merge_success" in system_lower
            or "merge_final" in system_lower
            or "merge failure" in system_lower
            or "merge success" in system_lower
            or "merge final" in system_lower
        ):
            return self._merge_rank_response(user)

        if "longitudinal" in user.lower() or "cross-epoch" in user.lower():
            return LLMResponse(
                content="- Verify format.\n- Use procedural rules only.",
                tokens_used=50,
            )

        edits: list[dict] = []

        if "lyon" in user.lower() or "capital" in user.lower():
            edits.append(
                {
                    "action": "add",
                    "content": "For geography questions, always state the official capital city name only.",
                    "rationale": "Agent failed to identify correct capital",
                    "priority": 0.9,
                }
            )

        if "approximately" in user.lower() or "arithmetic" in user.lower() or "2 + 2" in user:
            edits.append(
                {
                    "action": "add",
                    "content": "For arithmetic, compute step-by-step and state only the numeric result.",
                    "rationale": "Agent gave wrong arithmetic answer",
                    "priority": 0.85,
                }
            )

        if "FAILURE" in user or "the answer is" in user.lower():
            edits.append(
                {
                    "action": "add",
                    "content": "Before answering, verify your answer matches the expected concise format.",
                    "rationale": "Answer format mismatch",
                    "priority": 0.75,
                }
            )

        if (
            "harness: spreadsheet" in user.lower()
            or "cell_write" in user.lower()
            or "verification=" in user.lower()
        ):
            edits.append(
                {
                    "action": "add",
                    "content": (
                        "Inspect workbook structure from the preview. "
                        "Write static numeric evaluated values to every required target cell."
                    ),
                    "rationale": "Spreadsheet cell verification failed",
                    "priority": 0.95,
                }
            )

        if "office_qa" in user.lower() or "based on the document:" in user.lower():
            edits.append(
                {
                    "action": "add",
                    "content": (
                        "Reply with number only — no labels, currency, or commas. "
                        "For dates use YYYY-MM-DD. For text fields reply with the exact token only."
                    ),
                    "rationale": "Office document answer format mismatch",
                    "priority": 0.9,
                }
            )

        if "harness: alfworld" in user.lower() or "final_state=" in user.lower():
            edits.append(
                {
                    "action": "add",
                    "content": (
                        "Plan with go to / pick up / heat / put / clean actions. "
                        "Execute the full procedure step-by-step until the goal state is satisfied."
                    ),
                    "rationale": "ALFWorld goal not reached",
                    "priority": 0.92,
                }
            )

        if not edits:
            edits.append(
                {
                    "action": "add",
                    "content": "Be precise and concise in all answers.",
                    "rationale": "General improvement",
                    "priority": 0.5,
                }
            )

        return LLMResponse(
            content=json.dumps({"edits": edits}),
            tokens_used=100,
        )

    def _merge_rank_response(self, user: str) -> LLMResponse:
        """Preserve high-value candidate edits through merge/rank (mock teacher)."""
        lower = user.lower()
        candidates: list[dict] = []

        if "inspect workbook structure" in lower:
            candidates.append(
                {
                    "action": "add",
                    "content": (
                        "Inspect workbook structure from the preview. "
                        "Write static numeric evaluated values to every required target cell."
                    ),
                    "rationale": "Spreadsheet cell verification failed",
                    "priority": 0.95,
                }
            )
        if "plan with go to" in lower or "harness: alfworld" in lower:
            candidates.append(
                {
                    "action": "add",
                    "content": (
                        "Plan with go to / pick up / heat / put / clean actions. "
                        "Execute the full procedure step-by-step until the goal state is satisfied."
                    ),
                    "rationale": "ALFWorld goal not reached",
                    "priority": 0.92,
                }
            )
        if "number only" in lower and "no labels" in lower:
            candidates.append(
                {
                    "action": "add",
                    "content": (
                        "Reply with number only — no labels, currency, or commas. "
                        "For dates use YYYY-MM-DD. For text fields reply with the exact token only."
                    ),
                    "rationale": "Office document answer format mismatch",
                    "priority": 0.9,
                }
            )
        if "geography questions" in lower or "official capital" in lower:
            candidates.append(
                {
                    "action": "add",
                    "content": "For geography questions, always state the official capital city name only.",
                    "rationale": "Agent failed to identify correct capital",
                    "priority": 0.9,
                }
            )
        if "arithmetic" in lower or "numeric result" in lower:
            candidates.append(
                {
                    "action": "add",
                    "content": "For arithmetic, compute step-by-step and state only the numeric result.",
                    "rationale": "Agent gave wrong arithmetic answer",
                    "priority": 0.85,
                }
            )
        if "verify your answer matches" in lower:
            candidates.append(
                {
                    "action": "add",
                    "content": "Before answering, verify your answer matches the expected concise format.",
                    "rationale": "Answer format mismatch",
                    "priority": 0.75,
                }
            )

        if candidates:
            candidates.sort(key=lambda e: e["priority"], reverse=True)
            return LLMResponse(content=json.dumps({"edits": candidates[:3]}), tokens_used=80)

        return LLMResponse(
            content=json.dumps(
                {
                    "edits": [
                        {
                            "action": "add",
                            "content": "Be precise and concise in all answers.",
                            "rationale": "General improvement",
                            "priority": 0.5,
                        }
                    ]
                }
            ),
            tokens_used=80,
        )


class KimiLLMClient(LLMClient):
    """Kimi API (OpenAI-compatible).

    Supports:
    - Developer API: https://api.moonshot.cn/v1 or https://api.moonshot.ai/v1
    - Kimi Code Plan: https://api.kimi.com/coding/v1 (requires Coding Agent headers)
    """

    DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
    CODING_AGENT_HEADERS = {
        "User-Agent": "KimiCLI/1.5",
        "X-Msh-Platform": "kimi_cli",
    }

    def __init__(
        self,
        model: str = "kimi-k2.6",
        api_key: str | None = None,
        base_url: str | None = None,
        disable_thinking: bool = True,
        max_tokens: int = 8192,
        coding_agent: bool | None = None,
    ) -> None:
        from openai import OpenAI

        resolved_key = (
            api_key
            or os.environ.get("KIMI_API_KEY")
            or os.environ.get("MOONSHOT_API_KEY")
        )
        resolved_base = (
            base_url
            or os.environ.get("KIMI_BASE_URL")
            or os.environ.get("MOONSHOT_BASE_URL")
            or self.DEFAULT_BASE_URL
        )
        if coding_agent is None:
            coding_agent = (
                os.environ.get("KIMI_CODING_AGENT", "").lower() in {"1", "true", "yes"}
                or "kimi.com/coding" in resolved_base
                or (resolved_key or "").startswith("sk-kimi-")
            )

        self.model = model
        self.disable_thinking = disable_thinking
        self.max_tokens = max_tokens
        self.coding_agent = coding_agent

        default_headers = self.CODING_AGENT_HEADERS if coding_agent else None
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base,
            default_headers=default_headers,
        )

    def complete(self, system: str, user: str) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
        }
        if self.disable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content or getattr(message, "reasoning_content", "") or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return LLMResponse(content=content, tokens_used=tokens)


class OpenAILLMClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def complete(self, system: str, user: str) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return LLMResponse(content=content, tokens_used=tokens)


class AnthropicLLMClient(LLMClient):
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("Install anthropic: pip install skillopt[anthropic]") from e

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        content = response.content[0].text if response.content else ""
        tokens = (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
        return LLMResponse(content=content, tokens_used=tokens)


class AzureOpenAILLMClient(LLMClient):
    def __init__(
        self,
        deployment: str,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-02-01",
    ) -> None:
        from openai import AzureOpenAI

        self.deployment = deployment
        self.client = AzureOpenAI(
            azure_endpoint=endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=api_version,
        )

    def complete(self, system: str, user: str) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0
        return LLMResponse(content=content, tokens_used=tokens)


# Re-export for backward compatibility
from skillopt.optimizer.prompts import REFLECTION_SYSTEM  # noqa: F401


def _normalize_edit_action(raw: str) -> EditAction:
    mapping = {
        "add": EditAction.ADD,
        "append": EditAction.ADD,
        "insert_after": EditAction.INSERT_AFTER,
        "delete": EditAction.DELETE,
        "replace": EditAction.REPLACE,
    }
    key = raw.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown edit action: {raw}")
    return mapping[key]


def parse_edits_from_response(text: str) -> list[Edit]:
    """Extract edits from optimizer LLM response (flat or Appendix A nested schema)."""
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            raw_edits = data.get("edits")
            if raw_edits is None and isinstance(data.get("patch"), dict):
                raw_edits = data["patch"].get("edits", [])

            if raw_edits is not None:
                edits = [
                    Edit(
                        action=_normalize_edit_action(e.get("action", e.get("op", "add"))),
                        content=e.get("content", ""),
                        target=e.get("target", e.get("anchor", "")),
                        rationale=e.get("rationale", ""),
                        priority=float(e.get("priority", 0.5)),
                        source=e.get("source_type", e.get("source", "")),
                    )
                    for e in raw_edits
                    if e.get("content") or e.get("action") in ("delete",)
                ]
                indices = data.get("selected_indices")
                if indices and edits:
                    picked = []
                    for i in indices:
                        if 0 <= int(i) < len(edits):
                            picked.append(edits[int(i)])
                    if picked:
                        return picked
                return edits
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    from skillopt.core.edit import EditEngine

    return EditEngine.parse_rule_lines(text)


def parse_text_field_response(text: str, *keys: str) -> str:
    """Parse slow_update_content / meta_skill_content from JSON or raw text."""
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            for key in keys:
                if data.get(key):
                    return str(data[key]).strip()
        except json.JSONDecodeError:
            pass
    return text.strip()
