"""Minibatch reflection — paper-aligned backward pass with merge, rank, refinement."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from skillopt.core.edit import Edit, EditAction
from skillopt.core.skill import SkillDocument
from skillopt.core.trajectory import Trajectory
from skillopt.llm.client import LLMClient, parse_edits_from_response, parse_text_field_response
from skillopt.optimizer.prompts import (
    FAILURE_ANALYST_SYSTEM,
    MERGE_FAILURE_SYSTEM,
    MERGE_FINAL_SYSTEM,
    MERGE_SUCCESS_SYSTEM,
    META_SKILL_SYSTEM,
    RANK_SYSTEM,
    REWRITE_SYSTEM,
    SLOW_UPDATE_SYSTEM,
    SUCCESS_ANALYST_SYSTEM,
)
from skillopt.optimizer.slow_update import SlowUpdateEvidence


class ReflectionEngine:
    """Turn rollout trajectories into structured skill edit proposals."""

    def __init__(
        self,
        optimizer_client: LLMClient,
        minibatch_size: int = 4,
        workers: int = 1,
        refinement_rounds: int = 3,
        merge_batch_size: int = 8,
        workspace_root: str | None = None,
        merge_workers: int = 1,
    ) -> None:
        self.optimizer = optimizer_client
        self.minibatch_size = minibatch_size
        self.workers = max(1, workers)
        self.merge_workers = max(1, merge_workers)
        self.refinement_rounds = max(1, refinement_rounds)
        self.merge_batch_size = merge_batch_size
        self.workspace_root = Path(workspace_root) if workspace_root else None

    def reflect(
        self,
        trajectories: list[Trajectory],
        skill: SkillDocument,
        rejected_summary: str = "",
        meta_skill: str = "",
    ) -> list[Edit]:
        failures = [t for t in trajectories if not t.success]
        successes = [t for t in trajectories if t.success]

        failure_edits = self._reflect_group(
            failures, skill, "failure", rejected_summary, meta_skill
        )
        success_edits = self._reflect_group(
            successes, skill, "success", rejected_summary, meta_skill
        )

        merged_failures = self._merge_edits_llm(
            failure_edits, skill, mode="failure", rejected_summary=rejected_summary
        )
        merged_successes = self._merge_edits_llm(
            success_edits, skill, mode="success", rejected_summary=rejected_summary
        )

        combined = self._merge_final_llm(
            merged_failures, merged_successes, skill, rejected_summary
        )
        if not combined:
            combined = self._combine_with_failure_priority(merged_failures, merged_successes)
        return self._rank_edits(combined, skill, rejected_summary, meta_skill)

    def rewrite_skill(
        self,
        skill: SkillDocument,
        selected_edits: list[Edit],
    ) -> str:
        user = (
            f"Current skill:\n{skill.content}\n\n"
            f"Apply these suggestions:\n"
            + "\n".join(f"- {e.content} ({e.rationale})" for e in selected_edits)
        )
        response = self.optimizer.complete(REWRITE_SYSTEM, user)
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return text or skill.content

    def slow_update(
        self,
        prev_skill: SkillDocument,
        curr_skill: SkillDocument,
        evidence: SlowUpdateEvidence,
    ) -> str:
        user = (
            f"Previous skill:\n{prev_skill.content}\n\n"
            f"Current skill:\n{curr_skill.content}\n\n"
            f"{evidence.summary()}\n\n"
            "Write concise longitudinal guidance for the slow-update region."
        )
        response = self.optimizer.complete(SLOW_UPDATE_SYSTEM, user)
        return parse_text_field_response(response.content, "slow_update_content")

    def build_meta_skill(
        self,
        accepted_patterns: list[str],
        rejected_patterns: list[str],
        persistent_failures: list[str],
    ) -> str:
        user = (
            "Accepted edit patterns:\n"
            + "\n".join(f"- {p}" for p in accepted_patterns[-10:])
            + "\n\nRejected edit patterns:\n"
            + "\n".join(f"- {p}" for p in rejected_patterns[-10:])
            + "\n\nPersistent failure themes:\n"
            + "\n".join(f"- {p}" for p in persistent_failures[-10:])
        )
        response = self.optimizer.complete(META_SKILL_SYSTEM, user)
        return parse_text_field_response(response.content, "meta_skill_content")

    def _reflect_group(
        self,
        trajectories: list[Trajectory],
        skill: SkillDocument,
        mode: str,
        rejected_summary: str,
        meta_skill: str,
    ) -> list[Edit]:
        batches = _chunk(trajectories, self.minibatch_size)
        if not batches:
            return []

        all_edits: list[Edit] = []

        def process_batch(batch: list[Trajectory]) -> list[Edit]:
            edits: list[Edit] = []
            context_extra = ""
            for round_idx in range(self.refinement_rounds):
                batch_edits = self._reflect_batch(
                    batch,
                    skill,
                    mode,
                    rejected_summary,
                    meta_skill,
                    context_extra,
                )
                if not batch_edits:
                    break
                edits = batch_edits
                context_extra = (
                    f"Prior round {round_idx + 1} proposals:\n"
                    + "\n".join(str(e) for e in edits[:5])
                )
            return edits

        if self.workers <= 1 or len(batches) <= 1:
            for batch in batches:
                all_edits.extend(process_batch(batch))
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(process_batch, b) for b in batches]
                for future in as_completed(futures):
                    all_edits.extend(future.result())

        return self._dedupe_edits(all_edits)

    def _reflect_batch(
        self,
        batch: list[Trajectory],
        skill: SkillDocument,
        mode: str,
        rejected_summary: str = "",
        meta_skill: str = "",
        refinement_context: str = "",
    ) -> list[Edit]:
        if not batch:
            return []

        traj_text = "\n---\n".join(
            self._trajectory_context(t) for t in batch
        )
        system = FAILURE_ANALYST_SYSTEM if mode == "failure" else SUCCESS_ANALYST_SYSTEM
        context_parts = [
            f"Mode: {mode} analysis",
            f"Current skill ({skill.token_estimate} tokens):\n{skill.content}",
            f"Trajectories:\n{traj_text}",
        ]
        if rejected_summary:
            context_parts.append(rejected_summary)
        if meta_skill:
            context_parts.append(f"Meta guidance:\n{meta_skill}")
        if refinement_context:
            context_parts.append(refinement_context)

        response = self.optimizer.complete(
            system,
            "\n\n".join(context_parts),
        )
        return parse_edits_from_response(response.content)

    def _trajectory_context(self, traj: Trajectory) -> str:
        parts = [traj.summary()]
        if self.workspace_root:
            trace_path = self.workspace_root / traj.task_id / "codex_trace_summary.txt"
            if trace_path.is_file():
                parts.append(f"codex_trace_summary.txt:\n{trace_path.read_text(encoding='utf-8')[:2000]}")
        return "\n".join(parts)

    def _merge_edits_llm(
        self,
        edits: list[Edit],
        skill: SkillDocument,
        mode: str,
        rejected_summary: str = "",
    ) -> list[Edit]:
        if not edits:
            return []
        system = MERGE_FAILURE_SYSTEM if mode == "failure" else MERGE_SUCCESS_SYSTEM
        if len(edits) <= self.merge_batch_size:
            chunks = [edits]
        else:
            chunks = _chunk(edits, self.merge_batch_size)

        merged: list[Edit] = []
        chunks_list = chunks

        def merge_chunk(chunk: list[Edit]) -> list[Edit]:
            user = (
                f"Mode: {mode}\n"
                f"Current skill:\n{skill.content}\n\n"
                f"Edits to merge:\n"
                + "\n".join(str(e) for e in chunk)
            )
            if rejected_summary:
                user += f"\n\n{rejected_summary}"
            response = self.optimizer.complete(system, user)
            return parse_edits_from_response(response.content)

        if self.merge_workers <= 1 or len(chunks_list) <= 1:
            for chunk in chunks_list:
                merged.extend(merge_chunk(chunk))
        else:
            with ThreadPoolExecutor(max_workers=self.merge_workers) as pool:
                for result in pool.map(merge_chunk, chunks_list):
                    merged.extend(result)
        return self._dedupe_edits(merged)

    def _merge_final_llm(
        self,
        failure_edits: list[Edit],
        success_edits: list[Edit],
        skill: SkillDocument,
        rejected_summary: str,
    ) -> list[Edit]:
        if not failure_edits and not success_edits:
            return []
        user = (
            f"Current skill:\n{skill.content}\n\n"
            f"Failure edits:\n" + "\n".join(str(e) for e in failure_edits)
            + f"\n\nSuccess edits:\n" + "\n".join(str(e) for e in success_edits)
        )
        if rejected_summary:
            user += f"\n\n{rejected_summary}"
        response = self.optimizer.complete(MERGE_FINAL_SYSTEM, user)
        merged = parse_edits_from_response(response.content)
        return merged if merged else []

    def _rank_edits(
        self,
        edits: list[Edit],
        skill: SkillDocument,
        rejected_summary: str,
        meta_skill: str,
    ) -> list[Edit]:
        if not edits:
            return []
        user = (
            f"Current skill:\n{skill.content}\n\n"
            f"Candidate edits:\n"
            + "\n".join(str(e) for e in edits)
        )
        if rejected_summary:
            user += f"\n\n{rejected_summary}"
        if meta_skill:
            user += f"\n\nMeta guidance:\n{meta_skill}"

        response = self.optimizer.complete(RANK_SYSTEM, user)
        ranked = parse_edits_from_response(response.content)
        return ranked if ranked else sorted(edits, key=lambda e: e.priority, reverse=True)

    @staticmethod
    def _combine_with_failure_priority(
        failure_edits: list[Edit], success_edits: list[Edit]
    ) -> list[Edit]:
        combined = list(failure_edits)
        seen = {e.content.strip().lower() for e in failure_edits}
        for edit in success_edits:
            key = edit.content.strip().lower()
            if key not in seen:
                combined.append(edit)
                seen.add(key)
        return combined

    @staticmethod
    def _dedupe_edits(edits: list[Edit]) -> list[Edit]:
        seen: set[str] = set()
        merged: list[Edit] = []
        for edit in sorted(edits, key=lambda e: e.priority, reverse=True):
            key = edit.content.strip().lower()
            if key not in seen:
                seen.add(key)
                merged.append(edit)
        return merged


def _chunk(items: list, size: int) -> list[list]:
    if size <= 0:
        return [items] if items else []
    return [items[i : i + size] for i in range(0, len(items), size)]
