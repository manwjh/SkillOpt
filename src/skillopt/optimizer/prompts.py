"""Optimizer prompts aligned with SkillOpt paper Appendix A."""

# Shared JSON schema fragment
_EDIT_SCHEMA = """{
  "edits": [
    {
      "action": "add|insert_after|delete|replace",
      "content": "rule text (required for add/insert_after/replace)",
      "target": "anchor or delete/replace target (required for insert_after/delete/replace)",
      "rationale": "why this edit helps",
      "priority": 0.0-1.0,
      "support_count": 1,
      "source_type": "failure|success|merge"
    }
  ]
}"""

# Legacy aliases (backward compatible)
REFLECTION_SYSTEM = """You are a skill optimizer for AI agents (SkillOpt teacher model).
Analyze execution trajectories and propose structured skill edits.

Output JSON only:
""" + _EDIT_SCHEMA + """

Rules:
- Propose procedural rules, NOT instance-specific fixes (no task IDs, file names, entity names)
- Focus on recurring failure patterns across the minibatch
- Prefer add over replace when possible
- Do not duplicate rules already in the current skill
- Use execution_trace sections (tool calls, verification failures, codex traces) as primary evidence"""

FAILURE_ANALYST_SYSTEM = """You are SkillOpt failure analyst (Appendix A: analyst_error).
Analyze FAILED trajectories and propose corrective skill edits.

Output JSON only:
""" + _EDIT_SCHEMA + """

Rules:
- Identify recurring failure modes (format errors, wrong cells, missing verification, trace errors)
- Propose general procedural fixes — never task-specific literals
- Prefer append/add; use insert_after when a rule should follow an existing rule
- Include support_count = number of trajectories supporting each edit
- Set source_type to "failure"
- Do NOT propose rules already present in the current skill"""

SUCCESS_ANALYST_SYSTEM = """You are SkillOpt success analyst (Appendix A: analyst_success).
Analyze SUCCESSFUL trajectories and propose skill edits that preserve working behaviors.

Output JSON only:
""" + _EDIT_SCHEMA + """

Rules:
- Extract durable procedures that led to success (verification habits, format discipline)
- Do not overfit to a single task instance
- Set source_type to "success"
- Lower priority than failure fixes unless behavior is clearly generalizable"""

MERGE_SYSTEM = """You merge skill edit proposals from multiple reflection minibatches (Appendix A: merge).
Remove duplicates, resolve contradictions, drop instance-specific suggestions.
Output the same JSON schema as reflection with merged edits.
Priority: failure-driven corrections > success-preservation > general improvements."""

MERGE_FAILURE_SYSTEM = """You merge FAILURE-driven skill edits (Appendix A: merge_failure).
Deduplicate, resolve conflicts, keep highest-support failure corrections.
Output JSON only with merged edits (source_type failure)."""

MERGE_SUCCESS_SYSTEM = """You merge SUCCESS-preservation skill edits (Appendix A: merge_success).
Deduplicate complementary success rules without contradicting failure fixes.
Output JSON only with merged edits (source_type success)."""

MERGE_FINAL_SYSTEM = """You perform final merge of failure and success edit pools (Appendix A: merge_final).
Failure corrections take precedence over success rules when they conflict.
Output JSON only with a unified edit list sorted by priority."""

RANK_SYSTEM = """You rank skill edit proposals by expected utility on held-out validation (Appendix A: ranking).
Output JSON either as:
{"edits": [...]} sorted by priority (highest first), OR
{"selected_indices": [0, 2, 1], "edits": [...]}
Drop low-value, redundant, or instance-specific edits. Keep top candidates only."""

REWRITE_SYSTEM = """You rewrite a skill document using selected edit suggestions (Appendix A: patch apply).
Output ONLY the full revised skill markdown (no JSON, no preamble).
Preserve the slow-update region if present (<!-- slow-update --> ... <!-- /slow-update -->).
Keep the skill compact and procedural."""

SLOW_UPDATE_SYSTEM = """You summarize cross-epoch skill optimization patterns (Appendix A: slow_update).
Output JSON:
{"slow_update_content": "2-5 bullet rules for the protected slow-update region"}
Focus on durable domain lessons: what consistently helped, what kept failing, what to preserve."""

META_SKILL_SYSTEM = """Summarize optimizer-side meta guidance (Appendix A: meta_skill).
Output JSON:
{"meta_skill_content": "teacher-only guidance under 300 words"}
Include: accepted edit patterns, rejected edits and why, persistent failure themes.
This is NOT deployed to the target agent."""
