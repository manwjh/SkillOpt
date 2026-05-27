"""Benchmark-specific scorers (extensible registry)."""

from __future__ import annotations

import re
from typing import Callable

Scorer = Callable[[str, str | None, dict], float]


def exact_match(answer: str, expected: str | None, metadata: dict | None = None) -> float:
    if expected is None:
        return 1.0 if answer.strip() else 0.0
    a = _normalize(answer)
    e = _normalize(expected)
    if a == e:
        return 1.0
    if e in a:
        return 0.85
    return 0.0


def keyword_match(answer: str, expected: str | None, metadata: dict | None = None) -> float:
    """For procedural benchmarks where expected is a keyword tag."""
    if expected is None:
        return 1.0 if answer.strip() else 0.0
    meta = metadata or {}
    keywords = meta.get("keywords", [expected])
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    if hits >= len(keywords):
        return 1.0
    if hits > 0:
        return 0.5
    return 0.0


def spreadsheet_scorer(answer: str, expected: str | None, metadata: dict | None = None) -> float:
    """Heuristic scorer for spreadsheet-style procedural tasks."""
    meta = metadata or {}
    answer_lower = answer.lower()
    required = {
        "static_values": ["static", "evaluated", "materialized", "computed value"],
        "static_sum": ["sum", "static", "computed"],
        "normalized_lookup": ["normaliz", "lookup", "key"],
        "structure_first": ["structure", "sheet", "inspect", "workbook"],
        "full_range": ["full range", "blank", "complete", "all cells"],
    }
    tag = expected or ""
    kws = required.get(tag, meta.get("keywords", [tag]))
    if not kws or kws == [""]:
        return exact_match(answer, expected, metadata)
    hits = sum(1 for kw in kws if kw.lower() in answer_lower)
    ratio = hits / len(kws)
    if ratio >= 0.6:
        return 1.0
    if ratio >= 0.3:
        return 0.5
    return 0.0


SCORERS: dict[str, Scorer] = {
    "exact": exact_match,
    "keyword": keyword_match,
    "spreadsheet": spreadsheet_scorer,
}


def get_scorer(name: str) -> Scorer:
    return SCORERS.get(name, exact_match)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)
