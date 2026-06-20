"""MedQA answer extraction + exact-match scoring. No API key needed (pure parsing).

Mirrors the Nature-paper approach (a final-answer extraction with a regex fallback). We extract
the chosen option letter from free-text model output and compare to the gold letter.
"""
from __future__ import annotations

import re

_PATTERNS = [
    # "The answer is (C)" / "Answer: C" / "final answer: C."
    re.compile(r"(?:final\s+answer|answer)\s*(?:is)?\s*[:\-]?\s*\(?([A-E])\)?\b", re.I),
    # a lone "(C)" or "C)" near the end
    re.compile(r"\(([A-E])\)"),
    re.compile(r"\b([A-E])[\).:]"),
]


def extract_letter(text: str, options: dict | None = None) -> str | None:
    """Return the chosen option letter (A-E) or None if unparseable.

    Tries explicit answer phrasing first, then bare parenthesized letters, then (if options
    given) a unique option-text match."""
    if not text:
        return None
    tail = text.strip()
    for pat in _PATTERNS:
        m = None
        for m in pat.finditer(tail):
            pass  # take the LAST match (models often restate then conclude)
        if m:
            return m.group(1).upper()
    if options:
        hits = [k for k, v in options.items() if v and v.lower() in tail.lower()]
        if len(hits) == 1:
            return hits[0].upper()
    return None


def score_item(model_output: str, gold: str, options: dict | None = None) -> dict:
    pred = extract_letter(model_output, options)
    return {"pred": pred, "gold": (gold or "").upper(), "correct": bool(pred) and pred == (gold or "").upper(),
            "parsed": pred is not None}


def accuracy(records: list[dict]) -> dict:
    n = len(records)
    correct = sum(r["correct"] for r in records)
    parsed = sum(r["parsed"] for r in records)
    return {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0,
            "parse_rate": parsed / n if n else 0.0}
