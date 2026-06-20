"""MedQA answer extraction + exact-match scoring. No API key needed (pure parsing).

Mirrors the Nature-paper approach (a final-answer extraction with a regex fallback). We extract
the chosen option letter from free-text model output and compare to the gold letter.
"""
from __future__ import annotations

import re

# Markdown emphasis / formatting that models wrap answers in (e.g. "Answer: **D**", "`C`").
_FORMAT = re.compile(r"[*_`#>]+")

_PATTERNS = [
    # "The answer is (C)" / "Answer: C" / "final answer - C" (formatting already stripped).
    re.compile(r"(?:final\s+answer|answer|correct\s+option|best\s+option)\s*(?:is)?\s*[:\-]?\s*\(?([A-E])\)?\b", re.I),
    # a parenthesized letter "(C)"
    re.compile(r"\(([A-E])\)"),
    # a letter immediately followed by a delimiter, e.g. "C)" / "C." / "C:" / "C -"
    re.compile(r"\b([A-E])\s*[\).:\-]"),
    # a lone capital letter on its own (last resort)
    re.compile(r"\b([A-E])\b"),
]


def extract_letter(text: str, options: dict | None = None) -> str | None:
    """Return the chosen option letter (A-E) or None if unparseable.

    Robust to markdown formatting. Tries explicit answer phrasing first, then parenthesized /
    delimited letters, then a unique option-text match, then a lone trailing letter. Always takes
    the LAST match (models restate options then conclude)."""
    if not text:
        return None
    clean = _FORMAT.sub("", text).strip()
    for pat in _PATTERNS[:-1]:
        m = None
        for m in pat.finditer(clean):
            pass
        if m:
            return m.group(1).upper()
    if options:
        hits = [k for k, v in options.items() if v and v.lower() in clean.lower()]
        if len(hits) == 1:
            return hits[0].upper()
    # last resort: a lone trailing letter in the final two lines
    tail = "\n".join(clean.splitlines()[-2:])
    m = None
    for m in _PATTERNS[-1].finditer(tail):
        pass
    return m.group(1).upper() if m else None


def extract_letter_llm(text: str, options: dict, extractor_model: str = "gpt-4.1-mini") -> str | None:
    """LLM-based extraction fallback (the Nature paper's method: GPT-4.1 extracts the final answer,
    regex as cross-check). Real API call; used only when regex fails. Returns A-E or None."""
    from .providers import Model, generate

    opts = "\n".join(f"({k}) {v}" for k, v in options.items())
    prompt = (
        "Extract ONLY the single final answer letter (A-E) the response selected. "
        "If none is clearly selected, reply NONE.\n\n"
        f"Options:\n{opts}\n\nResponse:\n{text}\n\nFinal answer letter:"
    )
    out = generate(Model("openai", extractor_model), prompt, max_tokens=8).strip().upper()
    m = re.search(r"[A-E]", out)
    return m.group(0) if m else None


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
