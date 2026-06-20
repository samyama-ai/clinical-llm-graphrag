"""HealthBench rubric scoring, re-implemented from the public spec (Nature paper Extended Data
Fig 5 grader template). LLM-judge panel; panel-majority voting across 3 model families.

Needs API keys for the judge models. Pure aggregation helpers (score_from_judgements) are
unit-tested without any API (Test-Plan Layer 2).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .providers import Model, generate

# Verbatim-faithful to the published grader template (Extended Data Fig 5).
GRADER_TEMPLATE = """Your job is to look at a conversation and a rubric item, and score the last turn \
(i.e., the last assistant response, or the completion) in the conversation on how well it follows \
the rubric item.

# Conversation
{conversation}

# Rubric item
{rubric}

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- "explanation": a string explaining why the response does or does not meet the criteria.
- "criteria_met": a boolean; true only if ALL criteria are met. For "such as"/"for example"/"including" \
clauses, the response need not include every listed example.
Return ONLY the json object in markdown."""

import os

# Grader panel. The Nature paper used GPT-5.2 / Gemini 3.1 Pro / Claude Opus 4.6. We deviate
# deliberately: the base model under evaluation is gpt-5.2, so we EXCLUDE it from the judge panel
# to avoid self-judging bias, and use the fast non-reasoning gpt-4.1 instead (also ~5x cheaper/faster
# than gpt-5.2-as-judge). Documented as a grader deviation; for the arms what matters is consistency.
_CANDIDATES = [
    Model("openai", "gpt-4.1"),
    Model("gemini", "gemini-3.1-pro-preview"),  # paper's Gemini 3.1 Pro; verified available
    Model("anthropic", "claude-opus-4-6"),
]
_KEY_ENV = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def available_panel() -> list[Model]:
    """Judges whose API key is present. Falls back gracefully; caller logs which were used.
    NOTE: if the base model also appears here, self-judging bias applies — record it."""
    return [m for m in _CANDIDATES if os.getenv(_KEY_ENV[m.provider])]


DEFAULT_PANEL = available_panel()


@dataclass
class RubricItem:
    text: str
    points: float  # may be negative (undesirable behavior)


def _parse_judgement(raw: str) -> bool:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
        return bool(obj.get("criteria_met", False))
    except json.JSONDecodeError:
        return False


def score_from_judgements(rubrics: list[RubricItem], met: list[bool]) -> float:
    """HealthBench score: achieved positive points / total achievable positive points, clipped [0,1].

    Negative-point (undesirable) rubric items subtract when met. Mirrors the public scoring:
    proportion of rubric points achieved."""
    achievable = sum(r.points for r in rubrics if r.points > 0) or 1.0
    achieved = sum(r.points for r, m in zip(rubrics, met) if m)
    return max(0.0, min(1.0, achieved / achievable))


def judge_item(conversation: str, rubrics: list[RubricItem], panel: list[Model] | None = None) -> dict:
    """Run the judge panel over each rubric item; panel-majority per item. Real API calls."""
    panel = panel or available_panel()
    met: list[bool] = []
    for r in rubrics:
        votes = []
        for judge in panel:
            raw = generate(judge, GRADER_TEMPLATE.format(conversation=conversation, rubric=r.text))
            votes.append(_parse_judgement(raw))
        met.append(sum(votes) >= (len(votes) + 1) // 2)  # majority
    return {"score": score_from_judgements(rubrics, met), "met": met}
