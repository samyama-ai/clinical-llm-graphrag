"""Test-Plan Layer 2: MedQA extraction/scoring correctness on a real hand-checked fixture.
No service mocks — these are real model-output strings with known-correct labels."""
import json
from pathlib import Path

from cllm.score_medqa import accuracy, extract_letter, score_item

FIX = Path(__file__).parent / "fixtures" / "medqa_outputs.jsonl"


def _load():
    return [json.loads(l) for l in FIX.read_text().splitlines()]


def test_extraction_matches_hand_labels():
    rows = _load()
    for r in rows:
        rec = score_item(r["output"], r["gold"])
        assert rec["correct"] == r["expect_correct"], r


def test_last_match_wins():
    # models often restate options then conclude; we must take the final stated answer.
    assert extract_letter("Could be (A) or (B). Answer: B") == "B"


def test_markdown_bold_answer():
    # regression: gpt-5.2 emits "Answer: **D**"; markdown must not break extraction
    assert extract_letter("**D** — rationale.\nAnswer: **D**") == "D"
    assert extract_letter("the answer is **(B) Foo**") == "B"
    assert extract_letter("I pick `C`.") == "C"


def test_unparseable_returns_none():
    assert extract_letter("It depends on several factors.") is None


def test_accuracy_aggregate():
    rows = _load()
    recs = [score_item(r["output"], r["gold"]) for r in rows]
    agg = accuracy(recs)
    assert agg["n"] == 6
    assert agg["correct"] == 4  # four expect_correct=true rows are parseable + match
    assert 0.0 <= agg["accuracy"] <= 1.0
