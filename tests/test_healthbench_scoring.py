"""Test-Plan Layer 2: HealthBench score aggregation + judgement parsing (no API)."""
from cllm.score_healthbench import RubricItem, _parse_judgement, score_from_judgements


def test_score_all_met_is_one():
    rubrics = [RubricItem("a", 2), RubricItem("b", 3)]
    assert score_from_judgements(rubrics, [True, True]) == 1.0


def test_score_partial():
    rubrics = [RubricItem("a", 2), RubricItem("b", 2)]
    assert score_from_judgements(rubrics, [True, False]) == 0.5


def test_negative_points_penalize_when_met():
    # undesirable behavior (negative points) met -> lowers score below the positive-only fraction
    rubrics = [RubricItem("good", 4), RubricItem("harmful", -2)]
    s_clean = score_from_judgements(rubrics, [True, False])
    s_harm = score_from_judgements(rubrics, [True, True])
    assert s_clean == 1.0
    assert s_harm < s_clean


def test_score_clipped_nonnegative():
    rubrics = [RubricItem("good", 2), RubricItem("harmful", -5)]
    assert score_from_judgements(rubrics, [False, True]) == 0.0


def test_parse_judgement_json():
    assert _parse_judgement('```json\n{"explanation":"ok","criteria_met":true}\n```') is True
    assert _parse_judgement('{"criteria_met": false}') is False
    assert _parse_judgement("not json") is False
