"""Test-Plan Layer 2: subset selection matches the paper's method
`random.Random(seed).sample(rows_in_source_order, n)` — deterministic given (source order, seed, n).
No network."""
from cllm.data import _match_sample


def _rows(k):
    return [{"id": f"x-{i}"} for i in range(k)]


def test_sample_is_deterministic():
    a = _match_sample(_rows(1000), 50, seed=62)
    b = _match_sample(_rows(1000), 50, seed=62)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_sample_depends_on_source_order():
    # Matching the paper REQUIRES preserving source order (random.sample is order-sensitive).
    rows = _rows(200)
    a = {r["id"] for r in _match_sample(rows, 30, 62)}
    b = {r["id"] for r in _match_sample(list(reversed(rows)), 30, 62)}
    assert a != b


def test_matches_reference_random_sample():
    # exact equivalence to the paper's call on the same source order
    import random
    rows = _rows(500)
    expected = random.Random(62).sample(rows, 100)
    assert _match_sample(rows, 100, 62) == expected


def test_different_seed_changes_subset():
    a = [r["id"] for r in _match_sample(_rows(1000), 50, 62)]
    b = [r["id"] for r in _match_sample(_rows(1000), 50, 7)]
    assert a != b


def test_n_caps_at_available():
    assert len(_match_sample(_rows(10), 50, 62)) == 10
