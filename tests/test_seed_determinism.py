"""Test-Plan Layer 2: seed/subset selection is a deterministic function of (content, seed, n).
No network — exercises the pure sampler."""
from cllm.data import _sha256_ids, _stable_sample


def _rows(k):
    return [{"id": f"x-{i}", "_key": f"key-{i:04d}"} for i in range(k)]


def test_sample_is_deterministic():
    a = _stable_sample(_rows(1000), 50, seed=62)
    b = _stable_sample(_rows(1000), 50, seed=62)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_sample_independent_of_input_order():
    rows = _rows(200)
    shuffled = list(reversed(rows))
    a = _sha256_ids([r["id"] for r in _stable_sample(rows, 30, 62)])
    b = _sha256_ids([r["id"] for r in _stable_sample(shuffled, 30, 62)])
    assert a == b  # sorting by _key first removes order dependence


def test_different_seed_changes_subset():
    a = [r["id"] for r in _stable_sample(_rows(1000), 50, 62)]
    b = [r["id"] for r in _stable_sample(_rows(1000), 50, 7)]
    assert a != b


def test_n_caps_at_available():
    assert len(_stable_sample(_rows(10), 50, 62)) == 10
