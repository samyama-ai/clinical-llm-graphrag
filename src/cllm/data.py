"""Build the frozen, reproducible seed=62 subsets of MedQA and HealthBench.

No API key required. Writes:
  data/medqa.jsonl, data/healthbench.jsonl   (the sampled items)
  data/manifest.json                         (counts + sha256 of the ordered id lists)

The manifest hashes make the subsets byte-stable across runs/machines (Test-Plan Layer 2,
"seed/subset determinism"). Matches the Nature-paper protocol: 500 items each, seed=62.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

MEDQA_HF = "GBaker/MedQA-USMLE-4-options"
HEALTHBENCH_HF = "openai/healthbench"
# HealthBench ships as separate JSONL files with distinct schemas; pick explicitly.
HEALTHBENCH_FILES = {
    "healthbench": "2025-05-07-06-14-12_oss_eval.jsonl",   # full set (Gate B vs Nature 88.0)
    "healthbench_hard": "hard_2025-05-08-21-00-10.jsonl",  # HealthBench Hard (H1 headline)
}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _sha256_ids(ids: list[str]) -> str:
    h = hashlib.sha256()
    for i in ids:
        h.update(str(i).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _stable_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Deterministic sample: sort by a stable per-row key, then seeded shuffle, take n.

    Sorting first removes any dependence on the dataset's row order, so the subset is a
    pure function of (content, seed, n)."""
    rows = sorted(rows, key=lambda r: r["_key"])
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    take = idx[: min(n, len(rows))]
    take.sort()
    return [rows[i] for i in take]


def _load_medqa() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(MEDQA_HF, split="test")
    out = []
    for i, r in enumerate(ds):
        # schema: question, options (dict letter->text or list), answer (text), answer_idx (letter)
        opts = r.get("options")
        if isinstance(opts, dict):
            options = {k: opts[k] for k in sorted(opts)}
        else:  # list -> A,B,C,D
            options = {chr(65 + j): t for j, t in enumerate(opts)}
        gold = r.get("answer_idx") or r.get("answer_letter")
        if gold is None:  # fall back: match answer text to an option letter
            ans = r.get("answer")
            gold = next((k for k, v in options.items() if v == ans), None)
        item = {
            "id": f"medqa-{i:04d}",
            "question": r["question"],
            "options": options,
            "gold": gold,
            "_key": hashlib.sha256(r["question"].encode("utf-8")).hexdigest(),
        }
        out.append(item)
    return out


def _load_healthbench(variant: str) -> list[dict]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HEALTHBENCH_HF, HEALTHBENCH_FILES[variant], repo_type="dataset")
    ds = load_dataset("json", data_files=path, split="train")
    out = []
    for i, r in enumerate(ds):
        prompt = r.get("prompt") or r.get("conversation") or r.get("question")
        rubrics = r.get("rubrics") or r.get("rubric") or r.get("criteria") or []
        item = {
            "id": r.get("prompt_id") or f"{variant}-{i:04d}",
            "prompt": prompt,
            "rubrics": rubrics,
            "hard": variant == "healthbench_hard",
            "_key": hashlib.sha256(json.dumps(prompt, sort_keys=True, default=str).encode()).hexdigest(),
        }
        out.append(item)
    return out


def build(n: int, seed: int) -> dict:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"n": n, "seed": seed, "sources": {MEDQA_HF: "medqa", HEALTHBENCH_HF: list(HEALTHBENCH_FILES)}}
    loaders = [
        ("medqa", _load_medqa),
        ("healthbench", lambda: _load_healthbench("healthbench")),
        ("healthbench_hard", lambda: _load_healthbench("healthbench_hard")),
    ]
    for name, loader in loaders:
        rows = loader()
        sample = _stable_sample(rows, n, seed)
        for r in sample:
            r.pop("_key", None)
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w") as f:
            for r in sample:
                f.write(json.dumps(r, default=str) + "\n")
        manifest[name] = {
            "total_available": len(rows),
            "sampled": len(sample),
            "ids_sha256": _sha256_ids([r["id"] for r in sample]),
        }
        print(f"[data] {name}: sampled {len(sample)}/{len(rows)} -> {path}")
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[data] wrote {DATA_DIR/'manifest.json'}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=62)
    args = ap.parse_args()
    build(args.n, args.seed)


if __name__ == "__main__":
    main()
