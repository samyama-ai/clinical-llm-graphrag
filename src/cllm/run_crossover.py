"""Pillar-1 crossover experiment on MedQA: does KG grounding's lift shrink as the base model
strengthens? Predicts A_kg - A0 > 0 for weak models, -> 0 (or <0) for strong models.

Per-item PrimeKG retrieval is computed ONCE and reused across all base models. Arms:
  A0     = model alone           A_flat = + flat node names (no structure)
  A_kg   = + retrieved subgraph (relations)
Metric: MedQA exact-match (no judge cost). One process (embedded samyama-graph in-memory).
"""
from __future__ import annotations

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path

from . import primekg
from .providers import Model, generate
from .run_arms import embed_query, flat_context, setup_kg
from .score_medqa import accuracy, extract_letter, extract_letter_llm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# weak -> strong ladder (Pillar-1: lift should decrease left to right)
LADDER = [Model("openai", m) for m in ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-5.2"]]
WORKERS = 8

PROMPT = ("You are answering a USMLE-style medical question. Choose the single best option.\n\n"
          "{q}\n\n{opts}\n{ctx}Respond with the letter of the best option and a one-line rationale. "
          "End with 'Answer: <letter>'.")


def _ctx_block(kind: str, kg: str, flat: str) -> str:
    if kind == "A0":
        return ""
    body = kg if kind == "A_kg" else flat
    label = "Relevant knowledge-graph facts" if kind == "A_kg" else "Relevant medical concepts"
    return f"[{label} (use only if relevant):\n{body}\n]\n\n" if body else ""


def _gen_score(model, it, kind, kg, flat):
    opts = "\n".join(f"({k}) {v}" for k, v in it["options"].items())
    prompt = PROMPT.format(q=it["question"], opts=opts, ctx=_ctx_block(kind, kg, flat))
    try:
        out = generate(model, prompt, max_tokens=4000)
    except Exception as e:
        out = f"__error__: {e}"
    pred = extract_letter(out, it["options"])
    if pred is None and out and not out.startswith("__error__"):
        pred = extract_letter_llm(out, it["options"])
    gold = (it["gold"] or "").upper()
    return {"pred": pred, "gold": gold, "correct": bool(pred) and pred == gold, "parsed": pred is not None}


def run(n: int, limit_edges):
    client = setup_kg(limit_edges)
    items = [json.loads(l) for l in (DATA / "medqa.jsonl").read_text().splitlines()][:n]

    # Pre-compute retrieval ONCE per item (model-independent).
    print(f"[crossover] retrieving PrimeKG context for {len(items)} items...")
    ctx = {}
    for it in items:
        q = it["question"] + " " + " ".join(it["options"].values())
        emb = embed_query(q[:2000])
        ctx[it["id"]] = (primekg.retrieve(client, emb), flat_context(client, emb))

    summary = {}
    for model in LADDER:
        row = {}
        for kind in ("A0", "A_flat", "A_kg"):
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                recs = list(ex.map(lambda it: _gen_score(model, it, kind, *ctx[it["id"]]), items))
            agg = accuracy(recs)
            p = agg["accuracy"]; se = sqrt(p * (1 - p) / max(1, agg["n"]))
            row[kind] = {"acc": round(100 * p, 1), "ci": round(100 * 1.96 * se, 1), "parse": round(100 * agg["parse_rate"], 1)}
        row["lift_kg"] = round(row["A_kg"]["acc"] - row["A0"]["acc"], 1)
        row["lift_flat"] = round(row["A_flat"]["acc"] - row["A0"]["acc"], 1)
        summary[model.name] = row
        print(f"[crossover] {model.name}: A0={row['A0']['acc']} A_flat={row['A_flat']['acc']} "
              f"A_kg={row['A_kg']['acc']} | lift_kg={row['lift_kg']} lift_flat={row['lift_flat']}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"crossover_medqa_n{len(items)}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Pillar-1 crossover (lift_kg should decrease weak->strong) ===")
    for m in LADDER:
        print(f"  {m.name:14s} lift_kg={summary[m.name]['lift_kg']:+.1f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--limit-edges", type=int, default=None)
    a = ap.parse_args()
    run(a.n, a.limit_edges)


if __name__ == "__main__":
    main()
