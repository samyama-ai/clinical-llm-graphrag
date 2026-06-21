"""Agentic crossover on MedQA: A0 vs A_kg (naive triple-RAG) vs A_agent (LLM-written Cypher over
samyama-graph). Tests whether agentic graph-querying (paper-3 data-layer thesis) provides decisive
grounding where naive retrieval failed — and whether the benefit follows Pillar-1 across model
strengths. Per-question retrieval (naive + agentic) is computed ONCE and reused across base models.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path

from . import primekg
from .agentic import agentic_retrieve
from .providers import Model, generate
from .run_arms import embed_query, flat_context
from .score_medqa import accuracy, extract_letter, extract_letter_llm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LADDER = [Model("openai", m) for m in ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-5.2"]]
WORKERS = 8

PROMPT = ("You are answering a USMLE-style medical question. Choose the single best option.\n\n"
          "{q}\n\n{opts}\n{ctx}Respond with the letter of the best option and a one-line rationale. "
          "End with 'Answer: <letter>'.")


def _ctx_block(kind, naive, agent):
    if kind == "A0":
        return ""
    if kind == "A_kg":
        return f"[Knowledge-graph facts (use only if relevant):\n{naive}\n]\n\n" if naive else ""
    return f"[Knowledge-graph query results (use only if relevant):\n{agent}\n]\n\n" if agent else ""


def _gen_score(model, it, kind, naive, agent):
    opts = "\n".join(f"({k}) {v}" for k, v in it["options"].items())
    prompt = PROMPT.format(q=it["question"], opts=opts, ctx=_ctx_block(kind, naive, agent))
    try:
        out = generate(model, prompt, max_tokens=4000)
    except Exception as e:
        out = f"__error__: {e}"
    pred = extract_letter(out, it["options"])
    if pred is None and out and not out.startswith("__error__"):
        pred = extract_letter_llm(out, it["options"])
    gold = (it["gold"] or "").upper()
    return {"correct": bool(pred) and pred == gold, "parsed": pred is not None}


def setup_full(limit_edges=None):
    from samyama import SamyamaClient
    client = SamyamaClient.embedded()
    primekg.load_into(client, limit_edges=limit_edges)  # full edges (needed for agentic Cypher traversal)
    pids, embs = primekg.embed_nodes()
    nid2info = primekg.build_vector_index(client, pids, embs)
    adjacency = primekg.build_adjacency()
    return client, nid2info, adjacency


def run(n: int, limit_edges=None):
    client, nid2info, adjacency = setup_full(limit_edges)
    items = [json.loads(l) for l in (DATA / "medqa.jsonl").read_text().splitlines()][:n]

    print(f"[agentic] retrieving (naive + agentic Cypher) for {len(items)} items...")
    ctx = {}
    ag_rows = []
    for i, it in enumerate(items):
        q = it["question"] + " " + " ".join(it["options"].values())
        emb = embed_query(q[:2000])
        hits = client.vector_search("Entity", "emb", [float(x) for x in emb], 25)
        cands = [nid2info[nid][1] for nid, _ in hits if nid in nid2info]
        naive = primekg.retrieve(client, emb, nid2info, adjacency)
        ag = agentic_retrieve(client, q[:1500], cands)
        ctx[it["id"]] = (naive, ag["facts"])
        ag_rows.append({"id": it["id"], "n_rows": ag["n_rows"], "attempts": ag["attempts"], "cypher": ag["cypher"][:200]})
        if (i + 1) % 20 == 0:
            print(f"[agentic] retrieved {i+1}/{len(items)} (last n_rows={ag['n_rows']}, attempts={ag['attempts']})")
    hit_rate = sum(1 for r in ag_rows if r["n_rows"] > 0) / len(ag_rows)
    print(f"[agentic] Cypher non-empty rate: {100*hit_rate:.0f}%")
    (RESULTS / f"agentic_cypher_n{len(items)}.jsonl").write_text("\n".join(json.dumps(r) for r in ag_rows))

    summary = {}
    for model in LADDER:
        row = {}
        for kind in ("A0", "A_kg", "A_agent"):
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                recs = list(ex.map(lambda it: _gen_score(model, it, kind, *ctx[it["id"]]), items))
            agg = accuracy(recs)
            p = agg["accuracy"]; se = sqrt(p * (1 - p) / max(1, agg["n"]))
            row[kind] = {"acc": round(100 * p, 1), "ci": round(100 * 1.96 * se, 1)}
        row["lift_naive"] = round(row["A_kg"]["acc"] - row["A0"]["acc"], 1)
        row["lift_agent"] = round(row["A_agent"]["acc"] - row["A0"]["acc"], 1)
        summary[model.name] = row
        print(f"[agentic] {model.name}: A0={row['A0']['acc']} A_kg={row['A_kg']['acc']} "
              f"A_agent={row['A_agent']['acc']} | lift_naive={row['lift_naive']} lift_agent={row['lift_agent']}")
    summary["_cypher_nonempty_rate"] = round(100 * hit_rate, 1)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"agentic_medqa_n{len(items)}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== agentic vs naive lift (A_kg=naive triple-RAG, A_agent=LLM-written Cypher) ===")
    for m in LADDER:
        print(f"  {m.name:14s} lift_naive={summary[m.name]['lift_naive']:+.1f}  lift_agent={summary[m.name]['lift_agent']:+.1f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--limit-edges", type=int, default=None)
    a = ap.parse_args()
    run(a.n, a.limit_edges)


if __name__ == "__main__":
    main()
