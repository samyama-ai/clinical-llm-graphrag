"""Positive-case experiment: agentic KG grounding on a SYNTHETIC counterfactual KG (out-of-training).
Expect A0 ~ chance (25%) and A_agent high — grounding is decisive when the model can't know the fact.
Contrast with the public-PrimeKG null (RESULTS-AGENTIC). A0 vs A_agent across the weak->strong ladder.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path

from . import synthkg
from .agentic import agentic_retrieve
from .providers import Model, generate
from .run_arms import embed_query
from .score_medqa import accuracy, extract_letter, extract_letter_llm

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
LADDER = [Model("openai", m) for m in ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-5.2"]]
WORKERS = 8
EMB = "text-embedding-3-small"

PROMPT = ("You are answering a multiple-choice medical question. Choose the single best option.\n\n"
          "{q}\n\n{opts}\n{ctx}Respond with the letter of the best option. End with 'Answer: <letter>'.")


def _vector_index(client):
    from openai import OpenAI
    import numpy as np
    nodes = client.query("MATCH (n:Entity) RETURN id(n), n.name").records
    names = [nm for _, nm in nodes]
    oc = OpenAI()
    embs = []
    for i in range(0, len(names), 1000):
        embs.extend(e.embedding for e in oc.embeddings.create(model=EMB, input=names[i:i+1000]).data)
    client.create_vector_index("Entity", "emb", len(embs[0]))
    nid2name = {}
    for (nid, nm), e in zip(nodes, embs):
        client.add_vector("Entity", "emb", nid, [float(x) for x in e])
        nid2name[nid] = nm
    print(f"[synth] vector index: {len(nid2name)} nodes")
    return nid2name


def _score(model, it, kind, facts):
    opts = "\n".join(f"({k}) {v}" for k, v in it["options"].items())
    ctx = "" if kind == "A0" else (f"[Knowledge-graph query results:\n{facts}\n]\n\n" if facts else "")
    prompt = PROMPT.format(q=it["question"], opts=opts, ctx=ctx)
    try:
        out = generate(model, prompt, max_tokens=2000)
    except Exception as e:
        out = f"__error__: {e}"
    pred = extract_letter(out, it["options"])
    if pred is None and out and not out.startswith("__error__"):
        pred = extract_letter_llm(out, it["options"])
    return {"correct": bool(pred) and pred == it["gold"].upper(), "parsed": pred is not None}


def run(n: int):
    from samyama import SamyamaClient
    synthkg.generate()
    client = SamyamaClient.embedded()
    synthkg.load_into(client)
    nid2name = _vector_index(client)
    items = [json.loads(l) for l in (synthkg.OUT / "qa.jsonl").read_text().splitlines()][:n]

    print(f"[synth] agentic retrieval for {len(items)} items...")
    facts = {}
    nonempty = 0
    for it in items:
        emb = embed_query(it["question"])
        hits = client.vector_search("Entity", "emb", [float(x) for x in emb], 25)
        cands = [nid2name[nid] for nid, _ in hits if nid in nid2name]
        if it["entity"] not in cands:
            cands = [it["entity"]] + cands  # ensure the named disease is offered exactly
        ag = agentic_retrieve(client, it["question"], cands)
        facts[it["id"]] = ag["facts"]
        nonempty += ag["n_rows"] > 0
    print(f"[synth] Cypher non-empty rate: {100*nonempty/len(items):.0f}%")

    summary = {"_cypher_nonempty_rate": round(100 * nonempty / len(items), 1)}
    for model in LADDER:
        row = {}
        for kind in ("A0", "A_agent"):
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                recs = list(ex.map(lambda it: _score(model, it, kind, facts[it["id"]]), items))
            agg = accuracy(recs); p = agg["accuracy"]; se = sqrt(p * (1 - p) / max(1, agg["n"]))
            row[kind] = {"acc": round(100 * p, 1), "ci": round(100 * 1.96 * se, 1)}
        row["lift_agent"] = round(row["A_agent"]["acc"] - row["A0"]["acc"], 1)
        summary[model.name] = row
        print(f"[synth] {model.name}: A0={row['A0']['acc']} A_agent={row['A_agent']['acc']} | lift_agent=+{row['lift_agent']}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"synth_medqa_n{len(items)}.json").write_text(json.dumps(summary, indent=2))
    print("\n=== POSITIVE CASE: out-of-training KG (expect A0~25%, big lift_agent) ===")
    for m in LADDER:
        print(f"  {m.name:14s} A0={summary[m.name]['A0']['acc']:>5} -> A_agent={summary[m.name]['A_agent']['acc']:>5}  (+{summary[m.name]['lift_agent']})")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    run(a.n)


if __name__ == "__main__":
    main()
