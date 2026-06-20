"""A0 / A_flat / A_kg arms on a clinical benchmark, grounded in PrimeKG via samyama-graph.

  A0     = base model, no retrieval
  A_flat = base model + flat retrieved node names (vector hit list, NO graph structure)
  A_kg   = base model + retrieved SUBGRAPH (relations) — isolates the value of structure (Pillar-1)

All three share the same vector retrieval; A_flat vs A_kg differs only in whether graph structure
is included. Everything runs in ONE process (embedded samyama-graph is in-memory).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from . import primekg
from .providers import Model, generate
from .score_healthbench import RubricItem, judge_item

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
BASE = Model("openai", "gpt-5.2")
EMB_MODEL = "text-embedding-3-small"


def embed_query(text: str) -> list[float]:
    return OpenAI().embeddings.create(model=EMB_MODEL, input=[text]).data[0].embedding


def answer(conv: str, context: str | None, kind: str) -> str:
    if kind == "A0":
        return generate(BASE, conv, max_tokens=4000)
    label = "Relevant knowledge-graph facts" if kind == "A_kg" else "Relevant medical concepts"
    prompt = f"{conv}\n\n[{label} (use if relevant; ignore if not):\n{context}\n]"
    return generate(BASE, prompt, max_tokens=4000)


def setup_kg(limit_edges: int | None):
    from samyama import SamyamaClient
    client = SamyamaClient.embedded()
    primekg.load_into(client, limit_edges=limit_edges)
    pids, embs = primekg.embed_nodes(model=EMB_MODEL)
    info = primekg.build_vector_index(client, pids, embs)
    print(f"[arms] vector index: {info}")
    return client


def flat_context(client, q_emb, k=8) -> str:
    hits = client.vector_search("Entity", "emb", [float(x) for x in q_emb], k)
    names = []
    for nid, _ in hits:
        r = client.query("MATCH (n) WHERE id(n)=%d RETURN n.name, n.etype" % nid).records
        if r:
            names.append(f"- {r[0][0]} ({r[0][1]})")
    return "\n".join(names)


def conv_text(item) -> str:
    p = item["prompt"]
    return p if isinstance(p, str) else "\n".join(m["content"] for m in p)


def run(dataset: str, n: int, limit_edges: int | None, do_judge: bool):
    client = setup_kg(limit_edges)
    items = [json.loads(l) for l in (DATA / f"{dataset}.jsonl").read_text().splitlines()][:n]
    out = []
    for it in items:
        conv = conv_text(it)
        q_emb = embed_query(conv[:2000])
        ctx_kg = primekg.retrieve(client, q_emb)
        ctx_flat = flat_context(client, q_emb)
        rec = {"id": it["id"], "question": conv[:300], "kg_facts": ctx_kg.count("\n") + 1 if ctx_kg else 0,
               "ctx_kg": ctx_kg, "ctx_flat": ctx_flat}
        ans = {"A0": answer(conv, None, "A0"),
               "A_flat": answer(conv, ctx_flat, "A_flat"),
               "A_kg": answer(conv, ctx_kg, "A_kg")}
        rec["answers"] = ans
        if do_judge:
            rub = [RubricItem(x["criterion"], float(x.get("points", 1))) for x in it.get("rubrics", [])]
            for arm, a in ans.items():
                rec[f"score_{arm}"] = judge_item(f"{conv}\n\nassistant: {a}", rub)["score"] if rub else None
        out.append(rec)
        print(f"[arms] {it['id']}: kg_facts={rec['kg_facts']}"
              + (f" A0={rec.get('score_A0')} flat={rec.get('score_A_flat')} kg={rec.get('score_A_kg')}" if do_judge else ""))
    RESULTS.mkdir(exist_ok=True)
    tag = f"arms_{dataset}_n{len(items)}"
    (RESULTS / f"{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in out))
    if do_judge:
        import statistics
        for arm in ("A0", "A_flat", "A_kg"):
            sc = [r[f"score_{arm}"] for r in out if r.get(f"score_{arm}") is not None]
            if sc:
                print(f"[arms] {arm}: mean={100*statistics.mean(sc):.1f} (n={len(sc)})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="consensus")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--limit-edges", type=int, default=None)
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()
    run(a.dataset, a.n, a.limit_edges, a.judge)


if __name__ == "__main__":
    main()
