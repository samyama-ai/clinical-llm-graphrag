"""Synthetic counterfactual clinical KG — fictional drugs/diseases the models CANNOT have seen in
training. Tests the positive case of the thesis: when the decisive fact is OUT of the model's
parametric knowledge, KG grounding (agentic NLQ->Cypher over samyama-graph) is decisive. A0 should
be ~chance (25%); A_agent should be high. Same Entity/relation schema as PrimeKG so agentic.py works.

Deterministic (seed=62): no Date/random-at-import; pure function of the seed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "synthkg"

_C = "bdfgklmnprstvz"
_V = "aeiou"


def _word(rng, syll):
    return "".join(rng.choice(_C) + rng.choice(_V) for _ in range(syll)).capitalize()


def generate(seed: int = 62, n_disease: int = 150, n_drug: int = 250, n_pheno: int = 120):
    rng = random.Random(seed)
    diseases = [_word(rng, 3) + rng.choice(["", " syndrome", " disease", "osis", "itis"]) for _ in range(n_disease)]
    drugs = [_word(rng, 3) + rng.choice(["ib", "ine", "ol", "mab", "pril", "azil"]) for _ in range(n_drug)]
    phenos = [_word(rng, 2) + rng.choice([" rash", " fever", " palsy", " edema", " ataxia"]) for _ in range(n_pheno)]
    # dedup
    diseases = list(dict.fromkeys(diseases)); drugs = list(dict.fromkeys(drugs)); phenos = list(dict.fromkeys(phenos))
    edges = []  # (rel, x_name, y_name)
    qa = []
    for d in diseases:
        correct = rng.choice(drugs)
        edges.append(("INDICATION", correct, d))                      # drug treats disease
        contra = rng.choice([x for x in drugs if x != correct])
        edges.append(("CONTRAINDICATION", contra, d))
        for p in rng.sample(phenos, 2):
            edges.append(("DISEASE_PHENOTYPE_POSITIVE", d, p))
        # MCQ: which drug treats this (fictional) disease?
        distract = rng.sample([x for x in drugs if x != correct], 3)
        opts = distract + [correct]; rng.shuffle(opts)
        gold = "ABCD"[opts.index(correct)]
        qa.append({"id": f"synth-{len(qa):04d}",
                   "question": f"A patient is diagnosed with {d}. Which medication is indicated to treat {d}?",
                   "options": {"A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3]},
                   "gold": gold, "entity": d})
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = ([{"name": d, "etype": "disease"} for d in diseases]
             + [{"name": d, "etype": "drug"} for d in drugs]
             + [{"name": p, "etype": "effect/phenotype"} for p in phenos])
    (OUT / "nodes.jsonl").write_text("\n".join(json.dumps(n) for n in nodes))
    (OUT / "edges.jsonl").write_text("\n".join(json.dumps({"rel": r, "x": x, "y": y}) for r, x, y in edges))
    (OUT / "qa.jsonl").write_text("\n".join(json.dumps(q) for q in qa))
    print(f"[synthkg] {len(nodes)} nodes, {len(edges)} edges, {len(qa)} questions -> {OUT}")
    return {"nodes": len(nodes), "edges": len(edges), "questions": len(qa)}


def _q(s):
    return s.replace("'", " ")


def load_into(client):
    """Load synthetic KG into samyama-graph (nodes + edges)."""
    nodes = [json.loads(l) for l in (OUT / "nodes.jsonl").read_text().splitlines()]
    edges = [json.loads(l) for l in (OUT / "edges.jsonl").read_text().splitlines()]
    client.query("CREATE INDEX ON :Entity(name)")
    for i in range(0, len(nodes), 500):
        pat = ",".join("(:Entity {name:'%s',etype:'%s'})" % (_q(n["name"]), _q(n["etype"])) for n in nodes[i:i+500])
        client.query("CREATE " + pat)
    for e in edges:
        client.query("MATCH (a:Entity {name:'%s'}),(b:Entity {name:'%s'}) CREATE (a)-[:%s]->(b)"
                     % (_q(e["x"]), _q(e["y"]), e["rel"]))
    print(f"[synthkg] loaded {len(nodes)} nodes, {len(edges)} edges")
    return {"nodes": len(nodes), "edges": len(edges)}


if __name__ == "__main__":
    generate()
