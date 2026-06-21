"""HYBRID regime experiment (the missing third regime, per paper3's grounding substrate).

One benchmark, two strata in the SAME graph:
  - REAL stratum: canonical, textbook disease->drug facts the models KNOW (in-training).
  - FICTIONAL stratum: synthetic counterfactual facts the models CANNOT know (out-of-training).
Three arms: A0 (no knowledge / direct LLM), A_agent (GraphRAG: LLM writes Cypher), A_det (no-LLM
deterministic Cypher handler -- paper3 Arch C). Prediction (knowledge-boundary): lift ~0 on the REAL
stratum, large on the FICTIONAL stratum; A_det ~100% on both (it's the data layer, not the LLM).
"""
from __future__ import annotations

import argparse, json, random
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from pathlib import Path

from .agentic import agentic_retrieve
from .providers import Model, generate
from .run_arms import embed_query
from .score_medqa import accuracy, extract_letter, extract_letter_llm
from . import synthkg

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
LADDER = [Model("openai", m) for m in ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-5.2"]]
WORKERS = 8
EMB = "text-embedding-3-small"

# Canonical, uncontroversial textbook disease -> drug (in-training knowledge).
REAL_PAIRS = [
    ("type 1 diabetes mellitus", "Insulin"), ("hypothyroidism", "Levothyroxine"),
    ("Parkinson disease", "Levodopa"), ("acute gout", "Colchicine"),
    ("anaphylaxis", "Epinephrine"), ("opioid overdose", "Naloxone"),
    ("Wilson disease", "Penicillamine"), ("pernicious anemia", "Vitamin B12"),
    ("scurvy", "Vitamin C"), ("acetaminophen overdose", "N-acetylcysteine"),
    ("myasthenia gravis", "Pyridostigmine"), ("Addison disease", "Hydrocortisone"),
    ("central diabetes insipidus", "Desmopressin"), ("bipolar disorder", "Lithium"),
    ("attention deficit hyperactivity disorder", "Methylphenidate"),
    ("erectile dysfunction", "Sildenafil"), ("benzodiazepine overdose", "Flumazenil"),
    ("iron overdose", "Deferoxamine"), ("methanol poisoning", "Fomepizole"),
    ("malaria", "Artemisinin"), ("tuberculosis", "Isoniazid"),
    ("hemophilia A", "Factor VIII"), ("hyperkalemia", "Calcium gluconate"),
    ("warfarin-associated bleeding", "Vitamin K"),
]
REAL_DRUGS = [d for _, d in REAL_PAIRS]
SCHEMA_NOTE = None  # agentic.SCHEMA used


def _build_kg(client, seed, n_fict):
    rng = random.Random(seed)
    client.query("CREATE INDEX ON :Entity(name)")
    items = []
    # REAL stratum
    nodes = set()
    edges = []
    for dis, drug in REAL_PAIRS:
        nodes.add((dis, "disease")); nodes.add((drug, "drug"))
        edges.append((drug, "INDICATION", dis))
        distract = rng.sample([d for d in REAL_DRUGS if d != drug], 3)
        opts = distract + [drug]; rng.shuffle(opts); gold = "ABCD"[opts.index(drug)]
        items.append({"id": f"real-{len(items):03d}", "stratum": "real", "entity": dis,
                      "question": f"A patient is diagnosed with {dis}. Which medication is indicated to treat {dis}?",
                      "options": dict(zip("ABCD", opts)), "gold": gold})
    # FICTIONAL stratum (reuse synth generator's word maker)
    fdis = [synthkg._word(rng, 3) + rng.choice(["", " syndrome", "osis"]) for _ in range(n_fict)]
    fdrug = [synthkg._word(rng, 3) + rng.choice(["ib", "ine", "ol", "mab"]) for _ in range(n_fict * 4)]
    fdis = list(dict.fromkeys(fdis)); fdrug = list(dict.fromkeys(fdrug))
    for dis in fdis:
        drug = rng.choice(fdrug)
        nodes.add((dis, "disease")); nodes.add((drug, "drug"))
        edges.append((drug, "INDICATION", dis))
        distract = rng.sample([d for d in fdrug if d != drug], 3)
        opts = distract + [drug]; rng.shuffle(opts); gold = "ABCD"[opts.index(drug)]
        items.append({"id": f"fict-{len(items):03d}", "stratum": "fictional", "entity": dis,
                      "question": f"A patient is diagnosed with {dis}. Which medication is indicated to treat {dis}?",
                      "options": dict(zip("ABCD", opts)), "gold": gold})
    nodes = list(nodes)
    for i in range(0, len(nodes), 400):
        pat = ",".join("(:Entity {name:'%s',etype:'%s'})" % (n.replace("'"," "), t) for n, t in nodes[i:i+400])
        client.query("CREATE " + pat)
    for d, rel, dis in edges:
        client.query("MATCH (a:Entity {name:'%s'}),(b:Entity {name:'%s'}) CREATE (a)-[:%s]->(b)"
                     % (d.replace("'"," "), dis.replace("'"," "), rel))
    return items, nodes


def _vec_index(client):
    from openai import OpenAI
    rows = client.query("MATCH (n:Entity) RETURN id(n), n.name").records
    names = [nm for _, nm in rows]
    oc = OpenAI(); embs = []
    for i in range(0, len(names), 1000):
        embs.extend(e.embedding for e in oc.embeddings.create(model=EMB, input=names[i:i+1000]).data)
    client.create_vector_index("Entity", "emb", len(embs[0]))
    nid2name = {}
    for (nid, nm), e in zip(rows, embs):
        client.add_vector("Entity", "emb", nid, [float(x) for x in e]); nid2name[nid] = nm
    return nid2name


def _det_answer(client, it):
    """No-LLM deterministic handler (paper3 Arch C): query the indicated drug, match to an option."""
    dis = it["entity"].replace("'", " ")
    recs = client.query("MATCH (d:Entity)-[:INDICATION]->(:Entity {name:'%s'}) RETURN d.name" % dis).records
    found = {r[0] for r in recs}
    for letter, opt in it["options"].items():
        if opt in found:
            return letter
    return None


def _score(model, it, kind, facts):
    if kind == "A_det":
        pred = facts  # precomputed letter
        return {"correct": pred == it["gold"], "parsed": pred is not None}
    opts = "\n".join(f"({k}) {v}" for k, v in it["options"].items())
    ctx = "" if kind == "A0" else (f"[Knowledge-graph query results:\n{facts}\n]\n\n" if facts else "")
    prompt = (f"Answer the multiple-choice medical question. Choose the single best option.\n\n"
              f"{it['question']}\n\n{opts}\n{ctx}End with 'Answer: <letter>'.")
    try:
        out = generate(model, prompt, max_tokens=2000)
    except Exception as e:
        out = f"__error__: {e}"
    pred = extract_letter(out, it["options"])
    if pred is None and out and not out.startswith("__error__"):
        pred = extract_letter_llm(out, it["options"])
    return {"correct": bool(pred) and pred == it["gold"], "parsed": pred is not None}


def _agg(recs):
    a = accuracy(recs); p = a["accuracy"]; se = sqrt(p*(1-p)/max(1, a["n"]))
    return {"acc": round(100*p, 1), "ci": round(100*1.96*se, 1), "n": a["n"]}


def run(n_fict, seed=62):
    from samyama import SamyamaClient
    client = SamyamaClient.embedded()
    items, nodes = _build_kg(client, seed, n_fict)
    nid2name = _vec_index(client)
    real = [it for it in items if it["stratum"] == "real"]
    fict = [it for it in items if it["stratum"] == "fictional"]
    print(f"[hybrid] {len(real)} real + {len(fict)} fictional items; {len(nodes)} nodes")

    # precompute agentic facts + deterministic letters per item
    afacts, dets = {}, {}
    for it in items:
        emb = embed_query(it["question"])
        hits = client.vector_search("Entity", "emb", [float(x) for x in emb], 25)
        cands = [nid2name[nid] for nid, _ in hits if nid in nid2name]
        if it["entity"] not in cands: cands = [it["entity"]] + cands
        afacts[it["id"]] = agentic_retrieve(client, it["question"], cands)["facts"]
        dets[it["id"]] = _det_answer(client, it)
    det_real = _agg([_score(None, it, "A_det", dets[it["id"]]) for it in real])
    det_fict = _agg([_score(None, it, "A_det", dets[it["id"]]) for it in fict])

    summary = {"A_det": {"real": det_real, "fictional": det_fict}}
    for model in LADDER:
        row = {}
        for stratum, group in (("real", real), ("fictional", fict)):
            for kind in ("A0", "A_agent"):
                with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                    recs = list(ex.map(lambda it: _score(model, it, kind, afacts[it["id"]]), group))
                row[f"{kind}_{stratum}"] = _agg(recs)
            row[f"lift_{stratum}"] = round(row[f"A_agent_{stratum}"]["acc"] - row[f"A0_{stratum}"]["acc"], 1)
        summary[model.name] = row
        print(f"[hybrid] {model.name}: REAL A0={row['A0_real']['acc']}->A_agent={row['A_agent_real']['acc']} (lift {row['lift_real']:+}) | "
              f"FICT A0={row['A0_fictional']['acc']}->A_agent={row['A_agent_fictional']['acc']} (lift {row['lift_fictional']:+})")
    print(f"[hybrid] A_det (no LLM): real={det_real['acc']} fictional={det_fict['acc']}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"hybrid_n{len(items)}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-fict", type=int, default=80)
    a = ap.parse_args(); run(a.n_fict)


if __name__ == "__main__":
    main()
