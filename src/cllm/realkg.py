"""REAL out-of-training experiment: recent (2026) FDA novel-drug approvals -- genuine facts that
post-date most ladder models' training. Same pipeline as the synthetic/hybrid arms, but the facts are
REAL and verifiable (FDA Q1-2026 novel approvals; cross-checked against BioPatrika/FDA). Tests whether
agentic samyama-graph grounding rescues accuracy on real knowledge the model couldn't have, and whether
A0 is graded across the weak->strong ladder (newer models may know a few -> a real crossover signal).

Distractors are OTHER recent drugs, so the model must know the specific drug->indication mapping, not
just recognize the only real option.
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

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
LADDER = [Model("openai", m) for m in ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1", "gpt-5.2"]]
WORKERS = 6
EMB = "text-embedding-3-small"

# FDA novel approvals (2026), generic -> approved indication. Verified; navepegritide corrected to
# achondroplasia (a CNP analog) vs pegzilarginase for arginase-1 deficiency.
REAL_RECENT = [
    ("relacorilant", "platinum-resistant ovarian cancer"),
    ("tividenofusp alfa", "Hunter syndrome"),
    ("icotrokinra", "moderate-to-severe plaque psoriasis"),
    ("linerixibat", "cholestatic pruritus in primary biliary cholangitis"),
    ("navepegritide", "achondroplasia"),
    ("pegzilarginase", "arginase 1 deficiency"),
    ("milsaperidone", "schizophrenia"),
    ("difamilast", "atopic dermatitis"),
    ("copper histidinate", "Menkes disease"),
    ("pivekimab sunirine", "blastic plasmacytoid dendritic cell neoplasm"),
    ("ensitrelvir", "COVID-19 post-exposure prophylaxis"),
    ("tebipenem pivoxil", "complicated urinary tract infection"),
    ("baxdrostat", "hypertension"),
]
DRUGS = [d for d, _ in REAL_RECENT]


def _build(client, seed):
    rng = random.Random(seed)
    client.query("CREATE INDEX ON :Entity(name)")
    nodes, items = set(), []
    for drug, dis in REAL_RECENT:
        nodes.add((drug, "drug")); nodes.add((dis, "disease"))
    for drug, dis in REAL_RECENT:
        distract = rng.sample([d for d in DRUGS if d != drug], 3)
        opts = distract + [drug]; rng.shuffle(opts); gold = "ABCD"[opts.index(drug)]
        items.append({"id": dis[:20], "entity": dis, "drug": drug,
                      "question": f"A patient has {dis}. Which of the following medications is indicated to treat it?",
                      "options": dict(zip("ABCD", opts)), "gold": gold})
    nl = list(nodes)
    pat = ",".join("(:Entity {name:'%s',etype:'%s'})" % (n.replace("'", " "), t) for n, t in nl)
    client.query("CREATE " + pat)
    for drug, dis in REAL_RECENT:
        client.query("MATCH (a:Entity {name:'%s'}),(b:Entity {name:'%s'}) CREATE (a)-[:INDICATION]->(b)"
                     % (drug.replace("'", " "), dis.replace("'", " ")))
    return items, nl


def _vec(client):
    from openai import OpenAI
    rows = client.query("MATCH (n:Entity) RETURN id(n), n.name").records
    names = [nm for _, nm in rows]
    embs = [e.embedding for e in OpenAI().embeddings.create(model=EMB, input=names).data]
    client.create_vector_index("Entity", "emb", len(embs[0]))
    nid2name = {}
    for (nid, nm), e in zip(rows, embs):
        client.add_vector("Entity", "emb", nid, [float(x) for x in e]); nid2name[nid] = nm
    return nid2name


def _det(client, it):
    recs = client.query("MATCH (d:Entity)-[:INDICATION]->(:Entity {name:'%s'}) RETURN d.name"
                        % it["entity"].replace("'", " ")).records
    found = {r[0] for r in recs}
    return next((L for L, o in it["options"].items() if o in found), None)


def _score(model, it, kind, facts):
    if kind == "A_det":
        return {"correct": facts == it["gold"], "parsed": facts is not None}
    opts = "\n".join(f"({k}) {v}" for k, v in it["options"].items())
    ctx = "" if kind == "A0" else (f"[Knowledge-graph query results:\n{facts}\n]\n\n" if facts else "")
    out = generate(model, f"Answer the question. {it['question']}\n\n{opts}\n{ctx}End with 'Answer: <letter>'.", max_tokens=2000)
    pred = extract_letter(out, it["options"]) or extract_letter_llm(out, it["options"])
    return {"correct": pred == it["gold"], "parsed": pred is not None}


def _agg(recs):
    a = accuracy(recs); p = a["accuracy"]; se = sqrt(p*(1-p)/max(1, a["n"]))
    return {"acc": round(100*p, 1), "ci": round(100*1.96*se, 1), "n": a["n"]}


def run(seed=62):
    from samyama import SamyamaClient
    client = SamyamaClient.embedded()
    items, nodes = _build(client, seed)
    nid2name = _vec(client)
    afacts, dets = {}, {}
    for it in items:
        emb = embed_query(it["question"])
        hits = client.vector_search("Entity", "emb", [float(x) for x in emb], 12)
        cands = [nid2name[nid] for nid, _ in hits if nid in nid2name]
        if it["entity"] not in cands: cands = [it["entity"]] + cands
        afacts[it["id"]] = agentic_retrieve(client, it["question"], cands)["facts"]
        dets[it["id"]] = _det(client, it)
    summary = {"n": len(items), "A_det": _agg([_score(None, it, "A_det", dets[it["id"]]) for it in items])}
    for model in LADDER:
        row = {}
        for kind in ("A0", "A_agent"):
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                recs = list(ex.map(lambda it: _score(model, it, kind, afacts[it["id"]]), items))
            row[kind] = _agg(recs)
        row["lift_agent"] = round(row["A_agent"]["acc"] - row["A0"]["acc"], 1)
        summary[model.name] = row
        print(f"[realkg] {model.name}: A0={row['A0']['acc']}±{row['A0']['ci']} -> A_agent={row['A_agent']['acc']} (lift +{row['lift_agent']})")
    print(f"[realkg] A_det (no LLM) = {summary['A_det']['acc']}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"realkg_n{len(items)}_s{seed}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=62)
    a = ap.parse_args(); run(a.seed)


if __name__ == "__main__":
    main()
