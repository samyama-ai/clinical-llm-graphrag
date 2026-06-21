"""Architecture D -- Generation-Augmented Knowledge (GAK), per paper3's grounding substrate.

The realistic out-of-training loop: facts live in unstructured SOURCE monographs (not in training, not
yet in the graph). On a lookup miss, an agent reads the source, generates provenance-tagged Cypher
CREATEs to materialize the facts, caches them, and answers via deterministic query. Subsequent questions
about the same entity are cache hits (no LLM). This closes paper15's loop: the synthetic experiment showed
"facts-in-graph -> decisive"; GAK is the mechanism that puts them there.

Arms:
  A0     : question only (no source)            -> ~chance (out-of-training)
  A_text : question + raw source in context     -> flat-RAG; correct but pays an LLM call EVERY query
  A_GAK  : enrich graph from source (once) then deterministic query; repeat queries are cache hits
Reports accuracy + LLM-call economics + provenance tagging.
"""
from __future__ import annotations

import argparse, json, random, re
from pathlib import Path

from .providers import Model, generate
from .score_medqa import extract_letter, extract_letter_llm
from . import synthkg

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ENRICH = Model("openai", "gpt-4.1")   # the enrichment agent (fixed)
ANSWER = Model("openai", "gpt-4o-mini")  # answering model for A0/A_text (a mid model)

ENRICH_PROMPT = (
    "Extract facts from a clinical monograph into knowledge-graph form. "
    "Return ONLY JSON: {{\"nodes\":[{{\"name\":..,\"etype\":..}}], \"edges\":[{{\"src\":..,\"rel\":..,\"dst\":..}}]}}. "
    "etype in {{drug, disease, effect/phenotype}}. rel in {{INDICATION (drug->disease it treats), "
    "DISEASE_PHENOTYPE_POSITIVE (disease->symptom)}}. Use exact names from the monograph.\n\n"
    "Monograph:\n{src}\n\nJSON:")


def _parse(t):
    m = re.search(r"\{.*\}", t.strip().removeprefix("```json").removeprefix("```"), re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def _materialize(client, facts):
    """LLM extracted the facts; the engine materializes them (nodes + edges-via-MATCH), every node
    provenance-tagged source:'enriched'. Uses the working CREATE/MATCH pattern (not inline var-edges)."""
    if not facts:
        return
    q = lambda s: str(s).replace("'", " ")
    for nd in facts.get("nodes", []):
        client.query("CREATE (:Entity {name:'%s', etype:'%s', source:'enriched'})"
                     % (q(nd.get("name")), q(nd.get("etype", "entity"))))
    for e in facts.get("edges", []):
        rel = q(e.get("rel", "RELATED")).upper().replace(" ", "_")
        client.query("MATCH (a:Entity {name:'%s'}),(b:Entity {name:'%s'}) CREATE (a)-[:%s]->(b)"
                     % (q(e.get("src")), q(e.get("dst")), rel))


def _gen_bench(seed, n):
    rng = random.Random(seed)
    dis = list(dict.fromkeys(synthkg._word(rng, 3) + rng.choice(["", " syndrome", "osis"]) for _ in range(n)))
    drugs = list(dict.fromkeys(synthkg._word(rng, 3) + rng.choice(["ib", "ine", "ol", "mab"]) for _ in range(n * 5)))
    phenos = list(dict.fromkeys(synthkg._word(rng, 2) + rng.choice([" rash", " fever", " palsy"]) for _ in range(n * 3)))
    items = []
    for d in dis:
        drug = rng.choice(drugs); ph = rng.sample(phenos, 2)
        src = (f"{d} is a rare condition. First-line pharmacologic treatment for {d} is {drug}. "
               f"Patients with {d} typically present with {ph[0]} and {ph[1]}.")
        distract = rng.sample([x for x in drugs if x != drug], 3)
        # two questions per disease: Q1 triggers enrichment, Q2 is a cache hit
        for k in range(2):
            opts = distract + [drug]; rng.shuffle(opts); gold = "ABCD"[opts.index(drug)]
            items.append({"id": f"{d}-{k}", "disease": d, "drug": drug, "src": src,
                          "q": f"Which medication is indicated to treat {d}?",
                          "options": dict(zip("ABCD", opts)), "gold": gold})
    return items


def _ans(model, q, options, ctx):
    opts = "\n".join(f"({k}) {v}" for k, v in options.items())
    c = f"[Source:\n{ctx}\n]\n\n" if ctx else ""
    out = generate(model, f"Answer the question. {q}\n\n{opts}\n{c}End with 'Answer: <letter>'.", max_tokens=600)
    p = extract_letter(out, options) or extract_letter_llm(out, options)
    return p


def run(n, seed=62):
    from samyama import SamyamaClient
    client = SamyamaClient.embedded()
    client.query("CREATE INDEX ON :Entity(name)")
    items = _gen_bench(seed, n)
    enriched, calls = set(), {"enrich": 0, "answer_text": 0, "cache_hits": 0}
    res = {a: {"correct": 0} for a in ("A0", "A_text", "A_GAK")}
    for it in items:
        # A0: no source
        res["A0"]["correct"] += (_ans(ANSWER, it["q"], it["options"], None) == it["gold"])
        # A_text: flat RAG (source in context) -- pays an LLM call per query
        calls["answer_text"] += 1
        res["A_text"]["correct"] += (_ans(ANSWER, it["q"], it["options"], it["src"]) == it["gold"])
        # A_GAK: enrich-on-miss then deterministic query (cache hits skip the LLM)
        if it["disease"] not in enriched:
            facts = _parse(generate(ENRICH, ENRICH_PROMPT.format(src=it["src"]), max_tokens=400)); calls["enrich"] += 1
            try:
                _materialize(client, facts)
            except Exception:
                pass
            enriched.add(it["disease"])
        else:
            calls["cache_hits"] += 1
        recs = client.query("MATCH (d:Entity)-[:INDICATION]->(:Entity {name:'%s'}) RETURN d.name"
                            % it["disease"].replace("'", " ")).records
        found = {r[0] for r in recs}
        pred = next((L for L, o in it["options"].items() if o in found), None)
        res["A_GAK"]["correct"] += (pred == it["gold"])
    N = len(items)
    prov = client.query("MATCH (n:Entity {source:'enriched'}) RETURN count(n)").records[0][0]
    summary = {"n_questions": N, "n_diseases": n,
               "A0_acc": round(100 * res["A0"]["correct"] / N, 1),
               "A_text_acc": round(100 * res["A_text"]["correct"] / N, 1),
               "A_GAK_acc": round(100 * res["A_GAK"]["correct"] / N, 1),
               "llm_calls_A_text": calls["answer_text"], "llm_calls_A_GAK_enrich": calls["enrich"],
               "cache_hits_A_GAK": calls["cache_hits"], "enriched_nodes_provenance_tagged": prov}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"gak_n{N}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nGAK pays {calls['enrich']} enrichment calls for {N} questions "
          f"({calls['cache_hits']} cache hits); A_text pays {calls['answer_text']}. "
          f"All {prov} materialized nodes tagged source:'enriched'.")
    return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(); run(a.n)


if __name__ == "__main__":
    main()
