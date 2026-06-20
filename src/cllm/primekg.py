"""PrimeKG → clinical subset → samyama-graph loader + hybrid retriever for the A_kg arm.

PrimeKG (Harvard Dataverse datafile 6180620, kg.csv, 8.1M edges) is mostly molecular
(anatomy_protein 3M, protein_protein 642K) — irrelevant to clinical Q&A. We keep the
clinically-relevant relations only (~0.6M edges) so the grounding subgraph is precise
(Pillar-1: precision is what makes grounding help).
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PRIMEKG_CSV = ROOT / "data" / "primekg" / "kg.csv"
SUBSET_DIR = ROOT / "data" / "primekg"

# Clinically-relevant relations for grounding open-ended clinical questions + MedQA.
CLINICAL_RELATIONS = {
    "indication", "contraindication", "off-label use",
    "drug_effect", "drug_drug",
    "disease_phenotype_positive", "disease_phenotype_negative",
    "disease_disease", "phenotype_phenotype", "phenotype_protein",
    "disease_protein", "drug_protein",
}


def build_subset(max_drug_drug: int = 150_000) -> dict:
    """Filter kg.csv to clinical relations; cap drug_drug (2.67M) to keep it lean.
    Writes nodes.jsonl (id,type,name) and edges.csv (relation,x_id,y_id)."""
    nodes: dict[str, tuple[str, str]] = {}
    rel_counts: Counter = Counter()
    dd = 0
    edges_path = SUBSET_DIR / "edges.csv"
    with PRIMEKG_CSV.open() as f, edges_path.open("w", newline="") as out:
        r = csv.DictReader(f)
        w = csv.writer(out)
        w.writerow(["relation", "x_id", "y_id"])
        for row in r:
            rel = row["relation"]
            if rel not in CLINICAL_RELATIONS:
                continue
            if rel == "drug_drug":
                dd += 1
                if dd > max_drug_drug:
                    continue
            w.writerow([rel, row["x_id"], row["y_id"]])
            nodes[row["x_id"]] = (row["x_type"], row["x_name"])
            nodes[row["y_id"]] = (row["y_type"], row["y_name"])
            rel_counts[rel] += 1
    nodes_path = SUBSET_DIR / "nodes.jsonl"
    with nodes_path.open("w") as f:
        for nid, (ntype, name) in nodes.items():
            f.write(json.dumps({"id": nid, "type": ntype, "name": name}) + "\n")
    summary = {"edges": int(sum(rel_counts.values())), "nodes": len(nodes),
               "by_relation": dict(rel_counts.most_common()),
               "node_types": dict(Counter(t for t, _ in nodes.values()).most_common())}
    (SUBSET_DIR / "subset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _q(s: str) -> str:
    return s.replace("\\", " ").replace("'", " ").replace("\n", " ")


def load_into(client, node_batch: int = 500, log_every: int = 100_000, limit_edges: int | None = None) -> dict:
    """Load the clinical subset into an embedded samyama-graph client.
    Nodes via multi-pattern CREATE (~88K/s); edges via index-backed MATCH (~1.6K/s).
    limit_edges caps edges for smoke tests."""
    import time
    nodes = [json.loads(l) for l in (SUBSET_DIR / "nodes.jsonl").read_text().splitlines()]
    client.query("CREATE INDEX ON :Entity(pid)")
    t0 = time.time()
    for i in range(0, len(nodes), node_batch):
        chunk = nodes[i:i + node_batch]
        pat = ",".join("(:Entity {pid:'%s',name:'%s',etype:'%s'})"
                       % (_q(n["id"]), _q(n["name"]), _q(n["type"])) for n in chunk)
        client.query("CREATE " + pat)
    print(f"[primekg] {len(nodes)} nodes in {time.time()-t0:.1f}s")
    t0 = time.time(); n = 0
    with (SUBSET_DIR / "edges.csv").open() as f:
        r = csv.DictReader(f)
        for row in r:
            if limit_edges and n >= limit_edges:
                break
            client.query("MATCH (a:Entity {pid:'%s'}),(b:Entity {pid:'%s'}) CREATE (a)-[:%s]->(b)"
                         % (_q(row["x_id"]), _q(row["y_id"]), row["relation"].upper().replace("-", "_").replace(" ", "_")))
            n += 1
            if n % log_every == 0:
                print(f"[primekg] {n} edges ({n/(time.time()-t0):.0f}/s)")
    print(f"[primekg] {n} edges in {time.time()-t0:.1f}s")
    return {"nodes": len(nodes), "edges": n}


def embed_nodes(model: str = "text-embedding-3-small", batch: int = 1000) -> tuple[list[str], "object"]:
    """Embed node names (cached to data/primekg/node_emb.npz). Returns (pids, np.ndarray)."""
    import numpy as np
    from openai import OpenAI
    cache = SUBSET_DIR / "node_emb.npz"
    nodes = [json.loads(l) for l in (SUBSET_DIR / "nodes.jsonl").read_text().splitlines()]
    pids = [n["id"] for n in nodes]
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        if list(d["pids"]) == pids:
            print(f"[primekg] embeddings cache hit ({len(pids)})")
            return pids, d["emb"]
    client = OpenAI()
    texts = [f"{n['type']}: {n['name']}" for n in nodes]
    embs = []
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i:i + batch])
        embs.extend(e.embedding for e in resp.data)
        print(f"[primekg] embedded {min(i+batch,len(texts))}/{len(texts)}")
    arr = np.array(embs, dtype=np.float32)
    np.savez(cache, pids=np.array(pids), emb=arr)
    return pids, arr


def build_vector_index(client, pids, embs) -> dict:
    """Add node embeddings to the graph + build HNSW index. Maps pid->internal id via query."""
    from samyama import SamyamaClient  # noqa
    dim = len(embs[0])
    client.create_vector_index("Entity", "emb", dim)
    # internal id <- pid
    rows = client.query("MATCH (n:Entity) RETURN id(n), n.pid").records
    pid2nid = {pid: nid for nid, pid in rows}
    added = 0
    for pid, vec in zip(pids, embs):
        nid = pid2nid.get(pid)
        if nid is not None:
            client.add_vector("Entity", "emb", nid, [float(x) for x in vec])
            added += 1
    return {"indexed": added, "dim": dim}


# Clinically-meaningful relations for grounding (exclude noisy *_protein associations, which are
# molecular and unhelpful for clinical advice; they stay in the graph for mechanism queries).
_CLINICAL_RELS = {
    "INDICATION", "CONTRAINDICATION", "OFF_LABEL_USE", "DRUG_EFFECT", "DRUG_DRUG",
    "DISEASE_PHENOTYPE_POSITIVE", "DISEASE_PHENOTYPE_NEGATIVE", "DISEASE_DISEASE", "PHENOTYPE_PHENOTYPE",
}


def retrieve(client, q_emb, k: int = 8, max_neighbors: int = 30, keep=_CLINICAL_RELS, max_lines: int = 40) -> str:
    """Hybrid retrieval: ANN over node embeddings -> 1-hop subgraph (clinical relations only)
    -> text context. Filters out *_protein noise so grounding stays clinically precise (Pillar-1)."""
    hits = client.vector_search("Entity", "emb", [float(x) for x in q_emb], k)
    lines, seen = [], set()
    for nid, _ in hits:
        rows = client.query(
            "MATCH (a) WHERE id(a)=%d MATCH (a)-[r]-(b:Entity) "
            "RETURN a.name, type(r), b.name LIMIT %d" % (nid, max_neighbors)).records
        for a, rel, b in rows:
            if keep is not None and rel.upper() not in keep:
                continue
            key = (a, rel, b)
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {a} —[{rel.lower().replace('_', ' ')}]→ {b}")
    return "\n".join(lines[:max_lines])


if __name__ == "__main__":
    import sys
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000
    s = build_subset(cap)
    print(json.dumps(s, indent=2))
