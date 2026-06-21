"""Agentic A_kg smoke: LLM writes Cypher over PrimeKG (samyama-graph) and we execute it —
the real samyama-graph value prop (NLQ->graph), vs naive triple-RAG. Tests whether generated
Cypher returns RELEVANT facts for real MedQA questions (the entity-linking + graph-answerability
question). Run from repo root with PYTHONPATH=src after sourcing OpenAI env.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from samyama import SamyamaClient

from cllm import primekg

ROOT = Path(__file__).resolve().parents[1]

SCHEMA = (
    "Graph of clinical entities. Nodes: (:Entity {name, etype}); etype in "
    "{drug, disease, effect/phenotype, gene/protein}. Relationships (all (:Entity)->(:Entity)): "
    "[:INDICATION] drug->disease it treats; [:CONTRAINDICATION] drug->disease where unsafe; "
    "[:OFF_LABEL_USE]; [:DRUG_EFFECT] drug->side-effect/phenotype; [:DRUG_DRUG] interaction; "
    "[:DISEASE_PHENOTYPE_POSITIVE] disease->symptom; [:DISEASE_DISEASE]; [:PHENOTYPE_PHENOTYPE]; "
    "[:DISEASE_PROTEIN]; [:DRUG_PROTEIN]. Match entities by exact 'name' property."
)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    limit_edges = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
    client = SamyamaClient.embedded()
    primekg.load_into(client, limit_edges=limit_edges)
    items = [json.loads(l) for l in (ROOT / "data" / "medqa.jsonl").read_text().splitlines()][:n]
    for it in items:
        opts = " ".join(f"({k}){v}" for k, v in it["options"].items())
        q = f"{it['question']}\nOptions: {opts}"
        print("=" * 80)
        print("Q:", it["question"][:180])
        try:
            cy = client.nlq_to_cypher(
                f"Extract the key clinical entities and relationships needed to answer: {q}", SCHEMA)
            print("CYPHER:", cy[:300] if isinstance(cy, str) else cy)
            recs = client.query(cy).records if isinstance(cy, str) else []
            print(f"RESULTS ({len(recs)}):", recs[:8])
        except Exception as e:
            print("ERR:", str(e)[:200])


if __name__ == "__main__":
    main()
