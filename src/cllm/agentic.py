"""Agentic graph retrieval: LLM writes Cypher over PrimeKG (samyama-graph), executes, and
self-corrects on error/empty. The real samyama-graph value prop (NLQ -> multi-hop -> deterministic
facts), vs naive triple-RAG. Used for the A_agent arm.

Entity-linking: samyama vector_search resolves question terms to EXACT PrimeKG node names, injected
into the Cypher-gen prompt (fixes name-mismatch). A fixed capable model (gpt-4.1) writes the Cypher
so the crossover isolates the *answering* model's benefit from grounding, not its Cypher skill.
"""
from __future__ import annotations

import re

from .providers import Model, generate

SCHEMA = (
    "Property graph of clinical entities in samyama-graph (OpenCypher).\n"
    "Nodes: (:Entity {name, etype}); etype in {drug, disease, effect/phenotype, gene/protein}.\n"
    "Relationships (all (:Entity)->(:Entity)):\n"
    "  (:Entity{etype:'drug'})-[:INDICATION]->(:Entity{etype:'disease'})   // drug treats disease\n"
    "  (:Entity{etype:'drug'})-[:CONTRAINDICATION]->(:Entity{etype:'disease'}) // unsafe in disease\n"
    "  [:OFF_LABEL_USE], [:DRUG_EFFECT] (drug->side effect), [:DRUG_DRUG] (interaction),\n"
    "  [:DISEASE_PHENOTYPE_POSITIVE] (disease->symptom), [:DISEASE_DISEASE], [:PHENOTYPE_PHENOTYPE]\n"
    "Match entities by EXACT 'name'."
)

_RULES = (
    "Rules:\n"
    "1. Entity names: use ONLY names copied VERBATIM from the Candidate list above — exact casing, "
    "exact spelling, including any parenthetical suffix like '(disease)'. NEVER invent, translate, "
    "abbreviate, or rephrase a name. If the ideal entity isn't listed, pick the closest candidate(s).\n"
    "2. Pick the 1-4 candidates most relevant to the question and build the query around them.\n"
    "3. Use SPECIFIC relationship types, ONE per MATCH (e.g. MATCH (a)-[:INDICATION]->(b)). "
    "Do NOT use type(r) or multi-type [r:A|B] patterns or UNION.\n"
    "4. RETURN readable named columns; end with LIMIT 50. Output ONLY the Cypher, no prose/fences."
)

_EXAMPLES = (
    "Direction matters — worked examples:\n"
    "  Drugs that treat disease X:   MATCH (drug:Entity)-[:INDICATION]->(:Entity {name:'X'}) RETURN drug.name LIMIT 50\n"
    "  Symptoms of disease X:        MATCH (:Entity {name:'X'})-[:DISEASE_PHENOTYPE_POSITIVE]->(p:Entity) RETURN p.name LIMIT 50\n"
    "  Is drug Y unsafe in disease X: MATCH (:Entity {name:'Y'})-[:CONTRAINDICATION]->(:Entity {name:'X'}) RETURN 'contraindicated' AS flag LIMIT 50\n"
    "  Side effects of drug Y:       MATCH (:Entity {name:'Y'})-[:DRUG_EFFECT]->(e:Entity) RETURN e.name LIMIT 50\n"
    "  To gather options, query several candidate drugs:  MATCH (drug:Entity)-[:INDICATION]->(d:Entity)\n"
    "    WHERE drug.name IN ['A','B','C'] RETURN drug.name, d.name LIMIT 50"
)

GEN = Model("openai", "gpt-4.1")  # fixed retrieval agent


def _clean(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:cypher|sql)?", "", t).strip().removesuffix("```").strip()
    m = re.search(r"\b(MATCH|CALL|OPTIONAL MATCH)\b", t, re.I)
    return t[m.start():].strip() if m else t


def agentic_retrieve(client, question: str, candidate_names, gen_model: Model = GEN,
                     max_retries: int = 2, max_chars: int = 1800) -> dict:
    """LLM writes Cypher (using vector-linked candidate names), execute, self-correct. Returns
    {facts, cypher, n_rows, attempts}."""
    hints = ", ".join(f"'{n}'" for n in candidate_names[:25])
    base = (f"{SCHEMA}\n\n{_EXAMPLES}\n\nCandidate entity names (use exact matches):\n{hints}\n\n"
            f"Question:\n{question}\n\n{_RULES}\n\nCypher:")
    prompt = base
    cypher = ""
    for attempt in range(max_retries + 1):
        try:
            cypher = _clean(generate(gen_model, prompt, max_tokens=600))
        except Exception as e:
            return {"facts": "", "cypher": cypher, "n_rows": 0, "attempts": attempt + 1, "error": str(e)[:120]}
        try:
            recs = client.query(cypher).records
            if recs:
                lines = []
                for row in recs[:50]:
                    lines.append(" | ".join(str(x) for x in row if x is not None))
                facts = "\n".join(l for l in lines if l)[:max_chars]
                if facts.strip():
                    return {"facts": facts, "cypher": cypher, "n_rows": len(recs), "attempts": attempt + 1}
                fb = "Query ran but all values were null. Pick concrete entities from the candidates."
            else:
                fb = "Query returned no rows. Use the EXACT candidate names and a broader relationship."
        except Exception as e:
            fb = f"Query errored: {str(e)[:150]}. Fix it; do NOT use type(r) or multi-type [r:A|B]."
        prompt = base + f"\n\nYour previous query:\n{cypher}\n\nProblem: {fb}\nWrite a corrected single Cypher query:"
    return {"facts": "", "cypher": cypher, "n_rows": 0, "attempts": max_retries + 1}
