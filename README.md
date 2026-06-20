# clinical-llm-graphrag

**Does structured knowledge-graph grounding change the verdict that frontier LLMs beat specialized clinical AI tools?**

A reproducible harness comparing, on the **same public clinical benchmarks**, a frontier LLM:
- **A0** — alone (no retrieval),
- **A_flat** — with flat text-RAG,
- **A_kg** — with [samyama-graph](https://github.com/samyama-ai) knowledge-graph grounding (graph + vector + provenance).

Motivated by Vishwanath, Oermann et al., *"General-purpose large language models outperform specialized clinical AI tools on medical benchmarks,"* **Nature Medicine** 2026 (DOI 10.1038/s41591-026-04431-5), which found frontier LLMs beat OpenEvidence/UpToDate and that flat document-RAG *hurt* strong models. We test whether **retrieval precision** (not retrieval per se) is the deciding variable, and whether structured graph retrieval clears the precision floor strong models demand.

> **Honest claim (current stage):** this is a **reproducible baseline + pre-registered study**, not yet a validated result. Status below.

## Status

| Stage | State |
|---|---|
| Pre-registration (hypotheses + decision rules frozen) | ✅ `dbms_cloud/daily/clinical-llm-graphrag/HYPOTHESIS.md` |
| Test plan (5 layers) | ✅ `…/TEST-PLAN.md` |
| Data prep (MedQA + HealthBench, seed=62 subsets) | see `make data` |
| **Gate B — reproduce frontier A0 numbers** | ⏳ needs `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` |
| Headline (H1) + ablations + neg-controls | ⛔ gated on Gate B |

## Quickstart

```bash
make env        # install pinned deps into ~/projects/venv
make data       # download MedQA + HealthBench, build seed=62 subsets + hashes (no API key)
make test       # correctness tests on real fixtures (no API key)
make gateb      # reproduce A0 frontier numbers (needs API key) — the hard gate
make repro      # full pipeline: data -> gateb -> arms -> stats -> figures
```

## What's reproducible vs not
- **Public + reproducible:** MedQA (`GBaker/MedQA-USMLE-4-options`), HealthBench (`openai/healthbench`, CC-BY-4.0), PrimeKG (MIT). seed=62 subsets are byte-stable (hashes in `data/manifest.json`).
- **External reference only (no API):** OpenEvidence, UpToDate Expert AI — quoted from the Nature paper, not re-run.
- **License:** Apache-2.0 (code). HealthBench scoring is re-implemented from the public spec (we do **not** vendor the AGPL-3.0 reference harness `nyuolab/clinical-llm-benchmarks`).

Trail, hypotheses, and novelty analysis live in `dbms_cloud/daily/clinical-llm-graphrag/`.
