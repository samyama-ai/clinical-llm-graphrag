# clinical-llm-graphrag

**When does knowledge-graph grounding actually help an LLM?** A controlled clinical-QA study using
[samyama-graph](https://github.com/samyama-ai/samyama-graph) (graph + vector + OpenCypher) over the public
biomedical KG **PrimeKG** — plus a reproduction of a contested *Nature Medicine* 2026 result.

> **TL;DR — grounding helps only for knowledge the model can't already produce.** On public KGs (whose
> facts are in training) structured grounding gives **no lift**; on out-of-training knowledge it is
> **decisive** (chance → ~100%). And "out-of-training" ≠ "unknown": real recent drugs are largely
> inferable from drug-name conventions, so the boundary is **training ∪ inferable**.

Companion paper: *"Knowledge-Graph Grounding Helps LLMs Only for Out-of-Training Knowledge."*

## Findings

| Setting | Result | Reading |
|---|---|---|
| **Reproduction** | Nature's HealthBench **88** = the *Consensus* variant; full `oss_eval` ≈ **46** (ideal completions ≈47); MedQA ~90 | the headline number is a different (easier) scale; grader physician-calibrated at 82.5% |
| **Public KG** (PrimeKG, in-training) | naive **and** agentic NLQ→Cypher (82% query success): **no lift** at any model strength | public-KG facts are redundant with training |
| **Out-of-training** (synthetic KG) | A0 ~chance → **A_agent ~100%** (+75 to +79) | grounding decisive when the fact is novel |
| **Hybrid** (known + novel in one KG) | known lift **+0**, novel lift **+68 to +78**; no-LLM `A_det` = 100% both | the boundary, within one benchmark |
| **GAK** (Architecture D) | agent materializes facts from a source → provenance-tagged → cache → answer; **once per fact** vs per-query | the loop that *puts* novel facts in the graph |
| **Real recent drugs** (2026 FDA) | A0 **84.6%** (inferred from INN nomenclature); on the 3 uninferable facts A0 **0/3** → grounding **3/3** | boundary = training **∪ inferable** |

Maps to the four architectures of [the AssetOpsBench KG-data-layer paper](https://arxiv.org/abs/2605.26874):
A0 = A (LLM-only), A_agent = B (NLQ→Cypher), A_det = C (no-LLM deterministic), A_GAK = D (GAK).

## Reproduce

```bash
make env          # pinned deps into ~/projects/venv
make data         # MedQA + HealthBench seed=62 subsets (no API key)
make test         # 15 correctness/determinism tests (no API key)

# experiments (need OPENAI_API_KEY; deterministic temp=0, seed=62)
python -m cllm.run_crossover  --n 200          # public-KG null across the model ladder
python -m cllm.run_synth      --n 150          # out-of-training synthetic KG
python -m cllm.run_hybrid     --n-fict 80      # hybrid: known vs novel in one KG
python -m cllm.run_gak        --n 40           # GAK (Architecture D): enrich + cache + provenance
python -m cllm.realkg                          # real 2026 FDA approvals
```

Judge panel for HealthBench is **gpt-4.1 + Claude (via the `claude` CLI)** — no paid Gemini. PrimeKG
grounding loads into samyama-graph; subgraph expansion uses a precomputed adjacency (the engine's Cypher
`expand` lacks LIMIT-pushdown on high-degree nodes — one of three engine gaps documented in the paper).

## Layout
- `src/cllm/` — data prep, scorers (MedQA + HealthBench re-impl), providers, agentic NLQ→Cypher, the five experiment runners.
- `bench/` — figure + variance drivers.
- `results/` — committed result JSONs each runner regenerates.
- `tests/` — correctness fixtures (no service mocks).

## Honest scope
Single base-model family (OpenAI ladder gpt-4.1-nano→gpt-5.2); clinical multiple-choice is lookup-shaped,
so the engine's graph-algorithm/optimization primitives are not exercised; clinical tools from the source
study lack APIs and are cited, not re-run; HealthBench grading is LLM-as-judge (calibrated). See the paper's
limitations section. Data: MedQA, HealthBench (CC-BY-4.0), PrimeKG (MIT). Code: Apache-2.0.
