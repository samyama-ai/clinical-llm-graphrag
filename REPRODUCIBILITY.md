# Reproducibility

## Environment
- Python 3.12, deps pinned in `requirements.txt` (install into `~/projects/venv`: `make env`).
- Key libs: datasets 4.8.2, openai 2.26.0, anthropic 0.86.0.

## Data provenance (public, reproducible)
- **MedQA** — HF `GBaker/MedQA-USMLE-4-options`, split `test` (1,273 items). Subset = 500, `seed=62`.
- **HealthBench** — HF `openai/healthbench`, file `2025-05-07-06-14-12_oss_eval.jsonl` (5,000). Subset = 500, `seed=62`. CC-BY-4.0.
- **HealthBench Hard** — HF `openai/healthbench`, file `hard_2025-05-08-21-00-10.jsonl` (1,000). Subset = 500, `seed=62`.
- Subsets are a pure function of (content, seed, n): rows sorted by content hash, then `random.Random(62)` shuffle. Byte-stability verified by `data/manifest.json` `ids_sha256`:
  - medqa `23db6e71…`, healthbench `794fc407…`, healthbench_hard `9cf5a562…` (n=500, seed=62, this machine).

## Seeds / determinism
- Subset selection: `seed=62`. Generation: `temperature=0.0`, `seed=62` (OpenAI); Anthropic temp 0 (no seed param).
- `make test` (13 tests) verifies extraction, score aggregation, and seed determinism with **no API calls** (real fixtures, no service mocks).

## Hardware
- Dev: macOS (Apple Silicon). Data prep + tests are CPU-only, < 1 min. Generation/judging cost depends on the model/provider (logged per run).

## Gate B (the hard gate) — not yet run
Needs `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY`. `make gateb` reproduces A0 frontier MedQA/HealthBench numbers and checks them against the published references (MedQA ±2 pts, HealthBench ±3 pts). Exact model IDs + access dates are written to `results/gateB_report.json` for version provenance.

## Regenerate everything
`make repro` (= data → gateb → arms → stats → figures). Every number in the paper must come from this.
