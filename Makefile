# clinical-llm-graphrag — one command regenerates every number.
# Uses ~/projects/venv (global rule). Override PY/VENV if needed.
VENV ?= $(HOME)/projects/venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
N    ?= 500
SEED ?= 62

.PHONY: env data test gateb arms stats figures repro clean

env:
	$(PIP) install -r requirements.txt

# No API key needed: build the frozen seed=62 subsets + hashes.
data:
	$(PY) -m cllm.data --n $(N) --seed $(SEED)

# No API key needed: correctness tests on real fixtures (Test-Plan Layer 2).
test:
	cd $(CURDIR) && PYTHONPATH=src $(PY) -m pytest -q tests

# THE HARD GATE (Test-Plan Layer 1). Needs OPENAI_API_KEY / ANTHROPIC_API_KEY.
gateb:
	PYTHONPATH=src $(PY) -m cllm.run_gateb --n $(N) --seed $(SEED)

arms:
	PYTHONPATH=src $(PY) -m cllm.run_arms --n $(N) --seed $(SEED)

stats:
	PYTHONPATH=src $(PY) -m cllm.stats

figures:
	PYTHONPATH=src $(PY) -m cllm.figures

repro: data gateb arms stats figures
	@echo "Full pipeline complete; see results/"

clean:
	rm -rf data/*.jsonl data/raw outputs results/*.raw.jsonl __pycache__
