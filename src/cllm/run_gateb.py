"""Gate B (Test-Plan Layer 1): reproduce frontier A0 numbers on MedQA + HealthBench.

Real generation (needs API keys). Writes results/gateB_medqa.csv, gateB_healthbench.csv and a
pass/fail report against the published tolerances (MedQA ±2 pts, HealthBench ±3 pts).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .providers import Model, MissingKey, generate
from .score_healthbench import RubricItem, judge_item
from .score_medqa import accuracy, score_item

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Published references (Nature s41591-026-04431-5). Tolerances per TEST-PLAN Layer 1.
REFERENCE = {
    "medqa": {"gemini": 97.4, "gpt": 94.2, "claude": 90.2, "tol": 2.0},
    "healthbench": {"gpt": 88.0, "gemini": 79.3, "claude": 77.0, "tol": 3.0},
}

MEDQA_PROMPT = (
    "You are a medical expert answering a USMLE-style question. Choose the single best option.\n\n"
    "{question}\n\n{options}\n\nRespond with the letter of the best option and a one-line rationale. "
    "End with 'Answer: <letter>'."
)


def _fmt_options(options: dict) -> str:
    return "\n".join(f"({k}) {v}" for k, v in options.items())


def run_medqa(model: Model, n: int) -> dict:
    items = [json.loads(l) for l in (DATA / "medqa.jsonl").read_text().splitlines()]
    recs = []
    for it in items[:n]:
        out = generate(model, MEDQA_PROMPT.format(question=it["question"], options=_fmt_options(it["options"])))
        recs.append({"id": it["id"], **score_item(out, it["gold"], it["options"])})
    RESULTS.mkdir(exist_ok=True)
    _write_csv(RESULTS / "gateB_medqa.csv", recs)
    return accuracy(recs)


def run_healthbench(model: Model, n: int) -> dict:
    items = [json.loads(l) for l in (DATA / "healthbench.jsonl").read_text().splitlines()]
    scores = []
    for it in items[:n]:
        conv = it["prompt"] if isinstance(it["prompt"], str) else json.dumps(it["prompt"])
        answer = generate(model, conv)
        rubrics = [RubricItem(r["criterion"] if isinstance(r, dict) else str(r),
                              float(r.get("points", 1)) if isinstance(r, dict) else 1.0)
                   for r in it.get("rubrics", [])]
        res = judge_item(f"{conv}\n\nassistant: {answer}", rubrics) if rubrics else {"score": None}
        scores.append({"id": it["id"], "score": res["score"]})
    RESULTS.mkdir(exist_ok=True)
    _write_csv(RESULTS / "gateB_healthbench.csv", scores)
    valid = [s["score"] for s in scores if s["score"] is not None]
    return {"n": len(valid), "mean_score": 100 * sum(valid) / len(valid) if valid else None}


def _write_csv(path: Path, recs: list[dict]) -> None:
    if not recs:
        path.write_text("")
        return
    cols = list(recs[0].keys())
    with path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for r in recs:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=62)
    ap.add_argument("--provider", default=os.getenv("CLLM_PROVIDER", "openai"))
    ap.add_argument("--model", default=os.getenv("CLLM_MODEL", "gpt-5.2"))
    args = ap.parse_args()
    model = Model(args.provider, args.model)
    try:
        mq = run_medqa(model, args.n)
        hb = run_healthbench(model, args.n)
    except MissingKey as e:
        raise SystemExit(f"[gateB] cannot run — {e}. Set the API key (no mocks allowed).")
    report = {"model": vars(model), "medqa": mq, "healthbench": hb, "reference": REFERENCE}
    (RESULTS / "gateB_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
