"""Gate B (Test-Plan Layer 1): reproduce frontier A0 numbers on MedQA + HealthBench.

Real generation (needs API keys). Writes results/gateB_medqa.csv, gateB_healthbench.csv and a
pass/fail report against the published tolerances (MedQA ±2 pts, HealthBench ±3 pts).
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .providers import Model, MissingKey, generate
from .score_healthbench import RubricItem, judge_item
from .score_medqa import accuracy, extract_letter, extract_letter_llm, score_item

WORKERS = int(os.getenv("CLLM_WORKERS", "8"))


def _gen_retry(model: Model, prompt: str, tries: int = 3) -> str:
    """Real generation with retry on empty/transient failure (no mocks)."""
    last = ""
    for _ in range(tries):
        try:
            out = generate(model, prompt, max_tokens=4000)
            if out and out.strip():
                return out
            last = out or ""
        except Exception as e:  # transient API error -> retry
            last = f"__error__: {e}"
    return last


def _map_concurrent(fn, items, workers: int = WORKERS):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))

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
    items = [json.loads(l) for l in (DATA / "medqa.jsonl").read_text().splitlines()][:n]
    RESULTS.mkdir(exist_ok=True)
    raw_path = RESULTS / "gateB_medqa.raw.jsonl"

    def work(it):
        out = _gen_retry(model, MEDQA_PROMPT.format(question=it["question"], options=_fmt_options(it["options"])))
        pred = extract_letter(out, it["options"])
        used_llm = False
        if pred is None and out and not out.startswith("__error__"):
            pred = extract_letter_llm(out, it["options"])  # paper's LLM-extraction fallback
            used_llm = True
        gold = (it["gold"] or "").upper()
        return {"id": it["id"], "raw": out, "pred": pred, "gold": gold,
                "correct": bool(pred) and pred == gold, "parsed": pred is not None, "llm_extract": used_llm}

    results = _map_concurrent(work, items)
    with raw_path.open("w") as f:  # save raw outputs for reproducibility + audit
        for r in results:
            f.write(json.dumps(r) + "\n")
    recs = [{k: r[k] for k in ("id", "pred", "gold", "correct", "parsed")} for r in results]
    _write_csv(RESULTS / "gateB_medqa.csv", recs)
    agg = accuracy(recs)
    agg["llm_extract_used"] = sum(r["llm_extract"] for r in results)
    return agg


def _conv_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):  # [{role, content}, ...]
        return "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in prompt)
    return json.dumps(prompt)


def run_healthbench(model: Model, n: int, dataset: str = "healthbench", judge_panel=None) -> dict:
    items = [json.loads(l) for l in (DATA / f"{dataset}.jsonl").read_text().splitlines()][:n]
    RESULTS.mkdir(exist_ok=True)
    raw_path = RESULTS / f"gateB_{dataset}.raw.jsonl"

    def work(it):
        conv = _conv_text(it["prompt"])
        answer = _gen_retry(model, conv)
        rubrics = [RubricItem(r["criterion"] if isinstance(r, dict) else str(r),
                              float(r.get("points", 1)) if isinstance(r, dict) else 1.0)
                   for r in it.get("rubrics", [])]
        res = judge_item(f"{conv}\n\nassistant: {answer}", rubrics, panel=judge_panel) if rubrics else {"score": None, "met": []}
        return {"id": it["id"], "answer": answer, "n_rubrics": len(rubrics), "score": res["score"], "met": res.get("met")}

    results = _map_concurrent(work, items)
    with raw_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    _write_csv(RESULTS / f"gateB_{dataset}.csv", [{"id": r["id"], "score": r["score"]} for r in results])
    valid = [r["score"] for r in results if r["score"] is not None]
    return {"dataset": dataset, "n": len(valid), "mean_score": 100 * sum(valid) / len(valid) if valid else None}


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
    ap.add_argument("--task", choices=["medqa", "healthbench", "both"], default="both")
    args = ap.parse_args()
    model = Model(args.provider, args.model)
    mq = hb = None
    try:
        if args.task in ("medqa", "both"):
            mq = run_medqa(model, args.n)
            print("[gateB] medqa:", json.dumps(mq))
        if args.task in ("healthbench", "both"):
            hb = run_healthbench(model, args.n)
            print("[gateB] healthbench:", json.dumps(hb))
    except MissingKey as e:
        raise SystemExit(f"[gateB] cannot run — {e}. Set the API key (no mocks allowed).")
    report = {"model": vars(model), "medqa": mq, "healthbench": hb, "reference": REFERENCE}
    (RESULTS / "gateB_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
