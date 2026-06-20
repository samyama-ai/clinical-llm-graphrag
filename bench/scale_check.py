"""Diagnostic: is the paper's HealthBench 88.0 explained by the dataset VARIANT/scale?

Compares gpt-5.2 and the dataset's ideal completions on the full oss_eval vs the Consensus
variant (Consensus = few, high-agreement, all-positive criteria → much higher scores).
Saves results/scale_check.json so the number is never lost. Run from repo root with PYTHONPATH=src.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from cllm.providers import Model, generate
from cllm.score_healthbench import RubricItem, judge_item

FILES = {
    "oss_eval": "2025-05-07-06-14-12_oss_eval.jsonl",
    "consensus": "consensus_2025-05-09-20-00-46.jsonl",
}
OUT = Path("results/scale_check.json")


def run(variant: str, n: int) -> dict:
    p = hf_hub_download("openai/healthbench", FILES[variant], repo_type="dataset")
    ds = [r for r in load_dataset("json", data_files=p, split="train")
          if isinstance(r["prompt"], list) and len(r["prompt"]) == 1][:n]
    base = Model("openai", "gpt-5.2")
    gpt, ideal = [], []
    for r in ds:
        conv = r["prompt"][0]["content"]
        rub = [RubricItem(x["criterion"], float(x["points"])) for x in r["rubrics"]]
        ic = (r.get("ideal_completions_data") or {}).get("ideal_completion")
        if ic:
            ideal.append(judge_item(f"{conv}\n\nassistant: {ic}", rub)["score"])
        ans = generate(base, conv, max_tokens=4000)
        gpt.append(judge_item(f"{conv}\n\nassistant: {ans}", rub)["score"])
    return {"variant": variant, "n": len(ds),
            "gpt52_mean": 100 * statistics.mean(gpt) if gpt else None,
            "ideal_mean": 100 * statistics.mean(ideal) if ideal else None,
            "mean_rubrics": statistics.mean(len(r["rubrics"]) for r in ds)}


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["consensus", "oss_eval"]
    res = [run(v, n) for v in variants]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    for r in res:
        print(f"{r['variant']}: gpt-5.2={r['gpt52_mean']:.1f}  ideal={r['ideal_mean']:.1f}  "
              f"rubrics/item={r['mean_rubrics']:.1f}  (n={r['n']})")
    print("paper reports HealthBench = 88.0")


if __name__ == "__main__":
    main()
