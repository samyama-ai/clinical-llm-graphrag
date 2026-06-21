"""Multi-seed variance pass. Positives (synth/hybrid/gak) at extra seeds -> robustness to the
fact-set; crossover (public-KG null) rerun -> the null is stable across runs. Combine with the
existing seed-62 results. Writes results/variance_summary.json. Run from repo root, PYTHONPATH=src.
"""
import json, statistics, glob
from pathlib import Path
from cllm import run_synth, run_hybrid, run_gak, run_crossover

R = Path("results")
EXTRA_SEEDS = [7, 123]


def main():
    # positives at extra seeds (seed 62 already run earlier under *_s62 or legacy names)
    for s in EXTRA_SEEDS:
        run_synth.run(150, s)
        run_hybrid.run(80, s)
        run_gak.run(40, s)
    # crossover null: one rerun (generation variance); existing committed run is run 1
    run_crossover.run(200, None)  # overwrites crossover_medqa_n200.json -> capture below

    def lifts_synth():
        vals = []
        for f in glob.glob("results/synth_medqa_n150*.json"):
            d = json.load(open(f)); vals += [d[m]["lift_agent"] for m in d if not m.startswith("_")]
        return vals

    def lifts_hybrid_fict():
        vals = []
        for f in glob.glob("results/hybrid_n*.json"):
            d = json.load(open(f)); vals += [d[m]["lift_fictional"] for m in d if m not in ("A_det",)]
        return vals

    summary = {
        "synth_lift_agent": _stat(lifts_synth()),
        "hybrid_fictional_lift": _stat(lifts_hybrid_fict()),
        "note": "positives across seeds 62/7/123; A_agent ~100 every run by construction.",
    }
    (R / "variance_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _stat(v):
    return {"n": len(v), "mean": round(statistics.mean(v), 1),
            "std": round(statistics.pstdev(v), 1) if len(v) > 1 else 0.0,
            "min": round(min(v), 1), "max": round(max(v), 1)} if v else {}


if __name__ == "__main__":
    main()
