"""Per-drug A0 outcomes for the realkg experiment, to resolve the clean-3 split in paper15 S6.

A0 ignores retrieval (no graph context), so this needs NO engine -- only the LLM. We reproduce the
EXACT seed-62 items (same distractor/shuffle sequence as realkg._build) and reuse realkg._score for A0,
so the aggregate must reproduce the stored realkg run (A0=84.6%) if the models are unchanged. Then we
read off how many of the three non-name-encoding ("clean") drugs A0 actually gets.

Run:  PYTHONPATH=src OPENAI_API_KEY=... python -m cllm.realkg_a0_perdrug
"""
import random
from .realkg import REAL_RECENT, DRUGS, LADDER, _score

# the three approvals whose generic names do NOT encode the indication (paper S6)
CLEAN3 = {"Hunter syndrome", "achondroplasia", "blastic plasmacytoid dendritic cell neoplasm"}


def build_items(seed=62):
    """Replicates realkg._build's item construction exactly (rng used only here, same order)."""
    rng = random.Random(seed)
    items = []
    for drug, dis in REAL_RECENT:
        distract = rng.sample([d for d in DRUGS if d != drug], 3)
        opts = distract + [drug]
        rng.shuffle(opts)
        gold = "ABCD"[opts.index(drug)]
        items.append({"id": dis[:20], "entity": dis, "drug": drug,
                      "question": f"A patient has {dis}. Which of the following medications is indicated to treat it?",
                      "options": dict(zip("ABCD", opts)), "gold": gold})
    return items


def main(seed=62):
    items = build_items(seed)
    hdr = "".join(f"{m.name:>14}" for m in LADDER)
    print(f"{'disease':46}{'drug':22}{hdr}")
    tot = {m.name: 0 for m in LADDER}
    clean = {m.name: 0 for m in LADDER}
    for it in items:
        cells = ""
        for m in LADDER:
            ok = _score(m, it, "A0", None)["correct"]
            tot[m.name] += ok
            if it["entity"] in CLEAN3:
                clean[m.name] += ok
            cells += f"{('Y' if ok else '.'):>14}"
        tag = "  <-- clean-3" if it["entity"] in CLEAN3 else ""
        print(f"{it['entity'][:45]:46}{it['drug'][:21]:22}{cells}{tag}")
    n = len(items)
    print("\n--- A0 summary (no graph) ---")
    for m in LADDER:
        print(f"{m.name:14} A0 = {tot[m.name]}/{n} = {100*tot[m.name]/n:.1f}%   |   clean-3 A0 = {clean[m.name]}/3")


if __name__ == "__main__":
    main()
