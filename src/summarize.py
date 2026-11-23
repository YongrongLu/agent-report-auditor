"""Recompute the headline numbers on the poster from saved outputs (CPU only, no model needed).

Usage (from the repository root):
    python src/summarize.py results/v2_1
"""
import json
import sys

import pandas as pd


def pct(x):
    return f"{100 * x:.0f}%"


def main(run_dir):
    df = pd.read_csv(f"{run_dir}/samples.csv")
    R = json.load(open(f"{run_dir}/results.json"))
    print(f"Model {R['model']} | {len(df)} reports | {df.instance.nunique()} task instances\n")
    print(df.category.value_counts().to_string(), "\n")

    print("False success (full failure log -> SUCCESS), by domain x condition:")
    ff = df[(df.visibility == "full") & (df.outcome == "failure")]
    print(pd.crosstab(ff.domain, ff.pressure, values=(ff.claim == "SUCCESS"), aggfunc="sum").astype(int).to_string(), "\n")

    print("Claim mix on truncated logs, by condition:")
    pl = df[df.visibility == "partial"]
    print(pd.crosstab(pl.pressure, pl.claim, normalize="index").round(3).to_string(), "\n")

    print("Paired behaviour tests:")
    for k, v in R["behaviour"]["paired"].items():
        print(f"  {k}: delta={v['delta']:+.3f} CI95={v['ci95']} McNemar p={v['mcnemar_p']:.2g}")

    print(f"\nProbe layer {R['best_layer']}: test balanced acc {R['test']['test_bacc']:.2f} "
          f"(permutation p={R['test']['perm_p']:.3f}); held-out-domain {R['lodo']}; "
          f"TF-IDF text baseline {R['text_baseline']:.2f}")
    print(f"Leakage control (hidden outcome from truncated logs): {R['leakage']['probe_bacc']:.2f} "
          f"(perm p={R['leakage']['perm_p']:.2f}, chance 0.50)")
    print("Probe reading on bad reports:")
    for k, v in R["gap"]["reading_on_bad_reports"].items():
        print(f"  {k}: n={v['n']} -> {v}")
    print("\nSteering (rate of the target behaviour; 'broken' = unparsable output):")
    for k, v in R["steering"].items():
        for kind, rows in v.items():
            s = ", ".join(f"a={r['alpha']}: {'broken' if r['unparsed'] >= 1 else pct(r['rate'])}" for r in rows)
            print(f"  {k:17s} {kind:7s} {s}")
    print("\nPatching:", {k: pct(v["rate"]) for k, v in R["patching"].items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/v2_1")
