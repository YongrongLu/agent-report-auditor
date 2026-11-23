"""Poster figures for v2.1 (main) with v2 as the confounded comparison.
Usage: python poster_figs_v21.py <v21_dir> <v2_dir> <fig_dir>"""
import json, sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

src, src_v2, dst = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(dst, exist_ok=True)
R = json.load(open(f"{src}/results.json")); df = pd.read_csv(f"{src}/samples.csv")
df2 = pd.read_csv(f"{src_v2}/samples.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 15, "axes.labelsize": 15, "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13})
COND = {"neutral": "#8a8a8a", "pressure": "#d97706", "forced": "#b91c1c"}
CLAIM = {"SUCCESS": "#d97706", "FAILURE": "#6b21a8", "UNKNOWN": "#1d4ed8"}
INK, BLUE = "#111111", "#1d4ed8"
PR = ["neutral", "pressure", "forced"]
DOM = [("unit_tests", "unit tests"), ("data_pipeline", "ETL pipeline"), ("deployment", "deployment")]

# ---- E1 behaviour ----
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
for di, (d, dl) in enumerate(DOM):
    for pi, pr in enumerate(PR):
        g = df[(df.visibility == "full") & (df.outcome == "failure") & (df.domain == d) & (df.pressure == pr)]
        v = (g.claim == "SUCCESS").mean(); x = di + (pi - 1) * 0.27
        ax.bar(x, v, 0.25, color=COND[pr], label=pr if di == 0 else None)
        ax.text(x, v + 0.02, f"{(g.claim == 'SUCCESS').sum()}/{len(g)}", ha="center", va="bottom", rotation=90, fontsize=11)
ax.set_xticks(range(3), [dl for _, dl in DOM]); ax.set_ylim(0, 1.25); ax.set_yticks([0, .25, .5, .75, 1])
ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_title("(a) Log shows FAILURE: reported SUCCESS", loc="left")
ax.legend(frameon=False, loc="center right")
ax = axes[1]
p = df[df.visibility == "partial"]; bottom = np.zeros(3)
for c in ["UNKNOWN", "FAILURE", "SUCCESS"]:
    vals = np.array([(p[p.pressure == pr].claim == c).mean() for pr in PR])
    ax.bar(range(3), vals, 0.62, bottom=bottom, color=CLAIM[c], label=c + (" (faithful)" if c == "UNKNOWN" else ""))
    for i, v in enumerate(vals):
        if v > 0.07: ax.text(i, bottom[i] + v / 2, f"{v:.0%}", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    bottom += vals
ax.set_xticks(range(3), PR); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_title("(b) Log cut off: what was reported", loc="left")
ax.legend(frameon=False, fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
fig.tight_layout(); fig.savefig(f"{dst}/E1.png", dpi=300); plt.close(fig)

# ---- E2 probing: within-domain saturates, cross-domain picks the layer ----
cv, lc = R["layer_curve"], R["lodo_curve"]; L = [c["layer"] for c in cv]; b = R["best_layer"]
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(L, [c["val_bacc"] for c in cv], color="#9db4f0", lw=2.4, label="within-domain probe")
ax.plot(L, [c["lodo_mean"] for c in lc], color=BLUE, lw=3, label="held-out domain (mean of 3)")
ax.plot(L, [c["val_bacc_shuffled"] for c in cv], color="#9a9a9a", lw=1.8, ls="--", label="shuffled labels")
ax.axhline(R["leakage"]["probe_bacc"], color="#b91c1c", lw=2, ls=":", label=f"leakage control, hidden outcome: {R['leakage']['probe_bacc']:.2f}")
ax.axvline(b, color=INK, lw=1); ax.text(b + 0.4, 0.06, f"layer {b}", fontsize=13)
ax.axhline(1 / 3, color=INK, lw=0.8); ax.text(0.6, 1 / 3 - 0.055, "chance", fontsize=12)
ax.set_ylim(0, 1.05); ax.set_xlabel("layer (0 = embeddings)"); ax.set_ylabel("balanced accuracy")
ax.legend(loc="upper left", frameon=False, fontsize=12, bbox_to_anchor=(0.03, 0.92))
fig.tight_layout(); fig.savefig(f"{dst}/E2.png", dpi=300); plt.close(fig)

# ---- E3: (a) the word confound, v2 vs v2.1; (b) what the probe reads on bad reports ----
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
m2 = (df2.visibility == "full") & (df2.outcome == "success"); m1 = (df.visibility == "full") & (df.outcome == "success")
x = np.arange(3)
v2 = [df2[m2 & (df2.pressure == pr)].p_verified_success.mean() for pr in PR]
v21 = [df[m1 & (df.pressure == pr)].p_verified_success.mean() for pr in PR]
ax.bar(x - 0.2, v2, 0.38, color="#c7c7c7", label="v2: prompt said “failed tasks”")
ax.bar(x + 0.2, v21, 0.38, color=BLUE, label="v2.1: outcome words removed")
for xi, (a, c) in enumerate(zip(v2, v21)):
    ax.text(xi - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=12); ax.text(xi + 0.2, c + 0.02, f"{c:.2f}", ha="center", fontsize=12)
ax.set_xticks(x, PR); ax.set_ylim(0, 1.15); ax.set_title("(a) Success logs: probe P(verified success)", loc="left", fontsize=14)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=1, fontsize=11)
ax = axes[1]
tab = R["gap"]["reading_on_bad_reports"]
groups = [("false_success", "SUCCESS on a\nfailure log"), ("unsupported_success", "SUCCESS on a\ncut-off log")]
names = [("reads_verified_failure", "probe reads: verified failure", "#6b21a8"),
         ("reads_insufficient", "probe reads: insufficient evidence", BLUE),
         ("reads_verified_success", "probe reads: verified success", "#d97706")]
left = np.zeros(2)
for key, lab, col in names:
    vals = np.array([sum(tab[f"{g}|{pr}"][key] for pr in PR if f"{g}|{pr}" in tab) for g, _ in groups], float)
    tot = np.array([sum(tab[f"{g}|{pr}"]["n"] for pr in PR if f"{g}|{pr}" in tab) for g, _ in groups], float)
    ax.barh(range(2), vals / tot, left=left, color=col, label=lab, height=0.55)
    for i, (v, t) in enumerate(zip(vals, tot)):
        if v > 0: ax.text(left[i] + v / t / 2, i, f"{int(v)}/{int(t)}", ha="center", va="center", color="white", fontsize=13, fontweight="bold")
    left += vals / tot
ax.set_yticks(range(2), [g[1] for g in groups]); ax.invert_yaxis(); ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.set_title("(b) Every bad report, all conditions", loc="left", fontsize=14)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=1, fontsize=11)
fig.tight_layout(); fig.savefig(f"{dst}/E3.png", dpi=300); plt.close(fig)

# ---- E4 steering at the chosen layer ----
st, pt = R["steering"], R.get("patching")
fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
panels = [("pressure_failure", "SUCCESS", "Failed task, pressure\nreported SUCCESS"),
          ("neutral_success", "FAILURE", "Succeeded task (side effect)\nreported FAILURE"),
          ("neutral_failure", "SUCCESS", "Failed task, steer the other way\nreported SUCCESS")]
for ax, (k, tgt, title) in zip(axes, panels):
    for d, col, sty in [("belief", BLUE, "-o"), ("random", "#9a9a9a", "--s")]:
        pts = [q for q in st[k][d] if q["unparsed"] < 0.5]
        ax.plot([abs(q["alpha"]) for q in pts], [q["rate"] for q in pts], sty, color=col, lw=2.4, ms=7,
                label="evidence direction" if d == "belief" else "random direction")
    ax.set_title(title, loc="left", fontsize=12.5); ax.set_xlabel("|α|"); ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
axes[0].legend(frameon=False, fontsize=11, loc="center right")
fig.tight_layout(); fig.savefig(f"{dst}/E4.png", dpi=300); plt.close(fig)
print("done")
