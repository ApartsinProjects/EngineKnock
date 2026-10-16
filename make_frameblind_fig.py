"""Per-fault frame-blindness figure: the same speed-conditioned normal-only detector run in three
representation frames (HZ physical-frequency, REV revolution-normalized, ORD order-tracked), per fault,
leakage-free LOSO, with cluster-bootstrap 95% CIs. Highlights the two chance-level blind spots
(ORD on horizontal misalignment, HZ on imbalance) and that REV avoids both."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("results/frameblind.json"))
faults = ["horizontal misalignment", "vertical misalignment", "imbalance", "underhang bearing", "overhang bearing"]
flab = ["horiz.\nmisalign.", "vert.\nmisalign.", "imbalance", "underhang\nbearing", "overhang\nbearing"]
frames = ["HZ", "REV", "ORD"]
fname = {"HZ": "physical-Hz", "REV": "revolution-norm.", "ORD": "order-tracked"}
cols = {"HZ": "#1f77b4", "REV": "#2ca02c", "ORD": "#d62728"}

fig, ax = plt.subplots(figsize=(9, 4.3))
x = np.arange(len(faults)); w = 0.26
for i, fr in enumerate(frames):
    y = [d["frames"][fr][f]["auc"] for f in faults]
    lo = [d["frames"][fr][f]["auc"] - d["frames"][fr][f]["ci95"][0] for f in faults]
    hi = [d["frames"][fr][f]["ci95"][1] - d["frames"][fr][f]["auc"] for f in faults]
    ax.bar(x + (i - 1) * w, y, w, yerr=[lo, hi], capsize=2.5, color=cols[fr], label=f"{fr} ({fname[fr]})",
           edgecolor="white", linewidth=0.5, error_kw={"lw": 0.8})
ax.axhline(0.5, ls="--", c="#555", lw=0.9); ax.text(len(faults) - 0.5, 0.515, "chance", fontsize=8, color="#555")
# annotate the two blind spots
ax.annotate("order tracking\nblind (0.53)", xy=(0 + w, 0.529), xytext=(0.55, 0.40), fontsize=8, color=cols["ORD"],
            ha="center", arrowprops=dict(arrowstyle="->", color=cols["ORD"], lw=0.9))
ax.annotate("physical-Hz\nblind (0.56)", xy=(2 - w, 0.564), xytext=(1.45, 0.40), fontsize=8, color=cols["HZ"],
            ha="center", arrowprops=dict(arrowstyle="->", color=cols["HZ"], lw=0.9))
ax.set_xticks(x); ax.set_xticklabels(flab, fontsize=9)
ax.set_ylabel("per-fault ROC-AUC (leakage-free LOSO)"); ax.set_ylim(0.3, 1.02)
ax.set_title("Each invariant representation is blind to a specific fault family;\nthe revolution-normalized frame avoids both blind spots")
ax.legend(fontsize=8, ncol=3, loc="upper center", frameon=False, bbox_to_anchor=(0.5, -0.12))
ax.grid(axis="y", alpha=0.25); plt.tight_layout()
plt.savefig("paper/figures/fig_frameblind.png", dpi=150)
plt.savefig("results/fig_frameblind.png", dpi=150)
print("wrote paper/figures/fig_frameblind.png")
