"""fig3_cross_montage.py - Figure 3: the MD-dMRI vs DTI conductivity-model effect (dE_model) replicates
across all four montages. Per montage, the per-lobe cohort-median dE_model for the 4 cortical and 4 white-
matter lobes (95th-percentile |E|); white matter is consistently negative (MD-dMRI < DTI), cortex near zero.
Reads analysis/results/<subject>/roi_efield_<montage>.csv. Writes fig3_cross_montage.{png,pdf}.
"""
import os, csv, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "axes.linewidth": 0.9})

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "results")
MONTAGES = ["M1", "DLPFC", "HD_M1", "HD_DLPFC"]
LAB = {"M1": "M1 pad", "DLPFC": "DLPFC pad", "HD_M1": "M1 HD", "HD_DLPFC": "DLPFC HD"}
CTX = ["Ctx_Frontal", "Ctx_Parietal", "Ctx_Temporal", "Ctx_Occipital"]
WM = ["WM_Frontal", "WM_Parietal", "WM_Temporal", "WM_Occipital"]
ORANGE, BLUE, INK = "#d95f02", "#1f6fb2", "#1a1a1a"

def lobe_dE(M, lobe):
    vals = []
    for p in glob.glob(os.path.join(R, "PD*", f"roi_efield_{M}.csv")):
        row = next((r for r in csv.DictReader(open(p)) if r["ROI"] == lobe), None)
        if not row:
            continue
        try:
            dti, md = float(row["DTI_p95"]), float(row["MD-dMRI_p95"])
            if dti > 0:
                vals.append(100 * (md - dti) / dti)
        except (KeyError, ValueError):
            pass
    return float(np.median(vals)) if vals else np.nan

fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=300)
x = np.arange(len(MONTAGES))
for i, M in enumerate(MONTAGES):
    ctx = [lobe_dE(M, l) for l in CTX]; wm = [lobe_dE(M, l) for l in WM]
    ax.scatter([i - 0.12] * 4, ctx, color=ORANGE, s=42, edgecolors="white", linewidths=0.6, zorder=3,
               label="Cortex (GM)" if i == 0 else None)
    ax.scatter([i + 0.12] * 4, wm, color=BLUE, s=42, edgecolors="white", linewidths=0.6, zorder=3,
               label="White matter" if i == 0 else None)
    ax.plot([i - 0.22, i - 0.02], [np.median(ctx)] * 2, color=ORANGE, lw=2.4, zorder=2)
    ax.plot([i + 0.02, i + 0.22], [np.median(wm)] * 2, color=BLUE, lw=2.4, zorder=2)
ax.axhline(0, color="#888888", lw=0.9, ls="--", zorder=1)
ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in MONTAGES], fontsize=9.5)
ax.set_ylabel(r"$\Delta E_\mathrm{model}$  (MD-dMRI vs. DTI, %)", fontsize=10.5)
ax.set_title("Figure 3.   Conductivity-model effect replicates across montages   (n = 29, per-lobe medians)",
             fontsize=12, fontweight="bold", color=INK, pad=10)
ax.legend(frameon=False, fontsize=9.5, loc="lower right", handletextpad=0.3)
ax.tick_params(labelsize=9); ax.spines[["top", "right"]].set_visible(False)
ax.text(0.012, 0.04, "horizontal bars: median over the 4 lobes", transform=ax.transAxes, fontsize=7.8,
        color="#5f6368", style="italic", va="bottom")
fig.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.13)
fig.savefig(os.path.join(R, "fig3_cross_montage.png"), dpi=300, facecolor="white")
fig.savefig(os.path.join(R, "fig3_cross_montage.pdf"), facecolor="white")
print("wrote fig3_cross_montage.{png,pdf}")
for M in MONTAGES:
    print(f"  {M:9} WM median dE_model = {np.median([lobe_dE(M, l) for l in WM]):+.1f}%   "
          f"cortex = {np.median([lobe_dE(M, l) for l in CTX]):+.1f}%")
