"""fig_prelim_3models.py - preliminary 'first results' display: tDCS E-field across the three conductivity
models (ISO, DTI, MD-dMRI), M1 montage, n=29, from the 95th-percentile |E|. Produces a grouped-bar figure
(8 lobes) and a tier-grouped two-panel table (Tier 1+2). Foundational; precedes the H1 comparison (Table 2).
"""
import os, csv, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "axes.linewidth": 0.9})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "analysis", "results")
data = {}
for p in sorted(glob.glob(os.path.join(R, "PD*", "roi_efield_M1.csv"))):
    for row in csv.DictReader(open(p)):
        d = data.setdefault(row["ROI"], {"ISO": [], "DTI": [], "MD-dMRI": []})
        for m, col in [("ISO", "ISO_p95"), ("DTI", "DTI_p95"), ("MD-dMRI", "MD-dMRI_p95")]:
            try: d[m].append(float(row[col]))
            except (KeyError, ValueError): pass
dmod = {r["ROI"]: r["dE_model_median_pct"] for r in csv.DictReader(open(os.path.join(R, "m1_results_summary.csv")))}
def med(roi, m): return float(np.median(data[roi][m])) if data.get(roi, {}).get(m) else float("nan")
def msd(roi, m): a = np.array(data[roi][m]); return a.mean(), a.std()

ISO_C, DTI_C, MD_C, INK, SUB = "#9aa0a6", "#1f6fb2", "#d95f02", "#1a1a1a", "#5f6368"

# ---- bar figure: 4 cortical + 4 WM lobes ----
LOBES = [("Ctx_Frontal", "Frontal"), ("Ctx_Parietal", "Parietal"), ("Ctx_Temporal", "Temporal"), ("Ctx_Occipital", "Occipital"),
         ("WM_Frontal", "Frontal"), ("WM_Parietal", "Parietal"), ("WM_Temporal", "Temporal"), ("WM_Occipital", "Occipital")]
fig, ax = plt.subplots(figsize=(9.2, 4.7), dpi=300)
x = np.arange(len(LOBES)); w = 0.26
for i, (m, c) in enumerate([("ISO", ISO_C), ("DTI", DTI_C), ("MD-dMRI", MD_C)]):
    means = [msd(code, m)[0] for code, _ in LOBES]; sds = [msd(code, m)[1] for code, _ in LOBES]
    ax.bar(x + (i - 1) * w, means, w, yerr=sds, capsize=2.5, color=c, edgecolor="white", linewidth=0.5,
           label=m, error_kw=dict(lw=0.8, ecolor="#777"))
ax.set_xticks(x); ax.set_xticklabels([nm for _, nm in LOBES], fontsize=8.6)
ax.set_ylabel("95th-percentile |E|  (V/m)", fontsize=10)
ax.set_title("Predicted tDCS E-field across three conductivity models   (M1 montage, n = 29)",
             fontsize=11.5, fontweight="bold", color=INK, pad=22)
ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.06))
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8.6)
ax.axvline(3.5, color="#cccccc", lw=1.0, ls="--", zorder=0)
ax.text(1.5, -0.135, "Cortex (gray matter)", ha="center", va="top", transform=ax.get_xaxis_transform(),
        fontsize=8.8, color=SUB, fontstyle="italic")
ax.text(5.5, -0.135, "White matter", ha="center", va="top", transform=ax.get_xaxis_transform(),
        fontsize=8.8, color=SUB, fontstyle="italic")
fig.subplots_adjust(left=0.085, right=0.97, top=0.85, bottom=0.18)
fig.savefig(os.path.join(R, "prelim_efield_bars.png"), dpi=300, facecolor="white")
fig.savefig(os.path.join(R, "prelim_efield_bars.pdf"), facecolor="white")
plt.close(fig)

# ---- two-panel table (Tier 1 | Tier 2) ----
T1 = [("Ctx_Frontal", "Frontal cortex"), ("Ctx_Parietal", "Parietal cortex"), ("Ctx_Temporal", "Temporal cortex"),
      ("Ctx_Occipital", "Occipital cortex"), ("WM_Frontal", "Frontal white matter"), ("WM_Parietal", "Parietal white matter"),
      ("WM_Temporal", "Temporal white matter"), ("WM_Occipital", "Occipital white matter"), ("CC", "Corpus callosum"),
      ("Mesencephalon", "Mesencephalon (midbrain)"), ("Pons", "Pons")]
T2 = [("Thalamus_L", "Thalamus (L)"), ("Thalamus_R", "Thalamus (R)"), ("Caudate_L", "Caudate (L)"), ("Caudate_R", "Caudate (R)"),
      ("Putamen_L", "Putamen (L)"), ("Putamen_R", "Putamen (R)"), ("Pallidum_L", "Pallidum (L)"), ("Pallidum_R", "Pallidum (R)"),
      ("Accumbens_L", "Accumbens (L)"), ("Accumbens_R", "Accumbens (R)"), ("Hippocampus_L", "Hippocampus (L)"),
      ("Hippocampus_R", "Hippocampus (R)"), ("Amygdala_L", "Amygdala (L)"), ("Amygdala_R", "Amygdala (R)")]
BAND = "#e7e9ec"
def P(a, b, f): return a + (b - a) * f
CH = [("Region", "left", 0.005), ("ISO", "right", 0.50), ("DTI", "right", 0.66),
      ("MD-dMRI", "right", 0.86), (r"$\Delta$ (%)", "right", 0.99)]

figT = plt.figure(figsize=(11.6, 6.4), dpi=300)
ax = figT.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.patch.set_visible(False)
ax.text(0.5, 0.955, "Predicted tDCS E-field across three conductivity models  (M1 montage, n = 29)",
        ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
TOP, ROWH = 0.895, 0.046
def panel(members, px0, px1, tlabel):
    ax.add_patch(Rectangle((px0, TOP), px1 - px0, 0.030, facecolor=BAND, edgecolor="none"))
    ax.text(px0 + 0.006, TOP + 0.015, tlabel, ha="left", va="center", fontsize=9.6, fontweight="bold", color=INK)
    yh = TOP - 0.020
    for name, al, f in CH:
        ax.text(P(px0, px1, f), yh, name, ha=al, va="center", fontsize=8.7, fontweight="bold", color=SUB)
    ry = yh - 0.016; ax.plot([px0, px1], [ry, ry], color=INK, lw=1.0)
    y = ry - ROWH * 0.62
    for code, nm in members:
        iso, dti, mdd = med(code, "ISO"), med(code, "DTI"), med(code, "MD-dMRI")
        try: dd = float(dmod.get(nm, "nan"))
        except ValueError: dd = float("nan")
        ax.text(P(px0, px1, 0.005), y, nm, ha="left", va="center", fontsize=8.6, color=INK)
        for val, f in [(iso, 0.50), (dti, 0.66), (mdd, 0.86)]:
            ax.text(P(px0, px1, f), y, f"{val:.3f}", ha="right", va="center", fontsize=8.6, color=INK)
        dc = DTI_C if dd < 0 else MD_C if dd > 0 else SUB
        ax.text(P(px0, px1, 0.99), y, f"{dd:+.1f}" if dd == dd else "", ha="right", va="center", fontsize=8.6, color=dc)
        y -= ROWH
    ax.plot([px0, px1], [y + ROWH * 0.5, y + ROWH * 0.5], color=INK, lw=1.0)
    return y + ROWH * 0.5
b1 = panel(T1, 0.035, 0.49, "Tier 1   Cortical / white-matter lobes, corpus callosum, brainstem")
b2 = panel(T2, 0.51, 0.965, "Tier 2   Subcortical gray matter (FreeSurfer aseg)")
y0 = b1 - 0.040; fs = 7.6
ax.text(0.035, y0, "Cohort-median 95th-percentile |E| (V/m) over GM+WM in each region. ISO, isotropic literature",
        ha="left", va="top", fontsize=fs, color=SUB)
ax.text(0.035, y0 - 0.033, r"conductivity; DTI, single-shell diffusion tensor; MD-dMRI, QTI mean tensor $\langle D\rangle$.",
        ha="left", va="top", fontsize=fs, color=SUB)
ax.text(0.035, y0 - 0.066, r"$\Delta$, MD-dMRI vs. DTI (median per-subject percent difference; significance in Table 2).",
        ha="left", va="top", fontsize=fs, color=SUB)
W, Hh = figT.get_size_inches()
bb = Bbox.from_extents(0.02 * W, (min(b1, b2, y0 - 0.085) - 0.01) * Hh, 0.985 * W, 0.99 * Hh)
figT.savefig(os.path.join(R, "prelim_efield_table.png"), dpi=300, bbox_inches=bb, facecolor="white", pad_inches=0.1)
figT.savefig(os.path.join(R, "prelim_efield_table.pdf"), bbox_inches=bb, facecolor="white", pad_inches=0.1)
print("wrote prelim_efield_bars.{png,pdf} + prelim_efield_table.{png,pdf}")
