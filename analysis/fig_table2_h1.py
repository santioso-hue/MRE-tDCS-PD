"""fig_table2_h1.py - render Table 2 (H1 per-ROI summary, M1) as a NeuroImage-style figure.
Reads analysis/results/m1_results_summary.csv (Tier 1 + Tier 2). Writes m1_h1_table.{png,pdf}.
"""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "mathtext.fontset": "dejavusans"})

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "results")
rows = list(csv.DictReader(open(os.path.join(R, "m1_results_summary.csv"))))
t1 = [r for r in rows if r["Tier"] == "Tier 1"]
t2 = [r for r in rows if r["Tier"] == "Tier 2"]

INK, SUB, GRAY = "#1a1a1a", "#5f6368", "#9aa0a6"
BLUE, RED, HL, BAND = "#1f6fb2", "#c0392b", "#eef3fb", "#e7e9ec"
def P(a, b, f): return a + (b - a) * f
CH = [("Region", "left", 0.005), (r"$\Delta E_\mathrm{model}$ (%)", "right", 0.49),
      ("IQR (%)", "center", 0.61), (r"$r_\mathrm{rb}$", "right", 0.77),
      ("q", "right", 0.895), ("Dir", "center", 0.965)]

fig = plt.figure(figsize=(11.6, 6.6), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.patch.set_visible(False)   # so bbox_inches='tight' crops to the drawn content, not the full-figure axes
ax.text(0.5, 0.962, "Table 2.  H1 per-ROI summary: MD-dMRI vs. DTI electric field, M1 montage (n = 29)",
        ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
TOP, ROWH = 0.900, 0.045

def panel(data, px0, px1, tlabel):
    ax.add_patch(Rectangle((px0, TOP), px1 - px0, 0.030, facecolor=BAND, edgecolor="none"))
    ax.text(px0 + 0.006, TOP + 0.015, tlabel, ha="left", va="center", fontsize=9.6, fontweight="bold", color=INK)
    yh = TOP - 0.020
    for name, al, f in CH:
        ax.text(P(px0, px1, f), yh, name, ha=al, va="center", fontsize=8.8, fontweight="bold", color=SUB)
    ry = yh - 0.016; ax.plot([px0, px1], [ry, ry], color=INK, lw=1.0)
    y = ry - ROWH * 0.62
    for r in data:
        med = float(r["dE_model_median_pct"]); lo = float(r["dE_model_IQR_lo_pct"]); hi = float(r["dE_model_IQR_hi_pct"])
        rb = float(r["rank_biserial"]); sig = r["MDdMRI_vs_DTI_significant"] == "Y"
        if sig:
            ax.add_patch(Rectangle((px0, y - ROWH * 0.5), px1 - px0, ROWH, facecolor=HL, edgecolor="none", zorder=0))
        col = INK if sig else GRAY
        ax.text(P(px0, px1, 0.005), y, r["ROI"], ha="left", va="center", fontsize=8.5, color=col)
        ax.text(P(px0, px1, 0.49), y, f"{med:+.1f}", ha="right", va="center", fontsize=8.5, color=col,
                fontweight="bold" if sig else "normal")
        ax.text(P(px0, px1, 0.61), y, f"[{lo:+.1f}, {hi:+.1f}]", ha="center", va="center", fontsize=7.8, color=col)
        ax.text(P(px0, px1, 0.77), y, f"{rb:+.2f}", ha="right", va="center", fontsize=8.5, color=col)
        qv = float(r["q"]) if r.get("q") not in (None, "", "nan") else float("nan")
        qs = "" if qv != qv else ("<0.01" if qv < 0.01 else f"{qv:.2f}")
        ax.text(P(px0, px1, 0.895), y, qs, ha="right", va="center", fontsize=8.2, color=col)
        g, gc = ("▼", BLUE) if (sig and med < 0) else ("▲", RED) if (sig and med > 0) else ("ns", GRAY)
        ax.text(P(px0, px1, 0.965), y, g, ha="center", va="center", fontsize=8.3 if sig else 7.5, color=gc,
                style="normal" if sig else "italic")
        y -= ROWH
    ax.plot([px0, px1], [y + ROWH * 0.5, y + ROWH * 0.5], color=INK, lw=1.0)
    return y + ROWH * 0.5

b1 = panel(t1, 0.035, 0.49, "Tier 1   Cortical / white-matter lobes, corpus callosum, brainstem")
b2 = panel(t2, 0.51, 0.965, "Tier 2   Subcortical gray matter (FreeSurfer aseg)")

# definitions in the blank space under the left (Tier 1) column, wrapped to stay within the left panel
y0 = b1 - 0.038; dyl = 0.033; fs = 7.5
ax.text(0.035, y0, r"$\Delta E_\mathrm{model}$, percent difference in the 95th-percentile $|E|$ between the",
        ha="left", va="top", fontsize=fs, color=SUB)
ax.text(0.035, y0 - dyl, r"MD-dMRI and DTI models; cohort median and IQR (25–75th percentile).",
        ha="left", va="top", fontsize=fs, color=SUB)
ax.text(0.035, y0 - 2 * dyl, r"$r_\mathrm{rb}$, matched-pairs rank-biserial correlation;   "
        r"$q$, FDR-corrected Wilcoxon $p$.", ha="left", va="top", fontsize=fs, color=SUB)
y3 = y0 - 3 * dyl
ax.text(0.035, y3, "▼", color=BLUE, fontsize=8.0, ha="left", va="top")
ax.text(0.050, y3, "MD-dMRI < DTI,", color=SUB, fontsize=fs, ha="left", va="top")
ax.text(0.163, y3, "▲", color=RED, fontsize=8.0, ha="left", va="top")
ax.text(0.179, y3, "MD-dMRI > DTI  (q < 0.05);   ns, not significant.", color=SUB, fontsize=fs, ha="left", va="top")

# crop to the actual content (explicit bbox in inches; full-figure axes defeats bbox_inches='tight')
W, Hh = fig.get_size_inches()
bb = Bbox.from_extents(0.02 * W, (min(b1, b2, y3) - 0.020) * Hh, 0.985 * W, 0.99 * Hh)
fig.savefig(os.path.join(R, "m1_h1_table.png"), dpi=300, bbox_inches=bb, facecolor="white", pad_inches=0.1)
fig.savefig(os.path.join(R, "m1_h1_table.pdf"), bbox_inches=bb, facecolor="white", pad_inches=0.1)
print("wrote m1_h1_table.png + .pdf")
