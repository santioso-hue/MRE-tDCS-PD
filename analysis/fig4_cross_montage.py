"""Figure 4: per-subject conductivity-model effect (dE_model) on white matter vs cortex across the four
electrode montages, as paired boxplots.

For each subject and montage, dE_model is the lobe-mean of 100*(MD-dMRI_p95 - DTI_p95)/DTI_p95 over the
four white-matter lobes (one value) and over the four cortical lobes (one value), where p95 is the
95th-percentile electric-field magnitude. Each montage shows a boxplot of the per-subject white-matter
means (blue) beside a boxplot of the per-subject cortical means (purple). Reads
analysis/results/<subject>/roi_efield_<montage>.csv; writes fig4_cross_montage.{png,pdf}.
"""
import os, csv, glob
import numpy as np
from _figstyle import plt, INK, SUB, GRAY, WM_C, OKABE
from _figsave import save_fig

CTX_C = OKABE["purple"]   # cortex (GM): purple, distinct from both model colours
WM_COL = WM_C             # white matter: blue

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "results")
MONTAGES = ["M1", "DLPFC", "HD_M1", "HD_DLPFC"]
LAB = {"M1": "M1 pad", "DLPFC": "DLPFC pad", "HD_M1": "M1 HD", "HD_DLPFC": "DLPFC HD"}
CTX = ["Ctx_Frontal", "Ctx_Parietal", "Ctx_Temporal", "Ctx_Occipital"]
WM = ["WM_Frontal", "WM_Parietal", "WM_Temporal", "WM_Occipital"]


def lobe_de(row):
    """Per-lobe dE_model = 100*(MD-dMRI_p95 - DTI_p95)/DTI_p95, or nan if the row is missing/unparseable."""
    if not row:
        return np.nan
    try:
        dti, md = float(row["DTI_p95"]), float(row["MD-dMRI_p95"])
        return 100.0 * (md - dti) / dti if dti > 0 else np.nan
    except (KeyError, ValueError):
        return np.nan


def subject_means(M):
    """{subject: (wm_lobe_mean, ctx_lobe_mean)} of per-subject dE_model for montage M."""
    out = {}
    for p in sorted(glob.glob(os.path.join(R, "PD*", f"roi_efield_{M}.csv"))):
        subj = os.path.basename(os.path.dirname(p))
        rows = {r["ROI"]: r for r in csv.DictReader(open(p))}
        wm = np.nanmean([lobe_de(rows.get(l)) for l in WM])
        ctx = np.nanmean([lobe_de(rows.get(l)) for l in CTX])
        out[subj] = (wm, ctx)
    return out


# Gather per-subject means, keeping only subjects present in every montage (so a spaghetti line is complete).
per_montage = {M: subject_means(M) for M in MONTAGES}
subjects = sorted(set.intersection(*(set(per_montage[M]) for M in MONTAGES)))
n = len(subjects)

wm_mat = np.array([[per_montage[M][s][0] for M in MONTAGES] for s in subjects])    # (n, 4)
ctx_mat = np.array([[per_montage[M][s][1] for M in MONTAGES] for s in subjects])   # (n, 4)

CAPTION = (
    "Figure 4. Conductivity-model effect on white matter versus cortex across electrode montages, "
    f"Group 1 lobes, n = {n}. For each subject and montage, dE_model is the mean over the four "
    "white-matter lobes (blue) or the four cortical lobes (purple) of the per-lobe relative difference "
    "dE_model = 100 x (MD-dMRI minus DTI) / DTI, where the field summary is p95 |E|, the 95th-percentile "
    "electric-field magnitude over GM and WM elements in the lobe. Each montage shows a boxplot of the "
    "per-subject white-matter means beside a boxplot of the per-subject cortical means; boxes span the "
    "interquartile range with the median line, whiskers reach 1.5x the IQR, and the notch marks the "
    "median 95% confidence interval. Columns are the four montages: M1 pad (anode C3), "
    "DLPFC pad (anode F3), M1 HD (4x1 ring centred at C3), DLPFC HD (4x1 ring centred at F3); the pad "
    "montages return through Fp2, 2 mA total. The thin line marks dE_model = 0. The white-matter model "
    "effect is consistently negative (MD-dMRI below DTI) and the cortical effect sits near zero, in "
    "every montage and for essentially every subject."
)

MINUS = "−"   # true minus for the y tick labels

# One axis: the white-matter (negative) and cortical (near-zero) bands separate but the gap is narrow,
# so a single y-axis reads more cleanly than a break.
x = np.arange(len(MONTAGES), dtype=float)
OFF = 0.19          # half-separation of the WM / cortex box pair within a montage slot
BW = 0.26           # box width

fig, ax = plt.subplots(figsize=(3.4, 3.5), layout="constrained")

# Zero reference, behind everything.
ax.axhline(0, color=SUB, lw=0.7, zorder=1)


def draw_boxes(mat, center_off, color):
    """Notched boxplots of the per-subject means, one per montage, at x +/- center_off."""
    pos = x + center_off
    bp = ax.boxplot(
        [mat[:, i] for i in range(len(MONTAGES))], positions=pos, widths=BW,
        patch_artist=True, notch=True, showfliers=False, manage_ticks=False, zorder=4,
    )
    for box in bp["boxes"]:
        box.set(facecolor=color, alpha=0.32, edgecolor=color, linewidth=1.1)
    for med in bp["medians"]:
        med.set(color=color, linewidth=1.6)
    for el in bp["whiskers"] + bp["caps"]:
        el.set(color=color, linewidth=1.0)
    return pos


draw_boxes(wm_mat, -OFF, WM_COL)
draw_boxes(ctx_mat, +OFF, CTX_C)

ax.set_xticks(x)
ax.set_xticklabels([LAB[m] for m in MONTAGES])
ax.set_xlim(-0.5, len(MONTAGES) - 0.5)
ax.tick_params(axis="x", which="both", length=0)

ax.set_ylabel(r"$\Delta E_\mathrm{model}$ (%)")
ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0f}".replace("-", MINUS))

# Legend out of the data, in the top margin, frameless.
handles = [
    plt.Line2D([], [], color=WM_COL, lw=1.8, label="White matter"),
    plt.Line2D([], [], color=CTX_C, lw=1.8, label="Cortex (GM)"),
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=2, frameon=False, handletextpad=0.5, columnspacing=1.6, handlelength=1.6)

save_fig(fig, "fig4_cross_montage", CAPTION)
for M in MONTAGES:
    i = MONTAGES.index(M)
    print(f"  {M:9} WM median = {np.median(wm_mat[:, i]):+.1f}%   "
          f"cortex median = {np.median(ctx_mat[:, i]):+.1f}%   (n={n})")
