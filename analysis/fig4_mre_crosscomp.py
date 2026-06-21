"""fig4_mre_crosscomp.py - Figure 4 (3 panels), neutral panel labels (interpretation lives in the caption).
A: dE_model vs stiffness (null). B: |E| vs stiffness with the group+age line and the CSF-adjusted line (the
collapse), points colored by CSF fraction. C: nested-model partial-r forest for field~stiffness
(group -> +age -> +TIV -> +CSF) with 95% CI, collapsing to n.s. once CSF fraction is added.
Reads analysis/results/wholebrain_covariates.csv (+ the null q from the 09 whole-brain CSV).
"""
import os, csv, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "analysis")
from _stats import partial_pearson
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "mathtext.fontset": "dejavusans", "axes.linewidth": 0.9})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "analysis", "results")
rows = list(csv.DictReader(open(os.path.join(R, "wholebrain_covariates.csv"))))
grp = np.array([1.0 if r["group"] == "PD" else 0.0 for r in rows]); age = np.array([float(r["age"]) for r in rows])
stf = np.array([float(r["stiffness_kPa"]) for r in rows]); ef = np.array([float(r["E_MDdMRI"]) for r in rows])
de = np.array([float(r["dE_model_pct"]) for r in rows]); csf = np.array([float(r["CSF_frac"]) for r in rows])
tiv = np.array([float(r["TIV_cc"]) for r in rows]); pd_, hc = grp == 1, grp == 0; n = len(rows)
ORANGE, BLUE, LINE, INK, GRY = "#d95f02", "#1f6fb2", "#333333", "#1a1a1a", "#999999"

def adj_line(x, y, cov):
    X = np.column_stack([np.ones_like(x), x, cov]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    xr = np.array([x.min(), x.max()]); return xr, b[0] + b[1] * xr + cov.mean(0) @ b[2:]

wb09 = {row["pair"]: row for row in csv.DictReader(open(os.path.join(R, "mre_cohort_corr_wholebrain.csv")))}
Dn = wb09["dE_model_pct_vs_stiffness"]; r_null, q_null = float(Dn["rho"]), float(Dn["q"])

MODELS = [("group", grp.reshape(-1, 1), 1), ("+ age", np.column_stack([grp, age]), 2),
          ("+ TIV", np.column_stack([grp, age, tiv]), 3), ("+ CSF frac", np.column_stack([grp, age, csf]), 3)]
res = []
for lab, cov, k in MODELS:
    r, p = partial_pearson(stf, ef, cov)
    z, se = np.arctanh(r), 1 / np.sqrt(n - k - 3)        # Fisher-z 95% CI, covariate-adjusted SE
    res.append((lab, r, float(np.tanh(z - 1.96 * se)), float(np.tanh(z + 1.96 * se))))
r_ga, r_csf = res[1][1], res[3][1]

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.6, 4.3), dpi=300, gridspec_kw={"width_ratios": [1, 1.18, 0.95]})
fig.suptitle("Figure 4.   Electric field and conductivity-model difference vs. MRE stiffness (whole brain, n = 29)",
             fontsize=12.5, fontweight="bold", y=0.99, color=INK)

# A: the null
for m, c, lab in [(hc, BLUE, "HC"), (pd_, ORANGE, "PD")]:
    axA.scatter(stf[m], de[m], s=40, c=c, edgecolors="white", linewidths=0.5, alpha=0.9, label=lab, zorder=3)
xr, yr = adj_line(stf, de, grp.reshape(-1, 1)); axA.plot(xr, yr, color=LINE, lw=1.6, zorder=2)
axA.axhline(0, color="#bbbbbb", lw=0.8, ls="--", zorder=1)
axA.set_xlabel("MRE stiffness (kPa)", fontsize=9.5); axA.set_ylabel(r"$\Delta E_\mathrm{model}$  (%)", fontsize=9.5)
axA.set_title(r"A    $\Delta E_\mathrm{model}$ vs. MRE stiffness", fontsize=9.6, fontweight="bold", color=INK, pad=6, loc="left")
axA.text(0.04, 0.05, f"partial $r$ = {r_null:+.2f}\n$q$ = {q_null:.2f}  (n.s.)", transform=axA.transAxes, ha="left",
         va="bottom", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#f6f7f9", ec="#cfd4da", lw=0.8))
axA.legend(loc="upper right", frameon=False, fontsize=8.5, handletextpad=0.3)
axA.tick_params(labelsize=8); axA.spines[["top", "right"]].set_visible(False)

# B: the confound, with both regression lines
sc = axB.scatter(stf, ef, s=44, c=csf, cmap="viridis", edgecolors="white", linewidths=0.4, zorder=3)
xr, ys = adj_line(stf, ef, np.column_stack([grp, age])); axB.plot(xr, ys, color=LINE, lw=1.8, zorder=2, label="group + age")
xr, yd = adj_line(stf, ef, np.column_stack([grp, age, csf])); axB.plot(xr, yd, color=LINE, lw=1.6, ls="--", zorder=2, label="+ CSF fraction")
axB.set_xlabel("MRE stiffness (kPa)", fontsize=9.5); axB.set_ylabel("MD-dMRI  |E|  (V/m)", fontsize=9.5)
axB.set_title("B    |E| vs. MRE stiffness", fontsize=9.6, fontweight="bold", color=INK, pad=6, loc="left")
axB.text(0.04, 0.96, f"$r$ = +{r_ga:.2f}  (group + age)\n$r$ = +{r_csf:.2f}  (+ CSF, n.s.)", transform=axB.transAxes,
         ha="left", va="top", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="#f6f7f9", ec="#cfd4da", lw=0.8))
axB.legend(loc="lower right", frameon=False, fontsize=8, handlelength=1.9)
cb = fig.colorbar(sc, ax=axB, fraction=0.045, pad=0.02); cb.set_label("CSF fraction", fontsize=8); cb.ax.tick_params(labelsize=7)
axB.tick_params(labelsize=8); axB.spines[["top", "right"]].set_visible(False)

# C: nested-model coefficient forest
yy = np.arange(len(res))[::-1]
for (lab, r, lo, hi), y in zip(res, yy):
    sig = not (lo <= 0 <= hi); col = INK if sig else GRY
    axC.plot([lo, hi], [y, y], color=col, lw=2, zorder=2, solid_capstyle="round")
    axC.scatter([r], [y], s=46, color=col, zorder=3, edgecolors="white", linewidths=0.6)
axC.axvline(0, color="#bbbbbb", lw=0.9, ls="--")
axC.set_yticks(yy); axC.set_yticklabels([m[0] for m in res], fontsize=8.7)
axC.set_ylim(-0.6, len(res) - 0.4)
axC.set_xlabel(r"field-stiffness partial $r$  (95% CI)", fontsize=9.5)
axC.set_title("C    Adjustment for confounders", fontsize=9.6, fontweight="bold", color=INK, pad=6, loc="left")
axC.set_xlim(-0.25, 0.9); axC.tick_params(labelsize=8); axC.spines[["top", "right"]].set_visible(False)
axC.text(0.96, 0.04, "+ CSF: 95% CI spans 0", transform=axC.transAxes, ha="right", va="bottom", fontsize=7.6,
         color=GRY, style="italic")

fig.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.14, wspace=0.42)
fig.savefig(os.path.join(R, "fig4_mre_crosscomp.png"), dpi=300, facecolor="white")
fig.savefig(os.path.join(R, "fig4_mre_crosscomp.pdf"), facecolor="white")
print(f"wrote 3-panel fig4: null r={r_null:+.2f} q={q_null:.2f}; field-stiff {r_ga:+.2f} -> {r_csf:+.2f} (+CSF)")
