"""Figure 5 (2 panels) for the whole-brain MRE specificity check.
A: dE_model vs stiffness (null). B: |E| vs stiffness with the group+age line and the CSF-adjusted line,
points colored by CSF fraction.
Reads analysis/results/wholebrain_covariates.csv and the whole-brain MRE-correlation CSV from stage 09.
"""
import os, csv, sys
import numpy as np
from scipy import stats
sys.path.insert(0, "analysis")
from _figstyle import plt, INK, SUB, ISO_C, MDDMRI_C, panel_label
from _figsave import save_fig
from _stats import partial_pearson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(ROOT, "analysis", "results")
rows = list(csv.DictReader(open(os.path.join(R, "wholebrain_covariates.csv"))))
grp = np.array([1.0 if r["group"] == "PD" else 0.0 for r in rows]); age = np.array([float(r["age"]) for r in rows])
stf = np.array([float(r["stiffness_kPa"]) for r in rows]); ef = np.array([float(r["E_MDdMRI"]) for r in rows])
de = np.array([float(r["dE_model_pct"]) for r in rows]); csf = np.array([float(r["CSF_frac"]) for r in rows])
pd_, hc = grp == 1, grp == 0; n = len(rows)
LINE = INK
CSF_CMAP = "cividis"


def adj_line(x, y, cov):
    """Covariate-adjusted regression line: fit y ~ 1 + x + cov, then evaluate at cov held at its mean.
    Returns the two endpoint x's clipped to the data range and the fitted line there."""
    X = np.column_stack([np.ones_like(x), x, cov]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    xr = np.array([x.min(), x.max()]); return xr, b[0] + b[1] * xr + cov.mean(0) @ b[2:]


def adj_band(x, y, cov, level=0.95, npts=100):
    """95% CI band for the covariate-adjusted mean response, cov held at its mean. Returns (xg, lo, hi)
    over a dense grid clipped to the data range. The prediction variance is the textbook OLS form
    s^2 * xp (X'X)^-1 xp', with s^2 the residual mean square on df = n - p, so the band widens away from
    the design center exactly as the standard error of the fitted line does."""
    cov = cov.reshape(-1, 1) if cov.ndim == 1 else cov
    X = np.column_stack([np.ones_like(x), x, cov]); n, p = X.shape
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b; dof = n - p
    s2 = float(resid @ resid / dof)
    XtX_inv = np.linalg.inv(X.T @ X)
    xg = np.linspace(x.min(), x.max(), npts)
    covm = cov.mean(0)
    Xp = np.column_stack([np.ones_like(xg), xg, np.tile(covm, (npts, 1))])
    yhat = Xp @ b
    se = np.sqrt(s2 * np.einsum("ij,jk,ik->i", Xp, XtX_inv, Xp))
    tcrit = float(stats.t.ppf(0.5 + level / 2, dof))
    return xg, yhat - tcrit * se, yhat + tcrit * se


wb09 = {row["pair"]: row for row in csv.DictReader(open(os.path.join(R, "mre_cohort_corr_wholebrain.csv")))}
Dn = wb09["dE_model_pct_vs_stiffness"]; r_null, q_null = float(Dn["rho"]), float(Dn["q"])

r_ga, p_ga = partial_pearson(stf, ef, np.column_stack([grp, age]))
r_csf, p_csf = partial_pearson(stf, ef, np.column_stack([grp, age, csf]))
csf_tag = "" if (p_csf == p_csf and p_csf < 0.05) else "n.s."

CAPTION = (
    "Figure 5. Electric field and conductivity-model difference versus MRE stiffness, whole brain, n = 29. "
    "Each point is one subject. "
    "DeltaE_model = 100 x (MD-dMRI minus DTI) / DTI, the per-subject whole-brain percent difference in 95th-percentile "
    "electric-field magnitude between the MD-dMRI and DTI conductivity models. |E| is the MD-dMRI 95th-percentile "
    "electric-field magnitude in V/m over GM and WM elements. MRE stiffness is the whole-brain mean in kPa. "
    "CSF fraction is the per-subject cerebrospinal-fluid volume fraction of the head model. partial r is the partial "
    "Pearson correlation of |E| or DeltaE_model with MRE stiffness after the listed adjustment; q is the "
    "Benjamini-Hochberg FDR-adjusted p-value. In Panel A the partial r controls for group, PD versus HC. In Panel B "
    "the solid-line partial r controls for group and age, and the dashed-line partial r controls for group, age, and "
    "CSF fraction. "
    "Panel A plots DeltaE_model against MRE stiffness, points colored by group, with the group-adjusted regression "
    "line and the partial r and q annotated. Panel B plots |E| against MRE stiffness, points colored by CSF fraction, "
    "with two regression lines, group plus age, solid, and group plus age plus CSF fraction, dashed, and the "
    "corresponding partial r values annotated. "
    "Each regression line carries a shaded 95% confidence band for the covariate-adjusted mean response, computed "
    "from the standard error of the linear-model prediction with the adjustment covariates held at their cohort means."
)

MINUS = "−"  # true minus sign for signed text


def fmt_signed(v):
    return f"{v:+.2f}".replace("-", MINUS)


# Two equal data panels (A, B) plus a slim dedicated column for Panel B's colorbar, so the colorbar
# does not eat into B's width and both data boxes render at the same pixel width.
fig = plt.figure(figsize=(7.2, 3.2), layout="constrained")
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.04])
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[0, 2])

# A: dE_model vs stiffness. HC / PD as a deliberate colorblind-safe pair: HC = ISO gray, PD = MD-dMRI vermillion.
# Group-adjusted fit gets a faint 95% CI band (under the points, so it never obscures a subject).
xg, lo, hi = adj_band(stf, de, grp.reshape(-1, 1))
axA.fill_between(xg, lo, hi, color=LINE, alpha=0.07, lw=0, zorder=1)
for m, c, lab in [(hc, ISO_C, "HC"), (pd_, MDDMRI_C, "PD")]:
    axA.scatter(stf[m], de[m], s=28, c=c, alpha=0.85, label=lab, zorder=3)
xr, yr = adj_line(stf, de, grp.reshape(-1, 1))
axA.plot(xr, yr, color=LINE, lw=1.2, zorder=2)
axA.plot(xr, [0.0, 0.0], color=SUB, lw=0.6, ls=(0, (4, 3)), zorder=1)  # y=0 reference clipped to data range
axA.set_xlim(xr[0], xr[1])
axA.set_xlabel("MRE stiffness (kPa)")
axA.set_ylabel(r"$\Delta E_\mathrm{model}$ (%)")
axA.text(0.04, 0.90, f"partial $r$ = {fmt_signed(r_null)}, $q$ = {q_null:.2f} (n.s.)",
         transform=axA.transAxes, ha="left", va="top", color=SUB)
axA.legend(loc="upper right", handletextpad=0.3)
panel_label(axA, "A")

# B: |E| vs stiffness, points colored by CSF fraction on a perceptual continuous map.
# Each adjusted fit gets its own faint 95% CI band, drawn under the points so the CSF colors stay readable.
xg, lo, hi = adj_band(stf, ef, np.column_stack([grp, age]))
axB.fill_between(xg, lo, hi, color=LINE, alpha=0.07, lw=0, zorder=1)
xg, lo2, hi2 = adj_band(stf, ef, np.column_stack([grp, age, csf]))
axB.fill_between(xg, lo2, hi2, color=LINE, alpha=0.05, lw=0, zorder=1)
sc = axB.scatter(stf, ef, s=28, c=csf, cmap=CSF_CMAP, alpha=0.9, zorder=3)
xr, ys = adj_line(stf, ef, np.column_stack([grp, age]))
axB.plot(xr, ys, color=LINE, lw=1.2, zorder=2, label="group + age")
xr, yd = adj_line(stf, ef, np.column_stack([grp, age, csf]))
axB.plot(xr, yd, color=LINE, lw=1.0, ls=(0, (4, 3)), zorder=2, label="+ CSF fraction")
axB.set_xlim(xr[0], xr[1])
axB.set_xlabel("MRE stiffness (kPa)")
axB.set_ylabel("MD-dMRI |E| (V/m)")
csf_note = "" if not csf_tag else f" ({csf_tag})"
axB.text(0.04, 0.96, f"partial $r$ = {fmt_signed(r_ga)}\npartial $r$ = {fmt_signed(r_csf)}{csf_note}",
         transform=axB.transAxes, ha="left", va="top", color=SUB)
axB.legend(loc="lower right", handlelength=1.9, labelcolor=SUB)
panel_label(axB, "B")

cb = fig.colorbar(sc, cax=cax)
cb.set_label("CSF fraction")
cb.outline.set_visible(False)
cb.ax.tick_params(width=0.6)

save_fig(fig, "fig5_mre_specificity", CAPTION)
print(f"2-panel fig5: null r={r_null:+.2f} q={q_null:.2f}; field-stiff {r_ga:+.2f} -> {r_csf:+.2f} (+CSF)")
