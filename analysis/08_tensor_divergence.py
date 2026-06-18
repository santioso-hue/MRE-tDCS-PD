"""08_tensor_divergence.py — per-ROI divergence between the single-shell DTI tensor and the QTI <D>
tensor, computed BEFORE any E-field claim.

The contribution is a MORE PRINCIPLED tensor construction (kurtosis-aware multi-shell <D> vs single-shell
Gaussian DTI). This script QUANTIFIES where the two estimates diverge; it does NOT assert <D> is more
accurate or more anisotropic. The novelty is only non-trivial if the tensors actually diverge in a
structured way -- if they are near-identical, "more principled" collapses to "fancier acquisition, no
effect", so we measure the divergence directly.

'vn' divides each tensor by det^(1/3), so MAGNITUDE cancels and only the eigenvalue RATIOS (lam1/lam3,
lam2/lam3) and the orientation reach the E-field. Hence we report orientation angle, FA difference, and
the ratio differences -- the divergence that can actually change |E|, not a magnitude difference 'vn' discards.

Inputs (both 6-comp FSL order xx,xy,xz,yy,yz,zz, in T1/mesh space):
  DTI:  m2m/DTI_coregT1_tensor.nii.gz   (dwi2cond)   [gated on ParkMRE_DTI for the cohort]
  <D>:  work/tensor_MD_dMRI.nii.gz       (03 output)
  ROIs: analysis/_rois.load_labeled
Output: results/<subject>/tensor_divergence.csv (one row per ROI), console table sorted by divergence.

Run:  PIPELINE_CONFIG=<subject config.sh> simnibs_python analysis/08_tensor_divergence.py
"""
import os
import sys
import csv
import numpy as np
import nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from _config import cfg  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rois import load_labeled, _labels_on_grid, eigh_6comp, fa_from_evals, v1_angle_deg  # noqa: E402

# Orientation threshold anchored to the measured core-WM V1 agreement between <D> and the independent
# single-shell DTI (~8 deg; conductivity_models_derivation.md "Registration" / "DTI baseline"). A median
# angle at/below this is within the registration+acquisition noise floor -> tensors "effectively equal".
ANGLE_FLOOR_DEG = 8.0
RATIO_TOL = 0.10              # chosen fractional tolerance on lam1/lam3 and lam2/lam3 for the "equal" verdict
PD_EPS = 1e-3                 # positive-definite floor on the smallest eigenvalue (um2/ms), as in _rois/_vn_check

DTI = os.path.join(cfg["M2M_DIR"], "DTI_coregT1_tensor.nii.gz")
DD = os.path.join(cfg["WORK_DIR"], "tensor_MD_dMRI.nii.gz")


def _miqr(x):
    return float(np.median(x)), float(np.percentile(x, 75) - np.percentile(x, 25))


def main():
    if not os.path.exists(DTI):
        sys.exit(f"DTI tensor not found: {DTI}\n(the DTI arm is gated on ParkMRE_DTI; run 01_dwi2cond first)")
    if not os.path.exists(DD):
        sys.exit(f"<D> tensor not found: {DD}\n(run 03_build_conductivity_tensor.py first)")

    labeled, aff, names = load_labeled(cfg["REG_DIR"])
    dti_img = nib.load(DTI)
    tdti = np.asarray(dti_img.dataobj, float)
    tdd = np.asarray(nib.load(DD).dataobj, float)
    lab = _labels_on_grid(dti_img, labeled, aff)        # DTI is on the T1/mesh grid, same as <D>

    rows = []
    for k, n in names.items():
        sel = ((lab == k) & np.isfinite(tdti[..., 0]) & np.isfinite(tdd[..., 0])
               & (np.abs(tdti[..., 0]) > 1e-6) & (np.abs(tdd[..., 0]) > 1e-6))
        if not sel.any():
            continue
        ev_d, vec_d = eigh_6comp(tdd, sel)
        ev_t, vec_t = eigh_6comp(tdti, sel)
        pd = (ev_d[:, 0] > PD_EPS) & (ev_t[:, 0] > PD_EPS)   # both positive-definite
        if not pd.any():
            continue
        ev_d, vec_d, ev_t, vec_t = ev_d[pd], vec_d[pd], ev_t[pd], vec_t[pd]

        ang = v1_angle_deg(vec_d, vec_t)
        dfa = fa_from_evals(ev_d) - fa_from_evals(ev_t)
        dr1 = ev_d[:, 2] / ev_d[:, 0] - ev_t[:, 2] / ev_t[:, 0]   # delta(lam1/lam3)
        dr2 = ev_d[:, 1] / ev_d[:, 0] - ev_t[:, 1] / ev_t[:, 0]   # delta(lam2/lam3)

        a_med, a_iqr = _miqr(ang); fa_med, fa_iqr = _miqr(dfa)
        r1_med, r1_iqr = _miqr(dr1); r2_med, r2_iqr = _miqr(dr2)
        # "effectively equal" = within the orientation noise floor AND both ratios within tolerance
        ref_r1 = float(np.median(ev_t[:, 2] / ev_t[:, 0])); ref_r2 = float(np.median(ev_t[:, 1] / ev_t[:, 0]))
        equal = (a_med <= ANGLE_FLOOR_DEG
                 and abs(r1_med) / max(ref_r1, 1e-6) <= RATIO_TOL
                 and abs(r2_med) / max(ref_r2, 1e-6) <= RATIO_TOL)
        rows.append(dict(ROI=n, n_vox=int(pd.sum()), angle_med=a_med, angle_iqr=a_iqr,
                         dFA_med=fa_med, dFA_iqr=fa_iqr, dR1_med=r1_med, dR1_iqr=r1_iqr,
                         dR2_med=r2_med, dR2_iqr=r2_iqr, verdict="equal" if equal else "diverge"))

    rows.sort(key=lambda r: r["angle_med"], reverse=True)    # most divergent (WM) at the top

    out = os.path.join(ROOT, "analysis", "results", cfg["SUBJECT"], "tensor_divergence.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = ["ROI", "n_vox", "angle_med", "angle_iqr", "dFA_med", "dFA_iqr",
            "dR1_med", "dR1_iqr", "dR2_med", "dR2_iqr", "verdict"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4g}" if isinstance(r[c], float) else r[c]) for c in cols})
    print(f"Per-ROI tensor divergence -> {out}")
    print(f"(DTI vs <D>; 'equal' = median angle <= {ANGLE_FLOOR_DEG} deg AND |d ratio| within {100*RATIO_TOL:.0f}%)")
    print(f"{'ROI':22s}{'angle deg med/IQR':>20s}{'dFA':>8s}{'d(l1/l3)':>10s}{'verdict':>10s}")
    for r in rows:
        a = f"{r['angle_med']:.1f}/{r['angle_iqr']:.1f}"
        print(f"{r['ROI']:22s}{a:>20s}{r['dFA_med']:>8.3f}{r['dR1_med']:>10.2f}{r['verdict']:>10s}")
    print("\nMORE PRINCIPLED tensor (kurtosis-aware multi-shell <D> vs single-shell DTI): this QUANTIFIES")
    print("divergence, it does not assert <D> is correct. Single subject -> across-ROI pattern, not a statistic.")


if __name__ == "__main__":
    main()
