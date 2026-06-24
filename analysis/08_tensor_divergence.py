"""08_tensor_divergence.py - per-ROI divergence between the single-shell DTI tensor and the QTI <D>
tensor.

Quantifies where the two tensor estimates (kurtosis-aware multi-shell <D> vs single-shell Gaussian DTI)
diverge. 'vn' divides each tensor by det^(1/3), so magnitude cancels and only the eigenvalue ratios
(lam1/lam3, lam2/lam3) and orientation reach the E-field; we therefore report orientation angle, FA
difference, and the two ratio differences. The equal/diverge verdict is MEDIAN-only on angle + the two
ratios; dFA and the IQRs are explanatory readouts and do not enter the verdict.

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
from _rois import load_labeled, _labels_on_grid, eigh_6comp, fa_from_evals, v1_angle_deg, PD_EPS, FILL_EPS  # noqa: E402

# ANGLE_FLOOR_DEG = a conservative reporting floor for the V1 angle (registration/reorientation noise). The
# verdict is DESCRIPTIVE: the per-ROI angle and eigenvalue-ratio differences are the reported quantities.
# 'vn' discards tensor magnitude, so only tensor shape (anisotropy ratios + orientation) reaches |E|;
# registration sensitivity of the orientation readout is assessed separately and reported with the results.
ANGLE_FLOOR_DEG = 8.0
RATIO_TOL = 0.10            # chosen fractional tolerance on lam1/lam3 and lam2/lam3 for the "equal" verdict
ANISO_FLOOR = 1.2          # below this DTI lam1/lam3 the ROI is ~isotropic: the ratio test is unstable and
                           # barely moves 'vn' |E|, so the verdict falls back to orientation-only there
                           # (keeps RATIO_TOL from imposing a stricter absolute bar in near-isotropic GM).
# PD_EPS / FILL_EPS imported from _rois (one source of truth). PD_EPS is a um2/ms floor, so BOTH tensors
# must be converted to um2/ms (DTI_TO_UM2MS below) before the positive-definite gate is applied.

DTI = os.path.join(cfg["M2M_DIR"], "DTI_coregT1_tensor.nii.gz")
DD = os.path.join(cfg["WORK_DIR"], "tensor_MD_dMRI.nii.gz")
# FSL dtifit emits the DTI tensor in mm2/s; the QTI <D> (03 output) is in um2/ms (1 um2/ms = 1e-3 mm2/s).
# Convert DTI to um2/ms once at load so the PD_EPS=1e-3 floor is unit-correct for BOTH models. angle / FA /
# eigenvalue ratios are scale-invariant, so this rescale only affects the positive-definite gate.
DTI_TO_UM2MS = 1000.0


def _miqr(x):
    return float(np.median(x)), float(np.percentile(x, 75) - np.percentile(x, 25))


def main():
    if not os.path.exists(DTI):
        sys.exit(f"DTI tensor not found: {DTI}\n(the DTI model is gated on ParkMRE_DTI; run 01_dwi2cond first)")
    if not os.path.exists(DD):
        sys.exit(f"<D> tensor not found: {DD}\n(run 03_build_conductivity_tensor.py first)")

    labeled, aff, names = load_labeled(cfg["REG_DIR"])
    dti_img = nib.load(DTI)
    tdti = np.asarray(dti_img.dataobj, float) * DTI_TO_UM2MS   # mm2/s -> um2/ms (see DTI_TO_UM2MS)
    tdd = np.asarray(nib.load(DD).dataobj, float)
    lab = _labels_on_grid(dti_img, labeled, aff)        # DTI is on the T1/mesh grid, same as <D>
    # unit sanity: both median MDs should sit ~1e-1..1e0 um2/ms; a ~1e-3 value means a tensor is still mm2/s
    mdt = (tdti[..., 0] + tdti[..., 3] + tdti[..., 5]) / 3.0
    mdd = (tdd[..., 0] + tdd[..., 3] + tdd[..., 5]) / 3.0
    bt = np.isfinite(mdt) & (np.abs(tdti[..., 0]) > FILL_EPS)
    bd = np.isfinite(mdd) & (np.abs(tdd[..., 0]) > FILL_EPS)
    print(f"unit check (median MD, um2/ms): DTI={np.median(mdt[bt]):.3f}  <D>={np.median(mdd[bd]):.3f}")

    rows = []
    skipped = []   # (ROI name, reason) for every ROI that produced no row; asserted empty below
    for k, n in names.items():
        sel = ((lab == k) & np.isfinite(tdti[..., 0]) & np.isfinite(tdd[..., 0])
               & (np.abs(tdti[..., 0]) > FILL_EPS) & (np.abs(tdd[..., 0]) > FILL_EPS))
        if not sel.any():
            reason = "no finite non-fill voxels overlapping the labels (DTI or <D>)"
            print(f"WARN: ROI '{n}' skipped: {reason}", file=sys.stderr)
            skipped.append((n, reason))
            continue
        ev_d, vec_d = eigh_6comp(tdd, sel)
        ev_t, vec_t = eigh_6comp(tdti, sel)
        pd = (ev_d[:, 0] > PD_EPS) & (ev_t[:, 0] > PD_EPS)   # both positive-definite
        if not pd.any():
            reason = "no voxels positive-definite in both DTI and <D>"
            print(f"WARN: ROI '{n}' skipped: {reason}", file=sys.stderr)
            skipped.append((n, reason))
            continue
        ev_d, vec_d, ev_t, vec_t = ev_d[pd], vec_d[pd], ev_t[pd], vec_t[pd]

        ang = v1_angle_deg(vec_d, vec_t)
        dfa = fa_from_evals(ev_d) - fa_from_evals(ev_t)
        dr1 = ev_d[:, 2] / ev_d[:, 0] - ev_t[:, 2] / ev_t[:, 0]   # delta(lam1/lam3)
        dr2 = ev_d[:, 1] / ev_d[:, 0] - ev_t[:, 1] / ev_t[:, 0]   # delta(lam2/lam3)

        a_med, a_iqr = _miqr(ang); fa_med, fa_iqr = _miqr(dfa)
        r1_med, r1_iqr = _miqr(dr1); r2_med, r2_iqr = _miqr(dr2)
        # "effectively equal" = within the typical cross-method angle AND (for anisotropic ROIs) both
        # eigenvalue ratios within tolerance. In ~isotropic ROIs (DTI lam1/lam3 < ANISO_FLOOR) the ratio
        # test is unstable and 'vn'-irrelevant, so the verdict is orientation-only there.
        ref_r1 = float(np.median(ev_t[:, 2] / ev_t[:, 0])); ref_r2 = float(np.median(ev_t[:, 1] / ev_t[:, 0]))
        ratios_ok = (ref_r1 < ANISO_FLOOR
                     or (abs(r1_med) / ref_r1 <= RATIO_TOL and abs(r2_med) / ref_r2 <= RATIO_TOL))
        equal = a_med <= ANGLE_FLOOR_DEG and ratios_ok
        rows.append(dict(ROI=n, n_vox=int(pd.sum()), angle_med=a_med, angle_iqr=a_iqr,
                         dFA_med=fa_med, dFA_iqr=fa_iqr, dR1_med=r1_med, dR1_iqr=r1_iqr,
                         dR2_med=r2_med, dR2_iqr=r2_iqr, verdict="equal" if equal else "diverge"))

    rows.sort(key=lambda r: r["angle_med"], reverse=True)    # most divergent (WM) at the top

    # Fail loudly rather than write a TRUNCATED per-subject file: the cohort aggregate needs a consistent
    # n, so every ROI in the loaded label set must yield a row. A short file silently poisons the aggregate.
    expected = len(names)
    if len(rows) != expected:
        miss = "; ".join(f"{nm} ({why})" for nm, why in skipped)
        sys.exit(f"tensor_divergence: emitted {len(rows)}/{expected} ROI rows; "
                 f"{expected - len(rows)} skipped: {miss}\n"
                 "Refusing to write a truncated per-subject CSV (would corrupt the cohort aggregate).")

    out = os.path.join(ROOT, "analysis", "results", cfg["SUBJECT"], "tensor_divergence.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = ["ROI", "n_vox", "angle_med", "angle_iqr", "dFA_med", "dFA_iqr",
            "dR1_med", "dR1_iqr", "dR2_med", "dR2_iqr", "verdict"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4g}" if isinstance(r[c], float) else r[c]) for c in cols})
    print(f"Per-ROI tensor divergence -> {out}")
    print(f"(DTI vs <D>; 'equal' = median angle <= {ANGLE_FLOOR_DEG} deg AND, for anisotropic ROIs, "
          f"|d ratio| within {100*RATIO_TOL:.0f}%; orientation-only where DTI lam1/lam3 < {ANISO_FLOOR})")
    print(f"{'ROI':22s}{'angle deg med/IQR':>20s}{'dFA':>8s}{'d(l1/l3)':>10s}{'verdict':>10s}")
    for r in rows:
        a = f"{r['angle_med']:.1f}/{r['angle_iqr']:.1f}"
        print(f"{r['ROI']:22s}{a:>20s}{r['dFA_med']:>8.3f}{r['dR1_med']:>10.2f}{r['verdict']:>10s}")


if __name__ == "__main__":
    main()
