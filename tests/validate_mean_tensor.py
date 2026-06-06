"""
tests/validate_mean_tensor.py — Reproducible validation of the σ ∝ ⟨D⟩ model's data
foundation, recomputed from dps.mat so a reviewer can verify the docstring claims.

    simnibs_python tests/validate_mean_tensor.py
    (or any Python with numpy / scipy / nibabel and access to the configured dps.mat)

PART A (always): recompute from dps.mat and ASSERT the facts 01d/derivation claim
  - ⟨D⟩ built from md?? is the mean tensor:  trace(⟨D⟩)/3 == MD
  - λ1(⟨D⟩) == ad ;  (λ2+λ3)/2 == rd
  - principal eigenvector of ⟨D⟩ == dps.u  (median angle ≈ 0°)
  - positive-definite in ~100% of brain ; genuinely triaxial (λ2≠λ3) in ~88%
  - limiting cases: the most fibre-like voxel ≈ diag(ad, rd, rd) in its eigenframe;
                    the most isotropic voxel ≈ MD·I
It also PRINTS the single authoritative numbers used across the docs:
  - base anisotropy        = median λ1/λ3 of ⟨D⟩ over the QTI brain mask
  - free-water bin MD      = median trace/3 of QTI bin-2 over the mask

PART B (optional, registration QA — runs only if the pipeline outputs exist):
  - median angle between dps.u (the Model-2 principal axis) and the dwi2cond DTI V1
    at several FA thresholds. The agreement is ~20° in core/dense WM (FA>0.5-0.6) and
    ~30° across all WM (FA>0.3): Model 2 and the DTI baseline only moderately share
    orientation, so the Model-vs-DTI E-field contrast reflects BOTH the eigenvalue
    (magnitude) difference AND a ~20-30° orientation difference — not magnitude alone.
"""
import os
import sys
import numpy as np
import scipy.io
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402

TRIAXIAL_REL = 0.05      # |λ2-λ3| / MD > this  ⇒ "genuinely triaxial"


def _sym_tensor(g):
    """Stack md?? component arrays (each N-long) into N×3×3 symmetric tensors."""
    n = g('mdxx').shape[0]
    D = np.zeros((n, 3, 3))
    D[:, 0, 0], D[:, 1, 1], D[:, 2, 2] = g('mdxx'), g('mdyy'), g('mdzz')
    D[:, 0, 1] = D[:, 1, 0] = g('mdxy')
    D[:, 0, 2] = D[:, 2, 0] = g('mdxz')
    D[:, 1, 2] = D[:, 2, 1] = g('mdyz')
    return D


def main():
    dps = scipy.io.loadmat(os.path.join(cfg["FIT_DIR"], "dps.mat"))['dps']
    mask = dps['mask'][0, 0].astype(bool)
    g = lambda f: np.real(dps[f][0, 0]).astype(np.float64)[mask] * 1e9   # SI m²/s → µm²/ms
    D = _sym_tensor(g)                                   # N×3×3, µm²/ms
    MD = np.real(dps['MD'][0, 0]).astype(np.float64)[mask]               # already µm²/ms
    ad = np.real(dps['ad'][0, 0]).astype(np.float64)[mask]
    rd = np.real(dps['rd'][0, 0]).astype(np.float64)[mask]
    u  = np.real(dps['u'][0, 0]).astype(np.float64)[mask]                # N×3 principal axis

    evals, evecs = np.linalg.eigh(D)                     # ascending
    l3, l2, l1 = evals[:, 0], evals[:, 1], evals[:, 2]
    v1 = evecs[:, :, 2]                                  # principal eigenvector of ⟨D⟩
    n = mask.sum()

    print(f"PART A — dps.mat mean-tensor validation ({n:,} brain voxels)")
    checks = []

    def check(name, ok, detail):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    e_md = np.abs((l1 + l2 + l3) / 3.0 - MD)
    check("trace(<D>)/3 == MD", e_md.max() < 1e-3, f"max|err|={e_md.max():.2e} µm²/ms")

    e_ad = np.abs(l1 - ad)   # a handful of near-degenerate voxels have larger ordering error
    check("λ1 == ad", np.median(e_ad) < 1e-3 and np.mean(e_ad) < 1e-2,
          f"median|err|={np.median(e_ad):.2e}, mean={np.mean(e_ad):.2e} µm²/ms")

    e_rd = np.abs((l2 + l3) / 2.0 - rd)
    check("(λ2+λ3)/2 == rd", np.median(e_rd) < 1e-3 and e_rd.max() < 0.1,
          f"median|err|={np.median(e_rd):.2e}, max={e_rd.max():.2e} µm²/ms")

    un = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(v1 * un, 1)), 0, 1)))
    check("v1(<D>) == dps.u", np.median(ang) < 2.0,
          f"median angle={np.median(ang):.2f}°, p95={np.percentile(ang,95):.2f}°")

    pd = 100 * np.mean(l3 > 0)
    check("positive-definite ~100%", pd > 99.5, f"{pd:.2f}% have λ3>0")

    triax = 100 * np.mean(np.abs(l2 - l3) / np.maximum(MD, 1e-6) > TRIAXIAL_REL)
    check(f"genuinely triaxial (|λ2-λ3|/MD>{TRIAXIAL_REL})", 75 < triax < 95,
          f"{triax:.1f}% of brain")

    # limiting cases (data-driven): most fibre-like vs most isotropic voxel
    fa = np.sqrt(0.5 * ((l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2) /
                 np.maximum(l1**2 + l2**2 + l3**2, 1e-12))
    i_fib, i_iso = int(np.argmax(fa)), int(np.argmin(fa))
    fib_cyl = abs(l2[i_fib] - l3[i_fib]) / MD[i_fib]
    check("single-fibre voxel ≈ diag(ad,rd,rd)", fib_cyl < 0.10,
          f"most-anisotropic voxel: λ=({l1[i_fib]:.2f},{l2[i_fib]:.2f},{l3[i_fib]:.2f}), "
          f"|λ2-λ3|/MD={fib_cyl:.3f}")
    iso_sp = (l1[i_iso] - l3[i_iso]) / MD[i_iso]
    check("isotropic voxel ≈ MD·I", iso_sp < 0.10,
          f"most-isotropic voxel: λ=({l1[i_iso]:.2f},{l2[i_iso]:.2f},{l3[i_iso]:.2f}), "
          f"(λ1-λ3)/MD={iso_sp:.3f}")

    # ── authoritative single numbers (defined mask + metric) ──────────────────
    aniso = l1 / np.maximum(l3, 1e-6)
    print("\n  AUTHORITATIVE NUMBERS (QTI brain mask, median):")
    print(f"    base anisotropy  median λ1/λ3 of ⟨D⟩ = {np.median(aniso):.2f}  "
          f"(mean {np.mean(aniso):.2f})")

    bins = dps['bin'][0, 0]
    fw_md_per_bin = []
    for b in range(bins.shape[1]):
        bb = bins[0, b]
        gb = lambda f: np.real(bb[f][0, 0]).astype(np.float64)[mask] * 1e9
        tr = (gb('mdxx') + gb('mdyy') + gb('mdzz')) / 3.0
        fw_md_per_bin.append(np.nanmedian(tr))
    fw_bin = int(np.nanargmax(fw_md_per_bin))
    print(f"    free-water bin = bin-{fw_bin}, median MD = {fw_md_per_bin[fw_bin]:.2f} µm²/ms "
          f"(literature free water ≈ 3.0 at 37 °C)")

    # ── PART B: registration QA (optional) ────────────────────────────────────
    print("\nPART B — registration QA (dps.u vs dwi2cond DTI V1 in core WM)")
    v1_t1_p = os.path.join(cfg["REG_DIR"], "v1_T1.nii.gz")
    dti_p   = os.path.join(cfg["M2M_DIR"], "DTI_coregT1_tensor.nii.gz")
    if os.path.exists(v1_t1_p) and os.path.exists(dti_p):
        v1img = nib.load(v1_t1_p); v1t1 = np.asarray(v1img.dataobj, float)
        dimg = nib.load(dti_p); dt = np.asarray(dimg.dataobj, float)
        if dt.shape[:3] == v1t1.shape[:3]:
            M = np.zeros(dt.shape[:3] + (3, 3))
            M[..., 0, 0], M[..., 0, 1], M[..., 0, 2] = dt[..., 0], dt[..., 1], dt[..., 2]
            M[..., 1, 0], M[..., 1, 1], M[..., 1, 2] = dt[..., 1], dt[..., 3], dt[..., 4]
            M[..., 2, 0], M[..., 2, 1], M[..., 2, 2] = dt[..., 2], dt[..., 4], dt[..., 5]
            ev, evec = np.linalg.eigh(M.reshape(-1, 3, 3))
            ev = ev.reshape(dt.shape[:3] + (3,)); evec = evec.reshape(dt.shape[:3] + (3, 3))
            l = ev; fa = np.sqrt(0.5 * ((l[...,2]-l[...,1])**2 + (l[...,1]-l[...,0])**2 +
                 (l[...,2]-l[...,0])**2) / np.maximum((l**2).sum(-1), 1e-12))
            dti_v1 = evec[..., 2]
            nrm = np.linalg.norm(v1t1, axis=-1)
            print("  median angle(dps.u, dwi2cond DTI V1) by WM mask:")
            for thr in (0.3, 0.5, 0.6):
                core = (fa > thr) & (nrm > 0.5)
                a = v1t1[core] / nrm[core][:, None]
                ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(a * dti_v1[core], 1)), 0, 1)))
                tag = "all WM" if thr == 0.3 else ("core WM" if thr == 0.5 else "dense WM")
                print(f"    FA>{thr} ({tag:7s}): n={core.sum():7,}  median={np.median(ang):.1f}°")
            print("  NOTE: dps.u and the DTI V1 differ by ~20-30°, so the Model-2-vs-DTI E-field")
            print("        contrast reflects orientation AND eigenvalue differences (not magnitude alone).")
        else:
            print(f"  SKIP: grid mismatch v1_T1 {v1t1.shape[:3]} vs DTI {dt.shape[:3]}")
    else:
        print("  SKIP: pipeline outputs not found (run 00_dwi2cond.sh + 01_register_dMRI_to_T1.sh)")

    print(f"\n{'ALL PART-A CHECKS PASSED' if all(checks) else 'SOME CHECKS FAILED'}")
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
