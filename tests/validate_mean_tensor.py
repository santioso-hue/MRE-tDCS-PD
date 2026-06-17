"""
tests/validate_mean_tensor.py -- assert the QTI covariance mean tensor matches the facts the
sigma ~ <D> derivation claims, so a reviewer can verify them.

    simnibs_python tests/validate_mean_tensor.py

PART A: rebuild <D> from cov_mfs m(2:7) (Mandel Voigt) and check against cov_dps:
trace/3==MD, lambda1==ad, (lambda2+lambda3)/2==rd, v1==u (all over PD voxels).
PART B (optional): median V1 angle between registered QTI <D> and dwi2cond DTI in core WM.
"""
import os
import sys
import numpy as np
import scipy.io
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402


def main():
    L = lambda f, k: scipy.io.loadmat(f, squeeze_me=True, struct_as_record=False)[k]  # noqa: E731
    mfs = L(cfg["QTI_MFS"], "mfs")
    dps = L(cfg["QTI_DPS"], "dps")
    mask = np.real(np.asarray(mfs.mask)).astype(bool)
    s2 = np.sqrt(2.0)
    D6 = np.asarray(mfs.m)[..., 1:7][mask] * 1e9                 # Mandel Voigt, SI -> um2/ms
    n = int(mask.sum())
    D = np.zeros((n, 3, 3))
    D[:, 0, 0], D[:, 1, 1], D[:, 2, 2] = D6[:, 0], D6[:, 1], D6[:, 2]
    D[:, 0, 1] = D[:, 1, 0] = D6[:, 3] / s2
    D[:, 0, 2] = D[:, 2, 0] = D6[:, 4] / s2
    D[:, 1, 2] = D[:, 2, 1] = D6[:, 5] / s2
    g = lambda f: np.real(np.asarray(getattr(dps, f))).astype(np.float64)[mask]  # noqa: E731
    MD, ad, rd = g("MD"), g("ad"), g("rd")
    u = np.real(np.asarray(dps.u)).astype(np.float64)[mask]

    evals, evecs = np.linalg.eigh(D)                             # ascending
    l3, l2, l1 = evals[:, 0], evals[:, 1], evals[:, 2]
    v1 = evecs[:, :, 2]
    pdm = l3 > 0                                                 # PD voxels: ad/rd/u meaningful there

    print(f"PART A — QTI covariance mean-tensor validation ({n:,} brain voxels)")
    checks = []

    def check(name, ok, detail):
        checks.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    e_md = np.abs((l1 + l2 + l3) / 3.0 - MD)
    check("trace(<D>)/3 == cov.MD", np.median(e_md) < 1e-3, f"median|err|={np.median(e_md):.2e} um2/ms")

    e_ad = np.abs(l1[pdm] - ad[pdm])
    check("lambda1 == ad (PD voxels)", np.median(e_ad) < 1e-3, f"median|err|={np.median(e_ad):.2e} um2/ms")

    e_rd = np.abs((l2[pdm] + l3[pdm]) / 2.0 - rd[pdm])
    check("(lambda2+lambda3)/2 == rd (PD voxels)", np.median(e_rd) < 1e-3, f"median|err|={np.median(e_rd):.2e} um2/ms")

    un = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(v1 * un, 1)), 0, 1)))
    check("v1(<D>) == cov.u (PD voxels)", np.median(ang[pdm]) < 2.0, f"median angle={np.median(ang[pdm]):.2f} deg")

    # Negative control: the Mandel /sqrt2 de-scaling of the off-diagonals is load-bearing. Rebuild <D>
    # without it and confirm v1 then disagrees with cov.u, so a regression that drops the sqrt2 cannot
    # pass the checks above silently.
    Dbad = D.copy()
    Dbad[:, 0, 1] = Dbad[:, 1, 0] = D6[:, 3]
    Dbad[:, 0, 2] = Dbad[:, 2, 0] = D6[:, 4]
    Dbad[:, 1, 2] = Dbad[:, 2, 1] = D6[:, 5]
    vbad = np.linalg.eigh(Dbad)[1][:, :, 2]
    abad = np.degrees(np.arccos(np.clip(np.abs(np.sum(vbad * un, 1)), 0, 1)))
    check("sqrt2 de-scaling is load-bearing (negative control)", np.median(abad[pdm]) > 1.0,
          f"no-sqrt2 v1 disagrees with cov.u by median {np.median(abad[pdm]):.2f} deg (correct path is ~0)")

    pd = 100 * np.mean(pdm)
    degen = 100 * np.mean((l3 <= 1e-3) | (MD < 0.2))
    check("cumulant fit positive-definite", pd > 80,
          f"{pd:.1f}% PD; {degen:.1f}% degenerate (non-PD / MD<0.2) -> isotropic fallback in 03")

    aniso = l1[pdm] / np.maximum(l3[pdm], 1e-6)
    print(f"\n  AUTHORITATIVE: base anisotropy median lambda1/lambda3 of <D> (PD voxels) = {np.median(aniso):.2f}")

    print("\nPART B — orientation QA (registered QTI <D> vs dwi2cond DTI V1 in core WM)")
    dti_t = os.path.join(cfg["M2M_DIR"], "DTI_coregT1_tensor.nii.gz")
    qti_t = os.path.join(cfg["REG_DIR"], "tensor_triaxial_T1.nii.gz")
    seg_f = os.path.join(cfg["M2M_DIR"], "final_tissues.nii.gz")
    if all(os.path.exists(p) for p in (dti_t, qti_t, seg_f)):
        def to_T(p):
            t6 = nib.load(p).get_fdata()
            T = np.zeros(t6.shape[:3] + (3, 3))
            T[..., 0, 0], T[..., 0, 1], T[..., 0, 2] = t6[..., 0], t6[..., 1], t6[..., 2]
            T[..., 1, 0], T[..., 1, 1], T[..., 1, 2] = t6[..., 1], t6[..., 3], t6[..., 4]
            T[..., 2, 0], T[..., 2, 1], T[..., 2, 2] = t6[..., 2], t6[..., 4], t6[..., 5]
            return T
        Tq, Td = to_T(qti_t), to_T(dti_t)
        seg = nib.load(seg_f).get_fdata(); seg = seg[..., 0] if seg.ndim == 4 else seg
        m = (seg == 1) & (np.abs(Tq).sum((-1, -2)) > 1e-9) & (np.abs(Td).sum((-1, -2)) > 1e-9)

        def feat(T):
            w, V = np.linalg.eigh(T[m]); w2 = np.sort(w, 1)[:, ::-1]
            fa = np.sqrt(.5) * np.sqrt((w2[:, 0] - w2[:, 1]) ** 2 + (w2[:, 1] - w2[:, 2]) ** 2 +
                                       (w2[:, 2] - w2[:, 0]) ** 2) / (np.sqrt((w2 ** 2).sum(1)) + 1e-20)
            return V[:, :, 2], fa
        vq, faq = feat(Tq); vd, _ = feat(Td)
        a = np.degrees(np.arccos(np.clip(np.abs(np.sum(vq * vd, 1)), 0, 1)))
        print(f"  QTI <D> vs dwi2cond DTI V1: core WM FA>0.4 median = {np.median(a[faq > 0.4]):.1f} deg "
              f"(n={int((faq > 0.4).sum()):,})")
    else:
        print("  (skipped: run 01_dwi2cond + 02_register_dmri_to_T1 first)")

    ok = all(checks)
    print(f"\n{'ALL PASS' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
