"""
prepare_dmri_tensor.py — Reconstruct the QTI covariance mean tensor <D> and supporting maps from the
md-dmri toolbox covariance fit (Westin et al. 2016), in dMRI space, ready for registration to T1.
Called by 02_register_dmri_to_T1.sh.

Upstream: pipeline/run_qti_cov.m runs the md-dmri `dtd_covariance` fit (constrained, regularized,
heteroscedastic) on the motion/eddy-corrected data and writes qti_cov/cov_mfs.mat (model fit) and
cov_dps.mat (derived parameters). <D> is the first cumulant of the QTI signal model — the macroscopic
mean diffusion tensor — stored in cov_mfs.m(:,:,:,2:7) in Mandel (sqrt(2)) Voigt order, SI units. We
reconstruct the full triaxial <D>, take its eigenvalues, and use the toolbox's principal eigenvector u
and derived MD / uFA.

Why the covariance fit and not the full DTD (de Almeida Martins/Topgaard) Monte-Carlo fit: the cumulant
mean tensor is the standard QTI estimate, is not magnitude-inflated (MD ~0.97 vs ~1.53 um2/ms for the
Monte-Carlo mean here), and has a reliable eigenframe (its principal axis agrees with a robust low-b DTI
to ~14 deg in core WM, vs ~40 deg for the Monte-Carlo mean).

Outputs (registration/, dMRI space):
  tensor_triaxial_dMRI.nii.gz  6-comp <D> [Dxx,Dxy,Dxz,Dyy,Dyz,Dzz], um2/ms (vecreg -> orientation frame)
  lam1/lam2/lam3_dMRI.nii.gz   eigenvalues l1>=l2>=l3 (scalar; trilinear -> magnitude, preserves anisotropy)
  v1_dMRI.nii.gz               principal eigenvector cov_dps.u (vecreg -> T1; anchors the principal axis)
  MD_dps_dMRI.nii.gz           mean diffusivity (QA + post-hoc MRE microstructure comparison)
  C_mu_dps_dMRI.nii.gz         microscopic FA uFA (post-hoc MRE comparison only; see note)
  dMRI_mask.nii.gz             QTI brain mask

The conductivity model is sigma proportional to <D> via SimNIBS 'vn' (built in 03). Only the eigenvalue
RATIOS and orientation of <D> survive that mapping. uFA is orientation-invariant (no macroscopic
eigenframe) and is written only for the post-hoc MRE comparison.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg  # noqa: E402

import numpy as np       # noqa: E402
import nibabel as nib    # noqa: E402
import scipy.io          # noqa: E402

FDIR, RDIR = cfg["FIT_DIR"], cfg["REG_DIR"]
os.makedirs(RDIR, exist_ok=True)
ref = nib.load(os.path.join(FDIR, "dtd_covariance_C_mu.nii.gz"))   # dMRI-space affine/header


def save(arr, name):
    hdr = ref.header.copy()
    hdr.set_data_shape(arr.shape)
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), ref.affine, hdr), os.path.join(RDIR, name))
    print(f"  saved {name}")


def _load(path, key):
    return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)[key]


print("Loading QTI covariance fit (cov_mfs.mat, cov_dps.mat)...")
mfs = _load(cfg["QTI_MFS"], "mfs")
dps = _load(cfg["QTI_DPS"], "dps")
mask = np.real(np.asarray(mfs.mask)).astype(bool)
assert ref.shape == mask.shape, f"shape mismatch: C_mu {ref.shape} vs cov_mfs {mask.shape}"

# --- mean tensor <D> from cov_mfs.m(:,:,:,2:7): Mandel (sqrt2) Voigt, SI m2/s -> um2/ms ---
S2 = np.sqrt(2.0)
D6 = np.asarray(mfs.m)[..., 1:7] * 1.0e9                     # [Dxx, Dyy, Dzz, sqrt2*Dxy, sqrt2*Dxz, sqrt2*Dyz]
mdxx, mdyy, mdzz = D6[..., 0], D6[..., 1], D6[..., 2]
mdxy, mdxz, mdyz = D6[..., 3] / S2, D6[..., 4] / S2, D6[..., 5] / S2
MD = np.real(np.asarray(dps.MD)).astype(np.float64)         # toolbox MD (um2/ms)

D = np.zeros(mask.shape + (3, 3))
D[..., 0, 0] = mdxx; D[..., 1, 1] = mdyy; D[..., 2, 2] = mdzz
D[..., 0, 1] = D[..., 1, 0] = mdxy
D[..., 0, 2] = D[..., 2, 0] = mdxz
D[..., 1, 2] = D[..., 2, 1] = mdyz
ev = np.linalg.eigvalsh(D[mask])                            # ascending l3<=l2<=l1
l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]
trace_md = (mdxx + mdyy + mdzz) / 3.0
err = np.abs(trace_md[mask] - MD[mask])                     # median (cov.MD is toolbox-clamped in extremes)

# The QTI cumulant mean tensor is degenerate in a minority of voxels: non-positive-definite, or an
# implausibly low MD (< 0.2 um2/ms, ~8x below real WM) with FA ~ 1 — a failed fit, not real tissue.
# Anisotropy/orientation cannot be trusted there, so those voxels fall back to ISOTROPIC (l1=l2=l3 ->
# 'vn' yields the literature sigma0). Real high-anisotropy WM (PD, plausible MD) is kept and capped at
# 10:1 by SimNIBS like every model. Real tissue (FA<0.5) is ~98% PD; the failures are noise/edge voxels.
md_m = MD[mask]
degen = (l3 <= 1e-3) | (md_m < 0.2)
iso = np.clip(md_m, 1e-3, None)
l1 = np.where(degen, iso, l1); l2 = np.where(degen, iso, l2); l3 = np.where(degen, iso, l3)
print(f"  trace/3 vs cov.MD median|err|={np.median(err):.4f} um2/ms; "
      f"degenerate->isotropic {100 * np.mean(degen):.1f}%; "
      f"median lambda {np.median(l1):.3f}/{np.median(l2):.3f}/{np.median(l3):.3f}")
assert np.median(err) < 1e-2, "trace/3 != cov.MD (median): Mandel/scale convention bug"
assert np.mean(degen) < 0.25, "QTI cumulant fit degenerate in >25% of brain — check fit/data quality"

lam1 = np.zeros(mask.shape); lam2 = np.zeros(mask.shape); lam3 = np.zeros(mask.shape)
lam1[mask] = l1; lam2[mask] = l2; lam3[mask] = l3
for a in (mdxx, mdyy, mdzz, mdxy, mdxz, mdyz):
    a[~mask] = 0.0
tensor = np.stack([mdxx, mdxy, mdxz, mdyy, mdyz, mdzz], axis=-1)  # FSL dtifit order
save(tensor, "tensor_triaxial_dMRI.nii.gz")
save(lam1, "lam1_dMRI.nii.gz"); save(lam2, "lam2_dMRI.nii.gz"); save(lam3, "lam3_dMRI.nii.gz")

# FA(<D>) from the post-fallback eigenvalues — the nonlinear fnirt -> T1 driver (matches dwi2cond's
# FA->T1 registration). Degenerate voxels were isotropized above, so their FA is 0 (not a spurious 1),
# giving a clean registration contrast.
denom = np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)
fa_m = np.sqrt(0.5) * np.sqrt((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) / np.where(denom > 1e-9, denom, 1.0)
fa_full = np.zeros(mask.shape); fa_full[mask] = np.clip(fa_m, 0.0, 1.0)
save(fa_full, "FA_dMRI.nii.gz")

# --- principal eigenvector v1 = cov_dps.u (real, unit, masked) ---
u = np.real(np.asarray(dps.u)).astype(np.float64)
u[~mask] = 0.0
n = np.linalg.norm(u, axis=-1, keepdims=True)
save(u / np.where(n > 0.1, n, 1.0), "v1_dMRI.nii.gz")

# --- QA / post-hoc MRE-comparison maps ---
ufa = np.clip(np.nan_to_num(np.real(np.asarray(dps.uFA)).astype(np.float64)), 0.0, 1.0); ufa[~mask] = 0.0
mdmap = np.nan_to_num(MD.copy()); mdmap[~mask] = 0.0
save(mdmap, "MD_dps_dMRI.nii.gz")
save(ufa, "C_mu_dps_dMRI.nii.gz")
save(mask.astype(np.float32), "dMRI_mask.nii.gz")
print("Done. Next: 02_register_dmri_to_T1.sh registers these to T1.")
