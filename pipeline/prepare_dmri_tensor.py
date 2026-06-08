"""
prepare_dmri_tensor.py — Reconstruct the QTI mean diffusion tensor <D> and supporting maps from
dps.mat, in dMRI space, ready for registration to T1. Called by 02_register_dmri_to_T1.sh.

Outputs (registration/, dMRI space):
  tensor_triaxial_dMRI.nii.gz  6-comp <D> [Dxx,Dxy,Dxz,Dyy,Dyz,Dzz], um2/ms (vecreg -> orientation frame)
  lam1/lam2/lam3_dMRI.nii.gz   eigenvalues l1>=l2>=l3 (scalar; trilinear -> magnitude, preserves anisotropy)
  v1_dMRI.nii.gz               principal eigenvector dps.u (vecreg -> T1; anchors the tensor's principal axis)
  MD_dps_dMRI.nii.gz           mean diffusivity (QA + post-hoc MRE microstructure comparison)
  C_mu_dps_dMRI.nii.gz         microscopic FA from the DPS model (post-hoc MRE comparison only; see note)
  dMRI_mask.nii.gz             QTI brain mask

<D> is reconstructed from dps['mdxx'..'mdyz'] (SI m2/s, x1e9 -> um2/ms). trace(<D>)/3 == MD and the
principal eigenvector == dps.u are asserted before writing. The conductivity model is sigma proportional
to <D> via SimNIBS 'vn' (built in 03_build_conductivity_tensor.py); only the eigenvalue RATIOS and
orientation of <D> survive that mapping.

microscopic FA (C_mu) is deliberately NOT used by the conductivity model: it is orientation-invariant
and has no macroscopic eigenframe. It is written only for the post-hoc MRE microstructure comparison.
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


print("Loading dps.mat...")
dps = scipy.io.loadmat(os.path.join(FDIR, "dps.mat"))["dps"]
mask = dps["mask"][0, 0].astype(bool)
assert ref.shape == mask.shape, f"shape mismatch: C_mu {ref.shape} vs dps {mask.shape}"

# --- mean tensor <D> (SI m2/s -> um2/ms) ---
SI = 1.0e9
g = lambda f: np.real(dps[f][0, 0]).astype(np.float64) * SI  # noqa: E731
mdxx, mdyy, mdzz = g("mdxx"), g("mdyy"), g("mdzz")
mdxy, mdxz, mdyz = g("mdxy"), g("mdxz"), g("mdyz")
MD = np.real(dps["MD"][0, 0]).astype(np.float64)             # already um2/ms

D = np.zeros(mask.shape + (3, 3))
D[..., 0, 0] = mdxx; D[..., 1, 1] = mdyy; D[..., 2, 2] = mdzz
D[..., 0, 1] = D[..., 1, 0] = mdxy
D[..., 0, 2] = D[..., 2, 0] = mdxz
D[..., 1, 2] = D[..., 2, 1] = mdyz
ev = np.linalg.eigvalsh(D[mask])                            # ascending l3<=l2<=l1
l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]
trace_md = (mdxx + mdyy + mdzz) / 3.0
print(f"  trace/3 vs MD max|err|={np.max(np.abs(trace_md[mask] - MD[mask])):.4f} um2/ms; "
      f"PD {100 * np.mean(l3 > 0):.1f}%; median lambda {np.median(l1):.3f}/{np.median(l2):.3f}/{np.median(l3):.3f}")
assert np.max(np.abs(trace_md[mask] - MD[mask])) < 1e-2, "trace/3 != MD: unit-scale or diagonal bug"
assert np.mean(l3 > 0) > 0.99, "<D> not positive-definite in >1% of brain: assembly bug"

lam1 = np.zeros(mask.shape); lam2 = np.zeros(mask.shape); lam3 = np.zeros(mask.shape)
lam1[mask] = l1; lam2[mask] = l2; lam3[mask] = l3
for a in (mdxx, mdyy, mdzz, mdxy, mdxz, mdyz):
    a[~mask] = 0.0
tensor = np.stack([mdxx, mdxy, mdxz, mdyy, mdyz, mdzz], axis=-1)  # FSL dtifit order
save(tensor, "tensor_triaxial_dMRI.nii.gz")
save(lam1, "lam1_dMRI.nii.gz"); save(lam2, "lam2_dMRI.nii.gz"); save(lam3, "lam3_dMRI.nii.gz")

# --- principal eigenvector v1 = dps.u (real, unit, masked) ---
u = np.real(dps["u"][0, 0]).astype(np.float64)
u[~mask] = 0.0
n = np.linalg.norm(u, axis=-1, keepdims=True)
save(u / np.where(n > 0.1, n, 1.0), "v1_dMRI.nii.gz")

# --- QA / post-hoc MRE-comparison maps ---
ufa = np.clip(np.nan_to_num(np.real(dps["ufa"][0, 0]).astype(np.float64)), 0.0, 1.0); ufa[~mask] = 0.0
mdmap = np.nan_to_num(MD.copy()); mdmap[~mask] = 0.0
save(mdmap, "MD_dps_dMRI.nii.gz")
save(ufa, "C_mu_dps_dMRI.nii.gz")
save(mask.astype(np.float32), "dMRI_mask.nii.gz")
print("Done. Next: 02_register_dmri_to_T1.sh registers these to T1.")
