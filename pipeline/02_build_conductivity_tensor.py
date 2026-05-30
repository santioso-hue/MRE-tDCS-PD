"""
02_build_conductivity_tensor.py — Build MD-dMRI conductivity tensor for SimNIBS

Inputs (all in T1 space, from 01_register_dMRI_to_T1.sh):
  registration/C_mu_T1.nii.gz   — μFA (microscopic fractional anisotropy), [0, 1)
  registration/MD_T1.nii.gz     — Mean diffusivity (μm²/ms, from dps.mat)
  registration/v1_T1.nii.gz     — Principal eigenvector [X,Y,Z,3], unit-normalised
  registration/dMRI_mask_T1.nii.gz — Brain mask (from QTI fit)

Output:
  tensor_MD_dMRI.nii.gz  — Diffusion tensor [X,Y,Z,6] in FSL dtifit order [Dxx,Dxy,Dxz,Dyy,Dyz,Dzz]
                            Pass to SimNIBS via fn_tensor_nifti= with anisotropy_type='vn'

Physics:
  From QTI theory (Westin 2016), the per-compartment tensor shape gives eigenvalues:
    λ1 = MD × (1 + 2·C_mu / √(3 − 2·C_mu²))   ← along fibre
    λ2 = λ3 = MD × (1 − C_mu / √(3 − 2·C_mu²)) ← perpendicular

  Prolate tensor approximation (only v1 available from dps.mat):
    D_ij = (λ1−λ2)·v1_i·v1_j + λ2·δ_ij

  SimNIBS then applies volume-normalised conductivity: σ = σ_WM · D / det(D)^(1/3)

Usage:
  ~/Applications/SimNIBS-4.6/bin/simnibs_python 02_build_conductivity_tensor.py
"""

import numpy as np
import nibabel as nib
import os
import sys

WDIR = "/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation"
RDIR = os.path.join(WDIR, "registration")

# ── Load inputs ───────────────────────────────────────────────────────────────
print("Loading registered maps...")
c_mu_img = nib.load(os.path.join(RDIR, "C_mu_T1.nii.gz"))
md_img   = nib.load(os.path.join(RDIR, "MD_T1.nii.gz"))
v1_img   = nib.load(os.path.join(RDIR, "v1_T1.nii.gz"))
mask_img = nib.load(os.path.join(RDIR, "dMRI_mask_T1.nii.gz"))

C_mu = c_mu_img.get_fdata().astype(np.float64)
MD   = md_img.get_fdata().astype(np.float64)
v1   = v1_img.get_fdata().astype(np.float64)   # shape (X, Y, Z, 3)
mask = mask_img.get_fdata().astype(bool)

affine = c_mu_img.affine

print(f"  C_mu: shape={C_mu.shape}, range=[{C_mu.min():.4f}, {C_mu.max():.4f}]")
print(f"  MD:   shape={MD.shape},   range=[{MD.min():.6f}, {MD.max():.6f}] μm²/ms")
print(f"  v1:   shape={v1.shape}")
print(f"  mask: {mask.sum()} brain voxels")

# ── Apply QTI brain mask and clip C_mu ───────────────────────────────────────
# C_mu > 1 are numerical artefacts (16.5% of brain mask voxels in raw data).
# After registration + interpolation, values can drift outside [0,1] further.
# Clip to [0, 0.9999] — values near 1 cause denom → 0 (singular tensor).
print("\nApplying mask and clipping C_mu...")
n_bad = np.sum((C_mu > 1.0) & mask)
print(f"  Voxels with C_mu > 1 inside mask: {n_bad} ({100*n_bad/mask.sum():.1f}%)")

C_mu = np.clip(C_mu, 0.0, 0.9999)

# Zero out outside mask
C_mu[~mask] = 0.0
MD[~mask]   = 0.0

# ── Compute eigenvalues ────────────────────────────────────────────────────────
print("\nComputing eigenvalues from μFA formula...")
denom = np.sqrt(np.maximum(3.0 - 2.0 * C_mu**2, 1e-12))  # guard against div-by-0
lam1  = MD * (1.0 + 2.0 * C_mu / denom)   # along fibre
lam2  = MD * (1.0 - C_mu / denom)         # perpendicular (= lam3)

# Sanity check: MD should be recovered as (lam1 + 2*lam2) / 3
MD_check = (lam1 + 2.0 * lam2) / 3.0
err = np.abs(MD_check - MD)[mask]
print(f"  MD recovery error (brain voxels): max={err.max():.2e}, mean={err.mean():.2e}")

# Sanity check: C_mu should be recoverable
# FA of prolate tensor = sqrt(2) * |lam1 - lam2| / sqrt(lam1^2 + 2*lam2^2) * something
# Just verify lam1 >= lam2 everywhere in mask
neg = np.sum((lam1 < lam2) & mask)
if neg > 0:
    print(f"  WARNING: {neg} voxels have lam1 < lam2 — check C_mu clipping")

print(f"  lam1 range (brain): [{lam1[mask].min():.6f}, {lam1[mask].max():.6f}]")
print(f"  lam2 range (brain): [{lam2[mask].min():.6f}, {lam2[mask].max():.6f}]")

# ── Build diffusion tensor (prolate approximation) ─────────────────────────────
# D_ij = (λ1−λ2)·v1_i·v1_j + λ2·δ_ij
print("\nBuilding diffusion tensor...")

dl = lam1 - lam2                   # (X, Y, Z)
v1x = v1[..., 0]                   # (X, Y, Z)
v1y = v1[..., 1]
v1z = v1[..., 2]

# Re-zero eigenvectors outside mask (vecreg may bleed into background)
v1x[~mask] = 0.0
v1y[~mask] = 0.0
v1z[~mask] = 0.0

Dxx = dl * v1x * v1x + lam2
Dxy = dl * v1x * v1y
Dyy = dl * v1y * v1y + lam2
Dxz = dl * v1x * v1z
Dyz = dl * v1y * v1z
Dzz = dl * v1z * v1z + lam2

# ── Verify positive definiteness ───────────────────────────────────────────────
# det(D) = lam1 * lam2^2 for prolate tensor — must be > 0
det = lam1 * lam2**2
neg_det = np.sum((det <= 0) & mask)
if neg_det > 0:
    print(f"  WARNING: {neg_det} voxels have det(D) <= 0 — setting to isotropic MD")
    # Fall back to isotropic tensor at those voxels
    bad = (det <= 0) & mask
    Dxx[bad] = MD[bad]; Dxy[bad] = 0; Dyy[bad] = MD[bad]
    Dxz[bad] = 0;       Dyz[bad] = 0; Dzz[bad] = MD[bad]
else:
    print(f"  All brain voxels positive definite. det range: [{det[mask].min():.2e}, {det[mask].max():.2e}]")

# ── Stack into [X, Y, Z, 6] in FSL dtifit order: [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz] ──
# SimNIBS reads fn_tensor_nifti in the same format as FSL dtifit --save_tensor output:
#   Vol 0: T11=Dxx,  Vol 1: T12=Dxy,  Vol 2: T13=Dxz,
#   Vol 3: T22=Dyy,  Vol 4: T23=Dyz,  Vol 5: T33=Dzz
# (upper triangle, row by row — verified from dwi2cond.prepro.source.sh)
tensor = np.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], axis=-1).astype(np.float32)
print(f"\nTensor shape: {tensor.shape}  (FSL/SimNIBS order: Dxx, Dxy, Dxz, Dyy, Dyz, Dzz)")

# ── Save ────────────────────────────────────────────────────────────────────────
out_path = os.path.join(WDIR, "tensor_MD_dMRI.nii.gz")
hdr = c_mu_img.header.copy()
hdr.set_data_shape(tensor.shape)
hdr.set_data_dtype(np.float32)
nib.save(nib.Nifti1Image(tensor, affine, hdr), out_path)
print(f"Saved: {out_path}")
print("\nNext: run 03_run_simulations.py")
