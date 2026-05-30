"""
02_build_conductivity_tensor.py — Build MD-dMRI conductivity tensor for SimNIBS

Inputs (all in T1 space, from 01_register_dMRI_to_T1.sh):
  registration/C_mu_T1.nii.gz       — μFA (DPS model), [0, 1)
  registration/MD_T1.nii.gz         — Mean diffusivity (μm²/ms, from dps.mat)
  registration/v1_T1.nii.gz         — Principal eigenvector [X,Y,Z,3], unit-normalised
  registration/dMRI_mask_T1.nii.gz  — Brain mask (from QTI fit)

Output:
  tensor_MD_dMRI.nii.gz  — Diffusion tensor [X,Y,Z,6] in FSL dtifit order
                            [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz] (upper triangle, row-major)
                            Pass to SimNIBS via fn_tensor_nifti= with anisotropy_type='vn'

Physics:
  The DPS model fits an axially symmetric (prolate) mean compartment tensor.
  Eigenvalues are defined by the μFA from Westin 2016 / Topgaard 2017:
    λ₁ = MD × (1 + 2·μFA / √(3 − 2·μFA²))   ← along fibre
    λ₂ = λ₃ = MD × (1 − μFA / √(3 − 2·μFA²)) ← perpendicular

  This is not an approximation: the formula IS the model. The DPS μFA describes
  intra-voxel fibre coherence, and the prolate (λ₂=λ₃) output is the
  mathematically correct representation of the axially symmetric compartment
  geometry assumed by the DPS framework. Splitting λ₂/λ₃ using DTI data would
  mix macroscopic DTI asymmetry back into the microscopic QTI model —
  conceptually incoherent at fibre crossings.

  Prolate tensor construction:
    D_ij = (λ₁ − λ₂)·v1_i·v1_j + λ₂·δ_ij

  SimNIBS then applies volume-normalised conductivity: σ = σ_WM · D / det(D)^(1/3)

Usage:
  ~/Applications/SimNIBS-4.6/bin/simnibs_python 02_build_conductivity_tensor.py
"""

import numpy as np
import nibabel as nib
import os

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

print(f"  C_mu: shape={C_mu.shape}, range=[{C_mu[mask].min():.4f}, {C_mu[mask].max():.4f}]")
print(f"  MD:   shape={MD.shape},   range=[{MD[mask].min():.6f}, {MD[mask].max():.6f}] μm²/ms")
print(f"  v1:   shape={v1.shape}")
print(f"  mask: {mask.sum():,} brain voxels")

# ── Clip C_mu to [0, 0.9999] ──────────────────────────────────────────────────
# dps.ufa is [0,1] but trilinear interpolation at mask edges can push values
# slightly outside. Clip to avoid denominator → 0 at C_mu = 1.
print("\nClipping C_mu to [0, 0.9999]...")
n_over = np.sum((C_mu > 1.0) & mask)
print(f"  C_mu > 1 inside mask: {n_over} voxels ({100*n_over/mask.sum():.2f}%) — clipped")
C_mu = np.clip(C_mu, 0.0, 0.9999)

# Zero outside mask
C_mu[~mask] = 0.0
MD[~mask]   = 0.0
v1[~mask]   = 0.0

# ── Compute eigenvalues from DPS/QTI formula ──────────────────────────────────
print("\nComputing eigenvalues from DPS μFA formula...")
denom = np.sqrt(np.maximum(3.0 - 2.0 * C_mu**2, 1e-12))  # guard div-by-0 at C_mu→1
lam1  = MD * (1.0 + 2.0 * C_mu / denom)   # along fibre  (μm²/ms)
lam2  = MD * (1.0 - C_mu / denom)          # perpendicular (μm²/ms)

# Sanity: MD must be recovered as (λ₁ + 2λ₂) / 3
MD_check = (lam1 + 2.0 * lam2) / 3.0
err = np.abs(MD_check - MD)[mask]
print(f"  MD recovery error (brain): max={err.max():.2e}  mean={err.mean():.2e} μm²/ms")

n_inv = np.sum((lam1 < lam2) & mask)
if n_inv > 0:
    print(f"  WARNING: {n_inv} voxels have λ₁ < λ₂ — check μFA clipping")

print(f"  λ₁ (brain): [{lam1[mask].min():.6f}, {lam1[mask].max():.6f}] μm²/ms")
print(f"  λ₂ (brain): [{lam2[mask].min():.6f}, {lam2[mask].max():.6f}] μm²/ms")

# ── Build prolate diffusion tensor ────────────────────────────────────────────
# D_ij = (λ₁−λ₂)·v1_i·v1_j + λ₂·δ_ij
print("\nBuilding prolate diffusion tensor...")
dl  = lam1 - lam2
v1x = v1[..., 0]
v1y = v1[..., 1]
v1z = v1[..., 2]

Dxx = dl * v1x * v1x + lam2
Dxy = dl * v1x * v1y
Dxz = dl * v1x * v1z
Dyy = dl * v1y * v1y + lam2
Dyz = dl * v1y * v1z
Dzz = dl * v1z * v1z + lam2

# ── Verify positive definiteness ──────────────────────────────────────────────
# For a prolate tensor: det(D) = λ₁ · λ₂²
det = lam1 * lam2**2
n_neg = np.sum((det <= 0) & mask)
if n_neg > 0:
    print(f"  WARNING: {n_neg} non-positive-definite voxels → falling back to isotropic MD")
    bad = (det <= 0) & mask
    Dxx[bad] = MD[bad]; Dxy[bad] = 0.0; Dxz[bad] = 0.0
    Dyy[bad] = MD[bad]; Dyz[bad] = 0.0; Dzz[bad] = MD[bad]
else:
    print(f"  All brain voxels positive definite.")
    print(f"  det(D) range (brain): [{det[mask].min():.2e}, {det[mask].max():.2e}]")

# ── Stack into [X,Y,Z,6] — FSL/SimNIBS dtifit --save_tensor order ─────────────
# Vol 0: T11=Dxx  Vol 1: T12=Dxy  Vol 2: T13=Dxz
# Vol 3: T22=Dyy  Vol 4: T23=Dyz  Vol 5: T33=Dzz
tensor = np.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], axis=-1).astype(np.float32)
print(f"\nTensor shape: {tensor.shape}")
print(f"Tensor range: [{tensor[mask].min():.5f}, {tensor[mask].max():.5f}] μm²/ms")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(WDIR, "tensor_MD_dMRI.nii.gz")
hdr = c_mu_img.header.copy()
hdr.set_data_shape(tensor.shape)
hdr.set_data_dtype(np.float32)
nib.save(nib.Nifti1Image(tensor, affine, hdr), out_path)
print(f"Saved: {out_path}")
print("\nNext: run 03_run_simulations.py")
