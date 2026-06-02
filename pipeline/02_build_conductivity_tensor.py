"""
02_build_conductivity_tensor.py — DPS two-tier conductivity tensor for SimNIBS

Conductivity model: DPS prolate with isotropic fallback
  Two-tier logic per voxel (signaniso field from DPS):

  TIER 1 — Prolate (signaniso=+1 AND valid v1, ~55% of brain):
    λ_axial  = MD · (1 + 2μFA/d)   along v1_DPS (fibre axis, large eigenvalue)
    λ_radial = MD · (1 − μFA/d)    perpendicular to v1 (small, λ₂=λ₃)
    D = (λ_axial−λ_radial)·v1⊗v1 + λ_radial·I
    Valid for all μFA ∈ [0,1): λ_radial ≥ 0 always. ✓
    Uses μFA (orientation-dispersion-invariant) rather than macroscopic FA
    → preserves microscopic anisotropy in regions with fibre dispersion.

  TIER 2 — All other brain voxels (~45% of brain): D = MD · I  (isotropic fallback)
    Covers: oblate (signaniso=−1), spherical (signaniso=0), v1_invalid.
    Rationale for excluding oblate voxels:
      The DPS eigenvector u = direction of LARGEST eigenvalue in ALL voxels
      (confirmed: AD > RD in 100% of oblate voxels from dps.mat).
      For oblate (disc-shaped) tensors, all equatorial directions are degenerate
      (equal eigenvalue within the disc plane). The DPS fitting algorithm returns
      one arbitrary equatorial direction as u — it is NOT the disc normal v3 (the
      unique small-eigenvalue direction needed to orient the oblate tensor).
      Without v3, two incorrect alternatives arise:
        (a) Oblate formula along u: assigns λ_normal (minimum) along the maximum-
            diffusivity direction → conductivity inverted 16× relative to true tensor.
        (b) Prolate formula along u: assigns max conductivity along an arbitrary
            in-plane axis → adds spurious directional anisotropy where none exists.
      Isotropic fallback (D = MD·I) preserves mean conductivity without asserting
      an incorrect orientation. Disclosed in methods.

  where d = √(3 − 2μFA²) throughout.

Derivation of prolate formula:
  For a cylindrically symmetric tensor (λ_‖ along v1, λ_⊥ perpendicular):
    MD = (λ_‖ + 2λ_⊥)/3
    μFA = |λ_‖ − λ_⊥| / √(λ_‖² + 2λ_⊥²)   [standard FA for cylindrical tensor]
  Solving for λ_‖ and λ_⊥ in terms of MD and μFA:
    Δ ≡ λ_‖ − λ_⊥  →  λ_‖ = MD + 2Δ/3,  λ_⊥ = MD − Δ/3
    μFA² = Δ²/(3MD² + 2Δ²/3)  →  t ≡ Δ/MD = +3μFA/d   (prolate: Δ>0)
    → λ_axial = MD(1+2μFA/d), λ_radial = MD(1−μFA/d)
  MD conservation: (λ_axial + 2λ_radial)/3 = MD exactly. ✓

Inputs (all in T1 space):
  registration/C_mu_T1.nii.gz       — μFA from DPS model (NOT covariance C_mu)
  registration/MD_T1.nii.gz         — Mean diffusivity from DPS model (μm²/ms)
  registration/v1_T1.nii.gz         — Principal eigenvector [X,Y,Z,3], unit-normalised
  registration/dMRI_mask_T1.nii.gz  — Brain mask
  registration/signaniso_T1.nii.gz  — DPS tensor shape: +1=prolate, -1=oblate, 0=spherical

Output:
  tensor_MD_dMRI.nii.gz  — Diffusion tensor [X,Y,Z,6] in FSL dtifit order
                            [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz] (upper triangle, row-major)
                            Pass to SimNIBS via s.fname_tensor with anisotropy_type='vn'

Usage:
  ~/Applications/SimNIBS-4.6/bin/simnibs_python scripts/02_build_conductivity_tensor.py
"""

import numpy as np
import nibabel as nib
import os

WDIR = "/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation"
RDIR = os.path.join(WDIR, "registration")

# ── Load inputs ───────────────────────────────────────────────────────────────
print("Loading registered maps...")
c_mu_img   = nib.load(os.path.join(RDIR, "C_mu_T1.nii.gz"))
md_img     = nib.load(os.path.join(RDIR, "MD_T1.nii.gz"))
v1_img     = nib.load(os.path.join(RDIR, "v1_T1.nii.gz"))
mask_img   = nib.load(os.path.join(RDIR, "dMRI_mask_T1.nii.gz"))
signa_img  = nib.load(os.path.join(RDIR, "signaniso_T1.nii.gz"))

C_mu   = c_mu_img.get_fdata().astype(np.float64)
MD     = md_img.get_fdata().astype(np.float64)
v1     = v1_img.get_fdata().astype(np.float64)    # (X, Y, Z, 3)
mask   = mask_img.get_fdata().astype(bool)
signa  = signa_img.get_fdata().astype(np.float64)  # +1, -1, 0

affine = c_mu_img.affine

print(f"  C_mu:    shape={C_mu.shape}, range=[{C_mu[mask].min():.4f}, {C_mu[mask].max():.4f}]")
print(f"  MD:      shape={MD.shape},   range=[{MD[mask].min():.4f}, {MD[mask].max():.4f}] μm²/ms")
print(f"  v1:      shape={v1.shape}")
print(f"  mask:    {mask.sum():,} brain voxels")
print(f"  signaniso distribution (brain mask):")

is_prolate   = (signa > 0.5)  & mask
is_oblate    = (signa < -0.5) & mask
is_spherical = (np.abs(signa) < 0.5) & mask
print(f"    Prolate  (+1): {is_prolate.sum():,}  ({100*is_prolate.sum()/mask.sum():.1f}%)")
print(f"    Oblate   (-1): {is_oblate.sum():,}  ({100*is_oblate.sum()/mask.sum():.1f}%)")
print(f"    Spherical (0): {is_spherical.sum():,}  ({100*is_spherical.sum()/mask.sum():.1f}%)")

# ── Clip C_mu — numerical guard ───────────────────────────────────────────────
print(f"\nClipping C_mu to [0, 0.9999] (numerical guard)...")
n_over = np.sum((C_mu > 1.0) & mask)
print(f"  C_mu > 1.0 inside mask: {n_over:,} voxels clipped")
C_mu = np.clip(C_mu, 0.0, 0.9999)
C_mu[~mask] = 0.0
MD[~mask]   = 0.0
v1[~mask]   = 0.0

# ── v1 validity check (must come after zeroing outside-mask) ──────────────────
# After vecreg, v1_T1 norms are bimodal: interior voxels ≈1, boundary voxels ≈0.
# Cause: vecreg trilinearly interpolates between valid unit vectors and [0,0,0]
# (the zero-padding outside the dMRI mask). At the 2.5mm→1mm boundary, a T1
# voxel can be entirely covered by the zero-padded region → v1≈[0,0,0].
# The norm distribution has a perfect gap in (0.1, 0.990) — zero voxels there.
# Threshold 0.5 sits in the middle of that gap; any value in (0.1, 0.990) is identical.
v1_norm    = np.linalg.norm(v1, axis=-1)   # (X, Y, Z)
v1_invalid = (v1_norm < 0.5) & mask
n_v1_inv   = v1_invalid.sum()
print(f"\nv1 validity (after vecreg):")
print(f"  |v1| ≈ 1 (valid): {mask.sum()-n_v1_inv:,}  ({100*(mask.sum()-n_v1_inv)/mask.sum():.2f}%)")
print(f"  |v1| ≈ 0 (vecreg boundary): {n_v1_inv:,}  ({100*n_v1_inv/mask.sum():.2f}%) → isotropic fallback")

# ── Common denominator d = √(3 − 2μFA²) ───────────────────────────────────────
denom = np.sqrt(np.maximum(3.0 - 2.0 * C_mu**2, 1e-12))

# ── Prolate eigenvalues ────────────────────────────────────────────────────────
# λ_axial  = large eigenvalue, along v1 (fibre axis)
# λ_radial = small eigenvalue, perpendicular to v1
# Applied only to Tier 1 (signaniso=+1 AND valid v1).
lam_axial  = MD * (1.0 + 2.0 * C_mu / denom)
lam_radial = MD * (1.0 - C_mu / denom)

# ── Two-tier tensor assembly ───────────────────────────────────────────────────
# TIER 1 (prolate + valid v1): D = (λ_axial − λ_radial)·v1⊗v1 + λ_radial·I
# TIER 2 (all else):           D = MD · I
#   Tier 2 covers: oblate (signaniso=−1), spherical (signaniso=0), v1_invalid.
#   For oblate voxels: u = max equatorial eigenvector (arbitrary in degenerate plane);
#   disc normal v3 (unique small-eigenvalue direction) not available in dps.mat.
#   Without v3, neither the oblate formula nor a prolate approximation is correct.
dl       = np.zeros_like(MD)
lam_perp = np.zeros_like(MD)

tier1 = is_prolate & (~v1_invalid)
tier2 = ~tier1 & mask   # oblate + spherical + v1_invalid (everything else in brain)

dl[tier1]       = lam_axial[tier1] - lam_radial[tier1]   # > 0
lam_perp[tier1] = lam_radial[tier1]
lam_perp[tier2] = MD[tier2]   # dl remains 0 → D = MD·I

n_t1 = tier1.sum()
n_t2 = tier2.sum()
print(f"\nTier assignment:")
print(f"  Tier 1 — prolate (μFA+v1 used): {n_t1:,}  ({100*n_t1/mask.sum():.1f}% of brain)")
print(f"  Tier 2 — isotropic fallback:    {n_t2:,}  ({100*n_t2/mask.sum():.1f}% of brain)")
print(f"    breakdown: oblate={is_oblate.sum():,}, "
      f"spherical={is_spherical.sum():,}, "
      f"v1-invalid={n_v1_inv:,}")

# ── MD conservation sanity check ──────────────────────────────────────────────
err_p = np.abs((lam_axial[tier1] + 2*lam_radial[tier1]) / 3 - MD[tier1])
print(f"\nMD conservation check (Tier 1 prolate voxels):")
print(f"  max error = {err_p.max():.2e} μm²/ms ✓")

# ── Build tensor ──────────────────────────────────────────────────────────────
print("\nBuilding conductivity tensor (two-tier DPS model)...")

v1x = v1[..., 0];  v1y = v1[..., 1];  v1z = v1[..., 2]
v1x2 = v1x*v1x;  v1y2 = v1y*v1y;  v1z2 = v1z*v1z
v1xy = v1x*v1y;  v1xz = v1x*v1z;  v1yz = v1y*v1z

# Assemble 6-component tensor in FSL dtifit order:
# [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
Dxx = dl * v1x2 + lam_perp
Dxy = dl * v1xy
Dxz = dl * v1xz
Dyy = dl * v1y2 + lam_perp
Dyz = dl * v1yz
Dzz = dl * v1z2 + lam_perp

tensor = np.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], axis=-1).astype(np.float32)
print(f"  Tensor shape: {tensor.shape}")
print(f"  Tensor range (brain): [{tensor[mask].min():.5f}, {tensor[mask].max():.5f}] μm²/ms")

# ── Positive definiteness check ───────────────────────────────────────────────
# Tier 1 prolate: det = λ_axial · λ_radial²  (≥ 0 since λ_radial ≥ 0 for μFA < 1)
# Tier 2:         det = MD³ > 0
det_prolate = lam_axial[tier1] * lam_radial[tier1]**2
n_neg_det = np.sum(det_prolate < 0)
print(f"\nPositive definiteness check:")
print(f"  Tier 1 voxels with det ≤ 0: {n_neg_det} (should be 0)")

# ── Eigenvalue ratio report (for SimNIBS capping assessment) ──────────────────
ratio_p  = lam_axial[tier1] / np.maximum(lam_radial[tier1], 1e-12)
thresh8  = 4.0**1.5   # 8.0:1 at aniso_maxcond=4 (VN normalization)
n_capped = np.sum(ratio_p > thresh8)
print(f"\nEigenvalue ratio (λ_axial/λ_radial) for Tier 1 prolate voxels:")
print(f"  median={np.median(ratio_p):.2f}  "
      f"p95={np.percentile(ratio_p, 95):.2f}  "
      f"max={ratio_p.max():.2f}")
print(f"  ratio > 8.0 (capped at aniso_maxcond=4): "
      f"{n_capped:,}  ({100*n_capped/n_t1:.1f}% of Tier 1 voxels)")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(WDIR, "tensor_MD_dMRI.nii.gz")
hdr = c_mu_img.header.copy()
hdr.set_data_shape(tensor.shape)
hdr.set_data_dtype(np.float32)
nib.save(nib.Nifti1Image(tensor, affine, hdr), out_path)
print(f"\nSaved: {out_path}")
print("\nNext: run 03_run_simulations.py (MD-dMRI simulation only — tensor changed)")
