"""
02_build_conductivity_tensor.py — DPS oblate-corrected conductivity tensor for SimNIBS

Conductivity model: DPS oblate-corrected with isotropic fallback
  Three-tier logic per voxel (signaniso field from DPS):

  TIER 1 — Prolate (signaniso=+1, ~55% of brain):
    λ_axial  = MD · (1 + 2μFA/d)   along v1_DPS (fibre axis, large eigenvalue)
    λ_radial = MD · (1 − μFA/d)    perpendicular to v1 (small, λ₂=λ₃)
    D = (λ_axial−λ_radial)·v1⊗v1 + λ_radial·I
    Valid for all μFA ∈ [0,1): λ_radial ≥ 0 always. ✓

  TIER 2 — Oblate valid (signaniso=−1 AND μFA ≤ 1/√2 ≈ 0.707, ~28% of brain):
    λ_normal    = MD · (1 − 2μFA/d)  along v1_DPS (disc normal, small eigenvalue)
    λ_equatorial = MD · (1 + μFA/d)   perpendicular to v1 (large, λ₂=λ₃)
    D = (λ_normal−λ_equatorial)·v1⊗v1 + λ_equatorial·I
    Physical limit: λ_normal ≥ 0 ⟺ μFA ≤ 1/√2. ✓

  TIER 3 — Oblate invalid or spherical (μFA > 1/√2 OR signaniso=0, ~17% of brain):
    D = MD · I  (isotropic fallback)
    Reason: DPS fit outside physically valid range for oblate cylindrical geometry.
    The formula would require λ_normal < 0 (non-physical). Isotropic is the
    least-biased fallback; disclosed in methods.

  where d = √(3 − 2μFA²) throughout.

Derivation of oblate formula:
  For a cylindrically symmetric tensor (λ_‖ along v1, λ_⊥ perpendicular):
    MD = (λ_‖ + 2λ_⊥)/3
    μFA = |λ_‖ − λ_⊥| / √(λ_‖² + 2λ_⊥²)   [standard FA for cylindrical tensor]
  Solving for λ_‖ and λ_⊥ in terms of MD and μFA:
    Δ ≡ λ_‖ − λ_⊥  →  λ_‖ = MD + 2Δ/3,  λ_⊥ = MD − Δ/3
    μFA² = Δ²/(3MD² + 2Δ²/3)  →  t ≡ Δ/MD = ±3μFA/d
  Prolate (Δ>0): t = +3μFA/d  →  λ_‖ = MD(1+2μFA/d), λ_⊥ = MD(1−μFA/d)
  Oblate  (Δ<0): t = −3μFA/d  →  λ_‖ = MD(1−2μFA/d), λ_⊥ = MD(1+μFA/d)
  MD conservation: (λ_‖ + 2λ_⊥)/3 = MD exactly in both cases. ✓

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

# ── Physical limit for oblate formula ─────────────────────────────────────────
# λ_normal = MD(1-2μFA/d) ≥ 0  ⟺  μFA ≤ 1/√2 ≈ 0.7071
OBLATE_UFA_MAX = 1.0 / np.sqrt(2)   # 0.70711...

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

is_prolate = (signa > 0.5) & mask
is_oblate  = (signa < -0.5) & mask
is_spherical = (np.abs(signa) < 0.5) & mask
print(f"    Prolate  (+1): {is_prolate.sum():,}  ({100*is_prolate.sum()/mask.sum():.1f}%)")
print(f"    Oblate   (-1): {is_oblate.sum():,}  ({100*is_oblate.sum()/mask.sum():.1f}%)")
print(f"    Spherical (0): {is_spherical.sum():,}  ({100*is_spherical.sum()/mask.sum():.1f}%)")

# ── Clip C_mu — numerical guard; oblate limit applied per-voxel below ─────────
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
# These 64,693 voxels (3.13%) have no valid eigenvector — force to isotropic fallback.
v1_norm = np.linalg.norm(v1, axis=-1)   # (X, Y, Z)
v1_invalid = (v1_norm < 0.5) & mask     # bimodal: 0 or 1 — threshold 0.5 cleanly separates
n_v1_inv = v1_invalid.sum()
print(f"\nv1 validity (after vecreg):")
print(f"  |v1| ≈ 1 (valid, unit-normalised): {mask.sum()-n_v1_inv:,}  ({100*(mask.sum()-n_v1_inv)/mask.sum():.2f}%)")
print(f"  |v1| ≈ 0 (vecreg boundary, no eigenvector): {n_v1_inv:,}  ({100*n_v1_inv/mask.sum():.2f}%) → isotropic fallback")

# ── Common denominator d = √(3 − 2μFA²) ───────────────────────────────────────
denom = np.sqrt(np.maximum(3.0 - 2.0 * C_mu**2, 1e-12))

# ── TIER 1: Prolate eigenvalues (λ_axial=large, λ_radial=small) ───────────────
lam_axial  = MD * (1.0 + 2.0 * C_mu / denom)   # along v1 (fibre axis)
lam_radial = MD * (1.0 - C_mu / denom)           # perpendicular to v1

# ── TIER 2: Oblate eigenvalues (λ_normal=small, λ_equatorial=large) ───────────
# Physical validity: only where C_mu ≤ 1/√2 (otherwise λ_normal < 0)
lam_normal    = MD * (1.0 - 2.0 * C_mu / denom)  # along v1 (disc normal, SMALL)
lam_equatorial = MD * (1.0 + C_mu / denom)         # perpendicular to v1 (LARGE)

# Oblate validity mask (inside brain + signaniso=-1 + μFA ≤ 1/√2)
oblate_valid   = is_oblate & (C_mu <= OBLATE_UFA_MAX)
oblate_invalid = is_oblate & (C_mu >  OBLATE_UFA_MAX)

n_ob_valid   = oblate_valid.sum()
n_ob_invalid = oblate_invalid.sum()
print(f"\nOblate tier breakdown:")
print(f"  Valid   (μFA ≤ {OBLATE_UFA_MAX:.4f}): {n_ob_valid:,}  ({100*n_ob_valid/is_oblate.sum():.1f}% of oblate, "
      f"{100*n_ob_valid/mask.sum():.1f}% of brain)")
print(f"  Invalid (μFA > {OBLATE_UFA_MAX:.4f}): {n_ob_invalid:,}  ({100*n_ob_invalid/is_oblate.sum():.1f}% of oblate, "
      f"{100*n_ob_invalid/mask.sum():.1f}% of brain) → isotropic fallback")
print(f"  Spherical/invalid total (before v1 check): {(is_spherical | oblate_invalid).sum():,}  "
      f"({100*(is_spherical | oblate_invalid).sum()/mask.sum():.1f}% of brain)")

# ── MD conservation sanity check — only for voxels with valid v1 ──────────────
prolate_ok      = is_prolate & (~v1_invalid)
oblate_valid_ok = oblate_valid & (~v1_invalid)
err_p = np.abs((lam_axial[prolate_ok] + 2*lam_radial[prolate_ok])/3 - MD[prolate_ok])
err_o = np.abs((lam_normal[oblate_valid_ok] + 2*lam_equatorial[oblate_valid_ok])/3 - MD[oblate_valid_ok])
print(f"\nMD conservation check:")
print(f"  Prolate voxels:       max error = {err_p.max():.2e} μm²/ms ✓")
print(f"  Oblate valid voxels:  max error = {err_o.max():.2e} μm²/ms ✓")

# ── Sanity: verify λ_normal ≥ 0 for valid oblate voxels ──────────────────────
n_neg_valid = np.sum((lam_normal < 0) & oblate_valid)
n_neg_invalid = np.sum((lam_normal < 0) & oblate_invalid)
print(f"  λ_normal < 0 in valid oblate: {n_neg_valid} (should be 0) ✓")
print(f"  λ_normal < 0 in invalid oblate: {n_neg_invalid} (handled by fallback)")

# ── Build eigenvector components ──────────────────────────────────────────────
v1x = v1[..., 0]
v1y = v1[..., 1]
v1z = v1[..., 2]

print("\nBuilding conductivity tensor (three-tier DPS model)...")

# Outer product v1⊗v1 components
v1x2 = v1x * v1x
v1y2 = v1y * v1y
v1z2 = v1z * v1z
v1xy = v1x * v1y
v1xz = v1x * v1z
v1yz = v1y * v1z

# ── Tier 1: Prolate — D = (λ_axial − λ_radial)·v1⊗v1 + λ_radial·I ──────────
# ── Tier 2: Oblate valid — D = (λ_normal − λ_equatorial)·v1⊗v1 + λ_equatorial·I
# Both share the same outer-product structure; only the scalar coefficients differ.
# We build a unified coefficient array:
#   dl  = λ_‖ − λ_⊥  (the "anisotropic" factor applied to v1⊗v1)
#   lam_perp = λ_⊥   (the "isotropic" background added to diagonal)
dl       = np.zeros_like(MD)
lam_perp = np.zeros_like(MD)

# Prolate
dl[is_prolate]       = lam_axial[is_prolate] - lam_radial[is_prolate]   # > 0
lam_perp[is_prolate] = lam_radial[is_prolate]

# Oblate valid: λ_‖ = λ_normal (small, along v1), λ_⊥ = λ_equatorial (large)
dl[oblate_valid]       = lam_normal[oblate_valid] - lam_equatorial[oblate_valid]  # < 0
lam_perp[oblate_valid] = lam_equatorial[oblate_valid]

# Tier 3: Oblate invalid + spherical + v1_invalid → isotropic: dl=0, lam_perp=MD
# v1_invalid must override any prolate/oblate assignment above (comes last).
tier3 = oblate_invalid | is_spherical | v1_invalid
dl[tier3]       = 0.0
lam_perp[tier3] = MD[tier3]

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

# ── Verify positive definiteness ──────────────────────────────────────────────
# Prolate: det = λ_axial · λ_radial²  (always ≥ 0 since both ≥ 0)
# Oblate valid: det = λ_normal · λ_equatorial²  (always ≥ 0)
# Tier 3: det = MD³ > 0
det_prolate = lam_axial[prolate_ok] * lam_radial[prolate_ok]**2
det_oblate  = lam_normal[oblate_valid_ok] * lam_equatorial[oblate_valid_ok]**2
n_neg_det_p = np.sum(det_prolate < 0)
n_neg_det_o = np.sum(det_oblate < 0)
print(f"\nPositive definiteness check:")
print(f"  Prolate voxels with det ≤ 0:        {n_neg_det_p} (should be 0)")
print(f"  Oblate valid voxels with det ≤ 0:   {n_neg_det_o} (should be 0)")

# ── Eigenvalue ratio report (for SimNIBS clipping assessment) ─────────────────
# Effective max/min ratio per tier (anisotropic voxels with valid v1 only)
ratio_p = lam_axial[prolate_ok] / np.maximum(lam_radial[prolate_ok], 1e-12)
ratio_o = lam_equatorial[oblate_valid_ok] / np.maximum(lam_normal[oblate_valid_ok], 1e-12)
thresh8 = 4**1.5   # 8.0:1 at aniso_maxcond=4

print(f"\nEigenvalue ratio (λ_max/λ_min):")
print(f"  Prolate (valid v1):   median={np.median(ratio_p):.2f}  "
      f"p95={np.percentile(ratio_p,95):.2f}  "
      f"clipped@8.0: {np.sum(ratio_p>thresh8):,} ({100*np.sum(ratio_p>thresh8)/prolate_ok.sum():.1f}%)")
print(f"  Oblate valid (v1 ok): median={np.median(ratio_o):.2f}  "
      f"p95={np.percentile(ratio_o,95):.2f}  "
      f"clipped@8.0: {np.sum(ratio_o>thresh8):,} ({100*np.sum(ratio_o>thresh8)/oblate_valid_ok.sum():.1f}%)")
print(f"  Tier 3 (isotropic):   ratio=1.0  {tier3.sum():,} voxels ({100*tier3.sum()/mask.sum():.1f}% of brain)"
      f"  [components (union): oblate-invalid={oblate_invalid.sum():,}, "
      f"spherical={is_spherical.sum():,}, v1-invalid={n_v1_inv:,}]")
print(f"  Total clipped@aniso_maxcond=4: "
      f"{np.sum(ratio_p>thresh8)+np.sum(ratio_o>thresh8):,} / "
      f"{(prolate_ok|oblate_valid_ok).sum():,} anisotropic voxels "
      f"({100*(np.sum(ratio_p>thresh8)+np.sum(ratio_o>thresh8))/(prolate_ok|oblate_valid_ok).sum():.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(WDIR, "tensor_MD_dMRI.nii.gz")
hdr = c_mu_img.header.copy()
hdr.set_data_shape(tensor.shape)
hdr.set_data_dtype(np.float32)
nib.save(nib.Nifti1Image(tensor, affine, hdr), out_path)
print(f"\nSaved: {out_path}")
print("\nNext: run 03_run_simulations.py")
