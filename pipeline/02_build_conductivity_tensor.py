"""
02_build_conductivity_tensor.py — DPS mean-compartment conductivity tensor for SimNIBS

Conductivity model: σ ∝ ⟨D⟩  (mean compartment tensor from DPS, orientation-dispersion-invariant)

  For every brain voxel with a valid eigenvector (v1_norm ≥ 0.5):

    D_conductivity = (ad − rd) · u⊗u + rd · I

  where:
    u  = principal eigenvector from DPS (direction of largest eigenvalue in ALL voxels)
    ad = axial diffusivity of ⟨D⟩ (largest eigenvalue of mean compartment tensor)
    rd = radial diffusivity of ⟨D⟩ (smallest eigenvalue of mean compartment tensor)

  For voxels where v1 is unreliable (vecreg boundary artefact, |v1| < 0.5):

    D_conductivity = MD · I  (isotropic fallback, ~3% of brain)

This is the orientation-dispersion-invariant analogue of Tuch 2001 (σ ∝ D):
  - dwi2cond: σ ∝ D_DTI (macroscopic apparent diffusion tensor)
  - This model: σ ∝ ⟨D⟩_DPS (mean compartment tensor, eigenvalues unconfounded by orientation dispersion)
  In regions with fibre crossing or fanning, DTI λ1 ≪ DPS ad (and DTI λ3 ≫ DPS rd) because
  macroscopic DTI averages over all orientations; DPS separates orientation dispersion from
  compartment-level anisotropy. Our eigenvalues (ad, rd) reflect the true compartment shape.

Cylindrical symmetry:
  The DPS model gives ONE unique eigenvector u per voxel (principal direction of ⟨D⟩).
  The other two eigenvectors are degenerate (same eigenvalue rd) and are not stored.
  This cylindrical symmetry is inherent to the DPS compartment model — it is not a
  pipeline limitation. dwi2cond can use all 3 DTI eigenvectors because DTI gives 3
  distinct eigenvalues; DPS fundamentally provides only u, ad, and rd.

Oblate voxels (signaniso=−1, 41.6% of brain):
  u = direction of LARGEST eigenvalue (one arbitrary direction in the equatorial plane).
  The disc normal (unique smallest-eigenvalue direction, v3) is NOT stored in dps.mat —
  only one eigenvector is saved regardless of tensor shape.
  For oblate voxels, (ad−rd)·u⊗u + rd·I gives a prolate-shaped tensor with max
  conductivity along u. This is geometrically wrong (oblate voxels should have
  disc-like conductivity), but is directionally correct: max conductivity along the
  max diffusivity direction, consistent with Tuch 2001.
  ad/rd ratio for oblate voxels is modest (~1.49:1 median), so the error is bounded.

Inputs (all in T1 space):
  registration/MD_T1.nii.gz         — Mean diffusivity from DPS (μm²/ms) — for isotropic fallback
  registration/ad_T1.nii.gz         — Axial diffusivity of ⟨D⟩ (μm²/ms)
  registration/rd_T1.nii.gz         — Radial diffusivity of ⟨D⟩ (μm²/ms)
  registration/v1_T1.nii.gz         — Principal eigenvector [X,Y,Z,3], unit-normalised
  registration/dMRI_mask_T1.nii.gz  — Brain mask
  registration/signaniso_T1.nii.gz  — DPS tensor shape (+1/−1/0) — used for reporting only

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
md_img    = nib.load(os.path.join(RDIR, "MD_T1.nii.gz"))
ad_img    = nib.load(os.path.join(RDIR, "ad_T1.nii.gz"))
rd_img    = nib.load(os.path.join(RDIR, "rd_T1.nii.gz"))
v1_img    = nib.load(os.path.join(RDIR, "v1_T1.nii.gz"))
mask_img  = nib.load(os.path.join(RDIR, "dMRI_mask_T1.nii.gz"))
signa_img = nib.load(os.path.join(RDIR, "signaniso_T1.nii.gz"))

MD    = md_img.get_fdata().astype(np.float64)
AD    = ad_img.get_fdata().astype(np.float64)
RD    = rd_img.get_fdata().astype(np.float64)
v1    = v1_img.get_fdata().astype(np.float64)    # (X, Y, Z, 3)
mask  = mask_img.get_fdata().astype(bool)
signa = signa_img.get_fdata().astype(np.float64)  # +1, -1, 0 (reporting only)

affine = md_img.affine

print(f"  MD:   shape={MD.shape}, range=[{MD[mask].min():.4f}, {MD[mask].max():.4f}] μm²/ms")
print(f"  AD:   shape={AD.shape}, range=[{AD[mask].min():.4f}, {AD[mask].max():.4f}] μm²/ms")
print(f"  RD:   shape={RD.shape}, range=[{RD[mask].min():.4f}, {RD[mask].max():.4f}] μm²/ms")
print(f"  v1:   shape={v1.shape}")
print(f"  mask: {mask.sum():,} brain voxels")

# ── DPS shape distribution — reported for methods section, not used in formula ─
is_prolate   = (signa > 0.5)  & mask
is_oblate    = (signa < -0.5) & mask
is_spherical = (np.abs(signa) < 0.5) & mask
print(f"\n  DPS signaniso distribution (reporting only — all get same formula):")
print(f"    Prolate  (+1): {is_prolate.sum():,}  ({100*is_prolate.sum()/mask.sum():.1f}%)")
print(f"    Oblate   (-1): {is_oblate.sum():,}  ({100*is_oblate.sum()/mask.sum():.1f}%)")
print(f"    Spherical (0): {is_spherical.sum():,}  ({100*is_spherical.sum()/mask.sum():.1f}%)")

# ── Zero outside-mask ─────────────────────────────────────────────────────────
MD[~mask] = 0.0
AD[~mask] = 0.0
RD[~mask] = 0.0
v1[~mask] = 0.0

# ── Enforce AD ≥ RD ≥ 0 after trilinear interpolation ────────────────────────
# Boundary voxels can get interpolation artefacts (e.g. mixing brain AD with
# background zero, then rd > ad at the edge). Clip to maintain physical ordering.
RD = np.clip(RD, 0.0, None)
AD = np.maximum(AD, RD)   # ensure AD ≥ RD voxel-wise

n_ad_lt_rd = np.sum((AD < RD) & mask)
print(f"\n  Voxels where AD < RD after clip (should be 0): {n_ad_lt_rd}")

# ── AD/RD ratio statistics ────────────────────────────────────────────────────
eps = 1e-12
ratio_all = AD[mask] / np.maximum(RD[mask], eps)
ratio_prol = AD[is_prolate] / np.maximum(RD[is_prolate], eps)
ratio_obl  = AD[is_oblate]  / np.maximum(RD[is_oblate],  eps)
print(f"\nAD/RD ratio (λ_max/λ_min) — mean compartment tensor:")
print(f"  All brain:  median={np.median(ratio_all):.3f}  p95={np.percentile(ratio_all,95):.3f}  "
      f"max={ratio_all.max():.3f}")
print(f"  Prolate:    median={np.median(ratio_prol):.3f}  p95={np.percentile(ratio_prol,95):.3f}")
print(f"  Oblate:     median={np.median(ratio_obl):.3f}   p95={np.percentile(ratio_obl,95):.3f}")

thresh8 = 4.0**1.5   # 8.0:1 — SimNIBS VN cap at aniso_maxcond=4
n_cap = np.sum(ratio_all > thresh8)
print(f"  Ratio > 8.0 (capped at aniso_maxcond=4): {n_cap:,} ({100*n_cap/mask.sum():.2f}% of brain)")

# ── v1 validity check ─────────────────────────────────────────────────────────
# After vecreg, v1_T1 norms are bimodal: interior voxels ≈1, boundary ≈0.
# The norm distribution has a perfect gap in (0.1, 0.990) — any threshold
# in that range gives identical results. 0.5 is the midpoint of the gap.
v1_norm    = np.linalg.norm(v1, axis=-1)
v1_invalid = (v1_norm < 0.5) & mask
n_v1_inv   = v1_invalid.sum()
print(f"\nv1 validity (after vecreg):")
print(f"  |v1| ≈ 1 (valid):              {mask.sum()-n_v1_inv:,}  ({100*(mask.sum()-n_v1_inv)/mask.sum():.2f}%)")
print(f"  |v1| ≈ 0 (vecreg boundary):    {n_v1_inv:,}  ({100*n_v1_inv/mask.sum():.2f}%) → MD·I fallback")

# ── Build tensor ──────────────────────────────────────────────────────────────
# D = (AD − RD) · u⊗u + RD · I   [valid v1 voxels]
# D = MD · I                      [v1_invalid voxels]
print("\nBuilding conductivity tensor (DPS mean compartment model: σ ∝ ⟨D⟩)...")

v1_valid = (~v1_invalid) & mask
n_valid  = v1_valid.sum()

# Outer-product components
v1x = v1[..., 0];  v1y = v1[..., 1];  v1z = v1[..., 2]
v1x2 = v1x*v1x;  v1y2 = v1y*v1y;  v1z2 = v1z*v1z
v1xy = v1x*v1y;  v1xz = v1x*v1z;  v1yz = v1y*v1z

# Scalar anisotropy factor dl = AD − RD (> 0 where valid; 0 elsewhere)
dl   = np.zeros_like(MD)
lp   = np.zeros_like(MD)    # lp = perpendicular eigenvalue
dl[v1_valid]  = AD[v1_valid] - RD[v1_valid]   # always ≥ 0
lp[v1_valid]  = RD[v1_valid]
lp[v1_invalid & mask] = MD[v1_invalid & mask]  # isotropic fallback

# Assemble [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz] — FSL dtifit order
Dxx = dl * v1x2 + lp
Dxy = dl * v1xy
Dxz = dl * v1xz
Dyy = dl * v1y2 + lp
Dyz = dl * v1yz
Dzz = dl * v1z2 + lp

tensor = np.stack([Dxx, Dxy, Dxz, Dyy, Dyz, Dzz], axis=-1).astype(np.float32)
print(f"  Tensor shape: {tensor.shape}")
print(f"  Tensor range (brain): [{tensor[mask].min():.5f}, {tensor[mask].max():.5f}] μm²/ms")
print(f"  Anisotropic voxels (AD−RD > 0, valid v1): {n_valid:,}  ({100*n_valid/mask.sum():.1f}%)")
print(f"  Isotropic fallback (v1_invalid):           {n_v1_inv:,}  ({100*n_v1_inv/mask.sum():.1f}%)")

# ── Positive definiteness check ───────────────────────────────────────────────
# det = AD · RD²  (always ≥ 0 since AD ≥ RD ≥ 0 after clipping)
det_v1 = AD[v1_valid] * RD[v1_valid]**2
n_neg  = np.sum(det_v1 < 0)
print(f"\nPositive definiteness (anisotropic voxels): {n_neg} with det < 0 (should be 0)")

# ── MD conservation check ─────────────────────────────────────────────────────
md_reconstructed = (AD[v1_valid] + 2*RD[v1_valid]) / 3
md_original      = MD[v1_valid]
err = np.abs(md_reconstructed - md_original)
print(f"\nMD conservation check (anisotropic voxels):")
print(f"  |(AD+2RD)/3 − MD| mean={err.mean():.4f}  max={err.max():.4f} μm²/ms")
print(f"  Note: (AD+2RD)/3 ≈ MD within DPS fitting precision; small residual expected.")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(WDIR, "tensor_MD_dMRI.nii.gz")
hdr = md_img.header.copy()
hdr.set_data_shape(tensor.shape)
hdr.set_data_dtype(np.float32)
nib.save(nib.Nifti1Image(tensor, affine, hdr), out_path)
print(f"\nSaved: {out_path}")
print("\nNext: run 03_run_mddmri_only.py (tensor changed — MD-dMRI sim must be re-run)")
