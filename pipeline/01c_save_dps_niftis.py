"""
01c_save_dps_niftis.py — Extract μFA (ufa), MD, and signaniso from dps.mat; save as NIfTIs

Called automatically by 01_register_dMRI_to_T1.sh (Step 3).

WHY: dtd_covariance_C_mu.nii.gz has zeros outside the brain mask (safe for FLIRT),
but 24.3% of BRAIN voxels have non-physical extreme values (range -23,653 to
+13,774 within mask; mean = 26.0 vs expected 0.3-0.8). This is NOT a masking
artifact — it is a covariance model instability caused by insufficient QTI
encoding density (38 volumes for this pilot). The full 4th-order covariance
tensor is underdetermined at this sampling, producing an ill-conditioned fit.

The DPS model fields in dps.mat are better conditioned for sparse QTI sampling:
  ufa [0, 0.999] within mask — physically valid throughout
  MD  [0.005, 4.976] μm²/ms within mask — no negative values
  signaniso {-1, 0, +1} — tensor shape: +1=prolate, -1=oblate, 0=spherical

After clipping dtd_covariance_C_mu to [0,1], its Pearson correlation with
dps.ufa is only r=0.26, confirming these are measuring substantially different
quantities at this data quality level. Use dps.ufa as the primary μFA estimate
for this pilot; revisit the covariance estimator for the full-study protocol.

dps.mat fields used:
  .ufa       — μFA from DPS model, shape (96,96,48), float32, range [0, 1]
  .MD        — Mean diffusivity, shape (96,96,48), float32, range [0.005, 4.976] μm²/ms
  .mask      — Brain mask, shape (96,96,48), uint8
  .signaniso — Tensor shape indicator, shape (96,96,48), float64, values {-1, 0, +1}
  .ad        — Axial diffusivity (largest eigenvalue) of the mean tensor ⟨D⟩, μm²/ms.
  .rd        — Radial diffusivity ((λ2+λ3)/2) of ⟨D⟩, μm²/ms.
               ad/rd are saved here for QA only. The conductivity model uses the FULL triaxial
               ⟨D⟩ — reconstructed from dps.mat's mdxx..mdyz (the populated mean-tensor
               components) in 01d, NOT a cylindrical ad/rd tensor (that interim model is
               superseded; see the derivation-doc appendix).

Outputs (saved in registration/):
  C_mu_dps_dMRI.nii.gz      — μFA in dMRI space (NaN replaced with 0)
  MD_dps_dMRI.nii.gz        — MD in dMRI space, μm²/ms (NaN replaced with 0)
  dMRI_mask.nii.gz          — Brain mask in dMRI space
  signaniso_dMRI.nii.gz     — DPS shape indicator in dMRI space (+1/−1/0)
                              Register with nearestneighbour (not trilinear) to preserve discrete values.
  ad_dMRI.nii.gz            — Axial diffusivity in dMRI space, μm²/ms
  rd_dMRI.nii.gz            — Radial diffusivity in dMRI space, μm²/ms
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg  # noqa: E402  (paths/subject from config/config.sh)

import numpy as np
import nibabel as nib
import scipy.io
import os

FDIR = cfg["FIT_DIR"]
RDIR = cfg["REG_DIR"]

os.makedirs(RDIR, exist_ok=True)

# Load dps.mat
print("Loading dps.mat...")
mat = scipy.io.loadmat(os.path.join(FDIR, "dps.mat"))
dps = mat['dps']  # MATLAB struct stored as (1,1) object array

ufa       = dps['ufa'][0, 0].astype(np.float32)        # shape (96, 96, 48), μFA [0, 1]
MD        = dps['MD'][0, 0].astype(np.float32)        # shape (96, 96, 48), μm²/ms
mask      = dps['mask'][0, 0].astype(bool)            # shape (96, 96, 48)
signaniso = dps['signaniso'][0, 0].astype(np.float32) # shape (96, 96, 48), {-1, 0, +1}
ad        = dps['ad'][0, 0].astype(np.float32)        # shape (96, 96, 48), μm²/ms — largest eigenvalue of ⟨D⟩
rd        = dps['rd'][0, 0].astype(np.float32)        # shape (96, 96, 48), μm²/ms — radial diffusivity (λ2+λ3)/2 of ⟨D⟩

print(f"  ufa       (within mask): [{ufa[mask].min():.4f}, {ufa[mask].max():.4f}]")
print(f"  MD        (within mask): [{MD[mask].min():.4f},  {MD[mask].max():.4f}] μm²/ms")
print(f"  ad        (within mask): [{ad[mask].min():.4f},  {ad[mask].max():.4f}] μm²/ms")
print(f"  rd        (within mask): [{rd[mask].min():.4f},  {rd[mask].max():.4f}] μm²/ms")
n_ad_lt_rd = np.sum((ad < rd) & mask)
print(f"  voxels where ad < rd (should be 0): {n_ad_lt_rd}")
print(f"  mask: {mask.sum()} brain voxels")
prolate = (signaniso > 0.5) & mask
oblate  = (signaniso < -0.5) & mask
sph     = (np.abs(signaniso) < 0.5) & mask
print(f"  signaniso (brain): {prolate.sum()} prolate (+1), {oblate.sum()} oblate (-1), "
      f"{sph.sum()} spherical (0)")

# Replace NaN/inf with 0 outside mask
# dps.mat fields have valid values inside mask; NaN or garbage outside.
# Set to 0 so FLIRT trilinear/nearestneighbour interpolation cannot propagate NaN.
ufa[~mask]       = 0.0
MD[~mask]        = 0.0
signaniso[~mask] = 0.0
ad[~mask]        = 0.0
rd[~mask]        = 0.0

# Clip μFA to [0, 1] — should already be in range, but guard against edge voxels
ufa = np.clip(ufa, 0.0, 1.0)
# Clip ad ≥ rd ≥ 0 (enforce physical ordering; DPS should already satisfy this)
ad = np.maximum(ad, 0.0)
rd = np.clip(rd, 0.0, ad)   # rd cannot exceed ad

# Borrow affine from dtd_covariance_C_mu.nii.gz (same dMRI space)
ref_img = nib.load(os.path.join(FDIR, "dtd_covariance_C_mu.nii.gz"))
affine  = ref_img.affine
assert ref_img.shape == ufa.shape, \
    f"Shape mismatch: C_mu {ref_img.shape} vs ufa {ufa.shape}"

# Save
def save_nifti(data, path, ref_img):
    hdr = ref_img.header.copy()
    hdr.set_data_shape(data.shape)
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, ref_img.affine, hdr), path)
    print(f"  Saved: {path}")

save_nifti(ufa,                        os.path.join(RDIR, "C_mu_dps_dMRI.nii.gz"),  ref_img)
save_nifti(MD,                         os.path.join(RDIR, "MD_dps_dMRI.nii.gz"),    ref_img)
save_nifti(mask.astype(np.float32),   os.path.join(RDIR, "dMRI_mask.nii.gz"),      ref_img)
save_nifti(signaniso,                  os.path.join(RDIR, "signaniso_dMRI.nii.gz"), ref_img)
save_nifti(ad,                         os.path.join(RDIR, "ad_dMRI.nii.gz"),        ref_img)
save_nifti(rd,                         os.path.join(RDIR, "rd_dMRI.nii.gz"),        ref_img)

print("\nDone. Next: FLIRT will apply dMRI_to_T1.mat to these files.")
print("  NOTE: Register signaniso with -interp nearestneighbour to preserve discrete {-1,0,+1} values.")
print("  NOTE: ad and rd use trilinear interpolation (continuous quantities).")
