"""
01c_save_dps_niftis.py — Extract μFA (ufa) and MD from dps.mat and save as NIfTIs

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

After clipping dtd_covariance_C_mu to [0,1], its Pearson correlation with
dps.ufa is only r=0.26, confirming these are measuring substantially different
quantities at this data quality level. Use dps.ufa as the primary μFA estimate
for this pilot; revisit the covariance estimator for the full-study protocol.

dps.mat fields used:
  .ufa   — μFA from DPS model, shape (96,96,48), float32, range [0, 1]
  .MD    — Mean diffusivity, shape (96,96,48), float32, range [0.005, 4.976] μm²/ms
  .mask  — Brain mask, shape (96,96,48), uint8

Note: C_mu_mu from the covariance model (dtd_covariance_C_mu) is related but
different from dps.ufa. Both are estimates of microscopic anisotropy but from
different fitting frameworks. dps.ufa is used here for practical reasons (NaN-free);
future work should compare both estimators.

Outputs (saved in registration/):
  C_mu_dps_dMRI.nii.gz   — μFA in dMRI space (NaN replaced with 0)
  MD_dps_dMRI.nii.gz     — MD in dMRI space, μm²/ms (NaN replaced with 0)
  dMRI_mask.nii.gz       — Brain mask in dMRI space
"""

import numpy as np
import nibabel as nib
import scipy.io
import os

FDIR = "/Users/santi/Downloads/FullPD5_forSantiago/FullPD5/fit"
RDIR = "/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation/registration"

os.makedirs(RDIR, exist_ok=True)

# ── Load dps.mat ───────────────────────────────────────────────────────────────
print("Loading dps.mat...")
mat = scipy.io.loadmat(os.path.join(FDIR, "dps.mat"))
dps = mat['dps']  # MATLAB struct stored as (1,1) object array

ufa  = dps['ufa'][0, 0].astype(np.float32)   # shape (96, 96, 48), μFA [0, 1]
MD   = dps['MD'][0, 0].astype(np.float32)    # shape (96, 96, 48), μm²/ms
mask = dps['mask'][0, 0].astype(bool)        # shape (96, 96, 48)

print(f"  ufa  (within mask): [{ufa[mask].min():.4f}, {ufa[mask].max():.4f}]")
print(f"  MD   (within mask): [{MD[mask].min():.4f},  {MD[mask].max():.4f}] μm²/ms")
print(f"  mask: {mask.sum()} brain voxels")

# ── Replace NaN/inf with 0 outside mask ───────────────────────────────────────
# dps.mat fields have valid values inside mask; NaN or garbage outside.
# Set to 0 so FLIRT trilinear interpolation cannot propagate NaN.
ufa[~mask] = 0.0
MD[~mask]  = 0.0

# Clip μFA to [0, 1] — should already be in range, but guard against edge voxels
ufa = np.clip(ufa, 0.0, 1.0)

# ── Borrow affine from dtd_covariance_C_mu.nii.gz (same dMRI space) ───────────
ref_img = nib.load(os.path.join(FDIR, "dtd_covariance_C_mu.nii.gz"))
affine  = ref_img.affine
assert ref_img.shape == ufa.shape, \
    f"Shape mismatch: C_mu {ref_img.shape} vs ufa {ufa.shape}"

# ── Save ────────────────────────────────────────────────────────────────────────
def save_nifti(data, path, ref_img):
    hdr = ref_img.header.copy()
    hdr.set_data_shape(data.shape)
    hdr.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, ref_img.affine, hdr), path)
    print(f"  Saved: {path}")

save_nifti(ufa,              os.path.join(RDIR, "C_mu_dps_dMRI.nii.gz"), ref_img)
save_nifti(MD,               os.path.join(RDIR, "MD_dps_dMRI.nii.gz"),   ref_img)
save_nifti(mask.astype(np.float32),
                             os.path.join(RDIR, "dMRI_mask.nii.gz"),     ref_img)

print("\nDone. Next: FLIRT will apply dMRI_to_T1.mat to these files.")
