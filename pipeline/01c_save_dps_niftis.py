"""
01c_save_dps_niftis.py — Extract μFA (ufa) and MD from dps.mat and save as NIfTIs

Called automatically by 01_register_dMRI_to_T1.sh (Step 3).

WHY: dtd_covariance_C_mu.nii.gz and dtd_covariance_MD.nii.gz from the QTI fit
contain NaN outside the brain mask. FSL trilinear interpolation propagates NaN
into the brain region, corrupting the C_mu_T1 and MD_T1 maps. The DPS model
fields in dps.mat (ufa, MD) are clean: valid values within the mask,
identically zero outside. These are safe for FLIRT interpolation.

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
