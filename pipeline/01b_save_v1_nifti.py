"""
01b_save_v1_nifti.py — Save principal eigenvectors from dps.mat as NIfTI

Called automatically by 01_register_dMRI_to_T1.sh (Step 4).

dps.mat is a MATLAB struct: mat['dps'][0,0] gives the struct fields.
  .u   — shape (96, 96, 48, 3), complex64, principal eigenvectors of <D>
          imaginary part is numerical noise (RMS ~0.001) — take real part only.
          Each voxel: [vx, vy, vz] = unit direction of largest eigenvalue.
  .ufa — shape (96, 96, 48), float32, μFA [0, 1] with no artefacts outside mask
  .MD  — shape (96, 96, 48), float32, mean diffusivity
  .mask — shape (96, 96, 48), uint8, QTI brain mask

NOTE: dps['mdxx'], dps['mdyy'], etc. are ALL ZEROS for this pilot — the mean
diffusion tensor components were not computed. We must use the prolate approximation
(v1 eigenvector only) in 02_build_conductivity_tensor.py.

The output NIfTI uses the same affine as dtd_covariance_C_mu.nii.gz (dMRI space).
vecreg will rotate the direction vectors when transforming to T1 space.
"""

import numpy as np
import nibabel as nib
import scipy.io
import os

FDIR  = "/Users/santi/Downloads/FullPD5_forSantiago/FullPD5/fit"
RDIR  = "/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation/registration"

# ── Load eigenvectors from dps.mat (nested MATLAB struct) ─────────────────────
print("Loading dps.mat...")
mat = scipy.io.loadmat(os.path.join(FDIR, "dps.mat"))
dps = mat['dps']          # MATLAB struct stored as (1,1) object array

u_raw = dps['u'][0, 0]   # shape (96, 96, 48, 3), complex64
mask  = dps['mask'][0, 0].astype(bool)  # (96, 96, 48)

print(f"  u shape: {u_raw.shape}, dtype: {u_raw.dtype}")
print(f"  imaginary RMS: {np.sqrt(np.mean(np.imag(u_raw)**2)):.5f}  (should be ~0)")

u = np.real(u_raw)  # (96, 96, 48, 3) — drop negligible imaginary part

# ── Borrow affine from C_mu (same acquisition space as dps.mat) ─────────────
ref_img = nib.load(os.path.join(FDIR, "dtd_covariance_C_mu.nii.gz"))
affine  = ref_img.affine
header  = ref_img.header.copy()

print(f"\nReference affine (from C_mu):\n{affine}")
assert ref_img.shape == u.shape[:3], \
    f"Shape mismatch: C_mu {ref_img.shape} vs u {u.shape[:3]}"

# ── Zero out outside-mask voxels, then normalise to unit vectors ─────────────
u[~mask] = 0.0

norms = np.linalg.norm(u, axis=-1, keepdims=True)   # (96, 96, 48, 1)
norms_brain = np.linalg.norm(u[mask], axis=-1)       # (N_brain_voxels,)
print(f"\nEigenvector norm stats (brain mask voxels):")
print(f"  mean={norms_brain.mean():.4f}, min={norms_brain.min():.4f}, max={norms_brain.max():.4f}")

safe_norms = np.where(norms > 0.1, norms, 1.0)
v1_unit = u / safe_norms   # unit vectors inside mask; zeros outside

# ── Save as 4D NIfTI [X, Y, Z, 3] — vecreg expects [Vx, Vy, Vz] as 3 volumes ──
header.set_data_shape(v1_unit.shape)
header.set_data_dtype(np.float32)
out_path = os.path.join(RDIR, "v1_dMRI.nii.gz")
nib.save(nib.Nifti1Image(v1_unit.astype(np.float32), affine, header), out_path)
print(f"\nSaved: {out_path}  shape={v1_unit.shape}")
