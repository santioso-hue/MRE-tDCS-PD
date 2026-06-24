"""Build the MD-dMRI conductivity tensor (σ ∝ ⟨D⟩) -> tensor_MD_dMRI.nii.gz.

Usage: simnibs_python pipeline/03_build_conductivity_tensor.py

The MD-dMRI model maps the QTI mean diffusion tensor ⟨D⟩ to conductivity by the standard SimNIBS 'vn'
rule. Downstream physics is identical to dwi2cond's DTI model; only the input tensor differs (QTI ⟨D⟩
vs single-shell DTI). Rejected alternatives (FWE, magnitude preservation, μFA/covariance) are in the
manuscript Methods.

Reconstruct in T1 space D = Σ_k λk · v_k v_kᵀ from the separately-warped eigenvalues and frame:
eigenvalues λ1≥λ2≥λ3 are interpolated independently as scalar maps (02, FLIRT trilinear), the
orientation frame is carried by vecreg (PPD reorientation, Alexander 2001) with the principal axis
anchored to v1_T1 (= dps.u). Component-wise trilinear interpolation of the whole 6-vector would
average neighbouring tensors and shrink anisotropy (the "swelling" PPD avoids). After interpolation
the scalar maps are re-sorted to λ1≥λ2≥λ3 and re-paired with the frame by magnitude order, so a
boundary voxel where the maps cross cannot mis-assign.

σ ∝ D: Tuch 2001 effective medium + Güllmar 2010 / Rullmann 2009 volume normalization (SimNIBS 'vn').
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg  # noqa: E402  (paths/subject from config/config.sh)

import numpy as np
import nibabel as nib
import nibabel.processing as nibproc

WDIR = cfg["WORK_DIR"]
RDIR = cfg["REG_DIR"]   # the T1-space maps written by 02_register_dmri_to_T1.sh
M2M  = cfg["M2M_DIR"]
EPS  = 1e-3   # min eigenvalue, µm²/ms (VN-safe)

# Eigenvalue scalar maps (magnitude) + reoriented tensor frame (orientation), both from 02.
lam_fmt    = "lam{}_T1"
frame_name = "tensor_triaxial_T1.nii.gz"
out_name   = "tensor_MD_dMRI.nii.gz"

lam1 = nib.load(os.path.join(RDIR, lam_fmt.format(1) + ".nii.gz")).get_fdata().astype(np.float64)
lam2 = nib.load(os.path.join(RDIR, lam_fmt.format(2) + ".nii.gz")).get_fdata().astype(np.float64)
lam3 = nib.load(os.path.join(RDIR, lam_fmt.format(3) + ".nii.gz")).get_fdata().astype(np.float64)
timg = nib.load(os.path.join(RDIR, frame_name))
t = timg.get_fdata().astype(np.float64)
affine = timg.affine

# The conductivity maps may be on a coarser grid than the charm segmentation (TENSOR_GRID_RES).
# Resample the (3D) brain segmentation onto the eigenvalue grid by nearest neighbour so masks align.
lam1_img = nib.load(os.path.join(RDIR, lam_fmt.format(1) + ".nii.gz"))
seg_src = nib.load(os.path.join(M2M, "final_tissues.nii.gz"))
seg_data = seg_src.get_fdata()
seg_img = nib.Nifti1Image((seg_data[..., 0] if seg_data.ndim == 4 else seg_data), seg_src.affine)
if seg_img.shape[:3] != lam1_img.shape[:3] or not np.allclose(seg_img.affine, lam1_img.affine, atol=1e-3):
    seg_img = nibproc.resample_from_to(seg_img, (lam1_img.shape[:3], lam1_img.affine), order=0)
brain = seg_img.get_fdata() > 0
covered = brain & (lam1 > EPS)   # only reconstruct where dMRI actually covers (eigenvalues present)
print(f"  brain: {brain.sum():,}   dMRI-covered: {covered.sum():,} ({100*covered.sum()/brain.sum():.1f}%)")

# Orientation frame = eigenvectors of the vecreg-reoriented tensor
M = np.zeros(t.shape[:3] + (3, 3))
M[..., 0, 0]=t[...,0]; M[...,0,1]=t[...,1]; M[...,0,2]=t[...,2]
M[..., 1, 0]=t[...,1]; M[...,1,1]=t[...,3]; M[...,1,2]=t[...,4]
M[..., 2, 0]=t[...,2]; M[...,2,1]=t[...,4]; M[...,2,2]=t[...,5]

out = np.zeros(t.shape[:3] + (6,), dtype=np.float32)
idx = np.argwhere(covered)
Mc = M[covered]
finite = np.all(np.isfinite(Mc), axis=(1, 2))   # guard against any non-finite frame voxels
idx = idx[finite]; Mc = Mc[finite]
_, evec = np.linalg.eigh(Mc)                     # ascending; evec[:,:,k]

l1 = np.maximum(lam1[covered][finite], EPS)
l2 = np.maximum(lam2[covered][finite], EPS)
l3 = np.maximum(lam3[covered][finite], EPS)
ls = np.sort(np.stack([l1, l2, l3], 1), axis=1)  # enforce ordering after independent scalar interp
l3, l2, l1 = ls[:, 0], ls[:, 1], ls[:, 2]

# Anchor the principal axis to v1_T1 (the vecreg-reoriented dps.u). The tensor-vecreg principal axis
# and v1_T1 are two independent interpolations of dps.u and can disagree; v1_T1 is the reference frame.
# Use it as v1, keep the in-plane shape from the tensor frame, set λ1->v1_T1.
v1ref_all = nib.load(os.path.join(RDIR, "v1_T1.nii.gz")).get_fdata()
v1a = v1ref_all[idx[:, 0], idx[:, 1], idx[:, 2]]
nrm = np.linalg.norm(v1a, axis=1)
valid_ref = nrm > 0.5
v1a[valid_ref] /= nrm[valid_ref][:, None]
v1a[~valid_ref] = evec[~valid_ref, :, 2]         # where v1_T1 invalid, use tensor-frame principal axis

v2t = evec[:, :, 1]                              # tensor-frame middle eigenvector
v2a = v2t - np.sum(v2t * v1a, axis=1)[:, None] * v1a   # orthogonalize to v1a
n2 = np.linalg.norm(v2a, axis=1)
bad = n2 < 1e-6                                  # v2t ∥ v1a → pick any ⊥ vector
if bad.any():
    tmp = np.tile(np.array([1.0, 0, 0]), (bad.sum(), 1))
    alt = np.abs(v1a[bad, 0]) > 0.9
    tmp[alt] = np.array([0, 1.0, 0])
    v2a[bad] = tmp - np.sum(tmp * v1a[bad], axis=1)[:, None] * v1a[bad]
    n2 = np.linalg.norm(v2a, axis=1)
v2a /= n2[:, None]
v3a = np.cross(v1a, v2a)
v1, v2, v3 = v1a, v2a, v3a

# Reconstruct D = λ1 v1v1ᵀ + λ2 v2v2ᵀ + λ3 v3v3ᵀ
def outer(v): return v[:, :, None] * v[:, None, :]
Dt = l1[:, None, None]*outer(v1) + l2[:, None, None]*outer(v2) + l3[:, None, None]*outer(v3)

out[idx[:, 0], idx[:, 1], idx[:, 2], 0] = Dt[:, 0, 0]
out[idx[:, 0], idx[:, 1], idx[:, 2], 1] = Dt[:, 0, 1]
out[idx[:, 0], idx[:, 1], idx[:, 2], 2] = Dt[:, 0, 2]
out[idx[:, 0], idx[:, 1], idx[:, 2], 3] = Dt[:, 1, 1]
out[idx[:, 0], idx[:, 1], idx[:, 2], 4] = Dt[:, 1, 2]
out[idx[:, 0], idx[:, 1], idx[:, 2], 5] = Dt[:, 2, 2]

# QA + abort on a broken tensor
ratio = l1 / np.maximum(l3, 1e-6)
print(f"\nAnisotropy λ1/λ3 (covered brain): median={np.median(ratio):.3f}, p95={np.percentile(ratio,95):.3f}")
print(f"  > 8 (VN cap): {100*np.mean(ratio>8):.2f}%")
# Cross-check the two INDEPENDENT reorientation paths: the v1_T1 anchor (vecreg of v1_dMRI) vs the
# principal axis of the vecreg-reoriented tensor (evec[:,:,2]). Reoriented separately, so agreement is
# a real test that reorientation held (comparing v1 to v1_T1 would be a tautology - v1 IS the anchor).
# Restrict to voxels where the anchor was used, not the degenerate fallback (which set v1 = evec[:,:,2]).
tensor_axis = evec[valid_ref, :, 2]
ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(v1[valid_ref] * tensor_axis, axis=1)), 0, 1)))
print(f"v1_T1 anchor vs independent tensor-frame axis: median={np.median(ang):.2f}°, within15°={np.mean(ang<15)*100:.1f}%")
assert np.isfinite(out[idx[:, 0], idx[:, 1], idx[:, 2]]).all(), "non-finite values in the conductivity tensor"
assert np.mean(ang < 15) > 0.9, "v1_T1 anchor disagrees with the tensor-frame principal axis in >10% of WM - reorientation broke"

out_path = os.path.join(WDIR, out_name)
hdr = timg.header.copy(); hdr.set_data_shape(out.shape); hdr.set_data_dtype(np.float32)
nib.save(nib.Nifti1Image(out, affine, hdr), out_path)
print(f"\nSaved: {out_path}")
