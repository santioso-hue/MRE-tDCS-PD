"""
02_build_conductivity_tensor.py — Build THE MD-dMRI conductivity tensor (σ ∝ ⟨D⟩).

The single MD-dMRI model is the QTI mean diffusion tensor ⟨D⟩ (the first cumulant of the
intra-voxel diffusion-tensor distribution) mapped to conductivity by the standard SimNIBS 'vn'
(volume-normalized) rule. The downstream physics is IDENTICAL to dwi2cond's DTI model; only the
input tensor differs (QTI ⟨D⟩ vs single-shell DTI). So ISO / DTI / MD-dMRI is a controlled
comparison that isolates the input-tensor estimation, and nothing else departs from standard SimNIBS.

Alternatives evaluated and NOT used (free-water elimination, magnitude preservation, μFA/covariance)
are kept for internal reference under pipeline/internal/ — see that folder's README for why each was
rejected (FWE: ~0% field effect under 'vn' and ill-posed at this volume count; magnitude: injects
~46%-CoV partial-volume/noise; μFA & covariance: microscopic, no macroscopic eigenframe).

Registering a diffusion tensor to the structural grid must REORIENT it, not just resample the
components. Eigen-decomposition approach:
  • eigenVALUES λ1≥λ2≥λ3 — interpolated INDEPENDENTLY as three scalar maps (FLIRT trilinear).
  • orientation frame     — eigenvectors of the whole tensor carried by vecreg (PPD-style
                            reorientation; Alexander 2001), principal axis anchored to v1_T1 (= dps.u).
Component-wise trilinear interpolation of the whole 6-vector averages neighbouring tensors and
shrinks anisotropy (the tensor "swelling" that log-Euclidean/PPD schemes avoid). After interpolation
the three scalar maps are re-SORTED to λ1≥λ2≥λ3 and re-paired with the frame BY MAGNITUDE ORDER, and
λ1 is anchored to the validated v1_T1, so a boundary voxel where the maps cross cannot mis-assign.

Reconstruct (T1 space):  D = Σ_k λk_T1 · v_k v_kᵀ   (k=1,2,3)

Usage:
  simnibs_python pipeline/02_build_conductivity_tensor.py    ->  tensor_MD_dMRI.nii.gz

Model: σ ∝ D — Tuch 2001 effective medium (shared eigenvectors; σ-anisotropy = D-anisotropy) +
Güllmar 2010 / Rullmann 2009 volume normalization (SimNIBS 'vn'). Fully triaxial (λ2≠λ3 in ~93%).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg  # noqa: E402  (paths/subject from config/config.sh)

import numpy as np
import nibabel as nib

WDIR = cfg["WORK_DIR"]
RDIR = os.path.join(WDIR, "registration")
M2M  = cfg["M2M_DIR"]
EPS  = 1e-3   # min eigenvalue, µm²/ms (VN-safe)

# The single MD-dMRI model: plain QTI mean tensor ⟨D⟩, mapped by SimNIBS 'vn'.
# eigenvalue scalar maps (magnitude) + reoriented tensor frame (in-plane orientation), both from 01d.
lam_fmt    = "lam{}_T1"
frame_name = "tensor_triaxial_T1.nii.gz"
out_name   = "tensor_MD_dMRI.nii.gz"
print("MD-dMRI conductivity tensor — σ ∝ ⟨D⟩ (QTI mean tensor) -> SimNIBS 'vn'")

print("Loading scalar eigenvalues + orientation frame...")
lam1 = nib.load(os.path.join(RDIR, lam_fmt.format(1) + ".nii.gz")).get_fdata().astype(np.float64)
lam2 = nib.load(os.path.join(RDIR, lam_fmt.format(2) + ".nii.gz")).get_fdata().astype(np.float64)
lam3 = nib.load(os.path.join(RDIR, lam_fmt.format(3) + ".nii.gz")).get_fdata().astype(np.float64)
timg = nib.load(os.path.join(RDIR, frame_name))
t = timg.get_fdata().astype(np.float64)
affine = timg.affine

seg = nib.load(os.path.join(M2M, "final_tissues.nii.gz")).get_fdata()
brain = (seg[..., 0] if seg.ndim == 4 else seg) > 0
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

# Anchor principal axis to the validated v1_T1
# The tensor-vecreg principal axis disagrees with the vector-vecreg v1_T1 by ~8° median (two
# independent interpolations of dps.u). v1_T1 is the reference (agrees with dwi2cond DTI V1 to ~22°
# median in core WM); use it as v1, keep the in-plane shape from the tensor frame, set λ1→v1_T1.
v1ref_all = nib.load(os.path.join(RDIR, "v1_T1.nii.gz")).get_fdata()
v1a = v1ref_all[idx[:, 0], idx[:, 1], idx[:, 2]]
nrm = np.linalg.norm(v1a, axis=1)
valid_ref = nrm > 0.5
v1a[valid_ref] /= nrm[valid_ref][:, None]
v1a[~valid_ref] = evec[~valid_ref, :, 2]         # where v1_T1 invalid, use tensor-frame principal axis

v2t = evec[:, :, 1]                              # tensor-frame middle eigenvector
v2a = v2t - np.sum(v2t * v1a, axis=1)[:, None] * v1a   # orthogonalize to v1a
n2 = np.linalg.norm(v2a, axis=1)
bad = n2 < 1e-6                                  # v2t ∥ v1a (rare) → pick any ⊥ vector
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
v1ref = nib.load(os.path.join(RDIR, "v1_T1.nii.gz")).get_fdata()
nref = np.linalg.norm(v1ref, axis=-1)
mref = nref[idx[:, 0], idx[:, 1], idx[:, 2]] > 0.5
vr = v1ref[idx[mref, 0], idx[mref, 1], idx[mref, 2]]
vr = vr / np.linalg.norm(vr, axis=1)[:, None]
ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(v1[mref]*vr, 1)), 0, 1)))
print(f"Principal eigenvector vs validated v1_T1: median={np.median(ang):.2f}°, within15°={np.mean(ang<15)*100:.1f}%")
assert np.isfinite(out[idx[:, 0], idx[:, 1], idx[:, 2]]).all(), "non-finite values in the conductivity tensor"
assert np.mean(ang < 15) > 0.9, "principal axis disagrees with v1_T1 in >10% of WM — anchoring/registration broke"

out_path = os.path.join(WDIR, out_name)
hdr = timg.header.copy(); hdr.set_data_shape(out.shape); hdr.set_data_dtype(np.float32)
nib.save(nib.Nifti1Image(out, affine, hdr), out_path)
print(f"\nSaved: {out_path}")
print("Next: run 03_run_simulations.py (SimNIBS anisotropy_type='vn').")
