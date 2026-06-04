"""
02c_build_multicompartment_conductivity.py — Build the free-water-eliminated MD-dMRI tensor (σ ∝ ⟨D⟩_tissue)

Registration strategy (consistent with the ad/rd model, preserves anisotropy):
  • eigenVALUES λ1≥λ2≥λ3 : registered as SCALARS (FLIRT trilinear) → preserves magnitude
  • eigenVECTOR frame      : from the vecreg-reoriented full tensor → preserves orientation
  Whole-tensor trilinear interpolation alone averages neighbours and dilutes anisotropy
  (1.92→1.28 observed); this hybrid keeps the native anisotropy (~1.9) while still using
  proper tensor reorientation for the directions.

Reconstruct (T1 space):  ⟨D⟩ = Σ_k λk_T1 · v_k v_kᵀ   (k=1,2,3)

Inputs (registration/, T1 space):
  lam1_T1, lam2_T1, lam3_T1 .nii.gz   — scalar eigenvalues (µm²/ms)
  tensor_mc_T1.nii.gz           — vecreg-reoriented tensor (orientation frame only)
  v1_T1.nii.gz                        — validated principal direction (QA cross-check)

Output:
  tensor_MD_dMRI_triaxial.nii.gz      — [X,Y,Z,6] FSL order; SimNIBS anisotropy_type='vn'

Model: σ ∝ ⟨D⟩ — Tuch 2001 effective medium (shared eigenvectors; σ-anisotropy =
D-anisotropy) + Güllmar 2010 / Rullmann 2009 volume normalization (SimNIBS 'vn').
Full triaxial: retains λ2≠λ3 and gives oblate voxels their correct disc shape.
"""

import numpy as np
import nibabel as nib
import os

WDIR = "/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation"
RDIR = os.path.join(WDIR, "registration")
M2M  = os.path.join(WDIR, "m2m_FullPD5")
EPS  = 1e-3   # min eigenvalue, µm²/ms (VN-safe)

print("Loading scalar eigenvalues + orientation frame...")
lam1 = nib.load(os.path.join(RDIR, "lam1_mc_T1.nii.gz")).get_fdata().astype(np.float64)
lam2 = nib.load(os.path.join(RDIR, "lam2_mc_T1.nii.gz")).get_fdata().astype(np.float64)
lam3 = nib.load(os.path.join(RDIR, "lam3_mc_T1.nii.gz")).get_fdata().astype(np.float64)
timg = nib.load(os.path.join(RDIR, "tensor_mc_T1.nii.gz"))
t = timg.get_fdata().astype(np.float64)
affine = timg.affine

seg = nib.load(os.path.join(M2M, "final_tissues.nii.gz")).get_fdata()
brain = (seg[..., 0] if seg.ndim == 4 else seg) > 0
# only reconstruct where dMRI actually covers (eigenvalues present)
covered = brain & (lam1 > EPS)
print(f"  brain: {brain.sum():,}   dMRI-covered: {covered.sum():,} ({100*covered.sum()/brain.sum():.1f}%)")

# Orientation frame = eigenvectors of the vecreg-reoriented tensor
M = np.zeros(t.shape[:3] + (3, 3))
M[..., 0, 0]=t[...,0]; M[...,0,1]=t[...,1]; M[...,0,2]=t[...,2]
M[..., 1, 0]=t[...,1]; M[...,1,1]=t[...,3]; M[...,1,2]=t[...,4]
M[..., 2, 0]=t[...,2]; M[...,2,1]=t[...,4]; M[...,2,2]=t[...,5]

out = np.zeros(t.shape[:3] + (6,), dtype=np.float32)
idx = np.argwhere(covered)
Mc = M[covered]
# Guard against any non-finite frame voxels
finite = np.all(np.isfinite(Mc), axis=(1,2))
idx = idx[finite]; Mc = Mc[finite]
_, evec = np.linalg.eigh(Mc)                       # ascending; evec[:,:,k]

l1 = np.maximum(lam1[covered][finite], EPS)
l2 = np.maximum(lam2[covered][finite], EPS)
l3 = np.maximum(lam3[covered][finite], EPS)
# enforce ordering after independent scalar interpolation
ls = np.sort(np.stack([l1, l2, l3], 1), axis=1)    # ascending
l3, l2, l1 = ls[:,0], ls[:,1], ls[:,2]

# ── Anchor principal axis to the validated v1_T1 ─────────────────────────────
# The tensor-vecreg principal axis disagrees with the vector-vecreg v1_T1 by ~8°
# median (mean 14.4°, tail to ~20-24% conductivity error) — two independent
# interpolations of the same dps.u. v1_T1 is the validated direction (18° vs
# dwi2cond V1 in core WM), so use it as v1. Keep the in-plane shape from the
# tensor frame: project the tensor's middle eigenvector into the plane ⊥ v1,
# then v3 = v1 × v2. λ1 (largest) is assigned to the validated principal axis.
v1ref_all = nib.load(os.path.join(RDIR, "v1_T1.nii.gz")).get_fdata()
v1a = v1ref_all[idx[:, 0], idx[:, 1], idx[:, 2]]
nrm = np.linalg.norm(v1a, axis=1)
valid_ref = nrm > 0.5
v1a[valid_ref] /= nrm[valid_ref][:, None]
# where v1_T1 is invalid, fall back to the tensor-frame principal axis
v1a[~valid_ref] = evec[~valid_ref, :, 2]

v2t = evec[:, :, 1]                                # tensor-frame middle eigenvector
v2a = v2t - np.sum(v2t * v1a, axis=1)[:, None] * v1a   # orthogonalize to v1a
n2 = np.linalg.norm(v2a, axis=1)
bad = n2 < 1e-6                                     # v2t ∥ v1a (rare) → pick any ⊥ vector
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
Dt = l1[:,None,None]*outer(v1) + l2[:,None,None]*outer(v2) + l3[:,None,None]*outer(v3)

out[idx[:,0], idx[:,1], idx[:,2], 0] = Dt[:,0,0]
out[idx[:,0], idx[:,1], idx[:,2], 1] = Dt[:,0,1]
out[idx[:,0], idx[:,1], idx[:,2], 2] = Dt[:,0,2]
out[idx[:,0], idx[:,1], idx[:,2], 3] = Dt[:,1,1]
out[idx[:,0], idx[:,1], idx[:,2], 4] = Dt[:,1,2]
out[idx[:,0], idx[:,1], idx[:,2], 5] = Dt[:,2,2]

# ── QA: anisotropy + principal direction vs validated v1_T1 ──────────────────
ratio = l1/np.maximum(l3,1e-6)
print(f"\nAnisotropy λ1/λ3 (covered brain): median={np.median(ratio):.3f}, p95={np.percentile(ratio,95):.3f}")
print(f"  > 8 (VN cap): {100*np.mean(ratio>8):.2f}%")

v1ref = nib.load(os.path.join(RDIR, "v1_T1.nii.gz")).get_fdata()
nref = np.linalg.norm(v1ref, axis=-1)
covv = covered.copy(); covv[tuple(idx[~np.ones(len(idx),bool)].T)] = False  # noop guard
mref = nref[idx[:,0],idx[:,1],idx[:,2]] > 0.5
vr = v1ref[idx[mref,0],idx[mref,1],idx[mref,2]]
vr = vr/np.linalg.norm(vr,axis=1)[:,None]
ang = np.degrees(np.arccos(np.clip(np.abs(np.sum(v1[mref]*vr,1)),0,1)))
print(f"Principal eigenvector vs validated v1_T1: median={np.median(ang):.2f}°, within15°={np.mean(ang<15)*100:.1f}%")

# Canonical MD-dMRI conductivity tensor (replaces the earlier cylindrical ad/rd form).
out_path = os.path.join(WDIR, "tensor_MD_dMRI_mc.nii.gz")
hdr = timg.header.copy(); hdr.set_data_shape(out.shape); hdr.set_data_dtype(np.float32)
nib.save(nib.Nifti1Image(out, affine, hdr), out_path)
print(f"\nSaved: {out_path}")
print("Next: run the mc simulation (SimNIBS anisotropy_type='vn').")
