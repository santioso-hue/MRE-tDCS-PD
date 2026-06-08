"""
01d_save_triaxial_tensor.py — Save the FULL triaxial mean diffusion tensor ⟨D⟩ from dps.mat

KEY CORRECTION (supersedes the note in 01b/01c):
  dps['mdxx'..'mdyz'] are NOT zero. They are the full mean compartment tensor ⟨D⟩,
  stored in SI units (m²/s, ~1.5e-9). Earlier inspection mistook them for zero because
  1.5e-9 rounds to 0.000 at display precision. Multiplying by 1e9 gives µm²/ms.

  Verified against dps.mat (130,643 brain voxels; tests/validate_mean_tensor.py):
    trace(⟨D⟩)/3 == MD            (max|err| 6e-7 µm²/ms; confirms it is the mean tensor)
    λ1(⟨D⟩) == ad                 (median|err| 3e-5; a few near-degenerate outliers)
    (λ2+λ3)/2 == rd               (median|err| 1e-5)
    principal eigenvector == dps.u (median 0.01°)
    positive-definite              100% of brain
    genuinely triaxial (|λ2−λ3| > 0.05·MD)  93% of brain

  Its principal axis (dps.u) agrees with the INDEPENDENT single-shell dwi2cond DTI V1 to a
  median of ~22° in core WM (FA>0.5), ~30° across all WM (FA>0.3) — they share orientation only
  moderately, so the Model-2-vs-DTI E-field difference reflects both eigenvalue and orientation
  differences. dps.u carries its own validated orientation, so the comparison is not circular.

This is the σ ∝ ⟨D⟩ conductivity model in its full triaxial form — the established
Tuch (2001) effective-medium mapping (shared eigenvectors; conductivity anisotropy =
diffusion anisotropy) with Güllmar (2010) / Rullmann (2009) volume normalization
applied by SimNIBS (anisotropy_type='vn'). Sourcing ⟨D⟩ from the QTI LTE+STE mean
tensor — rather than single-shell DTI — is a more faithful realization of the
ensemble-averaged effective-medium tensor.

Element order written: FSL dtifit / vecreg convention [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
  D = [[mdxx, mdxy, mdxz],
       [mdxy, mdyy, mdyz],
       [mdxz, mdyz, mdzz]]

Output (dMRI space):
  registration/tensor_triaxial_dMRI.nii.gz  — 6-component ⟨D⟩ [X,Y,Z,6], µm²/ms
  Next step (01_register…): vecreg → T1 with proper tensor reorientation.
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

print("Loading dps.mat...")
dps = scipy.io.loadmat(os.path.join(FDIR, "dps.mat"))['dps']
mask = dps['mask'][0, 0].astype(bool)

# Mean tensor components — SI units (m²/s) → µm²/ms via ×1e9
SI_TO_UM2MS = 1.0e9
g = lambda f: np.real(dps[f][0, 0]).astype(np.float64) * SI_TO_UM2MS
mdxx, mdyy, mdzz = g('mdxx'), g('mdyy'), g('mdzz')
mdxy, mdxz, mdyz = g('mdxy'), g('mdxz'), g('mdyz')
MD = np.real(dps['MD'][0, 0]).astype(np.float64)          # already µm²/ms
ad = np.real(dps['ad'][0, 0]).astype(np.float64)
rd = np.real(dps['rd'][0, 0]).astype(np.float64)

# Validate the reconstruction before writing
print("\nValidating ⟨D⟩ reconstruction (brain voxels)...")
D = np.zeros(mask.shape + (3, 3))
D[..., 0, 0] = mdxx; D[..., 1, 1] = mdyy; D[..., 2, 2] = mdzz
D[..., 0, 1] = D[..., 1, 0] = mdxy
D[..., 0, 2] = D[..., 2, 0] = mdxz
D[..., 1, 2] = D[..., 2, 1] = mdyz
ev = np.linalg.eigvalsh(D[mask])            # ascending λ3≤λ2≤λ1
l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]
trace_md = (mdxx + mdyy + mdzz) / 3.0
print(f"  trace/3 vs MD:    max|err|={np.max(np.abs(trace_md[mask]-MD[mask])):.4f} µm²/ms")
print(f"  λ1 vs ad:         max|err|={np.max(np.abs(l1-ad[mask])):.4f} µm²/ms")
print(f"  (λ2+λ3)/2 vs rd:  max|err|={np.max(np.abs((l2+l3)/2-rd[mask])):.4f} µm²/ms")
print(f"  positive-definite: {100*np.mean(l3>0):.2f}%  (λ3 min={l3.min():.4f})")
print(f"  eigenvalues µm²/ms: λ1={np.median(l1):.3f}  λ2={np.median(l2):.3f}  λ3={np.median(l3):.3f} (median)")

# Abort on a broken reconstruction (unit-scale slip, mis-assembled diagonal) rather than silently
# writing a plausible-but-wrong tensor. trace/3 == MD is a definitional identity (exact). NOTE: ad/rd
# are the fit's CYLINDRICAL axial/radial summaries and differ from the triaxial eigenvalues by a few %
# (max|λ1-ad|≈0.16), so they stay informational prints above rather than strict asserts.
assert np.max(np.abs(trace_md[mask] - MD[mask])) < 1e-2, "trace/3 != MD — unit-scale or diagonal-component bug"
assert np.mean(l3 > 0) > 0.99, "reconstructed ⟨D⟩ not positive-definite in >1% of brain — assembly bug"

# Eigenvalue maps (scalars) — registered with trilinear to PRESERVE magnitude
# Whole-tensor vecreg interpolation averages neighbouring tensors and dilutes the
# anisotropy ratio (1.92→1.28 observed). Registering eigenvalues as scalars (as the
# ad/rd model does) preserves the eigenvalue magnitudes; vecreg of the full tensor is
# then used ONLY to carry the orientation frame to T1. Reconstruction in 02 combines
# scalar-registered eigenvalues with the vecreg orientation.
lam1 = np.zeros(mask.shape); lam2 = np.zeros(mask.shape); lam3 = np.zeros(mask.shape)
lam_brain = np.linalg.eigvalsh(D[mask])           # (Nbrain,3) ascending — mask avoids outside-NaN
lam1[mask] = lam_brain[:, 2]                       # largest
lam2[mask] = lam_brain[:, 1]                       # middle
lam3[mask] = lam_brain[:, 0]                       # smallest

# Zero outside mask so registration interpolation cannot pull in garbage
for a in (mdxx, mdyy, mdzz, mdxy, mdxz, mdyz):
    a[~mask] = 0.0

# Assemble FSL dtifit order [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz] (for vecreg frame)
tensor = np.stack([mdxx, mdxy, mdxz, mdyy, mdyz, mdzz], axis=-1).astype(np.float32)

# Affine from C_mu (same dMRI space as dps.mat, matches v1_dMRI)
ref = nib.load(os.path.join(FDIR, "dtd_covariance_C_mu.nii.gz"))
assert ref.shape == mask.shape, f"shape mismatch {ref.shape} vs {mask.shape}"

def save(arr, name):
    hdr = ref.header.copy(); hdr.set_data_shape(arr.shape); hdr.set_data_dtype(np.float32)
    p = os.path.join(RDIR, name)
    nib.save(nib.Nifti1Image(arr.astype(np.float32), ref.affine, hdr), p)
    print(f"  saved {p}")

save(tensor, "tensor_triaxial_dMRI.nii.gz")       # for vecreg → orientation frame
save(lam1,   "lam1_dMRI.nii.gz")                  # scalar eigenvalues → magnitude
save(lam2,   "lam2_dMRI.nii.gz")
save(lam3,   "lam3_dMRI.nii.gz")
print("\nNext: FLIRT(trilinear) lam1/2/3 → T1 ; vecreg tensor → T1 ; then 02 reconstructs.")
