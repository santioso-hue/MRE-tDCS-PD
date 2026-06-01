"""
03_run_mddmri_only.py — Re-run MD-dMRI simulation only (whole-brain tensor).

Used after reverting from WM-only to whole-brain anisotropy.
ISO and DTI results are unchanged and do not need re-running.
"""

import simnibs
import os
import nibabel as nib
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WDIR    = "/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation"
SUBPATH = "m2m_FullPD5"
TENSOR  = os.path.join(WDIR, "tensor_MD_dMRI.nii.gz")

os.chdir(WDIR)
print(f"Working directory: {os.getcwd()}")

if not os.path.exists(TENSOR):
    raise FileNotFoundError(f"Tensor not found: {TENSOR}")

_t = nib.load(TENSOR)
_t1 = nib.load(os.path.join(SUBPATH, "T1.nii.gz"))
assert _t.shape[:3] == _t1.shape[:3], f"Tensor/T1 spatial shape mismatch: {_t.shape[:3]} vs {_t1.shape[:3]}"
assert _t.shape[3] == 6, f"Tensor 4th dim={_t.shape[3]}, expected 6"
assert np.allclose(_t.affine, _t1.affine, atol=1e-3), "Tensor affine does not match T1 — wrong space!"
print(f"Tensor validation: shape={_t.shape} ✓  affine ✓")
del _t, _t1

print("=" * 60)
print("MODEL 3: MD-dMRI (μFA-based, whole-brain anisotropy)")
print("=" * 60)

s_mddmri = simnibs.sim_struct.SESSION()
s_mddmri.subpath       = SUBPATH
s_mddmri.pathfem       = "sim_MD_dMRI"
# CRITICAL: set fname_tensor on the SESSION, not fn_tensor_nifti on the TDCSLIST.
# SESSION._prepare() (sim_struct.py line 209) unconditionally overwrites
# PL.fn_tensor_nifti = self.fname_tensor, so the TDCSLIST attribute is silently
# ignored and the simulation falls back to the dwi2cond tensor.
s_mddmri.fname_tensor  = TENSOR

tdcs = s_mddmri.add_tdcslist()
tdcs.currents        = [0.002, -0.002]
tdcs.anisotropy_type = "vn"
# aniso_maxcond=4 (threshold 8.00:1) — matches DTI model for fair comparison.
# 8.0:1 is within ex vivo WM benchmark 7–10:1 (Nicholson 1965; Ranck & BeMent 1965).
# No μFA pre-cap in 02_build_conductivity_tensor.py; clipped fraction disclosed in methods.
tdcs.aniso_maxcond   = 4

anode            = tdcs.add_electrode()
anode.channelnr  = 1
anode.centre     = "C3"
anode.shape      = "rect"
anode.dimensions = [50, 50]
anode.thickness  = 4

cathode            = tdcs.add_electrode()
cathode.channelnr  = 2
cathode.centre     = "Fp2"
cathode.shape      = "rect"
cathode.dimensions = [50, 50]
cathode.thickness  = 4

simnibs.run_simnibs(s_mddmri)
print("\nMD-dMRI simulation complete -> sim_MD_dMRI/")
print("Done.")
