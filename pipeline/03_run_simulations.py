"""
03_run_simulations.py — Run all three tDCS models for FullPD5

Three-model design:
  ISO     — Isotropic scalar conductivity (literature values, no diffusion weighting)
  DTI     — Anisotropic, FA-based tensors from dwi2cond (classical approach)
  MD-dMRI — Anisotropic, μFA-based tensors from QTI fit (novel contribution)

Montage: C3 (anode, left M1) → Fp2 (cathode, right supraorbital), 2 mA
Same as sub04 reference simulation.

Requirements:
  - m2m_<subject>/ must exist (run CHARM first)
  - m2m_<subject>/dMRI_MNI_reg/ must exist (run dwi2cond first, for DTI model)
  - tensor_MD_dMRI.nii.gz must exist (run 02_build_conductivity_tensor.py first)

Usage:
  cd /Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation
  ~/Applications/SimNIBS-4.6/bin/simnibs_python scripts/03_run_simulations.py

NOTE: Do NOT pass fn_tensor_nifti with tms_flex_opt/tes_flex_opt — documented SimNIBS bug.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg  # noqa: E402  (paths/subject from config/config.sh)

import simnibs
import os
import nibabel as nib
import numpy as np

# ── macOS Apple Silicon: prevent OpenMP crash ─────────────────────────────────
# Must be set BEFORE any SimNIBS import triggers OpenMP initialisation.
# These are no-ops on Linux/Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WDIR    = cfg["WORK_DIR"]
SUBPATH = f'm2m_{cfg["SUBJECT"]}'
TENSOR  = os.path.join(WDIR, "tensor_MD_dMRI.nii.gz")

# SimNIBS resolves subpath and pathfem relative to CWD — must be in WDIR
os.chdir(WDIR)
print(f"Working directory: {os.getcwd()}")

# ── Pre-flight checks ─────────────────────────────────────────────────────────
if not os.path.isdir(SUBPATH):
    raise FileNotFoundError(f"Head model not found: {SUBPATH}  — run CHARM first.")

if not os.path.exists(TENSOR):
    raise FileNotFoundError(f"Tensor file not found: {TENSOR}  — run 02_build_conductivity_tensor.py first.")

# Validate tensor shape and affine match the subject T1
_t = nib.load(TENSOR)
_t1 = nib.load(os.path.join(SUBPATH, "T1.nii.gz"))
assert _t.shape[:3] == _t1.shape[:3], \
    f"Tensor spatial shape {_t.shape[:3]} ≠ T1 shape {_t1.shape[:3]} — affine mismatch!"
assert _t.shape[3] == 6, \
    f"Tensor 4th dim = {_t.shape[3]}, expected 6 (FSL dtifit order)"
assert np.allclose(_t.affine, _t1.affine, atol=1e-3), \
    "Tensor affine does not match T1 affine — tensor may be in wrong space!"
print(f"Tensor validation: shape={_t.shape} ✓  affine matches T1 ✓")
del _t, _t1

def make_electrode_pair(tdcs):
    """C3 anode, Fp2 cathode — 5×5 cm rectangular pads, 2 mA."""
    anode           = tdcs.add_electrode()
    anode.channelnr  = 1
    anode.centre     = "C3"
    anode.shape      = "rect"
    anode.dimensions = [50, 50]
    anode.thickness  = 4

    cathode           = tdcs.add_electrode()
    cathode.channelnr  = 2
    cathode.centre     = "Fp2"
    cathode.shape      = "rect"
    cathode.dimensions = [50, 50]
    cathode.thickness  = 4
    return tdcs


# ── ISO Model ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("MODEL 1: ISO (isotropic scalar conductivity)")
print("=" * 60)

s_iso = simnibs.sim_struct.SESSION()
s_iso.subpath  = SUBPATH
s_iso.pathfem  = "sim_ISO"

tdcs_iso = s_iso.add_tdcslist()
tdcs_iso.currents        = [0.002, -0.002]
tdcs_iso.anisotropy_type = "scalar"       # isotropic literature values
make_electrode_pair(tdcs_iso)

_errors = {}

try:
    simnibs.run_simnibs(s_iso)
    print("ISO simulation complete -> sim_ISO/\n")
except Exception as e:
    _errors["ISO"] = e
    print(f"ERROR in ISO simulation: {e}\n  Continuing with remaining models...\n")


# ── DTI Model ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("MODEL 2: DTI (FA-based tensors from dwi2cond)")
print("=" * 60)

s_dti = simnibs.sim_struct.SESSION()
s_dti.subpath  = SUBPATH
s_dti.pathfem  = "sim_DTI"

tdcs_dti = s_dti.add_tdcslist()
tdcs_dti.currents        = [0.002, -0.002]
tdcs_dti.anisotropy_type = "vn"           # volume-normalised from dwi2cond output
# Match MD-dMRI eigenvalue cap for a fair model comparison.
# max_cond=3 (effective ratio 5.20:1) is justified from ex vivo WM measurements
# of 7–10:1 (Nicholson 1965 Exp Neurol 13:386; Ranck & BeMent 1965 Exp Neurol 11:451).
# In practice this barely changes DTI results: median DTI eigenvalue ratio is ~1.46:1,
# far below both caps, so almost no DTI voxels are affected. But without this
# matched cap the DTI↔MD-dMRI E-field difference cannot be attributed cleanly to
# the tensor source vs. the clipping threshold.
tdcs_dti.aniso_maxcond   = 4   # 4^(3/2) = 8.0:1 — within ex vivo WM benchmark 7–10:1
make_electrode_pair(tdcs_dti)

try:
    simnibs.run_simnibs(s_dti)
    print("DTI simulation complete -> sim_DTI/\n")
except Exception as e:
    _errors["DTI"] = e
    print(f"ERROR in DTI simulation: {e}\n  Continuing with remaining models...\n")


# ── MD-dMRI Model ─────────────────────────────────────────────────────────────
print("=" * 60)
print("MODEL 3: MD-dMRI (μFA-based tensors — novel contribution)")
print("=" * 60)

if not os.path.exists(TENSOR):
    raise FileNotFoundError(
        f"Tensor file not found: {TENSOR}\n"
        "Run 02_build_conductivity_tensor.py first."
    )

s_mddmri = simnibs.sim_struct.SESSION()
s_mddmri.subpath       = SUBPATH
s_mddmri.pathfem       = "sim_MD_dMRI"
# CRITICAL: set fname_tensor on the SESSION, NOT fn_tensor_nifti on the TDCSLIST.
# SESSION._prepare() unconditionally overwrites PL.fn_tensor_nifti with
# self.fname_tensor (sim_struct.py line 209), so the TDCSLIST attribute is ignored.
s_mddmri.fname_tensor  = TENSOR

tdcs_mddmri = s_mddmri.add_tdcslist()
tdcs_mddmri.currents         = [0.002, -0.002]
tdcs_mddmri.anisotropy_type  = "vn"        # volume-normalised, tensor from file
# aniso_maxcond=4 (effective ratio 8.00:1 after VN normalization) — identical to
# the DTI model above.  Using the same cap for both models is essential for a fair
# comparison: any E-field difference between DTI and MD-dMRI reflects the tensor
# source, not a difference in SimNIBS clipping behaviour.
# Physiological justification: ex vivo WM σ_∥/σ_⊥ = 7–10:1 (Nicholson 1965 Exp
# Neurol 13:386; Ranck & BeMent 1965 Exp Neurol 11:451); 8.0:1 sits within this
# validated range and is the most faithful representation we can achieve.
# No μFA pre-cap is applied in 02_build_conductivity_tensor.py; the fraction of
# voxels clipped here is reported in that script's console output and disclosed in
# the methods section as a known limitation of sparse QTI acquisition (38 volumes).
# ISO is isotropic (max_cond irrelevant).
tdcs_mddmri.aniso_maxcond    = 4
make_electrode_pair(tdcs_mddmri)

try:
    simnibs.run_simnibs(s_mddmri)
    print("MD-dMRI simulation complete -> sim_MD_dMRI/\n")
except Exception as e:
    _errors["MD-dMRI"] = e
    print(f"ERROR in MD-dMRI simulation: {e}\n")


# ── Final status ──────────────────────────────────────────────────────────────
print("=" * 60)
_completed = [m for m in ["ISO", "DTI", "MD-dMRI"] if m not in _errors]
_failed    = list(_errors.keys())

if _completed:
    print(f"Completed ({len(_completed)}/3): {', '.join(_completed)}")
    for m in _completed:
        dirname = {"ISO": "sim_ISO", "DTI": "sim_DTI", "MD-dMRI": "sim_MD_dMRI"}[m]
        print(f"  {WDIR}/{dirname}/")

if _failed:
    print(f"\nFailed ({len(_failed)}/3): {', '.join(_failed)}")
    for m, exc in _errors.items():
        print(f"  {m}: {exc}")

if not _failed:
    print("\nAll three simulations complete.")
    print("Next: run 04_extract_roi_efield.py for quantitative comparison.")
else:
    import sys
    sys.exit(1)   # signal failure to the shell while still having run all models
