"""
03_run_simulations.py — Run all three tDCS models for FullPD5

Three-model design:
  ISO     — Isotropic scalar conductivity (literature values, no diffusion weighting)
  DTI     — Anisotropic, FA-based tensors from dwi2cond (classical approach)
  MD-dMRI — Anisotropic, μFA-based tensors from QTI fit (novel contribution)

Montage: C3 (anode, left M1) → Fp2 (cathode, right supraorbital), 2 mA
Same as sub04 reference simulation.

Requirements:
  - m2m_FullPD5/ must exist (run CHARM first)
  - m2m_FullPD5/dMRI_MNI_reg/ must exist (run dwi2cond first, for DTI model)
  - tensor_MD_dMRI.nii.gz must exist (run 02_build_conductivity_tensor.py first)

Usage:
  cd /Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation
  ~/Applications/SimNIBS-4.6/bin/simnibs_python scripts/03_run_simulations.py

NOTE: Do NOT pass fn_tensor_nifti with tms_flex_opt/tes_flex_opt — documented SimNIBS bug.
"""

import simnibs
import os

WDIR    = "/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation"
SUBPATH = "m2m_FullPD5"
TENSOR  = os.path.join(WDIR, "tensor_MD_dMRI.nii.gz")

# SimNIBS resolves subpath and pathfem relative to CWD — must be in WDIR
os.chdir(WDIR)
print(f"Working directory: {os.getcwd()}")

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

simnibs.run_simnibs(s_iso)
print("ISO simulation complete -> sim_ISO/\n")


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
make_electrode_pair(tdcs_dti)

simnibs.run_simnibs(s_dti)
print("DTI simulation complete -> sim_DTI/\n")


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
s_mddmri.subpath  = SUBPATH
s_mddmri.pathfem  = "sim_MD_dMRI"

tdcs_mddmri = s_mddmri.add_tdcslist()
tdcs_mddmri.currents         = [0.002, -0.002]
tdcs_mddmri.anisotropy_type  = "vn"        # volume-normalised, tensor from file
tdcs_mddmri.fn_tensor_nifti  = TENSOR      # bypasses dwi2cond entirely
# NOTE: Do NOT set tms_flex_opt or tes_flex_opt — documented bug with fn_tensor_nifti
make_electrode_pair(tdcs_mddmri)

simnibs.run_simnibs(s_mddmri)
print("MD-dMRI simulation complete -> sim_MD_dMRI/\n")


print("=" * 60)
print("All three simulations complete.")
print("Outputs:")
print(f"  {WDIR}/sim_ISO/")
print(f"  {WDIR}/sim_DTI/")
print(f"  {WDIR}/sim_MD_dMRI/")
print("")
print("Next: run 04_extract_roi_efield.py for quantitative comparison.")
