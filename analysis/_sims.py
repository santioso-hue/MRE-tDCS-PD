"""_sims.py — shared lookup for the per-(montage x model) SimNIBS output meshes.

Single source of the model -> (dir token, mesh-field suffix) map and the montage-aware path, so 04_extract,
05 and qc_harness all resolve 04_run_simulations.py's sim_<montage>_<token>/ meshes identically (with a
fallback to the legacy sim_<token> = old M1 layout).
"""
import os

# model name -> (sim-dir token, mesh-field suffix). ISO is scalar; both anisotropic arms are 'vn'.
MODEL_TOKENS = {"ISO": ("ISO", "scalar"), "DTI": ("DTI", "vn"), "MD-dMRI": ("MD_dMRI", "vn")}
MODELS = list(MODEL_TOKENS)


def sim_mesh(work_dir, montage, model, subject):
    """Path to the TDCS mesh for (montage, model), or None. Tries sim_<montage>_<token>, then legacy sim_<token>."""
    token, suf = MODEL_TOKENS[model]
    for d in (f"sim_{montage}_{token}", f"sim_{token}"):
        p = os.path.join(work_dir, d, f"{subject}_TDCS_1_{suf}.msh")
        if os.path.exists(p):
            return p
    return None
