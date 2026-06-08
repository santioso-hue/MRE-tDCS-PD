"""
03_run_simulations.py — Run the tDCS conductivity-model simulations for one subject.

Models (data-driven, see MODELS below):
  ISO      — isotropic scalar conductivity (literature values)
  DTI      — anisotropic 'vn' from dwi2cond (FA-based classical baseline)
  MD-dMRI  — anisotropic 'vn', σ ∝ ⟨D⟩ (QTI mean tensor)  ->  tensor_MD_dMRI.nii.gz

The ONLY thing that differs between the DTI and MD-dMRI models is the input tensor (QTI ⟨D⟩ vs
single-shell DTI); the 'vn' mapping, mesh, electrodes, caps, and FEM are identical standard SimNIBS.
Alternatives considered (free-water elimination, magnitude preservation, μFA) are kept under
pipeline/internal/ for reference and are not part of the model.

Montage: C3 (anode, left M1) → Fp2 (cathode, right supraorbital), 2 mA.

Usage (run in WORK_DIR; SimNIBS resolves subpath/pathfem relative to CWD):
  simnibs_python pipeline/03_run_simulations.py                # all models
  simnibs_python pipeline/03_run_simulations.py MD-dMRI        # re-run a single model only
  simnibs_python pipeline/03_run_simulations.py ISO DTI        # a subset

Requirements: m2m_<subject>/ (CHARM); dwi2cond output (DTI); tensor_MD_dMRI.nii.gz (02).
NOTE: do NOT pass fn_tensor_nifti with tms_flex_opt/tes_flex_opt — documented SimNIBS bug.
"""
import os
import sys

# macOS Apple Silicon: prevent the OpenMP duplicate-runtime crash. MUST be set BEFORE
# `import simnibs` triggers OpenMP initialisation. No-ops on Linux/Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg  # noqa: E402  (paths/subject from config/config.sh)

import simnibs            # noqa: E402
import nibabel as nib     # noqa: E402
import numpy as np        # noqa: E402

WDIR      = cfg["WORK_DIR"]
SUBPATH   = f'm2m_{cfg["SUBJECT"]}'
TENSOR_MD = os.path.join(WDIR, "tensor_MD_dMRI.nii.gz")   # σ ∝ ⟨D⟩ (QTI mean tensor), standard 'vn'

# Anisotropy caps applied IDENTICALLY to every 'vn' (anisotropic) model so a DTI↔MD-dMRI E-field
# difference reflects the tensor source, not the clip. SimNIBS has TWO separate caps:
#   aniso_maxratio = 10 : eigenvalue RATIO cap (the BINDING one), top of the 7–10:1 ex vivo WM
#       range (Nicholson 1965 Exp Neurol 13:386; Ranck & BeMent 1965 Exp Neurol 11:451).
#   aniso_maxcond  = 2  : eigenvalue MAGNITUDE cap (S/m, SimNIBS default), non-binding under vn.
ANISO_MAXRATIO = 10
ANISO_MAXCOND  = 2

# Data-driven model table. `tensor` = explicit conductivity-tensor NIfTI set on the SESSION;
# None → ISO (scalar) or DTI ('vn' uses the dwi2cond tensor SimNIBS finds inside m2m).
MODELS = [
    {"name": "ISO",     "pathfem": "sim_ISO",     "anisotropy": "scalar", "tensor": None},
    {"name": "DTI",     "pathfem": "sim_DTI",     "anisotropy": "vn",     "tensor": None},
    {"name": "MD-dMRI", "pathfem": "sim_MD_dMRI", "anisotropy": "vn",     "tensor": TENSOR_MD},
]


def make_electrode_pair(tdcs):
    """C3 anode, Fp2 cathode — 5×5 cm rectangular pads."""
    for channel, centre in ((1, "C3"), (2, "Fp2")):
        e = tdcs.add_electrode()
        e.channelnr  = channel
        e.centre     = centre
        e.shape      = "rect"
        e.dimensions = [50, 50]
        e.thickness  = 4
    return tdcs


def validate_tensor(path):
    """Assert the conductivity tensor is T1-space, [X,Y,Z,6] FSL order. Aborts on mismatch."""
    t  = nib.load(path)
    t1 = nib.load(os.path.join(SUBPATH, "T1.nii.gz"))
    assert t.shape[:3] == t1.shape[:3], f"Tensor shape {t.shape[:3]} ≠ T1 {t1.shape[:3]} (wrong space)"
    assert t.shape[3] == 6, f"Tensor 4th dim {t.shape[3]} ≠ 6 (FSL dtifit order)"
    assert np.allclose(t.affine, t1.affine, atol=1e-3), "Tensor affine ≠ T1 — tensor in wrong space!"
    print(f"  tensor validation: shape={t.shape} ✓  affine matches T1 ✓")


def run_model(m):
    """Build and run one SESSION for model dict `m`. Returns None on success, else the exception."""
    print("=" * 60); print(f"MODEL: {m['name']}  ->  {m['pathfem']}"); print("=" * 60)
    s = simnibs.sim_struct.SESSION()
    s.subpath = SUBPATH
    s.pathfem = m["pathfem"]
    if m["tensor"] is not None:
        if not os.path.exists(m["tensor"]):
            return FileNotFoundError(f"{m['tensor']} not found — run 02_build_conductivity_tensor.py")
        validate_tensor(m["tensor"])
        # CRITICAL: set fname_tensor on the SESSION, NOT fn_tensor_nifti on the TDCSLIST.
        # SESSION._prepare() unconditionally overwrites PL.fn_tensor_nifti = self.fname_tensor
        # (sim_struct.py ~line 209), so a TDCSLIST attribute is silently ignored.
        s.fname_tensor = m["tensor"]
    tdcs = s.add_tdcslist()
    tdcs.currents        = [0.002, -0.002]        # 2 mA
    tdcs.anisotropy_type = m["anisotropy"]
    if m["anisotropy"] == "vn":
        tdcs.aniso_maxratio = ANISO_MAXRATIO
        tdcs.aniso_maxcond  = ANISO_MAXCOND
    make_electrode_pair(tdcs)
    try:
        simnibs.run_simnibs(s)
        print(f"{m['name']} complete -> {m['pathfem']}/\n")
        return None
    except Exception as e:                          # noqa: BLE001 — report and continue other models
        print(f"ERROR in {m['name']}: {e}\n")
        return e


def main():
    if not os.path.isdir(os.path.join(WDIR, SUBPATH)):
        raise FileNotFoundError(f"Head model not found: {SUBPATH} — run CHARM first.")
    os.chdir(WDIR)                                   # SimNIBS resolves subpath/pathfem relative to CWD
    print(f"Working directory: {os.getcwd()}")

    # Optional CLI subset (e.g. "MD-dMRI" to re-run one model); default = all models.
    by_name   = {m["name"]: m for m in MODELS}
    requested = sys.argv[1:] or list(by_name)
    unknown   = [r for r in requested if r not in by_name]
    if unknown:
        raise SystemExit(f"Unknown model(s) {unknown}. Choose from: {', '.join(by_name)}")
    to_run = [by_name[r] for r in requested]
    print(f"Running {len(to_run)} model(s): {', '.join(requested)}")

    errors = {m["name"]: err for m in to_run if (err := run_model(m)) is not None}

    print("=" * 60)
    done = [m["name"] for m in to_run if m["name"] not in errors]
    if done:
        print(f"Completed ({len(done)}/{len(to_run)}): {', '.join(done)}")
        for m in to_run:
            if m["name"] in done:
                print(f"  {WDIR}/{m['pathfem']}/")
    if errors:
        print(f"\nFailed ({len(errors)}/{len(to_run)}):")
        for name, exc in errors.items():
            print(f"  {name}: {exc}")
        sys.exit(1)                                  # signal failure to the shell after running all
    print("\nAll requested simulations complete.\nNext: analysis/04_extract_roi_efield.py")


if __name__ == "__main__":
    main()
