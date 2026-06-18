"""
04_run_simulations.py - tDCS simulations over (montage x conductivity-model) for one subject.

Conductivity models (montage-independent; only the input tensor differs):
  ISO      - isotropic scalar conductivity (literature values)
  DTI      - anisotropic 'vn' from dwi2cond (FA-based classical baseline)
  MD-dMRI  - anisotropic 'vn', sigma proportional to <D> (QTI mean tensor) -> tensor_MD_dMRI.nii.gz

Montages. Each target gets a conventional pad and a focal 4x1 HD version, same +2 mA total dose
(HD vs pad = same dose, different focality):
  M1        - C3 anode / Fp2 cathode, 5x5 cm pads, 2 mA  (primary montage)
  DLPFC     - F3 anode / Fp2 cathode, 5x5 cm pads, 2 mA
  HD_M1     - 4x1 ring: centre C3 (+2 mA), returns Cz/F3/T7/P3 (-0.5 mA each), ~1 cm discs
  HD_DLPFC  - 4x1 ring: centre F3 (+2 mA), returns Fp1/Fz/C3/F7 (-0.5 mA each), ~1 cm discs
Left M1 (C3) is fixed across all subjects to avoid montage-laterality confounding the conductivity-model
contrast (C4 mirror optional). 4x1 ring: Datta et al. 2009 Brain Stimul 2:201; Villamar et al. 2013
(JoVE; J Pain 14:371).

Output dirs are namespaced: sim_<montage>_<model>/ (e.g. sim_DLPFC_MD-dMRI). analysis/04_extract_roi_efield
reads this convention. Free-water elimination was tested and dropped (null); see the FWE archive notes.

Usage (run in WORK_DIR):
  simnibs_python pipeline/04_run_simulations.py                              # all montages x all models
  simnibs_python pipeline/04_run_simulations.py --montage DLPFC              # one montage, all models
  simnibs_python pipeline/04_run_simulations.py --montage M1 --model MD-dMRI # one montage, one model
NOTE: do NOT pass fn_tensor_nifti with tms_flex_opt/tes_flex_opt - documented SimNIBS bug.
"""
import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg  # noqa: E402
import simnibs            # noqa: E402
import nibabel as nib     # noqa: E402
import numpy as np        # noqa: E402

WDIR    = cfg["WORK_DIR"]
SUBPATH = f'm2m_{cfg["SUBJECT"]}'
TENSOR_MD = os.path.join(WDIR, "tensor_MD_dMRI.nii.gz")
ANISO_MAXRATIO, ANISO_MAXCOND = 10, 2                  # SimNIBS defaults, identical for both 'vn' models

MODELS = {                                             # name -> (anisotropy_type, tensor or None)
    "ISO":     ("scalar", None),
    "DTI":     ("vn",     None),                       # dwi2cond tensor found inside m2m
    "MD-dMRI": ("vn",     TENSOR_MD),                  # sigma proportional to <D>
}


def pad(centre, channelnr):
    return dict(centre=centre, channelnr=channelnr, shape="rect", dimensions=[50, 50], thickness=4)


def disc(centre, channelnr):
    return dict(centre=centre, channelnr=channelnr, shape="ellipse", dimensions=[10, 10], thickness=4)


# 4x1 HD: centre carries the full +2 mA, four returns -0.5 mA each (sum to zero, Kirchhoff).
HD = [0.002, -0.0005, -0.0005, -0.0005, -0.0005]
MONTAGES = {
    "M1":       dict(currents=[0.002, -0.002], electrodes=[pad("C3", 1), pad("Fp2", 2)]),
    "DLPFC":    dict(currents=[0.002, -0.002], electrodes=[pad("F3", 1), pad("Fp2", 2)]),
    "HD_M1":    dict(currents=HD, electrodes=[disc("C3", 1), disc("Cz", 2), disc("F3", 3),
                                              disc("T7", 4), disc("P3", 5)]),
    "HD_DLPFC": dict(currents=HD, electrodes=[disc("F3", 1), disc("Fp1", 2), disc("Fz", 3),
                                              disc("C3", 4), disc("F7", 5)]),
}


def validate_tensor(path):
    t, t1 = nib.load(path), nib.load(os.path.join(SUBPATH, "T1.nii.gz"))
    assert t.shape[:3] == t1.shape[:3], f"Tensor shape {t.shape[:3]} != T1 {t1.shape[:3]} (wrong space)"
    assert t.shape[3] == 6, f"Tensor 4th dim {t.shape[3]} != 6 (FSL order)"
    assert np.allclose(t.affine, t1.affine, atol=1e-3), "Tensor affine != T1 - wrong space!"


def run(montage_name, model_name):
    mont = MONTAGES[montage_name]
    aniso, tensor = MODELS[model_name]
    pathfem = f"sim_{montage_name}_{model_name.replace('-', '_')}"   # token matches 04_extract lookup
    print(f"MONTAGE {montage_name}  x  MODEL {model_name}  ->  {pathfem}")
    s = simnibs.sim_struct.SESSION(); s.subpath = SUBPATH; s.pathfem = pathfem
    s.open_in_gmsh = False                             # headless/batch: never auto-open GMSH (it hangs without a display)
    if tensor is not None:
        if not os.path.exists(tensor):
            return FileNotFoundError(f"{tensor} not found - run 03_build_conductivity_tensor.py")
        validate_tensor(tensor)
        s.fname_tensor = tensor                        # on the SESSION, not the TDCSLIST (overwrite trap)
    tdcs = s.add_tdcslist()
    tdcs.currents = mont["currents"]
    tdcs.anisotropy_type = aniso
    if aniso == "vn":
        tdcs.aniso_maxratio, tdcs.aniso_maxcond = ANISO_MAXRATIO, ANISO_MAXCOND
    for spec in mont["electrodes"]:
        e = tdcs.add_electrode()
        e.channelnr, e.centre = spec["channelnr"], spec["centre"]
        e.shape, e.dimensions, e.thickness = spec["shape"], spec["dimensions"], spec["thickness"]
    try:
        simnibs.run_simnibs(s); print(f"{montage_name}/{model_name} complete\n"); return None
    except Exception as e:                             # noqa: BLE001
        print(f"ERROR {montage_name}/{model_name}: {e}\n"); return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", nargs="*", default=None, help="montage(s); default = all")
    ap.add_argument("--model", nargs="*", default=None, dest="models", help="model(s); default = all")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(WDIR, SUBPATH)):
        raise FileNotFoundError(f"Head model not found: {SUBPATH} - run CHARM first.")
    os.chdir(WDIR)

    montages = args.montage or list(MONTAGES)
    models = args.models or list(MODELS)
    for bad, pool in ((set(montages) - set(MONTAGES), "montages"), (set(models) - set(MODELS), "models")):
        if bad:
            raise SystemExit(f"Unknown {pool}: {bad}. Choose from {list(MONTAGES if pool=='montages' else MODELS)}")

    errors = {}
    for mont in montages:
        for model in models:
            err = run(mont, model)
            if err is not None:
                errors[f"{mont}/{model}"] = err

    if errors:
        print(f"Failed ({len(errors)}):")
        for k, e in errors.items():
            print(f"  {k}: {e}")
        sys.exit(1)
    print("All requested simulations complete.\nNext: analysis/04_extract_roi_efield.py --montage <name>")


if __name__ == "__main__":
    main()
