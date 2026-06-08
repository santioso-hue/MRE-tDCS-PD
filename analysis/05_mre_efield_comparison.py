"""
05_mre_efield_comparison.py — Post-hoc cross-modal comparison: MRE mechanics vs
MD-dMRI microstructure and the tDCS E-field, per ROI.

Purpose (stated honestly, see pipeline/conductivity_models_derivation.md):
  This is NOT a novel correlation — Olsson et al. already related MD-dMRI microstructure
  to MRE mechanics in a cohort, and our conductivity is a deterministic function of the
  diffusion tensor, so "conductivity vs mechanics" ~ "diffusion vs mechanics". This script
  serves two supporting roles:
    (1) Consistency / QC — does our subject reproduce the known microstructure<->mechanics
        relationship (MD vs stiffness negative, uFA vs stiffness positive)?  A data-quality
        check, not a new biological claim.
    (2) Relevance map — does the conductivity model's IMPACT on the E-field
        (dE = E_MD-dMRI - E_DTI, a local quantity where the montage largely cancels)
        land in the regions Olsson flags as tissue-altered in PD?

Designed for the cohort: extract_subject() returns a per-ROI table for one subject; the
main loop concatenates over subjects. With one subject the correlation is ACROSS ROIs
(n = number of ROIs) — a proof of concept, not a statistical result.

Run:  cd <repo>;  simnibs_python analysis/05_mre_efield_comparison.py
"""
import os
import sys
import csv
import numpy as np
from simnibs import mesh_io
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rois import (  # noqa: E402
    load_labeled, sample_volume_medians, sample_tensor_aniso_medians, assign_mesh_labels)

WORK = cfg["WORK_DIR"]; M2M = cfg["M2M_DIR"]; REG = cfg["REG_DIR"]

# ROIs are the FastSurfer-derived masks in mesh space (registration/fastsurfer_rois/), shared with
# 04 through _rois.py: cortical and WM lobes, corpus callosum, and aseg subcortical structures. Each
# ROI is sampled over all of its voxels/elements, so the median reflects the whole structure.
TENSOR = os.path.join(WORK, "tensor_MD_dMRI.nii.gz")

VOLUME_MAPS = {  # name -> T1-space NIfTI (microstructure + mechanics)
    "stiffness": f"{REG}/mre_stiffness_T1.nii.gz",
    "storage":   f"{REG}/mre_storage_T1.nii.gz",
    "loss":      f"{REG}/mre_loss_T1.nii.gz",
    "mre_conf":  f"{REG}/mre_confidence_T1.nii.gz",  # reported for reference, not used to gate
    "MD":        f"{REG}/MD_T1.nii.gz",
    "uFA":       f"{REG}/C_mu_T1.nii.gz",
}
MESHES = {  # E-field per model
    "E_ISO":     (f"{WORK}/sim_ISO/{cfg['SUBJECT']}_TDCS_1_scalar.msh"),
    "E_DTI":     (f"{WORK}/sim_DTI/{cfg['SUBJECT']}_TDCS_1_vn.msh"),
    "E_MDdMRI":  (f"{WORK}/sim_MD_dMRI/{cfg['SUBJECT']}_TDCS_1_vn.msh"),
}


def extract_subject():
    """Return {roi: {quantity: value}} for the configured subject, sampling each
    atlas ROI mask over all its voxels/elements (no spheres)."""
    labeled, lab_aff, names = load_labeled(REG)
    rows = {names[k]: {} for k in names}

    # scalar microstructure + mechanics maps: median within each ROI
    for mname, p in VOLUME_MAPS.items():
        if os.path.exists(p):
            for roi, val in sample_volume_medians(p, labeled, lab_aff, names).items():
                rows[roi][mname] = val

    # conductivity-tensor anisotropy (lambda1/lambda3) within each ROI
    if os.path.exists(TENSOR):
        for roi, val in sample_tensor_aniso_medians(TENSOR, labeled, lab_aff, names).items():
            rows[roi]["cond_aniso"] = val

    # viscosity phi = atan(G''/G')
    for r in rows.values():
        if np.isfinite(r.get("loss", np.nan)) and r.get("storage", 0):
            r["viscosity"] = float(np.degrees(np.arctan2(r["loss"], r["storage"])))

    # E-field per model: median over GM+WM elements whose barycentre lands in the ROI
    for m, path in MESHES.items():
        if not os.path.exists(path):
            continue
        msh = mesh_io.read_msh(path)
        elab = assign_mesh_labels(msh.elements_baricenters().value, labeled, lab_aff)
        gmwm = (msh.elm.tag1 == 1) | (msh.elm.tag1 == 2)
        ef = msh.field['magnE'].value
        for k, n in names.items():
            sel = (elab == k) & gmwm
            rows[n][m] = float(np.median(ef[sel])) if sel.any() else np.nan

    # model impact on the E-field (local quantity; the montage largely cancels)
    for r in rows.values():
        if np.isfinite(r.get("E_MDdMRI", np.nan)) and np.isfinite(r.get("E_DTI", np.nan)) and r.get("E_DTI", 0):
            r["dE_model"] = 100 * (r["E_MDdMRI"] - r["E_DTI"]) / r["E_DTI"]
    return rows


def main():
    rows = extract_subject()
    keys = ["stiffness", "viscosity", "MD", "uFA", "cond_aniso",
            "E_ISO", "E_DTI", "E_MDdMRI", "dE_model"]
    # write per-ROI table
    out_csv = os.path.join(os.path.dirname(__file__), "results", "mre_efield_per_roi.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ROI"] + keys)
        for roi, r in rows.items():
            w.writerow([roi] + [f"{r.get(k, np.nan):.4g}" for k in keys])
    print(f"Per-ROI table -> {out_csv}\n")

    roi_names = list(rows.keys())

    def col(k):
        return np.array([rows[r].get(k, np.nan) for r in roi_names])

    print("="*70)
    print(f"ACROSS-ROI consistency check (n={len(rows)} ROIs, single HC subject)")
    print("Expected from Olsson et al.: MD vs stiffness NEGATIVE; uFA vs stiffness POSITIVE")
    print("="*70)
    for a, b, exp in [("MD", "stiffness", "neg"), ("uFA", "stiffness", "pos"),
                      ("MD", "viscosity", "?"), ("cond_aniso", "stiffness", "pos")]:
        x, y = col(a), col(b); m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 4:
            rho, p = spearmanr(x[m], y[m])
            print(f"  {a:11s} vs {b:10s}: Spearman rho={rho:+.2f} (p={p:.2f}, n={m.sum()})  [expect {exp}]")

    print("\n"+"="*70)
    print("RELEVANCE: where does the conductivity model change the E-field most,")
    print("and does it track tissue alteration (stiffness / free water)?")
    print("="*70)
    for a in ["stiffness", "MD"]:
        x, y = col("dE_model"), col(a); m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 4:
            rho, p = spearmanr(x[m], y[m])
            print(f"  dE(MD-dMRI vs DTI) vs {a:10s}: rho={rho:+.2f} (p={p:.2f}, n={m.sum()})")
    print("\nNOTE: n=1 subject -> these are across-region trends (proof of concept), not")
    print("statistical results. The same script aggregates over subjects for the cohort.")


if __name__ == "__main__":
    main()
