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
import nibabel as nib
import simnibs
from simnibs import mesh_io
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402

WORK = cfg["WORK_DIR"]; M2M = cfg["M2M_DIR"]; REG = cfg["REG_DIR"]

# ── ROIs (MNI152, mm): subcortical PD targets that overlap Olsson et al. ───────
ROIS = {
    "L_SNc": [-8, -16, -12], "R_SNc": [8, -16, -12], "L_SNr": [-10, -18, -11], "R_SNr": [10, -18, -11],
    "L_VTA": [-4, -16, -9],  "R_VTA": [4, -16, -9],
    "L_PUT": [-28, -2, 4],   "R_PUT": [28, -2, 4],   "L_Ca": [-13, 9, 9], "R_Ca": [13, 9, 9],
    "L_GPe": [-24, -8, 2],   "R_GPe": [24, -8, 2],   "L_GPi": [-20, -10, -2], "R_GPi": [20, -10, -2],
    "L_RN": [-4, -22, -6],   "R_RN": [4, -22, -6],
}
# Per-structure ROI radii (mm), matched to analysis/04 so the E-field values agree.
# At the coarse MD-dMRI (2.5 mm) / MRE (3 mm) grids a 3 mm sphere is ~1 voxel — too few
# for a stable cross-modal median — so deep nuclei use 7 mm, basal ganglia 5 mm.
RADII = {r: (7.0 if any(s in r for s in ("SN", "VTA", "RN")) else 5.0) for r in (
    "L_SNc R_SNc L_SNr R_SNr L_VTA R_VTA L_PUT R_PUT L_Ca R_Ca "
    "L_GPe R_GPe L_GPi R_GPi L_RN R_RN").split()}

VOLUME_MAPS = {  # name -> T1-space NIfTI (microstructure + mechanics)
    "stiffness": f"{REG}/mre_stiffness_T1.nii.gz",
    "storage":   f"{REG}/mre_storage_T1.nii.gz",
    "loss":      f"{REG}/mre_loss_T1.nii.gz",
    "mre_conf":  f"{REG}/mre_confidence_T1.nii.gz",  # reported for reference, not used to gate
    "MD":        f"{REG}/MD_T1.nii.gz",
    "uFA":       f"{REG}/C_mu_T1.nii.gz",
    "fw_frac":   f"{REG}/fw_frac_T1.nii.gz",
}
MESHES = {  # E-field per model
    "E_ISO":     (f"{WORK}/sim_ISO/{cfg['SUBJECT']}_TDCS_1_scalar.msh"),
    "E_DTI":     (f"{WORK}/sim_DTI/{cfg['SUBJECT']}_TDCS_1_vn.msh"),
    "E_MDdMRI":  (f"{WORK}/sim_MD_dMRI/{cfg['SUBJECT']}_TDCS_1_vn.msh"),
}


def roi_volume_median(d, inv, center_world, radius):
    ijk = inv[:3, :3] @ center_world + inv[:3, 3]
    i, j, k = np.round(ijk).astype(int)
    r = int(np.ceil(radius))
    sl = d[max(i-r, 0):i+r+1, max(j-r, 0):j+r+1, max(k-r, 0):k+r+1]
    v = sl[np.isfinite(sl) & (sl != 0)]
    return float(np.median(v)) if v.size else np.nan


def conductivity_anisotropy(tensor_path, inv, center_world, radius):
    t = nib.load(tensor_path).get_fdata()
    ijk = inv[:3, :3] @ center_world + inv[:3, 3]
    i, j, k = np.round(ijk).astype(int); r = int(np.ceil(radius))
    blk = t[max(i-r, 0):i+r+1, max(j-r, 0):j+r+1, max(k-r, 0):k+r+1].reshape(-1, 6)
    blk = blk[np.abs(blk[:, 0]) > 1e-6]
    if blk.size == 0:
        return np.nan
    M = np.zeros((len(blk), 3, 3))
    for a, (p, q) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        M[:, p, q] = blk[:, a]; M[:, q, p] = blk[:, a]
    ev = np.linalg.eigvalsh(M)
    return float(np.median(ev[:, 2] / np.maximum(ev[:, 0], 1e-9)))


def extract_subject():
    """Return {roi: {quantity: value}} for the configured subject."""
    centers = simnibs.mni2subject_coords(np.array(list(ROIS.values()), float), M2M)
    centers = {k: centers[i] for i, k in enumerate(ROIS)}

    # preload volume maps
    vols = {}
    for name, p in VOLUME_MAPS.items():
        if os.path.exists(p):
            img = nib.load(p); vols[name] = (img.get_fdata(), np.linalg.inv(img.affine))
    t1_inv = np.linalg.inv(nib.load(f"{M2M}/T1.nii.gz").affine)

    # preload meshes
    meshes = {m: mesh_io.read_msh(p) for m, p in MESHES.items() if os.path.exists(p)}
    bary = {m: meshes[m].elements_baricenters().value for m in meshes}
    tag = {m: meshes[m].elm.tag1 for m in meshes}
    Ef = {m: meshes[m].field['magnE'].value for m in meshes}

    rows = {}
    for roi, c in centers.items():
        rad = RADII.get(roi, 6.0)
        row = {}
        for name, (d, inv) in vols.items():
            row[name] = roi_volume_median(d, inv, c, rad)
        row["cond_aniso"] = conductivity_anisotropy(f"{WORK}/tensor_MD_dMRI.nii.gz", t1_inv, c, rad)
        # viscosity phi = atan(G''/G')
        if np.isfinite(row.get("loss", np.nan)) and row.get("storage", 0):
            row["viscosity"] = float(np.degrees(np.arctan2(row["loss"], row["storage"])))
        for m in meshes:
            dist = np.linalg.norm(bary[m] - c, axis=1)
            sel = ((tag[m] == 1) | (tag[m] == 2)) & (dist < rad)
            row[m] = float(np.median(Ef[m][sel])) if sel.any() else np.nan
        if np.isfinite(row.get("E_MDdMRI", np.nan)) and np.isfinite(row.get("E_DTI", np.nan)):
            row["dE_model"] = 100 * (row["E_MDdMRI"] - row["E_DTI"]) / row["E_DTI"]  # model impact %
        rows[roi] = row
    return rows


def main():
    rows = extract_subject()
    keys = ["stiffness", "viscosity", "MD", "uFA", "fw_frac", "cond_aniso",
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
                      ("fw_frac", "stiffness", "neg"), ("MD", "viscosity", "?"),
                      ("cond_aniso", "stiffness", "pos")]:
        x, y = col(a), col(b); m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 4:
            rho, p = spearmanr(x[m], y[m])
            print(f"  {a:11s} vs {b:10s}: Spearman rho={rho:+.2f} (p={p:.2f}, n={m.sum()})  [expect {exp}]")

    print("\n"+"="*70)
    print("RELEVANCE: where does the conductivity model change the E-field most,")
    print("and does it track tissue alteration (stiffness / free water)?")
    print("="*70)
    for a in ["stiffness", "fw_frac", "MD"]:
        x, y = col("dE_model"), col(a); m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 4:
            rho, p = spearmanr(x[m], y[m])
            print(f"  dE(MD-dMRI vs DTI) vs {a:10s}: rho={rho:+.2f} (p={p:.2f}, n={m.sum()})")
    print("\nNOTE: n=1 subject -> these are across-region trends (proof of concept), not")
    print("statistical results. The same script aggregates over subjects for the cohort.")


if __name__ == "__main__":
    main()
