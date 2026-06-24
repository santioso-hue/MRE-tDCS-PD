"""05_mre_efield_comparison.py -- per-ROI comparison: MRE mechanics vs MD-dMRI microstructure vs tDCS E-field.

Two outputs (see the manuscript Methods): (1) consistency QC -- does the subject
reproduce MD vs stiffness negative, uFA vs stiffness positive? (2) relevance map -- does the model's
E-field impact (dE_model_pct = 100*(E_MD-dMRI - E_DTI)/E_DTI, a local PERCENT difference where the shared
montage field cancels, so it isolates the conductivity model) land where Olsson flags tissue alteration?
dE_model_pct is NaN until the DTI arm (ParkMRE_DTI) lands.

MRE gating: cohort has no per-voxel confidence map (QC was subject-level, not per-voxel), so MRE
maps are sampled over brain EXCLUDING the CSF-adjacent cortical surface where the Helmholtz inversion is
unreliable (Olsson); gated is primary, ungated is the sensitivity row. The gate does NOT touch MD/uFA
(QTI) or the E-field (FEM). E-field statistic is MEDIAN per ROI (Olsson convention); p95 is a sensitivity column.

Run:  PIPELINE_CONFIG=<subject config.sh> simnibs_python analysis/05_mre_efield_comparison.py
"""
import os, sys, csv
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from simnibs import mesh_io
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from _config import cfg  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rois import (load_labeled, sample_volume_medians, sample_tensor_aniso_medians,  # noqa: E402
                   sample_tensor_fa_medians, assign_mesh_labels)
from _sims import sim_mesh  # noqa: E402  (shared montage-aware mesh lookup)
from _stats import pct_delta  # noqa: E402

WORK, M2M, REG = cfg["WORK_DIR"], cfg["M2M_DIR"], cfg["REG_DIR"]
TENSOR = os.path.join(WORK, "tensor_MD_dMRI.nii.gz")
MRE_MAPS = {"stiffness": f"{REG}/mre_stiffness_T1.nii.gz",
            "alpha":     f"{REG}/mre_alpha_T1.nii.gz"}     # springpot exponent (elastic<->viscous)
MICRO_MAPS = {"MD": f"{REG}/MD_T1.nii.gz", "uFA": f"{REG}/uFA_T1.nii.gz"}   # NOT gated (QTI, not MRE)
MONTAGE = os.environ.get("MONTAGE", "M1")   # 05 reads one montage (default M1); set MONTAGE to switch
MESHES = {f"E_{mdl.replace('-', '')}": sim_mesh(WORK, MONTAGE, mdl, cfg["SUBJECT"])
          for mdl in ("ISO", "DTI", "MD-dMRI")}   # E_ISO / E_DTI / E_MDdMRI -> mesh path (or None)


def build_gate():
    """MRE gate: no per-voxel confidence map, so drop only the CSF-adjacent cortical surface where the
    Helmholtz inversion is unreliable (Olsson). Returns (gate, fraction of brain kept)."""
    seg = np.asarray(nib.load(os.path.join(M2M, "final_tissues.nii.gz")).dataobj)
    seg = seg[..., 0] if seg.ndim == 4 else seg
    brain = np.isin(seg, [1, 2, 3])
    csf_adj = ndi.binary_dilation(seg == 3, iterations=1) & (seg == 2)
    gate = ~csf_adj
    return gate, float((gate & brain).sum() / max(int(brain.sum()), 1))


def load_tensor_divergence():
    """Per-ROI (V1 angle, delta lam1/lam3) from 08_tensor_divergence.py's CSV, if present. Absent until 08
    has run (gated on the DTI arm); returns {} so the correlations degrade gracefully."""
    p = os.path.join(os.path.dirname(__file__), "results", cfg["SUBJECT"], "tensor_divergence.csv")
    out = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            try:
                out[r["ROI"]] = (float(r["angle_med"]), float(r["dR1_med"]))
            except (KeyError, ValueError):
                pass
    return out


def extract_subject(gate):
    labeled, lab_aff, names = load_labeled(REG)
    rows = {names[k]: {} for k in names}
    # MRE maps: gated (primary) + ungated (sensitivity, "_ung")
    for nm, p in MRE_MAPS.items():
        if not os.path.exists(p):
            continue
        for roi, val in sample_volume_medians(p, labeled, lab_aff, names, gate=gate).items():
            rows[roi][nm] = val
        for roi, val in sample_volume_medians(p, labeled, lab_aff, names).items():
            rows[roi][nm + "_ung"] = val
    for nm, p in MICRO_MAPS.items():                               # QTI: never gated
        if os.path.exists(p):
            for roi, val in sample_volume_medians(p, labeled, lab_aff, names).items():
                rows[roi][nm] = val
    if os.path.exists(TENSOR):
        for roi, val in sample_tensor_aniso_medians(TENSOR, labeled, lab_aff, names).items():
            rows[roi]["cond_aniso"] = val
        for roi, val in sample_tensor_fa_medians(TENSOR, labeled, lab_aff, names).items():
            rows[roi]["FA_meanD"] = val
    # tensor-divergence (08) per ROI, if 08 has run (gated on the DTI arm)
    for roi, (ang, dr1) in load_tensor_divergence().items():
        if roi in rows:
            rows[roi]["tdiv_angle"] = ang; rows[roi]["tdiv_dR1"] = dr1
    # E-field per model: median (primary) + p95 (sensitivity) over GM+WM elements in the ROI. The
    # whole-brain median is captured here so each mesh is read once, not re-read in the WholeBrain block.
    wb_efield = {}
    for m, path in MESHES.items():
        if not path or not os.path.exists(path):
            continue
        msh = mesh_io.read_msh(path)
        elab = assign_mesh_labels(msh.elements_baricenters().value, labeled, lab_aff)
        gmwm = (msh.elm.tag1 == 1) | (msh.elm.tag1 == 2)
        ef = msh.field["magnE"].value
        wb_efield[m] = float(np.median(ef[gmwm])) if gmwm.any() else np.nan
        for k, n in names.items():
            sel = (elab == k) & gmwm
            rows[n][m] = float(np.median(ef[sel])) if sel.any() else np.nan
            rows[n][m + "_p95"] = float(np.percentile(ef[sel], 95)) if sel.any() else np.nan
    for r in rows.values():
        if np.isfinite(r.get("E_MDdMRI", np.nan)) and r.get("E_DTI", 0):
            r["dE_model_pct"] = pct_delta(r["E_MDdMRI"], r["E_DTI"])   # PERCENT, not V/m
        # microstructure divergence: uFA (microscopic anisotropy) minus FA(<D>) (macroscopic). Large in
        # crossing/dispersed-fiber voxels, where single-shell DTI FA conflates micro-anisotropy with
        # orientation dispersion. uFA is an EXPLANATION variable here, NOT a conductivity input.
        if np.isfinite(r.get("uFA", np.nan)) and np.isfinite(r.get("FA_meanD", np.nan)):
            r["uFA_minus_FA"] = r["uFA"] - r["FA_meanD"]
    # WholeBrain: union of all ROI labels (GM+WM), sampled with the same CSF-excluding gate.
    wb_lab = np.where(labeled > 0, 1, 0).astype(labeled.dtype)
    wb_names = {1: "WholeBrain"}
    wb = {}
    for nm, p in MRE_MAPS.items():
        if os.path.exists(p):
            wb[nm] = sample_volume_medians(p, wb_lab, lab_aff, wb_names, gate=gate).get("WholeBrain", np.nan)
            wb[nm + "_ung"] = sample_volume_medians(p, wb_lab, lab_aff, wb_names).get("WholeBrain", np.nan)
    for nm, p in MICRO_MAPS.items():
        if os.path.exists(p):
            wb[nm] = sample_volume_medians(p, wb_lab, lab_aff, wb_names).get("WholeBrain", np.nan)
    if os.path.exists(TENSOR):
        wb["cond_aniso"] = sample_tensor_aniso_medians(TENSOR, wb_lab, lab_aff, wb_names).get("WholeBrain", np.nan)
        wb["FA_meanD"] = sample_tensor_fa_medians(TENSOR, wb_lab, lab_aff, wb_names).get("WholeBrain", np.nan)
    for m in MESHES:
        if m in wb_efield:
            wb[m] = wb_efield[m]
    if np.isfinite(wb.get("E_MDdMRI", np.nan)) and wb.get("E_DTI", 0):
        wb["dE_model_pct"] = pct_delta(wb["E_MDdMRI"], wb["E_DTI"])
    if np.isfinite(wb.get("uFA", np.nan)) and np.isfinite(wb.get("FA_meanD", np.nan)):
        wb["uFA_minus_FA"] = wb["uFA"] - wb["FA_meanD"]
    rows["WholeBrain"] = wb
    return rows


def main():
    gate, frac = build_gate()
    print(f"MRE gate: no per-voxel confidence map (cohort QC is subject-level); CSF-adjacent "
          f"cortex excluded -> {100*frac:.1f}% of brain kept")
    rows = extract_subject(gate)

    out_csv = os.path.join(os.path.dirname(__file__), "results", cfg["SUBJECT"], "mre_efield_per_roi.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = ["stiffness", "stiffness_ung", "alpha", "alpha_ung", "MD", "uFA", "FA_meanD", "uFA_minus_FA",
            "cond_aniso", "E_ISO", "E_DTI", "E_MDdMRI", "E_ISO_p95", "E_DTI_p95", "E_MDdMRI_p95",
            "dE_model_pct", "tdiv_angle", "tdiv_dR1"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ROI"] + keys)
        for roi, r in rows.items():
            w.writerow([roi] + [f"{r.get(k, np.nan):.4g}" for k in keys])
    print(f"Per-ROI table -> {out_csv}\n")

    names = list(rows.keys())
    col = lambda k: np.array([rows[r].get(k, np.nan) for r in names])

    def corr(a, b):
        x, y = col(a), col(b); m = np.isfinite(x) & np.isfinite(y)
        return (spearmanr(x[m], y[m])[0], int(m.sum())) if m.sum() >= 4 else (np.nan, int(m.sum()))

    print(f"Consistency (n={len(rows)} ROIs, this subject). Expect MD vs stiffness NEG, uFA vs stiffness POS.")
    print(f"  {'pair':<26}{'GATED':>14}{'ungated':>14}")
    for a in ["MD", "uFA", "cond_aniso", "alpha", "dE_model_pct"]:
        rg, ng = corr(a, "stiffness")          # GATED column (CSF-adjacent cortex excluded)
        ru, nu = corr(a, "stiffness_ung")      # ungated sensitivity
        print(f"  {a+' vs stiffness':<26}{f'rho={rg:+.2f}(n={ng})':>14}{f'rho={ru:+.2f}(n={nu})':>14}")

    # uFA-FA divergence vs the model effect: do the conductivity models diverge (tensor divergence + nonzero
    # dE_model) in ROIs where single-shell DTI FA and the QTI uFA disagree most (crossing/dispersed fibers)?
    # uFA is an explanation variable here, not a conductivity input.
    print("\nExplanation layer (uFA-FA divergence vs the model effect; across ROIs, this subject):")
    for b, lbl in [("dE_model_pct", "dE_model_pct = 100*(E_MD-dMRI - E_DTI)/E_DTI"),
                   ("tdiv_angle", "DTI-vs-<D> V1 angle (08)"),
                   ("tdiv_dR1", "delta(lam1/lam3) DTI-vs-<D> (08)")]:
        rho, nn = corr("uFA_minus_FA", b)
        note = "" if nn >= 4 else "  [needs the DTI arm / 08]"
        print(f"  (uFA-FA) vs {lbl:<32}{f'rho={rho:+.2f}(n={nn})':>16}{note}")
    print("  Expected POSITIVE (models diverge where uFA>FA = dispersion).")

    print("\nGATED is primary (Olsson MRE handling); ungated is the sensitivity. n=1 subject: across-region "
          "trend, not a statistical result. The cohort runner aggregates this per subject.")


if __name__ == "__main__":
    main()
