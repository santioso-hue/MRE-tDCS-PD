"""
05_mre_efield_comparison.py — Post-hoc cross-modal comparison: MRE mechanics vs MD-dMRI microstructure
and the tDCS E-field, per ROI. Cohort MRE = stiffness + alpha (springpot exponent); no per-voxel
confidence map, so QC is subject-level (GoodMRE + alphapositive) rather than confidence-gated.

Purpose (see pipeline/conductivity_models_derivation.md):
  (1) Consistency/QC — does the subject reproduce the known microstructure<->mechanics relationship
      (MD vs stiffness negative, uFA vs stiffness positive)?  A data-quality check, not a new claim.
  (2) Relevance map — does the conductivity model's IMPACT on the E-field (dE = E_MD-dMRI - E_DTI,
      a local quantity where the montage largely cancels) land where Olsson flags tissue alteration?

MRE gating: the cohort has no per-voxel confidence map (cohort QC is subject-level), so MRE maps
(stiffness, alpha) are sampled over the brain EXCLUDING the CSF-adjacent cortical surface, where the
Helmholtz inversion is unreliable (per Olsson). The script reports that gated result as primary and the
UNGATED (no exclusion) result as a sensitivity row. The gate does NOT touch MD/uFA (QTI) or the E-field (FEM).
The dE relevance map (item 2) needs the DTI arm (dE = E_MD-dMRI - E_DTI); it is NaN until ParkMRE_DTI lands.

E-field statistic: MEDIAN per ROI (Olsson's ROI convention; item F); p95 reported as a sensitivity column.

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
from _rois import load_labeled, sample_volume_medians, sample_tensor_aniso_medians, assign_mesh_labels  # noqa: E402
from _sims import sim_mesh  # noqa: E402  (shared montage-aware mesh lookup)

WORK, M2M, REG = cfg["WORK_DIR"], cfg["M2M_DIR"], cfg["REG_DIR"]
TENSOR = os.path.join(WORK, "tensor_MD_dMRI.nii.gz")
MRE_MAPS = {"stiffness": f"{REG}/mre_stiffness_T1.nii.gz",
            "alpha":     f"{REG}/mre_alpha_T1.nii.gz"}     # springpot exponent (elastic<->viscous)
MICRO_MAPS = {"MD": f"{REG}/MD_T1.nii.gz", "uFA": f"{REG}/uFA_T1.nii.gz"}   # NOT gated (QTI, not MRE)
MONTAGE = os.environ.get("MONTAGE", "M1")   # 05 reads one montage (default M1); set MONTAGE to switch
MESHES = {f"E_{mdl.replace('-', '')}": sim_mesh(WORK, MONTAGE, mdl, cfg["SUBJECT"])
          for mdl in ("ISO", "DTI", "MD-dMRI")}   # E_ISO / E_DTI / E_MDdMRI -> mesh path (or None)


def build_gate():
    """Cohort MRE gate. There is NO per-voxel confidence map (the cohort QC is subject-level:
    GoodMRE + alphapositive), so we only drop the CSF-adjacent cortical surface, where the Helmholtz
    inversion is unreliable (Olsson). Returns (gate, fraction of brain kept)."""
    seg = np.asarray(nib.load(os.path.join(M2M, "final_tissues.nii.gz")).dataobj)
    seg = seg[..., 0] if seg.ndim == 4 else seg
    brain = np.isin(seg, [1, 2, 3])
    csf_adj = ndi.binary_dilation(seg == 3, iterations=1) & (seg == 2)
    gate = ~csf_adj
    return gate, float((gate & brain).sum() / max(int(brain.sum()), 1))


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
    # (alpha, the springpot exponent, is sampled directly above as an MRE map -- no storage/loss arctan needed)
    # E-field per model: median (primary) + p95 (sensitivity) over GM+WM elements in the ROI
    for m, path in MESHES.items():
        if not path or not os.path.exists(path):
            continue
        msh = mesh_io.read_msh(path)
        elab = assign_mesh_labels(msh.elements_baricenters().value, labeled, lab_aff)
        gmwm = (msh.elm.tag1 == 1) | (msh.elm.tag1 == 2)
        ef = msh.field["magnE"].value
        for k, n in names.items():
            sel = (elab == k) & gmwm
            rows[n][m] = float(np.median(ef[sel])) if sel.any() else np.nan
            rows[n][m + "_p95"] = float(np.percentile(ef[sel], 95)) if sel.any() else np.nan
    for r in rows.values():
        if np.isfinite(r.get("E_MDdMRI", np.nan)) and r.get("E_DTI", 0):
            r["dE_model"] = 100 * (r["E_MDdMRI"] - r["E_DTI"]) / r["E_DTI"]
    return rows


def main():
    gate, frac = build_gate()
    print(f"MRE gate: no per-voxel confidence (subject-level GoodMRE+alphapositive QC); CSF-adjacent "
          f"cortex excluded -> {100*frac:.1f}% of brain kept")
    rows = extract_subject(gate)

    out_csv = os.path.join(os.path.dirname(__file__), "results", cfg["SUBJECT"], "mre_efield_per_roi.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = ["stiffness", "stiffness_ung", "alpha", "MD", "uFA", "cond_aniso",
            "E_ISO", "E_DTI", "E_MDdMRI", "E_MDdMRI_p95", "dE_model"]
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

    print(f"Consistency (n={len(rows)} ROIs, single HC). Expect MD vs stiffness NEG, uFA vs stiffness POS.")
    print(f"  {'pair':<26}{'GATED':>14}{'ungated':>14}")
    for a in ["MD", "uFA", "cond_aniso", "alpha", "dE_model"]:
        rg, ng = corr(a, "stiffness")          # GATED column (CSF-adjacent cortex excluded)
        ru, nu = corr(a, "stiffness_ung")      # ungated sensitivity
        print(f"  {a+' vs stiffness':<26}{f'rho={rg:+.2f}(n={ng})':>14}{f'rho={ru:+.2f}(n={nu})':>14}")
    print("\nGATED is primary (Olsson MRE handling); ungated is the sensitivity. n=1 -> across-region")
    print("trend, not a statistical result. The cohort runner aggregates this per subject.")


if __name__ == "__main__":
    main()
