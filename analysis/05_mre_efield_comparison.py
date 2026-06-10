"""
05_mre_efield_comparison.py — Post-hoc cross-modal comparison: MRE mechanics vs MD-dMRI microstructure
and the tDCS E-field, per ROI. Confidence-gated to match Olsson's MRE handling.

Purpose (see pipeline/conductivity_models_derivation.md):
  (1) Consistency/QC — does the subject reproduce the known microstructure<->mechanics relationship
      (MD vs stiffness negative, uFA vs stiffness positive)?  A data-quality check, not a new claim.
  (2) Relevance map — does the conductivity model's IMPACT on the E-field (dE = E_MD-dMRI - E_DTI,
      a local quantity where the montage largely cancels) land where Olsson flags tissue alteration?

MRE confidence gating (item H): MRE maps (stiffness, storage, loss) are sampled only over voxels with
mre_confidence >= threshold AND not on the CSF-adjacent cortical surface (Helmholtz inversion is
unreliable at the CSF interface, per Olsson). The script reports the gated result as primary and the
UNGATED result as a sensitivity row. Confidence does NOT gate MD/uFA (QTI) or the E-field (FEM).

E-field statistic: MEDIAN per ROI (Olsson's ROI convention; item F); p95 reported as a sensitivity column.

Run:  cd <repo>;  simnibs_python analysis/05_mre_efield_comparison.py [--conf-thresh V | --conf-pct P]
"""
import os, sys, csv, argparse
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

WORK, M2M, REG = cfg["WORK_DIR"], cfg["M2M_DIR"], cfg["REG_DIR"]
TENSOR = os.path.join(WORK, "tensor_MD_dMRI.nii.gz")
MRE_MAPS = {"stiffness": f"{REG}/mre_stiffness_T1.nii.gz",     # confidence-gated
            "storage":   f"{REG}/mre_storage_T1.nii.gz",
            "loss":      f"{REG}/mre_loss_T1.nii.gz"}
MICRO_MAPS = {"MD": f"{REG}/MD_T1.nii.gz", "uFA": f"{REG}/C_mu_T1.nii.gz"}   # NOT gated (QTI, not MRE)
MESHES = {"E_ISO": f"{WORK}/sim_ISO/{cfg['SUBJECT']}_TDCS_1_scalar.msh",
          "E_DTI": f"{WORK}/sim_DTI/{cfg['SUBJECT']}_TDCS_1_vn.msh",
          "E_MDdMRI": f"{WORK}/sim_MD_dMRI/{cfg['SUBJECT']}_TDCS_1_vn.msh"}


def build_gate(conf_thresh=None, conf_pct=10.0):
    """Gate MRE voxels: confidence >= threshold AND not CSF-adjacent cortical GM."""
    conf = np.asarray(nib.load(f"{REG}/mre_confidence_T1.nii.gz").dataobj, float)
    seg = np.asarray(nib.load(os.path.join(M2M, "final_tissues.nii.gz")).dataobj)
    seg = seg[..., 0] if seg.ndim == 4 else seg
    brain = np.isin(seg, [1, 2, 3])
    inbrain = conf[brain & np.isfinite(conf) & (conf > 0)]
    thr = float(conf_thresh) if conf_thresh is not None else float(np.percentile(inbrain, conf_pct))
    gate = np.isfinite(conf) & (conf >= thr)
    if seg.shape == conf.shape:                                    # drop CSF-adjacent cortical surface
        csf_adj = ndi.binary_dilation(seg == 3, iterations=1) & (seg == 2)
        gate &= ~csf_adj
    return gate, thr, float(gate[brain].mean())


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
    for r in rows.values():
        for suf in ("", "_ung"):
            if np.isfinite(r.get("loss" + suf, np.nan)) and r.get("storage" + suf, 0):
                r["viscosity" + suf] = float(np.degrees(np.arctan2(r["loss" + suf], r["storage" + suf])))
    # E-field per model: median (primary) + p95 (sensitivity) over GM+WM elements in the ROI
    for m, path in MESHES.items():
        if not os.path.exists(path):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf-thresh", type=float, default=None, help="absolute MRE confidence cutoff")
    ap.add_argument("--conf-pct", type=float, default=10.0, help="percentile cutoff if no absolute thresh")
    args = ap.parse_args()

    gate, thr, frac = build_gate(args.conf_thresh, args.conf_pct)
    print(f"MRE gate: confidence >= {thr:.1f} (p{args.conf_pct:g}) + CSF-adjacent cortex excluded -> "
          f"{100*frac:.1f}% of brain voxels kept")
    rows = extract_subject(gate)

    out_csv = os.path.join(os.path.dirname(__file__), "results", cfg["SUBJECT"], "mre_efield_per_roi.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    keys = ["stiffness", "stiffness_ung", "viscosity", "MD", "uFA", "cond_aniso",
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
    for a, b in [("MD", "stiffness"), ("uFA", "stiffness"), ("cond_aniso", "stiffness"),
                 ("dE_model", "stiffness")]:
        rg, ng = corr(a, b + ("" if b != "stiffness" else ""))
        ru, nu = corr(a, b + "_ung") if b == "stiffness" else (rg, ng)
        print(f"  {a+' vs '+b:<26}{f'rho={rg:+.2f}(n={ng})':>14}{f'rho={ru:+.2f}(n={nu})':>14}")
    print("\nGATED is primary (Olsson MRE handling); ungated is the sensitivity. n=1 -> across-region")
    print("trend, not a statistical result. The cohort runner aggregates this per subject.")


if __name__ == "__main__":
    main()
