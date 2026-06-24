"""_repsubj.py - deterministic representative-subject pick, shared by Figures 2 and 3 so both caption the
same subject. The representative is the subject whose mean white-matter-lobe conductivity-model effect
(dE_model) is nearest the cohort median; ties break by sorted subject id."""
import os, glob, csv
import numpy as np

WM = ["WM_Frontal", "WM_Parietal", "WM_Temporal", "WM_Occipital"]


def _wm_de(csv_path):
    vals = []
    for r in csv.DictReader(open(csv_path)):
        if r["ROI"] in WM:
            try:
                dti, md = float(r["DTI_p95"]), float(r["MD-dMRI_p95"])
                if dti > 0:
                    vals.append(100.0 * (md - dti) / dti)
            except (KeyError, ValueError):
                pass
    return float(np.mean(vals)) if vals else np.nan


def representative_subject(results_dir, montage="M1"):
    """Return (subject_id, wm_de, cohort_median) for the subject nearest the cohort-median WM dE_model.
    Candidate CSVs are globbed in sorted order so the pick is deterministic (ties -> first sorted id)."""
    cands = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*", f"roi_efield_{montage}.csv"))):
        sid = os.path.basename(os.path.dirname(p))
        d = _wm_de(p)
        if np.isfinite(d):
            cands.append((sid, d))
    if not cands:
        raise FileNotFoundError(f"no roi_efield_{montage}.csv under {results_dir}")
    med = float(np.median([d for _, d in cands]))
    sid, d = min(cands, key=lambda c: (abs(c[1] - med), c[0]))
    return sid, d, med
