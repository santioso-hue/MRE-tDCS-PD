"""fa_sigma_diagnostic.py - interior vs boundary white-matter FA of the QTI <D> tensor, cohort-wide.

Supporting artifact for Figure 2. For each subject the Group-1 white-matter mask (recon-all WM lobes,
label ids 11-14) is eroded by one voxel: the eroded core is the interior, the removed shell is the
boundary. Median FA of <D> (from work/tensor_MD_dMRI.nii.gz, via eigh_6comp + fa_from_evals, over
positive-definite voxels) is reported in each region. A cohort-median summary row is appended.

Reads:  data/cohort_local/<id>/work/tensor_MD_dMRI.nii.gz
        data/cohort_local/<id>/registration/freesurfer_rois/  (via _rois.load_labeled)
Writes: analysis/results/fa_sigma_diagnostic.csv

Run:  $SIMNIBS_BIN/simnibs_python analysis/fa_sigma_diagnostic.py
"""
import os
import sys
import csv
import glob
import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rois import load_labeled, _labels_on_grid, eigh_6comp, fa_from_evals, PD_EPS, FILL_EPS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHORT = os.path.join(ROOT, "data", "cohort_local")
OUT = os.path.join(ROOT, "analysis", "results", "fa_sigma_diagnostic.csv")
WM_IDS = (11, 12, 13, 14)   # recon-all Group-1 WM lobes (frontal/parietal/temporal/occipital)


def subject_fa(tensor_path, reg_dir):
    """Median WM FA(<D>) for interior vs boundary voxels. Returns dict or None if no usable WM voxels."""
    labeled, lab_aff, _ = load_labeled(reg_dir)
    img = nib.load(tensor_path)
    t = np.asarray(img.dataobj, dtype=float)
    lab = _labels_on_grid(img, labeled, lab_aff)
    wm = np.isin(lab, WM_IDS)
    if not wm.any():
        return None
    interior = ndimage.binary_erosion(wm, iterations=1)
    boundary = wm & ~interior

    def med_fa(mask):
        sel = mask & np.isfinite(t[..., 0]) & (np.abs(t[..., 0]) > FILL_EPS)
        if not sel.any():
            return np.nan, 0
        ev, _ = eigh_6comp(t, sel)
        pd = ev[:, 0] > PD_EPS
        if not pd.any():
            return np.nan, 0
        return float(np.median(fa_from_evals(ev[pd]))), int(pd.sum())

    fa_i, n_i = med_fa(interior)
    fa_b, n_b = med_fa(boundary)
    return dict(fa_interior=fa_i, fa_boundary=fa_b, n_interior=n_i, n_boundary=n_b)


def main():
    subj_dirs = sorted(glob.glob(os.path.join(COHORT, "PD*")))
    rows = []
    for d in subj_dirs:
        sid = os.path.basename(d)
        tensor = os.path.join(d, "work", "tensor_MD_dMRI.nii.gz")
        reg = os.path.join(d, "registration")
        if not (os.path.exists(tensor) and os.path.isdir(os.path.join(reg, "freesurfer_rois"))):
            print(f"WARN: skipping {sid} (missing tensor or freesurfer_rois)", file=sys.stderr)
            continue
        res = subject_fa(tensor, reg)
        if res is None:
            print(f"WARN: skipping {sid} (no usable WM voxels)", file=sys.stderr)
            continue
        rows.append(dict(subject=sid, **res))
        print(f"{sid:24s} interior FA={res['fa_interior']:.3f}  boundary FA={res['fa_boundary']:.3f}"
              f"  (n {res['n_interior']}/{res['n_boundary']})")

    if not rows:
        sys.exit("fa_sigma_diagnostic: no subjects produced a row.")

    fi = np.array([r["fa_interior"] for r in rows], float)
    fb = np.array([r["fa_boundary"] for r in rows], float)
    coh = dict(subject="cohort_median",
               fa_interior=float(np.nanmedian(fi)), fa_boundary=float(np.nanmedian(fb)),
               n_interior=int(np.nansum([r["n_interior"] for r in rows])),
               n_boundary=int(np.nansum([r["n_boundary"] for r in rows])))
    rows.append(coh)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cols = ["subject", "fa_interior", "fa_boundary", "n_interior", "n_boundary"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r[c], float) else r[c]) for c in cols})

    print(f"\nwrote {OUT}")
    print(f"cohort-median: interior FA={coh['fa_interior']:.3f}  boundary FA={coh['fa_boundary']:.3f}")


if __name__ == "__main__":
    main()
