"""
04_extract_roi_efield.py - E-field per ROI per conductivity model, namespaced by subject and montage.

Usage: ~/Applications/SimNIBS-4.6/bin/simnibs_python analysis/04_extract_roi_efield.py [--montage M1|DLPFC|...]
Output (gitignored): analysis/results/<subject>/roi_efield_<montage>.csv

ROIs from _rois.py (build_rois.py -> registration/<roi_dir>/): cortical/WM lobes, corpus callosum,
mesencephalon/pons (Group 1); sampled over every element whose barycentre falls inside. "WholeBrain" is
all GM+WM. Group 2 deep nuclei (CIT168, built by 07) are appended as extra E-field-only rows.

CHARM assigns the deep nuclei predominantly to WM (tag=1), not cortical GM (tag=2), so we sample both
WM and GM - the volumetric tissues where anisotropic conductivity matters.

Writes BOTH p95 (dosimetry headline, Huang 2017) and median (Olsson ROI convention) per model;
06_cohort_stats.py picks the column it needs.
"""
import os, sys, csv, argparse, glob
import numpy as np
import nibabel as nib
from simnibs import mesh_io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from _config import cfg  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rois import load_labeled, assign_mesh_labels, mask_select  # noqa: E402
from _sims import MODELS, sim_mesh                   # noqa: E402
from _stats import pct_delta                         # noqa: E402

WDIR, REG = cfg["WORK_DIR"], cfg["REG_DIR"]
TISSUE_TAGS = (1, 2)
WHOLE_BRAIN = "WholeBrain"
# Group 2 deep nuclei: overlap-allowed, E-field-only, sampled as SEPARATE binary masks
# (never an int-label / winner-take-all volume). Built by 07; absent on dev machines.
NUCLEI_DIR = os.path.join(REG, "atlas_rois", "nuclei")
NUCLEI = ("Pu", "Ca", "NAC", "GPe", "GPi", "SNc", "SNr", "VTA", "RN", "STN")
NUCLEI_SIDES = ("L", "R")


def load_nuclei_masks(nuclei_dir):
    """Glob the Group 2 nucleus masks as INDEPENDENT binary volumes (overlap allowed).
    Each entry is (name, bool array, own affine). Returns [] if the dir is absent."""
    out = []
    for side in NUCLEI_SIDES:
        for nuc in NUCLEI:
            p = os.path.join(nuclei_dir, f"roi_{side}_{nuc}.nii.gz")
            for f in sorted(glob.glob(p)):
                img = nib.load(f)
                out.append((f"{nuc}_{side}", np.asarray(img.dataobj) > 0, img.affine))
    return out


def roi_stats(roi, tag, e):
    if not roi.any():
        return dict(n=0, n_wm=0, n_gm=0, median=np.nan, p95=np.nan)
    er = e[roi]
    return dict(n=int(roi.sum()), n_wm=int(np.sum((tag == 1) & roi)), n_gm=int(np.sum((tag == 2) & roi)),
                median=float(np.median(er)), p95=float(np.percentile(er, 95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", default="M1")
    ap.add_argument("--subject", default=cfg["SUBJECT"])
    ap.add_argument("--results-root", default=os.path.join(ROOT, "analysis", "results"))
    args = ap.parse_args()

    labeled, lab_aff, names = load_labeled(REG)
    # Group 2 nuclei are extra ROWS after WholeBrain (overlap-allowed, E-field-only);
    # absent on dev machines -> skip silently so 04 still runs the Group 1 path.
    nuclei = load_nuclei_masks(NUCLEI_DIR)
    if nuclei:
        print(f"Group 2 nuclei (E-field only): {', '.join(n for n, _, _ in nuclei)}")
    roi_order = list(names.values()) + [WHOLE_BRAIN] + [n for n, _, _ in nuclei]

    results = {}
    for model in MODELS:
        p = sim_mesh(WDIR, args.montage, model, args.subject)
        if p is None:
            print(f"  [{model}/{args.montage}] mesh not found - SKIPPING")
            results[model] = None
            continue
        print(f"Loading {model}/{args.montage}: {os.path.relpath(p, WDIR)}")
        msh = mesh_io.read_msh(p)
        if "magnE" not in msh.field:
            print(f"  ERROR: 'magnE' missing. Fields: {list(msh.field.keys())}")
            results[model] = None
            continue
        tag, e = msh.elm.tag1, msh.field["magnE"].value
        bary = msh.elements_baricenters().value
        elab = assign_mesh_labels(bary, labeled, lab_aff)
        tissue = np.isin(tag, TISSUE_TAGS)
        results[model] = {n: roi_stats((elab == k) & tissue, tag, e) for k, n in names.items()}
        results[model][WHOLE_BRAIN] = roi_stats(tissue, tag, e)
        # Group 2 nuclei: each its own binary mask + affine, overlap allowed (no winner-take-all).
        # May be legitimately empty in some subjects -> roi_stats returns NaN (06 is NaN-safe).
        for nname, nmask, naff in nuclei:
            results[model][nname] = roi_stats(mask_select(bary, nmask, naff) & tissue, tag, e)

    available = [m for m in MODELS if results[m] is not None]
    if not available:
        sys.exit(f"No sim meshes found for montage '{args.montage}' / subject '{args.subject}'.")

    # Console table (p95, with the MD-dMRI deltas for human QC)
    print(f"\nE-field p95 (V/m) - montage {args.montage}, subject {args.subject}  (tissue WM+GM)")
    print(f"{'ROI':<16}" + "".join(f"{m:>12}" for m in available) + f"{'vs ISO':>9}{'vs DTI':>9}")
    for roi in roi_order:
        v = {m: results[m][roi]["p95"] for m in available}
        line = f"{roi:<16}" + "".join((f"{v[m]:>12.4f}" if not np.isnan(v[m]) else f"{'N/A':>12}") for m in available)
        md = v.get("MD-dMRI", np.nan)
        line += f"{pct_delta(md, v.get('ISO', np.nan)):>+8.1f}%{pct_delta(md, v.get('DTI', np.nan)):>+8.1f}%"
        print(line)

    # Machine CSV (p95 + median per model) - the schema 06_cohort_stats consumes
    outdir = os.path.join(args.results_root, args.subject)
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"roi_efield_{args.montage}.csv")
    cols = ["ROI"] + [f"{m}_{s}" for m in MODELS for s in ("p95", "median")]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for roi in roi_order:
            cells = [roi]
            for m in MODELS:
                st = results[m][roi] if results[m] is not None else {}
                cells += [f"{st.get('p95', float('nan')):.5f}", f"{st.get('median', float('nan')):.5f}"]
            w.writerow(cells)
    print(f"\nCSV: {csv_path}")
    if len(available) < len(MODELS):
        print(f"WARNING: only {available} present - cohort stats needs all of {MODELS}.")


if __name__ == "__main__":
    main()
