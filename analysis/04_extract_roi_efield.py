"""
04_extract_roi_efield.py — E-field per ROI for all conductivity models.

Montage: C3 (anode, left M1) -> Fp2 (cathode, right supraorbital), 2 mA.

ROIs: the FastSurfer-derived masks in mesh space (built by analysis/build_rois.py ->
registration/fastsurfer_rois/), loaded through _rois.py: cortical and white-matter lobes, corpus
callosum, and the aseg subcortical structures. Each ROI is sampled over every GM/WM element whose
barycentre falls inside it; "WholeBrain" is all GM+WM elements.

Tissue sampling: CHARM assigns the deep nuclei predominantly to WM (tag=1), not cortical GM (tag=2),
so we sample both WM and GM — the volumetric tissues where anisotropic conductivity matters.

Output: p95 and mean E-field per ROI per model, with (MD-dMRI vs ISO) and (vs DTI) deltas;
CSV analysis/results/roi_efield_4models.csv.

Usage:  cd <repo>;  ~/Applications/SimNIBS-4.6/bin/simnibs_python analysis/04_extract_roi_efield.py
"""
import os
import sys
import csv
import numpy as np
from simnibs import mesh_io

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rois import load_labeled, assign_mesh_labels  # noqa: E402

WDIR = cfg["WORK_DIR"]; M2M = cfg["M2M_DIR"]; REG = cfg["REG_DIR"]
TISSUE_TAGS = (1, 2)              # WM + GM (the volumetric tissues with anisotropic conductivity)
WHOLE_BRAIN = "WholeBrain"

MODELS = {
    "ISO":     os.path.join(WDIR, "sim_ISO",     f"{cfg['SUBJECT']}_TDCS_1_scalar.msh"),
    "DTI":     os.path.join(WDIR, "sim_DTI",     f"{cfg['SUBJECT']}_TDCS_1_vn.msh"),   # dwi2cond
    "MD-dMRI": os.path.join(WDIR, "sim_MD_dMRI", f"{cfg['SUBJECT']}_TDCS_1_vn.msh"),   # sigma ~ <D>
}


def roi_stats(roi, tag, e):
    """E-field magnitude stats over a boolean element-mask `roi`."""
    if not roi.any():
        return dict(n=0, n_wm=0, n_gm=0, mean=np.nan, median=np.nan, p95=np.nan)
    er = e[roi]
    return dict(
        n=int(roi.sum()),
        n_wm=int(np.sum((tag == 1) & roi)),
        n_gm=int(np.sum((tag == 2) & roi)),
        mean=float(np.mean(er)),
        median=float(np.median(er)),
        p95=float(np.percentile(er, 95)),
    )


labeled, lab_aff, names = load_labeled(REG)
ROI_ORDER = list(names.values()) + [WHOLE_BRAIN]

# Extract E-field per model
results = {}
for model, msh_path in MODELS.items():
    if not os.path.exists(msh_path):
        print(f"  [{model}] mesh not found: {msh_path}  — SKIPPING")
        results[model] = None
        continue
    print(f"Loading {model} mesh: {os.path.basename(msh_path)}")
    msh = mesh_io.read_msh(msh_path)
    if 'magnE' not in msh.field:
        print(f"  ERROR: 'magnE' not found. Fields: {list(msh.field.keys())}")
        results[model] = None
        continue
    tag = msh.elm.tag1
    e = msh.field['magnE'].value
    elab = assign_mesh_labels(msh.elements_baricenters().value, labeled, lab_aff)
    tissue = np.isin(tag, TISSUE_TAGS)
    results[model] = {n: roi_stats((elab == k) & tissue, tag, e) for k, n in names.items()}
    results[model][WHOLE_BRAIN] = roi_stats(tissue, tag, e)


def pct_delta(new, ref):
    if ref == 0 or np.isnan(ref) or np.isnan(new):
        return float('nan')
    return 100 * (new - ref) / ref


available = [m for m in MODELS if results[m] is not None]

print("\n" + "=" * 100)
print("E-field magnitude — C3(+2mA anode) -> Fp2(-2mA cathode)")
print("Statistic: p95 (V/m).  Tissue: WM(1)+GM(2).  ROIs: FastSurfer masks (mesh space)")
print("=" * 100)

hdr = f"{'ROI':<16}" + "".join(f"  {m:>10}(p95)" for m in available)
if "ISO" in available and "MD-dMRI" in available:
    hdr += f"  {'MD-dMRI vs ISO':>16}"
if "DTI" in available and "MD-dMRI" in available:
    hdr += f"  {'MD-dMRI vs DTI':>16}"
hdr += f"   {'WM/GM(n)':>14}"
print(hdr); print("-" * len(hdr))

for roi in ROI_ORDER:
    row = f"{roi:<16}"
    vals = {}
    for m in available:
        v = results[m].get(roi, {}).get('p95', np.nan)
        vals[m] = v
        row += f"  {v:>15.4f}" if not np.isnan(v) else f"  {'N/A':>15}"
    if "ISO" in vals and "MD-dMRI" in vals:
        d = pct_delta(vals.get("MD-dMRI", np.nan), vals.get("ISO", np.nan))
        row += f"  {d:>+14.1f}%" if not np.isnan(d) else f"  {'N/A':>15}"
    if "DTI" in vals and "MD-dMRI" in vals:
        d = pct_delta(vals.get("MD-dMRI", np.nan), vals.get("DTI", np.nan))
        row += f"  {d:>+14.1f}%" if not np.isnan(d) else f"  {'N/A':>15}"
    if available:
        st0 = results[available[0]][roi]
        row += f"   WM={st0.get('n_wm', 0)},GM={st0.get('n_gm', 0)}"
    print(row)

# CSV
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "roi_efield_4models.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ROI"] + [f"{m}_p95_Vm" for m in available] +
               [f"{m}_mean_Vm" for m in available] + ["MD-dMRI_vs_ISO_%", "MD-dMRI_vs_DTI_%"])
    for roi in ROI_ORDER:
        p95 = {m: results[m][roi].get('p95', np.nan) for m in available}
        mean = {m: results[m][roi].get('mean', np.nan) for m in available}
        md = p95.get("MD-dMRI", np.nan)
        w.writerow([roi] + [f"{p95[m]:.4f}" for m in available] + [f"{mean[m]:.4f}" for m in available] +
                   [f"{pct_delta(md, p95.get('ISO', np.nan)):+.1f}",
                    f"{pct_delta(md, p95.get('DTI', np.nan)):+.1f}"])
print(f"\nCSV written: {csv_path}")
print("MD-dMRI vs DTI isolates the conductivity-model effect (same montage/anatomy).")
