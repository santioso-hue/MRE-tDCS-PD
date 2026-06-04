"""
04_extract_roi_efield.py — Extract E-field statistics from ROIs for all three models

Montage: C3 (anode, left M1) → Fp2 (cathode, right supraorbital), 2 mA
ROIs: subcortical PD-relevant structures from Olsson 2025 + DISTAL atlas (Ewert 2018)
Method: radius spheres at MNI coordinates → transform to subject space → sample WM+GM elements

Tissue sampling:
  CHARM assigns brainstem/basal-ganglia structures predominantly to WM (tag=1), not
  cortical GM (tag=2). Using only tag=2 produces zero-element ROIs for SNc/VTA.
  We sample both WM (tag=1) and GM (tag=2) — these are the only volumetric tissue
  types in the FEM where anisotropic conductivity matters.

Radii:
  - Cortical (L_M1): 3mm (large, well-resolved)
  - Basal ganglia (PUT, NAc, GPe): 5mm (moderate-sized nuclei)
  - Mesencephalon (SNc, VTA): 7mm (small nuclei, partial-volume dominated, brainstem)

Outputs:
  - Table of mean / median / p95 E-field per ROI per model
  - Pairwise delta table: (MD-dMRI − ISO) / ISO and (MD-dMRI − DTI) / DTI
  - Tissue composition of each ROI (fraction WM vs GM elements)

Usage:
  cd /Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation
  ~/Applications/SimNIBS-4.6/bin/simnibs_python scripts/04_extract_roi_efield.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg  # noqa: E402  (paths/subject from config/config.sh)

import numpy as np
import simnibs
from simnibs import mesh_io
import os

WDIR    = cfg["WORK_DIR"]
SUBPATH = cfg["M2M_DIR"]

# ── ROI definitions (MNI152 space, mm) ───────────────────────────────────────
# References: Ewert et al. 2018 (DISTAL), Pauli et al. 2018 (CIT168 7T atlas)
# Side notation: L = anode hemisphere (C3, left), R = cathode hemisphere (Fp2, right)
ROIS = {
    # Subcortical — primary PD targets / Olsson 2025 key regions
    "L_SNc":  [-8,  -16, -12],   # left  substantia nigra pars compacta
    "R_SNc":  [ 8,  -16, -12],   # right SNc
    "L_VTA":  [-4,  -16,  -9],   # left  ventral tegmental area
    "R_VTA":  [ 4,  -16,  -9],   # right VTA
    "L_PUT":  [-28,  -2,   4],   # left  putamen
    "R_PUT":  [ 28,  -2,   4],   # right putamen
    "L_NAc":  [-12,  10,  -8],   # left  nucleus accumbens
    "R_NAc":  [ 12,  10,  -8],   # right nucleus accumbens
    "L_GPe":  [-24,  -8,   2],   # left  globus pallidus externus
    "R_GPe":  [ 24,  -8,   2],   # right GPe
    # Cortical ROI — directly under C3 electrode (anode, left M1)
    "L_M1":   [-40, -14,  60],   # left  primary motor cortex hand area
}

# Per-ROI radii (mm) — scaled to nucleus size and CHARM segmentation resolution
# Mesencephalon structures (SNc, VTA): 7mm — CHARM labels most of this region as WM;
#   the larger sphere is needed to capture any nearby anisotropic tissue effect.
#   Results are dominated by the surrounding WM field, not the nucleus itself.
# Basal ganglia (PUT, NAc, GPe): 5mm — moderate nuclei, partially GM-labelled.
# Cortical (L_M1): 3mm — well-resolved in CHARM, directly under C3 electrode.
ROI_RADII = {
    "L_SNc": 7.0,
    "R_SNc": 7.0,
    "L_VTA": 7.0,
    "R_VTA": 7.0,
    "L_PUT": 5.0,
    "R_PUT": 5.0,
    "L_NAc": 5.0,
    "R_NAc": 5.0,
    "L_GPe": 5.0,
    "R_GPe": 5.0,
    "L_M1":  3.0,
}

# Tissue tags to sample — both WM (1) and GM (2)
# CHARM labels subcortical structures (SNc, VTA, GPe, PUT) predominantly as WM.
# The primary effect of anisotropic conductivity is in WM; excluding it hides the
# main model comparison signal. Both tags are included throughout for consistency.
TISSUE_TAGS = (1, 2)

# ── Simulation mesh files ─────────────────────────────────────────────────────
MODELS = {
    "ISO":         os.path.join(WDIR, "sim_ISO",         f"{cfg['SUBJECT']}_TDCS_1_scalar.msh"),
    "DTI":         os.path.join(WDIR, "sim_DTI",         f"{cfg['SUBJECT']}_TDCS_1_vn.msh"),  # dwi2cond
    "MD-dMRI":     os.path.join(WDIR, "sim_MD_dMRI",     f"{cfg['SUBJECT']}_TDCS_1_vn.msh"),  # σ∝⟨D⟩ (Model 3)
    "MD-dMRI-FWE": os.path.join(WDIR, "sim_MD_dMRI_mc",  f"{cfg['SUBJECT']}_TDCS_1_vn.msh"),  # free-water-eliminated (Model 4)
}


def get_roi_stats(msh, center_subj, radius, tissue_tags=(1, 2)):
    """Extract E-field magnitude statistics within a sphere over WM+GM elements.

    tissue_tags: tuple of SimNIBS tags to include (1=WM, 2=GM).
    Returns n, mean, median, p95 E-field and tissue composition (n_wm, n_gm).
    """
    tag = msh.elm.tag1
    in_tissue = np.zeros(len(tag), dtype=bool)
    for t in tissue_tags:
        in_tissue |= (tag == t)
    ctr  = msh.elements_baricenters().value     # (N_elems, 3)
    dist = np.linalg.norm(ctr - center_subj[np.newaxis, :], axis=1)
    roi  = in_tissue & (dist < radius)
    if not roi.any():
        return dict(n=0, n_wm=0, n_gm=0, mean=np.nan, median=np.nan, p95=np.nan)
    E = msh.field['magnE'].value[roi]
    return dict(
        n      = int(roi.sum()),
        n_wm   = int(np.sum((tag == 1) & roi)),
        n_gm   = int(np.sum((tag == 2) & roi)),
        mean   = float(np.mean(E)),
        median = float(np.median(E)),
        p95    = float(np.percentile(E, 95)),
    )


# ── Transform MNI → subject coordinates ─────────────────────────────────────
print("Transforming MNI coordinates to subject space...")
mni_coords  = np.array(list(ROIS.values()), dtype=float)
try:
    subj_coords = simnibs.mni2subject_coords(mni_coords, SUBPATH)
    roi_centers = {name: subj_coords[i] for i, name in enumerate(ROIS)}
    print(f"  MNI → subject transform applied via m2m_<subject>/")
except Exception as e:
    print(f"  WARNING: mni2subject_coords failed ({e})")
    print(f"  Falling back to identity (MNI = subject space — approximate only)")
    roi_centers = {name: np.array(coord) for name, coord in ROIS.items()}

# ── Extract E-field per model ─────────────────────────────────────────────────
results = {}   # results[model_name][roi_name] = stats dict

for model, msh_path in MODELS.items():
    if not os.path.exists(msh_path):
        print(f"\n  [{model}] mesh not found: {msh_path}  — SKIPPING")
        results[model] = None
        continue

    print(f"\nLoading {model} mesh: {msh_path}")
    msh = mesh_io.read_msh(msh_path)
    print(f"  Elements: {msh.elm.nr:,}   Nodes: {msh.nodes.nr:,}")
    if 'magnE' not in msh.field:
        print(f"  ERROR: 'magnE' field not found. Fields: {list(msh.field.keys())}")
        results[model] = None
        continue

    results[model] = {}
    for roi_name, center in roi_centers.items():
        radius = ROI_RADII[roi_name]
        stats  = get_roi_stats(msh, center, radius, tissue_tags=TISSUE_TAGS)
        results[model][roi_name] = stats


# ── Print table ───────────────────────────────────────────────────────────────
def pct_delta(new, ref):
    if ref == 0 or np.isnan(ref) or np.isnan(new):
        return float('nan')
    return 100 * (new - ref) / ref


print("\n" + "=" * 100)
print(f"E-field magnitude — C3(+2mA anode) → Fp2(-2mA cathode)")
print(f"Statistic: p95 (V/m).  Tissue: WM (tag=1) + GM (tag=2).  Radii: 7mm SNc/VTA, 5mm BG, 3mm M1")
print("=" * 100)

available = [m for m in MODELS if results[m] is not None]

hdr = f"{'ROI':<10}"
for m in available:
    hdr += f"  {m:>10}(p95)"
if "ISO" in available and "MD-dMRI" in available:
    hdr += f"  {'MD-dMRI vs ISO':>16}"
if "DTI" in available and "MD-dMRI" in available:
    hdr += f"  {'MD-dMRI vs DTI':>16}"
hdr += f"   {'WM/GM(n)':>14}"
print(hdr)
print("-" * len(hdr))

for roi in ROIS:
    row = f"{roi:<10}"
    vals = {}
    for m in available:
        st = results[m].get(roi, {})
        v  = st.get('p95', np.nan)
        vals[m] = v
        row += f"  {v:>15.4f}" if not np.isnan(v) else f"  {'N/A':>15}"
    if "ISO" in vals and "MD-dMRI" in vals:
        d = pct_delta(vals.get("MD-dMRI", np.nan), vals.get("ISO", np.nan))
        row += f"  {d:>+14.1f}%" if not np.isnan(d) else f"  {'N/A':>15}"
    if "DTI" in vals and "MD-dMRI" in vals:
        d = pct_delta(vals.get("MD-dMRI", np.nan), vals.get("DTI", np.nan))
        row += f"  {d:>+14.1f}%" if not np.isnan(d) else f"  {'N/A':>15}"
    if available:
        st0   = results[available[0]][roi]
        n_wm  = st0.get('n_wm', 0)
        n_gm  = st0.get('n_gm', 0)
        r     = ROI_RADII[roi]
        row  += f"   WM={n_wm},GM={n_gm}(r={r:.0f}mm)"
    print(row)

print("\n" + "-" * 100)
print("Mean E-field version (complementary):")
print("-" * 100)

hdr2 = f"{'ROI':<10}"
for m in available:
    hdr2 += f"  {m:>10}(mean)"
if "ISO" in available and "MD-dMRI" in available:
    hdr2 += f"  {'MD-dMRI vs ISO':>16}"
if "DTI" in available and "MD-dMRI" in available:
    hdr2 += f"  {'MD-dMRI vs DTI':>16}"
print(hdr2)

for roi in ROIS:
    row = f"{roi:<10}"
    vals = {}
    for m in available:
        st = results[m].get(roi, {})
        v  = st.get('mean', np.nan)
        vals[m] = v
        row += f"  {v:>15.4f}" if not np.isnan(v) else f"  {'N/A':>15}"
    if "ISO" in vals and "MD-dMRI" in vals:
        d = pct_delta(vals.get("MD-dMRI", np.nan), vals.get("ISO", np.nan))
        row += f"  {d:>+14.1f}%" if not np.isnan(d) else f"  {'N/A':>15}"
    if "DTI" in vals and "MD-dMRI" in vals:
        d = pct_delta(vals.get("MD-dMRI", np.nan), vals.get("DTI", np.nan))
        row += f"  {d:>+14.1f}%" if not np.isnan(d) else f"  {'N/A':>15}"
    print(row)

# ── CSV output: all models + triaxial deltas ─────────────────────────────────
import csv
csv_path = os.path.join(WDIR, "..", "analysis", "results", "roi_efield_4models.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ROI"] + [f"{m}_p95_Vm" for m in available] +
               ["MD-dMRI_vs_ISO_%", "MD-dMRI_vs_DTI_%"])
    for roi in ROIS:
        vals = {m: results[m][roi].get('p95', np.nan) for m in available}
        row = [roi] + [f"{vals[m]:.4f}" for m in available]
        md = vals.get("MD-dMRI", np.nan)
        row += [f"{pct_delta(md, vals.get('ISO',np.nan)):+.1f}",
                f"{pct_delta(md, vals.get('DTI',np.nan)):+.1f}"]
        w.writerow(row)
print(f"\nCSV written: {csv_path}")

print("\n" + "=" * 100)
print("Physical consistency checks:")
print("  - L-side ROIs (anode hemisphere) should generally show HIGHER E-field than R-side")
print("  - MD-dMRI > ISO expected in anisotropic WM-adjacent regions (higher μFA → directed current)")
print("  - DTI↔MD-dMRI delta largest in mesencephalon/temporal WM where μFA/FA diverge most")
print("  - L_M1 (directly under C3 anode) should have highest E-field of all cortical ROIs")
print("  - CAVEAT (pilot): SNc/VTA sampled from WM surrounding the nucleus, not the nucleus itself.")
print("    CHARM at 1mm does not resolve these small nuclei. Results reflect peri-nigral WM field.")
print("    Full-cohort analysis should use atlas-based ROI labels (e.g., DISTAL applied to m2m/).")
