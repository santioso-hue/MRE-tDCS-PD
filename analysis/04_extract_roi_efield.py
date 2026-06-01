"""
04_extract_roi_efield.py — Extract and compare E-field across PD-relevant ROIs

Compares |E| (V/m) across three conductivity models:
  ISO     — isotropic scalar (sim_ISO/FullPD5_TDCS_1_scalar.msh)
  DTI     — FA-based anisotropy (sim_DTI/FullPD5_TDCS_1_vn.msh)
  MD-dMRI — μFA-based anisotropy (sim_MD_dMRI/FullPD5_TDCS_1_vn.msh)

ROIs: all subcortical structures from Olsson 2025 (Pauli et al. 2018 atlas),
bilateral, plus GPe/GPi for completeness.

  SNc  — Substantia Nigra pars compacta  (Olsson 2025: MD↑ d=1.35**, μFA↓ d=−1.01*)
  SNr  — Substantia Nigra pars reticulata (Olsson 2025: MD↑ d=1.16**)
  VTA  — Ventral Tegmental Area           (Olsson 2025: μFA↓ d=−0.95*, NM-CNR↓)
  PUT  — Putamen                          (Olsson 2025: μFA effect, stiffness↓)
  Ca   — Caudate Nucleus                  (Olsson 2025: stiffness↓, MD age effects)
  NAC  — Nucleus Accumbens                (Olsson 2025: FA↓ d=−1.15***, largest effect)
  GPi  — Globus Pallidus internus         (Olsson 2025: no significant PD effect)
  GPe  — Globus Pallidus externus         (Olsson 2025: no significant PD effect)
  RN   — Red Nucleus                      (Olsson 2025: MD correlation r=0.53**)

MNI coordinates: DISTAL atlas (Ewert et al. 2018) for SNc/VTA; Pauli et al. 2018
approximations for others. Transformed to subject space via CHARM warp.

NOTE on STN: excluded — not in Olsson 2025's ROI set and too small (≈5mm diameter)
for reliable extraction with 3mm radius spheres at 2.5mm QTI resolution.

ROI sphere radius: 3mm (see ROI_RADIUS_MM).
Statistics reported: mean, median, p25, p75, p95 (|E| in V/m).
The 95th percentile is reported as a robust peak-field estimate less sensitive
to outlier mesh elements than the maximum (Huang et al. 2017, eLife 6:e18834).

Output:
  results/roi_efield_table.csv   — statistics per ROI × model
  results/roi_efield_boxplot.png — comparison figure
  results/roi_efield_ratios.csv  — DTI/ISO and MD-dMRI/ISO ratios

Usage:
  cd /Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation
  ~/Applications/SimNIBS-4.6/bin/simnibs_python ../analysis/04_extract_roi_efield.py
"""

import os
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

import simnibs

# ── Paths ─────────────────────────────────────────────────────────────────────
WDIR   = "/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation"
OUTDIR = "/Users/santi/Documents/MRE_tDCS_PD/analysis/results"
os.makedirs(OUTDIR, exist_ok=True)
os.chdir(WDIR)

MODELS = {
    "ISO":     "sim_ISO/FullPD5_TDCS_1_scalar.msh",
    "DTI":     "sim_DTI/FullPD5_TDCS_1_vn.msh",
    "MD-dMRI": "sim_MD_dMRI/FullPD5_TDCS_1_vn.msh",
}

WARP_PATH = "m2m_FullPD5/toMNI/MNI2Conform_nonl.nii.gz"

# ── ROI definitions — MNI RAS coordinates (mm) ───────────────────────────────
# Coordinate sources:
#   SNc/SNr/VTA: DISTAL atlas (Ewert et al. 2018 NeuroImage)
#   Caudate/NAC/GPe/GPi: Pauli et al. 2018 (high-res 7T subcortical atlas)
#   RN (Red Nucleus): standard brainstem atlas coordinates
#   PUT: MNI Structural Atlas (Collins et al. 1994)
#
# All structures also reported in Olsson 2025 (Pauli 2018 atlas segmentation).
# STN excluded: not in Olsson 2025 ROI set; at ≈5mm diameter it is too small
# for reliable sampling with a 5mm radius sphere at 2.5mm QTI resolution.
#
# Montage: C3 (anode, left M1) → Fp2 (cathode, right supraorbital)
# Left hemisphere structures are ipsilateral to anode.

ROI_MNI = {
    # Primary PD pathology (dopaminergic degeneration)
    "L_SNc": np.array([-5.0,  -15.0, -11.0]),
    "R_SNc": np.array([ 5.0,  -15.0, -11.0]),
    "L_SNr": np.array([-5.0,  -25.0, -14.0]),   # inferior to SNc
    "R_SNr": np.array([ 5.0,  -25.0, -14.0]),
    "L_VTA": np.array([-9.0,  -14.0,  -8.0]),
    "R_VTA": np.array([ 9.0,  -14.0,  -8.0]),
    # Basal ganglia (motor circuit, all in Olsson 2025)
    "L_PUT": np.array([-28.0,   1.0,   2.0]),
    "R_PUT": np.array([ 28.0,   1.0,   2.0]),
    "L_Ca":  np.array([-12.0,  13.0,   8.0]),   # Caudate head
    "R_Ca":  np.array([ 12.0,  13.0,   8.0]),
    "L_NAC": np.array([ -9.0,  10.0,  -7.0]),   # Nucleus Accumbens
    "R_NAC": np.array([  9.0,  10.0,  -7.0]),
    "L_GPi": np.array([-17.0,  -5.0,  -1.0]),
    "R_GPi": np.array([ 17.0,  -5.0,  -1.0]),
    "L_GPe": np.array([-22.0,  -4.0,   2.0]),
    "R_GPe": np.array([ 22.0,  -4.0,   2.0]),
    # Brainstem reference (significant MD correlation in Olsson 2025)
    "L_RN":  np.array([ -4.0,  -22.0,  -9.0]),  # Red Nucleus
    "R_RN":  np.array([  4.0,  -22.0,  -9.0]),
}

ROI_RADIUS_MM = 3.0    # spherical ROI radius
# 3mm chosen for subcortical PD structures (SNc ≈5–7mm diam, VTA ≈3–5mm):
# 5mm spheres extend outside small nuclei into neighbouring tissue.
# 3mm samples the structural core while staying within typical nucleus boundaries.
# Reference: Saturnino et al. 2019 NeuroImage used 5mm for CORTICAL structures;
# smaller subcortical structures require proportionally smaller radii.
# With 3000 Monte Carlo points, ~500–1000 valid interpolated values per ROI.
N_SAMPLE_PTS  = 3000   # Monte Carlo sphere samples per ROI
RANDOM_SEED   = 42


# ── Coordinate transform: MNI → subject (Conform) space ──────────────────────
def mni_to_subject(mni_pts: np.ndarray, warp: np.ndarray,
                   warp_affine_inv: np.ndarray) -> np.ndarray:
    """
    Apply CHARM's MNI2Conform nonlinear warp to a set of MNI RAS coordinates.

    The warp field is defined on the MNI grid; each voxel stores the
    displacement vector from MNI space to Conform (subject T1) space,
    stored as absolute target coordinates.

    Args:
        mni_pts:          (N, 3) array of MNI RAS coordinates (mm)
        warp:             (X, Y, Z, 3) warp field array
        warp_affine_inv:  inverse affine of the warp NIfTI (MNI RAS → voxel)

    Returns:
        (N, 3) array of subject-space RAS coordinates (mm)
    """
    hom  = np.column_stack([mni_pts, np.ones(len(mni_pts))])
    vox  = (warp_affine_inv @ hom.T).T[:, :3]  # MNI voxel indices
    subj = np.zeros_like(mni_pts)
    for dim in range(3):
        subj[:, dim] = map_coordinates(
            warp[..., dim], vox.T, order=1, mode='nearest'
        )
    return subj


def make_sphere_samples(center: np.ndarray, radius: float,
                        n: int, seed: int) -> np.ndarray:
    """Return n uniformly-distributed points inside a sphere of given radius."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    radii = radius * rng.random(n) ** (1.0 / 3.0)   # volume-uniform
    return center + (radii[:, None] * vecs)


# ── Load warp ─────────────────────────────────────────────────────────────────
print("Loading MNI→Conform warp field...")
if not os.path.exists(WARP_PATH):
    raise FileNotFoundError(f"Warp not found: {WARP_PATH} — run CHARM first.")
warp_img = nib.load(WARP_PATH)
warp_data = warp_img.get_fdata()
warp_aff_inv = np.linalg.inv(warp_img.affine)
print(f"  Warp shape: {warp_data.shape}")

# ── Transform ROI centres to subject space ────────────────────────────────────
print("\nTransforming ROI centres: MNI → subject space")
mni_arr = np.array(list(ROI_MNI.values()))
subj_arr = mni_to_subject(mni_arr, warp_data, warp_aff_inv)
ROI_SUBJ = {name: subj_arr[i] for i, name in enumerate(ROI_MNI)}

for name, sc in ROI_SUBJ.items():
    print(f"  {name:8s}: MNI {ROI_MNI[name]} → subj [{sc[0]:+.1f}, {sc[1]:+.1f}, {sc[2]:+.1f}] mm")

# ── Pre-compute sphere sample sets (same points for all models) ───────────────
print("\nGenerating spherical ROI sample points...")
sphere_pts = {
    name: make_sphere_samples(center, ROI_RADIUS_MM, N_SAMPLE_PTS, RANDOM_SEED + i)
    for i, (name, center) in enumerate(ROI_SUBJ.items())
}

# ── Extract |E| per model per ROI ────────────────────────────────────────────
print("\nExtracting E-field magnitudes...")

# results[model][roi] = 1-D array of valid |E| values (V/m)
results = {}

for model, mesh_path in MODELS.items():
    if not os.path.exists(mesh_path):
        print(f"  SKIP {model}: {mesh_path} not found")
        results[model] = None
        continue

    print(f"\n  Loading {model}: {mesh_path}")
    m = simnibs.read_msh(mesh_path)
    field = m.field['magnE']

    results[model] = {}
    for roi_name, pts in sphere_pts.items():
        vals = field.interpolate_scattered(pts, out_fill=np.nan)
        valid = np.isfinite(vals) & (vals > 0)
        results[model][roi_name] = vals[valid]
        n_v = valid.sum()
        if n_v > 0:
            print(f"    {roi_name:8s}: n={n_v:4d}  "
                  f"mean={np.mean(vals[valid]):.3f}  "
                  f"median={np.median(vals[valid]):.3f}  "
                  f"p95={np.percentile(vals[valid], 95):.3f} V/m")
        else:
            print(f"    {roi_name:8s}: NO VALID POINTS — ROI may be outside mesh")

# ── Build statistics table ────────────────────────────────────────────────────
print("\n\nSummary statistics table (|E| in V/m)")
print("=" * 100)

stats_rows = []   # list of dicts for CSV

for roi_name in ROI_MNI:
    for model in MODELS:
        if results.get(model) is None or roi_name not in results[model]:
            continue
        v = results[model][roi_name]
        if len(v) == 0:
            continue
        row = {
            "ROI":   roi_name,
            "Model": model,
            "N":     len(v),
            "mean":  float(np.mean(v)),
            "median":float(np.median(v)),
            "p25":   float(np.percentile(v, 25)),
            "p75":   float(np.percentile(v, 75)),
            "p95":   float(np.percentile(v, 95)),
        }
        stats_rows.append(row)

# Print compact table
hdr = f"{'ROI':10s}  {'Model':8s}  {'N':>5s}  {'mean':>7s}  {'median':>7s}  {'p25':>7s}  {'p75':>7s}  {'p95':>7s}"
print(hdr)
print("-" * len(hdr))
for r in stats_rows:
    print(f"{r['ROI']:10s}  {r['Model']:8s}  {r['N']:5d}  "
          f"{r['mean']:7.4f}  {r['median']:7.4f}  "
          f"{r['p25']:7.4f}  {r['p75']:7.4f}  {r['p95']:7.4f}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
import csv

csv_path = os.path.join(OUTDIR, "roi_efield_table.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["ROI", "Model", "N", "mean", "median", "p25", "p75", "p95"])
    writer.writeheader()
    writer.writerows(stats_rows)
print(f"\nSaved: {csv_path}")

# ── Compute ISO-normalised ratios ─────────────────────────────────────────────
ratio_rows = []
for roi_name in ROI_MNI:
    iso_vals = results.get("ISO", {}).get(roi_name, np.array([]))
    if len(iso_vals) == 0:
        continue
    iso_med = np.median(iso_vals)
    row = {"ROI": roi_name, "ISO_median_Vm": round(iso_med, 5)}
    for model in ["DTI", "MD-dMRI"]:
        v = results.get(model, {}).get(roi_name, np.array([]))
        if len(v) == 0:
            continue
        med = np.median(v)
        row[f"{model}_median_Vm"] = round(med, 5)
        row[f"{model}/ISO_ratio"]  = round(med / iso_med, 4) if iso_med > 0 else np.nan
    ratio_rows.append(row)

print("\n|E| ratios (median) relative to ISO:")
print(f"{'ROI':10s}  {'ISO (V/m)':>10s}  {'DTI/ISO':>8s}  {'MDDMRI/ISO':>10s}")
print("-" * 50)
for r in ratio_rows:
    print(f"{r['ROI']:10s}  {r.get('ISO_median_Vm', 'N/A'):>10.4f}  "
          f"{r.get('DTI/ISO_ratio', 'N/A'):>8.4f}  "
          f"{r.get('MD-dMRI/ISO_ratio', 'N/A'):>10.4f}")

ratio_csv = os.path.join(OUTDIR, "roi_efield_ratios.csv")
all_keys = ["ROI", "ISO_median_Vm", "DTI_median_Vm", "DTI/ISO_ratio",
            "MD-dMRI_median_Vm", "MD-dMRI/ISO_ratio"]
with open(ratio_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(ratio_rows)
print(f"Saved: {ratio_csv}")

# ── Plot ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")   # headless
    import matplotlib.pyplot as plt

    n_rois = len(ROI_MNI)
    n_cols = 6
    n_rows = (n_rois + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4 * n_rows), sharey=False)
    fig.suptitle(
        "E-field magnitude (|E|) in PD-relevant ROIs — FullPD5\n"
        "Montage: C3 (anode, left M1) → Fp2 (cathode), 2 mA\n"
        "ISO, DTI, MD-dMRI: aniso_maxcond=4 (effective ratio cap 8.0:1)",
        fontsize=12, fontweight="bold"
    )

    roi_names = list(ROI_MNI.keys())
    colors = {"ISO": "#4e9af1", "DTI": "#f4a460", "MD-dMRI": "#6dbf67"}
    model_list = ["ISO", "DTI", "MD-dMRI"]

    for idx, roi_name in enumerate(roi_names):
        ax = axes[idx // n_cols][idx % n_cols]
        data_to_plot = []
        labels = []
        for model in model_list:
            v = results.get(model, {}).get(roi_name, np.array([]))
            if len(v) > 0:
                data_to_plot.append(v)
                labels.append(model)

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=labels,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            showfliers=False,
            widths=0.5,
        )
        for patch, model in zip(bp["boxes"], labels):
            patch.set_facecolor(colors.get(model, "gray"))
            patch.set_alpha(0.8)

        ax.set_title(roi_name.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_ylabel("|E| (V/m)" if idx % 6 == 0 else "", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.4, linestyle="--")

    # Hide any unused subplot panels
    for idx in range(n_rois, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(OUTDIR, "roi_efield_boxplot.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved: {fig_path}")

except ImportError:
    print("\nmatplotlib not available — skipping figure (install with pip install matplotlib)")

print("\n=== 04_extract_roi_efield.py complete ===")
print(f"Results in: {OUTDIR}/")
