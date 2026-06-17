"""
compare_sims.py — Global (tissue-level) comparison, validation, and visualization of the tDCS sims.

This is NOT an ROI analysis (no anatomical masks). It compares the three conductivity models at the
tissue level (GM / WM / whole-brain |E|) and produces viewable outputs:
  * per-model |E| statistics + a PASS/FLAG validation (physiological GM p95, no spikes)
  * DTI↔MD-dMRI and ISO↔MD-dMRI global deltas (the conductivity-model effect)
  * magnE interpolated to T1-space NIfTIs  -> open in FSLeyes over m2m_<subj>/T1.nii.gz
  * overlay PNGs (|E| per model in 3 planes + the MD-dMRI − DTI difference)

Usage:   simnibs_python analysis/compare_sims.py
Outputs: analysis/sim_compare/  (NIfTIs + PNGs + the console table below)
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sims import MODELS, sim_mesh  # noqa: E402  (shared montage-aware mesh lookup)
from simnibs import mesh_io  # noqa: E402

_ap = argparse.ArgumentParser(); _ap.add_argument("--montage", default="M1")
MONTAGE = _ap.parse_args().montage
WDIR = cfg["WORK_DIR"]; M2M = cfg["M2M_DIR"]; SUBJ = cfg["SUBJECT"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_compare", SUBJ)   # per-subject
os.makedirs(OUT, exist_ok=True)
EFIELD_P95 = (0.10, 0.60)   # physiological GM |E| band at 2 mA (V/m)
SPIKE_MAX = 5.0             # max/p95 — guards against a single hot mesh element

t1img = nib.load(os.path.join(M2M, "T1.nii.gz"))
T1 = np.asarray(t1img.dataobj, float); aff = t1img.affine; shape = t1img.shape
seg = np.asarray(nib.load(os.path.join(M2M, "segmentation", "labeling.nii.gz")).dataobj).astype(int)
brain = np.isin(seg, [2, 41, 3, 42])    # cerebral GM+WM — the region where |E| is meaningful


def load_mesh(name):
    p = sim_mesh(WDIR, MONTAGE, name, SUBJ)
    return mesh_io.read_msh(p) if p else None


def stats(v):
    return dict(mean=float(v.mean()), median=float(np.median(v)), p95=float(np.percentile(v, 95)),
                p99=float(np.percentile(v, 99)), max=float(v.max()))


# per-model: stats, validation, and magnE interpolated to the T1 grid
rows, vols = {}, {}
for name in MODELS:
    msh = load_mesh(name)
    if msh is None:
        print(f"  {name}: mesh missing -> skipped"); continue
    E = msh.field["magnE"].value
    tag = msh.elm.tag1
    gm, wm, br = stats(E[tag == 2]), stats(E[tag == 1]), stats(E[(tag == 1) | (tag == 2)])
    # spike = GM max/p95 (the target tissue; matches qc_harness). A whole-brain max/p95 is inflated
    # by single hot WM/electrode-boundary FEM elements and is not a meaningful field check.
    spike = gm["max"] / max(gm["p95"], 1e-9)
    ok = (EFIELD_P95[0] <= gm["p95"] <= EFIELD_P95[1]) and (spike <= SPIKE_MAX)
    rows[name] = dict(gm=gm, wm=wm, brain=br, spike=spike, ok=ok)
    vol = np.asarray(msh.field["magnE"].interpolate_to_grid(shape, aff, method="linear", continuous=False))
    vols[name] = vol
    nib.save(nib.Nifti1Image(vol.astype(np.float32), aff, t1img.header),
             os.path.join(OUT, f"magnE_{name.replace('-', '_')}_T1.nii.gz"))
    print(f"  {name}: interpolated -> magnE_{name.replace('-', '_')}_T1.nii.gz")

# comparison + validation table
print("\nPer-model |E| (V/m), tissue-level (no ROIs)")
print(f"{'model':9s}{'GM mean':>9s}{'GM med':>8s}{'GM p95':>8s}{'GM max':>8s}{'WM p95':>8s}{'spike':>7s}   valid")
for name in MODELS:
    if name not in rows:
        continue
    r = rows[name]
    print(f"{name:9s}{r['gm']['mean']:9.3f}{r['gm']['median']:8.3f}{r['gm']['p95']:8.3f}"
          f"{r['gm']['max']:8.3f}{r['wm']['p95']:8.3f}{r['spike']:7.1f}   {'PASS' if r['ok'] else 'FLAG'}")
print(f"  (validation: GM p95 in {EFIELD_P95} V/m, spike max/p95 <= {SPIKE_MAX})")

# model-vs-model deltas (the conductivity-model effect, brain GM+WM)
print("\nGlobal E-field deltas within cerebral GM+WM")
for a, b in [("MD-dMRI", "DTI"), ("MD-dMRI", "ISO"), ("DTI", "ISO")]:
    if a not in vols or b not in vols:
        continue
    diff = vols[a] - vols[b]
    m = brain & (vols[b] > 0.01) & np.isfinite(diff)
    rel = 100.0 * diff[m] / vols[b][m]
    print(f"  {a:8s} − {b:8s}: median|ΔE|={np.median(np.abs(diff[m])):.4f} V/m   "
          f"median Δ={np.median(rel):+5.1f}%   p95|Δ|={np.percentile(np.abs(rel), 95):4.1f}%   "
          f"max|ΔE|={np.max(np.abs(diff[m])):.3f}")
    if (a, b) == ("MD-dMRI", "DTI"):
        nib.save(nib.Nifti1Image(diff.astype(np.float32), aff, t1img.header),
                 os.path.join(OUT, "diff_MDdMRI_minus_DTI_T1.nii.gz"))

# visualization PNGs
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ci, cj, ck = (int(round(c)) for c in np.array(np.where(brain)).mean(axis=1))   # brain-centroid slices
    present = [n for n in MODELS if n in vols]
    vmax = np.percentile(np.concatenate([vols[n][brain] for n in present]), 99)    # shared scale

    # Figure 1: |E| per model, 3 planes, on T1
    fig, ax = plt.subplots(len(present), 3, figsize=(11, 3.2 * len(present)))
    ax = np.atleast_2d(ax)
    for r, n in enumerate(present):
        planes = [(T1[ci], vols[n][ci]), (T1[:, cj], vols[n][:, cj]), (T1[:, :, ck], vols[n][:, :, ck])]
        for c, (bg, ov) in enumerate(planes):
            ax[r, c].imshow(np.rot90(bg), cmap="gray")
            ax[r, c].imshow(np.rot90(np.ma.masked_less(ov, 0.01)), cmap="hot", vmin=0, vmax=vmax, alpha=0.75)
            ax[r, c].axis("off")
        ax[r, 0].set_ylabel(n, rotation=90, fontsize=12)
        ax[r, 0].axis("on"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
    fig.suptitle(f"{SUBJ}  |E| (V/m), shared scale 0–{vmax:.2f}  (C3→Fp2, 2 mA)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare_magnE.png"), dpi=90); plt.close(fig)
    print(f"\nPNG -> {OUT}/compare_magnE.png")

    # Figure 2: MD-dMRI − DTI difference (the conductivity-model effect)
    if "MD-dMRI" in vols and "DTI" in vols:
        diff = vols["MD-dMRI"] - vols["DTI"]
        dlim = np.percentile(np.abs(diff[brain]), 99)
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
        for c, (bg, ov) in enumerate([(T1[ci], diff[ci]), (T1[:, cj], diff[:, cj]), (T1[:, :, ck], diff[:, :, ck])]):
            ax[c].imshow(np.rot90(bg), cmap="gray")
            im = ax[c].imshow(np.rot90(np.ma.masked_where(np.abs(ov) < 1e-3, ov)),
                              cmap="seismic", vmin=-dlim, vmax=dlim, alpha=0.8)
            ax[c].axis("off")
        fig.colorbar(im, ax=ax, fraction=0.025, label="ΔE V/m")
        fig.suptitle(f"{SUBJ}  MD-dMRI − DTI  (red = MD-dMRI higher)")
        fig.savefig(os.path.join(OUT, "compare_diff_MDdMRI_DTI.png"), dpi=90); plt.close(fig)
        print(f"PNG -> {OUT}/compare_diff_MDdMRI_DTI.png")
except Exception as e:
    print(f"  PNG generation skipped: {type(e).__name__}: {e}")

print(f"\nVisualize interactively:  fsleyes {os.path.join(M2M,'T1.nii.gz')} "
      f"{os.path.join(OUT,'magnE_MD_dMRI_T1.nii.gz')} -cm hot")
