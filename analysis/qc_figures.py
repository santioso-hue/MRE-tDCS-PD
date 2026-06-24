"""QC figures for the three-model sims: per-model |E| overlay PNGs, the MD-dMRI minus DTI difference
PNG, and the T1-space magnE NIfTIs (for fsleyes). Drawing only; the quantitative analysis of record
is qc_harness.py (tissue |E|) and 04/05/06 (per-ROI E-field and model contrast).

Usage:   simnibs_python analysis/qc_figures.py [--montage M1]
Outputs: analysis/sim_compare/<subj>/  (magnE_<model>_T1.nii.gz, diff_MDdMRI_minus_DTI_T1.nii.gz, PNGs)
"""
import os
import sys
import argparse
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sims import MODELS, sim_mesh  # noqa: E402
from simnibs import mesh_io  # noqa: E402

_ap = argparse.ArgumentParser(); _ap.add_argument("--montage", default="M1")
MONTAGE = _ap.parse_args().montage
WDIR = cfg["WORK_DIR"]; M2M = cfg["M2M_DIR"]; SUBJ = cfg["SUBJECT"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_compare", SUBJ)
os.makedirs(OUT, exist_ok=True)

t1img = nib.load(os.path.join(M2M, "T1.nii.gz"))
T1 = np.asarray(t1img.dataobj, float); aff = t1img.affine; shape = t1img.shape
seg = np.asarray(nib.load(os.path.join(M2M, "segmentation", "labeling.nii.gz")).dataobj).astype(int)
brain = np.isin(seg, [2, 41, 3, 42])    # cerebral GM+WM - the region where |E| is meaningful


def load_mesh(name):
    p = sim_mesh(WDIR, MONTAGE, name, SUBJ)
    return mesh_io.read_msh(p) if p else None


# magnE per model -> T1 grid (NIfTI for fsleyes + the overlay figures)
vols = {}
for name in MODELS:
    msh = load_mesh(name)
    if msh is None:
        print(f"  {name}: mesh missing -> skipped"); continue
    vol = np.asarray(msh.field["magnE"].interpolate_to_grid(shape, aff, method="linear", continuous=False))
    vols[name] = vol
    nib.save(nib.Nifti1Image(vol.astype(np.float32), aff, t1img.header),
             os.path.join(OUT, f"magnE_{name.replace('-', '_')}_T1.nii.gz"))
    print(f"  {name}: magnE -> magnE_{name.replace('-', '_')}_T1.nii.gz")

# MD-dMRI minus DTI difference volume (quantified per ROI in 05/06)
diff = None
if "MD-dMRI" in vols and "DTI" in vols:
    diff = vols["MD-dMRI"] - vols["DTI"]
    nib.save(nib.Nifti1Image(diff.astype(np.float32), aff, t1img.header),
             os.path.join(OUT, "diff_MDdMRI_minus_DTI_T1.nii.gz"))

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
    fig.suptitle(f"{SUBJ}  |E| (V/m), shared scale 0-{vmax:.2f}  (montage {MONTAGE})")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "compare_magnE.png"), dpi=90); plt.close(fig)
    print(f"PNG -> {OUT}/compare_magnE.png")

    # Figure 2: MD-dMRI minus DTI difference (the conductivity-model effect)
    if diff is not None:
        dlim = np.percentile(np.abs(diff[brain]), 99)
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
        for c, (bg, ov) in enumerate([(T1[ci], diff[ci]), (T1[:, cj], diff[:, cj]), (T1[:, :, ck], diff[:, :, ck])]):
            ax[c].imshow(np.rot90(bg), cmap="gray")
            im = ax[c].imshow(np.rot90(np.ma.masked_where(np.abs(ov) < 1e-3, ov)),
                              cmap="seismic", vmin=-dlim, vmax=dlim, alpha=0.8)
            ax[c].axis("off")
        fig.colorbar(im, ax=ax, fraction=0.025, label="dE V/m")
        fig.suptitle(f"{SUBJ}  MD-dMRI minus DTI  (red = MD-dMRI higher)")
        fig.savefig(os.path.join(OUT, "compare_diff_MDdMRI_DTI.png"), dpi=90); plt.close(fig)
        print(f"PNG -> {OUT}/compare_diff_MDdMRI_DTI.png")
except Exception as e:
    print(f"  PNG generation skipped: {type(e).__name__}: {e}")

print(f"Visualize:  fsleyes {os.path.join(M2M,'T1.nii.gz')} {os.path.join(OUT,'magnE_MD_dMRI_T1.nii.gz')} -cm hot")
