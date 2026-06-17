"""
qc_visual.py — render visual-QC overlays for one subject (segmentation, registration, tensor orientation).

Produces PNGs under <DATA_DIR>/qc_visual/ for manual inspection of the three things worth eyeballing
before trusting/scaling the pipeline:
  qc_scalp.png        charm final_tissues over T1 (scalp/bone/brain) — the flagged scalp segmentation.
  qc_registration.png registered dMRI S0 + FA over T1 (brain-edge alignment) — the S0-affine to T1.
  qc_v1_orientation.png  DEC colour map of v1_T1 (R=L-R, G=A-P, B=S-I) in WM — tensor reorientation
                         (corpus callosum should be RED, brainstem/CST BLUE).

Usage:  PIPELINE_CONFIG=<subject config.sh> conda run -n neuro python analysis/qc_visual.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from _config import cfg  # noqa: E402
import numpy as np                 # noqa: E402
import nibabel as nib              # noqa: E402
import matplotlib                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt    # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402

REG, M2M, DATA = cfg["REG_DIR"], cfg["M2M_DIR"], cfg["DATA_DIR"]
OUT = os.path.join(DATA, "qc_visual"); os.makedirs(OUT, exist_ok=True)


def _load(p):
    return np.asarray(nib.load(p).dataobj)


def _panel(ax, bg2d, title):
    ax.imshow(bg2d.T, cmap="gray", origin="lower"); ax.set_title(title, fontsize=9); ax.axis("off")


def _slices(seg_brain):
    """Centroid voxel indices (i,j,k) of the brain, for axial/coronal/sagittal panels."""
    idx = np.argwhere(seg_brain)
    return tuple(int(round(c)) for c in idx.mean(0))


t1 = _load(os.path.join(M2M, "T1.nii.gz")).astype(float)
seg = _load(os.path.join(M2M, "final_tissues.nii.gz")); seg = seg[..., 0] if seg.ndim == 4 else seg
ci, cj, ck = _slices(np.isin(seg, [1, 2]))
vmax = np.percentile(t1[t1 > 0], 99)

# ---- 1) scalp / tissue segmentation over T1 ----
# final_tissues labels: 1 WM, 2 GM, 3 CSF, 5 scalp, 7/8 bone, others -> background.
lut = {1: 1, 2: 1, 3: 2, 5: 3, 7: 4, 8: 4}              # collapse to brain / CSF / scalp / bone
disp = np.zeros_like(seg, dtype=int)
for k, v in lut.items():
    disp[seg == k] = v
cmap = ListedColormap([(0, 0, 0, 0), (1, 0.2, 0.2, 0.45), (0.2, 0.6, 1, 0.45),
                       (1, 1, 0.2, 0.55), (0.2, 1, 0.3, 0.45)])   # bg / brain(red) / CSF(blue) / scalp(yellow) / bone(green)
norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5, 4.5], cmap.N)
fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
for ax, (sl_t1, sl_seg, ttl) in zip(axs, [
        (t1[:, :, ck], disp[:, :, ck], f"axial k={ck}"),
        (t1[:, cj, :], disp[:, cj, :], f"coronal j={cj}"),
        (t1[ci, :, :], disp[ci, :, :], f"sagittal i={ci}")]):
    ax.imshow(np.clip(sl_t1, 0, vmax).T, cmap="gray", origin="lower")
    ax.imshow(sl_seg.T, cmap=cmap, norm=norm, origin="lower"); ax.set_title(ttl, fontsize=9); ax.axis("off")
fig.suptitle("charm tissues over T1 — brain(red) CSF(blue) SCALP(yellow) bone(green). "
             "Check: scalp is a thin shell, not mislabeled air/neck.", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "qc_scalp.png"), dpi=120); plt.close(fig)

# ---- 2) registration: S0 + FA in T1 over T1 (brain-edge alignment) ----
s0 = _load(os.path.join(REG, "s0_T1.nii.gz")).astype(float)
fa = _load(os.path.join(REG, "FA_T1.nii.gz")).astype(float)
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
for row, (vol, name, cm) in enumerate([(s0, "S0->T1", "magma"), (fa, "FA->T1", "viridis")]):
    vm = np.percentile(vol[vol > 0], 99) if (vol > 0).any() else 1
    for ax, (bg, ov, ttl) in zip(axs[row], [
            (t1[:, :, ck], vol[:, :, ck], f"{name} axial"),
            (t1[:, cj, :], vol[:, cj, :], f"{name} coronal"),
            (t1[ci, :, :], vol[ci, :, :], f"{name} sagittal")]):
        ax.imshow(np.clip(bg, 0, vmax).T, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_less_equal(ov, 0).T, cmap=cm, alpha=0.5, vmax=vm, origin="lower")
        ax.set_title(ttl, fontsize=9); ax.axis("off")
fig.suptitle("Registered dMRI S0 (top) + FA (bottom) over T1 — check the dMRI brain edge tracks the T1 edge.",
             fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "qc_registration.png"), dpi=120); plt.close(fig)

# ---- 3) DEC colour map of v1_T1 (R=L-R, G=A-P, B=S-I), weighted by FA ----
v1 = _load(os.path.join(REG, "v1_T1.nii.gz")).astype(float)
w = np.clip(fa, 0, 1)[..., None]
dec = np.clip(np.abs(v1) * w * 1.4, 0, 1)                # standard directionally-encoded colour
fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
for ax, (sl, ttl) in zip(axs, [
        (dec[:, :, ck], f"axial k={ck} (CC should be RED = L-R)"),
        (dec[:, cj, :], f"coronal j={cj} (brainstem BLUE = S-I)"),
        (dec[ci, :, :], f"sagittal i={ci}")]):
    ax.imshow(np.transpose(sl, (1, 0, 2)), origin="lower"); ax.set_title(ttl, fontsize=9); ax.axis("off")
fig.suptitle("v1_T1 DEC map — R=left-right, G=ant-post, B=sup-inf, weighted by FA. "
             "Check: corpus callosum RED, corticospinal/brainstem BLUE.", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "qc_v1_orientation.png"), dpi=120); plt.close(fig)

print("wrote:")
for f in ("qc_scalp.png", "qc_registration.png", "qc_v1_orientation.png"):
    print("  " + os.path.join(OUT, f))
