"""_build_tier3_labels.py - warped Tier-3 nucleus prob maps -> per-nucleus L/R binary masks
(overlap ALLOWED: these nuclei are unresolvable at this resolution) plus a winner-take-all
labeled volume; reports volumes and pairwise overlap.
"""
import os, sys, json
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402

REG = cfg["REG_DIR"]; M2M = cfg["M2M_DIR"]
T3 = os.path.join(REG, "atlas_rois", "tier3")
THR = float(os.environ.get("TIER3_THR", "0.25"))           # prob threshold; set by 07 (single source)
NUCLEI = os.environ.get("TIER3_NUCLEI", "SNc SNr VTA RN STN").split()

t1 = nib.load(os.path.join(M2M, "T1.nii.gz"))
aff = t1.affine; shape = t1.shape
vx = abs(np.linalg.det(aff[:3, :3]))

# world-x of every voxel (for the L/R split); errstate silences BLAS matmul warnings
ii, jj, kk = np.indices(shape)
grid = np.vstack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)]).astype(np.float64)
with np.errstate(all="ignore"):
    xw = (np.asarray(aff, np.float64) @ grid)[0].reshape(shape)

masks = {}                      # name -> binary mask (overlap allowed)
probs = {}                      # name -> (prob masked to side) for the WTA labeled volume
for nm in NUCLEI:
    p = os.path.join(T3, f"sub_{nm}.nii.gz")
    if not os.path.exists(p):
        print(f"  WARN: {p} missing - skipping {nm}")
        continue
    d = nib.load(p).get_fdata()
    above = d >= THR
    for side, ssel in [("L", xw < 0), ("R", xw > 0)]:
        name = f"{side}_{nm}"
        m = above & ssel
        masks[name] = m
        probs[name] = np.where(m, d, 0.0)
        nib.save(nib.Nifti1Image(m.astype(np.uint8), aff, t1.header),
                 os.path.join(T3, f"roi_{name}.nii.gz"))

# winner-take-all labeled volume (visualisation only - non-overlapping)
order = list(masks.keys())
stack = np.stack([probs[n] for n in order], -1)
amax = stack.max(-1); arg = stack.argmax(-1)
labeled = np.zeros(shape, np.int16)
names = {}
for i, n in enumerate(order):
    names[i + 1] = n
    labeled[(amax > 0) & (arg == i)] = i + 1
nib.save(nib.Nifti1Image(labeled, aff, t1.header), os.path.join(T3, "tier3_labeled.nii.gz"))
with open(os.path.join(T3, "tier3_rois.json"), "w") as f:
    json.dump(names, f, indent=2)

print(f"\nTier-3 nucleus volumes (THR={THR}, overlap-allowed masks):")
for n in order:
    v = int(masks[n].sum())
    print(f"  {n:7s}: {v:4d} vox ({v*vx:.0f} mm3)")

print("\nPairwise overlap (Dice) - expected HIGH; documents the resolution caveat:")
shown = False
for a in range(len(order)):
    for b in range(a + 1, len(order)):
        na, nb = order[a], order[b]
        inter = int((masks[na] & masks[nb]).sum())
        if inter:
            va, vb = int(masks[na].sum()), int(masks[nb].sum())
            dice = 2 * inter / (va + vb)
            if dice > 0.05:
                print(f"  {na:7s} <-> {nb:7s}: Dice {dice:.2f} ({inter} shared vox)")
                shown = True
if not shown:
    print("  (no pair exceeds Dice 0.05)")
