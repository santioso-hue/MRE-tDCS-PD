"""_build_nuclei_labels.py - warped deep-nucleus (Group 2) prob maps -> per-nucleus L/R binary masks
(overlap ALLOWED) plus a winner-take-all labeled volume; reports volumes and pairwise overlap. The basal
ganglia barely overlap; the midbrain and subthalamic nuclei do (reported, not flagged: the resolution caveat).
"""
import os, sys, json
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402

REG = cfg["REG_DIR"]; M2M = cfg["M2M_DIR"]
ND = os.path.join(REG, "atlas_rois", "nuclei")
THR = float(os.environ.get("NUCLEI_THR", "0.25"))          # prob threshold; set by 07 (single source)
NUCLEI = os.environ.get("NUCLEI_LIST", "Pu Ca NAC GPe GPi SNc SNr VTA RN STN").split()

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
    p = os.path.join(ND, f"sub_{nm}.nii.gz")
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
                 os.path.join(ND, f"roi_{name}.nii.gz"))

# winner-take-all labeled volume (visualisation only - non-overlapping)
order = list(masks.keys())
stack = np.stack([probs[n] for n in order], -1)
amax = stack.max(-1); arg = stack.argmax(-1)
labeled = np.zeros(shape, np.int16)
names = {}
for i, n in enumerate(order):
    names[i + 1] = n
    labeled[(amax > 0) & (arg == i)] = i + 1
nib.save(nib.Nifti1Image(labeled, aff, t1.header), os.path.join(ND, "nuclei_labeled.nii.gz"))
with open(os.path.join(ND, "nuclei_rois.json"), "w") as f:
    json.dump(names, f, indent=2)

print(f"\nDeep-nucleus (Group 2) volumes (THR={THR}, overlap-allowed masks):")
for n in order:
    v = int(masks[n].sum())
    print(f"  {n:8s}: {v:5d} vox ({v*vx:.0f} mm3)")

# Pairwise overlap (Dice): basal ganglia should be LOW (well-resolved); midbrain/STN can be HIGH
# (sub-voxel, inter-nucleus overlap) - documents the §2.7 resolution caveat, reported not flagged.
print("\nPairwise overlap (Dice > 0.05):")
shown = False
for a in range(len(order)):
    for b in range(a + 1, len(order)):
        na, nb = order[a], order[b]
        inter = int((masks[na] & masks[nb]).sum())
        if inter:
            va, vb = int(masks[na].sum()), int(masks[nb].sum())
            dice = 2 * inter / (va + vb)
            if dice > 0.05:
                print(f"  {na:8s} <-> {nb:8s}: Dice {dice:.2f} ({inter} shared vox)")
                shown = True
if not shown:
    print("  (no pair exceeds Dice 0.05)")
