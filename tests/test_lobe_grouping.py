"""
test_lobe_grouping.py — validate the recon-all ROI label schemes without recon-all data.

Checks the DK->lobe mapping is a clean partition (no overlap, insula and CC excluded) and that
assemble_labels (build_rois) places cortical lobes, real-wmparc WM lobes, CC, aseg subcortical,
and the Iglesias brainstem substructures at the right ids.

Usage:  conda run -n neuro python tests/test_lobe_grouping.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
from build_rois import LOBE_IDX, LOBE_ID, WM_OFFSET, CC_ID, assemble_labels  # noqa: E402

checks = []
def ck(name, ok, detail=""):
    checks.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

# 1. lobe mapping is a clean partition of the DK cortical indices
allidx = [i for L in LOBE_IDX.values() for i in L]
ck("no DK index in two lobes", len(allidx) == len(set(allidx)))
ck("insula (35) excluded from lobes", 35 not in allidx)
ck("corpus-callosum index (4) excluded", 4 not in allidx)
ck("33 DK cortical regions grouped", len(set(allidx)) == 33, f"(got {len(set(allidx))})")

# 2. assemble_labels places everything correctly
sh = (4, 4, 4)
ap = np.zeros(sh, int); wm = np.zeros(sh, int); bs = np.zeros(sh, int)
ap[0, 0, 0] = 1000 + 3      # lh caudalmiddlefrontal  -> Ctx_Frontal
ap[0, 0, 1] = 2000 + 8      # rh inferiorparietal     -> Ctx_Parietal
ap[0, 1, 0] = 251           # corpus callosum
ap[0, 1, 1] = 10            # left thalamus
wm[1, 0, 0] = 3000 + 3      # lh frontal WM           -> WM_Frontal
wm[1, 0, 1] = 4000 + 5      # rh cuneus WM            -> WM_Occipital
bs[2, 0, 0] = 173           # midbrain                -> Mesencephalon
bs[2, 0, 1] = 174           # pons                    -> Pons
lab, names = assemble_labels(ap, wm, bs)

ck("Ctx_Frontal", lab[0, 0, 0] == LOBE_ID["Frontal"])
ck("Ctx_Parietal", lab[0, 0, 1] == LOBE_ID["Parietal"])
ck("CC", lab[0, 1, 0] == CC_ID)
ck("Thalamus_L (aseg 10 -> 41)", lab[0, 1, 1] == 41)
ck("WM_Frontal from wmparc", lab[1, 0, 0] == LOBE_ID["Frontal"] + WM_OFFSET)
ck("WM_Occipital from wmparc", lab[1, 0, 1] == LOBE_ID["Occipital"] + WM_OFFSET)
ck("Mesencephalon separate (173)", lab[2, 0, 0] == 55 and names[55] == "Mesencephalon")
ck("Pons separate (174)", lab[2, 0, 1] == 56 and names[56] == "Pons")
ck("no single 'Brainstem' label", "Brainstem" not in names.values())

print(f"\n{'ALL PASS' if all(checks) else 'SOME CHECKS FAILED'}")
sys.exit(0 if all(checks) else 1)
