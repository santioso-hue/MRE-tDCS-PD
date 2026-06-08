"""
build_rois.py — FastSurfer-based ROI masks in charm/mesh space for the tDCS E-field analysis.

The single, clean ROI builder. Takes a FastSurfer seg-only result (run locally, license-free) and
emits the cortical-lobe, white-matter-lobe, corpus-callosum, and whole-brain masks in the charm
mesh / E-field space. Subcortical nuclei (CIT168/Pauli) and brainstem (Iglesias) are NOT built here;
they come from the atlas-warp path, independent of FastSurfer.

Region set (mirrors Olsson et al. 2025, grouped to lobes):
  Ctx lobes x4   -> roi_Ctx_{Frontal,Parietal,Temporal,Occipital}.nii.gz   (DKT grouped to lobes)
  WM lobes x4    -> roi_WM_{Frontal,Parietal,Temporal,Occipital}.nii.gz    (cortical lobe propagated
                     into cerebral WM by nearest-label = a volume approximation of FreeSurfer wmparc)
  Corpus callosum-> roi_CC.nii.gz        (FastSurferCC labels 251-255, present in seg-only)
  Whole brain    -> roi_Brain.nii.gz     (all GM+WM, excl. ventricles/CSF)

Method: FastSurfer outputs live in FastSurfer-conformed space, which is NOT the charm/mesh space
where the E-field lives. We register FastSurfer orig.mgz -> charm T1 (FSL FLIRT, 6-DOF rigid, same
subject), build a single integer ROI-label volume in FastSurfer space, warp it once to mesh space
with nearest-neighbour interpolation, then split into binary masks. No surface pipeline, no license.

Lobe grouping uses the standard Desikan-Killiany-Tourville index->lobe map (documented below).
Cingulate is folded (anterior->Frontal, posterior/isthmus->Parietal); insula is excluded from the
four lobes (it still counts toward whole-brain). If exact parity with Olsson's lobe definition is
required, adjust LOBE_IDX.

Usage:
  <simnibs_python> analysis/build_rois.py --fs_dir /path/to/fastsurfer_out/<subject>
"""
import os
import sys
import argparse
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402

import numpy as np            # noqa: E402
import nibabel as nib         # noqa: E402
from scipy import ndimage     # noqa: E402

# DKT cortical region index -> lobe. Cortical label = 1000+idx (lh) / 2000+idx (rh).
LOBE_IDX = {
    "Frontal":   [3, 12, 14, 17, 18, 19, 20, 24, 27, 28, 32, 2, 26],
    "Parietal":  [8, 10, 22, 23, 25, 29, 31],
    "Temporal":  [1, 6, 7, 9, 15, 16, 30, 33, 34],
    "Occipital": [5, 11, 13, 21],
}
LOBES = ["Frontal", "Parietal", "Temporal", "Occipital"]
LOBE_ID = {n: i + 1 for i, n in enumerate(LOBES)}   # cortex lobe ids 1..4
WM_OFFSET = 10                                       # WM lobe ids 11..14
CC_ID, BRAIN_ID = 21, 31
CC_LABELS = [251, 252, 253, 254, 255]
# ventricles / CSF / background to exclude from whole-brain (FreeSurfer aseg numbering)
VENT_CSF = [0, 4, 5, 14, 15, 24, 43, 44, 31, 63, 72]


def ctx_labels(idxs):
    return [1000 + i for i in idxs] + [2000 + i for i in idxs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs_dir", required=True, help="FastSurfer subject dir (the one containing mri/)")
    args = ap.parse_args()

    mri = os.path.join(args.fs_dir, "mri")
    seg_f = os.path.join(mri, "aparc.DKTatlas+aseg.deep.withCC.mgz")
    orig_f = os.path.join(mri, "orig.mgz")
    charm_t1 = os.path.join(cfg["M2M_DIR"], "T1.nii.gz")
    outdir = os.path.join(cfg["REG_DIR"], "fastsurfer_rois")
    os.makedirs(outdir, exist_ok=True)
    for f in (seg_f, orig_f, charm_t1):
        if not os.path.exists(f):
            raise SystemExit(f"required input missing: {f}")

    fsl = cfg["FSLDIR"]
    flirt = os.path.join(fsl, "bin", "flirt")
    env = dict(os.environ, FSLDIR=fsl, FSLOUTPUTTYPE="NIFTI_GZ")

    seg_img = nib.load(seg_f)
    seg = np.asarray(seg_img.dataobj).astype(np.int32)

    # --- assemble one integer ROI-label volume in FastSurfer space ---
    roi = np.zeros(seg.shape, np.int16)
    lobe_vol = np.zeros(seg.shape, np.int16)
    for name in LOBES:
        lobe_vol[np.isin(seg, ctx_labels(LOBE_IDX[name]))] = LOBE_ID[name]
    roi[lobe_vol > 0] = lobe_vol[lobe_vol > 0]

    # WM lobes: nearest cortical-lobe label propagated into cerebral WM (2, 41) — wmparc-style
    wm = np.isin(seg, [2, 41])
    if lobe_vol.any():
        idx = ndimage.distance_transform_edt(lobe_vol == 0, return_distances=False, return_indices=True)
        nearest = lobe_vol[tuple(idx)]
        roi[wm] = nearest[wm] + WM_OFFSET            # 11..14

    roi[np.isin(seg, CC_LABELS)] = CC_ID             # corpus callosum
    brain = (seg > 0) & ~np.isin(seg, VENT_CSF)      # whole brain GM+WM (excl ventricles/CSF)
    roi[(roi == 0) & brain] = BRAIN_ID

    # --- register FastSurfer -> charm, warp the ROI-label volume to mesh space (one NN warp) ---
    with tempfile.TemporaryDirectory() as td:
        orig_nii = os.path.join(td, "orig.nii.gz")
        roi_nii = os.path.join(td, "roi.nii.gz")
        nib.save(nib.load(orig_f), orig_nii)                                   # mgz -> nii
        nib.save(nib.Nifti1Image(roi, seg_img.affine, seg_img.header), roi_nii)
        mat = os.path.join(td, "fs2charm.mat")
        print("Registering FastSurfer orig -> charm T1 (FLIRT 6-DOF rigid)...")
        subprocess.run([flirt, "-in", orig_nii, "-ref", charm_t1, "-omat", mat,
                        "-out", os.path.join(td, "orig_in_charm.nii.gz"),
                        "-dof", "6", "-cost", "mutualinfo",
                        "-searchrx", "-20", "20", "-searchry", "-20", "20",
                        "-searchrz", "-20", "20", "-interp", "trilinear"], check=True, env=env)
        roi_charm = os.path.join(outdir, "roi_labels_meshspace.nii.gz")
        subprocess.run([flirt, "-in", roi_nii, "-ref", charm_t1, "-applyxfm", "-init", mat,
                        "-interp", "nearestneighbour", "-out", roi_charm], check=True, env=env)

    # --- split into binary masks in mesh space + QC ---
    lab = np.asarray(nib.load(roi_charm).dataobj).astype(np.int16)
    aff = nib.load(charm_t1).affine
    hdr = nib.load(charm_t1).header

    def save(mask, fn):
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff, hdr), os.path.join(outdir, fn))
        return int(mask.sum())

    print(f"\n{'ROI (mesh space)':22s}{'voxels':>10s}")
    for name in LOBES:
        print(f"{'Ctx_' + name:22s}{save(lab == LOBE_ID[name], f'roi_Ctx_{name}.nii.gz'):10d}")
    for name in LOBES:
        print(f"{'WM_' + name:22s}{save(lab == LOBE_ID[name] + WM_OFFSET, f'roi_WM_{name}.nii.gz'):10d}")
    print(f"{'CC':22s}{save(lab == CC_ID, 'roi_CC.nii.gz'):10d}")
    print(f"{'Brain (all GM+WM)':22s}{save(lab > 0, 'roi_Brain.nii.gz'):10d}")

    # registration QC: overlap of warped brain with charm GM+WM tissues
    ft = os.path.join(cfg["M2M_DIR"], "final_tissues.nii.gz")
    if os.path.exists(ft):
        seg_ch = nib.load(ft).get_fdata()
        seg_ch = seg_ch[..., 0] if seg_ch.ndim == 4 else seg_ch
        charm_gmwm = np.isin(seg_ch, [1, 2])
        warped_brain = lab > 0
        inter = (warped_brain & charm_gmwm).sum()
        dice = 2 * inter / (warped_brain.sum() + charm_gmwm.sum())
        print(f"\nReg QC: Dice(FastSurfer brain, charm GM+WM) = {dice:.3f}  (expect > ~0.85)")
    print(f"\nMasks -> {outdir}")


if __name__ == "__main__":
    main()
