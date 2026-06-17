"""
build_rois.py — recon-all (FreeSurfer) ROI masks in charm/mesh space for the tDCS E-field analysis.

The single ROI builder, matching Olsson et al. 2025 (FreeSurfer 7.2). Needs a recon-all subject dir
(and FSL for the charm registration); the cohort ships recon-all PRE-COMPUTED in
sanoso/cohort_data/ReconAlls/<subj>, and the pilot's recon-all ran on the cluster, so this runs locally
with no recon-all step. Whenever a recon-all is needed for a new subject, it is run on the cluster.

Emits ROI masks in the charm mesh (E-field) space:
  Ctx lobes x4    roi_Ctx_{Frontal,Parietal,Temporal,Occipital}   (Desikan grouped to lobes)
  WM lobes x4     roi_WM_{Frontal,Parietal,Temporal,Occipital}    (from the REAL wmparc)
  Corpus callosum roi_CC                                           (aseg CC labels 251-255)
  Subcortical     roi_{Thalamus,Caudate,Putamen,Pallidum,Accumbens,Hippocampus,Amygdala}_{L,R}  (aseg)
  Brainstem       roi_Mesencephalon, roi_Pons                      (Iglesias 2015 brainstem module)

It also writes a single labeled volume `roi_labels_meshspace.nii.gz` and `rois.json` ({label: name}),
the interface the analysis scripts (04_extract_roi_efield.py, 05_mre_efield_comparison.py, _rois.py,
qc_harness.py) consume from registration/freesurfer_rois/. Whole-brain GM+WM is taken from the charm
tissue tags downstream (it overlaps every other ROI), not from this label volume.

Two things this does that a deep-learning seg-only parcellation cannot: WM lobes from the REAL wmparc
(3000+idx lh / 4000+idx rh), not a nearest-cortical propagation; and the brainstem split into
MESENCEPHALON (173) and PONS (174) from the Iglesias 2015 module (brainstemSsLabels), instead of the
single aseg Brain-Stem (16) which does not match Olsson. (Run `segmentBS.sh` /
`-brainstem-structures` in recon-all.) The recon-all aparc+aseg also resolves the SimNIBS atlas2subject
right-hemisphere bug (see state.md).

The fine midbrain nuclei (SNc/SNr/VTA/RN/STN) are NOT built here: the tier-3 nuclei live in
registration/atlas_rois/tier3/ (built by analysis/07_build_tier3_nuclei.sh, sampled separately as
overlap-allowed binary masks, E-field-only) and are never routed through this int-label volume.

The label-assembly logic (assemble_labels) is pure and is unit-tested without data in
tests/test_lobe_grouping.py.

Method: register recon-all orig.mgz -> charm T1 (FSL FLIRT 6-DOF rigid, same subject), assemble one
integer ROI-label volume in recon-all space, warp it once to mesh space (nearest-neighbour), split.
Lobe grouping uses the standard Desikan-Killiany index->lobe map (cingulate folded ant->Frontal /
post->Parietal; insula excluded from the four lobes). Adjust LOBE_IDX for exact parity with a
specific atlas.

Usage:  <simnibs_python> analysis/build_rois.py --fs_dir <recon-all subject dir (contains mri/)>
"""
import os, sys, json, glob, argparse, subprocess, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402

import numpy as np            # noqa: E402
import nibabel as nib         # noqa: E402

# Desikan cortical region index -> lobe (cortical label = 1000+idx lh / 2000+idx rh).
LOBE_IDX = {
    "Frontal":   [3, 12, 14, 17, 18, 19, 20, 24, 27, 28, 32, 2, 26],
    "Parietal":  [8, 10, 22, 23, 25, 29, 31],
    "Temporal":  [1, 6, 7, 9, 15, 16, 30, 33, 34],
    "Occipital": [5, 11, 13, 21],
}
LOBES = ["Frontal", "Parietal", "Temporal", "Occipital"]
LOBE_ID = {n: i + 1 for i, n in enumerate(LOBES)}   # cortex lobe ids 1..4
WM_OFFSET = 10                                       # WM lobe ids 11..14
CC_ID = 21
CC_LABELS = [251, 252, 253, 254, 255]

# Subcortical structures from the aseg: aseg label -> (ROI id, name). The aseg Brain-Stem (16) is
# replaced by the Iglesias brainstem split below, so it is dropped from the subcortical block.
ASEG = {
    10: (41, "Thalamus_L"),    49: (42, "Thalamus_R"),
    11: (43, "Caudate_L"),     50: (44, "Caudate_R"),
    12: (45, "Putamen_L"),     51: (46, "Putamen_R"),
    13: (47, "Pallidum_L"),    52: (48, "Pallidum_R"),
    26: (49, "Accumbens_L"),   58: (50, "Accumbens_R"),
    17: (51, "Hippocampus_L"), 53: (52, "Hippocampus_R"),
    18: (53, "Amygdala_L"),    54: (54, "Amygdala_R"),
    16: (55, "Brainstem"),
}
ASEG_NO_BS = {k: v for k, v in ASEG.items() if k != 16}

# Iglesias 2015 brainstem substructures (brainstemSsLabels): Midbrain=173, Pons=174 (Medulla=175,
# SCP=178 unused). Ids continue after the aseg block.
BRAINSTEM_SS = {173: (55, "Mesencephalon"), 174: (56, "Pons")}


def ctx_labels(idxs):
    return [1000 + i for i in idxs] + [2000 + i for i in idxs]


def wm_labels(idxs):
    # wmparc: the same Desikan indices, offset 3000 (lh) / 4000 (rh).
    return [3000 + i for i in idxs] + [4000 + i for i in idxs]


def assemble_labels(aparc_aseg, wmparc, brainstem_ss):
    """Pure: build one integer ROI-label volume (+ {id:name}) from three recon-all label volumes,
    all on the same grid. No I/O. The tier-3 midbrain nuclei (SNc/SNr/VTA/RN/STN) are NOT merged
    here: they live in registration/atlas_rois/tier3/ (built by 07), sampled separately as
    overlap-allowed binary masks, E-field-only."""
    assert aparc_aseg.shape == wmparc.shape == brainstem_ss.shape, "recon-all label grids differ"
    out = np.zeros(aparc_aseg.shape, np.int16)
    names = {}
    for lobe in LOBES:                                  # cortical lobes (1000/2000 + idx)
        out[np.isin(aparc_aseg, ctx_labels(LOBE_IDX[lobe]))] = LOBE_ID[lobe]
        names[LOBE_ID[lobe]] = f"Ctx_{lobe}"
    for lobe in LOBES:                                  # WM lobes from real wmparc (3000/4000 + idx)
        out[np.isin(wmparc, wm_labels(LOBE_IDX[lobe]))] = LOBE_ID[lobe] + WM_OFFSET
        names[LOBE_ID[lobe] + WM_OFFSET] = f"WM_{lobe}"
    out[np.isin(aparc_aseg, CC_LABELS)] = CC_ID
    names[CC_ID] = "CC"
    for aseg_lbl, (rid, nm) in ASEG_NO_BS.items():      # subcortical nuclei (aseg)
        out[aparc_aseg == aseg_lbl] = rid
        names[rid] = nm
    for ss_lbl, (rid, nm) in BRAINSTEM_SS.items():      # Iglesias brainstem substructures
        out[brainstem_ss == ss_lbl] = rid
        names[rid] = nm
    return out, names


def _load(mri, *cands):
    for c in cands:
        p = os.path.join(mri, c)
        if os.path.exists(p):
            return np.asarray(nib.load(p).dataobj).astype(np.int32), nib.load(p), c
    raise FileNotFoundError(f"none of {cands} in {mri}")


def _load_brainstem(mri):
    """Version-agnostic brainstemSsLabels loader. The Iglesias FS suffix (v13/v12/v2/v10) varies by
    FreeSurfer version, so glob instead of hardcoding it. The full-grid *.FSvoxelSpace.mgz is on the
    conformed grid (passes the same-grid assert); the bare *.mgz is cropped to a brainstem bounding box
    and must NOT be used."""
    fs = sorted(glob.glob(os.path.join(mri, "brainstemSsLabels*FSvoxelSpace.mgz")))
    if fs:
        p = fs[-1]
    else:
        # fall back to any brainstemSsLabels*.mgz that is NOT the cropped bare file
        cands = sorted(g for g in glob.glob(os.path.join(mri, "brainstemSsLabels*.mgz"))
                       if os.path.basename(g) != "brainstemSsLabels.mgz")
        if not cands:
            found = sorted(glob.glob(os.path.join(mri, "brainstemSsLabels*")))
            raise FileNotFoundError(
                f"no usable brainstemSsLabels*FSvoxelSpace.mgz in {mri}; "
                f"found brainstemSsLabels*: {[os.path.basename(g) for g in found]}")
        p = cands[-1]
    return np.asarray(nib.load(p).dataobj).astype(np.int32), nib.load(p), os.path.basename(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs_dir", required=True, help="recon-all subject dir (contains mri/)")
    ap.add_argument("--outdir", default=os.path.join(cfg["REG_DIR"], "freesurfer_rois"))
    args = ap.parse_args()
    mri = os.path.join(args.fs_dir, "mri")
    aparc, ref, aparc_f = _load(mri, "aparc+aseg.mgz", "aparc.DKTatlas+aseg.mgz")
    if aparc_f != "aparc+aseg.mgz":
        raise SystemExit(
            f"ERROR: LOBE_IDX is Desikan-indexed (needs aparc+aseg.mgz) but only {aparc_f} was found. "
            f"The DKT atlas drops cortical indices 1/4/6/32/33, so applying LOBE_IDX to it would SILENTLY "
            f"shrink the Frontal/Temporal lobes. Run recon-all to produce aparc+aseg.mgz (ParkMRE uses "
            f"recon-all -all, which does), or remap LOBE_IDX to DKT indices before using the DKT atlas.")
    wmparc, _, _ = _load(mri, "wmparc.mgz")
    bs, _, bs_f = _load_brainstem(mri)
    print(f"aparc used: {aparc_f}    brainstem used: {bs_f}")
    labels, names = assemble_labels(aparc, wmparc, bs)
    os.makedirs(args.outdir, exist_ok=True)

    fsl, charm_t1 = cfg["FSLDIR"], os.path.join(cfg["M2M_DIR"], "T1.nii.gz")
    flirt = os.path.join(fsl, "bin", "flirt")
    env = dict(os.environ, FSLDIR=fsl, FSLOUTPUTTYPE="NIFTI_GZ")
    roi_charm = os.path.join(args.outdir, "roi_labels_meshspace.nii.gz")
    with tempfile.TemporaryDirectory() as td:
        lab_nii = os.path.join(td, "labels_fs.nii.gz")
        nib.save(nib.Nifti1Image(labels, ref.affine, ref.header), lab_nii)
        orig_nii = os.path.join(td, "orig.nii.gz")
        nib.save(nib.load(os.path.join(mri, "orig.mgz")), orig_nii)
        mat = os.path.join(td, "fs2charm.mat")
        print("Registering recon-all orig -> charm T1 (FLIRT 6-DOF rigid)...")
        subprocess.run([flirt, "-in", orig_nii, "-ref", charm_t1, "-omat", mat,
                        "-out", os.path.join(td, "orig_in_charm.nii.gz"),
                        "-dof", "6", "-cost", "mutualinfo",
                        "-searchrx", "-20", "20", "-searchry", "-20", "20",
                        "-searchrz", "-20", "20", "-interp", "trilinear"], check=True, env=env)
        subprocess.run([flirt, "-in", lab_nii, "-ref", charm_t1, "-applyxfm", "-init", mat,
                        "-interp", "nearestneighbour", "-out", roi_charm], check=True, env=env)

    # --- split into binary masks in mesh space + write rois.json ---
    lab = np.asarray(nib.load(roi_charm).dataobj).astype(np.int16)
    aff = nib.load(charm_t1).affine
    hdr = nib.load(charm_t1).header

    def save(mask, fn):
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff, hdr), os.path.join(args.outdir, fn))
        return int(mask.sum())

    print(f"\n{'ROI (mesh space)':22s}{'voxels':>10s}")
    for rid, nm in sorted(names.items()):
        print(f"{nm:22s}{save(lab == rid, f'roi_{nm}.nii.gz'):10d}")
    with open(os.path.join(args.outdir, "rois.json"), "w") as jf:
        json.dump({int(k): v for k, v in names.items()}, jf, indent=2)

    # registration QC: overlap of warped labels with charm GM+WM tissues
    ft = os.path.join(cfg["M2M_DIR"], "final_tissues.nii.gz")
    if os.path.exists(ft):
        seg_ch = nib.load(ft).get_fdata()
        seg_ch = seg_ch[..., 0] if seg_ch.ndim == 4 else seg_ch
        warped, charm_gmwm = lab > 0, np.isin(seg_ch, [1, 2])
        dice = 2 * (warped & charm_gmwm).sum() / (warped.sum() + charm_gmwm.sum())
        print(f"\nReg QC: Dice(recon-all brain, charm GM+WM) = {dice:.3f}  (expect > ~0.85)")
    print(f"\nWrote {len(names)} ROIs + roi_labels_meshspace.nii.gz + rois.json -> {args.outdir} "
          f"(tier-3 nuclei built separately by 07).")


if __name__ == "__main__":
    main()
