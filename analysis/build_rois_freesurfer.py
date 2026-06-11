"""
build_rois_freesurfer.py — recon-all (FreeSurfer) ROI masks in charm/mesh space. The cohort-definitive
parcellation (replaces FastSurfer build_rois.py), to match Olsson 2025. CLUSTER-ONLY: needs recon-all
outputs (and FSL for the charm registration). The label-assembly logic (assemble_labels) is pure and is
unit-tested without data in tests/test_lobe_grouping.py.

Two things this does that the FastSurfer builder does not:
  - WM lobes from the REAL wmparc (3000+idx lh / 4000+idx rh), not a nearest-cortical propagation.
  - Brainstem from the Iglesias 2015 module (brainstemSsLabels): MESENCEPHALON (173) and PONS (174)
    SEPARATE, instead of the single aseg Brain-Stem (16) — which the aseg label does not split and
    which does not match Olsson. (Run `segmentBS.sh` / `-brainstem-structures` in recon-all.)
The recon-all aparc+aseg also resolves the SimNIBS atlas2subject right-hemisphere bug (state.md).

Inputs (SUBJECTS_DIR/<subj>/mri/): aparc+aseg.mgz (or aparc.DKTatlas+aseg.mgz), wmparc.mgz,
brainstemSsLabels*.mgz, orig.mgz. Output: registration/freesurfer_rois/{roi_labels_meshspace.nii.gz,
rois.json, roi_*.nii.gz}, the same format _rois.py / 04 / 05 consume (they prefer freesurfer_rois).

The fine midbrain nuclei (SNc/SNr/VTA/RN/STN) are NOT built here: the tier-3 nuclei live in
registration/atlas_rois/tier3/ (built by analysis/07_build_tier3_nuclei.sh, sampled separately as
overlap-allowed binary masks, E-field-only) and are never routed through this int-label volume.

Usage (cluster):  simnibs_python analysis/build_rois_freesurfer.py --fs_dir $SUBJECTS_DIR/<subj>
"""
import os, sys, json, glob, argparse, subprocess, tempfile
import numpy as np
import nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from _config import cfg  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# reuse the single-sourced lobe mapping + subcortical scheme from the FastSurfer builder
from build_rois import LOBE_IDX, LOBES, LOBE_ID, WM_OFFSET, CC_ID, CC_LABELS, ASEG, ctx_labels  # noqa: E402

# WM parcellation labels (wmparc): the same DK indices, offset 3000 (lh) / 4000 (rh).
def wm_labels(idxs):
    return [3000 + i for i in idxs] + [4000 + i for i in idxs]


# Iglesias 2015 brainstem substructures (brainstemSsLabels): Midbrain=173, Pons=174 (Medulla=175,
# SCP=178 unused). Replace the single aseg Brain-Stem (16); ids continue after the aseg block.
BRAINSTEM_SS = {173: (55, "Mesencephalon"), 174: (56, "Pons")}
ASEG_NO_BS = {k: v for k, v in ASEG.items() if k != 16}

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
    json.dump({int(k): v for k, v in names.items()},
              open(os.path.join(args.outdir, "rois.json"), "w"), indent=0)

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
