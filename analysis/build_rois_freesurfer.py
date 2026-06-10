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
brainstemSsLabels.v13.mgz, orig.mgz. Output: registration/freesurfer_rois/{roi_labels_meshspace.nii.gz,
rois.json, roi_*.nii.gz}, the same format _rois.py / 04 / 05 consume (they prefer freesurfer_rois).

Subcortical nuclei for the MRE cross-comparison come SEPARATELY from CIT168/Pauli via ANTs
(analysis/07_build_tier3_nuclei.sh, item E) and merge into this label volume; see CIT168_NUCLEI below.

Usage (cluster):  simnibs_python analysis/build_rois_freesurfer.py --fs_dir $SUBJECTS_DIR/<subj>
"""
import os, sys, json, argparse, subprocess, tempfile
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

# CIT168 / Pauli 2018 nuclei kept SEPARATE for MRE cross-comparison (item E). Warped to subject space by
# ANTs in 07_build_tier3_nuclei.sh; ids 60+. NOTE: the integer keys are the CIT168 atlas-volume indices
# and MUST be confirmed against the specific CIT168 release 07 uses before the cohort run.
CIT168_NUCLEI = {
    "Put": 60, "Cau": 61, "NAC": 62, "GPe": 63, "GPi": 64,
    "SNc": 65, "SNr": 66, "RN": 67, "VTA": 68,
}


def assemble_labels(aparc_aseg, wmparc, brainstem_ss):
    """Pure: build one integer ROI-label volume (+ {id:name}) from three recon-all label volumes,
    all on the same grid. No I/O. CIT168 nuclei are merged later (after the ANTs warp)."""
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
            return np.asarray(nib.load(p).dataobj).astype(np.int32), nib.load(p)
    raise FileNotFoundError(f"none of {cands} in {mri}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs_dir", required=True, help="recon-all subject dir (contains mri/)")
    ap.add_argument("--outdir", default=os.path.join(cfg["REG_DIR"], "freesurfer_rois"))
    args = ap.parse_args()
    mri = os.path.join(args.fs_dir, "mri")
    aparc, ref = _load(mri, "aparc+aseg.mgz", "aparc.DKTatlas+aseg.mgz")
    wmparc, _ = _load(mri, "wmparc.mgz")
    bs, _ = _load(mri, "brainstemSsLabels.v13.mgz", "brainstemSsLabels.v12.mgz", "brainstemSsLabels.mgz")
    labels, names = assemble_labels(aparc, wmparc, bs)
    os.makedirs(args.outdir, exist_ok=True)

    fsl, charm_t1 = cfg["FSLDIR"], os.path.join(cfg["M2M_DIR"], "T1.nii.gz")
    flirt = os.path.join(fsl, "share", "fsl", "bin", "flirt")
    env = dict(os.environ, FSLDIR=fsl, FSLOUTPUTTYPE="NIFTI_GZ")
    with tempfile.TemporaryDirectory() as td:
        lab_nii = os.path.join(td, "labels_fs.nii.gz")
        nib.save(nib.Nifti1Image(labels, ref.affine, ref.header), lab_nii)
        orig_nii = os.path.join(td, "orig.nii.gz")
        nib.save(nib.load(os.path.join(mri, "orig.mgz")), orig_nii)
        mat = os.path.join(td, "fs2charm.mat")
        subprocess.run([flirt, "-in", orig_nii, "-ref", charm_t1, "-omat", mat,
                        "-dof", "6", "-cost", "mutualinfo"], check=True, env=env)
        roi_charm = os.path.join(args.outdir, "roi_labels_meshspace.nii.gz")
        subprocess.run([flirt, "-in", lab_nii, "-ref", charm_t1, "-applyxfm", "-init", mat,
                        "-interp", "nearestneighbour", "-out", roi_charm], check=True, env=env)
    json.dump({int(k): v for k, v in names.items()},
              open(os.path.join(args.outdir, "rois.json"), "w"), indent=0)
    print(f"Wrote {len(names)} ROIs -> {args.outdir} (then merge CIT168 nuclei from 07).")


if __name__ == "__main__":
    main()
