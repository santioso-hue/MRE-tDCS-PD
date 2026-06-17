"""
cerebellum_coverage_check.py — is the MD-dMRI cerebellum coverage gap caused by the Synb0 correction,
or is it inherent to the dMRI acquisition (and therefore present in the uncorrected data too)?

MD-dMRI conductivity is anisotropic only where the QTI tensor reached (lam1 > EPS); elsewhere it falls
back to scalar sigma0, but the |E| view shows the cerebellum thinning because the tensor is absent there.
Synb0+topup is a within-FOV distortion correction (it cannot add un-acquired tissue), so the test is:
does the UNCORRECTED b0 (pre-Synb0) have cerebellum signal that the corrected fit lacks (=> Synb0
artifact), or do BOTH b0s lack it (=> dMRI FOV/SNR limit, present regardless of correction)?

Usage:  PIPELINE_CONFIG=<subject config.sh> simnibs_python analysis/cerebellum_coverage_check.py
"""
import os, sys, tempfile, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from _config import cfg  # noqa: E402
import numpy as np        # noqa: E402
import nibabel as nib     # noqa: E402

REG, M2M, DATA, FSLDIR, SUBJ = cfg["REG_DIR"], cfg["M2M_DIR"], cfg["DATA_DIR"], cfg["FSLDIR"], cfg["SUBJECT"]
FIT = os.path.join(os.path.dirname(cfg["QTI_MFS"]))   # .../fit/qti_cov
T1_REF = os.path.join(M2M, "T1.nii.gz")
B0_CORR = os.path.join(FIT, "b0_ref_dMRI.nii.gz")     # corrected (post Synb0+topup) fit b0
B0_UNCORR = os.environ.get("INPUTS_B0",
    f"/Volumes/med-avbildning-1/sanoso/cohort_data/MUDI2024fit/{SUBJ}_lin/INPUTS/b0.nii.gz")
flirt = os.path.join(FSLDIR, "bin", "flirt")
cxfm = os.path.join(FSLDIR, "bin", "convert_xfm")
env = dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ")


def _arr(p): return np.asarray(nib.load(p).dataobj, dtype=np.float64)


def main():
    with tempfile.TemporaryDirectory() as td:
        # 1) cerebellum mask in FreeSurfer space (aparc+aseg labels 7/8 L, 46/47 R), -> charm via orig->T1 flirt
        aseg_img = nib.load(os.path.join(DATA, "recon", "mri", "aparc+aseg.mgz"))
        cb = np.isin(np.asarray(aseg_img.dataobj).astype(int), [7, 8, 46, 47]).astype(np.float32)
        cb_fs = os.path.join(td, "cb_fs.nii.gz"); nib.save(nib.Nifti1Image(cb, aseg_img.affine), cb_fs)
        orig = os.path.join(td, "orig.nii.gz"); nib.save(nib.load(os.path.join(DATA, "recon", "mri", "orig.mgz")), orig)
        fs2charm = os.path.join(td, "fs2charm.mat")
        subprocess.run([flirt, "-in", orig, "-ref", T1_REF, "-omat", fs2charm, "-dof", "6",
                        "-cost", "mutualinfo", "-searchrx", "-25", "25", "-searchry", "-25", "25",
                        "-searchrz", "-25", "25"], check=True, env=env)
        cb_charm = os.path.join(td, "cb_charm.nii.gz")
        subprocess.run([flirt, "-in", cb_fs, "-ref", T1_REF, "-applyxfm", "-init", fs2charm,
                        "-interp", "nearestneighbour", "-out", cb_charm], check=True, env=env)
        cbc = _arr(cb_charm) > 0.5

        # 2) MD-dMRI tensor coverage within the cerebellum (charm space)
        lam1 = _arr(os.path.join(REG, "lam1_T1.nii.gz"))
        cov = float((cbc & (lam1 > 1e-3)).sum() / max(int(cbc.sum()), 1))
        print(f"cerebellum (charm): {int(cbc.sum()):,} voxels; MD-dMRI tensor coverage = {cov*100:.1f}% "
              f"(rest falls back to scalar sigma0)")

        # 3) cerebellum -> dMRI space (invert the production affine), compare corrected vs uncorrected b0 there
        if not os.path.exists(B0_UNCORR):
            print("\n[uncorrected INPUTS/b0 unavailable -- share not mounted; corrected-vs-uncorrected skipped]")
            return
        inv = os.path.join(td, "charm2dmri.mat")
        subprocess.run([cxfm, "-inverse", os.path.join(REG, "dMRI_to_T1_aff.mat"), "-omat", inv], check=True, env=env)
        cb_dmri = os.path.join(td, "cb_dmri.nii.gz")
        subprocess.run([flirt, "-in", cb_charm, "-ref", B0_CORR, "-applyxfm", "-init", inv,
                        "-interp", "nearestneighbour", "-out", cb_dmri], check=True, env=env)
        cbd = _arr(cb_dmri) > 0.5
        b0c, b0u = _arr(B0_CORR), _arr(B0_UNCORR)
        thr_c = 0.10 * np.percentile(b0c[b0c > 0], 95)
        thr_u = 0.10 * np.percentile(b0u[b0u > 0], 95)
        sig_c = float((cbd & (b0c > thr_c)).sum() / max(int(cbd.sum()), 1))
        sig_u = float((cbd & (b0u > thr_u)).sum() / max(int(cbd.sum()), 1))
        print(f"\ncerebellum in dMRI space: {int(cbd.sum()):,} voxels")
        print(f"  with signal in CORRECTED   b0 (post Synb0+topup): {sig_c*100:.1f}%")
        print(f"  with signal in UNCORRECTED b0 (pre  Synb0+topup): {sig_u*100:.1f}%")
        print("\nVerdict:")
        if abs(sig_c - sig_u) < 0.05:
            print("  corrected and uncorrected cerebellum coverage MATCH -> the gap is the dMRI ACQUISITION")
            print("  (FOV/SNR at the inferior edge), present in BOTH; Synb0 did not cause or worsen it.")
        elif sig_u - sig_c > 0.05:
            print("  uncorrected has MORE cerebellum than corrected -> Synb0+topup pushed/dropped cerebellum signal.")
        else:
            print("  corrected has more (correction recovered edge signal); not a Synb0 loss.")


if __name__ == "__main__":
    main()
