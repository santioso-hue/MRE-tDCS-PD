"""
synb0_circularity_check.py — does the Synb0 (T1-derived synthetic blip-down) correction inflate the
dMRI->T1 registration, and does residual distortion reach the PD targets?

The cohort dMRI was EPI-distortion-corrected with Synb0+topup, and Synb0 synthesised the missing
blip-down b0 FROM THE T1. So the corrected b0/S0 (our registration driver) is partly T1-informed, which
could make registering it back to T1 partly circular and inflate the registration score.

Disentangling test (the raw corrected-vs-uncorrected gap alone is ambiguous — it conflates legitimate
distortion correction with circular T1-reshaping):
  - Register the UNCORRECTED b0 (MUDI2024fit/INPUTS/b0, pre-Synb0+topup) to T1 with the same affine, and
    compare its T1 edge alignment (gradient-magnitude correlation) to the CORRECTED S0's, REGIONALLY:
      * high-susceptibility (orbitofrontal + temporal): EPI distortion is large here. A gain here is
        LEGITIMATE distortion correction.
      * deep/central (basal ganglia + midbrain — the PD targets): EPI distortion is small here. A gain
        here would indicate UNIFORM T1-reshaping, i.e. circularity.
  Improvement concentrated in high-susceptibility regions => the correction fixed real distortion, not
  circular T1-pull; the registration r is genuine. (The 12-DOF affine also cannot exploit local
  T1-pre-alignment the way a nonlinear warp could — another reason the affine choice is robust here.)

Independent backstop: the peduncle S-I orientation gate (qc_harness reg_peduncle_siz) validates the
registration via the diffusion v1 DIRECTION, not b0<->T1 edge similarity, so it is immune to this
circularity. It passed (0.854).

Usage:  PIPELINE_CONFIG=<subject config.sh> simnibs_python analysis/synb0_circularity_check.py
"""
import os, sys, tempfile, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from _config import cfg  # noqa: E402
import numpy as np        # noqa: E402
import nibabel as nib     # noqa: E402
from scipy import ndimage # noqa: E402

REG, M2M, FSLDIR, SUBJ = cfg["REG_DIR"], cfg["M2M_DIR"], cfg["FSLDIR"], cfg["SUBJECT"]
ROI = os.path.join(REG, "freesurfer_rois")
T1_REF = os.path.join(M2M, "T1.nii.gz")
INPUTS_B0 = os.environ.get(
    "INPUTS_B0",
    f"/Volumes/med-avbildning-1/sanoso/cohort_data/MUDI2024fit/{SUBJ}_lin/INPUTS/b0.nii.gz")


def _arr(p):
    return np.asarray(nib.load(p).dataobj, dtype=np.float64)


def _gmag(v, sig=1.0):
    g = np.gradient(ndimage.gaussian_filter(v.astype(np.float64), sig))
    return np.sqrt(sum(gi * gi for gi in g))


def grad_corr(a, b, mask):
    """Gradient-magnitude correlation of two volumes within mask (BBR-style edge-alignment metric)."""
    ga, gb = _gmag(a)[mask], _gmag(b)[mask]
    if ga.size < 200:
        return np.nan
    ga, gb = ga - ga.mean(), gb - gb.mean()
    return float(np.corrcoef(ga, gb)[0, 1])


def roi_union(*names):
    m = None
    for n in names:
        p = os.path.join(ROI, f"roi_{n}.nii.gz")
        if not os.path.exists(p):
            continue
        d = _arr(p) > 0.5
        m = d if m is None else (m | d)
    return m


def register_uncorrected_b0():
    """FLIRT the uncorrected INPUTS/b0 -> charm T2 (same recipe as 02), warp into T1 space. Returns the
    b0-in-T1 array, or None if the share/inputs are unavailable."""
    t2b = os.path.join(REG, "T2_brain.nii.gz")
    if not (os.path.exists(INPUTS_B0) and os.path.exists(t2b)):
        return None
    flirt = os.path.join(FSLDIR, "bin", "flirt")
    env = dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ")
    with tempfile.TemporaryDirectory() as td:
        mat = os.path.join(td, "uncorr.mat")
        out = os.path.join(td, "b0_uncorr_T1.nii.gz")
        subprocess.run([flirt, "-in", INPUTS_B0, "-ref", t2b, "-omat", mat, "-dof", "12",
                        "-cost", "mutualinfo", "-searchrx", "-25", "25", "-searchry", "-25", "25",
                        "-searchrz", "-25", "25"], check=True, env=env)
        subprocess.run([flirt, "-in", INPUTS_B0, "-ref", T1_REF, "-applyxfm", "-init", mat,
                        "-interp", "trilinear", "-out", out], check=True, env=env)
        return _arr(out)


def main():
    seg = nib.load(os.path.join(M2M, "final_tissues.nii.gz")).get_fdata()
    seg = seg[..., 0] if seg.ndim == 4 else seg
    brain = ndimage.binary_erosion(np.isin(seg, [1, 2]), iterations=2)   # interior: avoid mask boundary
    t1 = _arr(T1_REF)
    s0c = _arr(os.path.join(REG, "s0_T1.nii.gz"))                        # corrected S0, registered (02)

    deep = roi_union("Mesencephalon", "Pallidum_L", "Pallidum_R", "Thalamus_L", "Thalamus_R",
                     "Putamen_L", "Putamen_R", "Caudate_L", "Caudate_R")  # low EPI distortion (PD targets)
    hisus = roi_union("Ctx_Frontal", "WM_Frontal", "Ctx_Temporal", "WM_Temporal")  # high EPI distortion
    deep = deep & brain if deep is not None else None
    hisus = hisus & brain if hisus is not None else None

    print(f"Subject {SUBJ}\n")
    print(f"CORRECTED S0 -> T1 edge alignment (grad-corr vs T1):")
    print(f"  whole brain        : {grad_corr(s0c, t1, brain):.3f}")
    if deep is not None:  print(f"  deep/central (PD)  : {grad_corr(s0c, t1, deep):.3f}  (n={int(deep.sum())})")
    if hisus is not None: print(f"  frontal+temporal   : {grad_corr(s0c, t1, hisus):.3f}  (n={int(hisus.sum())})")

    b0u = register_uncorrected_b0()
    if b0u is None:
        print("\n[uncorrected b0 unavailable -- share not mounted; circularity comparison skipped]")
        return
    print(f"\nUNCORRECTED b0 (pre-Synb0+topup) -> T1 edge alignment (same affine recipe):")
    gw_c, gw_u = grad_corr(s0c, t1, brain), grad_corr(b0u, t1, brain)
    print(f"  whole brain        : {gw_u:.3f}   (corrected {gw_c:.3f}; gain {gw_c - gw_u:+.3f})")
    if deep is not None:
        gd_c, gd_u = grad_corr(s0c, t1, deep), grad_corr(b0u, t1, deep)
        print(f"  deep/central (PD)  : {gd_u:.3f}   (corrected {gd_c:.3f}; gain {gd_c - gd_u:+.3f})")
    if hisus is not None:
        gh_c, gh_u = grad_corr(s0c, t1, hisus), grad_corr(b0u, t1, hisus)
        print(f"  frontal+temporal   : {gh_u:.3f}   (corrected {gh_c:.3f}; gain {gh_c - gh_u:+.3f})")

    print("\nInterpretation:")
    if deep is not None and hisus is not None:
        gain_deep = grad_corr(s0c, t1, deep) - grad_corr(b0u, t1, deep)
        gain_hi = grad_corr(s0c, t1, hisus) - grad_corr(b0u, t1, hisus)
        print(f"  correction gain: frontal/temporal {gain_hi:+.3f} vs deep/central {gain_deep:+.3f}")
        if gain_hi > gain_deep + 0.03:
            print("  -> gain concentrated in high-susceptibility regions = LEGITIMATE distortion correction,")
            print("     not uniform T1-reshaping. Registration r is genuine; deep PD targets not circularly inflated.")
        elif abs(gain_deep) < 0.03 and abs(gain_hi) < 0.03:
            print("  -> negligible gain everywhere = Synb0 correction was small; little distortion AND little")
            print("     circularity. Registration r is genuine.")
        else:
            print("  -> gain is substantial in low-distortion deep regions too = possible uniform T1-pull;")
            print("     treat the b0<->T1 r as partly circular and rely on the (independent) peduncle S-I gate.")


if __name__ == "__main__":
    main()
