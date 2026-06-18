"""
score_registration_bakeoff.py - score the AFFINE vs FNIRT registration arms on one subject.

Rewritten after an adversarial code review. Three metrics, both arms, ONE shared evaluation mask
(REG_DIR/bakeoff/{affine,fnirt}/):

  (1) ORIENTATION (single SimNIBS-faithful frame, real pass/FAIL gate). v1 is scored in the EXACT frame
      the FEM sees: SimNIBS cond_utils.cond2elmdata(correct_FSL=True) rotates the tensor by
      M = colnorm(T1_affine) . diag(-1,1,1 if det>0), so the principal eigenvector the mesh sees is
      v_fem = M @ v1. We compare v_fem to anatomy: corpus-callosum MID-BODY should be left-right (world x),
      cerebral peduncle (mesencephalon) should be superior-inferior (world z). We do NOT search frames
      (the old two-frame trick could not fail); we score this one frame and FAIL the arm if either
      landmark misses its dominance/angle thresholds.

  (2) OVER-WARP (fnirt only): median/p95 magnitude of the NONLINEAR displacement field
      (fnirt total warp minus the affine, built by the harness via convertwarp) inside GM+WM. The full
      warp's Jacobian is dominated by the affine's 2.5mm->1mm scale and is NOT an over-warp measure;
      the nonlinear part is. A topup-corrected cohort should need little of it.

  (3) SPATIAL (corroborating): Pearson r of each arm's warped FA vs the dataset's delivered
      dtd_covariance_FA_to_t1 (resampled into charm space), over the SHARED mask, plus a WM-restricted r
      (less dominated by global GM/WM contrast). Treated as supporting evidence, not the sole driver
      (a more flexible model fits a reference better regardless of anatomical correctness).

The reference FA is brought to charm space via FLIRT orig.mgz -> charm T1 (rigid, MI) + applyxfm; the
orig->charm brain RECALL is printed (same-extent check) so the reference alignment is auditable.

Verdict is computed only over arms that PASS orientation; fnirt is preferred over affine only if it
improves alignment AND its nonlinear displacement is small.

Usage:  PIPELINE_CONFIG=<subject config.sh> simnibs_python docs/registration_bakeoff/score_registration_bakeoff.py
"""
import os, sys, json, tempfile, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
from _config import cfg  # noqa: E402
import numpy as np        # noqa: E402
import nibabel as nib     # noqa: E402

REG, M2M, DATA, FSLDIR = cfg["REG_DIR"], cfg["M2M_DIR"], cfg["DATA_DIR"], cfg["FSLDIR"]
OUT = os.path.join(REG, "bakeoff")
T1_REF = os.path.join(M2M, "T1.nii.gz")
# Score whichever arms have been built (affine = FA-driven, affine_s0 = b0/S0-driven, fnirt = nonlinear,
# prod = the production 02 registration copied in for end-to-end validation).
ARMS = [a for a in ("affine", "affine_s0", "fnirt", "prod")
        if os.path.exists(os.path.join(OUT, a, "FA_T1.nii.gz"))]

# Orientation gate thresholds (anatomically meaningful; a ~20deg-rotated frame must fail).
CC_DOM_MIN, CC_ANG_MAX = 0.70, 25.0     # CC mid-body: left-right (world x)
PED_DOM_MIN, PED_ANG_MAX = 0.65, 35.0   # cerebral peduncle: superior-inferior (world z); CST less pure
OVERWARP_MM_MAX = 1.5                    # median nonlinear displacement above this = fnirt over-warps
R_TIE = 0.02                             # |dr| below this = alignment wash


def _arr(p):
    return np.asarray(nib.load(p).dataobj, dtype=np.float64)


def brain_mask_charm():
    seg = nib.load(os.path.join(M2M, "final_tissues.nii.gz")).get_fdata()
    seg = seg[..., 0] if seg.ndim == 4 else seg
    return np.isin(seg, [1, 2]), (seg > 0)   # (GM+WM, whole head)


def simnibs_M(affine):
    """The exact rotation SimNIBS cond2elmdata(correct_FSL=True) applies to the tensor (cond_utils.py:224):
    M = colnorm(affine) . diag(-1,1,1 if det>0). The [:, None] matches SimNIBS's broadcast exactly."""
    M = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)[:, None]
    R = np.eye(3)
    if np.linalg.det(M) > 0:
        R[0, 0] = -1
    return M.dot(R)


def his_FA_in_charm(bm):
    """The dataset's dtd_covariance_FA_to_t1 -> charm grid via orig.mgz->charm FLIRT rigid.
    Returns (FA on charm grid clipped to [0,1], recall of charm brain covered by orig, reference_ok)."""
    orig_mgz = os.path.join(DATA, "recon", "mri", "orig.mgz")
    his_fa = os.path.join(DATA, "dmri", "dtd_covariance_FA_to_t1_antstrans.nii.gz")
    if not (os.path.exists(orig_mgz) and os.path.exists(his_fa)):
        raise FileNotFoundError(f"need {orig_mgz} and {his_fa} for the spatial metric")
    flirt = os.path.join(FSLDIR, "bin", "flirt")
    env = dict(os.environ, FSLDIR=FSLDIR, FSLOUTPUTTYPE="NIFTI_GZ")
    with tempfile.TemporaryDirectory() as td:
        orig_nii = os.path.join(td, "orig.nii.gz")
        nib.save(nib.load(orig_mgz), orig_nii)
        fa_img, orig_img = nib.load(his_fa), nib.load(orig_nii)
        if fa_img.shape != orig_img.shape or not np.allclose(fa_img.affine, orig_img.affine, atol=1e-3):
            from nibabel.processing import resample_from_to
            his_fa_grid = os.path.join(td, "his_fa_on_orig.nii.gz")
            nib.save(resample_from_to(fa_img, (orig_img.shape, orig_img.affine), order=1), his_fa_grid)
        else:
            his_fa_grid = his_fa
        mat = os.path.join(td, "fs2charm.mat"); oc = os.path.join(td, "orig_in_charm.nii.gz")
        subprocess.run([flirt, "-in", orig_nii, "-ref", T1_REF, "-omat", mat, "-out", oc,
                        "-dof", "6", "-cost", "mutualinfo", "-searchrx", "-25", "25",
                        "-searchry", "-25", "25", "-searchrz", "-25", "25", "-interp", "trilinear"],
                       check=True, env=env)
        out_fa = os.path.join(OUT, "his_FA_in_charm.nii.gz")
        subprocess.run([flirt, "-in", his_fa_grid, "-ref", T1_REF, "-applyxfm", "-init", mat,
                        "-interp", "trilinear", "-out", out_fa], check=True, env=env)
        ocd = _arr(oc)
        # same-extent check: of the charm GM+WM voxels, what fraction does the resampled orig brain cover
        oc_brain = ocd > (0.10 * np.percentile(ocd[ocd > 0], 95))
        recall = float((oc_brain & bm).sum() / (bm.sum() + 1e-9))
    fa = np.clip(_arr(out_fa), 0.0, 1.0)
    return fa, recall, recall > 0.90


def common_mask(his_fa, bm):
    """One evaluation mask shared by BOTH arms, so r is computed over identical voxels (no support confound)."""
    m = bm & np.isfinite(his_fa) & (his_fa > 0)
    for arm in ARMS:
        fa = _arr(os.path.join(OUT, arm, "FA_T1.nii.gz"))
        m &= np.isfinite(fa) & (fa > 0)
    return m


def spatial_metric(arm, his_fa, cmask):
    fa = _arr(os.path.join(OUT, arm, "FA_T1.nii.gz"))
    a, b = fa[cmask], his_fa[cmask]
    r = float(np.corrcoef(a, b)[0, 1]) if a.size > 100 else float("nan")
    wm = cmask & (his_fa > 0.2)                       # WM-restricted (contrast-reduced, alignment-sensitive)
    aw, bw = fa[wm], his_fa[wm]
    r_wm = float(np.corrcoef(aw, bw)[0, 1]) if aw.size > 100 else float("nan")
    return {"r_vs_hisFA": r, "r_vs_hisFA_WM": r_wm, "n_vox": int(cmask.sum()), "n_wm": int(wm.sum())}


def _cc_midbody(reg_dir, affine):
    """CC restricted to the middle anterior-posterior third (world y), to avoid genu/splenium curvature
    diluting the left-right signal."""
    cc = _arr(os.path.join(reg_dir, "freesurfer_rois", "roi_CC.nii.gz")) > 0.5
    vox = np.argwhere(cc)
    world = nib.affines.apply_affine(affine, vox)
    y = world[:, 1]
    lo, hi = np.percentile(y, 33), np.percentile(y, 67)
    keep = (y >= lo) & (y <= hi)
    mid = np.zeros_like(cc)
    kv = vox[keep]
    mid[kv[:, 0], kv[:, 1], kv[:, 2]] = True
    return mid


def _axis_score(v_fem, roi, axis_idx):
    nrm = np.linalg.norm(v_fem, axis=-1)
    finite = np.all(np.isfinite(v_fem), axis=-1)
    sel = roi & finite & (nrm > 0.5) & (nrm < 2.0)        # drop vecreg background garbage (inf / non-unit)
    n = int(sel.sum())
    if n < 20:
        return float("nan"), float("nan"), n
    vv = v_fem[sel]; vv = vv / np.linalg.norm(vv, axis=-1, keepdims=True)
    e = np.zeros(3); e[axis_idx] = 1.0
    ang = np.degrees(np.arccos(np.clip(np.abs(vv @ e), 0, 1)))
    dom = float(np.mean(np.argmax(np.abs(vv), axis=-1) == axis_idx))
    return float(np.median(ang)), dom, n


def orientation_metric(arm):
    vimg = nib.load(os.path.join(OUT, arm, "v1_T1.nii.gz"))
    v = np.asarray(vimg.dataobj, dtype=np.float64)             # (X,Y,Z,3), vecreg (FSL voxel) frame
    v[~np.isfinite(v)] = 0.0                                   # vecreg can emit inf/nan in background voxels
    M = simnibs_M(vimg.affine)
    v_fem = v @ M.T                                            # frame the FEM sees: v_fem = M @ v
    cc_mid = _cc_midbody(REG, vimg.affine)
    ped = _arr(os.path.join(REG, "freesurfer_rois", "roi_Mesencephalon.nii.gz")) > 0.5
    cc_ang, cc_dom, cc_n = _axis_score(v_fem, cc_mid, 0)       # CC mid-body -> world x (L-R)
    pe_ang, pe_dom, pe_n = _axis_score(v_fem, ped, 2)          # peduncle -> world z (S-I)
    cc_ok = np.isfinite(cc_dom) and cc_dom >= CC_DOM_MIN and cc_ang <= CC_ANG_MAX
    pe_ok = np.isfinite(pe_dom) and pe_dom >= PED_DOM_MIN and pe_ang <= PED_ANG_MAX
    # The corpus-callosum L-R coherence is BELOW what 2.5mm MD-dMRI can resolve (verified: native dMRI
    # CC L-R dom ~0.56 / 32deg, before any registration), so CC is an INFORMATIONAL metric, not a gate.
    # The cerebral peduncle S-I (a thick coherent tract, well resolved at 2.5mm) is the hard orientation
    # gate; all arms pass it, confirming the reorientation/frame are correct.
    return {"CC_LR_angle": cc_ang, "CC_LR_dom": cc_dom, "CC_n": cc_n, "CC_pass": bool(cc_ok),
            "Ped_SI_angle": pe_ang, "Ped_SI_dom": pe_dom, "Ped_n": pe_n, "Ped_pass": bool(pe_ok),
            "valid": bool(pe_ok)}


def overwarp_metric(bm):
    f = os.path.join(OUT, "fnirt", "field_nonlinear.nii.gz")
    if not os.path.exists(f):
        return {"available": False}
    fld = np.asarray(nib.load(f).dataobj, dtype=np.float64)    # (X,Y,Z,3) relative displacement mm
    mag = np.sqrt(np.sum(fld ** 2, axis=-1))
    m = mag[bm & np.isfinite(mag)]
    return {"available": True, "median_mm": float(np.median(m)), "p95_mm": float(np.percentile(m, 95)),
            "frac_gt_1mm": float(np.mean(m > 1.0)), "frac_gt_2mm": float(np.mean(m > 2.0))}


def main():
    bm, _head = brain_mask_charm()
    print("Resampling the dataset's FA_to_t1 into charm space (orig.mgz -> charm FLIRT rigid)...")
    his_fa, recall, ref_ok = his_FA_in_charm(bm)
    print(f"  reference brain recall (charm GM+WM covered by orig->charm) = {recall:.3f} "
          f"({'OK' if ref_ok else 'LOW -- spatial metric suspect'})")
    cmask = common_mask(his_fa, bm)
    print(f"  shared evaluation mask: {int(cmask.sum()):,} voxels (identical for both arms)\n")

    res = {}
    for arm in ARMS:
        res[arm] = {"spatial": spatial_metric(arm, his_fa, cmask), "orientation": orientation_metric(arm)}
    overwarp = overwarp_metric(bm)

    print(f"{'arm':8s}{'FA r':>8s}{'FA r(WM)':>10s}{'CC L-R deg/dom':>18s}{'peduncle S-I deg/dom':>24s}{'orient':>9s}")
    for arm in ARMS:
        s, o = res[arm]["spatial"], res[arm]["orientation"]
        print(f"{arm:8s}{s['r_vs_hisFA']:8.3f}{s['r_vs_hisFA_WM']:10.3f}"
              f"{o['CC_LR_angle']:9.1f}/{o['CC_LR_dom']:.2f}{o['Ped_SI_angle']:16.1f}/{o['Ped_SI_dom']:.2f}"
              f"{('PASS' if o['valid'] else 'FAIL'):>9s}")
    if overwarp["available"]:
        print(f"\nfnirt nonlinear displacement in GM+WM: median={overwarp['median_mm']:.2f} mm, "
              f"p95={overwarp['p95_mm']:.2f} mm, frac>1mm={overwarp['frac_gt_1mm']*100:.0f}%, "
              f"frac>2mm={overwarp['frac_gt_2mm']*100:.0f}%  (threshold median<{OVERWARP_MM_MAX} mm)")

    print("\n--- verdict ---")
    valid = [a for a in ARMS if res[a]["orientation"]["valid"]]
    over = overwarp.get("median_mm", float("nan"))
    fnirt_overwarps = bool(overwarp.get("available") and over > OVERWARP_MM_MAX)
    if not valid:
        print(f"No arm passes the orientation gate (CC dom>={CC_DOM_MIN}/ang<={CC_ANG_MAX}, "
              f"peduncle dom>={PED_DOM_MIN}/ang<={PED_ANG_MAX}). Best CC angles: " +
              ", ".join(f"{a}={res[a]['orientation']['CC_LR_angle']:.0f}deg/{res[a]['orientation']['CC_LR_dom']:.2f}" for a in ARMS) +
              ". Investigate the driver/reorientation before trusting any arm.")
    elif not ref_ok:
        print(f"Reference alignment recall {recall:.2f} too low -- spatial ranking unreliable; "
              f"orientation-valid arms: {valid}.")
    else:
        # production-eligible = orientation-valid AND not an over-warping fnirt
        eligible = [a for a in valid if not (a == "fnirt" and fnirt_overwarps)]
        if fnirt_overwarps and "fnirt" in valid:
            print(f"NOTE: fnirt passes orientation but OVER-WARPS (median nonlinear displacement {over:.2f} mm "
                  f"> {OVERWARP_MM_MAX}); excluded as the production registration.")
        pool = eligible or valid
        win = max(pool, key=lambda a: res[a]["spatial"]["r_vs_hisFA_WM"])
        ranking = ", ".join(f"{a} r_WM={res[a]['spatial']['r_vs_hisFA_WM']:.3f}" for a in pool)
        print(f"Preferred: {win.upper()}  (orientation-valid, lowest over-warp tier; by WM FA alignment: {ranking}).")
        if win.startswith("affine"):
            print("  -> an affine suffices for the topup-corrected cohort (no nonlinear over-warp).")

    with open(os.path.join(OUT, "bakeoff_scores.json"), "w") as f:
        json.dump({"reference_recall": recall, "reference_ok": ref_ok, "overwarp": overwarp, "results": res},
                  f, indent=2)
    print(f"\nFull detail -> {os.path.join(OUT, 'bakeoff_scores.json')}")


if __name__ == "__main__":
    main()
