"""
qc_harness.py - post-pipeline per-subject QC: one qc_summary.csv row per subject with sanity
metrics, a PASS/FLAG per stage, and an overall verdict. Beyond the absolute thresholds, any
metric > 3 MAD from the cohort median is also flagged. An outlier is a signal to inspect/fix.

Usage:
  simnibs_python analysis/qc_harness.py                       # the subject in config/config.sh
  simnibs_python analysis/qc_harness.py --cohort cohort.json  # [{"id","work","m2m"}...]
  simnibs_python analysis/qc_harness.py --calibrate           # suggest cohort-percentile thresholds, then exit
"""
import os, sys, csv, json, glob, argparse
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from _config import cfg  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _sims import sim_mesh, MODELS  # noqa: E402  (shared montage-aware mesh lookup)
from _rois import eigvals_6comp, fa_from_evals  # noqa: E402  (shared eigendecomp + FA; one source of truth)

# Absolute thresholds seeded from a single-subject pilot (n=1). [PROV] items are provisional
# single-subject cutoffs to recalibrate as a cohort percentile (--calibrate) once 5-8 subjects
# exist. The MAD>3 outlier pass in main() (active only at n>=4) is the real per-subject guard;
# on one subject only these absolute thresholds are in force.
THR = dict(
    charm_tet_q_med_min=0.10, charm_tet_q_badfrac_max=0.05,  # broadly-poor mesh / too many slivers
    tissue_plausible={"WM": (3e5, 7e5), "GM": (4e5, 8e5), "CSF": (5e4, 5e5),
                      "bone": (2e5, 8e5), "scalp": (5e5, 2e6)},   # mm3 ranges
    cc_fa_min=0.45, csf_fa_max=0.25,                 # dwi2cond
    reg_containment_min=0.95,                        # dMRI FOV covers brain (gross misreg / cutoff)
    cc_v1x_min=0.6,                                  # [PROV] CC V1 left-right (tensor reorientation)
    peduncle_siz_min=0.55,                           # [PROV] cerebral-peduncle V1 superior-inferior in the SimNIBS frame
    reg_grad_corr_min=0.08,                          # [PROV] b0<->T1 edge alignment (mid-range shift)
    geom_mean_tol=0.01,                              # vn reconstruction must land on sigma0 within 1%
    efield_p95_range=(0.10, 0.60), efield_spike=8.0, # 03: p95 |E| range; max/p95 spike [PROV] default mesh ~6 (tail artifact, not the median readout); recalibrate at cohort scale
    electrode_tol_mm=15.0,
)
SIGMA0 = {"WM": 0.126, "GM": 0.275}                  # tissue baseline conductivities (S/m), vn anchors
SAMSEG = dict(wm=(2, 41), gm=(3, 42), csf=(4, 43, 14, 15, 24), brainstem=(16,))
NUCLEI = ["SNc", "SNr", "VTA", "RN", "STN"]


def _load(p):
    return nib.load(p) if os.path.exists(p) else None


def _data(p):
    img = _load(p)
    return np.asarray(img.dataobj) if img is not None else None


def _roi_dir(reg):
    """Tier-1 ROI dir: the recon-all parcellation (freesurfer_rois) built by analysis/build_rois.py.
    Mirrors analysis/_rois.load_labeled so QC reads the same masks."""
    return os.path.join(reg, "freesurfer_rois")


def _eigs_masked(t, sel):
    """Ascending eigenvalues (N,3) of the 6-comp tensor `t` over the selected voxels only. Delegates to
    _rois.eigvals_6comp (one source of truth for the FSL pack + the eigvalsh values-only fast path)."""
    return eigvals_6comp(t, sel)


def _tensor_eigs(path, mask=None):
    """Return sorted eigenvalues (N,3) of a 6-comp tensor NIfTI, optionally within mask."""
    t = _data(path)
    if t is None:
        return None
    sel = (np.abs(t[..., 0]) > 1e-9) if mask is None else mask
    return _eigs_masked(t, sel)


def _fa(eigs):
    """Mean FA over voxels, from ascending eigenvalues (N,3); uses _rois.fa_from_evals (per-voxel)."""
    return np.nanmean(fa_from_evals(eigs))


def _vn_check(e, sigma0):
    """Semi-direct volume-normalization check on diffusion-tensor eigenvalues `e` (N,3, ascending).

    SimNIBS's vn step scales each voxel's D to sigma = D * sigma0 / geomean(eig D), forcing
    geomean(eig sigma) == sigma0. We can't read SimNIBS's internal sigma, only verify D supports it.
    Returns (degfrac, vn_err, sigma_hi, sigma_lo):
      degfrac  - fraction NOT positive-definite (vn undefined there). MUST be read from the raw
                 smallest eigenvalue e[:,0]: a clamped geom-mean is always > 0 and would silently
                 report 0 (the bug this function exists to prevent).
      vn_err   - |median(geomean(sigma)) - sigma0| / sigma0 over valid voxels (~0 if vn holds).
      sigma_hi/lo - 99th/1st percentile of reconstructed conductivity eigenvalues (physiological?).
    """
    good = np.isfinite(e).all(axis=1) & (e[:, 0] > 0)
    degfrac = float(np.mean(~good))
    if not good.any():
        return degfrac, float("nan"), float("nan"), float("nan")
    eg = e[good]                                       # positive-definite -> no clamp needed
    g = np.exp(np.mean(np.log(eg), axis=1))
    sig = eg * (sigma0 / g[:, None])                   # reconstructed conductivity eigenvalues
    sg = np.exp(np.mean(np.log(sig), axis=1))          # == sigma0 if vn holds
    vn_err = float(abs(np.median(sg) - sigma0) / sigma0)
    return degfrac, vn_err, float(np.percentile(sig, 99)), float(np.percentile(sig, 1))


# stage checks: each returns (metrics dict, list of flag strings)
def qc_charm(P):
    m, f = {}, []
    m2m = P["m2m"]
    mesh = next(iter(glob.glob(os.path.join(m2m, "*.msh"))), None)
    ft = os.path.join(m2m, "final_tissues.nii.gz")
    m["charm_m2m"] = int(os.path.isdir(m2m)); m["charm_mesh"] = int(mesh is not None)
    m["charm_tissues"] = int(os.path.exists(ft))
    if not (m["charm_m2m"] and mesh and m["charm_tissues"]):
        f.append("charm:missing_output"); return m, f
    try:
        from simnibs import mesh_io
        msh = mesh_io.read_msh(mesh)
        m["charm_n_tet"] = int((msh.elm.elm_type == 4).sum())
        try:
            # tetrahedra_quality() -> (radius_edge_ratio, inscribed/circumscribed ratio); [1] is the
            # standard tet quality (1.0 = ideal, low = sliver), sized to ALL elements so triangles are
            # NaN -> keep finite (the tets). Gate on median + sliver fraction, not min (a few slivers
            # are normal and the FEM handles them).
            q = msh.tetrahedra_quality()[1].value
            qt = q[np.isfinite(q)]
            m["charm_tet_q_med"] = round(float(np.median(qt)), 3)
            m["charm_tet_q_badfrac"] = round(float(np.mean(qt < 0.1)), 4)
            if np.median(qt) < THR["charm_tet_q_med_min"] or np.mean(qt < 0.1) > THR["charm_tet_q_badfrac_max"]:
                f.append("charm:low_tet_quality")
        except Exception:
            m["charm_tet_q_med"] = np.nan; m["charm_tet_q_badfrac"] = np.nan
    except Exception as e:
        m["charm_n_tet"] = np.nan; m["charm_tet_q_med"] = np.nan; m["charm_tet_q_badfrac"] = np.nan
        f.append(f"charm:mesh_read({type(e).__name__})")
    seg = _data(ft); vx = abs(np.linalg.det(_load(ft).affine[:3, :3]))
    # SimNIBS charm final_tissues labels: WM=1, GM=2, CSF=3, scalp=5; bone is SPLIT into 7=compact +
    # 8=spongy (generic bone label 4 is EMPTY in charm output), so total bone = 7+8.
    lut = {"WM": [1], "GM": [2], "CSF": [3], "bone": [7, 8], "scalp": [5]}
    for name, labs in lut.items():
        v = float(np.isin(seg, labs).sum() * vx); m[f"vol_{name}_mm3"] = round(v)
        lo, hi = THR["tissue_plausible"].get(name, (0, 1e12))
        if not (lo <= v <= hi):
            f.append(f"charm:vol_{name}_implausible")
    return m, f


def qc_dwi2cond(P):
    m, f = {}, []
    dti = os.path.join(P["m2m"], "DTI_coregT1_tensor.nii.gz")
    if not os.path.exists(dti):
        f.append("dwi2cond:missing"); return m, f
    seg = _data(P["samseg"])
    cc = _data(P.get("cc_mask")) if P.get("cc_mask") else None
    cc_mask = (cc > 0) if cc is not None else None
    csf_mask = np.isin(seg, SAMSEG["csf"]) if seg is not None else None
    if cc_mask is not None:
        ec = _tensor_eigs(dti, cc_mask); m["dwi2cond_cc_fa"] = round(_fa(ec), 3)
        if m["dwi2cond_cc_fa"] < THR["cc_fa_min"]:
            f.append("dwi2cond:cc_fa_low")
    if csf_mask is not None:
        es = _tensor_eigs(dti, csf_mask); m["dwi2cond_csf_fa"] = round(_fa(es), 3)
        if m["dwi2cond_csf_fa"] > THR["csf_fa_max"]:
            f.append("dwi2cond:csf_fa_high")
    eall = _tensor_eigs(dti)
    m["dwi2cond_neg_eig_frac"] = round(float(np.mean(eall[:, 0] <= 0)), 4) if eall is not None else np.nan
    if eall is not None and (eall[:, 0] <= 0).mean() > 0.001:
        f.append("dwi2cond:neg_eigenvalues")
    return m, f


def qc_register(P):
    m, f = {}, []
    seg = _data(P["samseg"])
    dm = _data(os.path.join(P["reg"], "dMRI_mask_T1.nii.gz"))
    if seg is None or dm is None:
        f.append("register:missing"); return m, f
    # The registered dMRI brain mask is ~2x the SAMSEG brain (includes neck/inferior FOV), so Dice or
    # centroid distance is size-confounded (caps ~0.62 / ~8 mm even when aligned). Use size-INDEPENDENT
    # signals instead:
    #   (1) containment - does the dMRI FOV cover the brain? Catches gross misreg / FOV cutoff.
    #   (2) CC reorientation - do corpus-callosum V1 point left-right? Catches tensor-reorient errors
    #       (highest-risk step: a flip/rotation collapses |V1_x| in the CC).
    brain = np.isin(seg, [2, 41, 3, 42, 10, 49, 11, 50, 12, 51, 13, 52, 16, 17, 18, 53, 54, 26, 58, 28, 60])
    dmm = dm > 0
    m["reg_containment"] = round(float((dmm & brain).sum() / max(int(brain.sum()), 1)), 4)
    if m["reg_containment"] < THR["reg_containment_min"]:
        f.append("register:brain_not_covered")
    v1 = _data(os.path.join(P["reg"], "v1_T1.nii.gz"))
    cc = _data(P.get("cc_mask")) if P.get("cc_mask") else None
    if v1 is not None and cc is not None and (cc > 0).any():
        ccm = cc > 0
        nrm = np.linalg.norm(v1[ccm], axis=1)
        ok = nrm > 1e-6
        if ok.any():
            v1x = np.abs(v1[ccm][ok, 0]) / nrm[ok]
            m["reg_cc_v1x"] = round(float(np.median(v1x)), 3)   # median: robust to CC-edge partial volume
            if np.median(v1x) < THR["cc_v1x_min"]:
                f.append("register:cc_not_LR")   # tensor reorientation failed
    # (2b) cerebral-peduncle reorientation in the SimNIBS frame. CC V1x above is voxel-frame and the
    # CC L-R signal is resolution-limited at 2.5 mm; the peduncle is a thick coherent S-I tract, the
    # reliable orientation gate. SimNIBS rotates the tensor by M = colnorm(affine).diag(-1,1,1 if det>0)
    # (cond_utils.cond2elmdata) before the FEM, so we score M@v1 against world-z (superior-inferior).
    ped = _data(os.path.join(_roi_dir(P["reg"]), "roi_Mesencephalon.nii.gz"))
    if v1 is not None and ped is not None and (ped > 0).any():
        aff = nib.load(os.path.join(P["reg"], "v1_T1.nii.gz")).affine[:3, :3]
        M = aff / np.linalg.norm(aff, axis=0)[:, None]
        if np.linalg.det(M) > 0:
            M = M.dot(np.diag([-1.0, 1.0, 1.0]))
        vw = v1[ped > 0] @ M.T
        nrm = np.linalg.norm(vw, axis=1); ok = nrm > 1e-6
        if ok.any():
            siz = np.abs(vw[ok, 2]) / nrm[ok]
            m["reg_peduncle_siz"] = round(float(np.median(siz)), 3)
            if np.median(siz) < THR["peduncle_siz_min"]:
                f.append("register:peduncle_not_SI")   # reorientation / registration off
    # (3) edge alignment - gradient-magnitude correlation of registered b0 vs T1 inside the brain.
    # Containment and CC-V1 miss a few-mm rigid shift that still covers the brain; a shift misaligns
    # tissue boundaries, so |grad| correlation drops sharply (BBR principle): ~0.19 aligned, ~-40% at
    # a 4 mm shift. (NMI on this b0/T1 pair sits near the information floor and barely moves, so unused.)
    try:
        from scipy import ndimage
        b0 = _data(os.path.join(P["reg"], "s0_T1.nii.gz"))   # registered dMRI S0 in T1 space
        t1 = _data(os.path.join(P["m2m"], "T1.nii.gz"))
        if b0 is not None and t1 is not None:
            inner = ndimage.binary_erosion(brain, iterations=2)   # avoid the mask boundary itself

            def _gmag(v):
                gx, gy, gz = np.gradient(ndimage.gaussian_filter(v.astype(float), 1.0))
                return np.sqrt(gx * gx + gy * gy + gz * gz)
            a = _gmag(b0)[inner]; b = _gmag(t1)[inner]
            a = a - a.mean(); b = b - b.mean()
            gc = float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-30))
            m["reg_grad_corr"] = round(gc, 3)
            if gc < THR["reg_grad_corr_min"]:
                f.append("register:edge_misaligned")   # [PROV] likely a few-mm shift
    except Exception:
        pass   # scipy missing or b0 absent -> skip; containment+CC still gate
    return m, f


def qc_conductivity(P):
    m, f = {}, []
    seg = _data(P["samseg"])
    # The MD-dMRI conductivity input is the plain QTI mean tensor <D> (tensor_MD_dMRI.nii.gz).
    tensors = [("MDdMRI", "tensor_MD_dMRI.nii.gz")]
    for tag, fname in tensors:
        path = os.path.join(P["work"], fname)
        if not os.path.exists(path):
            f.append(f"cond:{tag}_missing"); continue
        t = _data(path)                                  # load once; reuse for global + per-tissue
        naninf = bool(np.isnan(t).any() or np.isinf(t).any())
        m[f"cond_{tag}_nan"] = int(naninf)
        if naninf:
            f.append(f"cond:{tag}_naninf")
        valid = np.abs(t[..., 0]) > 1e-9
        if not valid.any():
            f.append(f"cond:{tag}_empty"); continue
        eigs = _eigs_masked(t, valid)
        negfrac = float(np.mean(eigs[:, 0] <= 0))
        m[f"cond_{tag}_neg_eig_frac"] = round(negfrac, 5)
        if negfrac > 0:
            f.append(f"cond:{tag}_neg_eig")
        ratio = eigs[:, 2] / np.maximum(eigs[:, 0], 1e-9)
        m[f"cond_{tag}_cap8_frac"] = round(float(np.mean(ratio > 8)), 4)
        # Per-tissue mean diffusivity = geom-mean of the D eigenvalues. tensor_MD_dMRI is the DIFFUSION
        # tensor; SimNIBS applies vn (geom-mean sigma -> sigma0) at sim time, so sigma0 is verified
        # INDIRECTLY via the E-field range in qc_sims. Here we sanity-check the tensor: positive, real
        # MD-scale, not corrupt/zeroed.
        if seg is not None:
            for tis, labs in [("WM", SAMSEG["wm"]), ("GM", SAMSEG["gm"])]:
                tm = np.isin(seg, labs) & valid
                if not tm.any():
                    continue
                e = _eigs_masked(t, tm)
                md_med = float(np.median(np.exp(np.mean(np.log(np.maximum(e, 1e-12)), axis=1))))
                m[f"cond_{tag}_{tis}_md"] = round(md_med, 3)
                if not (0.2 <= md_med <= 3.0):   # plausible mean diffusivity band (um^2/ms)
                    f.append(f"cond:{tag}_{tis}_md_implausible")
                # semi-direct vn check (see _vn_check): reconstruct sigma from D and verify it lands on
                # the sigma0 anchor, count voxels that break vn, confirm sigma is physiological.
                s0 = SIGMA0.get(tis)
                if s0:
                    degfrac, vn_err, sig_hi, sig_lo = _vn_check(e, s0)
                    m[f"cond_{tag}_{tis}_vn_degfrac"] = round(degfrac, 5)
                    if degfrac > 0.001:
                        f.append(f"cond:{tag}_{tis}_vn_degenerate")    # D cannot be vn-normalized here
                    if np.isfinite(vn_err):
                        m[f"cond_{tag}_{tis}_vn_err"] = round(vn_err, 5)
                        m[f"cond_{tag}_{tis}_sigma_hi"] = round(sig_hi, 3)
                        if vn_err > THR["geom_mean_tol"]:
                            f.append(f"cond:{tag}_{tis}_vn_inconsistent")   # arithmetic / NaN propagation
                        if sig_hi > 2.0 or sig_lo < 1e-3:
                            f.append(f"cond:{tag}_{tis}_sigma_nonphysio")   # runaway / insulating sigma
    return m, f


def qc_sims(P):
    m, f = {}, []
    try:
        from simnibs import mesh_io
    except Exception:
        f.append("sims:no_simnibs"); return m, f
    montage = P.get("montage", "M1")
    for model in MODELS:                       # ISO, DTI, MD-dMRI
        mod = model.replace("-", "")           # metric label: ISO / DTI / MDdMRI
        mp = sim_mesh(P["work"], montage, model, P["id"])
        if not mp:
            f.append(f"sims:{mod}_missing"); continue
        msh = mesh_io.read_msh(mp)
        gm = (msh.elm.tag1 == 2)
        E = msh.field["magnE"].value
        p95 = float(np.percentile(E[gm], 95)); mx = float(np.max(E[gm]))
        m[f"E_{mod}_p95"] = round(p95, 3); m[f"E_{mod}_spike"] = round(mx / max(p95, 1e-9), 1)
        if not (THR["efield_p95_range"][0] <= p95 <= THR["efield_p95_range"][1]):
            f.append(f"sims:{mod}_p95_out_of_range")
        if mx / max(p95, 1e-9) > THR["efield_spike"]:
            f.append(f"sims:{mod}_spike")
    return m, f


def qc_rois(P):
    m, f = {}, []
    n_empty = 0; n_total = 0
    for p in glob.glob(os.path.join(_roi_dir(P["reg"]), "roi_*.nii.gz")):
        n_total += 1
        if (_data(p) > 0).sum() == 0:
            n_empty += 1; f.append(f"rois:empty({os.path.basename(p)})")
    m["roi_n_total"] = n_total; m["roi_n_empty"] = n_empty
    if n_total == 0:
        f.append("rois:none_built")
    return m, f


def qc_tier3(P):
    # Tier-3 midbrain nuclei (CIT168/Pauli) are OVERLAP-ALLOWED separate binary masks
    # (07_build_tier3_nuclei.sh). Glob the per-nucleus masks; do NOT read tier3_labeled.nii.gz
    # (winner-take-all int labels destroy the intended overlap). Pairwise overlap is expected, logged
    # not flagged.
    m, f = {}, []
    t3 = os.path.join(P["reg"], "atlas_rois", "tier3")
    paths = sorted(glob.glob(os.path.join(t3, "roi_*.nii.gz")))
    paths = [p for p in paths if os.path.basename(p) != "tier3_labeled.nii.gz"]
    masks = {}
    for p in paths:
        d = _data(p)
        if d is None:
            continue
        b = d > 0
        masks[os.path.basename(p)] = b
        if b.sum() == 0:
            f.append(f"tier3:empty({os.path.basename(p)})")
    m["tier3_n_total"] = len(masks)
    m["tier3_n_empty"] = sum(1 for b in masks.values() if b.sum() == 0)
    names = sorted(masks)
    n_ov = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if (masks[names[i]] & masks[names[j]]).any():
                n_ov += 1
    m["tier3_n_overlap_pairs"] = n_ov   # expected (overlap-allowed), reported not flagged
    if paths and not masks:
        f.append("tier3:unreadable")
    # presence: each nucleus should have an L and R mask (07 emits roi_{L,R}_{nucleus}.nii.gz)
    if masks:
        for nuc in NUCLEI:
            for side in ("L", "R"):
                if f"roi_{side}_{nuc}.nii.gz" not in masks:
                    f.append(f"tier3:missing(roi_{side}_{nuc})")
    return m, f


STAGES = [("00_charm", qc_charm), ("01_dwi2cond", qc_dwi2cond), ("02_register", qc_register),
          ("03_conductivity", qc_conductivity), ("04_sims", qc_sims), ("05_rois", qc_rois),
          ("06_tier3", qc_tier3)]


def resolve_subjects(args):
    if args.cohort and os.path.exists(args.cohort):
        return json.load(open(args.cohort))
    return [{"id": cfg["SUBJECT"], "work": cfg["WORK_DIR"], "m2m": cfg["M2M_DIR"],
             "reg": cfg["REG_DIR"],
             "samseg": os.path.join(cfg["M2M_DIR"], "segmentation", "labeling.nii.gz"),
             "cc_mask": os.path.join(_roi_dir(cfg["REG_DIR"]), "roi_CC.nii.gz")}]


# Directional metrics -> which tail to guard once a cohort exists ("low" = flag below the lower tail,
# "high" = above the upper). Used by --calibrate to turn single-subject cutoffs into cohort percentiles.
CALIB = {"reg_grad_corr": "low", "reg_cc_v1x": "low", "reg_containment": "low",
         "dwi2cond_cc_fa": "low", "charm_tet_q_med": "low",
         "dwi2cond_csf_fa": "high", "charm_tet_q_badfrac": "high"}


def _calibrate(out):
    p = os.path.join(out, "qc_summary.csv")
    if not os.path.exists(p):
        print(f"no cohort CSV at {p} - run the harness over the cohort first"); return
    rows = list(csv.DictReader(open(p)))
    print(f"Cohort threshold calibration from {len(rows)} subject(s) in {p}")
    if len(rows) < 5:
        print("  WARNING: < 5 subjects - percentile tails are unstable; treat as indicative only.")
    for k, side in CALIB.items():
        vals = np.array([float(r[k]) for r in rows
                         if r.get(k) not in (None, "", "nan")], float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 3:
            continue
        med = float(np.median(vals))
        if side == "low":
            print(f"  {k:22s} flag-below: median={med:.3f}  p10={np.percentile(vals,10):.3f}  "
                  f"p5={np.percentile(vals,5):.3f}   <- set THR near p5")
        else:
            print(f"  {k:22s} flag-above: median={med:.3f}  p90={np.percentile(vals,90):.3f}  "
                  f"p95={np.percentile(vals,95):.3f}   <- set THR near p95")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort"); ap.add_argument("--out", default="analysis/qc")
    ap.add_argument("--montage", default="M1", help="which montage's sims to QC (default M1)")
    ap.add_argument("--calibrate", action="store_true",
                    help="read out/qc_summary.csv and suggest cohort-percentile thresholds, then exit")
    ap.add_argument("--emit-metrics", help="write the per-subject numeric metrics to this JSON and exit")
    args = ap.parse_args()
    if args.calibrate:
        _calibrate(args.out); return
    subjects = resolve_subjects(args)
    os.makedirs(args.out, exist_ok=True)

    rows, stage_flags = [], []
    for P in subjects:
        P.setdefault("reg", os.path.join(P["work"], "registration"))
        P.setdefault("samseg", os.path.join(P["m2m"], "segmentation", "labeling.nii.gz"))
        P.setdefault("cc_mask", os.path.join(_roi_dir(P["reg"]), "roi_CC.nii.gz"))
        P.setdefault("montage", args.montage)
        row = {"subject": P["id"]}; flags_by_stage = {}
        for stage_name, fn in STAGES:
            try:
                metrics, flags = fn(P)
            except Exception as e:
                metrics, flags = {}, [f"{stage_name}:exception({type(e).__name__})"]
            row.update(metrics)
            flags_by_stage[stage_name] = flags
            row[f"{stage_name}_PASS"] = "PASS" if not flags else "FLAG"
        row["overall"] = "PASS" if all(row[f"{s}_PASS"] == "PASS" for s, _ in STAGES) else "FLAG"
        rows.append(row); stage_flags.append(flags_by_stage)

    # cohort MAD outlier flagging, gated at n>=4 (MAD is meaningless below that); a lone subject is
    # judged ONLY by the absolute THR. The `numeric` list is still built for the CSV.
    numeric = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))})
    if len(rows) >= 4:
        for k in numeric:
            vals = np.array([r.get(k, np.nan) for r in rows], float)
            med = np.nanmedian(vals); mad = np.nanmedian(np.abs(vals - med)) * 1.4826
            if mad > 0:
                for i, r in enumerate(rows):
                    if np.isfinite(vals[i]) and abs(vals[i] - med) / mad > 3:
                        stage_flags[i].setdefault("MAD", []).append(f"{k}_outlier")
                        r["overall"] = "FLAG"

    # write CSV
    cols = ["subject"] + numeric + [f"{s}_PASS" for s, _ in STAGES] + ["overall"]
    csv_path = os.path.join(args.out, "qc_summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)

    if args.emit_metrics:
        snap = {r["subject"]: {k: r[k] for k in numeric if k in r} for r in rows}
        with open(args.emit_metrics, "w") as fh:
            json.dump(snap, fh, indent=2, sort_keys=True)
        print(f"metrics -> {args.emit_metrics}")
        return

    # PNGs: flagged subjects + random 10% sample
    rng = np.random.default_rng(0)
    flagged = [i for i, r in enumerate(rows) if r["overall"] == "FLAG"]
    sample = set(flagged) | set(rng.choice(len(rows), max(1, len(rows) // 10), replace=False).tolist())
    png_dir = os.path.join(args.out, "qc_report"); os.makedirs(png_dir, exist_ok=True)
    for i in sorted(sample):
        try:
            _overlay_png(subjects[i], png_dir)
        except Exception as e:
            print(f"  PNG failed for {subjects[i]['id']}: {type(e).__name__}")

    # console summary
    print(f"\nQC summary ({len(rows)} subject(s)) -> {csv_path}")
    for i, r in enumerate(rows):
        if r["overall"] == "FLAG":
            fl = [f for s in stage_flags[i].values() for f in s]
            print(f"  FLAG {r['subject']}: {', '.join(fl)}")
        else:
            print(f"  PASS {r['subject']}")
    print(f"PNGs ({len(sample)}: flagged + 10% sample) -> {png_dir}")


def _overlay_png(P, png_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t1 = _data(os.path.join(P["m2m"], "T1.nii.gz"))
    if t1 is None:
        return
    lab = _data(os.path.join(_roi_dir(P["reg"]), "roi_labels_meshspace.nii.gz"))  # recon-all ROI labels
    # center slices on the ROI-label centroid so small ROIs (CC, mesencephalon) are in-plane
    if lab is not None and (lab > 0).any():
        ci, cj, ck = (int(round(c)) for c in np.array(np.where(lab > 0)).mean(axis=1))
    else:
        ci, cj, ck = (s // 2 for s in t1.shape[:3])
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    planes = [(t1[ci, :, :], None if lab is None else lab[ci, :, :]),
              (t1[:, cj, :], None if lab is None else lab[:, cj, :]),
              (t1[:, :, ck], None if lab is None else lab[:, :, ck])]
    for a, (bg, ov) in zip(ax, planes):
        a.imshow(np.rot90(bg), cmap="gray")
        if ov is not None:
            a.imshow(np.rot90(np.ma.masked_equal(ov, 0)), cmap="tab10", alpha=0.5, vmin=1, vmax=10)
        a.axis("off")
    fig.suptitle(f"{P['id']} - T1 + Tier-1 volume ROIs"); fig.tight_layout()
    fig.savefig(os.path.join(png_dir, f"{P['id']}_overlay.png"), dpi=80); plt.close(fig)


if __name__ == "__main__":
    main()
