"""_rois.py - Shared ROI loading + sampling for the analysis scripts.

ROIs are int-label masks on the charm/mesh grid, built from the recon-all parcellation by
analysis/build_rois.py:
  registration/freesurfer_rois/roi_labels_meshspace.nii.gz   int labels
  registration/freesurfer_rois/rois.json                     {label: name}
Coverage: cortical and WM lobes, corpus callosum, aseg subcortical (thalamus, caudate, putamen,
pallidum, accumbens, hippocampus, amygdala), brainstem split into Mesencephalon + Pons.
Consumed by 04 (E-field per ROI) and 05 (MRE comparison); each ROI is sampled over every
element/voxel it contains (no fixed-radius spheres).
"""
import os
import json
import numpy as np
import nibabel as nib

TENSOR_ORDER = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]  # FSL dtifit 6-comp -> symmetric 3x3
PD_EPS = 1e-3    # positive-definite floor on the smallest eigenvalue (um2/ms). Numeric anchor: 03's EPS=1e-3
                 # reconstruction floor (qc_harness._vn_check shares the >0 intent, not the value). ASSUMES the
                 # tensor is the um2/ms <D> from 03; do NOT pass an mm2/s dtifit tensor (08 rescales for that).
FILL_EPS = 1e-6  # |Dxx| reject for empty/background voxels (distinct from the PD floor above)


def _pack_6comp(t6, sel):
    """Assemble (N,3,3) symmetric matrices from a 6-component volume (FSL order xx,xy,xz,yy,yz,zz) over the
    selected voxels `sel`. Indexing t6[sel] first avoids a full-volume (X,Y,Z,3,3) allocation. One source of
    truth for the FSL pack, shared by eigh_6comp and eigvals_6comp."""
    comp = t6[sel]
    m = np.zeros((comp.shape[0], 3, 3))
    for a, (p, q) in enumerate(TENSOR_ORDER):
        m[:, p, q] = comp[:, a]; m[:, q, p] = comp[:, a]
    return m


def eigh_6comp(t6, sel):
    """Eigendecomposition of a 6-component symmetric-tensor volume over `sel`. Returns (evals ascending
    (N,3), evecs (N,3,3)); evecs[:, :, 2] is the principal eigenvector V1."""
    return np.linalg.eigh(_pack_6comp(t6, sel))


def eigvals_6comp(t6, sel):
    """Ascending eigenvalues (N,3) only, via eigvalsh (no eigenvectors) -- the values-only fast path."""
    return np.linalg.eigvalsh(_pack_6comp(t6, sel))


def fa_from_evals(ev):
    """Fractional anisotropy from ascending eigenvalues (N,3)."""
    l3, l2, l1 = ev[:, 0], ev[:, 1], ev[:, 2]
    num = np.sqrt(0.5 * ((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2))
    return num / np.maximum(np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2), 1e-12)


def v1_angle_deg(evecs_a, evecs_b):
    """Acute angle (deg) between principal eigenvectors evecs[:, :, 2], per voxel. Sign-agnostic
    (an eigenvector's sign is arbitrary), so it uses |cos|."""
    v1a, v1b = evecs_a[:, :, 2], evecs_b[:, :, 2]
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(v1a * v1b, axis=1)), 0.0, 1.0)))


def load_labeled(reg_dir):
    """Return (labeled int array, affine, {label:name}) in subject/mesh T1 space, from the
    recon-all parcellation (freesurfer_rois) built by analysis/build_rois.py."""
    fr = os.path.join(reg_dir, "freesurfer_rois")
    lab_fn = os.path.join(fr, "roi_labels_meshspace.nii.gz")
    if not os.path.exists(lab_fn):
        raise FileNotFoundError(
            f"{lab_fn} not found. Build the recon-all ROIs first: "
            "simnibs_python analysis/build_rois.py --fs_dir <recon-all subject dir>")
    img = nib.load(lab_fn)
    with open(os.path.join(fr, "rois.json")) as f:
        names = {int(k): v for k, v in json.load(f).items()}
    return np.asarray(img.dataobj).astype(int), img.affine, names


def _labels_on_grid(target_img, labeled, lab_affine):
    """Resample the label volume onto target_img's grid by world-coord nearest neighbour
    (labels are categorical). Fast-path if the grids already match."""
    if target_img.shape[:3] == labeled.shape and np.allclose(target_img.affine, lab_affine, atol=1e-3):
        return labeled
    nx, ny, nz = target_img.shape[:3]
    ii, jj, kk = np.indices((nx, ny, nz))
    grid = np.vstack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)]).astype(np.float64)
    with np.errstate(all="ignore"):
        world = np.asarray(target_img.affine, np.float64) @ grid
        vox = np.linalg.inv(lab_affine) @ world
    vi = np.round(vox[:3]).astype(int)
    ok = ((vi[0] >= 0) & (vi[0] < labeled.shape[0]) &
          (vi[1] >= 0) & (vi[1] < labeled.shape[1]) &
          (vi[2] >= 0) & (vi[2] < labeled.shape[2]))
    out = np.zeros(ii.size, int)
    out[ok] = labeled[vi[0][ok], vi[1][ok], vi[2][ok]]
    return out.reshape((nx, ny, nz))


def sample_volume_medians(map_path, labeled, lab_affine, names, gate=None):
    """Median of a scalar map within each ROI label. Ignores 0 / non-finite voxels.
    Optional `gate` (bool array on the map grid) further restricts which voxels count -
    used to drop low-confidence / CSF-adjacent MRE voxels (05_mre_efield_comparison)."""
    img = nib.load(map_path)
    d = np.asarray(img.dataobj, dtype=float)
    lab = _labels_on_grid(img, labeled, lab_affine)
    valid = np.isfinite(d) & (d != 0)
    if gate is not None and gate.shape == d.shape:
        valid &= gate
    out = {}
    for k, n in names.items():
        v = d[(lab == k) & valid]
        out[n] = float(np.median(v)) if v.size else np.nan
    return out


def sample_tensor_aniso_medians(tensor_path, labeled, lab_affine, names):
    """Median eigenvalue ratio lambda1/lambda3 of a 6-component symmetric tensor (order xx,xy,xz,yy,yz,zz,
    in um2/ms) within each ROI label, over POSITIVE-DEFINITE voxels only (degenerate voxels excluded)."""
    img = nib.load(tensor_path)
    t = np.asarray(img.dataobj, dtype=float)            # (X,Y,Z,6)
    lab = _labels_on_grid(img, labeled, lab_affine)
    out = {}
    for k, n in names.items():
        sel = (lab == k) & np.isfinite(t[..., 0]) & (np.abs(t[..., 0]) > FILL_EPS)
        if not sel.any():
            out[n] = np.nan
            continue
        ev, _ = eigh_6comp(t, sel)
        # PD gate on the raw smallest eigenvalue (PD_EPS): without it a near-zero lambda3 explodes lambda1/lambda3.
        pd = ev[:, 0] > PD_EPS
        out[n] = float(np.median(ev[pd, 2] / ev[pd, 0])) if pd.any() else np.nan
    return out


def sample_tensor_fa_medians(tensor_path, labeled, lab_affine, names):
    """Median FA of a 6-component tensor (um2/ms) within each ROI, over positive-definite voxels (same gate
    as sample_tensor_aniso_medians). Used by 05 for the FA(<D>) vs uFA divergence."""
    img = nib.load(tensor_path)
    t = np.asarray(img.dataobj, dtype=float)
    lab = _labels_on_grid(img, labeled, lab_affine)
    out = {}
    for k, n in names.items():
        sel = (lab == k) & np.isfinite(t[..., 0]) & (np.abs(t[..., 0]) > FILL_EPS)
        if not sel.any():
            out[n] = np.nan
            continue
        ev, _ = eigh_6comp(t, sel)
        pd = ev[:, 0] > PD_EPS
        out[n] = float(np.median(fa_from_evals(ev[pd]))) if pd.any() else np.nan
    return out


def assign_mesh_labels(bary_world, labeled, lab_affine):
    """ROI label per FEM element barycentre (N x 3, subject world mm) -> (N int)."""
    inv = np.linalg.inv(lab_affine)
    b = np.ascontiguousarray(bary_world, dtype=np.float64)
    with np.errstate(all="ignore"):
        vox = inv[:3, :3] @ b.T + inv[:3, 3:4]
    vi = np.round(vox).astype(int)
    ok = ((vi[0] >= 0) & (vi[0] < labeled.shape[0]) &
          (vi[1] >= 0) & (vi[1] < labeled.shape[1]) &
          (vi[2] >= 0) & (vi[2] < labeled.shape[2]))
    out = np.zeros(bary_world.shape[0], int)
    idx = np.where(ok)[0]
    out[idx] = labeled[vi[0][idx], vi[1][idx], vi[2][idx]]
    return out
