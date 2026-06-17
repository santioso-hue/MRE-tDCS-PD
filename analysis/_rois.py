"""_rois.py — Shared ROI loading + sampling for the analysis scripts.

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
    Optional `gate` (bool array on the map grid) further restricts which voxels count —
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
    """Median eigenvalue ratio (lambda1/lambda3) of a 6-component symmetric tensor
    (order xx,xy,xz,yy,yz,zz) within each ROI label."""
    img = nib.load(tensor_path)
    t = np.asarray(img.dataobj, dtype=float)            # (X,Y,Z,6)
    lab = _labels_on_grid(img, labeled, lab_affine)
    order = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    out = {}
    for k, n in names.items():
        sel = (lab == k) & np.isfinite(t[..., 0]) & (np.abs(t[..., 0]) > 1e-6)
        comp = t[sel]
        if comp.shape[0] == 0:
            out[n] = np.nan
            continue
        m = np.zeros((comp.shape[0], 3, 3))
        for a, (p, q) in enumerate(order):
            m[:, p, q] = comp[:, a]
            m[:, q, p] = comp[:, a]
        ev = np.linalg.eigvalsh(m)
        out[n] = float(np.median(ev[:, 2] / np.maximum(ev[:, 0], 1e-9)))
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
