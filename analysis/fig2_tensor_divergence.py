"""fig2_tensor_divergence.py - Figure 2: divergence between the single-shell DTI tensor and the QTI <D>
tensor across the Group-1 regions.

Panel A (lead): per-ROI principal-direction (V1) angle between the two tensors, cohort aggregate over the
29 results/<id>/tensor_divergence.csv files (col angle_med); median +/- IQR per ROI. White matter, corpus
callosum, and brainstem are clustered at the top with full-weight markers; the near-isotropic cortex rows
are de-emphasized at the bottom under a shaded band, with an in-panel note.
Panel B: per-voxel V1-angle map on one representative axial slice of one representative subject, computed
from the two tensors (eigh_6comp + v1_angle_deg) over the white-matter mask, over the subject T1.
The per-region FA and eigenvalue-ratio differences are reported in the supplement (supp_tables.py), not here.

Representative subject = the one whose mean white-matter-lobe dE_model
(100*(MD-dMRI_p95 - DTI_p95)/DTI_p95 over the 4 WM lobes, from results/<id>/roi_efield_M1.csv) is nearest
the cohort median. The DTI tensor is mm2/s and <D> is um2/ms, but the V1 angle is scale-invariant, so no
unit conversion is needed for the angle.

Reads:  analysis/results/<id>/tensor_divergence.csv
        analysis/results/<id>/roi_efield_M1.csv
        data/cohort_local/<id>/work/m2m_<id>/DTI_coregT1_tensor.nii.gz
        data/cohort_local/<id>/work/tensor_MD_dMRI.nii.gz
        data/cohort_local/<id>/work/m2m_<id>/T1.nii.gz
        data/cohort_local/<id>/registration/freesurfer_rois/  (via _rois)
Writes: analysis/results/fig2_tensor_divergence.{png,pdf} + _caption.txt
"""
import os
import sys
import csv
import glob
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import (plt, INK, SUB, GRAY, BAND, WM_C, CORTEX_C, BRAINSTEM_C,  # noqa: E402
                       SEQ_CMAP, panel_label)
from _figsave import save_fig  # noqa: E402
from _repsubj import representative_subject  # noqa: E402
from _rois import load_labeled, _labels_on_grid, eigh_6comp, v1_angle_deg, PD_EPS, FILL_EPS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "analysis", "results")
COHORT = os.path.join(ROOT, "data", "cohort_local")

GROUP1 = ["Ctx_Frontal", "Ctx_Parietal", "Ctx_Temporal", "Ctx_Occipital",
          "WM_Frontal", "WM_Parietal", "WM_Temporal", "WM_Occipital", "CC", "Mesencephalon", "Pons"]
WM_ROIS = {"WM_Frontal", "WM_Parietal", "WM_Temporal", "WM_Occipital"}
CTX_ROIS = {"Ctx_Frontal", "Ctx_Parietal", "Ctx_Temporal", "Ctx_Occipital"}
WM_LABEL_IDS = (11, 12, 13, 14)
DTI_TO_UM2MS = 1000.0   # mm2/s -> um2/ms, for the PD gate only; angle is scale-invariant
PRETTY = {"Ctx_Frontal": "Cortex frontal", "Ctx_Parietal": "Cortex parietal",
          "Ctx_Temporal": "Cortex temporal", "Ctx_Occipital": "Cortex occipital",
          "WM_Frontal": "WM frontal", "WM_Parietal": "WM parietal",
          "WM_Temporal": "WM temporal", "WM_Occipital": "WM occipital",
          "CC": "Corpus callosum", "Mesencephalon": "Mesencephalon", "Pons": "Pons"}


def cohort_perroi():
    """Per-ROI arrays of V1 angle_med across all subjects with a tensor_divergence.csv."""
    ang = {r: [] for r in GROUP1}
    n = 0
    for p in sorted(glob.glob(os.path.join(R, "PD*", "tensor_divergence.csv"))):
        n += 1
        for row in csv.DictReader(open(p)):
            roi = row["ROI"]
            if roi in ang:
                ang[roi].append(float(row["angle_med"]))
    return ang, n


def voxel_angle_slice(sid):
    """Per-voxel V1 angle (deg) over the WM mask on the best axial slice + the T1 background for that slice.
    Returns (t1_slice, angle_slice_masked, z, wm_median_angle, aspect). aspect is the imshow aspect for the
    displayed (transposed) axial slice, derived from the real voxel spacing so the slice is anatomically
    proportioned on anisotropic voxels."""
    base = os.path.join(COHORT, sid)
    dti_p = os.path.join(base, "work", "m2m_" + sid, "DTI_coregT1_tensor.nii.gz")
    dd_p = os.path.join(base, "work", "tensor_MD_dMRI.nii.gz")
    t1_p = os.path.join(base, "work", "m2m_" + sid, "T1.nii.gz")
    reg = os.path.join(base, "registration")

    dti_img = nib.load(dti_p)
    tdti = np.asarray(dti_img.dataobj, float) * DTI_TO_UM2MS
    tdd = np.asarray(nib.load(dd_p).dataobj, float)
    t1 = np.asarray(nib.load(t1_p).dataobj, float)
    labeled, lab_aff, _ = load_labeled(reg)
    lab = _labels_on_grid(dti_img, labeled, lab_aff)
    wm = np.isin(lab, WM_LABEL_IDS)

    sel = (wm & np.isfinite(tdti[..., 0]) & np.isfinite(tdd[..., 0])
           & (np.abs(tdti[..., 0]) > FILL_EPS) & (np.abs(tdd[..., 0]) > FILL_EPS))
    ev_d, vec_d = eigh_6comp(tdd, sel)
    ev_t, vec_t = eigh_6comp(tdti, sel)
    pd = (ev_d[:, 0] > PD_EPS) & (ev_t[:, 0] > PD_EPS)
    ang = v1_angle_deg(vec_d[pd], vec_t[pd])
    wm_median_angle = float(np.median(ang))

    # full-volume angle map (NaN outside the gated WM)
    amap = np.full(tdti.shape[:3], np.nan)
    idx = np.argwhere(sel)[pd]
    amap[idx[:, 0], idx[:, 1], idx[:, 2]] = ang
    # pick the axial slice with the most gated WM voxels
    per_z = np.array([np.isfinite(amap[:, :, z]).sum() for z in range(amap.shape[2])])
    z = int(per_z.argmax())
    # imshow aspect for the displayed (transposed) slice: rows are voxel axis 1, columns are voxel axis 0,
    # so aspect = (row spacing) / (column spacing) = zooms[1] / zooms[0] makes 1 mm read as 1 mm both ways.
    zx, zy = float(dti_img.header.get_zooms()[0]), float(dti_img.header.get_zooms()[1])
    aspect = zy / zx
    return t1[:, :, z].T, amap[:, :, z].T, z, wm_median_angle, aspect


def main():
    ang, n = cohort_perroi()
    rep, _, coh_med = representative_subject(R, "M1")
    print(f"cohort n (tensor_divergence files) = {n}; representative subject = {rep}")

    # Order so the load-bearing rows (white matter, callosum, brainstem) cluster at the TOP of the panel
    # and the near-isotropic cortex rows sit at the BOTTOM. y increases upward, so the load-bearing group
    # takes the high y indices. Each group is internally sorted by cohort-median angle.
    med_ang = {r: float(np.median(ang[r])) for r in GROUP1}
    load_bearing = [r for r in GROUP1 if r not in CTX_ROIS]
    cortex = [r for r in GROUP1 if r in CTX_ROIS]
    order = (sorted(cortex, key=lambda r: med_ang[r])
             + sorted(load_bearing, key=lambda r: med_ang[r]))
    n_ctx = len(cortex)   # cortex occupies the lowest n_ctx y indices

    fig = plt.figure(figsize=(7.2, 3.0), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0])
    axA = fig.add_subplot(gs[0, 0])   # lead panel
    axB = fig.add_subplot(gs[0, 1])
    axB.set_anchor("N")               # top-align Panel B with Panel A

    # ---- Panel A: per-ROI angle lollipop ----
    # Shade the cortex band (the lowest n_ctx rows) so the eye lands on the load-bearing rows above it.
    if n_ctx:
        axA.axhspan(-0.6, n_ctx - 0.5, color=BAND, alpha=0.7, zorder=0)
    ypos = np.arange(len(order))
    for i, roi in enumerate(order):
        a = np.array(ang[roi], float)
        m = float(np.median(a))
        q1, q3 = float(np.percentile(a, 25)), float(np.percentile(a, 75))
        # Tissue colors: WM blue, cortex gray, corpus callosum + brainstem green (never the model vermillion).
        c = WM_C if roi in WM_ROIS else (CORTEX_C if roi in CTX_ROIS else BRAINSTEM_C)
        is_ctx = roi in CTX_ROIS
        # cortex rows are drawn lighter and smaller; load-bearing rows get full-weight markers
        lw = 1.2 if is_ctx else 2.0
        bar_alpha = 0.35 if is_ctx else 0.65
        sz = 26 if is_ctx else 56
        mk_alpha = 0.55 if is_ctx else 1.0
        axA.plot([q1, q3], [i, i], color=c, lw=lw, alpha=bar_alpha, zorder=2, solid_capstyle="round")
        axA.scatter([m], [i], color=c, s=sz, alpha=mk_alpha, linewidths=0, zorder=3)
    axA.set_yticks(ypos)
    lbls = axA.set_yticklabels([PRETTY[r] for r in order])
    for i, lab in enumerate(lbls):
        if order[i] in CTX_ROIS:
            lab.set_color(SUB)
    # Hold xlim so every ROI marker (cortex medians reach the high 40s) is inside the data area.
    axA.set_xlim(0, 55)
    axA.set_ylim(-0.6, len(order) - 0.4)
    axA.set_xlabel(r"DTI vs $\langle D\rangle$ principal-direction angle ($\degree$)")
    axA.tick_params(axis="y", length=0)
    axA.grid(axis="x", color="#e7e9ec", lw=0.7, zorder=0)
    panel_label(axA, "A")
    # Factual note (cortex V1 is near-isotropic, principal direction weakly defined): placed left of the
    # cortex markers, inside the shaded band, where no data sits.
    if n_ctx:
        axA.text(2.0, (n_ctx - 1) / 2.0, "cortex V1 near-isotropic:\nprincipal direction weakly defined",
                 ha="left", va="center", fontsize=6.5, color=SUB, zorder=4)
    # Frameless legend in the empty upper-right corner of the data area (well above the cortex band and
    # clear of every marker).
    h_wm = plt.Line2D([], [], color=WM_C, marker="o", lw=0, markersize=6)
    h_ct = plt.Line2D([], [], color=CORTEX_C, marker="o", lw=0, markersize=5, alpha=0.55)
    h_ot = plt.Line2D([], [], color=BRAINSTEM_C, marker="o", lw=0, markersize=6)
    axA.legend([h_wm, h_ot, h_ct],
               ["White matter", "Callosum / brainstem", "Cortex (near-isotropic)"],
               frameon=False, loc="upper right", handletextpad=0.3, borderaxespad=0.4)

    # ---- Panel B: per-voxel angle map on a representative slice ----
    t1s, angs, z, wm_med, aspect = voxel_angle_slice(rep)
    print(f"Panel B: subject {rep}, slice z={z}, per-voxel WM median angle = {wm_med:.2f} deg, "
          f"aspect = {aspect:.4f}")
    # Crop to the head bounding box (drop the black background margins) so the slice fills the panel.
    fg = np.isfinite(t1s) & (t1s > t1s.max() * 0.06)
    rows = np.where(fg.any(axis=1))[0]
    cols = np.where(fg.any(axis=0))[0]
    if rows.size and cols.size:
        pad = 3
        r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + 1 + pad, t1s.shape[0])
        c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + 1 + pad, t1s.shape[1])
        t1s, angs = t1s[r0:r1, c0:c1], angs[r0:r1, c0:c1]
    extent = (0, t1s.shape[1], 0, t1s.shape[0])
    axB.imshow(t1s, cmap="gray", origin="lower", interpolation="nearest", aspect=aspect, extent=extent)
    im = axB.imshow(angs, cmap=SEQ_CMAP, origin="lower", interpolation="nearest", vmin=0, vmax=90,
                    aspect=aspect, extent=extent)
    axB.set_xticks([]); axB.set_yticks([])
    axB.set_xlim(extent[0], extent[1]); axB.set_ylim(extent[2], extent[3])
    for s in axB.spines.values():
        s.set_visible(False)
    panel_label(axB, "B")
    # Left/Right orientation labels (radiological axial: image left is anatomical right).
    axB.text(0.015, 0.5, "R", transform=axB.transAxes, ha="left", va="center",
             fontsize=8, fontweight="bold", color="white")
    axB.text(0.985, 0.5, "L", transform=axB.transAxes, ha="right", va="center",
             fontsize=8, fontweight="bold", color="white")
    # Colorbar matched to the rendered image height (the slice fills the panel width, leaving the box
    # taller than the image), brought in tight against the slice. Draw once to learn the image's height
    # fraction within the axes, then place the colorbar as an inset of exactly that height.
    fig.canvas.draw()
    disp_ratio = aspect * t1s.shape[0] / t1s.shape[1]       # displayed image height / width
    box = axB.get_position()
    img_h = min(1.0, disp_ratio * box.width / box.height)   # image height as a fraction of the axes box
    cax = axB.inset_axes([1.04, 1.0 - img_h, 0.05, img_h])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"DTI vs $\langle D\rangle$ V1 angle ($\degree$)")
    cb.set_ticks([0, 30, 45, 60, 90])
    cb.ax.tick_params(length=2)

    caption = (
        "Figure 2. Divergence between the single-shell DTI tensor and the QTI mean tensor <D> across the "
        "Group 1 regions. Group 1 = the four cortical lobes, the four white-matter lobes, the corpus "
        "callosum, the mesencephalon, and the pons. V1 is the principal eigenvector of a tensor. The "
        "DTI-vs-<D> V1 angle is the acute angle between the two principal eigenvectors, sign-agnostic. "
        "(A) Per-region V1 angle across the cohort (n = 29): each point is the cohort median of that "
        "region's median V1 angle, with the bar spanning the interquartile range over subjects; blue marks "
        "white-matter lobes, green marks the corpus callosum and brainstem, gray marks cortical lobes. The "
        "white-matter lobes, corpus callosum, and brainstem are clustered at the top with full-weight "
        "markers; the cortical lobes are placed at the bottom under a shaded band and drawn lighter, because "
        "cortex is near-isotropic and its principal direction is therefore weakly defined. Within each group "
        "regions are ordered by median angle. (B) Per-voxel V1 angle on one representative axial slice of "
        "the representative subject (the subject whose mean white-matter-lobe field difference is nearest "
        "the cohort median), over the white-matter mask, on the subject T1, displayed at the true voxel "
        "aspect ratio; color is the V1 angle in degrees. Units: degrees."
    )
    save_fig(fig, "fig2_tensor_divergence", caption)
    plt.close(fig)


if __name__ == "__main__":
    main()
