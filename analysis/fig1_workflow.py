"""Figure 1: real-data conductivity-modeling pipeline for one representative subject.

An austere data-flow strip built from this subject's actual images: T1/T2 structural inputs,
the SimNIBS charm tissue segmentation, the single-shell DTI fractional-anisotropy map and the
multi-shell QTI mean-tensor fractional-anisotropy map, and the three modeled electric-field
magnitude maps (ISO, DTI, MD-dMRI) on one shared scale. No banners, no colored panel frames,
no rounded boxes: thumbnails abut cleanly, connected by thin neutral single-weight arrows, with
small neutral labels beneath each and one shared magma colorbar. Every thumbnail is the same
axial slice, cropped to the head, so all panels register. Writes fig1_workflow.{png,pdf} +
caption to analysis/results/.
"""
import os
import sys
import glob
import numpy as np
import nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from _figstyle import plt, INK, SUB, SEQ_CMAP  # noqa: E402
from _figsave import save_fig  # noqa: E402
from _rois import eigh_6comp, fa_from_evals  # noqa: E402
from _repsubj import representative_subject  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402

RESULTS = os.path.join(ROOT, "analysis", "results")
COHORT = os.path.join(ROOT, "data", "cohort_local")
# One axial slice index, shared by every thumbnail (mid-axial: cortex ribbon, ventricle bodies, and
# major WM tracts so the conductivity direction-encoded-color panels read clearly).
KSLICE = 270

# SimNIBS default isotropic conductivities (S/m) for the ISO scalar-conductivity panel.
SIGMA0 = {1: 0.126, 2: 0.275, 3: 1.654}   # WM, GM, CSF


def resolve_subject():
    """Representative subject + its on-disk paths, picked by the shared cohort rule (matches Figures 2/3:
    nearest the cohort-median white-matter-lobe conductivity-model effect). Falls back to the single
    cohort_local work tree when the per-subject ROI CSVs needed by the rule are absent."""
    try:
        sid, _, _ = representative_subject(RESULTS, "M1")
    except FileNotFoundError:
        works = sorted(glob.glob(os.path.join(COHORT, "*", "work")))
        if not works:
            raise FileNotFoundError(f"no cohort work tree under {COHORT}")
        sid = os.path.basename(os.path.dirname(works[0]))
    coh = os.path.join(COHORT, sid, "work")
    return sid, coh, os.path.join(coh, f"m2m_{sid}"), os.path.join(RESULTS, sid)


SUBJ, COH, M2M, RES = resolve_subject()

# Tissue palette for the charm segmentation thumbnail (discrete, from final_tissues_LUT).
TISSUE = [
    (1, "WM", "#dcdcdc"),
    (2, "GM", "#9a9a9a"),
    (3, "CSF", "#4f86ff"),
    (5, "Scalp", "#ff9e7a"),
    (7, "Bone", "#e7c873"),
]

# Uniform imaging background for the brain-only thumbnails (FA, |E|): the magma colormap's dark end, so
# the masked-out rim reads as one continuous dark field that fills the frame (no white corners).
IMG_BG = "#0a0612"

# One arrow style for the whole figure (single head, single weight, neutral ink).
ARROW_C = SUB
ARROW_LW = 1.3
ARROW_MS = 11


def _axial(arr):
    """Display orientation for one axial slice: rot90 so nose is up and rows span A/P, cols span R/L."""
    return np.rot90(arr)


def _brain_bbox(brain2d, seg2d, pad_rl=14, pad_ap=12):
    """Bounding box centered on the brain (not the full face/neck head) with an even pad that keeps a
    rim of skull/scalp for context, clipped to the head mask so no black margin is introduced."""
    br = np.where(brain2d.any(axis=1))[0]
    bc = np.where(brain2d.any(axis=0))[0]
    head = seg2d > 0
    hr = np.where(head.any(axis=1))[0]
    hc = np.where(head.any(axis=0))[0]
    r0 = max(br.min() - pad_rl, hr.min())
    r1 = min(br.max() + pad_rl + 1, hr.max() + 1)
    c0 = max(bc.min() - pad_ap, hc.min())
    c1 = min(bc.max() + pad_ap + 1, hc.max() + 1)
    return r0, r1, c0, c1


def _fa_slice(tensor_path, k, brain_k):
    """Fractional anisotropy on one axial slice of a 6-component tensor, over the brain mask only.
    Tensor scale does not affect FA, so DTI (mm2/s) and QTI (um2/ms) are handled identically."""
    t = np.asarray(nib.load(tensor_path).dataobj, dtype=float)[:, :, k, :]
    fa = np.zeros(t.shape[:2])
    sel = brain_k & np.isfinite(t[..., 0]) & (np.abs(t[..., 0]) > 1e-12)
    if sel.any():
        ev, _ = eigh_6comp(t, sel)
        fa[sel] = np.clip(fa_from_evals(ev), 0.0, 1.0)
    return np.ma.masked_where(~sel, fa)


def _dec_slice(tensor_path, k, brain_k):
    """Direction-encoded color (RGB) of the conductivity principal axis on one axial slice. Volume
    normalization (sigma ∝ D) scales eigenvalues only, so the conductivity tensor shares the diffusion
    tensor's eigenvectors; the principal conductivity direction is the tensor V1. RGB = |V1| in voxel
    axes (R = x ≈ L-R, G = y ≈ A-P, B = z ≈ S-I), brightness weighted by FA so near-isotropic gray
    matter and CSF stay dark and only oriented white matter carries colour. Returns an (X, Y, 3) array."""
    t = np.asarray(nib.load(tensor_path).dataobj, dtype=float)[:, :, k, :]
    rgb = np.zeros(t.shape[:2] + (3,))
    sel = brain_k & np.isfinite(t[..., 0]) & (np.abs(t[..., 0]) > 1e-12)
    if sel.any():
        ev, evec = eigh_6comp(t, sel)
        v1 = np.abs(evec[:, :, 2])
        fa = np.clip(fa_from_evals(ev), 0.0, 1.0)
        rgb[sel] = v1 * fa[:, None]
    return rgb / max(rgb.max(), 1e-6)


def _iso_sigma_slice(seg2d, brain_k):
    """ISO scalar conductivity (S/m) per voxel from the tissue label, masked to the brain. No direction:
    a single value per tissue (WM < GM < CSF), shown in grayscale to contrast with the colored tensor
    panels. Returns a masked array."""
    sig = np.zeros(seg2d.shape, float)
    for lab, s in SIGMA0.items():
        sig[seg2d == lab] = s
    return np.ma.masked_where(~brain_k, sig)


def load_panels():
    """Load every real thumbnail on the shared axial slice, cropped to the head bounding box."""
    t1img = nib.load(os.path.join(M2M, "T1.nii.gz"))
    T1 = np.asarray(t1img.dataobj, float)[:, :, KSLICE]
    T2 = np.asarray(nib.load(os.path.join(M2M, "T2_reg.nii.gz")).dataobj, float)[:, :, KSLICE]
    seg = np.asarray(nib.load(os.path.join(M2M, "final_tissues.nii.gz")).dataobj)
    if seg.ndim == 4:
        seg = seg[..., 0]
    seg = seg.astype(int)
    seg2d = seg[:, :, KSLICE]
    brain_k = np.isin(seg2d, [1, 2, 3])

    r0, r1, c0, c1 = _brain_bbox(brain_k, seg2d)

    def crop(a):
        return a[r0:r1, c0:c1]

    dti_tensor = os.path.join(M2M, "DTI_coregT1_tensor.nii.gz")
    qti_tensor = os.path.join(COH, "tensor_MD_dMRI.nii.gz")
    fa_dti = _fa_slice(dti_tensor, KSLICE, brain_k)
    fa_qti = _fa_slice(qti_tensor, KSLICE, brain_k)

    out = {
        "T1": _axial(crop(T1)),
        "T2": _axial(crop(T2)),
        "seg": _axial(crop(seg2d)),
        "fa_dti": _axial(crop(fa_dti)),
        "fa_qti": _axial(crop(fa_qti)),
        "brain": _axial(crop(brain_k)),
        "sig_ISO": _axial(crop(_iso_sigma_slice(seg2d, brain_k))),
        "sig_DTI": _axial(crop(_dec_slice(dti_tensor, KSLICE, brain_k))),
        "sig_MD_dMRI": _axial(crop(_dec_slice(qti_tensor, KSLICE, brain_k))),
    }
    # physical displayed width/height (R/L mm over A/P mm): after rot90 the displayed rows span A/P
    # (zy) and cols span R/L (zx). Each thumbnail axes box is sized to this so it hugs the content.
    zx, zy = float(t1img.header.get_zooms()[0]), float(t1img.header.get_zooms()[1])
    disp = np.asarray(out["seg"])
    out["wh"] = (disp.shape[1] * zx) / (disp.shape[0] * zy)
    return out


# --------------------------------------------------------------------------------------------------
# drawing primitives
# --------------------------------------------------------------------------------------------------

def thumb(fig, rect, img, brain=None, cmap="gray", vmin=None, vmax=None, aspect="auto",
          norm=None, bg=None):
    """Place one thumbnail as an inset axes at figure-fraction rect [x, y, w, h]; the rect is already
    proportioned to the real image so aspect='auto' fills it with no internal black margin. No frame:
    the thumbnail abuts the strip cleanly. Optional brain mask + bg fill so a brain-only map fills its
    box on a uniform imaging background instead of leaving white corners; returns (ax, image handle)."""
    ax = fig.add_axes(rect)
    if bg is not None:
        ax.set_facecolor(bg)
    show = img
    if brain is not None:
        show = np.ma.masked_where(~brain, show)
    if norm is not None:
        im = ax.imshow(show, cmap=cmap, norm=norm, aspect=aspect, interpolation="nearest")
    else:
        im = ax.imshow(show, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ax, im


def arrow(fig, p0, p1, color=ARROW_C, lw=ARROW_LW, ms=ARROW_MS, rad=0.0):
    """One standardized arrow in figure-fraction coords."""
    cs = f"arc3,rad={rad}"
    fig.add_artist(FancyArrowPatch(
        p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, connectionstyle=cs,
        joinstyle="miter", capstyle="round", zorder=20))


def main():
    P = load_panels()
    brain = np.asarray(P["brain"])

    # discrete tissue colormap for the segmentation thumbnail
    labels = [t[0] for t in TISSUE]
    colors = [t[2] for t in TISSUE]
    seg_arr = np.asarray(P["seg"])
    seg_disp = np.ma.masked_all(seg_arr.shape)
    for i, lab in enumerate(labels):
        seg_disp[seg_arr == lab] = i
    seg_cmap = ListedColormap(colors)
    seg_cmap.set_bad((1, 1, 1, 0))
    seg_norm = BoundaryNorm(np.arange(-0.5, len(labels) + 0.5), len(labels))

    # ---- canvas + grid -----------------------------------------------------------------------
    FW, FH = 11.0, 5.0
    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor("white")

    # neutral label tokens for the spare strip: one small sans size beneath each thumbnail, one
    # slightly smaller for the secondary (FA) qualifier. No banners, no bold section headers.
    LAB = 7.2          # primary thumbnail label
    LAB2 = 6.6         # secondary qualifier
    LAB_DY = 0.026     # gap from thumbnail edge to its label

    # thumbnail height (figure fraction); width derived from the real image W/H so the box hugs
    # the content with no internal black margin (FW/FH converts the fraction-space aspect)
    TH = 0.250
    TW = P["wh"] * TH * FH / FW

    # column x-anchors (left edges of each thumbnail group)
    X_IN = 0.050
    X_HEAD = 0.215
    X_ARM = 0.405
    X_MAP = 0.580
    X_OUT = 0.775

    # row centers
    Y_TOP = 0.660
    Y_MID = 0.460
    Y_BOT = 0.260

    def rect(x, yc):
        return [x, yc - TH / 2.0, TW, TH]

    # ---- 1. structural inputs (T1 over T2) ---------------------------------------------------
    th = TH * 0.92
    tw_s = P["wh"] * th * FH / FW
    thumb(fig, [X_IN, Y_MID + 0.012, tw_s, th], P["T1"], cmap="gray")
    fig.text(X_IN + tw_s + 0.006, Y_MID + 0.012 + th / 2.0, "T1", ha="left", va="center",
             fontsize=LAB, color=SUB)
    thumb(fig, [X_IN, Y_MID - th - 0.012, tw_s, th], P["T2"], cmap="gray")
    fig.text(X_IN + tw_s + 0.006, Y_MID - 0.012 - th / 2.0, "T2", ha="left", va="center",
             fontsize=LAB, color=SUB)

    arrow(fig, (X_IN + tw_s + 0.034, Y_MID), (X_HEAD - 0.012, Y_MID))

    # ---- 2. head model (charm tissue segmentation) -------------------------------------------
    thumb(fig, rect(X_HEAD, Y_MID), seg_disp, cmap=seg_cmap, norm=seg_norm)
    fig.text(X_HEAD + TW / 2.0, Y_MID + TH / 2.0 + LAB_DY, "Head model", ha="center", va="bottom",
             fontsize=LAB, color=SUB)
    # tiny tissue key beneath the thumbnail (the only categorical color the figure carries)
    key_y = Y_MID - TH / 2.0 - 0.032
    kx = X_HEAD - 0.004
    for (lab, name, col) in TISSUE:
        fig.patches.append(plt.Rectangle((kx, key_y), 0.010, 0.016, transform=fig.transFigure,
                                         facecolor=col, edgecolor="none", zorder=15))
        fig.text(kx + 0.012, key_y + 0.008, name, ha="left", va="center", fontsize=6.0, color=SUB)
        kx += 0.012 + 0.010 + 0.0072 * len(name)

    # branch point: head -> two diffusion arms
    arrow(fig, (X_HEAD + TW + 0.006, Y_MID), (X_ARM - 0.012, Y_TOP), rad=-0.20)
    arrow(fig, (X_HEAD + TW + 0.006, Y_MID), (X_ARM - 0.012, Y_BOT), rad=0.20)

    # ---- 3. two diffusion arms (no frames; small neutral labels) -----------------------------
    thumb(fig, rect(X_ARM, Y_TOP), P["fa_dti"], brain=brain, cmap=SEQ_CMAP, vmin=0.0, vmax=0.8,
          bg=IMG_BG)
    fig.text(X_ARM + TW / 2.0, Y_TOP + TH / 2.0 + LAB_DY, "Single-shell DTI",
             ha="center", va="bottom", fontsize=LAB, color=SUB)
    fig.text(X_ARM + TW / 2.0, Y_TOP - TH / 2.0 - LAB_DY, "FA",
             ha="center", va="top", fontsize=LAB2, color=SUB)
    thumb(fig, rect(X_ARM, Y_BOT), P["fa_qti"], brain=brain, cmap=SEQ_CMAP, vmin=0.0, vmax=0.8,
          bg=IMG_BG)
    fig.text(X_ARM + TW / 2.0, Y_BOT + TH / 2.0 + LAB_DY, r"Multi-shell QTI  $\langle D \rangle$",
             ha="center", va="bottom", fontsize=LAB, color=SUB)
    fig.text(X_ARM + TW / 2.0, Y_BOT - TH / 2.0 - LAB_DY, "FA",
             ha="center", va="top", fontsize=LAB2, color=SUB)

    # both arms rejoin at the shared vn mapping (arrow tips just left of the text block)
    x_tag = X_MAP + TW / 2.0
    arrow(fig, (X_ARM + TW + 0.006, Y_TOP), (x_tag - 0.052, Y_MID + 0.020), rad=0.18)
    arrow(fig, (X_ARM + TW + 0.006, Y_BOT), (x_tag - 0.052, Y_MID - 0.020), rad=-0.18)

    # ---- 4. shared vn mapping (tensor -> conductivity). FEM is downstream; |E| is Figure 3. ----
    fig.text(x_tag, Y_MID + 0.013, "vn mapping", ha="center", va="center", fontsize=LAB, color=INK)
    fig.text(x_tag, Y_MID - 0.013, r"$\sigma \propto D$", ha="center", va="center",
             fontsize=LAB2, color=SUB)

    # vn mapping -> the two tensor-conductivity panels (DTI top, MD-dMRI bottom)
    arrow(fig, (x_tag + 0.046, Y_MID + 0.012), (X_OUT - 0.012, Y_TOP), rad=-0.18)
    arrow(fig, (x_tag + 0.046, Y_MID - 0.012), (X_OUT - 0.012, Y_BOT), rad=0.18)

    # ISO scalar-conductivity path: along the floor, bypassing the diffusion arms, up into the ISO panel
    iso_y = 0.080
    pts = [(X_HEAD + TW / 2.0, key_y - 0.024), (X_HEAD + TW / 2.0, iso_y),
           (X_OUT - 0.034, iso_y), (X_OUT - 0.034, Y_MID), (X_OUT - 0.012, Y_MID)]
    for i in range(len(pts) - 1):
        if i == len(pts) - 2:
            arrow(fig, pts[i], pts[i + 1], ms=9)
        else:
            fig.add_artist(FancyArrowPatch(pts[i], pts[i + 1], transform=fig.transFigure,
                                           arrowstyle="-", linewidth=ARROW_LW, color=ARROW_C,
                                           shrinkA=0, shrinkB=0, zorder=19))
    fig.text((X_HEAD + TW / 2.0 + X_OUT) / 2.0, iso_y - 0.020,
             "scalar conductivity (ISO)", ha="center", va="top", fontsize=LAB2, color=SUB)

    # ---- 5. three conductivity models (the FEM input). DTI / MD-dMRI: direction-encoded color of
    #         the principal conductivity axis; ISO: the scalar value in grayscale. ----------------
    fig.text(X_OUT + TW / 2.0, Y_TOP + TH / 2.0 + LAB_DY, "Conductivity",
             ha="center", va="bottom", fontsize=LAB, color=INK)
    out_specs = [(Y_TOP, "sig_DTI", "DTI", True), (Y_MID, "sig_ISO", "ISO", False),
                 (Y_BOT, "sig_MD_dMRI", "MD-dMRI", True)]
    for (yc, key, label, is_dec) in out_specs:
        if is_dec:
            thumb(fig, rect(X_OUT, yc), np.asarray(P[key]), bg=IMG_BG)
        else:
            thumb(fig, rect(X_OUT, yc), np.asarray(P[key]), brain=brain, cmap="gray",
                  vmin=0.0, vmax=0.32, bg=IMG_BG)
        fig.text(X_OUT + TW + 0.008, yc, label, ha="left", va="center", fontsize=LAB, color=SUB)

    # direction key for the DEC panels (replaces the |E| colorbar) + the downstream FEM/Figure 3 note
    ky = Y_BOT - TH / 2.0 - 0.050
    kx = X_OUT
    for col, lab in [("#cc2222", "L-R"), ("#1f9e3a", "A-P"), ("#2a5bd7", "S-I")]:
        fig.patches.append(plt.Rectangle((kx, ky), 0.011, 0.016, transform=fig.transFigure,
                                         facecolor=col, edgecolor="none", zorder=15))
        fig.text(kx + 0.013, ky + 0.008, lab, ha="left", va="center", fontsize=6.0, color=SUB)
        kx += 0.013 + 0.011 + 0.0098 * len(lab)
    fig.text(X_OUT, ky - 0.026, "principal conductivity axis (ISO: scalar, gray)",
             ha="left", va="top", fontsize=6.0, color=SUB)
    fig.text(X_OUT, ky - 0.050, r"$\rightarrow$ FEM solver (SimNIBS) $\rightarrow$ $|E|$ : Figure 3",
             ha="left", va="top", fontsize=LAB2, color=INK)

    caption = (
        "Figure 1. Subject-specific conductivity-modeling pipeline. Three tDCS simulations are built "
        "from the same head model that differ only in the conductivity input, which is what the figure "
        "follows. T1 and T2 structural images feed the SimNIBS charm head segmentation (white matter, "
        "gray matter, cerebrospinal fluid, scalp, bone). The pipeline then branches into two diffusion "
        "arms, a single-shell DTI tensor and the multi-shell QTI mean diffusion tensor (angle bracket "
        "D), shown as fractional-anisotropy maps on one axial slice. Both tensors pass through the same "
        "volume-normalized mapping (sigma proportional to D) to give the DTI and MD-dMRI conductivity "
        "tensors, shown as direction-encoded color of the principal conductivity axis (red left-right, "
        "green anterior-posterior, blue superior-inferior; brightness scaled by fractional anisotropy). "
        "The ISO model bypasses diffusion and uses scalar literature conductivities, shown in grayscale. "
        "All three conductivities enter the same SimNIBS finite-element solver; the resulting "
        "electric-field magnitude is the subject of Figure 3. Same axial slice and subject throughout."
    )
    save_fig(fig, "fig1_workflow", caption)
    plt.close(fig)
    print(f"  slice k={KSLICE}; conductivity panels (DTI/MD-dMRI DEC, ISO scalar grayscale)")


if __name__ == "__main__":
    main()
