"""fig3_field_maps.py - Figure 3: electrode montage placement plus the tDCS |E| field on the cortical
surface for the three conductivity models (ISO, DTI, MD-dMRI) and the MD-dMRI minus DTI difference, for one
representative subject, M1 montage.

Panel A: the scalp with the electrodes (anode C3 red, cathode Fp2 blue) on the watertight (neck-capped) head
(analysis/_montagerender.py). Panel B: |E| solved on the FEM volume mesh and DISPLAYED on the smooth middle-
gray-matter central surface (FreeSurfer lh/rh.central.gii) via gmsh - watertight, no holes - the way SimNIBS-
team figures are made (Mosayebi-Samani 2025). Each model and the difference are shown at two oblique views.
Colormap convention follows the tDCS/SimNIBS literature: a shared perceptual sequential map (parula) for |E|
across the three models on one fixed scale (pooled p95), and a symmetric diverging map (blue-white-red) for the
difference. Each panel is rendered in its own process (repeated gmsh init in one process segfaults).

Representative subject: nearest the cohort median white-matter dE_model (analysis/_repsubj.py).

Reads:  data/cohort_local/<id>/work/sim_M1_<model>/<id>_TDCS_1_{scalar,vn}.msh
        data/cohort_local/<id>/work/m2m_<id>/{T1.nii.gz, surfaces/lh.central.gii, rh.central.gii}
Writes: analysis/results/<id>/magnE_<model>_T1.nii.gz, head_montage.msh  (cached)
        analysis/results/fig3_field_maps.{png,pdf} + _caption.txt
"""
import os
import sys
import subprocess
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import plt, INK, SUB  # noqa: E402
from _figsave import save_fig  # noqa: E402
from _repsubj import representative_subject  # noqa: E402
from _surfrender import load_central_surface, sample_volume_to_surface  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "analysis", "results")
COHORT = os.path.join(ROOT, "data", "cohort_local")
SURF_WORKER = os.path.join(ROOT, "analysis", "_surfrender.py")
MON_WORKER = os.path.join(ROOT, "analysis", "_montagerender.py")
MODELS = ["ISO", "DTI", "MD_dMRI"]
SUFFIX = {"ISO": "scalar", "DTI": "vn", "MD_dMRI": "vn"}
TITLES = {"ISO": "ISO", "DTI": "DTI", "MD_dMRI": "MD-dMRI", "DIFF": "MD-dMRI − DTI"}
VIEWS = [(320.0, 345.0, 150.0), (325.0, 0.0, 0.0)]  # lateral (top row) + superior (bottom row), user-set
VIEW_NAMES = ["Lateral", "Superior"]
ANODE_C, CATHODE_C = (211 / 255, 47 / 255, 47 / 255), (25 / 255, 118 / 255, 210 / 255)


def ensure_magnE(subj, model, m2m):
    """Per-voxel |E| on the T1 grid for one model, cached as a NIfTI; built from the sim mesh if absent."""
    out = os.path.join(R, subj, f"magnE_{model}_T1.nii.gz")
    if os.path.exists(out):
        return out
    from simnibs import mesh_io
    mesh = os.path.join(COHORT, subj, "work", f"sim_M1_{model}", f"{subj}_TDCS_1_{SUFFIX[model]}.msh")
    t1 = nib.load(os.path.join(m2m, "T1.nii.gz"))
    m = mesh_io.read_msh(mesh)
    grid = m.field["magnE"].interpolate_to_grid(np.array(t1.shape), t1.affine)
    os.makedirs(os.path.join(R, subj), exist_ok=True)
    nib.save(nib.Nifti1Image(grid.astype(np.float32), t1.affine), out)
    return out


def render_surface_panel(m2m, out_png, vmax, niftis, rot, cmap="viridis", symmetric=False):
    """Render one cortical-surface panel (one process) at the given camera rotation."""
    args = [sys.executable, SURF_WORKER, "--m2m", m2m, "--out", out_png, "--cmap", cmap,
            "--vmax", str(vmax), "--rot", str(rot[0]), str(rot[1]), str(rot[2]), "--nifti"] + list(niftis)
    if symmetric:
        args.append("--symmetric")
    _run(args, out_png)


def render_montage_panel(subj, out_png):
    """Render the scalp+electrode montage (one process), building the capped head mesh if needed."""
    sim = os.path.join(COHORT, subj, "work", "sim_M1_ISO", f"{subj}_TDCS_1_scalar.msh")
    msh = os.path.join(R, subj, "head_montage.msh")
    _run([sys.executable, MON_WORKER, "--sim", sim, "--msh", msh, "--out", out_png,
          "--rot", "290", "350", "140"], out_png)


def _run(args, out_png):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_png):
        raise RuntimeError(f"render worker failed for {out_png}: {r.stderr[-400:]}")


def crop_white(path, pad=8):
    """Trim the white margin around a rendered surface to its bounding box, keeping a white border so the
    brain never touches the cell edge (a flush edge causes faint interpolation-ringing lines on compose)."""
    im = plt.imread(path)[..., :3].copy()
    # gmsh leaves a thin black strip on the render border; blank the outer edge to white so crop_white
    # locks onto the brain, not that strip (the strip otherwise aligns across panels as a faint line).
    b = max(3, int(0.03 * min(im.shape[:2])))
    im[:b] = 1.0
    im[-b:] = 1.0
    im[:, :b] = 1.0
    im[:, -b:] = 1.0
    fg = (im < 0.985).any(-1)
    ys, xs = np.where(fg)
    return im[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


def main():
    subj, _, _ = representative_subject(R, "M1")
    m2m = os.path.join(COHORT, subj, "work", f"m2m_{subj}")
    pdir = os.path.join(R, subj)
    print(f"representative subject = {subj}")

    nii = {m: ensure_magnE(subj, m, m2m) for m in MODELS}
    V, _ = load_central_surface(m2m)
    mag = {m: sample_volume_to_surface(nii[m], V) for m in MODELS}
    vmax = round(float(np.nanpercentile(np.concatenate(list(mag.values())), 95)), 2)  # ROAST-style p95 cap
    diff = mag["MD_dMRI"] - mag["DTI"]
    dmax = max(round(float(np.nanpercentile(np.abs(diff), 99)), 2), 0.01)
    med_abs = float(np.median(np.abs(diff)))
    pct = 100.0 * med_abs / float(np.median(mag["MD_dMRI"]))
    print(f"|E| vmax={vmax} V/m; diff +/-{dmax} V/m; median|delta|={med_abs:.4f} ({pct:.1f}% of median |E|)")

    # render the 8 cortical-surface panels (3 models + difference, each at two oblique views) + the montage
    reuse = os.environ.get("FIG3_REUSE_RENDERS") == "1"   # skip re-rendering (layout-only iteration)
    panels = {}
    for col in ["ISO", "DTI", "MD_dMRI", "DIFF"]:
        for r_i, rot in enumerate(VIEWS):
            p = os.path.join(pdir, f"surf_{col}_v{r_i}.png")
            if not (reuse and os.path.exists(p)):
                if col == "DIFF":
                    render_surface_panel(m2m, p, dmax, [nii["MD_dMRI"], nii["DTI"]], rot,
                                         cmap="bwr", symmetric=True)
                else:
                    render_surface_panel(m2m, p, vmax, [nii[col]], rot, cmap="parula")
            panels[(col, r_i)] = p
    head_png = os.path.join(pdir, "head_montage_panel.png")
    if not (reuse and os.path.exists(head_png)):
        render_montage_panel(subj, head_png)

    def pad_center(im, Ht, Wt):
        h, w = im.shape[:2]
        out = np.ones((max(Ht, h), max(Wt, w), im.shape[2]), im.dtype)
        y0 = (out.shape[0] - h) // 2
        x0 = (out.shape[1] - w) // 2
        out[y0:y0 + h, x0:x0 + w] = im
        return out

    # crop each panel, then pad PER ROW (not globally) so brains in a row share size; the row aspect drives
    # height_ratios so each row's cells match its brain shape - removes the vertical letterbox white + gap.
    imgs = {k: crop_white(p) for k, p in panels.items()}
    cols4 = ["ISO", "DTI", "MD_dMRI", "DIFF"]
    SCALE = 1.30   # pad each row's canvas to SCALE x the brain bbox -> brain fills ~1/SCALE of its cell (smaller)
    rowH = {r: int(max(imgs[(c, r)].shape[0] for c in cols4) * SCALE) for r in (0, 1)}
    rowW = {r: int(max(imgs[(c, r)].shape[1] for c in cols4) * SCALE) for r in (0, 1)}
    imgs = {k: pad_center(v, rowH[k[1]], rowW[k[1]]) for k, v in imgs.items()}
    h_ratios = [rowH[0] / rowW[0], rowH[1] / rowW[1]]

    # size the figure height so the brain cells match the brain aspect (no vertical letterbox / row gap)
    wr = [0.78, 1, 1, 1, 0.05, 1, 0.05]
    LEFT, RIGHT, TOP, BOT, FIG_W = 0.012, 0.965, 0.91, 0.04, 11.0
    col_w = 1.0 / sum(wr) * (RIGHT - LEFT)
    fig_h = FIG_W * sum(h_ratios) * col_w / (TOP - BOT)
    fig = plt.figure(figsize=(FIG_W, fig_h), dpi=300)
    gs = fig.add_gridspec(2, 7, width_ratios=wr, height_ratios=h_ratios,
                          wspace=0.02, hspace=0.0, left=LEFT, right=RIGHT, top=TOP, bottom=BOT)

    # Panel A: montage head sized to its own aspect (no wasted column height), key directly beneath
    head_img = crop_white(head_png)
    hh, ww = head_img.shape[:2]
    HW = 0.125
    HH = min(HW * (hh / ww) * (fig.get_figwidth() / fig.get_figheight()), 0.62)
    head_y1 = 0.85
    ax_head = fig.add_axes([0.014, head_y1 - HH, HW, HH])
    ax_head.imshow(head_img, interpolation="antialiased")
    ax_head.set_axis_off()
    ax_head.set_title("M1 montage", fontsize=9, color=INK, fontweight="bold")
    ax_head.text(0.0, 1.0, "A", transform=ax_head.transAxes, fontsize=12, fontweight="bold",
                 va="bottom", ha="left", color=INK)
    for i, (color, lab) in enumerate([(ANODE_C, "Anode (C3)"), (CATHODE_C, "Cathode (Fp2)")]):
        y = head_y1 - HH - 0.05 - i * 0.045
        fig.patches.append(Rectangle((0.03, y - 0.012), 0.015, 0.024, facecolor=color,
                                     edgecolor="none", transform=fig.transFigure, clip_on=False))
        fig.text(0.052, y, lab, fontsize=7, va="center", color=INK)

    # brain grid: models in columns 1-3, difference in column 5; vertical colorbars in 4 and 6
    grid_col = {"ISO": 1, "DTI": 2, "MD_dMRI": 3, "DIFF": 5}
    top_axes = {}
    for col, gc in grid_col.items():
        for r_i in range(2):
            ax = fig.add_subplot(gs[r_i, gc])
            ax.imshow(imgs[(col, r_i)], interpolation="antialiased")
            ax.set_axis_off()
            if r_i == 0:
                ax.set_title(TITLES[col], fontsize=9, color=INK, fontweight="bold")
                top_axes[col] = ax
            if gc == 1:
                ax.text(-0.045, 0.5, VIEW_NAMES[r_i], transform=ax.transAxes, fontsize=8,
                        color=SUB, rotation=90, va="center", ha="center")
                if r_i == 0:
                    ax.text(0.0, 1.0, "B", transform=ax.transAxes, fontsize=12, fontweight="bold",
                            va="bottom", ha="left", color=INK)

    # clean header rule under each column title (consistent across all four columns)
    for ax in top_axes.values():
        bb = ax.get_position()
        fig.add_artist(Line2D([bb.x0, bb.x1], [bb.y1, bb.y1], color="#9a9a9a", lw=0.7,
                              transform=fig.transFigure, clip_on=False))

    # short vertical colorbars on the right, centered against the upper (|E|) / mid block, reference-style
    CH = 0.40   # colorbar height as a figure fraction (short, not full-height)
    cax1 = fig.add_subplot(gs[:, 4])
    p1 = cax1.get_position()
    cax1.set_position([p1.x0, p1.y0 + (p1.height - CH) / 2, p1.width, CH])
    cb1 = fig.colorbar(ScalarMappable(Normalize(0, vmax), "parula"), cax=cax1, orientation="vertical")
    cb1.set_label("|E| (V/m)", fontsize=8)
    cb1.set_ticks([0, 0.1, 0.2, vmax])
    cb1.ax.tick_params(labelsize=7)
    cax2 = fig.add_subplot(gs[:, 6])
    p2 = cax2.get_position()
    cax2.set_position([p2.x0, p2.y0 + (p2.height - CH) / 2, p2.width, CH])
    cb = fig.colorbar(ScalarMappable(TwoSlopeNorm(0, -dmax, dmax), "bwr"), cax=cax2,
                      orientation="vertical")
    cb.set_label("Δ|E| (V/m)", fontsize=8, labelpad=-8)   # pull label in (wider ticks push it out)
    cb.set_ticks([-dmax, 0, dmax])
    cb.ax.set_yticklabels([f"−{dmax:.2f}", "0", f"{dmax:.2f}"])
    cb.ax.tick_params(labelsize=7)

    caption = (
        "Figure 3. Electrode montage and transcranial direct-current stimulation electric-field magnitude on "
        "the cortical surface for the three conductivity models. (A) The M1 montage on the scalp of a "
        "representative subject: anode over C3 (red), cathode over Fp2 (blue). (B) |E| from the M1 montage "
        "rendered on the middle gray-matter (central) surface for the isotropic (ISO), single-shell DTI, and "
        "b-tensor-encoded MD-dMRI (QTI) conductivity models, and the MD-dMRI minus DTI difference, each shown "
        "at a lateral (top) and a superior (bottom) view of the left hemisphere over the C3 target. The three "
        "model columns share one magnitude scale (parula, V/m), fixed and capped at the pooled 95th "
        f"percentile; the difference column uses a symmetric blue-white-red scale (V/m, +/-{dmax:.2f}) centered at "
        f"zero. The model difference is near zero across the cortex (median |delta E| = {med_abs:.3f} V/m, "
        f"about {pct:.0f}% of the local field) and is dominated by tissue-boundary effects; the white-matter "
        "model effect is reported in Table 2. The representative subject is the one whose mean white-matter-"
        "lobe model effect is nearest the cohort median.")
    save_fig(fig, "fig3_field_maps", caption)
    print("wrote fig3_field_maps.{png,pdf} + _caption.txt")


if __name__ == "__main__":
    main()
