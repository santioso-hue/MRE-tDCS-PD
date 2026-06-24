"""_figstyle.py - the single house style for every manuscript figure: global rcParams, a colorblind-safe
palette, perceptual colormaps, and a panel-label helper. Importing this module applies the style, so the
whole figure set looks like one set. Fonts are DejaVu Sans (matplotlib's bundled sans; no system-font
dependency). Do NOT bake a "Figure N." title into the artwork - the number and title belong in the caption.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm  # noqa: F401  re-exported for zero-centered difference panels

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",              # Times-compatible math glyphs
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 400,                       # high-res raster (guide: not low quality)
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,                       # embed editable TrueType (no outlined text)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# Neutral ink tokens.
INK, SUB, GRAY = "#1a1a1a", "#5f6368", "#9aa0a6"
BAND, HL = "#e9ecef", "#eef3fb"

# Colorblind-safe categorical palette (Okabe & Ito 2008). MODEL identity is consistent across the whole set:
# DTI = blue, MD-dMRI = vermillion, ISO = gray. The Figure 2 tissue split uses blue (white matter) + gray
# (cortex) + green (brainstem) so it never reuses the MD-dMRI vermillion, which always means "model".
OKABE = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73", "sky": "#56B4E9",
         "orange": "#E69F00", "purple": "#CC79A7", "yellow": "#F0E442", "gray": "#7f7f7f"}
DTI_C, MDDMRI_C, ISO_C = OKABE["blue"], OKABE["vermillion"], OKABE["gray"]
WM_C, CORTEX_C, BRAINSTEM_C = OKABE["blue"], OKABE["gray"], OKABE["green"]
BLUE, ORANGE, RED = OKABE["blue"], OKABE["vermillion"], OKABE["vermillion"]   # back-compat aliases

# Perceptual, print- and colorblind-safe colormaps.
SEQ_CMAP = "magma"     # sequential scalar fields (|E|, V1 angle); dark->bright reads as "more"
DIV_CMAP = "RdBu_r"    # diverging, zero-centered differences (pair with TwoSlopeNorm or symmetric +/-vmax)


def panel_label(ax, letter, dx=-0.02, dy=1.04, fontsize=11, color=None):
    """Bold panel letter at a consistent top-left position (axes fraction) for every figure."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
            color=color or INK, va="bottom", ha="right")
