"""_figsave.py - one writer for every publication figure: 300 dpi PNG + vector PDF + standalone caption."""
import os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "results")


def save_fig(fig, basename, caption):
    """Write <basename>.png (300 dpi, white bg) + .pdf + _caption.txt to analysis/results/; return the PNG path."""
    os.makedirs(R, exist_ok=True)
    png = os.path.join(R, basename + ".png")
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.join(R, basename + ".pdf"), facecolor="white", bbox_inches="tight")
    with open(os.path.join(R, basename + "_caption.txt"), "w") as f:
        f.write(caption.strip() + "\n")
    print(f"wrote {basename}.png + .pdf + _caption.txt")
    return png
