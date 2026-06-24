"""Methods table: the three conductivity models (ISO, DTI, MD-dMRI) by the assumptions that define them.

This is a MODEL-DEFINITION table, not a results table: every cell is a fixed property of how each arm
maps a diffusion measurement to a conductivity tensor (source quantity, acquisition, SimNIBS
anisotropy_type, the anisotropy it carries, the per-tissue scale, and the two SimNIBS caps). It contains no
per-subject or cohort numbers, so it is safe for the tracked repo. Columns are the three models; rows are the
model-defining assumptions, with a left rowname column. Emits a booktabs .tex + a .csv + a standalone
self-contained _caption.txt to analysis/results/ (the .tex needs \\usepackage{booktabs} downstream).

Source: pipeline/conductivity_models_derivation.md (the methods source of truth; canonical model table + caps).
"""
import os
import csv

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "results")

MINUS = "−"   # true minus U+2212 for signed text and the "not applicable" dash
DASH = MINUS       # the "-" cell (not applicable to the isotropic model) is a true minus

MODELS = ["ISO", "DTI", "MD-dMRI"]

# Each row: (rowname, [ISO cell, DTI cell, MD-dMRI cell]). Plain-text (CSV / caption) form; the .tex
# re-renders the few cells that need math (ratios, S/m) from the same content below.
ROWS = [
    ("Conductivity source",
        ["isotropic literature σ0",
         "single-shell DTI tensor (dwi2cond)",
         "QTI mean tensor ⟨D⟩"]),
    ("Diffusion acquisition",
        [DASH,
         "single-shell, ~b1500, 80-dir",
         "multi-shell b-tensor (QTI)"]),
    ("SimNIBS anisotropy_type",
        ["scalar", "vn", "vn"]),
    ("Anisotropy",
        ["none",
         "eigenvalue ratios + orientation",
         "eigenvalue ratios + orientation"]),
    ("Per-tissue scale σ0",
        ["WM 0.126 / GM 0.275 S/m",
         "WM 0.126 / GM 0.275 S/m",
         "WM 0.126 / GM 0.275 S/m"]),
    ("Eigenvalue-ratio cap",
        [DASH, "10:1", "10:1"]),
    ("Magnitude cap",
        [DASH, "2 S/m", "2 S/m"]),
]

# ---- LaTeX rendering of each cell (math where it helps; ASCII hyphen typesets as a true minus in math) ----
TEX_DASH = "$-$"
TEX_ROWS = [
    (r"Conductivity source",
        [r"isotropic literature $\sigma_0$",
         r"single-shell DTI tensor (dwi2cond)",
         r"QTI mean tensor $\langle D \rangle$"]),
    (r"Diffusion acquisition",
        [TEX_DASH,
         r"single-shell, ${\sim}b1500$, 80-dir",
         r"multi-shell $b$-tensor (QTI)"]),
    (r"SimNIBS \texttt{anisotropy\_type}",
        [r"\texttt{scalar}", r"\texttt{vn}", r"\texttt{vn}"]),
    (r"Anisotropy",
        [r"none",
         r"eigenvalue ratios + orientation",
         r"eigenvalue ratios + orientation"]),
    (r"Per-tissue scale $\sigma_0$",
        [r"WM 0.126 / GM 0.275\,S/m",
         r"WM 0.126 / GM 0.275\,S/m",
         r"WM 0.126 / GM 0.275\,S/m"]),
    (r"Eigenvalue-ratio cap",
        [TEX_DASH, r"$10{:}1$", r"$10{:}1$"]),
    (r"Magnitude cap",
        [TEX_DASH, r"$2$\,S/m", r"$2$\,S/m"]),
]

CAPTION = (
    "Table. Conductivity models compared, by their defining assumptions. Three finite-element tDCS arms "
    "share one head mesh, electrode montage, current, and solver, and (for the two anisotropic arms) one "
    "SimNIBS conductivity mapping; only the input tensor changes. ISO is the isotropic baseline; DTI is the "
    "SimNIBS-standard single-shell anisotropy baseline; MD-dMRI is the contribution, the same volume-"
    "normalized mapping fed the QTI mean diffusion tensor instead of the single-shell DTI tensor. "
    "anisotropy_type is the SimNIBS conductivity mode: scalar assigns each tissue its isotropic literature "
    "conductivity, while vn (volume-normalized) builds an anisotropic tensor sigma proportional to "
    "D / det(D)^(1/3), which normalizes the tensor eigenvalues to geometric mean 1 and then scales by the "
    "per-tissue literature value sigma0, so vn keeps only the tensor shape (eigenvalue ratios and "
    "orientation) and discards its magnitude. The per-tissue scale sigma0 (white-matter 0.126, grey-matter "
    "0.275 S/m) is identical across all three arms. Both anisotropic arms use the SimNIBS default caps, "
    "identical between them: the eigenvalue-ratio cap (aniso_maxratio = 10) limits the "
    "largest-to-smallest eigenvalue ratio to 10:1 and is the binding one, and the magnitude cap "
    "(aniso_maxcond = 2 S/m) limits the eigenvalue magnitude and is non-binding under vn. Because the caps "
    "and sigma0 are shared, the DTI-versus-MD-dMRI contrast reflects the input tensor, not the clip. A dash "
    "(" + DASH + ") marks an assumption that does not apply to the isotropic model. Conductivities are in "
    "siemens per metre (S/m). ISO, isotropic; DTI, single-shell diffusion tensor imaging; MD-dMRI, "
    "multidimensional diffusion MRI; QTI, q-space trajectory imaging; vn, volume-normalized; WM, white "
    "matter; GM, grey matter."
)


def write_table():
    os.makedirs(R, exist_ok=True)
    basename = "table_models"

    # ---- CSV (rowname column + one column per model; true minus and Unicode glyphs preserved) ----
    with open(os.path.join(R, basename + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Assumption"] + MODELS)
        for rowname, cells in ROWS:
            w.writerow([rowname] + cells)

    # ---- LaTeX (booktabs; left rowname column l + three left-aligned model columns) ----
    colspec = r"@{}llll@{}"
    header = r"\textbf{Assumption} & \textbf{ISO} & \textbf{DTI} & \textbf{MD-dMRI} \\"
    out = [r"% requires \usepackage{booktabs}",
           r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{" + CAPTION + r"}",
           r"\label{tab:models}",
           r"\begin{tabular}{" + colspec + r"}", r"\toprule",
           header, r"\midrule"]
    for rowname, cells in TEX_ROWS:
        out.append(f"{rowname} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(R, basename + ".tex"), "w") as f:
        f.write("\n".join(out) + "\n")

    # ---- standalone, self-contained caption ----
    with open(os.path.join(R, basename + "_caption.txt"), "w") as f:
        f.write(CAPTION + "\n")

    print(f"wrote {basename}.tex + .csv + _caption.txt ({len(ROWS)} assumption rows x {len(MODELS)} models)")


if __name__ == "__main__":
    write_table()
