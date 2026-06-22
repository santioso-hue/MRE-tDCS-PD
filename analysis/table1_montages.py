"""table1_montages.py - Table 1 (electrode montages) for the manuscript Methods 2.6.
Exposes the 2x2 design: two cortical targets (M1, DLPFC) crossed with two electrode types (conventional
5x5 cm two-pad, 4x1 HD ring). No redundant montage-ID column: each row is identified by (Target x Type).
Matches the house style of fig_table2_h1.py (DejaVu Sans, INK/SUB tokens, bold headers, thin booktabs rules,
faint target-group shading band consistent with the H1 tier shading, no gridlines, explicit-bbox crop).
Emits PNG, PDF, LaTeX (booktabs + multirow), CSV, and a standalone caption .txt. Values are fixed.
"""
import os, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans"})

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "results")
INK, SUB, BAND = "#1a1a1a", "#5f6368", "#e7e9ec"   # house tokens (fig_table2_h1.py); BAND is the H1 tier shade

# header label + left-x anchor (fraction of the table width). Target is rendered once per group (merged look).
HEADERS = [("Target", 0.005), ("Type", 0.115), ("Anode", 0.405), ("Return electrode(s)", 0.490), ("Current", 0.700)]
DATA_ANCHORS = [0.115, 0.405, 0.490, 0.700]   # Type, Anode, Return, Current
# grouped by target: each group is (target, [ [Type, Anode, Return, Current], ... ])
GROUPS = [
    ("M1", [
        ["Conventional 2-pad, 5×5 cm", "C3", "Fp2",            "+2 mA / −2 mA"],
        ["4×1 HD ring",                "C3", "Cz, F3, T7, P3", "+2 mA anode; −0.5 mA each return"],
    ]),
    ("DLPFC", [
        ["Conventional 2-pad, 5×5 cm", "F3", "Fp2",             "+2 mA / −2 mA"],
        ["4×1 HD ring",                "F3", "Fp1, Fz, C3, F7", "+2 mA anode; −0.5 mA each return"],
    ]),
]
CAPTION = (
    "Table 1. Electrode montages simulated. The four montages span two cortical targets (primary motor "
    "cortex, M1; dorsolateral prefrontal cortex, DLPFC) crossed with two electrode types: a conventional "
    "two-pad configuration (5 × 5 cm rectangular pads) and a 4 × 1 high-definition (HD) ring of small disc "
    "electrodes. All montages used an identical head mesh, finite-element solver, and 2 mA total current, "
    "and the two electrode types at each target share the anode position. Electrode labels are 10–10 EEG "
    "positions. Each pad montage delivers +2 mA at the anode and −2 mA at the single return; each HD montage "
    "delivers +2 mA at the central anode and −0.5 mA at each of four surrounding returns, summing to −2 mA. "
    "The left motor target (C3) is fixed across all subjects so that montage laterality does not confound "
    "the conductivity-model contrast. HD, high-definition 4 × 1 ring.")

# ---------- styled render (PNG + PDF) ----------
fig = plt.figure(figsize=(11.0, 3.0), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.patch.set_visible(False)
x0, x1 = 0.02, 0.98
def X(f): return x0 + (x1 - x0) * f
ax.text(0.5, 0.94, "Table 1.  Electrode montages", ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
yh = 0.78
ax.plot([x0, x1], [yh + 0.05, yh + 0.05], color=INK, lw=1.2)        # top rule
for name, fx in HEADERS:
    ax.text(X(fx), yh, name, ha="left", va="center", fontsize=9, fontweight="bold", color=SUB)
ax.plot([x0, x1], [yh - 0.05, yh - 0.05], color=INK, lw=1.0)        # under-header rule

ROWH, GAP = 0.135, 0.052
y = yh - 0.05 - ROWH * 0.60
for target, rows in GROUPS:
    top = y + ROWH * 0.5
    bot = y - ROWH * (len(rows) - 1) - ROWH * 0.5
    ax.add_patch(Rectangle((x0, bot), x1 - x0, top - bot, facecolor=BAND, edgecolor="none",
                           alpha=0.45, zorder=0))                   # faint target-group band (H1 tier shade)
    ax.text(X(0.005), (top + bot) / 2, target, ha="left", va="center", fontsize=9.5, fontweight="bold", color=INK)
    for row in rows:
        for val, fx in zip(row, DATA_ANCHORS):
            ax.text(X(fx), y, val, ha="left", va="center", fontsize=9, color=INK)
        y -= ROWH
    y -= GAP
y_bot = y + GAP + ROWH * 0.4
ax.plot([x0, x1], [y_bot, y_bot], color=INK, lw=1.2)               # bottom rule
W, H = fig.get_size_inches()
bb = Bbox.from_extents(0.01 * W, (y_bot - 0.03) * H, 0.99 * W, 0.99 * H)
fig.savefig(f"{R}/table1_montages.png", dpi=300, bbox_inches=bb, facecolor="white", pad_inches=0.08)
fig.savefig(f"{R}/table1_montages.pdf", bbox_inches=bb, facecolor="white", pad_inches=0.08)

# ---------- CSV (Target repeated per row for spreadsheet completeness; true minus, x as cross) ----------
with open(f"{R}/table1_montages.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow([h for h, _ in HEADERS])
    for target, rows in GROUPS:
        for row in rows:
            w.writerow([target] + row)

# ---------- LaTeX (booktabs + multirow; \addlinespace separates the two target blocks) ----------
tex = r"""% requires \usepackage{booktabs, multirow}
\begin{table}[t]
\centering
\small
\caption{__CAPTION__}
\label{tab:montages}
\begin{tabular}{@{}lllll@{}}
\toprule
Target & Type & Anode & Return electrode(s) & Current \\
\midrule
\multirow{2}{*}{M1} & Conventional 2-pad, $5\times5$\,cm & C3 & Fp2 & $+2$ / $-2$\,mA \\
 & $4\times1$ HD ring & C3 & Cz, F3, T7, P3 & $+2$\,mA anode; $-0.5$\,mA each return \\
\addlinespace
\multirow{2}{*}{DLPFC} & Conventional 2-pad, $5\times5$\,cm & F3 & Fp2 & $+2$ / $-2$\,mA \\
 & $4\times1$ HD ring & F3 & Fp1, Fz, C3, F7 & $+2$\,mA anode; $-0.5$\,mA each return \\
\bottomrule
\end{tabular}
\end{table}
""".replace("__CAPTION__", CAPTION)
open(f"{R}/table1_montages.tex", "w").write(tex)
open(f"{R}/table1_caption.txt", "w").write(CAPTION + "\n")
print("wrote table1_montages.{png,pdf,tex,csv} + table1_caption.txt (2x2: Target x Type)")
