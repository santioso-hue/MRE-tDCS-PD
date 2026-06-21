"""table2_to_latex.py - emit Table 2 (H1 per-ROI summary) as an editable LaTeX booktabs table for submission.
Reads analysis/results/m1_results_summary.csv; writes analysis/results/m1_h1_table.tex. Needs \\usepackage{booktabs}.
"""
import os, csv

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "results")
rows = list(csv.DictReader(open(os.path.join(R, "m1_results_summary.csv"))))
TIERHDR = {"Tier 1": r"\textit{Tier 1 --- cortical / white-matter lobes, corpus callosum, brainstem}",
           "Tier 2": r"\textit{Tier 2 --- subcortical gray matter (FreeSurfer aseg)}"}

def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")

def qfmt(q):
    return r"$<0.01$" if q < 0.01 else f"${q:.2f}$"

def dir_sym(sig, med):
    return r"$\blacktriangle$" if (sig and med > 0) else r"$\blacktriangledown$" if (sig and med < 0) else "ns"

caption = (r"Region-wise MD-dMRI vs.\ DTI electric-field comparison (H1), M1 montage "
           r"(anode C3 / cathode Fp2, 2\,mA), $n = 29$. $\Delta E_{\mathrm{model}}$, percent difference in the "
           r"95th-percentile $|E|$ between the MD-dMRI and DTI models; values are cohort median and IQR "
           r"(25th--75th percentile). $r_{\mathrm{rb}}$, matched-pairs rank-biserial effect size "
           r"(Wilcoxon signed-rank); $q$, Benjamini--Hochberg FDR-adjusted $p$. Dir: $\blacktriangle$ "
           r"MD-dMRI $>$ DTI, $\blacktriangledown$ MD-dMRI $<$ DTI ($q<0.05$); ns, not significant. "
           r"L/R, left/right hemisphere.")

out = [r"\begin{table}[t]", r"\centering", r"\small",
       r"\caption{" + caption + r"}", r"\label{tab:h1}",
       r"\begin{tabular}{@{}lrrrrc@{}}", r"\toprule",
       r"Region & $\Delta E_{\mathrm{model}}$ (\%) & IQR (\%) & $r_{\mathrm{rb}}$ & $q$ & Dir \\", r"\midrule"]
cur = None
for r in rows:
    if r["Tier"] != cur:
        if cur is not None:
            out.append(r"\addlinespace")
        out.append(r"\multicolumn{6}{@{}l}{" + TIERHDR[r["Tier"]] + r"} \\")
        cur = r["Tier"]
    med, lo, hi = fnum(r["dE_model_median_pct"]), fnum(r["dE_model_IQR_lo_pct"]), fnum(r["dE_model_IQR_hi_pct"])
    rb, q, sig = fnum(r["rank_biserial"]), fnum(r["q"]), r["MDdMRI_vs_DTI_significant"] == "Y"
    out.append(f"{r['ROI']} & ${med:+.1f}$ & $[{lo:+.1f}, {hi:+.1f}]$ & ${rb:+.2f}$ & "
               f"{qfmt(q)} & {dir_sym(sig, med)} \\\\")
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

with open(os.path.join(R, "m1_h1_table.tex"), "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote m1_h1_table.tex ({len(rows)} ROI rows)")
