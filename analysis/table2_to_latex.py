"""Emit Table 2 (H1 primary, Group 1, M1 montage) as editable deliverables.
Reads analysis/results/cohort_stats_M1_<stat>.csv (stage 06, default p95); writes
analysis/results/m1_h1_table.tex (booktabs) + .csv + m1_h1_table_caption.txt.
The .tex + .csv + caption are the deliverable; needs \\usepackage{booktabs} downstream.
Group 1 only (the 20 CIT168 nuclei and WholeBrain go to the supplement).

Layout (paper review: Caiani/Lampinen): the three |E| columns share a single spanning
header "p95 |E| (V/m)" via \\multicolumn + \\cmidrule, with ISO / DTI / MD-dMRI sub-headers;
then the contrast columns r_rb, p, q and the paired sample size n. Rows significant at
q < 0.05 are set in bold (Region cell + the four stat cells). The .csv stays flat (one
header row) so it remains machine-readable.
"""
import os, csv, argparse

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "results")
GROUP2 = {f"{n}_{s}" for n in ("Pu", "Ca", "NAC", "GPe", "GPi", "SNc", "SNr", "VTA", "RN", "STN")
          for s in ("L", "R")}

# Flat .csv header (one row, machine-readable). ISO/DTI/MD-dMRI |E| are the cohort-median p95 fields;
# n is the number of paired subjects in the MD-dMRI vs DTI Wilcoxon test (n_h1_dti).
COLS = ["Region", "ISO p95 |E| (V/m)", "DTI p95 |E| (V/m)", "MD-dMRI p95 |E| (V/m)",
        "r_rb", "p", "q", "n"]

SIG_Q = 0.05   # rows with q < SIG_Q are typeset in bold


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def inum(x):
    f = fnum(x)
    return "" if f != f else str(int(round(f)))


def efmt(x):
    return "" if x != x else f"{x:.3f}"


def rrb(x):
    # signed rank-biserial, 2 decimals, true minus sign (U+2212) for the plain-text .csv
    if x != x:
        return ""
    return f"{x:+.2f}".replace("-", "−")


def pfmt(p):
    return "" if p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")


def qfmt(q):
    return "" if q != q else ("<0.01" if q < 0.01 else f"{q:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", choices=["p95", "median"], default="p95")
    args = ap.parse_args()
    src = os.path.join(R, f"cohort_stats_M1_{args.stat}.csv")
    rows = [r for r in csv.DictReader(open(src))
            if r["roi"] != "WholeBrain" and r["roi"] not in GROUP2]

    recs = []
    for r in rows:
        recs.append({
            "roi": r["roi"],
            "iso": fnum(r["iso_med"]),
            "dti": fnum(r["dti_med"]),
            "md": fnum(r["md_med"]),
            "rrb": fnum(r["es_h1_dti"]),
            "p": fnum(r["p_h1_dti"]),
            "q": fnum(r["q_h1_dti"]),
            "n": inum(r["n_h1_dti"]),
        })

    # ---------- CSV (flat, one header row; true minus in r_rb; plain-text symbols) ----------
    with open(os.path.join(R, "m1_h1_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in recs:
            w.writerow([r["roi"], efmt(r["iso"]), efmt(r["dti"]), efmt(r["md"]),
                        rrb(r["rrb"]), pfmt(r["p"]), qfmt(r["q"]), r["n"]])

    # ---------- LaTeX (booktabs; grouped spanning header + bold significant rows) ----------
    def tp(p):
        return "" if p != p else (r"<0.001" if p < 0.001 else f"{p:.3f}")

    def tq(q):
        return "" if q != q else (r"<0.01" if q < 0.01 else f"{q:.2f}")

    def trrb(x):
        # ASCII hyphen typesets as a true minus inside math mode
        return "" if x != x else f"{x:+.2f}"

    def emph(s, bold):
        # wrap a plain-text cell in \textbf when the row is significant
        if s == "":
            return s
        return r"\textbf{" + s + "}" if bold else s

    def memph(s, bold):
        # wrap a math cell; \boldsymbol bolds the digits, sign and relational symbols
        if s == "":
            return s
        return (r"$\boldsymbol{" + s + "}$") if bold else (f"${s}$")

    out = [r"% requires \usepackage{booktabs}",
           r"% requires \usepackage{amsmath} for \boldsymbol (bold = q<0.05)",
           r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{__CAPTION__}", r"\label{tab:h1}",
           r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
           (r"Region & \multicolumn{3}{c}{p95 $|E|$ (V/m)} & "
            r"$r_{\mathrm{rb}}$ & $p$ & $q$ & $n$ \\"),
           r"\cmidrule(lr){2-4}",
           r" & ISO & DTI & MD-dMRI & & & & \\", r"\midrule"]
    for r in recs:
        bold = r["q"] == r["q"] and r["q"] < SIG_Q
        roi = r["roi"].replace("_", r"\_")
        out.append(
            f"{emph(roi, bold)} & {emph(efmt(r['iso']), bold)} & {emph(efmt(r['dti']), bold)} & "
            f"{emph(efmt(r['md']), bold)} & {memph(trrb(r['rrb']), bold)} & {memph(tp(r['p']), bold)} & "
            f"{memph(tq(r['q']), bold)} & {emph(r['n'], bold)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    caption = (
        "Table 2. Region-wise electric-field comparison across conductivity models for the 11 Group 1 "
        "regions (four cortical lobes, four white-matter lobes, corpus callosum, mesencephalon, pons), "
        "M1 montage (anode C3 / cathode Fp2, 2 mA), n = 29. Ctx, cortical lobe; WM, white-matter lobe; "
        "CC, corpus callosum. The three p95 |E| columns (V/m), grouped under the spanning header, are the "
        "cohort-median 95th-percentile electric-field magnitude over the gray-matter and white-matter "
        "elements of each region, for the isotropic (ISO), single-shell DTI, and MD-dMRI conductivity "
        "models. r_rb is the matched-pairs rank-biserial effect size for MD-dMRI versus DTI (Wilcoxon "
        "signed-rank, paired across subjects). p is the two-sided Wilcoxon signed-rank p-value. q is the "
        "Benjamini-Hochberg false-discovery-rate-adjusted p-value within the Group 1 family. n is the "
        "number of paired subjects in that test. p < 0.001 is shown as <0.001 and q < 0.01 as <0.01. "
        "Rows in bold are significant at q < 0.05.")

    # The .txt caption stays plain text (machine/human readable); the .tex caption needs LaTeX-safe
    # math for |E|, r_rb, and the inequality phrases. Apply targeted, order-sensitive replacements
    # (longer/inequality phrases first so the bare thresholds are not double-wrapped).
    tex_caption = caption
    for a, b in (
        ("|E|", r"$|E|$"),
        ("r_rb", r"$r_{\mathrm{rb}}$"),
        ("p < 0.001 is shown as <0.001", r"$p < 0.001$ is shown as ${<}0.001$"),
        ("q < 0.01 as <0.01", r"$q < 0.01$ as ${<}0.01$"),
        ("q < 0.05", r"$q < 0.05$"),
    ):
        tex_caption = tex_caption.replace(a, b)
    tex = "\n".join(out).replace("__CAPTION__", tex_caption)
    with open(os.path.join(R, "m1_h1_table.tex"), "w") as f:
        f.write(tex + "\n")
    with open(os.path.join(R, "m1_h1_table_caption.txt"), "w") as f:
        f.write(caption + "\n")
    print(f"wrote m1_h1_table.tex + .csv + m1_h1_table_caption.txt ({len(recs)} Group 1 rows)")


if __name__ == "__main__":
    main()
