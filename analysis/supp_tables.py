"""Emit the supplement tables as editable deliverables (booktabs .tex + .csv + standalone caption).
S-Tables 1-3 read analysis/results/cohort_stats_<montage>_<stat>.csv (stage 06, default p95);
S-Table 4 reads the per-subject analysis/results/<id>/tensor_divergence.csv (stage 08):

  S-Table 1 - H3 PD-vs-HC exploratory, MD-dMRI model, M1, Group 1 (Region | d | p | q).
  S-Table 2 - Group 2 nuclei H1 field results (the 20 CIT168 nuclei), M1 (Region | r_rb | p | q | n).
  S-Table 3 - non-M1 H1, Group 1, one block per montage DLPFC / HD_M1 / HD_DLPFC (Region | r_rb | q).
  S-Table 4 - DTI-vs-<D> FA and eigenvalue-ratio difference per Group 1 region (Region | dFA | d(lam1/lam3)).

The .tex + .csv + caption are the deliverable; the .tex needs \\usepackage{booktabs} downstream.
Number formatting and underscore escaping match table2_to_latex.py (Table 2).
"""
import os, csv, glob, argparse
import numpy as np

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis", "results")
GROUP2 = {f"{n}_{s}" for n in ("Pu", "Ca", "NAC", "GPe", "GPi", "SNc", "SNr", "VTA", "RN", "STN")
          for s in ("L", "R")}
WHOLE_BRAIN = "WholeBrain"
NONM1 = ["DLPFC", "HD_M1", "HD_DLPFC"]
MONTAGE_LABEL = {"DLPFC": "DLPFC pad", "HD_M1": "HD M1", "HD_DLPFC": "HD DLPFC"}
G1_ORDER = ["Ctx_Frontal", "Ctx_Parietal", "Ctx_Temporal", "Ctx_Occipital",
            "WM_Frontal", "WM_Parietal", "WM_Temporal", "WM_Occipital", "CC", "Mesencephalon", "Pons"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def signed2(x):
    # signed effect size (rank-biserial / Cohen's d), 2 decimals, true minus sign (U+2212) for the CSV
    return "" if x != x else f"{x:+.2f}".replace("-", "−")


def pfmt(p):
    return "" if p != p else ("<0.001" if p < 0.001 else f"{p:.3f}")


def qfmt(q):
    return "" if q != q else ("<0.01" if q < 0.01 else f"{q:.2f}")


def t_signed2(x):
    # LaTeX math mode: the ASCII hyphen typesets as a true minus
    return "" if x != x else f"${x:+.2f}$"


def tp(p):
    return "" if p != p else (r"$<0.001$" if p < 0.001 else f"${p:.3f}$")


def tq(q):
    return "" if q != q else (r"$<0.01$" if q < 0.01 else f"${q:.2f}$")


def signed3(x):
    return "" if x != x else f"{x:+.3f}".replace("-", "−")


def t_signed3(x):
    return "" if x != x else f"${x:+.3f}$"


def tex_roi(roi):
    return roi.replace("_", r"\_")


def load_rows(montage, stat):
    src = os.path.join(R, f"cohort_stats_{montage}_{stat}.csv")
    return list(csv.DictReader(open(src)))


def group1(rows):
    return [r for r in rows if r["roi"] != WHOLE_BRAIN and r["roi"] not in GROUP2]


def group2(rows):
    return [r for r in rows if r["roi"] in GROUP2]


def write_table(basename, csv_header, csv_records, tex_body, colspec, tex_header, caption):
    """Write basename.csv + basename.tex (booktabs) + basename_caption.txt."""
    with open(os.path.join(R, basename + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(csv_header)
        for rec in csv_records:
            w.writerow(rec)
    out = [r"% requires \usepackage{booktabs}",
           r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{" + caption.replace("|E|", r"$|E|$") + r"}",
           r"\label{tab:" + basename + r"}",
           r"\begin{tabular}{" + colspec + r"}", r"\toprule",
           tex_header, r"\midrule"]
    out += tex_body
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(R, basename + ".tex"), "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(R, basename + "_caption.txt"), "w") as f:
        f.write(caption + "\n")


def s_table1_h3(stat):
    rows = group1(load_rows("M1", stat))
    csv_records, tex_body = [], []
    for r in rows:
        d, p, q = fnum(r["d_h3"]), fnum(r["p_h3"]), fnum(r["q_h3"])
        csv_records.append([r["roi"], signed2(d), pfmt(p), qfmt(q)])
        tex_body.append(f"{tex_roi(r['roi'])} & {t_signed2(d)} & {tp(p)} & {tq(q)} \\\\")
    caption = (
        "Supplementary Table 1. Exploratory Parkinson-disease-versus-healthy-control comparison of the "
        "MD-dMRI p95 electric-field magnitude across the 11 Group 1 regions, M1 montage (anode C3 / "
        "cathode Fp2, 2 mA), n = 29. Region values were adjusted for a linear age effect (ordinary "
        "least-squares residuals across all subjects); d is Cohen's d on the residuals (positive = "
        "Parkinson disease higher); p is the two-sided Mann-Whitney U p-value; q is the Benjamini-Hochberg "
        "false-discovery-rate-adjusted p-value within the Group 1 family. Ctx, cortical lobe; WM, "
        "white-matter lobe; CC, corpus callosum. p < 0.001 is shown as <0.001 and q < 0.01 as <0.01. "
        "This comparison is exploratory and underpowered; it is reported as a null and is not promoted to "
        "the main text.")
    write_table("s_table1_h3",
                ["Region", "d", "p", "q"], csv_records,
                tex_body, r"@{}lrrr@{}",
                r"Region & $d$ & $p$ & $q$ \\", caption)
    print(f"wrote s_table1_h3.tex + .csv + _caption.txt ({len(csv_records)} Group 1 rows)")


def s_table2_nuclei_h1(stat):
    rows = group2(load_rows("M1", stat))
    csv_records, tex_body = [], []
    for r in rows:
        es, p, q = fnum(r["es_h1_dti"]), fnum(r["p_h1_dti"]), fnum(r["q_h1_dti"])
        n = r["n_h1_dti"]
        csv_records.append([r["roi"], signed2(es), pfmt(p), qfmt(q), n])
        tex_body.append(f"{tex_roi(r['roi'])} & {t_signed2(es)} & {tp(p)} & {tq(q)} & {n} \\\\")
    caption = (
        "Supplementary Table 2. MD-dMRI versus DTI p95 electric-field comparison (H1) for the 20 deep "
        "stimulation-relevant nuclei (CIT168 atlas; Pu putamen, Ca caudate, NAC nucleus accumbens, GPe / "
        "GPi external / internal globus pallidus, SNc / SNr substantia nigra pars compacta / reticulata, "
        "VTA ventral tegmental area, RN red nucleus, STN subthalamic nucleus; L / R left / right "
        "hemisphere), M1 montage (anode C3 / cathode Fp2, 2 mA). r_rb is the matched-pairs rank-biserial "
        "effect size for MD-dMRI versus DTI (Wilcoxon signed-rank, paired across subjects; positive = "
        "MD-dMRI higher); p is the two-sided Wilcoxon signed-rank p-value; q is the Benjamini-Hochberg "
        "false-discovery-rate-adjusted p-value within the exploratory Group 2 family; n is the number of "
        "paired subjects. p < 0.001 is shown as <0.001 and q < 0.01 as <0.01. These nuclei are at or near "
        "the diffusion voxel size and are reported as exploratory, field-only.")
    write_table("s_table2_nuclei_h1",
                ["Region", "r_rb", "p", "q", "n"], csv_records,
                tex_body, r"@{}lrrrr@{}",
                r"Region & $r_{\mathrm{rb}}$ & $p$ & $q$ & $n$ \\", caption)
    print(f"wrote s_table2_nuclei_h1.tex + .csv + _caption.txt ({len(csv_records)} nuclei rows)")


def s_table3_nonM1_h1(stat):
    csv_records, tex_body = [], []
    for i, montage in enumerate(NONM1):
        rows = group1(load_rows(montage, stat))
        if i:
            tex_body.append(r"\addlinespace")
        tex_body.append(r"\multicolumn{3}{@{}l}{\textit{" + MONTAGE_LABEL[montage] + r"}} \\")
        for r in rows:
            es, q = fnum(r["es_h1_dti"]), fnum(r["q_h1_dti"])
            csv_records.append([MONTAGE_LABEL[montage], r["roi"], signed2(es), qfmt(q)])
            tex_body.append(f"{tex_roi(r['roi'])} & {t_signed2(es)} & {tq(q)} \\\\")
    caption = (
        "Supplementary Table 3. MD-dMRI versus DTI p95 electric-field comparison (H1) across the 11 Group 1 "
        "regions for the three non-M1 montages (DLPFC pad, anode F3 / cathode Fp2; HD M1, 4x1 ring at C3; "
        "HD DLPFC, 4x1 ring at F3; all 2 mA), n = 29. r_rb is the matched-pairs rank-biserial effect size "
        "for MD-dMRI versus DTI (Wilcoxon signed-rank, paired across subjects; positive = MD-dMRI higher); "
        "q is the Benjamini-Hochberg false-discovery-rate-adjusted p-value within the Group 1 family of "
        "each montage. Ctx, cortical lobe; WM, white-matter lobe; CC, corpus callosum. q < 0.01 is shown "
        "as <0.01. The M1 montage is reported in Table 2 of the main text.")
    write_table("s_table3_nonM1_h1",
                ["Montage", "Region", "r_rb", "q"], csv_records,
                tex_body, r"@{}lrr@{}",
                r"Region & $r_{\mathrm{rb}}$ & $q$ \\", caption)
    print(f"wrote s_table3_nonM1_h1.tex + .csv + _caption.txt "
          f"({len(csv_records)} rows over {len(NONM1)} montages)")


def s_table4_tensor_divergence():
    """Cohort-median FA and lambda1/lambda3-ratio difference (<D> minus DTI) per Group 1 region, from the
    per-subject tensor_divergence.csv (stage 08). Complements Figure 2, which shows the V1 angle."""
    dfa = {r: [] for r in G1_ORDER}
    dr1 = {r: [] for r in G1_ORDER}
    for p in sorted(glob.glob(os.path.join(R, "PD*", "tensor_divergence.csv"))):
        for row in csv.DictReader(open(p)):
            roi = row["ROI"]
            if roi in dfa:
                dfa[roi].append(fnum(row["dFA_med"]))
                dr1[roi].append(fnum(row["dR1_med"]))
    csv_records, tex_body = [], []
    for roi in G1_ORDER:
        fa = float(np.nanmedian(dfa[roi])) if dfa[roi] else float("nan")
        r1 = float(np.nanmedian(dr1[roi])) if dr1[roi] else float("nan")
        csv_records.append([roi, signed3(fa), signed2(r1)])
        tex_body.append(f"{tex_roi(roi)} & {t_signed3(fa)} & {t_signed2(r1)} \\\\")
    caption = (
        "Supplementary Table 4. Per-region difference in fractional anisotropy (FA) and the principal "
        "eigenvalue ratio between the QTI mean tensor <D> and the single-shell DTI tensor, across the 11 "
        "Group 1 regions, n = 29. dFA is the cohort median of each subject's median FA(<D>) minus FA(DTI) "
        "over positive-definite voxels in the region (positive = <D> more anisotropic); d(lambda1/lambda3) "
        "is the corresponding cohort-median difference in the largest-to-smallest eigenvalue ratio. Ctx, "
        "cortical lobe; WM, white-matter lobe; CC, corpus callosum. FA is dimensionless. The "
        "principal-direction angle between the two tensors is shown in Figure 2.")
    write_table("s_table4_tensor_divergence",
                ["Region", "dFA", "d(lambda1/lambda3)"], csv_records,
                tex_body, r"@{}lrr@{}",
                r"Region & $\Delta$FA & $\Delta(\lambda_1/\lambda_3)$ \\", caption)
    print(f"wrote s_table4_tensor_divergence.tex + .csv + _caption.txt ({len(csv_records)} Group 1 rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", choices=["p95", "median"], default="p95")
    args = ap.parse_args()
    os.makedirs(R, exist_ok=True)
    s_table1_h3(args.stat)
    s_table2_nuclei_h1(args.stat)
    s_table3_nonM1_h1(args.stat)
    s_table4_tensor_divergence()


if __name__ == "__main__":
    main()
