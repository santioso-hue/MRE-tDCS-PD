"""
06_cohort_stats.py - Cohort statistics for the tDCS conductivity-model study (the main group deliverable).

Concatenates per-subject ROI E-field CSVs and computes, per ROI per montage:

  H1 (PRIMARY, powered): paired MD-dMRI vs DTI |E| across all subjects (Wilcoxon signed-rank), also
     MD-dMRI vs ISO. Each subject is its own control. Effect size = matched-pairs rank-biserial r.

  H3 (EXPLORATORY): PD vs HC, mirroring Olsson 2025: each ROI value is adjusted for a linear age effect
     (OLS residuals across all subjects), Mann-Whitney U on the residuals, Cohen's d. No gender
     correction (matches Olsson).

  Age: partial Pearson between the ROI value and age, controlling for group (PD/HC).

  Multiple comparisons: Benjamini-Hochberg FDR within each (montage, test-family) across ROIs.

Input schema (produced by the cohort runner, item A):
  config/cohort.json                          {montages:[...], subjects:[{id,group,age,...}]}
  results/<id>/roi_efield_<montage>.csv       columns: ROI, ISO_p95, ISO_median, DTI_p95, DTI_median,
                                              MD-dMRI_p95, MD-dMRI_median
Headline dosimetry uses p95 (Huang 2017); pass --stat median to mirror the MRE-comparison convention.

Output: results/cohort_stats_<montage>_<stat>.csv (one row per ROI; H1/H3/age columns with FDR q-values).

Usage:  conda run -n neuro python analysis/06_cohort_stats.py [--stat p95|median] [--cohort config/cohort.json]
"""
import os, sys, csv, json, argparse
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["ISO", "DTI", "MD-dMRI"]
WHOLE_BRAIN = "WholeBrain"   # aggregate row; excluded from per-family FDR so it does not inflate m
MIN_PAIRS = 6                # minimum finite paired observations to run a Wilcoxon test


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values. NaN p-values pass through as NaN and are excluded from ranking."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return q
    order = idx[np.argsort(p[idx])]
    m = idx.size
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = m - rank + 1
        val = p[i] * m / k
        prev = min(prev, val)
        q[i] = prev
    return q


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def residualize(y, X):
    """Return residuals of y on design X (with intercept added)."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_pearson(x, y, covar):
    """Pearson(x, y) controlling for covar (one or more columns)."""
    if len(x) < 4:
        return np.nan, np.nan
    rx = residualize(np.asarray(x, float), np.asarray(covar, float))
    ry = residualize(np.asarray(y, float), np.asarray(covar, float))
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(rx, ry)
    return r, p


def rank_biserial_paired(a, b):
    """Matched-pairs rank-biserial effect size for a-b (drops non-finite and zero differences)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    tot = ranks.sum()
    return (ranks[d > 0].sum() - ranks[d < 0].sum()) / tot


def load_matrix(subjects, results_dir, montage, stat):
    """Return rois (list), and {model: array[n_subj, n_roi]}, aligned to subjects, for the given stat."""
    rois, per_subj = None, {m: [] for m in MODELS}
    for s in subjects:
        path = os.path.join(results_dir, s["id"], f"roi_efield_{montage}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path} (run the cohort runner for {s['id']}/{montage})")
        rows = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                rows[row["ROI"]] = row
        names = list(rows)
        if rois is None:
            rois = names
        elif names != rois:
            # order matters for the matrix; report the set-diff (e.g. {'Brainstem'} vs
            # {'Mesencephalon','Pons'}) since a reorder gives the same diff.
            extra = set(names) - set(rois)
            missing = set(rois) - set(names)
            raise ValueError(
                f"ROI set for {s['id']}/{montage} differs from the first subject: "
                f"extra={extra or '{}'} missing={missing or '{}'} "
                f"(or same ROIs in a different order)")
        for m in MODELS:
            per_subj[m].append([float(rows[r][f"{m}_{stat}"]) for r in rois])
    return rois, {m: np.asarray(v, float) for m, v in per_subj.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", choices=["p95", "median"], default="p95")
    ap.add_argument("--cohort", default=os.path.join(ROOT, "config", "cohort.json"))
    ap.add_argument("--results", default=os.path.join(ROOT, "analysis", "results"))
    args = ap.parse_args()

    with open(args.cohort) as f:
        cohort = json.load(f)
    subjects = cohort["subjects"]
    group = np.array([1 if s["group"] == "PD" else 0 for s in subjects])   # 1=PD, 0=HC
    age = np.array([float(s["age"]) for s in subjects])
    n_pd, n_hc = int(group.sum()), int((1 - group).sum())
    print(f"Cohort: {len(subjects)} subjects ({n_pd} PD, {n_hc} HC). Statistic: {args.stat}")

    for montage in cohort["montages"]:
        rois, mats = load_matrix(subjects, args.results, montage, args.stat)
        md = mats["MD-dMRI"]
        rows = []
        for j, roi in enumerate(rois):
            v = md[:, j]
            # H1 paired (every subject as own control); NaN-safe over finite pairs only
            h1 = {}
            for ref in ("DTI", "ISO"):
                a, b = md[:, j], mats[ref][:, j]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() < MIN_PAIRS:
                    print(f"  [{montage}] {roi}: H1 MD-vs-{ref} skipped, "
                          f"only {int(m.sum())} finite pairs (< {MIN_PAIRS})")
                    h1[ref] = (np.nan, np.nan)
                    continue
                try:
                    _, p = stats.wilcoxon(a[m], b[m])
                except ValueError:
                    p = np.nan
                h1[ref] = (p, rank_biserial_paired(a[m], b[m]))
            # H3 PD vs HC on age-adjusted residuals; over finite residuals only
            resid = residualize(v, age)
            rp, rh = resid[group == 1], resid[group == 0]
            rp, rh = rp[np.isfinite(rp)], rh[np.isfinite(rh)]
            if rp.size >= 1 and rh.size >= 1 and len(set(v)) > 1:
                try:
                    _, p_h3 = stats.mannwhitneyu(rp, rh, alternative="two-sided")
                except ValueError:
                    p_h3 = np.nan
                d_h3 = cohens_d(rp, rh) if rp.size >= 2 and rh.size >= 2 else np.nan
            else:
                p_h3, d_h3 = np.nan, np.nan
            # Age, controlling for group
            r_age, p_age = partial_pearson(v, age, group.reshape(-1, 1))
            rows.append(dict(roi=roi,
                             md_med=float(np.median(v)),
                             p_h1_dti=h1["DTI"][0], es_h1_dti=h1["DTI"][1],
                             p_h1_iso=h1["ISO"][0], es_h1_iso=h1["ISO"][1],
                             p_h3=p_h3, d_h3=d_h3, r_age=r_age, p_age=p_age))
        # FDR-BH per (montage, family) across ROIs. WholeBrain is excluded so it does not inflate m
        # (passed as NaN -> NaN q).
        def fdr_no_wholebrain(key):
            ps = [(r[key] if r["roi"] != WHOLE_BRAIN else np.nan) for r in rows]
            return bh_fdr(ps)
        q_h1_dti = fdr_no_wholebrain("p_h1_dti")
        q_h1_iso = fdr_no_wholebrain("p_h1_iso")
        q_h3 = fdr_no_wholebrain("p_h3")
        q_age = fdr_no_wholebrain("p_age")
        for i, r in enumerate(rows):
            r.update(q_h1_dti=q_h1_dti[i], q_h1_iso=q_h1_iso[i], q_h3=q_h3[i], q_age=q_age[i])

        os.makedirs(args.results, exist_ok=True)
        out = os.path.join(args.results, f"cohort_stats_{montage}_{args.stat}.csv")
        cols = ["roi", "md_med", "p_h1_dti", "q_h1_dti", "es_h1_dti", "p_h1_iso", "q_h1_iso", "es_h1_iso",
                "p_h3", "q_h3", "d_h3", "r_age", "p_age", "q_age"]
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in rows:
                w.writerow([r["roi"]] + [f"{r[c]:.5g}" if isinstance(r[c], float) else r[c] for c in cols[1:]])
        sig = sum(1 for r in rows if np.isfinite(r["q_h1_dti"]) and r["q_h1_dti"] < 0.05)
        print(f"  [{montage}] {len(rois)} ROIs -> {out}  (H1 MD-vs-DTI: {sig} ROIs at q<0.05)")


if __name__ == "__main__":
    main()
