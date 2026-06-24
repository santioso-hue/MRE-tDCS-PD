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

Usage:  <simnibs_python> analysis/06_cohort_stats.py [--stat p95|median] [--cohort config/cohort.json]
"""
import os, sys, csv, json, glob, argparse
import numpy as np
from scipy import stats
from _stats import bh_fdr, cohens_d, residualize, partial_pearson, rank_biserial_paired

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["ISO", "DTI", "MD-dMRI"]
WHOLE_BRAIN = "WholeBrain"   # aggregate row; excluded from per-family FDR so it does not inflate m
GROUP2 = {f"{n}_{s}" for n in ("Pu", "Ca", "NAC", "GPe", "GPi", "SNc", "SNr", "VTA", "RN", "STN")
          for s in ("L", "R")}   # Group 2 deep nuclei: exploratory, own FDR family
MIN_PAIRS = 6                # minimum finite paired observations to run a Wilcoxon test


def load_matrix(subjects, results_dir, montage, stat):
    """Return rois (list), {model: array[n_kept, n_roi]}, and the kept-subject list (those with a CSV).
    A subject missing its per-montage CSV is skipped with a warning, not fatal, so a partial cohort (e.g.
    27/29 succeeded overnight) still aggregates; raises only if NO subject has a CSV for this montage.
    The kept list lets the caller re-align the parallel group/age arrays to the surviving subjects."""
    rois, per_subj, kept, dropped = None, {m: [] for m in MODELS}, [], []
    for s in subjects:
        path = os.path.join(results_dir, s["id"], f"roi_efield_{montage}.csv")
        if not os.path.exists(path):
            dropped.append(s["id"]); continue
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
        kept.append(s)
    if not kept:
        raise FileNotFoundError(f"no subject has results/<id>/roi_efield_{montage}.csv "
                                f"(run the cohort runner for {montage})")
    if dropped:
        print(f"  [{montage}] skipped {len(dropped)} subject(s) with no CSV: {', '.join(dropped)}")
    return rois, {m: np.asarray(v, float) for m, v in per_subj.items()}, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stat", choices=["p95", "median"], default="p95")
    ap.add_argument("--cohort", default=os.path.join(ROOT, "config", "cohort.json"))
    ap.add_argument("--results", default=os.path.join(ROOT, "analysis", "results"))
    args = ap.parse_args()

    with open(args.cohort) as f:
        cohort = json.load(f)
    subjects = cohort["subjects"]
    manifest_ids = {s["id"] for s in subjects}
    n_pd_all = sum(1 for s in subjects if s["group"] == "PD")
    print(f"Cohort: {len(subjects)} subjects ({n_pd_all} PD, {len(subjects) - n_pd_all} HC). "
          f"Statistic: {args.stat}")

    for montage in cohort["montages"]:
        # A subject with results on disk but no manifest entry is read by nobody (load_matrix iterates the
        # manifest), so it drops out of the stats silently. Surface it; the fix is to regenerate the
        # manifest with analysis/build_cohort_manifest.py so the subject set tracks the share.
        on_disk = {os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(args.results, "*", f"roi_efield_{montage}.csv"))}
        unlisted = sorted(on_disk - manifest_ids)
        if unlisted:
            print(f"  [{montage}] WARNING: {len(unlisted)} subject(s) have results but are absent from "
                  f"the manifest -> excluded from stats: {', '.join(unlisted)}")
        rois, mats, kept = load_matrix(subjects, args.results, montage, args.stat)
        group = np.array([1 if s["group"] == "PD" else 0 for s in kept])   # 1=PD, 0=HC, aligned to kept
        age = np.array([float(s["age"]) for s in kept])
        print(f"  [{montage}] aggregating {len(kept)} subjects "
              f"({int(group.sum())} PD, {int((1 - group).sum())} HC)")
        md = mats["MD-dMRI"]
        rows = []
        for j, roi in enumerate(rois):
            v = md[:, j]
            fin = np.isfinite(v)
            # H1 paired (every subject as own control); NaN-safe over finite pairs only
            h1, n_h1 = {}, {}
            for ref in ("DTI", "ISO"):
                a, b = md[:, j], mats[ref][:, j]
                m = np.isfinite(a) & np.isfinite(b)
                n_h1[ref] = int(m.sum())
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
            # H3 PD vs HC on age-adjusted residuals. Finite-mask v/age/group together FIRST: a single NaN
            # subject would otherwise make lstsq return an all-NaN beta and null the whole ROI (the Group 2
            # nuclei, which 04 legitimately writes NaN for when empty, are exactly the affected ROIs).
            vf, agef, gf = v[fin], age[fin], group[fin]
            if vf.size >= MIN_PAIRS and len(set(vf)) > 1:
                resid = residualize(vf, agef)
                rp, rh = resid[gf == 1], resid[gf == 0]
                if rp.size >= 1 and rh.size >= 1:
                    try:
                        _, p_h3 = stats.mannwhitneyu(rp, rh, alternative="two-sided")
                    except ValueError:
                        p_h3 = np.nan
                    d_h3 = cohens_d(rp, rh) if rp.size >= 2 and rh.size >= 2 else np.nan
                else:
                    p_h3, d_h3 = np.nan, np.nan
            else:
                p_h3, d_h3 = np.nan, np.nan
            # Age, controlling for group; finite-masked for the same reason
            if vf.size >= 4:
                r_age, p_age = partial_pearson(vf, agef, gf.reshape(-1, 1))
            else:
                r_age, p_age = np.nan, np.nan
            rows.append(dict(roi=roi,
                             md_med=float(np.nanmedian(v)) if fin.any() else np.nan,
                             iso_med=float(np.nanmedian(mats["ISO"][:, j]))
                             if np.isfinite(mats["ISO"][:, j]).any() else np.nan,
                             dti_med=float(np.nanmedian(mats["DTI"][:, j]))
                             if np.isfinite(mats["DTI"][:, j]).any() else np.nan,
                             n_h1_dti=n_h1["DTI"], n_h1_iso=n_h1["ISO"],
                             p_h1_dti=h1["DTI"][0], es_h1_dti=h1["DTI"][1],
                             p_h1_iso=h1["ISO"][0], es_h1_iso=h1["ISO"][1],
                             p_h3=p_h3, d_h3=d_h3, r_age=r_age, p_age=p_age))
        # FDR-BH within each family SEPARATELY: the primary Group 1 cross-comparison regions, and the
        # exploratory Group 2 deep nuclei. WholeBrain is excluded from both. Keeping Group 2 in its own
        # family avoids inflating the primary m.
        def fdr_family(key):
            q = np.full(len(rows), np.nan)
            primary = [i for i, r in enumerate(rows) if r["roi"] not in GROUP2 and r["roi"] != WHOLE_BRAIN]
            exploratory = [i for i, r in enumerate(rows) if r["roi"] in GROUP2]
            for members in (primary, exploratory):
                if members:
                    fq = bh_fdr([rows[i][key] for i in members])
                    for j, i in enumerate(members):
                        q[i] = fq[j]
            return q
        q_h1_dti = fdr_family("p_h1_dti")
        q_h1_iso = fdr_family("p_h1_iso")
        q_h3 = fdr_family("p_h3")
        q_age = fdr_family("p_age")
        for i, r in enumerate(rows):
            r.update(q_h1_dti=q_h1_dti[i], q_h1_iso=q_h1_iso[i], q_h3=q_h3[i], q_age=q_age[i])

        # DTI coverage: a subject with no matched DTI scan runs ISO+MD-dMRI only -> NaN DTI column -> fewer
        # MD-vs-DTI pairs. Surface it so the primary hypothesis is not silently underpowered.
        n_dti_cov = max((r["n_h1_dti"] for r in rows), default=0)
        if n_dti_cov < len(kept):
            print(f"  [{montage}] WARNING: DTI model present for at most {n_dti_cov}/{len(kept)} subjects "
                  f"-> H1 MD-vs-DTI underpowered where DTI is absent")

        os.makedirs(args.results, exist_ok=True)
        out = os.path.join(args.results, f"cohort_stats_{montage}_{args.stat}.csv")
        cols = ["roi", "md_med", "iso_med", "dti_med", "n_h1_dti", "p_h1_dti", "q_h1_dti", "es_h1_dti",
                "n_h1_iso", "p_h1_iso", "q_h1_iso", "es_h1_iso",
                "p_h3", "q_h3", "d_h3", "r_age", "p_age", "q_age"]
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for r in rows:
                w.writerow([r["roi"]] + [f"{r[c]:.5g}" if isinstance(r[c], float) else r[c] for c in cols[1:]])
        sig = sum(1 for r in rows if np.isfinite(r["q_h1_dti"]) and r["q_h1_dti"] < 0.05)
        print(f"  [{montage}] {len(rois)} ROIs -> {out}  (H1 MD-vs-DTI: {sig} ROIs at q<0.05)")


if __name__ == "__main__":
    main()
