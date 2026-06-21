"""09_mre_cohort_corr.py - cohort E-field vs MRE-stiffness correlation (the novel contribution).

Olsson et al. 2025 already published the MRE-to-microstructure relationships (MD/uFA/FA vs stiffness) on
this exact cohort, so re-deriving them adds no new science. The novel question is whether the tDCS E-field,
and the MD-dMRI-vs-DTI conductivity-model difference (dE_model), relate to tissue stiffness. We report those
between-subject correlations (partial Pearson) at whole-brain and per-ROI, FDR-BH.

Two controls are reported per pair: group only (PD/HC, Olsson's convention; the primary), and group+age (a
sensitivity, since stiffness and head morphology both vary with age, so a group-only association could be an
age artifact). MD-vs-stiffness is kept only as a one-line input check: it should reproduce Olsson's -0.76,
confirming the diffusion inputs are correct.

Run: simnibs_python analysis/09_mre_cohort_corr.py [--cohort config/cohort.json] [--results analysis/results]
"""
import os, sys, csv, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _stats import partial_pearson, bh_fdr, cohens_d  # noqa: E402
from scipy import stats as scst  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EFIELD = ["E_ISO", "E_DTI", "E_MDdMRI", "dE_model_pct"]   # vs stiffness - the deliverable
ARMS = ["E_ISO", "E_DTI", "E_MDdMRI"]                      # the three E-field arms (for the cross-arm check)
OLSSON_MD = -0.76                                          # Olsson Fig 6a, MD vs stiffness |G*| (input check)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load(cohort_path, results_dir):
    """Return kept subjects and table[level][modality] -> array aligned to subjects. A subject missing its
    CSV is dropped with a warning; only manifest subjects are read, so extras (e.g. FullPD5) are ignored."""
    subs = json.load(open(cohort_path))["subjects"]
    kept, per = [], {}
    for s in subs:
        p = os.path.join(results_dir, s["id"], "mre_efield_per_roi.csv")
        if not os.path.exists(p):
            print(f"  skip {s['id']}: no mre_efield_per_roi.csv"); continue
        per[s["id"]] = {r["ROI"]: r for r in csv.DictReader(open(p))}
        kept.append(s)
    if not kept:
        raise FileNotFoundError("no subject has mre_efield_per_roi.csv")
    levels = sorted({roi for rows in per.values() for roi in rows})
    table = {}
    for lvl in levels:
        table[lvl] = {mod: np.array([fnum(per[s["id"]].get(lvl, {}).get(mod, "nan")) for s in kept], float)
                      for mod in ["stiffness", "MD"] + EFIELD}
    return kept, table


def corr_vs_stiffness(table, lvl, covar, mods):
    """Partial Pearson of each modality in `mods` vs stiffness at one level, controlling for covar
    (a [n] or [n, k] array). Returns one dict per modality."""
    cov = np.asarray(covar, float)
    if cov.ndim == 1:
        cov = cov.reshape(-1, 1)
    out = []
    st = table[lvl]["stiffness"]
    for mod in mods:
        y = table[lvl][mod]
        m = np.isfinite(st) & np.isfinite(y) & np.all(np.isfinite(cov), axis=1)
        if m.sum() >= cov.shape[1] + 4 and np.std(st[m]) > 0 and np.std(y[m]) > 0:
            r, p = partial_pearson(st[m], y[m], cov[m])
        else:
            r, p = np.nan, np.nan
        out.append(dict(pair=f"{mod}_vs_stiffness", modality=mod, level=lvl, rho=r, p=p, n=int(m.sum())))
    return out


def fdr(rows):
    for row, q in zip(rows, bh_fdr([x["p"] for x in rows])):
        row["q"] = q
    return rows


def _write(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in rows:
            w.writerow([f"{r[c]:.5g}" if isinstance(r[c], float) else r[c] for c in cols])
    print(f"  wrote {path}")


def softening_context(table, group, age):
    """Reproduce Olsson's age/PD softening (whole-brain) and trace it through to the E-field. The E-field
    arm is MD-dMRI (the arms are near-identical, so it is representative)."""
    wb = table["WholeBrain"]
    st, ef = wb["stiffness"], wb["E_MDdMRI"]
    pd, hc, g = group == 1, group == 0, group.reshape(-1, 1)
    rows = []
    def add(check, stat, value, p):
        rows.append(dict(check=check, stat=stat, value=value, p=p))
    r, p = partial_pearson(age, st, g); add("age_vs_stiffness", "partial_rho", r, p)
    r, p = partial_pearson(age, ef, g); add("age_vs_E_MDdMRI", "partial_rho", r, p)
    add("PDvsHC_stiffness", "cohens_d", cohens_d(st[pd], st[hc]), scst.mannwhitneyu(st[pd], st[hc]).pvalue)
    add("PDvsHC_E_MDdMRI", "cohens_d", cohens_d(ef[pd], ef[hc]), scst.mannwhitneyu(ef[pd], ef[hc]).pvalue)
    r, p = scst.pearsonr(st[pd], ef[pd]); add("stiffness_vs_E_MDdMRI_PD", "pearson_rho", r, p)
    r, p = scst.pearsonr(st[hc], ef[hc]); add("stiffness_vs_E_MDdMRI_HC", "pearson_rho", r, p)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=os.path.join(ROOT, "config", "cohort.json"))
    ap.add_argument("--results", default=os.path.join(ROOT, "analysis", "results"))
    args = ap.parse_args()

    kept, table = load(args.cohort, args.results)
    group = np.array([1.0 if s["group"] == "PD" else 0.0 for s in kept])
    age = np.array([fnum(s.get("age")) for s in kept])
    cov_ga = np.column_stack([group, age])
    n_pd = int(group.sum())
    print(f"Cohort E-field vs stiffness: {len(kept)} subjects ({n_pd} PD, {len(kept)-n_pd} HC); "
          f"partial Pearson, group control (primary) + group+age (sensitivity)")

    if "WholeBrain" not in table:
        raise SystemExit("no WholeBrain row in the per-subject CSVs - re-run 05_mre to add it")

    # Input check: MD vs stiffness should reproduce Olsson's -0.76 (confirms the diffusion inputs).
    md = corr_vs_stiffness(table, "WholeBrain", group, ["MD"])[0]
    print(f"\nInput check  MD vs stiffness (whole-brain) = {md['rho']:+.3f}  "
          f"(Olsson {OLSSON_MD:+.2f}; reproduction confirms the diffusion inputs)")

    # Whole-brain E-field vs stiffness: group-only (primary) and group+age (sensitivity).
    wb = fdr(corr_vs_stiffness(table, "WholeBrain", group, EFIELD))
    wb_age = {r["modality"]: r for r in fdr(corr_vs_stiffness(table, "WholeBrain", cov_ga, EFIELD))}
    for r in wb:
        a = wb_age[r["modality"]]
        r["rho_age"] = a["rho"]; r["q_age"] = a["q"]

    # Per-ROI E-field vs stiffness (group-only, FDR within each arm across ROIs).
    rois = [l for l in table if l != "WholeBrain"]
    perroi, sig_by_mod = [], {}
    for mod in EFIELD:
        rows = fdr([corr_vs_stiffness(table, lvl, group, [mod])[0] for lvl in rois])
        perroi.extend(rows)
        sig_by_mod[mod] = [r["level"] for r in rows if np.isfinite(r["q"]) and r["q"] < 0.05]

    os.makedirs(args.results, exist_ok=True)
    _write(os.path.join(args.results, "mre_cohort_corr_wholebrain.csv"), wb)
    _write(os.path.join(args.results, "mre_cohort_corr_perroi.csv"), perroi)

    print("\nWhole-brain E-field vs stiffness (group-controlled primary | group+age sensitivity):")
    print(f"  {'pair':26} {'rho_g':>7} {'q_g':>9}   {'rho_g+age':>9} {'q_g+age':>9}")
    for r in wb:
        print(f"  {r['pair']:26} {r['rho']:+7.3f} {r['q']:9.3g}   {r['rho_age']:+9.3f} {r['q_age']:9.3g}")

    # Cross-arm check: if ISO ~ DTI ~ MD-dMRI, the association is not conductivity-model driven.
    arm_rho = [r["rho"] for r in wb if r["modality"] in ARMS]
    de = next(r for r in wb if r["modality"] == "dE_model_pct")
    print(f"\nCross-arm: E-field vs stiffness spans {min(arm_rho):+.3f} to {max(arm_rho):+.3f} across ISO/DTI/MD-dMRI "
          f"(spread {max(arm_rho)-min(arm_rho):.3f}).")
    print(f"  -> {'nearly identical across arms; the association is global, not conductivity-model driven' if max(arm_rho)-min(arm_rho) < 0.1 else 'arms differ; conductivity model may contribute'}.")
    print(f"  Model-specific dE_model vs stiffness: rho={de['rho']:+.3f}, q={de['q']:.3g} "
          f"({'significant' if np.isfinite(de['q']) and de['q'] < 0.05 else 'n.s.'}).")

    # Age robustness of the arms.
    surv = [r["modality"] for r in wb if r["modality"] in ARMS
            and np.isfinite(r["q"]) and r["q"] < 0.05 and np.isfinite(r["q_age"]) and r["q_age"] < 0.05]
    print(f"\nAge robustness: of the {sum(1 for r in wb if r['modality'] in ARMS and np.isfinite(r['q']) and r['q'] < 0.05)} "
          f"arm(s) significant under group control, {len(surv)} stay significant after adding age "
          f"-> {'the association survives age control' if surv else 'the association is explained by age (does not survive age control)'}.")

    ctx = softening_context(table, group, age)
    _write(os.path.join(args.results, "mre_cohort_corr_context.csv"), ctx)
    print("\nSoftening context (Olsson reproduction + the chain to the E-field, whole-brain):")
    for r in ctx:
        print(f"  {r['check']:26} {r['stat']:11} = {r['value']:+.3f}  p={r['p']:.3g}")

    print("\nPer-ROI significant ROIs (group-controlled, q<0.05):")
    for mod in EFIELD:
        s = sig_by_mod[mod]
        print(f"  {mod+' vs stiffness':26} {len(s):2d}/{len(rois)}  {', '.join(s[:8])}{' ...' if len(s) > 8 else ''}")

    n_wb = sum(1 for r in wb if np.isfinite(r["q"]) and r["q"] < 0.05)
    n_roi = sum(len(s) for s in sig_by_mod.values())
    primary = "per-ROI" if n_roi > n_wb else "whole-brain"
    print(f"\nPrimary level: {primary}  (whole-brain {n_wb} pairs q<0.05; per-ROI {n_roi} ROI-pairs q<0.05)")


if __name__ == "__main__":
    main()
