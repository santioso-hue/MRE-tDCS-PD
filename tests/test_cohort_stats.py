"""
test_cohort_stats.py — validate analysis/06_cohort_stats.py against synthetic cohorts with KNOWN effects.

Builds a fake 29-subject cohort (12 PD, 17 HC) with planted signals and asserts the stats engine
recovers them: a consistent paired MD-dMRI>DTI ROI is H1-significant, a noisy one is not; a planted
PD>HC age-adjusted effect is H3-significant, a null ROI is not. No real data required.

Usage:  conda run -n neuro python tests/test_cohort_stats.py
"""
import os, sys, csv, json, subprocess, tempfile, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(tmp, seed=0):
    rng = np.random.default_rng(seed)
    subs = [{"id": f"s{i:02d}", "group": "PD" if i < 12 else "HC", "age": int(rng.integers(55, 76))}
            for i in range(29)]
    age = np.array([s["age"] for s in subs], float)
    pd = np.array([s["group"] == "PD" for s in subs])
    res = os.path.join(tmp, "results")
    rois = ["ROI_H1pos", "ROI_H1null", "ROI_H3pos", "ROI_null"]
    for k, s in enumerate(subs):
        dti = 0.30 + rng.normal(0, 0.03, len(rois))
        iso = 0.28 + rng.normal(0, 0.03, len(rois))
        md = dti.copy()
        md[0] = dti[0] + 0.05 + rng.normal(0, 0.005)            # H1pos: consistent MD>DTI
        md[1] = dti[1] + rng.normal(0, 0.05)                    # H1null: random sign
        md[2] = 0.30 + 0.06 * pd[k] + 0.004 * (age[k] - 65) + rng.normal(0, 0.02)  # H3: PD>HC + age
        md[3] = 0.30 + rng.normal(0, 0.02)                      # null
        d = os.path.join(res, s["id"]); os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "roi_efield_M1.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ROI", "ISO_p95", "ISO_median", "DTI_p95", "DTI_median", "MD-dMRI_p95", "MD-dMRI_median"])
            for r in range(len(rois)):
                w.writerow([rois[r], iso[r], iso[r] * 0.9, dti[r], dti[r] * 0.9, md[r], md[r] * 0.9])
    cohort = {"montages": ["M1"], "subjects": subs}
    cj = os.path.join(tmp, "cohort.json")
    json.dump(cohort, open(cj, "w"))
    return cj, res


def main():
    tmp = tempfile.mkdtemp()
    try:
        cj, res = build(tmp)
        subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "06_cohort_stats.py"),
                        "--cohort", cj, "--results", res, "--stat", "p95"], check=True)
        out = {}
        with open(os.path.join(res, "cohort_stats_M1_p95.csv")) as f:
            for row in csv.DictReader(f):
                out[row["roi"]] = {k: float(v) for k, v in row.items() if k != "roi"}

        checks = []
        def ck(name, ok, detail):
            checks.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

        ck("H1 consistent MD>DTI is significant", out["ROI_H1pos"]["q_h1_dti"] < 0.05,
           f"q={out['ROI_H1pos']['q_h1_dti']:.3g}, es={out['ROI_H1pos']['es_h1_dti']:.2f}")
        ck("H1 effect-size positive for MD>DTI", out["ROI_H1pos"]["es_h1_dti"] > 0.7,
           f"rank-biserial={out['ROI_H1pos']['es_h1_dti']:.2f}")
        ck("H1 noisy ROI is NOT significant", not (out["ROI_H1null"]["q_h1_dti"] < 0.05),
           f"q={out['ROI_H1null']['q_h1_dti']:.3g}")
        ck("H3 planted PD>HC is significant", out["ROI_H3pos"]["q_h3"] < 0.05,
           f"q={out['ROI_H3pos']['q_h3']:.3g}, d={out['ROI_H3pos']['d_h3']:.2f}")
        ck("H3 Cohen's d positive (PD>HC)", out["ROI_H3pos"]["d_h3"] > 0.5,
           f"d={out['ROI_H3pos']['d_h3']:.2f}")
        ck("null ROI is NOT H3-significant", not (out["ROI_null"]["q_h3"] < 0.05),
           f"q={out['ROI_null']['q_h3']:.3g}")

        print(f"\n{'ALL PASS' if all(checks) else 'SOME CHECKS FAILED'}")
        sys.exit(0 if all(checks) else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
