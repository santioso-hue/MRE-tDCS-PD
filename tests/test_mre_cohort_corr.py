"""test_mre_cohort_corr.py - validate analysis/09_mre_cohort_corr.py on a synthetic cohort.

Plants a between-subject E-field-vs-stiffness correlation plus a group confound and asserts the engine
recovers the planted partial correlation, removes a purely group-driven spurious correlation, and does
not flag a null pair. Run: simnibs_python tests/test_mre_cohort_corr.py
"""
import os, sys, csv, json, subprocess, tempfile, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROIS = ["Ctx_Frontal", "WM_Frontal", "Putamen_L", "WholeBrain"]


def build(tmp, seed=0):
    rng = np.random.default_rng(seed)
    subs = [{"id": f"s{i:02d}", "group": "PD" if i < 12 else "HC", "age": int(rng.integers(55, 76))}
            for i in range(29)]
    grp = np.array([s["group"] == "PD" for s in subs], float)
    res = os.path.join(tmp, "results")
    for k, s in enumerate(subs):
        d = os.path.join(res, s["id"]); os.makedirs(d, exist_ok=True)
        rows = []
        for r in ROIS:
            stiff = 2000 + rng.normal(0, 80)
            md = 1.3 - 0.002 * (stiff - 2000) + rng.normal(0, 0.02)         # input check (NEG, like Olsson)
            e_md = 0.27 + 0.0004 * (stiff - 2000) + rng.normal(0, 0.004)    # planted POS E_MDdMRI vs stiffness
            e_iso = 0.27 + 0.03 * grp[k] + rng.normal(0, 0.01)             # group-only (should vanish w/ control)
            e_dti = 0.27 + rng.normal(0, 0.01)                             # null
            de = rng.normal(0, 3.0)                                         # null
            rows.append(dict(ROI=r, stiffness=stiff, MD=md, E_ISO=e_iso, E_DTI=e_dti,
                             E_MDdMRI=e_md, dE_model_pct=de))
        cols = ["ROI", "stiffness", "MD", "E_ISO", "E_DTI", "E_MDdMRI", "dE_model_pct"]
        with open(os.path.join(d, "mre_efield_per_roi.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
            for row in rows: w.writerow(row)
    cj = os.path.join(tmp, "cohort.json")
    json.dump({"subjects": subs}, open(cj, "w"))
    return cj, res


def main():
    tmp = tempfile.mkdtemp()
    try:
        cj, res = build(tmp)
        subprocess.run([sys.executable, os.path.join(ROOT, "analysis", "09_mre_cohort_corr.py"),
                        "--cohort", cj, "--results", res], check=True)
        wb = {}
        with open(os.path.join(res, "mre_cohort_corr_wholebrain.csv")) as f:
            for row in csv.DictReader(f):
                wb[row["pair"]] = row
        checks = []
        def ck(name, ok, detail):
            checks.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

        r_md = float(wb["E_MDdMRI_vs_stiffness"]["rho"]); q_md = float(wb["E_MDdMRI_vs_stiffness"]["q"])
        ck("planted E_MDdMRI-vs-stiffness recovered positive + significant", r_md > 0.3 and q_md < 0.05,
           f"rho={r_md:.2f}, q={q_md:.3g}")
        r_iso = float(wb["E_ISO_vs_stiffness"]["rho"]); q_iso = float(wb["E_ISO_vs_stiffness"]["q"])
        ck("group-only E_ISO correlation NOT significant after group control", not (q_iso < 0.05),
           f"rho={r_iso:.2f}, q={q_iso:.3g}")
        q_null = float(wb["dE_model_pct_vs_stiffness"]["q"])
        ck("null dE_model pair NOT significant", not (q_null < 0.05), f"q={q_null:.3g}")

        print(f"\n{'ALL PASS' if all(checks) else 'SOME CHECKS FAILED'}")
        sys.exit(0 if all(checks) else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
