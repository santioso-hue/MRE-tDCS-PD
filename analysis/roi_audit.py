"""roi_audit.py - Phase-1 audit: every analysis ROI with its atlas source and cohort-median voxel count,
plus Group 2 (CIT168 nuclei) pairwise mask overlap (Dice + fraction-of-smaller). Confirms the two-group
ROI scheme and the Section 2.7 "overlap minimally" wording. Reporting only; consumed by no other script.

Run: $SIMNIBS_BIN/simnibs_python analysis/roi_audit.py
Outputs (gitignored): analysis/results/roi_audit.csv, analysis/results/nuclei_dice.csv
"""
import os, glob, csv, itertools
import numpy as np
import nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHORT = os.path.join(ROOT, "data", "cohort_local")
OUT = os.path.join(ROOT, "analysis", "results")

G1 = [("Ctx_Frontal", "FreeSurfer recon-all Desikan (cortex)"),
      ("Ctx_Parietal", "FreeSurfer recon-all Desikan (cortex)"),
      ("Ctx_Temporal", "FreeSurfer recon-all Desikan (cortex)"),
      ("Ctx_Occipital", "FreeSurfer recon-all Desikan (cortex)"),
      ("WM_Frontal", "FreeSurfer wmparc (white matter)"),
      ("WM_Parietal", "FreeSurfer wmparc (white matter)"),
      ("WM_Temporal", "FreeSurfer wmparc (white matter)"),
      ("WM_Occipital", "FreeSurfer wmparc (white matter)"),
      ("CC", "FreeSurfer aseg corpus callosum"),
      ("Mesencephalon", "FreeSurfer Iglesias brainstem"),
      ("Pons", "FreeSurfer Iglesias brainstem")]
BG = ["Pu", "Ca", "NAC", "GPe", "GPi"]      # larger / well-resolved basal ganglia
MB = ["SNc", "SNr", "VTA", "RN", "STN"]     # midbrain + STN
NUCLEI = BG + MB
ATLAS_G2 = "CIT168/Pauli 2018"


def subjects():
    return sorted(d for d in glob.glob(os.path.join(COHORT, "*"))
                  if os.path.isdir(os.path.join(d, "registration")))


def vox(path):
    return int((np.asarray(nib.load(path).dataobj) > 0).sum()) if os.path.exists(path) else None


def main():
    os.makedirs(OUT, exist_ok=True)
    subs = subjects()
    print(f"ROI audit over {len(subs)} subjects\n")

    # ---- voxel-count audit ----
    rows = []
    for name, src in G1:
        counts = [vox(os.path.join(d, "registration", "freesurfer_rois", f"roi_{name}.nii.gz")) for d in subs]
        counts = [c for c in counts if c]
        rows.append(("Group 1", name, src, int(np.median(counts)) if counts else 0))
    for nuc in NUCLEI:
        for side in ("L", "R"):
            counts = [vox(os.path.join(d, "registration", "atlas_rois", "nuclei", f"roi_{side}_{nuc}.nii.gz"))
                      for d in subs]
            counts = [c for c in counts if c]
            rows.append(("Group 2", f"{nuc}_{side}", ATLAS_G2, int(np.median(counts)) if counts else 0))
    with open(os.path.join(OUT, "roi_audit.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "roi", "atlas_source", "vox_median"])
        w.writerows(rows)
    n_g1 = sum(1 for r in rows if r[0] == "Group 1"); n_g2 = sum(1 for r in rows if r[0] == "Group 2")
    print(f"roi_audit.csv: {n_g1} Group 1 + {n_g2} Group 2 ROIs")
    print(f"{'group':9}{'roi':16}{'vox_med':>9}  atlas")
    for g, name, src, v in rows:
        print(f"{g:9}{name:16}{v:>9}  {src}")

    # ---- Group 2 pairwise overlap (within hemisphere) ----
    def dice_sweep(group, label):
        pairs = list(itertools.combinations(group, 2))
        dv = {p: [] for p in pairs}; ov = {p: [] for p in pairs}
        for d in subs:
            nd = os.path.join(d, "registration", "atlas_rois", "nuclei")
            for side in ("L", "R"):
                m = {}
                ok = True
                for n in group:
                    p = os.path.join(nd, f"roi_{side}_{n}.nii.gz")
                    if not os.path.exists(p):
                        ok = False; break
                    m[n] = np.asarray(nib.load(p).dataobj) > 0
                if not ok:
                    continue
                for a, b in pairs:
                    inter = int(np.logical_and(m[a], m[b]).sum())
                    s = int(m[a].sum()) + int(m[b].sum())
                    sm = min(int(m[a].sum()), int(m[b].sum()))
                    dv[(a, b)].append(2.0 * inter / s if s else np.nan)
                    ov[(a, b)].append(inter / sm if sm else np.nan)
        out = []
        for p in pairs:
            out.append((label, f"{p[0]}-{p[1]}", float(np.nanmedian(dv[p])), float(np.nanmax(dv[p])),
                        float(np.nanmedian(ov[p])), float(np.nanmax(ov[p]))))
        return out

    drows = dice_sweep(BG, "basal_ganglia") + dice_sweep(MB, "midbrain")
    with open(os.path.join(OUT, "nuclei_dice.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "pair", "dice_med", "dice_max", "frac_smaller_med", "frac_smaller_max"])
        for r in drows:
            w.writerow([r[0], r[1]] + [f"{x:.4f}" for x in r[2:]])
    bg_worst = max((r for r in drows if r[0] == "basal_ganglia"), key=lambda r: r[3])
    print(f"\nnuclei_dice.csv written. Basal-ganglia worst pairwise Dice: {bg_worst[1]} = {bg_worst[3]:.3f} "
          f"({'PASS' if bg_worst[3] <= 0.05 else 'CHECK'}: Section 2.7 'overlap minimally' wants <= 0.05).")
    gpe = next(r for r in drows if r[1] == "GPe-GPi")
    print(f"GPe-GPi: dice med {gpe[2]:.3f}, max {gpe[3]:.3f}")


if __name__ == "__main__":
    main()
