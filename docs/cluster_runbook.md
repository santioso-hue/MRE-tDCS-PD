# MAIA cluster runbook — full cohort (12 PD + 17 HC)

Execute on the MAIA workspace once SSH access is set up. The per-subject forward pipeline was validated
on a single HC pilot (kept locally, untracked); this scales it to the cohort and adds the recon-all
parcellation, the four montages, and the group statistics. Do NOT commit/push from the cluster — the user handles git.

## Decisions to lock BEFORE the batch
- **FreeSurfer version:** match Olsson's 7.2 if available; else document 8.2.0 (`analysis/build_rois.py`
  is version-agnostic, but the parcellation differs slightly between versions).
- **CIT168 label indices:** confirm the CIT168/Pauli 4D volume indices in `analysis/07_build_tier3_nuclei.sh`
  against the exact CIT168/Pauli release it warps (the integer indices are release-specific).
- **MRE confidence threshold:** set `05_mre_efield_comparison.py --conf-thresh` to Olsson's cutoff
  (default is the in-brain 10th percentile).
- HD montage geometry: locked (M1 ring C3+Cz/F3/T7/P3; DLPFC ring F3+Fp1/Fz/C3/F7).

## Phase 0 — environment + data availability (once)
1. Clone the repo; `cp config/config.example.sh config/config.sh` and set MAIA paths + `SUBJECT`.
   Recreate `CLAUDE.md` + `state.md` (git-excluded, do not travel with the clone).
2. Verify the toolchain: SimNIBS 4.x, FSL, FreeSurfer (+ `segmentBS.sh`/`-brainstem-structures`), ANTs,
   MATLAB **with** the md-dmri toolbox, and the `neuro` conda env. Note module-load commands for the
   job scripts.
3. **(item J) Data availability check across all 29:** for each subject confirm T1, T2, the single-shell
   `sDTI_opt_80` (`.nii.gz/.bval/.bvec` — the DTI arm needs it per subject), the MD-dMRI `fit/` inputs,
   and the MRE maps (stiffness/storage/loss/confidence). Tabulate gaps before launching anything.
4. Build `config/cohort.json` from `config/cohort.example.json`: `{id, group(PD/HC), age, work, m2m, fit}`
   for all 29, montages `["M1","DLPFC","HD_M1","HD_DLPFC"]`.

## Phase 1 — extended pilots (item I.13): 2–3 REAL subjects, >=1 PD and >=1 HC
Run the full Phase-2 chain on 2–3 subjects with **manual QC at every stage** (FSLeyes/freeview via the
Remote Desktop). PD brains (atrophy, enlarged ventricles, iron) break segmentation/registration in ways
a healthy-control pilot will not. Gate the full batch on these passing. Check especially: charm tissue seg,
S0-affine dMRI->T1 alignment, recon-all parcellation in atrophied cortex, and sim |E| in the physiological band.

## Phase 2 — per-subject forward pipeline (SLURM array over the 29)
Dependency-ordered. **charm and recon-all are independent (both from T1) — run in parallel; recon-all is
the long pole, so launch it first.**

| # | Stage | Script | Needs | ~Runtime |
|---|---|---|---|---|
| 1 | Head model | `pipeline/00_charm.sh` | T1, T2 | 1–2 h |
| 2 | Parcellation | `recon-all -all` then `segmentBS.sh` | T1 | **6–12 h** (long pole) |
| 3 | DTI conductivity | `pipeline/01_dwi2cond.sh` | m2m (1) | ~30 min |
| 4 | QTI covariance fit | `pipeline/run_qti_cov_cohort.m` (via run_qti_cov_cohort.sh) | fit/ inputs | 10–30 min |
| 5 | Build ⟨D⟩ maps | `pipeline/prepare_dmri_tensor.py` | (4) | fast (invoked automatically by stage 6, step 1) |
| 6 | Register dMRI→T1 | `pipeline/02_register_dmri_to_T1.sh` | m2m (1), (5) | S0-driven 12-DOF affine, ~1 min; emits s0_T1/FA_T1 (QC), MD_T1/uFA_T1, lam{1,2,3}_T1, tensor_triaxial_T1, v1_T1, dMRI_mask_T1 |
| 7 | Conductivity tensor | `pipeline/03_build_conductivity_tensor.py` | (6) | fast |
| 8 | recon-all ROIs | `analysis/build_rois.py --fs_dir …` | recon-all (2), m2m (1) | ~5 min |
| 9 | CIT168 nuclei | `analysis/07_build_tier3_nuclei.sh` (ANTs) | T1 | 30–60 min; merge 9 nuclei into (8) |
| 10 | MRE→T1 | `pipeline/05_register_mre_to_T1.sh` | m2m (1), MRE maps | ~10 min |
| 11 | Simulations | `pipeline/04_run_simulations.py` | m2m (1), tensors (3,7) | ~1 h (4 montages × 3 models) |
| 12 | ROI E-field | `analysis/04_extract_roi_efield.py --montage {M1,DLPFC,HD_M1,HD_DLPFC}` | sims (11), ROIs (8/9) | fast → `results/<id>/roi_efield_<montage>.csv` |
| 13 | MRE comparison | `analysis/05_mre_efield_comparison.py` | sims, ROIs, MRE_T1 (10) | fast → `results/<id>/mre_efield_per_roi.csv` |

Ordering within a subject: {1,2} parallel → {3,4,5,6,7} and {8,9} and {10} → 11 → {12,13}.
Parallelize across subjects with a SLURM array (one array task per subject); recon-all (2) can be its own
earlier array so its 6–12 h overlaps everything else.

## Phase 3 — QC (item I.12), after 5–8 subjects are through
`simnibs_python analysis/qc_harness.py --cohort config/cohort.json --calibrate` — recalibrates thresholds
on cohort percentiles (MAD>3 cohort-outlier flag activates at n>=4). Review every FLAG before trusting
the batch. Repoint qc_harness to the recon-all ROIs (it reads `_rois.py`, already prefers freesurfer_rois).
Add a per-arm mixed-sign-rate-per-ROI check here (derivation-doc note) so atrophied subjects don't
silently take the SimNIBS prolate branch.

## Phase 4 — cohort statistics (item B), after the batch (or rolling)
```
conda run -n neuro python analysis/06_cohort_stats.py --stat p95     # dosimetry headline (Huang)
conda run -n neuro python analysis/06_cohort_stats.py --stat median  # MRE-convention sensitivity
```
Per ROI per montage → `results/cohort_stats_<montage>_<stat>.csv`:
- **H1 (primary, powered):** paired MD-dMRI vs DTI and vs ISO (Wilcoxon signed-rank + rank-biserial).
- **H3 (exploratory):** PD vs HC, age-adjusted residuals → Mann-Whitney U + Cohen's d (Olsson's method).
- Age partial Pearson controlling for group; Benjamini-Hochberg FDR per (montage, family).
Frame H1 as the powered primary; H3 + clinical correlations as exploratory.

## Validation already in place (run as smoke tests on the cluster checkout)
- `conda run -n neuro python tests/test_cohort_stats.py`   (stats engine, synthetic)
- `conda run -n neuro python tests/test_lobe_grouping.py`  (recon-all label schemes)
- `conda run -n neuro python tests/validate_mean_tensor.py` (per subject after step 6)

## Not adopted (record, do not re-run)
Free-water elimination (4th arm) was built, simulated, and found null under 'vn' (see
`MRE_archive/fwe_experiment/` on the dev machine). The model is the three-arm ISO / DTI / MD-dMRI only.
