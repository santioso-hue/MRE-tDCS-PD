# analysis/ — from |E| to a reported number

Reading guide for a reviewer: *can I trust that a number in a results table came from this code?*
This folder turns the SimNIBS sims into the per-ROI numbers and figures. The per-subject pipeline that
builds the head model and runs the sims is in `pipeline/` (see `docs/cluster_runbook.md` for the full
charm -> sims chain); this README covers the analysis half.

## DAG (consumes -> emits)

```
pipeline/04_run_simulations.py  ->  work/sim_<montage>_<model>/   (FEM solve; magnE per element)
        |
        v
analysis/04_extract_roi_efield.py  ->  results/<subj>/roi_efield_<montage>.csv
        |                                (per-ROI |E| median + p95, per model ISO/DTI/MD-dMRI)
        |
        +--> analysis/05_mre_efield_comparison.py  ->  results/<subj>/mre_efield_per_roi.csv
        |       per ROI: MRE stiffness/alpha, MD, uFA, FA(<D>), cond anisotropy, |E| per model, and
        |       dE = E_MD-dMRI - E_DTI; plus the across-ROI consistency block and the FA/uFA-divergence
        |       explanation (n=1 across-ROI trend per subject, not a statistic).
        |
        +--> analysis/08_tensor_divergence.py  ->  results/<subj>/tensor_divergence.csv
        |       per ROI: DTI-vs-<D> principal-direction angle, dFA, d(lam1/lam3), d(lam2/lam3),
        |       and a trivial-vs-nontrivial verdict. (Needs the DTI tensor; gated on ParkMRE_DTI.)
        |
        v
analysis/06_cohort_stats.py  ->  aggregates the per-subject CSVs across the cohort
        H1: within-subject paired ISO/DTI/MD-dMRI contrast.   H3: PD-vs-HC, age-adjusted (gated on demographics).
```

## Supporting scripts
- `qc_harness.py` — per-subject QC across mesh / conductivity / registration / E-field. The p95-range and
  spike checks are SANITY GATES (flag an artifactual sim), not reported numbers. `--emit-metrics` dumps the
  metric snapshot. This is the single validation authority.
- `qc_figures.py` — figures only: |E| overlay PNGs, the MD-dMRI-minus-DTI difference PNG, and T1-space
  magnE NIfTIs. No stats of record.
- `07_build_tier3_nuclei.sh` + `_build_tier3_labels.py` — CIT168 SNc/SNr/VTA/RN/STN masks. EXPLORATORY,
  E-field-only, overlap-allowed; not headline numbers. Every tier-3 magic number is in `07`'s config block.
- `build_rois.py` — recon-all parcellation -> Tier-1/2 ROI masks in mesh space (`registration/freesurfer_rois/`).
- `_rois.py`, `_sims.py` — shared helpers (ROI loading/sampling; montage-aware mesh lookup). One copy, imported.

## Unit-tested (the credibility signal)
Label assembly (`tests/test_lobe_grouping.py`), the cohort stats engine (`tests/test_cohort_stats.py`), the
QC metric extraction including the vn/degeneracy guard (`tests/test_qc_harness.py`), and the
tensor-divergence math (`tests/test_tensor_divergence.py`) are unit-tested. Each runs as a script, e.g.
`simnibs_python tests/test_qc_harness.py`.

## Framing
The MD-dMRI arm uses a MORE PRINCIPLED tensor (kurtosis-aware multi-shell QTI <D>) than the single-shell
Gaussian DTI tensor. The analysis QUANTIFIES where the two diverge; it does not assert <D> is "more
accurate" or "more anisotropic".
