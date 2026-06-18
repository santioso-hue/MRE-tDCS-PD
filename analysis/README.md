# analysis/ — from |E| to a reported number

Reading guide for a reviewer: *can I trust that a number in a results table came from this code?*
This folder turns the SimNIBS sims into the per-ROI numbers and figures. The per-subject pipeline that
builds the head model and runs the sims is in `pipeline/` (see `docs/cluster_runbook.md` for the full
charm -> sims chain); this README covers the analysis half.

## DAG (consumes -> emits)

The SimNIBS work meshes (and, for 08, the DTI/<D> tensors) are the shared upstream. 04, 05, and 08 each
read them DIRECTLY -- 05 and 08 do NOT consume 04's CSV; 06 consumes ONLY 04's CSV.

```
pipeline/04_run_simulations.py         ->  work/sim_<montage>_<model>/      (FEM solve; magnE per element)
pipeline/03_build_conductivity_tensor  ->  work/tensor_MD_dMRI.nii.gz       (<D>, um2/ms)
pipeline/01_dwi2cond (gated)           ->  m2m/DTI_coregT1_tensor.nii.gz    (DTI, gated on ParkMRE_DTI)
        |
        |  each reader below loads the meshes/tensors directly (not 04's CSV):
        |
        +--> analysis/04_extract_roi_efield.py  ->  results/<subj>/roi_efield_<montage>.csv
        |        per-ROI |E| median + p95, per model ISO/DTI/MD-dMRI
        |            |
        |            v
        |        analysis/06_cohort_stats.py  ->  aggregates roi_efield_<montage>.csv across the cohort
        |            H1: within-subject paired ISO/DTI/MD-dMRI.  H3: PD-vs-HC, age-adjusted (gated).
        |
        +--> analysis/08_tensor_divergence.py  ->  results/<subj>/tensor_divergence.csv
        |        per ROI: DTI-vs-<D> principal-direction angle, dFA, d(lam1/lam3), d(lam2/lam3), and a
        |        trivial-vs-nontrivial verdict. (Needs the DTI tensor; gated on ParkMRE_DTI.)
        |            |
        |            v  (08 -> 05: 05 also reads tensor_divergence.csv when present)
        +--> analysis/05_mre_efield_comparison.py  ->  results/<subj>/mre_efield_per_roi.csv
                 per ROI: MRE stiffness/alpha, MD, uFA, FA(<D>), cond anisotropy, |E| per model, and
                 dE_model_pct = 100*(E_MD-dMRI - E_DTI)/E_DTI; plus the across-ROI consistency block and the
                 FA/uFA-divergence explanation (n=1 across-ROI trend per subject, not a statistic).
```

## Supporting scripts
- `qc_harness.py` — per-subject QC across mesh / conductivity / registration / E-field. The p95-range and
  spike checks are SANITY GATES (flag an artifactual sim), not reported numbers. `--emit-metrics` dumps the
  metric snapshot. This is the single validation authority.
- `qc_figures.py` — figures only: |E| overlay PNGs, the MD-dMRI-minus-DTI difference PNG, and T1-space
  magnE NIfTIs. No stats of record.
- `07_build_tier3_nuclei.sh` + `_build_tier3_labels.py` — CIT168 SNc/SNr/VTA/RN/STN masks. EXPLORATORY,
  E-field-only, overlap-allowed; not headline numbers. Every tier-3 magic number is in `07`'s config block.
  The CIT168 atlas cache it consumes (`_atlas_cache/`: the 2009c->NLin6 affine + `pauli_prob`) is a
  cluster/manual artifact staged into `registration/atlas_rois/`, not produced by any in-repo script.
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
