# MD-dMRI–informed tDCS electric-field modelling

A subject-specific pipeline for transcranial direct-current stimulation (tDCS) electric-field
simulation in which white-matter conductivity tensors are derived from b-tensor-encoded
multidimensional diffusion MRI (MD-dMRI / QTI) rather than conventional single-shell DTI. It
targets deep subcortical structures (substantia nigra, STN, globus pallidus) relevant to
Parkinson's disease.

## What it does

The pipeline builds a finite-element head model, derives an anisotropic conductivity tensor in each
white-/grey-matter voxel, runs the tDCS simulation in SimNIBS, and compares the induced electric
field across three conductivity models:

| Model | Conductivity source |
|-------|---------------------|
| ISO | Scalar literature conductivities (no anisotropy) |
| DTI | Diffusion tensor from single-shell DTI (`dwi2cond`) |
| MD-dMRI | QTI covariance mean tensor ⟨D⟩ (`dtd_covariance`) |

The conductivity mapping (σ ∝ D, volume-normalised; SimNIBS `'vn'`) and everything downstream are
identical across the two anisotropic models; they differ only in the input diffusion tensor, which
makes ISO / DTI / MD-dMRI a controlled comparison. The QTI covariance mean tensor ⟨D⟩ is a less
kurtosis-biased estimate of the same macroscopic tensor than the single-shell DTI tensor. See
[pipeline/conductivity_models_derivation.md](pipeline/conductivity_models_derivation.md) for the
theory, the mapping, and limitations.

Regions of interest (cortical and white-matter lobes, corpus callosum, brainstem substructures,
subcortical nuclei) are derived from a [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) `recon-all`
parcellation (matching Olsson et al. 2025) and brought into the mesh space; the fine midbrain nuclei
(SNc/SNr/VTA/RN/STN) come from the CIT168/Pauli atlas. `recon-all` is run on the cluster when needed.

## Setup

1. Install [FSL](https://fsl.fmrib.ox.ac.uk/) (≥ 6.0) and [SimNIBS](https://simnibs.github.io/simnibs/)
   (4.6). ROIs are built from a [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) `recon-all`
   parcellation (`recon-all` + `-brainstem-structures`, run on the cluster).
2. Copy the config template and edit it for your machine and subject:

   ```bash
   cp config/config.example.sh config/config.sh
   # edit SUBJECT, DATA_DIR, WORK_DIR, input file names, tool paths
   ```

   `config/config.sh` is gitignored, so paths and subject IDs stay out of version control. The bash
   scripts source it; the Python scripts read it through `pipeline/_config.py` (the single source of
   truth for paths and the subject ID).

## Running

Scripts run in order; each reads its paths from the config.

```bash
bash           pipeline/00_charm.sh "$SUBJECT" T1.nii T2.nii "$WORK_DIR"  # FEM head model
bash           pipeline/01_dwi2cond.sh                                     # DTI baseline tensor (dwi2cond)
bash           pipeline/run_qti_cov_cohort.sh <subject_dir>                   # QTI covariance fit <D> (MATLAB via md-dmri)
bash           pipeline/02_register_dmri_to_T1.sh                          # QTI <D> -> T1 (mesh) space
simnibs_python pipeline/03_build_conductivity_tensor.py                    # MD-dMRI tensor (sigma ~ <D>)
simnibs_python pipeline/04_run_simulations.py                             # ISO + DTI + MD-dMRI FEM
simnibs_python analysis/build_rois.py --fs_dir <recon-all_subject_dir>             # ROI masks in mesh space (recon-all)
simnibs_python analysis/04_extract_roi_efield.py                          # per-ROI E-field table
```

Quality control and the post-hoc MRE comparison:

```bash
simnibs_python analysis/qc_harness.py            # per-subject QC across all stages
simnibs_python analysis/qc_figures.py            # |E| overlay + MD-dMRI-minus-DTI difference figures
bash           pipeline/05_register_mre_to_T1.sh # MRE maps -> T1 (post-hoc)
```

## Layout

```
config/    config.example.sh (template), cohort.example.json (cohort manifest)
pipeline/  00_charm, 01_dwi2cond, run_qti_cov_cohort.m + .sh (QTI covariance fit, MATLAB), 02_register_dmri_to_T1,
           03_build_conductivity_tensor, 04_run_simulations, 05_register_mre_to_T1,
           prepare_dmri_tensor.py (called by 02), _config.py, conductivity_models_derivation.md (methods)
analysis/  build_rois (recon-all ROIs), _rois (ROI resolver), extract_roi_efield,
           mre_efield_comparison, qc_harness (stats/QC), qc_figures (PNGs)
docs/      references
tests/     mean-tensor reconstruction + QC-harness checks
```

## Notes

- Subject data (NIfTIs, meshes, `.mat` fits, working directories) and derived outputs are gitignored;
  the repository contains code and documentation only.
- The models are motivated by bias arguments and compared against each other; direct validation against
  MR current-density imaging (MRCDI/MREIT) is future work.
