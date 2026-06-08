# MD-dMRI–informed tDCS electric-field modelling

A subject-specific pipeline for transcranial direct-current stimulation (tDCS) electric-field
simulation, where white-matter conductivity tensors are derived from b-tensor-encoded
multidimensional diffusion MRI (MD-dMRI / QTI) rather than conventional single-shell DTI. It
targets deep subcortical structures (substantia nigra, STN, globus pallidus) relevant to
Parkinson's disease.

## What it does

The pipeline builds a finite-element head model, derives an anisotropic conductivity tensor in
each white-/grey-matter voxel, runs the tDCS simulation in SimNIBS, and compares the induced
electric field across three conductivity models:

| Model | Conductivity source |
|-------|---------------------|
| **ISO** | Scalar literature conductivities (no anisotropy) |
| **DTI** | Diffusion tensor from single-shell DTI (`dwi2cond`) |
| **MD-dMRI** | Mean diffusion tensor ⟨D⟩ from QTI, with free-water elimination |

The conductivity mapping (σ ∝ D, volume-normalised) is identical across the anisotropic
models; they differ only in the diffusion tensor they start from. The MD-dMRI tensor is less
biased by diffusional kurtosis and CSF partial volume than the DTI tensor. See
[pipeline/conductivity_models_derivation.md](pipeline/conductivity_models_derivation.md) for
the theory, the three models, and limitations.

## Setup

1. Install [FSL](https://fsl.fmrib.ox.ac.uk/) (≥ 6.0) and
   [SimNIBS](https://simnibs.github.io/simnibs/) (4.6).
2. Copy the config template and edit it for your machine and subject:

   ```bash
   cp config/config.example.sh config/config.sh
   # edit config/config.sh: SUBJECT, DATA_DIR, WORK_DIR, input file names, tool paths
   ```

   `config/config.sh` is gitignored — paths and subject IDs stay out of version control. The
   bash scripts source it; the Python scripts read it through `pipeline/_config.py`.

## Running

Scripts run in order; each reads its paths from the config.

```bash
bash    pipeline/00_charm.sh "$SUBJECT" T1.nii T2.nii "$WORK_DIR"   # FEM head model
bash    pipeline/00_dwi2cond.sh                                      # DTI tensor (dwi2cond)
bash    pipeline/01_register_dMRI_to_T1.sh                           # QTI maps/tensors -> T1
simnibs_python pipeline/02_build_conductivity_tensor.py              # MD-dMRI tensor: free-water-eliminated <D>
simnibs_python pipeline/03_run_simulations.py                        # ISO + DTI + MD-dMRI FEM
simnibs_python analysis/04_extract_roi_efield.py                     # ROI E-field comparison
```

## Layout

```
config/      config.example.sh (template), charm_highquality.ini
pipeline/    00-03 pipeline scripts + _config.py (shared config loader)
             conductivity_models_derivation.md  (methods)
analysis/    04_extract_roi_efield.py + results/
docs/        references
```

## Notes

- On macOS/Apple Silicon the scripts set `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE`
  for SimNIBS.
- Subject data (NIfTIs, meshes, `.mat` fits, working directories) are gitignored; the
  repository contains code and documentation only.
- The models are motivated by bias arguments and validated against each other; direct
  validation against MR current-density imaging (MRCDI/MREIT) is future work.
