# MRE-tDCS-PD: MD-dMRI–Informed tDCS Simulation in Parkinson's Disease

Computational pipeline for subject-specific tDCS electric-field modelling using
diffusion MRI–derived conductivity tensors, applied to Parkinson's disease patients.

## Key Idea

Standard tDCS simulations assume isotropic tissue conductivity or use FA-based
anisotropy from conventional DTI. This project introduces a third model driven by
**microscopic fractional anisotropy (μFA)** estimated from Q-space trajectory imaging
(QTI / MD-dMRI). μFA captures intra-voxel fibre architecture that is invisible to
conventional DTI, potentially improving conductivity estimates in regions of fibre
crossings and dispersion — common in PD-relevant white matter tracts.

## Three-Model Comparison

| Model | Conductivity | Anisotropy driver |
|-------|-------------|-------------------|
| **ISO** | Scalar literature values | None |
| **DTI** | FA-based tensors via `dwi2cond` | Conventional DWI |
| **MD-dMRI** | μFA-based tensors from QTI | Spherical dMRI (QTI+) |

## Repository Structure

```
pipeline/           Pipeline scripts (00–03), run in order
  00_charm.sh         Head model (SimNIBS charm, T1+T2)
  00_dwi2cond.sh      DTI preprocessing (dtifit + vecreg)
  01_register_dMRI_to_T1.sh  Register QTI outputs to T1 space
  01b_save_v1_nifti.py       Save principal eigenvectors from dps.mat
  01c_save_dps_niftis.py     Save μFA and MD from dps.mat (NaN-free)
  02_build_conductivity_tensor.py  Build 6-component diffusion tensor
  03_run_simulations.py      Run ISO + DTI + MD-dMRI simulations

config/
  charm_highquality.ini   CHARM settings (T1+T2, macOS Apple Silicon)

analysis/
  04_extract_roi_efield.py   Regional E-field comparison (TODO)

docs/
  pipeline_overview.md    Detailed pipeline documentation, QC notes, open questions
```

## Requirements

- [FSL](https://fsl.fmrib.ox.ac.uk/) ≥ 6.0 — installed at `~/fsl`
- [SimNIBS](https://simnibs.github.io/simnibs/) 4.6.0 — installed at `~/Applications/SimNIBS-4.6`
- Python packages (via SimNIBS environment): `nibabel`, `scipy`, `numpy`

**macOS Apple Silicon:** `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` are
required for SimNIBS. All scripts set these automatically.

## Status

- [x] Pilot: FullPD5 — head model, DTI tensor, MD-dMRI tensor, all 3 simulations ✓
- [ ] Visual QC: b0→T1 registration in FSLeyes
- [ ] Regional E-field comparison (Step 4)
- [ ] Additional PD subjects

See [docs/pipeline_overview.md](docs/pipeline_overview.md) for full details, QC
metrics, and known issues.
