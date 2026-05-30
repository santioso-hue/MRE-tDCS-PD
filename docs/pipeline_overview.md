# Pipeline Overview — MD-dMRI–Informed tDCS Simulation

## Goal

Compare three models of white-matter conductivity for tDCS electric-field simulation
in Parkinson's disease (PD) patients:

| Model | Conductivity source | Anisotropy |
|-------|-------------------|------------|
| **ISO** | Literature scalar values | None |
| **DTI** | FA-weighted tensors from conventional DWI (`dwi2cond`) | FA-based |
| **MD-dMRI** | μFA-weighted tensors from Q-space trajectory imaging (QTI) | μFA-based |

The MD-dMRI model is the novel contribution: it uses the *microscopic* fractional
anisotropy (μFA) from the DPS fit of QTI data to build diffusion tensors that
reflect intra-voxel fibre architecture beyond what conventional DTI captures.

---

## Data Requirements

Per subject:
- **T1w** — 1 mm isotropic, reoriented to standard space
- **T2w** — 1 mm isotropic, reoriented to standard space (improves skull segmentation)
- **DWI (DTI)** — Multi-shell with b=0 and ≥1 b>0 shell, bvec/bval files
- **QTI acquisition** — Spherical dMRI (LTE+STE+PTE) producing `dps.mat` from QTI+ pipeline:
  - `dps.mat` — MATLAB struct containing: `ufa` (μFA), `MD` (mean diffusivity), `u` (principal eigenvectors), `mask`

---

## Pipeline Steps

### Step 0a — Head model (CHARM)
```bash
bash pipeline/00_charm.sh <SUBID> T1w_1mm.nii T2w_1mm.nii <subject_dir>
```
- Builds 5-tissue tetrahedral FEM mesh in `m2m_<SUBID>/`
- Settings in `config/charm_highquality.ini`
- Runtime: ~1 hour on Apple M-series

### Step 0b — DTI preprocessing
```bash
bash pipeline/00_dwi2cond.sh
```
- Runs `dtifit --save_tensor` on the DWI series
- Registers DTI tensor to T1 space via FLIRT + `vecreg`
- Copies `DTI_coregT1_tensor.nii.gz` to `m2m_<SUBID>/` for SimNIBS auto-detection
- **Note (FullPD5 pilot):** `dwi2cond` crashes due to a single b=0 volume edge case
  in the eddy current wrapper. Bypassed by running `dtifit --save_tensor` directly,
  then FLIRT + vecreg manually. This skips eddy current correction — acceptable for
  pilot but should be revisited for the full study.

### Step 1 — Register MD-dMRI outputs to T1 space
```bash
bash pipeline/01_register_dMRI_to_T1.sh
```
Sub-steps:
1. Extract b=0 from spherical dMRI
2. FLIRT 6-DOF rigid registration: b0 → T1 (`dMRI_to_T1.mat`)
3. Extract clean μFA (ufa) and MD from `dps.mat` — **NaN-free** (01c)
4. Apply transform to μFA, MD, and brain mask
5. Save principal eigenvectors from `dps.mat` as NIfTI (01b)
6. `vecreg` eigenvectors to T1 space (rotation-corrected)

**Why dps.mat instead of dtd_covariance_C_mu.nii.gz?**
The issue is NOT NaN values outside the mask — those are zero and safe for
interpolation. The actual failure is within the brain mask: 24.3% of brain voxels
contain non-physical extreme values (range −23,653 to +13,774; mean = 26.0),
caused by covariance model instability from sparse QTI sampling (38 volumes, pilot
protocol). After clipping to [0, 1], Pearson r between the covariance estimate and
`dps.ufa` is only 0.26 — confirming they produce different estimates. This is a
data-insufficiency problem, not fixable by masking. `dps.ufa` is the only
physically valid μFA estimate for this pilot dataset.

### Step 2 — Build conductivity tensor
```bash
simnibs_python pipeline/02_build_conductivity_tensor.py
```
- Computes prolate eigenvalues from QTI formula:
  - λ₁ = MD × (1 + 2·μFA / √(3 − 2·μFA²))
  - λ₂ = λ₃ = MD × (1 − μFA / √(3 − 2·μFA²))
- Builds diffusion tensor: D = (λ₁−λ₂)·v₁v₁ᵀ + λ₂·I
- Output: `tensor_MD_dMRI.nii.gz` in FSL `dtifit --save_tensor` format
  `[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]` (upper triangle, row-major)

### Step 3 — Run simulations
```bash
simnibs_python pipeline/03_run_simulations.py
```
- Runs all three models: ISO, DTI, MD-dMRI
- Montage: C3 (anode) → Fp2 (cathode), 2 mA, 5×5 cm pads
- Uses `anisotropy_type='vn'` (volume normalization) for DTI and MD-dMRI
- Outputs: `sim_ISO/`, `sim_DTI/`, `sim_MD_dMRI/` with `.msh` mesh files

### Step 4 — Regional E-field comparison (TODO)
```bash
simnibs_python analysis/04_extract_roi_efield.py
```
- Extract E-field from `.msh` in PD-relevant ROIs (SN, STN, CST, etc.)
- Compare ISO vs DTI vs MD-dMRI percentile distributions

---

## macOS Apple Silicon Notes

SimNIBS 4.6.0 requires these environment variables to avoid OpenMP crashes:
```bash
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```
Use `caffeinate -i` for any run > 15 min to prevent macOS sleep.

---

## Simulation Quality Metrics (FullPD5 pilot)

| Metric | ISO | DTI | MD-dMRI |
|--------|-----|-----|---------|
| GM E99.9 (V/m) | 0.648 | 0.672 | 0.672 |
| GM E99.0 (V/m) | 0.530 | 0.544 | 0.544 |
| GM E95.0 (V/m) | 0.432 | 0.433 | 0.433 |
| Focality 75% (mm³) | 11,300 | 9,850 | 9,850 |
| Focality 50% (mm³) | 108,000 | 88,400 | 88,400 |
| Solver calibration error | 2.4% | 2.4% | 2.4% |

DTI = MD-dMRI at global GM percentiles — expected. Regional differences (the key
result) require `04_extract_roi_efield.py`.

Registration QC (FullPD5 pilot):
- dMRI→T1: Rx=4.6°, Ry=−0.7°, Rz=3.0°, Tz=79.7 mm — anatomically reasonable
- DTI→T1:  Rx=6.2°, Ry=−0.3°, Rz=2.6°, Tz=76.5 mm — consistent
- Both transforms: |det(R)| = 1.000000, R^T·R = I — perfect rigid bodies
- **PENDING: Visual QC in FSLeyes** — b0_spherical_T1 overlaid on m2m_FullPD5/T1.nii.gz

---

## Known Limitations

### Eddy Current Correction (DTI)
Cannot be applied because only 1 b=0 volume was acquired in the DTI series
(series 801: 1 b=0 + 80 b=1500, MB3) and no reversed phase-encoding b=0 was
collected to enable topup susceptibility correction. Motion correction only could
be added via `mcflirt`. The full study acquisition protocol should include a
reversed-PE b=0 pair to enable both topup and eddy correction via `eddy_openmp`.

### Prolate Approximation Error
DTI data for FullPD5 shows λ₂/λ₃ mean = 1.32 (24% fractional asymmetry), meaning
the prolate approximation (λ₂ = λ₃) introduces ~12% error in perpendicular
conductivity per voxel on average. Can be addressed via a hybrid model: set λ₁
from the QTI μFA formula, then split λ₂/λ₃ from DTI eigenvector decomposition,
combining all three eigenvectors. Not implemented for this pilot.

---

## Key Scientific Decisions & Open Questions

1. **DPS μFA vs. covariance μFA**: We use `dps.ufa` (DPS estimator) instead of
   `dtd_covariance_C_mu.nii.gz` (covariance estimator). The covariance estimator
   is unstable for this pilot's sparse QTI sampling (38 volumes): 24.3% of brain
   voxels contain non-physical values; Pearson r with `dps.ufa` = 0.26 after
   clipping to [0, 1]. `dps.ufa` is the only physically valid μFA estimate.
   Methods must specify the DPS estimator explicitly.

2. **Prolate approximation**: Only the principal eigenvector v₁ is available from
   `dps.mat` (the full mean diffusion tensor components are zero in this pilot).
   The prolate approximation (λ₂=λ₃) assumes axial symmetry around v₁. DTI data
   shows λ₂/λ₃ mean = 1.32 (24% fractional asymmetry), giving ~12% error in
   perpendicular conductivity per voxel on average. See Known Limitations for the
   hybrid-model fix.

3. **Eddy current correction skipped (DTI)**: Cannot be applied — only 1 b=0
   volume acquired, no reversed-PE b=0 for topup. Motion correction only could be
   added via `mcflirt`. See Known Limitations.

4. **Unit consistency**: MD-dMRI tensor values are in μm²/ms; DTI tensor (from
   dtifit) is in mm²/s. SimNIBS `vn` normalization is scale-invariant so this
   does not affect results — but document clearly in methods.
