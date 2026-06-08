# Anisotropic conductivity for tDCS simulation: the MD-dMRI model

Three finite-element tDCS simulations are compared per subject. They share the same head model,
electrodes, current, solver, and (for the two anisotropic models) the same SimNIBS conductivity
mapping. **Only the input changes**, which is what makes this a controlled comparison:

| Model | Conductivity input | SimNIBS setting |
|---|---|---|
| **ISO** | isotropic literature values | `anisotropy_type='scalar'` |
| **DTI** | single-shell DTI tensor (dwi2cond) | `'vn'` |
| **MD-dMRI** | QTI mean tensor ⟨D⟩ | `'vn'` |

The DTI model is the SimNIBS-standard anisotropy baseline. The **MD-dMRI model is the contribution:
the same Tuch effective-medium / volume-normalized mapping, but fed the mean diffusion tensor ⟨D⟩
from b-tensor-encoded MD-dMRI (QTI) instead of the single-shell DTI tensor.** Nothing downstream of
the input tensor departs from standard SimNIBS.

## From a diffusion tensor to a conductivity tensor

A diffusion tensor `D` is a symmetric positive-definite 3×3 matrix. Its eigen-decomposition

    D = λ1 v1 v1ᵀ + λ2 v2 v2ᵀ + λ3 v3 v3ᵀ ,   λ1 ≥ λ2 ≥ λ3 > 0

gives three orthogonal eigenvectors v_k (the principal diffusion directions) and eigenvalues λ_k
(diffusivities along them). The mean diffusivity is MD = (λ1+λ2+λ3)/3 = trace(D)/3.

Tuch et al. (2001) argued that in tissue the same microgeometry (cell membranes, axon walls)
obstructs both water self-diffusion and ionic conduction, so the conductivity tensor σ and the
diffusion tensor D **share eigenvectors** and have **linearly related eigenvalues**. There is no
such link in free fluid; it exists only because both transport processes respect the same tissue
boundaries. We use the eigenvector-sharing and proportionality, σ ∝ D.

SimNIBS applies this with `anisotropy_type='vn'` (volume normalization; Güllmar 2010, Rullmann 2009):

    σ(x) = σ0(x) · D(x) / det[D(x)]^(1/3)

`det(D)^(1/3)` is the geometric mean of the eigenvalues. Dividing by it makes the three conductivity
eigenvalues have geometric mean 1; multiplying by the tissue's literature isotropic conductivity σ0
sets the scale. So `'vn'` keeps only the **shape** of D (its anisotropy ratios and orientation) and
pins the per-tissue geometric-mean conductivity to σ0 (WM 0.126, GM 0.275 S/m). Two consequences,
both deliberate:
- The mapping is robust to the absolute magnitude of D, which is the least reliable part of a 2.5 mm
  diffusion measurement; magnitude is replaced by the trusted literature σ0.
- It is the proportional / "volume-constrained" variant (Hallez 2009; what SimNIBS implements), which
  approximates Tuch's affine eigenvalue relation σ_v = k(d_v − d_e) and slightly understates its
  anisotropy. This is identical for the DTI and MD-dMRI models, so it does not affect their contrast.

SimNIBS bounds the tensor with two parameters, left at their **defaults** and identical for both
anisotropic models: `aniso_maxratio = 10` (caps the eigenvalue ratio at 10:1; the binding one; top of
the 7–10:1 ex-vivo WM range, Nicholson 1965 / Ranck & BeMent 1965; clips ~0.1% of DTI and ~0.7% of
⟨D⟩ voxels) and `aniso_maxcond = 2` S/m (caps eigenvalue magnitude; non-binding under `'vn'`). The
tensor is supplied as a 6-component NIfTI in FSL order `[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]`.

## DTI model (baseline)

A separate single-shell acquisition (`sDTI_opt_80`, 80 directions, b≈1500, same subject and session)
is fitted with FSL `dtifit` and registered to T1 with `dwi2cond --all` (eddy correction, nonlinear
FA→T1 registration, vecreg tensor reorientation) — the standard SimNIBS anisotropy path. Its
limitation is intrinsic to single-shell DTI: the mono-exponential fit averages over all fibre
orientations in a voxel and is biased by non-Gaussian diffusion (kurtosis), so crossing fibres and
dispersion depress the measured anisotropy.

## MD-dMRI model (the contribution)

b-tensor-encoded MD-dMRI (linear + spherical encoding; the QTI framework of Westin et al. 2016,
Topgaard 2017) is fitted with the `md-dmri` toolbox. The fit yields the **mean diffusion tensor ⟨D⟩**
— the first cumulant (mean) of the intra-voxel diffusion-tensor distribution — stored in `dps.mat`
(fields `mdxx..mdyz`, SI units). ⟨D⟩ is the *same macroscopic quantity* DTI estimates, with
MD = trace(⟨D⟩)/3, but estimated within a model that places the non-Gaussian variance in a separate
covariance term rather than letting it bias the mean. So ⟨D⟩ is a **less kurtosis-biased** estimate
of the macroscopic mean tensor than the single-shell DTI tensor. (It is *not* free-water corrected:
like the DTI tensor it still contains any CSF partial volume; MD is high in the ventricles.)

The full triaxial ⟨D⟩ is reconstructed from the six `md??` fields and used directly: σ ∝ ⟨D⟩. It is
fully triaxial (three distinct eigenvalues in ~93% of brain voxels), with principal eigenvector equal
to the fit's `u`, trace 3·MD, and extreme eigenvalues ad/rd. `tests/validate_mean_tensor.py`
reproduces these identities from `dps.mat`.

The principal axis (`dps.u`) agrees with the *independent* single-shell DTI V1 only moderately —
~22° median in core WM (FA>0.5), ~30° across all WM. So the DTI↔MD-dMRI E-field contrast reflects
**both** the eigenvalue (tensor-estimation) difference **and** a ~22–30° orientation difference from
the separate DTI acquisition; it is not a magnitude-only comparison. Empirically the two
conductivities deviate ~5% (median) to ~17% (p95) in white matter, concentrated where the DTI
mono-exponential assumption fails, consistent with ⟨D⟩ being the less-biased estimate.

## Registration

The diffusion tensors live on the 2.5 mm dMRI grid and must be brought to the 1 mm structural/mesh
grid, which requires *reorienting* the tensor, not just resampling its components. We decompose it:
the three eigenvalues are interpolated **independently** as scalar maps (FLIRT trilinear), and the
orientation frame is carried by `vecreg` (preservation-of-principal-direction reorientation,
Alexander 2001), with the principal axis anchored to `dps.u` (`v1_T1`). After interpolation the
eigenvalue maps are re-sorted to λ1≥λ2≥λ3 and paired with the frame **by magnitude order**, so a
boundary voxel where two maps cross cannot mis-assign eigenvalue to eigenvector.

Whole-tensor component-wise interpolation, by contrast, averages neighbouring tensors and shrinks the
anisotropy (median λ1/λ3 1.92 → 1.28 here) — the tensor "swelling" that log-Euclidean interpolation
(Arsigny 2006) and PPD reorientation are designed to avoid. Our scheme is a pragmatic stand-in for
full log-Euclidean tensor interpolation (e.g. DTI-TK): it preserves the native anisotropy at the cost
of not reproducing the partial-volume blurring the 2.5 mm resolution incurs — a disclosed trade-off,
and the one step where the MD-dMRI pipeline is less standardized than dwi2cond's fnirt+vecreg path.

## ROI definition

E-field and microstructure are read out over the FastSurfer-derived ROI masks in mesh space
(`analysis/build_rois.py` -> `registration/fastsurfer_rois/`, loaded by the analysis scripts through
`analysis/_rois.py`): cortical and white-matter lobes (DKT grouped to frontal/parietal/temporal/
occipital), corpus callosum, and the aseg subcortical structures (thalamus, caudate, putamen,
pallidum, accumbens, hippocampus, amygdala, brainstem), plus a whole-brain GM+WM mask. FastSurfer is
run seg-only (deep-learning, license-free); its conformed segmentation is registered to the charm T1
(FLIRT 6-DOF rigid) so the masks land in the FEM/E-field space. The fine midbrain nuclei
(SNc/SNr/VTA/RN/STN), below aseg resolution, come from the CIT168/Pauli 2017 atlas warped via ANTs
(`analysis/07_build_tier3_nuclei.sh`) and merge into the ROI set when needed. Each ROI is sampled over
every GM/WM element it contains, not a fixed-radius sphere.

## Limitations

- **Resolution.** MD-dMRI is acquired at 2.5 mm; small deep nuclei (SN, STN) are near or below a
  voxel, so partial volume biases the per-voxel tensor at exactly the PD targets of interest.
- **Magnitude is discarded by `'vn'`.** Conductivity magnitude is set by literature σ0, not by the
  measured ⟨D⟩. This is deliberate (per-voxel diffusion magnitude at 2.5 mm is unreliable: within-WM
  geometric-mean CoV ≈ 46%, mostly partial volume and noise), but it means disease-related MD changes
  enter only through anisotropy/orientation, not magnitude.
- **Validation.** No in-vivo conductivity ground truth exists; the model is motivated by bias
  arguments, not direct validation. MR current-density imaging (MRCDI/MREIT; Gregersen 2024) is the
  planned validation route.
- **DTI baseline.** The DTI model is a separate single-shell acquisition registered independently, so
  its principal direction differs from ⟨D⟩ by ~22–30°; the contrast is not a same-data comparison of
  magnitude alone.

## References

Tuch et al. (2001) PNAS — conductivity ∝ diffusion tensor (effective medium).
Güllmar et al. (2010) NeuroImage 51 — volume-normalized mapping (Eq. 3–4).
Rullmann et al. (2009) NeuroImage 44 — volume-constrained anisotropic FEM head model.
Westin et al. (2016) NeuroImage 135 — QTI; ⟨D⟩ + covariance; MD = tr⟨D⟩/3 (Eq. 14).
Topgaard (2017) J Magn Reson 275 — diffusion tensor distributions; b-tensor signal model (Eq. 40).
Lampinen et al. (2017) NeuroImage 147 — microscopic anisotropy needs variable b-tensor shape; d_FW = 3.0 µm²/ms.
Nicholson (1965) / Ranck & BeMent (1965) Exp Neurol — ex-vivo WM conductivity anisotropy 7–10:1.
Alexander et al. (2001) IEEE TMI — preservation-of-principal-direction reorientation.
Arsigny et al. (2006) MRM — log-Euclidean tensor interpolation.
Gregersen et al. (2024) Imaging Neuroscience — MRCDI for head-model validation.

## Appendix — alternatives evaluated and not used

Free-water elimination (σ ∝ ⟨D⟩_tissue), magnitude preservation (`'dir'` / per-tissue), and μFA /
covariance were each tested and rejected. In brief: FWE is ~null under `'vn'` and ill-posed at this
volume count; magnitude preservation injects
~46%-CoV partial-volume/noise (and SimNIBS `'dir'` also distorts the GM/WM contrast); μFA and the
covariance tensor are microscopic measures with no macroscopic eigenframe, so they cannot define a
conductivity tensor (they remain useful for interpretation).
