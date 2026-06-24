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
sets the scale. (The SimNIBS GUI and TDCSLIST documentation describe this as a *trace* normalization,
but the implemented code divides by the **geometric mean** `det(D)^(1/3)`, not the trace - verified in
SimNIBS 4.6 `cond_utils.cond2elmdata`; the "trace" wording is a documentation error.) So `'vn'` keeps
only the **shape** of D (its anisotropy ratios and orientation) and
pins the per-tissue geometric-mean conductivity to σ0 (WM 0.126, GM 0.275 S/m). Two consequences,
both deliberate:
- The mapping is robust to the absolute magnitude of D, which is the least reliable part of a 2.5 mm
  diffusion measurement; magnitude is replaced by the trusted literature σ0.
- It is the proportional / "volume-constrained" variant (Hallez 2009; what SimNIBS implements), which
  approximates Tuch's affine eigenvalue relation σ_v = k(d_v − d_e) and slightly understates its
  anisotropy. This is identical for the DTI and MD-dMRI models, so it does not affect their contrast.

SimNIBS bounds the tensor with two parameters, left at their **defaults** and identical for both
anisotropic models: `aniso_maxratio = 10` (caps the eigenvalue ratio at 10:1; the binding one; top of
the 7–10:1 ex-vivo WM range, Nicholson 1965 / Ranck & BeMent 1965) and `aniso_maxcond = 2` S/m (caps
eigenvalue magnitude; non-binding under `'vn'`). The tensor is supplied as a 6-component NIfTI in FSL
order `[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]`.

Exact `'vn'` implementation, verified against SimNIBS 4.6 `cond_utils.cond2elmdata` / `_fix_eigv`
(master-branch code - pin the methods to this version). Per element: (1) eigenvalues are normalized to
geometric mean 1 by dividing by `|det|^(1/3)`; (2) clamped - magnitude to `aniso_maxcond`, then any
eigenvalue below `λ1/aniso_maxratio` is raised to `λ1/aniso_maxratio` (the binding 10:1 ratio cap);
(3) re-normalized to geometric mean 1; (4) clamped again; then scaled per-tissue by σ0. The
non-positive-definite handling is asymmetric and worth stating: a tensor with **all three eigenvalues
≤ 0** is replaced by isotropic σ0, but a **mixed-sign** tensor (λ1>0, λ2/λ3≤0) is NOT isotropized - its
negative eigenvalues are raised to `λ1/10`, yielding a spurious 10:1 prolate along an unreliable axis.
The MD-dMRI model isotropizes its non-positive-definite ⟨D⟩ voxels upstream in `03` (these fall in
CSF/skull, not WM/GM). For the cohort, a per-model mixed-sign rate per ROI is logged so noisier or
atrophied subjects do not silently take the prolate branch.

## DTI model (baseline)

A separate single-shell acquisition (`sDTI_opt_80`, 80 directions, b≈1500, same subject and session),
already eddy/topup-corrected, is fitted with FSL `dtifit --save_tensor`; the tensor is handed to
`dwi2cond --all --regmthd=12dof` for its standard T1 coregistration + vecreg reorientation - the documented
SimNIBS anisotropy path (dwi2cond accepts a preprocessed dtifit tensor as input). The 12-DoF affine, not
dwi2cond's fnirt default, is the option the SimNIBS docs prescribe when "the distortion correction during
preprocessing is good enough"; our registration bake-off found fnirt over-warps the corrected data
without improving alignment. Its
limitation is intrinsic to single-shell DTI: the mono-exponential fit averages over all fibre
orientations in a voxel and is biased by non-Gaussian diffusion (kurtosis), so crossing fibres and
dispersion depress the measured anisotropy.

## MD-dMRI model (the contribution)

b-tensor-encoded MD-dMRI (linear + spherical encoding; the QTI framework of Westin et al. 2016,
Topgaard 2017) is fitted with the `md-dmri` toolbox. We use the **QTI covariance fit**
(`dtd_covariance`: constrained, regularized, heteroscedastic), whose first cumulant is the
**mean diffusion tensor ⟨D⟩** - the macroscopic diffusion tensor - stored in `cov_mfs.m(:,:,:,2:7)`
(Mandel Voigt, SI). `pipeline/run_qti_cov_cohort.m` runs the fit; `prepare_dmri_tensor.py` reconstructs the
full triaxial ⟨D⟩ and maps it by `'vn'`: σ ∝ ⟨D⟩. ⟨D⟩ is the *same macroscopic quantity* DTI
estimates, with MD = trace(⟨D⟩)/3, but estimated within a model that places the non-Gaussian variance
in a separate covariance term rather than letting it bias the mean. So ⟨D⟩ is a **less kurtosis-biased**
estimate of the macroscopic mean tensor than the single-shell DTI tensor. (It is *not* free-water
corrected: like the DTI tensor it still contains any CSF partial volume.)

**Why the covariance cumulant and not the full DTD Monte-Carlo mean.** The toolbox also produces a
non-parametric diffusion-tensor-distribution fit (de Almeida Martins & Topgaard; `dps.mat`), whose
mean tensor we used initially. On this data that Monte-Carlo mean is magnitude-inflated relative to the
covariance fit (whose MD matches the toolbox's own `dtd_covariance_MD`) and its eigenframe is unreliable:
the Monte-Carlo inversion has poor angular resolution for the mean tensor, so its principal axis sits well
off a robust low-b DTI and worsens with FA. The covariance cumulant mean is the standard QTI estimate,
de-inflated and better oriented against the same-data DTI. Its one cost: the 2nd-order cumulant overshoots
into non-positive-definite mean tensors in a minority of low-SNR/edge voxels (MD far below real tissue,
FA ≈ 1 - failed fits, not real anisotropy). Those voxels fall back to **isotropic** (σ0 under `'vn'`), so
they inject neither spurious anisotropy nor orientation. The per-subject isotropized fraction and the
fraction reaching the 10:1 cap are reported with the results.

The principal axes of the *independent* single-shell DTI and the MD-dMRI ⟨D⟩ tensors differ; the magnitude of
this divergence and its robustness to the registration method are reported with the results. So the
DTI↔MD-dMRI E-field contrast reflects **both** the tensor-estimation difference **and** a residual
orientation difference from the separate DTI acquisition; it is not a magnitude-only comparison.
Replacing the full-DTD mean with the QTI covariance mean shifts the deep-target E-field within
conductivity uncertainty - the tDCS field is intrinsically weakly sensitive to white-matter anisotropy
(Suh, Lee & Kim 2012, Phys Med Biol 57:6961).

## Registration

The md-dmri toolbox keeps analysis in native diffusion space (its motion/eddy correction, the Elastix
`AffineDTITransform`, is already applied upstream; the fit runs on the corrected series). A FEM head
model needs the conductivity tensor on the 1 mm T1/mesh grid, so we bring ⟨D⟩ to T1 - the
SimNIBS-standard direction. The cohort dMRI is EPI-distortion-corrected upstream (Synb0 synthetic
reverse-PE + `topup` + eddy), so an **affine dMRI → T1** is the right class (see the distortion-correction
note below). The registration is a **12-DOF affine driven by the dMRI model S0**, FLIRT to the charm T2
(mutual information), then applied to FA / eigenvalues / v1 by `vecreg`; the S0 driver is chosen by a
per-subject orientation bake-off (S0 beats FA as a driver, and `fnirt` over-warps the already-corrected
data). The DTI baseline (`dwi2cond`) registers by the same affine *class* (12-DOF) but a different driver
and target (FA → T1, correlation-ratio, vs the MD-dMRI S0 → T2, mutual-information), so the registration is
harmonized in class, not identical (quantified by the matched-registration control under Limitations).
(Historical: an earlier
single-HC pilot lacked a reverse-PE volume, so it used the `dwi2cond`-default nonlinear `fnirt` FA → T1
substitute - kept locally, not in this repo.)

Bringing the tensor across requires *reorienting* it, not just resampling components. We decompose:
the three eigenvalues are warped **independently** as scalar maps (`applywarp`, trilinear), and the
orientation frame is carried by `vecreg` with the warp (preservation-of-principal-direction
reorientation, Alexander 2001), with the principal axis anchored to the covariance fit's principal eigenvector
(`cov_dps.u`, registered as `v1_T1`). After interpolation the eigenvalue maps are re-sorted to
λ1≥λ2≥λ3 and paired with the frame **by magnitude order**, so a boundary voxel where two maps cross
cannot mis-assign eigenvalue to eigenvector.

Whole-tensor component-wise interpolation, by contrast, averages neighbouring tensors and shrinks the
anisotropy - the tensor "swelling" that log-Euclidean interpolation (Arsigny 2006) and PPD
reorientation are designed to avoid. Our scheme is a pragmatic stand-in for full log-Euclidean tensor
interpolation (e.g. DTI-TK): it preserves the native anisotropy at the cost of not reproducing the
partial-volume blurring the 2.5 mm resolution incurs (a disclosed trade-off). Both anisotropic models
register by the same affine class (12-DOF), with model-specific drivers and targets (DTI FA → T1; MD-dMRI
S0 → T2); the sensitivity of the orientation comparison to the registration method is assessed by a
matched-registration check and reported with the results. Because `'vn'` discards tensor magnitude, the
E-field depends on tensor shape (anisotropy ratios and orientation) only.

**Distortion correction governs the registration choice.** The gold-standard EPI-distortion correction
is FSL `topup` with a reverse-phase-encode b=0, after which a *rigid/affine* dMRI→T1 coregistration
suffices - this is what the SimNIBS team does (Mosayebi-Samani et al. 2025). The cohort is already
distortion-corrected upstream (Synb0 synthetic reverse-PE + `topup` + eddy), so an **affine dMRI→T1** is
the right class; over-warping with `fnirt` would risk re-introducing spurious deformation on
already-corrected data. The registration is **validated per subject** by orientation scoring (the warped
⟨D⟩ principal axis against anatomy - corpus callosum left–right, cerebral peduncle superior–inferior -
and against the delivered scalar references), rather than assumed.

## ROI definition

E-field and microstructure are read out over ROI masks in mesh space built from a FreeSurfer
`recon-all` parcellation (`analysis/build_rois.py` -> `registration/freesurfer_rois/`, loaded by the
analysis scripts through `analysis/_rois.py`). The ROIs fall in two functional groups. Group 1, the
cross-comparison regions, are the four cortical lobes and four white-matter lobes (Desikan grouped to
frontal/parietal/temporal/occipital, WM from the real wmparc), corpus callosum, and the
mesencephalon/pons split from the Iglesias 2015 brainstem subsegmentation, all bilateral by
construction. Group 1 carries both the H1 field result and the MRE cross-comparison. Whole-brain GM+WM
is taken from the charm tissue tags downstream. This matches Olsson et al. 2025, which used FreeSurfer
7.2 for the same structures; the cohort ships `recon-all` pre-computed, so the parcellation is consumed
directly (`recon-all` is run on the cluster whenever a new subject needs it). The recon-all segmentation
is registered to the charm T1 (FLIRT 6-DOF rigid) so the masks land in the FEM/E-field space. Group 2,
the deep stimulation-relevant nuclei, are the putamen, caudate, nucleus accumbens, external and internal
globus pallidus, substantia nigra pars compacta and reticulata, ventral tegmental area, red nucleus, and
subthalamic nucleus (Pu Ca NAC GPe GPi SNc SNr VTA RN STN, split left and right), all from the
CIT168/Pauli 2017 atlas warped via ANTs (`analysis/07_build_nuclei.sh`); they are field-only and
exploratory. The aseg subcortical structures used previously (thalamus, hippocampus, amygdala, and the
aseg basal ganglia) are dropped, so every subcortical readout is drawn from the single CIT168 atlas.
Each ROI is sampled over every GM/WM element it contains, not a fixed-radius sphere.

## Limitations

- **Resolution.** MD-dMRI is acquired at 2.5 mm; small deep nuclei (SN, STN) are near or below a
  voxel, so partial volume biases the per-voxel tensor at exactly the PD targets of interest. This is
  not fixable without reacquisition. All Group 2 nuclei come from CIT168 with overlap-allowed masks and
  are reported as exploratory field-only; the basal ganglia are well-resolved (overlap minimal,
  partial-volume modest) while the midbrain and subthalamic nuclei are at or below voxel size and
  additionally subject to inter-nucleus overlap, so the latter should be read as order-of-magnitude
  field exposure rather than precise per-nucleus values. Primary inference therefore leans on the
  coarser Group 1 cross-comparison regions and the cohort, with the Group 2 nuclei used for
  E-field-only reporting (no MRE cross-correlation at this resolution).
- **Magnitude is discarded by `'vn'`.** Conductivity magnitude is set by literature σ0, not by the
  measured ⟨D⟩. This is deliberate (per-voxel diffusion magnitude at 2.5 mm is unreliable, dominated by
  partial volume and noise), but it means disease-related MD changes enter only through
  anisotropy/orientation, not magnitude.
- **Anisotropy is a small effect; the scalar conductivities dominate.** For cortical targets, modeling
  brain anisotropy from DTI/QTI changes the tES E-field comparatively little, while uncertainty in the
  ohmic tissue conductivities is the main source of E-field variability, and MREIT is only weakly
  sensitive to brain anisotropy (Mosayebi-Samani et al. 2025, Imaging Neuroscience - SimNIBS team, same
  charm/dwi2cond/`'vn'` framework). The DTI↔MD-dMRI contrast here sits in that small-effect regime: the
  value of the QTI input is a more principled tensor estimate (kurtosis-aware multi-shell estimation +
  orientation), expected to be less biased, not a larger field.
- **De-kurtosis is a motivation, not a magnitude claim.** ⟨D⟩ is used as a less kurtosis-biased estimate of
  the macroscopic tensor than single-shell DTI (the first cumulant of the covariance fit separates kurtosis
  into a second-cumulant term; Westin 2016). The diffusion-model E-field comparison is sensitive to diffusion
  preprocessing (spatial smoothing, and the difference in native resolution between the two acquisitions), so
  it is treated as a controlled-input comparison rather than a magnitude result; the preprocessing-sensitivity
  analysis is reported with the results.
- **Validation.** No in-vivo conductivity ground truth exists; the model is motivated by bias
  arguments, not direct validation. MR current-density imaging (MRCDI/MREIT; Gregersen 2024) is the
  planned validation route.
- **DTI baseline.** The DTI model is a separate single-shell acquisition, registered by the same affine
  class (12-DOF) as the MD-dMRI model but with a model-specific driver and target (dwi2cond FA→T1 vs S0→T2).
  Its principal direction differs from ⟨D⟩, reflecting the genuine tensor-estimation/acquisition difference;
  the registration-method sensitivity and the divergence magnitude are assessed by a matched-registration
  check and reported with the results.

## References

Tuch et al. (2001) PNAS - conductivity ∝ diffusion tensor (effective medium).
Güllmar et al. (2010) NeuroImage 51 - volume-normalized mapping (Eq. 3–4).
Rullmann et al. (2009) NeuroImage 44 - volume-constrained anisotropic FEM head model.
Westin et al. (2016) NeuroImage 135 - QTI; ⟨D⟩ + covariance; MD = tr⟨D⟩/3 (Eq. 14).
Topgaard (2017) J Magn Reson 275 - diffusion tensor distributions; b-tensor signal model (Eq. 40).
Lampinen et al. (2017) NeuroImage 147 - microscopic anisotropy needs variable b-tensor shape; d_FW = 3.0 µm²/ms.
Nicholson (1965) / Ranck & BeMent (1965) Exp Neurol - ex-vivo WM conductivity anisotropy 7–10:1.
Alexander et al. (2001) IEEE TMI - preservation-of-principal-direction reorientation.
Arsigny et al. (2006) MRM - log-Euclidean tensor interpolation.
Gregersen et al. (2024) Imaging Neuroscience - MRCDI for head-model validation.
Mosayebi-Samani et al. (2025) Imaging Neuroscience - brain anisotropy has a small effect on the tES E-field (scalar conductivity uncertainty dominates); MREIT weakly sensitive to anisotropy; uses charm + dwi2cond + `'vn'` (Eq. 4) with topup + rigid dMRI→T1 coregistration.

## Appendix - alternatives evaluated and not used

Free-water elimination (σ ∝ ⟨D⟩_tissue), magnitude preservation (`'dir'` / per-tissue), and μFA /
covariance were each tested and rejected. In brief: FWE is ~null under `'vn'` and ill-posed at this
volume count; magnitude preservation injects
partial-volume and noise variance (and SimNIBS `'dir'` also distorts the GM/WM contrast); μFA and the
covariance tensor are microscopic measures with no macroscopic eigenframe, so they cannot define a
conductivity tensor (they remain useful for interpretation).
