# Anisotropic conductivity models for tDCS simulation

Three conductivity models drive the finite-element tDCS simulations in this pipeline,
ordered by how much diffusion information they use:

1. **DTI** — the conventional diffusion-tensor mapping (SimNIBS `dwi2cond`), our baseline.
2. **σ ∝ ⟨D⟩** — the mean diffusion tensor from b-tensor-encoded MD-dMRI (QTI).
3. **Free-water-eliminated σ ∝ ⟨D⟩_tissue** — a multi-compartment refinement that removes
   the CSF partial-volume contribution.

All three share the same physics for turning a diffusion tensor into a conductivity tensor;
they differ only in which diffusion tensor they start from.

## From diffusion to conductivity

Tuch et al. (2001) showed that in tissue, where the same microstructure (axons, membranes)
restricts both water diffusion and ionic current, the conductivity tensor σ and the diffusion
tensor D share eigenvectors and have linearly related eigenvalues:

    σ ∝ D

We pass D to SimNIBS with `anisotropy_type='vn'`, which applies the volume-normalised mapping
of Güllmar et al. (2010) and Rullmann et al. (2009):

    σ(x) = σ_tissue(x) · D(x) / det[D(x)]^(1/3)

The geometric-mean normalisation makes σ depend only on the *shape* (anisotropy and
orientation) of D; the tissue's isotropic baseline conductivity sets the overall scale.
SimNIBS caps the eigenvalue ratio at `aniso_maxcond=4` (an 8:1 input ratio); we keep this
cap for all three models so they are directly comparable. The tensor is supplied as a
6-component NIfTI in FSL order `[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]`.

## Model 1 — DTI (baseline)

Single-shell diffusion (80 directions, b≈1500) fitted with FSL `dtifit`, registered to the
structural T1 with `dwi2cond --all` (eddy-current correction, nonlinear FA→T1 registration,
tensor reorientation). This is the standard SimNIBS anisotropy input. Its limitation is
intrinsic to single-shell DTI: the apparent diffusion tensor averages over all fibre
orientations in a voxel, so crossing fibres and orientation dispersion depress the measured
anisotropy, and the mono-exponential fit is biased by non-Gaussian diffusion (kurtosis).

## Model 2 — σ ∝ ⟨D⟩ (MD-dMRI mean tensor)

b-tensor-encoded MD-dMRI (linear + spherical encoding, the QTI framework of Westin et al.
2016) is fitted with the Topgaard `md-dmri` toolbox. The fit yields the mean diffusion tensor
⟨D⟩ — the first cumulant of the intra-voxel diffusion-tensor distribution — stored in
`dps.mat`. ⟨D⟩ is the same macroscopic quantity DTI estimates, but the spherical encoding
separates isotropic from anisotropic variance, so ⟨D⟩ is less biased by diffusional kurtosis
and free water than the single-shell DTI tensor.

The full tensor is reconstructed from the six `md??` fields (stored in SI units, m²/s) and
used directly: σ ∝ ⟨D⟩. It is fully triaxial (three distinct eigenvalues), with principal
eigenvector equal to the fit's `u`, trace equal to 3·MD, and extreme eigenvalues equal to the
fit's `ad`/`rd`.

Empirically, the MD-dMRI and DTI conductivities deviate from each other by about as much as
either deviates from an isotropic model (~5% median in brain tissue, ~20% at the 95th
percentile, element-wise on a shared mesh). The deviation grows with diffusional kurtosis —
the two models agree in Gaussian tissue and diverge where the DTI mono-exponential assumption
fails — which is consistent with ⟨D⟩ being the less-biased estimate.

## Model 3 — free-water-eliminated σ ∝ ⟨D⟩_tissue

The QTI fit also resolves each voxel into three fixed-diffusivity compartments with signal
fractions f₀+f₁+f₂ = 1: anisotropic tissue (low MD), restricted tissue, and free water
(MD ≈ 3 µm²/ms, the CSF compartment). This decomposition is impossible with single-shell DTI;
separating compartments by *shape* requires the spherical encoding.

Free water has high, isotropic diffusivity, so any CSF partial volume biases ⟨D⟩ toward
isotropy and masks the true tissue anisotropy. We remove it (free-water elimination,
Pasternak et al. 2009) and renormalise over the tissue compartments:

    ⟨D⟩_tissue = (f₀·D₀ + f₁·D₁) / (f₀ + f₁) ,   σ ∝ ⟨D⟩_tissue

The principal axis is anchored to the validated Model-2 direction, so only the eigenvalues
(the free-water-corrected anisotropy) change relative to Model 2 — isolating the free-water
effect. Where the tissue fraction is too small to be reliable (f₀+f₁ < 0.30, near-pure CSF)
the model falls back to ⟨D⟩.

This is a de-biasing correction, not an anisotropy inflation: where there is no free water it
reduces to Model 2 exactly; where CSF contaminates the voxel it recovers the masked tissue
anisotropy (whole-brain anisotropy rises from 1.89 to 2.06; the per-voxel correction scales
with free-water fraction). SimNIBS already models bulk CSF as a separate tissue, so removing
the CSF compartment from the WM/GM tensor is consistent with how the head model is built. The
effect on the E-field is small (~2% median) but concentrated in the high-CSF deep targets
(substantia nigra, VTA), where partial volume is worst.

## Registration

The diffusion tensors live on the 2.5 mm MD-dMRI grid and must be brought to the 1 mm
structural/mesh grid. Following the principle used by the `md-dmri` toolbox (rotate
directions, do not interpolate whole tensors), we register the eigenvalues as scalars
(trilinear, which preserves the anisotropy magnitude) and carry the orientation as a
direction (`vecreg`), anchoring the principal axis to the diffusion direction validated
against the DTI fit (~18° median agreement in core white matter). Whole-tensor trilinear
interpolation, by contrast, averages neighbouring tensors and dilutes anisotropy.

## Limitations

- **Resolution.** The MD-dMRI is acquired at 2.5 mm (an SNR-versus-time trade-off on this
  protocol, not a scanner limit). Conductivity tensors inherit this grid and are always
  coarser than the 1 mm segmentation. Small deep nuclei (substantia nigra, STN) are smaller
  than a voxel, so partial volume biases both the segmentation and the per-voxel tensor at
  exactly the targets of interest. Per-voxel tensors in and around these nuclei should not be
  over-interpreted; free-water elimination mitigates but cannot remove this bias.
- **Validation.** No in-vivo conductivity ground truth exists, so the choice between models is
  motivated by bias arguments, not direct validation. MR current-density imaging
  (MRCDI/MREIT; Gregersen et al. 2024) is the planned validation route — comparing simulated
  and measured current-induced magnetic fields.
- **Scope.** The conductivity mapping is the macroscopic Tuch relation; microscopic anisotropy
  (μFA) is deliberately not used (see appendix).

## References

Tuch et al. (2001) PNAS — conductivity ∝ diffusion tensor.
Güllmar et al. (2010) NeuroImage — volume-normalised mapping.
Rullmann et al. (2009) NeuroImage — anisotropic FEM head model.
Westin et al. (2016) NeuroImage — q-space trajectory imaging (QTI).
Topgaard (2017) J Magn Reson — diffusion tensor distributions.
Pasternak et al. (2009) MRM — free-water elimination.
Gregersen et al. (2024) Imaging Neuroscience — MRCDI for head-model validation.

---

## Appendix — model development notes

Approaches evaluated and not carried forward:

- **μFA-derived microscopic anisotropy.** μFA measures per-compartment anisotropy and is
  larger than macroscopic FA wherever fibres disperse or cross. But those are exactly the
  voxels with no coherent current direction, where the effective (voxel-scale) conductivity
  the FEM needs is near-isotropic. μFA would impose strong anisotropy along an arbitrary axis,
  so it is the wrong quantity for conductivity despite being the unique MD-dMRI measurement.
- **Cylindrical ad/rd tensor.** An interim model used only the two extreme eigenvalues
  (ad, rd) with λ₂=λ₃, on the mistaken assumption that the full mean-tensor components were
  unavailable. They were present in `dps.mat` (in SI units); Model 2 uses the full triaxial
  tensor and supersedes this.
- **Oblate-corrected / two-tier prolate variants.** Earlier attempts to special-case oblate
  voxels and to fall back to isotropic conductivity in dispersed voxels. The full triaxial
  mean tensor handles oblate geometry directly, making these unnecessary.
- **Fraction-weighted multi-compartment mixing** (σ_eff = Σ fₖ·D̂ₖ with per-compartment
  normalisation). This over-diluted the anisotropic compartment and rotated the principal
  axis ~23° off the validated direction. Free-water elimination (Model 3) is the cleaner,
  better-behaved multi-compartment formulation.

A note on the data: every field in `dps.mat` is populated inside the brain mask. Diffusivities
are stored in SI units (≈10⁻⁹ m²/s), so variances are ≈10⁻¹⁸ and fourth-order moments ≈10⁻³⁶ —
small in magnitude but not zero. Any inspection must test at SI precision. The QTI second
moments (free-water variance, kurtosis) are available but unused by the current models.
