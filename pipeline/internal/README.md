# Internal reference — alternatives evaluated and NOT used

The published MD-dMRI conductivity model is **one model**: σ ∝ ⟨D⟩, the plain QTI mean diffusion
tensor mapped to conductivity by the standard SimNIBS `'vn'` (volume-normalized) rule. The downstream
physics is identical to dwi2cond's DTI model; only the input tensor differs (QTI ⟨D⟩ vs single-shell
DTI). Everything except that input tensor follows standard SimNIBS procedure.

This folder keeps the alternatives we tested so the decisions are not relitigated. None are part of
the pipeline or the documentation. Pilot evidence (FullPD5, C3→Fp2, 2 mA) is summarized below.

## 1. Free-water elimination — `01e_save_multicompartment_tensor.py`
σ ∝ ⟨D⟩_tissue: remove the QTI free-water compartment (highest-MD bin) and renormalize over the
tissue compartments. Rejected as the model for two reasons:
- **Negligible effect under `'vn'`.** `'vn'` discards magnitude, so FWE can only act through the
  anisotropy ratio (median λ1/λ3 1.92 → 2.13). Pilot E-field delta vs plain ⟨D⟩: GM median −0.3%,
  brain median −0.0%, p95 ~4.6%. Essentially null where it is used.
- **Ill-posed at this volume count.** The 3-compartment fit is a Laplace/NNLS inversion (Topgaard
  2017 calls it "notorious"); validated bin separation needs ~1000 measurements at SNR≈1000. The
  pilot has 82 volumes (44 LTE + 38 STE); the project's own covariance maps are non-physical at this
  count. ⟨D⟩ (first moment) is robust; the per-bin fractions are not.
FWE only does meaningful work when paired with magnitude preservation (below), which is also rejected.

## 2. Magnitude preservation (`'dir'` and a custom per-tissue map)
Let conductivity magnitude track the measured ⟨D⟩ (or MD) instead of pinning each element's geometric
mean to literature σ₀ (what `'vn'` does). Rejected:
- **SimNIBS `'dir'` introduces a between-tissue artifact.** `'dir'` applies a single global Rullmann
  scalar across WM+GM, collapsing the literature GM/WM conductivity contrast (2.18×) toward the
  diffusivity ratio (~1), spuriously raising the GM field +15% (a contrast artifact, not signal).
- **The within-tissue magnitude is mostly noise.** A clean per-tissue map (inter-tissue contrast
  preserved) still changed the field ~11% and produced the highest WM peak field of any model. The
  within-tissue geometric-mean CoV is ~46% (WM and GM), far above plausible physiological conductivity
  heterogeneity (WM MD CoV ~10-20%); the excess is partial volume + FWE noise + the noisy λ3. Injecting
  it is over-enhancement of unvalidated magnitude.
`'vn'` is correct precisely because it is robust to this unreliable magnitude. MD's real value is for
interpretation (it is where the PD signal lives, Olsson 2025), not for driving conductivity.

## 3. μFA and the QTI covariance tensor ℂ
Microscopic anisotropy (μFA) and the covariance tensor ℂ describe intra-voxel dispersion/variance,
not a macroscopic current direction. An effective-medium conductivity tensor needs an eigenframe;
μFA is orientation-invariant (powder-averaged) and ℂ is a 4th-order variance with no conductivity
axis (Lampinen 2017; Westin 2016). Where μFA ≫ FA (crossing/dispersion) the macroscopic current has
no preferred direction, so the near-isotropic ⟨D⟩ is the physically correct effective behavior.
ℂ is also poorly conditioned at 38 spherical volumes. The covariance is valuable for interpretation
(μFA, V_iso, kurtosis) and is implicitly what lets QTI estimate ⟨D⟩ with less kurtosis bias, but it
does not belong in the conductivity tensor.

## Reproducing the pilot evidence
The 2×2 sensitivity sims (vn/dir × meanD/FWE) and the per-tissue magnitude test were run in the
gitignored work dir as `exp_magnitude_sensitivity.py`, `exp_compare_magnitude.py`, and
`exp_pertissue_mag.py`. They are scratch experiments, not pipeline code.
