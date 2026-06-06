# Anisotropic conductivity models for tDCS simulation

Three conductivity models drive the finite-element tDCS simulations in this pipeline,
ordered by how much diffusion information they use:

1. **DTI** — the conventional diffusion-tensor mapping (SimNIBS `dwi2cond`), our baseline.
2. **σ ∝ ⟨D⟩** — the mean diffusion tensor from b-tensor-encoded MD-dMRI (QTI).
3. **Free-water-eliminated σ ∝ ⟨D⟩_tissue** — a multi-compartment refinement that removes
   the CSF partial-volume contribution.

All three share the same physics for turning a diffusion tensor into a conductivity tensor;
they differ only in which diffusion tensor they start from.

The novelty is methodological: (i) the first tDCS conductivity tensor derived from b-tensor-
encoded MD-dMRI / QTI rather than single-shell DTI; (ii) a mean tensor ⟨D⟩ that is provably less
biased by diffusional kurtosis and free water than the DTI tensor — the same macroscopic Tuch
target, but a cleaner estimate; and (iii) free-water elimination applied to the conductivity
tensor itself. This is an effective-medium model — one tensor per finite element — so σ depends
on the macroscopic ⟨D⟩ only and makes no claim about intra-voxel/microscopic conductivity (μFA
is therefore the wrong quantity and is excluded; see appendix).

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
SimNIBS bounds the conductivity tensor with two separate parameters, set identically for both
anisotropic models so any E-field difference reflects the tensor source, not the clip. The
binding one is `aniso_maxratio = 10`, which caps the eigenvalue **ratio** at 10:1 — the top of
the 7–10:1 ex-vivo white-matter range (Nicholson 1965; Ranck & BeMent 1965); it clips ~0.1% of
the DTI tensor and ~0.7% of the σ ∝ ⟨D⟩ tensor (median ratio 1.89, p95 4.10). The second,
`aniso_maxcond = 2` S/m (the SimNIBS default), caps the eigenvalue **magnitude**; under
volume-normalisation the conductivity eigenvalues sit at ~σ_tissue (≈0.13 S/m in WM), so it is
essentially non-binding (<0.05% of voxels) and is *not* a ratio cap. (The ISO model has no
tensor.) The tensor is supplied as a 6-component NIfTI in FSL order `[Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]`.

## Model 1 — DTI (baseline)

The baseline comes from a **separate single-shell DTI acquisition** (`sDTI_opt_80`, 80
directions, b≈1500), collected in the **same subject and session** as the MD-dMRI (it is not the
LTE subset of the b-tensor data). It is fitted with FSL `dtifit` and registered to the structural
T1 with `dwi2cond --all` (eddy-current correction, nonlinear FA→T1 registration, tensor
reorientation) — the standard SimNIBS anisotropy input. Its limitation is intrinsic to
single-shell DTI: the apparent diffusion tensor averages over all fibre orientations in a voxel,
so crossing fibres and orientation dispersion depress the measured anisotropy, and the
mono-exponential fit is biased by non-Gaussian diffusion (kurtosis). Because it is a distinct
sequence with its own distortions and registration, the DTI tensor differs from the MD-dMRI
tensor in orientation as well as eigenvalues (quantified below), which the Model-vs-DTI contrast
must be read in light of (see Limitations).

## Model 2 — MD-dMRI (σ ∝ ⟨D⟩_tissue, free-water-eliminated)

> This is the **single MD-dMRI model**. The QTI mean tensor ⟨D⟩ is described first as the basis, then
> the free-water elimination that *defines* the model. Plain ⟨D⟩ without free-water elimination is the
> degenerate **sensitivity** case (`02_build_conductivity_tensor.py --meanD`), not a separate model.

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

The MD-dMRI principal axis is the fit's `u` exactly: the mean tensor's first eigenvector equals
`dps.u` to a median 0.01°. After registration to T1, `u` agrees with the *independent* single-shell
DTI principal direction (dwi2cond V1) only moderately — a median of ~22° in core white matter
(FA>0.5), rising to ~30° across all WM (FA>0.3; reproduced by `tests/validate_mean_tensor.py`).
So the two anisotropic models differ in **orientation as well as eigenvalues**: the MD-dMRI-vs-DTI
E-field difference is not a pure magnitude effect. It is *not* circular — each model carries its
own independently estimated orientation (DTI from the sDTI fit, MD-dMRI from the QTI `u`).

Empirically, the MD-dMRI and DTI conductivities deviate from each other by about as much as
either deviates from an isotropic model (~5% median in brain tissue, ~20% at the 95th
percentile, element-wise on a shared mesh). The deviation grows with diffusional kurtosis —
the two models agree in Gaussian tissue and diverge where the DTI mono-exponential assumption
fails — which is consistent with ⟨D⟩ being the less-biased estimate.

### Free-water elimination — the defining step of the MD-dMRI model

The QTI fit also resolves each voxel into three fixed-diffusivity compartments with signal
fractions f₀+f₁+f₂ = 1: anisotropic tissue (low MD), restricted tissue, and free water (the CSF
compartment, observed bin MD ≈ 3.3 µm²/ms — consistent with the literature free-water value of
≈ 3.0 µm²/ms at 37 °C; median fraction f₂ ≈ 0.24 brain-wide). This decomposition is impossible
with single-shell DTI; separating compartments by *shape* requires the spherical encoding.

Free water has high, isotropic diffusivity, so any CSF partial volume biases ⟨D⟩ toward
isotropy and masks the true tissue anisotropy. We remove it (free-water elimination,
Pasternak et al. 2009) and renormalise over the tissue compartments:

    ⟨D⟩_tissue = (f₀·D₀ + f₁·D₁) / (f₀ + f₁) ,   σ ∝ ⟨D⟩_tissue

The principal axis is anchored to the validated Model-2 direction, so only the eigenvalues
(the free-water-corrected anisotropy) change relative to plain ⟨D⟩ — isolating the free-water
effect. Where the tissue fraction is too small to be reliable (f₀+f₁ < 0.30, near-pure CSF)
the model falls back to ⟨D⟩.

This is a de-biasing correction, not an anisotropy inflation: where there is no free water it
reduces to plain ⟨D⟩ exactly; where CSF contaminates the voxel it recovers the masked tissue
anisotropy (median λ1/λ3 over the QTI brain mask rises from 1.92 for ⟨D⟩ to 2.13 for ⟨D⟩_tissue;
the per-voxel correction scales with free-water fraction). SimNIBS already models bulk CSF as a separate tissue, so removing
the CSF compartment from the WM/GM tensor is consistent with how the head model is built. The
effect on the E-field is small (~2% median) but concentrated in the high-CSF deep targets
(substantia nigra, VTA), where partial volume is worst.

## Registration

The diffusion tensors live on the 2.5 mm MD-dMRI grid and must be brought to the 1 mm
structural/mesh grid, which requires *reorienting* the tensor, not just resampling its six
components. We decompose it: the three eigenvalues are interpolated **independently** as scalar
maps (FLIRT trilinear), and the orientation frame is carried by `vecreg` (preservation-of-
principal-direction reorientation, Alexander et al. 2001), with the principal axis anchored to
`dps.u` (`v1_T1`).

Whole-tensor component-wise trilinear interpolation, by contrast, averages neighbouring tensors
and shrinks the anisotropy (median λ1/λ3 1.92 → 1.28 here) — the tensor "swelling"/dilution that
log-Euclidean interpolation (Arsigny et al. 2006) and PPD reorientation are designed to avoid.
Our scheme is a pragmatic stand-in for full log-Euclidean tensor interpolation (e.g. DTI-TK):
it preserves the native anisotropy at the cost of not reproducing the genuine partial-volume
blurring that the 2.5 mm resolution incurs — a deliberate, disclosed trade-off.

**Pairing validity.** Because the eigenvalue maps are interpolated independently, they could in
principle mis-pair with the separately reoriented frame. They cannot here: after interpolation
the three maps are re-sorted to λ1≥λ2≥λ3 and paired with the frame **by magnitude order**
(largest → principal axis), not by map identity, so a boundary voxel where two maps cross cannot
mis-assign eigenvalue to eigenvector. The principal axis is the validated `dps.u`; its agreement
with the independent single-shell DTI V1 (~22° median in core WM, ~30° across all WM) is given in
the Model 2 section and reproduced by `tests/validate_mean_tensor.py`.

## ROI definition

E-field and microstructure are read out over anatomical atlas masks, not coordinate
spheres. Spheres at MNI coordinates overlapped heavily in the midbrain — substantia
nigra and VTA sit only a few mm apart — so neighbouring nuclei shared most of their
voxels. Atlas masks follow the real anatomy and (after a winner-take-all assignment)
never share a voxel.

- **Basal ganglia** — HarvardOxford-Subcortical (caudate, putamen, pallidum). The
  pallidum label is the whole globus pallidus; GPe and GPi are deliberately merged
  (they are not separable here and the atlas does not split them). HarvardOxford is on
  the MNI152 182×218×182 grid, identical to CHARM's MNI output, so masks warp into
  subject space through the CHARM deformation with no intermediate resampling.
- **Midbrain** — CIT168/Pauli 2017, the only standard atlas that resolves SNc/SNr/VTA.
  The three are merged into one SN/VTA ROI (inseparable at the 2.5/3 mm diffusion/MRE
  resolution, and alike in Olsson et al. 2025); the red nucleus is dropped. CIT168 is in
  MNI152-2009c, bridged to the NLin6/CHARM space with one affine registration (the
  midbrain is central, where the two MNI variants differ by ~1–2 mm) before the CHARM
  warp. Round-tripping each warped ROI centroid back to MNI lands within 1–2 mm of the
  atlas coordinate.
- **Cortex** — the L_M1 reference under the C3 anode stays a small sphere; cortical GM is
  well resolved and has no neighbouring-structure overlap.

Built by `analysis/06_build_atlas_rois.sh`; shared by `04` and `05` via `_atlas_rois.py`.

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
- **DTI baseline acquisition.** The DTI model uses a *separate* single-shell `sDTI_opt_80`
  acquisition (same subject and session), registered independently of the MD-dMRI. Its principal
  direction differs from the MD-dMRI `u` by ~22–30° (above), so the Model-vs-DTI E-field contrast
  reflects acquisition and orientation differences as well as the tensor-estimation method — it
  is not a controlled same-data comparison of magnitude alone.
- **Scope.** The conductivity mapping is the macroscopic Tuch relation; microscopic anisotropy
  (μFA) is deliberately not used (see appendix).

## References

Tuch et al. (2001) PNAS — conductivity ∝ diffusion tensor.
Güllmar et al. (2010) NeuroImage — volume-normalised mapping.
Rullmann et al. (2009) NeuroImage — anisotropic FEM head model.
Westin et al. (2016) NeuroImage 135 — q-space trajectory imaging (QTI); ⟨D⟩ + covariance; MD = tr⟨D⟩/3 (Eq. 14).
Topgaard (2017) J Magn Reson 275 — diffusion tensor distributions; b-tensor encoding (Eq. 20–24); DTD signal S=S0∫P(D)e^(−b:D)dD (Eq. 40, 46); NNLS components s=K·p (Eq. 43).
Lampinen et al. (2017) NeuroImage 147 — microscopic anisotropy (μFA) requires variable b-tensor shape; isotropic free-water compartment d_FW = 3.0 µm²/ms (Eq. 4).
Lasič et al. (2014) Front Phys / Szczepankiewicz et al. (2016) NeuroImage — μFA definition (the `dps.ufa` field; deliberately NOT used here).
Pasternak et al. (2009) MRM — free-water elimination.
Nicholson (1965) Exp Neurol — ex-vivo WM conductivity anisotropy ~9:1.
Ranck & BeMent (1965) Exp Neurol — ex-vivo WM conductivity anisotropy ~7:1.
Alexander et al. (2001) IEEE TMI — preservation-of-principal-direction tensor reorientation.
Arsigny et al. (2006) MRM — log-Euclidean tensor interpolation.
Gregersen et al. (2024) Imaging Neuroscience — MRCDI for head-model validation.
Makris et al. (2006) / HarvardOxford-Subcortical atlas (FSL) — basal-ganglia ROIs.
Pauli et al. (2018) Sci Data — CIT168 probabilistic subcortical atlas (SN/VTA ROI).

### Provenance of the MD-dMRI constants (verified field-by-field against the papers + `md-dmri` source)

`fit/dps.mat` is produced by the Topgaard `md-dmri` toolbox, method `dtd`, function
`methods/dtd/dtd_dtds2dps.m` → `tools/tensor_maths/tm_dt_to_dps.m`. Each field maps to QTI/DTD theory:

| `dps` field | Quantity | Source | Units stored |
|---|---|---|---|
| `mdxx..mdyz` | mean tensor ⟨D⟩ = Σ_k f_k D_k — first moment of the DTD, **over ALL compartments** | Westin 2016; Topgaard 2017 Eq. 56; code lines 27–32 | SI m²/s (~1e-9) → ×1e9 = µm²/ms |
| `MD` | tr(⟨D⟩)/3 | Westin 2016 Eq. 14; `tm_md` | µm²/ms (code ×1e9, line 73) |
| `ad`, `rd` | λ1 and (λ2+λ3)/2 of ⟨D⟩ | `tm_dt_to_dps.m` L37–38 | µm²/ms |
| `u` | principal eigenvector of ⟨D⟩ | `tm_dt_to_dps.m` L39 | unit vector |
| `bin` | DTD NNLS components (f_k, D_k); free water = highest-MD isotropic bin (here MD≈3.34) | Topgaard 2017 Eq. 43 | SI m²/s |
| `ufa` | μFA (microscopic FA) — **deliberately unused** | Szczepankiewicz 2016 (`dtds2dps.m` L58) | — |

- **⟨D⟩ is NOT free-water-corrected.** Westin 2016 (Fig. 9: "MD is high in the ventricles … where there is
  CSF") and Topgaard 2017 (Fig. 7: ⟨D⟩ identical across CSF-containing voxels) state the mean tensor folds the
  free-water bin in. Empirically `dps.MD = Σ_k f_k·MD_k` exactly (incl. the 3.34-µm²/ms free-water bin),
  brain-median 1.53 with free water vs 0.95 after FWE. **This is the justification for the free-water elimination** (FWE recombines
  only the tissue bins; Pasternak 2009; Lampinen 2017 Eq. 4).
- **μFA is rejected for conductivity** because it is a *microscopic*, orientation-dispersion-invariant measure
  (from the powder-averaged signal; Lampinen 2017) with no coherent macroscopic direction — whereas the
  effective-medium conductivity (Tuch 2001) needs the macroscopic ⟨D⟩ whose principal eigenvector is the current
  direction.
- **Fully triaxial:** three independent eigenvalues; verified 90–95% of brain voxels genuinely triaxial
  (λ2≠λ3 in 99–100%; ~75% prolate-leaning, ~25% oblate-leaning) — a cylindrical (λ2=λ3) ad/rd form would misfit
  the oblate/planar voxels (the scrapped Model in the appendix).
- **Units verified end-to-end:** `dtd_dtds2dps.m:73` derives MD/ad/rd from the ×1e9-scaled tensor while
  `mdxx..mdyz` stay SI — matching `01d` (scales ×1e9) vs `01c` (reads as-is); `trace(mdxx×1e9)/3 == dps.MD` to
  machine zero.

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
  axis ~23° off the validated direction. Free-water elimination is the cleaner,
  better-behaved multi-compartment formulation.

A note on the data: every field in `dps.mat` is populated inside the brain mask. Diffusivities
are stored in SI units (≈10⁻⁹ m²/s), so variances are ≈10⁻¹⁸ and fourth-order moments ≈10⁻³⁶ —
small in magnitude but not zero. Any inspection must test at SI precision. The QTI second
moments (free-water variance, kurtosis) are available but unused by the current models.
