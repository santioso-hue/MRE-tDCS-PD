# Anisotropic Conductivity Tensor Models for tDCS Simulation
## Full Mathematical Derivation — All Models Attempted

**Project:** MRE-tDCS in Parkinson's Disease — FullPD5 pilot subject  
**Tool chain:** SimNIBS 4.6 (FEM), FSL (registration), md-dmri toolbox (DPS fitting)  
**Electrode montage:** C3 (+2 mA, anode) / Fp2 (−2 mA, cathode)  
**Target structures:** Basal ganglia (SNc, SNr, VTA, PUT, Ca, NAc, GPi, GPe, RN)

---

## Part I — Theoretical Background

### 1.1 Why Anisotropic Conductivity Matters

Standard tDCS simulations assign isotropic conductivity per tissue type (WM ≈ 0.126 S/m,
GM ≈ 0.276 S/m, CSF ≈ 1.65 S/m). In reality, white matter is highly anisotropic: current
flows preferentially along myelinated axon bundles, so σ_∥/σ_⊥ ≈ 3–10:1. Ignoring this
changes the spatial distribution of the E-field in deep grey matter targets by up to 10–20%
(Saturnino et al., 2019).

The standard approach (Tuch et al., 2001) exploits diffusion MRI as a proxy for
microstructural tissue anisotropy.

### 1.2 Tuch 2001 Linear Relationship

Tuch et al. (2001, Ann. Biomed. Eng.) proposed:

    σ ∝ D

where **σ** is the 3×3 symmetric positive-definite conductivity tensor (S/m) and **D** is the
diffusion tensor (m²/s or μm²/ms). The proportionality constant is absorbed into the
tissue-specific baseline conductivity.

Physical rationale: in tissues where both electrical and diffusive transport are dominated
by the same microstructural channels (axons, myelin sheaths), their tensors should share
the same eigenvectors with proportional eigenvalues.

### 1.3 SimNIBS Volume-Normalized (VN) Implementation

SimNIBS 4.6 (`anisotropy_type = 'vn'`) implements the Volume-Normalized normalization
from Güllmar et al. (2010, NeuroImage):

    σ(x) = σ_tissue(x) · D(x) / det[D(x)]^(1/3)

where:
- **σ_tissue(x)** is the scalar tissue baseline conductivity at voxel x (from SimNIBS tissue table)
- **D(x)** is the input diffusion tensor at voxel x (passed via `fname_tensor`)
- **det[D]^(1/3)** is the geometric mean eigenvalue (the "isotropic equivalent scale")

This normalization makes σ scale-invariant: only the anisotropy shape of D matters, not
its absolute magnitude. The tissue conductivity σ_tissue provides the absolute scale.

**SimNIBS parameter `aniso_maxcond = 4` (default):**  
Caps the ratio:  max(eigenvalues of σ) / det(σ)^(1/3)  ≤  aniso_maxcond = 4

For a cylindrically symmetric tensor D with eigenvalues (λ_1, λ_2, λ_2) where λ_1 ≥ λ_2,
this translates to an effective input eigenvalue ratio cap of:

    λ_1 / λ_2  ≤  aniso_maxcond^(3/2)  =  4^(3/2)  =  8.0 : 1

Any input tensor with λ_1/λ_2 > 8:1 is silently clamped to 8:1 in the simulation.

**Input format:** `fname_tensor` must point to a NIfTI file with 6 components per voxel in
FSL dtifit order: [D_xx, D_xy, D_xz, D_yy, D_yz, D_zz] (upper triangle, row-major).
Units: any consistent unit (SimNIBS normalizes via det, so μm²/ms works fine).

---

## Part II — Available Data

### 2.1 DTI Acquisition

**File:** `NiiFiles/FullPD5_WIP_dMRI_Linear_medium_601.nii.gz`  
**Protocol:** Linear tensor encoding (LTE), 44 volumes, b = 0/100/700/1400/2000 s/mm²  
**Signal model:** Stejskal-Tanner monoexponential

    S(b, g) = S_0 · exp(−b · g^T D g)

where **g** is the gradient unit vector and **D** is the macroscopic apparent diffusion tensor.
Fitted with FSL `dtifit` (least-squares). Outputs in `registration/lte_dti/`.

**FullPD5 empirical statistics (after registration to T1 space):**

| Quantity | Median | Notes |
|----------|--------|-------|
| L1 (λ_1) | 0.931 μm²/ms | Largest eigenvalue; "axial diffusivity" |
| L2 (λ_2) | 0.697 μm²/ms | Middle eigenvalue |
| L3 (λ_3) | 0.561 μm²/ms | Smallest eigenvalue; "radial diffusivity" |
| FA | 0.234 | Macroscopic fractional anisotropy |
| λ_1 / λ_3 | 1.66 : 1 | End-to-end eigenvalue ratio |
| λ_1 / λ_geom | 1.414 : 1 | where λ_geom = (λ_2·λ_3)^0.5 |
| |λ_1−λ_2| | 0.252 μm²/ms | Non-zero triaxiality |
| |λ_2−λ_3| | 0.156 μm²/ms | Non-zero triaxiality |

**Critical limitation:** D_DTI is the macroscopic APPARENT diffusion tensor — an ensemble
average over all fiber populations within the voxel. In voxels with crossing fibers or
orientation dispersion, D_DTI eigenvalues are confounded:
- λ_1 is depressed (dominant direction mixes with off-axis signal)
- λ_2 and λ_3 are inflated
This is the "orientation dispersion confound." D_DTI does NOT reflect the anisotropy of
individual axons — it reflects the average of all orientations within the voxel.

### 2.2 MD-dMRI (DPS) Acquisition and Fitting

**Files:**
- LTE: `NiiFiles/FullPD5_WIP_dMRI_Linear_medium_601.nii.gz` (44 vol)
- STE: `NiiFiles/FullPD5_WIP_dMRI_Spherical_medium_...501.nii.gz`
- Encoding structure: `output_directory/LTE_STE_PTE_xps.mat` (experimental parameter structure)

**Fitting framework:** md-dmri toolbox (Topgaard group), DPS model (Distribution of Particle
Shapes). Joint analysis of LTE + STE under the QTI (Q-space Trajectory Imaging) framework
(Westin et al., 2016). 38 volumes total, pilot dataset.

**DPS model outputs — all stored in `fit/dps.mat`:**

| Field | Shape | Range (brain) | Description |
|-------|-------|---------------|-------------|
| `ufa` | (96,96,48) | [0.000, 0.999] | μFA (microscopic FA), corrected estimate |
| `ufa_old` | (96,96,48) | [0.000, ~1.0] | μFA, older/alternative estimate; median 0.776 |
| `MD` | (96,96,48) | [0.005, 4.976] μm²/ms | Mean diffusivity of mean compartment tensor ⟨D⟩ |
| `FA` | (96,96,48) | [0, 1] | Macroscopic FA (from QTI mean tensor fit) |
| **`ad`** | (96,96,48) | [0.005, 5.008] μm²/ms | **Largest eigenvalue of ⟨D⟩** — "axial diffusivity" of mean compartment tensor |
| **`rd`** | (96,96,48) | [0.005, 4.976] μm²/ms | **Smallest eigenvalue of ⟨D⟩** — "radial diffusivity" |
| `FA_col` | (3,96,96,48) | [0, 254] | RGB visualization: FA × ‖u‖ × 255 — NOT eigenvectors |
| `signaniso` | (96,96,48) | {−1, 0, +1} | Tensor shape: +1 = prolate (58.4%), −1 = oblate (41.6%), 0 = spherical (<1%) |
| `vdiso` | (96,96,48) | (μm²/ms)² | Variance of isotropic diffusivity across compartments |
| `vsdaniso` | (96,96,48) | (μm²/ms)² | Variance of normalized anisotropy (related to C_µ) |
| `mdxx` | (96,96,48) | ~1.5×10⁻⁹ (SI) | Mean tensor D_xx — **POPULATED** in brain (SI m²/s → ×1e9 µm²/ms); NaN only outside mask. See Model 3. |
| `mdyy` | (96,96,48) | ~1.5×10⁻⁹ (SI) | Mean tensor D_yy — populated in brain |
| `mdzz` | (96,96,48) | ~1.5×10⁻⁹ (SI) | Mean tensor D_zz — populated in brain |
| `mdxy` | (96,96,48) | ~±10⁻¹⁰ (SI) | Mean tensor D_xy — populated in brain |
| `mdxz` | (96,96,48) | ~±10⁻¹⁰ (SI) | Mean tensor D_xz — populated in brain |
| `mdyz` | (96,96,48) | ~±10⁻¹⁰ (SI) | Mean tensor D_yz — populated in brain |

**Key distinction: ufa vs ad/rd**

These two describe different levels of the microstructural hierarchy:

- **μFA (ufa):** The ensemble-average per-compartment FA — "how elongated is the average
  individual axon?" Orientation-dispersion-invariant: a voxel with many parallel axons
  and one with equally many crossing axons at 90° will have the same μFA if the axons
  are identically shaped.

- **⟨D⟩ eigenvalues (ad, rd):** The eigenvalues of the mean compartment tensor — the
  average over all compartments' orientations. When multiple fiber orientations coexist,
  ⟨D⟩ is "smeared" across orientations, so ad/rd ratio is SMALLER than the
  per-compartment anisotropy implied by μFA. It sits between DTI (fully macroscopic,
  maximally orientation-confounded) and μFA (fully per-compartment).

**Note on the mean tensor components (CORRECTED):**  
Fields mdxx/mdyy/mdzz/mdxy/mdxz/mdyz are **populated and valid inside the brain mask** —
they hold the full mean diffusion tensor ⟨D⟩ in SI units (m²/s, ≈1.5×10⁻⁹), and are NaN
only *outside* the mask. An earlier reading mistook them for "zero/NaN" because the
SI-unit magnitudes (1.5×10⁻⁹) round to 0.000 at display precision. Multiplying by 10⁹
gives µm²/ms. The reconstructed 3×3 tensor is positive-definite in 100% of brain voxels,
its trace equals 3·MD exactly, its largest eigenvalue equals `ad`, its (λ₂+λ₃)/2 equals
`rd`, and its principal eigenvector equals `u` — i.e. it is internally consistent with the
other dps fields and genuinely triaxial (88%). This is the basis of the implemented
**Model 3** (full triaxial σ ∝ ⟨D⟩). No re-export or richer protocol was required.

**QTI fit/ folder outputs (from `fit/`):**

| File | Description | Note |
|------|-------------|------|
| `dtd_covariance_C_mu.nii.gz` | C_µ = normalized µFA variance | Has NaN outside mask; extreme values (+13,774 to −23,653) inside — unstable fit for 38-vol protocol |
| `dtd_covariance_MD.nii.gz` | MD from covariance fit | NaN outside mask |
| `dtd_covariance_C_MD.nii.gz` | Normalized MD variance (C_MD) | Isotropic variance component |
| `dtd_covariance_MKa.nii.gz` | Mean kurtosis, anisotropic | Sensitivity to µFA (Westin 2016) |
| `dtd_covariance_MKi.nii.gz` | Mean kurtosis, isotropic | Sensitivity to MD variance |
| `dtd_covariance_MKt.nii.gz` | Mean kurtosis, total | MKt = MKa + MKi |

**Eigenvector storage:**  
DPS stores exactly ONE eigenvector per voxel, regardless of tensor shape:
`u` = principal eigenvector = direction of LARGEST eigenvalue of ⟨D⟩.  
- For prolate (signaniso=+1): u = fiber axis (unique direction of max diffusivity) ✓  
- For oblate (signaniso=−1): u = one arbitrary direction in the equatorial disc plane; the
  disc normal (v3, direction of min eigenvalue) is NOT stored.  
This is a fundamental property of the DPS compartment model: ⟨D⟩ has cylindrical symmetry
(one unique eigenvalue ad, one degenerate eigenvalue rd), so only one eigenvector is
physically informative per voxel.

---

## Part III — Model Derivations

### Model 0: Isotropic Baseline (ISO) — Reference Condition

**D_input = MD_tissue · I**

No diffusion tensor is used. SimNIBS assigns the standard scalar conductivity to each
tissue type from its lookup table. This is the isotropic control.

- Conductivity: σ = σ_tissue · I
- No diffusion data required
- Voxel coverage: 100% by definition

---

### Model 1: DTI-Based Anisotropy (dwi2cond) — Macroscopic Triaxial

**Data:** LTE acquisition → FSL dtifit → lte_dti_tensor.nii.gz

**Signal model (Stejskal-Tanner):**

    S(b, g) = S_0 · exp(−b · g^T D_DTI g)

Fitting: minimize ‖log S_measured − log S_predicted‖² over D_DTI (6 free parameters).

**Conductivity tensor (Tuch 2001 + VN):**

    D_cond = D_DTI = λ_1 e_1⊗e_1 + λ_2 e_2⊗e_2 + λ_3 e_3⊗e_3

where {λ_1 ≥ λ_2 ≥ λ_3, e_1, e_2, e_3} are eigenvalues and eigenvectors of D_DTI.

**Triaxial:** YES — three distinct eigenvalues per voxel.

**Intra-voxel anisotropy (per-compartment):** NO. D_DTI is the voxel-averaged apparent
tensor, confounded by orientation dispersion.

**FullPD5 empirical results:**
- Median λ_1/λ_3 ratio: 1.66:1; λ_1/λ_geom: 1.414:1
- Fraction capped at 8:1: ~0% (ratios well within cap)

**SimNIBS call:**  
`s.anisotropy_type = 'vn'`, `s.fname_tensor = 'lte_dti_tensor.nii.gz'`

---

### Model 2a: DPS Prolate-for-All (FIRST MD-dMRI MODEL — ABANDONED)

**Historical note:** This was the original implementation of anisotropic MD-dMRI conductivity.
It is now known to be incorrect for oblate voxels (~41.6% of brain).

**Physical motivation:** Under the assumption that all compartments are prolate (rod-like,
axon-like), the mean compartment tensor ⟨D⟩ for a voxel with principal eigenvector u has
eigenvalues (λ_∥, λ_⊥, λ_⊥) where λ_∥ = ad, λ_⊥ = rd. 

The conductivity tensor is then:

    D_cond = (ad − rd) · u⊗u  +  rd · I

But in Model 2a, ad and rd were NOT used directly. Instead, they were derived by inverting
the μFA definition. For a cylindrically symmetric tensor with eigenvalues (λ_∥, λ_⊥, λ_⊥):

    MD   = (λ_∥ + 2λ_⊥) / 3
    μFA  = √(3/2) · |λ_∥ − λ_⊥| / √(λ_∥² + 2λ_⊥²)

Letting δ = λ_∥ − λ_⊥ and expressing in terms of MD and μFA:

    δ = 3·MD·μFA·√2 / √(9 − 4·μFA²)

    λ_∥ = MD + (2/3)·δ  =  MD · [1 + 2·μFA·√2 / √(9 − 4·μFA²)]
    λ_⊥ = MD − (1/3)·δ  =  MD · [1 − μFA·√2 / √(9 − 4·μFA²)]

For FullPD5 median μFA = 0.714:
    δ/MD ≈ 1.149  →  λ_∥ ≈ 1.766·MD,  λ_⊥ ≈ 0.617·MD  →  ratio ≈ 2.86:1 (exact formula)

Empirically computed ratio from the implementation was higher (median ~7.74:1), suggesting
the code used a simplified formula (e.g., λ_∥ = MD·(1+2·μFA), λ_⊥ = MD·(1−μFA)) that is
not the exact mathematical inverse of the μFA definition — a common simplified approximation.

**Critical bug — oblate inversion:**  
DPS stores u = direction of LARGEST eigenvalue for ALL voxels including oblate ones.  
For oblate voxels (signaniso=−1, disc-like, ~41.6% of brain):
- Physical meaning of u: one arbitrary direction in the disc plane (max diffusivity)
- Physical meaning of v3 (NOT stored): disc normal (min diffusivity direction)
- What the code does: assigns λ_∥ (large value ≈ 2.428·MD) along u

This is correct only if the voxel is prolate. For oblate voxels, u already has the large
eigenvalue (ad ≈ 2.67 μm²/ms), and the disc normal has the small eigenvalue (≈ 0.167 μm²/ms).
Ratio λ_normal/AD ≈ 0.062 — the assignment is roughly correct (max along max) but assigns
a prolate shape to what is an oblate tissue geometry.

More critically: μFA-derived eigenvalue ratio median ~7.74:1 → 48.7% of brain voxels
exceeded the 8:1 SimNIBS cap (aniso_maxcond=4), saturating nearly half the brain to
maximum allowed anisotropy.

**Summary of failures:**
1. Oblate voxels (~41.6%): geometrically wrong shape, though directionally consistent
2. 48.7% voxels capped: physically unrealistic, homogenizes E-field
3. μFA overestimated: 38-volume unconstrained QTI produces +0.1 μFA bias (Herberthson 2021,
   Kerkelä 2021); further inflated in subcortical GM (partial volume with free water)

---

### Model 2b: Three-Tier Oblate-Corrected (ATTEMPTED FIX — ABANDONED)

**Motivation:** Correct the oblate inversion error in Model 2a by handling the three tensor
shapes separately.

**Proposed formula:**

For prolate (signaniso=+1) with valid v1:
    D_cond = λ_∥·u⊗u + λ_⊥·(I − u⊗u)    [same as 2a]

For oblate (signaniso=−1) with valid v1:
    D_cond = RD_oblate·u⊗u + AD_oblate·(I − u⊗u)    [attempted inversion]

For spherical or invalid v1:
    D_cond = MD · I

**Why this fails:**  
For oblate voxels, the physically correct representation is:

    ⟨D⟩_oblate = λ_disc · (I − v3⊗v3)  +  λ_normal · v3⊗v3

where v3 is the disc normal (direction of MINIMUM eigenvalue) and λ_disc > λ_normal.
The inverted formula assigns max conductivity in the plane perpendicular to u, but u is
only ONE direction in the equatorial disc — any rotation within that plane is equally
valid. Without v3, we cannot orient the oblate tensor correctly.

Furthermore, empirically: RD_oblate/AD_oblate ≈ 0.062 for oblate voxels. The "inverted"
tensor (RD along u, AD in the perpendicular plane) still has a 16:1 ratio — worse than
the original. The cap problem is not resolved.

**The fundamental issue:** v3 is not stored in dps.mat. Any oblate correction model
requires the disc normal, which the DPS fitting stores but does not output for this dataset.

**Abandoned in favor of Model 2d (ad/rd), which accepts the bounded oblate error.**

---

### Model 2c: Two-Tier μFA Prolate + Isotropic Fallback (INTERMEDIATE — ABANDONED)

**Motivation:** Avoid the oblate directional error entirely by falling back to isotropic
for all non-prolate voxels.

**Formula:**

For prolate (signaniso=+1) with valid v1 (~55% of brain):
    D_cond = (λ_∥ − λ_⊥) · u⊗u  +  λ_⊥ · I
    with λ_∥, λ_⊥ derived from μFA and MD as in Model 2a

For oblate (signaniso=−1), spherical, or invalid v1 (~45% of brain):
    D_cond = MD · I    (isotropic fallback)

**Why abandoned:**  
- ~45% of brain is isotropic — the model does not provide meaningful anisotropy in oblate GM
- The μFA-to-eigenvalue derivation still produces 48.7% capping in the prolate fraction
- This was the "44.9% isotropic fallback" version that motivated the full redesign

---

### Model 2d: DPS Mean Compartment Tensor σ∝⟨D⟩ — cylindrical ad/rd (SUPERSEDED by Model 3)

> **Superseded.** This cylindrical form assumed the full mean-tensor components
> (mdxx..mdyz) were unavailable. They were in fact present in `dps.mat` all along —
> stored in SI units (~1.5×10⁻⁹ m²/s), which rounded to 0.000 at display precision and
> were mistaken for zero. Model 3 (below) uses the full triaxial ⟨D⟩ directly and
> **replaces this model in the repository.**

**Physical motivation:**  
The DPS model provides the mean compartment tensor ⟨D⟩ — the ensemble average of individual
compartment diffusion tensors. Unlike D_DTI (which averages signal across all compartments
and orientations), ⟨D⟩ preserves orientation-dependent per-compartment diffusivity while
still providing a single tensor per voxel. The eigenvalues (ad, rd) are stored directly
in dps.mat, eliminating the need to invert μFA.

This makes the model the orientation-dispersion-invariant analogue of the Tuch 2001 DTI
approach: σ ∝ ⟨D⟩ instead of σ ∝ D_DTI.

**Conductivity formula:**

For every brain voxel with valid principal eigenvector (|v1_norm| ≥ 0.5, ~97% of brain):

    D_cond = (ad − rd) · u⊗u  +  rd · I

where:
- **u** = principal eigenvector of ⟨D⟩ (direction of max diffusivity; stored in dps.mat)
- **ad** = largest eigenvalue of ⟨D⟩ (μm²/ms; stored directly in dps.mat)
- **rd** = smallest eigenvalue of ⟨D⟩ (μm²/ms; stored directly in dps.mat)

Note: (ad − rd) ≥ 0 is guaranteed by construction (DPS enforces ad ≥ rd). After
trilinear registration to T1 space, edge interpolation artefacts may produce rd > ad at
brain boundaries; these are corrected by clipping:
    rd = clip(rd, 0, ad)    (boundary correction, ~0 interior voxels affected)

For voxels with invalid v1 (~3% of brain; vecreg boundary artefacts where |v1_norm| < 0.5):

    D_cond = MD · I    (isotropic fallback using MD = (ad + 2·rd)/3)

**v1_norm validity:**  
After `vecreg` registration, the norm distribution of v1_T1 is bimodal with a clean gap:
- Interior brain voxels: |v1_norm| ∈ [0.990, 1.001) — valid unit vectors
- Boundary artefacts: |v1_norm| ∈ [0, 0.1) — registration failure at brain edge
- Empty gap: (0.1, 0.99) — no voxels in this range
The threshold 0.5 lies in the empty gap and is thus non-arbitrary (any value in (0.1, 0.99)
gives identical results).

**Tensor assembly — FSL dtifit component order [D_xx, D_xy, D_xz, D_yy, D_yz, D_zz]:**

    D_xx = (ad − rd)·u_x² + rd
    D_xy = (ad − rd)·u_x·u_y
    D_xz = (ad − rd)·u_x·u_z
    D_yy = (ad − rd)·u_y² + rd
    D_yz = (ad − rd)·u_y·u_z
    D_zz = (ad − rd)·u_z² + rd

where (u_x, u_y, u_z) are the components of u (unit vector).

**Note on cylindrical symmetry:**  
The DPS model is fundamentally cylindrically symmetric: the mean compartment tensor ⟨D⟩
has one unique eigenvector u and one degenerate eigenvalue rd in the perpendicular plane.
This is NOT a pipeline limitation — it is inherent to the DPS compartment model, because
⟨D⟩ is averaged over all orientations within the voxel and cylindrical averaging preserves
only the principal axis. Two distinct transverse eigenvalues (triaxiality) would require
the FULL mean tensor components mdxx/mdyy/etc., which are NaN for this dataset.

**Note on oblate voxels (~41.6% of brain):**  
u is the direction of LARGEST eigenvalue even for oblate (disc-like) voxels. The formula
assigns a prolate-shaped conductivity tensor (max conductivity along u). This is
geometrically incorrect (disc shape requires max conductivity in a plane, not along a line),
but the error is bounded:
- Oblate ad/rd median = 1.461:1 (prolate: 1.651:1; both well within 8:1 cap)
- The assignment is directionally consistent with Tuch 2001: max conductivity along
  max diffusivity direction
- The error is modest compared to the gross 16:1 inversion in Models 2a/2b

**FullPD5 empirical statistics:**

| Quantity | Value |
|----------|-------|
| Anisotropic voxels (valid v1) | 96.87% of brain |
| Isotropic fallback (invalid v1) | 3.13% |
| AD/RD ratio, all brain — median | 1.537:1 |
| AD/RD ratio, all brain — 95th percentile | 2.914:1 |
| AD range (brain) | [0.005, 5.008] μm²/ms |
| RD range (brain) | [0.005, 4.976] μm²/ms |
| Voxels exceeding 8:1 cap | 0.09% |
| DPS signaniso: prolate (+1) | ~58.4% |
| DPS signaniso: oblate (−1) | ~41.6% |
| Negative-definite tensors | 0 (by construction: ad ≥ rd ≥ 0) |

**Comparison to DTI model:**
- DTI λ_1/λ_geom = 1.414:1 (macroscopic, orientation-confounded)
- DPS ad/rd = 1.537:1 (mean compartment, partially orientation-corrected)
- DPS ad > DTI λ_1 and DPS rd < DTI λ_3 (consistent with orientation dispersion
  deflating the DTI primary eigenvalue and inflating the secondary/tertiary)

**SimNIBS call:**  
`s.anisotropy_type = 'vn'`, `s.fname_tensor = 'tensor_MD_dMRI.nii.gz'`  
Output shape: (180, 239, 239, 6), float32, FSL dtifit order.

**Positive-definiteness proof:**  
det(D_cond) = ad · rd² ≥ 0 since ad ≥ rd ≥ 0 (from DPS + clipping). Equality to zero
only at brain boundary voxels with ad = rd = 0 (outside-mask zeros), which SimNIBS
handles via tissue segmentation masking.

---

### Future Model 2e: QTI Mean Tensor Triaxial (IDEAL — NOT YET AVAILABLE)

**What it would provide:**  
If the full mean tensor components (mdxx, mdyy, mdzz, mdxy, mdxz, mdyz) were valid (non-NaN),
we could build a fully triaxial, orientation-dispersion-invariant conductivity tensor:

    D_cond = ⟨D⟩ = sum over full symmetric 3×3 tensor
           = [mdxx  mdxy  mdxz]
             [mdxy  mdyy  mdyz]
             [mdxz  mdyz  mdzz]

Diagonalize → three eigenvalues (λ_1 ≥ λ_2 ≥ λ_3) and three eigenvectors (e_1, e_2, e_3).
These eigenvalues would reflect per-compartment anisotropy WITHOUT orientation dispersion
confound, AND without cylindrical symmetry constraint — giving three distinct values like
DTI but at the compartment level.

This would be the ideal model: intra-voxel anisotropy + triaxial shape + OD-invariant.

**Status:** mdxx/mdyy/mdzz/mdxy/mdxz/mdyz = NaN in FullPD5 dps.mat. Requires re-running
md-dmri fit with mean tensor computation enabled, or a richer acquisition protocol.

---

### Future Model 3: LTE-DTI Triaxial (AVAILABLE, NOT YET RUN)

The tensor from the LTE-DTI fit (`registration/lte_dti/lte_dti_tensor.nii.gz`) could be
passed to SimNIBS directly:

    D_cond = D_LTE-DTI = λ_1·e_1⊗e_1 + λ_2·e_2⊗e_2 + λ_3·e_3⊗e_3

This is identical in concept to the dwi2cond approach (Model 1) but uses the LTE
acquisition fitted in-house. Would serve as a cross-check. Macroscopic, triaxial,
orientation-confounded — same limitations as Model 1.

---

## Part IV — Model Comparison Table

| Model | Tensor type | Triaxial? | Intra-voxel (OD-invariant)? | Coverage (anisotropic) | % Cap (>8:1) | Status |
|-------|------------|-----------|----------------------------|-----------------------|-------------|--------|
| ISO | Isotropic | — | — | 0% | 0% | ✅ Control |
| DTI (dwi2cond) | Macroscopic apparent | YES (3 eigenvalues) | NO (OD-confounded) | ~100% WM/GM | ~0% | ✅ Implemented |
| 2a: Prolate-for-all (μFA) | Cylindrical | NO | YES (per-compartment) | 96.9% | **48.7%** | ❌ Abandoned |
| 2b: Three-tier oblate | Cylindrical | NO | YES (prolate) / wrong (oblate) | 96.9% | **48.7%** | ❌ Abandoned (no v3) |
| 2c: Two-tier μFA+ISO | Cylindrical | NO | YES (prolate only) | ~55% | **48.7%** prolate | ❌ Abandoned (45% iso) |
| **2d: ⟨D⟩ ad/rd** | **Cylindrical** | **NO** | **YES (OD-corrected)** | **96.9%** | **0.09%** | **✅ CURRENT** |
| 2e: QTI mean tensor | Full symmetric 3×3 | YES | YES (OD-invariant) | ~97% | TBD | ⛔ NaN data |
| LTE-DTI (Model 3) | Macroscopic apparent | YES | NO (OD-confounded) | ~100% | ~0% | 🔲 Available, not run |

**Eigenvalue ratio hierarchy (FullPD5):**  
ISO 1:1 < DTI λ_1/λ_geom 1.414:1 < DPS ad/rd 1.537:1 < μFA-derived (correct formula) ~2.86:1

The DTI < DPS ordering is physically expected: DTI eigenvalues are smeared by orientation
dispersion (reducing apparent anisotropy), while ⟨D⟩ eigenvalues partially correct for this.

---

## Part V — E-field Results (FullPD5)

SimNIBS VN simulation results, E-field magnitude median per ROI (V/m):

| ROI | ISO | DTI | DTI/ISO | MD-dMRI (⟨D⟩) | MD-dMRI/ISO |
|-----|-----|-----|---------|---------------|------------|
| L_SNc | 0.406 | 0.422 | 1.039 | 0.428 | 1.054 |
| R_SNc | 0.397 | 0.412 | 1.036 | 0.409 | 1.030 |
| L_SNr | 0.304 | 0.298 | 0.982 | 0.308 | 1.015 |
| R_SNr | 0.324 | 0.317 | 0.979 | 0.325 | 1.004 |
| L_VTA | 0.341 | 0.336 | 0.984 | 0.347 | 1.018 |
| R_VTA | 0.368 | 0.367 | 0.999 | 0.377 | 1.024 |
| L_PUT | 0.304 | 0.326 | 1.073 | 0.320 | 1.051 |
| R_PUT | 0.250 | 0.231 | 0.924 | 0.237 | 0.947 |
| L_Ca | 0.229 | 0.232 | 1.015 | 0.234 | 1.022 |
| R_Ca | 0.317 | 0.315 | 0.992 | 0.318 | 1.002 |
| L_NAC | 0.227 | 0.254 | 1.120 | 0.231 | 1.018 |
| R_NAC | 0.336 | 0.332 | 0.989 | 0.331 | 0.985 |
| L_GPi | 0.335 | 0.339 | 1.011 | 0.334 | 0.997 |
| R_GPi | 0.365 | 0.378 | 1.034 | 0.362 | 0.990 |
| L_GPe | 0.362 | 0.390 | 1.076 | 0.358 | 0.987 |
| R_GPe | 0.346 | 0.340 | 0.982 | 0.331 | 0.955 |
| L_RN | 0.331 | 0.315 | 0.952 | 0.327 | 0.988 |
| R_RN | 0.367 | 0.349 | 0.953 | 0.361 | 0.985 |

**Summary statistics across 18 ROIs:**
- DTI/ISO: range [0.924, 1.120], mean ≈ 1.000 (symmetric around ISO)
- MD-dMRI/ISO: range [0.947, 1.054], narrower spread than DTI

The MD-dMRI model produces more conservative anisotropy effects (1.537:1 ratio) than DTI
in some regions (especially caudate, GP, RN) but slightly larger effects than DTI in
subcortical targets (SNc, VTA). This is consistent with ⟨D⟩ providing a more accurate
voxel-scale anisotropy estimate than macroscopic DTI.

---

## Part VI — Key Scientific Questions for Review

1. **Is the VN normalization correctly matched to ⟨D⟩ units?**  
   SimNIBS VN: σ = σ_tissue · D/det(D)^(1/3). Since D is scale-normalized, units cancel.
   We pass tensor in μm²/ms; SimNIBS normalizes by det, so units do not affect the result.
   VERIFIED: `s.anisotropy_type = 'vn'` → SimNIBS takes shape only.

2. **Is u from dps.mat in the same coordinate system as the FEM mesh after vecreg?**  
   vecreg applies the affine rotation from the FLIRT dMRI-to-T1 matrix to the eigenvectors,
   correctly rotating the direction vectors (unlike flirt -applyxfm which only resamples
   values). After vecreg, v1_T1 is in T1 anatomical space, matching the CHARM FEM mesh.

3. **Why does the ad/rd model give larger E-field in SNc/VTA but smaller in GP/RN vs ISO?**  
   This likely reflects real differences in local fiber orientation relative to the
   applied E-field direction (C3→Fp2). The VN normalization preserves tissue-average
   conductivity, so anisotropic redirection of current flow produces local increases
   where fibers align with the field and decreases where they are perpendicular.

4. **Is the cylindrical symmetry constraint (no triaxiality) a significant limitation?**  
   For WM: less so — individual axons ARE cylindrically symmetric to a good approximation.
   For GM: potentially more significant — cells and dendrites may have genuinely oblate
   or irregular shapes. However, the oblate error in our model (ad/rd ≈ 1.46:1) is small.
   The ideal correction would require valid mean tensor components (mdxx etc.) — Future Model 2e.

5. **How does 38-volume QTI underdetermination affect ad and rd (not just μFA)?**  
   μFA overestimation from sparse QTI is well documented (Herberthson 2021, Kerkelä 2021).
   Whether ad and rd are similarly biased is less clear from the literature. If the DPS
   mean tensor fit is regularized (as typically implemented), ad/rd may be underestimated
   relative to the true per-compartment eigenvalues, providing a conservative anisotropy
   estimate. This is a potential systematic bias not yet quantified.

---

## Model 3: MD-dMRI Conductivity Model — σ ∝ ⟨D⟩ (FINAL — IMPLEMENTED)

This is **the MD-dMRI model**. Like the DTI model, its conductivity tensor is fully
triaxial (three distinct eigenvalues / eigenvectors); "triaxial" is a property of the
tensor, not a separate model name. The distinction from the DTI model is the *source* of
the tensor: the QTI mean diffusion tensor ⟨D⟩ from LTE+STE b-tensor encoding, rather than
the single-shell mono-exponential DTI tensor (see *Novelty / accuracy* below).

**Key data correction.** The QTI fit (`dps.mat`) stores the full mean diffusion tensor as
six fields `mdxx, mdyy, mdzz, mdxy, mdxz, mdyz`, in SI units (m²/s, ≈1.5×10⁻⁹). At display
precision these rounded to 0.000 and were earlier mistaken for "not computed." They are
valid; multiplying by 10⁹ gives µm²/ms. This provides the full mean tensor directly,
removing the cylindrical limitation of Model 2d.

**Verification (130,643 brain voxels):**

| Check | Result |
|---|---|
| trace(⟨D⟩)/3 = MD | exact (confirms it is the mean tensor) |
| λ₁(⟨D⟩) = ad | r = 0.9999 (element order confirmed) |
| (λ₂+λ₃)/2 = rd | r = 1.0000 |
| principal eigenvector = dps.u | median 0.0° |
| positive-definite | 100% |
| genuinely triaxial (λ₂≠λ₃) | 88% (median in-plane index 0.51) |

**Model.** For every brain voxel,

    σ ∝ ⟨D⟩ = [[mdxx, mdxy, mdxz],
               [mdxy, mdyy, mdyz],
               [mdxz, mdyz, mdzz]] × 10⁹   (µm²/ms)

passed to SimNIBS with `anisotropy_type='vn'` (Güllmar 2010 / Rullmann 2009 volume
normalization, geometric mean preserved). This is the Tuch (2001) effective-medium mapping
(σ and ⟨D⟩ share eigenvectors; σ-anisotropy ratio = ⟨D⟩-eigenvalue ratio) applied to the
QTI LTE+STE mean tensor — a more faithful realization of the ensemble-averaged
effective-medium tensor than single-shell DTI.

**Eigenvalues (T1 space, covered brain):** λ₁/λ₂/λ₃ median 2.05/1.53/1.00 µm²/ms;
anisotropy λ₁/λ₃ median 1.89 (p95 4.10); 1.1% above the 8:1 VN cap.

**Agreement with the conductivity-mapping literature.**

| Principle | Tuch 2001 | Güllmar 2010 / Rullmann 2009 | Model 3 |
|---|---|---|---|
| σ, D share eigenvectors | ✓ | ✓ | ✓ (σ ∝ ⟨D⟩) |
| eigenvalue mapping | linear σ = 0.844·d | volume-normalized | ✓ SimNIBS vn |
| σ-anisotropy = D-anisotropy | ✓ | ✓, capped ~10:1 | ✓ 1.89 median, cap 8:1 |

**Advantages over Model 2d (cylindrical ad/rd):**
- retains the middle eigenvalue (λ₂≠λ₃) — the cylindrical model discards it via rd=(λ₂+λ₃)/2;
- gives the 42% oblate voxels (signaniso=−1) their correct disc shape (λ₁≈λ₂>λ₃) instead of
  a geometrically wrong prolate;
- uses the actual fitted ⟨D⟩ rather than a 2-eigenvalue approximation;
- higher whole-brain anisotropy (1.89 vs 1.52) → more directed current.

**Registration (proper-handling notes).**
- Upstream motion/eddy correction was done by the md-dmri toolbox (extrapolation-based,
  Nilsson 2015, `mdm_s_mec`) — the data we consume (`LTE_STE_PTE_mc`) is already corrected.
- dMRI→T1 (this pipeline) uses FSL, on a grid verified identical across the QTI fit space
  and the registration source (Spherical b0): same affine, 96×96×48, 2.5 mm iso.
- **Eigenvalues** λ₁,λ₂,λ₃ are registered as *scalars* (FLIRT trilinear) to preserve
  anisotropy magnitude; whole-tensor interpolation alone dilutes it (1.92→1.28 observed).
- **Orientation**: principal axis anchored to the validated `v1_T1` (vecreg; 18° median vs
  dwi2cond V1 in core WM); in-plane v₂,v₃ taken from the vecreg-reoriented tensor frame.
  Anchoring removes a residual 8°-median two-registration discrepancy (≈1% conductivity
  error at the median, p95 ≈ 20% in low-anisotropy voxels).

**ROI E-field result (C3→Fp2, p95).** Model 3 tracks the gold-standard dwi2cond DTI within
±3% in deep WM structures (SNc, VTA, NAc, GPe) while raising the field +6–10% over isotropic.
Larger divergences (L_PUT −26%, R_GPe −14%) occur where single-shell DTI and the
dispersion-aware QTI mean tensor genuinely differ (free water / crossing fibres).

**Pipeline:** `01d_save_triaxial_tensor.py` → `01_register_dMRI_to_T1.sh`
(FLIRT λ₁/λ₂/λ₃ + vecreg v1, tensor) → `02_build_conductivity_tensor.py` (anchored
reconstruction → `tensor_MD_dMRI.nii.gz`) → `03_run_mddmri_only.py` → `04_extract_roi_efield.py`.

### Model 3 — Novelty and accuracy (vs the DTI model)

**Is this an intra-voxel (microscopic) anisotropy model?** No — and deliberately so, for a
precise physical reason. ⟨D⟩ is the *macroscopic* mean diffusion tensor (the ensemble
average over compartment orientations), the same conceptual level as the DTI tensor. Tuch's
effective-medium relation σ ∝ D is a macroscopic statement: current at the FEM voxel scale
responds to the applied field through the *effective-medium* conductivity, which follows the
orientation-averaged (mean) tensor.

The microscopic anisotropy μFA (median 0.76 here, implying a ≈3.2:1 axial ratio vs ⟨D⟩'s
≈1.54:1) is tempting as "the unique MD-dMRI advantage," but it is the *wrong* quantity for
conductivity. The key argument: μFA exceeds the macroscopic FA *only* in voxels with
orientation dispersion or fibre crossing — that is the sole condition where micro ≠ macro.
But those are exactly the voxels with **no coherent current direction**: in a crossing-fibre
voxel current can take either population, so the effective conductivity is near-isotropic
(which ⟨D⟩ gives), whereas μFA would impose a large anisotropy along an arbitrary axis that
no single compartment population defines. Microscopic anisotropy is therefore largest exactly
where it is least usable for conductivity. This is why Tuch (2001), Güllmar (2010),
Rullmann (2009) and dwi2cond all use the macroscopic tensor. A μFA-based model would be more
"novel" but physically incorrect and unvalidated — it cannot be as rigorous as dwi2cond.
We use μFA only as a QA/reporting scalar, never to build σ.

A fully microstructural conductivity model (mapping each compartment's diffusion tensor to a
per-compartment conductivity and homogenising to an effective tensor) is conceivable and
would be the largest possible novelty, but it requires (i) reliable per-compartment
orientations — which the DTD export here does not provide (Monte-Carlo orientations are
30–52° unreliable, see Model 2 discussion), and (ii) unvalidated assumptions about
per-compartment conductivities and the homogenisation scheme. It is out of scope for a
model intended to match dwi2cond's rigour.

**Then what is the novelty vs the DTI model?** The *source and bias* of the tensor, not the
conductivity physics (which is the validated Tuch / Güllmar / Rullmann mapping for both):

- **DTI tensor**: apparent diffusion tensor from a single-shell, mono-exponential fit.
  Biased by non-Gaussian diffusion (diffusional kurtosis) and inflated by free-water
  partial volume.
- **MD-dMRI ⟨D⟩**: the *first cumulant* of the diffusion tensor distribution, estimated
  jointly with the covariance (kurtosis) from LTE+STE b-tensor encoding (Westin 2016;
  Topgaard 2017). It is the kurtosis-corrected, free-water-disentangled estimate of the
  same macroscopic mean tensor — i.e. a less-biased input to the Tuch mapping.

**Empirical accuracy signature.** Across the whole brain the two anisotropy models deviate
from ISO by ~5% (median; p95 ~17–20%) and differ *from each other* by a comparable amount
(~5–6% median, p95 ~20%; element-wise, shared mesh) — so the choice of diffusion model
matters about as much as whether to model anisotropy at all. Crucially, the MD-dMRI–vs–DTI
E-field divergence **rises with diffusional kurtosis** (median |Δ| 4.1%→6.6% from the
lowest to highest MKt quartile; corr ≈ +0.20) and is *smallest* in high free-water/CSF
voxels (where both models correctly collapse to isotropic). That is the predicted
signature: the two tensors agree in Gaussian tissue and diverge where DTI's mono-exponential
assumption fails — precisely where the kurtosis-corrected ⟨D⟩ is the correction. This is
consistent with the MD-dMRI tensor being the more accurate input, though no in-vivo
conductivity ground truth exists, so the accuracy claim is theoretically motivated rather
than directly validated.

**Registration note (re: the md-dmri / Olsson 2025 methodology).** The md-dmri toolbox and
the Olsson 2025 pipeline register *scalar* parameter maps to T1 (FreeSurfer) and rotate
b-vectors for motion correction (`mdm_mec_rotate_bvec`) — they never interpolate a diffusion
*tensor*. Our conductivity model is the first step here to need a registered tensor, and it
follows the same principle: eigenvalues are moved as scalars (no anisotropy dilution) and
the principal axis is carried as a *direction* (vecreg, anchored to the validated v1_T1),
rather than resampling the tensor as a whole. The principal-axis anchoring aligns ⟨D⟩'s
largest-eigenvalue direction with the validated v1_T1 (the previous ~8° offset was an
artifact of two independent tensor/vector interpolations, not a physical direction; anchoring
removes it and, if anything, improves the principal direction since v1_T1 is the validated one).

**Method difference vs dwi2cond — disclosed for transparency.** dwi2cond registers the DTI
tensor with straight `vecreg` (whole-tensor trilinear interpolation), which *dilutes* the
anisotropy when upsampling to the 1 mm FEM grid: the dwi2cond DTI tensor sits at median
λ1/λ3 ≈ 1.42 in T1 space. Our eigenvalue-as-scalar approach preserves the measured tissue
anisotropy, leaving the MD-dMRI tensor at median λ1/λ3 ≈ 1.89. The scalar approach is the
more physically faithful one (trilinear tensor interpolation is a known anisotropy-dilution
artifact, not a feature), but the two models are therefore registered by *different* tensor-
resampling schemes. Consequently, the MD-dMRI-vs-DTI E-field difference partly reflects this
registration choice in addition to the underlying diffusion-model difference. This is a
deliberate, documented choice (preserve the measured anisotropy rather than replicate
dwi2cond's interpolation dilution); a strict method-matched comparison would register both
tensors identically. The MD-dMRI-vs-ISO contrast is unaffected by this caveat.

---

## Model 4: Free-water-eliminated MD-dMRI (multi-compartment, σ ∝ ⟨D⟩_tissue)

**The genuine intra-voxel contribution.** The QTI tensor-distribution fit decomposes each
voxel into 3 fixed-diffusivity compartments ("bins") with signal fractions f_k (Σf_k=1) and
per-compartment tensors — a decomposition single-shell DTI fundamentally cannot make
(separating compartments by SHAPE/μFA requires the spherical STE encoding):

| bin | MD (µm²/ms) | f (median) | character |
|---|---|---|---|
| 0 | 1.24 | 0.35 | anisotropic tissue (WM-like, λ1/λ3≈4.2) |
| 1 | 0.62 | 0.25 | restricted tissue (GM-like) |
| 2 | 3.34 (2.68–4.51) | 0.24 | **free water / CSF** (f2↔MD r=0.89) |

**Model.** Free-water elimination (Pasternak 2009, generalised to the QTI compartments):
remove the free-water compartment and renormalise over tissue, then Tuch + VN:

    ⟨D⟩_tissue = (f0·D0 + f1·D1)/(f0+f1) ;   σ ∝ ⟨D⟩_tissue   (SimNIBS 'vn')

Principal axis anchored to the validated v1_T1 (same as Model 3), so only the eigenvalues
(the free-water-corrected anisotropy) differ — isolating the free-water effect.
Where f0+f1 < 0.30 (near-pure CSF) the model falls back to ⟨D⟩ so a tiny noisy tissue
fraction cannot create spurious anisotropy.

**Scientific correctness.** This is a de-biasing correction, not an anisotropy inflation:
- Free-water elimination is an established diffusion method; the QTI/STE separation is more
  robust than bi-exponential FWE-DTI because the isotropic component is measured directly.
- The bins are fixed-range (Topgaard); bin 2 is consistently the CSF compartment, so removal
  is well-defined and identical across voxels.
- SimNIBS models *bulk* CSF as a separate tissue; for a WM/GM-labelled element the dMRI tensor
  should carry the *tissue* conductivity shape. CSF partial-volume biases ⟨D⟩ toward isotropy;
  FWE removes that bias. Consistent with Tuch's tissue rationale and SimNIBS's tissue model.
- Behaviour: a **no-op where there is no free water** (Δλ1/λ3≈0 for f_FW<0.1), recovering
  tissue anisotropy only where CSF contaminates (Δ rises to +1.67 in λ1/λ3 for f_FW>0.35).
  The goal is a *more accurate*, de-biased tensor — not a larger E-field.

**Empirical effect (vs Model 3, σ∝⟨D⟩).** Anisotropy 1.89→2.06; whole-brain E-field |Δ|
~1.8% median (p95 ~7%); largest and positive in the high-CSF deep targets (SNc f_FW≈0.42–0.52:
+0.8–1.5%), negligible elsewhere — i.e. the correction lands where partial volume is worst,
which is exactly the PD targets (SN/VTA). Within the VN framework only the *shape* change is
captured; the additional effect of CSF's high *conductivity magnitude* would require a direct
(non-VN) mapping — noted as future work.

**Caveats / limitations.** (i) FWE assumes the free-water compartment should be excluded from
the WM/GM element shape; (ii) bin 2 has mild residual anisotropy (fit noise) — forced removal
is conservative; (iii) not directly validated for conductivity (roadmap: MRCDI/MREIT,
Gregersen 2024); (iv) acquisition-resolution partial volume at small nuclei (see below).

**Data-coverage note.** The DTD fit (`dps.mat`, source of all compartment tensors/bins) is
defined on a 130,643-voxel brain mask and is empty (NaN/zero) outside it at all scales. The
*covariance* maps (C_mu, MD, MKa/i/t) used a slightly wider mask (133,577 voxels), so ~3,904
voxels carry valid scalar QTI metrics beyond the DTD mask — but no tensors, so they do not
extend the conductivity tensor coverage.

---

## Acquisition-resolution limitation (for Methods / Limitations / Future directions)

Source: Olsson et al. MRE-PD dataset, Philips Ingenia CX 3T, single session.

| modality | native resolution | constraint |
|---|---|---|
| T1w / T2w / QSM | 1.0 mm iso | structural/segmentation grid |
| NMI (neuromelanin) | 0.6 mm in-plane, 1.3 mm slice | — |
| **MD-dMRI** | **2.5 mm iso** | SNR-vs-time (TE=111 ms; LTE+STE; 6.5 min) |
| MRE | 3.0 mm iso | shear-wave-physics floor (~25–30 mm wavelength at 60 Hz) |

The 2.5/3.0 mm figures are **not** a scanner spatial-encoding limit — the same scanner/session
resolves 1 mm (T1/T2/QSM) and 0.6 mm in-plane (NMI). The coarseness is a contrast-mechanism +
scan-time trade-off:
- **MD-dMRI @ 2.5 mm**: TE already long (STE waveforms + b=2000) → heavy T2 decay, SNR-limited.
  SNR ∝ voxel volume: 2.5→2.0 mm ≈ halves SNR → ~4× averaging → ~25 min; 1.5 mm → hours;
  1 mm-class QTI infeasible. Finer res also lengthens the EPI echo train → longer TE / worse
  distortion. ~2 mm is reachable at large time cost; 1 mm is not, on this acquisition class.
- **MRE @ 3.0 mm**: at the/below the meaningful mechanical resolution (3 mm already oversamples
  a ~25–30 mm shear wavelength). Finer voxels add noise, not mechanical information; only higher
  drive frequency would help, at the cost of deep penetration (bad for SN/STN).

**Implication for the conductivity pipeline.** Conductivity tensors inherit the 2.5 mm dMRI
grid and are always coarser than the 1 mm structural/segmentation grid — not fixable by a
scanner setting on this data. PD targets (substantia nigra, STN) are small; 2.5 mm voxels
straddle their boundaries, so partial volume biases both segmentation and the per-voxel tensor
exactly at the targets. **Mitigations applied:** tensor-aware registration to T1 (eigenvalues
as scalars, direction via vecreg/anchoring), free-water elimination (Model 4) to de-bias the
CSF partial-volume contribution, and the FT_MIN fallback near pure CSF. **Reporting:** per-voxel
tensors in/near SN/STN should not be over-trusted; partial volume is a stated limitation. A
future acquisition could reach ~2 mm dMRI at higher scan-time cost, but never the 1 mm grid;
this is intrinsic to the FullPD cohort, not a processing choice.
