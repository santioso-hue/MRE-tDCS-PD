# References — MRE_tDCS_PD Project

Organized by role in the project. All citations are used in scripts, docs, or manuscript methods.

---

## WM Conductivity Anisotropy — Empirical Basis

**Nicholson PW (1965).** Specific impedance of cerebral white matter.
*Experimental Neurology* 13:386–401.
→ **Primary source for 7–10:1 WM conductivity anisotropy.** Ex vivo AC impedance measurement of corpus callosum. Reported ~9:1 longitudinal:transverse conductivity **ratio**. Justifies our `aniso_maxratio` cap of 10:1 (the eigenvalue-RATIO cap; SimNIBS default 10). NB: `aniso_maxcond` is a separate eigenvalue-MAGNITUDE cap (S/m), not a ratio.

**Ranck JB Jr, BeMent SL (1965).** The specific impedance of the dorsal columns of cat: an anisotropic medium.
*Experimental Neurology* 11:451–463.
→ Ex vivo AC impedance, spinal cord WM. Found ~7:1 anisotropy. Companion to Nicholson 1965 as empirical foundation of the 7–10:1 range.

**Haueisen J, Ramon C, Eiselt M, Brauer H, Nowak H (1997).** Influence of tissue resistivities on neuromagnetic fields and electric potentials studied with a finite element model of the head.
*IEEE Transactions on Biomedical Engineering* 44(8):727–735.
→ Used 10:1 WM anisotropy in FEM head model, citing Nicholson 1965. Early precedent for using ex vivo ratios in electromagnetic simulation.

---

## DTI–Conductivity Effective Medium Theory

**Tuch DS, Wedeen VJ, Dale AM, George JS, Belliveau JW (2001).** Conductivity tensor mapping of the human brain using diffusion tensor MRI.
*PNAS* 98(20):11697–11701.
→ **Foundational theoretical paper.** Derived σ ∝ D from Maxwell-Garnett effective medium / tortuosity theory. Basis for dwi2cond and our MD-dMRI conductivity approach alike. Note: the 7–10:1 range is NOT from this paper — Tuch reports in vivo DTI ratios of 2–4:1. Cite Nicholson 1965 and Ranck 1965 for the 7–10:1 range.

**Opitz A, Paulus W, Will S, Thielscher A (2015).** Deterministic and probabilistic analysis of the electric field during transcranial direct current stimulation.
*NeuroImage* 109:266–279.
→ dwi2cond methodology and validation in SimNIBS. Confirms σ ∝ D relationship and volume-normalized (VN) conductivity scheme.

**Güllmar D, Haueisen J, Reichenbach JR (2010).** Influence of anisotropic electrical conductivity in white matter tissue on the EEG/MEG forward and inverse solution.
*NeuroImage* 51(1):145–163.
→ VN conductivity formula: σ = σ₀ · D / det(D)^(1/3). Removes the absolute diffusivity scale; only eigenvalue ratios matter.

---

## SimNIBS Methodology

**Windhoff M, Opitz A, Thielscher A (2013).** Electric field calculations in brain stimulation based on finite elements: an optimized processing pipeline for the generation of accurate individual head models.
*Human Brain Mapping* 34(4):923–935.
→ Main SimNIBS methodology paper. Eigenvalue clipping discussed as numerical stability + physiological plausibility measure. Two caps: `aniso_maxratio` (ratio, default 10) and `aniso_maxcond` (magnitude, default 2 S/m); both empirically calibrated.

**Thielscher A, Antunes A, Saturnino GB (2015).** Field modeling for transcranial magnetic stimulation: A useful tool to understand the physiological effects of TMS?
*Engineering in Medicine and Biology Society (EMBC)*, 37th Annual Conference of the IEEE.
→ SimNIBS 2015 release paper. Tetrahedral FEM mesh generation and solver.

---

## Tissue Conductivity Values

**Gabriel S, Lau RW, Gabriel C (1996).** The dielectric properties of biological tissues: III. Parametric models for the dielectric spectrum of tissues.
*Physics in Medicine & Biology* 41(11):2271–2293.
→ Comprehensive tissue conductivity measurements. Source for isotropic baseline values used in SimNIBS ISO model: WM=0.126 S/m, GM=0.275 S/m, CSF=1.654 S/m, skull=0.010 S/m, scalp=0.465 S/m.

---

## MD-dMRI / QTI Framework

**Westin CF, Knutsson H, Pasternak O, Szczepankiewicz F, Özarslan E, van Westen D, Nilsson M (2016).** Q-space trajectory imaging for multidimensional diffusion MRI of the human brain.
*NeuroImage* 135:345–362.
→ The QTI framework underlying the `fit/` outputs. Defines μFA (C_mu in code) and the covariance tensor formalism.

**Topgaard D (2017).** Multidimensional diffusion MRI.
*Journal of Magnetic Resonance* 275:98–113.
→ DPS (Diffusion Propagator Spectroscopy) model. Single mean compartment tensor fit under prolate (axially symmetric) constraint: D_ij = (λ₁−λ₂)·v1_i·v1_j + λ₂·δ_ij. Outputs μFA, MD, v₁. This is the analysis framework Christoffer's group uses. NOT a SimNIBS concept.

**Lasič S, Szczepankiewicz F, Eriksson S, Nilsson M, Topgaard D (2014).** Microanisotropy imaging: quantification of microscopic diffusion anisotropy and orientational order parameter by diffusion MRI with magic-angle spinning of the q-vector.
*Frontiers in Physics* 2:11.
→ Definition and estimation of μFA. The key insight: μFA measures intra-compartment anisotropy independent of fiber orientation dispersion, unlike DTI FA.

**de Almeida Martins JP, Topgaard D (2016).** Two-dimensional correlation of isotropic and directional diffusion using NMR.
*Physical Review Letters* 116:087601.
→ 2D MD-FA correlation: precursor to QTI and DPS; theoretical basis for separating isotropic and anisotropic diffusion contributions.

---

## Free-Water Elimination

**Pasternak O, Sochen N, Gur Y, Intrator N, Assaf Y (2009).** Free water elimination and mapping from diffusion MRI.
*Magnetic Resonance in Medicine* 62(3):717–730.
→ FWE removes the isotropic free-water compartment. **It defines the canonical MD-dMRI model** (`01e`/`02`),
generalised to the QTI 3-compartment fit: ⟨D⟩_tissue = (Σ_{k≠FW} f_k D_k)/(Σ_{k≠FW} f_k), raising the median
⟨D⟩ anisotropy from 1.92 to 2.13 in this subject. Plain ⟨D⟩ (no FWE) is the `--meanD` sensitivity.

---

## ROI Analysis Methods

**Saturnino GB, Siebner HR, Thielscher A, Madsen KH (2019).** Accessibility of cortical regions to focal TES: Dependence on spatial position, safety, and practical constraints.
*NeuroImage* 203:116183.
→ 5mm spherical ROI approach around MNI coordinates — used by SimNIBS group for CORTICAL targets. We follow this only for the cortical M1 reference; the subcortical ROIs are anatomical atlas masks (below), not spheres.

**Huang Y, Liu AA, Lafon B, Friedman D, Dayan M, Wang X, Bikson M, Doyle WK, Devinsky O, Parra LC (2017).** Measurements and models of electric fields in the in vivo human brain during transcranial electrical stimulation.
*eLife* 6:e18834.
→ Direct E-field measurements in vivo. Recommends 95th percentile statistic over mean or maximum for robust ROI characterization. Justification for reporting p95 alongside mean/median.

**Antonenko D, Thielscher A, Saturnino GB, Nierhaus T, Meinzer M, Prehn K, Flöel A (2019).** Towards precise brain stimulation: Is electric field simulation related to neuromodulation?
*Brain Stimulation* 12(5):1159–1168.
→ Atlas-mask approach for cortical ROI E-field extraction using FreeSurfer parcellations.

---

## ROI Definitions — Subcortical Atlases

**Pauli WM, Nili AN, Tyszka JM (2018).** A high-resolution probabilistic in vivo atlas of human subcortical brain nuclei (CIT168).
*Scientific Data* 5:180063.
→ **The atlas used for the midbrain ROI** (`06_build_atlas_rois.sh`): CIT168 SNc+SNr+VTA are merged into one SN/VTA mask, warped from MNI152-2009c to subject space. The basal-ganglia ROIs (caudate, putamen, pallidum) come from the HarvardOxford-Subcortical atlas instead. Olsson 2025 also uses CIT168.

**Ewert S, Plettig P, Li N, Bhatt M, Bhatt DL, Kühn AA, Volkmann J, Horn A (2018).** Toward defining deep brain stimulation targets in MNI space: A subcortical atlas based on multimodal MRI, histology and structural connectivity.
*NeuroImage* 170:271–282.
→ DISTAL atlas — a high-quality brainstem alternative for the dopaminergic nuclei; evaluated but not used in the final pipeline (CIT168 covers the same nuclei and was already used by the source dataset).

---

## PD-Specific Diffusion and Mechanical Properties

**Olsson F, Nilsson M, Stening EM, Lövdén M, Persson J (2025).** Effects of Parkinson's Disease on Mechanical and Microstructural Properties of the Brain.
*[Journal TBD — from PMC12351346]*
→ **The dataset paper.** MRE + MD-dMRI in PD (n=12) vs HC (n=17), Philips Ingenia CX 3T. Key PD findings: SNc/VTA μFA↓, MD↑; NAC FA↓ (d=−1.15, largest effect); temporal/occipital WM softening (MRE). Defines our ROI set and primary scientific context.

**Atkinson-Clement C, Pinto S, Eusebio A, Coulon O (2017).** Diffusion tensor imaging in Parkinson's disease: Review and meta-analysis.
*NeuroImage: Clinical* 16:98–110.
→ DTI systematic review in PD. Confirms WM microstructural alterations but uses FA (not μFA), motivating our MD-dMRI approach.

---

## tDCS Effects and WM Anisotropy

**Lee WH, Lisanby SH, Laine AF, Parra LC (2016).** Minimum electric field intensity in the brain at standard intensity for transcranial direct current stimulation.
*Brain Stimulation* — [check exact citation]
→ WM anisotropy effects on tDCS E-field distribution.

**Datta A, Bansal V, Diaz J, Patel J, Reato D, Bikson M (2009).** Gyri-precise head model of transcranial direct current stimulation: Improved spatial focality using a ring electrode versus conventional rectangular pad.
*Brain Stimulation* 2(4):201–207.
→ Individual-specific tDCS head modeling; inter-subject E-field variability.

**Stroke DTI-tDCS paper (Miraglia et al. or similar).** PMC12811680.
→ Closest methodological precedent to our approach: DTI-based anisotropic conductivity in a disease model. Compare methodology.

---

## Key Scripting / Pipeline

**Seibt O, Brunoni AR, Huang Y, Bikson M (2015).** The effect of tDCS waveform on current flow in the brain.
[ALSO: Bikson lab SimNIBS tutorial, biorxiv 704940]
→ SimNIBS pipeline tutorial from Bikson's own lab.

---

*Last updated: 2026-05-31. Add new references here as the project expands.*
