# References - MRE_tDCS_PD Project

Organized by role in the project. All citations are used in scripts, docs, or manuscript methods.

---

## WM Conductivity Anisotropy - Empirical Basis

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
→ **Foundational theoretical paper.** Derived σ ∝ D from Maxwell-Garnett effective medium / tortuosity theory. Basis for dwi2cond and our MD-dMRI conductivity approach alike. Note: the 7–10:1 range is NOT from this paper - Tuch reports in vivo DTI ratios of 2–4:1. Cite Nicholson 1965 and Ranck 1965 for the 7–10:1 range.

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
→ DPS (Diffusion Propagator Spectroscopy) model. Single mean compartment tensor fit under prolate (axially symmetric) constraint: D_ij = (λ₁−λ₂)·v1_i·v1_j + λ₂·δ_ij. Outputs μFA, MD, v₁. This is the analysis framework the dataset's group (Olsson et al. 2025) uses. NOT a SimNIBS concept.

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
generalised to the QTI 3-compartment fit: ⟨D⟩_tissue = (Σ_{k≠FW} f_k D_k)/(Σ_{k≠FW} f_k). FWE was
evaluated and not adopted under `'vn'`; plain ⟨D⟩ (no FWE) is the `--meanD` sensitivity.

---

## ROI Analysis Methods

**Saturnino GB, Siebner HR, Thielscher A, Madsen KH (2019).** Accessibility of cortical regions to focal TES: Dependence on spatial position, safety, and practical constraints.
*NeuroImage* 203:116183.
→ 5mm spherical ROI approach around MNI coordinates - used by SimNIBS group for CORTICAL targets. We follow this only for the cortical M1 reference; the subcortical ROIs are anatomical atlas masks (below), not spheres.

**Huang Y, Liu AA, Lafon B, Friedman D, Dayan M, Wang X, Bikson M, Doyle WK, Devinsky O, Parra LC (2017).** Measurements and models of electric fields in the in vivo human brain during transcranial electrical stimulation.
*eLife* 6:e18834.
→ Direct E-field measurements in vivo. Recommends 95th percentile statistic over mean or maximum for robust ROI characterization. Justification for reporting p95 alongside mean/median.

**Antonenko D, Thielscher A, Saturnino GB, Nierhaus T, Meinzer M, Prehn K, Flöel A (2019).** Towards precise brain stimulation: Is electric field simulation related to neuromodulation?
*Brain Stimulation* 12(5):1159–1168.
→ Atlas-mask approach for cortical ROI E-field extraction using FreeSurfer parcellations.

---

## ROI Definitions - Subcortical Atlases

**Pauli WM, Nili AN, Tyszka JM (2018).** A high-resolution probabilistic in vivo atlas of human subcortical brain nuclei (CIT168).
*Scientific Data* 5:180063.
→ **The atlas used for all Group 2 deep nuclei** (`07_build_nuclei.sh`): the basal ganglia (Pu, Ca, NAC, GPe, GPi) and the midbrain/subthalamic nuclei (SNc, SNr, VTA, RN, STN), split left and right, are warped from MNI152-2009c to subject space with overlap-allowed masks. Every subcortical readout comes from this single atlas (the aseg subcortical structures are dropped). Olsson 2025 also uses CIT168.

**Ewert S, Plettig P, Li N, Bhatt M, Bhatt DL, Kühn AA, Volkmann J, Horn A (2018).** Toward defining deep brain stimulation targets in MNI space: A subcortical atlas based on multimodal MRI, histology and structural connectivity.
*NeuroImage* 170:271–282.
→ DISTAL atlas - a high-quality brainstem alternative for the dopaminergic nuclei; evaluated but not used in the final pipeline (CIT168 covers the same nuclei and was already used by the source dataset).

---

## PD-Specific Diffusion and Mechanical Properties

**Olsson C, Nilsson M, Stening EM, Lövdén M, Persson J (2025).** Effects of Parkinson's Disease on Mechanical and Microstructural Properties of the Brain.
*[Journal TBD - from PMC12351346]*
→ **The dataset paper.** MRE + MD-dMRI in PD (n=12) vs HC (n=17), Philips Ingenia CX 3T. Defines our ROI set and primary scientific context (the deep-nuclei microstructure and WM-softening regions it reports).

**Atkinson-Clement C, Pinto S, Eusebio A, Coulon O (2017).** Diffusion tensor imaging in Parkinson's disease: Review and meta-analysis.
*NeuroImage: Clinical* 16:98–110.
→ DTI systematic review in PD. Confirms WM microstructural alterations but uses FA (not μFA), motivating our MD-dMRI approach.

---

## tDCS Effects and WM Anisotropy

**Lee WH, Lisanby SH, Laine AF, Parra LC (2016).** Minimum electric field intensity in the brain at standard intensity for transcranial direct current stimulation.
*Brain Stimulation* - [check exact citation]
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

## QTI / b-tensor Encoding and Kurtosis Bias (Methods 2.2, 2.4)

**Szczepankiewicz F, Sjölund J, Ståhlberg F, Lätt J, Nilsson M (2019).** Tensor-valued diffusion encoding for diffusional variance decomposition (DIVIDE): Technical feasibility in clinical MRI systems.
*PLoS ONE* 14(3):e0214238.
→ Linear + spherical b-tensor encoding on clinical scanners; the acquisition basis for the QTI/MD-dMRI model.

**Nilsson M, Szczepankiewicz F, Lampinen B, Ahlgren A, de Almeida Martins JP, Lasič S, Westin CF, Topgaard D (2018).** An open-source framework for analysis of multidimensional diffusion MRI data implemented in MATLAB.
*Proc. Intl. Soc. Mag. Reson. Med.* 26:5355. [verify abstract number against reference manager]
→ The `md-dmri` toolbox used for the QTI covariance fit (`run_qti_cov_cohort.m`).

**Veraart J, Poot DHJ, Van Hecke W, Blockx I, Van der Linden A, Verhoye M, Sijbers J (2011).** More accurate estimation of diffusion tensor parameters using diffusion kurtosis imaging.
*Magnetic Resonance in Medicine* 65(1):138–145.
→ Single-shell mono-exponential DTI is kurtosis-biased; supports ⟨D⟩ (covariance first cumulant) as a less-biased estimate of the same macroscopic tensor.

**Jensen JH, Helpern JA, Ramani A, Lu H, Kaczynski K (2005).** Diffusional kurtosis imaging: the quantification of non-Gaussian water diffusion by means of magnetic resonance imaging.
*Magnetic Resonance in Medicine* 53(6):1432–1440.
→ Defines non-Gaussian (kurtosis) diffusion that biases the single-shell tensor estimate.

**Lampinen B, Szczepankiewicz F, Mårtensson J, van Westen D, Sundgren PC, Nilsson M (2017).** Neurite density imaging versus imaging of microscopic anisotropy in diffusion MRI: A model comparison using spherical tensor encoding.
*NeuroImage* 147:517–531.
→ Microscopic anisotropy requires variable b-tensor shape; context for why μFA is not a macroscopic conductivity input.

---

## Registration and Diffusion Preprocessing (Methods 2.5)

**Alexander DC, Pierpaoli C, Basser PJ, Gee JC (2001).** Spatial transformations of diffusion tensor magnetic resonance images.
*IEEE Transactions on Medical Imaging* 20(11):1131–1139.
→ Preservation-of-principal-direction (PPD) reorientation: a tensor must be reoriented, not merely resampled, when warped. Basis for the `vecreg` step.

**Jenkinson M, Smith S (2001).** A global optimisation method for robust affine registration of brain images.
*Medical Image Analysis* 5(2):143–156.
→ FLIRT affine registration (the 12-DOF dMRI→T1 driver).

**Jenkinson M, Beckmann CF, Behrens TEJ, Woolrich MW, Smith SM (2012).** FSL.
*NeuroImage* 62(2):782–790.
→ FSL toolbox: dtifit, FLIRT, vecreg, topup, eddy.

**Andersson JLR, Skare S, Ashburner J (2003).** How to correct susceptibility distortions in spin-echo echo-planar images: application to diffusion tensor imaging.
*NeuroImage* 20(2):870–888.
→ `topup` susceptibility-distortion correction (applied upstream to the cohort dMRI).

**Andersson JLR, Sotiropoulos SN (2016).** An integrated approach to correction for off-resonance effects and subject movement in diffusion MR imaging.
*NeuroImage* 125:1063–1078.
→ `eddy` eddy-current and motion correction (applied upstream).

**Schilling KG, Blaber J, Huo Y, Newton A, Hansen C, Nath V, Shafer AT, Williams O, Resnick SM, Rogers B, Anderson AW, Landman BA (2019).** Synthesized b0 for diffusion distortion correction (Synb0-DisCo).
*Magnetic Resonance Imaging* 64:62–70.
→ Synthetic reverse-phase-encode b0 from T1 (the cohort distortion-correction route; the 2020 PLoS ONE companion is the deep-learning validation).

**Arsigny V, Fillard P, Pennec X, Ayache N (2006).** Log-Euclidean metrics for fast and simple calculus on diffusion tensors.
*Magnetic Resonance in Medicine* 56(2):411–421.
→ Log-Euclidean tensor interpolation (the gold standard our per-eigenvalue + PPD scheme approximates without tensor swelling).

---

## Head Segmentation and Atlases (Methods 2.3, 2.7)

**Puonti O, Van Leemput K, Saturnino GB, Siebner HR, Madsen KH, Thielscher A (2020).** Accurate and robust whole-head segmentation from magnetic resonance images for individualized head modeling.
*NeuroImage* 219:117044.
→ The `charm` head-segmentation pipeline (SimNIBS 4.x) used for the FEM mesh.

**Fischl B (2012).** FreeSurfer.
*NeuroImage* 62(2):774–781.
→ FreeSurfer `recon-all` (cortical/WM parcellation; matches Olsson et al. 2025).

**Desikan RS, Ségonne F, Fischl B, et al. (2006).** An automated labeling system for subdividing the human cerebral cortex on MRI scans into gyral based regions of interest.
*NeuroImage* 31(3):968–980.
→ Desikan-Killiany atlas; grouped to frontal/parietal/temporal/occipital lobes (Group 1 cross-comparison regions).

**Fischl B, Salat DH, Busa E, et al. (2002).** Whole brain segmentation: automated labeling of neuroanatomical structures in the human brain.
*Neuron* 33(3):341–355.
→ FreeSurfer `aseg` subcortical segmentation. No longer the subcortical ROI source: under the regrouped scheme all subcortical readouts come from the CIT168 Group 2 nuclei, not the aseg.

**Iglesias JE, Van Leemput K, Bhatt P, Casillas C, Dutt S, Schuff N, Truran-Sacrey D, Boxer A, Fischl B (2015).** Bayesian segmentation of brainstem structures in MRI.
*NeuroImage* 113:184–195.
→ Brainstem substructures; the mesencephalon/pons split (Group 1 cross-comparison regions).

**Avants BB, Tustison NJ, Song G, Cook PA, Klein A, Gee JC (2011).** A reproducible evaluation of ANTs similarity metric performance in brain image registration.
*NeuroImage* 54(3):2033–2044.
→ ANTs; the MNI→subject warp for the Group 2 CIT168 nuclei.

---

## Statistics and Scientific Software (Methods 2.8, 2.9)

**Benjamini Y, Hochberg Y (1995).** Controlling the false discovery rate: a practical and powerful approach to multiple testing.
*Journal of the Royal Statistical Society: Series B* 57(1):289–300.
→ BH-FDR multiple-comparison control, per montage and per test family.

**Harris CR, Millman KJ, van der Walt SJ, et al. (2020).** Array programming with NumPy.
*Nature* 585:357–362.
→ NumPy (all numerical analysis).

**Virtanen P, Gommers R, Oliphant TE, et al. (2020).** SciPy 1.0: fundamental algorithms for scientific computing in Python.
*Nature Methods* 17:261–272.
→ SciPy (Wilcoxon signed-rank, Mann-Whitney U, partial-correlation t-test, Spearman).

---

## MRE Mechanics, Aging, and the CSF / Atrophy Confound (Figure 4, Discussion)

All five verified against PubMed (DOIs below).

**Sack I, Beierbach B, Wuerfel J, Klatt D, Hamhaber U, Papazoglou S, Martus P, Braun J (2009).** The impact of aging and gender on brain viscoelasticity.
*NeuroImage* 46(3):652–657. DOI: 10.1016/j.neuroimage.2009.02.040
→ Healthy brain stiffness (springpot μ) declines ~0.8%/yr; the structure parameter α is the same springpot exponent as the cohort's `alpha` map. Establishes age as a driver of MRE stiffness.

**Hiscox LV, Johnson CL, Barnhill E, McGarry MDJ, Huston J, van Beek EJR, Starr JM, Roberts N (2016).** Magnetic resonance elastography (MRE) of the human brain: technique, findings and clinical applications.
*Physics in Medicine and Biology* 61(24):R401–R437. DOI: 10.1088/0031-9155/61/24/R401
→ MRE methods review and baseline grey/white-matter stiffness (kPa) and loss-tangent values across studies.

**Hiscox LV, Schwarb H, McGarry MDJ, Johnson CL (2021).** Aging brain mechanics: Progress and promise of magnetic resonance elastography.
*NeuroImage* 232:117889. DOI: 10.1016/j.neuroimage.2021.117889
→ Review of MRE in healthy aging and neurodegeneration; subcortical/regional softening with age.

**Indahlastari A, Albizu A, O'Shea A, Forbes MA, Nissim NR, Kraft JN, Evangelista ND, Hausman HK, Woods AJ (2020).** Modeling transcranial electrical stimulation in the aging brain.
*Brain Stimulation* 13(3):664–674. DOI: 10.1016/j.brs.2020.02.007
→ **Direct support for the Figure 4 confound:** in 587 older adults, computed tES field is inversely correlated with atrophy and the age→current-density relationship is *partially mediated by brain-to-CSF ratio*.

**Unal G, Ficek B, Webster K, Shahabuddin S, Truong D, Hampstead B, Bikson M, Tsapkini K (2020).** Impact of brain atrophy on tDCS and HD-tDCS current flow: a modeling study in three variants of primary progressive aphasia.
*Neurological Sciences* 41(7):1781–1789. DOI: 10.1007/s10072-019-04229-z
→ Local atrophy does not, in isolation, predict local E-field; holistic head anatomy drives current flow. Supports the Figure 4 reframing (whole-head/CSF morphology, not local mechanics).

---

## Additional Conductivity and tDCS References

**Rullmann M, Anwander A, Dannhauer M, Warfield SK, Duffy FH, Wolters CH (2009).** EEG source analysis of epileptiform activity using a 1 mm anisotropic hexahedra finite element head model.
*NeuroImage* 44(2):399–410.
→ Volume-constrained anisotropic FEM head model; precedent for the `'vn'` mapping (with Güllmar 2010).

**Mosayebi-Samani M, et al. (2025).** [tES brain-anisotropy sensitivity study]
*Imaging Neuroscience* (SimNIBS group). [verify full author list / pages against reference manager]
→ Brain anisotropy has a small effect on the tES E-field (scalar-conductivity uncertainty dominates); MREIT weakly sensitive to anisotropy. Same charm/dwi2cond/`'vn'` framework. Cited in `conductivity_models_derivation.md`.

**Deuschl G, Schade-Brittinger C, Krack P, et al. (2006).** A randomized trial of deep-brain stimulation for Parkinson's disease.
*New England Journal of Medicine* 355(9):896–908.
→ STN is the principal DBS target in advanced PD; motivates the Group 2 STN E-field readout.

---

*Last updated: 2026-06-22. The Figure 4 / MRE-aging cluster (Sack, Hiscox x2, Indahlastari, Unal) is PubMed-verified with DOIs; entries marked [verify] should be cross-checked in the reference manager. Add new references here as the project expands.*
