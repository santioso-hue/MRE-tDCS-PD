# Results-section deliverable plan

Planning note for the manuscript Results (not a task to run; no figures or tables are generated from this).
The Results mirror the Methods order (2.1-2.9) and tell a three-legged story: H1 primary positive; the MRE
morphology-not-conductivity dissection; H3 and deep nuclei exploratory. Stay at or under about 8 main display
items, put demographics and QC first, and keep null and exploratory material in the supplement. Most items gate
on completing the three remaining montage runs (HD-M1 and the two prefrontal montages); the only genuinely new
analysis is the cross-subject variability decomposition.

## Main text (4-5 figures + 2 tables)

- Figure 1 - E-field maps for ISO/DTI/MD-dMRI plus the MD-dMRI minus DTI difference, representative cohort
  subject, M1 pad. Mirrors 2.6/2.8. Status: pilot prototype exists; need the cohort version.
- Figure 2 - H1 primary: per-ROI dE_model across the cohort, tier-grouped, signed, with significance, plus an
  ISO-vs-anisotropic context panel. Mirrors 2.8. Status: have M1; need the full montage set.
- Figure 3 - Cross-montage summary: compact effect across all four montages; show one pad and one HD as maps in
  the main text, the full set in the supplement. Mirrors 2.6. Status: needs the montage runs.
- Figure 4 - MRE cross-comparison: field-stiffness vs the dE_model-stiffness null, Olsson Fig 6 style. Mirrors
  2.9. Status: have M1; need the final render.
- Figure 5 - Cross-subject variability decomposition (Pundik framing): p95 |E| variability across the 29
  subjects, conductivity-model vs anatomy contribution. Main text only if it lands cleanly, otherwise drop.
  Status: new analysis to build.
- Table 1 - Demographics (group, n, age, sex, UPDRS). Status: trivial.
- Table 2 - H1 per-ROI summary: tier sub-headers, dE_model median/IQR, significance, direction, rank-biserial,
  plus the significant-count and white-matter-consensus line. Status: M1 specced; needs the full montage set.

## Supplement

- S-Fig 1 - Model construction schematic (shared mesh to three conductivity assignments). Demoted from main;
  orienting.
- S-Fig 2 - Full four-montage difference maps (behind Figure 3's representative pair).
- S-Fig 3 - H3 exploratory: per-ROI PD-vs-HC effect sizes, null, FDR, Olsson Fig 5 style.
- S-Fig 4 - QC: representative segmentations plus registration orientation validation (corpus callosum
  left-right, cerebral peduncle superior-inferior). CONDITIONAL: justified only if it shows the
  failure-relevant check passing on the hard region (the peduncle orientation gate), not pretty slices.
- S-Table 1 - Full per-ROI x per-montage H1 statistics.
- S-Table 2 - Full H3 statistics across ROIs and montages.
- S-Table 3 - Full MRE correlations per ROI (group-only and group+age) for field-stiffness and
  dE_model-stiffness. CONDITIONAL: justified only if Figure 4 is summarized; if Figure 4 shows the full
  per-ROI correlation matrix in the main text, S-Table 3 is redundant and moves to the repo.

## Repo (not supplement)

- Voxel-isotropization and eigenvalue-clamping maps and counts.
- Per-subject QC manifest (registration pass, percent voxels isotropized, ISO-fallback flags).
- Tier-3 deep-nuclei field exposure, unless the STN is actually discussed (then a text sentence plus its
  S-Table 1 row suffices).

## Load-bearing minimum (priority when developing later)

The minimum publishable Results set is Table 1, Figure 1, the Figure 2 plus Table 2 H1 pair, and Figure 4, with
H3 in the supplement. Everything else strengthens or completes. The critical path is the three remaining montage
runs (HD-M1 and the two prefrontal montages); after them, Figures 2/3/5 and Tables 2/S-T1/S-T2 fall out of one
group-stats pass, the MRE figure is a re-render, and the QC and clamping items are assembly from existing logs.
