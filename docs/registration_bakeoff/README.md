# Registration bake-off + Synb0 circularity check (decision record)

These scripts settled the cohort dMRI->T1 registration method and are kept as the **evidentiary record** (the
chosen method now lives in `pipeline/02_register_dmri_to_T1.sh`). The decision numbers below stand on their
own; the scripts are the regeneration recipe and stay runnable against a built `REG_DIR/bakeoff/` tree (they
read paths from `config/config.sh`).

## Decision: S0-driven AFFINE
On the Synb0+topup distortion-corrected cohort, a 12-DOF affine (dMRI S0 → charm T2) is the registration.
Scored on the dry-run subject (one HC):

| arm | WM FA-corr vs his FA | over-warp | peduncle S-I | verdict |
|---|---|---|---|---|
| FA-driven affine | 0.61 | none | pass | FA is a weak driver |
| **S0-driven affine** | **0.84** | **none** | **pass** | **chosen** |
| fnirt (nonlinear) | 0.76 | **median 5.17 mm** | pass | rejected (over-warps) |

fnirt over-warps already-corrected data without improving alignment; the S0 driver (the ParkMRE pipeline's
`dtd_s0`) beats the FA driver. CC L-R is resolution-limited at 2.5 mm (native dMRI ≈ 0.56), so the
cerebral-peduncle S-I check is the orientation gate (now automated in `qc_harness`, `reg_peduncle_siz`).

- `registration_bakeoff.sh` - FSL affine vs fnirt arms (shared flirt init).
- `registration_bakeoff_s0.sh` - the S0-driven affine arm.
- `score_registration_bakeoff.py` - FA-correlation + SimNIBS-frame orientation + nonlinear-displacement scorer.

## Synb0 circularity check
`synb0_circularity_check.py` tested whether the T1-derived synthetic blip-down (Synb0) makes the S0→T1
registration circular. Regional edge-alignment gain from the correction was +0.088 in high-susceptibility
frontal/temporal but ≈0 in the deep PD targets → legitimate distortion correction, not uniform T1-reshaping;
the registration is genuine.

## Cerebellum coverage check
`cerebellum_coverage_check.py` tested whether the MD-dMRI cerebellum thinning is a Synb0 artifact or an
inherent dMRI FOV/SNR limit. Cerebellum b0 signal in the dMRI FOV is 100% (corrected) vs 98.9% (uncorrected
pre-Synb0) -> a match, so the correction did not cause it; it is a tensor-coverage gap (only ~60% of the
cerebellum gets a QTI tensor, a low-SNR foliated inferior edge), where the model falls back to scalar sigma0
and still gets a field. The cerebellum is not a tDCS target here (SN/STN/GP are), so there is no study impact.
