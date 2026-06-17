# Registration bake-off + Synb0 circularity check (decision record)

These scripts settled the cohort dMRI→T1 registration method. They are the **evidentiary record**, not
runnable pipeline stages — the chosen method lives in `pipeline/02_register_dmri_to_T1.sh`. (They reference
fit/registration outputs by relative path and are not maintained for re-running.)

## Decision: S0-driven AFFINE
On the Synb0+topup distortion-corrected cohort, a 12-DOF affine (dMRI S0 → charm T2) is the registration.
Scored on the dry-run subject (one HC):

| arm | WM FA-corr vs his FA | over-warp | peduncle S-I | verdict |
|---|---|---|---|---|
| FA-driven affine | 0.61 | none | pass | FA is a weak driver |
| **S0-driven affine** | **0.84** | **none** | **pass** | **chosen** |
| fnirt (nonlinear) | 0.76 | **median 5.17 mm** | pass | rejected (over-warps) |

fnirt over-warps already-corrected data without improving alignment; the S0 driver (Christoffer's
`dtd_s0`) beats the FA driver. CC L-R is resolution-limited at 2.5 mm (native dMRI ≈ 0.56), so the
cerebral-peduncle S-I check is the orientation gate (now automated in `qc_harness`, `reg_peduncle_siz`).

- `registration_bakeoff.sh` — FSL affine vs fnirt arms (shared flirt init).
- `registration_bakeoff_s0.sh` — the S0-driven affine arm.
- `score_registration_bakeoff.py` — FA-correlation + SimNIBS-frame orientation + nonlinear-displacement scorer.

## Synb0 circularity check
`synb0_circularity_check.py` tested whether the T1-derived synthetic blip-down (Synb0) makes the S0→T1
registration circular. Regional edge-alignment gain from the correction was +0.088 in high-susceptibility
frontal/temporal but ≈0 in the deep PD targets → legitimate distortion correction, not uniform T1-reshaping;
the registration is genuine. See `state.md` for the full result.
