#!/bin/bash
# 01_dwi2cond.sh — Fit the DTI conductivity tensor for the DTI baseline model (SimNIBS dwi2cond)
#
# CORRECTION (supersedes the earlier "bypass" version of this script):
#   A previous version claimed dwi2cond had a bvec/bval argument-swap bug and implemented
#   a manual dtifit+FLIRT+vecreg bypass. That was WRONG. Verified against the dwi2cond
#   source (simnibs/external/dwi2cond + dwi2cond.prepro.source.sh): the positional order is
#       dwi2cond [options] <subID> <DWI> <bval> <bvec>
#   and it is mapped correctly internally (BVALS=$3 → DWIbvals, BVECS=$4 → DWIbvecs).
#   `dwi2cond --all` runs cleanly on FullPD5 and IS the SimNIBS-standard, validated path:
#       eddy_correct  →  dtifit  →  FA→T1 registration  →  vecreg (tensor reorientation)  →  brain-mask.
#   The original failure was a mis-ordered manual call, not a tool bug.
#
# REGISTRATION CLASS (decided with the MD-dMRI bake-off, 2026-06-16): the cohort MD-dMRI arm registers by
# an S0-driven AFFINE, because fnirt over-warps the Synb0+topup-corrected data (~5 mm). For the
# DTI↔MD-dMRI E-field contrast to isolate the tensor and not the registration, the DTI arm must register
# the same affine class — so we pass `-r 12dof` (affine FA→T1) instead of dwi2cond's nonlinear default.
# TODO (when the separate ParkMRE_DTI scan arrives — the arm is gated on Rodrigo): consider the tighter
# match of fitting the tensor with dwi2cond and re-registering it with the MD-dMRI S0-affine (02).
#
# Prerequisites: CHARM has run (m2m_${SUBJECT}/ exists).
# Idempotent:    skips if the coregistered tensor already exists.
#
# Usage:
#   bash pipeline/01_dwi2cond.sh
#   # inspect registration afterwards:  dwi2cond --check "$SUBJECT"
#
# Runtime: ~6 min

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"
export FSLDIR
export PATH="$SIMNIBS_BIN:$FSLDIR/bin:$PATH"
source "$FSLDIR/etc/fslconf/fsl.sh"


# Single-shell sDTI (801 series): 1×b=0 + 80×b≈1500, MB3
DWI="$DWI_NII"
BVAL="$DWI_BVAL"
BVEC="$DWI_BVEC"

M2M="$M2M_DIR"
TENSOR="$M2M/DTI_coregT1_tensor.nii.gz"

# Pre-flight
if [ ! -d "$M2M" ]; then
    echo "ERROR: $M2M not found — run CHARM (00_charm.sh) first."; exit 1
fi
for f in "$DWI" "$BVAL" "$BVEC"; do
    [ -f "$f" ] || { echo "ERROR: required file not found: $f"; exit 1; }
done

# Idempotency
if [ -f "$TENSOR" ]; then
    echo "DTI tensor already exists: $TENSOR"
    echo "Skipping (delete it to force a re-run)."
    exit 0
fi

cd "$WORK_DIR"

# Remove any locked/partial prep from an interrupted run (avoids permission errors)
rm -rf "$M2M/dMRI_prep"

# Standard SimNIBS DTI preparation. -r 12dof = affine FA→T1, matching the MD-dMRI arm's affine class
# (see the registration-class note above). Argument order subID DWI bval bvec (verified vs dwi2cond source).
echo "Running dwi2cond --all -r 12dof (eddy + dtifit + affine FA→T1 + vecreg + mask)..."
dwi2cond --all -r 12dof "$SUBJECT" "$DWI" "$BVAL" "$BVEC"

echo ""
echo "dwi2cond complete."
echo "  Tensor:  $TENSOR  (SimNIBS reads this for anisotropy_type='vn')"
echo "  Check registration visually:  dwi2cond --check $SUBJECT"
echo "Next: run the DTI simulation (04_run_simulations.py / SimNIBS with anisotropy_type='vn')."
