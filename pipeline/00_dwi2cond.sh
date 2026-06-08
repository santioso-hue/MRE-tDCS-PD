#!/bin/bash
# 00_dwi2cond.sh — Fit DTI conductivity tensors for the DTI model (SimNIBS dwi2cond)
#
# CORRECTION (supersedes the earlier "bypass" version of this script):
#   A previous version claimed dwi2cond had a bvec/bval argument-swap bug and implemented
#   a manual dtifit+FLIRT+vecreg bypass. That was WRONG. Verified against the dwi2cond
#   source (simnibs/external/dwi2cond + dwi2cond.prepro.source.sh): the positional order is
#       dwi2cond [options] <subID> <DWI> <bval> <bvec>
#   and it is mapped correctly internally (BVALS=$3 → DWIbvals, BVECS=$4 → DWIbvecs).
#   `dwi2cond --all` runs cleanly on FullPD5 and IS the SimNIBS-standard, validated path:
#       eddy_correct  →  dtifit  →  nonlinear fnirt (FA→T1)  →  vecreg (tensor reorientation)
#       →  brain-mask.
#   The original failure was a mis-ordered manual call, not a tool bug. dwi2cond's nonlinear
#   FA→T1 registration is more accurate than the previous 6-DOF rigid bypass; its principal
#   eigenvector agrees with the QTI dps.u to ~22° median in core WM (FA>0.5), ~30° across all WM
#   — see tests/validate_mean_tensor.py. (Separate sDTI acquisition, so this is a moderate, not
#   tight, agreement.)
#
# Prerequisites: CHARM has run (m2m_${SUBJECT}/ exists).
# Idempotent:    skips if the coregistered tensor already exists.
#
# Usage:
#   cd /Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation
#   bash scripts/00_dwi2cond.sh
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

# Standard SimNIBS DTI preparation
# Argument order: subID DWI bval bvec  (verified correct against dwi2cond source)
echo "Running dwi2cond --all (eddy + dtifit + fnirt FA→T1 + vecreg + mask)..."
dwi2cond --all "$SUBJECT" "$DWI" "$BVAL" "$BVEC"

echo ""
echo "dwi2cond complete."
echo "  Tensor:  $TENSOR  (SimNIBS reads this for anisotropy_type='vn')"
echo "  Check registration visually:  dwi2cond --check $SUBJECT"
echo "Next: run the DTI simulation (03_run_simulations.py / SimNIBS with anisotropy_type='vn')."
