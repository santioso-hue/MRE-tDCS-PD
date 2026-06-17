#!/bin/bash
# 01_dwi2cond.sh — Fit the DTI conductivity tensor for the DTI baseline model (SimNIBS dwi2cond)
#
# Usage: bash pipeline/01_dwi2cond.sh   (then inspect with: dwi2cond --check "$SUBJECT")
#
# dwi2cond argument order is <subID> <DWI> <bval> <bvec>, verified against the source
# (simnibs/external/dwi2cond): BVALS=$3, BVECS=$4. An earlier "bypass" version assumed a
# bvec/bval swap bug and reimplemented the prep manually; that was a mis-ordered manual call,
# not a tool bug. `dwi2cond --all` is the validated path: eddy_correct → dtifit → FA→T1 → vecreg → mask.
#
# We pass `-r 12dof` (affine FA→T1) instead of dwi2cond's nonlinear default so the DTI arm shares the
# MD-dMRI arm's registration class (S0-driven affine; fnirt over-warps the Synb0+topup data ~5 mm,
# bake-off 2026-06-16). This keeps the DTI↔MD-dMRI E-field contrast on the tensor, not the registration.
# TODO (gated on the separate ParkMRE_DTI scan): re-register dwi2cond's tensor with the MD-dMRI S0-affine (02).

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

if [ ! -d "$M2M" ]; then
    echo "ERROR: $M2M not found — run CHARM (00_charm.sh) first."; exit 1
fi
for f in "$DWI" "$BVAL" "$BVEC"; do
    [ -f "$f" ] || { echo "ERROR: required file not found: $f"; exit 1; }
done

if [ -f "$TENSOR" ]; then
    echo "DTI tensor already exists, skipping (delete to force re-run): $TENSOR"
    exit 0
fi

cd "$WORK_DIR"

# Remove any locked/partial prep from an interrupted run (avoids permission errors)
rm -rf "$M2M/dMRI_prep"

dwi2cond --all -r 12dof "$SUBJECT" "$DWI" "$BVAL" "$BVEC"

echo "dwi2cond complete. Tensor: $TENSOR  (SimNIBS reads this for anisotropy_type='vn')"
echo "Check registration:  dwi2cond --check $SUBJECT"
