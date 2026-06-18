#!/bin/bash
# 01_dwi2cond.sh -- DTI baseline conductivity tensor via standard SimNIBS dwi2cond.
#
# The DTI arm is the SEPARATE single-shell ParkMRE DTI scan (not the MUDI multi-shell). That scan is already
# eddy/topup-corrected upstream, so we do NOT re-run dwi2cond's preprocessing; we fit the tensor with FSL
# dtifit (--save_tensor) and hand the fitted tensor to dwi2cond for its validated T1 coregistration +
# reorientation (the standard SimNIBS path, no reimplementation):
#   dtifit --save_tensor   ->   dwi2cond --all --regmthd=12dof <subjectID> <DTI_tensor>
#
# Why this exact path (validated 2026-06-18 on PD_20230125_Control1 + Patient1):
#  * dwi2cond --help: "A preprocessed DTI tensor (written out by FSL dtifit using --save_tensor) ... will be
#    coregistered to the T1 image of the subject." So feeding the fitted tensor uses dwi2cond's own
#    reorientation and skips the eddy/distortion steps our data does not need.
#  * Registration option is --regmthd=12dof (affine). NOT the nonlinear default (fnirt over-warps the
#    corrected data, registration bake-off 2026-06-16) and NOT `-r 12dof` (wrong flag for this dwi2cond build).
#  * The registration DRIVER does not matter for DTI: a DTI bake-off (FA->T1 vs b0->T2) moved the DTI-vs-<D>
#    orientation by <2 deg, and standard dwi2cond matched a hand-rolled S0/b0 affine to ~1 deg. So dwi2cond's
#    own FA->T1 coregistration is used as-is (unlike the MD-dMRI arm, which needs the S0/b0 affine of 02).
#  * The fitted tensor was cross-checked against the MRtrix DWI2Tensor (FA r=0.997, V1 within 1.6 deg in WM).
#
# Config inputs: DTI_DWI (eddy-corrected single-shell DWI), DTI_BVEC (eddy-rotated bvecs). bvals default to a
# single b0 + N*b1500 if DTI_BVAL is unset; DTI_MASK optional. (Cohort staging of these is the cluster sub-project.)
#
# Usage: PIPELINE_CONFIG=<subject config.sh> bash pipeline/01_dwi2cond.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${PIPELINE_CONFIG:-$SCRIPT_DIR/../config/config.sh}"
# shellcheck disable=SC1090
source "$CFG"
export FSLDIR; export PATH="$SIMNIBS_BIN:$FSLDIR/bin:$PATH"
source "$FSLDIR/etc/fslconf/fsl.sh"
export FSLOUTPUTTYPE=NIFTI_GZ

M2M="$M2M_DIR"
TENSOR_OUT="$M2M/DTI_coregT1_tensor.nii.gz"
[ -d "$M2M" ] || { echo "ERROR: $M2M not found -- run 00_charm.sh first."; exit 1; }
: "${DTI_DWI:?set DTI_DWI to the eddy-corrected single-shell DTI DWI}"
: "${DTI_BVEC:?set DTI_BVEC to the (eddy-rotated) bvecs}"
for f in "$DTI_DWI" "$DTI_BVEC"; do [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }; done
if [ -f "$TENSOR_OUT" ]; then echo "DTI tensor exists, skipping (delete to force): $TENSOR_OUT"; exit 0; fi

FITDIR="$WORK_DIR/dti_arm"; mkdir -p "$FITDIR"
BVAL="${DTI_BVAL:-}"
if [ -z "$BVAL" ]; then
  NV="$(fslnvols "$DTI_DWI")"; BVAL="$FITDIR/dti.bval"
  "$SIMNIBS_BIN/simnibs_python" -c "n=int('$NV'); open('$BVAL','w').write(' '.join(['0']+['1500']*(n-1))+'\n')"
fi
MASKOPT=""; [ -n "${DTI_MASK:-}" ] && MASKOPT="-m ${DTI_MASK}"

echo "Step 1: FSL dtifit (--save_tensor) on the corrected DTI"
# shellcheck disable=SC2086
dtifit -k "$DTI_DWI" -o "$FITDIR/dti" $MASKOPT -r "$DTI_BVEC" -b "$BVAL" --save_tensor

echo "Step 2: dwi2cond T1 coregistration of the fitted tensor (standard SimNIBS reorientation)"
cd "$WORK_DIR"; rm -rf "$M2M/dMRI_prep"
dwi2cond --all --regmthd=12dof "$SUBJECT" "$FITDIR/dti_tensor.nii.gz"
[ -f "$TENSOR_OUT" ] && echo "Done: $TENSOR_OUT (SimNIBS reads this for anisotropy_type='vn')" \
    || { echo "ERROR: dwi2cond produced no tensor"; exit 1; }
