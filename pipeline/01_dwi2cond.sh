#!/bin/bash
# 01_dwi2cond.sh -- DTI baseline conductivity tensor via standard SimNIBS dwi2cond.
#
# The DTI arm is the SEPARATE single-shell DTI scan (not the MUDI multi-shell). It is already
# eddy/topup-corrected upstream, so we do NOT re-run dwi2cond's preprocessing; we fit the tensor with FSL
# dtifit (--save_tensor) and hand the fitted tensor to dwi2cond for its validated T1 coregistration +
# reorientation (the standard SimNIBS path, no reimplementation):
#   dtifit --save_tensor   ->   dwi2cond --all --regmthd=12dof <subjectID> <DTI_tensor>
#
# Why this path (validated 2026-06-18 on two held-out subjects, one HC + one PD):
#  * dwi2cond --help: a preprocessed dtifit --save_tensor tensor is accepted and only coregistered to T1,
#    so the eddy/distortion steps our data does not need are skipped.
#  * --regmthd=12dof (affine). NOT the nonlinear default (fnirt over-warps the corrected data) and NOT the
#    old `-r 12dof` (wrong flag for this dwi2cond build).
#  * Registration driver does not matter for DTI (bake-off FA->T1 vs b0->T2 moved orientation <2 deg; standard
#    dwi2cond matched a hand-rolled affine ~1 deg), so dwi2cond's own FA->T1 coregistration is used as-is.
#  * The fitted tensor was cross-checked vs MRtrix DWI2Tensor (FA r=0.997, V1 1.6 deg WM).
#
# Config inputs (see config.example.sh): DTI_DWI (eddy-corrected single-shell DTI DWI), DTI_BVEC (eddy-rotated
# bvecs). Optional: DTI_BVAL (real bvals; used only when its token count matches the DWI, else single-shell
# bvals are synthesized from the bvec with b = DTI_BVALUE, default 1500), DTI_MASK. Config aliases DTI_* to
# DWI_* for the standard single-subject case; the cohort points them at the independent ParkMRE_DTI scan
# (staged by run_cohort -- the cluster sub-project).
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
: "${DTI_DWI:?set DTI_DWI to the single-shell DTI DWI (see config.example.sh)}"
: "${DTI_BVEC:?set DTI_BVEC to the (eddy-rotated) bvecs (see config.example.sh)}"
# Idempotent: a built tensor short-circuits BEFORE the source-file check, so re-runs (and cohort
# re-runs over already-done subjects) do not need the raw DTI scan to still be mounted.
[ -f "$TENSOR_OUT" ] && { echo "DTI tensor exists, skipping (delete to force): $TENSOR_OUT"; exit 0; }
for f in "$DTI_DWI" "$DTI_BVEC"; do [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }; done

FITDIR="$WORK_DIR/dti_arm"; mkdir -p "$FITDIR"
NV="$(fslnvols "$DTI_DWI")"
BVAL="${DTI_BVAL:-}"
if [ -n "$BVAL" ] && [ -f "$BVAL" ] && [ "$(wc -w < "$BVAL")" -eq "$NV" ]; then
  echo "Using provided bvals: $BVAL ($NV vols)"
else
  # No matching real bvals (the ParkMRE_DTI export omits/mismatches them). Synthesize a single-shell bval
  # from the bvec: b0 at the zero-norm column(s), b=DTI_BVALUE elsewhere. Aborts unless >=1 b0 and the bvec
  # column count matches the DWI -- no silent mis-scale (the b0 position is read, not assumed).
  BVAL="$FITDIR/dti.bval"
  "$SIMNIBS_BIN/simnibs_python" - "$DTI_BVEC" "$NV" "${DTI_BVALUE:-1500}" "$BVAL" <<'PY'
import sys, numpy as np
bvec, nv, b, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
v = np.loadtxt(bvec)
if v.ndim != 2: raise SystemExit("bvec is not 2D")
if v.shape[0] != 3: v = v.T
if v.shape[1] != nv: raise SystemExit(f"bvec cols {v.shape[1]} != DWI vols {nv}")
b0 = np.linalg.norm(v, axis=0) < 0.1
if b0.sum() < 1: raise SystemExit("no b0 (zero-norm bvec column) found -- refusing to synthesize bvals")
vals = np.where(b0, 0.0, b)
open(out, "w").write(" ".join("%g" % x for x in vals) + "\n")
print(f"synthesized single-shell bvals: {int(b0.sum())} b0 + {int((~b0).sum())} x b{b:g} ({nv} vols)")
PY
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
