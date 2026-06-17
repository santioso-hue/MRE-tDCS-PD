#!/bin/bash
# registration_bakeoff_s0.sh — add an S0(b0)-driven AFFINE arm to the bake-off.
#
# Diagnostic for the weak FA-driven affine (CC ~45deg off, r=0.71): FA is a poor registration driver
# (noisy, WM-only). The ParkMRE pipeline (Olsson et al. 2025) drives registration off the dMRI S0/b0
# (whole-brain, T2-like contrast). This registers the dMRI b0 -> charm T2 (same T2 weighting => robust,
# matching that S0->T2 pairing) to get a dMRI->charm affine, then applies it to FA + v1 (vecreg -t). Output: REG_DIR/bakeoff/
# affine_s0/. Score with analysis/score_registration_bakeoff.py (affine_s0 is in its ARMS list).
#
# Usage:  PIPELINE_CONFIG=<subject config.sh> bash pipeline/registration_bakeoff_s0.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${PIPELINE_CONFIG:-$SCRIPT_DIR/../config/config.sh}"
# shellcheck disable=SC1090
source "$CFG"
export FSLDIR; export PATH="$FSLDIR/bin:$PATH"; export FSLOUTPUTTYPE=NIFTI_GZ

REG="$REG_DIR"; OUT="$REG/bakeoff"; T1_REF="$M2M_DIR/T1.nii.gz"
B0="$FIT_DIR/qti_cov/b0_ref_dMRI.nii.gz"          # dMRI b0, on the FA_dMRI/v1_dMRI grid
FA_DMRI="$REG/FA_dMRI.nii.gz"; V1_DMRI="$REG/v1_dMRI.nii.gz"
T2_REG="$M2M_DIR/T2_reg.nii.gz"                    # charm T2 (charm/T1 space) = same contrast as b0
for f in "$B0" "$FA_DMRI" "$V1_DMRI" "$T2_REG" "$T1_REF" "$M2M_DIR/final_tissues.nii.gz"; do
  [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
mkdir -p "$OUT/affine_s0"

echo "=== brain-extract charm T2 (registration target, parallels the FA arm's T1_brain) ==="
T2_BRAIN="$OUT/T2_brain.nii.gz"
"$SIMNIBS_BIN/simnibs_python" - "$M2M_DIR" "$T2_BRAIN" <<'PY'
import sys, os, numpy as np, nibabel as nib
m2m, out = sys.argv[1], sys.argv[2]
t2 = nib.load(os.path.join(m2m, "T2_reg.nii.gz"))
seg = nib.load(os.path.join(m2m, "final_tissues.nii.gz")).get_fdata(); seg = seg[...,0] if seg.ndim==4 else seg
brain = np.isin(seg, [1,2,3]).astype(np.float32)
nib.save(nib.Nifti1Image((t2.get_fdata()*brain).astype(np.float32), t2.affine, t2.header), out)
print("  wrote", out)
PY

echo "=== FLIRT 12-DOF affine: dMRI b0 -> charm T2_brain (same T2 contrast; Mattes MI) ==="
AFF="$OUT/dMRI_to_T1_aff_s0.mat"
flirt -in "$B0" -ref "$T2_BRAIN" -omat "$AFF" -dof 12 -cost mutualinfo \
      -searchrx -25 25 -searchry -25 25 -searchrz -25 25
[ -s "$AFF" ] || { echo "ERROR: b0->T2 affine failed"; exit 1; }

echo "=== apply the S0-driven affine to FA + v1 (vecreg -t) ==="
flirt -in "$FA_DMRI" -ref "$T1_REF" -applyxfm -init "$AFF" -interp trilinear -out "$OUT/affine_s0/FA_T1.nii.gz"
vecreg -i "$V1_DMRI" -o "$OUT/affine_s0/v1_T1.nii.gz" -r "$T1_REF" -t "$AFF"
echo "=== done -> $OUT/affine_s0/ (FA_T1, v1_T1). Re-run the scorer. ==="
