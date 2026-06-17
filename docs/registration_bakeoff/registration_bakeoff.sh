#!/bin/bash
# registration_bakeoff.sh — per-subject AFFINE vs FNIRT bake-off for dMRI(<D>) -> charm-T1.
#
# Decides the registration CLASS for one subject by running both arms from the SAME flirt affine init,
# so the ONLY difference is the fnirt nonlinear step (does it help, or over-warp already-distortion-
# corrected data?). Both arms reorient v1 with vecreg (identical tool/frame), so the orientation score
# compares like with like. (An ANTs-affine toolkit cross-check arm may be added later; not implemented
# yet.) Scoring is done by analysis/score_registration_bakeoff.py.
#
# Reads the dMRI-space maps prepared by prepare_dmri_tensor.py (REG_DIR): FA_dMRI, v1_dMRI, dMRI_mask.
# Target = charm m2m T1 (so outputs live in the FEM mesh grid). Writes REG_DIR/bakeoff/{affine,fnirt}/.
#
# Usage:  PIPELINE_CONFIG=<subject config.sh> bash pipeline/registration_bakeoff.sh
# Runtime: ~8 min (fnirt dominates).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${PIPELINE_CONFIG:-$SCRIPT_DIR/../config/config.sh}"
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG (set PIPELINE_CONFIG)"; exit 1; }
# shellcheck disable=SC1090
source "$CFG"
export FSLDIR
export PATH="$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ

REG="$REG_DIR"
T1_REF="$M2M_DIR/T1.nii.gz"
FA_DMRI="$REG/FA_dMRI.nii.gz"
V1_DMRI="$REG/v1_dMRI.nii.gz"
for f in "$T1_REF" "$FA_DMRI" "$V1_DMRI" "$M2M_DIR/final_tissues.nii.gz"; do
  [ -f "$f" ] || { echo "ERROR: required input missing: $f"; exit 1; }
done

OUT="$REG/bakeoff"; mkdir -p "$OUT/affine" "$OUT/fnirt"

require_nonzero() {
  local f="$1"
  [ -f "$f" ] || { echo "ERROR: expected output missing: $f"; exit 1; }
  local nz; nz="$(fslstats "$f" -V | awk '{print $1}')"
  [ "${nz:-0}" -gt 0 ] 2>/dev/null || { echo "ERROR: $f has 0 non-zero voxels"; exit 1; }
}

echo "=== Brain-extract charm T1 (registration cost target) ==="
T1_BRAIN="$OUT/T1_brain.nii.gz"
"$SIMNIBS_BIN/simnibs_python" - "$M2M_DIR" "$T1_BRAIN" <<'PY'
import sys, os, numpy as np, nibabel as nib
m2m, out = sys.argv[1], sys.argv[2]
t1 = nib.load(os.path.join(m2m, "T1.nii.gz"))
seg = nib.load(os.path.join(m2m, "final_tissues.nii.gz")).get_fdata()
seg = seg[..., 0] if seg.ndim == 4 else seg
brain = np.isin(seg, [1, 2, 3]).astype(np.float32)   # WM, GM, CSF
nib.save(nib.Nifti1Image((t1.get_fdata() * brain).astype(np.float32), t1.affine, t1.header), out)
print("  wrote", out)
PY
require_nonzero "$T1_BRAIN"

echo ""
echo "=== Shared flirt 12-DOF affine init: FA(<D>) -> charm T1_brain (Mattes MI) ==="
AFF="$OUT/dMRI_to_T1_aff.mat"
flirt -in "$FA_DMRI" -ref "$T1_BRAIN" -omat "$AFF" -dof 12 -cost mutualinfo \
      -searchrx -25 25 -searchry -25 25 -searchrz -25 25
[ -s "$AFF" ] || { echo "ERROR: flirt affine init failed"; exit 1; }

echo ""
echo "=== ARM 1 (affine): apply the affine to FA + v1 (vecreg -t affine) ==="
flirt -in "$FA_DMRI" -ref "$T1_REF" -applyxfm -init "$AFF" -interp trilinear -out "$OUT/affine/FA_T1.nii.gz"
vecreg -i "$V1_DMRI" -o "$OUT/affine/v1_T1.nii.gz" -r "$T1_REF" -t "$AFF"
require_nonzero "$OUT/affine/FA_T1.nii.gz"
require_nonzero "$OUT/affine/v1_T1.nii.gz"

echo ""
echo "=== ARM 2 (fnirt): nonlinear warp from the same affine init, then vecreg -w warp ==="
WARP="$OUT/dMRI_to_T1_warp"
fnirt --in="$FA_DMRI" --ref="$T1_BRAIN" --aff="$AFF" --cout="$WARP" --subsamp=8,4,2,2
[ -f "$WARP.nii.gz" ] || { echo "ERROR: fnirt did not produce $WARP"; exit 1; }
applywarp -i "$FA_DMRI" -r "$T1_REF" -w "$WARP" -o "$OUT/fnirt/FA_T1.nii.gz" --interp=trilinear
vecreg -i "$V1_DMRI" -o "$OUT/fnirt/v1_T1.nii.gz" -r "$T1_REF" -w "$WARP"
require_nonzero "$OUT/fnirt/FA_T1.nii.gz"
require_nonzero "$OUT/fnirt/v1_T1.nii.gz"

echo ""
echo "=== NONLINEAR over-warp field: |total fnirt warp - affine| (isolates deformation beyond the affine) ==="
# The full fnirt warp includes the affine (which also carries the 2.5mm->1mm scale change), so the full
# warp's Jacobian/displacement is dominated by the affine and is NOT an over-warp measure. The nonlinear
# part = total relative field minus the affine's relative field. A topup-corrected cohort should need
# little of this; a large nonlinear displacement means fnirt is inventing deformation.
convertwarp --ref="$T1_REF" --premat="$AFF"   --relout --out="$OUT/fnirt/field_affine.nii.gz"
convertwarp --ref="$T1_REF" --warp1="$WARP"   --relout --out="$OUT/fnirt/field_total.nii.gz"
fslmaths "$OUT/fnirt/field_total.nii.gz" -sub "$OUT/fnirt/field_affine.nii.gz" "$OUT/fnirt/field_nonlinear.nii.gz"
echo "  wrote fnirt/field_nonlinear.nii.gz (3-vol relative displacement, mm); scorer reports its magnitude in GM+WM"

echo ""
echo "=== Bake-off arms written to $OUT/{affine,fnirt}/ ==="
echo "  affine/FA_T1.nii.gz affine/v1_T1.nii.gz   (12-DOF affine)"
echo "  fnirt/FA_T1.nii.gz  fnirt/v1_T1.nii.gz    (affine + nonlinear warp)"
echo "Next: <simnibs_python> analysis/score_registration_bakeoff.py"
