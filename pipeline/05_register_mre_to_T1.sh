#!/bin/bash
# 05_register_mre_to_T1.sh — Bring MRE mechanical maps into T1 space for the
# post-hoc cross-modal comparison (mechanics vs MD-dMRI microstructure / E-field).
#
# MRE is acquired in its own low-resolution space (here 1.5x1.5x3 mm). We rigidly
# register the stiffness map (which carries tissue contrast) to the structural T1
# and apply the same transform to the storage/loss moduli and confidence maps.
#
# NOTE: a dedicated MRE magnitude/T2*-EPI image, if available, is a better
# registration source than the stiffness map. Swap MRE_REF below if you have one.
#
# Prerequisites: CHARM (m2m_${SUBJECT}/T1.nii.gz). Reads paths from config/config.sh.
# Output: registration/mre_{stiffness,storage,loss,confidence}_T1.nii.gz

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"
export FSLDIR
export PATH="$SIMNIBS_BIN:$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ

T1_REF="$M2M_DIR/T1.nii.gz"
MRE_REF="$MRE_STIFFNESS"      # registration source (swap for an MRE magnitude if available)
mkdir -p "$REG_DIR"; cd "$REG_DIR"

[ -f "$T1_REF" ] || { echo "ERROR: $T1_REF not found (run CHARM first)."; exit 1; }
[ -f "$MRE_REF" ] || { echo "ERROR: $MRE_REF not found (check MRE_* in config)."; exit 1; }

echo "=== Register MRE stiffness -> T1 (6-DOF, mutual information) ==="
flirt -in "$MRE_REF" -ref "$T1_REF" -out mre_stiffness_T1.nii.gz \
      -omat mre_to_T1.mat -dof 6 -cost mutualinfo \
      -searchrx -30 30 -searchry -30 30 -searchrz -30 30 -interp trilinear
echo "  transform: mre_to_T1.mat   (VISUALLY CHECK mre_stiffness_T1 vs T1 in FSLeyes)"

echo "=== Apply transform to storage / loss / confidence ==="
for pair in "MRE_STORAGE mre_storage_T1" "MRE_LOSS mre_loss_T1" "MRE_CONFIDENCE mre_confidence_T1"; do
  set -- $pair; src="${!1}"; out="$2"
  if [ -f "$src" ]; then
    flirt -in "$src" -ref "$T1_REF" -out "${out}.nii.gz" \
          -applyxfm -init mre_to_T1.mat -interp trilinear
    echo "  $out.nii.gz"
  fi
done
echo "Done. Next: simnibs_python analysis/05_mre_efield_comparison.py"
