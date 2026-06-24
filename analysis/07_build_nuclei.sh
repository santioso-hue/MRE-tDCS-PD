#!/bin/bash
# 07_build_nuclei.sh - Group 2 deep stimulation-relevant nuclei ROIs from CIT168/Pauli, warped to subject
# space (L/R split). Basal ganglia (Pu, Ca, NAC, GPe, GPi) plus midbrain and subthalamic nuclei (SNc, SNr,
# VTA, RN, STN). Reuses the CIT168 atlas cache (2009c->NLin6 affine + pauli_prob) staged under
# registration/atlas_rois/_atlas_cache/, a cluster/manual artifact not built by an in-repo script.
#
# Usage: PIPELINE_CONFIG=<subject config.sh> bash analysis/07_build_nuclei.sh
#
# All Group 2 nuclei are EXPLORATORY and E-field-only (overlap-allowed masks, no MRE cross-corr). The basal
# ganglia are well-resolved; the midbrain and subthalamic nuclei are at or below a 2.5 mm dMRI / 3 mm MRE
# voxel and additionally overlap each other, so read those as order-of-magnitude field exposure (see §2.7).
# Only CIT168 resolves these (FreeSurfer/HarvardOxford cannot).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${PIPELINE_CONFIG:-$SCRIPT_DIR/../config/config.sh}"
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG (set PIPELINE_CONFIG)"; exit 1; }
source "$CFG"
export FSLDIR
export PATH="$SIMNIBS_BIN:$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ

T1REF="$M2M_DIR/T1.nii.gz"
NLIN6="$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz"
OUT="$REG_DIR/atlas_rois"; CACHE="$OUT/_atlas_cache"; ND="$OUT/nuclei"; mkdir -p "$ND"
MAT="$CACHE/2009c_to_nlin6.mat"; PAULI="$CACHE/pauli_prob.nii.gz"

for f in "$MAT" "$PAULI" "$T1REF"; do
  [ -f "$f" ] || { echo "ERROR: missing $f. Stage the CIT168 atlas cache (_atlas_cache/) and run charm for T1.nii.gz first; the cache is a cluster/manual artifact, not produced by any in-repo script."; exit 1; }
done

# ---- Group 2 config: every magic number lives here (PROB_THR is passed to _build_nuclei_labels.py) ----
# CIT168/Pauli 4D volume indices, verified from _atlas_cache/pauli_labels.txt.
NUCLEI=(Pu Ca NAC GPe GPi SNc SNr VTA RN STN)
IDX=(   0  1  2   4   5   6   8   10  7  15)
PROB_THR=0.25                    # CIT168 probability cutoff (tiny nuclei -> permissive)

echo "Warp Group 2 nuclei (CIT168 2009c -> NLin6 -> subject)"
for n in "${!NUCLEI[@]}"; do
  nm="${NUCLEI[$n]}"; i="${IDX[$n]}"
  fslroi "$PAULI" "$ND/_mni_${nm}.nii.gz" "$i" 1
  flirt -in "$ND/_mni_${nm}.nii.gz" -ref "$NLIN6" -applyxfm -init "$MAT" \
        -out "$ND/_nlin6_${nm}.nii.gz" -interp trilinear >/dev/null 2>&1
  mni2subject -i "$ND/_nlin6_${nm}.nii.gz" -m "$M2M_DIR" -o "$ND/sub_${nm}.nii.gz" \
              -r "$T1REF" --interpolation_order 1 >/dev/null 2>&1
  echo "  $nm warped"
done

NUCLEI_THR="$PROB_THR" NUCLEI_LIST="${NUCLEI[*]}" "$SIMNIBS_BIN/simnibs_python" "$SCRIPT_DIR/_build_nuclei_labels.py"
echo "Done. Group 2 nuclei -> $ND/roi_{L,R}_{Pu,Ca,NAC,GPe,GPi,SNc,SNr,VTA,RN,STN}.nii.gz (+ nuclei_labeled.nii.gz)."
