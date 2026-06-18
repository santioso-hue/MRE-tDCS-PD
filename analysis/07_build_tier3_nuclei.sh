#!/bin/bash
# 07_build_tier3_nuclei.sh — Tier-3 small PD nuclei ROIs (SNc, SNr, VTA, RN, STN L/R) from
# CIT168/Pauli, warped to subject space. Reuses the CIT168 atlas cache (2009c->NLin6 affine + pauli_prob)
# staged under registration/atlas_rois/_atlas_cache/, a cluster/manual artifact not built by an in-repo script.
#
# Usage: bash analysis/07_build_tier3_nuclei.sh
#
# These nuclei are smaller than a 2.5 mm dMRI / 3 mm MRE voxel and sit within a few mm of each other,
# so they are EXPLORATORY and E-field-only: overlap-allowed masks, no MRE cross-corr, NOT headline
# numbers (an SNc |E| is not a reportable result). Only CIT168 resolves them (FreeSurfer/HarvardOxford cannot).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"
export FSLDIR
export PATH="$SIMNIBS_BIN:$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ

T1REF="$M2M_DIR/T1.nii.gz"
NLIN6="$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz"
OUT="$REG_DIR/atlas_rois"; CACHE="$OUT/_atlas_cache"; T3="$OUT/tier3"; mkdir -p "$T3"
MAT="$CACHE/2009c_to_nlin6.mat"; PAULI="$CACHE/pauli_prob.nii.gz"

for f in "$MAT" "$PAULI" "$T1REF"; do
  [ -f "$f" ] || { echo "ERROR: missing $f. Stage the CIT168 atlas cache (_atlas_cache/) and run charm for T1.nii.gz first; the cache is a cluster/manual artifact, not produced by any in-repo script."; exit 1; }
done

# ---- Tier-3 config: every tier-3 magic number lives here (PROB_THR is passed to _build_tier3_labels.py) ----
# CIT168/Pauli 4D volume indices, verified from _atlas_cache/pauli_labels.txt.
NUCLEI=(SNc SNr VTA RN STN)
IDX=(   6   8   10  7  15)
PROB_THR=0.25                    # CIT168 probability cutoff (tiny nuclei -> permissive)

echo "Warp Tier-3 nuclei (CIT168 2009c -> NLin6 -> subject)"
for n in "${!NUCLEI[@]}"; do
  nm="${NUCLEI[$n]}"; i="${IDX[$n]}"
  fslroi "$PAULI" "$T3/_mni_${nm}.nii.gz" "$i" 1
  flirt -in "$T3/_mni_${nm}.nii.gz" -ref "$NLIN6" -applyxfm -init "$MAT" \
        -out "$T3/_nlin6_${nm}.nii.gz" -interp trilinear >/dev/null 2>&1
  mni2subject -i "$T3/_nlin6_${nm}.nii.gz" -m "$M2M_DIR" -o "$T3/sub_${nm}.nii.gz" \
              -r "$T1REF" --interpolation_order 1 >/dev/null 2>&1
  echo "  $nm warped"
done

TIER3_THR="$PROB_THR" TIER3_NUCLEI="${NUCLEI[*]}" "$SIMNIBS_BIN/simnibs_python" "$SCRIPT_DIR/_build_tier3_labels.py"
echo "Done. Tier-3 nuclei -> $T3/roi_{L,R}_{SNc,SNr,VTA,RN,STN}.nii.gz (+ tier3_labeled.nii.gz)."
