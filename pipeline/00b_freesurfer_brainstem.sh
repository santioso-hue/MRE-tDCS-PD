#!/bin/bash
# 00b_freesurfer_brainstem.sh — FreeSurfer recon-all + brainstem substructure
# segmentation (Iglesias et al. 2015) to obtain the MESENCEPHALON (Midbrain) ROI for
# the Tier-1 E-field/MRE analysis.
#
# Why: CHARM/SAMSEG (our head-model segmentation) gives tissue classes + cortical
# surfaces but NOT brainstem sub-parcels. The midbrain block is only available from
# FreeSurfer's brainstem module, so we run FreeSurfer once on the subject T1.
#
# Runtime: recon-all -all -parallel ~3 h (Apple Silicon, FS 8) + brainstem ~30-60 min.
# Idempotent: skips recon-all if it already finished (aseg.mgz present).
#
# Output: $WORK_DIR/freesurfer/$SUBJECT/mri/brainstemSsLabels*.mgz
#   Labels (typical Iglesias IDs — the ROI builder reads them, don't hardcode):
#     Midbrain = the mesencephalon ROI; also Pons, Medulla, SCP, Whole_brainstem.
#   The label is in FreeSurfer conformed space; map it to T1 space in the ROI builder
#   (mri_vol2vol --regheader against rawavg/orig, or via the m2m T1 affine).
#
# Usage:  bash pipeline/00b_freesurfer_brainstem.sh [n_threads]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"

# FreeSurfer env (falls back to the standard macOS install + home license).
export FREESURFER_HOME="${FREESURFER_HOME:-/Applications/freesurfer/8.1.0}"
export FS_LICENSE="${FS_LICENSE:-$HOME/license.txt}"
[ -f "$FS_LICENSE" ] || { echo "ERROR: FreeSurfer license not found at $FS_LICENSE"; exit 1; }
# FreeSurfer's setup script isn't written for "strict mode" — it references unset variables
# (fatal under `set -u`) and can return non-zero (fatal under `set -e`). Disable BOTH just for
# the source, then restore them for our own code.
set +eu
source "$FREESURFER_HOME/SetUpFreeSurfer.sh" >/dev/null
set -eu

# SUBJECTS_DIR default (/Applications/...) is root-owned/read-only — use a writable project dir.
export SUBJECTS_DIR="$WORK_DIR/freesurfer"
mkdir -p "$SUBJECTS_DIR"

T1="$M2M_DIR/T1.nii.gz"           # CHARM-conformed full-head T1 (recon-all re-conforms internally)
THREADS="${1:-6}"
[ -f "$T1" ] || { echo "ERROR: $T1 not found (run CHARM first)."; exit 1; }

echo "FREESURFER_HOME=$FREESURFER_HOME"
echo "SUBJECTS_DIR=$SUBJECTS_DIR   subject=$SUBJECT   T1=$T1   threads=$THREADS"

if [ -f "$SUBJECTS_DIR/$SUBJECT/mri/aseg.mgz" ]; then
  echo "=== recon-all already complete for $SUBJECT (aseg.mgz present) — skipping ==="
else
  echo "=== recon-all -all  (~3 h with -parallel) ==="
  recon-all -all -s "$SUBJECT" -i "$T1" -parallel -threads "$THREADS"
fi

echo "=== brainstem substructures (segment_subregions brainstem, Iglesias) ==="
segment_subregions brainstem --cross "$SUBJECT" --threads "$THREADS"

echo "Done. Brainstem labels:"
ls -1 "$SUBJECTS_DIR/$SUBJECT/mri/"brainstemSsLabels*.mgz 2>/dev/null || echo "  (expected brainstemSsLabels*.mgz not found — check the segment_subregions log)"
echo "Next: map the Midbrain label to T1 space and add it to the Tier-1 ROI set."
