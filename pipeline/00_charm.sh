#!/bin/bash
# 00_charm.sh — Build 5-tissue head model from T1+T2 MRI using SimNIBS charm
#
# Produces a tetrahedral FEM mesh in m2m_<SUBJECT_ID>/ for tDCS simulation.
# Uses high-quality settings (see config/charm_highquality.ini) tuned for
# macOS Apple Silicon and T1+T2 input.
#
# Prerequisites:
#   SimNIBS 4.6 at ~/Applications/SimNIBS-4.6/
#   T1 and T2 images reoriented to standard space and resampled to 1 mm isotropic
#   (use fslreorient2std + flirt -applyisoxfm 1 + --forceqform)
#
# Usage:
#   bash 00_charm.sh <SUBJECT_ID> <T1.nii> <T2.nii> <output_dir> <ini_file>
#
# Example:
#   bash pipeline/00_charm.sh "$SUBJECT" \
#        $WORK_DIR/T1w.nii \
#        $WORK_DIR/T2w.nii \
#        $WORK_DIR \
#        config/charm_highquality.ini
#
# Runtime: ~1 hour on Apple Silicon M-series
# macOS fixes applied:
#   OMP_NUM_THREADS=1       — prevents OpenMP deadlock
#   KMP_DUPLICATE_LIB_OK=TRUE — prevents OpenMP library conflict
#   caffeinate -i           — prevents macOS sleep during long run
#
# IMPORTANT: If T1/T2 were prepared with flirt -applyisoxfm, add --forceqform
# to charm to avoid a qform/sform mismatch error.

set -euo pipefail

# Resolve the repo root BEFORE any cd, so --ini is stable regardless of CWD or how the script is
# invoked. After `cd "$OUT_DIR"`, a `$0`-relative path would resolve against the wrong directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SUBJECT_ID="${1:-}"
T1="${2:-}"
T2="${3:-}"
OUT_DIR="${4:-}"
INI="${5:-config/charm_highquality.ini}"

if [ -z "$SUBJECT_ID" ] || [ -z "$T1" ] || [ -z "$T2" ] || [ -z "$OUT_DIR" ]; then
    echo "Usage: bash 00_charm.sh <SUBJECT_ID> <T1.nii> <T2.nii> <output_dir> [ini_file]"
    exit 1
fi

# INI path: keep absolute paths as-is, otherwise resolve relative to the repo root; verify it exists.
case "$INI" in
    /*) INI_PATH="$INI" ;;
    *)  INI_PATH="$REPO_ROOT/$INI" ;;
esac
[ -f "$INI_PATH" ] || { echo "ERROR: charm INI not found: $INI_PATH"; exit 1; }

export PATH=~/Applications/SimNIBS-4.6/bin:$PATH
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "=== SimNIBS charm — 5-tissue head model ==="
echo "Subject:  $SUBJECT_ID"
echo "T1:       $T1"
echo "T2:       $T2"
echo "Output:   $OUT_DIR"
echo "Settings: $INI"
echo "Started:  $(date)"
echo ""

caffeinate -i \
  charm "$SUBJECT_ID" "$T1" "$T2" \
    --forceqform \
    --ini "$INI_PATH" \
    2>&1 | tee "charm_${SUBJECT_ID}.log"

# `set -o pipefail` makes this pipeline return charm's exit status, not tee's, so a failed
# segmentation now aborts instead of being silently reported as success. Belt-and-suspenders:
# confirm the head-model directory was actually produced.
[ -d "m2m_${SUBJECT_ID}" ] || { echo "ERROR: charm did not produce m2m_${SUBJECT_ID}/ — see charm_${SUBJECT_ID}.log"; exit 1; }

echo ""
echo "=== charm complete: $(date) ==="
echo "Head model: $OUT_DIR/m2m_${SUBJECT_ID}/"
echo ""
echo "Next steps:"
echo "  1. bash pipeline/01_dwi2cond.sh  (DTI tensor for the DTI baseline)"
echo "  2. bash pipeline/02_register_dmri_to_T1.sh"
echo "  3. simnibs_python pipeline/03_build_conductivity_tensor.py"
echo "  4. simnibs_python pipeline/04_run_simulations.py"
