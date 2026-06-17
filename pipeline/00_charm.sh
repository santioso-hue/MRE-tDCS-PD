#!/bin/bash
# 00_charm.sh — build the 5-tissue head model from T1+T2 with SimNIBS charm (high-quality settings
# in config/charm_highquality.ini). Produces a tetrahedral FEM mesh in m2m_<SUBJECT_ID>/.
#
# Usage:   bash pipeline/00_charm.sh <SUBJECT_ID> <T1.nii> <T2.nii> <output_dir> [ini_file]
# Example: bash pipeline/00_charm.sh "$SUBJECT" "$WORK_DIR/T1w.nii" "$WORK_DIR/T2w.nii" "$WORK_DIR" config/charm_highquality.ini
#
# charm runs with --forceqform so a qform/sform mismatch (e.g. after resampling T1/T2 to 1 mm
# isotropic with flirt -applyisoxfm) does not abort segmentation.

set -euo pipefail

# Resolve the repo root BEFORE any cd, so --usesettings is stable regardless of CWD or how the script is
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

echo "charm: subject=$SUBJECT_ID  T1=$T1  T2=$T2  output=$OUT_DIR  settings=$INI"

caffeinate -i \
  charm "$SUBJECT_ID" "$T1" "$T2" \
    --forceqform \
    --usesettings "$INI_PATH" \
    2>&1 | tee "charm_${SUBJECT_ID}.log"

# `set -o pipefail` makes this pipeline return charm's exit status, not tee's, so a failed
# segmentation now aborts instead of being silently reported as success. Belt-and-suspenders:
# confirm the head-model directory was actually produced.
[ -d "m2m_${SUBJECT_ID}" ] || { echo "ERROR: charm did not produce m2m_${SUBJECT_ID}/ — see charm_${SUBJECT_ID}.log"; exit 1; }

echo "Done. Head model: $OUT_DIR/m2m_${SUBJECT_ID}/"
