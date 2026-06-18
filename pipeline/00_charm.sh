#!/bin/bash
# 00_charm.sh — build the 5-tissue head model from T1+T2 with SimNIBS charm (standard defaults).
# Produces a tetrahedral FEM mesh in m2m_<SUBJECT_ID>/.
#
# Usage:   bash pipeline/00_charm.sh <SUBJECT_ID> <T1.nii> <T2.nii> <output_dir> [settings.ini]
# Example: bash pipeline/00_charm.sh "$SUBJECT" "$WORK_DIR/T1w.nii" "$WORK_DIR/T2w.nii" "$WORK_DIR"
#
# With no settings.ini, charm uses the SimNIBS defaults; pass a custom .ini as the 5th arg to override.
# --forceqform avoids a qform/sform mismatch abort (e.g. after resampling T1/T2 to 1 mm with flirt -applyisoxfm).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SUBJECT_ID="${1:-}"
T1="${2:-}"
T2="${3:-}"
OUT_DIR="${4:-}"
INI="${5:-}"

if [ -z "$SUBJECT_ID" ] || [ -z "$T1" ] || [ -z "$T2" ] || [ -z "$OUT_DIR" ]; then
    echo "Usage: bash 00_charm.sh <SUBJECT_ID> <T1.nii> <T2.nii> <output_dir> [settings.ini]"
    exit 1
fi

if [ -n "$INI" ]; then
    case "$INI" in /*) INI_PATH="$INI" ;; *) INI_PATH="$REPO_ROOT/$INI" ;; esac
    [ -f "$INI_PATH" ] || { echo "ERROR: charm settings file not found: $INI_PATH"; exit 1; }
fi

export PATH=~/Applications/SimNIBS-4.6/bin:$PATH
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "charm: subject=$SUBJECT_ID  T1=$T1  T2=$T2  output=$OUT_DIR  settings=${INI:-defaults}"

# pipefail propagates charm's exit through the tee, so a failed segmentation aborts here
if [ -n "$INI" ]; then
    charm "$SUBJECT_ID" "$T1" "$T2" --forceqform --usesettings "$INI_PATH" 2>&1 | tee "charm_${SUBJECT_ID}.log"
else
    charm "$SUBJECT_ID" "$T1" "$T2" --forceqform 2>&1 | tee "charm_${SUBJECT_ID}.log"
fi

[ -d "m2m_${SUBJECT_ID}" ] || { echo "ERROR: charm did not produce m2m_${SUBJECT_ID}/ — see charm_${SUBJECT_ID}.log"; exit 1; }

echo "Done. Head model: $OUT_DIR/m2m_${SUBJECT_ID}/"
