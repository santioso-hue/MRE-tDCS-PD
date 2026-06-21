#!/bin/bash
# 00_charm.sh - build the 5-tissue head model from T1+T2 with SimNIBS charm (standard defaults).
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

# Single-thread charm: the multithreaded samseg affine registration can livelock on macOS (spins at full CPU,
# never finishing). The OMP/KMP/ITK thread limits avoid it; scoped here so other stages stay multithreaded.
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
# Watchdog: kill charm if it exceeds CHARM_TIMEOUT so a hung subject fails fast instead of blocking the cohort.
CHARM_TIMEOUT="${CHARM_TIMEOUT:-10800}"
LOG="charm_${SUBJECT_ID}.log"
if [ -n "$INI" ]; then
    charm "$SUBJECT_ID" "$T1" "$T2" --forceqform --usesettings "$INI_PATH" > "$LOG" 2>&1 &
else
    charm "$SUBJECT_ID" "$T1" "$T2" --forceqform > "$LOG" 2>&1 &
fi
cpid=$!
( set +e
  e=0
  while kill -0 "$cpid" 2>/dev/null; do
      sleep 30; e=$((e+30))
      if [ "$e" -ge "$CHARM_TIMEOUT" ]; then
          echo "WATCHDOG: charm exceeded ${CHARM_TIMEOUT}s, killing $cpid" >> "$LOG"
          pkill -9 -P "$cpid" 2>/dev/null   # reap charm's children first, then charm itself
          kill -9 "$cpid" 2>/dev/null
          break
      fi
  done ) &
wpid=$!
rc=0; wait "$cpid" || rc=$?
kill "$wpid" 2>/dev/null || true
if [ "$rc" -ne 0 ]; then
    echo "ERROR: charm failed or timed out (rc=$rc) - see $OUT_DIR/$LOG"
    tail -8 "$LOG" 2>/dev/null || true
    exit 1
fi

[ -d "m2m_${SUBJECT_ID}" ] || { echo "ERROR: charm did not produce m2m_${SUBJECT_ID}/ - see charm_${SUBJECT_ID}.log"; exit 1; }

echo "Done. Head model: $OUT_DIR/m2m_${SUBJECT_ID}/"
