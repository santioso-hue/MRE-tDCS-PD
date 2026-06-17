#!/bin/bash
# run_qti_cov_cohort.sh — drive the cohort QTI covariance fit (run_qti_cov_cohort.m) for one subject.
#
# Input:  <subject_dir>/dmri/ (linear_corrected + spherical_corrected + bval/bvec)
# Output: <subject_dir>/fit/qti_cov/{cov_mfs.mat, cov_dps.mat} (oriented <D>)
# Usage:  bash pipeline/run_qti_cov_cohort.sh <subject_dir>
set -euo pipefail
SUBJDIR="${1:?usage: run_qti_cov_cohort.sh <subject_dir with dmri/ and fit/>}"
SUBJDIR="$(cd "$SUBJDIR" && pwd)"   # absolutize: md-dmri setup_paths.m changes cwd, breaking relative paths
export DMRI_IN="$SUBJDIR/dmri"
export FIT_OUT="$SUBJDIR/fit"
export MDDMRI_DIR="${MDDMRI_DIR:-$HOME/md-dmri-master}"
[ -d "$MDDMRI_DIR" ] || { echo "ERROR: md-dmri toolbox not found at $MDDMRI_DIR (set MDDMRI_DIR to your md-dmri checkout)"; exit 1; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${ELASTIX_BIN:-$HOME/Applications/elastix-5.3.1/bin}:$PATH"   # elastix + transformix for step2 MC
command -v elastix >/dev/null || { echo "ERROR: elastix not on PATH (needed for step2 inter-series MC)"; exit 1; }
cd "$REPO"
/Applications/MATLAB_R2026a.app/bin/matlab -batch "run('pipeline/run_qti_cov_cohort.m')"
