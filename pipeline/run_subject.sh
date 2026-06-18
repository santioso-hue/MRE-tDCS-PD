#!/bin/bash
# run_subject.sh -- run the per-subject pipeline (ISO + MD-dMRI, plus the DTI arm when a DTI scan is wired
# in the config) for one subject, then check the outputs reproduce the recorded baseline. Also the unit the
# cohort scale-out calls per subject.
#
# Usage: PIPELINE_CONFIG=<subject config.sh> bash pipeline/run_subject.sh [--from fit|charm] [--montage M1]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG="${PIPELINE_CONFIG:?set PIPELINE_CONFIG to the subject config.sh}"
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
# shellcheck disable=SC1090
source "$CFG"
export PIPELINE_CONFIG="$CFG"
export MDDMRI_DIR="${MDDMRI_DIR:?set MDDMRI_DIR in the subject config (md-dmri toolbox path)}"

FROM="fit"; MONTAGE="M1"
while [ $# -gt 0 ]; do
    case "$1" in
        --from)    FROM="$2";    shift 2 ;;
        --montage) MONTAGE="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

SP="$SIMNIBS_BIN/simnibs_python"
RESULTS="$REPO/analysis/results/$SUBJECT"
mkdir -p "$RESULTS"

# provenance: best-effort version capture, isolated so a missing tool never aborts the run
record_env() (
    set +e +o pipefail
    {
        echo "subject=$SUBJECT from=$FROM montage=$MONTAGE"
        "$SP" -c "import simnibs; print('simnibs', simnibs.__version__)" 2>/dev/null
        "$FSLDIR/bin/flirt" -version 2>/dev/null | head -1
        recon-all --version 2>/dev/null | head -1
        elastix --version 2>/dev/null | head -1
        echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset} KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-unset}"
    } > "$RESULTS/provenance.txt" 2>&1
)

stage() { echo; echo ">>> $*"; }

record_env; echo "wrote $RESULTS/provenance.txt"

stage "unit tests"
for t in test_lobe_grouping test_cohort_stats test_qc_harness test_tensor_divergence; do
    "$SP" "$REPO/tests/$t.py"
done

if [ "$FROM" = "charm" ]; then
    : "${CHARM_T1:?set CHARM_T1 in the config for --from charm}"
    : "${CHARM_T2:?set CHARM_T2 in the config for --from charm}"
    # Build charm only when the head model is absent (final_tissues.nii.gz is its completion marker) or
    # FORCE_CHARM=1. A bare cohort re-run after a crash then keeps the hours-long charm of done subjects
    # instead of rebuilding it from scratch.
    if [ ! -f "$M2M_DIR/final_tissues.nii.gz" ] || [ "${FORCE_CHARM:-0}" = "1" ]; then
        stage "00 charm head model"
        rm -rf "$M2M_DIR"                          # rebuild from scratch (set FORCE_CHARM=1 to force)
        bash "$REPO/pipeline/00_charm.sh" "$SUBJECT" "$CHARM_T1" "$CHARM_T2" "$WORK_DIR"
    else
        stage "00 charm head model (exists, skipping; FORCE_CHARM=1 to rebuild)"
    fi
fi

stage "recon-all ROIs"
"$SP" "$REPO/analysis/build_rois.py" --fs_dir "$DATA_DIR/recon"

stage "QTI covariance fit"
bash "$REPO/pipeline/run_qti_cov_cohort.sh" "$DATA_DIR"

stage "validate regenerated mean tensor"
"$SP" "$REPO/tests/validate_mean_tensor.py"

stage "register dMRI to T1"
bash "$REPO/pipeline/02_register_dmri_to_T1.sh"

stage "build conductivity tensor"
"$SP" "$REPO/pipeline/03_build_conductivity_tensor.py"

# DTI arm: only when a DTI scan is wired in the config (run_cohort stages it when ParkMRE_DTI is present).
# Build the dwi2cond tensor in m2m and score its divergence from <D> (08); the DTI sim joins ISO+MD-dMRI
# below so the single extract call picks up all three.
MODELS_RUN=(ISO MD-dMRI)
SIM_DIRS=("$WORK_DIR/sim_${MONTAGE}_ISO" "$WORK_DIR/sim_${MONTAGE}_MD_dMRI")
if [ -n "${DTI_DWI:-}" ]; then
    stage "DTI tensor (standard dwi2cond)"
    bash "$REPO/pipeline/01_dwi2cond.sh"
    stage "tensor divergence DTI vs <D>"
    "$SP" "$REPO/analysis/08_tensor_divergence.py"
    MODELS_RUN+=(DTI)
    SIM_DIRS+=("$WORK_DIR/sim_${MONTAGE}_DTI")
fi

stage "simulations ${MODELS_RUN[*]}"
rm -rf "${SIM_DIRS[@]}"
"$SP" "$REPO/pipeline/04_run_simulations.py" --montage "$MONTAGE" --model "${MODELS_RUN[@]}"

stage "extract ROI E-field"
"$SP" "$REPO/analysis/04_extract_roi_efield.py" --montage "$MONTAGE"

stage "MRE to T1 + cross-modal compare"
bash "$REPO/pipeline/05_register_mre_to_T1.sh"
MONTAGE="$MONTAGE" "$SP" "$REPO/analysis/05_mre_efield_comparison.py"

stage "emit metrics (per-subject QC snapshot)"
"$SP" "$REPO/analysis/qc_harness.py" --montage "$MONTAGE" --emit-metrics "$RESULTS/metrics.json"

echo; echo "Done. metrics -> $RESULTS/metrics.json  provenance -> $RESULTS/provenance.txt"
