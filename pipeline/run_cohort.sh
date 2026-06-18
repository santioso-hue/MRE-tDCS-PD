#!/bin/bash
# run_cohort.sh — scale-out over the cohort. Stages each subject from the mounted KTH share into the
# per-subject layout (symlinks; no copy), writes a per-subject config, and runs run_subject.sh
# (default charm, ISO+MD-dMRI; DTI gated). Per-subject failures are logged and skipped, not fatal.
#
# Usage: COHORT_SHARE=<cohort_data dir> bash pipeline/run_cohort.sh [SUBJECT ...]
#        (no SUBJECT args = every MUDI_synb0/<S> on the share)
# Env:   COHORT_SHARE (required), MONTAGE (default M1), FROM (default charm), MDDMRI_DIR (default $HOME/md-dmri-master)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
SHARE="${COHORT_SHARE:?set COHORT_SHARE to the mounted cohort_data dir}"
MONTAGE="${MONTAGE:-M1}"
FROM="${FROM:-charm}"
MDDMRI_DIR="${MDDMRI_DIR:-$HOME/md-dmri-master}"
COHORT_LOCAL="$REPO/data/cohort_local"
[ -d "$SHARE/MUDI_synb0" ] || { echo "ERROR: $SHARE/MUDI_synb0 not found (mount the cohort share)"; exit 1; }

if [ $# -gt 0 ]; then SUBJECTS=("$@"); else SUBJECTS=($(ls -1 "$SHARE/MUDI_synb0")); fi
echo "Cohort scale-out: ${#SUBJECTS[@]} subjects; share=$SHARE; montage=$MONTAGE; from=$FROM"

ok=0; fail=0; FAILED=()
for S in "${SUBJECTS[@]}"; do
    echo; echo "================ $S ================"
    D="$COHORT_LOCAL/$S"
    mkdir -p "$D/anat" "$D/dmri" "$D/mre" "$D/fit" "$D/work"
    ln -sf "$SHARE/T1T2QSMandSWI/$S/T1.nii.gz"       "$D/anat/T1.nii.gz"
    ln -sf "$SHARE/T1T2QSMandSWI/$S/t2_to_t1.nii.gz" "$D/anat/t2_to_t1.nii.gz"
    for f in linear_corrected spherical_corrected; do
        for e in nii.gz bval bvec; do ln -sf "$SHARE/MUDI_synb0/$S/$f.$e" "$D/dmri/$f.$e"; done
    done
    ln -sfn "$SHARE/ReconAlls/$S" "$D/recon"
    ln -sf "$SHARE/MRE_ToT1_202402/$S/MRE_stiffness_ToT1_202402.nii.gz"         "$D/mre/MRE_stiffness_ToT1.nii.gz"
    ln -sf "$SHARE/MRE_ToT1_202402/$S/MRE_alphaparam_masked_ToT1_202402.nii.gz" "$D/mre/MRE_alpha_ToT1.nii.gz"
    cat > "$D/config.sh" <<CFG
SUBJECT="$S"
DATA_DIR="$D"
WORK_DIR="\${DATA_DIR}/work"
NII_DIR="\${DATA_DIR}/dmri"
FIT_DIR="\${DATA_DIR}/fit"
M2M_DIR="\${WORK_DIR}/m2m_\${SUBJECT}"
REG_DIR="\${DATA_DIR}/registration"
QTI_MFS="\${FIT_DIR}/qti_cov/cov_mfs.mat"
QTI_DPS="\${FIT_DIR}/qti_cov/cov_dps.mat"
DWI_NII="\${DATA_DIR}/dmri/linear_corrected.nii.gz"
DWI_BVAL="\${DATA_DIR}/dmri/linear_corrected.bval"
DWI_BVEC="\${DATA_DIR}/dmri/linear_corrected.bvec"
STE_B0_NII="\${DATA_DIR}/dmri/spherical_corrected.nii.gz"
SIMNIBS_BIN="\${HOME}/Applications/SimNIBS-4.6/bin"
FSLDIR="\${HOME}/fsl"
MRE_STIFFNESS="\${DATA_DIR}/mre/MRE_stiffness_ToT1.nii.gz"
MRE_ALPHA="\${DATA_DIR}/mre/MRE_alpha_ToT1.nii.gz"
MRE_STORAGE=""; MRE_LOSS=""; MRE_CONFIDENCE=""
CHARM_T1="\${DATA_DIR}/anat/T1.nii.gz"
CHARM_T2="\${DATA_DIR}/anat/t2_to_t1.nii.gz"
MDDMRI_DIR="$MDDMRI_DIR"
CFG
    if PIPELINE_CONFIG="$D/config.sh" bash "$REPO/pipeline/run_subject.sh" --from "$FROM" --montage "$MONTAGE"; then
        ok=$((ok+1)); echo "OK $S"
    else
        fail=$((fail+1)); FAILED+=("$S"); echo "FAILED $S (continuing)"
    fi
done
echo; echo "Cohort done: $ok ok, $fail failed."
[ $fail -gt 0 ] && printf 'FAILED: %s\n' "${FAILED[*]:-}"
echo "Next: simnibs_python analysis/06_cohort_stats.py to aggregate per-ROI E-field across subjects."
