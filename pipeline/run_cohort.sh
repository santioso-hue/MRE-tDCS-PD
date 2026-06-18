#!/bin/bash
# run_cohort.sh - scale-out over the cohort. Stages each subject from the mounted KTH share into the
# per-subject layout (symlinks; no copy), writes a per-subject config, and runs run_subject.sh
# (default charm, ISO+MD-dMRI, plus the DTI arm when a matching ParkMRE_DTI scan is found on the share).
# Per-subject failures are logged and skipped, not fatal.
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

ok=0; fail=0; skipped=0; dti_matched=0; dti_skipped=0; FAILED=()
for S in "${SUBJECTS[@]}"; do
    echo; echo "================ $S ================"
    D="$COHORT_LOCAL/$S"
    # Resumable: a finished subject (metrics.json present) is skipped so a re-run after a crash only
    # recomputes the failed ids. Set REDO=1 to force recompute.
    if [ -f "$REPO/analysis/results/$S/metrics.json" ] && [ "${REDO:-0}" != "1" ]; then
        echo "SKIP $S (results/$S/metrics.json exists; set REDO=1 to recompute)"; skipped=$((skipped+1)); continue
    fi
    mkdir -p "$D/anat" "$D/dmri" "$D/dti" "$D/mre" "$D/fit" "$D/work"
    ln -sf "$SHARE/T1T2QSMandSWI/$S/T1.nii.gz"       "$D/anat/T1.nii.gz"
    ln -sf "$SHARE/T1T2QSMandSWI/$S/t2_to_t1.nii.gz" "$D/anat/t2_to_t1.nii.gz"
    for f in linear_corrected spherical_corrected; do
        for e in nii.gz bval bvec; do ln -sf "$SHARE/MUDI_synb0/$S/$f.$e" "$D/dmri/$f.$e"; done
    done
    ln -sfn "$SHARE/ReconAlls/$S" "$D/recon"
    ln -sf "$SHARE/MRE_ToT1_202402/$S/MRE_stiffness_ToT1_202402.nii.gz"         "$D/mre/MRE_stiffness_ToT1.nii.gz"
    ln -sf "$SHARE/MRE_ToT1_202402/$S/MRE_alphaparam_masked_ToT1_202402.nii.gz" "$D/mre/MRE_alpha_ToT1.nii.gz"

    # DTI arm source (independent single-shell scan under ParkMREDTI_Backup). The DTI folder names drift
    # from the MUDI subject IDs (a _DTI suffix, dropped underscores, or case), so match on a normalised
    # alnum-lowercase key. Missing/incomplete -> leave DTI_* unset and run_subject runs ISO+MD-dMRI only.
    dti_dwi_src=""; dti_bvec_src=""; dti_mask_src=""
    DTI_ROOT="$SHARE/DTI/ParkMREDTI_Backup"
    if [ -d "$DTI_ROOT" ]; then
        key=$(printf '%s' "$S" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')
        dti_dir=""; nmatch=0
        for cand in "$DTI_ROOT"/*/; do
            [ -d "$cand" ] || continue
            nm=$(basename "$cand"); nkey=$(printf '%s' "${nm%_DTI}" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]')
            if [ "$nkey" = "$key" ]; then nmatch=$((nmatch+1)); [ -z "$dti_dir" ] && dti_dir="$cand"; fi
        done
        if [ "$nmatch" -gt 1 ]; then
            echo "  WARN: $S matches $nmatch DTI dirs by normalized key -> DTI arm skipped (disambiguate names)"
            dti_dir=""
        fi
        if [ -n "$dti_dir" ]; then
            [ -f "$dti_dir/eddy_corrected.nii.gz" ] && dti_dwi_src="$dti_dir/eddy_corrected.nii.gz"
            [ -z "$dti_dwi_src" ] && dti_dwi_src=$(ls -1 "$dti_dir"/*eddy*corrected*.nii.gz 2>/dev/null | head -1)
            [ -f "$dti_dir/eddy_corrected.eddy_rotated_bvecs" ] && dti_bvec_src="$dti_dir/eddy_corrected.eddy_rotated_bvecs"
            [ -z "$dti_bvec_src" ] && dti_bvec_src=$(ls -1 "$dti_dir"/*rotated_bvec* 2>/dev/null | head -1)
            [ -z "$dti_bvec_src" ] && dti_bvec_src=$(ls -1 "$dti_dir"/*.bvec 2>/dev/null | head -1)
            [ -f "$dti_dir/my_hifi_b0_brain_mask.nii.gz" ] && dti_mask_src="$dti_dir/my_hifi_b0_brain_mask.nii.gz"
            [ -z "$dti_mask_src" ] && dti_mask_src=$(ls -1 "$dti_dir"/*brain_mask*.nii.gz 2>/dev/null | head -1)
        fi
    fi
    if [ -n "$dti_dwi_src" ] && [ -n "$dti_bvec_src" ]; then
        ln -sf "$dti_dwi_src" "$D/dti/dwi.nii.gz"; ln -sf "$dti_bvec_src" "$D/dti/dwi.bvec"
        [ -n "$dti_mask_src" ] && ln -sf "$dti_mask_src" "$D/dti/mask.nii.gz"
        echo "  DTI: $S <- $(basename "${dti_dir%/}")"; dti_matched=$((dti_matched+1))
    else
        dti_dwi_src=""   # incomplete -> DTI arm skipped for this subject (ISO+MD-dMRI still run)
        echo "  note: no DTI source for $S under ParkMREDTI_Backup; DTI arm skipped"; dti_skipped=$((dti_skipped+1))
    fi

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
    if [ -n "$dti_dwi_src" ]; then
        cat >> "$D/config.sh" <<'DTICFG'
DTI_DWI="${DATA_DIR}/dti/dwi.nii.gz"
DTI_BVEC="${DATA_DIR}/dti/dwi.bvec"
DTI_BVALUE=1500
DTICFG
        [ -n "$dti_mask_src" ] && echo 'DTI_MASK="${DATA_DIR}/dti/mask.nii.gz"' >> "$D/config.sh"
    fi
    if PIPELINE_CONFIG="$D/config.sh" bash "$REPO/pipeline/run_subject.sh" --from "$FROM" --montage "$MONTAGE"; then
        ok=$((ok+1)); echo "OK $S"
    else
        fail=$((fail+1)); FAILED+=("$S"); echo "FAILED $S (continuing)"
    fi
done
echo; echo "Cohort done: $ok ok, $fail failed, $skipped skipped (already had metrics.json)."
echo "DTI arm: $dti_matched matched a ParkMRE_DTI scan, $dti_skipped without one (ran ISO+MD-dMRI only)."
[ $fail -gt 0 ] && printf 'FAILED: %s\n' "${FAILED[*]:-}"
echo "Next: simnibs_python analysis/06_cohort_stats.py to aggregate per-ROI E-field across subjects."
