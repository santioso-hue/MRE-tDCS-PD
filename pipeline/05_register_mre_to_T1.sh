#!/bin/bash
# 05_register_mre_to_T1.sh — bring the MRE mechanical maps into charm/mesh T1 space for the post-hoc
# cross-modal comparison (mechanics vs MD-dMRI microstructure / E-field).
#
# Cohort MRE (stiffness + alpha) is delivered ALREADY in the FreeSurfer-conformed T1 (256^3, "_ToT1"),
# the same grid as recon-all's orig.mgz. So this resamples it to the charm/mesh T1 via the orig->charm
# rigid affine -- the identical transform analysis/build_rois.py uses to place the ROIs -- so the MRE
# maps, the ROIs, and the E-field all live in one space. (No raw-MRE->T1 registration is needed; the
# upstream pipeline already did MRE->T1.)
#
# Usage:   PIPELINE_CONFIG=<subject config.sh> bash pipeline/05_register_mre_to_T1.sh
# Output:  registration/mre_stiffness_T1.nii.gz, mre_alpha_T1.nii.gz   (charm T1 space)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${PIPELINE_CONFIG:-$SCRIPT_DIR/../config/config.sh}"
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG (set PIPELINE_CONFIG)"; exit 1; }
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
# shellcheck disable=SC1090
source "$CFG"
export FSLDIR
export PATH="$SIMNIBS_BIN:$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ

require_nonzero() {
    local f="$1"
    [ -f "$f" ] || { echo "ERROR: expected output missing: $f"; exit 1; }
    local nz; nz="$(fslstats "$f" -V | awk '{print $1}')"
    [ "${nz:-0}" -gt 0 ] 2>/dev/null || { echo "ERROR: $f has 0 non-zero voxels"; exit 1; }
}

T1_REF="$M2M_DIR/T1.nii.gz"
ORIG="$DATA_DIR/recon/mri/orig.mgz"           # recon-all orig = the FS-conformed grid the MRE maps live on
for f in "$T1_REF" "$ORIG" "$MRE_STIFFNESS" "$MRE_ALPHA"; do
    [ -f "$f" ] || { echo "ERROR: required input missing: $f"; exit 1; }
done
mkdir -p "$REG_DIR"; cd "$REG_DIR"

echo "=== FS-conformed T1 -> charm T1 affine (orig.mgz, 6-DOF MI; same transform build_rois uses) ==="
"$SIMNIBS_BIN/simnibs_python" -c "import nibabel as nib; nib.save(nib.load('$ORIG'), 'orig_fs.nii.gz')"
flirt -in orig_fs.nii.gz -ref "$T1_REF" -omat fs_to_charm.mat -dof 6 -cost mutualinfo \
      -searchrx -20 20 -searchry -20 20 -searchrz -20 20 -interp trilinear   # match build_rois' orig->charm
[ -s fs_to_charm.mat ] || { echo "ERROR: orig->charm flirt failed"; exit 1; }

echo "=== resample MRE stiffness + alpha (his FS-T1 grid) into charm T1 ==="
flirt -in "$MRE_STIFFNESS" -ref "$T1_REF" -applyxfm -init fs_to_charm.mat -interp trilinear -out mre_stiffness_T1.nii.gz
flirt -in "$MRE_ALPHA"     -ref "$T1_REF" -applyxfm -init fs_to_charm.mat -interp trilinear -out mre_alpha_T1.nii.gz
# alpha (springpot exponent) is physically in (0,1]; the delivered map carries Helmholtz-inversion
# artifacts outside that range (~ -1.5..1.5). Mask them to 0 (= dropped downstream like masked background),
# matching the subject-level "alphapositive" QC, so only valid alpha enters the medians/correlations.
"$SIMNIBS_BIN/simnibs_python" - <<'PY'
import nibabel as nib, numpy as np
im = nib.load("mre_alpha_T1.nii.gz"); a = np.asarray(im.dataobj, np.float64)
bad = ~np.isfinite(a) | (a <= 0) | (a > 1)
a[bad] = 0.0
nib.save(nib.Nifti1Image(a.astype(np.float32), im.affine, im.header), "mre_alpha_T1.nii.gz")
print(f"  alpha validity mask: zeroed {int(bad.sum()):,} out-of-(0,1] voxels")
PY
require_nonzero mre_stiffness_T1.nii.gz
require_nonzero mre_alpha_T1.nii.gz
rm -f orig_fs.nii.gz

echo "Done -> registration/mre_{stiffness,alpha}_T1.nii.gz. Next: simnibs_python analysis/05_mre_efield_comparison.py"
