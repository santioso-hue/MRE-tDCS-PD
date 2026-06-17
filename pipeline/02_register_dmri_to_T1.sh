#!/bin/bash
# 02_register_dmri_to_T1.sh — bring the QTI mean-tensor maps from dMRI space to T1 (charm/mesh) space.
#
# Registration is an S0-driven 12-DOF affine: FLIRT the dMRI model S0 onto the charm T2 (same T2-like
# contrast, so the fit drives a robust whole-brain affine), then apply that one transform to every map.
# This matches Christoffer's S0->T2 pairing and stays affine because the cohort dMRI is already
# Synb0+topup distortion-corrected; the registration bake-off (docs/registration_bakeoff/) showed a
# nonlinear warp over-warps it (~5 mm) without improving alignment, so an affine is the faithful choice.
#
# Runs after 00_charm and the QTI fit. prepare_dmri_tensor.py reconstructs <D>, eigenvalues, v1, S0 and
# the QA maps in dMRI space; here the S0 drives the affine and each map is carried to T1 by type:
# scalars trilinear, the brain mask nearest-neighbour, and v1 + the triaxial tensor reoriented with
# vecreg -t (the covariant reorientation for an affine). 03 then assembles the conductivity tensor.
#
# Usage:   PIPELINE_CONFIG=<subject config.sh> bash pipeline/02_register_dmri_to_T1.sh
# Runtime: ~1 min.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${PIPELINE_CONFIG:-$SCRIPT_DIR/../config/config.sh}"
[ -f "$CFG" ] || { echo "ERROR: config not found: $CFG (set PIPELINE_CONFIG)"; exit 1; }
# Absolutize so the Python child (prepare_dmri_tensor, via _config.py) resolves it after we cd into REG_DIR.
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"
export PIPELINE_CONFIG="$CFG"
# shellcheck disable=SC1090
source "$CFG"
export FSLDIR
export PATH="$SIMNIBS_BIN:$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ

# A registration output that "succeeds" but is all-zero would otherwise flow silently into stage 03.
require_nonzero() {
    local f="$1"
    [ -f "$f" ] || { echo "ERROR: expected output missing: $f"; exit 1; }
    local nz; nz="$(fslstats "$f" -V | awk '{print $1}')"
    [ "${nz:-0}" -gt 0 ] 2>/dev/null || { echo "ERROR: $f has 0 non-zero voxels — registration failed"; exit 1; }
    # fslstats -V counts NaN/Inf as nonzero, so also reject a non-finite max (an all-garbage map "passes" otherwise).
    local mx; mx="$(fslstats "$f" -R | awk '{print $2}')"
    case "$mx" in *[iI]nf*|*[nN]a[nN]*) echo "ERROR: $f has non-finite values (max=$mx) — registration failed"; exit 1;; esac
}

T1_REF="$M2M_DIR/T1.nii.gz"
T2_REG="$M2M_DIR/T2_reg.nii.gz"
for f in "$T1_REF" "$T2_REG" "$M2M_DIR/final_tissues.nii.gz"; do
    [ -f "$f" ] || { echo "ERROR: $f not found. Run 00_charm.sh first."; exit 1; }
done
mkdir -p "$REG_DIR"
cd "$REG_DIR"

echo "=== Step 1: reconstruct <D> + eigenvalues + v1 + S0 + QA maps from the QTI fit (dMRI space) ==="
# prepare_dmri_tensor needs a 3D image on the fit grid for the output affine/header. run_qti_cov_cohort
# writes a stable dmri_grid_ref.nii.gz for exactly this; fall back to the fit auto-mask for older fits.
QDIR="$(dirname "$QTI_MFS")"
if [ -z "${DMRI_REF:-}" ]; then
    if   [ -f "$QDIR/dmri_grid_ref.nii.gz" ];     then DMRI_REF="$QDIR/dmri_grid_ref.nii.gz"
    else DMRI_REF="$QDIR/LTE_STE_mc_s_mask.nii.gz"; fi
fi
export DMRI_REF
[ -f "$DMRI_REF" ] || { echo "ERROR: dMRI grid reference not found: $DMRI_REF"; exit 1; }
"$SIMNIBS_BIN/simnibs_python" "$SCRIPT_DIR/prepare_dmri_tensor.py"

echo ""
echo "=== Step 2: brain-extract the charm T2 (the S0 contrast match, registration target) ==="
T2_BRAIN="T2_brain.nii.gz"
"$SIMNIBS_BIN/simnibs_python" - "$M2M_DIR" "$T2_BRAIN" <<'PY'
import sys, os, numpy as np, nibabel as nib
m2m, out = sys.argv[1], sys.argv[2]
t2 = nib.load(os.path.join(m2m, "T2_reg.nii.gz"))
seg = nib.load(os.path.join(m2m, "final_tissues.nii.gz")).get_fdata()
seg = seg[..., 0] if seg.ndim == 4 else seg
brain = np.isin(seg, [1, 2, 3]).astype(np.float32)   # WM, GM, CSF
nib.save(nib.Nifti1Image((t2.get_fdata() * brain).astype(np.float32), t2.affine, t2.header), out)
PY
require_nonzero "$T2_BRAIN"

echo ""
echo "=== Step 3: S0-driven 12-DOF affine — dMRI S0 -> charm T2 (Mattes MI) ==="
flirt -in s0_dMRI.nii.gz -ref "$T2_BRAIN" -omat dMRI_to_T1_aff.mat -dof 12 -cost mutualinfo \
      -searchrx -25 25 -searchry -25 25 -searchrz -25 25
[ -s dMRI_to_T1_aff.mat ] || { echo "ERROR: S0 affine failed"; exit 1; }

echo ""
echo "=== Step 4: carry every map to T1 with that one affine ==="
# Scalars (trilinear): eigenvalues for the conductivity tensor, plus QA/QC maps. Eigenvalues are warped
# as scalars so the anisotropy magnitude is preserved (whole-tensor interpolation would dilute it); the
# orientation is carried separately by vecreg below.
for s in FA lam1 lam2 lam3 MD uFA s0; do
    flirt -in "${s}_dMRI.nii.gz" -ref "$T1_REF" -applyxfm -init dMRI_to_T1_aff.mat \
          -interp trilinear -out "${s}_T1.nii.gz"
done
flirt -in dMRI_mask.nii.gz -ref "$T1_REF" -applyxfm -init dMRI_to_T1_aff.mat \
      -interp nearestneighbour -out dMRI_mask_T1.nii.gz
# Direction data: vecreg -t applies the affine's rotation to the vectors/tensor (covariant reorientation).
vecreg -i v1_dMRI.nii.gz             -o v1_T1.nii.gz             -r "$T1_REF" -t dMRI_to_T1_aff.mat
vecreg -i tensor_triaxial_dMRI.nii.gz -o tensor_triaxial_T1.nii.gz -r "$T1_REF" -t dMRI_to_T1_aff.mat

for out in FA_T1 lam1_T1 lam2_T1 lam3_T1 MD_T1 uFA_T1 s0_T1 dMRI_mask_T1 v1_T1 tensor_triaxial_T1; do
    require_nonzero "${out}.nii.gz"
done

echo ""
echo "=== Registration complete (outputs in $REG_DIR) ==="
echo "  lam1/lam2/lam3_T1.nii.gz   mean-tensor eigenvalues (um2/ms, scalar)"
echo "  tensor_triaxial_T1.nii.gz  reoriented <D> frame (in-plane v2,v3 source)"
echo "  v1_T1.nii.gz               principal eigenvector (vecreg)"
echo "  MD_T1, uFA_T1.nii.gz       QA / post-hoc MRE-comparison maps"
echo "  s0_T1, FA_T1.nii.gz        registration QC (VISUALLY CHECK s0_T1 / FA_T1 over T1 before trusting)"
echo ""
echo "Next: simnibs_python pipeline/03_build_conductivity_tensor.py"
