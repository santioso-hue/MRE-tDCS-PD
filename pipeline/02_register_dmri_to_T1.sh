#!/bin/bash
# 02_register_dmri_to_T1.sh — Register the QTI mean-tensor maps to T1 (charm/mesh) space.
#
# Runs AFTER 00_charm (uses m2m_${SUBJECT}/T1.nii.gz as the target so everything lives in the FEM
# mesh coordinate space). prepare_dmri_tensor.py reconstructs <D> + eigenvalues + v1 + QA maps in
# dMRI space; this script brings them to T1: eigenvalues l1/l2/l3 as SCALARS (trilinear, preserves
# anisotropy magnitude) and the orientation by vecreg (proper tensor reorientation). The conductivity
# tensor is then assembled in 03_build_conductivity_tensor.py.
#
# Usage:   bash pipeline/02_register_dmri_to_T1.sh
# Runtime: ~2 min

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"
export FSLDIR
export PATH="$SIMNIBS_BIN:$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ

# Post-condition guard: a registration output that "succeeds" but is all-zero would otherwise flow
# silently into stage 03.
require_nonzero() {
    local f="$1"
    [ -f "$f" ] || { echo "ERROR: expected registration output missing: $f"; exit 1; }
    local nz; nz="$(fslstats "$f" -V | awk '{print $1}')"
    [ "${nz:-0}" -gt 0 ] 2>/dev/null || { echo "ERROR: $f has 0 non-zero voxels — registration failed"; exit 1; }
}

T1_REF="$M2M_DIR/T1.nii.gz"
[ -f "$T1_REF" ] || { echo "ERROR: $T1_REF not found. Run 00_charm.sh first."; exit 1; }
mkdir -p "$WORK_DIR/registration"
cd "$WORK_DIR/registration"

echo "=== Step 1: reconstruct <D> + eigenvalues + v1 + QA maps from dps.mat (dMRI space) ==="
"$SIMNIBS_BIN/simnibs_python" "$SCRIPT_DIR/prepare_dmri_tensor.py"

echo ""
echo "=== Step 2: extract b=0 from the spherical series (registration source) ==="
fslroi "$STE_B0_NII" b0_spherical.nii.gz 0 1

echo ""
echo "=== Step 3: FLIRT rigid b0 -> T1 (6-DOF, mutual information) ==="
flirt -in b0_spherical.nii.gz -ref "$T1_REF" -out b0_spherical_T1.nii.gz -omat dMRI_to_T1.mat \
      -dof 6 -cost mutualinfo -searchrx -20 20 -searchry -20 20 -searchrz -20 20 -interp trilinear
[ -s dMRI_to_T1.mat ] || { echo "ERROR: FLIRT did not produce dMRI_to_T1.mat"; exit 1; }
require_nonzero b0_spherical_T1.nii.gz
echo "  Transform: dMRI_to_T1.mat. VISUALLY CHECK b0_spherical_T1.nii.gz over T1 before trusting it."

echo ""
echo "=== Step 4: scalar maps -> T1 (eigenvalues trilinear; mask nearestneighbour) ==="
# Eigenvalues registered as SCALARS preserves anisotropy magnitude (whole-tensor interpolation
# would dilute it); vecreg below carries the orientation.
for L in lam1 lam2 lam3 C_mu_dps MD_dps; do
  flirt -in "${L}_dMRI.nii.gz" -ref "$T1_REF" -out "${L%_dps}_T1.nii.gz" \
        -applyxfm -init dMRI_to_T1.mat -interp trilinear
done
flirt -in dMRI_mask.nii.gz -ref "$T1_REF" -out dMRI_mask_T1.nii.gz \
      -applyxfm -init dMRI_to_T1.mat -interp nearestneighbour
for out in lam1_T1 lam2_T1 lam3_T1 C_mu_T1 MD_T1 dMRI_mask_T1; do require_nonzero "${out}.nii.gz"; done

echo ""
echo "=== Step 5: reorient direction data to T1 with vecreg (covariant rotation) ==="
vecreg -i v1_dMRI.nii.gz -o v1_T1.nii.gz -r "$T1_REF" -t dMRI_to_T1.mat
vecreg -i tensor_triaxial_dMRI.nii.gz -o tensor_triaxial_T1.nii.gz -r "$T1_REF" -t dMRI_to_T1.mat
require_nonzero v1_T1.nii.gz
require_nonzero tensor_triaxial_T1.nii.gz

echo ""
echo "=== Registration complete (outputs in $WORK_DIR/registration/) ==="
echo "  lam1/lam2/lam3_T1.nii.gz   mean-tensor eigenvalues (um2/ms, scalar)"
echo "  tensor_triaxial_T1.nii.gz  reoriented <D> frame (in-plane v2,v3 source)"
echo "  v1_T1.nii.gz               validated principal eigenvector (vecreg)"
echo "  C_mu_T1.nii.gz, MD_T1.nii.gz, dMRI_mask_T1.nii.gz   QA / post-hoc MRE-comparison maps"
echo ""
echo "Next: simnibs_python pipeline/03_build_conductivity_tensor.py"
