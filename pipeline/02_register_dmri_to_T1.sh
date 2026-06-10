#!/bin/bash
# 02_register_dmri_to_T1.sh — Register the QTI mean-tensor maps to T1 (charm/mesh) space.
#
# Runs AFTER 00_charm (uses m2m_${SUBJECT}/T1.nii.gz as the target so everything lives in the FEM
# mesh coordinate space). prepare_dmri_tensor.py reconstructs <D> + eigenvalues + v1 + QA maps in
# dMRI space; this script brings them to T1 by NONLINEAR fnirt of FA(<D>)->T1 — identical to dwi2cond's
# DTI path (regmthd=nonl), so both anisotropic arms register the same way and the only DTI<->MD-dMRI
# difference is the tensor. Eigenvalues l1/l2/l3 are warped as SCALARS (trilinear, preserves anisotropy
# magnitude) and the orientation is carried by vecreg (warp-based tensor reorientation). The
# conductivity tensor is then assembled in 03_build_conductivity_tensor.py.
#
# Usage:   bash pipeline/02_register_dmri_to_T1.sh
# Runtime: ~7 min (fnirt)

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

echo "=== Step 1: reconstruct <D> + eigenvalues + v1 + QA maps from the QTI covariance fit (dMRI space) ==="
"$SIMNIBS_BIN/simnibs_python" "$SCRIPT_DIR/prepare_dmri_tensor.py"

echo ""
echo "=== Step 2: skull-stripped T1 (fnirt reference) + b0 (QC) ==="
fslroi "$STE_B0_NII" b0_spherical.nii.gz 0 1
"$SIMNIBS_BIN/simnibs_python" - "$M2M_DIR" <<'PY'
import sys, os, numpy as np, nibabel as nib
m2m = sys.argv[1]; t1 = nib.load(os.path.join(m2m, "T1.nii.gz"))
seg = nib.load(os.path.join(m2m, "final_tissues.nii.gz")).get_fdata(); seg = seg[..., 0] if seg.ndim == 4 else seg
brain = np.isin(seg, [1, 2, 3]).astype(np.float32)
nib.save(nib.Nifti1Image((t1.get_fdata() * brain).astype(np.float32), t1.affine, t1.header),
         os.path.join(m2m, "T1_brain.nii.gz"))
PY
T1_BRAIN="$M2M_DIR/T1_brain.nii.gz"
[ -f "$T1_BRAIN" ] || { echo "ERROR: failed to build T1_brain.nii.gz"; exit 1; }

echo ""
echo "=== Step 3: nonlinear FA(<D>) -> T1 (flirt affine init + fnirt) — IDENTICAL to dwi2cond's DTI path ==="
# The DTI arm (dwi2cond --all, regmthd=nonl) registers FA->T1 with fnirt; we match it so the only
# DTI<->MD-dMRI difference is the tensor, not the registration. fnirt also corrects the EPI distortion
# that a rigid map leaves (verified: QTI v1 vs dwi2cond DTI improves 20.2 deg [rigid] -> 8.5 deg [fnirt]).
flirt -in FA_dMRI.nii.gz -ref "$T1_BRAIN" -omat dMRI_to_T1_aff.mat -dof 6 -cost mutualinfo \
      -searchrx -20 20 -searchry -20 20 -searchrz -20 20
[ -s dMRI_to_T1_aff.mat ] || { echo "ERROR: FLIRT affine init failed"; exit 1; }
fnirt --in=FA_dMRI.nii.gz --ref="$T1_BRAIN" --aff=dMRI_to_T1_aff.mat --cout=dMRI_to_T1_warp --subsamp=8,4,2,2
[ -f dMRI_to_T1_warp.nii.gz ] || { echo "ERROR: fnirt did not produce dMRI_to_T1_warp"; exit 1; }
applywarp -i b0_spherical.nii.gz -r "$T1_REF" -w dMRI_to_T1_warp -o b0_spherical_T1.nii.gz --interp=trilinear
echo "  Warp: dMRI_to_T1_warp.nii.gz. VISUALLY CHECK b0_spherical_T1.nii.gz / FA over T1 before trusting it."

echo ""
echo "=== Step 4: scalar maps -> T1 via the warp (eigenvalues trilinear; mask nearestneighbour) ==="
# Eigenvalues registered as SCALARS preserves anisotropy magnitude (whole-tensor interpolation
# would dilute it); vecreg below carries the orientation.
for L in lam1 lam2 lam3 C_mu_dps MD_dps; do
  applywarp -i "${L}_dMRI.nii.gz" -r "$T1_REF" -w dMRI_to_T1_warp -o "${L%_dps}_T1.nii.gz" --interp=trilinear
done
applywarp -i dMRI_mask.nii.gz -r "$T1_REF" -w dMRI_to_T1_warp -o dMRI_mask_T1.nii.gz --interp=nn
for out in lam1_T1 lam2_T1 lam3_T1 C_mu_T1 MD_T1 dMRI_mask_T1; do require_nonzero "${out}.nii.gz"; done

echo ""
echo "=== Step 5: reorient direction data to T1 with vecreg (warp-based covariant rotation) ==="
vecreg -i v1_dMRI.nii.gz -o v1_T1.nii.gz -r "$T1_REF" -w dMRI_to_T1_warp
vecreg -i tensor_triaxial_dMRI.nii.gz -o tensor_triaxial_T1.nii.gz -r "$T1_REF" -w dMRI_to_T1_warp
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
