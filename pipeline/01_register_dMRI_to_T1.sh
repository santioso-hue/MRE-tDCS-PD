#!/bin/bash
# Step 1: Register MD-dMRI outputs (C_mu, MD, mask) to T1 space using FSL FLIRT
#
# MUST run AFTER charm — uses m2m_${SUBJECT}/T1.nii.gz as the registration target
# so everything lives in the same coordinate space as the FEM mesh.
#
# Usage: bash 01_register_dMRI_to_T1.sh
# Runtime: ~2 min

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config/config.sh"
export FSLDIR
export PATH="$SIMNIBS_BIN:$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ

WDIR="$WORK_DIR"
NDIR="$NII_DIR"
FDIR="$FIT_DIR"

# T1 reference: use the bias-corrected T1 from CHARM output for consistency
# with the FEM mesh coordinate space
T1_REF="$M2M_DIR/T1.nii.gz"
if [ ! -f "$T1_REF" ]; then
    echo "ERROR: $T1_REF not found. Run CHARM first."
    exit 1
fi

mkdir -p "$WDIR/registration"
cd "$WDIR/registration"

echo "=== Step 1: Extract b=0 from dMRI_Spherical (registration source) ==="
# Volume 0 of dMRI_Spherical is b=0 — same space as the QTI fit/ outputs
fslroi "$STE_B0_NII" \
       b0_spherical.nii.gz 0 1
echo "  b0 extracted: $(fslinfo b0_spherical.nii.gz | grep -E 'dim[1-4]|pixdim[1-4]' | head -8)"

echo ""
echo "=== Step 2: FLIRT rigid registration — b0 -> T1 ==="
# 6-DOF rigid (no scaling/shearing between same-subject modalities)
# Mutual information handles T1/EPI contrast mismatch well
flirt -in  b0_spherical.nii.gz \
      -ref "$T1_REF" \
      -out b0_spherical_T1.nii.gz \
      -omat dMRI_to_T1.mat \
      -dof 6 \
      -cost mutualinfo \
      -searchrx -20 20 \
      -searchry -20 20 \
      -searchrz -20 20 \
      -interp trilinear
echo "  Transform saved: dMRI_to_T1.mat"
echo "  Registered b0:   b0_spherical_T1.nii.gz"
echo "  VISUALLY CHECK this registration in FSLeyes before proceeding!"

echo ""
echo "=== Step 3a: Extract μFA (ufa), MD, signaniso from dps.mat (QA/reporting) ==="
# Use dps.mat fields directly (ufa, MD already zero-masked → FLIRT-safe).
"$SIMNIBS_BIN/simnibs_python" "$SCRIPT_DIR/01c_save_dps_niftis.py"

echo ""
echo "=== Step 3b: Extract full triaxial mean tensor ⟨D⟩ + eigenvalues from dps.mat ==="
# dps['mdxx'..'mdyz'] hold ⟨D⟩ in SI units (×1e9 → µm²/ms). 01d writes the 6-comp
# tensor (for vecreg orientation) and λ1≥λ2≥λ3 scalar maps (for magnitude).
"$SIMNIBS_BIN/simnibs_python" "$SCRIPT_DIR/01d_save_triaxial_tensor.py"

echo ""
echo "=== Step 4: Transform scalar maps to T1 (FLIRT trilinear / nearestneighbour) ==="
# Eigenvalues λ1,λ2,λ3 registered as SCALARS — preserves anisotropy magnitude
# (whole-tensor interpolation would dilute it). vecreg below carries orientation.
for L in lam1 lam2 lam3; do
  flirt -in "${L}_dMRI.nii.gz" -ref "$T1_REF" -out "${L}_T1.nii.gz" \
        -applyxfm -init dMRI_to_T1.mat -interp trilinear
done

flirt -in C_mu_dps_dMRI.nii.gz -ref "$T1_REF" -out C_mu_T1.nii.gz \
      -applyxfm -init dMRI_to_T1.mat -interp trilinear
flirt -in MD_dps_dMRI.nii.gz   -ref "$T1_REF" -out MD_T1.nii.gz \
      -applyxfm -init dMRI_to_T1.mat -interp trilinear
flirt -in dMRI_mask.nii.gz     -ref "$T1_REF" -out dMRI_mask_T1.nii.gz \
      -applyxfm -init dMRI_to_T1.mat -interp nearestneighbour
# signaniso is discrete {-1,0,+1} — nearestneighbour preserves the labels.
flirt -in signaniso_dMRI.nii.gz -ref "$T1_REF" -out signaniso_T1.nii.gz \
      -applyxfm -init dMRI_to_T1.mat -interp nearestneighbour

echo ""
echo "=== Step 5: Save principal eigenvectors from dps.mat as NIfTI ==="
"$SIMNIBS_BIN/simnibs_python" "$SCRIPT_DIR/01b_save_v1_nifti.py"

echo ""
echo "=== Step 6: Reorient direction data to T1 with vecreg ==="
# vecreg applies the proper covariant (rotation) transform — not plain resampling.
vecreg -i v1_dMRI.nii.gz -o v1_T1.nii.gz -r "$T1_REF" -t dMRI_to_T1.mat
echo "  Registered principal eigenvector: v1_T1.nii.gz"
# Full triaxial tensor: vecreg detects the 6-component field and reorients it.
# Used ONLY for the in-plane (v2,v3) orientation frame in 02 (eigenvalues come
# from the scalar maps above; principal axis is anchored to the validated v1_T1).
vecreg -i tensor_triaxial_dMRI.nii.gz -o tensor_triaxial_T1.nii.gz -r "$T1_REF" -t dMRI_to_T1.mat
echo "  Registered triaxial tensor frame: tensor_triaxial_T1.nii.gz"

echo ""
echo "=== Registration complete ==="
echo "Outputs in $WDIR/registration/ (triaxial σ∝⟨D⟩ model):"
echo "  dMRI_to_T1.mat          — FLIRT transform (6-DOF rigid)"
echo "  lam1/lam2/lam3_T1.nii.gz — mean-tensor eigenvalues λ1≥λ2≥λ3 in T1 (µm²/ms, scalar)"
echo "  tensor_triaxial_T1.nii.gz — reoriented ⟨D⟩ frame (in-plane v2,v3 source)"
echo "  v1_T1.nii.gz            — validated principal eigenvector (vecreg)"
echo "  C_mu_T1.nii.gz, MD_T1.nii.gz, signaniso_T1.nii.gz, dMRI_mask_T1.nii.gz — QA/reporting maps"
echo ""
echo "Next: run 02_build_conductivity_tensor.py (builds tensor_MD_dMRI.nii.gz)"
