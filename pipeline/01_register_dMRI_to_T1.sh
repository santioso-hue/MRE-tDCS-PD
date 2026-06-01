#!/bin/bash
# Step 1: Register MD-dMRI outputs (C_mu, MD, mask) to T1 space using FSL FLIRT
#
# MUST run AFTER charm — uses m2m_FullPD5/T1.nii.gz as the registration target
# so everything lives in the same coordinate space as the FEM mesh.
#
# Usage: bash 01_register_dMRI_to_T1.sh
# Runtime: ~2 min

set -e

export FSLDIR=~/fsl
export PATH=$FSLDIR/bin:$PATH
export FSLOUTPUTTYPE=NIFTI_GZ

WDIR="/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation"
NDIR="/Users/santi/Downloads/FullPD5_forSantiago/FullPD5/NiiFiles"
FDIR="/Users/santi/Downloads/FullPD5_forSantiago/FullPD5/fit"

# T1 reference: use the bias-corrected T1 from CHARM output for consistency
# with the FEM mesh coordinate space
T1_REF="$WDIR/m2m_FullPD5/T1.nii.gz"
if [ ! -f "$T1_REF" ]; then
    echo "ERROR: $T1_REF not found. Run CHARM first."
    exit 1
fi

mkdir -p "$WDIR/registration"
cd "$WDIR/registration"

echo "=== Step 1: Extract b=0 from dMRI_Spherical (registration source) ==="
# Volume 0 of dMRI_Spherical is b=0 — same space as the QTI fit/ outputs
fslroi "$NDIR/FullPD5_WIP_dMRI_Spherical_medium_20230110134105_501.nii.gz" \
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
echo "=== Step 3: Extract clean μFA (ufa) and MD from dps.mat ==="
# IMPORTANT: dtd_covariance_C_mu.nii.gz and dtd_covariance_MD.nii.gz have NaN
# outside the QTI mask. FSL trilinear interpolation spreads NaN into the brain
# region. Instead, use dps.mat fields directly: ufa [0,1] and MD [μm²/ms] are
# already masked with zeros outside the brain mask — safe for FLIRT.
~/Applications/SimNIBS-4.6/bin/simnibs_python "$WDIR/scripts/01c_save_dps_niftis.py"
echo "  Saved: C_mu_dps_dMRI.nii.gz, MD_dps_dMRI.nii.gz, dMRI_mask.nii.gz"

echo ""
echo "=== Step 4: Apply transform to C_mu, MD, brain mask, and signaniso ==="
flirt -in  C_mu_dps_dMRI.nii.gz \
      -ref "$T1_REF" \
      -out C_mu_T1.nii.gz \
      -applyxfm -init dMRI_to_T1.mat \
      -interp trilinear

flirt -in  MD_dps_dMRI.nii.gz \
      -ref "$T1_REF" \
      -out MD_T1.nii.gz \
      -applyxfm -init dMRI_to_T1.mat \
      -interp trilinear

flirt -in  dMRI_mask.nii.gz \
      -ref "$T1_REF" \
      -out dMRI_mask_T1.nii.gz \
      -applyxfm -init dMRI_to_T1.mat \
      -interp nearestneighbour

# signaniso has discrete values {-1, 0, +1} — must use nearestneighbour to preserve them.
# Trilinear would create intermediate values (e.g. -0.5) that the tensor builder cannot interpret.
flirt -in  signaniso_dMRI.nii.gz \
      -ref "$T1_REF" \
      -out signaniso_T1.nii.gz \
      -applyxfm -init dMRI_to_T1.mat \
      -interp nearestneighbour

echo ""
echo "=== Step 5: Save principal eigenvectors from dps.mat as NIfTI ==="
~/Applications/SimNIBS-4.6/bin/simnibs_python "$WDIR/scripts/01b_save_v1_nifti.py"

echo ""
echo "=== Step 6: Register eigenvectors to T1 space with vecreg ==="
# vecreg properly rotates direction vectors (unlike flirt which just resamples)
vecreg -i v1_dMRI.nii.gz \
       -o v1_T1.nii.gz \
       -r "$T1_REF" \
       -t dMRI_to_T1.mat
echo "  Registered eigenvectors: v1_T1.nii.gz"

echo ""
echo "=== Registration complete ==="
echo "Outputs in $WDIR/registration/:"
echo "  dMRI_to_T1.mat        — FLIRT transform (6-DOF rigid)"
echo "  C_mu_dps_dMRI.nii.gz  — μFA from DPS model (dMRI space, NaN-free)"
echo "  MD_dps_dMRI.nii.gz    — MD from DPS model (dMRI space, μm²/ms, NaN-free)"
echo "  signaniso_dMRI.nii.gz — DPS shape indicator (dMRI space, {-1,0,+1})"
echo "  C_mu_T1.nii.gz        — μFA in T1 space"
echo "  MD_T1.nii.gz          — MD in T1 space (μm²/ms)"
echo "  dMRI_mask_T1.nii.gz   — Brain mask in T1 space"
echo "  signaniso_T1.nii.gz   — DPS shape indicator in T1 space (nearestneighbour)"
echo "  v1_T1.nii.gz          — Principal eigenvectors in T1 space (rotation-corrected)"
echo ""
echo "Next: run 02_build_conductivity_tensor.py"
