#!/bin/bash
# Step 0b: dwi2cond — fit DTI conductivity tensors for the DTI model
#
# MUST run AFTER charm — needs m2m_FullPD5/ for T1 coregistration.
# Creates m2m_FullPD5/dMRI_MNI_reg/ with conductivity tensors used by sim_DTI.
#
# Uses DTI series 801: b=0 + b=1500, multiband factor 3, 2mm slices.
# dwi2cond pipeline: eddy correction → tensor fit → volume normalisation.
#
# Usage: bash scripts/00_dwi2cond.sh
# Runtime: ~30 min

set -e

export PATH=~/Applications/SimNIBS-4.6/bin:$PATH
export FSLDIR=~/fsl
export PATH=$FSLDIR/bin:$PATH
source $FSLDIR/etc/fslconf/fsl.sh

WDIR="/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation"
NDIR="/Users/santi/Downloads/FullPD5_forSantiago/FullPD5/NiiFiles"

DWI="$NDIR/FullPD5_WIP_MB3_sDTI_opt_80_20230110134105_801.nii.gz"
BVEC="$NDIR/FullPD5_WIP_MB3_sDTI_opt_80_20230110134105_801.bvec"
BVAL="$NDIR/FullPD5_WIP_MB3_sDTI_opt_80_20230110134105_801.bval"

if [ ! -d "$WDIR/m2m_FullPD5" ]; then
    echo "ERROR: m2m_FullPD5/ not found. Run CHARM first."
    exit 1
fi

cd "$WDIR"

echo "=== Running dwi2cond on FullPD5 DTI data ==="
echo "  DWI:  $DWI"
echo "  bvec: $BVEC"
echo "  bval: $BVAL"
echo ""

# --all: full pipeline (eddy correction + tensor fit + volume normalisation)
# Output goes into m2m_FullPD5/dMRI_MNI_reg/
dwi2cond --all FullPD5 "$DWI" "$BVEC" "$BVAL" \
    2>&1 | tee dwi2cond_FullPD5.log

echo ""
echo "=== dwi2cond complete ==="
echo "Conductivity tensors: $WDIR/m2m_FullPD5/dMRI_MNI_reg/"
echo "Log: $WDIR/dwi2cond_FullPD5.log"
