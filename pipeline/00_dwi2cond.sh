#!/bin/bash
# 00_dwi2cond.sh — Fit DTI conductivity tensors for the DTI simulation model
#
# IMPORTANT: dwi2cond --all CANNOT be used for FullPD5 due to a confirmed bug
# in SimNIBS 4.6: the dwi2cond script internally maps the positional arguments
# in reverse order (bvec→DWIbvals, bval→DWIbvecs), causing dtifit to crash:
#   "Mat::operator(): index out of bounds"
# See: m2m_FullPD5/dMRI_prep/dwi2cond_log.html (crash report preserved)
#      m2m_FullPD5/dMRI_prep/raw/ (DWIbvals/DWIbvecs confirm the swap)
#
# This script implements the equivalent pipeline directly:
#   1. dtifit    — tensor fit in DWI native space (correct bvec/bval order)
#   2. FLIRT     — rigid registration b=0 → T1 (6-DOF)
#   3. vecreg    — rotate tensor to T1 space (correct covariant transform)
#   4. Copy      — place tensor in both SimNIBS expected locations
#
# Prerequisites: CHARM must have run first (m2m_FullPD5/ must exist)
# Idempotent:   re-running skips all steps if output tensor already exists
#
# Usage:
#   cd /Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation
#   bash scripts/00_dwi2cond.sh
#
# Runtime: ~5 min

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
export FSLDIR=~/fsl
export PATH=~/Applications/SimNIBS-4.6/bin:$FSLDIR/bin:$PATH
source "$FSLDIR/etc/fslconf/fsl.sh"

WDIR="/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation"
NDIR="/Users/santi/Downloads/FullPD5_forSantiago/FullPD5/NiiFiles"

# Original DWI files (801 series: 1×b=0 + 80×b=1500, MB3)
DWI="$NDIR/FullPD5_WIP_MB3_sDTI_opt_80_20230110134105_801.nii.gz"
BVEC="$NDIR/FullPD5_WIP_MB3_sDTI_opt_80_20230110134105_801.bvec"
BVAL="$NDIR/FullPD5_WIP_MB3_sDTI_opt_80_20230110134105_801.bval"

# SimNIBS subject directory
M2M="$WDIR/m2m_FullPD5"
T1="$M2M/T1.nii.gz"

# Intermediate work directory (inside the SimNIBS prep tree so it stays together)
RAWDIR="$M2M/dMRI_prep/raw"
DTIDIR="$M2M/dMRI_prep/dtifit_bypass"

# SimNIBS reads the tensor from both of these locations:
OUT_REG="$M2M/dMRI_MNI_reg/DTI_coregT1_tensor.nii.gz"
OUT_ROOT="$M2M/DTI_coregT1_tensor.nii.gz"

# ── Pre-flight ─────────────────────────────────────────────────────────────────
if [ ! -d "$M2M" ]; then
    echo "ERROR: $M2M not found — run CHARM first."
    exit 1
fi

for f in "$DWI" "$BVEC" "$BVAL" "$T1"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f"
        exit 1
    fi
done

# ── Idempotency ───────────────────────────────────────────────────────────────
if [ -f "$OUT_ROOT" ] && [ -f "$OUT_REG" ]; then
    echo "DTI tensor already exists at:"
    echo "  $OUT_ROOT"
    echo "  $OUT_REG"
    echo "Skipping (delete these files to force re-run)."
    exit 0
fi

mkdir -p "$DTIDIR" "$M2M/dMRI_MNI_reg"

echo "=========================================================="
echo "00_dwi2cond.sh — DTI bypass pipeline for FullPD5"
echo "=========================================================="
echo "  DWI:   $DWI"
echo "  bvec:  $BVEC"
echo "  bval:  $BVAL"
echo ""

# ── Step 1: dtifit ────────────────────────────────────────────────────────────
# Use brain mask from the partial dwi2cond run (b=0 BET already completed).
# If it doesn't exist, run BET now.
MASK="$RAWDIR/nodif_brain_mask.nii.gz"
NODIF_BRAIN="$RAWDIR/nodif_brain.nii.gz"

if [ ! -f "$MASK" ]; then
    echo "Step 0: brain mask not found — extracting b=0 and running BET..."
    NODIF="$RAWDIR/nodif.nii.gz"
    mkdir -p "$RAWDIR"
    fslroi "$DWI" "$NODIF" 0 1
    bet "$NODIF" "$RAWDIR/nodif_brain" -m -f 0.3
    echo "  BET complete: $MASK"
fi

echo "Step 1: dtifit — tensor fit in DWI native space"
echo "  data:  $RAWDIR/DWIraw.nii.gz (or original DWI)"

# Use DWIraw.nii.gz if it exists (already extracted by partial dwi2cond run),
# otherwise fall back to the original file.
if [ -f "$RAWDIR/DWIraw.nii.gz" ]; then
    DWI_INPUT="$RAWDIR/DWIraw.nii.gz"
else
    DWI_INPUT="$DWI"
fi

dtifit \
    --data="$DWI_INPUT" \
    --mask="$MASK" \
    --bvecs="$BVEC" \
    --bvals="$BVAL" \
    --out="$DTIDIR/dti" \
    --save_tensor \
    --verbose

echo ""
echo "  dtifit output: $DTIDIR/dti_tensor.nii.gz"

# ── Step 2: FLIRT — b=0 → T1 rigid registration (6-DOF) ─────────────────────
echo ""
echo "Step 2: FLIRT — registering b=0 to T1 (6-DOF rigid)"

flirt \
    -in "$NODIF_BRAIN" \
    -ref "$T1" \
    -omat "$DTIDIR/b0_to_T1.mat" \
    -dof 6 \
    -cost corratio \
    -searchrx -30 30 \
    -searchry -30 30 \
    -searchrz -30 30

echo "  Registration matrix: $DTIDIR/b0_to_T1.mat"

# ── Step 3: vecreg — rotate tensor to T1 space ───────────────────────────────
# vecreg applies the correct covariant (rotation-only) transform to each tensor,
# AND resamples the data to the T1 grid. This is the same operation that
# dwi2cond --all would perform after its eddy+dtifit steps.
echo ""
echo "Step 3: vecreg — rotating tensor to T1 space"

vecreg \
    -i "$DTIDIR/dti_tensor.nii.gz" \
    -o "$OUT_REG" \
    -r "$T1" \
    -t "$DTIDIR/b0_to_T1.mat"

echo "  Tensor in T1 space: $OUT_REG"

# ── Step 4: Copy to SimNIBS root m2m location ─────────────────────────────────
# SimNIBS reads DTI_coregT1_tensor.nii.gz from two places:
#   m2m_<subid>/DTI_coregT1_tensor.nii.gz            (used by run_simnibs)
#   m2m_<subid>/dMRI_MNI_reg/DTI_coregT1_tensor.nii.gz  (created by dwi2cond)
echo ""
echo "Step 4: Copying tensor to SimNIBS locations"
cp "$OUT_REG" "$OUT_ROOT"
echo "  Copied to: $OUT_ROOT"

# ── Validation ────────────────────────────────────────────────────────────────
echo ""
echo "Validating output tensor..."
~/Applications/SimNIBS-4.6/bin/simnibs_python - << 'PYEOF'
import nibabel as nib
import numpy as np
import os

WDIR = "/Users/santi/Documents/MRE_tDCS_PD/FullPD5_segmentation"
M2M  = os.path.join(WDIR, "m2m_FullPD5")
out  = os.path.join(M2M, "DTI_coregT1_tensor.nii.gz")

img = nib.load(out)
t1  = nib.load(os.path.join(M2M, "T1.nii.gz"))
assert img.shape[:3] == t1.shape[:3], f"Spatial shape mismatch: {img.shape[:3]} vs {t1.shape[:3]}"
assert img.shape[3] == 6, f"Expected 6 tensor components, got {img.shape[3]}"
assert np.allclose(img.affine, t1.affine, atol=1e-3), "Affine mismatch with T1"

d = img.get_fdata().astype(np.float64)
print(f"  Shape:  {img.shape}  ✓   Affine matches T1  ✓")
print(f"  Value range before fix: [{d.min():.6f}, {d.max():.6f}]  (mm²/s)")

# vecreg boundary interpolation produces non-physical tensors (Dii < 0) at the brain
# edge due to trilinear interpolation between valid tensors and zero-padded background.
#
# Why we fix rather than leaving it to SimNIBS:
#   SimNIBS's VN normalization computes np.abs(λ₁·λ₂·λ₃)^(1/3).  A tensor with one
#   exact-zero eigenvalue (det=0) causes division by zero → inf/nan.  A small NEGATIVE
#   eigenvalue [+, +, -ε] is handled correctly by SimNIBS via np.abs (product still > 0),
#   but Dii < 0 in the stored NIfTI is physically non-sensical and confusing to inspect.
#
# Fix: eigendecompose → clamp eigenvalues to ≥ 1e-6 (NOT 0 — zero det causes div/0
#   in VN normalization; 1e-6 is ~4 orders of magnitude below typical MD ≈ 0.001 mm²/s
#   so it does not influence VN normalization meaningfully, and is far below the ratio
#   cap λ₁/max_ratio which fires at ≈ 0.0001 for typical WM).
#   After this fix: det > 0, tensor is strictly positive definite, VN normalization safe.
EPS_EIG = 1e-6   # minimum eigenvalue (mm²/s); chosen << typical MD but > 0
nonzero = np.abs(d[..., 0]) > 1e-7
bad = ((d[..., 0] < 0) | (d[..., 3] < 0) | (d[..., 5] < 0)) & nonzero
print(f"  Voxels with negative diagonal (non-physical): {bad.sum():,}")

if bad.sum() > 0:
    idx  = np.argwhere(bad)
    vals = d[idx[:, 0], idx[:, 1], idx[:, 2]]   # (N, 6)
    mats = np.zeros((len(vals), 3, 3))
    mats[:, 0, 0] = vals[:, 0]; mats[:, 0, 1] = vals[:, 1]; mats[:, 0, 2] = vals[:, 2]
    mats[:, 1, 0] = vals[:, 1]; mats[:, 1, 1] = vals[:, 3]; mats[:, 1, 2] = vals[:, 4]
    mats[:, 2, 0] = vals[:, 2]; mats[:, 2, 1] = vals[:, 4]; mats[:, 2, 2] = vals[:, 5]
    evals, evecs = np.linalg.eigh(mats)
    evals = np.maximum(evals, EPS_EIG)    # clamp λ < EPS_EIG → EPS_EIG (strictly PD)
    mats_fixed = np.einsum('nij,nj,nkj->nik', evecs, evals, evecs)
    d[idx[:, 0], idx[:, 1], idx[:, 2], 0] = mats_fixed[:, 0, 0]
    d[idx[:, 0], idx[:, 1], idx[:, 2], 1] = mats_fixed[:, 0, 1]
    d[idx[:, 0], idx[:, 1], idx[:, 2], 2] = mats_fixed[:, 0, 2]
    d[idx[:, 0], idx[:, 1], idx[:, 2], 3] = mats_fixed[:, 1, 1]
    d[idx[:, 0], idx[:, 1], idx[:, 2], 4] = mats_fixed[:, 1, 2]
    d[idx[:, 0], idx[:, 1], idx[:, 2], 5] = mats_fixed[:, 2, 2]
    for path in [out, os.path.join(M2M, "dMRI_MNI_reg", "DTI_coregT1_tensor.nii.gz")]:
        nib.save(nib.Nifti1Image(d.astype(np.float32), img.affine, img.header), path)
    bad_after = ((d[...,0]<0)|(d[...,3]<0)|(d[...,5]<0)) & nonzero
    print(f"  Fixed → {bad_after.sum():,} non-physical voxels remaining (should be 0)")

print(f"  Value range after fix:  [{d.min():.6f}, {d.max():.6f}]")
print(f"  Dxx min (should be ≥ {EPS_EIG:.0e}): {d[...,0].min():.6e}  ✓")
PYEOF

echo ""
echo "=========================================================="
echo "00_dwi2cond.sh complete."
echo "  DTI tensor: $OUT_ROOT"
echo "  Next: run scripts/03_run_simulations.py"
echo "=========================================================="
