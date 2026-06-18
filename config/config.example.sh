# Pipeline configuration template.
#
#   cp config/config.example.sh config/config.sh   then edit for your machine.
#
# config/config.sh is gitignored - keep machine paths and subject IDs out of
# version control. Both the bash scripts (source this file) and the Python
# scripts (via pipeline/_config.py) read these values, so this is the single
# source of truth for paths and the subject ID.

# Subject ID - used by CHARM, dwi2cond and SimNIBS for output naming.
SUBJECT="sub-01"

# Raw scanner data root: contains NiiFiles/ (NIfTI) and fit/ (QTI/MD-dMRI fit).
DATA_DIR="/path/to/raw_data"

# Working directory: CHARM writes m2m_${SUBJECT}/ here, plus registration/ and sim_*/.
WORK_DIR="/path/to/work"

# Derived locations (usually no need to change).
NII_DIR="${DATA_DIR}/NiiFiles"
FIT_DIR="${DATA_DIR}/fit"
M2M_DIR="${WORK_DIR}/m2m_${SUBJECT}"
REG_DIR="${WORK_DIR}/registration"

# Subject-specific inputs - edit the file names to match your acquisition.
DWI_NII="${NII_DIR}/dti.nii.gz"           # single-shell DTI (for dwi2cond)
DWI_BVAL="${NII_DIR}/dti.bval"
DWI_BVEC="${NII_DIR}/dti.bvec"
STE_B0_NII="${NII_DIR}/dmri_spherical.nii.gz"  # spherical-encoding series; vol 0 = b0 used for dMRI->T1
# DTI baseline (01_dwi2cond.sh): the SEPARATE single-shell DTI scan. For a standard single-subject config
# this is the same scan as DWI_* above; the cohort points these at the independent ParkMRE_DTI scan.
DTI_DWI="${DWI_NII}"     # eddy/topup-corrected single-shell DTI DWI
DTI_BVEC="${DWI_BVEC}"   # eddy-rotated bvecs
DTI_BVAL="${DWI_BVAL}"   # real bvals; used when token count matches the DWI, else single-shell bvals synthesized from the bvec
# DTI_BVALUE=1500        # b-value for synthesized bvals (default 1500)
# DTI_MASK=""            # optional brain mask for dtifit
# MD-dMRI (QTI) covariance fit, produced by pipeline/run_qti_cov_cohort.m (md-dmri toolbox; run in MATLAB first; see run_qti_cov_cohort.sh).
QTI_MFS="${FIT_DIR}/qti_cov/cov_mfs.mat"  # model fit: oriented mean tensor <D> = m(:,:,:,2:7)
QTI_DPS="${FIT_DIR}/qti_cov/cov_dps.mat"  # derived params: MD, uFA, principal eigenvector u

# Tool locations.
SIMNIBS_BIN="${HOME}/Applications/SimNIBS-4.6/bin"
FSLDIR="${HOME}/fsl"

# MRE (magnetic resonance elastography) maps - for the post-hoc cross-modal comparison.
MRE_STIFFNESS="${NII_DIR}/stiffness.nii.gz"     # |G*| magnitude (kPa-scale)
MRE_ALPHA="${NII_DIR}/alpha.nii.gz"             # springpot exponent (elastic<->viscous); cohort uses this
MRE_STORAGE="${NII_DIR}/storage_modulus.nii.gz" # G' (real shear modulus); leave "" if not available
MRE_LOSS="${NII_DIR}/loss_modulus.nii.gz"       # G'' (loss modulus); viscosity ~ atan(G''/G'); "" if n/a
MRE_CONFIDENCE="${NII_DIR}/mre_confidence.nii.gz"  # leave "" -> no per-voxel gate (use subject-level QC)
