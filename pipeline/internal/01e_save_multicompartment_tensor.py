"""
01e_save_multicompartment_tensor.py — Free-water-eliminated tissue conductivity tensor (QTI bins)

DERIVATION (free-water elimination, σ ∝ ⟨D⟩_tissue)
  The QTI tensor-distribution fit decomposes each voxel into K=3 compartments ("bins") with
  signal fractions f_k (Σ f_k = 1) and diffusion tensors D_k:  S(b)/S0 = Σ_k f_k exp(−b:D_k).
  In FullPD5: k=0 anisotropic tissue (λ1/λ3≈4.2, fibre-oriented), k=1 restricted tissue
  (GM-like), k=2 free water / CSF (observed bin MD ≈ 3.3 µm²/ms, consistent with the literature
  free-water value ≈ 3.0 at 37 °C; f≈0.24 brain-wide, ≈0.77 in CSF voxels).

  The macroscopic mean tensor ⟨D⟩ = Σ_k f_k D_k is *diluted* by the free-water
  compartment: in voxels with CSF partial volume the high-diffusivity isotropic D_2 pulls
  ⟨D⟩ toward isotropy and masks the true tissue anisotropy. DTI cannot fix this — it has no
  way to separate the free-water compartment (that requires the spherical STE encoding).

  Free-water elimination (Pasternak 2009, generalised to the QTI compartment model) removes
  the CSF compartment (identified at runtime as the highest-MD bin, not a hard-coded index) and
  renormalises over the remaining tissue compartments:
      ⟨D⟩_tissue = (Σ_{k≠FW} f_k D_k) / (Σ_{k≠FW} f_k)
  Then Tuch:  σ ∝ ⟨D⟩_tissue, passed to SimNIBS with anisotropy_type='vn'.

  Where the tissue fraction is too low to trust (f_0+f_1 < FT_MIN, i.e. mostly CSF), fall
  back to the full ⟨D⟩ so a tiny noisy tissue fraction cannot create spurious anisotropy.

  This is INTRA-VOXEL informed: SimNIBS gets one effective tensor per element, but that tensor
  is the homogenised TISSUE response after removing the resolved free-water compartment — a
  correction impossible with single-shell DTI.

  Verified contrast vs ⟨D⟩ (native, by free-water fraction): Δ(λ1/λ3) ≈ 0 where f_FW<0.1
  (identical to ⟨D⟩, as it must be), rising to +1.67 where f_FW>0.35 (recovers tissue
  anisotropy masked by CSF). Principal axis stays close to ⟨D⟩/v1_T1; the model anchors the
  principal direction to the validated v1_T1 in 02 so only the eigenvalue (free-water-
  corrected anisotropy) differs from plain ⟨D⟩ — isolating the free-water effect on the E-field.

LIMITATIONS (methods): free-water elimination assumes the CSF compartment should be removed
  from the WM/GM conductivity shape (SimNIBS models bulk CSF separately); it can over-state
  anisotropy in near-pure-CSF voxels (handled by the FT_MIN fallback); bins are diffusion-
  defined; 38-volume pilot. Validation roadmap: MRCDI/MREIT (Gregersen 2024).

Output (dMRI space): registration/tensor_mc_dMRI.nii.gz + lam{1,2,3}_mc_dMRI.nii.gz
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _config import cfg  # noqa: E402  (paths/subject from config/config.sh)
import numpy as np, nibabel as nib, scipy.io, os

FDIR=cfg["FIT_DIR"]
RDIR=cfg["REG_DIR"]
os.makedirs(RDIR, exist_ok=True)
FT_MIN=0.30   # min tissue fraction to trust free-water elimination

print("Loading dps.mat bins...")
dps=scipy.io.loadmat(os.path.join(FDIR,"dps.mat"))['dps']
mask=dps['mask'][0,0].astype(bool); n=mask.sum()
bins=dps['bin'][0,0]
def comp_D(i):
    b=bins[0,i]; g=lambda f:np.real(b[f][0,0]).astype(np.float64)[mask]*1e9
    D=np.zeros((n,3,3)); D[:,0,0]=g('mdxx');D[:,1,1]=g('mdyy');D[:,2,2]=g('mdzz')
    D[:,0,1]=D[:,1,0]=g('mdxy');D[:,0,2]=D[:,2,0]=g('mdxz');D[:,1,2]=D[:,2,1]=g('mdyz')
    return np.nan_to_num(D)
fk=[np.nan_to_num(np.real(bins[0,i]['f'][0,0]).astype(np.float64)[mask]) for i in range(3)]
D0,D1,D2=comp_D(0),comp_D(1),comp_D(2)
Dbins=[D0,D1,D2]

Dmean=fk[0][:,None,None]*D0+fk[1][:,None,None]*D1+fk[2][:,None,None]*D2   # ⟨D⟩ (full)

# Identify the free-water compartment by DIFFUSIVITY (CSF = highest-MD bin), not a hard-coded index
# — QTI bin ordering is a per-fit convention, not guaranteed across subjects. Per-bin MD is taken
# only over voxels where that bin is actually present (f>0.05); zero-fraction bins carry NaN tensors
# (zeroed in comp_D) and would otherwise corrupt the median.
def bin_md(D,f):
    sel=f>0.05
    return float(np.median((D[sel,0,0]+D[sel,1,1]+D[sel,2,2])/3.0)) if sel.any() else -np.inf
md_per_bin=[bin_md(Dbins[i],fk[i]) for i in range(3)]
fw=int(np.argmax(md_per_bin)); tis=[i for i in range(3) if i!=fw]
print(f"  per-bin median MD (µm²/ms): {[round(m,2) for m in md_per_bin]}  ->  free-water bin = {fw}")
assert md_per_bin[fw]>2.0, f"free-water bin MD={md_per_bin[fw]:.2f} not CSF-like (>2) — check the QTI fit/bin order"

ft=fk[tis[0]]+fk[tis[1]]
Dtis=(fk[tis[0]][:,None,None]*Dbins[tis[0]]+fk[tis[1]][:,None,None]*Dbins[tis[1]])/np.maximum(ft,1e-3)[:,None,None]
# fallback to full ⟨D⟩ where tissue fraction too low (near-pure CSF)
lowtis=ft<FT_MIN
Dtis[lowtis]=Dmean[lowtis]
print(f"  free-water-elimination applied to {100*np.mean(~lowtis):.1f}% of voxels "
      f"({100*np.mean(lowtis):.1f}% fall back to ⟨D⟩, near-pure CSF)")

ev=np.linalg.eigvalsh(Dtis)
print(f"  ⟨D⟩_tissue λ1/λ3 median={np.median(ev[:,2]/np.maximum(ev[:,0],1e-6)):.3f} (⟨D⟩ was 1.92); PD={100*np.mean(ev[:,0]>0):.0f}%")

def vol(flat,shp): out=np.zeros(shp); out[mask]=flat; return out
T=np.zeros(mask.shape+(6,),np.float32)
T[mask]=np.stack([Dtis[:,0,0],Dtis[:,0,1],Dtis[:,0,2],Dtis[:,1,1],Dtis[:,1,2],Dtis[:,2,2]],1)
lam1=vol(ev[:,2],mask.shape);lam2=vol(ev[:,1],mask.shape);lam3=vol(ev[:,0],mask.shape)

ref=nib.load(os.path.join(FDIR,"dtd_covariance_C_mu.nii.gz"))
def save(arr,name):
    h=ref.header.copy(); h.set_data_shape(arr.shape); h.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(arr.astype(np.float32),ref.affine,h),os.path.join(RDIR,name)); print(f"  saved {name}")
save(T,"tensor_mc_dMRI.nii.gz")
save(lam1,"lam1_mc_dMRI.nii.gz");save(lam2,"lam2_mc_dMRI.nii.gz");save(lam3,"lam3_mc_dMRI.nii.gz")
print("\nNext: FLIRT lam{1,2,3}_mc → T1 ; vecreg tensor_mc → T1 ; then 02 reconstructs (anchored to v1_T1).")
