"""
04_extract_roi_efield.py — Extract and compare E-field across ROIs for ISO / DTI / MD-dMRI models

TODO: Implement regional E-field comparison between the three conductivity models.

Planned ROIs (PD-relevant white matter tracts):
  - Substantia Nigra (SN)
  - Subthalamic Nucleus (STN)
  - Corticospinal tract (CST)
  - Temporal white matter
  - Occipital white matter

Approach:
  1. Load each simulation mesh (.msh) from sim_ISO/, sim_DTI/, sim_MD_dMRI/
  2. Extract E-field magnitude at GM/WM tetrahedra
  3. Map ROIs from MNI atlas to subject space via m2m_<subid>/toMNI/ warps
  4. Compute percentile statistics within each ROI
  5. Plot ISO vs DTI vs MD-dMRI distributions per ROI

Usage:
  simnibs_python analysis/04_extract_roi_efield.py
"""

raise NotImplementedError("04_extract_roi_efield.py is not yet implemented — see docstring for plan.")
