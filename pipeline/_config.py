"""Load the shared pipeline configuration (config/config.sh) into Python.

Single source of truth for paths and the subject ID: the bash scripts source
config/config.sh directly; the Python scripts read the same values through this
helper, which sources the file in a subshell so ${VAR} expansions resolve exactly
as bash would.

    from _config import cfg
    cfg["WORK_DIR"], cfg["SUBJECT"], cfg["DPS_MAT"], ...
"""
import os
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_SH = os.path.join(_ROOT, "config", "config.sh")


def load_config(path=_CONFIG_SH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy config/config.example.sh to config/config.sh "
            "and edit it for your machine."
        )
    keys = ("SUBJECT DATA_DIR WORK_DIR NII_DIR FIT_DIR M2M_DIR REG_DIR "
            "DWI_NII DWI_BVAL DWI_BVEC STE_B0_NII DPS_MAT SIMNIBS_BIN FSLDIR "
            "MRE_STIFFNESS MRE_STORAGE MRE_LOSS MRE_CONFIDENCE").split()
    script = f'set -a; source "{path}"; ' + "; ".join(f'echo "${k}"' for k in keys)
    out = subprocess.check_output(["bash", "-c", script], text=True).splitlines()
    return dict(zip(keys, out))


cfg = load_config()
