"""Load the shared pipeline configuration (config/config.sh) into Python.

Single source of truth for paths and the subject ID, shared with the bash
scripts. Sources the file in a subshell so ${VAR} expansions resolve as bash would.

    from _config import cfg
    cfg["WORK_DIR"], cfg["SUBJECT"], cfg["QTI_DPS"], ...
"""
import os
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PIPELINE_CONFIG lets the cohort batch point each subject at its own config.
_CONFIG_SH = os.environ.get("PIPELINE_CONFIG") or os.path.join(_ROOT, "config", "config.sh")


def load_config(path=_CONFIG_SH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy config/config.example.sh to config/config.sh "
            "and edit it for your machine."
        )
    keys = ("SUBJECT DATA_DIR WORK_DIR NII_DIR FIT_DIR M2M_DIR REG_DIR "
            "DWI_NII DWI_BVAL DWI_BVEC STE_B0_NII QTI_MFS QTI_DPS SIMNIBS_BIN FSLDIR "
            "MRE_STIFFNESS MRE_ALPHA MRE_STORAGE MRE_LOSS MRE_CONFIDENCE").split()
    script = f'set -a; source "{path}"; ' + "; ".join(f'echo "${k}"' for k in keys)
    out = subprocess.check_output(["bash", "-c", script], text=True).splitlines()
    return dict(zip(keys, out))


cfg = load_config()
