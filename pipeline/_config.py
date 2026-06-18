"""Load the shared pipeline configuration (config/config.sh) into Python.

Single source of truth for paths and the subject ID, shared with the bash scripts. Sources the file in a
subshell so ${VAR} expansions resolve as bash would. Keys are parsed from the config's own KEY= assignments,
so a new config variable is picked up automatically (no hand-maintained allowlist to drift out of sync).

    from _config import cfg
    cfg["WORK_DIR"], cfg["SUBJECT"], cfg["QTI_DPS"], ...
"""
import os
import re
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# PIPELINE_CONFIG lets the cohort batch point each subject at its own config.
_CONFIG_SH = os.environ.get("PIPELINE_CONFIG") or os.path.join(_ROOT, "config", "config.sh")
_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=")   # uppercase KEY= (skips comments, lowercase, export-only)


def load_config(path=_CONFIG_SH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy config/config.example.sh to config/config.sh "
            "and edit it for your machine."
        )
    with open(path) as f:
        keys = list(dict.fromkeys(m.group(1) for m in map(_ASSIGN.match, f) if m))   # file order, deduped
    if not keys:
        raise ValueError(f"no KEY= assignments found in {path}")
    script = f'set -a; source "{path}"; ' + "; ".join(f'echo "${k}"' for k in keys)
    out = subprocess.check_output(["bash", "-c", script], text=True).splitlines()
    return dict(zip(keys, out))


cfg = load_config()
