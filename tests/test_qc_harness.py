"""Regression tests for qc_harness numeric checks.

Run:  simnibs_python tests/test_qc_harness.py        (assert-based, no pytest needed)
  or  simnibs_python -m pytest tests/test_qc_harness.py

These exercise the REAL extracted functions from qc_harness (not copies), so they guard the actual
code path. The headline test is the regression guard for the vacuous VN-degeneracy bug.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from qc_harness import _vn_check, _eigs_masked  # noqa: E402


def test_vn_check_flags_degenerate_voxels():
    """REGRESSION: the VN degeneracy fraction must be read from the raw smallest eigenvalue.

    Bug: `gmv = exp(mean(log(maximum(e, 1e-12))))` clamps eigenvalues positive before the geom-mean,
    so `gmv > 0` is always True and degfrac was silently 0 -> a non-positive-definite tensor would
    pass unnoticed. Here 2 of 3 voxels are not positive-definite.
    """
    e = np.array([[-0.2, 0.5, 1.0],     # negative smallest eigenvalue -> breaks vn
                  [0.0, 0.5, 1.0],      # zero smallest eigenvalue     -> breaks vn
                  [0.3, 0.6, 0.9]])     # valid
    degfrac, vn_err, hi, lo = _vn_check(e, 0.126)
    assert abs(degfrac - 2.0 / 3.0) < 1e-9, f"degfrac should be 2/3, got {degfrac}"
    # demonstrate the OLD clamped path would have reported 0 (the false-confidence we removed):
    gmv = np.exp(np.mean(np.log(np.maximum(e, 1e-12)), axis=1))
    assert float(np.mean(~(gmv > 0))) == 0.0, "clamped-geomean path reports 0 -> that was the bug"


def test_vn_check_valid_reconstructs_sigma0():
    """Valid positive-definite D: no degeneracy, reconstruction lands exactly on sigma0, sigma > 0."""
    rng = np.random.default_rng(1)
    e = np.sort(np.abs(rng.normal(1.0, 0.3, size=(500, 3))) + 0.1, axis=1)   # anisotropic, PD
    degfrac, vn_err, hi, lo = _vn_check(e, 0.275)
    assert degfrac == 0.0, f"no degenerate voxels expected, got {degfrac}"
    assert vn_err < 1e-9, f"geomean(sigma) must equal sigma0, err={vn_err}"
    assert 0.0 < lo <= hi, f"reconstructed sigma must be positive and ordered, got lo={lo} hi={hi}"


def test_vn_check_all_degenerate_returns_nan_err():
    """All voxels degenerate -> degfrac 1.0 and vn_err NaN (no valid voxel to reconstruct)."""
    e = np.array([[-1.0, -0.5, -0.1], [0.0, 0.0, 0.0]])
    degfrac, vn_err, hi, lo = _vn_check(e, 0.126)
    assert degfrac == 1.0
    assert np.isnan(vn_err) and np.isnan(hi)


def test_eigs_masked_matches_known_tensor():
    """6-comp [Dxx,Dxy,Dxz,Dyy,Dyz,Dzz]=[3,0,0,2,0,1] is diagonal -> eigenvalues {1,2,3} ascending."""
    t = np.zeros((1, 1, 1, 6)); t[0, 0, 0] = [3, 0, 0, 2, 0, 1]
    e = _eigs_masked(t, np.ones((1, 1, 1), bool))
    assert np.allclose(e[0], [1, 2, 3]), f"expected ascending [1,2,3], got {e[0]}"


def test_eigs_masked_off_diagonal():
    """Off-diagonal tensor [2,1,0,2,0,2] has eigenvalues {1,2,3} (2x2 block [[2,1],[1,2]] -> 1,3)."""
    t = np.zeros((1, 1, 1, 6)); t[0, 0, 0] = [2, 1, 0, 2, 0, 2]
    e = _eigs_masked(t, np.ones((1, 1, 1), bool))
    assert np.allclose(e[0], [1, 2, 3]), f"expected [1,2,3], got {e[0]}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\n{len(tests)}/{len(tests)} tests passed")
