"""test_tensor_divergence.py - synthetic-tensor checks for the divergence helpers used by
analysis/08_tensor_divergence.py (eigh_6comp, fa_from_evals, v1_angle_deg). Build tensors with KNOWN
eigenvalues / orientation and assert the helpers recover the angle, FA difference, and ratio difference.
Run: simnibs_python tests/test_tensor_divergence.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from _rois import eigh_6comp, fa_from_evals, v1_angle_deg  # noqa: E402

checks = []


def check(name, ok, detail=""):
    checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def rot_x(deg):
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def tensor6(evals_asc, R):
    """6-comp FSL tensor (xx,xy,xz,yy,yz,zz) from ascending eigenvalues + rotation R (columns=eigenvectors)."""
    D = R @ np.diag(evals_asc) @ R.T
    return np.array([D[0, 0], D[0, 1], D[0, 2], D[1, 1], D[1, 2], D[2, 2]])


def _eig1(t6):
    return eigh_6comp(t6[None, :], np.array([True]))     # one voxel -> (ev (1,3), vec (1,3,3))


def test_known_orientation_angle():
    # same prolate eigenvalues, principal axis tilted 30 deg -> angle must be 30
    _, vA = _eig1(tensor6([1.0, 1.0, 3.0], np.eye(3)))
    _, vB = _eig1(tensor6([1.0, 1.0, 3.0], rot_x(30)))
    ang = float(v1_angle_deg(vA, vB)[0])
    check("V1 angle recovers 30 deg", abs(ang - 30.0) < 0.5, f"got {ang:.2f}")


def test_sign_agnostic():
    _, vA = _eig1(tensor6([1.0, 1.0, 3.0], np.eye(3)))
    check("V1 vs -V1 is 0 deg", float(v1_angle_deg(vA, -vA)[0]) < 1e-6, "eigenvector sign is arbitrary")


def test_known_fa_difference():
    evA, _ = _eig1(tensor6([1.0, 1.0, 3.0], np.eye(3)))
    evB, _ = _eig1(tensor6([1.0, 1.0, 2.0], np.eye(3)))
    dfa = float(fa_from_evals(evA)[0] - fa_from_evals(evB)[0])
    expected = 2.0 / np.sqrt(11.0) - 1.0 / np.sqrt(6.0)   # analytic FA([1,1,3]) - FA([1,1,2])
    check("dFA matches analytic", abs(dfa - expected) < 1e-9, f"{dfa:.4f} vs {expected:.4f}")


def test_known_ratio_difference():
    evA, _ = _eig1(tensor6([1.0, 1.0, 3.0], np.eye(3)))   # lam1/lam3 = 3
    evB, _ = _eig1(tensor6([1.0, 1.0, 2.0], np.eye(3)))   # lam1/lam3 = 2
    dr1 = float(evA[0, 2] / evA[0, 0] - evB[0, 2] / evB[0, 0])
    check("d(lam1/lam3) recovers 3-2=1", abs(dr1 - 1.0) < 1e-9, f"got {dr1:.4f}")


def test_eigh_recovers_eigenvalues():
    ev, _ = _eig1(tensor6([0.5, 1.5, 4.0], rot_x(17)))    # non-degenerate, rotated
    check("eigh recovers ascending eigenvalues", np.allclose(ev[0], [0.5, 1.5, 4.0], atol=1e-9), f"got {ev[0]}")


if __name__ == "__main__":
    print("test_tensor_divergence (synthetic tensors, known angle/FA/ratio differences)")
    test_known_orientation_angle()
    test_sign_agnostic()
    test_known_fa_difference()
    test_known_ratio_difference()
    test_eigh_recovers_eigenvalues()
    print(f"\n{sum(checks)}/{len(checks)} tests passed")
    sys.exit(0 if all(checks) else 1)
