"""mesh_qc.py - adversarial head-mesh QC across the cohort: HOLES, QUALITY, CONSISTENCY.

This complements analysis/qc_harness.py (which scores the whole pipeline per subject) by interrogating
ONLY the charm head mesh geometry, and doing so adversarially: it hunts for the one broken mesh rather
than reporting reassuring cohort averages. Three axes:

  HOLES (topology)    per tissue-surface tag, an edge on a watertight closed surface is shared by exactly
                      2 triangles. edges on 1 triangle are BOUNDARY edges (a hole); edges on >2 are
                      NON-MANIFOLD. also extracts the tet mesh outer boundary (faces on exactly 1 tet) and
                      counts its connected components: >1 closed component means an enclosed interior void.
  QUALITY             tet quality (SimNIBS tetrahedra_quality, 1.0 ideal, low = sliver): median, p1, min,
                      frac < 0.1, count < 0.01; plus signed-volume sign to count INVERTED tets (vol <= 0),
                      which give negative Jacobians and silently corrupt the FEM solve.
  CONSISTENCY         node/tet/tri counts, tissue-label set, per-tissue volume (mm3). cohort-relative:
                      any subject > 3 MAD from the cohort median, or missing a label others have, is flagged.

Output: analysis/results/mesh_qc_cohort.csv (results/ is gitignored, so scan-date subject IDs are fine
here) plus an adversarial console report ranking the worst subjects and listing every failure.

Usage:
  $SIMNIBS_BIN/simnibs_python analysis/mesh_qc.py                 # all subjects under data/cohort_local
  $SIMNIBS_BIN/simnibs_python analysis/mesh_qc.py --subject <id>  # one subject (verification)
"""
import os, sys, csv, glob, gc, argparse
import numpy as np
from simnibs import mesh_io
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COHORT = os.path.join(ROOT, "data", "cohort_local")
OUT = os.path.join(ROOT, "analysis", "results", "mesh_qc_cohort.csv")
TISSUE = {1: "WM", 2: "GM", 3: "CSF", 5: "scalp", 6: "eyes", 7: "compactbone",
          8: "spongybone", 9: "blood", 10: "muscle"}   # charm volume tags (4.x)
VOLCOLS = ["WM", "GM", "CSF", "scalp", "compactbone", "spongybone"]   # reported per-subject volumes


def subjects():
    out = []
    for d in sorted(glob.glob(os.path.join(COHORT, "*"))):
        sid = os.path.basename(d)
        msh = os.path.join(d, "work", f"m2m_{sid}", f"{sid}.msh")
        if os.path.isfile(msh):
            out.append((sid, msh))
    return out


def _boundary_topology(tet_nodes):
    """Topology of the tet-mesh outer boundary (faces on exactly one tet) - the physically meaningful
    'holes' axis (SimNIBS +1000 surface tags are interface patches, not closed shells, so per-tag edge
    counts measure the storage convention, not defects). Returns:
      free_edges      boundary-face edges on exactly 1 boundary face -> a genuine OPEN HOLE/tear (should be 0)
      nonman_edges    boundary-face edges on >2 boundary faces -> non-manifold pinch (small counts normal)
      ncomp           connected components of the boundary surface (1 outer scalp + cavities/islands)
      n_tiny          components < 20 faces (floating specks)
      largest_frac    largest component as a fraction of all boundary faces (~1.0 for a clean single head)
      nbf             number of boundary faces
    """
    f = np.sort(np.vstack([tet_nodes[:, [0, 1, 2]], tet_nodes[:, [0, 1, 3]],
                           tet_nodes[:, [0, 2, 3]], tet_nodes[:, [1, 2, 3]]]), axis=1)
    uf, cnt = np.unique(f, axis=0, return_counts=True)
    bf = uf[cnt == 1]                                    # boundary faces
    ne = bf.shape[0]
    if ne == 0:
        return 0, 0, 0, 0, 0.0, 0
    edges = np.sort(np.vstack([bf[:, [0, 1]], bf[:, [1, 2]], bf[:, [0, 2]]]), axis=1)
    fid = np.tile(np.arange(ne), 3)
    ue, einv, ecnt = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    free_edges = int((ecnt == 1).sum())                 # open hole / tear (count==1)
    nonman_edges = int((ecnt > 2).sum())                # non-manifold pinch (count>2)
    order = np.argsort(einv, kind="stable")             # group face-ids by shared edge for adjacency
    fid_s, ei_s = fid[order], einv[order]
    same = ei_s[1:] == ei_s[:-1]
    a, b = fid_s[:-1][same], fid_s[1:][same]
    g = coo_matrix((np.ones(a.size), (a, b)), shape=(ne, ne))
    ncomp, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    return free_edges, nonman_edges, int(ncomp), int((sizes < 20).sum()), round(float(sizes.max() / ne), 4), ne


def evaluate(sid, mshpath):
    msh = mesh_io.read_msh(mshpath)
    et = msh.elm.elm_type
    tet = et == 4
    tri = et == 2
    nl = msh.elm.node_number_list                       # (Nelm, 4); triangles use cols 0:3
    coords = msh.nodes.node_coord
    tags_v = msh.elm.tag1[tet]
    m = dict(subject=sid, n_nodes=int(msh.nodes.nr), n_tet=int(tet.sum()), n_tri=int(tri.sum()))

    # --- QUALITY ---
    q = msh.tetrahedra_quality()[1].value
    qt = q[np.isfinite(q)]
    tn = nl[tet][:, :4] - 1                              # node indices (1-based -> 0-based)
    p = coords[tn]                                       # (Ntet,4,3)
    vol6 = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    # inverted = non-positive signed volume (gmsh positive-orientation convention). Guard against a global
    # opposite convention: genuine inverted tets are always a small minority, so if MOST are 'negative' the
    # whole mesh uses the opposite sign and the test is flipped (avoids a false "all tets inverted" alarm).
    inv = (vol6 <= 0) if (vol6 < 0).mean() <= 0.5 else (vol6 >= 0)
    m.update(tetq_med=round(float(np.median(qt)), 4), tetq_p1=round(float(np.percentile(qt, 1)), 4),
             tetq_min=round(float(qt.min()), 5), tetq_badfrac=round(float(np.mean(qt < 0.1)), 5),
             n_q_lt_001=int((qt < 0.01).sum()), n_inverted=int(inv.sum()))

    # --- HOLES (volume-boundary topology) ---
    free_e, nonman_e, ncomp, n_tiny, largest_frac, nbf = _boundary_topology(tn)
    m.update(bnd_free_edges=free_e, bnd_nonmanifold_edges=nonman_e, outer_components=ncomp,
             n_tiny_comp=n_tiny, largest_comp_frac=largest_frac, outer_boundary_faces=nbf)

    # --- CONSISTENCY ---
    present = sorted(int(t) for t in np.unique(tags_v))
    m["tags_present"] = "|".join(TISSUE.get(t, str(t)) for t in present)
    # per-tet volume = |vol6|/6; sum per tissue tag (mm3)
    av = np.abs(vol6) / 6.0
    for tg, name in TISSUE.items():
        if name in VOLCOLS:
            m[f"vol_{name}_mm3"] = int(av[tags_v == tg].sum()) if (tags_v == tg).any() else 0
    del msh, q, qt, tn, p, vol6, av
    gc.collect()
    return m


def mad_flags(rows, col):
    x = np.array([r[col] for r in rows], float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) or 1e-9
    return {rows[i]["subject"] for i in range(len(rows)) if abs(x[i] - med) / (1.4826 * mad) > 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default=None)
    args = ap.parse_args()
    subs = subjects()
    if args.subject:
        subs = [(s, m) for s, m in subs if s == args.subject]
        if not subs:
            sys.exit(f"subject {args.subject} not found under {COHORT}")
    print(f"adversarial mesh QC over {len(subs)} subject(s)\n")
    rows = []
    for i, (sid, mshpath) in enumerate(subs, 1):
        print(f"[{i}/{len(subs)}] {sid} ...", flush=True)
        try:
            rows.append(evaluate(sid, mshpath))
        except Exception as e:                           # noqa: BLE001
            print(f"  *** FAILED to evaluate {sid}: {type(e).__name__}: {e}")
    if not rows:
        sys.exit("no meshes evaluated")

    cols = ["subject", "n_nodes", "n_tet", "n_tri", "tags_present", "tetq_med", "tetq_p1", "tetq_min",
            "tetq_badfrac", "n_q_lt_001", "n_inverted", "bnd_free_edges", "bnd_nonmanifold_edges",
            "outer_components", "n_tiny_comp", "largest_comp_frac", "outer_boundary_faces"] + \
           [f"vol_{n}_mm3" for n in VOLCOLS]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}  ({len(rows)} rows)")

    # ---- adversarial report ----
    # ABSOLUTE failures: convention-independent, must be zero on any valid mesh.
    # COHORT-RELATIVE flags: every mesh shares the same charm version/conventions, so a real defect is an
    # outlier, not a nonzero count. (>3 MAD, active only at n>=4.)
    print("\n================ ADVERSARIAL FINDINGS ================")
    hard = []
    for r in rows:
        if r["bnd_free_edges"] > 0:
            hard.append(f"OPEN HOLE  {r['subject']}: {r['bnd_free_edges']} free boundary edges (solid is torn, not closed)")
        if r["n_inverted"] > 0:
            hard.append(f"INVERTED   {r['subject']}: {r['n_inverted']} tets with signed volume <= 0 (negative Jacobian)")
    label_sets = [set(r["tags_present"].split("|")) for r in rows]
    union = set().union(*label_sets)
    for r, ls in zip(rows, label_sets):
        miss = union - ls
        if miss:
            hard.append(f"LABELSET   {r['subject']}: missing tissue label(s) {sorted(miss)} present in other subjects")

    soft = []
    if len(rows) >= 4:
        for col in ["n_tet", "n_nodes", "tetq_med", "bnd_nonmanifold_edges", "outer_components",
                    "n_tiny_comp", "vol_WM_mm3", "vol_GM_mm3", "vol_CSF_mm3"]:
            for s in sorted(mad_flags(rows, col)):
                v = next(r[col] for r in rows if r["subject"] == s)
                soft.append(f"OUTLIER  {s}: {col}={v} is > 3 MAD from cohort median")

    print("\n-- ABSOLUTE (must be clean) --")
    if hard:
        for line in hard:
            print("  " + line)
    else:
        print("  PASS: every mesh is closed (0 free edges), has 0 inverted tets, and shares the full tissue-label set.")
    print("\n-- COHORT OUTLIERS (>3 MAD; inspect, not necessarily broken) --")
    if soft:
        for line in soft:
            print("  " + line)
    else:
        print("  none: all subjects within 3 MAD on counts, quality, topology, and tissue volumes.")

    # adversarial: surface the tail, not the mean
    print("\n---- lowest median tet quality (worst 5) ----")
    for r in sorted(rows, key=lambda r: r["tetq_med"])[:5]:
        print(f"  {r['subject']:<28} tetq_med={r['tetq_med']}  badfrac={r['tetq_badfrac']}  "
              f"min={r['tetq_min']}  n<0.01={r['n_q_lt_001']}  inverted={r['n_inverted']}")
    print("\n---- topology spread (boundary closedness) ----")
    print(f"  free edges (open holes): max over cohort = {max(r['bnd_free_edges'] for r in rows)}  (want 0)")
    print(f"  largest_comp_frac: min = {min(r['largest_comp_frac'] for r in rows)}  (want ~1.0; low = fragmented)")
    print("\n---- count/volume spread (cohort consistency) ----")
    for col in ["n_tet", "vol_WM_mm3", "vol_GM_mm3", "vol_CSF_mm3"]:
        x = np.array([r[col] for r in rows], float)
        print(f"  {col:<14} median={np.median(x):,.0f}  min={x.min():,.0f}  max={x.max():,.0f}  "
              f"CV={100*x.std()/x.mean():.1f}%")
    print("======================================================")


if __name__ == "__main__":
    main()
