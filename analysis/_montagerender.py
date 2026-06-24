"""_montagerender.py - render the scalp with the tDCS electrodes (montage placement) via gmsh, offscreen.

The SimNIBS FEM scalp surface (tag 1005) is open at the neck (its only boundary), so it is first made
WATERTIGHT by fan-capping that boundary loop to a centroid - then no view shows a hole. The two electrode
patches (anode + cathode tags) are colored on top of the gray scalp via a 3-band color table
(gray scalp / red anode / blue cathode). One gmsh session per process (repeated init segfaults), so this is
invoked as a one-shot worker by fig3_field_maps.py.

Run with $SIMNIBS_BIN/simnibs_python.
"""
import os
import sys
import numpy as np
from simnibs import mesh_io

SCALP_TAG = 1005
ANODE_TAGS = (1501, 2101)     # C3 gel + rubber
CATHODE_TAGS = (1502, 2102)   # Fp2 gel + rubber
SCALP_RGB, ANODE_RGB, CATHODE_RGB = (228, 193, 162), (211, 47, 47), (25, 118, 210)  # natural skin tone


def build_capped_head(sim_mesh, out_msh):
    """Crop scalp+electrodes from a sim mesh, cap the scalp neck opening (watertight), tag a montage field
    (0 scalp / 1 anode / 2 cathode), and write the result. Returns out_msh."""
    m = mesh_io.read_msh(sim_mesh)
    sub = m.crop_mesh(tags=[SCALP_TAG, *ANODE_TAGS, *CATHODE_TAGS])
    nodes = sub.nodes.node_coord.copy()
    tris = sub.elm.node_number_list[:, :3].astype(np.int64)   # 1-based
    tags = sub.elm.tag1
    val = np.where(np.isin(tags, ANODE_TAGS), 1.0, np.where(np.isin(tags, CATHODE_TAGS), 2.0, 0.0))

    # cap the scalp neck: fan its boundary (free) edges to a centroid so the surface is closed
    sc = tris[tags == SCALP_TAG] - 1
    edges = np.sort(np.vstack([sc[:, [0, 1]], sc[:, [1, 2]], sc[:, [0, 2]]]), axis=1)
    uniq, cnt = np.unique(edges, axis=0, return_counts=True)
    free = uniq[cnt == 1]
    centroid = nodes[np.unique(free)].mean(0)
    centroid[2] -= 8.0
    ci = len(nodes) + 1
    nodes = np.vstack([nodes, centroid])
    cap = np.column_stack([free[:, 0] + 1, free[:, 1] + 1, np.full(len(free), ci)])
    tris = np.vstack([tris, cap])
    val = np.concatenate([val, np.zeros(len(cap))])

    head = mesh_io.Msh()
    head.nodes = mesh_io.Nodes(nodes)
    head.elm = mesh_io.Elements(triangles=tris.astype(np.int32))
    head.add_element_field(val, "montage")
    mesh_io.write_msh(head, out_msh)
    return out_msh


def render_montage(msh, out_png, rot=(290.0, 350.0, 140.0), width=2000, height=2000):
    """Render a capped head+montage mesh: gray scalp, red anode, blue cathode."""
    import gmsh
    band = [(SCALP_RGB if i < 85 else (ANODE_RGB if i < 171 else CATHODE_RGB)) for i in range(256)]
    gmsh.initialize()
    try:
        gmsh.open(msh)
        nv = int(gmsh.option.getNumber("PostProcessing.NbViews"))
        vi = next(i for i in range(nv) if "montage" in gmsh.option.getString(f"View[{i}].Name").lower())
        optf = out_png + ".opt"
        with open(optf, "w") as fh:
            fh.write("View[%d].ColorTable={\n%s\n};\n" % (vi, ",\n".join("{%d,%d,%d}" % c for c in band)))
        gmsh.merge(optf)
        for k, v in {"RangeType": 2, "CustomMin": 0, "CustomMax": 2, "ShowScale": 0,
                     "Light": 1, "SmoothNormals": 1, "Visible": 1}.items():
            gmsh.option.setNumber(f"View[{vi}].{k}", v)
        for o in ("Mesh.SurfaceFaces", "Mesh.SurfaceEdges", "Mesh.Lines", "Mesh.Points",
                  "General.SmallAxes", "General.Axes"):
            gmsh.option.setNumber(o, 0)
        gmsh.option.setNumber("General.Antialiasing", 1)
        gmsh.option.setNumber("Mesh.LightTwoSide", 1)
        gmsh.option.setColor("General.Background", 255, 255, 255)
        gmsh.option.setNumber("General.Trackball", 0)
        gmsh.option.setNumber("General.RotationX", rot[0])
        gmsh.option.setNumber("General.RotationY", rot[1])
        gmsh.option.setNumber("General.RotationZ", rot[2])
        gmsh.option.setNumber("General.GraphicsWidth", width)
        gmsh.option.setNumber("General.GraphicsHeight", height)
        gmsh.fltk.initialize()
        gmsh.graphics.draw()
        gmsh.write(out_png)
    finally:
        gmsh.finalize()
        if os.path.exists(out_png + ".opt"):
            os.remove(out_png + ".opt")


def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, help="sim mesh to build the capped head from")
    ap.add_argument("--msh", required=True, help="cached capped head mesh path (built if absent)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rot", nargs=3, type=float, default=[250.0, 0.0, 160.0])
    a = ap.parse_args()
    if not os.path.exists(a.msh):
        build_capped_head(a.sim, a.msh)
    render_montage(a.msh, a.out, rot=tuple(a.rot))
    print("rendered montage", a.out)


if __name__ == "__main__":
    _main()
