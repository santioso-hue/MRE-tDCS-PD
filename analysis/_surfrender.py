"""_surfrender.py - render a scalar field on the smooth middle-gray-matter (central) surface via gmsh,
offscreen, and export a PNG. This is how SimNIBS/Mosayebi-style cortical |E| surface figures are made:
the field is SOLVED on the FEM volume mesh but DISPLAYED on FreeSurfer's watertight central surface
(lh/rh.central.gii), which is smooth and hole-free - unlike the FEM tissue-boundary surface (tag 1002),
whose irregular tessellation shows gaps.

Pipeline: sample the volume field (a |E| NIfTI on the T1 grid) onto the central-surface vertices
(trilinear), hand the surface + per-vertex values to gmsh as node data, apply a matplotlib colormap as a
gmsh ColorTable, set a clean lateral camera + white background, and write the screenshot. gmsh's Python
API ships with SimNIBS and renders offscreen on macOS (verified).

Used by fig3_field_maps.py. Run with $SIMNIBS_BIN/simnibs_python.
"""
import os
import numpy as np
import nibabel as nib
import matplotlib as _mpl
from matplotlib import colormaps as mpl_colormaps
from matplotlib.colors import LinearSegmentedColormap as _LSC
from scipy.ndimage import map_coordinates

# MATLAB 'parula' (blue-cyan-green-yellow) - the SimNIBS/Mosayebi-style |E| colormap; more colorful than
# viridis (no dark-purple low end). Registered by name so both this worker and fig3 can use cmap="parula".
PARULA_BRIGHT = 0.85   # overall brightness of the |E| parula palette (brains + colorbar dim together)
_PARULA_PTS = [
    (0.2081, 0.1663, 0.5292), (0.2116, 0.2875, 0.8053), (0.1786, 0.4049, 0.8835),
    (0.1305, 0.5101, 0.8943), (0.0796, 0.5806, 0.8545), (0.0381, 0.6418, 0.7916),
    (0.0879, 0.6925, 0.7239), (0.2487, 0.7374, 0.6347), (0.4708, 0.7686, 0.5151),
    (0.6720, 0.7793, 0.3624), (0.8418, 0.7872, 0.2438), (0.9514, 0.8169, 0.2667),
    (0.9763, 0.9831, 0.0538)]
_PARULA = _LSC.from_list("parula", [tuple(ch * PARULA_BRIGHT for ch in c) for c in _PARULA_PTS])
try:
    _mpl.colormaps.register(_PARULA)
except (ValueError, AttributeError):
    pass
from simnibs import mesh_io


def load_central_surface(m2m_dir):
    """Combined left+right central (middle-GM) surface. Returns (V Nx3 mm, F Mx3 0-based triangles)."""
    def _gii(p):
        g = nib.load(p)
        return g.darrays[0].data.astype(float), g.darrays[1].data.astype(np.int64)
    lv, lf = _gii(os.path.join(m2m_dir, "surfaces", "lh.central.gii"))
    rv, rf = _gii(os.path.join(m2m_dir, "surfaces", "rh.central.gii"))
    return np.vstack([lv, rv]), np.vstack([lf, rf + len(lv)])


def sample_volume_to_surface(nifti_path, V):
    """Trilinearly sample a scalar NIfTI at the surface vertices (world mm -> voxel)."""
    img = nib.load(nifti_path)
    d = np.asarray(img.dataobj, float)
    inv = np.linalg.inv(img.affine)
    vox = (inv[:3, :3] @ V.T + inv[:3, 3:4]).T
    return map_coordinates(d, vox.T, order=1, mode="nearest")


# camera presets (gmsh General.Rotation X/Y/Z) for a brain in subject RAS
VIEWS = {"left": (270.0, 0.0, 90.0), "right": (270.0, 0.0, 270.0)}

# SmoothNormals=1 = smooth Gouraud shading (clean); 0 = faceted (more fold texture, slightly triangulated).
# AngleSmoothNormals is left at the gmsh default (30) - lowering it to ~12 produced a dark silhouette rim.
SMOOTH_NORMALS = 1
LIGHT0 = (0.0, 0.25, 1.0)     # gmsh Light0 (key): in front of the camera, slightly above
LIGHT1 = None                 # fill light off (two lights washed the surface out)
LIGHT1_DIM = False
LIGHT_SHININESS = 0.0


def render_surface(V, F, values, out_png, cmap="viridis", vmin=0.0, vmax=None,
                   symmetric=False, view="left", rot=None, width=2400, height=1900):
    """Render per-vertex `values` on the surface (V, F) to out_png with a matplotlib colormap.
    symmetric=True centers a diverging map at 0 with +/- vmax. `rot` (rx, ry, rz) overrides the named
    `view` camera. Returns (vmin, vmax) actually used."""
    import gmsh
    if vmax is None:
        vmax = float(np.nanpercentile(np.abs(values) if symmetric else values, 99))
    lo, hi = (-vmax, vmax) if symmetric else (vmin, vmax)

    tmp = out_png + ".msh"
    m = mesh_io.Msh()
    m.nodes = mesh_io.Nodes(V)
    m.elm = mesh_io.Elements(triangles=(F + 1).astype(np.int32))
    m.add_node_field(np.asarray(values, float), "field")
    mesh_io.write_msh(m, tmp)

    gmsh.initialize()
    try:
        gmsh.open(tmp)
        nv = int(gmsh.option.getNumber("PostProcessing.NbViews"))
        vi = next(i for i in range(nv) if "field" in gmsh.option.getString(f"View[{i}].Name").lower())
        cobj = mpl_colormaps[cmap]
        rgb = np.clip(np.round(np.array([cobj(t)[:3] for t in np.linspace(0, 1, 256)]) * 255), 0, 255).astype(int)
        optf = out_png + ".opt"
        with open(optf, "w") as fh:
            fh.write("View[%d].ColorTable = {\n%s\n};\n" % (vi, ",\n".join("{%d,%d,%d}" % tuple(c) for c in rgb)))
        gmsh.merge(optf)
        for i in range(nv):
            gmsh.option.setNumber(f"View[{i}].Visible", 1 if i == vi else 0)
        for k, v in {"RangeType": 2, "CustomMin": lo, "CustomMax": hi, "ShowScale": 0,
                     "Light": 1, "SmoothNormals": SMOOTH_NORMALS, "Visible": 1}.items():
            gmsh.option.setNumber(f"View[{vi}].{k}", v)
        for o in ("Mesh.SurfaceFaces", "Mesh.SurfaceEdges", "Mesh.Lines", "Mesh.Points",
                  "General.SmallAxes", "General.Axes"):
            gmsh.option.setNumber(o, 0)
        # quality: anti-aliasing + two-sided lighting; the high render size is supersampled down on compose.
        # raking directional light + near-zero specular maximizes gyral/sulcal relief (fold contrast).
        gmsh.option.setNumber("General.Antialiasing", 1)
        gmsh.option.setNumber("Mesh.LightTwoSide", 1)
        gmsh.option.setNumber("General.Shininess", LIGHT_SHININESS)
        gmsh.option.setNumber("General.Light0X", LIGHT0[0])
        gmsh.option.setNumber("General.Light0Y", LIGHT0[1])
        gmsh.option.setNumber("General.Light0Z", LIGHT0[2])
        if LIGHT1 is not None:
            gmsh.option.setNumber("General.Light1", 1)
            gmsh.option.setNumber("General.Light1X", LIGHT1[0])
            gmsh.option.setNumber("General.Light1Y", LIGHT1[1])
            gmsh.option.setNumber("General.Light1Z", LIGHT1[2])
            if LIGHT1_DIM:
                try:
                    gmsh.option.setColor("General.Color.Light1", 95, 95, 95)
                except Exception:
                    pass
        gmsh.option.setColor("General.Background", 255, 255, 255)
        gmsh.option.setNumber("General.Trackball", 0)
        rx, ry, rz = rot if rot is not None else VIEWS[view]
        gmsh.option.setNumber("General.RotationX", rx)
        gmsh.option.setNumber("General.RotationY", ry)
        gmsh.option.setNumber("General.RotationZ", rz)
        gmsh.option.setNumber("General.GraphicsWidth", width)
        gmsh.option.setNumber("General.GraphicsHeight", height)
        gmsh.fltk.initialize()
        gmsh.graphics.draw()
        gmsh.write(out_png)
    finally:
        gmsh.finalize()
        for p in (tmp, out_png + ".opt"):
            if os.path.exists(p):
                os.remove(p)
    return lo, hi


def _main():
    """One-shot CLI worker: render ONE panel in its own process (gmsh init/finalize once, no segfault).
    Usage: _surfrender.py --m2m DIR --out PNG --view left|right --cmap NAME --vmax F [--symmetric]
                          --nifti A.nii.gz [B.nii.gz]   (two -> A minus B difference)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2m", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="left")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--vmax", type=float, required=True)
    ap.add_argument("--symmetric", action="store_true")
    ap.add_argument("--rot", nargs=3, type=float, default=None, help="rx ry rz camera override")
    ap.add_argument("--light", nargs=3, type=float, default=None, help="Light0 (key) x y z override")
    ap.add_argument("--light2", nargs=3, type=float, default=None, help="Light1 (fill) x y z; off if absent")
    ap.add_argument("--dimfill", action="store_true", help="soften the fill light")
    ap.add_argument("--shininess", type=float, default=None)
    ap.add_argument("--nifti", nargs="+", required=True)
    a = ap.parse_args()
    global LIGHT0, LIGHT1, LIGHT1_DIM, LIGHT_SHININESS
    if a.light:
        LIGHT0 = tuple(a.light)
    if a.light2:
        LIGHT1 = tuple(a.light2)
    LIGHT1_DIM = a.dimfill
    if a.shininess is not None:
        LIGHT_SHININESS = a.shininess
    V, F = load_central_surface(a.m2m)
    vals = sample_volume_to_surface(a.nifti[0], V)
    if len(a.nifti) == 2:
        vals = vals - sample_volume_to_surface(a.nifti[1], V)
    render_surface(V, F, vals, a.out, cmap=a.cmap, vmin=0.0, vmax=a.vmax,
                   symmetric=a.symmetric, view=a.view, rot=(tuple(a.rot) if a.rot else None))
    print("rendered", a.out)


if __name__ == "__main__":
    _main()
