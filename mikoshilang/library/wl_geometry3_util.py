"""Shared helpers for Wolfram batch X06 (geometry III): argument checks, option handling and conversion between
mesh / polyhedron expressions and NumPy arrays.  No symbols are registered here."""
from __future__ import annotations

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import Unevaluated, L, is_numeric_value
from .wl_graphics_color import K_input
from .wl_graphics_core import split_trailing_options, opt_name
from .wl_geomprims import sx, num, concrete
from . import wl_regions_mesh as M

# presentation options of MeshRegion that mesh constructors pass through unchanged
MESH_PRESENTATION = {"MeshCellHighlight", "MeshCellLabel", "MeshCellMarker", "MeshCellShapeFunction",
                     "MeshCellStyle", "PlotTheme", "PlotRange", "PlotRangePadding", "ImageSize", "Axes", "Frame",
                     "Background", "BaseStyle", "Boxed", "BoxRatios", "Lighting", "ViewPoint", "AspectRatio",
                     "Epilog", "Prolog", "PlotLabel", "AxesLabel", "Ticks", "FrameLabel", "SphericalRegion",
                     "ViewAngle", "ViewVertical", "ImagePadding", "ContentSelectable", "Method"}


def is_sym(x, name):
    return isinstance(x, Symbol) and x.name == name


def int_arg(x, f, what, lo=0, hi=None):
    """Integer argument with bounds: TypeError for a wrong kind, Unevaluated if symbolic, ValueError if out of
    range or not an integer."""
    if isinstance(x, (bool, str)) or K.is_list(x) or (isinstance(x, Expr) and x.head in ("Rule", "Association")):
        raise TypeError(f"{f}: {what} must be an integer, not {K_input(x)}")
    if K.is_int(x):
        v = int(x)
        if v < lo or (hi is not None and v > hi):
            rng = f">= {lo}" if hi is None else f"between {lo} and {hi}"
            raise ValueError(f"{f}: {what} must be an integer {rng}, not {v}")
        return v
    if is_numeric_value(x):
        raise ValueError(f"{f}: {what} must be an integer, not {K_input(x)}")
    raise Unevaluated()


def real_arg(x, f, what, lo=None, hi=None, lo_open=False, hi_open=False):
    """Concrete real -> float; symbolic -> Unevaluated; bad kind -> TypeError; out of range -> ValueError."""
    v = sx(x, f, what)
    concrete(v)
    r = num(v)
    bad = (lo is not None and (r < lo or (lo_open and r == lo))) or \
          (hi is not None and (r > hi or (hi_open and r == hi)))
    if bad:
        lo_s = "" if lo is None else (f"> {lo}" if lo_open else f">= {lo}")
        hi_s = "" if hi is None else (f"< {hi}" if hi_open else f"<= {hi}")
        raise ValueError(f"{f}: {what} must be {' and '.join(s for s in (lo_s, hi_s) if s)}, not {K_input(x)}")
    return r


def options(args, f, allowed, passthrough=()):
    """(positional args, {name: value}, passthrough option rules).  Unknown option names raise ValueError."""
    pos, opts = split_trailing_options(list(args))
    d, keep = {}, []
    for r in opts:
        n = opt_name(r)
        if n in allowed:
            d.setdefault(n, r.args[1])
        elif n in passthrough:
            keep.append(r)
        else:
            raise ValueError(f"{f}: {n} is not a known option of {f}")
    return pos, d, keep


def is_auto(v):
    return v is None or is_sym(v, "Automatic")


def fl(x):
    return float(num(x))


def mesh_np(x, f):
    """(Mesh, float coordinate array) of a MeshRegion / BoundaryMeshRegion expression."""
    import numpy as np
    m = M.mesh_of(x, f)
    P = np.array([[num(c) for c in p] for p in m.P], dtype=float)
    return m, P


def fnum(v):
    """Python float of a NumPy scalar, with -0.0 normalised to 0.0."""
    v = float(v)
    return 0.0 if v == 0 else v


def coords_out(P):
    return L(*[L(*[fnum(c) for c in p]) for p in P])


def mesh_out(P, cells, head="MeshRegion", opts=()):
    """MeshRegion / BoundaryMeshRegion expression from float coordinates and [(cell head, 0-based idx)]."""
    return Expr(head, coords_out(P), L(*[Expr(h, L(*[int(i) + 1 for i in idx])) for h, idx in cells]), *opts)


def compact(P, cells):
    """Drop unused coordinates (keeping their order) and re-index the cells."""
    used = sorted({int(i) for _, idx in cells for i in idx})
    pos = {k: j for j, k in enumerate(used)}
    return [P[k] for k in used], [(h, tuple(pos[int(i)] for i in idx)) for h, idx in cells]


def dedupe_points(P, tol=1e-12):
    """(unique points in order of first appearance, index map) for float points (relative tolerance)."""
    import numpy as np
    P = np.asarray(P, dtype=float)
    scale = max(1.0, float(np.abs(P).max())) if P.size else 1.0
    key = {}
    U, idx = [], []
    q = 1.0 / (tol * scale)
    for p in P:
        k = tuple(int(round(c * q)) for c in p)
        if k not in key:
            key[k] = len(U)
            U.append(p)
        idx.append(key[k])
    return np.array(U), idx


def require_scipy(f):
    try:
        import scipy  # noqa: F401
    except ImportError:
        raise NotImplementedError(f"{f}: requires SciPy (pip install scipy)")


def surface_tris(m, f, P=None):
    """Triangles (0-based index triples) of the 2-cells of a surface mesh in 3-D (polygons are fan-split)."""
    tris = []
    for h, idx in m.cells:
        d = M.CELL_DIM[h]
        if d != 2:
            raise NotImplementedError(f"{f}: only meshes of triangles and polygons are supported here (found "
                                      f"{h} cells)")
        for k in range(1, len(idx) - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))
    return tris


def require_points(x, f, dims=(2, 3), lo=1):
    """Float array of a point list {p1, ...} or Point[{...}] (Point[pts, VertexNormals -> ...] gives normals)."""
    import numpy as np
    from .wl_geomprims import pts
    normals = None
    if isinstance(x, Expr) and x.head == "Point" and len(x.args) >= 1:
        a, o = split_trailing_options(list(x.args))
        for r in o:
            if opt_name(r) == "VertexNormals":
                normals = r.args[1]
            else:
                raise ValueError(f"{f}: {opt_name(r)} is not supported in Point[...] here")
        if len(a) != 1:
            raise ValueError(f"{f}: expected Point[{{p1, p2, ...}}]")
        x = a[0] if K.is_list(a[0]) and a[0].args and K.is_list(a[0].args[0]) else L(a[0])
    P = pts(x, f, lo=lo)
    concrete(P)
    if len(P[0]) not in dims:
        raise ValueError(f"{f}: the points must be {' or '.join(map(str, dims))}-dimensional")
    A = np.array([[num(c) for c in p] for p in P], dtype=float)
    N = None
    if normals is not None:
        N = normals_arg(normals, len(A), A.shape[1], f)
    return A, N


def normals_arg(v, n, d, f):
    import numpy as np
    from .wl_geomprims import pts
    Nn = pts(v, f, "vertex normals")
    concrete(Nn)
    if len(Nn) != n or len(Nn[0]) != d:
        raise ValueError(f"{f}: VertexNormals must give one {d}-dimensional normal per point")
    N = np.array([[num(c) for c in p] for p in Nn], dtype=float)
    nn = np.linalg.norm(N, axis=1)
    if np.any(nn == 0):
        raise ValueError(f"{f}: vertex normals must be nonzero")
    return N / nn[:, None]
