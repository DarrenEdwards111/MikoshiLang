"""Wolfram batch W19: Geometry I -- primitives, special regions, splines, polyhedra and synthetic geometry.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).

The geometric primitives (``Point``, ``Line``, ``Polygon``, ``Disk``, ``Ball``, ``Cuboid``, ``Polyhedron``, ...)
are inert symbolic expressions, as in Wolfram: ``Disk[{0, 0}, 2]`` evaluates to itself after its arguments have
been validated (a string coordinate raises ``TypeError``, a negative radius ``ValueError``; symbolic coordinates
and radii are allowed, so the same expression works as a symbolic region).  They are registered with the W14
graphics core (``register_primitive``) so ``Graphics`` / ``Graphics3D`` check that a primitive matches the
container's dimension, and they are understood as regions by the computational heads of this batch
(``RandomPoint``, ``Circumsphere``, ``PolygonAngle``, ``ConvexPolyhedronQ``, ...).

Extension points for later batches (W22 regions)
------------------------------------------------
* ``polygon_data(expr, fname)`` -> list of vertex lists (SymPy) of a ``Polygon`` / ``Triangle`` / ``Rectangle`` /
  ``RegularPolygon`` / ``Parallelogram`` (2-D).
* ``polyhedron_data(expr, fname)`` -> ``(verts, faces)`` of ``Polyhedron`` / ``Tetrahedron`` / ``Cube`` /
  ``Cuboid`` / ``Hexahedron`` / ``Prism`` / ``Pyramid`` / ``Octahedron`` / ``Dodecahedron`` / ``Icosahedron``
  (faces are 0-based index lists, outward-oriented when the input is).
* ``region_sampler(expr, fname)`` -> ``(dim, sample(rng, n))`` uniform sampler used by ``RandomPoint``.
"""
from __future__ import annotations

import itertools
import math
import random as _random

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, R, has_float
from .wl_graphics_color import K_input
from .wl_graphics_core import register_primitive, opt_name, split_trailing_options, OPTION_VALIDATORS

AREA = "geometry: primitives & synthetic geometry (W19)"

_GRAPHICS_SCALED = ("Scaled", "ImageScaled", "Offset", "Dynamic")


# ───────────────────────────── conversion helpers ─────────────────────────────

def _bad_kind(x):
    return isinstance(x, (str, bool)) or (isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed", "Association"))


def sx(x, fname, what="argument"):
    """SymPy value of a scalar argument (may be symbolic). Strings, booleans, lists and rules raise TypeError."""
    if _bad_kind(x) or K.is_list(x):
        raise TypeError(f"{fname}: {what} must be a number or symbolic expression, not {K_input(x)}")
    if isinstance(x, Expr) and x.head in _GRAPHICS_SCALED:
        raise TypeError(f"{fname}: {what} must be an explicit number here, not {K_input(x)}")
    try:
        v = sp.sympify(K.to_sp(x))
    except Exception:
        raise TypeError(f"{fname}: {what} {K_input(x)} is not a number")
    if not isinstance(v, sp.Expr):
        raise TypeError(f"{fname}: {what} {K_input(x)} is not a number")
    if v.is_number and v.is_real is False:
        if v.is_finite is False or v.has(sp.zoo, sp.nan):
            raise ValueError(f"{fname}: {what} {K_input(x)} must be finite")
        raise ValueError(f"{fname}: {what} {K_input(x)} must be real")
    if v.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
        raise ValueError(f"{fname}: {what} {K_input(x)} must be finite")
    return v


def is_symbolic(v):
    return isinstance(v, sp.Basic) and bool(v.free_symbols)


def concrete(*vals):
    """Raise Unevaluated if any of the SymPy values (or nested lists of them) is symbolic."""
    for v in vals:
        if isinstance(v, (list, tuple)):
            concrete(*v)
        elif is_symbolic(v):
            raise Unevaluated()


def pt(x, fname, what="point", dim=None, point_head=True):
    """A coordinate list {x1, ..., xn} (or Point[{...}] when point_head) -> list of SymPy values."""
    if point_head and isinstance(x, Expr) and x.head == "Point" and len(x.args) == 1:
        x = x.args[0]
    if not K.is_list(x):
        if isinstance(x, Symbol) or (isinstance(x, Expr) and not _bad_kind(x) and not isinstance(x.head, str)):
            raise Unevaluated()
        if isinstance(x, Symbol) or (isinstance(x, Expr) and x.head not in ("Rule", "RuleDelayed", "Association")
                                     and not _numberish(x)):
            raise Unevaluated()
        raise TypeError(f"{fname}: {what} must be a list of coordinates, not {K_input(x)}")
    if not x.args:
        raise ValueError(f"{fname}: {what} must have at least one coordinate")
    if dim is not None and len(x.args) not in ((dim,) if isinstance(dim, int) else dim):
        want = dim if isinstance(dim, int) else " or ".join(map(str, dim))
        raise ValueError(f"{fname}: {what} {K_input(x)} must have {want} coordinates")
    return [sx(c, fname, what + " coordinate") for c in x.args]


def _numberish(x):
    try:
        v = sp.sympify(K.to_sp(x))
        return isinstance(v, sp.Expr) and v.is_number
    except Exception:
        return False


def pts(x, fname, what="points", n=None, dim=None, lo=1):
    """A list of points {p1, p2, ...} (Point[...] entries allowed) -> list of coordinate lists, same dimension."""
    if isinstance(x, Expr) and x.head == "Point" and len(x.args) == 1 and K.is_list(x.args[0]) and x.args[0].args \
            and K.is_list(x.args[0].args[0]):
        x = x.args[0]
    if not K.is_list(x):
        if isinstance(x, Symbol) or (isinstance(x, Expr) and not _bad_kind(x) and not _numberish(x)):
            raise Unevaluated()
        raise TypeError(f"{fname}: {what} must be a list of points, not {K_input(x)}")
    out = [pt(p, fname, "point", dim) for p in x.args]
    if n is not None and len(out) not in ((n,) if isinstance(n, int) else n):
        want = n if isinstance(n, int) else " or ".join(map(str, n))
        raise ValueError(f"{fname}: expected {want} points, got {len(out)}")
    if len(out) < lo:
        raise ValueError(f"{fname}: expected at least {lo} point(s), got {len(out)}")
    if len({len(p) for p in out}) > 1:
        raise ValueError(f"{fname}: all points must have the same dimension")
    return out


def inexact(*inputs):
    return any(has_float(i) for i in inputs)


def out(v, *inputs, simplify=True):
    """SymPy value (or nested list) -> MikoshiLang; machine numbers when any input was inexact."""
    flt = inexact(*inputs)

    def conv(a):
        if isinstance(a, (list, tuple)):
            return L(*[conv(b) for b in a])
        if isinstance(a, bool):
            return a
        if isinstance(a, Expr):
            return a
        a = sp.sympify(a)
        if flt:
            v_ = a.evalf(15)
            if v_.is_number:
                c = complex(v_)
                if abs(c.imag) > 1e-12 * max(1.0, abs(c.real)):
                    return R(v_)
                return float(c.real)
            return R(v_)
        if simplify and a.free_symbols == set() and not a.is_Rational:
            a = _simp(a)
        return R(a)
    return conv(v)


def _simp(a):
    try:
        b = sp.nsimplify(a) if a.is_Float else a
        c = sp.radsimp(sp.simplify(b))
        return c if sp.count_ops(c) <= sp.count_ops(b) else b
    except Exception:
        return a


def num(v):
    """Float value of a concrete real SymPy number."""
    return float(sp.N(v, 30))


def sign_of(v, fname, what="value"):
    """Exact sign (-1, 0, 1) of a concrete real expression; NotImplementedError if undecidable."""
    v = sp.sympify(v)
    if v.is_zero:
        return 0
    if v.is_positive:
        return 1
    if v.is_negative:
        return -1
    vs = sp.simplify(v)
    if vs.is_zero:
        return 0
    try:
        f = sp.N(vs, 50)
    except Exception:
        raise NotImplementedError(f"{fname}: cannot decide the sign of {what}")
    if not f.is_number or f.has(sp.I):
        raise NotImplementedError(f"{fname}: cannot decide the sign of {what}")
    if abs(f) < sp.Float(10) ** -40:
        if vs.has(sp.Float):
            return 0
        if sp.nsimplify(vs) == 0 or sp.minimal_polynomial(vs, sp.Symbol("_z")).as_expr() == sp.Symbol("_z"):
            return 0
    return 1 if f > 0 else -1


def fsign(v, tol=1e-12):
    """Sign of a value that may be inexact: exact sign for exact input, tolerance otherwise."""
    v = sp.sympify(v)
    if v.has(sp.Float):
        f = num(v)
        return 0 if abs(f) <= tol else (1 if f > 0 else -1)
    return sign_of(v, "sign")


def _string_arg(x, choices, fname, what="specification"):
    if isinstance(x, str):
        if x in choices:
            return x
        raise ValueError(f"{fname}: {what} {x!r} must be one of {', '.join(repr(c) for c in choices)}")
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {what} must be a string such as {choices[0]!r}, not {K_input(x)}")


# ───────────────────────────── primitive validation ─────────────────────────────
# Each checker validates the arguments of one head and returns the set of coordinate dimensions it uses
# (empty when unknown/implicit).  Checkers never raise Unevaluated: symbolic coordinates are fine in an inert
# primitive.  Option rules at the end are split off by the caller.

def _coord(x, fname, what="point"):
    """Validate one coordinate list (symbolic entries allowed); Scaled/Offset forms accepted. -> dim or None."""
    if isinstance(x, Expr) and x.head in _GRAPHICS_SCALED:
        return None
    if not K.is_list(x):
        if isinstance(x, Symbol) or (isinstance(x, Expr) and not _bad_kind(x) and not _numberish(x)):
            return None                                   # symbolic point p
        raise TypeError(f"{fname}: {what} must be a list of coordinates, not {K_input(x)}")
    if not x.args:
        raise ValueError(f"{fname}: {what} must have at least one coordinate")
    for c in x.args:
        if K.is_list(c) or _bad_kind(c) or (isinstance(c, Expr) and c.head in _GRAPHICS_SCALED):
            if isinstance(c, Expr) and c.head in _GRAPHICS_SCALED:
                continue
            raise TypeError(f"{fname}: {what} coordinate {K_input(c)} must be a number")
        sx(c, fname, what + " coordinate")
    return len(x.args)


def _is_coord_list(x):
    """True for {{..}, {..}, ...} (a list whose first element is itself a list or a Scaled/Offset form)."""
    return K.is_list(x) and bool(x.args) and (K.is_list(x.args[0]) or (isinstance(x.args[0], Expr) and
                                                                      x.args[0].head in _GRAPHICS_SCALED))


def _coords(x, fname, what="points", n=None, lo=0, nested=False):
    """Validate a list of points (or, with nested, a list of such lists). -> set of dims."""
    if not K.is_list(x):
        if isinstance(x, Symbol) or (isinstance(x, Expr) and not _bad_kind(x) and not _numberish(x)):
            return set()
        raise TypeError(f"{fname}: {what} must be a list of points, not {K_input(x)}")
    if nested and x.args and K.is_list(x.args[0]) and _is_coord_list(x.args[0]):
        dims = set()
        for part in x.args:
            dims |= _coords(part, fname, what, n, lo)
        return dims
    if n is not None and len(x.args) != n:
        raise ValueError(f"{fname}: expected {n} points, got {len(x.args)}")
    if len(x.args) < lo:
        raise ValueError(f"{fname}: expected at least {lo} points, got {len(x.args)}")
    dims = {d for d in (_coord(p, fname, "point") for p in x.args) if d is not None}
    if len(dims) > 1:
        raise ValueError(f"{fname}: all points must have the same dimension")
    return dims


def _nonneg(x, fname, what, strict=False):
    """A radius-like value: non-negative (positive with strict) when numeric; symbolic allowed."""
    if isinstance(x, Expr) and x.head in _GRAPHICS_SCALED:
        return
    v = sx(x, fname, what)
    if v.is_number:
        s = num(v)
        if s < 0 or (strict and s == 0):
            raise ValueError(f"{fname}: {what} must be {'positive' if strict else 'non-negative'}, not {K_input(x)}")
    return v


def _radii(x, fname, what="radius", k=None):
    """r or {r1, r2, ...}: non-negative values. -> number of components or None."""
    if K.is_list(x):
        if k is not None and len(x.args) != k:
            raise ValueError(f"{fname}: {what} must be a number or a list of {k} numbers")
        for c in x.args:
            _nonneg(c, fname, what)
        return len(x.args)
    _nonneg(x, fname, what)
    return None


def _angles(x, fname, what="angle range"):
    if not K.is_list(x) or len(x.args) != 2:
        if isinstance(x, Symbol):
            return
        raise TypeError(f"{fname}: {what} must be a list {{theta1, theta2}}, not {K_input(x)}")
    for c in x.args:
        sx(c, fname, "angle")


def _nargs(args, lo, hi, fname):
    if not lo <= len(args) <= hi:
        want = str(lo) if lo == hi else f"{lo} to {hi}"
        raise ValueError(f"{fname}: expected {want} argument(s), got {len(args)}")


def _pos_int(x, fname, what="dimension"):
    if isinstance(x, bool) or isinstance(x, str) or K.is_list(x):
        raise TypeError(f"{fname}: {what} must be a positive integer, not {K_input(x)}")
    if K.is_int(x):
        if int(x) < 1:
            raise ValueError(f"{fname}: {what} must be a positive integer, not {x}")
        return int(x)
    if _numberish(x):
        raise ValueError(f"{fname}: {what} must be a positive integer, not {K_input(x)}")
    return None


def _ck_point(a, f):
    _nargs(a, 1, 1, f)
    x = a[0]
    if _is_coord_list(x):
        return _coords(x, f)
    d = _coord(x, f)
    return {d} if d else set()


def _ck_line(a, f, lo=0):
    _nargs(a, 1, 1, f)
    x = a[0]
    if isinstance(x, Expr) and x.head in ("Line", "BezierCurve", "BSplineCurve", "JoinedCurve") and f in (
            "Arrow", "Tube", "HalfLine", "InfiniteLine"):
        return _check_args(x.head, list(x.args))
    return _coords(x, f, nested=True, lo=lo)


def _ck_polygon(a, f):
    _nargs(a, 1, 2, f)
    x = a[0]
    if len(a) == 2:                                   # Polygon[{p1, ...}, data]: integer indices into points
        dims = _coords(x, f)
        _index_data(a[1], len(x.args) if K.is_list(x) else None, f)
        return dims
    if isinstance(x, Expr) and x.head == "Rule" and len(x.args) == 2:  # outer -> {holes}
        dims = _coords(x.args[0], f)
        holes = x.args[1]
        if not K.is_list(holes):
            raise TypeError(f"{f}: the holes in outer -> holes must be a list of point lists")
        hs = holes.args if (holes.args and _is_coord_list(holes.args[0])) else ([holes] if holes.args else [])
        for h in hs:
            dims |= _coords(h, f)
        if len(dims) > 1:
            raise ValueError(f"{f}: all points must have the same dimension")
        return dims
    if K.is_list(x) and x.args and (K.is_list(x.args[0]) and _is_coord_list(x.args[0]) or
                                   (isinstance(x.args[0], Expr) and x.args[0].head == "Rule")):
        dims = set()
        for part in x.args:
            dims |= _ck_polygon([part], f)
        if len(dims) > 1:
            raise ValueError(f"{f}: all points must have the same dimension")
        return dims
    return _coords(x, f)


def _index_data(d, n, f):
    if K.is_list(d):
        for v in d.args:
            _index_data(v, n, f)
        return
    if isinstance(d, Expr) and d.head == "Rule":
        for v in d.args:
            _index_data(v, n, f)
        return
    if K.is_int(d) and not isinstance(d, bool):
        if n is not None and not 1 <= int(d) <= n:
            raise ValueError(f"{f}: vertex index {d} is out of range 1..{n}")
        return
    raise TypeError(f"{f}: the face data must contain integer vertex indices, not {K_input(d)}")


def _ck_round(a, f, dim):
    """Circle / Disk / DiskSegment-like: [c, r | {rx, ry}, {t1, t2}]."""
    _nargs(a, 0, 3, f)
    dims = set()
    if a:
        c = a[0]
        dims = _coords(c, f) if _is_coord_list(c) else ({_coord(c, f, "centre")} - {None})
    if len(a) > 1:
        _radii(a[1], f, "radius", dim)
    if len(a) > 2:
        _angles(a[2], f)
    if dims and dims != {dim}:
        raise ValueError(f"{f}: the centre must be a {dim}-dimensional point")
    return dims or {dim}


def _ck_ball(a, f):
    _nargs(a, 0, 2, f)
    if not a:
        return {3}
    c = a[0]
    if K.is_int(c) and not isinstance(c, bool):
        _pos_int(c, f)
        if len(a) > 1:
            raise ValueError(f"{f}: the form {f}[n] takes no radius")
        return {int(c)}
    dims = _coords(c, f) if _is_coord_list(c) else ({_coord(c, f, "centre")} - {None})
    if len(a) > 1:
        _nonneg(a[1], f, "radius")
    return dims


def _ck_regular_solid(a, f):
    """Cube / Tetrahedron / Octahedron / Dodecahedron / Icosahedron: [], [l], [{t, p}, l], [c, ...]; Tetrahedron
    also [{p1, p2, p3, p4}] and a list of such corner lists."""
    _nargs(a, 0, 3, f)
    rest = list(a)
    if f == "Tetrahedron" and len(rest) == 1 and _is_coord_list(rest[0]):
        if len(rest[0].args) == 4 or not _is_coord_list(rest[0].args[0]):
            dims = _coords(rest[0], f, n=4)
        else:
            dims = set()
        if _is_coord_list(rest[0].args[0]):
            for t in rest[0].args:
                dims |= _coords(t, f, n=4)
        if dims - {3}:
            raise ValueError(f"{f}: the corners must be three-dimensional points")
        return {3}
    if rest and K.is_list(rest[0]) and len(rest[0].args) == 3:
        _coord(rest[0], f, "centre")
        rest = rest[1:]
    if rest and K.is_list(rest[0]):
        if len(rest[0].args) != 2:
            raise ValueError(f"{f}: the orientation must be a pair of angles {{theta, phi}}")
        _angles(rest[0], f, "orientation")
        rest = rest[1:]
    if len(rest) > 1:
        raise ValueError(f"{f}: invalid specification {K_input(Expr(f, *a))}")
    if rest:
        _nonneg(rest[0], f, "edge length", strict=True)
    return {3}


def _bad_dim(f):
    raise ValueError(f"{f}: the points must be three-dimensional")


def _ck_fixed_pts(n, dim=None):
    def ck(a, f):
        _nargs(a, 1, 1, f)
        x = a[0]
        if K.is_list(x) and x.args and K.is_list(x.args[0]) and x.args[0].args and _is_coord_list(x.args[0]):
            dims = set()
            for part in x.args:
                dims |= _coords(part, f, n=n)
        else:
            dims = _coords(x, f, n=n)
        if dim is not None and dims - {dim}:
            raise ValueError(f"{f}: the points must be {dim}-dimensional")
        return dims
    return ck


def _ck_axis_solid(a, f, default=True):
    """Cone / Cylinder / CapsuleShape / StadiumShape: [{p1, p2}, r] (or [r] / [])."""
    _nargs(a, 0, 2, f)
    if not a:
        return set()
    if len(a) == 1 and not K.is_list(a[0]):
        if f in ("CapsuleShape", "StadiumShape"):
            _nonneg(a[0], f, "radius")
            return set()
        raise TypeError(f"{f}: the axis must be a list of two points, not {K_input(a[0])}")
    dims = _coords(a[0], f, n=2)
    if len(a) > 1:
        _nonneg(a[1], f, "radius")
    return dims


def _ck_rect(a, f, dim):
    _nargs(a, 0, 2, f)
    dims = set()
    for i, p in enumerate(a):
        d = _coord(p, f, "corner")
        if d:
            dims.add(d)
    if len(dims) > 1:
        raise ValueError(f"{f}: the corners must have the same dimension")
    if dim and dims and dims != {dim}:
        raise ValueError(f"{f}: the corners must be {dim}-dimensional points")
    return dims


def _ck_parallelotope(a, f, k=None):
    _nargs(a, 0, 2, f)
    if not a:
        return {2} if f == "Parallelogram" else set()
    if len(a) != 2:
        raise ValueError(f"{f}: expected {f}[p, {{v1, v2, ...}}]")
    d = _coord(a[0], f, "origin")
    vd = _coords(a[1], f, "direction vectors", n=k)
    dims = ({d} if d else set()) | vd
    if len(dims) > 1:
        raise ValueError(f"{f}: the origin and the direction vectors must have the same dimension")
    return dims


def _ck_simplex(a, f):
    _nargs(a, 1, 1, f)
    x = a[0]
    if K.is_int(x) and not isinstance(x, bool):
        n = _pos_int(x, f, "dimension")
        return {n}
    if K.is_list(x) and x.args and _is_coord_list(x.args[0]) if K.is_list(x) else False:
        dims = set()
        for s in x.args:
            dims |= _ck_simplex([s], f)
        return dims
    dims = _coords(x, f, lo=1)
    if dims and K.is_list(x) and len(x.args) > list(dims)[0] + 1:
        raise ValueError(f"{f}: a simplex in {list(dims)[0]} dimensions has at most {list(dims)[0] + 1} points")
    return dims


def _ck_polyhedron(a, f):
    _nargs(a, 1, 2, f)
    if len(a) == 2:
        dims = _coords(a[0], f)
        n = len(a[0].args) if K.is_list(a[0]) else None
        _index_data(a[1], n, f)
        if dims - {3}:
            raise ValueError(f"{f}: the vertex coordinates must be three-dimensional")
        return {3}
    x = a[0]
    if isinstance(x, Expr) and x.head == "Rule":
        for part in x.args:
            _ck_polyhedron([part], f)
        return {3}
    if not K.is_list(x):
        if isinstance(x, Symbol):
            return {3}
        raise TypeError(f"{f}: the faces must be a list of polygons, not {K_input(x)}")
    if x.args and (K.is_list(x.args[0]) and x.args[0].args and _is_coord_list(x.args[0].args[0])
                   or (isinstance(x.args[0], Expr) and x.args[0].head == "Rule")):
        for part in x.args:                            # collection of polyhedra
            _ck_polyhedron([part], f)
        return {3}
    for face in x.args:
        if _coords(face, f, "face", lo=3) - {3}:
            raise ValueError(f"{f}: the faces must have three-dimensional vertices")
    return {3}


def _ck_spline_opts(opts, f, surface=False):
    for r in opts:
        n = opt_name(r)
        if n not in ("SplineDegree", "SplineKnots", "SplineWeights", "SplineClosed", "VertexColors",
                     "VertexNormals", "VertexTextureCoordinates"):
            raise ValueError(f"{f}: {n} is not a known option of {f}")
        v = r.args[1]
        if n == "SplineDegree":
            vals = v.args if (surface and K.is_list(v)) else [v]
            for d in vals:
                if not (isinstance(d, Symbol) and d.name == "Automatic"):
                    if not K.is_int(d) or isinstance(d, bool) or int(d) < 1:
                        raise ValueError(f"{f}: SplineDegree must be a positive integer or Automatic, "
                                         f"not {K_input(d)}")
        elif n == "SplineClosed":
            vals = v.args if (surface and K.is_list(v)) else [v]
            for d in vals:
                if d is not True and d is not False:
                    raise ValueError(f"{f}: SplineClosed must be True or False, not {K_input(d)}")


def _ck_bezier(a, f):
    pts_, opts = split_trailing_options(a)
    _nargs(pts_, 1, 1, f)
    _ck_spline_opts(opts, f)
    return _coords(pts_[0], f, nested=False, lo=0)


def _ck_bspline_curve(a, f):
    pts_, opts = split_trailing_options(a)
    _nargs(pts_, 1, 1, f)
    _ck_spline_opts(opts, f)
    return _coords(pts_[0], f, lo=0)


def _ck_surface(a, f):
    pts_, opts = split_trailing_options(a)
    _nargs(pts_, 1, 1, f)
    _ck_spline_opts(opts, f, surface=True)
    x = pts_[0]
    if not K.is_list(x) or not all(K.is_list(r) for r in x.args):
        raise TypeError(f"{f}: the control points must be an array {{{{p11, p12, ...}}, ...}}")
    if len({len(r.args) for r in x.args}) > 1:
        raise ValueError(f"{f}: the control point array must be rectangular")
    dims = set()
    for r in x.args:
        dims |= _coords(r, f)
    return dims


def _ck_curve_list(a, f):
    pts_, opts = split_trailing_options(a)
    _nargs(pts_, 1, 1, f)
    for r in opts:
        if opt_name(r) not in ("CurveClosed",):
            raise ValueError(f"{f}: {opt_name(r)} is not a known option of {f}")
    x = pts_[0]
    if not K.is_list(x):
        raise TypeError(f"{f}: expected a list of curve segments, not {K_input(x)}")
    comps = x.args if (x.args and K.is_list(x.args[0])) else [x]
    dims = set()
    for comp in comps:
        if not K.is_list(comp):
            raise TypeError(f"{f}: a component must be a list of curve segments")
        for seg in comp.args:
            if not (isinstance(seg, Expr) and seg.head in ("Line", "BezierCurve", "BSplineCurve")):
                raise TypeError(f"{f}: segments must be Line, BezierCurve or BSplineCurve, not {K_input(seg)}")
            dims |= _check_args(seg.head, list(seg.args))
    return dims


def _ck_tube(a, f):
    _nargs(a, 1, 2, f)
    x = a[0]
    if isinstance(x, Expr) and x.head in ("Line", "BezierCurve", "BSplineCurve"):
        dims = _check_args(x.head, list(x.args))
    else:
        dims = _coords(x, f, nested=True)
    if len(a) > 1:
        r = a[1]
        if K.is_list(r):
            for c in r.args:
                _nonneg(c, f, "radius")
        else:
            _nonneg(r, f, "radius")
    if dims - {3}:
        raise ValueError(f"{f}: the curve must be three-dimensional")
    return {3}


def _ck_arrow(a, f):
    pts_, opts = split_trailing_options(a)
    _nargs(pts_, 1, 2, f)
    x = pts_[0]
    if isinstance(x, Expr) and x.head in ("Line", "BezierCurve", "BSplineCurve", "JoinedCurve", "Tube"):
        dims = _check_args(x.head, list(x.args))
    else:
        dims = _coords(x, f, nested=True)
    if len(pts_) > 1:
        s = pts_[1]
        if K.is_list(s):
            if len(s.args) != 2:
                raise ValueError(f"{f}: the setback must be s or {{s1, s2}}")
            for c in s.args:
                _nonneg(c, f, "setback")
        else:
            _nonneg(s, f, "setback")
    return dims


def _ck_line_through(a, f, mode):
    """HalfLine / InfiniteLine: [{p1, p2}] or [p, v]; InfinitePlane: [{p1, p2, p3}] or [p, {v1, v2}]."""
    _nargs(a, 1, 2, f)
    if len(a) == 1:
        x = a[0]
        if f == "HalfLine" and K.is_list(x) and x.args and _is_coord_list(x.args[0]):
            pass
        n = 3 if mode == "plane" else 2
        if K.is_list(x) and x.args and K.is_list(x.args[0]) and x.args[0].args and _is_coord_list(x.args[0]):
            dims = set()
            for part in x.args:
                dims |= _coords(part, f, n=n)
            return dims
        return _coords(x, f, n=n)
    d = _coord(a[0], f, "point")
    if mode == "plane":
        vd = _coords(a[1], f, "direction vectors", n=2)
    else:
        if _is_coord_list(a[1]):
            raise ValueError(f"{f}: the direction must be a single vector")
        vd = {_coord(a[1], f, "direction")} - {None}
    dims = ({d} if d else set()) | vd
    if len(dims) > 1:
        raise ValueError(f"{f}: the point and direction must have the same dimension")
    return dims


def _ck_halfplane(a, f):
    _nargs(a, 2, 3, f)
    if len(a) == 2:
        dims = _coords(a[0], f, n=2) | ({_coord(a[1], f, "direction")} - {None})
    else:
        dims = {_coord(v, f, w) for v, w in zip(a, ("point", "direction", "direction"))} - {None}
    if len(dims) > 1:
        raise ValueError(f"{f}: the points and directions must have the same dimension")
    return dims


def _ck_normal_form(a, f):
    """HalfSpace / Hyperplane: [n, p] or [n, c] (Hyperplane[n] too)."""
    _nargs(a, 1 if f == "Hyperplane" else 2, 2, f)
    d = _coord(a[0], f, "normal")
    dims = {d} if d else set()
    if len(a) > 1:
        if K.is_list(a[1]):
            d2 = _coord(a[1], f, "point")
            if d2:
                dims.add(d2)
        else:
            sx(a[1], f, "offset")
    if len(dims) > 1:
        raise ValueError(f"{f}: the normal and the point must have the same dimension")
    if K.is_list(a[0]) and all(_numberish(c) for c in a[0].args) and \
            all(sp.sympify(K.to_sp(c)).is_zero for c in a[0].args):
        raise ValueError(f"{f}: the normal vector must be non-zero")
    return dims


def _ck_affine(a, f, halfspace=False, conic=False):
    lo, hi = (1, 3) if conic else ((2, 3) if halfspace else (1, 2))
    _nargs(a, lo, hi, f)
    rest = list(a)
    dims = set()
    if _is_coord_list(rest[0]) or (K.is_list(rest[0]) and not rest[0].args):
        dims |= _coords(rest[0], f, lo=1)
        rest = rest[1:]
        if conic and rest:
            dims |= _coords(rest[0], f, "generator vectors")
            rest = rest[1:]
    else:
        d = _coord(rest[0], f, "point")
        if d:
            dims.add(d)
        if len(rest) < 2:
            raise ValueError(f"{f}: expected {f}[p, {{v1, ...}}{', w' if halfspace else ''}]")
        dims |= _coords(rest[1], f, "direction vectors")
        rest = rest[2:]
        if conic and rest:
            dims |= _coords(rest[0], f, "generator vectors")
            rest = rest[1:]
    if halfspace:
        if len(rest) != 1:
            raise ValueError(f"{f}: missing the extension direction w")
        d = _coord(rest[0], f, "direction")
        if d:
            dims.add(d)
        rest = []
    if rest:
        raise ValueError(f"{f}: too many arguments")
    if len(dims) > 1:
        raise ValueError(f"{f}: all points and vectors must have the same dimension")
    return dims


def _ck_shell(a, f, dim):
    """Annulus / SphericalShell / Torus / FilledTorus: [], [r], [{rin, rout}], [c, {rin, rout}], Annulus also
    [c, {rin, rout}, {t1, t2}]."""
    _nargs(a, 0, 3 if f == "Annulus" else 2, f)
    rest = list(a)
    if len(rest) == 1 and f in ("Annulus", "SphericalShell"):
        x = rest[0]
        if not K.is_list(x):
            _nonneg(x, f, "radius")
            return {dim} if dim else set()
        _shell_radii(x, f)
        return {dim} if dim else {3}
    dims = set()
    if rest:
        d = _coord(rest[0], f, "centre")
        if d:
            dims.add(d)
        rest = rest[1:]
    if rest:
        _shell_radii(rest[0], f)
        rest = rest[1:]
    if rest:
        _angles(rest[0], f)
    if dim and dims and dims != {dim}:
        raise ValueError(f"{f}: the centre must be a {dim}-dimensional point")
    return dims or ({dim} if dim else set())


def _shell_radii(x, f):
    if not K.is_list(x) or len(x.args) != 2:
        raise TypeError(f"{f}: the radii must be a list {{r_inner, r_outer}}, not {K_input(x)}")
    a = _nonneg(x.args[0], f, "inner radius")
    b = _nonneg(x.args[1], f, "outer radius")
    if a is not None and b is not None and a.is_number and b.is_number and num(a) > num(b):
        raise ValueError(f"{f}: the inner radius must not exceed the outer radius")


def _ck_ellipsoid(a, f):
    _nargs(a, 2, 2, f)
    d = _coord(a[0], f, "centre")
    s = a[1]
    if K.is_list(s) and s.args and K.is_list(s.args[0]):     # weight matrix
        n = len(s.args)
        if any(not K.is_list(r) or len(r.args) != n for r in s.args):
            raise ValueError(f"{f}: the weight matrix must be square")
        M = sp.Matrix([[sx(c, f, "matrix entry") for c in r.args] for r in s.args])
        if not M.free_symbols:
            if M != M.T:
                raise ValueError(f"{f}: the weight matrix must be symmetric")
            if not M.is_positive_definite:
                raise ValueError(f"{f}: the weight matrix must be positive definite")
        k = n
    else:
        if not K.is_list(s):
            raise TypeError(f"{f}: the semi-axes must be a list {{r1, r2, ...}} or a matrix")
        for c in s.args:
            _nonneg(c, f, "semi-axis", strict=True)
        k = len(s.args)
    if d and d != k:
        raise ValueError(f"{f}: the centre and the semi-axes must have the same dimension")
    return {k}


def _ck_disk_segment(a, f):
    _nargs(a, 3, 3, f)
    return _ck_round(a, f, 2)


def _ck_regular_polygon(a, f):
    _nargs(a, 1, 3, f)
    n = a[-1]
    if K.is_int(n) and not isinstance(n, bool):
        if int(n) < 3:
            raise ValueError(f"{f}: a regular polygon needs at least 3 vertices")
    elif K.is_list(n) or _bad_kind(n):
        raise TypeError(f"{f}: the number of vertices must be an integer, not {K_input(n)}")
    elif _numberish(n):
        raise ValueError(f"{f}: the number of vertices must be an integer >= 3, not {K_input(n)}")
    rest = a[:-1]
    dims = set()
    if len(rest) == 2:
        d = _coord(rest[0], f, "centre")
        if d and d != 2:
            raise ValueError(f"{f}: the centre must be a 2-dimensional point")
        rest = rest[1:]
    if rest:
        r = rest[0]
        if K.is_list(r):
            if len(r.args) != 2:
                raise ValueError(f"{f}: the radius specification must be r or {{r, theta}}")
            _nonneg(r.args[0], f, "radius", strict=True)
            sx(r.args[1], f, "angle")
        else:
            _nonneg(r, f, "radius", strict=True)
    return dims | {2}


def _ck_uniform(a, f):
    _nargs(a, 1, 3, f)
    s = a[-1]
    if isinstance(s, str):
        pass
    elif K.is_list(s) and len(s.args) == 2:
        pass
    elif K.is_int(s) and not isinstance(s, bool):
        pass
    elif isinstance(s, Symbol):
        pass
    else:
        raise TypeError(f"{f}: the polyhedron must be given by name, Schlafli symbol {{n, m}} or Wenninger number")
    rest = a[:-1]
    for r in rest:
        if not K.is_list(r) or len(r.args) != 3:
            raise ValueError(f"{f}: expected {f}[{{x, y, z}}, {{r, theta, phi}}, name]")
        _coord(r, f, "position")
    return {3}


def _ck_empty_full(a, f):
    _nargs(a, 1, 1, f)
    n = _pos_int(a[0], f)
    return {n} if n else set()


def _ck_polar(a, f):
    _nargs(a, 2, 2, f)
    r, spec = a
    if _bad_kind(r) or K.is_list(r):
        raise TypeError(f"{f}: the radius must be an expression in the angle variable, not {K_input(r)}")
    if K.is_list(spec):
        if len(spec.args) != 3 or not isinstance(spec.args[0], Symbol):
            raise ValueError(f"{f}: the angle range must be {{theta, theta_min, theta_max}}")
        for c in spec.args[1:]:
            sx(c, f, "angle bound")
    elif not isinstance(spec, Symbol):
        raise TypeError(f"{f}: the angle must be a variable or {{theta, theta_min, theta_max}}")
    return {2}


def _ck_cuboid(a, f):
    return _ck_rect(a, f, None)


_CHECKS = {
    "Point": lambda a, f: _ck_point(a, f),
    "Line": lambda a, f: _ck_line(a, f),
    "Polygon": _ck_polygon,
    "Triangle": lambda a, f: (_nargs(a, 0, 1, f), _ck_fixed_pts(3)(a, f) if a else {2})[1],
    "Rectangle": lambda a, f: _ck_rect(a, f, 2),
    "Cuboid": _ck_cuboid,
    "Circle": lambda a, f: _ck_round(a, f, 2),
    "Disk": lambda a, f: _ck_round(a, f, 2),
    "DiskSegment": _ck_disk_segment,
    "Sphere": _ck_ball,
    "Ball": _ck_ball,
    "Ellipsoid": _ck_ellipsoid,
    "Annulus": lambda a, f: _ck_shell(a, f, 2),
    "SphericalShell": lambda a, f: _ck_shell(a, f, None),
    "Torus": lambda a, f: _ck_shell(a, f, 3),
    "FilledTorus": lambda a, f: _ck_shell(a, f, 3),
    "Cube": _ck_regular_solid, "Tetrahedron": _ck_regular_solid, "Octahedron": _ck_regular_solid,
    "Dodecahedron": _ck_regular_solid, "Icosahedron": _ck_regular_solid,
    "Hexahedron": _ck_fixed_pts(8, 3), "Prism": _ck_fixed_pts(6, 3), "Pyramid": _ck_fixed_pts(5, 3),
    "Cone": _ck_axis_solid, "Cylinder": _ck_axis_solid, "CapsuleShape": _ck_axis_solid,
    "StadiumShape": _ck_axis_solid,
    "Parallelogram": lambda a, f: _ck_parallelotope(a, f, 2),
    "Parallelepiped": _ck_parallelotope,
    "Simplex": _ck_simplex,
    "Polyhedron": _ck_polyhedron,
    "UniformPolyhedron": _ck_uniform,
    "RegularPolygon": _ck_regular_polygon,
    "BezierCurve": _ck_bezier,
    "BSplineCurve": _ck_bspline_curve,
    "BSplineSurface": _ck_surface,
    "BezierSurface": _ck_surface,
    "FilledCurve": _ck_curve_list,
    "JoinedCurve": _ck_curve_list,
    "Tube": _ck_tube,
    "Arrow": _ck_arrow,
    "HalfLine": lambda a, f: _ck_line_through(a, f, "line"),
    "InfiniteLine": lambda a, f: _ck_line_through(a, f, "line"),
    "InfinitePlane": lambda a, f: _ck_line_through(a, f, "plane"),
    "HalfPlane": _ck_halfplane,
    "HalfSpace": _ck_normal_form,
    "Hyperplane": _ck_normal_form,
    "AffineSpace": lambda a, f: _ck_affine(a, f),
    "AffineHalfSpace": lambda a, f: _ck_affine(a, f, halfspace=True),
    "ConicHullRegion": lambda a, f: _ck_affine(a, f, conic=True),
    "EmptyRegion": _ck_empty_full,
    "FullRegion": _ck_empty_full,
    "PolarCurve": _ck_polar,
    "FilledPolarCurve": _ck_polar,
}

# Options a primitive accepts (graphics options); anything else after the arguments raises ValueError.
_PRIM_OPTS = {
    "Polygon": ("VertexColors", "VertexNormals", "VertexTextureCoordinates"),
    "Polyhedron": ("VertexColors", "VertexNormals", "VertexTextureCoordinates"),
    "Line": ("VertexColors", "VertexNormals"),
    "Point": ("VertexColors", "VertexNormals"),
    "Rectangle": ("RoundingRadius",),
    "Triangle": ("VertexColors", "VertexNormals", "VertexTextureCoordinates"),
    "Simplex": ("VertexColors", "VertexNormals", "VertexTextureCoordinates"),
    "Tube": ("VertexColors", "VertexNormals"),
}
for _h in ("Tetrahedron", "Hexahedron", "Prism", "Pyramid", "Cube", "Octahedron", "Dodecahedron", "Icosahedron",
           "UniformPolyhedron"):
    _PRIM_OPTS[_h] = ("VertexColors", "VertexNormals", "VertexTextureCoordinates")
_SPLINE_HEADS = ("BezierCurve", "BSplineCurve", "BSplineSurface", "BezierSurface", "FilledCurve", "JoinedCurve",
                 "Arrow")


def _check_args(head, args):
    """Validate head[args] (options split off); returns the set of coordinate dimensions."""
    if head in _SPLINE_HEADS:
        return _CHECKS[head](args, head)
    a, opts = split_trailing_options(args)
    allowed = _PRIM_OPTS.get(head, ())
    for r in opts:
        n = opt_name(r)
        if n not in allowed:
            raise ValueError(f"{head}: {n} is not a known option of {head}")
        if n in OPTION_VALIDATORS and r.head == "Rule":
            OPTION_VALIDATORS[n](r.args[1], head)
        if n == "RoundingRadius":
            v = r.args[1]
            if K.is_list(v):
                for c in v.args:
                    _nonneg(c, head, "RoundingRadius")
            elif not (isinstance(v, Symbol) and v.name in ("None", "Automatic")):
                _nonneg(v, head, "RoundingRadius")
    return _CHECKS[head](a, head)


# Graphics dimensions each primitive may appear in.
_DIMS = {h: (2, 3) for h in _CHECKS}
for _h in ("Circle", "Disk", "DiskSegment", "Rectangle", "Annulus", "StadiumShape", "RegularPolygon", "Parallelogram",
           "PolarCurve", "FilledPolarCurve", "HalfPlane"):
    _DIMS[_h] = (2,)
for _h in ("Cube", "Tetrahedron", "Octahedron", "Dodecahedron", "Icosahedron", "Hexahedron", "Prism", "Pyramid",
           "Cone", "Cylinder", "CapsuleShape", "Tube", "Polyhedron", "UniformPolyhedron", "Torus", "FilledTorus",
           "BSplineSurface", "BezierSurface", "InfinitePlane"):
    _DIMS[_h] = (3,)
_DIMS["HalfPlane"] = (2, 3)
_DIMS["Parallelogram"] = (2, 3)


def _graphics_validator(head):
    def validate(expr, dim):
        dims = _check_args(head, list(expr.args))
        if dims and dims != {dim}:
            raise ValueError(f"{head}: the coordinates of {K_input(expr)} are not {dim}-dimensional, as required in "
                             f"{'Graphics3D' if dim == 3 else 'Graphics'}")
    return validate


def _register_prim(head, notes):
    def impl(*args: Expr | Option) -> Expr:
        _check_args(head, list(args))
        raise Unevaluated()
    fn(head, area=AREA, status="added", notes=notes)(impl)
    register_primitive(head, _DIMS[head], _graphics_validator(head))


_PRIM_COMMON = (" Inert geometric region and graphics primitive: arguments are validated (strings or booleans as "
                "coordinates raise TypeError; wrong point counts, mixed dimensions or negative radii ValueError; "
                "unknown options ValueError) and the expression then evaluates to itself, with symbolic "
                "coordinates allowed. Inside Graphics/Graphics3D the coordinate dimension must match the "
                "container (else ValueError). Deviation: forms that Wolfram autoevaluates (Disk[] to "
                "Disk[{0, 0}], ...) stay as written; they denote the same region.")


_PRIM_NOTES = {
    "Point": "Point[p] / Point[{p1, p2, ...}] (any dimension; Point[pts, VertexColors -> ...]).",
    "Line": "Line[{p1, p2, ...}] / Line[{{p11, ...}, {p21, ...}, ...}] (any dimension; options VertexColors, "
            "VertexNormals).",
    "Polygon": "Polygon[{p1, ...}], Polygon[{p1, ...} -> {{q1, ...}, ...}] (holes), Polygon[{poly1, ...}] and "
               "Polygon[{p1, ...}, data] with integer vertex indices in data (indices outside 1..n raise "
               "ValueError); options VertexColors / VertexNormals / VertexTextureCoordinates.",
    "Triangle": "Triangle[] (the unit triangle {{0, 0}, {1, 0}, {0, 1}}), Triangle[{p1, p2, p3}] and a list of "
                "corner triples; exactly three points per triangle (else ValueError).",
    "Rectangle": "Rectangle[] / Rectangle[{xmin, ymin}] (unit square) / Rectangle[pmin, pmax]; option "
                 "RoundingRadius -> r (non-negative).",
    "Cuboid": "Cuboid[] / Cuboid[pmin] (unit hypercube) / Cuboid[pmin, pmax], any dimension.",
    "Circle": "Circle[], Circle[{x, y}], Circle[{x, y}, r], Circle[{x, y}, {rx, ry}], Circle[c, r, {t1, t2}] "
              "(arc), and a list of centres; 2-D only.",
    "Disk": "Disk[], Disk[{x, y}], Disk[{x, y}, r | {rx, ry}], Disk[c, r, {t1, t2}] (sector), and a list of "
            "centres; 2-D only.",
    "DiskSegment": "DiskSegment[{x, y}, r | {rx, ry}, {t1, t2}] (circular or elliptic segment); 2-D only.",
    "Sphere": "Sphere[] (Sphere[{0, 0, 0}]), Sphere[n] (unit sphere in R^n), Sphere[p], Sphere[p, r], "
              "Sphere[{p1, ...}, r]; any dimension.",
    "Ball": "Ball[] (Ball[{0, 0, 0}]), Ball[n] (unit ball in R^n), Ball[p], Ball[p, r], Ball[{p1, ...}, r]; any "
            "dimension.",
    "Ellipsoid": "Ellipsoid[p, {r1, ...}] (positive semi-axes) and Ellipsoid[p, Sigma] (Sigma symmetric positive "
                 "definite, else ValueError).",
    "Annulus": "Annulus[] (= Annulus[{0, 0}, {1/2, 1}]), Annulus[r], Annulus[{rin, rout}], Annulus[c, {rin, rout}] "
               "and Annulus[c, {rin, rout}, {t1, t2}]; rin <= rout (else ValueError); 2-D only.",
    "SphericalShell": "SphericalShell[] (= SphericalShell[{0, 0, 0}, {1/2, 1}]), SphericalShell[r], "
                      "SphericalShell[{rin, rout}], SphericalShell[c, {rin, rout}] with rin <= rout; any dimension.",
    "Torus": "Torus[] (= Torus[{0, 0, 0}, {1/2, 1}]) and Torus[{x, y, z}, {rin, rout}] (surface); 3-D only.",
    "FilledTorus": "FilledTorus[] (= FilledTorus[{0, 0, 0}, {1/2, 1}]) and FilledTorus[{x, y, z}, {rin, rout}] "
                   "(solid); 3-D only.",
    "Cube": "Cube[], Cube[l], Cube[{theta, phi}, l], Cube[{x, y, z}, ...] (regular cube, unit edge by default); "
            "the edge length must be positive.",
    "Tetrahedron": "Tetrahedron[] / Tetrahedron[l] / Tetrahedron[{theta, phi}, l] / Tetrahedron[c, ...] (regular) "
                   "and Tetrahedron[{p1, p2, p3, p4}] (general, three-dimensional corners).",
    "Octahedron": "Octahedron[], Octahedron[l], Octahedron[{theta, phi}, l], Octahedron[{x, y, z}, ...].",
    "Dodecahedron": "Dodecahedron[], Dodecahedron[l], Dodecahedron[{theta, phi}, l], Dodecahedron[{x, y, z}, ...].",
    "Icosahedron": "Icosahedron[], Icosahedron[l], Icosahedron[{theta, phi}, l], Icosahedron[{x, y, z}, ...].",
    "Hexahedron": "Hexahedron[{p1, ..., p8}] (faces {p4, p3, p2, p1}, {p1, p2, p6, p5}, ...) and a list of corner "
                  "lists; exactly 8 three-dimensional points.",
    "Prism": "Prism[{p1, ..., p6}] (triangles {p1, p2, p3} and {p4, p5, p6}); exactly 6 three-dimensional points.",
    "Pyramid": "Pyramid[{p1, ..., p5}] (base {p1, ..., p4}, apex p5); exactly 5 three-dimensional points.",
    "Cone": "Cone[] (= Cone[{{0, 0, -1}, {0, 0, 1}}]), Cone[{p1, p2}] and Cone[{p1, p2}, r] (base centre p1, "
            "tip p2).",
    "Cylinder": "Cylinder[] (= Cylinder[{{0, 0, -1}, {0, 0, 1}}]), Cylinder[{p1, p2}] and Cylinder[{p1, p2}, r].",
    "CapsuleShape": "CapsuleShape[] (= CapsuleShape[{{-1, 0, 0}, {1, 0, 0}}, 1]), CapsuleShape[r], "
                    "CapsuleShape[{p1, p2}, r]: points within r of the segment.",
    "StadiumShape": "StadiumShape[] (= StadiumShape[{{-1, 0}, {1, 0}}, 1]), StadiumShape[r], "
                    "StadiumShape[{p1, p2}, r]; 2-D.",
    "Parallelogram": "Parallelogram[] (= Parallelogram[{0, 0}, {{1, 0}, {1, 1}}]) and Parallelogram[p, {v1, v2}].",
    "Parallelepiped": "Parallelepiped[p, {v1, ..., vk}] (any dimension).",
    "Simplex": "Simplex[{p1, ..., pk}] (at most n + 1 points in R^n, else ValueError), Simplex[n] (the standard "
               "simplex in R^n) and a list of simplices.",
    "Polyhedron": "Polyhedron[{f1, ...}] (faces given by three-dimensional points), Polyhedron[{f1, ...} -> "
                  "{{g1, ...}, ...}] (voids), Polyhedron[{poly1, ...}] and Polyhedron[{p1, ...}, {{i1, ...}, ...}] "
                  "(index faces; indices outside 1..n raise ValueError).",
    "UniformPolyhedron": "UniformPolyhedron[\"name\"], UniformPolyhedron[{n, m}] (Schlafli symbol), Wenninger "
                         "numbers, and the positioned forms UniformPolyhedron[{r, theta, phi}, spec] / "
                         "UniformPolyhedron[{x, y, z}, {r, theta, phi}, spec]. Names are not resolved to "
                         "coordinates (geometric computations on it raise NotImplementedError).",
    "RegularPolygon": "RegularPolygon[n], RegularPolygon[r, n], RegularPolygon[{r, theta}, n], "
                      "RegularPolygon[{x, y}, rspec, n] with integer n >= 3 (else ValueError).",
    "BezierCurve": "BezierCurve[{p1, p2, ...}] (composite Bezier curve; option SplineDegree -> d, default 3).",
    "BSplineCurve": "BSplineCurve[{p1, ...}] with options SplineDegree (positive integer), SplineKnots, "
                    "SplineWeights, SplineClosed (True/False).",
    "BSplineSurface": "BSplineSurface[{{p11, p12, ...}, ...}] (rectangular control array) with the spline options "
                      "(per-direction lists allowed).",
    "BezierSurface": "BezierSurface[{{p11, p12, ...}, ...}] (rectangular control array).",
    "FilledCurve": "FilledCurve[{seg1, seg2, ...}] / FilledCurve[{{seg11, ...}, ...}] with segments Line, "
                   "BezierCurve or BSplineCurve (other heads raise TypeError).",
    "JoinedCurve": "JoinedCurve[{seg1, seg2, ...}] / JoinedCurve[{{seg11, ...}, ...}] with segments Line, "
                   "BezierCurve or BSplineCurve; option CurveClosed.",
    "Tube": "Tube[{p1, p2, ...}], Tube[pts, r], Tube[{{...}, ...}, r], Tube[curve, r] (Line, BezierCurve, "
            "BSplineCurve), r a non-negative radius or a list of radii; 3-D only.",
    "Arrow": "Arrow[{p1, p2, ...}], Arrow[pts, s], Arrow[pts, {s1, s2}] (non-negative setbacks) and "
             "Arrow[curve, ...] (Line, BezierCurve, BSplineCurve, JoinedCurve, Tube).",
    "HalfLine": "HalfLine[{p1, p2}] (from p1 through p2) and HalfLine[p, v].",
    "InfiniteLine": "InfiniteLine[{p1, p2}] and InfiniteLine[p, v].",
    "InfinitePlane": "InfinitePlane[{p1, p2, p3}] and InfinitePlane[p, {v1, v2}].",
    "HalfPlane": "HalfPlane[{p1, p2}, w] and HalfPlane[p, v, w].",
    "HalfSpace": "HalfSpace[n, p] (points x with (x - p).n <= 0) and HalfSpace[n, c] (x.n <= c); a zero normal "
                 "raises ValueError.",
    "Hyperplane": "Hyperplane[n] (through the origin), Hyperplane[n, p] and Hyperplane[n, c] (x.n == c); a zero "
                  "normal raises ValueError.",
    "AffineSpace": "AffineSpace[{p1, ..., pk+1}] and AffineSpace[p, {v1, ..., vk}].",
    "AffineHalfSpace": "AffineHalfSpace[{p1, ..., pk+1}, w] and AffineHalfSpace[p, {v1, ..., vk}, w].",
    "ConicHullRegion": "ConicHullRegion[{p1, ...}], ConicHullRegion[p, {v1, ...}], ConicHullRegion[{p1, ...}, "
                       "{w1, ...}] and ConicHullRegion[p, {v1, ...}, {w1, ...}].",
    "EmptyRegion": "EmptyRegion[n], the empty subset of R^n (n a positive integer, else ValueError).",
    "FullRegion": "FullRegion[n], all of R^n (n a positive integer, else ValueError).",
    "PolarCurve": "PolarCurve[r, {theta, tmin, tmax}] and PolarCurve[r, theta] (= {theta, 0, 2 Pi}): the curve "
                  "{r Cos[theta], r Sin[theta]}.",
    "FilledPolarCurve": "FilledPolarCurve[r, theta] / FilledPolarCurve[r, {theta, tmin, tmax}]: the region "
                        "swept by the radius vector.",
}

for _h, _n in _PRIM_NOTES.items():
    _register_prim(_h, _n + _PRIM_COMMON + " Check: reference (Wolfram documentation forms) in tests.")
