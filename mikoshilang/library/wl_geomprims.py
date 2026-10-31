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


def nice_angle(a):
    """Exact angle a: replaced by r Pi when it equals a rational multiple of Pi with denominator <= 720, decided
    by a 60-digit numerical comparison (|a - r Pi| < 1e-50)."""
    a = sp.sympify(a)
    if a.has(sp.Float) or a.free_symbols or not a.is_number:
        return a
    try:
        v = sp.N(a, 60)
        q = sp.Rational(str(sp.N(v / sp.pi, 30))).limit_denominator(720)
        if abs(v - sp.N(q * sp.pi, 60)) < sp.Float(10) ** -50:
            return q * sp.pi
    except Exception:
        pass
    return a


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


# ───────────────────────────── option symbols and scene heads ─────────────────────────────

def _inert(name, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def _opt(*args: Expr) -> Expr:
        raise Unevaluated()
    return _opt


_inert("RoundingRadius", "Option of Rectangle (and Framed-like boxes) giving the radius of rounded corners: a "
                         "non-negative number, {rx, ry}, None or Automatic; Rectangle validates it (negative values "
                         "raise ValueError). Inert as an expression. Check: reference (Wolfram documentation) in "
                         "tests.")
_inert("SplineDegree", "Option of BSplineCurve, BSplineSurface, BezierCurve, BSplineFunction and BezierFunction: a "
                       "positive integer (per-direction list for surfaces) or Automatic; validated by those heads "
                       "(other values raise ValueError). Inert as an expression. Check: reference in tests.")
_inert("SplineKnots", "Option of BSplineCurve / BSplineSurface / BSplineFunction: Automatic, \"Clamped\", "
                      "\"Unclamped\" or an explicit non-decreasing knot list (length n + d + 1, else ValueError in "
                      "BSplineFunction). Inert as an expression. Check: reference in tests.")
_inert("SplineWeights", "Option of BSplineCurve / BSplineSurface / BSplineFunction: Automatic or one positive "
                        "weight per control point (rational B-splines). Inert as an expression. Check: reference "
                        "in tests.")
_inert("SplineClosed", "Option of BSplineCurve / BSplineSurface / BSplineFunction: True or False (whether the "
                       "spline is closed); other values raise ValueError. Inert as an expression. Check: reference "
                       "in tests.")
_inert("CurveClosed", "Option of JoinedCurve / FilledCurve: True, False or a list per component. Inert as an "
                      "expression. Check: reference in tests.")
_inert("RevolutionAxis", "Option of RevolutionPlot3D / RevolutionSurface giving the axis of revolution ({x, y, z} "
                         "or {{x1, y1, z1}, {x2, y2, z2}}). Inert as an expression; no function of this batch reads "
                         "it. Check: reference in tests.")
_inert("GeometricStylingRules", "Option of GeometricScene giving styling rules for the diagram of a scene. Inert as "
                                "an expression. Check: reference in tests.")


def _ck_scene(args):
    _nargs(args, 1, 4, "GeometricScene")
    a, opts = split_trailing_options(args)
    for r in opts:
        if opt_name(r) not in ("GeometricStylingRules", "RandomSeeding", "PlotLabel", "ImageSize"):
            raise ValueError(f"GeometricScene: {opt_name(r)} is not a known option of GeometricScene")
    if not a:
        raise ValueError("GeometricScene: expected GeometricScene[points, hypotheses]")
    p = a[0]
    if not K.is_list(p):
        raise TypeError("GeometricScene: the first argument must be a list of points (or {points, scalars})")
    for x in a[1:]:
        if not K.is_list(x):
            raise TypeError(f"GeometricScene: hypotheses and conclusions must be lists, not {K_input(x)}")


@fn("GeometricScene", area=AREA, status="added",
    notes="GeometricScene[{p1, ...}, {hyp1, ...}], GeometricScene[{{p1, ...}, {k1, ...}}, hyps] and "
          "GeometricScene[params, hyps, {con1, ...}] (option GeometricStylingRules): an inert symbolic scene. "
          "The structure is validated (a non-list point or hypothesis specification raises TypeError, unknown "
          "options ValueError). Solving a scene (RandomInstance, GeometricSolveValues, FindGeometricConjectures) "
          "is not provided by this batch. Check: reference (Wolfram documentation forms) in tests.")
def _geometric_scene(*args: Expr | Option) -> Expr:
    _ck_scene(list(args))
    raise Unevaluated()


@fn("GeometricStep", area=AREA, status="added",
    notes="GeometricStep[{hyp1, ...}] / GeometricStep[hyps, label]: inert step of a GeometricScene; hyps must be a "
          "list (else TypeError) and may not contain another GeometricStep (ValueError). "
          "Check: reference in tests.")
def _geometric_step(hyps: ListOf[Expr], *label: Expr) -> Expr:
    if len(label) > 1:
        raise ValueError("GeometricStep: expected GeometricStep[hyps] or GeometricStep[hyps, label]")
    if not K.is_list(hyps):
        if isinstance(hyps, Symbol):
            raise Unevaluated()
        raise TypeError(f"GeometricStep: the hypotheses must be a list, not {K_input(hyps)}")
    for h in hyps.args:
        if isinstance(h, Expr) and h.head == "GeometricStep":
            raise ValueError("GeometricStep: a step cannot contain another GeometricStep")
    raise Unevaluated()


# ───────────────────────────── vector helpers ─────────────────────────────

def _sub(p, q):
    return [a - b for a, b in zip(p, q)]


def _add(p, q):
    return [a + b for a, b in zip(p, q)]


def _mul(c, p):
    return [c * a for a in p]


def _dot(p, q):
    return sum((a * b for a, b in zip(p, q)), sp.Integer(0))


def _norm(p):
    return sp.sqrt(_dot(p, p))


def _cross(u, v):
    return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]


def _is_zero_vec(v):
    return all(fsign(c) == 0 for c in v)


def _float_mode(*inputs):
    return inexact(*inputs)


def _zero(v, flt):
    v = sp.sympify(v)
    if flt or v.has(sp.Float):
        return abs(num(v)) <= 1e-10
    return sign_of(v, "zero test") == 0


def _rank(vectors, flt):
    if not vectors:
        return 0
    if flt:
        import numpy as np
        A = np.array([[num(c) for c in v] for v in vectors], dtype=float)
        scale = max(1.0, float(np.abs(A).max()))
        return int(np.linalg.matrix_rank(A, tol=1e-10 * scale))
    M = sp.Matrix(vectors)
    return M.rank(simplify=True)


def _point_arg(x, fname, what="point"):
    p = pt(x, fname, what)
    return p


def _point_list(x, fname, what="points", **kw):
    if isinstance(x, Expr) and x.head in ("Line", "Polygon", "Triangle", "Simplex", "Point") and len(x.args) == 1 \
            and K.is_list(x.args[0]):
        x = x.args[0]
    return pts(x, fname, what, **kw)


# ───────────────────────────── angles and vectors ─────────────────────────────

@fn("AngleVector", area=AREA, status="added",
    notes="AngleVector[theta] = {Cos[theta], Sin[theta]}; AngleVector[{r, theta}] = r {Cos, Sin}; "
          "AngleVector[{x, y}, theta] and AngleVector[{x, y}, {r, theta}] add the point. Symbolic arguments give "
          "the formula (as in Wolfram); strings raise TypeError, lists of the wrong length ValueError. "
          "Check: exact; reference values and the identity Norm[AngleVector[{r, t}]] == r in tests.")
def _angle_vector(*args: Real | ListOf[Real]) -> ListOf[Real]:
    _nargs(args, 1, 2, "AngleVector")
    if len(args) == 2:
        p = pt(args[0], "AngleVector", "point", dim=2)
        spec = args[1]
    else:
        p = [sp.Integer(0), sp.Integer(0)]
        spec = args[0]
    if K.is_list(spec):
        if len(spec.args) != 2:
            raise ValueError("AngleVector: the polar specification must be {r, theta}")
        r, t = (sx(c, "AngleVector", w) for c, w in zip(spec.args, ("radius", "angle")))
    else:
        r, t = sp.Integer(1), sx(spec, "AngleVector", "angle")
    return out([p[0] + r * sp.cos(t), p[1] + r * sp.sin(t)], *args, simplify=False)


@fn("CirclePoints", area=AREA, status="added",
    notes="CirclePoints[n], CirclePoints[r, n], CirclePoints[{r, theta1}, n], CirclePoints[{x, y}, rspec, n]: n "
          "points on a circle at angles theta1 + 2 Pi k/n, with theta1 = Pi/n - Pi/2 by default (horizontal "
          "base). n must be a positive integer (symbolic n stays unevaluated; non-integer n raises "
          "NotImplementedError, Wolfram accepts it); r, theta1 and the centre may be symbolic. "
          "Check: exact; reference values (Wolfram documentation) and equal distances to the centre in tests.")
def _circle_points(*args: Real | ListOf[Real] | PosInteger) -> MatrixOf[Real]:
    _nargs(args, 1, 3, "CirclePoints")
    n = args[-1]
    if isinstance(n, (bool, str)) or K.is_list(n):
        raise TypeError(f"CirclePoints: the number of points must be a positive integer, not {K_input(n)}")
    if not K.is_int(n):
        if _numberish(n):
            raise NotImplementedError("CirclePoints: a non-integer number of points is not supported")
        raise Unevaluated()
    n = int(n)
    if n < 1:
        raise ValueError("CirclePoints: the number of points must be positive")
    c = [sp.Integer(0), sp.Integer(0)]
    rest = list(args[:-1])
    if len(rest) == 2:
        c = pt(rest[0], "CirclePoints", "centre", dim=2)
        rest = rest[1:]
    r, t1 = sp.Integer(1), sp.pi / n - sp.pi / 2
    if rest:
        spec = rest[0]
        if K.is_list(spec):
            if len(spec.args) != 2:
                raise ValueError("CirclePoints: the radius specification must be r or {r, theta1}")
            r, t1 = sx(spec.args[0], "CirclePoints", "radius"), sx(spec.args[1], "CirclePoints", "angle")
        else:
            r = sx(spec, "CirclePoints", "radius")
    res = [[c[0] + r * sp.cos(t1 + 2 * sp.pi * k / n), c[1] + r * sp.sin(t1 + 2 * sp.pi * k / n)] for k in range(n)]
    return out(res, *args, simplify=False)


def _planar_interior(p, q1, q2, fname):
    u, v = _sub(q1, p), _sub(q2, p)
    if _is_zero_vec(u) or _is_zero_vec(v):
        raise ValueError(f"{fname}: the angle is undefined because two of the points coincide")
    c = _dot(u, v) / (_norm(u) * _norm(v))
    if len(p) == 2:
        return nice_angle(sp.Abs(sp.atan2(u[0] * v[1] - u[1] * v[0], _dot(u, v))))
    return nice_angle(sp.acos(sp.simplify(c)))


def _planar_ccw(p, q1, q2, fname):
    u, v = _sub(q1, p), _sub(q2, p)
    if len(p) != 2:
        raise ValueError(f"{fname}: oriented angles are only defined in 2D")
    if _is_zero_vec(u) or _is_zero_vec(v):
        raise ValueError(f"{fname}: the angle is undefined because two of the points coincide")
    a = sp.atan2(u[0] * v[1] - u[1] * v[0], _dot(u, v))
    return nice_angle(a if fsign(a) >= 0 else a + 2 * sp.pi)


_PLANAR_SPECS = ("Interior", "Exterior", "FullExterior", "Counterclockwise", "Clockwise")


@fn("PlanarAngle", area=AREA, status="added",
    notes="PlanarAngle[{q1, p, q2}] (interior angle at p, in [0, Pi], any dimension), PlanarAngle[p -> {q1, q2}] "
          "(counterclockwise angle from q1 to q2 around p, in [0, 2 Pi), 2-D), and the specs \"Interior\", "
          "\"Exterior\" (Pi - interior), \"FullExterior\" (2 Pi - interior), \"Counterclockwise\", \"Clockwise\". "
          "Points may be Point[...] objects. Symbolic points stay unevaluated (Wolfram: GeometricScene use); "
          "coincident points raise ValueError, oriented specs in 3D ValueError, unknown specs ValueError. "
          "Check: exact; reference values and interior + exterior == Pi in tests.")
def _planar_angle(spec: ListOf[Expr] | Rule, *kind: String) -> Real:
    _nargs((spec,) + kind, 1, 2, "PlanarAngle")
    k = _string_arg(kind[0], _PLANAR_SPECS, "PlanarAngle") if kind else None
    if isinstance(spec, Expr) and spec.head == "Rule":
        p = pt(spec.args[0], "PlanarAngle")
        qs = _point_list(spec.args[1], "PlanarAngle", n=2)
        q1, q2 = qs
        k = k or "Counterclockwise"
    else:
        ps = _point_list(spec, "PlanarAngle", n=3)
        q1, p, q2 = ps
        k = k or "Interior"
    if len(p) != len(q1):
        raise ValueError("PlanarAngle: all points must have the same dimension")
    concrete(p, q1, q2)
    if k == "Counterclockwise":
        a = _planar_ccw(p, q1, q2, "PlanarAngle")
    elif k == "Clockwise":
        a = _planar_ccw(p, q2, q1, "PlanarAngle")
    else:
        a = _planar_interior(p, q1, q2, "PlanarAngle")
        a = {"Interior": a, "Exterior": sp.pi - a, "FullExterior": 2 * sp.pi - a}[k]
    return out(a, spec)


@fn("AngleBisector", area=AREA, status="added",
    notes="AngleBisector[{q1, p, q2}] and AngleBisector[p -> {q1, q2}] give the bisector of the interior angle at p "
          "as InfiniteLine[p, v] with v = u1/|u1| + u2/|u2| (unit vectors towards q1, q2); \"Exterior\" uses "
          "v = u1/|u1| - u2/|u2|. 2-D only (else ValueError, as in Wolfram); coincident points or a straight angle "
          "for \"Interior\" (where v = 0) take the perpendicular direction. Deviation: the direction vector's "
          "scaling may differ from Wolfram's (same line). Symbolic points stay unevaluated. "
          "Check: self-check (the result direction makes equal angles with both sides, ArithmeticError otherwise).")
def _angle_bisector(spec: ListOf[Expr] | Rule, *kind: String) -> Expr:
    _nargs((spec,) + kind, 1, 2, "AngleBisector")
    k = _string_arg(kind[0], ("Interior", "Exterior"), "AngleBisector", "bisector type") if kind else "Interior"
    if isinstance(spec, Expr) and spec.head == "Rule":
        p = pt(spec.args[0], "AngleBisector")
        q1, q2 = _point_list(spec.args[1], "AngleBisector", n=2)
    else:
        q1, p, q2 = _point_list(spec, "AngleBisector", n=3)
    if len(p) != 2:
        raise ValueError("AngleBisector: only 2-D points are supported (as in Wolfram)")
    concrete(p, q1, q2)
    u, w = _sub(q1, p), _sub(q2, p)
    if _is_zero_vec(u) or _is_zero_vec(w):
        raise ValueError("AngleBisector: the angle is undefined because two of the points coincide")
    a = _mul(1 / _norm(u), u)
    b = _mul(1 / _norm(w), w)
    v = _add(a, b) if k == "Interior" else _sub(a, b)
    if _is_zero_vec(v):
        base = a
        v = [-base[1], base[0]]
    v = [sp.radsimp(sp.simplify(c)) for c in v]
    # self-check: equal (unsigned) angles between the direction and the two sides
    if k == "Interior":
        c1 = _dot(v, a) / _norm(v)
        c2 = _dot(v, b) / _norm(v)
        if abs(num(c1 - c2)) > 1e-9:
            raise ArithmeticError("AngleBisector: result failed verification")
    return Expr("InfiniteLine", out(p, spec), out(v, spec))


@fn("PerpendicularBisector", area=AREA, status="added",
    notes="PerpendicularBisector[{p1, p2}] and PerpendicularBisector[Line[{p1, p2}]] give InfiniteLine[m, v] with "
          "m the midpoint and v = p2 - p1 rotated by +90 degrees. 2-D only (else ValueError, as in Wolfram); equal "
          "points raise ValueError; symbolic points stay unevaluated. "
          "Check: exact; tests verify equidistance of the line points from p1 and p2.")
def _perpendicular_bisector(seg: ListOf[Expr] | Expr) -> Expr:
    if isinstance(seg, Expr) and seg.head == "Line" and len(seg.args) == 1:
        seg = seg.args[0]
    p1, p2 = _point_list(seg, "PerpendicularBisector", n=2)
    if len(p1) != 2:
        raise ValueError("PerpendicularBisector: only 2-D points are supported (as in Wolfram)")
    concrete(p1, p2)
    d = _sub(p2, p1)
    if _is_zero_vec(d):
        raise ValueError("PerpendicularBisector: the two points coincide")
    m = _mul(sp.Rational(1, 2), _add(p1, p2))
    return Expr("InfiniteLine", out(m, seg), out([-d[1], d[0]], seg))


@fn("InfiniteLineThrough", area=AREA, status="added",
    notes="InfiniteLineThrough[{p1, p2, ...}] gives InfiniteLine[{p, q}] through the first two distinct points when "
          "all points are collinear; non-collinear points or fewer than two distinct points raise ValueError. "
          "Symbolic points stay unevaluated (GeometricScene constraint). Check: exact (rank test of p_i - p1).")
def _infinite_line_through(points: ListOf[ListOf[Real]]) -> Expr:
    ps = _point_list(points, "InfiniteLineThrough", lo=2)
    concrete(ps)
    flt = _float_mode(points)
    q = next((p for p in ps[1:] if not all(_zero(c, flt) for c in _sub(p, ps[0]))), None)
    if q is None:
        raise ValueError("InfiniteLineThrough: the points do not determine a line (they coincide)")
    if _rank([_sub(p, ps[0]) for p in ps[1:]], flt) > 1:
        raise ValueError("InfiniteLineThrough: the points are not collinear")
    return Expr("InfiniteLine", out([ps[0], q], points))


def _orientation_det(ps, fname):
    d = len(ps[0])
    if len(ps) != d + 1:
        raise NotImplementedError(f"{fname}: exactly d + 1 points in d dimensions are supported (got {len(ps)} "
                                  f"points in {d} dimensions)")
    return sp.Matrix([_sub(p, ps[0]) for p in ps[1:]]).det()


def _oriented(x, fname, want):
    ps = _point_list(x, fname, lo=2)
    concrete(ps)
    det = _orientation_det(ps, fname)
    return fsign(det) == want


@fn("PositivelyOrientedPoints", area=AREA, status="added",
    notes="PositivelyOrientedPoints[{p1, ..., pd+1}] in R^d is True when Det[{p2 - p1, ..., pd+1 - p1}] > 0 "
          "(counterclockwise in 2-D, right-hand rule in 3-D); Line[...] / Point[...] / Polygon[...] wrappers are "
          "accepted. Other point counts raise NotImplementedError; symbolic points stay unevaluated (Wolfram "
          "returns a condition). Check: exact (sign of the exact determinant; tolerance 1e-12 for machine numbers).")
def _positively_oriented(points: ListOf[ListOf[Real]]) -> Boolean:
    return _oriented(points, "PositivelyOrientedPoints", 1)


@fn("NegativelyOrientedPoints", area=AREA, status="added",
    notes="NegativelyOrientedPoints[{p1, ..., pd+1}] in R^d is True when Det[{p2 - p1, ..., pd+1 - p1}] < 0 "
          "(clockwise in 2-D, left-hand rule in 3-D); wrappers and errors as for PositivelyOrientedPoints. "
          "Check: exact (sign of the exact determinant).")
def _negatively_oriented(points: ListOf[ListOf[Real]]) -> Boolean:
    return _oriented(points, "NegativelyOrientedPoints", -1)


@fn("CollinearPoints", area=AREA, status="added",
    notes="CollinearPoints[{p1, p2, ...}] is True when MatrixRank[{p2 - p1, p3 - p1, ...}] <= 1 (any dimension; "
          "fewer than three points are trivially collinear); Point/Line wrappers accepted. Symbolic points stay "
          "unevaluated (Wolfram returns a condition); mixed dimensions raise ValueError. "
          "Check: exact (exact rank; relative tolerance 1e-10 for machine numbers).")
def _collinear_points(points: ListOf[ListOf[Real]]) -> Boolean:
    ps = _point_list(points, "CollinearPoints", lo=1)
    concrete(ps)
    return _rank([_sub(p, ps[0]) for p in ps[1:]], _float_mode(points)) <= 1


@fn("CoplanarPoints", area=AREA, status="added",
    notes="CoplanarPoints[{p1, p2, ...}] is True when MatrixRank[{p2 - p1, p3 - p1, ...}] <= 2 (any dimension); "
          "Point/Line wrappers accepted; symbolic points stay unevaluated. "
          "Check: exact (exact rank; relative tolerance 1e-10 for machine numbers).")
def _coplanar_points(points: ListOf[ListOf[Real]]) -> Boolean:
    ps = _point_list(points, "CoplanarPoints", lo=1)
    concrete(ps)
    return _rank([_sub(p, ps[0]) for p in ps[1:]], _float_mode(points)) <= 2


@fn("DihedralAngle", area=AREA, status="promoted",
    notes="DihedralAngle[{p1, p2}, {v, w}]: the angle in [0, Pi] between the half-planes bounded by the line "
          "through p1, p2 and extended in the directions v and w (both projected orthogonally to p2 - p1); any "
          "dimension >= 2. Equal points p1 = p2, or v or w parallel to the edge, raise ValueError; symbolic input "
          "stays unevaluated. Replaces the quarantined legacy rule. "
          "Check: exact; reference values (cube and tetrahedron dihedral angles) in tests.")
def _dihedral_angle(edge: ListOf[ListOf[Real]], dirs: ListOf[ListOf[Real]]) -> Real:
    p1, p2 = _point_list(edge, "DihedralAngle", n=2)
    v, w = pts(dirs, "DihedralAngle", "directions", n=2)
    if len(v) != len(p1):
        raise ValueError("DihedralAngle: the points and directions must have the same dimension")
    concrete(p1, p2, v, w)
    e = _sub(p2, p1)
    if _is_zero_vec(e):
        raise ValueError("DihedralAngle: the two edge points coincide")
    ee = _dot(e, e)
    vp = _sub(v, _mul(_dot(v, e) / ee, e))
    wp = _sub(w, _mul(_dot(w, e) / ee, e))
    if _is_zero_vec(vp) or _is_zero_vec(wp):
        raise ValueError("DihedralAngle: a direction is parallel to the edge, so its half-plane is undefined")
    c = sp.simplify(_dot(vp, wp) / (_norm(vp) * _norm(wp)))
    return out(nice_angle(sp.acos(c)), edge, dirs)


def _solid_triangle(a, b, c):
    """Signed solid angle of the cone spanned by vectors a, b, c (Van Oosterom & Strackee 1983)."""
    na, nb, nc = _norm(a), _norm(b), _norm(c)
    numr = _dot(a, _cross(b, c))
    den = na * nb * nc + _dot(a, b) * nc + _dot(a, c) * nb + _dot(b, c) * na
    return 2 * sp.atan2(numr, den)


@fn("SolidAngle", area=AREA, status="added",
    notes="SolidAngle[p, {u1, u2}] in 2-D (angle in [0, Pi] between the vectors) and SolidAngle[p, {u1, u2, u3}] in "
          "3-D (solid angle of the cone they span, Van Oosterom-Strackee formula); SolidAngle[p, reg] for reg a "
          "Ball/Sphere (3-D: 2 Pi (1 - Sqrt[1 - r^2/d^2]), 4 Pi from inside) or Disk/Circle (2-D: 2 ArcSin[r/d], "
          "2 Pi from inside) and for a planar Polygon/Triangle in 3-D (triangle fan; p in the polygon's plane "
          "raises ValueError). Other region heads, more vectors than dimensions, or other dimensions raise "
          "NotImplementedError; zero vectors ValueError; symbolic input stays unevaluated. "
          "Check: exact; reference values (octant = Pi/2, hemisphere = 2 Pi) and the identity that the 8 octants "
          "sum to 4 Pi in tests.")
def _solid_angle(p: ListOf[Real], spec: MatrixOf[Real] | Expr) -> Real:
    pp = pt(p, "SolidAngle", "apex")
    if isinstance(spec, Expr) and isinstance(spec.head, str) and spec.head in (
            "Ball", "Sphere", "Disk", "Circle", "Polygon", "Triangle"):
        return out(_solid_angle_region(pp, spec), p, spec)
    us = pts(spec, "SolidAngle", "vectors")
    if len(us[0]) != len(pp):
        raise ValueError("SolidAngle: the apex and the vectors must have the same dimension")
    concrete(pp, us)
    d = len(pp)
    if any(_is_zero_vec(u) for u in us):
        raise ValueError("SolidAngle: the spanning vectors must be non-zero")
    if d == 2 and len(us) == 2:
        return out(_planar_interior([0, 0], us[0], us[1], "SolidAngle"), p, spec)
    if d == 3 and len(us) == 3:
        return out(sp.Abs(_solid_triangle(*us)), p, spec)
    raise NotImplementedError("SolidAngle: only d vectors in d = 2 or 3 dimensions are supported")


def _solid_angle_region(p, reg):
    h = reg.head
    if h in ("Ball", "Sphere", "Disk", "Circle"):
        c, r = _center_radius(reg, "SolidAngle")
        if len(c) != len(p):
            raise ValueError("SolidAngle: the apex and the region must have the same dimension")
        concrete(p, c, [r])
        dd = _norm(_sub(c, p))
        if len(p) == 3 and h in ("Ball", "Sphere"):
            if fsign(dd - r) <= 0:
                if h == "Sphere" and fsign(dd - r) == 0:
                    raise NotImplementedError("SolidAngle: an apex on the sphere is not supported")
                return 4 * sp.pi
            return 2 * sp.pi * (1 - sp.sqrt(1 - r ** 2 / dd ** 2))
        if len(p) == 2 and h in ("Disk", "Circle", "Ball", "Sphere"):
            if fsign(dd - r) <= 0:
                if fsign(dd - r) == 0 and h in ("Circle", "Sphere"):
                    raise NotImplementedError("SolidAngle: an apex on the circle is not supported")
                return 2 * sp.pi
            return 2 * sp.asin(r / dd)
        raise NotImplementedError(f"SolidAngle: {h} regions in {len(p)} dimensions are not supported")
    rings = polygon_data(reg, "SolidAngle", dim=None)
    if len(rings) != 1 or len(rings[0][0]) != 3 or len(p) != 3:
        raise NotImplementedError("SolidAngle: only a single planar polygon in 3-D is supported")
    vs = rings[0]
    concrete(p, vs)
    n = _cross(_sub(vs[1], vs[0]), _sub(vs[2], vs[0]))
    if fsign(_dot(n, _sub(p, vs[0]))) == 0:
        raise ValueError("SolidAngle: the apex lies in the plane of the polygon")
    tot = sp.Integer(0)
    for i in range(1, len(vs) - 1):
        tot += _solid_triangle(_sub(vs[0], p), _sub(vs[i], p), _sub(vs[i + 1], p))
    return sp.Abs(tot)


def _center_radius(reg, fname):
    a = list(reg.args)
    if reg.head in ("Ball", "Sphere"):
        if not a:
            c = [sp.Integer(0)] * 3
        elif K.is_int(a[0]) and not isinstance(a[0], bool):
            c = [sp.Integer(0)] * int(a[0])
        else:
            c = pt(a[0], fname, "centre")
    else:
        c = pt(a[0], fname, "centre", dim=2) if a else [sp.Integer(0)] * 2
    if len(a) > 1 and K.is_list(a[1]):
        raise NotImplementedError(f"{fname}: elliptical {reg.head} regions are not supported")
    if len(a) > 2:
        raise NotImplementedError(f"{fname}: {reg.head} sectors/arcs are not supported")
    r = sx(a[1], fname, "radius") if len(a) > 1 and not (K.is_int(a[0]) and reg.head in ("Ball", "Sphere")) \
        else sp.Integer(1)
    return c, r


# ───────────────────────────── spherical geometry ─────────────────────────────

def _sph_to_cart(c):
    """Hyperspherical angles {t1, ..., t(n-2), phi} (CoordinateTransformData conventions, r dropped) -> unit
    vector.  Only inner products are used, so the axis order does not matter."""
    out_ = []
    s = sp.Integer(1)
    for t in c[:-1]:
        out_.append(s * sp.cos(t))
        s = s * sp.sin(t)
    out_ += [s * sp.cos(c[-1]), s * sp.sin(c[-1])]
    return out_


def _sph_point(x, fname):
    if isinstance(x, Expr) and x.head == "GeoPosition":
        if len(x.args) != 1 or not K.is_list(x.args[0]) or len(x.args[0].args) < 2:
            raise ValueError(f"{fname}: GeoPosition must be GeoPosition[{{lat, lon}}]")
        lat, lon = (sx(c, fname, "coordinate") for c in x.args[0].args[:2])
        return [sp.pi / 2 - lat * sp.pi / 180, lon * sp.pi / 180]
    p = pt(x, fname, "point", point_head=False)
    if len(p) < 2:
        raise ValueError(f"{fname}: a point on the sphere needs at least two angles {{theta, phi}}")
    return p


def _sph_dist(p, q):
    """Great-circle distance by the haversine formula (accurate for small and antipodal distances)."""
    if len(p) == 2:
        h = sp.sin((q[0] - p[0]) / 2) ** 2 + sp.sin(p[0]) * sp.sin(q[0]) * sp.sin((q[1] - p[1]) / 2) ** 2
        return 2 * sp.asin(sp.sqrt(h))
    a, b = _sph_to_cart(p), _sph_to_cart(q)
    d = _norm(_sub(a, b))
    return 2 * sp.asin(d / 2)


def _is_point_list_arg(x):
    return K.is_list(x) and bool(x.args) and (K.is_list(x.args[0]) or
                                              (isinstance(x.args[0], Expr) and x.args[0].head == "GeoPosition"))


@fn("SphericalDistance", area=AREA, status="added",
    notes="SphericalDistance[{t1, p1}, {t2, p2}]: great-circle distance on the unit sphere between points given by "
          "polar angle theta and azimuth phi (Wolfram's spherical-coordinate convention with r dropped), by the "
          "haversine formula; hyperspherical points {t1, ..., phi} use the chord length 2 ArcSin[|a - b|/2]; "
          "GeoPosition[{lat, lon}] (degrees) is accepted. Threads: SphericalDistance[p, {q1, ...}] gives a list, "
          "SphericalDistance[pts1, pts2] a matrix. Points of different dimension raise ValueError, strings "
          "TypeError; symbolic angles give the closed form. "
          "Check: exact; reference values (quarter circle Pi/2, antipodes Pi) and symmetry in tests.")
def _spherical_distance(p: ListOf[Real] | MatrixOf[Real], q: ListOf[Real] | MatrixOf[Real]) -> Real | ListOf[Real]:
    f = "SphericalDistance"
    pl, ql = _is_point_list_arg(p), _is_point_list_arg(q)
    P = [_sph_point(x, f) for x in p.args] if pl else [_sph_point(p, f)]
    Q = [_sph_point(x, f) for x in q.args] if ql else [_sph_point(q, f)]
    if len({len(x) for x in P + Q}) > 1:
        raise ValueError(f"{f}: the points must have the same dimension")
    M = [[_sph_dist(a, b) for b in Q] for a in P]
    if not pl and not ql:
        return out(M[0][0], p, q)
    if not pl:
        return out(M[0], p, q)
    if not ql:
        return out([r[0] for r in M], p, q)
    return out(M, p, q)


@fn("SphericalAngle", area=AREA, status="added",
    notes="SphericalAngle[p -> {q, r}]: on the 2-sphere (points {theta, phi}, Wolfram's convention) the signed "
          "angle at p from the great circle towards q to the one towards r, positive counterclockwise around the "
          "outward normal at p; on higher-dimensional hyperspheres the unsigned angle. SphericalAngle[p -> {{q1, "
          "r1}, ...}] and SphericalAngle[{p1, ...} -> {{q1, r1}, ...}] give lists. q or r equal or antipodal to p "
          "raise ValueError; symbolic angles stay unevaluated. "
          "Check: exact; reference values (octant triangle angles Pi/2, sign reversal under q <-> r) in tests.")
def _spherical_angle(spec: Rule) -> Real | ListOf[Real]:
    f = "SphericalAngle"
    if not (isinstance(spec, Expr) and spec.head == "Rule"):
        if isinstance(spec, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: expected SphericalAngle[p -> {{q, r}}]")
    lhs, rhs = spec.args
    if not K.is_list(rhs) or not rhs.args:
        raise TypeError(f"{f}: expected p -> {{q, r}}")
    pairs = rhs.args if (K.is_list(rhs.args[0]) and rhs.args[0].args and _is_point_list_arg(rhs.args[0])) else None
    if pairs is None:
        if len(rhs.args) != 2:
            raise ValueError(f"{f}: expected p -> {{q, r}}")
        return out(_sph_angle(_sph_point(lhs, f), _sph_point(rhs.args[0], f), _sph_point(rhs.args[1], f), f), spec)
    plist = [_sph_point(x, f) for x in lhs.args] if _is_point_list_arg(lhs) else None
    if plist is not None and len(plist) != len(pairs):
        raise ValueError(f"{f}: the number of points p must match the number of pairs {{q, r}}")
    res = []
    for i, pr in enumerate(pairs):
        if not K.is_list(pr) or len(pr.args) != 2:
            raise ValueError(f"{f}: each pair must be {{q, r}}")
        p_ = plist[i] if plist is not None else _sph_point(lhs, f)
        res.append(_sph_angle(p_, _sph_point(pr.args[0], f), _sph_point(pr.args[1], f), f))
    return out(res, spec)


def _sph_angle(p, q, r, f):
    if not len(p) == len(q) == len(r):
        raise ValueError(f"{f}: the points must have the same dimension")
    concrete(p, q, r)
    P, Qv, Rv = _sph_to_cart(p), _sph_to_cart(q), _sph_to_cart(r)
    if len(p) == 2:
        # standard axes: x = sin t cos f, y = sin t sin f, z = cos t (right-handed)
        def cart(c):
            return [sp.sin(c[0]) * sp.cos(c[1]), sp.sin(c[0]) * sp.sin(c[1]), sp.cos(c[0])]
        P, Qv, Rv = cart(p), cart(q), cart(r)
    t1 = _sub(Qv, _mul(_dot(Qv, P), P))
    t2 = _sub(Rv, _mul(_dot(Rv, P), P))
    if _is_zero_vec(t1) or _is_zero_vec(t2):
        raise ValueError(f"{f}: the great circle through p and a point equal or antipodal to p is undefined")
    if len(p) == 2:
        return nice_angle(sp.atan2(_dot(P, _cross(t1, t2)), _dot(t1, t2)))
    return nice_angle(sp.acos(sp.simplify(_dot(t1, t2) / (_norm(t1) * _norm(t2)))))


# ───────────────────────────── triangles ─────────────────────────────

def _pos_side(x, f, what="side length"):
    v = sx(x, f, what)
    if v.is_number and fsign(v) <= 0:
        raise ValueError(f"{f}: the {what} must be positive, not {K_input(x)}")
    return v


def _angle_val(x, f):
    v = sx(x, f, "angle")
    if v.is_number and not (0 < num(v) < math.pi):
        raise ValueError(f"{f}: the angles must lie strictly between 0 and Pi, not {K_input(x)}")
    return v


def _tri(points, inputs):
    return Expr("Triangle", out(points, *inputs, simplify=True))


@fn("SSSTriangle", area=AREA, status="added",
    notes="SSSTriangle[a, b, c] = Triangle[{{0, 0}, {c, 0}, {(b^2 + c^2 - a^2)/(2 c), Sqrt[(a + b - c)(a - b + c)"
          "(-a + b + c)(a + b + c)]/(2 c)}}] (Wolfram documentation formula; a, b, c opposite the vertices at the "
          "origin, at {c, 0} and in the upper half-plane). Non-positive sides or a violated triangle inequality "
          "raise ValueError, strings TypeError; symbolic sides give the symbolic triangle (as in Wolfram). "
          "Check: self-check (the side lengths of the result are compared with a, b, c; ArithmeticError).")
def _sss_triangle(a: Real, b: Real, c: Real) -> Expr:
    f = "SSSTriangle"
    a, b, c = (_pos_side(v, f) for v in (a, b, c))
    if all(v.is_number for v in (a, b, c)):
        if fsign(a + b - c) <= 0 or fsign(a - b + c) <= 0 or fsign(-a + b + c) <= 0:
            raise ValueError(f"{f}: the side lengths violate the triangle inequality")
    y = (-a ** 2 + b ** 2 + c ** 2) / (2 * c)
    z = sp.sqrt((a + b - c) * (a - b + c) * (-a + b + c) * (a + b + c)) / (2 * c)
    P = [[0, 0], [c, 0], [y, z]]
    if all(v.is_number for v in (a, b, c)):
        _check_sides(P, (a, b, c), f)
    return _tri(P, (a, b, c))


def _check_sides(P, abc, f):
    A, B, C = ([sp.sympify(t) for t in p] for p in P)
    got = (_norm(_sub(B, C)), _norm(_sub(A, C)), _norm(_sub(A, B)))
    for g, w in zip(got, abc):
        if abs(num(g - w)) > 1e-9 * max(1.0, abs(num(w))):
            raise ArithmeticError(f"{f}: result failed verification")


@fn("SASTriangle", area=AREA, status="added",
    notes="SASTriangle[a, gamma, b] = Triangle[{{0, 0}, {x, 0}, {y, z}}] with x = Sqrt[a^2 + b^2 - 2 a b Cos[gamma]], "
          "y = (b^2 - a b Cos[gamma])/x, z = a b Sin[gamma]/x (Wolfram documentation formula). Sides must be "
          "positive and 0 < gamma < Pi (else ValueError); symbolic arguments give the symbolic triangle. "
          "Check: self-check (side lengths a, b of the result are verified; ArithmeticError).")
def _sas_triangle(a: Real, gamma: Real, b: Real) -> Expr:
    f = "SASTriangle"
    a, b = _pos_side(a, f), _pos_side(b, f)
    g = _angle_val(gamma, f)
    x = sp.sqrt(a ** 2 + b ** 2 - 2 * a * b * sp.cos(g))
    P = [[0, 0], [x, 0], [(b ** 2 - a * b * sp.cos(g)) / x, a * b * sp.sin(g) / x]]
    if all(v.is_number for v in (a, b, g)):
        _check_sides(P, (a, b, x), f)
    return _tri(P, (a, gamma, b))


@fn("ASATriangle", area=AREA, status="added",
    notes="ASATriangle[alpha, c, beta] = Triangle[{{0, 0}, {c, 0}, {c x, c y}}] with x = Cos[alpha] Csc[alpha + beta] "
          "Sin[beta], y = Csc[alpha + beta] Sin[alpha] Sin[beta] (Wolfram documentation formula). Angles must lie in "
          "(0, Pi) with alpha + beta < Pi and c > 0 (else ValueError); symbolic arguments give the symbolic triangle. "
          "Check: self-check (the angle at the origin is verified; ArithmeticError).")
def _asa_triangle(alpha: Real, c: Real, beta: Real) -> Expr:
    f = "ASATriangle"
    al, be = _angle_val(alpha, f), _angle_val(beta, f)
    c_ = _pos_side(c, f)
    if al.is_number and be.is_number and fsign(al + be - sp.pi) >= 0:
        raise ValueError(f"{f}: the two angles must sum to less than Pi")
    s = 1 / sp.sin(al + be)
    P = [[0, 0], [c_, 0], [c_ * sp.cos(al) * s * sp.sin(be), c_ * s * sp.sin(al) * sp.sin(be)]]
    if all(v.is_number for v in (al, be, c_)):
        ang = sp.atan2(P[2][1], P[2][0])
        if abs(num(ang - al)) > 1e-9:
            raise ArithmeticError(f"{f}: result failed verification")
    return _tri(P, (alpha, c, beta))


@fn("AASTriangle", area=AREA, status="added",
    notes="AASTriangle[alpha, beta, a] = Triangle[{{0, 0}, {a Csc[alpha] Sin[alpha + beta], 0}, {a Cot[alpha] "
          "Sin[beta], a Sin[beta]}}] (Wolfram documentation formula; a is opposite alpha). Angles in (0, Pi) with "
          "alpha + beta < Pi and a > 0 (else ValueError); symbolic arguments give the symbolic triangle. "
          "Check: self-check (the side opposite the origin angle has length a; ArithmeticError).")
def _aas_triangle(alpha: Real, beta: Real, a: Real) -> Expr:
    f = "AASTriangle"
    al, be = _angle_val(alpha, f), _angle_val(beta, f)
    a_ = _pos_side(a, f)
    if al.is_number and be.is_number and fsign(al + be - sp.pi) >= 0:
        raise ValueError(f"{f}: the two angles must sum to less than Pi")
    P = [[0, 0], [a_ / sp.sin(al) * sp.sin(al + be), 0], [a_ * sp.cot(al) * sp.sin(be), a_ * sp.sin(be)]]
    if all(v.is_number for v in (al, be, a_)):
        side = _norm(_sub([sp.sympify(t) for t in P[1]], [sp.sympify(t) for t in P[2]]))
        if abs(num(side - a_)) > 1e-9 * max(1.0, abs(num(a_))):
            raise ArithmeticError(f"{f}: result failed verification")
    return _tri(P, (alpha, beta, a))


def _triangle_pts(tri, f):
    if isinstance(tri, Expr) and tri.head in ("Triangle", "Polygon") and len(tri.args) == 1:
        tri = tri.args[0]
    elif isinstance(tri, Expr) and tri.head == "Triangle" and not tri.args:
        return [[sp.Integer(0), sp.Integer(0)], [sp.Integer(1), sp.Integer(0)], [sp.Integer(0), sp.Integer(1)]]
    P = pts(tri, f, "triangle vertices", n=3)
    concrete(P)
    if _rank([_sub(P[1], P[0]), _sub(P[2], P[0])], False) < 2:
        raise ValueError(f"{f}: the triangle is degenerate (collinear vertices)")
    return P


def _vertex_index(spec, P, f):
    """Vertex spec: index i (1..3), an explicit vertex {x, y} / Point[...]; returns 0-based index."""
    if K.is_int(spec) and not isinstance(spec, bool):
        i = int(spec)
        if not 1 <= i <= 3:
            raise ValueError(f"{f}: vertex index {i} must be 1, 2 or 3")
        return i - 1
    if isinstance(spec, Symbol):
        raise Unevaluated()
    v = pt(spec, f, "vertex")
    for i, p in enumerate(P):
        if len(p) == len(v) and all(fsign(a - b) == 0 for a, b in zip(p, v)):
            return i
    raise ValueError(f"{f}: {K_input(spec)} is not a vertex of the triangle")


def _abc(P):
    A, B, C = P
    return _norm(_sub(B, C)), _norm(_sub(A, C)), _norm(_sub(A, B))


def _tri_area(P):
    u, v = _sub(P[1], P[0]), _sub(P[2], P[0])
    return sp.sqrt(sp.simplify(_dot(u, u) * _dot(v, v) - _dot(u, v) ** 2)) / 2


def _bary(P, w):
    s = sum(w, sp.Integer(0))
    if fsign(s) == 0:
        raise ValueError("TriangleCenter: the center is at infinity for this triangle")
    return [sp.simplify(sum((w[i] * P[i][k] for i in range(3)), sp.Integer(0)) / s) for k in range(len(P[0]))]


def _rot(i):
    """Vertex order (i, j, k) with j, k the other two vertices in cyclic order."""
    return i, (i + 1) % 3, (i + 2) % 3


def _center(P, typ, f):
    a, b, c = _abc(P)
    a2, b2, c2 = (sp.simplify(t ** 2) for t in (a, b, c))
    SA, SB, SC = (b2 + c2 - a2) / 2, (a2 + c2 - b2) / 2, (a2 + b2 - c2) / 2
    if typ == "Centroid":
        return _bary(P, [1, 1, 1])
    if typ == "Incenter":
        return _bary(P, [a, b, c])
    if typ == "Circumcenter":
        return _bary(P, [a2 * SA, b2 * SB, c2 * SC])
    if typ == "Orthocenter":
        if any(fsign(s) == 0 for s in (SA, SB, SC)):          # right triangle: the right-angle vertex
            return list(P[[fsign(s) == 0 for s in (SA, SB, SC)].index(True)])
        return _bary(P, [SB * SC, SC * SA, SA * SB])
    if typ == "NinePointCenter":
        O = _center(P, "Circumcenter", f)
        H = _center(P, "Orthocenter", f)
        return [sp.simplify((o + h) / 2) for o, h in zip(O, H)]
    if typ == "SymmedianPoint":
        return _bary(P, [a2, b2, c2])
    raise ValueError(f"{f}: unknown center type {typ!r}")


_CENTER_NAMES = ("Centroid", "Circumcenter", "Incenter", "Orthocenter", "NinePointCenter", "SymmedianPoint")
_VERTEX_CENTERS = ("AngleBisectingCevianEndpoint", "CevianEndpoint", "Excenter", "Foot", "Midpoint",
                   "SymmedianEndpoint")


def _vertex_center(P, typ, i, f, center=None):
    i, j, k = _rot(i)
    A, B, C = P[i], P[j], P[k]
    sides = _abc(P)
    a, b, c = sides[i], sides[j], sides[k]           # a opposite A, b opposite B, c opposite C
    if typ == "Midpoint":
        return [sp.simplify((x + y) / 2) for x, y in zip(B, C)]
    if typ == "Foot":
        d = _sub(C, B)
        t = _dot(_sub(A, B), d) / _dot(d, d)
        return [sp.simplify(x) for x in _add(B, _mul(t, d))]
    if typ == "Excenter":
        w = [0, 0, 0]
        w[i], w[j], w[k] = -a, b, c
        return _bary(P, w)
    if typ == "AngleBisectingCevianEndpoint":
        return [sp.simplify((b * x + c * y) / (b + c)) for x, y in zip(B, C)]
    if typ == "SymmedianEndpoint":
        return [sp.simplify((b ** 2 * x + c ** 2 * y) / (b ** 2 + c ** 2)) for x, y in zip(B, C)]
    if typ == "CevianEndpoint":
        X = center
        # intersection of line A X with line B C (in the plane of the triangle)
        M = sp.Matrix([_sub(X, A), _sub(B, C)]).T
        rhs = sp.Matrix(_sub(B, A))
        try:
            sol = (M.T * M).LUsolve(M.T * rhs)
        except Exception:
            raise ValueError(f"{f}: the cevian through the center is parallel to the opposite side")
        s = sol[1]
        return [sp.simplify(x) for x in _sub(B, _mul(s, _sub(B, C)))]
    raise ValueError(f"{f}: unknown center type {typ!r}")


def _parse_type(spec, P, f, names, vnames, extra=()):
    """-> (typ, vertex index list or None, center point or None, all_flag)."""
    if isinstance(spec, str):
        if spec in names or spec in extra:
            return spec, None, None, False
        if spec in vnames:
            return spec, [1], None, False                 # short form: vertex p2
        raise ValueError(f"{f}: unknown type {spec!r}")
    if not K.is_list(spec) or not spec.args or not isinstance(spec.args[0], str):
        if isinstance(spec, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: the type must be a string or {{\"type\", vertex}}, not {K_input(spec)}")
    typ = spec.args[0]
    rest = list(spec.args[1:])
    if typ not in vnames:
        raise ValueError(f"{f}: unknown type {typ!r}")
    center = None
    if typ in ("CevianEndpoint", "Cevian"):
        if not rest:
            raise ValueError(f"{f}: {typ} needs a center")
        cs = rest.pop(0)
        center = _center(P, cs, f) if isinstance(cs, str) else pt(cs, f, "center")
        concrete(center)
    if len(rest) > 1:
        raise ValueError(f"{f}: too many elements in {K_input(spec)}")
    if not rest:
        return typ, [1], center, False
    v = rest[0]
    if isinstance(v, Symbol) and v.name == "All":
        return typ, [0, 1, 2], center, True
    return typ, [_vertex_index(v, P, f)], center, False


@fn("TriangleCenter", area=AREA, status="added",
    notes="TriangleCenter[tri] (centroid) and TriangleCenter[tri, type] for tri = {p1, p2, p3}, Triangle[...] or "
          "Polygon[...] in any dimension; types \"Centroid\", \"Circumcenter\", \"Incenter\", \"Orthocenter\", "
          "\"NinePointCenter\", \"SymmedianPoint\" and {\"Midpoint\" | \"Foot\" | \"Excenter\" | "
          "\"AngleBisectingCevianEndpoint\" | \"SymmedianEndpoint\", p} / {\"CevianEndpoint\", center, p} with p a "
          "vertex index, an explicit vertex or All (short forms use vertex p2, as documented). Computed with exact "
          "barycentric coordinates. Degenerate triangles, unknown types or non-vertices raise ValueError; symbolic "
          "vertices stay unevaluated (GeometricScene use in Wolfram). "
          "Check: self-check (circumcenter equidistant from the vertices, incenter equidistant from the sides; "
          "ArithmeticError).")
def _triangle_center(tri: MatrixOf[Real] | Expr, *typ: String | ListOf[Expr]) -> ListOf[Real] | MatrixOf[Real]:
    f = "TriangleCenter"
    _nargs((tri,) + typ, 1, 2, f)
    P = _triangle_pts(tri, f)
    spec = typ[0] if typ else "Centroid"
    t, idx, center, all_ = _parse_type(spec, P, f, _CENTER_NAMES, _VERTEX_CENTERS)
    if idx is None:
        res = _center(P, t, f)
        _verify_center(P, t, res, f)
        return out(res, tri)
    vals = [_vertex_center(P, t, i, f, center) for i in idx]
    return out(vals if all_ else vals[0], tri)


def _verify_center(P, t, c, f):
    if t == "Circumcenter":
        ds = [num(_dot(_sub(c, p), _sub(c, p))) for p in P]
    elif t == "Incenter":
        ds = []
        for i in range(3):
            _, j, k = _rot(i)
            d = _sub(P[k], P[j])
            w = _sub(c, P[j])
            ds.append(num((_dot(w, w) * _dot(d, d) - _dot(w, d) ** 2) / _dot(d, d)))
    else:
        return
    if max(ds) - min(ds) > 1e-9 * max(1.0, max(abs(x) for x in ds)):
        raise ArithmeticError(f"{f}: result failed verification")


_MEASURES = ("Area", "Circumradius", "Inradius", "NinePointRadius", "Perimeter", "Semiperimeter")
_VERTEX_MEASURES = ("Exradius", "ExteriorAngle", "FullExteriorAngle", "Height", "InteriorAngle")


def _measure(P, t, i, f):
    a, b, c = _abc(P)
    K_ = _tri_area(P)
    s = (a + b + c) / 2
    if t == "Area":
        return K_
    if t == "Perimeter":
        return a + b + c
    if t == "Semiperimeter":
        return s
    if t == "Circumradius":
        return a * b * c / (4 * K_)
    if t == "NinePointRadius":
        return a * b * c / (8 * K_)
    if t == "Inradius":
        return K_ / s
    i_, j, k = _rot(i)
    sides = (a, b, c)
    ai, bj, ck = sides[i_], sides[j], sides[k]
    if t == "Exradius":
        return K_ / (s - ai)
    if t == "Height":
        return 2 * K_ / ai
    ang = nice_angle(sp.acos(sp.simplify((bj ** 2 + ck ** 2 - ai ** 2) / (2 * bj * ck))))
    return {"InteriorAngle": ang, "ExteriorAngle": sp.pi - ang, "FullExteriorAngle": 2 * sp.pi - ang}[t]


@fn("TriangleMeasurement", area=AREA, status="added",
    notes="TriangleMeasurement[tri, type] for tri = {p1, p2, p3}, Triangle[...] or Polygon[...] (any dimension): "
          "\"Area\", \"Perimeter\", \"Semiperimeter\", \"Circumradius\" (abc/(4K)), \"Inradius\" (K/s), "
          "\"NinePointRadius\" (R/2) and {\"InteriorAngle\" | \"ExteriorAngle\" | \"FullExteriorAngle\" | "
          "\"Height\" | \"Exradius\", p} with p a vertex index, explicit vertex or All (short forms use vertex p2). "
          "Degenerate triangles and unknown types raise ValueError; symbolic vertices stay unevaluated. "
          "Check: exact; reference values (3-4-5 triangle) and the identities K = r s and sum of angles = Pi in "
          "tests.")
def _triangle_measurement(tri: MatrixOf[Real] | Expr, typ: String | ListOf[Expr]) -> Real | ListOf[Real]:
    f = "TriangleMeasurement"
    P = _triangle_pts(tri, f)
    t, idx, _c, all_ = _parse_type(typ, P, f, _MEASURES, _VERTEX_MEASURES)
    if idx is None:
        return out(sp.simplify(_measure(P, t, 0, f)), tri)
    vals = [sp.simplify(_measure(P, t, i, f)) for i in idx]
    return out(vals if all_ else vals[0], tri)


_POINT_CONSTRUCTS = _CENTER_NAMES
_LINE_CONSTRUCTS = ("Altitude", "AngleBisectingCevian", "AngleBisector", "Cevian", "ExteriorAngleBisector", "Median",
                    "OppositeSide", "PerpendicularBisector", "Symmedian")
_CIRCLE_CONSTRUCTS = ("Circumcircle", "Incircle", "NinePointCircle")
_TRI_CONSTRUCTS = ("AntimedialTriangle", "MedialTriangle", "Triangle")


def _construct(P, t, i, center, f, inputs):
    if t in _CENTER_NAMES:
        return Expr("Point", out(_center(P, t, f), *inputs))
    if t in _VERTEX_CENTERS:
        return Expr("Point", out(_vertex_center(P, t, i, f, center), *inputs))
    if t == "Boundary":
        return Expr("Line", out([P[0], P[1], P[2], P[0]], *inputs))
    if t in _TRI_CONSTRUCTS:
        if t == "Triangle":
            Q = P
        elif t == "MedialTriangle":
            Q = [_mul(sp.Rational(1, 2), _add(P[j], P[k])) for (_, j, k) in map(_rot, range(3))]
        else:
            Q = [_sub(_add(P[j], P[k]), P[i_]) for (i_, j, k) in map(_rot, range(3))]
        return Expr("Triangle", out(Q, *inputs))
    if t in _CIRCLE_CONSTRUCTS or t == "Excircle":
        if len(P[0]) != 2:
            raise NotImplementedError(f"{f}: circles of triangles in {len(P[0])} dimensions are not supported")
        a, b, c = _abc(P)
        K_ = _tri_area(P)
        s = (a + b + c) / 2
        if t == "Circumcircle":
            cc, r = _center(P, "Circumcenter", f), a * b * c / (4 * K_)
        elif t == "Incircle":
            cc, r = _center(P, "Incenter", f), K_ / s
        elif t == "NinePointCircle":
            cc, r = _center(P, "NinePointCenter", f), a * b * c / (8 * K_)
        else:
            cc, r = _vertex_center(P, "Excenter", i, f), _measure(P, "Exradius", i, f)
        return Expr("Circle", out(cc, *inputs), out(sp.simplify(r), *inputs))
    i_, j, k = _rot(i if i is not None else 1)
    A, B, C = P[i_], P[j], P[k]
    if t == "OppositeSide":
        return Expr("Line", out([B, C], *inputs))
    if t == "Altitude":
        return Expr("Line", out([A, _vertex_center(P, "Foot", i_, f)], *inputs))
    if t == "Median":
        return Expr("Line", out([A, _vertex_center(P, "Midpoint", i_, f)], *inputs))
    if t == "AngleBisectingCevian":
        return Expr("Line", out([A, _vertex_center(P, "AngleBisectingCevianEndpoint", i_, f)], *inputs))
    if t == "Symmedian":
        return Expr("Line", out([A, _vertex_center(P, "SymmedianEndpoint", i_, f)], *inputs))
    if t == "Cevian":
        return Expr("Line", out([A, _vertex_center(P, "CevianEndpoint", i_, f, center)], *inputs))
    if t in ("AngleBisector", "ExteriorAngleBisector"):
        u, w = _sub(B, A), _sub(C, A)
        e1, e2 = _mul(1 / _norm(u), u), _mul(1 / _norm(w), w)
        v = _add(e1, e2) if t == "AngleBisector" else _sub(e1, e2)
        return Expr("InfiniteLine", out(A, *inputs), out([sp.radsimp(sp.simplify(x)) for x in v], *inputs))
    if t == "PerpendicularBisector":
        m = _mul(sp.Rational(1, 2), _add(B, C))
        d = _sub(C, B)
        if len(d) != 2:
            raise NotImplementedError(f"{f}: perpendicular bisectors are only supported in 2D")
        return Expr("InfiniteLine", out(m, *inputs), out([-d[1], d[0]], *inputs))
    raise ValueError(f"{f}: unknown construct {t!r}")


@fn("TriangleConstruct", area=AREA, status="added",
    notes="TriangleConstruct[tri, type] for tri = {p1, p2, p3}, Triangle[...] or Polygon[...]: point types (as in "
          "TriangleCenter) give Point[...]; \"Boundary\" Line[{p1, p2, p3, p1}]; {\"Altitude\" | \"Median\" | "
          "\"AngleBisectingCevian\" | \"Symmedian\" | \"OppositeSide\", p} and {\"Cevian\", center, p} give "
          "Line segments; {\"AngleBisector\" | \"ExteriorAngleBisector\" | \"PerpendicularBisector\", p} and "
          "\"EulerLine\" give InfiniteLine; \"Circumcircle\", \"Incircle\", \"NinePointCircle\", {\"Excircle\", p} "
          "give Circle (2-D only, NotImplementedError otherwise); \"MedialTriangle\", \"AntimedialTriangle\", "
          "\"Triangle\" give Triangle. p is a vertex index, explicit vertex or All (short form: vertex p2). "
          "Deviation: the vertex order of derived triangles and the scaling of InfiniteLine directions may differ "
          "from Wolfram's (same regions). Degenerate triangles, an undefined Euler line (equilateral triangle) and "
          "unknown types raise ValueError; symbolic vertices stay unevaluated. "
          "Check: exact; tests verify incidence relations (feet on the opposite side, circle through the vertices).")
def _triangle_construct(tri: MatrixOf[Real] | Expr, typ: String | ListOf[Expr]) -> Expr | ListOf[Expr]:
    f = "TriangleConstruct"
    P = _triangle_pts(tri, f)
    if typ == "EulerLine":
        O, H = _center(P, "Circumcenter", f), _center(P, "Orthocenter", f)
        if _is_zero_vec(_sub(O, H)):
            raise ValueError(f"{f}: the Euler line of an equilateral triangle is undefined")
        return Expr("InfiniteLine", out([O, H], tri))
    vnames = _VERTEX_CENTERS + _LINE_CONSTRUCTS + ("Excircle",)
    t, idx, center, all_ = _parse_type(typ, P, f, _POINT_CONSTRUCTS + _CIRCLE_CONSTRUCTS + _TRI_CONSTRUCTS,
                                       vnames, ("Boundary",))
    if idx is None:
        return _construct(P, t, None, center, f, (tri,))
    vals = [_construct(P, t, i, center, f, (tri,)) for i in idx]
    return L(*vals) if all_ else vals[0]


# ───────────────────────────── circles and spheres through points ─────────────────────────────

def _circumcenter_affine(P, flt=False):
    """Centre (in the affine hull) and squared radius of the sphere through affinely independent points P."""
    p0 = P[0]
    V = [_sub(p, p0) for p in P[1:]]
    if not V:
        return list(p0), sp.Integer(0)
    G = sp.Matrix([[_dot(u, w) for w in V] for u in V])
    rhs = sp.Matrix([_dot(u, u) / 2 for u in V])
    if flt:
        import numpy as np
        Gn = np.array(G.evalf(30).tolist(), dtype=float)
        if abs(np.linalg.det(Gn)) <= 1e-12 * max(1.0, float(np.abs(Gn).max())) ** len(V):
            raise ZeroDivisionError
        lam = np.linalg.solve(Gn, np.array(rhs.evalf(30).tolist(), dtype=float).ravel())
        lam = [sp.Float(x) for x in lam]
    else:
        if fsign(G.det()) == 0:
            raise ZeroDivisionError
        lam = list(G.LUsolve(rhs))
    c = list(p0)
    for l_, u in zip(lam, V):
        c = _add(c, _mul(l_, u))
    c = [sp.simplify(x) if not flt else x for x in c]
    d = _sub(P[0], c)
    return c, sp.simplify(_dot(d, d)) if not flt else _dot(d, d)


def _sphere_through(P, f, inputs):
    """Sphere through all P (P spans the space); ValueError if they are not cospherical. -> (c, r)."""
    flt = _float_mode(*inputs)
    d = len(P[0])
    # choose d + 1 affinely independent points
    base = [P[0]]
    for p in P[1:]:
        cand = base + [p]
        if _rank([_sub(q, cand[0]) for q in cand[1:]], flt) == len(cand) - 1:
            base = cand
        if len(base) == d + 1:
            break
    if len(base) < d + 1:
        raise ValueError(f"{f}: the points are affinely dependent (degenerate), so the circumsphere is undefined")
    try:
        c, r2 = _circumcenter_affine(base, flt)
    except ZeroDivisionError:
        raise ValueError(f"{f}: the points are degenerate")
    for p in P:
        e = _sub(p, c)
        if not _zero(_dot(e, e) - r2, flt):
            raise ValueError(f"{f}: the points do not lie on a common sphere")
    return c, sp.sqrt(r2)


def _vertices_of(x, f):
    """Coordinates of a point list, polygon or polyhedron argument."""
    if isinstance(x, Expr) and isinstance(x.head, str) and x.head in _POLYHEDRON_HEADS:
        V, _F = polyhedron_data(x, f)
        return V
    if isinstance(x, Expr) and isinstance(x.head, str) and x.head in _POLYGON_HEADS:
        rings = polygon_data(x, f, dim=None)
        if len(rings) != 1:
            raise NotImplementedError(f"{f}: polygons with several components or holes are not supported")
        return rings[0]
    return pts(x, f, lo=1)


@fn("Circumsphere", area=AREA, status="added",
    notes="Circumsphere[{p1, ..., pn+1}] gives Sphere[c, r] through n + 1 affinely independent points of R^n "
          "(circumcircle in 2-D); Circumsphere[poly] for a Polygon/Triangle/Rectangle or polyhedron uses its "
          "vertices, which must lie on a common sphere. Wrong point counts or degenerate (affinely dependent) points "
          "raise ValueError, as do vertices that are not cospherical; symbolic points stay unevaluated. "
          "Check: self-check (every vertex lies on the sphere; ValueError otherwise) and reference values.")
def _circumsphere(points: MatrixOf[Real] | Expr) -> Expr:
    f = "Circumsphere"
    P = _vertices_of(points, f)
    concrete(P)
    d = len(P[0])
    if K.is_list(points) and len(P) != d + 1:
        raise ValueError(f"{f}: expected {d + 1} points in {d} dimensions, got {len(P)}")
    c, r = _sphere_through(P, f, (points,))
    return Expr("Sphere", out(c, points), out(r, points))


def _facet_measure(F):
    """(k-1)-volume of the simplex F (k points)."""
    V = [_sub(p, F[0]) for p in F[1:]]
    G = sp.Matrix([[_dot(u, w) for w in V] for u in V])
    return sp.sqrt(sp.simplify(G.det())) / math.factorial(len(V))


@fn("Insphere", area=AREA, status="added",
    notes="Insphere[{p1, ..., pn+1}] gives Sphere[c, r] inscribed in the simplex (incircle of a triangle): c is "
          "the facet-measure weighted mean of the opposite vertices and r = n Volume/Sum[facet measures]. "
          "Insphere[poly] for a convex Polygon/Triangle/Rectangle in 2-D or a convex polyhedron tangential to all "
          "edges/faces gives the sphere tangent to them; a polygon or polyhedron without an insphere raises "
          "ValueError, a non-convex one NotImplementedError; degenerate simplices raise ValueError; symbolic points "
          "stay unevaluated. Check: self-check (the centre is at distance r from every facet; ArithmeticError).")
def _insphere(points: MatrixOf[Real] | Expr) -> Expr:
    f = "Insphere"
    if isinstance(points, Expr) and isinstance(points.head, str) and points.head in _POLYHEDRON_HEADS + _POLYGON_HEADS \
            and not (points.head in ("Triangle", "Simplex") and points.args):
        return _insphere_poly(points, f)
    P = _vertices_of(points, f)
    concrete(P)
    d = len(P[0])
    if len(P) != d + 1:
        raise ValueError(f"{f}: expected {d + 1} points in {d} dimensions, got {len(P)}")
    if _rank([_sub(p, P[0]) for p in P[1:]], _float_mode(points)) < d:
        raise ValueError(f"{f}: the simplex is degenerate")
    w = [_facet_measure(P[:i] + P[i + 1:]) for i in range(d + 1)]
    S = sum(w, sp.Integer(0))
    c = [sp.simplify(sum((w[i] * P[i][k] for i in range(d + 1)), sp.Integer(0)) / S) for k in range(d)]
    vol = _facet_measure(P)
    r = sp.simplify(d * vol / S)
    return Expr("Sphere", out(c, points), out(r, points))


def _insphere_poly(poly, f):
    facets = _facet_planes(poly, f)              # list of (unit outward normal n, offset b): n.x <= b
    d = len(facets[0][0])
    # solve n_i.c + r = b_i for the first d + 1 independent facets, then verify the rest
    rows = []
    for n_, b_ in facets:
        cand = rows + [(n_, b_)]
        M = sp.Matrix([list(n) + [1] for n, _ in cand])
        if M.rank(simplify=True) == len(cand):
            rows = cand
        if len(rows) == d + 1:
            break
    if len(rows) < d + 1:
        raise ValueError(f"{f}: the region is unbounded or degenerate")
    M = sp.Matrix([list(n) + [1] for n, _ in rows])
    sol = M.LUsolve(sp.Matrix([b for _, b in rows]))
    c, r = [sp.simplify(x) for x in sol[:d]], sp.simplify(sol[d])
    for n_, b_ in facets:
        if not _zero(_dot(n_, c) + r - b_, _float_mode(poly)):
            raise ValueError(f"{f}: the {('polygon' if d == 2 else 'polyhedron')} has no inscribed sphere tangent to "
                             f"all its {'edges' if d == 2 else 'faces'}")
    if fsign(r) <= 0:
        raise ValueError(f"{f}: the region has no inscribed sphere")
    return Expr("Sphere", out(c, poly), out(r, poly))


def _facet_planes(poly, f):
    if isinstance(poly.head, str) and poly.head in _POLYHEDRON_HEADS:
        V, F = polyhedron_data(poly, f)
        concrete(V)
        if not _convex_polyhedron(V, F, f):
            raise NotImplementedError(f"{f}: non-convex polyhedra are not supported")
        cen = [sum((v[k] for v in V), sp.Integer(0)) / len(V) for k in range(3)]
        out_ = []
        for face in F:
            a, b, c = (V[i] for i in face[:3])
            n_ = _cross(_sub(b, a), _sub(c, a))
            if fsign(_dot(n_, _sub(a, cen))) < 0:
                n_ = _mul(-1, n_)
            nn = _norm(n_)
            u = [sp.simplify(x / nn) for x in n_]
            out_.append((u, sp.simplify(_dot(u, a))))
        return out_
    rings = polygon_data(poly, f, dim=2)
    if len(rings) != 1:
        raise NotImplementedError(f"{f}: polygons with several components or holes are not supported")
    R_ = rings[0]
    concrete(R_)
    if not _convex_ring(R_):
        raise NotImplementedError(f"{f}: non-convex polygons are not supported")
    s = fsign(_signed_area(R_))
    out_ = []
    for i in range(len(R_)):
        a, b = R_[i], R_[(i + 1) % len(R_)]
        e = _sub(b, a)
        n_ = [e[1], -e[0]] if s > 0 else [-e[1], e[0]]
        nn = _norm(n_)
        u = [sp.simplify(x / nn) for x in n_]
        out_.append((u, sp.simplify(_dot(u, a))))
    return out_


@fn("CircleThrough", area=AREA, status="added",
    notes="CircleThrough[{p1, p2, p3, ...}] gives Circle[c, r] through the 2-D points (all must be concyclic, else "
          "ValueError); CircleThrough[pts, q] gives Circle[q, r] (all points equidistant from q, else ValueError); "
          "CircleThrough[pts, q, r] gives Circle[q, r] after checking the points lie on it. Fewer than three points "
          "without a centre do not determine a circle (ValueError); collinear points ValueError; non-2-D points "
          "ValueError; symbolic points stay unevaluated (GeometricScene constraint in Wolfram). "
          "Check: self-check (every point lies on the returned circle).")
def _circle_through(points: MatrixOf[Real], *rest: ListOf[Real] | Real) -> Expr:
    f = "CircleThrough"
    _nargs((points,) + rest, 1, 3, f)
    P = _point_list(points, f, lo=1, dim=2)
    concrete(P)
    flt = _float_mode(points, *rest)
    if rest:
        q = pt(rest[0], f, "centre", dim=2)
        concrete(q)
        if len(rest) > 1:
            r = _pos_side(rest[1], f, "radius")
            concrete(r)
        else:
            r = _norm(_sub(P[0], q))
        for p in P:
            if not _zero(_dot(_sub(p, q), _sub(p, q)) - r ** 2, flt):
                raise ValueError(f"{f}: the points do not lie on a circle with the given centre and radius")
        return Expr("Circle", out(q, points, *rest), out(r, points, *rest))
    if len(P) < 3:
        raise ValueError(f"{f}: at least three points (or a centre) are needed to determine a circle")
    c, r = _sphere_through(P, f, (points,))
    return Expr("Circle", out(c, points), out(r, points))


@fn("CircularArcThrough", area=AREA, status="added",
    notes="CircularArcThrough[{p1, p2, ..., pk}] (k >= 3 concyclic 2-D points in order along the arc) gives "
          "Circle[c, r, {t1, t2}] with t1 < t2, the counterclockwise arc from p1 to pk passing through the others "
          "(from pk to p1 when the points run clockwise); t1 is normalised to (-Pi, Pi]. Points that are not "
          "concyclic, not in arc order, or fewer than three raise ValueError; the centre/radius forms "
          "CircularArcThrough[pts, q(, r)] raise NotImplementedError; symbolic points stay unevaluated. "
          "Deviation: Wolfram may choose a different but equivalent angle branch. "
          "Check: self-check (all points lie on the circle within the angle range).")
def _circular_arc_through(points: MatrixOf[Real], *rest: ListOf[Real] | Real) -> Expr:
    f = "CircularArcThrough"
    if rest:
        raise NotImplementedError(f"{f}: the forms with a given centre or radius are not supported")
    P = _point_list(points, f, lo=1, dim=2)
    concrete(P)
    if len(P) < 3:
        raise ValueError(f"{f}: at least three points are needed to determine an arc")
    c, r = _sphere_through(P, f, (points,))
    angs = [sp.atan2(p[1] - c[1], p[0] - c[0]) for p in P]
    # unwrap counterclockwise from p1
    a0 = angs[0]
    rel = []
    for a in angs:
        d = a - a0
        while fsign(d) < 0:
            d += 2 * sp.pi
        while fsign(d - 2 * sp.pi) >= 0:
            d -= 2 * sp.pi
        rel.append(d)
    nums = [num(x) for x in rel]
    if all(nums[i] < nums[i + 1] for i in range(len(nums) - 1)):
        t1, t2 = a0, a0 + rel[-1]
    else:
        back = [(2 * math.pi - x) % (2 * math.pi) for x in nums]
        if not all(back[i] < back[i + 1] for i in range(len(back) - 1)):
            raise ValueError(f"{f}: the points are not in order along a circular arc")
        t1 = angs[-1]
        t2 = t1 + (2 * sp.pi - rel[-1])
    t1, t2 = sp.simplify(t1), sp.simplify(t2)
    return Expr("Circle", out(c, points), out(r, points), out([t1, t2], points))


def _min_ball_numeric(Pn):
    """Welzl's algorithm (move-to-front, floats) -> support point indices."""
    import numpy as np
    pts_ = [np.array(p, dtype=float) for p in Pn]

    def ball(R):
        if not R:
            return np.zeros(len(Pn[0])), -1.0
        A = [pts_[i] for i in R]
        p0 = A[0]
        V = [a - p0 for a in A[1:]]
        if not V:
            return p0, 0.0
        G = np.array([[u @ w for w in V] for u in V])
        rhs = np.array([u @ u / 2 for u in V])
        lam = np.linalg.lstsq(G, rhs, rcond=None)[0]
        c = p0 + sum(l_ * u for l_, u in zip(lam, V))
        return c, float(np.linalg.norm(c - p0))

    def inside(c, r, i):
        return r >= 0 and np.linalg.norm(pts_[i] - c) <= r * (1 + 1e-9) + 1e-12

    def welzl(Pidx, R):
        if not Pidx or len(R) == len(Pn[0]) + 1:
            return R
        p = Pidx[0]
        S = welzl(Pidx[1:], R)
        c, r = ball(S)
        if inside(c, r, p):
            return S
        return welzl(Pidx[1:], R + [p])

    idx = list(range(len(pts_)))
    _random.Random(1).shuffle(idx)
    import sys
    lim = sys.getrecursionlimit()
    sys.setrecursionlimit(max(lim, 10 * len(idx) + 1000))
    try:
        return welzl(idx, [])
    finally:
        sys.setrecursionlimit(lim)


@fn("CircumscribedBall", area=AREA, status="added",
    notes="CircumscribedBall[{p1, p2, ...}] gives Ball[c, r], the smallest ball containing the points (minimal "
          "enclosing circle in 2-D), any dimension: Welzl's algorithm selects the support points numerically and "
          "the ball is then computed exactly from them (exact input gives exact c and r). Symbolic points stay "
          "unevaluated; strings raise TypeError, mixed dimensions ValueError; more than 2000 points raise "
          "NotImplementedError. Check: self-check (every point lies in the ball; ArithmeticError).")
def _circumscribed_ball(points: MatrixOf[Real]) -> Expr:
    f = "CircumscribedBall"
    P = _point_list(points, f, lo=1)
    concrete(P)
    if len(P) > 2000:
        raise NotImplementedError(f"{f}: more than 2000 points are not supported")
    flt = _float_mode(points)
    Pn = [[num(c) for c in p] for p in P]
    S = _min_ball_numeric(Pn)
    sup = [P[i] for i in S]
    c, r2 = _circumcenter_affine(sup, flt) if len(sup) > 1 else (list(sup[0]), sp.Integer(0))
    rn = math.sqrt(max(num(r2), 0.0))
    for p in Pn:
        if math.dist(p, [num(x) for x in c]) > rn * (1 + 1e-8) + 1e-10:
            raise ArithmeticError(f"{f}: result failed verification")
    return Expr("Ball", out(c, points), out(sp.sqrt(r2), points))


@fn("InscribedBall", area=AREA, status="added",
    notes="InscribedBall[{p1, p2, ...}] gives Ball[c, r], the largest ball inside the convex hull of the points "
          "(Chebyshev centre), in 2-D and 3-D: the hull (SciPy Qhull) and a linear program (SciPy HiGHS) select the "
          "touching facets, then c and r are solved exactly from them (exact input gives exact c, r). When the "
          "largest ball is not unique (e.g. a long rectangle) the LP's centre is returned, computed in machine "
          "numbers. Degenerate hulls raise ValueError, other dimensions NotImplementedError, symbolic points stay "
          "unevaluated; SciPy is required (ImportError otherwise). "
          "Check: self-check (the ball lies inside every hull facet; ArithmeticError).")
def _inscribed_ball(points: MatrixOf[Real]) -> Expr:
    f = "InscribedBall"
    P = _point_list(points, f, lo=1)
    concrete(P)
    d = len(P[0])
    if d not in (2, 3):
        raise NotImplementedError(f"{f}: only 2-D and 3-D points are supported")
    import numpy as np
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    Pn = np.array([[num(c) for c in p] for p in P])
    try:
        hull = ConvexHull(Pn)
    except Exception:
        raise ValueError(f"{f}: the convex hull of the points is degenerate")
    eqs = np.unique(np.round(hull.equations, 12), axis=0)
    A = eqs[:, :d]
    b = -eqs[:, d]
    res = linprog(np.r_[np.zeros(d), -1.0], A_ub=np.c_[A, np.ones(len(A))], b_ub=b,
                  bounds=[(None, None)] * d + [(0, None)], method="highs")
    if not res.success:
        raise ArithmeticError(f"{f}: the linear program failed")
    xc, rr = res.x[:d], res.x[d]
    slack = b - A @ xc - rr
    active = np.where(np.abs(slack) < 1e-7 * max(1.0, abs(rr)))[0]
    flt = _float_mode(points)
    exact_ok = not flt
    planes = []
    if exact_ok:
        for k in active:
            simplex = None
            for s_, eq in zip(hull.simplices, hull.equations):
                if np.allclose(eq, eqs[k], atol=1e-9):
                    simplex = s_
                    break
            if simplex is None:
                exact_ok = False
                break
            V = [P[i] for i in simplex]
            if d == 2:
                e = _sub(V[1], V[0])
                n_ = [e[1], -e[0]]
            else:
                n_ = _cross(_sub(V[1], V[0]), _sub(V[2], V[0]))
            if float(np.dot([num(x) for x in n_], eqs[k][:d])) < 0:
                n_ = _mul(-1, n_)
            nn = _norm(n_)
            planes.append(([x / nn for x in n_], _dot(n_, V[0]) / nn))
    if exact_ok and len(planes) >= d + 1:
        M = sp.Matrix([list(n_) + [1] for n_, _ in planes])
        if M.rank(simplify=True) == d + 1:
            sol = (M.T * M).LUsolve(M.T * sp.Matrix([b_ for _, b_ in planes]))
            c = [sp.simplify(x) for x in sol[:d]]
            r = sp.simplify(sol[d])
            for n_, b_ in planes:
                if abs(num(_dot(n_, c) + r - b_)) > 1e-8:
                    exact_ok = False
            if exact_ok:
                _check_ball_inside(A, b, [num(x) for x in c], num(r), f)
                return Expr("Ball", out(c, points), out(r, points))
    _check_ball_inside(A, b, xc, rr, f)
    return Expr("Ball", L(*[float(x) for x in xc]), float(rr))


def _check_ball_inside(A, b, c, r, f):
    import numpy as np
    if np.any(A @ np.array(c, dtype=float) + r > b + 1e-7 * max(1.0, abs(r))):
        raise ArithmeticError(f"{f}: result failed verification")


# ───────────────────────────── polygon data ─────────────────────────────

_POLYGON_HEADS = ("Polygon", "Triangle", "Rectangle", "RegularPolygon", "Parallelogram", "Simplex")


def _regular_polygon_pts(a, f):
    n = a[-1]
    if not K.is_int(n) or isinstance(n, bool):
        if _numberish(n) or _bad_kind(n):
            raise ValueError(f"{f}: RegularPolygon needs an integer number of vertices")
        raise Unevaluated()
    n = int(n)
    if n < 3:
        raise ValueError(f"{f}: RegularPolygon needs at least 3 vertices")
    c, r, t1 = [sp.Integer(0), sp.Integer(0)], sp.Integer(1), sp.pi / n - sp.pi / 2
    rest = list(a[:-1])
    if len(rest) == 2:
        c = pt(rest[0], f, "centre", dim=2)
        rest = rest[1:]
    if rest:
        if K.is_list(rest[0]):
            r, t1 = sx(rest[0].args[0], f, "radius"), sx(rest[0].args[1], f, "angle")
        else:
            r = sx(rest[0], f, "radius")
    return [[c[0] + r * sp.cos(t1 + 2 * sp.pi * k / n), c[1] + r * sp.sin(t1 + 2 * sp.pi * k / n)] for k in range(n)]


def polygon_components(x, f):
    """[(outer ring, [hole rings])] of a polygon-like region (rings as lists of SymPy points)."""
    if not isinstance(x, Expr) or not isinstance(x.head, str) or x.head not in _POLYGON_HEADS:
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: expected a polygon (Polygon, Triangle, Rectangle, RegularPolygon, Parallelogram), "
                        f"not {K_input(x)}")
    _check_args(x.head, list(x.args))
    h = x.head
    a, _opts = split_trailing_options(list(x.args))
    if h == "Triangle":
        if not a:
            return [([[sp.Integer(0), sp.Integer(0)], [sp.Integer(1), sp.Integer(0)], [sp.Integer(0), sp.Integer(1)]],
                     [])]
        if _is_coord_list(a[0]) and _is_coord_list(a[0].args[0]):
            return [c for t in a[0].args for c in polygon_components(Expr("Triangle", t), f)]
        return [(pts(a[0], f, n=3), [])]
    if h == "Simplex":
        P = pts(a[0], f) if K.is_list(a[0]) else None
        if P is None or len(P) != 3:
            raise NotImplementedError(f"{f}: only triangular simplices are supported as polygons")
        return [(P, [])]
    if h == "Rectangle":
        p = pt(a[0], f, "corner", dim=2) if a else [sp.Integer(0), sp.Integer(0)]
        q = pt(a[1], f, "corner", dim=2) if len(a) > 1 else [p[0] + 1, p[1] + 1]
        return [([[p[0], p[1]], [q[0], p[1]], [q[0], q[1]], [p[0], q[1]]], [])]
    if h == "Parallelogram":
        if not a:
            p, v1, v2 = [sp.Integer(0)] * 2, [sp.Integer(1), sp.Integer(0)], [sp.Integer(1), sp.Integer(1)]
        else:
            p = pt(a[0], f, "origin")
            v1, v2 = pts(a[1], f, "vectors", n=2)
        return [([p, _add(p, v1), _add(_add(p, v1), v2), _add(p, v2)], [])]
    if h == "RegularPolygon":
        return [(_regular_polygon_pts(a, f), [])]
    # Polygon
    if len(a) == 2:
        P = pts(a[0], f)
        return _index_components(a[1], P, f)
    x0 = a[0]
    if isinstance(x0, Expr) and x0.head == "Rule":
        outer = pts(x0.args[0], f)
        hs = x0.args[1]
        holes = [pts(hh, f) for hh in (hs.args if hs.args and _is_coord_list(hs.args[0]) else [hs])] \
            if hs.args else []
        return [(outer, holes)]
    if K.is_list(x0) and x0.args and (K.is_list(x0.args[0]) and _is_coord_list(x0.args[0]) or
                                     (isinstance(x0.args[0], Expr) and x0.args[0].head == "Rule")):
        return [c for part in x0.args for c in polygon_components(Expr("Polygon", part), f)]
    return [(pts(x0, f), [])]


def _index_components(d, P, f):
    def ring(r):
        if not K.is_list(r):
            raise TypeError(f"{f}: a face must be a list of vertex indices")
        out_ = []
        for i in r.args:
            if not K.is_int(i) or isinstance(i, bool) or not 1 <= int(i) <= len(P):
                raise ValueError(f"{f}: invalid vertex index {K_input(i)}")
            out_.append(P[int(i) - 1])
        return out_
    if isinstance(d, Expr) and d.head == "Rule":
        hs = d.args[1]
        holes = [ring(h) for h in (hs.args if hs.args and K.is_list(hs.args[0]) else [hs])] if hs.args else []
        return [(ring(d.args[0]), holes)]
    if K.is_list(d) and d.args and (K.is_list(d.args[0]) or (isinstance(d.args[0], Expr) and d.args[0].head == "Rule")):
        return [c for part in d.args for c in _index_components(part, P, f)]
    return [(ring(d), [])]


def polygon_data(x, f, dim=2):
    """All rings (outer rings and holes) of a polygon-like region; dim checks the embedding dimension."""
    comps = polygon_components(x, f)
    rings = []
    for o, hs in comps:
        rings.append(o)
        rings += hs
    for r in rings:
        if dim is not None and len(r[0]) != dim:
            raise NotImplementedError(f"{f}: only {dim}-dimensional polygons are supported")
    return rings


def _signed_area(R_):
    n = len(R_)
    return sum((R_[i][0] * R_[(i + 1) % n][1] - R_[(i + 1) % n][0] * R_[i][1] for i in range(n)),
               sp.Integer(0)) / 2


def _orient2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _dedup_ring(R_):
    out_ = []
    for p in R_:
        if not out_ or not all(fsign(a - b) == 0 for a, b in zip(p, out_[-1])):
            out_.append(p)
    while len(out_) > 1 and all(fsign(a - b) == 0 for a, b in zip(out_[0], out_[-1])):
        out_.pop()
    return out_


def _to_plane(R_, f):
    """2-D coordinates of a planar ring (3-D rings are projected onto their plane's dominant axes)."""
    d = len(R_[0])
    if d == 2:
        return R_
    if d != 3:
        raise NotImplementedError(f"{f}: only 2-D and 3-D polygons are supported")
    n_ = [sp.Integer(0)] * 3
    for i in range(len(R_)):
        n_ = _add(n_, _cross(R_[i], R_[(i + 1) % len(R_)]))
    if _is_zero_vec(n_):
        raise ValueError(f"{f}: the polygon is degenerate")
    for p in R_:
        if fsign(_dot(n_, _sub(p, R_[0]))) != 0:
            raise ValueError(f"{f}: the polygon is not planar")
    k = max(range(3), key=lambda i: abs(num(n_[i])))
    keep = [i for i in range(3) if i != k]
    sgn = 1 if num(n_[k]) > 0 else -1
    if (k == 1):
        sgn = -sgn
    return [[p[keep[0]], p[keep[1]]] if sgn > 0 else [p[keep[1]], p[keep[0]]] for p in R_]


def _segments_intersect(p1, p2, q1, q2, proper_only=False):
    d1, d2 = fsign(_orient2(q1, q2, p1)), fsign(_orient2(q1, q2, p2))
    d3, d4 = fsign(_orient2(p1, p2, q1)), fsign(_orient2(p1, p2, q2))
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    if proper_only:
        return False

    def on(a, b, c):
        return all(fsign(min(a[i], b[i]) - c[i]) <= 0 and fsign(c[i] - max(a[i], b[i])) <= 0 for i in range(2))
    return (d1 == 0 and on(q1, q2, p1)) or (d2 == 0 and on(q1, q2, p2)) or \
        (d3 == 0 and on(p1, p2, q1)) or (d4 == 0 and on(p1, p2, q2))


def _simple_ring(R_):
    n = len(R_)
    if n < 3:
        return False
    keys = {tuple(num(c) for c in p) for p in R_}
    if len(keys) != n:
        return False
    if fsign(_signed_area(R_)) == 0:
        return False
    for i in range(n):
        a, b = R_[i], R_[(i + 1) % n]
        for j in range(i + 1, n):
            c, d = R_[j], R_[(j + 1) % n]
            if j == i + 1 or (i == 0 and j == n - 1):
                # adjacent edges: only overlapping (collinear and folding back) is a violation
                shared = b if j == i + 1 else a
                other1 = a if j == i + 1 else b
                other2 = d if j == i + 1 else c
                if fsign(_orient2(shared, other1, other2)) == 0 and \
                        fsign(_dot(_sub(other1, shared), _sub(other2, shared))) > 0:
                    return False
                continue
            if _segments_intersect(a, b, c, d):
                return False
    return True


def _convex_ring(R_):
    R_ = _dedup_ring(R_)
    n = len(R_)
    if n < 3:
        return False
    sgn = 0
    for i in range(n):
        s = fsign(_orient2(R_[i], R_[(i + 1) % n], R_[(i + 2) % n]))
        if s == 0:
            continue
        if sgn == 0:
            sgn = s
        elif s != sgn:
            return False
    if sgn == 0:
        return False
    # total turning must be one revolution (rules out star-shaped self-intersecting rings)
    tot = 0.0
    for i in range(n):
        a, b, c = ([num(t) for t in R_[(i + k) % n]] for k in range(3))
        tot += math.atan2((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]),
                          (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]))
    return abs(abs(tot) - 2 * math.pi) < 1e-6


def _polygon_arg(poly, f):
    comps = polygon_components(poly, f)
    for o, hs in comps:
        concrete(o, hs)
    return comps


@fn("ConvexPolygonQ", area=AREA, status="added",
    notes="ConvexPolygonQ[poly] for Polygon (all forms), Triangle, Rectangle, RegularPolygon, Parallelogram: True for "
          "a single boundary without holes whose turns all have the same sign and whose total turning is one "
          "revolution (planar 3-D polygons are projected onto their plane); polygons with holes or several "
          "components give False; collinear vertices are allowed. Non-planar 3-D polygons raise ValueError, "
          "non-polygons TypeError; symbolic coordinates stay unevaluated. "
          "Check: exact (exact orientation signs; tolerance 1e-12 for machine numbers).")
def _convex_polygon_q(poly: Expr) -> Boolean:
    f = "ConvexPolygonQ"
    comps = _polygon_arg(poly, f)
    if len(comps) != 1 or comps[0][1]:
        return False
    return _convex_ring(_to_plane(_dedup_ring(comps[0][0]), f))


@fn("SimplePolygonQ", area=AREA, status="added",
    notes="SimplePolygonQ[poly]: True when the polygon has a single boundary, no holes, at least three distinct "
          "vertices, non-zero area and no two non-adjacent edges touching or crossing (adjacent edges may not fold "
          "back); planar 3-D polygons are projected onto their plane. Non-polygons raise TypeError; symbolic "
          "coordinates stay unevaluated. Check: exact (exact segment-intersection predicates).")
def _simple_polygon_q(poly: Expr) -> Boolean:
    f = "SimplePolygonQ"
    comps = _polygon_arg(poly, f)
    if len(comps) != 1 or comps[0][1]:
        return False
    R_ = _dedup_ring(comps[0][0])
    try:
        R_ = _to_plane(R_, f)
    except ValueError:
        return False
    return _simple_ring(R_)


def _ring_angles(R_, f):
    """Interior angles of a simple ring (any orientation)."""
    s = fsign(_signed_area(R_))
    n = len(R_)
    res = []
    for i in range(n):
        a, b, c = R_[(i - 1) % n], R_[i], R_[(i + 1) % n]
        u, v = _sub(a, b), _sub(c, b)
        # counterclockwise angle from (c - b) to (a - b) for a counterclockwise ring is the interior angle
        ang = sp.atan2(v[0] * u[1] - v[1] * u[0], _dot(u, v)) if s > 0 else \
            sp.atan2(u[0] * v[1] - u[1] * v[0], _dot(u, v))
        if fsign(ang) < 0:
            ang = ang + 2 * sp.pi
        res.append(nice_angle(sp.simplify(ang)))
    return res


@fn("PolygonAngle", area=AREA, status="added",
    notes="PolygonAngle[poly] gives the interior angles at the vertices of a simple polygon (in vertex order), "
          "PolygonAngle[poly, p] the angle at vertex p, PolygonAngle[poly, i] at the i-th point of the coordinate "
          "list (Polygon[pts, data] form: index into pts); spec \"Interior\" (default), \"Exterior\" (Pi - a), "
          "\"FullExterior\" (2 Pi - a). Reflex angles exceed Pi. Planar 3-D polygons are projected onto their "
          "plane. Polygons with holes, several components or self-intersections raise NotImplementedError; a point "
          "that is not a vertex or an index out of range ValueError; symbolic coordinates stay unevaluated. "
          "Check: exact; the identity Sum[angles] == (n - 2) Pi in tests.")
def _polygon_angle(poly: Expr, *rest: ListOf[Real] | PosInteger | String) -> Real | ListOf[Real]:
    f = "PolygonAngle"
    _nargs((poly,) + rest, 1, 3, f)
    rest = list(rest)
    spec = "Interior"
    if rest and isinstance(rest[-1], str):
        spec = _string_arg(rest.pop(), ("Interior", "Exterior", "FullExterior"), f)
    comps = _polygon_arg(poly, f)
    if len(comps) != 1 or comps[0][1]:
        raise NotImplementedError(f"{f}: polygons with holes or several components are not supported")
    R0 = comps[0][0]
    R_ = _to_plane(R0, f)
    if not _simple_ring(R_):
        raise NotImplementedError(f"{f}: self-intersecting or degenerate polygons are not supported")
    angs = _ring_angles(R_, f)
    tr = {"Interior": lambda a: a, "Exterior": lambda a: sp.pi - a, "FullExterior": lambda a: 2 * sp.pi - a}[spec]
    if not rest:
        return out([tr(a) for a in angs], poly)
    sel = rest[0]
    if K.is_int(sel) and not isinstance(sel, bool):
        coords = _coord_list_of(poly, f)
        i = int(sel)
        if not 1 <= i <= len(coords):
            raise ValueError(f"{f}: index {i} is out of range 1..{len(coords)}")
        target = coords[i - 1]
    else:
        target = pt(sel, f, "vertex")
    for k, p in enumerate(R0):
        if len(p) == len(target) and all(fsign(a - b) == 0 for a, b in zip(p, target)):
            return out(tr(angs[k]), poly)
    raise ValueError(f"{f}: {K_input(sel)} is not a vertex of the polygon")


def _coord_list_of(poly, f):
    if poly.head == "Polygon":
        a, _ = split_trailing_options(list(poly.args))
        if len(a) == 2:
            return pts(a[0], f)
    return polygon_data(poly, f, dim=None)[0]


def _sorted_unique(points):
    seen, res = set(), []
    for p in points:
        key = tuple(sp.srepr(sp.nsimplify(c) if False else c) for c in p)
        if key in seen:
            continue
        dup = False
        for q in res:
            if all(fsign(a - b) == 0 for a, b in zip(p, q)):
                dup = True
                break
        if not dup:
            seen.add(key)
            res.append(p)
    return sorted(res, key=lambda p: [num(c) for c in p])


@fn("PolygonCoordinates", area=AREA, status="added",
    notes="PolygonCoordinates[poly] gives the distinct vertex coordinates of a polygon (Polygon in all forms, "
          "Triangle, Rectangle, RegularPolygon, Parallelogram), in Sort order as in Wolfram's canonical form. "
          "Self-intersecting polygons (whose intersection points Wolfram adds) raise NotImplementedError. "
          "RegularPolygon[n] has its vertices at CirclePoints[n] (horizontal base). Non-polygons raise TypeError; "
          "symbolic coordinates stay unevaluated. Check: exact; reference values in tests.")
def _polygon_coordinates(poly: Expr) -> MatrixOf[Real]:
    f = "PolygonCoordinates"
    comps = _polygon_arg(poly, f)
    _require_noncrossing(comps, f)
    allp = [p for o, hs in comps for r in [o] + hs for p in r]
    return out(_sorted_unique(allp), poly)


def _require_noncrossing(comps, f):
    rings = [_dedup_ring(r) for o, hs in comps for r in [o] + hs]
    if any(len(r[0]) != 2 for r in rings):
        raise NotImplementedError(f"{f}: only 2-D polygons are supported")
    for r in rings:
        if not _simple_ring(r):
            raise NotImplementedError(f"{f}: self-intersecting or degenerate polygons are not supported")
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            A, B = rings[i], rings[j]
            for k in range(len(A)):
                for m in range(len(B)):
                    if _segments_intersect(A[k], A[(k + 1) % len(A)], B[m], B[(m + 1) % len(B)]):
                        raise NotImplementedError(f"{f}: polygons whose boundaries touch or cross are not supported")
    return rings


def _point_in_ring(p, R_):
    """Winding number of the closed ring around p (exact predicates); None if p is on the ring."""
    wn = 0
    n = len(R_)
    for i in range(n):
        a, b = R_[i], R_[(i + 1) % n]
        o = fsign(_orient2(a, b, p))
        if o == 0 and all(fsign(min(a[k], b[k]) - p[k]) <= 0 and fsign(p[k] - max(a[k], b[k])) <= 0 for k in range(2)):
            return None
        if fsign(a[1] - p[1]) <= 0:
            if fsign(b[1] - p[1]) > 0 and o > 0:
                wn += 1
        elif fsign(b[1] - p[1]) <= 0 and o < 0:
            wn -= 1
    return wn


def _nesting(rings):
    """Depth (number of rings containing it) of each ring, for mutually non-crossing simple rings."""
    depth = []
    for i, r in enumerate(rings):
        d = 0
        for j, s in enumerate(rings):
            if i != j and _point_in_ring(r[0], s) not in (None, 0):
                d += 1
        depth.append(d)
    return depth


def _orient_ring(R_, ccw=True):
    s = fsign(_signed_area(R_))
    return R_ if (s > 0) == ccw else R_[::-1]


def _canonical(comps_rings, keep, f, inputs, empty_if_none=False):
    """Polygon[pts, {outer -> {holes}, ...}] from rings with fill flags; points in Sort order."""
    rings = [r for r in comps_rings]
    depth = _nesting(rings)
    filled = keep(rings, depth)
    # filled region = union over rings with fill parity; build outer rings (filled inside, parent not filled)
    outers = [i for i in range(len(rings)) if filled[i]]
    if not outers:
        if empty_if_none:
            return Expr("EmptyRegion", 2)
        raise ValueError(f"{f}: the polygon is empty")
    allp = _sorted_unique([p for i in range(len(rings)) for p in rings[i]])

    def idx(p):
        for k, q in enumerate(allp):
            if all(fsign(a - b) == 0 for a, b in zip(p, q)):
                return k + 1
        raise ArithmeticError(f"{f}: internal error")

    used = set()
    parts = []
    for i in outers:
        holes = [j for j in range(len(rings)) if depth[j] == depth[i] + 1 and not filled[j]
                 and _point_in_ring(rings[j][0], rings[i]) not in (None, 0)]
        o = [idx(p) for p in _orient_ring(rings[i], True)]
        hs = [[idx(p) for p in _orient_ring(rings[j], False)] for j in holes]
        used.update(o)
        for h in hs:
            used.update(h)
        parts.append(Expr("Rule", L(*o), L(*[L(*h) for h in hs])) if hs else L(*o))
    order = sorted(used)
    remap = {old: new + 1 for new, old in enumerate(order)}

    def re(x):
        if isinstance(x, Expr):
            return Expr(x.head, *[re(a) for a in x.args])
        return remap[x]
    parts = [re(p) for p in parts]
    P = [allp[k - 1] for k in order]
    data = parts[0] if len(parts) == 1 else L(*parts)
    return Expr("Polygon", out(P, *inputs), data)


def _rings_of(poly, f, allow_list=True):
    if K.is_list(poly):
        P = pts(poly, f, lo=3) if not (poly.args and _is_coord_list(poly.args[0])) else None
        rings = [P] if P is not None else [pts(r, f, lo=3) for r in poly.args]
        comps = [(r, []) for r in rings]
    else:
        comps = polygon_components(poly, f)
    for o, hs in comps:
        concrete(o, hs)
    return comps


def _rings_for_rule(comps, f):
    rings = [_dedup_ring(r) for o, hs in comps for r in [o] + hs]
    if any(len(r[0]) != 2 for r in rings):
        raise NotImplementedError(f"{f}: only 2-D polygons are supported")
    _require_noncrossing([(r, []) for r in rings], f)
    return rings


@fn("CanonicalizePolygon", area=AREA, status="added",
    notes="CanonicalizePolygon[poly] (2-D Polygon in all forms, Triangle, Rectangle, RegularPolygon, Parallelogram) "
          "gives Polygon[{p1, ...}, data] with the distinct points in Sort order and data the outer boundaries "
          "(counterclockwise) with their holes (clockwise) as index lists (outer -> {holes} where present); the "
          "filled set follows the polygon's own rule (Polygon[outer -> holes] and collections are unions of "
          "components minus holes). Filter \"Full\" (default) or All. Polygons whose boundaries cross or touch "
          "raise NotImplementedError (Wolfram splits them at the intersections). Deviation: the starting vertex of "
          "each boundary may differ from Wolfram's (same region). Symbolic coordinates stay unevaluated. "
          "Check: exact; the area of the result equals the input's in tests.")
def _canonicalize_polygon(poly: Expr, *flt: String) -> Expr:
    f = "CanonicalizePolygon"
    if flt:
        _string_arg(flt[0], ("Full", "All"), f, "filter")
    comps = _polygon_arg(poly, f)
    for o, hs in comps:
        if len(o[0]) != 2:
            raise NotImplementedError(f"{f}: only 2-D polygons are supported")
    _require_noncrossing(comps, f)
    rings, fill = [], []
    for o, hs in comps:
        rings.append(_dedup_ring(o))
        fill.append(True)
        for h in hs:
            rings.append(_dedup_ring(h))
            fill.append(False)
    return _canonical(rings, lambda rs, d: fill, f, (poly,))


@fn("OuterPolygon", area=AREA, status="added",
    notes="OuterPolygon[poly] gives Polygon[{p1, ...}, data] of the outer boundaries of the canonical form "
          "(CanonicalizePolygon) with the holes removed. Same support and limitations as CanonicalizePolygon. "
          "Check: exact; Area[OuterPolygon] - Area[InnerPolygon] == Area[poly] relations in tests.")
def _outer_polygon(poly: Expr) -> Expr:
    f = "OuterPolygon"
    comps = _polygon_arg(poly, f)
    _require_noncrossing(comps, f)
    rings = [_dedup_ring(o) for o, hs in comps]
    return _canonical(rings, lambda rs, d: [True] * len(rs), f, (poly,))


@fn("InnerPolygon", area=AREA, status="added",
    notes="InnerPolygon[poly] gives Polygon[{p1, ...}, data] of the holes of the canonical form (each hole as a "
          "counterclockwise outer boundary), or EmptyRegion[2] when there are none. Same support and limitations "
          "as CanonicalizePolygon. Check: exact; reference cases in tests.")
def _inner_polygon(poly: Expr) -> Expr:
    f = "InnerPolygon"
    comps = _polygon_arg(poly, f)
    _require_noncrossing(comps, f)
    rings = [_dedup_ring(h) for o, hs in comps for h in hs]
    if not rings:
        return Expr("EmptyRegion", 2)
    return _canonical(rings, lambda rs, d: [x == 0 for x in d], f, (poly,), empty_if_none=True)


def _contour_rings(contour, f):
    """Closed rings of a contour: Line[{p1, ..., p1}] / Line[{{...}, ...}] / Polygon-like / point list."""
    if isinstance(contour, Expr) and contour.head == "Line" and contour.args:
        x = contour.args[0]
        parts = x.args if (K.is_list(x) and x.args and _is_coord_list(x.args[0])) else [x]
        rings = []
        for part in parts:
            P = pts(part, f, lo=2)
            if len(P) == 2:                           # a segment list {{p, q}, ...} is joined below
                rings.append(P)
                continue
            rings.append(P)
        if parts is not None and all(len(r) == 2 for r in rings) and len(rings) > 1:
            # Line[{{p1, p2}, {p2, p3}, ...}]: chain segments
            chain = [rings[0][0]]
            for s in rings:
                chain.append(s[1])
            rings = [chain]
        out_ = []
        for r in rings:
            if not all(fsign(a - b) == 0 for a, b in zip(r[0], r[-1])):
                raise ValueError(f"{f}: the contour must be closed (Line[{{p1, ..., p1}}])")
            out_.append(r[:-1])
        return out_
    if isinstance(contour, Expr) and isinstance(contour.head, str) and contour.head in _POLYGON_HEADS:
        return [r for o, hs in polygon_components(contour, f) for r in [o] + hs]
    if K.is_list(contour):
        return [pts(contour, f, lo=3)]
    if isinstance(contour, Symbol):
        raise Unevaluated()
    raise TypeError(f"{f}: expected a closed Line contour or a polygon, not {K_input(contour)}")


def _query_points(p, f):
    if K.is_list(p) and p.args and K.is_list(p.args[0]):
        return [pt(q, f, "point", dim=2) for q in p.args], True
    return [pt(p, f, "point", dim=2)], False


@fn("WindingCount", area=AREA, status="added",
    notes="WindingCount[contour, p]: the number of counterclockwise revolutions of the closed contour around p, summed "
          "over its closed curves; contour is a closed Line[{p1, ..., p1}] (or several), a list of segments, or a "
          "polygon (its boundary rings as given); p may be a list of points (list result). 2-D only (else "
          "ValueError); a point on the contour raises ValueError; symbolic input stays unevaluated. "
          "Check: exact (exact orientation predicates); reference values (documentation star, nested rings) in "
          "tests.")
def _winding_count(contour: Expr, p: ListOf[Real] | MatrixOf[Real]) -> Integer | ListOf[Integer]:
    f = "WindingCount"
    rings = _contour_rings(contour, f)
    Q, many = _query_points(p, f)
    concrete(rings, Q)
    if any(len(r[0]) != 2 for r in rings):
        raise ValueError(f"{f}: only planar 2-D contours are supported")
    res = []
    for q in Q:
        tot = 0
        for r in rings:
            w = _point_in_ring(q, r)
            if w is None:
                raise ValueError(f"{f}: the point {K_input(out(q, p))} lies on the contour")
            tot += w
        res.append(tot)
    return L(*res) if many else res[0]


def _ray_crossings(q, rings, direction):
    cnt = 0
    for r in rings:
        n = len(r)
        for i in range(n):
            a, b = r[i], r[(i + 1) % n]
            ax, ay, bx, by = a[0] - q[0], a[1] - q[1], b[0] - q[0], b[1] - q[1]
            dx, dy = direction
            den = (bx - ax) * dy - (by - ay) * dx
            if abs(den) < 1e-300:
                continue
            t = -(ax * dy - ay * dx) / den           # parameter along the segment
            if 0 <= t < 1:
                px, py = ax + t * (bx - ax), ay + t * (by - ay)
                if px * dx + py * dy > 0:
                    cnt += 1
    return cnt


@fn("CrossingCount", area=AREA, status="added",
    notes="CrossingCount[contour, p]: the minimal number of crossings of a ray from p with the closed contour, over "
          "all ray directions (directions are sampled between consecutive vertex directions as seen from p, so "
          "every combinatorially distinct ray is tried; the crossings are counted in machine arithmetic); contour "
          "and p as for WindingCount (p may be a list of points). A point on the contour raises ValueError, non-2-D "
          "input ValueError; symbolic input stays unevaluated. Check: reference values (documentation examples) and "
          "parity agreement with WindingCount for simple contours in tests.")
def _crossing_count(contour: Expr, p: ListOf[Real] | MatrixOf[Real]) -> Integer | ListOf[Integer]:
    f = "CrossingCount"
    rings = _contour_rings(contour, f)
    Q, many = _query_points(p, f)
    concrete(rings, Q)
    if any(len(r[0]) != 2 for r in rings):
        raise ValueError(f"{f}: only planar 2-D contours are supported")
    rn = [[[num(c) for c in v] for v in r] for r in rings]
    res = []
    for q in Q:
        for r in rings:
            if _point_in_ring(q, r) is None:
                raise ValueError(f"{f}: the point {K_input(out(q, p))} lies on the contour")
        qn = [num(c) for c in q]
        angs = sorted({math.atan2(v[1] - qn[1], v[0] - qn[0]) for r in rn for v in r})
        cands = [(angs[i] + angs[(i + 1) % len(angs)] + (2 * math.pi if i == len(angs) - 1 else 0)) / 2
                 for i in range(len(angs))] or [0.123]
        best = min(_ray_crossings(qn, rn, (math.cos(t), math.sin(t))) for t in cands)
        res.append(best)
    return L(*res) if many else res[0]


_WINDING_RULES = ("NonzeroRule", "PositiveRule", "NegativeRule", "OddRule", "EvenOddRule", "AbsGeqTwoRule")


def _winding_fill(rule):
    return {"NonzeroRule": lambda w: w != 0, "PositiveRule": lambda w: w > 0, "NegativeRule": lambda w: w < 0,
            "OddRule": lambda w: w % 2 == 1, "EvenOddRule": lambda w: w % 2 == 1,
            "AbsGeqTwoRule": lambda w: abs(w) >= 2}[rule]


def _fill_by_rule(rings, depth, rule, oriented):
    """Fill flag of the face just inside each ring: winding number there decides."""
    flags = []
    for i, r in enumerate(rings):
        # winding number just inside ring i = sum of orientations of rings containing it plus its own
        w = 0
        for j, s in enumerate(rings):
            if j == i or _point_in_ring(r[0], s) not in (None, 0):
                w += (1 if fsign(_signed_area(s)) > 0 else -1) if oriented else 1
        outside = w - ((1 if fsign(_signed_area(r)) > 0 else -1) if oriented else 1)
        flags.append((_winding_fill(rule)(w), _winding_fill(rule)(outside)))
    return flags


def _rule_polygon(spec, rule, f, oriented):
    comps = _rings_of(spec, f)
    rings = _rings_for_rule(comps, f)
    flags = _fill_by_rule(rings, None, rule, oriented)
    # a ring is an outer boundary when inside is filled and outside is not; a hole when the reverse
    keep_rings = [r for r, (fi, fo) in zip(rings, flags) if fi != fo]
    fill = [fi for (fi, fo) in flags if fi != fo]
    if not keep_rings:
        return Expr("EmptyRegion", 2)
    return _canonical(keep_rings, lambda rs, d: fill, f, (spec,))


@fn("WindingPolygon", area=AREA, status="added",
    notes="WindingPolygon[{p1, ...}] and WindingPolygon[{{p11, ...}, ...}] give the polygon of points around which "
          "the closed contours wind (\"NonzeroRule\", default) or by a winding rule \"PositiveRule\", "
          "\"NegativeRule\", \"OddRule\", \"EvenOddRule\", \"AbsGeqTwoRule\", in the canonical form of "
          "CanonicalizePolygon (EmptyRegion[2] when nothing is filled). 2-D contours that do not cross or touch "
          "each other or themselves are supported; crossing contours raise NotImplementedError (Wolfram splits "
          "them). Unknown rules raise ValueError; symbolic coordinates stay unevaluated. "
          "Check: exact; reference cases (nested contours with either orientation) in tests.")
def _winding_polygon(spec: MatrixOf[Real] | ListOf[MatrixOf[Real]], *rule: String) -> Expr:
    f = "WindingPolygon"
    r = _string_arg(rule[0], _WINDING_RULES, f, "winding rule") if rule else "NonzeroRule"
    if len(rule) > 1:
        raise ValueError(f"{f}: too many arguments")
    return _rule_polygon(spec, r, f, oriented=True)


@fn("CrossingPolygon", area=AREA, status="added",
    notes="CrossingPolygon[{p1, ...}] and CrossingPolygon[{{p11, ...}, ...}] give the polygon of points from which a "
          "ray crosses the closed contours an odd number of times (even-odd rule), in the canonical form of "
          "CanonicalizePolygon. 2-D contours that do not cross or touch are supported; crossing contours raise "
          "NotImplementedError. Symbolic coordinates stay unevaluated. "
          "Check: exact; reference cases (nested contours give holes) in tests.")
def _crossing_polygon(spec: MatrixOf[Real] | ListOf[MatrixOf[Real]]) -> Expr:
    return _rule_polygon(spec, "EvenOddRule", "CrossingPolygon", oriented=False)


def _triangulate(R_, f):
    """Ear clipping of a simple ring (CCW) -> list of index triples."""
    R_ = _orient_ring(R_, True)
    idx = list(range(len(R_)))
    tris = []
    guard = 0
    while len(idx) > 3:
        guard += 1
        if guard > 10 * len(R_) ** 2:
            raise ArithmeticError(f"{f}: triangulation failed")
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = R_[i0], R_[i1], R_[i2]
            if fsign(_orient2(a, b, c)) <= 0:
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                q = R_[j]
                if fsign(_orient2(a, b, q)) >= 0 and fsign(_orient2(b, c, q)) >= 0 and fsign(_orient2(c, a, q)) >= 0:
                    ok = False
                    break
            if ok:
                tris.append([R_[i0], R_[i1], R_[i2]])
                idx.pop(k)
                break
        else:
            # only collinear ears remain: drop a collinear vertex
            for k in range(len(idx)):
                if fsign(_orient2(R_[idx[k - 1]], R_[idx[k]], R_[idx[(k + 1) % len(idx)]])) == 0:
                    idx.pop(k)
                    break
            else:
                raise ArithmeticError(f"{f}: triangulation failed")
    tris.append([R_[i] for i in idx])
    return tris


def _merge_convex(tris):
    """Hertel-Mehlhorn: merge adjacent convex pieces while the union stays convex."""
    pieces = [list(t) for t in tris]
    changed = True
    while changed:
        changed = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                m = _merge_two(pieces[i], pieces[j])
                if m is not None and _convex_ring(m):
                    pieces[i] = m
                    pieces.pop(j)
                    changed = True
                    break
            if changed:
                break
    return pieces


def _same(p, q):
    return all(fsign(a - b) == 0 for a, b in zip(p, q))


def _merge_two(A, B):
    n, m = len(A), len(B)
    for i in range(n):
        a1, a2 = A[i], A[(i + 1) % n]
        for j in range(m):
            b1, b2 = B[j], B[(j + 1) % m]
            if _same(a1, b2) and _same(a2, b1):
                # A ... a1 | (B from b2's successor around to b1's predecessor) | a2 ...
                merged = A[:i + 1] + [B[(j + 2 + k) % m] for k in range(m - 2)] + A[i + 1:]
                return merged
    return None


@fn("PolygonDecomposition", area=AREA, status="added",
    notes="PolygonDecomposition[poly] / PolygonDecomposition[poly, \"Convex\"] gives Polygon[{piece1, ...}], a "
          "union of convex polygons with disjoint interiors (ear-clipping triangulation followed by Hertel-Mehlhorn "
          "merging); \"Triangle\" gives the triangulation; \"Simple\" gives the simple components. Supports 2-D "
          "simple polygons, collections of them (Triangle, Rectangle, RegularPolygon, Parallelogram too); holes or "
          "self-intersections raise NotImplementedError, unknown types ValueError. A convex input is returned as a "
          "single piece. Deviation: the decomposition is not unique and may differ from Wolfram's (same region). "
          "Check: self-check (the piece areas sum to the polygon area; ArithmeticError).")
def _polygon_decomposition(poly: Expr, *typ: String) -> Expr:
    f = "PolygonDecomposition"
    t = _string_arg(typ[0], ("Convex", "Triangle", "Simple"), f, "decomposition type") if typ else "Convex"
    comps = _polygon_arg(poly, f)
    pieces = []
    total = sp.Integer(0)
    for o, hs in comps:
        if hs:
            raise NotImplementedError(f"{f}: polygons with holes are not supported")
        R_ = _dedup_ring(o)
        if len(R_[0]) != 2:
            raise NotImplementedError(f"{f}: only 2-D polygons are supported")
        if not _simple_ring(R_):
            raise NotImplementedError(f"{f}: self-intersecting or degenerate polygons are not supported")
        total += sp.Abs(_signed_area(R_))
        if t == "Simple":
            pieces.append(R_)
            continue
        if t == "Convex" and _convex_ring(R_):
            pieces.append(R_)
            continue
        tris = _triangulate(R_, f)
        pieces += tris if t == "Triangle" else _merge_convex(tris)
    s = sum((sp.Abs(_signed_area(p)) for p in pieces), sp.Integer(0))
    if abs(num(s - total)) > 1e-9 * max(1.0, abs(num(total))):
        raise ArithmeticError(f"{f}: result failed verification")
    return Expr("Polygon", out(pieces, poly))
