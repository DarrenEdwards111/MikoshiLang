"""Wolfram batch W09 -- paths, space-filling curves and Mandelbrot/Julia set functions.

* AnglePath (2D turtle paths; Position, FrameAngle, RelativeAngle, FrameMatrix, RelativeMatrix forms).
* HilbertCurve (2D), PeanoCurve, KochCurve (the standard Koch curve) and SierpinskiCurve return Line[...] with the
  coordinates documented by the Wolfram Language (HilbertCurve[n] runs from {0, 0} to {2^n - 1, 0}; PeanoCurve[n]
  from {0, 0} to {3^n - 1, 3^n - 1}; KochCurve[n] from {0., 0.} to {1., 0.}; SierpinskiCurve[n] is Wirth's closed
  curve with diagonal steps {32, 32} and straight steps of length 64 starting at {0, 0}); DataRange rescales.
* MandelbrotSetIterationCount, MandelbrotSetMemberQ, MandelbrotSetDistance (exterior estimate),
  MandelbrotSetBoettcher and JuliaSetBoettcher iterate z -> z^2 + c (escape radius 2 for the counts; the Boettcher
  coordinates use a large escape radius and the product formula).  Exact arguments of the Boettcher functions stay
  unevaluated (N evaluates them), as in the Wolfram Language.
"""
from __future__ import annotations

import cmath
import math

import mpmath
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, record, Unevaluated, REGISTRY, S, R, L, is_list, has_float
from .wl_common import ev, split_options, is_sym
from . import wl_numbertheory2 as NT

AREA = "fractals and paths (W09)"
ADDED = dict(area=AREA, status="added")


def _intq(x):
    return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, sp.Integer)


def _nonneg_int(n, fname):
    if not _intq(n) or int(n) < 0:
        raise ValueError(f"{fname}: the number of steps must be a non-negative integer")
    return int(n)


# ─────────────────────────── AnglePath ───────────────────────────

_FORMS = ("Position", "FrameAngle", "RelativeAngle", "FrameMatrix", "RelativeMatrix")


def _num(x):
    return S(x) if not isinstance(x, (int, float, complex)) else sp.sympify(x)


def _rot(a):
    return L(L(R(sp.cos(a)), R(-sp.sin(a))), L(R(sp.sin(a)), R(sp.cos(a))))


@fn("AnglePath", **ADDED, notes="AnglePath[{t1, t2, ...}] / AnglePath[{{r1, t1}, ...}] turtle path from {0, 0} "
    "turning by ti then stepping ri (default 1); AnglePath[t0 | {x, y} | {{x, y}, t0} | {{x, y}, {dx, dy}}, steps]; "
    "AnglePath[init, steps, form] with form Position, FrameAngle, RelativeAngle, FrameMatrix, RelativeMatrix or a "
    "list; exact input gives exact (unsimplified) coordinates, approximate input machine numbers; "
    "WorkingPrecision -> MachinePrecision forces numbers. Transformation forms raise (no TransformationFunction)")
def _anglepath(*args: Expr) -> Expr:
    args, opts = split_options(args, {"WorkingPrecision"}, "AnglePath")
    if not 1 <= len(args) <= 3:
        raise ValueError("AnglePath takes 1 to 3 arguments")
    form = "Position"
    if len(args) == 3 or (len(args) == 2 and (isinstance(args[1], str) or
                                              (is_list(args[1]) and args[1].args and
                                               all(isinstance(f, str) for f in args[1].args)))):
        form = args[-1]
        args = args[:-1]
    if len(args) == 1:
        init, steps = None, args[0]
    else:
        init, steps = args
    if not is_list(steps):
        raise ValueError("AnglePath: the steps must be a list")
    x0, y0, a0 = sp.Integer(0), sp.Integer(0), sp.Integer(0)
    if init is not None:
        if is_list(init) and len(init.args) == 2 and is_list(init.args[0]):
            p, d = init.args
            if len(p.args) != 2:
                raise ValueError("AnglePath: invalid initial position")
            x0, y0 = _num(p.args[0]), _num(p.args[1])
            if is_list(d):
                if len(d.args) != 2:
                    raise ValueError("AnglePath: invalid initial direction")
                a0 = sp.atan2(_num(d.args[1]), _num(d.args[0]))
            else:
                a0 = _num(d)
        elif is_list(init):
            if len(init.args) != 2:
                raise ValueError("AnglePath: invalid initial position")
            x0, y0 = _num(init.args[0]), _num(init.args[1])
        else:
            a0 = _num(init)
    rs, ts = [], []
    for s in steps.args:
        if is_list(s):
            if len(s.args) != 2:
                raise ValueError("AnglePath: each step must be an angle or {r, angle}")
            rs.append(_num(s.args[0]))
            ts.append(_num(s.args[1]))
        else:
            rs.append(sp.Integer(1))
            ts.append(_num(s))
    numeric = has_float(steps) or (init is not None and has_float(init))
    wp = opts.get("WorkingPrecision")
    if wp is not None and not is_sym(wp, "Automatic"):
        if is_sym(wp, "MachinePrecision"):
            numeric = True
        else:
            raise NotImplementedError("AnglePath: only WorkingPrecision -> Automatic or MachinePrecision is supported")
    pts = [(x0, y0)]
    frames = [a0]
    x, y, a = x0, y0, a0
    for r, t in zip(rs, ts):
        a = a + t
        x = x + r * sp.cos(a)
        y = y + r * sp.sin(a)
        pts.append((x, y))
        frames.append(a)

    def out(v):
        if numeric:
            v = sp.N(v, 15)
            if v.is_number:
                c = complex(v)
                return c.real if abs(c.imag) < 1e-300 or c.imag == 0 else c
        return R(v)

    def one(f):
        if f == "Position":
            return L(*(L(out(px), out(py)) for px, py in pts))
        if f == "FrameAngle":
            return L(*(out(fa - a0) for fa in frames))
        if f == "RelativeAngle":
            return L(*([out(sp.Integer(0))] + [out(t) for t in ts]))
        if f == "FrameMatrix":
            return L(*(_rotm(fa - a0, out) for fa in frames))
        if f == "RelativeMatrix":
            return L(*([_rotm(sp.Integer(0), out)] + [_rotm(t, out) for t in ts]))
        if f in ("Translation", "Rotation", "RotationTranslation"):
            raise NotImplementedError(f"AnglePath: the form {f!r} needs TransformationFunction, which MikoshiLang "
                                      f"does not implement")
        raise ValueError(f"AnglePath: unknown form {f!r}")
    if is_list(form):
        return L(*(one(f) for f in form.args))
    return one(form)


def _rotm(a, out):
    return L(L(out(sp.cos(a)), out(-sp.sin(a))), L(out(sp.sin(a)), out(sp.cos(a))))


# ─────────────────────────── space-filling curves ───────────────────────────

def _data_range(pts, dr, fname, box=None):
    """Rescale points (tuples of numbers) to DataRange {{x0, x1}, {y0, y1}}; returns floats."""
    if not (is_list(dr) and len(dr.args) == 2 and all(is_list(r) and len(r.args) == 2 for r in dr.args)):
        raise ValueError(f"{fname}: DataRange must be {{{{xmin, xmax}}, {{ymin, ymax}}}}")
    rng = [(float(S(r.args[0])), float(S(r.args[1]))) for r in dr.args]
    if box is None:
        box = [(min(p[i] for p in pts), max(p[i] for p in pts)) for i in range(2)]
    out = []
    for p in pts:
        q = []
        for i in range(2):
            lo, hi = box[i]
            a, b = rng[i]
            q.append(a + (float(p[i]) - lo) * (b - a) / (hi - lo) if hi != lo else a)
        out.append(q)
    return out


def _line(pts, floats=False):
    if floats:
        return Expr("Line", L(*(L(*(float(c) for c in p)) for p in pts)))
    return Expr("Line", L(*(L(*p) for p in pts)))


def _curve_opts(args, fname):
    args, opts = split_options(args, {"DataRange"}, fname)
    return list(args), opts.get("DataRange")


def hilbert_points(n):
    N = 2 ** n
    pts = []
    for d in range(N * N):
        x = y = 0
        t = d
        s = 1
        while s < N:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        pts.append((x, y))
    return pts


@fn("HilbertCurve", **ADDED, notes="HilbertCurve[n] is Line[...] through all 4^n integer points of the "
    "(2^n - 1)-square, from {0, 0} to {2^n - 1, 0} (unit steps); HilbertCurve[n, 2] the same; other dimensions "
    "raise; DataRange -> {{x0, x1}, {y0, y1}} rescales")
def _hilbert(*args: Expr) -> Expr:
    args, dr = _curve_opts(args, "HilbertCurve")
    if not 1 <= len(args) <= 2:
        raise ValueError("HilbertCurve takes 1 or 2 arguments")
    n = _nonneg_int(args[0], "HilbertCurve")
    if len(args) == 2 and not (_intq(args[1]) and int(args[1]) == 2):
        raise NotImplementedError("HilbertCurve: only the 2D Hilbert curve is implemented")
    if n > 10:
        raise ValueError("HilbertCurve: n > 10 is too large")
    pts = hilbert_points(n)
    if dr is not None:
        return _line(_data_range(pts, dr, "HilbertCurve"), True)
    return _line(pts)


def peano_points(n):
    N = 3 ** n
    pts = []
    for d in range(N * N):
        # base-3 digits of d, most significant first, in pairs (x digit, y digit) of a serpentine order
        digs = []
        t = d
        for _ in range(2 * n):
            digs.append(t % 3)
            t //= 3
        digs.reverse()
        x = y = 0
        flip_x = flip_y = False
        for k in range(n):
            a, b = digs[2 * k], digs[2 * k + 1]
            # within a block: rows (by) of a serpentine; the x digit of the block is b' and y digit a
            by = a
            bx = b if by % 2 == 0 else 2 - b
            ex = 2 - bx if flip_x else bx
            ey = 2 - by if flip_y else by
            x = 3 * x + ex
            y = 3 * y + ey
            if bx % 2 == 1:
                flip_y = not flip_y
            if by % 2 == 1:
                flip_x = not flip_x
        pts.append((x, y))
    return pts


@fn("PeanoCurve", **ADDED, notes="PeanoCurve[n] is Line[...] through all 9^n integer points of the "
    "(3^n - 1)-square from {0, 0} to {3^n - 1, 3^n - 1} (PeanoCurve[1] = Line[{{0, 0}, {1, 0}, {2, 0}, {2, 1}, "
    "{1, 1}, {0, 1}, {0, 2}, {1, 2}, {2, 2}}]); DataRange rescales")
def _peano(*args: Expr) -> Expr:
    args, dr = _curve_opts(args, "PeanoCurve")
    if len(args) != 1:
        raise ValueError("PeanoCurve takes 1 argument")
    n = _nonneg_int(args[0], "PeanoCurve")
    if n > 6:
        raise ValueError("PeanoCurve: n > 6 is too large")
    pts = peano_points(n)
    if dr is not None:
        return _line(_data_range(pts, dr, "PeanoCurve"), True)
    return _line(pts)


def koch_points(n):
    pts = [complex(0, 0), complex(1, 0)]
    w = cmath.exp(1j * math.pi / 3)
    for _ in range(n):
        new = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            d = (b - a) / 3
            new += [a + d, a + d + d * w, a + 2 * d, b]
        pts = new
    return [(p.real, p.imag) for p in pts]


@fn("KochCurve", **ADDED, notes="KochCurve[n] is Line[...] of the n-step Koch curve from {0., 0.} to {1., 0.} "
    "(KochCurve[1] = Line[{{0., 0.}, {0.333333, 0.}, {0.5, 0.288675}, {0.666667, 0.}, {1., 0.}}]); DataRange "
    "rescales; custom step specifications KochCurve[n, {t1, ...}] raise")
def _koch(*args: Expr) -> Expr:
    args, dr = _curve_opts(args, "KochCurve")
    if len(args) != 1:
        raise NotImplementedError("KochCurve: only KochCurve[n] (the standard Koch curve) is implemented")
    n = _nonneg_int(args[0], "KochCurve")
    if n > 9:
        raise ValueError("KochCurve: n > 9 is too large")
    pts = [(0.0 if abs(x) < 1e-15 else x, 0.0 if abs(y) < 1e-15 else y) for x, y in koch_points(n)]
    if dr is not None:
        return _line(_data_range(pts, dr, "KochCurve", box=[(0.0, 1.0), (0.0, 1.0)]), True)
    return _line(pts, True)


_SE, _SW, _NW, _NE = (1, -1), (-1, -1), (-1, 1), (1, 1)
_E, _S, _W, _N = (2, 0), (0, -2), (-2, 0), (0, 2)


def _wirth(kind, n, out):
    """Wirth's Sierpinski curve pieces A, B, C, D (moves in units of h = 32)."""
    if n == 0:
        return
    if kind == "A":
        _wirth("A", n - 1, out); out.append(_SE); _wirth("B", n - 1, out); out.append(_E)
        _wirth("D", n - 1, out); out.append(_NE); _wirth("A", n - 1, out)
    elif kind == "B":
        _wirth("B", n - 1, out); out.append(_SW); _wirth("C", n - 1, out); out.append(_S)
        _wirth("A", n - 1, out); out.append(_SE); _wirth("B", n - 1, out)
    elif kind == "C":
        _wirth("C", n - 1, out); out.append(_NW); _wirth("D", n - 1, out); out.append(_W)
        _wirth("B", n - 1, out); out.append(_SW); _wirth("C", n - 1, out)
    else:
        _wirth("D", n - 1, out); out.append(_NE); _wirth("A", n - 1, out); out.append(_N)
        _wirth("C", n - 1, out); out.append(_NW); _wirth("D", n - 1, out)


def sierpinski_points(n):
    moves = []
    for kind, conn in (("A", _SE), ("B", _SW), ("C", _NW), ("D", _NE)):
        _wirth(kind, n, moves)
        moves.append(conn)
    x = y = 0
    pts = [(0, 0)]
    for dx, dy in moves:
        x, y = x + 32 * dx, y + 32 * dy
        pts.append((x, y))
    return pts


@fn("SierpinskiCurve", **ADDED, notes="SierpinskiCurve[n] is the closed Line[...] of the n-step Sierpinski curve "
    "(Wirth's construction, diagonal steps {+-32, +-32}, straight steps of length 64, starting and ending at {0, 0}; "
    "SierpinskiCurve[1] has 17 points); DataRange rescales the bounding box")
def _sierpinski_curve(*args: Expr) -> Expr:
    args, dr = _curve_opts(args, "SierpinskiCurve")
    if len(args) != 1:
        raise ValueError("SierpinskiCurve takes 1 argument")
    n = _nonneg_int(args[0], "SierpinskiCurve")
    if n < 1:
        raise ValueError("SierpinskiCurve: the number of steps must be a positive integer")
    if n > 7:
        raise ValueError("SierpinskiCurve: n > 7 is too large")
    pts = sierpinski_points(n)
    if dr is not None:
        return _line(_data_range(pts, dr, "SierpinskiCurve"), True)
    return _line(pts)


# ─────────────────────────── Mandelbrot and Julia sets ───────────────────────────

def _iter_opts(args, fname, default_max):
    args, opts = split_options(args, {"MaxIterations", "WorkingPrecision"}, fname)
    mx = opts.get("MaxIterations", default_max)
    if is_sym(mx, "Infinity"):
        raise NotImplementedError(f"{fname}: MaxIterations -> Infinity is not supported")
    if not _intq(mx) or int(mx) < 1:
        raise ValueError(f"{fname}: MaxIterations must be a positive integer")
    wp = opts.get("WorkingPrecision")
    prec = None
    if wp is not None and not is_sym(wp, "Automatic"):
        if is_sym(wp, "MachinePrecision"):
            prec = None
        elif isinstance(wp, (int, float)) or (isinstance(wp, sp.Basic) and wp.is_number):
            prec = int(float(S(wp)))
        else:
            raise ValueError(f"{fname}: invalid WorkingPrecision")
    return list(args), int(mx), prec


def _complex_value(c, fname, prec=None):
    """A Python complex (or mpmath mpc when prec is given) for a numeric c, else None."""
    if isinstance(c, bool):
        return None
    try:
        v = S(c)
    except Exception:
        return None
    if not isinstance(v, sp.Basic) or not v.is_number or v.free_symbols:
        return None
    if prec is not None:
        with mpmath.workdps(prec):
            return mpmath.mpc(complex(sp.N(v, prec + 5))) if False else mpmath.mpmathify(sp.N(v, prec + 5))
    vv = complex(sp.N(v, 20))
    return vv


def _known_inside(c):
    """Main cardioid and period-2 bulb tests."""
    x, y = c.real, c.imag
    q = (x - 0.25) ** 2 + y * y
    if q * (q + (x - 0.25)) <= 0.25 * y * y:
        return True
    return (x + 1) ** 2 + y * y <= 1 / 16


def mandel_count(c, mx):
    if _known_inside(c):
        return mx + 1
    z = 0j
    for k in range(mx):
        z = z * z + c
        if abs(z) > 2:
            return k
    return mx


def _thread(x, f):
    if is_list(x):
        return L(*(_thread(y, f) for y in x.args))
    return f(x)


@fn("MandelbrotSetIterationCount", **ADDED, notes="MandelbrotSetIterationCount[c]: the number of iterations "
    "z -> z^2 + c from 0 before |z| > 2 (MaxIterations -> 1000 by default; the maximum if never escaping; "
    "MaxIterations + 1 for points known to be inside: the main cardioid and the period-2 disk); threads over "
    "lists; machine-precision iteration")
def _msic(*args: Expr) -> Expr:
    args, mx, prec = _iter_opts(args, "MandelbrotSetIterationCount", 1000)
    if len(args) != 1:
        raise ValueError("MandelbrotSetIterationCount takes 1 argument")

    def one(c):
        v = _complex_value(c, "MandelbrotSetIterationCount")
        if v is None:
            raise ValueError("MandelbrotSetIterationCount: the argument must be a number")
        return mandel_count(v, mx)
    return _thread(args[0], one)


@fn("MandelbrotSetMemberQ", **ADDED, notes="MandelbrotSetMemberQ[c] is True if the orbit of 0 under z^2 + c does "
    "not leave |z| <= 2 within MaxIterations (default 1000) iterations or c is known to be inside (cardioid, "
    "period-2 disk); threads over lists; non-numeric arguments give False")
def _msmq(*args: Expr) -> Boolean:
    args, mx, prec = _iter_opts(args, "MandelbrotSetMemberQ", 1000)
    if len(args) != 1:
        raise ValueError("MandelbrotSetMemberQ takes 1 argument")

    def one(c):
        v = _complex_value(c, "MandelbrotSetMemberQ")
        if v is None:
            return False
        return mandel_count(v, mx) >= mx
    return _thread(args[0], one)


def _mandel_distance(c, mx):
    if _known_inside(c):
        return 0
    z, dz = 0j, 0j
    for _ in range(mx):
        dz = 2 * z * dz + 1
        z = z * z + c
        if abs(z) > 1e10:
            a = abs(z)
            return 2 * a * math.log(a) / abs(dz)
    return 0.0


@fn("MandelbrotSetDistance", **ADDED, notes="MandelbrotSetDistance[c] estimates the distance from c to the "
    "Mandelbrot set with the exterior distance estimate 2 |z_n| Log|z_n| / |z_n'| (MaxIterations -> 100; 0 for "
    "points that do not escape, exact 0 for points known to be inside); the \"Interior\" estimate raises")
def _msd(*args: Expr) -> Expr:
    args, mx, prec = _iter_opts(args, "MandelbrotSetDistance", 100)
    if len(args) == 2:
        if args[1] == "Interior":
            raise NotImplementedError("MandelbrotSetDistance: the \"Interior\" estimate is not implemented")
        raise ValueError("MandelbrotSetDistance: the second argument must be \"Interior\"")
    if len(args) != 1:
        raise ValueError("MandelbrotSetDistance takes 1 or 2 arguments")

    def one(c):
        v = _complex_value(c, "MandelbrotSetDistance")
        if v is None:
            raise ValueError("MandelbrotSetDistance: the argument must be a number")
        return _mandel_distance(v, mx)
    return _thread(args[0], one)


def _boettcher(c, z, mx, dps):
    """Boettcher coordinate of z for z -> z^2 + c (mpmath), or None if the orbit does not escape."""
    with mpmath.workdps(dps + 10):
        c = mpmath.mpmathify(c)
        z = mpmath.mpmathify(z)
        big = mpmath.mpf(10) ** (dps + 10)
        res = z
        zn = z
        for k in range(1, mx + 1):
            if zn == 0:
                return None
            # B = z prod (1 + c / z_{n-1}^2)^(1/2^n)
            res *= (1 + c / (zn * zn)) ** (mpmath.mpf(1) / 2 ** k)
            zn = zn * zn + c
            if abs(zn) > big:
                return res
        return None


def _out_num(v, dps, real_hint):
    if dps <= 15:
        cv = complex(v)
        if real_hint and abs(cv.imag) <= 1e-12 * max(1.0, abs(cv.real)):
            return complex(cv.real, 0.0)
        return cv
    with mpmath.workdps(dps):
        return R(sp.Float(mpmath.re(v), dps) + sp.I * sp.Float(mpmath.im(v), dps)) if mpmath.im(v) != 0 else \
            sp.Float(mpmath.re(v), dps)


def _boett_common(fname, c, z, mx, prec_from_n=None):
    """Evaluate when z (and c) are approximate numbers; raise Unevaluated for exact or non-escaping input."""
    vals = [c, z]
    if not all(_complex_value(v, fname) is not None for v in vals):
        raise Unevaluated()
    if not any(has_float(v) for v in vals) and prec_from_n is None:
        raise Unevaluated()
    dps = 15
    for v in vals:
        sv = S(v)
        for f in sp.preorder_traversal(sv):
            if isinstance(f, sp.Float) and f._prec > 60:
                dps = max(dps, int(f._prec * math.log10(2)))
    if prec_from_n is not None:
        dps = max(dps, prec_from_n)
    cc = mpmath.mpmathify(sp.N(S(c), dps + 10)) if True else None
    zz = mpmath.mpmathify(sp.N(S(z), dps + 10))
    r = _boettcher(cc, zz, mx, dps)
    if r is None:
        raise Unevaluated()
    return r, dps


@fn("MandelbrotSetBoettcher", **ADDED, listable=True, notes="MandelbrotSetBoettcher[c] is the Boettcher "
    "coordinate Phi(c) = lim z_n^(1/2^n) (z_0 = c, z -> z^2 + c) for c outside the Mandelbrot set; approximate c "
    "evaluates (any precision), exact c stays unevaluated (N evaluates it); points whose orbit does not escape "
    "within MaxIterations (100) stay unevaluated")
def _msb(*args: Expr) -> Expr:
    args, mx, prec = _iter_opts(args, "MandelbrotSetBoettcher", 100)
    if len(args) != 1:
        raise ValueError("MandelbrotSetBoettcher takes 1 argument")
    c = args[0]
    r, dps = _boett_common("MandelbrotSetBoettcher", c, c, mx)
    return _out_num(r, dps, isinstance(c, float) or (isinstance(c, complex) and False))


@fn("JuliaSetBoettcher", **ADDED, notes="JuliaSetBoettcher[c, z] is the Boettcher coordinate lim z_n^(1/2^n) of z "
    "for the quadratic map z -> z^2 + c (z_0 = z); approximate input evaluates, exact input stays unevaluated (N "
    "evaluates it); threads over lists of z; non-escaping orbits stay unevaluated")
def _jsb(*args: Expr) -> Expr:
    args, mx, prec = _iter_opts(args, "JuliaSetBoettcher", 100)
    if len(args) != 2:
        raise ValueError("JuliaSetBoettcher takes 2 arguments")
    c, z = args
    if is_list(z):
        return L(*(ev(Expr("JuliaSetBoettcher", c, w, *args[2:])) for w in z.args))
    r, dps = _boett_common("JuliaSetBoettcher", c, z, mx)
    return _out_num(r, dps, False)


def _n_boettcher(x, prec):
    """N[...] of an unevaluated Boettcher expression."""
    try:
        if x.head == "MandelbrotSetBoettcher" and len(x.args) == 1:
            r, dps = _boett_common("MandelbrotSetBoettcher", x.args[0], x.args[0], 100, prec_from_n=prec)
        elif x.head == "JuliaSetBoettcher" and len(x.args) == 2:
            r, dps = _boett_common("JuliaSetBoettcher", x.args[0], x.args[1], 100, prec_from_n=prec)
        else:
            return None
    except Unevaluated:
        return None
    return _out_num(r, dps, False) if prec > 15 else _out_num(r, 15, False)


NT.N_PRE["MandelbrotSetBoettcher"] = _n_boettcher
NT.N_PRE["JuliaSetBoettcher"] = _n_boettcher


# ─────────────────────────── AnglePath3D ───────────────────────────

_FORMS3D = ("Position", "FrameMatrix", "EulerMatrix", "EulerAngles")


def _scalarq(x):
    return not is_list(x)


def _euler_matrix(a, b, g):
    """EulerMatrix[{a, b, g}, {3, 2, 1}] = RotationMatrix[a, z].RotationMatrix[b, y].RotationMatrix[g, x]."""
    rz = sp.Matrix([[sp.cos(a), -sp.sin(a), 0], [sp.sin(a), sp.cos(a), 0], [0, 0, 1]])
    ry = sp.Matrix([[sp.cos(b), 0, sp.sin(b)], [0, 1, 0], [-sp.sin(b), 0, sp.cos(b)]])
    rx = sp.Matrix([[1, 0, 0], [0, sp.cos(g), -sp.sin(g)], [0, sp.sin(g), sp.cos(g)]])
    return rz * ry * rx


def _matrixq(x):
    return is_list(x) and len(x.args) == 3 and all(is_list(r) and len(r.args) == 3 and
                                                    all(_scalarq(v) for v in r.args) for r in x.args)


def _rot3(x, fname):
    """(3x3 SymPy matrix, Euler angles or None) for a rotation spec: {a, b}, {a, b, g} or a 3x3 matrix."""
    if _matrixq(x):
        return sp.Matrix([[_num(v) for v in r.args] for r in x.args]), None
    if is_list(x) and len(x.args) in (2, 3) and all(_scalarq(v) for v in x.args):
        angs = [_num(v) for v in x.args] + ([sp.Integer(0)] if len(x.args) == 2 else [])
        return _euler_matrix(*angs), angs
    raise ValueError(f"{fname}: a rotation must be {{alpha, beta}}, {{alpha, beta, gamma}} or a 3x3 matrix")


def _steps3d(steps, fname):
    if not is_list(steps):
        raise ValueError(f"{fname}: the steps must be a list")
    items = list(steps.args)
    # {{r1, r2, ...}, {rot1, rot2, ...}}
    if len(items) == 2 and is_list(items[0]) and is_list(items[1]) and items[0].args and \
            all(_scalarq(v) for v in items[0].args) and len(items[0].args) == len(items[1].args) and \
            all(is_list(v) for v in items[1].args):
        return [(_num(r), _rot3(rot, fname)) for r, rot in zip(items[0].args, items[1].args)]
    out = []
    for s in items:
        if is_list(s) and len(s.args) == 2 and _scalarq(s.args[0]) and is_list(s.args[1]):
            out.append((_num(s.args[0]), _rot3(s.args[1], fname)))
        else:
            out.append((sp.Integer(1), _rot3(s, fname)))
    return out


def _init3d(init, fname):
    """(position, frame matrix) for the initial specification."""
    def pos(p):
        if not (is_list(p) and len(p.args) == 3 and all(_scalarq(v) for v in p.args)):
            raise ValueError(f"{fname}: the initial position must be {{x, y, z}}")
        return [_num(v) for v in p.args]
    if not is_list(init):
        raise ValueError(f"{fname}: invalid initial specification")
    if len(init.args) == 3 and all(_scalarq(v) for v in init.args):
        return pos(init), sp.eye(3)
    if len(init.args) == 1:
        return [sp.Integer(0)] * 3, _rot3(init.args[0], fname)[0]
    if len(init.args) == 2 and is_list(init.args[0]) and len(init.args[0].args) == 3 and \
            all(_scalarq(v) for v in init.args[0].args):
        return pos(init.args[0]), _rot3(init.args[1], fname)[0]
    raise ValueError(f"{fname}: the initial specification must be {{x, y, z}}, {{rot}} or {{{{x, y, z}}, rot}}")


@fn("AnglePath3D", **ADDED, notes="AnglePath3D[steps] / AnglePath3D[init, steps] / AnglePath3D[init, steps, "
    "form]: a 3D turtle path from {0, 0, 0} (or the given point/initial frame); each step rotates the local frame "
    "by EulerMatrix[{alpha, beta, gamma}, {3, 2, 1}] (gamma = 0 for pairs) or by a 3x3 rotation matrix, then "
    "advances r (default 1) along the local x axis; steps {{r, rot}, ...} or {{r1, ...}, {rot1, ...}}; forms "
    "Position, FrameMatrix (relative to the initial frame), EulerMatrix, EulerAngles or a list of them; the "
    "transformation forms and FrameAngles raise; exact input gives exact coordinates for up to 200 steps, "
    "machine numbers otherwise or for approximate input")
def _anglepath3d(*args: Expr) -> Expr:
    fname = "AnglePath3D"
    args, opts = split_options(args, {"WorkingPrecision"}, fname)
    args = list(args)
    form = "Position"
    if len(args) >= 2 and (isinstance(args[-1], str) or (is_list(args[-1]) and args[-1].args and
                                                          all(isinstance(f, str) for f in args[-1].args))):
        form = args.pop()
    if len(args) == 1:
        init, steps = None, args[0]
    elif len(args) == 2:
        init, steps = args
    else:
        raise ValueError("AnglePath3D takes 1 to 3 arguments")
    forms = list(form.args) if is_list(form) else [form]
    for f in forms:
        if f in ("FrameAngles", "Translation", "Rotation", "RotationTranslation"):
            raise NotImplementedError(f"AnglePath3D: the form {f!r} is not supported")
        if f not in _FORMS3D:
            raise ValueError(f"AnglePath3D: unknown form {f!r}")
    st = _steps3d(steps, fname)
    p0, f0 = ([sp.Integer(0)] * 3, sp.eye(3)) if init is None else _init3d(init, fname)
    numeric = has_float(steps) or (init is not None and has_float(init)) or len(st) > 200
    wp = opts.get("WorkingPrecision")
    if wp is not None and not is_sym(wp, "Automatic"):
        if is_sym(wp, "MachinePrecision"):
            numeric = True
        elif is_sym(wp, "Infinity"):
            numeric = False
        else:
            raise NotImplementedError("AnglePath3D: only WorkingPrecision -> Automatic, MachinePrecision or "
                                      "Infinity is supported")
    if "EulerAngles" in forms and any(a is None for _r, (_m, a) in st):
        raise ValueError("AnglePath3D: the form \"EulerAngles\" needs the steps given as Euler angles")

    def val(v):
        if numeric:
            v = sp.N(v, 15)
            if v.is_number:
                c = complex(v)
                return c.real if c.imag == 0 or abs(c.imag) < 1e-14 * max(1.0, abs(c.real)) else c
            return R(v)
        if v.is_number:
            v = sp.nsimplify(sp.simplify(v)) if v.has(sp.sin, sp.cos) else sp.simplify(v)
        return R(v)

    if numeric:
        p0 = [sp.N(v, 20) for v in p0]
        f0 = f0.evalf(20)
    pos = [list(p0)]
    frames = [f0]
    p, f = list(p0), f0
    f0inv = f0.T
    for r, (m, _a) in st:
        if numeric:
            m = m.evalf(20)
            r = sp.N(r, 20)
        f = f * m
        d = f[:, 0] * r
        p = [p[i] + d[i] for i in range(3)]
        pos.append(p)
        frames.append(f)

    def mat(M):
        return L(*(L(*(val(M[i, j]) for j in range(3))) for i in range(3)))

    def one(fm):
        if fm == "Position":
            return L(*(L(*(val(v) for v in q)) for q in pos))
        if fm == "FrameMatrix":
            return L(*(mat(f0inv * F) for F in frames))
        if fm == "EulerMatrix":
            return L(*([mat(sp.eye(3))] + [mat(m) for _r, (m, _a) in st]))
        return L(*([L(0, 0, 0)] + [L(*(val(x) for x in a)) for _r, (_m, a) in st]))
    if is_list(form):
        per = [one(fm) for fm in forms]
        return L(*(L(*(pf.args[i] for pf in per)) for i in range(len(pos))))
    return one(form)


# ─────────────────────────── JuliaSetPoints ───────────────────────────

def _rational_map(f, z, fname):
    """(numerator coefficients, denominator coefficients) of the rational function f of z (highest first)."""
    zs = sp.Symbol("_jz")
    try:
        e = sp.together(S(f).subs(S(z), zs))
    except Exception:
        raise ValueError(f"{fname}: the first argument must be a rational function of the variable")
    num, den = sp.fraction(e)
    if not (num.is_polynomial(zs) and den.is_polynomial(zs)) or (e.free_symbols - {zs}):
        raise ValueError(f"{fname}: the first argument must be a rational function of the variable with numeric "
                         f"coefficients")
    P, Q = sp.Poly(num, zs), sp.Poly(den, zs)
    g = sp.gcd(P, Q)
    if g.degree() > 0:
        P, Q = sp.div(P, g)[0], sp.div(Q, g)[0]
    if max(P.degree(), Q.degree()) < 2:
        raise ValueError(f"{fname}: the function must have degree at least 2")
    pc = [complex(sp.N(c)) for c in P.all_coeffs()]
    qc = [complex(sp.N(c)) for c in Q.all_coeffs()]
    return pc, qc


def _julia_points(pc, qc, tol, bound, polynomial, limit=400000):
    import numpy as np
    deg = max(len(pc), len(qc)) - 1
    P = np.zeros(deg + 1, dtype=complex)
    Q = np.zeros(deg + 1, dtype=complex)
    P[deg + 1 - len(pc):] = pc
    Q[deg + 1 - len(qc):] = qc
    dP, dQ = np.polyder(P), np.polyder(Q)

    def preimages(w):
        return np.roots(P - w * Q) if np.any(P - w * Q) else np.array([], dtype=complex)

    def deriv(z):
        q = np.polyval(Q, z)
        return (np.polyval(dP, z) * q - np.polyval(P, z) * np.polyval(dQ, z)) / (q * q)
    # a repelling fixed point lies on the Julia set; otherwise pull a point back onto it
    start = None
    for z in np.roots(np.polysub(P, np.polymul(Q, [1, 0]))):
        with np.errstate(all="ignore"):
            q = np.polyval(Q, z)
            if abs(q) > 1e-12 and abs(deriv(z)) > 1 + 1e-9 and abs(z) <= bound:
                start = complex(z)
                break
    if start is None:
        rng = np.random.default_rng(12345)
        w = complex(0.123456 + 0.654321j)
        for _ in range(200):
            pre = preimages(w)
            if not len(pre):
                break
            w = complex(pre[rng.integers(len(pre))])
        start = w
    cells = {}
    pts = []

    def near(z):
        cx, cy = int(math.floor(z.real / tol)), int(math.floor(z.imag / tol))
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for k in cells.get((i, j), ()):
                    if abs(pts[k] - z) < tol:
                        return True
        return False

    def add(z):
        cells.setdefault((int(math.floor(z.real / tol)), int(math.floor(z.imag / tol))), []).append(len(pts))
        pts.append(z)
    add(start)
    queue = [start]
    head = 0
    while head < len(queue):
        w = queue[head]
        head += 1
        for z in preimages(w):
            z = complex(z)
            if not (math.isfinite(z.real) and math.isfinite(z.imag)) or abs(z) > bound:
                continue
            if not near(z):
                add(z)
                queue.append(z)
                if len(pts) > limit:
                    raise ValueError("JuliaSetPoints: too many points; increase \"ClosenessTolerance\"")
    return pts


@fn("JuliaSetPoints", **ADDED, notes="JuliaSetPoints[c] (the map z^2 + c) or JuliaSetPoints[f, z] for a rational "
    "function f of degree >= 2: points {x, y} approximating the Julia set by inverse iteration from a repelling "
    "fixed point, keeping preimages at least \"ClosenessTolerance\" (default 0.004) apart and within \"Bound\" "
    "(default 6; unbounded for polynomials) of the origin; the points and their order may differ from the "
    "Wolfram Language's, every point lies (numerically) on the Julia set")
def _juliasetpoints(*args: Expr) -> ListOf[Expr]:
    fname = "JuliaSetPoints"
    pos, opts = [], {}
    for a in args:
        if isinstance(a, Expr) and a.head in ("Rule", "RuleDelayed") and isinstance(a.args[0], str):
            if a.args[0] not in ("ClosenessTolerance", "Bound"):
                raise ValueError(f"{fname}: unknown option {a.args[0]!r}")
            opts[a.args[0]] = a.args[1]
        else:
            pos.append(a)
    tol = opts.get("ClosenessTolerance", 0.004)
    try:
        tol = float(S(tol))
    except Exception:
        raise ValueError(f"{fname}: \"ClosenessTolerance\" must be a positive number")
    if not tol > 0:
        raise ValueError(f"{fname}: \"ClosenessTolerance\" must be a positive number")
    if len(pos) == 1:
        cv = _complex_value(pos[0], fname)
        if cv is None:
            raise Unevaluated()
        pc, qc = [1, 0, cv], [1]
    elif len(pos) == 2:
        if not isinstance(pos[1], Symbol):
            raise ValueError(f"{fname}: the second argument must be a variable")
        if (S(pos[0]).free_symbols - {S(pos[1])}):
            raise Unevaluated()
        pc, qc = _rational_map(pos[0], pos[1], fname)
    else:
        raise ValueError("JuliaSetPoints takes 1 or 2 arguments plus options")
    polynomial = len(qc) == 1
    if "Bound" in opts:
        try:
            bound = float(S(opts["Bound"]))
        except Exception:
            raise ValueError(f"{fname}: \"Bound\" must be a positive number")
    else:
        bound = 6.0
    if polynomial and "Bound" not in opts:
        # every point of the Julia set of a polynomial lies within its escape radius
        lead = abs(pc[0] / qc[0])
        rest = sum(abs(c / qc[0]) for c in pc[1:])
        bound = max(1.0, (1 + rest) / lead) + 1.0
    pts = _julia_points(pc, qc, tol, bound, polynomial)
    return L(*(L(z.real, z.imag) for z in pts))
