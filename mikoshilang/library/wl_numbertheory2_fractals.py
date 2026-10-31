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
def _anglepath(*args):
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
def _hilbert(*args):
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
def _peano(*args):
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
def _koch(*args):
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
def _sierpinski_curve(*args):
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
def _msic(*args):
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
def _msmq(*args):
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
def _msd(*args):
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
def _msb(*args):
    args, mx, prec = _iter_opts(args, "MandelbrotSetBoettcher", 100)
    if len(args) != 1:
        raise ValueError("MandelbrotSetBoettcher takes 1 argument")
    c = args[0]
    r, dps = _boett_common("MandelbrotSetBoettcher", c, c, mx)
    return _out_num(r, dps, isinstance(c, float) or (isinstance(c, complex) and False))


@fn("JuliaSetBoettcher", **ADDED, notes="JuliaSetBoettcher[c, z] is the Boettcher coordinate lim z_n^(1/2^n) of z "
    "for the quadratic map z -> z^2 + c (z_0 = z); approximate input evaluates, exact input stays unevaluated (N "
    "evaluates it); threads over lists of z; non-escaping orbits stay unevaluated")
def _jsb(*args):
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
