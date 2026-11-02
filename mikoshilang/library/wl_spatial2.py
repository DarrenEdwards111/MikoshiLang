"""Wolfram batch X08: spatial statistics II -- cluster and Gibbs point processes, spatial point data, observation
regions.

Builds on the W25 point-process model (``wl_processes_spatial``): the new process classes are added to its
``PCLASSES`` table, so ``RandomPointConfiguration``, ``MeanPointDensity``, ``PointCountDistribution``,
``RipleyK`` and friends, ``PointProcessParameterQ`` and ``PointProcessParameterAssumptions`` accept them.

* Cluster processes (Neyman-Scott family): ``CauchyPointProcess`` (multivariate t displacements with one degree of
  freedom), ``VarianceGammaPointProcess`` (normal variance mixture with gamma variances) and
  ``NeymanScottPointProcess`` (any radial or multivariate cluster distribution).  Simulation places Poisson
  centres in the window dilated by a tail radius R chosen so that the expected number of missed points,
  mu lambda |W| P(|D| > R), is below 1e-10; the heavy-tailed Cauchy process instead uses an exact scheme for the
  far parents (a dominating Pareto-type Poisson process of parent proposals, thinned by the displacement density
  and corrected for size bias by keeping each proposed cluster with probability 1/(cluster size), which is exact
  by the Slivnyak-Mecke theorem).
* Gibbs (Markov) processes: ``GibbsPointProcess`` (pair potential, pair interaction, Papangelou density or
  density forms) and the pairwise special cases ``HardcorePointProcess``, ``StraussPointProcess``,
  ``StraussHardcorePointProcess``, ``PenttinenPointProcess``, ``DiggleGatesPointProcess`` and
  ``DiggleGrattonPointProcess``.  Simulation is the Geyer-Moller (1994) birth-death Metropolis-Hastings sampler,
  started from the empty configuration and run for a stated number of steps in the observation region (free
  boundary).
* ``SpatialPointData`` construction and properties (annotations, ensembles, Ripley-Rasson regions),
  ``RipleyRassonRegion``, ``SpatialObservationRegionQ`` and ``SpatialPointSelect``.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel_lists as KL
from .core import Unevaluated, fn, L, R, S, is_list, rule
from . import wl_processes_spatial as PS
from .wl_processes_spatial import _p, _pos, _dim, _num, _f, _CONDS, _bad, _unit_ball

AREA = "spatial statistics II (X08)"
TAIL_EPS = 1e-10          # bound on the expected number of points missed by window dilation


def _ev(e):
    from ..evaluate import evaluate
    return evaluate(e)


def _np():
    import numpy as np
    return np


def _need_scipy(fname, what):
    try:
        import scipy  # noqa: F401
    except ImportError:
        raise NotImplementedError(f"{fname}: {what} needs SciPy, which is not installed")


def _opt_name(o):
    if isinstance(o, Expr) and o.head in ("Rule", "RuleDelayed") and isinstance(o.args[0], Symbol):
        return o.args[0].name
    if isinstance(o, Expr) and o.head in ("Rule", "RuleDelayed") and isinstance(o.args[0], str):
        return o.args[0]
    return None


def split_opts(args, names, fname):
    """positional arguments and a dict of the trailing options (names); unknown options raise"""
    args = list(args)
    opts = {}
    while args and _opt_name(args[-1]) is not None:
        o = args.pop()
        k = _opt_name(o)
        if k not in names:
            raise ValueError(f"{fname}: unknown option {k}")
        opts[k] = o.args[1]
    for a in args:
        if _opt_name(a) in names:
            raise ValueError(f"{fname}: options must come after the positional arguments")
    return args, opts


def is_auto(v):
    return v is None or v == Symbol("Automatic")


def fnum(v, fname, what):
    """a concrete real number as float; symbolic -> Unevaluated"""
    if isinstance(v, (bool, str)) or is_list(v):
        raise TypeError(f"{fname}: {what} must be a real number")
    x = S(v)
    if not isinstance(x, sp.Basic):
        raise TypeError(f"{fname}: {what} must be a real number")
    if x.free_symbols:
        raise Unevaluated()
    if not x.is_number:
        raise TypeError(f"{fname}: {what} must be a real number")
    x = sp.N(x)
    if not x.is_real:
        raise ValueError(f"{fname}: {what} must be real")
    return float(x)


# ─────────────────────────── observation windows ───────────────────────────

class Win:
    """A numeric observation window: kind box (any dimension), ball (equal radii), ellipsoid, or poly (convex
    polygon, counterclockwise vertices).  Floating point; the region expression is kept in ``e``."""

    def __init__(self, e, fname):
        np = _np()
        self.e = e
        h = e.head if isinstance(e, Expr) else None
        if h == "Interval" and len(e.args) == 1 and is_list(e.args[0]) and len(e.args[0].args) == 2:
            a, b = (fnum(v, fname, "the interval end") for v in e.args[0].args)
            if not b > a:
                raise ValueError(f"{fname}: the interval must have positive length")
            self.kind, self.lo, self.hi, self.dim = "box", np.array([a]), np.array([b]), 1
        elif h == "Polygon":
            if len(e.args) != 1 or not is_list(e.args[0]) or len(e.args[0].args) < 3 or \
                    not all(is_list(p) and len(p.args) == 2 for p in e.args[0].args):
                raise NotImplementedError(f"{fname}: only simple Polygon[{{p1, p2, ...}}] windows in 2D are supported")
            V = np.array([[fnum(c, fname, "a polygon vertex") for c in p.args] for p in e.args[0].args])
            area = 0.5 * float(np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1]))
            if abs(area) < 1e-300:
                raise ValueError(f"{fname}: the polygon has zero area")
            if area < 0:
                V = V[::-1]
            E = np.roll(V, -1, axis=0) - V
            cr = E[:, 0] * np.roll(E, -1, axis=0)[:, 1] - E[:, 1] * np.roll(E, -1, axis=0)[:, 0]
            if np.any(cr < -1e-12 * float(np.max(np.abs(V)) ** 2 + 1e-300)):
                raise NotImplementedError(f"{fname}: non-convex polygon windows are not supported")
            self.kind, self.V, self.dim = "poly", V, 2
            self.lo, self.hi = V.min(axis=0), V.max(axis=0)
            nrm = np.stack([E[:, 1], -E[:, 0]], axis=1)          # outward normals (counterclockwise)
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
            self.nrm, self.off = nrm, np.sum(nrm * V, axis=1)    # x . n <= off inside
            self.area = abs(area)
        else:
            if isinstance(e, Symbol):
                raise Unevaluated()
            rg = PS.region_of(e, fname)
            if not rg.numeric:
                raise Unevaluated()
            self.rg = rg
            self.dim = rg.dim
            if rg.kind == "box":
                self.kind = "box"
                self.lo = np.array([_f(v) for v in rg.lo])
                self.hi = np.array([_f(v) for v in rg.hi])
            else:
                self.c = np.array([_f(v) for v in rg.c])
                self.radii = np.array([_f(v) for v in rg.radii])
                self.kind = "ball" if len(set(self.radii.tolist())) == 1 else "ellipsoid"
                self.lo, self.hi = self.c - self.radii, self.c + self.radii
        self.measure = self._measure()

    def _measure(self):
        np = _np()
        if self.kind == "box":
            return float(np.prod(self.hi - self.lo))
        if self.kind == "poly":
            return self.area
        return float(_f(_unit_ball(self.dim))) * float(np.prod(self.radii))

    def contains(self, P):
        np = _np()
        if self.kind == "box":
            return np.all((P >= self.lo) & (P <= self.hi), axis=1)
        if self.kind == "poly":
            return np.all(P @ self.nrm.T <= self.off + 1e-12, axis=1)
        return (((P - self.c) / self.radii) ** 2).sum(axis=1) <= 1.0

    def bdist(self, P, fname="SpatialBoundaryCorrection"):
        """distance from points inside the window to its boundary"""
        np = _np()
        if self.kind == "box":
            return np.minimum(P - self.lo, self.hi - P).min(axis=1)
        if self.kind == "poly":
            return (self.off - P @ self.nrm.T).min(axis=1)
        if self.kind == "ball":
            return self.radii[0] - np.linalg.norm(P - self.c, axis=1)
        raise NotImplementedError(f"{fname}: boundary distances in ellipsoidal (non-circular) windows are not "
                                  "supported")

    def eroded_measure(self, s, fname="SpatialBoundaryCorrection"):
        """measure of {x in W : distance to the boundary >= s}"""
        np = _np()
        if self.kind == "box":
            return float(np.prod(np.maximum(self.hi - self.lo - 2 * s, 0.0)))
        if self.kind == "ball":
            return float(_f(_unit_ball(self.dim))) * max(self.radii[0] - s, 0.0) ** self.dim
        if self.kind == "poly":
            Q = clip_poly(self.V, self.nrm, self.off - s)
            return poly_area(Q)
        raise NotImplementedError(f"{fname}: erosion of ellipsoidal windows is not supported")

    def uniform(self, rng, n):
        np = _np()
        out = np.empty((0, self.dim))
        while len(out) < n:
            m = max(16, int((n - len(out)) * 1.6) + 8)
            cand = rng.uniform(self.lo, self.hi, size=(m, self.dim))
            out = np.vstack([out, cand[self.contains(cand)]])
        return out[:n]

    def circ_frac(self, p, r, fname="RipleyK"):
        """fraction of the circle of radius r (array) about p (2D) lying in the window (Ripley's isotropic
        correction weight is its reciprocal)"""
        np = _np()
        if self.dim != 2:
            raise NotImplementedError(f"{fname}: Ripley's isotropic correction is only available in 2D")
        r = np.asarray(r, dtype=float)
        if self.kind == "ball":
            t = float(np.linalg.norm(p - self.c))
            Rw = self.radii[0]
            out = np.ones_like(r)
            if t == 0:
                out[r > Rw] = 0.0
                return out
            cut = t + r > Rw
            cosv = np.clip((t * t + r * r - Rw * Rw) / (2 * t * np.where(r > 0, r, 1.0)), -1.0, 1.0)
            out[cut] = np.arccos(-cosv[cut]) / math.pi
            out[cut & (r >= Rw + t)] = 0.0
            return out
        if self.kind == "ellipsoid":
            raise NotImplementedError(f"{fname}: Ripley's correction in elliptical windows is not supported")
        if self.kind == "box":
            nrm = np.array([[-1.0, 0], [1.0, 0], [0, -1.0], [0, 1.0]])
            off = np.array([-self.lo[0], self.hi[0], -self.lo[1], self.hi[1]])
        else:
            nrm, off = self.nrm, self.off
        dist = off - nrm @ p                       # distance from p to each edge line (>= 0 inside)
        ang = np.arctan2(nrm[:, 1], nrm[:, 0])
        out = np.empty_like(r)
        for k, rv in enumerate(r):
            if rv <= 0:
                out[k] = 1.0
                continue
            iv = []
            for d_, a_ in zip(dist, ang):
                if d_ < rv:
                    w = math.acos(max(-1.0, min(1.0, d_ / rv)))
                    iv.append((a_ - w, a_ + w))
            out[k] = 1.0 - _arc_union(iv) / (2 * math.pi)
        return out

    def halfspaces(self, fname):
        np = _np()
        if self.kind == "box":
            d = self.dim
            A = np.vstack([-np.eye(d), np.eye(d)])
            b = np.concatenate([-self.lo, self.hi])
            return A, b
        if self.kind == "poly":
            return self.nrm, self.off
        raise NotImplementedError(f"{fname}: this needs a box or convex polygon window")


def _arc_union(iv):
    """total length of a union of angular intervals (each shorter than 2 Pi)"""
    if not iv:
        return 0.0
    segs = []
    for a, b in iv:
        w = b - a
        if w >= 2 * math.pi:
            return 2 * math.pi
        s = a % (2 * math.pi)
        e = s + w
        if e > 2 * math.pi:
            segs.append((s, 2 * math.pi))
            segs.append((0.0, e - 2 * math.pi))
        else:
            segs.append((s, e))
    segs.sort()
    tot, cs, ce = 0.0, segs[0][0], segs[0][1]
    for s, e in segs[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return min(2 * math.pi, tot + ce - cs)


def clip_poly(V, nrm, off):
    """the convex polygon V clipped by the half-planes x . nrm_k <= off_k (Sutherland-Hodgman)"""
    np = _np()
    P = [tuple(v) for v in V]
    for n_, o_ in zip(nrm, off):
        if not P:
            break
        Q = []
        for i in range(len(P)):
            a, b = np.array(P[i]), np.array(P[(i + 1) % len(P)])
            fa, fb = a @ n_ - o_, b @ n_ - o_
            if fa <= 0:
                Q.append(tuple(a))
            if (fa < 0 < fb) or (fb < 0 < fa):
                t = fa / (fa - fb)
                Q.append(tuple(a + t * (b - a)))
        P = Q
    return np.array(P) if P else np.zeros((0, 2))


def poly_area(Q):
    np = _np()
    if len(Q) < 3:
        return 0.0
    return abs(0.5 * float(np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1])))


def window(e, fname):
    return Win(e, fname)


# ─────────────────────────── point data ───────────────────────────

def _coords(p, fname):
    if not is_list(p) or not p.args:
        if isinstance(p, Expr) and p.head in ("GeoPosition", "GeoPositionXYZ", "GeoPositionENU"):
            raise NotImplementedError(f"{fname}: geographic points are not supported")
        if isinstance(p, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: a point must be a list of coordinates, not {p}")
    return [fnum(c, fname, "a coordinate") for c in p.args]


def point_array(pts, fname, dim=None):
    """an n x d float array from a list of points"""
    np = _np()
    if isinstance(pts, Symbol):
        raise Unevaluated()
    if isinstance(pts, Expr) and pts.head in ("GeoPosition", "GeoPositionXYZ", "GeoPositionENU"):
        raise NotImplementedError(f"{fname}: geographic points are not supported")
    if not is_list(pts):
        raise TypeError(f"{fname}: the points must be a list of coordinate lists")
    rows = [_coords(p, fname) for p in pts.args]
    if not rows:
        return np.zeros((0, dim or 0))
    d = len(rows[0])
    if any(len(r_) != d for r_ in rows):
        raise ValueError(f"{fname}: all points must have the same dimension")
    if dim is not None and d != dim:
        raise ValueError(f"{fname}: the points have dimension {d} but the region has dimension {dim}")
    return np.array(rows, dtype=float)


def _is_point(p):
    return is_list(p) and p.args and all(not is_list(c) for c in p.args)


def _is_pointlist(x):
    return is_list(x) and all(_is_point(p) for p in x.args)


def _ann_assoc(vals, n, fname):
    """annotation association key -> list of n values from vals (list, list of lists or association)"""
    if isinstance(vals, Expr) and vals.head == "Association":
        out = []
        for r_ in vals.args:
            v = r_.args[1]
            if not is_list(v) or len(v.args) != n:
                raise ValueError(f"{fname}: each annotation must have one value per point ({n})")
            out.append((r_.args[0], v))
        return out
    if not is_list(vals):
        raise TypeError(f"{fname}: annotations must be a list or an association")
    if len(vals.args) == n and not (n and all(is_list(v) and len(v.args) == n for v in vals.args) and n != 1):
        return [(1, vals)]
    if all(is_list(v) and len(v.args) == n for v in vals.args):
        return [(i + 1, v) for i, v in enumerate(vals.args)]
    raise ValueError(f"{fname}: annotations must give one value per point ({n})")


class Config:
    """one point configuration: points (MikoshiLang point lists) and annotations [(key, list)]"""

    def __init__(self, pts, ann=None):
        self.pts = list(pts)
        self.ann = ann or []

    def expr(self):
        P = L(*self.pts)
        if self.ann:
            return rule(P, Expr("Association", *(rule(k, v) for k, v in self.ann)))
        return P

    def select(self, mask):
        pts = [p for p, m in zip(self.pts, mask) if m]
        ann = [(k, L(*(a for a, m in zip(v.args, mask) if m))) for k, v in self.ann]
        return Config(pts, ann)


def _config_of(x, fname):
    """Config from points, points -> vals, points -> <|k -> vals|>, {p1 -> d1, ...}"""
    if isinstance(x, Expr) and x.head == "Rule":
        pts, vals = x.args
        if not _is_pointlist(pts):
            if isinstance(pts, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: the points must be a list of coordinate lists")
        return Config(pts.args, _ann_assoc(vals, len(pts.args), fname))
    if is_list(x) and x.args and all(isinstance(r_, Expr) and r_.head == "Rule" for r_ in x.args):
        pts = [r_.args[0] for r_ in x.args]
        if not all(_is_point(p) for p in pts):
            raise TypeError(f"{fname}: the points must be coordinate lists")
        vals = L(*(r_.args[1] for r_ in x.args))
        if all(is_list(v) for v in vals.args) and len({len(v.args) for v in vals.args}) == 1:
            k = len(vals.args[0].args)
            ann = [(i + 1, L(*(v.args[i] for v in vals.args))) for i in range(k)]
        elif all(isinstance(v, Expr) and v.head == "Association" for v in vals.args):
            keys = [r_.args[0] for r_ in vals.args[0].args]
            ann = []
            for key in keys:
                col = []
                for v in vals.args:
                    hit = [r_.args[1] for r_ in v.args if r_.args[0] == key]
                    if len(hit) != 1:
                        raise ValueError(f"{fname}: all points need the same annotation keys")
                    col.append(hit[0])
                ann.append((key, L(*col)))
        else:
            ann = [(1, vals)]
        return Config(pts, ann)
    if is_list(x) and (not x.args or _is_pointlist(x)):
        return Config(x.args)
    if isinstance(x, Symbol):
        raise Unevaluated()
    if isinstance(x, Expr) and x.head in ("GeoPosition", "GeoPositionXYZ", "GeoPositionENU"):
        raise NotImplementedError(f"{fname}: geographic points are not supported")
    raise TypeError(f"{fname}: expected a list of points, points -> annotations or {{p -> data, ...}}")


class SPD:
    """parsed SpatialPointData: configs (list of Config), region expr or None, ensemble flag"""

    def __init__(self, configs, reg, ens):
        self.configs, self.reg, self.ens = configs, reg, ens

    def expr(self):
        if self.ens:
            data = Expr("Association", *(rule(i + 1, c.expr()) for i, c in enumerate(self.configs)))
        else:
            data = self.configs[0].expr()
        return Expr("SpatialPointData", data, self.reg) if self.reg is not None else Expr("SpatialPointData", data)

    def dim(self):
        for c in self.configs:
            if c.pts:
                return len(c.pts[0].args)
        if self.reg is not None:
            return window(self.reg, "SpatialPointData").dim
        return None


def parse_spd(x, fname="SpatialPointData"):
    """SPD from a SpatialPointData expression (canonical or not)"""
    if not (isinstance(x, Expr) and x.head == "SpatialPointData" and 1 <= len(x.args) <= 2):
        raise TypeError(f"{fname}: expected SpatialPointData")
    data = x.args[0]
    reg = x.args[1] if len(x.args) == 2 else None
    if isinstance(data, Expr) and data.head == "Association":
        return SPD([_config_of(r_.args[1], fname) for r_ in data.args], reg, True)
    if is_list(data) and data.args and all(is_list(c) and (not c.args or _is_pointlist(c)) for c in data.args) \
            and not _is_pointlist(data):
        return SPD([_config_of(c, fname) for c in data.args], reg, True)
    if is_list(data) and data.args and all(isinstance(c, Expr) and c.head == "Rule" and _is_pointlist(c.args[0])
                                          for c in data.args):
        return SPD([_config_of(c, fname) for c in data.args], reg, True)
    return SPD([_config_of(data, fname)], reg, False)


def ripley_rasson(P, fname="RipleyRassonRegion", exact_pts=None):
    """Ripley-Rasson (1977) estimate of a planar convex window: the convex hull scaled about its centroid by
    1/Sqrt[1 - m/n] (m hull vertices, n points)"""
    np = _np()
    n = len(P)
    if P.shape[1] != 2:
        raise NotImplementedError(f"{fname}: the Ripley-Rasson region is only available for 2D points")
    H = hull2d(P)
    m = len(H)
    if m < 3:
        raise ValueError(f"{fname}: the points are collinear; no full-dimensional region can be estimated")
    if n <= m:
        raise ValueError(f"{fname}: all {n} points are hull vertices; the Ripley-Rasson scale 1/Sqrt[1 - m/n] is "
                         "undefined")
    if exact_pts is not None:
        V = [exact_pts[i] for i in H]
        A2 = sum(V[i][0] * V[(i + 1) % m][1] - V[(i + 1) % m][0] * V[i][1] for i in range(m))
        cx = sum((V[i][0] + V[(i + 1) % m][0]) * (V[i][0] * V[(i + 1) % m][1] - V[(i + 1) % m][0] * V[i][1])
                 for i in range(m)) / (3 * A2)
        cy = sum((V[i][1] + V[(i + 1) % m][1]) * (V[i][0] * V[(i + 1) % m][1] - V[(i + 1) % m][0] * V[i][1])
                 for i in range(m)) / (3 * A2)
        s = 1 / sp.sqrt(1 - sp.Rational(m, n))
        return Expr("Polygon", L(*(L(R(sp.radsimp(cx + s * (v[0] - cx))), R(sp.radsimp(cy + s * (v[1] - cy))))
                                   for v in V)))
    V = P[H]
    x, y = V[:, 0], V[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = cr.sum() / 2
    c = np.array([((x + np.roll(x, -1)) * cr).sum(), ((y + np.roll(y, -1)) * cr).sum()]) / (6 * A)
    s = 1.0 / math.sqrt(1.0 - m / n)
    Q = c + s * (V - c)
    return Expr("Polygon", R(Q.tolist()))


def hull2d(P):
    """indices of the convex hull vertices of 2D points, counterclockwise (Andrew's monotone chain; collinear
    points dropped)"""
    idx = sorted(range(len(P)), key=lambda i: (P[i][0], P[i][1]))
    uniq = []
    for i in idx:
        if not uniq or (P[i][0], P[i][1]) != (P[uniq[-1]][0], P[uniq[-1]][1]):
            uniq.append(i)
    if len(uniq) < 3:
        return uniq

    def cross(o, a, b):
        return (P[a][0] - P[o][0]) * (P[b][1] - P[o][1]) - (P[a][1] - P[o][1]) * (P[b][0] - P[o][0])
    lower, upper = [], []
    for i in uniq:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    for i in reversed(uniq):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def _exact_points(pts):
    """SymPy rational coordinates if every coordinate is exact, else None"""
    out = []
    for p in pts:
        row = []
        for c in p.args:
            v = S(c)
            if not (isinstance(v, sp.Basic) and v.is_Rational):
                return None
            row.append(v)
        out.append(row)
    return out


def _region_check(reg, fname):
    w = window(reg, fname)
    return w


@fn("SpatialPointData", area=AREA, status="added",
    notes="SpatialPointData[pts], SpatialPointData[pts, reg], SpatialPointData[pts -> vals | <|k -> vals, ...|>, "
          "...], SpatialPointData[{p1 -> d1, ...}, ...], ensembles SpatialPointData[{pts1, pts2, ...}, ...] (stored "
          "as an Association 1 -> pts1, ...) and SpatialPointData[spd, newreg]. Points are numeric coordinate "
          "lists of one dimension; with a region (Rectangle, Cuboid, Disk, Ball, Interval[{a, b}] or a convex "
          "Polygon) only the points inside are kept; without one the 2D Ripley-Rasson region is computed (other "
          "dimensions keep no region and statistics that need one raise). List annotations get the keys 1, 2, ... "
          "Properties: \"Points\", \"PointsList\", \"PointCount\", \"PointCountList\", \"ConfigurationCount\", "
          "\"MeanPointCount\", \"MeanPointDensity\", \"Dimension\", \"ObservationRegion\", \"RegionMeasure\", "
          "\"Annotations\", {\"Annotations\", n}, {\"Annotations\", key}, {\"Annotations\", \"Keys\"}, "
          "\"AnnotationsList\", \"FryPlot\" (Graphics of all difference vectors p_i - p_j), \"Properties\"; "
          "\"Summary\" (a formatted table) and geographic points raise NotImplementedError. "
          "SpatialPointData[spd, newreg] intersects regions only when one contains the other or both are boxes "
          "(NotImplementedError otherwise). Check: exact; reference (Wolfram documentation: points outside the "
          "region are dropped, annotation keys {1, 2}) in tests.")
def _spd(data: Expr, reg: Expr = None) -> Expr:
    fname = "SpatialPointData"
    if isinstance(data, Expr) and data.head == "SpatialPointData":
        if reg is None:
            raise Unevaluated()
        old = parse_spd(data, fname)
        neww = window(reg, fname)
        newreg = reg
        if old.reg is not None:
            ow = window(old.reg, fname)
            newreg = _intersect(ow, neww, old.reg, reg, fname)
        out = SPD(old.configs, newreg, old.ens)
        return _filter(out, fname)
    if isinstance(data, Symbol):
        raise Unevaluated()
    sd = parse_spd(Expr("SpatialPointData", data) if reg is None else Expr("SpatialPointData", data, reg), fname)
    d = None
    for c in sd.configs:
        if c.pts:
            A = point_array(L(*c.pts), fname)
            if d is not None and A.shape[1] != d:
                raise ValueError(f"{fname}: all configurations must have the same dimension")
            d = A.shape[1]
    if reg is None:
        if d == 2:
            allp = [p for c in sd.configs for p in c.pts]
            P = point_array(L(*allp), fname)
            sd.reg = ripley_rasson(P, fname, _exact_points(allp))
            return sd.expr()
        canon = sd.expr()
        if canon == Expr("SpatialPointData", data):
            raise Unevaluated()
        return canon
    w = window(reg, fname)
    if d is not None and d != w.dim:
        raise ValueError(f"{fname}: the points have dimension {d} but the region has dimension {w.dim}")
    out = _filter(sd, fname)
    if out == Expr("SpatialPointData", data, reg):
        raise Unevaluated()
    return out


def _filter(sd, fname):
    w = window(sd.reg, fname)
    np = _np()
    cfgs = []
    for c in sd.configs:
        if not c.pts:
            cfgs.append(c)
            continue
        A = point_array(L(*c.pts), fname, w.dim)
        m = w.contains(A)
        cfgs.append(c if bool(np.all(m)) else c.select(m.tolist()))
    return SPD(cfgs, sd.reg, sd.ens).expr()


def _intersect(ow, nw, oe, ne, fname):
    np = _np()
    if ow.dim != nw.dim:
        raise ValueError(f"{fname}: the new region has dimension {nw.dim}, the data {ow.dim}")
    if ow.kind == "box" and nw.kind == "box":
        lo, hi = np.maximum(ow.lo, nw.lo), np.minimum(ow.hi, nw.hi)
        if np.any(hi <= lo):
            raise ValueError(f"{fname}: the new region does not overlap the observation region")
        if np.all(lo == ow.lo) and np.all(hi == ow.hi):
            return oe
        if np.all(lo == nw.lo) and np.all(hi == nw.hi):
            return ne
        if ow.dim == 2:
            return Expr("Rectangle", R(lo.tolist()), R(hi.tolist()))
        return Expr("Cuboid", R(lo.tolist()), R(hi.tolist()))
    if _inside(ow, nw):
        return oe
    if _inside(nw, ow):
        return ne
    raise NotImplementedError(f"{fname}: the intersection of {oe} and {ne} is not supported")


def _inside(a, b):
    """True if window a lies inside window b (checked on a's extreme points / sampled boundary)"""
    np = _np()
    if a.kind == "box":
        import itertools
        pts = np.array(list(itertools.product(*zip(a.lo, a.hi))))
    elif a.kind == "poly":
        pts = a.V
    elif a.kind == "ball" and b.kind in ("ball", "box", "poly"):
        if b.kind == "ball":
            return bool(np.linalg.norm(a.c - b.c) + a.radii[0] <= b.radii[0] + 1e-12)
        return bool(b.contains(a.c[None, :])[0] and b.bdist(a.c[None, :])[0] >= a.radii[0] - 1e-12)
    else:
        return False
    if b.kind == "ellipsoid":
        return bool(np.all(b.contains(pts)))
    return bool(np.all(b.contains(pts)))


def spd_of(x, fname, need_region=True):
    """(SPD, window or None) from point data forms: SpatialPointData, {pts, reg}, {p1, p2, ...}"""
    if isinstance(x, Expr) and x.head == "SpatialPointData":
        if len(x.args) == 1:
            x = _ev(x)
        sd = parse_spd(x, fname)
    elif is_list(x) and len(x.args) == 2 and _is_pointlist(x.args[0]) and not is_list(x.args[1]) and \
            not _is_point(x):
        sd = parse_spd(_ev(Expr("SpatialPointData", x.args[0], x.args[1])), fname)
    elif _is_pointlist(x) and x.args:
        sd = parse_spd(_ev(Expr("SpatialPointData", x)), fname)
    elif isinstance(x, Symbol):
        raise Unevaluated()
    elif isinstance(x, Expr) and x.head in ("GeoPosition", "GeoPositionXYZ", "GeoPositionENU"):
        raise NotImplementedError(f"{fname}: geographic points are not supported")
    else:
        raise TypeError(f"{fname}: expected point data ({{p1, p2, ...}}, {{pts, reg}} or SpatialPointData)")
    if sd.reg is None:
        if need_region:
            raise NotImplementedError(f"{fname}: point data without an observation region are only supported in 2D "
                                      "(Ripley-Rasson region); give the region")
        return sd, None
    return sd, window(sd.reg, fname)


def config_array(c, w, fname):
    return point_array(L(*c.pts), fname, w.dim if w is not None else None)


def _spd_props(head, args, ev):
    from .. import kernel as K
    if len(args) != 1:
        return K.UNHANDLED
    prop = args[0]
    if not (isinstance(prop, str) or (is_list(prop) and prop.args and isinstance(prop.args[0], str))):
        return K.UNHANDLED
    fname = "SpatialPointData"
    try:
        sd = parse_spd(head, fname)
    except TypeError:
        return K.UNHANDLED
    np = _np()
    cs = sd.configs
    first = cs[0] if cs else Config([])
    if is_list(prop):
        if prop.args[0] != "Annotations" or len(prop.args) != 2:
            raise ValueError(f"{fname}: unknown property {prop}")
        sel = prop.args[1]
        if sel == "Keys":
            return L(*(k for k, _ in first.ann))
        if isinstance(sel, int) and not isinstance(sel, bool):
            if not 1 <= sel <= len(cs):
                raise ValueError(f"{fname}: there is no configuration {sel}")
            return _ann_expr(cs[sel - 1])
        hit = [v for k, v in first.ann if k == sel]
        if not hit:
            raise ValueError(f"{fname}: no annotation with key {sel}")
        return hit[0]
    if prop == "Points":
        return L(*first.pts)
    if prop == "PointsList":
        return L(*(L(*c.pts) for c in cs))
    if prop == "PointCount":
        return len(first.pts)
    if prop == "PointCountList":
        return L(*(len(c.pts) for c in cs))
    if prop == "ConfigurationCount":
        return len(cs)
    if prop == "MeanPointCount":
        return R(sp.Rational(sum(len(c.pts) for c in cs), len(cs)))
    if prop in ("ObservationRegion",):
        return sd.reg if sd.reg is not None else Expr("Missing", "NotAvailable")
    if prop in ("RegionMeasure", "MeanPointDensity"):
        if sd.reg is None:
            raise NotImplementedError(f"{fname}: \"{prop}\" needs an observation region")
        meas = _ev(Expr("RegionMeasure", sd.reg))
        if prop == "RegionMeasure":
            return meas
        return _ev(Expr("Divide", R(sp.Rational(sum(len(c.pts) for c in cs), len(cs))), meas))
    if prop == "Dimension":
        d = sd.dim()
        return d if d is not None else Expr("Missing", "NotAvailable")
    if prop == "Annotations":
        return _ann_expr(first)
    if prop == "AnnotationsList":
        return L(*(_ann_expr(c) for c in cs))
    if prop == "FryPlot":
        A = point_array(L(*first.pts), fname)
        if A.shape[1] not in (2, 3):
            raise NotImplementedError(f"{fname}: Fry plots need 2D or 3D points")
        D = (A[:, None, :] - A[None, :, :])[~np.eye(len(A), dtype=bool)]
        return Expr("Graphics" if A.shape[1] == 2 else "Graphics3D", Expr("Point", R(D.tolist())),
                    rule(Symbol("Axes"), True))
    if prop == "Summary":
        raise NotImplementedError(f"{fname}: the formatted \"Summary\" table is not supported")
    if prop == "Properties":
        return L("Annotations", "AnnotationsList", "ConfigurationCount", "Dimension", "FryPlot", "MeanPointCount",
                 "MeanPointDensity", "ObservationRegion", "PointCount", "PointCountList", "Points", "PointsList",
                 "RegionMeasure")
    raise ValueError(f"{fname}: unknown property {prop!r}")


def _ann_expr(c):
    return Expr("Association", *(rule(k, v) for k, v in c.ann))


KL.COMPOUND_HEAD_HOOKS["SpatialPointData"] = _spd_props


def _spd_parts_x08(x, fname):
    """replacement for W25's spd_parts that understands annotations and list ensembles"""
    if not (isinstance(x, Expr) and x.head == "SpatialPointData" and 1 <= len(x.args) <= 2):
        raise TypeError(f"{fname}: expected SpatialPointData")
    sd = parse_spd(x, fname)
    return [list(c.pts) for c in sd.configs], sd.reg, sd.ens


PS.spd_parts = _spd_parts_x08


@fn("RipleyRassonRegion", area=AREA, status="added",
    notes="RipleyRassonRegion[pdata] for 2D points ({p1, ...}, {pts, reg} uses only pts, SpatialPointData uses all "
          "configurations): the convex hull scaled about its area centroid by 1/Sqrt[1 - m/n] (m hull vertices, n "
          "points; Ripley & Rasson 1977, J. Appl. Prob. 14), returned as a Polygon (exact for exact coordinates, "
          "machine reals otherwise). Collinear points or n = m raise ValueError; other dimensions and geographic "
          "points raise NotImplementedError. Check: exact; reference (hand computation for a square with an "
          "interior point) in tests.")
def _rrr(pdata: Expr) -> Expr:
    fname = "RipleyRassonRegion"
    if isinstance(pdata, Expr) and pdata.head == "SpatialPointData":
        pts = [p for c in parse_spd(pdata, fname).configs for p in c.pts]
    elif is_list(pdata) and len(pdata.args) == 2 and _is_pointlist(pdata.args[0]) and not is_list(pdata.args[1]):
        pts = list(pdata.args[0].args)
    elif _is_pointlist(pdata) and pdata.args:
        pts = list(pdata.args)
    elif isinstance(pdata, Symbol):
        raise Unevaluated()
    else:
        raise TypeError(f"{fname}: expected a list of points or SpatialPointData")
    P = point_array(L(*pts), fname)
    return ripley_rasson(P, fname, _exact_points(pts))


@fn("SpatialObservationRegionQ", area=AREA, status="added",
    notes="SpatialObservationRegionQ[reg]: True if reg is a constant (parameter-free), bounded region whose "
          "dimension equals its embedding dimension (RegionQ, BoundedRegionQ, RegionDimension == "
          "RegionEmbeddingDimension, no symbolic parameters), else False; geographic entities raise "
          "NotImplementedError. Check: exact; reference (Wolfram documentation: Disk[] True, Circle[] False, "
          "HalfPlane False) in tests.")
def _soq(reg: Expr) -> Boolean:
    fname = "SpatialObservationRegionQ"
    if isinstance(reg, Expr) and reg.head in ("Entity", "GeoGroup", "GeoBoundsRegion", "Polygon") and \
            reg.args and isinstance(reg.args[0], Expr) and reg.args[0].head in ("GeoPosition", "Entity"):
        raise NotImplementedError(f"{fname}: geographic regions are not supported")
    if isinstance(reg, Expr) and reg.head in ("Entity", "GeoBoundsRegion", "GeoGroup"):
        raise NotImplementedError(f"{fname}: geographic regions are not supported")
    if not isinstance(reg, Expr) or isinstance(reg, Symbol):
        return False
    if _has_free_symbols(reg):
        return False
    try:
        if _ev(Expr("RegionQ", reg)) is not True:
            return False
        if _ev(Expr("BoundedRegionQ", reg)) is not True:
            return False
        dm = _ev(Expr("RegionDimension", reg))
        em = _ev(Expr("RegionEmbeddingDimension", reg))
    except (ValueError, TypeError):
        return False
    return isinstance(dm, int) and dm == em


def _has_free_symbols(e):
    if isinstance(e, Symbol):
        return e.name not in ("Pi", "E", "Automatic", "True", "False", "None", "Infinity")
    if isinstance(e, Expr):
        return any(_has_free_symbols(a) for a in e.args)
    return False


@fn("SpatialPointSelect", area=AREA, status="added",
    notes="SpatialPointSelect[spdata, crit]: the SpatialPointData of the points for which crit[<|\"Point\" -> p, "
          "key1 -> ann1, ...|>] is True (so #Point and #key work in pure functions), keeping the observation "
          "region and the annotations of the kept points; applied to each configuration of an ensemble. Criteria "
          "that give neither True nor False count as False (like Select). Non-SpatialPointData input raises "
          "TypeError. Check: exact; reference (selection by Norm[#Point] and by an annotation) in tests.")
def _sps(spd: Expr, crit: Fn) -> Expr:
    fname = "SpatialPointSelect"
    if isinstance(spd, Symbol):
        raise Unevaluated()
    if not (isinstance(spd, Expr) and spd.head == "SpatialPointData"):
        raise TypeError(f"{fname}: the first argument must be SpatialPointData")
    sd = parse_spd(spd, fname)
    out = []
    for c in sd.configs:
        mask = []
        for i, p in enumerate(c.pts):
            a = Expr("Association", rule("Point", p),
                     *(rule(k, v.args[i]) for k, v in c.ann))
            mask.append(_ev(Expr(crit, a)) is True)
        out.append(c.select(mask))
    return SPD(out, sd.reg, sd.ens).expr()


# ─────────────────────────── cluster processes ───────────────────────────

def _cond(e, ok, msg):
    """record a parameter condition (for PointProcessParameterAssumptions) and check it for numeric values"""
    _CONDS.append(e)
    if ok is False:
        _bad(msg)


def _nonneg(v, fname, what):
    ok = None
    if _num(v):
        ok = bool(v.is_real and v >= 0)
    _cond(Expr("GreaterEqual", R(v), 0), ok, f"{fname}: {what} must be a non-negative real number")


def _le(a, b, fname, msg):
    ok = None
    if _num(a) and _num(b) and a.is_real and b.is_real:
        ok = bool(a <= b)
    _cond(Expr("LessEqual", R(a), R(b)), ok, f"{fname}: {msg}")


def _win_ball(reg):
    """centre and radius of a ball containing the W25 region"""
    np = _np()
    lo, hi = reg.bbox()
    lo, hi = np.array(lo), np.array(hi)
    if reg.kind == "box":
        return (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2
    c = np.array([_f(v) for v in reg.c])
    return c, float(max(_f(v) for v in reg.radii))


def _dilated_box(reg, pad):
    lo, hi = reg.bbox()
    return [x - pad for x in lo], [x + pad for x in hi]


def _poisson_cluster(rng, reg, mu, lam, pad, displace, d, centres=None):
    """offspring in reg of Poisson(mu) centres in the bounding box dilated by pad; displace(m) -> m x d array"""
    np = _np()
    lo2, hi2 = _dilated_box(reg, pad)
    if centres is None:
        vol = math.prod(b - a for a, b in zip(lo2, hi2))
        k = int(rng.poisson(mu * vol))
        if k > 10 ** 6:
            raise NotImplementedError("RandomPointConfiguration: more than 10^6 cluster centres are needed (the "
                                      "dilated window is too large); this is not supported")
        cen = rng.uniform(lo2, hi2, size=(k, d))
    else:
        cen = centres(lo2, hi2)
    counts = rng.poisson(lam, size=len(cen))
    if counts.sum() == 0:
        return np.zeros((0, d))
    pts = np.repeat(cen, counts, axis=0) + displace(int(counts.sum()))
    return pts[reg.contains(pts)]


class CauchyPP(PS.PP):
    def parse(self, a, fname):
        if len(a) != 4:
            raise ValueError(f"{fname}: CauchyPointProcess expects [mu, lambda, b, d]")
        self.mu = _p(a[0], fname, "the cluster centre density")
        self.lam = _p(a[1], fname, "the cluster mean")
        self.s = _p(a[2], fname, "the scale")
        for v, w in ((self.mu, "the cluster centre density"), (self.lam, "the cluster mean"), (self.s, "the scale")):
            _pos(v, fname, w)
        self.d = _dim(a[3], fname)
        self.p = [self.mu, self.lam, self.s]

    def intensity(self):
        return self.mu * self.lam

    def count(self, reg):
        raise NotImplementedError("PointCountDistribution: the count distribution of CauchyPointProcess has no "
                                  "closed form")

    def K(self, r):
        d, b = self.d, self.s
        x = r ** 2 / (r ** 2 + 4 * b ** 2)
        if d == 1:
            F = 2 * sp.atan(r / (2 * b)) / sp.pi
        elif d == 2:
            F = 1 - 2 * b / sp.sqrt(r ** 2 + 4 * b ** 2)
        elif d == 3:
            F = 2 * (sp.asin(sp.sqrt(x)) - sp.sqrt(x) * sp.sqrt(1 - x)) / sp.pi
        else:
            F = sp.betainc_regularized(sp.Rational(d, 2), sp.Rational(1, 2), 0, x)
        return _unit_ball(d) * r ** d + F / self.mu

    def g(self, r):
        d, b = self.d, self.s
        fD = sp.gamma(sp.Rational(d + 1, 2)) / (sp.pi ** sp.Rational(d + 1, 2) * (2 * b) ** d) * \
            (1 + r ** 2 / (4 * b ** 2)) ** (-sp.Rational(d + 1, 2))
        return 1 + fD / self.mu

    def _disp(self, rng, m):
        np = _np()
        z = rng.standard_normal((m, self.d))
        g = np.abs(rng.standard_normal((m, 1)))
        return _f(self.s) * z / g

    def _fD(self, t):
        d, b = self.d, _f(self.s)
        c = math.gamma((d + 1) / 2) / (math.pi ** ((d + 1) / 2) * b ** d)
        return c * (1 + (t / b) ** 2) ** (-(d + 1) / 2)

    def sample(self, rng, reg):
        """exact: Poisson centres in a ball around the window; far centres by a dominating Pareto-type Poisson
        proposal thinned by the displacement density, with siblings added (Slivnyak) and size-bias correction"""
        np = _np()
        d, mu, lam, b = self.d, _f(self.mu), _f(self.lam), _f(self.s)
        c0, rho = _win_ball(reg)
        R0 = max(10 * b, rho, 1e-300)
        Rin = rho + R0
        vol_in = _f(_unit_ball(d)) * Rin ** d
        k = int(rng.poisson(mu * vol_in))
        if k > 10 ** 6:
            raise NotImplementedError("RandomPointConfiguration: more than 10^6 cluster centres are needed; this is "
                                      "not supported")
        dirs = rng.standard_normal((k, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) if k else 1
        cen = c0 + dirs * (Rin * rng.uniform(0, 1, size=(k, 1)) ** (1.0 / d))
        counts = rng.poisson(lam, size=k)
        parts = []
        if counts.sum():
            P = np.repeat(cen, counts, axis=0) + self._disp(rng, int(counts.sum()))
            parts.append(P[reg.contains(P)])
        # far centres (|c - c0| > Rin)
        Wm = _f(reg.measure())
        Sd = d * _f(_unit_ball(d))
        cd = math.gamma((d + 1) / 2) / math.pi ** ((d + 1) / 2)
        A = mu * lam * Wm * Sd * cd * b * (1 + rho / R0) ** (d - 1)
        m = int(rng.poisson(A / R0))
        for _ in range(m):
            t = R0 / (1.0 - rng.uniform())
            s_ = t + rho
            gval = mu * lam * Wm * self._fD(t) * Sd * s_ ** (d - 1)
            if rng.uniform() * A / t ** 2 > gval:
                continue
            u = rng.standard_normal(d)
            c = c0 + s_ * u / np.linalg.norm(u)
            Y = reg.uniform(rng, 1)[0]
            if rng.uniform() * self._fD(t) > self._fD(float(np.linalg.norm(Y - c))):
                continue
            ns = int(rng.poisson(lam))
            sib = c + self._disp(rng, ns) if ns else np.zeros((0, d))
            sib = sib[reg.contains(sib)] if ns else sib
            if rng.uniform() * (1 + len(sib)) >= 1:
                continue
            parts.append(np.vstack([Y[None, :], sib]))
        return np.vstack(parts) if parts else np.zeros((0, d))


_TAILS = {}


class VarianceGammaPP(PS.PP):
    def parse(self, a, fname):
        if len(a) != 5:
            raise ValueError(f"{fname}: VarianceGammaPointProcess expects [mu, lambda, alpha, beta, d]")
        self.mu = _p(a[0], fname, "the cluster centre density")
        self.lam = _p(a[1], fname, "the cluster mean")
        self.al = _p(a[2], fname, "the shape parameter alpha")
        self.be = _p(a[3], fname, "the shape parameter beta")
        for v, w in ((self.mu, "the cluster centre density"), (self.lam, "the cluster mean"),
                     (self.al, "alpha"), (self.be, "beta")):
            _pos(v, fname, w)
        self.d = _dim(a[4], fname)
        self.p = [self.mu, self.lam, self.al, self.be]

    def intensity(self):
        return self.mu * self.lam

    def count(self, reg):
        raise NotImplementedError("PointCountDistribution: the count distribution of VarianceGammaPointProcess has "
                                  "no closed form")

    def fD(self, r):
        """density (in R^d) of the difference of two displacements at distance r: N(0, U I) with U ~ Gamma(2 alpha,
        beta)"""
        d, a2, be = self.d, 2 * self.al, self.be
        nu = a2 - sp.Rational(d, 2)
        return 2 * (r ** 2 * be / 2) ** (nu / 2) * sp.besselk(nu, r * sp.sqrt(2 / be)) / \
            (sp.gamma(a2) * be ** a2 * (2 * sp.pi) ** sp.Rational(d, 2))

    def g(self, r):
        return 1 + self.fD(r) / self.mu

    def K(self, r):
        if not (self.numeric() and _num(r)):
            raise NotImplementedError("RipleyK: the K function of VarianceGammaPointProcess is only available for "
                                      "numeric parameters and radii (numerical integration)")
        import mpmath
        d, a2, be = self.d, 2 * _f(self.al), _f(self.be)
        rv = _f(r)
        if rv == 0:
            return sp.Integer(0)
        # P(|D| <= r) = E[P(ChiSquare(d) <= r^2/U)], U ~ Gamma(2 alpha, beta)
        f = lambda u: mpmath.gammainc(d / 2.0, 0, rv * rv / (2 * u), regularized=True) * \
            u ** (a2 - 1) * mpmath.exp(-u / be) / (mpmath.gamma(a2) * be ** a2)  # noqa: E731
        F = mpmath.quad(f, [0, be * a2, mpmath.inf])
        return sp.Float(float(_unit_ball(d).evalf()) * rv ** d + float(F) / _f(self.mu), 15)

    def tail_radius(self, nexp):
        import mpmath
        d, a2, be = self.d, 2 * _f(self.al), _f(self.be)
        a1 = _f(self.al)

        def tail(R_):
            f = lambda u: mpmath.gammainc(d / 2.0, R_ * R_ / (2 * u), mpmath.inf, regularized=True) * \
                u ** (a1 - 1) * mpmath.exp(-u / be) / (mpmath.gamma(a1) * be ** a1)  # noqa: E731
            return float(mpmath.quad(f, [0, be * a1, 10 * be * (a1 + 10), mpmath.inf]))
        R_ = math.sqrt(be * a1 * d) + 1e-12
        while tail(R_) * nexp > TAIL_EPS:
            R_ *= 1.5
            if R_ > 1e8:
                raise NotImplementedError("RandomPointConfiguration: the cluster tail is too wide to simulate")
        return R_

    def sample(self, rng, reg):
        np = _np()
        d = self.d
        nexp = _f(self.mu) * _f(self.lam) * _f(reg.measure())
        key = (self.e, reg.e)
        if key not in _TAILS:
            _TAILS[key] = self.tail_radius(max(nexp, 1.0))
        pad = _TAILS[key]
        a1, be = _f(self.al), _f(self.be)

        def disp(m):
            u = rng.gamma(a1, be, size=(m, 1))
            return rng.standard_normal((m, d)) * np.sqrt(u)
        return _poisson_cluster(rng, reg, _f(self.mu), _f(self.lam), pad, disp, d)


class NeymanScottPP(PS.PP):
    def parse(self, a, fname):
        if len(a) != 4:
            raise ValueError(f"{fname}: NeymanScottPointProcess expects [mu, lambda, dist, d]")
        mu = a[0]
        self.mu_fn = None
        if isinstance(mu, (str, bool)) or is_list(mu):
            raise TypeError(f"{fname}: the centre density must be a positive number or a function")
        if isinstance(mu, Expr) and mu.head == "Function":
            self.mu_fn = mu
            self.mu = None
        else:
            self.mu = _p(mu, fname, "the cluster centre density")
            _pos(self.mu, fname, "the cluster centre density")
        self.lam = _p(a[1], fname, "the cluster mean")
        _pos(self.lam, fname, "the cluster mean")
        self.dist = a[2]
        if not (isinstance(self.dist, Expr) and isinstance(self.dist.head, str) and
                self.dist.head.endswith("Distribution")):
            if isinstance(self.dist, Symbol):
                self.dist_sym = True
            else:
                raise TypeError(f"{fname}: the cluster distribution must be a distribution")
        self.d = _dim(a[3], fname)
        self.p = [self.lam] + ([self.mu] if self.mu is not None else [])
        self.stationary = self.mu_fn is None

    def numeric(self):
        return all(_num(v) for v in self.p) and isinstance(self.dist, Expr) and \
            not _has_free_symbols(self.dist)

    def intensity(self):
        if self.mu_fn is not None:
            raise NotImplementedError("MeanPointDensity: a NeymanScottPointProcess with a centre density function has "
                                      "no constant mean density")
        return self.mu * self.lam

    def count(self, reg):
        raise NotImplementedError("PointCountDistribution: the count distribution of NeymanScottPointProcess has no "
                                  "closed form")

    def _multi(self):
        dm = _ev(Expr("RandomVariate", self.dist))
        return is_list(dm)

    def _disp_fn(self, rng):
        np = _np()
        d = self.d
        multi = self._multi()

        def disp(m):
            if m == 0:
                return np.zeros((0, d))
            v = _ev(Expr("RandomVariate", self.dist, m))
            A = np.array(S(v) if not is_list(v) else [[_f(S(c)) for c in (row.args if is_list(row) else [row])]
                                                       for row in v.args], dtype=float)
            if multi:
                A = A.reshape(m, -1)
                if A.shape[1] != d:
                    raise ValueError(f"RandomPointConfiguration: the cluster distribution has dimension {A.shape[1]}, "
                                     f"the process {d}")
                return A
            A = A.reshape(m, 1)
            u = rng.standard_normal((m, d))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            return u * A
        return disp, multi

    def tail_radius(self, nexp, multi):
        fname = "RandomPointConfiguration"
        eps = TAIL_EPS / max(nexp, 1.0)
        if multi:
            h = self.dist.head
            if h in ("MultinormalDistribution", "BinormalDistribution"):
                import numpy as np
                cov = _ev(Expr("Covariance", self.dist))
                mean = _ev(Expr("Mean", self.dist))
                C = np.array([[_f(S(c)) for c in row.args] for row in cov.args], dtype=float)
                m_ = np.array([_f(S(c)) for c in mean.args], dtype=float)
                lmax = float(np.linalg.eigvalsh(C).max())
                q = _chi2_quantile(self.d, eps)
                return float(np.linalg.norm(m_)) + math.sqrt(lmax * q)
            raise NotImplementedError(f"{fname}: multivariate cluster distributions other than MultinormalDistribution "
                                      "and BinormalDistribution are not supported (no tail bound)")

        def tailp(t):
            sf = _ev(Expr("SurvivalFunction", self.dist, sp.Float(t)))
            cdf = _ev(Expr("CDF", self.dist, sp.Float(-t)))
            return _f(S(sf)) + _f(S(cdf))
        t = 1.0
        while tailp(t) > eps:
            t *= 2
            if t > 1e9:
                raise NotImplementedError(f"{fname}: the cluster distribution is too heavy-tailed to simulate by "
                                          "window dilation")
        lo = 0.0
        for _ in range(60):
            mid = (lo + t) / 2
            if tailp(mid) > eps:
                lo = mid
            else:
                t = mid
        return t

    def sample(self, rng, reg):
        np = _np()
        d = self.d
        disp, multi = self._disp_fn(rng)
        if self.mu_fn is None:
            mu = _f(self.mu)
            nexp = mu * _f(self.lam) * _f(reg.measure())
            key = (self.e, reg.e)
            if key not in _TAILS:
                _TAILS[key] = self.tail_radius(nexp, multi)
            pad = _TAILS[key]
            return _poisson_cluster(rng, reg, mu, _f(self.lam), pad, disp, d)
        ip = PS.InhomPP(Expr("InhomogeneousPoissonPointProcess", self.mu_fn, d), "RandomPointConfiguration")
        lo, hi = reg.bbox()
        probe_reg = PS.Region(Expr("Cuboid", R(lo), R(hi)), "RandomPointConfiguration")
        xs = sp.symbols(f"x08c1:{d + 1}", real=True)
        f = sp.lambdify(xs, ip._expr(xs), "numpy")
        P = probe_reg.uniform(rng, 4096)
        mmax = float(np.max(np.broadcast_to(np.asarray(f(*P.T), dtype=float), (len(P),))))
        nexp = 1.25 * mmax * _f(self.lam) * _f(reg.measure())
        pad = self.tail_radius(nexp, multi)

        def centres(lo2, hi2):
            box = PS.Region(Expr("Cuboid", R(lo2), R(hi2)), "RandomPointConfiguration")
            return ip.sample(rng, box)
        return _poisson_cluster(rng, reg, None, _f(self.lam), pad, disp, d, centres=centres)


def _chi2_quantile(d, eps):
    """t with P(ChiSquare(d) > t) = eps (bisection on the regularized upper gamma function)"""
    import mpmath
    lo, hi = 0.0, max(10.0, 4.0 * d)
    while float(mpmath.gammainc(d / 2.0, hi / 2, mpmath.inf, regularized=True)) > eps:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if float(mpmath.gammainc(d / 2.0, mid / 2, mpmath.inf, regularized=True)) > eps:
            lo = mid
        else:
            hi = mid
    return hi


# ─────────────────────────── Gibbs processes ───────────────────────────

class GibbsBase(PS.PP):
    """pairwise or general Gibbs process; subclasses define parse, act(P) (activity at points), pair(r) (pair
    interaction factors) or papangelou(u, X)"""
    takes_method = True
    pairwise = True
    rng_ = None

    def intensity(self):
        raise NotImplementedError(f"MeanPointDensity: the intensity of {self.name} has no closed form")

    def count(self, reg):
        raise NotImplementedError(f"PointCountDistribution: the count distribution of {self.name} has no closed form")

    def act(self, P):
        np = _np()
        return np.full(len(P), _f(self.mu))

    def papangelou(self, u, X):
        np = _np()
        a = float(self.act(u[None, :])[0])
        if a == 0 or len(X) == 0:
            return a
        h = self.pair(np.linalg.norm(X - u, axis=1))
        return a * float(np.prod(h))

    def default_steps(self, reg):
        try:
            e = _f(self.mu) * _f(reg.measure())
        except (TypeError, AttributeError):
            e = 100.0
        return int(10000 + 100 * math.ceil(e))

    def sample(self, rng, reg, method=None):
        np = _np()
        fname = "RandomPointConfiguration"
        steps = None
        if method is not None:
            mname, sub = method, []
            if is_list(method) and method.args:
                mname, sub = method.args[0], list(method.args[1:])
            if mname == "Exact":
                raise NotImplementedError(f"{fname}: Method -> \"Exact\" (coupling from the past) is not supported; "
                                          "use \"MCMC\"")
            if mname != "MCMC":
                raise ValueError(f"{fname}: unknown Method {method} (use \"MCMC\")")
            for o in sub:
                if _opt_name(o) == "Steps":
                    v = o.args[1]
                    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                        raise ValueError(f"{fname}: \"Steps\" must be a positive integer")
                    steps = v
                else:
                    raise ValueError(f"{fname}: unknown \"MCMC\" suboption {o}")
        if steps is None:
            steps = self.default_steps(reg)
        return birth_death(self, rng, reg, steps)


def birth_death(proc, rng, reg, steps):
    """Geyer-Moller birth-death Metropolis-Hastings sampler from the empty configuration"""
    np = _np()
    d = proc.d
    W = _f(reg.measure())
    X = np.zeros((64, d))
    n = 0
    B = 4096
    cand = reg.uniform(rng, B)
    ci = 0
    U = rng.uniform(size=(steps, 3))
    for k in range(steps):
        if U[k, 0] < 0.5:
            if ci >= B:
                cand, ci = reg.uniform(rng, B), 0
            u = cand[ci]
            ci += 1
            lam = proc.papangelou(u, X[:n])
            _check_lam(lam)
            if U[k, 1] * (n + 1) < lam * W:
                if n == len(X):
                    if n >= 10 ** 5:
                        raise ValueError("RandomPointConfiguration: more than 10^5 points; the Gibbs process appears "
                                         "to be unstable")
                    X = np.vstack([X, np.zeros_like(X)])
                X[n] = u
                n += 1
        elif n:
            i = int(U[k, 2] * n)
            xi = X[i].copy()
            X[i] = X[n - 1]
            lam = proc.papangelou(xi, X[:n - 1])
            _check_lam(lam)
            if U[k, 1] * W * lam < n:
                n -= 1
            else:
                X[n - 1] = X[i]
                X[i] = xi
    return X[:n].copy()


def _check_lam(lam):
    if not (lam >= 0 and math.isfinite(lam)):
        raise ValueError("RandomPointConfiguration: the Papangelou conditional density must be finite and "
                         f"non-negative (got {lam})")


def _gpos(v, fname, what):
    _pos(v, fname, what)


def _unit_interval(v, fname, what):
    ok = None
    if _num(v):
        ok = bool(v.is_real and 0 <= v <= 1)
    _cond(Expr("LessEqual", 0, R(v), 1), ok, f"{fname}: {what} must be between 0 and 1")


class HardcorePP(GibbsBase):
    def parse(self, a, fname):
        if len(a) != 3:
            raise ValueError(f"{fname}: HardcorePointProcess expects [mu, rh, d]")
        self.mu = _p(a[0], fname, "the intensity")
        self.rh = _p(a[1], fname, "the hard-core radius")
        _gpos(self.mu, fname, "the intensity")
        _gpos(self.rh, fname, "the hard-core radius")
        self.d = _dim(a[2], fname)
        self.p = [self.mu, self.rh]

    def pair(self, r):
        return (r >= _f(self.rh)).astype(float)

    def range_(self):
        return _f(self.rh)


class StraussPP(GibbsBase):
    def parse(self, a, fname):
        if len(a) != 4:
            raise ValueError(f"{fname}: StraussPointProcess expects [mu, gamma, rs, d]")
        self.mu = _p(a[0], fname, "the intensity")
        self.ga = _p(a[1], fname, "the interaction parameter")
        self.rs = _p(a[2], fname, "the interaction radius")
        _gpos(self.mu, fname, "the intensity")
        _unit_interval(self.ga, fname, "the interaction parameter gamma")
        _gpos(self.rs, fname, "the interaction radius")
        self.d = _dim(a[3], fname)
        self.p = [self.mu, self.ga, self.rs]

    def pair(self, r):
        np = _np()
        return np.where(r < _f(self.rs), _f(self.ga), 1.0)

    def range_(self):
        return _f(self.rs)


class StraussHardcorePP(GibbsBase):
    def parse(self, a, fname):
        if len(a) != 5:
            raise ValueError(f"{fname}: StraussHardcorePointProcess expects [mu, gamma, rh, rs, d]")
        self.mu = _p(a[0], fname, "the intensity")
        self.ga = _p(a[1], fname, "the interaction parameter")
        self.rh = _p(a[2], fname, "the hard-core radius")
        self.rs = _p(a[3], fname, "the interaction radius")
        _gpos(self.mu, fname, "the intensity")
        _unit_interval(self.ga, fname, "the interaction parameter gamma")
        _gpos(self.rh, fname, "the hard-core radius")
        _gpos(self.rs, fname, "the interaction radius")
        _le(self.rh, self.rs, fname, "the hard-core radius must not exceed the interaction radius")
        self.d = _dim(a[4], fname)
        self.p = [self.mu, self.ga, self.rh, self.rs]

    def pair(self, r):
        np = _np()
        return np.where(r < _f(self.rh), 0.0, np.where(r < _f(self.rs), _f(self.ga), 1.0))

    def range_(self):
        return _f(self.rs)


def lens_fraction(r, rp, d, fname="PenttinenPointProcess"):
    """|B(0, rp) n B(r, rp)| / |B(0, rp)| for an array of distances r (0 beyond 2 rp)"""
    np = _np()
    z = np.clip(np.asarray(r, dtype=float) / (2 * rp), 0.0, 1.0)
    if d == 1:
        return 1 - z
    if d == 2:
        return (2 / math.pi) * (np.arccos(z) - z * np.sqrt(1 - z * z))
    if d == 3:
        return (1 - z) ** 2 * (2 + z) / 2
    try:
        from scipy.special import betainc
    except ImportError:
        raise NotImplementedError(f"{fname}: dimensions above 3 need SciPy (incomplete beta function), which is not "
                                  "installed")
    return betainc((d + 1) / 2, 0.5, 1 - z * z)


class PenttinenPP(GibbsBase):
    def parse(self, a, fname):
        if len(a) != 4:
            raise ValueError(f"{fname}: PenttinenPointProcess expects [mu, gamma, rp, d]")
        self.mu = _p(a[0], fname, "the intensity")
        self.ga = _p(a[1], fname, "the interaction parameter")
        self.rp = _p(a[2], fname, "the interaction radius")
        _gpos(self.mu, fname, "the intensity")
        _unit_interval(self.ga, fname, "the interaction parameter gamma")
        _gpos(self.rp, fname, "the interaction radius")
        self.d = _dim(a[3], fname)
        self.p = [self.mu, self.ga, self.rp]

    def pair(self, r):
        np = _np()
        f = lens_fraction(r, _f(self.rp), self.d)
        g = _f(self.ga)
        if g == 0:
            return np.where(f > 0, 0.0, 1.0)
        return g ** f

    def range_(self):
        return 2 * _f(self.rp)


class DiggleGatesPP(GibbsBase):
    def parse(self, a, fname):
        if len(a) != 3:
            raise ValueError(f"{fname}: DiggleGatesPointProcess expects [mu, rho, d]")
        self.mu = _p(a[0], fname, "the intensity")
        self.rho = _p(a[1], fname, "the interaction radius")
        _gpos(self.mu, fname, "the intensity")
        _gpos(self.rho, fname, "the interaction radius")
        self.d = _dim(a[2], fname)
        self.p = [self.mu, self.rho]

    def pair(self, r):
        np = _np()
        rho = _f(self.rho)
        return np.where(r < rho, np.sin(math.pi * r / (2 * rho)) ** 2, 1.0)

    def range_(self):
        return _f(self.rho)


class DiggleGrattonPP(GibbsBase):
    def parse(self, a, fname):
        if len(a) != 5:
            raise ValueError(f"{fname}: DiggleGrattonPointProcess expects [mu, kappa, delta, rho, d]")
        self.mu = _p(a[0], fname, "the intensity")
        self.ka = _p(a[1], fname, "the interaction parameter kappa")
        self.de = _p(a[2], fname, "the hard-core radius delta")
        self.rho = _p(a[3], fname, "the interaction radius rho")
        _gpos(self.mu, fname, "the intensity")
        _nonneg(self.ka, fname, "kappa")
        _nonneg(self.de, fname, "the hard-core radius delta")
        _gpos(self.rho, fname, "the interaction radius rho")
        _le(self.de, self.rho, fname, "delta must not exceed rho")
        self.d = _dim(a[4], fname)
        self.p = [self.mu, self.ka, self.de, self.rho]

    def pair(self, r):
        np = _np()
        de, rho, ka = _f(self.de), _f(self.rho), _f(self.ka)
        mid = (r >= de) & (r < rho)
        out = np.where(r < de, 0.0, 1.0)
        if rho > de:
            out = np.where(mid, np.clip((r - de) / (rho - de), 0, 1) ** ka, out)
        return out

    def range_(self):
        return _f(self.rho)


_CMP = {"Less": "__lt__", "Greater": "__gt__", "LessEqual": "__le__", "GreaterEqual": "__ge__"}


def np_compile(e, names):
    """a numpy function of the variables `names` (arrays) for an evaluated MikoshiLang expression with If,
    Piecewise, comparisons and And/Or/Not around SymPy-convertible arithmetic; ValueError if not compilable"""
    np = _np()
    syms = [sp.Symbol(n) for n in names]
    if isinstance(e, Expr) and e.head == "If" and len(e.args) == 3:
        c, a, b = (np_compile(x, names) for x in e.args)
        return lambda *v: np.where(np.asarray(c(*v), dtype=bool), a(*v), b(*v))
    if isinstance(e, Expr) and e.head == "Piecewise" and e.args and is_list(e.args[0]):
        pcs = [(np_compile(pc.args[1], names), np_compile(pc.args[0], names)) for pc in e.args[0].args]
        dflt = np_compile(e.args[1], names) if len(e.args) > 1 else (lambda *v: 0.0)

        def pw(*v):
            shape = np.broadcast(*[np.asarray(x) for x in v]).shape
            out = np.broadcast_to(np.asarray(dflt(*v), dtype=float), shape).copy()
            done = np.zeros(shape, dtype=bool)
            for cv, vv in pcs:
                m = np.broadcast_to(np.asarray(cv(*v), dtype=bool), shape) & ~done
                out = np.where(m, np.broadcast_to(np.asarray(vv(*v), dtype=float), shape), out)
                done |= m
            return out
        return pw
    if isinstance(e, Expr) and e.head in _CMP and len(e.args) == 2:
        a, b = (np_compile(x, names) for x in e.args)
        op = _CMP[e.head]
        return lambda *v: getattr(np.asarray(a(*v), dtype=float), op)(np.asarray(b(*v), dtype=float))
    if isinstance(e, Expr) and e.head in ("And", "Or") and e.args:
        fs = [np_compile(x, names) for x in e.args]
        red = np.logical_and if e.head == "And" else np.logical_or
        return lambda *v: red.reduce([np.asarray(f(*v), dtype=bool) for f in fs])
    if isinstance(e, Expr) and e.head == "Not" and len(e.args) == 1:
        f = np_compile(e.args[0], names)
        return lambda *v: ~np.asarray(f(*v), dtype=bool)
    if e is True or e is False:
        return lambda *v: e
    v = S(e)
    if isinstance(v, list) or not isinstance(v, sp.Basic):
        raise ValueError("not a scalar expression")
    if v.free_symbols - set(syms):
        raise ValueError("the function has unknown symbols")
    return sp.lambdify(syms, v, "numpy")


_GIBBS_FORMS = ("PairPotential", "PairInteraction", "Papangelou", "Density")


class GibbsPP(GibbsBase):
    def parse(self, a, fname):
        if len(a) != 2:
            raise ValueError(f"{fname}: GibbsPointProcess expects [spec, d]")
        spec = a[0]
        if not (is_list(spec) and spec.args and isinstance(spec.args[0], str)):
            raise TypeError(f"{fname}: the specification must be {{\"PairPotential\", mu, phi}}, {{\"PairInteraction\", "
                            "mu, h}, {\"Papangelou\", lambda}, or {\"Density\", f}")
        self.form = spec.args[0]
        if self.form not in _GIBBS_FORMS:
            raise ValueError(f"{fname}: unknown specification {self.form!r} (use one of {', '.join(_GIBBS_FORMS)})")
        need = 3 if self.form in ("PairPotential", "PairInteraction") else 2
        if len(spec.args) != need:
            raise ValueError(f"{fname}: {{\"{self.form}\", ...}} needs {need - 1} element(s)")
        self.d = _dim(a[1], fname)
        self.p = []
        self.mu = None
        self.pairwise = self.form in ("PairPotential", "PairInteraction")
        if self.pairwise:
            mu = spec.args[1]
            if isinstance(mu, (str, bool)) or is_list(mu):
                raise TypeError(f"{fname}: the density must be a non-negative number or a function")
            if isinstance(mu, Expr) and mu.head == "Function":
                self.mu_fn = mu
            else:
                self.mu_fn = None
                self.mu = _p(mu, fname, "the density")
                _gpos(self.mu, fname, "the density")
                self.p = [self.mu]
            self.fn_ = spec.args[2]
        else:
            self.fn_ = spec.args[1]
        if isinstance(self.fn_, (str, bool, int, float)) or is_list(self.fn_):
            raise TypeError(f"{fname}: the {self.form} function must be a function")
        self._cache = {}

    def numeric(self):
        return all(_num(v) for v in self.p)

    def default_steps(self, reg):
        if self.mu is not None:
            return super().default_steps(reg)
        return 10000

    def act(self, P):
        np = _np()
        if self.mu is not None:
            return np.full(len(P), _f(self.mu))
        if "act" not in self._cache:
            names = [f"x08g{i + 1}" for i in range(self.d)]
            e = _ev(Expr(self.mu_fn, *(Symbol(x) for x in names)))
            try:
                self._cache["act"] = np_compile(e, names)
            except (TypeError, ValueError, sp.SympifyError):
                raise ValueError("GibbsPointProcess: the density function must give numbers")
        out = np.broadcast_to(np.asarray(self._cache["act"](*P.T), dtype=float), (len(P),))
        if np.any(out < 0) or not np.all(np.isfinite(out)):
            raise ValueError("RandomPointConfiguration: the density must be finite and non-negative")
        return out

    def pair(self, r):
        np = _np()
        if "pair" not in self._cache:
            self._cache["pair"] = self._compile_pair()
        with np.errstate(all="ignore"):
            out = np.broadcast_to(np.asarray(self._cache["pair"](r), dtype=float), r.shape).copy()
        out[np.isnan(out)] = 0.0 if self.form == "PairPotential" else np.nan
        if np.any(out < 0) or np.any(np.isnan(out)) or np.any(np.isinf(out)):
            raise ValueError("RandomPointConfiguration: the pair interaction must be finite and non-negative")
        return out

    def _compile_pair(self):
        """vectorised pair interaction: SymPy lambdify when the function has a symbolic form, else evaluation of the
        function at each distance"""
        np = _np()
        pot = self.form == "PairPotential"
        try:
            e = _ev(Expr(self.fn_, Symbol("x08r")))
            f0 = np_compile(e, ["x08r"])
            if pot:
                return lambda r: np.exp(-np.asarray(f0(r), dtype=float))
            return f0
        except (TypeError, ValueError, NotImplementedError, sp.SympifyError):
            pass

        def f(r):
            out = []
            for rv in np.atleast_1d(r):
                val = _ev(Expr(self.fn_, R(float(rv))))
                if val == Symbol("Infinity") and pot:
                    out.append(0.0)
                    continue
                try:
                    y = _f(S(val))
                except (TypeError, ValueError, AttributeError):
                    raise ValueError(f"GibbsPointProcess: the pair function gave {val} (a number is needed)")
                out.append(math.exp(-y) if pot else y)
            return np.array(out)
        return f

    def papangelou(self, u, X):
        if self.pairwise:
            return super().papangelou(u, X)
        q = L(*(R(float(c)) for c in u))
        pts = L(*(L(*(R(float(c)) for c in p)) for p in X))
        if self.form == "Papangelou":
            v = _ev(Expr(self.fn_, pts, q))
        else:
            num = _ev(Expr(self.fn_, L(*(list(pts.args) + [q]))))
            den = _ev(Expr(self.fn_, pts))
            dn = _f(S(den))
            if dn == 0:
                raise ValueError("RandomPointConfiguration: the density is zero at the current configuration")
            v = R(_f(S(num)) / dn)
        try:
            return _f(S(v))
        except (TypeError, ValueError, AttributeError):
            raise ValueError(f"RandomPointConfiguration: the {self.form} function must give non-negative numbers")


_NEW = {"CauchyPointProcess": CauchyPP, "VarianceGammaPointProcess": VarianceGammaPP,
        "NeymanScottPointProcess": NeymanScottPP, "GibbsPointProcess": GibbsPP,
        "HardcorePointProcess": HardcorePP, "StraussPointProcess": StraussPP,
        "StraussHardcorePointProcess": StraussHardcorePP, "PenttinenPointProcess": PenttinenPP,
        "DiggleGatesPointProcess": DiggleGatesPP, "DiggleGrattonPointProcess": DiggleGrattonPP}
PS.PCLASSES.update(_NEW)

_SIM = ("RandomPointConfiguration simulates it reproducibly with SeedRandom (Wolfram's random stream differs)")
_GIBBS_SIM = ("RandomPointConfiguration[proc, reg] uses the Geyer-Moller birth-death Metropolis-Hastings sampler "
              "(birth and death proposals with probability 1/2 each, uniform birth locations in reg) from the empty "
              "configuration, with no window expansion, for 10000 + 100 Ceiling[mu RegionMeasure[reg]] steps; "
              "Method -> {\"MCMC\", \"Steps\" -> n} sets the steps, Method -> \"Exact\" (coupling from the past) "
              "raises NotImplementedError. MeanPointDensity, PointCountDistribution and RipleyK-type functions of "
              "the process raise NotImplementedError (no closed form); estimates from simulated data work")

_NOTES = {
    "CauchyPointProcess": "CauchyPointProcess[mu, lambda, b, d]: Neyman-Scott process with Poisson(mu) centres, "
                          "PoissonDistribution[lambda] points per cluster and MultivariateTDistribution[b^2 "
                          "IdentityMatrix[d], 1] displacements. Closed forms: MeanPointDensity mu lambda; RipleyK "
                          "Pi^(d/2)/Gamma[d/2 + 1] r^d + I(d/2, 1/2; r^2/(r^2 + 4 b^2))/mu (2D: (1 - 2 b/Sqrt[r^2 + "
                          "4 b^2])/mu; the difference of two displacements is Cauchy with scale 2 b); PairCorrelationG "
                          "1 + f(r)/mu with f the density of that difference. " + _SIM + "; far cluster centres "
                          "are handled exactly (dominating Pareto-type proposals thinned by the displacement density, "
                          "siblings added and size bias removed). PointCountDistribution raises NotImplementedError",
    "VarianceGammaPointProcess": "VarianceGammaPointProcess[mu, lambda, alpha, beta, d]: Neyman-Scott process with "
                                 "Poisson(mu) centres, PoissonDistribution[lambda] points per cluster and displacements "
                                 "MultinormalDistribution[u IdentityMatrix[d]] with u ~ GammaDistribution[alpha, beta] "
                                 "drawn per point. MeanPointDensity mu lambda; PairCorrelationG 1 + f(r)/mu with the "
                                 "closed form f(r) = 2 (r^2 beta/2)^(nu/2) BesselK[nu, r Sqrt[2/beta]]/(Gamma[2 alpha] "
                                 "beta^(2 alpha) (2 Pi)^(d/2)), nu = 2 alpha - d/2; RipleyK numerically (mpmath "
                                 "quadrature) for numeric parameters and radius only (symbolic raises "
                                 "NotImplementedError). " + _SIM + " (centres in the window dilated by the tail radius "
                                 "with expected missed points < 1e-10)",
    "NeymanScottPointProcess": "NeymanScottPointProcess[mu, lambda, rdist, d] (displacement R u with R ~ rdist, u "
                               "uniform on the unit sphere) and NeymanScottPointProcess[mu, lambda, mdist, d] "
                               "(displacements from the d-variate mdist); mu a positive number or a function of the "
                               "d coordinates (centres simulated as an inhomogeneous Poisson process by thinning). "
                               "MeanPointDensity mu lambda for constant mu. " + _SIM + "; the window is dilated by the "
                               "tail radius of rdist (from its CDF and SurvivalFunction) or, for Multinormal/Binormal "
                               "mdist, by |mean| + Sqrt[largest covariance eigenvalue chi-square(d) quantile], keeping "
                               "the expected number of missed points below 1e-10; other multivariate distributions "
                               "raise NotImplementedError. RipleyK-type functions and PointCountDistribution raise "
                               "NotImplementedError",
    "GibbsPointProcess": "GibbsPointProcess[{\"PairPotential\", mu, phi}, d] (pair interaction Exp[-phi[r]]), "
                         "GibbsPointProcess[{\"PairInteraction\", mu, h}, d] (Papangelou density mu Product[h[|p_i - "
                         "q|]]), GibbsPointProcess[{\"Papangelou\", lam}, d] (lam[{p1, ...}, q]) and "
                         "GibbsPointProcess[{\"Density\", f}, d] (lam = f[x U {q}]/f[x]); mu a number or a function of "
                         "the coordinates. phi, h and mu are compiled with SymPy's lambdify; Papangelou and Density "
                         "functions are evaluated for every proposal (slow; 10000 steps by default). Non-finite or "
                         "negative values raise ValueError; more than 10^5 points raises ValueError (unstable). "
                         + _GIBBS_SIM,
    "HardcorePointProcess": "HardcorePointProcess[mu, rh, d]: pair interaction h(r) = 0 for r < rh, 1 otherwise; "
                            "density mu^n Product[h] with respect to PoissonPointProcess[1, d]. " + _GIBBS_SIM,
    "StraussPointProcess": "StraussPointProcess[mu, gamma, rs, d] with 0 <= gamma <= 1: h(r) = gamma for r < rs, "
                           "else 1; density mu^n gamma^s with s the number of (unordered) close pairs, Papangelou "
                           "density mu gamma^m (m = number of points within rs of q). " + _GIBBS_SIM,
    "StraussHardcorePointProcess": "StraussHardcorePointProcess[mu, gamma, rh, rs, d] with 0 <= gamma <= 1 and rh <= "
                                   "rs: h(r) = 0 for r < rh, gamma for rh <= r < rs, 1 otherwise. " + _GIBBS_SIM,
    "PenttinenPointProcess": "PenttinenPointProcess[mu, gamma, rp, d] with 0 <= gamma <= 1: h(r) = gamma^f(r) for "
                             "r < 2 rp, f(r) the overlap fraction of two balls of radius rp at distance r (2D (2/Pi)"
                             "(ArcCos[r/(2 rp)] - r/(2 rp) Sqrt[1 - r^2/(4 rp^2)]), 3D (r - 2 rp)^2 (4 rp + r)/(16 "
                             "rp^3), general d the regularized incomplete beta I(1 - r^2/(4 rp^2); (d + 1)/2, 1/2), "
                             "which needs SciPy for d > 3). " + _GIBBS_SIM,
    "DiggleGatesPointProcess": "DiggleGatesPointProcess[mu, rho, d]: h(r) = Sin[Pi r/(2 rho)]^2 for r < rho, else 1. "
                               + _GIBBS_SIM,
    "DiggleGrattonPointProcess": "DiggleGrattonPointProcess[mu, kappa, delta, rho, d] with kappa >= 0 and 0 <= delta <= "
                                 "rho: h(r) = 0 for r < delta, ((r - delta)/(rho - delta))^kappa for delta <= r < rho, "
                                 "1 otherwise. " + _GIBBS_SIM,
}


def _register(name):
    @fn(name, area=AREA, status="added",
        notes=_NOTES[name] + ". Inert; the constructor validates numeric parameters (positive rates and radii, "
              "parameter ranges as stated, positive integer dimension; invalid values raise ValueError, wrong kinds "
              "TypeError); PointProcessParameterQ and PointProcessParameterAssumptions report the conditions. "
              "Check: reference (closed forms, Wolfram documentation) and statistical (seeded simulations against "
              "exact means, hard-core distances, Poisson special cases) in tests.")
    def _ctor(*args: Expr) -> Expr:
        PS.PCLASSES[name](Expr(name, *args), name)
        raise Unevaluated()
    return _ctor


for _n in _NEW:
    _register(_n)
