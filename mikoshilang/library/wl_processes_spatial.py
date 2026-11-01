"""Wolfram batch W25: spatial point processes.

PoissonPointProcess, BinomialPointProcess, InhomogeneousPoissonPointProcess, ThomasPointProcess and
MaternPointProcess are inert expressions validated by their constructors.  Observation regions are Rectangle,
Cuboid, Disk (circle or axis-aligned ellipse) and Ball in any dimension, with exact measures.
RandomPointConfiguration simulates (reproducibly with SeedRandom) and returns an inert ``SpatialPointData`` object
with a few properties; PointCountDistribution, RipleyK, BesagL, PairCorrelationG, NearestNeighborG, EmptySpaceF
and SpatialJ give exact closed forms where they are known for the process.  Estimates from point data need
Wolfram's boundary-correction defaults and are not supported (NotImplementedError).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel_lists as KL
from .core import Unevaluated, fn, L, R, S, sym, is_list, has_float, numeric_out, rule
from . import wl_stats as W

AREA = "spatial point processes (W25)"


def _ev(e):
    from ..evaluate import evaluate
    return evaluate(e)


def _num(v):
    return isinstance(v, sp.Basic) and v.is_number


def _f(v):
    return float(sp.N(v))


def _p(v, fname, what="parameter"):
    if isinstance(v, (str, bool)) or is_list(v):
        raise TypeError(f"{fname}: {what} must be a number or symbol")
    return sym(v, fname)


_CONDS = []
_LAX = [False]


def _bad(msg):
    if not _LAX[0]:
        raise ValueError(msg)


def _pos(v, fname, what):
    _CONDS.append(Expr("Greater", R(v), 0))
    if _num(v) and not (v.is_real and v > 0):
        _bad(f"{fname}: {what} must be a positive real number")


def _dim(d, fname):
    if isinstance(d, bool) or not isinstance(d, int):
        if isinstance(d, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the dimension must be a positive integer")
    if d < 1:
        raise ValueError(f"{fname}: the dimension must be a positive integer")
    return d


def _unit_ball(d):
    return sp.pi ** sp.Rational(d, 2) / sp.gamma(sp.Rational(d, 2) + 1)


# ─────────────────────────── regions ───────────────────────────

def _vec(v, fname, what):
    if not is_list(v) or not v.args:
        raise TypeError(f"{fname}: {what} must be a list of coordinates")
    out = [_p(x, fname, what) for x in v.args]
    return out


class Region:
    """kind in box / ellipsoid; lo, hi (box) or c, radii (ellipsoid), SymPy values"""

    def __init__(self, e, fname):
        self.e = e
        h = e.head if isinstance(e, Expr) else None
        a = list(e.args) if isinstance(e, Expr) else []
        if h in ("Rectangle", "Cuboid"):
            dflt = 2 if h == "Rectangle" else 3
            if not a:
                lo = [sp.Integer(0)] * dflt
                hi = [sp.Integer(1)] * dflt
            elif len(a) == 1:
                lo = _vec(a[0], fname, "the corner")
                hi = [x + 1 for x in lo]
            elif len(a) == 2:
                lo, hi = _vec(a[0], fname, "the corner"), _vec(a[1], fname, "the corner")
            else:
                raise ValueError(f"{fname}: {h} expects [], [pmin] or [pmin, pmax]")
            if h == "Rectangle" and len(lo) != 2:
                raise ValueError(f"{fname}: Rectangle needs 2-D corners")
            if len(lo) != len(hi):
                raise ValueError(f"{fname}: the corners of {h} must have the same dimension")
            self.kind, self.lo, self.hi = "box", lo, hi
            self.dim = len(lo)
            vals = lo + hi
        elif h in ("Disk", "Ball"):
            dflt = 2 if h == "Disk" else 3
            c = _vec(a[0], fname, "the centre") if a else [sp.Integer(0)] * dflt
            if len(a) > 2:
                raise NotImplementedError(f"{fname}: sectors of {h} are not supported")
            r = a[1] if len(a) == 2 else 1
            if is_list(r):
                if h == "Ball":
                    raise ValueError(f"{fname}: Ball takes a scalar radius")
                radii = _vec(r, fname, "the radii")
            else:
                radii = [_p(r, fname, "the radius")] * len(c)
            if h == "Disk" and len(c) != 2:
                raise ValueError(f"{fname}: Disk needs a 2-D centre")
            if len(radii) != len(c):
                raise ValueError(f"{fname}: the radii must match the dimension")
            self.kind, self.c, self.radii = "ellipsoid", c, radii
            self.dim = len(c)
            vals = c + radii
        else:
            if isinstance(e, Symbol):
                raise Unevaluated()
            raise NotImplementedError(f"{fname}: the region {e} is not supported (use Rectangle, Cuboid, Disk or "
                                      "Ball)")
        for v in vals:
            if _num(v) and not v.is_real:
                raise ValueError(f"{fname}: region coordinates must be real")
        self.numeric = all(_num(v) for v in vals)
        if self.numeric:
            if self.kind == "box" and any(not h_ > l_ for l_, h_ in zip(self.lo, self.hi)):
                raise ValueError(f"{fname}: the region must be full-dimensional (pmin < pmax)")
            if self.kind == "ellipsoid" and any(not r_ > 0 for r_ in self.radii):
                raise ValueError(f"{fname}: the radius must be positive")

    def measure(self):
        if self.kind == "box":
            return sp.Mul(*(h - l for l, h in zip(self.lo, self.hi)))
        return _unit_ball(self.dim) * sp.Mul(*self.radii)

    def bbox(self):
        if self.kind == "box":
            return [_f(v) for v in self.lo], [_f(v) for v in self.hi]
        return [_f(c - r) for c, r in zip(self.c, self.radii)], [_f(c + r) for c, r in zip(self.c, self.radii)]

    def contains(self, pts):
        import numpy as np
        if self.kind == "box":
            lo, hi = self.bbox()
            return np.all((pts >= np.array(lo)) & (pts <= np.array(hi)), axis=1)
        c = np.array([_f(v) for v in self.c])
        r = np.array([_f(v) for v in self.radii])
        return (((pts - c) / r) ** 2).sum(axis=1) <= 1.0

    def uniform(self, rng, n):
        import numpy as np
        lo, hi = self.bbox()
        out = np.empty((0, self.dim))
        while len(out) < n:
            m = max(16, int((n - len(out)) * 1.5) + 8)
            cand = rng.uniform(lo, hi, size=(m, self.dim))
            out = np.vstack([out, cand[self.contains(cand)]])
        return out[:n]

    def inside(self, other):
        """True if self is provably contained in other, None if unknown"""
        if self.dim != other.dim or not (self.numeric and other.numeric):
            return None
        if self.e == other.e:
            return True
        if other.kind == "box":
            lo, hi = self.bbox()
            olo, ohi = other.bbox()
            return all(a >= b - 1e-15 for a, b in zip(lo, olo)) and all(a <= b + 1e-15 for a, b in zip(hi, ohi))
        if self.kind == "ellipsoid" and len(set(self.radii)) == 1 and len(set(other.radii)) == 1:
            dist = sp.sqrt(sum((a - b) ** 2 for a, b in zip(self.c, other.c)))
            return bool(dist + self.radii[0] <= other.radii[0])
        if self.kind == "box" and len(set(other.radii)) == 1:
            import itertools
            corners = itertools.product(*zip(self.lo, self.hi))
            return all(sum((x - c) ** 2 for x, c in zip(p, other.c)) <= other.radii[0] ** 2 for p in corners)
        return None


def region_of(e, fname):
    return Region(e, fname)


# ─────────────────────────── point processes ───────────────────────────

class PP:
    stationary = True

    def __init__(self, e, fname):
        self.e, self.name = e, e.head
        self.parse(list(e.args), fname)

    def numeric(self):
        return all(_num(v) for v in self.p)


class PoissonPP(PP):
    def parse(self, a, fname):
        if len(a) != 2:
            raise ValueError(f"{fname}: PoissonPointProcess expects [mu, d]")
        self.mu = _p(a[0], fname, "the intensity")
        _pos(self.mu, fname, "the intensity")
        self.d = _dim(a[1], fname)
        self.p = [self.mu]

    def intensity(self):
        return self.mu

    def K(self, r):
        return _unit_ball(self.d) * r ** self.d

    def G(self, r):
        return 1 - sp.exp(-self.mu * _unit_ball(self.d) * r ** self.d)

    F = G

    def count(self, reg):
        return Expr("PoissonDistribution", R(sp.simplify(self.mu * reg.measure())))

    def sample(self, rng, reg):
        n = int(rng.poisson(_f(self.mu * reg.measure())))
        return reg.uniform(rng, n)


class BinomialPP(PP):
    stationary = False

    def parse(self, a, fname):
        if len(a) != 2:
            raise ValueError(f"{fname}: BinomialPointProcess expects [n, reg]")
        n = a[0]
        if isinstance(n, Symbol):
            self.n = sp.Symbol(n.name)
        elif isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"{fname}: the number of points must be a positive integer")
        else:
            if n < 1:
                _bad(f"{fname}: the number of points must be a positive integer")
            self.n = sp.Integer(n)
        _CONDS.append(Expr("And", Expr("Element", R(self.n), Symbol("Integers")), Expr("Greater", R(self.n), 0)))
        self.reg = region_of(a[1], fname)
        self.d = self.reg.dim
        self.p = [self.n]

    def numeric(self):
        return _num(self.n) and self.reg.numeric

    def intensity(self):
        return self.n / self.reg.measure()

    def count(self, reg):
        inside = reg.inside(self.reg)
        if inside is None or inside is False:
            raise NotImplementedError("PointCountDistribution: the count region must lie inside the region of the "
                                      "BinomialPointProcess (provably, for boxes and balls)")
        return Expr("BinomialDistribution", R(self.n), R(sp.simplify(reg.measure() / self.reg.measure())))

    def sample(self, rng, reg):
        pts = self.reg.uniform(rng, int(self.n))
        if reg.e == self.reg.e:
            return pts
        return pts[reg.contains(pts)]


class InhomPP(PP):
    stationary = False

    def parse(self, a, fname):
        if len(a) != 2:
            raise ValueError(f"{fname}: InhomogeneousPoissonPointProcess expects [f, d]")
        self.fn = a[0]
        if isinstance(a[0], (int, float, str)) or is_list(a[0]):
            raise TypeError(f"{fname}: the density must be a function of the coordinates")
        self.d = _dim(a[1], fname)
        self.p = []

    def numeric(self):
        return True

    def intensity(self):
        raise NotImplementedError("MeanPointDensity: an inhomogeneous Poisson point process has no constant mean "
                                  "density")

    def _expr(self, xs):
        e = _ev(Expr(self.fn, *(Symbol(x.name) for x in xs)))
        v = sp.sympify(S(e)).subs({sp.Symbol(x.name): x for x in xs})
        if v.free_symbols - set(xs):
            raise Unevaluated()
        return v

    def count(self, reg):
        if reg.kind != "box" or not reg.numeric:
            raise NotImplementedError("PointCountDistribution: inhomogeneous counts are supported over Rectangle "
                                      "and Cuboid regions")
        xs = sp.symbols(f"w25x1:{self.d + 1}", real=True)
        v = self._expr(xs)
        for x, lo, hi in zip(xs, reg.lo, reg.hi):
            v = sp.integrate(v, (x, lo, hi))
        if v.has(sp.Integral):
            raise NotImplementedError("PointCountDistribution: the density has no closed-form integral over the "
                                      "region")
        return Expr("PoissonDistribution", R(sp.simplify(v)))

    def sample(self, rng, reg):
        import numpy as np
        xs = sp.symbols(f"w25x1:{self.d + 1}", real=True)
        v = self._expr(xs)
        f = sp.lambdify(xs, v, "numpy")

        def dens(p):
            out = np.asarray(f(*p.T), dtype=float)
            if out.shape == ():
                out = np.full(len(p), float(out))
            if np.any(out < 0) or not np.all(np.isfinite(out)):
                raise ValueError("RandomPointConfiguration: the density must be finite and non-negative")
            return out
        probe = reg.uniform(rng, 4096)
        bound = 1.25 * float(dens(probe).max()) + 1e-300
        while True:
            n = int(rng.poisson(bound * _f(reg.measure())))
            pts = reg.uniform(rng, n)
            dv = dens(pts) if n else np.zeros(0)
            if n and dv.max() > bound:
                bound = 2 * float(dv.max())
                continue
            return pts[rng.uniform(0, bound, size=n) < dv]


class ClusterPP(PP):
    def parse(self, a, fname):
        h = self.name
        if len(a) != 4:
            raise ValueError(f"{fname}: {h} expects [mu, lambda, {'sigma' if h.startswith('Thomas') else 'r'}, d]")
        self.mu = _p(a[0], fname, "the cluster centre density")
        self.lam = _p(a[1], fname, "the cluster mean")
        self.s = _p(a[2], fname, "the scale" if h.startswith("Thomas") else "the cluster radius")
        for v, w in ((self.mu, "the cluster centre density"), (self.lam, "the cluster mean"), (self.s, "the scale")):
            _pos(v, fname, w)
        self.d = _dim(a[3], fname)
        self.p = [self.mu, self.lam, self.s]

    def intensity(self):
        return self.mu * self.lam

    def count(self, reg):
        raise NotImplementedError(f"PointCountDistribution: the count distribution of {self.name} has no closed "
                                  "form")

    def _displace(self, rng, m):
        raise NotImplementedError

    def sample(self, rng, reg):
        import numpy as np
        lo, hi = reg.bbox()
        pad = self._pad()
        lo2, hi2 = [x - pad for x in lo], [x + pad for x in hi]
        vol = math.prod(b - a for a, b in zip(lo2, hi2))
        k = int(rng.poisson(_f(self.mu) * vol))
        if k > 10 ** 6:
            raise NotImplementedError("RandomPointConfiguration: more than 10^6 cluster centres are not supported")
        cen = rng.uniform(lo2, hi2, size=(k, self.d))
        counts = rng.poisson(_f(self.lam), size=k)
        pts = np.repeat(cen, counts, axis=0) + self._displace(rng, int(counts.sum()))
        return pts[reg.contains(pts)] if len(pts) else pts.reshape(0, self.d)


class ThomasPP(ClusterPP):
    def _pad(self):
        return 8 * _f(self.s)          # Gaussian displacements beyond 8 sigma have probability < 1e-14

    def _displace(self, rng, m):
        return rng.normal(0.0, _f(self.s), size=(m, self.d))

    def K(self, r):
        d = self.d
        return _unit_ball(d) * r ** d + sp.lowergamma(sp.Rational(d, 2), r ** 2 / (4 * self.s ** 2)) / \
            (sp.gamma(sp.Rational(d, 2)) * self.mu)


class MaternPP(ClusterPP):
    def _pad(self):
        return _f(self.s)

    def _displace(self, rng, m):
        import numpy as np
        g = rng.standard_normal((m, self.d))
        g /= np.linalg.norm(g, axis=1, keepdims=True) if m else 1
        rad = _f(self.s) * rng.uniform(0, 1, size=(m, 1)) ** (1.0 / self.d)
        return g * rad


PCLASSES = {"PoissonPointProcess": PoissonPP, "BinomialPointProcess": BinomialPP,
            "InhomogeneousPoissonPointProcess": InhomPP, "ThomasPointProcess": ThomasPP,
            "MaternPointProcess": MaternPP}


def is_pproc(x):
    return isinstance(x, Expr) and isinstance(x.head, str) and x.head in PCLASSES


def pproc_of(x, fname):
    if not is_pproc(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {x} is not a spatial point process")
    return PCLASSES[x.head](x, fname)


_PNOTES = {
    "PoissonPointProcess": "PoissonPointProcess[mu, d]: homogeneous Poisson point process (complete spatial "
                           "randomness) with intensity mu > 0 in R^d; counts in a region of volume v follow "
                           "PoissonDistribution[mu v]",
    "BinomialPointProcess": "BinomialPointProcess[n, reg]: n independent uniform points in reg (Rectangle, Cuboid, "
                            "Disk or Ball); counts in a subregion follow BinomialDistribution[n, |sub|/|reg|]",
    "InhomogeneousPoissonPointProcess": "InhomogeneousPoissonPointProcess[f, d]: Poisson point process with "
                                        "density function f (a function of d coordinates); counts follow "
                                        "PoissonDistribution[Integral of f]",
    "ThomasPointProcess": "ThomasPointProcess[mu, lambda, sigma, d]: Neyman-Scott cluster process with Poisson(mu) "
                          "centres, PoissonDistribution[lambda] points per cluster and NormalDistribution[0, sigma] "
                          "coordinate displacements",
    "MaternPointProcess": "MaternPointProcess[mu, lambda, r, d]: Neyman-Scott cluster process with Poisson(mu) "
                          "centres, PoissonDistribution[lambda] points per cluster uniform in a ball of radius r",
}


def _register(name):
    @fn(name, area=AREA, status="added",
        notes=_PNOTES[name] + ". Inert; the constructor validates numeric parameters (positive rates/scales, positive "
              "integer dimension; invalid values raise ValueError, non-numeric kinds TypeError). "
              "Check: exact; reference (Wolfram documentation) in tests.")
    def _ctor(*args: Expr) -> Expr:
        PCLASSES[name](Expr(name, *args), name)
        raise Unevaluated()
    return _ctor


for _n in PCLASSES:
    _register(_n)


# ─────────────────────────── SpatialPointData ───────────────────────────

def make_spd(configs, reg, ensemble):
    def pts(c):
        return L(*(L(*(R(sp.Float(float(x))) for x in p)) for p in c))
    if ensemble:
        return Expr("SpatialPointData", Expr("Association", *(rule(i + 1, pts(c)) for i, c in enumerate(configs))),
                    reg)
    return Expr("SpatialPointData", pts(configs[0]), reg)


def spd_parts(x, fname):
    """(list of configurations as lists of point tuples (MikoshiLang values), region expr or None, ensemble)"""
    if not (isinstance(x, Expr) and x.head == "SpatialPointData" and 1 <= len(x.args) <= 2):
        raise TypeError(f"{fname}: expected SpatialPointData")
    data = x.args[0]
    reg = x.args[1] if len(x.args) == 2 else None
    if isinstance(data, Expr) and data.head == "Association":
        confs = [list(r.args[1].args) for r in data.args]
        return confs, reg, True
    if not is_list(data):
        raise TypeError(f"{fname}: SpatialPointData needs a list of points")
    return [list(data.args)], reg, False


def _spd_apply(head, args, ev):
    from .. import kernel as K
    if len(args) != 1 or not isinstance(args[0], str):
        return K.UNHANDLED
    try:
        confs, reg, ens = spd_parts(head, "SpatialPointData")
    except TypeError:
        return K.UNHANDLED
    prop = args[0]
    if prop == "Points":
        return L(*(L(*c) for c in confs)) if ens else L(*confs[0])
    if prop == "ConfigurationCount":
        return len(confs)
    if prop == "PointCountList":
        return L(*(len(c) for c in confs))
    if prop == "PointCount":
        return L(*(len(c) for c in confs)) if ens else len(confs[0])
    if prop == "ObservationRegion":
        return reg if reg is not None else Expr("Missing", "NotAvailable")
    if prop == "Dimension":
        if reg is not None:
            return region_of(reg, "SpatialPointData").dim
        for c in confs:
            if c:
                return len(c[0].args)
        return Expr("Missing", "NotAvailable")
    if prop == "Properties":
        return L("ConfigurationCount", "Dimension", "ObservationRegion", "PointCount", "PointCountList", "Points")
    raise ValueError(f"SpatialPointData: unknown property {prop!r}")


KL.COMPOUND_HEAD_HOOKS["SpatialPointData"] = _spd_apply


# ─────────────────────────── functions ───────────────────────────

def _np_rng():
    import numpy as np
    return np.random.default_rng(W._rng().getrandbits(63))


@fn("RandomPointConfiguration", area=AREA, status="added",
    notes="RandomPointConfiguration[pproc, reg] and RandomPointConfiguration[pproc, reg, n] simulate point "
          "configurations of PoissonPointProcess (Poisson count, uniform points), BinomialPointProcess (n uniform "
          "points in its region, kept inside reg), InhomogeneousPoissonPointProcess (thinning with an adaptive "
          "bound from 4096 probe points), ThomasPointProcess and MaternPointProcess (centres in reg dilated by "
          "8 sigma / r) in Rectangle, Cuboid, Disk or Ball regions, reproducibly with SeedRandom (Wolfram's random "
          "stream differs). The result is an inert SpatialPointData[{pt, ...}, reg] (an Association of "
          "configurations for n) with the properties \"Points\", \"PointCount\", \"PointCountList\", "
          "\"ConfigurationCount\", \"ObservationRegion\", \"Dimension\", \"Properties\". Region and process "
          "dimensions must agree (ValueError); symbolic parameters stay unevaluated; Method and WorkingPrecision "
          "raise NotImplementedError. Check: statistical (mean point counts against PointCountDistribution) in "
          "tests.")
def _rpc(proc: Expr, reg: Expr, n: Integer = None, *opts: Expr) -> Expr:
    fname = "RandomPointConfiguration"
    if isinstance(n, Expr) and n.head in ("Rule", "RuleDelayed"):
        opts, n = (n,) + opts, None
    method = None
    for o in opts:
        if isinstance(o, Expr) and o.head in ("Rule", "RuleDelayed") and isinstance(o.args[0], Symbol) and \
                o.args[0].name in ("Method", "WorkingPrecision"):
            if o.args[1] == Symbol("Automatic") or (o.args[0].name == "WorkingPrecision" and
                                                    o.args[1] == Symbol("MachinePrecision")):
                continue
            if o.args[0].name == "Method" and is_pproc(proc) and getattr(PCLASSES[proc.head], "takes_method", False):
                method = o.args[1]       # Gibbs samplers (wl_spatial2, X08)
                continue
            raise NotImplementedError(f"{fname}: {o.args[0].name} is not supported")
        raise ValueError(f"{fname}: unexpected argument {o}")
    pr = pproc_of(proc, fname)
    rg = region_of(reg, fname)
    if rg.dim != pr.d:
        raise ValueError(f"{fname}: the region dimension {rg.dim} differs from the process dimension {pr.d}")
    if n is not None:
        if isinstance(n, Symbol):
            raise Unevaluated()
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"{fname}: the number of configurations must be a positive integer")
        if n < 1:
            raise ValueError(f"{fname}: the number of configurations must be positive")
    if not (pr.numeric() and rg.numeric):
        raise Unevaluated()
    rng = _np_rng()
    confs = [pr.sample(rng, rg) if method is None else pr.sample(rng, rg, method) for _ in range(n or 1)]
    return make_spd(confs, reg, n is not None)


@fn("PointCountDistribution", area=AREA, status="added",
    notes="PointCountDistribution[pproc, reg]: the distribution of the number of points in reg. "
          "PoissonPointProcess[mu, d]: PoissonDistribution[mu RegionMeasure[reg]]; BinomialPointProcess[n, reg0] "
          "(reg provably inside reg0; PointCountDistribution[proc] uses reg0): BinomialDistribution[n, "
          "|reg|/|reg0|]; InhomogeneousPoissonPointProcess over boxes: PoissonDistribution[Integral of f] "
          "(exact). Cluster and Gibbs processes and joint counts {reg1, ...} raise NotImplementedError. "
          "Check: exact; reference (Wolfram documentation: PoissonDistribution[mu Pi] for Disk[]) in tests.")
def _pcd(proc: Expr, reg: Expr = None) -> Expr:
    fname = "PointCountDistribution"
    pr = pproc_of(proc, fname)
    if reg is None:
        if isinstance(pr, BinomialPP):
            return Expr("BinomialDistribution", R(pr.n), 1)
        raise ValueError(f"{fname}: a region is required for {pr.name}")
    if is_list(reg):
        raise NotImplementedError(f"{fname}: joint count distributions are not supported")
    rg = region_of(reg, fname)
    if rg.dim != pr.d:
        raise ValueError(f"{fname}: the region dimension {rg.dim} differs from the process dimension {pr.d}")
    return pr.count(rg)


def _radius(r, fname):
    if is_list(r):
        return [_radius(x, fname) for x in r.args]
    v = _p(r, fname, "the radius")
    if _num(v) and not (v.is_real and v >= 0):
        raise ValueError(f"{fname}: the radius must be non-negative")
    return v


DATA_STATS = {}      # name -> fn(pdata, r, opts): point-data estimates, registered by wl_spatial2_stats (X08)


def _stat(fname, which):
    def impl(proc, r=None, opts=()):
        if (isinstance(proc, Expr) and proc.head == "SpatialPointData" or is_list(proc)) and fname in DATA_STATS and \
                not (is_list(proc) and proc.args and is_pproc(proc.args[0])):
            return DATA_STATS[fname](proc, r, opts)       # estimates from point data (wl_spatial2_stats, X08)
        if opts:
            raise ValueError(f"{fname}: options are only used with point data")
        if r is None:
            raise NotImplementedError(f"{fname}: PointStatisticFunction objects are not supported; give a radius")
        if isinstance(proc, Expr) and proc.head == "SpatialPointData" or is_list(proc):
            if is_list(proc) and proc.args and is_pproc(proc.args[0]):
                raise NotImplementedError(f"{fname}: {{proc, reg}} specifications are not supported")
            raise NotImplementedError(f"{fname}: estimation from point data (boundary-corrected) is not supported")
        pr = pproc_of(proc, fname)
        rs = _radius(r, fname)

        def one(rv):
            # parameters and radii are positive reals: compute with positive symbols, then restore the names
            free = set(rv.free_symbols)
            for v_ in getattr(pr, "p", []):
                if isinstance(v_, sp.Basic):
                    free |= v_.free_symbols
            pos = {s_: sp.Symbol(s_.name, positive=True) for s_ in free}
            back = {b: a for a, b in pos.items()}
            if pos:
                saved = pr.p, getattr(pr, "mu", None), getattr(pr, "lam", None), getattr(pr, "s", None)
                for attr in ("mu", "lam", "s"):
                    if hasattr(pr, attr):
                        setattr(pr, attr, getattr(pr, attr).subs(pos))
                try:
                    out = _one(rv.subs(pos))
                finally:
                    pr.p = saved[0]
                    for attr, val in zip(("mu", "lam", "s"), saved[1:]):
                        if val is not None:
                            setattr(pr, attr, val)
                return R(sp.sympify(S(out)).subs(back)) if not isinstance(out, bool) else out
            return _one(rv)

        def _one(rv):
            d = pr.d
            if which in ("K", "L", "g"):
                if not hasattr(pr, "K"):
                    raise NotImplementedError(f"{fname}: no closed form is available for {pr.name}")
                if not pr.stationary:
                    raise NotImplementedError(f"{fname}: {pr.name} is not stationary")
                if which == "K":
                    v = pr.K(rv)
                elif which == "L":
                    v = (pr.K(rv) / _unit_ball(d)) ** sp.Rational(1, d)
                elif hasattr(pr, "g"):
                    v = pr.g(rv)
                else:
                    x = sp.Dummy("r", positive=True)
                    v = (sp.diff(pr.K(x), x) / (d * _unit_ball(d) * x ** (d - 1))).subs(x, rv)
            else:
                if not isinstance(pr, PoissonPP):
                    raise NotImplementedError(f"{fname}: no closed form is available for {pr.name}")
                v = pr.G(rv) if which in ("G", "F") else sp.Integer(1)
            v = sp.simplify(v)
            return R(v if not has_float(R(v)) else sp.N(v))
        if isinstance(rs, list):
            return L(*(one(x) for x in rs))
        return one(rs)
    return impl


_STATNOTES = {
    "RipleyK": ("K", "RipleyK[pproc, r]: Ripley's K function. PoissonPointProcess[mu, d]: Pi^(d/2)/Gamma[d/2 + 1] "
                     "r^d; ThomasPointProcess[mu, lambda, sigma, d]: that plus "
                     "GammaRegularized-type term Gamma[d/2, 0, r^2/(4 sigma^2)]/(Gamma[d/2] mu) (2-D: (1 - "
                     "Exp[-r^2/(4 sigma^2)])/mu)"),
    "BesagL": ("L", "BesagL[pproc, r]: Besag's L function (K/omega_d)^(1/d) (= r for Poisson)"),
    "PairCorrelationG": ("g", "PairCorrelationG[pproc, r]: the pair correlation K'(r)/(d omega_d r^(d-1)) (1 for "
                              "Poisson; 1 + Exp[-r^2/(4 sigma^2)]/(mu (4 Pi sigma^2)^(d/2)) for Thomas)"),
    "NearestNeighborG": ("G", "NearestNeighborG[pproc, r]: the nearest-neighbour distance CDF; Poisson: 1 - "
                              "Exp[-mu omega_d r^d]"),
    "EmptySpaceF": ("F", "EmptySpaceF[pproc, r]: the empty-space (spherical contact) CDF; Poisson: 1 - "
                         "Exp[-mu omega_d r^d]"),
    "SpatialJ": ("J", "SpatialJ[pproc, r]: (1 - G)/(1 - F); 1 for Poisson"),
}


def _register_stat(name):
    which, note = _STATNOTES[name]
    impl = _stat(name, which)

    @fn(name, area=AREA, status="added",
        notes=note + ". The radius may be a list. Other processes (Matern, binomial, inhomogeneous, Gibbs ...), "
                     "point-data estimates, {proc, reg} forms and PointStatisticFunction (no radius) raise "
                     "NotImplementedError; negative radii raise ValueError. Check: exact; reference (closed forms; "
                     "Wolfram documentation) in tests.")
    def _s(proc: Expr, r: Real | ListOf[Real] | Var | Option = None, *opts: Option) -> Expr:
        if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed"):
            r, opts = None, (r,) + opts
        return impl(proc, r, opts)
    return _s


for _n in _STATNOTES:
    _register_stat(_n)


@fn("MeanPointDensity", area=AREA, status="added",
    notes="MeanPointDensity[pproc]: the intensity (PoissonPointProcess: mu; BinomialPointProcess: n/|reg|; "
          "Thomas and Matern: mu lambda); MeanPointDensity[SpatialPointData[pts, reg]] and MeanPointDensity[{pts, "
          "reg}]: the estimate (number of points)/RegionMeasure[reg] (a list for ensembles). Point data without a "
          "region (Ripley-Rasson region) and inhomogeneous processes raise NotImplementedError. "
          "Check: exact; reference (n/area by hand) in tests.")
def _mpd(x: Expr) -> Expr:
    fname = "MeanPointDensity"
    if is_pproc(x):
        pr = pproc_of(x, fname)
        return R(sp.simplify(pr.intensity()))
    if isinstance(x, Expr) and x.head == "SpatialPointData":
        confs, reg, ens = spd_parts(x, fname)
    elif is_list(x) and len(x.args) == 2 and is_list(x.args[0]) and not is_list(x.args[1]):
        confs, reg, ens = [list(x.args[0].args)], x.args[1], False
    elif isinstance(x, Symbol):
        raise Unevaluated()
    elif is_list(x):
        raise NotImplementedError(f"{fname}: point data without an observation region are not supported")
    else:
        raise TypeError(f"{fname}: expected a point process, SpatialPointData or {{points, region}}")
    if reg is None:
        raise NotImplementedError(f"{fname}: point data without an observation region are not supported")
    rg = region_of(reg, fname)
    vals = []
    for c in confs:
        for p in c:
            if not (is_list(p) and len(p.args) == rg.dim):
                raise ValueError(f"{fname}: points must have the dimension of the region ({rg.dim})")
        vals.append(sp.Integer(len(c)) / rg.measure())
    out = [R(sp.simplify(v)) for v in vals]
    return L(*out) if ens else out[0]


from .. import kernel as _KK  # noqa: E402

for _h in ("PointProcessParameterQ", "PointProcessParameterAssumptions"):
    _KK.BUILTIN_ATTRIBUTES[_h] = {"HoldAll", "Protected"}


def _held(proc):
    if isinstance(proc, Expr) and isinstance(proc.head, str) and proc.head in PCLASSES:
        return Expr(proc.head, *(_ev(a) for a in proc.args))
    return _ev(proc)


def _lax(proc, fname):
    _CONDS.clear()
    _LAX[0] = True
    try:
        try:
            pr = PCLASSES[proc.head](proc, fname)
        except (ValueError, TypeError):
            return None, None
        return pr, list(_CONDS)
    finally:
        _LAX[0] = False
        _CONDS.clear()


@fn("PointProcessParameterAssumptions", area=AREA, status="added", hold=True,
    notes="PointProcessParameterAssumptions[pproc]: the conjunction of the constructor's parameter conditions "
          "(positive intensities, cluster means and scales; positive integer point counts), evaluated: True or "
          "False for numeric parameters, a logical expression otherwise; invalid structure (dimension, region) "
          "gives False. Check: exact; reference (Wolfram documentation) in tests.")
def _pppa(proc: Held) -> Expr:
    fname = "PointProcessParameterAssumptions"
    proc = _held(proc)
    if not is_pproc(proc):
        if isinstance(proc, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {proc} is not a spatial point process")
    pr, conds = _lax(proc, fname)
    if pr is None:
        return False
    return _ev(Expr("And", *conds)) if conds else True


@fn("PointProcessParameterQ", area=AREA, status="added", hold=True,
    notes="PointProcessParameterQ[pproc]: True if pproc is a spatial point process whose numeric parameters are "
          "valid (symbolic parameters are assumed valid), else False. Check: exact; reference (Wolfram "
          "documentation) in tests.")
def _pppq(proc: Held) -> Boolean:
    proc = _held(proc)
    if not is_pproc(proc):
        return False
    try:
        pr, conds = _lax(proc, "PointProcessParameterQ")
    except Unevaluated:
        return True
    if pr is None:
        return False
    return not any(_ev(c) is False for c in conds)
