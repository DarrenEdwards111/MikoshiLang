"""Wolfram batch X08: point density estimation -- PointDensity (Delaunay / Voronoi partitions), SmoothPointDensity
(product-kernel estimate with Diggle's edge correction), HistogramPointDensity, the PointDensityFunction object and
PointValuePlot.

Methods (Baddeley, Rubak & Turner, Spatial Point Patterns, CRC 2015, sec. 6.4-6.5; Schaap & van de Weygaert 2000):

* "Delaunay" (default; Delaunay tessellation field estimator, DTFE): the density at data point x_i is
  (d + 1) / V_i, V_i the total volume of the Delaunay simplices with vertex x_i; it is interpolated linearly in each
  simplex and is 0 outside the convex hull, so it integrates to n.
* "Voronoi": 1/|C_i| on the Voronoi cell C_i of x_i clipped to the observation region (1D and 2D); 0 outside it.
* SmoothPointDensity: f(x) = Sum_i Prod_k K((x_k - x_ik)/h_k)/h_k / e_i with Diggle's (1985) edge correction e_i (the
  kernel mass at x_i inside W; exact for boxes, a 64^2 / 24^3 midpoint rule over the kernel support otherwise), so
  it integrates to n over W.  Kernels on [-1, 1] (Gaussian: standard normal, h is its standard deviation).
  Bandwidths per coordinate: "Silverman" (default) 0.9 min(s, IQR/1.34) n^(-1/5), "Scott" 1.06 s n^(-1/5),
  "StandardDeviation" s (s the sample standard deviation; constant coordinates use s = 1).
* HistogramPointDensity: count / |bin n W| for coordinate-wise bins over the bounding box of W.

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
from . import wl_spatial2 as SP
from .wl_spatial2 import _np, _ev, _need_scipy, fnum, is_auto, window, spd_of, config_array, split_opts

AREA = "spatial statistics II (X08)"


# ─────────────────────────── PointDensityFunction ───────────────────────────

def _pdf(method, **parts):
    return Expr("PointDensityFunction", Expr("Association", rule("Method", method),
                                             *(rule(k, v) for k, v in parts.items())))


def _pdf_parts(x, fname="PointDensityFunction"):
    if not (isinstance(x, Expr) and x.head == "PointDensityFunction" and len(x.args) == 1 and
            isinstance(x.args[0], Expr) and x.args[0].head == "Association"):
        raise TypeError(f"{fname}: expected PointDensityFunction[<|\"Method\" -> ..., ...|>] as produced by "
                        "PointDensity, SmoothPointDensity or HistogramPointDensity")
    d = {r_.args[0]: r_.args[1] for r_ in x.args[0].args}
    need = {"Exact": ("Density", "Dimension"), "Delaunay": ("Points", "VertexDensity", "Region"),
            "Voronoi": ("Points", "CellDensity", "Region"),
            "Kernel": ("Points", "Bandwidth", "Kernel", "EdgeWeights", "Region"),
            "Histogram": ("BinEdges", "Densities", "Region")}
    m = d.get("Method")
    if m not in need:
        raise ValueError(f"{fname}: unknown method {m!r}")
    for k in need[m]:
        if k not in d:
            raise ValueError(f"{fname}: the key \"{k}\" is missing")
    return d


def _tolist(e):
    if is_list(e):
        return [_tolist(a) for a in e.args]
    return float(sp.N(S(e)))


def _locs(a, dim, fname):
    """query locations: ({points}, list?) or Unevaluated for symbolic input"""
    np = _np()
    if is_list(a) and a.args and all(is_list(p) for p in a.args):
        pts, lst = list(a.args), True
    elif is_list(a):
        pts, lst = [a], False
    elif dim == 1 and not isinstance(a, str):
        pts, lst = [L(a)], False
    else:
        raise TypeError(f"{fname}: the location must be a coordinate list (or a number in 1D)")
    out = []
    for p in pts:
        if len(p.args) != dim:
            raise ValueError(f"{fname}: the location must have {dim} coordinates")
        out.append([fnum(c, fname, "a coordinate") for c in p.args])
    return np.array(out, dtype=float).reshape(-1, dim), lst


_DTRI = {}


def _delaunay(P):
    from scipy.spatial import Delaunay
    key = (P.shape, P.tobytes())
    t = _DTRI.get(key)
    if t is None:
        if len(_DTRI) > 32:
            _DTRI.clear()
        t = _DTRI[key] = Delaunay(P)
    return t


def _eval_pdf(d, X, fname):
    np = _np()
    m = d["Method"]
    if m == "Histogram":
        w = window(d["Region"], fname)
        edges = [np.array(_tolist(e)) for e in d["BinEdges"].args]
        D = np.array(_tolist(d["Densities"]), dtype=float).reshape([len(e) - 1 for e in edges])
        out = np.zeros(len(X))
        ins = w.contains(X)
        for i, x in enumerate(X):
            if not ins[i]:
                continue
            idx = []
            for k, e in enumerate(edges):
                j = int(np.searchsorted(e, x[k], side="right")) - 1
                if j == len(e) - 1 and x[k] == e[-1]:
                    j -= 1
                if j < 0 or j >= len(e) - 1:
                    idx = None
                    break
                idx.append(j)
            if idx is not None:
                out[i] = D[tuple(idx)]
        return out
    w = window(d["Region"], fname)
    P = np.array(_tolist(d["Points"]), dtype=float).reshape(-1, w.dim)
    ins = w.contains(X)
    if m == "Kernel":
        h = np.array(_tolist(d["Bandwidth"]), dtype=float)
        wt = np.array(_tolist(d["EdgeWeights"]), dtype=float)
        K = KERNELS[d["Kernel"]][0]
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            if ins[i]:
                U = (x - P) / h
                out[i] = float((wt * np.prod(K(U), axis=1)).sum() / np.prod(h))
        return out
    if m == "Voronoi":
        rho = np.array(_tolist(d["CellDensity"]), dtype=float)
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            if ins[i]:
                out[i] = rho[int(np.argmin(((P - x) ** 2).sum(axis=1)))]
        return out
    rho = np.array(_tolist(d["VertexDensity"]), dtype=float)
    if w.dim == 1:
        o = np.argsort(P[:, 0])
        xs, rs = P[o, 0], rho[o]
        out = np.where((X[:, 0] >= xs[0]) & (X[:, 0] <= xs[-1]) & ins, np.interp(X[:, 0], xs, rs), 0.0)
        return out
    _need_scipy(fname, "the Delaunay density estimate")
    tri = _delaunay(P)
    s = tri.find_simplex(X)
    out = np.zeros(len(X))
    for i, x in enumerate(X):
        if s[i] < 0 or not ins[i]:
            continue
        T = tri.transform[s[i]]
        b = T[:w.dim] @ (x - T[w.dim])
        lam = np.append(b, 1 - b.sum())
        out[i] = float(lam @ rho[tri.simplices[s[i]]])
    return out


def _pdf_apply(head, args, ev):
    from .. import kernel as K
    fname = "PointDensityFunction"
    if len(args) != 1:
        return K.UNHANDLED
    d = _pdf_parts(head)
    a = args[0]
    if isinstance(a, str):
        if a == "Properties":
            return L(*sorted(k for k in d))
        if a in d:
            return d[a]
        raise ValueError(f"{fname}: unknown property {a!r}")
    if isinstance(a, Symbol):
        return K.UNHANDLED
    if d["Method"] == "Exact":
        dim = d["Dimension"]
        pts = list(a.args) if is_list(a) and a.args and all(is_list(p) for p in a.args) else [a]
        lst = len(pts) > 1 or (is_list(a) and a.args and all(is_list(p) for p in a.args))
        vals = []
        for p in pts:
            cs = list(p.args) if is_list(p) else [p]
            if len(cs) != dim:
                raise ValueError(f"{fname}: the location must have {dim} coordinates")
            dens = d["Density"]
            if "Region" in d:
                if any(not S(c).is_number for c in cs):
                    return K.UNHANDLED
                inside = bool(window(d["Region"], fname).contains(
                    _np().array([[fnum(c, fname, "a coordinate") for c in cs]]))[0])
                vals.append(dens if inside else 0)
            elif isinstance(dens, Expr) and dens.head == "Function":
                vals.append(ev(Expr(dens, *cs)))
            else:
                vals.append(dens)
        return L(*vals) if lst else vals[0]
    w = window(d["Region"], fname)
    try:
        X, lst = _locs(a, w.dim, fname)
    except Unevaluated:
        return K.UNHANDLED
    v = _eval_pdf(d, X, fname)
    out = [R(float(x)) for x in v]
    return L(*out) if lst else out[0]


KL.COMPOUND_HEAD_HOOKS["PointDensityFunction"] = _pdf_apply


@fn("PointDensityFunction", area=AREA, status="added",
    notes="PointDensityFunction[<|\"Method\" -> m, ...|>] is produced by PointDensity, SmoothPointDensity and "
          "HistogramPointDensity; pdf[{x, y, ...}] (or pdf[x] in 1D) gives the density at a location and "
          "pdf[{p1, p2, ...}] a list (machine numbers; 0 outside the observation region, and outside the convex hull "
          "for the Delaunay estimate); pdf[\"Properties\"] and pdf[key] give its parts. Inert; malformed objects "
          "raise TypeError/ValueError, symbolic locations stay unevaluated. Check: exact (piecewise-constant and "
          "interpolated values by hand) and integrals equal to the number of points in tests.")
def _pdf_ctor(spec: Assoc) -> Expr:
    _pdf_parts(Expr("PointDensityFunction", spec))
    raise Unevaluated()


# ─────────────────────────── shared input handling ───────────────────────────

def _process_density(x, fname, extra):
    pr = PS.pproc_of(x, fname)
    if isinstance(pr, PS.InhomPP):
        if fname == "HistogramPointDensity":
            raise NotImplementedError(f"{fname}: histogram densities of inhomogeneous processes are not supported "
                                      "(PointDensity gives the exact density function)")
        return _pdf("Exact", Density=pr.fn, Dimension=pr.d)
    if not pr.numeric():
        raise Unevaluated()
    mu = R(sp.simplify(pr.intensity()))
    if isinstance(pr, PS.BinomialPP):
        return _pdf("Exact", Density=mu, Dimension=pr.d, Region=x.args[1])
    return _pdf("Exact", Density=mu, Dimension=pr.d)


def _data(x, fname):
    if is_list(x) and len(x.args) == 2 and PS.is_pproc(x.args[0]):
        raise NotImplementedError(f"{fname}: simulation-based densities of {{proc, reg}} are not supported; use the "
                                  "exact density of proc or estimate from RandomPointConfiguration[proc, reg]")
    if isinstance(x, Expr) and x.head == "SpatialBinnedPointData":
        raise NotImplementedError(f"{fname}: binned point data are not supported")
    sd, w = spd_of(x, fname)
    if sd.ens:
        raise NotImplementedError(f"{fname}: ensembles of configurations are not supported; use each configuration")
    P = config_array(sd.configs[0], w, fname)
    if len(P) == 0:
        raise ValueError(f"{fname}: the point data contain no points")
    return P, w, sd.reg


def _check_integral(total, n, fname, what):
    if not abs(total - n) <= 1e-7 * max(1.0, n):
        raise ArithmeticError(f"{fname}: internal check failed ({what} integrates to {total}, not {n})")


# ─────────────────────────── PointDensity ───────────────────────────

def _dups(P, fname):
    np = _np()
    if len(np.unique(P, axis=0)) != len(P):
        raise ValueError(f"{fname}: the points must be distinct for a partition-based estimate")


def _voronoi(P, w, reg, fname):
    np = _np()
    n = len(P)
    if w.dim == 1:
        o = np.argsort(P[:, 0])
        xs = P[o, 0]
        mids = (xs[1:] + xs[:-1]) / 2
        lo = np.concatenate([[w.lo[0]], mids])
        hi = np.concatenate([mids, [w.hi[0]]])
        rho = np.empty(n)
        rho[o] = 1.0 / (hi - lo)
        _check_integral(float((hi - lo).sum()), w.measure, fname, "the cell measures (region)")
        return rho
    if w.dim != 2:
        raise NotImplementedError(f"{fname}: Voronoi cells are supported in 1D and 2D")
    if w.kind == "box":
        V = np.array([[w.lo[0], w.lo[1]], [w.hi[0], w.lo[1]], [w.hi[0], w.hi[1]], [w.lo[0], w.hi[1]]])
        nrm0 = np.array([[0.0, -1], [1.0, 0], [0, 1.0], [-1.0, 0]])
        off0 = np.array([-w.lo[1], w.hi[0], w.hi[1], -w.lo[0]])
    elif w.kind == "poly":
        V, nrm0, off0 = w.V, w.nrm, w.off
    else:
        raise NotImplementedError(f"{fname}: Voronoi cells need a rectangle or convex polygon region")
    if n >= 4:
        _need_scipy(fname, "the Voronoi partition")
        try:
            tri = _delaunay(P)
        except Exception:
            tri = None
    else:
        tri = None
    if tri is not None:
        ip, ix = tri.vertex_neighbor_vertices
        nb = [ix[ip[i]:ip[i + 1]] for i in range(n)]
    else:
        nb = [np.array([j for j in range(n) if j != i]) for i in range(n)]
    areas = np.empty(n)
    for i in range(n):
        J = nb[i]
        nrm = P[J] - P[i]
        off = ((P[J] ** 2).sum(axis=1) - (P[i] ** 2).sum()) / 2
        Q = SP.clip_poly(V, np.vstack([nrm0, nrm]), np.concatenate([off0, off]))
        areas[i] = SP.poly_area(Q)
    if np.any(areas <= 0):
        raise ValueError(f"{fname}: degenerate Voronoi cells (repeated points)")
    _check_integral(float(areas.sum()), w.measure, fname, "the cell areas (region)")
    return 1.0 / areas


def _dtfe(P, w, fname):
    np = _np()
    n, d = P.shape
    if d == 1:
        if n < 2:
            raise ValueError(f"{fname}: the Delaunay estimate needs at least 2 points in 1D")
        o = np.argsort(P[:, 0])
        xs = P[o, 0]
        gaps = np.diff(xs)
        V = np.zeros(n)
        V[:-1] += gaps
        V[1:] += gaps
        rho = np.empty(n)
        rho[o] = 2.0 / V
        tot = float((gaps * (rho[o][1:] + rho[o][:-1]) / 2).sum())
        _check_integral(tot, n, fname, "the Delaunay density")
        return rho
    _need_scipy(fname, "the Delaunay density estimate")
    if n < d + 1:
        raise ValueError(f"{fname}: the Delaunay estimate needs at least {d + 1} points in {d}D")
    try:
        tri = _delaunay(P)
    except Exception:
        raise ValueError(f"{fname}: the points are degenerate (they do not span {d}D space)")
    S_ = P[tri.simplices]
    vol = np.abs(np.linalg.det(S_[:, 1:, :] - S_[:, :1, :])) / math.factorial(d)
    V = np.zeros(n)
    for k in range(d + 1):
        np.add.at(V, tri.simplices[:, k], vol)
    if np.any(V <= 0):
        raise ValueError(f"{fname}: degenerate Delaunay cells (repeated points)")
    rho = (d + 1) / V
    tot = float((vol * rho[tri.simplices].mean(axis=1)).sum())
    _check_integral(tot, n, fname, "the Delaunay density")
    return rho


@fn("PointDensity", area=AREA, status="added",
    notes="PointDensity[pdata] and PointDensity[pdata, \"Delaunay\"]: the Delaunay tessellation field estimator "
          "(Schaap & van de Weygaert 2000; density (d+1)/V_i at the points, V_i the volume of the Delaunay simplices "
          "at x_i, linear in each simplex, 0 outside the convex hull; any dimension, SciPy for d >= 2); "
          "PointDensity[pdata, \"Voronoi\"]: 1/|cell| on the Voronoi cells clipped to the region (1D; 2D in "
          "rectangles and convex polygons). pdata is {p1, ...} (2D: Ripley-Rasson region), {pts, reg} or "
          "SpatialPointData; the result is a PointDensityFunction. PointDensity[pproc] gives the exact density "
          "(constant intensity, InhomogeneousPoissonPointProcess's function; Gibbs processes raise "
          "NotImplementedError). {\"Voronoi\", n} aggregation, binned data, {proc, reg}, geographic points and "
          "ensembles raise NotImplementedError; repeated or degenerate points raise ValueError. Check: exact "
          "(Voronoi and Delaunay values by hand) and self-check (the estimate integrates to n, ArithmeticError "
          "otherwise) in tests.")
def _point_density(pdata: Expr, method: Expr = None) -> Expr:
    fname = "PointDensity"
    if PS.is_pproc(pdata):
        return _process_density(pdata, fname, method)
    m = "Delaunay" if is_auto(method) else method
    if is_list(m) and len(m.args) == 2 and m.args[0] == "Voronoi":
        raise NotImplementedError(f"{fname}: aggregated Voronoi cells {{\"Voronoi\", n}} are not supported")
    if m not in ("Delaunay", "Voronoi"):
        raise ValueError(f"{fname}: the method must be \"Delaunay\" or \"Voronoi\"")
    P, w, reg = _data(pdata, fname)
    _dups(P, fname)
    pts = R([list(map(float, p)) for p in P])
    if m == "Voronoi":
        rho = _voronoi(P, w, reg, fname)
        return _pdf("Voronoi", Points=pts, CellDensity=R([float(v) for v in rho]), Region=reg)
    rho = _dtfe(P, w, fname)
    return _pdf("Delaunay", Points=pts, VertexDensity=R([float(v) for v in rho]), Region=reg)


# ─────────────────────────── SmoothPointDensity ───────────────────────────

def _k_poly(c, pw, extra=None):
    def k(U):
        np = _np()
        return np.where(np.abs(U) <= 1, c * (1 - U ** 2) ** pw if extra is None else extra(U), 0.0)
    return k


def _gauss_cdf(U):
    np = _np()
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(np.asarray(U) / math.sqrt(2)))


def _cdf(f):
    def c(U):
        np = _np()
        U = np.clip(U, -1.0, 1.0)
        return f(U)
    return c


KERNELS = {
    "Gaussian": (lambda U: _np().exp(-U ** 2 / 2) / math.sqrt(2 * math.pi), _gauss_cdf),
    "Rectangular": (lambda U: _np().where(_np().abs(U) <= 1, 0.5, 0.0), _cdf(lambda U: (U + 1) / 2)),
    "Triangular": (lambda U: _np().where(_np().abs(U) <= 1, 1 - _np().abs(U), 0.0),
                   _cdf(lambda U: _np().where(U < 0, (1 + U) ** 2 / 2, 1 - (1 - U) ** 2 / 2))),
    "Epanechnikov": (_k_poly(0.75, 1), _cdf(lambda U: (2 + 3 * U - U ** 3) / 4)),
    "Biweight": (_k_poly(15 / 16, 2), _cdf(lambda U: 0.5 + 15 / 16 * (U - 2 * U ** 3 / 3 + U ** 5 / 5))),
    "Triweight": (_k_poly(35 / 32, 3),
                  _cdf(lambda U: 0.5 + 35 / 32 * (U - U ** 3 + 3 * U ** 5 / 5 - U ** 7 / 7))),
    "Cosine": (lambda U: _np().where(_np().abs(U) <= 1, math.pi / 4 * _np().cos(math.pi * U / 2), 0.0),
               _cdf(lambda U: (1 + _np().sin(math.pi * U / 2)) / 2)),
    "SemiCircle": (lambda U: _np().where(_np().abs(U) <= 1, 2 / math.pi * _np().sqrt(_np().clip(1 - U ** 2, 0, None)),
                                         0.0),
                   _cdf(lambda U: 0.5 + (U * _np().sqrt(1 - U ** 2) + _np().arcsin(U)) / math.pi)),
}


def _kernel_name(ker, fname):
    if is_auto(ker):
        return "Gaussian"
    if is_list(ker) and len(ker.args) == 2 and ker.args[0] in ("Product", "Radial"):
        if ker.args[0] == "Radial":
            raise NotImplementedError(f"{fname}: radial kernels are not supported (product kernels are)")
        ker = ker.args[1]
    if isinstance(ker, str):
        if ker not in KERNELS:
            raise ValueError(f"{fname}: unknown kernel \"{ker}\" (use {', '.join(sorted(KERNELS))})")
        return ker
    if isinstance(ker, Symbol):
        raise Unevaluated()
    raise NotImplementedError(f"{fname}: custom kernel functions are not supported")


def _sd(x):
    np = _np()
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return s if s > 0 else 1.0


def _iqr(x):
    np = _np()
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def _bw1(spec, x, fname):
    n = len(x)
    if is_auto(spec):
        spec = "Silverman"
    if isinstance(spec, str):
        s = _sd(x)
        if spec == "Silverman":
            q = _iqr(x) / 1.34
            return 0.9 * (min(s, q) if q > 0 else s) * n ** (-0.2)
        if spec == "Scott":
            return 1.06 * s * n ** (-0.2)
        if spec == "StandardDeviation":
            return s
        if spec in ("LeastSquaresCrossValidation", "SheatherJones", "Oversmooth", "StandardGaussian"):
            raise NotImplementedError(f"{fname}: the bandwidth method \"{spec}\" is not supported (use \"Silverman\", "
                                      "\"Scott\", \"StandardDeviation\" or a number)")
        raise ValueError(f"{fname}: unknown bandwidth method \"{spec}\"")
    if is_list(spec):
        raise NotImplementedError(f"{fname}: standardized, adaptive and matrix bandwidths are not supported")
    h = fnum(spec, fname, "the bandwidth")
    if not h > 0:
        raise ValueError(f"{fname}: the bandwidth must be positive")
    return h


def bandwidths(bw, P, fname):
    d = P.shape[1]
    if is_list(bw) and len(bw.args) == d and not (bw.args and bw.args[0] in ("Standardized", "Adaptive")) and \
            not all(is_list(r_) for r_ in bw.args):
        if d == 1 and len(bw.args) == 1 and is_list(bw.args[0]):
            raise NotImplementedError(f"{fname}: matrix bandwidths are not supported")
        return [_bw1(b, P[:, k], fname) for k, b in enumerate(bw.args)]
    if is_list(bw) and bw.args and all(is_list(r_) for r_ in bw.args):
        raise NotImplementedError(f"{fname}: matrix bandwidths are not supported")
    return [_bw1(bw, P[:, k], fname) for k in range(d)]


def edge_weights(P, w, h, kname, fname):
    """Diggle's correction e_i = kernel mass at x_i inside W"""
    np = _np()
    K, C = KERNELS[kname]
    h = np.asarray(h)
    if w.kind == "box":
        e = np.prod(C((w.hi - P) / h) - C((w.lo - P) / h), axis=1)
    else:
        d = w.dim
        m = 64 if d <= 2 else 24
        span = 4.0 if kname == "Gaussian" else 1.0
        t = (np.arange(m) + 0.5) / m * 2 - 1
        grid = np.stack(np.meshgrid(*([t] * d), indexing="ij"), axis=-1).reshape(-1, d)
        U = grid * span                                   # in kernel units
        kv = np.prod(K(U), axis=1) * (2 * span / m) ** d
        e = np.empty(len(P))
        for i, p in enumerate(P):
            e[i] = float(kv[w.contains(p + U * h)].sum())
        e = e / float(kv.sum())                           # Gaussian: truncated at 4 standard deviations
    if np.any(e <= 0) or np.any(e > 1 + 1e-6):
        raise ArithmeticError(f"{fname}: internal check failed (edge-correction masses outside (0, 1])")
    return np.minimum(e, 1.0)


@fn("SmoothPointDensity", area=AREA, status="added",
    notes="SmoothPointDensity[pdata, bw, ker]: product-kernel estimate Sum_i Prod_k K((x_k - x_ik)/h_k)/h_k / e_i "
          "with Diggle's edge correction e_i (kernel mass at x_i inside W: exact for boxes, 64^2/24^3-point midpoint "
          "rule over the kernel support otherwise), 0 outside W, so it integrates to n. Kernels \"Gaussian\" "
          "(default; h = standard deviation), \"Rectangular\", \"Triangular\", \"Epanechnikov\", \"Biweight\", "
          "\"Triweight\", \"Cosine\", \"SemiCircle\" (support [-h, h]) and {\"Product\", ker}. Bandwidth (per "
          "coordinate or {bwx, bwy, ...}): a number, \"Silverman\" (default, 0.9 min(s, IQR/1.34) n^(-1/5)), "
          "\"Scott\" (1.06 s n^(-1/5)) or \"StandardDeviation\" (s); Wolfram's multivariate rules may differ. "
          "pdata is {p1, ...} (2D: Ripley-Rasson region), {pts, reg} or SpatialPointData; result is a "
          "PointDensityFunction. SmoothPointDensity[pproc] gives the exact density. Radial and custom kernels, "
          "\"LeastSquaresCrossValidation\", \"SheatherJones\", \"Oversmooth\", \"StandardGaussian\", standardized, "
          "adaptive and matrix bandwidths, binned data, {proc, reg}, geographic points and ensembles raise "
          "NotImplementedError. Check: reference (single-point Gaussian and Epanechnikov values by hand) and "
          "numerical (integral over W equals n) in tests.")
def _smooth_point_density(pdata: Expr, bw: Expr = None, ker: Expr = None) -> Expr:
    fname = "SmoothPointDensity"
    if PS.is_pproc(pdata):
        return _process_density(pdata, fname, bw)
    kname = _kernel_name(ker, fname)
    P, w, reg = _data(pdata, fname)
    h = bandwidths(bw, P, fname)
    e = edge_weights(P, w, h, kname, fname)
    return _pdf("Kernel", Points=R([list(map(float, p)) for p in P]), Bandwidth=R([float(v) for v in h]),
                Kernel=kname, EdgeWeights=R([float(1.0 / v) for v in e]), Region=reg)


# ─────────────────────────── HistogramPointDensity ───────────────────────────

def _edges1(spec, x, lo, hi, fname):
    np = _np()
    n = len(x)

    def by_width(wd):
        if not wd > 0:
            raise ValueError(f"{fname}: the bin width must be positive")
        nb = max(1, int(math.ceil((hi - lo) / wd - 1e-9)))
        return np.linspace(lo, hi, nb + 1)

    if is_auto(spec):
        spec = "Sturges"
    if isinstance(spec, str):
        if spec == "Sturges":
            return np.linspace(lo, hi, int(math.ceil(math.log2(n))) + 2 if n > 1 else 2)
        if spec == "Scott":
            return by_width(3.49 * _sd(x) * n ** (-1 / 3))
        if spec == "FreedmanDiaconis":
            q = _iqr(x)
            return by_width(2 * (q if q > 0 else _sd(x)) * n ** (-1 / 3))
        if spec in ("Knuth", "Wand"):
            raise NotImplementedError(f"{fname}: the binning method \"{spec}\" is not supported (use \"Sturges\", "
                                      "\"Scott\" or \"FreedmanDiaconis\")")
        if spec in ("Triangle", "Square", "Hexagon", "ObservationMesh"):
            raise NotImplementedError(f"{fname}: shaped and mesh bins are not supported (give coordinate-wise bins)")
        raise ValueError(f"{fname}: unknown binning method \"{spec}\"")
    if isinstance(spec, int) and not isinstance(spec, bool):
        if spec < 1:
            raise ValueError(f"{fname}: the number of bins must be positive")
        return np.linspace(lo, hi, spec + 1)
    if is_list(spec) and len(spec.args) == 1 and is_list(spec.args[0]):
        e = np.array([fnum(v, fname, "a bin edge") for v in spec.args[0].args])
        if len(e) < 2 or np.any(np.diff(e) <= 0):
            raise ValueError(f"{fname}: bin edges must be strictly increasing (at least two)")
        return e
    if is_list(spec) and len(spec.args) == 1:
        wd = fnum(spec.args[0], fname, "the bin width")
        if not wd > 0:
            raise ValueError(f"{fname}: the bin width must be positive")
        a = math.floor(lo / wd) * wd
        b = math.ceil(hi / wd) * wd
        nb = max(1, int(round((b - a) / wd)))
        return a + wd * np.arange(nb + 1)
    if is_list(spec) and len(spec.args) == 3:
        a, b, wd = (fnum(v, fname, "a bin specification") for v in spec.args)
        if not (wd > 0 and b > a):
            raise ValueError(f"{fname}: {{min, max, w}} needs min < max and w > 0")
        e = a + wd * np.arange(int(math.floor((b - a) / wd + 1e-9)) + 1)
        if len(e) < 2:
            raise ValueError(f"{fname}: the bin width exceeds max - min")
        return e
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: invalid bin specification {spec}")


def _bin_measure(w, box_lo, box_hi, fname):
    np = _np()
    if w.kind == "box":
        return float(np.prod(np.clip(np.minimum(box_hi, w.hi) - np.maximum(box_lo, w.lo), 0, None)))
    if w.kind == "poly":
        V = np.array([[box_lo[0], box_lo[1]], [box_hi[0], box_lo[1]], [box_hi[0], box_hi[1]],
                      [box_lo[0], box_hi[1]]])
        return SP.poly_area(SP.clip_poly(V, w.nrm, w.off))
    raise NotImplementedError(f"{fname}: histogram densities need an interval, box or convex polygon region")


@fn("HistogramPointDensity", area=AREA, status="added",
    notes="HistogramPointDensity[pdata, bspec]: (points in bin)/|bin n W| on coordinate-wise bins spanning the "
          "bounding box of W (bins [b_k, b_k+1), the last closed); bspec per coordinate or {xspec, yspec, ...}: n "
          "(equal bins), {w} (multiples of w), {min, max, w}, {{b1, b2, ...}}, Automatic/\"Sturges\" "
          "(ceil(log2 n) + 1 bins), \"Scott\" (width 3.49 s n^(-1/3)) or \"FreedmanDiaconis\" (2 IQR n^(-1/3)), "
          "widths rounded to equal bins over the range. pdata is {p1, ...} (2D: Ripley-Rasson region), {pts, reg} "
          "or SpatialPointData in intervals, boxes or convex polygons; result is a PointDensityFunction (0 outside "
          "W and the bins). HistogramPointDensity[pproc] gives the exact constant density of homogeneous processes. "
          "Shaped bins (\"Square\", \"Hexagon\", ...), meshes, \"Knuth\", \"Wand\", bin functions, disk regions, "
          "binned data, {proc, reg}, geographic points and ensembles raise NotImplementedError. Check: exact "
          "(counts / bin areas by hand) and numerical (integral equals the binned count) in tests.")
def _histogram_point_density(pdata: Expr, bspec: Expr = None) -> Expr:
    np = _np()
    fname = "HistogramPointDensity"
    if PS.is_pproc(pdata):
        pr = PS.pproc_of(pdata, fname)
        if isinstance(pr, PS.InhomPP):
            raise NotImplementedError(f"{fname}: histogram densities of inhomogeneous processes are not supported "
                                      "(PointDensity gives the exact density function)")
        return _process_density(pdata, fname, bspec)
    P, w, reg = _data(pdata, fname)
    d = w.dim
    if isinstance(bspec, Expr) and bspec.head in ("MeshRegion", "BoundaryMeshRegion"):
        raise NotImplementedError(f"{fname}: mesh bins are not supported (give coordinate-wise bins)")
    if is_list(bspec) and len(bspec.args) == d and d > 1 and not (
            len(bspec.args) in (1, 3) and all(not is_list(a) and not isinstance(a, str) for a in bspec.args)):
        specs = list(bspec.args)
    elif is_list(bspec) and len(bspec.args) == d and d > 1 and len(bspec.args) not in (1, 3):
        specs = list(bspec.args)
    else:
        specs = [bspec] * d
    edges = [_edges1(s, P[:, k], float(w.lo[k]), float(w.hi[k]), fname) for k, s in enumerate(specs)]
    shape = [len(e) - 1 for e in edges]
    counts = np.zeros(shape)
    for p in P:
        idx = []
        for k, e in enumerate(edges):
            j = int(np.searchsorted(e, p[k], side="right")) - 1
            if j == len(e) - 1 and p[k] == e[-1]:
                j -= 1
            if j < 0 or j >= len(e) - 1:
                idx = None
                break
            idx.append(j)
        if idx is not None:
            counts[tuple(idx)] += 1
    dens = np.zeros(shape)
    tot = 0.0
    for idx in np.ndindex(*shape):
        lo = np.array([edges[k][j] for k, j in enumerate(idx)])
        hi = np.array([edges[k][j + 1] for k, j in enumerate(idx)])
        m = _bin_measure(w, lo, hi, fname)
        if m > 0:
            dens[idx] = counts[idx] / m
            tot += dens[idx] * m
        elif counts[idx] > 0:
            raise ArithmeticError(f"{fname}: internal check failed (points in a bin outside the region)")
    _check_integral(tot, float(counts.sum()), fname, "the histogram density")
    return _pdf("Histogram", BinEdges=R([[float(v) for v in e] for e in edges]), Densities=R(dens.tolist()),
                Region=reg)


# ─────────────────────────── PointValuePlot ───────────────────────────

_GRAPHICS_OPTS = {"AspectRatio", "Axes", "AxesLabel", "AxesOrigin", "AxesStyle", "Background", "Frame", "FrameLabel",
                  "FrameStyle", "FrameTicks", "GridLines", "GridLinesStyle", "ImageSize", "ImagePadding",
                  "PlotLabel", "PlotRange", "PlotRangePadding", "PlotRangeClipping", "Ticks", "BaseStyle",
                  "Epilog", "Prolog", "PlotStyle"}
_PALETTE = [(0.368417, 0.506779, 0.709798), (0.880722, 0.611041, 0.142051), (0.560181, 0.691569, 0.194885),
            (0.922526, 0.385626, 0.209179), (0.528488, 0.470624, 0.701351), (0.772079, 0.431554, 0.102387),
            (0.363898, 0.618501, 0.782349), (1.0, 0.75, 0.0), (0.647624, 0.37816, 0.614037),
            (0.571589, 0.586483, 0.0)]
_LOW, _HIGH = (0.18, 0.3, 0.75), (0.9, 0.25, 0.1)


def _pvp_data(data, fname):
    if isinstance(data, Expr) and data.head == "SpatialPointData":
        sd = SP.parse_spd(_ev(data) if len(data.args) == 1 else data, fname)
        if sd.ens:
            raise NotImplementedError(f"{fname}: ensembles of configurations are not supported")
        c = sd.configs[0]
        if not c.ann:
            raise ValueError(f"{fname}: the SpatialPointData has no annotations to plot")
        return list(c.pts), list(c.ann[0][1].args)
    if isinstance(data, Expr) and data.head == "Rule" and is_list(data.args[0]) and is_list(data.args[1]):
        pts, vals = list(data.args[0].args), list(data.args[1].args)
        if len(pts) != len(vals):
            raise ValueError(f"{fname}: the numbers of points and values differ")
        return pts, vals
    if is_list(data) and data.args and all(isinstance(r_, Expr) and r_.head == "Rule" for r_ in data.args):
        return [r_.args[0] for r_ in data.args], [r_.args[1] for r_ in data.args]
    if isinstance(data, Expr) and data.head == "Association":
        return [r_.args[0] for r_ in data.args], [r_.args[1] for r_ in data.args]
    if isinstance(data, Symbol):
        raise Unevaluated()
    if isinstance(data, Expr) and data.head in ("WeightedData", "GeoPosition", "Dataset", "QuantityArray"):
        raise NotImplementedError(f"{fname}: {data.head} input is not supported")
    raise TypeError(f"{fname}: expected {{pt -> val, ...}}, pts -> vals, <|pt -> val, ...|> or SpatialPointData")


def _color(c):
    return Expr("RGBColor", *(R(float(round(v, 6))) for v in c))


@fn("PointValuePlot", area=AREA, status="added",
    notes="PointValuePlot[{pt1 -> v1, ...}], PointValuePlot[pts -> vals], PointValuePlot[<|pt -> v, ...|>] and "
          "PointValuePlot[SpatialPointData] (first annotation) with encoding enc: \"Color\" (default for numbers; "
          "linear blend from RGBColor[0.18, 0.3, 0.75] at the minimum to RGBColor[0.9, 0.25, 0.1] at the maximum), "
          "\"Size\" (PointSize[0.01 + 0.03 Sqrt[t]], t the min-max scaled value, bubble area ~ value), "
          "\"DiscreteColor\" (default for non-numbers; one palette colour per distinct value in sorted order) or "
          "\"Label\" (Text labels next to the points); explicit colour values are used as given. Returns "
          "Graphics (2D) or Graphics3D (3D points) with Frame -> True and the given Graphics options (PlotStyle "
          "is prepended to every point). \"Shape\", encoding functions and n -> enc forms, list values, "
          "geographic points, quantities, wrappers, WeightedData and ColorFunction raise NotImplementedError. "
          "Check: structural (one styled Point per datum, colours and sizes ordered like the values) in tests.")
def _point_value_plot(data: Expr, *rest: Expr) -> Expr:
    fname = "PointValuePlot"
    rest = list(rest)
    opts = []
    while rest and isinstance(rest[-1], Expr) and rest[-1].head in ("Rule", "RuleDelayed") and \
            isinstance(rest[-1].args[0], Symbol):
        opts.insert(0, rest.pop())
    for o in opts:
        nm = o.args[0].name
        if nm not in _GRAPHICS_OPTS:
            if nm in ("ColorFunction", "PlotLegends", "PlotMarkers", "BubbleSizes", "BubbleScale",
                      "ColorFunctionScaling", "LabelingFunction", "ScalingFunctions", "PlotTheme") or \
                    nm.startswith("Geo"):
                raise NotImplementedError(f"{fname}: the option {nm} is not supported")
            raise ValueError(f"{fname}: unknown option {nm}")
    if len(rest) > 1:
        raise TypeError(f"{fname}: expected PointValuePlot[data] or PointValuePlot[data, enc]")
    enc = rest[0] if rest else Symbol("Automatic")
    pts, vals = _pvp_data(data, fname)
    if not pts:
        raise ValueError(f"{fname}: no data to plot")
    d = None
    for p in pts:
        if isinstance(p, Expr) and p.head in ("GeoPosition", "Entity", "Dated"):
            raise NotImplementedError(f"{fname}: geographic locations are not supported")
        if not is_list(p) or not all(not is_list(c) and not isinstance(c, str) for c in p.args):
            raise TypeError(f"{fname}: each point must be a list of coordinates")
        for c in p.args:
            if isinstance(c, Expr) and c.head in ("Quantity", "Around", "Interval"):
                raise NotImplementedError(f"{fname}: {c.head} coordinates are not supported")
            fnum(c, fname, "a coordinate")
        if d is None:
            d = len(p.args)
        elif len(p.args) != d:
            raise ValueError(f"{fname}: all points must have the same dimension")
    if d not in (2, 3):
        raise NotImplementedError(f"{fname}: only 2D and 3D points are supported")
    for v in vals:
        if is_list(v):
            raise NotImplementedError(f"{fname}: list values (several encodings) are not supported")
        if isinstance(v, Expr) and v.head in ("Quantity", "Around"):
            raise NotImplementedError(f"{fname}: {v.head} values are not supported")
    is_col = [isinstance(v, Expr) and v.head in ("RGBColor", "Hue", "GrayLevel", "CMYKColor", "LABColor") for v in vals]
    nums = []
    numeric = True
    for v in vals:
        try:
            if isinstance(v, (str, bool)):
                raise TypeError
            x = S(v)
            if not (isinstance(x, sp.Basic) and x.is_number and x.is_real):
                raise TypeError
            nums.append(float(x))
        except (TypeError, ValueError, Unevaluated):
            numeric = False
            break
    if isinstance(enc, Symbol) and enc.name == "Automatic":
        enc = "Color" if numeric else "DiscreteColor"
    if not isinstance(enc, str):
        raise NotImplementedError(f"{fname}: encoding functions and n -> enc forms are not supported")
    if enc == "Shape":
        raise NotImplementedError(f"{fname}: the \"Shape\" encoding is not supported")
    if enc not in ("Color", "Size", "DiscreteColor", "Label"):
        raise ValueError(f"{fname}: unknown encoding \"{enc}\" (use \"Color\", \"Size\", \"DiscreteColor\" or "
                         "\"Label\")")
    if enc in ("Color", "Size") and not numeric:
        if enc == "Color" and all(is_col):
            pass
        else:
            raise TypeError(f"{fname}: the \"{enc}\" encoding needs real numeric values")
    style = [o.args[1] for o in opts if o.args[0].name == "PlotStyle"]
    gopts = [o for o in opts if o.args[0].name != "PlotStyle"]
    if numeric and nums:
        lo, hi = min(nums), max(nums)
        ts = [(x - lo) / (hi - lo) if hi > lo else 0.5 for x in nums]
    prims = []
    if enc == "DiscreteColor":
        keys = []
        for v in vals:
            if v not in keys:
                keys.append(v)
        try:
            keys = sorted(keys, key=lambda k: (0, float(S(k))) if numeric else (1, str(k)))
        except Exception:
            pass
    for i, (p, v) in enumerate(zip(pts, vals)):
        if enc == "Color":
            col = v if is_col[i] else _color(tuple(a + (b - a) * ts[i] for a, b in zip(_LOW, _HIGH)))
            dirs = [col, Expr("PointSize", R(0.02))]
        elif enc == "Size":
            dirs = [_color(_PALETTE[0]), Expr("PointSize", R(round(0.01 + 0.03 * math.sqrt(ts[i]), 6)))]
        elif enc == "DiscreteColor":
            dirs = [v if is_col[i] else _color(_PALETTE[keys.index(v) % len(_PALETTE)]),
                    Expr("PointSize", R(0.02))]
        else:
            dirs = [_color(_PALETTE[0]), Expr("PointSize", R(0.015))]
        item = list(style) + dirs + [Expr("Point", p)]
        if enc == "Label":
            item.append(Expr("Text", v if isinstance(v, str) else Expr("ToString", v), p, L(R(-1.5), R(0))))
        prims.append(L(*item))
    head = "Graphics" if d == 2 else "Graphics3D"
    extra = [] if any(o.args[0].name == "Frame" for o in gopts) or d == 3 else \
        [rule(Symbol("Frame"), True)]
    return _ev(Expr(head, L(*prims), *extra, *gopts))
