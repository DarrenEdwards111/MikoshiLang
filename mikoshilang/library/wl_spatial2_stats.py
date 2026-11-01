"""Wolfram batch X08: spatial statistics II -- summary statistics of point data with boundary corrections,
PointStatisticFunction, point-process estimation and goodness-of-fit tests.

Estimators (Baddeley, Rubak & Turner, Spatial Point Patterns, CRC 2015, ch. 7-8; Illian et al. 2008):

* RipleyK: K(r) = |W| / (n (n - 1)) Sum_{i != j} 1[d_ij <= r] e_ij with the correction e_ij of
  SpatialBoundaryCorrection: None (1), "Ripley" (2D isotropic: 1/fraction of the circle about x_i through x_j inside
  W), "Translation" (|W| / |W n (W + x_i - x_j)|, boxes and balls) or "BorderMargin" (reduced sample:
  |W| Sum_{b_i >= r} #{j : d_ij <= r} / (n #{i : b_i >= r})).  Automatic is "Ripley" in 2D, "Translation" otherwise.
  BesagL is (K / omega_d)^(1/d).
* PairCorrelationG: Epanechnikov kernel estimate with half-width h = 0.15 lambda^(-1/d) (Stoyan & Stoyan) and the
  Ripley / Translation weights.
* NearestNeighborG (nearest-neighbour distances d_i, boundary distances b_i): None (empirical CDF), "BorderMargin"
  (#{d_i <= r, b_i >= r}/#{b_i >= r}), "Hanisch" (Hanisch 1984), "KaplanMeier" (default; Baddeley & Gill 1997) and
  "NelsonAalen".  EmptySpaceF uses the distances from a regular grid of 16384 test locations in the bounding box
  (those inside W), with None, "BorderMargin", "KaplanMeier" (default) or "NelsonAalen"; SpatialJ = (1 - G)/(1 - F).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).  Results are machine numbers.
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel_lists as KL
from .core import Unevaluated, fn, L, R, S, is_list, rule
from . import wl_processes_spatial as PS
from . import wl_spatial2 as SP
from .wl_spatial2 import _np, _ev, _opt_name, fnum, is_auto, window, spd_of, config_array, split_opts

AREA = "spatial statistics II (X08)"

CORRECTIONS = {
    "RipleyK": ("None", "Ripley", "Translation", "BorderMargin"),
    "BesagL": ("None", "Ripley", "Translation", "BorderMargin"),
    "PairCorrelationG": ("None", "Ripley", "Translation"),
    "NearestNeighborG": ("None", "BorderMargin", "Hanisch", "KaplanMeier", "NelsonAalen"),
    "EmptySpaceF": ("None", "BorderMargin", "KaplanMeier", "NelsonAalen"),
    "SpatialJ": ("None", "BorderMargin", "KaplanMeier", "NelsonAalen"),
}


def _corr(name, opts, w, fname):
    v = opts.get("SpatialBoundaryCorrection")
    if v is None or v == Symbol("Automatic"):
        if name in ("RipleyK", "BesagL", "PairCorrelationG"):
            return "Ripley" if w.dim == 2 else "Translation"
        return "KaplanMeier"
    if v == Symbol("None"):
        return "None"
    if not isinstance(v, str):
        raise TypeError(f"{fname}: SpatialBoundaryCorrection must be Automatic, None or a method name")
    if v not in CORRECTIONS[name]:
        raise ValueError(f"{fname}: SpatialBoundaryCorrection -> \"{v}\" is not available for {name} (use "
                         f"{', '.join(CORRECTIONS[name])})")
    return v


# ─────────────────────────── geometry helpers ───────────────────────────

def circ_frac_vec(w, P, r, fname):
    """fraction of the circle of radius r[k] about P[k] (2D) inside the window, vectorised"""
    np = _np()
    if w.dim != 2:
        raise NotImplementedError(f"{fname}: Ripley's isotropic correction is only available in 2D")
    if w.kind == "ball":
        t = np.linalg.norm(P - w.c, axis=1)
        Rw = w.radii[0]
        out = np.ones(len(P))
        cut = t + r > Rw
        with np.errstate(all="ignore"):
            cosv = np.clip((t * t + r * r - Rw * Rw) / (2 * t * r), -1.0, 1.0)
        out[cut] = np.arccos(cosv[cut]) / math.pi
        return out
    if w.kind == "ellipsoid":
        raise NotImplementedError(f"{fname}: Ripley's correction in elliptical windows is not supported")
    if w.kind == "box":
        nrm = np.array([[-1.0, 0], [1.0, 0], [0, -1.0], [0, 1.0]])
        off = np.array([-w.lo[0], w.hi[0], -w.lo[1], w.hi[1]])
    else:
        nrm, off = w.nrm, w.off
    dist = off[None, :] - P @ nrm.T                     # N x m
    ang = np.arctan2(nrm[:, 1], nrm[:, 0])[None, :]
    with np.errstate(all="ignore"):
        a = np.where(dist < r[:, None], np.arccos(np.clip(dist / r[:, None], -1, 1)), 0.0)
    s = np.mod(ang - a, 2 * math.pi)
    e = s + 2 * a
    # split arcs crossing 2 Pi
    wrap = e > 2 * math.pi
    s2 = np.where(wrap, 0.0, 0.0)
    e2 = np.where(wrap, e - 2 * math.pi, 0.0)
    e = np.minimum(e, 2 * math.pi)
    S_ = np.concatenate([s, s2], axis=1)
    E_ = np.concatenate([e, e2], axis=1)
    order = np.argsort(S_, axis=1)
    S_ = np.take_along_axis(S_, order, axis=1)
    E_ = np.take_along_axis(E_, order, axis=1)
    cm = np.maximum.accumulate(E_, axis=1)
    prev = np.concatenate([np.zeros((len(P), 1)), cm[:, :-1]], axis=1)
    cover = np.maximum(0.0, E_ - np.maximum(S_, prev)).sum(axis=1)
    return 1.0 - np.minimum(cover, 2 * math.pi) / (2 * math.pi)


def trans_weight(w, Dv, fname):
    """|W| / |W n (W + v)| for difference vectors Dv (N x d)"""
    np = _np()
    if w.kind == "box":
        Lk = w.hi - w.lo
        inter = np.prod(np.maximum(Lk - np.abs(Dv), 0.0), axis=1)
    elif w.kind == "ball":
        t = np.linalg.norm(Dv, axis=1)
        inter = SP.lens_fraction(t, w.radii[0], w.dim, fname) * w.measure
    else:
        raise NotImplementedError(f"{fname}: the translation correction needs a box or ball window")
    with np.errstate(divide="ignore"):
        return np.where(inter > 0, w.measure / inter, np.inf)


def _dist_matrix(P):
    np = _np()
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    return D


def _limit(n, fname):
    if n > 5000:
        raise NotImplementedError(f"{fname}: more than 5000 points are not supported")


def _unit_ball_f(d):
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1)


def k_estimate(P, w, rs, corr, fname="RipleyK"):
    np = _np()
    n = len(P)
    if n < 2:
        raise ValueError(f"{fname}: at least two points are needed")
    _limit(n, fname)
    D = _dist_matrix(P)
    rmax = float(np.max(rs)) if len(rs) else 0.0
    I, J = np.nonzero(D <= rmax)
    dij = D[I, J]
    if corr == "None":
        wts = np.ones(len(I))
    elif corr == "Ripley":
        wts = 1.0 / np.maximum(circ_frac_vec(w, P[I], dij, fname), 1e-12)
    elif corr == "Translation":
        wts = trans_weight(w, P[I] - P[J], fname)
    else:   # BorderMargin
        b = w.bdist(P, fname)
        out = []
        for r in rs:
            keep = b >= r
            m = int(keep.sum())
            if m == 0:
                raise ValueError(f"{fname}: no point is at least r = {r} from the boundary (BorderMargin)")
            cnt = int(((D <= r) & keep[:, None]).sum())
            out.append(w.measure * cnt / (n * m))
        return np.array(out)
    order = np.argsort(dij)
    cw = np.concatenate([[0.0], np.cumsum(wts[order])])
    idx = np.searchsorted(dij[order], rs, side="right")
    return w.measure * cw[idx] / (n * (n - 1))


def pcf_estimate(P, w, rs, corr, fname="PairCorrelationG"):
    np = _np()
    n, d = P.shape
    if n < 2:
        raise ValueError(f"{fname}: at least two points are needed")
    if np.any(rs <= 0):
        raise ValueError(f"{fname}: the pair correlation estimate needs positive radii")
    _limit(n, fname)
    lam = n / w.measure
    h = 0.15 * lam ** (-1.0 / d)
    D = _dist_matrix(P)
    I, J = np.nonzero(D <= float(np.max(rs)) + h)
    dij = D[I, J]
    if corr == "None":
        wts = np.ones(len(I))
    elif corr == "Ripley":
        wts = 1.0 / np.maximum(circ_frac_vec(w, P[I], dij, fname), 1e-12)
    else:
        wts = trans_weight(w, P[I] - P[J], fname)
    out = []
    sd = d * _unit_ball_f(d)
    for r in rs:
        t = (r - dij) / h
        k = np.where(np.abs(t) < 1, 0.75 * (1 - t * t) / h, 0.0)
        out.append(w.measure * float((k * wts).sum()) / (n * (n - 1) * sd * r ** (d - 1)))
    return np.array(out)


def _km(t, ev, rs):
    """Kaplan-Meier CDF at rs from censored times t and event flags ev"""
    np = _np()
    order = np.argsort(t, kind="mergesort")
    t, ev = t[order], ev[order]
    n = len(t)
    ut = np.unique(t[ev])
    surv = []
    s = 1.0
    jumps = []
    for u in ut:
        at_risk = n - np.searchsorted(t, u, side="left")
        dk = int(((t == u) & ev).sum())
        s *= 1.0 - dk / at_risk
        jumps.append((u, s))
    out = []
    for r in rs:
        v = 1.0
        for u, sv in jumps:
            if u <= r:
                v = sv
            else:
                break
        out.append(1.0 - v)
    return np.array(out)


def _na(t, ev, rs):
    np = _np()
    order = np.argsort(t, kind="mergesort")
    t, ev = t[order], ev[order]
    n = len(t)
    H = []
    h = 0.0
    for u in np.unique(t[ev]):
        at_risk = n - np.searchsorted(t, u, side="left")
        h += int(((t == u) & ev).sum()) / at_risk
        H.append((u, h))
    out = []
    for r in rs:
        v = 0.0
        for u, hv in H:
            if u <= r:
                v = hv
            else:
                break
        out.append(1.0 - math.exp(-v))
    return np.array(out)


def _censored_cdf(dist, b, rs, corr, w, fname):
    np = _np()
    if corr == "None":
        return np.array([float((dist <= r).mean()) for r in rs])
    if corr == "BorderMargin":
        out = []
        for r in rs:
            keep = b >= r
            if not keep.any():
                raise ValueError(f"{fname}: no location is at least r = {r} from the boundary (BorderMargin)")
            out.append(float((keep & (dist <= r)).sum() / keep.sum()))
        return np.array(out)
    if corr == "Hanisch":
        ok = dist <= b
        if not ok.any():
            raise ValueError(f"{fname}: no nearest-neighbour distance is uncensored (Hanisch)")
        er = np.array([w.eroded_measure(float(x), fname) for x in dist[ok]])
        wt = np.where(er > 0, 1.0 / np.where(er > 0, er, 1.0), 0.0)
        tot = wt.sum()
        return np.array([float(wt[dist[ok] <= r].sum() / tot) for r in rs])
    t = np.minimum(dist, b)
    ev = dist <= b
    if not ev.any():
        raise ValueError(f"{fname}: all distances are censored by the boundary")
    return _km(t, ev, rs) if corr == "KaplanMeier" else _na(t, ev, rs)


def g_estimate(P, w, rs, corr, fname="NearestNeighborG"):
    np = _np()
    n = len(P)
    if n < 2:
        raise ValueError(f"{fname}: at least two points are needed")
    _limit(n, fname)
    dist = _dist_matrix(P).min(axis=1)
    b = w.bdist(P, fname) if corr != "None" else None
    return _censored_cdf(dist, b, rs, corr, w, fname)


def f_grid(w):
    np = _np()
    d = w.dim
    m = max(2, int(round(16384 ** (1.0 / d))))
    axes = [w.lo[k] + (np.arange(m) + 0.5) * (w.hi[k] - w.lo[k]) / m for k in range(d)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
    return G[w.contains(G)]


def f_estimate(P, w, rs, corr, fname="EmptySpaceF"):
    np = _np()
    if len(P) < 1:
        raise ValueError(f"{fname}: at least one point is needed")
    G = f_grid(w)
    dist = np.full(len(G), np.inf)
    for k in range(0, len(P), 256):
        Q = P[k:k + 256]
        dd = np.sqrt(((G[:, None, :] - Q[None, :, :]) ** 2).sum(-1)).min(axis=1)
        dist = np.minimum(dist, dd)
    b = w.bdist(G, fname) if corr != "None" else None
    return _censored_cdf(dist, b, rs, corr, w, fname)


def stat_values(name, P, w, rs, corr, fname):
    np = _np()
    if name in ("RipleyK", "BesagL"):
        k = k_estimate(P, w, rs, corr, fname)
        if name == "RipleyK":
            return k
        return (k / _unit_ball_f(w.dim)) ** (1.0 / w.dim)
    if name == "PairCorrelationG":
        return pcf_estimate(P, w, rs, corr, fname)
    if name == "NearestNeighborG":
        return g_estimate(P, w, rs, corr, fname)
    if name == "EmptySpaceF":
        return f_estimate(P, w, rs, corr, fname)
    G = g_estimate(P, w, rs, corr, fname)
    F = f_estimate(P, w, rs, corr, fname)
    if np.any(F >= 1):
        raise ValueError(f"{fname}: SpatialJ is undefined where EmptySpaceF = 1")
    return (1 - G) / (1 - F)


def _radii(r, fname):
    np = _np()
    if is_list(r):
        vals = [fnum(x, fname, "the radius") for x in r.args]
        lst = True
    else:
        vals = [fnum(r, fname, "the radius")]
        lst = False
    if any(v < 0 for v in vals):
        raise ValueError(f"{fname}: the radius must be non-negative")
    return np.array(vals, dtype=float), lst


def data_stat(name):
    def impl(pdata, r, opts):
        fname = name
        _, o = split_opts(list(opts), ("SpatialBoundaryCorrection",), fname)
        sd, w = spd_of(pdata, fname)
        corr = _corr(name, o, w, fname)
        if sd.ens:
            raise NotImplementedError(f"{fname}: ensembles of configurations are not supported; apply it to each "
                                      "configuration")
        if r is None:
            data = sd.expr()
            return Expr("PointStatisticFunction", Expr("Association", rule("Statistic", name), rule("Data", data),
                                                       rule("SpatialBoundaryCorrection", corr)))
        if isinstance(r, Symbol):
            raise Unevaluated()
        rs, lst = _radii(r, fname)
        P = config_array(sd.configs[0], w, fname)
        v = stat_values(name, P, w, rs, corr, fname)
        out = [R(float(x)) for x in v]
        return L(*out) if lst else out[0]
    return impl


for _n in CORRECTIONS:
    PS.DATA_STATS[_n] = data_stat(_n)


def _psf_parts(x, fname="PointStatisticFunction"):
    if not (isinstance(x, Expr) and x.head == "PointStatisticFunction" and len(x.args) == 1 and
            isinstance(x.args[0], Expr) and x.args[0].head == "Association"):
        raise TypeError(f"{fname}: expected PointStatisticFunction[<|\"Statistic\" -> ..., \"Data\" -> ..., "
                        "\"SpatialBoundaryCorrection\" -> ...|>]")
    d = {}
    for r_ in x.args[0].args:
        d[r_.args[0]] = r_.args[1]
    for k in ("Statistic", "Data", "SpatialBoundaryCorrection"):
        if k not in d:
            raise ValueError(f"{fname}: the key \"{k}\" is missing")
    if d["Statistic"] not in CORRECTIONS:
        raise ValueError(f"{fname}: unknown statistic {d['Statistic']}")
    return d


@fn("PointStatisticFunction", area=AREA, status="added",
    notes="PointStatisticFunction[<|\"Statistic\" -> name, \"Data\" -> spd, \"SpatialBoundaryCorrection\" -> c|>] is "
          "produced by RipleyK, BesagL, PairCorrelationG, NearestNeighborG, EmptySpaceF and SpatialJ applied to point "
          "data without a radius; psf[r] and psf[{r1, ...}] give the estimate (as the function with the radius), "
          "psf[\"Statistic\" | \"Data\" | \"SpatialBoundaryCorrection\" | \"Properties\"] its parts. Inert; other "
          "forms raise TypeError/ValueError, symbolic r stays unevaluated. Check: exact (same values as the direct "
          "call) in tests.")
def _psf(spec: Assoc) -> Expr:
    _psf_parts(Expr("PointStatisticFunction", spec))
    raise Unevaluated()


def _psf_apply(head, args, ev):
    from .. import kernel as K
    if len(args) != 1:
        return K.UNHANDLED
    d = _psf_parts(head)
    a = args[0]
    if isinstance(a, str):
        if a == "Properties":
            return L("Data", "SpatialBoundaryCorrection", "Statistic")
        if a in d:
            return d[a]
        raise ValueError(f"PointStatisticFunction: unknown property {a!r}")
    if isinstance(a, Symbol):
        return K.UNHANDLED
    name = d["Statistic"]
    return PS.DATA_STATS[name](d["Data"], a, (rule(Symbol("SpatialBoundaryCorrection"),
                                                  Symbol("None") if d["SpatialBoundaryCorrection"] == "None" else
                                                  d["SpatialBoundaryCorrection"]),))


KL.COMPOUND_HEAD_HOOKS["PointStatisticFunction"] = _psf_apply


def _inert_option(name, notes):
    def _opt(*args: Expr) -> Expr:
        raise Unevaluated()
    fn(name, area=AREA, status="added", notes=notes + " Check: inert (stays unevaluated as a symbol, an expression or "
                                                     "in a rule).")(_opt)


_inert_option("SpatialBoundaryCorrection",
              "option of RipleyK, BesagL, PairCorrelationG (Automatic, None, \"Ripley\", \"Translation\"), "
              "NearestNeighborG (Automatic, None, \"BorderMargin\", \"Hanisch\", \"KaplanMeier\", \"NelsonAalen\"), "
              "EmptySpaceF and SpatialJ (Automatic, None, \"BorderMargin\", \"KaplanMeier\", \"NelsonAalen\") and "
              "RipleyK/BesagL \"BorderMargin\" for point-data estimates; Automatic is \"Ripley\" (2D) or "
              "\"Translation\" for K, L and g and \"KaplanMeier\" for G, F and J; settings a statistic does not "
              "define raise ValueError.")
_inert_option("PointProcessEstimator",
              "option of EstimatedPointProcess and FindPointProcessParameters: Automatic, \"MaximumLikelihood\" "
              "(PoissonPointProcess: n/|W|), \"MethodOfMoments\" (minimum contrast on Ripley's K for Thomas, "
              "Matern, Cauchy and variance-gamma cluster processes), \"MaximumPseudoLikelihood\" (Gibbs pairwise "
              "processes); \"FindClusters\" and estimators a process does not support raise NotImplementedError.")


# ─────────────────────────── estimation ───────────────────────────

def _rmax(w):
    np = _np()
    if w.kind == "ball":
        return float(w.radii[0]) / 2
    return float(np.min(w.hi - w.lo)) / 4


def _sym_name(v):
    return v.name if isinstance(v, Symbol) else None


def _cluster_K(head, prm, d, r):
    """model K for the cluster processes (numpy, needs SciPy)"""
    np = _np()
    from scipy import special
    om = _unit_ball_f(d)
    mu = prm["mu"]
    if head == "ThomasPointProcess":
        F = special.gammainc(d / 2, r * r / (4 * prm["s"] ** 2))
    elif head == "CauchyPointProcess":
        b = prm["s"]
        F = special.betainc(d / 2, 0.5, r * r / (r * r + 4 * b * b))
    elif head == "MaternPointProcess":
        Rr = prm["s"]
        x, wq = np.polynomial.legendre.leggauss(64)
        F = []
        for rv in np.atleast_1d(r):
            top = min(rv, 2 * Rr)
            if top <= 0:
                F.append(0.0)
                continue
            s_ = (x + 1) * top / 2
            F.append(float((wq * SP.lens_fraction(s_, Rr, d) * d * s_ ** (d - 1)).sum() * top / 2 / Rr ** d))
        F = np.array(F)
    else:   # VarianceGamma
        a2, be = 2 * prm["al"], prm["be"]
        xk, wk = special.roots_genlaguerre(80, a2 - 1)
        U = be * xk
        F = (wk[None, :] * special.gammainc(d / 2, np.atleast_1d(r)[:, None] ** 2 / (2 * U[None, :]))).sum(axis=1) \
            / special.gamma(a2)
    return om * r ** d + F / mu


_CLUSTER = {"ThomasPointProcess": ["mu", "lam", "s"], "MaternPointProcess": ["mu", "lam", "s"],
            "CauchyPointProcess": ["mu", "lam", "s"], "VarianceGammaPointProcess": ["mu", "lam", "al", "be"]}
_GIBBS = {"HardcorePointProcess": ["mu", "rh"], "StraussPointProcess": ["mu", "ga", "rs"],
          "StraussHardcorePointProcess": ["mu", "ga", "rh", "rs"], "PenttinenPointProcess": ["mu", "ga", "rp"],
          "DiggleGatesPointProcess": ["mu", "rho"], "DiggleGrattonPointProcess": ["mu", "ka", "de", "rho"]}


def _positive_value(v, fname, what):
    x = fnum(v, fname, what)
    if not x > 0:
        raise ValueError(f"{fname}: the starting value of {what} must be positive")
    return x


def _parse_target(pproc, sd, w, starts, fname):
    """(head, names, values (float or None for free), symbols, d)"""
    if not (isinstance(pproc, Expr) and isinstance(pproc.head, str) and pproc.head in PS.PCLASSES):
        if isinstance(pproc, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {pproc} is not a spatial point process")
    h = pproc.head
    a = list(pproc.args)
    if h == "PoissonPointProcess":
        names = ["mu"]
    elif h in _CLUSTER:
        names = _CLUSTER[h]
    elif h in _GIBBS:
        names = _GIBBS[h]
    else:
        raise NotImplementedError(f"{fname}: estimation of {h} is not supported")
    if len(a) != len(names) + 1:
        raise ValueError(f"{fname}: {h} has {len(names) + 1} arguments")
    d = a[-1]
    if isinstance(d, Symbol):
        d = w.dim
    elif d != w.dim:
        raise ValueError(f"{fname}: the process dimension {d} differs from the data dimension {w.dim}")
    vals, syms = {}, {}
    for nm, v in zip(names, a[:-1]):
        if isinstance(v, Symbol):
            vals[nm], syms[nm] = None, v
        else:
            x = S(v)
            if isinstance(x, sp.Basic) and x.free_symbols:
                raise NotImplementedError(f"{fname}: parameters must be symbols or numbers, not {v}")
            vals[nm] = fnum(v, fname, "a parameter")
    for s_, v0 in starts.items():
        if s_ not in [x.name for x in syms.values()]:
            raise ValueError(f"{fname}: {s_} is not a free parameter of {pproc}")
    return h, names, vals, syms, d


def _starts(spec, fname):
    if spec is None:
        return {}
    if not is_list(spec):
        raise TypeError(f"{fname}: starting values must be {{{{p, p0}}, ...}}")
    out = {}
    for pr in spec.args:
        if not (is_list(pr) and len(pr.args) == 2 and isinstance(pr.args[0], Symbol)):
            raise TypeError(f"{fname}: starting values must be {{{{p, p0}}, ...}}")
        out[pr.args[0].name] = pr.args[1]
    return out


def _est_options(opts, fname):
    for k in ("AccuracyGoal", "PrecisionGoal", "WorkingPrecision"):
        v = opts.get(k)
        if v is not None and not is_auto(v) and not (k == "WorkingPrecision" and v == Symbol("MachinePrecision")):
            raise NotImplementedError(f"{fname}: {k} -> {v} is not supported")
    est = opts.get("PointProcessEstimator")
    if est is None or est == Symbol("Automatic"):
        return "Automatic"
    if is_list(est) and est.args:
        raise NotImplementedError(f"{fname}: estimator suboptions (Method -> ...) are not supported")
    if est not in ("MaximumLikelihood", "MaximumPseudoLikelihood", "MethodOfMoments", "FindClusters"):
        raise ValueError(f"{fname}: unknown PointProcessEstimator {est}")
    if est == "FindClusters":
        raise NotImplementedError(f"{fname}: PointProcessEstimator -> \"FindClusters\" is not supported (use "
                                  "\"MethodOfMoments\")")
    return est


def estimate(pdata, pproc, spec, opts, fname):
    np = _np()
    est = _est_options(opts, fname)
    sd, w = spd_of(pdata, fname)
    starts = _starts(spec, fname)
    h, names, vals, syms, d = _parse_target(pproc, sd, w, starts, fname)
    Ps = [config_array(c, w, fname) for c in sd.configs]
    ntot = sum(len(P) for P in Ps)
    if ntot == 0:
        raise ValueError(f"{fname}: the point data are empty")
    lam_hat = ntot / (len(Ps) * w.measure)
    if h == "PoissonPointProcess":
        if est not in ("Automatic", "MaximumLikelihood"):
            raise NotImplementedError(f"{fname}: PoissonPointProcess is estimated by maximum likelihood")
        if vals["mu"] is None:
            meas = _ev(Expr("RegionMeasure", sd.reg))
            mu = _ev(Expr("Divide", sp.Integer(ntot), Expr("Times", len(Ps), meas)))
            return h, names, {"mu": mu}, syms, d
        return h, names, {}, syms, d
    if h in _CLUSTER:
        if est not in ("Automatic", "MethodOfMoments"):
            raise NotImplementedError(f"{fname}: {h} is estimated with \"MethodOfMoments\" (minimum contrast)")
        SP._need_scipy(fname, "minimum-contrast estimation")
        fitted = _min_contrast(h, names, vals, syms, d, Ps, w, lam_hat, starts, fname)
    else:
        if est not in ("Automatic", "MaximumPseudoLikelihood"):
            raise NotImplementedError(f"{fname}: {h} is estimated with \"MaximumPseudoLikelihood\"")
        if len(Ps) != 1:
            raise NotImplementedError(f"{fname}: pseudo-likelihood estimation from ensembles is not supported")
        fitted = _mple(h, vals, syms, d, Ps[0], w, fname)
    return h, names, {k: R(float(v)) for k, v in fitted.items() if vals.get(k) is None}, syms, d


def _min_contrast(h, names, vals, syms, d, Ps, w, lam_hat, starts, fname):
    """Diggle-Gratton minimum contrast: minimise Sum (Khat(r)^(1/4) - K(r)^(1/4))^2 on 64 radii in (0, rmax]"""
    np = _np()
    from scipy.optimize import minimize
    rmax = _rmax(w)
    rs = rmax * np.arange(1, 65) / 64
    corr = "Ripley" if w.dim == 2 else "Translation"
    Kh = np.mean([k_estimate(P, w, rs, corr, fname) for P in Ps], axis=0)
    q = 0.25
    free_mu = vals["mu"] is None and vals["lam"] is None
    shape = [nm for nm in names[2:] if vals[nm] is None]
    opt_names = (["mu"] if free_mu else []) + shape

    def params(x):
        p = dict(vals)
        for nm, v in zip(opt_names, x):
            p[nm] = math.exp(v)
        if p["mu"] is None:
            p["mu"] = lam_hat / p["lam"]
        if p["lam"] is None:
            p["lam"] = lam_hat / p["mu"]
        return p

    def obj(x):
        p = params(x)
        with np.errstate(all="ignore"):
            Km = _cluster_K(h, p, d, rs)
        v = float(np.sum((np.maximum(Kh, 0) ** q - np.maximum(Km, 0) ** q) ** 2))
        return v if math.isfinite(v) else 1e300
    if not opt_names:
        return params([])
    start_vals = {}
    for nm in opt_names:
        if syms[nm].name in starts:
            start_vals[nm] = [math.log(_positive_value(starts[syms[nm].name], fname, syms[nm].name))]
    grids = {"mu": [math.log(lam_hat / c) for c in (2, 5, 10, 20, 50)],
             "s": [math.log(rmax * c) for c in (0.02, 0.05, 0.1, 0.2, 0.5)],
             "al": [math.log(c) for c in (0.5, 1.0, 2.0)],
             "be": [math.log((rmax * c) ** 2) for c in (0.02, 0.05, 0.1, 0.3)]}
    import itertools
    cands = itertools.product(*[start_vals.get(nm, grids[nm]) for nm in opt_names])
    best = min(cands, key=lambda x: obj(np.array(x)))
    res = minimize(obj, np.array(best, dtype=float), method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-14, "maxiter": 4000, "maxfev": 8000})
    x = res.x if res.fun <= obj(np.array(best)) else np.array(best)
    p = params(x)
    for nm in opt_names:
        if not (math.isfinite(p[nm]) and p[nm] > 0):
            raise ArithmeticError(f"{fname}: the minimum-contrast fit failed")
    return p


def _dummy(w, B, fname):
    """quadrature points (cell centres of a regular grid with 16384 cells over the bounding box) inside the window
    eroded by B, and the cell volume"""
    np = _np()
    G = f_grid(w)
    m = max(2, int(round(16384 ** (1.0 / w.dim))))
    vol = float(np.prod((w.hi - w.lo) / m))
    if B > 0:
        G = G[w.bdist(G, fname) >= B]
    return G, vol


def _gibbs_terms(h, p, X, Q):
    """(covariate S, offset, allowed) of the Papangelou density at locations Q given the configuration X: log
    lambda* = log mu + theta S + offset"""
    np = _np()
    Dm = np.sqrt(((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)) if len(X) else np.zeros((len(Q), 0))
    off = np.zeros(len(Q))
    Sv = np.zeros(len(Q))
    allowed = np.ones(len(Q), dtype=bool)
    if h == "HardcorePointProcess":
        allowed = ~np.any(Dm < p["rh"], axis=1)
    elif h == "StraussPointProcess":
        Sv = (Dm < p["rs"]).sum(axis=1).astype(float)
    elif h == "StraussHardcorePointProcess":
        allowed = ~np.any(Dm < p["rh"], axis=1)
        Sv = ((Dm >= p["rh"]) & (Dm < p["rs"])).sum(axis=1).astype(float)
    elif h == "PenttinenPointProcess":
        Sv = SP.lens_fraction(Dm, p["rp"], X.shape[1] if len(X) else Q.shape[1]).sum(axis=1)
    elif h == "DiggleGatesPointProcess":
        rho = p["rho"]
        with np.errstate(divide="ignore"):
            lg = np.where(Dm < rho, 2 * np.log(np.sin(math.pi * Dm / (2 * rho))), 0.0)
        off = lg.sum(axis=1)
        allowed = np.isfinite(off)
        off = np.where(allowed, off, 0.0)
    else:   # DiggleGratton
        de, rho = p["de"], p["rho"]
        allowed = ~np.any(Dm < de, axis=1)
        mid = (Dm >= de) & (Dm < rho)
        with np.errstate(divide="ignore"):
            lg = np.where(mid, np.log(np.maximum(Dm - de, 0) / (rho - de)) if rho > de else 0.0, 0.0)
        Sv = lg.sum(axis=1)
        bad = ~np.isfinite(Sv)
        allowed &= ~bad
        Sv = np.where(bad, 0.0, Sv)
    return Sv, off, allowed


_THETA = {"StraussPointProcess": ("ga", "log", -1), "StraussHardcorePointProcess": ("ga", "log", -1),
          "PenttinenPointProcess": ("ga", "log", -1), "DiggleGrattonPointProcess": ("ka", "id", 1)}
_RANGE = {"HardcorePointProcess": "rh", "StraussPointProcess": "rs", "StraussHardcorePointProcess": "rs",
          "PenttinenPointProcess": "rp", "DiggleGatesPointProcess": "rho", "DiggleGrattonPointProcess": "rho"}
_HARD = {"HardcorePointProcess": "rh", "StraussHardcorePointProcess": "rh", "DiggleGrattonPointProcess": "de"}


def _range_of(h, p):
    k = _RANGE[h]
    return 2 * p[k] if h == "PenttinenPointProcess" else p[k]


def _mple(h, vals, syms, d, P, w, fname):
    """Baddeley-Turner maximum pseudo-likelihood with border correction (margin = interaction range, or the largest
    profiled range) and a regular-grid quadrature; irregular radii by profile pseudo-likelihood over 20 values"""
    np = _np()
    n = len(P)
    if n < 2:
        raise ValueError(f"{fname}: at least two points are needed")
    p = dict(vals)
    hk = _HARD.get(h)
    if hk is not None and p[hk] is None:
        nn = _dist_matrix(P).min(axis=1)
        p[hk] = float(nn.min()) * n / (n + 1)
    rk = _RANGE[h]
    rmax = _rmax(w)
    if p.get(rk) is None:
        lo = p.get(hk) or 0.0
        cands = [c for c in (rmax * k / 20 for k in range(1, 21)) if c > lo]
        if not cands:
            raise ValueError(f"{fname}: no interaction radius candidate exceeds the hard-core radius")
    else:
        cands = [p[rk]]
    if h in ("StraussHardcorePointProcess", "DiggleGrattonPointProcess") and p[hk] is not None and \
            p.get(rk) is not None and p[hk] > p[rk]:
        raise ValueError(f"{fname}: the hard-core radius exceeds the interaction radius")
    B = max(_range_of(h, dict(p, **{rk: c})) for c in cands)
    Q, vol = _dummy(w, B, fname)
    keep = w.bdist(P, fname) >= B
    if not keep.any() or len(Q) == 0:
        raise ValueError(f"{fname}: the window is too small for the border correction (margin {B})")
    best = None
    for c in cands:
        pc = dict(p, **{rk: c})
        res = _fit_loglinear(h, pc, P, keep, Q, vol, fname)
        if res is not None and (best is None or res[0] > best[0]):
            best = (res[0], pc, res[1])
    if best is None:
        raise ValueError(f"{fname}: the data are incompatible with every candidate interaction radius")
    out = dict(best[1])
    out.update(best[2])
    return out


def _fit_loglinear(h, p, P, keep, Q, vol, fname):
    """maximise the log pseudo-likelihood over log mu (if free) and theta (if free); returns (value, fitted dict)
    or None if a data point violates the hard core"""
    np = _np()
    idx = np.nonzero(keep)[0]
    Sd, od, ad = [], [], []
    for i in idx:
        s_, o_, a_ = _gibbs_terms(h, p, np.delete(P, i, axis=0), P[i:i + 1])
        Sd.append(s_[0])
        od.append(o_[0])
        ad.append(a_[0])
    if not all(ad):
        return None
    Sd, od = np.array(Sd), np.array(od)
    Sq, oq, aq = _gibbs_terms(h, p, P, Q)
    Sq, oq = Sq[aq], oq[aq]
    wq = np.full(len(Sq), vol)
    th = _THETA.get(h)
    free0 = p["mu"] is None
    free1 = th is not None and p[th[0]] is None
    t0 = math.log(p["mu"]) if not free0 else 0.0
    if th is not None and not free1:
        t1 = math.log(p[th[0]]) if th[1] == "log" and p[th[0]] > 0 else (p[th[0]] if th[1] == "id" else -np.inf)
    else:
        t1 = 0.0
    N = len(Sd)

    def lin(b, Sv):
        return b * Sv if math.isfinite(b) else np.where(Sv > 0, -np.inf, 0.0)

    def ll(a, b):
        with np.errstate(all="ignore"):
            return float((a + lin(b, Sd) + od).sum() - (wq * np.exp(a + lin(b, Sq) + oq)).sum())

    def closed_t0(b):
        with np.errstate(all="ignore"):
            tot = float((wq * np.exp(lin(b, Sq) + oq)).sum())
        if tot <= 0:
            raise ValueError(f"{fname}: the pseudo-likelihood is degenerate")
        return math.log(N / tot)
    if free1:
        lo_b = th[2] < 0
        if lo_b and not np.any(Sd > 0):
            t1 = -np.inf                         # no interacting pairs in the data: gamma = 0
            if free0:
                t0 = closed_t0(t1)
        else:
            a, b = (t0 if not free0 else closed_t0(0.0)), 0.0
            for _ in range(200):
                e = wq * np.exp(a + b * Sq + oq)
                g0 = N - e.sum() if free0 else 0.0
                g1 = float(Sd.sum() - (e * Sq).sum())
                H00, H01, H11 = -e.sum(), -(e * Sq).sum(), -(e * Sq * Sq).sum()
                if free0:
                    det = H00 * H11 - H01 * H01
                    if det == 0:
                        break
                    da = -(H11 * g0 - H01 * g1) / det
                    db = -(-H01 * g0 + H00 * g1) / det
                else:
                    if H11 == 0:
                        break
                    da, db = 0.0, -g1 / H11
                step, cur = 1.0, ll(a, b)
                while step > 1e-10 and ll(a + step * da, b + step * db) < cur - 1e-12:
                    step /= 2
                a, b = a + step * da, b + step * db
                if abs(da) + abs(db) < 1e-12:
                    break
            if th[2] < 0 and b > 0 or th[2] > 0 and b < 0:
                b = 0.0
                if free0:
                    a = closed_t0(0.0)
            t0, t1 = a, b
            e = wq * np.exp(t0 + t1 * Sq + oq)
            grad = abs(N - e.sum()) * free0 + (abs(float(Sd.sum() - (e * Sq).sum())) if t1 != 0 else 0.0)
            if not math.isfinite(grad) or grad > 1e-6 * max(1.0, N):
                raise ArithmeticError(f"{fname}: the pseudo-likelihood score failed verification")
    elif free0:
        t0 = closed_t0(t1)
    fitted = {}
    if free0:
        fitted["mu"] = math.exp(t0)
    if free1:
        fitted[th[0]] = (math.exp(t1) if math.isfinite(t1) else 0.0) if th[1] == "log" else t1
    for k in ("rh", "rs", "rp", "rho", "de"):
        if k in p:
            fitted.setdefault(k, p[k])
    return ll(t0, t1), fitted


_EST_NOTES = ("pdata is {p1, ...} (2D: Ripley-Rasson region), {pts, reg} or SpatialPointData (ensembles pooled for "
              "Poisson and cluster processes); pproc has symbols for the parameters to estimate (a symbolic "
              "dimension is taken from the data); {{p, p0}, ...} gives starting values (used by the minimum-contrast "
              "search). PoissonPointProcess: maximum likelihood n/|W| (exact). ThomasPointProcess, "
              "MaternPointProcess, CauchyPointProcess, VarianceGammaPointProcess (\"MethodOfMoments\", SciPy): "
              "Diggle-Gratton minimum contrast Sum (Khat^(1/4) - K^(1/4))^2 over 64 radii up to a quarter of the "
              "shortest window side (half the radius for balls), with the intensity mu lambda fixed at n/|W| when "
              "both are free, a start grid and Nelder-Mead polishing. Hardcore, Strauss, StraussHardcore, Penttinen, "
              "DiggleGates and DiggleGratton processes (\"MaximumPseudoLikelihood\"): Baddeley-Turner pseudo-"
              "likelihood with border correction and a 16384-cell grid quadrature, Newton iterations with the score "
              "self-checked, gamma restricted to [0, 1] and kappa >= 0; hard-core radii estimated by min "
              "nearest-neighbour distance n/(n + 1); interaction radii by profile pseudo-likelihood over 20 values "
              "up to the same rmax. Other processes, \"FindClusters\", estimator suboptions and non-Automatic "
              "AccuracyGoal/PrecisionGoal/WorkingPrecision raise NotImplementedError; estimators a process does not "
              "support raise NotImplementedError; wrong kinds raise TypeError")


def _est_args(args, fname):
    pos, o = split_opts(args, ("PointProcessEstimator", "AccuracyGoal", "PrecisionGoal", "WorkingPrecision"), fname)
    if len(pos) not in (2, 3):
        raise ValueError(f"{fname}: expected pdata, pproc and optionally {{{{p, p0}}, ...}}")
    return pos[0], pos[1], (pos[2] if len(pos) == 3 else None), o


@fn("EstimatedPointProcess", area=AREA, status="added",
    notes="EstimatedPointProcess[pdata, pproc] and EstimatedPointProcess[pdata, pproc, {{p, p0}, ...}]: pproc with "
          "the estimated values inserted. " + _EST_NOTES + ". Check: self-check (pseudo-likelihood score) and "
          "statistical (seeded simulations recover the parameters within tolerance) in tests.")
def _epp(pdata: Expr, pproc: Expr, *rest: Expr) -> Expr:
    fname = "EstimatedPointProcess"
    pdata, pproc, spec, o = _est_args([pdata, pproc] + list(rest), fname)
    h, names, fitted, syms, d = estimate(pdata, pproc, spec, o, fname)
    args = []
    for nm, v in zip(names, pproc.args[:-1]):
        args.append(fitted.get(nm, v) if nm in syms else v)
    return Expr(h, *args, d)


@fn("FindPointProcessParameters", area=AREA, status="added",
    notes="FindPointProcessParameters[pdata, pproc] and FindPointProcessParameters[pdata, pproc, {{p, p0}, ...}]: "
          "the rules {p -> value, ...} for the symbolic parameters of pproc (a symbolic dimension gets the data "
          "dimension). " + _EST_NOTES + ". Check: self-check (pseudo-likelihood score) and statistical (seeded "
          "simulations recover the parameters within tolerance; Poisson n/|W| exact) in tests.")
def _fppp(pdata: Expr, pproc: Expr, *rest: Expr) -> ListOf[Rule]:
    fname = "FindPointProcessParameters"
    pdata, pproc, spec, o = _est_args([pdata, pproc] + list(rest), fname)
    h, names, fitted, syms, d = estimate(pdata, pproc, spec, o, fname)
    out = [rule(syms[nm], fitted[nm]) for nm in names if nm in syms]
    if isinstance(pproc.args[-1], Symbol):
        out.append(rule(pproc.args[-1], d))
    return L(*out)


# ─────────────────────────── goodness-of-fit tests ───────────────────────────

TESTS = ("BesagL", "ChiSquare")
NSIM = 99


def _chisq(P, w, fname):
    np = _np()
    import mpmath
    if w.kind != "box":
        raise NotImplementedError(f"{fname}: the \"ChiSquare\" quadrat test needs a box (Rectangle/Cuboid/Interval) "
                                  "window")
    n, d = P.shape
    if n < 2:
        raise ValueError(f"{fname}: at least two points are needed")
    k = max(2, int((n / 5) ** (1.0 / d)))
    idx = np.clip(((P - w.lo) / (w.hi - w.lo) * k).astype(int), 0, k - 1)
    flat = np.ravel_multi_index(idx.T, (k,) * d)
    O = np.bincount(flat, minlength=k ** d).astype(float)
    E = n / k ** d
    X2 = float(((O - E) ** 2 / E).sum())
    df = k ** d - 1
    p = float(mpmath.gammainc(df / 2.0, X2 / 2, mpmath.inf, regularized=True))
    return X2, p, df


def _mad_test(P, w, sim, fname):
    """Monte Carlo maximum-absolute-deviation test on Besag's L over 32 radii in (0, rmax]"""
    np = _np()
    rs = _rmax(w) * np.arange(1, 33) / 32
    corr = "Ripley" if w.dim == 2 else "Translation"

    def Lh(X):
        if len(X) < 2:
            return np.zeros(len(rs))
        return (k_estimate(X, w, rs, corr, fname) / _unit_ball_f(w.dim)) ** (1.0 / w.dim)
    obs = Lh(P)
    sims = [Lh(sim()) for _ in range(NSIM)]
    if sim.theory is not None:
        ref = sim.theory(rs)
    else:
        ref = np.mean(sims, axis=0)
    T = float(np.max(np.abs(obs - ref)))
    Ts = np.array([np.max(np.abs(s_ - ref)) for s_ in sims])
    p = (1 + int((Ts >= T).sum())) / (NSIM + 1)
    return T, p


class _Sim:
    def __init__(self, f, theory=None):
        self.f, self.theory = f, theory

    def __call__(self):
        return self.f()


def _test_args(args, fname, nproc):
    from .wl_statmodels import TEST_PROPS
    pos, o = split_opts(args, ("SignificanceLevel", "Method"), fname)
    data = pos[0]
    proc = None
    rest = pos[1:]
    if nproc and rest and isinstance(rest[0], Expr) and not isinstance(rest[0], str) and not is_list(rest[0]):
        proc, rest = rest[0], rest[1:]
    if len(rest) > 1:
        raise ValueError(f"{fname}: too many arguments")
    spec = rest[0] if rest else None
    prop, tests = None, None
    if spec is None:
        pass
    elif spec == Symbol("All"):
        tests = list(TESTS)
    elif isinstance(spec, str) and spec in TESTS:
        tests = [spec]
    elif isinstance(spec, str):
        prop = spec
    elif is_list(spec) and len(spec.args) == 2 and isinstance(spec.args[0], str):
        prop = spec.args[0]
        t = spec.args[1]
        tests = list(TESTS) if t == Symbol("All") else [t]
    elif is_list(spec) and all(isinstance(t, str) and t in TESTS for t in spec.args):
        tests = list(spec.args)
    else:
        raise TypeError(f"{fname}: invalid test or property specification {spec}")
    m = o.get("Method")
    if m is not None and not is_auto(m):
        if m not in TESTS:
            raise ValueError(f"{fname}: unknown test {m} (use \"BesagL\" or \"ChiSquare\")")
        tests = [m]
    for t in tests or []:
        if t not in TESTS:
            if t == "ModifiedChiSquare":
                raise NotImplementedError(f"{fname}: the \"ModifiedChiSquare\" test is not supported")
            raise ValueError(f"{fname}: unknown test {t}")
    if prop is not None and prop not in TEST_PROPS + ("AllTests", "AutomaticTest", "HypothesisTestData",
                                                       "TestConclusion") and not prop.endswith("Table"):
        if prop == "ModifiedChiSquare":
            raise NotImplementedError(f"{fname}: the \"ModifiedChiSquare\" test is not supported")
        raise ValueError(f"{fname}: unknown test or property {prop!r}")
    return data, proc, tests, prop, o


def run_test(fname, data, proc, tests, prop, o):
    from .wl_statmodels import test_result
    np = _np()
    sd, w = spd_of(data, fname)
    if sd.ens:
        raise NotImplementedError(f"{fname}: ensembles of configurations are not supported")
    P = config_array(sd.configs[0], w, fname)
    n = len(P)
    poisson = proc is None
    if proc is not None:
        pr = PS.pproc_of(proc, fname)
        if not pr.numeric():
            raise NotImplementedError(f"{fname}: the null process must have numeric parameters (estimate them first "
                                      "with EstimatedPointProcess)")
        if pr.d != w.dim:
            raise ValueError(f"{fname}: the process dimension {pr.d} differs from the data dimension {w.dim}")
        poisson = isinstance(pr, PS.PoissonPP)
    auto = "ChiSquare" if poisson and w.kind == "box" else "BesagL"
    if prop == "AutomaticTest":
        return auto
    alltests = ["BesagL", "ChiSquare"] if poisson and w.kind == "box" else ["BesagL"]
    if prop == "AllTests":
        return L(*alltests)
    tests = tests or [auto]
    rng = PS._np_rng()
    res = []
    for t in tests:
        if t == "ChiSquare":
            if not poisson:
                raise NotImplementedError(f"{fname}: the \"ChiSquare\" test is only available for a Poisson null")
            X2, p, df = _chisq(P, w, fname)
            res.append((R(X2), p, df))
        else:
            if poisson:
                sim = _Sim(lambda: w.uniform(rng, n), lambda rs: rs)
            else:
                rg = PS.region_of(sd.reg, fname)
                sim = _Sim(lambda: pr.sample(rng, rg))
            T, p = _mad_test(P, w, sim, fname)
            res.append((R(T), p, None))
    if len(res) == 1:
        stat, p, df = res[0]
        return test_result(fname, stat, p, prop, o, df)
    if prop is None or prop == "PValue":
        return L(*(float(p) for _, p, _ in res))
    return L(*(test_result(fname, s_, p, prop, o, df) for s_, p, df in res))


_TEST_NOTES = ("pdata is {p1, ...} (2D: Ripley-Rasson region), {pts, reg} or SpatialPointData. Tests: "
               "\"ChiSquare\" (quadrat counts on a k^d grid of the box window, k = Max[2, Floor[(n/5)^(1/d)]], "
               "Pearson X^2 with k^d - 1 degrees of freedom; Poisson null and box windows only) and \"BesagL\" "
               "(Monte Carlo maximum-absolute-deviation test of Besag's L over 32 radii up to a quarter of the "
               "shortest side, 99 seeded simulations of the null, p = (1 + #{T_sim >= T})/100; the reference is "
               "L(r) = r for the Poisson null and the mean of the simulations otherwise). Automatic is \"ChiSquare\" "
               "for a Poisson null in a box, else \"BesagL\". Properties: \"PValue\" (default), \"TestStatistic\", "
               "\"TestData\", \"DegreesOfFreedom\" (ChiSquare), \"ShortTestConclusion\" (SignificanceLevel, default "
               "0.05), \"AllTests\", \"AutomaticTest\", or {prop, test}; \"ModifiedChiSquare\", \"HypothesisTestData\" "
               "and formatted tables raise NotImplementedError, unknown names ValueError")


@fn("SpatialRandomnessTest", area=AREA, status="added",
    notes="SpatialRandomnessTest[pdata], SpatialRandomnessTest[pdata, \"test\" | All | \"prop\" | {\"prop\", "
          "test}]: tests complete spatial randomness (homogeneous Poisson, conditioned on n). " + _TEST_NOTES +
          ". Check: reference (chi-square p-value by hand for quadrat counts) and statistical (seeded CSR data not "
          "rejected, clustered and regular data rejected) in tests.")
def _srt(pdata: Expr, *rest: Expr) -> Expr:
    fname = "SpatialRandomnessTest"
    data, proc, tests, prop, o = _test_args([pdata] + list(rest), fname, False)
    return run_test(fname, data, None, tests, prop, o)


@fn("PointProcessFitTest", area=AREA, status="added",
    notes="PointProcessFitTest[pdata], PointProcessFitTest[pdata, pproc], PointProcessFitTest[pdata, pproc, "
          "\"prop\"] (Method -> \"BesagL\" | \"ChiSquare\" selects the test): goodness of fit of pdata to pproc "
          "(default PoissonPointProcess with the estimated intensity); pproc must have numeric parameters "
          "(symbolic parameters raise NotImplementedError: estimate them first), and non-Poisson processes are "
          "simulated in the data's window (Rectangle, Cuboid, Disk or Ball). " + _TEST_NOTES +
          ". Check: statistical (seeded data from the null process not rejected, data from a different process "
          "rejected) in tests.")
def _ppft(pdata: Expr, *rest: Expr) -> Expr:
    fname = "PointProcessFitTest"
    data, proc, tests, prop, o = _test_args([pdata] + list(rest), fname, True)
    return run_test(fname, data, proc, tests, prop, o)
