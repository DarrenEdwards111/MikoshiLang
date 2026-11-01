"""Wolfram batch W25: random processes (parametric, time-series, Markov, queueing and SDE processes).

Process objects are inert expressions (``WienerProcess[mu, sigma]``, ``ARProcess[{a1, ...}, v]``, ...) whose
parameters are validated by their constructors.  A process slice ``proc[t]`` behaves like a distribution: Mean,
Variance, StandardDeviation, PDF, CDF, ... of ``proc[t]`` go through ``SliceDistribution[proc, t]`` (which gives
the named distribution where one exists), and ``CovarianceFunction``, ``CorrelationFunction``,
``AbsoluteCorrelationFunction`` and ``PartialCorrelationFunction`` use closed forms.  ``RandomFunction`` simulates
paths with exact transition laws where they exist (Euler-Maruyama only for general Ito/Stratonovich processes) and
is reproducible with SeedRandom (the random stream differs from Wolfram's).  Results are ``TemporalData`` objects
(see wl_processes_ts).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, L, R, S, sym, is_list, has_float, numeric_out, rule
from . import wl_stats as W

AREA = "random processes (W25)"
INF = Symbol("Infinity")


# ─────────────────────────── helpers ───────────────────────────

def _ev(e):
    from ..evaluate import evaluate
    return evaluate(e)


def _num(v):
    return isinstance(v, sp.Basic) and v.is_number


def _inexact(*vs):
    return any(has_float(v) for v in vs)


def _p(v, fname, what="parameter"):
    """SymPy value of a scalar process parameter."""
    if isinstance(v, str):
        raise TypeError(f"{fname}: {what} must be a number or symbol, not a string")
    if is_list(v):
        raise TypeError(f"{fname}: {what} must be a scalar, not a list")
    if isinstance(v, bool):
        raise TypeError(f"{fname}: {what} must be a number, not {v}")
    return sym(v, fname)


def _plist(v, fname, what="coefficient list"):
    if not is_list(v):
        if isinstance(v, str):
            raise TypeError(f"{fname}: {what} must be a list")
        raise TypeError(f"{fname}: {what} must be a list, got {v}")
    out = []
    for x in v.args:
        if is_list(x):
            raise NotImplementedError(f"{fname}: vector (matrix-coefficient) processes are not supported")
        out.append(_p(x, fname, what[:-5] if what.endswith(" list") else what))
    return out


_CONDS = []          # parameter conditions recorded while parsing (ProcessParameterAssumptions)
_LAX = [False]       # True: record conditions without raising (ProcessParameterQ / ProcessParameterAssumptions)


def _cond(c):
    _CONDS.append(c)


def _bad(msg):
    if not _LAX[0]:
        raise ValueError(msg)


def _pos(v, fname, what):
    _cond(Expr("Greater", R(v), 0))
    if _num(v) and not (v.is_real and v > 0):
        _bad(f"{fname}: {what} must be a positive real number")


def _nonzero_real(v, fname, what):
    _cond(Expr("And", Expr("Element", R(v), Symbol("Reals")), Expr("Unequal", R(v), 0)))
    if _num(v) and not (v.is_real and v != 0):
        _bad(f"{fname}: {what} must be a nonzero real number")


def _real(v, fname, what):
    _cond(Expr("Element", R(v), Symbol("Reals")))
    if _num(v) and v.is_real is False:
        _bad(f"{fname}: {what} must be real")


def _prob(v, fname, what="p", open_=False):
    _cond(Expr("Less" if open_ else "LessEqual", 0, R(v), 1))
    if _num(v):
        if not v.is_real or v < 0 or v > 1 or (open_ and v in (0, 1)):
            _bad(f"{fname}: {what} must be {'strictly ' if open_ else ''}between 0 and 1")


def _f(v):
    return float(sp.N(v))


def _numeric_all(vals):
    return all(_num(v) for v in vals)


def _out(v, *inputs):
    return numeric_out(v, *inputs)


def _dist(head, *params):
    return Expr(head, *(R(p) if isinstance(p, sp.Basic) or isinstance(p, (int, float)) else p for p in params))


def _np_rng():
    import numpy as np
    return np.random.default_rng(W._rng().getrandbits(63))


def _nonneg_time(t, fname, t0=0):
    if _num(t):
        if not t.is_real:
            raise ValueError(f"{fname}: the time {t} must be real")
        if t < t0:
            raise ValueError(f"{fname}: the time {t} must be at least {t0}")


def _int_time(t, fname, t0=None):
    if _num(t):
        if not t.is_Integer:
            raise ValueError(f"{fname}: discrete-time processes are defined at integer times, not {t}")
        if t0 is not None and t < t0:
            raise ValueError(f"{fname}: the time {t} must be at least {t0}")


# ─────────────────────────── process classes ───────────────────────────

class Proc:
    """A parametric random process.  ``p`` holds SymPy parameters.  Subclasses implement whatever applies:

    slice(t) -> distribution Expr; mean(t), var(t), cov(s, t) -> SymPy; stationary() -> bool;
    path(rng, times) -> list of float values (path processes); jumps(rng, t0, t1) -> [(t, v), ...] (jump processes);
    discrete-time processes set dtime = True.
    """
    dtime = False
    jump = False
    gaussian = False
    t0 = sp.Integer(0)
    has_cov = True

    def __init__(self, e, fname):
        self.e, self.fname = e, fname
        self.name = e.head
        self.parse(list(e.args), fname)

    def parse(self, a, fname):
        raise NotImplementedError

    def check_time(self, t, fname):
        if self.dtime:
            _int_time(t, fname)
        else:
            _nonneg_time(t, fname, self.t0)

    def numeric(self):
        return _numeric_all(self.flat())

    def flat(self):
        out = []
        for v in self.p:
            out.extend(v if isinstance(v, list) else [v])
        return out

    def slice(self, t):
        raise NotImplementedError(f"SliceDistribution: the slice distribution of {self.name} is not available in "
                                  "closed form")

    def mean(self, t):
        return sp.sympify(S(_ev(Expr("Mean", self.slice(t)))))

    def var(self, t):
        return sp.sympify(S(_ev(Expr("Variance", self.slice(t)))))

    def cov(self, s, t):
        raise NotImplementedError(f"CovarianceFunction: not available for {self.name}")

    def stationary(self):
        return False

    def describe(self):
        return self.name


def _min(s, t):
    return sp.Min(s, t)


class Wiener(Proc):
    gaussian = True

    def parse(self, a, fname):
        if len(a) == 0:
            a = [0, 1]
        if len(a) != 2:
            raise ValueError(f"{fname}: WienerProcess expects [] or [mu, sigma]")
        self.p = [_p(a[0], fname, "drift"), _p(a[1], fname, "volatility")]
        _real(self.p[0], fname, "the drift")
        _pos(self.p[1], fname, "the volatility")

    def slice(self, t):
        m, s = self.p
        return _dist("NormalDistribution", m * t, s * sp.sqrt(t))

    def mean(self, t):
        return self.p[0] * t

    def var(self, t):
        return self.p[1] ** 2 * t

    def cov(self, s, t):
        return self.p[1] ** 2 * _min(s, t)

    def path(self, rng, times):
        m, s = map(_f, self.p)
        out, x, prev = [], None, None
        for t in times:
            if x is None:
                x = rng.normal(m * t, s * math.sqrt(t)) if t > 0 else 0.0
            else:
                d = t - prev
                x += rng.normal(m * d, s * math.sqrt(d))
            out.append(x)
            prev = t
        return out


class BrownianBridge(Proc):
    gaussian = True

    def parse(self, a, fname):
        if len(a) == 0:
            a = [1, L(0, 0), L(1, 0)]
        elif len(a) == 2:
            a = [1] + a
        if len(a) != 3 or not (is_list(a[1]) and is_list(a[2]) and len(a[1].args) == 2 and len(a[2].args) == 2):
            raise ValueError(f"{fname}: BrownianBridgeProcess expects [], [{{t1, a}}, {{t2, b}}] or "
                             "[sigma, {t1, a}, {t2, b}]")
        sg = _p(a[0], fname, "volatility")
        t1, x1 = (_p(v, fname) for v in a[1].args)
        t2, x2 = (_p(v, fname) for v in a[2].args)
        _pos(sg, fname, "the volatility")
        if _num(t1) and _num(t2) and not t1 < t2:
            raise ValueError(f"{fname}: need t1 < t2")
        self.p = [sg, t1, x1, t2, x2]
        self.t0 = t1

    def check_time(self, t, fname):
        sg, t1, x1, t2, x2 = self.p
        if _num(t) and _num(t1) and _num(t2) and not (t1 <= t <= t2):
            raise ValueError(f"{fname}: the time {t} must lie in [{t1}, {t2}]")

    def mean(self, t):
        sg, t1, x1, t2, x2 = self.p
        return x1 + (x2 - x1) * (t - t1) / (t2 - t1)

    def var(self, t):
        sg, t1, x1, t2, x2 = self.p
        return sg ** 2 * (t - t1) * (t2 - t) / (t2 - t1)

    def slice(self, t):
        sg, t1, x1, t2, x2 = self.p
        if _num(t) and (t == t1 or t == t2):
            raise NotImplementedError(f"SliceDistribution: the slice of {self.name} at an end point is a point mass, "
                                      "which is not supported as a distribution")
        return _dist("NormalDistribution", sp.simplify(self.mean(t)), sp.sqrt(sp.factor(self.var(t))))

    def cov(self, s, t):
        sg, t1, x1, t2, x2 = self.p
        return sg ** 2 * (sp.Min(s, t) - t1) * (t2 - sp.Max(s, t)) / (t2 - t1)

    def path(self, rng, times):
        sg, t1, x1, t2, x2 = map(_f, self.p)
        out, x, u = [], x1, t1
        for t in times:
            if t >= t2:
                x = x2
            elif t > u:
                m = x + (x2 - x) * (t - u) / (t2 - u)
                v = sg ** 2 * (t - u) * (t2 - t) / (t2 - u)
                x = rng.normal(m, math.sqrt(v))
            out.append(x)
            u = t
        return out


class GBM(Proc):
    def parse(self, a, fname):
        if len(a) != 3:
            raise ValueError(f"{fname}: GeometricBrownianMotionProcess expects [mu, sigma, x0]")
        self.p = [_p(a[0], fname, "drift"), _p(a[1], fname, "volatility"), _p(a[2], fname, "initial value")]
        _real(self.p[0], fname, "the drift")
        _pos(self.p[1], fname, "the volatility")
        _pos(self.p[2], fname, "the initial value")

    def slice(self, t):
        m, s, x0 = self.p
        return _dist("LogNormalDistribution", sp.log(x0) + (m - s ** 2 / 2) * t, s * sp.sqrt(t))

    def mean(self, t):
        m, s, x0 = self.p
        return x0 * sp.exp(m * t)

    def var(self, t):
        m, s, x0 = self.p
        return x0 ** 2 * sp.exp(2 * m * t) * (sp.exp(s ** 2 * t) - 1)

    def cov(self, s_, t):
        m, s, x0 = self.p
        return x0 ** 2 * sp.exp(m * (s_ + t)) * (sp.exp(s ** 2 * sp.Min(s_, t)) - 1)

    def path(self, rng, times):
        m, s, x0 = map(_f, self.p)
        w = Wiener.__new__(Wiener)
        w.p = [sp.Float(m - s * s / 2), sp.Float(s)]
        return [x0 * math.exp(v) for v in Wiener.path(w, rng, times)]


class OU(Proc):
    gaussian = True

    def parse(self, a, fname):
        if len(a) not in (3, 4):
            raise ValueError(f"{fname}: OrnsteinUhlenbeckProcess expects [mu, sigma, theta] or [mu, sigma, theta, x0]")
        self.p = [_p(v, fname) for v in a]
        _real(self.p[0], fname, "mu")
        _pos(self.p[1], fname, "sigma")
        _pos(self.p[2], fname, "theta")
        if len(a) == 4:
            _real(self.p[3], fname, "x0")
        self.fixed = len(a) == 4

    def mean(self, t):
        m, s, th = self.p[:3]
        return self.p[3] * sp.exp(-th * t) + m * (1 - sp.exp(-th * t)) if self.fixed else m

    def var(self, t):
        m, s, th = self.p[:3]
        return s ** 2 / (2 * th) * (1 - sp.exp(-2 * th * t)) if self.fixed else s ** 2 / (2 * th)

    def slice(self, t):
        m, s, th = self.p[:3]
        if self.fixed:
            if _num(t) and t == 0:
                raise NotImplementedError("SliceDistribution: the slice at time 0 is a point mass, which is not "
                                          "supported as a distribution")
            return _dist("NormalDistribution", self.mean(t), s * sp.sqrt((1 - sp.exp(-2 * th * t)) / (2 * th)))
        return _dist("NormalDistribution", m, s / sp.sqrt(2 * th))

    def cov(self, s_, t):
        m, s, th = self.p[:3]
        c = s ** 2 / (2 * th) * sp.exp(-th * sp.Abs(t - s_))
        return c - s ** 2 / (2 * th) * sp.exp(-th * (s_ + t)) if self.fixed else c

    def stationary(self):
        return not self.fixed

    def path(self, rng, times):
        m, s, th = map(_f, self.p[:3])
        sd = s / math.sqrt(2 * th)
        out, x, prev = [], None, None
        for t in times:
            if x is None:
                if self.fixed:
                    x0 = _f(self.p[3])
                    x = x0 if t == 0 else rng.normal(m + (x0 - m) * math.exp(-th * t),
                                                     sd * math.sqrt(1 - math.exp(-2 * th * t)))
                else:
                    x = rng.normal(m, sd)
            else:
                e = math.exp(-th * (t - prev))
                x = rng.normal(m + (x - m) * e, sd * math.sqrt(1 - e * e))
            out.append(x)
            prev = t
        return out


class CIR(Proc):
    def parse(self, a, fname):
        if len(a) != 4:
            raise ValueError(f"{fname}: CoxIngersollRossProcess expects [mu, sigma, theta, x0]")
        self.p = [_p(v, fname) for v in a]
        m, s, th, x0 = self.p
        _nonzero_real(m, fname, "mu")
        _nonzero_real(s, fname, "sigma")
        _nonzero_real(th, fname, "theta")
        _pos(x0, fname, "x0")
        if _num(m) and _num(th) and (m > 0) != (th > 0):
            raise ValueError(f"{fname}: mu and theta must have the same sign")

    def mean(self, t):
        m, s, th, x0 = self.p
        return m + (x0 - m) * sp.exp(-th * t)

    def var(self, t):
        m, s, th, x0 = self.p
        e = sp.exp(-th * t)
        return x0 * s ** 2 / th * (e - e ** 2) + m * s ** 2 / (2 * th) * (1 - e) ** 2

    def slice(self, t):
        m, s, th, x0 = self.p
        if _num(t) and t == 0:
            raise NotImplementedError("SliceDistribution: the slice at time 0 is a point mass, which is not "
                                      "supported as a distribution")
        e = sp.exp(-th * t)
        c = s ** 2 * (1 - e) / (4 * th)
        k = 4 * th * m / s ** 2
        lam = 4 * th * x0 * e / (s ** 2 * (1 - e))
        x = Symbol("x")
        return Expr("TransformedDistribution", Expr("Times", R(c), x),
                    Expr("Distributed", x, _dist("NoncentralChiSquareDistribution", k, lam)))

    def cov(self, s_, t):
        m, s, th, x0 = self.p
        return sp.exp(-th * sp.Abs(t - s_)) * self.var(sp.Min(s_, t))

    def path(self, rng, times):
        m, s, th, x0 = map(_f, self.p)
        out, x, prev = [], x0, 0.0
        k = 4 * th * m / (s * s)
        for t in times:
            if t > prev:
                e = math.exp(-th * (t - prev))
                c = s * s * (1 - e) / (4 * th)
                x = c * rng.noncentral_chisquare(k, x * e / c)
            out.append(x)
            prev = t
        return out


class Poisson(Proc):
    jump = True

    def parse(self, a, fname):
        if len(a) != 1:
            raise ValueError(f"{fname}: PoissonProcess expects [mu]")
        self.p = [_p(a[0], fname, "rate")]
        _pos(self.p[0], fname, "the rate")

    def slice(self, t):
        return _dist("PoissonDistribution", self.p[0] * t)

    def mean(self, t):
        return self.p[0] * t

    def var(self, t):
        return self.p[0] * t

    def cov(self, s, t):
        return self.p[0] * _min(s, t)

    def start(self, rng, t0):
        lam = _f(self.p[0])
        return int(rng.poisson(lam * t0)) if t0 > 0 else 0

    def jumps(self, rng, t0, t1):
        lam = _f(self.p[0])
        x = self.start(rng, t0)
        out, t = [(t0, x)], t0
        while True:
            t += rng.exponential(1 / lam)
            if t > t1:
                break
            x += 1
            out.append((t, x))
        return out

    def path(self, rng, times):
        lam = _f(self.p[0])
        out, x, prev = [], None, None
        for t in times:
            x = (int(rng.poisson(lam * t)) if t > 0 else 0) if x is None else x + int(rng.poisson(lam * (t - prev)))
            out.append(x)
            prev = t
        return out


class Telegraph(Proc):
    jump = True

    def parse(self, a, fname):
        if len(a) != 1:
            raise ValueError(f"{fname}: TelegraphProcess expects [mu]")
        self.p = [_p(a[0], fname, "rate")]
        _pos(self.p[0], fname, "the rate")

    def mean(self, t):
        return sp.exp(-2 * self.p[0] * t)

    def var(self, t):
        return 1 - sp.exp(-4 * self.p[0] * t)

    def slice(self, t):
        x = Symbol("x")
        return Expr("TransformedDistribution", Expr("Plus", -1, Expr("Times", 2, x)),
                    Expr("Distributed", x, _dist("BernoulliDistribution", (1 + sp.exp(-2 * self.p[0] * t)) / 2)))

    def cov(self, s, t):
        mu = self.p[0]
        return sp.exp(-2 * mu * sp.Abs(t - s)) - sp.exp(-2 * mu * (s + t))

    def jumps(self, rng, t0, t1):
        lam = _f(self.p[0])
        n0 = int(rng.poisson(lam * t0)) if t0 > 0 else 0
        x = 1 if n0 % 2 == 0 else -1
        out, t = [(t0, x)], t0
        while True:
            t += rng.exponential(1 / lam)
            if t > t1:
                break
            x = -x
            out.append((t, x))
        return out

    def path(self, rng, times):
        lam = _f(self.p[0])
        out, x, prev = [], None, None
        for t in times:
            n = int(rng.poisson(lam * (t if x is None else t - prev)))
            x = (1 if x is None else x) * (1 if n % 2 == 0 else -1)
            out.append(x)
            prev = t
        return out


class CompoundPoisson(Poisson):
    def parse(self, a, fname):
        if len(a) != 2:
            raise ValueError(f"{fname}: CompoundPoissonProcess expects [lambda, jdist]")
        self.p = [_p(a[0], fname, "rate")]
        _pos(self.p[0], fname, "the rate")
        self.jd = a[1]
        if not W.is_dist(a[1]):
            raise TypeError(f"{fname}: the jump size distribution {a[1]} is not a distribution")
        self.jdd = W.dist_of(a[1], fname)

    def slice(self, t):
        return Expr("CompoundPoissonDistribution", R(self.p[0] * t), self.jd)

    def mean(self, t):
        return self.p[0] * t * self.jdd.mean()

    def var(self, t):
        return self.p[0] * t * W.raw_moment(self.jdd, 2)

    def cov(self, s, t):
        return self.p[0] * _min(s, t) * W.raw_moment(self.jdd, 2)

    def numeric(self):
        return _num(self.p[0]) and self.jdd.numeric()

    def _jump(self):
        v = self.jdd.sample(W._rng())
        return float(v) if not isinstance(v, (int, float)) else v

    def jumps(self, rng, t0, t1):
        lam = _f(self.p[0])
        x = sum(self._jump() for _ in range(int(rng.poisson(lam * t0)))) if t0 > 0 else 0
        out, t = [(t0, x)], t0
        while True:
            t += rng.exponential(1 / lam)
            if t > t1:
                break
            x = x + self._jump()
            out.append((t, x))
        return out

    def path(self, rng, times):
        lam = _f(self.p[0])
        out, x, prev = [], 0, 0
        for t in times:
            x = x + sum(self._jump() for _ in range(int(rng.poisson(lam * (t - prev)))))
            out.append(x)
            prev = t
        return out


class Renewal(Proc):
    jump = True
    has_cov = False

    def parse(self, a, fname):
        if len(a) != 1 or not W.is_dist(a[0]):
            raise ValueError(f"{fname}: RenewalProcess expects [rdist] with a distribution of positive inter-event "
                             "times")
        self.rd = a[0]
        self.rdd = W.dist_of(a[0], fname)
        self.p = []
        lo = self.rdd.bounds()[0]
        if _num(lo) and lo < 0:
            raise ValueError(f"{fname}: the renewal time distribution must have non-negative support")

    def numeric(self):
        return self.rdd.numeric()

    def _gap(self):
        v = self.rdd.sample(W._rng())
        return float(v)

    def _mark(self):
        return 1

    def jumps(self, rng, t0, t1):
        t, x, out = 0.0, 0, None
        while True:
            g = self._gap()
            if t + g > t1:
                break
            t += g
            if t > t0 and out is None:
                out = [(t0, x)]
            x = x + self._mark()
            if out is not None:
                out.append((t, x))
        return out if out is not None else [(t0, x)]

    def path(self, rng, times):
        ev = self.jumps(rng, 0.0, times[-1])
        out, k = [], 0
        for t in times:
            while k + 1 < len(ev) and ev[k + 1][0] <= t:
                k += 1
            out.append(ev[k][1])
        return out


class CompoundRenewal(Renewal):
    def parse(self, a, fname):
        if len(a) != 2 or not W.is_dist(a[0]) or not W.is_dist(a[1]):
            raise ValueError(f"{fname}: CompoundRenewalProcess expects [rdist, jdist]")
        Renewal.parse(self, a[:1], fname)
        self.jd, self.jdd = a[1], W.dist_of(a[1], fname)

    def numeric(self):
        return self.rdd.numeric() and self.jdd.numeric()

    def _mark(self):
        v = self.jdd.sample(W._rng())
        return float(v) if not isinstance(v, (int, float)) else v


class InhomPoisson(Proc):
    jump = True

    def parse(self, a, fname):
        if len(a) != 2 or not isinstance(a[1], Symbol):
            raise ValueError(f"{fname}: InhomogeneousPoissonProcess expects [lambda[t], t]")
        self.var_ = sp.Symbol(a[1].name)
        self.lam = _p(a[0], fname, "intensity")
        self.p = [self.lam]

    def numeric(self):
        return self.lam.free_symbols <= {self.var_}

    def big(self, t):
        u = sp.Dummy("u")
        r = sp.integrate(self.lam.subs(self.var_, u), (u, 0, t))
        if r.has(sp.Integral):
            raise NotImplementedError(f"{self.name}: the integrated intensity has no closed form")
        return sp.simplify(r)

    def slice(self, t):
        return _dist("PoissonDistribution", self.big(t))

    def mean(self, t):
        return self.big(t)

    def var(self, t):
        return self.big(t)

    def cov(self, s, t):
        return self.big(sp.Min(s, t))

    def _bigf(self):
        import mpmath
        f = sp.lambdify(self.var_, self.lam, "mpmath")
        return lambda t: float(mpmath.quad(f, [0, t])) if t > 0 else 0.0

    def _check_nonneg(self, t1):
        import mpmath
        f = sp.lambdify(self.var_, self.lam, "mpmath")
        for k in range(201):
            if float(mpmath.re(f(mpmath.mpf(t1) * k / 200))) < 0:
                raise ValueError(f"{self.name}: the intensity must be non-negative")

    def jumps(self, rng, t0, t1):
        self._check_nonneg(t1)
        Lf = self._bigf()
        L0, L1 = Lf(t0), Lf(t1)
        x = int(rng.poisson(L0)) if L0 > 0 else 0
        out, s = [(t0, x)], L0
        while True:
            s += rng.exponential(1.0)
            if s > L1:
                break
            lo, hi = t0, t1                       # invert the (monotone) integrated intensity by bisection
            for _ in range(80):
                mid = (lo + hi) / 2
                if Lf(mid) < s:
                    lo = mid
                else:
                    hi = mid
            x += 1
            out.append(((lo + hi) / 2, x))
        return out

    def path(self, rng, times):
        self._check_nonneg(times[-1])
        Lf = self._bigf()
        out, x, prev = [], 0, 0.0
        for t in times:
            x += int(rng.poisson(Lf(t) - Lf(prev)))
            out.append(x)
            prev = t
        return out


# discrete-time processes

class Bernoulli(Proc):
    dtime = True

    def parse(self, a, fname):
        if len(a) != 1:
            raise ValueError(f"{fname}: BernoulliProcess expects [p]")
        self.p = [_p(a[0], fname, "p")]
        _prob(self.p[0], fname)

    def slice(self, t):
        return _dist("BernoulliDistribution", self.p[0])

    def mean(self, t):
        return self.p[0]

    def var(self, t):
        return self.p[0] * (1 - self.p[0])

    def cov(self, s, t):
        return sp.Piecewise((self.p[0] * (1 - self.p[0]), sp.Eq(s, t)), (0, True))

    def stationary(self):
        return True

    def path(self, rng, times):
        p = _f(self.p[0])
        return [int(rng.random() < p) for _ in times]


class Binomial(Proc):
    dtime = True

    def parse(self, a, fname):
        if len(a) != 1:
            raise ValueError(f"{fname}: BinomialProcess expects [p]")
        self.p = [_p(a[0], fname, "p")]
        _prob(self.p[0], fname)

    def check_time(self, t, fname):
        _int_time(t, fname, 0)

    def slice(self, t):
        if _num(t) and t == 0:
            raise NotImplementedError("SliceDistribution: the slice at time 0 is a point mass, which is not "
                                      "supported as a distribution")
        return _dist("BinomialDistribution", t, self.p[0])

    def mean(self, t):
        return t * self.p[0]

    def var(self, t):
        return t * self.p[0] * (1 - self.p[0])

    def cov(self, s, t):
        return self.p[0] * (1 - self.p[0]) * _min(s, t)

    def path(self, rng, times):
        p = _f(self.p[0])
        out, x, prev = [], None, None
        for t in times:
            x = int(rng.binomial(t, p)) if x is None else x + int(rng.binomial(t - prev, p))
            out.append(x)
            prev = t
        return out


class RandomWalk(Proc):
    dtime = True

    def parse(self, a, fname):
        if len(a) != 1:
            raise ValueError(f"{fname}: RandomWalkProcess expects [p] or [{{p, q}}]")
        if is_list(a[0]):
            if len(a[0].args) != 2:
                raise ValueError(f"{fname}: RandomWalkProcess[{{p, q}}] needs two probabilities")
            p, q = (_p(v, fname) for v in a[0].args)
            _prob(p, fname, "p")
            _prob(q, fname, "q")
            if _num(p) and _num(q) and p + q > 1:
                raise ValueError(f"{fname}: p + q must not exceed 1")
            self.p, self.lazy = [p, q], True
        else:
            p = _p(a[0], fname, "p")
            _prob(p, fname)
            self.p, self.lazy = [p, 1 - p], False

    def check_time(self, t, fname):
        _int_time(t, fname, 0)

    def mean(self, t):
        p, q = self.p
        return t * (p - q)

    def var(self, t):
        p, q = self.p
        return sp.expand(t * (p + q - (p - q) ** 2))

    def slice(self, t):
        if self.lazy:
            raise NotImplementedError("SliceDistribution: the slice of a lazy RandomWalkProcess[{p, q}] is not "
                                      "available in closed form")
        if _num(t) and t == 0:
            raise NotImplementedError("SliceDistribution: the slice at time 0 is a point mass, which is not "
                                      "supported as a distribution")
        x = Symbol("x")
        return Expr("TransformedDistribution", Expr("Plus", Expr("Times", 2, x), R(-t)),
                    Expr("Distributed", x, _dist("BinomialDistribution", t, self.p[0])))

    def cov(self, s, t):
        p, q = self.p
        return sp.expand(p + q - (p - q) ** 2) * _min(s, t)

    def path(self, rng, times):
        p, q = map(_f, self.p)
        out, x, prev = [], 0, 0
        for t in times:
            for _ in range(t - prev):
                u = rng.random()
                x += 1 if u < p else (-1 if u < p + q else 0)
            out.append(x)
            prev = t
        return out


class WhiteNoise(Proc):
    dtime = True

    def parse(self, a, fname):
        if len(a) == 0:
            a = [_dist("NormalDistribution", 0, 1)]
        elif len(a) == 1 and not W.is_dist(a[0]):
            s = _p(a[0], fname, "sigma")
            _pos(s, fname, "sigma")
            a = [_dist("NormalDistribution", 0, s)]
        if len(a) != 1 or not W.is_dist(a[0]):
            raise ValueError(f"{fname}: WhiteNoiseProcess expects [], [sigma] or [dist]")
        self.d = a[0]
        self.p = []

    def numeric(self):
        return W.dist_of(self.d, self.fname).numeric() if self.d.head not in W.MULTI_BUILDERS else True

    def slice(self, t):
        return self.d

    def cov(self, s, t):
        return sp.Piecewise((self.var(s), sp.Eq(s, t)), (0, True))

    def stationary(self):
        v = _ev(Expr("Variance", self.d))
        return not (isinstance(v, Symbol) and v.name in ("Indeterminate", "Infinity"))

    def path(self, rng, times):
        dd = W.dist_of(self.d, "RandomFunction")
        out = []
        for _ in times:
            v = dd.sample(W._rng())
            out.append(float(v) if not isinstance(v, (int, float)) else v)
        return out


class FBM(Proc):
    gaussian = True

    def parse(self, a, fname):
        if len(a) == 1:
            a = [0, 1, a[0]]
        if len(a) != 3:
            raise ValueError(f"{fname}: FractionalBrownianMotionProcess expects [h] or [mu, sigma, h]")
        self.p = [_p(v, fname) for v in a]
        _real(self.p[0], fname, "mu")
        _pos(self.p[1], fname, "sigma")
        h = self.p[2]
        if _num(h) and not (h.is_real and 0 < h < 1):
            raise ValueError(f"{fname}: the Hurst index must be in (0, 1)")

    def slice(self, t):
        m, s, h = self.p
        return _dist("NormalDistribution", m * t, s * t ** h)

    def mean(self, t):
        return self.p[0] * t

    def var(self, t):
        return self.p[1] ** 2 * t ** (2 * self.p[2])

    def cov(self, s_, t):
        m, s, h = self.p
        return s ** 2 / 2 * (sp.Abs(s_) ** (2 * h) + sp.Abs(t) ** (2 * h) - sp.Abs(t - s_) ** (2 * h))

    def path(self, rng, times):
        import numpy as np
        m, s, h = map(_f, self.p)
        if len(times) > 4000:
            raise NotImplementedError("RandomFunction: FractionalBrownianMotionProcess paths are limited to 4000 "
                                      "points (exact Cholesky sampling)")
        ts = np.array(times, dtype=float)
        pos = ts > 0
        out = np.zeros(len(ts))
        tp = ts[pos]
        if len(tp):
            S_ = s * s / 2 * (tp[:, None] ** (2 * h) + tp[None, :] ** (2 * h) - np.abs(tp[:, None] - tp[None, :]) ** (2 * h))
            Lc = np.linalg.cholesky(S_ + 1e-14 * np.eye(len(tp)) * S_.max())
            out[pos] = Lc @ rng.standard_normal(len(tp))
        return list(out + m * ts)


class FGN(Proc):
    dtime = True
    gaussian = True

    def parse(self, a, fname):
        if len(a) == 1:
            a = [0, 1, a[0]]
        if len(a) != 3:
            raise ValueError(f"{fname}: FractionalGaussianNoiseProcess expects [h] or [mu, sigma, h]")
        self.p = [_p(v, fname) for v in a]
        _real(self.p[0], fname, "mu")
        _pos(self.p[1], fname, "sigma")
        h = self.p[2]
        if _num(h) and not (h.is_real and 0 < h < 1):
            raise ValueError(f"{fname}: the Hurst index must be in (0, 1)")

    def slice(self, t):
        return _dist("NormalDistribution", self.p[0], self.p[1])

    def mean(self, t):
        return self.p[0]

    def var(self, t):
        return self.p[1] ** 2

    def cov(self, s_, t):
        m, s, h = self.p
        k = t - s_
        return s ** 2 / 2 * (sp.Abs(k + 1) ** (2 * h) - 2 * sp.Abs(k) ** (2 * h) + sp.Abs(k - 1) ** (2 * h))

    def stationary(self):
        return True

    def path(self, rng, times):
        m, s, h = map(_f, self.p)
        g = [s * s / 2 * (abs(k + 1) ** (2 * h) - 2 * abs(k) ** (2 * h) + abs(k - 1) ** (2 * h))
             for k in range(len(times))]
        return [m + v for v in _gauss_toeplitz(rng, g)]


def _gauss_toeplitz(rng, g):
    """Exact sample of a stationary Gaussian sequence with autocovariances g[0..n-1] (Durbin-Levinson)."""
    n = len(g)
    out = []
    phi = []
    v = g[0]
    for k in range(n):
        if k == 0:
            out.append(rng.normal(0, math.sqrt(v)))
            continue
        # update Durbin-Levinson coefficients phi_{k,1..k}
        num = g[k] - sum(phi[j] * g[k - 1 - j] for j in range(k - 1))
        a = num / v
        phi = [phi[j] - a * phi[k - 2 - j] for j in range(k - 1)] + [a]
        v = v * (1 - a * a)
        m = sum(phi[j] * out[k - 1 - j] for j in range(k))
        out.append(rng.normal(m, math.sqrt(max(v, 0.0))))
    return out


# ─────────────────────────── ARMA family ───────────────────────────

def _polymul(a, b):
    """coefficients (index = power) of the product of two polynomials"""
    out = [sp.Integer(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return [sp.expand(c) for c in out]


def _ar_poly(phi):
    return [sp.Integer(1)] + [-c for c in phi]


def _ma_poly(theta):
    return [sp.Integer(1)] + list(theta)


def _seasonal(coeffs, s, sign):
    out = [sp.Integer(0)] * (s * len(coeffs) + 1)
    out[0] = sp.Integer(1)
    for i, c in enumerate(coeffs):
        out[s * (i + 1)] = sign * c
    return out


def _poly_to_ar(poly):
    return [sp.expand(-c) for c in poly[1:]]


def _trim(poly):
    while len(poly) > 1 and poly[-1] == 0:
        poly = poly[:-1]
    return poly


def stable(phi):
    """True iff 1 - sum phi_i z^i has every zero outside the unit circle (Schur-Cohn / Levinson step-down; exact for
    exact coefficients).  None for symbolic coefficients."""
    a = list(phi)
    while a and a[-1] == 0:
        a.pop()
    if not all(_num(c) for c in a):
        return None
    for k in range(len(a), 0, -1):
        kap = a[k - 1]
        if not abs(kap) < 1:
            return False
        a = [sp.nsimplify(0) + (a[i] + kap * sp.conjugate(a[k - 2 - i])) / (1 - abs(kap) ** 2) for i in range(k - 1)]
    return True


class ARMA(Proc):
    """ARProcess / MAProcess / ARMAProcess / ARIMAProcess / SARMAProcess / SARIMAProcess / FARIMAProcess with
    scalar coefficients: (1 - sum phi B^i)(1 - B)^d ... x_t = c + (1 + sum theta B^j) e_t, e_t ~ N(0, v)."""
    dtime = True
    gaussian = True

    def parse(self, a, fname):
        h = self.name
        self.c = sp.Integer(0)
        self.diff = [sp.Integer(1)]           # differencing polynomial
        self.frac = None
        forms = {"ARProcess": ("ar", "v"), "MAProcess": ("ma", "v"), "ARMAProcess": ("ar", "ma", "v"),
                 "ARIMAProcess": ("ar", "d", "ma", "v"), "FARIMAProcess": ("ar", "fd", "ma", "v"),
                 "SARMAProcess": ("ar", "ma", "sarma", "v"), "SARIMAProcess": ("ar", "d", "ma", "sarima", "v")}[h]
        if len(a) == 1 and not is_list(a[0]):
            raise NotImplementedError(f"{fname}: {h}[orders] is an estimation template; only EstimatedProcess and "
                                      "FindProcessParameters accept it") if fname != h else Unevaluated()
        if len(a) == len(forms) + 1 and not is_list(a[0]):
            self.c = _p(a[0], fname, "constant")
            a = a[1:]
        elif len(a) == len(forms) + 1:
            raise NotImplementedError(f"{fname}: {h} with initial data is not supported")
        if len(a) != len(forms):
            raise ValueError(f"{fname}: {h} expects {self._form(forms)}")
        phi, theta = [], []
        for spec, v in zip(forms, a):
            if spec == "ar":
                phi = _plist(v, fname, "AR coefficient list")
            elif spec == "ma":
                theta = _plist(v, fname, "MA coefficient list")
            elif spec == "d":
                if is_list(v):
                    raise NotImplementedError(f"{fname}: vector integration orders are not supported")
                d = _p(v, fname, "integration order")
                if not (_num(d) and d.is_Integer and d >= 0):
                    if _num(d):
                        raise ValueError(f"{fname}: the integration order must be a non-negative integer")
                    raise Unevaluated()
                for _ in range(int(d)):
                    self.diff = _polymul(self.diff, [sp.Integer(1), sp.Integer(-1)])
            elif spec == "fd":
                d = _p(v, fname, "fractional order")
                if _num(d) and not (d.is_real and -sp.Rational(1, 2) < d < sp.Rational(1, 2)):
                    raise ValueError(f"{fname}: the fractional differencing order must be in (-1/2, 1/2)")
                self.frac = d
            elif spec in ("sarma", "sarima"):
                n = 3 if spec == "sarma" else 4
                if not (is_list(v) and len(v.args) == n):
                    raise ValueError(f"{fname}: the seasonal part must be {{s, {{alpha}}, {{beta}}}}" if n == 3 else
                                     f"{fname}: the seasonal part must be {{s, {{alpha}}, D, {{beta}}}}")
                s = v.args[0]
                if not (isinstance(s, int) and not isinstance(s, bool) and s > 0):
                    raise ValueError(f"{fname}: the seasonal period must be a positive integer")
                sal = _plist(v.args[1], fname, "seasonal AR coefficient list")
                sma = _plist(v.args[-1], fname, "seasonal MA coefficient list")
                if n == 4:
                    D = v.args[2]
                    if not (isinstance(D, int) and not isinstance(D, bool) and D >= 0):
                        raise ValueError(f"{fname}: the seasonal integration order must be a non-negative integer")
                    for _ in range(D):
                        self.diff = _polymul(self.diff, _seasonal([sp.Integer(1)], s, -1))
                self.season = (s, sal, sma)
            else:
                if is_list(v):
                    raise NotImplementedError(f"{fname}: vector processes (covariance matrix noise) are not supported")
                self.v = _p(v, fname, "noise variance")
                _pos(self.v, fname, "the noise variance")
        arp, map_ = _ar_poly(phi), _ma_poly(theta)
        if hasattr(self, "season"):
            s, sal, sma = self.season
            arp = _polymul(arp, _seasonal(sal, s, -1))
            map_ = _polymul(map_, _seasonal(sma, s, 1))
        self.phi = _poly_to_ar(_trim(arp))
        self.theta = list(_trim(map_)[1:])
        self.orig_phi, self.orig_theta = phi, theta
        self.p = [self.c] + self.phi + self.theta + [self.v] + ([self.frac] if self.frac is not None else [])

    @staticmethod
    def _form(forms):
        names = {"ar": "{a1, ...}", "ma": "{b1, ...}", "v": "v", "d": "d", "fd": "d",
                 "sarma": "{s, {alpha1, ...}, {beta1, ...}}", "sarima": "{s, {alpha1, ...}, D, {beta1, ...}}"}
        return "[" + ", ".join(names[f] for f in forms) + "] or [c, ...]"

    @property
    def integrated(self):
        return len(self.diff) > 1

    def stationary(self):
        if self.integrated:
            return False
        r = stable(self.phi)
        if r is None:
            raise Unevaluated()
        return r

    def invertible(self):
        r = stable([-c for c in self.theta])
        if r is None:
            raise Unevaluated()
        return r

    def _need_stationary(self, fname):
        if self.integrated:
            raise NotImplementedError(f"{fname}: {self.name} with integration is not weakly stationary; its slices "
                                      "depend on initial data, which is not supported")
        if self.frac is not None and self.phi:
            raise NotImplementedError(f"{fname}: FARIMAProcess with an autoregressive part is not supported (only "
                                      "{{}} AR coefficients)")
        r = stable(self.phi)
        if r is False:
            raise ValueError(f"{fname}: {self.name} is not weakly stationary")

    def mean(self, t):
        self._need_stationary("Mean")
        return sp.simplify(self.c / (1 - sum(self.phi))) if self.phi else self.c

    def acvf(self, k):
        """autocovariance gamma(k), k a non-negative integer"""
        self._need_stationary("CovarianceFunction")
        if not hasattr(self, "_g"):
            self._g = {}
        if k in self._g:
            return self._g[k]
        if self.frac is not None:
            d = self.frac

            def g0(h):
                h = abs(h)
                return self.v * sp.gamma(1 - 2 * d) * sp.gamma(h + d) / (sp.gamma(d) * sp.gamma(1 - d) * sp.gamma(h + 1 - d)) \
                    if d != 0 else (self.v if h == 0 else sp.Integer(0))
            th = [sp.Integer(1)] + self.theta
            val = sum(th[i] * th[j] * g0(k + i - j) for i in range(len(th)) for j in range(len(th)))
            self._g[k] = sp.simplify(sp.gammasimp(val)) if not _inexact(val) else sp.N(val)
            return self._g[k]
        phi, th, v = self.phi, [sp.Integer(1)] + self.theta, self.v
        p, q = len(phi), len(th) - 1
        psi = []
        for j in range(q + 1):
            psi.append(th[j] + sum(phi[i - 1] * psi[j - i] for i in range(1, min(j, p) + 1)))

        def rhs(kk):
            return v * sum(th[j] * psi[j - kk] for j in range(kk, q + 1)) if kk <= q else sp.Integer(0)
        if "base" not in self._g:
            gs = sp.symbols(f"g0:{p + 1}")
            eqs = []
            for kk in range(p + 1):
                eqs.append(sp.Eq(gs[kk] - sum(phi[i - 1] * gs[abs(kk - i)] for i in range(1, p + 1)), rhs(kk)))
            sol = sp.solve(eqs, gs, dict=True)
            if not sol:
                raise ValueError(f"CovarianceFunction: {self.name} is not weakly stationary")
            self._g["base"] = True
            for kk in range(p + 1):
                val = sol[0][gs[kk]]
                self._g[kk] = sp.factor(sp.simplify(val)) if not _inexact(val) else val
        kk = p + 1
        while kk <= k:
            if kk not in self._g:
                val = sum(phi[i - 1] * self._g[kk - i] for i in range(1, p + 1)) + rhs(kk)
                self._g[kk] = sp.factor(sp.simplify(val)) if not _inexact(val) else sp.expand(val)
            kk += 1
        return self._g[k]

    def acvf_sym(self, h):
        """closed form in a symbolic lag h (pure MA and AR(1)-type processes), else Unevaluated"""
        self._need_stationary("CovarianceFunction")
        if not self.phi and self.frac is None:
            q = len(self.theta)
            return sp.Piecewise(*[(self.acvf(k), sp.Eq(sp.Abs(h), k)) for k in range(q + 1)], (0, True))
        if len(self.phi) == 1 and len(self.theta) <= 1 and self.frac is None:
            a = self.phi[0]
            g0, g1 = self.acvf(0), self.acvf(1)
            if not self.theta:
                return sp.simplify(g0) * a ** sp.Abs(h)
            return sp.Piecewise((g0, sp.Eq(h, 0)), (g1 * a ** (sp.Abs(h) - 1), True))
        raise Unevaluated()

    def var(self, t):
        return self.acvf(0)

    def slice(self, t):
        self._need_stationary("SliceDistribution")
        return _dist("NormalDistribution", self.mean(t), sp.sqrt(self.acvf(0)))

    def cov(self, s, t):
        k = t - s
        if _num(k):
            return self.acvf(int(abs(k)))
        return self.acvf_sym(k)

    # simulation
    def _burn(self):
        import numpy as np
        phi = [_f(c) for c in self.phi]
        if not phi:
            return len(self.theta)
        comp = np.zeros((len(phi), len(phi)))
        comp[0, :] = phi
        comp[1:, :-1] = np.eye(len(phi) - 1)
        rho = max(abs(np.linalg.eigvals(comp)))
        if self.integrated:
            return 0
        if rho >= 1:
            raise ValueError(f"RandomFunction: {self.name} is not weakly stationary")
        return min(10 ** 6, int(math.ceil(math.log(1e-16) / math.log(max(rho, 1e-300)))) + len(self.theta) + 10)

    def path(self, rng, times):
        n = len(times)
        sd = math.sqrt(_f(self.v))
        if self.frac is not None:
            if self.phi:
                raise NotImplementedError("RandomFunction: FARIMAProcess with an autoregressive part is not supported")
            g = [_f(self.acvf(k)) for k in range(n)]
            return [_f(self.c) + v for v in _gauss_toeplitz(rng, g)]
        phi = [_f(c) for c in self.phi]
        th = [_f(c) for c in self.theta]
        c = _f(self.c)
        if self.integrated:
            # stationary ARMA w_t driving delta(B) x_t = w_t from x = 0 before the first time
            w = ARMA.__new__(ARMA)
            w.__dict__.update(self.__dict__)
            w.diff = [sp.Integer(1)]
            w._g = {}
            ws = ARMA.path(w, rng, times)
            dl = [_f(x) for x in self.diff[1:]]
            xs = []
            for k in range(n):
                xs.append(ws[k] - sum(dl[i] * xs[k - 1 - i] for i in range(len(dl)) if k - 1 - i >= 0))
            return xs
        burn = self._burn()
        tot = burn + n
        e = rng.normal(0.0, sd, tot + len(th))
        mu = c / (1 - sum(phi)) if phi else c
        x = [mu] * len(phi)
        out = []
        q = len(th)
        for k in range(tot):
            ek = q + k
            val = c + sum(phi[i] * x[-1 - i] for i in range(len(phi))) + e[ek] + \
                sum(th[j] * e[ek - 1 - j] for j in range(q))
            x.append(val)
            if len(x) > len(phi) + 1:
                x.pop(0)
            if k >= burn:
                out.append(val)
        return out


class GARCH(Proc):
    dtime = True

    def parse(self, a, fname):
        h = self.name
        if h == "ARCHProcess":
            if len(a) != 2:
                raise ValueError(f"{fname}: ARCHProcess expects [kappa, {{alpha1, ...}}]")
            self.k, self.al, self.be = _p(a[0], fname, "kappa"), _plist(a[1], fname, "alpha list"), []
        else:
            if len(a) != 3:
                raise ValueError(f"{fname}: GARCHProcess expects [kappa, {{alpha1, ...}}, {{beta1, ...}}]")
            self.k, self.al, self.be = (_p(a[0], fname, "kappa"), _plist(a[1], fname, "alpha list"),
                                        _plist(a[2], fname, "beta list"))
        _pos(self.k, fname, "kappa")
        for v in self.al + self.be:
            if _num(v) and not (v.is_real and v >= 0):
                raise ValueError(f"{fname}: the coefficients must be non-negative")
        self.p = [self.k] + self.al + self.be

    def stationary(self):
        s = sum(self.al) + sum(self.be)
        if not _num(s):
            raise Unevaluated()
        return bool(s < 1)

    def mean(self, t):
        return sp.Integer(0)

    def var(self, t):
        if self.stationary() is False:
            raise ValueError(f"Variance: {self.name} is not weakly stationary")
        return sp.simplify(self.k / (1 - sum(self.al) - sum(self.be)))

    def cov(self, s, t):
        return sp.Piecewise((self.var(s), sp.Eq(s, t)), (0, True))

    def path(self, rng, times):
        if not self.stationary():
            raise ValueError(f"RandomFunction: {self.name} is not weakly stationary")
        k, al, be = _f(self.k), [_f(v) for v in self.al], [_f(v) for v in self.be]
        v0 = k / (1 - sum(al) - sum(be))
        xs, ss = [0.0] * len(al), [v0] * len(be)
        burn = 2000
        out = []
        for i in range(burn + len(times)):
            s2 = k + sum(al[j] * xs[-1 - j] ** 2 for j in range(len(al))) + sum(be[j] * ss[-1 - j] for j in range(len(be)))
            x = math.sqrt(s2) * rng.standard_normal()
            xs.append(x)
            ss.append(s2)
            xs, ss = xs[-max(1, len(al)):], ss[-max(1, len(be)):]
            if i >= burn:
                out.append(x)
        return out


# ─────────────────────────── finite Markov processes ───────────────────────────

def _matrix(v, fname, what="transition matrix"):
    if not (is_list(v) and v.args and all(is_list(r) for r in v.args)):
        raise TypeError(f"{fname}: the {what} must be a square matrix")
    n = len(v.args)
    if any(len(r.args) != n for r in v.args):
        raise ValueError(f"{fname}: the {what} must be square")
    return sp.Matrix([[_p(x, fname, "matrix entry") for x in r.args] for r in v.args])


def _close(a, b):
    if _inexact(a, b):
        return abs(_f(a) - _f(b)) < 1e-10
    return sp.simplify(a - b) == 0


class Markov(Proc):
    """DiscreteMarkovProcess[i0 | p0, m] and ContinuousMarkovProcess[i0 | p0, q] (rate matrix) or
    ContinuousMarkovProcess[i0 | p0, m, mu] (embedded chain m and holding rates mu)."""

    def parse(self, a, fname):
        cont = self.name == "ContinuousMarkovProcess"
        self.dtime = not cont
        self.jump = cont
        if len(a) == 1:
            raise NotImplementedError(f"{fname}: {self.name}[n] is an estimation template; only EstimatedProcess "
                                      "accepts it") if fname != self.name else Unevaluated()
        if not (len(a) == 2 or (cont and len(a) == 3)):
            raise ValueError(f"{fname}: {self.name} expects [i0 or p0, m]" + (" or [i0 or p0, m, mu]" if cont else ""))
        if isinstance(a[1], Expr) and a[1].head == "Graph":
            raise NotImplementedError(f"{fname}: {self.name} with a Graph is not supported; give the matrix")
        m = _matrix(a[1], fname)
        n = m.rows
        self.n = n
        if cont and len(a) == 3:
            mu = _plist(a[2], fname, "rate list")
            if len(mu) != n:
                raise ValueError(f"{fname}: the rate list must have one rate per state")
            for v in mu:
                _pos(v, fname, "each rate")
            self._check_stochastic(m, fname)
            self.m, self.mu = m, mu
            q = sp.zeros(n, n)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        q[i, j] = mu[i] * m[i, j]
                q[i, i] = -sum(q[i, j] for j in range(n) if j != i)
            self.q = q
        elif cont:
            for i in range(n):
                for j in range(n):
                    if i != j and _num(m[i, j]) and not m[i, j] >= 0:
                        raise ValueError(f"{fname}: off-diagonal rates must be non-negative")
                if all(_num(x) for x in m.row(i)) and not _close(sum(m.row(i)), 0):
                    raise ValueError(f"{fname}: the rows of the transition rate matrix must sum to 0")
            self.q = m
            self.mu = [-m[i, i] for i in range(n)]
            self.m = sp.Matrix(n, n, lambda i, j: (0 if i == j else (m[i, j] / self.mu[i])) if self.mu[i] != 0
                               else (1 if i == j else 0))
        else:
            self._check_stochastic(m, fname)
            self.m = m
        p0 = a[0]
        if is_list(p0):
            pv = _plist(p0, fname, "initial probability list")
            if len(pv) != n:
                raise ValueError(f"{fname}: the initial probability vector must have length {n}")
            for v in pv:
                _prob(v, fname, "each initial probability")
            if all(_num(v) for v in pv) and not _close(sum(pv), 1):
                raise ValueError(f"{fname}: the initial probabilities must sum to 1")
            self.p0 = sp.Matrix([pv])
        else:
            if not (isinstance(p0, int) and not isinstance(p0, bool)):
                if isinstance(p0, (Symbol, Expr)) and not isinstance(p0, str):
                    raise Unevaluated()
                raise TypeError(f"{fname}: the initial state must be an integer or a probability vector")
            if not 1 <= p0 <= n:
                raise ValueError(f"{fname}: the initial state must be between 1 and {n}")
            self.p0 = sp.Matrix([[1 if j == p0 - 1 else 0 for j in range(n)]])
        self.p = list(self.p0) + list(self.m) + (list(self.q) if cont else [])

    @staticmethod
    def _check_stochastic(m, fname):
        for i in range(m.rows):
            for v in m.row(i):
                if _num(v) and not (v.is_real and 0 <= v <= 1):
                    raise ValueError(f"{fname}: transition probabilities must be between 0 and 1")
            if all(_num(x) for x in m.row(i)) and not _close(sum(m.row(i)), 1):
                raise ValueError(f"{fname}: the rows of the transition matrix must sum to 1")

    def check_time(self, t, fname):
        if self.dtime:
            _int_time(t, fname, 0)
        else:
            _nonneg_time(t, fname)

    def probs(self, t):
        if not _num(t):
            raise Unevaluated()
        if self.dtime:
            pt = self.p0 * self.m ** int(t)
        else:
            pt = self.p0 * (self.q * t).exp()
            if not _inexact(*self.p, t):
                pt = pt.applyfunc(sp.simplify)
        return [sp.simplify(v) if not _inexact(v) else v for v in pt]

    def slice(self, t):
        pt = self.probs(t)
        return Expr("EmpiricalDistribution", rule(R(pt), L(*range(1, self.n + 1))))

    def mean(self, t):
        return sp.simplify(sum((i + 1) * v for i, v in enumerate(self.probs(t))))

    def var(self, t):
        pt = self.probs(t)
        m1 = sum((i + 1) * v for i, v in enumerate(pt))
        return sp.simplify(sum((i + 1) ** 2 * v for i, v in enumerate(pt)) - m1 ** 2)

    def trans(self, h):
        if self.dtime:
            return self.m ** int(h)
        return (self.q * h).exp()

    def cov(self, s, t):
        if not (_num(s) and _num(t)):
            raise Unevaluated()
        if s > t:
            s, t = t, s
        ps = self.probs(s)
        P = self.trans(t - s)
        e = sum((i + 1) * ps[i] * sum((j + 1) * P[i, j] for j in range(self.n)) for i in range(self.n))
        return sp.simplify(e - self.mean(s) * self.mean(t))

    def _draw(self, rng, probs):
        u, acc = rng.random(), 0.0
        for j, pj in enumerate(probs):
            acc += pj
            if u < acc:
                return j
        return len(probs) - 1

    def path(self, rng, times):
        P = [[_f(v) for v in self.m.row(i)] for i in range(self.n)]
        if not self.dtime:
            ev = self.jumps(rng, 0.0, times[-1])
            out, k = [], 0
            for t in times:
                while k + 1 < len(ev) and ev[k + 1][0] <= t:
                    k += 1
                out.append(ev[k][1])
            return out
        x = self._draw(rng, [_f(v) for v in self.p0])
        out, prev = [], 0
        for t in times:
            for _ in range(t - prev):
                x = self._draw(rng, P[x])
            out.append(x + 1)
            prev = t
        return out

    def jumps(self, rng, t0, t1):
        mu = [_f(v) for v in self.mu]
        P = [[_f(v) for v in self.m.row(i)] for i in range(self.n)]
        x = self._draw(rng, [_f(v) for v in self.p0])
        t, out = 0.0, None
        while True:
            if mu[x] == 0:
                break
            hold = rng.exponential(1 / mu[x])
            if t + hold > t1:
                break
            t += hold
            if t > t0 and out is None:
                out = [(t0, x + 1)]
            x = self._draw(rng, P[x])
            if out is not None and out[-1][1] != x + 1:
                out.append((t, x + 1))
        return out if out is not None else [(t0, x + 1)]

    # structure
    def graph_matrix(self):
        return self.m if self.dtime else self.q

    def adjacency(self):
        g = self.graph_matrix()
        for v in g:
            if not _num(v):
                raise Unevaluated()
        return [[(i != j or self.dtime) and g[i, j] != 0 for j in range(self.n)] for i in range(self.n)]

    def classes(self):
        adj = self.adjacency()
        n = self.n
        reach = [[i == j or adj[i][j] for j in range(n)] for i in range(n)]
        for k in range(n):
            for i in range(n):
                if reach[i][k]:
                    for j in range(n):
                        if reach[k][j]:
                            reach[i][j] = True
        seen, cls = set(), []
        for i in range(n):
            if i in seen:
                continue
            c = [j for j in range(n) if reach[i][j] and reach[j][i]]
            seen.update(c)
            cls.append(c)
        rec = [c for c in cls if all(not reach[i][j] or j in c for i in c for j in range(n))]
        return cls, rec, reach

    def period(self, c):
        adj = self.adjacency()
        if not self.dtime:
            return 1
        start = c[0]
        dist = {start: 0}
        frontier = [start]
        g = 0
        while frontier:
            nxt = []
            for i in frontier:
                for j in c:
                    if adj[i][j]:
                        if j not in dist:
                            dist[j] = dist[i] + 1
                            nxt.append(j)
                        else:
                            g = math.gcd(g, dist[i] + 1 - dist[j])
            frontier = nxt
        return g if g else 1

    def class_stationary(self, c):
        """stationary distribution of a closed class (exact linear solve)"""
        k = len(c)
        if self.dtime:
            A = sp.Matrix(k, k, lambda i, j: self.m[c[j], c[i]]) - sp.eye(k)
        else:
            A = sp.Matrix(k, k, lambda i, j: self.q[c[j], c[i]])
        A = A.col_join(sp.ones(1, k))
        b = sp.zeros(k, 1).col_join(sp.Matrix([[1]]))
        if _inexact(*A):
            import numpy as np
            sol = np.linalg.lstsq(np.array(A.tolist(), dtype=float), np.array(b.tolist(), dtype=float), rcond=None)[0]
            return [sp.Float(float(v)) for v in sol.ravel()]
        sols = list(sp.linsolve((A, b)))
        if not sols:
            raise ArithmeticError("StationaryDistribution: no stationary distribution found (result failed verification)")
        sol = sols[0]
        return [sp.simplify(v) if not _inexact(v) else v for v in sol]

    def absorption(self, rec):
        """matrix B[i][r]: probability of eventually entering recurrent class r from state i"""
        n = self.n
        recset = {i for c in rec for i in c}
        trans = [i for i in range(n) if i not in recset]
        P = self.m
        B = [[sp.Integer(0)] * len(rec) for _ in range(n)]
        for r, c in enumerate(rec):
            for i in c:
                B[i][r] = sp.Integer(1)
        if trans:
            Q = sp.Matrix(len(trans), len(trans), lambda a, b: P[trans[a], trans[b]])
            Rm = sp.Matrix(len(trans), len(rec), lambda a, r: sum(P[trans[a], j] for j in rec[r]))
            Bt = (sp.eye(len(trans)) - Q).solve(Rm)
            for a, i in enumerate(trans):
                for r in range(len(rec)):
                    B[i][r] = sp.simplify(Bt[a, r]) if not _inexact(Bt[a, r]) else Bt[a, r]
        return B

    def limit_matrix(self):
        cls, rec, _ = self.classes()
        B = self.absorption(rec)
        pis = [self.class_stationary(c) for c in rec]
        Lm = sp.zeros(self.n, self.n)
        for i in range(self.n):
            for r, c in enumerate(rec):
                for k, j in enumerate(c):
                    Lm[i, j] = B[i][r] * pis[r][k]
        return Lm.applyfunc(lambda v: sp.simplify(v) if not _inexact(v) else v)

    def stationary_dist(self):
        pi = list(self.p0 * self.limit_matrix())
        return Expr("EmpiricalDistribution", rule(R([sp.simplify(v) if not _inexact(v) else v for v in pi]),
                                                  L(*range(1, self.n + 1))))


# ─────────────────────────── queueing processes ───────────────────────────

class Queue(Proc):
    jump = True
    has_cov = False

    def parse(self, a, fname):
        if not 2 <= len(a) <= 5:
            raise ValueError(f"{fname}: QueueingProcess expects [a, s], [a, s, c], [a, s, c, k] or [a, s, c, k, x0]")

        def rate_or_dist(v, what):
            if W.is_dist(v):
                dd = W.dist_of(v, fname)
                lo = dd.bounds()[0]
                if _num(lo) and lo < 0:
                    raise ValueError(f"{fname}: the {what} distribution must have positive support")
                if v.head == "ExponentialDistribution":
                    return _p(v.args[0], fname), None
                return None, v
            r = _p(v, fname, f"{what} rate")
            _pos(r, fname, f"the {what} rate")
            return r, None
        self.lam, self.adist = rate_or_dist(a[0], "arrival")
        self.mu, self.sdist = rate_or_dist(a[1], "service")
        self.c = a[2] if len(a) > 2 else 1
        self.k = a[3] if len(a) > 3 else INF
        self.x0 = a[4] if len(a) > 4 else 0
        for v, what in ((self.c, "number of channels c"),):
            if not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
                if isinstance(v, Symbol) and v.name != "Infinity":
                    raise Unevaluated()
                if not (isinstance(v, Symbol) and v.name == "Infinity"):
                    raise ValueError(f"{fname}: the {what} must be a positive integer")
                raise NotImplementedError(f"{fname}: infinitely many service channels are not supported")
        if not (isinstance(self.k, Symbol) and self.k.name == "Infinity"):
            if not (isinstance(self.k, int) and not isinstance(self.k, bool) and self.k >= self.c):
                if isinstance(self.k, Symbol):
                    raise Unevaluated()
                raise ValueError(f"{fname}: the system capacity k must be an integer at least c")
        if not (isinstance(self.x0, int) and not isinstance(self.x0, bool) and self.x0 >= 0):
            raise ValueError(f"{fname}: the initial state must be a non-negative integer")
        self.finite = isinstance(self.k, int) and not isinstance(self.k, bool)
        if self.finite and self.x0 > self.k:
            raise ValueError(f"{fname}: the initial state exceeds the capacity")
        self.p = [v for v in (self.lam, self.mu) if v is not None]

    def markovian(self):
        return self.adist is None and self.sdist is None

    def numeric(self):
        ok = all(_num(v) for v in self.p)
        for d in (self.adist, self.sdist):
            if d is not None:
                ok = ok and W.dist_of(d, self.fname).numeric()
        return ok

    def _need_mm(self, fname):
        if not self.markovian():
            raise NotImplementedError(f"{fname}: stationary measures are only supported for M/M/c/k queues "
                                      "(exponential arrival and service times)")

    def rho(self):
        return self.lam / (self.c * self.mu)

    def is_stationary(self):
        if self.finite:
            return True
        r = self.rho()
        if not _num(r):
            return None
        return bool(r < 1)

    def pn_finite(self):
        """unnormalised stationary probabilities for a finite capacity queue"""
        lam, mu, c, k = self.lam, self.mu, self.c, self.k
        w = []
        for n in range(k + 1):
            if n <= c:
                w.append((lam / mu) ** n / sp.factorial(n))
            else:
                w.append((lam / mu) ** n / (sp.factorial(c) * c ** (n - c)))
        tot = sum(w)
        return [sp.simplify(v / tot) for v in w]

    def p0_inf(self):
        lam, mu, c = self.lam, self.mu, self.c
        a, r = lam / mu, self.rho()
        return sp.simplify(1 / (sum(a ** n / sp.factorial(n) for n in range(c)) + a ** c / (sp.factorial(c) * (1 - r))))

    def stationary_dist(self, fname="StationaryDistribution"):
        self._need_mm(fname)
        if self.finite:
            return Expr("EmpiricalDistribution", rule(R(self.pn_finite()), L(*range(0, self.k + 1))))
        if self.is_stationary() is False:
            raise ValueError(f"{fname}: the queue is not stationary (arrival rate >= total service rate)")
        if self.c == 1:
            return _dist("GeometricDistribution", 1 - self.rho())
        raise NotImplementedError(f"{fname}: the stationary distribution of an M/M/c queue with c > 1 and infinite "
                                  "capacity is not a named distribution; use QueueProperties")

    def measures(self, fname):
        self._need_mm(fname)
        lam, mu, c = self.lam, self.mu, self.c
        if self.finite:
            pn = self.pn_finite()
            k = self.k
            L_ = sum(n * pn[n] for n in range(k + 1))
            Lq = sum((n - c) * pn[n] for n in range(c + 1, k + 1))
            lam_eff = lam * (1 - pn[k])
            busy = sum(min(n, c) * pn[n] for n in range(k + 1))
        else:
            st = self.is_stationary()
            if st is False:
                return None
            r, a = self.rho(), lam / mu
            p0 = self.p0_inf()
            Lq = p0 * a ** c * r / (sp.factorial(c) * (1 - r) ** 2)
            L_ = Lq + a
            lam_eff = lam
            busy = a
        out = {"MeanSystemSize": L_, "MeanQueueSize": Lq, "MeanSystemTime": L_ / lam_eff,
               "MeanQueueTime": Lq / lam_eff, "Throughput": lam_eff, "UtilizationFactor": busy / c}
        return {k_: (sp.simplify(v) if not _inexact(v) else v) for k_, v in out.items()}

    def jumps(self, rng, t0, t1):
        cap = self.k if self.finite else None
        x = self.x0
        if self.markovian():
            lam, mu = _f(self.lam), _f(self.mu)
            t, out = 0.0, None
            while True:
                up = lam if cap is None or x < cap else 0.0
                dn = mu * min(x, self.c)
                tot = up + dn
                t += rng.exponential(1 / tot)
                if t > t1:
                    break
                if t > t0 and out is None:
                    out = [(t0, x)]
                x += 1 if rng.random() * tot < up else -1
                if out is not None:
                    out.append((t, x))
            return out if out is not None else [(t0, x)]
        # general arrival / service: event simulation with FIFO service
        pyr = W._rng()

        def draw(rate, d):
            if d is None:
                return rng.exponential(1 / _f(rate))
            return float(W.dist_of(d, "RandomFunction").sample(pyr))
        busy = [draw(self.mu, self.sdist) for _ in range(min(x, self.c))]
        waiting = max(0, x - self.c)
        t, nxt_a, out = 0.0, draw(self.lam, self.adist), None
        while True:
            nxt_d = min(busy) if busy else math.inf
            t = min(nxt_a, nxt_d)
            if t > t1:
                break
            if t > t0 and out is None:
                out = [(t0, x)]
            if nxt_a <= nxt_d:
                nxt_a = t + draw(self.lam, self.adist)
                if cap is not None and x >= cap:
                    continue
                x += 1
                if len(busy) < self.c:
                    busy.append(t + draw(self.mu, self.sdist))
                else:
                    waiting += 1
            else:
                busy.remove(nxt_d)
                x -= 1
                if waiting:
                    waiting -= 1
                    busy.append(t + draw(self.mu, self.sdist))
            if out is not None:
                out.append((t, x))
        return out if out is not None else [(t0, x)]
