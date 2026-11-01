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

    def scaled(self, t):
        """(c, k, lambda): X(t) = c Y with Y ~ NoncentralChiSquareDistribution[k, lambda]"""
        m, s, th, x0 = self.p
        e = sp.exp(-th * t)
        return s ** 2 * (1 - e) / (4 * th), 4 * th * m / s ** 2, 4 * th * x0 * e / (s ** 2 * (1 - e))

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
        if self.var_ not in self.lam.free_symbols:
            _cond(Expr("GreaterEqual", R(self.lam), 0))
            if _num(self.lam) and not (self.lam.is_real and self.lam >= 0):
                _bad(f"{fname}: the intensity must be non-negative")
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
        if len(a) not in (1, 2):
            raise ValueError(f"{fname}: RandomWalkProcess expects [p] or [p, q]")
        if any(is_list(v) for v in a):
            raise TypeError(f"{fname}: the step probabilities must be scalars")
        if len(a) == 2:
            p, q = (_p(v, fname) for v in a)
            _prob(p, fname, "p")
            _prob(q, fname, "q")
            _cond(Expr("LessEqual", Expr("Plus", R(p), R(q)), 1))
            if _num(p) and _num(q) and p + q > 1:
                _bad(f"{fname}: p + q must not exceed 1")
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
            raise NotImplementedError("SliceDistribution: the slice of a RandomWalkProcess[p, q] is not "
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

    def stationary_part(self):
        import copy
        sub = copy.copy(self)
        sub.diff = [sp.Integer(1)]
        sub.__dict__.pop("_g", None)
        return sub

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


# ─────────────────────────── SDE and transformed processes ───────────────────────────

def _var_spec(v, fname, what):
    if isinstance(v, Symbol):
        return v, sp.Integer(0)
    if is_list(v) and len(v.args) == 2 and isinstance(v.args[0], Symbol):
        return v.args[0], _p(v.args[1], fname, f"initial {what}")
    raise ValueError(f"{fname}: the {what} must be a symbol or {{symbol, initial value}}")


class Ito(Proc):
    """ItoProcess[{a, b}, x, t] / ItoProcess[{a, b, c}, {x, x0}, {t, t0}] (scalar), dx = a dt + b dw."""

    def parse(self, a, fname):
        h = self.name
        if len(a) == 1:
            raise NotImplementedError(f"{fname}: {h}[proc] (conversion of a named process) is not supported")
        if len(a) == 5 or (a and isinstance(a[0], Expr) and a[0].head in ("Equal", "List") and
                           any(isinstance(x, Expr) and x.head == "Equal" for x in (a[0].args if is_list(a[0]) else [a[0]]))):
            raise NotImplementedError(f"{fname}: the stochastic differential equation form of {h} is not supported; "
                                      "use {drift, diffusion}")
        if len(a) == 4:
            raise NotImplementedError(f"{fname}: a correlated Wiener process covariance is not supported")
        if len(a) != 3 or not is_list(a[0]) or len(a[0].args) not in (2, 3):
            raise ValueError(f"{fname}: {h} expects [{{a, b}}, x, t] or [{{a, b, c}}, {{x, x0}}, {{t, t0}}]")
        if any(is_list(c) for c in a[0].args):
            raise NotImplementedError(f"{fname}: vector {h} is not supported")
        xs, x0 = _var_spec(a[1], fname, "state")
        ts_, t0 = _var_spec(a[2], fname, "time")
        self.xs, self.ts = sp.Symbol(xs.name), sp.Symbol(ts_.name)
        self.x0, self.t0 = x0, t0
        co = [_p(c, fname, "coefficient") for c in a[0].args]
        self.drift, self.diff = co[0], co[1]
        self.outx = co[2] if len(co) == 3 else self.xs
        if h == "StratonovichProcess":
            self.drift = sp.simplify(self.drift + self.diff * sp.diff(self.diff, self.xs) / 2)
        self.p = [x0, t0]
        self.extra = (self.drift.free_symbols | self.diff.free_symbols | self.outx.free_symbols) - {self.xs, self.ts}

    def numeric(self):
        return not self.extra and _num(self.x0) and _num(self.t0)

    def check_time(self, t, fname):
        _nonneg_time(t, fname, self.t0)

    def path(self, rng, times):
        fa = sp.lambdify((self.xs, self.ts), self.drift, "math")
        fb = sp.lambdify((self.xs, self.ts), self.diff, "math")
        fc = sp.lambdify((self.xs, self.ts), self.outx, "math")
        x, t = _f(self.x0), _f(self.t0)
        if times[0] < t - 1e-12:
            raise ValueError(f"RandomFunction: the time range must start at or after t0 = {self.t0}")
        dt = times[1] - times[0] if len(times) > 1 else 0.0
        out = []
        for tt in times:
            while tt - t > 1e-12 * max(1.0, abs(tt)):
                h = min(dt if dt > 0 else tt - t, tt - t)
                x = x + fa(x, t) * h + fb(x, t) * math.sqrt(h) * rng.standard_normal()
                t += h
            out.append(float(fc(x, t)))
        return out


def _component_specs(spec, fname):
    items = list(spec.args) if is_list(spec) else [spec]
    out = []
    for d in items:
        if not (isinstance(d, Expr) and d.head == "Distributed" and len(d.args) == 2 and isinstance(d.args[0], Symbol)):
            raise ValueError(f"{fname}: expected x \\[Distributed] proc or a list of them")
        out.append((d.args[0], proc_of(d.args[1], fname)))
    return out


def _subst(e, reps):
    """replace subexpressions (MikoshiLang) using a dict keyed by expressions"""
    if e in reps:
        return reps[e]
    if isinstance(e, Expr):
        return Expr(_subst(e.head, reps) if isinstance(e.head, Expr) else e.head, *(_subst(a, reps) for a in e.args))
    return e


class Transformed(Proc):
    def parse(self, a, fname):
        if len(a) != 3 or not isinstance(a[2], Symbol):
            raise ValueError(f"{fname}: TransformedProcess expects [expr, x \\[Distributed] proc, t]")
        self.expr, self.tv = a[0], a[2]
        self.comps = _component_specs(a[1], fname)
        self.dtime = all(c.dtime for _, c in self.comps)
        self.p = []

    def numeric(self):
        return all(c.numeric() for _, c in self.comps)

    def check_time(self, t, fname):
        for _, c in self.comps:
            c.check_time(t, fname)

    def slice(self, t):
        if len(self.comps) != 1:
            raise NotImplementedError("SliceDistribution: slices of TransformedProcess with several component "
                                      "processes are not supported")
        x, c = self.comps[0]
        y = Symbol("x\u0303")
        e = _subst(self.expr, {Expr(x.name, self.tv): y, Expr(x, self.tv): y})
        e = _subst(e, {self.tv: R(t)})
        if _contains(e, x):
            raise NotImplementedError("SliceDistribution: the transformation must use the component only as "
                                      f"{x}[{self.tv}]")
        return Expr("TransformedDistribution", e, Expr("Distributed", y, c.slice(t)))

    def path(self, rng, times):
        vals = []
        for x, c in self.comps:
            vals.append(c.path(rng, times))
        out = []
        for k, tt in enumerate(times):
            reps = {}
            for i, (x, c) in enumerate(self.comps):
                v = R(sp.Float(vals[i][k]) if isinstance(vals[i][k], float) else vals[i][k])
                reps[Expr(x.name, self.tv)] = v
                reps[Expr(x, self.tv)] = v
            e = _subst(self.expr, reps)
            e = _subst(e, {self.tv: R(sp.Float(tt)) if isinstance(tt, float) else tt})
            v = _ev(Expr("N", e))
            out.append(v)
        return out


def _contains(e, x):
    if e == x:
        return True
    if isinstance(e, Expr):
        return _contains(e.head, x) or any(_contains(a, x) for a in e.args)
    return False


# ─────────────────────────── registry ───────────────────────────

CLASSES = {
    "WienerProcess": Wiener, "BrownianBridgeProcess": BrownianBridge, "GeometricBrownianMotionProcess": GBM,
    "OrnsteinUhlenbeckProcess": OU, "CoxIngersollRossProcess": CIR, "PoissonProcess": Poisson,
    "TelegraphProcess": Telegraph, "CompoundPoissonProcess": CompoundPoisson, "RenewalProcess": Renewal,
    "CompoundRenewalProcess": CompoundRenewal, "InhomogeneousPoissonProcess": InhomPoisson,
    "BernoulliProcess": Bernoulli, "BinomialProcess": Binomial, "RandomWalkProcess": RandomWalk,
    "WhiteNoiseProcess": WhiteNoise, "FractionalBrownianMotionProcess": FBM,
    "FractionalGaussianNoiseProcess": FGN,
    "ARProcess": ARMA, "MAProcess": ARMA, "ARMAProcess": ARMA, "ARIMAProcess": ARMA, "FARIMAProcess": ARMA,
    "SARMAProcess": ARMA, "SARIMAProcess": ARMA, "ARCHProcess": GARCH, "GARCHProcess": GARCH,
    "DiscreteMarkovProcess": Markov, "ContinuousMarkovProcess": Markov, "QueueingProcess": Queue,
    "ItoProcess": Ito, "StratonovichProcess": Ito, "TransformedProcess": Transformed,
}


def is_proc(x):
    return isinstance(x, Expr) and isinstance(x.head, str) and x.head in CLASSES


def proc_of(x, fname):
    if not is_proc(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {x} is not a random process")
    return CLASSES[x.head](x, fname)


def is_slice(x):
    return isinstance(x, Expr) and is_proc(x.head) and len(x.args) == 1


# ─────────────────────────── constructors ───────────────────────────

_CTOR_NOTES = {
    "WienerProcess": "WienerProcess[mu, sigma] (WienerProcess[] = WienerProcess[0, 1]); slice NormalDistribution[mu t, "
                     "sigma Sqrt[t]], covariance sigma^2 Min[s, t]; simulation by exact Gaussian increments",
    "BrownianBridgeProcess": "BrownianBridgeProcess[sigma, {t1, a}, {t2, b}] (also [{t1, a}, {t2, b}] and [] = [1, "
                             "{0, 0}, {1, 0}]); slices are normal on (t1, t2); exact sequential simulation",
    "GeometricBrownianMotionProcess": "GeometricBrownianMotionProcess[mu, sigma, x0]; slice LogNormalDistribution[Log[x0] "
                                      "+ (mu - sigma^2/2) t, sigma Sqrt[t]]; exact simulation",
    "OrnsteinUhlenbeckProcess": "OrnsteinUhlenbeckProcess[mu, sigma, theta] (stationary) and [mu, sigma, theta, x0]: "
                                "dx = theta (mu - x) dt + sigma dw; normal slices, exact transition simulation",
    "CoxIngersollRossProcess": "CoxIngersollRossProcess[mu, sigma, theta, x0]: dx = theta (mu - x) dt + sigma Sqrt[x] "
                               "dw; slice c x with x ~ NoncentralChiSquareDistribution (exact), exact simulation",
    "PoissonProcess": "PoissonProcess[mu]; slice PoissonDistribution[mu t], covariance mu Min[s, t]",
    "TelegraphProcess": "TelegraphProcess[mu]: (-1)^N(t), N a PoissonProcess[mu], starting at 1; slice "
                        "2 x - 1 with x ~ BernoulliDistribution[(1 + E^(-2 mu t))/2]",
    "CompoundPoissonProcess": "CompoundPoissonProcess[lambda, jdist]; slice CompoundPoissonDistribution[lambda t, "
                              "jdist], mean lambda t Mean[jdist], covariance lambda Min[s, t] E[J^2]",
    "RenewalProcess": "RenewalProcess[rdist] counting process with inter-event times rdist; RandomFunction only "
                      "(the slice distribution has no closed form and raises NotImplementedError)",
    "CompoundRenewalProcess": "CompoundRenewalProcess[rdist, jdist]; RandomFunction only (slices raise "
                              "NotImplementedError)",
    "InhomogeneousPoissonProcess": "InhomogeneousPoissonProcess[lambda[t], t]; slice PoissonDistribution[Integrate["
                                   "lambda, {u, 0, t}]] (NotImplementedError without a closed form); simulation by "
                                   "inverting the integrated intensity",
    "BernoulliProcess": "BernoulliProcess[p] (discrete time); slice BernoulliDistribution[p], i.i.d.",
    "BinomialProcess": "BinomialProcess[p] (discrete time, counts of successes); slice BinomialDistribution[t, p]",
    "RandomWalkProcess": "RandomWalkProcess[p] (steps +1 with probability p, -1 otherwise) and RandomWalkProcess[p, "
                         "q] (+1 with p, -1 with q, 0 otherwise); slice 2 x - t with x ~ BinomialDistribution[t, p]",
    "WhiteNoiseProcess": "WhiteNoiseProcess[] (standard normal), WhiteNoiseProcess[sigma] and WhiteNoiseProcess[dist]; "
                         "i.i.d. slices dist",
    "FractionalBrownianMotionProcess": "FractionalBrownianMotionProcess[mu, sigma, h] (and [h] = [0, 1, h]); slice "
                                       "NormalDistribution[mu t, sigma t^h], covariance sigma^2/2 (|s|^2h + |t|^2h - "
                                       "|t - s|^2h); exact Cholesky simulation (up to 4000 points)",
    "FractionalGaussianNoiseProcess": "FractionalGaussianNoiseProcess[mu, sigma, h] (and [h]); stationary normal "
                                      "slices, covariance sigma^2/2 (|k+1|^2h - 2|k|^2h + |k-1|^2h); exact simulation",
    "ARProcess": "ARProcess[{a1, ...}, v] and ARProcess[c, {a1, ...}, v]: x_t = c + Sum a_i x_(t-i) + e_t, e_t ~ "
                 "N(0, v); ARProcess[p] is an estimation template",
    "MAProcess": "MAProcess[{b1, ...}, v] and MAProcess[c, {b1, ...}, v]: x_t = c + e_t + Sum b_j e_(t-j)",
    "ARMAProcess": "ARMAProcess[{a...}, {b...}, v] and ARMAProcess[c, {a...}, {b...}, v]",
    "ARIMAProcess": "ARIMAProcess[{a...}, d, {b...}, v] (and with constant c): the d-th difference is "
                    "ARMAProcess[{a...}, {b...}, v]",
    "FARIMAProcess": "FARIMAProcess[{a...}, d, {b...}, v] with fractional -1/2 < d < 1/2; covariance, slices and "
                     "simulation need an empty AR part (others raise NotImplementedError)",
    "SARMAProcess": "SARMAProcess[{a...}, {b...}, {s, {alpha...}, {beta...}}, v]: multiplicative seasonal ARMA",
    "SARIMAProcess": "SARIMAProcess[{a...}, d, {b...}, {s, {alpha...}, D, {beta...}}, v]",
    "ARCHProcess": "ARCHProcess[kappa, {alpha1, ...}]: x_t = sigma_t z_t, sigma_t^2 = kappa + Sum alpha_i x_(t-i)^2",
    "GARCHProcess": "GARCHProcess[kappa, {alpha...}, {beta...}]: sigma_t^2 = kappa + Sum alpha_i x_(t-i)^2 + Sum "
                    "beta_j sigma_(t-j)^2; mean 0, variance kappa/(1 - Sum alpha - Sum beta) when weakly stationary",
    "DiscreteMarkovProcess": "DiscreteMarkovProcess[i0, m] / DiscreteMarkovProcess[p0, m] (states 1..n, rows of m sum "
                             "to 1); slice EmpiricalDistribution[p0.m^t -> Range[n]]; Graph input and the [n] template "
                             "are not supported",
    "ContinuousMarkovProcess": "ContinuousMarkovProcess[i0 | p0, q] (transition rate matrix, rows sum to 0) and "
                               "[i0 | p0, m, mu] (embedded chain m, holding rates mu); slice EmpiricalDistribution["
                               "p0.MatrixExp[q t] -> Range[n]]",
    "QueueingProcess": "QueueingProcess[a, s, c, k, x0] with a an arrival rate or distribution, s a service rate or "
                       "distribution, c servers, capacity k (default Infinity) and initial size x0; QueueProperties "
                       "and StationaryDistribution for M/M/c/k (and M/G/1 measures); RandomFunction by event "
                       "simulation",
    "ItoProcess": "ItoProcess[{a, b}, x, t] and ItoProcess[{a, b, c}, {x, x0}, {t, t0}] (scalar): dx = a dt + b dw, "
                  "output c; RandomFunction uses the Euler-Maruyama scheme with the given step (an approximation of "
                  "strong order 1/2); slices, the SDE-equation form, vector processes and ItoProcess[proc] raise "
                  "NotImplementedError",
    "StratonovichProcess": "StratonovichProcess[{a, b}, x, t] (and with {x, x0}, {t, t0}): simulated as the "
                           "equivalent Ito process with drift a + b b_x/2 (Euler-Maruyama)",
    "TransformedProcess": "TransformedProcess[expr, x \\[Distributed] proc, t] (or a list of components); slices "
                          "(one component) are TransformedDistribution[expr /. x[t] -> y, y ~ proc[t]]; RandomFunction "
                          "transforms simulated component paths",
}


def _register_ctor(name):
    def impl(*args: Expr) -> Expr:
        if name == "WienerProcess" and not args:
            return Expr(name, 0, 1)
        del _CONDS[:]
        CLASSES[name](Expr(name, *args), name)
        raise Unevaluated()
    impl.__name__ = "_ctor_" + name
    fn(name, area=AREA, status="added",
       notes=_CTOR_NOTES[name] + ". Inert process object; parameters are validated (invalid numeric parameters "
                                 "raise ValueError, wrong kinds TypeError, unsupported forms NotImplementedError); "
                                 "symbolic parameters are allowed. Check: exact (closed forms) and reference values "
                                 "(Wolfram documentation, SciPy/statsmodels) in tests.")(impl)


for _n in CLASSES:
    _register_ctor(_n)


# ─────────────────────────── slices ───────────────────────────

def _times_arg(t, fname):
    if is_list(t):
        return [_p(v, fname, "time") for v in t.args]
    return _p(t, fname, "time")


@fn("SliceDistribution", area=AREA, status="added",
    notes="SliceDistribution[proc, t] (also entered as proc[t]) gives the named distribution of the state at time t "
          "(NormalDistribution, PoissonDistribution, LogNormalDistribution, ...; see each process); "
          "SliceDistribution[proc, {t1, ...}] gives MultinormalDistribution[means, covariances] for Gaussian "
          "processes (others raise NotImplementedError). Times outside the process domain raise ValueError; "
          "processes whose slices have no closed form raise NotImplementedError. "
          "Check: exact closed forms; reference (Wolfram documentation, moment identities) in tests.")
def _slicedist(proc: Expr, t: Real | Var | ListOf[Real]) -> Expr:
    pr = proc_of(proc, "SliceDistribution")
    ts_ = _times_arg(t, "SliceDistribution")
    if isinstance(ts_, list):
        for v in ts_:
            pr.check_time(v, "SliceDistribution")
        if not pr.gaussian:
            raise NotImplementedError(f"SliceDistribution: multi-time slices of {pr.name} are not supported "
                                      "(Gaussian processes only)")
        means = [sp.simplify(pr.mean(v)) for v in ts_]
        cov = [[sp.simplify(pr.cov(a, b)) for b in ts_] for a in ts_]
        return Expr("MultinormalDistribution", R(means), R(cov))
    pr.check_time(ts_, "SliceDistribution")
    return pr.slice(ts_)


def slice_dist(x):
    pr = proc_of(x.head, "SliceDistribution")
    return _ev(Expr("SliceDistribution", x.head, x.args[0])), pr


def _slice_first(args):
    return bool(args) and is_slice(args[0])


def _moment_impl(which):
    def impl(x: Expr, *rest: Expr) -> Expr:
        pr = proc_of(x.head, which)
        t = _times_arg(x.args[0], which)
        if isinstance(t, list):
            d = _ev(Expr("SliceDistribution", x.head, x.args[0]))
            return _ev(Expr(which, d, *rest))
        pr.check_time(t, which)
        if which == "Mean":
            v = pr.mean(t)
        elif which == "Variance":
            v = pr.var(t)
        else:
            v = sp.sqrt(pr.var(t))
        if isinstance(v, sp.Basic):         # times are real and >= t0, so polar lifts are spurious
            v = v.replace(lambda e: isinstance(e, sp.Function) and type(e).__name__ == "polar_lift",
                          lambda e: e.args[0])
        return _out(sp.simplify(v) if not _inexact(v) else v, x)
    return impl


for _w in ("Mean", "Variance", "StandardDeviation"):
    W.extend(_w, _moment_impl(_w), _slice_first,
             f"{_w}[proc[t]] for random processes (closed form, also where the slice distribution is not named)")


def _via_slice(which):
    def impl(x: Expr, *rest: Expr) -> Expr:
        if which in ("PDF", "CDF", "SurvivalFunction") and len(rest) == 1 and x.head.head == "CoxIngersollRossProcess":
            # X(t) = c Y with Y noncentral chi-square: evaluate through Y at the point x/c
            pr = proc_of(x.head, which)
            t = _times_arg(x.args[0], which)
            if not isinstance(t, list):
                pr.check_time(t, which)
                c, k, lam = pr.scaled(t)
                pt = sym(rest[0], which)
                if _inexact(pt):
                    c, k, lam = (sp.Float(_f(v)) for v in (c, k, lam))
                y = _dist("NoncentralChiSquareDistribution", k, lam)
                v = _ev(Expr(which, y, R(pt / c)))
                return _ev(Expr("Times", v, R(1 / c))) if which == "PDF" else v
        d, _ = slice_dist(x)
        return _ev(Expr(which, d, *rest))
    return impl


for _w in ("PDF", "CDF", "InverseCDF", "SurvivalFunction", "HazardFunction", "Quantile", "Median", "Skewness",
           "Kurtosis", "Moment", "CentralMoment", "Cumulant", "CharacteristicFunction", "MomentGeneratingFunction",
           "RandomVariate"):
    W.extend(_w, _via_slice(_w), _slice_first, f"{_w}[proc[t], ...] uses SliceDistribution[proc, t]")


def _dist_second(args):
    return len(args) == 2 and isinstance(args[1], Expr) and args[1].head == "Distributed" and \
        len(args[1].args) == 2 and is_slice(args[1].args[1])


def _via_slice2(which):
    def impl(f: Expr, spec: Expr) -> Expr:
        d, _ = slice_dist(spec.args[1])
        return _ev(Expr(which, f, Expr("Distributed", spec.args[0], d)))
    return impl


for _w in ("Expectation", "Probability", "NExpectation", "NProbability"):
    W.extend(_w, _via_slice2(_w), _dist_second, f"{_w}[expr, x \\[Distributed] proc[t]] uses the slice distribution")


# ─────────────────────────── covariance and correlation functions ───────────────────────────

def _stationary_or_raise(pr, fname):
    try:
        st = pr.stationary()
    except Unevaluated:
        return
    if st is False:
        raise ValueError(f"{fname}: {pr.name} is not weakly stationary; give two times s, t")


def _proc_cf(kind, pr, s, t, fname):
    pr.check_time(s, fname)
    pr.check_time(t, fname)
    c = pr.cov(s, t)
    if kind == "cov":
        v = c
    elif kind == "abs":
        v = c + pr.mean(s) * pr.mean(t)
    else:
        v = c / sp.sqrt(pr.var(s) * pr.var(t))
    return v if _inexact(v) else sp.simplify(v)


def _lags(spec, fname, start=0):
    if is_list(spec):
        a = list(spec.args)
        if len(a) == 1 and is_list(a[0]):
            return [_p(v, fname, "lag") for v in a[0].args], True
        if len(a) == 1:
            lo, hi, st = sp.Integer(start), _p(a[0], fname, "lag"), sp.Integer(1)
        elif len(a) == 2:
            lo, hi, st = _p(a[0], fname, "lag"), _p(a[1], fname, "lag"), sp.Integer(1)
        elif len(a) == 3:
            lo, hi, st = (_p(v, fname, "lag") for v in a)
        else:
            raise ValueError(f"{fname}: invalid lag specification")
        if not (_num(lo) and _num(hi) and _num(st)):
            raise Unevaluated()
        if not st > 0:
            raise ValueError(f"{fname}: the lag step must be positive")
        out, k = [], 0
        while lo + k * st <= hi:
            out.append(lo + k * st)
            k += 1
        return out, True
    return [_p(spec, fname, "lag")], False


def _ts_out(lags, vals, listed):
    from .wl_processes_ts import make_ts
    if not listed:
        return vals[0]
    return make_ts(list(zip(lags, vals)))


def _data_values(x, fname):
    """list of paths (lists of SymPy values) from a list, TimeSeries or TemporalData"""
    from .wl_processes_ts import td_paths, series_of
    if isinstance(x, Expr) and x.head == "TemporalData" and x.args and is_list(x.args[0]):
        paths, _ = td_paths(x)
        vals = [[v for _, v in p] for p in paths]
    elif is_list(x) and x.args and not all(is_list(v) and len(v.args) == 2 for v in x.args):
        vals = [list(x.args)]
    else:
        h, pairs, _ = series_of(x, fname)
        vals = [[v for _, v in pairs]]
    out = []
    for path in vals:
        row = []
        for v in path:
            if is_list(v):
                raise NotImplementedError(f"{fname}: vector-valued data is not supported")
            sv = S(v)
            if not (isinstance(sv, sp.Basic) and sv.is_number):
                if isinstance(sv, sp.Basic) and sv.free_symbols:
                    raise Unevaluated()
                raise TypeError(f"{fname}: data values must be numbers")
            row.append(sv)
        out.append(row)
    return out


def _sample_cf(kind, xs, h, fname):
    n = len(xs)
    if not (_num(h) and h.is_Integer):
        if _num(h):
            raise ValueError(f"{fname}: lags must be integers for data")
        raise Unevaluated()
    h = abs(int(h))
    if h >= n:
        raise ValueError(f"{fname}: the lag {h} must be smaller than the data length {n}")
    if kind == "abs":
        return sum(xs[i + h] * xs[i] for i in range(n - h)) / n
    m = sum(xs) / n
    c = sum((xs[i + h] - m) * (xs[i] - m) for i in range(n - h)) / n
    if kind == "cov":
        return c
    c0 = sum((v - m) ** 2 for v in xs) / n
    if c0 == 0:
        raise ValueError(f"{fname}: the data has zero variance")
    return c / c0


def _durbin_levinson(r, hmax):
    """partial autocorrelations phi_kk, k = 1..hmax, from autocorrelations r[0..hmax]"""
    out, phi = [], []
    for k in range(1, hmax + 1):
        if k == 1:
            a = r[1] / r[0]
            phi = [a]
        else:
            num = r[k] - sum(phi[j] * r[k - 1 - j] for j in range(k - 1))
            den = r[0] - sum(phi[j] * r[j + 1] for j in range(k - 1))
            a = num / den
            phi = [phi[j] - a * phi[k - 2 - j] for j in range(k - 1)] + [a]
        out.append(sp.simplify(a) if not _inexact(a) else a)
    return out


def _cf_function(name, kind):
    def impl(x: Expr, h: Expr, t: Expr = None) -> Expr:
        if is_proc(x):
            pr = proc_of(x, name)
            if not pr.has_cov:
                raise NotImplementedError(f"{name}: not available for {pr.name}")
            if kind == "pacf":
                if t is not None:
                    raise ValueError(f"{name}: expected {name}[proc, h]")
                if not isinstance(pr, ARMA):
                    raise NotImplementedError(f"{name}: only weakly stationary time series processes are supported")
                pr._need_stationary(name)
                lags, listed = _lags(h, name, start=1)
                vals = []
                for k in lags:
                    if not (_num(k) and k.is_Integer):
                        if _num(k):
                            raise ValueError(f"{name}: lags must be integers")
                        raise Unevaluated()
                    if k < 1:
                        raise ValueError(f"{name}: lags must be positive")
                    r = [pr.acvf(j) for j in range(int(k) + 1)]
                    vals.append(_out(_durbin_levinson(r, int(k))[-1], x))
                return _ts_out(lags, vals, listed)
            if t is not None:
                return _out(_proc_cf(kind, pr, _p(h, name, "time"), _p(t, name, "time"), name), x, h, t)
            _stationary_or_raise(pr, name)
            lags, listed = _lags(h, name)
            vals = []
            for k in lags:
                if pr.dtime and _num(k) and not k.is_Integer:
                    raise ValueError(f"{name}: lags of a discrete-time process must be integers")
                ref = pr.t0 if not pr.dtime else sp.Integer(0)
                vals.append(_out(_proc_cf(kind, pr, ref + sp.Abs(k) if _num(k) else ref + k, ref, name)
                                 if not isinstance(pr, ARMA) else _proc_cf(kind, pr, k, sp.Integer(0), name), x, h))
            return _ts_out(lags, vals, listed)
        if t is not None:
            raise ValueError(f"{name}: data takes a single lag specification")
        paths = _data_values(x, name)
        lags, listed = _lags(h, name, start=1 if kind == "pacf" else 0)
        vals = []
        for k in lags:
            if kind == "pacf":
                if not (_num(k) and k.is_Integer):
                    if _num(k):
                        raise ValueError(f"{name}: lags must be integers")
                    raise Unevaluated()
                if k < 1:
                    raise ValueError(f"{name}: lags must be positive")
                tot = 0
                for xs in paths:
                    r = [_sample_cf("corr", xs, sp.Integer(j), name) for j in range(int(k) + 1)]
                    tot += _durbin_levinson(r, int(k))[-1]
            else:
                tot = sum(_sample_cf(kind, xs, k, name) for xs in paths)
            v = tot / len(paths)
            vals.append(_out(sp.nsimplify(v) if False else v, x))
        return _ts_out(lags, vals, listed)
    return impl


_CF_NOTES = {
    "CovarianceFunction": "CovarianceFunction[proc, s, t] = Cov[x[s], x[t]] in closed form; CovarianceFunction[proc, "
                          "hspec] for weakly stationary processes (Cov[x[t + h], x[t]]; non-stationary processes raise "
                          "ValueError); CovarianceFunction[data, hspec] = (1/n) Sum_(i=1)^(n-h) (x_(i+h) - xbar)(x_i - "
                          "xbar) (Wolfram's biased estimator), averaged over the paths of TemporalData",
    "CorrelationFunction": "CorrelationFunction[proc, s, t] = Cov[x[s], x[t]]/(sd[s] sd[t]); [proc, hspec] for weakly "
                           "stationary processes; [data, hspec] = sample covariance at lag h / sample covariance at 0",
    "AbsoluteCorrelationFunction": "AbsoluteCorrelationFunction[proc, s, t] = E[x[s] x[t]]; [proc, hspec] for weakly "
                                   "stationary processes; [data, hspec] = (1/n) Sum x_(i+h) x_i",
    "PartialCorrelationFunction": "PartialCorrelationFunction[proc, hspec] for weakly stationary ARMA-type processes "
                                  "and [data, hspec] (Durbin-Levinson recursion on the (sample) autocorrelations); "
                                  "lags start at 1 ({hmax} = 1..hmax); lag 0 raises ValueError",
}

for _name, _kind in (("CovarianceFunction", "cov"), ("CorrelationFunction", "corr"),
                     ("AbsoluteCorrelationFunction", "abs"), ("PartialCorrelationFunction", "pacf")):
    fn(_name, area=AREA, status="added",
       notes=_CF_NOTES[_name] + ". hspec: h, {hmax} (0..hmax), {hmin, hmax}, {hmin, hmax, dh} or {{h1, ...}}; a lag "
             "list gives a TimeSeries over the lags. Exact for exact input; vector-valued data raises "
             "NotImplementedError; symbolic lags stay unevaluated where no closed form exists. "
             "Check: exact; reference (Wolfram documentation identities, statsmodels acf/acovf/pacf_yw values) "
             "in tests.")(_cf_function(_name, _kind))


# ─────────────────────────── stationarity, invertibility ───────────────────────────

@fn("WeakStationarity", area=AREA, status="added",
    notes="WeakStationarity[proc]: True if proc is weakly stationary, else False. Time series processes: every zero "
          "of the AR polynomial lies outside the unit circle (decided exactly by the Schur-Cohn/Levinson step-down "
          "test for exact coefficients), integrated processes are not; ARCH/GARCH: Sum alpha + Sum beta < 1; "
          "named processes by their definition (OrnsteinUhlenbeckProcess without x0, BernoulliProcess, "
          "WhiteNoiseProcess with finite variance, FractionalGaussianNoiseProcess are; Wiener, Poisson, ... are not). "
          "Symbolic coefficients stay unevaluated (Wolfram gives inequality conditions). Markov, queueing and SDE "
          "processes raise NotImplementedError. Check: exact; reference (Wolfram documentation, numpy roots) in "
          "tests.")
def _weakstat(proc: Expr) -> Boolean:
    pr = proc_of(proc, "WeakStationarity")
    if isinstance(pr, (Markov, Queue, Ito, Transformed, Renewal)):
        raise NotImplementedError(f"WeakStationarity: not supported for {pr.name}")
    return bool(pr.stationary())


def _need_arma(proc, fname):
    pr = proc_of(proc, fname)
    if not isinstance(pr, ARMA):
        raise TypeError(f"{fname}: {pr.name} is not a time series (ARMA-type) process")
    return pr


@fn("TimeSeriesInvertibility", area=AREA, status="added",
    notes="TimeSeriesInvertibility[tproc]: True if every zero of the MA polynomial 1 + Sum b_j z^j (times the "
          "seasonal MA polynomial) lies outside the unit circle (exact Schur-Cohn test), else False; pure AR "
          "processes are invertible. Symbolic coefficients stay unevaluated; non time series processes raise "
          "TypeError. Check: exact; reference (Wolfram documentation, numpy roots) in tests.")
def _tsinv(proc: Expr) -> Boolean:
    return bool(_need_arma(proc, "TimeSeriesInvertibility").invertible())


@fn("ToInvertibleTimeSeries", area=AREA, status="added",
    notes="ToInvertibleTimeSeries[tproc] for MAProcess, ARMAProcess, ARIMAProcess and FARIMAProcess: reflects the "
          "zeros of the MA polynomial inside the unit circle to 1/Conjugate[z] and rescales the noise variance by "
          "Product |z|^-2, which preserves the mean and covariance function; exact (radicals) for exact "
          "coefficients, machine numbers otherwise. Zeros on the unit circle raise ValueError; seasonal processes "
          "and MA polynomials without radical zeros raise NotImplementedError; symbolic coefficients stay "
          "unevaluated. Check: self-check (the autocovariances at lags 0..q of the result equal the original ones, "
          "else ArithmeticError); reference (Wolfram documentation) in tests.")
def _toinv(proc: Expr) -> Expr:
    fname = "ToInvertibleTimeSeries"
    pr = _need_arma(proc, fname)
    if pr.name in ("SARMAProcess", "SARIMAProcess"):
        raise NotImplementedError(f"{fname}: seasonal processes are not supported")
    th = pr.theta
    if not all(_num(c) for c in th + [pr.v]):
        raise Unevaluated()
    if not th or pr.invertible():
        return proc
    inexact = _inexact(*th, pr.v)
    z = sp.Symbol("z")
    poly = sp.Poly(sum(c * z ** (j + 1) for j, c in enumerate(th)) + 1, z)
    if inexact:
        import numpy as np
        rts = [complex(r) for r in np.roots([complex(_f(c)) for c in poly.all_coeffs()])]
        scale = 1.0
        new = []
        for r in rts:
            if abs(abs(r) - 1) < 1e-12:
                raise ValueError(f"{fname}: a zero lies on the unit circle; no invertible representation exists")
            if abs(r) < 1:
                scale /= abs(r) ** 2
                r = 1 / r.conjugate()
            new.append(r)
        coeffs = [1.0 + 0j]
        for r in new:
            coeffs = [a - b / r for a, b in zip(coeffs + [0], [0] + coeffs)]
        nth = [sp.Float(c.real) for c in coeffs[1:]]
        nv = sp.Float(_f(pr.v) * scale)
    else:
        rmap = sp.roots(poly)
        if sum(rmap.values()) != poly.degree():
            raise NotImplementedError(f"{fname}: the zeros of the MA polynomial cannot be found in radicals")
        expr, scale = sp.Integer(1), sp.Integer(1)
        for r, mult in rmap.items():
            a = sp.Abs(r)
            if sp.simplify(a - 1) == 0:
                raise ValueError(f"{fname}: a zero lies on the unit circle; no invertible representation exists")
            if (a < 1) is sp.true or bool(a < 1):
                scale *= a ** (-2 * mult)
                r = 1 / sp.conjugate(r)
            expr *= (1 - z / r) ** mult
        cs = sp.Poly(sp.expand(expr), z).all_coeffs()[::-1]
        nth = [sp.nsimplify(sp.simplify(c)) if sp.simplify(c).is_rational else sp.simplify(c) for c in cs[1:]]
        nv = sp.simplify(pr.v * scale)
    # self-check: same autocovariances
    q = len(th)
    full_old = [sp.Integer(1)] + list(th)
    full_new = [sp.Integer(1)] + list(nth)
    for k in range(q + 1):
        g0 = pr.v * sum(full_old[j] * full_old[j + k] for j in range(q + 1 - k))
        g1 = nv * sum(full_new[j] * full_new[j + k] for j in range(q + 1 - k))
        if (abs(_f(g0) - _f(g1)) > 1e-8 * max(1.0, abs(_f(g0)))) if inexact else sp.simplify(g0 - g1) != 0:
            raise ArithmeticError(f"{fname}: result failed verification")
    a = list(proc.args)
    # positions: MA list and v are the last list before v (and v is last) in every supported form
    a[-1] = R(nv)
    a[-2] = R(nth)
    return Expr(proc.head, *a)


# ─────────────────────────── stationary distributions ───────────────────────────

@fn("StationaryDistribution", area=AREA, status="added",
    notes="StationaryDistribution[proc]: the limiting (stationary) distribution. OrnsteinUhlenbeckProcess (with or "
          "without x0): NormalDistribution[mu, sigma/Sqrt[2 theta]]; CoxIngersollRossProcess: GammaDistribution[2 "
          "theta mu/sigma^2, sigma^2/(2 theta)]; TelegraphProcess: 2 x - 1 with x ~ BernoulliDistribution[1/2]; "
          "Bernoulli, white noise, fractional Gaussian noise and weakly stationary ARMA-type processes: their slice; "
          "finite Markov processes: p0 times the Cesaro limit of the transition matrix, as "
          "EmpiricalDistribution[probabilities -> Range[n]] (exact linear algebra); M/M/c/k queues: the stationary "
          "system size (GeometricDistribution[1 - rho] for M/M/1, EmpiricalDistribution for finite capacity). "
          "Processes without a stationary distribution raise ValueError; others NotImplementedError. "
          "Check: exact; self-check (pi.m == pi for Markov chains, else ArithmeticError); reference in tests.")
def _statdist(proc: Expr) -> Expr:
    fname = "StationaryDistribution"
    pr = proc_of(proc, fname)
    if isinstance(pr, OU):
        m, s, th = pr.p[:3]
        return _dist("NormalDistribution", m, s / sp.sqrt(2 * th))
    if isinstance(pr, CIR):
        m, s, th, x0 = pr.p
        return _dist("GammaDistribution", 2 * th * m / s ** 2, s ** 2 / (2 * th))
    if isinstance(pr, Telegraph):
        x = Symbol("x")
        return Expr("TransformedDistribution", Expr("Plus", -1, Expr("Times", 2, x)),
                    Expr("Distributed", x, _dist("BernoulliDistribution", sp.Rational(1, 2))))
    if isinstance(pr, (Bernoulli, WhiteNoise, FGN)):
        return pr.slice(sp.Integer(1))
    if isinstance(pr, ARMA):
        if pr.stationary() is False:
            raise ValueError(f"{fname}: {pr.name} is not weakly stationary")
        return pr.slice(sp.Integer(1))
    if isinstance(pr, Markov):
        for v in pr.p:
            if not _num(v):
                raise Unevaluated()
        d = pr.stationary_dist()
        pi = sp.Matrix([[S(w) for w in d.args[0].args[0].args]])
        _, rec, _ = pr.classes()
        if len(rec) == 1:
            chk = pi * pr.m - pi if pr.dtime else pi * pr.q
            if any((abs(_f(c)) > 1e-9) if _inexact(c) else sp.simplify(c) != 0 for c in chk):
                raise ArithmeticError(f"{fname}: result failed verification")
        return d
    if isinstance(pr, Queue):
        return pr.stationary_dist(fname)
    if isinstance(pr, GARCH):
        raise NotImplementedError(f"{fname}: the stationary distribution of {pr.name} has no closed form")
    if isinstance(pr, (Ito, Transformed, Renewal)):
        raise NotImplementedError(f"{fname}: not supported for {pr.name}")
    if hasattr(pr, "stationary_distribution"):        # X02: hidden Markov processes, queueing networks
        return pr.stationary_distribution(fname)
    raise ValueError(f"{fname}: {pr.name} has no stationary distribution")


# ─────────────────────────── Markov process properties ───────────────────────────

_MPP = ["InitialProbabilities", "TransitionMatrix", "TransitionRateMatrix", "TransitionRateVector",
        "HoldingTimeMean", "HoldingTimeVariance", "CommunicatingClasses", "RecurrentClasses", "TransientClasses",
        "AbsorbingClasses", "PeriodicClasses", "Periods", "Irreducible", "Aperiodic", "Primitive",
        "LimitTransitionMatrix", "ReachabilityProbability", "Reversible", "TransientVisitMean"]


def _markov_prop(pr, prop, fname):
    n = pr.n
    if prop == "InitialProbabilities":
        return R(list(pr.p0))
    if prop == "TransitionMatrix":
        return R(pr.m.tolist())
    if prop in ("TransitionRateMatrix", "TransitionRateVector"):
        if pr.dtime:
            return Expr("Missing", "NotApplicable")
        return R(pr.q.tolist()) if prop == "TransitionRateMatrix" else R(list(pr.mu))
    if prop in ("HoldingTimeMean", "HoldingTimeVariance"):
        out = []
        for i in range(n):
            stay = pr.m[i, i] if pr.dtime else sp.Integer(0)
            rate = sp.Integer(1) if pr.dtime else pr.mu[i]
            if (sp.simplify(stay - 1) == 0) or rate == 0:
                out.append(INF)
                continue
            if pr.dtime:
                v = 1 / (1 - stay) if prop == "HoldingTimeMean" else stay / (1 - stay) ** 2
            else:
                v = 1 / rate if prop == "HoldingTimeMean" else 1 / rate ** 2
            out.append(R(sp.simplify(v)))
        return L(*out)
    cls, rec, reach = pr.classes()
    one = lambda c: L(*(i + 1 for i in sorted(c)))  # noqa: E731
    order = lambda cs: sorted(cs, key=min)  # noqa: E731
    if prop == "CommunicatingClasses":
        return L(*(one(c) for c in order(cls)))
    if prop == "RecurrentClasses":
        return L(*(one(c) for c in order(rec)))
    if prop == "TransientClasses":
        return L(*(one(c) for c in order([c for c in cls if c not in rec])))
    if prop == "AbsorbingClasses":
        return L(*(one(c) for c in order([c for c in rec if len(c) == 1])))
    per = [(c, pr.period(c)) for c in order(cls)]
    if prop == "PeriodicClasses":
        return L(*(one(c) for c, p in per if p > 1 and c in rec))
    if prop == "Periods":
        return L(*(p for c, p in per if p > 1 and c in rec))
    if prop == "Irreducible":
        return len(cls) == 1
    if prop == "Aperiodic":
        return all(p == 1 for c, p in per if c in rec)
    if prop == "Primitive":
        return len(cls) == 1 and per[0][1] == 1
    if prop == "LimitTransitionMatrix":
        return R(pr.limit_matrix().tolist())
    if prop == "ReachabilityProbability":
        # probability of ever reaching each state from the initial distribution
        out = []
        for j in range(n):
            h = _hit_prob(pr, j)
            out.append(sp.simplify(sum(pr.p0[i] * h[i] for i in range(n))))
        return R(out)
    if prop == "Reversible":
        if len(rec) != 1 or len(cls) != 1:
            return False
        pi = pr.class_stationary(rec[0])
        g = pr.graph_matrix()
        for i in range(n):
            for j in range(n):
                d = pi[i] * g[i, j] - pi[j] * g[j, i]
                if (abs(_f(d)) > 1e-12) if _inexact(d) else sp.simplify(d) != 0:
                    return False
        return True
    if prop == "TransientVisitMean":
        recset = {i for c in rec for i in c}
        tr = [i for i in range(n) if i not in recset]
        if not tr:
            return L()
        Q = sp.Matrix(len(tr), len(tr), lambda a, b: pr.m[tr[a], tr[b]])
        N = (sp.eye(len(tr)) - Q).inv()
        a0 = sp.Matrix([[pr.p0[i] for i in tr]])
        v = a0 * N
        return R([sp.simplify(x) for x in v])
    raise ValueError(f"{fname}: unknown property {prop!r}")


def _hit_prob(pr, j):
    """h[i] = probability of ever visiting state j starting from i (time 0 counts)"""
    n = pr.n
    _, _, reach = pr.classes()
    can = [reach[i][j] for i in range(n)]
    unk = [i for i in range(n) if i != j and can[i]]
    h = [sp.Integer(0)] * n
    h[j] = sp.Integer(1)
    if unk:
        A = sp.eye(len(unk)) - sp.Matrix(len(unk), len(unk), lambda a, b: pr.m[unk[a], unk[b]])
        b = sp.Matrix([pr.m[i, j] for i in unk])
        sol = A.solve(b)
        for a, i in enumerate(unk):
            h[i] = sp.simplify(sol[a])
    return h


@fn("MarkovProcessProperties", area=AREA, status="added",
    notes="MarkovProcessProperties[mproc, prop] for DiscreteMarkovProcess and ContinuousMarkovProcess with numeric "
          "matrices: \"InitialProbabilities\", \"TransitionMatrix\" (embedded chain for continuous time), "
          "\"TransitionRateMatrix\", \"TransitionRateVector\", \"HoldingTimeMean\", \"HoldingTimeVariance\" (Infinity "
          "for absorbing states), \"CommunicatingClasses\", \"RecurrentClasses\", \"TransientClasses\", "
          "\"AbsorbingClasses\", \"PeriodicClasses\", \"Periods\", \"Irreducible\", \"Aperiodic\", \"Primitive\", "
          "\"LimitTransitionMatrix\" (Cesaro limit), \"ReachabilityProbability\", \"Reversible\", "
          "\"TransientVisitMean\"; MarkovProcessProperties[mproc, \"Properties\"] lists them and "
          "MarkovProcessProperties[mproc] gives an association of the structural properties (Wolfram shows a "
          "summary table). Classes are listed by their smallest state. Unknown properties raise ValueError; "
          "\"SummaryTable\" and \"Description\" raise NotImplementedError. Check: exact; reference (Wolfram "
          "documentation examples) in tests.")
def _mpp(proc: Expr, prop: String = None, extra: String = None) -> Expr:
    fname = "MarkovProcessProperties"
    pr = proc_of(proc, fname)
    if not isinstance(pr, Markov):
        raise TypeError(f"{fname}: {pr.name} is not a finite Markov process")
    for v in pr.p:
        if not _num(v):
            raise Unevaluated()
    if extra is not None:
        raise NotImplementedError(f"{fname}: property descriptions are not supported")
    if prop is None:
        return Expr("Association", *(rule(k, _markov_prop(pr, k, fname)) for k in
                                     ("CommunicatingClasses", "RecurrentClasses", "TransientClasses",
                                      "AbsorbingClasses", "PeriodicClasses", "Periods", "Irreducible", "Aperiodic",
                                      "Primitive")))
    if prop == "Properties":
        return L(*_MPP)
    if prop == "SummaryTable":
        raise NotImplementedError(f"{fname}: \"SummaryTable\" is not supported")
    if not isinstance(prop, str):
        raise TypeError(f"{fname}: the property must be a string")
    return _markov_prop(pr, prop, fname)


# ─────────────────────────── first passage times ───────────────────────────

def _fpt_parts(d, fname):
    if len(d.args) != 2:
        raise ValueError(f"{fname}: FirstPassageTimeDistribution expects [mproc, f]")
    pr = proc_of(d.args[0], fname)
    if not isinstance(pr, Markov):
        raise TypeError(f"{fname}: FirstPassageTimeDistribution needs a finite Markov process")
    for v in pr.p:
        if not _num(v):
            raise Unevaluated()
    f = d.args[1]
    tgt = [f] if not is_list(f) else list(f.args)
    if not tgt or not all(isinstance(s, int) and not isinstance(s, bool) and 1 <= s <= pr.n for s in tgt):
        raise ValueError(f"{fname}: the final states must be states 1..{pr.n}")
    tset = {s - 1 for s in tgt}
    if any(pr.p0[i] != 0 for i in tset):
        raise NotImplementedError(f"{fname}: the initial distribution must not charge the final states")
    _, _, reach = pr.classes()
    start = [i for i in range(pr.n) if pr.p0[i] != 0]
    # states reachable from the start without entering the targets
    R_ = set(start)
    frontier = list(start)
    g = pr.graph_matrix()
    while frontier:
        i = frontier.pop()
        for j in range(pr.n):
            if j not in tset and j not in R_ and i != j and g[i, j] != 0:
                R_.add(j)
                frontier.append(j)
    idx = sorted(R_)
    if pr.dtime:
        T = sp.Matrix(len(idx), len(idx), lambda a, b: pr.m[idx[a], idx[b]])
    else:
        T = sp.Matrix(len(idx), len(idx), lambda a, b: pr.q[idx[a], idx[b]])
    alpha = sp.Matrix([[pr.p0[i] for i in idx]])
    M = sp.eye(len(idx)) - T if pr.dtime else -T
    if M.det() == 0:
        raise NotImplementedError(f"{fname}: the final states are not reached with probability 1")
    return pr, T, alpha, M


def _fpt_impl(which):
    def impl(d: Expr, *rest: Expr) -> Expr:
        pr, T, alpha, M = _fpt_parts(d, which)
        one = sp.ones(T.rows, 1)
        N = M.inv()
        if which == "Mean":
            v = (alpha * N * one)[0]
        elif which in ("Variance", "StandardDeviation"):
            m1 = (alpha * N * one)[0]
            m2 = (alpha * (sp.eye(T.rows) + T) * N * N * one)[0] if pr.dtime else 2 * (alpha * N * N * one)[0]
            v = m2 - m1 ** 2
            if which == "StandardDeviation":
                v = sp.sqrt(v)
        else:
            if len(rest) != 1:
                raise ValueError(f"{which}: expected {which}[dist, x]")
            x = _p(rest[0], which, "x")
            if not _num(x):
                if pr.dtime or not (isinstance(x, sp.Symbol) and len(x.free_symbols) == 1):
                    raise Unevaluated()
                E = (T * x).exp()
                v = (alpha * E * (-T) * one)[0] if which == "PDF" else 1 - (alpha * E * one)[0]
                return _out(sp.Piecewise((sp.simplify(v), x >= 0), (0, True)), d, *rest)
            if pr.dtime:
                if not x.is_Integer:
                    return sp.Integer(0) if which == "PDF" else _fpt_impl("CDF")(d, R(sp.floor(x)))
                k = int(x)
                if k < 1:
                    return sp.Integer(0)
                if which == "PDF":
                    t0 = (sp.eye(T.rows) - T) * one
                    v = (alpha * T ** (k - 1) * t0)[0]
                else:
                    v = 1 - (alpha * T ** k * one)[0]
            else:
                if x < 0:
                    return sp.Integer(0)
                E = (T * x).exp()
                v = (alpha * E * (-T) * one)[0] if which == "PDF" else 1 - (alpha * E * one)[0]
        return _out(sp.simplify(v) if not _inexact(v) else v, d, *rest)
    return impl


@fn("FirstPassageTimeDistribution", area=AREA, status="added",
    notes="FirstPassageTimeDistribution[mproc, f]: the (phase-type) distribution of the first time a finite Markov "
          "process enters the state f or one of the states f = {f1, ...}; inert, with Mean, Variance, "
          "StandardDeviation, PDF and CDF in closed form (fundamental-matrix formulas, exact); initial "
          "probability on f, or f not reached with probability 1, raise NotImplementedError. "
          "Check: exact; reference (Wolfram documentation mean first passage times, hand-computed values) in tests.")
def _fptd(proc: Expr, f: Integer | ListOf[Integer]) -> Expr:
    _fpt_parts(Expr("FirstPassageTimeDistribution", proc, f), "FirstPassageTimeDistribution")
    raise Unevaluated()


def _fpt_first(args):
    return bool(args) and isinstance(args[0], Expr) and args[0].head == "FirstPassageTimeDistribution"


for _w in ("Mean", "Variance", "StandardDeviation", "PDF", "CDF"):
    W.extend(_w, _fpt_impl(_w), _fpt_first, f"{_w}[FirstPassageTimeDistribution[...]] (phase-type formulas)")


# ─────────────────────────── queue properties ───────────────────────────

_QP = ["ArrivalRate", "ServiceRate", "ArrivalDistribution", "ServiceDistribution", "ServiceChannels",
       "SystemCapacity", "InitialState", "StationarySystem", "UtilizationFactor", "Throughput", "MeanSystemSize",
       "MeanQueueSize", "MeanSystemTime", "MeanQueueTime"]


def _queue_prop(q, prop, fname):
    if prop == "ArrivalRate":
        return R(q.lam) if q.lam is not None else _ev(Expr("Power", Expr("Mean", q.adist), -1))
    if prop == "ServiceRate":
        return R(q.mu) if q.mu is not None else _ev(Expr("Power", Expr("Mean", q.sdist), -1))
    if prop == "ArrivalDistribution":
        return _dist("ExponentialDistribution", q.lam) if q.lam is not None else q.adist
    if prop == "ServiceDistribution":
        return _dist("ExponentialDistribution", q.mu) if q.mu is not None else q.sdist
    if prop == "ServiceChannels":
        return q.c
    if prop == "SystemCapacity":
        return q.k
    if prop == "InitialState":
        return q.x0
    lam = S(_queue_prop(q, "ArrivalRate", fname))
    mu = S(_queue_prop(q, "ServiceRate", fname))
    if prop == "StationarySystem":
        if q.finite:
            return True
        r = lam / (q.c * mu)
        if not _num(r):
            raise Unevaluated()
        return bool(r < 1)
    if prop not in _QP:
        if prop in ("QueueNotation", "QueueDiagram", "SummaryTable", "NetworkType", "NodeCount", "SelectedNode",
                    "DataSource"):
            raise NotImplementedError(f"{fname}: the property {prop!r} is not supported")
        raise ValueError(f"{fname}: unknown property {prop!r}")
    if q.markovian():
        m = q.measures(fname)
    elif q.adist is None and q.c == 1 and not q.finite:
        # M/G/1: Pollaczek-Khinchine
        sd = W.dist_of(q.sdist, fname)
        es, es2 = sd.mean(), W.raw_moment(sd, 2)
        rho = q.lam * es
        if _num(rho) and not rho < 1:
            m = None
        else:
            Lq = q.lam ** 2 * es2 / (2 * (1 - rho))
            Wq = Lq / q.lam
            m = {"MeanQueueSize": Lq, "MeanQueueTime": Wq, "MeanSystemTime": Wq + es,
                 "MeanSystemSize": q.lam * (Wq + es), "Throughput": q.lam, "UtilizationFactor": rho}
            m = {k: sp.simplify(v) for k, v in m.items()}
    else:
        raise NotImplementedError(f"{fname}: stationary measures are only supported for M/M/c/k and M/G/1 queues")
    if m is None:
        raise ValueError(f"{fname}: the queue does not reach a steady state, so {prop!r} is not available")
    return R(m[prop])


QPROPS_HOOK = None      # X02: fn(q, prop, extra) for QueueingNetworkProcess / {net, i}; None when not handled


@fn("QueueProperties", area=AREA, status="added",
    notes="QueueProperties[qproc, prop] for QueueingProcess: \"ArrivalRate\", \"ServiceRate\", \"ArrivalDistribution\", "
          "\"ServiceDistribution\", \"ServiceChannels\", \"SystemCapacity\", \"InitialState\", \"StationarySystem\" and "
          "the steady-state measures \"UtilizationFactor\" (mean busy servers / c), \"Throughput\", "
          "\"MeanSystemSize\", \"MeanQueueSize\", \"MeanSystemTime\", \"MeanQueueTime\" (exact M/M/c/k formulas, "
          "Pollaczek-Khinchine for M/G/1); QueueProperties[qproc] gives an association (Wolfram shows a table); "
          "QueueProperties[qproc, \"Properties\"] the list. Non-stationary queues raise ValueError for the "
          "steady-state measures; other queue types, QueueingNetworkProcess, data input and diagram/notation "
          "properties raise NotImplementedError. Check: exact; reference (Little's law L = lambda W, textbook "
          "M/M/1 and M/M/c values) in tests.")
def _qprops(q: Expr, prop: String = None, extra: String = None) -> Expr:
    fname = "QueueProperties"
    if QPROPS_HOOK is not None:
        r = QPROPS_HOOK(q, prop, extra)
        if r is not None:
            return r
    if not (isinstance(q, Expr) and q.head == "QueueingProcess"):
        if isinstance(q, Expr) and q.head == "QueueingNetworkProcess":
            raise NotImplementedError(f"{fname}: queueing networks are not supported")
        if isinstance(q, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {q} is not a queueing process")
    pr = proc_of(q, fname)
    if extra is not None:
        raise NotImplementedError(f"{fname}: property descriptions are not supported")
    if prop is None:
        out = []
        for k in _QP:
            try:
                out.append(rule(k, _queue_prop(pr, k, fname)))
            except (ValueError, NotImplementedError):
                out.append(rule(k, Expr("Missing", "NotAvailable")))
        return Expr("Association", *out)
    if prop == "Properties":
        return L(*_QP)
    if not isinstance(prop, str):
        raise TypeError(f"{fname}: the property must be a string")
    return _queue_prop(pr, prop, fname)


# ─────────────────────────── simulation ───────────────────────────

_MAX_POINTS = 10 ** 6


def _sim_times(pr, spec, fname):
    """(times as SymPy values, uses_jumps, (tmin, tmax))"""
    if not is_list(spec) or not 1 <= len(spec.args) <= 3:
        raise ValueError(f"{fname}: the time specification must be {{tmax}}, {{tmin, tmax}} or {{tmin, tmax, dt}}")
    vals = [_p(v, fname, "time") for v in spec.args]
    if not all(_num(v) for v in vals):
        raise Unevaluated()
    if not all(v.is_real for v in vals):
        raise ValueError(f"{fname}: times must be real")
    if len(vals) == 1:
        tmin = pr.t0 if not pr.dtime else (pr.t0 if pr.t0 is not None else sp.Integer(0))
        tmax = vals[0]
    else:
        tmin, tmax = vals[0], vals[1]
    if tmax < tmin:
        raise ValueError(f"{fname}: tmax must be at least tmin")
    pr.check_time(tmin, fname)
    pr.check_time(tmax, fname)
    if pr.dtime:
        if len(vals) == 3:
            dt = vals[2]
            if not (dt.is_Integer and dt > 0):
                raise ValueError(f"{fname}: discrete-time processes need a positive integer step, not {dt}")
        else:
            dt = sp.Integer(1)
        n = int((tmax - tmin) // dt) + 1
        if n > _MAX_POINTS:
            raise NotImplementedError(f"{fname}: more than {_MAX_POINTS} time steps are not supported")
        return [tmin + k * dt for k in range(n)], False, (tmin, tmax)
    if len(vals) < 3:
        if pr.jump:
            return None, True, (tmin, tmax)
        raise ValueError(f"{fname}: {pr.name} is a continuous-time process; give {{tmin, tmax, dt}}")
    dt = vals[2]
    if not dt > 0:
        raise ValueError(f"{fname}: the time step must be positive")
    n = int(sp.floor((tmax - tmin) / dt + sp.Rational(1, 10 ** 9))) + 1
    if n > _MAX_POINTS:
        raise NotImplementedError(f"{fname}: more than {_MAX_POINTS} time steps are not supported")
    return [tmin + k * dt for k in range(n)], False, (tmin, tmax)


def _val_out(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return sp.Float(v)
    import numpy as np
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return sp.Float(float(v))
    return v


def _one_path(pr, rng, times, jumps, rng_ab, fname):
    if jumps:
        t0, t1 = rng_ab
        ev = pr.jumps(rng, _f(t0), _f(t1))
        inexact = _inexact(t0, t1)
        out = []
        for k, (t, v) in enumerate(ev):
            tt = (t0 if k == 0 else sp.Float(t))
            out.append((R(tt), R(_val_out(v)) if not isinstance(_val_out(v), Expr) else v))
        if not inexact or True:
            last_v = out[-1][1]
            if len(ev) == 1 or ev[-1][0] < _f(t1):
                out.append((R(t1), last_v))
        return out
    ft = [int(t) if pr.dtime else _f(t) for t in times]
    vals = pr.path(rng, ft)
    res = []
    for t, v in zip(times, vals):
        vv = _val_out(v)
        res.append((R(t), vv if isinstance(vv, Expr) else R(vv)))
    return res


@fn("RandomFunction", area=AREA, status="added",
    notes="RandomFunction[proc, {tmax}], RandomFunction[proc, {tmin, tmax}] (jump processes: exact event times), "
          "RandomFunction[proc, {tmin, tmax, dt}] and RandomFunction[proc, spec, n] simulate paths of a random "
          "process and return TemporalData[{path, ...}] (paths of {t, x} pairs; jump-process paths carry "
          "ResamplingMethod -> {\"Interpolation\", InterpolationOrder -> 0}). Simulation uses the exact transition "
          "law (Gaussian increments, exact Cholesky/Durbin-Levinson sampling for fractional and ARMA processes, "
          "noncentral chi-square for CIR, exponential holding times for Markov chains and M/M/c queues, event "
          "simulation for general queues and renewal processes) except Ito/Stratonovich processes, which use "
          "Euler-Maruyama on the grid dt. Reproducible with SeedRandom (the random stream differs from Wolfram's). "
          "Discrete-time processes reject non-integer steps; continuous processes without jumps require dt. "
          "Symbolic parameters stay unevaluated; WorkingPrecision other than Automatic and Method raise "
          "NotImplementedError. Check: statistical (sample means and variances of many paths against Mean and "
          "Variance of the slice) in tests.")
def _randomfunction(proc: Expr, spec: ListOf[Real] = None, n: Integer = None, *opts: Expr) -> Expr:
    fname = "RandomFunction"
    from .wl_processes_ts import make_td
    for o in opts:
        if isinstance(o, Expr) and o.head in ("Rule", "RuleDelayed") and isinstance(o.args[0], Symbol):
            if o.args[0].name == "WorkingPrecision" and o.args[1] != Symbol("Automatic"):
                raise NotImplementedError(f"{fname}: WorkingPrecision is not supported")
            if o.args[0].name == "Method":
                raise NotImplementedError(f"{fname}: the Method option is not supported")
            continue
        raise ValueError(f"{fname}: unexpected argument {o}")
    if isinstance(n, Expr) and n.head in ("Rule", "RuleDelayed"):
        return _randomfunction(proc, spec, None, n, *opts)
    pr = proc_of(proc, fname)
    if spec is None:
        raise ValueError(f"{fname}: a time specification {{tmin, tmax}} or {{tmin, tmax, dt}} is required")
    if n is None:
        count = 1
    else:
        if isinstance(n, Symbol):
            raise Unevaluated()
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError(f"{fname}: the number of paths must be a positive integer")
        if n < 1:
            raise ValueError(f"{fname}: the number of paths must be positive")
        count = n
    if not pr.numeric():
        raise Unevaluated()
    times, jumps, ab = _sim_times(pr, spec, fname)
    if times is not None and len(times) * count > _MAX_POINTS:
        raise NotImplementedError(f"{fname}: more than {_MAX_POINTS} simulated values are not supported")
    rng = _np_rng()
    paths = [_one_path(pr, rng, times, jumps, ab, fname) for _ in range(count)]
    o = {}
    if pr.jump and not pr.dtime:
        o["ResamplingMethod"] = L("Interpolation", rule(Symbol("InterpolationOrder"), 0))
    return make_td(paths, o)


# ─────────────────────────── forecasting ───────────────────────────

def _series_data(data, fname):
    """list of paths [(t, x) sympy pairs] from a list, list of pairs, TimeSeries or TemporalData"""
    from . import wl_processes_ts as T
    if is_list(data):
        if not data.args:
            raise ValueError(f"{fname}: the data must not be empty")
        if all(is_list(p) and len(p.args) == 2 for p in data.args):
            path = [(_p(p.args[0], fname, "time"), _p(p.args[1], fname, "value")) for p in data.args]
        else:
            path = [(sp.Integer(i), _p(x, fname, "value")) for i, x in enumerate(data.args)]
        paths = [path]
    elif isinstance(data, Expr) and data.head == "TemporalData":
        paths = [[(sp.sympify(S(t)), sp.sympify(S(v))) for t, v in p] for p in T.td_paths(data)[0]]
    elif isinstance(data, Expr) and data.head in ("TimeSeries", "EventSeries"):
        ser = T.series_of(data, fname)
        paths = [[(sp.sympify(S(t)), sp.sympify(S(v))) for t, v in ser[1]]]
    elif isinstance(data, Symbol):
        raise Unevaluated()
    else:
        raise TypeError(f"{fname}: the data must be a list of values, a list of {{t, x}} pairs, a TimeSeries or "
                        "TemporalData")
    for p in paths:
        for t, x in p:
            if not (_num(t) and _num(x)):
                raise Unevaluated()
            if not (t.is_real and x.is_real):
                raise ValueError(f"{fname}: data must be real")
    return paths


def _steps(spec, fname):
    if spec is None:
        return [1], True
    if isinstance(spec, int) and not isinstance(spec, bool):
        if spec < 1:
            raise ValueError(f"{fname}: the step must be a positive integer")
        return [spec], True
    if is_list(spec):
        a = spec.args
        if len(a) == 1 and is_list(a[0]):
            ks = list(a[0].args)
        elif len(a) == 1:
            ks = list(range(1, a[0] + 1)) if isinstance(a[0], int) else None
        elif len(a) == 2 and all(isinstance(x, int) for x in a):
            ks = list(range(a[0], a[1] + 1))
        else:
            ks = None
        if ks is None or not ks or not all(isinstance(k, int) and not isinstance(k, bool) and k >= 1 for k in ks):
            raise ValueError(f"{fname}: steps must be positive integers given as k, {{kmax}}, {{kmin, kmax}} or "
                             "{{{k1, k2, ...}}}")
        return ks, False
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: invalid forecast specification {spec}")


def _arma_predict(pr, xs, ks, exact):
    """E[x_{n-1+k} | x_0..x_{n-1}] for a weakly stationary (or pure AR) ARMA pr"""
    n = len(xs)
    p = len(pr.phi)
    pure_ar = not pr.theta and pr.frac is None
    if pure_ar and n >= p:
        hist = list(xs)
        out = {}
        for k in range(1, max(ks) + 1):
            nxt = pr.c + sum(pr.phi[i] * hist[-1 - i] for i in range(p))
            hist.append(sp.nsimplify(nxt) if exact else sp.Float(_f(nxt)))
            out[k] = hist[-1]
        return [out[k] for k in ks]
    if pr.stationary() is not True:
        raise NotImplementedError("TimeSeriesForecast: forecasting non-stationary ARMA processes with an MA part "
                                  "is not supported")
    mu = pr.mean(sp.Integer(0))
    K = max(ks)
    if exact and n <= 40:
        g = [pr.acvf(h) for h in range(n + K)]
        G = sp.Matrix(n, n, lambda i, j: g[abs(i - j)])
        w = G.LUsolve(sp.Matrix([x - mu for x in xs]))
        return [sp.simplify(mu + sum(g[n - 1 + k - i] * w[i] for i in range(n))) for k in ks]
    import numpy as np
    g = np.array([_f(pr.acvf(h)) for h in range(n + K)])
    idx = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    w = np.linalg.solve(g[idx], np.array([_f(x - mu) for x in xs]))
    m = _f(mu)
    return [sp.Float(m + float(sum(g[n - 1 + k - i] * w[i] for i in range(n)))) for k in ks]


TSM_FORECAST_HOOK = None    # X02: fn(TimeSeriesModel, spec, opts)


@fn("TimeSeriesForecast", area=AREA, status="added",
    notes="TimeSeriesForecast[tproc, data, spec] for ARProcess, MAProcess, ARMAProcess, ARIMAProcess, SARMAProcess, "
          "SARIMAProcess, ARCHProcess and GARCHProcess with numeric parameters: the conditional expectation "
          "E[x[m + k] | x[0] = x0, ..., x[m] = xm]. Weakly stationary processes use the exact Gaussian (covariance) "
          "predictor (exact arithmetic for exact data of length <= 40, machine numbers otherwise); pure AR processes "
          "with at least p observations use the AR recursion; (S)ARIMA processes difference the data, forecast the "
          "stationary part and integrate; (G)ARCH forecasts are 0. spec: k (default 1, gives the value), {kmax}, "
          "{kmin, kmax} or {{k1, ...}} (give TemporalData with the forecasts at times m + k). Data: a list of "
          "values (times 0, 1, ...), {t, x} pairs with unit spacing, a TimeSeries or TemporalData (first path). "
          "TimeSeriesForecast[tsmodel, spec] forecasts with a TimeSeriesModel's fitted process (X02). Mean squared "
          "errors (MetaInformation) and the Method option raise NotImplementedError. Check: exact; reference (the "
          "AR recursion, hand-computed MA(1) predictor) in tests.")
def _tsforecast(proc: Expr, data: Expr, spec: Expr = None, *opts: Expr) -> Expr:
    fname = "TimeSeriesForecast"
    from .wl_processes_ts import make_td
    if opts:
        raise NotImplementedError(f"{fname}: options are not supported")
    if isinstance(proc, Expr) and proc.head == "TimeSeriesModel":
        if TSM_FORECAST_HOOK is None or spec is not None:
            raise NotImplementedError(f"{fname}: use TimeSeriesForecast[tsmodel, spec]")
        return TSM_FORECAST_HOOK(proc, data, ())
    pr = proc_of(proc, fname)
    if not isinstance(pr, (ARMA, GARCH)):
        raise TypeError(f"{fname}: {pr.name} is not a time series process")
    ks, scalar = _steps(spec, fname)
    path = _series_data(data, fname)[0]
    ts = [t for t, _ in path]
    for a, b in zip(ts, ts[1:]):
        if sp.simplify(b - a - 1) != 0:
            raise NotImplementedError(f"{fname}: data must be regularly spaced with time step 1")
    if not pr.numeric():
        raise Unevaluated()
    xs = [x for _, x in path]
    exact = not _inexact(*xs, *pr.flat())
    if isinstance(pr, GARCH):
        preds = [sp.Integer(0) if exact else sp.Float(0)] * len(ks)
    else:
        dpoly = pr.diff
        if len(dpoly) > 1:
            d = len(dpoly) - 1
            if len(xs) <= d:
                raise ValueError(f"{fname}: need more than {d} observations to difference the data")
            ws = [sum(dpoly[j] * xs[i - j] for j in range(d + 1)) for i in range(d, len(xs))]
            sub = pr.stationary_part()
            K = max(ks)
            wf = _arma_predict(sub, ws, list(range(1, K + 1)), exact)
            hist = list(xs)
            for k in range(K):
                # (sum_j dpoly[j] B^j) x = w  =>  x_t = (w_t - sum_{j>=1} dpoly[j] x_{t-j}) / dpoly[0]
                hist.append((wf[k] - sum(dpoly[j] * hist[-j] for j in range(1, d + 1))) / dpoly[0])
            preds = [sp.simplify(hist[len(xs) - 1 + k]) if exact else sp.Float(_f(hist[len(xs) - 1 + k]))
                     for k in ks]
        else:
            preds = _arma_predict(pr, xs, ks, exact)
    last = ts[-1]
    if scalar:
        return R(preds[0])
    return make_td([[(R(last + k), R(v)) for k, v in zip(ks, preds)]])


# ─────────────────────────── parameter conditions ───────────────────────────

from .. import kernel as _KK  # noqa: E402

for _h in ("ProcessParameterQ", "ProcessParameterAssumptions"):
    _KK.BUILTIN_ATTRIBUTES[_h] = {"HoldAll", "Protected"}

def _held_proc(proc):
    """evaluate the arguments of a held process expression without running its constructor"""
    if isinstance(proc, Expr) and isinstance(proc.head, str) and proc.head in CLASSES:
        return Expr(proc.head, *(_ev(a) for a in proc.args))
    if isinstance(proc, Expr) and isinstance(proc.head, Expr) and is_proc(proc.head):
        return proc
    return _ev(proc)


def _lax_parse(proc, fname):
    """parse with validators recording conditions instead of raising; returns (proc or None, conditions)"""
    proc = _held_proc(proc)
    if not is_proc(proc):
        if isinstance(proc, Expr) and is_proc(proc.head):
            raise TypeError(f"{fname}: {proc} is a process slice; use DistributionParameterAssumptions")
        if isinstance(proc, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {proc} is not a random process")
    _CONDS.clear()
    _LAX[0] = True
    try:
        try:
            pr = CLASSES[proc.head](proc, fname)
        except ValueError:
            return None, None
        return pr, list(_CONDS)
    finally:
        _LAX[0] = False
        _CONDS.clear()


@fn("ProcessParameterAssumptions", area=AREA, status="added", hold=True,
    notes="ProcessParameterAssumptions[proc]: the conjunction of the conditions the constructor of proc places on "
          "its parameters (positivity of rates, volatilities and noise variances, 0 <= p <= 1, reality of drifts "
          "...), evaluated: True or False for numeric parameters, a logical expression otherwise. Structural "
          "conditions that are checked directly (matrix shapes and row sums, integer orders, list forms) give False "
          "when violated and are not listed. Slices proc[t] raise TypeError (Wolfram's Possible Issues). "
          "Check: exact; reference (Wolfram documentation examples) in tests.")
def _ppa(proc: Held) -> Expr:
    fname = "ProcessParameterAssumptions"
    try:
        pr, conds = _lax_parse(proc, fname)
    except Unevaluated:
        raise
    except NotImplementedError:
        raise
    if pr is None:
        return False
    if not conds:
        return True
    return _ev(Expr("And", *conds))


@fn("ProcessParameterQ", area=AREA, status="added", hold=True,
    notes="ProcessParameterQ[proc]: True if proc is a random process whose numeric parameters meet the parameter "
          "assumptions (symbolic parameters are assumed valid, as in Wolfram), False if a parameter is invalid or "
          "proc is not a random process. Check: exact; reference (Wolfram documentation examples) in tests.")
def _ppq(proc: Held) -> Boolean:
    fname = "ProcessParameterQ"
    proc = _held_proc(proc)
    if not is_proc(proc):
        return False
    try:
        pr, conds = _lax_parse(proc, fname)
    except (TypeError, Unevaluated):
        return False
    if pr is None:
        return False
    return not any(_ev(c) is False for c in conds)


# ─────────────────────────── estimation ───────────────────────────

def _est_opts(opts, fname):
    est = "Automatic"
    for o in opts:
        if not (isinstance(o, Expr) and o.head in ("Rule", "RuleDelayed") and isinstance(o.args[0], Symbol)):
            raise ValueError(f"{fname}: unexpected argument {o}")
        k, v = o.args[0].name, o.args[1]
        if k == "ProcessEstimator":
            if isinstance(v, Symbol) and v.name == "Automatic":
                est = "Automatic"
            elif isinstance(v, str):
                est = v
            else:
                raise NotImplementedError(f"{fname}: ProcessEstimator -> {v} is not supported")
        elif k in ("AccuracyGoal", "PrecisionGoal", "WorkingPrecision"):
            if not (isinstance(v, Symbol) and v.name == "Automatic"):
                raise NotImplementedError(f"{fname}: {k} is not supported")
        else:
            raise ValueError(f"{fname}: unknown option {k}")
    return est


def _with_origin(path):
    """prepend (0, 0) for processes that start at 0 when the data starts later"""
    if path and _num(path[0][0]) and path[0][0] > 0:
        return [(sp.Integer(0), sp.Integer(0))] + path
    return path


def _increments(paths):
    out = []
    for p in paths:
        for (t0, x0), (t1, x1) in zip(p, p[1:]):
            dt = t1 - t0
            if not dt > 0:
                raise ValueError("FindProcessParameters: times must be increasing")
            out.append((dt, x1 - x0))
    if not out:
        raise ValueError("FindProcessParameters: at least two observations are needed")
    return out


def _fix(v, exact):
    v = sp.simplify(v) if exact else sp.Float(_f(v))
    return v


def _param_slot(a, fname):
    """Symbol -> ('free', symbol); number -> ('fixed', sympy value)"""
    if isinstance(a, Symbol):
        return ("free", a)
    v = _p(a, fname)
    if not _num(v):
        raise NotImplementedError(f"{fname}: parameters must be symbols or numbers")
    return ("fixed", v)


def _yule_walker(xs, p, centered):
    n = len(xs)
    m = sum(xs) / n if centered else sp.Integer(0)
    d = [x - m for x in xs]
    g = [sum(d[i + h] * d[i] for i in range(n - h)) / n for h in range(p + 1)]
    if g[0] == 0:
        raise ValueError("FindProcessParameters: the data are constant")
    phi, v = [], g[0]
    for k in range(1, p + 1):
        num = g[k] - sum(phi[j] * g[k - 1 - j] for j in range(k - 1))
        a = num / v
        phi = [phi[j] - a * phi[k - 2 - j] for j in range(k - 1)] + [a]
        v = v * (1 - a ** 2)
    c = m * (1 - sum(phi))
    return c, phi, v


def _ar_ols(xs, p, with_c, exact):
    n = len(xs)
    if n - p < p + (1 if with_c else 0) + 1:
        raise ValueError("FindProcessParameters: too few observations for the AR order")
    rows = [([sp.Integer(1)] if with_c else []) + [xs[t - i] for i in range(1, p + 1)] for t in range(p, n)]
    y = [xs[t] for t in range(p, n)]
    if exact:
        X, Y = sp.Matrix(rows), sp.Matrix(y)
        A = X.T * X
        if A.det() == 0:
            raise ValueError("FindProcessParameters: the regression is singular (degenerate data)")
        beta = list(A.LUsolve(X.T * Y))
        res = Y - X * sp.Matrix(beta)
        rss = (res.T * res)[0]
    else:
        import numpy as np
        X = np.array([[_f(v) for v in r] for r in rows])
        Y = np.array([_f(v) for v in y])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        rss = float(((Y - X @ beta) ** 2).sum())
        beta = [sp.Float(float(b)) for b in beta]
        rss = sp.Float(rss)
    c = beta[0] if with_c else sp.Integer(0)
    phi = beta[1:] if with_c else beta
    return c, phi, rss / (n - p)


ESTIMATE_HOOKS = {}     # X02: head -> fn(data, proc, estimator, fname, template_ok) -> (sols, proc)


def _estimate(data, proc, opts, fname, template_ok):
    est = _est_opts(opts, fname)
    if not (isinstance(proc, Expr) and isinstance(proc.head, str) and proc.head in CLASSES):
        if isinstance(proc, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {proc} is not a random process")
    h = proc.head
    if h in ESTIMATE_HOOKS:
        return ESTIMATE_HOOKS[h](data, proc, est, fname, template_ok)
    paths = _series_data(data, fname)
    exact = not any(_inexact(t, x) for p in paths for t, x in p)
    args = list(proc.args)
    sols = {}

    def need(est_ok):
        if est not in est_ok:
            raise NotImplementedError(f"{fname}: ProcessEstimator -> {est!r} is not supported for {h} (supported: "
                                      + ", ".join(repr(e) for e in est_ok) + ")")

    if h in ("WienerProcess", "PoissonProcess", "BinomialProcess", "GeometricBrownianMotionProcess"):
        need(("Automatic", "MaximumLikelihood"))
    if h == "WienerProcess" or h == "GeometricBrownianMotionProcess":
        if h == "WienerProcess":
            if len(args) != 2:
                raise ValueError(f"{fname}: expected WienerProcess[mu, sigma]")
            ps = [_with_origin(p) for p in paths]
        else:
            if len(args) != 3:
                raise ValueError(f"{fname}: expected GeometricBrownianMotionProcess[mu, sigma, x0]")
            x0s = _param_slot(args[2], fname)
            if any(x <= 0 for p in paths for _, x in p):
                raise ValueError(f"{fname}: geometric Brownian motion data must be positive")
            if x0s[0] == "free":
                if any(p[0][0] != 0 for p in paths):
                    raise NotImplementedError(f"{fname}: estimating x0 needs data starting at time 0")
                firsts = {p[0][1] for p in paths}
                sols[x0s[1]] = paths[0][0][1] if len(firsts) == 1 else sum(p[0][1] for p in paths) / len(paths)
                x0v = None
            else:
                x0v = x0s[1]
            ps = []
            for p in paths:
                q = [(t, sp.log(x)) for t, x in p]
                if p[0][0] > 0 and x0v is not None:
                    q = [(sp.Integer(0), sp.log(x0v))] + q
                ps.append(q)
            exact = False
        inc = _increments(ps)
        ms, ss = _param_slot(args[0], fname), _param_slot(args[1], fname)
        T = sum(dt for dt, _ in inc)
        if ms[0] == "free":
            m = sum(dx for _, dx in inc) / T
        else:
            m = ms[1] if h == "WienerProcess" else None
        if h == "GeometricBrownianMotionProcess" and ms[0] == "fixed":
            raise NotImplementedError(f"{fname}: a fixed drift for geometric Brownian motion is not supported")
        if ss[0] == "free":
            s2 = sum((dx - m * dt) ** 2 / dt for dt, dx in inc) / len(inc)
            if s2 == 0:
                raise ValueError(f"{fname}: the data have no variability")
            s = sp.sqrt(s2)
        else:
            s = ss[1]
        if h == "GeometricBrownianMotionProcess":
            m = m + s ** 2 / 2
        if ms[0] == "free":
            sols[ms[1]] = _fix(m, exact)
        if ss[0] == "free":
            sols[ss[1]] = _fix(s, exact)
    elif h in ("PoissonProcess", "BinomialProcess"):
        if len(args) != 1:
            raise ValueError(f"{fname}: expected {h}[p]")
        sl = _param_slot(args[0], fname)
        inc = _increments([_with_origin(p) for p in paths])
        for dt, dx in inc:
            if not (dx.is_Integer and dx >= 0) and not (not exact and float(dx).is_integer() and dx >= 0):
                raise ValueError(f"{fname}: {h} data must be non-decreasing integer counts")
            if h == "BinomialProcess" and dx > dt:
                raise ValueError(f"{fname}: BinomialProcess counts can grow by at most 1 per step")
        if sl[0] == "free":
            sols[sl[1]] = _fix(sum(dx for _, dx in inc) / sum(dt for dt, _ in inc), exact)
    elif h == "BernoulliProcess":
        need(("Automatic", "MaximumLikelihood", "MethodOfMoments"))
        if len(args) != 1:
            raise ValueError(f"{fname}: expected BernoulliProcess[p]")
        sl = _param_slot(args[0], fname)
        xs = [x for p in paths for _, x in p]
        if not all(x in (0, 1) for x in xs):
            raise ValueError(f"{fname}: BernoulliProcess data must be 0 or 1")
        if sl[0] == "free":
            sols[sl[1]] = _fix(sp.Add(*xs) / len(xs), exact)
    elif h == "RandomWalkProcess":
        need(("Automatic", "MaximumLikelihood", "MethodOfMoments"))
        if len(args) != 1:
            raise NotImplementedError(f"{fname}: only RandomWalkProcess[p] is supported")
        sl = _param_slot(args[0], fname)
        inc = _increments([_with_origin(p) for p in paths])
        steps = 0
        ups = 0
        for dt, dx in inc:
            if dt != 1 or dx not in (1, -1):
                raise ValueError(f"{fname}: RandomWalkProcess data must move by +1 or -1 at each unit time step")
            steps += 1
            ups += dx == 1
        if sl[0] == "free":
            sols[sl[1]] = _fix(sp.Integer(ups) / steps, exact)
    elif h == "ARProcess":
        if len(paths) != 1:
            raise NotImplementedError(f"{fname}: AR estimation from several paths is not supported")
        xs = [x for _, x in paths[0]]
        if len(args) == 1 and isinstance(args[0], int) and not isinstance(args[0], bool):
            if not template_ok:
                raise NotImplementedError(f"{fname}: the template ARProcess[p] needs symbolic parameters; use "
                                          "ARProcess[{a1, ..., ap}, v] or EstimatedProcess")
            p_ = args[0]
            names = [Symbol(f"a{i}") for i in range(1, p_ + 1)]
            cs, vs = None, ("free", Symbol("v"))
        else:
            if len(args) == 3:
                cs, al, vv = _param_slot(args[0], fname), args[1], args[2]
            elif len(args) == 2:
                cs, al, vv = None, args[0], args[1]
            else:
                raise ValueError(f"{fname}: expected ARProcess[{{a1, ...}}, v] or ARProcess[c, {{a1, ...}}, v]")
            if not is_list(al) or not al.args or not all(isinstance(a, Symbol) for a in al.args):
                raise NotImplementedError(f"{fname}: all AR coefficients must be symbols")
            names = list(al.args)
            vs = _param_slot(vv, fname)
            if vs[0] != "free" or (cs is not None and cs[0] != "free"):
                raise NotImplementedError(f"{fname}: fixed AR constants or noise variances are not supported")
            p_ = len(names)
        if p_ < 1 or len(xs) <= 2 * p_ + 1:
            raise ValueError(f"{fname}: too few observations for an AR({p_}) model")
        with_c = cs is not None
        if est == "YuleWalker":
            c, phi, v = _yule_walker(xs, p_, with_c)
        elif est == "MaximumConditionalLikelihood":
            c, phi, v = _ar_ols(xs, p_, with_c, exact)
        else:
            raise NotImplementedError(f"{fname}: for ARProcess give ProcessEstimator -> \"YuleWalker\" or "
                                      "\"MaximumConditionalLikelihood\" (the Automatic and maximum likelihood "
                                      "estimators are not supported)")
        if with_c:
            sols[cs[1]] = _fix(c, exact)
        for nm, a in zip(names, phi):
            sols[nm] = _fix(a, exact)
        sols[vs[1]] = _fix(v, exact)
        if len(args) == 1:
            return sols, Expr("ARProcess", L(*names), vs[1])
    else:
        raise NotImplementedError(f"{fname}: estimation for {h} is not supported (supported: WienerProcess, "
                                  "GeometricBrownianMotionProcess, PoissonProcess, BernoulliProcess, BinomialProcess, "
                                  "RandomWalkProcess, ARProcess)")
    if not sols:
        raise ValueError(f"{fname}: {proc} has no free (symbolic) parameters to estimate")
    return sols, proc


def _sols_rules(sols):
    return L(*(rule(k, R(v)) for k, v in sols.items()))


@fn("FindProcessParameters", area=AREA, status="added",
    notes="FindProcessParameters[data, proc] gives {param -> estimate, ...} for the symbolic parameters of proc. "
          "Supported: WienerProcess (Gaussian-increment MLE: mu = (x_end - 0)/elapsed time, sigma^2 = mean of "
          "(dx - mu dt)^2/dt; the process starts at 0 at time 0), GeometricBrownianMotionProcess (Wiener MLE of log "
          "x, mu = drift + sigma^2/2, machine numbers), PoissonProcess and BinomialProcess (count/elapsed time), "
          "BernoulliProcess (sample mean), RandomWalkProcess[p] (fraction of up steps) and ARProcess with "
          "ProcessEstimator -> \"YuleWalker\" (biased 1/n autocovariances, Durbin-Levinson, as statsmodels "
          "yule_walker method=\"mle\") or \"MaximumConditionalLikelihood\" (least squares on lagged values, noise "
          "variance RSS/(n - p) as statsmodels AutoReg). Numeric parameters stay fixed (Wiener, GBM x0). Exact "
          "data give exact estimates (except GBM). Data: a list of values (times 0, 1, ...), {t, x} pairs, "
          "TimeSeries or TemporalData (several paths are pooled). Other processes, Automatic for ARProcess and "
          "other estimators raise NotImplementedError. Check: exact; reference (closed-form MLEs, statsmodels "
          "yule_walker / AutoReg) in tests.")
def _fpp(data: Expr, proc: Expr, *opts: Expr) -> ListOf[Rule]:
    sols, _ = _estimate(data, proc, opts, "FindProcessParameters", False)
    return _sols_rules(sols)


@fn("EstimatedProcess", area=AREA, status="added",
    notes="EstimatedProcess[data, proc]: proc with its symbolic parameters replaced by the estimates of "
          "FindProcessParameters (same processes, estimators and options); the template ARProcess[p] is accepted "
          "(fitted as ARProcess[{a1, ..., ap}, v] without constant). Check: exact; reference (equals "
          "proc /. FindProcessParameters[data, proc]) in tests.")
def _estproc(data: Expr, proc: Expr, *opts: Expr) -> Expr:
    sols, proc2 = _estimate(data, proc, opts, "EstimatedProcess", True)
    reps = {k: R(v) for k, v in sols.items()}
    return _ev(_subst(proc2, reps))


from .wl_graphics_color import inert_option as _inert_option  # noqa: E402

_inert_option("ProcessEstimator", "ProcessEstimator: option of FindProcessParameters and EstimatedProcess; "
              "Automatic, \"MaximumLikelihood\", \"MethodOfMoments\" (where they coincide with the closed forms), "
              "\"YuleWalker\" and \"MaximumConditionalLikelihood\" (ARProcess) are supported. Check: exact; "
              "reference in tests.", AREA)
