"""Wolfram batch W17: probability distributions II (remaining univariate, multivariate, matrix and derived
distributions).

Every distribution here plugs into the distribution machinery of ``wl_stats`` (batch 5), so PDF, CDF, InverseCDF,
SurvivalFunction, HazardFunction, Mean, Variance, StandardDeviation, Skewness, Kurtosis, Median, Moment,
CentralMoment, Cumulant, CharacteristicFunction, MomentGeneratingFunction, Expectation, Probability and
RandomVariate work on them without any change to those functions:

* plain univariate distributions are ``wl_stats.D`` subclasses added to ``wl_stats.TABLE`` (their parameters are
  plain expressions or lists of expressions);
* derived and data-based univariate distributions (TruncatedDistribution, MixtureDistribution,
  HistogramDistribution, ...) are built from the whole expression by functions in ``wl_stats.BUILDERS``;
* multivariate and matrix distributions are objects in ``wl_stats.MULTI_BUILDERS`` with ``pdf``, ``cdf``,
  ``moment`` and ``sample`` methods returning MikoshiLang values.

Later batches (W18 statistical models, W25 random processes) extend the same three tables.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).  Closed forms are exact for exact parameters; where no closed form exists the value is computed
numerically for machine-number (inexact) input and exact input raises NotImplementedError instead of returning an
approximation.
"""
from __future__ import annotations

import inspect
import math

import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, L, sym, is_list, has_float, numeric_out, R
from . import wl_stats as W
from .wl_stats import D, Discrete, _num, _pos

AREA = "statistics: distributions II (W17)"
X = sp.Symbol("x")


# ─────────────────────────── helpers ───────────────────────────

def _real(v, what, name):
    if _num(v) and v.is_real is False:
        raise ValueError(f"{name}: {what} must be real")


def _posint(v, what, name):
    if _num(v) and not (v.is_Integer and v > 0):
        raise ValueError(f"{name}: {what} must be a positive integer")


def _nonnegint(v, what, name):
    if _num(v) and not (v.is_Integer and v >= 0):
        raise ValueError(f"{name}: {what} must be a non-negative integer")


def _open01(v, what, name):
    if _num(v) and not (0 < v < 1):
        raise ValueError(f"{name}: {what} must be in (0, 1)")


def _scalar(v, name):
    if isinstance(v, list):
        raise TypeError(f"{name}: expected a scalar parameter, got a list")
    return v


def _nargs(a, allowed, name, forms):
    if len(a) not in allowed:
        raise ValueError(f"{name}: expected {forms}")
    for v in a:
        _scalar(v, name)


def _interval(v, name, what="{min, max}"):
    if not (isinstance(v, list) and len(v) == 2):
        raise TypeError(f"{name}: expected an interval {what}")
    lo, hi = v
    if _num(lo) and _num(hi) and not lo < hi:
        raise ValueError(f"{name}: need min < max")
    return lo, hi


def _f(v):
    return float(sp.N(v))


def _mp(v):
    import mpmath
    return mpmath.mpf(str(sp.N(v, 30))) if not isinstance(v, (int, float)) else mpmath.mpf(v)


def _inexact(x):
    return isinstance(x, sp.Basic) and x.has(sp.Float)


class UD(D):
    """Base class for W17 univariate continuous distributions.

    Subclasses set ``lo``/``hi`` (or ``bounds``), ``pdf`` and whatever closed forms exist (``cdf``, ``q``, ``mean``,
    ``var``, ``raw``, ``mgf``, ``cf``).  Without a closed-form CDF, numeric x with numeric parameters is integrated
    numerically (mpmath) when x or a parameter is inexact; exact input raises NotImplementedError.  Without a
    closed-form quantile, RandomVariate inverts the CDF numerically.
    """
    closed_cdf = True

    def num_cdf(self, x):
        import mpmath
        lo, hi = self.bounds()
        if not (_num(x) and self.numeric()):
            raise Unevaluated()
        if not (_inexact(x) or any(_inexact(v) for v in self.flat())):
            raise NotImplementedError(f"CDF: {self.name} has no closed-form CDF; use a machine-number argument")
        g = sp.lambdify(X, self.pdf(X), "mpmath")
        a = -mpmath.inf if lo == -sp.oo else _mp(lo)
        pts = [a, _mp(x)]
        if lo == -sp.oo and hi == sp.oo and self.numeric():
            pts = [-mpmath.inf, _mp(self.center()), _mp(x)] if _f(x) > _f(self.center()) else pts
        return sp.Float(mpmath.quad(g, pts))

    def center(self):
        return sp.Integer(0)

    def flat(self):
        out = []
        for v in self.a:
            out.extend(v if isinstance(v, list) else [v])
        return out

    def numeric(self):
        return all(_num(v) for v in self.flat())

    def sample(self, rng):
        u = rng.random()
        if hasattr(self, "q"):
            try:
                return _f(self.q(sp.Float(u)))
            except Unevaluated:
                pass
        return _invert(self, u)


def _invert(dd, u):
    """numeric inverse CDF by bracketing and bisection (machine precision)"""
    import mpmath
    lo, hi = dd.bounds()

    def F(x):
        v = W.cdf_value(dd, sp.Float(x))
        return float(sp.N(v))
    a = _f(lo) if lo != -sp.oo else None
    b = _f(hi) if hi != sp.oo else None
    step = 1.0
    if a is None:
        a = (b if b is not None else 0.0) - step
        while F(a) > u:
            step *= 2
            a -= step
    if b is None:
        b = a + 1.0
        step = 1.0
        while F(b) < u:
            step *= 2
            b += step
    for _ in range(200):
        m = (a + b) / 2
        if F(m) < u:
            a = m
        else:
            b = m
        if b - a <= 1e-15 * max(1.0, abs(m)):
            break
    return (a + b) / 2


def _numeric_moment(dd, f, what):
    """E[f(X)] by numeric integration, for inexact parameters only"""
    import mpmath
    if not dd.numeric():
        raise Unevaluated()
    if not any(_inexact(v) for v in dd.flat()):
        raise NotImplementedError(f"{what}: {dd.name} has no closed form here; use machine-number parameters")
    lo, hi = dd.bounds()
    g = sp.lambdify(X, f(X) * dd.pdf(X), "mpmath")
    return sp.Float(mpmath.quad(g, [-mpmath.inf if lo == -sp.oo else _mp(lo), mpmath.inf if hi == sp.oo else _mp(hi)]))


# ─────────────────────────── univariate continuous ───────────────────────────

class ArcSin(UD):
    def __init__(self, name, a):
        if len(a) == 0:
            a = [[sp.Integer(0), sp.Integer(1)]]
        if len(a) != 1:
            raise ValueError(f"{name}: expected ArcSinDistribution[{{xmin, xmax}}]")
        _interval(a[0], name)
        super().__init__(name, a)

    def bounds(self):
        return tuple(self.a[0])

    def pdf(self, x):
        a, b = self.a[0]
        return 1 / (sp.pi * sp.sqrt((x - a) * (b - x)))

    def cdf(self, x):
        a, b = self.a[0]
        return 2 * sp.asin(sp.sqrt((x - a) / (b - a))) / sp.pi

    def q(self, p):
        a, b = self.a[0]
        return a + (b - a) * sp.sin(sp.pi * p / 2) ** 2

    def mean(self):
        a, b = self.a[0]
        return (a + b) / 2

    def var(self):
        a, b = self.a[0]
        return (b - a) ** 2 / 8

    def raw(self, r):
        a, b = self.a[0]
        # X = a + (b - a) Y, Y ~ ArcSin(0, 1), E[Y^k] = binomial(2k, k)/4^k
        return sp.Add(*(sp.binomial(r, k) * a ** (r - k) * (b - a) ** k * sp.binomial(2 * k, k) / 4 ** k
                        for k in range(r + 1)))

    def skewness(self):
        return sp.Integer(0)


class _IrwinHall(UD):
    """sum (UniformSum) or mean (Bates) of n iid uniforms on [a, b]"""
    is_mean = False

    def __init__(self, name, a):
        if len(a) == 1:
            a = [a[0], [sp.Integer(0), sp.Integer(1)]]
        if len(a) != 2:
            raise ValueError(f"{name}: expected {name}[n] or {name}[n, {{min, max}}]")
        _scalar(a[0], name)
        _posint(a[0], "n", name)
        _interval(a[1], name)
        super().__init__(name, a)

    def _affine(self):
        n = self.a[0]
        lo, hi = self.a[1]
        # X = c + s Y with Y ~ Irwin-Hall(n) on [0, n]
        if self.is_mean:
            return lo, (hi - lo) / n
        return n * lo, hi - lo

    def bounds(self):
        n = self.a[0]
        c, s = self._affine()
        return c, c + s * n

    def _ih(self, y, deriv):
        n = self.a[0]
        if not (_num(n)):
            raise Unevaluated()
        n = int(n)
        if _num(y):
            # exact piecewise polynomial evaluated at numeric y
            if deriv:
                return sp.Add(*((-1) ** k * sp.binomial(n, k) * (y - k) ** (n - 1)
                                for k in range(0, int(sp.floor(y)) + 1) if k <= n)) / sp.factorial(n - 1)
            return sp.Add(*((-1) ** k * sp.binomial(n, k) * (y - k) ** n
                            for k in range(0, int(sp.floor(y)) + 1) if k <= n)) / sp.factorial(n)
        pieces = []
        for j in range(n):
            if deriv:
                e = sp.Add(*((-1) ** k * sp.binomial(n, k) * (y - k) ** (n - 1) for k in range(j + 1))) / sp.factorial(n - 1)
            else:
                e = sp.Add(*((-1) ** k * sp.binomial(n, k) * (y - k) ** n for k in range(j + 1))) / sp.factorial(n)
            pieces.append((sp.expand(e), y <= j + 1))
        return sp.Piecewise(*pieces)

    def pdf(self, x):
        c, s = self._affine()
        return self._ih((x - c) / s, True) / s

    def cdf(self, x):
        c, s = self._affine()
        return self._ih((x - c) / s, False)

    def mean(self):
        n = self.a[0]
        c, s = self._affine()
        return c + s * n / 2

    def var(self):
        n = self.a[0]
        c, s = self._affine()
        return s ** 2 * n / 12

    def mgf(self, t):
        n = self.a[0]
        c, s = self._affine()
        return sp.exp(c * t) * ((sp.exp(s * t) - 1) / (s * t)) ** n

    def skewness(self):
        return sp.Integer(0)

    def kurtosis(self):
        return 3 - sp.Rational(6, 5) / self.a[0]

    def center(self):
        return self.mean()

    def sample(self, rng):
        c, s = self._affine()
        return _f(c) + _f(s) * sum(rng.random() for _ in range(int(self.a[0])))


class Bates(_IrwinHall):
    is_mean = True


class UniformSum(_IrwinHall):
    is_mean = False


class BetaPrime(UD):
    lo = sp.Integer(0)
    open_lo = True

    def __init__(self, name, a):
        _nargs(a, (2, 3, 4), name, "BetaPrimeDistribution[p, q], [p, q, beta] or [p, q, alpha, beta]")
        if len(a) == 2:
            a = [a[0], a[1], sp.Integer(1), sp.Integer(1)]
        elif len(a) == 3:
            a = [a[0], a[1], sp.Integer(1), a[2]]
        for v, w in zip(a, ("p", "q", "alpha", "beta")):
            _pos(v, w, name)
        super().__init__(name, a)

    def pdf(self, x):
        p, q, al, be = self.a
        z = (x / be) ** al
        return al * (x / be) ** (al * p - 1) * (1 + z) ** (-p - q) / (be * sp.beta(p, q))

    def cdf(self, x):
        p, q, al, be = self.a
        z = (x / be) ** al
        return sp.betainc_regularized(p, q, 0, z / (1 + z))

    def raw(self, r):
        p, q, al, be = self.a
        if _num(q) and _num(al) and not (al * q > r):
            return sp.oo
        return be ** r * sp.gamma(p + sp.Rational(r) / al) * sp.gamma(q - sp.Rational(r) / al) / (sp.gamma(p) * sp.gamma(q))

    def mean(self):
        return _maybe_nan(self.raw(1))

    def var(self):
        m1, m2 = self.raw(1), self.raw(2)
        if m2 == sp.oo:
            return sp.oo if m1 != sp.oo else sp.nan
        return m2 - m1 ** 2

    def sample(self, rng):
        p, q, al, be = (_f(v) for v in self.a)
        g1, g2 = rng.gammavariate(p, 1.0), rng.gammavariate(q, 1.0)
        return be * (g1 / g2) ** (1 / al)


def _maybe_nan(v):
    return sp.oo if v == sp.oo else v


class Benini(UD):
    def __init__(self, name, a):
        _nargs(a, (3,), name, "BeniniDistribution[alpha, beta, sigma]")
        for v, w in zip(a, ("alpha", "beta", "sigma")):
            _pos(v, w, name)
        super().__init__(name, a)

    def bounds(self):
        return self.a[2], sp.oo

    def pdf(self, x):
        al, be, s = self.a
        lg = sp.log(x / s)
        return sp.exp(-al * lg - be * lg ** 2) * (al / x + 2 * be * lg / x)

    def cdf(self, x):
        al, be, s = self.a
        lg = sp.log(x / s)
        return 1 - sp.exp(-al * lg - be * lg ** 2)

    def q(self, p):
        al, be, s = self.a
        return s * sp.exp((-al + sp.sqrt(al ** 2 - 4 * be * sp.log(1 - p))) / (2 * be))

    def raw(self, r):
        al, be, s = self.a
        # E[X^r] = s^r E[e^{r L}], L = log(X/s); E[e^{rL}] = 1 + r Int_0^oo e^{r l} S(l) dl
        l = sp.Dummy("l", positive=True)
        if not (_num(al) and _num(be)):
            raise Unevaluated()
        v = 1 + r * sp.integrate(sp.exp(r * l - al * l - be * l ** 2), (l, 0, sp.oo))
        return s ** r * sp.simplify(v)

    def mean(self):
        return self.raw(1)

    def var(self):
        return sp.simplify(self.raw(2) - self.raw(1) ** 2)


class Dagum(UD):
    def __init__(self, name, a):
        _nargs(a, (3, 4), name, "DagumDistribution[p, a, b] or [p, a, b, mu]")
        if len(a) == 3:
            a = list(a) + [sp.Integer(0)]
        for v, w in zip(a[:3], ("p", "a", "b")):
            _pos(v, w, name)
        super().__init__(name, a)

    def bounds(self):
        return self.a[3], sp.oo

    def pdf(self, x):
        p, a, b, m = self.a
        z = (x - m) / b
        return a * p / (x - m) * z ** (a * p) / (z ** a + 1) ** (p + 1)

    def cdf(self, x):
        p, a, b, m = self.a
        return (1 + ((x - m) / b) ** (-a)) ** (-p)

    def q(self, u):
        p, a, b, m = self.a
        return m + b * (u ** (-1 / p) - 1) ** (-1 / a)

    def raw(self, r):
        p, a, b, m = self.a
        if m != 0:
            return sp.Add(*(sp.binomial(r, k) * m ** (r - k) * self._raw0(k) for k in range(r + 1)))
        return self._raw0(r)

    def _raw0(self, k):
        p, a, b, m = self.a
        if k == 0:
            return sp.Integer(1)
        if _num(a) and not a > k:
            return sp.oo
        return b ** k * sp.gamma(p + sp.Rational(k) / a) * sp.gamma(1 - sp.Rational(k) / a) / sp.gamma(p)

    def mean(self):
        return self.raw(1)

    def var(self):
        m1, m2 = self.raw(1), self.raw(2)
        if m2 == sp.oo:
            return sp.oo if m1 != sp.oo else sp.nan
        return m2 - m1 ** 2


class SinghMaddala(Dagum):
    def __init__(self, name, a):
        _nargs(a, (3, 4), name, "SinghMaddalaDistribution[q, a, b] or [q, a, b, mu]")
        D.__init__(self, name, list(a) + ([sp.Integer(0)] if len(a) == 3 else []))
        for v, w in zip(self.a[:3], ("q", "a", "b")):
            _pos(v, w, name)

    def pdf(self, x):
        q, a, b, m = self.a
        z = (x - m) / b
        return a * q / b * z ** (a - 1) * (1 + z ** a) ** (-q - 1)

    def cdf(self, x):
        q, a, b, m = self.a
        return 1 - (1 + ((x - m) / b) ** a) ** (-q)

    def q(self, u):
        q, a, b, m = self.a
        return m + b * ((1 - u) ** (-1 / q) - 1) ** (1 / a)

    def _raw0(self, k):
        q, a, b, m = self.a
        if k == 0:
            return sp.Integer(1)
        if _num(a) and _num(q) and not a * q > k:
            return sp.oo
        return b ** k * sp.gamma(1 + sp.Rational(k) / a) * sp.gamma(q - sp.Rational(k) / a) / sp.gamma(q)


class Davis(UD):
    def __init__(self, name, a):
        _nargs(a, (3,), name, "DavisDistribution[b, n, mu]")
        _pos(a[0], "b", name)
        _pos(a[1], "n", name)
        if _num(a[1]) and not a[1] > 1:
            raise ValueError(f"{name}: n must be greater than 1")
        super().__init__(name, a)

    def bounds(self):
        return self.a[2], sp.oo

    open_lo = True

    def pdf(self, x):
        b, n, m = self.a
        return b ** n * (x - m) ** (-1 - n) / ((sp.exp(b / (x - m)) - 1) * sp.gamma(n) * sp.zeta(n))

    closed_cdf = False

    def cdf(self, x):
        return self.num_cdf(x)

    def center(self):
        return self.a[2] + self.a[0]

    def raw(self, r):
        b, n, m = self.a
        # E[(X - mu)^k] = b^k Gamma(n - k) zeta(n - k)/(Gamma(n) zeta(n)) for k < n - 1
        def c(k):
            if k == 0:
                return sp.Integer(1)
            if _num(n) and not n - k > 1:
                return sp.oo
            return b ** k * sp.gamma(n - k) * sp.zeta(n - k) / (sp.gamma(n) * sp.zeta(n))
        terms = [c(k) for k in range(r + 1)]
        if any(t == sp.oo for t in terms):
            return sp.oo
        return sp.Add(*(sp.binomial(r, k) * m ** (r - k) * terms[k] for k in range(r + 1)))

    def mean(self):
        return self.raw(1)

    def var(self):
        m1, m2 = self.raw(1), self.raw(2)
        if m2 == sp.oo:
            return sp.oo if m1 != sp.oo else sp.nan
        return m2 - m1 ** 2


class ExpGamma(UD):
    def __init__(self, name, a):
        _nargs(a, (3,), name, "ExpGammaDistribution[kappa, theta, mu]")
        _pos(a[0], "kappa", name)
        _pos(a[1], "theta", name)
        super().__init__(name, a)

    def pdf(self, x):
        k, th, m = self.a
        z = (x - m) / th
        return sp.exp(k * z - sp.exp(z)) / (th * sp.gamma(k))

    def cdf(self, x):
        k, th, m = self.a
        return sp.lowergamma(k, sp.exp((x - m) / th)) / sp.gamma(k)

    def mgf(self, t):
        k, th, m = self.a
        return sp.exp(m * t) * sp.gamma(k + th * t) / sp.gamma(k)

    def mean(self):
        k, th, m = self.a
        return m + th * sp.polygamma(0, k)

    def var(self):
        k, th, m = self.a
        return th ** 2 * sp.polygamma(1, k)

    def center(self):
        return self.a[2]

    def sample(self, rng):
        k, th, m = (_f(v) for v in self.a)
        return m + th * math.log(rng.gammavariate(k, 1.0))


class ExponentialPower(UD):
    def __init__(self, name, a):
        _nargs(a, (1, 2, 3), name, "ExponentialPowerDistribution[kappa], [kappa, sigma] or [kappa, mu, sigma]")
        if len(a) == 1:
            a = [a[0], sp.Integer(0), sp.Integer(1)]
        elif len(a) == 2:
            a = [a[0], sp.Integer(0), a[1]]
        _pos(a[0], "kappa", name)
        _pos(a[2], "sigma", name)
        super().__init__(name, a)

    def pdf(self, x):
        k, m, s = self.a
        return sp.exp(-sp.Abs(x - m) ** k / (k * s ** k)) / (2 * k ** (1 / k) * s * sp.gamma(1 + 1 / k))

    def cdf(self, x):
        k, m, s = self.a
        g = sp.lowergamma(1 / k, sp.Abs(x - m) ** k / (k * s ** k)) / sp.gamma(1 / k)
        return sp.Rational(1, 2) + sp.sign(x - m) * g / 2

    def raw(self, r):
        k, m, s = self.a
        # central moments: E[|Z|^j] = (k^(1/k) s)^j Gamma((j + 1)/k)/Gamma(1/k), odd central moments vanish
        def c(j):
            if j % 2:
                return sp.Integer(0)
            return (k ** (1 / k) * s) ** j * sp.gamma(sp.Integer(j + 1) / k) / sp.gamma(1 / k)
        return sp.Add(*(sp.binomial(r, j) * m ** (r - j) * c(j) for j in range(r + 1)))

    def mean(self):
        return self.a[1]

    def var(self):
        k, m, s = self.a
        return k ** (2 / k) * s ** 2 * sp.gamma(3 / k) / sp.gamma(1 / k)

    def skewness(self):
        return sp.Integer(0)

    def center(self):
        return self.a[1]

    def sample(self, rng):
        k, m, s = (_f(v) for v in self.a)
        g = rng.gammavariate(1 / k, 1.0)
        z = (k * g) ** (1 / k) * s
        return m + (z if rng.random() < 0.5 else -z)


class FisherZ(UD):
    def __init__(self, name, a):
        _nargs(a, (2,), name, "FisherZDistribution[n, m]")
        _pos(a[0], "n", name)
        _pos(a[1], "m", name)
        super().__init__(name, a)

    def pdf(self, x):
        n, m = self.a
        return 2 * n ** (n / 2) * m ** (m / 2) * sp.exp(n * x) / (sp.beta(n / 2, m / 2) * (m + n * sp.exp(2 * x)) ** ((n + m) / 2))

    def cdf(self, x):
        n, m = self.a
        e = n * sp.exp(2 * x)
        return sp.betainc_regularized(n / 2, m / 2, 0, e / (m + e))

    def mgf(self, t):
        n, m = self.a
        return (m / n) ** (t / 2) * sp.gamma((n + t) / 2) * sp.gamma((m - t) / 2) / (sp.gamma(n / 2) * sp.gamma(m / 2))

    def mean(self):
        n, m = self.a
        return (sp.log(m / n) + sp.polygamma(0, n / 2) - sp.polygamma(0, m / 2)) / 2

    def var(self):
        n, m = self.a
        return (sp.polygamma(1, n / 2) + sp.polygamma(1, m / 2)) / 4

    def sample(self, rng):
        n, m = (_f(v) for v in self.a)
        f = (rng.gammavariate(n / 2, 2.0) / n) / (rng.gammavariate(m / 2, 2.0) / m)
        return math.log(f) / 2


class Frechet(UD):
    def __init__(self, name, a):
        _nargs(a, (2, 3), name, "FrechetDistribution[alpha, beta] or [alpha, beta, mu]")
        if len(a) == 2:
            a = list(a) + [sp.Integer(0)]
        _pos(a[0], "alpha", name)
        _pos(a[1], "beta", name)
        super().__init__(name, a)

    open_lo = True

    def bounds(self):
        return self.a[2], sp.oo

    def pdf(self, x):
        al, be, m = self.a
        z = (x - m) / be
        return al / be * z ** (-1 - al) * sp.exp(-z ** (-al))

    def cdf(self, x):
        al, be, m = self.a
        return sp.exp(-((x - m) / be) ** (-al))

    def q(self, p):
        al, be, m = self.a
        return m + be * (-sp.log(p)) ** (-1 / al)

    def raw(self, r):
        al, be, m = self.a

        def c(k):
            if k == 0:
                return sp.Integer(1)
            if _num(al) and not al > k:
                return sp.oo
            return be ** k * sp.gamma(1 - sp.Rational(k) / al)
        terms = [c(k) for k in range(r + 1)]
        if any(t == sp.oo for t in terms):
            return sp.oo
        return sp.Add(*(sp.binomial(r, k) * m ** (r - k) * terms[k] for k in range(r + 1)))

    def mean(self):
        return self.raw(1)

    def var(self):
        m1, m2 = self.raw(1), self.raw(2)
        if m2 == sp.oo:
            return sp.oo if m1 != sp.oo else sp.nan
        return m2 - m1 ** 2


class HalfNormal(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (1,), name, "HalfNormalDistribution[theta]")
        _pos(a[0], "theta", name)
        super().__init__(name, a)

    def pdf(self, x):
        th = self.a[0]
        return 2 * th / sp.pi * sp.exp(-x ** 2 * th ** 2 / sp.pi)

    def cdf(self, x):
        th = self.a[0]
        return sp.erf(th * x / sp.sqrt(sp.pi))

    def q(self, p):
        return sp.sqrt(sp.pi) * sp.erfinv(p) / self.a[0]

    def raw(self, r):
        th = self.a[0]
        return sp.pi ** sp.Rational(r - 1, 2) * th ** (-r) * sp.gamma(sp.Rational(r + 1, 2))

    def mean(self):
        return 1 / self.a[0]

    def var(self):
        return (sp.pi - 2) / (2 * self.a[0] ** 2)

    def sample(self, rng):
        return abs(rng.gauss(0, 1)) * math.sqrt(math.pi / 2) / _f(self.a[0])


class Hoyt(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "HoytDistribution[q, omega]")
        if _num(a[0]) and not (0 < a[0] <= 1):
            raise ValueError(f"{name}: q must be in (0, 1]")
        _pos(a[1], "omega", name)
        super().__init__(name, a)

    def pdf(self, x):
        q, w = self.a
        return ((1 + q ** 2) * x / (q * w) * sp.exp(-(1 + q ** 2) ** 2 * x ** 2 / (4 * q ** 2 * w))
                * sp.besseli(0, (1 - q ** 4) * x ** 2 / (4 * q ** 2 * w)))

    closed_cdf = False

    def cdf(self, x):
        return self.num_cdf(x)

    def raw(self, r):
        q, w = self.a
        # X = sqrt(A^2 + B^2), A ~ N(0, s1), B ~ N(0, s2), s1^2 = w/(1 + q^2), s2^2 = q^2 w/(1 + q^2)
        if r % 2 == 0:
            k = r // 2
            s1, s2 = w / (1 + q ** 2), q ** 2 * w / (1 + q ** 2)
            # E[(A^2 + B^2)^k] by the binomial expansion of the Gaussian moments
            return sp.expand(sp.Add(*(sp.binomial(k, j) * sp.factorial2(2 * j - 1) * s1 ** j
                                      * sp.factorial2(2 * (k - j) - 1) * s2 ** (k - j) for j in range(k + 1))))
        return _numeric_moment(self, lambda x: x ** r, "Moment")

    def mean(self):
        q, w = self.a
        if not self.numeric():
            raise Unevaluated()
        # E[X] = sqrt(2 s1^2/pi) * 2 EllipticE[1 - s2^2/s1^2] * sqrt(pi/2)/sqrt(pi) ... closed via elliptic E
        s1 = sp.sqrt(w / (1 + q ** 2))
        return sp.sqrt(2 / sp.pi) * s1 * sp.elliptic_e(1 - q ** 2)

    def var(self):
        return sp.simplify(self.raw(2) - self.mean() ** 2)

    def sample(self, rng):
        q, w = (_f(v) for v in self.a)
        s1, s2 = math.sqrt(w / (1 + q * q)), math.sqrt(q * q * w / (1 + q * q))
        return math.hypot(rng.gauss(0, s1), rng.gauss(0, s2))


class Hyperbolic(UD):
    """HyperbolicDistribution[alpha, beta, delta, mu] and the generalised [lambda, alpha, beta, delta, mu]"""

    def __init__(self, name, a):
        _nargs(a, (4, 5), name, "HyperbolicDistribution[alpha, beta, delta, mu] or [lambda, alpha, beta, delta, mu]")
        if len(a) == 4:
            a = [sp.Integer(1)] + list(a)
        lam, al, be, de, m = a
        _pos(al, "alpha", name)
        _pos(de, "delta", name)
        if _num(al) and _num(be) and not abs(be) < al:
            raise ValueError(f"{name}: need |beta| < alpha")
        super().__init__(name, a)

    def gamma_(self):
        lam, al, be, de, m = self.a
        return sp.sqrt(al ** 2 - be ** 2)

    def pdf(self, x):
        lam, al, be, de, m = self.a
        g = self.gamma_()
        r = sp.sqrt(de ** 2 + (x - m) ** 2)
        if lam == 1:
            return g / (2 * al * de * sp.besselk(1, de * g)) * sp.exp(-al * r + be * (x - m))
        return ((g / de) ** lam / (sp.sqrt(2 * sp.pi) * sp.besselk(lam, de * g)) * sp.exp(be * (x - m))
                * sp.besselk(lam - sp.Rational(1, 2), al * r) / (r / al) ** (sp.Rational(1, 2) - lam))

    closed_cdf = False

    def cdf(self, x):
        return self.num_cdf(x)

    def center(self):
        return self.a[4]

    def mgf(self, t):
        lam, al, be, de, m = self.a
        g = self.gamma_()
        gt = sp.sqrt(al ** 2 - (be + t) ** 2)
        return sp.exp(m * t) * (g / gt) ** lam * sp.besselk(lam, de * gt) / sp.besselk(lam, de * g)

    def mean(self):
        lam, al, be, de, m = self.a
        g = self.gamma_()
        return m + be * de / g * sp.besselk(lam + 1, de * g) / sp.besselk(lam, de * g)

    def var(self):
        lam, al, be, de, m = self.a
        g = self.gamma_()
        z = de * g
        k0, k1, k2 = sp.besselk(lam, z), sp.besselk(lam + 1, z), sp.besselk(lam + 2, z)
        return de * k1 / (g * k0) + be ** 2 * de ** 2 / g ** 2 * (k2 / k0 - k1 ** 2 / k0 ** 2)

    def raw(self, r):
        if r <= 2:
            m = self.mean()
            return m if r == 1 else self.var() + m ** 2
        return _numeric_moment(self, lambda x: x ** r, "Moment")

    def sample(self, rng):
        return _invert(self, rng.random())


class InverseChiSquare(UD):
    lo = sp.Integer(0)
    open_lo = True

    def __init__(self, name, a):
        _nargs(a, (1, 2), name, "InverseChiSquareDistribution[nu] or [nu, xi]")
        _pos(a[0], "nu", name)
        if len(a) == 2:
            _pos(a[1], "xi", name)
        super().__init__(name, a)

    def ig(self):
        nu = self.a[0]
        xi = self.a[1] if len(self.a) == 2 else 1 / nu
        return nu / 2, nu * xi / 2

    def pdf(self, x):
        al, be = self.ig()
        return be ** al * x ** (-al - 1) * sp.exp(-be / x) / sp.gamma(al)

    def cdf(self, x):
        al, be = self.ig()
        return sp.uppergamma(al, be / x) / sp.gamma(al)

    def raw(self, r):
        al, be = self.ig()
        if _num(al) and not al > r:
            return sp.oo
        return be ** r * sp.gamma(al - r) / sp.gamma(al)

    def mean(self):
        al, be = self.ig()
        if _num(al) and not al > 1:
            return sp.oo
        return be / (al - 1)

    def var(self):
        al, be = self.ig()
        if _num(al) and not al > 2:
            return sp.oo if al > 1 else sp.nan
        return be ** 2 / ((al - 1) ** 2 * (al - 2))

    def sample(self, rng):
        al, be = (_f(v) for v in self.ig())
        return be / rng.gammavariate(al, 1.0)

    def q(self, p):
        raise Unevaluated()


class InverseGamma(InverseChiSquare):
    def __init__(self, name, a):
        _nargs(a, (2,), name, "InverseGammaDistribution[alpha, beta]")
        _pos(a[0], "alpha", name)
        _pos(a[1], "beta", name)
        D.__init__(self, name, a)

    def ig(self):
        return self.a[0], self.a[1]


class InverseGaussian(UD):
    lo = sp.Integer(0)
    open_lo = True

    def __init__(self, name, a):
        _nargs(a, (2,), name, "InverseGaussianDistribution[mu, lambda]")
        _pos(a[0], "mu", name)
        _pos(a[1], "lambda", name)
        super().__init__(name, a)

    def pdf(self, x):
        m, l = self.a
        return sp.sqrt(l / (2 * sp.pi * x ** 3)) * sp.exp(-l * (x - m) ** 2 / (2 * m ** 2 * x))

    def cdf(self, x):
        m, l = self.a
        Phi = lambda z: sp.erfc(-z / sp.sqrt(2)) / 2  # noqa: E731
        return Phi(sp.sqrt(l / x) * (x / m - 1)) + sp.exp(2 * l / m) * Phi(-sp.sqrt(l / x) * (x / m + 1))

    def mgf(self, t):
        m, l = self.a
        return sp.exp(l / m * (1 - sp.sqrt(1 - 2 * m ** 2 * t / l)))

    def mean(self):
        return self.a[0]

    def var(self):
        m, l = self.a
        return m ** 3 / l

    def sample(self, rng):
        m, l = (_f(v) for v in self.a)
        y = rng.gauss(0, 1) ** 2
        x = m + m * m * y / (2 * l) - m / (2 * l) * math.sqrt(4 * m * l * y + m * m * y * y)
        return x if rng.random() <= m / (m + x) else m * m / x


class Johnson(UD):
    TYPES = ("SN", "SL", "SU", "SB")

    def __init__(self, name, a, raw_type=None):
        if raw_type not in self.TYPES:
            raise ValueError(f"{name}: the type must be one of \"SN\", \"SL\", \"SU\", \"SB\"")
        if len(a) != 4:
            raise ValueError(f"{name}: expected JohnsonDistribution[type, gamma, delta, mu, sigma]")
        _pos(a[1], "delta", name)
        _pos(a[3], "sigma", name)
        self.t = raw_type
        super().__init__(name, a)

    def bounds(self):
        g, d, m, s = self.a
        if self.t == "SL":
            return m, sp.oo
        if self.t == "SB":
            return m, m + s
        return -sp.oo, sp.oo

    open_lo = open_hi = True

    def _g(self, u):
        return {"SN": lambda: u, "SL": lambda: sp.log(u), "SU": lambda: sp.asinh(u),
                "SB": lambda: sp.log(u / (1 - u))}[self.t]()

    def _gi(self, z):
        return {"SN": lambda: z, "SL": lambda: sp.exp(z), "SU": lambda: sp.sinh(z),
                "SB": lambda: 1 / (1 + sp.exp(-z))}[self.t]()

    def cdf(self, x):
        g, d, m, s = self.a
        z = g + d * self._g((x - m) / s)
        return sp.erfc(-z / sp.sqrt(2)) / 2

    def pdf(self, x):
        g, d, m, s = self.a
        u = (x - m) / s
        z = g + d * self._g(u)
        return sp.exp(-z ** 2 / 2) / sp.sqrt(2 * sp.pi) * d * sp.diff(self._g(X), X).subs(X, u) / s

    def q(self, p):
        g, d, m, s = self.a
        z = -sp.sqrt(2) * sp.erfcinv(2 * p)
        return m + s * self._gi((z - g) / d)

    def mean(self):
        g, d, m, s = self.a
        if self.t == "SN":
            return m - s * g / d
        if self.t == "SL":
            return m + s * sp.exp((1 - 2 * g * d) / (2 * d ** 2))
        if self.t == "SU":
            return m - s * sp.exp(1 / (2 * d ** 2)) * sp.sinh(g / d)
        return _numeric_moment(self, lambda x: x, "Mean")

    def var(self):
        g, d, m, s = self.a
        if self.t == "SN":
            return s ** 2 / d ** 2
        if self.t == "SL":
            w = sp.exp(1 / d ** 2)
            return s ** 2 * w * (w - 1) * sp.exp(-2 * g / d)
        if self.t == "SU":
            w = sp.exp(1 / d ** 2)
            return s ** 2 / 2 * (w - 1) * (w * sp.cosh(2 * g / d) + 1)
        return sp.simplify(_numeric_moment(self, lambda x: x ** 2, "Variance") - self.mean() ** 2)

    def center(self):
        return self.q(sp.Rational(1, 2))

    def sample(self, rng):
        g, d, m, s = (_f(v) for v in self.a)
        return _f(self.q(sp.Float(rng.random())))


class KDist(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "KDistribution[nu, w]")
        _pos(a[0], "nu", name)
        _pos(a[1], "w", name)
        super().__init__(name, a)

    def pdf(self, x):
        v, w = self.a
        return 2 * (v / w) ** ((1 + v) / 2) * x ** v * sp.besselk(v - 1, 2 * x * sp.sqrt(v / w)) / sp.gamma(v)

    def cdf(self, x):
        v, w = self.a
        z = 2 * x * sp.sqrt(v / w)
        return 1 - 2 * (z / 2) ** v * sp.besselk(v, z) / sp.gamma(v)

    def raw(self, r):
        v, w = self.a
        # X^2 = w/nu * G1 * G2 with G1 ~ Gamma(nu, 1), G2 ~ Gamma(1, 1)
        return (w / v) ** sp.Rational(r, 2) * sp.gamma(v + sp.Rational(r, 2)) * sp.gamma(1 + sp.Rational(r, 2)) / sp.gamma(v)

    def mean(self):
        return self.raw(1)

    def var(self):
        return self.a[1] - self.raw(1) ** 2

    def sample(self, rng):
        v, w = (_f(x) for x in self.a)
        return math.sqrt(w / v * rng.gammavariate(v, 1.0) * rng.expovariate(1.0))


class Kumaraswamy(UD):
    lo, hi = sp.Integer(0), sp.Integer(1)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "KumaraswamyDistribution[alpha, beta]")
        _pos(a[0], "alpha", name)
        _pos(a[1], "beta", name)
        super().__init__(name, a)

    def pdf(self, x):
        al, be = self.a
        return al * be * x ** (al - 1) * (1 - x ** al) ** (be - 1)

    def cdf(self, x):
        al, be = self.a
        return 1 - (1 - x ** al) ** be

    def q(self, p):
        al, be = self.a
        return (1 - (1 - p) ** (1 / be)) ** (1 / al)

    def raw(self, r):
        al, be = self.a
        return be * sp.gamma(1 + sp.Rational(r) / al) * sp.gamma(be) / sp.gamma(1 + be + sp.Rational(r) / al)

    def mean(self):
        return self.raw(1)

    def var(self):
        return self.raw(2) - self.raw(1) ** 2


class Levy(UD):
    open_lo = True

    def __init__(self, name, a):
        _nargs(a, (2,), name, "LevyDistribution[mu, sigma]")
        _pos(a[1], "sigma", name)
        super().__init__(name, a)

    def bounds(self):
        return self.a[0], sp.oo

    def pdf(self, x):
        m, s = self.a
        return sp.sqrt(s / (2 * sp.pi)) * sp.exp(-s / (2 * (x - m))) / (x - m) ** sp.Rational(3, 2)

    def cdf(self, x):
        m, s = self.a
        return sp.erfc(sp.sqrt(s / (2 * (x - m))))

    def q(self, p):
        m, s = self.a
        return m + s / (2 * sp.erfcinv(p) ** 2)

    def mean(self):
        return sp.oo

    def var(self):
        return sp.oo

    def raw(self, r):
        return sp.oo

    def skewness(self):
        return sp.nan

    def kurtosis(self):
        return sp.nan

    def cf(self, t):
        m, s = self.a
        return sp.exp(sp.I * m * t - sp.sqrt(-2 * sp.I * s * t))


class Lindley(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (1,), name, "LindleyDistribution[delta]")
        _pos(a[0], "delta", name)
        super().__init__(name, a)

    def pdf(self, x):
        d = self.a[0]
        return d ** 2 / (1 + d) * (1 + x) * sp.exp(-d * x)

    def cdf(self, x):
        d = self.a[0]
        return 1 - (1 + d + d * x) / (1 + d) * sp.exp(-d * x)

    def raw(self, r):
        d = self.a[0]
        return sp.factorial(r) * (d + r + 1) / (d ** r * (d + 1))

    def mean(self):
        return self.raw(1)

    def var(self):
        return sp.simplify(self.raw(2) - self.raw(1) ** 2)

    def mgf(self, t):
        d = self.a[0]
        return d ** 2 * (d - t + 1) / ((d + 1) * (d - t) ** 2)

    def sample(self, rng):
        d = _f(self.a[0])
        if rng.random() < d / (d + 1):
            return rng.expovariate(d)
        return rng.gammavariate(2.0, 1 / d)


class LogGamma(UD):
    def __init__(self, name, a):
        _nargs(a, (2, 3), name, "LogGammaDistribution[alpha, beta] or [alpha, beta, mu]")
        if len(a) == 2:
            a = list(a) + [sp.Integer(0)]
        _pos(a[0], "alpha", name)
        _pos(a[1], "beta", name)
        super().__init__(name, a)

    def bounds(self):
        return self.a[2], sp.oo

    def pdf(self, x):
        al, be, m = self.a
        return sp.log(1 + x - m) ** (al - 1) * (1 + x - m) ** (-1 - 1 / be) / (be ** al * sp.gamma(al))

    def cdf(self, x):
        al, be, m = self.a
        return sp.lowergamma(al, sp.log(1 + x - m) / be) / sp.gamma(al)

    def raw(self, r):
        al, be, m = self.a
        # X = mu - 1 + e^Y, Y ~ Gamma(alpha, beta); E[e^{kY}] = (1 - k beta)^-alpha for k beta < 1
        def e(k):
            if k == 0:
                return sp.Integer(1)
            if _num(be) and not k * be < 1:
                return sp.oo
            return (1 - k * be) ** (-al)
        terms = [e(k) for k in range(r + 1)]
        if any(t == sp.oo for t in terms):
            return sp.oo
        return sp.Add(*(sp.binomial(r, k) * (m - 1) ** (r - k) * terms[k] for k in range(r + 1)))

    def mean(self):
        return self.raw(1)

    def var(self):
        m1, m2 = self.raw(1), self.raw(2)
        if m2 == sp.oo:
            return sp.oo if m1 != sp.oo else sp.nan
        return sp.simplify(m2 - m1 ** 2)

    def sample(self, rng):
        al, be, m = (_f(v) for v in self.a)
        return m - 1 + math.exp(rng.gammavariate(al, be))


class LogLogistic(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "LogLogisticDistribution[gamma, sigma]")
        _pos(a[0], "gamma", name)
        _pos(a[1], "sigma", name)
        super().__init__(name, a)

    def pdf(self, x):
        g, s = self.a
        return g / s * (x / s) ** (g - 1) / (1 + (x / s) ** g) ** 2

    def cdf(self, x):
        g, s = self.a
        return 1 / (1 + (x / s) ** (-g))

    def q(self, p):
        g, s = self.a
        return s * (p / (1 - p)) ** (1 / g)

    def raw(self, r):
        g, s = self.a
        if _num(g) and not g > r:
            return sp.oo
        b = sp.pi * r / g
        return s ** r * b / sp.sin(b)

    def mean(self):
        return self.raw(1)

    def var(self):
        m1, m2 = self.raw(1), self.raw(2)
        if m2 == sp.oo:
            return sp.oo if m1 != sp.oo else sp.nan
        return m2 - m1 ** 2


class MarchenkoPastur(UD):
    def __init__(self, name, a):
        _nargs(a, (1, 2), name, "MarchenkoPasturDistribution[lambda] or [lambda, sigma]")
        if len(a) == 1:
            a = [a[0], sp.Integer(1)]
        _pos(a[0], "lambda", name)
        _pos(a[1], "sigma", name)
        if _num(a[0]) and a[0] > 1:
            raise NotImplementedError(f"{name}: lambda > 1 (an atom at 0) is not supported")
        super().__init__(name, a)

    def bounds(self):
        l, s = self.a
        return s ** 2 * (1 - sp.sqrt(l)) ** 2, s ** 2 * (1 + sp.sqrt(l)) ** 2

    def pdf(self, x):
        l, s = self.a
        a, b = self.bounds()
        return sp.sqrt((b - x) * (x - a)) / (2 * sp.pi * s ** 2 * l * x)

    closed_cdf = False

    def cdf(self, x):
        return self.num_cdf(x)

    def raw(self, r):
        l, s = self.a
        # moments of Marchenko-Pastur: sigma^(2r) Sum_k Narayana(r, k) lambda^(k-1)
        return s ** (2 * r) * sp.Add(*(sp.binomial(r, k) * sp.binomial(r, k - 1) / r * l ** (k - 1)
                                       for k in range(1, r + 1))) if r > 0 else sp.Integer(1)

    def mean(self):
        return self.a[1] ** 2

    def var(self):
        l, s = self.a
        return l * s ** 4

    def center(self):
        return self.mean()


class MaxStable(UD):
    """generalised extreme value distribution (maximum); MinStable is its mirror image"""
    sign = 1

    def __init__(self, name, a):
        _nargs(a, (3,), name, f"{name}[mu, sigma, xi]")
        _pos(a[1], "sigma", name)
        super().__init__(name, a)

    def bounds(self):
        m, s, xi = self.a
        if not _num(xi):
            raise Unevaluated()
        if xi == 0:
            return -sp.oo, sp.oo
        e = m - self.sign * s / xi
        if (xi > 0) == (self.sign > 0):
            return e, sp.oo
        return -sp.oo, e

    def _t(self, x):
        m, s, xi = self.a
        z = self.sign * (x - m) / s
        if xi == 0:
            return sp.exp(-z)
        return (1 + xi * z) ** (-1 / xi)

    def cdf(self, x):
        t = self._t(x)
        return sp.exp(-t) if self.sign > 0 else 1 - sp.exp(-t)

    def pdf(self, x):
        m, s, xi = self.a
        t = self._t(x)
        return t ** (xi + 1) * sp.exp(-t) / s

    def q(self, p):
        m, s, xi = self.a
        u = -sp.log(p) if self.sign > 0 else -sp.log(1 - p)
        if xi == 0:
            return m - self.sign * s * sp.log(u)
        return m + self.sign * s * (u ** (-xi) - 1) / xi

    def mean(self):
        m, s, xi = self.a
        if xi == 0:
            return m + self.sign * s * sp.EulerGamma
        if _num(xi) and not xi < 1:
            return sp.oo * self.sign
        return m + self.sign * s * (sp.gamma(1 - xi) - 1) / xi

    def var(self):
        m, s, xi = self.a
        if xi == 0:
            return s ** 2 * sp.pi ** 2 / 6
        if _num(xi) and not xi < sp.Rational(1, 2):
            return sp.oo
        return s ** 2 * (sp.gamma(1 - 2 * xi) - sp.gamma(1 - xi) ** 2) / xi ** 2

    def raw(self, r):
        if r <= 2:
            m1 = self.mean()
            return m1 if r == 1 else self.var() + m1 ** 2
        m, s, xi = self.a
        if xi == 0:
            return _numeric_moment(self, lambda x: x ** r, "Moment")
        if _num(xi) and not xi * r < 1:
            return sp.oo
        # X = m + sign s (E^-xi - 1)/xi with E ~ Exponential(1): E[E^(-k xi)] = Gamma(1 - k xi)
        c = self.sign * s / xi
        return sp.Add(*(sp.binomial(r, k) * (m - c) ** (r - k) * c ** k * sp.gamma(1 - k * xi) for k in range(r + 1)))


class MinStable(MaxStable):
    sign = -1


class Maxwell(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (1,), name, "MaxwellDistribution[sigma]")
        _pos(a[0], "sigma", name)
        super().__init__(name, a)

    def pdf(self, x):
        s = self.a[0]
        return sp.sqrt(2 / sp.pi) * x ** 2 * sp.exp(-x ** 2 / (2 * s ** 2)) / s ** 3

    def cdf(self, x):
        s = self.a[0]
        return sp.erf(x / (sp.sqrt(2) * s)) - sp.sqrt(2 / sp.pi) * x * sp.exp(-x ** 2 / (2 * s ** 2)) / s

    def raw(self, r):
        s = self.a[0]
        return 2 ** (sp.Rational(r, 2) + 1) * s ** r * sp.gamma(sp.Rational(r + 3, 2)) / sp.sqrt(sp.pi)

    def mean(self):
        return 2 * sp.sqrt(2 / sp.pi) * self.a[0]

    def var(self):
        return (3 * sp.pi - 8) * self.a[0] ** 2 / sp.pi

    def sample(self, rng):
        s = _f(self.a[0])
        return math.sqrt(sum(rng.gauss(0, s) ** 2 for _ in range(3)))


class Meixner(UD):
    def __init__(self, name, a):
        _nargs(a, (4,), name, "MeixnerDistribution[a, b, m, d]")
        _pos(a[0], "a", name)
        _pos(a[3], "d", name)
        if _num(a[1]) and not abs(a[1]) < sp.pi:
            raise ValueError(f"{name}: need -Pi < b < Pi")
        super().__init__(name, a)

    def pdf(self, x):
        a, b, m, d = self.a
        z = (x - m) / a
        return ((2 * sp.cos(b / 2)) ** (2 * d) / (2 * a * sp.pi * sp.gamma(2 * d)) * sp.exp(b * z)
                * sp.gamma(d + sp.I * z) * sp.gamma(d - sp.I * z))

    closed_cdf = False

    def cdf(self, x):
        return self.num_cdf(x)

    def center(self):
        return self.a[2]

    def cf(self, t):
        a, b, m, d = self.a
        return sp.exp(sp.I * m * t) * (sp.cos(b / 2) / sp.cosh((a * t - sp.I * b) / 2)) ** (2 * d)

    def mgf(self, t):
        a, b, m, d = self.a
        return sp.exp(m * t) * (sp.cos(b / 2) / sp.cos((a * t + b) / 2)) ** (2 * d)

    def mean(self):
        a, b, m, d = self.a
        return m + a * d * sp.tan(b / 2)

    def var(self):
        a, b, m, d = self.a
        return a ** 2 * d / (2 * sp.cos(b / 2) ** 2)

    def sample(self, rng):
        return _invert(self, rng.random())


class Moyal(UD):
    def __init__(self, name, a):
        _nargs(a, (0, 2), name, "MoyalDistribution[] or [mu, sigma]")
        if not a:
            a = [sp.Integer(0), sp.Integer(1)]
        _pos(a[1], "sigma", name)
        super().__init__(name, a)

    def pdf(self, x):
        m, s = self.a
        z = (x - m) / s
        return sp.exp(-(sp.exp(-z) + z) / 2) / (s * sp.sqrt(2 * sp.pi))

    def cdf(self, x):
        m, s = self.a
        return sp.erfc(sp.exp(-(x - m) / (2 * s)) / sp.sqrt(2))

    def q(self, p):
        m, s = self.a
        return m - 2 * s * sp.log(sp.sqrt(2) * sp.erfcinv(p))

    def mgf(self, t):
        m, s = self.a
        return sp.exp(m * t) * 2 ** (-s * t) * sp.gamma(sp.Rational(1, 2) - s * t) / sp.sqrt(sp.pi)

    def mean(self):
        m, s = self.a
        return m + s * (sp.EulerGamma + sp.log(2))

    def var(self):
        return sp.pi ** 2 * self.a[1] ** 2 / 2

    def skewness(self):
        return 28 * sp.sqrt(2) * sp.zeta(3) / sp.pi ** 3

    def kurtosis(self):
        return sp.Integer(7)


class Nakagami(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "NakagamiDistribution[mu, omega]")
        _pos(a[0], "mu", name)
        _pos(a[1], "omega", name)
        if _num(a[0]) and a[0] < sp.Rational(1, 2):
            raise ValueError(f"{name}: mu must be at least 1/2")
        super().__init__(name, a)

    def pdf(self, x):
        m, w = self.a
        return 2 * m ** m / (sp.gamma(m) * w ** m) * x ** (2 * m - 1) * sp.exp(-m * x ** 2 / w)

    def cdf(self, x):
        m, w = self.a
        return sp.lowergamma(m, m * x ** 2 / w) / sp.gamma(m)

    def raw(self, r):
        m, w = self.a
        return sp.gamma(m + sp.Rational(r, 2)) / sp.gamma(m) * (w / m) ** sp.Rational(r, 2)

    def mean(self):
        return self.raw(1)

    def var(self):
        return self.a[1] - self.raw(1) ** 2

    def sample(self, rng):
        m, w = (_f(v) for v in self.a)
        return math.sqrt(rng.gammavariate(m, w / m))


class _Noncentral(UD):
    """shared numeric CDF for the noncentral families: Poisson(lambda/2)-weighted series, inexact input only"""
    closed_cdf = False

    def cdf(self, x):
        import mpmath
        if not (_num(x) and self.numeric()):
            raise Unevaluated()
        if not (_inexact(x) or any(_inexact(v) for v in self.flat())):
            raise NotImplementedError(f"CDF: {self.name} has no closed-form CDF; use a machine-number argument")
        return sp.Float(self.series_cdf(_mp(x)))


class NoncentralChiSquare(_Noncentral):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "NoncentralChiSquareDistribution[nu, lambda]")
        _pos(a[0], "nu", name)
        if _num(a[1]) and a[1] < 0:
            raise ValueError(f"{name}: lambda must be non-negative")
        super().__init__(name, a)

    def pdf(self, x):
        v, l = self.a
        if l == 0:
            return x ** (v / 2 - 1) * sp.exp(-x / 2) / (2 ** (v / 2) * sp.gamma(v / 2))
        return sp.exp(-(x + l) / 2) * (x / l) ** (v / 4 - sp.Rational(1, 2)) * sp.besseli(v / 2 - 1, sp.sqrt(l * x)) / 2

    def series_cdf(self, x):
        import mpmath
        v, l = (_mp(t) for t in self.a)
        return mpmath.nsum(lambda j: mpmath.exp(-l / 2) * (l / 2) ** j / mpmath.factorial(j)
                           * mpmath.gammainc(v / 2 + j, 0, x / 2, regularized=True), [0, mpmath.inf])

    def mean(self):
        return self.a[0] + self.a[1]

    def var(self):
        return 2 * (self.a[0] + 2 * self.a[1])

    def mgf(self, t):
        v, l = self.a
        return sp.exp(l * t / (1 - 2 * t)) / (1 - 2 * t) ** (v / 2)

    def sample(self, rng):
        v, l = (_f(t) for t in self.a)
        k = W.Poisson("PoissonDistribution", [sp.Float(l / 2)]).sample(rng) if l > 0 else 0
        return rng.gammavariate(v / 2 + k, 2.0)


class NoncentralBeta(_Noncentral):
    lo, hi = sp.Integer(0), sp.Integer(1)

    def __init__(self, name, a):
        _nargs(a, (3,), name, "NoncentralBetaDistribution[alpha, beta, delta]")
        _pos(a[0], "alpha", name)
        _pos(a[1], "beta", name)
        if _num(a[2]) and a[2] < 0:
            raise ValueError(f"{name}: delta must be non-negative")
        super().__init__(name, a)

    def pdf(self, x):
        al, be, de = self.a
        return (sp.exp(-de / 2) * x ** (al - 1) * (1 - x) ** (be - 1) / sp.beta(al, be)
                * sp.hyper([al + be], [al], de * x / 2))

    def series_cdf(self, x):
        import mpmath
        al, be, de = (_mp(t) for t in self.a)
        return mpmath.nsum(lambda j: mpmath.exp(-de / 2) * (de / 2) ** j / mpmath.factorial(j)
                           * mpmath.betainc(al + j, be, 0, x, regularized=True), [0, mpmath.inf])

    def raw(self, r):
        al, be, de = self.a
        return (sp.exp(-de / 2) * sp.rf(al, r) / sp.rf(al + be, r)
                * sp.hyper([al + r, al + be], [al, al + be + r], de / 2))

    def mean(self):
        return self.raw(1)

    def var(self):
        return self.raw(2) - self.raw(1) ** 2

    def sample(self, rng):
        al, be, de = (_f(t) for t in self.a)
        k = W.Poisson("PoissonDistribution", [sp.Float(de / 2)]).sample(rng) if de > 0 else 0
        g1 = rng.gammavariate(al + k, 1.0)
        return g1 / (g1 + rng.gammavariate(be, 1.0))


class NoncentralFRatio(_Noncentral):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (3,), name, "NoncentralFRatioDistribution[n, m, lambda]")
        _pos(a[0], "n", name)
        _pos(a[1], "m", name)
        if _num(a[2]) and a[2] < 0:
            raise ValueError(f"{name}: lambda must be non-negative")
        super().__init__(name, a)

    def pdf(self, x):
        n, m, l = self.a
        return (sp.exp(-l / 2) * (n / m) ** (n / 2) * x ** (n / 2 - 1) * (1 + n * x / m) ** (-(n + m) / 2)
                / sp.beta(n / 2, m / 2) * sp.hyper([(n + m) / 2], [n / 2], l * n * x / (2 * (m + n * x))))

    def series_cdf(self, x):
        import mpmath
        n, m, l = (_mp(t) for t in self.a)
        z = n * x / (n * x + m)
        return mpmath.nsum(lambda j: mpmath.exp(-l / 2) * (l / 2) ** j / mpmath.factorial(j)
                           * mpmath.betainc(n / 2 + j, m / 2, 0, z, regularized=True), [0, mpmath.inf])

    def mean(self):
        n, m, l = self.a
        if _num(m) and not m > 2:
            return sp.oo
        return m * (n + l) / (n * (m - 2))

    def var(self):
        n, m, l = self.a
        if _num(m) and not m > 4:
            return sp.oo if m > 2 else sp.nan
        return 2 * (m / n) ** 2 * ((n + l) ** 2 + (n + 2 * l) * (m - 2)) / ((m - 2) ** 2 * (m - 4))

    def raw(self, r):
        if r <= 2:
            m1 = self.mean()
            if r == 1:
                return m1
            v = self.var()
            return v if v in (sp.oo, sp.nan) else v + m1 ** 2
        return _numeric_moment(self, lambda x: x ** r, "Moment")

    def sample(self, rng):
        n, m, l = (_f(t) for t in self.a)
        num = NoncentralChiSquare("NoncentralChiSquareDistribution", [sp.Float(n), sp.Float(l)]).sample(rng)
        return (num / n) / (rng.gammavariate(m / 2, 2.0) / m)


class NoncentralStudentT(_Noncentral):
    def __init__(self, name, a):
        _nargs(a, (2,), name, "NoncentralStudentTDistribution[nu, delta]")
        _pos(a[0], "nu", name)
        _real(a[1], "delta", name)
        super().__init__(name, a)

    def pdf(self, x):
        v, d = self.a
        s = v + x ** 2
        z = d ** 2 * x ** 2 / (2 * s)
        pre = v ** (v / 2) * sp.gamma(v + 1) * sp.exp(-d ** 2 / 2) / (2 ** v * s ** (v / 2) * sp.gamma(v / 2))
        return pre * (sp.sqrt(2) * d * x * sp.hyper([v / 2 + 1], [sp.Rational(3, 2)], z) / (s * sp.gamma((v + 1) / 2))
                      + sp.hyper([(v + 1) / 2], [sp.Rational(1, 2)], z) / (sp.sqrt(s) * sp.gamma(v / 2 + 1)))

    def series_cdf(self, x):
        import mpmath
        v, d = (_mp(t) for t in self.a)
        # P(T <= x) = E_V[Phi(x Sqrt[V/nu] - delta)], V ~ ChiSquare(nu)
        g = lambda w: (mpmath.ncdf(x * mpmath.sqrt(w / v) - d) * w ** (v / 2 - 1) * mpmath.exp(-w / 2)  # noqa: E731
                       / (2 ** (v / 2) * mpmath.gamma(v / 2)))
        return mpmath.quad(g, [0, v, mpmath.inf])

    def center(self):
        return self.a[1]

    def mean(self):
        v, d = self.a
        if _num(v) and not v > 1:
            return sp.nan
        return d * sp.sqrt(v / 2) * sp.gamma((v - 1) / 2) / sp.gamma(v / 2)

    def var(self):
        v, d = self.a
        if _num(v) and not v > 2:
            return sp.oo if v > 1 else sp.nan
        return v * (1 + d ** 2) / (v - 2) - self.mean() ** 2

    def raw(self, r):
        v, d = self.a
        if _num(v) and not v > r:
            return sp.nan
        # E[T^r] = (nu/2)^(r/2) Gamma((nu - r)/2)/Gamma(nu/2) E[(Z + delta)^r]
        ez = sp.Add(*(sp.binomial(r, k) * d ** (r - k) * (sp.factorial2(k - 1) if k % 2 == 0 else 0)
                      for k in range(r + 1)))
        return (v / 2) ** sp.Rational(r, 2) * sp.gamma((v - r) / 2) / sp.gamma(v / 2) * ez

    def sample(self, rng):
        v, d = (_f(t) for t in self.a)
        return (rng.gauss(0, 1) + d) / math.sqrt(rng.gammavariate(v / 2, 2.0) / v)


class _ScaledBeta(UD):
    """Beta(alpha, beta) on [a, b]"""

    def bounds(self):
        return self.ab()

    def shape(self):
        raise NotImplementedError

    def pdf(self, x):
        a, b = self.ab()
        al, be = self.shape()
        return (x - a) ** (al - 1) * (b - x) ** (be - 1) / (sp.beta(al, be) * (b - a) ** (al + be - 1))

    def cdf(self, x):
        a, b = self.ab()
        al, be = self.shape()
        return W.Beta("BetaDistribution", [al, be]).cdf((x - a) / (b - a))

    def raw(self, r):
        a, b = self.ab()
        al, be = self.shape()
        return sp.Add(*(sp.binomial(r, k) * a ** (r - k) * (b - a) ** k * sp.rf(al, k) / sp.rf(al + be, k)
                        for k in range(r + 1)))

    def mean(self):
        a, b = self.ab()
        al, be = self.shape()
        return a + (b - a) * al / (al + be)

    def var(self):
        a, b = self.ab()
        al, be = self.shape()
        return (b - a) ** 2 * al * be / ((al + be) ** 2 * (al + be + 1))

    def sample(self, rng):
        a, b = (_f(v) for v in self.ab())
        al, be = (_f(v) for v in self.shape())
        return a + (b - a) * rng.betavariate(al, be)


class PERT(_ScaledBeta):
    def __init__(self, name, a):
        if len(a) not in (2, 3):
            raise ValueError(f"{name}: expected PERTDistribution[{{min, max}}, c] or [{{min, max}}, c, lambda]")
        lo, hi = _interval(a[0], name)
        c = _scalar(a[1], name)
        lam = _scalar(a[2], name) if len(a) == 3 else sp.Integer(4)
        if all(_num(v) for v in (lo, hi, c)) and not lo <= c <= hi:
            raise ValueError(f"{name}: the mode c must lie in [min, max]")
        _pos(lam, "lambda", name)
        super().__init__(name, [[lo, hi], c, lam])

    def ab(self):
        return tuple(self.a[0])

    def shape(self):
        (a, b), c, lam = self.a
        return 1 + lam * (c - a) / (b - a), 1 + lam * (b - c) / (b - a)


class ParetoPickands(UD):
    def __init__(self, name, a):
        _nargs(a, (3,), name, "ParetoPickandsDistribution[mu, sigma, xi]")
        _pos(a[1], "sigma", name)
        _real(a[2], "xi", name)
        super().__init__(name, a)

    def bounds(self):
        m, s, xi = self.a
        if _num(xi) and xi < 0:
            return m, m - s / xi
        return m, sp.oo

    def cdf(self, x):
        m, s, xi = self.a
        if xi == 0:
            return 1 - sp.exp(-(x - m) / s)
        return 1 - (1 + xi * (x - m) / s) ** (-1 / xi)

    def pdf(self, x):
        m, s, xi = self.a
        if xi == 0:
            return sp.exp(-(x - m) / s) / s
        return (1 + xi * (x - m) / s) ** (-1 / xi - 1) / s

    def q(self, p):
        m, s, xi = self.a
        if xi == 0:
            return m - s * sp.log(1 - p)
        return m + s * ((1 - p) ** (-xi) - 1) / xi

    def raw(self, r):
        m, s, xi = self.a
        if xi == 0:
            return sp.Add(*(sp.binomial(r, k) * m ** (r - k) * s ** k * sp.factorial(k) for k in range(r + 1)))
        if _num(xi) and not xi * r < 1:
            return sp.oo
        # X = m + s (U^-xi - 1)/xi, U uniform; E[U^(-k xi)] = 1/(1 - k xi)
        c = s / xi
        return sp.Add(*(sp.binomial(r, k) * (m - c) ** (r - k) * c ** k / (1 - k * xi) for k in range(r + 1)))

    def mean(self):
        return self.raw(1)

    def var(self):
        m, s, xi = self.a
        if _num(xi) and not xi < sp.Rational(1, 2):
            return sp.oo if xi < 1 else sp.nan
        return s ** 2 / ((1 - xi) ** 2 * (1 - 2 * xi))


class PowerDist(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "PowerDistribution[k, a]")
        _pos(a[0], "k", name)
        _pos(a[1], "a", name)
        super().__init__(name, a)

    def bounds(self):
        return sp.Integer(0), 1 / self.a[0]

    def pdf(self, x):
        k, a = self.a
        return a * k ** a * x ** (a - 1)

    def cdf(self, x):
        k, a = self.a
        return (k * x) ** a

    def q(self, p):
        k, a = self.a
        return p ** (1 / a) / k

    def raw(self, r):
        k, a = self.a
        return a / ((a + r) * k ** r)

    def mean(self):
        return self.raw(1)

    def var(self):
        k, a = self.a
        return a / (k ** 2 * (a + 1) ** 2 * (a + 2))


class Rice(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        if len(a) == 3:
            raise NotImplementedError(f"{name}: the three-parameter form RiceDistribution[m, alpha, beta] is not supported")
        _nargs(a, (2,), name, "RiceDistribution[alpha, beta]")
        if _num(a[0]) and a[0] < 0:
            raise ValueError(f"{name}: alpha must be non-negative")
        _pos(a[1], "beta", name)
        super().__init__(name, a)

    def pdf(self, x):
        al, be = self.a
        return x / be ** 2 * sp.exp(-(x ** 2 + al ** 2) / (2 * be ** 2)) * sp.besseli(0, x * al / be ** 2)

    def cdf(self, x):
        al, be = self.a
        return 1 - sp.marcumq(1, al / be, x / be)

    def raw(self, r):
        al, be = self.a
        return (be ** r * 2 ** sp.Rational(r, 2) * sp.gamma(1 + sp.Rational(r, 2))
                * sp.hyper([-sp.Rational(r, 2)], [1], -al ** 2 / (2 * be ** 2)))

    def mean(self):
        return self.raw(1)

    def var(self):
        al, be = self.a
        return 2 * be ** 2 + al ** 2 - self.raw(1) ** 2

    def sample(self, rng):
        al, be = (_f(v) for v in self.a)
        return math.hypot(rng.gauss(al, be), rng.gauss(0, be))


class Sech(UD):
    def __init__(self, name, a):
        _nargs(a, (0, 2), name, "SechDistribution[] or [mu, sigma]")
        if not a:
            a = [sp.Integer(0), sp.Integer(1)]
        _pos(a[1], "sigma", name)
        super().__init__(name, a)

    def pdf(self, x):
        m, s = self.a
        return sp.sech(sp.pi * (x - m) / (2 * s)) / (2 * s)

    def cdf(self, x):
        m, s = self.a
        return 2 * sp.atan(sp.exp(sp.pi * (x - m) / (2 * s))) / sp.pi

    def q(self, p):
        m, s = self.a
        return m + 2 * s / sp.pi * sp.log(sp.tan(sp.pi * p / 2))

    def mgf(self, t):
        m, s = self.a
        return sp.exp(m * t) * sp.sec(s * t)

    def mean(self):
        return self.a[0]

    def var(self):
        return self.a[1] ** 2

    def skewness(self):
        return sp.Integer(0)

    def kurtosis(self):
        return sp.Integer(5)


class ShiftedGompertz(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "ShiftedGompertzDistribution[lambda, xi]")
        _pos(a[0], "lambda", name)
        _pos(a[1], "xi", name)
        super().__init__(name, a)

    def pdf(self, x):
        l, xi = self.a
        e = sp.exp(-l * x)
        return l * e * sp.exp(-xi * e) * (1 + xi * (1 - e))

    def cdf(self, x):
        l, xi = self.a
        e = sp.exp(-l * x)
        return (1 - e) * sp.exp(-xi * e)

    def mean(self):
        l, xi = self.a
        return (sp.EulerGamma + sp.log(xi) + sp.expint(1, xi) + (1 - sp.exp(-xi)) / xi) / l

    def var(self):
        return _numeric_moment(self, lambda x: x ** 2, "Variance") - self.mean() ** 2

    def raw(self, r):
        if r == 1:
            return self.mean()
        return _numeric_moment(self, lambda x: x ** r, "Moment")

    def sample(self, rng):
        l, xi = (_f(v) for v in self.a)
        # max of Exponential(lambda) and Y with CDF Exp[-xi e^(-lambda y)]
        y = -math.log(-math.log(1 - rng.random()) / xi) / l
        return max(rng.expovariate(l), y)


class SkewNormal(UD):
    closed_cdf = False

    def __init__(self, name, a):
        _nargs(a, (3,), name, "SkewNormalDistribution[mu, sigma, alpha]")
        _pos(a[1], "sigma", name)
        _real(a[2], "alpha", name)
        super().__init__(name, a)

    def pdf(self, x):
        m, s, al = self.a
        z = (x - m) / s
        return sp.exp(-z ** 2 / 2) / (s * sp.sqrt(2 * sp.pi)) * sp.erfc(-al * z / sp.sqrt(2))

    def cdf(self, x):
        if self.a[2] == 0:
            m, s, _ = self.a
            return sp.erfc((m - x) / (s * sp.sqrt(2))) / 2
        return self.num_cdf(x)

    def center(self):
        return self.a[0]

    def _d(self):
        al = self.a[2]
        return al / sp.sqrt(1 + al ** 2)

    def mean(self):
        m, s, al = self.a
        return m + s * self._d() * sp.sqrt(2 / sp.pi)

    def var(self):
        m, s, al = self.a
        return s ** 2 * (1 - 2 * self._d() ** 2 / sp.pi)

    def mgf(self, t):
        m, s, al = self.a
        return sp.exp(m * t + s ** 2 * t ** 2 / 2) * sp.erfc(-s * self._d() * t / sp.sqrt(2))

    def skewness(self):
        d = self._d()
        b = d * sp.sqrt(2 / sp.pi)
        return (4 - sp.pi) / 2 * b ** 3 / (1 - 2 * d ** 2 / sp.pi) ** sp.Rational(3, 2)

    def kurtosis(self):
        d = self._d()
        b = d * sp.sqrt(2 / sp.pi)
        return 3 + 2 * (sp.pi - 3) * b ** 4 / (1 - 2 * d ** 2 / sp.pi) ** 2

    def sample(self, rng):
        m, s, al = (_f(v) for v in self.a)
        d = al / math.sqrt(1 + al * al)
        return m + s * (d * abs(rng.gauss(0, 1)) + math.sqrt(1 - d * d) * rng.gauss(0, 1))


class TsallisQExponential(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "TsallisQExponentialDistribution[lambda, q]")
        _pos(a[0], "lambda", name)
        if _num(a[1]) and not a[1] < 2:
            raise ValueError(f"{name}: q must be less than 2")
        super().__init__(name, a)

    def bounds(self):
        l, q = self.a
        if _num(q) and q < 1:
            return sp.Integer(0), 1 / (l * (1 - q))
        return sp.Integer(0), sp.oo

    def pdf(self, x):
        l, q = self.a
        if q == 1:
            return l * sp.exp(-l * x)
        return (2 - q) * l * (1 - (1 - q) * l * x) ** (1 / (1 - q))

    def cdf(self, x):
        l, q = self.a
        if q == 1:
            return 1 - sp.exp(-l * x)
        return 1 - (1 - (1 - q) * l * x) ** ((2 - q) / (1 - q))

    def q(self, p):
        l, q = self.a
        if q == 1:
            return -sp.log(1 - p) / l
        return (1 - (1 - p) ** ((1 - q) / (2 - q))) / ((1 - q) * l)

    def raw(self, r):
        l, q = self.a
        if _num(q) and not q < sp.Rational(r + 2, r + 1):
            return sp.oo
        return sp.factorial(r) / sp.Mul(*(l * ((j + 2) - (j + 1) * q) for j in range(1, r + 1)))

    def mean(self):
        return self.raw(1)

    def var(self):
        l, q = self.a
        if _num(q) and not q < sp.Rational(4, 3):
            return sp.oo if q < sp.Rational(3, 2) else sp.nan
        return (2 - q) / (l ** 2 * (3 - 2 * q) ** 2 * (4 - 3 * q))


class TsallisQGaussian(UD):
    def __init__(self, name, a):
        _nargs(a, (1, 3), name, "TsallisQGaussianDistribution[q] or [mu, beta, q]")
        if len(a) == 1:
            a = [sp.Integer(0), sp.Integer(1), a[0]]
        _pos(a[1], "beta", name)
        if _num(a[2]) and not a[2] < 3:
            raise ValueError(f"{name}: q must be less than 3")
        super().__init__(name, a)

    def _cq(self):
        q = self.a[2]
        if q == 1:
            return sp.sqrt(sp.pi)
        if _num(q) and q < 1:
            return 2 * sp.sqrt(sp.pi) * sp.gamma(1 / (1 - q)) / ((3 - q) * sp.sqrt(1 - q) * sp.gamma((3 - q) / (2 * (1 - q))))
        return sp.sqrt(sp.pi) * sp.gamma((3 - q) / (2 * (q - 1))) / (sp.sqrt(q - 1) * sp.gamma(1 / (q - 1)))

    def bounds(self):
        m, b, q = self.a
        if not _num(q):
            raise Unevaluated()
        if q < 1:
            w = b * sp.sqrt(2 / (1 - q))
            return m - w, m + w
        return -sp.oo, sp.oo

    def pdf(self, x):
        m, b, q = self.a
        u = -(x - m) ** 2 / (2 * b ** 2)
        e = sp.exp(u) if q == 1 else (1 + (1 - q) * u) ** (1 / (1 - q))
        return e / (b * sp.sqrt(2) * self._cq())

    def _base(self):
        """(c, scale, D) with X = mu + scale * Y, Y ~ D"""
        m, b, q = self.a
        if q == 1:
            return m, b, W.Normal("NormalDistribution", [sp.Integer(0), sp.Integer(1)])
        if q < 1:
            a = (2 - q) / (1 - q)
            # X = mu + b Sqrt[2/(1 - q)] (2 U - 1), U ~ Beta(a, a)
            w = b * sp.sqrt(2 / (1 - q))
            return m - w, 2 * w, W.Beta("BetaDistribution", [a, a])
        nu = (3 - q) / (q - 1)
        return m, b * sp.sqrt(2) / sp.sqrt(3 - q), W.StudentT("StudentTDistribution", [nu])

    def cdf(self, x):
        if not _num(self.a[2]):
            raise Unevaluated()
        c, s, base = self._base()
        return W.cdf_value(base, (x - c) / s)

    def center(self):
        return self.a[0]

    def mean(self):
        m, b, q = self.a
        if _num(q) and not q < 2:
            return sp.nan
        return m

    def var(self):
        m, b, q = self.a
        if _num(q) and not q < sp.Rational(5, 3):
            return sp.oo if q < 2 else sp.nan
        return 2 * b ** 2 / (5 - 3 * q)

    def skewness(self):
        q = self.a[2]
        if _num(q) and not q < sp.Rational(7, 5):
            return sp.nan
        return sp.Integer(0)

    def kurtosis(self):
        q = self.a[2]
        if _num(q) and not q < sp.Rational(7, 5):
            return sp.nan
        return 3 + 6 * (q - 1) / (7 - 5 * q)

    def sample(self, rng):
        c, s, base = self._base()
        return _f(c) + _f(s) * base.sample(rng)


class NumericOnly(UD):
    """distributions whose PDF and CDF have no closed form: computed numerically (mpmath) for numeric x when x or
    a parameter is an inexact number; exact numeric x raises NotImplementedError, symbolic x stays unevaluated"""

    def _check(self, x, what):
        if not (_num(x) and self.numeric()):
            raise Unevaluated()
        if not (_inexact(x) or any(_inexact(v) for v in self.flat())):
            raise NotImplementedError(f"{what}: {self.name} has no closed form; use a machine-number argument")

    def pdf_value(self, x):
        self._check(x, "PDF")
        lo, hi = self.bounds()
        if (lo != -sp.oo and _f(x) < _f(lo)) or (hi != sp.oo and _f(x) > _f(hi)):
            return sp.Integer(0)
        return sp.Float(self.npdf(_mp(x)))

    def cdf_value(self, x):
        self._check(x, "CDF")
        lo, hi = self.bounds()
        if lo != -sp.oo and _f(x) <= _f(lo):
            return sp.Integer(0)
        if hi != sp.oo and _f(x) >= _f(hi):
            return sp.Integer(1)
        return sp.Float(self.ncdf(_mp(x)))

    def pdf(self, x):
        raise NotImplementedError(f"{self.name}: the density has no closed form")

    def cdf(self, x):
        return self.cdf_value(x)

    def expect(self, f, v, numeric):
        import mpmath
        if not self.numeric():
            raise Unevaluated()
        if not numeric:
            raise NotImplementedError(f"Expectation: {self.name} has no closed-form density; use NExpectation")
        g = sp.lambdify(v, f, "mpmath")
        lo, hi = self.bounds()
        return sp.Float(mpmath.quad(lambda t: g(t) * self.npdf(t),
                                    [-mpmath.inf if lo == -sp.oo else _mp(lo), mpmath.inf if hi == sp.oo else _mp(hi)]))


class _QuantileDefined(NumericOnly):
    """distributions defined by a closed-form quantile function Q(p): CDF by root finding, PDF = 1/Q'(p)"""

    def _p_of(self, x):
        import mpmath
        Q = sp.lambdify(sp.Symbol("p"), self.q(sp.Symbol("p")), "mpmath")
        a, b = mpmath.mpf(0), mpmath.mpf(1)
        for _ in range(200):
            m = (a + b) / 2
            if Q(m) < x:
                a = m
            else:
                b = m
        return (a + b) / 2

    def npdf(self, x):
        p = sp.Symbol("p")
        dq = sp.lambdify(p, sp.diff(self.q(p), p), "mpmath")
        return 1 / dq(self._p_of(x))

    def ncdf(self, x):
        return self._p_of(x)

    def quantile_value(self, p, inputs, fname):
        return self.q(p)


class TukeyLambda(_QuantileDefined):
    def __init__(self, name, a):
        if len(a) == 3 and isinstance(a[0], list):
            raise NotImplementedError(f"{name}: the generalised form [{{l1, l2}}, mu, {{s1, s2}}] is not supported")
        _nargs(a, (1, 3), name, "TukeyLambdaDistribution[lambda] or [lambda, mu, sigma]")
        if len(a) == 1:
            a = [a[0], sp.Integer(0), sp.Integer(1)]
        _real(a[0], "lambda", name)
        _pos(a[2], "sigma", name)
        super().__init__(name, a)

    def bounds(self):
        l, m, s = self.a
        if not _num(l):
            raise Unevaluated()
        if l > 0:
            return m - s / l, m + s / l
        return -sp.oo, sp.oo

    def q(self, p):
        l, m, s = self.a
        if l == 0:
            return m + s * sp.log(p / (1 - p))
        return m + s * (p ** l - (1 - p) ** l) / l

    def pdf_value(self, x):
        l, m, s = self.a
        if l == 0:
            return W.pdf_value(W.Logistic("LogisticDistribution", [m, s]), x)
        if l == 1:
            return W.pdf_value(W.Uniform("UniformDistribution", [m - s, m + s]), x)
        return super().pdf_value(x)

    def cdf_value(self, x):
        l, m, s = self.a
        if l == 0:
            return W.cdf_value(W.Logistic("LogisticDistribution", [m, s]), x)
        if l == 1:
            return W.cdf_value(W.Uniform("UniformDistribution", [m - s, m + s]), x)
        return super().cdf_value(x)

    def mean(self):
        l, m, s = self.a
        if _num(l) and not l > -1:
            return sp.nan
        return m

    def var(self):
        l, m, s = self.a
        if l == 0:
            return s ** 2 * sp.pi ** 2 / 3
        if _num(l) and not l > -sp.Rational(1, 2):
            return sp.nan
        return s ** 2 * 2 / l ** 2 * (1 / (1 + 2 * l) - sp.gamma(l + 1) ** 2 / sp.gamma(2 * l + 2))

    def raw(self, r):
        l, m, s = self.a
        if l == 0 or (_num(l) and not l * r > -1):
            if l == 0 and r <= 2:
                return m if r == 1 else self.var() + m ** 2
            return sp.nan if l != 0 else _numeric_moment(self, lambda x: x ** r, "Moment")
        # E[Q(U)^r] = Int_0^1 Q(p)^r dp, binomial expansion of (p^l - (1 - p)^l)^r: Beta integrals
        out = 0
        for j in range(r + 1):
            c = sp.binomial(r, j) * m ** (r - j) * (s / l) ** j
            out += c * sp.Add(*(sp.binomial(j, k) * (-1) ** (j - k) * sp.beta(k * l + 1, (j - k) * l + 1)
                                for k in range(j + 1)))
        return out

    def skewness(self):
        l = self.a[0]
        if _num(l) and not l > -sp.Rational(1, 3):
            return sp.nan
        return sp.Integer(0)

    def sample(self, rng):
        return _f(self.q(sp.Float(rng.random())))


class Wakeby(_QuantileDefined):
    def __init__(self, name, a):
        _nargs(a, (5,), name, "WakebyDistribution[alpha, beta, gamma, delta, mu]")
        al, be, ga, de, m = a
        for v, w in ((al, "alpha"), (be, "beta"), (ga, "gamma"), (de, "delta")):
            _real(v, w, name)
        if all(_num(v) for v in a):
            if not (be + de > 0 or (be == 0 and de == 0 and ga == 0)) and not (al == 0 or ga == 0):
                pass
            if al + ga < 0 or ga < 0 or de >= 1:
                raise ValueError(f"{name}: need gamma >= 0, alpha + gamma >= 0 and delta < 1")
        super().__init__(name, a)

    def q(self, p):
        al, be, ga, de, m = self.a
        t1 = al * (1 - (1 - p) ** be) / be if be != 0 else -al * sp.log(1 - p)
        t2 = ga * (1 - (1 - p) ** (-de)) / de if de != 0 else ga * sp.log(1 - p)
        return m + t1 - t2

    def bounds(self):
        al, be, ga, de, m = self.a
        if not self.numeric():
            raise Unevaluated()
        if de < 0 and ga > 0 or (ga == 0 and be > 0):
            hi = m + (al / be if be != 0 else 0) - (ga / de if de != 0 else 0)
            if ga == 0 and be > 0:
                hi = m + al / be
            return m, hi
        return m, sp.oo

    def mean(self):
        al, be, ga, de, m = self.a
        return m + al / (1 + be) + ga / (1 - de)

    def var(self):
        al, be, ga, de, m = self.a
        if _num(de) and not de < sp.Rational(1, 2):
            return sp.oo
        return (al ** 2 / ((1 + be) ** 2 * (1 + 2 * be)) + 2 * al * ga / ((1 + be) * (1 - de) * (1 + be - de))
                + ga ** 2 / ((1 - de) ** 2 * (1 - 2 * de)))

    def sample(self, rng):
        return _f(self.q(sp.Float(rng.random())))


class VarianceGamma(UD):
    closed_cdf = False

    def __init__(self, name, a):
        _nargs(a, (4,), name, "VarianceGammaDistribution[lambda, alpha, beta, mu]")
        lam, al, be, m = a
        _pos(lam, "lambda", name)
        _pos(al, "alpha", name)
        if _num(al) and _num(be) and not abs(be) < al:
            raise ValueError(f"{name}: need |beta| < alpha")
        super().__init__(name, a)

    def pdf(self, x):
        lam, al, be, m = self.a
        g = sp.sqrt(al ** 2 - be ** 2)
        d = sp.Abs(x - m)
        return (g ** (2 * lam) * d ** (lam - sp.Rational(1, 2)) * sp.besselk(lam - sp.Rational(1, 2), al * d)
                * sp.exp(be * (x - m)) / (sp.sqrt(sp.pi) * sp.gamma(lam) * (2 * al) ** (lam - sp.Rational(1, 2))))

    def cdf(self, x):
        return self.num_cdf(x)

    def center(self):
        return self.a[3]

    def mgf(self, t):
        lam, al, be, m = self.a
        return sp.exp(m * t) * ((al ** 2 - be ** 2) / (al ** 2 - (be + t) ** 2)) ** lam

    def mean(self):
        lam, al, be, m = self.a
        return m + 2 * be * lam / (al ** 2 - be ** 2)

    def var(self):
        lam, al, be, m = self.a
        g2 = al ** 2 - be ** 2
        return 2 * lam * (1 + 2 * be ** 2 / g2) / g2

    def sample(self, rng):
        lam, al, be, m = (_f(v) for v in self.a)
        g2 = al * al - be * be
        w = rng.gammavariate(lam, 2.0 / g2)
        return m + be * w + math.sqrt(w) * rng.gauss(0, 1)


class Voigt(UD):
    closed_cdf = False

    def __init__(self, name, a):
        _nargs(a, (2,), name, "VoigtDistribution[delta, sigma]")
        _pos(a[0], "delta", name)
        _pos(a[1], "sigma", name)
        super().__init__(name, a)

    def pdf(self, x):
        d, s = self.a
        z = (x + sp.I * d) / (s * sp.sqrt(2))
        return sp.re(sp.exp(-z ** 2) * sp.erfc(-sp.I * z)) / (s * sp.sqrt(2 * sp.pi))

    def cdf(self, x):
        return self.num_cdf(x)

    def cf(self, t):
        d, s = self.a
        return sp.exp(-d * sp.Abs(t) - s ** 2 * t ** 2 / 2)

    def mean(self):
        return sp.nan

    def var(self):
        return sp.nan

    def raw(self, r):
        return sp.nan

    def skewness(self):
        return sp.nan

    def kurtosis(self):
        return sp.nan

    def sample(self, rng):
        d, s = (_f(v) for v in self.a)
        return rng.gauss(0, s) + d * math.tan(math.pi * (rng.random() - 0.5))


class VonMises(UD):
    closed_cdf = False
    open_lo = open_hi = False

    def __init__(self, name, a):
        _nargs(a, (2,), name, "VonMisesDistribution[mu, kappa]")
        _real(a[0], "mu", name)
        _pos(a[1], "kappa", name)
        super().__init__(name, a)

    def bounds(self):
        m = self.a[0]
        return m - sp.pi, m + sp.pi

    def pdf(self, x):
        m, k = self.a
        return sp.exp(k * sp.cos(x - m)) / (2 * sp.pi * sp.besseli(0, k))

    def cdf(self, x):
        return self.num_cdf(x)

    def center(self):
        return self.a[0]

    def mean(self):
        return self.a[0]

    def var(self):
        return _numeric_moment(self, lambda x: (x - self.a[0]) ** 2, "Variance")

    def raw(self, r):
        if r == 1:
            return self.a[0]
        return _numeric_moment(self, lambda x: x ** r, "Moment")

    def skewness(self):
        return sp.Integer(0)

    def sample(self, rng):
        m, k = (_f(v) for v in self.a)
        v = rng.vonmisesvariate(0.0, k)
        if v > math.pi:
            v -= 2 * math.pi
        return m + v


class WignerSemicircle(UD):
    def __init__(self, name, a):
        _nargs(a, (1, 2), name, "WignerSemicircleDistribution[r] or [a, r]")
        if len(a) == 1:
            a = [sp.Integer(0), a[0]]
        _pos(a[1], "r", name)
        super().__init__(name, a)

    def bounds(self):
        a, r = self.a
        return a - r, a + r

    def pdf(self, x):
        a, r = self.a
        return 2 / (sp.pi * r ** 2) * sp.sqrt(r ** 2 - (x - a) ** 2)

    def cdf(self, x):
        a, r = self.a
        u = x - a
        return sp.Rational(1, 2) + u * sp.sqrt(r ** 2 - u ** 2) / (sp.pi * r ** 2) + sp.asin(u / r) / sp.pi

    def raw(self, r_):
        a, r = self.a
        c = lambda k: (r / 2) ** (2 * k) * sp.catalan(k)  # noqa: E731
        return sp.Add(*(sp.binomial(r_, j) * a ** (r_ - j) * (c(j // 2) if j % 2 == 0 else 0) for j in range(r_ + 1)))

    def mean(self):
        return self.a[0]

    def var(self):
        return self.a[1] ** 2 / 4

    def skewness(self):
        return sp.Integer(0)

    def kurtosis(self):
        return sp.Integer(2)

    def sample(self, rng):
        a, r = (_f(v) for v in self.a)
        return a + r * (2 * rng.betavariate(1.5, 1.5) - 1)


class BenktanderGibrat(UD):
    lo = sp.Integer(1)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "BenktanderGibratDistribution[a, b]")
        _pos(a[0], "a", name)
        _pos(a[1], "b", name)
        if all(_num(v) for v in a) and not a[1] <= a[0] * (a[0] + 1) / 2:
            raise ValueError(f"{name}: need b <= a (a + 1)/2")
        super().__init__(name, a)

    def cdf(self, x):
        a, b = self.a
        lg = sp.log(x)
        return 1 - (1 + 2 * b / a * lg) * x ** (-(a + 1 + b * lg))

    def pdf(self, x):
        a, b = self.a
        lg = sp.log(x)
        return x ** (-2 - a - b * lg) * ((1 + 2 * b * lg / a) * (1 + a + 2 * b * lg) - 2 * b / a)

    def mean(self):
        return 1 + 1 / self.a[0]

    def var(self):
        a, b = self.a
        return (-sp.sqrt(b) + a * sp.exp((a - 1) ** 2 / (4 * b)) * sp.sqrt(sp.pi) * sp.erfc((a - 1) / (2 * sp.sqrt(b)))) / (a ** 2 * sp.sqrt(b))

    def raw(self, r):
        if r <= 2:
            m = self.mean()
            return m if r == 1 else self.var() + m ** 2
        return _numeric_moment(self, lambda x: x ** r, "Moment")


class GompertzMakeham(UD):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        if len(a) == 4:
            raise NotImplementedError(f"{name}: the four-parameter Gompertz-Makeham form is not supported")
        _nargs(a, (2,), name, "GompertzMakehamDistribution[lambda, xi]")
        _pos(a[0], "lambda", name)
        _pos(a[1], "xi", name)
        super().__init__(name, a)

    def pdf(self, x):
        l, xi = self.a
        return l * xi * sp.exp(l * x + xi * (1 - sp.exp(l * x)))

    def cdf(self, x):
        l, xi = self.a
        return 1 - sp.exp(xi * (1 - sp.exp(l * x)))

    def q(self, p):
        l, xi = self.a
        return sp.log(1 - sp.log(1 - p) / xi) / l

    def mean(self):
        l, xi = self.a
        return sp.exp(xi) * sp.expint(1, xi) / l

    def var(self):
        return _numeric_moment(self, lambda x: x ** 2, "Variance") - self.mean() ** 2

    def raw(self, r):
        if r == 1:
            return self.mean()
        return _numeric_moment(self, lambda x: x ** r, "Moment")


# ── phase-type distributions (Hyperexponential, Hypoexponential, Coxian) ──

class PhaseType(UD):
    """continuous phase-type distribution with initial vector alpha and sub-generator T: S(x) = alpha e^{T x} 1"""
    lo = sp.Integer(0)

    def ph(self):
        raise NotImplementedError

    def pdf(self, x):
        al, T = self.ph()
        t0 = -T * sp.ones(T.rows, 1)
        return sp.simplify((al * (T * x).exp() * t0)[0])

    def cdf(self, x):
        al, T = self.ph()
        return sp.simplify(1 - (al * (T * x).exp() * sp.ones(T.rows, 1))[0])

    def raw(self, r):
        al, T = self.ph()
        return sp.simplify(sp.factorial(r) * (al * ((-T).inv() ** r) * sp.ones(T.rows, 1))[0])

    def mean(self):
        return self.raw(1)

    def var(self):
        return sp.simplify(self.raw(2) - self.raw(1) ** 2)

    def mgf(self, t):
        al, T = self.ph()
        t0 = -T * sp.ones(T.rows, 1)
        return sp.simplify((al * (-t * sp.eye(T.rows) - T).inv() * t0)[0])

    def numeric(self):
        return all(_num(v) for v in self.flat())

    def sample(self, rng):
        al, T = self.ph()
        n = T.rows
        a = [_f(v) for v in al]
        Tm = [[_f(T[i, j]) for j in range(n)] for i in range(n)]
        u, acc, state = rng.random(), 0.0, n - 1
        for i, p in enumerate(a):
            acc += p
            if u < acc:
                state = i
                break
        t = 0.0
        while state is not None:
            rate = -Tm[state][state]
            t += rng.expovariate(rate)
            u, acc, nxt = rng.random() * rate, 0.0, None
            for j in range(n):
                if j != state:
                    acc += Tm[state][j]
                    if u < acc:
                        nxt = j
                        break
            state = nxt
        return t


def _vec(v, name, what):
    if not isinstance(v, list):
        raise TypeError(f"{name}: {what} must be a list")
    return v


class Hyperexponential(PhaseType):
    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError(f"{name}: expected HyperexponentialDistribution[{{a1, ...}}, {{l1, ...}}]")
        ps, ls = _vec(a[0], name, "the phase probabilities"), _vec(a[1], name, "the rates")
        if len(ps) != len(ls) or not ps:
            raise ValueError(f"{name}: the probability and rate lists must have the same positive length")
        for l in ls:
            _pos(l, "each rate", name)
        if all(_num(p) for p in ps) and (any(p < 0 for p in ps) or sp.simplify(sp.Add(*ps) - 1) != 0):
            raise ValueError(f"{name}: phase probabilities must be non-negative and sum to 1")
        super().__init__(name, a)

    def ph(self):
        ps, ls = self.a
        return sp.Matrix([ps]), sp.diag(*[-l for l in ls])

    def pdf(self, x):
        ps, ls = self.a
        return sp.Add(*(p * l * sp.exp(-l * x) for p, l in zip(ps, ls)))

    def cdf(self, x):
        ps, ls = self.a
        return 1 - sp.Add(*(p * sp.exp(-l * x) for p, l in zip(ps, ls)))

    def raw(self, r):
        ps, ls = self.a
        return sp.Add(*(p * sp.factorial(r) / l ** r for p, l in zip(ps, ls)))

    def mgf(self, t):
        ps, ls = self.a
        return sp.Add(*(p * l / (l - t) for p, l in zip(ps, ls)))


class Hypoexponential(PhaseType):
    def __init__(self, name, a):
        if len(a) != 1:
            raise ValueError(f"{name}: expected HypoexponentialDistribution[{{l1, ...}}]")
        ls = _vec(a[0], name, "the rates")
        if not ls:
            raise ValueError(f"{name}: at least one rate is required")
        for l in ls:
            _pos(l, "each rate", name)
        super().__init__(name, a)

    def ph(self):
        ls = self.a[0]
        n = len(ls)
        T = sp.zeros(n, n)
        for i, l in enumerate(ls):
            T[i, i] = -l
            if i + 1 < n:
                T[i, i + 1] = l
        al = sp.zeros(1, n)
        al[0, 0] = 1
        return al, T

    def mean(self):
        return sp.Add(*(1 / l for l in self.a[0]))

    def var(self):
        return sp.Add(*(1 / l ** 2 for l in self.a[0]))

    def mgf(self, t):
        return sp.Mul(*(l / (l - t) for l in self.a[0]))

    def sample(self, rng):
        return sum(rng.expovariate(_f(l)) for l in self.a[0])


class Coxian(PhaseType):
    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError(f"{name}: expected CoxianDistribution[{{a1, ..., a(m-1)}}, {{l1, ..., lm}}]")
        ps, ls = _vec(a[0], name, "the phase probabilities"), _vec(a[1], name, "the rates")
        if len(ps) != len(ls) - 1:
            raise ValueError(f"{name}: need m - 1 phase probabilities for m rates")
        for l in ls:
            _pos(l, "each rate", name)
        for p in ps:
            if _num(p) and not (0 < p <= 1):
                raise ValueError(f"{name}: phase probabilities must be in (0, 1]")
        super().__init__(name, a)

    def ph(self):
        ps, ls = self.a
        n = len(ls)
        T = sp.zeros(n, n)
        for i, l in enumerate(ls):
            T[i, i] = -l
            if i + 1 < n:
                T[i, i + 1] = ps[i] * l
        al = sp.zeros(1, n)
        al[0, 0] = 1
        return al, T


# ── characteristic-function distributions (Stable, Landau) ──

class Stable(NumericOnly):
    """StableDistribution[type, alpha, beta, mu, sigma] (type 0 or 1, Wolfram's S0/S1 characteristic functions);
    PDF and CDF by Fourier inversion of the characteristic function (Gil-Pelaez)"""

    def __init__(self, name, a):
        if len(a) == 2:
            a = [sp.Integer(1), a[0], a[1], sp.Integer(0), sp.Integer(1)]
        elif len(a) == 4:
            a = [sp.Integer(1)] + list(a)
        _nargs(a, (5,), name, "StableDistribution[type, alpha, beta, mu, sigma], [alpha, beta, mu, sigma] or "
                              "[alpha, beta]")
        ty, al, be, m, s = a
        if _num(ty) and ty not in (0, 1):
            raise ValueError(f"{name}: the type must be 0 or 1")
        if _num(al) and not (0 < al <= 2):
            raise ValueError(f"{name}: alpha must be in (0, 2]")
        if _num(be) and not (-1 <= be <= 1):
            raise ValueError(f"{name}: beta must be in [-1, 1]")
        _pos(s, "sigma", name)
        super().__init__(name, a)

    def mu1(self):
        """location in the S1 parametrisation"""
        ty, al, be, m, s = self.a
        if ty == 1:
            return m
        if al == 1:
            return m - 2 / sp.pi * be * s * sp.log(s)
        return m - be * s * sp.tan(sp.pi * al / 2)

    def cf(self, t):
        ty, al, be, m, s = self.a
        m1 = self.mu1()
        if al == 1:
            return sp.exp(sp.I * m1 * t - s * sp.Abs(t) * (1 + 2 * sp.I * be / sp.pi * sp.sign(t) * sp.log(sp.Abs(t))))
        return sp.exp(sp.I * m1 * t - (s * sp.Abs(t)) ** al * (1 - sp.I * be * sp.tan(sp.pi * al / 2) * sp.sign(t)))

    def _phi(self):
        import mpmath
        ty, al, be, m, s = (_mp(v) for v in self.a)
        m1 = _mp(self.mu1())
        if al == 1:
            return lambda t: mpmath.exp(1j * m1 * t - s * t * (1 + 2j * be / mpmath.pi * mpmath.log(t)))
        tn = mpmath.tan(mpmath.pi * al / 2)
        return lambda t: mpmath.exp(1j * m1 * t - (s * t) ** al * (1 - 1j * be * tn))

    def bounds(self):
        ty, al, be, m, s = self.a
        if not self.numeric():
            raise Unevaluated()
        if al < 1 and abs(be) == 1:
            m1 = self.mu1()
            return (m1, sp.oo) if be == 1 else (-sp.oo, m1)
        return -sp.oo, sp.oo

    def _quad(self, g):
        import mpmath
        s = _mp(self.a[4])
        return mpmath.quad(g, [0, 1 / s, 10 / s, mpmath.inf])

    def npdf(self, x):
        import mpmath
        ph = self._phi()
        return self._quad(lambda t: mpmath.re(ph(t) * mpmath.exp(-1j * t * x))) / mpmath.pi

    def ncdf(self, x):
        import mpmath
        ph = self._phi()
        return mpmath.mpf(1) / 2 - self._quad(lambda t: mpmath.im(mpmath.exp(-1j * t * x) * ph(t)) / t) / mpmath.pi

    def mean(self):
        ty, al, be, m, s = self.a
        if _num(al) and not al > 1:
            return sp.nan
        return self.mu1()

    def var(self):
        ty, al, be, m, s = self.a
        if al == 2:
            return 2 * s ** 2
        if _num(al):
            return sp.oo if al > 1 else sp.nan
        raise Unevaluated()

    def raw(self, r):
        if r == 1:
            return self.mean()
        if r == 2:
            v = self.var()
            return v if v in (sp.oo, sp.nan) else v + self.mean() ** 2
        ty, al, be, m, s = self.a
        if al == 2:
            return W.raw_moment(W.Normal("NormalDistribution", [m, sp.sqrt(2) * s]), r)
        return sp.nan

    def skewness(self):
        return sp.Integer(0) if self.a[1] == 2 else sp.nan

    def kurtosis(self):
        return sp.Integer(3) if self.a[1] == 2 else sp.nan

    def sample(self, rng):
        ty, al, be, m, s = (_f(v) for v in self.a)
        m1 = _f(self.mu1())
        u = math.pi * (rng.random() - 0.5)
        w = rng.expovariate(1.0)
        if al == 1:
            x = 2 / math.pi * ((math.pi / 2 + be * u) * math.tan(u)
                               - be * math.log((math.pi / 2) * w * math.cos(u) / (math.pi / 2 + be * u)))
            return s * x + 2 / math.pi * be * s * math.log(s) + m1
        z = be * math.tan(math.pi * al / 2)
        b = math.atan(z) / al
        sc = (1 + z * z) ** (1 / (2 * al))
        x = sc * math.sin(al * (u + b)) / math.cos(u) ** (1 / al) * (math.cos(u - al * (u + b)) / w) ** ((1 - al) / al)
        return s * x + m1


class Landau(Stable):
    """LandauDistribution[mu, sigma] = StableDistribution[1, 1, 1, mu, sigma]"""

    def __init__(self, name, a):
        _nargs(a, (0, 2), name, "LandauDistribution[] or [mu, sigma]")
        if not a:
            a = [sp.Integer(0), sp.Integer(1)]
        _pos(a[1], "sigma", name)
        D.__init__(self, name, [sp.Integer(1), sp.Integer(1), sp.Integer(1), a[0], a[1]])
        self.shown = a


class Suzuki(NumericOnly):
    """Rayleigh distribution whose scale is LogNormal(mu, nu)"""
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (2,), name, "SuzukiDistribution[mu, nu]")
        _real(a[0], "mu", name)
        _pos(a[1], "nu", name)
        super().__init__(name, a)

    def _mix(self, h):
        import mpmath
        m, v = (_mp(t) for t in self.a)
        # E over s ~ LogNormal(m, v) of h(s), with s = e^(m + v z), z standard normal
        return mpmath.quad(lambda z: h(mpmath.exp(m + v * z)) * mpmath.npdf(z), [-mpmath.inf, 0, mpmath.inf])

    def npdf(self, x):
        import mpmath
        return self._mix(lambda s: x / s ** 2 * mpmath.exp(-x ** 2 / (2 * s ** 2)))

    def ncdf(self, x):
        import mpmath
        return 1 - self._mix(lambda s: mpmath.exp(-x ** 2 / (2 * s ** 2)))

    def raw(self, r):
        m, v = self.a
        return sp.exp(r * m + r ** 2 * v ** 2 / 2) * 2 ** sp.Rational(r, 2) * sp.gamma(1 + sp.Rational(r, 2))

    def mean(self):
        return self.raw(1)

    def var(self):
        return sp.simplify(self.raw(2) - self.raw(1) ** 2)

    def sample(self, rng):
        m, v = (_f(t) for t in self.a)
        s = math.exp(rng.gauss(m, v))
        return s * math.sqrt(-2 * math.log(1 - rng.random()))


class Beckmann(NumericOnly):
    """norm of a binormal vector: BeckmannDistribution[mu1, mu2, sigma1, sigma2] or [..., rho]"""
    lo = sp.Integer(0)

    def __init__(self, name, a):
        _nargs(a, (4, 5), name, "BeckmannDistribution[mu1, mu2, sigma1, sigma2] or [mu1, mu2, sigma1, sigma2, rho]")
        if len(a) == 4:
            a = list(a) + [sp.Integer(0)]
        _pos(a[2], "sigma1", name)
        _pos(a[3], "sigma2", name)
        if _num(a[4]) and not (-1 < a[4] < 1):
            raise ValueError(f"{name}: rho must be in (-1, 1)")
        super().__init__(name, a)

    def _bvn(self, x, y):
        import mpmath
        m1, m2, s1, s2, r = (_mp(t) for t in self.a)
        u, v = (x - m1) / s1, (y - m2) / s2
        return mpmath.exp(-(u * u - 2 * r * u * v + v * v) / (2 * (1 - r * r))) / (2 * mpmath.pi * s1 * s2 * mpmath.sqrt(1 - r * r))

    def npdf(self, x):
        import mpmath
        return mpmath.quad(lambda th: x * self._bvn(x * mpmath.cos(th), x * mpmath.sin(th)), [0, mpmath.pi / 2, mpmath.pi,
                                                                                          3 * mpmath.pi / 2, 2 * mpmath.pi])

    def ncdf(self, x):
        import mpmath
        return mpmath.quad(lambda rr: self.npdf(rr), [0, x])

    def raw(self, r):
        m1, m2, s1, s2, rho = self.a
        if r % 2 == 0:
            # E[(X^2 + Y^2)^k] from the bivariate normal moments
            xs, ys = sp.symbols("xs ys")
            k = r // 2
            mvn = W.MULTI  # noqa: F841  (documented dependency)
            poly = sp.expand((xs ** 2 + ys ** 2) ** k)
            out = 0
            for (i, j), c in sp.Poly(poly, xs, ys).terms():
                out += c * _bvn_moment(i, j, m1, m2, s1, s2, rho)
            return sp.simplify(out)
        return _numeric_moment_num(self, r)

    def mean(self):
        return self.raw(1)

    def var(self):
        return sp.simplify(self.raw(2) - self.raw(1) ** 2)

    def sample(self, rng):
        m1, m2, s1, s2, r = (_f(t) for t in self.a)
        z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
        return math.hypot(m1 + s1 * z1, m2 + s2 * (r * z1 + math.sqrt(1 - r * r) * z2))


def _bvn_moment(i, j, m1, m2, s1, s2, rho):
    """E[X^i Y^j] for a binormal vector, from the moment generating function"""
    t1, t2 = sp.symbols("t1 t2")
    M = sp.exp(m1 * t1 + m2 * t2 + (s1 ** 2 * t1 ** 2 + 2 * rho * s1 * s2 * t1 * t2 + s2 ** 2 * t2 ** 2) / 2)
    return sp.diff(M, t1, i, t2, j).subs({t1: 0, t2: 0})


def _numeric_moment_num(dd, r):
    import mpmath
    if not dd.numeric():
        raise Unevaluated()
    if not any(_inexact(v) for v in dd.flat()):
        raise NotImplementedError(f"Moment: {dd.name} odd moments have no closed form; use machine-number parameters")
    return sp.Float(mpmath.quad(lambda t: t ** r * dd.npdf(t), [0, mpmath.inf]))


# ─────────────────────────── univariate discrete ───────────────────────────

class UDisc(Discrete):
    def flat(self):
        out = []
        for v in self.a:
            out.extend(v if isinstance(v, list) else [v])
        return out

    def numeric(self):
        return all(_num(v) for v in self.flat())


class Benford(UDisc):
    lo = sp.Integer(1)

    def __init__(self, name, a):
        _nargs(a, (1,), name, "BenfordDistribution[b]")
        if _num(a[0]) and not (a[0].is_Integer and a[0] >= 2):
            raise ValueError(f"{name}: the base b must be an integer >= 2")
        super().__init__(name, a)

    def bounds(self):
        return sp.Integer(1), self.a[0] - 1

    def pdf(self, k):
        b = self.a[0]
        return sp.log(1 + 1 / k) / sp.log(b)

    def cdf_closed(self, k):
        return sp.log(k + 1) / sp.log(self.a[0])

    def mean(self):
        b = self.a[0]
        if not _num(b):
            raise Unevaluated()
        return b - 1 - sp.log(sp.factorial(b - 1)) / sp.log(b)

    def var(self):
        b = self.a[0]
        if not _num(b):
            raise Unevaluated()
        m2 = sp.Add(*(k ** 2 * self.pdf(sp.Integer(k)) for k in range(1, int(b))))
        return sp.simplify(m2 - self.mean() ** 2)


class BetaBinomial(UDisc):
    def __init__(self, name, a):
        _nargs(a, (3,), name, "BetaBinomialDistribution[alpha, beta, n]")
        _pos(a[0], "alpha", name)
        _pos(a[1], "beta", name)
        _nonnegint(a[2], "n", name)
        super().__init__(name, a)

    def bounds(self):
        return sp.Integer(0), self.a[2]

    def pdf(self, k):
        al, be, n = self.a
        return sp.binomial(n, k) * sp.beta(k + al, n - k + be) / sp.beta(al, be)

    def mean(self):
        al, be, n = self.a
        return n * al / (al + be)

    def var(self):
        al, be, n = self.a
        return n * al * be * (al + be + n) / ((al + be) ** 2 * (al + be + 1))

    def raw(self, r):
        al, be, n = self.a
        if _num(n):
            return sp.Add(*(sp.Integer(k) ** r * self.pdf(sp.Integer(k)) for k in range(int(n) + 1)))
        # factorial moments E[(X)_j] = (n)_j (alpha)_j/(alpha + beta)_j, combined with Stirling numbers
        from sympy.functions.combinatorial.numbers import stirling
        return sp.Add(*(stirling(r, j) * sp.ff(n, j) * sp.rf(al, j) / sp.rf(al + be, j) for j in range(r + 1)))

    def sample(self, rng):
        al, be, n = (_f(v) for v in self.a)
        p = rng.betavariate(al, be)
        return sum(1 for _ in range(int(n)) if rng.random() < p)


class BetaNegativeBinomial(UDisc):
    def __init__(self, name, a):
        _nargs(a, (3,), name, "BetaNegativeBinomialDistribution[alpha, beta, n]")
        _pos(a[0], "alpha", name)
        _pos(a[1], "beta", name)
        _pos(a[2], "n", name)
        super().__init__(name, a)

    def pdf(self, k):
        al, be, n = self.a
        return sp.gamma(n + k) / (sp.factorial(k) * sp.gamma(n)) * sp.beta(al + n, be + k) / sp.beta(al, be)

    def raw(self, r):
        al, be, n = self.a
        if _num(al) and not al > r:
            return sp.oo
        from sympy.functions.combinatorial.numbers import stirling
        # factorial moments E[(X)_j] = (n)_j (beta)_j/((alpha - 1)(alpha - 2)...(alpha - j))
        return sp.Add(*(stirling(r, j) * sp.rf(n, j) * sp.rf(be, j) / sp.ff(al - 1, j) for j in range(r + 1)))

    def mean(self):
        al, be, n = self.a
        if _num(al) and not al > 1:
            return sp.oo
        return n * be / (al - 1)

    def var(self):
        al, be, n = self.a
        if _num(al) and not al > 2:
            return sp.oo if al > 1 else sp.nan
        return n * be * (n + al - 1) * (be + al - 1) / ((al - 2) * (al - 1) ** 2)

    def sample(self, rng):
        al, be, n = (_f(v) for v in self.a)
        p = rng.betavariate(al, be)
        return W.NegativeBinomial("NegativeBinomialDistribution", [sp.Float(n), sp.Float(p)]).sample(rng)


class BorelTanner(UDisc):
    def __init__(self, name, a):
        _nargs(a, (2,), name, "BorelTannerDistribution[alpha, n]")
        _open01(a[0], "alpha", name)
        _posint(a[1], "n", name)
        super().__init__(name, a)

    def bounds(self):
        return self.a[1], sp.oo

    def pdf(self, k):
        al, n = self.a
        return n / k * sp.exp(-al * k) * (al * k) ** (k - n) / sp.factorial(k - n)

    def mean(self):
        al, n = self.a
        return n / (1 - al)

    def var(self):
        al, n = self.a
        return n * al / (1 - al) ** 3

    def sample(self, rng):
        # total progeny of a Poisson(alpha) branching process started by n individuals
        al, n = _f(self.a[0]), int(self.a[1])
        pois = W.Poisson("PoissonDistribution", [sp.Float(al)])
        total, queue = n, n
        while queue:
            queue -= 1
            c = pois.sample(rng)
            total += c
            queue += c
            if total > 10 ** 7:
                raise ValueError("RandomVariate: sampling did not terminate")
        return total


class FisherHypergeometric(UDisc):
    def __init__(self, name, a):
        _nargs(a, (4,), name, f"{name}[n, nsucc, ntot, w]")
        for v, w in zip(a[:3], ("n", "nsucc", "ntot")):
            _nonnegint(v, w, name)
        _pos(a[3], "w", name)
        if all(_num(v) for v in a[:3]) and not (a[0] <= a[2] and a[1] <= a[2]):
            raise ValueError(f"{name}: need n <= ntot and nsucc <= ntot")
        super().__init__(name, a)

    def bounds(self):
        n, K, N, w = self.a
        return sp.Max(0, n + K - N), sp.Min(n, K)

    def _norm(self):
        n, K, N, w = self.a
        lo, hi = self.bounds()
        if not self.numeric():
            raise Unevaluated()
        return sp.Add(*(sp.binomial(K, j) * sp.binomial(N - K, n - j) * w ** j for j in range(int(lo), int(hi) + 1)))

    def pdf(self, k):
        n, K, N, w = self.a
        return sp.binomial(K, k) * sp.binomial(N - K, n - k) * w ** k / self._norm()

    def mean(self):
        lo, hi = self.bounds()
        return sp.simplify(sp.Add(*(k * self.pdf(sp.Integer(k)) for k in range(int(lo), int(hi) + 1))))

    def var(self):
        lo, hi = self.bounds()
        m2 = sp.Add(*(k ** 2 * self.pdf(sp.Integer(k)) for k in range(int(lo), int(hi) + 1)))
        return sp.simplify(m2 - self.mean() ** 2)


class WalleniusHypergeometric(FisherHypergeometric):
    def pdf(self, k):
        n, K, N, w = self.a
        if not (self.numeric() and _num(k)):
            raise Unevaluated()
        m2 = N - K
        dd = w * (K - k) + (m2 - (n - k))
        if dd == 0:
            return sp.Integer(1) if k == n else sp.Integer(0)
        if isinstance(w, sp.Rational) and not w.has(sp.Float):
            # exact: Int_0^1 (1 - t^(w/D))^k (1 - t^(1/D))^(n-k) dt with t = u^(q D), w = p/q
            p_, q_ = int(sp.fraction(w)[0]), int(sp.fraction(w)[1])
            u = sp.Symbol("u")
            e = int(q_ * dd)
            if q_ * dd != e:
                raise NotImplementedError("WalleniusHypergeometricDistribution: unsupported weight")
            # t^(w/D) = u^(q D w/D) = u^p, t^(1/D) = u^q, dt = q D u^(q D - 1) du
            integrand = sp.expand(e * u ** (e - 1) * (1 - u ** p_) ** k * (1 - u ** q_) ** (n - k))
            val = sp.integrate(integrand, (u, 0, 1))
        else:
            import mpmath
            wf, df, kk, nk = _mp(w), _mp(dd), int(k), int(n - k)
            val = sp.Float(mpmath.quad(lambda t: (1 - t ** (wf / df)) ** kk * (1 - t ** (1 / df)) ** nk, [0, 1]))
        return sp.binomial(K, k) * sp.binomial(m2, n - k) * val


class LogSeries(UDisc):
    lo = sp.Integer(1)

    def __init__(self, name, a):
        _nargs(a, (1,), name, "LogSeriesDistribution[theta]")
        _open01(a[0], "theta", name)
        super().__init__(name, a)

    def pdf(self, k):
        th = self.a[0]
        return -th ** k / (k * sp.log(1 - th))

    def mean(self):
        th = self.a[0]
        return -th / ((1 - th) * sp.log(1 - th))

    def var(self):
        th = self.a[0]
        lg = sp.log(1 - th)
        return -th * (th + lg) / ((1 - th) ** 2 * lg ** 2)

    def mgf(self, t):
        th = self.a[0]
        return sp.log(1 - th * sp.exp(t)) / sp.log(1 - th)

    def sample(self, rng):
        th = _f(self.a[0])
        # Kemp's algorithm LK
        u = rng.random()
        if u > th:
            return 1
        q = 1 - math.exp(math.log(1 - th) * rng.random())
        if u < q * q:
            return int(1 + math.log(u) / math.log(q))
        return 1 if u > q else 2


class Pascal(UDisc):
    def __init__(self, name, a):
        _nargs(a, (2,), name, "PascalDistribution[n, p]")
        _posint(a[0], "n", name)
        if _num(a[1]) and not (0 < a[1] <= 1):
            raise ValueError(f"{name}: p must be in (0, 1]")
        super().__init__(name, a)

    def bounds(self):
        return self.a[0], sp.oo

    def pdf(self, k):
        n, p = self.a
        return sp.binomial(k - 1, n - 1) * p ** n * (1 - p) ** (k - n)

    def mean(self):
        n, p = self.a
        return n / p

    def var(self):
        n, p = self.a
        return n * (1 - p) / p ** 2

    def mgf(self, t):
        n, p = self.a
        return (p * sp.exp(t) / (1 - (1 - p) * sp.exp(t))) ** n

    def sample(self, rng):
        n, p = self.a
        return int(n) + W.NegativeBinomial("NegativeBinomialDistribution", [n, p]).sample(rng)


class PoissonConsul(UDisc):
    def __init__(self, name, a):
        _nargs(a, (2,), name, "PoissonConsulDistribution[mu, lambda]")
        _pos(a[0], "mu", name)
        if _num(a[1]) and not (0 <= a[1] < 1):
            raise ValueError(f"{name}: lambda must be in [0, 1)")
        super().__init__(name, a)

    def pdf(self, k):
        m, l = self.a
        return m * (m + k * l) ** (k - 1) * sp.exp(-m - k * l) / sp.factorial(k)

    def mean(self):
        m, l = self.a
        return m / (1 - l)

    def var(self):
        m, l = self.a
        return m / (1 - l) ** 3

    def sample(self, rng):
        # branching representation: Poisson(mu) founders, each with a Borel(lambda) family
        m, l = _f(self.a[0]), _f(self.a[1])
        n = W.Poisson("PoissonDistribution", [sp.Float(m)]).sample(rng)
        if n == 0 or l == 0:
            return n
        return BorelTanner("BorelTannerDistribution", [sp.Float(l), sp.Integer(n)]).sample(rng)


class PolyaAeppli(UDisc):
    def __init__(self, name, a):
        _nargs(a, (2,), name, "PolyaAeppliDistribution[theta, p]")
        _pos(a[0], "theta", name)
        if _num(a[1]) and not (0 <= a[1] < 1):
            raise ValueError(f"{name}: p must be in [0, 1)")
        super().__init__(name, a)

    def pdf(self, k):
        th, p = self.a
        if _num(k) and k == 0:
            return sp.exp(-th)
        if _num(k) and p == 0:
            return sp.exp(-th) * th ** k / sp.factorial(k)
        body = (sp.exp(-th) * th * (1 - p) * p ** (k - 1) * sp.hyper([1 - k], [2], -th * (1 - p) / p))
        if _num(k):
            return sp.simplify(sp.hyperexpand(body))
        return sp.Piecewise((sp.exp(-th), sp.Eq(k, 0)), (body, True))

    def mean(self):
        th, p = self.a
        return th / (1 - p)

    def var(self):
        th, p = self.a
        return th * (1 + p) / (1 - p) ** 2

    def mgf(self, t):
        th, p = self.a
        g = (1 - p) * sp.exp(t) / (1 - p * sp.exp(t))
        return sp.exp(th * (g - 1))

    def sample(self, rng):
        th, p = _f(self.a[0]), _f(self.a[1])
        n = W.Poisson("PoissonDistribution", [sp.Float(th)]).sample(rng)
        geo = W.Geometric("GeometricDistribution", [sp.Float(1 - p)])
        return sum(1 + geo.sample(rng) for _ in range(n))


class Skellam(UDisc):
    lo = -sp.oo

    def __init__(self, name, a):
        _nargs(a, (2,), name, "SkellamDistribution[mu1, mu2]")
        _pos(a[0], "mu1", name)
        _pos(a[1], "mu2", name)
        super().__init__(name, a)

    def pdf(self, k):
        m1, m2 = self.a
        return sp.exp(-m1 - m2) * (m1 / m2) ** (k / 2) * sp.besseli(k, 2 * sp.sqrt(m1 * m2))

    def pdf_value(self, k):
        if _num(k):
            return self.pdf(k) if k == sp.floor(k) else sp.Integer(0)
        return sp.Piecewise((self.pdf(k), sp.Eq(k, sp.floor(k))), (0, True))

    def cdf_value(self, x):
        import mpmath
        if not (_num(x) and self.numeric()):
            raise Unevaluated()
        if not (_inexact(x) or any(_inexact(v) for v in self.a)):
            raise NotImplementedError("CDF: SkellamDistribution has no closed-form CDF; use a machine-number argument")
        k = int(math.floor(_f(x)))
        m1, m2 = (_mp(v) for v in self.a)
        f = lambda j: mpmath.exp(-m1 - m2) * (m1 / m2) ** (mpmath.mpf(j) / 2) * mpmath.besseli(j, 2 * mpmath.sqrt(m1 * m2))  # noqa: E731
        return sp.Float(mpmath.nsum(f, [-mpmath.inf, k]))

    def quantile_value(self, p, inputs, fname):
        if not (_num(p) and self.numeric()):
            raise Unevaluated()
        if p in (0, 1):
            return -sp.oo if p == 0 else sp.oo
        import mpmath
        m1, m2 = (_mp(v) for v in self.a)
        k = int(math.floor(_f(self.mean())))
        F = lambda kk: float(self.cdf_value(sp.Float(kk)))  # noqa: E731
        while F(k) >= _f(p):
            k -= 1
        while F(k) < _f(p):
            k += 1
        return sp.Integer(k)

    def mean(self):
        return self.a[0] - self.a[1]

    def var(self):
        return self.a[0] + self.a[1]

    def mgf(self, t):
        m1, m2 = self.a
        return sp.exp(-m1 - m2 + m1 * sp.exp(t) + m2 * sp.exp(-t))

    def sample(self, rng):
        m1, m2 = self.a
        P = W.Poisson
        return P("PoissonDistribution", [m1]).sample(rng) - P("PoissonDistribution", [m2]).sample(rng)


class WaringYule(UDisc):
    def __init__(self, name, a):
        _nargs(a, (1, 2), name, "WaringYuleDistribution[alpha] or [alpha, beta]")
        _pos(a[0], "alpha", name)
        if len(a) == 2:
            _pos(a[1], "beta", name)
        super().__init__(name, a)

    def ab(self):
        return (self.a[0], self.a[1]) if len(self.a) == 2 else (self.a[0], sp.Integer(1))

    def pdf(self, k):
        al, be = self.ab()
        return al * sp.rf(be, k) / sp.rf(al + be, k + 1)

    def mean(self):
        al, be = self.ab()
        if _num(al) and not al > 1:
            return sp.oo
        return be / (al - 1)

    def var(self):
        al, be = self.ab()
        if _num(al) and not al > 2:
            return sp.oo if al > 1 else sp.nan
        return al * be * (al + be - 1) / ((al - 1) ** 2 * (al - 2))

    def raw(self, r):
        al, be = self.ab()
        if _num(al) and not al > r:
            return sp.oo
        from sympy.functions.combinatorial.numbers import stirling
        # factorial moments E[(X)_j] = j! (beta)_j/((alpha - 1)...(alpha - j))
        return sp.Add(*(stirling(r, j) * sp.factorial(j) * sp.rf(be, j) / sp.ff(al - 1, j) for j in range(r + 1)))

    def sample(self, rng):
        al, be = (_f(v) for v in self.ab())
        # mixture: X | p ~ Geometric(p) (failures), p ~ Beta(alpha, beta)
        p = rng.betavariate(al, be)
        return W.Geometric("GeometricDistribution", [sp.Float(p)]).sample(rng)


class Zipf(UDisc):
    lo = sp.Integer(1)

    def __init__(self, name, a):
        _nargs(a, (1, 2), name, "ZipfDistribution[rho] or [n, rho]")
        if len(a) == 2:
            _posint(a[0], "n", name)
        _pos(a[-1], "rho", name)
        super().__init__(name, a)

    def bounds(self):
        return (sp.Integer(1), self.a[0]) if len(self.a) == 2 else (sp.Integer(1), sp.oo)

    def _z(self):
        r = self.a[-1]
        if len(self.a) == 2:
            return sp.harmonic(self.a[0], r + 1)
        return sp.zeta(r + 1)

    def pdf(self, k):
        r = self.a[-1]
        return k ** (-r - 1) / self._z()

    def cdf_closed(self, k):
        r = self.a[-1]
        return sp.harmonic(k, r + 1) / self._z()

    def raw(self, j):
        r = self.a[-1]
        if len(self.a) == 2:
            return sp.harmonic(self.a[0], r + 1 - j) / self._z()
        if _num(r) and not r > j:
            return sp.oo
        return sp.zeta(r + 1 - j) / self._z()

    def mean(self):
        return self.raw(1)

    def var(self):
        m1, m2 = self.raw(1), self.raw(2)
        if m2 == sp.oo:
            return sp.oo if m1 != sp.oo else sp.nan
        return m2 - m1 ** 2


class Categorical(UDisc):
    """CategoricalDistribution[{c1, ...}] (equal weights) or [{c1, ...}, {p1, ...}]: PDF and RandomVariate over
    arbitrary category values"""

    def __init__(self, name, a, cats=None):
        cs, ws = cats
        super().__init__(name, [])
        self.cats, self.w = cs, ws

    def pdf_value(self, x):
        for c, w in zip(self.cats, self.w):
            if c == x:
                return w
        if isinstance(x, (Symbol, Expr)) and not any(isinstance(c, type(x)) and c == x for c in self.cats):
            if isinstance(x, Symbol) and not any(isinstance(c, Symbol) for c in self.cats):
                raise Unevaluated()
        return sp.Integer(0)

    def cdf_value(self, x):
        raise NotImplementedError("CDF: CategoricalDistribution has no order on its categories")

    def mean(self):
        raise NotImplementedError("Mean: CategoricalDistribution categories are not numbers")

    var = mean

    def numeric(self):
        return all(_num(w) for w in self.w)

    def sample(self, rng):
        u, acc = rng.random(), 0.0
        for c, w in zip(self.cats, self.w):
            acc += _f(w)
            if u < acc:
                return c
        return self.cats[-1]


# ─────────────────────────── registration ───────────────────────────

def _ctor(name, notes, defaults=None):
    """register the inert constructor of a distribution head: parameters are validated (raising on invalid
    numeric values), zero-argument forms expand to Wolfram's defaults, and the expression stays unevaluated"""
    def impl(*params: Numeric | ListOf[Numeric]) -> Expr:  # noqa: F821
        if defaults is not None and not params:
            return defaults
        W.dist_of(Expr(name, *params), name)
        raise Unevaluated()
    impl.__name__ = "_" + name
    W.NEW_DISTS.add(name)
    fn(name, area=AREA, status="added", notes=notes)(impl)


def _uni(name, cls, notes, defaults=None):
    W.TABLE[name] = cls
    _ctor(name, notes, defaults)


_STD = ("PDF, CDF, InverseCDF, Mean, Variance, moments, CharacteristicFunction/MomentGeneratingFunction where they "
        "exist, Expectation, Probability and RandomVariate through the batch-5 distribution functions")

_uni("ArcSinDistribution", ArcSin,
     "ArcSinDistribution[{a, b}] (default {0, 1}): PDF 1/(Pi Sqrt[(x - a)(b - x)]), CDF (2/Pi) ArcSin[Sqrt[(x - a)/(b - a)]], "
     "closed-form quantile and moments; " + _STD + ". Check: exact; reference values (SciPy arcsine) in tests.",
     Expr("ArcSinDistribution", L(0, 1)))
_uni("BatesDistribution", Bates,
     "BatesDistribution[n, {a, b}] (default {0, 1}): mean of n uniforms; exact piecewise-polynomial PDF/CDF "
     "(Irwin-Hall), mean, variance, MGF; integer n required. Check: exact; Monte-Carlo-free identities "
     "(CDF at the centre = 1/2, PDF integrates to 1) and SymPy reference values in tests.")
_uni("UniformSumDistribution", UniformSum,
     "UniformSumDistribution[n, {a, b}] (default {0, 1}): sum of n uniforms (Irwin-Hall); exact piecewise PDF/CDF, "
     "mean, variance, MGF. Check: exact; Irwin-Hall reference values and the identity PDF = CDF' in tests.")
_uni("BetaPrimeDistribution", BetaPrime,
     "BetaPrimeDistribution[p, q] , [p, q, beta] (scale) and [p, q, alpha, beta]: PDF alpha (x/beta)^(alpha p - 1) "
     "(1 + (x/beta)^alpha)^(-p - q)/(beta Beta[p, q]); CDF via BetaRegularized; moments (Infinity when they "
     "diverge). Check: exact; SciPy betaprime reference values in tests.")
_uni("BeniniDistribution", Benini,
     "BeniniDistribution[alpha, beta, sigma] on x >= sigma: CDF 1 - Exp[-alpha Log[x/sigma] - beta Log[x/sigma]^2], "
     "closed-form quantile; moments by exact integration (SymPy). Check: exact; quantile/CDF round trip in tests.")
_uni("DagumDistribution", Dagum,
     "DagumDistribution[p, a, b] and [p, a, b, mu]: CDF (1 + ((x - mu)/b)^-a)^-p, closed quantile, moments "
     "b^k Gamma[p + k/a] Gamma[1 - k/a]/Gamma[p] (Infinity for k >= a). Check: exact; SciPy mielke/burg reference in "
     "tests.")
_uni("SinghMaddalaDistribution", SinghMaddala,
     "SinghMaddalaDistribution[q, a, b] and [q, a, b, mu] (Burr XII): CDF 1 - (1 + ((x - mu)/b)^a)^-q, closed "
     "quantile and moments. Check: exact; SciPy burr12 reference values in tests.")
_uni("DavisDistribution", Davis,
     "DavisDistribution[b, n, mu] (n > 1): PDF b^n (x - mu)^(-1 - n)/((E^(b/(x - mu)) - 1) Gamma[n] Zeta[n]); exact "
     "moments b^k Gamma[n - k] Zeta[n - k]/(Gamma[n] Zeta[n]); the CDF has no closed form: numeric (mpmath) for "
     "machine-number x, exact x raises. Check: numeric cross-check (PDF integrates to 1) and reference moments.")
_uni("ExpGammaDistribution", ExpGamma,
     "ExpGammaDistribution[kappa, theta, mu]: Log of a gamma variate, PDF Exp[kappa z - E^z]/(theta Gamma[kappa]) with "
     "z = (x - mu)/theta; CDF GammaRegularized form; MGF, mean mu + theta PolyGamma[kappa]. Check: exact; SciPy "
     "loggamma reference values in tests.")
_uni("ExponentialPowerDistribution", ExponentialPower,
     "ExponentialPowerDistribution[kappa], [kappa, sigma] or [kappa, mu, sigma]: PDF Exp[-|x - mu|^kappa/(kappa "
     "sigma^kappa)]/(2 kappa^(1/kappa) sigma Gamma[1 + 1/kappa]) (kappa = 2 is NormalDistribution[mu, sigma]); "
     "CDF via incomplete gamma. Check: exact; the kappa = 2 normal identity and SciPy gennorm in tests.")
_uni("FisherZDistribution", FisherZ,
     "FisherZDistribution[n, m]: Log[F]/2 for F ~ FRatioDistribution[n, m]; PDF, CDF (BetaRegularized), MGF, "
     "mean and variance (PolyGamma). Check: exact; CDF[FisherZ, z] = CDF[FRatio, E^(2 z)] in tests.")
_uni("FrechetDistribution", Frechet,
     "FrechetDistribution[alpha, beta] and [alpha, beta, mu]: CDF Exp[-((x - mu)/beta)^-alpha], closed quantile, "
     "moments beta^k Gamma[1 - k/alpha] (Infinity for k >= alpha). Check: exact; SciPy invweibull reference values.")
_uni("HalfNormalDistribution", HalfNormal,
     "HalfNormalDistribution[theta]: PDF 2 theta/Pi Exp[-x^2 theta^2/Pi] on x >= 0 (mean 1/theta); CDF "
     "Erf[theta x/Sqrt[Pi]], closed quantile and moments. Check: exact; SciPy halfnorm reference values.")
_uni("HoytDistribution", Hoyt,
     "HoytDistribution[q, omega] (Nakagami-q, 0 < q <= 1, E[X^2] = omega): Bessel-I PDF; mean via EllipticE, even "
     "moments exact; the CDF has no closed form: numeric for machine-number input. Check: numeric cross-check "
     "(quadrature of the PDF) in tests.")
_uni("HyperbolicDistribution", Hyperbolic,
     "HyperbolicDistribution[alpha, beta, delta, mu] and the generalised [lambda, alpha, beta, delta, mu]: Bessel-K "
     "PDF, MGF, mean and variance; numeric CDF for machine-number input. Check: exact moments; SciPy genhyperbolic "
     "reference values in tests.")
_uni("InverseChiSquareDistribution", InverseChiSquare,
     "InverseChiSquareDistribution[nu] and the scaled [nu, xi] (= InverseGammaDistribution[nu/2, nu xi/2]): PDF, "
     "CDF (GammaRegularized), moments (Infinity when divergent). Check: exact; SciPy invgamma reference values.")
_uni("InverseGammaDistribution", InverseGamma,
     "InverseGammaDistribution[alpha, beta]: PDF beta^alpha x^(-alpha - 1) E^(-beta/x)/Gamma[alpha], CDF "
     "GammaRegularized[alpha, beta/x], moments; the four-parameter generalised form is not supported (raises). "
     "Check: exact; SciPy invgamma reference values.")
_uni("InverseGaussianDistribution", InverseGaussian,
     "InverseGaussianDistribution[mu, lambda]: PDF, closed-form CDF, MGF, mean mu, variance mu^3/lambda; the "
     "three-parameter generalised form is not supported. Check: exact; SciPy invgauss reference values.")
_uni("KDistribution", KDist,
     "KDistribution[nu, w] (E[X^2] = w): Bessel-K PDF and CDF, moments (w/nu)^(r/2) Gamma[nu + r/2] Gamma[1 + r/2]/"
     "Gamma[nu]. Check: exact; numeric cross-check of the PDF normalisation and moments in tests.")
_uni("KumaraswamyDistribution", Kumaraswamy,
     "KumaraswamyDistribution[alpha, beta]: PDF alpha beta x^(alpha - 1) (1 - x^alpha)^(beta - 1), CDF, quantile, "
     "moments. Check: exact; quantile/CDF round trip and SciPy kumaraswamy-free reference values in tests.")
_uni("LevyDistribution", Levy,
     "LevyDistribution[mu, sigma]: PDF, CDF Erfc[Sqrt[sigma/(2 (x - mu))]], quantile, characteristic function; "
     "mean and variance are Infinity. Check: exact; SciPy levy reference values.")
_uni("LindleyDistribution", Lindley,
     "LindleyDistribution[delta]: PDF delta^2/(1 + delta) (1 + x) E^(-delta x), CDF, MGF, moments. Check: exact; "
     "reference moments from the mixture representation in tests.")
_uni("LogGammaDistribution", LogGamma,
     "LogGammaDistribution[alpha, beta, mu] (and [alpha, beta] with mu = 0): mu - 1 + E^Y for Y ~ GammaDistribution"
     "[alpha, beta]; PDF Log[1 + x - mu]^(alpha - 1) (1 + x - mu)^(-1 - 1/beta)/(beta^alpha Gamma[alpha]), CDF, "
     "moments. Check: exact; the gamma transformation identity in tests.")
_uni("LogLogisticDistribution", LogLogistic,
     "LogLogisticDistribution[gamma, sigma]: CDF 1/(1 + (x/sigma)^-gamma), quantile, moments sigma^k (Pi k/gamma)/"
     "Sin[Pi k/gamma]. Check: exact; SciPy fisk reference values.")
_uni("MarchenkoPasturDistribution", MarchenkoPastur,
     "MarchenkoPasturDistribution[lambda, sigma] (and [lambda]) for 0 < lambda <= 1: semicircle-type PDF on "
     "[sigma^2 (1 - Sqrt[lambda])^2, sigma^2 (1 + Sqrt[lambda])^2], exact moments (Narayana polynomials), numeric "
     "CDF; lambda > 1 (an atom at 0) raises NotImplementedError. Check: exact moments; PDF normalisation in tests.")
_uni("MaxStableDistribution", MaxStable,
     "MaxStableDistribution[mu, sigma, xi] (generalised extreme value, maxima): CDF Exp[-(1 + xi (x - mu)/sigma)^"
     "(-1/xi)] (Gumbel for xi = 0), quantile, moments via Gamma[1 - k xi]. Check: exact; SciPy genextreme (c = -xi).")
_uni("MinStableDistribution", MinStable,
     "MinStableDistribution[mu, sigma, xi]: mirror image of MaxStableDistribution (minima); CDF 1 - Exp[-(1 - xi "
     "(x - mu)/sigma)^(-1/xi)]. Check: exact; symmetry with MaxStableDistribution in tests.")
_uni("MaxwellDistribution", Maxwell,
     "MaxwellDistribution[sigma]: PDF Sqrt[2/Pi] x^2 E^(-x^2/(2 sigma^2))/sigma^3, CDF, moments. Check: exact; SciPy "
     "maxwell reference values.")
_uni("MeixnerDistribution", Meixner,
     "MeixnerDistribution[a, b, m, d]: PDF (2 Cos[b/2])^(2 d)/(2 a Pi Gamma[2 d]) E^(b (x - m)/a) |Gamma[d + I (x - m)/"
     "a]|^2, characteristic function, MGF, mean, variance; numeric CDF. Check: exact moments; numeric cross-check of "
     "the PDF normalisation.")
_uni("MoyalDistribution", Moyal,
     "MoyalDistribution[] = [0, 1] and [mu, sigma]: PDF Exp[-(E^-z + z)/2]/(sigma Sqrt[2 Pi]), CDF Erfc[E^(-z/2)/"
     "Sqrt[2]], quantile, MGF, moments. Check: exact; SciPy moyal reference values.",
     Expr("MoyalDistribution", 0, 1))
_uni("NakagamiDistribution", Nakagami,
     "NakagamiDistribution[mu, omega] (mu >= 1/2): PDF, CDF (GammaRegularized), moments. Check: exact; SciPy nakagami.")
_uni("NoncentralChiSquareDistribution", NoncentralChiSquare,
     "NoncentralChiSquareDistribution[nu, lambda]: Bessel-I PDF, MGF, mean nu + lambda, variance 2 (nu + 2 lambda); "
     "CDF by the Poisson-weighted series for machine-number input (exact input raises). Check: exact moments; "
     "SciPy ncx2 reference values.")
_uni("NoncentralBetaDistribution", NoncentralBeta,
     "NoncentralBetaDistribution[alpha, beta, delta]: PDF with Hypergeometric1F1, moments with HypergeometricPFQ; "
     "series CDF for machine-number input. Check: reference values (SciPy ncf-derived/mpmath series) in tests.")
_uni("NoncentralFRatioDistribution", NoncentralFRatio,
     "NoncentralFRatioDistribution[n, m, lambda]: PDF with Hypergeometric1F1, mean and variance; series CDF for "
     "machine-number input. Check: SciPy ncf reference values.")
_uni("NoncentralStudentTDistribution", NoncentralStudentT,
     "NoncentralStudentTDistribution[nu, delta]: PDF with Hypergeometric1F1, moments; CDF by quadrature for "
     "machine-number input. Check: SciPy nct reference values.")
_uni("PERTDistribution", PERT,
     "PERTDistribution[{min, max}, c] (lambda = 4) and [{min, max}, c, lambda]: scaled BetaDistribution[1 + lambda "
     "(c - min)/(max - min), 1 + lambda (max - c)/(max - min)]; exact PDF, CDF, moments. Check: exact; mean (min + "
     "lambda c + max)/(lambda + 2) in tests.")
_uni("ParetoPickandsDistribution", ParetoPickands,
     "ParetoPickandsDistribution[mu, sigma, xi] (generalised Pareto): CDF 1 - (1 + xi (x - mu)/sigma)^(-1/xi), "
     "quantile, moments. Check: exact; SciPy genpareto reference values.")
_uni("PowerDistribution", PowerDist,
     "PowerDistribution[k, a] on 0 < x < 1/k: PDF a k^a x^(a - 1), CDF (k x)^a, quantile, moments. Check: exact.")
_uni("RiceDistribution", Rice,
     "RiceDistribution[alpha, beta]: PDF x/beta^2 E^(-(x^2 + alpha^2)/(2 beta^2)) BesselI[0, x alpha/beta^2], CDF "
     "1 - MarcumQ[1, alpha/beta, x/beta], moments via Hypergeometric1F1; the [m, alpha, beta] form raises "
     "NotImplementedError. Check: SciPy rice reference values.")
_uni("SechDistribution", Sech,
     "SechDistribution[] = [0, 1] and [mu, sigma]: PDF Sech[Pi (x - mu)/(2 sigma)]/(2 sigma), CDF, quantile, MGF; "
     "variance sigma^2. Check: exact; SciPy hypsecant reference values.", Expr("SechDistribution", 0, 1))
_uni("ShiftedGompertzDistribution", ShiftedGompertz,
     "ShiftedGompertzDistribution[lambda, xi]: CDF (1 - E^(-lambda x)) Exp[-xi E^(-lambda x)], exact mean with "
     "ExpIntegralE; higher moments numeric for machine-number parameters. Check: exact; numeric cross-check of the "
     "mean.")
_uni("SkewNormalDistribution", SkewNormal,
     "SkewNormalDistribution[mu, sigma, alpha]: PDF (2/sigma) phi(z) Phi(alpha z), mean, variance, skewness, "
     "kurtosis, MGF; the CDF (OwenT) is numeric for machine-number input (exact input raises unless alpha = 0). "
     "Check: SciPy skewnorm reference values.")
_uni("TsallisQExponentialDistribution", TsallisQExponential,
     "TsallisQExponentialDistribution[lambda, q] (q < 2): PDF (2 - q) lambda e_q(-lambda x), CDF, quantile, moments "
     "(Infinity when divergent); q = 1 is ExponentialDistribution[lambda]. Check: exact; quantile/CDF round trip.")
_uni("TsallisQGaussianDistribution", TsallisQGaussian,
     "TsallisQGaussianDistribution[mu, beta, q] (q < 3) and [q] = [0, 1, q]: PDF e_q(-(x - mu)^2/(2 beta^2))/"
     "(beta Sqrt[2] C_q); q = 1 is NormalDistribution[mu, beta], q = 2 CauchyDistribution[mu, beta Sqrt[2]]; CDF "
     "through the equivalent scaled Beta (q < 1) or StudentT (1 < q < 3) distribution. Check: exact; the normal and "
     "Cauchy special cases in tests.")
_uni("TukeyLambdaDistribution", TukeyLambda,
     "TukeyLambdaDistribution[lambda] and [lambda, mu, sigma]: quantile mu + sigma (p^lambda - (1 - p)^lambda)/"
     "lambda (logistic for lambda = 0); exact mean, variance and moments (Beta integrals); PDF and CDF numeric for "
     "machine-number x (closed for lambda = 0, 1); the generalised [{l1, l2}, mu, {s1, s2}] form raises. Check: "
     "exact moments; SciPy tukeylambda reference values.")
_uni("WakebyDistribution", Wakeby,
     "WakebyDistribution[alpha, beta, gamma, delta, mu]: quantile mu + alpha/beta (1 - (1 - p)^beta) - gamma/delta "
     "(1 - (1 - p)^-delta); exact mean and variance; PDF/CDF numeric for machine-number x. Check: exact; "
     "quantile/CDF round trip.")
_uni("VarianceGammaDistribution", VarianceGamma,
     "VarianceGammaDistribution[lambda, alpha, beta, mu]: Bessel-K PDF, MGF, mean, variance; numeric CDF. Check: "
     "exact moments; numeric cross-check of the normalisation.")
_uni("VoigtDistribution", Voigt,
     "VoigtDistribution[delta, sigma]: convolution of CauchyDistribution[0, delta] and NormalDistribution[0, sigma]; "
     "PDF via the Faddeeva function, characteristic function; moments Indeterminate; numeric CDF. Check: SciPy "
     "voigt_profile reference values.")
_uni("VonMisesDistribution", VonMises,
     "VonMisesDistribution[mu, kappa] on [mu - Pi, mu + Pi]: PDF E^(kappa Cos[x - mu])/(2 Pi BesselI[0, kappa]); "
     "mean mu; variance and CDF numeric for machine-number input. Check: SciPy vonmises reference values.")
_uni("WignerSemicircleDistribution", WignerSemicircle,
     "WignerSemicircleDistribution[r] and [a, r]: PDF 2/(Pi r^2) Sqrt[r^2 - (x - a)^2], CDF, moments (Catalan "
     "numbers). Check: exact; SciPy semicircular reference values.")
_uni("BenktanderGibratDistribution", BenktanderGibrat,
     "BenktanderGibratDistribution[a, b] (Benktander type I, x >= 1): survival (1 + 2 b Log[x]/a) x^-(a + 1 + b "
     "Log[x]); mean 1 + 1/a, variance with Erfc. Check: exact; numeric cross-check of the moments.")
_uni("GompertzMakehamDistribution", GompertzMakeham,
     "GompertzMakehamDistribution[lambda, xi] (Gompertz): CDF 1 - Exp[xi (1 - E^(lambda x))], quantile, mean "
     "E^xi ExpIntegralE[1, xi]/lambda; the four-parameter Makeham form raises NotImplementedError. Check: exact; "
     "truncated-Gumbel identity in tests.")
_uni("HyperexponentialDistribution", Hyperexponential,
     "HyperexponentialDistribution[{a1, ...}, {l1, ...}]: mixture of exponentials; PDF, CDF, moments, MGF. Check: "
     "exact.")
_uni("HypoexponentialDistribution", Hypoexponential,
     "HypoexponentialDistribution[{l1, ...}]: sum of independent exponentials (phase-type, matrix exponential); "
     "PDF, CDF, moments, MGF. Check: exact; equal rates reduce to ErlangDistribution in tests.")
_uni("CoxianDistribution", Coxian,
     "CoxianDistribution[{a1, ..., a(m-1)}, {l1, ..., lm}]: phase-type distribution with continuation "
     "probabilities a_i; PDF, CDF, moments, MGF via the sub-generator matrix. Check: exact; reduces to "
     "HypoexponentialDistribution when every a_i = 1.")
_uni("StableDistribution", Stable,
     "StableDistribution[type, alpha, beta, mu, sigma] (type 0 or 1; [alpha, beta, mu, sigma] and [alpha, beta] use "
     "type 1): Wolfram's S0/S1 characteristic functions; PDF and CDF by numerical Fourier inversion for machine-"
     "number x (exact x raises); mean and variance where they exist; RandomVariate by Chambers-Mallows-Stuck. "
     "Check: SciPy levy_stable (S1) reference values; alpha = 2 normal case.")
_uni("LandauDistribution", Landau,
     "LandauDistribution[] = [0, 1] and [mu, sigma] = StableDistribution[1, 1, 1, mu, sigma]; numeric PDF/CDF for "
     "machine-number x; mean and variance Indeterminate. Check: reference values from the documented integral "
     "(1/Pi) Integrate[Exp[-t x - (2/Pi) t Log[t]] Sin[2 t], {t, 0, Infinity}].",
     Expr("LandauDistribution", 0, 1))
_uni("SuzukiDistribution", Suzuki,
     "SuzukiDistribution[mu, nu] = ParameterMixtureDistribution[RayleighDistribution[s], s ~ LogNormalDistribution"
     "[mu, nu]]; exact moments, numeric PDF/CDF for machine-number x. Check: exact moments; numeric cross-check.")
_uni("BeckmannDistribution", Beckmann,
     "BeckmannDistribution[mu1, mu2, sigma1, sigma2] and [..., rho]: Norm of a BinormalDistribution vector; exact "
     "even moments; numeric PDF/CDF and odd moments for machine-number input. Check: exact even moments; "
     "Rice/Rayleigh special cases.")
# discrete
_uni("BenfordDistribution", Benford,
     "BenfordDistribution[b]: PDF Log[b, 1 + 1/k] for k = 1, ..., b - 1, CDF Log[b, k + 1], exact mean and variance. "
     "Check: exact.")
_uni("BetaBinomialDistribution", BetaBinomial,
     "BetaBinomialDistribution[alpha, beta, n]: PDF Binomial[n, k] Beta[k + alpha, n - k + beta]/Beta[alpha, beta], "
     "moments. Check: exact; SciPy betabinom reference values.")
_uni("BetaNegativeBinomialDistribution", BetaNegativeBinomial,
     "BetaNegativeBinomialDistribution[alpha, beta, n]: failures before the n-th success with p ~ Beta[alpha, beta]; "
     "PDF, moments (Infinity when divergent). Check: exact; SciPy betanbinom reference values.")
_uni("BorelTannerDistribution", BorelTanner,
     "BorelTannerDistribution[alpha, n] on k >= n: PDF n/k E^(-alpha k) (alpha k)^(k - n)/(k - n)!, mean n/(1 - "
     "alpha), variance n alpha/(1 - alpha)^3. Check: exact; PDF sums to 1 (numeric cross-check) in tests.")
_uni("FisherHypergeometricDistribution", FisherHypergeometric,
     "FisherHypergeometricDistribution[n, nsucc, ntot, w]: Fisher's noncentral hypergeometric PDF, CDF and moments "
     "(exact finite sums). Check: exact; SciPy nchypergeom_fisher reference values.")
_uni("WalleniusHypergeometricDistribution", WalleniusHypergeometric,
     "WalleniusHypergeometricDistribution[n, nsucc, ntot, w]: Wallenius' PDF Binomial[m1, k] Binomial[m2, n - k] "
     "Integrate[(1 - t^(w/D))^k (1 - t^(1/D))^(n - k), {t, 0, 1}]; exact for rational w, numeric for machine w. "
     "Check: exact; SciPy nchypergeom_wallenius reference values.")
_uni("LogSeriesDistribution", LogSeries,
     "LogSeriesDistribution[theta] on k >= 1: PDF -theta^k/(k Log[1 - theta]), mean, variance, MGF. Check: exact; "
     "SciPy logser reference values.")
_uni("PascalDistribution", Pascal,
     "PascalDistribution[n, p]: number of trials for n successes (k >= n); PDF Binomial[k - 1, n - 1] p^n (1 - p)^"
     "(k - n), MGF, moments. Check: exact; shift of NegativeBinomialDistribution in tests.")
_uni("PoissonConsulDistribution", PoissonConsul,
     "PoissonConsulDistribution[mu, lambda] (generalised Poisson, 0 <= lambda < 1): PDF mu (mu + k lambda)^(k - 1) "
     "E^(-mu - k lambda)/k!, mean mu/(1 - lambda), variance mu/(1 - lambda)^3. Check: exact.")
_uni("PolyaAeppliDistribution", PolyaAeppli,
     "PolyaAeppliDistribution[theta, p]: Poisson[theta] sum of geometric (k >= 1, success 1 - p) variables; PDF "
     "E^-theta theta (1 - p) p^(k - 1) Hypergeometric1F1[1 - k, 2, -theta (1 - p)/p] for k >= 1, mean theta/(1 - p), "
     "MGF. Check: exact; PDF sums and the compound-Poisson MGF in tests.")
_uni("SkellamDistribution", Skellam,
     "SkellamDistribution[mu1, mu2]: difference of independent Poisson variables, PDF E^(-mu1 - mu2) (mu1/mu2)^(k/2) "
     "BesselI[k, 2 Sqrt[mu1 mu2]] on all integers, MGF, moments; CDF and quantile numeric for machine-number "
     "input. Check: SciPy skellam reference values.")
_uni("WaringYuleDistribution", WaringYule,
     "WaringYuleDistribution[alpha] (= [alpha, 1]) and [alpha, beta] on k >= 0: PDF alpha Pochhammer[beta, k]/"
     "Pochhammer[alpha + beta, k + 1], moments (Infinity when divergent). Check: exact.")
_uni("ZipfDistribution", Zipf,
     "ZipfDistribution[rho] (PDF k^(-rho - 1)/Zeta[rho + 1], k >= 1) and the truncated [n, rho] (HarmonicNumber "
     "normalisation); CDF, moments. Check: exact; SciPy zipf (a = rho + 1) reference values.")
