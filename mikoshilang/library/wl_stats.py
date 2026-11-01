"""Coverage batch 5: statistics and probability with Wolfram semantics.

* Distribution objects with Wolfram's parametrisations (GammaDistribution[alpha, beta] uses the *scale* beta,
  WeibullDistribution[alpha, beta] shape/scale, GeometricDistribution[p] counts failures before the first success,
  NegativeBinomialDistribution[n, p] counts failures before the n-th success, HypergeometricDistribution[n, nsucc,
  ntot], GumbelDistribution is the *minimum* extreme value distribution and ExtremeValueDistribution the maximum one).
* Distribution functions PDF, CDF, InverseCDF, SurvivalFunction, HazardFunction, Mean, Variance, StandardDeviation,
  Skewness, Kurtosis, Median, Quantile, Moment, CentralMoment, Cumulant, CharacteristicFunction,
  MomentGeneratingFunction, Expectation, Probability, NExpectation, NProbability and RandomVariate; exact for exact
  parameters, symbolic where Wolfram gives a closed form, and left unevaluated (never guessed) otherwise.
* Descriptive statistics that were missing (Quartiles, InterquartileRange, MeanDeviation, MedianDeviation, Moment,
  CentralMoment, Cumulant, WinsorizedMean, RootMeanSquare, ContraharmonicMean, Standardize, SpearmanRho,
  KendallTau, ...), column-wise statistics of matrices, random numbers (RandomInteger, RandomReal, RandomChoice) and
  basic inference (EstimatedDistribution, FindDistributionParameters, KolmogorovSmirnovTest, HistogramList, ...).

The distribution heads that the legacy statistics module already handled keep their legacy PDF/CDF/InverseCDF/Mean/
Variance/StandardDeviation/RandomVariate implementations (reviewed in the library audit); every other function and
distribution goes through this module.
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, record, REGISTRY, R, L, rule, S, sym, number, is_list, numeric_out, has_float,
                   list_arg, nonneg_int, int_arg)
from .wl_common import split_options

AREA = "statistics (batch 5)"
LEGACY = {"NormalDistribution", "ExponentialDistribution", "PoissonDistribution", "UniformDistribution",
          "BinomialDistribution", "ChiSquaredDistribution", "TDistribution", "StudentTDistribution", "FDistribution"}
LEGACY_FUNCS = {"PDF", "CDF", "InverseCDF", "Mean", "Variance", "StandardDeviation", "RandomVariate"}
INDET = Symbol("Indeterminate")
t_ = sp.Symbol("t")
X_ = sp.Symbol("x")


def _rng():
    from .statistics import _RNG
    return _RNG


# ─────────────────────────── distribution table ───────────────────────────

class D:
    """A univariate distribution: name, parameter list (SymPy) and closed forms."""
    discrete = False
    lo = -sp.oo           # support bounds (expressions in the parameters)
    hi = sp.oo

    def __init__(self, name, a):
        self.name, self.a = name, a

    # each subclass defines pdf(x) (on the support), cdf(x) (on the support) where known, mean, var, raw(r) and/or
    # mgf(t)/cf(t), q(p) quantile closed form, sample(rng)

    def support_cond(self, x):
        lo, hi = self.bounds()
        conds = []
        if lo != -sp.oo:
            conds.append(x > lo if self.open_lo else x >= lo)
        if hi != sp.oo:
            conds.append(x < hi if self.open_hi else x <= hi)
        return sp.And(*conds) if conds else sp.true

    open_lo = False
    open_hi = False

    def bounds(self):
        return self.lo, self.hi

    def numeric(self):
        return all(v.is_number for v in self.a)


def _num(v):
    return isinstance(v, sp.Basic) and v.is_number


_POS = []


def _pos(v, what, name):
    if isinstance(v, sp.Symbol):
        _POS.append(v)
    if _num(v) and not v > 0:
        raise ValueError(f"{name}: {what} must be positive")


def _prob(v, what, name, open_=False):
    if _num(v) and not (0 <= v <= 1) or (_num(v) and open_ and v in (0, 1)):
        raise ValueError(f"{name}: {what} must be in [0, 1]" if not open_ else f"{name}: {what} must be in (0, 1)")


def _nonneg_int_param(v, what, name):
    if _num(v) and not (v.is_Integer and v >= 0):
        raise ValueError(f"{name}: {what} must be a non-negative integer")


class Normal(D):
    def __init__(self, name, a):
        if len(a) == 0:
            a = [sp.Integer(0), sp.Integer(1)]
        if len(a) != 2:
            raise ValueError("NormalDistribution expects [mu, sigma]")
        _pos(a[1], "sigma", name)
        super().__init__(name, a)

    def pdf(self, x):
        m, s = self.a
        return sp.exp(-(x - m) ** 2 / (2 * s ** 2)) / (s * sp.sqrt(2 * sp.pi))

    def cdf(self, x):
        m, s = self.a
        return sp.erfc((m - x) / (s * sp.sqrt(2))) / 2

    def q(self, p):
        m, s = self.a
        return m - sp.sqrt(2) * s * sp.erfcinv(2 * p)

    def mean(self):
        return self.a[0]

    def var(self):
        return self.a[1] ** 2

    def mgf(self, t):
        m, s = self.a
        return sp.exp(m * t + s ** 2 * t ** 2 / 2)

    def sample(self, rng):
        return rng.gauss(float(self.a[0]), float(self.a[1]))


class Uniform(D):
    def __init__(self, name, a):
        if len(a) == 1 and isinstance(a[0], list):
            a = a[0]
        if len(a) == 0:
            a = [sp.Integer(0), sp.Integer(1)]
        if len(a) != 2:
            raise ValueError("UniformDistribution expects {min, max}")
        if _num(a[0]) and _num(a[1]) and not a[0] < a[1]:
            raise ValueError("UniformDistribution: min must be less than max")
        super().__init__(name, a)

    def bounds(self):
        return self.a[0], self.a[1]

    def pdf(self, x):
        return 1 / (self.a[1] - self.a[0])

    def cdf(self, x):
        return (x - self.a[0]) / (self.a[1] - self.a[0])

    def q(self, p):
        return self.a[0] + p * (self.a[1] - self.a[0])

    def mean(self):
        return (self.a[0] + self.a[1]) / 2

    def var(self):
        return (self.a[1] - self.a[0]) ** 2 / 12

    def raw(self, r):
        a, b = self.a
        return (b ** (r + 1) - a ** (r + 1)) / ((r + 1) * (b - a))

    def mgf(self, t):
        a, b = self.a
        return (sp.exp(b * t) - sp.exp(a * t)) / ((b - a) * t)

    def sample(self, rng):
        return rng.uniform(float(self.a[0]), float(self.a[1]))


class Exponential(D):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        if len(a) != 1:
            raise ValueError("ExponentialDistribution expects [lambda]")
        _pos(a[0], "lambda", name)
        super().__init__(name, a)

    def pdf(self, x):
        return self.a[0] * sp.exp(-self.a[0] * x)

    def cdf(self, x):
        return 1 - sp.exp(-self.a[0] * x)

    def q(self, p):
        return -sp.log(1 - p) / self.a[0]

    def mean(self):
        return 1 / self.a[0]

    def var(self):
        return 1 / self.a[0] ** 2

    def raw(self, r):
        return sp.factorial(r) / self.a[0] ** r

    def mgf(self, t):
        return self.a[0] / (self.a[0] - t)

    def sample(self, rng):
        return rng.expovariate(float(self.a[0]))


class Gamma(D):
    lo = sp.Integer(0)
    open_lo = True

    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError(f"{name} expects 2 parameters")
        _pos(a[0], "the shape", name)
        _pos(a[1], "the scale", name)
        super().__init__(name, a)

    def pdf(self, x):
        k, th = self.a
        return x ** (k - 1) * sp.exp(-x / th) / (th ** k * sp.gamma(k))

    def cdf(self, x):
        k, th = self.a
        return sp.lowergamma(k, x / th) / sp.gamma(k)

    def mean(self):
        return self.a[0] * self.a[1]

    def var(self):
        return self.a[0] * self.a[1] ** 2

    def raw(self, r):
        k, th = self.a
        return th ** r * sp.rf(k, r)

    def mgf(self, t):
        return (1 - self.a[1] * t) ** (-self.a[0])

    def sample(self, rng):
        return rng.gammavariate(float(self.a[0]), float(self.a[1]))

    def scipy(self, st):
        return st.gamma(float(self.a[0]), scale=float(self.a[1]))


class Erlang(Gamma):
    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError("ErlangDistribution expects [k, lambda]")
        if _num(a[0]) and not (a[0].is_Integer and a[0] > 0):
            raise ValueError("ErlangDistribution: k must be a positive integer")
        _pos(a[1], "lambda", name)
        D.__init__(self, name, [a[0], 1 / a[1]])
        self.shown = a


class ChiSquare(Gamma):
    def __init__(self, name, a):
        if len(a) != 1:
            raise ValueError(f"{name} expects [nu]")
        _pos(a[0], "nu", name)
        D.__init__(self, name, [a[0] / 2, sp.Integer(2)])
        self.nu = a[0]

    def sample(self, rng):
        return rng.gammavariate(float(self.nu) / 2, 2.0)

    def scipy(self, st):
        return st.chi2(float(self.nu))


class Beta(D):
    lo, hi = sp.Integer(0), sp.Integer(1)
    open_lo = open_hi = True

    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError("BetaDistribution expects [alpha, beta]")
        _pos(a[0], "alpha", name)
        _pos(a[1], "beta", name)
        super().__init__(name, a)

    def pdf(self, x):
        al, be = self.a
        return x ** (al - 1) * (1 - x) ** (be - 1) / sp.beta(al, be)

    def cdf(self, x):
        al, be = self.a
        if al.is_Integer and be.is_Integer and al > 0 and be > 0:
            n = int(al + be - 1)
            return sp.Add(*(sp.binomial(n, j) * x ** j * (1 - x) ** (n - j) for j in range(int(al), n + 1)))
        return sp.betainc_regularized(al, be, 0, x)

    def mean(self):
        al, be = self.a
        return al / (al + be)

    def var(self):
        al, be = self.a
        return al * be / ((al + be) ** 2 * (al + be + 1))

    def raw(self, r):
        al, be = self.a
        return sp.Mul(*[(al + i) / (al + be + i) for i in range(r)])

    def sample(self, rng):
        return rng.betavariate(float(self.a[0]), float(self.a[1]))

    def scipy(self, st):
        return st.beta(float(self.a[0]), float(self.a[1]))


class StudentT(D):
    def __init__(self, name, a):
        if len(a) == 1:
            a = [sp.Integer(0), sp.Integer(1), a[0]]
        if len(a) != 3:
            raise ValueError("StudentTDistribution expects [nu] or [mu, sigma, nu]")
        _pos(a[1], "sigma", name)
        _pos(a[2], "nu", name)
        super().__init__(name, a)

    def pdf(self, x):
        m, s, v = self.a
        z = (x - m) / s
        return (v / (v + z ** 2)) ** ((v + 1) / 2) / (s * sp.sqrt(v) * sp.beta(v / 2, sp.Rational(1, 2)))

    def cdf(self, x):
        m, s, v = self.a
        if not (_num(x) and self.numeric()):
            raise Unevaluated()
        z = (x - m) / s
        if z == 0:
            return sp.Rational(1, 2)
        if v.is_Integer and v <= 30 and not x.is_Float:
            u = sp.Dummy("u", real=True)
            dens = (v / (v + u ** 2)) ** ((v + 1) / 2) / (sp.sqrt(v) * sp.beta(v / 2, sp.Rational(1, 2)))
            half = sp.integrate(dens, (u, 0, sp.Abs(z)))
            if not half.has(sp.Integral):
                half = sp.simplify(half)
                return sp.Rational(1, 2) + half if z > 0 else sp.Rational(1, 2) - half
        tail = Beta("BetaDistribution", [v / 2, sp.Rational(1, 2)]).cdf(v / (v + z ** 2)) / 2
        return 1 - tail if z > 0 else tail

    def need(self, k):
        v = self.a[2]
        if not _num(v):
            raise Unevaluated()
        return v > k

    def mean(self):
        return self.a[0] if self.need(1) else sp.nan

    def var(self):
        v = self.a[2]
        if self.need(2):
            return self.a[1] ** 2 * v / (v - 2)
        return sp.nan

    def raw(self, r):
        m, s, v = self.a
        if not self.need(r):
            return sp.nan
        # E[(m + s T)^r] with E[T^k] = v^(k/2) Gamma((k+1)/2) Gamma((v-k)/2)/(Sqrt[Pi] Gamma[v/2]) for even k
        out = 0
        for k in range(0, r + 1, 2):
            tk = v ** sp.Rational(k, 2) * sp.gamma(sp.Rational(k + 1, 2)) * sp.gamma((v - k) / 2) / (
                sp.sqrt(sp.pi) * sp.gamma(v / 2))
            out += sp.binomial(r, k) * m ** (r - k) * s ** k * tk
        return out

    def sample(self, rng):
        m, s, v = (float(x) for x in self.a)
        return m + s * rng.gauss(0, 1) / math.sqrt(2 * rng.gammavariate(v / 2, 1.0) / v)

    def scipy(self, st):
        return st.t(float(self.a[2]), loc=float(self.a[0]), scale=float(self.a[1]))


class FRatio(D):
    lo = sp.Integer(0)
    open_lo = True

    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError(f"{name} expects [n, m]")
        _pos(a[0], "n", name)
        _pos(a[1], "m", name)
        super().__init__(name, a)

    def pdf(self, x):
        n, m = self.a
        return sp.sqrt((n * x) ** n * m ** m / (n * x + m) ** (n + m)) / (x * sp.beta(n / 2, m / 2))

    def cdf(self, x):
        n, m = self.a
        if not _num(x):
            raise Unevaluated()
        return Beta("BetaDistribution", [n / 2, m / 2]).cdf(n * x / (n * x + m))

    def need(self, k):
        if not _num(self.a[1]):
            raise Unevaluated()
        return self.a[1] > k

    def mean(self):
        m = self.a[1]
        return m / (m - 2) if self.need(2) else sp.nan

    def var(self):
        n, m = self.a
        return 2 * m ** 2 * (n + m - 2) / (n * (m - 2) ** 2 * (m - 4)) if self.need(4) else sp.nan

    def raw(self, r):
        n, m = self.a
        if not self.need(2 * r):
            return sp.nan
        return (m / n) ** r * sp.gamma(n / 2 + r) * sp.gamma(m / 2 - r) / (sp.gamma(n / 2) * sp.gamma(m / 2))

    def sample(self, rng):
        n, m = float(self.a[0]), float(self.a[1])
        return (2 * rng.gammavariate(n / 2, 1.0) / n) / (2 * rng.gammavariate(m / 2, 1.0) / m)

    def scipy(self, st):
        return st.f(float(self.a[0]), float(self.a[1]))


class LogNormal(D):
    lo = sp.Integer(0)
    open_lo = True

    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError("LogNormalDistribution expects [mu, sigma]")
        _pos(a[1], "sigma", name)
        super().__init__(name, a)

    def pdf(self, x):
        m, s = self.a
        return sp.exp(-(sp.log(x) - m) ** 2 / (2 * s ** 2)) / (sp.sqrt(2 * sp.pi) * s * x)

    def cdf(self, x):
        m, s = self.a
        return sp.erfc((m - sp.log(x)) / (sp.sqrt(2) * s)) / 2

    def q(self, p):
        m, s = self.a
        return sp.exp(m - sp.sqrt(2) * s * sp.erfcinv(2 * p))

    def mean(self):
        return sp.exp(self.a[0] + self.a[1] ** 2 / 2)

    def var(self):
        m, s = self.a
        return (sp.exp(s ** 2) - 1) * sp.exp(2 * m + s ** 2)

    def raw(self, r):
        m, s = self.a
        return sp.exp(r * m + r ** 2 * s ** 2 / 2)

    def sample(self, rng):
        return rng.lognormvariate(float(self.a[0]), float(self.a[1]))


class Cauchy(D):
    def __init__(self, name, a):
        if len(a) == 0:
            a = [sp.Integer(0), sp.Integer(1)]
        if len(a) != 2:
            raise ValueError("CauchyDistribution expects [a, b]")
        _pos(a[1], "b", name)
        super().__init__(name, a)

    def pdf(self, x):
        a, b = self.a
        return 1 / (sp.pi * b * (1 + ((x - a) / b) ** 2))

    def cdf(self, x):
        a, b = self.a
        return sp.Rational(1, 2) + sp.atan((x - a) / b) / sp.pi

    def q(self, p):
        a, b = self.a
        return a + b * sp.tan(sp.pi * (p - sp.Rational(1, 2)))

    def mean(self):
        return sp.nan

    def var(self):
        return sp.nan

    def raw(self, r):
        return sp.nan

    def cf(self, t):
        a, b = self.a
        return sp.exp(sp.I * a * t - b * sp.Abs(t))

    def sample(self, rng):
        a, b = float(self.a[0]), float(self.a[1])
        return a + b * math.tan(math.pi * (rng.random() - 0.5))


class Weibull(D):
    lo = sp.Integer(0)
    open_lo = True

    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError("WeibullDistribution expects [alpha, beta]")
        _pos(a[0], "the shape alpha", name)
        _pos(a[1], "the scale beta", name)
        super().__init__(name, a)

    def pdf(self, x):
        al, be = self.a
        return al / be * (x / be) ** (al - 1) * sp.exp(-(x / be) ** al)

    def cdf(self, x):
        al, be = self.a
        return 1 - sp.exp(-(x / be) ** al)

    def q(self, p):
        al, be = self.a
        return be * (-sp.log(1 - p)) ** (1 / al)

    def mean(self):
        al, be = self.a
        return be * sp.gamma(1 + 1 / al)

    def var(self):
        al, be = self.a
        return be ** 2 * (sp.gamma(1 + 2 / al) - sp.gamma(1 + 1 / al) ** 2)

    def raw(self, r):
        al, be = self.a
        return be ** r * sp.gamma(1 + r / al)

    def sample(self, rng):
        return rng.weibullvariate(float(self.a[1]), float(self.a[0]))


class Laplace(D):
    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError("LaplaceDistribution expects [mu, beta]")
        _pos(a[1], "beta", name)
        super().__init__(name, a)

    def pdf(self, x):
        m, b = self.a
        return sp.exp(-sp.Abs(x - m) / b) / (2 * b)

    def cdf(self, x):
        m, b = self.a
        if not (_num(x) and self.numeric()):
            return sp.Piecewise((sp.exp((x - m) / b) / 2, x < m), (1 - sp.exp(-(x - m) / b) / 2, True))
        return sp.exp((x - m) / b) / 2 if x < m else 1 - sp.exp(-(x - m) / b) / 2

    def q(self, p):
        m, b = self.a
        if not _num(p):
            raise Unevaluated()
        return m + b * sp.log(2 * p) if p <= sp.Rational(1, 2) else m - b * sp.log(2 * (1 - p))

    def mean(self):
        return self.a[0]

    def var(self):
        return 2 * self.a[1] ** 2

    def mgf(self, t):
        m, b = self.a
        return sp.exp(m * t) / (1 - b ** 2 * t ** 2)

    def sample(self, rng):
        m, b = float(self.a[0]), float(self.a[1])
        u = rng.random() - 0.5
        return m - b * math.copysign(1, u) * math.log(1 - 2 * abs(u))


class Logistic(D):
    def __init__(self, name, a):
        if len(a) == 0:
            a = [sp.Integer(0), sp.Integer(1)]
        if len(a) != 2:
            raise ValueError("LogisticDistribution expects [mu, beta]")
        _pos(a[1], "beta", name)
        super().__init__(name, a)

    def pdf(self, x):
        m, b = self.a
        return sp.exp(-(x - m) / b) / (b * (1 + sp.exp(-(x - m) / b)) ** 2)

    def cdf(self, x):
        m, b = self.a
        return 1 / (1 + sp.exp(-(x - m) / b))

    def q(self, p):
        m, b = self.a
        return m + b * sp.log(p / (1 - p))

    def mean(self):
        return self.a[0]

    def var(self):
        return sp.pi ** 2 * self.a[1] ** 2 / 3

    def mgf(self, t):
        m, b = self.a
        return sp.exp(m * t) * sp.pi * b * t / sp.sin(sp.pi * b * t)

    def sample(self, rng):
        m, b = float(self.a[0]), float(self.a[1])
        u = rng.random()
        while u == 0.0:
            u = rng.random()
        return m + b * math.log(u / (1 - u))


class Rayleigh(D):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        if len(a) != 1:
            raise ValueError("RayleighDistribution expects [sigma]")
        _pos(a[0], "sigma", name)
        super().__init__(name, a)

    def pdf(self, x):
        s = self.a[0]
        return x * sp.exp(-x ** 2 / (2 * s ** 2)) / s ** 2

    def cdf(self, x):
        s = self.a[0]
        return 1 - sp.exp(-x ** 2 / (2 * s ** 2))

    def q(self, p):
        return self.a[0] * sp.sqrt(-2 * sp.log(1 - p))

    def mean(self):
        return self.a[0] * sp.sqrt(sp.pi / 2)

    def var(self):
        return (4 - sp.pi) * self.a[0] ** 2 / 2

    def raw(self, r):
        s = self.a[0]
        return s ** r * 2 ** sp.Rational(r, 2) * sp.gamma(1 + sp.Rational(r, 2))

    def sample(self, rng):
        return float(self.a[0]) * math.sqrt(-2 * math.log(1 - rng.random()))


class Pareto(D):
    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError("ParetoDistribution expects [k, alpha]")
        _pos(a[0], "k", name)
        _pos(a[1], "alpha", name)
        super().__init__(name, a)

    def bounds(self):
        return self.a[0], sp.oo

    def pdf(self, x):
        k, al = self.a
        return al * k ** al * x ** (-al - 1)

    def cdf(self, x):
        k, al = self.a
        return 1 - (k / x) ** al

    def q(self, p):
        k, al = self.a
        return k * (1 - p) ** (-1 / al)

    def need(self, r):
        if not _num(self.a[1]):
            raise Unevaluated()
        return self.a[1] > r

    def mean(self):
        k, al = self.a
        return al * k / (al - 1) if self.need(1) else sp.nan

    def var(self):
        k, al = self.a
        return k ** 2 * al / ((al - 1) ** 2 * (al - 2)) if self.need(2) else sp.nan

    def raw(self, r):
        k, al = self.a
        return al * k ** r / (al - r) if self.need(r) else sp.nan

    def sample(self, rng):
        k, al = float(self.a[0]), float(self.a[1])
        return k * (1 - rng.random()) ** (-1 / al)


class ExtremeValue(D):
    """ExtremeValueDistribution[alpha, beta]: the maximum (Gumbel, right-skewed); sign = -1 gives
    GumbelDistribution, Wolfram's minimum extreme value distribution."""
    sign = 1

    def __init__(self, name, a):
        if len(a) == 0:
            a = [sp.Integer(0), sp.Integer(1)]
        if len(a) != 2:
            raise ValueError(f"{name} expects [alpha, beta]")
        _pos(a[1], "beta", name)
        super().__init__(name, a)

    def pdf(self, x):
        al, be = self.a
        z = self.sign * (x - al) / be
        return sp.exp(-z - sp.exp(-z)) / be

    def cdf(self, x):
        al, be = self.a
        z = self.sign * (x - al) / be
        return sp.exp(-sp.exp(-z)) if self.sign == 1 else 1 - sp.exp(-sp.exp(-z))

    def q(self, p):
        al, be = self.a
        if self.sign == 1:
            return al - be * sp.log(-sp.log(p))
        return al + be * sp.log(-sp.log(1 - p))

    def mean(self):
        al, be = self.a
        return al + self.sign * sp.EulerGamma * be

    def var(self):
        return sp.pi ** 2 * self.a[1] ** 2 / 6

    def mgf(self, t):
        al, be = self.a
        return sp.exp(al * t) * sp.gamma(1 - self.sign * be * t)

    def sample(self, rng):
        al, be = float(self.a[0]), float(self.a[1])
        u = rng.random()
        while u == 0.0:
            u = rng.random()
        return al - be * math.log(-math.log(u)) if self.sign == 1 else al + be * math.log(-math.log(u))


class Gumbel(ExtremeValue):
    sign = -1


class Triangular(D):
    def __init__(self, name, a):
        if len(a) == 1 and isinstance(a[0], list):
            lo, hi = a[0]
            c = (lo + hi) / 2
        elif len(a) == 2 and isinstance(a[0], list):
            (lo, hi), c = a[0], a[1]
        elif len(a) == 0:
            lo, hi, c = sp.Integer(0), sp.Integer(1), sp.Rational(1, 2)
        else:
            raise ValueError("TriangularDistribution expects [{min, max}] or [{min, max}, c]")
        if all(_num(v) for v in (lo, hi, c)) and not (lo < hi and lo <= c <= hi):
            raise ValueError("TriangularDistribution: need min <= c <= max and min < max")
        super().__init__(name, [lo, hi, c])

    def bounds(self):
        return self.a[0], self.a[1]

    def pdf(self, x):
        a, b, c = self.a
        if not (_num(x) and self.numeric()):
            return sp.Piecewise((2 * (x - a) / ((b - a) * (c - a)), x <= c), (2 * (b - x) / ((b - a) * (b - c)), True))
        return 2 * (x - a) / ((b - a) * (c - a)) if (x < c or (x == c and c != a)) else 2 * (b - x) / ((b - a) * (b - c))

    def cdf(self, x):
        a, b, c = self.a
        if not (_num(x) and self.numeric()):
            raise Unevaluated()
        return (x - a) ** 2 / ((b - a) * (c - a)) if x <= c and c != a else 1 - (b - x) ** 2 / ((b - a) * (b - c))

    def q(self, p):
        a, b, c = self.a
        if not (_num(p) and self.numeric()):
            raise Unevaluated()
        f = (c - a) / (b - a)
        return a + sp.sqrt(p * (b - a) * (c - a)) if p <= f else b - sp.sqrt((1 - p) * (b - a) * (b - c))

    def mean(self):
        return sp.Add(*self.a) / 3

    def var(self):
        a, b, c = self.a
        return (a ** 2 + b ** 2 + c ** 2 - a * b - a * c - b * c) / 18

    def raw(self, r):
        a, b, c = self.a
        x = sp.Dummy("x")
        if not self.numeric():
            raise Unevaluated()
        f1 = 2 * (x - a) / ((b - a) * (c - a)) if c != a else 0
        f2 = 2 * (b - x) / ((b - a) * (b - c)) if c != b else 0
        return sp.integrate(x ** r * f1, (x, a, c)) + sp.integrate(x ** r * f2, (x, c, b))

    def sample(self, rng):
        a, b, c = (float(v) for v in self.a)
        return rng.triangular(a, b, c)


class Chi(D):
    lo = sp.Integer(0)

    def __init__(self, name, a):
        if len(a) != 1:
            raise ValueError("ChiDistribution expects [nu]")
        _pos(a[0], "nu", name)
        super().__init__(name, a)

    def pdf(self, x):
        v = self.a[0]
        return 2 ** (1 - v / 2) * x ** (v - 1) * sp.exp(-x ** 2 / 2) / sp.gamma(v / 2)

    def cdf(self, x):
        v = self.a[0]
        return sp.lowergamma(v / 2, x ** 2 / 2) / sp.gamma(v / 2)

    def mean(self):
        v = self.a[0]
        return sp.sqrt(2) * sp.gamma((v + 1) / 2) / sp.gamma(v / 2)

    def var(self):
        return self.a[0] - self.mean() ** 2

    def raw(self, r):
        v = self.a[0]
        return 2 ** sp.Rational(r, 2) * sp.gamma((v + r) / 2) / sp.gamma(v / 2)

    def sample(self, rng):
        return math.sqrt(2 * rng.gammavariate(float(self.a[0]) / 2, 1.0))

    def scipy(self, st):
        return st.chi(float(self.a[0]))


# ── discrete ──

class Discrete(D):
    discrete = True
    lo = sp.Integer(0)

    def in_support(self, k):
        lo, hi = self.bounds()
        return k.is_integer and k >= lo and (hi == sp.oo or k <= hi)

    def sample(self, rng):
        u = rng.random()
        lo, hi = self.bounds()
        k, acc = int(lo), 0.0
        while True:
            acc += float(self.pdf(sp.Integer(k)))
            if u < acc or (hi != sp.oo and k >= hi):
                return k
            k += 1
            if k - int(lo) > 10 ** 7:
                raise ValueError("RandomVariate: sampling did not terminate")


class Bernoulli(Discrete):
    hi = sp.Integer(1)

    def __init__(self, name, a):
        if len(a) != 1:
            raise ValueError("BernoulliDistribution expects [p]")
        _prob(a[0], "p", name)
        super().__init__(name, a)

    def pdf(self, k):
        p = self.a[0]
        return sp.Piecewise((1 - p, sp.Eq(k, 0)), (p, True)) if not _num(k) else (1 - p if k == 0 else p)

    def mean(self):
        return self.a[0]

    def var(self):
        p = self.a[0]
        return p * (1 - p)

    def mgf(self, t):
        p = self.a[0]
        return 1 - p + p * sp.exp(t)

    def sample(self, rng):
        return 1 if rng.random() < float(self.a[0]) else 0


class Binomial(Discrete):
    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError("BinomialDistribution expects [n, p]")
        _nonneg_int_param(a[0], "n", name)
        _prob(a[1], "p", name)
        super().__init__(name, a)

    def bounds(self):
        return sp.Integer(0), self.a[0]

    def pdf(self, k):
        n, p = self.a
        return sp.binomial(n, k) * p ** k * (1 - p) ** (n - k)

    def mean(self):
        return self.a[0] * self.a[1]

    def var(self):
        n, p = self.a
        return n * p * (1 - p)

    def mgf(self, t):
        n, p = self.a
        return (1 - p + p * sp.exp(t)) ** n

    def sample(self, rng):
        return sum(1 for _ in range(int(self.a[0])) if rng.random() < float(self.a[1]))


class Poisson(Discrete):
    def __init__(self, name, a):
        if len(a) != 1:
            raise ValueError("PoissonDistribution expects [mu]")
        _pos(a[0], "mu", name)
        super().__init__(name, a)

    def pdf(self, k):
        mu = self.a[0]
        return sp.exp(-mu) * mu ** k / sp.factorial(k)

    def mean(self):
        return self.a[0]

    def var(self):
        return self.a[0]

    def mgf(self, t):
        return sp.exp(self.a[0] * (sp.exp(t) - 1))

    def sample(self, rng):
        lam, kk, p = math.exp(-float(self.a[0])), 0, 1.0
        while True:
            p *= rng.random()
            if p <= lam:
                return kk
            kk += 1


class Geometric(Discrete):
    def __init__(self, name, a):
        if len(a) != 1:
            raise ValueError("GeometricDistribution expects [p]")
        if _num(a[0]) and not (0 < a[0] <= 1):
            raise ValueError("GeometricDistribution: p must be in (0, 1]")
        super().__init__(name, a)

    def pdf(self, k):
        p = self.a[0]
        return p * (1 - p) ** k

    def cdf_closed(self, k):
        return 1 - (1 - self.a[0]) ** (k + 1)

    def mean(self):
        p = self.a[0]
        return (1 - p) / p

    def var(self):
        p = self.a[0]
        return (1 - p) / p ** 2

    def mgf(self, t):
        p = self.a[0]
        return p / (1 - (1 - p) * sp.exp(t))

    def sample(self, rng):
        p = float(self.a[0])
        if p == 1:
            return 0
        u = 1 - rng.random()
        return int(math.floor(math.log(u) / math.log(1 - p)))


class NegativeBinomial(Discrete):
    def __init__(self, name, a):
        if len(a) != 2:
            raise ValueError("NegativeBinomialDistribution expects [n, p]")
        _pos(a[0], "n", name)
        if _num(a[1]) and not (0 < a[1] <= 1):
            raise ValueError("NegativeBinomialDistribution: p must be in (0, 1]")
        super().__init__(name, a)

    def pdf(self, k):
        n, p = self.a
        return sp.binomial(k + n - 1, n - 1) * p ** n * (1 - p) ** k

    def mean(self):
        n, p = self.a
        return n * (1 - p) / p

    def var(self):
        n, p = self.a
        return n * (1 - p) / p ** 2

    def mgf(self, t):
        n, p = self.a
        return (p / (1 - (1 - p) * sp.exp(t))) ** n

    def sample(self, rng):
        n, p = float(self.a[0]), float(self.a[1])
        if p == 1:
            return 0
        lam = rng.gammavariate(n, (1 - p) / p)
        return Poisson("PoissonDistribution", [sp.Float(lam)]).sample(rng) if lam > 0 else 0


class Hypergeometric(Discrete):
    def __init__(self, name, a):
        if len(a) != 3:
            raise ValueError("HypergeometricDistribution expects [n, nsucc, ntot]")
        for v, w in zip(a, ("n", "nsucc", "ntot")):
            _nonneg_int_param(v, w, name)
        if all(_num(v) for v in a) and not (a[0] <= a[2] and a[1] <= a[2]):
            raise ValueError("HypergeometricDistribution: need n <= ntot and nsucc <= ntot")
        super().__init__(name, a)

    def bounds(self):
        n, K, N = self.a
        return sp.Max(0, n + K - N), sp.Min(n, K)

    def pdf(self, k):
        n, K, N = self.a
        return sp.binomial(K, k) * sp.binomial(N - K, n - k) / sp.binomial(N, n)

    def mean(self):
        n, K, N = self.a
        return n * K / N

    def var(self):
        n, K, N = self.a
        return n * K * (N - K) * (N - n) / (N ** 2 * (N - 1))


class DiscreteUniform(Discrete):
    def __init__(self, name, a):
        if len(a) != 1 or not isinstance(a[0], list) or len(a[0]) != 2:
            raise ValueError("DiscreteUniformDistribution expects [{imin, imax}]")
        lo, hi = a[0]
        if _num(lo) and _num(hi) and not (lo.is_Integer and hi.is_Integer and lo <= hi):
            raise ValueError("DiscreteUniformDistribution: need integers imin <= imax")
        super().__init__(name, [lo, hi])

    def bounds(self):
        return self.a[0], self.a[1]

    def pdf(self, k):
        return 1 / (self.a[1] - self.a[0] + 1)

    def mean(self):
        return (self.a[0] + self.a[1]) / 2

    def var(self):
        n = self.a[1] - self.a[0] + 1
        return (n ** 2 - 1) / 12

    def sample(self, rng):
        return rng.randint(int(self.a[0]), int(self.a[1]))


class Empirical(Discrete):
    """EmpiricalDistribution[{x1, ...}] (equal weights) or [{w1, ...} -> {x1, ...}]."""

    def __init__(self, name, a, raw_arg=None):
        vals, weights = raw_arg
        pts = {}
        order = []
        tot = sp.Add(*weights)
        for v, w in zip(vals, weights):
            key = v
            if key not in pts:
                pts[key] = sp.Integer(0)
                order.append(key)
            pts[key] += w / tot
        self.pts = sorted(order)
        self.w = pts
        super().__init__(name, [])

    def bounds(self):
        return self.pts[0], self.pts[-1]

    def in_support(self, k):
        return k in self.w

    def pdf(self, k):
        return self.w.get(k, sp.Integer(0))

    def cdf_value(self, x):
        return sp.Add(*(self.w[p] for p in self.pts if p <= x))

    def mean(self):
        return sp.Add(*(p * self.w[p] for p in self.pts))

    def var(self):
        m = self.mean()
        return sp.Add(*((p - m) ** 2 * self.w[p] for p in self.pts))

    def raw(self, r):
        return sp.Add(*(p ** r * self.w[p] for p in self.pts))

    def q(self, p):
        if not _num(p):
            raise Unevaluated()
        acc = sp.Integer(0)
        for x in self.pts:
            acc += self.w[x]
            if acc >= p:
                return x
        return self.pts[-1]

    def sample(self, rng):
        u, acc = rng.random(), 0.0
        for x in self.pts:
            acc += float(self.w[x])
            if u < acc:
                return x
        return self.pts[-1]


TABLE = {
    "NormalDistribution": Normal, "UniformDistribution": Uniform, "ExponentialDistribution": Exponential,
    "GammaDistribution": Gamma, "ErlangDistribution": Erlang, "ChiSquareDistribution": ChiSquare,
    "ChiSquaredDistribution": ChiSquare, "BetaDistribution": Beta, "StudentTDistribution": StudentT,
    "TDistribution": StudentT, "FRatioDistribution": FRatio, "FDistribution": FRatio,
    "LogNormalDistribution": LogNormal, "CauchyDistribution": Cauchy, "WeibullDistribution": Weibull,
    "LaplaceDistribution": Laplace, "LogisticDistribution": Logistic, "RayleighDistribution": Rayleigh,
    "ParetoDistribution": Pareto, "ExtremeValueDistribution": ExtremeValue, "GumbelDistribution": Gumbel,
    "TriangularDistribution": Triangular, "ChiDistribution": Chi,
    "BernoulliDistribution": Bernoulli, "BinomialDistribution": Binomial, "PoissonDistribution": Poisson,
    "GeometricDistribution": Geometric, "NegativeBinomialDistribution": NegativeBinomial,
    "HypergeometricDistribution": Hypergeometric, "DiscreteUniformDistribution": DiscreteUniform,
}
MULTI = {"MultinormalDistribution", "MultinomialDistribution"}
NEW_DISTS = (set(TABLE) | MULTI | {"EmpiricalDistribution", "TransformedDistribution"}) - LEGACY


def _param(v, name):
    if is_list(v):
        return [_param(x, name) for x in v.args]
    return sym(v, name)


def dist_of(d, fname):
    """D instance for a distribution expression; unknown distribution heads raise, non-distributions too."""
    if not isinstance(d, Expr) or not isinstance(d.head, str):
        if isinstance(d, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: {d} is not a distribution")
    h = d.head
    if h == "EmpiricalDistribution":
        return _empirical(d, fname)
    if h in TABLE:
        del _POS[:]
        dd = TABLE[h](h, [_param(v, fname) for v in d.args])
        dd.pos = list(_POS)
        return dd
    if h.endswith("Distribution"):
        raise NotImplementedError(f"{fname}: {h} is not supported")
    raise ValueError(f"{fname}: {d} is not a distribution")


def _empirical(d, fname):
    if len(d.args) != 1:
        raise ValueError("EmpiricalDistribution expects a list of data")
    a = d.args[0]
    if isinstance(a, Expr) and a.head == "Rule":
        ws = list_arg(a.args[0], fname, "weights", 1)
        vs = list_arg(a.args[1], fname, "data", 1)
        if len(ws) != len(vs):
            raise ValueError("EmpiricalDistribution: weights and data must have the same length")
        weights = [number(w, fname, "weight") for w in ws]
        if any(w < 0 for w in weights) or sp.Add(*weights) <= 0:
            raise ValueError("EmpiricalDistribution: weights must be non-negative with a positive total")
    else:
        vs = list_arg(a, fname, "data", 1)
        weights = [sp.Integer(1)] * len(vs)
    if any(is_list(v) for v in vs):
        raise NotImplementedError("EmpiricalDistribution: multivariate data is not supported")
    vals = [number(v, fname, "data value") for v in vs]
    return Empirical("EmpiricalDistribution", [], (vals, weights))


def is_dist(x):
    return isinstance(x, Expr) and isinstance(x.head, str) and (x.head in TABLE or x.head in MULTI or
                                                              x.head in ("EmpiricalDistribution", "TransformedDistribution"))


def _out(v, *inputs):
    if v is sp.nan or v == sp.nan:
        return INDET
    return numeric_out(v, *inputs)


def _simp(v, dd=None):
    """simplify, treating parameters that the distribution requires to be positive as positive symbols"""
    try:
        pos = getattr(dd, "pos", None) or []
        if pos and isinstance(v, sp.Basic):
            reps = {s: sp.Dummy(s.name, positive=True) for s in pos}
            back = {d_: s for s, d_ in reps.items()}
            return sp.simplify(sp.simplify(v.subs(reps)).subs(back))
        return sp.simplify(v)
    except Exception:  # pragma: no cover
        return v


# ─────────────────────────── distribution constructors ───────────────────────────

def _register_dist(name):
    def impl(*args):
        dist_of(Expr(name, *args), name)          # validates numeric parameters
        if name in ("CauchyDistribution", "LogisticDistribution", "ExtremeValueDistribution", "GumbelDistribution") \
                and not args:
            return Expr(name, 0, 1)
        if name == "UniformDistribution" and not args:
            return Expr(name, L(0, 1))
        if name == "TriangularDistribution" and not args:
            return Expr(name, L(0, 1))
        raise Unevaluated()
    import inspect
    impl.__signature__ = inspect.Signature([inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL)])
    fn(name, area=AREA, status="added", notes=f"inert distribution object {name}; parameters validated")(impl)


for _n in sorted(NEW_DISTS - {"MultinormalDistribution", "MultinomialDistribution", "EmpiricalDistribution",
                              "TransformedDistribution"}):
    _register_dist(_n)


@fn("EmpiricalDistribution", area=AREA, status="added",
    notes="EmpiricalDistribution[{x1, ...}] (equal weights) or [{w...} -> {x...}]: discrete distribution of the data "
          "(PDF = relative frequency, population moments)")
def _emp_ctor(a):
    dist_of(Expr("EmpiricalDistribution", a), "EmpiricalDistribution")
    raise Unevaluated()


# ─────────────────────────── core distribution functions ───────────────────────────

def _x(x, fname):
    if is_list(x):
        raise Unevaluated()
    return sym(x, fname)


def pdf_value(dd, x):
    if dd.discrete:
        if _num(x):
            if isinstance(dd, Empirical):
                return dd.pdf(x)
            lo, hi = dd.bounds()
            if not (x.is_integer and (not _num(lo) or x >= lo) and (not _num(hi) or x <= hi)):
                return sp.Integer(0)
            return dd.pdf(x)
        if isinstance(dd, Empirical):
            raise Unevaluated()
        lo, hi = dd.bounds()
        cond = sp.And(x >= lo, x <= hi) if hi != sp.oo else (x >= lo)
        return sp.Piecewise((dd.pdf(x), cond), (0, True))
    lo, hi = dd.bounds()
    if _num(x) and dd.numeric():
        c = dd.support_cond(x)
        if c in (sp.true, True):
            return dd.pdf(x)
        if c in (sp.false, False):
            return sp.Integer(0)
    c = dd.support_cond(x)
    if c is sp.true:
        return dd.pdf(x)
    return sp.Piecewise((dd.pdf(x), c), (0, True))


def cdf_value(dd, x):
    if isinstance(dd, Empirical):
        if not _num(x):
            raise Unevaluated()
        return dd.cdf_value(x)
    lo, hi = dd.bounds()
    if dd.discrete:
        if not (_num(x) and dd.numeric()):
            if hasattr(dd, "cdf_closed"):
                return sp.Piecewise((dd.cdf_closed(sp.floor(x)), x >= 0), (0, True))
            raise Unevaluated()
        if x < lo:
            return sp.Integer(0)
        if hi != sp.oo and x >= hi:
            return sp.Integer(1)
        k = int(sp.floor(x))
        if hasattr(dd, "cdf_closed"):
            return dd.cdf_closed(sp.Integer(k))
        return sp.Add(*(dd.pdf(sp.Integer(i)) for i in range(int(lo), k + 1)))
    if _num(x) and dd.numeric():
        if x <= lo:
            return sp.Integer(0)
        if hi != sp.oo and x >= hi:
            return sp.Integer(1)
        return dd.cdf(x)
    conds = []
    if lo != -sp.oo:
        conds.append((0, x <= lo if not dd.open_lo else x <= lo))
    inner = dd.cdf(x)
    if hi != sp.oo:
        conds.append((inner, x < hi))
        conds.append((1, True))
    else:
        conds.append((inner, True))
    if len(conds) == 1:
        return inner
    return sp.Piecewise(*conds)


def quantile_value(dd, p, inputs, fname):
    if _num(p) and not (0 <= p <= 1):
        raise ValueError(f"{fname}: the probability must be in [0, 1]")
    lo, hi = dd.bounds()
    if _num(p) and p == 0:
        if dd.discrete:
            raise Unevaluated()
        return lo
    if _num(p) and p == 1:
        return hi
    if dd.discrete:
        if not (_num(p) and dd.numeric()):
            raise Unevaluated()
        if isinstance(dd, Empirical):
            return dd.q(p)
        k = int(lo)
        acc = dd.pdf(sp.Integer(k))
        while acc < p:
            k += 1
            acc += dd.pdf(sp.Integer(k))
            if k - int(lo) > 10 ** 6:
                raise ValueError(f"{fname}: quantile too large")
        return sp.Integer(k)
    if hasattr(dd, "q"):
        return dd.q(p)
    if not (_num(p) and dd.numeric() and any(has_float(i) for i in inputs)):
        raise Unevaluated()
    try:
        import scipy.stats as st
    except ImportError:
        raise NotImplementedError(f"{fname}: numeric quantiles of {dd.name} require SciPy")
    return sp.Float(float(dd.scipy(st).ppf(float(p))))


# ── moments ──

def raw_moment(dd, r):
    if hasattr(dd, "raw"):
        return dd.raw(r)
    if r == 1:
        return dd.mean()
    if r == 2:
        return dd.var() + dd.mean() ** 2
    if dd.discrete and dd.numeric():
        lo, hi = dd.bounds()
        if hi != sp.oo:
            return sp.Add(*(sp.Integer(k) ** r * dd.pdf(sp.Integer(k)) for k in range(int(lo), int(hi) + 1)))
    if hasattr(dd, "mgf"):
        s = sp.series(dd.mgf(t_), t_, 0, r + 1).removeO()
        return sp.factorial(r) * s.coeff(t_, r)
    raise Unevaluated()


def central_moment(dd, r):
    m = dd.mean()
    if m is sp.nan:
        return sp.nan
    if r == 1:
        return sp.Integer(0)
    if r == 2:
        return dd.var()
    terms = [raw_moment(dd, k) for k in range(r + 1)]
    if any(x is sp.nan or x == sp.oo for x in terms[1:]):
        return sp.nan if any(x is sp.nan for x in terms[1:]) else sp.oo
    return sp.Add(*(sp.binomial(r, k) * terms[k] * (-m) ** (r - k) for k in range(r + 1)))


def cumulant(dd, r):
    raws = [sp.Integer(1)] + [raw_moment(dd, k) for k in range(1, r + 1)]
    if any(x is sp.nan or x in (sp.oo, -sp.oo) for x in raws):
        return sp.nan
    kap = [sp.Integer(0)] * (r + 1)
    for n in range(1, r + 1):
        kap[n] = raws[n] - sp.Add(*(sp.binomial(n - 1, m - 1) * kap[m] * raws[n - m] for m in range(1, n)))
    return kap[r]


def skewness(dd):
    v = dd.var()
    if v is sp.nan or v == sp.oo:
        return sp.nan
    if isinstance(dd, (Normal, Laplace, Logistic, Uniform, DiscreteUniform)):
        return sp.Integer(0)
    if isinstance(dd, ExtremeValue):
        return dd.sign * 12 * sp.sqrt(6) * sp.zeta(3) / sp.pi ** 3
    if isinstance(dd, StudentT):
        return sp.Integer(0) if dd.need(3) else sp.nan
    c3 = central_moment(dd, 3)
    if c3 is sp.nan or c3 == sp.oo:
        return sp.nan
    return c3 / v ** sp.Rational(3, 2)


def kurtosis(dd):
    v = dd.var()
    if v is sp.nan or v == sp.oo:
        return sp.nan
    if isinstance(dd, Normal):
        return sp.Integer(3)
    if isinstance(dd, Laplace):
        return sp.Integer(6)
    if isinstance(dd, Logistic):
        return sp.Rational(21, 5)
    if isinstance(dd, ExtremeValue):
        return sp.Rational(27, 5)
    if isinstance(dd, StudentT):
        nu = dd.a[2]
        return 3 + 6 / (nu - 4) if dd.need(4) else sp.nan
    c4 = central_moment(dd, 4)
    if c4 is sp.nan:
        return sp.nan
    return c4 / v ** 2


def cf_value(dd, t):
    if hasattr(dd, "cf"):
        return dd.cf(t)
    if hasattr(dd, "mgf"):
        return dd.mgf(sp.I * t)
    if isinstance(dd, Empirical) or (dd.discrete and dd.numeric() and dd.bounds()[1] != sp.oo):
        lo, hi = dd.bounds()
        pts = dd.pts if isinstance(dd, Empirical) else [sp.Integer(k) for k in range(int(lo), int(hi) + 1)]
        return sp.Add(*(dd.pdf(k) * sp.exp(sp.I * k * t) for k in pts))
    raise Unevaluated()


def mgf_value(dd, t):
    if isinstance(dd, Cauchy):
        raise Unevaluated()
    if hasattr(dd, "mgf"):
        return dd.mgf(t)
    if isinstance(dd, Empirical) or (dd.discrete and dd.numeric() and dd.bounds()[1] != sp.oo):
        lo, hi = dd.bounds()
        pts = dd.pts if isinstance(dd, Empirical) else [sp.Integer(k) for k in range(int(lo), int(hi) + 1)]
        return sp.Add(*(dd.pdf(k) * sp.exp(k * t) for k in pts))
    raise Unevaluated()


# ─────────────────────────── routing for legacy heads ───────────────────────────

def extend(name, impl, accepts, notes, status="fixed"):
    """Send calls accepted by ``accepts`` to ``impl``; others keep the existing implementation."""
    e = REGISTRY.get(name)
    old = list(e.impls) if e is not None else []

    def f(*args):
        if accepts(args):
            return impl(*args)
        for g, lo, hi in old:
            if len(args) >= lo and (hi is None or len(args) <= hi):
                return g(*args)
        raise ValueError(f"{name} called with {len(args)} argument(s)")
    if e is None:
        fn(name, area=AREA, status=status, notes=notes)(f)
        REGISTRY[name].impls = [(f, 0, None)]
    else:
        e.impls = [(f, 0, None)]
        e.notes = notes + ("; " + e.notes if e.notes else "")
        record(name, e.area, status, e.notes)
    return f


def _new_dist_first(args):
    return bool(args) and isinstance(args[0], Expr) and (
        args[0].head in NEW_DISTS or (args[0].head == "StudentTDistribution" and len(args[0].args) == 3))


def _any_dist_first(args):
    return bool(args) and is_dist(args[0])


def _matrix_first(args):
    return len(args) == 1 and is_list(args[0]) and args[0].args and all(is_list(r) for r in args[0].args)


# PDF / CDF / InverseCDF for the new distributions
def _pdf(d, x):
    if d.head in MULTI:
        return _multi_pdf(d, x)
    if d.head == "TransformedDistribution":
        raise Unevaluated()
    dd = dist_of(d, "PDF")
    return _out(pdf_value(dd, _x(x, "PDF")), d, x)


def _cdf(d, x):
    if d.head in MULTI:
        return _multi_cdf(d, x)
    if d.head == "TransformedDistribution":
        raise Unevaluated()
    dd = dist_of(d, "CDF")
    return _out(cdf_value(dd, _x(x, "CDF")), d, x)


def _invcdf(d, p):
    dd = dist_of(d, "InverseCDF")
    pv = sym(p, "InverseCDF")
    return _out(quantile_value(dd, pv, (d, p), "InverseCDF"), d, p)


extend("PDF", lambda d, x: _pdf(d, x), _new_dist_first,
       "batch 5 distributions: Wolfram densities/PMFs with Piecewise support conditions for symbolic x")
extend("CDF", lambda d, x: _cdf(d, x), _new_dist_first,
       "batch 5 distributions: exact CDFs (regularised gamma/beta forms; discrete CDFs at Floor[x])")
extend("InverseCDF", lambda d, p: _invcdf(d, p), _new_dist_first,
       "batch 5 distributions: closed-form quantiles where they exist, smallest k with CDF >= p for discrete "
       "distributions, SciPy for machine-number p otherwise (exact p without a closed form stays unevaluated)")


def _moment_fn(which):
    def impl(d, *rest):
        if d.head in MULTI:
            return _multi_moment(d, which)
        if d.head == "TransformedDistribution":
            return _transformed_moment(d, which)
        dd = dist_of(d, which)
        if which == "Mean":
            v = dd.mean()
        elif which == "Variance":
            v = dd.var()
        elif which == "StandardDeviation":
            v = dd.var()
            v = sp.nan if v is sp.nan else sp.sqrt(v)
        elif which == "Skewness":
            v = skewness(dd)
        elif which == "Kurtosis":
            v = kurtosis(dd)
        elif which == "Median":
            v = quantile_value(dd, sp.Rational(1, 2), (d,), "Median")
        if v is sp.nan:
            return INDET
        return _out(_simp(v, dd), d)
    return impl


for _w in ("Mean", "Variance", "StandardDeviation"):
    extend(_w, _moment_fn(_w), _new_dist_first,
           f"{_w}[dist] for the batch 5 distributions (Indeterminate when the moment does not exist)")
for _w in ("Skewness", "Kurtosis", "Median"):
    extend(_w, _moment_fn(_w), _any_dist_first,
           f"{_w}[dist] for every supported distribution (Indeterminate when undefined; Kurtosis is the non-excess "
           "kurtosis, 3 for the normal distribution)")


# ─────────────────────────── new distribution functions ───────────────────────────

@fn("SurvivalFunction", area=AREA, status="added",
    notes="1 - CDF[dist, x] (P(X > x)); exact; Piecewise for symbolic x")
def _surv(d, x):
    dd = dist_of(d, "SurvivalFunction")
    xv = _x(x, "SurvivalFunction")
    return _out(_one_minus(cdf_value(dd, xv)), d, x)


def _one_minus(v):
    if isinstance(v, sp.Piecewise):
        return sp.Piecewise(*((sp.simplify(1 - e), c) for e, c in v.args))
    return sp.simplify(1 - v)


@fn("HazardFunction", area=AREA, status="added",
    notes="PDF/SurvivalFunction; exact; numeric x outside the support gives 0")
def _hazard(d, x):
    dd = dist_of(d, "HazardFunction")
    xv = _x(x, "HazardFunction")
    if not (_num(xv) and dd.numeric()):
        if dd.discrete:
            raise Unevaluated()
        lo, hi = dd.bounds()
        c = dd.support_cond(xv)
        h = sp.simplify(dd.pdf(xv) / (1 - dd.cdf(xv)))
        return _out(h if c is sp.true else sp.Piecewise((h, c), (0, True)), d, x)
    p = pdf_value(dd, xv)
    if p == 0:
        return _out(sp.Integer(0), d, x)
    s = 1 - cdf_value(dd, xv) + (p if dd.discrete else 0)
    if s == 0:
        raise ValueError("HazardFunction: the survival function is 0 at this point")
    return _out(sp.simplify(p / s), d, x)


@fn("InverseSurvivalFunction", area=AREA, status="added", notes="InverseCDF[dist, 1 - q]")
def _isurv(d, q):
    dd = dist_of(d, "InverseSurvivalFunction")
    qv = sym(q, "InverseSurvivalFunction")
    return _out(quantile_value(dd, 1 - qv, (d, q), "InverseSurvivalFunction"), d, q)


@fn("CharacteristicFunction", area=AREA, status="added",
    notes="E[Exp[I t X]] in closed form (exact); distributions without an elementary closed form stay unevaluated")
def _cf(d, t):
    dd = dist_of(d, "CharacteristicFunction")
    return _out(_simp(cf_value(dd, sym(t, "CharacteristicFunction"))), d, t)


@fn("MomentGeneratingFunction", area=AREA, status="added",
    notes="E[Exp[t X]] in closed form (exact); distributions without one (Cauchy, LogNormal, ...) stay unevaluated")
def _mgf(d, t):
    dd = dist_of(d, "MomentGeneratingFunction")
    return _out(_simp(mgf_value(dd, sym(t, "MomentGeneratingFunction"))), d, t)


@fn("CentralMomentGeneratingFunction", area=AREA, status="added",
    notes="E[Exp[t (X - Mean)]] = Exp[-t mean] MomentGeneratingFunction")
def _cmgf(d, t):
    dd = dist_of(d, "CentralMomentGeneratingFunction")
    tv = sym(t, "CentralMomentGeneratingFunction")
    return _out(_simp(sp.exp(-tv * dd.mean()) * mgf_value(dd, tv)), d, t)


@fn("CumulantGeneratingFunction", area=AREA, status="added", notes="Log[MomentGeneratingFunction]")
def _cgf(d, t):
    dd = dist_of(d, "CumulantGeneratingFunction")
    m = _simp(mgf_value(dd, sym(t, "CumulantGeneratingFunction")), dd)
    v = m.args[0] if isinstance(m, sp.exp) else sp.log(m)
    if isinstance(v, sp.Add) and any(isinstance(a, sp.exp) for a in v.args) is False:
        v = sp.expand(v)
    return _out(_simp(v, dd), d, t)


# ─────────────────────────── descriptive statistics on data ───────────────────────────

def _data(x, fname, min_len=1):
    items = list_arg(x, fname, "data", min_len=min_len)
    if any(is_list(v) for v in items):
        raise NotImplementedError(f"{fname}: matrix data is not supported here")
    return [number(v, fname, "data value") for v in items], items


def _dout(v, items):
    if not any(has_float(i) for i in items):
        v = sp.simplify(v)
    return numeric_out(v, *items)


def _order_r(r, fname):
    if isinstance(r, bool) or not isinstance(r, int) or r < 0:
        if isinstance(r, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: the order must be a non-negative integer")
    return r


def _moment_data(x, r, central, fname):
    v, items = _data(x, fname)
    n = len(v)
    m = sp.Add(*v) / n
    return _dout(sp.Add(*(((t - m) if central else t) ** r for t in v)) / n, items)


@fn("Moment", area=AREA, status="added",
    notes="Moment[data, r] = Mean[data^r]; Moment[dist, r] the r-th raw moment (exact; Indeterminate when it does "
          "not exist)")
def _moment(x, r):
    r = _order_r(r, "Moment")
    if is_dist(x):
        if x.head == "TransformedDistribution":
            return _transformed_expect(x, lambda e: e ** r, "Moment")
        dd = dist_of(x, "Moment")
        v = raw_moment(dd, r) if r else sp.Integer(1)
        return INDET if v is sp.nan else _out(_simp(v), x)
    return _moment_data(x, r, False, "Moment")


@fn("CentralMoment", area=AREA, status="added",
    notes="CentralMoment[data, r] = Mean[(data - Mean[data])^r]; CentralMoment[dist, r] exact")
def _cmoment(x, r):
    r = _order_r(r, "CentralMoment")
    if is_dist(x):
        dd = dist_of(x, "CentralMoment")
        v = central_moment(dd, r) if r else sp.Integer(1)
        return INDET if v is sp.nan else _out(_simp(v), x)
    return _moment_data(x, r, True, "CentralMoment")


@fn("Cumulant", area=AREA, status="added",
    notes="Cumulant[dist, r] from the raw moments (exact); Cumulant[data, r] from the sample (biased) moments "
          "(Wolfram's moment-based definition: k1 = mean, k2 = second central moment, ...)")
def _cumulant(x, r):
    r = _order_r(r, "Cumulant")
    if r == 0:
        raise ValueError("Cumulant: the order must be a positive integer")
    if is_dist(x):
        dd = dist_of(x, "Cumulant")
        v = cumulant(dd, r)
        return INDET if v is sp.nan else _out(_simp(v), x)
    v, items = _data(x, "Cumulant")
    n = len(v)
    raws = [sp.Integer(1)] + [sp.Add(*(t ** k for t in v)) / n for k in range(1, r + 1)]
    kap = [sp.Integer(0)] * (r + 1)
    for m in range(1, r + 1):
        kap[m] = raws[m] - sp.Add(*(sp.binomial(m - 1, j - 1) * kap[j] * raws[m - j] for j in range(1, m)))
    return _dout(kap[r], items)


def _quantile_wolfram(v, q, params):
    from .statistics import _quantile
    return _quantile(v, q, params)


_QPARAMS = ((sp.Rational(1, 2), 0), (0, 1))


@fn("Quartiles", area=AREA, status="added",
    notes="Quantile[data, {1/4, 1/2, 3/4}, {{1/2, 0}, {0, 1}}] (Wolfram's definition); Quartiles[dist] = InverseCDF "
          "at 1/4, 1/2, 3/4")
def _quartiles(x):
    if is_dist(x):
        dd = dist_of(x, "Quartiles")
        return L(*(_out(quantile_value(dd, q, (x,), "Quartiles"), x) for q in
                   (sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4))))
    v, items = _data(x, "Quartiles")
    return L(*(numeric_out(_quantile_wolfram(v, q, _QPARAMS), *items) for q in
               (sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4))))


@fn("InterquartileRange", area=AREA, status="added",
    notes="Q3 - Q1 with the Quartiles parameters {{1/2, 0}, {0, 1}}; InterquartileRange[dist] from the quantiles")
def _iqr(x):
    if is_dist(x):
        dd = dist_of(x, "InterquartileRange")
        return _out(_simp(quantile_value(dd, sp.Rational(3, 4), (x,), "InterquartileRange")
                          - quantile_value(dd, sp.Rational(1, 4), (x,), "InterquartileRange")), x)
    v, items = _data(x, "InterquartileRange")
    return numeric_out(_quantile_wolfram(v, sp.Rational(3, 4), _QPARAMS) -
                       _quantile_wolfram(v, sp.Rational(1, 4), _QPARAMS), *items)


@fn("QuartileDeviation", area=AREA, status="added", notes="InterquartileRange/2")
def _qdev(x):
    r = _iqr(x)
    return numeric_out(S(r) / 2, x) if not is_dist(x) else _out(S(r) / 2, x)


@fn("QuartileSkewness", area=AREA, status="added", notes="(Q1 - 2 Q2 + Q3)/(Q3 - Q1) with the Quartiles parameters")
def _qskew(x):
    q = [S(a) for a in _quartiles(x).args]
    if q[2] == q[0]:
        raise ValueError("QuartileSkewness: the interquartile range is zero")
    return numeric_out(sp.simplify((q[0] - 2 * q[1] + q[2]) / (q[2] - q[0])), x)


@fn("MeanDeviation", area=AREA, status="added", notes="Mean[Abs[data - Mean[data]]], exact")
def _meandev(x):
    if _matrix_first((x,)):
        return _columns(x, _meandev)
    v, items = _data(x, "MeanDeviation")
    m = sp.Add(*v) / len(v)
    return _dout(sp.Add(*(sp.Abs(t - m) for t in v)) / len(v), items)


@fn("MedianDeviation", area=AREA, status="added", notes="Median[Abs[data - Median[data]]], exact")
def _meddev(x):
    if _matrix_first((x,)):
        return _columns(x, _meddev)
    from .statistics import _median
    v, items = _data(x, "MedianDeviation")
    m = _median(v)
    return _dout(_median([sp.Abs(t - m) for t in v]), items)


@fn("RootMeanSquare", area=AREA, status="added", notes="Sqrt[Mean[data^2]] (Abs[x]^2 for complex), exact")
def _rms(x):
    if _matrix_first((x,)):
        return _columns(x, _rms)
    items = list_arg(x, "RootMeanSquare", "data", 1)
    v = [number(t, "RootMeanSquare", "data value", real=False) for t in items]
    return _dout(sp.sqrt(sp.Add(*(sp.Abs(t) ** 2 for t in v)) / len(v)), items)


@fn("ContraharmonicMean", area=AREA, status="added", notes="Total[data^2]/Total[data], exact; zero total raises")
def _chm(x):
    v, items = _data(x, "ContraharmonicMean")
    s = sp.Add(*v)
    if s == 0:
        raise ValueError("ContraharmonicMean: the data sum to zero")
    return _dout(sp.Add(*(t ** 2 for t in v)) / s, items)


def _winsor(v, f):
    n = len(v)
    k = int(sp.floor(f * n))
    s = sorted(v)
    if k == 0:
        return s
    return [s[k]] * k + s[k:n - k] + [s[n - k - 1]] * k


def _frac(f, fname):
    ff = number(f, fname, "fraction")
    if not 0 <= ff < sp.Rational(1, 2):
        raise ValueError(f"{fname}: the fraction must be in [0, 1/2)")
    return ff


@fn("WinsorizedMean", area=AREA, status="added",
    notes="mean after replacing the Floor[f n] smallest and largest values by the nearest remaining values "
          "(default f = 1/5); exact")
def _wmean(x, f=None):
    v, items = _data(x, "WinsorizedMean")
    ff = sp.Rational(1, 5) if f is None else _frac(f, "WinsorizedMean")
    w = _winsor(v, ff)
    return _dout(sp.Add(*w) / len(w), items + ([f] if f is not None else []))


@fn("WinsorizedVariance", area=AREA, status="added",
    notes="unbiased variance of the winsorised data (default f = 1/5); exact")
def _wvar(x, f=None):
    v, items = _data(x, "WinsorizedVariance", 2)
    ff = sp.Rational(1, 5) if f is None else _frac(f, "WinsorizedVariance")
    w = _winsor(v, ff)
    m = sp.Add(*w) / len(w)
    return _dout(sp.Add(*((t - m) ** 2 for t in w)) / (len(w) - 1), items + ([f] if f is not None else []))


@fn("TrimmedVariance", area=AREA, status="added",
    notes="unbiased variance after dropping Floor[f n] values from each end (default f = 1/5); exact")
def _tvar(x, f=None):
    v, items = _data(x, "TrimmedVariance", 2)
    ff = sp.Rational(1, 5) if f is None else _frac(f, "TrimmedVariance")
    k = int(sp.floor(ff * len(v)))
    s = sorted(v)[k:len(v) - k]
    if len(s) < 2:
        raise ValueError("TrimmedVariance: fewer than two values remain")
    m = sp.Add(*s) / len(s)
    return _dout(sp.Add(*((t - m) ** 2 for t in s)) / (len(s) - 1), items + ([f] if f is not None else []))


@fn("Standardize", area=AREA, status="added",
    notes="(data - Mean)/StandardDeviation; Standardize[data, f1, f2] with location and scale functions; exact")
def _standardize(x, f1=None, f2=None):
    from .wl_common import apply
    items = list_arg(x, "Standardize", "data", 2)
    if any(is_list(i) for i in items):
        raise NotImplementedError("Standardize: matrix data is not supported")
    loc = apply(f1 if f1 is not None else Symbol("Mean"), x)
    sc = apply(f2 if f2 is not None else Symbol("StandardDeviation"), x)
    sv = S(sc)
    if not (isinstance(sv, sp.Basic) and sv.is_number):
        raise ValueError("Standardize: the scale must be numeric")
    if sv == 0:
        raise ValueError("Standardize: the scale is zero")
    lv = S(loc)
    return L(*(numeric_out((number(t, "Standardize") - lv) / sv, t, loc, sc) for t in items))


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [None] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = sp.Rational(i + j + 2, 2)
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def _pair(x, y, fname):
    a, ia = _data(x, fname, 2)
    b, ib = _data(y, fname, 2)
    if len(a) != len(b):
        raise ValueError(f"{fname}: the lists must have the same length")
    return a, b, ia + ib


@fn("SpearmanRho", area=AREA, status="added",
    notes="Pearson correlation of the (average) ranks, exact; SpearmanRho[matrix] the rank-correlation matrix")
def _spearman(x, y=None):
    if y is None:
        return _matrix_pairs(x, _spearman, "SpearmanRho")
    a, b, items = _pair(x, y, "SpearmanRho")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sp.Add(*ra) / len(ra), sp.Add(*rb) / len(rb)
    num = sp.Add(*((p - ma) * (q - mb) for p, q in zip(ra, rb)))
    den = sp.sqrt(sp.Add(*((p - ma) ** 2 for p in ra)) * sp.Add(*((q - mb) ** 2 for q in rb)))
    if den == 0:
        raise ValueError("SpearmanRho: a list is constant")
    return _dout(num / den, items)


@fn("KendallTau", area=AREA, status="added",
    notes="Kendall tau-b (ties corrected), exact; KendallTau[matrix] the matrix of coefficients")
def _kendall(x, y=None):
    if y is None:
        return _matrix_pairs(x, _kendall, "KendallTau")
    a, b, items = _pair(x, y, "KendallTau")
    n = len(a)
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = sp.sign(a[i] - a[j]), sp.sign(b[i] - b[j])
            if da == 0 and db == 0:
                continue
            if da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da == db:
                conc += 1
            else:
                disc += 1
    den = sp.sqrt((conc + disc + ta) * (conc + disc + tb))
    if den == 0:
        raise ValueError("KendallTau: a list is constant")
    return _dout(sp.Integer(conc - disc) / den, items)


def _matrix_pairs(m, f, fname):
    rows = list_arg(m, fname, "matrix", 2)
    if not all(is_list(r) for r in rows) or len({len(r.args) for r in rows}) != 1:
        raise ValueError(f"{fname}: expected two lists or a matrix")
    cols = [L(*(r.args[j] for r in rows)) for j in range(len(rows[0].args))]
    return L(*(L(*(f(ci, cj) for cj in cols)) for ci in cols))


def _columns(m, f):
    rows = list(m.args)
    if len({len(r.args) for r in rows}) != 1:
        raise ValueError("matrix data must be rectangular")
    return L(*(f(L(*(r.args[j] for r in rows))) for j in range(len(rows[0].args))))


def _colwise(name):
    e = REGISTRY[name]
    old = list(e.impls)

    def f(*args):
        if _matrix_first(args):
            return _columns(args[0], lambda col: _call_old(old, col))
        return _call_old(old, *args)

    def _call_old(old, *a):
        for g, lo, hi in old:
            if len(a) >= lo and (hi is None or len(a) <= hi):
                return g(*a)
        raise ValueError(f"{name} called with {len(a)} argument(s)")
    e.impls = [(f, 0, None)]
    e.notes = "matrix data: the statistic of each column (Wolfram); " + e.notes
    record(name, e.area, "fixed", e.notes)


for _n in ("Mean", "Median", "Variance", "StandardDeviation", "Skewness", "Kurtosis", "GeometricMean",
           "HarmonicMean"):
    _colwise(_n)


def _covariance_matrix(m, fname, corr=False):
    rows = list_arg(m, fname, "matrix", 2)
    if not all(is_list(r) for r in rows) or len({len(r.args) for r in rows}) != 1:
        raise ValueError(f"{fname}: expected a matrix")
    cols = [[number(r.args[j], fname, "data value") for r in rows] for j in range(len(rows[0].args))]
    n = len(rows)
    means = [sp.Add(*c) / n for c in cols]
    cov = [[sp.Add(*((ci[k] - mi) * (cj[k] - mj) for k in range(n))) / (n - 1)
            for cj, mj in zip(cols, means)] for ci, mi in zip(cols, means)]
    if corr:
        d = [sp.sqrt(cov[i][i]) for i in range(len(cols))]
        if any(x == 0 for x in d):
            raise ValueError(f"{fname}: a column is constant")
        cov = [[cov[i][j] / (d[i] * d[j]) for j in range(len(cols))] for i in range(len(cols))]
    inputs = [x for r in rows for x in r.args]
    return L(*(L(*(_dout(c, inputs) for c in row)) for row in cov))


extend("Covariance", lambda m: _covariance_matrix(m, "Covariance"), _matrix_first,
       "Covariance[matrix] = covariance matrix of the columns (unbiased), exact")
extend("Correlation", lambda m: _covariance_matrix(m, "Correlation", True), _matrix_first,
       "Correlation[matrix] = correlation matrix of the columns, exact")


# ─────────────────────────── multivariate distributions ───────────────────────────

def _mvn(d, fname):
    args = list(d.args)
    if len(args) == 1:
        sig = args[0]
        mu = None
    elif len(args) == 2:
        mu, sig = args
    else:
        raise ValueError("MultinormalDistribution expects [mu, Sigma] or [Sigma]")
    import sympy as _sp
    if not is_list(sig) or not all(is_list(r) for r in sig.args):
        raise ValueError("MultinormalDistribution: Sigma must be a matrix")
    S_ = _sp.Matrix([[sym(x, fname) for x in r.args] for r in sig.args])
    if S_.rows != S_.cols:
        raise ValueError("MultinormalDistribution: Sigma must be square")
    if mu is None:
        m = _sp.zeros(S_.rows, 1)
    else:
        if not is_list(mu) or len(mu.args) != S_.rows:
            raise ValueError("MultinormalDistribution: mu and Sigma dimensions differ")
        m = _sp.Matrix([sym(x, fname) for x in mu.args])
    if S_.is_symmetric() is False:
        raise ValueError("MultinormalDistribution: Sigma must be symmetric")
    if all(x.is_number for x in S_):
        try:
            if not all(ev_.is_positive for ev_ in S_.eigenvals()):
                raise ValueError("MultinormalDistribution: Sigma must be positive definite")
        except ValueError:
            raise
    return m, S_


def _multinomial(d, fname):
    if len(d.args) != 2 or not is_list(d.args[1]):
        raise ValueError("MultinomialDistribution expects [n, {p1, ..., pk}]")
    n = sym(d.args[0], fname)
    ps = [sym(x, fname) for x in d.args[1].args]
    _nonneg_int_param(n, "n", "MultinomialDistribution")
    if all(p.is_number for p in ps):
        if any(p < 0 for p in ps) or sp.simplify(sp.Add(*ps) - 1) != 0:
            raise ValueError("MultinomialDistribution: probabilities must be non-negative and sum to 1")
    return n, ps


@fn("MultinormalDistribution", area=AREA, status="added",
    notes="MultinormalDistribution[mu, Sigma] (and [Sigma] with zero mean): PDF, CDF (numeric, SciPy), Mean, "
          "Variance (diagonal), Covariance, RandomVariate; Sigma must be symmetric positive definite")
def _mvn_ctor(*args):
    _mvn(Expr("MultinormalDistribution", *args), "MultinormalDistribution")
    raise Unevaluated()


@fn("MultinomialDistribution", area=AREA, status="added",
    notes="MultinomialDistribution[n, {p1, ...}]: PDF, CDF, Mean, Variance, Covariance, RandomVariate")
def _mult_ctor(*args):
    _multinomial(Expr("MultinomialDistribution", *args), "MultinomialDistribution")
    raise Unevaluated()


def _vec(x, k, fname):
    if not is_list(x) or len(x.args) != k:
        raise ValueError(f"{fname}: expected a point with {k} coordinates")
    return [sym(v, fname) for v in x.args]


def _multi_pdf(d, x):
    if d.head == "MultinormalDistribution":
        m, S_ = _mvn(d, "PDF")
        xv = sp.Matrix(_vec(x, m.rows, "PDF"))
        k = m.rows
        diff = xv - m
        val = sp.exp(-(diff.T * S_.inv() * diff)[0] / 2) / sp.sqrt((2 * sp.pi) ** k * S_.det())
        return _out(sp.simplify(val), d, x)
    n, ps = _multinomial(d, "PDF")
    ks = _vec(x, len(ps), "PDF")
    if all(k.is_number for k in ks) and n.is_number:
        if not all(k.is_integer and k >= 0 for k in ks) or sp.Add(*ks) != n:
            return _out(sp.Integer(0), d, x)
    val = sp.factorial(n) * sp.Mul(*(p ** k / sp.factorial(k) for p, k in zip(ps, ks)))
    return _out(val, d, x)


def _multi_cdf(d, x):
    if d.head == "MultinormalDistribution":
        m, S_ = _mvn(d, "CDF")
        xv = _vec(x, m.rows, "CDF")
        if not all(v.is_number for v in xv) or not all(v.is_number for v in S_) or not all(v.is_number for v in m):
            raise Unevaluated()
        if m.rows == 1:
            return _out(sp.erfc((m[0] - xv[0]) / sp.sqrt(2 * S_[0, 0])) / 2, d, x)
        try:
            import scipy.stats as st
        except ImportError:
            raise NotImplementedError("CDF of a MultinormalDistribution requires SciPy")
        v = st.multivariate_normal(mean=[float(a) for a in m], cov=[[float(S_[i, j]) for j in range(S_.cols)]
                                                                   for i in range(S_.rows)]).cdf([float(a) for a in xv])
        return float(v)
    n, ps = _multinomial(d, "CDF")
    ks = _vec(x, len(ps), "CDF")
    if not (n.is_number and all(k.is_number for k in ks)):
        raise Unevaluated()
    import itertools
    lim = [int(sp.floor(k)) for k in ks]
    tot = sp.Integer(0)
    for combo in itertools.product(*(range(0, max(-1, l) + 1) for l in lim)):
        if sum(combo) == n:
            tot += sp.factorial(n) * sp.Mul(*(p ** c / sp.factorial(c) for p, c in zip(ps, combo)))
    return _out(tot, d, x)


def _multi_moment(d, which):
    if d.head == "MultinormalDistribution":
        m, S_ = _mvn(d, which)
        if which == "Mean":
            return _out(sp.Matrix(m), d) if False else L(*(_out(v, d) for v in m))
        if which == "Variance":
            return L(*(_out(S_[i, i], d) for i in range(S_.rows)))
        if which == "StandardDeviation":
            return L(*(_out(sp.sqrt(S_[i, i]), d) for i in range(S_.rows)))
        if which == "Covariance":
            return L(*(L(*(_out(S_[i, j], d) for j in range(S_.cols))) for i in range(S_.rows)))
        if which == "Skewness":
            return L(*([0] * S_.rows))
        if which == "Kurtosis":
            return L(*([3] * S_.rows))
        raise Unevaluated()
    n, ps = _multinomial(d, which)
    if which == "Mean":
        return L(*(_out(n * p, d) for p in ps))
    if which == "Variance":
        return L(*(_out(n * p * (1 - p), d) for p in ps))
    if which == "StandardDeviation":
        return L(*(_out(sp.sqrt(n * p * (1 - p)), d) for p in ps))
    if which == "Covariance":
        return L(*(L(*(_out(n * p * (1 - p) if i == j else -n * p * q, d) for j, q in enumerate(ps)))
                   for i, p in enumerate(ps)))
    raise Unevaluated()


extend("Covariance", lambda d: _multi_moment(d, "Covariance"),
       lambda a: len(a) == 1 and isinstance(a[0], Expr) and a[0].head in MULTI,
       "Covariance[dist] = covariance matrix of a multivariate distribution")


# ─────────────────────────── random numbers ───────────────────────────

def _dims(spec, fname):
    if spec is None:
        return None
    if isinstance(spec, int) and not isinstance(spec, bool) and spec >= 0:
        return [spec]
    if is_list(spec) and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in spec.args):
        return list(spec.args)
    raise ValueError(f"{fname}: the count must be a non-negative integer or a list of dimensions")


def _fill(dims, one):
    if not dims:
        return one()
    return L(*(_fill(dims[1:], one) for _ in range(dims[0])))


def _rv(d, n=None):
    rng = _rng()
    if d.head == "MultinormalDistribution":
        m, S_ = _mvn(d, "RandomVariate")
        if not (all(v.is_number for v in m) and all(v.is_number for v in S_)):
            raise ValueError("RandomVariate: distribution parameters must be numeric")
        Lc = S_.cholesky(hermitian=False)
        k = m.rows

        def one():
            z = sp.Matrix([rng.gauss(0, 1) for _ in range(k)])
            v = m + Lc * z
            return L(*(float(c) for c in v))
    elif d.head == "MultinomialDistribution":
        nn, ps = _multinomial(d, "RandomVariate")
        if not (nn.is_number and all(p.is_number for p in ps)):
            raise ValueError("RandomVariate: distribution parameters must be numeric")
        fp = [float(p) for p in ps]

        def one():
            counts = [0] * len(fp)
            for _ in range(int(nn)):
                u, acc = rng.random(), 0.0
                for i, p in enumerate(fp):
                    acc += p
                    if u < acc:
                        counts[i] += 1
                        break
                else:
                    counts[-1] += 1
            return L(*counts)
    elif d.head == "TransformedDistribution":
        f, var, base = _transformed(d, "RandomVariate")
        bd = dist_of(base, "RandomVariate")
        from ..math_ops import from_sympy

        def one():
            x = bd.sample(rng)
            return R(sp.N(f.subs(var, x)) if isinstance(x, float) else f.subs(var, x))
    else:
        dd = dist_of(d, "RandomVariate")
        if not dd.numeric():
            raise ValueError("RandomVariate: distribution parameters must be numeric")

        def one():
            v = dd.sample(rng)
            return R(v) if not isinstance(v, (int, float)) else v
    dims = _dims(n, "RandomVariate")
    return _fill(dims, one) if dims is not None else one()


extend("RandomVariate", _rv, _new_dist_first,
       "batch 5 distributions (including multivariate, empirical and transformed ones); RandomVariate[d, {n, m}] "
       "arrays; reproducible with SeedRandom (the random stream differs from Wolfram's)")


def _range_spec(spec, fname, integer):
    if spec is None:
        return (0, 1)
    if is_list(spec) and len(spec.args) == 2:
        a, b = spec.args
    else:
        a, b = 0, spec
    if integer:
        a, b = int_arg(a, fname, "bound"), int_arg(b, fname, "bound")
        if a > b:
            raise ValueError(f"{fname}: empty range")
        return a, b
    av, bv = number(a, fname, "bound"), number(b, fname, "bound")
    return float(av), float(bv)


@fn("RandomInteger", area=AREA, status="added",
    notes="RandomInteger[] (0 or 1), [imax] (0..imax), [{imin, imax}], with a count n or dimensions {n, m}; "
          "reproducible with SeedRandom (stream differs from Wolfram's)")
def _rint(spec=None, n=None):
    a, b = (0, 1) if spec is None else _range_spec(spec, "RandomInteger", True)
    rng = _rng()
    dims = _dims(n, "RandomInteger")
    one = lambda: rng.randint(a, b)  # noqa: E731
    return _fill(dims, one) if dims is not None else one()


@fn("RandomReal", area=AREA, status="added",
    notes="RandomReal[] in [0, 1), [xmax], [{xmin, xmax}], with a count n or dimensions; machine reals; "
          "reproducible with SeedRandom")
def _rreal(spec=None, n=None, *opts):
    if opts:
        raise ValueError("RandomReal: options such as WorkingPrecision are not supported")
    a, b = _range_spec(spec, "RandomReal", False)
    rng = _rng()
    dims = _dims(n, "RandomReal")
    one = lambda: a + (b - a) * rng.random()  # noqa: E731
    return _fill(dims, one) if dims is not None else one()


@fn("RandomChoice", area=AREA, status="added",
    notes="RandomChoice[list], [list, n] (with replacement), [{w...} -> list, n] weighted; reproducible with SeedRandom")
def _rchoice(x, n=None):
    rng = _rng()
    if isinstance(x, Expr) and x.head == "Rule":
        ws = [float(number(w, "RandomChoice", "weight")) for w in list_arg(x.args[0], "RandomChoice", "weights", 1)]
        items = list_arg(x.args[1], "RandomChoice", "list", 1)
        if len(ws) != len(items) or any(w < 0 for w in ws) or sum(ws) <= 0:
            raise ValueError("RandomChoice: weights must be non-negative, one per element, with a positive total")
        one = lambda: rng.choices(items, weights=ws)[0]  # noqa: E731
    else:
        items = list_arg(x, "RandomChoice", "list", 1)
        one = lambda: rng.choice(items)  # noqa: E731
    dims = _dims(n, "RandomChoice")
    return _fill(dims, one) if dims is not None else one()


# ─────────────────────────── Expectation and Probability ───────────────────────────

def _distributed(spec, fname):
    """[(sympy var, dist expr)] from x \\[Distributed] d or a list of them."""
    specs = list(spec.args) if is_list(spec) else [spec]
    out = []
    for s in specs:
        if not (isinstance(s, Expr) and s.head == "Distributed" and len(s.args) == 2):
            raise ValueError(f"{fname}: expected x \\[Distributed] dist (Distributed[x, dist])")
        v, d = s.args
        if not isinstance(v, Symbol):
            raise NotImplementedError(f"{fname}: only scalar random variables are supported")
        if not (isinstance(d, Expr) and isinstance(d.head, str) and (d.head in TABLE or d.head == "EmpiricalDistribution")):
            if isinstance(d, Expr) and isinstance(d.head, str) and d.head.endswith("Distribution"):
                raise NotImplementedError(f"{fname}: {d.head} is not supported")
            raise ValueError(f"{fname}: {d} is not a distribution")
        out.append((sp.Symbol(v.name), d))
    return out


def _expect(f, pairs, fname, numeric=False):
    if not pairs:
        return f
    (v, d), rest = pairs[0], pairs[1:]
    dd = dist_of(d, fname)
    inner = _expect(f, rest, fname, numeric) if rest else f
    if isinstance(dd, Empirical):
        return sp.Add(*(dd.w[p] * inner.subs(v, p) for p in dd.pts))
    lo, hi = dd.bounds()
    if dd.discrete:
        k = sp.Dummy("k", integer=True)
        if hi != sp.oo and dd.numeric():
            return sp.Add(*(inner.subs(v, sp.Integer(j)) * dd.pdf(sp.Integer(j)) for j in range(int(lo), int(hi) + 1)))
        if numeric:
            import mpmath
            g = sp.lambdify(k, inner.subs(v, k) * dd.pdf(k), "mpmath")
            return sp.Float(mpmath.nsum(g, [int(lo), mpmath.inf]))
        s = sp.summation(inner.subs(v, k) * dd.pdf(k), (k, lo, hi))
        if s.has(sp.Sum):
            raise Unevaluated()
        return s
    x = sp.Dummy("x", real=True)
    integrand = inner.subs(v, x) * dd.pdf(x)
    if numeric:
        import mpmath
        g = sp.lambdify(x, integrand, "mpmath")
        return sp.Float(mpmath.quad(g, [mpmath.mpf(float(lo)) if lo != -sp.oo else -mpmath.inf,
                                        mpmath.mpf(float(hi)) if hi != sp.oo else mpmath.inf]))
    r = sp.integrate(integrand, (x, lo, hi), conds="none")
    if r.has(sp.Integral):
        raise Unevaluated()
    return r


@fn("Distributed", area=AREA, status="added", notes="x \\[Distributed] dist, the inert random-variable declaration")
def _distr(x, d):
    raise Unevaluated()


def _cond_split(f, fname):
    if isinstance(f, Expr) and f.head == "Conditioned":
        raise NotImplementedError(f"{fname}: conditioned expectations are not supported")
    return f


@fn("Expectation", area=AREA, status="added",
    notes="Expectation[f, x \\[Distributed] dist] (and lists of independent variables) by exact integration or "
          "summation (SymPy); expectations SymPy cannot do in closed form stay unevaluated")
def _expectation(f, spec):
    _cond_split(f, "Expectation")
    pairs = _distributed(spec, "Expectation")
    fv = sym(f, "Expectation")
    r = _expect(fv, pairs, "Expectation")
    return _out(sp.simplify(r), f, *[d for _, d in pairs])


@fn("NExpectation", area=AREA, status="added",
    notes="numerical expectation (mpmath quadrature / summation) of f over x \\[Distributed] dist, machine reals")
def _nexpectation(f, spec):
    pairs = _distributed(spec, "NExpectation")
    if len(pairs) != 1:
        raise NotImplementedError("NExpectation: only one random variable is supported")
    fv = sym(f, "NExpectation")
    return float(sp.N(_expect(fv, pairs, "NExpectation", numeric=True)))


def _event_set(pred, v, fname):
    from sympy.solvers.inequalities import solve_univariate_inequality
    if isinstance(pred, sp.And):
        return sp.Intersection(*(_event_set(p, v, fname) for p in pred.args))
    if isinstance(pred, sp.Or):
        return sp.Union(*(_event_set(p, v, fname) for p in pred.args))
    if isinstance(pred, sp.Not):
        return sp.Complement(sp.S.Reals, _event_set(pred.args[0], v, fname))
    try:
        if isinstance(pred, sp.Eq):
            return sp.solveset(pred, v, sp.S.Reals)
        return solve_univariate_inequality(pred, v, relational=False, domain=sp.S.Reals)
    except Exception as exc:
        raise NotImplementedError(f"{fname}: cannot solve the event {pred} ({exc})")


def _prob_set(dd, st):
    if isinstance(st, sp.Union):
        return sp.Add(*(_prob_set(dd, s) for s in st.args))
    if st is sp.S.EmptySet:
        return sp.Integer(0)
    if isinstance(st, sp.FiniteSet):
        if not dd.discrete:
            return sp.Integer(0)
        return sp.Add(*(pdf_value(dd, p) for p in st))
    if isinstance(st, sp.Interval):
        a, b = st.start, st.end
        if dd.discrete:
            if isinstance(dd, Empirical):
                return sp.Add(*(dd.w[p] for p in dd.pts if p in st))
            kmin = None if a == -sp.oo else (a if (a.is_integer and not st.left_open) else sp.floor(a) + 1)
            kmax = None if b == sp.oo else (b if (b.is_integer and not st.right_open) else sp.ceiling(b) - 1)
            hi_v = sp.Integer(1) if kmax is None else cdf_value(dd, kmax)
            lo_v = sp.Integer(0) if kmin is None else cdf_value(dd, kmin - 1)
            return sp.Max(hi_v - lo_v, 0)
        hi_v = cdf_value(dd, b) if b != sp.oo else sp.Integer(1)
        lo_v = cdf_value(dd, a) if a != -sp.oo else sp.Integer(0)
        return hi_v - lo_v
    raise NotImplementedError(f"Probability: unsupported event set {st}")


_REL = {"Greater": sp.Gt, "GreaterEqual": sp.Ge, "Less": sp.Lt, "LessEqual": sp.Le, "Equal": sp.Eq,
        "Unequal": sp.Ne}


def _pred(p, fname):
    """SymPy boolean for an event built from comparisons, And, Or and Not."""
    if p is True or (isinstance(p, Symbol) and p.name == "True"):
        return sp.true
    if p is False or (isinstance(p, Symbol) and p.name == "False"):
        return sp.false
    if isinstance(p, Expr) and isinstance(p.head, str):
        h = p.head
        if h in _REL and len(p.args) >= 2:
            a = [sym(x, fname) for x in p.args]
            return sp.And(*(_REL[h](x, y) for x, y in zip(a, a[1:])))
        if h == "Inequality":
            a = p.args
            parts = []
            for i in range(0, len(a) - 2, 2):
                parts.append(_REL[a[i + 1].name](sym(a[i], fname), sym(a[i + 2], fname)))
            return sp.And(*parts)
        if h == "And":
            return sp.And(*(_pred(x, fname) for x in p.args))
        if h == "Or":
            return sp.Or(*(_pred(x, fname) for x in p.args))
        if h == "Not" and len(p.args) == 1:
            return sp.Not(_pred(p.args[0], fname))
    raise NotImplementedError(f"{fname}: events must be built from comparisons, And, Or and Not")


@fn("Probability", area=AREA, status="added",
    notes="Probability[pred, x \\[Distributed] dist] for a univariate event (inequalities, equalities and their "
          "And/Or combinations, solved over the reals) via the exact CDF; other events use exact expectation of "
          "Boole[pred]")
def _probability(pred, spec):
    pairs = _distributed(spec, "Probability")
    if len(pairs) != 1:
        raise NotImplementedError("Probability: only one random variable is supported")
    (v, d), = pairs
    dd = dist_of(d, "Probability")
    pv = _pred(pred, "Probability")
    if pv in (sp.true, True):
        return 1
    if pv in (sp.false, False):
        return 0
    if pv.free_symbols - {v}:
        raise Unevaluated()
    if not dd.numeric():
        raise Unevaluated()
    st = _event_set(pv, v, "Probability")
    return _out(sp.simplify(_prob_set(dd, st)), pred, d)


@fn("NProbability", area=AREA, status="added", notes="N[Probability[...]] (machine real)")
def _nprob(pred, spec):
    r = _probability(pred, spec)
    return float(sp.N(S(r)))


# ─────────────────────────── TransformedDistribution ───────────────────────────

def _transformed(d, fname):
    if len(d.args) != 2:
        raise ValueError("TransformedDistribution expects [expr, x \\[Distributed] dist]")
    pairs = _distributed(d.args[1], fname)
    if len(pairs) != 1:
        raise NotImplementedError("TransformedDistribution: only one random variable is supported")
    (v, base), = pairs
    return sym(d.args[0], fname), v, base


@fn("TransformedDistribution", area=AREA, status="added",
    notes="inert TransformedDistribution[f[x], x \\[Distributed] dist]; Mean, Variance, StandardDeviation, Moment, "
          "Expectation-based functions and RandomVariate (no simplification to a named distribution)")
def _td(*args):
    _transformed(Expr("TransformedDistribution", *args), "TransformedDistribution")
    raise Unevaluated()


def _transformed_expect(d, g, fname):
    f, v, base = _transformed(d, fname)
    r = _expect(g(f), [(v, base)], fname)
    return _out(sp.simplify(r), d)


def _transformed_moment(d, which):
    f, v, base = _transformed(d, which)
    m1 = _expect(f, [(v, base)], which)
    if which == "Mean":
        return _out(sp.simplify(m1), d)
    m2 = _expect(f ** 2, [(v, base)], which)
    var = sp.simplify(m2 - m1 ** 2)
    if which == "Variance":
        return _out(var, d)
    if which == "StandardDeviation":
        return _out(sp.sqrt(var), d)
    raise Unevaluated()


for _w in ("Mean", "Variance", "StandardDeviation"):
    pass


# ─────────────────────────── estimation and tests ───────────────────────────

def _symbols_of(d, fname):
    syms = []
    for a in d.args:
        for x in (a.args if is_list(a) else [a]):
            if not isinstance(x, Symbol):
                raise ValueError(f"{fname}: the distribution parameters must be symbols to estimate")
            syms.append(x)
    return syms


def _mle(data, d, fname):
    """Closed-form maximum-likelihood estimates (exact for exact data)."""
    v, items = _data(data, fname, 1)
    n = len(v)
    h = d.head
    syms = _symbols_of(d, fname)
    mean = sp.Add(*v) / n
    if h == "NormalDistribution":
        var = sp.Add(*((x - mean) ** 2 for x in v)) / n
        vals = [mean, sp.sqrt(var)]
    elif h == "ExponentialDistribution":
        vals = [1 / mean]
    elif h == "PoissonDistribution":
        vals = [mean]
    elif h == "BernoulliDistribution":
        vals = [mean]
    elif h == "GeometricDistribution":
        vals = [1 / (1 + mean)]
    elif h == "UniformDistribution":
        vals = [min(v), max(v)]
    elif h == "LogNormalDistribution":
        if any(x <= 0 for x in v):
            raise ValueError(f"{fname}: LogNormalDistribution needs positive data")
        lg = [sp.log(x) for x in v]
        m = sp.Add(*lg) / n
        vals = [m, sp.sqrt(sp.Add(*((x - m) ** 2 for x in lg)) / n)]
    elif h == "LaplaceDistribution":
        from .statistics import _median
        med = _median(v)
        if n % 2 == 0 and sorted(v)[n // 2 - 1] != sorted(v)[n // 2]:
            raise ValueError(f"{fname}: the Laplace location estimate is not unique for this data")
        vals = [med, sp.Add(*(sp.Abs(x - med) for x in v)) / n]
    elif h == "RayleighDistribution":
        vals = [sp.sqrt(sp.Add(*(x ** 2 for x in v)) / (2 * n))]
    else:
        raise NotImplementedError(f"{fname}: maximum-likelihood estimation for {h} is not supported")
    if len(vals) != len(syms):
        raise ValueError(f"{fname}: {h} has {len(vals)} parameter(s)")
    return syms, [_dout(x, items) for x in vals]


@fn("FindDistributionParameters", area=AREA, status="added",
    notes="maximum-likelihood estimates {p1 -> v1, ...} in closed form for Normal, Exponential, Poisson, Bernoulli, "
          "Geometric, Uniform, LogNormal, Laplace and Rayleigh; exact for exact data; other families raise")
def _fdp(data, d):
    syms, vals = _mle(data, d, "FindDistributionParameters")
    return L(*(rule(s, v) for s, v in zip(syms, vals)))


@fn("EstimatedDistribution", area=AREA, status="added",
    notes="the distribution with the maximum-likelihood parameters (see FindDistributionParameters)")
def _ed(data, d):
    syms, vals = _mle(data, d, "EstimatedDistribution")
    sub = {s.name: v for s, v in zip(syms, vals)}

    def rep(a):
        if isinstance(a, Symbol) and a.name in sub:
            return sub[a.name]
        if is_list(a):
            return L(*(rep(x) for x in a.args))
        return a
    return Expr(d.head, *(rep(a) for a in d.args))


@fn("KolmogorovSmirnovTest", area=AREA, status="added",
    notes="KolmogorovSmirnovTest[data, dist] = two-sided p-value of the one-sample KS test against a fully specified "
          "continuous distribution (exact distribution of D, SciPy); [data, dist, \"TestStatistic\"] the statistic D "
          "(exact for exact data and closed-form CDFs); estimated parameters and the two-sample form raise")
def _ks(data, d=None, prop=None):
    if d is None:
        raise NotImplementedError("KolmogorovSmirnovTest: testing against an estimated distribution is not supported")
    v, items = _data(data, "KolmogorovSmirnovTest", 1)
    if is_list(d):
        raise NotImplementedError("KolmogorovSmirnovTest: the two-sample form is not supported")
    dd = dist_of(d, "KolmogorovSmirnovTest")
    if dd.discrete or not dd.numeric():
        raise NotImplementedError("KolmogorovSmirnovTest: needs a fully specified continuous distribution")
    s = sorted(v)
    n = len(s)
    Fs = [cdf_value(dd, x) for x in s]
    dstat = max(max(sp.Rational(i + 1, n) - F, F - sp.Rational(i, n)) for i, F in enumerate(Fs))
    if prop == "TestStatistic":
        return _dout(dstat, items) if all(_num(F) for F in Fs) else float(dstat)
    if prop not in (None, "PValue"):
        raise ValueError('KolmogorovSmirnovTest: supported properties are "PValue" and "TestStatistic"')
    try:
        import scipy.stats as st
    except ImportError:
        raise NotImplementedError("KolmogorovSmirnovTest p-values require SciPy")
    return float(st.kstwo.sf(float(dstat), n))


@fn("LocationTest", area=AREA, status="added",
    notes="LocationTest[data, mu0, \"TTest\"] or \"ZTest\"-free explicit tests only: two-sided one-sample t-test p-value; "
          "Wolfram's automatic test selection is not reproduced, so LocationTest without an explicit test raises")
def _loctest(data, mu0=None, test=None):
    if test is None:
        raise NotImplementedError('LocationTest: give the test explicitly, e.g. LocationTest[data, mu0, "T"]; '
                                  "automatic test selection is not supported")
    if test not in ("T", "TTest"):
        raise NotImplementedError(f"LocationTest: test {test} is not supported (use \"T\")")
    try:
        import scipy.stats as st
    except ImportError:
        raise NotImplementedError("LocationTest requires SciPy")
    v, _ = _data(data, "LocationTest", 2)
    m0 = 0.0 if mu0 is None else float(number(mu0, "LocationTest", "mu0"))
    return float(st.ttest_1samp([float(x) for x in v], m0).pvalue)


@fn("VarianceTest", area=AREA, status="added",
    notes="VarianceTest[{d1, d2}, r0, \"FisherRatio\"]: two-sided F-test p-value for Var1/Var2 = r0; "
          "VarianceTest[data, s0, \"ChiSquare\"]: two-sided one-sample chi-square p-value; automatic test "
          "selection raises")
def _vartest(data, r0=None, test=None):
    if test is None:
        raise NotImplementedError('VarianceTest: give the test explicitly ("FisherRatio" or "ChiSquare")')
    try:
        import scipy.stats as st
    except ImportError:
        raise NotImplementedError("VarianceTest requires SciPy")
    items = list_arg(data, "VarianceTest", "data", 2)
    if test == "FisherRatio":
        if not (len(items) == 2 and all(is_list(i) for i in items)):
            raise ValueError("VarianceTest: the Fisher ratio test needs {data1, data2}")
        a, _ = _data(items[0], "VarianceTest", 2)
        b, _ = _data(items[1], "VarianceTest", 2)
        va = float(sp.Add(*((x - sp.Add(*a) / len(a)) ** 2 for x in a)) / (len(a) - 1))
        vb = float(sp.Add(*((x - sp.Add(*b) / len(b)) ** 2 for x in b)) / (len(b) - 1))
        r = 1.0 if r0 is None else float(number(r0, "VarianceTest", "ratio"))
        f = va / vb / r
        c = st.f.cdf(f, len(a) - 1, len(b) - 1)
        return float(2 * min(c, 1 - c))
    if test == "ChiSquare":
        a, _ = _data(data, "VarianceTest", 2)
        s0 = 1.0 if r0 is None else float(number(r0, "VarianceTest", "sigma0^2"))
        va = float(sp.Add(*((x - sp.Add(*a) / len(a)) ** 2 for x in a)) / (len(a) - 1))
        c = st.chi2.cdf((len(a) - 1) * va / s0, len(a) - 1)
        return float(2 * min(c, 1 - c))
    raise NotImplementedError(f"VarianceTest: test {test} is not supported")


@fn("HistogramList", area=AREA, status="added",
    notes="HistogramList[data, {dx}] (bins aligned at multiples of dx covering the data), [data, {min, max, dx}] and "
          "[data, {{b1, b2, ...}}]: {bin edges, counts} with bins [b_i, b_(i+1)); automatic binning raises")
def _histlist(data, spec=None):
    v, items = _data(data, "HistogramList")
    if spec is None:
        raise NotImplementedError("HistogramList: automatic bin selection is not supported; give {dx}")
    if not is_list(spec):
        raise NotImplementedError("HistogramList: give a bin specification {dx}, {min, max, dx} or {{edges}}")
    a = spec.args
    if len(a) == 1 and is_list(a[0]):
        edges = [number(e, "HistogramList", "bin edge") for e in a[0].args]
        if any(e2 <= e1 for e1, e2 in zip(edges, edges[1:])) or len(edges) < 2:
            raise ValueError("HistogramList: bin edges must be increasing")
    elif len(a) == 1:
        dx = number(a[0], "HistogramList", "bin width")
        if dx <= 0:
            raise ValueError("HistogramList: the bin width must be positive")
        lo = sp.floor(min(v) / dx) * dx
        hi = (sp.floor(max(v) / dx) + 1) * dx
        k = int((hi - lo) / dx)
        edges = [lo + i * dx for i in range(k + 1)]
    elif len(a) == 3:
        lo, hi, dx = (number(e, "HistogramList", "bin specification") for e in a)
        if dx <= 0 or hi <= lo:
            raise ValueError("HistogramList: need min < max and dx > 0")
        k = int(sp.ceiling((hi - lo) / dx))
        edges = [lo + i * dx for i in range(k + 1)]
    else:
        raise ValueError("HistogramList: invalid bin specification")
    counts = [0] * (len(edges) - 1)
    for x in v:
        for i in range(len(edges) - 1):
            if edges[i] <= x < edges[i + 1]:
                counts[i] += 1
                break
    return L(L(*(numeric_out(e, *items, spec) for e in edges)), L(*counts))


def _student3(*args):
    dist_of(Expr("StudentTDistribution", *args), "StudentTDistribution")
    raise Unevaluated()


extend("StudentTDistribution", _student3, lambda a: len(a) == 3,
       "StudentTDistribution[mu, sigma, nu] (location-scale form): every distribution function via batch 5")
