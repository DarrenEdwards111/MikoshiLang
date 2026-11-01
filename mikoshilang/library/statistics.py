"""Descriptive statistics, distributions and hypothesis tests.

Descriptive statistics are exact for exact data (legacy converted to floats and
rounded to 10 digits).  Hypothesis tests are numeric (SciPy) and are no longer
rounded."""
import math
import random as _random

import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, S, R, L, rule, numeric_out, sym, number, is_list, int_arg,
                   nonneg_int, pos_int, list_arg, has_float, matrix_arg)

AREA = "statistics"
EXACT = "exact for exact data (legacy converted to floats and rounded to 10 digits)"
_RNG = _random.Random()


def _data(x, fname, min_len=1):
    items = list_arg(x, fname, "data", min_len=min_len)
    if any(is_list(v) for v in items):
        raise NotImplementedError(f"{fname}: matrix (column-wise) data is not supported; pass a vector")
    return [number(v, fname, "data value") for v in items], items


def _out(v, items):
    return numeric_out(sp.simplify(v) if not any(has_float(i) for i in items) else v, *items)


def _mean(v):
    return sp.Add(*v) / len(v)


def _var(v):
    m = _mean(v)
    return sp.Add(*((x - m) ** 2 for x in v)) / (len(v) - 1)


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


# ─────────────────────────── distributions ───────────────────────────

DISTS = {"NormalDistribution", "ExponentialDistribution", "PoissonDistribution", "UniformDistribution",
         "BinomialDistribution", "ChiSquaredDistribution", "TDistribution", "StudentTDistribution", "FDistribution"}


def _dist(d, fname):
    if not (isinstance(d, Expr) and d.head in DISTS):
        if isinstance(d, Expr) and isinstance(d.head, str) and d.head.endswith("Distribution"):
            raise NotImplementedError(f"{fname}: {d.head} is not supported")
        raise ValueError(f"{fname}: first argument must be a distribution")
    h = d.head
    if h == "UniformDistribution":
        if len(d.args) == 1 and is_list(d.args[0]):
            a = [sym(v, fname) for v in d.args[0].args]
        else:
            a = [sym(v, fname) for v in d.args]
        if len(d.args) == 0:
            a = [sp.Integer(0), sp.Integer(1)]
        if len(a) != 2:
            raise ValueError("UniformDistribution expects {min, max}")
        if all(v.is_number for v in a) and not a[0] < a[1]:
            raise ValueError("UniformDistribution: min must be less than max")
    elif h == "NormalDistribution":
        a = [sym(v, fname) for v in d.args]
        if len(a) == 0:
            a = [sp.Integer(0), sp.Integer(1)]
        if len(a) != 2:
            raise ValueError("NormalDistribution expects {mu, sigma}")
        if a[1].is_number and not a[1] > 0:
            raise ValueError("NormalDistribution: sigma must be positive")
    else:
        a = [sym(v, fname) for v in d.args]
        need = {"ExponentialDistribution": 1, "PoissonDistribution": 1, "BinomialDistribution": 2,
                "ChiSquaredDistribution": 1, "TDistribution": 1, "StudentTDistribution": 1, "FDistribution": 2}[h]
        if len(a) != need:
            raise ValueError(f"{h} expects {need} parameter(s)")
        if h == "BinomialDistribution":
            if a[0].is_number and not (a[0].is_Integer and a[0] >= 0):
                raise ValueError("BinomialDistribution: n must be a non-negative integer")
            if a[1].is_number and not (0 <= a[1] <= 1):
                raise ValueError("BinomialDistribution: p must be in [0, 1]")
        elif any(v.is_number and not v > 0 for v in a):
            raise ValueError(f"{h}: parameters must be positive")
    if h == "StudentTDistribution":
        h = "TDistribution"
    return h, a


def _inert(name, nargs, notes):
    @fn(name, area=AREA, status="verified", notes=notes)
    def impl(*args):
        if len(args) not in nargs:
            raise ValueError(f"{name} expects {' or '.join(map(str, nargs))} parameter(s)")
        if name == "NormalDistribution" and not args:
            return Expr("NormalDistribution", 0, 1)
        _dist(Expr(name, *args), name)          # validates numeric parameters
        raise Unevaluated()
    return impl


_inert("NormalDistribution", (0, 2), "inert distribution object NormalDistribution[mu, sigma] (NormalDistribution[] = [0, 1])"
                                     ". Check: reference (differential cases in tests/library).")
_inert("ExponentialDistribution", (1,), "inert distribution object (rate lambda)")
_inert("PoissonDistribution", (1,), "inert distribution object (mean mu)")
_inert("BinomialDistribution", (2,), "inert distribution object [n, p]")
_inert("ChiSquaredDistribution", (1,), "inert distribution object (nu degrees of freedom)")
_inert("TDistribution", (1,), "inert Student t distribution (MikoshiLang name; StudentTDistribution is the Wolfram name)")
_inert("StudentTDistribution", (1,), "inert Student t distribution [nu]")
_inert("FDistribution", (2,), "inert F-ratio distribution [n, m] (Wolfram FRatioDistribution)")
_inert("UniformDistribution", (0, 1, 2),
       "inert distribution UniformDistribution[{min, max}] (Wolfram); the legacy form [min, max] is accepted")


def _pw(expr, cond, x):
    """Piecewise with 0 outside, evaluated when x is numeric."""
    if x.is_number:
        c = cond(x)
        if c in (True, False) or isinstance(c, sp.logic.boolalg.BooleanAtom):
            return expr(x) if bool(c) else sp.Integer(0)
    return sp.Piecewise((expr(x), cond(x)), (0, True))


def _pdf(h, a, x):
    if h == "NormalDistribution":
        m, s = a
        return sp.exp(-(x - m) ** 2 / (2 * s ** 2)) / (s * sp.sqrt(2 * sp.pi))
    if h == "ExponentialDistribution":
        lam = a[0]
        return _pw(lambda t: lam * sp.exp(-lam * t), lambda t: t >= 0, x)
    if h == "UniformDistribution":
        lo, hi = a
        return _pw(lambda t: 1 / (hi - lo), lambda t: sp.And(t >= lo, t <= hi), x)
    if h == "ChiSquaredDistribution":
        k = a[0]
        return _pw(lambda t: t ** (k / 2 - 1) * sp.exp(-t / 2) / (2 ** (k / 2) * sp.gamma(k / 2)), lambda t: t > 0, x)
    if h == "TDistribution":
        v = a[0]
        return sp.gamma((v + 1) / 2) / (sp.sqrt(v * sp.pi) * sp.gamma(v / 2)) * (1 + x ** 2 / v) ** (-(v + 1) / 2)
    if h == "FDistribution":
        d1, d2 = a
        return _pw(lambda t: sp.sqrt((d1 * t) ** d1 * d2 ** d2 / (d1 * t + d2) ** (d1 + d2)) / (t * sp.beta(d1 / 2, d2 / 2)),
                   lambda t: t > 0, x)
    if h == "PoissonDistribution":
        mu = a[0]
        if x.is_number:
            if not (x.is_integer and x >= 0):
                return sp.Integer(0)
        return sp.exp(-mu) * mu ** x / sp.factorial(x)
    if h == "BinomialDistribution":
        n, p = a
        if x.is_number:
            if not (x.is_integer and 0 <= x and (not n.is_number or x <= n)):
                return sp.Integer(0)
        return sp.binomial(n, x) * p ** x * (1 - p) ** (n - x)
    raise NotImplementedError(h)


def _betareg(a, b, z):
    """Regularized incomplete beta I_z(a, b); a closed-form polynomial for positive integer a, b."""
    if a.is_Integer and b.is_Integer and a > 0 and b > 0:
        n = int(a + b - 1)
        return sp.expand(sp.Add(*(sp.binomial(n, j) * z ** j * (1 - z) ** (n - j) for j in range(int(a), n + 1))))
    return sp.betainc_regularized(a, b, 0, z)


def _cdf(h, a, x):
    if h == "NormalDistribution":
        m, s = a
        return sp.erfc((m - x) / (s * sp.sqrt(2))) / 2
    if h == "ExponentialDistribution":
        lam = a[0]
        return _pw(lambda t: 1 - sp.exp(-lam * t), lambda t: t >= 0, x)
    if h == "UniformDistribution":
        lo, hi = a
        if x.is_number and all(v.is_number for v in a):
            return sp.Integer(0) if x < lo else (sp.Integer(1) if x > hi else (x - lo) / (hi - lo))
        return sp.Piecewise((0, x < lo), ((x - lo) / (hi - lo), x <= hi), (1, True))
    if h == "ChiSquaredDistribution":
        k = a[0]
        return _pw(lambda t: sp.lowergamma(k / 2, t / 2) / sp.gamma(k / 2), lambda t: t > 0, x)
    if h == "TDistribution":
        v = a[0]
        if not x.is_number:
            raise Unevaluated()
        if x == 0:
            return sp.Rational(1, 2)
        z = v / (v + x ** 2)
        tail = _betareg(v / 2, sp.Rational(1, 2), z) / 2
        return 1 - tail if x > 0 else tail
    if h == "FDistribution":
        d1, d2 = a
        if not x.is_number:
            raise Unevaluated()
        if x <= 0:
            return sp.Integer(0)
        return _betareg(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2))
    if h in ("PoissonDistribution", "BinomialDistribution"):
        if not x.is_number:
            raise Unevaluated()
        if x < 0:
            return sp.Integer(0)
        k = int(sp.floor(x))
        if h == "BinomialDistribution" and a[0].is_number and k >= a[0]:
            return sp.Integer(1)
        return sp.Add(*(_pdf(h, a, sp.Integer(i)) for i in range(k + 1)))
    raise NotImplementedError(h)


def _x_arg(x, fname):
    if is_list(x):
        raise Unevaluated()
    return sym(x, fname)


@fn("PDF", area=AREA, status="fixed",
    notes="exact/symbolic densities (legacy forced floats, rounded to 10 digits so PDF[NormalDistribution[0, 1], 10] "
          "was 0., and truncated Poisson/Binomial arguments with int()); PMF for discrete distributions")
def _pdf_fn(d, x):
    h, a = _dist(d, "PDF")
    xv = _x_arg(x, "PDF")
    return numeric_out(_pdf(h, a, xv), d, x)


@fn("CDF", area=AREA, status="fixed",
    notes="exact CDFs (normal via Erfc, chi-squared via incomplete gamma, t and F via regularised incomplete beta); "
          "legacy rounded to 10 digits and truncated discrete arguments")
def _cdf_fn(d, x):
    h, a = _dist(d, "CDF")
    xv = _x_arg(x, "CDF")
    return numeric_out(_cdf(h, a, xv), d, x)


def _numeric_inverse(h, a, p):
    import mpmath
    f = lambda t: float(sp.N(_cdf(h, a, sp.Float(t)), 20)) - p
    lo, hi = (-1.0, 1.0) if h == "TDistribution" else (0.0, 1.0)
    while f(hi) < 0:
        hi *= 2
    while h == "TDistribution" and f(lo) > 0:
        lo *= 2
    return float(mpmath.findroot(f, (lo, hi), solver="anderson"))


@fn("InverseCDF", area=AREA, status="fixed",
    notes="exact for normal, exponential and uniform (InverseCDF[NormalDistribution[0, 1], 1/2] = 0); numeric "
          "(mpmath root of the CDF) for t, chi-squared and F with machine-number p; legacy rounded to 10 digits "
          "and returned an Error expression for other distributions")
def _invcdf(d, q):
    h, a = _dist(d, "InverseCDF")
    p = number(q, "InverseCDF", "probability")
    if not 0 <= p <= 1:
        raise ValueError("InverseCDF: probability must be in [0, 1]")
    if h == "NormalDistribution":
        m, s = a
        v = -sp.oo if p == 0 else (sp.oo if p == 1 else m + s * sp.sqrt(2) * sp.erfinv(2 * p - 1))
        return numeric_out(v, d, q)
    if h == "ExponentialDistribution":
        return numeric_out(sp.oo if p == 1 else -sp.log(1 - p) / a[0], d, q)
    if h == "UniformDistribution":
        return numeric_out(a[0] + p * (a[1] - a[0]), d, q)
    if h in ("PoissonDistribution", "BinomialDistribution"):
        if not all(v.is_number for v in a):
            raise Unevaluated()
        k, acc = 0, _pdf(h, a, sp.Integer(0))
        while acc < p:
            k += 1
            acc += _pdf(h, a, sp.Integer(k))
            if k > 10 ** 6:
                raise ValueError("InverseCDF: quantile too large")
        return k
    if not has_float(q) or not all(v.is_number for v in a):
        raise Unevaluated()
    if p in (0, 1):
        return R((-sp.oo if h == "TDistribution" else 0) if p == 0 else sp.oo)
    return _numeric_inverse(h, a, float(p))


def _dist_moment(d, fname, which):
    h, a = _dist(d, fname)
    if h == "NormalDistribution":
        m, s = a
        mean, var = m, s ** 2
    elif h == "ExponentialDistribution":
        mean, var = 1 / a[0], 1 / a[0] ** 2
    elif h == "PoissonDistribution":
        mean, var = a[0], a[0]
    elif h == "BinomialDistribution":
        mean, var = a[0] * a[1], a[0] * a[1] * (1 - a[1])
    elif h == "UniformDistribution":
        mean, var = (a[0] + a[1]) / 2, (a[1] - a[0]) ** 2 / 12
    elif h == "ChiSquaredDistribution":
        mean, var = a[0], 2 * a[0]
    elif h == "TDistribution":
        v = a[0]
        if not v.is_number:
            raise Unevaluated()
        mean = 0 if v > 1 else sp.nan
        var = v / (v - 2) if v > 2 else (sp.oo if v > 1 else sp.nan)
    else:
        d1, d2 = a
        if not d2.is_number:
            raise Unevaluated()
        mean = d2 / (d2 - 2) if d2 > 2 else sp.nan
        var = 2 * d2 ** 2 * (d1 + d2 - 2) / (d1 * (d2 - 2) ** 2 * (d2 - 4)) if d2 > 4 else sp.nan
    v = {"mean": mean, "var": var, "sd": sp.sqrt(var)}[which]
    if v is sp.nan:
        raise ValueError(f"{fname}: the moment does not exist for these parameters")
    return numeric_out(sp.simplify(v), d)


def _is_dist(x):
    return isinstance(x, Expr) and isinstance(x.head, str) and x.head.endswith("Distribution")


# ─────────────────────────── descriptive ───────────────────────────

@fn("Mean", area=AREA, status="fixed", notes=EXACT + "; Mean[dist] gives the distribution mean")
def _mean_fn(x):
    if _is_dist(x):
        return _dist_moment(x, "Mean", "mean")
    v, items = _data(x, "Mean")
    return _out(_mean(v), items)


@fn("Median", area=AREA, status="fixed", notes=EXACT + "; even length averages the two middle values")
def _median_fn(x):
    v, items = _data(x, "Median")
    return _out(_median(v), items)


@fn("Variance", area=AREA, status="fixed", notes="unbiased sample variance (n - 1), " + EXACT + "; Variance[dist]")
def _variance(x):
    if _is_dist(x):
        return _dist_moment(x, "Variance", "var")
    v, items = _data(x, "Variance", 2)
    return _out(_var(v), items)


@fn("StandardDeviation", area=AREA, status="fixed", notes="Sqrt[Variance], " + EXACT + "; StandardDeviation[dist]")
def _sd(x):
    if _is_dist(x):
        return _dist_moment(x, "StandardDeviation", "sd")
    v, items = _data(x, "StandardDeviation", 2)
    return _out(sp.sqrt(_var(v)), items)


def _central(v, k):
    m = _mean(v)
    return sp.Add(*((x - m) ** k for x in v)) / len(v)


@fn("Skewness", area=AREA, status="fixed", notes="m3/m2^(3/2) with population moments (Wolfram), " + EXACT)
def _skew(x):
    v, items = _data(x, "Skewness", 2)
    m2 = _central(v, 2)
    if m2 == 0:
        raise ValueError("Skewness: data has zero variance")
    return _out(_central(v, 3) / m2 ** sp.Rational(3, 2), items)


@fn("Kurtosis", area=AREA, status="fixed",
    notes="m4/m2^2 (Wolfram; normal data -> 3); legacy returned SciPy's excess (Fisher) kurtosis, i.e. 3 too "
          "small, rounded to 10 digits")
def _kurt(x):
    v, items = _data(x, "Kurtosis", 2)
    m2 = _central(v, 2)
    if m2 == 0:
        raise ValueError("Kurtosis: data has zero variance")
    return _out(_central(v, 4) / m2 ** 2, items)


def _pair(x, y, fname):
    a, ia = _data(x, fname, 2)
    b, ib = _data(y, fname, 2)
    if len(a) != len(b):
        raise ValueError(f"{fname}: data lists must have the same length")
    return a, b, ia + ib


def _cov(a, b):
    ma, mb = _mean(a), _mean(b)
    return sp.Add(*((p - ma) * (q - mb) for p, q in zip(a, b))) / (len(a) - 1)


@fn("Covariance", area=AREA, status="fixed", notes="unbiased sample covariance, " + EXACT)
def _covariance(x, y):
    a, b, items = _pair(x, y, "Covariance")
    return _out(_cov(a, b), items)


def _pearson(a, b, fname):
    va, vb = _var(a), _var(b)
    if va == 0 or vb == 0:
        raise ValueError(f"{fname}: correlation is undefined for constant data")
    return _cov(a, b) / sp.sqrt(va * vb)


@fn("Correlation", area=AREA, status="fixed", notes="Pearson correlation, " + EXACT)
def _correlation(x, y):
    a, b, items = _pair(x, y, "Correlation")
    return _out(_pearson(a, b, "Correlation"), items)


def _scipy(fname):
    try:
        from scipy import stats
        return stats
    except ImportError:
        raise NotImplementedError(f"{fname} requires SciPy (pip install mikoshilang[signal])")


def _floats(v):
    return [float(x) for x in v]


@fn("PearsonCorrelation", area=AREA, status="fixed",
    notes="{r -> exact Pearson r, p -> two-sided p-value (SciPy)} (MikoshiLang)"
          ". Check: reference (differential cases in tests/library).")
def _pearsonr(x, y):
    a, b, items = _pair(x, y, "PearsonCorrelation")
    r = _out(_pearson(a, b, "PearsonCorrelation"), items)
    st = _scipy("PearsonCorrelation")
    return L(rule(Symbol("r"), r), rule(Symbol("p"), float(st.pearsonr(_floats(a), _floats(b))[1])))


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [sp.Integer(0)] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r = sp.Rational(i + j + 2, 2)
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


@fn("SpearmanCorrelation", area=AREA, status="fixed",
    notes="{rho -> exact Spearman rho (average ranks for ties), p -> p-value (SciPy)} (MikoshiLang)"
          ". Check: reference (differential cases in tests/library).")
def _spearman(x, y):
    a, b, items = _pair(x, y, "SpearmanCorrelation")
    rho = _out(_pearson(_ranks(a), _ranks(b), "SpearmanCorrelation"), items)
    st = _scipy("SpearmanCorrelation")
    return L(rule(Symbol("rho"), rho), rule(Symbol("p"), float(st.spearmanr(_floats(a), _floats(b))[1])))


@fn("KendallCorrelation", area=AREA, status="verified", notes="{tau -> Kendall tau-b, p -> p-value} (SciPy)"
                                                              ". Check: reference (differential cases in "
                                                              "tests/library).")
def _kendall(x, y):
    a, b, _ = _pair(x, y, "KendallCorrelation")
    st = _scipy("KendallCorrelation")
    t, p = st.kendalltau(_floats(a), _floats(b))
    return L(rule(Symbol("tau"), float(t)), rule(Symbol("p"), float(p)))


@fn("GeometricMean", area=AREA, status="fixed", notes="(x1 ... xn)^(1/n), " + EXACT + "; requires positive data")
def _gmean(x):
    v, items = _data(x, "GeometricMean")
    if any(not t > 0 for t in v):
        raise ValueError("GeometricMean: data must be positive")
    return _out(sp.Mul(*v) ** sp.Rational(1, len(v)), items)


@fn("HarmonicMean", area=AREA, status="fixed", notes="n / Sum[1/x], " + EXACT + "; zero values raise")
def _hmean(x):
    v, items = _data(x, "HarmonicMean")
    if any(t == 0 for t in v):
        raise ValueError("HarmonicMean: data must be non-zero")
    return _out(len(v) / sp.Add(*(1 / t for t in v)), items)


@fn("MAD", area=AREA, status="fixed", notes="median absolute deviation Median[Abs[x - Median[x]]] (Wolfram MedianDeviation), " + EXACT +
                                            ". Check: exact (exact input gives exact results), reference "
                                            "(differential cases in tests/library).")
def _mad(x):
    v, items = _data(x, "MAD")
    m = _median(v)
    return _out(_median([sp.Abs(t - m) for t in v]), items)


def _quantile(v, q, params):
    (a, b), (c, d) = params
    s = sorted(v)
    n = len(s)
    x = a + (n + b) * q
    fl = sp.floor(x)
    frac = x - fl
    lo = s[min(max(int(fl), 1), n) - 1]
    hi = s[min(max(int(sp.ceiling(x)), 1), n) - 1]
    return lo + (hi - lo) * (c + d * frac)


def _qparams(p, fname):
    if p is None:
        return ((0, 0), (1, 0))
    rows = list_arg(p, fname, "parameters")
    vals = [[number(v, fname, "parameter") for v in list_arg(r, fname, "parameters")] for r in rows]
    if len(vals) != 2 or any(len(r) != 2 for r in vals):
        raise ValueError(f"{fname}: parameters must be {{{{a, b}}, {{c, d}}}}")
    return tuple(tuple(r) for r in vals)


@fn("Quantile", area=AREA, status="fixed",
    notes="Wolfram definition s[[Ceiling[n q]]] (q = 0 gives the minimum), lists of q and the 4-parameter form "
          "Quantile[list, q, {{a, b}, {c, d}}]; legacy used NumPy's linear interpolation (a different estimator) "
          "returning floats"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _quantile_fn(x, q, params=None):
    v, items = _data(x, "Quantile")
    pr = _qparams(params, "Quantile")
    if is_list(q):
        return L(*(_quantile_fn(x, qq, params) for qq in q.args))
    qq = number(q, "Quantile", "probability")
    if not 0 <= qq <= 1:
        raise ValueError("Quantile: q must be in [0, 1]")
    return numeric_out(_quantile(v, qq, pr), *items, q)


@fn("Percentile", area=AREA, status="fixed",
    notes="Percentile[list, p] with p in [0, 100] using linear interpolation between order statistics "
          "(NumPy/Excel convention, = Quantile[list, p/100, {{1, -1}, {0, 1}}]); now exact for exact data"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _percentile(x, p):
    v, items = _data(x, "Percentile")
    pp = number(p, "Percentile", "percentage")
    if not 0 <= pp <= 100:
        raise ValueError("Percentile: p must be in [0, 100]")
    return numeric_out(_quantile(v, pp / 100, ((1, -1), (0, 1))), *items, p)


@fn("IQR", area=AREA, status="fixed",
    notes="Wolfram InterquartileRange: Q3 - Q1 with Quantile parameters {{1/2, 0}, {0, 1}}; legacy used NumPy's "
          "linear percentiles (a different estimator)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _iqr(x):
    v, items = _data(x, "IQR")
    pr = ((sp.Rational(1, 2), 0), (0, 1))
    return numeric_out(_quantile(v, sp.Rational(3, 4), pr) - _quantile(v, sp.Rational(1, 4), pr), *items)


@fn("Mode", area=AREA, status="fixed",
    notes="the unique most frequent value (MikoshiLang); multimodal data now raises (legacy returned the first "
          "mode silently) - use Commonest"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _mode(x):
    items = list_arg(x, "Mode", min_len=1)
    c = _commonest(items)
    if len(c) > 1:
        raise ValueError("Mode: data is multimodal; use Commonest[list]")
    return c[0]


def _commonest(items):
    from ..kernel import same
    keys, counts = [], []
    for it in items:
        for i, k in enumerate(keys):
            if same(it, k):
                counts[i] += 1
                break
        else:
            keys.append(it)
            counts.append(1)
    m = max(counts)
    return [k for k, c in zip(keys, counts) if c == m]


@fn("Commonest", area=AREA, status="promoted",
    notes="Commonest[list]: the most frequent elements in order of first appearance; Commonest[list, n]: the n "
          "most frequent, by decreasing frequency (ties by first appearance)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _commonest_fn(x, n=None):
    from ..kernel import same
    items = list_arg(x, "Commonest", min_len=1)
    if n is None:
        return L(*_commonest(items))
    k = pos_int(n, "Commonest", "n")
    counts, first = [], []
    for it in items:
        for j, u in enumerate(first):
            if same(u, it):
                counts[j] += 1
                break
        else:
            first.append(it)
            counts.append(1)
    order = sorted(range(len(first)), key=lambda j: -counts[j])
    return L(*(first[j] for j in order[:k]))


@fn("TrimmedMean", area=AREA, status="fixed",
    notes="mean after dropping Floor[f n] values from each end (0 <= f < 1/2), " + EXACT +
          ". Check: reference (differential cases in tests/library).")
def _tmean(x, f):
    v, items = _data(x, "TrimmedMean")
    ff = number(f, "TrimmedMean", "fraction")
    if not 0 <= ff < sp.Rational(1, 2):
        raise ValueError("TrimmedMean: fraction must be in [0, 1/2)")
    k = int(sp.floor(ff * len(v)))
    s = sorted(v)[k:len(v) - k]
    return _out(_mean(s), items)


@fn("CoefficientOfVariation", area=AREA, status="fixed", notes="StandardDeviation/Mean, " + EXACT +
                                                               ". Check: exact (exact input gives exact results), "
                                                               "reference (differential cases in tests/library).")
def _cv(x):
    v, items = _data(x, "CoefficientOfVariation", 2)
    m = _mean(v)
    if m == 0:
        raise ValueError("CoefficientOfVariation: mean is zero")
    return _out(sp.sqrt(_var(v)) / m, items)


@fn("StandardError", area=AREA, status="fixed", notes="StandardDeviation/Sqrt[n] (standard error of the mean), " + EXACT +
                                                      ". Check: exact (exact input gives exact results), reference "
                                                      "(differential cases in tests/library).")
def _sem(x):
    v, items = _data(x, "StandardError", 2)
    return _out(sp.sqrt(_var(v) / len(v)), items)


@fn("ZScore", area=AREA, status="fixed", notes="(x - mu)/sigma, exact; sigma <= 0 raises (legacy rounded floats)"
                                               ". Check: exact (exact input gives exact results), reference "
                                               "(differential cases in tests/library).")
def _zscore(x, mu, sigma):
    s = number(sigma, "ZScore", "sigma")
    if not s > 0:
        raise ValueError("ZScore: sigma must be positive")
    return numeric_out((number(x, "ZScore") - number(mu, "ZScore")) / s, x, mu, sigma)


@fn("ZScoreNormalize", area=AREA, status="fixed", notes="(x - Mean)/StandardDeviation (Wolfram Standardize), " + EXACT +
                                                        ". Check: exact (exact input gives exact results), reference "
                                                        "(differential cases in tests/library).")
def _znorm(x):
    v, items = _data(x, "ZScoreNormalize", 2)
    m, s = _mean(v), sp.sqrt(_var(v))
    if s == 0:
        raise ValueError("ZScoreNormalize: data has zero variance")
    return L(*(_out((t - m) / s, items) for t in v))


def _cohen(x, y, fname):
    a, ia = _data(x, fname, 2)
    b, ib = _data(y, fname, 2)
    na, nb = len(a), len(b)
    pooled = ((na - 1) * _var(a) + (nb - 1) * _var(b)) / (na + nb - 2)
    if pooled == 0:
        raise ValueError(f"{fname}: pooled standard deviation is zero")
    return _out((_mean(a) - _mean(b)) / sp.sqrt(pooled), ia + ib)


fn("CohensD", area=AREA, status="fixed", notes="(mean1 - mean2)/pooled SD, " + EXACT +
                                               ". Check: exact (exact input gives exact results), reference "
                                               "(differential cases in tests/library).")(lambda x, y: _cohen(x, y, "CohensD"))
fn("EffectSize", area=AREA, status="fixed", notes="alias of CohensD, " + EXACT +
                                                  ". Check: exact (exact input gives exact results), reference "
                                                  "(differential cases in tests/library).")(lambda x, y: _cohen(x, y, "EffectSize"))


@fn("ConfidenceInterval", area=AREA, status="fixed",
    notes="t-based interval for the mean at the given level; legacy silently used 0.95 when the level was not a "
          "machine number (e.g. 9/10); now any level in (0, 1) and no rounding"
          ". Check: reference (differential cases in tests/library).")
def _ci(x, level):
    v, items = _data(x, "ConfidenceInterval", 2)
    c = number(level, "ConfidenceInterval", "confidence level")
    if not 0 < c < 1:
        raise ValueError("ConfidenceInterval: level must be in (0, 1)")
    st = _scipy("ConfidenceInterval")
    m = float(_mean(v))
    se = float(sp.sqrt(_var(v) / len(v)))
    h = se * float(st.t.ppf((1 + float(c)) / 2, len(v) - 1))
    return L(m - h, m + h)


def _rules(**kw):
    return L(*(rule(Symbol(k), float(v)) for k, v in kw.items()))


@fn("TTest", area=AREA, status="fixed",
    notes="Wolfram TTest[data] and TTest[data, mu0] return the two-sided one-sample p-value; TTest[{x, y}] the "
          "Welch two-sample p-value; legacy returned rounded rule lists and TTest[x, y] used the pooled-variance "
          "test (now raises with a pointer to TTest[{x, y}])"
          ". Check: reference (differential cases in tests/library).")
def _ttest(x, mu0=None):
    st = _scipy("TTest")
    items = list_arg(x, "TTest", "data", min_len=2)
    if all(is_list(i) for i in items) and len(items) == 2 and mu0 is None:
        a, _ = _data(items[0], "TTest", 2)
        b, _ = _data(items[1], "TTest", 2)
        return float(st.ttest_ind(_floats(a), _floats(b), equal_var=False).pvalue)
    if is_list(mu0):
        raise ValueError("TTest: for two samples use TTest[{data1, data2}]")
    v, _ = _data(x, "TTest", 2)
    m0 = 0.0 if mu0 is None else float(number(mu0, "TTest", "mu0"))
    return float(st.ttest_1samp(_floats(v), m0).pvalue)


@fn("ChiSquaredTest", area=AREA, status="fixed",
    notes="{chi2 -> statistic, p_value -> p} for observed vs expected counts (SciPy chisquare), unrounded"
          ". Check: reference (differential cases in tests/library).")
def _chi2(o, e):
    st = _scipy("ChiSquaredTest")
    a, _ = _data(o, "ChiSquaredTest")
    b, _ = _data(e, "ChiSquaredTest")
    if len(a) != len(b):
        raise ValueError("ChiSquaredTest: lists must have the same length")
    if abs(sum(_floats(a)) - sum(_floats(b))) > 1e-8 * max(1.0, sum(_floats(a))):
        raise ValueError("ChiSquaredTest: observed and expected totals must agree")
    r = st.chisquare(_floats(a), _floats(b))
    return _rules(chi2=r[0], p_value=r[1])


def _two(fname, x, y):
    a, _ = _data(x, fname, 1)
    b, _ = _data(y, fname, 1)
    return _floats(a), _floats(b)


@fn("MannWhitneyU", area=AREA, status="fixed", notes="{U -> statistic, p_value -> two-sided p} (SciPy), unrounded"
                                                     ". Check: reference (differential cases in tests/library).")
def _mwu(x, y):
    st = _scipy("MannWhitneyU")
    r = st.mannwhitneyu(*_two("MannWhitneyU", x, y), alternative="two-sided")
    return _rules(U=r[0], p_value=r[1])


@fn("WilcoxonTest", area=AREA, status="fixed", notes="paired signed-rank test {W -> statistic, p_value -> p} (SciPy), unrounded"
                                                     ". Check: reference (differential cases in tests/library).")
def _wilcoxon(x, y):
    st = _scipy("WilcoxonTest")
    a, b = _two("WilcoxonTest", x, y)
    if len(a) != len(b):
        raise ValueError("WilcoxonTest: paired data must have the same length")
    r = st.wilcoxon(a, b)
    return _rules(W=r[0], p_value=r[1])


@fn("ShapiroWilk", area=AREA, status="fixed", notes="{W -> statistic, p_value -> p} (SciPy), unrounded; n >= 3"
                                                    ". Check: reference (differential cases in tests/library).")
def _shapiro(x):
    st = _scipy("ShapiroWilk")
    v, _ = _data(x, "ShapiroWilk", 3)
    r = st.shapiro(_floats(v))
    return _rules(W=r[0], p_value=r[1])


@fn("AndersonDarling", area=AREA, status="fixed",
    notes="{statistic, criticalValues, significanceLevels} for normality with estimated mean and sd (Stephens "
          "critical values with the 1 + 0.75/n + 2.25/n^2 correction, identical to SciPy anderson); computed directly, no SciPy needed (legacy relied "
          "on an API SciPy is removing)"
          ". Check: reference (differential cases in tests/library).")
def _anderson(x):
    import math as _m
    v, _ = _data(x, "AndersonDarling", 3)
    xs = sorted(_floats(v))
    n = len(xs)
    mu = sum(xs) / n
    sd = _m.sqrt(sum((t - mu) ** 2 for t in xs) / (n - 1))
    if sd == 0:
        raise ValueError("AndersonDarling: data must not be constant")
    z = [(t - mu) / sd for t in xs]
    logcdf = [_m.log(0.5 * _m.erfc(-t / _m.sqrt(2))) for t in z]
    logsf = [_m.log(0.5 * _m.erfc(t / _m.sqrt(2))) for t in z]
    a2 = -n - sum((2 * i - 1) * (logcdf[i - 1] + logsf[n - i]) for i in range(1, n + 1)) / n
    crit = [round(c / (1.0 + 0.75 / n + 2.25 / n / n), 3) for c in (0.561, 0.631, 0.752, 0.873, 1.035)]
    return L(rule(Symbol("statistic"), a2),
             rule(Symbol("criticalValues"), L(*crit)),
             rule(Symbol("significanceLevels"), L(15.0, 10.0, 5.0, 2.5, 1.0)))


@fn("ANOVA", area=AREA, status="fixed", notes="one-way ANOVA {F, p_value} for 2 or more groups (SciPy), unrounded"
                                              ". Check: reference (differential cases in tests/library).")
def _anova(*groups):
    st = _scipy("ANOVA")
    if len(groups) < 2:
        raise ValueError("ANOVA needs at least two groups")
    r = st.f_oneway(*(_floats(_data(g, "ANOVA", 2)[0]) for g in groups))
    return _rules(F=r[0], p_value=r[1])


@fn("KruskalWallis", area=AREA, status="fixed", notes="{H, p_value} for 2 or more groups (SciPy), unrounded"
                                                      ". Check: reference (differential cases in tests/library).")
def _kruskal(*groups):
    st = _scipy("KruskalWallis")
    if len(groups) < 2:
        raise ValueError("KruskalWallis needs at least two groups")
    r = st.kruskal(*(_floats(_data(g, "KruskalWallis", 1)[0]) for g in groups))
    return _rules(H=r[0], p_value=r[1])


@fn("LinearRegression", area=AREA, status="fixed",
    notes="{slope, intercept, r_squared} exact least squares plus p_value and std_error of the slope (t-test, as scipy.stats.linregress; no SciPy needed); legacy rounded"
          ". Check: reference (differential cases in tests/library).")
def _linreg(x, y):
    a, b, items = _pair(x, y, "LinearRegression")
    va = _var(a)
    if va == 0:
        raise ValueError("LinearRegression: x values must not all be equal")
    slope = _cov(a, b) / va
    intercept = _mean(b) - slope * _mean(a)
    vb = _var(b)
    r2 = (_cov(a, b) ** 2 / (va * vb)) if vb != 0 else sp.Integer(1)
    out = [rule(Symbol("slope"), _out(slope, items)), rule(Symbol("intercept"), _out(intercept, items)),
           rule(Symbol("r_squared"), _out(r2, items))]
    if len(a) > 2:
        # two-sided t-test of slope = 0 with n - 2 degrees of freedom (matches scipy.stats.linregress)
        import mpmath as mpm
        n, df = len(a), len(a) - 2
        sxx = sum((u - _mean(a)) ** 2 for u in a)
        ssr = sum((v - intercept - slope * u) ** 2 for u, v in zip(a, b))
        se = mpm.sqrt(mpm.mpf(sp.N(ssr / df / sxx, 30)))
        if se == 0:
            pv = 0.0
        else:
            t = mpm.mpf(sp.N(slope, 30)) / se
            pv = float(mpm.betainc(df / 2.0, 0.5, 0, df / (df + t * t), regularized=True))
        out += [rule(Symbol("p_value"), pv), rule(Symbol("std_error"), float(se))]
    return L(*out)


@fn("RandomSample", area=AREA, status="fixed",
    notes="Wolfram RandomSample[list] / [list, n]: a random sample without replacement; legacy drew numbers from a "
          "distribution (now RandomVariate) and fell back to N(0, 1) for unknown input"
          ". Check: reference (differential cases in tests/library).")
def _rsample(x, n=None):
    if _is_dist(x):
        raise ValueError("RandomSample samples elements of a list; use RandomVariate[dist, n] for distributions")
    items = list_arg(x, "RandomSample")
    k = len(items) if n is None else nonneg_int(n, "RandomSample", "count")
    return L(*_RNG.sample(items, min(k, len(items))))


@fn("RandomVariate", area=AREA, status="promoted",
    notes="RandomVariate[dist] and RandomVariate[dist, n] by inverse-CDF sampling (machine numbers; integers for "
          "discrete distributions)")
def _rvariate(d, n=None):
    h, a = _dist(d, "RandomVariate")
    if not all(v.is_number for v in a):
        raise ValueError("RandomVariate: distribution parameters must be numeric")
    k = None if n is None else nonneg_int(n, "RandomVariate", "count")

    def one():
        if h == "NormalDistribution":
            return _RNG.gauss(float(a[0]), float(a[1]))
        if h == "ExponentialDistribution":
            return _RNG.expovariate(float(a[0]))
        if h == "UniformDistribution":
            return _RNG.uniform(float(a[0]), float(a[1]))
        if h == "BinomialDistribution":
            return sum(1 for _ in range(int(a[0])) if _RNG.random() < float(a[1]))
        if h == "PoissonDistribution":
            lam, kk, p = math.exp(-float(a[0])), 0, 1.0
            while True:
                p *= _RNG.random()
                if p <= lam:
                    return kk
                kk += 1
        if h == "ChiSquaredDistribution":
            return 2 * _RNG.gammavariate(float(a[0]) / 2, 1.0)
        if h == "TDistribution":
            v = float(a[0])
            return _RNG.gauss(0, 1) / math.sqrt(2 * _RNG.gammavariate(v / 2, 1.0) / v)
        d1, d2 = float(a[0]), float(a[1])
        return (2 * _RNG.gammavariate(d1 / 2, 1.0) / d1) / (2 * _RNG.gammavariate(d2 / 2, 1.0) / d2)
    return one() if k is None else L(*(one() for _ in range(k)))


@fn("SeedRandom", area=AREA, status="promoted", notes="SeedRandom[n] seeds RandomSample/RandomVariate"
                                                      ". Check: reference (differential cases in tests/library).")
def _seed(n=None):
    if isinstance(n, str):
        _RNG.seed(n)                      # string seeds are deterministic, as in Wolfram
    else:
        _RNG.seed(None if n is None else int_arg(n, "SeedRandom", "seed"))
    return Symbol("Null")

record("StudentTDistribution", AREA, "promoted",
       "alias of TDistribution[nu] (Wolfram StudentTDistribution[nu]); inert distribution object for PDF/CDF/"
       "InverseCDF/Mean/Variance/RandomVariate")
