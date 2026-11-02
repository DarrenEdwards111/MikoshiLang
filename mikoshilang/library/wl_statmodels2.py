"""Wolfram batch X02 (part 1): multivariate normality and goodness-of-fit tests (BaringhausHenzeTest, the Mardia
tests, PearsonChiSquareTest), unit-root tests (UnitRootTest), PearsonDistribution, FindDistribution and Dendrogram.

Model specifications and ModelFit live in ``wl_statmodels2_models``; hidden Markov processes, time-series models
and queueing networks in ``wl_statmodels2_ts``.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).  Where Wolfram's choice is undocumented (bin counts, smoothing parameters, small-sample
corrections, candidate families) the method used here is a named published one and is stated in the notes; it is
never claimed to reproduce Wolfram's undocumented default.  p-values are machine numbers computed with mpmath.
"""
from __future__ import annotations

import math

import mpmath
import numpy as np
import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, L, R, rule, S, is_list, has_float
from . import wl_statmodels as SM
from . import wl_stats as W

AREA = "statistical models II (X02)"


def _ev(x):
    from ..evaluate import evaluate
    return evaluate(x)


def _sym(x, *names):
    return isinstance(x, Symbol) and (not names or x.name in names)


# ─────────────────────────── data helpers ───────────────────────────

def mv_data(x, fname, min_n=3):
    """n x d float array from a vector {x1, ...} (d = 1) or a matrix {{x11, ...}, ...}"""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x) or not x.args:
        raise TypeError(f"{fname}: the data must be a non-empty list of numbers or of numeric vectors")
    if all(not is_list(v) for v in x.args):
        a = np.array([[SM.fl(v)] for v in SM.vec(x, fname)[0]], dtype=float)
    elif SM.is_matrix(x):
        a = np.array([[SM.fl(v) for v in r] for r in SM.mat(x, fname)], dtype=float)
    else:
        raise TypeError(f"{fname}: the data must be a vector or a rectangular matrix of numbers")
    n, d = a.shape
    if n < max(min_n, d + 2):
        raise ValueError(f"{fname}: at least {max(min_n, d + 2)} observations are needed for {d}-dimensional data")
    return a


def _ml_cov(a, fname):
    xc = a - a.mean(axis=0)
    S_ = xc.T @ xc / a.shape[0]
    if np.linalg.matrix_rank(S_) < a.shape[1] or np.linalg.cond(S_) > 1e13:
        raise ValueError(f"{fname}: the sample covariance matrix is singular")
    return xc, S_


def _mahal_gram(a, fname):
    """centred data, ML covariance (divisor n) and the Gram matrix D_jk = (x_j - xbar)' S^-1 (x_k - xbar)"""
    xc, S_ = _ml_cov(a, fname)
    return xc @ np.linalg.solve(S_, xc.T)


def _normal_null(d, fname):
    """the null distribution argument of the multivariate normality tests: only the composite hypothesis (estimated
    parameters) is supported"""
    if d is None or _sym(d, "Automatic"):
        return
    if isinstance(d, Expr) and d.head in ("MultinormalDistribution", "NormalDistribution", "BinormalDistribution"):
        if not d.args or all(isinstance(v, Symbol) or (is_list(v) and all(isinstance(q, Symbol) for q in v.args))
                             for v in d.args):
            return
        raise NotImplementedError(f"{fname}: testing against a fully specified normal distribution is not "
                                  "supported (only the composite hypothesis with estimated mean and covariance)")
    if isinstance(d, Expr) and isinstance(d.head, str) and d.head.endswith("Distribution"):
        raise ValueError(f"{fname}: the null distribution must be a (multi)normal distribution, not {d.head}")
    raise TypeError(f"{fname}: invalid distribution argument {d}")


def _test_args(rest, fname, opt_names=("SignificanceLevel", "Method", "VerifyTestAssumptions")):
    """(dist or None, property or None, options) from the arguments after the data"""
    args, o = SM.opts(rest, opt_names, fname)
    d = None
    if args and not isinstance(args[0], str) and not (is_list(args[0]) and args[0].args and
                                                     all(isinstance(q, str) for q in args[0].args)):
        d = args.pop(0)
    (), prop = SM.prop_arg(args, fname, 0)
    if prop is not None:
        for q in (prop.args if is_list(prop) else [prop]):
            if q in ("FittedDistribution", "FittedDistributionParameters"):
                raise NotImplementedError(f'{fname}: the property "{q}" is not supported')
    return d, prop, o


def _method(o, fname, allowed=("Automatic", "Asymptotic")):
    """Method setting -> (name, suboptions dict)"""
    m = o.get("Method", "Automatic")
    sub = {}
    if is_list(m) and m.args:
        for r in m.args[1:]:
            if SM._is_rule(r) and isinstance(r.args[0], str):
                sub[r.args[0]] = r.args[1]
            else:
                raise ValueError(f"{fname}: invalid Method suboption {r}")
        m = m.args[0]
    m = SM.str_or_sym(m) if not isinstance(m, str) else m
    if m == "MonteCarlo":
        raise NotImplementedError(f"{fname}: Method -> \"MonteCarlo\" is not supported")
    if m not in allowed:
        raise ValueError(f"{fname}: unknown Method {m!r}")
    return m, sub


# ─────────────────────────── Mardia tests ───────────────────────────

def mardia(a, fname):
    """(b1, b2, n, d): Mardia's (1970) multivariate skewness and kurtosis with the ML covariance"""
    n, d = a.shape
    G = _mahal_gram(a, fname)
    b1 = float(np.sum(G ** 3)) / n ** 2
    b2 = float(np.sum(np.diag(G) ** 2)) / n
    return b1, b2, n, d


def _mardia_parts(a, fname):
    b1, b2, n, d = mardia(a, fname)
    df = d * (d + 1) * (d + 2) // 6
    chi = n * b1 / 6
    z = (b2 - d * (d + 2)) / math.sqrt(8 * d * (d + 2) / n)
    return chi, df, z


_MARDIA_NOTE = (" Data: a vector {x1, ...} or a matrix of observations {{x11, ...}, ...}; the second argument may be "
                "Automatic or a normal distribution with symbolic parameters (composite null hypothesis with the "
                "sample mean and the maximum-likelihood covariance, divisor n); fully specified distributions, "
                "\"FittedDistribution\", Method -> \"MonteCarlo\" and the HypothesisTestData/table properties raise "
                "NotImplementedError. Properties \"PValue\" (default), \"TestStatistic\", \"TestData\", "
                "\"DegreesOfFreedom\", \"ShortTestConclusion\"; option SignificanceLevel. Method -> Automatic and "
                "\"Asymptotic\" both use the asymptotic distribution (Mardia 1970): Wolfram's Automatic applies an "
                "undocumented small-sample correction, so for small samples its p-values can differ from these. "
                "A singular sample covariance raises ValueError.")


def _mardia_impl(name, which):
    def impl(x: ListOf[Real] | MatrixOf[Real], *rest: Expr | String | Option) -> Real | Expr:
        d, prop, o = _test_args(rest, name)
        _method(o, name)
        _normal_null(d, name)
        a = mv_data(x, name)
        chi, df, z = _mardia_parts(a, name)
        if which == "skew":
            return SM.test_result(name, chi, SM.chi2_sf(chi, df), prop, o, df)
        if which == "kurt":
            return SM.test_result(name, z, SM.p_normal(z, "Unequal"), prop, o, None)
        stat = chi + z * z
        return SM.test_result(name, stat, SM.chi2_sf(stat, df + 1), prop, o, df + 1)
    impl.__name__ = "_" + name
    return impl


fn("MardiaSkewnessTest", area=AREA, status="added",
   notes="MardiaSkewnessTest[data] / [data, dist] / [data, dist, prop]: Mardia's (1970) multivariate skewness b1 = "
         "Sum[(z_j . S^-1 . z_k)^3]/n^2 of the centred observations z; the test statistic n b1/6 is referred to a "
         "chi-square distribution with d (d + 1) (d + 2)/6 degrees of freedom (upper tail)." + _MARDIA_NOTE +
         " Check: reference values (an independent NumPy implementation of Mardia's statistics and the chi-square "
         "tail, and the univariate identity b1 = squared moment skewness) in tests.")(
    _mardia_impl("MardiaSkewnessTest", "skew"))
fn("MardiaKurtosisTest", area=AREA, status="added",
   notes="MardiaKurtosisTest[data] / [data, dist] / [data, dist, prop]: Mardia's (1970) multivariate kurtosis b2 = "
         "Sum[(z_j . S^-1 . z_j)^2]/n; the test statistic z = (b2 - d (d + 2))/Sqrt[8 d (d + 2)/n] is referred to "
         "the standard normal distribution (two-sided)." + _MARDIA_NOTE + " Check: reference values (independent "
         "NumPy implementation; the univariate identity b2 = moment kurtosis) in tests.")(
    _mardia_impl("MardiaKurtosisTest", "kurt"))
fn("MardiaCombinedTest", area=AREA, status="added",
   notes="MardiaCombinedTest[data] / [data, dist] / [data, dist, prop]: pools the two Mardia statistics as the "
         "omnibus statistic n b1/6 + z^2 (skewness chi-square plus the squared kurtosis z), referred to a "
         "chi-square distribution with d (d + 1) (d + 2)/6 + 1 degrees of freedom. Wolfram documents only that the "
         "test pools the skewness and kurtosis results, so the pooling rule is this implementation's (stated "
         "here)." + _MARDIA_NOTE + " Check: reference (the statistic equals MardiaSkewnessTest's plus the square "
         "of MardiaKurtosisTest's, tested; chi-square tail from an independent implementation).")(
    _mardia_impl("MardiaCombinedTest", "comb"))


# ─────────────────────────── Baringhaus-Henze (BHEP / Henze-Zirkler) ───────────────────────────

def hz_beta(n, d):
    """Henze-Zirkler (1990) smoothing parameter"""
    return (1 / math.sqrt(2)) * ((2 * d + 1) * n / 4) ** (1 / (d + 4))


def bhep(a, beta, fname):
    """the BHEP statistic T_beta (Henze & Zirkler 1990, eq. 2.3) with the ML covariance"""
    n, d = a.shape
    G = _mahal_gram(a, fname)
    dj = np.diag(G)
    Djk = dj[:, None] + dj[None, :] - 2 * G
    b2 = beta * beta
    t1 = float(np.sum(np.exp(-b2 / 2 * Djk))) / n
    t2 = 2 * (1 + b2) ** (-d / 2) * float(np.sum(np.exp(-b2 / (2 * (1 + b2)) * dj)))
    t3 = n * (1 + 2 * b2) ** (-d / 2)
    return t1 - t2 + t3


def bhep_p(T, beta, d):
    """lognormal approximation of the null distribution (Henze & Zirkler 1990, Theorem 2.1 moments)"""
    b2 = beta * beta
    a = 1 + 2 * b2
    w = (1 + b2) * (1 + 3 * b2)
    mu = 1 - a ** (-d / 2) * (1 + d * b2 / a + d * (d + 2) * b2 ** 2 / (2 * a ** 2))
    s2 = (2 * (1 + 4 * b2) ** (-d / 2) + 2 * a ** (-d) * (1 + 2 * d * b2 ** 2 / a ** 2 + 3 * d * (d + 2) * b2 ** 4 /
                                                           (4 * a ** 4))
          - 4 * w ** (-d / 2) * (1 + 3 * d * b2 ** 2 / (2 * w) + d * (d + 2) * b2 ** 4 / (2 * w ** 2)))
    lm = math.log(math.sqrt(mu ** 4 / (s2 + mu ** 2)))
    ls = math.sqrt(math.log((s2 + mu ** 2) / mu ** 2))
    if T <= 0:
        return 1.0
    return SM.norm_sf((math.log(T) - lm) / ls)


@fn("BaringhausHenzeTest", area=AREA, status="added",
    notes="BaringhausHenzeTest[data] / [data, dist] / [data, dist, prop]: the BHEP (Baringhaus-Henze-Epps-Pulley) "
          "statistic T_beta = (1/n) Sum[E^(-beta^2 D_jk/2)] - 2 (1 + beta^2)^(-d/2) Sum[E^(-beta^2 D_j/(2 (1 + "
          "beta^2)))] + n (1 + 2 beta^2)^(-d/2) of the standardized data (sample mean, ML covariance with divisor "
          "n; D_jk, D_j squared Mahalanobis distances), with the Henze-Zirkler smoothing parameter beta = (1/Sqrt[2]) "
          "((2 d + 1) n/4)^(1/(d + 4)) by default (as documented) or Method -> {Automatic, \"SmoothingParameter\" -> "
          "beta}. The p-value uses the lognormal approximation of the null distribution with the exact mean and "
          "variance of Henze & Zirkler (1990) (the method of R's MVN::hz); Wolfram does not document its p-value "
          "method. Properties \"PValue\" (default), \"TestStatistic\", \"TestData\", \"ShortTestConclusion\"; option "
          "SignificanceLevel. Fully specified null distributions, Method -> \"MonteCarlo\", \"FittedDistribution\" "
          "and table properties raise NotImplementedError; a singular sample covariance ValueError. Check: reference "
          "values (independent NumPy implementation of the Henze-Zirkler statistic and p-value, as R MVN::hz) and "
          "affine invariance of the statistic in tests.")
def _bh_test(x: ListOf[Real] | MatrixOf[Real], *rest: Expr | String | Option) -> Real | Expr:
    name = "BaringhausHenzeTest"
    d, prop, o = _test_args(rest, name)
    _, sub = _method(o, name)
    _normal_null(d, name)
    a = mv_data(x, name)
    n, dim = a.shape
    beta = hz_beta(n, dim)
    for k, v in sub.items():
        if k != "SmoothingParameter":
            raise ValueError(f"{name}: unknown Method suboption {k!r}")
        beta = SM.fl(SM._num(v, name, "SmoothingParameter"))
        if not beta > 0:
            raise ValueError(f"{name}: the smoothing parameter must be positive")
    T = bhep(a, beta, name)
    return SM.test_result(name, T, bhep_p(T, beta, dim), prop, o)


# ─────────────────────────── Pearson chi-square goodness of fit ───────────────────────────

def pearson_bins(n):
    """number of equiprobable classes: Moore's (1986) rule k = Ceiling[2 n^(2/5)]"""
    return max(2, math.ceil(2 * n ** 0.4))


@fn("PearsonChiSquareTest", area=AREA, status="added",
    notes="PearsonChiSquareTest[data, dist] / [data, dist, prop] for univariate data and a fully specified "
          "continuous univariate distribution: the data are classified into k classes of equal probability under "
          "dist (via the probability-integral transforms CDF[dist, x_i]); the statistic Sum[(O_j - n/k)^2/(n/k)] is "
          "referred to a chi-square distribution with k - 1 degrees of freedom. Wolfram documents the "
          "equal-probability bins but not their number; here k = Ceiling[2 n^(2/5)] (Moore 1986), at least 2. "
          "Properties \"PValue\" (default), \"TestStatistic\", \"TestData\", \"DegreesOfFreedom\", "
          "\"ShortTestConclusion\"; option SignificanceLevel. Distributions with symbolic (estimated) parameters, "
          "discrete distributions, data as the reference (two-sample), multivariate data, Method -> \"MonteCarlo\" "
          "and table properties raise NotImplementedError. Check: reference values (an independent count of the "
          "equiprobable classes from scipy.stats quantiles and the chi-square tail) in tests.")
def _pearson_chisq(x: ListOf[Real], *rest: Expr | String | Option) -> Real | Expr:
    name = "PearsonChiSquareTest"
    d, prop, o = _test_args(rest, name)
    _method(o, name)
    if isinstance(x, Symbol):
        raise Unevaluated()
    if d is None or _sym(d, "Automatic"):
        raise NotImplementedError(f"{name}: testing against an estimated (Automatic) distribution is not supported; "
                                  "give a fully specified distribution")
    if is_list(d):
        raise NotImplementedError(f"{name}: the two-sample form (data as the reference) is not supported")
    if is_list(x) and x.args and any(is_list(v) for v in x.args):
        raise NotImplementedError(f"{name}: multivariate data are not supported")
    v = SM._uni_data(x, name, 2)
    if W.is_dist(d) and W.dist_of(d, name).discrete:
        raise NotImplementedError(f"{name}: discrete distributions are not supported (no equiprobable classes)")
    u = SM._cdf_values(v, d, name)
    n = len(u)
    k = pearson_bins(n)
    counts = [0] * k
    for t in u:
        counts[min(k - 1, max(0, int(math.floor(t * k))))] += 1
    e = n / k
    stat = math.fsum((c - e) ** 2 / e for c in counts)
    return SM.test_result(name, stat, SM.chi2_sf(stat, k - 1), prop, o, k - 1)


# ─────────────────────────── unit-root tests ───────────────────────────
# MacKinnon (1994), "Approximate asymptotic distribution functions for unit-root and cointegration tests",
# J. Business & Economic Statistics 12, 167-176: Tables 3-4 (tau) and 5-6 (normalized bias z), N = 1 rows
# (the same coefficients as statsmodels.tsa.adfvalues).

_TAU = {
    "n": dict(star=-1.04, mn=-19.04, mx=math.inf, small=(0.6344, 1.2378, 3.2496e-2),
              large=(0.4797, 0.93557, -0.06999, 0.033066)),
    "c": dict(star=-1.61, mn=-18.83, mx=2.74, small=(2.1659, 1.4412, 3.8269e-2),
              large=(1.7339, 0.93202, -0.12745, -0.010368)),
    "ct": dict(star=-2.89, mn=-16.18, mx=0.7, small=(3.2512, 1.6047, 4.9588e-2),
               large=(2.5261, 0.61654, -0.37956, -0.060285)),
}
_Z = {
    "n": dict(star=-2.9, small=(.0342, -.6376, 0, -.03872), large=(0.4927, 0.6906, 0.132331, 0.012099, 0)),
    "c": dict(star=-8.9, small=(2.2142, -1.7863, .32828, -.07727),
              large=(1.717, 0.55243, 0.043463, 0.0016671, 0)),
    "ct": dict(star=-15.0, small=(4.6476, -2.8932, 0.5832, -0.0999),
               large=(2.7117, 0.45731, 0.022868, 0.0006362, 0.000005)),
}


def _poly(c, x):
    return math.fsum(ci * x ** i for i, ci in enumerate(c))


def mackinnon_tau_p(t, det):
    tb = _TAU[det]
    if t > tb["mx"]:
        return 1.0
    if t < tb["mn"]:
        return 0.0
    c = tb["small"] if t <= tb["star"] else tb["large"]
    return SM.norm_cdf(_poly(c, t))


def mackinnon_z_p(z, det):
    zb = _Z[det]
    if z <= zb["star"]:
        return SM.norm_cdf(_poly(zb["small"], math.log(abs(z))))
    return SM.norm_cdf(_poly(zb["large"], z))


_URT_TESTS = ("DickeyFullerF", "DickeyFullerT", "PhillipsPerronF", "PhillipsPerronT")


def _urt_model(m, fname):
    """(r, det) with det "n" (alpha = beta = 0), "c" (beta = 0) or "ct" (general)"""
    if m is None or _sym(m, "Automatic"):
        return 1, "n"
    if isinstance(m, int) and not isinstance(m, bool):
        if m < 1:
            raise ValueError(f"{fname}: the AR order must be a positive integer")
        return m, "n"
    if m == "Constant":
        return 1, "c"
    if m == "Drift":
        return 1, "ct"
    if is_list(m) and len(m.args) == 2 and m.args[0] in ("Constant", "Drift"):
        r = m.args[1]
        if isinstance(r, Symbol):
            raise Unevaluated()
        if not (isinstance(r, int) and not isinstance(r, bool) and r >= 1):
            raise ValueError(f"{fname}: the AR order must be a positive integer")
        return r, ("c" if m.args[0] == "Constant" else "ct")
    if isinstance(m, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: invalid model specification {m} (use Automatic, r, \"Constant\", \"Drift\", "
                     "{\"Constant\", r} or {\"Drift\", r})")


def _urt_series(x, fname):
    from . import wl_processes as P
    if is_list(x) and x.args and any(is_list(v) and len(v.args) != 2 for v in x.args):
        raise TypeError(f"{fname}: the data must be a list of numbers, time-value pairs, a TimeSeries or "
                        "TemporalData")
    paths = P._series_data(x, fname)
    if len(paths) != 1:
        raise NotImplementedError(f"{fname}: several paths are not supported")
    return np.array([float(v) for _, v in paths[0]])


def _ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return beta, res


def _adf(y, r, det, fname):
    """ADF regression dy_t = (rho - 1) y_(t-1) + Sum c_j dy_(t-j) + det; returns (tau, z, nobs)"""
    N = len(y)
    dy = np.diff(y)
    nobs = N - r
    k = 1 + (r - 1) + (0 if det == "n" else 1 if det == "c" else 2)
    if nobs - k < 2:
        raise ValueError(f"{fname}: too few observations ({N}) for the model")
    rows = []
    for t in range(r, N):
        row = [y[t - 1]] + [dy[t - 1 - j] for j in range(1, r)]
        if det in ("c", "ct"):
            row.append(1.0)
        if det == "ct":
            row.append(float(t))
        rows.append(row)
    X = np.array(rows)
    yy = dy[r - 1:]
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError(f"{fname}: the regression is degenerate (constant or perfectly trending data)")
    beta, res = _ols(X, yy)
    s2 = float(res @ res) / (nobs - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(cov[0, 0])
    tau = float(beta[0]) / se
    gsum = float(np.sum(beta[1:r])) if r > 1 else 0.0
    z = nobs * float(beta[0]) / (1 - gsum)
    return tau, z, nobs


def nw_lags(n):
    """Newey-West truncation lag used for the Phillips-Perron long-run variance: Ceiling[12 (n/100)^(1/4)]"""
    return int(math.ceil(12 * (n / 100) ** 0.25))


def _pp(y, det, fname):
    """Phillips-Perron (1988) Z_t and Z_alpha from the AR(1) regression with a Bartlett-kernel long-run variance"""
    N = len(y)
    n = N - 1
    k = 1 + (0 if det == "n" else 1 if det == "c" else 2)
    if n - k < 2:
        raise ValueError(f"{fname}: too few observations ({N}) for the model")
    rows = []
    for t in range(1, N):
        row = [y[t - 1]]
        if det in ("c", "ct"):
            row.append(1.0)
        if det == "ct":
            row.append(float(t))
        rows.append(row)
    X = np.array(rows)
    yy = y[1:]
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError(f"{fname}: the regression is degenerate (constant or perfectly trending data)")
    beta, u = _ols(X, yy)
    s2 = float(u @ u) / (n - k)
    se = math.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    rho = float(beta[0])
    tau = (rho - 1) / se
    g0 = float(u @ u) / n
    lags = min(nw_lags(n), n - 1)
    lam2 = g0 + 2 * math.fsum((1 - j / (lags + 1)) * float(u[j:] @ u[:-j]) / n for j in range(1, lags + 1))
    if lam2 <= 0:
        raise ArithmeticError(f"{fname}: the long-run variance estimate is not positive")
    lam = math.sqrt(lam2)
    zt = math.sqrt(g0 / lam2) * tau - (lam2 - g0) / (2 * lam) * n * se / math.sqrt(s2)
    za = n * (rho - 1) - n * n * se * se / (2 * s2) * (lam2 - g0)
    return zt, za


@fn("UnitRootTest", area=AREA, status="added",
    notes="UnitRootTest[data], [data, model], [data, model, test], [data, model, prop] and [data, model, {prop, "
          "test}] for a list of values, time-value pairs, a TimeSeries or single-path TemporalData. Models as "
          "documented: Automatic (r = 1, alpha = beta = 0), r (alpha = beta = 0), \"Constant\" (r = 1, beta = 0), "
          "\"Drift\" (r = 1, constant and linear trend), {\"Constant\", r}, {\"Drift\", r}. Tests: "
          "\"DickeyFullerF\" (default; the normalized bias n (a1 - 1), divided by 1 - Sum of the lagged-difference "
          "coefficients when r > 1), \"DickeyFullerT\" ((a1 - 1)/se), \"PhillipsPerronF\" and \"PhillipsPerronT\" "
          "(Phillips-Perron 1988 Z_alpha and Z_t from the AR(1) regression, Bartlett-kernel long-run variance with "
          "Ceiling[12 (n/100)^(1/4)] lags; r = 1 only); n is the number of regression observations. p-values (left "
          "tail) use MacKinnon's (1994) asymptotic response surfaces (Tables 3-6, as statsmodels' mackinnonp). "
          "Properties \"PValue\", \"TestStatistic\", \"TestData\", \"ShortTestConclusion\", \"AllTests\", "
          "\"AutomaticTest\"; All gives every applicable test. Phillips-Perron tests with r > 1, several paths and "
          "the HypothesisTestData/table properties raise NotImplementedError; too short or degenerate series "
          "ValueError; symbolic data stay unevaluated. Check: reference values (statsmodels adfuller "
          "statistics and p-values, Fuller's 5% asymptotic critical values -8.1/-14.1/-21.8 and -1.95/-2.86/-3.41) "
          "in tests.")
def _unit_root(x: ListOf[Real] | Expr, model: Expr = None, *rest: String | ListOf[String] | Option) -> Real | Expr:
    name = "UnitRootTest"
    rest = list(rest)
    if model is not None and SM._is_rule(model) and isinstance(model.args[0], Symbol):
        rest.insert(0, model)
        model = None
    args, o = SM.opts(rest, ("SignificanceLevel",), name)
    if isinstance(model, str) and (model in _URT_TESTS or model in SM.TEST_PROPS or model in
                                   ("AllTests", "AutomaticTest", "HypothesisTestData")):
        args.insert(0, model)
        model = None
    r, det = _urt_model(model, name)
    prop, tests, several = None, ["DickeyFullerF"], False
    if args:
        if len(args) > 1:
            raise ValueError(f"{name}: too many arguments")
        spec = args[0]
        if _sym(spec, "All"):
            tests, several = list(_URT_TESTS), True
        elif isinstance(spec, str) and spec in _URT_TESTS:
            tests = [spec]
        elif isinstance(spec, str):
            prop = spec
        elif is_list(spec) and len(spec.args) == 2 and isinstance(spec.args[0], str) and \
                spec.args[0] not in _URT_TESTS:
            prop = spec.args[0]
            t = spec.args[1]
            if _sym(t, "All"):
                tests, several = list(_URT_TESTS), True
            elif isinstance(t, str) and t in _URT_TESTS:
                tests = [t]
            elif is_list(t) and t.args and all(isinstance(q, str) and q in _URT_TESTS for q in t.args):
                tests, several = list(t.args), True
            else:
                raise ValueError(f"{name}: unknown test {t}")
        elif is_list(spec) and spec.args and all(isinstance(q, str) and q in _URT_TESTS for q in spec.args):
            tests, several = list(spec.args), True
        else:
            raise ValueError(f"{name}: invalid test or property specification {spec}")
    if prop == "AutomaticTest":
        return "DickeyFullerF"
    if prop == "AllTests":
        return L(*(_URT_TESTS if r == 1 else _URT_TESTS[:2]))
    if several and r > 1 and any(t.startswith("Phillips") for t in tests):
        raise NotImplementedError(f"{name}: the Phillips-Perron tests are only supported for r = 1")
    y = _urt_series(x, name)
    out = []
    for t in tests:
        if t.startswith("DickeyFuller"):
            tau, z, _ = _adf(y, r, det, name)
        else:
            if r > 1:
                raise NotImplementedError(f"{name}: the Phillips-Perron tests are only supported for r = 1")
            tau, z = _pp(y, det, name)
        if t.endswith("T"):
            out.append((tau, mackinnon_tau_p(tau, det)))
        else:
            out.append((z, mackinnon_z_p(z, det)))
    if prop in ("TestStatistic", "PValue", "TestData", "ShortTestConclusion", None):
        res = [SM.test_result(name, s, p, prop, o) for s, p in out]
        return L(*res) if several else res[0]
    return SM.test_result(name, out[0][0], out[0][1], prop, o)


# ─────────────────────────── PearsonDistribution ───────────────────────────

from . import wl_distributions2 as D2  # noqa: E402


def _base(head, *params):
    return W.dist_of(Expr(head, *(R(sp.nsimplify(p) if not isinstance(p, sp.Float) else p) for p in params)),
                     "PearsonDistribution")


class Pearson(D2.UD):
    """PearsonDistribution[a1, a0, b2, b1, b0] and PearsonDistribution[type, a1, a0, b2, b1, b0]: the density solves
    f'(x) = -(a1 x + a0)/(b2 x^2 + b1 x + b0) f(x).  For numeric parameters the solution is identified with an
    affine image of a standard distribution (Beta, Gamma, Normal, InverseGamma, BetaPrime) or the Pearson type IV
    closed form."""

    def __init__(self, name, a):
        if len(a) == 6:
            typ, pp = a[0], a[1:]
            if _num(typ) and not (typ.is_Integer and 1 <= typ <= 7):
                raise ValueError(f"{name}: the type must be an integer from 1 to 7")
        elif len(a) == 5:
            typ, pp = None, a
        else:
            raise ValueError(f"{name}: expected PearsonDistribution[a1, a0, b2, b1, b0] or PearsonDistribution[type, "
                             "a1, a0, b2, b1, b0]")
        for v in pp:
            D2._scalar(v, name)
            D2._real(v, "parameter", name)
        super().__init__(name, list(a))
        self.typ, self.pp = typ, list(pp)
        self.case = None
        if all(_num(v) for v in pp) and (typ is None or _num(typ)):
            self._setup(name)

    # identification ------------------------------------------------------
    def _setup(self, name):
        a1, a0, b2, b1, b0 = [sp.nsimplify(v) if not isinstance(v, sp.Float) else v for v in self.pp]
        inexact = any(isinstance(v, sp.Float) for v in self.pp)
        self.inexact = inexact
        if b2 != 0:
            c = b2
        elif b1 != 0:
            c = b1
        elif b0 != 0:
            c = b0
        else:
            raise ValueError(f"{name}: b2, b1 and b0 cannot all be zero")
        a1, a0, b2, b1, b0 = (v / c for v in (a1, a0, b2, b1, b0))
        bad = ValueError(f"{name}: the parameters do not define a normalizable density")
        typ = int(self.typ) if self.typ is not None else None
        cands = []
        if b2 == 0 and b1 == 0:          # Q = 1: normal
            if not a1 > 0:
                raise bad
            cands.append((3, ("affine", "NormalDistribution", (-a0 / a1, 1 / sp.sqrt(a1)), 0, 1)))
        elif b2 == 0:                     # Q = x - r: gamma
            r = -b0
            k = -(a1 * r + a0)
            if not (a1 != 0 and k > -1):
                raise bad
            sgn = 1 if a1 > 0 else -1
            cands.append((3, ("affine", "GammaDistribution", (k + 1, 1 / abs(a1)), r, sgn)))
        else:
            disc = b1 ** 2 - 4 * b0
            if disc < 0:                  # complex roots: type IV (VII when symmetric)
                lam = -b1 / 2
                aa = sp.sqrt(b0 - b1 ** 2 / 4)
                m = a1 / 2
                nu = (a0 + a1 * lam) / aa
                if not 2 * m > 1:
                    raise bad
                cands.append((4, ("iv", lam, aa, m, nu)))
                if nu == 0:
                    df = 2 * m - 1
                    cands.append((7, ("affine", "StudentTDistribution", (df,), lam, aa / sp.sqrt(df))))
            elif disc == 0:               # double root: type V
                r = -b1 / 2
                C = a1 * r + a0
                al = a1 - 1
                if not (al > 0 and C != 0):
                    raise bad
                cands.append((5, ("affine", "InverseGammaDistribution", (al, abs(C)), r, -1 if C > 0 else 1)))
            else:                         # two real roots
                sq = sp.sqrt(disc)
                r1, r2 = (-b1 - sq) / 2, (-b1 + sq) / 2
                A = -(a1 * r1 + a0) / (r1 - r2)
                B = -(a1 * r2 + a0) / (r2 - r1)
                w = r2 - r1
                if A > -1 and B > -1:
                    cands.append((1, ("affine", "BetaDistribution", (A + 1, B + 1), r1, w)))
                    if sp.simplify(A - B) == 0:
                        cands.append((2, ("affine", "BetaDistribution", (A + 1, B + 1), r1, w)))
                if A + B < -1:
                    hi_ok = B > -1
                    lo_ok = A > -1
                    if hi_ok and lo_ok:
                        if typ == 6 or not cands:
                            raise NotImplementedError(f"{name}: both tails beyond the roots of the quadratic give a "
                                                      "valid type VI density; the choice is ambiguous")
                    elif hi_ok:
                        cands.append((6, ("affine", "BetaPrimeDistribution", (B + 1, -A - B - 1), r2, w)))
                    elif lo_ok:
                        cands.append((6, ("affine", "BetaPrimeDistribution", (A + 1, -A - B - 1), r1, -w)))
        if not cands:
            raise bad
        order = [4, 1, 6, 3, 5, 2, 7]
        if typ is None:
            cands.sort(key=lambda c: order.index(c[0]))
            chosen = cands[0]
        else:
            ok = [c for c in cands if c[0] == typ]
            if not ok:
                raise ValueError(f"{name}: the parameters are not those of a type {typ} Pearson distribution (they "
                                 f"define type {', '.join(str(c[0]) for c in cands)})")
            chosen = ok[0]
        self.ptype, spec = chosen
        if spec[0] == "affine":
            _, head, params, loc, sc = spec
            if inexact:
                params = tuple(sp.Float(sp.N(p, 20)) for p in params)
                loc, sc = sp.Float(sp.N(loc, 20)), sp.Float(sp.N(sc, 20))
            self.case = ("affine", _base(head, *params), sp.sympify(loc), sp.sympify(sc))
        else:
            _, lam, aa, m, nu = spec
            if inexact:
                lam, aa, m, nu = (sp.Float(sp.N(v, 20)) for v in (lam, aa, m, nu))
            self.case = ("iv", lam, aa, m, nu)

    def _need(self):
        if self.case is None:
            raise Unevaluated()
        return self.case

    def bounds(self):
        c = self._need()
        if c[0] == "iv":
            return -sp.oo, sp.oo
        _, base, loc, sc = c
        lo, hi = base.bounds()
        a, b = loc + sc * lo, loc + sc * hi
        return (a, b) if sc > 0 else (b, a)

    @property
    def open_lo(self):
        return self.case is not None and self.case[0] == "affine"

    @property
    def open_hi(self):
        return self.case is not None and self.case[0] == "affine"

    def pdf(self, x):
        c = self._need()
        if c[0] == "iv":
            _, lam, aa, m, nu = c
            t = (x - lam) / aa
            k = sp.gamma(m + sp.I * nu / 2) * sp.gamma(m - sp.I * nu / 2) / (aa * sp.sqrt(sp.pi) * sp.gamma(m) *
                                                                             sp.gamma(m - sp.Rational(1, 2)))
            if not self.inexact:
                k = sp.simplify(k)
            return k * (1 + t ** 2) ** (-m) * sp.exp(-nu * sp.atan(t))
        _, base, loc, sc = c
        return base.pdf((x - loc) / sc) / abs(sc)

    @property
    def closed_cdf(self):
        return self.case is None or self.case[0] == "affine"

    def cdf(self, x):
        c = self._need()
        if c[0] == "iv":
            return self.num_cdf(x)
        _, base, loc, sc = c
        y = (x - loc) / sc
        F = W.cdf_value(base, y) if _num(y) else base.cdf(y)
        return F if sc > 0 else 1 - F

    def q(self, p):
        c = self._need()
        if c[0] == "iv" or not hasattr(c[1], "q"):
            raise Unevaluated()
        _, base, loc, sc = c
        return loc + sc * (base.q(p) if sc > 0 else base.q(1 - p))

    def mean(self):
        c = self._need()
        if c[0] == "iv":
            _, lam, aa, m, nu = c
            if not m > 1:
                return sp.nan
            return lam - aa * nu / (2 * (m - 1))
        _, base, loc, sc = c
        mu = base.mean()
        return mu if mu in (sp.nan, sp.oo) else loc + sc * mu

    def var(self):
        c = self._need()
        if c[0] == "iv":
            _, lam, aa, m, nu = c
            if not m > sp.Rational(3, 2):
                return sp.oo if m > 1 else sp.nan
            return aa ** 2 * (4 * (m - 1) ** 2 + nu ** 2) / (4 * (m - 1) ** 2 * (2 * m - 3))
        _, base, loc, sc = c
        v = base.var()
        return v if v in (sp.nan, sp.oo) else sc ** 2 * v

    def raw(self, r):
        c = self._need()
        if c[0] == "iv":
            if r == 1:
                return self.mean()
            if r == 2:
                v, m = self.var(), self.mean()
                return v if v in (sp.nan, sp.oo) else v + m ** 2
            raise NotImplementedError(f"Moment: moments of order {r} of a type IV Pearson distribution are not "
                                      "supported")
        _, base, loc, sc = c
        terms = []
        for k in range(r + 1):
            mk = W.raw_moment(base, k) if k else sp.Integer(1)
            if mk in (sp.nan, sp.oo, sp.zoo):
                return sp.oo if mk == sp.oo else sp.nan
            terms.append(sp.binomial(r, k) * loc ** (r - k) * sc ** k * mk)
        return sp.Add(*terms)

    def sample(self, rng):
        c = self._need()
        if c[0] == "iv":
            return D2._invert(self, rng.random())
        _, base, loc, sc = c
        return D2._f(loc) + D2._f(sc) * float(base.sample(rng))


def _num(v):
    return isinstance(v, sp.Basic) and v.is_number


W.TABLE["PearsonDistribution"] = Pearson
W.NEW_DISTS.add("PearsonDistribution")


@fn("PearsonDistribution", area=AREA, status="added",
    notes="PearsonDistribution[a1, a0, b2, b1, b0] and PearsonDistribution[type, a1, a0, b2, b1, b0]: the density "
          "solves f'(x) = -(a1 x + a0)/(b2 x^2 + b1 x + b0) f(x). For numeric parameters the solution is identified "
          "exactly: type 1/2 a rescaled BetaDistribution on the interval between the real roots, type 3 a shifted "
          "GammaDistribution (b2 = 0) or NormalDistribution (b2 = b1 = 0), type 4 the closed-form Pearson IV density "
          "k (1 + ((x - lambda)/a)^2)^-m E^(-nu ArcTan[(x - lambda)/a]) with k = |Gamma[m + I nu/2]|^2/(a Sqrt[Pi] "
          "Gamma[m] Gamma[m - 1/2]) (Heinrich 2004), type 5 a shifted InverseGammaDistribution, type 6 a shifted "
          "BetaPrimeDistribution beyond a root, type 7 a rescaled StudentTDistribution; without a type the first "
          "valid type in the documented order 4, 1, 6, 3, 5, 2, 7 is used. PDF, CDF (closed form except type 4, "
          "which is integrated numerically for machine-number input), InverseCDF, Mean, Variance, Moment and "
          "RandomVariate work through the distribution functions. Parameters that give no normalizable density, "
          "or a type that does not match them, raise ValueError; symbolic parameters stay unevaluated in PDF/CDF "
          "(Wolfram gives the symbolic piecewise forms; not supported). Check: reference (the ODE f' Q + (a1 x + "
          "a0) f = 0 verified symbolically, unit total mass by numerical integration, moments against the base "
          "distributions) in tests.")
def _pearson_ctor(*params: Real) -> Expr:
    W.dist_of(Expr("PearsonDistribution", *params), "PearsonDistribution")
    raise Unevaluated()


# ─────────────────────────── FindDistribution ───────────────────────────

def _lgamma(x):
    return math.lgamma(x)


def _digamma(x):
    return float(mpmath.digamma(x))


def _trigamma(x):
    return float(mpmath.psi(1, x))


def _gamma_mle(v):
    """(shape k, scale theta) maximum-likelihood estimates (Newton on log k - psi(k) = log mean - mean log)"""
    n = len(v)
    m = math.fsum(v) / n
    s = math.log(m) - math.fsum(math.log(x) for x in v) / n
    if s <= 0:
        raise ValueError("degenerate")
    k = (3 - s + math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(100):
        f = math.log(k) - _digamma(k) - s
        fp = 1 / k - _trigamma(k)
        k2 = k - f / fp
        if k2 <= 0:
            k2 = k / 2
        if abs(k2 - k) <= 1e-14 * k:
            k = k2
            break
        k = k2
    if abs(math.log(k) - _digamma(k) - s) > 1e-9:
        raise ArithmeticError("FindDistribution: the gamma maximum-likelihood equation did not converge")
    return k, m / k


def _weibull_mle(v):
    """(shape alpha, scale beta) maximum-likelihood estimates (Newton on the profile equation)"""
    lx = [math.log(x) for x in v]
    n = len(v)
    ml = math.fsum(lx) / n
    sd = math.sqrt(math.fsum((t - ml) ** 2 for t in lx) / n)
    if sd == 0:
        raise ValueError("degenerate")
    a = 1.2 / sd
    for _ in range(200):
        xa = [x ** a for x in v]
        S0 = math.fsum(xa)
        S1 = math.fsum(w * t for w, t in zip(xa, lx))
        S2 = math.fsum(w * t * t for w, t in zip(xa, lx))
        f = S1 / S0 - 1 / a - ml
        fp = (S2 * S0 - S1 * S1) / S0 ** 2 + 1 / a ** 2
        a2 = a - f / fp
        if a2 <= 0:
            a2 = a / 2
        if abs(a2 - a) <= 1e-14 * a:
            a = a2
            break
        a = a2
    xa = [x ** a for x in v]
    S0, S1 = math.fsum(xa), math.fsum(w * t for w, t in zip(xa, lx))
    if abs(S1 / S0 - 1 / a - ml) > 1e-9:
        raise ArithmeticError("FindDistribution: the Weibull maximum-likelihood equation did not converge")
    return a, (S0 / n) ** (1 / a)


def _fit_family(h, v, ints):
    """(params (floats), log-likelihood) of the maximum-likelihood fit, or None when the family does not apply"""
    n = len(v)
    pos = all(x > 0 for x in v)
    nonneg = all(x >= 0 for x in v)
    try:
        if h == "NormalDistribution":
            m = math.fsum(v) / n
            s2 = math.fsum((x - m) ** 2 for x in v) / n
            if s2 <= 0:
                return None
            return (m, math.sqrt(s2)), -n / 2 * math.log(2 * math.pi * s2) - n / 2
        if h == "LogNormalDistribution":
            if not pos:
                return None
            lx = [math.log(x) for x in v]
            m = math.fsum(lx) / n
            s2 = math.fsum((t - m) ** 2 for t in lx) / n
            if s2 <= 0:
                return None
            return (m, math.sqrt(s2)), -n / 2 * math.log(2 * math.pi * s2) - n / 2 - math.fsum(lx)
        if h == "ExponentialDistribution":
            if not nonneg:
                return None
            m = math.fsum(v) / n
            if m <= 0:
                return None
            return (1 / m,), -n * math.log(m) - n
        if h == "GammaDistribution":
            if not pos:
                return None
            k, th = _gamma_mle(v)
            ll = math.fsum((k - 1) * math.log(x) - x / th for x in v) - n * (_lgamma(k) + k * math.log(th))
            return (k, th), ll
        if h == "WeibullDistribution":
            if not pos:
                return None
            a, b = _weibull_mle(v)
            ll = math.fsum(math.log(a / b) + (a - 1) * math.log(x / b) - (x / b) ** a for x in v)
            return (a, b), ll
        if h == "UniformDistribution":
            lo, hi = min(v), max(v)
            if hi <= lo:
                return None
            return ((lo, hi),), -n * math.log(hi - lo)
        if h == "LaplaceDistribution":
            sv = sorted(v)
            med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
            b = math.fsum(abs(x - med) for x in v) / n
            if b <= 0:
                return None
            return (med, b), -n * math.log(2 * b) - n
        if h == "RayleighDistribution":
            if not nonneg:
                return None
            s2 = math.fsum(x * x for x in v) / (2 * n)
            if s2 <= 0 or not pos:
                return None
            return (math.sqrt(s2),), math.fsum(math.log(x / s2) - x * x / (2 * s2) for x in v)
        if not ints:
            return None
        if h == "PoissonDistribution":
            if not nonneg:
                return None
            m = math.fsum(v) / n
            if m <= 0:
                return None
            return (m,), math.fsum(x * math.log(m) - m - _lgamma(x + 1) for x in v)
        if h == "GeometricDistribution":
            if not nonneg:
                return None
            m = math.fsum(v) / n
            if m <= 0:
                return None
            p = 1 / (1 + m)
            return (p,), n * math.log(p) + math.fsum(v) * math.log(1 - p)
        if h == "DiscreteUniformDistribution":
            lo, hi = int(min(v)), int(max(v))
            if hi <= lo:
                return None
            return ((lo, hi),), -n * math.log(hi - lo + 1)
        if h == "BernoulliDistribution":
            if not all(x in (0, 1) for x in v):
                return None
            p = math.fsum(v) / n
            if p in (0.0, 1.0):
                return None
            k = math.fsum(v)
            return (p,), k * math.log(p) + (n - k) * math.log(1 - p)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    return None


_CONT = ("NormalDistribution", "LogNormalDistribution", "ExponentialDistribution", "GammaDistribution",
         "WeibullDistribution", "UniformDistribution", "LaplaceDistribution", "RayleighDistribution")
_DISC = ("PoissonDistribution", "GeometricDistribution", "DiscreteUniformDistribution", "BernoulliDistribution")
_FD_PROPS = ("BIC", "AIC", "HQIC", "LogLikelihood", "PearsonChiSquare", "CramerVonMises")


def _fd_dist(h, params, exact_ints):
    args = []
    for p in params:
        if isinstance(p, tuple):
            args.append(L(*(int(q) if exact_ints else float(q) for q in p)))
        else:
            args.append(float(p))
    return Expr(h, *args)


@fn("FindDistribution", area=AREA, status="added",
    notes="FindDistribution[data], [data, n] (up to n best, best first) and [data, n, prop | {prop, ...}] (rows "
          "{dist, value, ...}) for univariate numeric data. Each candidate family is fitted by maximum likelihood "
          "(closed forms; Newton iterations for the gamma and Weibull shapes) and the candidates are ranked by BIC "
          "= -2 LogLikelihood + k Log[n] (k free parameters). Candidates (TargetFunctions -> Automatic): integer "
          "data use the discrete families Poisson, Geometric, DiscreteUniform and Bernoulli; other data the "
          "continuous families Normal, LogNormal, Exponential, Gamma, Weibull, Uniform, Laplace and Rayleigh "
          "(families whose support excludes the data are skipped); \"Continuous\", \"Discrete\", All and explicit "
          "lists of these heads are accepted. Wolfram's Automatic ranking (BIC with undocumented priors), mixture "
          "and histogram candidates (MaxItems) are not reproduced; the \"Score\" and \"Complexity\" properties, other "
          "target heads, RandomSeeding/PerformanceGoal settings and multivariate data raise NotImplementedError. "
          "Properties: \"BIC\", \"AIC\", \"HQIC\" (-2 LogLikelihood + 2 k Log[Log[n]]), \"LogLikelihood\", "
          "\"PearsonChiSquare\" and \"CramerVonMises\" (the p-values of PearsonChiSquareTest/CramerVonMisesTest "
          "against the fitted distribution, continuous candidates only). Parameters are machine numbers. Check: "
          "reference values (scipy.stats maximum-likelihood fits and log-likelihoods) and the closed-form MLEs in "
          "tests.")
def _find_distribution(x: ListOf[Real], nn: PosInteger = None, prop: String | ListOf[String] = None,
                       *rest: Option) -> Expr:
    name = "FindDistribution"
    rest = list(rest)
    for slot in ("prop", "nn"):
        val = prop if slot == "prop" else nn
        if val is not None and SM._is_rule(val) and isinstance(val.args[0], Symbol):
            rest.insert(0, val)
            if slot == "prop":
                prop = None
            else:
                nn = None
    _, o = SM.opts(rest, ("TargetFunctions", "MaxItems", "PerformanceGoal", "RandomSeeding"), name)
    for k in ("PerformanceGoal", "RandomSeeding"):
        if k in o and not _sym(o[k], "Automatic"):
            raise NotImplementedError(f"{name}: the {k} option is not supported")
    if isinstance(x, Symbol):
        raise Unevaluated()
    if is_list(x) and x.args and any(is_list(v) for v in x.args):
        raise NotImplementedError(f"{name}: multivariate data are not supported")
    vals, items = SM.vec(x, name, min_len=2)
    v = [SM.fl(q) for q in vals]
    ints = all(isinstance(q, sp.Integer) for q in vals)
    tf = o.get("TargetFunctions", Symbol("Automatic"))
    if _sym(tf, "Automatic"):
        fams = _DISC if ints else _CONT
    elif tf == "Continuous":
        fams = _CONT
    elif tf == "Discrete":
        fams = _DISC
    elif _sym(tf, "All"):
        fams = _CONT + (_DISC if ints else ())
    elif is_list(tf) and tf.args:
        fams = []
        for h in tf.args:
            hn = h.name if isinstance(h, Symbol) else None
            if hn not in _CONT + _DISC:
                raise NotImplementedError(f"{name}: the target {h} is not supported (supported: "
                                          f"{', '.join(_CONT + _DISC)})")
            fams.append(hn)
    else:
        raise ValueError(f"{name}: invalid TargetFunctions setting {tf}")
    if nn is None:
        count = 1
    else:
        if isinstance(nn, Symbol):
            raise Unevaluated()
        if not (isinstance(nn, int) and not isinstance(nn, bool)):
            raise TypeError(f"{name}: the number of distributions must be a positive integer")
        if nn < 1:
            raise ValueError(f"{name}: the number of distributions must be positive")
        count = nn
    if prop is not None:
        props = list(prop.args) if is_list(prop) else ([*_FD_PROPS] if _sym(prop, "All") else [prop])
        for q in props:
            if q in ("Score", "Complexity"):
                raise NotImplementedError(f'{name}: the property "{q}" (Wolfram\'s internal score) is not supported')
            if q not in _FD_PROPS:
                raise ValueError(f"{name}: unknown property {q!r}")
    n = len(v)
    fits = []
    for h in fams:
        r = _fit_family(h, v, ints)
        if r is None:
            continue
        params, ll = r
        k = sum(len(p) if isinstance(p, tuple) else 1 for p in params)
        fits.append((-2 * ll + k * math.log(n), h, params, ll, k))
    if not fits:
        raise ValueError(f"{name}: no candidate family fits the data")
    fits.sort(key=lambda f: (f[0], f[1]))
    fits = fits[:count]
    dists = [_fd_dist(h, params, ints) for _, h, params, _, _ in fits]
    if prop is None:
        return dists[0] if nn is None else L(*dists)

    def pval(test, dist):
        if W.dist_of(dist, name).discrete:
            raise NotImplementedError(f"{name}: the {test} property is only available for continuous candidates")
        return _ev(Expr(test + "Test", x, dist))
    rows = []
    for (bic, h, params, ll, k), dist in zip(fits, dists):
        row = [dist]
        for q in props:
            row.append({"BIC": lambda: bic, "AIC": lambda: -2 * ll + 2 * k,
                        "HQIC": lambda: -2 * ll + 2 * k * math.log(math.log(n)), "LogLikelihood": lambda: ll,
                        "PearsonChiSquare": lambda: pval("PearsonChiSquare", dist),
                        "CramerVonMises": lambda: pval("CramerVonMises", dist)}[q]())
        rows.append(L(*row))
    return L(*rows)


# ─────────────────────────── Dendrogram ───────────────────────────

from . import wl_statmodels_fit as FIT  # noqa: E402

_ORIENT = ("Top", "Bottom", "Left", "Right")


def _tree_from_graph(g, fname):
    """(children {v: [c...]}, height {v: h}, labels {leaf: label}, root) from a ClusteringTree-style Graph"""
    if len(g.args) < 2 or not (is_list(g.args[0]) and is_list(g.args[1])):
        raise ValueError(f"{fname}: expected a tree Graph[{{v...}}, {{edges...}}, VertexWeight -> ...]")
    verts = list(g.args[0].args)
    children = {v: [] for v in verts}
    parents = {}
    for e in g.args[1].args:
        if not (isinstance(e, Expr) and e.head in ("DirectedEdge", "UndirectedEdge", "Rule") and len(e.args) == 2):
            raise ValueError(f"{fname}: invalid edge {e}")
        a, b = e.args
        children[a].append(b)
        if b in parents:
            raise ValueError(f"{fname}: the graph is not a rooted tree")
        parents[b] = a
    roots = [v for v in verts if v not in parents]
    if len(roots) != 1 or len(parents) != len(verts) - 1:
        raise ValueError(f"{fname}: the graph is not a rooted tree")
    height, labels = {}, {}
    for o in g.args[2:]:
        if SM._is_rule(o) and _sym(o.args[0], "VertexWeight"):
            w = o.args[1]
            if is_list(w) and all(SM._is_rule(r) for r in w.args):
                for r in w.args:
                    height[r.args[0]] = SM.fl(SM._num(r.args[1], fname, "vertex weight"))
            elif is_list(w) and len(w.args) == len(verts):
                for v, h in zip(verts, w.args):
                    height[v] = SM.fl(SM._num(h, fname, "vertex weight"))
            else:
                raise ValueError(f"{fname}: invalid VertexWeight setting")
        elif SM._is_rule(o) and _sym(o.args[0], "VertexLabels"):
            if is_list(o.args[1]):
                for r in o.args[1].args:
                    if SM._is_rule(r):
                        labels[r.args[0]] = r.args[1]
    if set(height) != set(verts):
        raise ValueError(f"{fname}: Dendrogram needs a tree weighted on every vertex (VertexWeight)")
    for v in verts:
        if not children[v]:
            labels.setdefault(v, v)
    return children, height, labels, roots[0]


def _layout(children, height, labels, root):
    """leaf x positions 1, 2, ... in depth-first order and the drawing primitives (Top orientation: leaves at 0)"""
    xs = {}
    lines = []
    leaves = []

    def walk(v):
        if not children[v]:
            leaves.append(v)
            xs[v] = float(len(leaves))
            return xs[v], height[v]
        pts = [walk(c) for c in children[v]]
        h = height[v]
        for (x, y) in pts:
            lines.append([(x, y), (x, h)])
        lines.append([(min(p[0] for p in pts), h), (max(p[0] for p in pts), h)])
        xs[v] = sum(p[0] for p in pts) / len(pts)
        return xs[v], h
    walk(root)
    return lines, [(xs[v], labels[v]) for v in leaves]


def _orient(p, o):
    x, y = p
    if o == "Top":
        return (x, y)
    if o == "Bottom":
        return (x, -y)
    if o == "Left":
        return (y, -x)
    return (-y, -x)


@fn("Dendrogram", area=AREA, status="added",
    notes="Dendrogram[{e1, ...}], [{e1 -> v1, ...}], [{e1, ...} -> {v1, ...}], [<|label1 -> e1, ...|>], [data, "
          "orientation] (Top (default), Bottom, Left, Right) and Dendrogram[tree] for a vertex-weighted tree Graph "
          "(as returned by ClusteringTree). The hierarchy is the agglomerative clustering of ClusteringTree "
          "(Lance-Williams updates; ClusterDissimilarityFunction \"Single\", \"Complete\", \"Average\", "
          "\"WeightedAverage\", \"Centroid\", \"Median\", \"Ward\"; DistanceFunction as in FindClusters). The result "
          "is Graphics[{{Line[...] ...}, {Text[label, pos, offset] ...}}, opts]: leaves at positions 1, 2, ... in "
          "the tree order (lower-index element first) at height 0, each merge drawn at its dissimilarity. The "
          "Automatic linkage is undocumented and raises NotImplementedError (give ClusterDissimilarityFunction), as "
          "in ClusteringTree; FeatureExtractor, custom linkage functions and images also raise NotImplementedError. "
          "Graphics options are passed through. Check: reference (merge heights equal ClusteringTree's vertex "
          "weights, which match scipy.cluster.hierarchy.linkage; leaf order and coordinates checked in tests).")
def _dendrogram(x: ListOf[Expr] | Rule | Expr, orientation: Expr = None, *rest: Option) -> Expr:
    name = "Dendrogram"
    rest = list(rest)
    if orientation is not None and SM._is_rule(orientation) and isinstance(orientation.args[0], Symbol):
        rest.insert(0, orientation)
        orientation = None
    o_ = "Top" if orientation is None else (orientation.name if isinstance(orientation, Symbol) else orientation)
    if o_ not in _ORIENT:
        raise ValueError(f"{name}: the orientation must be Top, Bottom, Left or Right, not {orientation}")
    from .wl_graphics_core import GRAPHICS_OPTIONS
    gnames = set(GRAPHICS_OPTIONS[2])
    copts = []
    gopts = []
    for r in rest:
        if not (SM._is_rule(r) and isinstance(r.args[0], Symbol)):
            raise ValueError(f"{name}: unexpected argument {r}")
        if r.args[0].name in gnames:
            gopts.append(r)
        else:
            copts.append(r)
    _, o = SM.opts(copts, FIT._CL_OPTS, name)
    if "FeatureExtractor" in o and not _sym(o["FeatureExtractor"], "Automatic", "Identity"):
        raise NotImplementedError(f"{name}: the FeatureExtractor option is not supported")
    if isinstance(x, Expr) and x.head == "Graph":
        children, height, labels, root = _tree_from_graph(x, name)
    else:
        FIT._cl_common(o, name)
        cdf = o.get("ClusterDissimilarityFunction")
        if cdf is None or SM.str_or_sym(cdf) == "Automatic":
            if isinstance(x, Symbol):
                raise Unevaluated()
            raise NotImplementedError(f"{name}: the Automatic linkage is undocumented; give "
                                      "ClusterDissimilarityFunction (e.g. \"Single\")")
        link = FIT._method_linkage(o, name, need_method=False)
        data, labs = FIT._cluster_input(x, name)
        elems = FIT._elements(data, name)
        n = len(elems)
        if n < 2:
            raise ValueError(f"{name}: at least two elements are needed")
        D = FIT._dmatrix(elems, FIT._dist_fn(o.get("DistanceFunction"), name, [e[1] for e in elems]))
        merges = FIT._agglomerate(D, link, name)
        children, height, members = {}, {}, {i: [i] for i in range(n)}
        for i in range(n):
            children[i], height[i] = [], 0.0
        for t, (a, b, h, _) in enumerate(merges):
            c = n + t
            members[c] = sorted(members[a] + members[b])
            children[c] = sorted((a, b), key=lambda q: members[q][0])
            height[c] = float(h)
        labels = {i: labs[i] for i in range(n)}
        root = n + len(merges) - 1
    lines, leaves = _layout(children, height, labels, root)
    offset = {"Top": L(0, 1), "Bottom": L(0, -1), "Left": L(1, 0), "Right": L(-1, 0)}[o_]
    prims = [Expr("Line", L(*(L(*_orient(p, o_)) for p in seg))) for seg in lines]
    texts = [Expr("Text", lab, L(*_orient((xx, 0.0), o_)), offset) for xx, lab in leaves]
    return Expr("Graphics", L(L(*prims), L(*texts)), *gopts)
