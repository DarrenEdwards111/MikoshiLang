"""Wolfram batch W18: statistical models -- robust estimators, dependence measures, hypothesis tests, uncertainty
(Around), likelihoods and the option symbols of the fitting and testing functions.

Model fitting (LinearModelFit, NonlinearModelFit, GeneralizedLinearModelFit, CoxModelFit, SurvivalModelFit, ...)
and clustering live in ``wl_statmodels_fit``.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).  Hypothesis tests return the p-value by default and support the properties "PValue",
"TestStatistic", "TestData", "DegreesOfFreedom" (where the test has one) and "ShortTestConclusion"; the
"HypothesisTestData" object and the formatted "...Table" properties raise NotImplementedError.  p-values are
machine numbers; they are computed with mpmath (normal, Student t, chi-square, F, binomial and exact rank
distributions), so no test needs SciPy unless its notes say so.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import mpmath
import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, record, R, L, rule, S, is_list, numeric_out, has_float)
from . import wl_stats as W

AREA = "statistical models (W18)"
INDET = Symbol("Indeterminate")


# ─────────────────────────── generic helpers ───────────────────────────

def ev(x):
    from ..evaluate import evaluate
    return evaluate(x, 1)


def _is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def opts(args, names, fname):
    """Split trailing Name -> value options; an unknown option name raises ValueError."""
    args = list(args)
    out = {}
    while args and _is_rule(args[-1]) and isinstance(args[-1].args[0], Symbol):
        r = args.pop()
        nm = r.args[0].name
        if nm not in names:
            raise ValueError(f"{fname}: unknown option {nm}")
        out[nm] = r.args[1]
    return args, out


def _num(x, fname, what="value", real=True):
    """SymPy number for a numeric argument: str/bool -> TypeError, symbolic -> Unevaluated."""
    if isinstance(x, (bool, str)):
        raise TypeError(f"{fname}: {what} must be a number, not {x!r}")
    if isinstance(x, (int, float)):
        return sp.sympify(x)
    if isinstance(x, Expr) and x.head == "List":
        raise TypeError(f"{fname}: {what} must be a number, not a list")
    try:
        v = S(x)
    except Exception:
        raise TypeError(f"{fname}: {what} {x!r} must be a number")
    if not isinstance(v, sp.Basic):
        raise TypeError(f"{fname}: {what} {x!r} must be a number")
    if v.free_symbols:
        raise Unevaluated()
    if not v.is_number:
        raise TypeError(f"{fname}: {what} {x!r} must be a number")
    if real and v.is_real is False:
        raise ValueError(f"{fname}: {what} {x!r} must be a real number")
    return v


def vec(x, fname, what="data", min_len=1, real=True):
    """A list of numbers -> (SymPy values, original items).  Non-list -> TypeError (symbol -> Unevaluated)."""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x):
        raise TypeError(f"{fname}: {what} must be a list of numbers")
    items = list(x.args)
    if any(is_list(v) for v in items):
        raise TypeError(f"{fname}: {what} must be a vector (a flat list of numbers)")
    if len(items) < min_len:
        raise ValueError(f"{fname}: {what} needs at least {min_len} element(s)")
    return [_num(v, fname, "data value", real) for v in items], items


def is_matrix(x):
    return is_list(x) and len(x.args) > 0 and all(is_list(r) for r in x.args) and \
        len({len(r.args) for r in x.args}) == 1 and len(x.args[0].args) > 0


def mat(x, fname, what="data", min_rows=1):
    """A numeric matrix -> list of rows of SymPy numbers."""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_matrix(x):
        raise TypeError(f"{fname}: {what} must be a rectangular matrix of numbers")
    if len(x.args) < min_rows:
        raise ValueError(f"{fname}: {what} needs at least {min_rows} rows")
    return [[_num(v, fname, "data value") for v in r.args] for r in x.args]


def columns(rows):
    return [list(c) for c in zip(*rows)]


def inexact(items):
    return any(has_float(i) for i in items)


def out(v, items):
    """Exact result for exact data, machine/bignum number for inexact data."""
    return numeric_out(v, *items)


def fl(v):
    return float(sp.N(v, 17)) if isinstance(v, sp.Basic) else float(v)


def mp(v):
    """mpmath value of a SymPy/Python number at the current precision"""
    if isinstance(v, sp.Basic):
        return mpmath.mpf(str(sp.N(v, mpmath.mp.dps + 5)))
    return mpmath.mpf(v)


def mean(v):
    return sp.Add(*v) / len(v)


def svar(v):
    m = mean(v)
    return sp.Add(*((x - m) ** 2 for x in v)) / (len(v) - 1)


def median(v):
    s = sorted(v, key=lambda t: fl(t))
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def per_column(x, fname, f, *rest):
    """Wolfram's column-wise convention: f[{{x1, y1}, ...}] = {f[{x1, ...}], f[{y1, ...}]}."""
    if is_matrix(x):
        return L(*(f(L(*c), *rest) for c in columns([list(r.args) for r in x.args])))
    return f(x, *rest)


def ranks(v):
    """mid-ranks (1-based) of the numbers v, as Fractions"""
    fv = [fl(a) for a in v]
    order = sorted(range(len(v)), key=lambda i: fv[i])
    r = [Fraction(0)] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and fv[order[j + 1]] == fv[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = Fraction(i + j + 2, 2)
        i = j + 1
    return r


def has_ties(v):
    fv = [fl(a) for a in v]
    return len(set(fv)) != len(fv)


def str_or_sym(x):
    if isinstance(x, str):
        return x
    if isinstance(x, Symbol):
        return x.name
    return None


# ─────────────────────────── distribution functions for p-values (mpmath) ───────────────────────────

def norm_cdf(z):
    return float(mpmath.ncdf(z))


def norm_sf(z):
    return float(mpmath.ncdf(-z))


def t_sf(t, df):
    """P(T > t) for Student t with df degrees of freedom"""
    t = mpmath.mpf(t)
    x = df / (df + t * t)
    tail = mpmath.betainc(mpmath.mpf(df) / 2, mpmath.mpf(1) / 2, 0, x, regularized=True) / 2
    return float(tail if t >= 0 else 1 - tail)


def chi2_sf(x, k):
    if x <= 0:
        return 1.0
    return float(mpmath.gammainc(mpmath.mpf(k) / 2, mpmath.mpf(x) / 2, mpmath.inf, regularized=True))


def f_sf(x, d1, d2):
    if x <= 0:
        return 1.0
    x = mpmath.mpf(x)
    return float(mpmath.betainc(mpmath.mpf(d2) / 2, mpmath.mpf(d1) / 2, 0, d2 / (d2 + d1 * x), regularized=True))


def binom_cdf(k, n, p=Fraction(1, 2)):
    """exact P(X <= k), X ~ Binomial(n, p)"""
    p = Fraction(p)
    return sum(Fraction(math.comb(n, i)) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1)) if k >= 0 else 0


HYP = ("Unequal", "Less", "Greater")


def alt_of(o, fname):
    a = o.get("AlternativeHypothesis", "Unequal")
    a = str_or_sym(a) if not isinstance(a, str) else a
    if a not in HYP:
        raise ValueError(f'{fname}: AlternativeHypothesis must be "Unequal", "Less" or "Greater", not {a!r}')
    return a


def p_normal(z, alt):
    """p-value of a statistic that is standard normal under H0"""
    if alt == "Less":
        return norm_cdf(z)
    if alt == "Greater":
        return norm_sf(z)
    return min(1.0, 2 * norm_sf(abs(z)))


def p_t(t, df, alt):
    if alt == "Less":
        return 1 - t_sf(t, df)
    if alt == "Greater":
        return t_sf(t, df)
    return min(1.0, 2 * t_sf(abs(t), df))


def sig_level(o, fname):
    a = o.get("SignificanceLevel", 0.05)
    v = _num(a, fname, "SignificanceLevel")
    if not (0 < v < 1):
        raise ValueError(f"{fname}: SignificanceLevel must be between 0 and 1")
    return fl(v)


TEST_PROPS = ("PValue", "TestStatistic", "TestData", "ShortTestConclusion", "DegreesOfFreedom")
TEST_OPTS = ("AlternativeHypothesis", "SignificanceLevel", "VerifyTestAssumptions", "Method", "MaxIterations")


def verify_opt(o, fname):
    v = o.get("VerifyTestAssumptions")
    if v is None:
        return
    s = str_or_sym(v)
    if s in ("Automatic", "None", "All") or isinstance(v, bool):
        return
    if is_list(v) or (isinstance(v, Expr) and v.head in ("Rule", "Association")):
        return
    raise ValueError(f"{fname}: invalid VerifyTestAssumptions setting {v!r}")


def test_result(fname, stat, p, prop, o, dof=None):
    """the requested property of a hypothesis test with statistic `stat` and p-value `p`"""
    if prop is None:
        return float(p)
    if is_list(prop):
        return L(*(test_result(fname, stat, p, q, o, dof) for q in prop.args))
    if prop in ("HypothesisTestData",) or (isinstance(prop, str) and prop.endswith("Table")) or prop == "TestConclusion":
        raise NotImplementedError(f'{fname}: the property "{prop}" (formatted output / HypothesisTestData) is not '
                                  "supported; use \"PValue\", \"TestStatistic\", \"TestData\" or \"ShortTestConclusion\"")
    if not isinstance(prop, str):
        raise TypeError(f"{fname}: the property must be a string, not {prop!r}")
    if prop == "PValue":
        return float(p)
    if prop == "TestStatistic":
        return stat
    if prop == "TestData":
        return L(stat, float(p))
    if prop == "ShortTestConclusion":
        return "Reject" if p <= sig_level(o, fname) else "Do not reject"
    if prop == "DegreesOfFreedom":
        if dof is None:
            raise ValueError(f'{fname}: this test has no "DegreesOfFreedom"')
        return dof
    raise ValueError(f'{fname}: unknown property "{prop}"')


def prop_arg(rest, fname, nmax):
    """positional arguments after the data: (args..., optional trailing property string or list of strings)"""
    rest = list(rest)
    prop = None
    if rest and (isinstance(rest[-1], str) or (is_list(rest[-1]) and rest[-1].args and
                                               all(isinstance(q, str) for q in rest[-1].args))):
        prop = rest.pop()
    if len(rest) > nmax:
        raise ValueError(f"{fname}: too many arguments")
    return rest + [None] * (nmax - len(rest)), prop


def fstat(x):
    """a test statistic as a machine number"""
    return float(x) if not isinstance(x, sp.Basic) else fl(x)


# ─────────────────────────── robust location and dispersion ───────────────────────────

def _c_arg(c, fname, default):
    if c is None:
        return None
    v = _num(c, fname, "scaling parameter c")
    if not v > 0:
        raise ValueError(f"{fname}: the scaling parameter c must be positive")
    return v


def _biweight_location(x, c=None):
    v, items = vec(x, "BiweightLocation", min_len=1)
    cc = _c_arg(c, "BiweightLocation", 6)
    cc = fl(cc) if cc is not None else 6.0
    fv = [fl(a) for a in v]
    xs = float(median(fv))
    for _ in range(500):
        M = float(median([abs(a - xs) for a in fv]))
        if M == 0:
            break
        u = [(a - xs) / (cc * M) for a in fv]
        den = sum((1 - ui * ui) ** 2 for ui in u if abs(ui) < 1)
        new = xs + sum((a - xs) * (1 - ui * ui) ** 2 for a, ui in zip(fv, u) if abs(ui) < 1) / den
        if abs(new - xs) <= 1e-15 * max(1.0, abs(xs)):
            xs = new
            break
        xs = new
    # self-check: x* is a fixed point of the biweight update (the estimating equation sum psi(u_i) = 0)
    M = float(median([abs(a - xs) for a in fv]))
    if M:
        u = [(a - xs) / (cc * M) for a in fv]
        psi = sum((a - xs) * (1 - ui * ui) ** 2 for a, ui in zip(fv, u) if abs(ui) < 1)
        if abs(psi) > 1e-8 * max(1.0, sum(abs(a - xs) for a in fv)):
            raise ArithmeticError("BiweightLocation: result failed verification")
    return xs


@fn("BiweightLocation", area=AREA, status="added",
    notes="BiweightLocation[list] and [list, c] (default c = 6): Tukey's biweight M-estimate of location, iterated "
          "from the median to the fixed point x* = x* + Sum[(x - x*)(1 - u^2)^2]/Sum[(1 - u^2)^2] over |u| < 1, "
          "u = (x - x*)/(c MedianDeviation about x*); a matrix gives the column-wise estimates; the result is a "
          "machine number (iterative). Symbolic data stays unevaluated; non-numeric data raises TypeError, c <= 0 "
          "ValueError; dates and quantities are not supported. Check: self-check (the estimating equation is "
          "verified at the returned fixed point) and reference values from the Wolfram documentation examples.")
def _bwloc(x: ListOf[Real] | MatrixOf[Real], c: Real = None) -> Real | ListOf[Real]:
    return per_column(x, "BiweightLocation", _biweight_location, c)


def _biweight_midvariance(x, c=None):
    v, items = vec(x, "BiweightMidvariance", min_len=1)
    cc = _c_arg(c, "BiweightMidvariance", 9)
    if cc is None:
        cc = sp.Integer(9)
    if inexact(items) or has_float(cc):
        v = [sp.Float(fl(a), 17) for a in v]
    n = len(v)
    M = median(v)
    mad = median([abs(a - M) for a in v])
    if mad == 0:
        return out(sp.Integer(0), items + [c] if c is not None else items)
    u = [(a - M) / (cc * mad) for a in v]
    sel = [(a, ui) for a, ui in zip(v, u) if abs(ui) < 1]
    num = n * sp.Add(*((a - M) ** 2 * (1 - ui ** 2) ** 4 for a, ui in sel))
    den = sp.Add(*((1 - ui ** 2) * (1 - 5 * ui ** 2) for a, ui in sel)) ** 2
    if den == 0:
        raise ValueError("BiweightMidvariance: the estimate is undefined for this data (zero denominator)")
    return out(sp.simplify(num / den) if not inexact(items) and not has_float(cc) else num / den,
               items + ([c] if c is not None else []))


@fn("BiweightMidvariance", area=AREA, status="added",
    notes="BiweightMidvariance[list] and [list, c] (default c = 9): n Sum[(x - M)^2 (1 - u^2)^4] / "
          "Sum[(1 - u^2)(1 - 5 u^2)]^2 over |u| < 1 with M = Median, u = (x - M)/(c MedianDeviation); exact for "
          "exact data and c; a matrix gives the column-wise values; 0 when the median deviation is 0. Symbolic data "
          "stays unevaluated; non-numeric data raises TypeError, c <= 0 ValueError. Check: exact; reference values "
          "from the Wolfram documentation examples and astropy.stats.biweight_midvariance.")
def _bwmv(x: ListOf[Real] | MatrixOf[Real], c: Real = None) -> Real | ListOf[Real]:
    return per_column(x, "BiweightMidvariance", _biweight_midvariance, c)


_QN_SMALL = {2: 0.399, 3: 0.994, 4: 0.512, 5: 0.844, 6: 0.611, 7: 0.857, 8: 0.669, 9: 0.872}
_SN_SMALL = {2: 0.743, 3: 1.851, 4: 0.954, 5: 1.351, 6: 0.993, 7: 1.198, 8: 1.005, 9: 1.131}


def _method_opt(o, fname):
    m = o.get("Method", "FiniteSample")
    m = str_or_sym(m)
    if m in ("Automatic",):
        m = "FiniteSample"
    if m not in ("FiniteSample", "None"):
        raise ValueError(f'{fname}: Method must be "FiniteSample" or "None"')
    return m


def _qn(x, c=None, method="FiniteSample"):
    v, items = vec(x, "QnDispersion", min_len=2)
    n = len(v)
    d = sorted((abs(a - b) for a, b in itertools.combinations(v, 2)), key=fl)
    h = n // 2 + 1
    k = h * (h - 1) // 2
    q = d[k - 1]
    if c is not None:
        cc = _c_arg(c, "QnDispersion", None)
        return out(cc * q, items + [c])
    # consistency factor 1/(Sqrt[2] InverseCDF[NormalDistribution[], 5/8]) and Croux-Rousseeuw (1992) small-sample
    # factors d_n
    cn = 1 / (math.sqrt(2) * float(mpmath.sqrt(2) * mpmath.erfinv(2 * mpmath.mpf(5) / 8 - 1)))
    dn = 1.0
    if method == "FiniteSample":
        dn = _QN_SMALL[n] if n <= 9 else (n / (n + 1.4) if n % 2 else n / (n + 3.8))
    return cn * dn * fl(q)


def _lomed(s):
    s = sorted(s, key=fl)
    return s[(len(s) + 1) // 2 - 1]


def _himed(s):
    s = sorted(s, key=fl)
    return s[len(s) // 2]


def _sn(x, c=None, method="FiniteSample"):
    v, items = vec(x, "SnDispersion", min_len=2)
    n = len(v)
    z = [_himed([abs(a - b) for b in v]) for a in v]
    s = _lomed(z)
    if c is not None:
        cc = _c_arg(c, "SnDispersion", None)
        return out(cc * s, items + [c])
    cn = 1.1926
    dn = 1.0
    if method == "FiniteSample":
        dn = _SN_SMALL[n] if n <= 9 else (n / (n - 0.9) if n % 2 else 1.0)
    return cn * dn * fl(s)


def _robust_disp(name, f):
    def impl(x: ListOf[Real] | MatrixOf[Real], *rest: Real | Option) -> Real | ListOf[Real]:
        args, o = opts(rest, ("Method",), name)
        if len(args) > 1:
            raise ValueError(f"{name}: too many arguments")
        m = _method_opt(o, name)
        c = args[0] if args else None
        return per_column(x, name, lambda col, cc: f(col, cc, m), c)
    return impl


fn("QnDispersion", area=AREA, status="added",
   notes="QnDispersion[list] = c_n d_n * (k-th smallest |x_i - x_j|, i < j, k = Binomial[Floor[n/2] + 1, 2]) with the "
         "normal-consistency factor c_n = 2.2219 and the Croux-Rousseeuw (1992) small-sample factors d_n (Method -> "
         "\"FiniteSample\", default; \"None\" drops d_n); QnDispersion[list, c] uses the factor c instead (exact for "
         "exact data and c); a matrix gives column values; machine number by default. Symbolic data stays "
         "unevaluated; non-numeric data raises TypeError, c <= 0 or a bad Method ValueError; fewer than 2 values "
         "ValueError. Check: reference values from the Wolfram documentation examples (0.760723 for the basic "
         "example) and the defining order statistic.")(_robust_disp("QnDispersion", _qn))
fn("SnDispersion", area=AREA, status="added",
   notes="SnDispersion[list] = c_n d_n lomed_i himed_j |x_i - x_j| (Rousseeuw-Croux Sn) with c_n = 1.1926 and the "
         "small-sample factors d_n (Method -> \"FiniteSample\", default; \"None\" drops d_n); SnDispersion[list, c] "
         "uses the factor c (exact for exact data and c); a matrix gives column values. Symbolic data stays "
         "unevaluated; non-numeric data raises TypeError, bad c or Method ValueError. Check: reference values "
         "from the Wolfram documentation examples and the defining order statistic.")(_robust_disp("SnDispersion", _sn))


# ─────────────────────────── location of multivariate data ───────────────────────────

def _points(x, fname):
    """data as a list of points (each a list of SymPy numbers); univariate data gives 1-element points"""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x) or not x.args:
        raise TypeError(f"{fname}: data must be a non-empty list")
    if is_matrix(x):
        return mat(x, fname), list(x.args), True
    v, items = vec(x, fname)
    return [[a] for a in v], items, False


@fn("SpatialMedian", area=AREA, status="added",
    notes="SpatialMedian[{x1, ...}] for univariate data is the Median (exact); for multivariate data "
          "{{x1, y1, ...}, ...} the geometric median ArgMin[Sum[EuclideanDistance[x, xi]]] by Weiszfeld's "
          "iteration (machine numbers); options AccuracyGoal, MaxIterations, Method, PrecisionGoal and "
          "WorkingPrecision are accepted only at Automatic; DistanceFunction other than Automatic/EuclideanDistance, "
          "GeoPosition and WeightedData raise NotImplementedError. Symbolic data stays unevaluated. Check: "
          "self-check (the gradient of the total distance vanishes at the returned point, or the point is a data "
          "point satisfying the subgradient condition) and reference values computed independently.")
def _spatial_median(x: ListOf[Real] | MatrixOf[Real], *rest: Option) -> Real | ListOf[Real]:
    _, o = opts(rest, ("AccuracyGoal", "MaxIterations", "Method", "PrecisionGoal", "WorkingPrecision",
                       "DistanceFunction"), "SpatialMedian")
    for k, val in o.items():
        if str_or_sym(val) not in ("Automatic", "EuclideanDistance"):
            raise NotImplementedError(f"SpatialMedian: the option {k} -> {val} is not supported")
    pts, items, multi = _points(x, "SpatialMedian")
    if not multi:
        return out(median([p[0] for p in pts]), items)
    import numpy as np
    P = np.array([[fl(a) for a in p] for p in pts], dtype=float)
    y = P.mean(axis=0)
    for _ in range(10000):
        d = np.linalg.norm(P - y, axis=1)
        if np.any(d < 1e-14):
            # at a data point: check the subgradient condition; if it holds the data point is the median
            i = int(np.argmin(d))
            others = np.delete(P, i, axis=0)
            dd = np.linalg.norm(others - P[i], axis=1)
            g = ((others - P[i]) / dd[:, None]).sum(axis=0)
            if np.linalg.norm(g) <= 1.0 + 1e-12:
                y = P[i]
                break
            y = P[i] + 1e-6 * g / np.linalg.norm(g)
            continue
        w = 1 / d
        new = (P * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(new - y) <= 1e-15 * max(1.0, np.linalg.norm(y)):
            y = new
            break
        y = new
    d = np.linalg.norm(P - y, axis=1)
    if np.all(d > 1e-12):
        g = ((P - y) / d[:, None]).sum(axis=0)
        if np.linalg.norm(g) > 1e-6 * len(P):
            raise ArithmeticError("SpatialMedian: result failed verification")
    return L(*(float(t) for t in y))


@fn("CentralFeature", area=AREA, status="added",
    notes="CentralFeature[{x1, ...}] the data element with the smallest total Euclidean distance to the others "
          "(numbers or numeric vectors; the first such element on ties; exact comparison for exact data); "
          "CentralFeature[{x1 -> v1, ...}] and [{x1, ...} -> {v1, ...}] give the corresponding v; "
          "[data -> Automatic] the index. DistanceFunction -> f uses f[xi, xj]; other data types (strings, images, "
          "dates, GeoPosition, WeightedData) raise NotImplementedError. Check: exact (the total distances are "
          "compared exactly) with reference values worked out by hand.")
def _central_feature(x: ListOf[Expr] | Rule, *rest: Option) -> Expr:
    _, o = opts(rest, ("DistanceFunction",), "CentralFeature")
    labels = None
    if isinstance(x, Expr) and x.head == "Rule":
        data, lab = x.args
        if isinstance(lab, Symbol) and lab.name == "Automatic":
            labels = list(range(1, len(data.args) + 1)) if is_list(data) else None
        elif is_list(lab) and is_list(data) and len(lab.args) == len(data.args):
            labels = list(lab.args)
        else:
            raise ValueError("CentralFeature: data -> labels needs a list of labels of the same length")
        x = data
    elif is_list(x) and x.args and all(_is_rule(r) for r in x.args):
        labels = [r.args[1] for r in x.args]
        x = L(*(r.args[0] for r in x.args))
    if not is_list(x) or not x.args:
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError("CentralFeature: data must be a non-empty list")
    if any(isinstance(a, str) for a in x.args):
        raise NotImplementedError("CentralFeature: string data is not supported")
    df = o.get("DistanceFunction")
    if df is not None and str_or_sym(df) in ("Automatic", "EuclideanDistance"):
        df = None
    pts, items, multi = _points(x, "CentralFeature")
    n = len(pts)
    if df is None:
        dist = lambda a, b: sp.sqrt(sp.Add(*((u - w) ** 2 for u, w in zip(a, b)))) if multi else sp.Abs(a[0] - b[0])  # noqa: E731
    else:
        from ..kernel import call

        def dist(a, b):
            r = ev(call(df, *(L(*(R(t) for t in p)) if multi else R(p[0]) for p in (a, b))))
            return _num(r, "CentralFeature", "distance")
    tot = [sp.Add(*(dist(pts[i], pts[j]) for j in range(n) if j != i)) for i in range(n)]
    ft = [sp.N(t, 30) for t in tot]
    best = 0
    for i in range(1, n):
        if ft[i] < ft[best] and not (sp.Abs(ft[i] - ft[best]) < sp.Float(10) ** -25 and sp.simplify(tot[i] - tot[best]) == 0):
            best = i
    if labels is not None:
        return labels[best]
    return x.args[best]


# ─────────────────────────── dependence measures ───────────────────────────

def _pairwise(name, f, dist_ok=False):
    """vector/matrix calling conventions of Wolfram's dependence measures"""
    def impl(a: ListOf[Real] | MatrixOf[Real], b: ListOf[Real] | MatrixOf[Real] = None) -> Real | MatrixOf[Real]:
        if isinstance(a, Expr) and isinstance(a.head, str) and a.head.endswith("Distribution"):
            raise NotImplementedError(f"{name}: distribution arguments are not supported")
        if b is None:
            if not is_matrix(a):
                if isinstance(a, Symbol):
                    raise Unevaluated()
                raise TypeError(f"{name}: a single argument must be a matrix")
            cols = columns([list(r.args) for r in a.args])
            return L(*(L(*(f(L(*ci), L(*cj)) for cj in cols)) for ci in cols))
        if is_matrix(a) and is_matrix(b):
            if len(a.args) != len(b.args):
                raise ValueError(f"{name}: the matrices must have the same number of rows")
            ca = columns([list(r.args) for r in a.args])
            cb = columns([list(r.args) for r in b.args])
            return L(*(L(*(f(L(*ci), L(*cj)) for cj in cb)) for ci in ca))
        return f(a, b)
    return impl


def _two(a, b, fname, min_len=2):
    u, ia = vec(a, fname, min_len=min_len)
    v, ib = vec(b, fname, min_len=min_len)
    if len(u) != len(v):
        raise ValueError(f"{fname}: the vectors must have the same length")
    return u, v, ia + ib


def _abscorr(a, b):
    u, v, items = _two(a, b, "AbsoluteCorrelation", 1)
    for x in u + v:
        pass
    uu = [S(t) for t in (a.args)]
    vv = [S(t) for t in (b.args)]
    return out(sp.expand(sp.Add(*(p * sp.conjugate(q) for p, q in zip(uu, vv))) / len(uu)), items)


def _abscorr_impl(a: ListOf[Number] | MatrixOf[Number], b: ListOf[Number] | MatrixOf[Number] = None) -> Number | MatrixOf[Number]:
    if isinstance(a, Expr) and isinstance(a.head, str) and a.head.endswith("Distribution"):
        raise NotImplementedError("AbsoluteCorrelation: distribution arguments are not supported")
    for t in (a, b):
        if t is not None and is_list(t):
            for e in (t.args if not is_matrix(t) else [y for r in t.args for y in r.args]):
                _num(e, "AbsoluteCorrelation", "data value", real=False)
    f = lambda p, q: _abscorr_c(p, q)  # noqa: E731
    return _pairwise("AbsoluteCorrelation", f)(a, b)


def _abscorr_c(a, b):
    ua = [_num(t, "AbsoluteCorrelation", "data value", real=False) for t in a.args]
    ub = [_num(t, "AbsoluteCorrelation", "data value", real=False) for t in b.args]
    if len(ua) != len(ub):
        raise ValueError("AbsoluteCorrelation: the vectors must have the same length")
    return out(sp.expand(sp.Add(*(p * sp.conjugate(q) for p, q in zip(ua, ub))) / len(ua)), list(a.args) + list(b.args))


fn("AbsoluteCorrelation", area=AREA, status="added",
   notes="AbsoluteCorrelation[v1, v2] = Mean[v1 Conjugate[v2]] (non-centred correlation); [m] the matrix of the "
         "values between columns of m, [m1, m2] between the columns of m1 and m2; exact for exact data; complex "
         "data allowed. Symbolic entries stay unevaluated; unequal lengths raise ValueError; non-numeric data "
         "TypeError; distribution arguments NotImplementedError. Check: exact; reference values worked out from "
         "the definition.")(_abscorr_impl)


def _gk_counts(u, v):
    C = D = 0
    for i in range(len(u)):
        for j in range(i + 1, len(u)):
            s = sp.sign(u[i] - u[j]) * sp.sign(v[i] - v[j])
            if s > 0:
                C += 1
            elif s < 0:
                D += 1
    return C, D


def _gkgamma(a, b):
    u, v, items = _two(a, b, "GoodmanKruskalGamma")
    C, D = _gk_counts(u, v)
    if C + D == 0:
        raise ValueError("GoodmanKruskalGamma: undefined (no concordant or discordant pairs)")
    return out(sp.Rational(C - D, C + D), items)


fn("GoodmanKruskalGamma", area=AREA, status="added",
   notes="GoodmanKruskalGamma[v1, v2] = (C - D)/(C + D) with C, D the numbers of concordant and discordant pairs "
         "(tied pairs ignored); exact rational for exact data; [m] and [m1, m2] give the matrices of the values "
         "between columns. Unequal lengths raise ValueError, as does data with no untied pairs; non-numeric data "
         "TypeError; symbolic data stays unevaluated; distributions raise NotImplementedError. Check: exact; "
         "reference values from scipy.stats.somersd-independent pair counts and the identity with KendallTau "
         "without ties.")(_pairwise("GoodmanKruskalGamma", _gkgamma))


def _blomqvist(a, b):
    u, v, items = _two(a, b, "BlomqvistBeta")
    mu, mv = median(u), median(v)
    su = [sp.sign(t - mu) for t in u]
    sv = [sp.sign(t - mv) for t in v]
    n = len(u)
    ma, mb = mean(su), mean(sv)
    cov = sp.Add(*((p - ma) * (q - mb) for p, q in zip(su, sv)))
    va = sp.Add(*((p - ma) ** 2 for p in su))
    vb = sp.Add(*((q - mb) ** 2 for q in sv))
    if va == 0 or vb == 0:
        raise ValueError("BlomqvistBeta: undefined, all values lie on one side of the median")
    return out(sp.simplify(cov / sp.sqrt(va * vb)) if not inexact(items) else cov / sp.sqrt(va * vb), items)


fn("BlomqvistBeta", area=AREA, status="added",
   notes="BlomqvistBeta[v1, v2] = Correlation[Sign[v1 - Median[v1]], Sign[v2 - Median[v2]]] (Blomqvist's medial "
         "correlation, as documented); exact for exact data; [m] and [m1, m2] give matrices between columns. "
         "Unequal lengths raise ValueError, as does a sign vector with zero variance; non-numeric data TypeError; "
         "symbolic data stays unevaluated; distributions raise NotImplementedError. Check: exact; reference "
         "values from the documented definition (Pearson correlation of the median signs, numpy.corrcoef).")(
    _pairwise("BlomqvistBeta", _blomqvist))


def _hoeffding(a, b):
    u, v, items = _two(a, b, "HoeffdingD", 5)
    n = len(u)
    Ru, Rv = ranks(u), ranks(v)
    fu, fv = [fl(t) for t in u], [fl(t) for t in v]

    def phi(p, q):
        return Fraction(1) if p < q else (Fraction(1, 2) if p == q else Fraction(0))
    c = [sum(phi(fu[j], fu[i]) * phi(fv[j], fv[i]) for j in range(n) if j != i) for i in range(n)]
    Q = sum((Ru[i] - 1) * (Ru[i] - 2) * (Rv[i] - 1) * (Rv[i] - 2) for i in range(n))
    Rr = sum((Ru[i] - 2) * (Rv[i] - 2) * c[i] for i in range(n))
    Ss = sum(c[i] * (c[i] - 1) for i in range(n))
    val = Fraction(30) * (-2 * (n - 2) * Rr + (n - 3) * (n - 2) * Ss + Q) / \
        (n * (n - 1) * (n - 2) * (n - 3) * (n - 4))
    return out(sp.Rational(val.numerator, val.denominator), items)


fn("HoeffdingD", area=AREA, status="added",
   notes="HoeffdingD[v1, v2] = 30 ((n-2)(n-3) S - 2 (n-2) R + Q) / Pochhammer[n-4, 5] (Hoeffding's D on the 30x "
         "scale, between -1/2 and 1) with mid-ranks R_i, S_i and bivariate counts c_i = Sum[phi(x_j, x_i) "
         "phi(y_j, y_i), j != i], phi = 1 for <, 1/2 for ties (Hollander-Wolfe); exact rational; vectors of length "
         ">= 5; [m] and [m1, m2] give matrices between columns. Unequal lengths or n < 5 ValueError; non-numeric data TypeError; symbolic data stays "
         "unevaluated. Check: exact; reference values from an independent O(n^2) implementation of Hoeffding "
         "(1948) and the value 1 for strictly monotone data.")(_pairwise("HoeffdingD", _hoeffding))


# ─────────────────────────── order statistics, entropy, queueing ───────────────────────────

def _ranked(name, largest):
    def impl(x: ListOf[Real], k: Integer) -> Real:
        if isinstance(x, Symbol):
            raise Unevaluated()
        if not is_list(x):
            raise TypeError(f"{name}: the first argument must be a list")
        if isinstance(k, bool) or not (isinstance(k, int) or isinstance(k, sp.Integer)):
            if isinstance(k, (Symbol, Expr)) and not is_list(k):
                raise Unevaluated()
            raise TypeError(f"{name}: the rank must be an integer")
        k = int(k)
        n = len(x.args)
        if k == 0 or abs(k) > n:
            raise ValueError(f"{name}: the rank {k} must be between 1 and {n} (or -{n} and -1)")
        vals = []
        for e in x.args:
            v = S(e) if not isinstance(e, (bool, str)) else None
            if v is None or isinstance(v, list):
                raise TypeError(f"{name}: list elements must be real numbers")
            if v.free_symbols:
                raise Unevaluated()
            if not v.is_number or v.is_real is False:
                raise TypeError(f"{name}: list elements must be real numbers")
            vals.append((sp.N(v, 30), e))
        order = sorted(vals, key=lambda t: t[0])
        big = largest if k > 0 else not largest
        kk = abs(k)
        return order[-kk][1] if big else order[kk - 1][1]
    return impl


fn("RankedMin", area=AREA, status="added",
   notes="RankedMin[list, n] the n-th smallest element (repeated values counted); RankedMin[list, -n] the n-th "
         "largest; exact numeric elements are compared numerically and returned unchanged. A symbolic element "
         "keeps the call unevaluated; |n| > Length[list] or n = 0 raises ValueError, non-real elements TypeError. "
         "Check: exact; reference cases from the Wolfram documentation and Sort.")(_ranked("RankedMin", False))
fn("RankedMax", area=AREA, status="added",
   notes="RankedMax[list, n] the n-th largest element (repeated values counted); RankedMax[list, -n] the n-th "
         "smallest; elements returned unchanged. A symbolic element keeps the call unevaluated; |n| > Length[list] "
         "or n = 0 raises ValueError, non-real elements TypeError. Check: exact; reference cases from the Wolfram "
         "documentation and Sort.")(_ranked("RankedMax", True))


@fn("Entropy", area=AREA, status="added",
    notes="Entropy[list] = -Sum[p Log[p]] over the frequencies p of the distinct elements (exact, of the form "
          "Log[n] - Sum[c Log[c]]/n); Entropy[k, list] uses base-k logarithms; Entropy[\"string\"] uses the "
          "characters. Elements are compared structurally (symbols allowed). A non-list raises TypeError, an empty "
          "list ValueError, a base k <= 0 or k = 1 ValueError. Image and other data types raise NotImplementedError. "
          "Check: exact; reference values from the Wolfram documentation (Entropy[{0, 1, 1, 4, 1, 1}] = Log[6] - "
          "2 Log[4]/3) and scipy.stats.entropy.")
def _entropy(a: Expr, b: Expr = None) -> Real:
    if b is None:
        base, x = None, a
    else:
        base, x = a, b
    if isinstance(x, str):
        items = list(x)
    elif is_list(x):
        items = list(x.args)
    elif isinstance(x, Symbol):
        raise Unevaluated()
    elif isinstance(x, Expr) and x.head in ("Image", "Image3D", "Audio"):
        raise NotImplementedError(f"Entropy: {x.head} data is not supported")
    else:
        raise TypeError("Entropy: the argument must be a list or a string")
    if not items:
        raise ValueError("Entropy: the data must not be empty")
    counts = {}
    order = []
    for e in items:
        key = repr(e) if not isinstance(e, str) else "s:" + e
        if key not in counts:
            counts[key] = 0
            order.append(key)
        counts[key] += 1
    n = len(items)
    h = sp.log(n) - sp.Add(*(c * sp.log(c) for c in counts.values() if c > 1)) / n
    inexact_ = any(isinstance(e, float) or has_float(e) for e in items if not isinstance(e, str))
    if base is not None:
        k = _num(base, "Entropy", "base")
        if not k > 0 or k == 1:
            raise ValueError("Entropy: the base must be positive and different from 1")
        h = h / sp.log(k)
        inexact_ = inexact_ or has_float(k)
    return R(sp.N(h, 15)) if inexact_ else R(h)


@fn("ErlangB", area=AREA, status="added", listable=True,
    notes="ErlangB[c, a] = (a^c/c!)/Sum[a^k/k!, {k, 0, c}], the blocking probability of an M/M/c/c loss system with c "
          "servers and offered traffic a = lambda/mu; exact for exact a; machine for machine a. Symbolic a or c stays "
          "unevaluated; c not a non-negative integer or a < 0 raises ValueError; non-numeric arguments TypeError. "
          "Check: exact; the Erlang recursion B(c) = a B(c-1)/(c + a B(c-1)) and reference values (Wolfram "
          "documentation: ErlangB[3, 4/5] = 32/1059).")
def _erlangb(c: NonNegInteger, a: Real) -> Real:
    cc = _num(c, "ErlangB", "number of servers")
    av = _num(a, "ErlangB", "traffic")
    if not (cc.is_Integer and cc >= 0):
        raise ValueError("ErlangB: the number of servers must be a non-negative integer")
    if av < 0:
        raise ValueError("ErlangB: the traffic must be non-negative")
    c_ = int(cc)
    num = av ** c_ / sp.factorial(c_)
    den = sp.Add(*(av ** k / sp.factorial(k) for k in range(c_ + 1)))
    return out(sp.cancel(num / den) if not has_float(av) else num / den, [a])


@fn("ErlangC", area=AREA, status="added", listable=True,
    notes="ErlangC[c, a] the probability of waiting in an M/M/c queue with c servers and traffic a < c: "
          "(a^c/c!) (c/(c - a)) / (Sum[a^k/k!, {k, 0, c - 1}] + (a^c/c!) (c/(c - a))); exact for exact a; 1 when "
          "a >= c (unstable queue: every customer waits). Symbolic a or c stays unevaluated; c not a positive "
          "integer or a < 0 raises ValueError. Check: exact; the identity C = c B/(c - a (1 - B)) with ErlangB and "
          "reference values from the Wolfram documentation.")
def _erlangc(c: PosInteger, a: Real) -> Real:
    cc = _num(c, "ErlangC", "number of servers")
    av = _num(a, "ErlangC", "traffic")
    if not (cc.is_Integer and cc >= 1):
        raise ValueError("ErlangC: the number of servers must be a positive integer")
    if av < 0:
        raise ValueError("ErlangC: the traffic must be non-negative")
    c_ = int(cc)
    if av >= c_:
        return out(sp.Integer(1), [a])
    t = av ** c_ / sp.factorial(c_) * c_ / (c_ - av)
    v = t / (sp.Add(*(av ** k / sp.factorial(k) for k in range(c_))) + t)
    return out(sp.cancel(v) if not has_float(av) else v, [a])
