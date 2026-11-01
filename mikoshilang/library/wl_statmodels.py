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


# ─────────────────────────── hypothesis tests: data specifications ───────────────────────────

def dspec(x, fname, two_ok=True, one_ok=True, min_len=2):
    """('one', v, items) for univariate data, ('two', v1, v2, items) for {data1, data2}; multivariate raises"""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x):
        raise TypeError(f"{fname}: the data must be a list")
    if len(x.args) == 2 and all(is_list(a) for a in x.args) and \
            not any(is_list(b) for a in x.args for b in a.args):
        if not two_ok:
            raise ValueError(f"{fname}: expected one dataset, not {{data1, data2}}")
        v1, i1 = vec(x.args[0], fname, min_len=min_len)
        v2, i2 = vec(x.args[1], fname, min_len=min_len)
        return ("two", v1, v2, i1 + i2)
    if any(is_list(a) for a in x.args):
        raise NotImplementedError(f"{fname}: multivariate data is not supported")
    if not one_ok:
        raise ValueError(f"{fname}: expected {{data1, data2}}")
    v, items = vec(x, fname, min_len=min_len)
    return ("one", v, items)


def groups(x, fname, k_min=2, min_len=2):
    """{data1, data2, ...} with at least k_min univariate groups"""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x) or len(x.args) < k_min or not all(is_list(a) for a in x.args):
        raise TypeError(f"{fname}: the data must be a list of at least {k_min} datasets {{data1, data2, ...}}")
    out_ = []
    for a in x.args:
        if any(is_list(b) for b in a.args):
            raise NotImplementedError(f"{fname}: multivariate data is not supported")
        out_.append([fl(t) for t in vec(a, fname, min_len=min_len)[0]])
    return out_


def _mu0(m, fname):
    return 0.0 if m is None else fl(_num(m, fname, "mu0"))


def _fvec(v):
    return [fl(t) for t in v]


def _fmean(v):
    return math.fsum(v) / len(v)


def _fvar(v):
    m = _fmean(v)
    return math.fsum((t - m) ** 2 for t in v) / (len(v) - 1)


# ─────────────────────────── location tests ───────────────────────────

def _ztest_core(x, s2, mu0, fname, paired):
    d = dspec(x, fname)
    m0 = _mu0(mu0, fname)
    if d[0] == "two" and paired:
        a, b = _fvec(d[1]), _fvec(d[2])
        if len(a) != len(b):
            raise ValueError(f"{fname}: paired samples must have the same length")
        v = [p - q for p, q in zip(a, b)]
        d = ("one", v)
    if d[0] == "one":
        v = _fvec(d[1])
        var = _fvar(v) if s2 is None else _variance_arg(s2, fname)
        return (_fmean(v) - m0) / math.sqrt(var / len(v))
    a, b = _fvec(d[1]), _fvec(d[2])
    if s2 is None:
        va, vb = _fvar(a), _fvar(b)
    elif is_list(s2) and len(s2.args) == 2:
        va, vb = (_variance_arg(t, fname) for t in s2.args)
    else:
        va = vb = _variance_arg(s2, fname)
    return (_fmean(a) - _fmean(b) - m0) / math.sqrt(va / len(a) + vb / len(b))


def _variance_arg(s2, fname):
    if is_list(s2):
        raise NotImplementedError(f"{fname}: covariance matrices (multivariate data) are not supported")
    v = _num(s2, fname, "variance")
    if not v > 0:
        raise ValueError(f"{fname}: the variance must be positive")
    return fl(v)


def _ztest_impl(name, paired):
    def impl(x: ListOf[Real] | ListOf[ListOf[Real]], *rest: Real | String | Option) -> Real | Expr:
        args, o = opts(rest, TEST_OPTS, name)
        (s2, mu0), prop = prop_arg(args, name, 2)
        if isinstance(s2, Symbol) and s2.name == "Automatic":
            s2 = None
        verify_opt(o, name)
        z = _ztest_core(x, s2, mu0, name, paired)
        return test_result(name, z, p_normal(z, alt_of(o, name)), prop, o)
    return impl


fn("ZTest", area=AREA, status="added",
   notes="ZTest[data], [data, s2], [data, s2, mu0] and [{data1, data2}, s2 | {s1^2, s2^2}, mu0] with an optional "
         "trailing property: the z-test of mean = mu0 (or difference of means = mu0) with known variance s2 (the "
         "sample variance when omitted or Automatic, as documented); p-value from NormalDistribution[]; options "
         "AlternativeHypothesis (\"Unequal\", \"Less\", \"Greater\"), SignificanceLevel, VerifyTestAssumptions "
         "(accepted; assumptions are not diagnosed). Multivariate data and covariance matrices raise "
         "NotImplementedError; s2 <= 0 or a bad option ValueError; symbolic data stays unevaluated. Check: "
         "reference values from the closed form and statsmodels.stats.weightstats.ztest.")(_ztest_impl("ZTest", False))
fn("PairedZTest", area=AREA, status="added",
   notes="PairedZTest[data] and [{data1, data2}, s2, mu0, prop]: the z-test of the mean of data (or of the paired "
         "differences data1 - data2) against mu0 with known variance s2 (sample variance when omitted); options as "
         "for ZTest. Unequal sample lengths raise ValueError; multivariate data NotImplementedError. Check: "
         "reference values from the closed form (identical to ZTest on the differences).")(
    _ztest_impl("PairedZTest", True))


@fn("PairedTTest", area=AREA, status="added",
    notes="PairedTTest[data], [data, mu0] and [{data1, data2}, mu0] with an optional trailing property: Student's t "
          "test of the mean of data (or of the paired differences) against mu0, n - 1 degrees of freedom "
          "(\"DegreesOfFreedom\"); options AlternativeHypothesis, SignificanceLevel, VerifyTestAssumptions "
          "(accepted, not diagnosed). Unequal paired lengths raise ValueError; multivariate data "
          "NotImplementedError; symbolic data stays unevaluated. Check: reference values from scipy.stats.ttest_rel "
          "/ ttest_1samp and the Wolfram documentation example.")
def _pairedt(x: ListOf[Real] | ListOf[ListOf[Real]], *rest: Real | String | Option) -> Real | Expr:
    name = "PairedTTest"
    args, o = opts(rest, TEST_OPTS, name)
    (mu0,), prop = prop_arg(args, name, 1)
    verify_opt(o, name)
    d = dspec(x, name)
    if d[0] == "two":
        a, b = _fvec(d[1]), _fvec(d[2])
        if len(a) != len(b):
            raise ValueError(f"{name}: paired samples must have the same length")
        v = [p - q for p, q in zip(a, b)]
    else:
        v = _fvec(d[1])
    n = len(v)
    s2 = _fvar(v)
    if s2 == 0:
        raise ValueError(f"{name}: the data has zero variance")
    t = (_fmean(v) - _mu0(mu0, name)) / math.sqrt(s2 / n)
    return test_result(name, t, p_t(t, n - 1, alt_of(o, name)), prop, o, n - 1)


def _diffs(x, mu0, fname):
    d = dspec(x, fname, min_len=1)
    m0 = _mu0(mu0, fname)
    if d[0] == "two":
        a, b = _fvec(d[1]), _fvec(d[2])
        if len(a) != len(b):
            raise ValueError(f"{fname}: paired samples must have the same length")
        return [p - q - m0 for p, q in zip(a, b)]
    return [t - m0 for t in _fvec(d[1])]


@fn("SignTest", area=AREA, status="added",
    notes="SignTest[data], [data, mu0] and [{data1, data2}, mu0] with an optional trailing property: the sign test "
          "of median = mu0 (or median of the paired differences); values equal to mu0 are dropped; the test "
          "statistic is the number of values above mu0 and the p-value is exact (BinomialDistribution[n, 1/2]; "
          "two-sided = Min[1, 2 Min[P(S <= s), P(S >= s)]]); options AlternativeHypothesis, SignificanceLevel. "
          "Multivariate data raises NotImplementedError; all values equal to mu0 ValueError. Check: exact binomial "
          "tail; reference values from scipy.stats.binomtest.")
def _signtest(x: ListOf[Real] | ListOf[ListOf[Real]], *rest: Real | String | Option) -> Real | Expr:
    name = "SignTest"
    args, o = opts(rest, TEST_OPTS, name)
    (mu0,), prop = prop_arg(args, name, 1)
    v = [t for t in _diffs(x, mu0, name) if t != 0]
    n = len(v)
    if n == 0:
        raise ValueError(f"{name}: every value equals mu0")
    s = sum(1 for t in v if t > 0)
    alt = alt_of(o, name)
    lo = binom_cdf(s, n)
    hi = 1 - binom_cdf(s - 1, n)
    p = lo if alt == "Less" else hi if alt == "Greater" else min(Fraction(1), 2 * min(lo, hi))
    return test_result(name, s, float(p), prop, o)


def _signed_rank_dist(n):
    """counts[w] = number of subsets of {1..n} with sum w"""
    c = [1] + [0] * (n * (n + 1) // 2)
    for k in range(1, n + 1):
        for w in range(len(c) - 1, k - 1, -1):
            c[w] += c[w - k]
    return c


def _method(o, fname, allowed=("Automatic", "Exact", "Asymptotic")):
    m = o.get("Method", "Automatic")
    m = str_or_sym(m) if not isinstance(m, str) else m
    if m not in allowed:
        raise NotImplementedError(f"{fname}: Method -> {m!r} is not supported (use one of {', '.join(allowed)})")
    return m


@fn("SignedRankTest", area=AREA, status="added",
    notes="SignedRankTest[data], [data, mu0] and [{data1, data2}, mu0] with an optional trailing property: the "
          "Wilcoxon signed-rank test; differences equal to 0 are dropped, ties get mid-ranks; the statistic is the "
          "sum of the ranks of the positive differences. Method -> \"Exact\" (the exact null distribution; only "
          "without ties) or \"Asymptotic\" (normal approximation with tie and continuity corrections); Automatic "
          "uses the exact distribution for n <= 50 without ties, else the asymptotic one; options "
          "AlternativeHypothesis, SignificanceLevel, VerifyTestAssumptions (accepted). \"Permutation\" and "
          "multivariate data raise NotImplementedError. Check: reference values from the Wolfram documentation "
          "(statistic 41.5, p = 0.238235) and scipy.stats.wilcoxon.")
def _signedrank(x: ListOf[Real] | ListOf[ListOf[Real]], *rest: Real | String | Option) -> Real | Expr:
    name = "SignedRankTest"
    args, o = opts(rest, TEST_OPTS, name)
    (mu0,), prop = prop_arg(args, name, 1)
    verify_opt(o, name)
    v = [t for t in _diffs(x, mu0, name) if t != 0]
    n = len(v)
    if n == 0:
        raise ValueError(f"{name}: every difference is zero")
    r = ranks([abs(t) for t in v])
    w = sum((ri for ri, t in zip(r, v) if t > 0), Fraction(0))
    alt = alt_of(o, name)
    m = _method(o, name)
    ties = has_ties([abs(t) for t in v])
    if m == "Exact" and ties:
        raise NotImplementedError(f"{name}: the exact distribution is not available with ties")
    stat = float(w) if w.denominator != 1 else int(w)
    if m == "Exact" or (m == "Automatic" and not ties and n <= 50):
        c = _signed_rank_dist(n)
        tot = 2 ** n
        wi = int(w)
        lo = Fraction(sum(c[:wi + 1]), tot)
        hi = Fraction(sum(c[wi:]), tot)
        p = lo if alt == "Less" else hi if alt == "Greater" else min(Fraction(1), 2 * min(lo, hi))
        return test_result(name, stat, float(p), prop, o)
    mean_ = n * (n + 1) / 4
    t = {}
    for ri in r:
        t[ri] = t.get(ri, 0) + 1
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(k ** 3 - k for k in t.values()) / 48
    p = _p_cc(float(w) - mean_, math.sqrt(var), alt)
    return test_result(name, stat, p, prop, o)


def _p_cc(diff, sd, alt):
    """normal p-value of (stat - mean)/sd with a continuity correction of 1/2"""
    if alt == "Greater":
        return norm_sf((diff - 0.5) / sd)
    if alt == "Less":
        return norm_cdf((diff + 0.5) / sd)
    return min(1.0, 2 * norm_sf((abs(diff) - 0.5) / sd))


def _mw_dist(n1, n2):
    """counts[u] of the Mann-Whitney U (number of (i, j) with x_i > y_j) under H0, no ties"""
    # number of partitions of u into at most n1 parts each <= n2 (Gaussian binomial coefficients)
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def f(u, a, b):
        if u < 0:
            return 0
        if a == 0 or b == 0:
            return 1 if u == 0 else 0
        return f(u - b, a - 1, b) + f(u, a, b - 1)
    return [f(u, n1, n2) for u in range(n1 * n2 + 1)]


@fn("MannWhitneyTest", area=AREA, status="added",
    notes="MannWhitneyTest[{data1, data2}] and [{data1, data2}, mu0] with an optional trailing property: the "
          "Mann-Whitney U test of median(data1) - median(data2) = mu0; the statistic is U = #{(i, j): x_i - mu0 > "
          "y_j} + #ties/2. Method -> \"Exact\" (exact null distribution, no ties) or \"Asymptotic\" (normal "
          "approximation with tie and continuity corrections, as documented); Automatic is exact when there are no "
          "ties and n1 n2 <= 2500. Options AlternativeHypothesis, SignificanceLevel. \"Permutation\" and multivariate "
          "data raise NotImplementedError; a single dataset ValueError. Check: reference values from "
          "scipy.stats.mannwhitneyu (method=\"exact\" / \"asymptotic\").")
def _mannwhitney(x: ListOf[ListOf[Real]], *rest: Real | String | Option) -> Real | Expr:
    name = "MannWhitneyTest"
    args, o = opts(rest, TEST_OPTS, name)
    (mu0,), prop = prop_arg(args, name, 1)
    d = dspec(x, name, one_ok=False, min_len=1)
    m0 = _mu0(mu0, name)
    a = [t - m0 for t in _fvec(d[1])]
    b = _fvec(d[2])
    n1, n2 = len(a), len(b)
    U = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in a for q in b)
    alt = alt_of(o, name)
    m = _method(o, name)
    ties = has_ties(a + b)
    if m == "Exact" and ties:
        raise NotImplementedError(f"{name}: the exact distribution is not available with ties")
    stat = int(U) if U == int(U) else U
    if m == "Exact" or (m == "Automatic" and not ties and n1 * n2 <= 2500):
        c = _mw_dist(n1, n2)
        tot = sum(c)
        u = int(U)
        lo = Fraction(sum(c[:u + 1]), tot)
        hi = Fraction(sum(c[u:]), tot)
        p = lo if alt == "Less" else hi if alt == "Greater" else min(Fraction(1), 2 * min(lo, hi))
        return test_result(name, stat, float(p), prop, o)
    r = ranks(a + b)
    N = n1 + n2
    t = {}
    for ri in r:
        t[ri] = t.get(ri, 0) + 1
    var = n1 * n2 / 12 * ((N + 1) - sum(k ** 3 - k for k in t.values()) / (N * (N - 1)))
    p = _p_cc(U - n1 * n2 / 2, math.sqrt(var), alt)
    return test_result(name, stat, p, prop, o)


# ─────────────────────────── dispersion tests ───────────────────────────

def _anova(gs):
    """one-way ANOVA F statistic and degrees of freedom"""
    k = len(gs)
    N = sum(len(g) for g in gs)
    allm = math.fsum(math.fsum(g) for g in gs) / N
    ssb = math.fsum(len(g) * (_fmean(g) - allm) ** 2 for g in gs)
    ssw = math.fsum(math.fsum((t - _fmean(g)) ** 2 for t in g) for g in gs)
    if ssw == 0:
        raise ValueError("the within-group variation is zero")
    return (ssb / (k - 1)) / (ssw / (N - k)), k - 1, N - k


def _chisq_var(v, s0, alt):
    n = len(v)
    stat = (n - 1) * _fvar(v) / s0
    lo = 1 - chi2_sf(stat, n - 1)
    hi = chi2_sf(stat, n - 1)
    p = lo if alt == "Less" else hi if alt == "Greater" else min(1.0, 2 * min(lo, hi))
    return stat, p


def _ratio_arg(r0, fname):
    if r0 is None:
        return 1.0
    v = _num(r0, fname, "ratio")
    if not v > 0:
        raise ValueError(f"{fname}: the dispersion ratio must be positive")
    return fl(v)


@fn("FisherRatioTest", area=AREA, status="added",
    notes="FisherRatioTest[data] and [data, s0^2]: (n - 1) s^2/s0^2 against ChiSquareDistribution[n - 1] (s0^2 = 1 "
          "by default); FisherRatioTest[{data1, data2}] and [{data1, data2}, r0]: F = (s1^2/s2^2)/r0 against "
          "FRatioDistribution[n1 - 1, n2 - 1]; two-sided p = 2 Min[CDF, SF]; \"DegreesOfFreedom\"; options "
          "AlternativeHypothesis, SignificanceLevel, VerifyTestAssumptions (accepted). Multivariate data raises "
          "NotImplementedError; r0 <= 0 ValueError. Check: reference values from scipy.stats.f / chi2 tails of the "
          "closed-form statistic.")
def _fisherratio(x: ListOf[Real] | ListOf[ListOf[Real]], *rest: Real | String | Option) -> Real | Expr:
    name = "FisherRatioTest"
    args, o = opts(rest, TEST_OPTS, name)
    (r0,), prop = prop_arg(args, name, 1)
    verify_opt(o, name)
    alt = alt_of(o, name)
    d = dspec(x, name)
    r = _ratio_arg(r0, name)
    if d[0] == "one":
        stat, p = _chisq_var(_fvec(d[1]), r, alt)
        return test_result(name, stat, p, prop, o, len(d[1]) - 1)
    a, b = _fvec(d[1]), _fvec(d[2])
    if _fvar(b) == 0:
        raise ValueError(f"{name}: the second dataset has zero variance")
    f = _fvar(a) / _fvar(b) / r
    hi = f_sf(f, len(a) - 1, len(b) - 1)
    lo = 1 - hi
    p = lo if alt == "Less" else hi if alt == "Greater" else min(1.0, 2 * min(lo, hi))
    return test_result(name, f, p, prop, o, L(len(a) - 1, len(b) - 1))


def _levene_like(name, center, doc_ref):
    def impl(x: ListOf[Real] | ListOf[ListOf[Real]], *rest: Real | String | Option) -> Real | Expr:
        args, o = opts(rest, TEST_OPTS, name)
        (r0,), prop = prop_arg(args, name, 1)
        verify_opt(o, name)
        alt = alt_of(o, name)
        if is_list(x) and x.args and all(is_list(a) for a in x.args) and len(x.args) >= 2 and \
                not any(is_list(b) for a in x.args for b in a.args):
            gs = groups(x, name)
            r = _ratio_arg(r0, name)
            if r != 1.0:
                if len(gs) != 2:
                    raise ValueError(f"{name}: a dispersion ratio needs exactly two datasets")
                gs = [[t / math.sqrt(r) for t in gs[0]], gs[1]]
            z = [[abs(t - center(g)) for t in g] for g in gs]
            try:
                f, d1, d2 = _anova(z)
            except ValueError as e:
                raise ValueError(f"{name}: {e}")
            if len(gs) == 2 and alt != "Unequal":
                raise NotImplementedError(f"{name}: one-sided alternatives are not supported")
            return test_result(name, f, f_sf(f, d1, d2), prop, o, L(d1, d2))
        d = dspec(x, name)
        stat, p = _chisq_var(_fvec(d[1]), _ratio_arg(r0, name), alt)
        return test_result(name, stat, p, prop, o, len(d[1]) - 1)
    return impl


fn("LeveneTest", area=AREA, status="added",
   notes="LeveneTest[{data1, data2, ...}] (and [{data1, data2}, r0], data1 scaled by 1/Sqrt[r0]): one-way ANOVA F "
         "statistic of z = Abs[x - Mean[data_i]] against FRatioDistribution[k - 1, N - k]; LeveneTest[data, s0^2]: "
         "(n - 1) s^2/s0^2 against ChiSquareDistribution[n - 1] (as documented); options AlternativeHypothesis "
         "(one-sided only for one dataset), SignificanceLevel, VerifyTestAssumptions. Multivariate data raises "
         "NotImplementedError; constant groups ValueError. Check: reference values from scipy.stats.levene("
         "center=\"mean\").")(_levene_like("LeveneTest", _fmean, ""))
fn("BrownForsytheTest", area=AREA, status="added",
   notes="BrownForsytheTest[{data1, data2, ...}] (and [{data1, data2}, r0]): Levene's test with the group medians, "
         "z = Abs[x - Median[data_i]], F against FRatioDistribution[k - 1, N - k]; [data, s0^2] the chi-square "
         "variance test; options as for LeveneTest. Multivariate data raises NotImplementedError. Check: reference "
         "values from the Wolfram documentation (statistic 12.9663, p = 0.00121149) and scipy.stats.levene("
         "center=\"median\").")(_levene_like("BrownForsytheTest", lambda g: fl(median(g)), ""))


def _bartlett(gs):
    k = len(gs)
    N = sum(len(g) for g in gs)
    vs = [_fvar(g) for g in gs]
    if any(v == 0 for v in vs):
        raise ValueError("Bartlett: a dataset has zero variance")
    sp2 = math.fsum((len(g) - 1) * v for g, v in zip(gs, vs)) / (N - k)
    num = (N - k) * math.log(sp2) - math.fsum((len(g) - 1) * math.log(v) for g, v in zip(gs, vs))
    den = 1 + (math.fsum(1 / (len(g) - 1) for g in gs) - 1 / (N - k)) / (3 * (k - 1))
    return num / den, k - 1


def _conover(gs):
    """Conover's squared-ranks statistic (z = |x - median|) and its chi-square degrees of freedom"""
    z = [[abs(t - fl(median(g))) for t in g] for g in gs]
    flat = [t for g in z for t in g]
    r = ranks(flat)
    r2 = [float(q) ** 2 for q in r]
    N = len(flat)
    pos = 0
    S = []
    for g in z:
        S.append(math.fsum(r2[pos:pos + len(g)]))
        pos += len(g)
    Sbar = math.fsum(r2) / N
    D2 = (math.fsum(q * q for q in r2) - N * Sbar ** 2) / (N - 1)
    if D2 == 0:
        raise ValueError("Conover: undefined for this data (all ranks equal)")
    T = (math.fsum(s * s / len(g) for s, g in zip(S, z)) - N * Sbar ** 2) / D2
    return T, len(gs) - 1, S, Sbar, D2, z


@fn("ConoverTest", area=AREA, status="added",
    notes="ConoverTest[{data1, data2, ...}]: Conover's squared-ranks test of equal dispersion, ranks of "
          "z = Abs[x - Median[data_i]] (as documented); for two datasets the statistic is the standardized sum of "
          "squared ranks of data1 against NormalDistribution[] (AlternativeHypothesis supported), for k > 2 "
          "datasets T against ChiSquareDistribution[k - 1]; [{data1, data2}, r0] scales data1 by 1/Sqrt[r0]. "
          "One dataset and multivariate data raise. Check: reference values from an independent implementation of "
          "Conover (1999, sec. 5.3) (R: conover.test / fligner-type squared ranks).")
def _conover_test(x: ListOf[ListOf[Real]], *rest: Real | String | Option) -> Real | Expr:
    name = "ConoverTest"
    args, o = opts(rest, TEST_OPTS, name)
    (r0,), prop = prop_arg(args, name, 1)
    verify_opt(o, name)
    alt = alt_of(o, name)
    gs = groups(x, name)
    r = _ratio_arg(r0, name)
    if r != 1.0:
        if len(gs) != 2:
            raise ValueError(f"{name}: a dispersion ratio needs exactly two datasets")
        gs = [[t / math.sqrt(r) for t in gs[0]], gs[1]]
    try:
        T, df, S, Sbar, D2, z = _conover(gs)
    except ValueError as e:
        raise ValueError(f"{name}: {e}")
    if len(gs) == 2:
        n1, n2 = len(gs[0]), len(gs[1])
        N = n1 + n2
        stat = (S[0] - n1 * Sbar) / math.sqrt(n1 * n2 / (N - 1) * D2 * (N - 1) / N) if False else \
            (S[0] - n1 * Sbar) / math.sqrt(n1 * n2 * D2 / N)
        return test_result(name, stat, p_normal(stat, alt), prop, o)
    return test_result(name, T, chi2_sf(T, df), prop, o, df)


def _siegel_tukey_ranks(vals):
    """Siegel-Tukey ranks of the pooled values (alternating from the extremes; ties averaged)"""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    pos_rank = [0] * n
    lo, hi = 0, n - 1
    rank = 1
    take_low = True
    while lo <= hi:
        if take_low:
            pos_rank[lo] = rank
            rank += 1
            lo += 1
            if lo <= hi:
                pos_rank[lo] = rank
                rank += 1
                lo += 1
        else:
            pos_rank[hi] = rank
            rank += 1
            hi -= 1
            if lo <= hi:
                pos_rank[hi] = rank
                rank += 1
                hi -= 1
        take_low = not take_low
    # first step takes only one low value
    pos_rank = [0] * n
    seq = []
    lo, hi = 0, n - 1
    step = 0
    while lo <= hi:
        if step % 4 in (0, 3):
            seq.append(lo)
            lo += 1
        else:
            seq.append(hi)
            hi -= 1
        step += 1
    for r_, p in enumerate(seq, start=1):
        pos_rank[p] = r_
    rk = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = sum(pos_rank[q] for q in range(i, j + 1)) / (j - i + 1)
        for q in range(i, j + 1):
            rk[order[q]] = avg
        i = j + 1
    return rk


@fn("SiegelTukeyTest", area=AREA, status="added",
    notes="SiegelTukeyTest[{data1, data2}] and [{data1, data2}, r0]: the Siegel-Tukey rank test of equal "
          "dispersion; data are centred at their medians, data1 is scaled by 1/Sqrt[r0], the pooled sample gets "
          "Siegel-Tukey ranks (1 to the smallest, 2 and 3 to the two largest, 4 and 5 to the next smallest, ...; "
          "ties averaged), and the standardized rank sum of data1 is compared with NormalDistribution[] (as "
          "documented); options AlternativeHypothesis, SignificanceLevel. More than two datasets, one dataset and "
          "multivariate data raise. Check: reference values from an independent implementation of Siegel and Tukey "
          "(1960) with the normal approximation.")
def _siegel(x: ListOf[ListOf[Real]], *rest: Real | String | Option) -> Real | Expr:
    name = "SiegelTukeyTest"
    args, o = opts(rest, TEST_OPTS, name)
    (r0,), prop = prop_arg(args, name, 1)
    verify_opt(o, name)
    gs = groups(x, name)
    if len(gs) != 2:
        raise ValueError(f"{name}: needs exactly two datasets")
    r = _ratio_arg(r0, name)
    a = [(t - fl(median(gs[0]))) / math.sqrt(r) for t in gs[0]]
    b = [t - fl(median(gs[1])) for t in gs[1]]
    rk = _siegel_tukey_ranks(a + b)
    n1, n2 = len(a), len(b)
    N = n1 + n2
    W1 = math.fsum(rk[:n1])
    mean_ = n1 * (N + 1) / 2
    var = n1 * n2 / (N * (N - 1)) * math.fsum(q * q for q in rk) - n1 * n2 * (N + 1) ** 2 / (4 * (N - 1))
    z = (W1 - mean_) / math.sqrt(var)
    return test_result(name, z, p_normal(z, alt_of(o, name)), prop, o)


_VE_TESTS = ("Bartlett", "BrownForsythe", "Conover", "FisherRatio", "Levene")


def _ve_one(test, gs, o):
    if test == "Bartlett":
        s, df = _bartlett(gs)
        return s, chi2_sf(s, df), df
    if test in ("Levene", "BrownForsythe"):
        c = _fmean if test == "Levene" else (lambda g: fl(median(g)))
        f, d1, d2 = _anova([[abs(t - c(g)) for t in g] for g in gs])
        return f, f_sf(f, d1, d2), L(d1, d2)
    if test == "Conover":
        T, df, *_ = _conover(gs)
        return T, chi2_sf(T, df), df
    if test == "FisherRatio":
        if len(gs) != 2:
            raise ValueError("VarianceEquivalenceTest: the FisherRatio test needs exactly two datasets")
        a, b = gs
        f = _fvar(a) / _fvar(b)
        hi = f_sf(f, len(a) - 1, len(b) - 1)
        return f, min(1.0, 2 * min(hi, 1 - hi)), L(len(a) - 1, len(b) - 1)
    raise ValueError(f"VarianceEquivalenceTest: unknown test {test!r}")


def _equiv_impl(name, tests, one):
    def impl(x: ListOf[ListOf[Real]], *rest: String | ListOf[String] | Expr | Option) -> Real | Expr:
        args, o = opts(rest, ("SignificanceLevel", "VerifyTestAssumptions"), name)
        verify_opt(o, name)
        gs = groups(x, name)
        prop, test = None, None
        if len(args) > 2:
            raise ValueError(f"{name}: too many arguments")
        if len(args) == 2:
            prop, test = args
        elif len(args) == 1:
            a = args[0]
            if isinstance(a, str) and a in tests:
                test = a
            elif isinstance(a, Symbol) and a.name in ("All", "Automatic"):
                test = a.name
            else:
                prop = a
        if isinstance(test, Symbol):
            test = test.name
        if test in (None, "Automatic", "All"):
            raise NotImplementedError(f"{name}: automatic test selection is not supported; name the test, e.g. "
                                      f"{name}[data, \"PValue\", \"{tests[0]}\"] (one of {', '.join(tests)})")
        if test not in tests:
            raise ValueError(f"{name}: unknown test {test!r} (use one of {', '.join(tests)})")
        try:
            s, p, df = one(test, gs, o)
        except ValueError as e:
            msg = str(e)
            raise ValueError(msg if msg.startswith(name) else f"{name}: {msg}")
        return test_result(name, s, p, prop if not (isinstance(prop, Symbol)) else None, o, df)
    return impl


fn("VarianceEquivalenceTest", area=AREA, status="added",
   notes="VarianceEquivalenceTest[{data1, data2, ...}, \"test\"] and [data, prop, \"test\"] for the tests "
         "\"Bartlett\" (chi-square), \"BrownForsythe\" and \"Levene\" (ANOVA F on absolute deviations), \"Conover\" "
         "(squared ranks, chi-square) and \"FisherRatio\" (two datasets, two-sided F); Wolfram's automatic test "
         "selection (no test, Automatic or All) raises NotImplementedError; unknown tests ValueError; options "
         "SignificanceLevel, VerifyTestAssumptions. Check: reference values from scipy.stats.bartlett and "
         "scipy.stats.levene.")(_equiv_impl("VarianceEquivalenceTest", _VE_TESTS, _ve_one))


def _kruskal(gs):
    flat = [t for g in gs for t in g]
    N = len(flat)
    r = ranks(flat)
    pos = 0
    H = 0.0
    for g in gs:
        rs = float(sum(r[pos:pos + len(g)]))
        H += rs * rs / len(g)
        pos += len(g)
    H = 12 / (N * (N + 1)) * H - 3 * (N + 1)
    t = {}
    for q in r:
        t[q] = t.get(q, 0) + 1
    c = 1 - sum(k ** 3 - k for k in t.values()) / (N ** 3 - N)
    if c == 0:
        raise ValueError("all values are equal")
    return H / c, N


def _le_one(test, gs, o):
    k = len(gs)
    if test == "KruskalWallis":
        H, N = _kruskal(gs)
        # p-value from the F approximation to the rank ANOVA, F = (N - k) H/((k - 1)(N - 1 - H)) (the
        # Wolfram documentation example: H = 43.6429, p = 1.17225*10^-14)
        if N - 1 - H <= 0:
            return H, 0.0, L(k - 1, N - k)
        F = (N - k) * H / ((k - 1) * (N - 1 - H))
        return H, f_sf(F, k - 1, N - k), L(k - 1, N - k)
    if test == "KSampleT":
        f, d1, d2 = _anova(gs)
        return f, f_sf(f, d1, d2), L(d1, d2)
    if test in ("CompleteBlockF", "FriedmanRank"):
        n = len(gs[0])
        if any(len(g) != n for g in gs):
            raise ValueError(f"the {test} test needs datasets of equal length (complete blocks)")
        if test == "CompleteBlockF":
            gm = math.fsum(t for g in gs for t in g) / (k * n)
            sst = math.fsum(n * (_fmean(g) - gm) ** 2 for g in gs)
            ssb = math.fsum(k * (math.fsum(g[j] for g in gs) / k - gm) ** 2 for j in range(n))
            sse = math.fsum((t - gm) ** 2 for g in gs for t in g) - sst - ssb
            d1, d2 = k - 1, (k - 1) * (n - 1)
            if sse <= 0:
                raise ValueError("the residual variation is zero")
            f = (sst / d1) / (sse / d2)
            return f, f_sf(f, d1, d2), L(d1, d2)
        # Friedman: ranks within each block (row), tie-corrected chi-square statistic
        R = [Fraction(0)] * k
        ties = 0
        for j in range(n):
            row = [g[j] for g in gs]
            rr = ranks(row)
            for i in range(k):
                R[i] += rr[i]
            cnt = {}
            for q in rr:
                cnt[q] = cnt.get(q, 0) + 1
            ties += sum(c ** 3 - c for c in cnt.values())
        Q = 12 * sum(float(Ri) ** 2 for Ri in R) / (n * k * (k + 1)) - 3 * n * (k + 1)
        Q = Q / (1 - ties / (n * k * (k * k - 1)))
        return Q, chi2_sf(Q, k - 1), k - 1
    raise ValueError(f"unknown test {test!r}")


fn("LocationEquivalenceTest", area=AREA, status="added",
   notes="LocationEquivalenceTest[{data1, data2, ...}, \"test\"] and [data, prop, \"test\"] for \"KruskalWallis\" "
         "(tie-corrected H; p-value from the F approximation (N - k) H/((k - 1)(N - 1 - H)) ~ FRatioDistribution[k - "
         "1, N - k], which reproduces the Wolfram documentation example), \"KSampleT\" (one-way ANOVA F), "
         "\"CompleteBlockF\" (randomized complete block ANOVA, equal lengths) and \"FriedmanRank\" (tie-corrected "
         "Friedman chi-square); automatic test selection raises NotImplementedError; options SignificanceLevel, "
         "VerifyTestAssumptions. Check: reference values from the Wolfram documentation (H = 43.6429, p = "
         "1.17225*10^-14), scipy.stats.f_oneway and scipy.stats.friedmanchisquare.")(
    _equiv_impl("LocationEquivalenceTest", ("KruskalWallis", "KSampleT", "CompleteBlockF", "FriedmanRank"), _le_one))


# ─────────────────────────── independence / correlation tests ───────────────────────────

def _two_vectors(v1, v2, fname, min_len=3):
    if is_matrix(v1) or is_matrix(v2):
        raise NotImplementedError(f"{fname}: matrix (multivariate) arguments are not supported")
    a, _ = vec(v1, fname, min_len=min_len)
    b, _ = vec(v2, fname, min_len=min_len)
    if len(a) != len(b):
        raise ValueError(f"{fname}: the vectors must have the same length")
    return _fvec(a), _fvec(b)


def _pearson_r(a, b):
    ma, mb = _fmean(a), _fmean(b)
    sab = math.fsum((p - ma) * (q - mb) for p, q in zip(a, b))
    saa = math.fsum((p - ma) ** 2 for p in a)
    sbb = math.fsum((q - mb) ** 2 for q in b)
    if saa == 0 or sbb == 0:
        raise ValueError("a vector is constant, so the correlation is undefined")
    return sab / math.sqrt(saa * sbb)


def _r_t_p(r, n, alt):
    if abs(r) >= 1:
        t = math.copysign(math.inf, r)
        return 0.0 if alt == "Unequal" else (0.0 if (alt == "Greater") == (r > 0) else 1.0)
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return p_t(t, n - 2, alt)


def _kendall_parts(a, b):
    n = len(a)
    S = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] > a[j]) - (a[i] < a[j])
            t = (b[i] > b[j]) - (b[i] < b[j])
            S += s * t

    def tie_terms(v):
        c = {}
        for t in v:
            c[t] = c.get(t, 0) + 1
        g = [k for k in c.values() if k > 1]
        return (sum(k * (k - 1) * (2 * k + 5) for k in g), sum(k * (k - 1) for k in g),
                sum(k * (k - 1) * (k - 2) for k in g))
    ta, tb = tie_terms(a), tie_terms(b)
    var = (n * (n - 1) * (2 * n + 5) - ta[0] - tb[0]) / 18 + ta[2] * tb[2] / (9 * n * (n - 1) * (n - 2)) + \
        ta[1] * tb[1] / (2 * n * (n - 1))
    n0 = n * (n - 1) / 2
    tau = S / math.sqrt((n0 - ta[1] / 2) * (n0 - tb[1] / 2))
    return S, var, tau


def _corr_test(name, kind):
    def impl(v1: ListOf[Real], v2: ListOf[Real], *rest: String | Option) -> Real | Expr:
        args, o = opts(rest, TEST_OPTS, name)
        (), prop = prop_arg(args, name, 0)
        verify_opt(o, name)
        alt = alt_of(o, name)
        a, b = _two_vectors(v1, v2, name)
        n = len(a)
        try:
            if kind == "Pearson":
                r = _pearson_r(a, b)
                return test_result(name, r, _r_t_p(r, n, alt), prop, o, n - 2)
            if kind == "Spearman":
                r = _pearson_r([float(q) for q in ranks(a)], [float(q) for q in ranks(b)])
                return test_result(name, r, _r_t_p(r, n, alt), prop, o, n - 2)
        except ValueError as e:
            raise ValueError(f"{name}: {e}")
        if kind == "Kendall":
            S, var, tau = _kendall_parts(a, b)
            if var <= 0:
                raise ValueError(f"{name}: a vector is constant")
            return test_result(name, tau, p_normal(S / math.sqrt(var), alt), prop, o)
        if kind == "GK":
            if has_ties(a) or has_ties(b):
                raise NotImplementedError(f"{name}: data with ties is not supported")
            S, var, _ = _kendall_parts(a, b)
            C_D = S
            gamma = C_D / (n * (n - 1) / 2)
            sd = math.sqrt(var)
            if alt == "Unequal":
                p = min(1.0, 2 * norm_sf((abs(C_D) - 1) / sd))
            elif alt == "Greater":
                p = norm_sf((C_D - 1) / sd)
            else:
                p = norm_cdf((C_D + 1) / sd)
            return test_result(name, gamma, p, prop, o)
        if kind == "Blomqvist":
            ma, mb = fl(median(a)), fl(median(b))
            pairs = [(p - ma, q - mb) for p, q in zip(a, b) if p != ma and q != mb]
            n11 = sum(1 for p, q in pairs if p > 0 and q > 0)
            n12 = sum(1 for p, q in pairs if p > 0 > q)
            n21 = sum(1 for p, q in pairs if p < 0 < q)
            n22 = sum(1 for p, q in pairs if p < 0 and q < 0)
            beta = _blomqvist(L(*a), L(*b))
            p = _fisher_exact(n11, n12, n21, n22, alt)
            return test_result(name, beta, p, prop, o)
        if kind == "Hoeffding":
            d = _hoeffding(L(*a), L(*b))
            if prop != "TestStatistic":
                raise NotImplementedError(f'{name}: only the "TestStatistic" property is supported (Wolfram\'s '
                                          "null distribution for the p-value is not documented)")
            return test_result(name, d, 0.0, prop, o)
        raise AssertionError(kind)
    return impl


def _fisher_exact(a, b, c, d, alt):
    """Fisher's exact test for the 2x2 table {{a, b}, {c, d}} (hypergeometric distribution of a)"""
    r1, c1, N = a + b, a + c, a + b + c + d
    lo_, hi_ = max(0, r1 + c1 - N), min(r1, c1)

    def pmf(k):
        return Fraction(math.comb(c1, k) * math.comb(N - c1, r1 - k), math.comb(N, r1))
    lower = sum(pmf(k) for k in range(lo_, a + 1))
    upper = sum(pmf(k) for k in range(a, hi_ + 1))
    if alt == "Greater":
        return float(upper)
    if alt == "Less":
        return float(lower)
    return float(min(Fraction(1), 2 * min(lower, upper)))


_CORR_NOTE = (" Matrix (multivariate) arguments raise NotImplementedError; unequal lengths or constant vectors "
              "ValueError; non-numeric data TypeError; symbolic data stays unevaluated.")
fn("PearsonCorrelationTest", area=AREA, status="added",
   notes="PearsonCorrelationTest[v1, v2] and [v1, v2, prop]: the statistic is Correlation[v1, v2] = r and the p-value "
         "comes from t = r Sqrt[(n - 2)/(1 - r^2)] ~ StudentTDistribution[n - 2] (as documented); options "
         "AlternativeHypothesis, SignificanceLevel, VerifyTestAssumptions." + _CORR_NOTE +
         " Check: reference values from scipy.stats.pearsonr.")(_corr_test("PearsonCorrelationTest", "Pearson"))
fn("SpearmanRankTest", area=AREA, status="added",
   notes="SpearmanRankTest[v1, v2] and [v1, v2, prop]: the statistic is SpearmanRho (Pearson correlation of mid-ranks) "
         "and the asymptotic p-value uses t = rho Sqrt[(n - 2)/(1 - rho^2)] ~ StudentTDistribution[n - 2] (this "
         "reproduces the Wolfram documentation example: rho = -0.748287, n = 100, p = 3.63*10^-19); options "
         "AlternativeHypothesis, SignificanceLevel." + _CORR_NOTE +
         " Check: reference values from scipy.stats.spearmanr and the Wolfram documentation example.")(
    _corr_test("SpearmanRankTest", "Spearman"))
fn("KendallTauTest", area=AREA, status="added",
   notes="KendallTauTest[v1, v2] and [v1, v2, prop]: the statistic is Kendall's tau-b and the asymptotic p-value uses "
         "S = C - D with the tie-corrected variance (no continuity correction; this reproduces the Wolfram "
         "documentation example tau = -0.602828, n = 100, p = 6.295*10^-19); options AlternativeHypothesis, "
         "SignificanceLevel." + _CORR_NOTE + " Check: reference values from scipy.stats.kendalltau(method="
         "\"asymptotic\") and the Wolfram documentation example.")(_corr_test("KendallTauTest", "Kendall"))
fn("GoodmanKruskalGammaTest", area=AREA, status="added",
   notes="GoodmanKruskalGammaTest[v1, v2] and [v1, v2, prop] for data without ties: the statistic is "
         "GoodmanKruskalGamma and the asymptotic p-value uses (|C - D| - 1)/Sqrt[n (n - 1)(2 n + 5)/18] ~ "
         "NormalDistribution[] (reproduces the Wolfram documentation example gamma = -0.567273, n = 100, p = "
         "6.29452*10^-17); options AlternativeHypothesis, SignificanceLevel. Ties raise NotImplementedError." +
         _CORR_NOTE + " Check: reference values from the Wolfram documentation example and the closed form.")(
    _corr_test("GoodmanKruskalGammaTest", "GK"))
fn("BlomqvistBetaTest", area=AREA, status="added",
   notes="BlomqvistBetaTest[v1, v2] and [v1, v2, prop]: the statistic is BlomqvistBeta and the p-value is Fisher's "
         "exact test on the 2x2 table of signs about the medians (values equal to a median dropped; two-sided = 2 "
         "Min[tails]; reproduces the Wolfram documentation example beta = -0.48, n = 100, p = 2.8183*10^-6); options "
         "AlternativeHypothesis, SignificanceLevel." + _CORR_NOTE + " Check: exact hypergeometric tail; reference "
         "values from scipy.stats.fisher_exact and the Wolfram documentation example.")(
    _corr_test("BlomqvistBetaTest", "Blomqvist"))
fn("HoeffdingDTest", area=AREA, status="added",
   notes="HoeffdingDTest[v1, v2, \"TestStatistic\"] gives HoeffdingD[v1, v2]; the p-value (the default property) "
         "and every other property raise NotImplementedError, because Wolfram's null distribution for the p-value is "
         "not documented (the Blum-Kiefer-Rosenblatt asymptotic does not reproduce the documentation example)." +
         _CORR_NOTE + " Check: exact (the statistic); reference value from the Wolfram documentation (211/672).")(
    _corr_test("HoeffdingDTest", "Hoeffding"))


_INDEP = {"PearsonCorrelation": "Pearson", "SpearmanRank": "Spearman", "KendallTau": "Kendall",
          "GoodmanKruskalGamma": "GK", "BlomqvistBeta": "Blomqvist", "HoeffdingD": "Hoeffding"}


def _select_test(name, rest, allowed):
    args, o = opts(rest, TEST_OPTS, name)
    prop, test = None, None
    if len(args) > 2:
        raise ValueError(f"{name}: too many arguments")
    if len(args) == 2:
        prop, test = args
    elif len(args) == 1:
        if isinstance(args[0], str) and args[0] in allowed:
            test = args[0]
        elif isinstance(args[0], Symbol) and args[0].name in ("All", "Automatic"):
            test = args[0].name
        else:
            prop = args[0]
    if isinstance(test, Symbol):
        test = test.name
    if test in (None, "Automatic", "All"):
        raise NotImplementedError(f"{name}: automatic test selection is not supported; name the test (one of "
                                  f"{', '.join(allowed)})")
    if test not in allowed:
        if test in ("PillaiTrace", "WilksW"):
            raise NotImplementedError(f"{name}: the {test} test is not supported here")
        raise ValueError(f"{name}: unknown test {test!r}")
    return prop, test, o


@fn("IndependenceTest", area=AREA, status="added",
    notes="IndependenceTest[v1, v2, \"test\"] and [v1, v2, prop, \"test\"] with test one of \"PearsonCorrelation\", "
          "\"SpearmanRank\", \"KendallTau\", \"GoodmanKruskalGamma\", \"BlomqvistBeta\" (as the corresponding *Test "
          "functions) and \"HoeffdingD\" (statistic only); automatic test selection (no test, Automatic, All) and the "
          "matrix tests \"PillaiTrace\" and \"WilksW\" raise NotImplementedError. Check: delegated to the individual "
          "tests (identical results, checked in tests).")
def _independence(v1: ListOf[Real], v2: ListOf[Real], *rest: String | Option) -> Real | Expr:
    prop, test, o = _select_test("IndependenceTest", rest, tuple(_INDEP))
    extra = [prop] if prop is not None else []
    optr = [rule(Symbol(k), v) for k, v in o.items()]
    return _corr_test("IndependenceTest", _INDEP[test])(v1, v2, *extra, *optr)


@fn("CorrelationTest", area=AREA, status="added",
    notes="CorrelationTest[{{x1, y1}, ...}, \"test\"], [data, 0, \"test\"] and [data, 0, prop, \"test\"] with test "
          "\"PearsonCorrelation\" or \"SpearmanRank\": the test of zero correlation (as PearsonCorrelationTest / "
          "SpearmanRankTest of the two columns); a non-zero rho0, the comparison of two populations and automatic "
          "test selection raise NotImplementedError. Check: delegated to PearsonCorrelationTest / SpearmanRankTest "
          "(identical results, checked in tests).")
def _correlation_test(data: MatrixOf[Real], *rest: Real | String | Option) -> Real | Expr:
    name = "CorrelationTest"
    rest = list(rest)
    if rest and not isinstance(rest[0], str) and not _is_rule(rest[0]) and not (isinstance(rest[0], Symbol) and
                                                                               rest[0].name in ("All", "Automatic")):
        r0 = rest.pop(0)
        if is_matrix(r0):
            raise NotImplementedError(f"{name}: comparing the correlations of two populations is not supported")
        if _num(r0, name, "rho0") != 0:
            raise NotImplementedError(f"{name}: testing a non-zero correlation rho0 is not supported")
    prop, test, o = _select_test(name, rest, ("PearsonCorrelation", "SpearmanRank"))
    m = mat(data, name, min_rows=3)
    if len(m[0]) != 2:
        raise ValueError(f"{name}: the data must be bivariate {{{{x1, y1}}, ...}}")
    cols = columns([list(r.args) for r in data.args])
    extra = [prop] if prop is not None else []
    optr = [rule(Symbol(k), v) for k, v in o.items()]
    return _corr_test(name, _INDEP[test])(L(*cols[0]), L(*cols[1]), *extra, *optr)


# ─────────────────────────── goodness-of-fit tests ───────────────────────────

def _poly(c, x):
    r = 0.0
    for k in reversed(c):
        r = r * x + k
    return r


def _ninv(p):
    return float(mpmath.sqrt(2) * mpmath.erfinv(2 * mpmath.mpf(p) - 1))


def shapiro_wilk(x):
    """Royston (1995, algorithm AS R94) W statistic and p-value for sorted machine data x, 3 <= n <= 5000"""
    n = len(x)
    if n < 3:
        raise ValueError("at least 3 values are needed")
    if n > 5000:
        raise NotImplementedError("samples larger than 5000 are not supported (Royston's approximation)")
    rng_ = x[-1] - x[0]
    if rng_ == 0:
        raise ValueError("the data is constant")
    nn2 = n // 2
    if n == 3:
        a = [math.sqrt(0.5)]
    else:
        an25 = n + 0.25
        m = [-_ninv((i - 0.375) / an25) for i in range(1, nn2 + 1)]   # positive, largest first
        summ2 = 2 * math.fsum(t * t for t in m)
        ssumm2 = math.sqrt(summ2)
        rsn = 1 / math.sqrt(n)
        a1 = _poly([0., .221157, -.147981, -2.07119, 4.434685, -2.706056], rsn) + m[0] / ssumm2
        if n > 5:
            a2 = m[1] / ssumm2 + _poly([0., .042981, -.293762, -1.752461, 5.682633, -3.582633], rsn)
            fac = math.sqrt((summ2 - 2 * m[0] ** 2 - 2 * m[1] ** 2) / (1 - 2 * a1 ** 2 - 2 * a2 ** 2))
            a = [a1, a2] + [m[i] / fac for i in range(2, nn2)]
        else:
            fac = math.sqrt((summ2 - 2 * m[0] ** 2) / (1 - 2 * a1 ** 2))
            a = [a1] + [m[i] / fac for i in range(1, nn2)]
    num = math.fsum(a[i] * (x[n - 1 - i] - x[i]) for i in range(nn2))
    mx = _fmean(x)
    W = num * num / math.fsum((t - mx) ** 2 for t in x)
    W = min(W, 1.0)
    if n == 3:
        p = 6 / math.pi * (math.asin(math.sqrt(W)) - math.asin(math.sqrt(0.75)))
        return W, max(0.0, min(1.0, p))
    w1 = math.log(1 - W) if W < 1 else -math.inf
    if n <= 11:
        gamma = _poly([-2.273, .459], n)
        if w1 >= gamma:
            return W, 1e-99
        y = -math.log(gamma - w1)
        mu = _poly([.544, -.39978, .025054, -6.714e-4], n)
        s = math.exp(_poly([1.3822, -.77857, .062767, -.0020322], n))
    else:
        xx = math.log(n)
        y = w1
        mu = _poly([-1.5861, -.31082, -.083751, .0038915], xx)
        s = math.exp(_poly([-.4803, -.082676, .0030302], xx))
    if y == -math.inf:
        return W, 1.0
    return W, norm_sf((y - mu) / s)


def _uni_data(x, fname, min_len=3):
    if is_matrix(x):
        raise NotImplementedError(f"{fname}: multivariate data is not supported")
    v, _ = vec(x, fname, min_len=min_len)
    return sorted(_fvec(v))


GOF_OPTS = ("SignificanceLevel", "VerifyTestAssumptions", "Method")


def _gof_method(o, fname):
    m = o.get("Method")
    if m is None or str_or_sym(m) == "Automatic":
        return
    raise NotImplementedError(f"{fname}: Method -> {m} (e.g. Monte Carlo p-values) is not supported")


@fn("ShapiroWilkTest", area=AREA, status="added",
    notes="ShapiroWilkTest[data] and [data, prop] (also [data, Automatic, prop]): the Shapiro-Wilk test of normality, "
          "W statistic and p-value by Royston's algorithm (AS R94, 3 <= n <= 5000). Multivariate data, the "
          "\"MonteCarlo\" method and n > 5000 raise NotImplementedError; constant data or n < 3 ValueError. Check: "
          "reference values from scipy.stats.shapiro (the same algorithm) and the exact n = 3 distribution.")
def _shapiro(x: ListOf[Real], *rest: Expr | String | Option) -> Real | Expr:
    name = "ShapiroWilkTest"
    args, o = opts(rest, GOF_OPTS, name)
    if args and isinstance(args[0], Symbol) and args[0].name == "Automatic":
        args = args[1:]
    (), prop = prop_arg(args, name, 0)
    _gof_method(o, name)
    try:
        W, p = shapiro_wilk(_uni_data(x, name))
    except ValueError as e:
        raise ValueError(f"{name}: {e}")
    except NotImplementedError as e:
        raise NotImplementedError(f"{name}: {e}")
    return test_result(name, W, p, prop, o)


@fn("JarqueBeraALMTest", area=AREA, status="added",
    notes="JarqueBeraALMTest[data] and [data, prop]: Urzua's adjusted Lagrange multiplier statistic ALM = b1/c1 + "
          "(b2 - c2)^2/c3 with b1 = Skewness[data]^2, b2 = Kurtosis[data] and the exact normal-theory moments c1 = "
          "6 (n - 2)/((n + 1)(n + 3)), c2 = 3 (n - 1)/(n + 1), c3 = 24 n (n - 2)(n - 3)/((n + 1)^2 (n + 3)(n + 5)); "
          "the p-value uses the asymptotic ChiSquareDistribution[2] (documented as the null distribution; Wolfram's "
          "finite-sample p-values can differ in the second digit for small n). Multivariate data and Monte Carlo "
          "methods raise NotImplementedError; constant data ValueError. Check: reference values from the closed form "
          "with numpy moments (Urzua 1996) and the chi-square tail.")
def _jbalm(x: ListOf[Real], *rest: String | Option) -> Real | Expr:
    name = "JarqueBeraALMTest"
    args, o = opts(rest, GOF_OPTS, name)
    (), prop = prop_arg(args, name, 0)
    _gof_method(o, name)
    v = _uni_data(x, name, 4)
    n = len(v)
    m = _fmean(v)
    m2 = math.fsum((t - m) ** 2 for t in v) / n
    if m2 == 0:
        raise ValueError(f"{name}: the data is constant")
    m3 = math.fsum((t - m) ** 3 for t in v) / n
    m4 = math.fsum((t - m) ** 4 for t in v) / n
    b1 = m3 * m3 / m2 ** 3
    b2 = m4 / m2 ** 2
    c1 = 6 * (n - 2) / ((n + 1) * (n + 3))
    c2 = 3 * (n - 1) / (n + 1)
    c3 = 24 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    alm = b1 / c1 + (b2 - c2) ** 2 / c3
    return test_result(name, alm, chi2_sf(alm, 2), prop, o, 2)


def _ad_inf(z):
    """Marsaglia & Marsaglia (2004) asymptotic Anderson-Darling CDF"""
    if z <= 0:
        return 0.0
    if z < 2:
        return math.exp(-1.2337141 / z) / math.sqrt(z) * (2.00012 + (.247105 - (.0649821 - (.0347962 - (
            .011672 - .00168691 * z) * z) * z) * z) * z)
    return math.exp(-math.exp(1.0776 - (2.30695 - (.43424 - (.082433 - (.008056 - .0003146 * z) * z) * z) * z) * z))


def _ad_errfix(n, x):
    c = .01265 + .1757 / n
    if x < c:
        t = x / c
        t = math.sqrt(t) * (1 - t) * (49 * t - 102)
        return t * (.0037 / (n * n) + .00078 / n + .00006) / n
    if x < .8:
        t = (x - c) / (.8 - c)
        t = -.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t
        return t * (.04213 / n + .01365 / (n * n)) / n
    return (-130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * x) * x) * x) * x) * x) / n


def ad_sf(z, n):
    """P(A^2 > z) for a sample of size n from a fully specified continuous distribution"""
    x = _ad_inf(z)
    return min(1.0, max(0.0, 1 - (x + _ad_errfix(n, x))))


def cvm_inf_cdf(x):
    """asymptotic Cramer-von Mises CDF (Anderson & Darling 1952 series)"""
    if x <= 0:
        return 0.0
    x = mpmath.mpf(x)
    tot = mpmath.mpf(0)
    for k in range(0, 60):
        y = 4 * k + 1
        q = mpmath.mpf(y) ** 2 / (16 * x)
        if q > 700:
            break
        term = mpmath.exp(mpmath.loggamma(k + 0.5) - mpmath.loggamma(k + 1)) / (mpmath.pi ** 1.5 * mpmath.sqrt(x)) \
            * mpmath.sqrt(y) * mpmath.exp(-q) * mpmath.besselk(0.25, q)
        tot += term
        if abs(term) < mpmath.mpf(10) ** -20:
            break
    return float(min(1, max(0, tot)))


def _cvm_psi1(x):
    """Csorgo & Faraway (1996) first-order finite-sample correction term"""
    x = mpmath.mpf(x)

    def ed2(y):
        z = y * y / 4
        return mpmath.exp(-z) * (y / 2) ** 1.5 * (mpmath.besselk(0.25, z) + mpmath.besselk(0.75, z)) / \
            mpmath.sqrt(mpmath.pi)

    def ed3(y):
        z = y * y / 4
        return mpmath.exp(-z) / mpmath.sqrt(mpmath.pi) * (y / 2) ** 2.5 * (
            2 * mpmath.besselk(0.25, z) + 3 * mpmath.besselk(0.75, z) - mpmath.besselk(1.25, z))

    tot = mpmath.mpf(0)
    sx = 2 * mpmath.sqrt(x)
    y1, y2 = x ** 0.75, x ** 1.25
    for k in range(0, 200):
        m = 2 * k + 1
        g = mpmath.gamma(k + 0.5)
        a = (m * g * ed2((4 * k + 3) / sx) / (9 * y1) + g * ed3((4 * k + 1) / sx) / (72 * y2)
             + 2 * (m + 2) * mpmath.gamma(k + 1.5) * ed3((4 * k + 5) / sx) / (12 * y2)
             + 7 * m * g * ed2((4 * k + 1) / sx) / (144 * y1) + 7 * m * g * ed2((4 * k + 5) / sx) / (144 * y1))
        z = -a / (mpmath.pi * mpmath.factorial(k))
        tot += z
        if abs(z) < 1e-12:
            break
    return tot


def cvm_sf(x, n):
    """P(W^2 > x), Csorgo-Faraway finite-n approximation"""
    if x <= 1 / (12 * n):
        return 1.0
    if x >= n / 3:
        return 0.0
    c = cvm_inf_cdf(x) * (1 + 1 / (12 * n)) + float(_cvm_psi1(x)) / n
    return min(1.0, max(0.0, 1 - c))


def kuiper_sf(lam):
    if lam < 0.4:
        return 1.0
    return float(min(1.0, sum(2 * (4 * k * k * lam * lam - 1) * math.exp(-2 * k * k * lam * lam)
                              for k in range(1, 200))))


def watson_sf(u):
    if u <= 0:
        return 1.0
    if u < 0.02:
        return 1.0
    return float(min(1.0, sum(2 * (-1) ** (k - 1) * math.exp(-2 * k * k * math.pi ** 2 * u) for k in range(1, 200))))


def _cdf_values(v, d, fname):
    dd = W.dist_of(d, fname)
    if dd.discrete:
        raise NotImplementedError(f"{fname}: discrete distributions are not supported")
    if not dd.numeric():
        raise NotImplementedError(f"{fname}: distributions with estimated (symbolic) parameters are not supported "
                                  "(Wolfram computes those p-values by Monte Carlo simulation)")
    out_ = []
    for t in v:
        F = W.cdf_value(dd, sp.Float(t, 30))
        out_.append(float(sp.N(F, 20)))
    return out_


def _edf_stats(u):
    """A^2, W^2, U^2 and V for the sorted probability-integral transforms u"""
    n = len(u)
    eps = 1e-300
    A = -n - math.fsum((2 * i - 1) * (math.log(max(u[i - 1], eps)) + math.log(max(1 - u[n - i], eps)))
                       for i in range(1, n + 1)) / n
    W2 = 1 / (12 * n) + math.fsum((u[i - 1] - (2 * i - 1) / (2 * n)) ** 2 for i in range(1, n + 1))
    U2 = W2 - n * (_fmean(u) - 0.5) ** 2
    Dp = max(i / n - u[i - 1] for i in range(1, n + 1))
    Dm = max(u[i - 1] - (i - 1) / n for i in range(1, n + 1))
    return A, W2, U2, Dp + Dm


def _gof_impl(name, which):
    def impl(x: ListOf[Real], *rest: Expr | String | Option) -> Real | Expr:
        args, o = opts(rest, GOF_OPTS, name)
        _gof_method(o, name)
        d = None
        if args and not isinstance(args[0], str) and not (is_list(args[0]) and args[0].args and
                                                         all(isinstance(q, str) for q in args[0].args)):
            d = args.pop(0)
        (), prop = prop_arg(args, name, 0)
        if d is None or (isinstance(d, Symbol) and d.name == "Automatic"):
            raise NotImplementedError(f"{name}: testing against an estimated (Automatic) distribution is not "
                                      "supported (Wolfram uses Monte Carlo p-values); give a fully specified "
                                      "distribution")
        if is_list(d):
            raise NotImplementedError(f"{name}: the two-sample form is not supported")
        v = _uni_data(x, name, 2)
        n = len(v)
        u = sorted(_cdf_values(v, d, name))
        A, W2, U2, V = _edf_stats(u)
        if which == "AD":
            return test_result(name, A, ad_sf(A, n), prop, o)
        if which == "CvM":
            return test_result(name, W2, cvm_sf(W2, n), prop, o)
        if which == "Watson":
            us = (U2 - 0.1 / n + 0.1 / n ** 2) * (1 + 0.8 / n)
            return test_result(name, U2, watson_sf(us), prop, o)
        if which == "Kuiper":
            lam = V * (math.sqrt(n) + 0.155 + 0.24 / math.sqrt(n))
            return test_result(name, V, kuiper_sf(lam), prop, o)
        raise AssertionError(which)
    return impl


_GOF_NOTE = (" The statistic is computed from the probability-integral transforms CDF[dist, x_i] of the sorted data. "
             "Only fully specified continuous univariate distributions are supported: AutomaticTest[data] / "
             "[data, Automatic] (normality with estimated parameters, Monte Carlo p-values in Wolfram), symbolic "
             "parameters, two-sample and multivariate forms, discrete distributions and Method settings raise "
             "NotImplementedError; options SignificanceLevel, VerifyTestAssumptions.")
fn("AndersonDarlingTest", area=AREA, status="added",
   notes="AndersonDarlingTest[data, dist] and [data, dist, prop]: A^2 = -n - Sum[(2 i - 1)(Log[u_i] + Log[1 - "
         "u_(n+1-i)])]/n; p-value from Marsaglia & Marsaglia's (2004) finite-n Anderson-Darling distribution." +
         _GOF_NOTE.replace("AutomaticTest", "AndersonDarlingTest") + " Check: reference values from the classical "
         "asymptotic critical points (A^2 = 2.492 at 5%, 3.857 at 1%) and the closed-form statistic.")(
    _gof_impl("AndersonDarlingTest", "AD"))
fn("CramerVonMisesTest", area=AREA, status="added",
   notes="CramerVonMisesTest[data, dist] and [data, dist, prop]: W^2 = 1/(12 n) + Sum[(u_i - (2 i - 1)/(2 n))^2]; "
         "p-value from the Csorgo-Faraway (1996) finite-n approximation to the Cramer-von Mises distribution." + _GOF_NOTE.replace("AutomaticTest", "CramerVonMisesTest") +
         " Check: reference values from scipy.stats.cramervonmises (the same approximation) and the closed-form "
         "statistic.")(_gof_impl("CramerVonMisesTest", "CvM"))
fn("WatsonUSquareTest", area=AREA, status="added",
   notes="WatsonUSquareTest[data, dist] and [data, dist, prop]: U^2 = W^2 - n (Mean[u] - 1/2)^2; p-value from the "
         "asymptotic series 2 Sum[(-1)^(k-1) Exp[-2 k^2 Pi^2 U*]] of Stephens' modified U* = (U^2 - 0.1/n + "
         "0.1/n^2)(1 + 0.8/n)." + _GOF_NOTE.replace("AutomaticTest", "WatsonUSquareTest") + " Check: reference "
         "values from the closed-form statistic and Stephens' (1970) critical values (U* = 0.187 at 5%).")(
    _gof_impl("WatsonUSquareTest", "Watson"))
fn("KuiperTest", area=AREA, status="added",
   notes="KuiperTest[data, dist] and [data, dist, prop]: V = D+ + D-; p-value from the asymptotic Kuiper "
         "distribution of Stephens' modified V (Sqrt[n] + 0.155 + 0.24/Sqrt[n])." +
         _GOF_NOTE.replace("AutomaticTest", "KuiperTest") + " Check: reference values from the closed-form "
         "statistic and Stephens' (1970) critical value (1.747 at 5%).")(_gof_impl("KuiperTest", "Kuiper"))


_DFT = {"AndersonDarling": "AndersonDarlingTest", "CramerVonMises": "CramerVonMisesTest",
        "KolmogorovSmirnov": "KolmogorovSmirnovTest", "Kuiper": "KuiperTest", "WatsonUSquare": "WatsonUSquareTest",
        "ShapiroWilk": "ShapiroWilkTest", "JarqueBeraALM": "JarqueBeraALMTest"}


@fn("DistributionFitTest", area=AREA, status="added",
    notes="DistributionFitTest[data, dist, prop, \"test\"] (and [data, dist, \"test\"]) with test one of "
          "\"AndersonDarling\", \"CramerVonMises\", \"KolmogorovSmirnov\", \"Kuiper\", \"WatsonUSquare\" (fully "
          "specified dist) or \"ShapiroWilk\", \"JarqueBeraALM\" (normality; dist may be Automatic or "
          "NormalDistribution[]); identical to the corresponding *Test function. Automatic test selection, "
          "estimated distributions and the Pearson chi-square test raise NotImplementedError. Check: delegated to "
          "the named test functions (identical results checked in tests).")
def _dft(x: ListOf[Real], d: Expr = None, *rest: String | Option) -> Real | Expr:
    name = "DistributionFitTest"
    args, o = opts(rest, GOF_OPTS, name)
    prop, test = None, None
    if len(args) == 2:
        prop, test = args
    elif len(args) == 1:
        if isinstance(args[0], str) and args[0] in _DFT:
            test = args[0]
        else:
            prop = args[0]
    elif len(args) > 2:
        raise ValueError(f"{name}: too many arguments")
    if isinstance(test, Symbol):
        test = test.name
    if test in (None, "Automatic", "All"):
        raise NotImplementedError(f"{name}: automatic test selection is not supported; name the test (one of "
                                  f"{', '.join(_DFT)})")
    if test == "PearsonChiSquare":
        raise NotImplementedError(f"{name}: the Pearson chi-square test is not supported")
    if test not in _DFT:
        raise ValueError(f"{name}: unknown test {test!r}")
    optr = [rule(Symbol(k), v) for k, v in o.items()]
    extra = [prop] if prop is not None else []
    if test in ("ShapiroWilk", "JarqueBeraALM"):
        if d is not None and not (isinstance(d, Symbol) and d.name == "Automatic") and \
                not (isinstance(d, Expr) and d.head == "NormalDistribution" and not d.args):
            raise NotImplementedError(f"{name}: the {test} test only tests normality (use Automatic)")
        return ev(Expr(_DFT[test], x, *extra, *optr))
    if test == "KolmogorovSmirnov":
        if o:
            raise NotImplementedError(f"{name}: options are not supported with the KolmogorovSmirnov test")
        if prop is not None and prop not in ("PValue", "TestStatistic"):
            raise NotImplementedError(f'{name}: the KolmogorovSmirnov test supports "PValue" and "TestStatistic"')
        return ev(Expr("KolmogorovSmirnovTest", x, d, *extra))
    return ev(Expr(_DFT[test], x, d if d is not None else Symbol("Automatic"), *extra, *optr))
