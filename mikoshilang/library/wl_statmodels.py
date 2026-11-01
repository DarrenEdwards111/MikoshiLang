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
    if o and "SignificanceLevel" in o:
        sig_level(o, fname)          # validates the setting even when it is not used
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


def test_spec(fname, args, tests):
    """the Wolfram test specification (last argument of the *EquivalenceTest / IndependenceTest / CorrelationTest /
    DistributionFitTest families): "test", {"test1", ...}, "prop" or {"prop", test | {tests}}.
    Returns (prop or None, [tests], several?)"""
    auto = ("All", "Automatic", "AllTests", "AutomaticTest")
    if len(args) > 1:
        raise ValueError(f"{fname}: too many arguments")
    if not args:
        spec = None
    else:
        spec = args[0]
    if isinstance(spec, Symbol):
        spec = spec.name

    def tests_of(t):
        if isinstance(t, Symbol):
            t = t.name
        if isinstance(t, str):
            if t in auto:
                return None, False
            if t not in tests:
                raise ValueError(f"{fname}: unknown test {t!r} (use one of {', '.join(tests)})")
            return [t], False
        if is_list(t) and t.args:
            out_ = []
            for q in t.args:
                r, _ = tests_of(q)
                if r is None:
                    return None, True
                out_ += r
            return out_, True
        raise TypeError(f"{fname}: invalid test specification {t!r}")
    prop = None
    chosen, several = None, False
    if spec is None or (isinstance(spec, str) and spec in auto):
        chosen = None
    elif isinstance(spec, str) and spec in tests:
        chosen = [spec]
    elif isinstance(spec, str):
        if spec not in TEST_PROPS + ("HypothesisTestData", "TestConclusion") and not spec.endswith("Table"):
            raise ValueError(f"{fname}: unknown test or property {spec!r} (tests: {', '.join(tests)})")
        prop = spec
    elif is_list(spec) and len(spec.args) == 2 and not (isinstance(spec.args[0], str) and spec.args[0] in tests) \
            and not (isinstance(spec.args[0], str) and spec.args[0] in auto):
        prop = spec.args[0]
        chosen, several = tests_of(spec.args[1])
    elif is_list(spec):
        chosen, several = tests_of(spec)
    else:
        raise TypeError(f"{fname}: invalid test specification {spec!r}")
    if chosen is None:
        raise NotImplementedError(f"{fname}: automatic test selection (and All) is not supported; name the test, "
                                  f"e.g. {fname}[..., {{\"PValue\", \"{tests[0]}\"}}] (one of {', '.join(tests)})")
    return prop, chosen, several


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
        if s2 is None:
            var = _fvar(v)
        elif is_list(s2) and len(s2.args) == 2:
            var = math.fsum(_variance_arg(t, fname) for t in s2.args)
        else:
            # a scalar s2 is the (common) variance of each population, so Var[x - y] = 2 s2 (this reproduces
            # the Wolfram documentation example PairedZTest[{armSpan, ht}, .4^2, 0] = 1.73854)
            var = 2 * _variance_arg(s2, fname)
        return (_fmean(v) - m0) / math.sqrt(var / len(v))
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
         "differences data1 - data2) against mu0 with known population variance s2 (for paired data the variance of "
         "each population, so Var[data1 - data2] = 2 s2, or s1^2 + s2^2 for {s1^2, s2^2}; the sample variance of the "
         "differences when omitted); options as for ZTest. Unequal sample lengths raise ValueError; multivariate data NotImplementedError. Check: "
         "reference values from the Wolfram documentation (PairedZTest[{armSpan, ht}, .4^2, 0] = 1.73854, p = "
         "0.0410576 Greater) and the closed form.")(
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
        stat = (S[0] - n1 * Sbar) / math.sqrt(n1 * n2 * D2 / N)
        return test_result(name, stat, p_normal(stat, alt), prop, o)
    return test_result(name, T, chi2_sf(T, df), prop, o, df)


def _siegel_tukey_ranks(vals):
    """Siegel-Tukey ranks of the pooled values (alternating from the extremes; ties averaged)"""
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
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
        prop, chosen, several = test_spec(name, args, tests)
        res = []
        for test in chosen:
            try:
                s_, p_, df = one(test, gs, o)
            except ValueError as e:
                msg = str(e)
                raise ValueError(msg if msg.startswith(name) else f"{name}: {msg}")
            res.append(test_result(name, s_, p_, prop, o, df))
        return L(*res) if several else res[0]
    return impl


fn("VarianceEquivalenceTest", area=AREA, status="added",
   notes="VarianceEquivalenceTest[{data1, data2, ...}, spec] with spec \"test\", {\"test1\", ...} or {prop, test} "
         "(or {prop, {tests}}) for the tests "
         "\"Bartlett\" (chi-square), \"BrownForsythe\" and \"Levene\" (ANOVA F on absolute deviations), \"Conover\" "
         "(squared ranks, chi-square) and \"FisherRatio\" (two datasets, two-sided F); Wolfram's automatic test "
         "selection (no test, Automatic, All, \"AllTests\", or a property without a test) raises NotImplementedError; unknown tests ValueError; options "
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
        # Iman-Davenport F form of the Friedman statistic, F = (n - 1) Q/(n (k - 1) - Q) ~ F[k - 1, (n - 1)(k - 1)]
        # (reproduces the Wolfram documentation example: 56.0345, p = 2.21282*10^-8)
        d1, d2 = k - 1, (n - 1) * (k - 1)
        if n * (k - 1) - Q <= 0:
            return math.inf, 0.0, L(d1, d2)
        F = (n - 1) * Q / (n * (k - 1) - Q)
        return F, f_sf(F, d1, d2), L(d1, d2)
    raise ValueError(f"unknown test {test!r}")


fn("LocationEquivalenceTest", area=AREA, status="added",
   notes="LocationEquivalenceTest[{data1, data2, ...}, spec] with spec \"test\", {\"test1\", ...} or {prop, test} "
         "for \"KruskalWallis\" "
         "(tie-corrected H; p-value from the F approximation (N - k) H/((k - 1)(N - 1 - H)) ~ FRatioDistribution[k - "
         "1, N - k], which reproduces the Wolfram documentation example), \"KSampleT\" (one-way ANOVA F), "
         "\"CompleteBlockF\" (randomized complete block ANOVA, equal lengths; blocks are the positions) and "
         "\"FriedmanRank\" (tie-corrected Friedman Q reported in the Iman-Davenport form F = (n - 1) Q/(n (k - 1) - Q) "
         "~ FRatioDistribution[k - 1, (n - 1)(k - 1)], which reproduces the Wolfram documentation example); automatic test selection (no test, All, or a property without a test) raises "
         "NotImplementedError; options SignificanceLevel, "
         "VerifyTestAssumptions. Check: reference values from the Wolfram documentation (H = 43.6429, p = "
         "1.17225*10^-14; CompleteBlockF 5.9168, p = 0.00359111; FriedmanRank 56.0345, p = 2.21282*10^-8), "
         "scipy.stats.f_oneway and scipy.stats.friedmanchisquare.")(
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
    den = (n0 - ta[1] / 2) * (n0 - tb[1] / 2)
    if den <= 0:
        raise ValueError("a vector is constant, so the rank correlation is undefined")
    tau = S / math.sqrt(den)
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
            try:
                S, var, tau = _kendall_parts(a, b)
            except ValueError as e:
                raise ValueError(f"{name}: {e}")
            if var <= 0:
                raise ValueError(f"{name}: a vector is constant")
            return test_result(name, tau, p_normal(S / math.sqrt(var), alt), prop, o)
        if kind == "GK":
            if has_ties(a) or has_ties(b):
                raise NotImplementedError(f"{name}: data with ties is not supported")
            try:
                S, var, _ = _kendall_parts(a, b)
            except ValueError as e:
                raise ValueError(f"{name}: {e}")
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
    allowed_all = tuple(allowed) + ("PillaiTrace", "WilksW")
    prop, chosen, several = test_spec(name, args, allowed_all)
    for t in chosen:
        if t not in allowed:
            raise NotImplementedError(f"{name}: the {t} test is not supported here")
    return prop, chosen, several, o


@fn("IndependenceTest", area=AREA, status="added",
    notes="IndependenceTest[v1, v2, spec] with spec \"test\", {\"test1\", ...} or {prop, test} with test one of \"PearsonCorrelation\", "
          "\"SpearmanRank\", \"KendallTau\", \"GoodmanKruskalGamma\", \"BlomqvistBeta\" (as the corresponding *Test "
          "functions) and \"HoeffdingD\" (statistic only); automatic test selection (no test, Automatic, All, or a "
          "property without a test) and the "
          "matrix tests \"PillaiTrace\" and \"WilksW\" raise NotImplementedError. Check: delegated to the individual "
          "tests (identical results, checked in tests).")
def _independence(v1: ListOf[Real], v2: ListOf[Real], *rest: String | ListOf[String] | Option) -> Real | Expr:
    if isinstance(v1, Symbol) or isinstance(v2, Symbol):
        raise Unevaluated()
    prop, chosen, several, o = _select_test("IndependenceTest", rest, tuple(_INDEP))
    extra = [prop] if prop is not None else []
    optr = [rule(Symbol(k), v) for k, v in o.items()]
    res = [_corr_test("IndependenceTest", _INDEP[t])(v1, v2, *extra, *optr) for t in chosen]
    return L(*res) if several else res[0]


@fn("CorrelationTest", area=AREA, status="added",
    notes="CorrelationTest[{{x1, y1}, ...}, 0, spec] with spec \"test\", {\"test1\", ...} or {prop, test}, test "
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
    prop, chosen, several, o = _select_test(name, rest, ("PearsonCorrelation", "SpearmanRank"))
    m = mat(data, name, min_rows=3)
    if len(m[0]) != 2:
        raise ValueError(f"{name}: the data must be bivariate {{{{x1, y1}}, ...}}")
    cols = columns([list(r.args) for r in data.args])
    extra = [prop] if prop is not None else []
    optr = [rule(Symbol(k), v) for k, v in o.items()]
    res = [_corr_test(name, _INDEP[t])(L(*cols[0]), L(*cols[1]), *extra, *optr) for t in chosen]
    return L(*res) if several else res[0]


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
    if is_matrix(x) or (is_list(x) and any(is_list(a) for a in x.args)):
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
        "ShapiroWilk": "ShapiroWilkTest", "JarqueBeraALM": "JarqueBeraALMTest",
        # X02 (wl_statmodels2)
        "PearsonChiSquare": "PearsonChiSquareTest", "MardiaCombined": "MardiaCombinedTest",
        "MardiaKurtosis": "MardiaKurtosisTest", "MardiaSkewness": "MardiaSkewnessTest",
        "BaringhausHenze": "BaringhausHenzeTest"}
_NORMALITY = ("ShapiroWilk", "JarqueBeraALM", "MardiaCombined", "MardiaKurtosis", "MardiaSkewness", "BaringhausHenze")


_DFT_ALL = tuple(_DFT)


@fn("DistributionFitTest", area=AREA, status="added",
    notes="DistributionFitTest[data, dist, spec] with spec \"test\", {\"test1\", ...} or {prop, test} (or {prop, "
          "{tests}}), test one of \"AndersonDarling\", \"CramerVonMises\", \"KolmogorovSmirnov\", \"Kuiper\", "
          "\"WatsonUSquare\", \"PearsonChiSquare\" (fully specified dist) or \"ShapiroWilk\", \"JarqueBeraALM\", "
          "\"MardiaSkewness\", \"MardiaKurtosis\", \"MardiaCombined\", \"BaringhausHenze\" (normality; dist "
          "Automatic or NormalDistribution[]); identical to the corresponding *Test function. Automatic test "
          "selection (DistributionFitTest[data], [data, dist], All, a property without a test) and estimated "
          "distributions raise NotImplementedError; unknown tests ValueError. Check: "
          "delegated to the named test functions (identical results checked in tests).")
def _dft(x: ListOf[Real], d: Expr = None, *rest: String | ListOf[String] | Option) -> Real | Expr:
    name = "DistributionFitTest"
    if isinstance(x, Symbol):
        raise Unevaluated()
    args, o = opts(rest, GOF_OPTS, name)
    prop, chosen, several = test_spec(name, args, _DFT_ALL)
    optr = [rule(Symbol(k), v) for k, v in o.items()]
    extra = [prop] if prop is not None else []
    res = []
    for test in chosen:
        if test not in _DFT:
            raise NotImplementedError(f"{name}: the {test} test is not supported")
        if test in _NORMALITY:
            if d is not None and not (isinstance(d, Symbol) and d.name == "Automatic") and \
                    not (isinstance(d, Expr) and d.head == "NormalDistribution" and not d.args):
                raise NotImplementedError(f"{name}: the {test} test only tests normality (use Automatic)")
            res.append(ev(Expr(_DFT[test], x, *extra, *optr)))
        elif test == "KolmogorovSmirnov":
            if o:
                raise NotImplementedError(f"{name}: options are not supported with the KolmogorovSmirnov test")
            if prop is not None and prop not in ("PValue", "TestStatistic"):
                raise NotImplementedError(f'{name}: the KolmogorovSmirnov test supports "PValue" and "TestStatistic"')
            res.append(ev(Expr("KolmogorovSmirnovTest", x, d, *extra)))
        else:
            res.append(ev(Expr(_DFT[test], x, d if d is not None else Symbol("Automatic"), *extra, *optr)))
    return L(*res) if several else res[0]


# ─────────────────────────── uncertainty: Around, VectorAround, CenteredInterval ───────────────────────────

from .. import kernel as _K                        # noqa: E402
from .. import kernel_lists as _KL                 # noqa: E402
from . import wl_numbertheory2 as _NT              # noqa: E402

_UNH = _K.UNHANDLED
_ELEMENTARY = {"Exp": sp.exp, "Log": sp.log, "Sin": sp.sin, "Cos": sp.cos, "Tan": sp.tan, "ArcTan": sp.atan,
               "ArcSin": sp.asin, "ArcCos": sp.acos, "Sinh": sp.sinh, "Cosh": sp.cosh, "Tanh": sp.tanh,
               "Sqrt": sp.sqrt, "Abs": sp.Abs}


def _sp(x, fname, what):
    v = _num(x, fname, what)
    return v


def _around_parts(a):
    """(value, (dminus, dplus)) of a normalized Around[x, d] / Around[x, {dm, dp}]"""
    x, d = a.args
    if is_list(d):
        return S(x), (S(d.args[0]), S(d.args[1]))
    return S(x), (S(d), S(d))


def _mk_around(x, dm, dp, inexact_=False):
    def o(v):
        return fl(v) if inexact_ else R(v)
    if sp.simplify(dm - dp) == 0:
        return Expr("Around", o(x), o(dp))
    return Expr("Around", o(x), L(o(dm), o(dp)))


@fn("Around", area=AREA, status="added",
    notes="Around[x, d] is x with uncertainty d >= 0; Around[x, {dm, dp}] asymmetric; Around[x, Scaled[r]] = "
          "Around[x, r Abs[x]]; Around[Interval[{a, b}]] = Around[(a + b)/2, (b - a)/2]; Around[{x1, x2, ...}] = "
          "Around[Mean, StandardDeviation] of the samples. Arithmetic (Plus, Times, Power) and the elementary "
          "functions propagate uncertainty to first order assuming independent arguments (Sqrt[Sum[(df/dxi di)^2]], "
          "asymmetric parts propagated separately); a[\"Value\"], a[\"Uncertainty\"] and a[\"Interval\"] give the parts. "
          "Comparisons stay unevaluated; negative uncertainties raise ValueError; non-numeric values TypeError; "
          "symbolic arguments stay unevaluated; Around of complex numbers raises NotImplementedError. Check: "
          "first-order propagation formulas checked against hand derivatives in tests.")
def _around(x: Real | ListOf[Real] | Expr, d: Real | ListOf[Real] | Expr = None) -> Expr:
    name = "Around"
    if d is None:
        if isinstance(x, Expr) and x.head == "Interval" and len(x.args) == 1 and is_list(x.args[0]) and \
                len(x.args[0].args) == 2:
            a, b = (_sp(t, name, "endpoint") for t in x.args[0].args)
            if b < a:
                raise ValueError(f"{name}: the interval endpoints are out of order")
            return _mk_around((a + b) / 2, (b - a) / 2, (b - a) / 2, has_float(x))
        if is_list(x):
            v, items = vec(x, name, min_len=2)
            m = mean(v)
            s = sp.sqrt(svar(v))
            return _mk_around(m, s, s, inexact(items))
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{name}: expected Around[x, d], Around[{{x1, ...}}] or Around[Interval[{{a, b}}]]")
    xv = _num(x, name, "value", real=False)
    if xv.is_real is False:
        raise NotImplementedError(f"{name}: complex values are not supported")
    if isinstance(d, Expr) and d.head == "Scaled" and len(d.args) == 1:
        r = _sp(d.args[0], name, "relative uncertainty")
        if r < 0:
            raise ValueError(f"{name}: the uncertainty must be non-negative")
        u = r * abs(xv)
        return _mk_around(xv, u, u, has_float(x) or has_float(d))
    if is_list(d):
        if len(d.args) != 2:
            raise ValueError(f"{name}: an asymmetric uncertainty is {{dminus, dplus}}")
        dm, dp = (_sp(t, name, "uncertainty") for t in d.args)
        if dm < 0 or dp < 0:
            raise ValueError(f"{name}: the uncertainty must be non-negative")
        if dm == dp:
            return Expr("Around", x, d.args[0])
        raise Unevaluated()
    dv = _sp(d, name, "uncertainty")
    if dv < 0:
        raise ValueError(f"{name}: the uncertainty must be non-negative")
    raise Unevaluated()


def _is_around(a):
    return isinstance(a, Expr) and a.head == "Around" and len(a.args) == 2 and \
        all(isinstance(S(t), sp.Basic) and S(t).is_number for t in
            ([a.args[0]] + (list(a.args[1].args) if is_list(a.args[1]) else [a.args[1]])))


def _propagate(f, vals, deltas, inexact_):
    """first-order propagation of the sympy expression f in symbols z_i with values vals and (dm, dp) deltas"""
    zs = sp.symbols(f"w18z0:{len(vals)}")
    expr = f(*zs)
    sub = dict(zip(zs, vals))
    val = expr.subs(sub)
    lo2, hi2 = sp.Integer(0), sp.Integer(0)
    for z, (dm, dp) in zip(zs, deltas):
        g = sp.diff(expr, z).subs(sub)
        if g.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ArithmeticError("Around: the derivative is undefined at the value (uncertainty not propagable)")
        # a positive derivative maps the upper deviation to the upper side
        if (g >= 0) == True:  # noqa: E712
            hi2 += (g * dp) ** 2
            lo2 += (g * dm) ** 2
        else:
            hi2 += (g * dm) ** 2
            lo2 += (g * dp) ** 2
    if inexact_:
        val, lo, hi = sp.N(val, 17), sp.N(sp.sqrt(lo2), 17), sp.N(sp.sqrt(hi2), 17)
    else:
        val, lo, hi = sp.simplify(val), sp.simplify(sp.sqrt(lo2)), sp.simplify(sp.sqrt(hi2))
    return _mk_around(val, lo, hi, inexact_)


def _around_arith(expr, ev_):
    h = expr.head
    args = list(expr.args)
    if not any(_is_around(a) for a in args):
        return _UNH
    vals, deltas, slots, inexact_ = [], [], [], False
    for a in args:
        if _is_around(a):
            v, d = _around_parts(a)
            vals.append(v)
            deltas.append(d)
            slots.append(None)
            inexact_ = inexact_ or has_float(a)
        else:
            n = _NT.number_of(a)
            if n is None:
                return _UNH
            slots.append(sp.sympify(n))
            inexact_ = inexact_ or isinstance(n, float)

    def build(*zs):
        it = iter(zs)
        xs = [next(it) if s is None else s for s in slots]
        if h == "Plus":
            return sp.Add(*xs)
        if h == "Times":
            return sp.Mul(*xs)
        if h == "Power":
            return xs[0] ** xs[1]
        return _ELEMENTARY[h](*xs)
    if h in ("Plus", "Times", "Power") or (h in _ELEMENTARY and len(args) == 1):
        return _propagate(build, vals, deltas, inexact_)
    return _UNH


_NT.SPECIAL_HEADS.add("Around")
_NT.ARITH_HANDLERS.append(_around_arith)
_NT._ARITH_HEADS.update(_ELEMENTARY)


def _around_prop(head, args, ev_):
    if not _is_around(head) or len(args) != 1 or not isinstance(args[0], str):
        return _UNH
    p = args[0]
    v, (dm, dp) = _around_parts(head)
    if p == "Value":
        return head.args[0]
    if p == "Uncertainty":
        return head.args[1]
    if p == "Interval":
        return ev_(Expr("Interval", L(Expr("Plus", head.args[0], Expr("Times", -1, R(dm) if not has_float(head)
                                                                         else fl(dm))),
                                      Expr("Plus", head.args[0], R(dp) if not has_float(head) else fl(dp)))))
    raise ValueError(f'Around: unknown property "{p}" (use "Value", "Uncertainty" or "Interval")')


_KL.COMPOUND_HEAD_HOOKS["Around"] = _around_prop


@fn("AroundReplace", area=AREA, status="added",
    notes="AroundReplace[expr, {x1 -> Around[v1, d1], ...}] evaluates expr at the values and propagates the "
          "(symmetric) uncertainties to first order treating the xi as independent: Around[f(v), Sqrt[Sum[(df/dxi "
          "di)^2]]]; rules whose right side is a plain number substitute it exactly; a single rule need not be in a "
          "list. Asymmetric uncertainties and the Assumptions option raise NotImplementedError; a non-rule "
          "TypeError. Check: identical to Around arithmetic for expressions with each variable once (tested).")
def _around_replace(e: Expr, rules: Expr) -> Expr:
    name = "AroundReplace"
    rl = list(rules.args) if is_list(rules) else [rules]
    if not all(_is_rule(r) for r in rl):
        raise TypeError(f"{name}: the second argument must be a rule or a list of rules")
    from ..evaluate import evaluate
    syms, vals, ds, inexact_ = [], [], [], False
    plain = []
    for r in rl:
        lhs, rhs = r.args
        rhs = evaluate(rhs)
        if _is_around(rhs):
            if is_list(rhs.args[1]):
                raise NotImplementedError(f"{name}: asymmetric uncertainties are not supported")
            syms.append(lhs)
            v, (d, _) = _around_parts(rhs)
            vals.append(v)
            ds.append(d)
            inexact_ = inexact_ or has_float(rhs)
        else:
            plain.append(r)
    e2 = evaluate(Expr("ReplaceAll", e, L(*plain))) if plain else e
    if not syms:
        return evaluate(e2)
    zs = sp.symbols(f"w18z0:{len(syms)}")
    sub_rules = L(*(rule(s, Symbol(str(z))) for s, z in zip(syms, zs)))
    fe = S(evaluate(Expr("ReplaceAll", e2, sub_rules)))
    if not isinstance(fe, sp.Basic):
        raise TypeError(f"{name}: the expression is not numeric")
    fe = fe.subs({sp.Symbol(str(z)): z for z in zs})
    if fe.free_symbols - set(zs):
        raise Unevaluated()
    return _propagate(lambda *z: fe.subs(dict(zip(zs, z))), vals, [(d, d) for d in ds], inexact_ or
                      has_float(e2))


@fn("MeanAround", area=AREA, status="added",
    notes="MeanAround[{x1, x2, ...}] = Around[Mean, StandardDeviation/Sqrt[n]] (the mean with its standard error); "
          "for a list of Around values the inverse-variance weighted mean Around[Sum[w_i x_i]/Sum[w_i], "
          "1/Sqrt[Sum[w_i]]], w_i = 1/d_i^2 (zero uncertainties raise ArithmeticError). Matrices (VectorAround "
          "output), asymmetric uncertainties and Weights raise NotImplementedError; fewer than 2 plain values "
          "ValueError. Check: reference values from numpy mean/std(ddof=1)/sqrt(n) and the weighted-mean formula.")
def _mean_around(x: ListOf[Real] | ListOf[Expr]) -> Expr:
    name = "MeanAround"
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x):
        raise TypeError(f"{name}: the argument must be a list")
    if is_matrix(x):
        raise NotImplementedError(f"{name}: matrices (VectorAround results) are not supported")
    if x.args and all(_is_around(a) for a in x.args):
        num, den, inexact_ = sp.Integer(0), sp.Integer(0), False
        for a in x.args:
            if is_list(a.args[1]):
                raise NotImplementedError(f"{name}: asymmetric uncertainties are not supported")
            v, (d, _) = _around_parts(a)
            if d == 0:
                raise ArithmeticError(f"{name}: an uncertainty is zero (infinite weight)")
            num += v / d ** 2
            den += 1 / d ** 2
            inexact_ = inexact_ or has_float(a)
        m, u = num / den, 1 / sp.sqrt(den)
        if not inexact_:
            m, u = sp.simplify(m), sp.simplify(u)
        return _mk_around(m, u, u, inexact_)
    v, items = vec(x, name, min_len=2)
    m = mean(v)
    u = sp.sqrt(svar(v) / len(v))
    return _mk_around(m, u, u, inexact(items))


@fn("VectorAround", area=AREA, status="added",
    notes="VectorAround[{x1, ..., xn}, {d1, ..., dn}] and VectorAround[{x1, ...}, cov] (symmetric positive "
          "semidefinite covariance matrix) represent a vector with correlated uncertainties; the arguments are "
          "validated and the object stays as entered; v[\"Value\"], v[\"Uncertainty\"] (Sqrt of the covariance "
          "diagonal) and v[\"Covariance\"] give its parts. Arithmetic on VectorAround and the single-argument "
          "sample form raise NotImplementedError; mismatched lengths or an invalid covariance ValueError. Check: "
          "the covariance of the d form is DiagonalMatrix[d^2] (tested).")
def _vector_around(x: ListOf[Real], d: ListOf[Real] | MatrixOf[Real] = None) -> Expr:
    name = "VectorAround"
    if d is None:
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise NotImplementedError(f"{name}: the sample form VectorAround[{{v1, v2, ...}}] is not supported")
    if isinstance(x, Symbol) or isinstance(d, Symbol):
        raise Unevaluated()
    v, _ = vec(x, name, "value")
    if is_matrix(d):
        m = mat(d, name, "covariance")
        if len(m) != len(v) or len(m[0]) != len(v):
            raise ValueError(f"{name}: the covariance matrix must be {len(v)} x {len(v)}")
        M = sp.Matrix(m)
        if not all(sp.simplify(M[i, j] - M[j, i]) == 0 for i in range(len(v)) for j in range(len(v))):
            raise ValueError(f"{name}: the covariance matrix must be symmetric")
        if any(fl(e) < -1e-12 for e in M.evalf().eigenvals()):
            raise ValueError(f"{name}: the covariance matrix must be positive semidefinite")
    else:
        dd, _ = vec(d, name, "uncertainties")
        if len(dd) != len(v):
            raise ValueError(f"{name}: the values and uncertainties must have the same length")
        if any(t < 0 for t in dd):
            raise ValueError(f"{name}: uncertainties must be non-negative")
    raise Unevaluated()


def _vector_around_prop(head, args, ev_):
    if len(head.args) != 2 or len(args) != 1 or not isinstance(args[0], str) or not is_list(head.args[0]):
        return _UNH
    p = args[0]
    x, d = head.args
    if p == "Value":
        return x
    if is_matrix(d):
        cov = d
        unc = ev_(Expr("Sqrt", Expr("Diagonal", d)))
    else:
        cov = ev_(Expr("DiagonalMatrix", Expr("Power", d, 2)))
        unc = d
    if p == "Covariance":
        return cov
    if p == "Uncertainty":
        return unc
    raise ValueError(f'VectorAround: unknown property "{p}" (use "Value", "Uncertainty" or "Covariance")')


_KL.COMPOUND_HEAD_HOOKS["VectorAround"] = _vector_around_prop


def _is_ci(a):
    if not (isinstance(a, Expr) and a.head == "CenteredInterval" and len(a.args) == 2):
        return False
    for t in a.args:
        if has_float(t):
            return False
        v = S(t)
        if not (isinstance(v, sp.Basic) and v.is_number and v.is_real):
            return False
    return True


@fn("CenteredInterval", area=AREA, status="added",
    notes="CenteredInterval[c, r] is the set of numbers within r >= 0 of c (exact real c and r); "
          "CenteredInterval[Interval[{a, b}]] = CenteredInterval[(a + b)/2, (b - a)/2]. Exact interval arithmetic: "
          "sums add centres and radii, products give CenteredInterval[c1 c2, |c1| r2 + |c2| r1 + r1 r2] (a "
          "guaranteed enclosure), integer powers repeat the product; ci[\"Center\"], ci[\"Radius\"] and "
          "ci[\"Bounds\"] give the parts. Machine-number (binary ball) inputs, complex centres and other functions "
          "raise NotImplementedError; r < 0 ValueError. Check: every product enclosure contains the exact product "
          "of the endpoints (tested).")
def _centered_interval(c: Real | Expr, r: Real = None) -> Expr:
    name = "CenteredInterval"
    if r is None:
        if isinstance(c, Expr) and c.head == "Interval" and len(c.args) == 1 and is_list(c.args[0]) and \
                len(c.args[0].args) == 2:
            if has_float(c):
                raise NotImplementedError(f"{name}: machine-number intervals are not supported")
            a, b = (_sp(t, name, "endpoint") for t in c.args[0].args)
            if b < a:
                raise ValueError(f"{name}: the interval endpoints are out of order")
            return Expr("CenteredInterval", R((a + b) / 2), R((b - a) / 2))
        if isinstance(c, Symbol):
            raise Unevaluated()
        raise TypeError(f"{name}: expected CenteredInterval[c, r] or CenteredInterval[Interval[{{a, b}}]]")
    if isinstance(c, Symbol) or isinstance(r, Symbol):
        raise Unevaluated()
    if has_float(c) or has_float(r):
        raise NotImplementedError(f"{name}: machine-number (binary ball) intervals are not supported; use exact "
                                  "numbers")
    cv = _num(c, name, "centre", real=False)
    if cv.is_real is False:
        raise NotImplementedError(f"{name}: complex centres are not supported")
    rv = _sp(r, name, "radius")
    if rv < 0:
        raise ValueError(f"{name}: the radius must be non-negative")
    raise Unevaluated()


def _ci_parts(a):
    return S(a.args[0]), S(a.args[1])


def _ci_arith(expr, ev_):
    h = expr.head
    if h not in ("Plus", "Times", "Power") or not any(_is_ci(a) for a in expr.args):
        return _UNH
    parts = []
    for a in expr.args:
        if _is_ci(a):
            parts.append(_ci_parts(a))
        else:
            n = _NT.number_of(a)
            if n is None or isinstance(n, (float, complex)) or (isinstance(n, sp.Basic) and n.is_real is False):
                return _UNH
            parts.append((sp.sympify(n), sp.Integer(0)))
    if h == "Plus":
        c = sp.Add(*(p[0] for p in parts))
        r = sp.Add(*(p[1] for p in parts))
    elif h == "Times":
        c, r = parts[0]
        for c2, r2 in parts[1:]:
            c, r = c * c2, abs(c) * r2 + abs(c2) * r + r * r2
    else:
        (c0, r0), (e, er) = parts
        if er != 0 or not e.is_integer or e < 1:
            raise NotImplementedError("CenteredInterval: only positive integer powers are supported")
        c, r = c0, r0
        for _ in range(int(e) - 1):
            c, r = c * c0, abs(c) * r0 + abs(c0) * r + r * r0
    return Expr("CenteredInterval", R(sp.simplify(c)), R(sp.simplify(r)))


_NT.SPECIAL_HEADS.add("CenteredInterval")
_NT.ARITH_HANDLERS.append(_ci_arith)


def _ci_prop(head, args, ev_):
    if not _is_ci(head) or len(args) != 1 or not isinstance(args[0], str):
        return _UNH
    c, r = _ci_parts(head)
    p = args[0]
    if p == "Center":
        return head.args[0]
    if p == "Radius":
        return head.args[1]
    if p == "Bounds":
        return L(R(c - r), R(c + r))
    raise ValueError(f'CenteredInterval: unknown property "{p}" (use "Center", "Radius" or "Bounds")')


_KL.COMPOUND_HEAD_HOOKS["CenteredInterval"] = _ci_prop


# ─────────────────────────── likelihoods, moments and distribution utilities ───────────────────────────

def _dist_data(d, x, fname):
    dd = W.dist_of(d, fname)
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x):
        raise TypeError(f"{fname}: the observations must be a list")
    if any(is_list(a) for a in x.args):
        raise NotImplementedError(f"{fname}: multivariate observations are not supported")
    return dd


def _pdf_terms(d, x, fname):
    dd = _dist_data(d, x, fname)
    out_ = []
    for a in x.args:
        out_.append(S(W.pdf_value(dd, S(a))))
    return dd, out_


@fn("Likelihood", area=AREA, status="added",
    notes="Likelihood[dist, {x1, x2, ...}] = Product[PDF[dist, xi]] for univariate distributions (exact for exact "
          "input; symbolic parameters and observations give the symbolic product). Random processes and "
          "multivariate observations raise NotImplementedError; a non-list TypeError; a non-distribution "
          "ValueError. Check: reference values from scipy.stats (norm.pdf / poisson.pmf products) and the closed "
          "form.")
def _likelihood(d: Expr, x: ListOf[Real] | ListOf[Expr]) -> Real | Expr:
    _, terms = _pdf_terms(d, x, "Likelihood")
    v = sp.Mul(*terms)
    return out(sp.simplify(v) if not has_float(x) and not has_float(d) else v, [d, x])


@fn("LogLikelihood", area=AREA, status="added",
    notes="LogLikelihood[dist, {x1, x2, ...}] = Sum[Log[PDF[dist, xi]]] for univariate distributions (exact for "
          "exact input, logarithms of the positive density factors split; an observation outside the support gives -Infinity). Random "
          "processes and multivariate observations raise NotImplementedError; a non-list TypeError. Check: "
          "reference values from scipy.stats logpdf sums and Log[Likelihood] (tested).")
def _loglikelihood(d: Expr, x: ListOf[Real] | ListOf[Expr]) -> Real | Expr:
    _, terms = _pdf_terms(d, x, "LogLikelihood")
    if any(t == 0 for t in terms):
        return R(-sp.oo)
    inexact_ = has_float(x) or has_float(d)
    v = sp.Add(*(sp.log(t) for t in terms))
    if not inexact_:
        # the densities are positive here, so splitting the logarithms is valid
        v = sp.simplify(sp.expand_log(v, force=True))
    return out(v, [d, x])


def _falling(x, r):
    return sp.Mul(*(x - k for k in range(r)))


@fn("FactorialMoment", area=AREA, status="added",
    notes="FactorialMoment[data, r] = Mean[FactorialPower[x, r]] (column-wise for matrices) and "
          "FactorialMoment[dist, r] = Sum[StirlingS1[r, k] Moment[dist, k], {k, r}] for univariate distributions "
          "(symbolic parameters allowed); FactorialMoment[r] stays as the formal moment. Multivariate orders "
          "{r1, ...}, WeightedData and multivariate distributions raise NotImplementedError; r not a non-negative "
          "integer ValueError. Check: reference values from the Wolfram documentation (FactorialMoment[{1, 2, 3, "
          "4}, 1] = 5/2, FactorialMoment[{1., 2., 3., 4.}, 2] = 5., FactorialMoment[BinomialDistribution[n, p], 2] "
          "= n (n - 1) p^2).")
def _factorial_moment(x: ListOf[Real] | MatrixOf[Real] | Expr | Integer, r: Integer = None) -> Real | Expr:
    name = "FactorialMoment"
    if r is None:
        if isinstance(x, int) and not isinstance(x, bool) and x >= 0:
            raise Unevaluated()
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError(f"{name}: the formal moment order must be a non-negative integer")
    if is_list(r):
        raise NotImplementedError(f"{name}: multivariate orders are not supported")
    if isinstance(r, Symbol):
        raise Unevaluated()
    if not (isinstance(r, int) and not isinstance(r, bool)) or r < 0:
        raise ValueError(f"{name}: the order must be a non-negative integer")
    if isinstance(x, Expr) and x.head == "WeightedData":
        raise NotImplementedError(f"{name}: WeightedData is not supported")
    if W.is_dist(x):
        dd = W.dist_of(x, name)
        k = sp.Symbol("w18k")
        v = sp.Add(*(sp.functions.combinatorial.numbers.stirling(r, j, kind=1, signed=True) *
                     W.raw_moment(dd, j) for j in range(0, r + 1)))
        return out(sp.factor(sp.simplify(v)) if not has_float(x) else v, [x])
    if is_matrix(x):
        return per_column(x, name, lambda c: _factorial_moment(c, r))
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x) or not x.args:
        raise TypeError(f"{name}: the first argument must be a non-empty list or a distribution")
    v = [S(a) for a in x.args]
    m = sp.Add(*(_falling(t, r) for t in v)) / len(v)
    return out(m if inexact(x.args) else (sp.factor(m) if any(getattr(t, "free_symbols", None) for t in v)
                                           else sp.simplify(m)), list(x.args))


@fn("PowerSymmetricPolynomial", area=AREA, status="added",
    notes="PowerSymmetricPolynomial[r, data] = Sum[x^r] over the data (column-wise for matrices; r any number); "
          "PowerSymmetricPolynomial[r] stays as the formal power sum and PowerSymmetricPolynomial[{r1, ...}, data] "
          "is the list of the PowerSymmetricPolynomial[ri, data]. A non-list data TypeError. Check: the identity "
          "PowerSymmetricPolynomial[1, data] = Total[data] (tested) and direct sums.")
def _power_sym(r: Integer | ListOf[Integer] | Real, x: ListOf[Real] | MatrixOf[Real] = None) -> Real | Expr:
    name = "PowerSymmetricPolynomial"
    if x is None:
        raise Unevaluated()
    if is_list(r):
        return L(*(_power_sym(q, x) for q in r.args))
    if isinstance(x, Symbol):
        raise Unevaluated()
    if is_matrix(x):
        return per_column(x, name, lambda c: _power_sym(r, c))
    if not is_list(x):
        raise TypeError(f"{name}: the data must be a list")
    rr = S(r)
    v = sp.Add(*(S(a) ** rr for a in x.args))
    return out(v, list(x.args) + [r])


def _set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in _set_partitions(rest):
        yield [[first]] + p
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]


@fn("AugmentedSymmetricPolynomial", area=AREA, status="added",
    notes="AugmentedSymmetricPolynomial[{r1, ..., rk}, data] = Sum[x_i1^r1 ... x_ik^rk] over distinct indices "
          "i1, ..., ik, computed from power sums by Moebius inversion over the set partitions of {1, ..., k} "
          "(column-wise for matrices); AugmentedSymmetricPolynomial[{r1, ...}] stays as the formal object. Orders "
          "must be positive integers (ValueError); a non-list TypeError. Check: equals the brute-force sum over "
          "distinct index tuples (tested) and AugmentedSymmetricPolynomial[{r}, data] = "
          "PowerSymmetricPolynomial[r, data].")
def _aug_sym(rs: ListOf[Integer], x: ListOf[Real] | MatrixOf[Real] = None) -> Real | Expr:
    name = "AugmentedSymmetricPolynomial"
    if isinstance(rs, Symbol):
        raise Unevaluated()
    if not is_list(rs) or not rs.args:
        raise TypeError(f"{name}: the orders must be a list {{r1, r2, ...}}")
    if not all(isinstance(q, int) and not isinstance(q, bool) and q > 0 for q in rs.args):
        if any(isinstance(q, Symbol) for q in rs.args):
            raise Unevaluated()
        raise ValueError(f"{name}: the orders must be positive integers")
    if x is None:
        raise Unevaluated()
    if isinstance(x, Symbol):
        raise Unevaluated()
    if is_matrix(x):
        return per_column(x, name, lambda c: _aug_sym(rs, c))
    if not is_list(x):
        raise TypeError(f"{name}: the data must be a list")
    v = [S(a) for a in x.args]
    r = list(rs.args)
    tot = sp.Integer(0)
    for part in _set_partitions(list(range(len(r)))):
        mu = sp.Integer(1)
        prod = sp.Integer(1)
        for blk in part:
            mu *= (-1) ** (len(blk) - 1) * sp.factorial(len(blk) - 1)
            e = sum(r[i] for i in blk)
            prod *= sp.Add(*(t ** e for t in v))
        tot += mu * prod
    res = sp.expand(tot)
    return out(res, list(x.args))


_ASSUME = {
    "NormalDistribution": (2, "Element[#1, Reals] && #2 > 0"),
    "LogNormalDistribution": (2, "Element[#1, Reals] && #2 > 0"),
    "CauchyDistribution": (2, "Element[#1, Reals] && #2 > 0"),
    "LaplaceDistribution": (2, "Element[#1, Reals] && #2 > 0"),
    "LogisticDistribution": (2, "Element[#1, Reals] && #2 > 0"),
    "GammaDistribution": (2, "#1 > 0 && #2 > 0"),
    "BetaDistribution": (2, "#1 > 0 && #2 > 0"),
    "WeibullDistribution": (2, "#1 > 0 && #2 > 0"),
    "ParetoDistribution": (2, "#1 > 0 && #2 > 0"),
    "FRatioDistribution": (2, "#1 > 0 && #2 > 0"),
    "ExponentialDistribution": (1, "#1 > 0"),
    "PoissonDistribution": (1, "#1 > 0"),
    "ChiSquareDistribution": (1, "#1 > 0"),
    "StudentTDistribution": (1, "#1 > 0"),
    "GeometricDistribution": (1, "0 < #1 <= 1"),
    "BernoulliDistribution": (1, "0 <= #1 <= 1"),
    "BinormalDistribution": (1, "-1 < #1 < 1"),
}


@fn("DistributionParameterAssumptions", area=AREA, status="added",
    notes="DistributionParameterAssumptions[dist] gives the parameter conditions (Element[mu, Reals] && sigma > 0 "
          "for NormalDistribution[mu, sigma], 0 < p <= 1 for GeometricDistribution[p], n in Integers && n > 0 && "
          "pi >= 0 && Sum[pi] == 1 for MultinomialDistribution, ...), with conditions on numeric parameters "
          "evaluated (True when all hold, False when one fails). Covered: Normal, LogNormal, Cauchy, Laplace, "
          "Logistic, Gamma, Beta, Weibull, Pareto, FRatio (two parameters), Exponential, Poisson, ChiSquare, "
          "StudentT[nu], Geometric, Bernoulli, Binormal[rho] and Multinomial; other distributions raise "
          "NotImplementedError; non-distributions TypeError. Check: reference values from the Wolfram documentation "
          "(Normal, Gamma, Geometric, Pareto, Binormal, Multinomial examples).")
def _dpa(d: Expr) -> Expr:
    name = "DistributionParameterAssumptions"
    if isinstance(d, Symbol):
        raise Unevaluated()
    if not (isinstance(d, Expr) and isinstance(d.head, str) and d.head.endswith("Distribution")):
        raise TypeError(f"{name}: {d!r} is not a distribution")
    from ..parser import parse
    if d.head == "MultinomialDistribution" and len(d.args) == 2 and is_list(d.args[1]):
        n, ps = d.args
        parts = [Expr("Element", n, Symbol("Integers")), Expr("Greater", n, 0)]
        parts += [Expr("GreaterEqual", p, 0) for p in ps.args]
        parts.append(Expr("Equal", Expr("Plus", *ps.args), 1))
        return ev(Expr("And", *parts))
    spec = _ASSUME.get(d.head)
    if spec is None or len(d.args) != spec[0]:
        raise NotImplementedError(f"{name}: {d.head} with {len(d.args)} parameter(s) is not supported")
    tmpl = parse(spec[1])
    sub = L(*(rule(Expr("Slot", i + 1), a) for i, a in enumerate(d.args)))
    return ev(ev(Expr("ReplaceAll", tmpl, sub)))


@fn("DistributionParameterQ", area=AREA, status="added",
    notes="DistributionParameterQ[dist] is True when the numeric parameters of dist satisfy its parameter "
          "assumptions (symbolic parameters are assumed valid) and False otherwise; non-distribution arguments "
          "give False; distributions the library does not implement raise NotImplementedError. Check: agrees with "
          "DistributionParameterAssumptions on numeric parameters (tested).")
def _dpq(d: Expr) -> Boolean:
    if not (isinstance(d, Expr) and isinstance(d.head, str) and d.head.endswith("Distribution")):
        return False
    try:
        W.dist_of(d, "DistributionParameterQ")
    except ValueError:
        return False
    except Unevaluated:
        return True
    return True


_SYMMETRIC = {"NormalDistribution": 0, "StudentTDistribution": None, "CauchyDistribution": 0,
              "LaplaceDistribution": 0, "LogisticDistribution": 0}


@fn("RarerProbability", area=AREA, status="added",
    notes="RarerProbability[dist, x] = Probability[PDF[dist, y] <= PDF[dist, x], y ~ dist] (and the list form "
          "RarerProbability[dist, {x1, ...}]): for the symmetric unimodal Normal, StudentT, Cauchy, Laplace and "
          "Logistic distributions 2 CDF[dist, c - |x - c|] (the two-tailed p-value); for discrete distributions "
          "with finite support (Bernoulli, Binomial, DiscreteUniform) the exact sum of the probabilities not "
          "exceeding PDF[dist, x]. Other distributions and multivariate ones raise NotImplementedError; symbolic "
          "parameters or points stay unevaluated. Check: reference values from scipy.stats (2 norm.cdf(-|z|)) and "
          "direct summation of binom.pmf.")
def _rarer(d: Expr, x: Real | ListOf[Real]) -> Real | ListOf[Real]:
    name = "RarerProbability"
    if is_list(x):
        return L(*(_rarer(d, a) for a in x.args))
    dd = W.dist_of(d, name)
    xv = _num(x, name, "point")
    if not dd.numeric():
        raise Unevaluated()
    h = d.head
    if h in _SYMMETRIC:
        c = dd.a[0] if (h != "StudentTDistribution" or len(dd.a) == 3) else sp.Integer(0)
        if h == "StudentTDistribution" and len(dd.a) == 3:
            c = dd.a[0]
        F = W.cdf_value(dd, c - abs(xv - c))
        return out(sp.simplify(2 * F), [d, x])
    if h in ("BernoulliDistribution", "BinomialDistribution", "DiscreteUniformDistribution"):
        lo, hi = dd.bounds()
        px = W.pdf_value(dd, xv)
        tot = sp.Integer(0)
        for k in range(int(lo), int(hi) + 1):
            pk = W.pdf_value(dd, sp.Integer(k))
            if (sp.N(pk - px, 30) <= sp.Float(1e-25)) if (has_float(d) or has_float(x)) else (pk <= px):
                tot += pk
        return out(sp.simplify(tot), [d, x])
    raise NotImplementedError(f"{name}: {h} is not supported")


# ─────────────────────────── conditioning, reliability importance, data utilities ───────────────────────────

@fn("Conditioned", area=AREA, status="added",
    notes="Conditioned[expr, cond] (expr \\[Conditioned] cond) is inert; Probability[Conditioned[a, c], x ~ dist] = "
          "Probability[a && c]/Probability[c] and Expectation[Conditioned[f, c], x ~ dist] = Expectation[f "
          "Boole[c]]/Probability[c] (univariate; the expectation integrates f PDF over the event, continuous "
          "distributions with numeric parameters only, discrete ones raise NotImplementedError); a "
          "zero-probability condition raises ArithmeticError; a wrong number of arguments ValueError. Check: "
          "conditional probabilities of exponential memorylessness (tested) and Bayes' rule.")
def _conditioned(*args: Expr) -> Expr:
    if len(args) != 2:
        raise ValueError(f"Conditioned: expected Conditioned[expr, cond], not {len(args)} argument(s)")
    raise Unevaluated()


def _is_conditioned_first(args):
    return len(args) == 2 and isinstance(args[0], Expr) and args[0].head == "Conditioned" and \
        len(args[0].args) == 2


def _cond_prob(*args: Expr) -> Expr:
    (a, c), spec = args[0].args, args[1]
    pc = ev(Expr("Probability", c, spec))
    if _NT.number_of(pc) is not None and sp.sympify(_NT.number_of(pc)) == 0:
        raise ArithmeticError("Probability: the condition has probability zero")
    if isinstance(pc, Expr) and pc.head == "Probability":
        raise Unevaluated()
    pac = ev(Expr("Probability", Expr("And", a, c), spec))
    if isinstance(pac, Expr) and pac.head == "Probability":
        raise Unevaluated()
    return ev(Expr("Simplify", Expr("Times", pac, Expr("Power", pc, -1))))


def _cond_expect(*args: Expr) -> Expr:
    (f, c), spec = args[0].args, args[1]
    pairs = W._distributed(spec, "Expectation")
    if len(pairs) != 1:
        raise NotImplementedError("Expectation: conditioned expectations need one random variable")
    (v, d), = pairs
    dd = W.dist_of(d, "Expectation")
    if not dd.numeric():
        raise Unevaluated()
    if dd.discrete:
        raise NotImplementedError("Expectation: conditioned expectations of discrete distributions are not supported")
    st = W._event_set(W._pred(c, "Expectation"), v, "Expectation")
    pc = sp.simplify(W._prob_set(dd, st))
    if pc == 0:
        raise ArithmeticError("Expectation: the condition has probability zero")
    fv = W.sym(f, "Expectation")
    x = sp.Dummy("x", real=True)
    dens = W.pdf_value(dd, x)

    def integ(s_):
        if isinstance(s_, sp.Union):
            return sp.Add(*(integ(t) for t in s_.args))
        if isinstance(s_, sp.Interval):
            return sp.integrate(fv.subs(v, x) * dens, (x, s_.start, s_.end))
        if s_ is sp.S.EmptySet or isinstance(s_, sp.FiniteSet):
            return sp.Integer(0)
        raise NotImplementedError(f"Expectation: unsupported conditioning event {s_}")
    num = integ(st)
    if num.has(sp.Integral):
        raise Unevaluated()
    return out(sp.simplify(num / pc), [f, d])


def _formal_order(args):
    return len(args) == 1 and ((isinstance(args[0], int) and not isinstance(args[0], bool) and args[0] >= 0) or
                               (is_list(args[0]) and all(isinstance(q, int) and q >= 0 for q in args[0].args)) or
                               isinstance(args[0], Symbol))


def _formal_moment(*args: Expr) -> Expr:
    raise Unevaluated()


for _mn in ("Moment", "CentralMoment", "Cumulant"):
    W.extend(_mn, _formal_moment, _formal_order,
             f"{_mn}[r] (and {_mn}[{{r1, ...}}]) stays as the formal moment for MomentConvert / MomentEvaluate (W18)",
             status="added")


W.extend("Probability", _cond_prob, _is_conditioned_first,
         "Probability[Conditioned[a, c], x \\[Distributed] dist] = Probability[a && c]/Probability[c] (W18)",
         status="added")
W.extend("Expectation", _cond_expect, _is_conditioned_first,
         "Expectation[Conditioned[f, c], x \\[Distributed] dist] = Expectation[f Boole[c]]/Probability[c] (W18)",
         status="added")


@fn("BirnbaumImportance", area=AREA, status="added",
    notes="BirnbaumImportance[rdist, t] for ReliabilityDistribution and FailureDistribution: the list (in component "
          "order) of R(t | component i works) - R(t | component i failed), by exact enumeration of the Boolean "
          "structure function with independent component survival functions; t may be symbolic. Other "
          "distributions raise TypeError; discrete components (rejected by the distribution) NotImplementedError. "
          "Check: for a series system x && y the importance of x is the survival of y (tested), parallel x || y "
          "gives 1 - F_y.")
def _birnbaum(d: Expr, t: Real | Expr) -> ListOf[Expr]:
    name = "BirnbaumImportance"
    if isinstance(d, Symbol):
        raise Unevaluated()
    if not (isinstance(d, Expr) and d.head in ("ReliabilityDistribution", "FailureDistribution")):
        raise TypeError(f"{name}: expected a ReliabilityDistribution or FailureDistribution")
    dd = W.dist_of(d, name)
    tv = S(t)
    if isinstance(tv, sp.Basic) and tv.is_number and tv < 0:
        raise ValueError(f"{name}: the time must be non-negative")
    from . import wl_distributions2 as W2
    Sv = [sp.simplify(1 - W2._cdf_of(c, tv)) for c in dd.ds]
    n = len(dd.ds)

    def works(state):
        if dd.failure:
            return not dd.f({nm: not w for nm, w in zip(dd.names, state)})
        return dd.f(dict(zip(dd.names, state)))

    res = []
    for i in range(n):
        tot = sp.Integer(0)
        for bits in itertools.product((True, False), repeat=n - 1):
            st1 = list(bits[:i]) + [True] + list(bits[i:])
            st0 = list(bits[:i]) + [False] + list(bits[i:])
            w = sp.Mul(*(Sv[j] if st1[j] else 1 - Sv[j] for j in range(n) if j != i))
            tot += w * (int(works(st1)) - int(works(st0)))
        res.append(out(sp.simplify(tot), [d, t]))
    return L(*res)


@fn("MeanShift", area=AREA, status="added",
    notes="MeanShift[list, d] replaces each element by the mean of the elements within distance d of it "
          "(|xi - xj| <= d, Euclidean distance for vectors; one iteration, MaxIterations -> k repeats); "
          "MeanShift[list, d, {p1, ...}] shifts only those positions; options MaxIterations, Tolerance, "
          "DistanceFunction (EuclideanDistance, ManhattanDistance, ChessboardDistance). Images, Weights and "
          "symbolic elements raise NotImplementedError / stay unevaluated; d < 0 ValueError. Check: reference "
          "values from the Wolfram documentation (MeanShift[{0, 10, 1, 11}, 1] = {1/2, 21/2, 1/2, 21/2}).")
def _mean_shift(x: ListOf[Real] | MatrixOf[Real], d: Real, *rest: ListOf[Integer] | Option) -> ListOf[Real] | Expr:
    name = "MeanShift"
    args, o = opts(rest, ("MaxIterations", "Tolerance", "DistanceFunction", "Weights"), name)
    if isinstance(x, Expr) and x.head == "Image":
        raise NotImplementedError(f"{name}: images are not supported")
    if "Weights" in o and str_or_sym(o["Weights"]) != "Automatic":
        raise NotImplementedError(f"{name}: Weights are not supported")
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x) or not x.args:
        raise TypeError(f"{name}: the first argument must be a non-empty list")
    dv = _num(d, name, "distance")
    if dv < 0:
        raise ValueError(f"{name}: the distance must be non-negative")
    vecs = is_matrix(x)
    rows = mat(x, name) if vecs else [[v] for v in vec(x, name)[0]]
    df = str_or_sym(o.get("DistanceFunction", Symbol("EuclideanDistance")))
    if df not in ("EuclideanDistance", "ManhattanDistance", "ChessboardDistance", "Automatic"):
        raise NotImplementedError(f"{name}: DistanceFunction -> {o['DistanceFunction']} is not supported")
    its = o.get("MaxIterations", 1)
    if not isinstance(its, int) or its < 1:
        raise ValueError(f"{name}: MaxIterations must be a positive integer")
    tol = _num(o.get("Tolerance", 0), name, "tolerance")
    parts = None
    if args:
        if len(args) > 1 or not is_list(args[0]):
            raise ValueError(f"{name}: the third argument must be a list of positions")
        n = len(rows)
        parts = set()
        for p in args[0].args:
            if not isinstance(p, int) or not (1 <= abs(p) <= n):
                raise ValueError(f"{name}: invalid position {p!r}")
            parts.add(p - 1 if p > 0 else n + p)

    def dist(a, b):
        diffs = [abs(p - q) for p, q in zip(a, b)]
        if df == "ManhattanDistance":
            return sp.Add(*diffs)
        if df == "ChessboardDistance":
            return sp.Max(*diffs)
        return sp.sqrt(sp.Add(*(t ** 2 for t in diffs)))
    cur = rows
    for _ in range(its):
        new = []
        for i, a in enumerate(cur):
            if parts is not None and i not in parts:
                new.append(a)
                continue
            nb = [b for b in cur if (dist(a, b) - dv) <= 0]
            new.append([sp.Add(*(b[k] for b in nb)) / len(nb) for k in range(len(a))])
        delta = max((dist(p, q) for p, q in zip(new, cur)), key=lambda z: fl(z))
        cur = new
        if fl(delta) <= fl(tol):
            break
    items = list(x.args)
    if vecs:
        return L(*(L(*(out(v, items) for v in r)) for r in cur))
    return L(*(out(r[0], items) for r in cur))


def _missingq(x):
    return isinstance(x, Expr) and x.head == "Missing"


@fn("MissingFallback", area=AREA, status="added",
    notes="MissingFallback[arg1, arg2, ...] gives the first argument that is not Missing[...] (the last argument "
          "when all are Missing); MissingFallback[] stays unevaluated. Check: reference values from the Wolfram "
          "documentation (MissingFallback[Missing[], 2] = 2, MissingFallback[Missing[\"Unknown\"], "
          "Missing[\"NotAvailable\"]] = Missing[\"NotAvailable\"]).")
def _missing_fallback(*args: Expr) -> Expr:
    if not args:
        raise Unevaluated()
    for a in args:
        if not _missingq(a):
            return a
    return args[-1]


@fn("TransformMissing", area=AREA, status="added",
    notes="TransformMissing[data, spec] replaces Missing[...] elements of a vector, or column-wise in a matrix, by "
          "spec \"Mean\", \"Median\", \"Mode\" (first most frequent), {\"Constant\", c}, \"Previous\" or \"Next\" "
          "(the nearest known value before / after, falling back to the other direction); TransformMissing[data, "
          "{spec1, spec2, ...}] gives one spec per column. Tabular, Dataset, TimeSeries and associations raise "
          "NotImplementedError; a column with no known values ValueError; unknown specs ValueError. Check: the "
          "known values are unchanged and the result has no Missing elements (asserted in code).")
def _transform_missing(x: ListOf[Expr] | MatrixOf[Expr], spec: String | ListOf[Expr] = None) -> ListOf[Expr]:
    name = "TransformMissing"
    if isinstance(x, Symbol):
        raise Unevaluated()
    if isinstance(x, Expr) and x.head in ("Tabular", "Dataset", "TimeSeries", "EventSeries", "Association"):
        raise NotImplementedError(f"{name}: {x.head} input is not supported (use a list or matrix)")
    if not is_list(x) or not x.args:
        raise TypeError(f"{name}: the data must be a non-empty list or matrix")
    if spec is None:
        raise NotImplementedError(f"{name}: the automatic specification is not supported; give \"Mean\", "
                                  "\"Median\", {\"Constant\", c}, ...")

    def fill(col, sp_):
        known = [(i, v) for i, v in enumerate(col) if not _missingq(v)]
        if not known:
            raise ValueError(f"{name}: a column has no known values")
        kv = [v for _, v in known]
        sname = str_or_sym(sp_)
        if is_list(sp_) and len(sp_.args) == 2 and sp_.args[0] == "Constant":
            rep = [sp_.args[1]] * len(col)
        elif sname in ("Mean", "Median"):
            nums = [S(v) for v in kv]
            if not all(isinstance(t, sp.Basic) and t.is_number for t in nums):
                raise TypeError(f"{name}: \"{sname}\" needs numeric known values")
            val = mean(nums) if sname == "Mean" else median(nums)
            rep = [out(val, kv)] * len(col)
        elif sname == "Mode":
            cnt = {}
            for v in kv:
                cnt[repr(v)] = cnt.get(repr(v), 0) + 1
            best = max(cnt.values())
            m = next(v for v in kv if cnt[repr(v)] == best)
            rep = [m] * len(col)
        elif sname in ("Previous", "Next"):
            rep = []
            for i in range(len(col)):
                if not _missingq(col[i]):
                    rep.append(col[i])
                    continue
                prev = [v for j, v in known if j < i]
                nxt = [v for j, v in known if j > i]
                if sname == "Previous":
                    rep.append(prev[-1] if prev else nxt[0])
                else:
                    rep.append(nxt[0] if nxt else prev[-1])
        else:
            raise ValueError(f"{name}: unknown specification {sp_!r}")
        res = [rep[i] if _missingq(v) else v for i, v in enumerate(col)]
        assert not any(_missingq(v) for v in res) and all(res[i] is col[i] for i, _ in known)
        return res
    if all(is_list(r) for r in x.args):
        ncol = len(x.args[0].args)
        if any(len(r.args) != ncol for r in x.args):
            raise ValueError(f"{name}: the matrix must be rectangular")
        cols = [[r.args[j] for r in x.args] for j in range(ncol)]
        if is_list(spec) and not (len(spec.args) == 2 and spec.args[0] == "Constant"):
            if len(spec.args) != ncol:
                raise ValueError(f"{name}: give one specification per column")
            specs = list(spec.args)
        else:
            specs = [spec] * ncol
        filled = [fill(c, s_) for c, s_ in zip(cols, specs)]
        return L(*(L(*(filled[j][i] for j in range(ncol))) for i in range(len(x.args))))
    if any(is_list(r) for r in x.args):
        raise TypeError(f"{name}: ragged data is not supported")
    return L(*fill(list(x.args), spec))


# formal moments: conversions through raw moments

def _formal(e):
    """sympy expression with formal moments replaced by symbols M_r (raw), C_r (central), K_r (cumulant), F_r"""
    syms = {}

    def walk(x):
        if isinstance(x, Expr) and x.head in ("Moment", "CentralMoment", "Cumulant", "FactorialMoment") and \
                len(x.args) == 1:
            r = x.args[0]
            if is_list(r):
                raise NotImplementedError("MomentConvert: multivariate formal moments are not supported")
            if not (isinstance(r, int) and r >= 0):
                raise ValueError(f"MomentConvert: invalid moment order {r!r}")
            s = sp.Symbol(f"w18{x.head}_{r}")
            syms[s] = (x.head, r)
            return Symbol(str(s))
        if isinstance(x, Expr) and x.head in ("PowerSymmetricPolynomial", "AugmentedSymmetricPolynomial"):
            raise NotImplementedError("MomentConvert: formal sample moments are not supported")
        if isinstance(x, Expr):
            return Expr(x.head, *(walk(a) for a in x.args))
        return x
    w = walk(e)
    return S(w), syms


def _raw_of(kind, r, cache):
    """the formal moment (kind, r) as a polynomial in raw moments m[k] (sympy symbols)"""
    m = lambda k: sp.Integer(1) if k == 0 else sp.Symbol(f"w18Moment_{k}")  # noqa: E731
    if kind == "Moment":
        return m(r)
    if kind == "CentralMoment":
        return sp.expand(sp.Add(*(sp.binomial(r, k) * m(k) * (-m(1)) ** (r - k) for k in range(r + 1))))
    if kind == "FactorialMoment":
        return sp.expand(sp.Add(*(sp.functions.combinatorial.numbers.stirling(r, k, kind=1, signed=True) * m(k)
                                  for k in range(r + 1))))
    if kind == "Cumulant":
        if r == 0:
            return sp.Integer(0)
        if r in cache:
            return cache[r]
        v = m(r) - sp.Add(*(sp.binomial(r - 1, j - 1) * _raw_of("Cumulant", j, cache) * m(r - j)
                            for j in range(1, r)))
        cache[r] = sp.expand(v)
        return cache[r]
    raise AssertionError(kind)


def _raw_in(target, r, cache):
    """raw moment m[r] in terms of the target family"""
    T = lambda k: sp.Symbol(f"w18{target}_{k}")  # noqa: E731
    if r == 0:
        return sp.Integer(1)
    if target == "Moment":
        return T(r)
    if target == "CentralMoment":
        m1 = sp.Symbol("w18Moment_1")
        return sp.expand(sp.Add(*(sp.binomial(r, k) * (T(k) if k >= 2 else (1 if k == 0 else 0)) * m1 ** (r - k)
                                  for k in range(r + 1))))
    if target == "FactorialMoment":
        return sp.expand(sp.Add(*(sp.functions.combinatorial.numbers.stirling(r, k) * T(k) for k in range(1, r + 1))))
    if target == "Cumulant":
        if r in cache:
            return cache[r]
        v = sp.Add(*(sp.binomial(r - 1, j - 1) * T(j) * _raw_in("Cumulant", r - j, cache) for j in range(1, r + 1)))
        cache[r] = sp.expand(v)
        return cache[r]
    raise AssertionError(target)


@fn("MomentConvert", area=AREA, status="added",
    notes="MomentConvert[mexpr, form] for polynomials in the univariate formal moments Moment[r], CentralMoment[r], "
          "Cumulant[r] and FactorialMoment[r], with form \"Moment\", \"CentralMoment\" (raw moments written with "
          "central moments and Moment[1]), \"Cumulant\" or \"FactorialMoment\" (exact polynomial identities through "
          "raw moments, e.g. Cumulant[2] -> Moment[2] - Moment[1]^2). Multivariate formal moments, formal sample "
          "moments (PowerSymmetricPolynomial, AugmentedSymmetricPolynomial) and the sample-estimator forms raise "
          "NotImplementedError; unknown forms ValueError. Check: reference values from the Wolfram documentation "
          "(Cumulant[2] -> -Moment[1]^2 + Moment[2]) and round trips (tested).")
def _moment_convert(e: Expr, form: String) -> Expr:
    name = "MomentConvert"
    forms = ("Moment", "CentralMoment", "Cumulant", "FactorialMoment")
    if form in ("UnbiasedSampleEstimator", "SampleEstimator", "PowerSymmetricPolynomial",
                "AugmentedSymmetricPolynomial"):
        raise NotImplementedError(f'{name}: the "{form}" form is not supported')
    if form not in forms:
        raise ValueError(f"{name}: unknown form {form!r} (use one of {', '.join(forms)})")
    fe, syms = _formal(e)
    if not isinstance(fe, sp.Basic):
        raise TypeError(f"{name}: the moment expression must be algebraic")
    if not syms:
        return e
    cache = {}
    raw = fe.subs({s: _raw_of(k, r, cache) for s, (k, r) in syms.items()})
    raw = sp.expand(raw)
    if form != "Moment":
        cache2 = {}
        ms = sorted({s for s in raw.free_symbols if str(s).startswith("w18Moment_")}, key=lambda s: -int(str(s)[10:]))
        sub = {}
        for s in ms:
            k = int(str(s)[10:])
            if form == "CentralMoment" and k == 1:
                continue
            sub[s] = _raw_in(form, k, cache2)
        raw = sp.expand(raw.subs(sub, simultaneous=True))
    from ..parser import parse
    txt = str(raw)
    back = {}
    for s in raw.free_symbols:
        nm = str(s)
        if nm.startswith("w18"):
            k, r = nm[3:].rsplit("_", 1)
            back[nm] = Expr(k, int(r))
    res = R(raw)
    return ev(ev(Expr("ReplaceAll", res, L(*(rule(Symbol(k), v) for k, v in back.items())))))


@fn("MomentEvaluate", area=AREA, status="added",
    notes="MomentEvaluate[mexpr, dist] replaces the formal moments Moment[r], CentralMoment[r], Cumulant[r] and "
          "FactorialMoment[r] by their values for the distribution (Moment[dist, r], ...); MomentEvaluate[mexpr, "
          "list] evaluates formal moments and the formal sample moments PowerSymmetricPolynomial[r] and "
          "AugmentedSymmetricPolynomial[{r1, ...}] on the data; MomentEvaluate[mexpr, dist, list] uses dist for "
          "the moments and list for the sample moments. Multivariate orders follow the underlying functions. "
          "Check: reference values from the Wolfram documentation (Moment[2] on NormalDistribution[mu, sigma] = "
          "mu^2 + sigma^2, PowerSymmetricPolynomial[2] on {a, b, c} = a^2 + b^2 + c^2).")
def _moment_evaluate(e: Expr, src: Expr | ListOf[Expr], data: ListOf[Expr] = None) -> Expr:
    name = "MomentEvaluate"
    if isinstance(src, Symbol) or isinstance(data, Symbol):
        raise Unevaluated()
    if data is None:
        if W.is_dist(src):
            dist, lst = src, None
        elif is_list(src):
            dist, lst = None, src
        else:
            raise TypeError(f"{name}: the second argument must be a distribution or a data list")
    else:
        if not W.is_dist(src) or not is_list(data):
            raise TypeError(f"{name}: expected MomentEvaluate[mexpr, dist, list]")
        dist, lst = src, data

    def walk(x):
        if isinstance(x, Expr) and x.head in ("Moment", "CentralMoment", "Cumulant", "FactorialMoment") and \
                len(x.args) == 1:
            s_ = dist if dist is not None else lst
            return Expr(x.head, s_, x.args[0])
        if isinstance(x, Expr) and x.head in ("PowerSymmetricPolynomial", "AugmentedSymmetricPolynomial") and \
                len(x.args) == 1:
            if lst is None:
                raise ValueError(f"{name}: formal sample moments need a data list")
            return Expr(x.head, x.args[0], lst)
        if isinstance(x, Expr):
            return Expr(x.head, *(walk(a) for a in x.args))
        return x
    return ev(Expr("Simplify", walk(e)))


# ─────────────────────────── multivariate dependence, autocorrelation, log-rank ───────────────────────────

def _two_mats(m1, m2, fname):
    def cols(m):
        if isinstance(m, Symbol):
            raise Unevaluated()
        if is_matrix(m):
            return [[fl(v) for v in r] for r in mat(m, fname)]
        v, _ = vec(m, fname, min_len=2)
        return [[fl(t)] for t in v]
    a, b = cols(m1), cols(m2)
    if len(a) != len(b):
        raise ValueError(f"{fname}: the arguments must have the same length")
    return a, b


def _pooled_cov(a, b, fname):
    import numpy as np
    X = np.hstack([np.array(a), np.array(b)])
    if X.shape[0] < 2:
        raise ValueError(f"{fname}: at least two observations are needed")
    C = np.cov(X, rowvar=False)
    p = len(a[0])
    S11, S12, S21, S22 = C[:p, :p], C[:p, p:], C[p:, :p], C[p:, p:]
    if abs(np.linalg.det(S11)) < 1e-300 or abs(np.linalg.det(S22)) < 1e-300:
        raise ValueError(f"{fname}: a covariance block is singular")
    return C, S11, S12, S21, S22


@fn("PillaiTrace", area=AREA, status="added",
    notes="PillaiTrace[m1, m2] = Tr[Inverse[S11].S12.Inverse[S22].S21] for the blocks of the pooled covariance "
          "matrix of the columns of m1 and m2 (vectors are one-column matrices); machine-precision result. Unequal "
          "lengths or singular blocks raise ValueError; non-numeric data TypeError; symbolic arguments stay "
          "unevaluated. Check: for two vectors PillaiTrace = Correlation^2 (tested).")
def _pillai(m1: ListOf[Real] | MatrixOf[Real], m2: ListOf[Real] | MatrixOf[Real]) -> Real:
    import numpy as np
    a, b = _two_mats(m1, m2, "PillaiTrace")
    _, S11, S12, S21, S22 = _pooled_cov(a, b, "PillaiTrace")
    return float(np.trace(np.linalg.solve(S11, S12) @ np.linalg.solve(S22, S21)))


@fn("WilksW", area=AREA, status="added",
    notes="WilksW[m1, m2] = 1 - Det[S]/(Det[S11] Det[S22]) for the pooled covariance matrix S of the columns of m1 "
          "and m2 (as documented); machine-precision result. Unequal lengths or singular blocks raise ValueError; "
          "non-numeric data TypeError; symbolic arguments stay unevaluated. Check: for two vectors WilksW = "
          "Correlation^2 (tested).")
def _wilks(m1: ListOf[Real] | MatrixOf[Real], m2: ListOf[Real] | MatrixOf[Real]) -> Real:
    import numpy as np
    a, b = _two_mats(m1, m2, "WilksW")
    C, S11, _, _, S22 = _pooled_cov(a, b, "WilksW")
    return float(1 - np.linalg.det(C) / (np.linalg.det(S11) * np.linalg.det(S22)))


def _stat_only_test(name, f):
    def impl(m1: ListOf[Real] | MatrixOf[Real], m2: ListOf[Real] | MatrixOf[Real], *rest: String | Option) -> Real:
        args, o = opts(rest, TEST_OPTS, name)
        (), prop = prop_arg(args, name, 0)
        if prop != "TestStatistic":
            raise NotImplementedError(f'{name}: only the "TestStatistic" property is supported (the null '
                                      "distribution Wolfram uses for the p-value is not documented)")
        return f(m1, m2)
    return impl


fn("PillaiTraceTest", area=AREA, status="added",
   notes="PillaiTraceTest[m1, m2, \"TestStatistic\"] gives PillaiTrace[m1, m2]; the p-value and other properties "
         "raise NotImplementedError (Wolfram's null approximation is not documented). Check: identical to "
         "PillaiTrace (tested).")(_stat_only_test("PillaiTraceTest", _pillai))
fn("WilksWTest", area=AREA, status="added",
   notes="WilksWTest[m1, m2, \"TestStatistic\"] gives WilksW[m1, m2]; the p-value and other properties raise "
         "NotImplementedError (Wolfram's null approximation is not documented). Check: identical to WilksW "
         "(tested).")(_stat_only_test("WilksWTest", _wilks))


@fn("AutocorrelationTest", area=AREA, status="added",
    notes="AutocorrelationTest[data], [data, k] and [data, k, spec] for univariate data: the Ljung-Box statistic "
          "n (n + 2) Sum[r_i^2/(n - i), {i, k}] (the automatic test, as documented) or Box-Pierce n Sum[r_i^2] "
          "against ChiSquareDistribution[k], r_i the sample autocorrelations; k = Automatic is Ceiling[Log[n]]; "
          "spec is \"LjungBox\", \"BoxPierce\", a list of them or {prop, test}. Multivariate data, TemporalData and "
          "All raise NotImplementedError; k >= n or k < 1 ValueError. Check: reference values from "
          "statsmodels.stats.diagnostic.acorr_ljungbox (lb and bp statistics).")
def _autocorr_test(x: ListOf[Real], *rest: Integer | Expr | String | ListOf[String] | Option) -> Real | Expr:
    name = "AutocorrelationTest"
    args, o = opts(rest, ("SignificanceLevel",), name)
    if isinstance(x, Expr) and x.head in ("TemporalData", "TimeSeries", "EventSeries"):
        raise NotImplementedError(f"{name}: TemporalData input is not supported (use a list of values)")
    if is_matrix(x):
        raise NotImplementedError(f"{name}: multivariate data is not supported")
    v, _ = vec(x, name, min_len=3)
    xs = _fvec(v)
    n = len(xs)
    k = args.pop(0) if args else Symbol("Automatic")
    if str_or_sym(k) == "Automatic":
        k = math.ceil(math.log(n))
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"{name}: the lag must be a positive integer or Automatic")
    if not 1 <= k < n:
        raise ValueError(f"{name}: the lag must satisfy 1 <= k < n")
    if args and str_or_sym(args[0]) in ("Automatic",):
        args = [] if len(args) == 1 else args
    # the documented automatic test for univariate data is Ljung-Box
    a0 = args[0] if args else None
    if not args:
        prop, chosen, several = None, ["LjungBox"], False
    elif isinstance(a0, str) and a0 not in ("LjungBox", "BoxPierce"):
        if len(args) > 1:
            raise ValueError(f"{name}: too many arguments")
        prop, chosen, several = a0, ["LjungBox"], False
    elif is_list(a0) and len(a0.args) == 2 and str_or_sym(a0.args[1]) == "Automatic":
        prop, chosen, several = a0.args[0], ["LjungBox"], False
    else:
        prop, chosen, several = test_spec(name, args, ("LjungBox", "BoxPierce"))
    m = _fmean(xs)
    c0 = math.fsum((t - m) ** 2 for t in xs)
    if c0 == 0:
        raise ValueError(f"{name}: the data is constant")
    r = [math.fsum((xs[i] - m) * (xs[i + h] - m) for i in range(n - h)) / c0 for h in range(1, k + 1)]
    res = []
    for t in chosen:
        if t == "LjungBox":
            q = n * (n + 2) * math.fsum(r[h - 1] ** 2 / (n - h) for h in range(1, k + 1))
        else:
            q = n * math.fsum(rr ** 2 for rr in r)
        res.append(test_result(name, q, chi2_sf(q, k), prop, o, k))
    return L(*res) if several else res[0]


def _event_groups(x, fname):
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x) or len(x.args) < 2:
        raise TypeError(f"{fname}: expected a list of at least two datasets {{data1, data2, ...}}")
    out_ = []
    for g in x.args:
        if isinstance(g, Expr) and g.head == "EventData" and len(g.args) == 2 and is_list(g.args[0]) and \
                is_list(g.args[1]):
            t, _ = vec(g.args[0], fname, "event times")
            c = list(g.args[1].args)
            if len(c) != len(t) or not all(q in (0, 1) for q in c):
                raise ValueError(f"{fname}: EventData censoring flags must be 0 (event) or 1 (right-censored), one "
                                 "per time")
            out_.append([(fl(a), q == 0) for a, q in zip(t, c)])
        elif isinstance(g, Expr) and g.head == "EventData":
            raise NotImplementedError(f"{fname}: only EventData[times, {{0 | 1, ...}}] (right censoring) is supported")
        elif is_list(g):
            if any(is_list(a) for a in g.args):
                raise NotImplementedError(f"{fname}: interval-censored or multivariate data is not supported")
            t, _ = vec(g, fname, "event times")
            out_.append([(fl(a), True) for a in t])
        else:
            raise TypeError(f"{fname}: each dataset must be a list or EventData")
    return out_


@fn("LogRankTest", area=AREA, status="added",
    notes="LogRankTest[{data1, data2, ...}] and [{data1, ...}, wspec, prop]: the weighted log-rank test of equal "
          "hazards, U = Sum[w (O - E)] over the distinct event times with the hypergeometric covariance, statistic "
          "U'.Inverse[V].U (first k - 1 groups) against ChiSquareDistribution[k - 1]; datasets are lists of event "
          "times or EventData[times, flags] (flag 1 = right-censored); wspec is rho or {rho, gamma} "
          "(Fleming-Harrington S(t-)^rho (1 - S(t-))^gamma with the pooled Kaplan-Meier estimate; default 0 = "
          "log-rank), \"Equal\", \"Gehan\" (number at risk), \"TaroneWare\" (its square root), \"Peto\" and "
          "\"AndersenPeto\". Interval censoring and other EventData forms raise NotImplementedError. Check: "
          "reference values from the Wolfram documentation (p = 0.192633; Gehan 3.35128, p = 0.0671529; {1, 0} "
          "3.54507, p = 0.0597226).")
def _logrank(x: ListOf[ListOf[Real]] | ListOf[Expr], *rest: Real | String | ListOf[Real] | Option) -> Real | Expr:
    import numpy as np
    name = "LogRankTest"
    args, o = opts(rest, ("SignificanceLevel",), name)
    gs = _event_groups(x, name)
    w = args.pop(0) if args else 0
    (), prop = prop_arg(args, name, 0)
    k = len(gs)
    allv = sorted({t for g in gs for t, e in g if e})
    if not allv:
        raise ValueError(f"{name}: there are no events")
    # pooled Kaplan-Meier (left-continuous) for the weights
    S_left, S = [], 1.0
    St_left = []
    St = 1.0
    stats_ = []
    for t in allv:
        y = [sum(1 for tt, _ in g if tt >= t) for g in gs]
        d = [sum(1 for tt, e in g if e and tt == t) for g in gs]
        Y, Dn = sum(y), sum(d)
        S_left.append(S)
        St_left.append(St)
        stats_.append((y, d, Y, Dn))
        S *= 1 - Dn / Y
        St *= 1 - Dn / (Y + 1)
    ws = []
    for (y, d, Y, Dn), sl, stl, st_after in zip(stats_, S_left, St_left, St_left[1:] + [St]):
        ws_name = str_or_sym(w) if isinstance(w, (str, Symbol)) else None
        if ws_name == "Equal":
            ws.append(1.0)
        elif ws_name == "Gehan":
            ws.append(float(Y))
        elif ws_name == "TaroneWare":
            ws.append(math.sqrt(Y))
        elif ws_name == "Peto":
            ws.append(st_after)
        elif ws_name == "AndersenPeto":
            ws.append(st_after * Y / (Y + 1))
        elif ws_name is not None:
            raise ValueError(f"{name}: unknown weight specification {w!r}")
        else:
            if is_list(w):
                if len(w.args) != 2:
                    raise ValueError(f"{name}: Fleming-Harrington weights are {{rho, gamma}}")
                rho, gam = (fl(_num(q, name, "weight parameter")) for q in w.args)
            else:
                rho, gam = fl(_num(w, name, "weight parameter")), 0.0
            if rho < 0 or gam < 0:
                raise ValueError(f"{name}: rho and gamma must be non-negative")
            ws.append(sl ** rho * (1 - sl) ** gam)
    U = np.zeros(k)
    V = np.zeros((k, k))
    for (y, d, Y, Dn), wt in zip(stats_, ws):
        for i in range(k):
            U[i] += wt * (d[i] - Dn * y[i] / Y)
            if Y > 1:
                for j in range(k):
                    V[i, j] += wt * wt * Dn * (Y - Dn) / (Y - 1) * (y[i] / Y) * ((1.0 if i == j else 0.0) - y[j] / Y)
    Ur, Vr = U[:-1], V[:-1, :-1]
    if abs(np.linalg.det(Vr)) < 1e-300:
        raise ValueError(f"{name}: the variance matrix is singular")
    chi = float(Ur @ np.linalg.solve(Vr, Ur))
    return test_result(name, chi, chi2_sf(chi, k - 1), prop, o, k - 1)


# ─────────────────────────── option symbols and Censoring ───────────────────────────

def _inert_option(name, notes):
    def _opt(*args: Expr) -> Expr:
        raise Unevaluated()
    fn(name, area=AREA, status="added", notes=notes + " Check: inert (stays unevaluated as a symbol, an expression or "
                                                     "in a rule).")(_opt)


for _n, _note in (
        ("AlternativeHypothesis", "option of the hypothesis tests (ZTest, SignTest, MannWhitneyTest, the correlation "
                                  "tests, ...): \"Unequal\" (default), \"Less\" or \"Greater\"; other values raise "
                                  "ValueError in the tests; inert as an expression."),
        ("SignificanceLevel", "option of the hypothesis tests: the cutoff (default 0.05) used by "
                              "\"ShortTestConclusion\"; must be in (0, 1) (ValueError otherwise); inert as an "
                              "expression."),
        ("VerifyTestAssumptions", "option of the hypothesis tests (Automatic, All, None, True/False, or a list or "
                                  "rules of assumptions); accepted, assumptions are not diagnosed; inert as an "
                                  "expression."),
        ("ConfidenceLevel", "option of fitting functions (LinearModelFit, NonlinearModelFit, "
                            "GeneralizedLinearModelFit, ...) for confidence intervals (default 0.95); inert as an "
                            "expression."),
        ("ConfidenceRange", "option of SurvivalModelFit: the range for simultaneous confidence bands (All, Full or "
                            "{tmin, tmax}); inert as an expression."),
        ("ConfidenceTransform", "option of SurvivalModelFit: the transformation used for confidence intervals "
                                "(Automatic, \"Log\", \"LogLog\", ...); inert as an expression."),
        ("Weights", "option of the fitting functions and of MeanShift / estimators: weights for the data "
                    "(Automatic, a list or a pure function); inert as an expression."),
        ("IncludeGroupAggregates", "option of PivotTable and related functions specifying whether to add group "
                                   "aggregates; inert as an expression."),
        ("IntervalMarkers", "option of plotting functions for Around values (\"Bars\", \"Bands\", \"Ellipses\", "
                            "...); inert as an expression."),
        ("IntervalMarkersStyle", "option of plotting functions giving the style of interval markers; inert as an "
                                 "expression."),
        ("LinearOffsetFunction", "option of GeneralizedLinearModelFit: a known offset added to the linear "
                                 "predictor (None by default); inert as an expression."),
        ("LinkFunction", "option of GeneralizedLinearModelFit: the link (Automatic = canonical, \"LogitLink\", "
                         "\"ProbitLink\", \"LogLink\", \"IdentityLink\", \"InverseLink\", ...); inert as an "
                         "expression."),
        ("ExponentialFamily", "option of GeneralizedLinearModelFit: \"Gaussian\" (default), \"Binomial\", "
                              "\"Poisson\", \"Gamma\", \"InverseGaussian\", ...; inert as an expression."),
        ("DispersionEstimatorFunction", "option of the fitting functions: how the dispersion is estimated; inert "
                                        "as an expression."),
        ("VarianceEstimatorFunction", "option of LinearModelFit / NonlinearModelFit: how the error variance is "
                                      "estimated (default residual sum of squares / degrees of freedom); inert as an "
                                      "expression."),
        ("CovarianceEstimatorFunction", "option of the fitting functions: \"ExpectedInformation\" or "
                                        "\"ObservedInformation\" for the parameter covariance; inert as an "
                                        "expression."),
        ("NominalVariables", "option of LinearModelFit and related functions: variables to treat as nominal "
                             "(Automatic, All, None, {v1, ...}); inert as an expression."),
        ("MissingDataMethod", "option of statistical functions specifying how Missing data is handled; inert as an "
                              "expression."),
        ("MissingDataRules", "option of SemanticImport and related functions: rules for values to read as "
                             "Missing; inert as an expression."),
        ("MissingValuePattern", "option of SynthesizeMissingValues and ToTabular: which elements count as missing "
                                "(Automatic: Missing[...] and Indeterminate); inert as an expression."),
        ("ParameterEstimator", "option of EstimatedDistribution / FindDistributionParameters: the estimator "
                               "(\"MaximumLikelihood\", \"MethodOfMoments\", ...); inert as an expression."),
        ("StrataVariables", "option of CoxModelFit: variables used to stratify the baseline hazard; inert as an "
                            "expression."),
        ("FitDegree", "option of LocalModelFit: the degree of the local polynomial; inert as an expression."),
        ("ClusterDissimilarityFunction", "option of FindClusters / ClusteringTree / Dendrogram: the linkage "
                                         "(\"Single\", \"Complete\", \"Average\", \"Centroid\", \"Ward\", ...); "
                                         "inert as an expression."),
        ("CriterionFunction", "option of ClusterClassify and related functions: the criterion used to choose a "
                              "method (\"StandardDeviation\", \"RSquared\", \"Dunn\", ...); inert as an expression."),
        ("DataReversed", "option of ArrayPlot and related functions: whether rows (and columns) are plotted in "
                         "reverse; inert as an expression."),
        ("ExcludedLines", "option of SemanticImport: line indices to ignore; inert as an expression."),
        ("PlotFitElements", "option of ListPlot / ListPlot3D: which elements to show for fitted models; inert as "
                            "an expression."),
):
    _inert_option(_n, _note)


_CENS = {0: 0, 1: 1, -1: -1, "None": 0, "Right": 1, "Left": -1}


def _cens_code(c, fname):
    key = c if isinstance(c, int) and not isinstance(c, bool) else str_or_sym(c)
    if key not in _CENS:
        if isinstance(c, Symbol) and c.name not in _CENS:
            raise Unevaluated()
        raise ValueError(f"{fname}: the censoring must be 0/None, 1/Right or -1/Left, not {c!r}")
    return _CENS[key]


@fn("Censoring", area=AREA, status="added",
    notes="Censoring[t, c], Censoring[{t1, ...}, c] and Censoring[{t1, ...}, {c1, ...}] represent censored event "
          "times with c = 0 or None (no censoring), 1 or Right (right censored), -1 or Left (left censored); the "
          "arguments are validated and the object stays as entered (Censoring[{...}, c] with one c is kept as "
          "given). Invalid censoring codes and mismatched lengths raise ValueError; non-numeric times TypeError; "
          "symbolic times stay unevaluated. Check: validated against the documented codes (tested).")
def _censoring(t: Real | ListOf[Real], c: Integer | Expr | ListOf[Integer]) -> Expr:
    name = "Censoring"
    if is_list(t):
        vec(t, name, "event times")
        if is_list(c):
            if len(c.args) != len(t.args):
                raise ValueError(f"{name}: the times and censoring codes must have the same length")
            for q in c.args:
                _cens_code(q, name)
        else:
            _cens_code(c, name)
    else:
        _num(t, name, "event time")
        if is_list(c):
            raise ValueError(f"{name}: a single time needs a single censoring code")
        _cens_code(c, name)
    raise Unevaluated()
