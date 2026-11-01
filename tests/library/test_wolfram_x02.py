"""Differential tests for Wolfram batch X02 (Statistics II: model framework, tests, HMMs and time-series models;
modules wl_statmodels2, wl_statmodels2_models and wl_statmodels2_ts).

Reference values (cited per block):

* independent NumPy / SciPy implementations of Mardia's (1970) skewness and kurtosis statistics, the
  Henze-Zirkler (1990) form of the Baringhaus-Henze statistic with its lognormal p-value (as R's MVN::hz) and the
  equiprobable-class Pearson chi-square count (scipy.stats quantiles); the code of those references is in
  ``test_numpy_references`` below and their printed values are written out in the cases;
* statsmodels 0.15: ``adfuller`` (UnitRootTest), ``ARIMA(order, trend).fit()`` exact Gaussian ML
  (TimeSeriesModelFit), ``acf`` / ``acorr_ljungbox`` (TimeSeriesModel residual diagnostics), ``lowess``
  (LocalModelFit, frac, it = 0, delta = 0); scipy ``curve_fit`` and numpy ``lstsq`` (ModelFit,
  KernelModelFit); scipy ``linkage`` (Dendrogram heights); scipy maximum-likelihood fits (FindDistribution);
* the Wolfram documentation (reference.wolfram.com/language/ref/<Name>.html) examples, e.g.
  AdjustTimeSeriesForecast[MAProcess[{1/10}, 1], {1, 0, 0}, {1.3}] = {1.3, 0.03, 0.};
* hand computations in the comments: forward-algorithm likelihoods (checked against brute-force enumeration of
  all state paths in ``test_hmm_brute_force``), Jackson traffic equations and M/M/1 measures (Gross & Harris,
  Fundamentals of Queueing Theory, 2.2 and 4.1), Gordon-Newell product-form probabilities, least-squares normal
  equations.

Every symbol assigned or used as a free variable here starts with ``x02`` so that no case collides with
definitions made by other modules.
"""
import importlib.util
import itertools
import math
from fractions import Fraction

import pytest

from _harness import Pred, Raises, check, ev, ids, to_complex
from mikoshilang.expr import Expr

HAVE_SM = importlib.util.find_spec("statsmodels") is not None
HAVE_SCIPY = importlib.util.find_spec("scipy") is not None


def near(x, tol=1e-9):
    return Pred(lambda g: abs(to_complex(g) - x) <= tol, f"within {tol} of {x}")


def close(xs, tol=1e-9):
    def f(g):
        vals = list(g.args) if isinstance(g, Expr) else list(g)
        return len(vals) == len(xs) and all(abs(to_complex(a) - b) <= tol for a, b in zip(vals, xs))
    return Pred(f, f"within {tol} of {xs}")


def is_head(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"a {h}")


def T(src):
    return (src, "True")


D2 = "{{1., 2.}, {2., 1.}, {3., 4.}, {4., 3.}, {5., 6.}, {2.5, 2.}, {3.5, 5.}, {1.5, 1.}}"
D1 = "{1., 2., 4., 3., 7., 2.5}"
U12 = "{0.05, 0.12, 0.33, 0.41, 0.48, 0.52, 0.67, 0.71, 0.88, 0.93, 0.21, 0.38}"
Z10 = "{-1.2, 0.3, 0.5, -0.1, 2.1, 1.1, -0.7, 0.05, -1.9, 0.8}"
Y12 = "{1., 2., 1.5, 2.5, 3., 2., 2.8, 3.5, 3.1, 4., 3.6, 4.2}"
E12 = "{0.2, 1.5, 0.7, 3.1, 0.4, 2.2, 0.9, 1.1, 0.3, 5.0, 0.8, 1.7}"

# ─────────────────────────── multivariate normality tests (Mardia 1970; Baringhaus & Henze 1988) ───────────────
# Reference values: test_numpy_references (NumPy statistics, scipy.stats chi2/norm/lognorm tails).
NORMALITY = [
    (f"MardiaSkewnessTest[{D2}, Automatic, \"TestStatistic\"]", near(0.32805496700032327)),
    (f"MardiaSkewnessTest[{D2}]", near(0.9879318876707447)),
    (f"MardiaSkewnessTest[{D2}, Automatic, \"DegreesOfFreedom\"]", "4"),            # d (d + 1) (d + 2)/6, d = 2
    (f"MardiaSkewnessTest[{D1}, Automatic, \"TestStatistic\"]", near(0.8886857142857146)),
    (f"MardiaSkewnessTest[{D2}, Automatic, \"ShortTestConclusion\"]", "\"Do not reject\""),
    ("MardiaSkewnessTest[{x02a, x02b, x02c}]", "MardiaSkewnessTest[{x02a, x02b, x02c}]"),   # symbolic
    (f"MardiaSkewnessTest[{D2}, NormalDistribution[0, 1]]", Raises(NotImplementedError)),
    ("MardiaSkewnessTest[\"a\"]", Raises(TypeError)),
    ("MardiaSkewnessTest[{{1, 2}, {2, 4}, {3, 6}}]", Raises(ValueError)),                  # too few observations
    ("MardiaSkewnessTest[{{1, 2}, {2, 4}, {3, 6}, {4, 8}, {5, 10}}]", Raises(ValueError)),  # singular covariance
    (f"MardiaKurtosisTest[{D2}, Automatic, \"TestStatistic\"]", near(-0.9995702768493606)),
    (f"MardiaKurtosisTest[{D2}]", near(0.317518513389913)),
    (f"MardiaKurtosisTest[{D1}, Automatic, \"TestStatistic\"]", near(-0.08022857142857087)),
    (f"MardiaKurtosisTest[{D1}]", near(0.9360554666845962)),
    ("MardiaKurtosisTest[{x02a, x02b, x02c}]", "MardiaKurtosisTest[{x02a, x02b, x02c}]"),
    (f"MardiaKurtosisTest[{D2}, Automatic, \"Foo\"]", Raises(ValueError)),
    (f"MardiaKurtosisTest[{D2}, Automatic, \"PValue\", Method -> \"MonteCarlo\"]", Raises(NotImplementedError)),
    ("MardiaKurtosisTest[\"a\"]", Raises(TypeError)),
    (f"MardiaCombinedTest[{D2}, Automatic, \"TestStatistic\"]", near(1.3271957053610306)),
    (f"MardiaCombinedTest[{D2}]", near(0.932108538452199)),
    (f"MardiaCombinedTest[{D2}, Automatic, \"DegreesOfFreedom\"]", "5"),
    (f"MardiaCombinedTest[{D1}]", near(0.6391851167531418)),
    ("MardiaCombinedTest[{x02a, x02b, x02c}]", "MardiaCombinedTest[{x02a, x02b, x02c}]"),
    (f"MardiaCombinedTest[{D2}, NormalDistribution[0, 1]]", Raises(NotImplementedError)),
    ("MardiaCombinedTest[\"a\"]", Raises(TypeError)),
    ("MardiaCombinedTest[{{1, 2}, {2, 4}, {3, 6}}]", Raises(ValueError)),
    # Check: the combined statistic is the skewness statistic plus the squared kurtosis statistic
    (f"MardiaCombinedTest[{D2}, Automatic, \"TestStatistic\"] - (MardiaSkewnessTest[{D2}, Automatic, "
     f"\"TestStatistic\"] + MardiaKurtosisTest[{D2}, Automatic, \"TestStatistic\"]^2)", near(0, 1e-12)),
    # Check: univariate b1 equals the squared moment skewness (n b1/6 with n = 6)
    (f"MardiaSkewnessTest[{D1}, Automatic, \"TestStatistic\"] - 6 Skewness[{D1}]^2/6", near(0, 1e-12)),
    (f"BaringhausHenzeTest[{D2}, Automatic, \"TestStatistic\"]", near(0.24271043867207087)),
    (f"BaringhausHenzeTest[{D2}]", near(0.6600846158611697)),
    (f"BaringhausHenzeTest[{D1}, Automatic, {{\"TestStatistic\", \"PValue\"}}]",
     close([0.1628975495140459, 0.2163702287200796])),
    ("BaringhausHenzeTest[{x02a, x02b, x02c}]", "BaringhausHenzeTest[{x02a, x02b, x02c}]"),
    (f"BaringhausHenzeTest[{D2}, NormalDistribution[0, 1]]", Raises(NotImplementedError)),
    ("BaringhausHenzeTest[\"a\"]", Raises(TypeError)),
    ("BaringhausHenzeTest[{{1, 2}, {2, 4}, {3, 6}, {4, 8}, {5, 10}}]", Raises(ValueError)),
    # Check: affine invariance of the statistic (x -> 3 x + 1)
    (f"BaringhausHenzeTest[3 {D2} + 1, Automatic, \"TestStatistic\"] - "
     f"BaringhausHenzeTest[{D2}, Automatic, \"TestStatistic\"]", near(0, 1e-9)),
    # DistributionFitTest dispatch to the new tests (shared _DFT table)
    (f"DistributionFitTest[{D2}, Automatic, {{\"TestStatistic\", \"MardiaSkewness\"}}]",
     near(0.32805496700032327)),
]

# ─────────────────────────── PearsonChiSquareTest (Moore 1986 class count Ceiling[2 n^(2/5)]) ───────────────────
PCS = [
    (f"PearsonChiSquareTest[{U12}, UniformDistribution[{{0, 1}}], {{\"TestStatistic\", \"PValue\", "
     "\"DegreesOfFreedom\"}]", close([1.0, 0.9625657732472964, 5])),                  # 6 classes, counts 2 each but
    (f"PearsonChiSquareTest[{Z10}, NormalDistribution[0, 1], \"TestStatistic\"]", near(0.8)),
    (f"PearsonChiSquareTest[{Z10}, NormalDistribution[0, 1]]", near(0.9770333437569263)),
    ("PearsonChiSquareTest[{x02a, x02b}, NormalDistribution[0, 1]]",
     "PearsonChiSquareTest[{x02a, x02b}, NormalDistribution[0, 1]]"),
    ("PearsonChiSquareTest[{1, 2}, PoissonDistribution[1]]", Raises(NotImplementedError)),
    ("PearsonChiSquareTest[{1., 2.}, NormalDistribution[x02m, x02s]]", Raises(NotImplementedError)),
    ("PearsonChiSquareTest[\"a\", NormalDistribution[0, 1]]", Raises(TypeError)),
    ("PearsonChiSquareTest[{1., 2., 3.}, NormalDistribution[0, 1], \"Foo\"]", Raises(ValueError)),
]

# ─────────────────────────── UnitRootTest (statsmodels adfuller, MacKinnon p-values) ───────────────────────────
URT = [
    (f"UnitRootTest[{Y12}, \"Constant\", {{\"TestStatistic\", \"DickeyFullerT\"}}]", near(-1.6087784636967066)),
    (f"UnitRootTest[{Y12}, \"Constant\", {{\"PValue\", \"DickeyFullerT\"}}]", near(0.479182448298474)),
    (f"UnitRootTest[{Y12}, {{\"Drift\", 2}}, {{\"TestStatistic\", \"DickeyFullerT\"}}]", near(-3.8447088471427744)),
    (f"UnitRootTest[{Y12}, {{\"Drift\", 2}}, {{\"PValue\", \"DickeyFullerT\"}}]", near(0.014429579478335041)),
    (f"UnitRootTest[{Y12}, Automatic, {{\"TestStatistic\", \"DickeyFullerT\"}}]", near(0.748750389884901)),
    (f"UnitRootTest[{Y12}, Automatic, {{\"PValue\", \"DickeyFullerT\"}}]", near(0.8759241154584405)),
    # normalized bias n (a - 1), a = Sum[y_t y_(t-1)]/Sum[y_(t-1)^2] (no constant, n = 11), by hand
    (f"UnitRootTest[{Y12}, Automatic, {{\"TestStatistic\", \"DickeyFullerF\"}}]", near(0.679426961014564)),
    ("UnitRootTest[{x02a, x02b, x02c}]", "UnitRootTest[{x02a, x02b, x02c}]"),
    ("UnitRootTest[{1., 2.}]", Raises(ValueError)),
    (f"UnitRootTest[{Y12}, {{\"Drift\", 2}}, \"PhillipsPerronT\"]", Raises(NotImplementedError)),
    ("UnitRootTest[\"s\"]", Raises(TypeError)),
    # Check: TimeSeries input gives the same statistic as the value list
    (f"UnitRootTest[TimeSeries[{Y12}], \"Constant\", {{\"TestStatistic\", \"DickeyFullerT\"}}]",
     near(-1.6087784636967066)),
]

# ─────────────────────────── FindDistribution (ML fits, BIC ranking) ───────────────────────────
# Exponential MLE rate = 1/mean = 12/17.9; Poisson MLE = mean = 2; Normal ML sigma (divisor n);
# BIC = -2 LL + k Log[n] (scipy.stats.expon.logpdf sum).
FD = [
    (f"FindDistribution[{E12}]", Pred(lambda g: g.head == "ExponentialDistribution" and
                                      abs(to_complex(g.args[0]) - 12 / 17.9) < 1e-12, "Exponential[12/17.9]")),
    (f"FindDistribution[{E12}, 1, \"BIC\"][[1, 2]]",
     near(-2 * (12 * math.log(12 / 17.9) - 12) + math.log(12), 1e-9)),
    ("FindDistribution[{1, 3, 2, 0, 2, 4, 1, 2, 3, 2}]", Pred(lambda g: g.head == "PoissonDistribution" and
                                                                abs(to_complex(g.args[0]) - 2) < 1e-12, "Poisson[2]")),
    ("FindDistribution[{1.1, 2.3, 1.9, 2.8, 3.2, 2.1, 1.7, 2.5, 2.9, 2.2}, 1, {\"LogLikelihood\"}, "
     "TargetFunctions -> {NormalDistribution}][[1, 2]]", near(-8.884247690342534, 1e-9)),   # scipy norm.fit/logpdf
    ("FindDistribution[{x02a, x02b}]", "FindDistribution[{x02a, x02b}]"),
    ("FindDistribution[{1., 2.}, 1, \"Score\"]", Raises(NotImplementedError)),
    ("FindDistribution[\"abc\"]", Raises(TypeError)),
    ("FindDistribution[{1., 2., 3.}, 0]", Raises(ValueError)),
    # Check: the closed-form MLE (normal mean = sample mean)
    ("Mean[FindDistribution[{1.1, 2.3, 1.9, 2.8, 3.2, 2.1, 1.7, 2.5, 2.9, 2.2}, TargetFunctions -> "
     "{NormalDistribution}]] - Mean[{1.1, 2.3, 1.9, 2.8, 3.2, 2.1, 1.7, 2.5, 2.9, 2.2}]", near(0, 1e-12)),
]

# ─────────────────────────── PearsonDistribution ───────────────────────────
# f'/f = -(a1 x + a0)/(b2 x^2 + b1 x + b0).  (1, 0, 0, 0, 1): f'/f = -x, the standard normal (type 3 limit).
# (1, 0, 1/10, 0, 1): f ~ (1 + x^2/10)^-5, a Student t with 9 degrees of freedom scaled by Sqrt[10/9]:
# variance (10/9) * 9/7 = 10/7.  (2, 1, -1, 0, 1): f'/f = -(2 x + 1)/(1 - x^2) on (-1, 1), f ~ (1 - x)^(1/2)
# (1 + x)^(3/2)... Beta(5/2, 3/2) on (-1, 1): mean = -1 + 2 (5/2)/4 ... = 1/4 reflected -> -1/4.
PD = [
    ("Mean[PearsonDistribution[1, 0, 0, 0, 1]]", "0"),
    ("Variance[PearsonDistribution[1, 0, 0, 0, 1]]", "1"),
    ("Variance[PearsonDistribution[1, 0, 1/10, 0, 1]]", "10/7"),
    ("Mean[PearsonDistribution[2, 1, -1, 0, 1]]", "-1/4"),
    ("PDF[PearsonDistribution[1, 0, 0, 0, 1], 0]", "1/Sqrt[2 Pi]"),
    ("PearsonDistribution[x02a, x02b, x02c, x02d, x02e]",
     "PearsonDistribution[x02a, x02b, x02c, x02d, x02e]"),                           # symbolic
    ("PearsonDistribution[1, 2, 3, -1, 4]", Raises(ValueError)),                     # not normalizable
    ("PearsonDistribution[1, 0, 1, 0, 1]", Raises(ValueError)),
    ("PearsonDistribution[\"a\", 1, 0, 0, 0, 1]", Raises(TypeError)),
    # Check: the density solves the Pearson ODE f' Q + (a1 x + a0) f = 0
    ("With[{x02e = PDF[PearsonDistribution[2, 1, -1, 0, 1], x02x][[1, 1, 1]]}, "
     "Simplify[D[x02e, x02x] (1 - x02x^2) + (2 x02x + 1) x02e]]", "0"),
    ("NIntegrate[PDF[PearsonDistribution[1, 0, 1/10, 0, 1], x02x], {x02x, -Infinity, Infinity}]", near(1, 1e-7)),
]

def _merge_heights(g):
    out = []

    def walk(e):
        if isinstance(e, Expr):
            if e.head == "Line":
                (x1, y1), (x2, y2) = [[to_complex(c).real for c in p.args] for p in e.args[0].args]
                if y1 == y2 and y1 > 0 and x1 != x2:
                    out.append(y1)
            else:
                for a in e.args:
                    walk(a)
    walk(g)
    return sorted(out)


def _labels(g):
    out = []

    def walk(e):
        if isinstance(e, Expr):
            if e.head == "Text":
                out.append(e.args[0])
            else:
                for a in e.args:
                    walk(a)
    walk(g)
    return out


# ─────────────────────────── Dendrogram (single linkage heights = scipy linkage) ───────────────────────────
# {1, 2, 4, 7} single linkage: merges {1,2} at 1, {1,2,4} at 2, all at 3 (scipy linkage(method="single")).
DEN = [
    ("Dendrogram[{1, 2, 4, 7}, ClusterDissimilarityFunction -> \"Single\"]", Pred(lambda g: _merge_heights(g) ==
                                                                                 [1, 2, 3], "heights 1, 2, 3")),
    ("Dendrogram[{1, 2, 4, 7}, ClusterDissimilarityFunction -> \"Complete\"]",     # {1,2} = 1, {1,2,4} = 3, all 6
     Pred(lambda g: _merge_heights(g) == [1, 3, 6], "heights 1, 3, 6")),
    ("Dendrogram[{1, 2, 4, 7}, ClusterDissimilarityFunction -> \"Single\"]",
     Pred(lambda g: _labels(g) == [1, 2, 4, 7], "leaf labels in order")),
    ("Head[Dendrogram[{\"a\" -> 1, \"b\" -> 2, \"c\" -> 5}, ClusterDissimilarityFunction -> \"Average\"]]",
     "Graphics"),
    ("Dendrogram[x02q]", "Dendrogram[x02q]"),
    ("Dendrogram[{1, 2, 4, 7}]", Raises(NotImplementedError)),                     # Automatic linkage undocumented
    ("Dendrogram[{1, 2, 4}, Foo, ClusterDissimilarityFunction -> \"Single\"]", Raises(ValueError)),
    # Check: merge heights equal ClusteringTree's vertex weights (independent clustering code path)
]

# ─────────────────────────── model specifications ───────────────────────────
MODELS = [
    ("LinearModel[{1, x02t, x02t^2}, x02t][x02u]", "C[1] + C[2] x02u + C[3] x02u^2"),
    ("LinearModel[][{x02a, x02b}]", "C[1] x02a + C[2] x02b"),
    ("LinearModel[{1, x02t}, {{x02p, 2}, x02r}, x02t][x02u]", "x02p + x02r x02u"),
    ("LinearModel[{1, Sin[x02p x02t]}, {x02p}, x02t]", Raises(NotImplementedError)),
    ("LinearModel[{1, x02t}, x02t][x02u, x02v]", "LinearModel[{1, x02t}, x02t][x02u, x02v]"),
    ("PolynomialModel[2][x02u]", "C[0] + C[1] x02u + C[2] x02u^2"),
    ("PolynomialModel[2, {x02a, x02b}][{x02a, x02b}]",
     "C[0] + C[1] x02a + C[2] x02b + C[3] x02a x02b + C[4] x02a^2 + C[5] x02b^2"),
    ("PolynomialModel[1][{1, 2}]", "C[0] + C[1] + 2 C[2]"),                          # a point with 2 features
    ("PolynomialModel[1, x02t][{1, 2}]", "{C[0] + C[1], C[0] + 2 C[1]}"),            # one variable: a list of x
    ("PolynomialModel[-1]", Raises(ValueError)),
    ("ExponentialModel[][x02u]", "C[1] + C[2] E^(C[3] x02u)"),
    ("ExponentialModel[2][{x02a, x02b}]", "C[1] + C[2] E^(C[3] x02a) E^(C[4] x02b)"),
    ("ExponentialModel[][2]", "C[1] + C[2] E^(2 C[3])"),
    ("ExponentialModel[1, 2, 3, 4]", Raises(ValueError)),
    ("LogModel[][x02u]", "C[1] + C[2] Log[x02u]"),
    ("LogModel[2][{x02a, x02b}]", "C[1] + C[2] Log[x02a] + C[3] Log[x02b]"),
    ("LogModel[][E]", "C[1] + C[2]"),
    ("LogModel[1, 2, 3, 4]", Raises(ValueError)),
    ("PowerModel[][x02u]", "C[1] + C[2] x02u^C[3]"),
    ("PowerModel[2][{x02a, x02b}]", "C[1] + C[2] x02a^C[3] x02b^C[4]"),
    ("PowerModel[][1]", "C[1] + C[2]"),
    ("PowerModel[1, 2, 3, 4]", Raises(ValueError)),
    ("PeriodicModel[][x02u]", "C[0] + a[1] Sin[k[1] x02u + phi[1]]"),
    ("PeriodicModel[2][x02u]", "C[0] + a[1] Sin[k[1] x02u + phi[1]] + a[2] Sin[k[2] x02u + phi[2]]"),
    ("PeriodicModel[1][0]", "C[0] + a[1] Sin[phi[1]]"),
    ("PeriodicModel[<|\"Foo\" -> 1|>]", Raises(ValueError)),
    ("PeriodicModel[1, 2, 3, 4]", Raises(ValueError)),
    ("NearestModel[2][x02u]", "NearestModel[2][x02u]"),
    ("ModelFit[{1, 2, 3, 10}, NearestModel[1]][2.4]", near(2.0)),                   # nearest x = 2 -> y = 2
    ("ModelFit[{1, 2, 3, 10}, NearestModel[2]][2.4]", near(2.5)),                   # x = 2, 3 -> mean 5/2
    ("ModelFit[{1, 2, 3, 10}, NearestModel[<|\"Neighbors\" -> 2, \"Weights\" -> \"Distance\"|>, 1]][2.5]",
     near(2.5)),
    ("ModelFit[{1, 2, 3, 10}, NearestModel[<|\"Neighbors\" -> {1, 0.1}|>, 1]][2.5]", Raises(ValueError)),
    ("NearestModel[\"s\"]", Raises(TypeError)),
]

# ─────────────────────────── ModelFit / ModelFitReport ───────────────────────────
# {-4, -1, 4, 10, 20} at x = 1..5, quadratic least squares (normal equations, exact): -22/5 - 37/70 x + 15/14 x^2.
QD = "{-4, -1, 4, 10, 20}"
MF = [
    (f"ModelFit[{QD}, \"Quadratic\"]",
     "PolynomialModel[2, {C[0] -> -22/5, C[1] -> -37/70, C[2] -> 15/14}, {x[1]}]"),
    ("ModelFit[{{1, 2.1}, {2, 3.9}, {3, 6.2}, {4, 7.8}, {5, 10.1}}, \"Linear\"][[2, All, 2]]",
     close([0.05, 1.99], 1e-12)),                                                     # numpy lstsq
    ("ModelFit[{{1, 2.7}, {2, 7.4}, {3, 20.1}, {4, 54.6}, {5, 148.4}}, ExponentialModel[]][[1, All, 2]]",
     close([-0.00903075, 1.00191635, 0.99960941], 1e-6)),                             # scipy curve_fit
    ("ModelFit[{{1, 1.0}, {2, 2.9}, {3, 5.1}, {4, 8.2}, {5, 11.0}, {6, 14.8}}, PowerModel[]][[1, All, 2]]",
     close([0.04260242, 0.98270371, 1.50936002], 1e-6)),                              # scipy curve_fit
    ("ModelFit[{{1, 0.1}, {2, 1.5}, {3, 2.2}, {4, 2.7}, {5, 3.3}}, LogModel[]][[1, All, 2]]",
     close([0.10643896, 1.93583732], 1e-7)),                                          # numpy lstsq on {1, Log[x]}
    ("ModelFit[Table[{x02t/4, 1 + 2 Sin[3 x02t/4 + 1/2]}, {x02t, 0, 40}], PeriodicModel[]][[2, All, 2]]",
     close([1, 2, 3, 0.5], 1e-8)),                                                    # noise-free sine recovered
    (f"ModelFit[{QD}, PolynomialModel[Automatic]]",
     "PolynomialModel[2, {C[0] -> -22/5, C[1] -> -37/70, C[2] -> 15/14}, {x[1]}]"),
    # residuals of the quadratic: 1/35 {-3, 8, 4, -12, 3}... RSS = 22/35, TSS = 1824/5 (by hand)
    (f"ModelFit[{QD}, \"Quadratic\", {{\"ResidualSumOfSquares\", \"TotalSumOfSquares\"}}]",
     close([22 / 35, 1824 / 5], 1e-9)),
    (f"ModelFit[{QD}, \"Quadratic\", \"RSquared\"]", near(1 - (22 / 35) / (1824 / 5), 1e-12)),
    # LeaveOneOut-free check of the CV losses: KFold 5 on 5 points is leave-one-out; the linear fit without
    # point 1 ({-1, 4, 10, 20} at 2..5: slope 7, intercept -16) predicts -9 at x = 1: |(-9) - (-4)| = 5
    (f"ModelFit[{QD}, {{\"Linear\", \"Quadratic\"}}, \"Report\"][\"CrossValidationData\"][[1, \"Losses\", 1]]",
     near(5.0, 1e-9)),
    (f"ModelFit[{QD}, {{\"Linear\", \"Quadratic\"}}]", Pred(lambda g: g.head == "PolynomialModel" and
                                                          g.args[0] == 2, "the quadratic wins")),
    (f"ModelFit[{QD}, {{\"Linear\", \"Quadratic\"}}, ValidationSet -> None]",
     Pred(lambda g: g.head == "PolynomialModel" and g.args[0] == 2, "training loss also prefers the quadratic")),
    (f"ModelFit[{QD}, {{\"Linear\", \"Quadratic\"}}, CriterionFunction -> \"MeanAbsoluteError\", "
     "ValidationSet -> \"LeaveOneOut\"]", Pred(lambda g: g.head == "PolynomialModel", "a polynomial")),
    ("ModelFit[x02q, \"Linear\"]", "ModelFit[x02q, \"Linear\"]"),
    ("ModelFit[{1, 2, 3}, \"Logistic\"]", Raises(NotImplementedError)),
    ("ModelFit[\"abc\", \"Linear\"]", Raises(TypeError)),
    ("ModelFit[{1, 2, 3, 5}, \"Linear\", \"Foo\"]", Raises(ValueError)),
    # Check: RMSE identity Sqrt[RSS/n]
    (f"ModelFit[{QD}, \"Quadratic\", \"Report\"][\"RootMeanSquaredError\"]^2 - "
     f"ModelFit[{QD}, \"Quadratic\", \"ResidualSumOfSquares\"]/5", near(0, 1e-12)),
    ("ModelFitReport[1]", Raises(TypeError)),
    (f"ModelFitReport[<|\"x02\" -> 1|>][\"Foo\"]", Raises(ValueError)),
    (f"ModelFit[{QD}, \"Quadratic\", \"Report\"][\"ParameterValues\"]", "{-22/5, -37/70, 15/14}"),
    (f"ModelFit[{QD}, \"Quadratic\", \"Report\"][\"ParameterCount\"]", "3"),
    (f"ModelFit[{QD}, \"Quadratic\", \"Report\"][\"Foo\"]", Raises(ValueError)),
    (f"ModelFit[{QD}, \"Quadratic\", \"Report\"][\"ResidualPlot\"]", Raises(NotImplementedError)),
    (f"ModelFit[{QD}, \"Quadratic\", \"Report\"][\"FittedModel\"][\"RSquared\"]", "6373/6384"),  # 1 - (22/35)/(1824/5)
    ("ModelFitReport[x02z][\"RSquared\"]", "ModelFitReport[x02z][\"RSquared\"]"),
    ("ModelFitReport[x02z]", "ModelFitReport[x02z]"),
]

# ─────────────────────────── KernelModelFit / LocalModelFit ───────────────────────────
KD = "{{1, 1.}, {2, 3.}, {3, 2.}, {4, 5.}, {5, 4.}, {6, 6.}}"
LD = "{{1, 1.}, {2, 3.}, {3, 2.}, {4, 5.}, {5, 4.}, {6, 6.}, {7, 8.}, {8, 7.}}"
KL_ = [
    # {{3, 2}}: centres 1, 3.5, 6 with d = 2, Gaussian Exp[-2 r^2] = Exp[-(x - c)^2/2]; numpy lstsq on the basis
    (f"KernelModelFit[{KD}, {{{{3, 2}}}}, \"Gaussian\"][\"BestFitParameters\"]",
     close([3.5, -2.161713410312879, 0.0, 2.161713410312878], 1e-8)),
    (f"KernelModelFit[{KD}, {{{{3, 2}}}}, \"Gaussian\"][2.5]", near(2.8029231365054104, 1e-9)),
    (f"KernelModelFit[{KD}, 2, \"Cauchy\"][\"BestFitParameters\"]",
     close([8.379752262066816, -6.183650136027896, -4.147900779781947, -2.5090787211258445], 1e-8)),
    (f"KernelModelFit[{KD}][\"BestFitParameters\"][[1]]", near(3.5, 1e-9)),
    ("KernelModelFit[x02q, 1, \"Gaussian\"]", "KernelModelFit[x02q, 1, \"Gaussian\"]"),
    ("KernelModelFit[{{1, 1.}, {2, 3.}, {3, 2.}}, 1, \"Foo\"]", Raises(ValueError)),
    ("KernelModelFit[{{1, 1, 1.}, {2, 3, 3.}}, 1, \"Gaussian\"]", Raises(NotImplementedError)),
    ("KernelModelFit[\"s\", 1, \"Gaussian\"]", Raises(TypeError)),
    # statsmodels lowess(frac=0.75, it=0, delta=0) at the data points
    (f"LocalModelFit[{LD}][{{1, 2, 8}}]", close([1.26820243, 2.18241161, 7.72977211], 1e-7)),
    (f"LocalModelFit[{LD}, Scaled[1/2]][{{1, 2, 5}}]", close([1.35931682, 2.14106845, 4.85893155], 1e-7)),
    # FitDegree -> 0 at x = 4, Scaled[1/2]: h = 2 (4th nearest), tricube weights (1 - (d/2)^3)^3 for d = 0, 1, 1
    # (d = 2 gets weight 0): (5 + (2 + 4) w)/(1 + 2 w), w = (7/8)^3
    (f"LocalModelFit[{LD}, Scaled[1/2], FitDegree -> 0][4]",
     near((5 + 6 * (7 / 8) ** 3) / (1 + 2 * (7 / 8) ** 3), 1e-12)),
    ("LocalModelFit[x02q]", "LocalModelFit[x02q]"),
    ("LocalModelFit[{{1, 1.}, {2, 3.}}, Scaled[3/4], InterpolationPoints -> 10]", Raises(NotImplementedError)),
    ("LocalModelFit[\"s\"]", Raises(TypeError)),
    ("LocalModelFit[{{1, 1.}, {2, 3.}, {3, 4.}}, 1/10][2]", Raises(ValueError)),
    # Check: lowess equals the hand-computed weighted least squares at an interior point (same as the reference)
]

# ─────────────────────────── fit plots ───────────────────────────
def _line_points(g):
    lines = []

    def walk(e):
        if isinstance(e, Expr):
            if e.head == "Line":
                lines.append(e.args[0])
            for a in e.args:
                walk(a)
    walk(g)
    return lines


def _fit_line_matches(f, tol=1e-9):
    def pred(g):
        ls = _line_points(g)
        if not ls:
            return False
        pts = [[to_complex(c).real for c in p.args] for p in ls[0].args]
        return len(pts) == 101 and all(abs(y - f(x)) < tol for x, y in pts)
    return Pred(pred, "fitted curve vertices")


PLOTS = [
    # least-squares line through (1..6, {1, 3, 2, 5, 4, 6}): Sxy = 31/2, Sxx = 35/2, slope 31/35, intercept 2/5
    ("ListPlot[{1, 3, 2, 5, 4, 6}, PlotFit -> \"Linear\"]",
     _fit_line_matches(lambda x: 2 / 5 + 31 / 35 * x)),
    ("ListFitPlot[{1, 3, 2, 5, 4, 6}, PlotFit -> \"Linear\"]",
     _fit_line_matches(lambda x: 2 / 5 + 31 / 35 * x)),
    ("ListLinePlot[{1, 3, 2, 5, 4, 6}, PlotFit -> \"Quadratic\"]", is_head("Graphics")),
    ("ListFitPlot[{1, 3, 2, 5, 4, 6}]", is_head("Graphics")),
    ("ListFitPlot[{1, 3, 2, 5, 4, 6}, PlotFitElements -> \"Residuals\"]", is_head("Graphics")),
    ("ListFitPlot[x02q]", "ListFitPlot[x02q]"),
    ("ListFitPlot[{1, 3, 2}, PlotFit -> \"Logistic\"]", Raises(NotImplementedError)),
    ("ListFitPlot[{1, 3, 2}, PlotFitElements -> \"Bands\"]", Raises(NotImplementedError)),
    ("ListFitPlot3D[{{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 6, 7}, {4, 5, 7, 9}}]", is_head("Graphics3D")),
    ("Count[ListFitPlot3D[{{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 6, 7}, {4, 5, 7, 9}}, PlotFit -> \"Linear\"], "
     "_Polygon, Infinity]", "256"),                                                   # 16 x 16 grid of quads
    ("ListFitPlot3D[{{0, 0, 1}, {1, 0, 2}, {0, 1, 3}, {1, 1, 4}, {2, 1, 5}}, PlotFit -> LinearModel[{1, x02a, x02b}, "
     "{x02a, x02b}]]", is_head("Graphics3D")),
    ("ListFitPlot3D[x02q]", "ListFitPlot3D[x02q]"),
    ("ListFitPlot3D[{{1, 2, 3}, {2, 3, 4}, {3, 4, 6}}]", Raises(ValueError)),       # 3 triples, LOESS too small
    ("ListFitPlot3D[{{1, 2}, {3, 4}}, PlotFitElements -> \"Bands\"]", Raises(NotImplementedError)),
    ("PlotFit", "PlotFit"),
    ("PlotFit[x02q]", "PlotFit[x02q]"),
    ("ListPlot[{1, 3, 2, 5, 4, 6}, PlotFit -> \"Foo\"]", Raises(NotImplementedError)),
    ("ListPlot3D[{{1, 2}, {3, 4}}, PlotFit -> \"Linear\"]", Raises(NotImplementedError)),
]

# ─────────────────────────── hidden Markov processes ───────────────────────────
HM = "HiddenMarkovProcess[{1/2, 1/2}, {{9/10, 1/10}, {1/5, 4/5}}, {{1/2, 1/2}, {1/10, 9/10}}]"
HM2 = "HiddenMarkovProcess[1, {{1/2, 1/2}, {1/4, 3/4}}, {{1, 0}, {1/3, 2/3}}]"
HG = "HiddenMarkovProcess[1, {{0.95, 0.05}, {0.1, 0.9}}, {NormalDistribution[0, 1], NormalDistribution[3, 1]}]"
HMM = [
    (f"LogLikelihood[{HM}, {{1, 2, 2, 1}}]", "Log[182493/5000000]"),                # brute force: 182493/5000000
    (f"LogLikelihood[{HM}, {{1., 2, 2, 1}}]", near(math.log(182493 / 5000000), 1e-12)),
    # {1, missing, 2}: state 1 emits 1 surely; p(x2) = (1/2, 1/2).m = (3/8, 5/8); P(y2 = 2) = 5/8 * 2/3 = 5/12
    (f"LogLikelihood[{HM2}, {{1, Missing[], 2}}]", "Log[5/12]"),
    (f"PDF[{HM2}[1], 2]", "1/3"),                                                     # (1/2, 1/2).em[[All, 2]]
    (f"StationaryDistribution[{HM}]", "EmpiricalDistribution[{11/30, 19/30} -> {1, 2}]"),   # pi = (2/3, 1/3)
    (f"LogLikelihood[{HG}, {{0.1, -0.5, 3.2}}]", near(-5.842131589571119, 1e-9)),
    ("HiddenMarkovProcess[x02a, {{1}}, {{1}}]", "HiddenMarkovProcess[x02a, {{1}}, {{1}}]"),
    ("HiddenMarkovProcess[1, {{1/2, 1/3}, {1, 0}}, {{1, 0}, {0, 1}}]", Raises(ValueError)),
    ("HiddenMarkovProcess[1, {{1/2, 1/2}, {1, 0}}, {{1, 0}, {0, 0}}]", Raises(NotImplementedError)),
    ("HiddenMarkovProcess[\"s\", {{1}}, {{1}}]", Raises(TypeError)),
    ("SeedRandom[5]; Length[RandomFunction[" + HM + ", {0, 20}][\"Values\"]]", "21"),
    # Check: simulated emission frequencies at t = 3 vs the slice distribution (4000 paths, within 4 sd)
    ("SeedRandom[7]; Mean[Table[RandomFunction[" + HM2 + ", {0, 3}][\"Values\"][[4]], {4000}]] - Mean[" + HM2
     + "[3]]", Pred(lambda g: abs(to_complex(g)) < 4 * math.sqrt(0.25 / 4000), "within 4 standard errors")),
    ("SeedRandom[3]; EstimatedProcess[RandomFunction[" + HG + ", {0, 400}], HiddenMarkovProcess[2, \"Gaussian\"]]"
     "[[3, All, 1]]", Pred(lambda g: sorted(round(to_complex(v).real) for v in g.args) == [0, 3],
                           "Baum-Welch recovers the means 0 and 3")),
    (f"FindHiddenMarkovStates[{{1, 2, 2, 2, 1, 1}}, {HM}]", "{1, 1, 1, 1, 1, 1}"),     # brute force (test below)
    (f"FindHiddenMarkovStates[{{2, 2, 2, 2, 2, 2}}, {HM}]", "{2, 2, 2, 2, 2, 2}"),
    (f"FindHiddenMarkovStates[{{1, 2, 2, 2, 1, 1}}, {HM}, \"PosteriorDecoding\"]", "{1, 1, 1, 1, 1, 1}"),
    (f"FindHiddenMarkovStates[{{0.1, -0.5, 3.2, 2.9, 3.5, 0.2, -1.}}, {HG}]", "{1, 1, 2, 2, 2, 1, 1}"),
    (f"FindHiddenMarkovStates[TimeSeries[{{1, 2, 2, 2, 1, 1}}], {HM}]",
     "TimeSeries[{{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}}]"),
    (f"FindHiddenMarkovStates[{{x02a, 1}}, {HM}]", f"FindHiddenMarkovStates[{{x02a, 1}}, {HM}]"),
    (f"FindHiddenMarkovStates[{{1, 3}}, {HM}]", Raises(ValueError)),
    (f"FindHiddenMarkovStates[{{1, 2}}, {HM}, \"Foo\"]", Raises(ValueError)),
    ("FindHiddenMarkovStates[{1, 2}, WienerProcess[]]", Raises(TypeError)),
    (f"FindHiddenMarkovStates[{{1, 2}}, HiddenMarkovProcess[1, {{{{1, 0}}, {{0, 1}}}}, {{{{1, 0}}, {{0, 1}}}}]]",
     Raises(ValueError)),                                                              # impossible emission path
]

# ─────────────────────────── TimeSeriesModelFit / TimeSeriesModel ───────────────────────────
# statsmodels ARIMA(order, trend="c").fit() (exact ML): AR(1) const (mean) 0.46727501, phi 0.2897406,
# sigma2 0.26542952, llf -18.93748110988186, aic 43.87496221976372, aicc 45.01781936262086, bic 47.53158969436832;
# ARMA(1,1): mean 0.44978329, phi 0.06659617, theta 0.72036397, sigma2 0.17893993; MA(2): mean 0.47658668,
# theta -0.15589208, -0.84392325, sigma2 0.1344205; AR(2) aic 17.879641457999952.  The Wolfram ARProcess
# constant is c = mean (1 - Sum[phi]).
XS = ("{0.5, 1.2, 0.3, -0.4, 0.8, 1.5, 0.9, 0.1, -0.2, 0.6, 1.1, 0.4, -0.3, 0.2, 0.9, 1.3, 0.7, 0.0, -0.5, 0.3, "
      "0.8, 1.0, 0.6, 0.2, -0.1}")
Y9 = "{1., 2., 1.5, 2.5, 1., 2.2, 1.4, 2., 1.7}"
TSM = [
    (f"TimeSeriesModelFit[{XS}, {{\"AR\", 1}}][\"BestFitParameters\"]",
     Pred(lambda g: abs(to_complex(g.args[0]) - 0.46727501 * (1 - 0.2897406)) < 1e-5 and
          abs(to_complex(g.args[1].args[0]) - 0.2897406) < 1e-5 and abs(to_complex(g.args[2]) - 0.26542952) < 1e-5,
          "statsmodels AR(1)")),
    (f"TimeSeriesModelFit[{XS}, {{\"AR\", 1}}][\"LogLikelihood\"]", near(-18.93748110988186, 1e-6)),
    (f"TimeSeriesModelFit[{XS}, {{\"AR\", 1}}][{{\"AIC\", \"AICc\", \"BIC\"}}]",
     close([43.87496221976372, 45.01781936262086, 47.53158969436832], 1e-5)),
    (f"TimeSeriesModelFit[{XS}, {{\"ARMA\", {{1, 1}}}}][\"BestFit\"][[{{2, 3, 4}}]]",
     Pred(lambda g: abs(to_complex(g.args[0].args[0]) - 0.06659617) < 1e-4 and
          abs(to_complex(g.args[1].args[0]) - 0.72036397) < 1e-4 and abs(to_complex(g.args[2]) - 0.17893993) < 1e-4,
          "statsmodels ARMA(1,1)")),
    (f"TimeSeriesModelFit[{XS}, {{\"MA\", 2}}][\"BestFit\"][[2]]", close([-0.15589208, -0.84392325], 1e-3)),
    (f"TimeSeriesModelFit[{XS}, \"AR\"][\"CandidateModelSelectionValues\"][[1]]", near(17.879641457999952, 1e-5)),
    (f"TimeSeriesModelFit[Accumulate[{XS}], {{\"ARIMA\", {{1, 1, 0}}}}][\"BestFit\"]",
     Pred(lambda g: g.head == "ARIMAProcess" and abs(to_complex(g.args[0].args[0]) - 0.61100244) < 1e-4 and
          abs(to_complex(g.args[3]) - 0.33691859) < 1e-4, "statsmodels ARIMA(1,1,0), no trend")),
    (f"TimeSeriesModelFit[{XS}][\"BestFit\"]", Pred(lambda g: g.head == "ARProcess" and len(g.args[1].args) == 2,
                                                   "AIC picks AR(2) (statsmodels aic 17.88 is the smallest)")),
    ("TimeSeriesModelFit[{x02a, x02b, x02c}]", "TimeSeriesModelFit[{x02a, x02b, x02c}]"),
    ("TimeSeriesModelFit[{1., 2.}, {\"AR\", 1}]", Raises(ValueError)),
    ("TimeSeriesModelFit[\"s\"]", Raises(TypeError)),
    (f"TimeSeriesModelFit[{XS}, \"GARCH\"]", Raises(NotImplementedError)),
    # forecasts: statsmodels r.forecast(2) = 0.30291241, 0.4196525 (times 25, 26)
    (f"TimeSeriesModelFit[{XS}, {{\"AR\", 1}}][26]", near(0.4196525, 1e-5)),
    (f"TimeSeriesModelFit[{XS}, {{\"AR\", 1}}][3]", near(-0.4)),                        # data value
    (f"TimeSeriesModelFit[{XS}, {{\"AR\", 1}}][\"ModelFamily\"]", "\"AR\""),
    (f"TimeSeriesForecast[TimeSeriesModelFit[{Y9}, {{\"AR\", 1}}], {{3}}][\"Values\"]",
     close([1.7726636, 1.71165612, 1.76287726], 1e-5)),
    (f"TimeSeriesModelFit[{Y9}, {{\"AR\", 1}}][x02t]", Pred(lambda g: g.head != "TimeSeriesModel" and
                                                          isinstance(g, Expr), "symbolic time: unevaluated")),
    # Check: residual diagnostics vs statsmodels acf / acorr_ljungbox of the same innovations
    (f"TimeSeriesModelFit[{XS}, {{\"AR\", 1}}][\"ACFValues\", \"LagMax\" -> 3][[All, 2]]",
     close([0.23973391, -0.64918262, -0.55442735], 1e-5)),
    (f"TimeSeriesModelFit[{XS}, {{\"AR\", 1}}][\"LjungBoxValues\", \"LagMax\" -> 2][[All, 2]]",
     close([0.203593, 0.000919], 1e-5)),
    (f"TimeSeriesModelFit[{XS}, {{\"AR\", 1}}][\"FitResiduals\"][\"Values\"][[{{1, 2, 3}}]]",
     close([0.03272499, 0.72324323, -0.37957519], 1e-5)),
    ("TimeSeriesModel[1]", Raises(TypeError)),
    ("TimeSeriesModel[x02q]", "TimeSeriesModel[x02q]"),
    (f"TimeSeriesModelFit[{Y9}, {{\"AR\", 1}}][\"Foo\"]", Raises(ValueError)),
    (f"TimeSeriesModelFit[{Y9}, {{\"AR\", 1}}][\"PredictionLimits\"]", Raises(NotImplementedError)),
    ("TimeSeriesModel[<|\"x02\" -> 1|>]", "TimeSeriesModel[<|\"x02\" -> 1|>]"),
]

# ─────────────────────────── AdjustTimeSeriesForecast ───────────────────────────
ATF = [
    ("AdjustTimeSeriesForecast[MAProcess[{1/10}, 1], {1, 0, 0}, {1.3}]", close([1.3, 0.03, 0.0], 1e-12)),  # doc
    # psi_1 = phi + theta = 0.3, psi_2 = phi psi_1 = 0.03; error s - 0.1 (doc example with one symbolic point)
    ("Expand[AdjustTimeSeriesForecast[ARMAProcess[{.1}, {.2}, 1], {.1, .2, .3}, {x02s}] /. x02s -> 2]",
     close([2, 0.77, 0.357], 1e-12)),
    ("AdjustTimeSeriesForecast[ARMAProcess[{.1}, {.2}, 1], TimeSeries[{.1, .2, .3}, {{5, 6, 7}}], "
     "TimeSeries[{{6, 1.}}]][\"Values\"]", close([0.1, 1.0, 0.54], 1e-12)),            # 0.3 + 0.3 (1 - 0.2)
    # Check: AR(1) identity: adjusted forecasts equal fresh forecasts from the new last value
    ("AdjustTimeSeriesForecast[ARProcess[{1/3}, 1], {x02x/3, x02x/9, x02x/27}, {x02a}]",
     "{x02a, x02a/3, x02a/9}"),
    ("AdjustTimeSeriesForecast[x02q, {1, 2}, {1}]", "AdjustTimeSeriesForecast[x02q, {1, 2}, {1}]"),
    ("AdjustTimeSeriesForecast[ARProcess[{1/3}, 1], {1, 2}, {1, 2, 3}]", Raises(ValueError)),
    ("AdjustTimeSeriesForecast[WienerProcess[], {1, 2}, {1}]", Raises(TypeError)),
    ("AdjustTimeSeriesForecast[ARProcess[{1/3}, 1], \"s\", {1}]", Raises(TypeError)),
]

# ─────────────────────────── TimeSeriesSummary / Structure / Events / EventSeriesAccumulate / window ─────────────
TS6 = "TimeSeries[{1, 3, 2, 5, 4, 6}]"
TSX = [
    # Check: TabularSummary / TabularStructure of the equivalent Tabular
    (f"TimeSeriesSummary[{TS6}] === TabularSummary[Tabular[{{{{1}}, {{3}}, {{2}}, {{5}}, {{4}}, {{6}}}}, "
     "{\"Value\"}]]", "True"),
    (f"TimeSeriesSummary[{TS6}, \"Value\" -> \"Max\"] === TabularSummary[Tabular[{{{{1}}, {{3}}, {{2}}, {{5}}, "
     "{4}, {6}}, {\"Value\"}], \"Value\" -> \"Max\"]", "True"),
    ("TimeSeriesSummary[EventSeries[{{1, 2}, {3, 5}}]] === TabularSummary[Tabular[{{2}, {5}}, {\"Value\"}]]",
     "True"),
    ("TimeSeriesSummary[x02q]", "TimeSeriesSummary[x02q]"),
    ("TimeSeriesSummary[TimeSeries[{1, 3, 2}], \"Foo\"]", Raises(ValueError)),
    ("TimeSeriesSummary[TimeSeries[{{0, {1, 2}}, {1, {3, 4}}}]]", Raises(NotImplementedError)),
    ("TimeSeriesSummary[\"s\"]", Raises(TypeError)),
    ("TimeSeriesStructure[TimeSeries[{1, 3, 2}]] === TabularStructure[Tabular[{{0, 1}, {1, 3}, {2, 2}}, "
     "{\"Timestamp\", \"Value\"}]]", "True"),
    ("TimeSeriesStructure[EventSeries[{{1, 2}, {3, 5}}]] === TabularStructure[Tabular[{{1, 2}, {3, 5}}, "
     "{\"Timestamp\", \"Value\"}]]", "True"),
    ("TimeSeriesStructure[TimeSeries[{1.5, 2.}]] === TabularStructure[Tabular[{{0, 1.5}, {1, 2.}}, "
     "{\"Timestamp\", \"Value\"}]]", "True"),
    ("TimeSeriesStructure[x02q]", "TimeSeriesStructure[x02q]"),
    ("TimeSeriesStructure[TimeSeries[{{0, {1, 2}}, {1, {3, 4}}}]]", Raises(NotImplementedError)),
    ("TimeSeriesStructure[\"s\"]", Raises(TypeError)),
    # events (by hand on the piecewise-linear path 1, 3, 2, 5, 4, 6 at t = 0..5)
    (f"TimeSeriesEvents[{TS6}, \"LocalMaximum\"]", "EventSeries[{{1, 3}, {3, 5}}]"),
    (f"TimeSeriesEvents[{TS6}, \"LocalMinimum\"]", "EventSeries[{{2, 2}, {4, 4}}]"),
    (f"TimeSeriesEvents[{TS6}, {{\"CrossingFromBelow\", 7/2}}]", "EventSeries[{{5/2, 7/2}}]"),   # 2 -> 5 at 1/2
    (f"TimeSeriesEvents[{TS6}, \"Crossing\", # - 3 &]", "EventSeries[{{7/3, 3}}]"),              # touches at t = 1
    (f"TimeSeriesEvents[{TS6}, \"LocalMaximum\", Identity, #Value > 3 &]", "EventSeries[{{3, 5}}]"),
    (f"TimeSeriesEvents[{TS6}, \"LocalExtremum\", Identity, #Timestamp > 1 &]",
     "EventSeries[{{2, 2}, {3, 5}, {4, 4}}]"),
    ("TimeSeriesEvents[TimeSeries[{1., -1., 2.}], \"Crossing\"]", "EventSeries[{{0.5, 0.}, {1.3333333333333333, 0.}}]"),
    (f"TimeSeriesEvents[{TS6}, {{\"Crossing\", 100}}]", "Missing[\"NotFound\"]"),
    ("TimeSeriesEvents[x02q, \"Crossing\"]", "TimeSeriesEvents[x02q, \"Crossing\"]"),
    (f"TimeSeriesEvents[{TS6}, \"Bogus\"]", Raises(ValueError)),
    ("TimeSeriesEvents[TimeSeries[{{0, {1, 2}}, {1, {3, 4}}}], \"Crossing\"]", Raises(NotImplementedError)),
    ("TimeSeriesEvents[\"s\", \"Crossing\"]", Raises(TypeError)),
    ("EventSeriesAccumulate[EventSeries[{{1, 2}, {3, 5}, {4, 1}}]]", "TimeSeries[{{1, 1}, {3, 2}, {4, 3}}]"),
    ("EventSeriesAccumulate[EventSeries[{{1, 2}, {3, 5}, {4, 1}}] -> \"Value\"]",
     "TimeSeries[{{1, 2}, {3, 7}, {4, 8}}]"),
    # Check: the values are Accumulate of the event values
    ("EventSeriesAccumulate[EventSeries[{{1, 2}, {3, 5}, {4, 1}, {9, 7}}] -> \"Value\"][\"Values\"] == "
     "Accumulate[{2, 5, 1, 7}]", "True"),
    ("EventSeriesAccumulate[x02q]", "EventSeriesAccumulate[x02q]"),
    ("EventSeriesAccumulate[EventSeries[{{1, 2}, {3, 5}}] -> \"foo\"]", Raises(ValueError)),
    ("EventSeriesAccumulate[{2, 5, 1}]", Raises(TypeError)),
    ("TimeSeries[{1, 2, 3}, {{1, 2, 3}}, TimeObservationWindow -> Interval[{0, 5}]][\"TimeObservationWindow\"]",
     "Interval[{0, 5}]"),
    ("TimeSeries[{1, 2, 3}, TimeObservationWindow -> {0, 4}][\"TimeObservationWindow\"]", "Interval[{0, 4}]"),
    ("EventSeries[{{1, 2}, {3, 5}}, TimeObservationWindow -> {0, 4}][\"TimeObservationWindow\"]",
     "Interval[{0, 4}]"),
    ("TimeSeries[{1, 2, 3}][\"TimeObservationWindow\"]", "Interval[{0, 2}]"),
    ("TimeObservationWindow[x02a]", "TimeObservationWindow[x02a]"),
    ("TimeSeries[{1, 2, 3}, {{1, 2, 3}}, TimeObservationWindow -> Interval[{2, 5}]]", Raises(ValueError)),
    ("TimeSeries[{1, 2, 3}, TimeObservationWindow -> {3, 1}]", Raises(ValueError)),
    ("TimeSeries[{1, 2, 3}, TimeObservationWindow -> DateInterval[{x02a, x02b}]]", Raises(NotImplementedError)),
]

# ─────────────────────────── QueueingNetworkProcess ───────────────────────────
# Open Jackson network gamma = {4, 5}, r = {{0, 1/2}, {1/4, 0}}, mu = {8, 10}: lambda = gamma (I - r)^-1 = {6, 8};
# rho = {3/4, 4/5}; M/M/1 L = rho/(1 - rho) = {3, 4}; Lq = {9/4, 16/5}; total L = 7, throughput 9, W = 7/9.
QN = "QueueingNetworkProcess[{4, 5}, {{0, 1/2}, {1/4, 0}}, {8, 10}, {1, 1}]"
QNM = "QueueingNetworkProcess[{4, 5}, {{0, 1/2}, {1/4, 0}}, {8, 10}, {2, 1}]"
# Closed network, two nodes in a cycle, mu = {2, 3}, k = 3: P(n1) ~ (1/2)^n1 (1/3)^(3 - n1) -> {8, 12, 18, 27}/65.
QC = "QueueingNetworkProcess[{0, 0}, {{0, 1}, {1, 0}}, {2, 3}, {1, 1}, 3]"
QNP = [
    (f"QueueProperties[{QN}, \"MeanSystemSize\"]", "7"),
    (f"QueueProperties[{QN}, \"MeanQueueSize\"]", "109/20"),
    (f"QueueProperties[{QN}, \"Throughput\"]", "9"),
    (f"QueueProperties[{QN}, \"MeanSystemTime\"]", "7/9"),
    (f"QueueProperties[{{{QN}, 1}}, \"MeanSystemSize\"]", "3"),
    (f"StationaryDistribution[{QN}]", "ProductDistribution[GeometricDistribution[1/4], GeometricDistribution[1/5]]"),
    (f"Mean[{QN}[Infinity]]", "{3, 4}"),
    (f"Variance[{QN}[Infinity]]", "{12, 20}"),                                      # rho/(1 - rho)^2
    # M/M/2 node 1 (lambda 6, mu 8): P0 = 5/11, Lq = P0 a^2 rho/(2 (1 - rho)^2) = 27/220 (Gross & Harris 2.3)
    (f"QueueProperties[{{{QNM}, 1}}, \"MeanQueueSize\"]", "27/220"),
    (f"QueueProperties[{{{QC}, 1}}, \"UtilizationFactor\"]", "57/65"),
    (f"QueueProperties[{{{QC}, 1}}, \"MeanSystemSize\"]", "129/65"),               # (12 + 2*18 + 3*27)/65
    (f"QueueProperties[{{{QC}, 1}}, \"Throughput\"]", "114/65"),
    # Check: node 2 populations complete node 1's (k = 3 jobs in total) and Little's law
    (f"QueueProperties[{{{QC}, 1}}, \"MeanSystemSize\"] + QueueProperties[{{{QC}, 2}}, \"MeanSystemSize\"]", "3"),
    (f"QueueProperties[{QN}, \"MeanSystemTime\"] QueueProperties[{QN}, \"Throughput\"] - "
     f"QueueProperties[{QN}, \"MeanSystemSize\"]", "0"),
    (f"QueueProperties[QueueingNetworkProcess[{{20, 5}}, {{{{0, 1/2}}, {{1/4, 0}}}}, {{8, 10}}, {{1, 1}}], "
     "\"StationarySystem\"]", "False"),
    (f"SeedRandom[2]; Head[RandomFunction[{QN}, {{0, 2}}]]", "TemporalData"),
    ("QueueingNetworkProcess[{x02a, 5}, {{0, 1/2}, {1/4, 0}}, {8, 10}, {1, 1}]",
     "QueueingNetworkProcess[{x02a, 5}, {{0, 1/2}, {1/4, 0}}, {8, 10}, {1, 1}]"),
    ("QueueingNetworkProcess[{1, 2}, {{0, 1}}, {1, 2}, {1, 1}]", Raises(ValueError)),
    ("QueueingNetworkProcess[{1, 2}, {{0, 2}, {0, 0}}, {1, 2}, {1, 1}]", Raises(ValueError)),
    ("QueueingNetworkProcess[\"s\", {{0, 1}, {0, 0}}, {1, 2}, {1, 1}]", Raises(TypeError)),
    ("QueueingNetworkProcess[{1, 2}, {{0, 1}, {0, 0}}, {1, 2}, {1, 1}, 3]", Raises(NotImplementedError)),
    (f"StationaryDistribution[{QC}]", Raises(NotImplementedError)),
    (f"QueueProperties[{{{QN}, 3}}]", Raises(ValueError)),
]

CASES = NORMALITY + PCS + URT + FD + PD + DEN + MODELS + MF + KL_ + PLOTS + HMM + TSM + ATF + TSX + QNP


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_x02(case):
    check(case)


def test_hmm_brute_force():
    """forward algorithm and Viterbi vs enumeration of all hidden-state paths"""
    p0 = [Fraction(1, 2)] * 2
    m = [[Fraction(9, 10), Fraction(1, 10)], [Fraction(1, 5), Fraction(4, 5)]]
    e = [[Fraction(1, 2), Fraction(1, 2)], [Fraction(1, 10), Fraction(9, 10)]]

    def prob(y, s):
        p = p0[s[0]] * e[s[0]][y[0] - 1]
        for t in range(1, len(y)):
            p *= m[s[t - 1]][s[t]] * e[s[t]][y[t] - 1]
        return p
    for y in ([1, 2, 2, 1], [2, 1, 1, 2, 2], [1, 1, 2]):
        tot = sum(prob(y, s) for s in itertools.product(range(2), repeat=len(y)))
        got = ev(f"Exp[LogLikelihood[{HM}, {{{', '.join(map(str, y))}}}]]")
        assert ev(f"{got} - {tot.numerator}/{tot.denominator}") == 0
        best = max(itertools.product(range(2), repeat=len(y)), key=lambda s: prob(y, s))
        vit = ev(f"FindHiddenMarkovStates[{{{', '.join(map(str, y))}}}, {HM}]")
        assert [int(v) for v in vit.args] == [b + 1 for b in best]


@pytest.mark.skipif(not HAVE_SCIPY, reason="SciPy not installed")
def test_dendrogram_matches_scipy_linkage():
    import numpy as np
    from scipy.cluster.hierarchy import linkage
    pts = [1, 2, 4.2, 8, 13, 15.5]     # no tied dissimilarities (ties may merge in either order)
    for method, name in (("average", "Average"), ("single", "Single"), ("complete", "Complete"),
                         ("weighted", "WeightedAverage")):
        g = ev(f"Dendrogram[{{{', '.join(map(str, pts))}}}, ClusterDissimilarityFunction -> \"{name}\"]")
        ref = sorted(linkage(np.array(pts)[:, None], method)[:, 2])
        assert _merge_heights(g) == pytest.approx(ref)


@pytest.mark.skipif(not HAVE_SCIPY, reason="SciPy not installed")
def test_numpy_references():
    import numpy as np
    from scipy import stats
    from scipy.cluster.hierarchy import linkage

    d = np.array([[1., 2.], [2., 1.], [3., 4.], [4., 3.], [5., 6.], [2.5, 2.], [3.5, 5.], [1.5, 1.]])
    n, p = d.shape
    z = d - d.mean(0)
    g = z @ np.linalg.inv(z.T @ z / n) @ z.T
    sk = n * (g ** 3).sum() / n ** 2 / 6
    ku = ((np.diag(g) ** 2).sum() / n - p * (p + 2)) / math.sqrt(8 * p * (p + 2) / n)
    assert abs(sk - 0.32805496700032327) < 1e-12 and abs(ku + 0.9995702768493606) < 1e-12
    assert abs(stats.chi2.sf(sk, 4) - 0.9879318876707447) < 1e-12
    assert abs(stats.chi2.sf(sk + ku * ku, 5) - 0.932108538452199) < 1e-12
    # Henze-Zirkler statistic and lognormal p-value (R MVN::hz)
    b = (1 / math.sqrt(2)) * ((2 * p + 1) * n / 4) ** (1 / (p + 4))
    si = np.linalg.inv(z.T @ z / n)
    djk = np.array([[(z[j] - z[k]) @ si @ (z[j] - z[k]) for k in range(n)] for j in range(n)])
    dj = np.diag(z @ si @ z.T)
    hz = (np.exp(-b * b / 2 * djk).sum() / n - 2 * (1 + b * b) ** (-p / 2) *
          np.exp(-b * b / (2 * (1 + b * b)) * dj).sum() + n * (1 + 2 * b * b) ** (-p / 2))
    a, w = 1 + 2 * b * b, (1 + b * b) * (1 + 3 * b * b)
    mu = 1 - a ** (-p / 2) * (1 + p * b ** 2 / a + p * (p + 2) * b ** 4 / (2 * a * a))
    s2 = (2 * (1 + 4 * b * b) ** (-p / 2) + 2 * a ** (-p) * (1 + 2 * p * b ** 4 / a ** 2 + 3 * p * (p + 2) * b ** 8 /
          (4 * a ** 4)) - 4 * w ** (-p / 2) * (1 + 3 * p * b ** 4 / (2 * w) + p * (p + 2) * b ** 8 / (2 * w * w)))
    pmu, psd = math.log(math.sqrt(mu ** 4 / (s2 + mu ** 2))), math.sqrt(math.log((s2 + mu ** 2) / mu ** 2))
    assert abs(hz - 0.24271043867207087) < 1e-12
    assert abs(stats.lognorm.sf(hz, psd, scale=math.exp(pmu)) - 0.6600846158611697) < 1e-12
    # equiprobable classes for the normal sample (k = Ceiling[2 10^(2/5)] = 6)
    zz = np.array([-1.2, 0.3, 0.5, -0.1, 2.1, 1.1, -0.7, 0.05, -1.9, 0.8])
    o = np.bincount(np.minimum((stats.norm.cdf(zz) * 6).astype(int), 5), minlength=6)
    assert abs(((o - 10 / 6) ** 2 / (10 / 6)).sum() - 0.8) < 1e-12
    # single-linkage heights
    assert list(linkage(np.array([[1.], [2.], [4.], [7.]]), "single")[:, 2]) == [1, 2, 3]
    assert list(linkage(np.array([[1.], [2.], [4.], [7.]]), "complete")[:, 2]) == [1, 3, 6]


@pytest.mark.skipif(not HAVE_SM, reason="statsmodels not installed")
@pytest.mark.filterwarnings("ignore")
def test_statsmodels_references():
    import numpy as np
    from statsmodels.nonparametric.smoothers_lowess import lowess
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import acf, adfuller

    y = [1., 2., 1.5, 2.5, 3., 2., 2.8, 3.5, 3.1, 4., 3.6, 4.2]
    r = adfuller(y, maxlag=0, regression="c", autolag=None)
    assert abs(r[0] + 1.6087784636967066) < 1e-9 and abs(r[1] - 0.479182448298474) < 1e-9
    x = [float(v) for v in XS.strip("{}").split(",")]
    fit = ARIMA(x, order=(1, 0, 0), trend="c").fit()
    assert abs(fit.llf + 18.93748110988186) < 1e-6
    assert np.allclose(acf(fit.resid, nlags=3, fft=False)[1:], [0.23973391, -0.64918262, -0.55442735], atol=1e-6)
    assert np.allclose(acorr_ljungbox(fit.resid, lags=2)["lb_pvalue"], [0.203593, 0.000919], atol=1e-6)
    X = np.arange(1, 9.)
    Y = np.array([1, 3, 2, 5, 4, 6, 8, 7.])
    assert np.allclose(lowess(Y, X, frac=0.75, it=0, delta=0, return_sorted=False)[[0, 1, 7]],
                       [1.26820243, 2.18241161, 7.72977211], atol=1e-7)
