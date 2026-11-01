"""Differential tests for Wolfram batch W18 (statistical models; modules wl_statmodels and wl_statmodels_fit).

Reference values come from

* the Wolfram documentation examples (reference.wolfram.com/language/ref/<Name>.html, the rendered outputs of
  the Basic Examples / Scope sections), cited as "WL doc";
* SciPy 1.18 / statsmodels 0.14 / scikit-learn / lifelines, run separately in a reference environment; the values
  are written into this file as literals and cited by the call that produced them (so no case needs those
  libraries at test time);
* exact values worked out by hand from the defining formulas (strings, compared exactly).

Symbol names used here start with ``w18`` so that no case collides with definitions made by other modules.
"""
import importlib.util

import pytest

from _harness import Pred, Raises, check, ids

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None


def needs_scipy(case):
    """A case whose evaluation needs SciPy: without it, expect the clear 'requires SciPy' error."""
    return case if HAVE_SCIPY else (case[0], Raises(NotImplementedError, match="requires SciPy"))


def approx(v, tol=1e-5):
    return Pred(lambda g, v=v: abs(float(g) - v) <= tol * max(1.0, abs(v)), f"~{v}")


def approx_list(vs, tol=1e-5):
    def ok(g, vs=vs):
        flat = lambda x: [y for a in x.args for y in flat(a)] if getattr(x, "head", None) == "List" else [x]  # noqa
        gv = flat(g)
        return len(gv) == len(vs) and all(abs(complex(a) - b) <= tol * max(1.0, abs(b)) for a, b in zip(gv, vs))
    return Pred(ok, f"~{vs}")


BASIC = "{6.5, 3.8, 6.6, 5.7, 6.0, 6.4, 5.3}"
MAT = "{{1., 2.}, {4., 8.}, {5., 3.}, {2., 15.}}"
V1 = "{3, -4, 1, 4, 22, 17, -2, 2, 13, -11}"
V2 = "{-20, -24, 0, 4, 24, 36, -12, -12, 56, -14}"

# hypothesis-test data (references: /workspace scipy/statsmodels runs, quoted per case)
TA = "{12.1, 9.8, 11.4, 10.2, 13.5, 8.7, 10.9, 12.8, 11.1, 9.5}"
TB = "{10.3, 8.1, 9.7, 11.9, 7.6, 9.2, 8.8, 10.5, 9.0, 8.4, 10.1}"
TB10 = "{10.3, 8.1, 9.7, 11.9, 7.6, 9.2, 8.8, 10.5, 9.0, 8.4}"
TC = "{14.2, 12.5, 15.1, 11.8, 13.3, 16.0, 12.9}"
AB = "{" + TA + ", " + TB + "}"
AB10 = "{" + TA + ", " + TB10 + "}"
ABC = "{" + TA + ", " + TB + ", " + TC + "}"
# WL doc data sets
ARM = "{6.27, 5.82, 6.13, 6.14, 5.96, 6.57, 6.03, 6.06, 6.16, 6.72}"
HT = "{6.21, 5.42, 5.75, 5.97, 5.46, 6.24, 5.62, 5.66, 6.11, 6.31}"
SLEEP = "{12, 9, 11, 21, 17, 22, 18, 11}"
PLACEBO = "{21, 16, 8, 36, 28, 20, 29, 22}"
FIRST = "{86, 71, 77, 68, 91, 72, 77, 91, 70, 71, 88, 87}"
SECOND = "{88, 77, 76, 64, 96, 72, 65, 90, 65, 80, 81, 72}"
BF1 = "{-0.509, 2.25, -0.553, 1.234, 0.527, 0.319, 1.315, 1.024, 0.212, 0.835, 0.215, 0.652, 0.705, 0.683, -0.804}"
BF2 = "{-0.714, 4.559, 1.546, 2.819, -0.814, -4.146, 0.409, -3.241, -3.987, -2.739, 0.254, -4.887, -7.827, 0.424, -0.095}"
CBF = ("{{36.05, 52.47, 56.55, 45.2, 35.25, 66.38, 40.57, 57.15, 28.34}, {42.47, 85.15, 63.20, 52.10, 66.20, 73.25, "
       "44.50, 57.17, 35.05}, {51.50, 65, 73.1, 64.4, 57.45, 76.49, 40.55, 66.47, 33.17}, {37.55, 59.3, 79.12, 58.33, "
       "70.54, 69.47, 46.48, 66.35, 36.2}}")
FRD = "{{70, 77, 76, 80, 84, 78}, {61, 75, 67, 63, 66, 68}, {82, 88, 90, 96, 92, 98}, {74, 76, 80, 76, 84, 86}}"
WAT = ("{25.1, 28.51, 15.58, 7.98, 7.87, 28.9, 28.72, 10.64, 27.33, 23.86, 6.49, 11.02, 8.66, 28.84, 31.14, 24.05, "
       "15.92, 12.66, 20.31, 17.06, 17.45, 21.94, 9.8, 7.26, 11.44, 23.78, 18.83, 13.17, 23.12, 18.06, 21.79, 26.57, "
       "22.11, 21.12, 13.24, 24.26, 18.48, 28.63, 20.51, 8.21, 21.82, 17.09, 30.67, 14.99, 28.42, 21.64, 16.75, 8.06, "
       "9.5, 30.55}")
# the doc's conv[d] := With[{r = d/2 Pi}, 2 Pi (r - IntegerPart[r])] (note d/2 Pi = d Pi/2)
WATC = "Map[With[{w18r = # Pi/2}, N[2 Pi (w18r - IntegerPart[w18r])]] &, " + WAT + "]"
UNI7 = "{0.1, 0.5, 0.7, 0.2, 0.9, 0.33, 0.41}"
MV = "{{1., 2.}, {2., 1.5}, {3., 4.}, {4., 3.}}"

# fitting data
LMD = "{{0, 1}, {1, 0}, {3, 2}, {5, 4}}"            # WL doc LinearModelFit basic example
LMDF = "{{0., 1.}, {1., 0.}, {3., 2.}, {5., 4.}}"
POIS = "{{1, 2}, {2, 3}, {3, 6}, {4, 7}, {5, 12}}"
BIN = "{{1, 0}, {2, 0}, {3, 1}, {4, 0}, {5, 1}, {6, 1}}"
GAM = "{{1, 2.1}, {2, 1.3}, {3, 0.9}, {4, 0.7}, {5, 0.6}}"
EXPD = "{{0, 1.}, {1, 2.7}, {2, 7.4}, {3, 20.1}, {4, 54.6}}"

CMP = ("{{{0.79, 0.56}, {0.71, 0.48}}, {{0.48, 0.033}, {0.66, -0.18}, {0.71, -0.021}}, "
       "{{-0.88, 0.12}, {-0.90, 0.092}}}")                     # WL doc ClusteringMeasurements example
CT5 = "{0.1, 2.2, 3.4, 5.6, 8.9}"                              # WL doc ClusteringTree scope data
AGG = 'Method -> "Agglomerate"'
CDF = "ClusterDissimilarityFunction"

# survival data (WL doc SurvivalModelFit / CoxModelFit examples)
ED4 = "EventData[{1, 2, 3, 4}, {0, 0, 1, 0}]"
D10 = "{30, 32, 26, {37, Infinity}, 29, {32, Infinity}, 24, {31, Infinity}, {31, Infinity}, 42}"
XI = ("{{67, 1}, {11, 0}, {32, 0}, {50, 1}, {65, 1}, {44, 1}, {20, 1}, {26, 0}, {69, 1}, {50, 1}, {25, 1}, {18, 0}, "
      "{63, 1}, {41, 0}, {30, 0}, {32, 1}, {46, 1}, {35, 0}, {33, 0}, {22, 0}}")
E20 = ("EventData[Automatic, {{8, 22, 6, 4, 12, 11, 2, 34, 25, 15, 8, 6, 34, 6, 32, 1, 15, 9, 9, 6}, "
       "{0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0}, None}]")
COX2 = "CoxModelFit[{" + XI + ", " + E20 + "}, {w18u, w18v}, {w18u, w18v}]"
COX1 = "CoxModelFit[{{2, 1, 3, 1}, " + ED4 + "}, w18x, w18x]"

CASES = [
    # ── BiweightLocation (WL doc examples) ──
    (f"BiweightLocation[{BASIC}]", approx(6.10018)),
    (f"BiweightLocation[{MAT}]", approx_list([3.0, 6.68769])),
    ("BiweightLocation[{1., 2., 3., 2., 1.}, 7]", approx(1.78989)),
    ("BiweightLocation[N[{1, 2, 3, 4, 9, 10}]]", approx(4.70144)),
    ("BiweightLocation[{w18a, 1, 2}]", "BiweightLocation[{w18a, 1, 2}]"),
    ('BiweightLocation[{"a", 1}]', Raises(TypeError)),
    ("BiweightLocation[{1, 2, 3}, -1]", Raises(ValueError)),
    # Check: the fixed point satisfies the estimating equation (verified in code); symmetric data -> centre
    ("BiweightLocation[{1, 2, 3, 4, 5}]", approx(3.0, 1e-12)),
    # ── BiweightMidvariance (WL doc) ──
    (f"BiweightMidvariance[{BASIC}]", approx(0.677818)),
    (f"BiweightMidvariance[{MAT}]", approx_list([2.74623, 31.2777])),
    ("BiweightMidvariance[{1, 2, 3, 2, 1}, 8]", "236294415/374151649"),
    ("BiweightMidvariance[{w18a, 1, 2}]", "BiweightMidvariance[{w18a, 1, 2}]"),
    ("BiweightMidvariance[{1, 2, 3}, 0]", Raises(ValueError)),
    ("BiweightMidvariance[w18x]", "BiweightMidvariance[w18x]"),
    ("BiweightMidvariance[3]", Raises(TypeError)),
    # ── QnDispersion (WL doc) ──
    (f"QnDispersion[{BASIC}]", approx(0.760723)),
    (f"QnDispersion[{MAT}]", approx_list([2.2724, 6.81721])),
    ("QnDispersion[{1, 2, 3, 2, 1}, 1]", "1"),
    ("QnDispersion[{1, 20, 3, 4}, 1]", "3"),   # WL doc
    ("QnDispersion[{Sqrt[2], E, Pi, Pi^2, 1, 2, 3}, 2]", "2 (-2 + E)"),   # WL doc
    ("QnDispersion[{1.2, 3.8, 4.2, -0.5, -5.2}, 3]", approx(7.8)),   # WL doc
    (f"QnDispersion[{BASIC}, Method -> \"None\"]", approx(0.760723 / 0.857, 1e-5)),
    ("QnDispersion[{1, 2, w18a}]", "QnDispersion[{1, 2, w18a}]"),
    ("QnDispersion[{1}]", Raises(ValueError)),
    ('QnDispersion[{1, 2, 3}, Method -> "Foo"]', Raises(ValueError)),
    # ── SnDispersion (WL doc) ──
    (f"SnDispersion[{BASIC}]", approx(0.85724)),
    (f"SnDispersion[{MAT}]", approx_list([2.27548, 6.82643])),
    ("SnDispersion[{1, 2, 3, 2, 1}, 1]", "1"),
    ("SnDispersion[{1, 20, 3, 4}, 1]", "3"),
    ("SnDispersion[{Sqrt[2], E, Pi, Pi^2, 1, 2, 3}, 2]", "2 (-2 + Pi)"),
    ("SnDispersion[{w18a, 2}]", "SnDispersion[{w18a, 2}]"),
    ('SnDispersion[{"x", 2}]', Raises(TypeError)),
    # ── HoeffdingD (WL doc: 211/672 for v1, v2; ties get weight 1/2) ──
    (f"HoeffdingD[{V1}, {V2}]", "211/672"),
    ("HoeffdingD[{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}]", "1"),   # strictly monotone: D = 1
    (f"HoeffdingD[Transpose[{{{V1}, {V2}}}]][[1, 2]]", "211/672"),
    (f"HoeffdingD[Transpose[{{{V1}, {V2}}}]][[1, 1]]", "1"),
    ("HoeffdingD[{1, 2, 3}, {1, 2, 3}]", Raises(ValueError)),
    ("HoeffdingD[{1, 2, 3, 4, 5}, {1, 2, 3, 4}]", Raises(ValueError)),
    ("HoeffdingD[{1, 2, 3, 4, 5, w18a}, {1, 2, 3, 4, 5, 6}]", "HoeffdingD[{1, 2, 3, 4, 5, w18a}, {1, 2, 3, 4, 5, 6}]"),
    # ── BlomqvistBeta (WL doc: 3/5) ──
    (f"BlomqvistBeta[{V1}, {V2}]", "3/5"),
    ("BlomqvistBeta[{1, 2, 3, 4}, {4, 3, 2, 1}]", "-1"),
    ("BlomqvistBeta[{{1, 4}, {2, 3}, {3, 2}, {4, 1}}]", "{{1, -1}, {-1, 1}}"),
    ("BlomqvistBeta[{1, 2}, {1, 2, 3}]", Raises(ValueError)),
    ("BlomqvistBeta[{1, 1, 1}, {1, 2, 3}]", Raises(ValueError)),
    ("BlomqvistBeta[{w18a, 1}, {1, 2}]", "BlomqvistBeta[{w18a, 1}, {1, 2}]"),
    # ── GoodmanKruskalGamma (pair counts by hand; equals KendallTau without ties) ──
    ("GoodmanKruskalGamma[{1, 2, 3, 4}, {1, 3, 2, 4}]", "2/3"),
    ("GoodmanKruskalGamma[{1, 2, 2, 3}, {1, 2, 3, 3}]", "1"),
    ("GoodmanKruskalGamma[{1, 2, 3, 4, 5}, {2, 1, 4, 3, 5}] == KendallTau[{1, 2, 3, 4, 5}, {2, 1, 4, 3, 5}]", "True"),
    ("GoodmanKruskalGamma[{{1, 1}, {2, 3}, {3, 2}, {4, 4}}]", "{{1, 2/3}, {2/3, 1}}"),
    ("GoodmanKruskalGamma[{1, 1}, {2, 2}]", Raises(ValueError)),
    ('GoodmanKruskalGamma["a", {1, 2}]', Raises(TypeError)),
    ("GoodmanKruskalGamma[{w18a, 1}, {1, 2}]", "GoodmanKruskalGamma[{w18a, 1}, {1, 2}]"),
    # ── AbsoluteCorrelation (definition Mean[v1 Conjugate[v2]]) ──
    ("AbsoluteCorrelation[{1, 2, 3}, {4, 5, 6}]", "32/3"),
    ("AbsoluteCorrelation[{I, 1}, {I, 2}]", "3/2"),
    ("AbsoluteCorrelation[{{1, 2}, {3, 4}}]", "{{5, 7}, {7, 10}}"),
    ("AbsoluteCorrelation[{1., 2.}, {3., 4.}]", approx(5.5)),
    ("AbsoluteCorrelation[{1, 2}, {1, 2, 3}]", Raises(ValueError)),
    ("AbsoluteCorrelation[{w18a, 1}, {1, 2}]", "AbsoluteCorrelation[{w18a, 1}, {1, 2}]"),
    ("AbsoluteCorrelation[NormalDistribution[]]", Raises(NotImplementedError)),
    # ── SpatialMedian ──
    ("SpatialMedian[{1, 5, 2, 9}]", "7/2"),
    ("SpatialMedian[{{0, 0}, {1, 0}, {0, 1}, {1, 1}}]", approx_list([0.5, 0.5])),
    ("SpatialMedian[{{0, 0}, {4, 0}, {0, 3}}]", approx_list([0.69578856, 0.75117611])),   # scipy.optimize.minimize (Nelder-Mead) of the total distance
    ("SpatialMedian[{{0, 0}, {10, 0}, {0, 10}, {1, 1}}]", approx_list([1.0, 1.0])),     # median at a data point
    ("SpatialMedian[{{0, 0}, {1, 1}}, DistanceFunction -> ManhattanDistance]", Raises(NotImplementedError)),
    ("SpatialMedian[w18x]", "SpatialMedian[w18x]"),
    ('SpatialMedian["abc"]', Raises(TypeError)),
    # ── CentralFeature ──
    ("CentralFeature[{1, 2, 3, 10}]", "2"),
    ("CentralFeature[{{0, 0}, {1, 0}, {5, 5}, {0, 1}}]", "{1, 0}"),   # totals 9.07, 8.81, 18.1, 9.47
    ('CentralFeature[{1 -> "a", 2 -> "b", 3 -> "c", 10 -> "d"}]', '"b"'),
    ("CentralFeature[{1, 2, 3, 10} -> Automatic]", "2"),
    ("CentralFeature[{1, 2, 3, 10}, DistanceFunction -> (Abs[#1 - #2]^2 &)]", "3"),
    ('CentralFeature[{"cat", "dog"}]', Raises(NotImplementedError)),
    ("CentralFeature[w18x]", "CentralFeature[w18x]"),
    # ── RankedMin / RankedMax (WL doc) ──
    ("RankedMin[{12, 13, 11}, 2]", "12"),
    ("RankedMin[{Pi, Sqrt[2], E, 3}, 3]", "3"),
    ("RankedMin[{12.6555, 3.5265, 25.65}, 2]", "12.6555"),
    ("RankedMin[{1, 1, 2, 5}, -1]", "5"),
    ("RankedMin[{1, 2}, 3]", Raises(ValueError)),
    ("RankedMin[{1, w18a}, 1]", "RankedMin[{1, w18a}, 1]"),
    ("RankedMin[{1, 2}, 1.5]", Raises(TypeError)),
    ("RankedMax[{12, 13, 11}, 2]", "12"),
    ("RankedMax[{Pi, Sqrt[2], E, 3}, 1]", "Pi"),
    ("RankedMax[{1, 1, 2, 5}, 3]", "1"),
    ("RankedMax[{4, 2, 7}, -1] == Min[{4, 2, 7}]", "True"),
    ("RankedMax[{1, 2}, 0]", Raises(ValueError)),
    ("RankedMax[{1, w18a}, 1]", "RankedMax[{1, w18a}, 1]"),
    ("RankedMax[{1, I}, 1]", Raises(TypeError)),
    # ── Entropy (WL doc) ──
    ("Entropy[{0, 1, 1, 4, 1, 1}]", "-2 Log[4]/3 + Log[6]"),
    ('Entropy["A quick brown fox jumps over the lazy dog"]',
     "8/41 Log[41/8] + 4/41 Log[41/4] + 6/41 Log[41/2] + 23 Log[41]/41"),
    ("N[Entropy[{w18a, w18b, w18b, w18c, w18c, w18c, w18c}] - (-1/7 Log[1/7] - 2/7 Log[2/7] - 4/7 Log[4/7])]", approx(0.0)),
    ("Entropy[2, {0, 1, 0, 1}]", "1"),
    ("N[Entropy[{1, 2, 2, 3, 3, 3}]]", approx(1.0114042647073518)),   # scipy.stats.entropy([1, 2, 3])
    ("Entropy[{}]", Raises(ValueError)),
    ("Entropy[1, {1, 2}]", Raises(ValueError)),
    ("Entropy[5]", Raises(TypeError)),
    ("Entropy[w18x]", "Entropy[w18x]"),
    # ── ErlangB / ErlangC (WL doc: ErlangB[3, 12/15] = 32/827) ──
    ("ErlangB[3, 12/15]", "32/827"),
    ("ErlangB[1, 1]", "1/2"),
    ("ErlangB[4, 12/7] == (12/7) ErlangB[3, 12/7]/(4 + (12/7) ErlangB[3, 12/7])", "True"),  # Check: recursion
    ("ErlangB[10, 5.]", approx(0.018384570336598)),   # independent recursion, B(0) = 1
    ("ErlangB[w18c, 2]", "ErlangB[w18c, 2]"),
    ("ErlangB[2.5, 1]", Raises(ValueError)),
    ("ErlangB[2, -1]", Raises(ValueError)),
    ('ErlangB["a", 1]', Raises(TypeError)),
    ("ErlangC[3, 4/5]", "32/615"),   # WL doc
    ("ErlangC[4, 12/7]", "216/1939"),   # WL doc
    ("ErlangB[4, 12/7]", "864/12925"),   # WL doc
    ("ErlangC[2, 1] == 2 ErlangB[2, 1]/(2 - 1 (1 - ErlangB[2, 1]))", "True"),   # Check: identity with ErlangB
    ("ErlangC[2, 3]", "1"),
    ("ErlangC[1, 1/2]", "1/2"),   # M/M/1: P(wait) = rho
    ("ErlangC[0, 1]", Raises(ValueError)),
    ("ErlangC[w18c, 1]", "ErlangC[w18c, 1]"),
    # ── ZTest ──
    (f'ZTest[{TA}, Automatic, 10, "TestData"]', approx_list([2.0952908873087344, 0.03614514204656228])),  # statsmodels ztest(A, value=10)
    (f'ZTest[{AB}, Automatic, 0, "TestData"]', approx_list([2.6103915881095276, 0.009043863240222078])),  # ztest(A, B, usevar="unequal")
    (f'ZTest[{TA}, 2, 10, "TestData"]', approx_list([2.23606797749979, 0.025347318677468252])),  # closed form, norm.sf
    (f'ZTest[{TA}, 2, 10, "PValue", AlternativeHypothesis -> "Greater"]', approx(0.012673659338734126)),
    (f'ZTest[{TA}, 2, 10, "ShortTestConclusion", SignificanceLevel -> 0.05]', '"Reject"'),
    ("ZTest[w18data]", "ZTest[w18data]"),
    (f"ZTest[{TA}, -1]", Raises(ValueError)),
    (f'ZTest[{MV}]', Raises(NotImplementedError)),
    ('ZTest["abc"]', Raises(TypeError)),
    (f'ZTest[{TA}, 2, 10, AlternativeHypothesis -> "Sideways"]', Raises(ValueError)),
    # ── PairedZTest (WL doc: armSpan/ht, s2 = .4^2, Greater -> 1.73854, 0.0410576) ──
    (f'PairedZTest[{{{ARM}, {HT}}}, .4^2, 0, "TestData", AlternativeHypothesis -> "Greater"]',
     approx_list([1.73854, 0.0410576])),
    (f'PairedZTest[{AB10}, "TestStatistic"]', approx(2.6434328173316954)),  # mean/sd of differences (numpy)
    # Check: equal to ZTest on the differences with the variance of the differences
    (f'PairedZTest[{AB10}, "TestStatistic"] - ZTest[{TA} - {TB10}, "TestStatistic"]', approx(0.0)),
    ("PairedZTest[w18data]", "PairedZTest[w18data]"),
    (f'PairedZTest[{AB}]', Raises(ValueError)),
    (f'PairedZTest[{MV}]', Raises(NotImplementedError)),
    # ── PairedTTest (WL doc examples) ──
    (f'PairedTTest[{{({{25, 24, 18, 22, 22, 25, 28, 22, 22, 28, 21, 24, 33, 28, 24}} - 21.7)/4.1, '
     f'({{649, 518, 300, 497, 550, 656, 700, 574, 479, 661, 476, 641, 793, 667, 520}} - 528.5)/117.2}}, 0, "TestData", '
     f'AlternativeHypothesis -> "Greater"]', approx_list([2.10163, 0.0270849])),
    (f'PairedTTest[{{{SLEEP}, {PLACEBO}}}, 0, "TestData"]', approx_list([-3.2085, 0.0148891])),
    (f'PairedTTest[{AB10}, "TestData"]', approx_list([2.6434328173316954, 0.026762749064111222])),  # scipy ttest_rel
    (f'PairedTTest[{TA}, 10, "PValue", AlternativeHypothesis -> "Greater"]', approx(0.03280887680475913)),  # ttest_1samp
    (f'PairedTTest[{TA}, "DegreesOfFreedom"]', "9"),
    ("PairedTTest[w18data]", "PairedTTest[w18data]"),
    ("PairedTTest[{1, 1, 1}]", Raises(ValueError)),
    (f"PairedTTest[{MV}]", Raises(NotImplementedError)),
    (f'PairedTTest[{TA}, "HypothesisTestData"]', Raises(NotImplementedError)),
    # ── SignTest (WL doc: sleepAid/placebo -> 2, 0.289063) ──
    (f'SignTest[{{{SLEEP}, {PLACEBO}}}, 0, "TestData"]', approx_list([2, 0.289063])),
    (f'SignTest[{TA}, 10, "TestData"]', approx_list([7, 0.34375])),  # scipy binomtest(7, 10)
    (f'SignTest[{AB10}, "TestData"]', approx_list([8, 0.109375])),  # scipy binomtest(8, 10)
    ('SignTest[{1, 2, 3, 4, 5, 6}, 0, "PValue", AlternativeHypothesis -> "Greater"]', approx(1 / 64)),  # exact
    ("SignTest[w18data]", "SignTest[w18data]"),
    ("SignTest[{2, 2}, 2]", Raises(ValueError)),
    (f"SignTest[{MV}]", Raises(NotImplementedError)),
    # ── SignedRankTest (WL doc: first/second, Greater -> 41.5, 0.238235) ──
    (f'SignedRankTest[{{{FIRST}, {SECOND}}}, 0, "TestData", AlternativeHypothesis -> "Greater"]',
     approx_list([41.5, 0.238235])),
    (f'SignedRankTest[{TA}, 10, "TestData"]', approx_list([44.5, 0.09238885661268335])),  # wilcoxon approx, corrected
    (f'SignedRankTest[{AB10}, "PValue", AlternativeHypothesis -> "Less"]', approx(0.9906642642509477)),
    ('SignedRankTest[{1.5, -0.5, 2.5, 3.5, 4.5}, "TestData"]', approx_list([14, 0.125])),  # exact: 2 P(W >= 14) = 2 * 2/32
    ("SignedRankTest[w18data]", "SignedRankTest[w18data]"),
    ("SignedRankTest[{1, 1}, 1]", Raises(ValueError)),
    ('SignedRankTest[{1, 2, 2, 3}, Method -> "Exact"]', Raises(NotImplementedError)),
    # ── MannWhitneyTest ──
    (f'MannWhitneyTest[{AB}, "TestData"]', approx_list([87, 0.02415541115231518])),  # scipy exact
    (f'MannWhitneyTest[{{{TA}, {TC}}}, "PValue", AlternativeHypothesis -> "Less"]', approx(0.002313862607980255)),
    ('MannWhitneyTest[{{1, 2}, {3, 4}}, "TestData"]', approx_list([0, 1 / 3])),  # exact: 2 * 1/6
    ("MannWhitneyTest[w18data]", "MannWhitneyTest[w18data]"),
    (f"MannWhitneyTest[{TA}, 1, 2, 3]", Raises(ValueError)),
    (f"MannWhitneyTest[{{{MV}, {MV}}}]", Raises(NotImplementedError)),
    # ── FisherRatioTest ──
    (f'FisherRatioTest[{AB}, "TestData"]', approx_list([1.4852137258776261, 0.5454030139877727])),  # scipy f tails
    (f'FisherRatioTest[{TA}, "TestData"]', approx_list([20.500000000000004, 0.030130747291967446])),  # chi2 tails
    (f'FisherRatioTest[{AB}, "DegreesOfFreedom"]', "{9, 10}"),
    ("FisherRatioTest[w18data]", "FisherRatioTest[w18data]"),
    (f"FisherRatioTest[{AB}, -2]", Raises(ValueError)),
    (f"FisherRatioTest[{MV}]", Raises(NotImplementedError)),
    # ── LeveneTest ──
    (f'LeveneTest[{ABC}, "TestData"]', approx_list([0.26008642544338145, 0.7730453348057715])),  # scipy levene mean
    (f'LeveneTest[{{{BF1}, {BF2}}}, "TestData"]', approx_list([17.304587423845504, 0.0002733218856103001])),
    (f'LeveneTest[{ABC}, "DegreesOfFreedom"]', "{2, 25}"),
    ("LeveneTest[w18data]", "LeveneTest[w18data]"),
    ("LeveneTest[{{1, 1}, {2, 2}}]", Raises(ValueError)),
    (f'LeveneTest[{AB}, AlternativeHypothesis -> "Less"]', Raises(NotImplementedError)),
    # ── BrownForsytheTest (WL doc: 12.9663, 0.00121149) ──
    (f'BrownForsytheTest[{{{BF1}, {BF2}}}, "TestData"]', approx_list([12.9663, 0.00121149])),
    (f'BrownForsytheTest[{AB}, "TestData"]', approx_list([0.3812173326589111, 0.5442819632153038])),  # scipy median
    # Check: a ratio r0 is the same as rescaling data1 by 1/Sqrt[r0]
    (f'BrownForsytheTest[{{{BF1}, {BF2}}}, 4, "PValue"] - BrownForsytheTest[{{{BF1}/2, {BF2}}}, "PValue"]', approx(0.0)),
    ("BrownForsytheTest[w18data]", "BrownForsytheTest[w18data]"),
    (f"BrownForsytheTest[{ABC}, 2]", Raises(ValueError)),
    (f"BrownForsytheTest[{{{MV}, {MV}}}]", Raises(NotImplementedError)),
    # ── ConoverTest ──
    (f'ConoverTest[{AB}, "TestData"]', approx_list([0.6919412634015294, 0.48897421601994806])),  # independent numpy
    (f'ConoverTest[{ABC}, "DegreesOfFreedom"]', "2"),
    (f'ConoverTest[{AB}, "PValue", AlternativeHypothesis -> "Greater"]', approx(0.24448710800997403)),
    ("ConoverTest[w18data]", "ConoverTest[w18data]"),
    ("ConoverTest[{{1, 1}, {2, 2}}]", Raises(ValueError)),
    (f"ConoverTest[{TA}]", Raises(TypeError)),
    (f"ConoverTest[{{{MV}, {MV}}}]", Raises(NotImplementedError)),
    # ── SiegelTukeyTest ──
    (f'SiegelTukeyTest[{AB}, "TestData"]', approx_list([-0.49292515315367125, 0.6220654737597032])),  # numpy impl
    (f'SiegelTukeyTest[{AB}, "PValue", AlternativeHypothesis -> "Less"]', approx(0.3110327368798516)),
    (f'SiegelTukeyTest[{{{TB}, {TA}}}, "TestStatistic"]', Pred(lambda g: float(g) > 0, "positive")),
    ("SiegelTukeyTest[w18data]", "SiegelTukeyTest[w18data]"),
    (f"SiegelTukeyTest[{ABC}]", Raises(ValueError)),
    (f"SiegelTukeyTest[{{{MV}, {MV}}}]", Raises(NotImplementedError)),
    # ── VarianceEquivalenceTest ──
    (f'VarianceEquivalenceTest[{ABC}, "Bartlett"]', approx(0.814034826293456)),  # scipy bartlett
    (f'VarianceEquivalenceTest[{ABC}, {{"PValue", {{"Levene", "Bartlett"}}}}]',
     approx_list([0.7730453348057715, 0.814034826293456])),
    (f'VarianceEquivalenceTest[{ABC}, {{"TestStatistic", "Bartlett"}}]', approx(0.4115042595010344)),
    (f'VarianceEquivalenceTest[{AB}, "FisherRatio"] - FisherRatioTest[{AB}]', approx(0.0)),  # Check: same test
    ("VarianceEquivalenceTest[w18data]", "VarianceEquivalenceTest[w18data]"),
    (f"VarianceEquivalenceTest[{ABC}]", Raises(NotImplementedError)),
    (f'VarianceEquivalenceTest[{ABC}, "Nonsense"]', Raises(ValueError)),
    (f'VarianceEquivalenceTest[{ABC}, 5]', Raises(TypeError)),
    # ── LocationEquivalenceTest (WL doc: CompleteBlockF 5.9168/0.00359111, FriedmanRank 56.0345/2.21282e-8) ──
    (f'LocationEquivalenceTest[{CBF}, {{"TestData", "CompleteBlockF"}}]', approx_list([5.9168, 0.00359111])),
    (f'LocationEquivalenceTest[{FRD}, {{"TestData", "FriedmanRank"}}]', approx_list([56.0345, 2.21282e-8])),
    (f'LocationEquivalenceTest[{ABC}, {{"TestStatistic", "KruskalWallis"}}]', approx(15.570539312903833)),  # kruskal
    (f'LocationEquivalenceTest[{ABC}, {{"TestData", "KSampleT"}}]',
     approx_list([19.80918162038847, 6.995208958518676e-06])),  # scipy f_oneway
    ("LocationEquivalenceTest[w18data]", "LocationEquivalenceTest[w18data]"),
    (f'LocationEquivalenceTest[{ABC}, All]', Raises(NotImplementedError)),
    (f'LocationEquivalenceTest[{ABC}, "CompleteBlockF"]', Raises(ValueError)),
    # ── correlation / independence tests ──
    (f'PearsonCorrelationTest[{TA}, {TB10}, "TestData"]', approx_list([0.007457084610434455, 0.9836885344846332])),
    (f'PearsonCorrelationTest[{TA}, 2 {TA} + 1, "TestStatistic"]', approx(1.0)),
    (f'PearsonCorrelationTest[{TA}, {TB10}, "DegreesOfFreedom"]', "8"),
    ("PearsonCorrelationTest[w18v, {1, 2, 3}]", "PearsonCorrelationTest[w18v, {1, 2, 3}]"),
    ("PearsonCorrelationTest[{1, 2, 3}, {1, 2}]", Raises(ValueError)),
    (f"PearsonCorrelationTest[{MV}, {MV}]", Raises(NotImplementedError)),
    ('PearsonCorrelationTest[{1, 2, 3}, {"a", "b", "c"}]', Raises(TypeError)),
    (f'SpearmanRankTest[{TA}, {TB10}, "TestData"]', approx_list([0.11515151515151514, 0.7514196523258485])),
    (f'SpearmanRankTest[{TA}, Exp[{TA}], "TestStatistic"]', approx(1.0)),
    (f'SpearmanRankTest[{TA}, {TB10}, "PValue", AlternativeHypothesis -> "Greater"]', approx(0.7514196523258485 / 2)),
    ("SpearmanRankTest[w18v, {1, 2, 3}]", "SpearmanRankTest[w18v, {1, 2, 3}]"),
    ("SpearmanRankTest[{1, 1, 1}, {1, 2, 3}]", Raises(ValueError)),
    (f"SpearmanRankTest[{MV}, {MV}]", Raises(NotImplementedError)),
    (f'KendallTauTest[{TA}, {TB10}, "TestData"]', approx_list([0.15555555555555553, 0.5312499859948482])),
    (f'KendallTauTest[{TA}, -{TA}, "TestStatistic"]', approx(-1.0)),
    (f'KendallTauTest[{TA}, {TB10}, "PValue"] - IndependenceTest[{TA}, {TB10}, "KendallTau"]', approx(0.0)),
    ("KendallTauTest[w18v, {1, 2, 3}]", "KendallTauTest[w18v, {1, 2, 3}]"),
    ("KendallTauTest[{1, 1, 1}, {1, 2, 3}]", Raises(ValueError)),
    (f"KendallTauTest[{MV}, {MV}]", Raises(NotImplementedError)),
    # GoodmanKruskalGammaTest: (|C - D| - 1)/Sqrt[n (n - 1)(2 n + 5)/18]; C - D = 7, n = 10 -> 2 norm.sf(6/Sqrt[125])
    (f'GoodmanKruskalGammaTest[{TA}, {TB10}, "TestData"]', approx_list([0.15555555555555556, 0.5915050369949165])),
    (f'GoodmanKruskalGammaTest[{TA}, {TA}, "TestStatistic"]', approx(1.0)),
    (f'GoodmanKruskalGammaTest[{TA}, {TB10}, "TestStatistic"] - GoodmanKruskalGamma[{TA}, {TB10}]', approx(0.0)),
    ("GoodmanKruskalGammaTest[w18v, {1, 2, 3}]", "GoodmanKruskalGammaTest[w18v, {1, 2, 3}]"),
    ("GoodmanKruskalGammaTest[{1, 1, 2}, {1, 2, 3}]", Raises(NotImplementedError)),
    ("GoodmanKruskalGammaTest[{1, 2}, {1, 2, 3}]", Raises(ValueError)),
    # BlomqvistBetaTest: table {{3, 2}, {2, 3}} -> 2 Min[scipy fisher_exact greater 0.5, less 0.8968] = 1
    (f'BlomqvistBetaTest[{TA}, {TB10}, "TestData"]', approx_list([0.2, 1.0])),
    (f'BlomqvistBetaTest[{TA}, {TB10}, "PValue", AlternativeHypothesis -> "Greater"]', approx(0.5)),
    ('BlomqvistBetaTest[{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}, "TestData"]',
     approx_list([1.0, 2 / 70])),   # hypergeometric: 2/C(8, 4)
    ("BlomqvistBetaTest[w18v, {1, 2, 3}]", "BlomqvistBetaTest[w18v, {1, 2, 3}]"),
    ("BlomqvistBetaTest[{1, 2}, {1, 2, 3}]", Raises(ValueError)),
    (f"BlomqvistBetaTest[{MV}, {MV}]", Raises(NotImplementedError)),
    (f'HoeffdingDTest[{TA}, {TB10}, "TestStatistic"] - HoeffdingD[{TA}, {TB10}]', approx(0.0)),
    ('HoeffdingDTest[{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, "TestStatistic"]', approx(1.0)),
    ('HoeffdingDTest[{1, 2, 3, 4, 5}, {2, 1, 4, 3, 5}, "TestStatistic"]', Pred(lambda g: float(g) < 1, "<1")),
    ("HoeffdingDTest[w18v, {1, 2, 3}]", "HoeffdingDTest[w18v, {1, 2, 3}]"),
    (f"HoeffdingDTest[{TA}, {TB10}]", Raises(NotImplementedError)),
    ('HoeffdingDTest[{1, 2}, {1, 2, 3}, "TestStatistic"]', Raises(ValueError)),
    (f'IndependenceTest[{TA}, {TB10}, {{"PearsonCorrelation", "KendallTau"}}]',
     approx_list([0.9836885344846332, 0.5312499859948482])),
    (f'IndependenceTest[{TA}, {TB10}, {{"TestData", "SpearmanRank"}}]',
     approx_list([0.11515151515151514, 0.7514196523258485])),
    (f'IndependenceTest[{TA}, {TB10}, "BlomqvistBeta"]', approx(1.0)),
    ("IndependenceTest[w18v, {1, 2, 3}]", "IndependenceTest[w18v, {1, 2, 3}]"),
    (f"IndependenceTest[{TA}, {TB10}]", Raises(NotImplementedError)),
    (f'IndependenceTest[{TA}, {TB10}, "WilksW"]', Raises(NotImplementedError)),
    (f'IndependenceTest[{TA}, {TB10}, "Bogus"]', Raises(ValueError)),
    (f'CorrelationTest[Transpose[{{{TA}, {TB10}}}], 0, {{"TestData", "SpearmanRank"}}]',
     approx_list([0.11515151515151514, 0.7514196523258485])),
    (f'CorrelationTest[Transpose[{{{TA}, {TB10}}}], 0, "PearsonCorrelation"]', approx(0.9836885344846332)),
    (f'CorrelationTest[Transpose[{{{TA}, {TB10}}}], 0, "PearsonCorrelation"] - '
     f'PearsonCorrelationTest[{TA}, {TB10}]', approx(0.0)),
    ("CorrelationTest[w18m, 0, \"PearsonCorrelation\"]", "CorrelationTest[w18m, 0, \"PearsonCorrelation\"]"),
    (f'CorrelationTest[Transpose[{{{TA}, {TB10}}}], 0.5, "PearsonCorrelation"]', Raises(NotImplementedError)),
    ('CorrelationTest[{{1, 2, 3}, {4, 5, 6}, {7, 8, 0}}, 0, "PearsonCorrelation"]', Raises(ValueError)),
    # ── goodness of fit ──
    (f'ShapiroWilkTest[{TA}, "TestData"]', approx_list([0.9824564696380026, 0.9769481138115588])),  # scipy shapiro
    (f'ShapiroWilkTest[{TB}, "TestData"]', approx_list([0.9756730463888568, 0.9373681084274752])),  # scipy shapiro
    # Check: n = 3 has the exact null distribution 6/Pi (ArcSin[Sqrt[W]] - ArcSin[Sqrt[3/4]])
    ('ShapiroWilkTest[{1., 2., 4.}, "TestData"]', approx_list([0.9642857142857143, 0.6368868450289702])),
    ("ShapiroWilkTest[w18data]", "ShapiroWilkTest[w18data]"),
    ("ShapiroWilkTest[{1, 1, 1, 1}]", Raises(ValueError)),
    (f"ShapiroWilkTest[{MV}]", Raises(NotImplementedError)),
    ('ShapiroWilkTest[{"a", "b", "c"}]', Raises(TypeError)),
    (f'JarqueBeraALMTest[{TA}, "TestData"]', approx_list([0.3619000704474695, 0.8344770521133871])),  # numpy/scipy
    (f'JarqueBeraALMTest[{TA}, "DegreesOfFreedom"]', "2"),
    (f'JarqueBeraALMTest[{TA}, "PValue"] - Exp[-JarqueBeraALMTest[{TA}, "TestStatistic"]/2]', approx(0.0)),  # Check
    ("JarqueBeraALMTest[w18data]", "JarqueBeraALMTest[w18data]"),
    ("JarqueBeraALMTest[{2, 2, 2, 2}]", Raises(ValueError)),
    (f"JarqueBeraALMTest[{MV}]", Raises(NotImplementedError)),
    (f'AndersonDarlingTest[{TA}, NormalDistribution[11, 1.5], "TestData"]',
     approx_list([0.12748312585305044, 0.9998340174086877], 1e-4)),
    (f'AndersonDarlingTest[{UNI7}, UniformDistribution[], "TestStatistic"]', approx(0.2784381386483643)),
    # Check: the 5% asymptotic critical value 2.492 (Marsaglia & Marsaglia, large n)
    ("AndersonDarlingTest[Table[N[(2 w18i - 1)/400], {w18i, 200}], UniformDistribution[], \"TestStatistic\"]",
     Pred(lambda g: abs(float(g)) < 0.01, "~0")),
    ("AndersonDarlingTest[w18data, NormalDistribution[]]", "AndersonDarlingTest[w18data, NormalDistribution[]]"),
    (f"AndersonDarlingTest[{TA}]", Raises(NotImplementedError)),
    (f"AndersonDarlingTest[{TA}, NormalDistribution[w18mu, 1]]", Raises(NotImplementedError)),
    (f'AndersonDarlingTest[{TA}, NormalDistribution[], Method -> "MonteCarlo"]', Raises(NotImplementedError)),
    (f"AndersonDarlingTest[{TA}, NormalDistribution[], Bogus -> 1]", Raises(ValueError)),
    (f'CramerVonMisesTest[{UNI7}, UniformDistribution[], "TestData"]',
     approx_list([0.050333333333333355, 0.8895337333354527])),  # scipy cramervonmises
    (f'CramerVonMisesTest[{TA}, NormalDistribution[11, 1.5], "TestData"]',
     approx_list([0.017512123397014218, 0.9995389719743557])),  # scipy cramervonmises
    (f'CramerVonMisesTest[{TA}, NormalDistribution[11, 1.5]] - DistributionFitTest[{TA}, NormalDistribution[11, 1.5], '
     f'"CramerVonMises"]', approx(0.0)),
    ("CramerVonMisesTest[w18data, NormalDistribution[]]", "CramerVonMisesTest[w18data, NormalDistribution[]]"),
    (f"CramerVonMisesTest[{TA}, PoissonDistribution[3]]", Raises(NotImplementedError)),
    # WL doc: Watson U^2 = 0.0350114, p = 0.893843 (asymptotic approximation agrees to 1e-3)
    (f'WatsonUSquareTest[{WATC}, UniformDistribution[{{0, 2 Pi}}], "TestData"]',
     approx_list([0.0350114, 0.893843], 2e-3)),
    (f'WatsonUSquareTest[{UNI7}, UniformDistribution[], "TestStatistic"]', approx(0.031819047619047636)),
    # Check: U^2 = W^2 - n (Mean[u] - 1/2)^2
    (f'WatsonUSquareTest[{UNI7}, UniformDistribution[], "TestStatistic"] - (CramerVonMisesTest[{UNI7}, '
     f'UniformDistribution[], "TestStatistic"] - 7 (Mean[{UNI7}] - 1/2)^2)', approx(0.0)),
    ("WatsonUSquareTest[w18data, NormalDistribution[]]", "WatsonUSquareTest[w18data, NormalDistribution[]]"),
    (f"WatsonUSquareTest[{TA}, Automatic]", Raises(NotImplementedError)),
    (f'KuiperTest[{UNI7}, UniformDistribution[], "TestStatistic"]', approx(0.3142857142857143)),  # D+ + D-
    (f'KuiperTest[{TA}, NormalDistribution[11, 1.5], "TestStatistic"]', approx(0.1880289011744407)),
    (f'KuiperTest[{UNI7}, UniformDistribution[], "PValue"]', Pred(lambda g: 0.8 < float(g) <= 1, "large")),
    ("KuiperTest[w18data, NormalDistribution[]]", "KuiperTest[w18data, NormalDistribution[]]"),
    (f"KuiperTest[{{{TA}, {TB}}}, NormalDistribution[]]", Raises(NotImplementedError)),
    (f'DistributionFitTest[{TA}, Automatic, "ShapiroWilk"]', approx(0.9769481138115588)),
    (f'DistributionFitTest[{TA}, NormalDistribution[11, 1.5], {{"TestData", "AndersonDarling"}}]',
     approx_list([0.12748312585305044, 0.9998340174086877], 1e-4)),
    (f'DistributionFitTest[{TA}, NormalDistribution[11, 1.5], {{"PValue", {{"CramerVonMises", "Kuiper"}}}}]',
     Pred(lambda g: len(g.args) == 2, "two p-values")),
    ("DistributionFitTest[w18data, NormalDistribution[], \"AndersonDarling\"]",
     "DistributionFitTest[w18data, NormalDistribution[], \"AndersonDarling\"]"),
    (f"DistributionFitTest[{TA}]", Raises(NotImplementedError)),
    (f'DistributionFitTest[{TA}, PoissonDistribution[3], "PearsonChiSquare"]', Raises(NotImplementedError)),  # X02: continuous dists now supported
    (f'DistributionFitTest[{TA}, NormalDistribution[], "Bogus"]', Raises(ValueError)),
    # ── Around family (first-order propagation; hand derivatives) ──
    ("Around[1, 1/10] + Around[2, 1/5]", "Around[3, Sqrt[5]/10]"),
    ("Around[1, 0.1] + Around[2, 0.2]", Pred(lambda g: g.head == "Around" and abs(float(g.args[1]) - 0.223606797749979) < 1e-12, "Around[3., 0.2236]")),
    ("3 Around[2, 1/10]", "Around[6, 3/10]"),
    ("Around[1, 1/10] Around[2, 1/5]", "Around[2, Sqrt[2]/5]"),   # Check: d(xy) = Sqrt[(y dx)^2 + (x dy)^2]
    ("Around[2., 0.1]^2", Pred(lambda g: abs(float(g.args[1]) - 0.4) < 1e-12, "Around[4., 0.4]")),
    ("Exp[Around[0, 1/10]]", "Around[1, 1/10]"),
    ("Around[Interval[{1, 3}]]", "Around[2, 1]"),
    ("Around[5, Scaled[1/10]]", "Around[5, 1/2]"),
    ("Around[{1, 2, 3, 4}]", "Around[5/2, Sqrt[5/3]]"),   # Mean, StandardDeviation
    ("Around[1, {1/10, 1/5}] + 1", "Around[2, {1/10, 1/5}]"),
    ('Around[2, 1/10]["Interval"]', "Interval[{19/10, 21/10}]"),
    ('Around[2, 1/10]["Value"]', "2"),
    ('Around[2, 1/10]["Uncertainty"]', "1/10"),
    ("Around[w18x, 1]", "Around[w18x, 1]"),
    ("Around[1, -1]", Raises(ValueError)),
    ('Around["a", 1]', Raises(TypeError)),
    ("Around[1 + I, 1]", Raises(NotImplementedError)),
    ("Around[1, I]", Raises(ValueError)),
    ('Around[2, 1/10]["Bogus"]', Raises(ValueError)),
    ("AroundReplace[w18x w18y, {w18x -> Around[2, 1/10], w18y -> Around[3, 1/5]}]", "Around[6, 1/2]"),
    ("AroundReplace[w18x^2 + w18y, {w18x -> Around[2., 0.1], w18y -> Around[1., 0.2]}]",
     Pred(lambda g: abs(float(g.args[0]) - 5) < 1e-12 and abs(float(g.args[1]) - 0.2 * 5 ** 0.5) < 1e-12, "Around[5., Sqrt[0.4^2 + 0.2^2]]")),
    # Check: identical to Around arithmetic when each variable occurs once
    ("AroundReplace[w18x + w18y, {w18x -> Around[1, 1/10], w18y -> Around[2, 1/5]}] == Around[1, 1/10] + Around[2, 1/5]", "True"),
    ("AroundReplace[w18x + w18z, w18x -> Around[1, 1/10]]", "AroundReplace[w18x + w18z, w18x -> Around[1, 1/10]]"),
    ("AroundReplace[w18x, {w18x -> Around[1, {1/10, 1/5}]}]", Raises(NotImplementedError)),
    ("AroundReplace[w18x, 5]", Raises(TypeError)),
    ("MeanAround[{1.2, 2.3, 1.9, 2.8, 2.1}]",
     Pred(lambda g: abs(float(g.args[0]) - 2.06) < 1e-12 and abs(float(g.args[1]) - 0.26191601707417583) < 1e-12, "numpy mean/sem")),
    ("MeanAround[{Around[1, 1/10], Around[2, 1/10]}]", "Around[3/2, Sqrt[2]/20]"),   # weighted mean
    ("MeanAround[{1, 2, 3}]", "Around[2, 1/Sqrt[3]]"),
    ("MeanAround[w18x]", "MeanAround[w18x]"),
    ("MeanAround[{Around[1, 0], Around[2, 1]}]", Raises(ArithmeticError)),
    ("MeanAround[{1}]", Raises(ValueError)),
    ("MeanAround[{{1, 2}, {3, 4}}]", Raises(NotImplementedError)),
    ("MeanAround[5]", Raises(TypeError)),
    ('VectorAround[{1, 2}, {1/10, 1/5}]["Covariance"]', "{{1/100, 0}, {0, 1/25}}"),   # Check: DiagonalMatrix[d^2]
    ('VectorAround[{1, 2}, {{1, 0}, {0, 4}}]["Uncertainty"]', "{1, 2}"),
    ('VectorAround[{1, 2}, {1/10, 1/5}]["Value"]', "{1, 2}"),
    ("VectorAround[w18v, {1, 2}]", "VectorAround[w18v, {1, 2}]"),
    ("VectorAround[{1, 2}, {1/10}]", Raises(ValueError)),
    ("VectorAround[{1, 2}, {{1, 2}, {3, 4}}]", Raises(ValueError)),
    ("VectorAround[{{1, 2}, {3, 4}}]", Raises(NotImplementedError)),
    ('VectorAround[{"a"}, {1}]', Raises(TypeError)),
    ("CenteredInterval[1, 1/10] CenteredInterval[2, 1/5]", "CenteredInterval[2, 21/50]"),
    ("CenteredInterval[1, 1/2] + 3", "CenteredInterval[4, 1/2]"),
    ("CenteredInterval[Interval[{1, 3}]]", "CenteredInterval[2, 1]"),
    ('CenteredInterval[1, 1/10]["Bounds"]', "{9/10, 11/10}"),
    # Check: the product enclosure contains every product of endpoints (9/10 * 9/5 and 11/10 * 11/5)
    ('(CenteredInterval[1, 1/10] CenteredInterval[2, 1/5])["Bounds"][[1]] <= 81/50 && 121/50 <= '
     '(CenteredInterval[1, 1/10] CenteredInterval[2, 1/5])["Bounds"][[2]]', "True"),
    ("CenteredInterval[1, 1/10]^2", "CenteredInterval[1, 21/100]"),
    ("CenteredInterval[w18c, 1]", "CenteredInterval[w18c, 1]"),
    ("CenteredInterval[1, -1]", Raises(ValueError)),
    ("CenteredInterval[1.5, 0.1]", Raises(NotImplementedError)),
    ('CenteredInterval["a", 1]', Raises(TypeError)),
    # ── Likelihood / LogLikelihood ──
    ("Likelihood[NormalDistribution[0, 1], {0.5, -1., 2.}]", approx(0.004599463560297635)),  # prod norm.pdf
    ("Likelihood[PoissonDistribution[2], {1, 3, 0}]", "8/(3 E^6)"),
    ("N[Likelihood[PoissonDistribution[2], {1, 3, 0}]]", approx(0.006610005804443622)),  # prod poisson.pmf
    ("Likelihood[w18d, {1, 2}]", "Likelihood[w18d, {1, 2}]"),
    ("Likelihood[NormalDistribution[0, 1], 3]", Raises(TypeError)),
    ("Likelihood[NormalDistribution[0, 1], {{1, 2}}]", Raises(NotImplementedError)),
    ("Likelihood[5, {1, 2}]", Raises(ValueError)),
    ("LogLikelihood[NormalDistribution[1, 2], {0.5, -1., 2.}]", approx(-5.492507141293855)),  # sum norm.logpdf
    ("LogLikelihood[ExponentialDistribution[2], {1, 3}]", "2 Log[2] - 8"),
    ("LogLikelihood[ExponentialDistribution[2], {1, 3}] - Log[Likelihood[ExponentialDistribution[2], {1, 3}]]", "0"),  # Check
    ("LogLikelihood[ExponentialDistribution[2], {-1, 3}]", "-Infinity"),
    ("LogLikelihood[w18d, {1, 2}]", "LogLikelihood[w18d, {1, 2}]"),
    ("LogLikelihood[NormalDistribution[0, 1], 3]", Raises(TypeError)),
    ("LogLikelihood[NormalDistribution[0, 1], {{1, 2}}]", Raises(NotImplementedError)),
    # ── FactorialMoment (WL doc) ──
    ("FactorialMoment[{1, 2, 3, 4}, 1]", "5/2"),
    ("FactorialMoment[{1., 2., 3., 4.}, 2]", approx(5.0)),
    ("FactorialMoment[BinomialDistribution[w18n, w18p], 2]", "(w18n - 1) w18n w18p^2"),
    ("FactorialMoment[PoissonDistribution[w18m], 3]", "w18m^3"),   # Check: Poisson factorial moments m^r
    ("FactorialMoment[{Pi, E, 2}, 2] - (2 - E + E^2 - Pi + Pi^2)/3", "0"),
    ("FactorialMoment[3]", "FactorialMoment[3]"),
    ("FactorialMoment[{1, 2}, -1]", Raises(ValueError)),
    ("FactorialMoment[{1, 2}, {1, 2}]", Raises(NotImplementedError)),
    ("FactorialMoment[5, 2]", Raises(TypeError)),
    # ── PowerSymmetricPolynomial / AugmentedSymmetricPolynomial (WL doc: on {a, b, c}) ──
    ("PowerSymmetricPolynomial[2, {1, 2, 3}]", "14"),
    ("PowerSymmetricPolynomial[2, {w18a, w18b, w18c}]", "w18a^2 + w18b^2 + w18c^2"),
    ("PowerSymmetricPolynomial[1, {3, 5, 9}] == Total[{3, 5, 9}]", "True"),   # Check
    ("PowerSymmetricPolynomial[{1, 2}, {1, 2}]", "{3, 5}"),
    ("PowerSymmetricPolynomial[2]", "PowerSymmetricPolynomial[2]"),
    ("PowerSymmetricPolynomial[2, 5]", Raises(TypeError)),
    ("AugmentedSymmetricPolynomial[{2, 1}, {w18a, w18b, w18c}]",
     "w18a^2 w18b + w18a w18b^2 + w18a^2 w18c + w18b^2 w18c + w18a w18c^2 + w18b w18c^2"),
    ("AugmentedSymmetricPolynomial[{1, 2}, {1, 2, 3}]", "48"),   # brute force: sum over i != j of x_i x_j^2
    ("AugmentedSymmetricPolynomial[{1, 1, 1}, {1, 2, 3, 4}]", "300"),   # 3! e3 = 6 * 50
    ("AugmentedSymmetricPolynomial[{3}, {1, 2}] == PowerSymmetricPolynomial[3, {1, 2}]", "True"),
    ("AugmentedSymmetricPolynomial[{2, 1}]", "AugmentedSymmetricPolynomial[{2, 1}]"),
    ("AugmentedSymmetricPolynomial[{0, 1}, {1, 2}]", Raises(ValueError)),
    ("AugmentedSymmetricPolynomial[2, {1, 2}]", Raises(TypeError)),
    # ── DistributionParameterAssumptions / DistributionParameterQ (WL doc) ──
    ("DistributionParameterAssumptions[NormalDistribution[w18m, w18s]]", "Element[w18m, Reals] && w18s > 0"),
    ("DistributionParameterAssumptions[GammaDistribution[w18a, 2]]", "w18a > 0"),
    ("DistributionParameterAssumptions[GammaDistribution[1, 2]]", "True"),
    ("DistributionParameterAssumptions[GeometricDistribution[w18p]]", "0 < w18p <= 1"),
    ("DistributionParameterAssumptions[BinormalDistribution[w18r]]", "-1 < w18r < 1"),
    ("DistributionParameterAssumptions[MultinomialDistribution[w18n, {w18p1, w18p2}]]",
     "Element[w18n, Integers] && w18n > 0 && w18p1 >= 0 && w18p2 >= 0 && w18p1 + w18p2 == 1"),
    ("DistributionParameterAssumptions[w18d]", "DistributionParameterAssumptions[w18d]"),
    ("DistributionParameterAssumptions[ZipfDistribution[w18r]]", Raises(NotImplementedError)),
    ("DistributionParameterAssumptions[5]", Raises(TypeError)),
    ("DistributionParameterQ[NormalDistribution[w18m, w18s]]", "True"),
    ("DistributionParameterQ[NormalDistribution[0, 1]]", "True"),
    ("DistributionParameterQ[5]", "False"),
    ("DistributionParameterQ[w18d]", "False"),
    # ── RarerProbability ──
    ("RarerProbability[NormalDistribution[0, 1], 1.96]", approx(0.04999579029644087)),   # 2 norm.cdf(-1.96)
    ("RarerProbability[StudentTDistribution[5], 2.]", approx(0.10193947882985835)),   # 2 t.cdf(-2, 5)
    ("RarerProbability[LaplaceDistribution[0, 1], 1.5]", approx(0.22313016014842982)),
    ("RarerProbability[BinomialDistribution[4, 1/2], {0, 1, 2}]", "{1/8, 5/8, 1}"),   # Check: direct pmf sums
    ("RarerProbability[NormalDistribution[w18m, 1], 1]", "RarerProbability[NormalDistribution[w18m, 1], 1]"),
    ("RarerProbability[GammaDistribution[2, 1], 1]", Raises(NotImplementedError)),
    ('RarerProbability[NormalDistribution[0, 1], "a"]', Raises(TypeError)),
    # ── Conditioned (memorylessness) ──
    ("Probability[Conditioned[w18x > 3, w18x > 1], Distributed[w18x, ExponentialDistribution[2]]]", "E^-4"),
    ("Expectation[Conditioned[w18x, w18x > 1], Distributed[w18x, ExponentialDistribution[2]]]", "3/2"),
    ("Probability[Conditioned[w18x > 3, w18x > 1], Distributed[w18x, ExponentialDistribution[2]]] == "
     "Probability[w18x > 2, Distributed[w18x, ExponentialDistribution[2]]]", "True"),   # Check: memoryless
    ("Conditioned[w18a, w18b]", "Conditioned[w18a, w18b]"),
    ("Conditioned[w18a]", Raises(ValueError)),
    ("Probability[Conditioned[w18x > 3, w18x > 5], Distributed[w18x, UniformDistribution[{0, 1}]]]", Raises(ArithmeticError)),
    ("Expectation[Conditioned[w18x, w18x > 1], Distributed[w18x, PoissonDistribution[2]]]", Raises(NotImplementedError)),
    # ── BirnbaumImportance ──
    ("BirnbaumImportance[ReliabilityDistribution[w18x && w18y, {{w18x, ExponentialDistribution[1]}, "
     "{w18y, ExponentialDistribution[2]}}], w18t]", "{E^(-2 w18t), E^-w18t}"),   # Check: series -> survival of the other
    ("BirnbaumImportance[ReliabilityDistribution[w18x || w18y, {{w18x, ExponentialDistribution[1]}, "
     "{w18y, ExponentialDistribution[2]}}], w18t]", "{1 - E^(-2 w18t), 1 - E^-w18t}"),
    ("BirnbaumImportance[FailureDistribution[w18x && w18y, {{w18x, ExponentialDistribution[1]}, "
     "{w18y, ExponentialDistribution[2]}}], 1]", "{1 - E^-2, 1 - E^-1}"),
    ("BirnbaumImportance[w18r, 1]", "BirnbaumImportance[w18r, 1]"),
    ("BirnbaumImportance[NormalDistribution[0, 1], 1]", Raises(TypeError)),
    ("BirnbaumImportance[ReliabilityDistribution[w18x, {{w18x, ExponentialDistribution[1]}}], -1]", Raises(ValueError)),
    # ── MeanShift (WL doc) ──
    ("MeanShift[{0, 10, 1, 11}, 1]", "{1/2, 21/2, 1/2, 21/2}"),
    ("MeanShift[{{0, 1}, {2, 3}, {10, 11}, {12, 13}}, 3]", "{{1, 2}, {1, 2}, {11, 12}, {11, 12}}"),
    ("MeanShift[{0, 10, 1, 11}, 1, {1}]", "{1/2, 10, 1, 11}"),
    ("MeanShift[{0, 1, 2}, 1, MaxIterations -> 2]", "{1, 1, 1}"),   # hand: {1/2, 1, 3/2} then {1, 1, 1}
    ("MeanShift[w18l, 1]", "MeanShift[w18l, 1]"),
    ("MeanShift[{1, 2}, -1]", Raises(ValueError)),
    ("MeanShift[{1, 2}, 1, Weights -> {1, 2}]", Raises(NotImplementedError)),
    ("MeanShift[5, 1]", Raises(TypeError)),
    # ── MissingFallback (WL doc) ──
    ("MissingFallback[1, 2]", "1"),
    ("MissingFallback[Missing[], 2]", "2"),
    ('MissingFallback[Missing["reason"], Missing[], 2]', "2"),
    ('MissingFallback[Missing["Unknown"], Missing["NotAvailable"]]', 'Missing["NotAvailable"]'),
    ("MissingFallback[]", "MissingFallback[]"),
    # ── TransformMissing ──
    ('TransformMissing[{1, Missing[], 3, 6}, "Mean"]', "{1, 10/3, 3, 6}"),
    ('TransformMissing[{1, Missing[], 3, 6}, "Median"]', "{1, 3, 3, 6}"),
    ('TransformMissing[{{1, Missing[]}, {Missing[], 4}, {5, 6}}, {"Previous", {"Constant", 0}}]', "{{1, 0}, {1, 4}, {5, 6}}"),
    ('TransformMissing[{Missing[], 2, Missing[]}, "Next"]', "{2, 2, 2}"),
    ('TransformMissing[w18t, "Mean"]', 'TransformMissing[w18t, "Mean"]'),
    ('TransformMissing[{Missing[], Missing[]}, "Mean"]', Raises(ValueError)),
    ('TransformMissing[{1, Missing[]}, "Bogus"]', Raises(ValueError)),
    ('TransformMissing[Tabular[{{1}}], "Mean"]', Raises(NotImplementedError)),
    ('TransformMissing[5, "Mean"]', Raises(TypeError)),
    # ── MomentConvert (WL doc) / MomentEvaluate (WL doc) ──
    ('MomentConvert[Cumulant[2], "Moment"]', "-Moment[1]^2 + Moment[2]"),
    ('MomentConvert[CentralMoment[3], "Moment"]', "2 Moment[1]^3 - 3 Moment[1] Moment[2] + Moment[3]"),
    ('MomentConvert[Moment[2], "CentralMoment"]', "CentralMoment[2] + Moment[1]^2"),
    ('MomentConvert[Moment[3], "FactorialMoment"]', "FactorialMoment[1] + 3 FactorialMoment[2] + FactorialMoment[3]"),
    ('MomentConvert[MomentConvert[Moment[3], "Cumulant"], "Moment"]', "Moment[3]"),   # Check: round trip
    ('MomentConvert[Cumulant[{2, 2}], "Moment"]', Raises(NotImplementedError)),
    ('MomentConvert[Cumulant[2], "UnbiasedSampleEstimator"]', Raises(NotImplementedError)),
    ('MomentConvert[Cumulant[2], "Bogus"]', Raises(ValueError)),
    ('MomentConvert[w18e, "Moment"]', "w18e"),
    ("MomentEvaluate[Moment[1], NormalDistribution[w18m, w18s]]", "w18m"),
    ("MomentEvaluate[Moment[2], NormalDistribution[w18m, w18s]]", "w18m^2 + w18s^2"),
    ("MomentEvaluate[PowerSymmetricPolynomial[2], {w18a, w18b, w18c}]", "w18a^2 + w18b^2 + w18c^2"),
    ("MomentEvaluate[Cumulant[2] + Moment[1]^2, BinomialDistribution[w18n, w18p]] - "
     "Moment[BinomialDistribution[w18n, w18p], 2]", "0"),   # Check: Cumulant[2] + Moment[1]^2 = Moment[2]
    ("MomentEvaluate[Moment[2], w18d]", "MomentEvaluate[Moment[2], w18d]"),
    ("MomentEvaluate[Moment[2], 5]", Raises(TypeError)),
    ("MomentEvaluate[PowerSymmetricPolynomial[2], NormalDistribution[0, 1]]", Raises(ValueError)),
    # ── PillaiTrace / WilksW (documented formulas) ──
    ("PillaiTrace[{1., 2, 3, 4.5}, {2., 1, 4, 3}]", approx(0.31588785046728973)),
    ("PillaiTrace[{1., 2, 3, 4.5}, {2., 1, 4, 3}] - Correlation[{1., 2, 3, 4.5}, {2., 1, 4, 3}]^2", approx(0.0)),  # Check
    ("PillaiTrace[{{1., 2.}, {2., 1.}, {3., 5.}, {4., 3.}, {5., 7.}}, {1., 3., 2., 5., 4.}]",
     approx(0.9963636363636363)),   # R^2 of the regression of the vector on both columns (numpy lstsq)
    ("PillaiTrace[w18m, {1, 2}]", "PillaiTrace[w18m, {1, 2}]"),
    ("PillaiTrace[{1, 2, 3}, {1, 2}]", Raises(ValueError)),
    ('PillaiTrace[{1, 2, 3}, {"a", "b", "c"}]', Raises(TypeError)),
    ("WilksW[{1., 2, 3, 4.5}, {2., 1, 4, 3}]", approx(0.31588785046728973)),
    ("WilksW[{1., 2, 3, 4.5}, {2., 1, 4, 3}] - PillaiTrace[{1., 2, 3, 4.5}, {2., 1, 4, 3}]", approx(0.0)),  # Check: 1 var
    ("WilksW[{{1., 2.}, {2., 1.}, {3., 5.}, {4., 3.}, {5., 7.}}, {1., 3., 2., 5., 4.}]", approx(0.9963636363636363)),
    ("WilksW[w18m, {1, 2}]", "WilksW[w18m, {1, 2}]"),
    ("WilksW[{1, 1, 1}, {1, 2, 3}]", Raises(ValueError)),
    ('PillaiTraceTest[{1., 2, 3, 4.5}, {2., 1, 4, 3}, "TestStatistic"]', approx(0.31588785046728973)),
    ('PillaiTraceTest[{1., 2, 3, 4.5}, {2., 1, 4, 3}, "TestStatistic"] - PillaiTrace[{1., 2, 3, 4.5}, {2., 1, 4, 3}]', approx(0.0)),
    ('PillaiTraceTest[{1., 2, 3}, {1., 3, 2}, "TestStatistic"]', approx(0.25)),
    ('PillaiTraceTest[w18m, {1, 2}, "TestStatistic"]', 'PillaiTraceTest[w18m, {1, 2}, "TestStatistic"]'),
    ("PillaiTraceTest[{1., 2, 3, 4.5}, {2., 1, 4, 3}]", Raises(NotImplementedError)),
    ('PillaiTraceTest[{1., 2, 3}, {1., 3}, "TestStatistic"]', Raises(ValueError)),
    ('WilksWTest[{1., 2, 3, 4.5}, {2., 1, 4, 3}, "TestStatistic"]', approx(0.31588785046728973)),
    ('WilksWTest[{1., 2, 3}, {1., 3, 2}, "TestStatistic"]', approx(0.25)),
    ('WilksWTest[{1., 2, 3}, {1., 3, 2}, "TestStatistic"] - WilksW[{1., 2, 3}, {1., 3, 2}]', approx(0.0)),
    ('WilksWTest[w18m, {1, 2}, "TestStatistic"]', 'WilksWTest[w18m, {1, 2}, "TestStatistic"]'),
    ("WilksWTest[{1., 2, 3}, {1., 3, 2}]", Raises(NotImplementedError)),
    ('WilksWTest[{1., 2, 3}, {1., 3}, "TestStatistic"]', Raises(ValueError)),
    # ── AutocorrelationTest (statsmodels acorr_ljungbox, lags = Ceiling[Log[10]] = 3) ──
    ('AutocorrelationTest[{1., 3, 2, 5, 4, 6, 5, 8, 7, 9}, Automatic, {"TestData", {"LjungBox", "BoxPierce"}}]',
     approx_list([6.469048, 0.090891, 4.536111, 0.209092], 1e-5)),
    ("AutocorrelationTest[{1., 3, 2, 5, 4, 6, 5, 8, 7, 9}]", approx(0.09089121460960711)),   # Automatic = LjungBox
    ('AutocorrelationTest[{1., 3, 2, 5, 4, 6, 5, 8, 7, 9}, 3, "DegreesOfFreedom"]', "3"),
    ("AutocorrelationTest[w18d]", "AutocorrelationTest[w18d]"),
    ("AutocorrelationTest[{1, 2, 3, 4}, 7]", Raises(ValueError)),
    ("AutocorrelationTest[{{1, 2}, {3, 4}, {5, 6}}]", Raises(NotImplementedError)),
    ("AutocorrelationTest[{1, 2, 3, 4}, 1.5]", Raises(TypeError)),
    # ── LogRankTest (WL doc) ──
    ("LogRankTest[{{32, 34, 28, 26, 31, 25, 29, 31, 28, 25}, {30, 19, 28, 35, 35, 29, 33, 32, 31, 29}}]",
     approx(0.192633)),
    ('LogRankTest[{EventData[{32, 34, 28, 26, 31, 25, 29, 31, 28, 25}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}], '
     'EventData[{30, 19, 28, 35, 35, 29, 33, 32, 31, 29}, {0, 0, 1, 0, 0, 1, 0, 1, 0, 1}]}, "Gehan", "TestData"]',
     approx_list([3.35128, 0.0671529])),
    ('LogRankTest[{EventData[{32, 34, 28, 26, 31, 25, 29, 31, 28, 25}, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}], '
     'EventData[{30, 19, 28, 35, 35, 29, 33, 32, 31, 29}, {0, 0, 1, 0, 0, 1, 0, 1, 0, 1}]}, {1, 0}, "TestData"]',
     approx_list([3.54507, 0.0597226])),
    # Check: "Equal" weights are the plain log-rank (rho = 0)
    ("LogRankTest[{{3, 5, 7, 9}, {4, 6, 8, 10, 12}}, \"Equal\"] - LogRankTest[{{3, 5, 7, 9}, {4, 6, 8, 10, 12}}, 0]", approx(0.0)),
    ("LogRankTest[w18d]", "LogRankTest[w18d]"),
    ("LogRankTest[{{1, 2}, {3, 4}}, \"Bogus\"]", Raises(ValueError)),
    ("LogRankTest[{EventData[{1, 2}, {0, 2}], {3, 4}}]", Raises(ValueError)),
    ("LogRankTest[{{{1, 2}}, {3, 4}}]", Raises(NotImplementedError)),
    ("LogRankTest[{1, 2}]", Raises(TypeError)),
    # ── option symbols: inert as a symbol, an expression and in a rule ──
    ("AlternativeHypothesis", "AlternativeHypothesis"),
    ("AlternativeHypothesis[w18a]", "AlternativeHypothesis[w18a]"),
    ("AlternativeHypothesis -> w18a", "AlternativeHypothesis -> w18a"),
    ("SignificanceLevel", "SignificanceLevel"),
    ("SignificanceLevel[w18a]", "SignificanceLevel[w18a]"),
    ("SignificanceLevel -> w18a", "SignificanceLevel -> w18a"),
    ("VerifyTestAssumptions", "VerifyTestAssumptions"),
    ("VerifyTestAssumptions[w18a]", "VerifyTestAssumptions[w18a]"),
    ("VerifyTestAssumptions -> w18a", "VerifyTestAssumptions -> w18a"),
    ("ConfidenceLevel", "ConfidenceLevel"),
    ("ConfidenceLevel[w18a]", "ConfidenceLevel[w18a]"),
    ("ConfidenceLevel -> w18a", "ConfidenceLevel -> w18a"),
    ("ConfidenceRange", "ConfidenceRange"),
    ("ConfidenceRange[w18a]", "ConfidenceRange[w18a]"),
    ("ConfidenceRange -> w18a", "ConfidenceRange -> w18a"),
    ("ConfidenceTransform", "ConfidenceTransform"),
    ("ConfidenceTransform[w18a]", "ConfidenceTransform[w18a]"),
    ("ConfidenceTransform -> w18a", "ConfidenceTransform -> w18a"),
    ("Weights", "Weights"),
    ("Weights[w18a]", "Weights[w18a]"),
    ("Weights -> w18a", "Weights -> w18a"),
    ("IncludeGroupAggregates", "IncludeGroupAggregates"),
    ("IncludeGroupAggregates[w18a]", "IncludeGroupAggregates[w18a]"),
    ("IncludeGroupAggregates -> w18a", "IncludeGroupAggregates -> w18a"),
    ("IntervalMarkers", "IntervalMarkers"),
    ("IntervalMarkers[w18a]", "IntervalMarkers[w18a]"),
    ("IntervalMarkers -> w18a", "IntervalMarkers -> w18a"),
    ("IntervalMarkersStyle", "IntervalMarkersStyle"),
    ("IntervalMarkersStyle[w18a]", "IntervalMarkersStyle[w18a]"),
    ("IntervalMarkersStyle -> w18a", "IntervalMarkersStyle -> w18a"),
    ("LinearOffsetFunction", "LinearOffsetFunction"),
    ("LinearOffsetFunction[w18a]", "LinearOffsetFunction[w18a]"),
    ("LinearOffsetFunction -> w18a", "LinearOffsetFunction -> w18a"),
    ("LinkFunction", "LinkFunction"),
    ("LinkFunction[w18a]", "LinkFunction[w18a]"),
    ("LinkFunction -> w18a", "LinkFunction -> w18a"),
    ("ExponentialFamily", "ExponentialFamily"),
    ("ExponentialFamily[w18a]", "ExponentialFamily[w18a]"),
    ("ExponentialFamily -> w18a", "ExponentialFamily -> w18a"),
    ("DispersionEstimatorFunction", "DispersionEstimatorFunction"),
    ("DispersionEstimatorFunction[w18a]", "DispersionEstimatorFunction[w18a]"),
    ("DispersionEstimatorFunction -> w18a", "DispersionEstimatorFunction -> w18a"),
    ("VarianceEstimatorFunction", "VarianceEstimatorFunction"),
    ("VarianceEstimatorFunction[w18a]", "VarianceEstimatorFunction[w18a]"),
    ("VarianceEstimatorFunction -> w18a", "VarianceEstimatorFunction -> w18a"),
    ("CovarianceEstimatorFunction", "CovarianceEstimatorFunction"),
    ("CovarianceEstimatorFunction[w18a]", "CovarianceEstimatorFunction[w18a]"),
    ("CovarianceEstimatorFunction -> w18a", "CovarianceEstimatorFunction -> w18a"),
    ("NominalVariables", "NominalVariables"),
    ("NominalVariables[w18a]", "NominalVariables[w18a]"),
    ("NominalVariables -> w18a", "NominalVariables -> w18a"),
    ("MissingDataMethod", "MissingDataMethod"),
    ("MissingDataMethod[w18a]", "MissingDataMethod[w18a]"),
    ("MissingDataMethod -> w18a", "MissingDataMethod -> w18a"),
    ("MissingDataRules", "MissingDataRules"),
    ("MissingDataRules[w18a]", "MissingDataRules[w18a]"),
    ("MissingDataRules -> w18a", "MissingDataRules -> w18a"),
    ("MissingValuePattern", "MissingValuePattern"),
    ("MissingValuePattern[w18a]", "MissingValuePattern[w18a]"),
    ("MissingValuePattern -> w18a", "MissingValuePattern -> w18a"),
    ("ParameterEstimator", "ParameterEstimator"),
    ("ParameterEstimator[w18a]", "ParameterEstimator[w18a]"),
    ("ParameterEstimator -> w18a", "ParameterEstimator -> w18a"),
    ("StrataVariables", "StrataVariables"),
    ("StrataVariables[w18a]", "StrataVariables[w18a]"),
    ("StrataVariables -> w18a", "StrataVariables -> w18a"),
    ("FitDegree", "FitDegree"),
    ("FitDegree[w18a]", "FitDegree[w18a]"),
    ("FitDegree -> w18a", "FitDegree -> w18a"),
    ("ClusterDissimilarityFunction", "ClusterDissimilarityFunction"),
    ("ClusterDissimilarityFunction[w18a]", "ClusterDissimilarityFunction[w18a]"),
    ("ClusterDissimilarityFunction -> w18a", "ClusterDissimilarityFunction -> w18a"),
    ("CriterionFunction", "CriterionFunction"),
    ("CriterionFunction[w18a]", "CriterionFunction[w18a]"),
    ("CriterionFunction -> w18a", "CriterionFunction -> w18a"),
    ("DataReversed", "DataReversed"),
    ("DataReversed[w18a]", "DataReversed[w18a]"),
    ("DataReversed -> w18a", "DataReversed -> w18a"),
    ("ExcludedLines", "ExcludedLines"),
    ("ExcludedLines[w18a]", "ExcludedLines[w18a]"),
    ("ExcludedLines -> w18a", "ExcludedLines -> w18a"),
    ("PlotFitElements", "PlotFitElements"),
    ("PlotFitElements[w18a]", "PlotFitElements[w18a]"),
    ("PlotFitElements -> w18a", "PlotFitElements -> w18a"),
    (f'ZTest[{TA}, 2, 10, "PValue", AlternativeHypothesis -> "Less"]', approx(1 - 0.012673659338734126)),
    (f'ZTest[{TA}, 2, 10, AlternativeHypothesis -> Sideways]', Raises(ValueError)),
    (f'ZTest[{TA}, 2, 10, "ShortTestConclusion", SignificanceLevel -> 0.01]', '"Do not reject"'),
    (f'ZTest[{TA}, 2, 10, SignificanceLevel -> 2]', Raises(ValueError)),
    (f'ZTest[{TA}, 2, 10, "PValue", VerifyTestAssumptions -> None] - ZTest[{TA}, 2, 10]', approx(0.0)),
    (f'ZTest[{TA}, 2, 10, VerifyTestAssumptions -> 5]', Raises(ValueError)),
    ("MeanShift[{1, 2}, 1, Weights -> {1, 2}]", Raises(NotImplementedError)),
    # ── Censoring ──
    ("Censoring[3, 1]", "Censoring[3, 1]"),
    ("Censoring[{1, 2, 3}, {0, Right, -1}]", "Censoring[{1, 2, 3}, {0, Right, -1}]"),
    ("Censoring[{1, 2}, Left]", "Censoring[{1, 2}, Left]"),
    ("Censoring[w18t, 1]", "Censoring[w18t, 1]"),
    ("Censoring[3, 2]", Raises(ValueError)),
    ("Censoring[{1, 2}, {0}]", Raises(ValueError)),
    ('Censoring["a", 0]', Raises(TypeError)),
    # ── DesignMatrix (WL doc form; rows {1, f1, ...}) ──
    ("DesignMatrix[{{1, 2}, {2, 3}, {4, 7}}, {w18x, w18x^2}, w18x]", "{{1, 1, 1}, {1, 2, 4}, {1, 4, 16}}"),
    ("DesignMatrix[{{1, 2}, {2, 3}, {4, 7}}, {w18x}, w18x, IncludeConstantBasis -> False]", "{{1}, {2}, {4}}"),
    ("DesignMatrix[{{1, 2, 0}, {3, 4, 1}}, {w18x, w18y, w18x w18y}, {w18x, w18y}]", "{{1, 1, 2, 2}, {1, 3, 4, 12}}"),
    ("DesignMatrix[{5, 7, 9}, w18x, w18x]", "{{1, 1}, {1, 2}, {1, 3}}"),
    ("DesignMatrix[{{1.5, 2}, {2, 3}}, Log[w18x], w18x]", approx_list([1, 0.4054651081081644, 1, 0.6931471805599453])),
    ("DesignMatrix[w18d, w18x, w18x]", "DesignMatrix[w18d, w18x, w18x]"),
    ("DesignMatrix[{{1, 2}, {2, 3}}, w18y, w18x]", Raises(ValueError)),
    ("DesignMatrix[{{1, 2}, {2, 3}}, w18x, 3]", Raises(TypeError)),
    ("DesignMatrix[{{1, 2}, {2, 3}}, w18x, w18x, NominalVariables -> w18x]", Raises(NotImplementedError)),
    # Check: DesignMatrix . BestFitParameters == PredictedResponse
    (f"DesignMatrix[{LMD}, w18x, w18x] . LinearModelFit[{LMD}, w18x, w18x][\"BestFitParameters\"] == "
     f"LinearModelFit[{LMD}, w18x, w18x][\"PredictedResponse\"]", "True"),
    # ── LinearModelFit (WL doc: 0.186441 + 0.694915 x; statsmodels OLS on the same data) ──
    (f"LinearModelFit[{LMD}, w18x, w18x][\"BestFit\"]", "11/59 + 41 w18x/59"),
    (f"LinearModelFit[{LMD}, w18x, w18x][2]", "93/59"),
    (f"LinearModelFit[{LMD}, w18x, w18x][\"FitResiduals\"]", "{48/59, -52/59, -16/59, 20/59}"),
    (f"LinearModelFit[{LMD}, w18x, w18x][\"RSquared\"]", "1681/2065"),
    (f"LinearModelFit[{LMD}, w18x, w18x][\"EstimatedVariance\"]", "48/59"),
    (f"LinearModelFit[{LMD}, w18x, w18x][\"ParameterErrors\"]", "{4 Sqrt[105]/59, 8 Sqrt[3]/59}"),
    # statsmodels OLS: bse [0.69470853, 0.23485435], pvalues [0.81355932, 0.09775636], rsquared_adj 0.72106538
    (f"LinearModelFit[{LMDF}, w18x, w18x][\"ParameterErrors\"]", approx_list([0.69470853, 0.23485435])),
    (f"LinearModelFit[{LMDF}, w18x, w18x][\"ParameterPValues\"]", approx_list([0.81355932, 0.09775636])),
    (f"LinearModelFit[{LMDF}, w18x, w18x][\"AdjustedRSquared\"]", approx(0.7210653753026635)),
    (f"LinearModelFit[{LMDF}, w18x, w18x][\"ParameterConfidenceIntervals\"]",
     approx_list([-2.80264886, 3.17553022, -0.31558144, 1.70541195])),
    # statsmodels llf -3.8767869; WL counts the variance as a parameter: AIC = -2 llf + 2 (p + 1)
    (f"LinearModelFit[{LMDF}, w18x, w18x][\"LogLikelihood\"]", approx(-3.8767869057031428)),
    (f"LinearModelFit[{LMDF}, w18x, w18x][\"AIC\"]", approx(2 * 3.8767869057031428 + 6)),
    (f"LinearModelFit[{LMDF}, w18x, w18x][\"BIC\"]", approx(2 * 3.8767869057031428 + 3 * 1.3862943611198906)),
    # statsmodels WLS(weights = {1, 2, 1, 3}): params [-0.14851485, 0.8019802], bse [0.69926509, 0.19949942]
    (f"LinearModelFit[{LMD}, w18x, w18x, Weights -> {{1, 2, 1, 3}}][\"BestFitParameters\"]", "{-15/101, 81/101}"),
    (f"N[LinearModelFit[{LMD}, w18x, w18x, Weights -> {{1, 2, 1, 3}}][\"ParameterErrors\"]]",
     approx_list([0.69926509, 0.19949942])),
    (f"LinearModelFit[{LMD}, w18x, w18x, Weights -> {{1, 2, 1, 3}}][\"RSquared\"]", "6561/7373"),   # = 0.88986844
    ("LinearModelFit[{1, 3, 2, 5}, {w18x, w18x^2}, w18x][\"BestFit\"]", "5/4 - 3 w18x/20 + w18x^2/4"),
    ("LinearModelFit[{{1, 2, 3}, {2, 1, 5}, {3, 3, 4}, {4, 1, 1}}, {w18x, w18y}, {w18x, w18y}][\"BestFit\"]",
     "13/3 - 2 w18x/3 + w18y/3"),
    ("LinearModelFit[{{{1, 0}, {1, 1}, {1, 3}, {1, 5}}, {1, 0, 2, 4}}][\"BestFitParameters\"]", "{11/59, 41/59}"),
    ("LinearModelFit[{{{1, 0}, {1, 1}, {1, 3}, {1, 5}}, {1, 0, 2, 4}}][1, 2]", "93/59"),
    (f"LinearModelFit[{LMD}, w18x, w18x, IncludeConstantBasis -> False][\"BestFit\"]", "26 w18x/35"),
    (f"LinearModelFit[{LMD}, w18x, w18x][\"Function\"][2]", "93/59"),
    (f"LinearModelFit[{LMD}, w18x, w18x][\"Data\"]", LMD),
    (f"LinearModelFit[{LMD}, w18x, w18x][{{\"RSquared\", \"EstimatedVariance\"}}]", "{1681/2065, 48/59}"),
    ("LinearModelFit[w18d, w18x, w18x]", "LinearModelFit[w18d, w18x, w18x]"),
    ("LinearModelFit[{{1, 2}, {1, 2}}, w18x, w18x]", Raises(ValueError)),
    ("LinearModelFit[{{1, 2}, {2, 3}}, w18x, w18x, w18Foo -> 1]", Raises(ValueError)),
    ('LinearModelFit["a", w18x, w18x]', Raises(TypeError)),
    ("LinearModelFit[{{1, 2}, {2, 3}, {3, 5}}, w18x, w18x, NominalVariables -> w18x]", Raises(NotImplementedError)),
    # ── FittedModel ──
    (f"Head[LinearModelFit[{LMD}, w18x, w18x]]", "FittedModel"),
    (f"LinearModelFit[{LMD}, w18x, w18x][\"CovarianceMatrix\"]", "{{1680/3481, -432/3481}, {-432/3481, 192/3481}}"),
    (f"LinearModelFit[{LMD}, w18x, w18x][\"AdjustedRSquared\"]", "1489/2065"),
    (f"LinearModelFit[{LMD}, w18x, w18x][\"ParameterTable\"]", Raises(NotImplementedError)),
    (f"LinearModelFit[{LMD}, w18x, w18x][\"w18Foo\"]", Raises(ValueError)),
    ("FittedModel[3]", Raises(TypeError)),
    ('FittedModel["x"]', Raises(TypeError)),
    ('FittedModel[<|"a" -> 1|>]["BestFit"]', Raises(ValueError)),
    # Check: the exact covariance matrix equals EstimatedVariance Inverse[X'X]
    (f"LinearModelFit[{LMD}, w18x, w18x][\"CovarianceMatrix\"] == 48/59 Inverse[Transpose[DesignMatrix[{LMD}, "
     f"w18x, w18x]] . DesignMatrix[{LMD}, w18x, w18x]]", "True"),
    # ── NonlinearModelFit (scipy.optimize.curve_fit: [1.00159214, 0.999618], errors [0.00205462, 0.00053049]) ──
    (f"Values[NonlinearModelFit[{EXPD}, w18a Exp[w18b w18x], {{w18a, w18b}}, w18x][\"BestFitParameters\"]]",
     approx_list([1.00159214, 0.999618])),
    (f"NonlinearModelFit[{EXPD}, w18a Exp[w18b w18x], {{w18a, w18b}}, w18x][\"ParameterErrors\"]",
     approx_list([0.00205462, 0.00053049], 1e-4)),
    (f"NonlinearModelFit[{EXPD}, w18a Exp[w18b w18x], {{{{w18a, 2}}, {{w18b, 1/2}}}}, w18x][1]",
     approx(1.00159214 * 2.71724365)),
    ("Values[NonlinearModelFit[{{0, 1.}, {1, 2.7}, {2, 7.4}}, w18a Exp[w18b w18x] + w18c, "
     "{{w18a, 1}, {w18b, 1}, {w18c, 0}}, w18x][\"BestFitParameters\"]]", approx_list([0.96333333, 1.01693426, 0.03666667])),
    # Check: a model linear in its parameters reproduces LinearModelFit
    (f"Values[NonlinearModelFit[{LMDF}, w18a + w18b w18x, {{w18a, w18b}}, w18x][\"BestFitParameters\"]]",
     approx_list([11 / 59, 41 / 59])),
    ("NonlinearModelFit[w18d, w18a Exp[w18b w18x], {w18a, w18b}, w18x]",
     "NonlinearModelFit[w18d, w18a Exp[w18b w18x], {w18a, w18b}, w18x]"),
    (f"NonlinearModelFit[{EXPD}, {{w18a Exp[w18b w18x], w18a > 0}}, {{w18a, w18b}}, w18x]", Raises(NotImplementedError)),
    ("NonlinearModelFit[{{0, 1.}, {1, 2.7}}, w18a Exp[w18b w18x] + w18c, {w18a, w18b, w18c}, w18x]", Raises(ValueError)),
    ("NonlinearModelFit[{{0, 1.}, {1, 2.7}, {2, 3}}, w18a Exp[w18b w18x] + w18q, {w18a, w18b}, w18x]", Raises(ValueError)),
    ("NonlinearModelFit[{{0, 1.}, {1, 2.7}, {2, 3}}, w18a w18x, {3}, w18x]", Raises(TypeError)),
    ("NonlinearModelFit[{{0, 1.}, {1, 2.7}, {2, 3}}, Exp[w18a w18x], {{w18a, 800}}, w18x]", Raises(ArithmeticError)),
    # ── GeneralizedLinearModelFit (statsmodels GLM) ──
    # Poisson: params [0.31876907, 0.43131527], bse [0.57919854, 0.14465024], deviance 0.27985317, aic 21.68704284
    (f'GeneralizedLinearModelFit[{POIS}, w18x, w18x, ExponentialFamily -> "Poisson"]["BestFitParameters"]',
     approx_list([0.31876907, 0.43131527])),
    (f'GeneralizedLinearModelFit[{POIS}, w18x, w18x, ExponentialFamily -> "Poisson"]["ParameterErrors"]',
     approx_list([0.57919854, 0.14465024])),
    (f'GeneralizedLinearModelFit[{POIS}, w18x, w18x, ExponentialFamily -> "Poisson"][{{"Deviance", "AIC", '
     f'"LogLikelihood", "NullDeviance"}}]', approx_list([0.27985317, 21.68704284, -8.84352142, 10.24030961])),
    # Gamma (inverse link): params [0.16725108, 0.30783288], bse [0.01581905, 0.007417], scale 0.00055574197
    (f'GeneralizedLinearModelFit[{GAM}, w18x, w18x, ExponentialFamily -> "Gamma"]["BestFitParameters"]',
     approx_list([0.16725108, 0.30783288])),
    (f'GeneralizedLinearModelFit[{GAM}, w18x, w18x, ExponentialFamily -> "Gamma"]["ParameterErrors"]',
     approx_list([0.01581905, 0.007417])),
    (f'GeneralizedLinearModelFit[{GAM}, w18x, w18x, ExponentialFamily -> "Gamma"]["EstimatedDispersion"]',
     approx(0.0005557419684972075, 1e-6)),
    # InverseGaussian: params [-0.36441801, 0.56568429], bse [0.09673732, 0.06046635], scale 0.0090342499
    (f'GeneralizedLinearModelFit[{GAM}, w18x, w18x, ExponentialFamily -> "InverseGaussian"]["ParameterErrors"]',
     approx_list([0.09673732, 0.06046635])),
    (f'GeneralizedLinearModelFit[{GAM}, w18x, w18x, ExponentialFamily -> "InverseGaussian"]["EstimatedDispersion"]',
     approx(0.009034249897115848, 1e-6)),
    # Gaussian: params [2.2, -0.36], bse [0.26532998, 0.08], scale 0.064, llf 1.05455188
    (f'GeneralizedLinearModelFit[{GAM}, w18x, w18x][{{"EstimatedDispersion", "LogLikelihood"}}]',
     approx_list([0.064, 1.0545518824477762])),
    (f'GeneralizedLinearModelFit[{GAM}, w18x, w18x]["ParameterErrors"]', approx_list([0.26532998, 0.08])),
    # Binomial with cloglog link: params [-3.64211867, 0.88611191], bse [2.63145255, 0.62244104]
    (f'GeneralizedLinearModelFit[{BIN}, w18x, w18x, ExponentialFamily -> "Binomial", '
     f'LinkFunction -> "ComplementaryLogLogLink"]["BestFitParameters"]', approx_list([-3.64211867, 0.88611191], 1e-4)),
    # Check: the Gaussian identity-link GLM equals LinearModelFit
    (f'GeneralizedLinearModelFit[{GAM}, w18x, w18x]["BestFitParameters"] - '
     f'LinearModelFit[{GAM}, w18x, w18x]["BestFitParameters"]', approx_list([0, 0])),
    ("GeneralizedLinearModelFit[w18d, w18x, w18x]", "GeneralizedLinearModelFit[w18d, w18x, w18x]"),
    ('GeneralizedLinearModelFit[{{1, 2}, {2, 3}}, w18x, w18x, ExponentialFamily -> "w18Foo"]', Raises(ValueError)),
    ('GeneralizedLinearModelFit[{{1, 2}, {2, 3}, {3, 4}}, w18x, w18x, LinkFunction -> (Log[#] &)]',
     Raises(NotImplementedError)),
    ('GeneralizedLinearModelFit[{{1, -2}, {2, 3}, {3, 4}}, w18x, w18x, ExponentialFamily -> "Poisson"]', Raises(ValueError)),
    ('GeneralizedLinearModelFit[{{1, 2}, {2, 3}, {3, 4}}, w18x, 5]', Raises(TypeError)),
    ('GeneralizedLinearModelFit[{{1, 1}, {2, 2}, {3, 3}}, w18x, w18x, ExponentialFamily -> "Gamma", '
     'LinkFunction -> "IdentityLink", Weights -> {1, 1, -1}]', Raises(ValueError)),
    # ── LogitModelFit (statsmodels GLM Binomial, tol 1e-14: params [-4.24909655, 1.21402759],
    #    bse [3.38785022, 0.91258556], deviance 4.95597367, llf -2.47798684, aic 8.95597367) ──
    (f'LogitModelFit[{BIN}, w18x, w18x]["BestFitParameters"]', approx_list([-4.24909655, 1.21402759])),
    (f'LogitModelFit[{BIN}, w18x, w18x]["ParameterErrors"]', approx_list([3.38785022, 0.91258556])),
    (f'LogitModelFit[{BIN}, w18x, w18x][{{"Deviance", "LogLikelihood", "AIC"}}]',
     approx_list([4.95597367, -2.47798684, 8.95597367])),
    (f'LogitModelFit[{BIN}, w18x, w18x]["ParameterPValues"]', approx_list([0.2097631, 0.18341368], 1e-4)),
    (f'LogitModelFit[{BIN}, w18x, w18x][3.5]', approx(0.5, 1e-6)),   # Check: symmetric data -> 1/2 at the centre
    ("LogitModelFit[w18d, w18x, w18x]", "LogitModelFit[w18d, w18x, w18x]"),
    ("LogitModelFit[{{1, 2}, {2, 0}, {3, 1}}, w18x, w18x]", Raises(ValueError)),
    ("LogitModelFit[{{1, 0}, {2, 1}, {3, 1}}, w18x, w18x, w18Foo -> 2]", Raises(ValueError)),
    ('LogitModelFit["x", w18x, w18x]', Raises(TypeError)),
    ("LogitModelFit[{{1, 0}, {2, 1}, {3, 1}}, w18x, w18x, LinearOffsetFunction -> w18x]", Raises(NotImplementedError)),
    # ── ProbitModelFit (statsmodels GLM Binomial(probit), tol 1e-14: [-2.65923414, 0.75978118],
    #    bse [1.87154306, 0.50042696]) ──
    (f'ProbitModelFit[{BIN}, w18x, w18x]["BestFitParameters"]', approx_list([-2.65923414, 0.75978118])),
    (f'ProbitModelFit[{BIN}, w18x, w18x]["ParameterErrors"]', approx_list([1.87154306, 0.50042696])),
    (f'ProbitModelFit[{BIN}, w18x, w18x][3.5]', approx(0.5, 1e-6)),   # Check: symmetric data -> 1/2 at the centre
    (f'ProbitModelFit[{BIN}, w18x, w18x][4] - CDF[NormalDistribution[], -2.65923414 + 4 0.75978118]', approx(0.0, 1e-6)),
    ("ProbitModelFit[w18d, w18x, w18x]", "ProbitModelFit[w18d, w18x, w18x]"),
    ("ProbitModelFit[{{1, 2}, {2, 0}, {3, 1}}, w18x, w18x]", Raises(ValueError)),
    ('ProbitModelFit["x", w18x, w18x]', Raises(TypeError)),
    ("ProbitModelFit[{{1, 0}, {2, 1}, {3, 1}}, w18x, w18x, NominalVariables -> w18x]", Raises(NotImplementedError)),
    # ── FindClusters (WL doc data; "Agglomerate" = single linkage; references scipy fcluster(maxclust)) ──
    (f"FindClusters[{{1, 2, 10, 12, 3, 1, 13, 25}}, 3, {AGG}]", "{{1, 2, 3, 1}, {10, 12, 13}, {25}}"),   # WL doc
    (f"FindClusters[{{1 -> w18a, 2 -> w18b, 10 -> w18c, 12 -> w18d, 3 -> w18e, 1 -> w18f, 13 -> w18g, 25 -> w18h}}, 3, "
     f"{AGG}]", "{{w18a, w18b, w18e, w18f}, {w18c, w18d, w18g}, {w18h}}"),   # WL doc
    (f'FindClusters[<|"a" -> 1, "b" -> 2, "c" -> 10, "d" -> 12, "e" -> 3, "f" -> 1, "g" -> 13, "h" -> 25|>, 3, {AGG}]',
     '{{"a", "b", "e", "f"}, {"c", "d", "g"}, {"h"}}'),
    (f"FindClusters[{{1, 2, 10, 12, 3, 1, 13, 25}}, 4, {AGG}]", "{{1, 2, 3, 1}, {10}, {12, 13}, {25}}"),  # single linkage
    # scipy linkage(ward / average / complete) + fcluster(3, maxclust): [1 1 2 1 3]
    ('FindClusters[{{2.5, 3.1}, {5.9, 3.4}, {10, 15}, {2.2, 1.5}, {100, 7.5}}, 3, Method -> {"Agglomerate", '
     '"Linkage" -> "Ward"}, DistanceFunction -> EuclideanDistance]',
     "{{{2.5, 3.1}, {5.9, 3.4}, {2.2, 1.5}}, {{10, 15}}, {{100, 7.5}}}"),
    (f'FindClusters[{{"GGTTT", "GGGGT", "GTTTT", "VVBBS", "VVVBB", "VVVVB", "LLAFF", "LLLAA", "LLLLL"}}, 3, {AGG}, '
     f'DistanceFunction -> HammingDistance]',
     '{{"GGTTT", "GGGGT", "GTTTT"}, {"VVBBS", "VVVBB", "VVVVB"}, {"LLAFF", "LLLAA", "LLLLL"}}'),   # WL doc data
    (f"FindClusters[{{1, 2, 3}}, UpTo[5], {AGG}]", "{{1}, {2}, {3}}"),
    (f"FindClusters[{{1, 2, 10, 11}}, 2, {AGG}, DistanceFunction -> (Abs[#1 - #2] &)]", "{{1, 2}, {10, 11}}"),
    (f"FindClusters[{{1, w18a, 3}}, 2, {AGG}]", f"FindClusters[{{1, w18a, 3}}, 2, {AGG}]"),
    ("FindClusters[{1, 2, 3}, 2]", Raises(NotImplementedError)),
    ('FindClusters[{1, 2, 3}, 2, Method -> "KMeans"]', Raises(NotImplementedError)),
    (f"FindClusters[{{1, 2, 3}}, {AGG}]", Raises(NotImplementedError)),
    (f"FindClusters[{{1, 2, 3}}, 5, {AGG}]", Raises(ValueError)),
    ('FindClusters[{1, 2, 3}, 2, Method -> "w18Foo"]', Raises(ValueError)),
    (f"FindClusters[{{1, 2, 3}}, 0, {AGG}]", Raises(TypeError)),
    (f"FindClusters[3, 2, {AGG}]", Raises(TypeError)),
    # Check: ClusteringComponents labels induce the same partition
    (f"GatherBy[Transpose[{{{{1, 2, 10, 12, 3, 1, 13, 25}}, ClusteringComponents[{{1, 2, 10, 12, 3, 1, 13, 25}}, 3, "
     f"{AGG}]}}], Last][[All, All, 1]] == FindClusters[{{1, 2, 10, 12, 3, 1, 13, 25}}, 3, {AGG}]", "True"),
    # ── ClusteringComponents (WL doc examples) ──
    (f"ClusteringComponents[{{{{1, 2}}, 3, {{10, 11}}, {{12, {{13}}}}, 14}}, 2, {AGG}]", "{{1, 1}, 1, {2, 2}, {2, {2}}, 2}"),
    (f"ClusteringComponents[{{{{1, 2}}, {{3, 4}}, {{10, 11}}, {{12, 13}}}}, 2, 2, {AGG}]", "{{1, 1}, {1, 1}, {2, 2}, {2, 2}}"),
    (f"ClusteringComponents[{{{{1, 2}}, {{3, 4}}, {{10, 11}}, {{12, 13}}}}, 2, 1, {AGG}]", "{1, 1, 2, 2}"),
    (f"ClusteringComponents[{{1, 2, 3, 7, 8}}, 2, {AGG}]", "{1, 1, 1, 2, 2}"),
    (f"ClusteringComponents[w18d, 2, {AGG}]", f"ClusteringComponents[w18d, 2, {AGG}]"),
    ("ClusteringComponents[{1, 2, 3}, 2]", Raises(NotImplementedError)),
    (f"ClusteringComponents[{{1, 2, 3}}, 4, {AGG}]", Raises(ValueError)),
    (f"ClusteringComponents[{{1, 2, 3}}, 2, -1, {AGG}]", Raises(TypeError)),
    # ── ClusteringTree (merge heights: scipy linkage(single) [1, 3] on {1, 2, 5}; ward on CT5) ──
    (f'ClusteringTree[{{1, 2, 5}}, {CDF} -> "Single"]',
     "Graph[{1, 2, 3, 4, 5}, {DirectedEdge[1, 2], DirectedEdge[2, 3], DirectedEdge[2, 4], DirectedEdge[1, 5]}, "
     "VertexWeight -> {1 -> 3., 2 -> 1., 3 -> 0, 4 -> 0, 5 -> 0}, VertexLabels -> {3 -> 1, 4 -> 2, 5 -> 5}]"),
    (f'ClusteringTree[{{1, 2, 5}}, 1.3, {CDF} -> "Single"]',   # WL doc: leaves {1, 2} and {5}
     "Graph[{1, 2, 3}, {DirectedEdge[1, 2], DirectedEdge[1, 3]}, VertexWeight -> {1 -> 3., 2 -> 0, 3 -> 0}, "
     "VertexLabels -> {2 -> {1, 2}, 3 -> {5}}]"),
    # scipy linkage(ward) heights [1.2, 3.11769145, 3.3, 8.28818436]
    (f'Sort[Select[Values[AnnotationValue[ClusteringTree[{CT5}, {CDF} -> "Ward"], VertexWeight]], Positive]]',
     approx_list([1.2, 3.11769145, 3.3, 8.28818436])),
    # scipy linkage(average) [1.2, 2.7, 3.3, 5.35]; complete [1.2, 3.3, 3.3, 8.8]; weighted [1.2, 2.7, 3.3, 5.8]
    (f'Sort[Select[Values[AnnotationValue[ClusteringTree[{CT5}, {CDF} -> "Average"], VertexWeight]], Positive]]',
     approx_list([1.2, 2.7, 3.3, 5.35])),
    (f'Sort[Select[Values[AnnotationValue[ClusteringTree[{CT5}, {CDF} -> "Complete"], VertexWeight]], Positive]]',
     approx_list([1.2, 3.3, 3.3, 8.8])),
    (f'Sort[Select[Values[AnnotationValue[ClusteringTree[{CT5}, {CDF} -> "WeightedAverage"], VertexWeight]], Positive]]',
     approx_list([1.2, 2.7, 3.3, 5.8])),
    (f'TreeGraphQ[ClusteringTree[{CT5}, {CDF} -> "Centroid"]]', "True"),   # Check: the result is a tree
    (f'ClusteringTree[w18d, {CDF} -> "Single"]', f'ClusteringTree[w18d, {CDF} -> "Single"]'),
    ("ClusteringTree[{1, 2, 5}]", Raises(NotImplementedError)),
    (f'ClusteringTree[{{1, 2, 5}}, {CDF} -> "w18Foo"]', Raises(ValueError)),
    (f'ClusteringTree[{{1, 2, 5}}, -1, {CDF} -> "Single"]', Raises(ValueError)),
    (f'ClusteringTree[5, {CDF} -> "Single"]', Raises(TypeError)),
    # ── ClusteringMeasurements (WL doc "Summary": CH 128.42, Dunn 1.79653, RSquared 0.888558,
    #    Silhouette 0.775515, StandardDeviation 0.123269; sklearn calinski_harabasz_score 128.41977007) ──
    (f'ClusteringMeasurements[{CMP}, "CalinskiHarabasz"]', approx(128.41977007099686)),
    (f'ClusteringMeasurements[{CMP}, "Dunn"]', approx(1.79653, 1e-5)),
    (f'ClusteringMeasurements[{CMP}, "RSquared"]', approx(0.888558, 1e-5)),
    (f'ClusteringMeasurements[{CMP}, "Silhouette"]', approx(0.7755149377118608)),
    (f'ClusteringMeasurements[{CMP}, "StandardDeviation"]', approx(0.123269, 1e-5)),
    # sklearn silhouette_samples
    (f'ClusteringMeasurements[{CMP}, "SilhouetteScoreList"]',
     approx_list([0.82587415, 0.79622196, 0.53763841, 0.68472976, 0.62948308, 0.97720216, 0.97745504])),
    (f'ClusteringMeasurements[{CMP}, "SilhouetteScore"]',
     approx_list([(0.82587415 + 0.79622196) / 2, (0.53763841 + 0.68472976 + 0.62948308) / 3, (0.97720216 + 0.97745504) / 2])),
    ('ClusteringMeasurements[{{0.79, 0.56}, {0.71, 0.48}, {0.48, 0.033}, {0.66, -0.18}, {0.71, -0.021}, {-0.88, 0.12}, '
     '{-0.9, 0.092}} -> {1, 1, 2, 2, 2, 3, 3}, "Dunn"]', approx(1.79653, 1e-5)),     # WL doc data format
    ('ClusteringMeasurements[<|1 -> {{0.79, 0.56}, {0.71, 0.48}}, 2 -> {{0.48, 0.033}, {0.66, -0.18}, {0.71, -0.021}}, '
     '3 -> {{-0.88, 0.12}, {-0.9, 0.092}}|>, "Silhouette"]', approx(0.7755149377118608)),
    ('ClusteringMeasurements[{{"king", "kingdom", "kingly"}, {"dutchess", "duchy"}}, "Dunn", '
     'DistanceFunction -> EditDistance]', approx(1.25)),     # WL doc
    # Purity 5/6 by hand; sklearn rand_score([0,0,0,1,1,2],[0,0,1,1,1,2]) = 0.73333333
    ('ClusteringMeasurements[{{1, 2, 3}, {4, 5}, {6}}, {{1, 2}, {3, 4, 5}, {6}}, {"Purity", "Rand"}]',
     approx_list([5 / 6, 0.7333333333333333])),
    # Check: Silhouette is the mean of SilhouetteScoreList
    (f'ClusteringMeasurements[{CMP}, "Silhouette"] - Mean[Flatten[ClusteringMeasurements[{CMP}, "SilhouetteScoreList"]]]',
     approx(0.0)),
    (f'ClusteringMeasurements[w18c, "Dunn"]', f'ClusteringMeasurements[w18c, "Dunn"]'),
    (f'ClusteringMeasurements[{CMP}, "DaviesBouldin"]', Raises(NotImplementedError)),
    (f'ClusteringMeasurements[{CMP}, "w18Foo"]', Raises(ValueError)),
    (f'ClusteringMeasurements[{CMP}, "Purity"]', Raises(ValueError)),
    ('ClusteringMeasurements[{{1, 1}, {1, 1}}, "Dunn"]', Raises(ArithmeticError)),
    ('ClusteringMeasurements[5, "Dunn"]', Raises(TypeError)),
    # ── SurvivalModelFit / SurvivalModel (WL doc values) ──
    (f"SurvivalModelFit[{ED4}][3]", approx(0.5)),
    (f'SurvivalModelFit[{ED4}]["EstimationPoints"]', "{1, 2, 3, 4}"),
    ("SurvivalModelFit[{1, 2, 3, 4}] /@ {1, 2, 3, 4}", approx_list([0.75, 0.5, 0.25, 0.0])),
    ('SurvivalModelFit[{1, 2, 3, 4}]["StandardErrors"]', "{0.21650635094610965, 0.25, 0.21650635094610965, Indeterminate}", 1e-6),
    (f"SurvivalModelFit[{D10}][24]", approx(0.9)),
    (f'SurvivalModelFit[{D10}]["SF"][35]', approx(0.45)),
    (f'SurvivalModelFit[{D10}]["CDF"][{{20, 30}}]', approx_list([0.0, 0.4])),
    # WL doc PointwiseIntervals (log-log transform, Greenwood); the last (S = 0) is Indeterminate
    (f'Most[SurvivalModelFit[{D10}]["PointwiseIntervals"]]',
     approx_list([0.473009, 0.985281, 0.408691, 0.945873, 0.328717, 0.891949, 0.252669, 0.827221, 0.252669, 0.827221,
                  0.126688, 0.73446, 0.126688, 0.73446], 1e-5)),
    (f'Last[SurvivalModelFit[{D10}]["PointwiseIntervals"]]', "{Indeterminate, Indeterminate}"),
    (f'SurvivalModelFit[{D10}]["MedianSurvival"]', "32"),
    (f'SurvivalModelFit[{D10}][{{"AtRisk", "EventCounts", "RightCensoringCounts"}}]',
     "{{10, 9, 8, 7, 6, 4, 2, 1}, {1, 1, 1, 1, 0, 1, 0, 1}, {0, 0, 0, 0, 2, 1, 1, 0}}"),
    (f'SurvivalModelFit[{D10}]["ObservationCount"]', "10"),
    # lifelines KaplanMeierFitter (exponential Greenwood = log-log) gives the same intervals at 24: (0.473009, 0.985281);
    # the linear transform is S -+ 1.96 se = 0.9 -+ 1.959964 * 0.0948683
    (f'First[SurvivalModelFit[{D10}, ConfidenceTransform -> "Linear"]["PointwiseIntervals"]]',
     approx_list([0.9 - 1.959964 * 0.0948683, 1.0], 1e-5)),
    # Check: S(t) is the product of (1 - d/r) over the event table
    (f'SurvivalModelFit[{D10}]["SF"][32] - Times @@ (1 - SurvivalModelFit[{D10}]["EventCounts"][[;; 6]]/'
     f'SurvivalModelFit[{D10}]["AtRisk"][[;; 6]])', approx(0.0)),
    (f"SurvivalModelFit[{ED4}][w18t]", f"SurvivalModelFit[{ED4}][w18t]"),
    ("SurvivalModelFit[w18d]", "SurvivalModelFit[w18d]"),
    ("SurvivalModelFit[{1, {2, 3}}]", Raises(NotImplementedError)),
    ("SurvivalModelFit[EventData[{1, 2}, {0, -1}]]", Raises(NotImplementedError)),
    ("SurvivalModelFit[{1, -2}]", Raises(ValueError)),
    ("SurvivalModelFit[EventData[{1, 2}, {0, 2}]]", Raises(ValueError)),
    ('SurvivalModelFit[{1, 2}, ConfidenceTransform -> "w18Foo"]', Raises(ValueError)),
    ("SurvivalModelFit[5]", Raises(TypeError)),
    ('SurvivalModelFit[{1, 2}]["EventTable"]', "{{1, 2, 1, 0, 0.5}, {2, 1, 1, 0, 0.}}"),
    ('SurvivalModelFit[{1, 2}]["HallWellnerBands"]', Raises(NotImplementedError)),
    ('SurvivalModelFit[{1, 2}]["w18Foo"]', Raises(ValueError)),
    ("SurvivalModel[3]", Raises(TypeError)),
    ('SurvivalModel["x"]', Raises(TypeError)),
    ('SurvivalModel[<|"a" -> 1|>]["Nope"]', Raises(ValueError)),
    ('Head[SurvivalModelFit[{1, 2}]]', "SurvivalModel"),
    ('SurvivalModelFit[{1, 2}]["CHF"][1.5]', approx(0.6931471805599453)),
    ('SurvivalModelFit[{1, 2}]["CHF"]["a"]', Raises(TypeError)),
    # ── CoxModelFit / CoxModel (WL doc values; statsmodels PHReg on the same data) ──
    (f'CoxModelFit[{ED4}]["EventTimes"]', "{1, 2, 4}"),
    (f"CoxModelFit[{ED4}][3]", approx(0.558035, 1e-5)),
    ("CoxModelFit[{1, 2, 3, 4}] /@ {1, 2, 3, 4}", approx_list([0.778801, 0.558035, 0.338465, 0.124514], 1e-5)),
    ('CoxModelFit[{1, 2, 3, 4}]["StandardErrors"][]', approx_list([0.1947, 0.232515, 0.220292, 0.148565], 1e-4)),
    ('CoxModelFit[{1, 2, 3, 4}][{"MartingaleResiduals", "CoxSnellResiduals"}]',
     approx_list([0.75, 0.416667, -0.0833333, -1.08333, 0.25, 0.583333, 1.08333, 2.08333], 1e-5)),
    # WL doc ParameterTable / TestTable; statsmodels PHReg: -0.29167361, bse 0.88880596
    (f'{COX1}["BestFitParameters"]', approx_list([-0.29167361])),
    (f'{COX1}["ParameterStandardErrors"]', approx_list([0.88880596])),
    (f'{COX1}[{{"RelativeRisk", "WaldStatistic", "LikelihoodRatioStatistic", "ScoreStatistic"}}]',
     approx_list([0.747012, 0.107691, 0.117096, 0.110132], 1e-5)),
    (f'{COX1}["LogLikelihood"]', approx(-2.426358511648194)),     # statsmodels PHReg llf
    # WL doc: estimates {-0.0535922, 2.11712}, errors {0.0212829, 0.823307}; statsmodels llf -28.46786993
    (f'{COX2}["BestFitParameters"]', approx_list([-0.0535922, 2.11711687])),
    (f'{COX2}["ParameterStandardErrors"]', approx_list([0.02128287, 0.82330689])),
    (f'{COX2}[{{"LikelihoodRatioStatistic", "WaldStatistic", "ScoreStatistic", "LogLikelihood"}}]',
     approx_list([7.54027, 7.83201, 8.93641, -28.467869927708662], 1e-5)),
    (f'{COX2}["RelativeRiskConfidenceIntervals"]', approx_list([0.909095, 0.988192, 1.65444, 41.7112], 1e-5)),
    (f'{COX2}["ParameterConfidenceIntervals"]', approx_list([-0.09530585, -0.01187855, 0.50346501, 3.73076873])),
    (f'{COX2}["AIC"]', approx(2 * 28.467869927708662 + 4)),
    (f'{COX2}["SF"][{{30, 0}}][20]', approx(0.628764, 1e-5)),
    (f'{COX2}["MedianSurvival"] /@ {{{{25, 0}}, {{25, 1}}}}', "{22, 6}"),
    (f'{COX2}["MedianSurvival"][]', "6"),
    # statsmodels PHReg(ties="efron"): [-0.05496437, 2.17936292], bse [0.02126995, 0.83386988]
    (f'CoxModelFit[{{{XI}, {E20}}}, {{w18u, w18v}}, {{w18u, w18v}}, Method -> "Efron"]["BestFitParameters"]',
     approx_list([-0.05496437, 2.17936292])),
    (f'CoxModelFit[{{{XI}, {E20}}}, {{w18u, w18v}}, {{w18u, w18v}}, Method -> "Efron"]["ParameterStandardErrors"]',
     approx_list([0.02126995, 0.83386988])),
    # statsmodels PHReg basic example (xi = {2, 0, 0, 1, ...}): 1.11193428, bse 0.38346782
    ('CoxModelFit[{{2, 0, 0, 1, 1, 1, 3, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0}, ' + E20 + '}, w18x, w18x]'
     '[{"BestFitParameters", "ParameterStandardErrors"}]', approx_list([1.11193428, 0.38346782])),
    # Check: CHF = -Log[SF], and SF at x0 = baseline SF^RelativeRisk^x0
    (f'{COX2}["CHF"][{{30, 0}}][20] + Log[{COX2}["SF"][{{30, 0}}][20]]', approx(0.0)),
    (f'{COX2}["SF"][{{0, 1}}][20] - {COX2}["SF"][][20]^{COX2}["RelativeRisk"][[2]]', approx(0.0)),
    (f'{COX2}["SF"][{{w18a, 0}}]', f'{COX2}["SF"][{{w18a, 0}}]'),
    ("CoxModelFit[w18d]", "CoxModelFit[w18d]"),
    (f'CoxModelFit[{{{{2, 1, 3, 1}}, {ED4}}}, w18x, w18x, Method -> "Exact"]', Raises(NotImplementedError)),
    (f'CoxModelFit[{{{{2, 1, 3, 1}}, {ED4}}}, w18x, w18x, StrataVariables -> {{w18x}}]', Raises(NotImplementedError)),
    ("CoxModelFit[{{0, 0, 1, 1}, {1, 2, 3, 4}}, w18x, w18x]", Raises(ArithmeticError)),
    ("CoxModelFit[{{1, 2}, {1, 2, 3}}, w18x, w18x]", Raises(ValueError)),
    (f'CoxModelFit[{{{{2, 1, 3, 1}}, {ED4}}}, w18x, w18x, Method -> "w18Foo"]', Raises(ValueError)),
    ("CoxModelFit[EventData[{1, 2}, {1, 1}]]", Raises(ValueError)),
    ("CoxModelFit[{1, 2, 3}, 5, w18x]", Raises(TypeError)),
    (f'{COX1}["ParameterTable"]', Raises(NotImplementedError)),
    (f'{COX1}["w18Foo"]', Raises(ValueError)),
    ("CoxModel[3]", Raises(TypeError)),
    ('CoxModel["x"]', Raises(TypeError)),
    ('CoxModel[<|"a" -> 1|>]["Nope"]', Raises(ValueError)),
    (f"Head[{COX1}]", "CoxModel"),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w18(case):
    check(case)
