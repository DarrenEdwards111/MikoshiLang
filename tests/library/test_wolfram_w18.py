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
    (f'DistributionFitTest[{TA}, NormalDistribution[], "PearsonChiSquare"]', Raises(NotImplementedError)),
    (f'DistributionFitTest[{TA}, NormalDistribution[], "Bogus"]', Raises(ValueError)),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w18(case):
    check(case)
