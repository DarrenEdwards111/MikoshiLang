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
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w18(case):
    check(case)
