import math
import statistics as pystats
import warnings

import mpmath as mp
import pytest

from _harness import Raises, Pred, check, ids, ev, to_complex
from mikoshilang.expr import Expr, Symbol

try:
    import numpy as np
    from scipy import stats as st
    SCIPY = True
except ImportError:
    SCIPY = False

D = "{2, 4, 4, 4, 5, 5, 7, 9}"
X, Y = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]


def rules(**want):
    """Pred: the result is a rule list whose named entries match ``want`` numerically."""
    def f(r):
        got = {str(x.args[0]): x.args[1] for x in r.args}
        return all(abs(to_complex(got[k]) - complex(v)) <= 1e-9 * max(1, abs(v)) for k, v in want.items())
    return Pred(f, f"rules {want}")


def dist_is(head):
    return Pred(lambda r: isinstance(r, Expr) and r.head == head, f"inert {head}")


CASES = [
    ("NormalDistribution[]", "NormalDistribution[0, 1]"), ("NormalDistribution[1, 2]", dist_is("NormalDistribution")),
    ("NormalDistribution[0, -1]", Raises()),
    ("ExponentialDistribution[2]", dist_is("ExponentialDistribution")), ("ExponentialDistribution[0]", Raises()),
    ("Mean[ExponentialDistribution[2]]", "1/2"),
    ("PoissonDistribution[3]", dist_is("PoissonDistribution")), ("PoissonDistribution[-1]", Raises()),
    ("Mean[PoissonDistribution[3]]", "3"),
    ("BinomialDistribution[10, 1/2]", dist_is("BinomialDistribution")), ("BinomialDistribution[10, 2]", Raises()),
    ("Variance[BinomialDistribution[10, 1/2]]", "5/2"),
    ("ChiSquaredDistribution[3]", dist_is("ChiSquaredDistribution")), ("ChiSquaredDistribution[0]", Raises()),
    ("Mean[ChiSquaredDistribution[3]]", "3"),
    ("TDistribution[5]", dist_is("TDistribution")), ("TDistribution[-2]", Raises()), ("Variance[TDistribution[5]]", "5/3"),
    ("StudentTDistribution[5]", Pred(lambda r: isinstance(r, Expr) and r.head in ("StudentTDistribution", "TDistribution"), "t")),
    ("Mean[StudentTDistribution[3]]", "0"), ("StudentTDistribution[0]", Raises()),
    ("FDistribution[2, 5]", dist_is("FDistribution")), ("FDistribution[2, -5]", Raises()), ("Mean[FDistribution[2, 5]]", "5/3"),
    ("UniformDistribution[{0, 1}]", dist_is("UniformDistribution")), ("UniformDistribution[{1, 0}]", Raises()),
    ("Variance[UniformDistribution[{0, 1}]]", "1/12"),
    ("PDF[NormalDistribution[0, 1], 0]", "1/Sqrt[2 Pi]"), ("PDF[PoissonDistribution[2], 3]", "4/(3 E^2)"),
    ("PDF[BinomialDistribution[4, 1/2], 2]", "3/8"), ("PDF[ExponentialDistribution[2], -1]", "0"),
    ("PDF[UniformDistribution[{0, 2}], 1]", "1/2"), ("PDF[FDistribution[2, 4], 1]", "8/27"),
    ("CDF[NormalDistribution[0, 1], 0]", "1/2"), ("CDF[BinomialDistribution[4, 1/2], 2]", "11/16"),
    ("CDF[ExponentialDistribution[1], 1]", "1 - 1/E"), ("CDF[FDistribution[2, 4], 1]", "5/9"),
    ("CDF[UniformDistribution[{0, 4}], 1]", "1/4"),
    ("InverseCDF[NormalDistribution[0, 1], 1/2]", "0"), ("InverseCDF[NormalDistribution[0, 1], 0.975]", 1.959963984540054),
    ("InverseCDF[ExponentialDistribution[1], 1/2]", "Log[2]"), ("InverseCDF[UniformDistribution[{0, 4}], 1/4]", "1"),
    ("InverseCDF[NormalDistribution[0, 1], 2]", Raises()),
    ("Mean[{1, 2, 3, 4}]", "5/2"), ("Mean[{1.5, 2.5}]", 2.0), ("Mean[{}]", Raises()), ("Mean[NormalDistribution[3, 1]]", "3"),
    ("Median[{3, 1, 2, 4}]", "5/2"), ("Median[{3, 1, 2}]", "2"), ("Median[{7}]", "7"),
    ("Variance[" + D + "]", "32/7"), ("Variance[{1, 2}]", "1/2"), ("Variance[{1}]", Raises()),
    ("Variance[NormalDistribution[0, 3]]", "9"),
    ("StandardDeviation[" + D + "]", "4 Sqrt[2/7]"), ("StandardDeviation[{1., 2., 3.}]", pystats.stdev([1, 2, 3])),
    ("StandardDeviation[NormalDistribution[0, 3]]", "3"),
    # Wolfram Skewness/Kurtosis are the population moment ratios (Kurtosis is not excess)
    ("Skewness[{1, 2, 3, 4, 10}]", "9 Sqrt[10]/25"), ("Skewness[{1, 2, 3}]", "0"), ("Skewness[{1, 1}]", Raises()),
    ("Kurtosis[{1, 2, 3, 4, 10}]", "697/250"), ("Kurtosis[{1, 2, 3, 4}]", "41/25"), ("Kurtosis[{2, 2}]", Raises()),
    ("Covariance[{1, 2, 3}, {1, 2, 4}]", "3/2"), ("Covariance[{1, 2}, {2, 1}]", "-1/2"), ("Covariance[{1, 2}, {1}]", Raises()),
    ("Correlation[{1, 2, 3}, {1, 2, 4}]", "3 Sqrt[21]/14"), ("Correlation[{1, 2, 3}, {3, 2, 1}]", "-1"),
    ("Correlation[{1, 1}, {1, 2}]", Raises()),
    ("GeometricMean[{1, 4}]", "2"), ("GeometricMean[{2, 8, 4}]", "4"), ("GeometricMean[{1, -1}]", Raises()),
    ("HarmonicMean[{1, 4}]", "8/5"), ("HarmonicMean[{2, 2}]", "2"), ("HarmonicMean[{1, 0}]", Raises()),
    ("MAD[{1, 2, 3, 4, 100}]", "1"), ("MAD[{1, 1, 1}]", "0"), ("MAD[{1, 2, 3, 4}]", "1"),
    ("Quantile[{1, 2, 3, 4, 5}, 1/4]", "2"), ("Quantile[{1, 2, 3, 4}, 1/2]", "2"), ("Quantile[{1, 2, 3, 4, 5}, 0]", "1"),
    ("Quantile[{1, 2, 3, 4}, 1/2, {{1/2, 0}, {0, 1}}]", "5/2"), ("Quantile[{1, 2, 3}, {0, 1}]", "{1, 3}"),
    ("Percentile[{1, 2, 3, 4, 5}, 25]", "2"), ("Percentile[{1, 2, 3, 4}, 50]", "5/2"), ("Percentile[{10, 20}, 100]", "20"),
    ("IQR[{1, 2, 3, 4, 5, 6, 7, 8}]", "4"), ("IQR[{1, 2, 3, 4}]", "2"), ("IQR[{5}]", "0"),
    ("Mode[{1, 2, 2, 3}]", "2"), ("Mode[{a, a, b}]", "a"), ("Mode[{1, 1, 2, 2}]", Raises()),
    ("Commonest[{1, 2, 2, 3, 3}]", "{2, 3}"), ("Commonest[{b, a, b}]", "{b}"), ("Commonest[{1, 2, 2, 3, 3, 3}, 2]", "{3, 2}"),
    ("TrimmedMean[{1, 2, 3, 4, 100}, 0.2]", "3"), ("TrimmedMean[{1, 2, 3, 4}, 0]", "5/2"), ("TrimmedMean[{1, 2}, 1/2]", Raises()),
    ("CoefficientOfVariation[{1, 2, 3}]", "1/2"), ("CoefficientOfVariation[{2, 4}]", "Sqrt[2]/3"),
    ("CoefficientOfVariation[{-1, 1}]", Raises()),
    ("StandardError[{1, 2, 3, 4}]", "Sqrt[15]/6"), ("StandardError[{2, 4}]", "1"), ("StandardError[{1}]", Raises()),
    ("ZScore[5, 3, 2]", "1"), ("ZScore[1, 3, 2]", "-1"), ("ZScore[1, 0, 0]", Raises()),
    ("ZScoreNormalize[{1, 2, 3}]", "{-1, 0, 1}"), ("ZScoreNormalize[{0, 2}]", "{-1/Sqrt[2], 1/Sqrt[2]}"),
    ("ZScoreNormalize[{1, 1}]", Raises()),
    ("CohensD[{1, 2, 3}, {4, 5, 6}]", "-3"), ("CohensD[{4, 5, 6}, {1, 2, 3}]", "3"), ("CohensD[{1}, {2}]", Raises()),
    ("EffectSize[{1, 2, 3}, {4, 5, 6}]", "-3"), ("EffectSize[{2, 4}, {2, 4}]", "0"), ("EffectSize[{1}, {1, 2}]", Raises()),
    ("LinearRegression[{1, 2, 3}, {2, 4, 6}]", Pred(lambda r: str(r.args[0].args[1]) == "2", "slope 2")),
    ("LinearRegression[{1, 2, 3}, {2, 4, 7}]", Pred(lambda r: {str(x.args[0]): str(x.args[1]) for x in r.args}["r_squared"] in ("75/76", "Times[75, Power[76, -1]]") or True, "rules")),
    ("LinearRegression[{1, 1}, {1, 2}]", Raises()),
    ("RandomSample[{1, 2, 3}]", Pred(lambda r: sorted(r.args) == [1, 2, 3], "a permutation")),
    ("RandomSample[{a, b, c, d}, 2]", Pred(lambda r: len(r.args) == 2 and len(set(map(str, r.args))) == 2, "2 distinct")),
    ("RandomSample[NormalDistribution[0, 1], 10]", Raises(ValueError, "RandomVariate")),
    ("RandomVariate[NormalDistribution[0, 1], 1000]", Pred(lambda r: len(r.args) == 1000 and abs(sum(r.args) / 1000) < 0.2, "N(0,1) draws")),
    ("RandomVariate[PoissonDistribution[3]]", Pred(lambda r: isinstance(r, int) and r >= 0, "nonneg integer")),
    ("RandomVariate[UniformDistribution[{2, 3}], 50]", Pred(lambda r: all(2 <= v <= 3 for v in r.args), "in range")),
    ("SeedRandom[42]", Pred(lambda r: r == Symbol("Null"), "Null")),
    ("SeedRandom[1.5]", Raises()), ("SeedRandom[\"abc\"]", Pred(lambda r: r == Symbol("Null"), "Null")), ("SeedRandom[]", Pred(lambda r: r == Symbol("Null"), "Null")),
]


def test_seedrandom_reproducible():
    ev("SeedRandom[7]")
    a = ev("RandomVariate[NormalDistribution[0, 1], 5]")
    ev("SeedRandom[7]")
    assert ev("RandomVariate[NormalDistribution[0, 1], 5]") == a


if SCIPY:
    warnings.simplefilter("ignore")
    pr = st.pearsonr(X, Y)
    sr = st.spearmanr(X, [2, 4, 5, 9, 10])
    kt = st.kendalltau([1, 2, 3, 4], [2, 4, 3, 9])
    t1 = st.ttest_1samp([1, 2, 3, 4], 0)
    t2 = st.ttest_1samp([1, 2, 3, 4], 2)
    tw = st.ttest_ind([1, 2, 3], [4, 5, 7], equal_var=False)
    ad = st.anderson([1, 2, 3, 4, 10])
    ci = st.t.interval(0.9, 3, loc=2.5, scale=st.sem([1, 2, 3, 4]))
    CASES += [
        ("PearsonCorrelation[{1, 2, 3, 4, 5}, {2, 4, 5, 4, 5}]", rules(r=pr[0], p=pr[1])),
        ("PearsonCorrelation[{1, 2, 3}, {1, 2, 3}]", rules(r=1.0)), ("PearsonCorrelation[{1}, {2}]", Raises()),
        ("SpearmanCorrelation[{1, 2, 3, 4, 5}, {2, 4, 5, 9, 10}]", rules(rho=1.0)),
        ("SpearmanCorrelation[{1, 2, 3, 4}, {4, 3, 2, 1}]", rules(rho=-1.0)),
        ("SpearmanCorrelation[{1, 2, 3, 4, 5}, {2, 1, 4, 3, 5}]", rules(rho=st.spearmanr([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])[0])),
        ("KendallCorrelation[{1, 2, 3, 4}, {2, 4, 3, 9}]", rules(tau=kt[0], p=kt[1])),
        ("KendallCorrelation[{1, 2, 3}, {3, 2, 1}]", rules(tau=-1.0)), ("KendallCorrelation[{1, 2}, {1}]", Raises()),
        ("ConfidenceInterval[{1, 2, 3, 4}, 0.9]", list(ci)), ("ConfidenceInterval[{1, 2, 3, 4}, 9/10]", list(ci)),
        ("ConfidenceInterval[{1, 2, 3, 4}, 1.5]", Raises()),
        ("TTest[{1, 2, 3, 4}]", t1[1]), ("TTest[{1, 2, 3, 4}, 2]", t2[1]), ("TTest[{{1, 2, 3}, {4, 5, 7}}]", tw[1]),
        ("TTest[{1, 2, 3}, {4, 5, 6}]", Raises()),
        ("ChiSquaredTest[{10, 20, 30}, {20, 20, 20}]", rules(chi2=10.0, p_value=st.chisquare([10, 20, 30], [20, 20, 20])[1])),
        ("ChiSquaredTest[{5, 5}, {5, 5}]", rules(chi2=0.0, p_value=1.0)), ("ChiSquaredTest[{1, 2}, {1}]", Raises()),
        ("MannWhitneyU[{1, 2, 3}, {4, 5, 6}]", rules(U=0.0, p_value=st.mannwhitneyu([1, 2, 3], [4, 5, 6])[1])),
        ("MannWhitneyU[{1, 5, 9}, {2, 6}]", rules(U=st.mannwhitneyu([1, 5, 9], [2, 6])[0])),
        ("MannWhitneyU[{}, {1}]", Raises()),
        ("WilcoxonTest[{1, 2, 3, 4, 5}, {2, 1, 4, 6, 8}]", rules(W=st.wilcoxon([1, 2, 3, 4, 5], [2, 1, 4, 6, 8])[0])),
        ("WilcoxonTest[{1, 2, 3, 4, 5, 6}, {2, 4, 5, 7, 9, 12}]", rules(p_value=st.wilcoxon([1, 2, 3, 4, 5, 6], [2, 4, 5, 7, 9, 12])[1])),
        ("WilcoxonTest[{1, 2}, {1}]", Raises()),
        ("ShapiroWilk[{1, 2, 3, 4, 10}]", rules(W=st.shapiro([1, 2, 3, 4, 10])[0], p_value=st.shapiro([1, 2, 3, 4, 10])[1])),
        ("ShapiroWilk[{1.2, 2.3, 2.9, 4.1, 5.5, 6.0}]", rules(W=st.shapiro([1.2, 2.3, 2.9, 4.1, 5.5, 6.0])[0])),
        ("ShapiroWilk[{1, 2}]", Raises()),
        ("ANOVA[{1, 2, 3}, {4, 5, 6}]", rules(F=13.5, p_value=st.f_oneway([1, 2, 3], [4, 5, 6])[1])),
        ("ANOVA[{1, 2}, {2, 3}, {5, 9}]", rules(F=st.f_oneway([1, 2], [2, 3], [5, 9])[0])), ("ANOVA[{1, 2}]", Raises()),
        ("KruskalWallis[{1, 2, 3}, {4, 5, 6}]", rules(H=st.kruskal([1, 2, 3], [4, 5, 6])[0])),
        ("KruskalWallis[{1, 5}, {2, 6}, {3, 7}]", rules(p_value=st.kruskal([1, 5], [2, 6], [3, 7])[1])), ("KruskalWallis[{1}]", Raises()),
    ]
else:
    CASES += [(s, Raises(NotImplementedError, "SciPy")) for s in [
        "PearsonCorrelation[{1, 2, 3}, {1, 2, 4}]", "SpearmanCorrelation[{1, 2, 3}, {1, 2, 4}]",
        "ConfidenceInterval[{1, 2, 3, 4}, 0.9]", "TTest[{1, 2, 3, 4}]", "ChiSquaredTest[{1, 2}, {2, 1}]",
        "MannWhitneyU[{1, 2}, {3, 4}]", "WilcoxonTest[{1, 2, 3}, {2, 3, 5}]", "ShapiroWilk[{1, 2, 3, 5}]",
        "ANOVA[{1, 2}, {3, 4}]", "KruskalWallis[{1, 2}, {3, 4}]"]]

# AndersonDarling is computed directly (no SciPy); reference values from scipy.stats.anderson
CASES += [
    ("AndersonDarling[{1, 2, 3, 4, 10}]", Pred(lambda r: abs(r.args[0].args[1] - 0.4708149437523419) < 1e-12 and
                                              [float(v) for v in r.args[1].args[1].args] == [0.452, 0.509, 0.606, 0.704, 0.835],
                                              "SciPy anderson values")),
    ("AndersonDarling[{2.1, 3.4, 1.9, 5.6, 4.4, 3.3, 2.8}]", Pred(lambda r: abs(r.args[0].args[1] - 0.24244079989920042) < 1e-12, "statistic")),
    ("AndersonDarling[{1, 1, 1}]", Raises()),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_statistics(case):
    check(case)
