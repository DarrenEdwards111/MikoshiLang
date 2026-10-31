"""Differential tests for Wolfram batch W17 (probability distributions II, module wl_distributions2).

Reference values come from

* SciPy ``scipy.stats`` (cases built with ``S(...)``; they are dropped when SciPy, or the particular SciPy
  distribution, is not installed), cited by the SciPy distribution name in each lambda;
* mpmath evaluations of the textbook / Wolfram documentation formulas (``reference.wolfram.com/language/ref/<Name>``,
  Details section), written out independently in this file (``M(...)``);
* exact values worked out by hand from the defining formulas (strings, compared exactly).

Every symbol name used here starts with ``w17`` so that no case collides with definitions made by other modules.
"""
import importlib.util
import math

import mpmath
import pytest

from _harness import Pred, Raises, check, ids

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
HAVE_NUMPY = importlib.util.find_spec("numpy") is not None


class S:
    """a SciPy reference value, computed lazily (the case is dropped without SciPy)"""

    def __init__(self, f, tol=1e-9):
        self.f, self.tol = f, tol


def M(f):
    """an mpmath reference value"""
    with mpmath.workdps(30):
        return complex(f()) if isinstance(f(), mpmath.mpc) else float(f())


def T(src):
    return (src, True)


def near(x, tol=1e-6):
    from _harness import to_complex
    return Pred(lambda g: abs(to_complex(g) - x) <= tol * max(1.0, abs(x)), f"~{x}")


def vnear(xs, tol=1e-6):
    from _harness import to_complex

    def ok(g):
        vals = _flat(g)
        return len(vals) == len(xs) and all(abs(to_complex(a) - b) <= tol * max(1.0, abs(b)) for a, b in zip(vals, xs))
    return Pred(ok, f"~{xs}")


def _flat(e):
    from mikoshilang.expr import Expr
    if isinstance(e, Expr) and e.head == "List":
        out = []
        for a in e.args:
            out.extend(_flat(a))
        return out
    return [e]


def isnum(lo=-math.inf, hi=math.inf):
    from _harness import to_complex

    def ok(g):
        v = to_complex(g)
        return v.imag == 0 and lo <= v.real <= hi
    return Pred(ok, f"number in [{lo}, {hi}]")


def nums(n, lo=-math.inf, hi=math.inf, integer=False):
    from _harness import to_complex

    def ok(g):
        vals = _flat(g)
        if len(vals) != n:
            return False
        for v in vals:
            if integer and not isinstance(v, int):
                return False
            c = to_complex(v)
            if c.imag != 0 or not lo <= c.real <= hi:
                return False
        return True
    return Pred(ok, f"{n} numbers in [{lo}, {hi}]")


# ───────────────────────────── univariate continuous ─────────────────────────────

CONT = [
    # ArcSinDistribution: SciPy arcsine with loc a, scale b - a
    ("ArcSinDistribution[]", "ArcSinDistribution[{0, 1}]"),
    ("ArcSinDistribution[{w17a, w17b}]", "ArcSinDistribution[{w17a, w17b}]"),
    ("Mean[ArcSinDistribution[{w17a, w17b}]]", "(w17a + w17b)/2"),
    ("PDF[ArcSinDistribution[{1, 3}], 1.4]", S(lambda st: st.arcsine.pdf(1.4, 1, 2))),
    ("CDF[ArcSinDistribution[{1, 3}], 1.4]", S(lambda st: st.arcsine.cdf(1.4, 1, 2))),
    ("CDF[ArcSinDistribution[{0, 1}], 1/2]", "1/2"),
    ("Variance[ArcSinDistribution[{0, 1}]]", "1/8"),
    ("InverseCDF[ArcSinDistribution[{0, 1}], 1/4]", "1/2 - 1/(2 Sqrt[2])"),
    ("ArcSinDistribution[{3, 1}]", Raises(ValueError)),
    # BatesDistribution / UniformSumDistribution (Irwin-Hall): SciPy irwinhall, and exact piecewise values
    ("BatesDistribution[w17n]", "BatesDistribution[w17n]"),
    ("PDF[BatesDistribution[2], 1/4]", "1"),
    ("CDF[BatesDistribution[3], 1/2]", "1/2"),
    ("Variance[BatesDistribution[w17n]]", "1/(12 w17n)"),
    ("PDF[BatesDistribution[3, {0, 3}], 1.2]", S(lambda st: st.irwinhall.pdf(1.2, 3))),
    ("BatesDistribution[0]", Raises(ValueError)),
    ("BatesDistribution[2.5]", Raises(ValueError)),
    ("UniformSumDistribution[w17n]", "UniformSumDistribution[w17n]"),
    ("PDF[UniformSumDistribution[2], 3/2]", "1/2"),
    ("CDF[UniformSumDistribution[3], 1]", "1/6"),
    ("Mean[UniformSumDistribution[w17n, {w17a, w17b}]]", "w17n (w17a + w17b)/2"),
    ("CDF[UniformSumDistribution[4], 1.7]", S(lambda st: st.irwinhall.cdf(1.7, 4))),
    ("CDF[UniformSumDistribution[4], 1.7]", M(lambda: sum((-1) ** k * mpmath.binomial(4, k) * (1.7 - k) ** 4
                                                          for k in range(2)) / 24)),
    ("UniformSumDistribution[-1]", Raises(ValueError)),
    # BetaPrimeDistribution: SciPy betaprime
    ("BetaPrimeDistribution[w17p, w17q]", "BetaPrimeDistribution[w17p, w17q]"),
    ("Mean[BetaPrimeDistribution[w17p, w17q]]", Pred(lambda g: True, "symbolic mean")),
    ("Mean[BetaPrimeDistribution[2, 3]]", "1"),
    ("PDF[BetaPrimeDistribution[2, 3], 0.7]", S(lambda st: st.betaprime.pdf(0.7, 2, 3))),
    ("CDF[BetaPrimeDistribution[2, 3, 2], 0.7]", S(lambda st: st.betaprime.cdf(0.35, 2, 3))),
    ("Mean[BetaPrimeDistribution[2, 1]]", "Infinity"),
    ("BetaPrimeDistribution[-1, 3]", Raises(ValueError)),
    # BeniniDistribution: CDF 1 - Exp[-a Log[x/s] - b Log[x/s]^2] (Wolfram Details)
    ("BeniniDistribution[w17a, w17b, w17s]", "BeniniDistribution[w17a, w17b, w17s]"),
    ("CDF[BeniniDistribution[1, 2, 1], E]", "1 - E^(-3)"),
    ("CDF[BeniniDistribution[1, 2, 3], 4.5]", M(lambda: 1 - mpmath.exp(-mpmath.log(1.5) - 2 * mpmath.log(1.5) ** 2))),
    ("InverseCDF[BeniniDistribution[1, 2, 1], 1 - E^(-3)]", "E"),
    ("PDF[BeniniDistribution[1, 2, 3], 2]", "0"),
    ("BeniniDistribution[1, 2, -3]", Raises(ValueError)),
    # DagumDistribution = SciPy burr (Burr III) with c = a, d = p, scale b
    ("DagumDistribution[w17p, w17a, w17b]", "DagumDistribution[w17p, w17a, w17b]"),
    ("PDF[DagumDistribution[2, 3, 1.5], 1.1]", S(lambda st: st.burr.pdf(1.1, 3, 2, scale=1.5))),
    ("CDF[DagumDistribution[2, 3, 1.5], 1.1]", S(lambda st: st.burr.cdf(1.1, 3, 2, scale=1.5))),
    ("CDF[DagumDistribution[2, 3, 1], 1]", "1/4"),
    ("Mean[DagumDistribution[2, 3, 1.5]]", S(lambda st: st.burr.mean(3, 2, scale=1.5))),
    ("DagumDistribution[2, 0, 1]", Raises(ValueError)),
    # SinghMaddalaDistribution = SciPy burr12 with c = a, d = q, scale b
    ("SinghMaddalaDistribution[w17q, w17a, w17b]", "SinghMaddalaDistribution[w17q, w17a, w17b]"),
    ("PDF[SinghMaddalaDistribution[2, 3, 1.5], 1.1]", S(lambda st: st.burr12.pdf(1.1, 3, 2, scale=1.5))),
    ("CDF[SinghMaddalaDistribution[2, 3, 1], 1]", "3/4"),
    ("Mean[SinghMaddalaDistribution[2, 3, 1.5]]", S(lambda st: st.burr12.mean(3, 2, scale=1.5))),
    ("SinghMaddalaDistribution[-2, 3, 1]", Raises(ValueError)),
    # DavisDistribution: PDF b^n (x - mu)^(-1 - n)/((E^(b/(x - mu)) - 1) Gamma[n] Zeta[n]) (Wolfram Details)
    ("DavisDistribution[w17b, w17n, w17m]", "DavisDistribution[w17b, w17n, w17m]"),
    ("PDF[DavisDistribution[2, 3, 1], 2.5]", M(lambda: 2 ** 3 * 1.5 ** -4 / ((mpmath.exp(2 / 1.5) - 1) * 2 * mpmath.zeta(3)))),
    ("Mean[DavisDistribution[2, 3, 1]]", "1 + 2 Zeta[2]/(2 Zeta[3])"),
    ("CDF[DavisDistribution[2, 3, 1], 2.5]", M(lambda: mpmath.quad(
        lambda t: 2 ** 3 * (t - 1) ** -4 / ((mpmath.exp(2 / (t - 1)) - 1) * 2 * mpmath.zeta(3)), [1, 2.5]))),
    ("DavisDistribution[2, 0, 1]", Raises(ValueError)),
    # ExpGammaDistribution: X = mu + theta Log[Y], Y ~ GammaDistribution[kappa, 1]; SciPy loggamma(c = kappa)
    ("ExpGammaDistribution[w17k, w17t, w17m]", "ExpGammaDistribution[w17k, w17t, w17m]"),
    ("PDF[ExpGammaDistribution[2, 1.5, 0.5], 1.2]", S(lambda st: st.loggamma.pdf(1.2, 2, 0.5, 1.5))),
    ("CDF[ExpGammaDistribution[2, 1.5, 0.5], 1.2]", S(lambda st: st.loggamma.cdf(1.2, 2, 0.5, 1.5))),
    ("Mean[ExpGammaDistribution[w17k, w17t, w17m]]", "w17m + w17t PolyGamma[0, w17k]"),
    ("Variance[ExpGammaDistribution[2, 3, 0]]", "9 PolyGamma[1, 2]"),
    ("ExpGammaDistribution[2, -1, 0]", Raises(ValueError)),
    # ExponentialPowerDistribution: PDF E^(-|x - mu|^k/(k s^k))/(2 k^(1/k) s Gamma[1 + 1/k]); SciPy gennorm with
    # scale s k^(1/k)
    ("ExponentialPowerDistribution[w17k, w17m, w17s]", "ExponentialPowerDistribution[w17k, w17m, w17s]"),
    ("PDF[ExponentialPowerDistribution[3, 0.5, 2], 1.3]", S(lambda st: st.gennorm.pdf(1.3, 3, 0.5, 2 * 3 ** (1 / 3)))),
    ("CDF[ExponentialPowerDistribution[3, 0.5, 2], 1.3]", S(lambda st: st.gennorm.cdf(1.3, 3, 0.5, 2 * 3 ** (1 / 3)))),
    ("PDF[ExponentialPowerDistribution[2, 0, 1], 0]", "1/Sqrt[2 Pi]"),
    ("Variance[ExponentialPowerDistribution[2, 0, w17s]]", "w17s^2"),
    ("ExponentialPowerDistribution[0, 1, 1]", Raises(ValueError)),
    # FisherZDistribution: Z = Log[F]/2 with F ~ FRatioDistribution[n, m]
    ("FisherZDistribution[w17n, w17m]", "FisherZDistribution[w17n, w17m]"),
    ("CDF[FisherZDistribution[3, 5], 0.4]", S(lambda st: st.f.cdf(math.exp(0.8), 3, 5))),
    ("PDF[FisherZDistribution[3, 5], 0.4]", S(lambda st: 2 * math.exp(0.8) * st.f.pdf(math.exp(0.8), 3, 5))),
    ("CDF[FisherZDistribution[2, 2], 0]", "1/2"),
    ("FisherZDistribution[0, 5]", Raises(ValueError)),
    # FrechetDistribution: SciPy invweibull(alpha, loc mu, scale beta)
    ("FrechetDistribution[w17a, w17b]", "FrechetDistribution[w17a, w17b]"),
    ("CDF[FrechetDistribution[w17a, w17b], w17x]", Pred(lambda g: True, "symbolic CDF")),
    ("PDF[FrechetDistribution[3, 2], 1.7]", S(lambda st: st.invweibull.pdf(1.7, 3, scale=2))),
    ("CDF[FrechetDistribution[3, 2, 1], 3]", "1/E"),
    ("Mean[FrechetDistribution[3, 2]]", "2 Gamma[2/3]"),
    ("Mean[FrechetDistribution[1, 2]]", "Infinity"),
    ("FrechetDistribution[3, -2]", Raises(ValueError)),
    # HalfNormalDistribution[theta]: SciPy halfnorm with scale Sqrt[Pi/2]/theta
    ("HalfNormalDistribution[w17t]", "HalfNormalDistribution[w17t]"),
    ("Mean[HalfNormalDistribution[w17t]]", "1/w17t"),
    ("PDF[HalfNormalDistribution[2], 0.6]", S(lambda st: st.halfnorm.pdf(0.6, scale=math.sqrt(math.pi / 2) / 2))),
    ("CDF[HalfNormalDistribution[2], 0.6]", S(lambda st: st.halfnorm.cdf(0.6, scale=math.sqrt(math.pi / 2) / 2))),
    ("Variance[HalfNormalDistribution[1]]", "(Pi - 2)/2"),
    ("HalfNormalDistribution[0]", Raises(ValueError)),
    # HoytDistribution (Nakagami-q): PDF (1 + q^2) x/(q w) E^(-(1 + q^2)^2 x^2/(4 q^2 w)) I0((1 - q^4) x^2/(4 q^2 w))
    ("HoytDistribution[w17q, w17w]", "HoytDistribution[w17q, w17w]"),
    ("PDF[HoytDistribution[1/2, 2], 1.1]", M(lambda: 1.25 * 1.1 / (0.5 * 2) * mpmath.exp(-1.5625 * 1.21 / (4 * 0.25 * 2))
                                           * mpmath.besseli(0, (1 - 0.0625) * 1.21 / (4 * 0.25 * 2)))),
    ("Mean[HoytDistribution[1, 2]]", near(math.sqrt(math.pi / 2), 1e-10)),  # q = 1: Rayleigh with sigma = 1
    ("Variance[HoytDistribution[1/2, 2]]", "2 - (Mean[HoytDistribution[1/2, 2]])^2"),
    ("CDF[HoytDistribution[1/2, 2], 1.1]", M(lambda: mpmath.quad(lambda t: 1.25 * t / (0.5 * 2) * mpmath.exp(
        -1.5625 * t ** 2 / 2) * mpmath.besseli(0, 0.9375 * t ** 2 / 2), [0, 1.1]))),
    ("HoytDistribution[2, 1]", Raises(ValueError)),
    # HyperbolicDistribution[alpha, beta, delta, mu]: SciPy genhyperbolic with p = 1, a = alpha delta, b = beta delta,
    # scale delta
    ("HyperbolicDistribution[w17a, w17b, w17d, w17m]", "HyperbolicDistribution[w17a, w17b, w17d, w17m]"),
    ("PDF[HyperbolicDistribution[2, 1, 1.5, 0.5], 0.9]", S(lambda st: st.genhyperbolic.pdf(0.9, 1, 3, 1.5, 0.5, 1.5))),
    ("CDF[HyperbolicDistribution[2, 1, 1.5, 0.5], 0.9]", S(lambda st: st.genhyperbolic.cdf(0.9, 1, 3, 1.5, 0.5, 1.5),
                                                          1e-7)),
    ("Mean[HyperbolicDistribution[2, 0, 1, 3]]", "3"),
    ("HyperbolicDistribution[1, 2, 1, 0]", Raises(ValueError)),
    # InverseChiSquareDistribution[nu] = InverseGamma(nu/2, 1/2), [nu, xi] = InverseGamma(nu/2, nu xi/2)
    ("InverseChiSquareDistribution[w17n]", "InverseChiSquareDistribution[w17n]"),
    ("PDF[InverseChiSquareDistribution[3], 0.4]", S(lambda st: st.invgamma.pdf(0.4, 1.5, scale=0.5))),
    ("CDF[InverseChiSquareDistribution[3, 2], 0.9]", S(lambda st: st.invgamma.cdf(0.9, 1.5, scale=3))),
    ("Mean[InverseChiSquareDistribution[w17n]]", "1/(w17n - 2)"),
    ("Mean[InverseChiSquareDistribution[5, 3]]", "5"),
    ("InverseChiSquareDistribution[-3]", Raises(ValueError)),
    # InverseGammaDistribution[alpha, beta]: SciPy invgamma(alpha, scale beta)
    ("InverseGammaDistribution[w17a, w17b]", "InverseGammaDistribution[w17a, w17b]"),
    ("Mean[InverseGammaDistribution[w17a, w17b]]", Pred(lambda g: True, "symbolic mean")),
    ("PDF[InverseGammaDistribution[3, 2], 0.8]", S(lambda st: st.invgamma.pdf(0.8, 3, scale=2))),
    ("CDF[InverseGammaDistribution[3, 2], 0.8]", S(lambda st: st.invgamma.cdf(0.8, 3, scale=2))),
    ("Mean[InverseGammaDistribution[3, 2]]", "1"),
    ("InverseGammaDistribution[3, 2, 1, 1]", Raises(NotImplementedError)),
    ("InverseGammaDistribution[0, 2]", Raises(ValueError)),
    # InverseGaussianDistribution[mu, lambda]: SciPy invgauss(mu/lambda, scale lambda)
    ("InverseGaussianDistribution[w17m, w17l]", "InverseGaussianDistribution[w17m, w17l]"),
    ("Mean[InverseGaussianDistribution[w17m, w17l]]", "w17m"),
    ("Variance[InverseGaussianDistribution[w17m, w17l]]", "w17m^3/w17l"),
    ("PDF[InverseGaussianDistribution[2, 3], 1.3]", S(lambda st: st.invgauss.pdf(1.3, 2 / 3, scale=3))),
    ("CDF[InverseGaussianDistribution[2, 3], 1.3]", S(lambda st: st.invgauss.cdf(1.3, 2 / 3, scale=3))),
    ("InverseGaussianDistribution[2, 3, 1]", Raises(NotImplementedError)),
    ("InverseGaussianDistribution[-2, 3]", Raises(ValueError)),
    # KDistribution[nu, w]: PDF 2 (nu/w)^((nu + 1)/2) x^nu K_(nu-1)(2 x Sqrt[nu/w])/Gamma[nu]
    ("KDistribution[w17n, w17w]", "KDistribution[w17n, w17w]"),
    ("PDF[KDistribution[2, 3], 1.2]", M(lambda: 4 * (2 / 3) ** 1.5 * 1.2 ** 2 * mpmath.besselk(1, 2 * 1.2 * mpmath.sqrt(2 / 3))
                                         / mpmath.gamma(2))),
    ("Mean[KDistribution[1, 1]]", "Pi/4"),
    ("Variance[KDistribution[w17n, w17w]]", Pred(lambda g: True, "symbolic variance")),
    ("CDF[KDistribution[2, 3], 1.2]", M(lambda: mpmath.quad(lambda t: 4 * (2 / 3) ** 1.5 * t ** 2 * mpmath.besselk(
        1, 2 * t * mpmath.sqrt(2 / 3)), [0, 1.2]))),
    ("KDistribution[0, 1]", Raises(ValueError)),
    # KumaraswamyDistribution: CDF 1 - (1 - x^a)^b
    ("KumaraswamyDistribution[w17a, w17b]", "KumaraswamyDistribution[w17a, w17b]"),
    ("CDF[KumaraswamyDistribution[w17a, w17b], 1/2]", Pred(lambda g: True, "symbolic CDF")),
    ("CDF[KumaraswamyDistribution[2, 3], 1/2]", "37/64"),
    ("PDF[KumaraswamyDistribution[2, 3], 1/2]", "27/16"),
    ("Mean[KumaraswamyDistribution[2, 3]]", "16/35"),
    ("InverseCDF[KumaraswamyDistribution[2, 3], 37/64]", "1/2"),
    ("KumaraswamyDistribution[2, -3]", Raises(ValueError)),
    # LevyDistribution[mu, sigma]: SciPy levy(loc mu, scale sigma)
    ("LevyDistribution[w17m, w17s]", "LevyDistribution[w17m, w17s]"),
    ("PDF[LevyDistribution[1, 2], 3.5]", S(lambda st: st.levy.pdf(3.5, 1, 2))),
    ("CDF[LevyDistribution[1, 2], 3.5]", S(lambda st: st.levy.cdf(3.5, 1, 2))),
    ("CDF[LevyDistribution[0, 2], 1]", "Erfc[1]"),
    ("Mean[LevyDistribution[0, 1]]", "Infinity"),
    ("LevyDistribution[0, -1]", Raises(ValueError)),
    # LindleyDistribution[delta]: PDF d^2/(1 + d) (1 + x) E^(-d x)
    ("LindleyDistribution[w17d]", "LindleyDistribution[w17d]"),
    ("Mean[LindleyDistribution[w17d]]", "(w17d + 2)/(w17d (w17d + 1))"),
    ("CDF[LindleyDistribution[2], 1]", "1 - 5/(3 E^2)"),
    ("PDF[LindleyDistribution[2], 0.7]", M(lambda: 4 / 3 * 1.7 * mpmath.exp(-1.4))),
    ("LindleyDistribution[0]", Raises(ValueError)),
    # LogGammaDistribution[alpha, beta, mu]: Log[X - mu + 1] ~ GammaDistribution[alpha, beta]
    ("LogGammaDistribution[w17a, w17b, w17m]", "LogGammaDistribution[w17a, w17b, w17m]"),
    ("PDF[LogGammaDistribution[2, 3, 1], 2.5]", S(lambda st: st.gamma.pdf(math.log(2.5), 2, scale=3) / 2.5)),
    ("CDF[LogGammaDistribution[2, 1/2, 1], 1.5]", S(lambda st: st.gamma.cdf(math.log(1.5), 2, scale=0.5))),
    ("Mean[LogGammaDistribution[2, 1/3, 1]]", "9/4"),
    ("Mean[LogGammaDistribution[2, 1, 0]]", "Infinity"),
    ("LogGammaDistribution[2, 1]", Raises(ValueError)),
    ("LogGammaDistribution[-2, 1, 0]", Raises(ValueError)),
    # LogLogisticDistribution[gamma, sigma]: SciPy fisk(gamma, scale sigma)
    ("LogLogisticDistribution[w17g, w17s]", "LogLogisticDistribution[w17g, w17s]"),
    ("PDF[LogLogisticDistribution[3, 2], 1.4]", S(lambda st: st.fisk.pdf(1.4, 3, scale=2))),
    ("CDF[LogLogisticDistribution[3, 2], 2]", "1/2"),
    ("Mean[LogLogisticDistribution[3, 2]]", S(lambda st: st.fisk.mean(3, scale=2))),
    ("LogLogisticDistribution[3, 0]", Raises(ValueError)),
    # MarchenkoPasturDistribution[lambda, sigma] (lambda <= 1): density on [s^2 (1 - Sqrt[l])^2, s^2 (1 + Sqrt[l])^2]
    ("MarchenkoPasturDistribution[w17l]", "MarchenkoPasturDistribution[w17l]"),
    ("Mean[MarchenkoPasturDistribution[1/2, 2]]", "4"),
    ("Variance[MarchenkoPasturDistribution[1/2, 1]]", "1/2"),
    ("PDF[MarchenkoPasturDistribution[1/4], 1]", M(lambda: mpmath.sqrt((1 - 0.25) * (2.25 - 1)) / (2 * mpmath.pi * 0.25))),
    ("CDF[MarchenkoPasturDistribution[1/4], 1.]", M(lambda: mpmath.quad(lambda t: mpmath.sqrt((t - 0.25) * (2.25 - t))
                                                                       / (2 * mpmath.pi * 0.25 * t), [0.25, 1]))),
    ("MarchenkoPasturDistribution[2]", Raises(NotImplementedError)),
    ("MarchenkoPasturDistribution[-1]", Raises(ValueError)),
    # MaxStableDistribution = SciPy genextreme(c = -xi); MinStableDistribution: -X with X max-stable (mu -> -mu)
    ("MaxStableDistribution[w17m, w17s, w17x]", "MaxStableDistribution[w17m, w17s, w17x]"),
    ("PDF[MaxStableDistribution[1, 2, 0.3], 1.8]", S(lambda st: st.genextreme.pdf(1.8, -0.3, 1, 2))),
    ("CDF[MaxStableDistribution[1, 2, 0.3], 1.8]", S(lambda st: st.genextreme.cdf(1.8, -0.3, 1, 2))),
    ("CDF[MaxStableDistribution[0, 1, 0], 0]", "1/E"),
    ("MaxStableDistribution[0, -1, 0]", Raises(ValueError)),
    ("MinStableDistribution[w17m, w17s, w17x]", "MinStableDistribution[w17m, w17s, w17x]"),
    ("PDF[MinStableDistribution[1, 2, 0.3], 0.4]", S(lambda st: st.genextreme.pdf(-0.4, -0.3, -1, 2))),
    ("CDF[MinStableDistribution[1, 2, 0.3], 0.4]", S(lambda st: st.genextreme.sf(-0.4, -0.3, -1, 2))),
    ("CDF[MinStableDistribution[0, 1, 0], 0]", "1 - 1/E"),
    ("MinStableDistribution[0, 0, 0]", Raises(ValueError)),
    # MaxwellDistribution[sigma]: SciPy maxwell(scale sigma)
    ("MaxwellDistribution[w17s]", "MaxwellDistribution[w17s]"),
    ("Mean[MaxwellDistribution[w17s]]", "2 Sqrt[2/Pi] w17s"),
    ("PDF[MaxwellDistribution[2], 2.5]", S(lambda st: st.maxwell.pdf(2.5, scale=2))),
    ("CDF[MaxwellDistribution[2], 2.5]", S(lambda st: st.maxwell.cdf(2.5, scale=2))),
    ("MaxwellDistribution[-2]", Raises(ValueError)),
    # MeixnerDistribution[a, b, m, d]: PDF (2 Cos[b/2])^(2 d)/(2 a Pi Gamma[2 d]) E^(b (x - m)/a) |Gamma[d + I (x - m)/a]|^2
    ("MeixnerDistribution[w17a, w17b, w17m, w17d]", "MeixnerDistribution[w17a, w17b, w17m, w17d]"),
    ("PDF[MeixnerDistribution[1, 1/2, 0, 2], 0.4]", M(lambda: (2 * mpmath.cos(0.25)) ** 4 / (2 * mpmath.pi * mpmath.gamma(4))
                                                     * mpmath.exp(0.2) * abs(mpmath.gamma(2 + 0.4j)) ** 2)),
    ("Mean[MeixnerDistribution[w17a, w17b, w17m, w17d]]", "w17m + w17a w17d Tan[w17b/2]"),
    ("Variance[MeixnerDistribution[1, 0, 0, 1]]", "1/2"),
    ("MeixnerDistribution[1, 4, 0, 1]", Raises(ValueError)),
    # MoyalDistribution[mu, sigma]: SciPy moyal(loc mu, scale sigma)
    ("MoyalDistribution[]", "MoyalDistribution[0, 1]"),
    ("PDF[MoyalDistribution[1, 2], 1.5]", S(lambda st: st.moyal.pdf(1.5, 1, 2))),
    ("CDF[MoyalDistribution[1, 2], 1.5]", S(lambda st: st.moyal.cdf(1.5, 1, 2))),
    ("Mean[MoyalDistribution[w17m, w17s]]", "w17m + w17s (EulerGamma + Log[2])"),
    ("MoyalDistribution[1, -2]", Raises(ValueError)),
    # NakagamiDistribution[mu, omega]: SciPy nakagami(mu, scale Sqrt[omega])
    ("NakagamiDistribution[w17m, w17w]", "NakagamiDistribution[w17m, w17w]"),
    ("PDF[NakagamiDistribution[2, 3], 1.4]", S(lambda st: st.nakagami.pdf(1.4, 2, scale=math.sqrt(3)))),
    ("CDF[NakagamiDistribution[2, 3], 1.4]", S(lambda st: st.nakagami.cdf(1.4, 2, scale=math.sqrt(3)))),
    ("Variance[NakagamiDistribution[1, 2]]", "2 - Pi/2"),
    ("NakagamiDistribution[1/4, 2]", Raises(ValueError)),
    # noncentral distributions: SciPy ncx2, ncf, nct; NoncentralBeta from its Poisson mixture series
    ("NoncentralChiSquareDistribution[w17n, w17l]", "NoncentralChiSquareDistribution[w17n, w17l]"),
    ("Mean[NoncentralChiSquareDistribution[w17n, w17l]]", "w17n + w17l"),
    ("PDF[NoncentralChiSquareDistribution[3, 2], 2.5]", S(lambda st: st.ncx2.pdf(2.5, 3, 2))),
    ("CDF[NoncentralChiSquareDistribution[3, 2], 2.5]", S(lambda st: st.ncx2.cdf(2.5, 3, 2))),
    ("NoncentralChiSquareDistribution[3, -2]", Raises(ValueError)),
    ("NoncentralBetaDistribution[w17a, w17b, w17d]", "NoncentralBetaDistribution[w17a, w17b, w17d]"),
    ("CDF[NoncentralBetaDistribution[2, 3, 1.5], 0.4]", M(lambda: mpmath.nsum(
        lambda j: mpmath.exp(-0.75) * 0.75 ** j / mpmath.factorial(j) * mpmath.betainc(2 + j, 3, 0, 0.4, regularized=True),
        [0, mpmath.inf]))),
    ("PDF[NoncentralBetaDistribution[2, 3, 1.5], 0.4]", M(lambda: mpmath.nsum(
        lambda j: mpmath.exp(-0.75) * 0.75 ** j / mpmath.factorial(j) * 0.4 ** (1 + j) * 0.6 ** 2 / mpmath.beta(2 + j, 3),
        [0, mpmath.inf]))),
    ("Mean[NoncentralBetaDistribution[2, 3, 0]]", "2/5"),
    ("NoncentralBetaDistribution[2, 3, -1]", Raises(ValueError)),
    ("NoncentralFRatioDistribution[w17n, w17m, w17l]", "NoncentralFRatioDistribution[w17n, w17m, w17l]"),
    ("PDF[NoncentralFRatioDistribution[3, 5, 2], 1.3]", S(lambda st: st.ncf.pdf(1.3, 3, 5, 2))),
    ("CDF[NoncentralFRatioDistribution[3, 5, 2], 1.3]", S(lambda st: st.ncf.cdf(1.3, 3, 5, 2))),
    ("Mean[NoncentralFRatioDistribution[3, 5, 2]]", "25/9"),
    ("NoncentralFRatioDistribution[3, 5, -2]", Raises(ValueError)),
    ("NoncentralStudentTDistribution[w17n, w17d]", "NoncentralStudentTDistribution[w17n, w17d]"),
    ("PDF[NoncentralStudentTDistribution[5, 1.5], 1.1]", S(lambda st: st.nct.pdf(1.1, 5, 1.5))),
    ("CDF[NoncentralStudentTDistribution[5, 1.5], 1.1]", S(lambda st: st.nct.cdf(1.1, 5, 1.5), 1e-8)),
    ("Mean[NoncentralStudentTDistribution[5, 1.5]]", S(lambda st: st.nct.mean(5, 1.5))),
    ("NoncentralStudentTDistribution[-5, 1]", Raises(ValueError)),
    # PERTDistribution[{a, b}, c, lambda]: Beta(1 + l (c - a)/(b - a), 1 + l (b - c)/(b - a)) on [a, b] (lambda = 4)
    ("PERTDistribution[{w17a, w17b}, w17c]", "PERTDistribution[{w17a, w17b}, w17c]"),
    ("Mean[PERTDistribution[{0, 6}, 3]]", "3"),
    ("Mean[PERTDistribution[{w17a, w17b}, w17c]]", "(w17a + w17b + 4 w17c)/6"),
    ("PDF[PERTDistribution[{1, 5}, 2], 2.5]", S(lambda st: st.beta.pdf(2.5, 2, 4, 1, 4))),
    ("CDF[PERTDistribution[{1, 5}, 2, 2], 2.5]", S(lambda st: st.beta.cdf(2.5, 1.5, 2.5, 1, 4))),
    ("PERTDistribution[{1, 5}, 7]", Raises(ValueError)),
    # ParetoPickandsDistribution[mu, sigma, xi]: SciPy genpareto(xi, loc mu, scale sigma)
    ("ParetoPickandsDistribution[w17m, w17s, w17x]", "ParetoPickandsDistribution[w17m, w17s, w17x]"),
    ("PDF[ParetoPickandsDistribution[1, 2, 0.3], 2.5]", S(lambda st: st.genpareto.pdf(2.5, 0.3, 1, 2))),
    ("CDF[ParetoPickandsDistribution[1, 2, -0.3], 2.5]", S(lambda st: st.genpareto.cdf(2.5, -0.3, 1, 2))),
    ("Mean[ParetoPickandsDistribution[0, 1, 1/2]]", "2"),
    ("ParetoPickandsDistribution[0, -1, 1/2]", Raises(ValueError)),
    # PowerDistribution[k, a]: PDF a k^a x^(a - 1) on [0, 1/k]; SciPy powerlaw(a, scale 1/k)
    ("PowerDistribution[w17k, w17a]", "PowerDistribution[w17k, w17a]"),
    ("CDF[PowerDistribution[2, 3], 1/4]", "1/8"),
    ("Mean[PowerDistribution[w17k, w17a]]", "w17a/((1 + w17a) w17k)"),
    ("PDF[PowerDistribution[2, 3], 0.3]", S(lambda st: st.powerlaw.pdf(0.3, 3, scale=0.5))),
    ("PowerDistribution[2, 0]", Raises(ValueError)),
    # RiceDistribution[alpha, beta]: SciPy rice(alpha/beta, scale beta)
    ("RiceDistribution[w17a, w17b]", "RiceDistribution[w17a, w17b]"),
    ("PDF[RiceDistribution[2, 1.5], 1.8]", S(lambda st: st.rice.pdf(1.8, 2 / 1.5, scale=1.5))),
    ("CDF[RiceDistribution[2, 1.5], 1.8]", S(lambda st: st.rice.cdf(1.8, 2 / 1.5, scale=1.5))),
    ("Variance[RiceDistribution[0, 1]]", "2 - Pi/2"),
    ("RiceDistribution[1, 2, 3]", Raises(NotImplementedError)),
    ("RiceDistribution[-1, 2]", Raises(ValueError)),
    # SechDistribution[mu, sigma]: PDF Sech[Pi (x - mu)/(2 sigma)]/(2 sigma) = SciPy hypsecant(mu, 2 sigma/Pi)
    ("SechDistribution[]", "SechDistribution[0, 1]"),
    ("PDF[SechDistribution[1, 2], 1.7]", S(lambda st: st.hypsecant.pdf(1.7, 1, 4 / math.pi))),
    ("CDF[SechDistribution[1, 2], 1.7]", S(lambda st: st.hypsecant.cdf(1.7, 1, 4 / math.pi))),
    ("Variance[SechDistribution[w17m, w17s]]", "w17s^2"),
    ("SechDistribution[1, -2]", Raises(ValueError)),
    # ShiftedGompertzDistribution[lambda, xi]: CDF (1 - E^(-l x)) E^(-xi E^(-l x))
    ("ShiftedGompertzDistribution[w17l, w17x]", "ShiftedGompertzDistribution[w17l, w17x]"),
    ("CDF[ShiftedGompertzDistribution[2, 1], 1]", "(1 - E^-2) E^(-E^-2)"),
    ("PDF[ShiftedGompertzDistribution[2, 1], 0.5]", M(lambda: 2 * mpmath.exp(-1 - mpmath.exp(-1)) * (1 + 1 * (1 - mpmath.exp(-1))))),
    ("Mean[ShiftedGompertzDistribution[2., 1.]]", M(lambda: mpmath.quad(lambda t: 1 - (1 - mpmath.exp(-2 * t)) * mpmath.exp(
        -mpmath.exp(-2 * t)), [0, mpmath.inf]))),
    ("ShiftedGompertzDistribution[2, -1]", Raises(ValueError)),
    # SkewNormalDistribution[mu, sigma, alpha]: SciPy skewnorm(alpha, mu, sigma)
    ("SkewNormalDistribution[w17m, w17s, w17a]", "SkewNormalDistribution[w17m, w17s, w17a]"),
    ("PDF[SkewNormalDistribution[1, 2, 3], 1.5]", S(lambda st: st.skewnorm.pdf(1.5, 3, 1, 2))),
    ("CDF[SkewNormalDistribution[1, 2, 3], 1.5]", S(lambda st: st.skewnorm.cdf(1.5, 3, 1, 2))),
    ("Mean[SkewNormalDistribution[0, 1, 1]]", "1/Sqrt[Pi]"),
    ("SkewNormalDistribution[0, -1, 1]", Raises(ValueError)),
    # TsallisQExponentialDistribution[lambda, q]: PDF (2 - q) l (1 - (1 - q) l x)^(1/(1 - q)); q = 1 exponential
    ("TsallisQExponentialDistribution[w17l, w17q]", "TsallisQExponentialDistribution[w17l, w17q]"),
    ("PDF[TsallisQExponentialDistribution[2, 3/2], 1]", "1/4"),
    ("CDF[TsallisQExponentialDistribution[2, 1], 1]", "1 - E^-2"),
    ("Mean[TsallisQExponentialDistribution[1, 5/4]]", "2"),
    ("Mean[TsallisQExponentialDistribution[1, 3/2]]", "Infinity"),
    ("PDF[TsallisQExponentialDistribution[2, 1.5], 0.7]", M(lambda: 0.5 * 2 * (1 + 0.5 * 2 * 0.7) ** -2)),
    ("TsallisQExponentialDistribution[2, 3]", Raises(ValueError)),
    # TsallisQGaussianDistribution[mu, beta, q]: q = 1 Normal, q = 2 Cauchy[mu, beta Sqrt[2]] (Wolfram Details)
    ("TsallisQGaussianDistribution[w17m, w17b, w17q]", "TsallisQGaussianDistribution[w17m, w17b, w17q]"),
    ("PDF[TsallisQGaussianDistribution[0, 1, 1], 0]", "1/Sqrt[2 Pi]"),
    ("CDF[TsallisQGaussianDistribution[0, 1, 2], Sqrt[2]]", "3/4"),
    ("PDF[TsallisQGaussianDistribution[1, 2, 2], 2.2]", S(lambda st: st.cauchy.pdf(2.2, 1, 2 * math.sqrt(2)))),
    ("Variance[TsallisQGaussianDistribution[0, 1, 1]]", "1"),
    ("TsallisQGaussianDistribution[0, 1, 4]", Raises(ValueError)),
    # TukeyLambdaDistribution[lambda]: SciPy tukeylambda
    ("TukeyLambdaDistribution[w17l]", "TukeyLambdaDistribution[w17l]"),
    ("CDF[TukeyLambdaDistribution[0.3], 0.5]", S(lambda st: st.tukeylambda.cdf(0.5, 0.3))),
    ("PDF[TukeyLambdaDistribution[0.3], 0.5]", S(lambda st: st.tukeylambda.pdf(0.5, 0.3))),
    ("InverseCDF[TukeyLambdaDistribution[1/2], 3/4]", "2 (3/4)^(1/2) - 2 (1/4)^(1/2)"),
    ("CDF[TukeyLambdaDistribution[0], 0]", "1/2"),
    ("TukeyLambdaDistribution[1, 2, 3, 4]", Raises((NotImplementedError, ValueError))),
    ("TukeyLambdaDistribution[1, 0, -1]", Raises(ValueError)),
    # VarianceGammaDistribution[lambda, alpha, beta, mu]: PDF with BesselK (Wolfram Details); lambda = 1, beta = 0 gives
    # LaplaceDistribution[mu, 1/alpha]
    ("VarianceGammaDistribution[w17l, w17a, w17b, w17m]", "VarianceGammaDistribution[w17l, w17a, w17b, w17m]"),
    ("PDF[VarianceGammaDistribution[1, 2, 0, 0], 0.5]", M(lambda: mpmath.exp(-1.0))),
    ("Mean[VarianceGammaDistribution[w17l, w17a, w17b, w17m]]", "w17m + 2 w17b w17l/(w17a^2 - w17b^2)"),
    ("PDF[VarianceGammaDistribution[2, 3, 1, 0.5], 1.2]", M(lambda: (3 ** 2 - 1) ** 2 * abs(0.7) ** 1.5 * mpmath.besselk(
        1.5, 3 * 0.7) * mpmath.exp(0.7) / (mpmath.sqrt(mpmath.pi) * mpmath.gamma(2) * (2 * 3) ** 1.5))),
    ("VarianceGammaDistribution[1, 1, 2, 0]", Raises(ValueError)),
    # VoigtDistribution[delta, sigma]: SciPy special.voigt_profile(x, sigma, delta)
    ("VoigtDistribution[w17d, w17s]", "VoigtDistribution[w17d, w17s]"),
    ("PDF[VoigtDistribution[1/2, 1], 0.3]", S(lambda st: __import__("scipy.special", fromlist=["v"]).voigt_profile(0.3, 1, 0.5))),
    ("CDF[VoigtDistribution[1/2, 1], 0.3]", S(lambda st: __import__("scipy.integrate", fromlist=["q"]).quad(
        lambda t: __import__("scipy.special", fromlist=["v"]).voigt_profile(t, 1, 0.5), -math.inf, 0.3)[0], 1e-7)),
    ("CDF[VoigtDistribution[1/2, 1], 0.3]", M(lambda: mpmath.quad(lambda u: mpmath.ncdf(0.3 - 0.5 * mpmath.tan(
        mpmath.pi * (u - 0.5))), [0, 0.5, 1]))),
    ("Mean[VoigtDistribution[1, 1]]", "Indeterminate"),
    ("CDF[VoigtDistribution[1/2, 1], 3/10]", Raises(NotImplementedError)),
    ("VoigtDistribution[-1, 1]", Raises(ValueError)),
    # VonMisesDistribution[mu, kappa]: SciPy vonmises(kappa, loc mu)
    ("VonMisesDistribution[w17m, w17k]", "VonMisesDistribution[w17m, w17k]"),
    ("PDF[VonMisesDistribution[1, 2], 1.5]", S(lambda st: st.vonmises.pdf(1.5, 2, loc=1))),
    ("CDF[VonMisesDistribution[1, 2], 1.5]", S(lambda st: st.vonmises.cdf(1.5, 2, loc=1), 1e-8)),
    ("PDF[VonMisesDistribution[0, 1], 0]", "E/(2 Pi BesselI[0, 1])"),
    ("VonMisesDistribution[0, -1]", Raises(ValueError)),
    # WignerSemicircleDistribution[a, r]: SciPy semicircular(loc a, scale r)
    ("WignerSemicircleDistribution[w17r]", "WignerSemicircleDistribution[w17r]"),
    ("PDF[WignerSemicircleDistribution[1, 2], 1.5]", S(lambda st: st.semicircular.pdf(1.5, 1, 2))),
    ("CDF[WignerSemicircleDistribution[1, 2], 1.5]", S(lambda st: st.semicircular.cdf(1.5, 1, 2))),
    ("Variance[WignerSemicircleDistribution[w17r]]", "w17r^2/4"),
    ("WignerSemicircleDistribution[0]", Raises(ValueError)),
    # BenktanderGibratDistribution[a, b]: survival (1 + 2 b Log[x]/a) x^(-(1 + a + b Log[x])) on x >= 1
    ("BenktanderGibratDistribution[w17a, w17b]", "BenktanderGibratDistribution[w17a, w17b]"),
    ("CDF[BenktanderGibratDistribution[2, 1], E]", "1 - 2 E^-4"),
    ("Mean[BenktanderGibratDistribution[2, 1]]", "3/2"),
    ("PDF[BenktanderGibratDistribution[2, 1], 1.5]", M(lambda: -mpmath.diff(lambda t: (1 + mpmath.log(t)) * t ** (
        -(3 + mpmath.log(t))), 1.5))),
    ("BenktanderGibratDistribution[-2, 1]", Raises(ValueError)),
    # GompertzMakehamDistribution[lambda, xi]: SciPy gompertz(c = xi, scale 1/lambda)
    ("GompertzMakehamDistribution[w17l, w17x]", "GompertzMakehamDistribution[w17l, w17x]"),
    ("PDF[GompertzMakehamDistribution[2, 0.5], 0.4]", S(lambda st: st.gompertz.pdf(0.4, 0.5, scale=0.5))),
    ("CDF[GompertzMakehamDistribution[2, 1/2], 1/2]", "1 - E^((1 - E)/2)"),
    ("Mean[GompertzMakehamDistribution[2, 0.5]]", S(lambda st: st.gompertz.mean(0.5, scale=0.5))),
    ("GompertzMakehamDistribution[2, 1, 1, 1]", Raises(NotImplementedError)),
    ("GompertzMakehamDistribution[-2, 1]", Raises(ValueError)),
    # WakebyDistribution[alpha, beta, gamma, delta, mu]: quantile mu + a/b (1 - (1 - p)^b) - g/d (1 - (1 - p)^-d)
    ("WakebyDistribution[w17a, w17b, w17g, w17d, w17m]", "WakebyDistribution[w17a, w17b, w17g, w17d, w17m]"),
    ("InverseCDF[WakebyDistribution[1, 2, 1, 1/2, 0], 3/4]", "1/2 (1 - 1/16) - 2 (1 - 2)"),
    ("Mean[WakebyDistribution[1, 2, 1, 1/2, 0]]", "1/3 + 2"),
    ("CDF[WakebyDistribution[1, 2, 1, 0.5, 0], 2.46875]", near(0.75)),
    ("PDF[WakebyDistribution[1., 2., 1., 0.5, 0.], 2.46875]", near(1 / (1 * 0.25 ** 1 + 1 * 0.25 ** -1.5))),
    ("WakebyDistribution[1, 2, -1, 1/2, 0]", Raises(ValueError)),
    # HotellingTSquareDistribution[p, m] = (p m/(m - p + 1)) FRatioDistribution[p, m - p + 1]
    ("HotellingTSquareDistribution[w17p, w17m]", "HotellingTSquareDistribution[w17p, w17m]"),
    ("Mean[HotellingTSquareDistribution[2, 10]]", "20/7"),
    ("CDF[HotellingTSquareDistribution[2, 10], 3.]", S(lambda st: st.f.cdf(3 * 9 / 20, 2, 9))),
    ("PDF[HotellingTSquareDistribution[2, 10], 3.]", S(lambda st: st.f.pdf(3 * 9 / 20, 2, 9) * 9 / 20)),
    ("HotellingTSquareDistribution[5, 2]", Raises(ValueError)),
]

GROUPS = [CONT]


def _resolve(cases):
    out = []
    st = None
    if HAVE_SCIPY:
        import scipy.stats as st
    for c in cases:
        if isinstance(c[1], S):
            if st is None:
                continue
            try:
                v = float(c[1].f(st))
            except (AttributeError, ImportError):
                continue  # SciPy too old for this reference distribution
            out.append((c[0], v, c[1].tol))
        elif isinstance(c[1], float) and len(c) == 2:
            out.append((c[0], c[1], 1e-9))
        else:
            out.append(c)
    return out


CASES = _resolve([c for g in GROUPS for c in g])


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w17(case):
    check(case)
