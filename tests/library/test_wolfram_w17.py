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


def _landau_pdf(x, mu=0, s=1):
    """Landau density from the integral in the LandauDistribution Details (Sin[2 t] Exp[-t (x - mu)/s - 2/Pi t
    Log[t/s]]), normalised by 1/(Pi s)"""
    return mpmath.quad(lambda t: mpmath.exp(-t * (x - mu) / s - 2 / mpmath.pi * t * mpmath.log(t / s)) * mpmath.sin(2 * t),
                       [0, mpmath.inf]) / (mpmath.pi * s)


def _rayleigh_mix(f):
    """E over s ~ LogNormal(1/2, 1/4) of f(s) (SuzukiDistribution[1/2, 1/4])"""
    return mpmath.quad(lambda z: f(mpmath.exp(0.5 + 0.25 * z)) * mpmath.npdf(z), [-12, 0, 12])  # npdf(12) < 1e-31


def _polya_aeppli(theta, p, k):
    """P(X = k), X = sum of N ~ Poisson(theta) geometric(1 - p) counts on {1, 2, ...}, by direct convolution"""
    from math import comb
    if k == 0:
        return mpmath.exp(-theta)
    return sum(mpmath.exp(-theta) * mpmath.mpf(theta) ** n / mpmath.factorial(n)
               * comb(k - 1, n - 1) * (1 - mpmath.mpf(p)) ** n * mpmath.mpf(p) ** (k - n) for n in range(1, k + 1))


SPECIAL = [
    # StableDistribution: SciPy levy_stable uses Nolan's S1 parametrisation (Wolfram type 1)
    ("StableDistribution[1, w17a, w17b, w17m, w17s]", "StableDistribution[1, w17a, w17b, w17m, w17s]"),
    ("PDF[StableDistribution[1, 1.5, 0.5, 0, 1], 0.7]", S(lambda st: st.levy_stable.pdf(0.7, 1.5, 0.5), 1e-6)),
    ("CDF[StableDistribution[1, 1.5, 0.5, 0, 1], 0.7]", S(lambda st: st.levy_stable.cdf(0.7, 1.5, 0.5), 1e-6)),
    ("PDF[StableDistribution[1, 2, 0, 0, 1], 0.5]", M(lambda: mpmath.npdf(0.5, 0, mpmath.sqrt(2)))),  # alpha = 2: N(0, Sqrt[2] s)
    ("Mean[StableDistribution[1, 1.5, 0.5, 2, 1]]", "2."),
    ("Mean[StableDistribution[1, 1/2, 0, 0, 1]]", "Indeterminate"),
    ("PDF[StableDistribution[1, 3/2, 1/2, 0, 1], 7/10]", Raises(NotImplementedError)),
    ("StableDistribution[1, 3, 0, 0, 1]", Raises(ValueError)),
    ("StableDistribution[2, 1, 0, 0, 1]", Raises(ValueError)),
    # LandauDistribution: the documented integral; SciPy landau (1.15+)
    ("LandauDistribution[]", "LandauDistribution[0, 1]"),
    ("LandauDistribution[w17m, w17s]", "LandauDistribution[w17m, w17s]"),
    ("PDF[LandauDistribution[], 0.5]", M(lambda: _landau_pdf(0.5))),
    ("PDF[LandauDistribution[1, 2], 2.]", M(lambda: _landau_pdf(2, 1, 2))),  # log(t/sigma) term: not a pure rescale
    ("PDF[LandauDistribution[], 0.5]", S(lambda st: st.landau.pdf(0.5), 1e-7)),
    ("CDF[LandauDistribution[], 0.5]", S(lambda st: st.landau.cdf(0.5), 1e-6)),
    ("Mean[LandauDistribution[]]", "Indeterminate"),
    ("LandauDistribution[0, -1]", Raises(ValueError)),
    # SuzukiDistribution[mu, nu]: RayleighDistribution[s] with s ~ LogNormalDistribution[mu, nu]
    ("SuzukiDistribution[w17m, w17n]", "SuzukiDistribution[w17m, w17n]"),
    ("Mean[SuzukiDistribution[w17m, w17n]]", "Sqrt[Pi/2] E^(w17m + w17n^2/2)"),
    ("PDF[SuzukiDistribution[1/2, 1/4], 1.5]", M(lambda: _rayleigh_mix(lambda sg: 1.5 / sg ** 2 * mpmath.exp(-1.5 ** 2 / (2 * sg ** 2))))),
    ("CDF[SuzukiDistribution[1/2, 1/4], 1.5]", M(lambda: _rayleigh_mix(lambda sg: 1 - mpmath.exp(-1.5 ** 2 / (2 * sg ** 2))))),
    ("Variance[SuzukiDistribution[0, 1]]", "2 E^2 - Pi E/2"),
    ("SuzukiDistribution[0, -1]", Raises(ValueError)),
    # BeckmannDistribution: Norm of a binormal vector; rho = 0, equal sigmas, zero means = Rayleigh; mu != 0 = Rice
    ("BeckmannDistribution[w17a, w17b, w17c, w17d]", "BeckmannDistribution[w17a, w17b, w17c, w17d]"),
    ("PDF[BeckmannDistribution[0, 0, 1, 1], 1.2]", M(lambda: 1.2 * mpmath.exp(-0.72))),
    ("CDF[BeckmannDistribution[1, 0, 1, 1], 1.2]", S(lambda st: st.rice.cdf(1.2, 1), 1e-8)),
    ("PDF[BeckmannDistribution[1, 1, 1, 2], 1.5]", M(lambda: mpmath.quad(
        lambda t: 1.5 * mpmath.npdf(1.5 * mpmath.cos(t), 1, 1) * mpmath.npdf(1.5 * mpmath.sin(t), 1, 2), [0, mpmath.pi, 2 * mpmath.pi]))),
    ("Mean[BeckmannDistribution[0., 0., 1., 1.]]", near(math.sqrt(math.pi / 2), 1e-8)),
    ("Variance[BeckmannDistribution[0., 0., 1., 1.]] + Mean[BeckmannDistribution[0., 0., 1., 1.]]^2", near(2.0, 1e-8)),
    ("Mean[BeckmannDistribution[0, 1, 1, 2]]", Raises(NotImplementedError)),  # exact odd moments: no closed form
    ("BeckmannDistribution[0, 0, -1, 1]", Raises(ValueError)),
    # phase-type distributions (exact)
    ("HyperexponentialDistribution[{w17p, 1 - w17p}, {w17a, w17b}]", "HyperexponentialDistribution[{w17p, 1 - w17p}, {w17a, w17b}]"),
    ("PDF[HyperexponentialDistribution[{1/3, 2/3}, {1, 2}], w17x]", "Piecewise[{{E^(-w17x)/3 + 4 E^(-2 w17x)/3, w17x >= 0}}, 0]"),
    ("Mean[HyperexponentialDistribution[{1/3, 2/3}, {1, 2}]]", "2/3"),
    ("CDF[HyperexponentialDistribution[{1/3, 2/3}, {1, 2}], 1]", "1 - 1/(3 E) - 2/(3 E^2)"),
    ("HyperexponentialDistribution[{1/2, 1/3}, {1, 2}]", Raises(ValueError)),
    ("HypoexponentialDistribution[{w17a, w17b}]", "HypoexponentialDistribution[{w17a, w17b}]"),
    ("CDF[HypoexponentialDistribution[{1, 2}], 1]", "1 - 2/E + 1/E^2"),
    ("PDF[HypoexponentialDistribution[{2, 2}], 1]", "4/E^2"),  # = ErlangDistribution[2, 2]
    ("Variance[HypoexponentialDistribution[{1, 2, 4}]]", "1 + 1/4 + 1/16"),
    ("HypoexponentialDistribution[{1, -2}]", Raises(ValueError)),
    ("CoxianDistribution[{w17p}, {w17a, w17b}]", "CoxianDistribution[{w17p}, {w17a, w17b}]"),
    ("Mean[CoxianDistribution[{1/2}, {1, 2}]]", "5/4"),
    ("CDF[CoxianDistribution[{1}, {1, 2}], 1]", "1 - 2/E + 1/E^2"),
    ("PDF[CoxianDistribution[{1/2}, {1, 2}], 1]", "1/(2 E) + (2/E - 2/E^2)/2"),
    ("CoxianDistribution[{1/2, 1/2}, {1, 2}]", Raises(ValueError)),
    # JohnsonDistribution: SciPy johnsonsu / johnsonsb (a = gamma, b = delta, loc = mu, scale = sigma)
    ("JohnsonDistribution[\"SU\", w17g, w17d, w17m, w17s]", "JohnsonDistribution[\"SU\", w17g, w17d, w17m, w17s]"),
    ("PDF[JohnsonDistribution[\"SU\", 1, 2, 0.5, 1.5], 0.3]", S(lambda st: st.johnsonsu.pdf(0.3, 1, 2, 0.5, 1.5))),
    ("CDF[JohnsonDistribution[\"SB\", 1, 2, 0, 3], 1.2]", S(lambda st: st.johnsonsb.cdf(1.2, 1, 2, 0, 3))),
    ("CDF[JohnsonDistribution[\"SN\", 0, 1, 0, 1], 0]", "1/2"),
    ("CDF[JohnsonDistribution[\"SL\", 0, 1, 0, 1], 1]", "1/2"),
    ("Mean[JohnsonDistribution[\"SL\", 0, 1, 0, 1]]", "Sqrt[E]"),
    ("JohnsonDistribution[\"SX\", 0, 1, 0, 1]", Raises(ValueError)),
    ("JohnsonDistribution[\"SU\", 0, -1, 0, 1]", Raises(ValueError)),
]

DISC = [
    ("BenfordDistribution[w17b]", "BenfordDistribution[w17b]"),
    ("PDF[BenfordDistribution[10], 1]", "Log[2]/Log[10]"),
    ("CDF[BenfordDistribution[10], 3]", "Log[4]/Log[10]"),
    ("N[Mean[BenfordDistribution[10]]]", M(lambda: sum(k * mpmath.log10(1 + 1 / mpmath.mpf(k)) for k in range(1, 10)))),
    ("BenfordDistribution[1]", Raises(ValueError)),
    # BetaBinomialDistribution[alpha, beta, n]: SciPy betabinom(n, alpha, beta)
    ("BetaBinomialDistribution[w17a, w17b, w17n]", "BetaBinomialDistribution[w17a, w17b, w17n]"),
    ("Mean[BetaBinomialDistribution[w17a, w17b, w17n]]", "w17n w17a/(w17a + w17b)"),
    ("PDF[BetaBinomialDistribution[2, 3, 5], 2]", S(lambda st: st.betabinom.pmf(2, 5, 2, 3))),
    ("PDF[BetaBinomialDistribution[1, 1, 4], 3]", "1/5"),
    ("CDF[BetaBinomialDistribution[2, 3, 5], 2.5]", S(lambda st: st.betabinom.cdf(2, 5, 2, 3))),
    ("BetaBinomialDistribution[2, 3, -5]", Raises(ValueError)),
    # BetaNegativeBinomialDistribution[alpha, beta, n]: SciPy betanbinom(n, alpha, beta)
    ("BetaNegativeBinomialDistribution[w17a, w17b, w17n]", "BetaNegativeBinomialDistribution[w17a, w17b, w17n]"),
    ("PDF[BetaNegativeBinomialDistribution[3, 2, 4], 2]", S(lambda st: st.betanbinom.pmf(2, 4, 3, 2))),
    ("Mean[BetaNegativeBinomialDistribution[3, 2, 4]]", "4"),
    ("PDF[BetaNegativeBinomialDistribution[1, 1, 1], 1]", "1/6"),
    ("BetaNegativeBinomialDistribution[-3, 2, 4]", Raises(ValueError)),
    # BorelTannerDistribution[alpha, n]
    ("BorelTannerDistribution[w17a, w17n]", "BorelTannerDistribution[w17a, w17n]"),
    ("PDF[BorelTannerDistribution[1/2, 2], 3]", "2/3 E^(-3/2) (3/2)"),
    ("Mean[BorelTannerDistribution[w17a, w17n]]", "w17n/(1 - w17a)"),
    ("Total[Table[PDF[BorelTannerDistribution[0.3, 2], w17k], {w17k, 2, 80}]]", near(1.0, 1e-12)),
    ("BorelTannerDistribution[2, 1]", Raises(ValueError)),
    # Fisher / Wallenius noncentral hypergeometric: SciPy nchypergeom_fisher / nchypergeom_wallenius (M, n, N, odds)
    ("FisherHypergeometricDistribution[w17n, w17s, w17t, w17w]", "FisherHypergeometricDistribution[w17n, w17s, w17t, w17w]"),
    ("PDF[FisherHypergeometricDistribution[4, 5, 12, 2], 2]", S(lambda st: st.nchypergeom_fisher.pmf(2, 12, 5, 4, 2))),
    ("Mean[FisherHypergeometricDistribution[4, 5, 12, 2.]]", S(lambda st: st.nchypergeom_fisher.mean(12, 5, 4, 2))),
    ("PDF[FisherHypergeometricDistribution[2, 2, 4, 1], 1]", "2/3"),
    ("FisherHypergeometricDistribution[4, 13, 12, 2]", Raises(ValueError)),
    ("WalleniusHypergeometricDistribution[w17n, w17s, w17t, w17w]", "WalleniusHypergeometricDistribution[w17n, w17s, w17t, w17w]"),
    ("PDF[WalleniusHypergeometricDistribution[4, 5, 12, 2.], 2]", S(lambda st: st.nchypergeom_wallenius.pmf(2, 12, 5, 4, 2), 1e-8)),
    ("CDF[WalleniusHypergeometricDistribution[4, 5, 12, 2.], 2]", S(lambda st: st.nchypergeom_wallenius.cdf(2, 12, 5, 4, 2), 1e-8)),
    ("PDF[WalleniusHypergeometricDistribution[2, 2, 4, 1], 1]", "2/3"),
    ("WalleniusHypergeometricDistribution[4, 5, 12, -2]", Raises(ValueError)),
    # LogSeriesDistribution[theta]: SciPy logser
    ("LogSeriesDistribution[w17t]", "LogSeriesDistribution[w17t]"),
    ("PDF[LogSeriesDistribution[1/2], 2]", "1/(8 Log[2])"),
    ("Mean[LogSeriesDistribution[0.4]]", S(lambda st: st.logser.mean(0.4))),
    ("CDF[LogSeriesDistribution[0.4], 3]", S(lambda st: st.logser.cdf(3, 0.4))),
    ("LogSeriesDistribution[1]", Raises(ValueError)),
    # PascalDistribution[n, p]: number of trials = SciPy nbinom(n, p) + n
    ("PascalDistribution[w17n, w17p]", "PascalDistribution[w17n, w17p]"),
    ("Mean[PascalDistribution[w17n, w17p]]", "w17n/w17p"),
    ("PDF[PascalDistribution[3, 0.4], 7]", S(lambda st: st.nbinom.pmf(4, 3, 0.4))),
    ("CDF[PascalDistribution[3, 2/5], 3]", "8/125"),
    ("PascalDistribution[3, 2]", Raises(ValueError)),
    # PoissonConsulDistribution[mu, lambda] (generalised Poisson)
    ("PoissonConsulDistribution[w17m, w17l]", "PoissonConsulDistribution[w17m, w17l]"),
    ("PDF[PoissonConsulDistribution[2, 1/2], 2]", "2 3 E^-3/2"),
    ("Mean[PoissonConsulDistribution[w17m, w17l]]", "w17m/(1 - w17l)"),
    ("Total[Table[PDF[PoissonConsulDistribution[1.5, 0.3], w17k], {w17k, 0, 100}]]", near(1.0, 1e-12)),
    ("PoissonConsulDistribution[2, 3/2]", Raises(ValueError)),
    # PolyaAeppliDistribution[theta, p]: Poisson sum of geometrics (direct convolution)
    ("PolyaAeppliDistribution[w17t, w17p]", "PolyaAeppliDistribution[w17t, w17p]"),
    ("PDF[PolyaAeppliDistribution[2, 1/3], 3]", M(lambda: _polya_aeppli(2, mpmath.mpf(1) / 3, 3))),
    ("PDF[PolyaAeppliDistribution[2.5, 0.4], 5]", M(lambda: _polya_aeppli(2.5, 0.4, 5))),
    ("Mean[PolyaAeppliDistribution[w17t, w17p]]", "w17t/(1 - w17p)"),
    ("PDF[PolyaAeppliDistribution[2, 1/3], 0]", "E^-2"),
    ("PolyaAeppliDistribution[2, 1]", Raises(ValueError)),
    # SkellamDistribution: SciPy skellam
    ("SkellamDistribution[w17a, w17b]", "SkellamDistribution[w17a, w17b]"),
    ("Mean[SkellamDistribution[w17a, w17b]]", "w17a - w17b"),
    ("PDF[SkellamDistribution[2, 3], -1]", S(lambda st: st.skellam.pmf(-1, 2, 3))),
    ("CDF[SkellamDistribution[2., 3.], 1]", S(lambda st: st.skellam.cdf(1, 2, 3))),
    ("PDF[SkellamDistribution[2, 3.], 2]", S(lambda st: st.skellam.pmf(2, 2, 3))),
    ("SkellamDistribution[-2, 3]", Raises(ValueError)),
    # WaringYuleDistribution[alpha] = SciPy yulesimon(alpha) shifted by -1
    ("WaringYuleDistribution[w17a]", "WaringYuleDistribution[w17a]"),
    ("PDF[WaringYuleDistribution[2.5], 3]", S(lambda st: st.yulesimon.pmf(4, 2.5))),
    ("PDF[WaringYuleDistribution[2, 3], 0]", "2/5"),
    ("Mean[WaringYuleDistribution[3]]", "1/2"),
    ("Mean[WaringYuleDistribution[1]]", "Infinity"),
    ("WaringYuleDistribution[-1]", Raises(ValueError)),
    # ZipfDistribution[rho] = SciPy zipf(rho + 1); [n, rho] = zipfian(rho + 1, n)
    ("ZipfDistribution[w17r]", "ZipfDistribution[w17r]"),
    ("PDF[ZipfDistribution[1], 2]", "3/(2 Pi^2)"),  # 2^-2/Zeta[2]
    ("PDF[ZipfDistribution[1.5], 3]", S(lambda st: st.zipf.pmf(3, 2.5))),
    ("CDF[ZipfDistribution[5, 1.5], 3]", S(lambda st: st.zipfian.cdf(3, 2.5, 5))),
    ("Mean[ZipfDistribution[1]]", "Infinity"),
    ("ZipfDistribution[0]", Raises(ValueError)),
]


# ───────────────────────────── multivariate ─────────────────────────────

def _mvpois_cdf(m0, m1, m2, a, b):
    """independent reference: P[Y0 + Y1 <= a, Y0 + Y2 <= b] by direct summation"""
    def po(m, k):
        return mpmath.exp(-m) * mpmath.mpf(m) ** k / mpmath.factorial(k)
    return float(sum(po(m0, y0) * po(m1, y1) * po(m2, y2) for y0 in range(min(a, b) + 1)
                     for y1 in range(a - y0 + 1) for y2 in range(b - y0 + 1)))


MULTI = [
    # BinormalDistribution: SciPy multivariate_normal; orthant probability 1/4 + ArcSin[rho]/(2 Pi)
    ("BinormalDistribution[w17r]", "BinormalDistribution[w17r]"),
    ("PDF[BinormalDistribution[{1, 2}, {1, 2}, 1/2], {1, 2}]", "Sqrt[3]/(6 Pi)"),  # 1/(2 Pi s1 s2 Sqrt[1 - rho^2])
    ("PDF[BinormalDistribution[{1, 2}, {1, 2}, 1/2], {1.5, 1.}]",
     S(lambda st: st.multivariate_normal.pdf([1.5, 1.0], [1, 2], [[1, 1], [1, 4]]))),
    ("CDF[BinormalDistribution[{0, 0}, {1, 1}, 0.5], {0., 0.}]", 1 / 3, 1e-7),
    ("Covariance[BinormalDistribution[{1, 2}, {1, 2}, 1/2]]", "{{1, 1}, {1, 4}}"),
    ("Mean[BinormalDistribution[{1, 2}, {1, 2}, 1/2]]", "{1, 2}"),
    ("MarginalDistribution[BinormalDistribution[{1, 2}, {1, 3}, 1/2], 2]", "NormalDistribution[2, 3]"),
    ("BinormalDistribution[2]", Raises(ValueError)),
    ("BinormalDistribution[{1, 2}, {1, -2}, 1/2]", Raises(ValueError)),
    # DirichletDistribution: Gamma[6]/(Gamma[1] Gamma[2] Gamma[3]) x2 x3^2; SciPy dirichlet
    ("DirichletDistribution[{w17a, w17b}]", "DirichletDistribution[{w17a, w17b}]"),
    ("PDF[DirichletDistribution[{1, 2, 3}], {1/4, 1/4}]", "15/4"),
    ("PDF[DirichletDistribution[{1.5, 2, 3}], {0.2, 0.3}]",
     S(lambda st: st.dirichlet.pdf([0.2, 0.3, 0.5], [1.5, 2, 3]))),
    ("Mean[DirichletDistribution[{1, 2, 3}]]", "{1/6, 1/3}"),
    ("Covariance[DirichletDistribution[{1, 2, 3}]]", "{{5/252, -1/126}, {-1/126, 2/63}}"),  # a_i(a0 - a_i)/(a0^2 (a0 + 1))
    ("MarginalDistribution[DirichletDistribution[{1, 2, 3}], 1]", "BetaDistribution[1, 5]"),
    ("CDF[DirichletDistribution[{2, 3}], {1/2}]", "11/16"),  # BetaRegularized[1/2, 2, 3]
    ("CDF[DirichletDistribution[{1, 2, 3}], {1/2, 1/2}]", Raises(NotImplementedError)),
    ("DirichletDistribution[{1, -2, 3}]", Raises(ValueError)),
    ("Total[RandomVariate[DirichletDistribution[{1, 2, 3}], 20], {2}]", nums(20, 0, 1)),
    # MultivariateTDistribution: SciPy multivariate_t
    ("MultivariateTDistribution[{{1, 0}, {0, 1}}, w17n]", "MultivariateTDistribution[{{1, 0}, {0, 1}}, w17n]"),
    ("PDF[MultivariateTDistribution[{{1, 0}, {0, 1}}, 3], {0, 0}]", "1/(2 Pi)"),  # Gamma[5/2]/(Gamma[3/2] 3 Pi)
    ("PDF[MultivariateTDistribution[{1, 2}, {{2, 1/2}, {1/2, 1}}, 5], {0.5, 1.5}]",
     S(lambda st: st.multivariate_t.pdf([0.5, 1.5], [1, 2], [[2, 0.5], [0.5, 1]], df=5))),
    ("Covariance[MultivariateTDistribution[{{2, 1/2}, {1/2, 1}}, 5]]", "{{10/3, 5/6}, {5/6, 5/3}}"),  # nu/(nu - 2) Sigma
    ("Mean[MultivariateTDistribution[{1, 2}, {{2, 1/2}, {1/2, 1}}, 5]]", "{1, 2}"),
    ("MarginalDistribution[MultivariateTDistribution[{{4, 1/2}, {1/2, 1}}, 5], 1]", "StudentTDistribution[0, 2, 5]"),
    ("MultivariateTDistribution[{{1, 2}, {2, 1}}, 3]", Raises(ValueError)),
    # LogMultinormalDistribution: PDF = Multinormal PDF at Log[x] / Product[x]; E[X_i] = E^(mu_i + S_ii/2)
    ("LogMultinormalDistribution[{w17a, w17b}, {{1, 0}, {0, 1}}]",
     "LogMultinormalDistribution[{w17a, w17b}, {{1, 0}, {0, 1}}]"),
    ("Mean[LogMultinormalDistribution[{0, 1}, {{1, 0}, {0, 2}}]]", "{Sqrt[E], E^2}"),
    ("PDF[LogMultinormalDistribution[{0, 1}, {{1, 1/2}, {1/2, 2}}], {1.2, 3.}]",
     S(lambda st: st.multivariate_normal.pdf([math.log(1.2), math.log(3.0)], [0, 1], [[1, 0.5], [0.5, 2]]) / 3.6)),
    ("MarginalDistribution[LogMultinormalDistribution[{0, 1}, {{1, 0}, {0, 2}}], 2]",
     "LogNormalDistribution[1, Sqrt[2]]"),
    ("Covariance[LogMultinormalDistribution[{0, 0}, {{1, 0}, {0, 1}}]]", "{{E^2 - E, 0}, {0, E^2 - E}}"),
    ("LogMultinormalDistribution[{0, 1}, {{1, 2}, {2, 1}}]", Raises(ValueError)),
    # MultivariateHypergeometricDistribution: Product[Binomial[m_i, x_i]]/Binomial[12, 4]
    ("MultivariateHypergeometricDistribution[w17n, {3, 4, 5}]", "MultivariateHypergeometricDistribution[w17n, {3, 4, 5}]"),
    ("PDF[MultivariateHypergeometricDistribution[4, {3, 4, 5}], {1, 1, 2}]", "8/33"),
    ("PDF[MultivariateHypergeometricDistribution[4, {3, 4, 5}], {1, 1, 1}]", "0"),
    ("PDF[MultivariateHypergeometricDistribution[4, {3, 4, 5}], {0, 2, 2}]",
     S(lambda st: st.multivariate_hypergeom.pmf([0, 2, 2], [3, 4, 5], 4))),
    ("Mean[MultivariateHypergeometricDistribution[4, {3, 4, 5}]]", "{1, 4/3, 5/3}"),
    ("Covariance[MultivariateHypergeometricDistribution[4, {3, 4, 5}]][[1, 2]]", "-8/33"),  # -n m1 m2 (N-n)/(N^2 (N-1))
    ("MarginalDistribution[MultivariateHypergeometricDistribution[4, {3, 4, 5}], 1]", "HypergeometricDistribution[4, 3, 12]"),
    ("MultivariateHypergeometricDistribution[20, {3, 4, 5}]", Raises(ValueError)),
    ("MultivariateHypergeometricDistribution[-1, {3, 4, 5}]", Raises(ValueError)),
    # MultivariatePoissonDistribution: {Y0 + Y1, Y0 + Y2}
    ("MultivariatePoissonDistribution[w17m, {2, 3}]", "MultivariatePoissonDistribution[w17m, {2, 3}]"),
    ("PDF[MultivariatePoissonDistribution[1, {2, 3}], {1, 2}]", "12/E^6"),  # 9 E^-6 (Y0 = 0) + 3 E^-6 (Y0 = 1)
    ("CDF[MultivariatePoissonDistribution[1, {2, 3}], {2, 3}]", _mvpois_cdf(1, 2, 3, 2, 3)),
    ("Covariance[MultivariatePoissonDistribution[1, {2, 3}]]", "{{3, 1}, {1, 4}}"),
    ("MarginalDistribution[MultivariatePoissonDistribution[1, {2, 3}], 2]", "PoissonDistribution[4]"),
    ("MultivariatePoissonDistribution[-1, {2, 3}]", Raises(ValueError)),
    # NegativeMultinomialDistribution: Gamma[4]/(Gamma[2]) (1/2)^2 (1/4)^2
    ("NegativeMultinomialDistribution[w17n, {1/4, 1/4}]", "NegativeMultinomialDistribution[w17n, {1/4, 1/4}]"),
    ("PDF[NegativeMultinomialDistribution[2, {1/4, 1/4}], {1, 1}]", "3/32"),
    ("Mean[NegativeMultinomialDistribution[2, {1/4, 1/4}]]", "{1, 1}"),  # n p_i/p0
    ("Covariance[NegativeMultinomialDistribution[2, {1/4, 1/4}]]", "{{3/2, 1/2}, {1/2, 3/2}}"),
    ("MarginalDistribution[NegativeMultinomialDistribution[2, {1/4, 1/4}], 1]", "NegativeBinomialDistribution[2, 2/3]"),
    ("NegativeMultinomialDistribution[2, {1/2, 1/2}]", Raises(ValueError)),
    # CategoricalDistribution
    ("CategoricalDistribution[{w17x, w17y, w17z}]", "CategoricalDistribution[{w17x, w17y, w17z}]"),
    ("PDF[CategoricalDistribution[{w17x, w17y, w17z}, {1, 2, 3}], w17y]", "1/3"),
    ("PDF[CategoricalDistribution[{w17x, w17y, w17z}], w17z]", "1/3"),
    ("PDF[CategoricalDistribution[{w17x, w17y, w17z}], w17q]", "0"),
    ("RandomVariate[CategoricalDistribution[{w17x, w17y}, {1, 0}], 3]", "{w17x, w17x, w17x}"),
    ("Mean[CategoricalDistribution[{1, 2, 3}]]", Raises(TypeError)),
    ("CategoricalDistribution[{w17x, w17y}, {1, -1}]", Raises(ValueError)),
    # MarginalDistribution
    ("MarginalDistribution[w17d, 1]", "MarginalDistribution[w17d, 1]"),
    ("MarginalDistribution[MultinormalDistribution[{1, 2}, {{1, 0}, {0, 4}}], 2]", "NormalDistribution[2, 2]"),
    ("MarginalDistribution[MultinomialDistribution[5, {1/2, 1/3, 1/6}], 2]", "BinomialDistribution[5, 1/3]"),
    ("MarginalDistribution[ProductDistribution[NormalDistribution[], ExponentialDistribution[2], PoissonDistribution[3]], "
     "{1, 3}]", "ProductDistribution[NormalDistribution[0, 1], PoissonDistribution[3]]"),
    # check: marginal means agree with Mean[dist]
    ("Mean[MarginalDistribution[MultivariatePoissonDistribution[1, {2, 3}], 1]] == "
     "Mean[MultivariatePoissonDistribution[1, {2, 3}]][[1]]", "True"),
    ("MarginalDistribution[NormalDistribution[], 1]", Raises(ValueError)),
    ("MarginalDistribution[MultinormalDistribution[{1, 2}, {{1, 0}, {0, 4}}], 3]", Raises(ValueError)),
]

GROUPS = [CONT, SPECIAL, DISC, MULTI]


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
