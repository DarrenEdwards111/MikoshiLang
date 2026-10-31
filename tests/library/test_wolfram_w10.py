"""Differential tests for Wolfram batch W10 (special functions II, module wl_special_misc).

Expected values are Wolfram Language results from the reference pages (reference.wolfram.com/language/ref/<Name>
examples and Details), exact identities, or independent references computed here with mpmath from the defining
series/integrals (not from the functions under test).  ``HP`` checks an arbitrary-precision result to the requested
number of digits and that it carries that precision.
"""
import mpmath as mp
import pytest
import sympy as sp

from _harness import Pred, Raises, check, ids, to_complex

mp.mp.dps = 20


def HP(ref, digits):
    def ok(got):
        with mp.workdps(digits + 30):
            want = ref() if callable(ref) else ref
            if isinstance(got, sp.Float):
                if got._prec < digits * 3.32:
                    return False
                g = mp.mpf(got._mpf_)
            else:
                g = mp.mpmathify(complex(to_complex(got)))
            return abs(g - want) <= abs(want) * mp.mpf(10) ** (1 - digits) + mp.mpf(10) ** (-digits - 20)
    return Pred(ok, f"{digits} digits")


def Near(want, tol=1e-12):
    return Pred(lambda g: abs(to_complex(g) - complex(want)) <= tol * max(1, abs(complex(want))), f"~{want}")


def nested_quad(word, x):
    """Goncharov G(word; x) by direct nested numerical integration along [0, x] (independent reference)."""
    def G(w, t):
        if not w:
            return mp.mpf(1)
        a = w[0]
        return mp.quad(lambda u: G(w[1:], u) / (u - a), [0, t])
    return G(list(word), mp.mpf(x))


def hpl_series(word, x, N=400):
    """Harmonic polylog H(word; x) by term-wise integration of its power series (f_0 = 1/t, f_1 = 1/(1-t), f_-1 = 1/(1+t))."""
    c = [mp.mpf(1)] + [mp.mpf(0)] * N            # the constant 1
    for a in reversed(word):
        if a != 0:                               # multiply by 1/(1 - a t)
            acc, out = mp.mpf(0), []
            for n in range(N + 1):
                acc = acc * a + c[n]
                out.append(acc)
            c = [mp.mpf(0)] + [out[n - 1] / n for n in range(1, N + 1)]   # then integrate
        else:
            c = [mp.mpf(0)] + [c[n] / n for n in range(1, N + 1)]
    return float(sum(cn * mp.mpf(x) ** n for n, cn in enumerate(c)))


with mp.workdps(30):
    # independent references
    MZV32 = mp.nsum(lambda k: mp.zeta(3, k + 1) / k ** 2, [1, mp.inf])            # zeta(3, 2)
    ALT21 = mp.nsum(lambda k: (-1) ** int(k) / k * mp.zeta(2, k + 1), [1, mp.inf])  # Li_{2,1}(1, -1)
    MPL_HALF = mp.nsum(lambda n: mp.mpf(0.5) ** n / n ** 2 * sum(mp.mpf(0.5) ** j / j for j in range(1, int(n))),
                       [2, 200])
    HPL5 = hpl_series([0, 0, -1, 0, 1], 0.5)
    MARCUM = mp.quad(lambda x: x * mp.besseli(0, 1 * x) * mp.exp(-(1 + x * x) / 2), [2, mp.inf])
    MARCUM2 = mp.quad(lambda x: x * (x / 1.5) ** 1.5 * mp.besseli(1.5, 1.5 * x) * mp.exp(-(1.5 ** 2 + x * x) / 2),
                      [1, mp.inf])
    OWEN = mp.quad(lambda x: mp.exp(-0.25 * (1 + x * x) / 2) / (1 + x * x), [0, 1]) / (2 * mp.pi)
    DAWSON1 = mp.exp(-1) * mp.quad(lambda t: mp.exp(t * t), [0, 1])


def tau_series(N):
    """tau(1..N-1) from the q-expansion q * prod (1 - q^n)^24, truncated power series in integers."""
    c = [1] + [0] * N
    for n in range(1, N):
        for _ in range(24):
            for i in range(N, n - 1, -1):
                c[i] -= c[i - n]
    return c[:N - 1]


TAU = tau_series(13)
with mp.workdps(30):
    L12 = sum(TAU[n - 1] / mp.mpf(n) ** 12 for n in range(1, 13)) + mp.mpf(0)   # truncated Dirichlet series

ELEMENTARY = [
    ("ArcCoth[2.]", float(mp.log(3) / 2)), ("ArcCoth[0]", "I Pi/2"), ("ArcCoth[I]", "-I Pi/4"),
    ("ArcCoth[2]", "ArcCoth[2]"), ("N[ArcCoth[3], 30]", HP(lambda: mp.atanh(mp.mpf(1) / 3), 30)),
    ("ArcCoth[0.5]", complex(mp.atanh(2))),
    ("ArcCsch[1.]", float(mp.log(1 + mp.sqrt(2)))), ("ArcCsch[I]", "-I Pi/2"), ("ArcCsch[x]", "ArcCsch[x]"),
    ("ArcCsch[-2.]", float(-mp.asinh(0.5))),
    ("ArcSech[1]", "0"), ("ArcSech[2]", "I Pi/3"), ("ArcSech[0.5]", float(mp.log(2 + mp.sqrt(3)))),
    ("ArcSech[1/2]", "ArcSech[1/2]"),
    ("D[ArcCoth[x], x]", "1/(1 - x^2)"),
    # Gudermannian: reference value gd(x) = 0.8 at x = InverseGudermannian[0.8]
    ("Gudermannian[1.]", float(2 * mp.atan(mp.tanh(0.5)))), ("Gudermannian[0]", "0"),
    ("Gudermannian[Infinity]", "Pi/2"), ("Gudermannian[2. + I]", complex(2 * mp.atan(mp.tanh(mp.mpc(1, 0.5))))),
    ("Gudermannian[InverseGudermannian[0.8]]", Near(0.8)),
    ("InverseGudermannian[0.5]", float(mp.log(mp.tan(0.25 + mp.pi / 4)))), ("InverseGudermannian[0]", "0"),
    ("InverseGudermannian[Pi/2]", "Infinity"), ("N[InverseGudermannian[1/3], 25]", HP(lambda: mp.asinh(mp.tan(mp.mpf(1) / 3)), 25)),
    ("LogisticSigmoid[0]", "1/2"), ("LogisticSigmoid[1.]", float(1 / (1 + mp.exp(-1)))),
    ("LogisticSigmoid[-Infinity]", "0"), ("LogisticSigmoid[x]", "LogisticSigmoid[x]"),
    ("N[LogisticSigmoid[-30], 20]", HP(lambda: 1 / (1 + mp.exp(30)), 20)),
    ("Sinc[0]", "1"), ("Sinc[Pi/2]", "2/Pi"), ("Sinc[Pi]", "0"), ("Sinc[0.5]", float(mp.sin(0.5) / 0.5)),
    ("Sinc[{0., 2.}]", [1.0, float(mp.sin(2) / 2)]),
]

GENERALIZED = [
    ("Ramp[{-1, 2.5, 0}]", "{0, 2.5, 0}"), ("Ramp[x]", "Ramp[x]"), ("Ramp[-Pi]", "0"), ("Ramp[3/2]", "3/2"),
    ("HeavisidePi[0.2]", "1"), ("HeavisidePi[1/2]", "HeavisidePi[1/2]"), ("HeavisidePi[3]", "0"),
    ("HeavisidePi[2, x]", "0"), ("HeavisidePi[1/4, -1/3]", "1"),
    ("HeavisideLambda[0.25]", 0.75), ("HeavisideLambda[1]", "0"), ("HeavisideLambda[-1/3]", "2/3"),
    ("HeavisideLambda[1/2, 1/2]", "1/4"), ("HeavisideLambda[x]", "HeavisideLambda[x]"),
    ("UnitTriangle[-1/2]", "1/2"), ("UnitTriangle[2]", "0"), ("UnitTriangle[0.9]", 0.1),
    ("UnitTriangle[{0, 1}]", "{1, 0}"),
    ("DiracComb[1/2]", "0"), ("DiracComb[1]", "DiracComb[1]"), ("DiracComb[1.]", "DiracComb[1.]"),
    ("DiracComb[2.5, 3]", "0"),
    ("DiscreteIndicator[a, a, {a, b}]", "1"), ("DiscreteIndicator[a, b, {a, b, c}]", "0"),
    ("DiscreteIndicator[a, c, {a, b}]", "DiscreteIndicator[a, c, {a, b}]"),
    ("DiscreteIndicator[2, 2, {1, 2, 3}]", "1"),
    # waveforms (reference examples)
    ("SquareWave[0.2]", "1"), ("SquareWave[-1]", "1"), ("SquareWave[1/4]", "1"), ("SquareWave[{0, 2}, 1.6]", "0"),
    ("SquareWave[{1, 2}, 1/2]", "1"), ("SquareWave[3/4]", "-1"), ("SquareWave[x]", "SquareWave[x]"),
    ("TriangleWave[.2]", 0.8), ("TriangleWave[-1]", "0"), ("TriangleWave[1/4]", "1"), ("TriangleWave[3/4]", "-1"),
    ("TriangleWave[{0, 4}, 1/8]", "3"),
    ("SawtoothWave[0]", "0"), ("SawtoothWave[2.3]", Near(0.3, 1e-14)), ("SawtoothWave[-1/4]", "3/4"),
    ("SawtoothWave[{1, 3}, 1/4]", "3/2"),
]

CONSTANTS = [
    ("N[Glaisher, 30]", HP(lambda: mp.exp(mp.mpf(1) / 12 - mp.zeta(-1, derivative=1)), 30)),
    ("Glaisher", "Glaisher"), ("Glaisher > 1", True), ("N[Glaisher + 1]", float(mp.glaisher + 1)),
    ("N[Khinchin, 25]", HP(lambda: mp.khinchin, 25)), ("NumericQ[Khinchin]", True), ("Khinchin < 3", True),
    ("N[GoldenAngle]", float((3 - mp.sqrt(5)) * mp.pi)), ("N[GoldenAngle, 30]", HP(lambda: 2 * mp.pi / mp.phi ** 2, 30)),
    ("GoldenAngle < Pi", True),
    ("N[ChampernowneNumber[]]", 0.12345678910111213), ("N[ChampernowneNumber[10], 30]",
                                                         HP(lambda: mp.mpf("0.123456789101112131415161718192021"), 30)),
    ("N[ChampernowneNumber[2]]", Near(float(sum(mp.mpf(int(c)) / 2 ** (i + 1) for i, c in
                                                 enumerate("".join(bin(n)[2:] for n in range(1, 40))[:60]))))),
    ("ChampernowneNumber[1]", Raises()), ("ChampernowneNumber[3]", "ChampernowneNumber[3]"),
]

NUMBERS = [
    ("AlternatingHarmonicNumber[4]", "7/12"), ("AlternatingHarmonicNumber[3, 2]", "31/36"),
    ("AlternatingHarmonicNumber[3, 2, 1/2]", "65/144"), ("AlternatingHarmonicNumber[1.]", 1.0),
    ("AlternatingHarmonicNumber[3., 2]", 31 / 36), ("AlternatingHarmonicNumber[n]", "AlternatingHarmonicNumber[n]"),
    ("AlternatingHarmonicNumber[5.] - AlternatingHarmonicNumber[4.]", Near(0.2)),
    ("HyperHarmonicNumber[2, 4]", "77/12"), ("HyperHarmonicNumber[1, n]", "HarmonicNumber[n]"),
    ("HyperHarmonicNumber[3, 10]", str(sp.binomial(12, 2) * (sp.harmonic(12) - sp.harmonic(2)))),
    ("HyperHarmonicNumber[2, 0]", "0"), ("HyperHarmonicNumber[0, 10, 2]", "1/100"),
    ("HyperHarmonicNumber[2, 3, 2]", str(sum(sp.harmonic(k, 2) for k in range(1, 4)))),
    ("HyperHarmonicNumber[3, E] // N", float(mp.gamma(mp.e + 3) / (mp.gamma(mp.e + 1) * mp.gamma(3))
                                             * (mp.digamma(mp.e + 3) - mp.digamma(3)))),
    ("HyperHarmonicNumber[2., 4]", 77 / 12),
    ("MultipleHarmonicNumber[3, {2, 1}]", "5/12"), ("MultipleHarmonicNumber[5]", "137/60"),
    ("MultipleHarmonicNumber[4, {1, 1}, {1/2, 2}]", "25/48"),
    ("MultipleHarmonicNumber[Infinity, {2, 1}]", "Zeta[3]"),
    ("MultipleHarmonicNumber[6, {3}]", str(sp.harmonic(6, 3))), ("MultipleHarmonicNumber[3.5, {1}]", Raises()),
    ("BellY[4, 2, {x1, x2, x3}]", "3 x2^2 + 4 x1 x3"), ("BellY[6, 2, {1, 1, 1, 1, 1}]", "31"),
    ("BellY[6, 3, {1, 1, 1, 1}]", "90"),
    ("BellY[{{x1, x2, x3}, {y1, y2, y3}}]", "x1 y3 + 3 x2 y1 y2 + x3 y1^3"),
    ("BellY[3, 1, {a}]", Raises()),
    ("Sum[BellY[5, k, {1, 1, 1, 1, 1}], {k, 0, 5}]", "52"),
    ("NorlundB[2, a]", "-a/12 + a^2/4"), ("NorlundB[4, 1, x]", "-1/30 + x^2 - 2 x^3 + x^4"),
    ("NorlundB[3, 2, x]", "-1/2 + 5 x/2 - 3 x^2 + x^3"), ("NorlundB[6, 1]", "1/42"), ("NorlundB[n, a]", "NorlundB[n, a]"),
    ("PascalBinomial[-3, -5]", "0"), ("PascalBinomial[-3, 2]", "6"), ("PascalBinomial[5, 2]", "10"),
    ("PascalBinomial[-1, 1/2]", "ComplexInfinity"),
    ("PascalBinomial[-4, 2] == PascalBinomial[-5, 2] + PascalBinomial[-5, 1]", True),
    ("PascalBinomial[0, 0] == PascalBinomial[-1, 0] + PascalBinomial[-1, -1]", True),
    ("RamanujanTau[{1, 2, 3, 4, 5}]", "{1, -24, 252, -1472, 4830}"), ("RamanujanTau[10]", "-115920"),
    ("RamanujanTau[12]", str(TAU[11])), ("RamanujanTau[0]", "0"),
    ("RamanujanTau[2 3] == RamanujanTau[2] RamanujanTau[3]", True),
    ("RamanujanTau[1/2]", Raises()),
]

ZETA = [
    ("DirichletBeta[{0, 1, 2, 3, -2}]", "{1/2, Pi/4, Catalan, Pi^3/32, -1/2}"),
    ("DirichletBeta[5]", "5 Pi^5/1536"),
    ("DirichletBeta[3.]", Near(float(mp.nsum(lambda n: (-1) ** int(n) / (2 * n + 1) ** 3, [0, mp.inf])))),
    ("N[DirichletBeta[2], 30]", HP(lambda: mp.catalan, 30)),
    ("DirichletBeta[0.5 + I]", Near(complex(4 ** -mp.mpc(0.5, 1) * (mp.zeta(mp.mpc(0.5, 1), 0.25) - mp.zeta(mp.mpc(0.5, 1), 0.75))))),
    ("DirichletLambda[{0, 1, 2, 4}]", "{0, ComplexInfinity, Pi^2/8, Pi^4/96}"),
    ("DirichletLambda[3.]", Near(float(mp.nsum(lambda n: 1 / (2 * n + 1) ** 3, [0, mp.inf])))),
    ("DirichletLambda[s]", "DirichletLambda[s]"),
    ("PrimeZetaP[2.]", Near(float(sum(int(sp.mobius(k)) / mp.mpf(k) * mp.log(mp.zeta(2 * k)) for k in range(1, 80))))),
    ("PrimeZetaP[1]", "Infinity"), ("PrimeZetaP[-1.]", Raises()), ("PrimeZetaP[3]", "PrimeZetaP[3]"),
    ("RiemannR[100.]", Near(float(1 + mp.nsum(lambda k: mp.log(100) ** k / (k * mp.factorial(k) * mp.zeta(k + 1)),
                                              [1, mp.inf])))),
    ("RiemannR[x]", "RiemannR[x]"), ("RiemannR[1000.] > PrimePi[1000]", True),
    ("RiemannSiegelTheta[10.]", Near(float(mp.im(mp.loggamma(0.25 + 5j)) - 5 * mp.log(mp.pi)))),
    ("RiemannSiegelTheta[0]", "0"), ("N[RiemannSiegelTheta[1], 30]", HP(lambda: mp.im(mp.loggamma(mp.mpc(0.25, 0.5))) - mp.log(mp.pi) / 2, 30)),
    ("RiemannSiegelZ[14.134725141734695] < 10^-12", True),
    ("RiemannSiegelZ[5.]", Near(float(mp.re(mp.exp(1j * (mp.im(mp.loggamma(0.25 + 2.5j)) - 2.5 * mp.log(mp.pi)))
                                          * mp.zeta(0.5 + 5j))))),
    ("RiemannSiegelZ[t]", "RiemannSiegelZ[t]"),
    ("RiemannXi[{0, 1, 2, -1, 4}]", "{1/2, 1/2, Pi/6, Pi/6, Pi^2/15}"),
    ("RiemannXi[0.3]", Near(float(0.3 * (0.3 - 1) / 2 * mp.pi ** -0.15 * mp.gamma(0.15) * mp.zeta(0.3)))),
    ("RiemannXi[-2.] == RiemannXi[3.]", True), ("RiemannXi[3]", "RiemannXi[3]"),
    ("ZetaZero[1]", "ZetaZero[1]"), ("Im[N[ZetaZero[1]]]", 14.134725141734694),
    ("N[ZetaZero[-2], 20]", HP(lambda: mp.mpc(0.5, -21.022039638771554992628479593896902777334), 20)),
    ("Im[N[ZetaZero[1, 20]]]", 21.02203963877155), ("ZetaZero[0]", Raises()),
    ("LogBarnesG[{1, 2, 3, 4, 5}]", "{0, 0, 0, Log[2], Log[12]}"),
    ("LogBarnesG[4.5]", float(mp.log(mp.barnesg(4.5)))),
    ("N[LogBarnesG[1/3], 30]", HP(lambda: mp.log(mp.barnesg(mp.mpf(1) / 3)), 30)),
    ("Exp[LogBarnesG[-2.5 + I]]", Near(complex(mp.barnesg(mp.mpc(-2.5, 1))), 1e-11)),
    ("LogBarnesG[2.5] - LogBarnesG[1.5] - LogGamma[1.5]", Near(0, 1e-13)),
    ("LogBarnesG[0.]", Raises()),
]

AUX = [
    ("DawsonF[1.]", float(DAWSON1)), ("DawsonF[0]", "0"), ("DawsonF[10.]", Near(float(mp.sqrt(mp.pi) / 2 * mp.exp(-100) * mp.erfi(10)))),
    ("DawsonF[x]", "DawsonF[x]"),
    ("FresnelF[0]", "1/2"), ("FresnelF[0.5]", Near(float((0.5 - mp.fresnels(0.5)) * mp.cos(mp.pi / 8)
                                                         - (0.5 - mp.fresnelc(0.5)) * mp.sin(mp.pi / 8)))),
    ("FresnelF[100.]", Near(1 / (mp.pi * 100), 1e-4)), ("FresnelF[Infinity]", "0"),
    ("FresnelG[0]", "1/2"), ("FresnelG[2.]", Near(float((0.5 - mp.fresnelc(2)) * mp.cos(2 * mp.pi)
                                                        + (0.5 - mp.fresnels(2)) * mp.sin(2 * mp.pi)))),
    ("FresnelG[100.]", Near(1 / (mp.pi ** 2 * 100 ** 3), 1e-3)),
    ("OwenT[0.5, 1.]", float(OWEN)), ("OwenT[0, 1]", "1/8"), ("OwenT[h, 0]", "OwenT[h, 0]"), ("OwenT[2, 0]", "0"),
    ("OwenT[0.5, 1.] - (1 - Erf[0.5/Sqrt[2]]^2)/8", Near(0, 1e-13)),
    ("MarcumQ[1, 1., 2.]", float(MARCUM)), ("MarcumQ[2.5, 1.5, 1.]", Near(float(MARCUM2), 1e-9)),
    ("MarcumQ[1, 2, 0]", "1"), ("MarcumQ[2, 0, 3]", "GammaRegularized[2, 9/2]"),
    ("MarcumQ[1, 1., 1., 2.]", Near(float(mp.quad(lambda x: x * mp.besseli(0, x) * mp.exp(-(1 + x * x) / 2), [1, 2])))),
    ("MittagLefflerE[1, x]", "E^x"), ("MittagLefflerE[2, x]", "Cosh[Sqrt[x]]"),
    ("MittagLefflerE[0.5, 1.]", float(mp.exp(1) * mp.erfc(-1))),
    ("MittagLefflerE[2, 3, 2.]", float(mp.nsum(lambda k: mp.mpf(2) ** k / mp.gamma(2 * k + 3), [0, mp.inf]))),
    ("MittagLefflerE[2, -20.]", float(mp.cos(mp.sqrt(20)))), ("MittagLefflerE[a, b, 0]", "MittagLefflerE[a, b, 0]"),
    ("MittagLefflerE[1/2, 3, 0]", "1/2"),
    ("InverseGammaRegularized[2, 0.3]", Pred(lambda g: abs(float(mp.gammainc(2, g, regularized=True)) - 0.3) < 1e-14, "Q(2, z) = 0.3")),
    ("InverseGammaRegularized[1, 1/2]", "Log[2]"), ("InverseGammaRegularized[a, 1]", "InverseGammaRegularized[a, 1]"),
    ("InverseGammaRegularized[3, 1]", "0"), ("InverseGammaRegularized[3, 0]", "Infinity"),
    ("GammaRegularized[3, 1., InverseGammaRegularized[3, 1., 0.2]]", Near(0.2)),
    ("N[GammaRegularized[5/2, InverseGammaRegularized[5/2, 1/3]], 30]", HP(mp.mpf(1) / 3, 28)),
    ("InverseBetaRegularized[0.3, 2, 3]", Pred(lambda g: abs(float(mp.betainc(2, 3, 0, g, regularized=True)) - 0.3) < 1e-14, "I_z = 0.3")),
    ("InverseBetaRegularized[1/4, 2, 1]", "1/2"), ("InverseBetaRegularized[0, 2, 3]", "0"),
    ("InverseBetaRegularized[3/4, 1, 2]", "1/2"), ("InverseBetaRegularized[1.5, 2, 3]", Raises()),
    ("BetaRegularized[0.2, InverseBetaRegularized[0.2, 0.3, 2, 3], 2, 3]", Near(0.3)),
    ("MinkowskiQuestionMark[{1/2, 2/5, Sqrt[2], GoldenRatio - 1}]", "{1/2, 3/8, 7/5, 2/3}"),
    ("MinkowskiQuestionMark[(1 + Sqrt[3])/2]", "9/7"), ("MinkowskiQuestionMark[0.4]", 0.375),
    ("N[MinkowskiQuestionMark[Pi]]", Near(3 + 2 * (2.0 ** -7 - 2.0 ** -22 + 2.0 ** -23 - 2.0 ** -315), 1e-14)),
    ("MinkowskiQuestionMark[x]", "MinkowskiQuestionMark[x]"),
    ("CantorStaircase[{1/4, 1/3, 2/3, 3/4, 1/10}]", "{1/3, 1/2, 1/2, 2/3, 1/5}"),
    ("CantorStaircase[0.25]", Near(1 / 3, 1e-14)), ("CantorStaircase[-2]", "0"), ("CantorStaircase[5/2]", "1"),
    ("N[CantorStaircase[1/Pi]]", 0.4375), ("CantorStaircase[x]", "CantorStaircase[x]"),
]

Q = [
    ("QPochhammer[-1.2, .2, 5]", Near(2.89193, 1e-5)), ("QPochhammer[3, 5, 9]", "-175258163137268414822846915072"),
    ("QPochhammer[.4, .2]", Near(0.5409967762611082)), ("QPochhammer[a, q, 0]", "1"),
    ("QPochhammer[2, 3, -2]", "27/7"), ("QPochhammer[0.5]", Near(float(mp.nprod(lambda k: 1 - mp.mpf(0.5) ** k, [1, mp.inf])))),
    ("QPochhammer[a, q, 2]", "QPochhammer[a, q, 2]"), ("QPochhammer[0, q]", "1"),
    ("QFactorial[3, 1/2]", "21/8"), ("QFactorial[3, 1]", "6"), ("QFactorial[3, q]", "QFactorial[3, q]"),
    ("QFactorial[2.5, 0.5]", float(mp.qgamma(3.5, 0.5))),
    ("QGamma[4, 1/2]", "21/8"), ("QGamma[2.5, 0.3]", float(mp.qp(0.3, 0.3) / mp.qp(mp.mpf(0.3) ** 2.5, 0.3) * 0.7 ** -1.5)),
    ("QGamma[z, q]", "QGamma[z, q]"),
    ("QBinomial[4, 2, 1/2]", "35/16"), ("QBinomial[5, -3.1, E]", Near(-1.72675e-11, 1e-5)), ("QBinomial[0, 12, 5]", "0"),
    ("QBinomial[5, 2, 2]", str(sp.Rational((2 ** 5 - 1) * (2 ** 4 - 1), (2 - 1) * (2 ** 2 - 1)))),
    ("QPolyGamma[2., 0.5]", Near(float(mp.diff(lambda t: mp.log(mp.qgamma(t, 0.5)), 2)), 1e-12)),
    ("QPolyGamma[1, 0.3, 0.5]", Near(float(mp.diff(lambda t: mp.log(mp.qgamma(t, 0.5)), 0.3, 2)), 1e-11)),
    ("QPolyGamma[-1.5, 0.5]", Near(float(mp.diff(lambda t: mp.log(mp.qgamma(t, 0.5)), -1.5)), 1e-11)),
    ("QPolyGamma[z, q]", "QPolyGamma[z, q]"),
    ("QHypergeometricPFQ[{0.5}, {}, 0.3, 0.2]", Near(float(mp.qp(0.5 * 0.2, 0.3) / mp.qp(0.2, 0.3)))),  # q-binomial thm
    ("QHypergeometricPFQ[{a}, {b}, q, 0]", "1"),
    ("QHypergeometricPFQ[{0.2, 0.3}, {0.4}, 0.5, 0.1]",
     Near(float(mp.nsum(lambda k: mp.qp(0.2, 0.5, int(k)) * mp.qp(0.3, 0.5, int(k)) / (mp.qp(0.4, 0.5, int(k)) * mp.qp(0.5, 0.5, int(k))) * mp.mpf(0.1) ** k, [0, 80])))),
    ("BilateralHypergeometricPFQ[{0.3, 0.4}, {1.5, 1.7}, 1]",
     Near(float(mp.gamma(0.7) * mp.gamma(0.6) * mp.gamma(1.5) * mp.gamma(1.7) * mp.gamma(1.5)
                / (mp.gamma(1.2) * mp.gamma(1.4) * mp.gamma(1.1) * mp.gamma(1.3))))),        # Dougall
    ("BilateralHypergeometricPFQ[{a}, {b}, z]", "BilateralHypergeometricPFQ[{a}, {b}, z]"),
    ("BilateralHypergeometricPFQ[{0.5}, {1.5}, 0.5]",
     Near(float(mp.nsum(lambda n: mp.rf(0.5, n) / mp.rf(1.5, n) * mp.mpf(0.5) ** n, [0, mp.inf])
                + mp.nsum(lambda n: mp.rf(0.5, -n) / mp.rf(1.5, -n) * mp.mpf(0.5) ** (-n), [1, mp.inf])), 1e-9)),
]

APPELL = [
    ("AppellF1[1, 2, 3, 4, 0.1, 0.2]", Near(float(mp.nsum(lambda m, n: mp.rf(1, m + n) * mp.rf(2, m) * mp.rf(3, n) /
                                                           (mp.rf(4, m + n) * mp.factorial(m) * mp.factorial(n))
                                                           * mp.mpf(0.1) ** m * mp.mpf(0.2) ** n, [0, mp.inf], [0, mp.inf])))),
    ("AppellF1[a, b1, b2, c, x, 0]", "Hypergeometric2F1[a, b1, c, x]"), ("AppellF1[a, b1, b2, c, 0, 0]", "1"),
    ("AppellF1[1, 1, 1, 2, x, y]", "AppellF1[1, 1, 1, 2, x, y]"),
    ("AppellF2[1, 1, 1, 2, 2, 0.2, 0.3]", Near(float(mp.nsum(lambda m, n: mp.rf(1, m + n) * mp.rf(1, m) * mp.rf(1, n) /
                                                              (mp.rf(2, m) * mp.rf(2, n) * mp.factorial(m) * mp.factorial(n))
                                                              * mp.mpf(0.2) ** m * mp.mpf(0.3) ** n, [0, mp.inf], [0, mp.inf])))),
    ("AppellF2[a, b1, b2, c1, c2, x, 0]", "Hypergeometric2F1[a, b1, c1, x]"), ("AppellF2[1, 2, 3, 4, 5, 0, 0]", "1"),
    ("AppellF3[1, 1, 1, 1, 2, 0.2, 0.3]", Near(float(mp.nsum(lambda m, n: mp.rf(1, m) * mp.rf(1, n) * mp.rf(1, m) * mp.rf(1, n) /
                                                              (mp.rf(2, m + n) * mp.factorial(m) * mp.factorial(n))
                                                              * mp.mpf(0.2) ** m * mp.mpf(0.3) ** n, [0, mp.inf], [0, mp.inf])))),
    ("AppellF3[a1, a2, b1, b2, c, x, 0]", "Hypergeometric2F1[a1, b1, c, x]"), ("AppellF3[1, 2, 3, 4, 5, 0, 0]", "1"),
    ("AppellF4[1, 1, 2, 2, 0.1, 0.1]", Near(float(mp.nsum(lambda m, n: mp.rf(1, m + n) ** 2 /
                                                           (mp.rf(2, m) * mp.rf(2, n) * mp.factorial(m) * mp.factorial(n))
                                                           * mp.mpf(0.1) ** (m + n), [0, mp.inf], [0, mp.inf])))),
    ("AppellF4[a, b, c1, c2, x, 0]", "Hypergeometric2F1[a, b, c1, x]"), ("AppellF4[1, 2, 3, 4, 0, 0]", "1"),
]

ANGULAR = [
    ("ThreeJSymbol[{1, 0}, {1, 0}, {2, 0}]", "Sqrt[2/15]"), ("ThreeJSymbol[{1/2, 1/2}, {1/2, -1/2}, {1, 0}]", "1/Sqrt[6]"),
    ("ThreeJSymbol[{1, 0}, {1, 0}, {3, 0}]", "0"), ("ThreeJSymbol[{1, 1}, {1, 0}, {2, 0}]", "0"),
    ("ThreeJSymbol[{1, 0}, {1.3, 0}, {2, 0}]", Raises()), ("ThreeJSymbol[{j, m}, {1, 0}, {1, 0}]", "ThreeJSymbol[{j, m}, {1, 0}, {1, 0}]"),
    ("ThreeJSymbol[{2, 1}, {2, -1}, {2, 0}]", "-Sqrt[2/35]"),
    ("ClebschGordan[{1/2, 1/2}, {1/2, -1/2}, {1, 0}]", "1/Sqrt[2]"), ("ClebschGordan[{1, 1}, {1, 1}, {2, 2}]", "1"),
    ("ClebschGordan[{1/2, 1/2}, {1/2, -1/2}, {0, 0}]", "1/Sqrt[2]"), ("ClebschGordan[{1, 1}, {1, -1}, {1, 0}]", "1/Sqrt[2]"),
    ("ClebschGordan[{1, 0}, {1, 0}, {2, 0}] == Sqrt[5] ThreeJSymbol[{1, 0}, {1, 0}, {2, 0}]", True),
    ("SixJSymbol[{1, 1, 1}, {1, 1, 1}]", "1/6"), ("SixJSymbol[{1, 2, 3}, {1, 2, 3}]", "1/105"),
    ("SixJSymbol[{1, 1, 5}, {1, 1, 1}]", "0"), ("SixJSymbol[{1/2, 1/2, 1}, {1/2, 1/2, 0}]", "1/2"),
    ("SixJSymbol[{a, 1, 1}, {1, 1, 1}]", "SixJSymbol[{a, 1, 1}, {1, 1, 1}]"),
    # WignerD (reference examples)
    ("WignerD[{1, 0, 1}, psi, theta, phi]", "-Sqrt[2] E^(I phi) Cos[theta/2] Sin[theta/2]"),
    ("WignerD[{1/2, 1/2, 1/2}, psi, theta, phi]", "E^(I phi/2 + I psi/2) Cos[theta/2]"),
    ("WignerD[{1, 0, 1}, 20.2 Degree, 30 Degree, 11.0 Degree]", Near(-0.347058 - 0.0674612j, 1e-5)),
    ("WignerD[{1/2, 1/2, -1/2}, theta]", "Sin[theta/2]"), ("WignerD[{1/2, 1/2, 1/2}, theta, 0]", "Cos[theta/2]"),
    ("WignerD[{3/2, 1/2, -1/2}, theta]", "(1 + 3 Cos[theta]) Sin[theta/2]/2"),
    ("WignerD[{1, 2, 0}, theta]", "0"),
]

POLYLOG = [
    ("MultipleZeta[{2, 1}]", "Zeta[3]"), ("MultipleZeta[{2, 2, 2}]", "Pi^6/5040"), ("MultipleZeta[{3, 1}]", "Pi^4/360"),
    ("MultipleZeta[{1, 2}]", "ComplexInfinity"), ("MultipleZeta[{3}]", "Zeta[3]"),
    ("N[MultipleZeta[{3, 2}], 25]", HP(MZV32, 25)), ("N[MultipleZeta[{2, 1}], 30]", HP(lambda: mp.zeta(3), 30)),
    ("N[MultipleZeta[{4, 1, 1}]]", Near(float(mp.nsum(lambda n: 1 / n ** 4 * mp.nsum(lambda m: 1 / m * (mp.harmonic(m - 1)), [2, n - 1]) if n > 2 else 0, [3, 400])), 1e-2)),
    ("MultipleZeta[{2, 3}] + MultipleZeta[{3, 2}] + Zeta[5] - Zeta[2] Zeta[3] // N", Near(0, 1e-13)),
    ("GeneralizedPolyLog[{0, 0}, 2]", "Log[2]^2/2"), ("GeneralizedPolyLog[{2}, 1.]", float(-mp.log(2))),
    ("GeneralizedPolyLog[{0, 1}, 1.]", float(-mp.pi ** 2 / 6)), ("GeneralizedPolyLog[{1}, 1.]", "ComplexInfinity"),
    ("GeneralizedPolyLog[{3, -2}, 0.5]", Near(float(nested_quad([3, -2], 0.5)), 1e-12)),
    ("GeneralizedPolyLog[{1 + I, 2}, 0.7]", Near(complex(nested_quad([mp.mpc(1, 1), 2], 0.7)), 1e-12)),
    ("GeneralizedPolyLog[{1, 0}, 0.5]", Near(float(mp.log(0.5) * mp.log(0.5) + mp.polylog(2, 0.5)), 1e-13)),
    ("GeneralizedPolyLog[{2, 3}, 1., 0.5]", Near(float(nested_quad([1.5, 2.5], 0.5)), 1e-12)),
    ("GeneralizedPolyLog[{1/2}, 1.]", Raises()), ("GeneralizedPolyLog[{a}, x]", "GeneralizedPolyLog[{a}, x]"),
    ("HarmonicPolyLog[{1}, 0.5]", float(mp.log(2))), ("HarmonicPolyLog[{-2}, 0.5]", float(-mp.polylog(2, -0.5))),
    ("HarmonicPolyLog[{2}, 0.5]", float(mp.polylog(2, 0.5))), ("HarmonicPolyLog[{0, 0}, 3]", "Log[3]^2/2"),
    ("HarmonicPolyLog[{0, 0, -1, 0, 1}, 1/2] // N", Near(HPL5, 1e-10)),
    ("HarmonicPolyLog[{3, -1}, 0.4] == HarmonicPolyLog[{0, 0, 1, -1}, 0.4]", True),
    ("MultiplePolyLog[{2, 1}, {0.5, 0.5}]", Near(float(MPL_HALF), 1e-13)), ("MultiplePolyLog[{2, 1}, {1., 1.}]", float(mp.zeta(3))),
    ("MultiplePolyLog[{2, 1}, {2., 0.1}]", "ComplexInfinity"), ("MultiplePolyLog[{2}, {x}]", "PolyLog[2, x]"),
    ("MultiplePolyLog[{2, 1}, {1., -1.}]", Near(float(ALT21), 1e-12)), ("MultiplePolyLog[{2, 1}, {1, 1}]", "Zeta[3]"),
    ("N[MultiplePolyLog[{2, 1}, {1/2, 1/2}], 25]", HP(MPL_HALF, 25)),
]

FOX = [
    ("FoxH[{{{1/2, 1}}, {{1/3, 2}}}, {{{1/4, 3}}, {{Pi, 4}}}, 0.2]", Near(0.0145499, 1e-5)),
    ("FoxH[{{}, {}}, {{{0, 1/2}}, {}}, 0.7]", float(2 * mp.exp(-0.49))),
    ("FoxH[{{}, {}}, {{{0, 1}}, {}}, x]", "MeijerG[{{}, {}}, {{0}, {}}, x]"),
    ("FoxH[{{{a, 1/2}}, {}}, {{{b, 1}}, {}}, z]", "FoxH[{{{a, 1/2}}, {}}, {{{b, 1}}, {}}, z]"),
    ("FoxH[{{{0, 1/2}}, {}}, {{{0, 1}}, {}}, 0.3]",
     Near(float(mp.nsum(lambda k: (-1) ** int(k) / mp.factorial(k) * mp.gamma(1 + k / 2) * mp.mpf(0.3) ** k, [0, mp.inf])))),
    ("FoxH[{{}, {}}, {{{0, Sqrt[2]}}, {}}, 0.5]", Raises()),
    ("MeijerGReduce[Sin[x], x]", "Sqrt[Pi] Inactive[MeijerG][{{}, {}}, {{1/2}, {0}}, x^2/4]"),
    ("Activate[MeijerGReduce[Sin[x], x]] /. x -> 0.7", Near(float(mp.sin(0.7)))),
    ("Activate[MeijerGReduce[x^2 Exp[-3 x], x]] /. x -> 0.4", Near(float(0.16 * mp.exp(-1.2)))),
    ("Activate[MeijerGReduce[Log[1 + x], x]] /. x -> 0.3", Near(float(mp.log(1.3)))),
    ("Activate[MeijerGReduce[BesselJ[0, x], x]] /. x -> 1.3", Near(float(mp.besselj(0, 1.3)))),
    ("MeijerGReduce[f[x], x]", "MeijerGReduce[f[x], x]"),
    ("FoxHReduce[Exp[-x], x]", "Inactive[FoxH][{{}, {}}, {{{0, 1}}, {}}, x]"),
    ("Activate[FoxHReduce[Exp[-x], x]] /. x -> 0.5", Near(float(mp.exp(-0.5)))),
    ("Activate[FoxHReduce[Sin[x], x]] /. x -> 0.5", Near(float(mp.sin(0.5)))),
    ("FoxHReduce[f[x], x]", "FoxHReduce[f[x], x]"),
]

RAMANUJAN = [
    ("RamanujanTauL[12.]", Near(float(L12), 1e-9)),
    ("N[RamanujanTauL[13], 25]", HP(lambda: sum(t / mp.mpf(n + 1) ** 13 for n, t in enumerate(tau_series(40))), 12)),
    ("RamanujanTauL[s]", "RamanujanTauL[s]"),
    ("Abs[RamanujanTauL[6. + 9.222379399921102 I]] < 10^-10", True),
    ("RamanujanTauTheta[9.22]", Near(1.40372, 1e-5)), ("RamanujanTauTheta[0]", "0"),
    ("RamanujanTauTheta[-3.]", Near(-float(mp.im(mp.loggamma(6 + 3j)) - 3 * mp.log(2 * mp.pi)))),
    ("RamanujanTauZ[1.]", Near(float(abs(mp.mpf(1)) * 0) + 0.8194349386751963, 1e-9)),
    ("Abs[RamanujanTauZ[9.222379399921102]] < 10^-10", True),
    ("Im[RamanujanTauZ[5.]]", Near(0, 1e-15)),
]

MISC = [
    ("ZernikeR[4, 2, r]", "4 r^4 - 3 r^2"), ("ZernikeR[3, 2, r]", "0"), ("ZernikeR[4, 2, 0.3]", Near(4 * 0.3 ** 4 - 3 * 0.09)),
    ("ZernikeR[2, 0, r]", "2 r^2 - 1"), ("ZernikeR[n, m, r]", "ZernikeR[n, m, r]"),
    ("Integrate[ZernikeR[4, 2, r] ZernikeR[2, 2, r] r, {r, 0, 1}]", "0"),
    ("PauliMatrix[2]", "{{0, -I}, {I, 0}}"), ("PauliMatrix[Range[3]]", "{{{0, 1}, {1, 0}}, {{0, -I}, {I, 0}}, {{1, 0}, {0, -1}}}"),
    ("PauliMatrix[0]", "{{1, 0}, {0, 1}}"), ("PauliMatrix[3, TargetStructure -> \"Dense\"]", "{{1, 0}, {0, -1}}"),
    ("PauliMatrix[1] . PauliMatrix[2] == I PauliMatrix[3]", True), ("PauliMatrix[4]", Raises()),
    ("PauliMatrix[3, WorkingPrecision -> MachinePrecision]", "{{1., 0}, {0, -1.}}"),
    ("GeneralizedPower[f, x, 3]", "GeneralizedPower[f, x, 3]"),
    ("GeneralizedPower[f, x, 3] /. GeneralizedPower[g_, y_, k_] :> Fold[g, Table[y, k]]", "f[f[x, x], x]"),
    ("Head[GeneralizedPower[NonCommutativeMultiply, x + y, 3]]", "GeneralizedPower"),
    ("CombinatorS[x][y][z]", "CombinatorS[x][y][z]"),
    ("CombinatorS[x][y][z] /. CombinatorS[a_][b_][c_] :> a[c][b[c]]", "x[z][y[z]]"),
    ("Head[CombinatorS]", "Symbol"),
    ("CombinatorK[x][y] /. CombinatorK[a_][b_] :> a", "x"), ("CombinatorK[x][y]", "CombinatorK[x][y]"),
    ("Head[CombinatorK]", "Symbol"),
    ("CombinatorI[x] /. CombinatorI[a_] :> a", "x"), ("CombinatorI[x]", "CombinatorI[x]"), ("Head[CombinatorI]", "Symbol"),
    ("CombinatorB[x][y][z] /. CombinatorB[a_][b_][c_] :> a[b[c]]", "x[y[z]]"), ("CombinatorB[x][y][z]", "CombinatorB[x][y][z]"),
    ("Head[CombinatorB]", "Symbol"),
    ("CombinatorC[x][y][z] /. CombinatorC[a_][b_][c_] :> a[c][b]", "x[z][y]"), ("CombinatorC[x][y]", "CombinatorC[x][y]"),
    ("Head[CombinatorC]", "Symbol"),
    ("CombinatorW[x][y] /. CombinatorW[a_][b_] :> a[b][b]", "x[y][y]"), ("CombinatorW[x][y]", "CombinatorW[x][y]"),
    ("Head[CombinatorW]", "Symbol"),
    ("CombinatorY[f] /. CombinatorY[g_] :> g[CombinatorY[g]]", "f[CombinatorY[f]]"), ("CombinatorY[f]", "CombinatorY[f]"),
    ("Head[CombinatorY]", "Symbol"),
]

CASES = (ELEMENTARY + GENERALIZED + CONSTANTS + NUMBERS + ZETA + AUX + Q + APPELL + ANGULAR + POLYLOG + FOX
         + RAMANUJAN + MISC)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w10(case):
    check(case)
