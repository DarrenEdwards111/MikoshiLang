import mpmath as mp
import pytest

from _harness import Raises, Pred, check, ids

mp.mp.dps = 30


def sbj(n, z):
    return mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + 0.5, z)


CASES = [
    ("Gamma[5]", "24"), ("Gamma[1/2]", "Sqrt[Pi]"), ("Gamma[2.5]", mp.gamma(2.5)), ("Gamma[-1]", "ComplexInfinity"),
    ("Gamma[2, 1.5]", mp.gammainc(2, 1.5)), ("Gamma[1, x]", "E^(-x)"),
    ("Beta[2, 3]", "1/12"), ("Beta[0.5, 1.5]", mp.beta(0.5, 1.5)), ("Beta[1/2, 1/2]", "Pi"),
    ("LogGamma[10]", "Log[362880]"), ("LogGamma[2.5]", mp.loggamma(2.5)), ("LogGamma[0.3]", mp.loggamma(0.3)),
    ("DiGamma[1]", "-EulerGamma"), ("DiGamma[2.5]", mp.digamma(2.5)), ("DiGamma[1/2]", "-EulerGamma - 2 Log[2]"),
    ("TriGamma[1]", "Pi^2/6"), ("TriGamma[2.5]", mp.psi(1, 2.5)), ("TriGamma[0.7]", mp.psi(1, 0.7)),
    ("PolyGamma[0, 1]", "-EulerGamma"), ("PolyGamma[2, 1.5]", mp.psi(2, 1.5)), ("PolyGamma[1, 1]", "Pi^2/6"),
    ("Zeta[2]", "Pi^2/6"), ("Zeta[3.]", mp.zeta(3)), ("Zeta[-1]", "-1/12"), ("Zeta[0.5]", mp.zeta(0.5)),
    ("HurwitzZeta[2, 1]", "Pi^2/6"), ("HurwitzZeta[3., 0.5]", mp.zeta(3, 0.5)), ("HurwitzZeta[2.5, 2.]", mp.zeta(2.5, 2)),
    ("DirichletEta[1]", "Log[2]"), ("DirichletEta[2]", "Pi^2/12"), ("DirichletEta[0.5]", mp.altzeta(0.5)),
    ("Erf[0]", "0"), ("Erf[0.5]", mp.erf(0.5)), ("Erf[-1.2]", mp.erf(-1.2)),
    ("Erfc[0]", "1"), ("Erfc[0.5]", mp.erfc(0.5)), ("Erfc[2.]", mp.erfc(2)),
    ("Erfi[0]", "0"), ("Erfi[0.5]", mp.erfi(0.5)), ("Erfi[1.5]", mp.erfi(1.5)),
    ("ExpIntegralEi[1.]", mp.ei(1)), ("ExpIntegralEi[2.5]", mp.ei(2.5)), ("ExpIntegralEi[-1.]", mp.ei(-1)),
    ("ExpIntegralE[1, 1.]", mp.expint(1, 1)), ("ExpIntegralE[2, 0.5]", mp.expint(2, 0.5)), ("ExpIntegralE[3, 2.]", mp.expint(3, 2)),
    ("LogIntegral[2.]", mp.li(2)), ("LogIntegral[10.]", mp.li(10)), ("LogIntegral[0]", "0"),
    ("SinIntegral[0]", "0"), ("SinIntegral[1.]", mp.si(1)), ("SinIntegral[5.]", mp.si(5)),
    ("CosIntegral[1.]", mp.ci(1)), ("CosIntegral[5.]", mp.ci(5)), ("CosIntegral[0.2]", mp.ci(0.2)),
    ("BesselJ[0, 0]", "1"), ("BesselJ[0, 2.5]", mp.besselj(0, 2.5)), ("BesselJ[2, 1.5]", mp.besselj(2, 1.5)),
    ("BesselJ[1/2, 1.]", mp.besselj(0.5, 1)),
    ("BesselY[0, 2.5]", mp.bessely(0, 2.5)), ("BesselY[1, 1.]", mp.bessely(1, 1)), ("BesselY[2, 4.]", mp.bessely(2, 4)),
    ("BesselI[0, 0]", "1"), ("BesselI[0, 1.5]", mp.besseli(0, 1.5)), ("BesselI[2, 3.]", mp.besseli(2, 3)),
    ("BesselK[0, 1.5]", mp.besselk(0, 1.5)), ("BesselK[1, 2.]", mp.besselk(1, 2)), ("BesselK[2, 0.5]", mp.besselk(2, 0.5)),
    ("HankelH1[0, 1.]", mp.hankel1(0, 1)), ("HankelH1[1, 2.5]", mp.hankel1(1, 2.5)), ("HankelH1[2, 3.]", mp.hankel1(2, 3)),
    ("HankelH2[0, 1.]", mp.hankel2(0, 1)), ("HankelH2[1, 2.5]", mp.hankel2(1, 2.5)), ("HankelH2[2, 3.]", mp.hankel2(2, 3)),
    ("SphericalBesselJ[0, 1.]", sbj(0, 1)), ("SphericalBesselJ[1, 2.]", sbj(1, 2)), ("SphericalBesselJ[2, 3.5]", sbj(2, 3.5)),
    ("SphericalBesselJ[0, x]", "SphericalBesselJ[0, x]"),
    ("AiryAi[0.]", mp.airyai(0)), ("AiryAi[1.5]", mp.airyai(1.5)), ("AiryAi[-2.]", mp.airyai(-2)),
    ("AiryBi[0.]", mp.airybi(0)), ("AiryBi[1.5]", mp.airybi(1.5)), ("AiryBi[-2.]", mp.airybi(-2)),
    ("ChebyshevT[3, x]", "4 x^3 - 3 x"), ("ChebyshevT[5, 0.3]", mp.chebyt(5, 0.3)), ("ChebyshevT[0, x]", "1"),
    ("ChebyshevU[2, x]", "4 x^2 - 1"), ("ChebyshevU[4, 0.3]", mp.chebyu(4, 0.3)), ("ChebyshevU[0, x]", "1"),
    ("HermiteH[3, x]", "8 x^3 - 12 x"), ("HermiteH[4, 0.5]", mp.hermite(4, 0.5)), ("HermiteH[0, x]", "1"),
    ("LaguerreL[2, x]", "1 - 2 x + x^2/2"), ("LaguerreL[3, 0.5]", mp.laguerre(3, 0, 0.5)), ("LaguerreL[2, 1, x]", "3 - 3 x + x^2/2"),
    ("LegendreP[2, x]", "(3 x^2 - 1)/2"), ("LegendreP[3, 0.4]", mp.legendre(3, 0.4)), ("LegendreP[0, x]", "1"),
    ("GegenbauerC[2, 1, x]", "4 x^2 - 1"), ("GegenbauerC[3, 0.5, 0.3]", mp.gegenbauer(3, 0.5, 0.3)), ("GegenbauerC[2, 2, 0.5]", mp.gegenbauer(2, 2, 0.5)),
    ("JacobiP[1, 1, 1, x]", "2 x"), ("JacobiP[2, 0.5, 1.5, 0.3]", mp.jacobi(2, 0.5, 1.5, 0.3)), ("JacobiP[3, 1, 2, 0.7]", mp.jacobi(3, 1, 2, 0.7)),
    # Wolfram elliptic functions use the parameter m = k^2
    ("EllipticK[0]", "Pi/2"), ("EllipticK[0.5]", mp.ellipk(0.5)), ("EllipticK[-1.]", mp.ellipk(-1)),
    ("EllipticE[0]", "Pi/2"), ("EllipticE[0.5]", mp.ellipe(0.5)), ("EllipticE[1]", "1"),
    ("EllipticE[0.7, 0.4]", mp.ellipe(0.7, 0.4)), ("EllipticE[Pi/2, 1/2]", "EllipticE[1/2]"),
    ("EllipticEIncomplete[0.7, 0.4]", mp.ellipe(0.7, 0.4)), ("EllipticEIncomplete[1.2, 0.3]", mp.ellipe(1.2, 0.3)),
    ("EllipticEIncomplete[0, 0.3]", "0"),
    ("EllipticF[0.7, 0.4]", mp.ellipf(0.7, 0.4)), ("EllipticF[1.2, 0.9]", mp.ellipf(1.2, 0.9)), ("EllipticF[0, m]", "0"),
    ("EllipticPi[0.3, 0.5]", mp.ellippi(0.3, 0.5)), ("EllipticPi[0, 0.5]", mp.ellipk(0.5)),
    ("EllipticPi[0.3, 0.8, 0.5]", mp.ellippi(0.3, 0.8, 0.5)),
    ("KroneckerDelta[1, 1]", "1"), ("KroneckerDelta[1, 2]", "0"), ("KroneckerDelta[2, 2, 2]", "1"), ("KroneckerDelta[0]", "1"), ("KroneckerDelta[2, 2, 3]", "0"), ("KroneckerDelta[i, j]", "KroneckerDelta[i, j]"),
    ("SphericalHarmonicY[0, 0, t, p]", "1/(2 Sqrt[Pi])"),
    ("SphericalHarmonicY[1, 0, 0.7, 0.2]", mp.spherharm(1, 0, 0.7, 0.2)),
    ("SphericalHarmonicY[2, 1, 0.7, 0.2]", mp.spherharm(2, 1, 0.7, 0.2)),
    ("SphericalHarmonicY[1, 2, 0.7, 0.2]", "0"),
    ("Hypergeometric0F1[1, 0.5]", mp.hyp0f1(1, 0.5)), ("Hypergeometric0F1[2.5, -1.]", mp.hyp0f1(2.5, -1)),
    ("Hypergeometric0F1[b, 0]", "1"),
    ("Hypergeometric1F1[1, 2, 0.5]", mp.hyp1f1(1, 2, 0.5)), ("Hypergeometric1F1[a, a, z]", "E^z"),
    ("Hypergeometric1F1[0.5, 1.5, -2.]", mp.hyp1f1(0.5, 1.5, -2)),
    ("Hypergeometric2F1[1, 1, 2, 0.5]", mp.hyp2f1(1, 1, 2, 0.5)), ("Hypergeometric2F1[0.5, 0.5, 1.5, 0.25]", mp.hyp2f1(0.5, 0.5, 1.5, 0.25)),
    ("Hypergeometric2F1[1, 1, 2, -1/2]", mp.hyp2f1(1, 1, 2, -0.5)),
    ("LegendreQ[0, 0.5]", mp.legenq(0, 0, 0.5)), ("LegendreQ[2, 0.3]", mp.legenq(2, 0, 0.3)), ("LegendreQ[1, x]", "x ArcTanh[x] - 1"),
    ("LambertW[1.]", mp.lambertw(1)), ("LambertW[0]", "0"), ("LambertW[E]", "1"),
    ("ProductLog[1.]", mp.lambertw(1)), ("ProductLog[-1, -0.2]", mp.lambertw(-0.2, -1)), ("ProductLog[2.5]", mp.lambertw(2.5)),
    ("StruveH[0, 1.]", mp.struveh(0, 1)), ("StruveH[1, 2.5]", mp.struveh(1, 2.5)), ("StruveH[2, 0.5]", mp.struveh(2, 0.5)),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_special(case):
    check(case)
