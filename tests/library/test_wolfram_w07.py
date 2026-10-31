"""Differential tests for Wolfram batch W07 (special functions I: elliptic, theta, Weierstrass, Carlson, Heun,
Mathieu, Lame, spheroidal, Coulomb, Kelvin, Lommel, Scorer, ...; module wl_special_elliptic).

Expected values are the outputs printed on the Wolfram Language reference pages
(reference.wolfram.com/language/ref/<Name>.html, "Basic Examples" / "Numerical Evaluation"), at the precision
shown there (6 significant digits for machine numbers, 20-50 digits for N[..., p]), or mathematically verifiable
results (defining identities, inverse round trips, Wronskians, special values, independent mpmath formulas).
"""
import mpmath as mp
import pytest
import sympy as sp

from _harness import Pred, Raises, check, ev, ids, to_complex
from mikoshilang.expr import Expr
from mikoshilang.math_ops import to_sympy


def _c(x):
    return to_complex(x)


def W(want, sig=6):
    """A Wolfram machine-precision output shown with `sig` significant digits (complex allowed)."""
    want = complex(want)
    tol = 0.6 * 10 ** (1 - sig)

    def ok(g):
        return abs(_c(g) - want) <= tol * max(abs(want), 1e-300)
    return Pred(ok, f"~{want} ({sig} digits)")


def Near(want, rtol=1e-12, atol=0.0):
    want = complex(want)
    return Pred(lambda g: abs(_c(g) - want) <= rtol * abs(want) + atol, f"~{want}")


def Zero(atol=1e-12):
    return Pred(lambda g: abs(_c(g)) <= atol, f"~0 (|.| <= {atol})")


def WL(*wants, sig=6):
    """A List of Wolfram machine outputs."""
    preds = [W(w, sig) for w in wants]

    def ok(g):
        return isinstance(g, Expr) and g.head == "List" and len(g.args) == len(preds) and \
            all(p.f(a) for p, a in zip(preds, g.args))
    return Pred(ok, f"~{list(wants)}")


def _ndig(s):
    return len(s.lstrip("-").split("e")[0].replace(".", "").lstrip("0"))


def HP(re, im=None, digits=None):
    """A high-precision Wolfram output (decimal strings); checked to all but the last 2 shown digits."""
    d = digits or min(_ndig(re), _ndig(im) if im else 10 ** 6)
    with mp.workdps(d + 20):
        w = mp.mpc(mp.mpf(re), mp.mpf(im) if im else 0)

    def ok(g):
        v = to_sympy(g) if not isinstance(g, sp.Basic) else g
        with mp.workdps(d + 20):
            gv = mp.mpc(str(sp.N(sp.re(v), d + 15)), str(sp.N(sp.im(v), d + 15)))
            return abs(gv - w) <= mp.mpf(10) ** (-(d - 2)) * abs(w)
    return Pred(ok, f"~{re}{' + I ' + im if im else ''} ({d} digits)")


def HPL(*pairs):
    preds = [HP(*p) if isinstance(p, tuple) else HP(p) for p in pairs]

    def ok(g):
        return isinstance(g, Expr) and g.head == "List" and len(g.args) == len(preds) and \
            all(p.f(a) for p, a in zip(preds, g.args))
    return Pred(ok, "list of high-precision values")


def Unev(head):
    return Pred(lambda g: isinstance(g, Expr) and g.head == head, f"unevaluated {head}[...]")


R = Raises()

# ───────────────────────────── zeros, Anger/Weber, AGM, Struve ─────────────────────────────
ZEROS = [
    ("N[AiryAiZero[1]]", W(-2.33811)),
    ("N[AiryAiZero[3]]", W(-5.52056)),
    ("N[AiryAiZero[2, -500]]", W(-500.17, 5)),
    ("AiryAiZero[1]", Unev("AiryAiZero")),
    ("AiryAiZero[0]", Unev("AiryAiZero")),
    ("AiryAiZero[1.5]", R),
    ("AiryAi[AiryAiZero[4.]]", Zero(1e-12)),
    ("N[AiryBiZero[1]]", W(-1.17371)),
    ("N[AiryBiZero[3]]", W(-4.83074)),
    ("N[AiryBiZero[2, -500]]", W(-500.24, 5)),
    ("AiryBi[AiryBiZero[5.]]", Zero(1e-12)),
    ("AiryBiZero[-2]", Unev("AiryBiZero")),
    ("N[BesselJZero[0, 1]]", W(2.40483)),
    ("BesselJZero[0., 2]", W(5.52008)),
    ("N[BesselJZero[0, 1, 40]]", W(40.0584)),
    ("BesselJ[3, BesselJZero[3, 4.]]", Zero(1e-12)),
    ("N[BesselYZero[0, 1]]", W(0.893577)),
    ("N[BesselYZero[0, 2]]", W(3.95768)),
    ("BesselY[1, BesselYZero[1, 2.]]", Zero(1e-12)),
    ("BesselYZero[0, 0]", Unev("BesselYZero")),
    ("AngerJ[1/2, 5.0]", W(-0.285734)),
    ("AngerJ[1, 2.0]", W(0.576725)),
    ("AngerJ[1, -3.0]", W(-0.339059)),
    ("N[AngerJ[2, 3], 50]", HP("0.48609126058589107690783109411498403480166226564330")),
    ("AngerJ[3, 2.5]", Near(float(mp.besselj(3, 2.5)), 1e-12)),        # integer order: AngerJ = BesselJ
    ("WeberE[1/2, 5.0]", W(0.0333642)),
    ("WeberE[1, 2.0]", W(-0.010144, 5)),
    ("WeberE[0.5, -4.0]", W(-0.233603)),
    ("N[WeberE[1, 3], 50]", HP("-0.38348979681886901773728810558623864755329968210139")),
    ("ArithmeticGeometricMean[1.8, 1.2]", W(1.48481)),
    ("ArithmeticGeometricMean[1.2, 5]", W(2.76517)),
    ("ArithmeticGeometricMean[-2., -5]", W(-3.329, 4)),
    ("N[ArithmeticGeometricMean[52, 5], 30]", HP("21.8724836267417540342866215963")),
    ("ArithmeticGeometricMean[3, 3]", "3"),
    ("ArithmeticGeometricMean[0, 7]", "0"),
    ("StruveL[0, 2.5]", W(3.01121)),
    ("N[StruveL[0, 4], 50]", HP("11.131050203248583430971000405203573790363219235996")),
    ("StruveL[1, 2.]", Near(float(mp.struvel(1, 2)), 1e-12)),
]

# ───────────────────────────── Carlson ─────────────────────────────
CARLSON = [
    ("CarlsonRC[4., 5.]", W(0.463648)),
    ("CarlsonRC[5, 4.]", W(0.481212)),
    ("N[CarlsonRC[Sqrt[3], -2], 50]", HP("0.43038653843824667506780670352825472156472471657126")),
    ("CarlsonRC[4., 4.]", Near(0.5)),
    ("CarlsonRD[2., 3., 5.]", W(0.133211)),
    ("CarlsonRD[3/5, 11, 22/7.]", W(0.169773)),
    ("N[CarlsonRD[1, 7, 11], 50]", HP("0.053673859634013448581165319888961078106350312636220")),
    ("CarlsonRD[2., 2., 2.]", Near(2 ** -1.5)),
    ("CarlsonRE[3., 5.]", W(1.99207)),
    ("CarlsonRE[1.0, 1/GoldenRatio]", W(0.896279)),
    ("N[CarlsonRE[2, 7], 50]", HP("2.0769537676050639362132413613615290943447003386077")),
    ("N[Pi/2 CarlsonRE[1, 1 - 3/4]]", W(1.21106)),
    ("CarlsonRF[1., 2., 3.]", W(0.726946)),
    ("N[CarlsonRF[4, 6, 8]]", W(0.413009)),
    ("N[CarlsonRF[4, -6, 8]]", W(0.455848 - 0.261018j)),
    ("CarlsonRF[3., 3., 3.]", Near(3 ** -0.5)),
    ("CarlsonRG[3., 5., 11.]", W(2.48078)),
    ("CarlsonRG[5, 1.0, 7]", W(2.04027)),
    ("N[CarlsonRG[3/4, 4, 5], 50]", HP("1.7683569172007606057005120565827933724840481663269")),
    ("CarlsonRJ[2., 3., 5., 7.2]", W(0.104459)),
    ("CarlsonRJ[3, 5, 1, 0.2]", W(0.815304)),
    ("CarlsonRJ[3, 5, 1., -5]", W(-0.146216)),
    ("CarlsonRJ[2., 2., 2., 2.]", Near(2 ** -1.5)),
    ("CarlsonRK[2., 3.]", W(0.637306)),
    ("CarlsonRK[1.0, Sqrt[2] - 1]", W(1.23154)),
    ("N[CarlsonRK[5, 7], 50]", HP("0.41040751430462836940653167883696572076715396994946")),
    ("N[Pi/2 CarlsonRK[1, 1 - 3/10]]", W(1.71389)),
    ("CarlsonRM[3., 5., 1.]", W(0.341318)),
    ("N[CarlsonRM[Sqrt[3], 4, 2]]", W(0.284061)),
    ("N[CarlsonRM[5, 1, 2], 50]", HP("0.31253507895849908826754681942772385132977517380606")),
]

# ───────────────────────────── Coulomb, Kelvin, Whittaker, parabolic cylinder, Scorer, Lommel ─────────────────────
WAVES = [
    ("CoulombF[3, 0.12, 7.]", W(0.158992)),
    ("N[CoulombF[3, 2, 5], 50]", HP("0.43131593114087869012387960004177852871854471041151")),
    ("N[CoulombF[1/2, 7, 1]]", W(9.2647e-8, 5)),
    ("CoulombG[3, 0.12, 7.]", W(-1.06931)),
    ("N[CoulombG[1/2, 7, 1]]", W(1.44498e6)),
    ("N[CoulombG[1/2, -7, 1]]", W(0.355213)),
    ("CoulombH1[2, 0.2, 3.]", W(0.978448 + 0.811798j)),
    ("N[CoulombH1[3, 2, 5], 25]", HP("2.048694713777195592942385", "0.4313159311408786901238800")),
    ("CoulombH1[1, 0.8, 5]", W(-0.689415 + 0.89243j, 5)),
    ("CoulombH2[2, 0.2, 3.]", W(0.978448 - 0.811798j)),
    ("N[CoulombH2[3, 2, 5], 25]", HP("2.048694713777195592942385", "-0.4313159311408786901238800")),
    ("CoulombH2[3, 4 + I, 0.8]", W(-23662.5 + 32682.9j)),
    ("KelvinBer[2.5]", W(0.399968)),
    ("KelvinBer[7.]", W(-3.63293)),
    ("N[KelvinBer[-6, 8], 50]", HP("-6.1597937151765940539549479941618455027596413267910")),
    ("KelvinBei[2.5]", W(1.45718)),
    ("KelvinBei[5.]", Near(float(mp.bei(0, 5)), 1e-12)),
    ("N[KelvinBei[-7, 8], 50]", HP("1.5844868110163385629902574783376235624040854463360")),
    ("KelvinKer[2.5]", W(-0.069688, 5)),
    ("KelvinKer[2, 2.5]", W(0.188804)),
    ("KelvinKer[-1, 5.]", Near(float(mp.ker(-1, 5)), 1e-12)),
    ("KelvinKei[2.5]", W(-0.110696)),
    ("KelvinKei[2, 2.5]", W(0.12424, 5)),
    ("KelvinKei[7.]", W(0.00270037)),
    ("WhittakerM[2, 3, 4.5]", W(68.967, 5)),
    ("WhittakerM[2, 3, 1.7]", W(4.07202)),
    ("WhittakerM[1.5, 0.25, 2.]", Near(float(mp.exp(-1) * mp.mpf(2) ** 0.75 * mp.hyp1f1(-0.75, 1.5, 2)), 1e-12)),
    ("WhittakerW[2, 3, 4.5]", W(11.3063)),
    ("WhittakerW[6, 4, 1.7]", W(1740.55)),
    ("WhittakerW[1.5, 0.25, 2.]", Near(float(mp.exp(-1) * mp.mpf(2) ** 0.75 * mp.hyperu(-0.75, 1.5, 2)), 1e-12)),
    ("ParabolicCylinderD[0, 1.5]", W(0.569783)),
    ("ParabolicCylinderD[7., 5]", W(48.4544)),
    ("ParabolicCylinderD[.51, .87]", W(0.848114)),
    ("ScorerGi[5.]", W(0.0649198)),
    ("N[ScorerGi[Pi/2]]", W(0.19902, 5)),
    ("N[ScorerGi[1/3], 50]", HP("0.23848942143728333037399388668076539114324489688891")),
    ("ScorerGiPrime[5.]", W(-0.0138595)),
    ("ScorerGiPrime[25.]", W(-0.000509557)),
    ("N[ScorerGiPrime[Pi/2]]", W(-0.0710516)),
    ("N[ScorerGiPrime[1/3], 50]", HP("0.056097743902404760517474720533206074397558099109960")),
    ("ScorerHi[3.]", W(13.9231)),
    ("ScorerHi[7.]", W(80327.7)),
    ("N[ScorerHi[Pi]]", W(17.6388)),
    ("N[ScorerHi[1/3], 50]", HP("0.53012879936183198297470954961873272510840675231864")),
    ("ScorerHiPrime[3.]", W(22.9639)),
    ("ScorerHiPrime[7.]", W(209553.)),
    ("N[ScorerHiPrime[Pi/3]]", W(1.04402)),
    ("N[ScorerHiPrime[1/3], 50]", HP("0.43198059101649421611880410993296779169258587211483")),
    ("LommelS1[-0.25, 2/3, 1]", W(7.3332, 5)),
    ("N[LommelS1[2, 0, 4], 50]", HP("3.7879196040041784935926236016310124924725479873781")),
    ("LommelS1[0.5, 0.3, 2.]", Near(float(mp.mpf(2) ** 1.5 / (1.5 ** 2 - 0.09) * mp.hyp1f2(1, 1.6, 1.9, -1)), 1e-12)),
    ("LommelS2[-0.25, 2/3, 1]", W(0.749078)),
    ("N[LommelS2[2, 2/5, 4], 50]", HP("3.7980224304089304202056468516223743660998938401039")),
    ("LommelS2[7/3 + I, 1.2, 4.5 - I]", W(4.30491 + 9.55209j)),
    ("LommelT1[-0.25, 2/3, 1]", W(9.71561)),
    ("N[LommelT1[2, 0, 4], 50]", HP("13.484612772632462293130137197599014533323208955720")),
    ("LommelT1[1/2, 1/2, 0.7]", Near(float((mp.cosh(0.7) - 1) / mp.sqrt(0.7)), 1e-12)),
    ("LommelT2[-0.25, 2/3, 1]", W(0.838191)),
    ("N[LommelT2[2, 3/2, 4], 50]", HP("3.7165183624360502655026928435275546798402424410910")),
    ("LommelT2[7/3 + I, 1.2, 4.5 - I]", W(2.01919 + 8.76817j)),
    ("LommelT2[1/2, 1/3, {0.5, 1.0, 1.5}]", WL(1.34473, 1.00454, 0.831855)),
]

# ───────────────────────────── theta, modular, nome ─────────────────────────────
THETA = [
    ("EllipticTheta[1, 2., 1/3]", W(1.42788)),
    ("EllipticTheta[2, 5, .5]", W(0.183328)),
    ("EllipticTheta[3, .4, .5]", W(1.69015)),
    ("N[EllipticTheta[3, 1/4, 1/3], 25]", HP("1.598403007243784280861280")),
    ("EllipticTheta[1, 0, 0.3]", Zero(0)),
    ("N[EllipticTheta[3, Pi/2, Exp[-Pi/1000]], 20]",            # q near 1: needs ~1100 extra bits, no garbage
     HP("5.0926230946447430804e-340", digits=18)),
    ("EllipticTheta[3, 0.3, 1.2]", R),
    ("EllipticThetaPrime[2, 2.3, 1/3]", W(-1.41711)),
    ("EllipticThetaPrime[2, 5, .5]", W(0.811083)),
    ("EllipticThetaPrime[1, .4, .5]", W(1.03999)),
    ("N[EllipticThetaPrime[3, 15, 1/3], 25]", HP("1.346934998869974619488281")),
    ("EllipticThetaPrime[3, 0, 0.4]", Zero(0)),
    ("DedekindEta[1 + 2. I]", W(0.572198 + 0.15332j, 5)),
    ("DedekindEta[14.3 + I]", W(-0.635443 - 0.435071j)),
    ("DedekindEta[1.5 + I]", W(0.712404 + 0.295087j)),
    ("DedekindEta[I]", "Gamma[1/4]/(2 Pi^(3/4))"),
    ("DedekindEta[-I]", Unev("DedekindEta")),
    ("KleinInvariantJ[1 + 2. I]", W(166.375)),
    ("KleinInvariantJ[7.5 + I]", W(-0.051849, 5)),
    ("KleinInvariantJ[I]", "1"),
    ("N[KleinInvariantJ[1/9 + I], 50]", HP("0.83670412943246801957501016527762084761776225413542",
                                          "-0.016840596474901280663655935531428853147223205275", digits=45)),
    ("ModularLambda[1 + 3. I]", W(-0.00129202623999478, 14)),
    ("ModularLambda[14.3 + I]", W(0.4347361265876 + 0.3543146131269j, 13)),
    ("ModularLambda[1.5 + I]", W(0.22856559796511 - 0.63630885846189j, 14)),
    ("N[ModularLambda[1/7 + I], 50]", HP("0.48805792165803031281027381610666812671981914754952",
                                         "0.15930383175685144348362383600974058689665644411190")),
    ("ModularLambda[I]", "1/2"),
    ("EllipticNomeQ[-2.]", W(-0.0682378)),
    ("EllipticNomeQ[.5]", W(0.0432139)),
    ("N[EllipticNomeQ[5/16]]", W(0.0234012)),
    ("N[EllipticNomeQ[3/17], 50]", HP("0.012132369498116773570690565387863227120612596642326")),
    ("EllipticNomeQ[1/2]", "E^(-Pi)"),
    ("InverseEllipticNomeQ[0.1]", W(0.802403)),
    ("InverseEllipticNomeQ[.2]", W(0.965852)),
    ("N[InverseEllipticNomeQ[5/17], 50]", HP("0.99498241811420340694821403645941543408178720619728")),
    ("InverseEllipticNomeQ[EllipticNomeQ[0.37]]", Near(0.37, 1e-12)),
    ("NevilleThetaC[2., 0.3]", W(-0.258749)),
    ("NevilleThetaC[-45., -.3]", W(-0.557984)),
    ("NevilleThetaC[E, .7]", W(-0.459525)),
    ("N[NevilleThetaC[1/6, 1/3], 50]", HP("0.98856573814785930430827922299271141746387030835102")),
    ("NevilleThetaD[2., 0.3]", W(0.920424)),
    ("NevilleThetaD[2, .5]", W(0.843315)),
    ("NevilleThetaD[E, .7]", W(0.796848)),
    ("N[NevilleThetaD[6 Pi, 7/11], 50]", HP("0.87691640051378579606943404382187959575269199816857")),
    ("NevilleThetaN[2., 1/3]", W(1.1006, 5)),
    ("NevilleThetaN[Pi/3., 0]", W(1)),
    ("NevilleThetaN[E, .7]", W(1.2743, 5)),
    ("N[NevilleThetaN[5/6, 1/3], 50]", HP("1.0500915231173482381157096441538140787289122601943")),
    ("NevilleThetaS[2.5, 0.3]", W(0.820869)),
    ("NevilleThetaS[-5., 1]", W(-74.2032)),
    ("NevilleThetaS[Pi/6., 0]", W(0.5)),
    ("N[NevilleThetaS[5/7, 2/3], 50]", HP("0.68196864793213303477041345719863087191282615140591")),
    ("N[NevilleThetaS[2 Pi, 7/11], 50]", HP("-1.2481688456045470708942994054203836101493368380236")),
    ("SiegelTheta[{{I, -1/2}, {-1/2, I}}, {0.5, 0}]", W(1.00748)),
    ("N[SiegelTheta[IdentityMatrix[2] I, {1, 2}]]", W(1.18034)),
    ("N[SiegelTheta[{{2, 4}, {4, 2}}, IdentityMatrix[2] I, {1, 2}]]", W(1.18034)),
    ("N[SiegelTheta[{{2 I, 0}, {0, 2 I}}, {9, 2}], 20]", HP("1.0074837203450847062", digits=19)),
    ("SiegelTheta[{{I + 1, -2}, {-2, I + 4}}, {1.2222222222222222222, I + 1}]", W(24.763374406950077, 14)),
    ("SiegelTheta[{{0.5 + I}}, {2}]", Near(complex(mp.jtheta(3, 2 * mp.pi, mp.exp(1j * mp.pi * (0.5 + 1j)))), 1e-13)),
    ("N[SiegelTheta[{{I/1000}}, {1/2}], 20]", HP("5.0926230946447430804e-340", digits=18)),
    ("SiegelTheta[{{I, 1}, {0, 2 I}}, {1, 2.}]", R),
    ("SiegelTheta[{{-I}}, {0.5}]", R),
]

# ───────────────────────────── Jacobi amplitude/zeta/epsilon, inverses ─────────────────────────────
JACOBI = [
    ("JacobiAmplitude[4., 2/3]", W(3.0837, 5)),
    ("N[JacobiAmplitude[Pi/3, 1/5], 50]", HP("1.0165586491576824670186942551100936068189550267837")),
    ("JacobiAmplitude[u, 0]", "u"),
    ("JacobiSN[0.7, 0.4] - Sin[JacobiAmplitude[0.7, 0.4]]", Zero(1e-14)),
    ("JacobiEpsilon[2.5, 0.5]", W(1.69598)),
    ("N[JacobiEpsilon[2, 1/3], 50]", HP("1.6090927588599231302882559930934382552029018176665")),
    ("JacobiEpsilon[0.7, 0]", Near(0.7)),
    ("JacobiZN[1.23, 0.3]", W(0.0615763)),
    ("N[JacobiZN[1, 1/2], 50]", HP("0.14380198649913294348031582657890642012147872394084")),
    ("JacobiZN[0.8, 0.]", Zero(1e-15)),
    ("JacobiZeta[2, 0.5]", W(-0.117772)),
    ("N[JacobiZeta[2, 3], 30]", HP("-0.210943789958196804870285060445", "0.483396461384815292", digits=17)),
    ("JacobiZeta[2 + 3 I, 0.8 I]", W(8.18736 + 2.13147j)),
    ("InverseJacobiDC[1.3, 0.5]", W(0.919865)),
    ("JacobiDC[InverseJacobiDC[1.3, 0.5], 0.5]", W(1.3)),
    ("N[InverseJacobiDC[3, 1/2], 50]", HP("1.5109586514187672605502044573978781162982167772403")),
    ("InverseJacobiDS[1.3, 0.5]", W(0.776223)),
    ("JacobiDS[InverseJacobiDS[1.3, 0.5], 0.5]", W(1.3)),
    ("N[InverseJacobiDS[3, 1/2], 50]", HP("0.33343634653321716663936302039474043773274071234187")),
    ("JacobiND[InverseJacobiND[1.1, 0.3], 0.3]", Near(1.1, 1e-13)),
    ("InverseJacobiND[1., 0.6]", Zero(1e-15)),
    ("InverseJacobiND[0.5, 0.3]", R),                    # Wolfram: 0. + 1.53449 I (complex branch not implemented)
]

# ───────────────────────────── Weierstrass, EllipticExp/Log ─────────────────────────────
WEIER = [
    ("WeierstrassHalfPeriods[{1., 2.}]", WL(1.30836, 0.654182 + 1.22937j)),
    ("N[WeierstrassHalfPeriods[{1, 2}], 50]", HPL("1.3083647626370968653240501853340822887833812579379",
                                                  ("0.65418238131854843266202509266704114439169062896895",
                                                   "1.2293687516329150991198952049855018530054666213339"))),
    ("WeierstrassHalfPeriods[{1, 2.}]", WL(1.30836, 0.654182 + 1.22937j)),
    ("WeierstrassHalfPeriods[{3, 1}]", Unev("WeierstrassHalfPeriods")),
    ("WeierstrassHalfPeriodW1[{11.0, 7}]", W(0.927037)),
    ("N[WeierstrassHalfPeriodW1[{11, 7}], 50]", HP("0.92703733865068595921692517359763002310879941176088")),
    ("WeierstrassP[WeierstrassHalfPeriodW1[{11., 7.}], {11., 7.}]", W(1.91421)),
    ("WeierstrassHalfPeriodW1[{1., 2.}]", W(1.30836)),
    ("WeierstrassHalfPeriodW2[{11.2, 7}]", W(-0.924589 - 1.5114j, 5)),
    ("N[WeierstrassHalfPeriodW2[{11, 7}], 20]", HP("-0.92703733865068595921", "-1.8540746773013719184")),
    ("WeierstrassP[WeierstrassHalfPeriodW2[{11., 7.}], {11., 7.}]", W(-0.914214)),
    ("WeierstrassHalfPeriodW3[{11.2, 7.1}]", W(1.60357j)),
    ("N[WeierstrassHalfPeriodW3[{11, 7}], 20]", HP("0", "1.8540746773013719184", digits=20)),
    ("WeierstrassHalfPeriodW3[{1., 2.}]", W(0.654182 + 1.22937j)),
    ("WeierstrassInvariantG2[{1.1, 0.5 I}]", W(129.91, 5)),
    ("N[WeierstrassInvariantG2[{1, 2 I}], 50]", HP("8.1242184430530170907174818823599001851017295858586")),
    ("WeierstrassInvariantG2[{1, I}]", "Gamma[1/4]^8/(256 Pi^2)"),
    ("WeierstrassInvariantG3[{1., 2. I}]", W(4.44305)),
    ("N[WeierstrassInvariantG3[{1, 2 I}], 50]", HP("4.4430520449459499906285496780843793161520527567568")),
    ("WeierstrassInvariantG3[{1, I}]", "0"),
    ("WeierstrassInvariants[{1., 2. I}]", WL(8.12422, 4.44305)),
    ("N[WeierstrassInvariants[{1, 2 I}], 50]", HPL("8.1242184430530170907174818823599001851017295858586",
                                                   "4.4430520449459499906285496780843793161520527567568")),
    ("WeierstrassP[2.6, WeierstrassInvariants[{1.5, 1 - I}]]", W(6.2388 + 0.0336876j, 5)),
    ("WeierstrassInvariants[{1, I}]", "{Gamma[1/4]^8/(256 Pi^2), 0}"),
    ("WeierstrassE1[{11.5, 7.2}]", W(1.94899)),
    ("N[WeierstrassE1[{11, 7}], 20]", HP("1.9142135623730950488")),
    ("WeierstrassE1[{0, 1}]", "2^(-2/3)"),
    ("WeierstrassE2[{11.5, 7.2}]", W(-0.812992)),
    ("N[WeierstrassE2[{11, 7}], 20]", HP("-0.91421356237309504880")),
    ("WeierstrassE1[{1., 2.}] + WeierstrassE2[{1., 2.}] + WeierstrassE3[{1., 2.}]", Zero(1e-14)),
    ("WeierstrassE3[{11.5, 7.2}]", W(-1.136, 4)),
    ("N[WeierstrassE3[{11, 7}], 20]", HP("-1.0000000000000000000")),
    ("WeierstrassE3[{4., 0.}]", W(-1.0)),
    ("WeierstrassEta1[{9., 3.}]", W(0.805442)),
    ("WeierstrassEta1[{0.5 + I, 1.}]", W(0.285445 - 0.51725j, 5)),
    ("N[WeierstrassEta1[{23, 7}], 50]", HP("0.98398141010786928852371623983711870355773272676397")),
    ("WeierstrassEta2[{9., 3.}]", W(-0.805442 + 0.604487j)),
    ("WeierstrassEta2[{1 + 0.5 I, -0.7}]", W(0.457289 + 0.834123j)),
    ("N[WeierstrassEta2[{24, 7}], 50]", HP("-0.991386217975990631491745820156510653777",
                                           "0.86549409983037837614577279018705392574024724534725", digits=45)),
    ("WeierstrassEta3[{9., 3.}]", W(-0.604487j)),
    ("WeierstrassEta3[{0.3 - 0.4 I, 0.5}]", W(-0.258456 - 0.456944j)),
    ("N[WeierstrassEta3[{25, 7}], 50]", HP("0", "-0.87947779477689592720590813065963186945576655456512", digits=50)),
    ("WeierstrassSigma[2.2, {1, 2}]", W(1.37488)),
    ("WeierstrassSigma[.7, {1, 2}]", W(0.699103)),
    ("WeierstrassSigma[32., {3, 2}]", W(-7.16695e126)),
    ("N[WeierstrassSigma[2/3, {1, 2}], 50]", HP("0.66597844582403648084932694059313699423529330401322")),
    ("WeierstrassSigma[0., {1, 2}]", Zero(0)),
    ("WeierstrassZeta[3., {1, 2}]", W(3.94698)),
    ("WeierstrassZeta[.7, {1, 2}]", W(1.42044)),
    ("WeierstrassZeta[32., {3, 2}]", W(19.9917)),
    ("N[WeierstrassZeta[2/3, {1, 2}], 50]", HP("1.4931702747494321088274560836451392289389614443335")),
    ("WeierstrassZeta[WeierstrassHalfPeriodW1[{g2, g3}], {g2, g3}]", Unev("WeierstrassZeta")),
    ("InverseWeierstrassP[7., {1, 2}]", W(-0.378197)),
    ("InverseWeierstrassP[11., {3, 2}]", W(-0.301707)),
    ("WeierstrassP[InverseWeierstrassP[2., {1, 2}], {1, 2}]", Near(2.0, 1e-12)),
    ("WeierstrassPPrime[InverseWeierstrassP[{2., Sqrt[28.]}, {1, 2}], {1, 2}]", Near(28 ** 0.5, 1e-12)),
    ("InverseWeierstrassP[{2., 3.}, {1., 2.}]", R),
    ("EllipticExp[-0.4, {4, 1}]", WL(5.04383, 15.3336)),
    ("EllipticExp[5., {1, 2}]", WL(1.01198, 2.02099)),
    ("EllipticExp[-.4, {3, 2}]", WL(5.28205, 15.5446)),
    ("EllipticLog[EllipticExp[-0.4, {4, 1}], {4, 1}]", Near(-0.4, 1e-12)),
    ("EllipticExpPrime[-0.4, {4., 1}]", WL(30.6673, 117.671)),
    ("EllipticExpPrime[6., {4, 1}]", WL(-51.7169, 233.033)),
    ("EllipticExpPrime[-.2, {3, 2}]", WL(249.92, 1875.2, sig=5)),
    ("N[EllipticExpPrime[1/3, {5, 1}], 50]", HPL("-53.166675218874014288173620125692182325446361440655",
                                                 "243.86126775585559395177885255870578504738440988907")),
    ("EllipticLog[{0.3, Sqrt[0.3^3 - 5 0.3^2 + 0.3]}, {-5, 1}]", W(0.725631 + 1.06818j)),
    ("EllipticLog[{0.3, -Sqrt[0.3^3 - 5 0.3^2 + 0.3]}, {-5, 1}]", W(0.725631 - 1.06818j)),
    ("EllipticLog[{2, 5}, {3, 2.5}]", W(-0.581447)),
    ("EllipticLog[{1, 2}, {1, 2.}]", W(-0.814338)),
    ("N[EllipticLog[{1, 3}, {2, 6}], 25]", HP("-0.7098518800801157325978000")),
    ("EllipticLog[{1., 5.}, {1, 2}]", R),
]

# ───────────────────────────── Heun ─────────────────────────────
HEUN_A = "9/10, -14/10, 12/100, 3/100, -2/10"
HEUN = [
    ("HeunB[-0.2, 1.3, 0.12, -0.14, 4.32, -.3]", W(1.418, 4)),
    (f"N[HeunB[{HEUN_A}, 1/10], 50]", HP("1.7878024313493235428677")),
    ("HeunB[1.2 + I, -1.4, 0.12, 0.03, -0.2, 0.1]", W(2.0228 + 0.925694j, 5)),
    ("HeunB[q, al, ga, de, ep, 0]", "1"),
    ("HeunBPrime[1.2, -0.6, -0.7, -0.18, 0.3, 0.12]", W(-2.47884)),
    (f"N[HeunBPrime[{HEUN_A}, 1/10], 50]", HP("8.2809937699640175389115837491871161916628359551840")),
    ("HeunBPrime[1.2 + I, -1.4, 0.12, 0.03, -0.2, 0.1]", W(10.4714 + 10.2243j)),
    ("HeunC[4, -0.6, -0.7, -0.18, 0.3, 0.12]", W(1.05543)),
    (f"N[HeunC[{HEUN_A}, 1/10], 50]", HP("0.26449228718001586011180211657813936597828909635640")),
    ("HeunC[1.2 + I, -1.4, 0.12, 0.03, -0.2, 0.1]", W(-0.00170352 - 0.753041j)),
    ("HeunC[q, al, ga, de, ep, 0]", "1"),
    ("HeunCPrime[1.3, 0.12 + 3 I, -0.14, 4.32, 0.12, -.3]", W(4.29966 + 0.0426695j)),
    (f"N[HeunCPrime[{HEUN_A}, 1/10], 50]", HP("-7.1858848911545103559309253388473269534619213360227")),
    ("HeunCPrime[1.2 + I, -1.4, 0.12, 0.03, -0.2, 0.1]", W(-9.99377 - 6.68678j)),
    ("HeunD[4, -0.6, -0.7, -0.18, 0.3, 0.12]", W(6.94565)),
    (f"N[HeunD[{HEUN_A}, 1/10], 50]", HP("5.8303554938604013966662462975483651991652666629913")),
    ("HeunD[1.2 + I, -1.4, 0.12, 0.03, -0.2, 0.1]", W(5.84296 + 5.5737j, 5)),
    ("HeunDPrime[1.2, -0.6, -0.7, -0.18, 0.3, 0.12]", W(-3.69194)),
    (f"N[HeunDPrime[{HEUN_A}, 1/10], 50]", HP("-57.0614777043271726450648802762755964394804679934")),
    ("HeunDPrime[1.2 + I, -1.4, 0.12, 0.03, -0.2, 0.1]", W(-45.7891 - 91.053j, 5)),
    ("HeunG[4.3 + 0.1 I, -0.2, 1.3, 0.12, -0.14, 4.32, 0.12]", W(1.05232 - 0.00117174j)),
    ("N[HeunG[12/10, 9/10, -14/10, 12/100, 3/100, -2/10, 1/10], 50]",
     HP("3.56908496265667212886240725085997269076479622461")),
    ("HeunG[1.2 + I, 0.9, -1.4, 0.12, 0.03, -0.2, 0.1]", W(2.47167 - 1.27074j)),
    ("HeunG[a, q, al, be, ga, de, 0]", "1"),
    ("HeunGPrime[4.3 + 0.1 I, -0.2, 1.3, 0.12, -0.14, 4.32, 0.12]", W(0.565564 - 0.012258j, 5)),
    ("N[HeunGPrime[12/10, 9/10, -14/10, 12/100, 3/10, -2/10, 1/10], 50]",
     HP("2.6835505304834765021328295059135211396162298321832")),
    ("HeunGPrime[1.2 + I, 0.9, -1.4, 0.12, 0.03, -0.2, 0.1]", W(14.6702 - 13.1505j)),
    ("HeunT[-0.2, 1.3, 0.12, -0.14, 4.32, -.3]", W(0.996747)),
    (f"N[HeunT[{HEUN_A}, 1/10], 50]", HP("1.0047183377847068369792470512980993201156669326703")),
    ("HeunT[1.2 + I, -1.4, 0.12, 0.03, -0.2, 0.1]", W(1.00621 + 0.00499033j)),
    ("HeunTPrime[4, -0.6, -0.7, -0.18, 0.3, 0.12]", W(0.510465)),
    (f"N[HeunTPrime[{HEUN_A}, 1/10], 50]", HP("0.096585327479854003676331464822631958273079786670257")),
    ("HeunTPrime[1.2 + I, -1.4, 0.12, 0.03, -0.2, 0.1]", W(0.12635 + 0.0998185j, 5)),
]

# ───────────────────────────── Mathieu ─────────────────────────────
MATHIEU = [
    ("MathieuC[2, 1, 3.2]", W(-0.899531)),
    ("N[MathieuC[2, 1, 37/10], 50]", HP("-0.70674734130376727046760504036431438649802416523011")),
    ("MathieuC[a, 0, z]", "Cos[Sqrt[a] z]"),
    ("MathieuC[I, -2, 2.5 + I]", R),
    ("MathieuS[2, 1, 3.2]", W(-0.768098)),
    ("N[MathieuS[2, 1, 37/10], 50]", HP("-0.94825674959404574160708085731501659004752955180031")),
    ("MathieuS[a, 0, z]", "Sin[Sqrt[a] z]"),
    ("MathieuS[2, 1, 0]", "0"),
    ("MathieuCPrime[2, 1, 3.2]", W(0.335938)),
    ("N[MathieuCPrime[2, 1, 37/10], 50]", HP("0.50336559268757053224923503277363582282642163845699")),
    ("MathieuCPrime[a, 0, z]", "-Sqrt[a] Sin[Sqrt[a] z]"),
    ("MathieuSPrime[2, 1, 3.2]", W(-0.414591)),
    ("N[MathieuSPrime[2, 1, 37/10], 50]", HP("-0.217405424269757753397762113805770412822845895118")),
    ("MathieuSPrime[a, 0, z]", "Sqrt[a] Cos[Sqrt[a] z]"),
    ("MathieuCharacteristicA[1, 0.5]", W(1.46677)),
    ("MathieuCharacteristicA[-2 Pi, 0.5]", W(39.4817)),
    ("MathieuCharacteristicA[Pi, 0.]", W(9.8696, 5)),
    ("N[MathieuCharacteristicA[15/17, 5], 25]", HP("-5.790418614934502217370044")),
    ("MathieuCharacteristicB[1, 0.5]", W(0.470654)),
    ("N[MathieuCharacteristicB[Pi/2, Pi/4]]", W(2.63935)),
    ("MathieuCharacteristicB[Pi, 0.5]", W(9.88373)),
    ("N[MathieuCharacteristicB[15/17, 5/3], 25]", HP("-0.9554651931775240002199916")),
    ("MathieuCharacteristicB[0, 1.]", R),
    ("MathieuCharacteristicExponent[2, 0.5]", W(1.36951)),
    ("MathieuCharacteristicExponent[0.2, 0.4]", W(0.557654)),
    ("N[MathieuCharacteristicExponent[Pi, E]]", W(2.30738)),
    ("MathieuCharacteristicExponent[5, {0.5, 1.0, 1.5}]", WL(2.22879, 2.203, 2.14004)),
    ("N[MathieuCharacteristicExponent[17/3, 1/2], 30]", HP("2.37472448001814669297498099483")),
    ("Cos[Pi MathieuCharacteristicExponent[MathieuCharacteristicA[1.3, 2.], 2.]] - Cos[1.3 Pi]", Zero(1e-10)),
]

# ───────────────────────────── Lame ─────────────────────────────
LAME = [
    ("LameC[0.9, 2, 0.7, 0.3]", W(-0.234963)),
    ("N[LameC[9/10, 2, 7/10, 3/10], 50]", HP("-0.234963250953280039708551130650")),
    ("LameC[1.2 + I, 2, 0.7, 0.3]", R),
    ("LameCPrime[0.9, 2, 0.7, 0.3]", W(1.78341)),
    ("N[LameCPrime[9/10, 2, 7/10, 3/10], 50]", HP("1.78341265394981401944197249599")),
    ("LameCPrime[1.2 + I, 2, 0.7, 0.3]", R),
    ("LameS[0.9, 2, 0.7, 0.3]", W(0.965484)),
    ("N[LameS[9/10, 2, 7/10, 3/10], 50]", HP("0.965484211027996951737059622400")),
    ("LameS[1.2 + I, 2, 0.7, 0.3]", R),
    ("LameSPrime[0.9, 2, 0.7, 0.3]", W(0.479848)),
    ("N[LameSPrime[9/10, 2, 7/10, 3/10], 50]", HP("0.479848337653796306310665129362")),
    ("LameSPrime[1.2 + I, 2, 0.7, 0.3]", R),
    ("LameEigenvalueA[0.9, 2, 0.3]", W(3.63031)),
    ("N[LameEigenvalueA[9/10, 2, 3/10], 50]", HP("3.6303057475980740694316986183233411788696149665951")),
    ("LameEigenvalueA[2, 1, 0.5]", Near(3.0, 1e-12)),            # nu = 2: Lame polynomial eigenvalue 1 + 4 m
    ("LameEigenvalueA[1.2 + I, 2, 0.3]", R),
    ("LameEigenvalueB[0.9, 2, 0.3]", W(3.63196)),
    ("N[LameEigenvalueB[9/10, 2, 3/10], 50]", HP("3.63195531518681351881143360184228098739702")),
    ("LameEigenvalueB[2, 1, 0.5]", Near(1.5, 1e-12)),            # nu = 2: eigenvalue 1 + m
    ("LameEigenvalueB[1.2 + I, 2, 0.3]", R),
]

# ───────────────────────────── spheroidal ─────────────────────────────
SPHEROIDAL = [
    ("SpheroidalEigenvalue[2, 1, 0.5]", W(5.8569, 5)),
    ("SpheroidalEigenvalue[5, 0, 0]", "30"),
    ("N[SpheroidalEigenvalue[117, 5, 1], 50]", HP("13805.499105841203161740672967055183681641063997782")),
    ("SpheroidalEigenvalue[2, 0.5, -5]", R),
    ("SpheroidalPS[3, 2, 1, 0.5]", W(5.65037)),
    ("SpheroidalPS[5, 1, 0, .3]", W(0.160798)),
    ("SpheroidalPS[1, 0, 4, 2.]", W(-0.0267447)),
    ("N[SpheroidalPS[7, 5, 4, 0], 50]", HP("5197.9491917931803048477313438355929826745200256550")),
    ("N[SpheroidalPS[2, 1, 1/3, 1/2], 20]", HP("-1.3008757335723961523")),
    ("SpheroidalPSPrime[3, 2, 1, 0.5]", W(3.45433)),
    ("SpheroidalPSPrime[5, 1, 0, .3]", W(10.9149)),
    ("N[SpheroidalPSPrime[7, 0, 1/4, 1/3], 50]", HP("1.9599516004934291475641508479629864322857174354952")),
    ("SpheroidalPSPrime[1, 0.5, -5, 3]", R),
    ("SpheroidalS1[1, 0, 2.5, 1]", W(0.584507)),
    ("SpheroidalS1[5, 1, .1, .3]", W(-3.93471e-12j)),
    ("SpheroidalS1[1, 0, 1.5, 2]", W(0.394908)),
    ("N[SpheroidalS1[7, 5, 1, 2], 50]", HP("0.000027389830517676850780382752441683552371410431832")),
    ("SpheroidalS1Prime[1, 0, 2, 2.5]", W(-0.382992)),
    ("SpheroidalS1Prime[5, 1, .1, .3]", W(-2.66681e-10j)),
    ("SpheroidalS1Prime[1, 0, 1.5, 2]", W(-0.2875297849816411, 15)),
    ("N[SpheroidalS1Prime[7, 5, 1, 2], 50]", HP("0.00011596617199370606635908649991099482322269365627795")),
    ("N[SpheroidalS1Prime[2, 1, 1/3, -11], 20]", HP("-0.015058898228028541446")),
    ("SpheroidalS2[2, 0, 2.5, 5]", W(0.0819121)),
    ("SpheroidalS2[1, 0, 1.5, 2]", W(0.0110657)),
    ("N[SpheroidalS2[7, 5, 1, 2], 50]", HP("-1350.9438761697030096210897423232373668564299204611")),
    ("SpheroidalS2[1, 0, 1.5, 0.5]", R),
    ("SpheroidalS2Prime[3, 0, 10, 1.2]", W(-1.50116)),
    ("N[SpheroidalS2Prime[2, 1, 1/3, -11], 20]", HP("-0.087077652227300202593")),
    ("SpheroidalS2Prime[5, 1, .1, .3]", R),
    # Wronskian W{S1, S2} = 1/(c (z^2 - 1)) (DLMF 30.11.9)
    ("SpheroidalS1[3, 1, 1.5, 2.] SpheroidalS2Prime[3, 1, 1.5, 2.] - "
     "SpheroidalS1Prime[3, 1, 1.5, 2.] SpheroidalS2[3, 1, 1.5, 2.]", Near(1 / (1.5 * 3), 1e-11)),
    ("SpheroidalJoiningFactor[1, 0, 0.5]", W(0.1658506183761293, 16)),
    ("SpheroidalJoiningFactor[5, 1, .1]", W(2.443201135853705e-11, 15)),
    ("N[SpheroidalJoiningFactor[247, 5, 1], 50]", HP("1.6598693698064881972828216206973998269588995876050e-646")),
    ("N[SpheroidalJoiningFactor[2, -1, -1], 20]", HP("0.13478393896276848251")),
    ("SpheroidalS1[2, 0, 1., 0.7] - SpheroidalJoiningFactor[2, 0, 1.] SpheroidalPS[2, 0, 1., 0.7]", Zero(1e-14)),
    ("SpheroidalJoiningFactor[1, 0.5, -5]", R),
    # SpheroidalQS: documented values (DLMF 30.8.9 series)
    ("SpheroidalQS[3, 2, 1, 0.5]", W(-0.263019)),
    ("SpheroidalQS[5, 1, .1, .3]", W(-3.00656)),
    ("SpheroidalQS[3, 2, 1., .2]", W(-6.66162)),
    ("N[SpheroidalQS[23, 5, 2 I, 2/3]]", W(-1.7082389401816712e6, 14)),
    ("SpheroidalQS[1, 0, 0, 0.5]", W(-0.7253469278329727, 15)),
    ("SpheroidalQS[2, 1, 0.5 + I, 0.3]", R),
    ("SpheroidalQS[2, 1, 1/3, -11.]", W(-190.54416091822214467 + 0.00036531273120174j, 12)),
    ("SpheroidalQS[2, 1, 0, 1/2]", Pred(lambda g: str(g) == "LegendreQ[2, 1, 1/2]", "LegendreQ[2, 1, 1/2]")),
    ("SpheroidalQS[3, 2, 0., 0.4] - LegendreQ[3, 2, 0.4]", Zero(1e-13)),
    ("SpheroidalQS[3/2, 1, 1, 0.5]", R),
    ("SpheroidalQS[2, 1, 1, 1.]", R),
    # parity Qs(-x) = (-1)^(n-m+1) Qs(x) (DLMF 30.5.2)
    ("SpheroidalQS[4, 1, 1.3, -0.35] - SpheroidalQS[4, 1, 1.3, 0.35]", Zero(1e-12)),
    # Wronskian (1 - x^2) W{Ps, Qs} is constant (DLMF 30.5.4)
    ("With[{x = 0.3, y = -0.6}, (1 - x^2) (SpheroidalPS[3, 1, 1.2, x] SpheroidalQSPrime[3, 1, 1.2, x] - "
     "SpheroidalPSPrime[3, 1, 1.2, x] SpheroidalQS[3, 1, 1.2, x]) - (1 - y^2) (SpheroidalPS[3, 1, 1.2, y] "
     "SpheroidalQSPrime[3, 1, 1.2, y] - SpheroidalPSPrime[3, 1, 1.2, y] SpheroidalQS[3, 1, 1.2, y])]", Zero(1e-11)),
    ("SpheroidalQSPrime[3, 2, 1, 0.5]", W(28.5884)),
    ("SpheroidalQSPrime[5, 1, .1, .3]", W(0.93037)),
    ("SpheroidalQSPrime[3, 2, -.3, .2]", W(12.5829)),
    ("N[SpheroidalQSPrime[5, 3, I 2, 1/2]]", W(-281.371)),
    ("SpheroidalQSPrime[2, 2, 1/3, 0.2111111111111111]", W(8.0651323397698955, 14)),
    ("SpheroidalQSPrime[3, 2, 1, .2`40]", HP("12.976371063736039748663845236251542136510")),
    ("SpheroidalQSPrime[1, 0, 0, 0.5]", W(1.2159728110007215, 15)),
    # the ODE (1-x^2) y'' - 2 x y' + (lambda + c^2 (1-x^2) - m^2/(1-x^2)) y = 0, y'' by a central difference
    ("With[{x = 0.4, h = 10^-5, c = 1.7}, (1 - x^2) (SpheroidalQSPrime[3, 2, c, x + h] - SpheroidalQSPrime[3, 2, c, "
     "x - h])/(2 h) - 2 x SpheroidalQSPrime[3, 2, c, x] + (SpheroidalEigenvalue[3, 2, c] + c^2 (1 - x^2) - "
     "4/(1 - x^2)) SpheroidalQS[3, 2, c, x]]", Zero(1e-6)),
]

def _ci_ok(pde, u, xs, nconst):
    """A CompleteIntegral result: substituted back the PDE holds, with nconst distinct constants C[k]."""
    def ok(g):
        if not (isinstance(g, Expr) and g.head == "List" and g.args):
            return False
        for sol in g.args:
            chk = ev(f"Simplify[({pde}) /. {sol}]")
            if str(chk) not in ("True", "{True}"):
                return False
            if len(set(__import__("re").findall(r"C\[(\d+)\]", str(sol)))) != nconst:
                return False
        return True
    return Pred(ok, f"complete integral of {pde}")


def _str(want):
    return Pred(lambda g: str(g) == want, want)


COMPLETE = [
    # documented outputs (reference.wolfram.com/language/ref/CompleteIntegral.html)
    ("CompleteIntegral[D[u[x, y], x] - D[u[x, y], y]^2 == Cos[x], u[x, y], {x, y}]",
     _str("{{u[x, y] -> x*C[1]^2 + y*C[1] + C[2] + Sin[x]}}")),
    ("CompleteIntegral[D[u[x, y, z], x]^3 == D[u[x, y, z], y] + D[u[x, y, z], z], u[x, y, z], {x, y, z}]",
     _str("{{u[x, y, z] -> x*C[1] + y*C[2] + z*(C[1]^3 - C[2]) + C[3]}}")),
    ("CompleteIntegral[D[u[x, y, z], x]^3 == D[u[x, y, z], y] + D[u[x, y, z], z] && u[0, 1, 1] == 1 && "
     "u[1, 0, 1] == 1 && u[0, 0, 1] == 1/2, u[x, y, z], {x, y, z}]",
     Pred(lambda g: str(ev(f"Simplify[(u[x, y, z] /. {g}[[1]]) - (7 + 4 x + 4 y - 3 z)/8]")) == "0", "(7+4x+4y-3z)/8")),
    ("CompleteIntegral[u[x, y] == x D[u[x, y], x] + y D[u[x, y], y] + (D[u[x, y], x]^2 + D[u[x, y], y]^2)/2, "
     "u, {x, y}]", _str("{{u -> Function[{x, y}, x*C[1] + y*C[2] + C[1]^2/2 + C[2]^2/2]}}")),
    # same family as the documented C[1] + x C[2] - t C[2]^2 and C[1] + y C[2] + x (1 - C[2]^2), constants renumbered
    ("CompleteIntegral[D[u[t, x], t] + D[u[t, x], x]^2 == 0, u[t, x], {t, x}]",
     _str("{{u[t, x] -> -t*C[1]^2 + x*C[1] + C[2]}}")),
    ("CompleteIntegral[D[u[x, y], x] + D[u[x, y], y]^2 == 1, u, {x, y}]",
     _ci_ok("D[u[x, y], x] + D[u[x, y], y]^2 == 1", "u", "{x, y}", 2)),
    ("CompleteIntegral[D[u[x, y], x] - x D[u[x, y], y] == 0, u, {x, y}]",
     _ci_ok("D[u[x, y], x] - x D[u[x, y], y] == 0", "u", "{x, y}", 2)),
    ("CompleteIntegral[D[u[x, y], x] D[u[x, y], y] == 1, u, {x, y}]",
     _ci_ok("D[u[x, y], x] D[u[x, y], y] == 1", "u", "{x, y}", 2)),
    ("CompleteIntegral[D[u[x, y], x]^2 + D[u[x, y], y]^2 == 1, u, {x, y}]",
     _ci_ok("D[u[x, y], x]^2 + D[u[x, y], y]^2 == 1", "u", "{x, y}", 2)),
    # outside the supported classes it stays unevaluated (never a wrong answer)
    ("CompleteIntegral[D[u[x, y], x] + D[u[x, y], y] == u[x, y]^2, u[x, y], {x, y}]",
     Pred(lambda g: isinstance(g, Expr) and g.head == "CompleteIntegral", "unevaluated")),
    ("CompleteIntegral[D[u[x, y], {x, 2}] == 0, u[x, y], {x, y}]",
     Pred(lambda g: isinstance(g, Expr) and g.head == "CompleteIntegral", "unevaluated")),
    ("CompleteIntegral[D[u[x, y], x] == 1, u[x, y], x]", R),
]


CASES = ZEROS + CARLSON + WAVES + THETA + JACOBI + WEIER + HEUN + MATHIEU + LAME + SPHEROIDAL + COMPLETE


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w07(case):
    check(case)
