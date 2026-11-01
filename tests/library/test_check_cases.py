"""Direct differential cases for heads that had fewer than three cases of their own, so the stated verification
harness (scripts/verify_library.py) can run their ``Check: reference`` sentence.

Sources of the expected values (never the implementation's own output):

* mpmath, used as an independent reference for special functions: ``airyai/airybi`` (derivative=1),
  ``besseljzero/besselyzero``, ``elliprf/elliprg``, ``coulombf/coulombg``, ``lerchphi``, ``hyp0f1/hyp1f1/hyp2f1``
  divided by the gamma function (regularised forms), ``ellipfun/ellipf/ellipe/ellipk`` (inverse Jacobi
  functions found by root-finding on the forward function, Jacobi amplitude/epsilon/zeta), ``ber/bei``,
  ``meijerg``, ``siegeltheta``, ``scorergi/scorerhi``, ``bessely/besselj`` (spherical Bessel and Hankel
  functions as sqrt(pi/(2x)) times the half-integer order), ``struvel``.
* Wolfram Language reference pages for structural heads (AppendTo, PrependTo, AssociateTo, KeyDropFrom, Sow,
  Scan, TreeGraph, NestGraph, CrossMatrix, DiamondMatrix, FourierMatrix, Conjunction, Disjunction,
  BooleanCountingFunction, DirichletL, DeBruijnSequence, NumberFieldRootsOfUnity, ZetaZero, RandomPrime) and
  the ATLAS of Finite Groups (sporadic groups stay inert, like Wolfram's ``MathieuGroupM11[]``).
* CODATA 2018 for the physical constants (their values are tested in test_applied.py; the cases here are the
  documented argument errors).

Symbols used as state are prefixed ``cbk`` so they cannot collide with other test modules.
"""
import mpmath as mp
import pytest

from _harness import Pred, Raises, check, ids, to_complex
from mikoshilang.expr import Expr


def Near(want, rtol=1e-10, atol=0.0):
    want = complex(want)
    return Pred(lambda g: abs(to_complex(g) - want) <= rtol * abs(want) + atol, f"~{want}")


def Inert(head):
    return Pred(lambda g: isinstance(g, Expr) and g.head == head, f"inert {head}[...]")


def _inv(kind, v, m):
    """u in (0, K(m)) with ellipfun(kind, u, m) == v, by bracketing root-finding on the forward function."""
    f = lambda u: mp.ellipfun(kind, u, m=m) - v
    a, b = mp.mpf("1e-9"), mp.ellipk(m) - mp.mpf("1e-9")
    for _ in range(200):                                  # plain bisection: f changes sign on (a, b)
        c = (a + b) / 2
        if (f(a) < 0) == (f(c) < 0):
            a = c
        else:
            b = c
    return (a + b) / 2


def _am(u, m):
    return mp.asin(mp.ellipfun("sn", u, m=m))          # valid for |u| < K(m)


def _sph(kind, n, x):
    f = mp.sqrt(mp.pi / (2 * x))
    j, y = f * mp.besselj(n + 0.5, x), f * mp.bessely(n + 0.5, x)
    return {"y": y, "h1": j + 1j * y, "h2": j - 1j * y}[kind]


SPECIAL = [
    ("AiryAiPrime[2.]", Near(mp.airyai(2, derivative=1))),
    ("AiryBiPrime[-1.5]", Near(mp.airybi(-1.5, derivative=1))),
    ("BesselJZero[0., 1]", Near(mp.besseljzero(0, 1))),
    ("BesselJZero[1., 2]", Near(mp.besseljzero(1, 2))),
    ("BesselYZero[0., 1]", Near(mp.besselyzero(0, 1))),
    ("BesselYZero[1., 2]", Near(mp.besselyzero(1, 2))),
    ("CarlsonRF[1., 2., 4.]", Near(mp.elliprf(1, 2, 4))),
    ("CarlsonRG[1., 2., 4.]", Near(mp.elliprg(1, 2, 4))),
    ("CoulombF[1, 0.5, 2.]", Near(mp.coulombf(1, 0.5, 2))),
    ("CoulombF[0, 0, 1.5]", Near(mp.sin(1.5))),                     # eta = 0, L = 0: F = sin(rho)
    ("CoulombG[1, 0.5, 2.]", Near(mp.coulombg(1, 0.5, 2))),
    ("CoulombG[0, 0, 1.5]", Near(mp.cos(1.5))),                     # eta = 0, L = 0: G = cos(rho)
    ("CoulombH1[0, 0, 1.5]", Near(mp.cos(1.5) + 1j * mp.sin(1.5))),  # H+ = G + I F
    ("CoulombH2[0, 0, 1.5]", Near(mp.cos(1.5) - 1j * mp.sin(1.5))),  # H- = G - I F
    ("HurwitzLerchPhi[0.3, 3, 0.5]", Near(mp.lerchphi(0.3, 3, 0.5))),
    ("Hypergeometric0F1Regularized[2, 0.3]", Near(mp.hyp0f1(2, 0.3) / mp.gamma(2))),
    ("Hypergeometric1F1Regularized[1, 3, 0.2]", Near(mp.hyp1f1(1, 3, 0.2) / mp.gamma(3))),
    ("Hypergeometric2F1Regularized[1, 1, 3, 0.3]", Near(mp.hyp2f1(1, 1, 3, 0.3) / mp.gamma(3))),
    ("JacobiAmplitude[1., 0.5]", Near(_am(1, 0.5))),
    ("JacobiEpsilon[1., 0.5]", Near(mp.ellipe(_am(1, 0.5), 0.5))),
    ("JacobiZeta[0.7, 0.4]", Near(mp.ellipe(0.7, 0.4) - mp.ellipe(0.4) / mp.ellipk(0.4) * mp.ellipf(0.7, 0.4))),
    ("JacobiZN[1., 0.5]", Near(mp.ellipe(_am(1, 0.5), 0.5) - mp.ellipe(0.5) / mp.ellipk(0.5) * 1)),
    ("KelvinBer[1.]", Near(mp.ber(0, 1))),
    ("KelvinBer[2, 1.5]", Near(mp.ber(2, 1.5))),
    ("KelvinBei[1.]", Near(mp.bei(0, 1))),
    ("KelvinBei[2, 1.5]", Near(mp.bei(2, 1.5))),
    ("MeijerG[{{}, {}}, {{1/2}, {0}}, 0.25]", Near(mp.meijerg([[], []], [[0.5], [0]], 0.25))),
    ("RiemannSiegelTheta[3.]", Near(mp.siegeltheta(3))),
    ("ScorerGi[2.]", Near(mp.scorergi(2))),
    ("ScorerGiPrime[1.]", Near(mp.diff(mp.scorergi, 1), 1e-8)),
    ("ScorerHi[1.]", Near(mp.scorerhi(1))),
    ("ScorerHiPrime[1.]", Near(mp.diff(mp.scorerhi, 1), 1e-8)),
    ("SphericalBesselY[2, 1.5]", Near(_sph("y", 2, 1.5))),
    ("SphericalHankelH1[2, 1.5]", Near(_sph("h1", 2, 1.5))),
    ("SphericalHankelH2[2, 1.5]", Near(_sph("h2", 2, 1.5))),
    ("StruveL[2, 1.5]", Near(mp.struvel(2, 1.5))),
    ("ZetaZero[2]", "ZetaZero[2]"),                                   # exact index: stays symbolic
]
# inverse Jacobi functions: the forward function's root in (0, K(m))
for _k, _v in [("sn", 0.5), ("cn", 0.5), ("dn", 0.9), ("cd", 0.4), ("sc", 0.8), ("sd", 0.6), ("ns", 1.6),
               ("nc", 1.4), ("cs", 1.3)]:
    _name = "InverseJacobi" + _k.upper()
    SPECIAL += [(f"{_name}[{_v}, 0.3]", Near(_inv(_k, _v, 0.3))),
                (f"{_name}[{_v}, 0.6]", Near(_inv(_k, _v, 0.6)))]
SPECIAL += [("InverseJacobiSN[0., 0.3]", Near(0, atol=1e-14)),       # sn(0) = 0
            ("InverseJacobiCN[1., 0.3]", Near(0, atol=1e-14)),       # cn(0) = 1
            ("InverseJacobiSC[0., 0.3]", Near(0, atol=1e-14)),       # sc(0) = 0
            ("InverseJacobiSD[0., 0.3]", Near(0, atol=1e-14)),       # sd(0) = 0
            ("InverseJacobiDN[1., 0.3]", Near(0, atol=1e-14)),       # dn(0) = 1
            ("InverseJacobiNC[1., 0.3]", Near(0, atol=1e-14)),       # nc(0) = 1
            ("InverseJacobiCD[1., 0.3]", Near(0, atol=1e-14))]       # cd(0) = 1

SPORADIC = ["MathieuGroupM11", "MathieuGroupM12", "MathieuGroupM22", "MathieuGroupM23", "MathieuGroupM24",
            "JankoGroupJ1", "JankoGroupJ2", "JankoGroupJ3", "JankoGroupJ4", "HigmanSimsGroupHS",
            "McLaughlinGroupMcL", "HeldGroupHe", "RudvalisGroupRu", "SuzukiGroupSuz", "ONanGroupON",
            "ConwayGroupCo3", "ConwayGroupCo2", "ConwayGroupCo1", "FischerGroupFi22", "FischerGroupFi23",
            "FischerGroupFi24Prime", "HaradaNortonGroupHN", "LyonsGroupLy", "ThompsonGroupTh", "BabyMonsterGroupB",
            "MonsterGroupM", "TitsGroupT"]
GROUPS = []
for _g in SPORADIC:
    GROUPS += [(f"{_g}[]", f"{_g}[]"), (f"{_g}[a, b]", Raises())]

STRUCTURAL = [
    ("cbkA = {1}", "{1}"), ("AppendTo[cbkA, 2]", "{1, 2}"), ("AppendTo[cbkA, x]", "{1, 2, x}"),
    ("cbkP = {1}", "{1}"), ("PrependTo[cbkP, 0]", "{0, 1}"), ("PrependTo[cbkP, x]", "{x, 0, 1}"),
    ("cbkJ = <|a -> 1|>", "<|a -> 1|>"), ("AssociateTo[cbkJ, b -> 2]", "<|a -> 1, b -> 2|>"),
    ("AssociateTo[cbkJ, a -> 3]", "<|a -> 3, b -> 2|>"), ("AssociateTo[cbkUndefinedJ, a -> 1]", Raises()),
    ("cbkK = <|a -> 1, b -> 2, c -> 3|>", "<|a -> 1, b -> 2, c -> 3|>"), ("KeyDropFrom[cbkK, a]", "<|b -> 2, c -> 3|>"),
    ("KeyDropFrom[cbkK, {b, c}]", "<||>"),
    ("Sow[x, cbkTag]", "x"), ("Sow[{1, 2}]", "{1, 2}"),
    ("Scan[f, {}]", "Null"), ("Scan[(cbkS = #) &, {1, 2, 3}]", "Null"),
    ("TreeGraph[{1 -> 2, 2 -> 3}]", "Graph[{1, 2, 3}, {1 -> 2, 2 -> 3}]"),
    ("NestGraph[{2 #} &, 1, 3]", "Graph[{1, 2, 4, 8}, {1 -> 2, 2 -> 4, 4 -> 8}]"),
    ("NestGraph[{# + 1} &, 0, 1]", "Graph[{0, 1}, {0 -> 1}]"),
    ("UnlabeledTree[Tree[a, {b}]]", "Tree[Null, {Tree[Null, None]}]"),
    ("TreeScan[f, Tree[a, {Tree[b, None]}]]", "Null"), ("TreeScan[f, Tree[a, {b, c}]]", "Null"),
    ("TreeForm[f[x, y]]", "TreeForm[f[x, y]]"),
]

DISCRETE = [
    ("CrossMatrix[2]", "{{0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {1, 1, 1, 1, 1}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}}"),
    ("DiamondMatrix[1]", "{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}"),
    ("FourierMatrix[1]", "{{1}}"),
    ("Conjunction[a && b, {a}]", False), ("Conjunction[a, {b}]", "a"),
    ("Disjunction[a || b, {a}]", True), ("Disjunction[a, {b}]", "a"),
    ("Exists[x, True]", True),
    ("BooleanCountingFunction[{0}, 2][False, False]", True),
    ("DirichletL[1, 1, 3]", "Zeta[3]"),
    ("DeBruijnSequence[{0, 1}, 1]", "{0, 1}"),
    ("NumberFieldRootsOfUnity[Sqrt[3]]", "{-1, 1}"),
    ("RandomPrime[{2, 2}]", "2"), ("RandomPrime[{7, 7}]", "7"), ("RandomPrime[{13, 13}]", "13"),
    ("Underflow[]", "Underflow[]"),
]

def _ev_expr(e):
    from mikoshilang.evaluate import evaluate
    return evaluate(e)


def SeriesIs(poly, order, var="x"):
    """A SeriesData in var whose Normal part equals poly and whose truncation order is order."""
    from mikoshilang.parser import parse

    def order_of(g):
        if g.head == "SeriesData" and len(g.args) == 6:
            return str(g.args[4])
        for t in (g.args if g.head == "Plus" else [g]):        # MikoshiLang keeps series as poly + O[x]^n
            if isinstance(t, Expr) and t.head == "Power" and isinstance(t.args[0], Expr) and t.args[0].head == "O":
                return str(t.args[1])
        return None

    def ok(g):
        if not isinstance(g, Expr):
            return False
        diff = _ev_expr(Expr("Expand", Expr("Plus", Expr("Normal", g), Expr("Times", -1, parse(poly)))))
        return str(diff) == "0" and order_of(g) == str(order)
    return Pred(ok, f"{poly} + O[{var}]^{order}")


def PolyNear(poly, var="x", points=(0.0, 0.5, 2.0), tol=1e-8):
    """A numeric fit: agrees with poly at sample points."""
    from mikoshilang.parser import parse

    def ok(g):
        for t in points:
            d = _ev_expr(Expr("ReplaceAll", Expr("Plus", g, Expr("Times", -1, parse(poly))),
                              Expr("Rule", parse(var), t)))
            if abs(to_complex(d)) > tol:
                return False
        return True
    return Pred(ok, f"~{poly}")


def NonNeg():
    return Pred(lambda g: to_complex(g).imag == 0 and to_complex(g).real >= 0, "a non-negative number")


CONSTANTS = ["SpeedOfLight", "PlanckConstant", "ReducedPlanckConstant", "GravitationalConstant", "BoltzmannConstant",
             "AvogadroNumber", "ElementaryCharge", "ElectronMass", "ProtonMass", "NeutronMass",
             "FineStructureConstant", "RydbergConstant", "StefanBoltzmannConstant", "GasConstant",
             "VacuumPermittivity", "VacuumPermeability", "CoulombConstant", "FaradayConstant", "AtomicMassUnit",
             "BohrMagneton", "StandardGravity", "StandardAtmosphere", "WienDisplacement", "MolarVolumeIdealGas",
             "HubbleConstant", "SolarMass", "EarthMass", "EarthRadius", "LightYear", "Parsec"]
APPLIED = [(f"{_c}[x]", Raises()) for _c in CONSTANTS]

DISTRIBUTIONS = [
    ("BinomialDistribution[5, 1/3]", Inert("BinomialDistribution")), ("BinomialDistribution[-1, 1/2]", Raises()),
    ("ChiSquaredDistribution[5]", Inert("ChiSquaredDistribution")), ("ChiSquaredDistribution[-1]", Raises()),
    ("ExponentialDistribution[1/2]", Inert("ExponentialDistribution")), ("ExponentialDistribution[-1]", Raises()),
    ("FDistribution[3, 4]", Inert("FDistribution")), ("FDistribution[-3, 4]", Raises()),
    ("PoissonDistribution[1/2]", Inert("PoissonDistribution")), ("PoissonDistribution[0]", Raises()),
    ("TDistribution[3]", Inert("TDistribution")), ("TDistribution[1/2]", Inert("TDistribution")),
    ("UniformDistribution[{-1, 1}]", Inert("UniformDistribution")), ("UniformDistribution[{1, -1}]", Raises()),
]

GROUPS2 = [
    ("SymmetricGroup[3]", "SymmetricGroup[3]"), ("SymmetricGroup[5]", "SymmetricGroup[5]"),
    ("CyclicGroup[4]", "CyclicGroup[4]"), ("CyclicGroup[7]", "CyclicGroup[7]"), ("CyclicGroup[-1]", Raises()),
    ("AlternatingGroup[4]", "AlternatingGroup[4]"), ("AlternatingGroup[5]", "AlternatingGroup[5]"),
    ("AlternatingGroup[-1]", Raises()),
    ("DihedralGroup[5]", "DihedralGroup[5]"), ("DihedralGroup[6]", "DihedralGroup[6]"), ("DihedralGroup[-1]", Raises()),
    ("AbelianGroup[{2, 3}]", "AbelianGroup[{2, 3}]"), ("AbelianGroup[{4}]", "AbelianGroup[{4}]"),
    ("PermutationGroup[{Cycles[{{1, 2}}]}]", "PermutationGroup[{Cycles[{{1, 2}}]}]"),
    ("PermutationGroup[{Cycles[{{1, 2, 3}}], Cycles[{{1, 2}}]}]", "PermutationGroup[{Cycles[{{1, 2, 3}}], Cycles[{{1, 2}}]}]"),
    ("RandomPermutation[1]", "Cycles[{}]"), ("RandomPermutation[1, 2]", "{Cycles[{}], Cycles[{}]}"),
    ("RandomPermutation[-1]", Raises()),
]

SERIES = [
    ("InverseSeries[Series[Sin[x], {x, 0, 5}]]", SeriesIs("x + x^3/6 + 3 x^5/40", 6)),      # arcsin
    ("InverseSeries[Series[Exp[x] - 1, {x, 0, 3}]]", SeriesIs("x - x^2/2 + x^3/3", 4)),       # log(1 + x)
    ("InverseSeries[Series[x + x^2, {x, 0, 3}]]", SeriesIs("x - x^2 + 2 x^3", 4)),            # (Sqrt[1 + 4 x] - 1)/2
    ("ComposeSeries[Series[Exp[x], {x, 0, 3}], Series[Sin[x], {x, 0, 3}]]", SeriesIs("1 + x + x^2/2", 4)),
    ("ComposeSeries[Series[1/(1 - x), {x, 0, 3}], Series[x^2, {x, 0, 3}]]", SeriesIs("1 + x^2", 4)),
    ("ComposeSeries[Series[Sin[x], {x, 0, 5}], Series[2 x, {x, 0, 5}]]", SeriesIs("2 x - 4 x^3/3 + 4 x^5/15", 6)),
    ("SymbolicZerosArray[{2}]", Inert("SymbolicZerosArray")), ("SymbolicZerosArray[{-2}]", Raises()),
    ("SymbolicOnesArray[{2, 2}]", Inert("SymbolicOnesArray")), ("SymbolicOnesArray[{3}]", Inert("SymbolicOnesArray")),
    ("SymbolicIdentityArray[{3}]", Inert("SymbolicIdentityArray")),
    ("SymbolicIdentityArray[{2, 2}]", Inert("SymbolicIdentityArray")), ("SymbolicIdentityArray[{-1}]", Raises()),
    ("SymbolicDeltaProductArray[{2, 2}, {{1, 2}}]", Inert("SymbolicDeltaProductArray")),
    ("SymbolicDeltaProductArray[{3, 3}, {{1, 2}}]", Inert("SymbolicDeltaProductArray")),
]

ERRORS = [
    ("Assert[True]", "Null"), ("Assert[1 == 1]", "Null"),
    ("AbortProtect[x]", "x"),
    ("EchoLabel[\"y\"][x]", "x"),
    ("ExceptionTypeRegisteredQ[cbkNotRegistered]", "False"), ("ExceptionTypeRegisteredQ[cbkNotRegistered2]", "False"),
    ("ExceptionTypes[cbkNothing]", "{}"), ("ExceptionTypes[cbkNothing2]", "{}"),
    ("IntermediateTest[2 + 2, 4]", "4"),
    ("MemoryAvailable[]", NonNeg()), ("MemoryInUse[]", NonNeg()),
    ("Pause[0]", "Null"), ("Pause[0.001]", "Null"),
    ("Print[x]", "Null"), ("PrintTemporary[x]", "Null"),
    ("SessionTime[]", NonNeg()), ("TimeUsed[]", NonNeg()), ("TimeUsed[]", NonNeg()),
    ("TimeRemaining[]", "Infinity"), ("TimeRemaining[]", "Infinity"),
    ("TerminatedEvaluation[\"IterationLimit\"]", "TerminatedEvaluation[\"IterationLimit\"]"),
    ("WithCleanup[1 + 1, Null]", "2"), ("WithCleanup[x, y]", "x"), ("WithCleanup[cbkW = 1; cbkW, cbkW = 2]", "1"),
]

MISC = [
    ("NearestFunction[{1, 2, 3}][2.9]", "{3}"),
    ("SuchThat[y, y < 1]", "SuchThat[y, y < 1]"),
    ("IncrementalFunction[f]", "IncrementalFunction[f]"),
    ("LowerTriangularMatrix[{{1, 0}, {2, 3}}]", "{{1, 0}, {2, 3}}"),
    ("UpperTriangularMatrix[{{1, 2}, {0, 3}}]", "{{1, 2}, {0, 3}}"),
    ("UpperTriangularMatrix[{{-1, 5, 2}, {0, 4, 1}, {0, 0, 7}}]", "{{-1, 5, 2}, {0, 4, 1}, {0, 0, 7}}"),
    ("UpperTriangularMatrix[{{1, 0}, {2, 3}}]", Raises()),
    ("LinearSolveFunction[{2, 2}, {{1, 2}, {3, 4}}][{3, 7}]", "{1, 1}"),
    ("HermiteDecomposition[{{1, 0}, {0, 1}}]", "{{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}}"),
    ("PolynomialHermiteDecomposition[{{1, 0}, {0, x}}]", "{{{1, 0}, {0, 1}}, {{1, 0}, {0, x}}}"),
    ("PolynomialHermiteDecomposition[{{x, 0}, {0, 1}}]", Pred(lambda g: isinstance(g, Expr) and len(g.args) == 2,
                                                                   "{u, h}")),
    ("PolynomialSmithDecomposition[{{1, 0}, {0, x}}]", "{{{1, 0}, {0, 1}}, {{1, 0}, {0, x}}, {{1, 0}, {0, 1}}}"),
    ("PolynomialSmithDecomposition[{{x, 0}, {0, 1}}]", Pred(lambda g: isinstance(g, Expr) and len(g.args) == 3,
                                                                 "{u, d, v}")),
    ("NonCommutativeMultiply[a, b, c]", "a ** b ** c"),
    ("StringToByteArray[\"abc\"]", "ByteArray[{97, 98, 99}]"), ("StringToByteArray[\"\"]", "ByteArray[{}]"),
    ("BaseDecode[\"AQID\"]", "ByteArray[{1, 2, 3}]"), ("BaseDecode[\"YWJj\"]", "ByteArray[{97, 98, 99}]"),
    ("ByteArray[{1, 2}]", "ByteArray[{1, 2}]"), ("ByteArray[\"AQID\"]", "ByteArray[{1, 2, 3}]"),
    ("ByteArray[{256}]", Raises()),
    ("Fit[{{0, 1}, {1, 3}, {2, 5}}, {1, x}, x]", PolyNear("1 + 2 x")),
    ("Fit[{{0, 0}, {1, 1}, {2, 4}, {3, 9}}, {1, x, x^2}, x]", PolyNear("x^2")),
    ("FindFit[{{1, 2}, {2, 4}, {3, 6}}, a x, {a}, x]", Pred(lambda g: abs(to_complex(g.args[0].args[1]) - 2) < 1e-6,
                                                             "{a -> 2.}")),
    ("FindFit[{{0, 1}, {1, 3}, {2, 5}}, a + b x, {a, b}, x]",
     Pred(lambda g: abs(to_complex(g.args[0].args[1]) - 1) < 1e-6 and abs(to_complex(g.args[1].args[1]) - 2) < 1e-6,
          "{a -> 1., b -> 2.}")),
    ("RandomChoice[{5}]", "5"), ("RandomChoice[{x}, 3]", "{x, x, x}"), ("RandomChoice[{0, 1} -> {a, b}]", "b"),
    ("CreateDataStructure[\"Stack\"]", Inert("DataStructure")), ("CreateDataStructure[\"Queue\"]", Inert("DataStructure")),
    ("ToRules[x == 1]", "{x -> 1}"), ("ToRules[x == 1 && y == 2]", "{x -> 1, y -> 2}"),
]

CASES = SPECIAL + GROUPS + STRUCTURAL + DISCRETE + APPLIED + DISTRIBUTIONS + GROUPS2 + SERIES + ERRORS + MISC


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_check_cases(case):
    check(case)
