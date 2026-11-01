"""Direct differential cases, part 2: heads that had fewer than three cases of their own, so the stated
verification harness (scripts/verify_library.py) can run their ``Check: reference`` sentence.

Sources of the expected values (never the implementation's own output):

* Wolfram Language reference pages for kernel arithmetic and predicates (Plus, Times, Power, Positive, Negative,
  NonNegative, NumberQ, OddQ, Equal, List, StringJoin, Apart, Cancel, Together), operator forms (GreaterThan,
  LessThan, EqualTo, SameAs, Key, ...), inert wrappers that stay unevaluated (OutputForm, Skeleton, Defer,
  HoldCompleteForm, MixedRadix, UpTo, Missing, combinators, C[k], AiryBiZero[k] with an exact index) and the
  ``{}`` returned by DownValues/SubValues/DefaultValues for a symbol with no definitions.
* The Heun equations as written on the HeunG/HeunC pages: substituting z = 0 gives HeunG'(0) = q/(a gamma) and
  HeunC'(0) = -q/gamma. Mathieu functions with q = 0: MathieuC[a, 0, z] = Cos[Sqrt[a] z], so its derivative is
  -Sqrt[a] Sin[Sqrt[a] z] (and MathieuSPrime is Sqrt[a] Cos[Sqrt[a] z]).
* mpmath ``ellipfun`` for the inverse Jacobi functions: root of the forward function on (0, K(m)).
* Hand-computed Smith normal form (gcd of the entries, then |det|/gcd) and polar decomposition of a diagonal
  matrix; optimisation optima found by hand.
"""
import mpmath as mp
import pytest

from _harness import Pred, check, ev, ids, same, to_complex
from mikoshilang.expr import Expr


def Near(want, rtol=1e-10, atol=0.0):
    want = complex(want)
    return Pred(lambda g: abs(to_complex(g) - want) <= rtol * abs(want) + atol, f"~{want}")


def _inv(kind, v, m):
    f = lambda u: mp.ellipfun(kind, u, m=m) - v
    a, b = mp.mpf("1e-9"), mp.ellipk(m) - mp.mpf("1e-9")
    for _ in range(200):                                  # bisection: f changes sign on (a, b)
        c = (a + b) / 2
        if (f(a) < 0) == (f(c) < 0):
            a = c
        else:
            b = c
    return (a + b) / 2


def Part(i, want):
    """the i-th element (1-based) of a list result equals want"""
    return Pred(lambda g: isinstance(g, Expr) and len(g.args) >= i and same(g.args[i - 1], ev(want)), f"[[{i}]] == {want}")


def NumMatrix(i, rows, tol=1e-9):
    def ok(g):
        try:
            m = g.args[i - 1]
            return all(abs(to_complex(m.args[r].args[c]) - rows[r][c]) <= tol
                       for r in range(len(rows)) for c in range(len(rows[0])))
        except Exception:
            return False
    return Pred(ok, f"[[{i}]] ~ {rows}")


KERNEL = [
    ("Apart[1/(x (x + 1)), x]", "1/x - 1/(1 + x)"), ("Apart[1/(x^2 - 1), x]", "1/(2 (x - 1)) - 1/(2 (x + 1))"),
    ("Cancel[(x^2 - 1)/(x - 1)]", "1 + x"), ("Cancel[x/x^2]", "1/x"),
    ("Negative[-2]", True), ("Negative[3]", False), ("Negative[0]", False),
    ("NonNegative[0]", True), ("NonNegative[-1]", False), ("NonNegative[5/2]", True),
    ("Positive[2]", True), ("Positive[-1/2]", False), ("Positive[0]", False),
    ("NumberQ[1.5]", True), ("NumberQ[x]", False), ("NumberQ[Pi]", False),
    ("OddQ[3]", True), ("OddQ[4]", False), ("OddQ[x]", False),
    ("Plus[1, 2]", "3"), ("Plus[x, x]", "2 x"), ("Plus[]", "0"),
    ("Power[2, 10]", "1024"), ("Power[x, 0]", "1"), ("Power[4, 1/2]", "2"),
    ("Times[2, 3]", "6"), ("Times[x, x]", "x^2"), ("Times[]", "1"),
    ("StringJoin[\"a\", \"b\"]", '"ab"'), ("StringJoin[{\"a\", \"b\"}, \"c\"]", '"abc"'), ("StringJoin[]", '""'),
    ("Together[1/x + 1/y]", "(x + y)/(x y)"), ("Together[1/(x + 1) + 1]", "(2 + x)/(1 + x)"),
    ("List[1, 2]", "{1, 2}"), ("List[]", "{}"),
    ("Equal[1, 1]", True), ("Equal[1, 2]", False), ("Equal[x, x]", True),
]

OPERATORS = [
    ("GreaterThan[3][5]", True), ("GreaterThan[3][1]", False),
    ("LessThan[3][1]", True), ("LessThan[3][5]", False),
    ("GreaterEqualThan[3][3]", True), ("GreaterEqualThan[3][2]", False),
    ("LessEqualThan[3][4]", False), ("LessEqualThan[3][3]", True),
    ("EqualTo[2][2]", True), ("EqualTo[2][3]", False),
    ("UnequalTo[2][3]", True), ("UnequalTo[2][2]", False),
    ("SameAs[x][x]", True), ("SameAs[x][y]", False),
    ("Key[\"a\"][<|\"a\" -> 1|>]", "1"), ("Key[b][<|b -> 2|>]", "2"),
    ("Missing[\"NotFound\"]", 'Missing["NotFound"]'), ("Missing[]", "Missing[]"),
    ("Splice[{1, 2}]", "Splice[{1, 2}]"), ("Splice[{}]", "Splice[{}]"),
    ("Span[1, 3]", "1 ;; 3"),
    ("UpTo[3]", "UpTo[3]"), ("UpTo[0]", "UpTo[0]"),
]

INERT = [
    ("CombinatorS[a]", "CombinatorS[a]"), ("CombinatorK[a]", "CombinatorK[a]"), ("CombinatorI[f][g]", "CombinatorI[f][g]"),
    ("CombinatorB[a]", "CombinatorB[a]"), ("CombinatorC[a]", "CombinatorC[a]"), ("CombinatorW[a]", "CombinatorW[a]"),
    ("CombinatorY[g][h]", "CombinatorY[g][h]"),
    ("GeneralizedPower[g, y, 2]", "GeneralizedPower[g, y, 2]"),
    ("ChampernowneNumber[10]", "ChampernowneNumber[10]"), ("ChampernowneNumber[2]", "ChampernowneNumber[2]"),
    ("MeijerGReduce[g[y], y]", "MeijerGReduce[g[y], y]"), ("FoxHReduce[g[y], y]", "FoxHReduce[g[y], y]"),
    ("OutputForm[y]", "OutputForm[y]"), ("OutputForm[{1, 2}]", "OutputForm[{1, 2}]"),
    ("StandardForm[y]", "StandardForm[y]"), ("StandardForm[{1, 2}]", "StandardForm[{1, 2}]"),
    ("TraditionalForm[y]", "TraditionalForm[y]"), ("TraditionalForm[{1, 2}]", "TraditionalForm[{1, 2}]"),
    ("TeXForm[y]", "TeXForm[y]"), ("TeXForm[{1, 2}]", "TeXForm[{1, 2}]"),
    ("Skeleton[3]", "Skeleton[3]"), ("Skeleton[12]", "Skeleton[12]"),
    ("StringSkeleton[3]", "StringSkeleton[3]"), ("StringSkeleton[12]", "StringSkeleton[12]"),
    ("DisableFormatting[y]", "y"),
    ("Defer[1 + 1]", "Defer[1 + 1]"), ("HoldCompleteForm[1 + 1]", "HoldCompleteForm[1 + 1]"),
    ("MixedRadix[{24, 60}]", "MixedRadix[{24, 60}]"), ("MixedRadix[{10}]", "MixedRadix[{10}]"),
    ("C[1]", "C[1]"), ("C[2]", "C[2]"),
    ("AiryBiZero[1]", "AiryBiZero[1]"), ("AiryBiZero[3]", "AiryBiZero[3]"),
    ("Wedge[a, b]", "Wedge[a, b]"), ("Wedge[a, b, c]", "Wedge[a, b, c]"),
    ("NextValue[y]", "NextValue[y]"),
    ("MatrixForm[{{1}}]", "MatrixForm[{{1}}]"), ("MatrixForm[{{1, 2}, {3, 4}}]", "MatrixForm[{{1, 2}, {3, 4}}]"),
    ("SetAccuracy[y, 10]", "y"),
    ("DownValues[cbk2Undefined]", "{}"), ("SubValues[cbk2Undefined]", "{}"), ("DefaultValues[cbk2Undefined]", "{}"),
    ("DownValues[cbk2Undefined2]", "{}"), ("SubValues[cbk2Undefined2]", "{}"),
    ("Messages[cbk2Sym]", "{}"), ("Messages[cbk2Sym2]", "{}"),
    ("MessageName[cbk2f, \"usage\"]", 'MessageName[cbk2f, "usage"]'),
    ("MessageName[cbk2g, \"argx\"]", 'MessageName[cbk2g, "argx"]'),
]

NUMERIC = [
    ("HeunGPrime[2., 1., 1., 1., 1., 1., 0.]", Near(0.5)),          # q/(a gamma)
    ("HeunGPrime[3., 1.5, 0.5, 1., 2., 1., 0.]", Near(0.25)),       # 1.5/(3*2)
    ("HeunCPrime[1., 0.5, 2., 0.3, 0.2, 0.]", Near(-0.5)),          # -q/gamma
    ("HeunCPrime[0.6, 0.5, 3., 0.3, 0.2, 0.]", Near(-0.2)),
    ("MathieuCPrime[4., 0, 0.3]", Near(-2 * mp.sin(0.6))),
    ("MathieuSPrime[4., 0, 0.3]", Near(2 * mp.cos(0.6))),
    ("InverseJacobiDC[1.3, 0.3]", Near(_inv("dc", 1.3, 0.3))), ("InverseJacobiDC[1.1, 0.6]", Near(_inv("dc", 1.1, 0.6))),
    ("InverseJacobiDS[2., 0.3]", Near(_inv("ds", 2, 0.3))), ("InverseJacobiDS[1.5, 0.6]", Near(_inv("ds", 1.5, 0.6))),
    ("InverseJacobiND[1.05, 0.3]", Near(_inv("nd", 1.05, 0.3))),
    ("SmithDecomposition[{{2, 4}, {6, 8}}]", Part(2, "{{2, 0}, {0, 4}}")),
    ("PolarDecomposition[{{2., 0.}, {0., 3.}}]", NumMatrix(2, [[2, 0], [0, 3]])),
    ("ConvexOptimization[x^2, {x >= 1}, {x}, \"PrimalMinimumValue\"]", Near(1, 1e-6)),
    ("GeometricOptimization[x, {x >= 2}, {x}, \"PrimalMinimumValue\"]", Near(2, 1e-6)),
    ("SymmetrizedReplacePart[{{a, b}, {c, d}}, {{1, 2} -> y}, Symmetric[{1, 2}]]", "{{a, y}, {y, d}}"),
]

CASES = KERNEL + OPERATORS + INERT + NUMERIC


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_check_cases2(case):
    check(case)
