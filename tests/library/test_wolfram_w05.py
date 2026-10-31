"""Differential tests for Wolfram batch W05 (polynomials, equation manipulation, Piecewise, cylindrical
decomposition, function properties, non-commutative algebra and the related option names).

Expected values are Wolfram Language results (reference.wolfram.com/language/ref/<Name>.html examples, or values
derived from the definitions in their Details sections), or are checked independently here: decompositions are
multiplied back, sums of squares re-expanded, reductions re-assembled, solutions substituted back, and sample points
tested against the original conditions with SymPy.  String expectations compare structurally or by a SymPy-verified
zero difference (see _harness.py).
"""
import pytest
import sympy as sp

from _harness import Pred, Raises, check, ev, ids, to_complex
from mikoshilang.expr import Expr, Symbol
from mikoshilang.math_ops import to_sympy

x, y, z, a, b, c = sp.symbols("x y z a b c")


def S(g):
    return to_sympy(g)


def Equals(expr):
    """The result is mathematically equal to the SymPy expression expr."""
    return Pred(lambda g: sp.simplify(S(g) - expr) == 0, f"== {expr}")


def EqualsAnd(expr, test, what):
    return Pred(lambda g: sp.simplify(S(g) - expr) == 0 and test(g), f"== {expr} and {what}")


def is_list(g, n=None):
    return isinstance(g, Expr) and g.head == "List" and (n is None or len(g.args) == n)


def has_head(g, h):
    if isinstance(g, Expr):
        if g.head == h:
            return True
        return any(has_head(a_, h) for a_ in g.args) or (isinstance(g.head, Expr) and has_head(g.head, h))
    return False


def Solves(cond, names):
    """A list of rule lists, each satisfying cond (a SymPy boolean / relation) exactly."""
    def ok(g):
        if not is_list(g) or not g.args:
            return False
        for sol in g.args:
            vals = {sp.Symbol(r.args[0].name): S(r.args[1]) for r in sol.args}
            if set(map(str, vals)) != set(names):
                return False
            v = cond.subs(vals)
            if v is not sp.true and not (isinstance(v, sp.Basic) and sp.simplify(v) is sp.true):
                return False
        return True
    return Pred(ok, f"instances of {cond}")


def ProductIs(expr):
    """A TrigFactorList-style list {{c, 1}, {f1, e1}, ...} whose product equals expr."""
    def ok(g):
        if not is_list(g):
            return False
        prod = sp.Integer(1)
        for p in g.args:
            prod *= S(p.args[0]) ** S(p.args[1])
        return sp.simplify(sp.expand_trig(prod - expr)) == 0
    return Pred(ok, f"product == {expr}")


def SumOfSquares(expr, vs):
    def ok(g):
        return is_list(g) and sp.expand(sum(S(q) ** 2 for q in g.args) - expr) == 0
    return Pred(ok, f"sum of squares == {expr}")


def Floats(vals, tol=1e-12):
    def ok(g):
        flat = []

        def walk(h):
            if is_list(h):
                for t in h.args:
                    walk(t)
            else:
                flat.append(to_complex(h))
        walk(g)
        return len(flat) == len(vals) and all(abs(u - v) < tol for u, v in zip(flat, vals))
    return Pred(ok, f"~{vals}")


def Contains(intervals_value):
    """An isolating interval {lo, hi} with rational endpoints containing the value."""
    def ok(g):
        lo, hi = S(g.args[0]), S(g.args[1])
        return lo.is_Rational and hi.is_Rational and lo <= intervals_value <= hi
    return Pred(ok, f"interval containing {intervals_value}")


# ── predicates and system variables ──
PREDICATES = [
    ("PolynomialExpressionQ[x^2 + 3 x + 1, x]", True),
    ("PolynomialExpressionQ[x Sin[x]^2 + x Cos[x]^2, x]", False),
    ("PolynomialExpressionQ[x - 1/x, x]", False),
    ("PolynomialExpressionQ[(x + y)^3, {x, y}]", True),
    ("RationalExpressionQ[(x^2 + 1)/(x - 1), x]", True),
    ("RationalExpressionQ[Sqrt[x], x]", False),
    ("RationalExpressionQ[1/(x + y) + y^-2, {x, y}]", True),
    ("$Assumptions", True),
    ("Assuming[x > 0, $Assumptions]", "x > 0"),
    ("Block[{$Assumptions = x > 0}, Simplify[Sqrt[x^2]]]", "x"),
    ("$MaxPiecewiseCases", "100"),
    ("IntegerQ[$MaxPiecewiseCases]", True),
    ("Block[{$MaxPiecewiseCases = 5}, $MaxPiecewiseCases]", "5"),
    ("$MaxRootDegree", "1024"),
    ("IntegerQ[$MaxRootDegree]", True),
    ("$MaxRootDegree > 100", True),
    ("Undefined", Pred(lambda g: isinstance(g, Symbol) and g.name == "Undefined", "Undefined")),
    ("ConditionalExpression[x, False]", Pred(lambda g: isinstance(g, Symbol) and g.name == "Undefined",
                                             "Undefined")),
    ("Piecewise[{{1, x > 0}}, Undefined] /. x -> -1", Pred(lambda g: g == Symbol("Undefined"), "Undefined")),
    ("Head[Undefined]", Pred(lambda g: isinstance(g, Symbol) and g.name == "Symbol", "Symbol")),
]

# ── equation manipulation ──
SIDES = [
    ("AddSides[a == b, c]", "a + c == b + c"),
    ("AddSides[x < 3, 2]", "x + 2 < 5"),
    ("AddSides[a == b, c == d]", "a + c == b + d"),
    ("AddSides[1 < x < 2, -1]", "0 < x - 1 < 1"),
    ("ApplySides[f, a < b]", "f[a] < f[b]"),
    ("ApplySides[#^2 &, a == b]", "a^2 == b^2"),
    ("ApplySides[Sqrt, x^2 == 4]", "Sqrt[x^2] == 2"),
    ("SubtractSides[a == b]", "a - b == 0"),
    ("SubtractSides[x + 3 == 5, 3]", "x == 2"),
    ("SubtractSides[a < b, c]", "a - c < b - c"),
    ("MultiplySides[x/2 == 3, 2]", "x == 6"),
    ("MultiplySides[x < 1, -2]", "-2 < -2 x"),
    ("MultiplySides[a > b, 3]", "3 a > 3 b"),
    ("DivideSides[2 x == 6, 2]", "x == 3"),
    ("DivideSides[a < b, -1]", "-b < -a"),
    ("DivideSides[a == b]", "Piecewise[{{a/b == 1, b != 0}}, a == b]"),
]

# ── polynomial algebra ──
POLY = [
    ("ApartSquareFree[(x^2 - 2)/(x^4 - 2 x^2 + 1)]", "-1/(-1 + x^2)^2 + 1/(-1 + x^2)"),
    ("ApartSquareFree[1/(x (x + 1)^2), x]", EqualsAnd(1 / (x * (x + 1) ** 2), lambda g: g.head == "Plus", "a sum")),
    ("ApartSquareFree[(x^3 + 1)/(x^2 - 1), x]", "x + 1/(x - 1)"),
    ("CoefficientArrays[{x + 2 y + 3, 4 x - y}, {x, y}]", "{{3, 0}, {{1, 2}, {4, -1}}}"),
    ("CoefficientArrays[x^2 + x y + 1, {x, y}]", "{1, {0, 0}, {{1, 1}, {0, 0}}}"),
    ("CoefficientArrays[{a x == 5}, {x}]", "{{-5}, {{a}}}"),
    ("FromCoefficientRules[{{2, 0} -> 1, {0, 1} -> 3}, {x, y}]", "x^2 + 3 y"),
    ("FromCoefficientRules[CoefficientRules[x^2 y + 5 x, {x, y}], {x, y}]", "x^2 y + 5 x"),
    ("FromCoefficientRules[{{3} -> 2, {0} -> -1}, x]", "2 x^3 - 1"),
    ("PolynomialExtendedGCD[x^2 - 1, x^2 - 3 x + 2, x]", "{x - 1, {1/3, -1/3}}"),
    ("PolynomialExtendedGCD[x^3 + 1, x^2 + 1, x]", "{1, {1/2 + x/2, 1/2 - x/2 - x^2/2}}"),
    ("PolynomialExtendedGCD[x^2 + 1, x + 1, x, Modulus -> 2]", "{1 + x, {0, 1}}"),
    ("PolynomialReduction[x^2 + y^2, {x - y}, {x, y}]", "2 y^2"),
    ("PolynomialReduction[x^3, {x^2 + 1}, {x}]", "-x"),
    ("PolynomialReduction[x y - 1, {x - 1, y - 2}, {x, y}]", "1"),
    ("SubresultantPolynomialRemainders[x^3 + 1, x^2 + 2, x]", "{1 + x^3, 2 + x^2, 1 - 2 x, 9}"),
    ("Last[SubresultantPolynomialRemainders[x^2 - 1, x^2 - 3 x + 2, x]]",
     Pred(lambda g: sp.rem(S(g), x - 1, x) == 0 and sp.degree(S(g), x) == 1, "a multiple of the GCD x - 1")),
    ("First[SubresultantPolynomialRemainders[x^4 + x, x^2 + 1, x]]", "x^4 + x"),
    ("SubresultantPolynomials[x^2 - 1, x - 1, x]", "{0, x - 1}"),
    ("SubresultantPolynomials[x^2 + 1, x + 2, x]", "{5, x + 2}"),
    ("First[SubresultantPolynomials[x^3 - 2, x^2 + x + 1, x]]", "Resultant[x^3 - 2, x^2 + x + 1, x]"),
    ("Subresultants[x^2 - 1, x - 1, x]", "{0, 1}"),
    ("Subresultants[(x - 1)(x - 2)(x - 3), (x - 1)(x - 2)(x - 4), x]", "{0, 0, -1, 1}"),
    ("Subresultants[x^2 + 1, x + 2, x]", "{5, 1}"),
    ("PadeApproximant[Exp[x], {x, 0, {1, 1}}]", "(1 + x/2)/(1 - x/2)"),
    ("PadeApproximant[Exp[x], {x, 0, 2}]", "(1 + x/2 + x^2/12)/(1 - x/2 + x^2/12)"),
    ("PadeApproximant[Cos[x], {x, 0, 1}]", "1"),
    ("PadeApproximant[Log[1 + x], {x, 0, {1, 1}}]", "x/(1 + x/2)"),
    ("PartialFractionElements[1/(x^2 - 1), x, \"Poles\"]", "{-1, 1}"),
    ("PartialFractionElements[(x^3 + 1)/(x^2 - 4), x, \"PolynomialPart\"]", "x"),
    ("PartialFractionElements[1/(x (x - 1)^2), x, \"PoleMultiplicities\"]", "{1, 2}"),
    ("PartialFractionElements[1/(x^2 - 1), x, \"Decomposition\"]", Equals(1 / (x ** 2 - 1))),
    ("PolynomialSumOfSquaresList[x^2 + 2 x + 1, {x}]", "{1 + x}"),
    ("PolynomialSumOfSquaresList[x^2 + x y + y^2, {x, y}]", "{x/2 + y, Sqrt[3] x/2}"),
    ("PolynomialSumOfSquaresList[x^4 + 1, {x}]", SumOfSquares(x ** 4 + 1, [x])),
    ("PolynomialSumOfSquaresList[x^2 - 1, {x}]", Raises()),
    ("MatrixPolynomialValue[x^2 + 1, {{1, 2}, {0, 1}}, x]", "{{2, 4}, {0, 2}}"),
    ("MatrixPolynomialValue[{3, 2, -1}, {{3, -1}, {4, 1}}]",
     "{3, 2, -1} . {IdentityMatrix[2], {{3, -1}, {4, 1}}, MatrixPower[{{3, -1}, {4, 1}}, 2]}"),
    ("MatrixPolynomialValue[CharacteristicPolynomial[{{0, -1, 1}, {1, 2, -1}, {1, 1, 0}}, t], "
     "{{0, -1, 1}, {1, 2, -1}, {1, 1, 0}}, t]", "{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}"),
    ("MatrixPolynomialValue[a x^2 + b x + c, IdentityMatrix[2], x]",
     "{{a + b + c, 0}, {0, a + b + c}}"),
    ("TrigFactorList[Sin[x]^2 - Cos[x]^2]", ProductIs(sp.sin(x) ** 2 - sp.cos(x) ** 2)),
    ("TrigFactorList[Sin[2 x]]", ProductIs(sp.sin(2 * x))),
    ("TrigFactorList[Tan[x] Cos[x]]", ProductIs(sp.sin(x))),
]

# ── integer / polynomial matrix normal forms ──
NORMAL_FORMS = [
    ("HermiteDecomposition[{{2, 4}, {1, 3}}]", "{{{1, -1}, {-1, 2}}, {{1, 1}, {0, 2}}}"),
    ("Module[{u, r}, {u, r} = HermiteDecomposition[{{4, 6, 2}, {6, 9, 1}}]; {Abs[Det[u]], u . {{4, 6, 2}, {6, 9, 1}} == r}]",
     "{1, True}"),
    ("HermiteDecomposition[{{1/2, 1}, {1, 2}}]", Raises()),
    ("HermiteReduce[{{2, 4}, {1, 3}}]", "{{1, 1}, {0, 2}}"),
    ("HermiteReduce[{{3, 1}, {0, 2}}]", "{{3, 1}, {0, 2}}"),
    ("HermiteReduce[{{4, 6}, {6, 9}}]", "{{2, 3}, {0, 0}}"),
    ("PolynomialHermiteReduce[{{x, 1}, {0, x}}]", "{{x, 1}, {0, x}}"),
    ("PolynomialHermiteReduce[{{x, 0}, {x^2, 1}}]", "{{x, 0}, {0, 1}}"),
    ("PolynomialHermiteReduce[{{x - 1, x}, {x, x + 1}}]",
     Pred(lambda g: sp.simplify(sp.Matrix(S(g)).det()) != 0 and S(g.args[1].args[0]) == 0, "upper triangular")),
    ("PolynomialHermiteDecomposition[{{x, 1}, {0, x}}]", "{{{1, 0}, {0, 1}}, {{x, 1}, {0, x}}}"),
    ("Module[{u, h, m = {{x, 0}, {x^2, 1}}}, {u, h} = PolynomialHermiteDecomposition[m]; "
     "{Expand[u . m - h], FreeQ[Det[u], x]}]", "{{{0, 0}, {0, 0}}, True}"),
    ("Module[{u, h, m = {{x + 1, x}, {x, x - 1}}}, {u, h} = PolynomialHermiteDecomposition[m]; "
     "{Expand[u . m - h], h[[2, 1]]}]", "{{{0, 0}, {0, 0}}, 0}"),
    ("PolynomialSmithReduce[{{x, 1}, {0, x}}]", "{{1, 0}, {0, x^2}}"),
    ("PolynomialSmithReduce[{{x - 1, 0}, {0, x - 2}}]", "{{1, 0}, {0, x^2 - 3 x + 2}}"),
    ("PolynomialSmithReduce[{{x, 0}, {0, x^2}}]", "{{x, 0}, {0, x^2}}"),
    ("PolynomialSmithDecomposition[{{x, 0}, {0, x^2}}]", "{{{1, 0}, {0, 1}}, {{x, 0}, {0, x^2}}, {{1, 0}, {0, 1}}}"),
    ("Module[{u, s, v, m = {{x, 1}, {0, x}}}, {u, s, v} = PolynomialSmithDecomposition[m]; Expand[u . m . v - s]]",
     "{{0, 0}, {0, 0}}"),
    ("Module[{u, s, v, m = {{x - 1, 0}, {0, x - 2}}}, {u, s, v} = PolynomialSmithDecomposition[m]; "
     "{FreeQ[Det[u], x], FreeQ[Det[v], x]}]", "{True, True}"),
]

# ── Piecewise, DirectedInfinity ──
PIECEWISE = [
    ("Piecewise[{{1, True}}]", "1"),
    ("Piecewise[{{a, False}, {b, True}}]", "b"),
    ("Piecewise[{{x, x > 0}}, -x] /. x -> -3", "3"),
    ("Piecewise[{}, 7]", "7"),
    ("Piecewise[{{1, x > 0}}] /. x -> -1", "0"),
    ("Head[Piecewise[{{x^2, x < 0}, {x, x > 0}}]]", Pred(lambda g: g == Symbol("Piecewise"), "Piecewise")),
    ("PiecewiseExpand[Max[x, 0]]", "Piecewise[{{0, x <= 0}}, x]"),
    ("PiecewiseExpand[Abs[x], Reals]", "Piecewise[{{-x, x < 0}}, x]"),
    ("PiecewiseExpand[UnitStep[x] x]", "Piecewise[{{x, x >= 0}}]"),
    ("PiecewiseExpand[Min[x, 1] /. x -> 3]", "1"),
    ("DirectedInfinity[1]", "Infinity"),
    ("DirectedInfinity[-2]", "-Infinity"),
    ("DirectedInfinity[]", "ComplexInfinity"),
    ("DirectedInfinity[1 + I]", Pred(lambda g: g.head == "DirectedInfinity"
                                     and abs(to_complex(g.args[0]) - (1 + 1j) / 2 ** 0.5) < 1e-12, "unit direction")),
]

# ── solving ──
SOLVING = [
    ("FindInstance[x^2 == 4 && x > 0, x]", "{{x -> 2}}"),
    ("FindInstance[x + y == 3, {x, y}, Integers]", Solves(sp.Eq(x + y, 3), ["x", "y"])),
    ("FindInstance[x^2 + y^2 < 1, {x, y}]", Solves(x ** 2 + y ** 2 < 1, ["x", "y"])),
    ("FindInstance[x^2 == -1, x, Reals]", "{}"),
    ("SolveAlways[a x + b == 0, x]", "{{a -> 0, b -> 0}}"),
    ("SolveAlways[(a - 1) x^2 + b x + c - 2 == 0, x]", "{{a -> 1, b -> 0, c -> 2}}"),
    ("SolveAlways[a x + b == x, x]", "{{a -> 1, b -> 0}}"),
    ("NSolveValues[x^2 == 2, x]", Floats([-2 ** 0.5, 2 ** 0.5])),
    ("NSolveValues[{x + y == 1, x - y == 0}, {x, y}]", Floats([0.5, 0.5])),
    ("NSolveValues[x^3 == 1, x, Reals]", Floats([1.0])),
    ("NSolveValues[x^3 == 1, x, MaxRoots -> 1]", Floats([complex(-0.5, -3 ** 0.5 / 2)])),
    ("NSolveValues[x^4 == 16, x, MaxRoots -> 2]", Floats([-2, -2j])),
    ("MaxRoots[1]", Raises()),
    ("RootIntervals[x^2 - 2]", Pred(lambda g: [S(i.args[0]) <= v <= S(i.args[1]) for i, v in
                                               zip(g.args[0].args, (-sp.sqrt(2), sp.sqrt(2)))] == [True, True],
                                    "isolating intervals of -Sqrt[2], Sqrt[2]")),
    ("RootIntervals[{x^2 - 1, x - 1}]", "{{{-1, -1}, {1, 1}}, {{1}, {1, 2}}}"),
    ("RootIntervals[x^3 - x]", "{{{-1, -1}, {0, 0}, {1, 1}}, {{1}, {1}, {1}}}"),
    ("IsolatingInterval[Sqrt[2]]", Contains(sp.sqrt(2))),
    ("IsolatingInterval[3/4]", "{3/4, 3/4}"),
    ("IsolatingInterval[Root[#^3 - 2 &, 1], 1/100]",
     Pred(lambda g: S(g.args[0]) <= sp.cbrt(2) <= S(g.args[1]) and S(g.args[1]) - S(g.args[0]) < sp.Rational(1, 100),
          "width < 1/100 around 2^(1/3)")),
]

# ── cylindrical decomposition ──
CAD = [
    ("CylindricalDecomposition[x^2 + y^2 < 1, {x, y}]", "Less[-1, x, 1] && Less[-Sqrt[1 - x^2], y, Sqrt[1 - x^2]]"),
    ("CylindricalDecomposition[x^2 < 4, x]", "Less[-2, x, 2]"),
    ("CylindricalDecomposition[x^2 - 3 x + 2 <= 0, x]", "LessEqual[1, x, 2]"),
    ("CylindricalDecomposition[x^2 + y^2 <= 1 && y >= 0, {x, y}]", "LessEqual[-1, x, 1] && LessEqual[0, y, Sqrt[1 - x^2]]"),
    ("GenericCylindricalDecomposition[x^2 + y^2 < 1, {x, y}]",
     "{Less[-1, x, 1] && Less[-Sqrt[1 - x^2], y, Sqrt[1 - x^2]], False}"),
    ("GenericCylindricalDecomposition[x^2 + y^2 <= 1, {x, y}]",
     "{Less[-1, x, 1] && Less[-Sqrt[1 - x^2], y, Sqrt[1 - x^2]], -1 + x^2 + y^2 == 0}"),
    ("GenericCylindricalDecomposition[y > x^2, {x, y}]", "{y > x^2, False}"),
    ("CylindricalDecompositionFunction[x^2 + y^2 < 1, {x, y}][0, 0]", True),
    ("CylindricalDecomposition[x^2 + y^2 < 1, {x, y}, \"Function\"][1, 1]", False),
    ("CylindricalDecompositionFunction[x^2 < 4, {x}][1]", True),
    ("CylindricalDecompositionFunction[x^2 < 4, {x}][3]", False),
    ("Normal[CylindricalDecomposition[x^2 < 4, x, \"Function\"]]", "Less[-2, x, 2]"),
    ("Head[CylindricalDecomposition[x^2 < 4, x, \"Function\"]]",
     Pred(lambda g: g == Symbol("CylindricalDecompositionFunction"), "CylindricalDecompositionFunction")),
    ("SemialgebraicComponentInstances[x^2 > 1, x]", Pred(lambda g: len(g.args) == 2 and all(
        S(s.args[0].args[1]) ** 2 > 1 for s in g.args), "one point in each of x < -1, x > 1")),
    ("SemialgebraicComponentInstances[x^2 + y^2 < 1, {x, y}]", Solves(x ** 2 + y ** 2 < 1, ["x", "y"])),
    ("SemialgebraicComponentInstances[(x^2 - 1)(x^2 - 4) < 0, x]", Pred(lambda g: len(g.args) == 2 and all(
        1 < S(s.args[0].args[1]) ** 2 < 4 for s in g.args), "one point in each component")),
]

# ── function properties ──
PROPERTIES = [
    ("FunctionContinuous[Sin[x], x]", True),
    ("FunctionContinuous[1/x, x]", False),
    ("FunctionContinuous[{1/x, x > 0}, x]", True),
    ("FunctionContinuous[Tan[x], x]", False),
    ("FunctionAnalytic[Exp[x], x]", True),
    ("FunctionAnalytic[Abs[x], x]", False),
    ("FunctionAnalytic[1/x, x, Complexes]", False),
    ("FunctionAnalytic[x^2 + 1, x]", True),
    ("FunctionMeromorphic[1/x, x]", True),
    ("FunctionMeromorphic[Tan[x], x]", True),
    ("FunctionMeromorphic[Log[x], x]", False),
    ("FunctionMeromorphic[Gamma[x], x]", True),
    ("FunctionSign[x^2 + 1, x]", "1"),
    ("FunctionSign[x^2, x]", "1"),
    ("FunctionSign[x^2, x, StrictInequalities -> True]", "Indeterminate"),
    ("FunctionSign[Sin[x], x]", "Indeterminate"),
    ("FunctionSign[-Exp[x], x]", "-1"),
    ("FunctionMonotonicity[Exp[x], x]", "1"),
    ("FunctionMonotonicity[x^2, x]", "Indeterminate"),
    ("FunctionMonotonicity[{x^2, x > 0}, x]", "1"),
    ("FunctionMonotonicity[-x^3, x, StrictInequalities -> True]", "-1"),
    ("FunctionConvexity[x^2, x]", "1"),
    ("FunctionConvexity[Abs[x], x]", "1"),
    ("FunctionConvexity[Abs[x], x, StrictInequalities -> True]", "Indeterminate"),
    ("FunctionConvexity[Log[x], x]", "Indeterminate"),
    ("FunctionConvexity[{Log[x], x > 0}, x]", "-1"),
    ("FunctionConvexity[{Abs[x], x > 0}, x]", "0"),
    ("FunctionConvexity[{Sin[x], 0 <= x <= Pi}, x, StrictInequalities -> True]", "-1"),
    ("FunctionConvexity[x - E^x, x]", "-1"),
    ("FunctionConvexity[x^2 + y^2, {x, y}]", "1"),
    ("FunctionConvexity[x^2 - y^2, {x, y}]", "Indeterminate"),
    ("FunctionInjective[x^3, x]", True),
    ("FunctionInjective[x^2, x]", False),
    ("FunctionInjective[{x^2, x > 0}, x]", True),
    ("FunctionInjective[Exp[x], x, Complexes]", False),
    ("FunctionSurjective[x^3, x]", True),
    ("FunctionSurjective[x^2, x]", False),
    ("FunctionSurjective[Exp[x], x]", False),
    ("FunctionSurjective[Sin[z], z, Complexes]", True),
    ("FunctionBijective[x^3, x]", True),
    ("FunctionBijective[Exp[x], x]", False),
    ("FunctionBijective[2 x + 1, x]", True),
    ("FunctionDiscontinuities[Tan[x], x]", "Cos[x] == 0"),
    ("FunctionDiscontinuities[1/x, x]", "x == 0"),
    ("FunctionDiscontinuities[Sqrt[x], x]", "x <= 0"),
    ("FunctionDiscontinuities[Floor[x], x]", "Sin[Pi x] == 0"),
    ("FunctionSingularities[1/x, x]", "x == 0"),
    ("FunctionSingularities[Log[x], x]", "x <= 0"),
    ("FunctionSingularities[1/(x^2 - 1), x]", "x^2 == 1"),
    ("FunctionPoles[1/x, x]", "{{0, 1}}"),
    ("FunctionPoles[1/(x^2 (x - 1)), x]", "{{0, 2}, {1, 1}}"),
    ("FunctionPoles[x^2, x]", "{}"),
    ("Length[FunctionPoles[Tan[x], x]]", "2"),
]

# ── Boolean functions ──
BOOLEAN = [
    ("BooleanConsecutiveFunction[2, 3][a, b, c]", "(a && b) || (b && c)"),
    ("BooleanConsecutiveFunction[2, 3][True, True, False]", True),
    ("BooleanConsecutiveFunction[2, 3][True, False, True]", False),
    ("BooleanConsecutiveFunction[{2, True}, 3][True, False, True]", True),
    ("BooleanMinterms[{1, 3, 5}, {a, b, c}]", "(!a && c) || (!b && c)"),
    ("BooleanMinterms[{0}, {a, b}]", "!a && !b"),
    ("BooleanMinterms[{3}, {a, b}]", "a && b"),
    ("BooleanMaxterms[{1, 3, 5}, {a, b, c}]", "(!a || c) && (!b || c)"),
    ("BooleanMaxterms[{5}, {a, b, c}]", "a || !b || c"),
    ("BooleanMaxterms[{0}, {a, b}]", "!a || !b"),
    ("BooleanMaxterms[{3}, {a, b}]", "a || b"),
    ("UnateQ[a && b]", True),
    ("UnateQ[Xor[a, b]]", False),
    ("UnateQ[a && !b, {a, !b}]", True),
    ("UnateQ[a || b]", True),
]

# ── non-commutative algebra ──
NONCOMM = [
    ("NonCommutativeMultiply[a, b] === a ** b", True),
    ("a ** (b ** c) === NonCommutativeMultiply[a, b, c]", True),
    ("NonCommutativeExpand[(a + b) ** (a - b)]", "a ** a - a ** b + b ** a - b ** b"),
    ("NonCommutativeExpand[(a + b) ** c]", "a ** c + b ** c"),
    ("NonCommutativeExpand[a ** (2 b)]", "2 a ** b"),
    ("NonCommutativeCollect[a ** b + a ** c, {a}]", "a ** (b + c)"),
    ("NonCommutativeCollect[b ** a + c ** a, {a}]", "(b + c) ** a"),
    ("NonCommutativeCollect[a ** b + 2 a ** c, {a}]", "a ** (b + 2 c)"),
    ("NonCommutativeVariables[a ** b + c ** a]", "{a, b, c}"),
    ("NonCommutativeVariables[x ** y]", "{x, y}"),
    ("NonCommutativeVariables[2 ** x]", Pred(lambda g: is_list(g), "a list")),
    ("NonCommutativePolynomialQ[a ** b + 2 c, {a, b, c}]", True),
    ("NonCommutativePolynomialQ[Sin[a ** b], {a, b}]", False),
    ("NonCommutativePolynomialQ[a ** b + 2 c, {a, b}]", False),
    ("Module[{l = NonCommutativeMonomialList[a ** b + 3 b ** a + 1]}, {Length[l], MemberQ[l, 3 b ** a], MemberQ[l, a ** b], MemberQ[l, 1]}]", "{3, True, True, True}"),
    ("Length[NonCommutativeMonomialList[x ** y ** x + y]]", "2"),
    ("NonCommutativeMonomialList[2 x ** y]", "{2 x ** y}"),
    ("NonCommutativeGroebnerBasis[{a ** b - b ** a}, {a, b}]", "{b ** a - a ** b}"),
    ("NonCommutativeGroebnerBasis[{y**y - 2y**x**y, y**x - 2 x**x}, {x, y}]",
     "{x**x**x**x - x**x**x/2, y**x - 2 x**x, y**y - 4 x**x**y}"),
    ("NonCommutativePolynomialReduce[y**y - 2y**x**y, NonCommutativeGroebnerBasis[{y**y - 2y**x**y, "
     "y**x - 2 x**x}, {x, y}], {x, y}][[2]]", "0"),
    ("NonCommutativePolynomialReduce[b ** a, {a ** b - b ** a}, {a, b}]", "{{-# &}, a ** b}"),
    ("Module[{p = x**x**y**x**y + 2 y**x**y**x + 3 x**y, qq = {2 x**y**x + 3 x**x + 4 y + 5, "
     "3 y**x**x + 4 y**x - 5 x + 6}, fs, r}, {fs, r} = NonCommutativePolynomialReduce[p, qq, {x, y}]; "
     "NonCommutativeExpand[p - Plus @@ MapThread[Construct, {fs, qq}] - r]]", "0"),
    ("NonCommutativePolynomialReduction[b ** a, {a ** b - b ** a}, {a, b}]", "a ** b"),
    ("NonCommutativePolynomialReduction[z**y**x**y**z + 2x**y**z, {z**y - 3x, y**z + 5x**x}, {x, y, z}]",
     "-10 x**x**x - 15 x**x**x**x"),
    ("NonCommutativePolynomialReduction[x ** y, {y ** x - 1}, {x, y}]", "x ** y"),
    ("Head[NonCommutativeAlgebra[]]", Pred(lambda g: g == Symbol("NonCommutativeAlgebra"), "NonCommutativeAlgebra")),
    ("NonCommutativeExpand[(x + y) ** (x - y), NonCommutativeAlgebra[<|\"CommutativeVariables\" -> {x, y}|>]]",
     "x ** x - y ** y"),
    ("NonCommutativeExpand[(a + c) ** b, NonCommutativeAlgebra[<|\"ScalarVariables\" -> {c}|>]]",
     "c b + a ** b"),
    ("NonCommutativeExpand[e1 ** e1, CliffordAlgebra[{{e1, e2}}]]", "1"),
    ("NonCommutativeExpand[e2 ** e1, CliffordAlgebra[{{e1, e2}}]]", "-e1 ** e2"),
    ("NonCommutativeExpand[(x + 2y + 3z)**(4x + 5y + 6z), CliffordAlgebra[{{x}, {y}, {z}}]]",
     "-6 - 3 x ** y - 6 x ** z - 3 y ** z"),
    ("NonCommutativeExpand[(i - j) ** (i - j), CliffordAlgebra[{{}, {i, j}}]]", "-2"),
    ("CliffordAlgebra[{{e1, e2}, {1, 1}}]", Raises()),
    ("NonCommutativeExpand[Wedge[a, a], GrassmannAlgebra[{a, b}]]", "0"),
    ("NonCommutativeExpand[Wedge[b, a], GrassmannAlgebra[{a, b}]]", "-Wedge[a, b]"),
    ("NonCommutativeExpand[Wedge[a + b, a + b], GrassmannAlgebra[{a, b}]]", "0"),
    ("NonCommutativeExpand[b ** a, GrassmannAlgebra[{a, b}, <|\"Multiplication\" -> NonCommutativeMultiply|>]]",
     "-a ** b"),
    ("NonCommutativeExpand[dx ** x, WeylAlgebra[{{x}, {dx}}]]", "1 + x ** dx"),
    ("NonCommutativeExpand[dx ** x ** x, WeylAlgebra[{{x}, {dx}}]]", "2 x + x ** x ** dx"),
    ("NonCommutativeExpand[x ** dx, WeylAlgebra[{{x}, {dx}}]]", "x ** dx"),
]

# ── symbolic arrays, proofs ──
MISC = [
    ("Expand[(ComponentExpand[Det[a], Element[a, Matrices[{2, 2}]]] /. Indexed[a, {i_, j_}] :> m[i, j]) - "
     "(m[1, 1] m[2, 2] - m[1, 2] m[2, 1])]", "0"),
    ("Expand[(ComponentExpand[v . v, Element[v, Vectors[3]]] /. Indexed[v, {i_}] :> w[i]) - "
     "(w[1]^2 + w[2]^2 + w[3]^2)]", "0"),
    ("ComponentExpand[VectorLess[{{x, y}, {1, 2}}]]", "x < 1 && y < 2"),
    ("Length[ComponentExpand[VectorSymbol[v, {3}]]]", "3"),
    ("ComponentExpand[MatrixSymbol[m, x]]", Raises()),
    ("FindEquationalProof[f[a] == a, {ForAll[x, f[x] == x]}][\"Theorem\"]", "f[a] == a"),
    ("FindEquationalProof[g[a] == g[b], {a == b}][\"Logic\"]", Pred(lambda g: g == "EquationalLogic",
                                                                    "\"EquationalLogic\"")),
    ("Head[FindEquationalProof[a == c, {a == b, b == c}]]", Pred(lambda g: g == Symbol("ProofObject"),
                                                                 "ProofObject")),
    ("FindEquationalProof[a == c, {a == b}]", Raises()),
    ("FindEquationalProof[a == c, {a == b, b == c}][\"Axioms\"]", "{a == b, b == c}"),
    ("ProofObject[<|\"Theorem\" -> p == q|>][\"Theorem\"]", "p == q"),
    ("ProofObject[<|\"Theorem\" -> p == q|>][\"Theorems\"]", "{p == q}"),
    ("ProofObject[<|\"Theorem\" -> p == q|>][\"ProofGraph\"]", Raises()),
]

# ── option names and the functions that honour them ──
OPTIONS = [
    ("Solve[x^3 + x + 1 == 0, x, Cubics -> False]",
     "{{x -> Root[1 + # + #^3 &, 1]}, {x -> Root[1 + # + #^3 &, 2]}, {x -> Root[1 + # + #^3 &, 3]}}"),
    ("Roots[x^3 + x + 1 == 0, x, Cubics -> False]",
     "x == Root[1 + # + #^3 &, 1] || x == Root[1 + # + #^3 &, 2] || x == Root[1 + # + #^3 &, 3]"),
    ("FreeQ[Solve[x^3 + x + 1 == 0, x, Cubics -> True], Root]", True),
    ("Cubics[1]", Raises()),
    ("Solve[x^4 + x + 1 == 0, x, Quartics -> False]",
     "{{x -> Root[1 + # + #^4 &, 1]}, {x -> Root[1 + # + #^4 &, 2]}, {x -> Root[1 + # + #^4 &, 3]}, "
     "{x -> Root[1 + # + #^4 &, 4]}}"),
    ("FreeQ[Solve[x^4 + x + 1 == 0, x, Quartics -> True], Root]", True),
    ("Quartics[1]", Raises()),
    ("Solve[Sqrt[x] == 2, x, VerifySolutions -> True]", "{{x -> 4}}"),
    ("Solve[x^2 == 4, x, VerifySolutions -> False]", "{{x -> -2}, {x -> 2}}"),
    ("VerifySolutions[1]", Raises()),
    ("Integrate[Exp[-a x], {x, 0, Infinity}, GenerateConditions -> True]",
     "ConditionalExpression[1/a, Re[a] > 0]"),
    ("Integrate[Exp[-a x], {x, 0, Infinity}, GenerateConditions -> False]", "1/a"),
    ("Sum[r^k, {k, 0, Infinity}, GenerateConditions -> True]", "ConditionalExpression[1/(1 - r), Abs[r] < 1]"),
    ("FunctionMonotonicity[x^3, x, StrictInequalities -> True]", "1"),
    ("FunctionSign[{x^2, x > 0}, x, StrictInequalities -> True]", "1"),
    ("StrictInequalities[1]", Raises()),
    ("FunctionSpace", Pred(lambda g: g == Symbol("FunctionSpace"), "inert symbol")),
    ("(FunctionSpace -> \"Polynomial\")[[1]]", Pred(lambda g: g == Symbol("FunctionSpace"), "FunctionSpace")),
    ("FunctionSpace[\"Polynomial\"]", Raises()),
    ("BooleanStrings", Pred(lambda g: g == Symbol("BooleanStrings"), "inert symbol")),
    ("(BooleanStrings -> {\"yes\", \"no\"})[[2]]", "{\"yes\", \"no\"}"),
    ("BooleanStrings[True]", Raises()),
    ("Solve[a x == 1, x, MaxExtraConditions -> 0]", "{{x -> 1/a}}"),
    ("Solve[a x == 1, x, MaxExtraConditions -> All]", Raises()),
    ("MaxExtraConditions[0]", Raises()),
    ("Simplify[(1 + x)^3, ComplexityFunction -> (-LeafCount[#] &)]", "1 + 3 x + 3 x^2 + x^3"),
    ("Simplify[x^2 + 2 x + 1, ComplexityFunction -> LeafCount]", "(1 + x)^2"),
    ("Simplify[Sin[x]^2 + Cos[x]^2, ComplexityFunction -> (Count[{#}, Sin, Infinity] &)]", "1"),
    ("Simplify[g[x] + g[x]^2, TransformationFunctions -> {Automatic, (# /. g[u_] :> u &)}]", "x (1 + x)"),
    ("Simplify[g[x], TransformationFunctions -> {(# /. g[u_] :> u^2 &)}]", "g[x]"),
    ("Simplify[g[x]^2 + 2 g[x] + 1, TransformationFunctions -> {Automatic}]", "(1 + g[x])^2"),
    ("Simplify[(x - 1)(x + 1)(x^2 + 1)(x - 2)^10, ExcludedForms -> {(_Plus)^_}]", "(-2 + x)^10 (-1 + x^4)"),
    ("FullSimplify[h[Sin[x]^2 + Cos[x]^2] + Sin[x]^2 + Cos[x]^2, ExcludedForms -> {h[_]}]",
     "1 + h[Sin[x]^2 + Cos[x]^2]"),
    ("FullSimplify[h[Sin[x]^2 + Cos[x]^2] + Sin[x]^2 + Cos[x]^2]", "1 + h[1]"),
    ("ExcludedForms[1]", Raises()),
    ("Simplify[4 Sin[x]^2 Cos[x]^2 + 4 Sin[x] Cos[x] + 1, Trig -> False]", "(1 + 2 Cos[x] Sin[x])^2"),
    ("Simplify[Sin[x]^2 + Cos[x]^2, Trig -> False]", "Sin[x]^2 + Cos[x]^2"),
    ("Expand[Sin[x + y], Trig -> True]", "Cos[y] Sin[x] + Cos[x] Sin[y]"),
    ("Expand[4 (1 + Sin[x + y])^2 Cos[x - y], Trig -> True]",
     EqualsAnd(4 * (1 + sp.sin(x + y)) ** 2 * sp.cos(x - y),
               lambda g: not has_head(g, "Plus") or "x + y" not in str(g), "multiple angles expanded")),
    ("Factor[Sin[2 x], Trig -> True]", "2 Cos[x] Sin[x]"),
    ("Factor[Sin[x]^2 - Cos[x]^2, Trig -> True]", "-(Cos[x] - Sin[x]) (Cos[x] + Sin[x])"),
    ("Factor[x^2 - 1, Trig -> False]", "(x - 1) (x + 1)"),
    ("RowReduce[{{1, 2}, {1, 2 + 10^-20}}, ZeroTest -> (Abs[#] < 10^-10 &)]", "{{1, 2}, {0, 0}}"),
    ("RowReduce[{{1, 2}, {1, 2 + 10^-20}}]", "{{1, 0}, {0, 1}}"),
    ("MatrixRank[{{1, 2}, {1, 2 + 10^-20}}, ZeroTest -> (Abs[#] < 10^-10 &)]", "1"),
    ("NullSpace[{{1, 1, 1}}, ZeroTest -> (# == 0 &)]", "NullSpace[{{1, 1, 1}}]"),
    ("RowReduce[{{2, 4, 6}, {1, 2, 4}}, ZeroTest -> (# == 0 &)]", "RowReduce[{{2, 4, 6}, {1, 2, 4}}]"),
    ("ZeroTest[0]", Raises()),
    ("DSolve[y[x] == x y'[x] + y'[x]^2, y[x], x, IncludeSingularSolutions -> True][[1]]", "{y[x] -> -x^2/4}"),
    ("DSolve[y'[x]^2 == 4 y[x], y[x], x, IncludeSingularSolutions -> True][[1]]", "{y[x] -> 0}"),
    ("DSolve[y'[x] == y[x], y[x], x, IncludeSingularSolutions -> True]", "DSolve[y'[x] == y[x], y[x], x]"),
    ("Length[DSolve[y'[x]^2 == 4 y[x], y[x], x, IncludeSingularSolutions -> False]]", "1"),
]

CASES = (PREDICATES + SIDES + POLY + NORMAL_FORMS + PIECEWISE + SOLVING + CAD + PROPERTIES + BOOLEAN + NONCOMM + MISC
         + OPTIONS)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w05(case):
    check(case)
