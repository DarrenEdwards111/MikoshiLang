import pytest

from _harness import Raises, Pred, check, ids

CASES = [
    ("Simplify[(x^2 - 1)/(x - 1)]", "1 + x"), ("Simplify[Sin[x]^2 + Cos[x]^2]", "1"), ("Simplify[2 x + 3 x]", "5 x"),
    ("Simplify[x > 0, x > 1]", Raises()),
    ("Expand[(x + 1)^2]", "1 + 2 x + x^2"), ("Expand[(a + b) (a - b)]", "a^2 - b^2"), ("Expand[x (y + 1)]", "x + x y"),
    ("Expand[Sin[2 x]]", "Sin[2 x]"),
    ("Factor[x^2 - 1]", "(x - 1) (x + 1)"), ("Factor[x^3 - x^2 + x - 1]", "(x - 1) (1 + x^2)"), ("Factor[6]", "6"),
    ("Factor[2 x^2 + 4 x]", "2 x (2 + x)"),
    ("Collect[x y + x z + y, x]", "x (y + z) + y"), ("Collect[a x^2 + b x^2 + c x, x]", "(a + b) x^2 + c x"),
    ("Collect[x y + x + y^2 y, {x, y}]", "x (1 + y) + y^3"),
    ("Coefficient[3 x^2 + 2 x + 1, x, 2]", "3"), ("Coefficient[3 x^2 + 2 x + 1, x, 0]", "1"),
    ("Coefficient[a x + b x y, x]", "a + b y"), ("Coefficient[x + 2/x, x, -1]", "2"), ("Coefficient[x^2 + 1, x^2]", "1"),
    ("CoefficientList[1 + 2 x + 3 x^3, x]", "{1, 2, 0, 3}"), ("CoefficientList[0, x]", "{}"),
    ("CoefficientList[1 + x y + y^2, {x, y}]", "{{1, 0, 1}, {0, 1, 0}}"),
    ("Degree[x^3 + 1, x]", Raises(NotImplementedError, "Exponent")),
    ("Exponent[x^3 + x, x]", "3"), ("Exponent[5, x]", "0"), ("Exponent[0, x]", "-Infinity"), ("Exponent[x^2 y^5, y]", "5"),
    ("Discriminant[a x^2 + b x + c, x]", "b^2 - 4 a c"), ("Discriminant[x^3 + p x + q, x]", "-4 p^3 - 27 q^2"),
    ("Discriminant[x^2 - 2 x + 1, x]", "0"),
    ("Resultant[x^2 - 1, x - 1, x]", "0"), ("Resultant[x^2 + 1, x - 2, x]", "5"), ("Resultant[x - a, x - b, x]", "a - b"),
    ("PolynomialQuotient[x^3 + 1, x + 1, x]", "x^2 - x + 1"), ("PolynomialQuotient[x^2 + 3, x - 1, x]", "x + 1"),
    ("PolynomialQuotient[x, 2 x, x]", "1/2"),
    ("PolynomialRemainder[x^3 + 1, x + 1, x]", "0"), ("PolynomialRemainder[x^2 + 3, x - 1, x]", "4"),
    ("PolynomialRemainder[x^5, x^2 + 1, x]", "x"),
    ("PolynomialDivision[x^2 + 3, x - 1, x]", "{x + 1, 4}"), ("PolynomialDivision[x^3, x, x]", "{x^2, 0}"),
    ("PolynomialDivision[2 x + 1, 2, x]", "{x + 1/2, 0}"),
    ("PartialFractions[1/(x^2 - 1), x]", "1/(2 (x - 1)) - 1/(2 (x + 1))"),
    ("PartialFractions[(x + 2)/(x (x + 1)), x]", "2/x - 1/(1 + x)"), ("PartialFractions[x^2/(x - 1), x]", "x + 1 + 1/(x - 1)"),
    # ComplexExpand treats variables as real (Wolfram)
    ("ComplexExpand[(x + I y)^2]", "x^2 - y^2 + 2 I x y"), ("ComplexExpand[Exp[I x]]", "Cos[x] + I Sin[x]"),
    ("ComplexExpand[Abs[x + I y]]", "Sqrt[x^2 + y^2]"),
    ("PowerExpand[Sqrt[x^2]]", "x"), ("PowerExpand[Log[x y]]", "Log[x] + Log[y]"), ("PowerExpand[(a b)^n]", "a^n b^n"),
    ("PowerExpand[Log[x^3]]", "3 Log[x]"),
    ("TrigExpand[Sin[2 x]]", "2 Sin[x] Cos[x]"), ("TrigExpand[Cos[x + y]]", "Cos[x] Cos[y] - Sin[x] Sin[y]"),
    ("TrigExpand[Sin[x]]", "Sin[x]"),
    ("TrigReduce[Sin[x]^2]", "1/2 - Cos[2 x]/2"), ("TrigReduce[Sin[x] Cos[x]]", "Sin[2 x]/2"), ("TrigReduce[Cos[x]^2]", "1/2 + Cos[2 x]/2"),
    ("TrigToExp[Cos[x]]", "E^(I x)/2 + E^(-I x)/2"), ("TrigToExp[Sinh[x]]", "E^x/2 - E^(-x)/2"), ("TrigToExp[x]", "x"),
    ("ExpToTrig[E^(I x)]", "Cos[x] + I Sin[x]"), ("ExpToTrig[(E^x + E^-x)/2]", "Cosh[x]"), ("ExpToTrig[1]", "1"),
    ("CompleteTheSquare[x^2 + 6 x + 5, x]", "(x + 3)^2 - 4"), ("CompleteTheSquare[2 x^2 - 4 x, x]", "2 (x - 1)^2 - 2"),
    ("CompleteTheSquare[x^3, x]", Raises()),
    ("QuadraticFormula[1, -3, 2]", "{2, 1}"), ("QuadraticFormula[1, 0, 1]", "{I, -I}"), ("QuadraticFormula[0, 1, 1]", Raises()),
    ("QuadraticFormula[1., 0, -2]", [2 ** 0.5, -(2 ** 0.5)]),
    ("RationalizeDenominator[1/(1 + Sqrt[2])]", "-1 + Sqrt[2]"), ("RationalizeDenominator[1/Sqrt[3]]", "Sqrt[3]/3"),
    ("RationalizeDenominator[2/(Sqrt[3] - 1)]", "1 + Sqrt[3]"),
    ("Refine[Sqrt[x^2], x > 0]", Raises(NotImplementedError)), ("Assumptions[x]", Raises(NotImplementedError)),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_algebra(case):
    check(case)
