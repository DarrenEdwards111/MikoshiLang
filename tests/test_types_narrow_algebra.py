"""Hand-narrowed types of the algebra and calculus modules (algebra, calculus, wl_algebra, wl_calculus,
wl_calculus2, wl_polyalg): variables, orders, lists and matrices."""
import pytest

from mikoshilang import Session, types as T
from mikoshilang.library.core import REGISTRY


def _report(src):
    """Mismatches of one evaluation in report mode; the function's own error (if any) still surfaces after."""
    s = Session(check_types="report")
    try:
        s.evaluate(src)
    except (ValueError, TypeError, ArithmeticError, NotImplementedError):
        pass
    return [(m["function"], m["expected"]) for m in s.type_mismatches]


def _clean(src):
    s = Session(check_types="report")
    s.evaluate(src)
    return s.type_mismatches


def _types(name):
    """Parameter and result type texts of every implementation of a library head."""
    return [([T.type_text(p) for p in sig.params] + ([T.type_text(sig.rest) + "..."] if sig.rest else []),
             T.type_text(sig.result)) for sig in REGISTRY[name].signature]


@pytest.mark.parametrize("name, params, result", [
    ("Discriminant", ["Expr", "Var"], "Expr"),
    ("PolynomialDivision", ["Expr", "Expr", "Var"], "ListOf[Expr]"),
    ("Collect", ["Expr", "Var | ListOf[Var]"], "Expr"),
    ("NthDerivative", ["Expr", "Var", "NonNegInteger"], "Expr"),
    ("Gradient", ["Expr", "ListOf[Var]"], "ListOf[Expr]"),
    ("ExactODEQ", ["Expr", "Expr", "Var", "Var"], "Boolean"),
])
def test_narrowed_signatures(name, params, result):
    assert (params, result) in _types(name)


@pytest.mark.parametrize("src, head, expected", [
    ("Discriminant[x^2 + 1, 2]", "Discriminant", "Var"),
    ('Coefficient[x^2 + x, x, "2"]', "Coefficient", "Integer"),
    ("Collect[x y + x, {x, 1}]", "Collect", "Var | ListOf[Var]"),
    ("NthDerivative[x^3, x, -1]", "NthDerivative", "NonNegInteger"),
    ("Gradient[x y, 3]", "Gradient", "ListOf[Var]"),
    ("Divergence[5, {x}]", "Divergence", "ListOf[Expr]"),
    ("LinearProgramming[{1, 1}, {1, 2}, {3}]", "LinearProgramming", "MatrixOf[Expr]"),
    ("Roots[x^2 == 1, 2]", "Roots", "Var"),
    ("Casoratian[{1, 2^n}, 3]", "Casoratian", "Var"),
    ('MatrixSymbol[1, {2, 2}]', "MatrixSymbol", "String | Var"),
    ("Subresultants[x^2 - 1, x - 1, 1]", "Subresultants", "Var"),
])
def test_narrowed_mismatches(src, head, expected):
    assert (head, expected) in _report(src)


@pytest.mark.parametrize("src", [
    "Discriminant[a x^2 + b x + c, x]", "Coefficient[3 x^2 + 2 x + 1, x, 2]", "Collect[x y + x z + y, {x, y}]",
    "PolynomialDivision[x^2 + 3, x - 1, x]", "QuadraticFormula[a, b, c]", "NthDerivative[Sin[x], x, n]",
    "Gradient[x^2 y, {x, y}]", "JacobianMatrix[{x y, x + y}, {x, y}]", "TaylorSeries[Exp[x], {x, 0, 3}]",
    "ExactODEQ[2 x y, x^2, x, y]", "Exponent[x^3 + x, x]", "GroebnerBasis[{x^2 - y, y^2 - 1}, {x, y}]",
    "Casoratian[{1, 2^n}, n]", 'MatrixSymbol["A", {2, 3}]', "VectorSymbol[v, n]",
])
def test_symbolic_and_valid_calls_are_clean(src):
    assert _clean(src) == []


def test_narrowing_does_not_change_unchecked_results():
    s = Session()
    assert s.evaluate("PolynomialDivision[x^2 + 3, x - 1, x] == {1 + x, 4}") is True
    assert s.evaluate("NthDerivative[x^5, x, 3] == 60 x^2") is True
    assert s.evaluate("Gradient[x^2 y, {x, y}] == {2 x y, x^2}") is True


def test_raise_mode_names_the_head():
    s = Session(check_types=True)
    with pytest.raises(TypeError, match="^Discriminant: argument 2 2 is not of type Var"):
        s.evaluate("Discriminant[x^2 + 1, 2]")
