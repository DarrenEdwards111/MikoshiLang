"""Bounded direct-Python math audit; independent polynomial/numeric oracles."""
import math
import cmath
import subprocess
import sys
from pathlib import Path
from fractions import Fraction
import pytest
import sympy as sp
from mikoshilang import (Expr, Symbol, Session, simplify, expand, factor, diff,
                        integrate, solve, limit, series, N, numerical,
                        Pi, E, I, Infinity, GoldenRatio)

COVERED_EXPORTS = {"simplify", "expand", "factor", "diff", "integrate", "solve",
                   "limit", "series", "N", "numerical", "Pi", "E", "I",
                   "Infinity", "GoldenRatio"}


@pytest.mark.parametrize("operation,expected", [
    (lambda x: diff(x ** 2, x), Expr("Times", 2, Symbol("x"))),
    (lambda x: integrate(x, x, 0, 2), 2),
    (lambda x: solve(x - 3, x), Expr("List", 3)),
    (lambda x: limit(x, x, 0), 0),
    (lambda x: series(x ** 3, x, 0, 2), 0),
])
def test_calculus_preserves_sympy_variable_assumptions(operation, expected):
    real_x = sp.Symbol("x", real=True)
    assert operation(real_x) == expected


@pytest.mark.parametrize("precision", [0, -1, True, 1.5, "30"])
def test_machine_precision_symbol_validates_requested_precision(precision):
    with pytest.raises(ValueError, match="Precision must be a positive integer"):
        numerical(Symbol("MachinePrecision"), precision)


@pytest.mark.parametrize("precision", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_precision_rejected_consistently(precision):
    with pytest.raises(ValueError, match="Precision must be a positive integer"):
        N(Pi, precision)


x = Symbol("x")


def _value(expr, value=0):
    """Small independent exact arithmetic evaluator, not a SymPy comparison."""
    if isinstance(expr, (int, float, complex)):
        return expr
    if expr == x:
        return value
    if expr == I:
        return 1j
    if isinstance(expr, Expr):
        args = [_value(a, value) for a in expr.args]
        if expr.head == "Plus":
            return sum(args)
        if expr.head == "Times":
            result = 1
            for arg in args:
                result *= arg
            return result
        if expr.head == "Power":
            a, b = args
            if isinstance(a, int) and isinstance(b, int) and b < 0:
                return Fraction(a) ** b
            return a ** b
    raise AssertionError("Unexpected oracle input: %r" % (expr,))


def test_simplify_exact_trig_complex_and_branch_safety():
    assert simplify(Expr("Sin", x) ** 2 + Expr("Cos", x) ** 2) == 1
    assert simplify(I ** 2) == -1
    assert simplify(Expr("Rational", 2, 6)) == Expr("Times", 1, Expr("Power", 3, -1))
    # No positivity assumptions: sqrt(x^2) must not silently become x.
    assert _value(simplify((x ** 2) ** Expr("Rational", 1, 2)), -3) == 3
    huge = 10 ** 400 + 7
    assert simplify(huge) == huge


def test_expand_and_factor_independent_polynomial_values():
    expanded = expand((x + 2) ** 3)
    factored = factor(x ** 3 + 6 * x ** 2 + 12 * x + 8)
    for value in [-5, -2, 0, Fraction(1, 3), 7, 2 + 3j]:
        assert _value(expanded, value) == pytest.approx((value + 2) ** 3)
        assert _value(factored, value) == pytest.approx((value + 2) ** 3)
    assert expand(0) == factor(0) == 0
    assert factor(1) == expand(1) == 1


def test_diff_orders_mixed_variables_and_complex():
    assert diff(x ** 4, x, 0) == x ** 4
    assert diff(x ** 4, "x", 4) == 24
    assert diff(x ** 4, x, 5) == 0
    assert diff(Symbol("y"), x) == 0
    assert _value(diff(I * x ** 2, x), 3) == 6j
    with pytest.raises(ValueError):
        diff(x, x, -1)


def test_integrate_exact_orientation_improper_and_complex():
    assert _value(integrate(x ** 2, x, 0, 3)) == 9
    assert _value(integrate(x ** 2, x, 3, 0)) == -9
    assert integrate(x, "x", 2, 2) == 0
    assert _value(integrate(x, x), 4) == 8
    assert integrate(Expr("Exp", -x), x, 0, Infinity) == 1
    assert _value(integrate(I * x, x, 0, 2)) == 2j


def test_solve_linear_repeated_complex_and_no_roots():
    assert solve(2 * x - 6, "x") == Expr("List", 3)
    assert solve((x - 2) ** 2, x) == Expr("List", 2)
    roots = solve(x ** 2 + 1, x)
    assert roots.head == "List"
    assert {_value(r) for r in roots.args} == {-1j, 1j}
    assert solve(1, x) == Expr("List")
    assert solve(Expr("Equal", x ** 2, 4), x) == Expr("List", -2, 2)


def test_limit_known_values_and_documented_right_hand_default():
    assert limit(Expr("Sin", x) / x, x, 0) == 1
    assert limit((1 - Expr("Cos", x)) / x ** 2, "x", 0) == Expr("Times", 1, Expr("Power", 2, -1))
    assert limit(1 / x, x, Infinity) == 0
    # Direct helper inherits SymPy's right-hand default; not a two-sided claim.
    assert limit(1 / x, x, 0) == Infinity
    assert _value(limit(I * x, x, 2)) == 2j


def test_series_inclusive_order_nonzero_center_and_laurent():
    assert series(Expr("Exp", x), x, 0, 0) == 1
    cubic = series(Expr("Exp", x), x, 0, 3)
    for value in [0, 1, -2, Fraction(1, 2)]:
        assert _value(cubic, value) == 1 + value + Fraction(1, 2) * value ** 2 + Fraction(1, 6) * value ** 3
    centered = series(x ** 3, "x", 2, 2)
    for value in [0, 2, 3]:
        assert _value(centered, value) == 8 + 12 * (value - 2) + 6 * (value - 2) ** 2
    laurent = series(1 / x, x, 0, 2)
    for value in [-2, 1, 3]:
        assert _value(laurent, value) == Fraction(1, value)


@pytest.mark.parametrize("constant,name,expected", [
    (Pi, "Pi", math.pi), (E, "E", math.e), (I, "I", 1j),
    (Infinity, "Infinity", math.inf),
    (GoldenRatio, "GoldenRatio", (1 + math.sqrt(5)) / 2),
])
def test_constants_direct_symbolic_and_numeric(constant, name, expected):
    assert isinstance(constant, Symbol) and constant.name == name
    assert simplify(constant) == constant
    assert N(constant) == pytest.approx(expected)


@pytest.mark.parametrize("evaluator", [N, numerical])
def test_numeric_exact_complex_tiny_and_symbolic(evaluator):
    assert evaluator(Expr("Rational", 1, 8)) == 0.125
    assert evaluator(2 + 3j) == 2 + 3j
    assert evaluator(Expr("Log", -1)) == pytest.approx(cmath.log(-1))
    assert evaluator(Expr("Power", 10, -100)) == pytest.approx(1e-100, rel=1e-14, abs=0)
    assert evaluator(x) == x
    assert evaluator(Expr("List", Pi, Expr("List", I, 0))) == Expr("List", evaluator(Pi), Expr("List", 1j, 0.0))


@pytest.mark.parametrize("evaluator", [N, numerical])
def test_numeric_arbitrary_precision_independent_decimal_reference(evaluator):
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 65
        expected = Decimal(2).sqrt()
        actual = evaluator(Expr("Power", 2, Expr("Rational", 1, 2)), 60)
        assert isinstance(actual, sp.Float)
        assert abs(Decimal(str(actual)) - expected) < Decimal("1e-59")
        third = evaluator(Expr("Rational", 1, 3), 60)
        assert abs(Decimal(str(third)) - Decimal(1) / 3) < Decimal("1e-59")
    assert isinstance(evaluator(0.1, 60), float)  # never invent precision
    assert evaluator(Pi, 15.0) == evaluator(Pi)
    assert evaluator(Pi, Symbol("MachinePrecision")) == evaluator(Pi)


@pytest.mark.parametrize("evaluator", [N, numerical])
@pytest.mark.parametrize("precision", [0, -1, True, 1.5, "20", None])
def test_numeric_invalid_precision(evaluator, precision):
    with pytest.raises(ValueError, match="Precision must be a positive integer"):
        evaluator(Pi, precision)


@pytest.mark.parametrize("operation", [simplify, expand, factor, N, numerical])
def test_math_rejects_unparsed_source_strings(operation):
    with pytest.raises(TypeError):
        operation("x + 1")


def test_session_parity_only_for_equal_contracts():
    session = Session()
    assert simplify(Expr("Sin", x) ** 2 + Expr("Cos", x) ** 2) == session.evaluate("Simplify[Sin[x]^2+Cos[x]^2]")
    assert diff(x ** 3, x) == session.evaluate("D[x^3,x]")
    assert integrate(x ** 2, x, 0, 3) == session.evaluate("Integrate[x^2,{x,0,3}]")
    assert N(Pi) == pytest.approx(float(session.evaluate("N[Pi]")))
    # Solve uses plain roots here, language Solve uses rules.
    assert solve(x - 2, x) == Expr("List", 2)
    assert session.evaluate("Solve[x-2==0,x]") != solve(x - 2, x)
    # Python series deliberately removes O terms; Session Series retains them.
    assert series(Expr("Exp", x), x, 0, 2) != session.evaluate("Series[Exp[x],{x,0,2}]")
    assert series(Expr("Exp", x), x, 0, 2) == session.evaluate("Normal[Series[Exp[x],{x,0,2}]]")


def test_bounded_symbolic_work_in_killable_subprocess():
    result = subprocess.run([sys.executable, "-c", "from mikoshilang import *; x=Symbol('x'); assert solve(x**4-1,x).head=='List'; assert integrate(Expr('Exp',-x),x,0,Infinity)==1; assert series(Expr('Exp',x),x,0,8) is not None"],
                            cwd=str(Path(__file__).resolve().parents[1]),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_numeric_precision_boundaries_complex_and_nonfinite_values():
    assert isinstance(N(Pi, 1), float)
    # Low decimal precision controls binary accuracy, not decimal rounding.
    assert N(Pi, 1) == pytest.approx(math.pi, rel=0.05)
    assert N(0, 60) == 0
    assert numerical(Infinity, 60) == sp.oo
    assert numerical(-Infinity) == -math.inf
    assert cmath.isnan(N(Symbol("Indeterminate")))
    exact_complex = N(Expr("Rational", 1, 3) + I * Expr("Rational", 2, 7), 60)
    from decimal import Decimal, localcontext
    with localcontext() as context:
        context.prec = 65
        real, imag = exact_complex.as_real_imag()
        assert abs(Decimal(str(real)) - Decimal(1) / 3) < Decimal("1e-59")
        assert abs(Decimal(str(imag)) - Decimal(2) / 7) < Decimal("1e-59")
    known_30_digits = sp.Float("1.23456789012345678901234567890", 30)
    limited = numerical(known_30_digits, 80)
    assert limited._prec <= known_30_digits._prec
    assert N(Symbol("MachinePrecision"), 15.0) == pytest.approx(53 * math.log10(2))
    assert float(N(Symbol("MachinePrecision"), 60)) == pytest.approx(53 * math.log10(2))


def test_constants_exact_independent_identities():
    assert simplify(Expr("Exp", I * Pi) + 1) == 0
    assert simplify(Expr("Log", E)) == 1
    assert simplify(GoldenRatio ** 2 - GoldenRatio - 1) == 0
    assert N(Pi, 60) > 3
    assert str(N(Pi, 60)).startswith("3.14159265358979323846264338327950288419716939937510")


def test_property_expand_factor_preserve_independent_values():
    # Skip this test only, not module collection, in the minimal install.
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings, strategies as st

    @settings(max_examples=30, deadline=None, derandomize=True)
    @given(st.integers(-8, 8), st.integers(-8, 8), st.integers(-5, 5))
    def check(a, b, value):
        original = (x + a) * (x + b)
        assert _value(expand(original), value) == (value + a) * (value + b)
        assert _value(factor(expand(original)), value) == (value + a) * (value + b)
    check()


def test_property_polynomial_calculus_known_coefficients():
    pytest.importorskip("hypothesis")
    from hypothesis import given, settings, strategies as st

    @settings(max_examples=30, deadline=None, derandomize=True)
    @given(st.integers(0, 7), st.integers(-10, 10), st.integers(-3, 3))
    def check(degree, coefficient, value):
        polynomial = coefficient * x ** degree
        expected = coefficient * degree * value ** (degree - 1) if degree else 0
        assert _value(diff(polynomial, x), value) == expected
        expected_integral = Fraction(coefficient, degree + 1) * (2 ** (degree + 1))
        assert _value(integrate(polynomial, x, 0, 2)) == expected_integral
    check()
