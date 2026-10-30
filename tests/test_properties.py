"""Property-based tests (hypothesis): algebraic laws and list identities."""
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st, HealthCheck  # noqa: E402

import sympy as sp  # noqa: E402

from mikoshilang import Session, parse  # noqa: E402
from mikoshilang.expr import Expr  # noqa: E402
from mikoshilang.formatting import input_form, full_form  # noqa: E402
from mikoshilang.math_ops import to_sympy  # noqa: E402

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
ints = st.integers(min_value=-50, max_value=50)
small_lists = st.lists(ints, min_size=0, max_size=8)
X = sp.Symbol("x")


def ml(xs):
    return "{" + ", ".join(str(v) for v in xs) + "}"


def poly_src(coeffs):
    return " + ".join(f"({c})*x^{i}" for i, c in enumerate(coeffs)) or "0"


def poly_sp(coeffs):
    return sum(c * X ** i for i, c in enumerate(coeffs))


@SETTINGS
@given(ints, ints, st.integers(min_value=1, max_value=30))
def test_rational_arithmetic_exact(p, q, r):
    s = Session()
    assert to_sympy(s.evaluate(f"({p})/({r}) + ({q})/({r + 1})")) == sp.Rational(p, r) + sp.Rational(q, r + 1)


@SETTINGS
@given(st.lists(ints, min_size=1, max_size=5))
def test_derivative_matches_sympy(coeffs):
    s = Session()
    assert sp.expand(to_sympy(s.evaluate(f"D[{poly_src(coeffs)}, x]")) - sp.diff(poly_sp(coeffs), X)) == 0


@SETTINGS
@given(st.lists(ints, min_size=1, max_size=5))
def test_integrate_then_differentiate_roundtrip(coeffs):
    s = Session()
    r = s.evaluate(f"D[Integrate[{poly_src(coeffs)}, x], x]")
    assert sp.expand(to_sympy(r) - poly_sp(coeffs)) == 0


@SETTINGS
@given(st.lists(ints, min_size=1, max_size=4))
def test_expand_factor_equivalent(coeffs):
    s = Session()
    e = poly_src(coeffs)
    assert sp.expand(to_sympy(s.evaluate(f"Factor[Expand[{e}]]")) - poly_sp(coeffs)) == 0


@SETTINGS
@given(st.lists(ints, min_size=1, max_size=5))
def test_inputform_roundtrips(coeffs):
    """Display must be re-parseable to the same value (catches formatting bugs like 1*3^-1*x^3)."""
    s = Session()
    value = s.evaluate(f"Expand[({poly_src(coeffs)})/3]")
    again = Session().evaluate(input_form(value))
    assert sp.expand(to_sympy(again) - to_sympy(value)) == 0
    assert "^-1" not in input_form(value)


@SETTINGS
@given(small_lists)
def test_sort_is_sorted_permutation(xs):
    r = Session().evaluate(f"Sort[{ml(xs)}]")
    assert list(r.args) == sorted(xs)


@SETTINGS
@given(small_lists)
def test_reverse_involution_and_length(xs):
    s = Session()
    assert list(s.evaluate(f"Reverse[Reverse[{ml(xs)}]]").args) == xs
    assert s.evaluate(f"Length[{ml(xs)}]") == len(xs)


@SETTINGS
@given(small_lists, st.integers(min_value=0, max_value=8))
def test_take_drop_partition(xs, k):
    k = min(k, len(xs))
    r = Session().evaluate(f"Join[Take[{ml(xs)}, {k}], Drop[{ml(xs)}, {k}]]")
    assert list(r.args) == xs


@SETTINGS
@given(small_lists)
def test_total_equals_fold_and_python_sum(xs):
    s = Session()
    assert s.evaluate(f"Total[{ml(xs)}]") == sum(xs)
    assert s.evaluate(f"Fold[Plus, 0, {ml(xs)}]") == sum(xs)


@SETTINGS
@given(small_lists)
def test_select_matches_filter(xs):
    s = Session()
    assert list(s.evaluate(f"Select[{ml(xs)}, # > 0 &]").args) == [v for v in xs if v > 0]
    assert list(s.evaluate(f"Cases[{ml(xs)}, _?EvenQ]").args) == [v for v in xs if v % 2 == 0]
    assert s.evaluate(f"Count[{ml(xs)}, u_ /; u < 0]") == sum(1 for v in xs if v < 0)


@SETTINGS
@given(st.lists(ints, min_size=1, max_size=8), st.data())
def test_part_matches_python_indexing(xs, data):
    i = data.draw(st.integers(min_value=-len(xs), max_value=len(xs)).filter(lambda v: v != 0))
    assert Session().evaluate(f"{ml(xs)}[[{i}]]") == xs[i - 1 if i > 0 else i]


@SETTINGS
@given(st.integers(min_value=0, max_value=25))
def test_recursive_definitions_match_closed_forms(k):
    s = Session()
    s.evaluate("fac[0] = 1; fac[m_Integer] := m fac[m - 1]")
    s.evaluate("fib[0] = 0; fib[1] = 1; fib[m_] := fib[m] = fib[m - 1] + fib[m - 2]")
    assert s.evaluate(f"fac[{k}]") == sp.factorial(k)
    assert s.evaluate(f"fib[{k}]") == sp.fibonacci(k)


@SETTINGS
@given(ints, ints, ints, ints, ints, ints)
def test_linear_system_solution_satisfies_equations(a, b, c, d, e, f):
    if a * d - b * c == 0:
        return
    r = Session().evaluate(f"Solve[{{({a}) x + ({b}) y == {e}, ({c}) x + ({d}) y == {f}}}, {{x, y}}]")
    assert len(r.args) == 1
    sol = {str(rule.args[0]): to_sympy(rule.args[1]) for rule in r.args[0].args}
    assert a * sol["x"] + b * sol["y"] == e and c * sol["x"] + d * sol["y"] == f


@SETTINGS
@given(st.integers(min_value=0, max_value=8))
def test_series_order_inclusive(k):
    r = Session().evaluate(f"Normal[Series[Exp[x], {{x, 0, {k}}}]]")
    assert sp.expand(to_sympy(r) - sum(X ** i / sp.factorial(i) for i in range(k + 1))) == 0


@SETTINGS
@given(st.integers(min_value=1, max_value=4))
def test_module_never_leaks(k):
    s = Session()
    assert s.evaluate(f"Module[{{t = {k}}}, t^3]") == k ** 3
    assert str(s.evaluate("t")) == "t"


@SETTINGS
@given(st.lists(ints, min_size=1, max_size=5))
def test_fullform_parses_back(coeffs):
    s = Session()
    value = s.evaluate(f"Expand[{poly_src(coeffs)}]")
    assert sp.expand(to_sympy(Session().evaluate(full_form(value))) - to_sympy(value)) == 0
