"""Prototype tests for the Python interop pillar (PYTHON_INTEROP.md)."""
import fractions
import math

import numpy as np
import pytest
import sympy as sp
from hypothesis import given, settings, strategies as st

from mikoshilang import Session, Expr, Symbol
from mikoshilang.interop import from_python, to_python, mtype


# ── conversion table: round trips ──

def _data():
    scalars = (st.integers(-10**30, 10**30) | st.floats(allow_nan=False) | st.text(max_size=8) | st.booleans()
               | st.none() | st.fractions(max_denominator=10**6).filter(lambda f: f.denominator != 1)
               | st.complex_numbers(allow_nan=False, allow_infinity=False).filter(lambda c: c.imag != 0))
    keys = st.text(max_size=5) | st.integers(-100, 100)
    return st.recursive(scalars, lambda c: st.lists(c, max_size=4) | st.dictionaries(keys, c, max_size=4),
                        max_leaves=12)


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b and math.copysign(1, a) == math.copysign(1, b) or (a == b == 0.0)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


@settings(max_examples=300, deadline=None)
@given(_data())
def test_python_roundtrip(v):
    back = to_python(from_python(v))
    assert _same(back, v), (v, back)


@settings(max_examples=200, deadline=None)
@given(_data())
def test_roundtrip_survives_evaluation(v):
    """Converted data is already in evaluated (normal) form: evaluating it does not change the Python value."""
    s = Session()     # wrapped in Identity: a bare Python str passed to eval would be parsed as source code
    assert _same(s.eval(Expr("Identity", from_python(v))), v)


@pytest.mark.parametrize("value, expected", [
    (np.int64(2), 2), (np.float32(0.5), 0.5), (np.bool_(True), True),
    (np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]), (np.array([1.5, 2.5]), [1.5, 2.5]),
    ((1, 2), [1, 2]), (sp.Rational(1, 3), fractions.Fraction(1, 3)), (sp.Matrix([[1, 2]]), [[1, 2]]),
    (float("inf"), math.inf), (float("-inf"), -math.inf),
])
def test_one_way_and_special_rows(value, expected):
    assert _same(to_python(from_python(value)), expected)


def test_bignum_precision_kept():
    s = Session()
    v = s.eval("N[Pi, 40]")
    assert isinstance(v, sp.Float) and v._prec > 100
    assert s.eval(Expr("Plus", from_python(v), 0)) == v


def test_symbolic_modes():
    s = Session()
    r = s.eval("Integrate[x^2, x]")
    assert isinstance(r, Expr) and str(r) == "x^3/3"
    assert s.eval("Integrate[x^2, x]", symbolic="sympy") == sp.Symbol("x") ** 3 / 3
    with pytest.raises(TypeError):
        s.eval("Integrate[x^2, x]", symbolic="error")
    assert str(s.eval("Sqrt[2]")) == "Sqrt[2]"            # exact algebraic numbers stay symbolic


def test_numpy_out():
    s = Session()
    a = s.eval("{{1., 2.}, {3., 4.}} . {{1., 0.}, {0., 1.}}", arrays="numpy")
    assert isinstance(a, np.ndarray) and a.dtype == float and a.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert isinstance(s.eval("{1, x}", arrays="numpy"), list)    # not numeric: stays a list


def test_pandas_rows():
    pd = pytest.importorskip("pandas")
    assert to_python(from_python(pd.Series([1, 2]))) == [1, 2]
    assert to_python(from_python(pd.Series([1, 2], index=["a", "b"]))) == {"a": 1, "b": 2}
    assert to_python(from_python(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))) == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


@pytest.mark.parametrize("bad, exc", [(float("nan"), ValueError), ({1, 2}, TypeError), (object(), TypeError),
                                      (lambda v: v, TypeError), (np.array(["2026-10-31"], dtype="datetime64[D]"), TypeError)])
def test_unconvertible_values_raise(bad, exc):
    with pytest.raises(exc):
        from_python(bad)


def test_foreign_atoms_no_longer_inert():
    """Today Expr("Plus", Fraction(1, 3), 1) stays as the inert '1/3 + 1'; through from_python it evaluates."""
    s = Session()
    assert s.eval(Expr("Plus", fractions.Fraction(1, 3), 1)) == fractions.Fraction(4, 3)
    assert s.eval(Expr("Plus", np.int64(2), 1)) == 3
    assert s.eval(Expr("Total", np.arange(5))) == 10


# ── calling MikoshiLang from Python ──

def test_namespace_calls():
    s = Session()
    ml = s.ns
    x = ml.symbols("x")
    assert str(ml.Integrate(x ** 2, x)) == "x^3/3"
    assert ml.Total([1, 2, 3]) == 6
    assert ml.Solve(Expr("Equal", x ** 2, 4), x) == [[Expr("Rule", x, -2)], [Expr("Rule", x, 2)]]
    assert ml.N(Symbol("Pi"), 30) == s.eval("N[Pi, 30]")
    assert ml.Mean(np.array([1.0, 2.0, 3.0])) == 2.0
    assert ml.Keys({"a": 1, "b": 2}) == ["a", "b"]


# ── calling Python from MikoshiLang ──

def test_function_decorator_types_and_errors():
    s = Session()

    @s.function
    def hypot2(a: float, b: float) -> float:
        return a * a + b * b

    @s.function(listable=True, check="reference")
    def collatz_steps(n: int) -> int:
        if n < 1:
            raise ValueError("n must be positive")
        k = 0
        while n != 1:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            k += 1
        return k

    assert hypot2.mikoshi.name == "Hypot2" and hypot2.mikoshi.signature == "{Real, Real} -> Real"
    assert "Check: reference" in collatz_steps.mikoshi.notes
    assert s.eval("Hypot2[3, 4]") == 25.0
    assert s.eval("Hypot2[1/2, 0]") == 0.25
    assert s.eval("CollatzSteps[{1, 6, 27}]") == [0, 8, 111]
    assert str(s.eval("CollatzSteps[n]")) == "CollatzSteps[n]"           # symbolic: unevaluated
    with pytest.raises(TypeError, match=r"^CollatzSteps: argument 1 must be Integer"):
        s.eval("CollatzSteps[2.5]")
    with pytest.raises(ValueError, match=r"^CollatzSteps: n must be positive"):
        s.eval("CollatzSteps[0]")
    with pytest.raises(ValueError, match="called with 2 argument"):
        s.eval("CollatzSteps[1, 2]")
    assert s.eval('Check[CollatzSteps[0], "caught"]') == "caught"      # catchable inside the language
    assert s.eval("Map[CollatzSteps, Range[3]]") == [0, 1, 7]          # composes with the language


def test_function_is_session_local_and_cannot_shadow_builtins():
    s, other = Session(), Session()

    @s.function
    def twice(v: int) -> int:
        return 2 * v

    assert s.eval("Twice[21]") == 42
    assert str(other.eval("Twice[21]")) == "Twice[21]"
    with pytest.raises(ValueError, match="built-in head"):
        s.function(lambda v: v, name="Sin")


def test_function_foreign_exception_wrapped():
    s = Session()

    @s.function
    def lookup_code(key: str) -> int:
        return {"a": 1}[key]

    with pytest.raises(ValueError, match=r"^LookupCode: Python raised KeyError"):
        s.eval('LookupCode["zz"]')


def test_mtype_vocabulary():
    from typing import List, Optional
    assert mtype(int) == "Integer" and mtype(float) == "Real" and mtype(str) == "String"
    assert mtype(List[int]) == "ListOf[Integer]" and mtype(Optional[int]) == "Integer"
    assert mtype(np.ndarray) == "ListOf[Number]" and mtype(sp.Expr) == "Expr" and mtype("PosInteger") == "PosInteger"


def test_python_function_gating():
    assert str(Session(allow_python=True).eval('PythonFunction["math.gcd"][12, 18]')) == "6"
    with pytest.raises(PermissionError):
        Session().eval('PythonFunction["math.gcd"][12, 18]')
    s = Session(allow_python={"math"})
    assert s.eval('PythonFunction["math.sqrt"][16]') == 4.0
    with pytest.raises(PermissionError):
        s.eval('PythonFunction["os.getcwd"][]')
    assert str(s.eval('PythonFunction["math.sqrt"][y]')) == 'PythonFunction["math.sqrt"][y]'
    with pytest.raises(ValueError):
        Session(allow_python=True).eval('PythonFunction["os.__dict__"][]')
