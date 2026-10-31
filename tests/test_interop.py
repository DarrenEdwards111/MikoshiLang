"""Python interop: conversion table, Session.eval / ns, @session.function, gated PythonFunction (docs/python_interop.md)."""
import fractions
import math

import numpy as np
import pytest
import sympy as sp

from mikoshilang import Session, Expr, Symbol
from mikoshilang.interop import from_python, to_python, mtype


# ── conversion table (round-trip properties: tests/test_interop_properties.py) ──

def _same(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a == b and math.copysign(1, a) == math.copysign(1, b) or (a == b == 0.0)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


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
    with pytest.raises(ValueError, match="built-in|library head"):
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
    with pytest.raises(PermissionError, match="private and dunder"):
        Session(allow_python=True).eval('PythonFunction["os.__dict__"][]')


def test_python_function_allow_list_forms():
    s = Session(allow_python="math")                               # a single module name is an allow-list
    assert s.allow_python == frozenset({"math"})
    with pytest.raises(PermissionError):
        s.eval('PythonFunction["cmath.sqrt"][4]')
    s = Session(allow_python={"numpy.linalg"})                     # a submodule allows only that submodule
    assert s.eval('PythonFunction["numpy.linalg.det"][{{2, 0}, {0, 3}}]') == pytest.approx(6.0)
    assert type(s.eval('PythonFunction["numpy.linalg.det"][{{2, 0}, {0, 3}}]')) is float   # no numpy scalars leak
    with pytest.raises(PermissionError):
        s.eval('PythonFunction["numpy.sum"][{1, 2}]')
    with pytest.raises(PermissionError, match="private"):
        Session(allow_python=True).eval('PythonFunction["math._private"][1]')
    with pytest.raises(ValueError, match="^PythonFunction: math has no attribute"):
        Session(allow_python=True).eval('PythonFunction["math.nope"][1]')
    with pytest.raises(TypeError):
        Session(allow_python=3)
    with pytest.raises(ValueError):
        Session(allow_python={"os; rm"})


def test_python_function_options_and_symbolic_lists():
    s = Session(allow_python={"math"})
    assert s.eval('PythonFunction["math.isclose"][1, 1.05, RelTol -> 0.1]') is True       # RelTol -> rel_tol
    assert s.eval('PythonFunction["math.isclose"][1, 1.05, "rel_tol" -> 0.01]') is False
    assert str(s.eval('PythonFunction["math.fsum"][{1, y}]')) == 'PythonFunction["math.fsum"][{1, y}]'
    with pytest.raises(ValueError, match="^PythonFunction: math domain error"):
        s.eval('PythonFunction["math.sqrt"][-1]')


def test_permission_errors_are_catchable_by_check():
    s = Session()
    assert s.eval('Check[PythonFunction["math.gcd"][4, 6], "denied"]') == "denied"
    assert s.eval('Check[EntitySearch["x"], "offline"]') == "offline"


@pytest.mark.parametrize("head", ["ExternalEvaluate", "ExternalFunction", "StartExternalSession"])
def test_external_code_strings_raise(head):
    s = Session(allow_python=True)
    with pytest.raises(NotImplementedError, match=f"^{head}: running external code strings is not supported"):
        s.eval(f'{head}["Python", "1 + 1"]')
    assert s.eval(f'Check[{head}["Python", "1 + 1"], "no"]') == "no"


def test_python_function_works_in_main_loop():
    """evaluate_input (REPL, Jupyter) honours allow_python like evaluate."""
    s = Session(allow_python={"math"})
    assert s.evaluate_input('PythonFunction["math.gcd"][12, 18]')[0][1] == 6
    with pytest.raises(PermissionError):
        Session().evaluate_input('PythonFunction["math.gcd"][12, 18]')


# ── Session.eval result forms ──

def test_eval_result_forms():
    s = Session()
    assert s.eval("1 + 2") == 3 and type(s.eval("1 + 2")) is int
    assert s.eval("1/3") == fractions.Fraction(1, 3) and type(s.eval("1/3")) is fractions.Fraction
    assert type(s.eval("2.5")) is float and s.eval('"hi"') == "hi" and s.eval("Null") is None
    assert s.eval("{1, 2.5, \"a\", True}") == [1, 2.5, "a", True]
    assert s.eval('<|"a" -> 1/2, "b" -> {1}|>') == {"a": fractions.Fraction(1, 2), "b": [1]}
    assert s.eval("1. + 2. I") == complex(1, 2)
    assert s.eval("Infinity") == math.inf and s.eval("-Infinity") == -math.inf
    assert isinstance(s.eval("1 + 2 I"), Expr) and isinstance(s.eval("x + 1"), Expr)   # exact/symbolic: Expr
    assert s.eval("x + 1", symbolic="sympy") == sp.Symbol("x") + 1
    assert isinstance(s.eval("1/3", out="expr"), Expr)            # raw MikoshiLang result on request
    assert s.eval(fractions.Fraction(1, 3)) == fractions.Fraction(1, 3)
    assert s.eval({"k": np.array([1, 2])}) == {"k": [1, 2]}
    assert s.evaluate("1/3") == s.eval("1/3", out="expr")           # Session.evaluate is unchanged
    with pytest.raises(ValueError):
        s.eval("1", out="numpy")
    with pytest.raises(ValueError):
        s.eval("1", symbolic="text")


def test_eval_errors_keep_their_class():
    s = Session()
    with pytest.raises(TypeError):
        s.eval(object())
    with pytest.raises(ValueError):
        s.eval(float("nan"))


def test_python_operators_convert_foreign_operands():
    x = Symbol("x")
    assert x + fractions.Fraction(1, 2) == x + Expr("Times", 1, Expr("Power", 2, -1)) or \
        Session().eval(x + fractions.Fraction(1, 2) - x) == fractions.Fraction(1, 2)
    assert Session().eval((x * np.int64(3)) / x) == 3
    with pytest.raises(TypeError):
        x + object()


def test_namespace_options_and_modes():
    s = Session()
    ml = s.ns
    x = ml.symbols("x")
    assert ml.NIntegrate(x ** 2, [x, 0, 1], WorkingPrecision=30) == s.eval("NIntegrate[x^2, {x, 0, 1}, WorkingPrecision -> 30]")
    a = s.ns(arrays="numpy").Range(4)
    assert isinstance(a, np.ndarray) and a.tolist() == [1, 2, 3, 4]
    assert s.ns(symbolic="sympy").D(x ** 3, x) == 3 * sp.Symbol("x") ** 2
    with pytest.raises(AttributeError):
        ml._private


# ── typed functions: more of the boundary ──

def test_function_vocabulary_hints_and_keywords():
    s = Session()

    @s.function(name="Binom", check="exact")
    def binom(n: "NonNegInteger", k: "NonNegInteger") -> "Integer":
        return math.comb(n, k)

    @s.function
    def scale_list(v: list, *, factor: float = 1.0) -> list:
        return [factor * t for t in v]

    @s.function
    def as_real(t: float) -> float:
        return t

    assert binom.mikoshi.signature == "{NonNegInteger, NonNegInteger} -> Integer"
    assert s.eval("Binom[5, 2]") == 10                               # reference: C(5, 2) = 10
    with pytest.raises(ValueError, match=r"^Binom: argument 1 must be a non-negative integer"):
        s.eval("Binom[-1, 2]")
    with pytest.raises(TypeError, match=r"^Binom: argument 2 must be NonNegInteger"):
        s.eval('Binom[5, "a"]')
    assert str(s.eval("Binom[n, 2]")) == "Binom[n, 2]"
    assert s.eval("ScaleList[{1, 2}, Factor -> 2]") == [2.0, 4.0]
    with pytest.raises(ValueError, match="^ScaleList: unknown option Bogus"):
        s.eval("ScaleList[{1, 2}, Bogus -> 2]")
    assert s.eval("AsReal[Pi]") == pytest.approx(math.pi)          # numeric constants convert to float
    assert str(s.eval("AsReal[y]")) == "AsReal[y]"
    assert set(s.functions()) == {"Binom", "ScaleList", "AsReal"}


def test_function_registration_guards():
    s = Session()
    with pytest.raises(ValueError, match="kernel built-in"):
        s.function(lambda v: v, name="Plus", override=True)
    with pytest.raises(ValueError, match="not a valid MikoshiLang symbol"):
        s.function(lambda v: v, name="bad name")
    with pytest.raises(ValueError, match="check must name"):
        s.function(lambda v: v, name="Fine", check="trust me")

    @s.function(name="Mean", override=True)                         # library heads: shadow only on request
    def mean(v: list) -> str:
        return "mine"

    assert s.eval("Mean[{1, 2}]") == "mine"
    assert Session().eval("Mean[{1, 2}]") == fractions.Fraction(3, 2)   # other sessions unaffected
    s.clear()
    assert s.eval("Mean[{1, 2}]") == fractions.Fraction(3, 2)          # clear() drops registrations


def test_function_error_classes_kept():
    s = Session()

    @s.function
    def boom(kind: str) -> int:
        if kind == "nie":
            raise NotImplementedError("not yet")
        if kind == "arith":
            raise ZeroDivisionError("division by zero")
        if kind == "unicode":
            b"\xff".decode("utf-8")
        return object()

    with pytest.raises(NotImplementedError, match="^Boom: not yet"):
        s.eval('Boom["nie"]')
    with pytest.raises(ZeroDivisionError, match="^Boom: division by zero"):
        s.eval('Boom["arith"]')
    with pytest.raises(ValueError, match="^Boom: "):                 # UnicodeDecodeError -> its ValueError base
        s.eval('Boom["unicode"]')
    with pytest.raises(TypeError, match="^Boom: the Python result cannot be converted"):
        s.eval('Boom["other"]')


# ── inexact zero (Wolfram: 0.5 - 0.5 is 0., not exact 0) ──

@pytest.mark.parametrize("src, expected", [
    ("0.5 - 0.5", 0.0), ("0. x", 0.0), ("0.5 x - 0.5 x", 0.0), ("1.0 - 1", 0.0), ("1. I - 1. I", 0.0),
    ("Times[0., 3]", 0.0),
])
def test_inexact_zero_stays_inexact(src, expected):
    v = Session().evaluate(src)
    assert type(v) is float and v == expected


@pytest.mark.parametrize("src", ["x - x", "0 * 1.5", "2.0 * 0", "1 - 1"])
def test_exact_zero_stays_exact(src):
    """An exact 0 factor makes a product exact 0 (as in Wolfram); exact arithmetic is untouched."""
    v = Session().evaluate(src)
    assert type(v) is int and v == 0


def test_inexact_zero_bignum_and_head():
    s = Session()
    v = s.evaluate("N[Pi, 30] - N[Pi, 30]")
    assert isinstance(v, sp.Float) and v.is_zero and v._prec > 53
    assert str(s.evaluate("Head[0.5 - 0.5]")) == "Real"
    assert s.eval("{0.5 - 0.5}") == [0.0] and type(s.eval("0.5 - 0.5")) is float
