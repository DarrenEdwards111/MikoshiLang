"""Direct public builtin regressions and bounded robustness checks."""
import itertools
import math

import pytest
import sympy as sp

from mikoshilang import (
    Expr, Symbol, Session, Sin, Cos, Tan, ArcSin, ArcCos, ArcTan,
    Exp, Log, Log2, Log10, Prime, PrimeQ, FactorInteger, GCD, LCM,
    Mod, Factorial, Binomial, Fibonacci, And, Or, Not, Xor, Implies,
    Equal, Less, Greater, LessEqual, GreaterEqual,
)
from mikoshilang.math_ops import to_sympy

COVERED_EXPORTS = {
    "Sin", "Cos", "Tan", "ArcSin", "ArcCos", "ArcTan", "Exp", "Log",
    "Log2", "Log10", "Prime", "PrimeQ", "FactorInteger", "GCD", "LCM",
    "Mod", "Factorial", "Binomial", "Fibonacci", "And", "Or", "Not", "Xor",
    "Implies", "Equal", "Less", "Greater", "LessEqual", "GreaterEqual",
}


def test_gcd_normalizes_unary_sign_and_validates_integer():
    assert GCD(-6) == 6
    assert GCD(0) == 0
    assert GCD() == 0
    with pytest.raises(TypeError):
        GCD(2.5)
    with pytest.raises(TypeError):
        GCD(0, 2.5)


def test_lcm_zero_sign_and_integer_domain():
    assert LCM(0, 0) == 0
    assert LCM(-4, 6) == 12
    assert LCM(-6) == 6
    assert LCM() == 1
    with pytest.raises(TypeError):
        LCM(2.5)
    with pytest.raises(TypeError):
        LCM(0, 2.5)


@pytest.mark.parametrize("n", [2.5, sp.Rational(5, 2), Symbol("n")])
def test_fibonacci_rejects_noninteger_indices(n):
    with pytest.raises((TypeError, ValueError)):
        Fibonacci(n)


@pytest.mark.parametrize("func,args", [
    (And, (Symbol("p"), Symbol("q"))),
    (Or, (Symbol("p"), Symbol("q"))),
    (Not, (Symbol("p"),)),
    (Xor, (Symbol("p"), Symbol("q"))),
    (Implies, (Symbol("p"), Symbol("q"))),
])
def test_logic_preserves_unknown_symbolic_operands(func, args):
    expected = Expr(func.__name__, *args)
    assert func(*args) == expected
    assert Session().evaluate(func(*args)) == Session().evaluate(expected)


@pytest.mark.parametrize("func", [Equal, Less, Greater, LessEqual, GreaterEqual])
def test_comparisons_preserve_symbolic_relations(func):
    x, y = Symbol("x"), Symbol("y")
    assert func(x, y) == Expr(func.__name__, x, y)
    relation = Expr("Plus", x, 1)
    assert func(relation, 0) == Expr(func.__name__, relation, 0)
    assert Session().evaluate(func(x, y)) == Expr(func.__name__, x, y)


@pytest.mark.parametrize("func,arg,expected", [
    (Sin, 0, 0), (Cos, 0, 1), (Tan, 0, 0),
    (ArcSin, 1, sp.pi / 2), (ArcCos, -1, sp.pi),
    (ArcTan, 1, sp.pi / 4), (Exp, 0, 1), (Log, 1, 0),
    (Log2, 8, 3), (Log10, 1000, 3),
])
def test_elementary_known_values_and_session_parity(func, arg, expected):
    direct = func(arg)
    assert isinstance(direct, Expr)
    assert to_sympy(direct) == expected
    assert sp.simplify(to_sympy(Session().evaluate(direct)) - expected) == 0


@pytest.mark.parametrize("func,head", [
    (Sin, "Sin"), (Cos, "Cos"), (Tan, "Tan"), (ArcSin, "ArcSin"),
    (ArcCos, "ArcCos"), (ArcTan, "ArcTan"), (Exp, "Exp"), (Log, "Log"),
])
def test_elementary_symbolic_constructors(func, head):
    x = Symbol("x")
    assert func(x) == Expr(head, x)
    assert to_sympy(Session().evaluate(func(x))) == to_sympy(Expr(head, x))


def test_log_base_order_symbolic_and_known_values():
    x, b = Symbol("x"), Symbol("b")
    assert Log(x, base=b) == Expr("Log", b, x)
    assert Log2(x) == Expr("Log", 2, x)
    assert Log10(x) == Expr("Log", 10, x)
    assert to_sympy(Log(81, base=3)) == 4
    assert Session().evaluate(Log(81, base=3)) == 4


@pytest.mark.parametrize("func,reference", [
    (Sin, "sin"), (Cos, "cos"), (Tan, "tan"), (ArcSin, "asin"),
    (ArcCos, "acos"), (ArcTan, "atan"), (Exp, "exp"), (Log, "log"),
])
def test_elementary_complex_values_against_cmath(func, reference):
    import cmath
    z = 0.3 + 0.7j
    assert complex(to_sympy(func(z)).evalf()) == pytest.approx(getattr(cmath, reference)(z))


def test_principal_complex_branches_and_log_bases():
    assert to_sympy(Log(-1)) == sp.I * sp.pi
    assert complex(to_sympy(ArcSin(2)).evalf()) == pytest.approx(math.pi / 2 - 1j * math.log(2 + math.sqrt(3)))
    assert complex(to_sympy(ArcCos(2)).evalf()) == pytest.approx(1j * math.log(2 + math.sqrt(3)))
    assert complex(to_sympy(Exp(Symbol("I") * Symbol("Pi")))) == -1
    assert to_sympy(Log2(-8)) == 3 + sp.I * sp.pi / sp.log(2)
    assert to_sympy(Log10(-100)) == 2 + sp.I * sp.pi / sp.log(10)


def test_number_theory_known_values_and_session_parity():
    assert [Prime(n) for n in range(1, 7)] == [2, 3, 5, 7, 11, 13]
    assert [PrimeQ(n) for n in [-7, 0, 1, 2, 9, 97]] == [False, False, False, True, False, True]
    assert FactorInteger(-12) == Expr("List", Expr("List", -1, 1), Expr("List", 2, 2), Expr("List", 3, 1))
    assert FactorInteger(1) == Expr("List")
    assert FactorInteger(0) == Expr("List", Expr("List", 0, 1))
    assert GCD(-18, 24, 0) == 6
    assert LCM(-4, 6, 0) == 0
    for direct, source in [(Prime(6), "Prime[6]"), (PrimeQ(97), "PrimeQ[97]"),
                           (FactorInteger(360), "FactorInteger[360]"),
                           (GCD(-18, 24), "GCD[-18,24]"), (LCM(-4, 6), "LCM[-4,6]")]:
        assert direct == Session().evaluate(source)


@pytest.mark.parametrize("func", [Prime, PrimeQ, FactorInteger])
@pytest.mark.parametrize("value", [2.5, sp.Rational(5, 2), Symbol("n")])
def test_integer_number_theory_rejects_nonintegers(func, value):
    with pytest.raises((TypeError, ValueError)):
        func(value)


@pytest.mark.parametrize("n", [0, -1, -10])
def test_prime_rejects_nonpositive_indices(n):
    with pytest.raises(ValueError):
        Prime(n)


def test_mod_signed_zero_and_symbolic_inputs():
    assert Mod(-7, 3) == 2
    assert Mod(7, -3) == -2
    assert Mod(0, 3) == 0
    assert Mod(5.5, 2) == 1.5  # Existing real-number remainder support.
    assert Mod(sp.Symbol("n", integer=True), 3) == sp.Mod(sp.Symbol("n", integer=True), 3)
    with pytest.raises(ZeroDivisionError):
        Mod(1, 0)
    with pytest.raises(TypeError):
        Mod(Symbol("n"), 3)
    assert Mod(-7, 3) == Session().evaluate("Mod[-7,3]")


def test_combinatorics_known_values_domains_and_session_parity():
    assert Factorial(0) == 1
    assert Factorial(10) == 3628800
    assert Binomial(5, 2) == 10
    assert Binomial(3, 4) == 0
    assert Fibonacci(0) == 0
    assert Fibonacci(10) == 55
    assert Fibonacci(-10) == -55
    assert Fibonacci(sp.Integer(8)) == 21
    for call in [lambda: Factorial(-1), lambda: Binomial(-1, 2), lambda: Binomial(3, -1)]:
        with pytest.raises(ValueError):
            call()
    for call in [lambda: Factorial(2.5), lambda: Binomial(3.5, 1),
                 lambda: Binomial(3, 1.5), lambda: Factorial(Symbol("n"))]:
        with pytest.raises(TypeError):
            call()
    assert Factorial(10) == Session().evaluate("Factorial[10]")
    assert Binomial(5, 2) == Session().evaluate("Binomial[5,2]")
    assert Fibonacci(-10) == Session().evaluate("Fibonacci[-10]")


def test_boolean_truth_tables_and_identities():
    # Literal output table ordered (False,False), (False,True), (True,False), (True,True).
    expected = {
        And: [False, False, False, True], Or: [False, True, True, True],
        Xor: [False, True, True, False], Implies: [True, True, False, True],
    }
    for func, values in expected.items():
        for (a, b), value in zip(itertools.product([False, True], repeat=2), values):
            assert func(a, b) is value
            assert Session().evaluate(Expr(func.__name__, a, b)) is value
    assert Not(False) is True
    assert Not(True) is False
    assert And() is True
    assert Or() is False
    assert And(True, True, False) is False
    assert Or(False, False, True) is True


def test_boolean_composite_and_mixed_symbolic_operands():
    p = Symbol("p")
    relation = Expr("Less", Symbol("x"), 0)
    for arg in [p, relation]:
        assert And(arg, True) == Expr("And", arg, True)
        assert Or(arg, False) == Expr("Or", arg, False)
        assert Not(arg) == Expr("Not", arg)
        assert Xor(arg, True) == Expr("Xor", arg, True)
        assert Implies(arg, False) == Expr("Implies", arg, False)
        assert Session().evaluate(And(arg, True)) == arg
        assert Session().evaluate(Or(arg, False)) == arg


@pytest.mark.parametrize("a,b,expected", [(1, 2, [False, True, False, True, False]),
                                         (2, 2, [True, False, False, True, True]),
                                         (3, -1, [False, False, True, False, True])])
def test_comparison_known_values_and_session_parity(a, b, expected):
    for func, value in zip([Equal, Less, Greater, LessEqual, GreaterEqual], expected):
        assert func(a, b) is value
        assert Session().evaluate(Expr(func.__name__, a, b)) is value
    assert Equal(Symbol("x"), Symbol("x")) is True
    assert Equal(1 + 2j, 1 + 2j) is True
    with pytest.raises(TypeError):
        Less(1j, 2j)


def test_bounded_integer_properties():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    @hypothesis.settings(max_examples=60, deadline=None)
    @hypothesis.given(st.integers(-1000, 1000), st.integers(-1000, 1000))
    def check(a, b):
        g, l = GCD(a, b), LCM(a, b)
        assert g >= 0 and l >= 0
        assert g * l == abs(a * b)
        assert GCD(a, b) == GCD(b, a)
        assert Less(a, b) == Greater(b, a)
        assert LessEqual(a, b) == Not(Greater(a, b))
        assert GreaterEqual(a, b) == Not(Less(a, b))
        assert Equal(a, b) == And(LessEqual(a, b), GreaterEqual(a, b))
        assert Implies(Less(a, b), Not(Equal(a, b))) is True
        assert Xor(Less(a, b), GreaterEqual(a, b)) is True
        assert Or(Less(a, b), GreaterEqual(a, b)) is True
        if b:
            r = Mod(a, b)
            assert a == (a // b) * b + r
            assert abs(r) < abs(b)
        if g:
            assert a % g == 0 and b % g == 0
        if a:
            assert l % a == 0
        factors = FactorInteger(a).args
        assert math.prod(p ** e for p, e in (pair.args for pair in factors)) == a
    check()


def test_bounded_combinatorics_properties():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    @hypothesis.settings(max_examples=35, deadline=None)
    @hypothesis.given(st.integers(1, 25))
    def check(n):
        assert Factorial(n) == n * Factorial(n - 1)
        assert Fibonacci(n + 1) == Fibonacci(n) + Fibonacci(n - 1)
        assert sum(Binomial(n, k) for k in range(n + 1)) == 2 ** n
        p = Prime(n)
        assert PrimeQ(p) is True
        assert all(p % d for d in range(2, math.isqrt(p) + 1))
    check()


def test_bounded_trigonometric_and_log_identities():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st

    @hypothesis.settings(max_examples=25, deadline=None)
    @hypothesis.given(st.integers(-10, 10), st.integers(1, 10))
    def check(a, b):
        x = Expr("Rational", a, b)
        s, c = to_sympy(Sin(x)), to_sympy(Cos(x))
        assert sp.simplify(s ** 2 + c ** 2) == 1
        assert sp.simplify(to_sympy(Tan(x)) * c - s) == 0
        assert sp.simplify(to_sympy(Log(Exp(x))) - sp.Rational(a, b)) == 0
    check()
