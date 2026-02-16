"""Tests for built-in functions."""

import pytest
import math
from mikoshilang.expr import Expr, Symbol
from mikoshilang.builtins import (
    Sin, Cos, Tan, ArcSin, ArcCos, ArcTan,
    Exp, Log, Log2, Log10,
    Prime, PrimeQ, FactorInteger, GCD, LCM, Mod,
    Factorial, Binomial, Fibonacci,
    And, Or, Not, Xor, Implies,
    Equal, Less, Greater, LessEqual, GreaterEqual,
    BUILTIN_FUNCTIONS,
)


x = Symbol("x")


class TestTrig:
    def test_sin_creates_expr(self):
        e = Sin(x)
        assert isinstance(e, Expr) and e.head == "Sin"

    def test_cos_creates_expr(self):
        e = Cos(x)
        assert isinstance(e, Expr) and e.head == "Cos"

    def test_tan_creates_expr(self):
        e = Tan(x)
        assert isinstance(e, Expr) and e.head == "Tan"

    def test_arcsin_creates_expr(self):
        e = ArcSin(x)
        assert isinstance(e, Expr) and e.head == "ArcSin"

    def test_arccos_creates_expr(self):
        e = ArcCos(x)
        assert isinstance(e, Expr) and e.head == "ArcCos"

    def test_arctan_creates_expr(self):
        e = ArcTan(x)
        assert isinstance(e, Expr) and e.head == "ArcTan"

    def test_sin_nested(self):
        e = Sin(Cos(x))
        assert e.head == "Sin"
        assert e.args[0].head == "Cos"


class TestExponential:
    def test_exp(self):
        e = Exp(x)
        assert e.head == "Exp"

    def test_log(self):
        e = Log(x)
        assert e.head == "Log" and e.args == (x,)

    def test_log_base(self):
        e = Log(x, 2)
        assert e.head == "Log" and e.args == (2, x)

    def test_log2(self):
        e = Log2(x)
        assert e.head == "Log" and e.args == (2, x)

    def test_log10(self):
        e = Log10(x)
        assert e.head == "Log" and e.args == (10, x)


class TestNumberTheory:
    def test_prime_1(self):
        assert Prime(1) == 2

    def test_prime_5(self):
        assert Prime(5) == 11

    def test_prime_10(self):
        assert Prime(10) == 29

    def test_primeq_true(self):
        assert PrimeQ(7) is True

    def test_primeq_false(self):
        assert PrimeQ(4) is False

    def test_primeq_2(self):
        assert PrimeQ(2) is True

    def test_primeq_1(self):
        assert PrimeQ(1) is False

    def test_factor_integer(self):
        r = FactorInteger(12)
        assert r.head == "List"
        # 12 = 2^2 * 3
        factors = {(a.args[0], a.args[1]) for a in r.args}
        assert factors == {(2, 2), (3, 1)}

    def test_factor_prime(self):
        r = FactorInteger(7)
        assert len(r.args) == 1
        assert r.args[0].args == (7, 1)

    def test_gcd(self):
        assert GCD(12, 8) == 4
        assert GCD(15, 25) == 5

    def test_gcd_multiple(self):
        assert GCD(12, 18, 24) == 6

    def test_lcm(self):
        assert LCM(4, 6) == 12
        assert LCM(3, 5) == 15

    def test_lcm_multiple(self):
        assert LCM(2, 3, 4) == 12

    def test_mod(self):
        assert Mod(10, 3) == 1
        assert Mod(15, 5) == 0


class TestCombinatorics:
    def test_factorial_0(self):
        assert Factorial(0) == 1

    def test_factorial_5(self):
        assert Factorial(5) == 120

    def test_factorial_10(self):
        assert Factorial(10) == 3628800

    def test_binomial(self):
        assert Binomial(5, 2) == 10
        assert Binomial(10, 0) == 1
        assert Binomial(10, 10) == 1

    def test_binomial_pascal(self):
        # Pascal's rule: C(n,k) = C(n-1,k-1) + C(n-1,k)
        assert Binomial(10, 5) == Binomial(9, 4) + Binomial(9, 5)

    def test_fibonacci_0(self):
        assert Fibonacci(0) == 0

    def test_fibonacci_1(self):
        assert Fibonacci(1) == 1

    def test_fibonacci_10(self):
        assert Fibonacci(10) == 55

    def test_fibonacci_20(self):
        assert Fibonacci(20) == 6765


class TestLogic:
    def test_and_true(self):
        assert And(True, True) is True

    def test_and_false(self):
        assert And(True, False) is False

    def test_and_multiple(self):
        assert And(True, True, True) is True
        assert And(True, True, False) is False

    def test_or_true(self):
        assert Or(False, True) is True

    def test_or_false(self):
        assert Or(False, False) is False

    def test_or_multiple(self):
        assert Or(False, False, True) is True

    def test_not(self):
        assert Not(True) is False
        assert Not(False) is True

    def test_xor(self):
        assert Xor(True, False) is True
        assert Xor(True, True) is False
        assert Xor(False, False) is False

    def test_implies(self):
        assert Implies(True, True) is True
        assert Implies(True, False) is False
        assert Implies(False, True) is True
        assert Implies(False, False) is True


class TestComparison:
    def test_equal(self):
        assert Equal(3, 3) is True
        assert Equal(3, 4) is False

    def test_less(self):
        assert Less(3, 4) is True
        assert Less(4, 3) is False

    def test_greater(self):
        assert Greater(4, 3) is True
        assert Greater(3, 4) is False

    def test_less_equal(self):
        assert LessEqual(3, 3) is True
        assert LessEqual(3, 4) is True
        assert LessEqual(4, 3) is False

    def test_greater_equal(self):
        assert GreaterEqual(3, 3) is True
        assert GreaterEqual(4, 3) is True
        assert GreaterEqual(3, 4) is False


class TestRegistry:
    def test_all_registered(self):
        expected = [
            "Sin", "Cos", "Tan", "ArcSin", "ArcCos", "ArcTan",
            "Exp", "Log", "Log2", "Log10",
            "Prime", "PrimeQ", "FactorInteger", "GCD", "LCM", "Mod",
            "Factorial", "Binomial", "Fibonacci",
            "And", "Or", "Not", "Xor", "Implies",
            "Equal", "Less", "Greater", "LessEqual", "GreaterEqual",
        ]
        for name in expected:
            assert name in BUILTIN_FUNCTIONS, f"{name} not registered"

    def test_callable(self):
        for name, fn in BUILTIN_FUNCTIONS.items():
            assert callable(fn), f"{name} is not callable"
