"""Tests for symbolic math operations."""

import pytest
from mikoshilang.expr import Expr, Symbol
from mikoshilang.math_ops import simplify, expand, factor, diff, integrate, solve, limit, series, to_sympy, from_sympy
import sympy as sp


x = Symbol("x")
y = Symbol("y")


class TestToSympy:
    def test_int(self):
        assert to_sympy(42) == 42

    def test_float(self):
        assert to_sympy(3.14) == pytest.approx(3.14)

    def test_symbol(self):
        assert to_sympy(x) == sp.Symbol("x")

    def test_pi(self):
        assert to_sympy(Symbol("Pi")) == sp.pi

    def test_e(self):
        assert to_sympy(Symbol("E")) == sp.E

    def test_plus(self):
        e = Expr("Plus", x, 1)
        assert to_sympy(e) == sp.Symbol("x") + 1

    def test_times(self):
        e = Expr("Times", 3, x)
        assert to_sympy(e) == 3 * sp.Symbol("x")

    def test_power(self):
        e = Expr("Power", x, 2)
        assert to_sympy(e) == sp.Symbol("x") ** 2

    def test_sin(self):
        e = Expr("Sin", x)
        assert to_sympy(e) == sp.sin(sp.Symbol("x"))

    def test_cos(self):
        e = Expr("Cos", x)
        assert to_sympy(e) == sp.cos(sp.Symbol("x"))

    def test_exp(self):
        e = Expr("Exp", x)
        assert to_sympy(e) == sp.exp(sp.Symbol("x"))

    def test_log(self):
        e = Expr("Log", x)
        assert to_sympy(e) == sp.log(sp.Symbol("x"))


class TestFromSympy:
    def test_int(self):
        assert from_sympy(sp.Integer(42)) == 42

    def test_symbol(self):
        r = from_sympy(sp.Symbol("x"))
        assert isinstance(r, Symbol) and r.name == "x"

    def test_pi(self):
        r = from_sympy(sp.pi)
        assert isinstance(r, Symbol) and r.name == "Pi"

    def test_add(self):
        r = from_sympy(sp.Symbol("x") + 1)
        assert isinstance(r, Expr)

    def test_mul(self):
        r = from_sympy(3 * sp.Symbol("x"))
        assert isinstance(r, Expr)

    def test_pow(self):
        r = from_sympy(sp.Symbol("x") ** 2)
        assert isinstance(r, Expr) and r.head == "Power"


class TestDiff:
    def test_polynomial(self):
        # d/dx(x^2) = 2x
        e = Expr("Power", x, 2)
        r = diff(e, x)
        # Should be 2*x in some form
        s = to_sympy(r)
        assert s == 2 * sp.Symbol("x")

    def test_sin(self):
        # d/dx(sin(x)) = cos(x)
        e = Expr("Sin", x)
        r = diff(e, x)
        s = to_sympy(r)
        assert s == sp.cos(sp.Symbol("x"))

    def test_exp(self):
        e = Expr("Exp", x)
        r = diff(e, x)
        s = to_sympy(r)
        assert s == sp.exp(sp.Symbol("x"))

    def test_higher_order(self):
        e = Expr("Power", x, 3)
        r = diff(e, x, 2)
        s = to_sympy(r)
        assert s == 6 * sp.Symbol("x")

    def test_constant(self):
        r = diff(5, x)
        assert to_sympy(r) == 0


class TestIntegrate:
    def test_polynomial(self):
        # ∫x dx = x²/2
        r = integrate(x, x)
        s = to_sympy(r)
        assert s == sp.Symbol("x") ** 2 / 2

    def test_sin(self):
        e = Expr("Sin", x)
        r = integrate(e, x)
        s = to_sympy(r)
        assert s == -sp.cos(sp.Symbol("x"))

    def test_definite(self):
        # ∫₀¹ x dx = 1/2
        r = integrate(x, x, 0, 1)
        s = to_sympy(r)
        assert s == sp.Rational(1, 2)


class TestSolve:
    def test_linear(self):
        # x - 3 = 0 -> x = 3
        e = Expr("Plus", x, -3)
        r = solve(e, x)
        assert isinstance(r, Expr) and r.head == "List"
        assert 3 in r.args

    def test_quadratic(self):
        # x^2 - 4 = 0 -> x = ±2
        e = Expr("Plus", Expr("Power", x, 2), -4)
        r = solve(e, x)
        assert len(r.args) == 2
        vals = set(r.args)
        assert -2 in vals and 2 in vals


class TestSimplify:
    def test_trig_identity(self):
        # sin²x + cos²x = 1
        e = Expr("Plus",
                  Expr("Power", Expr("Sin", x), 2),
                  Expr("Power", Expr("Cos", x), 2))
        r = simplify(e)
        assert to_sympy(r) == 1

    def test_cancel(self):
        # (x^2 - 1)/(x - 1) = x + 1
        num = Expr("Plus", Expr("Power", x, 2), -1)
        den = Expr("Plus", x, -1)
        e = Expr("Times", num, Expr("Power", den, -1))
        r = simplify(e)
        assert to_sympy(r) == sp.Symbol("x") + 1


class TestExpand:
    def test_square(self):
        # (x + 1)^2 = x^2 + 2x + 1
        e = Expr("Power", Expr("Plus", x, 1), 2)
        r = expand(e)
        s = to_sympy(r)
        assert sp.expand(s) == sp.Symbol("x") ** 2 + 2 * sp.Symbol("x") + 1


class TestFactor:
    def test_difference_of_squares(self):
        # x^2 - 1 = (x-1)(x+1)
        e = Expr("Plus", Expr("Power", x, 2), -1)
        r = factor(e)
        s = to_sympy(r)
        assert s == sp.factor(sp.Symbol("x") ** 2 - 1)


class TestLimit:
    def test_simple(self):
        # lim x->0 sin(x)/x = 1
        e = Expr("Times", Expr("Sin", x), Expr("Power", x, -1))
        r = limit(e, x, 0)
        assert to_sympy(r) == 1


class TestSeries:
    def test_exp_series(self):
        # e^x around 0 to order 4: 1 + x + x²/2 + x³/6
        e = Expr("Exp", x)
        r = series(e, x, 0, 4)
        s = to_sympy(r)
        sx = sp.Symbol("x")
        expected = 1 + sx + sx ** 2 / 2 + sx ** 3 / 6
        assert sp.simplify(s - expected) == 0
