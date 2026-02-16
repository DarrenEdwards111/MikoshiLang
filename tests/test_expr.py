"""Tests for the expression system."""

import pytest
from mikoshilang.expr import Expr, Symbol, symbols


class TestSymbol:
    def test_create(self):
        x = Symbol("x")
        assert x.name == "x"

    def test_repr(self):
        assert repr(Symbol("x")) == "x"

    def test_equality(self):
        assert Symbol("x") == Symbol("x")
        assert Symbol("x") != Symbol("y")

    def test_hash(self):
        assert hash(Symbol("x")) == hash(Symbol("x"))
        s = {Symbol("x"), Symbol("x"), Symbol("y")}
        assert len(s) == 2

    def test_neg(self):
        x = Symbol("x")
        r = -x
        assert isinstance(r, Expr)
        assert r.head == "Times"
        assert r.args == (-1, x)

    def test_pos(self):
        x = Symbol("x")
        assert +x is x


class TestExpr:
    def test_create(self):
        e = Expr("Plus", 1, 2)
        assert e.head == "Plus"
        assert e.args == (1, 2)

    def test_equality(self):
        assert Expr("Plus", 1, 2) == Expr("Plus", 1, 2)
        assert Expr("Plus", 1, 2) != Expr("Plus", 2, 1)

    def test_hash(self):
        e1 = Expr("Plus", 1, 2)
        e2 = Expr("Plus", 1, 2)
        assert hash(e1) == hash(e2)

    def test_nested(self):
        e = Expr("Plus", Expr("Times", 2, 3), 4)
        assert e.head == "Plus"
        assert e.args[0].head == "Times"


class TestArithmetic:
    def test_symbol_add(self):
        x = Symbol("x")
        r = x + 1
        assert isinstance(r, Expr)
        assert r.head == "Plus"

    def test_symbol_radd(self):
        x = Symbol("x")
        r = 1 + x
        assert r.head == "Plus"
        assert r.args == (1, x)

    def test_symbol_mul(self):
        x = Symbol("x")
        r = x * 3
        assert r.head == "Times"

    def test_symbol_rmul(self):
        x = Symbol("x")
        r = 3 * x
        assert r.head == "Times"
        assert r.args == (3, x)

    def test_symbol_sub(self):
        x = Symbol("x")
        r = x - 1
        assert r.head == "Plus"

    def test_symbol_rsub(self):
        x = Symbol("x")
        r = 1 - x
        assert r.head == "Plus"

    def test_symbol_pow(self):
        x = Symbol("x")
        r = x ** 2
        assert r.head == "Power"
        assert r.args == (x, 2)

    def test_symbol_div(self):
        x = Symbol("x")
        r = x / 2
        assert r.head == "Times"

    def test_expr_add(self):
        x = Symbol("x")
        r = (x + 1) + 2
        assert r.head == "Plus"

    def test_expr_mul(self):
        x = Symbol("x")
        r = (x + 1) * 2
        assert r.head == "Times"

    def test_expr_pow(self):
        x = Symbol("x")
        r = (x + 1) ** 2
        assert r.head == "Power"

    def test_complex_expr(self):
        x = Symbol("x")
        r = x + 3 * x ** 2
        assert isinstance(r, Expr)


class TestRepr:
    def test_plus(self):
        x = Symbol("x")
        assert repr(x + 1) == "x + 1"

    def test_times(self):
        x = Symbol("x")
        assert repr(3 * x) == "3*x"

    def test_power(self):
        x = Symbol("x")
        assert repr(x ** 2) == "x^2"

    def test_list(self):
        e = Expr("List", 1, 2, 3)
        assert repr(e) == "{1, 2, 3}"

    def test_function(self):
        x = Symbol("x")
        e = Expr("Sin", x)
        assert repr(e) == "Sin[x]"

    def test_nested_repr(self):
        x = Symbol("x")
        e = Expr("Sin", x ** 2)
        assert repr(e) == "Sin[x^2]"


class TestSymbols:
    def test_single(self):
        x = symbols("x")
        assert isinstance(x, Symbol)
        assert x.name == "x"

    def test_multiple(self):
        x, y, z = symbols("x", "y", "z")
        assert x.name == "x"
        assert y.name == "y"
        assert z.name == "z"

    def test_space_separated(self):
        x, y = symbols("x y")
        assert x.name == "x"
        assert y.name == "y"
