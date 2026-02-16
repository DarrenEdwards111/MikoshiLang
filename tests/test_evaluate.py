"""Tests for the evaluation engine."""

import pytest
from mikoshilang.expr import Expr, Symbol
from mikoshilang.evaluate import evaluate, set_hold_all, set_hold_first, clear_memo


x = Symbol("x")


class TestEvaluate:
    def test_atom_int(self):
        assert evaluate(42) == 42

    def test_atom_float(self):
        assert evaluate(3.14) == 3.14

    def test_atom_string(self):
        assert evaluate("hello") == "hello"

    def test_atom_symbol(self):
        assert evaluate(x) == x

    def test_plus_zero(self):
        assert evaluate(Expr("Plus", x, 0)) == x

    def test_times_one(self):
        assert evaluate(Expr("Times", x, 1)) == x

    def test_times_zero(self):
        assert evaluate(Expr("Times", x, 0)) == 0

    def test_power_one(self):
        assert evaluate(Expr("Power", x, 1)) == x

    def test_power_zero(self):
        assert evaluate(Expr("Power", x, 0)) == 1

    def test_numeric_add(self):
        assert evaluate(Expr("Plus", 3, 4)) == 7

    def test_numeric_mul(self):
        assert evaluate(Expr("Times", 3, 4)) == 12

    def test_numeric_power(self):
        assert evaluate(Expr("Power", 2, 10)) == 1024

    def test_nested(self):
        # (x + 0) * 1 -> x
        e = Expr("Times", Expr("Plus", x, 0), 1)
        assert evaluate(e) == x

    def test_depth_limit(self):
        clear_memo()
        # Shouldn't crash
        e = Expr("Plus", Expr("Plus", Expr("Plus", 1, 2), 3), 4)
        assert evaluate(e) == 10

    def test_unknown_head(self):
        e = Expr("Foo", 1, 2)
        r = evaluate(e)
        assert isinstance(r, Expr) and r.head == "Foo"

    def test_clear_memo(self):
        clear_memo()
        assert evaluate(42) == 42
