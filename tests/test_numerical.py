"""Tests for numerical evaluation."""

import pytest
import math
from mikoshilang.expr import Expr, Symbol
from mikoshilang.numerical import N, numerical, Pi, E, I, Infinity, GoldenRatio


class TestConstants:
    def test_pi(self):
        assert abs(N(Pi) - math.pi) < 1e-10

    def test_e(self):
        assert abs(N(E) - math.e) < 1e-10

    def test_i(self):
        assert N(I) == complex(0, 1)

    def test_infinity(self):
        assert N(Infinity) == float("inf")

    def test_golden_ratio(self):
        phi = (1 + math.sqrt(5)) / 2
        assert abs(N(GoldenRatio) - phi) < 1e-10

    def test_pi_symbol(self):
        assert Pi.name == "Pi"

    def test_e_symbol(self):
        assert E.name == "E"


class TestNumerical:
    def test_int(self):
        assert N(42) == 42.0

    def test_float(self):
        assert N(3.14) == 3.14

    def test_symbol_unknown(self):
        x = Symbol("x")
        assert N(x) == x

    def test_expr_numeric(self):
        e = Expr("Plus", 1, 2)
        r = N(e)
        assert abs(r - 3.0) < 1e-10

    def test_sin_pi(self):
        e = Expr("Sin", Symbol("Pi"))
        r = N(e)
        assert abs(r) < 1e-10

    def test_cos_zero(self):
        e = Expr("Cos", 0)
        r = N(e)
        assert abs(r - 1.0) < 1e-10

    def test_exp_zero(self):
        e = Expr("Exp", 0)
        r = N(e)
        assert abs(r - 1.0) < 1e-10

    def test_log_e(self):
        e = Expr("Log", Symbol("E"))
        r = N(e)
        assert abs(r - 1.0) < 1e-10

    def test_power(self):
        e = Expr("Power", 2, 10)
        r = N(e)
        assert abs(r - 1024.0) < 1e-10

    def test_precision(self):
        r = N(Pi, 30)
        assert abs(r - math.pi) < 1e-14

    def test_numerical_alias(self):
        assert numerical(42) == N(42)

    def test_complex_expr(self):
        # sin(pi/4) ≈ sqrt(2)/2
        e = Expr("Sin", Expr("Times", Symbol("Pi"), Expr("Power", 4, -1)))
        r = N(e)
        assert abs(r - math.sqrt(2) / 2) < 1e-10
