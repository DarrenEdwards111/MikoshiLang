"""Tests for LaTeX rendering."""

import pytest
from mikoshilang.latex import to_latex
from mikoshilang.expr import Expr, Symbol


def test_integer():
    assert to_latex(42) == "42"

def test_float():
    assert to_latex(3.14) == "3.14"

def test_symbol():
    assert to_latex(Symbol("x")) == "x"

def test_pi():
    assert to_latex(Symbol("Pi")) == r"\pi"

def test_e():
    assert to_latex(Symbol("E")) == "e"

def test_infinity():
    assert to_latex(Symbol("Infinity")) == r"\infty"

def test_plus():
    expr = Expr("Plus", Symbol("x"), Symbol("y"))
    assert to_latex(expr) == "x + y"

def test_times_fraction():
    expr = Expr("Times", Symbol("x"), Expr("Power", Symbol("y"), -1))
    assert to_latex(expr) == r"\frac{x}{y}"

def test_power():
    expr = Expr("Power", Symbol("x"), 2)
    assert to_latex(expr) == "x^{2}"

def test_sin():
    expr = Expr("Sin", Symbol("x"))
    assert to_latex(expr) == r"\sin\left(x\right)"

def test_cos():
    expr = Expr("Cos", Symbol("x"))
    assert to_latex(expr) == r"\cos\left(x\right)"

def test_exp():
    expr = Expr("Exp", Symbol("x"))
    assert to_latex(expr) == r"e^{x}"

def test_log():
    expr = Expr("Log", Symbol("x"))
    assert to_latex(expr) == r"\ln\left(x\right)"

def test_log_base():
    expr = Expr("Log", 2, Symbol("x"))
    assert to_latex(expr) == r"\log_{2}\left(x\right)"

def test_sqrt():
    expr = Expr("Sqrt", Symbol("x"))
    assert to_latex(expr) == r"\sqrt{x}"

def test_abs():
    expr = Expr("Abs", Symbol("x"))
    assert to_latex(expr) == r"\left|x\right|"

def test_list():
    expr = Expr("List", 1, 2, 3)
    assert to_latex(expr) == r"\{1, 2, 3\}"

def test_matrix():
    expr = Expr("Matrix", Expr("List", 1, 2), Expr("List", 3, 4))
    result = to_latex(expr)
    assert r"\begin{pmatrix}" in result
    assert "1 & 2" in result

def test_equal():
    expr = Expr("Equal", Symbol("x"), Symbol("y"))
    assert to_latex(expr) == "x = y"

def test_less():
    expr = Expr("Less", Symbol("x"), 0)
    assert to_latex(expr) == "x < 0"

def test_and():
    expr = Expr("And", Symbol("x"), Symbol("y"))
    assert to_latex(expr) == r"x \land y"

def test_or():
    expr = Expr("Or", Symbol("x"), Symbol("y"))
    assert to_latex(expr) == r"x \lor y"

def test_not():
    expr = Expr("Not", Symbol("x"))
    assert to_latex(expr) == r"\lnot x"

def test_integrate():
    expr = Expr("Integrate", Symbol("x"), Symbol("x"))
    assert r"\int" in to_latex(expr)

def test_diff():
    expr = Expr("Diff", Expr("Power", Symbol("x"), 2), Symbol("x"))
    assert r"\frac{d}{dx}" in to_latex(expr)

def test_limit():
    expr = Expr("Limit", Symbol("x"), Symbol("x"), 0)
    result = to_latex(expr)
    assert r"\lim" in result

def test_rule():
    expr = Expr("Rule", Symbol("x"), 3)
    assert to_latex(expr) == r"x \to 3"

def test_negative_coeff():
    expr = Expr("Times", -1, Symbol("x"))
    assert to_latex(expr) == "-x"

def test_boolean_true():
    assert to_latex(True) == r"\text{True}"

def test_boolean_false():
    assert to_latex(False) == r"\text{False}"

def test_string():
    assert r'"hello"' in to_latex("hello")

def test_greek_alpha():
    assert to_latex(Symbol("alpha")) == r"\alpha"

def test_generic_function():
    expr = Expr("Foo", Symbol("x"), Symbol("y"))
    result = to_latex(expr)
    assert "Foo" in result
