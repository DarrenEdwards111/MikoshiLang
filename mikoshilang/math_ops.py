"""Symbolic math operations — wrapping SymPy through MikoshiLang's expression system."""

from __future__ import annotations
from typing import Any, Optional
import sympy as sp
from .expr import Expr, Symbol


# ── Conversion: MikoshiLang <-> SymPy ──

def to_sympy(expr) -> Any:
    """Convert a MikoshiLang expression to a SymPy expression."""
    if isinstance(expr, (int, float)):
        return sp.sympify(expr)
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, Symbol):
        if expr.name == "Pi":
            return sp.pi
        if expr.name == "E":
            return sp.E
        if expr.name == "I":
            return sp.I
        if expr.name == "Infinity":
            return sp.oo
        return sp.Symbol(expr.name)
    if isinstance(expr, Expr):
        head = expr.head
        args = [to_sympy(a) for a in expr.args]
        _map = {
            "Plus": lambda a: sum(a[1:], a[0]),
            "Times": lambda a: a[0] if len(a) == 1 else a[0] * a[1] if len(a) == 2 else sp.Mul(*a),
            "Power": lambda a: a[0] ** a[1],
            "Sin": lambda a: sp.sin(a[0]),
            "Cos": lambda a: sp.cos(a[0]),
            "Tan": lambda a: sp.tan(a[0]),
            "ArcSin": lambda a: sp.asin(a[0]),
            "ArcCos": lambda a: sp.acos(a[0]),
            "ArcTan": lambda a: sp.atan(a[0]),
            "Exp": lambda a: sp.exp(a[0]),
            "Log": lambda a: sp.log(*a),
            "Sqrt": lambda a: sp.sqrt(a[0]),
            "Abs": lambda a: sp.Abs(a[0]),
        }
        if head in _map:
            return _map[head](args)
        # Fallback: try SymPy function
        if hasattr(sp, head.lower()):
            return getattr(sp, head.lower())(*args)
        raise ValueError(f"Cannot convert {head} to SymPy")
    return sp.sympify(expr)


def from_sympy(sexpr) -> Any:
    """Convert a SymPy expression back to MikoshiLang."""
    if sexpr is sp.pi:
        return Symbol("Pi")
    if sexpr is sp.E:
        return Symbol("E")
    if sexpr is sp.I:
        return Symbol("I")
    if sexpr is sp.oo:
        return Symbol("Infinity")
    if sexpr is sp.zoo:
        return Symbol("ComplexInfinity")
    if isinstance(sexpr, sp.Integer):
        return int(sexpr)
    if isinstance(sexpr, sp.Rational):
        if sexpr.q == 1:
            return int(sexpr.p)
        return Expr("Times", int(sexpr.p), Expr("Power", int(sexpr.q), -1))
    if isinstance(sexpr, sp.Float):
        return float(sexpr)
    if isinstance(sexpr, sp.Symbol):
        return Symbol(str(sexpr))
    if isinstance(sexpr, sp.Add):
        args = [from_sympy(a) for a in sexpr.args]
        if len(args) == 1:
            return args[0]
        result = args[0]
        for a in args[1:]:
            result = Expr("Plus", result, a)
        return result
    if isinstance(sexpr, sp.Mul):
        args = [from_sympy(a) for a in sexpr.args]
        if len(args) == 1:
            return args[0]
        result = args[0]
        for a in args[1:]:
            result = Expr("Times", result, a)
        return result
    if isinstance(sexpr, sp.Pow):
        return Expr("Power", from_sympy(sexpr.base), from_sympy(sexpr.exp))
    if isinstance(sexpr, sp.sin):
        return Expr("Sin", from_sympy(sexpr.args[0]))
    if isinstance(sexpr, sp.cos):
        return Expr("Cos", from_sympy(sexpr.args[0]))
    if isinstance(sexpr, sp.tan):
        return Expr("Tan", from_sympy(sexpr.args[0]))
    if isinstance(sexpr, sp.exp):
        return Expr("Exp", from_sympy(sexpr.args[0]))
    if isinstance(sexpr, sp.log):
        return Expr("Log", *(from_sympy(a) for a in sexpr.args))
    if isinstance(sexpr, sp.Number):
        return float(sexpr)
    # Fallback
    if hasattr(sexpr, 'args') and hasattr(sexpr, 'func'):
        fname = type(sexpr).__name__
        args = [from_sympy(a) for a in sexpr.args]
        return Expr(fname, *args)
    return str(sexpr)


# ── High-level operations ──

def simplify(expr) -> Any:
    """Algebraic simplification."""
    return from_sympy(sp.simplify(to_sympy(expr)))


def expand(expr) -> Any:
    """Expand products and powers."""
    return from_sympy(sp.expand(to_sympy(expr)))


def factor(expr) -> Any:
    """Factor an expression."""
    return from_sympy(sp.factor(to_sympy(expr)))


def diff(expr, var, n: int = 1) -> Any:
    """Differentiate expr with respect to var, n times."""
    sv = to_sympy(var) if isinstance(var, Symbol) else sp.Symbol(str(var))
    return from_sympy(sp.diff(to_sympy(expr), sv, n))


def integrate(expr, var, *limits) -> Any:
    """Integrate expr with respect to var. Optional limits for definite integrals."""
    sv = to_sympy(var) if isinstance(var, Symbol) else sp.Symbol(str(var))
    if limits:
        return from_sympy(sp.integrate(to_sympy(expr), (sv, *[to_sympy(l) for l in limits])))
    return from_sympy(sp.integrate(to_sympy(expr), sv))


def solve(expr, var) -> Any:
    """Solve expr == 0 for var."""
    sv = to_sympy(var) if isinstance(var, Symbol) else sp.Symbol(str(var))
    solutions = sp.solve(to_sympy(expr), sv)
    return Expr("List", *(from_sympy(s) for s in solutions))


def limit(expr, var, point) -> Any:
    """Compute limit of expr as var -> point."""
    sv = to_sympy(var) if isinstance(var, Symbol) else sp.Symbol(str(var))
    return from_sympy(sp.limit(to_sympy(expr), sv, to_sympy(point)))


def series(expr, var, point=0, order=6) -> Any:
    """Taylor series expansion."""
    sv = to_sympy(var) if isinstance(var, Symbol) else sp.Symbol(str(var))
    s = sp.series(to_sympy(expr), sv, to_sympy(point), order)
    # Remove O() term
    return from_sympy(s.removeO())
