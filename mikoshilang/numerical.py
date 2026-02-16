"""Numerical evaluation and constants."""

from __future__ import annotations
from typing import Any
import sympy as sp
from .expr import Expr, Symbol
from .math_ops import to_sympy, from_sympy

# Mathematical constants
Pi = Symbol("Pi")
E = Symbol("E")
I = Symbol("I")
Infinity = Symbol("Infinity")
GoldenRatio = Symbol("GoldenRatio")


def N(expr, precision: int = 15) -> Any:
    """Evaluate expression numerically to given precision."""
    return numerical(expr, precision)


def numerical(expr, precision: int = 15) -> Any:
    """Evaluate expression to floating point with given precision."""
    if isinstance(expr, (int, float)):
        return float(expr)
    if isinstance(expr, Symbol):
        _constants = {
            "Pi": lambda p: float(sp.pi.evalf(p)),
            "E": lambda p: float(sp.E.evalf(p)),
            "I": lambda p: complex(0, 1),
            "Infinity": lambda p: float("inf"),
            "GoldenRatio": lambda p: float(sp.GoldenRatio.evalf(p)),
        }
        if expr.name in _constants:
            return _constants[expr.name](precision)
        return expr
    try:
        sexpr = to_sympy(expr)
        result = sexpr.evalf(precision)
        if result.is_real:
            return float(result)
        if result.is_number:
            return complex(result)
        return from_sympy(result)
    except Exception:
        return expr
