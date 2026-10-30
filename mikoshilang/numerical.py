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
    if isinstance(precision, Symbol) and precision.name == "MachinePrecision":
        precision = 15
    if isinstance(expr, Symbol) and expr.name == "MachinePrecision":
        return 53 * 0.30102999566398120 if precision <= 15 else sp.Float(53, precision) * sp.log(2, 10).evalf(precision)
    if isinstance(precision, float) and precision == int(precision):
        precision = int(precision)
    if isinstance(precision, bool) or not isinstance(precision, int) or precision < 1:
        raise ValueError("Precision must be a positive integer")
    if isinstance(expr, Symbol) and expr.name == "GoldenRatio":
        sexpr = sp.GoldenRatio
    else:
        sexpr = to_sympy(expr)
    result = sexpr.evalf(precision)
    if not result.is_number:
        return from_sympy(result)
    if precision > 15:
        return result
    if result.is_real or result in (sp.oo, -sp.oo):
        return float(result)
    return complex(result)
