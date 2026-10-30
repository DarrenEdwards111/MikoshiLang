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
    expr = _numericize_special(expr, precision)
    from .precision import input_precision
    import mpmath
    pin = input_precision([expr])
    if pin == "m":
        precision = min(precision, 15)       # N never raises the precision of a machine number (Wolfram)
    elif pin is not None:
        precision = min(precision, max(16, int(mpmath.libmp.prec_to_dps(pin))))
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


def _leaves(args):
    out = []
    for a in args:
        if isinstance(a, Expr):
            out.extend(_leaves(a.args))
        else:
            out.append(a)
    return out


def _numericize_special(expr, precision):
    """Evaluate calls of mpmath-backed special functions (which stay symbolic for exact arguments) with
    approximate arguments at ``precision`` digits, so that N[f[exact args], p] gives p digits."""
    if not isinstance(expr, Expr):
        return expr
    from .library.wl_special import MP_NUMERIC
    args = tuple(_numericize_special(a, precision) for a in expr.args)
    if isinstance(expr.head, str) and expr.head in MP_NUMERIC:
        from .precision import input_precision
        import mpmath
        pin = input_precision(_leaves(args))
        if pin == "m" or (pin is not None and mpmath.libmp.prec_to_dps(pin) <= 15):
            precision = 15          # a machine number among the inputs makes the result a machine number
        elif pin is not None:
            precision = min(precision, int(mpmath.libmp.prec_to_dps(pin)))
        num = []
        for a in args:
            try:
                num.append(numerical(a, precision + 5) if not isinstance(a, (str, bool)) else a)
            except Exception:
                num.append(a)
        try:
            from .library.core import Unevaluated
            out = MP_NUMERIC[expr.head](*num)
        except Unevaluated:
            return Expr(expr.head, *num)
        from .precision import round_to
        import mpmath
        return round_to(out, "m" if precision <= 15 else mpmath.libmp.dps_to_prec(precision))
    if any(n is not o for n, o in zip(args, expr.args)):
        return Expr(expr.head, *args)
    return expr
