"""Precision propagation for approximate numbers (Wolfram semantics, fixed-precision model).

The result of an arithmetic operation or a numeric function on approximate numbers carries the lowest
precision of its inputs: any machine number makes the result a machine number, otherwise the result is
rounded to the smallest binary precision among the arbitrary-precision inputs (N[Pi, 20] + N[E, 50] has
precision 20; N[Pi, 20] + 1.5 is a machine number).  Exact inputs do not limit precision.
"""
import sympy as sp

from .expr import Expr


def input_precision(values):
    """None (all exact), "m" (a machine number is present) or the smallest bignum binary precision."""
    best = None
    stack = list(values)
    while stack:
        x = stack.pop()
        if isinstance(x, bool):
            continue
        if isinstance(x, (float, complex)):
            return "m"
        if isinstance(x, sp.Float):
            if x._prec <= 53:
                return "m"
            best = x._prec if best is None else min(best, x._prec)
        elif isinstance(x, Expr):
            stack.extend(x.args)
            if isinstance(x.head, Expr):
                stack.append(x.head)
        elif isinstance(x, sp.Basic) and x.args:
            for a in sp.preorder_traversal(x):
                if isinstance(a, sp.Float):
                    if a._prec <= 53:
                        return "m"
                    best = a._prec if best is None else min(best, a._prec)
    return best


def round_to(x, prec):
    """Round every approximate number in ``x`` (MikoshiLang expression) to ``prec`` ("m" or bits)."""
    if prec is None:
        return x
    if isinstance(x, sp.Float):
        if prec == "m":
            return float(x)
        return x if x._prec <= prec else sp.Float(x, precision=prec)
    if isinstance(x, Expr):
        new = [round_to(a, prec) for a in x.args]
        if any(n is not o for n, o in zip(new, x.args)):
            return Expr(x.head, *new)
        return x
    return x
