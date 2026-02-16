"""Evaluation engine — recursive evaluation with simplification."""

from __future__ import annotations
from typing import Any, Dict, Set
from .expr import Expr, Symbol
from .rules import SIMPLIFICATION_RULES, replace_all

# Hold attributes
_HOLD_ALL: Set[str] = set()
_HOLD_FIRST: Set[str] = set()

_MAX_DEPTH = 256
_memo: Dict[int, Any] = {}


def set_hold_all(head: str):
    _HOLD_ALL.add(head)


def set_hold_first(head: str):
    _HOLD_FIRST.add(head)


def clear_memo():
    _memo.clear()


def evaluate(expr, depth: int = 0) -> Any:
    """Recursively evaluate an expression, applying simplification rules."""
    if depth > _MAX_DEPTH:
        raise RecursionError(f"Evaluation depth exceeded {_MAX_DEPTH}")

    # Atoms evaluate to themselves
    if isinstance(expr, (int, float, str, bool)):
        return expr
    if isinstance(expr, Symbol):
        return expr

    if not isinstance(expr, Expr):
        return expr

    # Memoization
    key = hash(expr)
    if key in _memo:
        cached = _memo[key]
        if cached == expr:
            return cached

    head = expr.head

    # Evaluate arguments (respecting Hold attributes)
    if head in _HOLD_ALL:
        new_args = expr.args
    elif head in _HOLD_FIRST:
        new_args = (expr.args[0],) + tuple(evaluate(a, depth + 1) for a in expr.args[1:]) if expr.args else ()
    else:
        new_args = tuple(evaluate(a, depth + 1) for a in expr.args)

    new_expr = Expr(head, *new_args)

    # Apply simplification rules
    result = replace_all(new_expr, SIMPLIFICATION_RULES)

    # If result changed, evaluate again
    if isinstance(result, Expr) and result != new_expr:
        result = evaluate(result, depth + 1)

    _memo[key] = result
    return result
