"""Evaluation engine — recursive evaluation with simplification."""

from __future__ import annotations
from typing import Any, Dict, Set
from .expr import Expr, Symbol
from .rules import SIMPLIFICATION_RULES, replace

# Lazy load extended rules to avoid circular imports
_extended_loaded = False
_extended_rules = []
_rules_by_head = {}

def _ensure_extended():
    global _extended_loaded, _extended_rules
    if not _extended_loaded:
        from .extended import EXTENDED_RULES
        from .extended2 import EXTENDED2_RULES
        from .extended3 import EXTENDED3_RULES
        from .visualization import VISUALIZATION_RULES
        _extended_rules = (EXTENDED_RULES + EXTENDED2_RULES + EXTENDED3_RULES
                           + VISUALIZATION_RULES)
        for rule in _extended_rules:
            _rules_by_head.setdefault(rule.pattern.head, []).append(rule)
        _extended_loaded = True

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
        from .runtime import values
        return evaluate(values[expr.name], depth + 1) if expr.name in values else expr

    if not isinstance(expr, Expr):
        return expr


    from .runtime import special, UNHANDLED, values
    if isinstance(expr.head, str) and expr.head in values:
        target = evaluate(values[expr.head], depth + 1)
        if isinstance(target, Symbol):
            return evaluate(Expr(target.name, *expr.args), depth + 1)
        if isinstance(target, Expr) and target.head == "Function":
            return evaluate(Expr(target, *expr.args), depth + 1)
        raise ValueError("Assigned head is not callable")
    handled = special(expr, lambda x: evaluate(x, depth + 1))
    if handled is not UNHANDLED:
        return handled
    head = expr.head

    # Evaluate arguments (respecting Hold attributes)
    if head in _HOLD_ALL:
        new_args = expr.args
    elif head in _HOLD_FIRST:
        new_args = (expr.args[0],) + tuple(evaluate(a, depth + 1) for a in expr.args[1:]) if expr.args else ()
    else:
        new_args = tuple(evaluate(a, depth + 1) for a in expr.args)

    new_expr = Expr(head, *new_args)

    from .runtime import apply_definition
    defined = apply_definition(new_expr)
    if defined is not UNHANDLED:
        return evaluate(defined, depth + 1)

    # Apply simplification rules
    result = replace(new_expr, SIMPLIFICATION_RULES)

    # If simplification didn't change anything, try extended rules
    if isinstance(result, Expr) and result == new_expr:
        _ensure_extended()
        if _extended_rules:
            ext_result = replace(new_expr, _rules_by_head.get(head, []))
            if ext_result != new_expr:
                result = ext_result

    # If result changed, evaluate again
    if isinstance(result, Expr) and result != new_expr:
        result = evaluate(result, depth + 1)

    from .pattern import match
    if result == new_expr and not any(match(r.pattern, new_expr) is not None for r in _rules_by_head.get(head, [])):
        from .runtime import reject_quarantined
        reject_quarantined(head)
    return result
