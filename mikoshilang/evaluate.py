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

_MAX_DEPTH = 1024  # matches Wolfram's default $RecursionLimit
_memo: Dict[int, Any] = {}

import sys as _sys
if _sys.getrecursionlimit() < 20000:
    _sys.setrecursionlimit(20000)


def set_hold_all(head: str):
    _HOLD_ALL.add(head)


def set_hold_first(head: str):
    _HOLD_FIRST.add(head)


def clear_memo():
    _memo.clear()


def evaluate(expr, depth: int = 0) -> Any:
    """Evaluate an expression with Wolfram-style semantics.

    Order: head, held special forms, arguments (respecting Hold* attributes),
    Sequence splicing, Flat/Orderless/Listable, user DownValues (most specific
    first), kernel built-ins, then the legacy rule library.
    """
    if depth == 0:
        from .kernel import ReturnSignal, BreakSignal, ContinueSignal, ThrowSignal
        try:
            return _evaluate(expr, 0)
        except ReturnSignal as r:
            return r.value
        except (BreakSignal, ContinueSignal):
            raise ValueError("Break[] or Continue[] used outside a loop") from None
        except ThrowSignal as t:
            raise ValueError(f"Uncaught Throw[{t.value!r}]") from None
    return _evaluate(expr, depth)


def _evaluate(expr, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        raise RecursionError(f"Evaluation depth exceeded {_MAX_DEPTH}")

    # Atoms evaluate to themselves
    if isinstance(expr, (int, float, str, bool)) or expr is None:
        return expr
    from . import kernel
    if isinstance(expr, Symbol):
        own = kernel.state().own
        return evaluate(own[expr.name], depth + 1) if expr.name in own else expr
    if not isinstance(expr, Expr):
        return expr

    ev = lambda x: evaluate(x, depth + 1)  # noqa: E731
    head = expr.head

    # 1. Compound heads evaluate first: f[a][b], (#&)[x], Derivative[1][f][x]
    if isinstance(head, Expr):
        new_head = ev(head)
        if isinstance(new_head, Symbol):
            head = new_head.name
        elif isinstance(new_head, Expr):
            head = new_head
        else:
            raise ValueError(f"{new_head!r} cannot be used as a function head")
        if head is not expr.head:
            expr = Expr(head, *expr.args)

    # 2. A symbol head with an OwnValue: square = #^2 &; square[3]
    own = kernel.state().own
    if isinstance(head, str) and head in own:
        target = ev(own[head])
        if isinstance(target, Symbol):
            return ev(Expr(target.name, *expr.args))
        if isinstance(target, Expr):
            return ev(Expr(target, *expr.args))
        raise ValueError("Assigned head is not callable")

    # 3. Special forms that see unevaluated arguments
    handled = kernel.held(expr, ev)
    if handled is not kernel.UNHANDLED:
        return handled
    from .runtime import special
    handled = special(expr, ev)
    if handled is not kernel.UNHANDLED:
        return handled

    # 4. Arguments, respecting Hold attributes
    attrs = kernel.attributes(head) if isinstance(head, str) else set()
    if head in _HOLD_ALL or "HoldAll" in attrs:
        new_args = list(expr.args)
    elif head in _HOLD_FIRST or "HoldFirst" in attrs:
        new_args = list(expr.args[:1]) + [ev(a) for a in expr.args[1:]]
    elif "HoldRest" in attrs:
        new_args = [ev(a) for a in expr.args[:1]] + list(expr.args[1:])
    else:
        new_args = [ev(a) for a in expr.args]

    # 5. Sequence splicing, Flat, Orderless, Listable
    if "SequenceHold" not in attrs and any(isinstance(a, Expr) and a.head == "Sequence" for a in new_args):
        spliced = []
        for a in new_args:
            if isinstance(a, Expr) and a.head == "Sequence":
                spliced.extend(a.args)
            else:
                spliced.append(a)
        new_args = spliced
    uattrs = kernel.user_attributes(head)
    if "Flat" in uattrs:
        flat = []
        for a in new_args:
            if isinstance(a, Expr) and a.head == head:
                flat.extend(a.args)
            else:
                flat.append(a)
        new_args = flat
    if "Orderless" in uattrs:
        new_args = sorted(new_args, key=kernel.order_key)
    if "OneIdentity" in uattrs and "Flat" in uattrs and len(new_args) == 1:
        pass
    if "Listable" in attrs and any(kernel.is_list(a) for a in new_args):
        lengths = {len(a.args) for a in new_args if kernel.is_list(a)}
        if len(lengths) != 1:
            raise ValueError(f"{head}: cannot thread over lists of unequal length")
        n = lengths.pop()
        return ev(Expr("List", *(Expr(head, *(a.args[i] if kernel.is_list(a) else a for a in new_args))
                                 for i in range(n))))

    new_expr = Expr(head, *new_args)

    # 6. Pure functions and derivatives as heads
    if isinstance(head, Expr):
        if head.head == "Function":
            try:
                return kernel.apply_function(head, new_args, ev)
            except kernel.ReturnSignal as r:
                return r.value
        if isinstance(head.head, Expr) and head.head.head == "Derivative":
            r = kernel.apply_derivative(head, new_args, ev)
            if r is not kernel.UNHANDLED:
                return r

    # 7. User definitions (DownValues, most specific first)
    defined = kernel.apply_downvalues(new_expr, ev)
    if defined is not kernel.UNHANDLED:
        try:
            return ev(defined)
        except kernel.ReturnSignal as r:
            return r.value

    # 8. Kernel built-ins
    r = kernel.apply_builtin(new_expr, ev)
    if r is not kernel.UNHANDLED:
        return r

    if not isinstance(head, str):
        return new_expr

    # 9. Legacy rule library
    result = replace(new_expr, SIMPLIFICATION_RULES)
    if isinstance(result, Expr) and result == new_expr:
        _ensure_extended()
        if _extended_rules:
            ext_result = replace(new_expr, _rules_by_head.get(head, []))
            if ext_result != new_expr:
                result = ext_result

    if isinstance(result, Expr) and result != new_expr:
        result = evaluate(result, depth + 1)

    from .pattern import match
    if result == new_expr and not any(match(r.pattern, new_expr) is not None for r in _rules_by_head.get(head, [])):
        from .runtime import reject_quarantined
        reject_quarantined(head)
    return result
