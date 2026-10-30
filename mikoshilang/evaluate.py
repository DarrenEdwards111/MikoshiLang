"""Evaluation engine — recursive evaluation with simplification."""

from __future__ import annotations
from typing import Any, Dict, Set
from .expr import Expr, Symbol
from .rules import SIMPLIFICATION_RULES, replace

# The reviewed function library (mikoshilang.library) replaces the legacy
# extended/extended2/extended3/visualization rule tables, which are no longer
# loaded: every exposed function is registered there with an audit status.
_rules_by_head = {}   # kept for backward compatibility; always empty


def _ensure_extended():
    """Backward-compatible no-op: the legacy rule tables are no longer loaded."""
    from . import library  # noqa: F401  (registers the reviewed library)


# Hold attributes
_HOLD_ALL: Set[str] = {"Trace", "Assuming"}   # library heads that receive unevaluated arguments
_HOLD_FIRST: Set[str] = set()

_MAX_DEPTH = 1024  # matches Wolfram's default $RecursionLimit
_memo: Dict[int, Any] = {}

import sys as _sys
from contextvars import ContextVar as _ContextVar

# Evaluation tracing for Trace[expr]: a stack of frames (None when not tracing). Each frame collects
# ("nested", chain) for argument evaluations, ("form", expr) for the expression after its arguments
# were evaluated, and ("splice", chain) for re-evaluations of a rewritten expression.
_TRACE = _ContextVar("mikoshilang_trace", default=None)
_TRACE_ARG = _ContextVar("mikoshilang_trace_arg", default=False)
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
    if depth > 0 and _TRACE.get() is not None:
        return _traced(expr, depth)
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


def _traced(expr, depth):
    stack = _TRACE.get()
    is_arg = _TRACE_ARG.get()
    tok = _TRACE_ARG.set(False)
    frame = []
    stack.append(frame)
    try:
        r = _evaluate(expr, depth)
    finally:
        stack.pop()
        _TRACE_ARG.reset(tok)
    chain = _trace_chain(expr, frame, r, start=bool(stack) and not is_arg and len(stack) > 1)
    if stack:
        if not is_arg:
            stack[-1].append(("splice", chain))
        elif len(chain) > 1 or any(isinstance(c, list) for c in chain):
            stack[-1].append(("nested", chain))
    return r


def _trace_chain(expr, items, r, start=False):
    out = []

    def add(x):
        if not (out and not isinstance(out[-1], list) and _same(out[-1], x)):
            out.append(x)

    if start or not any(k == "nested" for k, _ in items):
        add(expr)
    for k, c in items:
        if k == "nested":
            out.append(c)
        elif k == "form":
            add(c)
        else:
            for el in c:
                if isinstance(el, list):
                    out.append(el)
                else:
                    add(el)
    add(r)
    return out


def _same(a, b):
    try:
        return type(a) is type(b) and a == b
    except Exception:
        return False


def trace_evaluation(expr):
    """Evaluate expr while recording the Wolfram-style evaluation chain; returns (chain, value)."""
    stack = [[]]
    tok = _TRACE.set(stack)
    atok = _TRACE_ARG.set(False)
    try:
        r = evaluate(expr, 1)
    finally:
        _TRACE.reset(tok)
        _TRACE_ARG.reset(atok)
    top = stack[0]
    chain = top[0][1] if top else [expr, r]
    if len(chain) == 1:       # nothing happened: Wolfram returns {}
        chain = []
    return chain, r


def _ev_arg(x, depth):
    if _TRACE.get() is None:
        return evaluate(x, depth)
    tok = _TRACE_ARG.set(True)
    try:
        return evaluate(x, depth)
    finally:
        _TRACE_ARG.reset(tok)


def _evaluate(expr, depth: int) -> Any:
    if depth > _MAX_DEPTH:
        raise RecursionError(f"Evaluation depth exceeded {_MAX_DEPTH}")

    # Atoms evaluate to themselves
    if isinstance(expr, (int, float, str, bool)) or expr is None:
        return expr
    from . import kernel
    if isinstance(expr, Symbol):
        own = kernel.state().own
        if expr.name in own:
            return evaluate(own[expr.name], depth + 1)
        if expr.name == "Null":
            return None
        if expr.name.startswith("$"):
            from .library.wl_numerics import SYSTEM_CONSTANTS
            if expr.name in SYSTEM_CONSTANTS:
                return SYSTEM_CONSTANTS[expr.name]                     # the symbol Null is the null value (what x; evaluates to)
        return True if expr.name == "$Assumptions" else expr     # Wolfram default $Assumptions = True
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
    eva = lambda x: _ev_arg(x, depth + 1)  # noqa: E731
    if head in _HOLD_ALL or "HoldAll" in attrs:
        new_args = list(expr.args)
    elif head in _HOLD_FIRST or "HoldFirst" in attrs:
        new_args = list(expr.args[:1]) + [eva(a) for a in expr.args[1:]]
    elif "HoldRest" in attrs:
        new_args = [eva(a) for a in expr.args[:1]] + list(expr.args[1:])
    else:
        new_args = [eva(a) for a in expr.args]

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
    if _TRACE.get() is not None and _TRACE.get():
        _TRACE.get()[-1].append(("form", new_expr))

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

    # 8. Kernel built-ins (an operator form such as Select[crit] stays unevaluated until applied)
    from .kernel_lists import OPERATOR_FORMS
    op = OPERATOR_FORMS.get(head) if isinstance(head, str) else None
    if op is not None and len(new_args) == op[0]:
        return new_expr
    r = kernel.apply_builtin(new_expr, ev)
    if r is not kernel.UNHANDLED:
        return r

    if not isinstance(head, str):
        # 8b. Associations as functions, Key[k][assoc], compositions and operator forms (Select[crit][list])
        if isinstance(head, Expr) and isinstance(head.head, str):
            from .kernel_lists import apply_compound_head
            r = apply_compound_head(head, new_args, ev)
            if r is not kernel.UNHANDLED:
                return r
        return new_expr

    # 9. Reviewed function library
    result = replace(new_expr, SIMPLIFICATION_RULES)
    if isinstance(result, Expr) and result == new_expr:
        from . import library
        lib = library.dispatch(head, new_expr.args)
        if lib is library.UNHANDLED:
            if head not in library.REGISTRY:
                from .runtime import reject_quarantined
                reject_quarantined(head)
            return new_expr
        result = lib

    if isinstance(result, Expr) and result != new_expr:
        result = evaluate(result, depth + 1)
    return result
