"""Registry, dispatch and argument helpers for the reviewed function library.

Every function exposed by the library is registered here with an audit status:

* ``verified`` - the legacy behaviour was checked against an independent
  reference (SymPy, mpmath, SciPy or Wolfram reference values) and kept;
* ``fixed``    - the legacy implementation gave wrong answers and was replaced;
* ``error``    - the name is reserved but raises a clear NotImplementedError
  (the legacy behaviour was wrong and no correct implementation is offered);
* ``kernel``   - the head is implemented by the core kernel; the legacy
  fallback was removed;
* ``quarantined`` - the legacy rule was removed and the name now raises the
  quarantine error;
* ``promoted`` - a previously quarantined name with a new, tested implementation;
* ``added``    - a Wolfram built-in that MikoshiLang did not have, implemented in a coverage batch
  (see docs/wolfram_coverage_roadmap.md) with differential tests.

Library functions receive evaluated arguments and return MikoshiLang
expressions.  They raise :class:`Unevaluated` to leave a call unevaluated
(Wolfram's behaviour for symbolic arguments such as ``Fibonacci[n]``) and
``ValueError``/``NotImplementedError`` with a clear message otherwise.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import sympy as sp

from ..expr import Expr, Symbol

UNHANDLED = object()


class Unevaluated(Exception):
    """Raised by an implementation to leave the call unevaluated."""


@dataclass
class Entry:
    name: str
    area: str
    status: str
    notes: str
    impls: List[tuple] = field(default_factory=list)   # (fn, min_args, max_args or None)
    listable: bool = False
    hold: bool = False                                    # receives unevaluated args


REGISTRY: Dict[str, Entry] = {}
STATUS: Dict[str, tuple] = {}     # every audited name -> (area, status, notes)
_VALID = {"verified", "fixed", "error", "kernel", "quarantined", "promoted", "added"}


def _arity(fn):
    params = list(inspect.signature(fn).parameters.values())
    lo = sum(1 for p in params if p.kind == p.POSITIONAL_OR_KEYWORD and p.default is p.empty)
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return lo, None
    return lo, sum(1 for p in params if p.kind == p.POSITIONAL_OR_KEYWORD)


def record(name, area, status, notes=""):
    if status not in _VALID:
        raise ValueError(status)
    STATUS[name] = (area, status, notes)


def fn(*names, area, status, notes="", listable=False, hold=False):
    """Register an implementation for one or more heads (aliases share it)."""
    def deco(f):
        lo, hi = _arity(f)
        for n in names:
            e = REGISTRY.get(n)
            if e is None:
                e = REGISTRY[n] = Entry(n, area, status, notes, listable=listable, hold=hold)
            e.impls.append((f, lo, hi))
            if n not in STATUS or STATUS[n][1] != status:
                record(n, area, status, notes)
        return f
    return deco


ERRORS: Dict[str, str] = {}


def error(name, area, message, notes=None):
    ERRORS[name] = message
    record(name, area, "error", notes or message)


def _arity_text(e):
    parts = []
    for _, lo, hi in e.impls:
        parts.append(f"{lo}+" if hi is None else (str(lo) if lo == hi else f"{lo}-{hi}"))
    return " or ".join(sorted(set(parts)))


def dispatch(head, args):
    """Apply the library implementation of ``head`` to evaluated ``args``."""
    msg = ERRORS.get(head)
    if msg is not None:
        raise NotImplementedError(f"{head}: {msg}")
    e = REGISTRY.get(head)
    if e is None:
        return UNHANDLED
    if e.listable and any(is_list(a) for a in args):
        lengths = {len(a.args) for a in args if is_list(a)}
        if len(lengths) != 1:
            raise ValueError(f"{head}: cannot thread over lists of unequal length")
        n = lengths.pop()
        return Expr("List", *(Expr(head, *(a.args[i] if is_list(a) else a for a in args)) for i in range(n)))
    n = len(args)
    for f, lo, hi in e.impls:
        if n >= lo and (hi is None or n <= hi):
            try:
                return f(*args)
            except Unevaluated:
                return UNHANDLED
    raise ValueError(f"{head} called with {n} argument(s); expected {_arity_text(e)}")


# ─────────────────────────── conversion helpers ───────────────────────────

def is_list(x) -> bool:
    return isinstance(x, Expr) and x.head == "List"


def S(x):
    """MikoshiLang -> SymPy (lists become Python lists)."""
    from ..math_ops import to_sympy
    if isinstance(x, (sp.Basic, sp.MatrixBase)):
        return x
    return to_sympy(x)


def R(x):
    """SymPy/Python -> MikoshiLang."""
    from ..math_ops import from_sympy

    if isinstance(x, bool):
        return x
    if isinstance(x, (list, tuple)):
        return Expr("List", *(R(v) for v in x))
    if isinstance(x, dict):
        raise TypeError("dict results must be converted explicitly")
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return R(x.tolist())
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.complexfloating):
            x = complex(x)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(x, complex):
        if x.imag == 0:
            return float(x.real)
        return from_sympy(sp.Float(x.real) + sp.Float(x.imag) * sp.I)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return x
    return from_sympy(sp.sympify(x))


def L(*items):
    return Expr("List", *items)


def rule(lhs, rhs):
    return Expr("Rule", lhs, rhs)


def has_float(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (float, complex)):
        return True
    if isinstance(x, sp.Float):
        return True
    if isinstance(x, sp.Basic):
        return any(isinstance(a, sp.Float) for a in sp.preorder_traversal(x))
    if isinstance(x, Expr):
        return any(has_float(a) for a in x.args)
    return False


def numeric_out(value, *inputs):
    """Evaluate ``value`` numerically when any input was inexact, else return it exactly.  The result has the
    lowest input precision: machine if any input is a machine number, else the smallest bignum precision."""
    if any(has_float(i) for i in inputs):
        from ..precision import input_precision
        p = input_precision(inputs)
        if p == "m" or p is None:
            v = sp.sympify(value).evalf(15)
            return R(v)
        import mpmath
        dps = mpmath.libmp.prec_to_dps(p)
        v = sp.sympify(value).evalf(dps)
        from ..precision import round_to
        return round_to(R(v), p)
    return R(value)


def is_numeric_value(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float, complex)):
        return True
    try:
        v = S(x)
    except Exception:
        return False
    return isinstance(v, sp.Basic) and v.is_number


def int_arg(x, fname, what="argument"):
    """Exact integer argument; symbolic -> Unevaluated; non-integer number -> ValueError."""
    if isinstance(x, bool):
        raise ValueError(f"{fname}: {what} {x!r} is not an integer")
    if isinstance(x, int):
        return x
    if isinstance(x, sp.Integer):
        return int(x)
    if is_numeric_value(x):
        raise ValueError(f"{fname}: {what} {x!r} is not an integer")
    raise Unevaluated()


def nonneg_int(x, fname, what="argument"):
    n = int_arg(x, fname, what)
    if n < 0:
        raise ValueError(f"{fname}: {what} {n} must be a non-negative integer")
    return n


def pos_int(x, fname, what="argument"):
    n = int_arg(x, fname, what)
    if n <= 0:
        raise ValueError(f"{fname}: {what} {n} must be a positive integer")
    return n


def number(x, fname, what="argument", real=True):
    """SymPy number for a numeric argument; symbolic -> Unevaluated."""
    if isinstance(x, bool):
        raise ValueError(f"{fname}: {what} must be a number")
    if isinstance(x, (int, float)):
        return sp.sympify(x)
    try:
        v = S(x)
    except Exception:
        raise ValueError(f"{fname}: {what} {x!r} must be a number")
    if not isinstance(v, sp.Basic):
        raise ValueError(f"{fname}: {what} {x!r} must be a number")
    if v.free_symbols:
        raise Unevaluated()
    if not v.is_number:
        raise ValueError(f"{fname}: {what} {x!r} must be a number")
    if real and v.is_real is False:
        raise ValueError(f"{fname}: {what} {x!r} must be a real number")
    return v


def sym(x, fname):
    """SymPy expression for an arbitrary symbolic argument."""
    if isinstance(x, str):
        raise TypeError(f"{fname}: strings are not symbolic expressions; parse source explicitly")
    try:
        v = S(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{fname}: cannot interpret {x!r} as a mathematical expression ({exc})")
    if isinstance(v, list):
        raise ValueError(f"{fname}: expected an expression, got a list")
    return v


def variable(x, fname):
    if isinstance(x, Symbol):
        return sp.Symbol(x.name)
    raise ValueError(f"{fname}: {x!r} is not a valid variable")


def list_arg(x, fname, what="argument", min_len=0):
    if is_list(x):
        if len(x.args) < min_len:
            raise ValueError(f"{fname}: {what} needs at least {min_len} element(s)")
        return list(x.args)
    if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
        raise Unevaluated()
    raise ValueError(f"{fname}: {what} {x!r} must be a list")


def num_list(x, fname, what="argument", min_len=1):
    items = list_arg(x, fname, what, min_len)
    out = []
    for v in items:
        s = number(v, fname, "list element")
        out.append(s)
    return out


def matrix_arg(x, fname, square=False):
    rows = list_arg(x, fname, "matrix")
    if not rows or not all(is_list(r) for r in rows):
        raise ValueError(f"{fname}: argument must be a matrix (a list of equal-length lists)")
    width = len(rows[0].args)
    if width == 0 or any(len(r.args) != width for r in rows):
        raise ValueError(f"{fname}: argument must be a rectangular matrix")
    m = sp.Matrix([[S(v) for v in r.args] for r in rows])
    if square and m.rows != m.cols:
        raise ValueError(f"{fname}: matrix must be square")
    return m


def string_arg(x, fname, what="argument"):
    if isinstance(x, str):
        return x
    if isinstance(x, (Symbol, Expr)):
        raise Unevaluated()
    raise ValueError(f"{fname}: {what} {x!r} must be a string")


def to_float(v) -> float:
    return float(sp.sympify(v).evalf(17))


def mat_out(m):
    return R([[m[i, j] for j in range(m.cols)] for i in range(m.rows)])
