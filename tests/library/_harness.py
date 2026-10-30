"""Shared differential-test harness for the reviewed library.

Each area module defines ``CASES``: a list of ``(source, expected)`` tuples
(optionally ``(source, expected, tol)``).  ``expected`` is one of

* ``str``      -- a Wolfram Language reference result, parsed and evaluated;
                  the result must be exactly equal (structurally, or with a
                  SymPy-verified zero difference and the same exactness);
* a number     -- a float/complex/mpmath/SymPy reference computed by an
                  independent library; the result is compared numerically
                  (relative tolerance ``tol``, default 1e-10);
* ``Raises``   -- the call must raise a clear error;
* ``Pred``     -- a predicate on the result (used for structural checks such
                  as figures, random draws or sign-ambiguous decompositions).
"""
import numbers
import re

import mpmath
import pytest
import sympy as sp

from mikoshilang.evaluate import evaluate
from mikoshilang.expr import Expr, Symbol
from mikoshilang.math_ops import to_sympy
from mikoshilang.parser import parse

HEAD = re.compile(r"\s*([A-Za-z][A-Za-z0-9]*)\[")


def ev(src):
    return evaluate(parse(src))


class Raises:
    def __init__(self, exc=(ValueError, NotImplementedError, TypeError), match=None):
        self.exc, self.match = exc, match

    def __repr__(self):
        return f"Raises({self.match!r})"


class Pred:
    def __init__(self, f, what="predicate"):
        self.f, self.what = f, what

    def __repr__(self):
        return self.what


def head_of(case):
    m = HEAD.match(case[0])
    return m.group(1) if m else None


def _has_float(x):
    if isinstance(x, bool):
        return False
    if isinstance(x, (float, complex)):
        return True
    if isinstance(x, Expr):
        return any(_has_float(a) for a in x.args)
    if isinstance(x, sp.Basic):
        return any(isinstance(a, sp.Float) for a in sp.preorder_traversal(x))
    return False


def to_complex(x):
    if isinstance(x, bool):
        raise AssertionError(f"expected a number, got {x!r}")
    if isinstance(x, numbers.Number):
        return complex(x)
    if isinstance(x, (mpmath.mpf, mpmath.mpc)):
        return complex(x)
    v = x if isinstance(x, sp.Basic) else to_sympy(x)
    v = sp.N(v, 30)
    if v.free_symbols or not v.is_number:
        raise AssertionError(f"expected a number, got {x!r}")
    return complex(v)


def same(got, want):
    if isinstance(want, Expr) and want.head == "List":
        return isinstance(got, Expr) and got.head == "List" and len(got.args) == len(want.args) and \
            all(same(g, w) for g, w in zip(got.args, want.args))
    if isinstance(want, Expr) and want.head == "Rule":
        return isinstance(got, Expr) and got.head == "Rule" and same(got.args[0], want.args[0]) and \
            same(got.args[1], want.args[1])
    if got == want:
        return _has_float(got) == _has_float(want)
    if isinstance(want, (bool, str)) or isinstance(got, (bool, str)):
        return False
    if _has_float(got) != _has_float(want):
        return False
    try:
        g, w = to_sympy(got), to_sympy(want)
        if isinstance(g, list) or isinstance(w, list):
            return False
        return sp.simplify(g - w) == 0
    except Exception:
        return False


def close(got, want, tol):
    if isinstance(want, (list, tuple)):
        assert isinstance(got, Expr) and got.head == "List" and len(got.args) == len(want), (got, want)
        for g, w in zip(got.args, want):
            close(g, w, tol)
        return
    g, w = to_complex(got), to_complex(want)
    assert abs(g - w) <= tol * max(1.0, abs(w)), f"{got!r} != {want!r}"


def check(case):
    src, want = case[0], case[1]
    tol = case[2] if len(case) > 2 else 1e-10
    if isinstance(want, Raises):
        with pytest.raises(want.exc, match=want.match):
            ev(src)
        return
    got = ev(src)
    if isinstance(want, Pred):
        assert want.f(got), f"{src} => {got!r} fails {want.what}"
    elif isinstance(want, str):
        ref = ev(want)
        assert same(got, ref), f"{src} => {got!r}, expected {ref!r}"
    elif isinstance(want, bool):
        assert got is want, f"{src} => {got!r}, expected {want!r}"
    else:
        close(got, want, tol)


def ids(cases):
    return [c[0][:80] for c in cases]
