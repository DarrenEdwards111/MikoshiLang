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

Cases are grouped by head (``head_of``):

* ``F[...]``         -- the outermost head ``F``;
* a bare symbol or ``$``-variable (``"Pi"``, ``"$MachinePrecision"``) -- that
                        symbol; the case is a value check (``str`` / number) or
                        a type check (``Pred``);
* ``Via("G", src)``  -- a wrapper case such as ``ToString[G[...]]`` or
                        ``N[G[...]]`` that counts towards the inner head ``G``.
                        It counts only if the expected value depends on ``G``'s
                        own output: ``check`` re-evaluates ``src`` with ``G``
                        replaced by an inert probe head and fails the case if the
                        expectation still holds;
* anything else (``x = 3`` setup lines) -- no head.
"""
import numbers
import re

import mpmath
import pytest
import sympy as sp

from mikoshilang.evaluate import evaluate
from mikoshilang.kernel import SessionState, _state, state
from mikoshilang.expr import Expr, Symbol
from mikoshilang.math_ops import to_sympy
from mikoshilang.parser import parse

HEAD = re.compile(r"\s*([A-Za-z][A-Za-z0-9]*)\[")
SYMBOL = re.compile(r"\s*(\$?[A-Za-z][A-Za-z0-9]*)\s*$")
PROBE = "ViaProbe"


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


class Via(str):
    """Source of a wrapper case counted towards the inner head ``inner`` (see the module docstring)."""

    def __new__(cls, inner, src):
        if not re.search(r"(?<![A-Za-z0-9$])" + re.escape(inner) + r"\[", src):
            raise ValueError(f"Via: {inner}[...] does not occur in {src!r}")
        if HEAD.match(src) and HEAD.match(src).group(1) == inner:
            raise ValueError(f"Via: {src!r} already has head {inner}; use a plain case")
        s = str.__new__(cls, src)
        s.inner = inner
        return s

    def __getnewargs__(self):
        return (self.inner, str(self))

    def probe(self):
        """``src`` with every ``inner[`` replaced by the inert ``ViaProbe[``."""
        return re.sub(r"(?<![A-Za-z0-9$])" + re.escape(self.inner) + r"\[", PROBE + "[", str(self))


def head_of(case):
    src = case[0]
    if isinstance(src, Via):
        return src.inner
    m = HEAD.match(src) or SYMBOL.match(src)
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
    _check(case[0], case)
    if isinstance(case[0], Via) and not isinstance(case[1], Raises):
        tok = _state.set(_snapshot(state()))          # the probe must not change the session later cases see
        try:
            _check(case[0].probe(), case)
        except Exception:
            return
        finally:
            _state.reset(tok)
        raise AssertionError(f"{case[0]}: the expected value does not depend on {case[0].inner}'s own output "
                             f"(it also holds with {case[0].inner} replaced by {PROBE})")


def _snapshot(st):
    """A copy of session state ``st`` whose definitions can change without affecting ``st``."""
    new = SessionState()
    new.own = dict(st.own)
    new.exact = dict(st.exact)
    new.down = {k: list(v) for k, v in st.down.items()}
    new.attrs = {k: set(v) for k, v in st.attrs.items()}
    new.counter, new.escape_seq = st.counter, st.escape_seq
    new.temporaries = set(st.temporaries)
    new.ext = {k: dict(v) for k, v in st.ext.items()}
    return new


def _check(src, case):
    want = case[1]
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
