"""Expression system — the foundation of MikoshiLang.

Everything is an expression with a head and arguments, or an atom.
"""

from __future__ import annotations
from typing import Any, Union

Atom = Union[int, float, str, bool, "Symbol"]


class Symbol:
    """A symbolic variable like x, y, Pi."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.name == other.name
        return NotImplemented

    def __hash__(self):
        return hash(("Symbol", self.name))

    # Arithmetic — produce Expr trees
    def __add__(self, other):
        return Expr("Plus", self, _wrap(other))

    def __radd__(self, other):
        return Expr("Plus", _wrap(other), self)

    def __mul__(self, other):
        return Expr("Times", self, _wrap(other))

    def __rmul__(self, other):
        return Expr("Times", _wrap(other), self)

    def __sub__(self, other):
        return Expr("Plus", self, Expr("Times", -1, _wrap(other)))

    def __rsub__(self, other):
        return Expr("Plus", _wrap(other), Expr("Times", -1, self))

    def __truediv__(self, other):
        return Expr("Times", self, Expr("Power", _wrap(other), -1))

    def __rtruediv__(self, other):
        return Expr("Times", _wrap(other), Expr("Power", self, -1))

    def __pow__(self, other):
        return Expr("Power", self, _wrap(other))

    def __rpow__(self, other):
        return Expr("Power", _wrap(other), self)

    def __neg__(self):
        return Expr("Times", -1, self)

    def __pos__(self):
        return self


class Expr:
    """A composite expression with a head and arguments."""

    __slots__ = ("head", "args")

    def __init__(self, head: str, *args: Any):
        self.head = head
        self.args = tuple(args)

    def __eq__(self, other):
        if isinstance(other, Expr):
            return self.head == other.head and self.args == other.args
        return NotImplemented

    def __hash__(self):
        return hash((self.head, self.args))

    def __repr__(self):
        return _format_expr(self)

    # Arithmetic
    def __add__(self, other):
        return Expr("Plus", self, _wrap(other))

    def __radd__(self, other):
        return Expr("Plus", _wrap(other), self)

    def __mul__(self, other):
        return Expr("Times", self, _wrap(other))

    def __rmul__(self, other):
        return Expr("Times", _wrap(other), self)

    def __sub__(self, other):
        return Expr("Plus", self, Expr("Times", -1, _wrap(other)))

    def __rsub__(self, other):
        return Expr("Plus", _wrap(other), Expr("Times", -1, self))

    def __truediv__(self, other):
        return Expr("Times", self, Expr("Power", _wrap(other), -1))

    def __rtruediv__(self, other):
        return Expr("Times", _wrap(other), Expr("Power", self, -1))

    def __pow__(self, other):
        return Expr("Power", self, _wrap(other))

    def __rpow__(self, other):
        return Expr("Power", _wrap(other), self)

    def __neg__(self):
        return Expr("Times", -1, self)

    def __pos__(self):
        return self


def _wrap(x):
    """Ensure x is an Expr or Symbol; pass through atoms."""
    if isinstance(x, (Expr, Symbol, int, float, str, bool)):
        return x
    return x


def _format_atom(a):
    from .formatting import input_form
    return input_form(a)


def _format_expr(e: Expr, parent_prec: int = 0) -> str:
    from .formatting import input_form
    return input_form(e)


def _fmt(a, parent_prec: int = 0) -> str:
    from .formatting import input_form
    return input_form(a)


def symbols(*names: str):
    """Create multiple symbols: x, y, z = symbols('x', 'y', 'z')"""
    if len(names) == 1 and " " in names[0]:
        names = tuple(names[0].split())
    syms = tuple(Symbol(n) for n in names)
    return syms if len(syms) > 1 else syms[0]
