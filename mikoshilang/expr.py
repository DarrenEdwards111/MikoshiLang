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
    if isinstance(a, Symbol):
        return a.name
    return repr(a)


_PRECEDENCE = {"Plus": 1, "Times": 2, "Power": 3}


def _format_expr(e: Expr, parent_prec: int = 0) -> str:
    prec = _PRECEDENCE.get(e.head, 10)

    if e.head == "Plus":
        parts = []
        for i, a in enumerate(e.args):
            s = _fmt(a, prec)
            if i > 0:
                # Check for negative coefficient
                if isinstance(a, Expr) and a.head == "Times" and len(a.args) >= 1 and isinstance(a.args[0], (int, float)) and a.args[0] < 0:
                    s = _fmt(a, prec)
                    parts.append(f" {s}" if s.startswith("-") else f" + {s}")
                else:
                    parts.append(f" + {s}")
            else:
                parts.append(s)
        result = "".join(parts)
        if parent_prec > prec:
            result = f"({result})"
        return result

    if e.head == "Times":
        parts = []
        for a in e.args:
            s = _fmt(a, prec)
            parts.append(s)
        result = "*".join(parts)
        if parent_prec > prec:
            result = f"({result})"
        return result

    if e.head == "Power":
        base = _fmt(e.args[0], prec + 1)
        exp = _fmt(e.args[1], 0)
        result = f"{base}^{exp}"
        if parent_prec > prec:
            result = f"({result})"
        return result

    if e.head == "List":
        inner = ", ".join(_fmt(a, 0) for a in e.args)
        return "{" + inner + "}"

    if e.head == "Matrix":
        rows = []
        for a in e.args:
            if isinstance(a, Expr) and a.head == "List":
                rows.append("{" + ", ".join(_fmt(x, 0) for x in a.args) + "}")
            else:
                rows.append(_fmt(a, 0))
        return "Matrix[" + ", ".join(rows) + "]"

    # Generic function notation
    inner = ", ".join(_fmt(a, 0) for a in e.args)
    return f"{e.head}[{inner}]"


def _fmt(a, parent_prec: int = 0) -> str:
    if isinstance(a, Expr):
        return _format_expr(a, parent_prec)
    return _format_atom(a)


def symbols(*names: str):
    """Create multiple symbols: x, y, z = symbols('x', 'y', 'z')"""
    if len(names) == 1 and " " in names[0]:
        names = tuple(names[0].split())
    syms = tuple(Symbol(n) for n in names)
    return syms if len(syms) > 1 else syms[0]
