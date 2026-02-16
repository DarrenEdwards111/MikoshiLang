"""Pattern matching — Wolfram-style patterns for MikoshiLang."""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple, Union
from .expr import Expr, Symbol


class Blank:
    """Matches any single expression. Optionally binds a name and/or constrains head type."""
    def __init__(self, name: Optional[str] = None, head: Optional[str] = None):
        self.name = name
        self.head = head

    def __repr__(self):
        n = self.name or ""
        h = f"_{self.head}" if self.head else "_"
        return f"{n}{h}"


class BlankSequence:
    """Matches one or more expressions."""
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __repr__(self):
        return f"{self.name or ''}__"


class BlankNullSequence:
    """Matches zero or more expressions."""
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __repr__(self):
        return f"{self.name or ''}___"


class Pattern:
    """A pattern expression — an Expr-like structure with Blank slots."""
    def __init__(self, head: str, *args):
        self.head = head
        self.args = tuple(args)

    def __repr__(self):
        inner = ", ".join(repr(a) for a in self.args)
        return f"{self.head}[{inner}]"


class Condition:
    """Pattern with a condition: pattern /; test."""
    def __init__(self, pattern, test_fn: Callable[[Dict[str, Any]], bool]):
        self.pattern = pattern
        self.test_fn = test_fn

    def __repr__(self):
        return f"{self.pattern!r} /; <condition>"


def match(pattern, expr, bindings: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Try to match expr against pattern. Returns bindings dict or None."""
    if bindings is None:
        bindings = {}

    if isinstance(pattern, Condition):
        result = match(pattern.pattern, expr, bindings.copy())
        if result is not None and pattern.test_fn(result):
            return result
        return None

    if isinstance(pattern, Blank):
        # Check head constraint
        if pattern.head is not None:
            if pattern.head == "Integer" and not isinstance(expr, int):
                return None
            elif pattern.head == "Real" and not isinstance(expr, float):
                return None
            elif pattern.head == "String" and not isinstance(expr, str):
                return None
            elif pattern.head == "Symbol" and not isinstance(expr, Symbol):
                return None
            elif pattern.head not in ("Integer", "Real", "String", "Symbol"):
                if not (isinstance(expr, Expr) and expr.head == pattern.head):
                    return None
        if pattern.name is not None:
            if pattern.name in bindings:
                if bindings[pattern.name] != expr:
                    return None
            else:
                bindings[pattern.name] = expr
        return bindings

    if isinstance(pattern, (int, float, str, bool)):
        if pattern == expr:
            return bindings
        return None

    if isinstance(pattern, Symbol):
        if isinstance(expr, Symbol) and pattern.name == expr.name:
            return bindings
        return None

    if isinstance(pattern, Pattern):
        if not isinstance(expr, Expr):
            return None
        if pattern.head != expr.head:
            return None
        return _match_args(pattern.args, expr.args, bindings)

    if isinstance(pattern, Expr):
        if not isinstance(expr, Expr):
            return None
        if pattern.head != expr.head:
            return None
        return _match_args(pattern.args, expr.args, bindings)

    return None


def _match_args(pat_args: Tuple, expr_args: Tuple, bindings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Match argument lists, handling BlankSequence and BlankNullSequence."""
    return _match_args_rec(pat_args, 0, expr_args, 0, bindings)


def _match_args_rec(pat_args, pi, expr_args, ei, bindings):
    # Both exhausted
    if pi == len(pat_args) and ei == len(expr_args):
        return bindings

    # Pattern exhausted but exprs remain
    if pi == len(pat_args):
        return None

    pat = pat_args[pi]

    # BlankNullSequence — match 0 or more
    if isinstance(pat, BlankNullSequence):
        for end in range(ei, len(expr_args) + 1):
            seq = expr_args[ei:end]
            new_bindings = bindings.copy()
            if pat.name is not None:
                if pat.name in new_bindings:
                    if new_bindings[pat.name] != seq:
                        continue
                else:
                    new_bindings[pat.name] = seq
            result = _match_args_rec(pat_args, pi + 1, expr_args, end, new_bindings)
            if result is not None:
                return result
        return None

    # BlankSequence — match 1 or more
    if isinstance(pat, BlankSequence):
        for end in range(ei + 1, len(expr_args) + 1):
            seq = expr_args[ei:end]
            new_bindings = bindings.copy()
            if pat.name is not None:
                if pat.name in new_bindings:
                    if new_bindings[pat.name] != seq:
                        continue
                else:
                    new_bindings[pat.name] = seq
            result = _match_args_rec(pat_args, pi + 1, expr_args, end, new_bindings)
            if result is not None:
                return result
        return None

    # Expr args exhausted but pattern remains
    if ei == len(expr_args):
        return None

    # Regular match
    result = match(pat, expr_args[ei], bindings.copy())
    if result is not None:
        return _match_args_rec(pat_args, pi + 1, expr_args, ei + 1, result)
    return None


def parse_pattern(s: str):
    """Parse a simple pattern string like 'x_', 'x_Integer', '_', '__', '___'."""
    s = s.strip()
    if s == "___":
        return BlankNullSequence()
    if s == "__":
        return BlankSequence()
    if s == "_":
        return Blank()
    if s.endswith("___"):
        return BlankNullSequence(name=s[:-3])
    if s.endswith("__"):
        return BlankSequence(name=s[:-2])
    if "_" in s:
        parts = s.split("_", 1)
        name = parts[0] if parts[0] else None
        head = parts[1] if parts[1] else None
        return Blank(name=name, head=head)
    return Symbol(s)
