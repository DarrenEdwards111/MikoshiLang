"""Core Wolfram-language semantics for MikoshiLang.

This module owns the parts of the language that must be *correct* rather than
merely present:

* DownValues ordered by specificity (``f[0] = 1`` beats ``f[n_]``), memoisation
  ``f[n_] := f[n] = ...``, conditional definitions ``lhs := rhs /; test``.
* A backtracking pattern matcher (Blank/BlankSequence/BlankNullSequence with
  head constraints, Condition ``/;``, PatternTest ``?``, Alternatives,
  Orderless and Flat heads).
* Scoping: Module (lexical renaming), Block (dynamic), With (substitution).
* Pure functions ``Function[x, body]``, ``Function[{x, y}, body]``, ``#``,
  ``#n``, ``##``.
* Control flow (If, Which, Switch, Do, While, For, Break, Continue, Return,
  Catch/Throw) and attributes (HoldAll, HoldFirst, HoldRest, Listable, Flat,
  Orderless).
* List and structural functions (Part, Length, Take, Drop, Select, Cases,
  Position, Count, Sort, Fold, Nest, Thread, Outer, Inner, ...).
* Solve on systems, DSolve on y[x], Series with Wolfram's order semantics.

Anything not implemented raises ``NotImplementedError`` instead of returning a
plausible-looking wrong answer.
"""
from __future__ import annotations

import functools
import itertools
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

import sympy as sp

from .expr import Expr, Symbol

UNHANDLED = object()

# ─────────────────────────── session state ───────────────────────────


class SessionState:
    """All mutable definitions of one evaluation session."""

    def __init__(self):
        self.own: Dict[str, Any] = {}               # OwnValues: x = 5
        self.exact: Dict[Any, tuple] = {}           # pattern-free DownValues/SubValues: tag(lhs) -> (lhs, rhs)
        self.down: Dict[str, List[tuple]] = {}      # pattern DownValues/SubValues: key -> [(lhs, rhs, generality)]
        self.attrs: Dict[str, set] = {}             # user attributes
        self.counter = 0                            # Module variable counter
        self.temporaries: set = set()               # live Module-local symbols (Wolfram's Temporary)
        self.escape_seq = 0                         # bumped when a definition stores a reference to a temporary

    # Backwards compatible tuple-style access: state()[0] = values, state()[1] = definitions
    def __getitem__(self, index):
        if index == 0:
            return self.own
        if index == 1:
            return [(l, r) for l, r in self.exact.values()] + [
                (l, r) for defs in self.down.values() for l, r, _ in defs]
        raise IndexError(index)

    def clear(self):
        self.own.clear()
        self.exact.clear()
        self.down.clear()
        self.attrs.clear()
        self.temporaries.clear()

    # ── whole-symbol snapshots (used by Block) ──

    def snapshot(self, name: str):
        """Remove and return every definition attached to ``name``.

        Covers OwnValues, DownValues and SubValues (exact and pattern-based,
        all stored under the innermost head) and attributes.
        """
        exact = {k: v for k, v in self.exact.items() if _def_key(v[0]) == name}
        for k in exact:
            del self.exact[k]
        return (name in self.own, self.own.pop(name, None), exact,
                self.down.pop(name, None), self.attrs.pop(name, None))

    def restore(self, name: str, snap):
        had_own, own, exact, down, attrs = snap
        # Drop whatever was defined for name while it was localised ...
        self.own.pop(name, None)
        for k in [k for k, v in self.exact.items() if _def_key(v[0]) == name]:
            del self.exact[k]
        self.down.pop(name, None)
        self.attrs.pop(name, None)
        # ... and put the original definitions back exactly.
        if had_own:
            self.own[name] = own
        self.exact.update(exact)
        if down is not None:
            self.down[name] = down
        if attrs is not None:
            self.attrs[name] = attrs

    # ── Module temporaries ──

    def note_write(self, owner: str, *content):
        """Record that a definition for ``owner`` now refers to ``content``.

        If that content mentions a live Module temporary (other than owner
        itself), the temporary may have escaped its Module and must not be
        cleared when the Module returns.
        """
        if not self.temporaries:
            return
        names: set = set()
        for c in content:
            symbol_names(c, names)
        names.discard(owner)
        if names & self.temporaries:
            self.escape_seq += 1

    def definitions_of(self, name: str):
        """All stored expressions belonging to symbol ``name``."""
        out = []
        if name in self.own:
            out.append(self.own[name])
        for l, r in self.exact.values():
            if _def_key(l) == name:
                out.extend((l, r))
        for l, r, _ in self.down.get(name, ()):
            out.extend((l, r))
        return out

    def referenced_from_outside(self, names: set) -> set:
        """Which of ``names`` are referenced by a definition of a symbol not in ``names``."""
        found: set = set()

        def scan(owner, *content):
            if owner in names:
                return
            refs: set = set()
            for c in content:
                symbol_names(c, refs)
            found.update(refs & names)

        for k, v in self.own.items():
            scan(k, v)
        for l, r in self.exact.values():
            scan(_def_key(l), l, r)
        for k, defs in self.down.items():
            for l, r, _ in defs:
                scan(k, l, r)
        return found


_state: ContextVar[Optional[SessionState]] = ContextVar("mikoshilang_state", default=None)
_default_state = SessionState()


def state() -> SessionState:
    return _state.get() or _default_state


# ─────────────────────────── control-flow signals ───────────────────────────


class BreakSignal(Exception):
    pass


class ContinueSignal(Exception):
    pass


class ReturnSignal(Exception):
    def __init__(self, value):
        super().__init__("Return")
        self.value = value


class ThrowSignal(Exception):
    def __init__(self, value, tag=None):
        super().__init__("Throw")
        self.value = value
        self.tag = tag


MAX_ITERATIONS = 1_000_000
MAX_TABLE = 100_000

# ─────────────────────────── attributes ───────────────────────────

_LISTABLE_MATH = {
    "Sin", "Cos", "Tan", "Cot", "Sec", "Csc", "ArcSin", "ArcCos", "ArcTan", "ArcCot", "Sinh", "Cosh",
    "Tanh", "Coth", "ArcSinh", "ArcCosh", "ArcTanh", "Exp", "Log", "Sqrt", "Abs", "Sign", "Floor",
    "Ceiling", "Round", "Factorial", "Gamma", "Mod", "Plus", "Times", "Power", "Subtract", "Divide",
    "PrimeQ", "Re", "Im", "Conjugate", "Binomial", "Erf", "Fibonacci", "Quotient",
}

BUILTIN_ATTRIBUTES: Dict[str, set] = {
    "Plus": {"Flat", "Listable", "NumericFunction", "OneIdentity", "Orderless", "Protected"},
    "Times": {"Flat", "Listable", "NumericFunction", "OneIdentity", "Orderless", "Protected"},
    "Power": {"Listable", "NumericFunction", "OneIdentity", "Protected"},
    "Hold": {"HoldAll", "Protected"}, "HoldForm": {"HoldAll", "Protected"},
    "HoldComplete": {"HoldAll", "Protected"}, "HoldPattern": {"HoldAll", "Protected"},
    "Function": {"HoldAll", "Protected"}, "Module": {"HoldAll", "Protected"},
    "Block": {"HoldAll", "Protected"}, "With": {"HoldAll", "Protected"},
    "If": {"HoldRest", "Protected"}, "Which": {"HoldAll", "Protected"},
    "Switch": {"HoldRest", "Protected"}, "Do": {"HoldAll", "Protected"},
    "While": {"HoldAll", "Protected"}, "For": {"HoldAll", "Protected"},
    "Table": {"HoldAll", "Protected"}, "Set": {"HoldFirst", "Protected", "SequenceHold"},
    "SetDelayed": {"HoldAll", "Protected", "SequenceHold"}, "RuleDelayed": {"HoldRest", "Protected", "SequenceHold"},
    "Rule": {"Protected", "SequenceHold"}, "Attributes": {"HoldAll", "Listable", "Protected"},
    "SetAttributes": {"HoldFirst", "Protected"}, "Clear": {"HoldAll", "Protected"},
    "CompoundExpression": {"HoldAll", "Protected"}, "And": {"Flat", "HoldAll", "OneIdentity", "Protected"},
    "Or": {"Flat", "HoldAll", "OneIdentity", "Protected"},
}
for _h in _LISTABLE_MATH:
    BUILTIN_ATTRIBUTES.setdefault(_h, {"Listable", "NumericFunction", "Protected"}).add("Listable")

KNOWN_ATTRIBUTES = {"HoldAll", "HoldFirst", "HoldRest", "Listable", "Flat", "Orderless", "OneIdentity",
                    "Protected", "NumericFunction", "SequenceHold", "Locked", "ReadProtected", "Constant"}
# Attributes that change evaluation and are implemented for user symbols:
SUPPORTED_USER_ATTRIBUTES = {"HoldAll", "HoldFirst", "HoldRest", "Listable", "Flat", "Orderless",
                             "OneIdentity", "Protected", "NumericFunction", "SequenceHold", "Constant",
                             "ReadProtected"}


def attributes(head) -> set:
    if not isinstance(head, str):
        return set()
    user = state().attrs.get(head)
    builtin = BUILTIN_ATTRIBUTES.get(head)
    if user and builtin:
        return user | builtin
    return user or builtin or set()


def user_attributes(head) -> set:
    return state().attrs.get(head, set()) if isinstance(head, str) else set()


# ─────────────────────────── helpers ───────────────────────────


def is_int(x) -> bool:
    return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, sp.Integer)


def is_list(x) -> bool:
    return isinstance(x, Expr) and x.head == "List"


def L(*items):
    return Expr("List", *items)


def as_head(f):
    """Convert a value into something usable as an Expr head."""
    if isinstance(f, Symbol):
        return f.name
    if isinstance(f, Expr):
        return f
    if isinstance(f, str):
        raise TypeError(f"String {f!r} cannot be used as a function")
    raise TypeError(f"{f!r} cannot be used as a function")


def call(f, *args):
    return Expr(as_head(f), *args)


def tag(e):
    """Structural identity key (distinguishes 1 from 1.0, like SameQ)."""
    if isinstance(e, bool):
        return ("b", e)
    if isinstance(e, int):
        return ("i", e)
    if isinstance(e, float):
        return ("r", e)
    if isinstance(e, str):
        return ("s", e)
    if isinstance(e, Symbol):
        return ("y", e.name)
    if isinstance(e, Expr):
        h = e.head if isinstance(e.head, str) else tag(e.head)
        return ("e", h, tuple(tag(a) for a in e.args))
    if isinstance(e, sp.Integer):
        return ("i", int(e))
    if isinstance(e, sp.Float):
        return ("r", str(e), e._prec)
    if e is None:
        return ("n",)
    if isinstance(e, tuple):
        return ("q", tuple(tag(a) for a in e))
    return ("o", repr(e))


def same(a, b) -> bool:
    return tag(a) == tag(b)


def is_number(e) -> bool:
    """Explicit numbers (NumberQ): integers, reals, exact rationals, complex numbers."""
    if isinstance(e, bool):
        return False
    if isinstance(e, (int, float, complex, sp.Number)):
        return True
    from .formatting import rational_value
    if rational_value(e) is not None:
        return True
    if isinstance(e, Expr) and e.head in ("Plus", "Times", "Complex"):
        try:
            from .math_ops import to_sympy
            v = to_sympy(e)
            return bool(v.is_number) and not v.free_symbols and v.atoms(sp.NumberSymbol) == set() \
                and (v.is_Number or (v.as_real_imag()[1] != 0 and all(t.is_Number for t in v.as_real_imag())))
        except Exception:
            return False
    if isinstance(e, Symbol) and e.name == "I":
        return True
    return False


def to_sp(e):
    from .math_ops import to_sympy
    return to_sympy(e)


def from_sp(e):
    from .math_ops import from_sympy
    return from_sympy(e)


def contains(expr, pred) -> bool:
    if pred(expr):
        return True
    if isinstance(expr, Expr):
        if isinstance(expr.head, Expr) and contains(expr.head, pred):
            return True
        return any(contains(a, pred) for a in expr.args)
    return False


_PATTERN_HEADS = {"Pattern", "Blank", "BlankSequence", "BlankNullSequence", "Condition", "PatternTest",
                  "Alternatives", "Optional", "HoldPattern"}


def has_pattern(e) -> bool:
    return contains(e, lambda x: isinstance(x, Expr) and isinstance(x.head, str) and x.head in _PATTERN_HEADS)


# ─────────────────────────── substitution ───────────────────────────
#
# ``substitute`` is the single place where values are inserted into held code:
# Function application, With, Module renaming, and the right-hand sides of
# definitions and rules.  It follows Wolfram's lexical scoping rules:
#
# * a variable bound by an inner scoping construct (Function, Module, With,
#   Block, or a pattern name on the left of Rule/RuleDelayed/Set/SetDelayed/
#   Condition) shadows the outer one, so it is never replaced inside that
#   construct;
# * substitution is capture-avoiding: if a value being inserted mentions a
#   symbol that an inner lexical construct binds, that inner variable is
#   renamed to a fresh ``name$`` first (as Wolfram does), so
#   ``Function[y, Function[x, x + y]][x]`` gives ``Function[x$, x$ + x]``.
#
# Block is dynamic, so its variables shadow but are never renamed: renaming
# would change which global symbol the Block localises.

_LEXICAL_SCOPES = ("Function", "Module", "With")
_PATTERN_SCOPES = ("Rule", "RuleDelayed", "Set", "SetDelayed", "Condition", "UpSet", "UpSetDelayed")


def symbol_names(t, out=None) -> set:
    """Every symbol name (including string heads) occurring anywhere in t."""
    if out is None:
        out = set()
    stack = [t]
    while stack:
        e = stack.pop()
        if isinstance(e, Symbol):
            out.add(e.name)
        elif isinstance(e, Expr):
            if isinstance(e.head, str):
                out.add(e.head)
            else:
                stack.append(e.head)
            stack.extend(e.args)
        elif isinstance(e, tuple):
            stack.extend(e)
    return out


def pattern_names(t) -> List[str]:
    """Names bound by Pattern[name, ...] anywhere inside t (in order, no duplicates)."""
    out: List[str] = []
    stack = [t]
    while stack:
        e = stack.pop()
        if isinstance(e, Expr):
            if e.head == "Pattern" and len(e.args) == 2 and isinstance(e.args[0], Symbol):
                if e.args[0].name not in out:
                    out.append(e.args[0].name)
            if isinstance(e.head, Expr):
                stack.append(e.head)
            stack.extend(reversed(e.args))
    return out


def rename(t, old: str, new: str):
    """Structural alpha-renaming of every occurrence of symbol ``old`` (binders included)."""
    if isinstance(t, Symbol):
        return Symbol(new) if t.name == old else t
    if isinstance(t, Expr):
        h = t.head
        if isinstance(h, str):
            h = new if h == old else h
        else:
            h = rename(h, old, new)
        return Expr(h, *(rename(a, old, new) for a in t.args))
    return t


def _fresh(name: str, avoid: set) -> str:
    cand = name + "$"
    while cand in avoid:
        cand += "$"
    return cand


def _function_params(t) -> Optional[List[str]]:
    if len(t.args) < 2:
        return None
    p = t.args[0]
    if isinstance(p, Symbol):
        return [p.name]
    if is_list(p) and all(isinstance(x, Symbol) for x in p.args):
        return [x.name for x in p.args]
    return None


def _scope_spec_names(spec) -> Optional[List[str]]:
    if not is_list(spec):
        return None
    out = []
    for v in spec.args:
        if isinstance(v, Symbol):
            out.append(v.name)
        elif (isinstance(v, Expr) and v.head in ("Set", "SetDelayed") and len(v.args) == 2
              and isinstance(v.args[0], Symbol)):
            out.append(v.args[0].name)
        else:
            return None
    return out


class _Subst:
    __slots__ = ("b", "_value_names")

    def __init__(self, b):
        self.b = b
        self._value_names = None

    def value_names(self) -> set:
        if self._value_names is None:
            names: set = set()
            for v in self.b.values():
                symbol_names(v, names)
            self._value_names = names
        return self._value_names


def substitute(t, b: Dict[str, Any]):
    """Capture-avoiding substitution of the symbols bound in b (tuple values splice)."""
    if not b:
        return t
    return _subst(t, b, _Subst(b))


def _narrow(b, ctx, bound):
    """Bindings visible inside a construct that binds ``bound``."""
    if not any(n in b for n in bound):
        return b, ctx
    nb = {k: v for k, v in b.items() if k not in bound}
    return nb, _Subst(nb)


def _alpha(t, bound, b, ctx, renamable):
    """Rename bound variables of construct t that would capture free symbols of b's values."""
    clash = [n for n in bound if n in renamable] if renamable else []
    if not clash:
        return t, bound
    vn = ctx.value_names()
    clash = [n for n in clash if n in vn]
    if not clash:
        return t, bound
    avoid = symbol_names(t) | vn | set(b)
    bound = list(bound)
    for n in clash:
        fresh = _fresh(n, avoid)
        avoid.add(fresh)
        t = rename(t, n, fresh)
        bound[bound.index(n)] = fresh
    return t, bound


def _subst(t, b, ctx):
    if isinstance(t, Symbol):
        if t.name in b:
            v = b[t.name]
            return Expr("Sequence", *v) if isinstance(v, tuple) else v
        return t
    if not isinstance(t, Expr):
        return t
    h = t.head
    if isinstance(h, str) and t.args:
        # ── lexical scoping constructs: shadow, and rename to avoid capture ──
        if h == "Function":
            params = _function_params(t)
            if params is not None:
                t, params = _alpha(t, params, b, ctx, params)
                nb, nctx = _narrow(b, ctx, params)
                return Expr("Function", t.args[0], *(_subst(a, nb, nctx) for a in t.args[1:]))
        elif h in ("Module", "With", "Block") and len(t.args) == 2:
            names = _scope_spec_names(t.args[0])
            if names is not None:
                if h != "Block":
                    t, names = _alpha(t, names, b, ctx, names)
                nb, nctx = _narrow(b, ctx, names)
                spec = []
                for v in t.args[0].args:
                    if isinstance(v, Symbol):
                        spec.append(v)
                    else:  # initial values live in the enclosing scope
                        spec.append(Expr(v.head, v.args[0], _subst(v.args[1], b, ctx)))
                return Expr(h, Expr("List", *spec), _subst(t.args[1], nb, nctx))
        elif h in _PATTERN_SCOPES and len(t.args) == 2:
            names = pattern_names(t.args[0])
            if names:
                t, names = _alpha(t, names, b, ctx, names)
                nb, nctx = _narrow(b, ctx, names)
                return Expr(h, _subst(t.args[0], nb, nctx), _subst(t.args[1], nb, nctx))
        if h == "Pattern" and len(t.args) == 2:
            return Expr("Pattern", t.args[0], _subst(t.args[1], b, ctx))
    if isinstance(h, str):
        if h in b and not isinstance(b[h], tuple):
            nh = b[h]
            h = nh.name if isinstance(nh, Symbol) else nh if isinstance(nh, Expr) else h
    else:
        h = _subst(h, b, ctx)
    args = []
    for a in t.args:
        if isinstance(a, Symbol) and a.name in b and isinstance(b[a.name], tuple):
            args.extend(b[a.name])
        else:
            args.append(_subst(a, b, ctx))
    return Expr(h, *args)


def replace_slots(body, args):
    """Substitute #n and ##n in a pure-function body (stops at nested Function)."""
    if isinstance(body, Expr):
        if body.head == "Slot" and len(body.args) == 1:
            if isinstance(body.args[0], str):          # #name -> #1["name"]
                if not args:
                    raise ValueError(f"Function: slot #{body.args[0]} requested but no argument given")
                if isinstance(args[0], Expr):
                    return Expr(args[0], body.args[0])
                raise ValueError(f"Function: #{body.args[0]} needs an association argument, got {args[0]!r}")
            i = int(body.args[0])
            if i > len(args):
                raise ValueError(f"Function: slot #{i} requested but only {len(args)} argument(s) given")
            return args[i - 1]
        if body.head == "SlotSequence":
            return Expr("Sequence", *args[int(body.args[0]) - 1:])
        if body.head == "Function":
            return body
        h = replace_slots(body.head, args) if isinstance(body.head, Expr) else body.head
        new_args = []
        for a in body.args:
            r = replace_slots(a, args)
            if isinstance(a, Expr) and a.head == "SlotSequence":
                new_args.extend(r.args)
            else:
                new_args.append(r)
        return Expr(h if not isinstance(h, Symbol) else h.name, *new_args)
    return body


# ─────────────────────────── pattern matching ───────────────────────────


def _head_ok(expr, hsym) -> bool:
    from .formatting import rational_value
    name = hsym.name if isinstance(hsym, Symbol) else str(hsym)
    if name == "Integer":
        return is_int(expr)
    if name == "Real":
        return isinstance(expr, (float, sp.Float))
    if name == "Rational":
        return rational_value(expr) is not None and not is_int(expr)
    if name == "Complex":
        return isinstance(expr, complex)
    if name == "String":
        return isinstance(expr, str)
    if name == "Symbol":
        return isinstance(expr, (Symbol, bool))
    if isinstance(expr, Expr):
        return expr.head == name
    return False


def _seq_info(p):
    """Describe a sequence pattern: (min_len, name, head, wrappers) or None."""
    wrappers = []
    name = None
    while isinstance(p, Expr) and p.head in ("Condition", "PatternTest") and len(p.args) == 2:
        wrappers.append(p)
        p = p.args[0]
    if isinstance(p, Expr) and p.head == "Pattern" and len(p.args) == 2:
        name = p.args[0].name
        p = p.args[1]
    if isinstance(p, Expr) and p.head in ("BlankSequence", "BlankNullSequence"):
        return (1 if p.head == "BlankSequence" else 0, name, p.args[0] if p.args else None, wrappers)
    return None


def _is_single_blank(p) -> bool:
    while isinstance(p, Expr) and p.head in ("Condition", "PatternTest") and len(p.args) == 2:
        p = p.args[0]
    if isinstance(p, Expr) and p.head == "Pattern" and len(p.args) == 2:
        p = p.args[1]
    return isinstance(p, Expr) and p.head == "Blank"


def match(pat, expr, b: Dict[str, Any], ev) -> Iterator[Dict[str, Any]]:
    """Yield every binding dict under which pat matches expr."""
    if isinstance(pat, Expr):
        h = pat.head
        if h == "Pattern" and len(pat.args) == 2 and isinstance(pat.args[0], Symbol):
            name = pat.args[0].name
            for b2 in match(pat.args[1], expr, b, ev):
                if name in b2:
                    if not isinstance(b2[name], tuple) and same(b2[name], expr):
                        yield b2
                else:
                    b3 = dict(b2)
                    b3[name] = expr
                    yield b3
            return
        if h == "Blank":
            if not pat.args or _head_ok(expr, pat.args[0]):
                yield b
            return
        if h in ("BlankSequence", "BlankNullSequence"):
            # Outside an argument list a sequence pattern matches one expression.
            if not pat.args or _head_ok(expr, pat.args[0]):
                yield b
            return
        if h == "Condition" and len(pat.args) == 2:
            for b2 in match(pat.args[0], expr, b, ev):
                if ev(substitute(pat.args[1], b2)) is True:
                    yield b2
            return
        if h == "PatternTest" and len(pat.args) == 2:
            for b2 in match(pat.args[0], expr, b, ev):
                if ev(call(pat.args[1], expr)) is True:
                    yield b2
            return
        if h == "Alternatives":
            for alt in pat.args:
                yield from match(alt, expr, b, ev)
            return
        if h == "HoldPattern" and len(pat.args) == 1:
            yield from match(pat.args[0], expr, b, ev)
            return
        if h == "Optional":
            raise NotImplementedError("Optional patterns (x_.) are not supported")
        if not isinstance(expr, Expr):
            return
        eh = expr.head
        if isinstance(h, str):
            if h != eh:
                return
            head_bindings = [b]
        else:
            head_bindings = list(match(h, Symbol(eh) if isinstance(eh, str) else eh, b, ev))
        attrs = attributes(eh)
        for bh in head_bindings:
            yield from match_sequence(pat.args, expr.args, bh, ev, eh, attrs)
        return
    if isinstance(pat, Symbol):
        if isinstance(expr, Symbol) and expr.name == pat.name:
            yield b
        return
    if same(pat, expr):
        yield b


def match_sequence(pats, args, b, ev, head=None, attrs=()):
    if "Orderless" in attrs and len(args) > 1:
        if len(args) > 8:
            raise NotImplementedError("Orderless matching is limited to 8 arguments")
        seen = set()
        for perm in itertools.permutations(args):
            key = tuple(tag(a) for a in perm)
            if key in seen:
                continue
            seen.add(key)
            yield from _match_seq(pats, 0, perm, 0, b, ev, head, "Flat" in attrs)
        return
    yield from _match_seq(pats, 0, tuple(args), 0, b, ev, head, "Flat" in attrs)


def _match_seq(pats, i, args, j, b, ev, head, flat):
    if i == len(pats):
        if j == len(args):
            yield b
        return
    p = pats[i]
    info = _seq_info(p)
    remaining = len(args) - j
    if info is not None:
        min_len, name, hconstraint, wrappers = info
        for k in range(min_len, remaining + 1):
            chunk = tuple(args[j:j + k])
            if hconstraint is not None and not all(_head_ok(a, hconstraint) for a in chunk):
                continue
            b2 = b
            if name is not None:
                if name in b:
                    if not (isinstance(b[name], tuple) and len(b[name]) == k
                            and all(same(x, y) for x, y in zip(b[name], chunk))):
                        continue
                else:
                    b2 = dict(b)
                    b2[name] = chunk
            ok = True
            for w in reversed(wrappers):
                if w.head == "PatternTest":
                    if not all(ev(call(w.args[1], a)) is True for a in chunk):
                        ok = False
                        break
                else:
                    if ev(substitute(w.args[1], b2)) is not True:
                        ok = False
                        break
            if ok:
                yield from _match_seq(pats, i + 1, args, j + k, b2, ev, head, flat)
        return
    if flat and isinstance(head, str) and _is_single_blank(p):
        for k in range(1, remaining + 1):
            value = args[j] if k == 1 else Expr(head, *args[j:j + k])
            for b2 in match(p, value, b, ev):
                yield from _match_seq(pats, i + 1, args, j + k, b2, ev, head, flat)
        return
    if remaining <= 0:
        return
    for b2 in match(p, args[j], b, ev):
        yield from _match_seq(pats, i + 1, args, j + 1, b2, ev, head, flat)


def first_match(pat, expr, ev):
    for b in match(pat, expr, {}, ev):
        return b
    return None


# ─────────────────────────── definitions ───────────────────────────


def _generality(p) -> float:
    """Lower = more specific. Literal parts cost 0, constrained blanks less than bare ones."""
    if isinstance(p, Expr):
        h = p.head
        if h == "Pattern" and len(p.args) == 2:
            return _generality(p.args[1])
        if h == "Blank":
            return 1.0 if p.args else 2.0
        if h == "BlankSequence":
            return 3.0 if p.args else 4.0
        if h == "BlankNullSequence":
            return 4.0 if p.args else 5.0
        if h in ("Condition", "PatternTest"):
            return _generality(p.args[0]) - 0.5
        if h == "Alternatives":
            return max((_generality(a) for a in p.args), default=0.0)
        g = sum(_generality(a) for a in p.args)
        if isinstance(h, Expr):
            g += _generality(h)
        return g
    return 0.0


def _def_key(lhs):
    h = lhs.head
    while isinstance(h, Expr):
        h = h.head
    return h if isinstance(h, str) else repr(h)


def _condition_tag(rhs):
    if isinstance(rhs, Expr) and rhs.head == "Condition" and len(rhs.args) == 2:
        return tag(rhs.args[1])
    return None


def store_definition(lhs, rhs):
    st = state()
    st.note_write(_def_key(lhs), lhs, rhs)
    if not has_pattern(lhs):
        st.exact[tag(lhs)] = (lhs, rhs)
        return
    key = _def_key(lhs)
    defs = st.down.setdefault(key, [])
    # Wolfram treats "lhs := rhs /; test" as distinct from "lhs := rhs": the condition is part of the key.
    lt = (tag(lhs), _condition_tag(rhs))
    for i, (l, r, g) in enumerate(defs):
        if (tag(l), _condition_tag(r)) == lt:
            defs[i] = (lhs, rhs, g)
            return
    g = _generality(lhs)
    pos = len(defs)
    for i, (_, _, gi) in enumerate(defs):
        if gi > g:
            pos = i
            break
    defs.insert(pos, (lhs, rhs, g))


def apply_downvalues(expr, ev):
    st = state()
    if st.exact:
        hit = st.exact.get(tag(expr))
        if hit is not None:
            return hit[1]
    defs = st.down.get(_def_key(expr))
    if not defs:
        return UNHANDLED
    for lhs, rhs, _ in list(defs):
        for b in match(lhs, expr, {}, ev):
            if isinstance(rhs, Expr) and rhs.head == "Condition" and len(rhs.args) == 2:
                if ev(substitute(rhs.args[1], b)) is not True:
                    continue
                return substitute(rhs.args[0], b)
            return substitute(rhs, b)
    return UNHANDLED


def clear_symbol(name: str, all_: bool = False):
    st = state()
    st.own.pop(name, None)
    st.down.pop(name, None)
    for k in [k for k, (l, _) in st.exact.items() if _def_key(l) == name]:
        del st.exact[k]
    if all_:
        st.attrs.pop(name, None)


# ─────────────────────────── canonical order ───────────────────────────


def order_key(e):
    """Total order used for Orderless arguments and Sort (numbers < atoms < compounds)."""
    if isinstance(e, bool):
        s = "True" if e else "False"
        return (1, s.lower(), s.swapcase())
    if isinstance(e, (int, float, sp.Number)):
        return (0, sp.Float(e, 30) if isinstance(e, float) else sp.sympify(e), 0, 0)
    from .formatting import rational_value
    q = rational_value(e)
    if q is not None:
        return (0, sp.Rational(q.numerator, q.denominator), 0, 1)
    if isinstance(e, complex):
        return (0, sp.Float(e.real), sp.Float(e.imag), 0)
    if isinstance(e, str):
        return (1, e.lower(), e.swapcase(), 0)
    if isinstance(e, Symbol):
        return (1, e.name.lower(), e.name.swapcase(), 1)
    if isinstance(e, Expr):
        if e.head == "List":
            return (2, "", len(e.args), tuple(order_key(a) for a in e.args))
        return (3, e.head if isinstance(e.head, str) else repr(e.head), len(e.args),
                tuple(order_key(a) for a in e.args))
    return (4, repr(e))


def _sortable_canonically(e) -> bool:
    if isinstance(e, Expr):
        from .formatting import rational_value
        if rational_value(e) is not None:
            return True
        return e.head == "List" and all(_sortable_canonically(a) for a in e.args)
    return True


# ─────────────────────────── iterators (Table / Do) ───────────────────────────


def _iterator_values(spec, ev):
    """Return (variable name or None, list of values) for an iterator spec."""
    if not is_list(spec):
        n = ev(spec)
        if not is_int(n):
            raise ValueError("Iterator count must be an explicit integer")
        return None, [None] * int(n)
    items = spec.args
    if len(items) == 1:
        n = ev(items[0])
        if not is_int(n):
            raise ValueError("Iterator count must be an explicit integer")
        return None, [None] * int(n)
    var = items[0]
    if not isinstance(var, Symbol):
        raise ValueError("Iterator variable must be a symbol")
    if len(items) == 2:
        bound = ev(items[1])
        if is_list(bound):
            return var.name, list(bound.args)
        lo, hi, step = 1, bound, 1
    elif len(items) == 3:
        lo, hi, step = ev(items[1]), ev(items[2]), 1
    elif len(items) == 4:
        lo, hi, step = ev(items[1]), ev(items[2]), ev(items[3])
    else:
        raise ValueError("Invalid iterator specification")
    for v in (lo, hi, step):
        if not is_int(v):
            raise ValueError("Iterator bounds must be explicit integers")
    lo, hi, step = int(lo), int(hi), int(step)
    if step == 0:
        raise ValueError("Iterator step cannot be zero")
    count = max(0, (hi - lo) // step + 1)
    if count > MAX_TABLE:
        raise ValueError(f"Iterator would produce more than {MAX_TABLE} values")
    return var.name, list(range(lo, hi + 1 if step > 0 else hi - 1, step))


class _Bound:
    """Block-style temporary binding of a symbol."""

    def __init__(self, name):
        self.name = name
        self.st = state()
        self.had = name in self.st.own if name else False
        self.old = self.st.own.get(name) if name else None

    def set(self, value):
        if self.name is not None:
            self.st.own[self.name] = value

    def restore(self):
        if self.name is None:
            return
        if self.had:
            self.st.own[self.name] = self.old
        else:
            self.st.own.pop(self.name, None)


def _table(body, specs, ev):
    if not specs:
        return ev(body)
    name, vals = _iterator_values(specs[0], ev)
    bound = _Bound(name)
    out = []
    try:
        for v in vals:
            bound.set(v)
            out.append(_table(body, specs[1:], ev))
    finally:
        bound.restore()
    return L(*out)


def _do(body, specs, ev):
    name, vals = _iterator_values(specs[0], ev)
    bound = _Bound(name)
    try:
        for v in vals:
            bound.set(v)
            if len(specs) > 1:
                _do(body, specs[1:], ev)
            else:
                try:
                    ev(body)
                except ContinueSignal:
                    continue
    finally:
        bound.restore()


# ─────────────────────────── assignment ───────────────────────────


def _assign(lhs, rhs, ev):
    st = state()
    if isinstance(lhs, Symbol):
        if lhs.name in ("Pi", "E", "I", "Infinity") or "Protected" in BUILTIN_ATTRIBUTES.get(lhs.name, ()):
            raise ValueError(f"Symbol {lhs.name} is protected")
        st.own[lhs.name] = rhs
        st.note_write(lhs.name, rhs)
        return
    if isinstance(lhs, Expr):
        if lhs.head == "List":
            if not is_list(rhs) or len(rhs.args) != len(lhs.args):
                raise ValueError("Set: list assignment needs a list of the same length")
            for l, r in zip(lhs.args, rhs.args):
                _assign(l, r, ev)
            return
        if lhs.head == "Part":
            target = lhs.args[0]
            if not isinstance(target, Symbol) or target.name not in st.own:
                raise ValueError("Part assignment needs a symbol with a value")
            idx = [ev(i) for i in lhs.args[1:]]
            st.own[target.name] = _set_part(ev(target), idx, rhs)
            st.note_write(target.name, rhs)
            return
        if isinstance(lhs.head, str) and lhs.head in ("Pattern", "Condition"):
            raise ValueError("Cannot assign to a pattern")
        if (isinstance(lhs.head, str) and len(lhs.args) == 1 and lhs.head in st.own
                and isinstance(st.own[lhs.head], Expr) and st.own[lhs.head].head == "Association"):
            # assoc[key] = value updates the association stored in the symbol (Wolfram semantics)
            from .kernel_lists import make_assoc
            key = ev(lhs.args[0])
            st.own[lhs.head] = make_assoc(list(st.own[lhs.head].args) + [Expr("Rule", key, rhs)])
            st.note_write(lhs.head, rhs)
            return
        head = lhs.head
        if not isinstance(head, str) and not isinstance(head, Expr):
            raise ValueError("Invalid assignment target")
        target_head = _def_key(lhs)
        if "Protected" in BUILTIN_ATTRIBUTES.get(target_head, ()):
            raise ValueError(f"{target_head} is protected; definitions for built-in heads are not allowed")
        if not has_pattern(lhs):
            hold = attributes(head) if isinstance(head, str) else set()
            args = list(lhs.args)
            for i, a in enumerate(args):
                if "HoldAll" in hold or (i == 0 and "HoldFirst" in hold) or (i > 0 and "HoldRest" in hold):
                    continue
                args[i] = ev(a)
            lhs = Expr(head, *args)
        store_definition(lhs, rhs)
        return
    raise ValueError(f"Cannot assign to {lhs!r}")


def _set_part(expr, idx, value):
    if not idx:
        return value
    if not isinstance(expr, Expr):
        raise ValueError("Part assignment: target is not an expression")
    i = idx[0]
    if not is_int(i):
        raise NotImplementedError("Part assignment supports integer indices only")
    i = int(i)
    n = len(expr.args)
    k = i - 1 if i > 0 else n + i
    if not 0 <= k < n:
        raise ValueError(f"Part {i} of an expression with {n} parts does not exist")
    args = list(expr.args)
    args[k] = _set_part(args[k], idx[1:], value)
    return Expr(expr.head, *args)


def _modify(target, op, amount, ev, post=False):
    if not isinstance(target, Symbol):
        raise ValueError(f"{op} needs a symbol")
    st = state()
    if target.name not in st.own:
        raise ValueError(f"{target.name} has no value")
    old = ev(target)
    new = ev(Expr(op, old, amount) if op != "Subtract" else Expr("Plus", old, Expr("Times", -1, amount)))
    if op == "Divide":
        new = ev(Expr("Times", old, Expr("Power", amount, -1)))
    st.own[target.name] = new
    st.note_write(target.name, new)
    return old if post else new


# ─────────────────────────── scoping ───────────────────────────


def _scope_vars(spec):
    if not is_list(spec):
        raise ValueError("Scoping construct needs a list of variables")
    out = []
    for v in spec.args:
        if isinstance(v, Symbol):
            out.append((v.name, None))
        elif isinstance(v, Expr) and v.head in ("Set", "SetDelayed") and isinstance(v.args[0], Symbol):
            out.append((v.args[0].name, v))
        else:
            raise ValueError(f"Invalid local variable specification {v!r}")
    return out


def _module(spec, body, ev):
    """Module: lexical scoping with fresh ``x$n`` symbols.

    Like Wolfram's Temporary symbols, locals that are still referenced when the
    Module returns (closures, values stored in global definitions, values
    carried out by Return/Throw) keep their definitions; unreferenced ones are
    removed.
    """
    st = state()
    vars_ = _scope_vars(spec)
    st.counter += 1
    n = st.counter
    mapping = {name: Symbol(f"{name}${n}") for name, _ in vars_}
    inits = {}
    for name, init in vars_:
        if init is not None:
            inits[name] = ev(init.args[1]) if init.head == "Set" else init.args[1]
    locals_ = {sym.name for sym in mapping.values()}
    st.temporaries.update(locals_)
    for name, value in inits.items():
        local = mapping[name].name
        st.own[local] = value
        st.note_write(local, value)
    seq = st.escape_seq
    roots: list = []
    ok = False
    try:
        result = ev(substitute(body, mapping))
        roots.append(result)
        ok = True
        return result
    except (ReturnSignal, ThrowSignal) as sig:
        roots.append(sig.value)
        ok = True
        raise
    finally:
        _release_temporaries(st, locals_, roots if ok else [], st.escape_seq != seq)


def _release_temporaries(st, locals_, roots, scan_state):
    """Clear Module locals unless they are reachable from roots or other definitions."""
    live: set = set()
    if roots or scan_state:
        names: set = set()
        for r in roots:
            symbol_names(r, names)
        if scan_state:
            names |= st.referenced_from_outside(locals_)
        frontier = list(names & st.temporaries)
        # Follow references through the definitions of reachable temporaries.
        while frontier:
            t = frontier.pop()
            if t in live:
                continue
            live.add(t)
            refs: set = set()
            for c in st.definitions_of(t):
                symbol_names(c, refs)
            frontier.extend((refs & st.temporaries) - live)
    for name in locals_:
        if name not in live:
            clear_symbol(name, all_=True)
            st.temporaries.discard(name)


def _block(spec, body, ev):
    """Block: dynamic scoping.  Every definition of each variable (OwnValues,
    DownValues, SubValues, attributes) is set aside for the duration of the
    body and restored afterwards, whatever way the body exits."""
    st = state()
    vars_ = _scope_vars(spec)
    # Initial values are evaluated before any variable is localised (Wolfram semantics).
    inits = []
    for name, init in vars_:
        if init is not None:
            inits.append((name, ev(init.args[1]) if init.head == "Set" else init.args[1]))
    saved = []
    try:
        for name, _ in vars_:
            saved.append((name, st.snapshot(name)))
        for name, value in inits:
            st.own[name] = value
        return ev(body)
    finally:
        for name, snap in reversed(saved):
            st.restore(name, snap)


def _with(spec, body, ev):
    vars_ = _scope_vars(spec)
    b = {}
    for name, init in vars_:
        if init is None:
            raise ValueError("With needs initial values for every variable")
        b[name] = ev(init.args[1]) if init.head == "Set" else init.args[1]
    return ev(substitute(body, b))


# ─────────────────────────── functions ───────────────────────────


def apply_function(f, args, ev):
    fa = f.args
    if len(fa) == 1:
        return ev(replace_slots(fa[0], tuple(args)))
    if len(fa) == 3:
        raise NotImplementedError("Function[params, body, attributes] is not supported")
    if len(fa) == 2:
        params = fa[0]
        if isinstance(params, Symbol):
            names = [params.name]
        elif is_list(params) and all(isinstance(p, Symbol) for p in params.args):
            names = [p.name for p in params.args]
        else:
            raise ValueError("Function parameters must be a symbol or a list of symbols")
        if len(args) < len(names):
            raise ValueError(f"Function expects {len(names)} argument(s), got {len(args)}")
        return ev(substitute(fa[1], dict(zip(names, args))))
    raise ValueError("Function expects Function[body] or Function[params, body]")


def apply_derivative(head, args, ev):
    """Derivative[n...][f][args]: differentiate a known function, else stay symbolic."""
    orders = head.head.args
    f = head.args[0] if head.args else None
    if not all(is_int(o) for o in orders) or len(orders) != len(args):
        return UNHANDLED
    known = False
    if isinstance(f, Expr) and f.head == "Function":
        known = True
    elif isinstance(f, Symbol):
        st = state()
        from .math_ops import _FUNCS
        known = (f.name in st.down or f.name in _FUNCS or f.name in ("Log", "ArcTan")
                 or any(l.head == f.name for l, _ in st.exact.values()))
    if not known:
        return UNHANDLED
    dummies = [Symbol(f"$d{i}") for i in range(len(args))]
    body = ev(Expr(as_head(f), *dummies))
    if contains(body, lambda x: isinstance(x, Expr) and isinstance(f, Symbol) and x.head == f.name):
        return UNHANDLED
    spec = [L(d, int(n)) for d, n in zip(dummies, orders) if int(n)]
    d = ev(Expr("D", body, *spec)) if spec else body
    return ev(substitute(d, {s.name: a for s, a in zip(dummies, args)}))


# ─────────────────────────── replacement ───────────────────────────


def _rules_list(r):
    if is_list(r):
        if r.args and all(is_list(x) for x in r.args):
            return None  # list of rule lists handled by caller
        rules = list(r.args)
    else:
        rules = [r]
    for rule in rules:
        if not (isinstance(rule, Expr) and rule.head in ("Rule", "RuleDelayed") and len(rule.args) == 2):
            raise ValueError(f"{rule!r} is not a valid replacement rule")
    return rules


def _try_rules(e, rules, ev):
    for rule in rules:
        b = first_match(rule.args[0], e, ev)
        if b is not None:
            return True, substitute(rule.args[1], b)
    return False, None


def replace_all_once(e, rules, ev):
    hit, value = _try_rules(e, rules, ev)
    if hit:
        return value
    if isinstance(e, Expr):
        h = e.head
        if isinstance(h, str):
            hhit, hv = _try_rules(Symbol(h), rules, ev)
            if hhit:
                h = as_head(hv) if isinstance(hv, (Symbol, Expr)) else h
        else:
            h = replace_all_once(h, rules, ev)
        return Expr(h, *(replace_all_once(a, rules, ev) for a in e.args))
    return e


def _replace(expr, rules_expr, ev, repeated=False):
    rules_expr = ev(rules_expr)
    rules = _rules_list(rules_expr)
    if rules is None:
        return L(*(_replace(expr, r, ev, repeated) for r in rules_expr.args))
    current = ev(expr)
    if not repeated:
        return ev(replace_all_once(current, rules, ev))
    for _ in range(65536):
        nxt = ev(replace_all_once(current, rules, ev))
        if same(nxt, current):
            return current
        current = nxt
    raise RecursionError("ReplaceRepeated did not converge after 65536 iterations")


# ─────────────────────────── held special forms ───────────────────────────

_SELF_HELD = {"Hold", "HoldForm", "HoldComplete", "HoldPattern", "Function", "Slot", "SlotSequence",
              "Pattern", "Blank", "BlankSequence", "BlankNullSequence", "PatternTest", "Alternatives",
              "Optional", "Condition"}


def _truth(v):
    return v is True


def held(expr, ev):
    """Special forms that receive unevaluated arguments. Returns UNHANDLED if not one."""
    h = expr.head
    if not isinstance(h, str):
        return UNHANDLED
    a = expr.args
    if h in _SELF_HELD:
        return expr
    fn = _HELD.get(h)
    if fn is None:
        return UNHANDLED
    return fn(a, ev, expr)


def _h_compound(a, ev, expr):
    result = None
    for x in a:
        result = ev(x)
    return result


def _h_set(a, ev, expr):
    if len(a) != 2:
        raise ValueError("Set expects lhs = rhs")
    rhs = ev(a[1])
    _assign(a[0], rhs, ev)
    return rhs


def _h_setdelayed(a, ev, expr):
    if len(a) != 2:
        raise ValueError("SetDelayed expects lhs := rhs")
    _assign(a[0], a[1], ev)
    return None


def _h_unset(a, ev, expr):
    lhs = a[0]
    st = state()
    if isinstance(lhs, Symbol):
        st.own.pop(lhs.name, None)
    elif isinstance(lhs, Expr):
        st.exact.pop(tag(lhs), None)
        defs = st.down.get(_def_key(lhs), [])
        st.down[_def_key(lhs)] = [d for d in defs if not same(d[0], lhs)]
    return None


def _h_clear(a, ev, expr, all_=False):
    for s in a:
        if isinstance(s, Symbol):
            clear_symbol(s.name, all_)
        elif isinstance(s, str):
            clear_symbol(s, all_)
        else:
            raise ValueError("Clear expects symbols")
    return None


def _h_rule(a, ev, expr):
    if len(a) != 2:
        return expr
    return Expr("Rule", ev(a[0]), ev(a[1]))


def _h_ruledelayed(a, ev, expr):
    if len(a) != 2:
        return expr
    return Expr("RuleDelayed", ev(a[0]), a[1])


def _h_if(a, ev, expr):
    if not 2 <= len(a) <= 4:
        raise ValueError("If expects 2, 3 or 4 arguments")
    c = ev(a[0])
    if c is True:
        return ev(a[1])
    if c is False:
        return ev(a[2]) if len(a) >= 3 else None
    if len(a) == 4:
        return ev(a[3])
    return Expr("If", c, *a[1:])


def _h_which(a, ev, expr):
    if len(a) % 2:
        raise ValueError("Which expects an even number of arguments")
    for i in range(0, len(a), 2):
        c = ev(a[i])
        if c is True:
            return ev(a[i + 1])
        if c is not False:
            return Expr("Which", c, *a[i + 1:])
    return None


def _h_switch(a, ev, expr):
    if len(a) < 3 or len(a) % 2 == 0:
        raise ValueError("Switch expects Switch[expr, form1, value1, ...]")
    value = ev(a[0])
    for i in range(1, len(a), 2):
        b = first_match(a[i], value, ev)
        if b is not None:
            return ev(a[i + 1])
    return Expr("Switch", value, *a[1:])


def _h_do(a, ev, expr):
    if len(a) < 2:
        raise ValueError("Do expects Do[body, iterator...]")
    try:
        _do(a[0], list(a[1:]), ev)
    except BreakSignal:
        pass
    return None


def _h_while(a, ev, expr):
    if not 1 <= len(a) <= 2:
        raise ValueError("While expects While[test, body]")
    for _ in range(MAX_ITERATIONS):
        c = ev(a[0])
        if c is not True:
            return None
        if len(a) == 2:
            try:
                ev(a[1])
            except BreakSignal:
                return None
            except ContinueSignal:
                continue
    raise RecursionError(f"While exceeded {MAX_ITERATIONS} iterations")


def _h_for(a, ev, expr):
    if not 3 <= len(a) <= 4:
        raise ValueError("For expects For[start, test, incr, body]")
    ev(a[0])
    for _ in range(MAX_ITERATIONS):
        if ev(a[1]) is not True:
            return None
        try:
            if len(a) == 4:
                ev(a[3])
        except BreakSignal:
            return None
        except ContinueSignal:
            pass
        ev(a[2])
    raise RecursionError(f"For exceeded {MAX_ITERATIONS} iterations")


def _h_table(a, ev, expr):
    if not a:
        raise ValueError("Table expects Table[expr, iterator...]")
    if len(a) == 1:
        raise ValueError("Table needs at least one iterator")
    return _table(a[0], list(a[1:]), ev)


def _h_module(a, ev, expr):
    if len(a) != 2:
        raise ValueError("Module expects Module[{vars}, body]")
    return _module(a[0], a[1], ev)


def _h_block(a, ev, expr):
    if len(a) != 2:
        raise ValueError("Block expects Block[{vars}, body]")
    return _block(a[0], a[1], ev)


def _h_with(a, ev, expr):
    if len(a) != 2:
        raise ValueError("With expects With[{x = v, ...}, body]")
    return _with(a[0], a[1], ev)


def _h_setattributes(a, ev, expr, clear=False):
    if len(a) != 2:
        raise ValueError("SetAttributes expects SetAttributes[symbol, attrs]")
    syms = a[0].args if is_list(a[0]) else (a[0],)
    attrs = ev(a[1])
    names = [x.name for x in (attrs.args if is_list(attrs) else (attrs,)) if isinstance(x, Symbol)]
    if len(names) != (len(attrs.args) if is_list(attrs) else 1):
        raise ValueError("Attributes must be symbols")
    for n in names:
        if n not in KNOWN_ATTRIBUTES:
            raise ValueError(f"Unknown attribute {n}")
        if n not in SUPPORTED_USER_ATTRIBUTES:
            raise NotImplementedError(f"Attribute {n} is not supported")
    st = state()
    for s in syms:
        if not isinstance(s, Symbol):
            raise ValueError("SetAttributes expects symbols")
        current = st.attrs.setdefault(s.name, set())
        if clear:
            current.difference_update(names)
        else:
            current.update(names)
    return None


def _h_attributes(a, ev, expr):
    if len(a) != 1 or not isinstance(a[0], Symbol):
        raise ValueError("Attributes expects a symbol")
    return L(*(Symbol(n) for n in sorted(attributes(a[0].name))))


def _h_increment(op, post):
    def f(a, ev, expr):
        return _modify(a[0], op, 1, ev, post=post)
    return f


def _h_modify(op):
    def f(a, ev, expr):
        return _modify(a[0], op, ev(a[1]), ev)
    return f


def _h_replace_all(a, ev, expr):
    if len(a) != 2:
        raise ValueError("ReplaceAll expects expr /. rules")
    return _replace(a[0], a[1], ev)


def _h_replace_repeated(a, ev, expr):
    if len(a) != 2:
        raise ValueError("ReplaceRepeated expects expr //. rules")
    return _replace(a[0], a[1], ev, repeated=True)


def _h_catch(a, ev, expr):
    try:
        return ev(a[0])
    except ThrowSignal as t:
        if len(a) >= 2:
            if t.tag is None or first_match(a[1], t.tag, ev) is None:
                raise
        elif t.tag is not None:
            raise
        return t.value


def _h_break(a, ev, expr):
    raise BreakSignal()


def _h_continue(a, ev, expr):
    raise ContinueSignal()


def _h_return(a, ev, expr):
    raise ReturnSignal(ev(a[0]) if a else None)


def _h_throw(a, ev, expr):
    raise ThrowSignal(ev(a[0]), ev(a[1]) if len(a) > 1 else None)


def _h_and(a, ev, expr):
    rest = []
    for x in a:
        v = ev(x)
        if v is False:
            return False
        if v is not True:
            rest.append(v)
    if not rest:
        return True
    return rest[0] if len(rest) == 1 else Expr("And", *rest)


def _h_or(a, ev, expr):
    rest = []
    for x in a:
        v = ev(x)
        if v is True:
            return True
        if v is not False:
            rest.append(v)
    if not rest:
        return False
    return rest[0] if len(rest) == 1 else Expr("Or", *rest)


_HELD = {
    "CompoundExpression": _h_compound, "Set": _h_set, "SetDelayed": _h_setdelayed, "Unset": _h_unset,
    "Clear": _h_clear, "ClearAll": lambda a, ev, e: _h_clear(a, ev, e, all_=True),
    "Rule": _h_rule, "RuleDelayed": _h_ruledelayed, "If": _h_if, "Which": _h_which, "Switch": _h_switch,
    "Do": _h_do, "While": _h_while, "For": _h_for, "Table": _h_table, "Module": _h_module,
    "Block": _h_block, "With": _h_with, "SetAttributes": _h_setattributes,
    "ClearAttributes": lambda a, ev, e: _h_setattributes(a, ev, e, clear=True),
    "Attributes": _h_attributes, "Increment": _h_increment("Plus", True),
    "Decrement": _h_increment("Subtract", True), "PreIncrement": _h_increment("Plus", False),
    "PreDecrement": _h_increment("Subtract", False), "AddTo": _h_modify("Plus"),
    "SubtractFrom": _h_modify("Subtract"), "TimesBy": _h_modify("Times"), "DivideBy": _h_modify("Divide"),
    "ReplaceAll": _h_replace_all, "ReplaceRepeated": _h_replace_repeated, "Catch": _h_catch,
    "Break": _h_break, "Continue": _h_continue, "Return": _h_return, "Throw": _h_throw,
    "And": _h_and, "Or": _h_or,
}

# ─────────────────────────── builtins on evaluated arguments ───────────────────────────

_BUILTINS = {}


def builtin(name, *names):
    def deco(fn):
        for n in (name,) + names:
            _BUILTINS[n] = fn
        return fn
    return deco


def apply_builtin(expr, ev):
    fn = _BUILTINS.get(expr.head) if isinstance(expr.head, str) else None
    if fn is None:
        return UNHANDLED
    return fn(expr.args, ev, expr)


def _need(args, n, name, upto=None):
    if len(args) < n or len(args) > (upto if upto is not None else n):
        exp = f"{n}" if upto is None or upto == n else f"{n}-{upto}"
        raise ValueError(f"{name} expects {exp} argument(s), got {len(args)}")


def _listarg(x, name):
    if not isinstance(x, Expr):
        raise ValueError(f"{name}: {x!r} is not a list or expression")
    return x


# arithmetic ---------------------------------------------------------------


def _sympy_arith(expr):
    try:
        from .math_ops import to_sympy, from_sympy
        return from_sympy(to_sympy(expr))
    except (ValueError, TypeError, AttributeError):
        return expr


@builtin("Plus")
def _b_plus(a, ev, expr):
    if not a:
        return 0
    if len(a) == 1:
        return a[0]
    return _sympy_arith(expr)


@builtin("Times")
def _b_times(a, ev, expr):
    if not a:
        return 1
    if len(a) == 1:
        return a[0]
    return _sympy_arith(expr)


@builtin("Power")
def _b_power(a, ev, expr):
    if len(a) != 2:
        raise ValueError("Power expects 2 arguments")
    return _sympy_arith(expr)


@builtin("Subtract")
def _b_subtract(a, ev, expr):
    _need(a, 2, "Subtract")
    return ev(Expr("Plus", a[0], Expr("Times", -1, a[1])))


@builtin("Divide")
def _b_divide(a, ev, expr):
    _need(a, 2, "Divide")
    return ev(Expr("Times", a[0], Expr("Power", a[1], -1)))


@builtin("Sequence")
def _b_sequence(a, ev, expr):
    return expr


# structural ----------------------------------------------------------------


@builtin("Length")
def _b_length(a, ev, expr):
    _need(a, 1, "Length")
    from .formatting import rational_value
    x = a[0]
    if not isinstance(x, Expr) or rational_value(x) is not None:
        return 0
    return len(x.args)


@builtin("Head")
def _b_head(a, ev, expr):
    _need(a, 1, "Head")
    from .formatting import rational_value
    x = a[0]
    if isinstance(x, bool) or isinstance(x, Symbol):
        return Symbol("Complex") if isinstance(x, Symbol) and x.name == "I" else Symbol("Symbol")
    if is_int(x):
        return Symbol("Integer")
    if isinstance(x, (float, sp.Float)):
        return Symbol("Real")
    if isinstance(x, complex):
        return Symbol("Complex")
    if isinstance(x, str):
        return Symbol("String")
    if rational_value(x) is not None:
        return Symbol("Rational")
    if isinstance(x, Expr):
        return Symbol(x.head) if isinstance(x.head, str) else x.head
    raise ValueError(f"Head: unsupported value {x!r}")


def _part1(x, i):
    if is_int(i):
        i = int(i)
        if i == 0:
            return Symbol(x.head) if isinstance(x.head, str) else x.head
        n = len(x.args)
        k = i - 1 if i > 0 else n + i
        if not 0 <= k < n:
            raise ValueError(f"Part {i} of {x!r} does not exist")
        return x.args[k]
    raise ValueError("bad part index")


def _part(x, idx):
    if not idx:
        return x
    if not isinstance(x, Expr):
        raise ValueError(f"Part specification is longer than depth of object {x!r}")
    i = idx[0]
    if isinstance(i, Symbol) and i.name == "All":
        return Expr(x.head, *(_part(y, idx[1:]) for y in x.args))
    if is_list(i):
        return Expr(x.head, *(_part(_part1(x, j), idx[1:]) for j in i.args))
    if is_int(i):
        return _part(_part1(x, i), idx[1:])
    raise NotImplementedError(f"Part index {i!r} is not supported (use integers, lists or All)")


@builtin("Part")
def _b_part(a, ev, expr):
    if len(a) < 2:
        raise ValueError("Part expects expr[[i, ...]]")
    return _part(a[0], list(a[1:]))


@builtin("First")
def _b_first(a, ev, expr):
    _need(a, 1, "First")
    return _part(_listarg(a[0], "First"), [1])


@builtin("Last")
def _b_last(a, ev, expr):
    _need(a, 1, "Last")
    return _part(_listarg(a[0], "Last"), [-1])


@builtin("Rest")
def _b_rest(a, ev, expr):
    _need(a, 1, "Rest")
    x = _listarg(a[0], "Rest")
    if not x.args:
        raise ValueError("Rest: expression has no elements")
    return Expr(x.head, *x.args[1:])


@builtin("Most")
def _b_most(a, ev, expr):
    _need(a, 1, "Most")
    x = _listarg(a[0], "Most")
    if not x.args:
        raise ValueError("Most: expression has no elements")
    return Expr(x.head, *x.args[:-1])


@builtin("Append")
def _b_append(a, ev, expr):
    _need(a, 2, "Append")
    x = _listarg(a[0], "Append")
    return Expr(x.head, *x.args, a[1])


@builtin("Prepend")
def _b_prepend(a, ev, expr):
    _need(a, 2, "Prepend")
    x = _listarg(a[0], "Prepend")
    return Expr(x.head, a[1], *x.args)


@builtin("Join")
def _b_join(a, ev, expr):
    if not a:
        return L()
    heads = {x.head if isinstance(x, Expr) else None for x in a}
    if len(heads) != 1 or None in heads:
        raise ValueError("Join: all arguments must be expressions with the same head")
    return Expr(a[0].head, *(y for x in a for y in x.args))


@builtin("Reverse")
def _b_reverse(a, ev, expr):
    _need(a, 1, "Reverse")
    x = _listarg(a[0], "Reverse")
    return Expr(x.head, *reversed(x.args))


def _span(spec, n, name):
    if is_int(spec):
        k = int(spec)
        if abs(k) > n:
            raise ValueError(f"{name}: cannot take {k} elements from {n}")
        return (0, k) if k >= 0 else (n + k, n)
    if is_list(spec) and len(spec.args) == 1 and is_int(spec.args[0]):
        k = int(spec.args[0])
        k0 = k - 1 if k > 0 else n + k
        if not 0 <= k0 < n:
            raise ValueError(f"{name}: position {k} out of range")
        return (k0, k0 + 1)
    if is_list(spec) and len(spec.args) == 2 and all(is_int(v) for v in spec.args):
        m, k = (int(v) for v in spec.args)
        m0 = m - 1 if m > 0 else n + m
        k1 = k if k > 0 else n + k + 1
        if not (0 <= m0 <= n and 0 <= k1 <= n):
            raise ValueError(f"{name}: span {m}..{k} out of range")
        return (m0, k1)
    if isinstance(spec, Symbol) and spec.name == "All":
        return (0, n)
    raise NotImplementedError(f"{name}: unsupported sequence specification {spec!r}")


@builtin("Take")
def _b_take(a, ev, expr):
    _need(a, 2, "Take")
    x = _listarg(a[0], "Take")
    i, j = _span(a[1], len(x.args), "Take")
    return Expr(x.head, *x.args[i:j])


@builtin("Drop")
def _b_drop(a, ev, expr):
    _need(a, 2, "Drop")
    x = _listarg(a[0], "Drop")
    i, j = _span(a[1], len(x.args), "Drop")
    return Expr(x.head, *(x.args[:i] + x.args[j:]))


@builtin("Sort")
def _b_sort(a, ev, expr):
    _need(a, 1, "Sort", 2)
    x = _listarg(a[0], "Sort")
    if len(a) == 2:
        p = a[1]

        def cmp(u, v):
            if ev(call(p, u, v)) is True:
                return -1
            if ev(call(p, v, u)) is True:
                return 1
            return 0
        return Expr(x.head, *sorted(x.args, key=functools.cmp_to_key(cmp)))
    if not all(_sortable_canonically(e) for e in x.args):
        raise NotImplementedError("Sort: canonical ordering of general symbolic expressions is not "
                                  "implemented; pass an ordering function, e.g. Sort[list, Less]")
    return Expr(x.head, *sorted(x.args, key=order_key))


@builtin("Total")
def _b_total(a, ev, expr):
    _need(a, 1, "Total", 2)
    x = _listarg(a[0], "Total")
    depth = 1
    if len(a) == 2:
        if isinstance(a[1], Symbol) and a[1].name == "Infinity":
            depth = 10 ** 6
        elif is_int(a[1]) and int(a[1]) >= 1:
            depth = int(a[1])
        else:
            raise NotImplementedError("Total: level specification must be a positive integer or Infinity")
    items = list(x.args)
    for _ in range(depth - 1):
        if not any(is_list(i) for i in items):
            break
        items = [y for i in items for y in (i.args if is_list(i) else (i,))]
    return ev(Expr("Plus", *items)) if items else 0


@builtin("Flatten")
def _b_flatten(a, ev, expr):
    _need(a, 1, "Flatten", 2)
    x = _listarg(a[0], "Flatten")
    depth = 10 ** 6
    if len(a) == 2:
        if not is_int(a[1]):
            raise NotImplementedError("Flatten: only integer levels are supported")
        depth = int(a[1])

    def flat(e, d):
        out = []
        for y in e.args:
            if d > 0 and isinstance(y, Expr) and y.head == e.head:
                out.extend(flat(y, d - 1))
            else:
                out.append(y)
        return out
    return Expr(x.head, *flat(x, depth))


def _level_spec(spec):
    """Return (min_level, max_level)."""
    if spec is None:
        return (1, 1)
    if isinstance(spec, Symbol) and spec.name == "Infinity":
        return (1, 10 ** 9)
    if is_int(spec):
        return (1, int(spec))
    if is_list(spec) and len(spec.args) == 1 and is_int(spec.args[0]):
        return (int(spec.args[0]), int(spec.args[0]))
    if is_list(spec) and len(spec.args) == 2 and all(is_int(v) for v in spec.args):
        return (int(spec.args[0]), int(spec.args[1]))
    raise NotImplementedError(f"Unsupported level specification {spec!r}")


def _walk(e, lo, hi, level=0, pos=()):
    """Post-order traversal yielding (position, subexpression) within the level range."""
    if isinstance(e, Expr) and level < hi:
        for i, x in enumerate(e.args, 1):
            yield from _walk(x, lo, hi, level + 1, pos + (i,))
    if lo <= level <= hi:
        yield pos, e


def _pattern_and_rhs(p):
    if isinstance(p, Expr) and p.head in ("Rule", "RuleDelayed") and len(p.args) == 2:
        return p.args[0], p.args[1]
    return p, None


@builtin("Cases")
def _b_cases(a, ev, expr):
    _need(a, 2, "Cases", 3)
    x = _listarg(a[0], "Cases")
    pat, rhs = _pattern_and_rhs(a[1])
    lo, hi = _level_spec(a[2] if len(a) == 3 else None)
    out = []
    for _, sub in _walk(x, lo, hi):
        b = first_match(pat, sub, ev)
        if b is not None:
            out.append(sub if rhs is None else ev(substitute(rhs, b)))
    return L(*out)


@builtin("DeleteCases")
def _b_deletecases(a, ev, expr):
    _need(a, 2, "DeleteCases")
    x = _listarg(a[0], "DeleteCases")
    return Expr(x.head, *(y for y in x.args if first_match(a[1], y, ev) is None))


@builtin("Count")
def _b_count(a, ev, expr):
    _need(a, 2, "Count", 3)
    x = _listarg(a[0], "Count")
    lo, hi = _level_spec(a[2] if len(a) == 3 else None)
    return sum(1 for _, sub in _walk(x, lo, hi) if first_match(a[1], sub, ev) is not None)


@builtin("Position")
def _b_position(a, ev, expr):
    _need(a, 2, "Position")
    x = a[0]
    out = []

    def visit(e, pos):
        if isinstance(e, Expr):
            hd = Symbol(e.head) if isinstance(e.head, str) else e.head
            if first_match(a[1], hd, ev) is not None:
                out.append(pos + (0,))
            for i, y in enumerate(e.args, 1):
                visit(y, pos + (i,))
        if first_match(a[1], e, ev) is not None:
            out.append(pos)
    visit(x, ())
    return L(*(L(*p) for p in out))


@builtin("MemberQ")
def _b_memberq(a, ev, expr):
    _need(a, 2, "MemberQ")
    x = a[0]
    if not isinstance(x, Expr):
        return False
    return any(first_match(a[1], y, ev) is not None for y in x.args)


@builtin("FreeQ")
def _b_freeq(a, ev, expr):
    _need(a, 2, "FreeQ")
    return not any(first_match(a[1], sub, ev) is not None for _, sub in _walk(a[0], 0, 10 ** 9))


@builtin("MatchQ")
def _b_matchq(a, ev, expr):
    _need(a, 2, "MatchQ")
    return first_match(a[1], a[0], ev) is not None


@builtin("SameQ")
def _b_sameq(a, ev, expr):
    return all(same(a[0], y) for y in a[1:])


@builtin("UnsameQ")
def _b_unsameq(a, ev, expr):
    return not all(same(a[0], y) for y in a[1:])


@builtin("Equal", "Unequal")
def _b_equal(a, ev, expr):
    """Structural/list/string equality; numeric & symbolic cases fall through to SymPy."""
    if len(a) != 2:
        return UNHANDLED
    x, y = a
    eq = None
    if same(x, y):
        eq = True
    elif isinstance(x, str) or isinstance(y, str):
        eq = isinstance(x, str) and isinstance(y, str) and x == y
    elif isinstance(x, bool) or isinstance(y, bool):
        eq = x is y if isinstance(x, bool) and isinstance(y, bool) else None
    elif is_list(x) or is_list(y):
        if is_list(x) and is_list(y):
            if len(x.args) != len(y.args):
                eq = False
            else:
                parts = [ev(Expr("Equal", u, v)) for u, v in zip(x.args, y.args)]
                if all(p is True for p in parts):
                    eq = True
                elif any(p is False for p in parts):
                    eq = False
        else:
            eq = None
    if eq is None:
        return UNHANDLED
    return eq if expr.head == "Equal" else not eq


# functional -----------------------------------------------------------------


@builtin("Map")
def _b_map(a, ev, expr):
    _need(a, 2, "Map", 3)
    f, x = a[0], a[1]
    if len(a) == 3:
        lo, hi = _level_spec(a[2])
        if (lo, hi) != (1, 1):
            raise NotImplementedError("Map: only level {1} is supported")
    if not isinstance(x, Expr):
        return x
    return ev(Expr(x.head, *(call(f, y) for y in x.args)))


@builtin("Apply")
def _b_apply(a, ev, expr):
    _need(a, 2, "Apply", 3)
    f, x = a[0], a[1]
    if len(a) == 3:
        lo, hi = _level_spec(a[2])
        if (lo, hi) != (1, 1):
            raise NotImplementedError("Apply: only levels {0} and {1} are supported")
        if not isinstance(x, Expr):
            return x
        return ev(Expr(x.head, *(call(f, *y.args) if isinstance(y, Expr) else y for y in x.args)))
    if not isinstance(x, Expr):
        return x
    return ev(call(f, *x.args))


@builtin("MapThread")
def _b_mapthread(a, ev, expr):
    _need(a, 2, "MapThread")
    lists = _listarg(a[1], "MapThread").args
    if not lists or not all(is_list(l) for l in lists):
        raise ValueError("MapThread expects a list of lists")
    n = len(lists[0].args)
    if any(len(l.args) != n for l in lists):
        raise ValueError("MapThread: lists must have equal length")
    return ev(L(*(call(a[0], *(l.args[i] for l in lists)) for i in range(n))))


@builtin("MapIndexed")
def _b_mapindexed(a, ev, expr):
    _need(a, 2, "MapIndexed")
    x = _listarg(a[1], "MapIndexed")
    return ev(Expr(x.head, *(call(a[0], y, L(i)) for i, y in enumerate(x.args, 1))))


@builtin("Select")
def _b_select(a, ev, expr):
    _need(a, 2, "Select", 3)
    x = _listarg(a[0], "Select")
    limit = None
    if len(a) == 3:
        if not is_int(a[2]) or int(a[2]) < 0:
            raise ValueError("Select: the third argument must be a non-negative integer")
        limit = int(a[2])
    out = []
    for y in x.args:
        if limit is not None and len(out) >= limit:
            break
        if ev(call(a[1], y)) is True:
            out.append(y)
    return Expr(x.head, *out)


def _count_arg(n, name):
    if not is_int(n) or int(n) < 0:
        raise ValueError(f"{name}: iteration count must be a non-negative integer")
    if int(n) > MAX_ITERATIONS:
        raise ValueError(f"{name}: iteration count too large")
    return int(n)


@builtin("Nest")
def _b_nest(a, ev, expr):
    _need(a, 3, "Nest")
    x = a[1]
    for _ in range(_count_arg(a[2], "Nest")):
        x = ev(call(a[0], x))
    return x


@builtin("NestList")
def _b_nestlist(a, ev, expr):
    _need(a, 3, "NestList")
    x = a[1]
    out = [x]
    for _ in range(_count_arg(a[2], "NestList")):
        x = ev(call(a[0], x))
        out.append(x)
    return L(*out)


def _fixed_point(a, ev, keep):
    x = a[1]
    limit = _count_arg(a[2], "FixedPoint") if len(a) >= 3 else 65536
    out = [x]
    for _ in range(limit):
        y = ev(call(a[0], x))
        out.append(y)
        if same(x, y) or (is_number(x) and is_number(y) and ev(Expr("Equal", x, y)) is True):
            return L(*out) if keep else y
        x = y
    if len(a) >= 3:
        return L(*out) if keep else x
    raise RecursionError("FixedPoint did not converge in 65536 steps")


@builtin("FixedPoint")
def _b_fixedpoint(a, ev, expr):
    _need(a, 2, "FixedPoint", 3)
    return _fixed_point(a, ev, False)


@builtin("FixedPointList")
def _b_fixedpointlist(a, ev, expr):
    _need(a, 2, "FixedPointList", 3)
    return _fixed_point(a, ev, True)


def _fold_args(a, name):
    _need(a, 2, name, 3)
    if len(a) == 2:
        x = _listarg(a[1], name)
        if not x.args:
            raise ValueError(f"{name}[f, {{}}] has no starting value")
        return a[0], x.args[0], x.args[1:]
    return a[0], a[1], _listarg(a[2], name).args


@builtin("Fold")
def _b_fold(a, ev, expr):
    f, acc, items = _fold_args(a, "Fold")
    for y in items:
        acc = ev(call(f, acc, y))
    return acc


@builtin("FoldList")
def _b_foldlist(a, ev, expr):
    f, acc, items = _fold_args(a, "FoldList")
    out = [acc]
    for y in items:
        acc = ev(call(f, acc, y))
        out.append(acc)
    return L(*out)


@builtin("Identity")
def _b_identity(a, ev, expr):
    _need(a, 1, "Identity")
    return a[0]


@builtin("Thread")
def _b_thread(a, ev, expr):
    _need(a, 1, "Thread")
    x = a[0]
    if not isinstance(x, Expr):
        return x
    lens = {len(y.args) for y in x.args if is_list(y)}
    if not lens:
        return x
    if len(lens) != 1:
        raise ValueError("Thread: lists must have equal length")
    n = lens.pop()
    return ev(L(*(Expr(x.head, *(y.args[i] if is_list(y) else y for y in x.args)) for i in range(n))))


def _outer(f, lists, ev):
    first, rest = lists[0], lists[1:]
    if is_list(first):
        return L(*(_outer(f, [y] + rest, ev) for y in first.args))
    if not rest:
        return [first]
    inner = _outer(f, rest, ev)
    return _bind_outer(f, first, inner)


def _bind_outer(f, x, inner):
    if isinstance(inner, list):
        return call(f, x, *inner)
    return L(*(_bind_outer(f, x, y) for y in inner.args))


@builtin("Outer")
def _b_outer(a, ev, expr):
    if len(a) < 3:
        raise ValueError("Outer expects Outer[f, list1, list2, ...]")
    if not all(is_list(x) for x in a[1:]):
        raise ValueError("Outer: arguments must be lists")
    return ev(_outer(a[0], list(a[1:]), ev))


def _depth(x):
    return 1 + max((_depth(y) for y in x.args), default=0) if is_list(x) else 0


@builtin("Inner")
def _b_inner(a, ev, expr):
    _need(a, 3, "Inner", 4)
    f, u, v = a[0], a[1], a[2]
    g = a[3] if len(a) == 4 else Symbol("Plus")
    if not (is_list(u) and is_list(v)):
        raise ValueError("Inner: arguments must be lists")
    du, dv = _depth(u), _depth(v)
    if du > 2 or dv > 2:
        raise NotImplementedError("Inner is implemented for vectors and matrices only")
    rows = u.args if du == 2 else None
    cols_src = v.args
    n = len(cols_src)

    def vec_inner(row):
        if len(row) != n:
            raise ValueError("Inner: incompatible dimensions")
        if dv == 1:
            return call(g, *(call(f, row[k], cols_src[k]) for k in range(n)))
        m = len(cols_src[0].args)
        return L(*(call(g, *(call(f, row[k], cols_src[k].args[j]) for k in range(n))) for j in range(m)))
    if rows is None:
        return ev(vec_inner(u.args))
    return ev(L(*(vec_inner(r.args) for r in rows)))


# strings ----------------------------------------------------------------------


@builtin("StringJoin")
def _b_stringjoin(a, ev, expr):
    parts = []

    def collect(x):
        if isinstance(x, str):
            parts.append(x)
            return True
        if is_list(x):
            return all(collect(y) for y in x.args)
        return False
    if not all(collect(x) for x in a):
        raise TypeError("StringJoin: all arguments must be strings")
    return "".join(parts)


@builtin("ToString")
def _b_tostring(a, ev, expr):
    _need(a, 1, "ToString", 2)
    from .formatting import input_form, full_form
    x = a[0]
    if isinstance(x, str):
        return x
    if len(a) == 2 and isinstance(a[1], Symbol) and a[1].name == "FullForm":
        return full_form(x)
    return input_form(x)


@builtin("StringLength")
def _b_stringlength(a, ev, expr):
    _need(a, 1, "StringLength")
    if not isinstance(a[0], str):
        raise TypeError("StringLength expects a string")
    return len(a[0])


@builtin("InputForm", "FullForm")
def _b_forms(a, ev, expr):
    _need(a, 1, expr.head)
    return expr


# holding ------------------------------------------------------------------------


_HOLDS = {"Hold", "HoldForm", "HoldComplete", "HoldPattern"}


@builtin("ReleaseHold")
def _b_releasehold(a, ev, expr):
    _need(a, 1, "ReleaseHold")

    def strip(x):
        if isinstance(x, Expr):
            if x.head in _HOLDS:
                return x.args[0] if len(x.args) == 1 else Expr("Sequence", *x.args)
            return Expr(x.head, *(strip(y) for y in x.args))
        return x
    return ev(strip(a[0]))


@builtin("Evaluate")
def _b_evaluate(a, ev, expr):
    return a[0] if len(a) == 1 else Expr("Sequence", *a)


# predicates -----------------------------------------------------------------------


def _num(x):
    try:
        v = to_sp(x)
        return v if getattr(v, "is_number", False) else None
    except Exception:
        return None


@builtin("IntegerQ")
def _b_integerq(a, ev, expr):
    return is_int(a[0])


@builtin("EvenQ")
def _b_evenq(a, ev, expr):
    return is_int(a[0]) and int(a[0]) % 2 == 0


@builtin("OddQ")
def _b_oddq(a, ev, expr):
    return is_int(a[0]) and int(a[0]) % 2 == 1


@builtin("NumberQ")
def _b_numberq(a, ev, expr):
    return is_number(a[0])


@builtin("NumericQ")
def _b_numericq(a, ev, expr):
    v = _num(a[0])
    return v is not None and not isinstance(a[0], str)


@builtin("StringQ")
def _b_stringq(a, ev, expr):
    return isinstance(a[0], str)


@builtin("ListQ")
def _b_listq(a, ev, expr):
    return is_list(a[0])


@builtin("AtomQ")
def _b_atomq(a, ev, expr):
    from .formatting import rational_value
    return not isinstance(a[0], Expr) or rational_value(a[0]) is not None


@builtin("TrueQ")
def _b_trueq(a, ev, expr):
    return a[0] is True


def _sign_pred(name, test):
    @builtin(name)
    def f(a, ev, expr):
        _need(a, 1, name)
        v = _num(a[0])
        if v is None or not v.is_real:
            if v is not None and v.is_real is False:
                return False
            return expr
        return bool(test(v))
    return f


_sign_pred("Positive", lambda v: v > 0)
_sign_pred("Negative", lambda v: v < 0)
_sign_pred("NonNegative", lambda v: v >= 0)
_sign_pred("NonPositive", lambda v: v <= 0)


@builtin("PrimeQ")
def _b_primeq(a, ev, expr):
    return is_int(a[0]) and bool(sp.isprime(abs(int(a[0]))))


@builtin("Not")
def _b_not(a, ev, expr):
    _need(a, 1, "Not")
    if a[0] is True:
        return False
    if a[0] is False:
        return True
    return expr


# calculus & algebra -----------------------------------------------------------------


def _equations(e):
    """Flatten an equation, list of equations or And[...] into a list."""
    if is_list(e) or (isinstance(e, Expr) and e.head == "And"):
        return [x for y in e.args for x in _equations(y)]
    return [e]


def _to_eq(e):
    if e is True or e is False:
        return e
    if isinstance(e, Expr) and e.head == "Equal" and len(e.args) == 2:
        return sp.Eq(to_sp(e.args[0]), to_sp(e.args[1]), evaluate=False)
    if isinstance(e, Expr) and e.head in ("Less", "Greater", "LessEqual", "GreaterEqual", "Unequal"):
        raise NotImplementedError("Solve with inequalities is not supported; use Reduce-style tools")
    if isinstance(e, str):
        raise TypeError("Solve: strings are not equations")
    # Convenience (SymPy convention): a bare expression means expr == 0.
    return sp.Eq(to_sp(e), 0, evaluate=False)


@builtin("Solve")
def _b_solve(a, ev, expr):
    _need(a, 1, "Solve", 3)
    if len(a) == 3:
        raise NotImplementedError("Solve with a domain argument is not supported")
    eqs = [_to_eq(e) for e in _equations(a[0])]
    if any(e is False for e in eqs):
        return L()
    eqs = [e for e in eqs if e is not True]
    if len(a) == 2:
        vs = a[1].args if is_list(a[1]) else (a[1],)
    else:
        syms = set()
        for e in eqs:
            syms |= e.free_symbols
        vs = tuple(Symbol(str(s)) for s in sorted(syms, key=str))
    if not vs:
        raise ValueError("Solve: no variables to solve for")
    svars = [to_sp(v) for v in vs]
    if not eqs:
        return L(L())
    for e in eqs:
        if isinstance(e, sp.Basic) and e.has(sp.Float):
            pass
    try:
        sols = sp.solve(eqs, svars, dict=True)
    except NotImplementedError as exc:
        raise NotImplementedError(f"Solve: SymPy cannot solve this system ({exc})") from exc
    if not isinstance(sols, list):
        raise NotImplementedError("Solve: unexpected solution form from SymPy")
    out = []
    for sol in sols:
        rules = [Expr("Rule", v, from_sp(sol[sv])) for v, sv in zip(vs, svars) if sv in sol]
        out.append(L(*rules))
    out = [r for r in out if r.args or not sols]
    return ev(L(*out))


def _constants_to_c(e):
    """SymPy integration constants C1, C2 -> C[1], C[2]."""
    if isinstance(e, Symbol) and len(e.name) > 1 and e.name[0] == "C" and e.name[1:].isdigit():
        return Expr("C", int(e.name[1:]))
    if isinstance(e, Expr):
        h = _constants_to_c(e.head) if isinstance(e.head, Expr) else e.head
        return Expr(h, *(_constants_to_c(x) for x in e.args))
    return e


def _func_name(f):
    if isinstance(f, Symbol):
        return f.name
    if isinstance(f, Expr) and isinstance(f.head, str):
        return f.head
    raise ValueError(f"DSolve: cannot interpret {f!r} as an unknown function")


@builtin("DSolve")
def _b_dsolve(a, ev, expr):
    _need(a, 3, "DSolve")
    eqs = _equations(a[0])
    funcs = list(a[1].args) if is_list(a[1]) else [a[1]]
    var = a[2]
    if not isinstance(var, Symbol):
        raise NotImplementedError("DSolve: only ODEs in one independent variable are supported")
    x = to_sp(var)
    names = [_func_name(f) for f in funcs]
    sfuncs = [sp.Function(n)(x) for n in names]
    odes, ics = [], {}
    for e in eqs:
        if not (isinstance(e, Expr) and e.head == "Equal" and len(e.args) == 2):
            raise ValueError(f"DSolve expects equations, got {e!r}")
        lhs, rhs = e.args
        ic = _initial_condition(lhs, names, x)
        if ic is not None:
            ics[ic] = to_sp(rhs)
            continue
        odes.append(sp.Eq(to_sp(lhs), to_sp(rhs)))
    if not odes:
        raise ValueError("DSolve: no differential equation given")
    try:
        if len(odes) == 1 and len(sfuncs) == 1:
            res = sp.dsolve(odes[0], sfuncs[0], ics=ics or None)
        else:
            res = sp.dsolve(odes, sfuncs, ics=ics or None)
    except NotImplementedError as exc:
        raise NotImplementedError(f"DSolve: SymPy cannot solve this equation ({exc})") from exc
    solutions = res if isinstance(res, list) else [res]
    # A system gives one list of equations; a single ODE may give several alternatives.
    if len(sfuncs) > 1:
        groups = [solutions]
    else:
        groups = [[s] for s in solutions]
    out = []
    for grp in groups:
        rules = []
        for s in grp:
            fname = type(s.lhs).__name__ if not isinstance(s.lhs, sp.Derivative) else None
            if fname is None:
                raise NotImplementedError("DSolve: implicit solution not supported")
            value = _constants_to_c(from_sp(s.rhs))
            orig = funcs[names.index(fname)]
            if isinstance(orig, Symbol):
                rules.append(Expr("Rule", orig, Expr("Function", L(var), value)))
            else:
                rules.append(Expr("Rule", orig, value))
        out.append(L(*rules))
    return L(*out)


def _initial_condition(lhs, names, x):
    """y[0] or y'[0] (argument not the independent variable) -> SymPy ics key."""
    if isinstance(lhs, Expr) and isinstance(lhs.head, str) and lhs.head in names and len(lhs.args) == 1:
        arg = lhs.args[0]
        if not (isinstance(arg, Symbol) and to_sp(arg) == x):
            return sp.Function(lhs.head)(to_sp(arg))
    if (isinstance(lhs, Expr) and isinstance(lhs.head, Expr) and isinstance(lhs.head.head, Expr)
            and lhs.head.head.head == "Derivative" and len(lhs.args) == 1):
        f = lhs.head.args[0]
        arg = lhs.args[0]
        if isinstance(f, Symbol) and f.name in names and not (isinstance(arg, Symbol) and to_sp(arg) == x):
            n = int(lhs.head.head.args[0])
            return sp.Function(f.name)(x).diff(x, n).subs(x, to_sp(arg))
    return None


@builtin("Series")
def _b_series(a, ev, expr):
    _need(a, 2, "Series", 3)
    if len(a) == 3:
        raise NotImplementedError("Multivariate Series is not supported")
    spec = a[1]
    if not (is_list(spec) and len(spec.args) == 3 and isinstance(spec.args[0], Symbol) and is_int(spec.args[2])):
        raise ValueError("Series expects Series[f, {x, x0, n}] with integer n")
    var, x0, n = spec.args
    if int(n) < 0:
        raise ValueError("Series order must be non-negative")
    f = to_sp(a[0])
    s = sp.series(f, to_sp(var), to_sp(x0), int(n) + 1)
    return from_sp(s)


@builtin("Normal")
def _b_normal(a, ev, expr):
    _need(a, 1, "Normal")
    x = a[0]
    if not contains(x, lambda y: isinstance(y, Expr) and y.head == "O"):
        return x
    return from_sp(to_sp(x).removeO())


@builtin("Limit")
def _b_limit(a, ev, expr):
    _need(a, 2, "Limit", 3)
    if len(a) == 3:
        raise NotImplementedError("Limit options (Direction) are not supported")
    r = a[1]
    if not (isinstance(r, Expr) and r.head == "Rule" and len(r.args) == 2):
        raise ValueError("Limit expects Limit[f, x -> x0]")
    f, x, x0 = to_sp(a[0]), to_sp(r.args[0]), to_sp(r.args[1])
    if x0 in (sp.oo, -sp.oo):
        return from_sp(sp.limit(f, x, x0))
    right = sp.limit(f, x, x0, dir="+")
    left = sp.limit(f, x, x0, dir="-")
    if right != left:
        # one-sided limits differ: the two-sided limit does not exist (Wolfram: Indeterminate)
        return Symbol("Indeterminate")
    return from_sp(right)


@builtin("Inverse")
def _b_inverse(a, ev, expr):
    _need(a, 1, "Inverse")
    m = sp.Matrix(to_sp(a[0]))
    if not m.is_square:
        raise ValueError("Inverse: matrix must be square")
    if m.det() == 0:
        raise ValueError("Inverse: matrix is singular")
    return from_sp(m.inv())


@builtin("Transpose")
def _b_transpose(a, ev, expr):
    _need(a, 1, "Transpose")
    x = a[0]
    if not (is_list(x) and x.args and all(is_list(r) for r in x.args)):
        raise ValueError("Transpose expects a matrix (list of equal-length lists)")
    n = len(x.args[0].args)
    if any(len(r.args) != n for r in x.args):
        raise ValueError("Transpose: rows have different lengths")
    return L(*(L(*(r.args[j] for r in x.args)) for j in range(n)))


@builtin("Det")
def _b_det(a, ev, expr):
    _need(a, 1, "Det")
    return from_sp(sp.Matrix(to_sp(a[0])).det())


# numeric / legacy-core builtins (moved from runtime.special) ---------------------


@builtin("N")
def _b_n(a, ev, expr):
    from .numerical import numerical
    _need(a, 1, "N", 2)
    if is_list(a[0]):
        return L(*(ev(Expr("N", y, *a[1:])) for y in a[0].args))
    return numerical(a[0], a[1] if len(a) == 2 else 15)


@builtin("Range")
def _b_range(a, ev, expr):
    _need(a, 1, "Range", 3)
    for x in a:
        if not is_int(x):
            raise ValueError("Range bounds must be explicit integers")
    vals = [int(x) for x in a]
    lo, hi, step = (1, vals[0], 1) if len(vals) == 1 else (vals[0], vals[1], vals[2] if len(vals) == 3 else 1)
    if step == 0:
        raise ValueError("Range step cannot be zero")
    it = range(lo, hi + (1 if step > 0 else -1), step)
    if len(it) > MAX_TABLE:
        raise ValueError(f"Range exceeds {MAX_TABLE} element limit")
    return L(*it)


@builtin("D", "Diff", "PartialD")
def _b_d(a, ev, expr):
    if len(a) < 2:
        raise ValueError("D expects D[f, x] or D[f, {x, n}]")
    f = to_sp(a[0])
    for spec in a[1:]:
        if is_list(spec):
            if len(spec.args) != 2:
                raise ValueError("Derivative expects {variable, order}")
            var, order = spec.args
            if not is_int(order) or int(order) < 0:
                raise ValueError("Invalid derivative order")
            f = sp.diff(f, to_sp(var), int(order))
        else:
            f = sp.diff(f, to_sp(spec))
    return from_sp(f)


@builtin("Dot", "DotProduct")
def _b_dot(a, ev, expr):
    if len(a) != 2:
        raise ValueError("Dot expects two arguments")
    left, right = (to_sp(x) for x in a)
    if not isinstance(left, list) or not isinstance(right, list):
        raise ValueError("Dot expects lists")
    lm = bool(left) and isinstance(left[0], list)
    rm = bool(right) and isinstance(right[0], list)
    if not lm and not rm:
        if len(left) != len(right):
            raise ValueError("Vector dimensions differ")
        return from_sp(sp.Add(*(x * y for x, y in zip(left, right))))
    lmat = sp.Matrix(left) if lm else sp.Matrix([left])
    rmat = sp.Matrix(right) if rm else sp.Matrix(right)
    if lmat.cols != rmat.rows:
        raise ValueError("Dot: incompatible dimensions")
    result = lmat * rmat
    if lm and rm:
        return from_sp(result)
    return L(*(from_sp(x) for x in result))


@builtin("Distance2D")
def _b_distance2d(a, ev, expr):
    if len(a) != 2 or any(not is_list(x) or len(x.args) != 2 for x in a):
        raise ValueError("Distance2D expects two two-coordinate lists")
    return from_sp(sp.sqrt(sum((to_sp(x) - to_sp(y)) ** 2 for x, y in zip(a[0].args, a[1].args))))


_REL = {"Equal": sp.Eq, "Unequal": sp.Ne, "Less": sp.Lt, "Greater": sp.Gt,
        "LessEqual": sp.Le, "GreaterEqual": sp.Ge}


def _compare(a, ev, expr):
    if len(a) < 2:
        return True                       # Wolfram: Equal[x] and Less[] etc. are True
    if len(a) > 2:
        # Equal[a, b, c] / Less[a, b, c] chain adjacent pairs; Unequal[a, b, c] needs all pairs distinct
        pairs = ([(a[i], a[j]) for i in range(len(a)) for j in range(i + 1, len(a))] if expr.head == "Unequal"
                 else list(zip(a, a[1:])))
        vals = [ev(Expr(expr.head, x, y)) for x, y in pairs]
        if any(v is False for v in vals):
            return False
        if all(v is True for v in vals):
            return True
        return expr
    if expr.head in ("Equal", "Unequal"):
        r = _b_equal(a, ev, expr)
        if r is not UNHANDLED:
            return r
    try:
        result = _REL[expr.head](*(to_sp(x) for x in a))
    except (ValueError, TypeError):
        return expr
    if result is sp.true:
        return True
    if result is sp.false:
        return False
    return expr


for _name in _REL:
    _BUILTINS[_name] = _compare


@builtin("Apart")
def _b_apart(a, ev, expr):
    _need(a, 1, "Apart", 2)
    return from_sp(sp.apart(to_sp(a[0]), *(to_sp(v) for v in a[1:])))


@builtin("Together")
def _b_together(a, ev, expr):
    _need(a, 1, "Together")
    return from_sp(sp.factor_terms(sp.together(to_sp(a[0]))))


@builtin("Cancel")
def _b_cancel(a, ev, expr):
    _need(a, 1, "Cancel")
    return from_sp(sp.cancel(to_sp(a[0])))


# Wolfram level / sequence specifications, associations and operator forms (replaces several builtins above).
from . import kernel_lists  # noqa: E402,F401
