"""Types for MikoshiLang: the CONVENTIONS §2 vocabulary, signatures and an opt-in checker.

Types are the names of CONVENTIONS.md §2. Library heads take their signature from the Python annotations of
their implementations (``def _f(n: PosInteger, *opts: Option) -> Integer``); user functions declare one with
``TypeSignature[f, {Integer, Integer} -> Integer]``.

Checking is opt-in: ``Session(check_types=True)`` raises ``TypeError("f: ...")`` when an argument (or a result)
is a concrete value of the wrong kind, ``Session(check_types="report")`` records mismatches in
``session.type_mismatches`` and carries on, and ``TypeCheck[expr]`` does the same from inside the language.
Symbolic arguments (``x``, ``n + 1``, ``f[y]``) are never mismatches: a function may legitimately leave such a
call unevaluated (CONVENTIONS §3). Sign constraints (``PosInteger``, ``NonNegInteger``) are value constraints the
function enforces itself with ``ValueError``, so argument checks only test that the value is an integer. ``TypeMatchQ[value, type]`` is the strict test: ``True`` only for a value
that is known to have the type.
"""
from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

from . import _typestate
from .expr import Expr, Symbol

# ─────────────────────────── the vocabulary ───────────────────────────

ATOMS = ("Expr", "Integer", "PosInteger", "NonNegInteger", "Rational", "Real", "Number", "Numeric", "String",
         "Boolean", "Var", "Assoc", "Rule", "Option", "Fn", "Held")
CONTAINERS = ("ListOf", "MatrixOf")
EXPR_T = ("Expr",)
_PY_TYPES = {"int": "Integer", "str": "String", "float": "Real", "bool": "Boolean", "list": "ListOf",
             "complex": "Number", "Any": "Expr", "object": "Expr"}
_CONSTANT_SYMBOLS = {"True", "False", "Pi", "E", "I", "Infinity", "ComplexInfinity", "Indeterminate", "Degree",
                     "EulerGamma", "GoldenRatio", "Catalan", "Null"}


class TypeSyntaxError(ValueError):
    pass


def parse_annotation(a) -> tuple:
    """Internal type tuple for a Python annotation (a string under ``from __future__ import annotations``)."""
    if a is inspect.Parameter.empty or a is None:
        return EXPR_T
    if not isinstance(a, str):
        name = getattr(a, "__name__", None)
        if name in _PY_TYPES:
            t = _PY_TYPES[name]
            return ("ListOf", EXPR_T) if t == "ListOf" else (t,)
        return EXPR_T
    try:
        node = ast.parse(a, mode="eval").body
    except SyntaxError:
        return EXPR_T
    return _from_ast(node)


def _from_ast(node) -> tuple:
    if isinstance(node, ast.Name):
        if node.id in ATOMS:
            return (node.id,)
        if node.id in _PY_TYPES:
            t = _PY_TYPES[node.id]
            return ("ListOf", EXPR_T) if t == "ListOf" else (t,)
        return EXPR_T                     # outside the vocabulary: unconstrained
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return parse_annotation(node.value)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in CONTAINERS:
        inner = node.slice
        if isinstance(inner, ast.Index):  # Python 3.8 compatibility (not reached on 3.9+)
            inner = inner.value          # pragma: no cover
        return (node.value.id, _from_ast(inner))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union(_from_ast(node.left), _from_ast(node.right))
    return EXPR_T


def _union(*ts) -> tuple:
    parts: List[tuple] = []
    for t in ts:
        for u in (t[1] if t[0] == "Or" else (t,)):
            if u == EXPR_T:
                return EXPR_T
            if u not in parts:
                parts.append(u)
    return parts[0] if len(parts) == 1 else ("Or", tuple(parts))


def from_expr(t) -> tuple:
    """Internal type tuple for a MikoshiLang type expression (``Integer``, ``ListOf[Real]`` or ``{Real}``,
    ``A | B``)."""
    if isinstance(t, Symbol):
        if t.name in ATOMS:
            return (t.name,)
        if t.name == "List":
            return ("ListOf", EXPR_T)
        raise TypeSyntaxError(f"TypeSignature: unknown type {t.name}; types are {', '.join(ATOMS)}, "
                              "ListOf[t], MatrixOf[t] and t1 | t2")
    if isinstance(t, Expr):
        if t.head == "List" and len(t.args) == 1:      # List[Integer] evaluates to {Integer}: a list type
            return ("ListOf", from_expr(t.args[0]))
        if t.head in CONTAINERS:
            if len(t.args) != 1:
                raise TypeSyntaxError(f"TypeSignature: {t.head} takes one element type")
            return (t.head, from_expr(t.args[0]))
        if t.head == "Alternatives" and t.args:
            return _union(*(from_expr(a) for a in t.args))
    raise TypeSyntaxError(f"TypeSignature: {t!r} is not a type")


def to_expr(t: tuple):
    """MikoshiLang expression for an internal type tuple."""
    if t[0] in CONTAINERS:
        return Expr(t[0], to_expr(t[1]))
    if t[0] == "Or":
        return Expr("Alternatives", *(to_expr(u) for u in t[1]))
    return Symbol(t[0])


def type_text(t: tuple) -> str:
    if t[0] in CONTAINERS:
        return f"{t[0]}[{type_text(t[1])}]"
    if t[0] == "Or":
        return " | ".join(type_text(u) for u in t[1])
    return t[0]


# ─────────────────────────── matching ───────────────────────────

def _is_list(x) -> bool:
    return isinstance(x, Expr) and x.head == "List"


def _sym(x):
    from .library.core import S
    try:
        v = S(x)
    except Exception:
        return None
    return v if isinstance(v, sp.Basic) else None


_NUMERIC_KINDS = ("Integer", "PosInteger", "NonNegInteger", "Rational", "Real", "Number", "Numeric")


def _numval(x):
    """SymPy value of a numeric value or numeric expression (1/3, 1 + 2 I, Sqrt[2] + Pi), else None."""
    if isinstance(x, bool) or isinstance(x, str):
        return None
    if isinstance(x, (int, float, complex)):
        return sp.sympify(x)
    if isinstance(x, sp.Basic):
        return x if x.is_number else None
    if isinstance(x, Symbol) and x.name not in _CONSTANT_SYMBOLS:
        return None
    if isinstance(x, Expr) and x.head in ("List", "Association", "Rule", "RuleDelayed", "String"):
        return None
    v = _sym(x)
    return v if v is not None and v.is_number else None


def _concrete(x) -> bool:
    """A value whose kind is settled: no further evaluation turns it into something else."""
    if isinstance(x, (bool, int, float, str, complex)) or isinstance(x, sp.Number):
        return True
    if isinstance(x, Symbol):
        return x.name in ("True", "False")
    if isinstance(x, Expr):
        return x.head in ("List", "Association", "Rule", "RuleDelayed", "Rational", "Complex", "String")
    return False


def _is_bool(x) -> bool:
    return isinstance(x, bool) or (isinstance(x, Symbol) and x.name in ("True", "False"))


def match(x, t: tuple) -> Optional[bool]:
    """True: x has type t. False: x is a concrete value of another kind. None: not known (symbolic)."""
    k = t[0]
    if k in ("Expr", "Held"):
        return True
    if k == "Or":
        seen_none = False
        for u in t[1]:
            r = match(x, u)
            if r:
                return True
            seen_none |= r is None
        return None if seen_none else False
    if k in _NUMERIC_KINDS:
        if _is_bool(x) or isinstance(x, str):
            return False
        v = _numval(x)
        if v is None:
            return _unknown_unless_concrete(x)
        if v.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            return None                     # infinities: let the function decide
        if k in ("Integer", "PosInteger", "NonNegInteger"):
            if not v.is_Integer:
                return False
            n = int(v)
            return n > 0 if k == "PosInteger" else (n >= 0 if k == "NonNegInteger" else True)
        if k == "Rational":
            return bool(v.is_Rational)
        if k == "Real":
            return None if v.is_real is None else bool(v.is_real)
        if k == "Number":
            re_, im_ = v.as_real_imag()
            return bool(re_.is_Number and im_.is_Number)
        return True                         # Numeric
    if k == "String":
        if isinstance(x, str):
            return True
        return _unknown_unless_concrete(x)
    if k == "Boolean":
        if _is_bool(x):
            return True
        return _unknown_unless_concrete(x)
    if k == "Var":
        if isinstance(x, Symbol):
            return x.name not in _CONSTANT_SYMBOLS
        if isinstance(x, Expr) and not _concrete(x):
            return None                     # x[1], Subscript[x, 1] can be variables
        return False
    if k == "Assoc":
        if isinstance(x, Expr) and x.head == "Association":
            return True
        return _unknown_unless_concrete(x)
    if k == "Rule":
        if isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed"):
            return True
        return _unknown_unless_concrete(x)
    if k == "Option":
        if isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed"):
            return True
        if _is_list(x):
            return _all(match(a, ("Option",)) for a in x.args)
        return _unknown_unless_concrete(x)
    if k == "Fn":
        if isinstance(x, (int, float, str, complex)) and not isinstance(x, bool) or isinstance(x, sp.Number):
            return False
        return True
    if k == "ListOf":
        if _is_list(x):
            return _all(match(a, t[1]) for a in _elements(x))
        return _unknown_unless_concrete(x)
    if k == "MatrixOf":
        if _is_list(x):
            rows = x.args
            if not rows or not all(_is_list(r) for r in rows):
                return False if rows and any(_concrete(r) for r in rows if not _is_list(r)) else (None if rows else False)
            if len({len(r.args) for r in rows}) != 1:
                return False
            return _all(match(a, t[1]) for r in rows for a in r.args)
        return _unknown_unless_concrete(x)
    return True


def _elements(x):
    try:
        from .packed import PackedArray
    except ImportError:                     # packed arrays arrive with interop phase 2
        PackedArray = None
    if PackedArray is not None and isinstance(x, PackedArray):
        return x.args
    return x.args


def _all(results) -> Optional[bool]:
    out: Optional[bool] = True
    for r in results:
        if r is False:
            return False
        if r is None:
            out = None
    return out


def _unknown_unless_concrete(x) -> Optional[bool]:
    return False if _concrete(x) or (isinstance(x, Symbol) and x.name in _CONSTANT_SYMBOLS) else None


# ─────────────────────────── signatures ───────────────────────────

class Signature:
    """Parameter types and result type of one implementation (or one user declaration)."""
    __slots__ = ("params", "rest", "result", "names")

    def __init__(self, params, rest, result, names=None):
        self.params: List[tuple] = list(params)     # positional parameter types
        self.rest: Optional[tuple] = rest           # type of each extra argument (*args), or None
        self.result: tuple = result
        self.names = names or [None] * len(self.params)

    def param_type(self, i: int) -> Optional[tuple]:
        if i < len(self.params):
            return self.params[i]
        return self.rest

    def accepts_count(self, n: int, required: Optional[int] = None) -> bool:
        lo = len(self.params) if required is None else required
        return n >= lo and (self.rest is not None or n <= len(self.params))

    def to_expr(self):
        ps = [to_expr(t) for t in self.params]
        if self.rest is not None:
            ps.append(Expr("RepeatedNull", to_expr(self.rest)))
        return Expr("Rule", Expr("List", *ps), to_expr(self.result))

    def __repr__(self):
        ps = [type_text(t) for t in self.params] + ([type_text(self.rest) + "..."] if self.rest else [])
        return f"({', '.join(ps)}) -> {type_text(self.result)}"


_FN_CACHE: Dict[Any, Signature] = {}


def function_signature(f, held: bool = False) -> Signature:
    """Signature of a Python implementation from its annotations (cached per function object)."""
    key = (f, held)
    sig = _FN_CACHE.get(key)
    if sig is not None:
        return sig
    try:
        s = inspect.signature(f)
    except (TypeError, ValueError):
        sig = Signature([], EXPR_T, EXPR_T)
    else:
        params, rest, names = [], None, []
        for p in s.parameters.values():
            t = ("Held",) if held else parse_annotation(p.annotation)
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                params.append(t)
                names.append(p.name)
            elif p.kind == p.VAR_POSITIONAL:
                rest = t
        sig = Signature(params, rest, parse_annotation(s.return_annotation), names)
    _FN_CACHE[key] = sig
    return sig


def entry_signature(entry) -> List[Signature]:
    """Signatures of a registry entry, one per implementation."""
    return [function_signature(f, entry.hold) for f, _lo, _hi in entry.impls]


# ─────────────────────────── user declarations ───────────────────────────

TYPES_KEY = "\x00types"


def declare(name: str, sig_expr) -> Signature:
    """Store ``TypeSignature[name, {T...} -> R]`` in the current session."""
    from .kernel import state
    if not (isinstance(sig_expr, Expr) and sig_expr.head in ("Rule", "RuleDelayed") and len(sig_expr.args) == 2
            and _is_list(sig_expr.args[0])):
        raise TypeSyntaxError(f"TypeSignature: the signature must be {{t1, t2, ...}} -> result, not {sig_expr!r}")
    params, rest = [], None
    items = list(sig_expr.args[0].args)
    for i, a in enumerate(items):
        if isinstance(a, Expr) and a.head in ("RepeatedNull", "Repeated") and len(a.args) == 1:
            if i != len(items) - 1:
                raise TypeSyntaxError("TypeSignature: RepeatedNull[t] must be the last parameter type")
            rest = from_expr(a.args[0])
        else:
            params.append(from_expr(a))
    sig = Signature(params, rest, from_expr(sig_expr.args[1]))
    st = state()
    decls = st.ext.setdefault(TYPES_KEY, {}).setdefault(name, [])
    decls[:] = [d for d in decls if repr(d) != repr(sig)] + [sig]
    return sig


def declared(name: str) -> Optional[List[Signature]]:
    from .kernel import state
    d = state().ext.get(TYPES_KEY)
    return d.get(name) if d else None


def signatures_of(name: str) -> List[Signature]:
    """User declarations of ``name`` in the current session, else the library signatures."""
    d = declared(name)
    if d:
        return list(d)
    from .library.core import REGISTRY
    e = REGISTRY.get(name)
    return entry_signature(e) if e is not None else []


# ─────────────────────────── checking ───────────────────────────

_mode: ContextVar[Optional[list]] = ContextVar("mikoshilang_type_mode", default=None)
# value: None (off) or [mode, report_list] with mode "raise" or "report"


@contextmanager
def checking(mode="raise", report: Optional[list] = None):
    """Run the body with type checking on (``mode`` "raise" or "report"); yields the report list."""
    if mode not in ("raise", "report"):
        raise ValueError(f"check_types must be True, False or \"report\", not {mode!r}")
    report = [] if report is None else report
    token = _mode.set([mode, report])
    _typestate.ACTIVE += 1
    try:
        yield report
    finally:
        _typestate.ACTIVE -= 1
        _mode.reset(token)


@contextmanager
def unchecked():
    """Turn checking off for the body (used while the checker evaluates type tests)."""
    token = _mode.set(None)
    try:
        yield
    finally:
        _mode.reset(token)


def _mismatch(head: str, where, value, t: tuple):
    m = _mode.get()
    if m is None:
        return
    mode, report = m
    what = "result" if where == "result" else f"argument {where}"
    if mode == "raise":
        raise TypeError(f"{head}: {what} {_short(value)} is not of type {type_text(t)} (type check)")
    report.append({"function": head, "position": where, "value": value, "expected": type_text(t)})


def _short(v) -> str:
    s = repr(v) if isinstance(v, str) else str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def check_args(head: str, sig: Signature, args) -> None:
    """Report arguments that are concrete values of the wrong kind."""
    if _mode.get() is None:
        return
    for i, a in enumerate(args):
        t = sig.param_type(i)
        if t is None or t == EXPR_T or t[0] == "Held":
            continue
        if match(a, _kind(t)) is False:
            _mismatch(head, i + 1, a, t)


_SIGN_FREE = {"PosInteger": ("Integer",), "NonNegInteger": ("Integer",)}


def _kind(t: tuple) -> tuple:
    """The type with sign constraints dropped: a negative integer for PosInteger is a bad *value* (the function
    raises ValueError, CONVENTIONS §2), not a wrong kind, so argument checks leave it to the function."""
    if t[0] in _SIGN_FREE:
        return _SIGN_FREE[t[0]]
    if t[0] in CONTAINERS:
        return (t[0], _kind(t[1]))
    if t[0] == "Or":
        return _union(*(_kind(u) for u in t[1]))
    return t


def check_result(head: str, sig: Signature, value) -> None:
    if _mode.get() is None or sig.result == EXPR_T:
        return
    if match(value, sig.result) is False:
        _mismatch(head, "result", value, sig.result)


def check_library_call(head: str, f, args, held: bool):
    """Hook for :func:`mikoshilang.library.core.dispatch`: returns the signature to check the result with."""
    if _mode.get() is None:
        return None
    sig = function_signature(f, held)
    check_args(head, sig, args)
    return sig


def select_declared(head: str, args) -> Optional[List[Signature]]:
    """Declared signatures for a user call; reports a mismatch when no declaration fits the arguments."""
    decls = declared(head)
    if not decls or _mode.get() is None:
        return None
    fitting = [d for d in decls if d.accepts_count(len(args))]
    if not fitting:
        _mismatch_count(head, len(args), decls)
        return None
    if len(fitting) == 1:
        check_args(head, fitting[0], args)
        return fitting
    ok = [d for d in fitting if all(match(a, d.param_type(i)) is not False for i, a in enumerate(args))]
    if not ok:
        check_args(head, fitting[0], args)     # report against the first declaration
        return None
    return ok


def _mismatch_count(head, n, decls):
    m = _mode.get()
    if m is None:
        return
    mode, report = m
    text = " or ".join(repr(d) for d in decls)
    if mode == "raise":
        raise TypeError(f"{head}: called with {n} argument(s); declared {text} (type check)")
    report.append({"function": head, "position": "count", "value": n, "expected": text})


def check_declared_result(head: str, sigs: List[Signature], value) -> None:
    if _mode.get() is None or not sigs:
        return
    results = [s.result for s in sigs]
    if any(r == EXPR_T for r in results):
        return
    if all(match(value, r) is False for r in results):
        _mismatch(head, "result", value, results[0] if len(results) == 1 else _union(*results))
