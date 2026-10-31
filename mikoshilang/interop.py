"""Python interop: one conversion table, a Pythonic call proxy, typed Python functions, gated PythonFunction.

User guide: docs/python_interop.md. Entry points on :class:`mikoshilang.Session`: ``eval``, ``ns``, ``function``
and the ``allow_python`` capability.

Conversion table (Python -> MikoshiLang -> Python round-trips exactly for every row marked <->):

    bool, numpy.bool_       <->  True / False                         (Boolean)
    int, numpy.integer      <->  Integer (int, arbitrary size)        (Integer)
    float, numpy.floating   <->  machine Real (float)                 (Real)
    float('inf'), -inf      <->  Infinity / -Infinity                 (Real)
    float('nan')             ->  ValueError (MikoshiLang has no NaN; never silently Indeterminate)
    complex                 <->  machine complex a + b I              (Number)
    fractions.Fraction      <->  Rational                             (Rational)
    sympy.Float (bignum)    <->  bignum Real, precision kept          (Real)
    str                     <->  String                               (String)
    None                    <->  Null
    list                    <->  List                                 (ListOf[T]);  tuple -> List (one way)
    dict                    <->  Association                          (Assoc)
    numpy.ndarray            ->  nested List (bool/int/float/complex/str dtypes; object dtype element-wise)
                                 List -> ndarray with arrays="numpy" (rectangular, machine numbers)
    pandas.Series            ->  List (default RangeIndex) or Association (index -> value)
    pandas.DataFrame         ->  List of Associations (rows)
    sympy.Basic             <->  expression (back with symbolic="sympy")
    sympy.MatrixBase         ->  nested List
    Expr / Symbol           <->  itself (arguments converted recursively)
    callable                 ->  TypeError (register it with @session.function, or call it via PythonFunction)
    anything else            ->  TypeError (no inert foreign atoms)

Exact algebraic numbers (Sqrt[2], 1 + 2 I) have no exact Python type and stay symbolic.
"""
from __future__ import annotations

import collections.abc
import fractions
import importlib
import inspect
import math
import re
import sys
import typing
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional

import sympy as sp

from .expr import Expr, Symbol
from .packed import PackedArray, from_array

PYFUNCS_KEY = "\x00python_functions"  # SessionState.ext kind (session-wide) holding registered Python functions

_NAME = re.compile(r"^[A-Za-z$][A-Za-z0-9$]*$")
_ARGERRORS = (TypeError, ValueError, NotImplementedError, ArithmeticError)


def check_modes(out: str = "python", symbolic: str = "expr", arrays: str = "list") -> None:
    if out not in ("python", "expr"):
        raise ValueError(f"Session.eval: out must be 'python' or 'expr', not {out!r}")
    if symbolic not in ("expr", "sympy", "error"):
        raise ValueError(f"Session.eval: symbolic must be 'expr', 'sympy' or 'error', not {symbolic!r}")
    if arrays not in ("list", "numpy"):
        raise ValueError(f"Session.eval: arrays must be 'list' or 'numpy', not {arrays!r}")


# ─────────────────────────── Python -> MikoshiLang ───────────────────────────

def from_python(obj: Any) -> Any:
    """Convert a Python value to a MikoshiLang expression (see the module table); unknown types raise TypeError."""
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj if type(obj) is str else str(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return _from_float(float(obj))                  # float subclasses (numpy.float64) become plain float
    if isinstance(obj, complex):
        if math.isnan(obj.real) or math.isnan(obj.imag):
            raise ValueError("from_python: NaN has no MikoshiLang value")
        if math.isinf(obj.real) or math.isinf(obj.imag):
            raise ValueError(f"from_python: complex infinity {obj!r} has no exact MikoshiLang counterpart")
        return _from_sympy(sp.Float(obj.real) + sp.Float(obj.imag) * sp.I) if obj.imag else _from_float(obj.real)
    if isinstance(obj, fractions.Fraction):
        return obj.numerator if obj.denominator == 1 else _from_sympy(sp.Rational(obj.numerator, obj.denominator))
    if isinstance(obj, Symbol):
        return obj
    if obj.__class__ is PackedArray:
        return obj
    if isinstance(obj, Expr):
        head = obj.head if isinstance(obj.head, str) else from_python(obj.head)
        args = [from_python(a) for a in obj.args]
        if head is obj.head and all(n is o for n, o in zip(args, obj.args)):
            return obj                                  # already MikoshiLang data: no copy
        return Expr(head, *args)
    if isinstance(obj, (list, tuple)):
        return Expr("List", *(from_python(a) for a in obj))
    if isinstance(obj, dict):
        from .kernel_lists import make_assoc
        return make_assoc([(from_python(k), from_python(v)) for k, v in obj.items()])
    np = _numpy()
    if np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return _from_float(float(obj))
        if isinstance(obj, np.complexfloating):
            return from_python(complex(obj))
        if isinstance(obj, np.ndarray):
            packed = from_array(obj)
            if packed is not None:
                return packed                           # machine numbers stay one NumPy array (packed List)
            if obj.dtype.kind in "biufcU":
                return from_python(obj.tolist())
            if obj.dtype.kind == "O":
                return from_python(obj.tolist())
            raise TypeError(f"from_python: numpy dtype {obj.dtype} is not supported")
    pd = _pandas()
    if pd is not None:
        if isinstance(obj, pd.Series):
            if isinstance(obj.index, pd.RangeIndex) and obj.index.start == 0 and obj.index.step == 1:
                return from_python(obj.to_numpy())
            return from_python(dict(zip(obj.index.tolist(), obj.to_numpy().tolist())))
        if isinstance(obj, pd.DataFrame):
            return from_python(obj.to_dict(orient="records"))
    if isinstance(obj, sp.MatrixBase):
        return from_python(obj.tolist())
    if isinstance(obj, sp.Basic):
        if isinstance(obj, sp.Float):
            if obj._prec <= 53:
                return _from_float(float(obj))          # a machine-precision sympy.Float is a machine Real
            return obj                                  # bignum Real, precision kept
        return _from_sympy(obj)
    if callable(obj):
        raise TypeError("from_python: Python callables must be registered with @session.function or called "
                        "through PythonFunction")
    raise TypeError(f"from_python: cannot convert a {type(obj).__name__} to a MikoshiLang value")


def _from_float(x: float):
    if math.isnan(x):
        raise ValueError("from_python: NaN has no MikoshiLang value")
    if math.isinf(x):
        return Symbol("Infinity") if x > 0 else Expr("Times", -1, Symbol("Infinity"))
    return x


def _from_sympy(e):
    from .math_ops import from_sympy
    return from_sympy(e)


# ─────────────────────────── MikoshiLang -> Python ───────────────────────────

def to_python(e: Any, symbolic: str = "expr", arrays: str = "list") -> Any:
    """Convert a MikoshiLang value to Python.

    symbolic: what to do with non-data expressions: "expr" (keep the Expr), "sympy" (convert) or "error" (raise
              TypeError).
    arrays:   "list" (nested lists) or "numpy" (a rectangular List of machine numbers becomes an ndarray).

    Exact rationals become fractions.Fraction, machine numbers float/complex, bignum reals sympy.Float.
    """
    if e is None or isinstance(e, (bool, str, int, float, complex)):
        return e
    if isinstance(e, sp.Float):
        return float(e) if e._prec <= 53 else e
    if isinstance(e, Symbol):
        if e.name == "Null":
            return None
        if e.name == "Infinity":
            return math.inf
        return _symbolic(e, symbolic)
    if e.__class__ is PackedArray:
        return e.array.copy() if arrays == "numpy" else e.array.tolist()
    if isinstance(e, Expr):
        if e.head == "PythonObject":
            found = _handle_lookup(e)
            if found is not None:
                return found[0]
        if e.head == "List":
            if arrays == "numpy":
                arr = _as_array(e)
                if arr is not None:
                    return arr
            return [to_python(a, symbolic, arrays) for a in e.args]
        if e.head == "Association":
            out = {}
            for r in e.args:
                k = to_python(r.args[0], symbolic, arrays)
                try:
                    hash(k)
                except TypeError:
                    raise TypeError(f"to_python: association key {r.args[0]!r} has no hashable Python form") from None
                out[k] = to_python(r.args[1], symbolic, arrays)
            return out
        num = _number(e)
        if num is not None:
            return num
    elif isinstance(e, sp.Basic):
        if symbolic == "sympy":
            return e
        return to_python(_from_sympy(e), symbolic, arrays)
    return _symbolic(e, symbolic)


def _number(e):
    """Exact rationals, machine complex numbers and -Infinity as Python values; None for anything else."""
    if e == Expr("Times", -1, Symbol("Infinity")):
        return -math.inf
    if not (isinstance(e, Expr) and e.head in ("Times", "Plus", "Power", "Rational", "Complex")):
        return None
    try:
        from .math_ops import to_sympy
        v = to_sympy(e)
    except Exception:          # not convertible to SymPy: not a number, stays symbolic
        return None
    if not isinstance(v, sp.Basic) or not v.is_number:
        return None
    if isinstance(v, sp.Rational):
        return fractions.Fraction(int(v.p), int(v.q))
    re_, im_ = v.as_real_imag()
    if isinstance(re_, (sp.Float, sp.Integer)) and isinstance(im_, (sp.Float, sp.Integer)) and v.has(sp.Float):
        if all(getattr(f, "_prec", 53) <= 53 for f in v.atoms(sp.Float)):
            return complex(float(re_), float(im_))
    return None                                         # exact algebraic numbers (Sqrt[2], 1 + 2 I) stay symbolic


def _symbolic(e, mode):
    if mode == "expr":
        return e
    if mode == "sympy":
        from .math_ops import to_sympy
        return to_sympy(e)
    raise TypeError(f"to_python: {e} is symbolic and has no Python data form")


def _as_array(e):
    np = _numpy()
    if np is None:
        return None

    def scalars(x):
        if isinstance(x, Expr) and x.head == "List":
            return [scalars(a) for a in x.args]
        v = x if isinstance(x, (int, float, bool)) else to_python(x, symbolic="error")
        if isinstance(v, (bool, int, float, complex)):
            return v
        raise TypeError
    try:
        arr = np.array(scalars(e))
    except (TypeError, ValueError):
        return None
    return arr if arr.dtype.kind in "biufc" else None


def _numpy():
    try:
        import numpy
        return numpy
    except ImportError:  # pragma: no cover
        return None


def _pandas():
    return sys.modules.get("pandas")                  # only if the caller already uses pandas: never import it


def _is_symbolic(v) -> bool:
    """True when a to_python value still holds an expression (a symbolic leaf)."""
    if isinstance(v, (Expr, Symbol, sp.Basic)) and not isinstance(v, sp.Float):
        return True
    if isinstance(v, list):
        return any(_is_symbolic(x) for x in v)
    if isinstance(v, dict):
        return any(_is_symbolic(x) for x in v.values())
    return False


def _numeric_value(value):
    """A SymPy number for an exact or inexact numeric expression (Sqrt[2], Pi), else None."""
    try:
        from .math_ops import to_sympy
        v = to_sympy(value)
    except Exception:          # not convertible: not numeric
        return None
    return v if isinstance(v, sp.Basic) and v.is_number and not v.has(sp.nan, sp.zoo) else None


# ─────────────────────────── types from Python hints ───────────────────────────

VOCABULARY = ("Expr", "Integer", "PosInteger", "NonNegInteger", "Rational", "Real", "Number", "Numeric",
              "String", "Boolean", "Var", "Assoc", "Rule", "Fn", "Held")

_SEQUENCES = (list, tuple, typing.List, typing.Tuple, typing.Sequence, collections.abc.Sequence)


def mtype(hint) -> str:
    """The CONVENTIONS.md type-vocabulary name for a Python type hint (or a vocabulary string)."""
    if hint is inspect.Parameter.empty or hint is Any or hint is None:
        return "Expr"
    if isinstance(hint, str):
        return hint
    np = _numpy()
    simple = {bool: "Boolean", int: "Integer", float: "Real", complex: "Number", str: "String",
              fractions.Fraction: "Rational", dict: "Assoc", Expr: "Expr", Symbol: "Var"}
    if hint in simple:
        return simple[hint]
    if np is not None and hint is np.ndarray:
        return "ListOf[Number]"
    if isinstance(hint, type) and issubclass(hint, sp.Basic):
        return "Expr"
    origin = typing.get_origin(hint)
    targs = typing.get_args(hint)
    if origin in _SEQUENCES or hint in (list, tuple):
        return f"ListOf[{mtype(targs[0]) if targs else 'Expr'}]"
    if origin is dict:
        return "Assoc"
    if origin is typing.Union:
        return " | ".join(mtype(t) for t in targs if t is not type(None))
    return "Expr"


def _coerce(name, pos, value, hint):
    """Convert an evaluated MikoshiLang argument for a hinted Python parameter (CONVENTIONS.md §2 order).

    Wrong kind -> TypeError; symbolic input -> Unevaluated; right kind but bad value -> ValueError.
    """
    from .library.core import Unevaluated
    if hint is inspect.Parameter.empty or hint is Any or hint is Expr:
        return value
    if isinstance(hint, type) and issubclass(hint, sp.Basic):
        from .math_ops import to_sympy
        return to_sympy(value)
    np = _numpy()
    py = to_python(value, symbolic="expr")
    want = mtype(hint)

    def bad():
        if _is_symbolic(py) and _numeric_value(value) is None and not isinstance(py, (list, dict)):
            raise Unevaluated()                      # symbolic input (n, n + 1): stay unevaluated
        raise TypeError(f"{name}: argument {pos} must be {want}, not {_show(value)}")

    if isinstance(hint, str):
        return _coerce_vocab(name, pos, value, py, hint, bad)
    if hint is bool:
        return py if isinstance(py, bool) else bad()
    if hint is int:
        return py if isinstance(py, int) and not isinstance(py, bool) else bad()
    if hint is float:
        if isinstance(py, (int, float, fractions.Fraction, sp.Float)) and not isinstance(py, bool):
            return float(py)
        num = _numeric_value(value)
        if num is not None and num.is_real:
            return float(num)                        # Pi, Sqrt[2]: numeric, converted to a machine real
        return bad()
    if hint is complex:
        if isinstance(py, (int, float, complex, fractions.Fraction, sp.Float)) and not isinstance(py, bool):
            return complex(py)
        num = _numeric_value(value)
        if num is not None:
            return complex(num)
        return bad()
    if hint is fractions.Fraction:
        return fractions.Fraction(py) if isinstance(py, (int, fractions.Fraction)) and not isinstance(py, bool) \
            else bad()
    if hint is str:
        return py if isinstance(py, str) else bad()
    if hint is Symbol:
        return value if isinstance(value, Symbol) else bad()
    if np is not None and hint is np.ndarray:
        arr = to_python(value, arrays="numpy")
        return arr if isinstance(arr, np.ndarray) else bad()
    origin = typing.get_origin(hint) or hint
    if origin in _SEQUENCES:
        if not isinstance(py, list):
            return bad()
        targs = [t for t in typing.get_args(hint) if t is not Ellipsis]
        items = value.args
        out = [_coerce(name, f"{pos}[{i + 1}]", v, targs[0]) for i, v in enumerate(items)] if targs else py
        return tuple(out) if origin in (tuple, typing.Tuple) else out
    if origin is dict:
        return py if isinstance(py, dict) else bad()
    if origin is typing.Union:
        options = [t for t in typing.get_args(hint) if t is not type(None)]
        if py is None and len(options) < len(typing.get_args(hint)):
            return None
        for t in options:
            try:
                return _coerce(name, pos, value, t)
            except (TypeError, ValueError, Unevaluated):
                continue
        return bad()
    return py


def _coerce_vocab(name, pos, value, py, hint, bad):
    """Vocabulary strings ("PosInteger", "Real", ...) used as hints: check the kind, pass the Python value."""
    if hint in ("Integer", "PosInteger", "NonNegInteger"):
        if not (isinstance(py, int) and not isinstance(py, bool)):
            return bad()
        if hint == "PosInteger" and py <= 0:
            raise ValueError(f"{name}: argument {pos} must be a positive integer, not {py}")
        if hint == "NonNegInteger" and py < 0:
            raise ValueError(f"{name}: argument {pos} must be a non-negative integer, not {py}")
        return py
    if hint == "Rational":
        return py if isinstance(py, (int, fractions.Fraction)) and not isinstance(py, bool) else bad()
    if hint in ("Real", "Number", "Numeric"):
        if isinstance(py, (int, float, fractions.Fraction, sp.Float)) and not isinstance(py, bool):
            return py
        if hint != "Real" and isinstance(py, complex):
            return py
        num = _numeric_value(value)
        if num is not None and (hint != "Real" or num.is_real):
            return value                              # exact numeric expression (Pi, Sqrt[2]) kept as is
        return bad()
    if hint == "String":
        return py if isinstance(py, str) else bad()
    if hint == "Boolean":
        return py if isinstance(py, bool) else bad()
    if hint == "Assoc":
        return py if isinstance(py, dict) else bad()
    if hint == "Var":
        return value if isinstance(value, Symbol) else bad()
    if hint.startswith("ListOf[") or hint.startswith("MatrixOf["):
        return py if isinstance(py, list) else bad()
    return value                                      # Expr, Fn, Rule, Held, unions: passed through


def _show(v):
    try:
        from .formatting import input_form
        return input_form(v)
    except Exception:          # formatting must never mask the real error
        return repr(v)


# ─────────────────────────── registration and calls ───────────────────────────

class PythonFunctionInfo:
    """A Python function registered as a session head (``fn.mikoshi`` after ``@session.function``)."""

    __slots__ = ("name", "fn", "params", "returns", "signature", "listable", "notes", "keywords", "var_keyword")

    def __init__(self, name, fn, params, returns, signature, listable, notes, keywords, var_keyword):
        self.name, self.fn, self.params, self.returns = name, fn, params, returns
        self.signature, self.listable, self.notes = signature, listable, notes
        self.keywords, self.var_keyword = keywords, var_keyword

    def __repr__(self):
        return f"<PythonFunctionInfo {self.name} {self.signature}>"


_CHECKS = ("exact", "self-check", "numeric cross-check", "reference", "delegated")


def register(state, fn: Callable, name: Optional[str] = None, listable: bool = False, notes: str = "",
             check: str = "delegated", override: bool = False) -> PythonFunctionInfo:
    """Register a typed Python function as a session-local MikoshiLang head."""
    if not callable(fn):
        raise TypeError(f"function: {fn!r} is not callable")
    name = name or _wl_name(getattr(fn, "__name__", ""))
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"function: {name!r} is not a valid MikoshiLang symbol name; pass name=...")
    methods = [m.strip() for m in str(check).split(",")]
    if not all(m in _CHECKS for m in methods):
        raise ValueError(f"function: check must name one or more of {', '.join(_CHECKS)}, not {check!r}")
    from . import library
    from . import kernel as K
    if name in getattr(K, "_BUILTINS", {}):
        raise ValueError(f"function: {name} is a kernel built-in and cannot be redefined from Python")
    if not override and name in library.REGISTRY:
        raise ValueError(f"function: {name} is a library head; pass override=True to shadow it")
    sig = inspect.signature(fn)
    hints = _hints(fn, sig, name)
    params, keywords, var_keyword = [], {}, False
    for p in sig.parameters.values():
        hint = hints.get(p.name, p.annotation)
        if p.kind == p.KEYWORD_ONLY:
            keywords[p.name] = hint
        elif p.kind == p.VAR_KEYWORD:
            var_keyword = True
        else:
            params.append((p.name, hint, p.kind, p.default))
    returns = hints.get("return", sig.return_annotation)
    sig_args = [mtype(h) + ("..." if k == inspect.Parameter.VAR_POSITIONAL else "") for _, h, k, _ in params]
    if keywords or var_keyword:
        sig_args.append("Option...")
    signature = "{" + ", ".join(sig_args) + "} -> " + mtype(returns)
    qual = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', name)}"
    notes = (notes.rstrip() + " " if notes else "") + f"Check: {check} (user Python function {qual})."
    info = PythonFunctionInfo(name, fn, params, returns, signature, listable, notes, keywords, var_keyword)
    state.ext.setdefault(PYFUNCS_KEY, {})[name] = info
    return info


_VOCAB_HINT = re.compile(r"^(%s|(ListOf|MatrixOf)\[.+\])(\s*\|\s*(%s|(ListOf|MatrixOf)\[.+\]))*$"
                         % ("|".join(VOCABULARY), "|".join(VOCABULARY)))


class _Annotated:
    """Carrier for one string annotation, resolved with typing.get_type_hints in the function's globals."""


def _hints(fn, sig, name):
    """Resolved type hints; CONVENTIONS vocabulary strings ("PosInteger", "ListOf[Real]") are kept as strings."""
    globalns = getattr(inspect.unwrap(fn), "__globals__", {})
    out = {}
    items = [(p.name, p.annotation) for p in sig.parameters.values()] + [("return", sig.return_annotation)]
    for key, ann in items:
        if ann is inspect.Parameter.empty or ann is inspect.Signature.empty:
            continue
        if isinstance(ann, str):
            text = ann.strip()
            if _VOCAB_HINT.match(text):
                out[key] = text
                continue
            holder = _Annotated()
            holder.__annotations__ = {key: text}
            try:
                ann = typing.get_type_hints(holder, globalns=dict(globalns))[key]
            except Exception as exc:   # e.g. "int | None" under Python 3.9, or an undefined name
                raise TypeError(f"function: cannot read the type hint {text!r} of {name}: {exc}") from exc
        out[key] = ann
    return out


_GLOBAL_PREVIOUS: Dict[str, Any] = {}


def library_function(fn: Optional[Callable] = None, *, name: Optional[str] = None, area: str = "user library",
                     listable: bool = False, notes: str = "", check: str = "delegated", override: bool = False):
    """Decorator for library authors: register a typed Python function as a head in every session.

    Same typing, error and listable rules as ``@session.function``; the head is recorded in
    ``library.STATUS`` with status ``added`` and its ``Check:`` note, like the built-in library. Kernel built-ins
    are never redefined; an existing library head is replaced only with ``override=True`` (and restored by
    ``unregister``). Prefer ``@session.function`` in applications: a global head is visible to every session.
    """
    def deco(f):
        from .library import core
        holder = _Holder()
        info = register(holder, f, name=name, listable=listable, notes=notes, check=check, override=override)

        def impl(*args):
            return call_registered(info, list(args))
        impl.__name__ = info.name
        if info.name in core.REGISTRY or info.name in core.STATUS:
            if info.name not in _GLOBAL_PREVIOUS:
                _GLOBAL_PREVIOUS[info.name] = (core.REGISTRY.pop(info.name, None), core.STATUS.pop(info.name, None))
            else:
                core.REGISTRY.pop(info.name, None)
        core.fn(info.name, area=area, status="added", notes=info.notes)(impl)
        f.mikoshi = info
        return f
    return deco(fn) if fn is not None else deco


def unregister(name: str) -> None:
    """Remove a head registered with ``library_function`` (restoring a library head it overrode)."""
    from .library import core
    entry = core.REGISTRY.get(name)
    if entry is None or not any(getattr(f, "__name__", "") == name and f.__module__ == __name__
                                for f, _lo, _hi in entry.impls):
        raise ValueError(f"unregister: {name} was not registered with mikoshilang.function")
    core.REGISTRY.pop(name, None)
    core.STATUS.pop(name, None)
    prev = _GLOBAL_PREVIOUS.pop(name, None)
    if prev is not None:
        if prev[0] is not None:
            core.REGISTRY[name] = prev[0]
        if prev[1] is not None:
            core.STATUS[name] = prev[1]


class _Holder:
    """Stand-in session state for library_function (register writes into .ext)."""

    def __init__(self):
        self.ext = {}


def registered(state) -> Dict[str, PythonFunctionInfo]:
    return dict(state.ext.get(PYFUNCS_KEY) or {})


def _wl_name(pyname: str) -> str:
    """snake_case -> CamelCase (my_func -> MyFunc); names already CamelCase are kept."""
    return pyname if pyname[:1].isupper() else "".join(p[:1].upper() + p[1:] for p in pyname.split("_") if p)


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def lookup(state, head):
    table = state.ext.get(PYFUNCS_KEY)
    return table.get(head) if table else None


def _split_options(name, args, keywords, var_keyword):
    """Trailing ``Name -> value`` rules become keyword arguments (only for functions that take keywords)."""
    if not (keywords or var_keyword):
        return list(args), {}
    args, kwargs = list(args), {}
    while args and isinstance(args[-1], Expr) and args[-1].head in ("Rule", "RuleDelayed") \
            and len(args[-1].args) == 2 and isinstance(args[-1].args[0], (Symbol, str)):
        lhs, rhs = args.pop().args
        key = lhs.name if isinstance(lhs, Symbol) else lhs
        # RelTol -> rel_tol; a lower-case symbol or a string ("rel_tol" -> 0.1) is used as written
        pykey = _snake(key) if isinstance(lhs, Symbol) and key[:1].isupper() else key
        if key in keywords:
            pykey = key
        if not pykey.isidentifier() or (pykey not in keywords and not var_keyword):
            raise ValueError(f"{name}: unknown option {key}")
        kwargs[pykey] = rhs
    return args, kwargs


def call_registered(info: PythonFunctionInfo, args):
    """Apply a registered Python function to evaluated MikoshiLang arguments (raises per CONVENTIONS.md §3)."""
    args, options = _split_options(info.name, args, info.keywords, info.var_keyword)
    if info.listable and any(isinstance(a, Expr) and a.head == "List" for a in args):
        n = {len(a.args) for a in args if isinstance(a, Expr) and a.head == "List"}
        if len(n) != 1:
            raise ValueError(f"{info.name}: cannot thread over lists of unequal length")
        k = n.pop()
        rules = [Expr("Rule", Symbol(key), v) for key, v in options.items()]
        return Expr("List", *(Expr(info.name, *(a.args[i] if isinstance(a, Expr) and a.head == "List" else a
                                                for a in args), *rules) for i in range(k)))
    fixed = [p for p in info.params if p[2] in (inspect.Parameter.POSITIONAL_ONLY,
                                                 inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    var = next((p for p in info.params if p[2] == inspect.Parameter.VAR_POSITIONAL), None)
    lo = sum(1 for p in fixed if p[3] is inspect.Parameter.empty)
    if len(args) < lo or (var is None and len(args) > len(fixed)):
        raise ValueError(f"{info.name}: called with {len(args)} argument(s); signature {info.signature}")
    pyargs = [_coerce(info.name, i + 1, a, fixed[i][1] if i < len(fixed) else var[1]) for i, a in enumerate(args)]
    pykw = {k: _coerce(info.name, k, v, info.keywords.get(k, inspect.Parameter.empty)) for k, v in options.items()}
    return _invoke(info.name, info.fn, pyargs, pykw)


def _invoke(name, fn, pyargs, pykw=None):
    try:
        result = fn(*pyargs, **(pykw or {}))
    except PermissionError:
        raise
    except _ARGERRORS as exc:
        text = str(exc)
        if text.startswith(f"{name}:"):
            raise
        raise _same_class(exc, f"{name}: {text or type(exc).__name__}") from exc
    except Exception as exc:                          # foreign failure: keep the class name, make it catchable
        raise ValueError(f"{name}: Python raised {type(exc).__name__}: {exc}") from exc
    try:
        return from_python(result)
    except (TypeError, ValueError) as exc:
        raise type(exc)(f"{name}: the Python result cannot be converted ({exc})") from exc


def _same_class(exc, message):
    """The exception's own class with a new message, or its MikoshiLang base class if that is not possible."""
    try:
        return type(exc)(message)
    except Exception:          # classes such as UnicodeDecodeError need extra constructor arguments
        base = next(c for c in _ARGERRORS if isinstance(exc, c))
        return base(message)


# ─────────────────────────── PythonFunction["module.attr"] (capability-gated) ───────────────────────────

_python = ContextVar("mikoshilang_allow_python", default=False)


def python_permission(value):
    """Normalise Session(allow_python=...): False, True, or a frozenset of module names."""
    if value is None or value is False:
        return False
    if value is True:
        return True
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (set, frozenset, list, tuple)):
        mods = frozenset(value)
        for m in mods:
            if not isinstance(m, str) or not m or any(not p.isidentifier() for p in m.split(".")):
                raise ValueError(f"Session: allow_python entries must be module names such as 'math', not {m!r}")
        return mods
    raise TypeError(f"Session: allow_python must be False, True or a set of module names, not {value!r}")


def _allowed(path: str) -> bool:
    perm = _python.get()
    if perm is True:
        return True
    if not perm:
        return False
    parts = path.split(".")
    return any(".".join(parts[:i]) in perm for i in range(1, len(parts) + 1))


def _check_path(path: Any, fname: str = "PythonFunction"):
    if not isinstance(path, str):
        raise TypeError(f"{fname}: expected a dotted Python name as a string, not {_show(path)}")
    parts = path.split(".")
    if not path or any(not p.isidentifier() for p in parts):
        raise ValueError(f"{fname}: {path!r} is not a dotted Python name")
    if any(p.startswith("_") for p in parts):
        raise PermissionError(f"{fname}: private and dunder names such as {path!r} cannot be used")
    return parts


def _gate(path: str, fname: str = "PythonFunction"):
    if not _allowed(path):
        raise PermissionError(f"{fname}: using {path} requires Session(allow_python=True) or an allow-list "
                              f"containing {path.split('.')[0]!r}")


def _getattr_path(obj, base: str, attrs, fname):
    for i, attr in enumerate(attrs):
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            raise ValueError(f"{fname}: {base} has no attribute {'.'.join(attrs[:i + 1])}") from None
    return obj


def resolve(path: Any):
    """Import module.attr (longest importable module prefix), subject to Session(allow_python=...)."""
    parts = _check_path(path)
    _gate(path)
    for cut in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:cut])
        try:
            obj = importlib.import_module(mod_name)
        except ImportError:
            continue
        obj = _getattr_path(obj, mod_name, parts[cut:], "PythonFunction")
        if not callable(obj):
            raise TypeError(f"PythonFunction: {path} is not callable")
        return obj
    raise ValueError(f"PythonFunction: cannot import {path}")


# ── PythonObject handles: Python values with no MikoshiLang form, kept in the session ──

PYOBJECTS_KEY = "\x00python_objects"


def _objects(state=None):
    from . import kernel as K
    st = state if state is not None else K.state()
    return st.ext.setdefault(PYOBJECTS_KEY, {})


def make_handle(obj, origin: str) -> Expr:
    """Store obj in the current session and return its handle PythonObject[id, "description"]."""
    table = _objects()
    hid = len(table) + 1
    while hid in table:
        hid += 1
    table[hid] = (obj, origin)
    if isinstance(obj, type(importlib)):
        desc = f"module {obj.__name__}"
    else:
        desc = f"{type(obj).__module__}.{type(obj).__qualname__}"
    return Expr("PythonObject", hid, desc)


def is_handle(e) -> bool:
    return (isinstance(e, Expr) and e.head == "PythonObject" and len(e.args) == 2 and isinstance(e.args[0], int)
            and not isinstance(e.args[0], bool) and isinstance(e.args[1], str))


def _handle_lookup(e):
    if not is_handle(e):
        return None
    try:
        return _objects().get(e.args[0])
    except Exception:          # no active session state
        return None


def handle_object(e, fname: str):
    """(object, origin) for a PythonObject handle of the current session; raises for anything else."""
    if not is_handle(e):
        raise TypeError(f"{fname}: expected a PythonObject handle, not {_show(e)}")
    found = _objects().get(e.args[0])
    if found is None:
        raise ValueError(f"{fname}: PythonObject {e.args[0]} does not exist in this session (cleared, or "
                         f"created in another session)")
    _gate(found[1], fname)                           # the session must still allow where it came from
    return found


def python_import(name):
    """PythonImport["module"]: a handle to an allowed module."""
    _check_path(name, "PythonImport")
    _gate(name, "PythonImport")
    try:
        mod = importlib.import_module(name)
    except ImportError as exc:
        raise ValueError(f"PythonImport: cannot import {name}: {exc}") from None
    return make_handle(mod, name)


def python_attribute(handle, path):
    """PythonAttribute[handle, "a.b"]: an attribute of a handle's object, converted (or a new handle)."""
    obj, origin = handle_object(handle, "PythonAttribute")
    parts = _check_path(path, "PythonAttribute")
    full = f"{origin}.{path}"
    if isinstance(obj, type(importlib)):
        _gate(f"{obj.__name__}.{path}", "PythonAttribute")
        full = f"{obj.__name__}.{path}"
    value = _getattr_path(obj, origin, parts, "PythonAttribute")
    return _result(value, full)


def _target(head: Expr):
    """The callable and its origin path for PythonFunction["mod.f"], PythonFunction[handle] or
    PythonFunction[handle, "attr.path"]."""
    a = head.args
    if len(a) == 1 and isinstance(a[0], str):
        return resolve(a[0]), a[0]
    if len(a) in (1, 2) and is_handle(a[0]):
        obj, origin = handle_object(a[0], "PythonFunction")
        if len(a) == 2:
            parts = _check_path(a[1])
            if isinstance(obj, type(importlib)):
                origin = f"{obj.__name__}.{a[1]}"
                _gate(origin)
            else:
                origin = f"{origin}.{a[1]}"
            obj = _getattr_path(obj, origin, parts, "PythonFunction")
        if not callable(obj):
            raise TypeError(f"PythonFunction: {origin} is not callable")
        return obj, origin
    if len(a) == 1 and isinstance(a[0], (Expr, Symbol)) and not is_handle(a[0]):
        raise TypeError(f"PythonFunction: expected a dotted name or a PythonObject handle, not {_show(a[0])}")
    raise ValueError('PythonFunction: expected PythonFunction["module.name"], PythonFunction[handle] or '
                     'PythonFunction[handle, "attribute"]')


def _arg(a):
    if is_handle(a):
        return handle_object(a, "PythonFunction")[0]
    if isinstance(a, Expr) and a.head == "List" and a.__class__ is not PackedArray:
        return [_arg(x) for x in a.args]
    return to_python(a, symbolic="expr")


def _result(value, origin):
    """A Python result: converted by the table, or kept as a PythonObject handle."""
    try:
        return from_python(value)
    except TypeError:
        return make_handle(value, origin)


def apply_python_function(head: Expr, args):
    """PythonFunction[...][args...]: symbolic arguments leave the call unevaluated."""
    from .library.core import Unevaluated
    fn, origin = _target(head)
    args, options = _split_options("PythonFunction", args, (), True)
    pyargs = [_arg(a) for a in args]
    pykw = {k: _arg(v) for k, v in options.items()}
    if any(_is_symbolic(v) for v in pyargs) or any(_is_symbolic(v) for v in pykw.values()):
        raise Unevaluated()
    try:
        result = fn(*pyargs, **pykw)
    except PermissionError:
        raise
    except _ARGERRORS as exc:
        raise _same_class(exc, f"PythonFunction: {exc or type(exc).__name__}") from exc
    except Exception as exc:                          # foreign failure: keep the class name, make it catchable
        raise ValueError(f"PythonFunction: Python raised {type(exc).__name__}: {exc}") from exc
    try:
        return _result(result, origin)
    except ValueError as exc:
        raise ValueError(f"PythonFunction: the Python result cannot be converted ({exc})") from exc


INTEROP_HEADS = frozenset({"PythonImport", "PythonAttribute"})


def apply_interop_head(head: str, args):
    """PythonImport["mod"] and PythonAttribute[handle, "attr"]; a symbol argument leaves the call unevaluated."""
    from .library.core import Unevaluated
    if any(isinstance(a, Symbol) for a in args):
        raise Unevaluated()
    if head == "PythonImport":
        if len(args) != 1:
            raise ValueError(f"PythonImport: called with {len(args)} argument(s); expected 1")
        return python_import(args[0])
    if len(args) != 2:
        raise ValueError(f"PythonAttribute: called with {len(args)} argument(s); expected 2")
    return python_attribute(args[0], args[1])


# ─────────────────────────── Pythonic call proxy ───────────────────────────

class Namespace:
    """``ml = session.ns``; ``ml.Integrate(x**2, x)`` builds Integrate[x^2, x], evaluates it, converts the result.

    Python arguments go through ``from_python``; keyword arguments become options (``WorkingPrecision=30`` ->
    ``WorkingPrecision -> 30``).
    """

    def __init__(self, session, out: str = "python", symbolic: str = "expr", arrays: str = "list"):
        check_modes(out, symbolic, arrays)
        self._session, self._out, self._symbolic, self._arrays = session, out, symbolic, arrays

    def __getattr__(self, head: str):
        if head.startswith("_") or not _NAME.match(head):
            raise AttributeError(head)

        def call(*args, **options):
            expr = Expr(head, *(from_python(a) for a in args),
                        *(Expr("Rule", Symbol(k), from_python(v)) for k, v in options.items()))
            return self._session.eval(expr, out=self._out, symbolic=self._symbolic, arrays=self._arrays)
        call.__name__ = call.__qualname__ = head
        return call

    def __call__(self, out: str = "python", symbolic: str = "expr", arrays: str = "list") -> "Namespace":
        """A proxy with other conversion modes: ``session.ns(arrays="numpy").Range(5)``."""
        return Namespace(self._session, out, symbolic, arrays)

    def symbols(self, names: str):
        """Symbols for building expressions in Python: ``x, y = ml.symbols("x y")``."""
        from .expr import symbols
        return symbols(names)
