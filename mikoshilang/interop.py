"""Python interop prototype: one conversion table, a Pythonic call proxy, typed Python functions, PythonFunction.

Design: /workspace/mikoshilang-conventions/PYTHON_INTEROP.md. Prototype on a local branch; not part of any release.

Conversion table (Python -> MikoshiLang -> Python round-trips exactly for every row marked <->):

    bool               <->  True / False                         (Boolean)
    int, numpy.integer  ->  Integer (int)                        (Integer)        int <-> Integer
    float, numpy.floating -> machine Real (float)                (Real)           float <-> Real
    complex            <->  Plus[re, Times[im, I]] machine       (Number)
    fractions.Fraction <->  Rational                             (Rational)
    sympy.Float (prec) <->  bignum Real                          (Real)
    str                <->  String                               (String)
    None               <->  Null
    float('inf')       <->  Infinity / -Infinity
    float('nan')        ->  ValueError (MikoshiLang has no NaN; never silently Indeterminate)
    list               <->  List                                 (ListOf[T]);  tuple -> List (one way)
    dict               <->  Association                          (Assoc)
    numpy.ndarray       ->  nested List (numeric or bool dtype; object dtype element-wise)
                            List -> numpy.ndarray with arrays="numpy" (rectangular, machine numbers)
    pandas.Series       ->  List (RangeIndex) or Association (index -> value)
    pandas.DataFrame    ->  List of Associations (rows), until Tabular/Dataset land
    sympy.Basic        <->  expression (to_python(..., symbolic="sympy"))
    sympy.MatrixBase    ->  nested List
    Expr / Symbol      <->  itself (arguments converted recursively)
    anything else       ->  TypeError (no inert foreign atoms)
"""
from __future__ import annotations

import fractions
import functools
import importlib
import inspect
import math
import typing
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional

import sympy as sp

from .expr import Expr, Symbol

_PYFUNCS = "\x00python_functions"     # SessionState.ext kind (session-wide) holding registered Python functions


# ─────────────────────────── Python -> MikoshiLang ───────────────────────────

def from_python(obj: Any) -> Any:
    """Convert a Python value to a MikoshiLang expression (see the module table); unknown types raise TypeError."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return _from_float(obj)
    if isinstance(obj, complex):
        if math.isnan(obj.real) or math.isnan(obj.imag):
            raise ValueError("from_python: NaN has no MikoshiLang value")
        return _from_sympy(sp.Float(obj.real) + sp.Float(obj.imag) * sp.I) if obj.imag else _from_float(obj.real)
    if isinstance(obj, fractions.Fraction):
        return obj.numerator if obj.denominator == 1 else _from_sympy(sp.Rational(obj.numerator, obj.denominator))
    if isinstance(obj, (Symbol,)):
        return obj
    if isinstance(obj, Expr):
        head = obj.head if isinstance(obj.head, str) else from_python(obj.head)
        return Expr(head, *(from_python(a) for a in obj.args))
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
            if obj.dtype.kind in "biufc":
                return from_python(obj.tolist())
            if obj.dtype.kind == "O":
                return from_python([from_python(v) for v in obj.tolist()]) if obj.ndim else from_python(obj.item())
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
            return obj                                   # bignum (or machine) Real, precision kept
        return _from_sympy(obj)
    if callable(obj):
        raise TypeError("from_python: Python callables must be registered with @session.function or PythonFunction")
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

    symbolic: what to do with non-data expressions -- "expr" (keep the Expr), "sympy" (convert) or "error" (raise).
    arrays:   "list" (nested lists) or "numpy" (a rectangular List of machine numbers becomes an ndarray).
    """
    if e is None or isinstance(e, (bool, str, int, float)):
        return e
    if isinstance(e, complex):
        return e
    if isinstance(e, sp.Float):
        return e
    if isinstance(e, Symbol):
        if e.name == "Null":
            return None
        if e.name == "Infinity":
            return math.inf
        return _symbolic(e, symbolic)
    if isinstance(e, Expr):
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
        return e if symbolic == "sympy" else _symbolic(_from_sympy(e), symbolic)
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
    except Exception:
        return None
    if not isinstance(v, sp.Basic) or not v.is_number:
        return None
    if isinstance(v, sp.Rational):
        return fractions.Fraction(int(v.p), int(v.q))
    re_, im_ = v.as_real_imag()
    if isinstance(re_, (sp.Float, sp.Integer)) and isinstance(im_, (sp.Float, sp.Integer)) and v.has(sp.Float):
        if all(getattr(f, "_prec", 53) == 53 for f in v.atoms(sp.Float)):
            return complex(float(re_), float(im_))
    return None                                         # exact algebraic numbers (Sqrt[2], 1 + 2 I) stay symbolic


def _symbolic(e, mode):
    if mode == "expr":
        return e
    if mode == "sympy":
        from .math_ops import to_sympy
        return to_sympy(e)
    raise TypeError(f"to_python: {e!r} is symbolic and has no Python data form")


def _as_array(e):
    np = _numpy()
    if np is None:
        return None
    def scalars(x):
        if isinstance(x, Expr) and x.head == "List":
            return [scalars(a) for a in x.args]
        v = to_python(x, symbolic="error") if not isinstance(x, (int, float, bool)) else x
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
    import sys
    return sys.modules.get("pandas")                  # only if the caller already uses pandas: never import it


# ─────────────────────────── types from Python hints ───────────────────────────

_VOCAB = {"Expr", "Integer", "PosInteger", "NonNegInteger", "Rational", "Real", "Number", "Numeric", "String",
          "Boolean", "Var", "Assoc", "Rule", "Fn", "Held"}


def mtype(hint) -> str:
    """The CONVENTIONS.md type-vocabulary name for a Python type hint (or a vocabulary string)."""
    if hint is inspect.Parameter.empty or hint is Any:
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
    if origin in (list, tuple, typing.List, typing.Sequence) or hint in (list, tuple):
        return f"ListOf[{mtype(targs[0]) if targs else 'Expr'}]"
    if origin is dict:
        return "Assoc"
    if origin is typing.Union:
        return " | ".join(mtype(t) for t in targs if t is not type(None))
    return "Expr"


def _coerce(name, pos, value, hint):
    """Convert an evaluated MikoshiLang argument for a hinted Python parameter (CONVENTIONS.md §2 order)."""
    from .library.core import Unevaluated
    if hint is inspect.Parameter.empty or hint is Any or hint is Expr:
        return value
    np = _numpy()
    if isinstance(hint, type) and issubclass(hint, sp.Basic):
        from .math_ops import to_sympy
        return to_sympy(value)
    py = to_python(value, symbolic="expr")
    want = mtype(hint)
    def bad():
        if isinstance(py, (Expr, Symbol)) and not (isinstance(py, Expr) and py.head in ("List", "Association")):
            raise Unevaluated()                      # symbolic input: stay unevaluated
        raise TypeError(f"{name}: argument {pos} must be {want}, not {value!r}")
    if hint is bool:
        return py if isinstance(py, bool) else bad()
    if hint is int:
        return py if isinstance(py, int) and not isinstance(py, bool) else bad()
    if hint is float:
        if isinstance(py, bool):
            bad()
        if isinstance(py, (int, float, fractions.Fraction, sp.Float)):
            return float(py)
        return bad()
    if hint is complex:
        if isinstance(py, (int, float, complex, fractions.Fraction, sp.Float)) and not isinstance(py, bool):
            return complex(py)
        return bad()
    if hint is fractions.Fraction:
        return fractions.Fraction(py) if isinstance(py, (int, fractions.Fraction)) and not isinstance(py, bool) else bad()
    if hint is str:
        return py if isinstance(py, str) else bad()
    if np is not None and hint is np.ndarray:
        arr = to_python(value, arrays="numpy")
        return arr if isinstance(arr, np.ndarray) else bad()
    origin = typing.get_origin(hint) or hint
    if origin in (list, tuple):
        if not isinstance(py, list):
            return bad()
        targs = typing.get_args(hint)
        if targs:
            items = value.args
            return [ _coerce(name, f"{pos}[{i + 1}]", v, targs[0]) for i, v in enumerate(items)]
        return py
    if origin is dict:
        return py if isinstance(py, dict) else bad()
    return py


# ─────────────────────────── registration and calls ───────────────────────────

class PythonFunctionInfo:
    __slots__ = ("name", "fn", "params", "returns", "signature", "listable", "notes")

    def __init__(self, name, fn, params, returns, signature, listable, notes):
        self.name, self.fn, self.params, self.returns = name, fn, params, returns
        self.signature, self.listable, self.notes = signature, listable, notes


def register(state, fn: Callable, name: Optional[str] = None, listable: bool = False, notes: str = "",
             check: str = "delegated", override: bool = False) -> PythonFunctionInfo:
    """Register a typed Python function as a session-local MikoshiLang head."""
    name = name or _wl_name(fn.__name__)
    from . import library
    from . import kernel as K
    if not override and (name in library.REGISTRY or name in getattr(K, "_BUILTINS", {})):
        raise ValueError(f"function: {name} is a built-in head; pass override=True to shadow it in this session")
    hints = typing.get_type_hints(fn) if not isinstance(fn, functools.partial) else {}
    sig = inspect.signature(fn)
    params = [(p.name, hints.get(p.name, p.annotation), p.kind, p.default) for p in sig.parameters.values()]
    returns = hints.get("return", sig.return_annotation)
    signature = "{" + ", ".join(mtype(h) + ("..." if k == p_varargs() else "") for _, h, k, _ in params) + "} -> " \
                + mtype(returns)
    qual = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', name)}"
    notes = (notes + " " if notes else "") + f"Check: {check} (user Python function {qual})."
    info = PythonFunctionInfo(name, fn, params, returns, signature, listable, notes)
    state.ext.setdefault(_PYFUNCS, {})[name] = info
    return info


def p_varargs():
    return inspect.Parameter.VAR_POSITIONAL


def _wl_name(pyname: str) -> str:
    """snake_case -> CamelCase (my_func -> MyFunc); names already CamelCase are kept."""
    return pyname if pyname[:1].isupper() else "".join(p[:1].upper() + p[1:] for p in pyname.split("_") if p)


def lookup(state, head):
    table = state.ext.get(_PYFUNCS)
    return table.get(head) if table else None


def call_registered(info: PythonFunctionInfo, args):
    """Apply a registered Python function to evaluated MikoshiLang arguments (raises per CONVENTIONS.md §3)."""
    if info.listable and any(isinstance(a, Expr) and a.head == "List" for a in args):
        n = {len(a.args) for a in args if isinstance(a, Expr) and a.head == "List"}
        if len(n) != 1:
            raise ValueError(f"{info.name}: cannot thread over lists of unequal length")
        k = n.pop()
        return Expr("List", *(Expr(info.name, *(a.args[i] if isinstance(a, Expr) and a.head == "List" else a
                                                for a in args)) for i in range(k)))
    fixed = [p for p in info.params if p[2] in (p[2].POSITIONAL_ONLY, p[2].POSITIONAL_OR_KEYWORD)]
    var = next((p for p in info.params if p[2] == p[2].VAR_POSITIONAL), None)
    lo = sum(1 for p in fixed if p[3] is inspect.Parameter.empty)
    if len(args) < lo or (var is None and len(args) > len(fixed)):
        raise ValueError(f"{info.name} called with {len(args)} argument(s); signature {info.signature}")
    pyargs = []
    for i, a in enumerate(args):
        hint = fixed[i][1] if i < len(fixed) else var[1]
        pyargs.append(_coerce(info.name, i + 1, a, hint))
    return _invoke(info.name, info.fn, pyargs)


def _invoke(name, fn, pyargs):
    try:
        result = fn(*pyargs)
    except (TypeError, ValueError, NotImplementedError, ArithmeticError) as exc:
        text = str(exc)
        if text.startswith(f"{name}:"):
            raise
        raise type(exc)(f"{name}: {text or type(exc).__name__}") from exc
    except PermissionError:
        raise
    except Exception as exc:                          # foreign failure: keep the class name, make it catchable
        raise ValueError(f"{name}: Python raised {type(exc).__name__}: {exc}") from exc
    return from_python(result)


# ─────────────────────────── PythonFunction["module.attr"] (capability-gated) ───────────────────────────

_python = ContextVar("mikoshilang_allow_python", default=False)


def _allowed(module: str) -> bool:
    perm = _python.get()
    if perm is True:
        return True
    if not perm:
        return False
    top = module.split(".")[0]
    return module in perm or top in perm


def resolve(path: str):
    """Import module.attr (longest importable module prefix), subject to Session(allow_python=...)."""
    if not isinstance(path, str) or not path or path.startswith(".") or "__" in path:
        raise ValueError(f"PythonFunction: {path!r} is not a dotted Python name")
    parts = path.split(".")
    if not _allowed(path):
        raise PermissionError(f"PythonFunction: calling {path} requires Session(allow_python=True or a set "
                              f"containing {parts[0]!r})")
    for cut in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:cut])
        try:
            obj = importlib.import_module(mod_name)
        except ImportError:
            continue
        for attr in parts[cut:]:
            obj = getattr(obj, attr)
        if not callable(obj):
            raise TypeError(f"PythonFunction: {path} is not callable")
        return obj
    raise ValueError(f"PythonFunction: cannot import {path}")


def apply_python_function(head: Expr, args):
    """PythonFunction["mod.f"][args...]: symbolic arguments leave the call unevaluated."""
    from .library.core import Unevaluated
    if len(head.args) != 1:
        raise ValueError("PythonFunction: expected PythonFunction[\"module.name\"]")
    fn = resolve(head.args[0])
    pyargs = []
    for a in args:
        v = to_python(a, symbolic="expr")
        if isinstance(v, (Expr, Symbol)) and not (isinstance(v, Expr) and v.head in ("List", "Association")):
            raise Unevaluated()
        pyargs.append(to_python(a, symbolic="error"))
    return _invoke(f"PythonFunction[{head.args[0]!r}]", fn, pyargs)


# ─────────────────────────── Pythonic call proxy ───────────────────────────

class Namespace:
    """``ml = session.ns``; ``ml.Integrate(x**2, x)`` builds Integrate[x^2, x], evaluates it, converts the result."""

    def __init__(self, session, out: str = "python", symbolic: str = "expr", arrays: str = "list"):
        self._session, self._out, self._symbolic, self._arrays = session, out, symbolic, arrays

    def __getattr__(self, head: str):
        if head.startswith("_"):
            raise AttributeError(head)
        def call(*args, **options):
            expr = Expr(head, *(from_python(a) for a in args),
                        *(Expr("Rule", Symbol(k), from_python(v)) for k, v in options.items()))
            return self._session.eval(expr, out=self._out, symbolic=self._symbolic, arrays=self._arrays)
        call.__name__ = head
        return call

    def symbols(self, names: str):
        from .expr import symbols
        return symbols(names)
