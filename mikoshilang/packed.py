"""Packed arrays: machine-number lists backed by a NumPy array.

A :class:`PackedArray` *is* a ``List`` (``Head`` is ``List``, ``Part``, ``Map``, patterns and printing see
ordinary elements), but it stores its elements as one read-only ``numpy.ndarray`` of ``int64``, ``float64`` or
``complex128``. Elements are materialised only when something walks them. Like Wolfram's packed arrays the
representation is invisible to the language: a packed and an unpacked list with the same elements are equal.

Where packed arrays come from:

* Python: ``from_python(ndarray)`` (``Session.eval``, ``Session.ns``) packs machine-number arrays;
* ``Developer`ToPackedArray[list]`` and ``NumericArray[...]`` (its values are stored packed);
* results of the fast operations below;
* ``Dot``, ``Total`` and ``Mean`` pack plain machine-number lists of at least ``PACKING_THRESHOLD`` elements.

Fast operations (NumPy, never element by element): ``Dot``, ``Total``, ``Mean``, ``Max``, ``Min``,
``Transpose``, ``Length``, ``Dimensions``, ``N``, ``Normal``, ``Plus``, ``Times``, ``Power`` (packed base,
scalar exponent) and the machine-real elementary functions (``Sin``, ``Exp``, ``Log``, ``Sqrt``, ...). Each one
returns ``UNHANDLED`` (and the ordinary, exact code path runs) whenever NumPy could differ from MikoshiLang:
integer results that might overflow ``int64``, exact rationals, symbolic arguments, mismatched shapes, values
outside a function's real domain (``Sqrt`` of a negative, ``Log`` of a non-positive) or NaN/Infinity results.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import sympy as sp

from .expr import Expr, Symbol

PACKING_THRESHOLD = 250          # Dot / Total / Mean pack plain lists with at least this many elements
_INT_SAFE = 2 ** 62              # integer results must stay below this (int64 headroom)
UNHANDLED = object()

_DTYPES = {"i": np.int64, "u": np.int64, "f": np.float64, "c": np.complex128}


class PackedArray(Expr):
    """A List of machine numbers stored as one read-only ndarray (int64, float64 or complex128)."""

    __slots__ = ("array", "_cache")

    def __init__(self, array):
        arr = np.asarray(array)
        if arr.ndim == 0 or arr.size == 0:
            raise ValueError("PackedArray: a non-empty array of rank >= 1 is expected")
        dt = _DTYPES.get(arr.dtype.kind)
        if dt is None:
            raise TypeError(f"PackedArray: dtype {arr.dtype} is not a machine number type")
        if arr.dtype != dt:
            arr = arr.astype(dt)
        elif arr.flags.writeable:
            arr = arr.copy()
        if arr.dtype.kind != "i" and not np.isfinite(arr).all():
            raise ValueError("PackedArray: NaN and infinite values cannot be packed")
        arr.setflags(write=False)
        self.head = "List"
        self.array = arr
        self._cache = None

    @property
    def args(self):
        if self._cache is None:
            a = self.array
            if a.ndim > 1:
                self._cache = tuple(_view(row) for row in a)
            elif a.dtype.kind == "c":
                from .interop import from_python
                self._cache = tuple(from_python(z) for z in a.tolist())
            else:
                self._cache = tuple(a.tolist())
        return self._cache

    def __eq__(self, other):
        if isinstance(other, PackedArray):
            return self.array.shape == other.array.shape and bool(np.array_equal(self.array, other.array))
        return Expr.__eq__(self, other)

    def __hash__(self):
        return Expr.__hash__(self)

    def __reduce__(self):
        return (PackedArray, (np.array(self.array),))


def _view(row):
    p = PackedArray.__new__(PackedArray)
    p.head, p.array, p._cache = "List", row, None
    return p


def is_packed(x) -> bool:
    return x.__class__ is PackedArray


# ─────────────────────────── packing ───────────────────────────

class _NotPackable(Exception):
    pass


def _leaves(x):
    if x.__class__ is PackedArray:
        return x.array.tolist()
    if isinstance(x, Expr) and x.head == "List" and x.args:
        return [_leaves(a) for a in x.args]
    t = x.__class__
    if t is int or t is float:
        return x
    if t is sp.Float and x._prec <= 53:
        return float(x)
    raise _NotPackable


def pack(x, threshold: int = 0):
    """``x`` as a PackedArray if it is a non-empty rectangular list of machine integers or reals (int64 range)
    with at least ``threshold`` elements, else None. Exact rationals, complex and symbolic entries never pack."""
    if x.__class__ is PackedArray:
        return x
    if not (isinstance(x, Expr) and x.head == "List" and x.args):
        return None
    if threshold and _estimate(x) < threshold:
        return None
    try:
        data = _leaves(x)
    except _NotPackable:
        return None
    try:
        arr = np.array(data)
    except (ValueError, OverflowError):
        return None
    if arr.dtype.kind not in "if" or arr.size == 0:
        return None
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        return None
    return PackedArray(arr)


def _estimate(x):
    n = 1
    while isinstance(x, Expr) and x.head == "List" and x.args:
        n *= len(x.args)
        x = x.args[0]
    return n


def from_array(arr):
    """A Python ndarray as a PackedArray when possible (machine dtype, finite, non-empty), else None."""
    if arr.ndim == 0 or arr.size == 0 or arr.dtype.kind not in "iufc":
        return None
    if arr.dtype.kind == "u" and arr.size and int(arr.max()) >= 2 ** 63:
        return None
    if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
        return None
    return PackedArray(arr)


def _out(arr):
    """A NumPy result as a MikoshiLang value: a PackedArray, or a Python scalar for rank 0."""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        v = arr.item()
        if isinstance(v, complex):
            from .interop import from_python
            return from_python(v)
        return v
    return PackedArray(arr)


# ─────────────────────────── fast operations ───────────────────────────

def _bound(arr):
    return int(np.abs(arr).max()) if arr.size else 0


def _scalar(x):
    t = x.__class__
    if t is int or t is float:
        return x
    if t is sp.Float and x._prec <= 53:
        return float(x)
    return None


def _dot(args):
    if len(args) < 2:
        return UNHANDLED
    big = any(is_packed(a) for a in args) or any(_estimate(a) >= PACKING_THRESHOLD for a in args)
    if not big:
        return UNHANDLED
    arrs = [pack(a) for a in args]
    if any(p is None for p in arrs):
        return UNHANDLED
    arrs = [p.array for p in arrs]
    if any(a.ndim > 2 for a in arrs):
        return UNHANDLED
    for a, b in zip(arrs, arrs[1:]):
        if a.shape[-1] != b.shape[0]:
            return UNHANDLED                           # the library raises its own shape error
    if all(a.dtype.kind == "i" for a in arrs):
        bound = 1
        for a in arrs:
            bound *= max(_bound(a), 1) * a.shape[0]
        if bound >= _INT_SAFE:
            return UNHANDLED                           # could overflow int64: exact path
    out = arrs[0]
    for b in arrs[1:]:
        out = np.dot(out, b)
    return _out(out)


def _total(args):
    if len(args) != 1:
        return UNHANDLED
    p = args[0] if is_packed(args[0]) else pack(args[0], PACKING_THRESHOLD)
    if p is None:
        return UNHANDLED
    a = p.array
    if a.dtype.kind == "i" and _bound(a) * a.shape[0] >= _INT_SAFE:
        return UNHANDLED
    return _out(a.sum(axis=0))


def _mean(args):
    if len(args) != 1:
        return UNHANDLED
    p = args[0] if is_packed(args[0]) else pack(args[0], PACKING_THRESHOLD)
    if p is None:
        return UNHANDLED
    a = p.array
    if a.dtype.kind == "f" or a.dtype.kind == "c":
        return _out(a.mean(axis=0))
    if _bound(a) * a.shape[0] >= _INT_SAFE:
        return UNHANDLED
    from fractions import Fraction
    from .interop import from_python
    s = a.sum(axis=0)
    n = a.shape[0]
    if s.ndim == 0:
        return from_python(Fraction(int(s), n))        # exact mean of integers, as in Wolfram
    return from_python([Fraction(int(v), n) for v in s.ravel().tolist()]) if s.ndim == 1 else UNHANDLED


def _maxmin(f):
    def op(args):
        if len(args) != 1 or not is_packed(args[0]) or args[0].array.dtype.kind == "c":
            return UNHANDLED
        return _out(f(args[0].array))
    return op


def _transpose(args):
    if len(args) != 1 or not is_packed(args[0]) or args[0].array.ndim != 2:
        return UNHANDLED
    return PackedArray(args[0].array.T)


def _length(args):
    if len(args) != 1 or not is_packed(args[0]):
        return UNHANDLED
    return int(args[0].array.shape[0])


def _dimensions(args):
    if len(args) != 1 or not is_packed(args[0]):
        return UNHANDLED
    return PackedArray(np.array(args[0].array.shape, dtype=np.int64))


def _n(args):
    if len(args) != 1 or not is_packed(args[0]):
        return UNHANDLED
    a = args[0].array
    return args[0] if a.dtype.kind in "fc" else PackedArray(a.astype(np.float64))


def _normal(args):
    if len(args) != 1 or not is_packed(args[0]):
        return UNHANDLED
    return args[0]


def _operands(args):
    """Arrays (same shape) and machine scalars for an elementwise operation, or None."""
    if not any(is_packed(a) for a in args):
        return None
    out, shape, rationals = [], None, False
    for a in args:
        s = _scalar(a)
        if s is None and not isinstance(a, Expr) or (isinstance(a, Expr) and a.head in ("Times", "Power",
                                                                                        "Rational")):
            s = _exact_scalar(a)
            rationals = rationals or s is not None and s.__class__ is float
        if s is not None:
            out.append(s)
            continue
        p = pack(a)
        if p is None:
            return None
        if shape is None:
            shape = p.array.shape
        elif p.array.shape != shape:
            return None                                # Wolfram threads first levels: leave it to Listable
        out.append(p.array)
    if rationals and not any(isinstance(o, np.ndarray) and o.dtype.kind in "fc" for o in out):
        return None                                    # exact rational with integers: exact path
    return out


def _exact_scalar(a):
    """An exact integer (int) or rational (as float, only combined with machine reals) scalar, else None."""
    if isinstance(a, bool) or isinstance(a, str):
        return None
    try:
        from .math_ops import to_sympy
        v = to_sympy(a)
    except Exception:          # not a number
        return None
    if isinstance(v, sp.Integer):
        return int(v)
    if isinstance(v, sp.Rational):
        return float(v)
    return None


def _all_int(ops):
    return all((isinstance(o, np.ndarray) and o.dtype.kind == "i") or o.__class__ is int for o in ops)


def _absmax(o):
    return _bound(o) if isinstance(o, np.ndarray) else abs(o)


def _plus(args):
    ops = _operands(args)
    if ops is None:
        return UNHANDLED
    if _all_int(ops) and sum(_absmax(o) for o in ops) >= _INT_SAFE:
        return UNHANDLED
    out = ops[0]
    for o in ops[1:]:
        out = np.add(out, o)
    return _finite(out)


def _times(args):
    ops = _operands(args)
    if ops is None:
        return UNHANDLED
    if _all_int(ops):
        bound = 1
        for o in ops:
            bound *= max(_absmax(o), 1)
        if bound >= _INT_SAFE:
            return UNHANDLED
    elif any(not isinstance(o, np.ndarray) and o == 0 and o.__class__ is int for o in ops):
        return UNHANDLED                               # exact 0 times reals is exact 0 (Wolfram): exact path
    out = ops[0]
    for o in ops[1:]:
        out = np.multiply(out, o)
    return _finite(out)


def _power(args):
    if len(args) != 2 or not is_packed(args[0]):
        return UNHANDLED
    a, e = args[0].array, _scalar(args[1])
    if e is None or a.dtype.kind == "c":
        return UNHANDLED
    if a.dtype.kind == "i":
        if e.__class__ is not int or e < 0 or e > 64 or (max(_bound(a), 1) ** e) >= _INT_SAFE:
            return UNHANDLED                           # rationals, overflow: exact path
        return _out(np.power(a, e))
    if (e.__class__ is float or not float(e).is_integer()) and (a < 0).any():
        return UNHANDLED                               # complex results
    if e < 0 and (a == 0).any():
        return UNHANDLED                               # ComplexInfinity
    return _finite(np.power(a, e))


def _finite(out):
    out = np.asarray(out)
    if out.dtype.kind in "fc" and not np.isfinite(out).all():
        return UNHANDLED
    return _out(out)


def _real_fn(f, domain=None):
    def op(args):
        if len(args) != 1 or not is_packed(args[0]) or args[0].array.dtype.kind != "f":
            return UNHANDLED                           # exact integers stay exact (Sin[1]); complex: exact path
        a = args[0].array
        if domain is not None and not domain(a):
            return UNHANDLED
        with np.errstate(all="ignore"):
            return _finite(f(a))
    return op


def _abs(args):
    if len(args) != 1 or not is_packed(args[0]):
        return UNHANDLED
    return _out(np.abs(args[0].array))


FAST = {
    "Dot": _dot, "Total": _total, "Mean": _mean, "Max": _maxmin(np.max), "Min": _maxmin(np.min),
    "Transpose": _transpose, "Length": _length, "Dimensions": _dimensions, "N": _n, "Normal": _normal,
    "Plus": _plus, "Times": _times, "Power": _power, "Abs": _abs,
    "Sin": _real_fn(np.sin), "Cos": _real_fn(np.cos), "Tan": _real_fn(np.tan), "Exp": _real_fn(np.exp),
    "Sinh": _real_fn(np.sinh), "Cosh": _real_fn(np.cosh), "Tanh": _real_fn(np.tanh),
    "ArcTan": _real_fn(np.arctan),
    "Log": _real_fn(np.log, lambda a: bool((a > 0).all())),
    "Sqrt": _real_fn(np.sqrt, lambda a: bool((a >= 0).all())),
}
_AUTO = {"Dot", "Total", "Mean"}                         # also pack large plain lists


def apply(head: str, args) -> Any:
    """The NumPy result of head[args] for packed (or large machine-number) arguments, else UNHANDLED."""
    op = FAST.get(head)
    if op is None:
        return UNHANDLED
    if head not in _AUTO and not any(a.__class__ is PackedArray for a in args):
        return UNHANDLED
    return op(args)
