"""Python interop pillar, phase 2: reserved external-evaluation heads and packed-array utilities.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). The Python-calling heads themselves (PythonFunction, PythonImport, PythonAttribute) are
session-gated and live in ``mikoshilang.interop``; see docs/python_interop.md.
"""
from __future__ import annotations

import numpy as np

from ..expr import Expr, Symbol
from ..packed import PackedArray, is_packed, pack
from .core import Unevaluated, error, fn

AREA = "interop (I2)"

_EXTERNAL = ("running external code strings is not supported (there is no sandbox); call Python through "
             "PythonFunction[\"module.name\"] with Session(allow_python=...) or register a function with "
             "@session.function")

for _name in ("ExternalEvaluate", "ExternalFunction", "StartExternalSession"):
    error(_name, AREA, _EXTERNAL,
          notes=f"{_name} is reserved and always raises NotImplementedError: MikoshiLang never evaluates code "
                "strings in-process. Python is reachable only through PythonFunction / PythonImport (gated by "
                "Session(allow_python=...)) and @session.function. Check: reference (Raises cases in tests).")

_TYPES = {"Integer": np.int64, "Real": np.float64, "Complex": np.complex128}


def _type_name(t, fname):
    if isinstance(t, Symbol) and t.name in _TYPES:
        return t.name
    if isinstance(t, Symbol):
        raise ValueError(f"{fname}: the type must be Integer, Real or Complex, not {t.name}")
    raise TypeError(f"{fname}: the type must be the symbol Integer, Real or Complex, not {t!r}")


@fn("ToPackedArray", area=AREA, status="added",
    notes="ToPackedArray[list] stores a rectangular list of machine integers or reals as one NumPy array (a "
          "packed List: same elements, same Head, equal to the unpacked list); a list that cannot be packed "
          "(symbolic, exact rational, ragged, bignum, empty) is returned unchanged, as in Wolfram. "
          "ToPackedArray[list, type] with type Integer or Real converts machine numbers to that type; an "
          "unknown type raises ValueError, a non-symbol type TypeError, Integer for non-integers ValueError, "
          "Complex NotImplementedError (complex arrays are packed only when they come from NumPy). "
          "Also available as Developer`ToPackedArray. "
          "Check: exact (lossless for machine numbers); round trip through FromPackedArray in tests.")
def _to_packed_array(x: Expr, typ: Expr = None) -> Expr:
    if typ is None:
        p = pack(x)
        return p if p is not None else x
    t = _type_name(typ, "ToPackedArray")
    if t == "Complex":
        raise NotImplementedError("ToPackedArray: packing as Complex is not supported (complex arrays are packed "
                                  "when they come from NumPy)")
    p = pack(x)
    if p is None:
        return x
    a = p.array
    if t == "Integer":
        if a.dtype.kind != "i":
            raise ValueError("ToPackedArray: the list has non-integer entries and cannot be packed as Integer")
        return p
    return PackedArray(a.astype(_TYPES[t]))


@fn("FromPackedArray", area=AREA, status="added",
    notes="FromPackedArray[list] returns the same list with its elements stored one by one (an unpacked "
          "List); any other expression is returned unchanged. Also available as Developer`FromPackedArray. "
          "Check: exact; equality with the packed list in tests.")
def _from_packed_array(x: Expr) -> Expr:
    if not is_packed(x):
        return x

    def unpack(v):
        if is_packed(v):
            return Expr("List", *(unpack(a) for a in v.args))
        return v
    return unpack(x)


@fn("PackedArrayQ", area=AREA, status="added",
    notes="PackedArrayQ[expr] is True for a packed List; PackedArrayQ[expr, type] also tests the element type "
          "(Integer, Real or Complex) and PackedArrayQ[expr, type, rank] the array rank. Unknown types raise "
          "ValueError, a non-integer rank TypeError. Also available as Developer`PackedArrayQ. "
          "Check: exact; tested on packed and unpacked lists.")
def _packed_array_q(x: Expr, typ: Expr = None, rank: Integer = None) -> Boolean:
    t = _type_name(typ, "PackedArrayQ") if typ is not None else None
    if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int)):
        if isinstance(rank, Symbol):
            raise Unevaluated()
        raise TypeError(f"PackedArrayQ: the rank must be an integer, not {rank!r}")
    if not is_packed(x):
        return False
    a = x.array
    if t is not None and a.dtype != _TYPES[t]:
        return False
    return rank is None or a.ndim == rank


# Wolfram's context names (Developer`ToPackedArray, ...) evaluate as the plain heads; see evaluate.py step 9a.
CONTEXT_ALIASES = {"Developer`" + n: n for n in ("ToPackedArray", "FromPackedArray", "PackedArrayQ")}
