"""Kernel linear-algebra builtins (coverage batch 3): Wolfram-correct Dot, Det and Inverse.

Loaded at the end of kernel.py (like kernel_lists) so that they replace the legacy kernel builtins before any
evaluation.  Audit statuses and notes are recorded in mikoshilang/library/wl_linalg.py.
"""
from functools import reduce

import sympy as sp

from .expr import Expr
from . import kernel as K
from .kernel import is_list, L


def S(x):
    from .math_ops import to_sympy
    if isinstance(x, (sp.Basic, sp.MatrixBase)):
        return x
    return to_sympy(x)


def R(x):
    """SymPy / Python / NumPy value -> MikoshiLang."""
    from .math_ops import from_sympy
    if isinstance(x, bool):
        return x
    if isinstance(x, (list, tuple)):
        return L(*(R(v) for v in x))
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return R(x.tolist())
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.complexfloating):
            x = complex(x)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(x, complex):
        if x.imag == 0:
            return float(x.real)
        return from_sympy(sp.Float(x.real) + sp.Float(x.imag) * sp.I)
    if isinstance(x, (int, float)):
        return x
    return from_sympy(sp.sympify(x))


def has_float(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (float, complex, sp.Float)):
        return True
    if isinstance(x, sp.Basic):
        return any(isinstance(a, sp.Float) for a in sp.preorder_traversal(x))
    if isinstance(x, Expr):
        return any(has_float(a) for a in x.args)
    return False



def dims(x):
    """Full dimensions of a rectangular array (lists all the way down to non-list entries); None if ragged."""
    if not is_list(x):
        return []
    if not x.args:
        return [0]
    subs = [dims(a) for a in x.args]
    if any(s is None for s in subs) or any(s != subs[0] for s in subs):
        return None
    return [len(x.args)] + subs[0]


def is_array(x):
    return is_list(x) and dims(x) is not None


def to_nested(x, conv=S):
    """MikoshiLang array -> nested Python lists of SymPy values."""
    if is_list(x):
        return [to_nested(a, conv) for a in x.args]
    return conv(x)


def from_nested(v):
    if isinstance(v, list):
        return L(*(from_nested(a) for a in v))
    return R(v)


def _kind(*inputs):
    """'exact', 'machine' or a bignum binary precision for the inputs."""
    if not any(has_float(i) for i in inputs):
        return "exact"
    from .precision import input_precision
    p = input_precision(inputs)
    return "machine" if p in ("m", None) else p


def _round(value, kind):
    """Round an exact SymPy result (scalar or nested list) to the input precision."""
    if isinstance(value, list):
        return [_round(v, kind) for v in value]
    if kind == "exact":
        return value
    if kind == "machine":
        return sp.sympify(value).evalf(15)
    import mpmath
    return sp.sympify(value).evalf(mpmath.libmp.prec_to_dps(kind))


def _out_nested(value, kind):
    v = _round(value, kind)
    if kind not in ("exact", "machine"):
        from .precision import round_to
        return round_to(from_nested(v), kind)
    return from_nested(v)


def _np():
    try:
        import numpy as np
        return np
    except ImportError:  # pragma: no cover
        raise NotImplementedError("this operation on machine-precision matrices requires NumPy")


def _npmat(M):
    np = _np()
    a = np.array([[complex(v) for v in row] for row in M.evalf(17).tolist()])
    if not np.any(a.imag):
        a = a.real
    return a


def _np_out(a):
    np = _np()
    a = np.asarray(a)
    if np.iscomplexobj(a) and not np.any(a.imag):
        a = a.real
    return R(a.tolist())


def _zero(v):
    """Exact zero test (True / False / None when undecidable)."""
    v = sp.sympify(v)
    if v == 0:
        return True
    z = v.is_zero
    if z is not None:
        return z
    s = sp.simplify(v)
    if s == 0:
        return True
    z = s.is_zero
    if z is None and v.has(sp.exp, sp.Pow):
        c = sp.simplify(sp.expand_complex(v))
        if c == 0:
            return True
        z = c.is_zero
    return z


def _expand_poly(v):
    try:
        if v.free_symbols and v.is_polynomial(*v.free_symbols):
            return sp.expand(v)
    except Exception:
        pass
    return v


def _is_matrix(x):
    d = dims(x) if is_list(x) else None
    return d is not None and len(d) == 2 and d[1] > 0


# ─────────────────────────── Dot (kernel builtin) ───────────────────────────

def _tdot(a, b):
    """Contract the last index of nested list a with the first index of nested list b."""
    if isinstance(a[0], list) if a else False:
        return [_tdot(r, b) for r in a]
    if len(a) != len(b):
        raise ValueError("Dot: tensors have incompatible shapes")
    if not b:
        return sp.Integer(0)
    if isinstance(b[0], list):
        return reduce(_tadd, (_tscale(x, y) for x, y in zip(a, b)))
    return sp.Add(*(x * y for x, y in zip(a, b)))


def _tscale(c, t):
    if isinstance(t, list):
        return [_tscale(c, u) for u in t]
    return c * t


def _tadd(s, t):
    if isinstance(s, list):
        return [_tadd(u, v) for u, v in zip(s, t)]
    return s + t


@K.builtin("Dot")
def _b_dot(a, ev_, expr):
    if not a:
        return expr
    if len(a) == 1:
        return a[0]
    if not all(is_list(x) for x in a):
        return expr                                  # Dot[a, b] with symbolic factors stays unevaluated
    shapes = [dims(x) for x in a]
    if any(s is None for s in shapes):
        raise ValueError("Dot: arguments must be rectangular arrays")
    kind = _kind(*a)
    acc = to_nested(a[0])
    shape = shapes[0]
    for x, s in zip(a[1:], shapes[1:]):
        if shape[-1] != s[0]:
            raise ValueError(f"Dot: tensors with dimensions {shape} and {s} cannot be contracted")
        acc = _tdot(acc, to_nested(x))
        shape = shape[:-1] + s[1:]
    return _out_nested(acc, kind)


K.BUILTIN_ATTRIBUTES.setdefault("Dot", set()).update({"Flat", "OneIdentity", "Protected"})
K._BUILTINS["DotProduct"] = _b_dot


# ─────────────────────────── Det / Inverse (kernel builtins) ───────────────────────────

def _square_or_unevaluated(x, fname):
    if not is_list(x):
        return None
    d = dims(x)
    if d is None or len(d) != 2 or d[0] != d[1] or d[0] == 0:
        raise ValueError(f"{fname}: argument must be a non-empty square matrix")
    return sp.Matrix([[S(v) for v in r.args] for r in x.args])


@K.builtin("Det")
def _b_det(a, ev_, expr):
    K._need(a, 1, "Det")
    M = _square_or_unevaluated(a[0], "Det")
    if M is None:
        return expr
    kind = _kind(a[0])
    if kind == "machine":
        np = _np()
        return _np_out(np.linalg.det(_npmat(M)))
    d = M.det(method="bareiss")
    if kind == "exact":
        d = sp.sympify(d)
        d = _expand_poly(sp.cancel(d)) if d.free_symbols else (d if d.is_Rational else sp.simplify(d))
        return R(d)
    return _out_nested(d, kind)




@K.builtin("Inverse")
def _b_inverse(a, ev_, expr):
    K._need(a, 1, "Inverse")
    M = _square_or_unevaluated(a[0], "Inverse")
    if M is None:
        return expr
    kind = _kind(a[0])
    if kind == "machine":
        np = _np()
        A = _npmat(M)
        if np.linalg.matrix_rank(A) < A.shape[0]:
            raise ValueError("Inverse: matrix is singular")
        return _np_out(np.linalg.inv(A))
    if _zero(M.det(method="bareiss")) is True:
        raise ValueError("Inverse: matrix is singular")
    inv = M.inv(method="LU") if kind == "exact" else M.inv()
    if kind == "exact":
        inv = inv.applyfunc(lambda v: sp.cancel(v) if v.free_symbols else sp.simplify(v))
    return _out_nested(inv.tolist(), kind)
