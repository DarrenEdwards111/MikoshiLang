"""Coverage batch W06: calculus extras.

Asymptotics (Asymptotic, AsymptoticSum/Integrate/Product/Solve/DSolveValue/RSolveValue/Expectation/Probability and
the asymptotic relations AsymptoticLess, ...), discrete calculus (DiscreteLimit, DifferenceQuotient, Casoratian,
DifferenceRoot / DifferentialRoot, generating functions, FindSequenceFunction), vector calculus (LineIntegrate,
SurfaceIntegrate, ContourIntegrate and their numerical forms, ResidueSum, FrenetSerretSystem, coordinate charts and
TransformedField), fractional calculus (FractionalD, CaputoD and numerical forms), dynamical systems (DFixedPoints,
DStabilityConditions, RFixedPoints, RStabilityConditions, DEigenvalues, DEigensystem, GreenFunction), matrix equations
(LyapunovSolve, DiscreteLyapunovSolve, RiccatiSolve, DiscreteRiccatiSolve), interpolation (InterpolatingPolynomial,
FunctionInterpolation) and symbolic arrays (VectorSymbol, MatrixSymbol, ArraySymbol, Symbolic*Array, ArrayExpand,
ArraySimplify, TensorSymmetry, TensorWedge, HodgeDual, TensorExpand, TensorReduce).

Quality bar: every symbolic result is either produced by an exact algorithm whose output is checked (substituted
back, differentiated back, compared with an independent numerical evaluation) or the call stays unevaluated /
raises a clear error.
"""
import itertools
import math

import mpmath
import sympy as sp
from sympy.core.function import AppliedUndef

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import (fn, record, R, L, rule, sym, is_list, Unevaluated, nonneg_int, int_arg, pos_int, has_float,
                   matrix_arg, mat_out)
from .wl_symbolic import out, wl_fix, is_true, is_false, is_rule

AREA = "calculus extras (W06)"
_sym_ = Symbol


def _ev(x):
    from ..evaluate import evaluate
    return evaluate(x, 1)


def S(x, fname):
    if isinstance(x, str):
        raise ValueError(f"{fname}: strings are not expressions")
    return K.to_sp(x)


def SX(x, fname):
    v = S(x, fname)
    if isinstance(v, list):
        raise ValueError(f"{fname}: expected an expression, got a list")
    return v


def _is(x, name):
    return isinstance(x, Symbol) and x.name == name


def _head(x, name=None):
    return isinstance(x, Expr) and isinstance(x.head, str) and (name is None or x.head == name)


def _svar(x, fname, what="variable"):
    if isinstance(x, Symbol) and x.name not in ("True", "False", "Pi", "E", "I", "Infinity"):
        return sp.Symbol(x.name)
    raise ValueError(f"{fname}: {x!r} is not a valid {what}")


def _opts(args, names, fname):
    """Trailing Name -> value options (Name in names); other trailing rules whose lhs is a symbol raise."""
    args = list(args)
    o = {}
    while args and is_rule(args[-1]) and isinstance(args[-1].args[0], Symbol):
        nm = args[-1].args[0].name
        if nm not in names:
            if nm[:1].isupper() and nm in _KNOWN_OPTIONS:
                raise NotImplementedError(f"{fname}: the option {nm} is not supported")
            break
        o[nm] = args.pop().args[1]
    return args, o


_KNOWN_OPTIONS = {"Method", "WorkingPrecision", "Assumptions", "GenerateConditions", "PrincipalValue", "Direction",
                  "SeriesTermGoal", "AccuracyGoal", "PrecisionGoal", "PerformanceGoal", "MaxRecursion",
                  "GeneratedParameters", "Regularization", "InterpolationOrder", "InterpolationPoints",
                  "TargetStructure", "DependentVariables", "DiscreteVariables", "IncludeConstantBasis",
                  "FunctionSpace", "Modulus", "TimeConstraint", "MaxIterations"}


def _num_opt(v, fname, what):
    if K.is_int(v) and int(v) >= 0:
        return int(v)
    raise ValueError(f"{fname}: {what} must be a non-negative integer")


def _is_inf(v):
    return v in (sp.oo, -sp.oo)


# ───────────────────────────── inert symbols and options ─────────────────────────────

_OPTION_NOTES = {
    "PrincipalValue": "option of Integrate (and ContourIntegrate): PrincipalValue -> True gives Cauchy principal "
                      "values (default False); handled by Integrate (batch 4); an inert symbol otherwise",
    "SeriesTermGoal": "option of Asymptotic and the Asymptotic* functions: the number of terms of the asymptotic "
                      "approximation to return (default Automatic: the leading term for x -> x0)",
    "InterpolationOrder": "option of Interpolation, ListInterpolation and FunctionInterpolation: the polynomial "
                          "order of the interpolation (default 3)",
    "InterpolationPoints": "option of FunctionInterpolation: the number of sample points (default 31, refined until "
                           "the interpolant agrees with the function at the midpoints)",
    "TargetStructure": "option of symbolic-array functions (Normal of SymbolicIdentityArray etc.): \"Dense\" "
                       "(default) gives a dense list; other structures raise",
    "DependentVariables": "option of NDSolve/DSolve listing the dependent variables (inert symbol)",
    "DiscreteVariables": "option of NDSolve listing variables that change only at discrete events (inert symbol)",
    "IncludeConstantBasis": "option of DesignMatrix, LinearModelFit and related fitting functions: whether to "
                            "include a constant basis function (default True; inert symbol)",
    "C": "C[i] is the i-th constant of integration generated by DSolve, RSolve, AsymptoticDSolveValue and "
         "GreenFunction-related solvers (inert; C[i] + C[i] = 2 C[i])",
}


def _register_option_symbols():
    for _n, _d in _OPTION_NOTES.items():
        record(_n, AREA, "added", _d)


_register_option_symbols()


def _slots_ok(x, fname, exact=None):
    if not is_list(x) or not x.args or not all(K.is_int(v) and int(v) >= 1 for v in x.args):
        raise ValueError(f"{fname}: the slots must be a non-empty list of positive integers")
    s = [int(v) for v in x.args]
    if len(set(s)) != len(s):
        raise ValueError(f"{fname}: the slots must be distinct")
    if exact is not None and len(s) != exact:
        raise ValueError(f"{fname}: exactly {exact} slots are required")
    return s


for _n, _ex, _d in (("Symmetric", None, "Symmetric[{s1, ..., sn}]: symmetry under any permutation of the slots"),
                    ("Antisymmetric", None, "Antisymmetric[{s1, ..., sn}]: sign change under odd permutations"),
                    ("ZeroSymmetric", None, "ZeroSymmetric[{s1, ..., sn}]: the symmetry of a zero array"),
                    ("Hermitian", 2, "Hermitian[{s1, s2}]: invariance under conjugate transposition of the two "
                                     "slots"),
                    ("Antihermitian", 2, "Antihermitian[{s1, s2}]: sign change under conjugate transposition")):
    def _mk(name, exact):
        def f(slots):
            _slots_ok(slots, name, exact)
            raise Unevaluated()
        return f
    fn(_n, area=AREA, status="added", notes=_d + "; inert symmetry specification used by TensorSymmetry, "
       "ArraySymbol, MatrixSymbol, Matrices and Arrays; malformed slot lists raise")(_mk(_n, _ex))

_DOMAINS = {"Reals", "Complexes", "Integers", "Rationals", "Algebraics", "Booleans", "PositiveReals",
            "NonNegativeReals", "NegativeReals", "NonPositiveReals", "PositiveIntegers", "NonNegativeIntegers"}


def _dim_ok(d, fname):
    if K.is_int(d):
        if int(d) < 0:
            raise ValueError(f"{fname}: dimensions must be non-negative integers or symbols")
        return
    if isinstance(d, Symbol) or (isinstance(d, Expr) and not is_list(d)):
        return
    raise ValueError(f"{fname}: dimensions must be non-negative integers or symbols")


def _dims_ok(dims, fname):
    if not is_list(dims):
        raise ValueError(f"{fname}: the dimensions must be a list")
    for d in dims.args:
        _dim_ok(d, fname)
    return list(dims.args)


def _dom_ok(dom, fname):
    if not (isinstance(dom, Symbol) and dom.name in _DOMAINS):
        raise ValueError(f"{fname}: {dom!r} is not a supported domain")


_SYMHEADS = ("Symmetric", "Antisymmetric", "ZeroSymmetric", "Hermitian", "Antihermitian")


def _symspec_ok(s, rank, fname):
    if is_list(s) and not s.args:
        return
    items = s.args if is_list(s) and all(_head(v) for v in s.args) else (s,)
    for v in items:
        if _head(v) and v.head in _SYMHEADS and len(v.args) == 1:
            sl = _slots_ok(v.args[0], fname)
            if rank is not None and max(sl) > rank:
                raise ValueError(f"{fname}: symmetry slot {max(sl)} exceeds the rank {rank}")
            continue
        if is_list(v) and len(v.args) == 2 and _head(v.args[0], "Cycles"):
            continue
        raise ValueError(f"{fname}: {v!r} is not a symmetry specification")


@fn("Vectors", area=AREA, status="added",
    notes="Vectors[n] / Vectors[n, dom]: the domain of length-n vectors (dom Complexes by default, Reals, ...); inert "
          "domain used with Element and VectorSymbol; malformed specifications raise")
def _vectors(n, dom=None):
    _dim_ok(n, "Vectors")
    if dom is not None:
        _dom_ok(dom, "Vectors")
    raise Unevaluated()


@fn("Matrices", area=AREA, status="added",
    notes="Matrices[{m, n}] / Matrices[{m, n}, dom] / Matrices[{m, n}, dom, sym]: the domain of m x n matrices "
          "(sym a Symmetric/Antisymmetric/Hermitian/... specification); inert; malformed specifications raise")
def _matrices(dims, dom=None, symm=None):
    d = _dims_ok(dims, "Matrices")
    if len(d) != 2:
        raise ValueError("Matrices: the dimensions must be a list of two elements")
    if dom is not None:
        _dom_ok(dom, "Matrices")
    if symm is not None:
        _symspec_ok(symm, 2, "Matrices")
    raise Unevaluated()


@fn("Arrays", area=AREA, status="added",
    notes="Arrays[{d1, ..., dk}] / Arrays[dims, dom] / Arrays[dims, dom, sym]: the domain of arrays of the given "
          "dimensions; inert; malformed specifications raise")
def _arrays(dims, dom=None, symm=None):
    d = _dims_ok(dims, "Arrays")
    if dom is not None:
        _dom_ok(dom, "Arrays")
    if symm is not None:
        _symspec_ok(symm, len(d), "Arrays")
    raise Unevaluated()


@fn("VectorSymbol", area=AREA, status="added",
    notes="VectorSymbol[\"v\", n] / VectorSymbol[\"v\", n, dom]: a symbolic vector of length n (inert; used by "
          "ArrayExpand, ArraySimplify, TensorDimensions-style functions in this module); bad arguments raise")
def _vectorsymbol(name, n, dom=None):
    if not isinstance(name, (str, Symbol)):
        raise ValueError("VectorSymbol: the name must be a string or a symbol")
    _dim_ok(n, "VectorSymbol")
    if dom is not None:
        _dom_ok(dom, "VectorSymbol")
    raise Unevaluated()


@fn("MatrixSymbol", area=AREA, status="added",
    notes="MatrixSymbol[\"A\", {m, n}] / [.., dom] / [.., dom, sym]: a symbolic m x n matrix (inert); ArrayExpand / "
          "ArraySimplify treat it as a SymPy MatrixSymbol (symmetry Symmetric[{1, 2}] makes Transpose[A] = A)")
def _matrixsymbol(name, dims, dom=None, symm=None):
    if not isinstance(name, (str, Symbol)):
        raise ValueError("MatrixSymbol: the name must be a string or a symbol")
    d = _dims_ok(dims, "MatrixSymbol")
    if len(d) != 2:
        raise ValueError("MatrixSymbol: the dimensions must be a list of two elements")
    if dom is not None:
        _dom_ok(dom, "MatrixSymbol")
    if symm is not None:
        _symspec_ok(symm, 2, "MatrixSymbol")
    raise Unevaluated()


@fn("ArraySymbol", area=AREA, status="added",
    notes="ArraySymbol[\"A\", dims] / [.., dom] / [.., dom, sym]: a symbolic array (inert); rank-1 and rank-2 "
          "ArraySymbols take part in ArrayExpand / ArraySimplify like VectorSymbol / MatrixSymbol")
def _arraysymbol(name, dims, dom=None, symm=None):
    if not isinstance(name, (str, Symbol)):
        raise ValueError("ArraySymbol: the name must be a string or a symbol")
    d = _dims_ok(dims, "ArraySymbol")
    if dom is not None:
        _dom_ok(dom, "ArraySymbol")
    if symm is not None:
        _symspec_ok(symm, len(d), "ArraySymbol")
    raise Unevaluated()


def _int_dims(dims, fname):
    d = _dims_ok(dims, fname)
    if not all(K.is_int(v) for v in d):
        return None
    return [int(v) for v in d]


def _dense(f, dims):
    def build(i, idx):
        if i == len(dims):
            return f(idx)
        return L(*(build(i + 1, idx + [k]) for k in range(dims[i])))
    return build(0, [])


def _delta_pairs(pairs, rank, fname):
    if not is_list(pairs):
        raise ValueError(f"{fname}: the slot pairs must be a list of pairs")
    ps = []
    for p in pairs.args:
        if not (is_list(p) and len(p.args) == 2 and all(K.is_int(v) for v in p.args)):
            raise ValueError(f"{fname}: each slot pair must be a pair of integers")
        a, b = int(p.args[0]), int(p.args[1])
        if not (1 <= a <= rank and 1 <= b <= rank) or a == b:
            raise ValueError(f"{fname}: slot pair {{{a}, {b}}} is out of range")
        ps.append((a - 1, b - 1))
    used = [s for p in ps for s in p]
    if len(set(used)) != len(used):
        raise ValueError(f"{fname}: the slot pairs must be disjoint")
    return ps


def _sym_array_dense(e):
    """Dense list for a symbolic array with integer dimensions, or None."""
    h = e.head
    if h == "SymbolicZerosArray" and len(e.args) == 1:
        d = _int_dims(e.args[0], h)
        return None if d is None else _dense(lambda i: 0, d)
    if h == "SymbolicOnesArray" and len(e.args) == 1:
        d = _int_dims(e.args[0], h)
        return None if d is None else _dense(lambda i: 1, d)
    if h == "SymbolicIdentityArray" and len(e.args) == 1:
        d = _int_dims(e.args[0], h)
        if d is None:
            return None
        k = len(d)
        return _dense(lambda i: int(all(i[j] == i[j + k] for j in range(k))), d + d)
    if h == "SymbolicDeltaProductArray" and len(e.args) == 2:
        d = _int_dims(e.args[0], h)
        if d is None:
            return None
        ps = _delta_pairs(e.args[1], len(d), h)
        return _dense(lambda i: int(all(i[a] == i[b] for a, b in ps)), d)
    return None


@fn("SymbolicZerosArray", area=AREA, status="added",
    notes="SymbolicZerosArray[{d1, ...}]: a symbolic all-zero array (inert); Normal gives the dense array for "
          "integer dimensions; ArrayExpand/ArraySimplify use A.0 = 0, A + 0 = A")
def _szeros(dims):
    _dims_ok(dims, "SymbolicZerosArray")
    raise Unevaluated()


@fn("SymbolicOnesArray", area=AREA, status="added",
    notes="SymbolicOnesArray[{d1, ...}]: a symbolic all-ones array (inert); Normal gives the dense array for integer "
          "dimensions")
def _sones(dims):
    _dims_ok(dims, "SymbolicOnesArray")
    raise Unevaluated()


@fn("SymbolicIdentityArray", area=AREA, status="added",
    notes="SymbolicIdentityArray[{n1, ..., nk}]: the identity array of dimensions {n1, ..., nk, n1, ..., nk} "
          "(entries KroneckerDelta[i1, j1] ... KroneckerDelta[ik, jk]); SymbolicIdentityArray[{n}] is the n x n "
          "identity matrix; Normal gives the dense array; A.SymbolicIdentityArray[{n}] simplifies to A")
def _sident(dims):
    _dims_ok(dims, "SymbolicIdentityArray")
    raise Unevaluated()


@fn("SymbolicDeltaProductArray", area=AREA, status="added",
    notes="SymbolicDeltaProductArray[dims, {{s1, t1}, ...}]: the array whose entries are the product of "
          "KroneckerDelta[i_s, i_t] over the given slot pairs (pairs disjoint, slots in range); Normal gives the "
          "dense array")
def _sdelta(dims, pairs):
    d = _dims_ok(dims, "SymbolicDeltaProductArray")
    ps = _delta_pairs(pairs, len(d), "SymbolicDeltaProductArray")
    for a, b in ps:
        if K.is_int(d[a]) and K.is_int(d[b]) and int(d[a]) != int(d[b]):
            raise ValueError("SymbolicDeltaProductArray: paired slots must have equal dimensions")
    raise Unevaluated()


_SYMARRAY_HEADS = {"SymbolicZerosArray", "SymbolicOnesArray", "SymbolicIdentityArray", "SymbolicDeltaProductArray",
                   "MatrixSymbol", "VectorSymbol", "ArraySymbol"}


def _is_symarray(x):
    return _head(x) and x.head in _SYMARRAY_HEADS


def _contains_symarray(x):
    return K.contains(x, _is_symarray)


# Normal[SymbolicIdentityArray[{3}]] etc. (chained: every other Normal form is handled as before)
_prev_normal = K._BUILTINS.get("Normal")


def _normal_w06(a, ev_, expr):
    if a and _head(a[0]) and a[0].head in ("SymbolicZerosArray", "SymbolicOnesArray", "SymbolicIdentityArray",
                                           "SymbolicDeltaProductArray"):
        rest = a[1:]
        if rest and is_rule(rest[-1]) and _is(rest[-1].args[0], "TargetStructure"):
            ts = rest[-1].args[1]
            if ts != "Dense" and not _is(ts, "Automatic"):
                raise NotImplementedError("Normal: only TargetStructure -> \"Dense\" is supported for symbolic "
                                          "arrays")
            rest = rest[:-1]
        if not rest:
            d = _sym_array_dense(a[0])
            if d is not None:
                return d
            return a[0]
    if a and len(a) == 2 and is_rule(a[1]) and _is(a[1].args[0], "TargetStructure"):
        return _prev_normal([a[0]], ev_, Expr("Normal", a[0]))
    return _prev_normal(a, ev_, expr)


K._BUILTINS["Normal"] = _normal_w06

# Transpose of a symbolic array stays unevaluated (Wolfram) instead of raising "expects a list"
_prev_transpose = K._BUILTINS.get("Transpose")


def _transpose_w06(a, ev_, expr):
    if a and not is_list(a[0]) and _contains_symarray(a[0]):
        return expr
    return _prev_transpose(a, ev_, expr)


K._BUILTINS["Transpose"] = _transpose_w06
