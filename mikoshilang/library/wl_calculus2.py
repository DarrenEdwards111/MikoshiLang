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


# ───────────────────────────── symbolic matrix algebra (ArrayExpand / ArraySimplify) ─────────────────────────────

class _MX:
    """MikoshiLang symbolic-array expression <-> SymPy matrix expression."""

    def __init__(self, fname):
        self.fname = fname
        self.back = {}          # sympy MatrixSymbol name -> original MikoshiLang expression
        self.vec = set()        # names of vector symbols (stored as n x 1 columns)
        self.symm = {}          # name -> +1 symmetric / -1 antisymmetric

    def _dim(self, d):
        return sp.Integer(int(d)) if K.is_int(d) else SX(d, self.fname)

    def leaf(self, e):
        h = e.head
        if h in ("MatrixSymbol", "ArraySymbol", "VectorSymbol"):
            nm = e.args[0] if isinstance(e.args[0], str) else e.args[0].name
            if h == "VectorSymbol":
                dims = [e.args[1]]
            else:
                dims = list(e.args[1].args)
            key = f"{h}:{nm}:{len(dims)}"
            if len(dims) == 1:
                m = sp.MatrixSymbol(key, self._dim(dims[0]), 1)
                self.vec.add(key)
            elif len(dims) == 2:
                m = sp.MatrixSymbol(key, self._dim(dims[0]), self._dim(dims[1]))
            else:
                raise NotImplementedError(f"{self.fname}: arrays of rank {len(dims)} are not supported")
            self.back[key] = e
            if len(e.args) == 4 and len(dims) == 2:
                s = e.args[3]
                if _head(s, "Symmetric") or _head(s, "Hermitian"):
                    if _head(s, "Symmetric") or (len(e.args) > 2 and _is(e.args[2], "Reals")):
                        self.symm[key] = 1
                elif _head(s, "Antisymmetric"):
                    self.symm[key] = -1
            return m
        if h in ("SymbolicZerosArray", "SymbolicOnesArray", "SymbolicIdentityArray"):
            dims = [self._dim(d) for d in e.args[0].args]
            if h == "SymbolicIdentityArray":
                if len(dims) != 1:
                    raise NotImplementedError(f"{self.fname}: only SymbolicIdentityArray[{{n}}] is supported")
                return sp.Identity(dims[0])
            if len(dims) == 1:
                dims = dims + [sp.Integer(1)]
            if len(dims) != 2:
                raise NotImplementedError(f"{self.fname}: arrays of rank {len(dims)} are not supported")
            return (sp.ZeroMatrix if h == "SymbolicZerosArray" else sp.OneMatrix)(*dims)
        raise NotImplementedError(f"{self.fname}: {h} is not supported")

    def to(self, e, pos=None):
        if _is_symarray(e):
            return self.leaf(e)
        if not _contains_symarray(e):
            if is_list(e):
                raise NotImplementedError(f"{self.fname}: explicit lists cannot be mixed with symbolic arrays")
            return SX(e, self.fname)
        if not _head(e):
            raise NotImplementedError(f"{self.fname}: unsupported expression {e!r}")
        h = e.head
        a = e.args
        if h == "Plus":
            return sp.MatAdd(*[self.to(x) for x in a]) if True else None
        if h == "Times":
            return sp.MatMul(*[self.to(x) for x in a])
        if h == "Dot":
            fs = [self.to(x) for x in a]
            for i, (x, f) in enumerate(zip(a, fs)):
                if self._isvec(f):
                    if i == 0 and len(fs) > 1:
                        fs[0] = f.T
                    elif i != len(fs) - 1:
                        raise NotImplementedError(f"{self.fname}: a vector can only be the first or last factor "
                                                  "of Dot")
            return sp.MatMul(*fs)
        if h == "Transpose" and len(a) == 1:
            return sp.Transpose(self.to(a[0]))
        if h == "Inverse" and len(a) == 1:
            return sp.Inverse(self.to(a[0]))
        if h == "MatrixPower" and len(a) == 2 and K.is_int(a[1]):
            return sp.MatPow(self.to(a[0]), int(a[1]))
        if h == "Tr" and len(a) == 1:
            return sp.Trace(self.to(a[0]))
        if h == "Det" and len(a) == 1:
            return sp.Determinant(self.to(a[0]))
        raise NotImplementedError(f"{self.fname}: {h} of symbolic arrays is not supported")

    def _isvec(self, f):
        return isinstance(f, sp.MatrixSymbol) and f.name in self.vec or \
            (isinstance(f, sp.MatrixExpr) and f.shape[1] == 1 and any(
                isinstance(s, sp.MatrixSymbol) and s.name in self.vec for s in f.atoms(sp.MatrixSymbol)))

    def apply_symmetry(self, v):
        if not self.symm:
            return v
        reps = {}
        for k, s in self.symm.items():
            m = [x for x in v.atoms(sp.MatrixSymbol) if x.name == k]
            for x in m:
                reps[sp.Transpose(x)] = x if s == 1 else -x
        for _ in range(3):
            v = v.xreplace(reps).doit()
        return v

    def back_(self, v):
        if isinstance(v, sp.MatrixSymbol):
            return self.back[v.name]
        if isinstance(v, sp.Identity):
            return Expr("SymbolicIdentityArray", L(R(v.shape[0])))
        if isinstance(v, (sp.ZeroMatrix, sp.OneMatrix)):
            hd = "SymbolicZerosArray" if isinstance(v, sp.ZeroMatrix) else "SymbolicOnesArray"
            if v.shape[1] == 1:
                return Expr(hd, L(R(v.shape[0])))
            return Expr(hd, L(R(v.shape[0]), R(v.shape[1])))
        if isinstance(v, sp.MatAdd):
            return _ev(Expr("Plus", *(self.back_(x) for x in v.args)))
        if isinstance(v, sp.MatMul):
            sc, mats = v.as_coeff_matrices()
            fs = []
            for m in mats:
                if isinstance(m, sp.Transpose) and isinstance(m.arg, sp.MatrixSymbol) and m.arg.name in self.vec:
                    fs.append(self.back[m.arg.name])
                elif isinstance(m, sp.MatPow) and m.exp.is_Integer and m.exp > 0:
                    fs.extend([self.back_(m.base)] * int(m.exp))
                else:
                    fs.append(self.back_(m))
            d = fs[0] if len(fs) == 1 else Expr("Dot", *fs)
            return _ev(Expr("Times", R(sc), d)) if sc != 1 else d
        if isinstance(v, sp.Transpose):
            if isinstance(v.arg, sp.MatrixSymbol) and v.arg.name in self.vec:
                return self.back[v.arg.name]
            return Expr("Transpose", self.back_(v.arg))
        if isinstance(v, sp.Inverse):
            return Expr("Inverse", self.back_(v.arg))
        if isinstance(v, sp.MatPow):
            if v.exp == -1:
                return Expr("Inverse", self.back_(v.base))
            if v.exp.is_Integer and v.exp > 0:
                return Expr("Dot", *([self.back_(v.base)] * int(v.exp)))
            return Expr("MatrixPower", self.back_(v.base), R(v.exp))
        if isinstance(v, sp.Trace):
            return Expr("Tr", self.back_(v.arg))
        if isinstance(v, sp.Determinant):
            return Expr("Det", self.back_(v.arg))
        if isinstance(v, sp.MatrixExpr):
            raise NotImplementedError(f"{self.fname}: cannot convert {v} back")
        if isinstance(v, sp.Basic) and v.atoms(sp.MatrixExpr):
            # scalar expressions built from traces/determinants
            reps = {}
            for t in v.atoms(sp.Trace, sp.Determinant):
                d = sp.Dummy()
                reps[t] = d
            w = v.xreplace(reps)
            res = out(w)
            return K.substitute(res, {str(d): self.back_(t) for t, d in reps.items()}) if reps else res
        return out(v)


def _array_simp(e, fname, simplify):
    if is_list(e):
        return _ev(Expr("Simplify", e)) if simplify else _ev(Expr("Expand", e))
    if not _contains_symarray(e):
        return _ev(Expr("Simplify" if simplify else "Expand", e))
    c = _MX(fname)
    v = c.to(e)
    v = v.doit()
    v = c.apply_symmetry(v)
    if simplify:
        v = sp.simplify(v)
        v = c.apply_symmetry(v.doit())
    else:
        v = sp.expand(v).doit()
        v = c.apply_symmetry(v)
    return c.back_(v)


@fn("ArrayExpand", area=AREA, status="added",
    notes="ArrayExpand[expr]: expands symbolic-array expressions built from MatrixSymbol/VectorSymbol/ArraySymbol "
          "(rank <= 2) and Symbolic*Array with Plus, Times, Dot, Transpose, Inverse, MatrixPower, Tr, Det: "
          "distributes Dot over Plus, Transpose[a.b] -> Transpose[b].Transpose[a], Transpose[a + b] -> ..., "
          "Inverse[a.b] -> Inverse[b].Inverse[a] (SymPy matrix expressions); explicit arrays are expanded "
          "elementwise; anything else raises")
def _arrayexpand(e, *rest):
    pos, _o = _opts(rest, {"Assumptions"}, "ArrayExpand")
    if pos:
        raise ValueError("ArrayExpand takes one expression")
    return _array_simp(e, "ArrayExpand", False)


@fn("ArraySimplify", area=AREA, status="added",
    notes="ArraySimplify[expr]: simplifies symbolic-array expressions (Transpose[Transpose[a]] -> a, "
          "a.Inverse[a] -> identity, a.SymbolicIdentityArray[{n}] -> a, a.SymbolicZerosArray -> zeros, "
          "Transpose[a] -> a for MatrixSymbol[.., Symmetric[{1, 2}]] and -a for Antisymmetric) via SymPy matrix "
          "expressions; explicit arrays are simplified elementwise")
def _arraysimplify(e, *rest):
    pos, _o = _opts(rest, {"Assumptions"}, "ArraySimplify")
    if pos:
        raise ValueError("ArraySimplify takes one expression (and options)")
    return _array_simp(e, "ArraySimplify", True)


# ───────────────────────────── explicit tensors ─────────────────────────────

def _array(x, fname):
    """Explicit rectangular array -> (nested python lists of MikoshiLang values, dims); scalars -> (x, [])."""
    if not is_list(x):
        return x, []
    dims = KL.array_dims(x)
    def conv(e, d):
        if d == 0:
            return e
        return [conv(a, d - 1) for a in e.args]
    if not dims:
        raise ValueError(f"{fname}: the argument must be a rectangular array")
    # array_dims gives the full rectangular depth
    return conv(x, len(dims)), list(dims)


def _np_get(a, idx):
    for i in idx:
        a = a[i]
    return a


def _sp_array(x, fname):
    a, dims = _array(x, fname)
    if not dims:
        return sp.Array(SX(x, fname)), []
    flat = [SX(v, fname) for v in _flatten(a, len(dims))]
    return sp.MutableDenseNDimArray(flat, tuple(dims)), dims


def _flatten(a, depth):
    if depth == 0:
        return [a]
    r = []
    for v in a:
        r.extend(_flatten(v, depth - 1))
    return r


def _arr_out(A):
    if A.rank() == 0:
        return out(A[()])
    return out(A.tolist())


def _permute(A, perm):
    """B[i_1..i_r] = A[i_perm(1)..i_perm(r)]  (perm a tuple of 0-based slot indices)."""
    return sp.permutedims(A, perm)


@fn("TensorSymmetry", area=AREA, status="added",
    notes="TensorSymmetry[tensor]: the symmetry of an explicit array: Symmetric[{..}] / Antisymmetric[{..}] when the "
          "symmetry group is the full symmetric group on a set of slots (the others fixed), ZeroSymmetric[{1, ..., "
          "r}] for a zero array, {} for none, else a generating set {{Cycles[..], phase}, ...} (phases +-1 or roots of unity); also reads the "
          "symmetry of MatrixSymbol/ArraySymbol and SymbolicIdentityArray; TensorSymmetry[t, slots] restricts to "
          "the given slots")
def _tensorsymmetry(t, slots=None):
    if _head(t) and t.head in ("MatrixSymbol", "ArraySymbol"):
        if len(t.args) == 4:
            return t.args[3]
        return L()
    if _head(t, "SymbolicIdentityArray"):
        k = len(t.args[0].args)
        if k == 1:
            return Expr("Symmetric", L(1, 2))
        return L(*(L(Expr("Cycles", L(L(i + 1, i + k + 1))), 1) for i in range(k)))
    if _head(t, "SymbolicZerosArray"):
        return Expr("ZeroSymmetric", L(*range(1, len(t.args[0].args) + 1)))
    if not is_list(t):
        if isinstance(t, (Symbol, Expr)):
            raise Unevaluated()
        return L()
    A, dims = _sp_array(t, "TensorSymmetry")
    r = len(dims)
    sl = list(range(r)) if slots is None else [s - 1 for s in _slots_ok(slots, "TensorSymmetry")]
    if any(s >= r for s in sl):
        raise ValueError("TensorSymmetry: slot out of range")
    idxs = list(itertools.product(*[range(m) for m in dims]))
    if all(sp.simplify(A[i]) == 0 for i in idxs):
        return Expr("ZeroSymmetric", L(*(s + 1 for s in sl)))
    from sympy.combinatorics import Permutation, PermutationGroup
    elems = []
    for p in itertools.permutations(sl):
        perm = list(range(r))
        for s, q in zip(sl, p):
            perm[s] = q
        if perm == list(range(r)) or any(dims[i] != dims[perm[i]] for i in range(r)):
            continue
        B = _permute(A, perm)
        i0 = next(i for i in idxs if sp.simplify(A[i]) != 0)
        c = sp.simplify(B[i0] / A[i0])
        if c.free_symbols or c == 0:
            continue
        if all(sp.simplify(B[i] - c * A[i]) == 0 for i in idxs):
            elems.append((perm, c))
    if not elems:
        return L()
    G = PermutationGroup([Permutation(p) for p, _ in elems])
    moved = sorted({i for p, _ in elems for i in range(r) if p[i] != i})
    full = math.factorial(len(moved))
    if G.order() == full:
        sym_all = all(s == 1 for _, s in elems)
        anti = all(s == Permutation(p).signature() for p, s in elems)
        if sym_all:
            return Expr("Symmetric", L(*(m + 1 for m in moved)))
        if anti:
            return Expr("Antisymmetric", L(*(m + 1 for m in moved)))
    gens = []
    H = None
    for p, s in sorted(elems, key=lambda ps: (sum(1 for i in range(r) if ps[0][i] != i), ps[0])):
        P = Permutation(p)
        if H is not None and H.contains(P):
            continue
        gens.append((P, s))
        H = PermutationGroup([g for g, _ in gens])
        if H.order() == G.order():
            break
    res = []
    for P, s in gens:
        cyc = [c for c in P.cyclic_form]
        res.append(L(Expr("Cycles", L(*(L(*(i + 1 for i in c)) for c in cyc))), R(s)))
    return L(*res)


def _antisym_wedge(A, rank):
    """Alt(A) = (1/rank!) sum_sigma sgn(sigma) A^sigma."""
    if rank <= 1:
        return A
    from sympy.combinatorics import Permutation
    tot = None
    for p in itertools.permutations(range(rank)):
        B = _permute(A, list(p))
        s = Permutation(list(p)).signature()
        tot = B * s if tot is None else tot + B * s
    return tot / math.factorial(rank)


@fn("TensorWedge", area=AREA, status="added",
    notes="TensorWedge[t1, t2, ...]: the exterior product of explicit antisymmetric arrays of equal dimension, "
          "normalized as (p + q)!/(p! q!) Alt(t1 (x) t2) (TensorWedge[{a1, a2}, {b1, b2}] = "
          "{{0, a1 b2 - a2 b1}, {a2 b1 - a1 b2, 0}}); scalars multiply; symbolic arguments stay unevaluated")
def _tensorwedge(*ts):
    if not ts:
        return 1
    if any(not is_list(t) and isinstance(t, (Symbol, Expr)) and not K.is_number(t) and
           (_contains_symarray(t) or isinstance(t, Symbol) and t.name not in ("Pi", "E")) for t in ts):
        raise Unevaluated()
    if len(ts) == 1:
        return ts[0]
    acc, accr = None, 0
    dim = None
    for t in ts:
        A, dims = _sp_array(t, "TensorWedge")
        if dims:
            if len(set(dims)) != 1 or (dim is not None and dims[0] != dim):
                raise ValueError("TensorWedge: all slots of all arguments must have the same dimension")
            dim = dims[0]
        k = len(dims)
        if acc is None:
            acc, accr = A, k
            continue
        if accr == 0 or k == 0:
            acc = acc[()] * A if accr == 0 else acc * A[()]
            accr = accr + k
            continue
        P = sp.tensorproduct(acc, A)
        acc = _antisym_wedge(P, accr + k) * sp.Rational(math.factorial(accr + k),
                                                        math.factorial(accr) * math.factorial(k))
        accr += k
    if accr == 0:
        return out(acc[()] if isinstance(acc, sp.NDimArray) else acc)
    return _arr_out(acc.applyfunc(sp.expand))


@fn("HodgeDual", area=AREA, status="promoted",
    notes="HodgeDual[t]: the Hodge dual of an explicit array whose slots all have dimension n: "
          "(1/k!) LeviCivita_{i1..ik j1..j(n-k)} t_{i1..ik} (t is antisymmetrized in effect); HodgeDual[{a, b, c}] = "
          "{{0, c, -b}, {-c, 0, a}, {b, -a, 0}}; HodgeDual[x, n] for a scalar is x LeviCivitaTensor[n]; "
          "HodgeDual[t, n, slots] dualizes the given slots (the others become the last slots of the result); "
          "previously quarantined")
def _hodgedual(t, n=None, slots=None):
    if not is_list(t):
        if _contains_symarray(t) or (isinstance(t, Symbol) and n is None):
            raise Unevaluated()
        if n is None:
            raise ValueError("HodgeDual: the dimension must be given for a scalar")
        d = pos_int(n, "HodgeDual", "dimension")
        eps = sp.LeviCivita
        E = sp.MutableDenseNDimArray([eps(*i) for i in itertools.product(range(d), repeat=d)], (d,) * d)
        return _arr_out(E * SX(t, "HodgeDual"))
    A, dims = _sp_array(t, "HodgeDual")
    r = len(dims)
    if n is None:
        if len(set(dims)) != 1:
            raise ValueError("HodgeDual: all slots must have the same dimension; give the dimension explicitly")
        d = dims[0]
    else:
        d = pos_int(n, "HodgeDual", "dimension")
    if slots is None:
        sl = [i for i in range(r) if dims[i] == d]
    else:
        sl = [s - 1 for s in _slots_ok(slots, "HodgeDual")]
        if any(s >= r or dims[s] != d for s in sl):
            raise ValueError(f"HodgeDual: the given slots must have dimension {d}")
    k = len(sl)
    if k > d:
        raise ValueError("HodgeDual: more dualized slots than the dimension")
    rest = [i for i in range(r) if i not in sl]
    # bring dualized slots to the front
    A = _permute(A, sl + rest) if sl + rest != list(range(r)) else A
    rdims = [dims[i] for i in rest]
    outdims = [d] * (d - k) + rdims
    res = {}
    for J in itertools.product(range(d), repeat=d - k):
        for Rr in itertools.product(*[range(m) for m in rdims]):
            s = 0
            for I in itertools.product(range(d), repeat=k):
                e = sp.LeviCivita(*(I + J))
                if e:
                    s += e * A[I + Rr]
            res[J + Rr] = sp.expand(sp.sympify(s) / math.factorial(k))
    if not outdims:
        return out(res[()])
    flat = [res[i] for i in itertools.product(*[range(m) for m in outdims])]
    return _arr_out(sp.MutableDenseNDimArray(flat, tuple(outdims)))


def _tensor_terms(e):
    return list(e.args) if _head(e, "Plus") else [e]


def _scalar_factor(e):
    """(numeric coefficient, rest) for Times[c, rest]."""
    if _head(e, "Times"):
        nums = [a for a in e.args if K.is_number(a)]
        others = [a for a in e.args if not K.is_number(a)]
        if nums:
            c = _ev(Expr("Times", *nums))
            rest = others[0] if len(others) == 1 else Expr("Times", *others)
            return c, rest
    return 1, e


def _texpand(e):
    if not _head(e):
        return e
    if e.head in ("TensorProduct", "TensorWedge", "Dot"):
        args = [_texpand(a) for a in e.args]
        expanded = [_tensor_terms(a) for a in args]
        terms = []
        for combo in itertools.product(*expanded):
            c = 1
            fs = []
            for f in combo:
                cf, rf = _scalar_factor(f)
                c = _ev(Expr("Times", c, cf))
                fs.append(rf)
            terms.append(_ev(Expr("Times", c, Expr(e.head, *fs))))
        return _ev(Expr("Plus", *terms))
    if e.head in ("Plus", "Times"):
        return _ev(Expr(e.head, *(_texpand(a) for a in e.args)))
    return e


@fn("TensorExpand", area=AREA, status="added",
    notes="TensorExpand[expr]: distributes TensorProduct, TensorWedge and Dot over Plus and pulls numeric factors "
          "out of them (multilinearity), recursively; symbolic factors are treated as tensors (numeric coefficients "
          "only are scalars), so the result is valid for any ranks")
def _tensorexpand(e, *rest):
    pos, _o = _opts(rest, {"Assumptions"}, "TensorExpand")
    if pos:
        raise ValueError("TensorExpand takes one expression")
    return _texpand(e)


def _rank_of(x):
    if _head(x, "VectorSymbol"):
        return 1
    if _head(x, "MatrixSymbol"):
        return 2
    if _head(x, "ArraySymbol") and is_list(x.args[1]):
        return len(x.args[1].args)
    return None


def _treduce(e):
    e = _texpand(e)
    terms = _tensor_terms(e)
    acc = []
    for t in terms:
        c, r = _scalar_factor(t)
        if _head(r, "TensorWedge"):
            fs = list(r.args)
            ranks = [_rank_of(f) for f in fs]
            if all(k is not None for k in ranks):
                # canonical order with Koszul signs; repeated odd-rank factor -> 0
                sign = 1
                arr = list(zip(fs, ranks))
                key = lambda fr: K.canonical_order_key(fr[0]) if hasattr(K, "canonical_order_key") else str(fr[0])
                for i in range(len(arr)):
                    for j in range(len(arr) - 1 - i):
                        if key(arr[j]) > key(arr[j + 1]):
                            if arr[j][1] % 2 and arr[j + 1][1] % 2:
                                sign = -sign
                            arr[j], arr[j + 1] = arr[j + 1], arr[j]
                zero = any(str(arr[i][0]) == str(arr[i + 1][0]) and arr[i][1] % 2 for i in range(len(arr) - 1))
                if zero:
                    continue
                r = Expr("TensorWedge", *(f for f, _ in arr))
                c = _ev(Expr("Times", c, sign))
        acc.append((c, r))
    coll = {}
    order = []
    for c, r in acc:
        k = K.input_form(r) if hasattr(K, "input_form") else str(r)
        if k not in coll:
            coll[k] = [0, r]
            order.append(k)
        coll[k][0] = _ev(Expr("Plus", coll[k][0], c))
    terms = [_ev(Expr("Times", coll[k][0], coll[k][1])) for k in order if coll[k][0] != 0]
    return _ev(Expr("Plus", *terms)) if terms else 0


@fn("TensorReduce", area=AREA, status="added",
    notes="TensorReduce[expr]: canonical form of tensor polynomials: TensorExpand, then TensorWedge factors of known "
          "rank (VectorSymbol, MatrixSymbol, ArraySymbol) are sorted with the Koszul sign (-1)^(p q), a repeated "
          "odd-rank factor gives 0, and like terms are collected; explicit arrays evaluate directly")
def _tensorreduce(e, *rest):
    pos, _o = _opts(rest, {"Assumptions"}, "TensorReduce")
    if pos:
        raise ValueError("TensorReduce takes one expression")
    return _treduce(e)


# ───────────────────────────── coordinates ─────────────────────────────

def _vec(x, fname, n=None):
    if not is_list(x) or any(is_list(a) for a in x.args):
        if isinstance(x, (Symbol, Expr)) and not is_list(x):
            raise Unevaluated()
        raise ValueError(f"{fname}: expected a list of coordinates")
    if n is not None and len(x.args) != n:
        raise ValueError(f"{fname}: expected a list of {n} coordinates")
    return [SX(a, fname) for a in x.args]


def _to_polar(xs):
    n = len(xs)
    if n < 2:
        raise ValueError("ToPolarCoordinates: at least two coordinates are required")
    r = sp.sqrt(sum(v ** 2 for v in xs))
    ang = []
    for i in range(n - 2):
        ang.append(sp.acos(xs[i] / sp.sqrt(sum(v ** 2 for v in xs[i:]))))
    ang.append(sp.atan2(xs[n - 1], xs[n - 2]))
    return [r] + ang


def _from_polar(ps):
    n = len(ps)
    if n < 2:
        raise ValueError("FromPolarCoordinates: at least two coordinates are required")
    r, th = ps[0], ps[1:]
    res = []
    prod = r
    for i in range(n - 1):
        if i < n - 2:
            res.append(prod * sp.cos(th[i]))
            prod = prod * sp.sin(th[i])
        else:
            res.append(prod * sp.cos(th[i]))
            res.append(prod * sp.sin(th[i]))
    return res


def _to_sph(xs):
    x, y, z = xs
    return [sp.sqrt(x ** 2 + y ** 2 + z ** 2), sp.atan2(sp.sqrt(x ** 2 + y ** 2), z), sp.atan2(y, x)]


def _from_sph(ps):
    r, th, ph = ps
    return [r * sp.sin(th) * sp.cos(ph), r * sp.sin(th) * sp.sin(ph), r * sp.cos(th)]


def _coord_out(vs, inputs):
    return L(*(R(sp.N(v, 15)) if any(isinstance(i, sp.Float) for i in inputs) else out(v) for v in vs))


def _points_map(x, f, n, fname):
    if is_list(x) and x.args and all(is_list(a) for a in x.args):
        return L(*(_points_map(a, f, n, fname) for a in x.args))
    xs = _vec(x, fname, n)
    return _coord_out(f(xs), xs)


@fn("ToPolarCoordinates", area=AREA, status="added",
    notes="ToPolarCoordinates[{x, y}] = {Sqrt[x^2 + y^2], ArcTan[x, y]}; {x1, ..., xn} -> hyperspherical {r, "
          "ArcCos[x1/Sqrt[x1^2 + ... + xn^2]], ..., ArcTan[x(n-1), xn]}; threads over a list of points")
def _topolar(x):
    return _points_map(x, _to_polar, None, "ToPolarCoordinates")


@fn("FromPolarCoordinates", area=AREA, status="added",
    notes="FromPolarCoordinates[{r, t}] = {r Cos[t], r Sin[t]}; {r, t1, ..., t(n-1)} -> {r Cos[t1], r Sin[t1] "
          "Cos[t2], ..., r Sin[t1] ... Sin[t(n-1)]}; threads over a list of points")
def _frompolar(x):
    return _points_map(x, _from_polar, None, "FromPolarCoordinates")


@fn("ToSphericalCoordinates", area=AREA, status="added",
    notes="ToSphericalCoordinates[{x, y, z}] = {Sqrt[x^2 + y^2 + z^2], ArcTan[z, Sqrt[x^2 + y^2]], ArcTan[x, y]} "
          "(radius, polar angle from the z axis, azimuth); threads over a list of points")
def _tosph(x):
    return _points_map(x, _to_sph, 3, "ToSphericalCoordinates")


@fn("FromSphericalCoordinates", area=AREA, status="added",
    notes="FromSphericalCoordinates[{r, t, p}] = {r Cos[p] Sin[t], r Sin[t] Sin[p], r Cos[t]}; threads over a list "
          "of points")
def _fromsph(x):
    return _points_map(x, _from_sph, 3, "FromSphericalCoordinates")


def _chart(c, fname):
    """(name, dim or None) from "Polar" or {"Polar", n} (also {"Cartesian", n, metric}... not supported)."""
    if isinstance(c, str):
        nm, d = c, None
    elif is_list(c) and len(c.args) == 2 and isinstance(c.args[0], str) and K.is_int(c.args[1]):
        nm, d = c.args[0], int(c.args[1])
    else:
        raise ValueError(f"{fname}: {c!r} is not a coordinate chart specification")
    if nm == "Hyperspherical":
        nm = "Polar"
    if nm not in ("Cartesian", "Polar", "Spherical", "Cylindrical"):
        raise NotImplementedError(f"{fname}: the chart {nm!r} is not supported (Cartesian, Polar, Spherical, "
                                  "Cylindrical)")
    if nm in ("Spherical", "Cylindrical"):
        if d not in (None, 3):
            raise ValueError(f"{fname}: {nm} coordinates are three-dimensional")
        d = 3
    return nm, d


def _to_cart(nm, ps):
    if nm == "Cartesian":
        return list(ps)
    if nm == "Polar":
        return _from_polar(ps)
    if nm == "Spherical":
        return _from_sph(ps)
    r, th, z = ps
    return [r * sp.cos(th), r * sp.sin(th), z]


def _from_cart(nm, xs):
    if nm == "Cartesian":
        return list(xs)
    if nm == "Polar":
        return _to_polar(xs)
    if nm == "Spherical":
        return _to_sph(xs)
    x, y, z = xs
    return [sp.sqrt(x ** 2 + y ** 2), sp.atan2(y, x), z]


def _transform_spec(t, fname):
    if not is_rule(t):
        raise ValueError(f"{fname}: the transformation must be \"chart1\" -> \"chart2\"")
    a, da = _chart(t.args[0], fname)
    b, db = _chart(t.args[1], fname)
    d = da or db
    if da and db and da != db:
        raise ValueError(f"{fname}: the charts have different dimensions")
    return a, b, d


@fn("CoordinateTransform", area=AREA, status="added",
    notes="CoordinateTransform[\"chart1\" -> \"chart2\", pt]: the coordinates of pt in chart2 (charts Cartesian, "
          "Polar (n-dimensional hyperspherical), Spherical, Cylindrical; {\"chart\", n} gives the dimension); "
          "threads over a list of points")
def _coordtransform(t, pt):
    a, b, d = _transform_spec(t, "CoordinateTransform")

    def f(ps):
        if d is not None and len(ps) != d:
            raise ValueError(f"CoordinateTransform: expected {d} coordinates")
        if a in ("Spherical", "Cylindrical") or b in ("Spherical", "Cylindrical"):
            if len(ps) != 3:
                raise ValueError("CoordinateTransform: expected 3 coordinates")
        pos = _positive_assume(a, [v for v in ps if isinstance(v, sp.Symbol)]) if all(
            isinstance(v, sp.Symbol) for v in ps) else {}
        back = {v: k for k, v in pos.items()}
        ps = [pos.get(v, v) for v in ps]
        return [(sp.simplify(v) if not v.is_number else v).xreplace(back) for v in _from_cart(b, _to_cart(a, ps))]
    return _points_map(pt, f, None, "CoordinateTransform")


def _frame(nm, ps):
    """Columns: the orthonormal basis vectors of the chart at ps, in Cartesian components."""
    if nm == "Cartesian":
        return sp.eye(len(ps))
    if nm == "Polar" and len(ps) == 2:
        th = ps[1]
        return sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
    if nm == "Cylindrical":
        th = ps[1]
        return sp.Matrix([[sp.cos(th), -sp.sin(th), 0], [sp.sin(th), sp.cos(th), 0], [0, 0, 1]])
    if nm == "Spherical" or (nm == "Polar" and len(ps) == 3):
        th, ph = ps[1], ps[2]
        if nm == "Polar":    # hyperspherical {r, t1, t2}: x = r Cos[t1], y = r Sin[t1] Cos[t2], z = r Sin[t1] Sin[t2]
            X = sp.Matrix(_from_polar(ps))
        else:
            X = sp.Matrix(_from_sph(ps))
        cols = []
        for v in ps:
            c = X.diff(v)
            nrm = sp.sqrt(sum(sp.simplify(e ** 2) for e in c))
            cols.append(c / sp.simplify(nrm))
        return sp.Matrix.hstack(*cols)
    raise NotImplementedError("TransformedField: vector fields in this chart are not supported")


def _positive_assume(nm, syms):
    """Symbols with the standard chart assumptions (radius > 0, polar angle in (0, Pi))."""
    reps = {}
    if nm in ("Polar", "Spherical", "Cylindrical") and syms:
        reps[syms[0]] = sp.Symbol(syms[0].name, positive=True)
        if nm == "Spherical" or (nm == "Polar" and len(syms) > 2):
            for v in syms[1:-1]:
                reps[v] = sp.Symbol(v.name, positive=True)    # polar angles in (0, Pi): Sin > 0
    return reps


@fn("TransformedField", area=AREA, status="added",
    notes="TransformedField[\"chart1\" -> \"chart2\", f, {x1, ...} -> {y1, ...}]: a scalar field f (expression) or a "
          "vector field (list of components in the orthonormal frame of chart1) re-expressed in chart2 "
          "coordinates {y1, ...} (charts Cartesian, Polar, Spherical, Cylindrical); simplification assumes the "
          "standard chart ranges (r > 0, polar angles in (0, Pi)); rank >= 2 tensor fields raise")
def _transformedfield(t, f, vars_):
    a, b, d = _transform_spec(t, "TransformedField")
    if not (is_rule(vars_) and is_list(vars_.args[0]) and is_list(vars_.args[1])):
        raise ValueError("TransformedField: the variables must be given as {x1, ...} -> {y1, ...}")
    xs = [_svar(v, "TransformedField") for v in vars_.args[0].args]
    ys = [_svar(v, "TransformedField") for v in vars_.args[1].args]
    if len(xs) != len(ys) or (d is not None and len(xs) != d):
        raise ValueError("TransformedField: the variable lists must have the chart dimension")
    if (a in ("Spherical", "Cylindrical") or b in ("Spherical", "Cylindrical")) and len(xs) != 3:
        raise ValueError("TransformedField: Spherical and Cylindrical charts are three-dimensional")
    pos = _positive_assume(b, ys)
    yp = [pos.get(v, v) for v in ys]
    old_in_new = _from_cart(a, _to_cart(b, yp))
    sub = dict(zip(xs, old_in_new))
    back = {v: k for k, v in pos.items()}

    def simp(e):
        return sp.simplify(e).xreplace(back)
    if is_list(f):
        if any(is_list(c) for c in f.args):
            raise NotImplementedError("TransformedField: only scalar and vector fields are supported")
        comps = sp.Matrix([SX(c, "TransformedField") for c in f.args])
        if len(comps) != len(xs):
            raise ValueError("TransformedField: the vector field must have one component per coordinate")
        Fa = _frame(a, old_in_new)
        Fb = _frame(b, yp)
        cart = Fa * comps.xreplace(sub)
        res = Fb.T * cart
        return L(*(out(simp(sp.trigsimp(sp.expand(r)))) for r in res))
    v = SX(f, "TransformedField").xreplace(sub)
    return out(simp(v))


def _speed_norm(v, t):
    return sp.sqrt(sum(sp.expand(e ** 2) for e in v))


@fn("FrenetSerretSystem", area=AREA, status="added",
    notes="FrenetSerretSystem[{x[t], y[t]}, t] = {{k}, {T, N}} with the signed curvature k = (x' y'' - y' x'')/|r'|^3 "
          "and N = T rotated by Pi/2; FrenetSerretSystem[{x, y, z}, t] = {{k, tau}, {T, N, B}} (curvature, torsion, "
          "tangent, normal, binormal); t is assumed real; curves of other dimensions raise")
def _frenet(c, t):
    xs = _vec(c, "FrenetSerretSystem")
    tv = _svar(t, "FrenetSerretSystem")
    tr = sp.Symbol(tv.name, real=True)
    xs = [x.xreplace({tv: tr}) for x in xs]
    back = {tr: tv}

    def S_(e):
        return out(sp.simplify(e).xreplace(back))
    r1 = sp.Matrix([sp.diff(x, tr) for x in xs])
    r2 = r1.diff(tr)
    sp_ = sp.sqrt(sp.simplify(sum(e ** 2 for e in r1)))
    if sp.simplify(sp_) == 0:
        raise ValueError("FrenetSerretSystem: the curve is not regular (zero velocity)")
    T = (r1 / sp_).applyfunc(sp.simplify)
    if len(xs) == 2:
        k = sp.simplify((r1[0] * r2[1] - r1[1] * r2[0]) / sp_ ** 3)
        N = sp.Matrix([-T[1], T[0]])
        return L(L(S_(k)), L(L(*(S_(e) for e in T)), L(*(S_(e) for e in N))))
    if len(xs) == 3:
        r3 = r2.diff(tr)
        cr = r1.cross(r2)
        cn2 = sp.simplify(sum(e ** 2 for e in cr))
        if cn2 == 0:
            raise ValueError("FrenetSerretSystem: the curvature vanishes identically (a straight line has no "
                             "Frenet frame)")
        cn = sp.sqrt(cn2)
        k = sp.simplify(cn / sp_ ** 3)
        tau = sp.simplify(cr.dot(r3) / cn2)
        B = (cr / cn).applyfunc(sp.simplify)
        N = B.cross(T).applyfunc(sp.simplify)
        return L(L(S_(k), S_(tau)), L(L(*(S_(e) for e in T)), L(*(S_(e) for e in N)), L(*(S_(e) for e in B))))
    raise NotImplementedError("FrenetSerretSystem: only plane and space curves are supported")


# ───────────────────────────── asymptotic engine ─────────────────────────────

class _AsymFail(Exception):
    pass


def _point(v, fname):
    p = SX(v, fname)
    if p == sp.zoo:
        raise NotImplementedError(f"{fname}: expansions at ComplexInfinity are not supported")
    if p.free_symbols:
        raise NotImplementedError(f"{fname}: the expansion point must be numeric or Infinity")
    return p


def _parse_point_spec(spec, opts, fname):
    """x -> x0 (leading term unless SeriesTermGoal) or {x, x0, n} -> (x, x0, order or None)."""
    goal = opts.get("SeriesTermGoal")
    if goal is not None and not _is(goal, "Automatic"):
        goal = _num_opt(goal, fname, "SeriesTermGoal")
    else:
        goal = None
    if is_rule(spec):
        x = _svar(spec.args[0], fname)
        return x, _point(spec.args[1], fname), goal
    if is_list(spec) and len(spec.args) == 3:
        x = _svar(spec.args[0], fname)
        n = int_arg(spec.args[2], fname, "order")
        if n < 0:
            raise ValueError(f"{fname}: the order must be a non-negative integer")
        return x, _point(spec.args[1], fname), n
    raise ValueError(f"{fname}: the expansion point must be x -> x0 or {{x, x0, n}}")


def _rewrite_special(f, discrete=False):
    reps = {}
    for g in f.atoms(sp.Function):
        if isinstance(g, sp.factorial):
            reps[g] = sp.gamma(g.args[0] + 1)
        elif isinstance(g, sp.binomial):
            a, b = g.args
            reps[g] = sp.gamma(a + 1) / (sp.gamma(b + 1) * sp.gamma(a - b + 1))
        elif isinstance(g, sp.harmonic) and len(g.args) == 1:
            reps[g] = sp.digamma(g.args[0] + 1) + sp.EulerGamma
        elif isinstance(g, sp.fibonacci) and len(g.args) == 1:
            phi = (1 + sp.sqrt(5)) / 2
            reps[g] = (phi ** g.args[0] - (1 - phi) ** g.args[0]) / sp.sqrt(5)
        elif isinstance(g, sp.lucas):
            phi = (1 + sp.sqrt(5)) / 2
            reps[g] = phi ** g.args[0] + (1 - phi) ** g.args[0]
        elif isinstance(g, sp.RisingFactorial):
            a, b = g.args
            reps[g] = sp.gamma(a + b) / sp.gamma(a)
        elif isinstance(g, sp.FallingFactorial):
            a, b = g.args
            reps[g] = sp.gamma(a + 1) / sp.gamma(a - b + 1)
        elif isinstance(g, sp.catalan):
            nn = g.args[0]
            reps[g] = sp.gamma(2 * nn + 1) / (sp.gamma(nn + 1) * sp.gamma(nn + 2))
    if reps:
        f = f.xreplace(reps)
    return f


def _num_ratio_check(f, g, x, x0, tol=1e-2):
    """Numerical check that f/g -> 1 as x -> x0 (fallback when SymPy's limit cannot decide)."""
    if _is_inf(x0):
        pts = [sp.Integer(10) ** k for k in (3, 5, 8)]
        if x0 == -sp.oo:
            pts = [-p for p in pts]
    else:
        pts = [x0 + sp.Rational(1, 10 ** k) for k in (3, 5, 8)]
    errs = []
    for p in pts:
        try:
            v = complex(sp.N((f / g).subs(x, p), 40))
        except Exception:
            return False
        errs.append(abs(v - 1))
    return errs[-1] < tol and errs[-1] <= errs[0] + 1e-30


def _limit(e, x, x0, dirn=None):
    try:
        if _is_inf(x0):
            return sp.limit(e, x, x0)
        if dirn:
            return sp.limit(e, x, x0, dirn)
        try:
            return sp.limit(e, x, x0, "+-")
        except (ValueError, NotImplementedError):
            lp, lm = sp.limit(e, x, x0, "+"), sp.limit(e, x, x0, "-")
            if lp == lm:
                return lp
            return sp.nan
    except Exception:
        return None


def _verify_lead(f, g, x, x0):
    if g == 0:
        return f == 0
    r = _limit(sp.simplify(f / g) if f.count_ops() < 60 else f / g, x, x0, None if _is_inf(x0) else "+")
    if r is not None and r == 1:
        return True
    if r is not None and r.is_number and r != 1 and not isinstance(r, sp.AccumBounds) and r is not sp.nan:
        return False
    return _num_ratio_check(f, g, x, x0)


def _split_osc(t, x):
    """t = osc * rest where osc collects (-1)^x-type factors (negative or complex bases raised to x)."""
    osc, rest = sp.Integer(1), []
    for fct in sp.Mul.make_args(t):
        if isinstance(fct, sp.Pow) and fct.exp.has(x) and fct.base.is_number:
            b = fct.base
            if b.is_negative:
                osc *= sp.Integer(-1) ** fct.exp
                rest.append((-b) ** fct.exp)
                continue
            if b.is_real is False:
                raise _AsymFail("complex exponential bases are not supported")
        rest.append(fct)
    return osc, sp.Mul(*rest)


def _log_expand(t, x, order):
    c, P, corr = _log_parts(t, x, order)
    return c * sp.powsimp(sp.exp(P), force=True) * sp.expand(corr)


def _log_parts(t, x, order):
    """(c, P, corr): t ~ c * exp(P) * corr with corr = 1 + O(1/x) for a single positive-for-large-x term."""
    c, rest = t.as_independent(x, as_Add=False)
    if rest == 1:
        return c, sp.Integer(0), sp.Integer(1)
    lg = sp.expand_log(sp.log(rest.rewrite(sp.gamma).xreplace(
        {g: sp.exp(sp.loggamma(g.args[0])) for g in rest.rewrite(sp.gamma).atoms(sp.gamma)})), force=True)
    try:
        s = sp.series(lg, x, sp.oo, max(order, 1) + 2).removeO()
    except Exception as exc:
        raise _AsymFail(str(exc))
    P, Q = sp.Integer(0), sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(s)):
        lim = _limit(term, x, sp.oo)
        if lim == 0:
            Q += term
        else:
            P += term
    P = sp.expand_log(sp.logcombine(P, force=True), force=True)
    if order == 0:
        corr = sp.Integer(1)
    else:
        w = sp.Symbol("w_", positive=True)
        try:
            corr = sp.series(sp.exp(Q.subs(x, 1 / w)), w, 0, order + 1).removeO().subs(w, 1 / x)
        except Exception as exc:
            raise _AsymFail(str(exc))
    return c, P, corr


def _dominant(terms, x, x0):
    """Indices of the terms not dominated by another term (as x -> x0)."""
    keep = []
    for i, a in enumerate(terms):
        dominated = False
        for j, b in enumerate(terms):
            if i == j:
                continue
            r = _limit(sp.Abs(sp.simplify(a / b)), x, x0)
            if r is None:
                r = _limit(a / b, x, x0)
            if r is None:
                raise _AsymFail("cannot compare the growth of the terms")
            if r == 0:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def _lead_finite(f, x, x0):
    t = sp.Dummy("t")
    g = f.subs(x, x0 + t)
    try:
        ld = g.as_leading_term(t)
    except Exception as exc:
        raise _AsymFail(str(exc))
    ld = ld.subs(t, x - x0)
    if ld.has(sp.Order) or ld == 0:
        raise _AsymFail("no leading term")
    return ld


def asym(f, x, x0, order=None, discrete=False):
    """Asymptotic approximation of f as x -> x0: the leading term (order None) or the expansion to order n
    (Taylor/Laurent/Puiseux order n at finite x0; at Infinity, terms down to x^-n, or relative corrections
    down to x^-n times the leading exponential/power factor).  Raises _AsymFail when no verified result."""
    f = _rewrite_special(sp.sympify(f), discrete)
    if not f.has(x):
        return f
    if not _is_inf(x0):
        if order is None:
            ld = _lead_finite(f, x, x0)
            if not _verify_lead(f, ld, x, x0):
                raise _AsymFail("leading term could not be verified")
            return ld
        try:
            s = sp.series(f, x, x0, order + 1)
        except Exception as exc:
            raise _AsymFail(str(exc))
        return s.removeO()
    sgn = 1
    if x0 == -sp.oo:
        y = sp.Dummy("y", positive=True)
        return asym(f.subs(x, -y), y, sp.oo, order, discrete).subs(y, -x)
    xp = sp.Symbol(x.name, positive=True)
    g = f.xreplace({x: xp})
    res = _asym_inf(g, xp, order)
    return res.xreplace({xp: x})


def _special_inf(f, x, order):
    """Replace erfc/erf/E_nu of arguments -> +Infinity by their (exact) asymptotic series."""
    J = (order if order is not None else 1) + 3

    def to_inf(g):
        r = _limit(g, x, sp.oo)
        return r == sp.oo

    def erfc_s(z):
        return sp.exp(-z ** 2) / (z * sp.sqrt(sp.pi)) * sum(
            (-1) ** k * sp.factorial2(2 * k - 1) / (2 * z ** 2) ** k for k in range(J))

    def expint_s(nu, z):
        return sp.exp(-z) / z * sum((-1) ** k * sp.rf(nu, k) / z ** k for k in range(J))
    f = f.replace(lambda e: isinstance(e, sp.erfc) and e.args[0].has(x) and to_inf(e.args[0]),
                  lambda e: erfc_s(e.args[0]))
    f = f.replace(lambda e: isinstance(e, sp.erf) and e.args[0].has(x) and to_inf(e.args[0]),
                  lambda e: 1 - erfc_s(e.args[0]))
    f = f.replace(lambda e: isinstance(e, sp.expint) and e.args[1].has(x) and not e.args[0].has(x)
                  and to_inf(e.args[1]), lambda e: expint_s(e.args[0], e.args[1]))
    return f


def _asym_inf(f, x, order):
    if f.has(sp.erfc, sp.erf, sp.expint):
        g = _special_inf(f, x, order)
        if g != f:
            return _asym_inf(sp.expand(g) if order is not None else g, x, order)
    # 1. SymPy's own expansion when it is informative
    direct = None
    if not any(isinstance(b, sp.Pow) and b.exp.has(x) and b.base.is_number and b.base.is_negative
               for b in f.atoms(sp.Pow)) and not f.has(sp.gamma, sp.loggamma):
        try:
            s = sp.series(f, x, sp.oo, (order if order is not None else 2) + 1)
            direct = s.removeO()
            if direct.has(sp.Order):
                direct = sp.expand(s).removeO()
            if direct == 0 or direct.has(sp.Order):
                direct = None
        except Exception:
            direct = None
    if direct is not None:
        if order is not None:
            terms = sp.Add.make_args(sp.expand(direct))
            keep = []
            for t in terms:
                # keep terms down to x^-order (relative to x^0)
                r = _limit(t * x ** order, x, sp.oo)
                if r is None or r != 0 or t.has(sp.exp):
                    keep.append(t)
            return sp.Add(*keep)
        terms = list(sp.Add.make_args(sp.expand(direct)))
        idx = _dominant(terms, x, sp.oo)
        ld = sp.Add(*[terms[i] for i in idx])
        if _verify_lead(f, ld, x, sp.oo):
            return ld
    # 2. term-wise: oscillating factors kept, log-exp expansion of each term, dominance
    terms = list(sp.Add.make_args(sp.expand(f, power_exp=False, log=False, mul=True, multinomial=False)))
    if order is not None and len(terms) > 1:
        return _grouped_expansion(terms, x, order)
    reps = []
    for t in terms:
        osc, rest = _split_osc(t, x)
        reps.append((osc, _log_expand(rest, x, order or 0)))
    mags = [r for _o, r in reps]
    idx = _dominant(mags, x, sp.oo) if len(mags) > 1 else [0]
    res = sp.Add(*[reps[i][0] * reps[i][1] for i in idx])
    if order is None:
        chk = sp.Add(*[reps[i][1] * reps[i][0] for i in idx])
        if not _verify_lead(sp.Add(*[terms[i] for i in idx]), chk, x, sp.oo):
            raise _AsymFail("leading term could not be verified")
    return res


def _grouped_expansion(terms, x, order):
    """Expansion to relative order: terms grouped by their exponential scale exp(Q) (Q the part of the log not
    of the form a Log[x] + b); within a group the power corrections are summed and truncated at x^-order relative to
    the group's leading power; dominated (exponentially smaller) groups are dropped."""
    groups = {}
    keys = []
    for t in terms:
        osc, rest = _split_osc(t, x)
        c, P, corr = _log_parts(rest, x, order)
        Q, Rr = sp.Integer(0), sp.Integer(0)
        for term in sp.Add.make_args(sp.expand(P)):
            cc, pw = term.as_independent(x, as_Add=False)
            if pw == 1 or (isinstance(pw, sp.log) and pw.args[0] == x):
                Rr += term
            else:
                Q += term
        key = (sp.srepr(sp.simplify(Q)), sp.srepr(osc))
        if key not in groups:
            groups[key] = [Q, osc, sp.Integer(0)]
            keys.append(key)
        groups[key][2] += c * sp.powsimp(sp.exp(Rr), force=True) * sp.expand(corr)
    reps = []
    for k in keys:
        Q, osc, S = groups[k]
        S = sp.expand(S)
        ts = list(sp.Add.make_args(S))
        if not ts or S == 0:
            continue
        # leading power within the group
        lead = ts[_dominant(ts, x, sp.oo)[0]] if len(ts) > 1 else ts[0]
        kept = [u for u in ts if _limit(sp.simplify(u / lead) * x ** order, x, sp.oo) not in (0,)]
        reps.append((osc, sp.exp(Q), sp.Add(*kept)))
    if not reps:
        return sp.Integer(0)
    mags = [e * sp.Add.make_args(S_)[0] for _o, e, S_ in reps]
    idx = _dominant(mags, x, sp.oo) if len(mags) > 1 else [0]
    return sp.Add(*[reps[i][0] * reps[i][1] * reps[i][2] for i in idx])


def _exp_to_pow(v):
    """E^(a Log[b] + rest) -> b^a E^rest for numeric b (GoldenRatio^n rather than E^(n Log[GoldenRatio]))."""
    def conv(e):
        terms = sp.Add.make_args(sp.expand(e.args[0]))
        pw, rest = sp.Integer(1), []
        for t in terms:
            c, lg = t.as_independent(sp.log, as_Add=False)
            if isinstance(lg, sp.log) and lg.args[0].is_number and lg.args[0].is_positive:
                pw *= lg.args[0] ** c
            else:
                rest.append(t)
        return pw * (sp.exp(sp.factor_terms(sp.Add(*rest))) if rest else 1)
    return v.replace(lambda e: isinstance(e, sp.exp), conv)


def _asym_out(v):
    return out(_exp_to_pow(v) if isinstance(v, sp.Basic) else v)


def _run_asym(f, x, x0, order, fname, discrete=False):
    try:
        return asym(f, x, x0, order, discrete)
    except _AsymFail as exc:
        raise NotImplementedError(f"{fname}: no asymptotic approximation found ({exc})")


@fn("Asymptotic", area=AREA, status="added",
    notes="Asymptotic[f, x -> x0]: the leading term of the asymptotic expansion (verified: f/term -> 1); "
          "Asymptotic[f, {x, x0, n}] / SeriesTermGoal -> n: the expansion to order n (Taylor, Laurent or Puiseux "
          "at finite x0; at Infinity down to x^-n, with exponential/power prefactors and Stirling-type expansions "
          "of Gamma, Factorial, Binomial, HarmonicNumber via LogGamma); unexpandable input raises")
def _asymptotic(f, spec, *rest):
    pos, o = _opts(rest, {"SeriesTermGoal", "Assumptions"}, "Asymptotic")
    if pos:
        raise ValueError("Asymptotic: too many arguments")
    x, x0, n = _parse_point_spec(spec, o, "Asymptotic")
    return _asym_out(_run_asym(SX(f, "Asymptotic"), x, x0, n, "Asymptotic"))


@fn("DiscreteAsymptotic", area=AREA, status="added",
    notes="DiscreteAsymptotic[f, n -> Infinity] / {n, Infinity, k}: asymptotic approximation of a sequence (n an "
          "integer): Fibonacci, LucasL, factorials, binomials, HarmonicNumber, CatalanNumber are rewritten "
          "(Binet, Stirling/LogGamma), (-1)^n-type factors are kept as oscillating factors; dominated terms are "
          "dropped; leading terms are verified")
def _discasym(f, spec, *rest):
    pos, o = _opts(rest, {"SeriesTermGoal", "Assumptions"}, "DiscreteAsymptotic")
    if pos:
        raise ValueError("DiscreteAsymptotic: too many arguments")
    x, x0, n = _parse_point_spec(spec, o, "DiscreteAsymptotic")
    if not _is_inf(x0):
        raise ValueError("DiscreteAsymptotic: the limit point must be Infinity or -Infinity")
    return _asym_out(_run_asym(SX(f, "DiscreteAsymptotic"), x, x0, n, "DiscreteAsymptotic", discrete=True))


# ───────────────────────────── asymptotic relations ─────────────────────────────

def _ratio_limit(f, g, x, x0, discrete=False):
    """(lim |f/g|) as a SymPy value, an AccumBounds of |f/g|, or None (undecided)."""
    if g == 0:
        raise ValueError("comparison with the zero function")
    r = sp.simplify(f / g)
    if discrete:
        v = _disc_limits(r, x)
        if v is None:
            return None
        lo, hi = v
        if lo == hi:
            return sp.Abs(lo)
        return ("osc", lo, hi)
    L_ = _limit(sp.Abs(r), x, x0)
    if L_ is None or L_ is sp.nan:
        L_ = _limit(r, x, x0)
        if L_ is None or L_ is sp.nan:
            return None
        if isinstance(L_, sp.AccumBounds):
            return ("osc", L_.min, L_.max)
        return sp.Abs(L_)
    if isinstance(L_, sp.AccumBounds):
        return ("osc", L_.min, L_.max)
    return L_


def _relation(kind, f, g, spec, rest, fname):
    pos, _o = _opts(rest, {"Assumptions"}, fname)
    if pos:
        raise ValueError(f"{fname}: too many arguments")
    if not is_rule(spec):
        raise ValueError(f"{fname}: the third argument must be x -> x0")
    x = _svar(spec.args[0], fname)
    x0 = _point(spec.args[1], fname)
    F, G = SX(f, fname), SX(g, fname)
    other = (F.free_symbols | G.free_symbols) - {x}
    if other:
        raise Unevaluated()
    L_ = _ratio_limit(_rewrite_special(F), _rewrite_special(G), x, x0)
    if L_ is None:
        raise Unevaluated()
    if isinstance(L_, tuple):
        _t, lo, hi = L_
        bounded = lo.is_finite and hi.is_finite
        if kind == "LessEqual":
            if bounded:
                return True
            raise Unevaluated()
        if kind in ("Less", "Equivalent"):
            return False
        if kind == "Greater":
            if bounded:
                return False
            raise Unevaluated()
        raise Unevaluated()
    if L_.free_symbols:
        raise Unevaluated()
    zero = L_ == 0
    inf = L_ in (sp.oo, sp.zoo) or L_.is_infinite
    finite_nz = not zero and not inf
    return {"Less": zero, "LessEqual": not inf, "Equal": finite_nz, "GreaterEqual": not zero, "Greater": inf,
            "Equivalent": finite_nz and sp.simplify(L_ - 1) == 0 and _signed_one(F, G, x, x0)}[kind]


def _signed_one(F, G, x, x0):
    r = _limit(sp.simplify(F / G), x, x0)
    return r is not None and r == 1


for _k, _n, _d in (("Less", "AsymptoticLess", "f = o(g): f/g -> 0"),
                   ("LessEqual", "AsymptoticLessEqual", "f = O(g): |f/g| bounded"),
                   ("Equal", "AsymptoticEqual", "f = Theta(g): |f/g| bounded above and away from 0"),
                   ("GreaterEqual", "AsymptoticGreaterEqual", "f = Omega(g): |f/g| bounded away from 0"),
                   ("Greater", "AsymptoticGreater", "f = omega(g): |f/g| -> Infinity"),
                   ("Equivalent", "AsymptoticEquivalent", "f ~ g: f/g -> 1")):
    def _mkrel(kind, name):
        def f(a, b, spec, *rest):
            return _relation(kind, a, b, spec, rest, name)
        return f
    fn(_n, area=AREA, status="added",
       notes=f"{_n}[f, g, x -> x0]: True when {_d} as x -> x0 (decided by the limit of |f/g|; oscillating ratios "
             "with finite bounds are decided where the bounds suffice), False otherwise; unevaluated when the "
             "limit cannot be decided or depends on parameters")(_mkrel(_k, _n))


# ───────────────────────────── discrete limits ─────────────────────────────

def _period(f, n):
    """A period p such that f's oscillating parts depend only on n mod p ((-1)^n, Cos[Pi q n], E^(I Pi q n)),
    or None when the oscillation is not periodic."""
    p = 1
    for a in f.atoms(sp.Pow):
        if a.exp.has(n) and a.base.is_number and a.base.is_negative:
            if a.base != -1:
                return None
            k = sp.Poly(a.exp, n) if a.exp.is_polynomial(n) else None
            if k is None or k.degree() != 1:
                return None
            c = k.LC()
            if not c.is_rational:
                return None
            p = sp.ilcm(p, 2 * sp.Rational(c).q)
    for a in f.atoms(sp.sin, sp.cos, sp.tan, sp.exp):
        arg = a.args[0] if not isinstance(a, sp.exp) else a.args[0] / sp.I
        if not arg.has(n):
            continue
        if not arg.is_polynomial(n) or sp.degree(arg, n) != 1:
            return None
        c = sp.Poly(arg, n).LC() / sp.pi
        if isinstance(a, sp.exp) and not (a.args[0] / sp.I).is_real:
            if (a.args[0]).is_real:
                continue          # a real exponential, not oscillating
            return None
        if not c.is_rational:
            return None
        p = sp.ilcm(p, 2 * sp.Rational(c).q)
    return int(p)


def _neg_bases(f, n):
    """b^e with b a negative number and e depending on n -> (-1)^e (-b)^e."""
    return f.replace(lambda e: isinstance(e, sp.Pow) and e.exp.has(n) and e.base.is_number and e.base.is_negative
                     and e.base != -1, lambda e: sp.Pow(-1, e.exp, evaluate=False) * (-e.base) ** e.exp)


def _osc_atom(a, n):
    if isinstance(a, sp.Pow):
        return a.base == -1 and a.exp.has(n)
    if isinstance(a, (sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc)):
        return a.args[0].has(n)
    if isinstance(a, sp.exp):
        return a.args[0].has(n) and (a.args[0] / sp.I).is_real is True
    return False


def _disc_limits(f, n):
    """(liminf, limsup) of the sequence f(n) as n -> oo, or None when undecided."""
    f = _neg_bases(f, n)
    p = _period(f, n)
    if p is None:
        return None
    m = sp.Symbol("m_", positive=True)
    vals = []
    atoms = [a for a in sp.preorder_traversal(f) if _osc_atom(a, n)]
    for r in range(p):
        reps = {a: a.subs(n, r) for a in atoms}
        g = f.xreplace(reps).subs(n, p * m + r)
        try:
            v = sp.limit(g, m, sp.oo)
        except Exception:
            v = None
        if v is None or isinstance(v, (sp.AccumBounds, sp.Limit)) or v is sp.nan:
            try:
                from sympy.series.limitseq import limit_seq
                v = limit_seq(g.subs(m, sp.Symbol("m_", integer=True, positive=True)),
                              sp.Symbol("m_", integer=True, positive=True))
            except Exception:
                v = None
        if v is None or isinstance(v, (sp.AccumBounds, sp.Limit)) or v is sp.nan:
            return None
        if v.free_symbols:
            return None
        vals.append(sp.nsimplify(v) if isinstance(v, sp.Float) else v)
    if any(v == sp.zoo for v in vals):
        return None

    def key(v):
        if v == sp.oo:
            return (1, 0)
        if v == -sp.oo:
            return (-1, 0)
        return (0, float(v))
    try:
        return min(vals, key=key), max(vals, key=key)
    except TypeError:
        return None


def _disc_spec(spec, fname):
    if not is_rule(spec):
        raise ValueError(f"{fname}: the second argument must be n -> Infinity")
    nv = _svar(spec.args[0], fname)
    p = SX(spec.args[1], fname)
    if p != sp.oo:
        raise ValueError(f"{fname}: only limits n -> Infinity are supported")
    return nv


def _disc_common(f, spec, rest, fname):
    pos, _o = _opts(rest, {"Assumptions", "GenerateConditions"}, fname)
    if pos:
        raise ValueError(f"{fname}: too many arguments")
    if is_list(f):
        return [(_disc_common(g, spec, rest, fname)) for g in f.args]
    if is_list(spec.args[0]) if is_rule(spec) else False:
        raise NotImplementedError(f"{fname}: multivariate discrete limits are not supported")
    nv = _disc_spec(spec, fname)
    F = _rewrite_special(SX(f, fname), True)
    ni = sp.Symbol(nv.name, integer=True, positive=True)
    F = F.xreplace({nv: ni})
    if (F.free_symbols - {ni}):
        try:
            v = sp.limit(F, ni, sp.oo)
            if not isinstance(v, (sp.AccumBounds, sp.Limit)) and v is not sp.nan and _period(F, ni) == 1:
                return (v, v)
        except Exception:
            pass
        raise Unevaluated()
    r = _disc_limits(F, ni)
    if r is None:
        raise Unevaluated()
    return r


def _disc_wrap(f, spec, rest, fname, pick):
    r = _disc_common(f, spec, rest, fname)
    if isinstance(r, list):
        return L(*(out(pick(v)) for v in r))
    return out(pick(r))


@fn("DiscreteLimit", area=AREA, status="added",
    notes="DiscreteLimit[f, n -> Infinity]: the limit of the sequence f(n) over integers n; sequences with periodic "
          "oscillation ((-1)^n, Sin[Pi q n], ...) are split into residue classes n = p m + r (Indeterminate when the "
          "subsequence limits differ); Fibonacci/factorial/binomial forms are rewritten; threads over lists; "
          "unevaluated when undecided (e.g. Sin[n])")
def _disclimit(f, spec, *rest):
    return _disc_wrap(f, spec, rest, "DiscreteLimit", lambda r: r[0] if r[0] == r[1] else sp.nan)


@fn("DiscreteMaxLimit", area=AREA, status="added",
    notes="DiscreteMaxLimit[f, n -> Infinity]: lim sup of the sequence f(n) (residue-class splitting for periodic "
          "oscillation; e.g. (-1)^n -> 1, Sin[n Pi/3] -> Sqrt[3]/2); unevaluated when undecided")
def _discmaxlimit(f, spec, *rest):
    return _disc_wrap(f, spec, rest, "DiscreteMaxLimit", lambda r: r[1])


@fn("DiscreteMinLimit", area=AREA, status="added",
    notes="DiscreteMinLimit[f, n -> Infinity]: lim inf of the sequence f(n) (residue-class splitting for periodic "
          "oscillation); unevaluated when undecided")
def _discminlimit(f, spec, *rest):
    return _disc_wrap(f, spec, rest, "DiscreteMinLimit", lambda r: r[0])


# ───────────────────────────── Asymptotic* solvers ─────────────────────────────

def _c_to_sym(e):
    """C[k] -> symbol __C_k (for SymPy) ; returns the converted expression."""
    if isinstance(e, Expr):
        if e.head == "C" and len(e.args) == 1 and K.is_int(e.args[0]):
            return Symbol(f"__C_{int(e.args[0])}")
        h = _c_to_sym(e.head) if isinstance(e.head, Expr) else e.head
        return Expr(h, *(_c_to_sym(a) for a in e.args))
    return e


def _sym_to_c(e):
    if isinstance(e, Symbol) and e.name.startswith("__C_"):
        return Expr("C", int(e.name[4:]))
    if isinstance(e, Expr):
        h = _sym_to_c(e.head) if isinstance(e.head, Expr) else e.head
        return Expr(h, *(_sym_to_c(a) for a in e.args))
    return e


def _csyms(v):
    return sorted([s for s in v.free_symbols if s.name.startswith("__C_")], key=lambda s: int(s.name[4:]))


def _asym_with_constants(v, x, x0, order, fname, discrete=False):
    """Asymptotics of a general solution sum C_i y_i(x): each constant's part is expanded separately."""
    cs = _csyms(v)
    if not cs:
        return _run_asym(v, x, x0, order, fname, discrete)
    v = sp.expand(v)
    parts = []
    rest = v
    for c in cs:
        part = sp.expand(v).coeff(c)
        rest = sp.expand(rest - c * part)
        if part != 0:
            parts.append(c * _run_asym(part, x, x0, order, fname, discrete))
    if sp.simplify(rest) != 0:
        parts.append(_run_asym(rest, x, x0, order, fname, discrete))
    return sp.Add(*parts)


def _num_close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1, abs(b))


def _check_sum_closed(S, f, k, a, nvar, fname):
    for N in (a + 3, a + 6):
        try:
            direct = sum(complex(sp.N(f.subs(k, j), 30)) for j in range(int(a), int(N) + 1))
            closed = complex(sp.N(S.subs(nvar, N), 30))
        except Exception:
            raise NotImplementedError(f"{fname}: the closed form could not be checked")
        if not _num_close(closed, direct, 1e-8):
            raise NotImplementedError(f"{fname}: the closed form of the sum failed a numerical check")


def _sum_spec(spec, fname):
    if not (is_list(spec) and len(spec.args) == 3):
        raise ValueError(f"{fname}: the summation range must be {{k, a, b}}")
    return _svar(spec.args[0], fname), SX(spec.args[1], fname), SX(spec.args[2], fname)


def _eval_point(spec, opts, fname):
    x, x0, n = _parse_point_spec(spec, opts, fname)
    return x, x0, n


def _hurwitz_em(v, x, order):
    """Replace Hurwitz zeta(s, z) (z -> oo with x) by its Euler-Maclaurin expansion (exact asymptotics)."""
    def em(e):
        if len(e.args) != 2 or not e.args[1].has(x):
            return e
        s, z = e.args
        J = order + 3
        r = z ** (1 - s) / (s - 1) + z ** (-s) / 2
        for j in range(1, J + 1):
            r += sp.bernoulli(2 * j) / sp.factorial(2 * j) * sp.rf(s, 2 * j - 1) * z ** (-s - 2 * j + 1)
        return r
    v = v.replace(lambda e: isinstance(e, sp.harmonic) and len(e.args) == 2,
                  lambda e: sp.zeta(e.args[1]) - sp.zeta(e.args[1], e.args[0] + 1))
    return v.replace(lambda e: isinstance(e, sp.zeta) and len(e.args) == 2, em)


def _asym_closed(S, x, x0, n, fname, discrete=True):
    if _is_inf(x0):
        S = _hurwitz_em(S, x, n if n is not None else 2)
    return _run_asym(S, x, x0, n, fname, discrete)


def _exp_pow(F, k):
    """E^(c k) -> (E^c)^k so that SymPy recognises geometric sums."""
    return F.replace(lambda e: isinstance(e, sp.exp) and e.args[0].has(k) and not sp.cancel(e.args[0] / k).has(k),
                     lambda e: sp.Pow(sp.exp(sp.cancel(e.args[0] / k)), k))


def _summation(F, k, a, b):
    """sp.summation with E^(c k) -> q^k (geometric sums with symbolic ratio)."""
    F = _exp_pow(F, k)
    reps = {}
    for pw in F.atoms(sp.Pow):
        if pw.exp.has(k) and not pw.base.has(k) and not pw.base.is_number:
            if pw.base not in reps:
                reps[pw.base] = sp.Dummy("q")
    if not reps:
        return sp.summation(F, (k, a, b))
    G = F.replace(lambda e: isinstance(e, sp.Pow) and e.base in reps, lambda e: reps[e.base] ** e.exp)
    S = sp.summation(G, (k, a, b))
    back = {v: kk for kk, v in reps.items()}
    return S.xreplace(back)


def _nsum(F, k, a, b):
    f = sp.lambdify(k, F, "mpmath")
    alt = any(isinstance(e, sp.Pow) and e.base == -1 for e in F.atoms(sp.Pow))
    hi = mpmath.inf if b == sp.oo else int(b)
    if alt:
        return complex(mpmath.nsum(lambda j: f(int(j)), [int(a), hi]))
    return complex(mpmath.nsum(f, [int(a), hi], method="euler-maclaurin"))


def _partial_sum(F, k, a, b):
    """Closed form of Sum[F, {k, a, b}] (b symbolic): partial fractions, power terms via Hurwitz zeta."""
    F = _exp_pow(F, k)
    S = _summation(F, k, a, b)
    if not S.has(sp.Sum) and not isinstance(S, sp.Piecewise):
        return S
    tot = 0
    Fe = sp.apart(F, k) if F.is_rational_function(k) else sp.expand(F)
    for t in sp.Add.make_args(Fe):
        c, pw = t.as_independent(k, as_Add=False)
        sh = None
        if isinstance(pw, sp.Pow) and pw.exp.is_number and pw.base.is_polynomial(k) and sp.degree(pw.base, k) == 1 \
                and sp.Poly(pw.base, k).LC() == 1 and sp.Poly(pw.base, k).TC().is_Integer:
            sh = int(sp.Poly(pw.base, k).TC())
        if sh is not None and a.is_Integer and a + sh >= 1:
            e = pw.exp
            a, b = a + sh, b + sh
            if e == -1:
                st = sp.harmonic(b)
            else:
                st = sp.zeta(-e) - sp.zeta(-e, b + 1)
            tot += c * (st - sum(sp.Integer(j) ** e for j in range(1, int(a))))
            a, b = a - sh, b - sh
            continue
        if pw == k and a.is_Integer:
            tot += c * sp.summation(k, (k, a, b))
            continue
        st = _summation(t, k, a, b)
        if st.has(sp.Sum) or isinstance(st, sp.Piecewise):
            return None
        tot += st
    return tot


def _rational_tail(F, k):
    """-Sum[F(j), {j, k, Infinity}] for a rational F = O(1/k^2), via polygamma functions of k - root."""
    if not F.is_rational_function(k):
        return None
    num, den = sp.fraction(sp.cancel(F))
    if sp.degree(den, k) < sp.degree(num, k) + 2:
        return None
    rts = sp.roots(sp.Poly(den, k))
    if sum(rts.values()) != sp.degree(den, k):
        return None
    tail = 0
    for r, mult in rts.items():
        for m in range(1, mult + 1):
            # coefficient of 1/(k - r)^m in the partial fraction expansion
            c = sp.limit(sp.diff(sp.cancel(F * (k - r) ** mult), k, mult - m) / sp.factorial(mult - m), k, r)
            # Sum_{j>=k} 1/(j - r)^m = (-1)^m/(m-1)! polygamma(m-1, k - r)   (m >= 2); digamma terms combine
            if m == 1:
                tail += -c * sp.polygamma(0, k - r)
            else:
                tail += c * (-1) ** m * sp.polygamma(m - 1, k - r) / sp.factorial(m - 1)
    return -tail


def _termwise_param_sum(F, k, a, b, x, x0, n):
    m = n if n is not None else 3
    try:
        s = sp.series(F, x, x0, m + 1).removeO()
    except Exception:
        return None
    res = 0
    xs = x - x0
    s = sp.expand(s)
    for p in range(m + 1):
        c = s.coeff(x, p) if x0 == 0 else sp.expand(s.subs(x, xs + x0)).coeff(xs, p)
        if c == 0:
            continue
        st = sp.summation(c, (k, a, b))
        if st.has(sp.Sum) or not st.is_finite:
            return None
        res += st * (x - x0) ** p
    return res


@fn("AsymptoticSum", area=AREA, status="added",
    notes="AsymptoticSum[f, {k, a, n}, n -> Infinity | {n, Infinity, m}]: asymptotics of a partial sum, "
          "AsymptoticSum[f, {k, a, Infinity}, {p, p0, m}]: expansion of a parametric infinite sum, "
          "AsymptoticSum[f, k, {k, Infinity, m}]: the indefinite sum (the tail convention F(Infinity) = 0 for "
          "convergent sums, else Sum over 0..k-1); computed from SymPy's closed form (checked numerically against "
          "direct summation) with Euler-Maclaurin expansions of Hurwitz zeta / generalized harmonic numbers; "
          "no closed form -> error")
def _asymsum(f, rng, spec, *rest):
    pos, o = _opts(rest, {"SeriesTermGoal", "Assumptions", "GenerateConditions"}, "AsymptoticSum")
    if pos:
        raise ValueError("AsymptoticSum: too many arguments")
    F = SX(f, "AsymptoticSum")
    x, x0, n = _eval_point(spec, o, "AsymptoticSum")
    if isinstance(rng, Symbol):
        k = _svar(rng, "AsymptoticSum")
        if x != k or x0 != sp.oo:
            raise NotImplementedError("AsymptoticSum: indefinite sums are supported for k -> Infinity only")
        j = sp.Dummy("j", integer=True)
        tail = sp.summation(F.subs(k, j), (j, k, sp.oo))
        rt = _rational_tail(F, k)
        if rt is not None:
            S = rt
        elif not tail.has(sp.Sum) and tail.is_finite is not False and not tail.has(sp.oo, sp.zoo):
            S = -tail
        else:
            lo = 0
            while lo < 5 and not sp.sympify(F.subs(k, lo)).is_finite:
                lo += 1
            S = _partial_sum(F.subs(k, j), j, sp.Integer(lo), k - 1)
            if S is None or S.has(sp.Sum):
                raise NotImplementedError("AsymptoticSum: no closed form for the indefinite sum")
        return _asym_out(_asym_closed(S, k, x0, n, "AsymptoticSum"))
    k, a, b = _sum_spec(rng, "AsymptoticSum")
    if b.has(x) and _is_inf(x0):
        S = _partial_sum(F, k, a, b)
        if S is None or S.has(sp.Sum):
            raise NotImplementedError("AsymptoticSum: no closed form for the partial sum")
        if a.is_Integer and b == x:
            _check_sum_closed(S, F, k, a, x, "AsymptoticSum")
        return _asym_out(_asym_closed(S, x, x0, n, "AsymptoticSum"))
    if not F.has(x):
        raise ValueError("AsymptoticSum: the expansion variable must be the upper limit or a parameter")
    S = _summation(F, k, a, b)
    pv = (x0 + sp.Rational(1, 7)) if not _is_inf(x0) else sp.Integer(7)
    if isinstance(S, sp.Piecewise):
        br = S.args[0]
        S = br[0] if br[1].subs(x, pv) is sp.true else S
    if not a.is_Integer:
        raise NotImplementedError("AsymptoticSum: the lower limit must be an integer")

    def numsum(v):
        return _nsum(F.subs(x, v), k, a, b)
    if not (S.has(sp.Sum) or S.has(sp.Piecewise)):
        try:
            ok = _num_close(numsum(pv), complex(sp.N(S.subs(x, pv), 30)), 1e-6)
        except Exception:
            ok = False
        if ok:
            try:
                return _asym_out(_asym_closed(S, x, x0, n, "AsymptoticSum"))
            except NotImplementedError:
                pass
    if _is_inf(x0):
        raise NotImplementedError("AsymptoticSum: no verified expansion of the parametric sum found")
    res = _termwise_param_sum(F, k, a, b, x, x0, n)
    if res is None:
        raise NotImplementedError("AsymptoticSum: the term-wise sums diverge or have no closed form")
    m = n if n is not None else 3
    try:
        h = sp.Rational(1, 1000)
        err = abs(numsum(x0 + h) - complex(sp.N(res.subs(x, x0 + h), 30)))
    except Exception:
        raise NotImplementedError("AsymptoticSum: the term-wise expansion could not be checked")
    if err > 1e-3 ** (m + 0.5) * 1e3 + 1e-12:
        raise NotImplementedError("AsymptoticSum: the term-wise expansion failed a numerical check")
    if n is None:
        return _asym_out(_lead_finite(res, x, x0))
    return _asym_out(res)


@fn("AsymptoticProduct", area=AREA, status="added",
    notes="AsymptoticProduct[f, {k, a, n}, n -> Infinity | {n, Infinity, m}], AsymptoticProduct[f, {k, a, "
          "Infinity}, {p, p0, m}] and the indefinite AsymptoticProduct[f, k, {k, Infinity, m}] (= Product over "
          "1..k-1): from SymPy's closed form (Gamma / rising factorials, checked numerically) expanded with "
          "Stirling's series (Product[k, {k, 2, n}] ~ Sqrt[2 Pi] n^(n + 1/2) E^-n); no closed form -> error")
def _asymprod(f, rng, spec, *rest):
    pos, o = _opts(rest, {"SeriesTermGoal", "Assumptions", "GenerateConditions"}, "AsymptoticProduct")
    if pos:
        raise ValueError("AsymptoticProduct: too many arguments")
    F = SX(f, "AsymptoticProduct")
    x, x0, n = _eval_point(spec, o, "AsymptoticProduct")
    if isinstance(rng, Symbol):
        k = _svar(rng, "AsymptoticProduct")
        if x != k or x0 != sp.oo:
            raise NotImplementedError("AsymptoticProduct: indefinite products are supported for k -> Infinity only")
        j = sp.Dummy("j", integer=True)
        P = sp.product(F.subs(k, j), (j, 1, k - 1))
        if P.has(sp.Product):
            raise NotImplementedError("AsymptoticProduct: no closed form for the indefinite product")
        return _asym_out(_run_asym(P, k, x0, n, "AsymptoticProduct", True))
    k, a, b = _sum_spec(rng, "AsymptoticProduct")
    P = sp.product(F, (k, a, b))
    if P.has(sp.Product) or P.has(sp.Piecewise):
        raise NotImplementedError("AsymptoticProduct: no closed form for the product")
    if b.has(x) and _is_inf(x0):
        if a.is_Integer and b == x:
            for N in (a + 3, a + 6):
                d = sp.prod([F.subs(k, j) for j in range(int(a), int(N) + 1)])
                if not _num_close(complex(sp.N(P.subs(x, N), 30)), complex(sp.N(d, 30)), 1e-8):
                    raise NotImplementedError("AsymptoticProduct: the closed form failed a numerical check")
        return _asym_out(_run_asym(P, x, x0, n, "AsymptoticProduct", True))
    return _asym_out(_run_asym(P, x, x0, n, "AsymptoticProduct", True))


def _antideriv(F, x):
    G = sp.integrate(F, x)
    if G.has(sp.Integral):
        return None
    if sp.simplify(sp.diff(G, x) - F) != 0:
        return None
    return G


def _num_integral(F, t, a, b):
    f = sp.lambdify(t, F, "mpmath")
    return complex(mpmath.quad(f, [float(a) if a.is_finite else (mpmath.inf if a == sp.oo else -mpmath.inf),
                                   float(b) if b.is_finite else (mpmath.inf if b == sp.oo else -mpmath.inf)]))


@fn("AsymptoticIntegrate", area=AREA, status="added",
    notes="AsymptoticIntegrate[f, x, x -> x0 | {x, x0, n}]: asymptotic expansion of an antiderivative (SymPy's "
          "antiderivative checked by differentiation, else term-wise integration of the expansion of f, constant "
          "0); AsymptoticIntegrate[f, {t, a, b}, {p, p0, n}]: expansion of a parametric integral (exact integral "
          "checked numerically, else term-wise integration of the expansion in p (finite p0, each term finite; the "
          "result is checked against numerical quadrature near p0)); divergent term-wise integrals raise")
def _asymint(f, rng, spec, *rest):
    pos, o = _opts(rest, {"SeriesTermGoal", "Assumptions", "GenerateConditions", "PrincipalValue"},
                   "AsymptoticIntegrate")
    if pos:
        raise ValueError("AsymptoticIntegrate: too many arguments")
    F = SX(f, "AsymptoticIntegrate")
    x, x0, n = _eval_point(spec, o, "AsymptoticIntegrate")
    if isinstance(rng, Symbol):
        t = _svar(rng, "AsymptoticIntegrate")
        if t != x:
            raise ValueError("AsymptoticIntegrate: the expansion variable must be the integration variable")
        G = _antideriv(F, x)
        if G is not None:
            try:
                return _asym_out(asym(G, x, x0, n))
            except _AsymFail:
                pass
        if _is_inf(x0):
            raise NotImplementedError("AsymptoticIntegrate: no antiderivative expansion at Infinity found")
        m = n if n is not None else 6
        try:
            s = sp.series(F, x, x0, m).removeO()
        except Exception as exc:
            raise NotImplementedError(f"AsymptoticIntegrate: cannot expand the integrand ({exc})")
        G = sp.integrate(sp.expand(s), x)
        if G.has(sp.Integral):
            raise NotImplementedError("AsymptoticIntegrate: the term-wise integral failed")
        if n is None:
            return _asym_out(_lead_finite(G, x, x0))
        return _asym_out(sp.series(G, x, x0, n + 1).removeO())
    t, a, b = _sum_spec(rng, "AsymptoticIntegrate")
    if t == x:
        raise ValueError("AsymptoticIntegrate: the expansion variable must be a parameter")
    pv = (x0 + sp.Rational(1, 7)) if not _is_inf(x0) else sp.Integer(7)
    exact = sp.integrate(F, (t, a, b))
    if not exact.has(sp.Integral) and not exact.has(sp.Piecewise):
        try:
            nv = _num_integral(F.subs(x, pv), t, a, b)
            ok = _num_close(complex(sp.N(exact.subs(x, pv), 30)), nv, 1e-6)
        except Exception:
            ok = False
        if ok:
            try:
                return _asym_out(asym(exact, x, x0, n))
            except _AsymFail:
                pass
    if _is_inf(x0):
        raise NotImplementedError("AsymptoticIntegrate: no verified expansion of the parametric integral found")
    m = n if n is not None else 4
    try:
        s = sp.series(F, x, x0, m + 1).removeO()
    except Exception as exc:
        raise NotImplementedError(f"AsymptoticIntegrate: cannot expand the integrand ({exc})")
    res = 0
    for p in range(0, m + 1):
        c = sp.expand(s).coeff(x - x0 if x0 != 0 else x, p) if p else s.subs(x, x0)
        if c == 0:
            continue
        I_ = sp.integrate(c, (t, a, b))
        if I_.has(sp.Integral) or not I_.is_finite:
            raise NotImplementedError("AsymptoticIntegrate: a term-wise integral diverges or has no closed form "
                                      "(non-uniform expansion)")
        res += I_ * (x - x0) ** p
    # numerical check near x0
    errs = []
    for h in (sp.Rational(1, 100), sp.Rational(1, 1000)):
        try:
            nv = _num_integral(F.subs(x, x0 + h), t, a, b)
            errs.append(abs(complex(sp.N(res.subs(x, x0 + h), 30)) - nv))
        except Exception:
            raise NotImplementedError("AsymptoticIntegrate: the term-wise expansion could not be checked")
    if errs[1] > 1e-3 ** (m + 0.5) * 1e3 + 1e-12:
        raise NotImplementedError("AsymptoticIntegrate: the term-wise expansion failed a numerical check")
    if n is None:
        return _asym_out(_lead_finite(res, x, x0))
    return _asym_out(res)


def _texp(term, t):
    c, pw = term.as_independent(t, as_Add=False)
    if pw == 1:
        return 0
    if pw == t:
        return 1
    if isinstance(pw, sp.Pow) and pw.base == t and pw.exp.is_number:
        return pw.exp
    return -sp.oo


def _branches(F, y, x, x0, n, fname):
    try:
        roots = sp.solve(F, y)
    except Exception:
        roots = []
    if not roots:
        raise NotImplementedError(f"{fname}: the branches could not be found")
    res = []
    for r in roots:
        try:
            if _is_inf(x0):
                v = asym(r, x, x0, n)
            else:
                v = sp.series(r, x, x0, (n if n is not None else 0) + 1).removeO() if n is not None \
                    else _lead_finite(r, x, x0)
                if n is not None:
                    t_ = sp.Dummy("t")
                    v = sp.Add(*[term for term in sp.Add.make_args(sp.expand(v.subs(x, x0 + t_)))
                                 if not (sp.Poly(term.subs(t_, t_), t_) if False else False)
                                 and _texp(term, t_) <= n]).subs(t_, x - x0)
        except Exception:
            continue
        res.append(sp.expand(v))
    return res


@fn("AsymptoticSolve", area=AREA, status="added",
    notes="AsymptoticSolve[eqn, {y, y0}, {x, x0, n}]: the branch of y(x) through (x0, y0) as a Taylor series (implicit "
          "function theorem, requires D[eqn, y] != 0 there; Newton iteration on truncated series, residual checked); "
          "AsymptoticSolve[eqn, y | {y}, {x, x0, n} | x -> x0]: every branch, from SymPy's closed-form roots "
          "expanded in Puiseux series; systems raise")
def _asymsolve(eqn, ys, spec, *rest):
    pos, o = _opts(rest, {"SeriesTermGoal", "Assumptions", "GenerateConditions"}, "AsymptoticSolve")
    if pos and not (len(pos) == 1 and isinstance(pos[0], Symbol) and pos[0].name in ("Reals", "Complexes")):
        raise ValueError("AsymptoticSolve: too many arguments")
    dom = pos[0].name if pos else "Complexes"
    if _head(eqn, "And") or is_list(eqn):
        raise NotImplementedError("AsymptoticSolve: systems of equations are not supported")
    E_ = SX(eqn, "AsymptoticSolve")
    F = E_.lhs - E_.rhs if isinstance(E_, sp.Equality) else E_
    x, x0, n = _eval_point(spec, o, "AsymptoticSolve")
    if is_list(ys) and len(ys.args) == 2 and not is_list(ys.args[0]):
        y = _svar(ys.args[0], "AsymptoticSolve")
        y0 = SX(ys.args[1], "AsymptoticSolve")
        if _is_inf(x0):
            raise NotImplementedError("AsymptoticSolve: branch selection at Infinity is not supported")
        if sp.simplify(F.subs({x: x0, y: y0})) != 0:
            raise ValueError("AsymptoticSolve: the point {x0, y0} does not satisfy the equation")
        Fy = sp.simplify(sp.diff(F, y).subs({x: x0, y: y0}))
        if Fy == 0:
            raise NotImplementedError("AsymptoticSolve: D[eqn, y] vanishes at the point (singular branch point)")
        m = n if n is not None else 6
        Y = y0
        for _ in range(m + 1):
            r = sp.series(F.subs(y, Y), x, x0, m + 1).removeO()
            Y = sp.expand(Y - r / Fy)
            Y = sp.series(Y, x, x0, m + 1).removeO() if Y.has(x) else Y
        chk = sp.series(F.subs(y, Y), x, x0, m + 1).removeO()
        if sp.simplify(chk) != 0:
            raise NotImplementedError("AsymptoticSolve: the series did not converge")
        if n is None:
            Y = Y - y0
            Y = y0 + (_lead_finite(Y, x, x0) if Y != 0 else 0)
        return L(L(rule(Symbol(y.name), _asym_out(Y))))
    if is_list(ys):
        if len(ys.args) != 1:
            raise NotImplementedError("AsymptoticSolve: only one unknown is supported")
        ys = ys.args[0]
    y = _svar(ys, "AsymptoticSolve")
    brs = _branches(F, y, x, x0, n, "AsymptoticSolve")
    if dom == "Reals":
        brs = [b for b in brs if not b.has(sp.I)]
    return L(*(L(rule(Symbol(y.name), _asym_out(b))) for b in brs))


def _flat_eqs(e):
    if is_list(e):
        r = []
        for a in e.args:
            r.extend(_flat_eqs(a))
        return r
    if _head(e, "And"):
        r = []
        for a in e.args:
            r.extend(_flat_eqs(a))
        return r
    return [e]


def _fname_of(t):
    """'y' from y, y[x] or y[n]."""
    if isinstance(t, Symbol):
        return t.name
    if _head(t) and len(t.args) == 1:
        return t.head
    raise ValueError(f"{t!r} is not an unknown function")


def _ode_parts(eqs, yname, x, fname):
    """(ODE as SymPy expression == 0, order, {k: value} initial conditions at points) for one unknown."""
    yf = sp.Function(yname)
    odes, ics = [], []
    for e in eqs:
        v = SX(e, fname)
        if not isinstance(v, sp.Equality):
            raise ValueError(f"{fname}: {e!r} is not an equation")
        d = v.lhs - v.rhs
        if d.has(yf(x)):
            odes.append(d)
        else:
            ics.append(v)
    if len(odes) != 1:
        raise NotImplementedError(f"{fname}: exactly one differential equation in {yname} is supported")
    ode = odes[0]
    order = 0
    for dv in ode.atoms(sp.Derivative):
        if dv.expr == yf(x):
            order = max(order, dv.derivative_count)
    if order == 0:
        raise ValueError(f"{fname}: the equation is not a differential equation")
    icd = {}
    for v in ics:
        lhs, rhs = v.lhs, v.rhs
        if rhs.has(yf) and not lhs.has(yf):
            lhs, rhs = rhs, lhs
        if isinstance(lhs, AppliedUndef) and lhs.func == yf:
            icd[(0, lhs.args[0])] = rhs
            continue
        if isinstance(lhs, sp.Subs) and isinstance(lhs.expr, sp.Derivative):
            icd[(lhs.expr.derivative_count, lhs.point[0])] = rhs
            continue
        if isinstance(lhs, sp.Derivative) and isinstance(lhs.expr, AppliedUndef):
            icd[(lhs.derivative_count, lhs.expr.args[0])] = rhs
            continue
        raise ValueError(f"{fname}: cannot interpret the condition {v}")
    return ode, order, icd


def _taylor_ode(ode, yname, x, order, x0, vals, N, fname):
    yf = sp.Function(yname)
    Ys = [sp.Symbol(f"__Y{i}") for i in range(order + 1)]
    e = ode
    for i in range(order, 0, -1):
        e = e.subs(sp.Derivative(yf(x), (x, i)), Ys[i])
    e = e.subs(yf(x), Ys[0])
    if e.has(yf):
        raise NotImplementedError(f"{fname}: delay or non-local terms are not supported")
    sol = sp.solve(e, Ys[order], dict=True)
    if len(sol) != 1:
        raise NotImplementedError(f"{fname}: the equation must determine the highest derivative uniquely")
    G = sp.cancel(sol[0][Ys[order]])
    point = {x: x0, **{Ys[i]: vals[i] for i in range(order)}}
    den = sp.fraction(sp.together(G))[1]
    g0 = sp.simplify(G.subs(point))
    if sp.simplify(den.subs(point)) == 0 or g0.has(sp.zoo, sp.oo, sp.nan) or not g0.is_finite:
        raise NotImplementedError(f"{fname}: {x0} is a singular point of the equation")
    derivs = list(vals)
    cur = G
    for k in range(order, N + 1):
        derivs.append(sp.simplify(cur.subs(point)))
        nxt = sp.diff(cur, x) + sum(Ys[i + 1] * sp.diff(cur, Ys[i]) for i in range(order - 1)) + \
            G * sp.diff(cur, Ys[order - 1])
        cur = sp.cancel(nxt) if k < N else cur
    return sum(derivs[k] / sp.factorial(k) * (x - x0) ** k for k in range(N + 1))


def _perturb_ode(ode, yname, x, order, icd, eps, e0, n, fname):
    if e0 != 0:
        raise NotImplementedError(f"{fname}: perturbation expansions are supported about 0 only")
    yf = sp.Function(yname)
    lead = [d for d in ode.atoms(sp.Derivative) if d.expr == yf(x) and d.derivative_count == order]
    hc = sp.diff(ode.subs(lead[0], sp.Symbol("__H")), sp.Symbol("__H")).subs(sp.Symbol("__H"), 0)
    if sp.simplify(hc.subs(eps, 0)) == 0:
        raise NotImplementedError(f"{fname}: singular perturbation (the small parameter multiplies the highest "
                                  "derivative); boundary-layer expansions are not supported")
    Yk = [sp.Function(f"__y{k}") for k in range(n + 1)]
    Ysum = sum(eps ** k * Yk[k](x) for k in range(n + 1))
    full = sp.expand(ode.subs(yf(x), Ysum).doit())
    ser = sp.series(full, eps, 0, n + 1).removeO()
    sols = {}
    for k in range(n + 1):
        eqk = sp.expand(ser).coeff(eps, k)
        for j in range(k):
            eqk = eqk.subs(Yk[j](x), sols[j]).doit()
        conds = {}
        for (d, pt), v in icd.items():
            vk = sp.series(v, eps, 0, n + 1).removeO()
            vk = sp.expand(vk).coeff(eps, k)
            if d == 0:
                conds[Yk[k](pt)] = vk
            else:
                conds[Yk[k](x).diff(x, d).subs(x, pt)] = vk
        try:
            s_ = sp.dsolve(sp.Eq(sp.simplify(eqk), 0), Yk[k](x), ics=conds)
        except Exception as exc:
            raise NotImplementedError(f"{fname}: could not solve the order-{k} problem ({exc})")
        if isinstance(s_, list):
            if len(s_) != 1:
                raise NotImplementedError(f"{fname}: the order-{k} problem has several solutions")
            s_ = s_[0]
        sols[k] = sp.simplify(s_.rhs)
        if sols[k].free_symbols & {sp.Symbol(f"C{i}") for i in range(1, 6)}:
            raise NotImplementedError(f"{fname}: give enough initial/boundary conditions to fix the solution")
    res = sum(eps ** k * sols[k] for k in range(n + 1))
    chk = sp.series(sp.expand(ode.subs(yf(x), res).doit()), eps, 0, n + 1).removeO()
    if sp.simplify(chk) != 0:
        raise NotImplementedError(f"{fname}: the perturbation series failed the residual check")
    return res


@fn("AsymptoticDSolveValue", area=AREA, status="added",
    notes="AsymptoticDSolveValue[{ode, ics}, y[x], {x, x0, n} | x -> x0]: the power-series solution of a (linear or "
          "nonlinear) ODE at an ordinary point x0 (Taylor coefficients by repeated differentiation of the solved "
          "equation; missing initial values become C[1], C[2], ...); AsymptoticDSolveValue[{ode, ics}, y[x], x, "
          "{e, 0, n}]: regular perturbation expansion in a parameter (each order solved with DSolve, residual "
          "checked); singular points, singular perturbations and systems raise")
def _asymdsolve(eqs, target, spec, *rest):
    pos, o = _opts(rest, {"SeriesTermGoal", "Assumptions", "GenerateConditions"}, "AsymptoticDSolveValue")
    fname = "AsymptoticDSolveValue"
    yname = _fname_of(target)
    eqlist = _flat_eqs(eqs)
    if pos:
        if len(pos) != 1:
            raise ValueError(f"{fname}: too many arguments")
        x = _svar(spec, fname)
        eps, e0, n = _parse_point_spec(pos[0], o, fname)
        ode, order, icd = _ode_parts(eqlist, yname, x, fname)
        res = _perturb_ode(ode, yname, x, order, icd, eps, e0, n if n is not None else 1, fname)
        return _asym_out(res)
    x, x0, n = _parse_point_spec(spec, o, fname)
    if _is_inf(x0):
        raise NotImplementedError(f"{fname}: expansions at Infinity are not supported")
    ode, order, icd = _ode_parts(eqlist, yname, x, fname)
    vals = []
    cno = 0
    for k in range(order):
        key = next((kk for kk in icd if kk[0] == k), None)
        if key is not None and sp.simplify(key[1] - x0) != 0:
            raise NotImplementedError(f"{fname}: initial conditions must be given at the expansion point")
        if key is None:
            cno += 1
            vals.append(sp.Symbol(f"__C_{cno}"))
        else:
            vals.append(icd[key])
    if any(k[0] >= order for k in icd):
        raise NotImplementedError(f"{fname}: boundary-value problems are not supported for series solutions")
    N = n if n is not None else 12
    ser = _taylor_ode(ode, yname, x, order, x0, vals, N, fname)
    if n is None:
        if ser == 0:
            return 0
        ser = _lead_finite(sp.expand(ser), x, x0) if not _csyms(ser) else sp.expand(ser)
    return _sym_to_c(_asym_out(sp.expand(ser)))


def _first_order_linear(eq, yname, nv):
    """y[n+1] == r(n) y[n] -> C[1] Product[r(k), {k, 1, n - 1}] in closed form (verified), else None."""
    yf = sp.Function(yname)
    d = eq.lhs - eq.rhs
    terms = {a for a in d.atoms(AppliedUndef) if a.func == yf}
    if len(terms) != 2:
        return None
    shifts = sorted(terms, key=lambda a: sp.simplify(a.args[0] - nv))
    lo, hi = shifts
    s0 = sp.simplify(lo.args[0] - nv)
    if sp.simplify(hi.args[0] - lo.args[0]) != 1 or not s0.is_Integer:
        return None
    A, B = sp.Symbol("__A"), sp.Symbol("__B")
    lin = d.subs({hi: B, lo: A})
    if sp.diff(lin, A, 2) != 0 or sp.diff(lin, B, 2) != 0 or sp.diff(lin, A, B) != 0 or lin.subs({A: 0, B: 0}) != 0:
        return None
    cA, cB = sp.diff(lin, A), sp.diff(lin, B)
    r = sp.cancel(-cA / cB).subs(nv, nv - s0)       # y(m + 1) = r(m) y(m)
    k = sp.Dummy("k", integer=True, positive=True)
    P = sp.product(r.subs(nv, k), (k, 1, nv - 1))
    if P.has(sp.Product) or P == 0:
        return None
    C1 = sp.Symbol("__C_1")
    P = _rewrite_special(P)
    sol = C1 * P
    try:
        chk = sp.simplify(sp.gammasimp(d.subs({hi: sol.subs(nv, hi.args[0]), lo: sol.subs(nv, lo.args[0])})))
    except Exception:
        chk = None
    if chk != 0:
        for m in (3, 5, 8):
            try:
                if abs(complex(sp.N(d.subs({hi: sol.subs(nv, hi.args[0]), lo: sol.subs(nv, lo.args[0])})
                                    .subs({nv: m, C1: 1}), 30))) > 1e-20:
                    return None
            except Exception:
                return None
    return sol


@fn("AsymptoticRSolveValue", area=AREA, status="added",
    notes="AsymptoticRSolveValue[{eqn, ics}, y[n], n -> Infinity | {n, Infinity, m}]: asymptotics of the solution "
          "of a recurrence: first-order linear homogeneous recurrences are solved as C[1] times a closed-form "
          "product (Gamma functions, verified); others via RSolveValue (verified by substitution); each free "
          "constant's part is expanded separately (Stirling / exponential scales; Fibonacci -> GoldenRatio^n/Sqrt[5]); "
          "parameter (perturbation) forms and recurrences without a verified closed form raise")
def _asymrsolve(eqs, target, spec, *rest):
    pos, o = _opts(rest, {"SeriesTermGoal", "Assumptions", "GenerateConditions"}, "AsymptoticRSolveValue")
    fname = "AsymptoticRSolveValue"
    if pos:
        raise NotImplementedError(f"{fname}: perturbation expansions in a parameter are not supported")
    x, x0, n = _parse_point_spec(spec, o, fname)
    if x0 != sp.oo:
        raise ValueError(f"{fname}: the expansion point must be Infinity")
    yname = _fname_of(target)
    eqlist = _flat_eqs(eqs)
    sol = None
    if len(eqlist) == 1:
        v = SX(eqlist[0], fname)
        if isinstance(v, sp.Equality):
            sol = _first_order_linear(v, yname, x)
    if sol is None:
        r = _ev(Expr("RSolveValue", L(*eqlist), Expr(yname, Symbol(x.name)), Symbol(x.name)))
        if _head(r, "RSolveValue"):
            raise NotImplementedError(f"{fname}: the recurrence could not be solved")
        if K.contains(r, lambda e: _head(e, "Pochhammer") and e.args and e.args[0] == 0):
            raise NotImplementedError(f"{fname}: the recurrence could not be solved")
        sol = SX(_c_to_sym(r), fname)
        if sol == 0 and len(eqlist) == 1:
            raise NotImplementedError(f"{fname}: only the trivial solution was found")
        _check_rsol(sol, eqlist, yname, x, fname)
    res = _asym_with_constants(sol, x, x0, n, fname, discrete=True)
    return _sym_to_c(_asym_out(res))


def _check_rsol(sol, eqlist, yname, x, fname):
    """The solution must satisfy the recurrence (numerically, at several n) and carry order - #ics free constants."""
    yf = sp.Function(yname)
    rec, ics = [], []
    for e in eqlist:
        v = SX(e, fname)
        if not isinstance(v, sp.Equality):
            raise ValueError(f"{fname}: equations expected")
        apps = [a for a in v.atoms(sp.Function) if a.func == yf]
        if any(a.args[0].has(x) for a in apps):
            rec.append((v, apps))
        else:
            ics.append(v)
    if len(rec) != 1:
        raise NotImplementedError(f"{fname}: exactly one recurrence equation is supported")
    v, apps = rec[0]
    shifts = [sp.simplify(a.args[0] - x) for a in apps]
    if not all(sh.is_integer for sh in shifts):
        raise NotImplementedError(f"{fname}: only shifts by integers are supported")
    order = int(max(shifts) - min(shifts))
    consts = [c for c in sol.free_symbols if c.name.startswith("__C_")]
    if len(consts) != order - len(ics):
        raise NotImplementedError(f"{fname}: the solver did not return the general solution "
                                  f"({len(consts)} of {order - len(ics)} free constants)")
    d = v.lhs - v.rhs
    cvals = {c: sp.Rational(k + 2, 3) for k, c in enumerate(consts)}
    for nv in (7, 11, 16):
        val = d.subs({a: sol.subs(x, a.args[0]) for a in apps}).subs(x, nv).subs(cvals)
        try:
            z = complex(sp.N(val, 30))
            sc = max(1.0, abs(complex(sp.N(sol.subs(x, nv).subs(cvals), 30))))
        except Exception:
            raise NotImplementedError(f"{fname}: the solution could not be verified")
        if abs(z) > 1e-12 * sc * 10 ** order:
            raise NotImplementedError(f"{fname}: the solver's solution failed verification")


def _pick_piecewise(r, xs, pv):
    """Replace Piecewise[...] sub-expressions by the branch whose condition holds at xs = pv (near the point)."""
    def walk(e):
        if _head(e, "Piecewise"):
            for br in e.args[0].args:
                c = _ev(K.substitute(br.args[1], {xs.name: pv}))
                if c is True:
                    return walk(br.args[0])
            return walk(e.args[1]) if len(e.args) > 1 else Symbol("Indeterminate")
        if _head(e):
            return Expr(e.head, *(walk(a) for a in e.args))
        return e
    return _ev(walk(r))


def _dist_arg(d, fname):
    if not (_head(d, "Distributed") and len(d.args) == 2):
        raise ValueError(f"{fname}: the second argument must be x \\[Distributed] dist")
    return d


@fn("AsymptoticExpectation", area=AREA, status="added",
    notes="AsymptoticExpectation[expr, x \\[Distributed] dist, a -> a0 | {a, a0, n}]: the asymptotic expansion in a "
          "parameter of the exact Expectation (computed by Expectation, then expanded); raises when the "
          "expectation has no closed form")
def _asymexp(expr, d, spec, *rest):
    pos, o = _opts(rest, {"SeriesTermGoal", "Assumptions"}, "AsymptoticExpectation")
    if pos:
        raise ValueError("AsymptoticExpectation: too many arguments")
    _dist_arg(d, "AsymptoticExpectation")
    x, x0, n = _parse_point_spec(spec, o, "AsymptoticExpectation")
    r = _ev(Expr("Expectation", expr, d))
    if _head(r, "Expectation") or K.contains(r, lambda e: _head(e, "Integrate") or _head(e, "Sum")):
        raise NotImplementedError("AsymptoticExpectation: the expectation has no closed form")
    R_ = SX(r, "AsymptoticExpectation")
    try:
        return _asym_out(asym(R_, x, x0, n))
    except _AsymFail:
        pass
    # term-wise: expand the integrand in the parameter, take expectations of the coefficients (moments)
    m = n if n is not None else 2
    E_ = SX(expr, "AsymptoticExpectation")
    try:
        if _is_inf(x0):
            w = sp.Dummy("w", positive=True)
            ser = sp.series(E_.subs(x, 1 / w), w, 0, m + 1).removeO()
        else:
            w = sp.Dummy("w")
            ser = sp.series(E_.subs(x, x0 + w), w, 0, m + 1).removeO()
    except Exception as exc:
        raise NotImplementedError(f"AsymptoticExpectation: cannot expand the expression ({exc})")
    tot = 0
    ser = sp.expand(ser)
    for p in range(m + 1):
        c = ser.coeff(w, p)
        if c == 0:
            continue
        mv = _ev(Expr("Expectation", out(c), d))
        if _head(mv, "Expectation"):
            raise NotImplementedError("AsymptoticExpectation: a term-wise expectation has no closed form")
        tot += SX(mv, "AsymptoticExpectation") * w ** p
    approx = tot.subs(w, 1 / x if _is_inf(x0) else x - x0)
    pv = sp.Integer(1000) if x0 == sp.oo else (sp.Integer(-1000) if x0 == -sp.oo else x0 + sp.Rational(1, 1000))
    try:
        err = abs(complex(sp.N(R_.subs(x, pv), 30)) - complex(sp.N(approx.subs(x, pv), 30)))
    except Exception:
        raise NotImplementedError("AsymptoticExpectation: the term-wise expansion could not be checked")
    if err > 1e-3 ** (m + 0.5) * 1e3 * max(1, abs(complex(sp.N(R_.subs(x, pv), 20)))):
        raise NotImplementedError("AsymptoticExpectation: the term-wise expansion failed a numerical check")
    return _asym_out(approx)


def _prob_closed(pred, d, fname):
    r = _ev(Expr("Probability", pred, d))
    if not _head(r, "Probability"):
        return r
    xv, dist = d.args
    if not (_head(pred) and pred.head in ("Greater", "Less", "GreaterEqual", "LessEqual") and len(pred.args) == 2):
        raise NotImplementedError(f"{fname}: only predicates x > c, x < c, x >= c, x <= c are supported")
    lhs, rhs = pred.args
    h = pred.head
    if rhs == xv:
        lhs, rhs = rhs, lhs
        h = {"Greater": "Less", "Less": "Greater", "GreaterEqual": "LessEqual", "LessEqual": "GreaterEqual"}[h]
    if lhs != xv:
        raise NotImplementedError(f"{fname}: the predicate must compare the random variable with a bound")
    cdf = _ev(Expr("CDF", dist, rhs))
    if _head(cdf, "CDF"):
        raise NotImplementedError(f"{fname}: the CDF has no closed form")
    pdfp = _ev(Expr("PDF", dist, rhs))
    if h in ("Less", "LessEqual"):
        return cdf
    return _ev(Expr("Plus", 1, Expr("Times", -1, cdf))) if not _head(pdfp, "PDF") else \
        _ev(Expr("Plus", 1, Expr("Times", -1, cdf)))


@fn("AsymptoticProbability", area=AREA, status="added",
    notes="AsymptoticProbability[pred, x \\[Distributed] dist, a -> a0 | {a, a0, n}]: the asymptotic expansion in a "
          "parameter of the exact probability (Probability, or 1 - CDF / CDF for one-sided bounds of continuous "
          "distributions); e.g. the Gaussian tail Erfc[a/Sqrt[2]]/2 ~ E^(-a^2/2)/(a Sqrt[2 Pi]); no closed form -> "
          "error")
def _asymprob(pred, d, spec, *rest):
    pos, o = _opts(rest, {"SeriesTermGoal", "Assumptions"}, "AsymptoticProbability")
    if pos:
        raise ValueError("AsymptoticProbability: too many arguments")
    _dist_arg(d, "AsymptoticProbability")
    x, x0, n = _parse_point_spec(spec, o, "AsymptoticProbability")
    r = _prob_closed(pred, d, "AsymptoticProbability")
    pv = (x0 + sp.Rational(1, 7)) if not _is_inf(x0) else (sp.Integer(10) if x0 == sp.oo else sp.Integer(-10))
    r = _pick_piecewise(r, Symbol(x.name), R(pv))
    v = sp.simplify(SX(r, "AsymptoticProbability"))
    return _asym_out(_run_asym(v, x, x0, n, "AsymptoticProbability"))


# ───────────────────────────── discrete calculus ─────────────────────────────

def _dq_spec(spec, fname):
    if not (is_list(spec) and len(spec.args) in (2, 3)):
        raise ValueError(f"{fname}: each specification must be {{x, h}} or {{x, n, h}}")
    x = _svar(spec.args[0], fname)
    if len(spec.args) == 2:
        return x, 1, SX(spec.args[1], fname)
    return x, _num_opt(spec.args[1], fname, "the order"), SX(spec.args[2], fname)


@fn("DifferenceQuotient", area=AREA, status="added",
    notes="DifferenceQuotient[f, {x, h}] = (f[x + h] - f[x])/h; {x, n, h}: the n-th forward difference over h^n; "
          "several specifications apply in turn; threads over lists")
def _diffquot(f, *specs):
    if not specs:
        raise ValueError("DifferenceQuotient: a specification {x, h} is required")
    if is_list(f):
        return L(*(_diffquot(e, *specs) for e in f.args))
    e = SX(f, "DifferenceQuotient")
    for sp_ in specs:
        x, n, h = _dq_spec(sp_, "DifferenceQuotient")
        if h == 0:
            raise ValueError("DifferenceQuotient: the step must be non-zero")
        e = sp.Add(*[(-1) ** (n - k) * sp.binomial(n, k) * e.subs(x, x + k * h) for k in range(n + 1)]) / h ** n
        e = sp.cancel(sp.together(e))
    return out(e)


@fn("Casoratian", area=AREA, status="added",
    notes="Casoratian[{y1, ..., yk}, n] = Det[{y_j[n + i]}] (i = 0..k-1), simplified; Casoratian[..., n, True] gives "
          "the matrix; the equation form Casoratian[eqn, y, n] raises (its normalisation constant is not determined "
          "by the equation)")
def _casoratian(ys, n, *rest):
    if not is_list(ys):
        raise NotImplementedError("Casoratian: the equation form (Casoratian[eqn, y, n]) is not supported; "
                                  "give an explicit list of solutions")
    if len(rest) > 1 or (rest and not isinstance(rest[0], bool)):
        raise ValueError("Casoratian: the optional third argument must be True or False")
    nv = _svar(n, "Casoratian")
    fs0 = [SX(y, "Casoratian") for y in ys.args]
    special = any(g.has(sp.fibonacci, sp.lucas) for g in fs0)
    fs = [_rewrite_special(g) if special else g for g in fs0]
    if not fs:
        raise ValueError("Casoratian: at least one function is required")
    k = len(fs)
    M = sp.Matrix(k, k, lambda i, j: fs[j].subs(nv, nv + i))
    if rest and rest[0] is True:
        return out(M.tolist())
    d = M.det(method="berkowitz")
    d = sp.simplify(sp.powsimp(sp.expand(d, power_base=False), force=True))
    d = sp.powsimp(sp.factor(d), combine="base", force=True) if not d.has(AppliedUndef) else sp.expand(d)
    if d.has(sp.gamma) and not any(g.has(sp.gamma) for g in fs0):
        d = d.replace(lambda e: isinstance(e, sp.gamma) and (e.args[0] - nv).is_integer is True,
                      lambda e: sp.factorial(e.args[0] - 1))
    def _primes(e):
        def f_(b, ex):
            if b.is_Integer and b < 0:
                return (-1) ** ex * f_(-b, ex)
            if b.is_Integer and b > 1:
                fa = sp.factorint(b)
                if len(fa) == 1:
                    (pr, k), = fa.items()
                    return sp.Pow(pr, k * ex)
            return sp.Pow(b, ex)
        return e.replace(lambda t: isinstance(t, sp.Pow) and t.base.is_Integer and not t.exp.is_number,
                         lambda t: f_(t.base, t.exp))
    d2 = sp.powsimp(sp.expand(_primes(d)), force=True)
    cands = [d, d2, sp.powsimp(sp.expand_power_base(d, force=True), force=True)]
    cands.append(sp.powsimp(sp.expand_power_base(cands[-1], force=True), combine="exp", force=True))
    d = min(cands, key=sp.count_ops)
    return out(d)


def _rec_parts(fun, fname, kind):
    """Function[{y, n}, {eqs...}] -> (yname, nsym, equations as SymPy Eq)."""
    if not (_head(fun, "Function") and len(fun.args) == 2 and is_list(fun.args[0]) and len(fun.args[0].args) == 2):
        raise ValueError(f"{fname}: the argument must be Function[{{y, {kind}}}, {{eqns}}]")
    yv, nvar = fun.args[0].args
    if not (isinstance(yv, Symbol) and isinstance(nvar, Symbol)):
        raise ValueError(f"{fname}: the formal parameters must be symbols")
    body = fun.args[1]
    eqs = list(body.args) if is_list(body) else [body]
    out_ = []
    for e in eqs:
        v = SX(e, fname)
        if not isinstance(v, sp.Equality):
            raise ValueError(f"{fname}: the body must be a list of equations")
        out_.append(v)
    return yv.name, sp.Symbol(nvar.name), out_


def _rec_values(fun, upto, fname="DifferenceRoot"):
    """Exact values of a DifferenceRoot sequence: dict index -> value, computed from the ICs up to `upto`."""
    yname, nv, eqs = _rec_parts(fun, fname, "n")
    yf = sp.Function(yname)
    rec, ic = None, {}
    for v in eqs:
        apps = [a for a in v.atoms(AppliedUndef) if a.func == yf]
        if any(a.args[0].has(nv) for a in apps):
            if rec is not None:
                raise ValueError(f"{fname}: exactly one recurrence equation is required")
            rec = (v.lhs - v.rhs, apps)
        else:
            if not (isinstance(v.lhs, AppliedUndef) and v.lhs.func == yf and v.lhs.args[0].is_integer
                    and v.lhs.args[0].is_number):
                raise ValueError(f"{fname}: initial conditions must be y[k] == value")
            ic[int(v.lhs.args[0])] = v.rhs
    if rec is None:
        raise ValueError(f"{fname}: no recurrence equation")
    d, apps = rec
    shifts = sorted({sp.simplify(a.args[0] - nv) for a in apps})
    if not all(s.is_integer for s in shifts):
        raise ValueError(f"{fname}: shifts must be integers")
    smax, smin = int(max(shifts)), int(min(shifts))
    Ys = {s: sp.Symbol(f"__Y{s - smin}") for s in range(smin, smax + 1)}
    dd = d.xreplace({a: Ys[int(sp.simplify(a.args[0] - nv))] for a in apps})
    if not all(sp.Poly(dd, *Ys.values()).total_degree() <= 1 for _ in [0]):
        raise NotImplementedError(f"{fname}: only linear recurrences are supported")
    lead = sp.Symbol("__LEAD")
    sol = sp.solve(dd, Ys[smax], dict=True)
    if len(sol) != 1:
        raise NotImplementedError(f"{fname}: the recurrence cannot be solved for the highest shift")
    nxt = sol[0][Ys[smax]]
    vals = dict(ic)
    if not vals:
        raise ValueError(f"{fname}: initial conditions are required")
    start = min(vals)
    order = smax - smin
    if any(start + i not in vals for i in range(order)):
        raise ValueError(f"{fname}: {order} consecutive initial values are required")
    m = start + order
    while m <= upto:
        if m not in vals:
            base = m - smax
            sub = {Ys[s]: vals[base + s] for s in range(smin, smax) if base + s in vals}
            if len(sub) != order:
                raise ValueError(f"{fname}: insufficient initial values")
            v = sp.simplify(nxt.subs(nv, base).subs(sub))
            if v.has(sp.zoo, sp.nan, sp.oo):
                raise ValueError(f"{fname}: the recurrence is singular at {base}")
            vals[m] = v
        m += 1
    del lead
    return vals


def _droot_hook(head, args, ev):
    if len(head.args) != 1 or len(args) != 1:
        return K.UNHANDLED
    k = args[0]
    if is_list(k):
        return L(*(_droot_hook(head, [a], ev) for a in k.args))
    if not K.is_int(k):
        return K.UNHANDLED
    kk = int(k)
    vals = _rec_values(head.args[0], kk)
    if kk not in vals:
        return K.UNHANDLED
    return out(vals[kk])


def _ode_from_function(fun, fname):
    yname, xv, eqs = _rec_parts(fun, fname, "x")
    yf = sp.Function(yname)
    ode, ics = None, []
    for v in eqs:
        if v.has(yf(xv)):
            if ode is not None:
                raise ValueError(f"{fname}: exactly one differential equation is required")
            ode = v.lhs - v.rhs
        else:
            ics.append(v)
    if ode is None:
        raise ValueError(f"{fname}: no differential equation")
    order = max([d.derivative_count for d in ode.atoms(sp.Derivative)] or [0])
    Ys = [sp.Symbol(f"__D{i}") for i in range(order + 1)]
    rep = {sp.Derivative(yf(xv), (xv, i)): Ys[i] for i in range(order, 0, -1)}
    dd = ode.subs(rep).subs(yf(xv), Ys[0])
    sol = sp.solve(dd, Ys[order], dict=True)
    if len(sol) != 1:
        raise NotImplementedError(f"{fname}: the equation cannot be solved for the highest derivative")
    G = sol[0][Ys[order]]
    x0 = None
    vals = [None] * order
    for v in ics:
        lhs = v.lhs
        if isinstance(lhs, AppliedUndef) and lhs.func == yf:
            i, p = 0, lhs.args[0]
        elif isinstance(lhs, sp.Subs) and isinstance(lhs.expr, sp.Derivative):
            i, p = lhs.expr.derivative_count, lhs.point[0]
        else:
            raise ValueError(f"{fname}: initial conditions must be y[x0] == v, y'[x0] == v, ...")
        if x0 is not None and p != x0:
            raise NotImplementedError(f"{fname}: all initial conditions must be at the same point")
        x0 = p
        if i >= order:
            raise ValueError(f"{fname}: too many initial conditions")
        vals[i] = v.rhs
    if any(v is None for v in vals):
        raise ValueError(f"{fname}: {order} initial conditions are required")
    return yname, xv, order, Ys, G, x0, vals, ode


def _diffroot_hook(head, args, ev):
    if len(head.args) != 1 or len(args) != 1:
        return K.UNHANDLED
    t = args[0]
    if is_list(t):
        return L(*(_diffroot_hook(head, [a], ev) for a in t.args))
    yname, xv, order, Ys, G, x0, vals, _ode = _ode_from_function(head.args[0], "DifferentialRoot")
    tv = SX(t, "DifferentialRoot")
    if tv == x0:
        return out(vals[0])
    if not (isinstance(t, float) or (tv.is_number and tv.has(sp.Float))):
        return K.UNHANDLED
    tf = float(sp.re(tv))
    f = sp.lambdify([xv] + Ys[:order], G, "mpmath")
    mpmath.mp.dps = 20
    try:
        sol = mpmath.odefun(lambda xx, yy: list(yy[1:]) + [f(xx, *yy)], float(x0), [complex(sp.N(v)) for v in vals]
                            if any(sp.N(v).has(sp.I) for v in vals) else [float(sp.N(v)) for v in vals])
        r = sol(tf)[0]
    except Exception as exc:
        raise ValueError(f"DifferentialRoot: numerical integration failed ({exc})")
    finally:
        mpmath.mp.dps = 15
    r = complex(r)
    return float(r.real) if abs(r.imag) < 1e-14 else R(r)


KL.COMPOUND_HEAD_HOOKS["DifferenceRoot"] = _droot_hook
KL.COMPOUND_HEAD_HOOKS["DifferentialRoot"] = _diffroot_hook
record("DifferenceRoot", AREA, "added",
       "DifferenceRoot[Function[{y, n}, {rec, ics}]][k]: the solution of a linear recurrence with initial values; "
       "evaluated exactly at integers (iterating the recurrence), unevaluated otherwise")
record("DifferentialRoot", AREA, "added",
       "DifferentialRoot[Function[{y, x}, {ode, ics}]][x]: the solution of an ODE with initial conditions; exact at "
       "the initial point, numerical (mpmath Taylor ODE solver) at inexact arguments, unevaluated otherwise")


def _guess_recurrence(vals, n0, max_order=4, max_deg=4, spare=8):
    """Smallest (order, degree) recurrence sum_i p_i(n) a(n+i) = 0 fitting the values (exact nullspace)."""
    N = len(vals)
    nv = sp.Symbol("n")
    for tot in range(1, max_order + max_deg + 1):
        for r in range(1, max_order + 1):
            d = tot - r
            if d < 0 or d > max_deg:
                continue
            unk = (r + 1) * (d + 1)
            rows = N - r
            if rows < unk + spare:
                continue
            M = sp.Matrix(rows, unk, lambda m, c: sp.Integer(n0 + m) ** (c % (d + 1)) * vals[m + c // (d + 1)])
            ns = M.nullspace()
            if len(ns) != 1:
                continue
            v = ns[0]
            if all(c.is_rational for c in v):
                den = sp.ilcm(*[sp.fraction(c)[1] for c in v])
                v = v * den
                g = sp.igcd(*[int(c) for c in v if c != 0])
                v = v / g
            else:
                piv = [c for c in v if c != 0][-1]
                v = (v / piv).applyfunc(sp.simplify)
            ps = [sp.Add(*[v[i * (d + 1) + j] * nv ** j for j in range(d + 1)]) for i in range(r + 1)]
            if ps[-1] == 0 or ps[0] == 0:
                continue
            return ps
    return None


def _verify_rec(f, ps, nv):
    """f satisfies sum_i ps[i](n) f(n+i) == 0: symbolically, else numerically at several real points."""
    expr = sp.Add(*[p.subs(sp.Symbol("n"), nv) * f.subs(nv, nv + i) for i, p in enumerate(ps)])
    g = _rewrite_special(expr)
    try:
        if sp.simplify(sp.expand(sp.gammasimp(g))) == 0:
            return True
    except Exception:
        pass
    for t in (sp.Rational(37, 7), sp.Rational(91, 11), sp.Rational(23, 3), 12):
        try:
            z = complex(sp.N(g.subs(nv, t), 40))
            sc = max(1.0, max(abs(complex(sp.N(f.subs(nv, t + i), 40))) for i in range(len(ps))))
        except Exception:
            return False
        if abs(z) > 1e-25 * sc * 10 ** len(ps):
            return False
    return True


def _seq_values(f, nv, fname, count=40):
    n0 = 0
    while n0 < 30:
        try:
            vs = [sp.simplify(_rewrite_special(f).subs(nv, n0 + i)) for i in range(count)]
            if all(v.is_finite and not v.has(sp.zoo, sp.nan) for v in vs):
                return n0, vs
        except Exception:
            pass
        n0 += 1
    raise NotImplementedError(f"{fname}: the sequence values could not be computed")


@fn("DifferenceRootReduce", area=AREA, status="added",
    notes="DifferenceRootReduce[expr, n]: DifferenceRoot[Function[{y, n}, {rec == 0, ics}]][n] with a linear "
          "recurrence with polynomial coefficients (guessed from exact values, then verified symbolically or "
          "numerically at non-integer points); DifferenceRootReduce[f &]; DifferenceRootReduce[lhs == rhs, n] gives "
          "True when both sides reduce to the same sequence; non-holonomic inputs raise")
def _drr(expr, *rest):
    fname = "DifferenceRootReduce"
    if not rest:
        if not _head(expr, "Function"):
            raise ValueError(f"{fname}: give DifferenceRootReduce[expr, n] or a pure function")
        n_ = Symbol("n")
        body = _ev(Expr(expr, n_))
        r = _drr(body, n_)
        return r.head
    if len(rest) != 1:
        raise ValueError(f"{fname}: too many arguments")
    nv = _svar(rest[0], fname)
    if _head(expr, "Equal") and len(expr.args) == 2:
        d = SX(expr.args[0], fname) - SX(expr.args[1], fname)
        d = _rewrite_special(d)
        if sp.simplify(sp.expand(d)) == 0:
            return True
        n0, vs = _seq_values(d, nv, fname)
        if all(v == 0 for v in vs) and _verify_rec(d, [sp.Integer(1)], nv) and False:
            return True
        raise NotImplementedError(f"{fname}: the identity could not be decided")
    f = SX(expr, fname)
    n0, vs = _seq_values(f, nv, fname)
    ps = _guess_recurrence(vs, n0)
    if ps is None or not _verify_rec(f, ps, nv):
        raise NotImplementedError(f"{fname}: no verified linear recurrence with polynomial coefficients was found")
    r = len(ps) - 1
    nn = sp.Symbol("n")
    # choose the start beyond integer roots of the leading coefficient
    roots = [int(z) for z in sp.Poly(ps[-1], nn).ground_roots() if z.is_integer] if ps[-1].has(nn) else []
    start = max([n0] + [z + 1 for z in roots])
    if ps[-1].subs(nn, sp.Symbol("t")).as_poly(sp.Symbol("t")).LC() < 0 if ps[-1].has(nn) else ps[-1] < 0:
        ps = [-p for p in ps]
    yS, nS = Symbol("FormalY"), Symbol("FormalN")
    yf = sp.Function("FormalY")
    nF = sp.Symbol("FormalN")
    rec = sp.Add(*[sp.expand(p.subs(nn, nF)) * yf(nF + i) for i, p in enumerate(ps)])
    ics = [Expr("Equal", Expr("FormalY", start + i), out(sp.simplify(_rewrite_special(f).subs(nv, start + i))))
           for i in range(r)]
    body = L(Expr("Equal", out(rec), 0), *ics)
    return Expr(Expr("DifferenceRoot", Expr("Function", L(yS, nS), body)), Symbol(nv.name))


@fn("DifferentialRootReduce", area=AREA, status="added",
    notes="DifferentialRootReduce[expr, x]: DifferentialRoot[Function[{y, x}, {ode == 0, ics}]][x], a linear ODE "
          "with polynomial coefficients (SymPy's holonomic conversion, verified by substitution) with initial "
          "conditions at a regular point (0, else 1, 2, ...); DifferentialRootReduce[f &]; non-holonomic inputs raise")
def _difrr(expr, *rest):
    fname = "DifferentialRootReduce"
    if not rest:
        if not _head(expr, "Function"):
            raise ValueError(f"{fname}: give DifferentialRootReduce[expr, x] or a pure function")
        x_ = Symbol("x")
        return _difrr(_ev(Expr(expr, x_)), x_).head
    if len(rest) != 1:
        raise ValueError(f"{fname}: too many arguments")
    xv = _svar(rest[0], fname)
    f = SX(expr, fname)
    from sympy.holonomic import expr_to_holonomic
    try:
        h = expr_to_holonomic(f, xv)
        ann = h.annihilator
        ps = [sp.sympify(ann.parent.base.to_sympy(c)) for c in ann.listofpoly]
    except Exception:
        raise NotImplementedError(f"{fname}: {expr} is not recognised as a holonomic function")
    ps = [sp.expand(p.subs(sp.Symbol("x"), xv)) for p in ps]
    chk = sp.Add(*[p * sp.diff(f, xv, i) for i, p in enumerate(ps)])
    if sp.simplify(chk) != 0:
        raise NotImplementedError(f"{fname}: the differential equation could not be verified")
    r = len(ps) - 1
    x0 = None
    for c in range(0, 20):
        if ps[-1].subs(xv, c) != 0:
            ok = True
            ivs = []
            for i in range(r):
                try:
                    v = sp.simplify(sp.limit(sp.diff(f, xv, i), xv, c))
                except Exception:
                    ok = False
                    break
                if v.has(sp.oo, -sp.oo, sp.zoo, sp.nan) or isinstance(v, sp.Limit):
                    ok = False
                    break
                ivs.append(v)
            if ok:
                x0 = c
                break
    if x0 is None:
        raise NotImplementedError(f"{fname}: no regular point for the initial conditions was found")
    from sympy import lcm, fraction
    den = sp.Integer(1)
    for p in ps:
        for t in sp.Add.make_args(p):
            den = lcm(den, fraction(t.as_coeff_Mul()[0])[1])
    ps = [sp.expand(p * den) for p in ps]
    yf = sp.Function("FormalY")
    xs = sp.Symbol("FormalX")
    ode = sp.Add(*[p.subs(xv, xs) * (yf(xs) if i == 0 else sp.Derivative(yf(xs), (xs, i))) for i, p in enumerate(ps)])
    ics = []
    for i, v in enumerate(ivs):
        lhs = Expr("FormalY", x0) if i == 0 else Expr(Expr(Expr("Derivative", i), Symbol("FormalY")), x0)
        ics.append(Expr("Equal", lhs, out(v)))
    body = L(Expr("Equal", out(ode), 0), *ics)
    return Expr(Expr("DifferentialRoot", Expr("Function", L(Symbol("FormalY"), Symbol("FormalX")), body)),
                Symbol(xv.name))


# ───────────────────────────── sequence recognition and generating functions ─────────────────────────────

def _pw_first(G):
    """Closed forms of sums come as Piecewise((value, convergence condition), (Sum, True)): take the value."""
    return G.replace(lambda e: isinstance(e, sp.Piecewise) and len(e.args) == 2 and e.args[1][0].has(sp.Sum),
                     lambda e: e.args[0][0])


def _gamma_to_factorial(f, nv):
    return f.replace(lambda e: isinstance(e, sp.gamma) and (e.args[0] - nv).is_integer is True
                     and (e.args[0] - nv) >= 1, lambda e: sp.factorial(e.args[0] - 1))


def _fit_rational(xs, ys, spare=2, max_tot=8):
    """Exact rational function p(n)/q(n) through all (x, y) points with at least `spare` confirming points."""
    nv = sp.Symbol("n")
    m = len(xs)
    for tot in range(0, max_tot + 1):
        for dq in range(0, tot + 1):
            dp = tot - dq
            unk = dp + 1 + dq + 1
            if m < unk - 1 + spare:
                continue
            M = sp.Matrix(m, unk, lambda i, c: xs[i] ** c if c <= dp else -ys[i] * xs[i] ** (c - dp - 1))
            ns = M.nullspace()
            if len(ns) != 1:
                continue
            v = ns[0]
            p = sp.Add(*[v[c] * nv ** c for c in range(dp + 1)])
            q = sp.Add(*[v[dp + 1 + c] * nv ** c for c in range(dq + 1)])
            if q == 0 or (dq > 0 and sp.Poly(q, nv).degree() < dq) or (dp > 0 and sp.Poly(p, nv).degree() < dp):
                continue
            if any(q.subs(nv, x) == 0 for x in xs):
                continue
            f = sp.cancel(p / q)
            if all(sp.simplify(f.subs(nv, x) - y) == 0 for x, y in zip(xs, ys)):
                return sp.factor(f) if dq > 0 else sp.expand(f)
    return None


def _check_seq(f, nv, xs, ys):
    for x, y in zip(xs, ys):
        try:
            v = _rewrite_special(f).subs(nv, x)
            if sp.simplify(v - y) != 0:
                z = complex(sp.N(v - y, 40))
                if abs(z) > 1e-30 * max(1, abs(complex(sp.N(y, 40)))):
                    return False
        except Exception:
            return False
    return True


def _hypergeometric_fit(xs, ys, nv):
    """a(n+1)/a(n) rational -> a(n) = a(n0) prod r(k) (consecutive integer points)."""
    if any(y == 0 for y in ys) or len(ys) < 3:
        return None
    rs = [sp.simplify(ys[i + 1] / ys[i]) for i in range(len(ys) - 1)]
    r = _fit_rational(xs[:-1], rs)
    if r is None:
        return None
    k = sp.Symbol("k", integer=True)
    nn = sp.Symbol("n")
    n0 = xs[0]
    if not r.has(nn):
        f = ys[0] * r ** (nv - n0)
    else:
        try:
            pr = sp.product(r.subs(nn, k), (k, n0, nv - 1))
        except Exception:
            return None
        if pr.has(sp.Product):
            return None
        f = ys[0] * pr
    f = sp.simplify(sp.combsimp(f))
    f = sp.powsimp(f, force=True)
    return f if _check_seq(f, nv, xs, ys) else None


def _cfinite_fit(ys, n0, nv):
    """Constant-coefficient linear recurrence (with >= 2 confirming terms) solved exactly."""
    m = len(ys)
    for r in range(1, m // 2 + 1):
        if m < 2 * r + 2:
            break
        M = sp.Matrix(m - r, r, lambda i, j: ys[i + j])
        b = sp.Matrix(m - r, 1, lambda i, j: ys[i + r])
        try:
            sol, params = M[:r, :].gauss_jordan_solve(b[:r, :])
        except ValueError:
            continue
        if params.shape[0]:
            continue
        if (M * sol - b).applyfunc(sp.simplify) != sp.zeros(m - r, 1):
            continue
        cs = list(sol)
        yf = sp.Function("y")
        rec = yf(nv + r) - sp.Add(*[cs[j] * yf(nv + j) for j in range(r)])
        try:
            f = sp.rsolve(rec, yf(nv), {yf(n0 + i): ys[i] for i in range(r)})
        except Exception:
            return None
        if f is None:
            return None
        f = sp.simplify(f)
        xs = list(range(n0, n0 + m))
        # Fibonacci / Lucas recognition
        for base in (sp.fibonacci, sp.lucas):
            for s_ in range(-2, 3):
                g = base(nv + s_)
                v0 = [base(x + s_) for x in xs]
                if v0[0] != 0:
                    c = sp.simplify(ys[0] / v0[0])
                    if all(sp.simplify(c * a - y) == 0 for a, y in zip(v0, ys)):
                        return sp.expand(c * g)
        return f if _check_seq(f, nv, xs, ys) else None
    return None


def _find_seq(data, nv, depth=0):
    if not is_list(data) or not data.args:
        raise ValueError("FindSequenceFunction: the data must be a non-empty list")
    if all(is_list(p) and len(p.args) == 2 for p in data.args):
        xs = [SX(p.args[0], "FindSequenceFunction") for p in data.args]
        ys = [SX(p.args[1], "FindSequenceFunction") for p in data.args]
    else:
        xs = list(range(1, len(data.args) + 1))
        ys = [SX(v, "FindSequenceFunction") for v in data.args]
        xs = [sp.Integer(x) for x in xs]
    return _find_seq_xy(xs, ys, nv, depth)


def _find_seq_xy(xs, ys, nv, depth=0):
    nn = sp.Symbol("n")
    if len(xs) < 3:
        return None
    f = _fit_rational(xs, ys)
    if f is not None:
        return f.subs(nn, nv)
    consecutive = all(xs[i + 1] - xs[i] == 1 for i in range(len(xs) - 1)) and all(x.is_integer for x in xs)
    step = sp.simplify(xs[1] - xs[0])
    equal_steps = all(sp.simplify(xs[i + 1] - xs[i] - step) == 0 for i in range(len(xs) - 1))
    if consecutive:
        f = _hypergeometric_fit(xs, ys, nv)
        if f is not None:
            return f
        f = _cfinite_fit(ys, int(xs[0]), nv)
        if f is not None:
            return f
        if depth == 0:
            ds = [sp.simplify(ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
            g = _find_seq_xy(xs[:-1], ds, nv, 1)
            if g is not None:
                k = sp.Symbol("k", integer=True)
                try:
                    s_ = sp.summation(g.subs(nv, k), (k, xs[0], nv - 1))
                except Exception:
                    s_ = None
                if s_ is not None and not s_.has(sp.Sum):
                    f = sp.simplify(ys[0] + s_)
                    f = sp.powsimp(sp.expand(f), force=True)
                    if _check_seq(f, nv, xs, ys):
                        return f
    elif equal_steps and len(xs) >= 3 and all(y != 0 for y in ys):
        rs = [sp.simplify(ys[i + 1] / ys[i]) for i in range(len(ys) - 1)]
        if all(sp.simplify(r - rs[0]) == 0 for r in rs):
            f = sp.powsimp(sp.simplify(ys[0] * rs[0] ** ((nv - xs[0]) / step)), force=True)
            f = sp.simplify(f)
            if _check_seq(f, nv, xs, ys):
                return f
    return None


@fn("FindSequenceFunction", area=AREA, status="added",
    notes="FindSequenceFunction[{a1, a2, ...}, n] (or {{n1, a1}, ...}): rational functions of n, hypergeometric "
          "terms (rational successive ratios: products, factorials, powers), constant-coefficient linear recurrences "
          "(solved exactly; Fibonacci/LucasL recognised) and sums of hypergeometric differences, each fitted exactly "
          "with at least two confirming terms; FindSequenceFunction[list] gives a pure function; stays unevaluated "
          "when nothing is found")
def _findseqfn(data, *rest):
    if len(rest) > 1:
        raise ValueError("FindSequenceFunction: too many arguments")
    if rest:
        nv = _svar(rest[0], "FindSequenceFunction")
        f = _find_seq(data, nv)
        if f is None:
            raise Unevaluated
        return out(_gamma_to_factorial(f, nv))
    nv = sp.Symbol("__slot1")
    f = _find_seq(data, nv)
    if f is None:
        raise Unevaluated
    body = K.substitute(out(_gamma_to_factorial(f, nv)), {"__slot1": Expr("Slot", 1)})
    return Expr("Function", body)


@fn("FindGeneratingFunction", area=AREA, status="added",
    notes="FindGeneratingFunction[{a0, a1, ...}, x]: a rational generating function from a constant-coefficient "
          "linear recurrence (at least two confirming terms), else the sum of the FindSequenceFunction term; "
          "verified against the series; unevaluated when nothing is found")
def _findgf(data, x):
    fname = "FindGeneratingFunction"
    if not is_list(data) or len(data.args) < 3:
        raise ValueError(f"{fname}: at least three terms are required")
    xv = _svar(x, fname)
    ys = [SX(v, fname) for v in data.args]
    m = len(ys)
    G = None
    for r in range(1, m // 2 + 1):
        if m < 2 * r + 1:
            break
        M = sp.Matrix(m - r, r, lambda i, j: ys[i + j])
        b = sp.Matrix(m - r, 1, lambda i, j: ys[i + r])
        try:
            sol, params = M[:r, :].gauss_jordan_solve(b[:r, :])
        except ValueError:
            continue
        if params.shape[0] or (M * sol - b).applyfunc(sp.simplify) != sp.zeros(m - r, 1):
            continue
        cs = list(sol)
        Q = 1 - sp.Add(*[cs[r - 1 - j] * xv ** (j + 1) for j in range(r)])
        A = sp.Add(*[ys[i] * xv ** i for i in range(m)])
        P = sp.expand(Q * A)
        P = sp.Add(*[P.coeff(xv, i) * xv ** i for i in range(r)])
        G = sp.factor(sp.cancel(P / Q))
        break
    if G is None:
        nv = sp.Symbol("n", integer=True, nonnegative=True)
        f = _find_seq_xy([sp.Integer(i) for i in range(m)], ys, nv)
        if f is None:
            raise Unevaluated
        try:
            G = sp.summation(f * xv ** nv, (nv, 0, sp.oo))
        except Exception:
            raise Unevaluated
        G = _pw_first(G)
        if G.has(sp.Sum):
            raise Unevaluated
        G = sp.factor(sp.simplify(G))
    try:
        ser = sp.series(G, xv, 0, m).removeO()
    except Exception:
        raise Unevaluated
    if any(sp.simplify(ser.coeff(xv, i) - ys[i]) != 0 for i in range(m)):
        raise Unevaluated
    return out(G)


@fn("ExponentialGeneratingFunction", area=AREA, status="added",
    notes="ExponentialGeneratingFunction[expr, n, x] = Sum[expr x^n/n!, {n, 0, Infinity}] in closed form (SymPy "
          "summation, checked against the first Taylor coefficients); raises when no closed form is found")
def _egf(expr, n, x, *rest):
    fname = "ExponentialGeneratingFunction"
    if rest:
        raise NotImplementedError(f"{fname}: multivariate generating functions and options are not supported")
    nv = _svar(n, fname)
    xv = _svar(x, fname)
    f = SX(expr, fname)
    k = sp.Symbol("k", integer=True, nonnegative=True)
    fk = f.subs(nv, k)
    try:
        G = sp.summation(fk * xv ** k / sp.factorial(k), (k, 0, sp.oo))
    except Exception:
        G = None
    G = _pw_first(G) if G is not None else None
    if G is None or G.has(sp.Sum):
        raise NotImplementedError(f"{fname}: no closed form was found")
    G = sp.simplify(G)
    ser = sp.series(G, xv, 0, 7).removeO()
    for i in range(7):
        want = sp.simplify(_rewrite_special(fk.subs(k, i)) / sp.factorial(i))
        if sp.simplify(ser.coeff(xv, i) - want) != 0:
            raise NotImplementedError(f"{fname}: the closed form failed verification")
    return out(G)


@fn("FactorialMomentGeneratingFunction", area=AREA, status="added",
    notes="FactorialMomentGeneratingFunction[dist, t] = Expectation[t^x] = MomentGeneratingFunction[dist, Log[t]], "
          "simplified; raises when the moment generating function is not known")
def _fmgf(dist, t, *rest):
    fname = "FactorialMomentGeneratingFunction"
    if rest:
        raise ValueError(f"{fname}: too many arguments")
    tv = _svar(t, fname)
    m = _ev(Expr("MomentGeneratingFunction", dist, Expr("Log", Symbol(tv.name))))
    if _head(m, "MomentGeneratingFunction"):
        raise NotImplementedError(f"{fname}: the moment generating function of {dist} is not available")
    v = SX(m, fname)
    tp = sp.Symbol(tv.name, positive=True)
    v = sp.simplify(sp.powsimp(sp.expand_log(v.subs(tv, tp), force=True), force=True)).subs(tp, tv)
    return out(v)


def _truncate(e, m, activate):
    if isinstance(e, Expr):
        h = e.head
        is_sum = h == "Sum" or (isinstance(h, Expr) and h.head == "Inactive" and h.args and _is(h.args[0], "Sum"))
        args = [_truncate(a, m, activate) for a in e.args]
        if is_sum and len(args) >= 2:
            body, its = args[0], args[1:]
            for it in reversed(its):
                if not (is_list(it) and len(it.args) in (2, 3) and isinstance(it.args[0], Symbol)):
                    raise NotImplementedError("TruncateSum: iterators must be {i, a, b} or {i, b}")
                i_ = it.args[0]
                a_ = it.args[1] if len(it.args) == 3 else 1
                b_ = it.args[-1]
                hi = _ev(Expr("Plus", a_, m, -1))
                if K.is_int(m) and K.is_int(a_) and (_is(b_, "Infinity") or
                                                     (K.is_int(b_) and int(b_) - int(a_) + 1 > int(m))):
                    if activate:
                        body = _ev(Expr("Plus", *[K.substitute(body, {i_.name: int(a_) + k}) for k in range(int(m))]))
                    else:
                        body = Expr(Expr("Inactive", Symbol("Sum")), body, L(i_, a_, hi))
                elif K.is_int(m):
                    body = Expr(e.head, body, it) if not activate else _ev(Expr("Sum", body, it))
                else:
                    if not (_is(b_, "Infinity") or not K.is_int(b_)):
                        body = Expr(Expr("Inactive", Symbol("Sum")), body, L(i_, a_, Expr("Min", hi, b_)))
                    else:
                        body = Expr(Expr("Inactive", Symbol("Sum")), body, L(i_, a_, hi))
            return body
        return Expr(e.head if not isinstance(e.head, Expr) else _truncate(e.head, m, activate), *args)
    return e


@fn("TruncateSum", area=AREA, status="added",
    notes="TruncateSum[expr, m]: every Sum and Inactive[Sum] in expr is replaced by the sum of its first m terms "
          "(explicit, activated); symbolic m gives Inactive[Sum][f, {i, a, a + m - 1}]; ActivateResult -> False keeps "
          "the truncated sums inactive")
def _truncsum(expr, m, *rest):
    activate = True
    for r_ in rest:
        if is_rule(r_) and _is(r_.args[0], "ActivateResult"):
            activate = r_.args[1] is True
        else:
            raise ValueError("TruncateSum: unexpected argument")
    if K.is_int(m) and int(m) < 0:
        raise ValueError("TruncateSum: the number of terms must be non-negative")
    return _ev(_truncate(expr, m, activate))
