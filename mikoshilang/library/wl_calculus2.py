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
