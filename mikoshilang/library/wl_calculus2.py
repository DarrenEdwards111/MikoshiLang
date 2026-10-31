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
        return L(*[_array_simp(a, fname, simplify) for a in e.args])
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
    for t in ts:
        if is_list(t):
            continue
        if _contains_symarray(t):
            raise Unevaluated()
        try:
            v = K.to_sp(t)
        except Exception:
            raise Unevaluated()
        if isinstance(v, list) or not isinstance(v, sp.Basic) or v.free_symbols:
            raise Unevaluated()      # symbolic factors have unknown rank
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
        if not arg.has(n) or _arg_settles(arg, n):
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
        return a.args[0].has(n) and not _arg_settles(a.args[0], n)
    if isinstance(a, sp.exp):
        return a.args[0].has(n) and (a.args[0] / sp.I).is_real is True and not _arg_settles(a.args[0] / sp.I, n)
    return False


def _arg_settles(e, n):
    """True when e(n) tends to a finite limit (so a trig function of it does not oscillate), e.g. 1/n."""
    if e.is_polynomial(n):
        return False
    try:
        v = sp.limit(e, n, sp.oo)
    except Exception:
        return False
    return v.is_finite is True


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


# ───────────────────────────── fixed points and stability ─────────────────────────────

def _dep_names(xs, fname):
    items = list(xs.args) if is_list(xs) else [xs]
    names = []
    for it in items:
        if isinstance(it, Symbol):
            names.append(it.name)
        elif _head(it) and len(it.args) == 1 and isinstance(it.head, str):
            names.append(it.head)
        else:
            raise ValueError(f"{fname}: dependent variables must be x or x[t]")
    return names


def _ode_system(eqs, xs, t, fname):
    """First-order system for {eqns} in x_i[t]: (state symbols, rhs list, per-variable orders)."""
    tv = _svar(t, fname)
    names = _dep_names(xs, fname)
    eql = list(eqs.args) if is_list(eqs) else [eqs]
    E = []
    for e in eql:
        v = SX(e, fname)
        if not isinstance(v, sp.Equality):
            raise ValueError(f"{fname}: equations expected")
        E.append(v.lhs - v.rhs)
    fs = [sp.Function(nm)(tv) for nm in names]
    orders = []
    for f in fs:
        o = 0
        for e in E:
            for d in e.atoms(sp.Derivative):
                if d.expr == f:
                    o = max(o, d.derivative_count)
        if o == 0:
            raise ValueError(f"{fname}: no derivative of {f} appears")
        orders.append(o)
    state, sub = [], {}
    for f, o, nm in zip(fs, orders, names):
        for k in range(o):
            s_ = sp.Symbol(f"__S_{nm}_{k}")
            state.append(s_)
            sub[sp.Derivative(f, (tv, k)) if k else f] = s_
    tops = [sp.Symbol(f"__T_{nm}") for nm in names]
    Es = []
    for e in E:
        for f, o, tp in zip(fs, orders, tops):
            e = e.subs(sp.Derivative(f, (tv, o)), tp)
        for f, o, nm in zip(fs, orders, names):
            for k in range(o - 1, 0, -1):
                e = e.subs(sp.Derivative(f, (tv, k)), sub[sp.Derivative(f, (tv, k))])
            e = e.subs(f, sub[f])
        if e.has(tv):
            pass
        Es.append(e)
    sol = sp.solve(Es, tops, dict=True)
    if len(sol) != 1:
        raise NotImplementedError(f"{fname}: the equations cannot be solved uniquely for the highest derivatives")
    rhs = []
    idx = 0
    for f, o, nm, tp in zip(fs, orders, names, tops):
        for k in range(o):
            if k < o - 1:
                rhs.append(state[idx + k + 1])
            else:
                rhs.append(sol[0][tp])
        idx += o
    if any(r.has(tv) for r in rhs):
        raise NotImplementedError(f"{fname}: only autonomous systems are supported")
    return state, rhs, tv


def _rec_system(eqs, xs, n, fname):
    nv = _svar(n, fname)
    names = _dep_names(xs, fname)
    eql = list(eqs.args) if is_list(eqs) else [eqs]
    E = []
    for e in eql:
        v = SX(e, fname)
        if not isinstance(v, sp.Equality):
            raise ValueError(f"{fname}: equations expected")
        E.append(v.lhs - v.rhs)
    fns = [sp.Function(nm) for nm in names]
    orders = []
    for f in fns:
        sh = set()
        for e in E:
            for a in e.atoms(AppliedUndef):
                if a.func == f:
                    d = sp.simplify(a.args[0] - nv)
                    if not d.is_integer:
                        raise ValueError(f"{fname}: shifts must be integers")
                    sh.add(int(d))
        if not sh:
            raise ValueError(f"{fname}: {f} does not appear")
        if min(sh) != 0:
            E = [e.subs({f(nv + k): f(nv + k - min(sh)) for k in sorted(sh)}) for e in E]
            sh = {k - min(sh) for k in sh}
        if max(sh) == 0:
            raise ValueError(f"{fname}: no shifted term of {f}")
        orders.append(max(sh))
    state = []
    sub = {}
    for f, o, nm in zip(fns, orders, names):
        for k in range(o):
            s_ = sp.Symbol(f"__S_{nm}_{k}")
            state.append(s_)
            sub[f(nv + k)] = s_
    tops = [sp.Symbol(f"__T_{nm}") for nm in names]
    Es = []
    for e in E:
        e = e.subs({f(nv + o): tp for f, o, tp in zip(fns, orders, tops)})
        e = e.subs(sub)
        Es.append(e)
    sol = sp.solve(Es, tops, dict=True)
    if len(sol) != 1:
        raise NotImplementedError(f"{fname}: the equations cannot be solved uniquely for the highest shifts")
    rhs = []
    idx = 0
    for f, o, tp in zip(fns, orders, tops):
        for k in range(o):
            rhs.append(state[idx + k + 1] if k < o - 1 else sol[0][tp])
        idx += o
    if any(r.has(nv) for r in rhs):
        raise NotImplementedError(f"{fname}: only autonomous recurrences are supported")
    return state, rhs


def _fixed(state, eqs, fname):
    try:
        sols = sp.solve(eqs, state, dict=True)
    except NotImplementedError:
        raise NotImplementedError(f"{fname}: the fixed-point equations could not be solved")
    pts = []
    for s_ in sols:
        if any(v not in s_ for v in state):
            if all(e.subs(s_) == 0 for e in eqs) and len(s_) < len(state):
                raise NotImplementedError(f"{fname}: there is a continuum of fixed points")
        pt = [sp.simplify(s_.get(v, v)) for v in state]
        if any(p.has(*state) for p in pt):
            raise NotImplementedError(f"{fname}: there is a continuum of fixed points")
        if pt not in pts:
            pts.append(pt)
    if not eqs or all(e == 0 for e in eqs):
        raise NotImplementedError(f"{fname}: every point is a fixed point")

    def key(p):
        try:
            return tuple(float(sp.re(sp.N(v))) for v in p)
        except Exception:
            return tuple(0.0 for _ in p)
    return sorted(pts, key=key)


def _pts_arg(p, n, fname):
    if not is_list(p):
        raise ValueError(f"{fname}: the fixed points must be a list")
    pts = []
    for q in p.args:
        vals = [SX(v, fname) for v in (q.args if is_list(q) else [q])]
        if len(vals) != n:
            raise ValueError(f"{fname}: each point must have {n} coordinates")
        pts.append(vals)
    return pts


def _stab_cond(eigs, discrete):
    conds = []
    for lam in eigs:
        lam = sp.simplify(lam)
        if discrete:
            a = sp.Abs(lam)
            c = a < 1
        else:
            c = sp.re(lam) < 0
        conds.append(c)
    num = [c for c in conds if c in (sp.true, sp.false)]
    if any(c is sp.false for c in num):
        return False
    rest = [c for c in conds if c not in (sp.true, sp.false)]
    return rest


@fn("DFixedPoints", area=AREA, status="added",
    notes="DFixedPoints[eqns, x[t] | {x1, ...}, t]: the equilibria of an autonomous ODE system, as points of the "
          "first-order state (x, x', ..., x^(k-1)) for higher-order equations; sorted; raises for continua")
def _dfixed(eqs, xs, t):
    state, rhs, _tv = _ode_system(eqs, xs, t, "DFixedPoints")
    return out(_fixed(state, rhs, "DFixedPoints"))


@fn("RFixedPoints", area=AREA, status="added",
    notes="RFixedPoints[eqns, a[n] | {a1, ...}, n]: the fixed points of an autonomous recurrence system (all shifts "
          "equal), as points of the state (a[n], ..., a[n + k - 1]); sorted; raises for continua")
def _rfixed(eqs, xs, n):
    state, rhs = _rec_system(eqs, xs, n, "RFixedPoints")
    return out(_fixed(state, [r - s_ for r, s_ in zip(rhs, state)], "RFixedPoints"))


def _stability(state, rhs, pts, discrete, fname):
    J = sp.Matrix(rhs).jacobian(sp.Matrix(state))
    res = []
    for p in pts:
        Jp = J.subs(dict(zip(state, p)))
        Jp = Jp.applyfunc(sp.simplify)
        try:
            ev = Jp.eigenvals(multiple=True)
        except Exception:
            raise NotImplementedError(f"{fname}: the eigenvalues of the linearisation could not be computed")
        if len(ev) != Jp.rows:
            raise NotImplementedError(f"{fname}: the eigenvalues of the linearisation could not be computed")
        free = set().union(*[e.free_symbols for e in ev]) if ev else set()
        if not free:
            vals = [complex(sp.N(e)) for e in ev]
            if discrete:
                mags = [abs(v) for v in vals]
                if all(m < 1 - 1e-12 for m in mags):
                    c = True
                elif any(m > 1 + 1e-12 for m in mags):
                    c = False
                else:
                    raise NotImplementedError(f"{fname}: the linearisation at {p} is not hyperbolic; stability "
                                              "cannot be decided from it")
            else:
                if all(v.real < -1e-12 for v in vals):
                    c = True
                elif any(v.real > 1e-12 for v in vals):
                    c = False
                else:
                    raise NotImplementedError(f"{fname}: the linearisation at {p} is not hyperbolic; stability "
                                              "cannot be decided from it")
            res.append(L(out(p), c))
            continue
        cs = []
        for e in ev:
            e = sp.simplify(e)
            if discrete:
                c = None
                if len(e.free_symbols) == 1:
                    (a,) = e.free_symbols
                    ar = sp.Symbol(a.name, real=True)
                    er = e.subs(a, ar)
                    if er.is_real is not False and sp.im(er) == 0:
                        try:
                            c = sp.solve_univariate_inequality(sp.Abs(er) < 1, ar, relational=True).subs(ar, a)
                        except Exception:
                            c = None
                cs.append(c if c is not None else sp.Abs(e) < 1)
            else:
                cs.append(Expr("Less", Expr("Re", out(e)), 0))
        cs = [out(c) if not isinstance(c, Expr) else c for c in cs]
        cond = cs[0] if len(cs) == 1 else Expr("And", *cs)
        res.append(L(out(p), _ev(cond)))
    return L(*res)


@fn("DStabilityConditions", area=AREA, status="added",
    notes="DStabilityConditions[eqns, x, t] (optionally with a list of points): {{point, condition}, ...} for "
          "asymptotic stability from the linearisation (all eigenvalues with negative real part: True/False for "
          "numeric Jacobians, Re[l] < 0 conditions otherwise); non-hyperbolic numeric cases raise")
def _dstab(eqs, xs, t, *rest):
    fname = "DStabilityConditions"
    state, rhs, _tv = _ode_system(eqs, xs, t, fname)
    if len(rest) > 1:
        raise ValueError(f"{fname}: too many arguments")
    pts = _pts_arg(rest[0], len(state), fname) if rest else _fixed(state, rhs, fname)
    return _stability(state, rhs, pts, False, fname)


@fn("RStabilityConditions", area=AREA, status="added",
    notes="RStabilityConditions[eqns, a, n] (optionally with points): {{point, condition}, ...} for asymptotic "
          "stability of the fixed points from the linearisation (all eigenvalues inside the unit circle; a single "
          "real parameter gives an interval such as -1 < a < 1, otherwise Abs[l] < 1 conditions)")
def _rstab(eqs, xs, n, *rest):
    fname = "RStabilityConditions"
    state, rhs = _rec_system(eqs, xs, n, fname)
    if len(rest) > 1:
        raise ValueError(f"{fname}: too many arguments")
    pts = _pts_arg(rest[0], len(state), fname) if rest else _fixed(state, [r - s_ for r, s_ in zip(rhs, state)],
                                                                    fname)
    return _stability(state, rhs, pts, True, fname)


# ───────────────────────────── Lyapunov and Riccati matrix equations ─────────────────────────────

def _mat(x, fname, square=False):
    return matrix_arg(x, fname, square=square)


def _ct(m):
    return m.H


def _vec_solve(terms, C, fname):
    """Solve sum_k A_k . X . B_k == C for X (vec / Kronecker form), exactly or numerically."""
    n, m = C.shape
    K_ = sp.zeros(n * m, n * m)
    from sympy.physics.quantum import TensorProduct
    for A, B in terms:
        if A.shape != (n, n) or B.shape != (m, m):
            raise ValueError(f"{fname}: the matrix dimensions are inconsistent")
        K_ += TensorProduct(B.T, A)
    vecC = sp.Matrix([C[i, j] for j in range(m) for i in range(n)])
    inexact = any(isinstance(v, sp.Float) or v.has(sp.Float) for v in list(K_) + list(vecC))
    if inexact:
        import numpy as np
        try:
            Kn = np.array(K_.evalf(), dtype=complex)
            cn = np.array(vecC.evalf(), dtype=complex).ravel()
        except TypeError:
            raise NotImplementedError(f"{fname}: mixed symbolic and inexact input is not supported")
        if np.linalg.matrix_rank(Kn) < Kn.shape[0]:
            raise ValueError(f"{fname}: the equation does not have a unique solution")
        xv = np.linalg.solve(Kn, cn)
        X = [[xv[j * n + i] for j in range(m)] for i in range(n)]
        return R([[float(v.real) if abs(v.imag) < 1e-13 * max(1, abs(v)) else complex(v) for v in row]
                  for row in X])
    if K_.det() == 0:
        raise ValueError(f"{fname}: the equation does not have a unique solution")
    xv = K_.LUsolve(vecC)
    X = sp.Matrix(n, m, lambda i, j: sp.simplify(xv[j * n + i]))
    return mat_out(X)


def _ad(x, fname):
    if is_list(x) and len(x.args) == 2 and all(is_list(r) and r.args and is_list(r.args[0]) for r in x.args):
        return _mat(x.args[0], fname), _mat(x.args[1], fname)
    return _mat(x, fname), None


@fn("LyapunovSolve", area=AREA, status="added",
    notes="LyapunovSolve[a, c]: a.x + x.a^H == c; [a, b, c]: a.x + x.b == c; [{a, d}, c]: a.x.d^H + d.x.a^H == c; "
          "[{a, d}, {b, e}, c]: a.x.e + d.x.b == c; exact (Kronecker linear system) or numerical for inexact input; "
          "raises when the solution is not unique")
def _lyap(*args):
    fname = "LyapunovSolve"
    if len(args) == 2:
        (A, D), C = _ad(args[0], fname), _mat(args[1], fname)
        if D is None:
            terms = [(A, sp.eye(C.rows)), (sp.eye(C.rows), _ct(A))]
        else:
            terms = [(A, _ct(D)), (D, _ct(A))]
    elif len(args) == 3:
        (A, D), (B, E_), C = _ad(args[0], fname), _ad(args[1], fname), _mat(args[2], fname)
        if D is None and E_ is None:
            terms = [(A, sp.eye(C.cols)), (sp.eye(C.rows), B)]
        elif D is not None and E_ is not None:
            terms = [(A, E_), (D, B)]
        else:
            raise ValueError(f"{fname}: give a, b, c or {{a, d}}, {{b, e}}, c")
    else:
        raise ValueError(f"{fname}: two or three arguments are expected")
    return _vec_solve(terms, C, fname)


@fn("DiscreteLyapunovSolve", area=AREA, status="added",
    notes="DiscreteLyapunovSolve[a, c]: a.x.a^H - x == c; [a, b, c]: a.x.b - x == c; [{a, d}, c]: "
          "a.x.a^H - d.x.d^H == c; [{a, d}, {b, e}, c]: a.x.b - d.x.e == c; exact or numerical; raises when the "
          "solution is not unique")
def _dlyap(*args):
    fname = "DiscreteLyapunovSolve"
    if len(args) == 2:
        (A, D), C = _ad(args[0], fname), _mat(args[1], fname)
        if D is None:
            terms = [(A, _ct(A)), (-sp.eye(C.rows), sp.eye(C.cols))]
        else:
            terms = [(A, _ct(A)), (-D, _ct(D))]
    elif len(args) == 3:
        (A, D), (B, E_), C = _ad(args[0], fname), _ad(args[1], fname), _mat(args[2], fname)
        if D is None and E_ is None:
            terms = [(A, B), (-sp.eye(C.rows), sp.eye(C.cols))]
        elif D is not None and E_ is not None:
            terms = [(A, B), (-D, E_)]
        else:
            raise ValueError(f"{fname}: give a, b, c or {{a, d}}, {{b, e}}, c")
    else:
        raise ValueError(f"{fname}: two or three arguments are expected")
    return _vec_solve(terms, C, fname)


def _riccati_args(ab, qr, fname):
    if not (is_list(ab) and len(ab.args) == 2 and is_list(qr) and len(qr.args) in (2, 3)):
        raise ValueError(f"{fname}: the arguments must be {{a, b}}, {{q, r}} or {{q, r, p}}")
    A = _mat(ab.args[0], fname, square=True)
    B = _mat(ab.args[1], fname)
    Q = _mat(qr.args[0], fname, square=True)
    Rm = _mat(qr.args[1], fname, square=True)
    P = _mat(qr.args[2], fname) if len(qr.args) == 3 else sp.zeros(A.rows, B.cols)
    n, m = A.rows, B.cols
    if B.rows != n or Q.shape != (n, n) or Rm.shape != (m, m) or P.shape != (n, m):
        raise ValueError(f"{fname}: the matrix dimensions are inconsistent")
    return A, B, Q, Rm, P


def _inexact(*ms):
    return any(v.has(sp.Float) for M in ms for v in M)


def _num_matrix(M):
    import numpy as np
    try:
        return np.array(M.evalf(), dtype=complex)
    except TypeError:
        return None


def _np_out(X):
    import numpy as np
    X = np.asarray(X)
    if np.all(np.abs(X.imag) < 1e-12 * max(1.0, np.abs(X).max())):
        return R([[float(v) for v in row] for row in X.real])
    return R([[complex(v) for v in row] for row in X])


@fn("RiccatiSolve", area=AREA, status="added",
    notes="RiccatiSolve[{a, b}, {q, r}] (or {q, r, p}): the stabilising solution of the continuous algebraic Riccati "
          "equation; exact via the stable invariant subspace of the Hamiltonian matrix (verified by substitution) "
          "for exact input, SciPy's solve_continuous_are for inexact input; unevaluated when no exact solution is found")
def _care(ab, qr):
    fname = "RiccatiSolve"
    A, B, Q, Rm, P = _riccati_args(ab, qr, fname)
    if _inexact(A, B, Q, Rm, P):
        mats = [_num_matrix(M) for M in (A, B, Q, Rm, P)]
        if any(M is None for M in mats):
            raise NotImplementedError(f"{fname}: mixed symbolic and inexact input is not supported")
        from scipy.linalg import solve_continuous_are
        try:
            X = solve_continuous_are(mats[0], mats[1], mats[2], mats[3], s=mats[4])
        except Exception as exc:
            raise ValueError(f"{fname}: no stabilising solution ({exc})")
        return _np_out(X)
    if (A, B, Q, Rm, P) and any(M.free_symbols for M in (A, B, Q, Rm, P)):
        raise Unevaluated
    n = A.rows
    Ri = Rm.inv()
    A_ = A - B * Ri * P.H
    Q_ = Q - P * Ri * P.H
    H = sp.Matrix(sp.BlockMatrix([[A_, -B * Ri * B.H], [-Q_, -A_.H]]))
    X = _stable_subspace(H, n, lambda lam: complex(sp.N(lam)).real < -1e-12)
    if X is None:
        raise Unevaluated
    res = (A.H * X + X * A - (X * B + P) * Ri * (B.H * X + P.H) + Q).applyfunc(sp.simplify)
    if res != sp.zeros(n, n):
        raise Unevaluated
    return mat_out(X)


def _stable_subspace(H, n, stable):
    try:
        evs = H.eigenvects()
    except Exception:
        return None
    cols = []
    for lam, mult, vecs in evs:
        if stable(lam):
            if len(vecs) != mult:
                return None
            cols.extend(vecs)
    if len(cols) != n:
        return None
    U = sp.Matrix.hstack(*cols)
    U1, U2 = U[:n, :], U[n:, :]
    if sp.simplify(U1.det()) == 0:
        return None
    X = (U2 * U1.inv()).applyfunc(lambda v: sp.nsimplify(sp.radsimp(sp.simplify(v))) if v.is_number else
                                  sp.simplify(v))
    return X.applyfunc(sp.simplify)


@fn("DiscreteRiccatiSolve", area=AREA, status="added",
    notes="DiscreteRiccatiSolve[{a, b}, {q, r}] (or {q, r, p}): the stabilising solution of the discrete algebraic "
          "Riccati equation; exact via the stable eigenvectors of the symplectic pencil (invertible a, verified by "
          "substitution) for exact input, SciPy's solve_discrete_are for inexact input; unevaluated otherwise")
def _dare(ab, qr):
    fname = "DiscreteRiccatiSolve"
    A, B, Q, Rm, P = _riccati_args(ab, qr, fname)
    if _inexact(A, B, Q, Rm, P):
        mats = [_num_matrix(M) for M in (A, B, Q, Rm, P)]
        if any(M is None for M in mats):
            raise NotImplementedError(f"{fname}: mixed symbolic and inexact input is not supported")
        from scipy.linalg import solve_discrete_are
        try:
            X = solve_discrete_are(mats[0], mats[1], mats[2], mats[3], s=mats[4])
        except Exception as exc:
            raise ValueError(f"{fname}: no stabilising solution ({exc})")
        return _np_out(X)
    if any(M.free_symbols for M in (A, B, Q, Rm, P)):
        raise Unevaluated
    n = A.rows
    Ri = Rm.inv()
    A_ = A - B * Ri * P.H
    Q_ = Q - P * Ri * P.H
    if A_.det() == 0:
        raise Unevaluated
    G = B * Ri * B.H
    AiH = A_.H.inv()
    Z = sp.Matrix(sp.BlockMatrix([[A_ + G * AiH * Q_, -G * AiH], [-AiH * Q_, AiH]]))
    X = _stable_subspace(Z, n, lambda lam: abs(complex(sp.N(lam))) < 1 - 1e-12)
    if X is None:
        raise Unevaluated
    S_ = Rm + B.H * X * B
    res = (A.H * X * A - X - (A.H * X * B + P) * S_.inv() * (B.H * X * A + P.H) + Q).applyfunc(sp.simplify)
    if res != sp.zeros(n, n):
        raise Unevaluated
    return mat_out(X)


# ───────────────────────────── eigenproblems and Green's functions ─────────────────────────────

def _interval_spec(rng, fname):
    if is_list(rng) and len(rng.args) == 3:
        return _svar(rng.args[0], fname), SX(rng.args[1], fname), SX(rng.args[2], fname)
    if _head(rng, "Element") and len(rng.args) == 2 and _head(rng.args[1], "Interval"):
        iv = rng.args[1].args[0]
        if is_list(iv) and len(iv.args) == 2:
            v = rng.args[0]
            if is_list(v) and len(v.args) == 1:
                v = v.args[0]
            return _svar(v, fname), SX(iv.args[0], fname), SX(iv.args[1], fname)
    raise NotImplementedError(f"{fname}: only one-dimensional intervals {{x, a, b}} are supported")


def _split_neumann(e, fname):
    """Drop NeumannValue[0, ...] terms; nonzero Neumann values raise."""
    if _head(e, "Plus"):
        keep = []
        for a in e.args:
            if _head(a, "NeumannValue"):
                if not (len(a.args) == 2 and a.args[0] == 0):
                    raise NotImplementedError(f"{fname}: only homogeneous NeumannValue[0, ...] is supported")
                continue
            keep.append(a)
        return _ev(Expr("Plus", *keep)) if keep else 0
    if _head(e, "NeumannValue"):
        raise ValueError(f"{fname}: the operator is missing")
    return e


def _lin_coeffs(Lop, uname, xv, fname, max_order=None):
    """L[u] = sum c_i(x) u^(i)(x): coefficients [c_0, c_1, ...] (raises for non-linear or inhomogeneous)."""
    uf = sp.Function(uname)(xv)
    e = sp.expand(SX(Lop, fname))
    order = max([d.derivative_count for d in e.atoms(sp.Derivative) if d.expr == uf] or [0])
    Ys = [sp.Symbol(f"__U{i}") for i in range(order + 1)]
    for i in range(order, 0, -1):
        e = e.subs(sp.Derivative(uf, (xv, i)), Ys[i])
    e = e.subs(uf, Ys[0])
    if e.has(sp.Function(uname)):
        raise NotImplementedError(f"{fname}: the operator must be linear in {uname}")
    try:
        P = sp.Poly(e, *Ys)
    except sp.PolynomialError:
        raise NotImplementedError(f"{fname}: the operator must be linear in {uname}")
    if P.total_degree() > 1 or any(sum(m) == 0 for m in P.monoms()):
        raise NotImplementedError(f"{fname}: the operator must be linear and homogeneous in {uname}")
    cs = [P.coeff_monomial(Ys[i]) for i in range(order + 1)]
    if cs[-1] == 0 or order == 0:
        raise NotImplementedError(f"{fname}: a differential operator is required")
    return cs


def _bc_list(items, uname, xv, a, b, fname):
    """Boundary conditions -> list of (point, {derivative order: coefficient}) homogeneous linear conditions."""
    out_ = []
    uf = sp.Function(uname)
    for c in items:
        if _head(c, "DirichletCondition"):
            if len(c.args) != 2 or not _head(c.args[0], "Equal"):
                raise ValueError(f"{fname}: DirichletCondition[u == value, predicate] expected")
            if SX(c.args[0].args[1], fname) != 0:
                raise NotImplementedError(f"{fname}: only homogeneous boundary conditions are supported")
            pred = c.args[1]
            pts = []
            if pred is True:
                pts = [a, b]
            else:
                preds = list(pred.args) if _head(pred, "Or") else [pred]
                for pr in preds:
                    if not (_head(pr, "Equal") and len(pr.args) == 2):
                        raise NotImplementedError(f"{fname}: Dirichlet predicates must be x == a or True")
                    pts.append(SX(pr.args[1], fname))
            for p in pts:
                out_.append((p, {0: sp.Integer(1)}))
            continue
        if _head(c, "PeriodicBoundaryCondition"):
            raise NotImplementedError(f"{fname}: periodic boundary conditions are not supported")
        v = SX(c, fname)
        if not isinstance(v, sp.Equality):
            raise ValueError(f"{fname}: boundary conditions must be equations or DirichletCondition[...]")
        d = sp.expand(v.lhs - v.rhs)
        terms = {}
        pt = None
        rest = d
        for t in sp.Add.make_args(d):
            coef, atom = sp.Integer(1), None
            subs_ = [f for f in t.atoms(sp.Subs) if isinstance(f.expr, sp.Derivative) and f.expr.expr.func == uf]
            if subs_:
                f = subs_[0]
                atom, k, p = f, f.expr.derivative_count, f.point[0]
            else:
                for f in t.atoms(AppliedUndef):
                    if f.func == uf and not f.args[0].has(xv):
                        atom, k, p = f, 0, f.args[0]
            if atom is None:
                raise NotImplementedError(f"{fname}: only homogeneous boundary conditions are supported")
            coef = sp.simplify(t / atom)
            if coef.has(uf):
                raise NotImplementedError(f"{fname}: boundary conditions must be linear")
            if pt is not None and p != pt:
                raise NotImplementedError(f"{fname}: each condition must involve a single boundary point")
            pt = p
            terms[k] = terms.get(k, 0) + coef
        del rest
        out_.append((pt, terms))
    return out_


def _deig_setup(op, u, rng, fname):
    x, a, b = _interval_spec(rng, fname)
    uname = _fname_of(u)
    items = list(op.args) if is_list(op) else [op]
    Lop = _split_neumann(items[0], fname)
    bcs = _bc_list(items[1:], uname, x, a, b, fname)
    cs = _lin_coeffs(Lop, uname, x, fname)
    if len(cs) != 3:
        raise NotImplementedError(f"{fname}: only second-order operators are supported")
    if any(c.has(x) or not c.is_number for c in cs):
        raise NotImplementedError(f"{fname}: only constant numeric coefficients are supported")
    if not (a.is_number and b.is_number and (b - a).is_positive):
        raise NotImplementedError(f"{fname}: the interval must be finite and numeric")
    kinds = {}
    for p, terms in bcs:
        side = "a" if sp.simplify(p - a) == 0 else ("b" if sp.simplify(p - b) == 0 else None)
        if side is None:
            raise ValueError(f"{fname}: boundary conditions must be at the interval ends")
        if set(terms) == {0}:
            kd = "D"
        elif set(terms) == {1}:
            kd = "N"
        else:
            raise NotImplementedError(f"{fname}: only Dirichlet and Neumann conditions are supported")
        if kinds.get(side, kd) != kd:
            raise ValueError(f"{fname}: conflicting conditions at one end")
        kinds[side] = kd
    ka, kb = kinds.get("a", "N"), kinds.get("b", "N")
    c0, c1, c2 = cs
    if c1 != 0 and (ka, kb) != ("D", "D"):
        raise NotImplementedError(f"{fname}: first-derivative terms are supported only with Dirichlet conditions")
    return x, a, b, c0, c1, c2, ka, kb


def _deig(op, u, rng, n, fname, vectors):
    n = _num_opt(n, fname, "the number of eigenvalues")
    x, a, b, c0, c1, c2, ka, kb = _deig_setup(op, u, rng, fname)
    ell = b - a
    beta = c0 - c1 ** 2 / (4 * c2)
    alpha = -c2
    half = (ka != kb)
    j0 = 1 if (ka, kb) == ("D", "D") else 0
    extra = int(math.ceil(math.sqrt(2 * abs(float(beta)) / abs(float(alpha))) / (math.pi / float(ell)))) + 3
    modes = []
    for j in range(j0, j0 + n + extra):
        k = (j + sp.Rational(1, 2)) * sp.pi / ell if half else j * sp.pi / ell
        lam = sp.simplify(alpha * k ** 2 + beta)
        if ka == "D":
            f = sp.sin(k * (x - a))
        else:
            f = sp.cos(k * (x - a))
        if c1 != 0:
            f = sp.exp(-c1 * x / (2 * c2)) * f
        modes.append((lam, sp.simplify(f)))
    modes.sort(key=lambda m: (abs(float(m[0])), float(m[0])))
    modes = modes[:n]
    vals = [m[0] for m in modes]
    if vectors:
        return L(out(vals), out([m[1] for m in modes]))
    return out(vals)


@fn("DEigenvalues", area=AREA, status="added",
    notes="DEigenvalues[{L[u[x]], bcs}, u[x], {x, a, b}, n]: the n smallest-magnitude eigenvalues of L u == lambda u "
          "for constant-coefficient second-order operators on an interval with Dirichlet (DirichletCondition or "
          "u[a] == 0) or Neumann (u'[a] == 0, NeumannValue[0, ...], the default) ends, in closed form; other "
          "regions, variable coefficients and Robin/periodic conditions raise")
def _deigenvalues(op, u, rng, n):
    return _deig(op, u, rng, n, "DEigenvalues", False)


@fn("DEigensystem", area=AREA, status="added",
    notes="DEigensystem[{L[u[x]], bcs}, u[x], {x, a, b}, n]: {eigenvalues, eigenfunctions} (unnormalised, e.g. "
          "Sin[k x]) for the same class of problems as DEigenvalues")
def _deigensystem(op, u, rng, n):
    return _deig(op, u, rng, n, "DEigensystem", True)


@fn("GreenFunction", area=AREA, status="added",
    notes="GreenFunction[{L[u[x]], bcs}, u[x] | u, {x, a, b}, y]: the Green's function of a linear ODE of any order "
          "with homogeneous two-point or initial conditions (homogeneous solutions by DSolve; continuity of the "
          "first n-2 derivatives and the jump 1/c_n(y) of the (n-1)th), as HeavisideTheta pieces; u gives "
          "Function[{x, y}, ...]; PDEs and problems without a unique Green's function raise")
def _green(op, u, rng, y, *rest):
    fname = "GreenFunction"
    if rest:
        raise NotImplementedError(f"{fname}: time-dependent and PDE forms are not supported")
    x, a, b = _interval_spec(rng, fname)
    yv = _svar(y, fname)
    uname = _fname_of(u)
    items = list(op.args) if is_list(op) else [op]
    cs = _lin_coeffs(items[0], uname, x, fname)
    bcs = _bc_list(items[1:], uname, x, a, b, fname)
    n = len(cs) - 1
    if len(bcs) != n:
        raise NotImplementedError(f"{fname}: {n} homogeneous boundary/initial conditions are required")
    uf = sp.Function("__u")
    ode = sp.Add(*[c * sp.diff(uf(x), x, i) for i, c in enumerate(cs)])
    try:
        sol = sp.dsolve(ode, uf(x))
    except Exception:
        raise NotImplementedError(f"{fname}: the homogeneous equation could not be solved")
    if isinstance(sol, list) or not isinstance(sol, sp.Equality):
        raise NotImplementedError(f"{fname}: the homogeneous equation could not be solved")
    gen = sp.expand(sol.rhs)
    Cs = sorted([s_ for s_ in gen.free_symbols if s_.name.startswith("C") and s_.name[1:].isdigit()],
                key=lambda s_: int(s_.name[1:]))
    if len(Cs) != n:
        raise NotImplementedError(f"{fname}: the homogeneous equation could not be solved")
    basis = [sp.simplify(gen.coeff(c)) for c in Cs]
    if any(sp.simplify(sp.Add(*[c * sp.diff(f, x, i) for i, c in enumerate(cs)])) != 0 for f in basis):
        raise NotImplementedError(f"{fname}: the homogeneous solutions could not be verified")
    P = sp.symbols(f"__p0:{n}")
    Q = sp.symbols(f"__q0:{n}")
    right = sp.Add(*[p * f for p, f in zip(P, basis)])     # x > y
    left = sp.Add(*[q * f for q, f in zip(Q, basis)])      # x < y
    eqs = []
    for k in range(n - 1):
        eqs.append(sp.diff(right - left, x, k).subs(x, yv))
    eqs.append(sp.diff(right - left, x, n - 1).subs(x, yv) - 1 / cs[-1].subs(x, yv))
    for p, terms in bcs:
        if sp.simplify(p - a) == 0:
            piece = left if not (_is_inf(b) and all(sp.simplify(q - a) == 0 for q, _t in bcs)) else left
        elif sp.simplify(p - b) == 0:
            piece = right
        else:
            raise ValueError(f"{fname}: conditions must be at the ends of the interval")
        if _is_inf(p):
            raise NotImplementedError(f"{fname}: conditions at infinity are not supported")
        eqs.append(sp.Add(*[c * sp.diff(piece, x, k).subs(x, p) for k, c in terms.items()]))
    sol = sp.solve(eqs, list(P) + list(Q), dict=True)
    if len(sol) != 1 or any(v not in sol[0] for v in list(P) + list(Q)):
        raise NotImplementedError(f"{fname}: there is no unique Green's function (the homogeneous problem has a "
                                  "non-trivial solution)")
    R_ = sp.simplify(sp.expand(right.subs(sol[0])))
    Lf = sp.simplify(sp.expand(left.subs(sol[0])))
    parts = []
    if R_ != 0:
        parts.append(Expr("Times", out(R_), Expr("HeavisideTheta", out(x - yv))))
    if Lf != 0:
        parts.append(Expr("Times", out(Lf), Expr("HeavisideTheta", out(yv - x))))
    G = _ev(Expr("Plus", *parts)) if parts else 0
    if isinstance(u, Symbol):
        return Expr("Function", L(Symbol(x.name), Symbol(yv.name)), G)
    return G


# ───────────────────────────── interpolation ─────────────────────────────

def _ip_data(data, fname):
    """[(node, [f, f', f'', ...]), ...] from the 1-D data forms of InterpolatingPolynomial."""
    if not is_list(data) or not data.args:
        raise ValueError(f"{fname}: the data must be a non-empty list")
    rows = []
    if not any(is_list(r) for r in data.args):
        for i, v in enumerate(data.args):
            rows.append((sp.Integer(i + 1), [SX(v, fname)]))
        return rows
    for r in data.args:
        if not is_list(r) or len(r.args) < 2:
            raise ValueError(f"{fname}: each data point must be {{x, f}} or {{x, f, df, ...}}")
        if is_list(r.args[0]):
            raise NotImplementedError(f"{fname}: multidimensional interpolation is not supported")
        node = SX(r.args[0], fname)
        if len(r.args) == 2 and is_list(r.args[1]):
            vals = [SX(v, fname) for v in r.args[1].args]
        else:
            vals = [SX(v, fname) for v in r.args[1:]]
        if not vals or any(isinstance(v, list) for v in vals):
            raise ValueError(f"{fname}: invalid derivative data at {r!r}")
        rows.append((node, vals))
    nodes = [n for n, _v in rows]
    for i in range(len(nodes)):
        for j in range(i):
            if sp.simplify(nodes[i] - nodes[j]) == 0:
                raise ValueError(f"{fname}: the abscissae must be distinct")
    return rows


def _hermite_newton(rows):
    """Nodes z_0..z_{n-1} (with repeats) and the Newton coefficients by confluent divided differences."""
    z, fv = [], []
    for node, vals in rows:
        for _k in range(len(vals)):
            z.append(node)
            fv.append(vals)
    n = len(z)
    table = [[None] * n for _ in range(n)]
    for i in range(n):
        table[i][i] = fv[i][0]
    for d in range(1, n):
        for i in range(n - d):
            j = i + d
            if z[i] is z[j] or (sp.simplify(z[j] - z[i]) == 0):
                table[i][j] = fv[i][d] / sp.factorial(d)
            else:
                table[i][j] = sp.simplify((table[i + 1][j] - table[i][j - 1]) / (z[j] - z[i]))
    return z, [table[0][j] for j in range(n)]


@fn("InterpolatingPolynomial", area=AREA, status="added",
    notes="InterpolatingPolynomial[{f1, f2, ...}, x] (nodes 1, 2, ...), [{{x1, f1}, ...}, x] and Hermite data "
          "{{x1, f1, df1, ...}, ...} or {{x1, {f1, df1, ...}}, ...}: the lowest-degree interpolating polynomial in "
          "Newton/Horner form by (confluent) divided differences; symbolic nodes and values allowed; "
          "multidimensional data and the Modulus option raise")
def _interppoly(data, x, *rest):
    fname = "InterpolatingPolynomial"
    rest, _o = _opts(rest, set(), fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    if is_list(x):
        if len(x.args) != 1:
            raise NotImplementedError(f"{fname}: multidimensional interpolation is not supported")
        x = x.args[0]
    rows = _ip_data(data, fname)
    z, c = _hermite_newton(rows)
    xs = x if not isinstance(x, Symbol) else x
    # Horner form c0 + (x - z0) (c1 + (x - z1) (c2 + ...))
    acc = out(c[-1])
    for k in range(len(c) - 2, -1, -1):
        acc = Expr("Plus", out(c[k]), Expr("Times", Expr("Plus", xs, out(-z[k])), acc))
    return _ev(acc) if not isinstance(xs, Symbol) or True else acc


def _fi_eval(expr, xname, fname):
    """A float -> complex evaluator for expr in the variable xname."""
    xs = sp.Symbol(xname)
    try:
        e = SX(expr, fname)
        if e.free_symbols - {xs}:
            raise ValueError(f"{fname}: the expression has symbolic parameters {sorted(map(str, e.free_symbols - {xs}))}")
        f = sp.lambdify(xs, e, "mpmath")
        f(mpmath.mpf("0.123456"))
        return lambda t: complex(f(mpmath.mpf(t)))
    except ValueError:
        raise
    except Exception:
        pass

    def g(t):
        v = _ev(Expr("N", Expr("ReplaceAll", expr, R(Expr("Rule", Symbol(xname), float(t))))))
        try:
            return complex(sp.N(K.to_sp(v)))
        except Exception:
            raise ValueError(f"{fname}: the expression does not evaluate to a number at {xname} = {t}")
    return g


@fn("FunctionInterpolation", area=AREA, status="added",
    notes="FunctionInterpolation[expr, {x, xmin, xmax}]: samples expr on a uniform grid (31 points, doubled until "
          "the piecewise interpolant of InterpolationOrder (default 3) agrees with expr at every midpoint to 1e-7 "
          "relative, at most 8193 points) and returns an InterpolatingFunction; several variables and region "
          "specifications raise")
def _funcinterp(expr, rng, *rest):
    from .wl_interp import _make, _Piecewise
    fname = "FunctionInterpolation"
    rest, o = _opts(rest, {"InterpolationOrder", "InterpolationPoints"}, fname)
    if rest:
        raise NotImplementedError(f"{fname}: multivariate interpolation is not supported")
    if not is_list(rng) or len(rng.args) != 3 or not isinstance(rng.args[0], Symbol):
        raise ValueError(f"{fname}: the range must be {{x, xmin, xmax}}")
    a = complex(sp.N(SX(rng.args[1], fname)))
    b = complex(sp.N(SX(rng.args[2], fname)))
    if a.imag or b.imag or not (a.real < b.real) or not all(map(math.isfinite, (a.real, b.real))):
        raise ValueError(f"{fname}: the range must be finite real numbers with xmin < xmax")
    a, b = a.real, b.real
    order = _num_opt(o.get("InterpolationOrder", 3), fname, "InterpolationOrder")
    n = _num_opt(o.get("InterpolationPoints", 31), fname, "InterpolationPoints")
    if n < 2:
        raise ValueError(f"{fname}: at least 2 interpolation points are needed")
    f = _fi_eval(expr, rng.args[0].name, fname)
    while True:
        xs = [a + (b - a) * i / (n - 1) for i in range(n)]
        ys = [f(t) for t in xs]
        if not all(math.isfinite(abs(y)) for y in ys):
            raise ValueError(f"{fname}: the expression is not finite on the interval")
        real = all(abs(y.imag) <= 1e-14 * max(1.0, abs(y)) for y in ys)
        ys_ = [y.real for y in ys] if real else ys
        pw = _Piecewise(xs, ys_, order)
        mids = [(xs[i] + xs[i + 1]) / 2 for i in range(n - 1)]
        scale = max(1.0, max(abs(y) for y in ys))
        if all(abs(complex(pw(m)) - f(m)) <= 1e-7 * scale for m in mids):
            return _make(pw, a, b)
        if n > 8193:
            raise NotImplementedError(f"{fname}: the interpolation did not converge (the function is not smooth "
                                      "enough on the interval)")
        n = 2 * n - 1


# ───────────────────────────── residues ─────────────────────────────

_NONMERO = (sp.log, sp.Abs, sp.re, sp.im, sp.conjugate, sp.arg, sp.asin, sp.acos, sp.atan, sp.acot, sp.asinh,
            sp.acosh, sp.atanh, sp.LambertW, sp.floor, sp.ceiling, sp.sign, sp.Heaviside, sp.erf, sp.zeta,
            sp.polylog, sp.loggamma, sp.besselj, sp.bessely, sp.uppergamma, sp.lowergamma, sp.Ei, sp.Si, sp.Ci,
            sp.li, sp.expint)
_ENTIRE = (sp.exp, sp.sin, sp.cos, sp.sinh, sp.cosh)


def _mero_kind(f, z):
    """'entire', 'mero' (built from rational operations, entire functions, tan/cot/sec/csc/tanh/.. and Gamma)
    or None (possibly non-meromorphic: branch points, essential singularities hidden in non-elementary heads)."""
    entire = True
    for a in sp.preorder_traversal(f):
        if not getattr(a, "has", None) or not a.has(z):
            continue
        if isinstance(a, sp.Pow):
            if not (a.exp.is_Integer):
                if a.exp.has(z) and not a.base.has(z) and a.base.is_positive:
                    continue     # b^g(z) = exp(g log b)
                return None
            if a.exp < 0:
                entire = False
        elif isinstance(a, (sp.Add, sp.Mul, sp.Symbol)) or isinstance(a, _ENTIRE):
            continue
        elif isinstance(a, (sp.tan, sp.cot, sp.sec, sp.csc, sp.tanh, sp.coth, sp.sech, sp.csch, sp.gamma)):
            entire = False
        else:
            return None
    if entire:
        return "entire"
    return "mero"


def _poly_bound(g, z, bound):
    P = sp.Poly(g, z)
    return sum(abs(complex(sp.N(c))) * bound ** k for (k,), c in P.terms())


def _zero_families(fac, z, bound, fname):
    """Exact zeros with |z| <= bound of one multiplicative factor of a denominator."""
    if not fac.has(z):
        return []
    if fac.is_polynomial(z):
        P = sp.Poly(fac, z)
        sols = sp.roots(P, z)
        if sum(sols.values()) != P.degree():
            if P.free_symbols - {z}:
                raise NotImplementedError(f"{fname}: the poles could not be found exactly")
            return list(P.all_roots()) if P.is_irreducible else [r for q, _m in sp.factor_list(P)[1]
                                                                  for r in sp.Poly(q, z).all_roots()]
        return list(sols)
    if isinstance(fac, sp.Pow) and fac.exp.is_Integer and fac.exp > 0:
        return _zero_families(fac.base, z, bound, fname)
    if isinstance(fac, sp.exp):
        return []
    zeros = {sp.sin: (0, sp.pi), sp.cos: (sp.pi / 2, sp.pi), sp.sinh: (0, sp.I * sp.pi),
             sp.cosh: (sp.I * sp.pi / 2, sp.I * sp.pi)}
    for h, (off, per) in zeros.items():
        if isinstance(fac, h):
            g = fac.args[0]
            if not g.is_polynomial(z):
                raise NotImplementedError(f"{fname}: the poles could not be enumerated")
            if bound == float("inf"):
                raise NotImplementedError(f"{fname}: the function has infinitely many poles")
            M = _poly_bound(g, z, bound)
            kmax = int(math.ceil(M / abs(complex(sp.N(per))))) + 1
            res = []
            for k in range(-kmax, kmax + 1):
                res.extend(_zero_families(sp.expand(g - off - k * per), z, bound, fname))
            return res
    if isinstance(fac, sp.Add) and fac.is_polynomial(z) is False:
        try:
            ss = sp.solveset(fac, z, sp.S.Complexes)
        except Exception:
            ss = None
        if isinstance(ss, sp.FiniteSet):
            return list(ss.args)
    raise NotImplementedError(f"{fname}: the poles could not be found")


def _poles_in(f, z, inside, bound, fname):
    """Poles of a meromorphic f with |z| <= bound for which inside(complex) is True (exact values)."""
    g = f.rewrite(sp.sin) if f.has(sp.tan, sp.cot, sp.sec, sp.csc) else f
    if g.has(sp.tanh, sp.coth, sp.sech, sp.csch):
        g = g.rewrite(sp.sinh)
    g = sp.together(g)
    num, den = sp.fraction(g)
    cands = []
    for a in f.atoms(sp.gamma):
        if not a.has(z):
            continue
        arg = a.args[0]
        if not arg.is_polynomial(z):
            raise NotImplementedError(f"{fname}: Gamma of a non-polynomial argument is not supported")
        if bound == float("inf"):
            raise NotImplementedError(f"{fname}: the function has infinitely many poles")
        M = _poly_bound(arg, z, bound)
        for n in range(0, int(math.ceil(M)) + 2):
            cands.extend(_zero_families(sp.expand(arg + n), z, bound, fname))
    for fac in sp.Mul.make_args(sp.factor_terms(den)):
        cands.extend(_zero_families(fac, z, bound, fname))
    res = []
    for c in cands:
        if c.free_symbols:
            raise NotImplementedError(f"{fname}: the location of the poles depends on symbolic parameters")
        cv = complex(sp.N(c, 30))
        if abs(cv) > bound + 1e-9:
            continue
        flag = inside(cv)
        if flag is None and getattr(inside, "exact", None):
            flag = inside.exact(c)
        if flag is None:
            if sp.limit(f * (z - c), z, c) == 0 if False else _is_pole(f, z, c):
                raise NotImplementedError(f"{fname}: a pole lies on the contour")
            continue
        if flag and not any(abs(complex(sp.N(c - r, 30))) < 1e-20 for r in res):
            res.append(c)
    return res


def _is_pole(f, z, c):
    try:
        v = sp.limit(f, z, c)
        return not v.is_finite
    except Exception:
        return True


def _residue(f, z, p, fname):
    try:
        r = sp.residue(f, z, p)
    except Exception:
        r = None
    if r is None or r.has(sp.O) or r.has(sp.nan) or r.has(sp.zoo):
        raise NotImplementedError(f"{fname}: the residue at {p} could not be computed")
    return r


def _residue_sum_all(f, z, fname):
    kind = _mero_kind(f, z)
    if kind is None:
        raise NotImplementedError(f"{fname}: the function is not known to be meromorphic in the plane")
    if kind == "entire":
        return sp.Integer(0)
    num, den = sp.fraction(sp.together(f))
    if num.is_polynomial(z) and den.is_polynomial(z):
        # sum of finite residues = -Res_{z=oo} f = Res_{w=0} f(1/w)/w^2
        w = sp.Dummy("w")
        r = sp.residue(sp.together(f.subs(z, 1 / w) / w ** 2), w, 0)
        return sp.simplify(r)
    if den.is_polynomial(z) and _mero_kind(num, z) == "entire":
        poles = _poles_in(f, z, lambda v: True, float("inf"), fname)
        return sp.simplify(sp.Add(*[_residue(f, z, p, fname) for p in poles]))
    raise NotImplementedError(f"{fname}: the function may have infinitely many poles; give a bounded constraint "
                              "such as Abs[z] < r")


def _log_cuts_avoid(f, z, pred, bound, fname):
    """Replace Log[g] (g polynomial) by placeholders when the branch cut {g <= 0} misses the region; else raise."""
    reps = {}
    for a in sorted(f.atoms(sp.log), key=sp.default_sort_key):
        if not a.has(z):
            continue
        g = a.args[0]
        if not g.is_polynomial(z) or (g.free_symbols - {z}):
            raise NotImplementedError(f"{fname}: the function is not known to be meromorphic in the region")
        P = sp.Poly(g, z)
        coeffs = [complex(sp.N(c)) for c in P.all_coeffs()]
        smax = (_poly_bound(g, z, bound) if bound not in (None, float("inf")) else 1e6) * 2 + 10
        import numpy as _np
        for s_ in _np.concatenate([_np.linspace(0, 1, 201), _np.geomspace(1, smax, 400)]):
            cc = list(coeffs)
            cc[-1] += s_
            for r in _np.roots(cc) if len(cc) > 1 else []:
                if pred(complex(r)) is True:
                    raise NotImplementedError(f"{fname}: the branch cut of {a} meets the region")
        reps[a] = sp.Dummy("L")
    return reps


_RELS = {"Less": sp.Lt, "LessEqual": sp.Le, "Greater": sp.Gt, "GreaterEqual": sp.Ge, "Equal": sp.Eq,
         "Unequal": sp.Ne}


def _rel_sp(c, fname):
    """A MikoshiLang constraint (relations, And, Or, Not, chained inequalities) as a SymPy boolean."""
    if _head(c) and c.head in _RELS:
        xs = [SX(a, fname) for a in c.args]
        if len(xs) < 2:
            raise ValueError(f"{fname}: invalid constraint {c!r}")
        return sp.And(*[_RELS[c.head](xs[i], xs[i + 1]) for i in range(len(xs) - 1)])
    if _head(c, "And"):
        return sp.And(*[_rel_sp(a, fname) for a in c.args])
    if _head(c, "Or"):
        return sp.Or(*[_rel_sp(a, fname) for a in c.args])
    if _head(c, "Not"):
        return sp.Not(_rel_sp(c.args[0], fname))
    if _head(c, "Inequality"):
        a = c.args
        parts = []
        for i in range(0, len(a) - 2, 2):
            parts.append(_RELS[a[i + 1].name](SX(a[i], fname), SX(a[i + 2], fname)))
        return sp.And(*parts)
    raise ValueError(f"{fname}: {c!r} is not a constraint")


def _cons_pred(cons, z, fname):
    """(predicate on complex -> True/False/None (boundary), bound on |z| or None) for a constraint."""
    zr = sp.Symbol(z.name + "__re", real=True)
    zi = sp.Symbol(z.name + "__im", real=True)
    c = _rel_sp(cons, fname).subs(z, zr + sp.I * zi)
    c = sp.simplify(c.replace(sp.Abs, lambda a: sp.sqrt(sp.re(a) ** 2 + sp.im(a) ** 2)))
    if c.free_symbols - {zr, zi}:
        raise NotImplementedError(f"{fname}: constraints with symbolic parameters are not supported")
    f = sp.lambdify((zr, zi), c, "math")
    rels = [a for a in sp.preorder_traversal(c) if isinstance(a, sp.core.relational.Relational)]
    lhs = [sp.lambdify((zr, zi), r.lhs - r.rhs, "math") for r in rels]

    def pred(v):
        try:
            for g in lhs:
                if abs(g(v.real, v.imag)) < 1e-12:
                    return None
            return bool(f(v.real, v.imag))
        except Exception:
            return None
    def exact(pt):
        try:
            v = sp.simplify(c.subs({zr: sp.re(pt), zi: sp.im(pt)}))
        except Exception:
            return None
        return True if v is sp.true else (False if v is sp.false else None)
    pred.exact = exact
    bound = None
    if isinstance(cons, Expr) and cons.head in ("Less", "LessEqual") and _head(cons.args[0], "Abs"):
        try:
            bound = float(sp.N(SX(cons.args[1], fname)))
        except Exception:
            bound = None
    return pred, bound


@fn("ResidueSum", area=AREA, status="added",
    notes="ResidueSum[f, z]: the sum of the residues at all finite poles (rational f: -Res at infinity; entire f: "
          "0; other functions with infinitely many poles raise); ResidueSum[{f, cons}, z]: the residues at the "
          "poles satisfying cons (bounded constraints Abs[z - c] < r, or any constraint for rational f), with "
          "poles found exactly (polynomial roots, periodic families enumerated, Gamma poles); non-meromorphic "
          "functions, poles on the constraint boundary and symbolic pole locations raise")
def _residuesum(fc, z, *rest):
    fname = "ResidueSum"
    rest, _o = _opts(rest, {"GenerateConditions", "PerformanceGoal"}, fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    zv = _svar(z, fname)
    if is_list(fc):
        if len(fc.args) != 2:
            raise ValueError(f"{fname}: expected {{f, cons}}")
        f = SX(fc.args[0], fname)
        pred, bound = _cons_pred(fc.args[1], zv, fname)
        reps = _log_cuts_avoid(f, zv, pred, bound, fname)
        fm = f.xreplace(reps)
        kind = _mero_kind(fm, zv)
        if kind is None:
            raise NotImplementedError(f"{fname}: the function is not known to be meromorphic in the region")
        if kind == "entire":
            return 0
        num, den = sp.fraction(sp.together(fm))
        finite = den.is_polynomial(zv) and _mero_kind(num, zv) == "entire" and not fm.has(sp.gamma)
        if bound is None and not finite:
            raise NotImplementedError(f"{fname}: the function has infinitely many poles; the constraint must bound "
                                      "the region (Abs[z] < r)")
        poles = _poles_in(fm, zv, pred, bound if bound is not None else float("inf"), fname)
        tot = sp.Add(*[_residue(f, zv, p, fname) for p in poles])
        return out(sp.simplify(tot))
    f = SX(fc, fname)
    return out(_residue_sum_all(f, zv, fname))


# ───────────────────────────── line, surface and contour integrals ─────────────────────────────

_T = sp.Dummy("t", real=True)
_U = sp.Dummy("u", real=True)
_V = sp.Dummy("v", real=True)


def _ptv(p, fname, n=None):
    if not is_list(p) or not p.args or any(is_list(a) for a in p.args):
        raise ValueError(f"{fname}: {p!r} is not a point")
    v = [SX(a, fname) for a in p.args]
    if n is not None and len(v) != n:
        raise ValueError(f"{fname}: the point {p!r} does not have dimension {n}")
    return v


def _pts(p, fname, n=None):
    if not is_list(p) or len(p.args) < 1 or not all(is_list(a) for a in p.args):
        raise ValueError(f"{fname}: {p!r} is not a list of points")
    return [_ptv(a, fname, n) for a in p.args]


def _seg(p, q):
    """Oriented segment p -> q, t in [0, 1]."""
    return ([pi + _T * (qi - pi) for pi, qi in zip(p, q)], _T, 0, 1)


def _int_spec(spec, fname):
    """Element[{x, y, ...}, reg] -> ([sympy vars], reg)."""
    if not _head(spec, "Element") or len(spec.args) != 2:
        raise ValueError(f"{fname}: the second argument must be {{x, y, ...}} \u2208 region (Element[...])")
    vs = spec.args[0]
    if not is_list(vs):
        raise NotImplementedError(f"{fname}: implicit vector variables are not supported; give {{x, y, ...}}")
    xs = [_svar(v, fname) for v in vs.args]
    if len(set(xs)) != len(xs):
        raise ValueError(f"{fname}: the variables must be distinct")
    reg = spec.args[1]
    if _head(reg, "Inactive") or (isinstance(reg, Expr) and _head(reg.head, "Inactive")):
        raise NotImplementedError(f"{fname}: inactive regions are not supported")
    return xs, reg


def _circle_parts(reg, fname, dim=2):
    """(center, rx, ry, theta0, theta1) for Circle / Disk / 2-D Sphere."""
    a = list(reg.args)
    c = _ptv(a[0], fname, dim) if a else [sp.Integer(0)] * dim
    if len(c) != 2:
        raise NotImplementedError(f"{fname}: {reg.head} must be two-dimensional here")
    r = a[1] if len(a) > 1 else 1
    if is_list(r):
        rx, ry = SX(r.args[0], fname), SX(r.args[1], fname)
    else:
        rx = ry = SX(r, fname)
    th = (sp.Integer(0), 2 * sp.pi)
    if len(a) > 2:
        if not is_list(a[2]) or len(a[2].args) != 2:
            raise ValueError(f"{fname}: the angle range must be {{theta1, theta2}}")
        th = (SX(a[2].args[0], fname), SX(a[2].args[1], fname))
    if len(a) > 3:
        raise ValueError(f"{fname}: invalid {reg.head} specification")
    return c, rx, ry, th


def _ellipse_piece(c, rx, ry, th):
    return ([c[0] + rx * sp.cos(_T), c[1] + ry * sp.sin(_T)], _T, th[0], th[1])


def _polygon_vertices(reg, fname):
    h = reg.head
    a = list(reg.args)
    if h == "Triangle":
        pts = _pts(a[0], fname, 2) if a else [[0, 0], [1, 0], [0, 1]]
        if len(pts) != 3:
            raise ValueError(f"{fname}: Triangle needs three points")
    elif h == "Rectangle":
        p = _ptv(a[0], fname, 2) if a else [sp.Integer(0), sp.Integer(0)]
        q = _ptv(a[1], fname, 2) if len(a) > 1 else [p[0] + 1, p[1] + 1]
        pts = [[p[0], p[1]], [q[0], p[1]], [q[0], q[1]], [p[0], q[1]]]
    elif h == "Polygon":
        if len(a) != 1 or (is_list(a[0]) and a[0].args and is_list(a[0].args[0]) and is_list(a[0].args[0].args[0])):
            raise NotImplementedError(f"{fname}: polygons with holes or several components are not supported")
        pts = _pts(a[0], fname, 2)
    else:
        return None
    pts = [[sp.sympify(x) for x in p] for p in pts]
    area2 = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
                for i in range(len(pts)))
    try:
        sgn = complex(sp.N(area2)).real
    except Exception:
        raise NotImplementedError(f"{fname}: the polygon orientation depends on symbolic parameters")
    if sgn == 0:
        raise ValueError(f"{fname}: degenerate polygon")
    if sgn < 0:
        pts = pts[::-1]
    return pts


def _boundary_closed(reg, fname):
    """Counterclockwise boundary pieces of a 2-D solid region plus an inside test (True/False/None on boundary)."""
    h = reg.head
    if h in ("Disk", "Circle") or (h == "Sphere" and reg.args and is_list(reg.args[0]) and len(reg.args[0].args) == 2):
        c, rx, ry, th = _circle_parts(reg, fname)
        if th != (0, 2 * sp.pi):
            if h == "Circle":
                return None
            raise NotImplementedError(f"{fname}: disk sectors are not supported")
        if (rx.free_symbols | ry.free_symbols | set().union(*[sp.sympify(v).free_symbols for v in c])):
            return [_ellipse_piece(c, rx, ry, th)], None
        cx, cy, a_, b_ = (complex(sp.N(v)).real for v in (c[0], c[1], rx, ry))

        def inside(z):
            d = ((z.real - cx) / a_) ** 2 + ((z.imag - cy) / b_) ** 2
            return None if abs(d - 1) < 1e-10 else d < 1
        return [_ellipse_piece(c, rx, ry, th)], inside
    pts = _polygon_vertices(reg, fname)
    if pts is None:
        return None
    pieces = [_seg(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    if any(sp.sympify(v).free_symbols for p in pts for v in p):
        return pieces, None
    P = [(complex(sp.N(p[0])).real, complex(sp.N(p[1])).real) for p in pts]

    def inside(z):
        x, y = z.real, z.imag
        n = len(P)
        for i in range(n):
            (x1, y1), (x2, y2) = P[i], P[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
            if math.hypot(x - x1 - t * dx, y - y1 - t * dy) < 1e-10:
                return None
        wn = 0
        for i in range(n):
            (x1, y1), (x2, y2) = P[i], P[(i + 1) % n]
            cr = (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)
            if y1 <= y < y2 and cr > 0:
                wn += 1
            elif y2 <= y < y1 and cr < 0:
                wn -= 1
        if wn not in (0, 1):
            raise NotImplementedError(f"{fname}: self-intersecting polygons are not supported")
        return wn == 1
    return pieces, inside


def _param_region(reg, fname, nparams):
    a = list(reg.args)
    if len(a) != 2 or not is_list(a[0]) or not is_list(a[1]):
        raise NotImplementedError(f"{fname}: ParametricRegion must be given as ParametricRegion[{{f1, ...}}, "
                                  "{{u, umin, umax}, ...}]")
    ranges = a[1].args
    if len(ranges) != nparams:
        raise ValueError(f"{fname}: expected a parametric region with {nparams} parameter(s)")
    dums = [_T] if nparams == 1 else [_U, _V]
    subs, lims = {}, []
    for r, d in zip(ranges, dums):
        if not is_list(r) or len(r.args) != 3 or not isinstance(r.args[0], Symbol):
            raise NotImplementedError(f"{fname}: parameter ranges must be {{u, umin, umax}}")
        subs[sp.Symbol(r.args[0].name)] = d
        lims.append((d, SX(r.args[1], fname).subs(subs), SX(r.args[2], fname).subs(subs)))
    for d, lo, hi in lims:
        if (lo.free_symbols | hi.free_symbols) & set(dums):
            raise NotImplementedError(f"{fname}: parameter ranges must not depend on other parameters")
    vec = [SX(e, fname).subs(subs, simultaneous=True) for e in a[0].args]
    return vec, lims


def _curve_pieces(reg, n, fname):
    """Oriented pieces (r(t), t, a, b) of a curve in R^n."""
    if not isinstance(reg, Expr) or not isinstance(reg.head, str):
        raise ValueError(f"{fname}: {reg!r} is not a region")
    h, a = reg.head, list(reg.args)
    if h == "Line":
        if not a:
            raise ValueError(f"{fname}: empty Line")
        if is_list(a[0]) and a[0].args and is_list(a[0].args[0]) and is_list(a[0].args[0].args[0]):
            out_ = []
            for comp in a[0].args:
                ps = _pts(comp, fname, n)
                out_ += [_seg(ps[i], ps[i + 1]) for i in range(len(ps) - 1)]
            return out_
        ps = _pts(a[0], fname, n)
        if len(ps) < 2:
            raise ValueError(f"{fname}: a Line needs at least two points")
        return [_seg(ps[i], ps[i + 1]) for i in range(len(ps) - 1)]
    if h in ("HalfLine", "InfiniteLine"):
        if len(a) == 1:
            ps = _pts(a[0], fname, n)
            if len(ps) != 2:
                raise ValueError(f"{fname}: {h} needs two points")
            p, v = ps[0], [q - r for q, r in zip(ps[1], ps[0])]
        elif len(a) == 2:
            p, v = _ptv(a[0], fname, n), _ptv(a[1], fname, n)
        else:
            raise ValueError(f"{fname}: invalid {h}")
        lo = 0 if h == "HalfLine" else -sp.oo
        return [([pi + _T * vi for pi, vi in zip(p, v)], _T, lo, sp.oo)]
    if h == "Circle" or (h == "Sphere" and n == 2):
        if n != 2:
            raise ValueError(f"{fname}: a Circle is a curve in two dimensions")
        c, rx, ry, th = _circle_parts(reg, fname)
        return [_ellipse_piece(c, rx, ry, th)]
    if h == "ParametricRegion":
        vec, lims = _param_region(reg, fname, 1)
        if len(vec) != n:
            raise ValueError(f"{fname}: the region has dimension {len(vec)}, not {n}")
        return [(vec, lims[0][0], lims[0][1], lims[0][2])]
    raise NotImplementedError(f"{fname}: integration over {h} curves is not supported (use Line, HalfLine, "
                              "InfiniteLine, Circle or a one-parameter ParametricRegion)")


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _surface_pieces(reg, n, fname):
    """Oriented pieces (r(u, v), normal(u, v) = r_u x r_v or an outward normal, area_element, [(u,..),(v,..)])."""
    if not isinstance(reg, Expr) or not isinstance(reg.head, str):
        raise ValueError(f"{fname}: {reg!r} is not a region")
    h, a = reg.head, list(reg.args)
    if n == 2:
        # hypersurfaces of the plane are curves
        if h == "ParametricRegion":
            vec, lims = _param_region(reg, fname, 1)
            if len(vec) != 2:
                raise ValueError(f"{fname}: the region has dimension {len(vec)}, not 2")
            d = [sp.diff(e, _T) for e in vec]
            return [(vec, [-d[1], d[0]], None, lims)]
        if h in ("Sphere", "Circle", "Disk", "Rectangle", "Triangle", "Polygon"):
            bd = _boundary_closed(reg if h != "Circle" else Expr("Disk", *reg.args), fname)
            if bd is None:
                raise NotImplementedError(f"{fname}: arcs are not closed boundaries")
            res = []
            for vec, t, lo, hi in bd[0]:
                d = [sp.diff(e, t) for e in vec]
                res.append((vec, [d[1], -d[0]], None, [(t, lo, hi)]))
            return res
        raise NotImplementedError(f"{fname}: {h} is not supported in two dimensions")
    if n != 3:
        raise NotImplementedError(f"{fname}: only surfaces in two and three dimensions are supported")
    if h in ("Sphere", "Ball"):
        c = _ptv(a[0], fname, 3) if a else [sp.Integer(0)] * 3
        r = SX(a[1], fname) if len(a) > 1 else sp.Integer(1)
        if len(a) > 2:
            raise ValueError(f"{fname}: invalid {h}")
        vec = [c[0] + r * sp.sin(_U) * sp.cos(_V), c[1] + r * sp.sin(_U) * sp.sin(_V), c[2] + r * sp.cos(_U)]
        nrm = [r ** 2 * sp.sin(_U) ** 2 * sp.cos(_V), r ** 2 * sp.sin(_U) ** 2 * sp.sin(_V),
               r ** 2 * sp.sin(_U) * sp.cos(_U)]
        return [(vec, nrm, r ** 2 * sp.sin(_U), [(_U, 0, sp.pi), (_V, 0, 2 * sp.pi)])]
    if h in ("Cuboid", "Cube"):
        if h == "Cuboid":
            p = _ptv(a[0], fname, 3) if a else [sp.Integer(0)] * 3
            q = _ptv(a[1], fname, 3) if len(a) > 1 else [x + 1 for x in p]
        else:
            if len(a) == 0:
                cen, l_ = [sp.Integer(0)] * 3, sp.Integer(1)
            elif len(a) == 1:
                if is_list(a[0]):
                    cen, l_ = _ptv(a[0], fname, 3), sp.Integer(1)
                else:
                    cen, l_ = [sp.Integer(0)] * 3, SX(a[0], fname)
            else:
                cen, l_ = _ptv(a[0], fname, 3), SX(a[1], fname)
            p = [x - l_ / 2 for x in cen]
            q = [x + l_ / 2 for x in cen]
        res = []
        for k in range(3):
            i, j = [m for m in range(3) if m != k]
            for side, sgn in ((q[k], 1), (p[k], -1)):
                # parameters (s_i, s_j) ordered so that r_u x r_v points along sgn * e_k
                vec = [None] * 3
                vec[k] = side
                first, second = (i, j) if sgn == 1 else (j, i)
                if (k == 1):
                    first, second = second, first
                vec[first], vec[second] = _U, _V
                nrm = [0, 0, 0]
                nrm[k] = sgn
                lims = [(_U, p[first], q[first]), (_V, p[second], q[second])]
                res.append((vec, nrm, sp.Integer(1), lims))
        return res
    if h == "ParametricRegion":
        vec, lims = _param_region(reg, fname, 2)
        if len(vec) != 3:
            raise ValueError(f"{fname}: the region has dimension {len(vec)}, not 3")
        nrm = _cross([sp.diff(e, _U) for e in vec], [sp.diff(e, _V) for e in vec])
        return [(vec, nrm, None, lims)]
    if h == "Triangle":
        ps = _pts(a[0], fname, 3)
        if len(ps) != 3:
            raise ValueError(f"{fname}: Triangle needs three points")
        p1, p2, p3 = ps
        vec = [p1[m] + _U * (p2[m] - p1[m]) + _U * _V * (p3[m] - p2[m]) for m in range(3)]
        cr = _cross([p2[m] - p1[m] for m in range(3)], [p3[m] - p1[m] for m in range(3)])
        area = sp.sqrt(sum(c_ ** 2 for c_ in cr))
        return [(vec, None, _U * area, [(_U, 0, 1), (_V, 0, 1)])]
    raise NotImplementedError(f"{fname}: integration over {h} surfaces is not supported (use Sphere, Ball, Cuboid, "
                              "Cube, Triangle or a two-parameter ParametricRegion)")


def _simp(e):
    try:
        e2 = sp.simplify(e)
    except Exception:
        return e
    return e2


def _speed(vec, t):
    d = [sp.diff(e, t) for e in vec]
    s2 = _simp(sp.trigsimp(sp.expand(sum(x ** 2 for x in d))))
    return _simp(sp.sqrt(s2))


def _integrate_ranges(e, lims, fname):
    v = e
    for d, lo, hi in reversed(lims) if len(lims) > 1 else lims:
        try:
            v = sp.integrate(v, (d, lo, hi))
        except Exception:
            raise NotImplementedError(f"{fname}: the integral could not be evaluated symbolically")
        if v.has(sp.Integral) or v.has(sp.nan):
            raise NotImplementedError(f"{fname}: the integral could not be evaluated symbolically")
    return v


def _quad(e, lims, fname, subs=None):
    """Numerical value of the integral of e over the (rectangular) parameter ranges."""
    if subs:
        e = e.subs(subs)
        lims = [(d, sp.sympify(lo).subs(subs), sp.sympify(hi).subs(subs)) for d, lo, hi in lims]
    ds = [d for d, _lo, _hi in lims]
    if e.free_symbols - set(ds):
        raise ValueError(f"{fname}: the integrand has symbolic parameters {sorted(map(str, e.free_symbols - set(ds)))}")
    f = sp.lambdify(ds, e, "mpmath")
    rngs = []
    for _d, lo, hi in lims:
        lo, hi = sp.sympify(lo), sp.sympify(hi)
        if lo.free_symbols or hi.free_symbols:
            raise ValueError(f"{fname}: the parameter ranges are not numeric")
        lo_ = -mpmath.inf if lo == -sp.oo else (mpmath.inf if lo == sp.oo else mpmath.mpf(float(sp.N(lo))))
        hi_ = mpmath.inf if hi == sp.oo else (-mpmath.inf if hi == -sp.oo else mpmath.mpf(float(sp.N(hi))))
        if mpmath.isinf(lo_) or mpmath.isinf(hi_):
            rngs.append([lo_, hi_])
        else:
            rngs.append(list(mpmath.linspace(lo_, hi_, 5)))
    with mpmath.workdps(20):
        v = mpmath.quad(f, *rngs)
    return complex(v)


def _num_total(parts, fname, subs=None):
    return sum(_quad(e, lims, fname, subs) for e, lims in parts)


def _check_parts(val, parts, fname, extra_syms=()):
    """Compare a symbolic value with independent numerical quadrature (parameters set to sample values)."""
    params = set()
    for e, lims in parts:
        params |= e.free_symbols - {d for d, _a, _b in lims}
        for _d, lo, hi in lims:
            params |= sp.sympify(lo).free_symbols | sp.sympify(hi).free_symbols
    params |= sp.sympify(val).free_symbols
    subs = {p: sp.Rational(7 + 3 * i, 11 + 2 * i) for i, p in enumerate(sorted(params, key=lambda s_: s_.name))}
    try:
        num = _num_total(parts, fname, subs)
        sv = complex(sp.N(sp.sympify(val).subs(subs), 20))
    except NotImplementedError:
        raise
    except Exception:
        raise NotImplementedError(f"{fname}: the result could not be verified numerically")
    if not (math.isfinite(abs(num)) and abs(num - sv) <= 1e-7 * max(1.0, abs(num))):
        raise NotImplementedError(f"{fname}: the symbolic result could not be verified numerically")


def _register_fresnel():
    # FresnelS/FresnelC already map MikoshiLang -> SymPy (extended10); add the reverse so results containing
    # sympy.fresnels/fresnelc print as Wolfram heads.
    from .. import math_ops as _mo
    _mo._INV_FUNCS.setdefault(sp.fresnels, "FresnelS")
    _mo._INV_FUNCS.setdefault(sp.fresnelc, "FresnelC")
    _mo._FUNCS.setdefault("FresnelS", sp.fresnels)
    _mo._FUNCS.setdefault("FresnelC", sp.fresnelc)


_register_fresnel()


def _wlf(v):
    """SymPy special functions without a MikoshiLang mapping -> the Wolfram heads."""
    v = sp.sympify(v)
    if v.has(sp.fresnelc, sp.fresnels):
        v = v.replace(sp.fresnelc, sp.Function("FresnelC")).replace(sp.fresnels, sp.Function("FresnelS"))
    return out(v)


def _finish(val, parts, fname):
    val = _simp(val)
    _check_parts(val, parts, fname)
    return _wlf(val)


def _line_parts(f, spec, fname):
    xs, reg = _int_spec(spec, fname)
    n = len(xs)
    pieces = _curve_pieces(reg, n, fname)
    vector = is_list(f)
    if vector:
        F = [SX(c, fname) for c in f.args]
        if len(F) != n:
            raise ValueError(f"{fname}: the vector field must have {n} components")
    else:
        F = SX(f, fname)
    parts = []
    for vec, t, lo, hi in pieces:
        sub = dict(zip(xs, vec))
        if vector:
            e = sum(Fi.subs(sub, simultaneous=True) * sp.diff(ri, t) for Fi, ri in zip(F, vec))
        else:
            e = F.subs(sub, simultaneous=True) * _speed(vec, t)
        parts.append((e, [(t, lo, hi)]))
    return parts


def _surface_parts(f, spec, fname):
    xs, reg = _int_spec(spec, fname)
    n = len(xs)
    pieces = _surface_pieces(reg, n, fname)
    vector = is_list(f)
    if vector:
        F = [SX(c, fname) for c in f.args]
        if len(F) != n:
            raise ValueError(f"{fname}: the vector field must have {n} components")
    else:
        F = SX(f, fname)
    parts = []
    for vec, nrm, area, lims in pieces:
        sub = dict(zip(xs, vec))
        if vector:
            if nrm is None:
                raise NotImplementedError(f"{fname}: this surface has no defined orientation")
            e = sum(Fi.subs(sub, simultaneous=True) * ni for Fi, ni in zip(F, nrm))
        else:
            if area is None:
                if nrm is not None and n == 3:
                    area = _simp(sp.sqrt(_simp(sp.trigsimp(sp.expand(sum(c_ ** 2 for c_ in nrm))))))
                else:
                    area = _speed(vec, lims[0][0])
            e = F.subs(sub, simultaneous=True) * area
        parts.append((e, lims))
    return parts


_INT_OPTS = {"GenerateConditions", "Assumptions"}


@fn("LineIntegrate", area=AREA, status="added",
    notes="LineIntegrate[f, {x, y, ...} \u2208 curve] (scalar, f |r'| dt) and LineIntegrate[{p, q, ...}, {x, y, ...} "
          "\u2208 curve] (vector, F.r' dt) over Line (points in order), HalfLine, InfiniteLine, Circle (counterclockwise; "
          "arcs and ellipses) and one-parameter ParametricRegion (increasing parameter); symbolic integration "
          "checked against numerical quadrature (parameters sampled); unevaluable integrals raise")
def _lineint(f, spec, *rest):
    fname = "LineIntegrate"
    rest, _o = _opts(rest, _INT_OPTS, fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    parts = _line_parts(f, spec, fname)
    val = sp.Add(*[_integrate_ranges(_simp(e), lims, fname) for e, lims in parts])
    return _finish(val, parts, fname)


@fn("NLineIntegrate", area=AREA, status="added",
    notes="NLineIntegrate[f | {p, q, ...}, {x, ...} \u2208 curve]: numerical scalar or vector line integral over the "
          "curves supported by LineIntegrate (adaptive Gauss-Legendre/tanh-sinh quadrature)")
def _nlineint(f, spec, *rest):
    fname = "NLineIntegrate"
    rest, _o = _opts(rest, {"WorkingPrecision", "AccuracyGoal", "PrecisionGoal"}, fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    return _num_result(_num_total(_line_parts(f, spec, fname), fname), fname)


@fn("SurfaceIntegrate", area=AREA, status="added",
    notes="SurfaceIntegrate[f, {x, y, z} \u2208 surface] (scalar, f dS) and SurfaceIntegrate[{p, q, r}, ...] (flux "
          "F.n dS) over Sphere, Ball (its boundary), Cuboid and Cube (outward normals), Triangle (scalar only) and "
          "two-parameter ParametricRegion (normal r_u x r_v); in the plane, boundaries of Disk/Circle/Rectangle/"
          "Triangle/Polygon (outward normal) and parametric curves; results checked by numerical quadrature")
def _surfint(f, spec, *rest):
    fname = "SurfaceIntegrate"
    rest, _o = _opts(rest, _INT_OPTS, fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    parts = _surface_parts(f, spec, fname)
    val = sp.Add(*[_integrate_ranges(_simp(e), lims, fname) for e, lims in parts])
    return _finish(val, parts, fname)


@fn("NSurfaceIntegrate", area=AREA, status="added",
    notes="NSurfaceIntegrate[f | {p, q, r}, {x, y, z} \u2208 surface]: numerical scalar or flux surface integral over "
          "the surfaces supported by SurfaceIntegrate")
def _nsurfint(f, spec, *rest):
    fname = "NSurfaceIntegrate"
    rest, _o = _opts(rest, {"WorkingPrecision", "AccuracyGoal", "PrecisionGoal"}, fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    return _num_result(_num_total(_surface_parts(f, spec, fname), fname), fname)


def _num_result(v, fname):
    if not math.isfinite(abs(v)):
        raise ValueError(f"{fname}: the integral does not converge")
    if abs(v.imag) <= 1e-13 * max(1.0, abs(v)):
        return float(v.real)
    return R(complex(v.real, v.imag))


# contour integrals

def _contour_spec(spec, fname):
    if not _head(spec, "Element") or len(spec.args) != 2:
        raise ValueError(f"{fname}: the second argument must be z \u2208 contour (Element[z, contour])")
    return _svar(spec.args[0], fname), spec.args[1]


def _contour_pieces(reg, fname):
    """(pieces as complex z(t), closed?, inside-test or None)."""
    if not isinstance(reg, Expr) or not isinstance(reg.head, str):
        raise ValueError(f"{fname}: {reg!r} is not a contour")
    h = reg.head
    closed = None
    if h in ("Disk", "Rectangle", "Triangle", "Polygon") or (h == "Circle" and (len(reg.args) < 3)):
        bd = _boundary_closed(reg if h != "Circle" else Expr("Disk", *reg.args), fname)
        pieces, inside = bd
        closed = True
    else:
        pieces = _curve_pieces(reg, 2, fname)
        inside = None
    zp = [(vec[0] + sp.I * vec[1], t, lo, hi) for vec, t, lo, hi in pieces]
    return zp, closed, inside


def _contour_parts(f, zp, z):
    return [(f.subs(z, zt) * sp.diff(zt, t), [(t, lo, hi)]) for zt, t, lo, hi in zp]


@fn("ContourIntegrate", area=AREA, status="added",
    notes="ContourIntegrate[f, z \u2208 cont]: complex contour integral; closed contours (Circle, Disk, Rectangle, "
          "Triangle, Polygon boundaries, counterclockwise) of meromorphic f by the residue theorem (poles found "
          "exactly); open contours (Line, HalfLine, InfiniteLine, arcs, ParametricRegion[{x[t], y[t]}, ...]) by an "
          "antiderivative or direct parametrised integration; every result is checked against numerical "
          "quadrature (parameters sampled); poles on the contour and symbolic pole locations raise")
def _contourint(f, spec, *rest):
    fname = "ContourIntegrate"
    rest, o = _opts(rest, _INT_OPTS, fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    z, reg = _contour_spec(spec, fname)
    F = SX(f, fname)
    zp, closed, inside = _contour_pieces(reg, fname)
    parts = _contour_parts(F, zp, z)
    kind = _mero_kind(F, z)
    val = None
    if closed and kind == "entire":
        val = sp.Integer(0)
    elif closed and kind == "mero" and inside is not None:
        bound = max(abs(complex(sp.N(zt.subs(t, lo + (hi - lo) * k / 16)))) for zt, t, lo, hi in zp
                    for k in range(17)) + 1e-6
        if reg.head in ("Disk", "Circle"):
            c, rx, ry, _th = _circle_parts(reg, fname)
            bound = abs(complex(sp.N(c[0] + sp.I * c[1]))) + max(abs(complex(sp.N(rx))), abs(complex(sp.N(ry))))
        poles = _poles_in(F, z, inside, bound + 1e-6, fname)
        val = 2 * sp.pi * sp.I * sp.Add(*[_residue(F, z, p, fname) for p in poles])
    else:
        Fz = None
        if kind == "entire" or (kind == "mero" and not closed):
            try:
                Fz = sp.integrate(F, z)
                if Fz.has(sp.Integral):
                    Fz = None
            except Exception:
                Fz = None
        if Fz is not None and all(sp.sympify(lo).is_finite and sp.sympify(hi).is_finite for _zt, _t, lo, hi in zp):
            cand = sp.Add(*[Fz.subs(z, zt.subs(t, hi)) - Fz.subs(z, zt.subs(t, lo)) for zt, t, lo, hi in zp])
            try:
                _check_parts(_simp(cand), parts, fname)
                val = cand
            except NotImplementedError:
                val = None
        if val is None:
            val = sp.Add(*[_integrate_ranges(_simp(e), lims, fname) for e, lims in parts])
    return _finish(val, parts, fname)


@fn("NContourIntegrate", area=AREA, status="added",
    notes="NContourIntegrate[f, z \u2208 cont]: numerical contour integral along the contours supported by "
          "ContourIntegrate (Line, HalfLine, InfiniteLine, Circle, arcs, Disk/Rectangle/Triangle/Polygon "
          "boundaries counterclockwise, ParametricRegion)")
def _ncontourint(f, spec, *rest):
    fname = "NContourIntegrate"
    rest, _o = _opts(rest, {"WorkingPrecision", "AccuracyGoal", "PrecisionGoal", "MaxRecursion"}, fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    z, reg = _contour_spec(spec, fname)
    zp, _c, _i = _contour_pieces(reg, fname)
    return _num_result(_num_total(_contour_parts(SX(f, fname), zp, z), fname), fname)


# ───────────────────────────── fractional calculus ─────────────────────────────

def _frac_spec(spec, fname):
    if not is_list(spec) or len(spec.args) != 2:
        raise ValueError(f"{fname}: the second argument must be {{x, alpha}}")
    x = _svar(spec.args[0], fname)
    al = SX(spec.args[1], fname)
    if not al.free_symbols and not al.is_real:
        raise ValueError(f"{fname}: the order must be real")
    return x, al


def _power_terms(f, x):
    """[(c, p)] with f = sum c x^p (c free of x), or None."""
    terms = []
    for t in sp.Add.make_args(sp.expand(f)):
        c, g = t.as_independent(x, as_Add=False)
        if g == 1:
            terms.append((c, sp.Integer(0)))
        elif g == x:
            terms.append((c, sp.Integer(1)))
        elif isinstance(g, sp.Pow) and g.base == x and not g.exp.has(x):
            terms.append((c, g.exp))
        else:
            return None
    return terms


def _rl_power(c, p, al, x, fname):
    if p.is_number and not (p > -1):
        raise ValueError(f"{fname}: the fractional derivative of x^{p} at the lower terminal 0 does not exist "
                         "(the exponent must exceed -1)")
    return c * sp.gamma(p + 1) / sp.gamma(p + 1 - al) * x ** (p - al)


def _rl_integral(f, beta, x, fname):
    """I^beta f (beta > 0, numeric) = 1/Gamma(beta) Integral_0^x (x - s)^(beta - 1) f(s) ds."""
    s_ = sp.Dummy("s", positive=True)
    xp = sp.Dummy("x", positive=True)
    v = None
    for integrand in ((s_ ** (beta - 1) * f.subs(x, xp - s_)), ((xp - s_) ** (beta - 1) * f.subs(x, s_))):
        try:
            v = sp.integrate(sp.expand_trig(integrand) if integrand.has(sp.sin, sp.cos) else integrand,
                             (s_, 0, xp), conds="none")
        except Exception:
            v = None
        if v is not None and not v.has(sp.Integral):
            break
    if v is None or v.has(sp.Integral):
        raise NotImplementedError(f"{fname}: the fractional integral could not be computed in closed form")
    return (v / sp.gamma(beta)).subs(xp, x)


def _frac_check(val, f, al, x, fname, caputo):
    """Compare with independent quadrature at two points (only when f, alpha are free of parameters)."""
    if sp.sympify(val).free_symbols - {x} or f.free_symbols - {x} or al.free_symbols:
        return
    for x0 in (sp.Rational(3, 10), sp.Rational(7, 10)):
        with mpmath.workdps(20):
            want = _nfrac(f, x, al, mpmath.mpf(float(x0)), fname, caputo)
        got = complex(sp.N(sp.sympify(val).subs(x, x0), 20))
        if not abs(got - want) <= 1e-7 * max(1.0, abs(want)):
            raise NotImplementedError(f"{fname}: the symbolic result could not be verified numerically")


def _fracd(f, spec, fname, caputo):
    if is_list(f):
        return L(*[_fracd(a, spec, fname, caputo) for a in f.args])
    x, al = _frac_spec(spec, fname)
    F = SX(f, fname)
    if al.is_number and al.is_integer:
        n = int(al)
        if n >= 0:
            return out(sp.diff(F, x, n))
        v = F
        for _ in range(-n):
            v = _rl_integral(v, 1, x, fname)
        return out(_simp(v))
    terms = _power_terms(F, x)
    if terms is not None:
        res = 0
        for c, p in terms:
            if caputo:
                if p.is_Integer and p >= 0:
                    if al.is_number:
                        res += 0 if al > p else _rl_power(c, p, al, x, fname)
                    else:
                        res += sp.Piecewise((_rl_power(c, p, al, x, fname), al <= p), (0, True))
                    continue
                if not al.is_number:
                    raise NotImplementedError(f"{fname}: symbolic orders are supported only for polynomial terms")
                if al > p + 1:
                    raise ValueError(f"{fname}: the Caputo derivative of x^{p} of order {al} does not exist")
            res += _rl_power(c, p, al, x, fname)
        return out(_simp(res) if res != 0 else 0)
    if not al.is_number:
        raise NotImplementedError(f"{fname}: symbolic orders are supported only for sums of powers of {x}")
    m = int(sp.ceiling(al)) if al > 0 else 0
    if caputo and al > 0:
        val = _rl_integral(sp.diff(F, x, m), m - al, x, fname)
    elif al > 0:
        val = sp.diff(_rl_integral(F, m - al, x, fname), x, m)
    else:
        val = _rl_integral(F, -al, x, fname)
    val = _simp(val)
    _frac_check(val, F, al, x, fname, caputo)
    return _wlf(val)


def _nfrac(F, x, al, x0, fname, caputo):
    """Numerical Riemann-Liouville / Caputo derivative of order al at x0 > 0 (lower terminal 0)."""
    al_ = float(sp.N(al))
    if x0 <= 0:
        raise ValueError(f"{fname}: the point must be positive (the lower terminal is 0)")
    f = sp.lambdify(x, F, "mpmath")
    if float(al_).is_integer():
        n = int(al_)
        if n >= 0:
            return complex(sp.N(sp.diff(F, x, n).subs(x, sp.Float(x0, 20)), 20))
        return complex(mpmath.quad(lambda s: (x0 - s) ** (-n - 1) * f(s), [0, x0]) / mpmath.gamma(-n))
    if al_ < 0:
        return complex(mpmath.quad(lambda s: (x0 - s) ** (-al_ - 1) * f(s), [0, x0]) / mpmath.gamma(-al_))
    m = int(math.ceil(al_))
    g = sp.lambdify(x, sp.diff(F, x, m), "mpmath")
    cap = mpmath.quad(lambda s: (x0 - s) ** (m - al_ - 1) * g(s), [0, x0]) / mpmath.gamma(m - al_)
    if caputo:
        return complex(cap)
    corr = 0
    for k in range(m):
        try:
            v0 = sp.limit(sp.diff(F, x, k), x, 0, "+")
        except Exception:
            v0 = sp.nan
        if not v0.is_finite:
            # f^(k)(0) is singular: differentiate the fractional integral numerically instead
            beta = m - al_

            def I(xx):
                return mpmath.quad(lambda s: (xx - s) ** (beta - 1) * f(s), [0, xx]) / mpmath.gamma(beta)
            return complex(mpmath.diff(I, x0, m))
        corr += complex(sp.N(v0)) * x0 ** (k - al_) / mpmath.gamma(k + 1 - al_)
    return complex(cap + corr)


@fn("FractionalD", area=AREA, status="added",
    notes="FractionalD[f, {x, alpha}]: the Riemann-Liouville differintegral with lower terminal 0; sums of powers "
          "c x^p (p > -1) in closed form for any (also symbolic) alpha, Gamma(p+1)/Gamma(p+1-alpha) x^(p-alpha); "
          "integer alpha gives D or repeated integrals; other f with numeric alpha via the defining integral "
          "(D^m I^(m-alpha) f), checked against quadrature; threads over lists")
def _fractionald(f, spec, *rest):
    rest, _o = _opts(rest, {"Assumptions", "GenerateConditions"}, "FractionalD")
    if rest:
        raise ValueError(f"FractionalD: unexpected arguments {rest}")
    return _fracd(f, spec, "FractionalD", False)


@fn("CaputoD", area=AREA, status="added",
    notes="CaputoD[f, {x, alpha}]: the Caputo differintegral I^(m-alpha) D^m f (m = Ceiling[alpha]) with lower "
          "terminal 0; powers in closed form (constants and x^k with k < alpha give 0; symbolic alpha gives a "
          "Piecewise for polynomial terms); other f with numeric alpha via the defining integral, checked against "
          "quadrature; threads over lists")
def _caputod(f, spec, *rest):
    rest, _o = _opts(rest, {"Assumptions", "GenerateConditions"}, "CaputoD")
    if rest:
        raise ValueError(f"CaputoD: unexpected arguments {rest}")
    return _fracd(f, spec, "CaputoD", True)


def _nfrac_fn(f, spec, x0, rest, fname, caputo):
    rest, o = _opts(rest, {"Method", "WorkingPrecision", "AccuracyGoal", "PrecisionGoal"}, fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    meth = o.get("Method")
    if meth is not None and not (isinstance(meth, Symbol) and meth.name == "Automatic") \
            and not (isinstance(meth, str) and meth == "RiemannLiouville"):
        raise NotImplementedError(f"{fname}: only the Riemann-Liouville method is supported")
    if is_list(f):
        return L(*[_nfrac_fn(a, spec, x0, (), fname, caputo) for a in f.args])
    if is_list(x0):
        return L(*[_nfrac_fn(f, spec, p, (), fname, caputo) for p in x0.args])
    x, al = _frac_spec(spec, fname)
    if al.free_symbols:
        raise ValueError(f"{fname}: the order must be numeric")
    F = SX(f, fname)
    if F.free_symbols - {x}:
        raise ValueError(f"{fname}: the function has symbolic parameters")
    p = complex(sp.N(SX(x0, fname)))
    if p.imag:
        raise ValueError(f"{fname}: the point must be real")
    with mpmath.workdps(20):
        v = _nfrac(F, x, al, mpmath.mpf(p.real), fname, caputo)
    return _num_result(v, fname)


@fn("NFractionalD", area=AREA, status="added",
    notes="NFractionalD[f, {x, alpha}, x0]: numerical Riemann-Liouville differintegral at x0 > 0 (Caputo integral "
          "of f^(m) by tanh-sinh quadrature plus the initial-value terms, or numerical differentiation of the "
          "fractional integral when f^(k)(0) is singular)")
def _nfractionald(f, spec, x0, *rest):
    return _nfrac_fn(f, spec, x0, rest, "NFractionalD", False)


@fn("NCaputoD", area=AREA, status="added",
    notes="NCaputoD[f, {x, alpha}, x0]: numerical Caputo differintegral at x0 > 0 (tanh-sinh quadrature of the "
          "defining integral of f^(m))")
def _ncaputod(f, spec, x0, *rest):
    return _nfrac_fn(f, spec, x0, rest, "NCaputoD", True)


# ───────────────────────────── changes of variables ─────────────────────────────

def _inactive_head(e, names):
    """The head name if e is Inactive[name][...] or an unevaluated name[...], else None."""
    if not isinstance(e, Expr):
        return None
    h = e.head
    if isinstance(h, Expr) and h.head == "Inactive" and len(h.args) == 1 and isinstance(h.args[0], Symbol) \
            and h.args[0].name in names:
        return h.args[0].name
    if isinstance(h, str) and h in names:
        return h
    return None


def _new_var(u, fname, what="new variable"):
    if is_list(u):
        if len(u.args) != 1:
            raise NotImplementedError(f"{fname}: only one {what} is supported")
        u = u.args[0]
    if not isinstance(u, Symbol):
        raise ValueError(f"{fname}: {u!r} is not a valid {what}")
    return u


def _eq_parts(tr, fname):
    if not _head(tr, "Equal") or len(tr.args) != 2:
        if _head(tr, "Rule") and isinstance(tr.args[0], str):
            raise NotImplementedError(f"{fname}: named coordinate-system transformations are not supported")
        raise ValueError(f"{fname}: the transformation must be an equation")
    return SX(tr.args[0], fname), SX(tr.args[1], fname)


def _inverse_maps(tvar, phi, x, fname):
    """Candidates psi(t) with phi(psi(t)) == t."""
    try:
        sols = sp.solve(sp.Eq(tvar, phi), x, dict=False)
    except Exception:
        sols = []
    if not sols:
        raise NotImplementedError(f"{fname}: the transformation could not be inverted")
    return list(sols)


def _indep_map(lhs, rhs, x, tvar, fname):
    """(phi or None, [psi candidates]) from t == phi(x) or x == psi(t) (either side order)."""
    if lhs == x and not rhs.has(x) and rhs.has(tvar):
        return None, [rhs]
    if rhs == x and not lhs.has(x) and lhs.has(tvar):
        return None, [lhs]
    if lhs == tvar and rhs.has(x) and not rhs.has(tvar):
        return rhs, _inverse_maps(tvar, rhs, x, fname)
    if rhs == tvar and lhs.has(x) and not lhs.has(tvar):
        return lhs, _inverse_maps(tvar, lhs, x, fname)
    raise ValueError(f"{fname}: the transformation must have the form t == phi(x) or x == psi(t)")


def _real_num(v):
    try:
        c = complex(sp.N(v, 20))
    except Exception:
        return None
    return c.real if abs(c.imag) < 1e-12 * max(1.0, abs(c)) else None


@fn("IntegrateChangeVariables", area=AREA, status="added",
    notes="IntegrateChangeVariables[Inactive[Integrate][f, x | {x, a, b}], u, u == phi[x] | x == psi[u]]: "
          "integration by substitution, f(psi(u)) psi'(u), returned as Inactive[Integrate][...]; for definite "
          "integrals the inverse branch mapping [ua, ub] onto [a, b] is chosen and the result is checked by "
          "quadrature; multiple integrals, regions and named coordinate charts raise")
def _intchange(integral, u, trans, *rest):
    fname = "IntegrateChangeVariables"
    rest, _o = _opts(rest, {"Assumptions", "GenerateConditions"}, fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    hn = _inactive_head(integral, ("Integrate", "NIntegrate"))
    if hn is None:
        raise ValueError(f"{fname}: the first argument must be Inactive[Integrate][...]")
    if len(integral.args) != 2:
        raise NotImplementedError(f"{fname}: multiple integrals are not supported")
    f = SX(integral.args[0], fname)
    it = integral.args[1]
    if _head(it, "Element"):
        raise NotImplementedError(f"{fname}: integrals over regions are not supported")
    if is_list(it):
        if len(it.args) != 3:
            raise ValueError(f"{fname}: invalid integration range {it!r}")
        x = _svar(it.args[0], fname)
        a, b = SX(it.args[1], fname), SX(it.args[2], fname)
    else:
        x, a, b = _svar(it, fname), None, None
    uu = _new_var(u, fname)
    t = sp.Symbol(uu.name, real=True)
    tplain = sp.Symbol(uu.name)
    f = f.subs(tplain, t)
    if t.name == x.name:
        raise ValueError(f"{fname}: the new variable must differ from {x}")
    if is_list(trans):
        if len(trans.args) != 1:
            raise NotImplementedError(f"{fname}: multivariate transformations are not supported")
        trans = trans.args[0]
    lhs, rhs = _eq_parts(trans, fname)
    lhs, rhs = lhs.subs(tplain, t), rhs.subs(tplain, t)
    phi, psis = _indep_map(lhs, rhs, x, t, fname)
    if a is None:
        psi = psis[0]
        g = _simp(f.subs(x, psi) * sp.diff(psi, t))
        return Expr(integral.head, _wlf(g.subs(t, tplain)), uu)
    lims = None
    choice = None
    for psi in psis:
        if phi is not None:
            ua, ub = phi.subs(x, a), phi.subs(x, b)
        else:
            sa = [s_ for s_ in sp.solve(sp.Eq(psi, a), t)]
            sb = [s_ for s_ in sp.solve(sp.Eq(psi, b), t)]
            if not sa or not sb:
                continue
            ua, ub = sa[0], sb[0]
            if len(sa) > 1 or len(sb) > 1:
                # pick the pair giving a monotone map onto [a, b]
                best = None
                for p in sa:
                    for q in sb:
                        if _mono_ok(psi, t, p, q, a, b):
                            best = (p, q)
                            break
                    if best:
                        break
                if best is None:
                    continue
                ua, ub = best
        if sp.simplify(psi.subs(t, ua) - a) != 0 or sp.simplify(psi.subs(t, ub) - b) != 0:
            if not (_real_num(psi.subs(t, ua) - a) == 0 and _real_num(psi.subs(t, ub) - b) == 0):
                continue
        if not _mono_ok(psi, t, ua, ub, a, b):
            continue
        choice, lims = psi, (ua, ub)
        break
    if choice is None:
        raise NotImplementedError(f"{fname}: no branch of the inverse transformation maps the new interval onto "
                                  "the old one")
    psi = choice
    ua, ub = (_simp(v) for v in lims)
    g = _sqrt_sq_fix(_simp(f.subs(x, psi) * sp.diff(psi, t)), t, ua, ub)
    # independent check
    if not (f.free_symbols - {x}) and all(_real_num(v) is not None for v in (a, b, ua, ub)):
        try:
            v1 = _quad(f, [(x, a, b)], fname)
            v2 = _quad(g, [(t, ua, ub)], fname)
        except Exception:
            v1 = v2 = None
        if v1 is not None and not abs(v1 - v2) <= 1e-7 * max(1.0, abs(v1)):
            raise NotImplementedError(f"{fname}: the transformed integral could not be verified")
    return Expr(integral.head, _wlf(g.subs(t, tplain)), L(uu, _wlf(ua), _wlf(ub)))


def _sqrt_sq_fix(g, t, ua, ub):
    """Sqrt[h^2] -> h or -h when h has a fixed sign on the (numeric) interval."""
    lo, hi = _real_num(ua), _real_num(ub)
    if lo is None or hi is None or g.free_symbols - {t}:
        return g

    def fix(e):
        h = e.args[0] if isinstance(e, sp.Abs) else e.base.base
        try:
            f = sp.lambdify(t, h, "mpmath")
            vals = [complex(f(mpmath.mpf(lo + (hi - lo) * k / 50))).real for k in range(51)]
        except Exception:
            return e
        if all(v >= 0 for v in vals):
            return h
        if all(v <= 0 for v in vals):
            return -h
        return e
    g2 = g.replace(lambda e: isinstance(e, sp.Abs) or (isinstance(e, sp.Pow) and e.exp == sp.Rational(1, 2)
                   and isinstance(e.base, sp.Pow) and e.base.exp == 2), fix)
    return _simp(g2) if g2 != g else g


def _mono_ok(psi, t, ua, ub, a, b):
    """psi maps the samples of [ua, ub] into [a, b] (numeric limits only; symbolic limits are accepted)."""
    vals = [_real_num(v) for v in (ua, ub, a, b)]
    if any(v is None for v in vals) or psi.free_symbols - {t}:
        return True
    ua_, ub_, a_, b_ = vals
    lo, hi = min(a_, b_), max(a_, b_)
    try:
        f = sp.lambdify(t, psi, "mpmath")
        for k in range(1, 40):
            s_ = ua_ + (ub_ - ua_) * k / 40
            v = complex(f(mpmath.mpf(s_)))
            if abs(v.imag) > 1e-9 or not (lo - 1e-9 <= v.real <= hi + 1e-9):
                return False
    except Exception:
        return False
    return True


@fn("DSolveChangeVariables", area=AREA, status="added",
    notes="DSolveChangeVariables[Inactive[DSolve][ode, y | y[x], x], u, t, trans]: changes the variables of a "
          "single ODE by the chain rule; trans is t == phi[x] or x == psi[t] (independent variable), "
          "y[x] == G[x, u[x]] (dependent variable, t = x) or a list of one of each; returns "
          "Inactive[DSolve][ode', u | u[t], t]; PDEs, systems and named coordinate charts raise")
def _dsolvechange(dsolve, u, t, trans, *rest):
    fname = "DSolveChangeVariables"
    rest, _o = _opts(rest, {"Assumptions"}, fname)
    if rest:
        raise ValueError(f"{fname}: unexpected arguments {rest}")
    if _inactive_head(dsolve, ("DSolve",)) is None:
        raise ValueError(f"{fname}: the first argument must be Inactive[DSolve][...]")
    if len(dsolve.args) != 3:
        raise ValueError(f"{fname}: expected Inactive[DSolve][eqn, y, x]")
    eqn, yspec, xspec = dsolve.args
    if is_list(xspec):
        if len(xspec.args) != 1:
            raise NotImplementedError(f"{fname}: partial differential equations are not supported")
        xspec = xspec.args[0]
    x = _svar(xspec, fname)
    if is_list(eqn):
        if len(eqn.args) != 1:
            raise NotImplementedError(f"{fname}: systems of equations are not supported")
        eqn = eqn.args[0]
    if is_list(yspec):
        if len(yspec.args) != 1:
            raise NotImplementedError(f"{fname}: systems of equations are not supported")
        yspec = yspec.args[0]
    applied = isinstance(yspec, Expr) and not isinstance(yspec, Symbol)
    yname = _fname_of(yspec)
    uu = _new_var(u, fname, "new function")
    tt = _new_var(t, fname)
    tv = sp.Symbol(tt.name, real=True) if tt.name != x.name else x
    tplain = sp.Symbol(tt.name)
    yF, uF = sp.Function(yname), sp.Function("__newfn")
    trs = list(trans.args) if is_list(trans) else [trans]
    psi = None
    Y = None
    for tr in trs:
        lhs, rhs = _eq_parts(tr, fname)
        unew = sp.Function(uu.name)
        lhs, rhs = (e.replace(lambda a: isinstance(a, AppliedUndef) and a.func == unew, lambda a: uF(*a.args))
                    .subs(tplain, tv) for e in (lhs, rhs))
        if lhs == yF(x) or rhs == yF(x):
            G = rhs if lhs == yF(x) else lhs
            if G.has(yF):
                raise ValueError(f"{fname}: the dependent-variable transformation must express {yname}[{x}]")
            Y = G
            continue
        if psi is not None:
            raise NotImplementedError(f"{fname}: only one independent-variable transformation is supported")
        phi, psis = _indep_map(lhs, rhs, x, tv, fname)
        psi = psis[0]
        if phi is not None and sp.simplify(phi.subs(x, psi) - tv) != 0:
            raise NotImplementedError(f"{fname}: the transformation could not be inverted")
    if psi is None:
        if tv != x:
            raise ValueError(f"{fname}: without an independent-variable transformation the new variable must be {x}")
        psi = x
    if Y is None:
        Ynew = uF(tv)
    else:
        if tv != x and Y.has(uF(x)):
            Y = Y.subs(uF(x), uF(tv))
        Ynew = Y.subs(x, psi)
        if not Ynew.has(uF(tv)):
            raise ValueError(f"{fname}: the transformation must involve {uu.name}[{tt.name}]")
    E = SX(eqn, fname)
    if not isinstance(E, sp.Equality):
        E = sp.Eq(E, 0)
    order = 0
    for d in E.atoms(sp.Derivative):
        if d.expr == yF(x):
            if d.variables != (x,) * len(d.variables):
                raise NotImplementedError(f"{fname}: partial differential equations are not supported")
            order = max(order, len(d.variables))
        elif d.expr.has(yF):
            raise NotImplementedError(f"{fname}: derivatives of composite expressions are not supported")
    dpsi = sp.diff(psi, tv)
    if dpsi == 0:
        raise ValueError(f"{fname}: the transformation is degenerate")
    ders = [Ynew]
    for _k in range(order):
        ders.append(sp.diff(ders[-1], tv) / dpsi)

    def tr_side(e):
        reps = {}
        for d in e.atoms(sp.Derivative):
            if d.expr == yF(x):
                reps[d] = ders[len(d.variables)]
        e = e.xreplace(reps)
        e = e.xreplace({yF(x): Ynew})
        if e.has(yF):
            raise NotImplementedError(f"{fname}: {yname} must appear only as {yname}[{x}] and its derivatives")
        return e.subs(x, psi) if psi != x else e
    lhs_, rhs_ = tr_side(E.lhs), tr_side(E.rhs)
    lhs_, rhs_ = _simp(lhs_), _simp(rhs_)
    unew = sp.Function(uu.name)

    def back(e):
        e = e.replace(lambda a: isinstance(a, AppliedUndef) and a.func == uF, lambda a: unew(*a.args))
        return e.subs(tv, tplain) if tv != x else e
    lhs_, rhs_ = back(lhs_), back(rhs_)
    head = dsolve.head
    return Expr(head, Expr("Equal", _wlf(lhs_), _wlf(rhs_)),
                Expr(uu.name, tt) if applied else uu, tt)
