"""Coverage batch 3: linear algebra with Wolfram Language semantics.

Exact and symbolic matrices are handled exactly with SymPy; matrices containing machine numbers use NumPy
(and SciPy where noted); arbitrary-precision matrices are evaluated exactly and rounded to the input precision.
Structured or sparse results (``SparseArray``, ``LeviCivitaTensor``, ``VandermondeMatrix``, ``CauchyMatrix``)
are returned as the equivalent dense (``Normal``) array; see each name's notes.

This module also replaces three kernel builtins (``Dot``, ``Det``, ``Inverse``) with Wolfram-correct versions
and extends ``Eigenvalues``/``Eigenvectors`` with the ``k`` argument.
"""

from __future__ import annotations
import itertools
from functools import reduce

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import (Unevaluated, fn, record, S, R, L, rule, sym, is_list, int_arg, pos_int, nonneg_int, list_arg,
                   variable, mat_out)
from .wl_common import split_options, ev, apply
from . import linalg as LA

AREA = "linear algebra (batch 3)"


# ─────────────────────────── shared helpers (kernel_linalg) ───────────────────────────

from ..kernel_linalg import (dims, to_nested, _kind, _out_nested, _np, _npmat, _np_out, _zero,  # noqa: E402
                             _expand_poly, _is_matrix, _tscale)


def _matrix(x, fname, square=False):
    """SymPy matrix for a matrix argument; a non-list leaves the call unevaluated."""
    if not is_list(x):
        raise Unevaluated()
    d = dims(x)
    if d is None or len(d) != 2 or d[1] == 0:
        raise ValueError(f"{fname}: argument must be a rectangular matrix")
    M = sp.Matrix([[S(v) for v in r.args] for r in x.args])
    if square and M.rows != M.cols:
        raise ValueError(f"{fname}: matrix must be square")
    return M



record("Dot", AREA, "fixed",
       "kernel builtin replaced in batch 3: any number of arguments and tensors of any rank (contracts the last "
       "index of each factor with the first of the next); symbolic factors leave Dot unevaluated (legacy raised); "
       "Dot[x] = x; incompatible shapes raise; infix a.b parses as Dot")

record("DotProduct", AREA, "fixed", "MikoshiLang alias of Dot for two vectors (now the batch-3 Dot)")

record("Det", AREA, "fixed",
       "kernel builtin replaced in batch 3: non-square input raises a clear error (legacy raised SymPy's "
       "NonSquareMatrixError), a symbol leaves Det unevaluated (legacy raised TypeError), machine matrices use "
       "LAPACK, arbitrary-precision matrices keep their precision, symbolic determinants are expanded")

record("Inverse", AREA, "fixed",
       "kernel builtin replaced in batch 3: a symbol leaves Inverse unevaluated (legacy raised), non-square and "
       "singular matrices raise clear errors, machine matrices use LAPACK, symbolic entries are cancelled")


# ─────────────────────────── transposes, parts of matrices ───────────────────────────

def _conj_leaves(x):
    if is_list(x):
        return L(*(_conj_leaves(a) for a in x.args))
    v = S(x)
    if isinstance(v, sp.Basic):
        return R(sp.conjugate(v))
    return ev(Expr("Conjugate", x))


@fn("ConjugateTranspose", area=AREA, status="added",
    notes="Conjugate[Transpose[m]] (and ConjugateTranspose[m, perm]); a vector or non-array raises like Transpose")
def _conjtranspose(m: ListOf[Expr], *perm: Expr) -> Expr:
    if not is_list(m):
        raise Unevaluated()
    t = ev(Expr("Transpose", m, *perm))
    if isinstance(t, Expr) and t.head == "Transpose":
        raise Unevaluated()
    return _conj_leaves(t)


def _offset(k, fname):
    if not K.is_int(k):
        raise Unevaluated() if isinstance(k, (Symbol, Expr)) else ValueError(f"{fname}: offset must be an integer")
    return int(k)


def _rows(m, fname):
    if not is_list(m):
        raise Unevaluated()
    d = dims(m)
    if d is None or len(d) < 2:
        raise ValueError(f"{fname}: argument must be a matrix")
    return [list(r.args) for r in m.args], d


@fn("Diagonal", area=AREA, status="added",
    notes="Diagonal[m] = {m[[1,1]], m[[2,2]], ...} (rectangular matrices too); Diagonal[m, k] is the k-th "
          "superdiagonal (k < 0: subdiagonal); entries are returned unevaluated-as-given")
def _diagonal(m: ListOf[Expr], k: Integer = 0) -> ListOf[Expr]:
    rows, d = _rows(m, "Diagonal")
    k = _offset(k, "Diagonal")
    out = []
    i = max(0, -k)
    while i < d[0] and i + k < d[1]:
        out.append(rows[i][i + k])
        i += 1
    return L(*out)


def _triangularize(m, k, keep, fname):
    rows, d = _rows(m, fname)
    if len(d) != 2:
        raise ValueError(f"{fname}: argument must be a matrix")
    k = _offset(k, fname)
    return L(*(L(*(rows[i][j] if keep(j - i, k) else 0 for j in range(d[1]))) for i in range(d[0])))


@fn("UpperTriangularize", area=AREA, status="added",
   notes="zeros below the k-th diagonal (keeps m[[i,j]] with j - i >= k; default k = 0); rectangular matrices")
def _upper_triangularize(m: MatrixOf[Expr], k: Integer = 0) -> MatrixOf[Expr]:
    return _triangularize(m, k, lambda dj, kk: dj >= kk, "UpperTriangularize")


@fn("LowerTriangularize", area=AREA, status="added",
   notes="zeros above the k-th diagonal (keeps m[[i,j]] with j - i <= k; default k = 0); rectangular matrices")
def _lower_triangularize(m: MatrixOf[Expr], k: Integer = 0) -> MatrixOf[Expr]:
    return _triangularize(m, k, lambda dj, kk: dj <= kk, "LowerTriangularize")


# ─────────────────────────── spectra ───────────────────────────

def _take_k(items, k, fname):
    n = len(items)
    if isinstance(k, Expr) and k.head == "UpTo" and len(k.args) == 1:
        c = nonneg_int(k.args[0], fname, "UpTo bound")
        return items[:min(c, n)]
    c = int_arg(k, fname, "count")
    if abs(c) > n:
        raise ValueError(f"{fname}: cannot take {abs(c)} of {n} values")
    return items[:c] if c >= 0 else items[n + c:]


def _generalized(m, fname):
    if is_list(m) and len(m.args) == 2 and all(_is_matrix(x) for x in m.args) and not _is_matrix(m):
        raise NotImplementedError(f"{fname}: generalized eigenproblems {{m, a}} are not supported")


def _eigvals(m):
    _generalized(m, "Eigenvalues")
    return LA._eigenvalues(m)


def _eigvecs(m):
    _generalized(m, "Eigenvectors")
    return LA._eigenvectors(m)


EIG_K = (" Eigenvalues[m, k] gives the k largest in absolute value (k < 0: the |k| smallest, i.e. "
         "Take[Eigenvalues[m], k]); Eigenvalues[m, UpTo[k]]; generalized {m, a} raises")


@fn("Eigenvalues", area="linear algebra", status="fixed",
    notes="eigenvalues with multiplicity in Wolfram order (decreasing absolute value; legacy returned SymPy's "
          "dict order)." + EIG_K)
def _eigenvalues_k(m: ListOf[Expr], k: Expr) -> ListOf[Expr]:
    return L(*_take_k(list(_eigvals(m).args), k, "Eigenvalues"))


@fn("Eigenvectors", area="linear algebra", status="fixed",
    notes="ordered like Eigenvalues, exact vectors scaled so the last non-zero entry is 1, zero vectors pad "
          "defective matrices (Wolfram); machine matrices give unit vectors. Eigenvectors[m, k] and UpTo[k] as "
          "for Eigenvalues")
def _eigenvectors_k(m: ListOf[Expr], k: Expr) -> ListOf[Expr]:
    return L(*_take_k(list(_eigvecs(m).args), k, "Eigenvectors"))


record("Eigenvalues", "linear algebra", "fixed",
       "eigenvalues with multiplicity in Wolfram order (decreasing absolute value; legacy returned SymPy's dict "
       "order)." + EIG_K)
record("Eigenvectors", "linear algebra", "fixed",
       "ordered like Eigenvalues, exact vectors scaled so the last non-zero entry is 1, zero vectors pad defective "
       "matrices (Wolfram); machine matrices give unit vectors; Eigenvectors[m, k] and UpTo[k] as for Eigenvalues")


@fn("Eigensystem", area=AREA, status="added",
    notes="{Eigenvalues[m], Eigenvectors[m]} with matching order (decreasing absolute value); Eigensystem[m, k] "
          "and UpTo[k]; generalized {m, a} raises")
def _eigensystem(m: ListOf[Expr], *k: Expr) -> ListOf[Expr]:
    if len(k) > 1:
        raise ValueError("Eigensystem called with too many arguments")
    vals, vecs = list(_eigvals(m).args), list(_eigvecs(m).args)
    if k:
        vals, vecs = _take_k(vals, k[0], "Eigensystem"), _take_k(vecs, k[0], "Eigensystem")
    return L(L(*vals), L(*vecs))


def _sv_exact(M, fname):
    if M.free_symbols:
        raise NotImplementedError(f"{fname}: singular values of symbolic matrices are not supported")
    A = M.H * M if M.cols <= M.rows else M * M.H
    vals = []
    for v, mult in A.eigenvals().items():
        s = sp.sqrt(sp.simplify(v))
        vals.extend([sp.simplify(s)] * mult)
    vals.sort(key=lambda v: -float(sp.N(v, 30)))
    return vals


@fn("SingularValueList", area=AREA, status="added",
    notes="non-zero singular values in decreasing order (exact for exact matrices, LAPACK for machine matrices, "
          "where values below 100*10^-p times the largest are dropped); SingularValueList[m, k] (k < 0: smallest) "
          "and UpTo[k] use all Min[Dimensions[m]] values, zeros included; Tolerance -> t keeps values >= t times "
          "the largest; symbolic and generalized {m, a} inputs raise")
def _svlist(m: ListOf[Expr], *rest: Expr) -> ListOf[Expr]:
    _generalized(m, "SingularValueList")
    args, opts = split_options(rest, {"Tolerance"}, "SingularValueList")
    if len(args) > 1:
        raise ValueError("SingularValueList called with too many arguments")
    M = _matrix(m, "SingularValueList")
    kind = _kind(m)
    if kind == "machine":
        np = _np()
        vals = [float(v) for v in np.linalg.svd(_npmat(M), compute_uv=False)]
        default_tol = 100 * 2.0 ** -52
    else:
        vals = _sv_exact(M, "SingularValueList")
        default_tol = 0
    tol = opts.get("Tolerance")
    if args:
        out = _take_k(vals, args[0], "SingularValueList")
    else:
        t = default_tol if tol is None else float(sym(tol, "SingularValueList"))
        top = float(abs(complex(sp.N(vals[0])))) if vals else 0.0
        if kind == "machine":
            out = [v for v in vals if v > t * top] if t > 0 else list(vals)
        elif tol is None:
            out = [v for v in vals if _zero(v) is not True]
        else:
            out = [v for v in vals if t == 0 or float(sp.N(v)) >= t * top]
    if kind == "machine":
        return L(*out)
    return _out_nested(out, kind)


@fn("JordanDecomposition", area=AREA, status="added",
    notes="{s, j} with m == s.j.Inverse[s] and j in Jordan normal form (exact, SymPy); a machine matrix must be "
          "numerically diagonalizable (eigenvector matrix condition number < 10^8) and gives "
          "{eigenvectors as columns, diagonal eigenvalues}; otherwise it raises. The order of the Jordan blocks and "
          "the scaling of s can differ from Wolfram's (the decomposition is not unique)")
def _jordan(m: MatrixOf[Expr]) -> ListOf[Expr]:
    M = _matrix(m, "JordanDecomposition", square=True)
    kind = _kind(m)
    if kind == "machine":
        np = _np()
        vals, vecs = np.linalg.eig(_npmat(M))
        if np.linalg.cond(vecs) > 1e8:
            raise NotImplementedError("JordanDecomposition: this machine-precision matrix is not numerically "
                                      "diagonalizable; use exact input")
        return L(_np_out(vecs), _np_out(np.diag(vals)))
    if kind != "exact":
        raise NotImplementedError("JordanDecomposition: arbitrary-precision matrices are not supported; use exact "
                                  "input")
    P, J = M.jordan_form()
    return L(mat_out(P.applyfunc(sp.simplify)), mat_out(J.applyfunc(sp.simplify)))


# ─────────────────────────── matrix functions ───────────────────────────

def _fn_of(f, t):
    """SymPy expression for f[t]."""
    from ..math_ops import to_sympy
    r = ev(K.call(f, Symbol(t.name)))
    v = to_sympy(r)
    if not isinstance(v, sp.Basic):
        raise ValueError("MatrixFunction: f must give a scalar")
    return v


@fn("MatrixFunction", area=AREA, status="added",
    notes="MatrixFunction[f, m] for a square matrix and any scalar function f (symbol or pure function): exact "
          "via the Jordan form (SymPy analytic_func, derivatives of f on Jordan blocks); machine matrices via the "
          "eigendecomposition, which must be well-conditioned (else raises); f is evaluated by MikoshiLang")
def _matrixfunction(f: Fn, m: MatrixOf[Expr]) -> Expr:
    M = _matrix(m, "MatrixFunction", square=True)
    kind = _kind(m)
    if kind == "machine":
        np = _np()
        vals, vecs = np.linalg.eig(_npmat(M))
        if np.linalg.cond(vecs) > 1e8:
            raise NotImplementedError("MatrixFunction: this machine-precision matrix is not numerically "
                                      "diagonalizable; use exact input")
        fv = []
        for v in vals:
            r = ev(K.call(f, R(complex(v)) if abs(complex(v).imag) > 0 else float(complex(v).real)))
            try:
                fv.append(complex(sp.N(S(r))))
            except (TypeError, ValueError):
                raise ValueError(f"MatrixFunction: f did not give a number at the eigenvalue {v}")
        out = vecs @ np.diag(fv) @ np.linalg.inv(vecs)
        if not np.iscomplexobj(_npmat(M)) and np.all(np.abs(out.imag) <= 1e-12 * max(1.0, np.abs(out).max())):
            out = out.real
        return _np_out(out)
    t = sp.Dummy("t")
    expr = _fn_of(f, Symbol("MatrixFunction$t")).subs(sp.Symbol("MatrixFunction$t"), t)
    try:
        res = M.analytic_func(expr, t)
    except Exception as exc:
        raise NotImplementedError(f"MatrixFunction: could not compute the exact matrix function ({exc})")
    res = res.applyfunc(sp.simplify)
    if res.has(sp.nan, sp.zoo):
        raise ValueError("MatrixFunction: f is not defined at an eigenvalue of m")
    return _out_nested(res.tolist(), kind)


@fn("MatrixMinimalPolynomial", area=AREA, status="added",
    notes="the monic polynomial p of least degree with p(m) == 0 (exact; Krylov dependence of I, m, m^2, ...); "
          "inexact matrices raise")
def _minpoly(m: MatrixOf[Expr], x: Var) -> Expr:
    M = _matrix(m, "MatrixMinimalPolynomial", square=True)
    xv = variable(x, "MatrixMinimalPolynomial")
    if _kind(m) != "exact":
        raise NotImplementedError("MatrixMinimalPolynomial: exact input is required")
    n = M.rows
    powers = [sp.eye(n)]
    for k in range(1, n + 1):
        powers.append((powers[-1] * M).applyfunc(sp.expand))
        A = sp.Matrix.hstack(*[p.reshape(n * n, 1) for p in powers])
        ns = A.nullspace(simplify=True)
        if ns:
            c = ns[0]
            c = c / c[k]
            return R(sp.expand(sum(sp.simplify(c[i]) * xv ** i for i in range(k + 1))))
    raise ValueError("MatrixMinimalPolynomial: failed to find the minimal polynomial")  # pragma: no cover


# ─────────────────────────── determinants and friends ───────────────────────────

@fn("Adjugate", area=AREA, status="added",
    notes="the adjugate (transposed cofactor matrix), Det[m] Inverse[m] for invertible m, defined for singular m "
          "too; exact (machine matrices are computed exactly from their binary values and rounded)")
def _adjugate(m: MatrixOf[Expr]) -> ListOf[Expr]:
    M = _matrix(m, "Adjugate", square=True)
    kind = _kind(m)
    if M.rows == 1:
        return L(L(1))
    A = M.adjugate(method="berkowitz")
    if kind == "exact":
        A = A.applyfunc(_expand_poly)
    return _out_nested(A.tolist(), kind)


@fn("Minors", area=AREA, status="added",
    notes="Minors[m, k]: determinants of all k x k submatrices, rows and columns subsets in lexicographic order "
          "(Wolfram); Minors[m] = Minors[m, Min[Dimensions[m]] - 1]; Minors[m, k, f] applies f instead of Det; "
          "Minors[m, 0] = {{1}}")
def _minors(m: MatrixOf[Expr], *rest: Expr) -> ListOf[Expr]:
    if len(rest) > 2:
        raise ValueError("Minors called with too many arguments")
    M = _matrix(m, "Minors")
    k = rest[0] if rest else min(M.rows, M.cols) - 1
    k = nonneg_int(k, "Minors", "size")
    if k > min(M.rows, M.cols):
        raise ValueError("Minors: k exceeds the matrix dimensions")
    f = rest[1] if len(rest) == 2 else None
    rows = list(itertools.combinations(range(M.rows), k))
    cols = list(itertools.combinations(range(M.cols), k))
    if f is not None:
        src = [list(r.args) for r in m.args]
        return L(*(L(*(apply(f, L(*(L(*(src[i][j] for j in c)) for i in r))) for c in cols)) for r in rows))
    kind = _kind(m)
    out = [[_expand_poly(M.extract(list(r), list(c)).det(method="berkowitz")) if k else sp.Integer(1)
            for c in cols] for r in rows]
    return _out_nested(out, kind)


@fn("Permanent", area=AREA, status="added",
    notes="sum over permutations of products of entries (SymPy Matrix.per); square matrices")
def _permanent(m: MatrixOf[Expr]) -> Expr:
    M = _matrix(m, "Permanent", square=True)
    return _out_nested(_expand_poly(sp.sympify(M.per())), _kind(m))


# ─────────────────────────── projections, Gram-Schmidt, least squares ───────────────────────────

def _vector(x, fname):
    if not is_list(x):
        raise Unevaluated()
    if any(is_list(a) for a in x.args) or not x.args:
        raise ValueError(f"{fname}: arguments must be non-empty vectors")
    return [S(a) for a in x.args]


def _cdot(u, v):
    """Conjugate[u].v"""
    return sp.Add(*(sp.conjugate(a) * b for a, b in zip(u, v)))


@fn("Projection", area=AREA, status="added",
    notes="Projection[u, v] = (u.Conjugate[v]/v.Conjugate[v]) v; Projection[u, v, f] uses the inner product "
          "f[u, v]/f[v, v] v; projection onto the zero vector raises")
def _projection(u: Expr, v: Expr, *f: Expr) -> Expr:
    if f:
        if len(f) > 1:
            raise ValueError("Projection called with too many arguments")
        num, den = apply(f[0], u, v), apply(f[0], v, v)
        if den == 0:
            raise ValueError("Projection: cannot project onto a vector of zero norm")
        return ev(Expr("Times", Expr("Times", num, Expr("Power", den, -1)), v))
    a, b = _vector(u, "Projection"), _vector(v, "Projection")
    if len(a) != len(b):
        raise ValueError("Projection: vectors must have the same length")
    den = _cdot(b, b)
    if _zero(den) is True:
        raise ValueError("Projection: cannot project onto the zero vector")
    c = sp.simplify(_cdot(b, a) / den)
    return _out_nested([sp.simplify(c * x) for x in b], _kind(u, v))


@fn("Orthogonalize", area=AREA, status="added",
    notes="Gram-Schmidt orthonormal basis using the standard inner product Conjugate[u].v; the output has as "
          "many vectors as the input, linearly dependent vectors become zero vectors; machine vectors whose "
          "residual norm is at most 100 $MachineEpsilon (about 2.2*10^-14) of the original are treated as dependent (Tolerance -> t overrides); "
          "Orthogonalize[list, f] uses the inner product f on arbitrary elements")
def _orthogonalize(vs: ListOf[Expr], *rest: Expr) -> ListOf[Expr]:
    args, opts = split_options(rest, {"Tolerance", "Method"}, "Orthogonalize")
    if "Method" in opts and not (isinstance(opts["Method"], str) and opts["Method"] == "GramSchmidt"):
        raise NotImplementedError("Orthogonalize: only Method -> \"GramSchmidt\" is supported")
    items = list_arg(vs, "Orthogonalize")
    if args:
        if len(args) > 1:
            raise ValueError("Orthogonalize called with too many arguments")
        return _orth_general(items, args[0])
    if not items:
        return L()
    vecs = [_vector(v, "Orthogonalize") for v in items]
    if len({len(v) for v in vecs}) != 1:
        raise ValueError("Orthogonalize: vectors must have the same length")
    kind = _kind(vs)
    tol = opts.get("Tolerance")
    tol = 100 * 2.0 ** -52 if tol is None else float(sym(tol, "Orthogonalize"))
    n = len(vecs[0])
    basis, out = [], []
    for v in vecs:
        w = list(v)
        for e in basis:
            c = _cdot(e, w)
            w = [wi - c * ei for wi, ei in zip(w, e)]
        if kind == "exact":
            w = [sp.simplify(x) for x in w]
            nn = sp.sqrt(sp.simplify(_cdot(w, w)))
            if _zero(nn) is True:
                out.append([sp.Integer(0)] * n)
                continue
            e = [sp.simplify(x / nn) for x in w]
        else:
            nn = sp.sqrt(_cdot(w, w)).evalf(30)
            orig = sp.sqrt(_cdot(v, v)).evalf(30)
            if orig == 0 or abs(nn) <= tol * abs(orig):
                out.append([sp.Integer(0)] * n)
                continue
            e = [x / nn for x in w]
        basis.append(e)
        out.append(e)
    return _out_nested(out, kind)


def _orth_general(items, f):
    basis, out = [], []
    for v in items:
        w = v
        for e in basis:
            c = apply(f, e, w)
            w = ev(Expr("Plus", w, Expr("Times", -1, c, e)))
        nn2 = apply(f, w, w)
        z = _zero(S(nn2))
        if z is True:
            out.append(ev(Expr("Times", 0, w)))
            continue
        e = ev(Expr("Simplify", Expr("Times", w, Expr("Power", nn2, R(sp.Rational(-1, 2))))))
        basis.append(e)
        out.append(e)
    return L(*out)


@fn("LeastSquares", area=AREA, status="added",
    notes="the x minimising Norm[m.x - b] (minimum-norm solution when several exist, = PseudoInverse[m].b); b a "
          "vector or matrix; exact for exact input, LAPACK lstsq for machine input")
def _leastsquares(m: MatrixOf[Expr], b: ListOf[Expr]) -> Expr:
    M = _matrix(m, "LeastSquares")
    if not is_list(b):
        raise Unevaluated()
    vec = not _is_matrix(b)
    B = sp.Matrix(_vector(b, "LeastSquares")) if vec else _matrix(b, "LeastSquares")
    if B.rows != M.rows:
        raise ValueError("LeastSquares: dimensions of m and b do not match")
    kind = _kind(m, b)
    if kind == "machine":
        np = _np()
        x = np.linalg.lstsq(_npmat(M), _npmat(B), rcond=None)[0]
        return _np_out(x[:, 0] if vec else x)
    X = (M.pinv() * B).applyfunc(sp.simplify)
    return _out_nested([X[i, 0] for i in range(X.rows)] if vec else X.tolist(), kind)


# ─────────────────────────── constructors ───────────────────────────

def _size2(n, fname):
    if is_list(n):
        a = list_arg(n, fname)
        if len(a) != 2:
            raise ValueError(f"{fname}: size must be n or {{m, n}}")
        return pos_int(a[0], fname, "size"), pos_int(a[1], fname, "size")
    k = pos_int(n, fname, "size")
    return k, k


@fn("HilbertMatrix", area=AREA, status="added",
    notes="entries 1/(i + j - 1); HilbertMatrix[n] and HilbertMatrix[{m, n}]; exact (use N for machine numbers)")
def _hilbert(n: PosInteger | ListOf[PosInteger]) -> MatrixOf[Expr]:
    r, c = _size2(n, "HilbertMatrix")
    return mat_out(sp.Matrix(r, c, lambda i, j: sp.Rational(1, i + j + 1)))


def _entries(x, fname):
    items = list_arg(x, fname, min_len=1)
    if any(is_list(v) for v in items):
        raise ValueError(f"{fname}: expected a vector")
    return items


@fn("ToeplitzMatrix", area=AREA, status="added",
    notes="ToeplitzMatrix[n] (symmetric, successive integers), ToeplitzMatrix[c] (symmetric, first column c), "
          "ToeplitzMatrix[c, r] (first column c, first row r; r[[1]] is ignored and c[[1]] used, as in Wolfram); "
          "always a dense matrix (Wolfram's TargetStructure option is not supported)")
def _toeplitz(c: PosInteger | ListOf[Expr], r: ListOf[Expr] = None) -> ListOf[Expr]:
    if r is None and not is_list(c):
        n = pos_int(c, "ToeplitzMatrix", "size")
        c = L(*range(1, n + 1))
    col = _entries(c, "ToeplitzMatrix")
    row = col if r is None else _entries(r, "ToeplitzMatrix")
    return L(*(L(*(col[i - j] if i >= j else row[j - i] for j in range(len(row)))) for i in range(len(col))))


@fn("HankelMatrix", area=AREA, status="added",
    notes="HankelMatrix[n] (successive integers, zeros below the antidiagonal), HankelMatrix[c] (first column c, "
          "zeros below the antidiagonal), HankelMatrix[c, r] (first column c, last row r; r[[1]] is ignored and "
          "c[[-1]] used, as in Wolfram); always a dense matrix")
def _hankel(c: PosInteger | ListOf[Expr], r: ListOf[Expr] = None) -> ListOf[Expr]:
    if r is None and not is_list(c):
        n = pos_int(c, "HankelMatrix", "size")
        c = L(*range(1, n + 1))
    col = _entries(c, "HankelMatrix")
    m = len(col)
    row = [0] * m if r is None else _entries(r, "HankelMatrix")
    n = len(row)

    def h(i, j):          # 0-based
        s = i + j
        return col[s] if s < m else row[s - m + 1]
    return L(*(L(*(h(i, j) for j in range(n))) for i in range(m)))


@fn("VandermondeMatrix", area=AREA, status="added",
    notes="rows {1, x_i, x_i^2, ...}: VandermondeMatrix[{x1, ..., xn}] (n x n) and VandermondeMatrix[x, k] "
          "(n x k), returned as the dense matrix (Wolfram returns a structured array whose Normal this is); "
          "confluent (repeated) nodes raise")
def _vandermonde(x: ListOf[Expr], k: Expr = None) -> ListOf[Expr]:
    xs = _entries(x, "VandermondeMatrix")
    cols = len(xs) if k is None else pos_int(k, "VandermondeMatrix", "column count")
    for i in range(len(xs)):
        for j in range(i):
            if K.same(xs[i], xs[j]) or _zero(S(xs[i]) - S(xs[j])) is True:
                raise NotImplementedError("VandermondeMatrix: confluent (repeated) nodes are not supported")
    kind = _kind(x)
    return _out_nested([[S(v) ** j if j else sp.Integer(1) for j in range(cols)] for v in xs], kind)


@fn("CauchyMatrix", area=AREA, status="added",
    notes="entries 1/(x_i + y_j) (CauchyMatrix[x] = CauchyMatrix[x, x]), returned as the dense matrix; a zero "
          "denominator raises")
def _cauchy(x: ListOf[Expr], y: ListOf[Expr] = None) -> ListOf[Expr]:
    xs = [S(v) for v in _entries(x, "CauchyMatrix")]
    ys = xs if y is None else [S(v) for v in _entries(y, "CauchyMatrix")]
    out = []
    for a in xs:
        row = []
        for b in ys:
            if _zero(a + b) is True:
                raise ValueError("CauchyMatrix: x_i + y_j must be non-zero")
            row.append(1 / (a + b))
        out.append(row)
    return _out_nested(out, _kind(x, *(() if y is None else (y,))))


@fn("FourierMatrix", area=AREA, status="added",
    notes="FourierMatrix[n]: entries Exp[2 Pi I (r - 1)(s - 1)/n]/Sqrt[n] (Wolfram's default FourierParameters "
          "{0, 1}); exact")
def _fouriermatrix(n: PosInteger) -> MatrixOf[Expr]:
    k = pos_int(n, "FourierMatrix", "size")
    return mat_out(sp.Matrix(k, k, lambda r, s: sp.exp(2 * sp.pi * sp.I * ((r * s) % k) / k) / sp.sqrt(k)))


def _numvec(v, fname):
    xs = _vector(v, fname)
    if any(x.free_symbols for x in xs):
        raise Unevaluated()
    return xs


def _norm2(xs):
    return sp.sqrt(sp.Add(*(sp.Abs(a) ** 2 for a in xs)))


def _simp_out(M, kind):
    return _out_nested(M.applyfunc(lambda v: sp.expand(v) if v.free_symbols else sp.simplify(v)).tolist(), kind)


@fn("RotationMatrix", area=AREA, status="added",
    notes="RotationMatrix[t] (2D, counterclockwise), RotationMatrix[t, w] (3D, right-handed about the numeric "
          "axis w), RotationMatrix[t, {u, v}] (n-D, in the plane of u and v, from u towards v), RotationMatrix[{u, "
          "v}] (rotation taking the direction of u to that of v, in their plane; antiparallel vectors raise)")
def _rotation(t: Numeric | ListOf[Expr], w: ListOf[Expr] = None) -> ListOf[Expr]:
    if w is None and is_list(t):
        return _rotation_uv(*_pair(t, "RotationMatrix"), None)
    th = sym(t, "RotationMatrix")
    kind = _kind(t) if w is None else _kind(t, w)
    if w is None:
        return _simp_out(sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]]), kind)
    if is_list(w) and w.args and all(is_list(a) for a in w.args):
        return _rotation_uv(*_pair(w, "RotationMatrix"), th, kind)
    ax = _numvec(w, "RotationMatrix")
    if len(ax) != 3:
        raise ValueError("RotationMatrix: the axis must be a 3D vector (use {u, v} for other dimensions)")
    nn = _norm2(ax)
    if nn == 0:
        raise ValueError("RotationMatrix: the axis must be non-zero")
    x, y, z = [a / nn for a in ax]
    c, s = sp.cos(th), sp.sin(th)
    Kx = sp.Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    u = sp.Matrix([x, y, z])
    return _simp_out(c * sp.eye(3) + s * Kx + (1 - c) * u * u.T, kind)


def _pair(x, fname):
    items = list_arg(x, fname)
    if len(items) != 2:
        raise ValueError(f"{fname}: expected a pair of vectors {{u, v}}")
    return _numvec(items[0], fname), _numvec(items[1], fname)


def _rotation_uv(u, v, th, kind=None):
    if len(u) != len(v):
        raise ValueError("RotationMatrix: u and v must have the same length")
    n = len(u)
    nu = _norm2(u)
    if nu == 0 or _norm2(v) == 0:
        raise ValueError("RotationMatrix: u and v must be non-zero")
    e1 = sp.Matrix([a / nu for a in u])
    vv = sp.Matrix(v)
    w = vv - (e1.T * vv)[0] * e1
    w = w.applyfunc(sp.simplify)
    nw = sp.simplify(_norm2(list(w)))
    if th is None:
        if nw == 0:
            if (e1.T * vv)[0] > 0:
                return mat_out(sp.eye(n))
            raise ValueError("RotationMatrix: the rotation taking u to -u is not unique")
        c = sp.simplify((e1.T * vv)[0] / _norm2(v))
        s = sp.simplify(nw / _norm2(v))
    else:
        if nw == 0:
            raise ValueError("RotationMatrix: u and v must be linearly independent")
        c, s = sp.cos(th), sp.sin(th)
    e2 = w / nw
    Rm = sp.eye(n) + (c - 1) * (e1 * e1.T + e2 * e2.T) + s * (e2 * e1.T - e1 * e2.T)
    return _simp_out(Rm, kind or "exact")


@fn("ReflectionMatrix", area=AREA, status="added",
    notes="IdentityMatrix[n] - 2 v.ConjugateTranspose[v]/(Conjugate[v].v): reflection in the hyperplane "
          "normal to the numeric vector v")
def _reflection(v: ListOf[Numeric]) -> ListOf[Expr]:
    xs = _numvec(v, "ReflectionMatrix")
    d = sp.simplify(_cdot(xs, xs))
    if d == 0:
        raise ValueError("ReflectionMatrix: the normal vector must be non-zero")
    u = sp.Matrix(xs)
    return _simp_out(sp.eye(len(xs)) - 2 * u * u.H / d, _kind(v))


@fn("ScalingMatrix", area=AREA, status="added",
    notes="ScalingMatrix[{s1, ..., sn}] = DiagonalMatrix[{s1, ..., sn}]; ScalingMatrix[s, v] scales by s along "
          "the numeric direction v: IdentityMatrix + (s - 1) v.Transpose[v]/v.v")
def _scaling(s: Numeric | ListOf[Expr], v: ListOf[Expr] = None) -> ListOf[Expr]:
    if v is None:
        xs = _entries(s, "ScalingMatrix")
        n = len(xs)
        return L(*(L(*(xs[i] if i == j else 0 for j in range(n))) for i in range(n)))
    sc = sym(s, "ScalingMatrix")
    xs = _numvec(v, "ScalingMatrix")
    u = sp.Matrix(xs)
    d = (u.T * u)[0]
    if d == 0:
        raise ValueError("ScalingMatrix: the direction must be non-zero")
    return _simp_out(sp.eye(len(xs)) + (sc - 1) * u * u.T / d, _kind(s, v))


@fn("ShearingMatrix", area=AREA, status="added",
    notes="ShearingMatrix[t, v, n] = IdentityMatrix + Tan[t] Outer[Times, vp, Normalize[n]] where vp is the unit "
          "vector along the part of v perpendicular to n (Wolfram uses the projection for non-orthogonal v); "
          "determinant 1")
def _shearing(t: Expr, v: ListOf[Numeric], nrm: ListOf[Numeric]) -> ListOf[Expr]:
    th = sym(t, "ShearingMatrix")
    a, b = _numvec(v, "ShearingMatrix"), _numvec(nrm, "ShearingMatrix")
    if len(a) != len(b):
        raise ValueError("ShearingMatrix: v and n must have the same length")
    nb = _norm2(b)
    if nb == 0:
        raise ValueError("ShearingMatrix: the normal must be non-zero")
    nn = sp.Matrix(b) / nb
    vp = sp.Matrix(a) - (nn.T * sp.Matrix(a))[0] * nn
    nv = sp.simplify(_norm2(list(vp)))
    if nv == 0:
        raise ValueError("ShearingMatrix: v must not be parallel to n")
    return _simp_out(sp.eye(len(a)) + sp.tan(th) * (vp / nv) * nn.T, _kind(t, v, nrm))


def _radii(r, fname):
    if is_list(r):
        rs = [nonneg_int(x, fname, "radius") for x in list_arg(r, fname, min_len=1)]
    else:
        rs = [nonneg_int(r, fname, "radius")] * 2
    return rs


def _shape(r, w, fname, inside):
    rs = _radii(r, fname)
    if w is None:
        ws = [2 * x + 1 for x in rs]
    elif is_list(w):
        ws = [pos_int(x, fname, "size") for x in list_arg(w, fname)]
        if len(ws) != len(rs):
            if not is_list(r):
                rs = [rs[0]] * len(ws)
            else:
                raise ValueError(f"{fname}: radii and sizes must have the same length")
    else:
        ws = [pos_int(w, fname, "size")] * len(rs)
    for x, n in zip(rs, ws):
        if (n - (2 * x + 1)) % 2:
            raise NotImplementedError(f"{fname}: the region cannot be centred in a dimension of size {n}; use an "
                                      "odd size or a size of the same parity as 2r + 1")
    centre = [(n - 1) // 2 for n in ws]

    def build(level, idx):
        if level == len(ws):
            d = [i - c for i, c in zip(idx, centre)]
            return 1 if inside(d, rs) else 0
        return L(*(build(level + 1, idx + [i]) for i in range(ws[level])))
    return build(0, [])


def _box(d, rs):
    return all(abs(a) <= r for a, r in zip(d, rs))


def _cross(d, rs):
    return _box(d, rs) and sum(1 for a in d if a != 0) <= 1


@fn("BoxMatrix", area=AREA, status="added",
   notes="1s in a box extending r index positions to each side: BoxMatrix[r] is (2r+1) x (2r+1) ones, "
         "BoxMatrix[r, w] (w x w, box centred), BoxMatrix[{r1, r2, ...}] and sizes {w1, ...}; integer radii only; "
         "an off-centre box (w - 2r - 1 odd) raises")
def _box_matrix(r: NonNegInteger | ListOf[Expr], w: Expr = None) -> ListOf[Expr]:
    return _shape(r, w, "BoxMatrix", _box)


@fn("CrossMatrix", area=AREA, status="added",
   notes="1s on the central row and column (central lines in higher dimensions) extending r positions: "
         "CrossMatrix[r] is (2r+1) x (2r+1), CrossMatrix[r, w], CrossMatrix[{r1, ...}]; integer radii only")
def _cross_matrix(r: NonNegInteger | ListOf[Expr], w: Expr = None) -> ListOf[Expr]:
    return _shape(r, w, "CrossMatrix", _cross)


def _diamond(r: NonNegInteger | ListOf[Expr], w: Expr = None) -> ListOf[Expr]:
    rs = _radii(r, "DiamondMatrix")
    if len(set(rs)) != 1:
        raise NotImplementedError("DiamondMatrix: different radii per dimension are not supported")
    return _shape(r, w, "DiamondMatrix", lambda d, rr: sum(abs(a) for a in d) <= rr[0])


fn("DiamondMatrix", area=AREA, status="added",
   notes="1s where the Manhattan distance from the centre is at most r: DiamondMatrix[r] is (2r+1) x (2r+1), "
         "DiamondMatrix[r, w]; integer radii only, equal radii per dimension")(_diamond)


def _perm_sign(p):
    p = list(p)
    s = 1
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            s = -s
    return s


@fn("LeviCivitaTensor", area=AREA, status="added",
    notes="the rank-d totally antisymmetric tensor (Signature of the index list), as a dense array: "
          "LeviCivitaTensor[d] (Wolfram returns the equivalent SparseArray; Normal gives this array) and "
          "LeviCivitaTensor[d, List]")
def _levicivita(d: PosInteger, h: Expr = None) -> Expr:
    n = pos_int(d, "LeviCivitaTensor", "rank")
    if h is not None and not (isinstance(h, Symbol) and h.name in ("List", "SparseArray")):
        raise ValueError("LeviCivitaTensor: the second argument must be List")
    if n > 8:
        raise ValueError("LeviCivitaTensor: rank above 8 is too large for a dense array")

    def build(idx):
        if len(idx) == n:
            return _perm_sign(idx) if len(set(idx)) == n else 0
        return L(*(build(idx + [i]) for i in range(n)))
    return build([])


# ─────────────────────────── predicates ───────────────────────────

_PRED_OPTS = {"Tolerance", "SameTest", "Normalized"}


def _pred_matrix(x, rest, fname):
    """(M, kind, opts) for a matrix argument, or None when x is not a matrix (the predicate is then False)."""
    args, opts = split_options(rest, _PRED_OPTS, fname)
    if args:
        raise ValueError(f"{fname} called with too many arguments")
    if not _is_matrix(x):
        return None
    d = dims(x)
    if len(d) != 2:
        return None
    try:
        M = sp.Matrix([[S(v) for v in r.args] for r in x.args])
    except Exception:
        return None
    if any(not isinstance(v, sp.Basic) for v in M):
        return None
    return M, _kind(x), opts


def _tolerance(opts, kind, M):
    t = opts.get("Tolerance")
    if t is not None and not (isinstance(t, Symbol) and t.name == "Automatic"):
        return float(sym(t, "Tolerance"))
    if kind == "exact":
        return 0.0
    if kind == "machine":
        return 1e-12
    import mpmath
    return 10.0 ** (-(mpmath.libmp.prec_to_dps(kind) - 3))


def _all_zero(D, kind, tol, same_test=None, ref=None):
    """Every entry of D (a difference matrix) is zero: exactly, or within the tolerance (infinity norm)."""
    if same_test is not None:
        return all(apply(same_test, R(a), R(b)) is True for a, b in zip(ref[0], ref[1]))
    if kind == "exact" and tol == 0:
        return all(_zero(v) is True for v in D)
    try:
        rows = [sum(abs(complex(sp.N(D[i, j]))) for j in range(D.cols)) for i in range(D.rows)]
    except (TypeError, ValueError):
        return False
    return max(rows, default=0.0) <= tol


def _matrix_pred(name, test, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def impl(x: Expr, *rest: Expr) -> Boolean:
        got = _pred_matrix(x, rest, name)
        if got is None:
            return False
        M, kind, opts = got
        return bool(test(M, kind, opts))
    return impl


def _sq(M):
    return M.rows == M.cols


def _eqtest(A, B, kind, opts, M):
    st = opts.get("SameTest")
    if st is not None and not (isinstance(st, Symbol) and st.name == "Automatic"):
        return _all_zero(None, kind, 0, st, (list(A), list(B)))
    return _all_zero(A - B, kind, _tolerance(opts, kind, M))


_matrix_pred("SquareMatrixQ", lambda M, k, o: _sq(M),
             "True for a matrix with as many rows as columns, False for anything else")
_matrix_pred("HermitianMatrixQ", lambda M, k, o: _sq(M) and _eqtest(M, M.H, k, o, M),
             "m == ConjugateTranspose[m] (exact zero test; Tolerance -> t for approximate matrices, default "
             "1e-12 at machine precision; SameTest -> f); False for non-matrices")
_matrix_pred("AntisymmetricMatrixQ", lambda M, k, o: _sq(M) and _eqtest(M, -M.T, k, o, M),
             "m == -Transpose[m] (exact; Tolerance / SameTest as for HermitianMatrixQ)")
_matrix_pred("AntihermitianMatrixQ", lambda M, k, o: _sq(M) and _eqtest(M, -M.H, k, o, M),
             "m == -ConjugateTranspose[m] (exact; Tolerance / SameTest as for HermitianMatrixQ)")
_matrix_pred("NormalMatrixQ", lambda M, k, o: _sq(M) and _eqtest(M * M.H, M.H * M, k, o, M),
             "m.ConjugateTranspose[m] == ConjugateTranspose[m].m")


def _orthotest(M, kind, opts, conj):
    T = M.H if conj else M.T
    P = T * M if M.rows >= M.cols else M * T
    n = P.rows
    norm = opts.get("Normalized", True)
    if norm is False:
        D = sp.Matrix(n, n, lambda i, j: 0 if i == j else P[i, j])
        return _all_zero(D, kind, _tolerance(opts, kind, M)) and all(_zero(P[i, i]) is not True for i in range(n))
    return _eqtest(P, sp.eye(n), kind, opts, M)


_matrix_pred("OrthogonalMatrixQ", lambda M, k, o: _orthotest(M, k, o, False),
             "p x q matrix: Transpose[m].m == IdentityMatrix[q] when p >= q, m.Transpose[m] == IdentityMatrix[p] "
             "when p <= q (no conjugation, as in Wolfram); options Normalized -> False (only orthogonal columns/"
             "rows), Tolerance -> t (infinity norm of the defect), SameTest -> f")
_matrix_pred("UnitaryMatrixQ", lambda M, k, o: _orthotest(M, k, o, True),
             "as OrthogonalMatrixQ with ConjugateTranspose; options Normalized, Tolerance, SameTest")


def _hermitian_part(M):
    return ((M + M.H) / 2).applyfunc(sp.simplify)


def _sign_of(v):
    """+1, 0, -1 for a real number, None when it cannot be decided (symbolic)."""
    v = sp.simplify(sp.re(v)) if v.free_symbols else sp.re(v)
    if v.free_symbols:
        return None
    if _zero(v) is True:
        return 0
    try:
        f = sp.N(v, 50)
    except Exception:
        return None
    if f.is_positive:
        return 1
    if f.is_negative:
        return -1
    return None


def _definiteness(M, kind):
    """Set of possible signs of the quadratic form: returns (pd, psd, nd, nsd) as booleans (False if unknown)."""
    if not _sq(M):
        return (False,) * 4
    H = _hermitian_part(M)
    n = H.rows
    if kind == "machine":
        np = _np()
        w = np.linalg.eigvalsh(_npmat(H))
        tol = n * 2.0 ** -52 * max(1.0, float(np.abs(w).max()))
        return (bool(w.min() > tol), bool(w.min() >= -tol), bool(w.max() < -tol), bool(w.max() <= tol))
    if H.free_symbols:
        # a symbolic matrix is only "explicitly" definite when every principal minor has a provable sign
        pass
    lead = [_sign_of(H[:k, :k].det(method="berkowitz")) for k in range(1, n + 1)]
    pd = all(s == 1 for s in lead)
    nd = all(s == (-1) ** k for k, s in enumerate(lead, 1))
    psd = nsd = True
    for k in range(1, n + 1):
        for idx in itertools.combinations(range(n), k):
            s = _sign_of(H.extract(list(idx), list(idx)).det(method="berkowitz"))
            if s is None:
                psd = nsd = False
                break
            if s < 0:
                psd = False
            if s * (-1) ** k < 0:
                nsd = False
        if not (psd or nsd):
            break
    return pd, psd, nd, nsd


DEFN = ("; the quadratic form is Re[Conjugate[x].m.x], so only the Hermitian part (m + ConjugateTranspose[m])/2 "
        "matters (Wolfram); exact matrices use principal minors, machine matrices the eigenvalues of the "
        "Hermitian part with a tolerance of n*$MachineEpsilon*max|eigenvalue|; False for symbolic matrices unless "
        "every sign can be proved, and for non-matrices")
_matrix_pred("PositiveDefiniteMatrixQ", lambda M, k, o: _definiteness(M, k)[0],
             "Re[Conjugate[x].m.x] > 0 for all non-zero x" + DEFN)
_matrix_pred("PositiveSemidefiniteMatrixQ", lambda M, k, o: _definiteness(M, k)[1],
             "Re[Conjugate[x].m.x] >= 0 for all x" + DEFN)
_matrix_pred("NegativeDefiniteMatrixQ", lambda M, k, o: _definiteness(M, k)[2],
             "Re[Conjugate[x].m.x] < 0 for all non-zero x" + DEFN)
_matrix_pred("NegativeSemidefiniteMatrixQ", lambda M, k, o: _definiteness(M, k)[3],
             "Re[Conjugate[x].m.x] <= 0 for all x" + DEFN)


def _indefinite(M, kind):
    if not _sq(M):
        return False
    if M.free_symbols:
        H = _hermitian_part(M)
        if H.free_symbols:
            return False
    pd, psd, nd, nsd = _definiteness(M, kind)
    return not psd and not nsd


_matrix_pred("IndefiniteMatrixQ", lambda M, k, o: _indefinite(M, k),
             "the quadratic form takes both signs (neither positive nor negative semidefinite)" + DEFN)


def _diagonalizable(M, kind):
    if not _sq(M):
        return False
    if kind == "machine":
        np = _np()
        _vals, vecs = np.linalg.eig(_npmat(M))
        return bool(np.linalg.cond(vecs) < 1e8)
    if kind != "exact":
        M = M.applyfunc(lambda v: sp.nsimplify(v, rational=True))
    try:
        return bool(M.is_diagonalizable())
    except Exception:
        return False


_matrix_pred("DiagonalizableMatrixQ", lambda M, k, o: _diagonalizable(M, k),
             "m == s.d.Inverse[s] for a diagonal d (over the complex numbers): exact via SymPy; machine matrices "
             "test that the eigenvector matrix is well-conditioned (condition number < 10^8)")


# ─────────────────────────── tensors ───────────────────────────

def _outer_nested(a, b):
    if isinstance(a, list):
        return [_outer_nested(x, b) for x in a]
    return _tscale(a, b)


@fn("TensorProduct", area=AREA, status="added",
    notes="tensor (outer) product of explicit arrays, dimensions concatenated (Outer[Times, a, b] for arrays); "
          "numbers act as rank-0 tensors; TensorProduct[t] = t; symbolic factors leave it unevaluated")
def _tensorproduct(*ts: Expr) -> Expr:
    if not ts:
        raise Unevaluated()
    if len(ts) == 1:
        return ts[0]
    vals = []
    for t in ts:
        if is_list(t):
            if dims(t) is None:
                raise ValueError("TensorProduct: arguments must be rectangular arrays")
            vals.append(to_nested(t))
        else:
            v = S(t)
            if not isinstance(v, sp.Basic) or not v.is_number:
                raise Unevaluated()
            vals.append(v)
    acc = vals[0]
    for v in vals[1:]:
        acc = _outer_nested(acc, v)
    return _out_nested(acc, _kind(*ts))


@fn("TensorRank", area=AREA, status="added",
    notes="the number of indices of an explicit rectangular array (ArrayDepth), 0 for a number; symbols and "
          "ragged lists stay unevaluated")
def _tensorrank(t: Expr) -> Integer:
    if is_list(t):
        d = dims(t)
        if d is None:
            raise Unevaluated()
        return len(d)
    v = S(t)
    if isinstance(v, sp.Basic) and v.is_number:
        return 0
    raise Unevaluated()


@fn("TensorDimensions", area=AREA, status="added",
    notes="Dimensions of an explicit rectangular array ({} for a number); symbolic tensors stay unevaluated")
def _tensordims(t: Expr) -> ListOf[Expr]:
    if is_list(t):
        d = dims(t)
        if d is None:
            raise Unevaluated()
        return L(*d)
    v = S(t)
    if isinstance(v, sp.Basic) and v.is_number:
        return L()
    raise Unevaluated()


@fn("TensorContract", area=AREA, status="added",
    notes="TensorContract[t, {{s1, s2}, ...}] sums over each pair of slots of an explicit array; the remaining "
          "slots keep their order; paired slots must have equal dimensions and appear once")
def _tensorcontract(t: ListOf[Expr], pairs: ListOf[Expr]) -> Expr:
    if not is_list(t):
        raise Unevaluated()
    d = dims(t)
    if d is None:
        raise ValueError("TensorContract: the tensor must be a rectangular array")
    ps = list_arg(pairs, "TensorContract")
    if ps and not is_list(ps[0]):
        ps = [pairs]
    slots = []
    for p in ps:
        q = list_arg(p, "TensorContract")
        if len(q) != 2:
            raise ValueError("TensorContract: each contraction is a pair of slots {s1, s2}")
        a, b = (pos_int(x, "TensorContract", "slot") for x in q)
        if a > len(d) or b > len(d):
            raise ValueError(f"TensorContract: slot out of range for a rank-{len(d)} tensor")
        slots.append((a - 1, b - 1))
    used = [s for p in slots for s in p]
    if len(set(used)) != len(used):
        raise ValueError("TensorContract: each slot can be contracted only once")
    for a, b in slots:
        if d[a] != d[b]:
            raise ValueError("TensorContract: contracted slots must have equal dimensions")
    free = [i for i in range(len(d)) if i not in used]
    T = to_nested(t)

    def get(idx):
        v = T
        for i in idx:
            v = v[i]
        return v

    def entry(free_idx):
        total = sp.Integer(0)
        for c in itertools.product(*(range(d[a]) for a, _b in slots)):
            idx = [0] * len(d)
            for i, f in zip(free, free_idx):
                idx[i] = f
            for (a, b), k in zip(slots, c):
                idx[a] = idx[b] = k
            total += get(idx)
        return total

    def build(level, prefix):
        if level == len(free):
            return entry(prefix)
        return [build(level + 1, prefix + [i]) for i in range(d[free[level]])]
    return _out_nested(build(0, []), _kind(t))


@fn("TensorTranspose", area=AREA, status="added",
    notes="TensorTranspose[t, perm] for an explicit array: Transpose[t, perm] with perm a full permutation "
          "list of the slots (Cycles specifications are not supported)")
def _tensortranspose(t: ListOf[Expr], perm: Expr) -> Expr:
    if not is_list(t):
        raise Unevaluated()
    d = dims(t)
    p = [pos_int(x, "TensorTranspose", "slot") for x in list_arg(perm, "TensorTranspose")]
    if d is None or sorted(p) != list(range(1, len(d) + 1)):
        raise ValueError("TensorTranspose: perm must be a permutation of the slots of an array")
    return ev(Expr("Transpose", t, perm))


# ─────────────────────────── sparse arrays (dense representation) ───────────────────────────

def _default_zero(v, default):
    return K.same(v, default) or (default == 0 and isinstance(v, (int, float)) and not isinstance(v, bool)
                                  and v == 0)


def _positions(x, prefix=()):
    if is_list(x):
        for i, a in enumerate(x.args, 1):
            yield from _positions(a, prefix + (i,))
    else:
        yield prefix, x


@fn("ArrayRules", area=AREA, status="added",
    notes="ArrayRules[array] gives {pos -> value, ...} for the entries that differ from 0 (in lexicographic "
          "position order) followed by {_, _, ...} -> 0; ArrayRules[array, default]; works on explicit "
          "rectangular arrays (SparseArray evaluates to its dense array in MikoshiLang)")
def _arrayrules(x: ListOf[Expr], default: Expr = 0) -> ListOf[Expr]:
    if not is_list(x):
        raise Unevaluated()
    d = dims(x)
    if d is None:
        raise ValueError("ArrayRules: argument must be a rectangular array")
    out = [rule(L(*p), v) for p, v in _positions(x) if not _default_zero(v, default)]
    out.append(rule(L(*(Expr("Blank") for _ in d)), default))
    return L(*out)


def _sparse_rules(r):
    if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed"):
        lhs, rhs = r.args
        if is_list(lhs) and lhs.args and is_list(lhs.args[0]):
            vals = list(rhs.args) if is_list(rhs) else None
            if vals is None or len(vals) != len(lhs.args):
                raise ValueError("SparseArray: {pos1, pos2, ...} -> {v1, v2, ...} needs as many values as positions")
            return [rule(p, v) for p, v in zip(lhs.args, vals)]
        return [r]
    if is_list(r) and all(isinstance(a, Expr) and a.head in ("Rule", "RuleDelayed") for a in r.args):
        out = []
        for a in r.args:
            out.extend(_sparse_rules(a))
        return out
    return None


@fn("SparseArray", area=AREA, status="added",
    notes="SparseArray[{pos -> v, ...}], SparseArray[rules, dims], SparseArray[rules, dims, default] and "
          "SparseArray[list] (positions may be patterns such as {i_, i_} -> 1; the first matching rule wins). "
          "MikoshiLang has no sparse storage: the result is the equivalent dense array, i.e. Normal[SparseArray[...]] "
          "in Wolfram (Head is List, not SparseArray); Band[...] -> v specifications are expanded (see Band)")
def _sparsearray(spec: Expr, *rest: Expr) -> Expr:
    if len(rest) > 2:
        raise ValueError("SparseArray called with too many arguments")
    rules = _sparse_rules(spec)
    if rules is None:
        if is_list(spec) and dims(spec) is not None and not rest:
            return spec
        raise Unevaluated() if isinstance(spec, Symbol) else ValueError(
            "SparseArray: expected rules {pos -> value, ...} or an array")
    default = rest[1] if len(rest) == 2 else 0
    explicit = {}
    patterns = []
    for r in rules:
        lhs, rhs = r.args
        if isinstance(lhs, Expr) and lhs.head == "Band":
            patterns.append(("band", lhs, rhs))       # expanded once the dimensions are known (W08)
            continue
        if K.is_int(lhs):
            lhs = L(lhs)
        if is_list(lhs) and all(K.is_int(i) for i in lhs.args):
            key = tuple(int(i) for i in lhs.args)
            if any(i <= 0 for i in key):
                raise ValueError("SparseArray: positions must be positive integers")
            explicit.setdefault(key, (len(patterns), rhs))
            patterns.append(("pos", key, rhs))
        else:
            patterns.append(("pat", lhs, r))
    from .wl_linalg_optim import band_entries, band_rank
    bands = [(a, b) for kind, a, b in patterns if kind == "band"]
    if rest:
        if is_list(rest[0]):
            shape = [pos_int(v, "SparseArray", "dimension") for v in rest[0].args]
        else:
            n = pos_int(rest[0], "SparseArray", "dimension")
            ranks = {len(k) for k in explicit} | {band_rank(a, b) for a, b in bands} or {1}
            if None in ranks or len(ranks) != 1:
                raise ValueError("SparseArray: the rules have inconsistent ranks")
            shape = [n] * ranks.pop()
    elif bands and not explicit:
        rank = band_rank(*bands[0])
        if rank is None:
            raise ValueError("SparseArray: give the dimensions for this Band specification")
        ext = [0] * rank
        for a, b in bands:
            if len(a.args) >= 2 or not is_list(b):
                raise ValueError("SparseArray: give the dimensions when a Band repeats to the edge of the array")
            for k in band_entries(a, b, rank, [None] * rank):
                ext = [max(e, i) for e, i in zip(ext, k)]
        shape = ext
    else:
        if not explicit or len({len(k) for k in explicit}) != 1:
            raise ValueError("SparseArray: give the dimensions when the rules use patterns or mixed ranks")
        rank = len(next(iter(explicit)))
        shape = [max(k[i] for k in explicit) for i in range(rank)]
    for k in explicit:
        if len(k) != len(shape) or any(a > b for a, b in zip(k, shape)):
            raise ValueError(f"SparseArray: position {list(k)} lies outside the dimensions {shape}")
    if len(shape) > 1 and reduce(lambda a, b: a * b, shape) > 10 ** 6:
        raise ValueError("SparseArray: the dense array would be too large")

    band_maps = {id(a): band_entries(a, b, len(shape), shape) for a, b in bands}

    def value(idx):
        pos = L(*idx)
        for kind, a, b in patterns:
            if kind == "band":
                m = band_maps[id(a)]
                if idx in m:
                    return m[idx]
                continue
            if kind == "pos":
                if a == idx:
                    return b
            else:
                bnd = next(iter(K.match(a, pos, {}, ev)), None)
                if bnd is not None:
                    res = ev(Expr("Replace", pos, b))
                    return res
        return default

    def build(level, idx):
        if level == len(shape):
            return value(tuple(idx))
        return L(*(build(level + 1, idx + [i]) for i in range(1, shape[level] + 1)))
    return ev(build(0, []))


# ─────────────────────────── display ───────────────────────────

@fn("MatrixForm", area=AREA, status="added",
    notes="display wrapper: MatrixForm[expr] stays as MatrixForm[expr] and does not affect evaluation of expr "
          "(as in Wolfram, it is not a matrix); the text REPL prints it as an aligned grid (matrix_form_text)")
def _matrixform(x: Expr, *opts: Option) -> Expr:
    raise Unevaluated()


def matrix_form_text(x):
    """Aligned text rendering of MatrixForm[m] (vectors as a column)."""
    from ..formatting import input_form
    if is_list(x) and x.args and all(is_list(r) for r in x.args) and dims(x) is not None and len(dims(x)) >= 2:
        cells = [[input_form(v) for v in r.args] for r in x.args]
    elif is_list(x):
        cells = [[input_form(v)] for v in x.args]
    else:
        return input_form(x)
    width = [max(len(r[j]) for r in cells) for j in range(len(cells[0]))]
    return "\n".join("( " + "  ".join(c.rjust(w) for c, w in zip(r, width)) + " )" for r in cells)
