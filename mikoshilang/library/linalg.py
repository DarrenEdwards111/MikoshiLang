"""Linear algebra.  Exact matrices are handled exactly with SymPy; matrices
containing machine numbers use NumPy/SciPy where noted.  Output conventions
follow the Wolfram Language."""
import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, S, R, L, rule, numeric_out, sym, number, is_list, int_arg,
                   nonneg_int, pos_int, list_arg, variable, has_float, matrix_arg, mat_out)

AREA = "linear algebra"


def _is_matrix(x):
    return is_list(x) and len(x.args) > 0 and all(is_list(r) for r in x.args) and \
        len({len(r.args) for r in x.args}) == 1 and len(x.args[0].args) > 0 and \
        not any(is_list(v) for r in x.args for v in r.args)


def _mat(x, fname, square=False):
    m = matrix_arg(x, fname, square=square)
    return m, any(isinstance(v, sp.Float) for v in m)


def _vec(x, fname, n=None):
    items = list_arg(x, fname, "vector", min_len=1)
    if any(is_list(v) for v in items):
        raise ValueError(f"{fname}: argument must be a vector")
    v = [sym(i, fname) for i in items]
    if n is not None and len(v) != n:
        raise ValueError(f"{fname}: vectors must have length {n}")
    return v


def _out(m, inexact):
    if inexact:
        m = m.evalf()
    return mat_out(m)


def _np():
    try:
        import numpy as np
        return np
    except ImportError:
        raise NotImplementedError("this operation on machine-precision matrices requires NumPy")


def _npmat(m):
    np = _np()
    a = np.array([[complex(v) for v in row] for row in m.tolist()])
    if not np.any(a.imag):
        a = a.real
    return a


@fn("IdentityMatrix", area=AREA, status="fixed",
    notes="IdentityMatrix[n] and IdentityMatrix[{m, n}]; non-integer sizes raise (legacy truncated)")
def _identity(n):
    if is_list(n):
        r, c = [pos_int(v, "IdentityMatrix", "size") for v in list_arg(n, "IdentityMatrix")]
        return mat_out(sp.eye(r, c))
    return mat_out(sp.eye(pos_int(n, "IdentityMatrix", "size")))


@fn("ZeroMatrix", area=AREA, status="fixed", notes="m x n zero matrix (MikoshiLang); ZeroMatrix[n] is n x n; " 
    "non-integer sizes raise")
def _zeros(m, n=None):
    r = pos_int(m, "ZeroMatrix", "size")
    c = r if n is None else pos_int(n, "ZeroMatrix", "size")
    return mat_out(sp.zeros(r, c))


@fn("DiagonalMatrix", area=AREA, status="fixed", notes="DiagonalMatrix[list] and DiagonalMatrix[list, k] (k-th diagonal)")
def _diag(v, k=0):
    items = list_arg(v, "DiagonalMatrix", min_len=1)
    off = int_arg(k, "DiagonalMatrix", "offset")
    n = len(items) + abs(off)
    m = sp.zeros(n, n)
    for i, x in enumerate(items):
        r, c = (i, i + off) if off >= 0 else (i - off, i)
        m[r, c] = S(x)
    return mat_out(m)


def _pred(fname, test):
    @fn(fname, area=AREA, status="fixed",
        notes="False for non-matrices (legacy raised); exact test on the entries")
    def impl(x):
        if not _is_matrix(x):
            return False
        m = matrix_arg(x, fname)
        return bool(test(m))
    return impl


_pred("SymmetricMatrixQ", lambda m: m.rows == m.cols and all(sp.simplify(m[i, j] - m[j, i]) == 0
                                                           for i in range(m.rows) for j in range(i)))
_pred("HermitianQ", lambda m: m.rows == m.cols and all(sp.simplify(m[i, j] - sp.conjugate(m[j, i])) == 0
                                                     for i in range(m.rows) for j in range(i + 1)))
_pred("DiagonalMatrixQ", lambda m: all(m[i, j] == 0 for i in range(m.rows) for j in range(m.cols) if i != j))
_pred("UpperTriangularMatrixQ", lambda m: all(m[i, j] == 0 for i in range(m.rows) for j in range(m.cols) if i > j))
_pred("LowerTriangularMatrixQ", lambda m: all(m[i, j] == 0 for i in range(m.rows) for j in range(m.cols) if i < j))


@fn("CharacteristicPolynomial", area=AREA, status="fixed",
    notes="legacy returned Det[x I - m]; Wolfram CharacteristicPolynomial[m, x] = Det[m - x I] "
          "(differs by (-1)^n)")
def _charpoly(m, x):
    M, _ = _mat(m, "CharacteristicPolynomial", square=True)
    xv = variable(x, "CharacteristicPolynomial")
    return R(sp.expand((M - xv * sp.eye(M.rows)).det()))


def _eig_key(c):
    """Wolfram order: decreasing |v|; ties by increasing real part, then decreasing imaginary part
    (Eigenvalues[{{0, 1}, {-1, 0}}] = {I, -I})."""
    return (-round(abs(c), 12), round(c.real, 12), -round(c.imag, 12))


def _eig_sorted(vals):
    def key(v):
        return _eig_key(complex(sp.N(v)))
    if all(v.is_number for v in vals):
        return sorted(vals, key=key)
    return vals


@fn("Eigenvalues", area=AREA, status="fixed",
    notes="eigenvalues with multiplicity in Wolfram order (decreasing absolute value; legacy returned SymPy's "
          "dict order)")
def _eigenvalues(m):
    M, inexact = _mat(m, "Eigenvalues", square=True)
    if inexact:
        np = _np()
        vals = list(np.linalg.eigvals(_npmat(M)))
        vals.sort(key=lambda c: _eig_key(complex(c)))
        return L(*(R(complex(v)) for v in vals))
    ev = M.eigenvals()
    vals = []
    for v, k in ev.items():
        vals.extend([v] * k)
    return L(*(R(v) for v in _eig_sorted(vals)))


def _scale_last(v):
    for x in reversed(list(v)):
        if x != 0:
            return (v / x).applyfunc(sp.simplify)
    return v


@fn("Eigenvectors", area=AREA, status="fixed",
    notes="ordered like Eigenvalues, exact vectors scaled so the last non-zero entry is 1, zero vectors pad "
          "defective matrices (Wolfram); machine matrices give unit vectors")
def _eigenvectors(m):
    M, inexact = _mat(m, "Eigenvectors", square=True)
    n = M.rows
    if inexact:
        np = _np()
        vals, vecs = np.linalg.eig(_npmat(M))
        order = sorted(range(n), key=lambda i: _eig_key(complex(vals[i])))
        return L(*(L(*(R(complex(x)) for x in vecs[:, i])) for i in order))
    groups = []
    for val, mult, vecs in M.eigenvects():
        vs = [_scale_last(v) for v in vecs] + [sp.zeros(n, 1)] * (mult - len(vecs))
        groups.append((val, vs))
    vals = _eig_sorted([g[0] for g in groups])
    out = []
    for v in vals:
        for g in groups:
            if g[0] == v:
                out.extend(L(*(R(x) for x in vec)) for vec in g[1])
                break
    return L(*out)


@fn("CholeskyDecomposition", area=AREA, status="fixed",
    notes="legacy returned the lower factor; Wolfram returns the upper triangular u with "
          "ConjugateTranspose[u].u = m; non-Hermitian or non-positive-definite matrices raise")
def _cholesky(m):
    M, inexact = _mat(m, "CholeskyDecomposition", square=True)
    if M != M.H:
        raise ValueError("CholeskyDecomposition: matrix must be Hermitian")
    if not M.is_positive_definite:
        raise ValueError("CholeskyDecomposition: matrix must be positive definite")
    Lm = M.cholesky(hermitian=True)
    return _out(Lm.H.applyfunc(sp.simplify), inexact)


@fn("LUDecomposition", area=AREA, status="fixed",
    notes="legacy ignored the row permutation (L.U != m whenever pivoting occurs); now Wolfram's {lu, p, c}: "
          "lu holds L (unit lower, below the diagonal) and U, m[[p]] == L.U, c = 0 for exact matrices and the "
          "infinity-norm condition number for machine matrices")
def _lu(m):
    M, inexact = _mat(m, "LUDecomposition", square=True)
    Lm, U, perm = M.LUdecomposition()
    p = list(range(1, M.rows + 1))
    for a, b in perm:
        p[a], p[b] = p[b], p[a]
    lu = U.copy()
    for i in range(M.rows):
        for j in range(i):
            lu[i, j] = Lm[i, j]
    c = 0
    if inexact:
        np = _np()
        a = _npmat(M)
        c = float(np.linalg.cond(a, np.inf))
    return L(_out(lu, inexact), L(*p), c)


@fn("QRDecomposition", area=AREA, status="fixed",
    notes="legacy returned {Q, R} with m = Q.R; Wolfram returns {q, r} with m = ConjugateTranspose[q].r "
          "(q has orthonormal rows)")
def _qr(m):
    M, inexact = _mat(m, "QRDecomposition")
    Q, Rm = M.QRdecomposition()
    return L(_out(Q.H.applyfunc(sp.simplify), inexact), _out(Rm.applyfunc(sp.simplify), inexact))


def _svd(m, fname):
    M, inexact = _mat(m, fname)
    if inexact:
        np = _np()
        u, s, vh = np.linalg.svd(_npmat(M), full_matrices=True)
        sig = np.zeros((M.rows, M.cols))
        for i, x in enumerate(s):
            sig[i, i] = x
        return L(R(u), R(sig), R(vh.conj().T))
    if M.rows != M.cols or M.rank() != M.rows:
        raise NotImplementedError(f"{fname}: exact SVD is supported for square non-singular matrices; "
                                  "apply N[] for the general case")
    U, Sg, V = M.singular_value_decomposition()
    order = sorted(range(Sg.rows), key=lambda i: -float(sp.N(Sg[i, i])))
    U = sp.Matrix.hstack(*[U[:, i] for i in order])
    V = sp.Matrix.hstack(*[V[:, i] for i in order])
    Sg = sp.diag(*[Sg[i, i] for i in order])
    return L(mat_out(U.applyfunc(sp.simplify)), mat_out(Sg), mat_out(V.applyfunc(sp.simplify)))


SVDN = ("Wolfram {u, s, v} with m = u.s.ConjugateTranspose[v] and singular values in decreasing order "
        "(legacy did not order them); machine matrices give the full SVD")
fn("SingularValueDecomposition", area=AREA, status="fixed", notes=SVDN)(lambda m: _svd(m, "SingularValueDecomposition"))
fn("SVD", area=AREA, status="fixed", notes="MikoshiLang alias of SingularValueDecomposition; " + SVDN)(
    lambda m: _svd(m, "SVD"))


def _spectral(M):
    if any(v.has(sp.Float) for v in M):
        np = _np()
        return float(np.linalg.norm(_npmat(M), 2))
    ev = (M.H * M).eigenvals()
    return sp.sqrt(sp.Max(*ev.keys()))


@fn("Norm", area=AREA, status="fixed",
    notes="vectors: Norm[v] and Norm[v, p] (p >= 1 or Infinity); matrices: Norm[m] is the spectral 2-norm as in "
          "Wolfram (legacy flattened the matrix, giving the Frobenius norm), Norm[m, 1], Norm[m, Infinity], "
          "Norm[m, \"Frobenius\"]")
def _norm(x, p=2):
    if _is_matrix(x):
        M, inexact = _mat(x, "Norm")
        if p == 2:
            return numeric_out(_spectral(M), x)
        if p == "Frobenius":
            return numeric_out(sp.sqrt(sp.Add(*(sp.Abs(v) ** 2 for v in M))), x)
        if p == 1:
            return numeric_out(sp.Max(*(sp.Add(*(sp.Abs(M[i, j]) for i in range(M.rows))) for j in range(M.cols))), x)
        if isinstance(p, Symbol) and p.name == "Infinity":
            return numeric_out(sp.Max(*(sp.Add(*(sp.Abs(M[i, j]) for j in range(M.cols))) for i in range(M.rows))), x)
        raise NotImplementedError("Norm: only matrix norms 1, 2, Infinity and \"Frobenius\" are supported")
    if is_list(x):
        v = _vec(x, "Norm")
        if isinstance(p, Symbol) and p.name == "Infinity":
            return numeric_out(sp.Max(*(sp.Abs(c) for c in v)), x)
        if p == "Frobenius":
            p = 2
        pp = sym(p, "Norm")
        if pp.is_number and pp < 1:
            raise ValueError("Norm: p must be >= 1")
        if pp == 2:
            return numeric_out(sp.sqrt(sp.Add(*(sp.Abs(c) ** 2 for c in v))), x, p)
        return numeric_out(sp.Add(*(sp.Abs(c) ** pp for c in v)) ** (1 / pp), x, p)
    v = sym(x, "Norm")
    return numeric_out(sp.Abs(v), x)


@fn("FrobeniusNorm", area=AREA, status="fixed", notes="exact Frobenius norm (legacy returned floats)")
def _frob(m):
    M, _ = _mat(m, "FrobeniusNorm")
    return numeric_out(sp.sqrt(sp.Add(*(sp.Abs(v) ** 2 for v in M))), m)


@fn("SpectralNorm", area=AREA, status="fixed", notes="largest singular value, exact for exact matrices "
    "(legacy always returned a float via NumPy)")
def _specnorm(m):
    M, _ = _mat(m, "SpectralNorm")
    return numeric_out(_spectral(M), m)


@fn("ConditionNumber", area=AREA, status="fixed",
    notes="2-norm condition number sigma_max/sigma_min (MikoshiLang); exact for exact matrices, Infinity for "
          "singular matrices (legacy returned a huge float or an Error expression without NumPy)")
def _cond(m):
    M, inexact = _mat(m, "ConditionNumber", square=True)
    if inexact:
        np = _np()
        return float(np.linalg.cond(_npmat(M), 2))
    if M.det() == 0:
        return R(sp.oo)
    ev = (M.H * M).eigenvals()
    return R(sp.simplify(sp.sqrt(sp.Max(*ev) / sp.Min(*ev))))


@fn("Normalize", area=AREA, status="fixed",
    notes="v / Norm[v]; the zero vector is returned unchanged as in Wolfram (legacy divided by zero); "
          "Normalize[z] for a number gives z/Abs[z]")
def _normalize(v):
    if not is_list(v):
        z = sym(v, "Normalize")
        return numeric_out(0 if z == 0 else z / sp.Abs(z), v)
    c = _vec(v, "Normalize")
    n = sp.sqrt(sp.Add(*(sp.Abs(x) ** 2 for x in c)))
    if n == 0:
        return v
    return L(*(numeric_out(sp.simplify(x / n), v) for x in c))


@fn("VectorAngle", area=AREA, status="fixed",
    notes="ArcCos[u.Conjugate[v]/(|u| |v|)]; zero vectors and different lengths raise")
def _vangle(u, v):
    a, b = _vec(u, "VectorAngle"), _vec(v, "VectorAngle")
    if len(a) != len(b):
        raise ValueError("VectorAngle: vectors must have the same length")
    na = sp.sqrt(sp.Add(*(sp.Abs(x) ** 2 for x in a)))
    nb = sp.sqrt(sp.Add(*(sp.Abs(x) ** 2 for x in b)))
    if na == 0 or nb == 0:
        raise ValueError("VectorAngle: vectors must be non-zero")
    dot = sp.Add(*(x * sp.conjugate(y) for x, y in zip(a, b)))
    return numeric_out(sp.simplify(sp.acos(sp.simplify(dot / (na * nb)))), u, v)


def _cross3(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _cross(fname, notes):
    @fn(fname, area=AREA, status="fixed", notes=notes)
    def impl(u, v=None):
        if v is None:
            a = _vec(u, fname, 2)
            return L(numeric_out(-a[1], u), numeric_out(a[0], u))
        a, b = _vec(u, fname, 3), _vec(v, fname, 3)
        return L(*(numeric_out(sp.expand(c), u, v) for c in _cross3(a, b)))
    return impl


_cross("Cross", "3D cross product, exact; Cross[{x, y}] = {-y, x}; non-3D vectors raise")
_cross("CrossProduct", "alias of Cross (exact)")
_cross("Cross3D", "alias of Cross; legacy converted to floats and returned an Error expression for bad input")


@fn("ScalarTripleProduct", area=AREA, status="fixed", notes="a.(b x c), exact (legacy converted to floats)")
def _triple(a, b, c):
    x, y, z = _vec(a, "ScalarTripleProduct", 3), _vec(b, "ScalarTripleProduct", 3), _vec(c, "ScalarTripleProduct", 3)
    return numeric_out(sp.expand(sp.Add(*(p * q for p, q in zip(x, _cross3(y, z))))), a, b, c)


def _as_matrix_or_vector(x, fname):
    if _is_matrix(x):
        return matrix_arg(x, fname), True
    return sp.Matrix([_vec(x, fname)]), False


@fn("KroneckerProduct", area=AREA, status="fixed",
    notes="variadic Kronecker product of matrices and/or vectors (vectors give vectors, as Wolfram)")
def _kron(*ms):
    if not ms:
        raise ValueError("KroneckerProduct needs at least one argument")
    mats = [_as_matrix_or_vector(m, "KroneckerProduct") for m in ms]
    out = mats[0][0]
    for m, _ in mats[1:]:
        out = sp.kronecker_product(out, m)
    if not any(is_mat for _, is_mat in mats):
        return L(*(numeric_out(v, *ms) for v in out))
    return L(*(L(*(numeric_out(out[i, j], *ms) for j in range(out.cols))) for i in range(out.rows)))


@fn("LinearSolve", area=AREA, status="fixed",
    notes="solves m.x == b for a vector or matrix b; underdetermined systems give one particular solution "
          "(free parameters 0, as Wolfram); inconsistent systems raise")
def _linsolve(m, b):
    M, inexact = _mat(m, "LinearSolve")
    if _is_matrix(b):
        B = matrix_arg(b, "LinearSolve")
        vec = False
    else:
        B = sp.Matrix(_vec(b, "LinearSolve"))
        vec = True
    if B.rows != M.rows:
        raise ValueError("LinearSolve: dimensions of m and b do not match")
    try:
        sol, params = M.gauss_jordan_solve(B)
    except ValueError:
        raise ValueError("LinearSolve: the linear system has no solution")
    if params.shape[0]:
        sol = sol.subs({p: 0 for p in params})
    sol = sol.applyfunc(sp.simplify)
    if inexact:
        sol = sol.evalf()
    if vec:
        return L(*(R(v) for v in sol))
    return mat_out(sol)


@fn("MatrixPower", area=AREA, status="fixed",
    notes="integer powers (negative powers of singular matrices raise); symbolic n via diagonalisation")
def _mpow(m, n):
    M, inexact = _mat(m, "MatrixPower", square=True)
    k = sym(n, "MatrixPower")
    if k.is_Integer and k < 0 and M.det() == 0:
        raise ValueError("MatrixPower: matrix is singular")
    if not k.is_Integer and k.is_number:
        raise NotImplementedError("MatrixPower: non-integer powers are not supported")
    return _out((M ** k).applyfunc(sp.simplify), inexact)


def _mfunc(fname, f, scipy_name, notes):
    @fn(fname, area=AREA, status="fixed", notes=notes)
    def impl(m):
        M, inexact = _mat(m, fname, square=True)
        if inexact:
            try:
                import scipy.linalg as sl
            except ImportError:
                raise NotImplementedError(f"{fname} of a machine-precision matrix requires SciPy")
            a = _npmat(M)
            if scipy_name == "tanm":
                r = sl.solve(sl.cosm(a), sl.sinm(a))
            else:
                r = getattr(sl, scipy_name)(a)
                if isinstance(r, tuple):
                    r = r[0]
            np = _np()
            r = np.asarray(r)
            if np.iscomplexobj(r) and not np.any(np.abs(r.imag) > 1e-12 * max(1.0, np.abs(r).max())):
                r = r.real
            return L(*(L(*(R(complex(x)) if np.iscomplexobj(r) else float(x) for x in row)) for row in r))
        if fname in ("MatrixLog",) and M.det() == 0:
            raise ValueError("MatrixLog: matrix is singular")
        try:
            res = M.analytic_func(f(sp.Symbol("_t")), sp.Symbol("_t"))
        except Exception as exc:
            raise NotImplementedError(f"{fname}: could not compute the exact matrix function ({exc})")
        return mat_out(res.applyfunc(lambda v: sp.simplify(v)))
    return impl


_mfunc("MatrixExp", sp.exp, "expm", "exact via the Jordan form (SymPy); machine matrices via SciPy expm")
_mfunc("MatrixLog", sp.log, "logm",
       "principal logarithm; legacy discarded the imaginary part (MatrixLog[{{-1, 0}, {0, 1}}] was wrong)")
_mfunc("MatrixSqrt", sp.sqrt, "sqrtm", "principal square root; legacy discarded the imaginary part")
_mfunc("MatrixSin", sp.sin, "sinm", "exact (legacy summed a truncated Taylor series)")
_mfunc("MatrixCos", sp.cos, "cosm", "exact (legacy summed a truncated Taylor series)")
_mfunc("MatrixTan", sp.tan, "tanm", "exact (legacy summed a truncated Taylor series)")


def _rank(m, fname):
    M, inexact = _mat(m, fname)
    if inexact:
        np = _np()
        return int(np.linalg.matrix_rank(_npmat(M)))
    return M.rank(simplify=True)


fn("MatrixRank", area=AREA, status="verified", notes="exact rank; machine matrices use an SVD tolerance")(
    lambda m: _rank(m, "MatrixRank"))
fn("Rank", area=AREA, status="verified", notes="MikoshiLang alias of MatrixRank")(lambda m: _rank(m, "Rank"))


@fn("MatrixNullity", area=AREA, status="verified", notes="columns - rank (MikoshiLang)")
def _nullity(m):
    M, _ = _mat(m, "MatrixNullity")
    return M.cols - _rank(m, "MatrixNullity")


@fn("NullSpace", area=AREA, status="fixed",
    notes="basis vectors in Wolfram order (legacy reversed it), exact; empty list for full column rank")
def _nullspace(m):
    M, inexact = _mat(m, "NullSpace")
    vs = list(reversed(M.nullspace(simplify=True)))
    return L(*(L(*(R(sp.N(x) if inexact else x) for x in v)) for v in vs))


@fn("RowReduce", area=AREA, status="verified", notes="reduced row echelon form")
def _rref(m):
    M, inexact = _mat(m, "RowReduce")
    return _out(M.rref(simplify=True)[0], inexact)


@fn("RowSpace", area=AREA, status="verified", notes="basis of the row space (non-zero rows of the RREF; MikoshiLang)")
def _rowspace(m):
    M, _ = _mat(m, "RowSpace")
    return L(*(L(*(R(x) for x in v)) for v in M.rowspace()))


@fn("ColumnSpace", area=AREA, status="verified", notes="basis of the column space (pivot columns; MikoshiLang)")
def _colspace(m):
    M, _ = _mat(m, "ColumnSpace")
    return L(*(L(*(R(x) for x in v)) for v in M.columnspace()))


@fn("PseudoInverse", area=AREA, status="verified", notes="Moore-Penrose pseudo-inverse (exact; NumPy for machine matrices)")
def _pinv(m):
    M, inexact = _mat(m, "PseudoInverse")
    if inexact:
        np = _np()
        return R(np.linalg.pinv(_npmat(M)))
    return mat_out(M.pinv().applyfunc(sp.simplify))


@fn("JordanForm", area=AREA, status="verified",
    notes="{s, j} with m = s.j.Inverse[s] (Wolfram JordanDecomposition convention)")
def _jordan(m):
    M, inexact = _mat(m, "JordanForm", square=True)
    if inexact:
        raise NotImplementedError("JordanForm is numerically unstable for machine-precision matrices; use exact input")
    P, J = M.jordan_form()
    return L(mat_out(P), mat_out(J))


def _scipy_decomp(m, fname):
    M, _ = _mat(m, fname, square=True)
    if M.free_symbols:
        raise ValueError(f"{fname}: matrix must be numeric")
    try:
        import scipy.linalg as sl
    except ImportError:
        raise NotImplementedError(f"{fname} requires SciPy (pip install mikoshilang[signal])")
    return sl, _npmat(M.evalf())


@fn("HessenbergDecomposition", area=AREA, status="fixed",
    notes="Wolfram {p, h} with m = p.h.ConjugateTranspose[p] (legacy returned {H -> h, Q -> q} rules); numeric")
def _hess(m):
    sl, a = _scipy_decomp(m, "HessenbergDecomposition")
    h, q = sl.hessenberg(a, calc_q=True)
    return L(R(q), R(h))


@fn("SchurDecomposition", area=AREA, status="fixed",
    notes="Wolfram {q, t} with m = q.t.ConjugateTranspose[q] (legacy returned {T -> t, Z -> z} rules); numeric, "
          "real Schur form for real matrices")
def _schur(m):
    sl, a = _scipy_decomp(m, "SchurDecomposition")
    t, z = sl.schur(a)
    return L(R(z), R(t))


def _trace_out(chain):
    return L(*(_trace_out(c) if isinstance(c, list) else Expr("HoldForm", c) for c in chain))


def _trace_filter(chain, form):
    from ..kernel import match
    out = []
    for c in chain:
        if isinstance(c, list):
            sub = _trace_filter(c, form)
            if sub:
                out.append(sub)
        elif next(iter(match(form, c, {}, lambda x: x)), None) is not None:
            out.append(c)
    return out


@fn("Trace", area=AREA, status="fixed",
    notes="Wolfram evaluation trace: Trace[expr] evaluates expr and returns the chain of forms as HoldForm[...] "
          "with subevaluations as nested lists (Trace[1 + 2 3] -> {{2 3, 6}, 1 + 6, 7}); Trace[expr, form] keeps "
          "only forms matching the pattern. Internal evaluations made by library functions can add extra steps "
          "that Wolfram would not show. For the matrix trace use Tr (legacy Trace[m] meant Tr[m])")
def _trace(e, *form):
    if len(form) > 1:
        raise NotImplementedError("Trace[expr, form, s] (a third argument) is not supported")
    from ..evaluate import trace_evaluation
    chain, _ = trace_evaluation(e)
    if form:
        chain = _trace_filter(chain, form[0])
    return _trace_out(chain)


@fn("Tr", area=AREA, status="promoted", notes="trace of a matrix (sum of the diagonal); Tr[list] sums a vector")
def _tr(m):
    if _is_matrix(m):
        M = matrix_arg(m, "Tr")
        return numeric_out(sp.Add(*(M[i, i] for i in range(min(M.rows, M.cols)))), m)
    return numeric_out(sp.Add(*_vec(m, "Tr")), m)


@fn("TransposeList", area=AREA, status="fixed",
    notes="transpose of a rectangular list of lists (MikoshiLang); ragged input now raises (legacy zip-truncated)")
def _tlist(x):
    rows = list_arg(x, "TransposeList", min_len=1)
    if not all(is_list(r) for r in rows) or len({len(r.args) for r in rows}) != 1:
        raise ValueError("TransposeList: argument must be a rectangular list of lists")
    return L(*(L(*c) for c in zip(*(r.args for r in rows))))
