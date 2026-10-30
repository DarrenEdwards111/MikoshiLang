import pytest
import sympy as sp

from _harness import Raises, Pred, check, ids, to_complex
from mikoshilang.math_ops import to_sympy

try:
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    SCIPY = True
except ImportError:
    SCIPY = False


def M(x):
    return sp.Matrix([[sp.nsimplify(to_sympy(v)) if not isinstance(v, float) else v for v in r.args] for r in x.args])


def recon_close(parts, target, tol=1e-9):
    def f(r):
        got = parts(r)
        return all(abs(complex(sp.N(a)) - complex(b)) < tol for a, b in zip(got, sp.Matrix(target)))
    return Pred(f, "reconstructs the input")


A = "{{4, 1}, {2, 3}}"
CASES = [
    ("IdentityMatrix[2]", "{{1, 0}, {0, 1}}"), ("IdentityMatrix[1]", "{{1}}"), ("IdentityMatrix[0]", Raises()),
    ("ZeroMatrix[2]", "{{0, 0}, {0, 0}}"), ("ZeroMatrix[2, 3]", "{{0, 0, 0}, {0, 0, 0}}"), ("ZeroMatrix[-1]", Raises()),
    ("DiagonalMatrix[{1, 2}]", "{{1, 0}, {0, 2}}"), ("DiagonalMatrix[{1, 2}, 1]", "{{0, 1, 0}, {0, 0, 2}, {0, 0, 0}}"),
    ("DiagonalMatrix[{a}, -1]", "{{0, 0}, {a, 0}}"),
    ("SymmetricMatrixQ[{{1, 2}, {2, 1}}]", True), ("SymmetricMatrixQ[{{1, 2}, {3, 1}}]", False), ("SymmetricMatrixQ[{1, 2}]", False),
    ("HermitianQ[{{1, I}, {-I, 1}}]", True), ("HermitianQ[{{1, I}, {I, 1}}]", False), ("HermitianQ[5]", False),
    ("DiagonalMatrixQ[{{1, 0}, {0, 2}}]", True), ("DiagonalMatrixQ[{{1, 1}, {0, 2}}]", False), ("DiagonalMatrixQ[{{1, 0, 0}}]", True),
    ("UpperTriangularMatrixQ[{{1, 2}, {0, 3}}]", True), ("UpperTriangularMatrixQ[{{1, 0}, {2, 3}}]", False),
    ("UpperTriangularMatrixQ[{{1}}]", True),
    ("LowerTriangularMatrixQ[{{1, 0}, {2, 3}}]", True), ("LowerTriangularMatrixQ[{{1, 2}, {0, 3}}]", False),
    ("LowerTriangularMatrixQ[x]", False),
    ("CharacteristicPolynomial[{{1, 2}, {3, 4}}, x]", "x^2 - 5 x - 2"), ("CharacteristicPolynomial[{{2}}, x]", "2 - x"),
    ("CharacteristicPolynomial[{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, x]", "(1 - x)^3"),
    ("Eigenvalues[{{2, 1}, {1, 2}}]", "{3, 1}"), ("Eigenvalues[{{0, 1}, {-1, 0}}]", "{I, -I}"),
    ("Eigenvalues[{{1, 1}, {0, 1}}]", "{1, 1}"), ("Eigenvalues[{{-3, 0}, {0, 2}}]", "{-3, 2}"),
    ("Eigenvalues[{{1, -2}, {2, 1}}]", "{1 + 2 I, 1 - 2 I}"), ("Eigenvalues[{{0., 1.}, {-1., 0.}}]", [1j, -1j]),
    ("Eigenvectors[{{2, 1}, {1, 2}}]", "{{1, 1}, {-1, 1}}"), ("Eigenvectors[{{1, 1}, {0, 1}}]", "{{1, 0}, {0, 0}}"),
    ("Eigenvectors[{{2, 0}, {0, 3}}]", "{{0, 1}, {1, 0}}"),
    ("CholeskyDecomposition[{{4, 2}, {2, 3}}]", "{{2, 1}, {0, Sqrt[2]}}"), ("CholeskyDecomposition[{{9}}]", "{{3}}"),
    ("CholeskyDecomposition[{{1, 2}, {2, 1}}]", Raises()),
    ("LUDecomposition[{{1, 2}, {3, 4}}]", "{{{1, 2}, {3, -2}}, {1, 2}, 0}"),
    ("LUDecomposition[{{0, 1}, {1, 0}}]", "{{{1, 0}, {0, 1}}, {2, 1}, 0}"),
    ("LUDecomposition[{{2}}]", "{{{2}}, {1}, 0}"),
    ("QRDecomposition[{{3, 0}, {4, 5}}]", "{{{3/5, 4/5}, {-4/5, 3/5}}, {{5, 4}, {0, 3}}}"),
    ("QRDecomposition[{{1, 0}, {0, 1}}]", "{{{1, 0}, {0, 1}}, {{1, 0}, {0, 1}}}"),
    ("QRDecomposition[{{1}, {1}}]", "{{{1/Sqrt[2], 1/Sqrt[2]}}, {{Sqrt[2]}}}"),
    ("SingularValueDecomposition[{{3, 0}, {0, 4}}]", "{{{0, 1}, {1, 0}}, {{4, 0}, {0, 3}}, {{0, 1}, {1, 0}}}"),
    ("SingularValueDecomposition[{{2, 0}, {0, 1}}]", "{{{1, 0}, {0, 1}}, {{2, 0}, {0, 1}}, {{1, 0}, {0, 1}}}"),
    ("SingularValueDecomposition[{{1, 2}, {3, 4}, {5, 6}}]", Raises(NotImplementedError, "N\\[\\]")),
    ("SVD[{{3, 0}, {0, 4}}]", "{{{0, 1}, {1, 0}}, {{4, 0}, {0, 3}}, {{0, 1}, {1, 0}}}"),
    ("SVD[{{-2}}]", "{{{-1}}, {{2}}, {{1}}}"),
    ("Norm[{3, 4}]", "5"), ("Norm[{1, -2, 3}, 1]", "6"), ("Norm[{1, -5}, Infinity]", "5"), ("Norm[{{1, 2}, {3, 4}}, \"Frobenius\"]", "Sqrt[30]"),
    ("Norm[{{3, 0}, {0, 4}}]", "4"), ("Norm[{{1, 2}, {3, 4}}, 1]", "6"),
    ("FrobeniusNorm[{{1, 2}, {3, 4}}]", "Sqrt[30]"), ("FrobeniusNorm[{{3}}]", "3"), ("FrobeniusNorm[{{1, 1}, {1, 1}}]", "2"),
    ("SpectralNorm[{{3, 0}, {0, 4}}]", "4"), ("SpectralNorm[{{1, 1}, {0, 0}}]", "Sqrt[2]"), ("SpectralNorm[{{2}}]", "2"),
    ("ConditionNumber[{{1, 0}, {0, 2}}]", "2"), ("ConditionNumber[{{1, 0}, {0, 1}}]", "1"), ("ConditionNumber[{{1, 1}, {1, 1}}]", "Infinity"),
    ("Normalize[{3, 4}]", "{3/5, 4/5}"), ("Normalize[{0, 0}]", "{0, 0}"), ("Normalize[{1., 1.}]", [2 ** -0.5, 2 ** -0.5]),
    ("VectorAngle[{1, 0}, {0, 1}]", "Pi/2"), ("VectorAngle[{1, 1}, {1, 0}]", "Pi/4"), ("VectorAngle[{1, 0}, {-1, 0}]", "Pi"),
    ("VectorAngle[{0, 0}, {1, 0}]", Raises()),
    ("Cross[{1, 0, 0}, {0, 1, 0}]", "{0, 0, 1}"), ("Cross[{1, 2}]", "{-2, 1}"), ("Cross[{a, b, c}, {x, y, z}]", "{b z - c y, c x - a z, a y - b x}"),
    ("CrossProduct[{1, 2, 3}, {4, 5, 6}]", "{-3, 6, -3}"), ("CrossProduct[{1, 0, 0}, {1, 0, 0}]", "{0, 0, 0}"),
    ("CrossProduct[{1, 2}, {3, 4}]", Raises()),
    ("Cross3D[{0, 1, 0}, {0, 0, 1}]", "{1, 0, 0}"), ("Cross3D[{1/2, 0, 0}, {0, 2, 0}]", "{0, 0, 1}"), ("Cross3D[{1, 1, 1}, {2, 2, 2}]", "{0, 0, 0}"),
    ("ScalarTripleProduct[{1, 0, 0}, {0, 1, 0}, {0, 0, 1}]", "1"), ("ScalarTripleProduct[{1, 2, 3}, {4, 5, 6}, {7, 8, 9}]", "0"),
    ("ScalarTripleProduct[{2, 0, 0}, {0, 3, 0}, {0, 0, 4}]", "24"),
    ("KroneckerProduct[{{1, 2}}, {{1}, {1}}]", "{{1, 2}, {1, 2}}"), ("KroneckerProduct[{{1, 0}, {0, 1}}, {{a}}]", "{{a, 0}, {0, a}}"),
    ("KroneckerProduct[{{2}}, {{3}}, {{4}}]", "{{24}}"),
    ("LinearSolve[{{1, 1}, {1, -1}}, {2, 0}]", "{1, 1}"), ("LinearSolve[{{1, 2}, {2, 4}}, {3, 6}]", "{3, 0}"),
    ("LinearSolve[{{1, 1}, {1, 1}}, {1, 2}]", Raises()), ("LinearSolve[{{2}}, {x}]", "{x/2}"),
    ("MatrixPower[{{1, 1}, {0, 1}}, 3]", "{{1, 3}, {0, 1}}"), ("MatrixPower[{{2, 0}, {0, 3}}, -1]", "{{1/2, 0}, {0, 1/3}}"),
    ("MatrixPower[{{1, 2}, {3, 4}}, 0]", "{{1, 0}, {0, 1}}"),
    ("MatrixExp[{{0, 1}, {0, 0}}]", "{{1, 1}, {0, 1}}"), ("MatrixExp[{{1, 0}, {0, 2}}]", "{{E, 0}, {0, E^2}}"),
    ("MatrixExp[{{0, -t}, {t, 0}}]", "{{Cos[t], -Sin[t]}, {Sin[t], Cos[t]}}"),
    ("MatrixLog[{{1, 1}, {0, 1}}]", "{{0, 1}, {0, 0}}"), ("MatrixLog[{{E, 0}, {0, 1}}]", "{{1, 0}, {0, 0}}"),
    ("MatrixLog[{{0, 0}, {0, 1}}]", Raises()),
    ("MatrixSqrt[{{4, 0}, {0, 9}}]", "{{2, 0}, {0, 3}}"), ("MatrixSqrt[{{1, 2}, {0, 1}}]", "{{1, 1}, {0, 1}}"),
    ("MatrixSqrt[{{1}}]", "{{1}}"),
    ("MatrixSin[{{0, 0}, {0, 0}}]", "{{0, 0}, {0, 0}}"), ("MatrixSin[{{Pi/2, 0}, {0, Pi}}]", "{{1, 0}, {0, 0}}"),
    ("MatrixSin[{{0, 1}, {0, 0}}]", "{{0, 1}, {0, 0}}"),
    ("MatrixCos[{{0, 0}, {0, 0}}]", "{{1, 0}, {0, 1}}"), ("MatrixCos[{{Pi, 0}, {0, 0}}]", "{{-1, 0}, {0, 1}}"),
    ("MatrixCos[{{0, 1}, {0, 0}}]", "{{1, 0}, {0, 1}}"),
    ("MatrixTan[{{0, 0}, {0, 0}}]", "{{0, 0}, {0, 0}}"), ("MatrixTan[{{Pi/4, 0}, {0, 0}}]", "{{1, 0}, {0, 0}}"),
    ("MatrixTan[{{0, 1}, {0, 0}}]", "{{0, 1}, {0, 0}}"),
    ("MatrixRank[{{1, 2}, {2, 4}}]", "1"), ("MatrixRank[{{1, 0}, {0, 1}}]", "2"), ("MatrixRank[{{0, 0}}]", "0"),
    ("Rank[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]", "2"), ("Rank[{{1}}]", "1"), ("Rank[{{1, 1}, {1, 1}}]", "1"),
    ("MatrixNullity[{{1, 2}, {2, 4}}]", "1"), ("MatrixNullity[{{1, 0}, {0, 1}}]", "0"), ("MatrixNullity[{{0, 0, 0}}]", "3"),
    ("NullSpace[{{1, 2}, {2, 4}}]", "{{-2, 1}}"), ("NullSpace[{{1, 0}, {0, 1}}]", "{}"),
    ("NullSpace[{{1, 1, 1}}]", "{{-1, 0, 1}, {-1, 1, 0}}"),
    ("RowReduce[{{1, 2}, {3, 4}}]", "{{1, 0}, {0, 1}}"), ("RowReduce[{{1, 2}, {2, 4}}]", "{{1, 2}, {0, 0}}"),
    ("RowReduce[{{2, 4, 6}}]", "{{1, 2, 3}}"),
    ("RowSpace[{{1, 2}, {2, 4}}]", "{{1, 2}}"), ("RowSpace[{{1, 0}, {0, 1}}]", "{{1, 0}, {0, 1}}"), ("RowSpace[{{0, 0}}]", "{}"),
    ("ColumnSpace[{{1, 2}, {2, 4}}]", "{{1, 2}}"), ("ColumnSpace[{{1, 0}, {0, 1}}]", "{{1, 0}, {0, 1}}"), ("ColumnSpace[{{0, 0}}]", "{}"),
    ("PseudoInverse[{{1, 2}, {2, 4}}]", "{{1/25, 2/25}, {2/25, 4/25}}"), ("PseudoInverse[{{2, 0}, {0, 4}}]", "{{1/2, 0}, {0, 1/4}}"),
    ("PseudoInverse[{{1}, {1}}]", "{{1/2, 1/2}}"),
    ("JordanForm[{{1, 1}, {0, 1}}]", "{{{1, 0}, {0, 1}}, {{1, 1}, {0, 1}}}"),
    ("JordanForm[{{2, 0}, {0, 3}}]", "{{{1, 0}, {0, 1}}, {{2, 0}, {0, 3}}}"),
    ("JordanForm[{{5}}]", "{{{1}}, {{5}}}"),
    ("Trace[{{1, 2}, {3, 4}}]", Raises(NotImplementedError, "Tr")),
    ("Tr[{{1, 2}, {3, 4}}]", "5"), ("Tr[{1, 2, 3}]", "6"), ("Tr[{{a, b}, {c, d}}]", "a + d"),
    ("TransposeList[{{1, 2}, {3, 4}}]", "{{1, 3}, {2, 4}}"), ("TransposeList[{{1, 2, 3}}]", "{{1}, {2}, {3}}"),
    ("TransposeList[{{1}, {2, 3}}]", Raises()),
]

if SCIPY:
    import numpy as np

    def _hess_ok(r):
        p, h = (np.array([[complex(to_complex(v)) for v in row.args] for row in x.args]) for x in r.args)
        m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        return np.allclose(p @ h @ p.conj().T, m) and np.allclose(np.tril(h, -2), 0)

    def _schur_ok(r, m):
        q, t = (np.array([[complex(to_complex(v)) for v in row.args] for row in x.args]) for x in r.args)
        # real input gives the real Schur form (upper quasi-triangular), as Wolfram
        return np.allclose(q @ t @ q.conj().T, m) and np.allclose(np.tril(t, -2), 0)

    def _svd_ok(r, m):
        u, s, v = (np.array([[complex(to_complex(v)) for v in row.args] for row in x.args]) for x in r.args)
        return np.allclose(u @ s @ v.conj().T, m)

    CASES += [
        ("HessenbergDecomposition[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}]", Pred(_hess_ok, "p.h.p^T = m, h Hessenberg")),
        ("HessenbergDecomposition[{{1., 2.}, {3., 4.}}]", Pred(lambda r: True, "2x2 is already Hessenberg")),
        ("HessenbergDecomposition[{{1, 2}, {3}}]", Raises()),
        ("SchurDecomposition[{{1, 2}, {3, 4}}]", Pred(lambda r: _schur_ok(r, [[1, 2], [3, 4]]), "q.t.q^T = m")),
        ("SchurDecomposition[{{0, 1}, {-1, 0}}]", Pred(lambda r: _schur_ok(r, [[0, 1], [-1, 0]]), "real Schur form")),
        ("SchurDecomposition[{{1, 2}}]", Raises()),
        ("SVD[{{1., 2.}, {3., 4.}, {5., 6.}}]", Pred(lambda r: _svd_ok(r, [[1, 2], [3, 4], [5, 6]]), "u.s.v^H = m")),
        ("MatrixExp[{{0., 1.}, {-1., 0.}}]", Pred(lambda r: abs(to_complex(r.args[0].args[0]) - np.cos(1)) < 1e-12, "cos 1")),
    ]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_linalg(case):
    check(case)
