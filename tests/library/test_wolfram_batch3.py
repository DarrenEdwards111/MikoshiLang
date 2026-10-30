"""Differential tests for coverage batch 3 (linear algebra).

Expected values are Wolfram Language results (examples on the reference pages
reference.wolfram.com/language/ref/<Name>.html, or values that follow from their Details sections), or
independent references computed here with NumPy / SciPy / SymPy.  Decompositions that are not unique are checked
through their defining properties (e.g. m == s.j.Inverse[s]).
"""
import numpy as np
import pytest
import sympy as sp

from _harness import Pred, Raises, check, ids
from mikoshilang.math_ops import to_sympy

try:
    import scipy.linalg as sla
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


def _sm(x):
    """MikoshiLang matrix/vector -> SymPy Matrix."""
    v = to_sympy(x)
    return sp.Matrix(v)


def _arr(x):
    return np.array(sp.Matrix(to_sympy(x)).evalf(30).tolist(), dtype=complex)


def NP(want, tol=1e-10, what=None):
    """A numeric array that agrees with the NumPy reference ``want``."""
    want = np.asarray(want, dtype=complex)

    def ok(got):
        g = _arr(got)
        if g.shape != want.reshape(g.shape).shape or g.size != want.size:
            return False
        return bool(np.allclose(g.reshape(want.shape), want, atol=tol, rtol=tol))
    return Pred(ok, what or f"~{want.tolist()}")


def ZERO(what="== 0"):
    """An exact (or within 1e-10) zero matrix."""
    def ok(got):
        m = _sm(got)
        return all(sp.simplify(v) == 0 or abs(complex(sp.N(v))) < 1e-10 for v in m)
    return Pred(ok, what)


A = [[1, 2], [3, 4]]
A3 = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
B = [[1.1, 1.3, 2.1], [2.2, 2.2, 3.3], [3.3, -3, 4.7]]
HB = [[2, 1 - 1j], [1 + 1j, 3]]

BASICS = [
    # Dot: any rank, any number of factors, infix "."
    ("{1, 2}.{3, 4}", "11"), ("{{1, 2}, {3, 4}}.{5, 6}", "{17, 39}"), ("{5, 6}.{{1, 2}, {3, 4}}", "{23, 34}"),
    ("{{1, 2}, {3, 4}}.{{1, 0}, {0, 1}}.{1, 1}", "{3, 7}"), ("Dot[{a, b}, {c, d}]", "a c + b d"),
    ("{a, b}.{{p, q}, {r, s}}", "{a p + b r, a q + b s}"),
    ("Dot[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {1, 1}]",
     NP(np.dot(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), [1, 1]))),
    ("Dot[{1, 2, 3}, {{{1, 0}, {0, 1}}, {{2, 0}, {0, 2}}, {{3, 0}, {0, 3}}}]",
     NP(np.tensordot([1, 2, 3], np.array([np.eye(2), 2 * np.eye(2), 3 * np.eye(2)]), 1))),
    ("Dot[{1., 2}, {3, 4}]", 11.0), ("Head[Dot[{1., 2}, {3, 4}]] === Real", True),
    ("Dot[a, b]", Pred(lambda g: str(g) == "Dot[a, b]", "unevaluated")), ("a.b.c", Pred(lambda g: g.head == "Dot" and len(g.args) == 3, "flat Dot[a, b, c]")),
    ("Dot[{1, 2}]", "{1, 2}"), ("Dot[{1, 2}, {1, 2, 3}]", Raises()), ("{1, 2}.x", Pred(lambda g: g.head == "Dot", "unevaluated")),
    ("2 {1, 2}.{3, 4}", "22"), ("-{1, 2}.{3, 4}", "-11"), ("{1, 2}.{3, 4}^2", "41"),
    ("f /@ {1, 2}.{3, 4}", "3 f[1] + 4 f[2]"), ("x /. x -> 2", "2"), ("1.5 + .5", 2.0),
    ("DotProduct[{1, 2, 3}, {4, 5, 6}]", "32"), ("DotProduct[{a, b}, {c, d}]", "a c + b d"),
    ("DotProduct[{1, 2}, {3}]", Raises()),
    ("{1, I}.{1, I}", "0"), ("{{1, 2}, {3, 4}}.{{1, 2}, {3, 4}}", "{{7, 10}, {15, 22}}"),
    # Det
    ("Det[{{1, 2}, {3, 4}}]", "-2"), ("Det[{{a, b}, {c, d}}]", "a d - b c"), ("Det[{{1.5, 2}, {3, 4}}]", 0.0),
    ("Det[{{2.2, 1.2, 3.1}, {1.1, 3.1, 2.1}, {3.2, 2.6, 1.7}}]",
     float(np.linalg.det([[2.2, 1.2, 3.1], [1.1, 3.1, 2.1], [3.2, 2.6, 1.7]]))),
    ("Det[{{2, 3, 2, 3}, {1, 12, 1, 2}, {13, 3, 3, 1}, {1, 2, 12, 11}}]",
     str(sp.Matrix([[2, 3, 2, 3], [1, 12, 1, 2], [13, 3, 3, 1], [1, 2, 12, 11]]).det())),
    ("Det[{{Sqrt[2], 1}, {1, Sqrt[2]}}]", "1"), ("Det[HilbertMatrix[4]]", "1/6048000"),
    ("Det[{{1, 2, 3}, {4, 5, 6}}]", Raises()), ("Det[x]", Pred(lambda g: str(g) == "Det[x]", "unevaluated")),
    ("Det[{1, 2}]", Raises()), ("Det[{{1 + I, 2}, {3, 4}}]", "-2 + 4 I"), ("Det[{{x}}]", "x"),
    ("Precision[Det[N[{{1, 2}, {3, 4}}, 30]]]", Pred(lambda g: abs(float(g) - 30) < 1, "30 digits")),
    ("Det[N[{{1, 2}, {3, 4}}, 30]] + 2", Pred(lambda g: abs(complex(to_sympy(g))) < 1e-25, "-2 to 30 digits")),
    # Inverse
    ("Inverse[{{1, 2}, {3, 4}}]", "{{-2, 1}, {3/2, -1/2}}"), ("Inverse[{{1., 2}, {3, 4}}]", NP(np.linalg.inv(A))),
    ("Inverse[{{a, b}, {c, d}}]", "{{d/(a d - b c), -b/(a d - b c)}, {-c/(a d - b c), a/(a d - b c)}}"),
    ("Inverse[{{1, 2}, {2, 4}}]", Raises()), ("Inverse[{{1, 2, 3}, {4, 5, 6}}]", Raises()),
    ("Inverse[x]", Pred(lambda g: str(g) == "Inverse[x]", "unevaluated")),
    ("Inverse[HilbertMatrix[3]]", "{{9, -36, 30}, {-36, 192, -180}, {30, -180, 180}}"),
    ("Inverse[{{1, I}, {0, 1}}]", "{{1, -I}, {0, 1}}"), ("Inverse[{{2.}}]", NP([[0.5]])),
    # ConjugateTranspose
    ("ConjugateTranspose[{{1, 2 + I}, {3, 4}}]", "{{1, 3}, {2 - I, 4}}"),
    ("ConjugateTranspose[{{a, b}, {c, d}}]", "{{Conjugate[a], Conjugate[c]}, {Conjugate[b], Conjugate[d]}}"),
    ("ConjugateTranspose[{{1, 2, 3}, {4, 5, 6}}]", "{{1, 4}, {2, 5}, {3, 6}}"),
    ("ConjugateTranspose[{{1. + 2. I}}]", NP([[1 - 2j]])), ("ConjugateTranspose[x]", Pred(lambda g: g.head == "ConjugateTranspose", "unevaluated")),
    # Diagonal, triangularize
    ("Diagonal[{{1, 2, 3}, {4, 5, 6}}]", "{1, 5}"), ("Diagonal[{{1, 2, 3}, {4, 5, 6}}, 1]", "{2, 6}"),
    ("Diagonal[{{1, 2, 3}, {4, 5, 6}}, -1]", "{4}"), ("Diagonal[{{a, b}, {c, d}}, 5]", "{}"),
    ("Diagonal[IdentityMatrix[3]]", "{1, 1, 1}"), ("Diagonal[{1, 2}]", Raises()),
    ("UpperTriangularize[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]", "{{1, 2, 3}, {0, 5, 6}, {0, 0, 9}}"),
    ("UpperTriangularize[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]", "{{0, 2, 3}, {0, 0, 6}, {0, 0, 0}}"),
    ("UpperTriangularize[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, -1]", "{{1, 2, 3}, {4, 5, 6}, {0, 8, 9}}"),
    ("UpperTriangularize[{{a, b, c}, {d, e, f}}]", "{{a, b, c}, {0, e, f}}"),
    ("LowerTriangularize[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]", "{{1, 0, 0}, {4, 5, 0}, {7, 8, 9}}"),
    ("LowerTriangularize[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, -1]", "{{0, 0, 0}, {4, 0, 0}, {7, 8, 0}}"),
    ("LowerTriangularize[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, 1]", "{{1, 2, 0}, {4, 5, 6}, {7, 8, 9}}"),
    ("LowerTriangularize[{{a, b, c}, {d, e, f}}]", "{{a, 0, 0}, {d, e, 0}}"),
    ("UpperTriangularize[m] ", Pred(lambda g: g.head == "UpperTriangularize", "unevaluated")),
]

_EV3 = sorted(np.linalg.eigvals(np.array(A3, float)), key=lambda v: -abs(v))
SPECTRA = [
    # Eigenvalues / Eigenvectors with k; Eigensystem
    ("Eigenvalues[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}, 2]", "{3, 2}"),
    ("Eigenvalues[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}, -2]", "{2, 1}"),
    ("Eigenvalues[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}, UpTo[5]]", "{3, 2, 1}"),
    ("Eigenvalues[{{1, 0}, {0, 2}}, 3]", Raises()), ("Eigenvalues[{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}}, 1]", "{2 + Sqrt[2]}"),
    ("Eigenvalues[N[{{2, -1, 0}, {-1, 2, -1}, {0, -1, 2}}]]", NP(_EV3)),
    ("Eigenvalues[{{{1, 0}, {0, 1}}, {{2, 0}, {0, 1}}}, 1]", Raises()),
    ("Eigenvectors[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}, 1]", "{{0, 0, 1}}"),
    ("Eigenvectors[{{1, 0, 0}, {0, 2, 0}, {0, 0, 3}}, -1]", "{{1, 0, 0}}"),
    ("Eigenvectors[{{2, 1}, {1, 2}}, UpTo[1]]", "{{1, 1}}"),
    ("Eigensystem[{{1, 2}, {3, 4}}]",
     "{{5/2 + Sqrt[33]/2, 5/2 - Sqrt[33]/2}, {{-1/2 + Sqrt[33]/6, 1}, {-1/2 - Sqrt[33]/6, 1}}}"),
    ("Eigensystem[{{2, 0}, {0, 3}}, 1]", "{{3}, {{0, 1}}}"), ("Eigensystem[{{0, 1}, {-1, 0}}]", "{{I, -I}, {{-I, 1}, {I, 1}}}"),
    ("Eigensystem[{{2, 1}, {0, 2}}]", "{{2, 2}, {{1, 0}, {0, 0}}}"),
    ("Eigensystem[{{1., 2.}, {3., 4.}}]", Pred(lambda g: all(
        np.allclose(np.array(A) @ _arr(g.args[1].args[i]).ravel(), complex(to_sympy(g.args[0].args[i])) *
                    _arr(g.args[1].args[i]).ravel()) for i in range(2)), "A.v == lambda v")),
    ("Eigensystem[x]", Pred(lambda g: g.head == "Eigensystem", "unevaluated")),
    # CharacteristicPolynomial / MatrixMinimalPolynomial
    ("MatrixMinimalPolynomial[{{2, 0}, {0, 2}}, x]", "x - 2"),
    ("MatrixMinimalPolynomial[{{2, 1}, {0, 2}}, x]", "(x - 2)^2"),
    ("MatrixMinimalPolynomial[{{1, 2}, {3, 4}}, x]", "x^2 - 5 x - 2"),
    ("MatrixMinimalPolynomial[{{1, 0, 0}, {0, 1, 0}, {0, 0, 2}}, t]", "(t - 1) (t - 2)"),
    ("MatrixMinimalPolynomial[{{1., 0}, {0, 1}}, x]", Raises()),
    ("MatrixMinimalPolynomial[{{a, 0}, {0, a}}, x]", "x - a"),
    # SingularValueList (reference page examples)
    ("SingularValueList[{{1, 2}, {3, 4}}]", "{Sqrt[15 + Sqrt[221]], Sqrt[15 - Sqrt[221]]}"),
    ("SingularValueList[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]", Pred(lambda g: len(g.args) == 2 and np.allclose(
        [complex(to_sympy(v)) for v in g.args], np.linalg.svd(np.arange(1, 10).reshape(3, 3), compute_uv=False)[:2]), "2 values")),
    ("SingularValueList[{{1.1, 1.3, 2.1}, {2.2, 2.2, 3.3}, {3.3, -3, 4.7}}]", NP(np.linalg.svd(B, compute_uv=False))),
    ("SingularValueList[{{0.5, 1., 1.5}, {2., 2.5, 3.}, {3.5, 4., 4.5}}]",
     NP(np.linalg.svd([[0.5, 1, 1.5], [2, 2.5, 3], [3.5, 4, 4.5]], compute_uv=False)[:2])),
    ("Length[SingularValueList[{{0.5, 1., 1.5}, {2., 2.5, 3.}, {3.5, 4., 4.5}}, -2]]", "2"),
    ("SingularValueList[{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}}]", "{2, 2, 2}"),
    ("SingularValueList[{{7/2, 0, 1/2, 0}, {0, 3, 0, 1}, {1/2, 0, 7/2, 0}, {0, 1, 0, 3}}, 3]", "{4, 4, 3}"),
    ("SingularValueList[{{7/2, 0, 1/2, 0}, {0, 3, 0, 1}, {1/2, 0, 7/2, 0}, {0, 1, 0, 3}}, -3]", "{4, 3, 2}"),
    ("SingularValueList[{{3, 2, 2, 7}, {9, 3, -2, 1}}]", NP(np.linalg.svd([[3, 2, 2, 7], [9, 3, -2, 1]], compute_uv=False))),
    ("SingularValueList[IdentityMatrix[4]]", "{1, 1, 1, 1}"),
    ("SingularValueList[{{1, 0}, {0, 0}}, Tolerance -> 0]", "{1, 0}"),
    ("SingularValueList[{{a, b}, {c, d}}]", Raises()),
    # JordanDecomposition (defining property, Jordan form)
    ("JordanDecomposition[{{2, 1}, {0, 2}}][[2]]", "{{2, 1}, {0, 2}}"),
    ("With[{r = JordanDecomposition[{{1, 2}, {3, 4}}]}, Map[Simplify, r[[1]].r[[2]].Inverse[r[[1]]] - {{1, 2}, {3, 4}}, {2}]]",
     "{{0, 0}, {0, 0}}"),
    ("With[{r = JordanDecomposition[{{5, 4, 2, 1}, {0, 1, -1, -1}, {-1, -1, 3, 0}, {1, 1, -1, 2}}]}, "
     "{r[[1]].r[[2]].Inverse[r[[1]]], Sort[Diagonal[r[[2]]]], Total[Diagonal[r[[2]], 1]]}]",
     "{{{5, 4, 2, 1}, {0, 1, -1, -1}, {-1, -1, 3, 0}, {1, 1, -1, 2}}, {1, 2, 4, 4}, 1}"),
    ("With[{r = JordanDecomposition[{{1., 2.}, {3., 4.}}]}, r[[1]].r[[2]].Inverse[r[[1]]]]", NP(A)),
    ("JordanDecomposition[{{1., 1.}, {0., 1.}}]", Raises()),
    ("JordanDecomposition[{{1, 2, 3}, {4, 5, 6}}]", Raises()),
    # MatrixFunction
    ("MatrixFunction[Sin, {{0, 1}, {0, 0}}]", "{{0, 1}, {0, 0}}"),
    ("MatrixFunction[Exp, {{1, 2}, {3, 4}}] == MatrixExp[{{1, 2}, {3, 4}}]", True),
    ("MatrixFunction[#^2 &, {{1, 2}, {3, 4}}]", "{{7, 10}, {15, 22}}"),
    ("MatrixFunction[Exp, {{a, 1}, {0, a}}]", "{{E^a, E^a}, {0, E^a}}"),
    ("MatrixFunction[Sqrt, {{4., 1.}, {1., 3.}}]", NP(sla.sqrtm([[4., 1.], [1., 3.]]) if HAVE_SCIPY else None)),
    ("MatrixFunction[Cos, {{1., 2.}, {3., 4.}}]", NP(sla.cosm(np.array(A, float)) if HAVE_SCIPY else None)),
    ("MatrixFunction[Log, {{0, 0}, {0, 1}}]", Raises()),
]

MATRIX_OPS = [
    # Adjugate (reference identity Adjugate[m] == Det[m] Inverse[m]; defined for singular m)
    ("Adjugate[{{a, b}, {c, d}}]", "{{d, -b}, {-c, a}}"), ("Adjugate[{{1, 2}, {2, 4}}]", "{{4, -2}, {-2, 1}}"),
    ("Adjugate[{{2, 3, 2}, {1, 12, 1}, {13, 3, 3}}] == Det[{{2, 3, 2}, {1, 12, 1}, {13, 3, 3}}] Inverse[{{2, 3, 2}, {1, 12, 1}, {13, 3, 3}}]", True),
    ("Adjugate[{{1.5, 2.}, {3., 4.}}]", NP([[4, -2], [-3, 1.5]])), ("Adjugate[{{7}}]", "{{1}}"),
    ("Adjugate[{{1, 2, 3}, {4, 5, 6}}]", Raises()),
    # Minors (lexicographic subsets; Reverse[Minors[m], {1, 2}] gives the deletion minors)
    ("Minors[{{a, b, c}, {d, e, f}, {g, h, i}}]", "{{a e - b d, a f - c d, b f - c e}, {a h - b g, a i - c g, b i - c h}, "
                                                  "{d h - e g, d i - f g, e i - f h}}"),
    ("Minors[{{a, b, c}, {d, e, f}}, 2]", "{{a e - b d, a f - c d, b f - c e}}"), ("Minors[{{1, 2}, {3, 4}}, 0]", "{{1}}"),
    ("Minors[{{1, 2}, {3, 4}}, 2]", "{{-2}}"), ("Minors[IdentityMatrix[3], 2]", "IdentityMatrix[3]"),
    ("Dimensions[Minors[Partition[Range[20], 5], 3]]", "{4, 10}"),
    ("Minors[{{1, 2}, {3, 4}}, 1, f]", "{{f[{{1}}], f[{{2}}]}, {f[{{3}}], f[{{4}}]}}"),
    ("Minors[{{1, 2}, {3, 4}}, 3]", Raises()),
    ("Minors[Table[10 i + j, {i, 4}, {j, 5}], 3]", Pred(lambda g: all(v == 0 for v in _sm(g)), "all zero (rank 2)")),
    # Permanent
    ("Permanent[{{1, 2}, {3, 4}}]", "10"), ("Permanent[{{a, b}, {c, d}}]", "a d + b c"),
    ("Permanent[ConstantArray[1, {4, 4}]]", "24"), ("Permanent[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]", "450"),
    ("Permanent[{{1, 2}}]", Raises()),
    # Projection
    ("Projection[{5, 6, 7}, {1, 0, 0}]", "{5, 0, 0}"), ("Projection[{x, y}, {1, 1}]", "{(x + y)/2, (x + y)/2}"),
    ("Projection[{1, 2, 3}, {2, 1, 0}]", "{8/5, 4/5, 0}"), ("Projection[{1., 2.}, {3., 1.}]", NP([1.5, 0.5])),
    ("Projection[{1, 2}, {0, 0}]", Raises()), ("Projection[{1, 2}, {1, 2, 3}]", Raises()),
    ("Projection[x^2, 1, Integrate[#1 #2, {x, -1, 1}] &]", "1/3"),
    # Orthogonalize (reference examples and properties)
    ("Orthogonalize[{{1, 0, 1}, {1, 1, 1}}]", "{{1/Sqrt[2], 0, 1/Sqrt[2]}, {0, 1, 0}}"),
    ("Orthogonalize[{{1, 0, 1}, {1, 1, 1}, {2, 1, 2}}]", "{{1/Sqrt[2], 0, 1/Sqrt[2]}, {0, 1, 0}, {0, 0, 0}}"),
    ("Orthogonalize[{{2, 3}, {0, 2}, {3, 1}}]", "{{2/Sqrt[13], 3/Sqrt[13]}, {-3/Sqrt[13], 2/Sqrt[13]}, {0, 0}}"),
    ("Orthogonalize[{{1.25, 3}, {7.9, -1.4}}]", NP([[0.384615384615, 0.923076923077], [0.923076923077, -0.384615384615]], 1e-9)),
    ("Orthogonalize[{{1, 0}, {1, 1.*^-10}}, Tolerance -> 10^-5]", NP([[1, 0], [0, 0]])),
    ("Orthogonalize[{{1, 0}, {1, 1.*^-10}}]", NP([[1, 0], [0, 1]])),
    ("Orthogonalize[{1, x, x^2}, Integrate[#1 #2, {x, -1, 1}] &]",
     "{1/Sqrt[2], Sqrt[3/2] x, Sqrt[5/2] (3 x^2 - 1)/2}"),
    ("With[{u = Orthogonalize[{{1 + I, 2, 3 - 2 I}, {0, 4, 5 I}, {1, 0, 1}}]}, Map[Simplify, u.ConjugateTranspose[u], {2}]]",
     "IdentityMatrix[3]"),
    ("Orthogonalize[{{1, 2}, {3}}]", Raises()),
    # LeastSquares (reference: LeastSquares[{{1, 2}, {3, -4}, {1, -1}}, {0, -2, 1}])
    ("LeastSquares[{{1, 2}, {3, -4}, {1, -1}}, {0, -2, 1}]", "{-14/55, 1/5}"),
    ("LeastSquares[{{1., 2}, {3, -4}, {1, -1}}, {0, -2, 1}]",
     NP(np.linalg.lstsq([[1, 2], [3, -4], [1, -1]], [0, -2, 1], rcond=None)[0])),
    ("LeastSquares[{{1, 1}, {1, 1}}, {1, 3}]", "{1, 1}"), ("LeastSquares[{{1, 2}, {3, 4}}, {5, 6}]", "{-4, 9/2}"),
    ("LeastSquares[{{1, 0}, {0, 1}, {1, 1}}, {{1, 0}, {0, 1}, {1, 1}}]", "{{1, 0}, {0, 1}}"),
    ("LeastSquares[{{1, 2}, {3, 4}}, {1, 2, 3}]", Raises()),
]

CONSTRUCTORS = [
    ("HilbertMatrix[3]", "{{1, 1/2, 1/3}, {1/2, 1/3, 1/4}, {1/3, 1/4, 1/5}}"),
    ("HilbertMatrix[{2, 3}]", "{{1, 1/2, 1/3}, {1/2, 1/3, 1/4}}"), ("HilbertMatrix[0]", Raises()),
    ("N[HilbertMatrix[2]]", NP([[1, 0.5], [0.5, 1 / 3]])),
    ("ToeplitzMatrix[4]", "{{1, 2, 3, 4}, {2, 1, 2, 3}, {3, 2, 1, 2}, {4, 3, 2, 1}}"),
    ("ToeplitzMatrix[{a, b, c}]", "{{a, b, c}, {b, a, b}, {c, b, a}}"),
    ("ToeplitzMatrix[{1, 2, 3, 4, 5}, {1, 6, 7}]", "{{1, 6, 7}, {2, 1, 6}, {3, 2, 1}, {4, 3, 2}, {5, 4, 3}}"),
    ("ToeplitzMatrix[{1, 2, 3}, {1, 4, 5, 6, 7}]", "{{1, 4, 5, 6, 7}, {2, 1, 4, 5, 6}, {3, 2, 1, 4, 5}}"),
    ("ToeplitzMatrix[{c1, c2}, {r1, r2}]", "{{c1, r2}, {c2, c1}}"),
    ("Table[Det[ToeplitzMatrix[n]], {n, 6}]", "Table[(-2)^(n - 1) (n + 1)/2, {n, 6}]"),
    ("HankelMatrix[4]", "{{1, 2, 3, 4}, {2, 3, 4, 0}, {3, 4, 0, 0}, {4, 0, 0, 0}}"),
    ("HankelMatrix[{a, b, c}]", "{{a, b, c}, {b, c, 0}, {c, 0, 0}}"),
    ("HankelMatrix[{x, y, z}, {z, a, b, c, d}]", "{{x, y, z, a, b}, {y, z, a, b, c}, {z, a, b, c, d}}"),
    ("HankelMatrix[{1, 2, 3, 4}, {a, b, c, d}]", "{{1, 2, 3, 4}, {2, 3, 4, b}, {3, 4, b, c}, {4, b, c, d}}"),
    ("Table[Det[HankelMatrix[n]], {n, 6}]", "Table[(-1)^Floor[n/2] n^n, {n, 6}]"),
    ("VandermondeMatrix[{a, b, c}]", "{{1, a, a^2}, {1, b, b^2}, {1, c, c^2}}"),
    ("VandermondeMatrix[{1, 2, 3}, 4]", "{{1, 1, 1, 1}, {1, 2, 4, 8}, {1, 3, 9, 27}}"),
    ("Det[VandermondeMatrix[{a, b, c}]] == (b - a) (c - a) (c - b) // Simplify", True),
    ("VandermondeMatrix[{0.5, 1., 2.}]", NP(np.vander([0.5, 1, 2], increasing=True))),
    ("VandermondeMatrix[{1, 1, 2}]", Raises()),
    ("CauchyMatrix[{1, 2}, {3, 4}]", "{{1/4, 1/5}, {1/5, 1/6}}"), ("CauchyMatrix[{a, b}, {c, d}]", "{{1/(a + c), 1/(a + d)}, {1/(b + c), 1/(b + d)}}"),
    ("CauchyMatrix[{1, 2}]", "{{1/2, 1/3}, {1/3, 1/4}}"), ("CauchyMatrix[{1}, {-1}]", Raises()),
    ("FourierMatrix[4]", "{{1, 1, 1, 1}, {1, I, -1, -I}, {1, -1, 1, -1}, {1, -I, -1, I}}/2"),
    ("FourierMatrix[2]", "{{1, 1}, {1, -1}}/Sqrt[2]"),
    ("N[FourierMatrix[5]]", NP(np.exp(2j * np.pi * np.outer(range(5), range(5)) / 5) / np.sqrt(5))),
    ("UnitaryMatrixQ[FourierMatrix[6]]", True),
    # geometric transforms
    ("RotationMatrix[t]", "{{Cos[t], -Sin[t]}, {Sin[t], Cos[t]}}"),
    ("RotationMatrix[Pi/2, {0, 0, 1}]", "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"),
    ("RotationMatrix[t, {0, 0, 1}]", "{{Cos[t], -Sin[t], 0}, {Sin[t], Cos[t], 0}, {0, 0, 1}}"),
    ("RotationMatrix[t, {1, 0, 0}]", "{{1, 0, 0}, {0, Cos[t], -Sin[t]}, {0, Sin[t], Cos[t]}}"),
    ("RotationMatrix[2.0, {1, 2, 2}]", NP(sla.expm(2.0 * np.array([[0, -2, 2], [2, 0, -1], [-2, 1, 0]]) / 3) if HAVE_SCIPY else None)),
    ("RotationMatrix[{{1, 0}, {0, 1}}]", "{{0, -1}, {1, 0}}"), ("RotationMatrix[{{1, 0, 0}, {0, 1, 0}}]", "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"),
    ("RotationMatrix[{{1, 1}, {1, 1}}]", "{{1, 0}, {0, 1}}"),
    ("RotationMatrix[{{1, 0, 0}, {1, 1, 1}}].{1, 0, 0}", "{1, 1, 1}/Sqrt[3]"),
    ("RotationMatrix[Pi/2, {{1, 0, 0, 0}, {0, 0, 0, 1}}].{1, 0, 0, 0}", "{0, 0, 0, 1}"),
    ("RotationMatrix[{{1, 0}, {-1, 0}}]", Raises()), ("RotationMatrix[t, {0, 0, 0}]", Raises()),
    ("OrthogonalMatrixQ[RotationMatrix[0.7, {1., 2., 3.}]]", True),
    ("ReflectionMatrix[{1, 0}]", "{{-1, 0}, {0, 1}}"), ("ReflectionMatrix[{1, 1}]", "{{0, -1}, {-1, 0}}"),
    ("ReflectionMatrix[{1, 2, 2}]", "IdentityMatrix[3] - 2 Outer[Times, {1, 2, 2}, {1, 2, 2}]/9"),
    ("Det[ReflectionMatrix[{3, 1, 4, 1}]]", "-1"), ("ReflectionMatrix[{0, 0}]", Raises()),
    ("ScalingMatrix[{2, 3}]", "{{2, 0}, {0, 3}}"), ("ScalingMatrix[2, {1, 0}]", "{{2, 0}, {0, 1}}"),
    ("ScalingMatrix[s, {1, 1}]", "{{(1 + s)/2, (s - 1)/2}, {(s - 1)/2, (1 + s)/2}}"),
    ("ShearingMatrix[t, {1, 0}, {0, 1}]", "{{1, Tan[t]}, {0, 1}}"), ("ShearingMatrix[t, {0, 1}, {1, 0}]", "{{1, 0}, {Tan[t], 1}}"),
    ("ShearingMatrix[t, {1, 0, 0}, {0, 0, 1}]", "{{1, 0, Tan[t]}, {0, 1, 0}, {0, 0, 1}}"),
    ("ShearingMatrix[t, {1, 1}, {1, -1}]", "IdentityMatrix[2] + Tan[t] {{1, -1}, {1, -1}}/2"),
    ("Simplify[Det[ShearingMatrix[t, {1, 2, 0}, {0, 1, 1}]]]", "1"),
    ("ShearingMatrix[a, {1, 0}, {1, 1}] == ShearingMatrix[a, Projection[{1, 0}, {1, -1}], {1, 1}] // Simplify", True),
    ("ShearingMatrix[t, {1, 0}, {1, 0}]", Raises()),
    # structuring matrices
    ("BoxMatrix[1]", "ConstantArray[1, {3, 3}]"), ("BoxMatrix[1, 5]", "ArrayPad[ConstantArray[1, {3, 3}], 1]"
     if False else "{{0, 0, 0, 0, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {0, 0, 0, 0, 0}}"),
    ("BoxMatrix[{1, 0}]", "{{1}, {1}, {1}}"), ("Dimensions[BoxMatrix[{2, 3}]]", "{5, 7}"), ("BoxMatrix[1, 4]", Raises()),
    ("CrossMatrix[1]", "{{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}"), ("Total[CrossMatrix[3], 2]", "13"),
    ("CrossMatrix[1, 5][[3]]", "{0, 1, 1, 1, 0}"),
    ("DiamondMatrix[2]", "{{0, 0, 1, 0, 0}, {0, 1, 1, 1, 0}, {1, 1, 1, 1, 1}, {0, 1, 1, 1, 0}, {0, 0, 1, 0, 0}}"),
    ("Dimensions[DiamondMatrix[5]]", "{11, 11}"), ("Total[DiamondMatrix[4], 2]", "41"), ("DiamondMatrix[{1, 2}]", Raises()),
    # LeviCivitaTensor
    ("LeviCivitaTensor[2]", "{{0, 1}, {-1, 0}}"),
    ("LeviCivitaTensor[3]", "{{{0, 0, 0}, {0, 0, 1}, {0, -1, 0}}, {{0, 0, -1}, {0, 0, 0}, {1, 0, 0}}, "
                            "{{0, 1, 0}, {-1, 0, 0}, {0, 0, 0}}}"),
    ("LeviCivitaTensor[4, List][[1, 2, 3, 4]]", "1"), ("LeviCivitaTensor[4][[2, 1, 3, 4]]", "-1"),
    ("Total[Abs[Flatten[LeviCivitaTensor[4]]]]", "24"),
    ("LeviCivitaTensor[3].{a, b, c}", "{{0, c, -b}, {-c, 0, a}, {b, -a, 0}}"),
    ("LeviCivitaTensor[0]", Raises()),
]

PREDICATES = [
    ("SquareMatrixQ[{{1, 2}, {3, 4}}]", True), ("SquareMatrixQ[{{1, 2, 3}, {3, 4, 5}}]", False), ("SquareMatrixQ[x]", False),
    ("SquareMatrixQ[{1, 2}]", False),
    ("HermitianMatrixQ[{{1, I}, {-I, 2}}]", True), ("HermitianMatrixQ[{{1, I}, {I, 2}}]", False),
    ("HermitianMatrixQ[{{1., 2. + 1.*^-14 I}, {2., 3.}}]", True), ("HermitianMatrixQ[{{a, b}, {b, a}}]", False),
    ("HermitianMatrixQ[{{1, 2}, {2, 1}}]", True), ("HermitianMatrixQ[x]", False),
    ("AntisymmetricMatrixQ[{{0, 1}, {-1, 0}}]", True), ("AntisymmetricMatrixQ[{{0, a}, {-a, 0}}]", True),
    ("AntisymmetricMatrixQ[{{1, 1}, {-1, 0}}]", False), ("AntisymmetricMatrixQ[LeviCivitaTensor[3].{2, -1, 1}]", True),
    ("AntihermitianMatrixQ[{{I, 1}, {-1, 0}}]", True), ("AntihermitianMatrixQ[{{I, 1}, {1, 0}}]", False),
    ("AntihermitianMatrixQ[{{0, 1 + I}, {-1 + I, 2 I}}]", True),
    ("NormalMatrixQ[{{1, 2, -1}, {-1, 1, 2}, {2, -1, 1}}]", True), ("NormalMatrixQ[{{1, 1}, {0, 1}}]", False),
    ("NormalMatrixQ[{{1, 2}, {2, 1}}]", True), ("NormalMatrixQ[{{0, 1}, {-1, 0}}]", True),
    ("OrthogonalMatrixQ[{{1, -1}, {1, 1}}/Sqrt[2]]", True),
    ("OrthogonalMatrixQ[{{1, 0, 0}, {0, Cos[a], -Sin[a]}, {0, Sin[a], Cos[a]}}]", True),
    ("OrthogonalMatrixQ[{{1, 2}, {-2, 1}}/Sqrt[5]]", True), ("OrthogonalMatrixQ[{{4, -5, 2}, {3, -3, -3}, {5, 5, 5}}]", False),
    ("OrthogonalMatrixQ[{{2, -I}, {I, 2}}/Sqrt[3]]", True), ("OrthogonalMatrixQ[{{1, 1, 1, -1}, {-1, 1, 1, 1}}/2]", True),
    ("OrthogonalMatrixQ[{{1, -1}, {1, 1}, {1, 1}, {-1, 1}}/2]", True), ("OrthogonalMatrixQ[IdentityMatrix[{3, 4}]]", True),
    ("OrthogonalMatrixQ[{{a, b}, {b, -a}}]", False), ("OrthogonalMatrixQ[{{a, b}, {b, -a}}, Normalized -> False]", True),
    ("OrthogonalMatrixQ[HilbertMatrix[3]]", False),
    ("OrthogonalMatrixQ[{{1, 1.*^-13}, {0, 1}}]", True), ("OrthogonalMatrixQ[{{1, 1.*^-13}, {0, 1}}, Tolerance -> 10^-15]", False),
    ("UnitaryMatrixQ[{{2, -I}, {I, 2}}/Sqrt[3]]", False), ("UnitaryMatrixQ[{{1, I}, {I, 1}}/Sqrt[2]]", True),
    ("UnitaryMatrixQ[{{1, 2}, {-2, 1}}/Sqrt[5]]", True), ("UnitaryMatrixQ[{{1, 1}, {0, 1}}]", False),
    ("PositiveDefiniteMatrixQ[{{5, -1}, {-1, 4}}]", True), ("PositiveDefiniteMatrixQ[{{2.3, -1.2}, {0.6, 3.7}}]", True),
    ("PositiveDefiniteMatrixQ[{{1, 2 I}, {-I, 4}}]", True), ("PositiveDefiniteMatrixQ[{{1, a}, {b, 2}}]", False),
    ("PositiveDefiniteMatrixQ[{{4, -1}, {2, 0}}]", False), ("PositiveDefiniteMatrixQ[HilbertMatrix[5]]", True),
    ("PositiveDefiniteMatrixQ[N[HilbertMatrix[16]]]", False), ("PositiveDefiniteMatrixQ[Sqrt[3]]", False),
    ("PositiveDefiniteMatrixQ[{{6, 4}, {-2, 8}}]", True), ("PositiveDefiniteMatrixQ[{{1, 0, 0}, {0, 2, 0}, {0, 0, 0}}]", False),
    ("PositiveDefiniteMatrixQ[{{11, 6 + 4 I, 7 + 3 I, 10 + 3 I}, {6 - 4 I, 12, 8 - I, 8 - 2 I}, {7 - 3 I, 8 + I, 7, 9 - I}, "
     "{10 - 3 I, 8 + 2 I, 9 + I, 17}}]", True),
    ("PositiveDefiniteMatrixQ[{{Pi, -5, 2}, {E, -3, -3}, {5, Sqrt[2], 5}}]",
     bool(np.all(np.linalg.eigvalsh((lambda m: (m + m.T) / 2)(np.array([[np.pi, -5, 2], [np.e, -3, -3], [5, 2 ** .5, 5]]))) > 0))),
    ("PositiveSemidefiniteMatrixQ[{{6, 4}, {-2, 8}}]", True), ("PositiveSemidefiniteMatrixQ[N[HilbertMatrix[16]]]", True),
    ("PositiveSemidefiniteMatrixQ[{{1, 0, 0}, {0, 2, 0}, {0, 0, 0}}]", True), ("PositiveSemidefiniteMatrixQ[{{1, 0}, {0, -1}}]", False),
    ("PositiveSemidefiniteMatrixQ[{{1, 1}, {1, 1}}]", True),
    ("NegativeDefiniteMatrixQ[-{{5, -1}, {-1, 4}}]", True), ("NegativeDefiniteMatrixQ[{{5, -1}, {-1, 4}}]", False),
    ("NegativeDefiniteMatrixQ[{{-1., 0}, {0, -2.}}]", True),
    ("NegativeSemidefiniteMatrixQ[{{-1, 0}, {0, 0}}]", True), ("NegativeSemidefiniteMatrixQ[{{1, 0}, {0, 0}}]", False),
    ("NegativeSemidefiniteMatrixQ[{{3, -1}, {1, 2}}]", False),
    ("IndefiniteMatrixQ[{{1, 0}, {0, -1}}]", True), ("IndefiniteMatrixQ[{{1, 0}, {0, 1}}]", False),
    ("IndefiniteMatrixQ[{{3, -1}, {1, 2}}]", False), ("IndefiniteMatrixQ[{{0, 1}, {1, 0}}]", True),
    ("IndefiniteMatrixQ[{{a, 0}, {0, 1}}]", False),
    ("DiagonalizableMatrixQ[{{1, 1}, {0, 1}}]", False), ("DiagonalizableMatrixQ[{{1, 2}, {3, 4}}]", True),
    ("DiagonalizableMatrixQ[{{1., 1.}, {0., 1.}}]", False), ("DiagonalizableMatrixQ[{{0, 1}, {-1, 0}}]", True),
    ("DiagonalizableMatrixQ[{{1, 2, 3}, {4, 5, 6}}]", False),
]

TENSORS = [
    ("TensorProduct[{a, b}, {c, d}]", "{{a c, a d}, {b c, b d}}"),
    ("TensorProduct[{1, 2}, {{1, 0}, {0, 1}}]", "{{{1, 0}, {0, 1}}, {{2, 0}, {0, 2}}}"),
    ("TensorProduct[x, y]", Pred(lambda g: g.head == "TensorProduct", "unevaluated")),
    ("Dimensions[TensorProduct[{1, 2}, {1, 2, 3}, {1, 2, 3, 4}]]", "{2, 3, 4}"),
    ("TensorProduct[{1., 2.}, {3, 4}]", NP(np.outer([1, 2], [3, 4]))), ("TensorProduct[2, {1, 2}]", "{2, 4}"),
    ("TensorRank[{{1, 2}, {3, 4}}]", "2"), ("TensorRank[5]", "0"), ("TensorRank[LeviCivitaTensor[4]]", "4"),
    ("TensorRank[x]", Pred(lambda g: g.head == "TensorRank", "unevaluated")),
    ("TensorDimensions[{{1, 2, 3}, {4, 5, 6}}]", "{2, 3}"), ("TensorDimensions[7]", "{}"),
    ("TensorDimensions[Array[a, {2, 3, 4}]]", "{2, 3, 4}"),
    ("TensorContract[{{1, 2}, {3, 4}}, {{1, 2}}]", "5"),
    ("TensorContract[Array[a, {2, 2, 2}], {{1, 3}}]", "{a[1, 1, 1] + a[2, 1, 2], a[1, 2, 1] + a[2, 2, 2]}"),
    ("TensorContract[TensorProduct[{a, b}, {c, d}], {{1, 2}}]", "a c + b d"),
    ("TensorContract[LeviCivitaTensor[3], {{1, 2}}]", "{0, 0, 0}"),
    ("TensorContract[Array[a, {2, 3}], {{1, 2}}]", Raises()), ("TensorContract[{{1, 2}, {3, 4}}, {{1, 1}}]", Raises()),
    ("TensorTranspose[{{1, 2, 3}, {4, 5, 6}}, {2, 1}]", "{{1, 4}, {2, 5}, {3, 6}}"),
    ("Dimensions[TensorTranspose[Array[a, {2, 3, 4}], {2, 3, 1}]]", "{4, 2, 3}"),
    ("TensorTranspose[Array[a, {2, 3, 4}], {2, 3, 1}] == Transpose[Array[a, {2, 3, 4}], {2, 3, 1}]", True),
    ("TensorTranspose[{{1, 2}, {3, 4}}, {1, 1}]", Raises()),
    # ArrayRules / SparseArray (dense representation)
    ("ArrayRules[{{1, 0}, {0, 2}}]", "{{1, 1} -> 1, {2, 2} -> 2, {_, _} -> 0}"),
    ("ArrayRules[{1, 0, 3}]", "{{1} -> 1, {3} -> 3, {_} -> 0}"), ("ArrayRules[{1, 1, 3}, 1]", "{{3} -> 3, {_} -> 1}"),
    ("ArrayRules[{{1, 2}, {3}}]", Raises()),
    ("SparseArray[{{1, 1} -> 1, {2, 2} -> 2}]", "{{1, 0}, {0, 2}}"),
    ("SparseArray[{{1, 1} -> 1, {2, 2} -> 2}, {3, 3}]", "{{1, 0, 0}, {0, 2, 0}, {0, 0, 0}}"),
    ("SparseArray[{{i_, i_} -> 2, {i_, j_} /; Abs[i - j] == 1 -> -1}, {4, 4}]",
     "{{2, -1, 0, 0}, {-1, 2, -1, 0}, {0, -1, 2, -1}, {0, 0, -1, 2}}"),
    ("SparseArray[{1 -> a, 3 -> b}]", "{a, 0, b}"), ("SparseArray[{{1, 2}, {3, 1}} -> {x, y}]", "{{0, x}, {0, 0}, {y, 0}}"),
    ("SparseArray[{2 -> 1}, 4, x]", "{x, 1, x, x}"), ("SparseArray[{{1, 1} -> 1}, {2, 2}] . {3, 4}", "{3, 0}"),
    ("Normal[SparseArray[{{1, 2} -> 5}, {2, 2}]]", "{{0, 5}, {0, 0}}"),
    ("ArrayRules[SparseArray[{{1, 2} -> 5}, {2, 2}]]", "{{1, 2} -> 5, {_, _} -> 0}"),
    ("SparseArray[{{3, 3} -> 1}, {2, 2}]", Raises()),
    # MatrixForm is an inert display wrapper
    ("Head[MatrixForm[{{1, 2}, {3, 4}}]]", "MatrixForm"), ("First[MatrixForm[{{1, 2}, {3, 4}}]]", "{{1, 2}, {3, 4}}"),
    ("MatrixForm[{1, 2}] // Head", "MatrixForm"),
    # exact versus numeric
    ("Inverse[N[{{1, 2}, {3, 4}}]] - N[Inverse[{{1, 2}, {3, 4}}]]", ZERO("~0")),
    ("Head[Det[N[{{1, 2}, {3, 4}}]]]", "Real"), ("Head[Det[{{1, 2}, {3, 4}}]]", "Integer"),
]

CASES = [c for c in BASICS + SPECTRA + MATRIX_OPS + CONSTRUCTORS + PREDICATES + TENSORS
         if HAVE_SCIPY or not ("MatrixFunction[Sqrt" in c[0] or "MatrixFunction[Cos" in c[0] or "RotationMatrix[2.0" in c[0])]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_batch3(case):
    check(case)


def test_matrix_form_text():
    from mikoshilang.library.wl_linalg import matrix_form_text
    from mikoshilang.parser import parse
    from mikoshilang.evaluate import evaluate
    assert matrix_form_text(evaluate(parse("{{1, 20}, {300, 4}}"))) == "(   1  20 )\n( 300   4 )"
    assert matrix_form_text(evaluate(parse("{a, b}"))) == "( a )\n( b )"
