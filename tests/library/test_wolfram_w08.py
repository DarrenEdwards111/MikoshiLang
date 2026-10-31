"""Differential tests for Wolfram batch W08 (linear algebra extras, distances, optimisation, games).

Expected values are Wolfram Language results from the reference pages (reference.wolfram.com/language/ref/<Name>),
or results verified here through defining properties (m == s.j.Inverse[s], u.m.v == r, optimality conditions,
...) or independent NumPy / SciPy references.
"""
import importlib.util

import pytest

from _harness import Pred, Raises, check, ids, ev

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None


def T(src):
    """The expression must evaluate to True."""
    return (src, True)


def near(x, tol=1e-6):
    return Pred(lambda g: _close(g, x, tol), f"~{x}")


def _flat(e):
    from mikoshilang.expr import Expr
    if isinstance(e, Expr) and e.head == "List":
        out = []
        for a in e.args:
            out.extend(_flat(a))
        return out
    return [e]


def _close(got, want, tol):
    from _harness import to_complex
    if isinstance(want, (list, tuple)):
        g = _flat(got)
        w = _flatten_py(want)
        return len(g) == len(w) and all(abs(to_complex(a) - complex(b)) <= tol * max(1.0, abs(complex(b)))
                                        for a, b in zip(g, w))
    return abs(to_complex(got) - complex(want)) <= tol * max(1.0, abs(complex(want)))


def _flatten_py(x):
    if isinstance(x, (list, tuple)):
        out = []
        for a in x:
            out.extend(_flatten_py(a))
        return out
    return [x]


N50 = "N[{1, 5, 2, 3, 10}, 50], N[{4, 15, 20, 5, 5}, 50]"


def hp(digits, prec=50):
    """A bignum result agreeing with the Wolfram value to 45 digits and keeping ~prec digits of precision."""
    import mpmath
    import sympy as sp
    from mikoshilang.math_ops import to_sympy

    def ok(g):
        v = to_sympy(g)
        if not isinstance(v, sp.Float) or v._prec < (prec - 2) * 3.32:
            return False
        with mpmath.workdps(60):
            return abs(mpmath.mpf(str(v)) - mpmath.mpf(digits)) < mpmath.mpf(10) ** (-44) * max(1, abs(mpmath.mpf(digits)))
    return Pred(ok, f"~{digits}`{prec}")

DISTANCES = [
    ("EuclideanDistance[{a, b, c}, {x, y, z}]", "Sqrt[Abs[a - x]^2 + Abs[b - y]^2 + Abs[c - z]^2]"),
    ("EuclideanDistance[{1, 2, 3}, {2, 4, 6}]", "Sqrt[14]"),
    (f"EuclideanDistance[{N50}]", hp("21.494185260204677039244010575481712995963818467845")),
    ("EuclideanDistance[{a, b}, {a, c}]", "Abs[b - c]"),
    ("EuclideanDistance[3, 7.5]", "4.5"),
    ("EuclideanDistance[{1, 2}, {1, 2, 3}]", Raises()),
    ("SquaredEuclideanDistance[{a, b, c}, {x, y, z}]", "Abs[a - x]^2 + Abs[b - y]^2 + Abs[c - z]^2"),
    ("SquaredEuclideanDistance[{1, 2, 3}, {2, 4, 6}]", "14"),
    (f"SquaredEuclideanDistance[{N50}]", hp("462.0000000000000000000000000000000000000000000000")),
    ("ManhattanDistance[{a, b, c}, {x, y, z}]", "Abs[a - x] + Abs[b - y] + Abs[c - z]"),
    ("ManhattanDistance[{1, 2, 3}, {2, 4, 6}]", "6"),
    (f"ManhattanDistance[{N50}]", hp("38.00000000000000000000000000000000000000000000000")),
    ("ChessboardDistance[{a, b, c}, {x, y, z}]", "Max[Abs[a - x], Abs[b - y], Abs[c - z]]"),
    ("ChessboardDistance[{1, 2, 3}, {2, 4, 6}]", "3"),
    (f"ChessboardDistance[{N50}]", hp("18.000000000000000000000000000000000000000000000000")),
    ("BrayCurtisDistance[{1, 2, 3}, {2, 4, 6}]", "1/3"),
    ("BrayCurtisDistance[{a, b}, {x, y}]", "(Abs[a - x] + Abs[b - y])/(Abs[a + x] + Abs[b + y])"),
    (f"BrayCurtisDistance[{N50}]", hp("0.5428571428571428571428571428571428571428571428571")),
    ("CanberraDistance[{1, 2, 3}, {2, 4, 6}]", "1"),
    ("CanberraDistance[{a, b}, {a, c}]", "Abs[b - c]/(Abs[b] + Abs[c])"),
    (f"CanberraDistance[{N50}]", hp("2.5015151515151515151515151515151515151515151515152")),
    ("CorrelationDistance[{1, 2, 3}, {3, 5, 10}]", "1 - 7/(2 Sqrt[13])"),
    (f"CorrelationDistance[{N50}]", hp("1.2106635188398813533035274347761676650148587810028")),
    ("CorrelationDistance[{1, 1, 1}, {1, 2, 3}]", Raises()),
    ("CosineDistance[{1, 2, 3}, {3, 5, 7}]", "1 - 17 Sqrt[2/581]"),
    (f"CosineDistance[{N50}]", hp("0.4062940529572788872406931125582100673373718289257")),
    ("CosineDistance[{a, b}, {x, y}]",
     "1 - (a Conjugate[x] + b Conjugate[y])/(Sqrt[Abs[a]^2 + Abs[b]^2] Sqrt[Abs[x]^2 + Abs[y]^2])"),
    ("CosineDistance[{0, 0}, {1, 2}]", Raises()),
    ("NormalizedSquaredEuclideanDistance[{1, 2, 3}, {3, 5, 10}]", "1/4"),
    (f"NormalizedSquaredEuclideanDistance[{N50}]", hp("0.583333333333333333333333333333333333333333333333")),
    ("NormalizedSquaredEuclideanDistance[{1, 2, 3}, {2, 3, 4}]", "0"),
    ("BinaryDistance[{1, 2, 3}, {2, 4, 6}]", "1"), ("BinaryDistance[{a, b, c}, {a, b, c}]", "0"),
    ("BinaryDistance[\"text\", {1, 2, 3}]", "1"), ("BinaryDistance[{1, 2, 3}, {1, 2}]", "1"),
    ("BinaryDistance[a, b]", "Boole[a != b]"), ("BinaryDistance[1, 1.]", "0"),
    ("DiceDissimilarity[{1, 0, 1, 1, 0}, {1, 1, 0, 1, 1}]", "3/7"),
    ("DiceDissimilarity[{True, False, True}, {True, True, False}]", "1/2"),
    ("DiceDissimilarity[{1, 1, 1, 1}, {1, 1, 1, 1}]", "0"), ("DiceDissimilarity[{0, 0, 0, 0}, {1, 1, 1, 1}]", "1"),
    ("JaccardDissimilarity[{1, 0, 1, 1, 0}, {1, 1, 0, 1, 1}]", "3/5"),
    ("JaccardDissimilarity[{True, False, True}, {True, True, False}]", "2/3"),
    ("JaccardDissimilarity[{1, 1, 1, 1}, {1, 1, 1, 1}]", "0"),
    ("JaccardDissimilarity[{1, 2}, {1, 0}]", Raises()),
    ("MatchingDissimilarity[{1, 0, 1, 1, 0}, {1, 1, 0, 1, 1}]", "3/5"),
    ("MatchingDissimilarity[{True, False, True}, {True, True, False}]", "2/3"),
    ("MatchingDissimilarity[{0, 0, 0, 0}, {1, 1, 1, 1}]", "1"),
    ("RogersTanimotoDissimilarity[{1, 0, 1, 1, 0}, {1, 1, 0, 1, 1}]", "3/4"),
    ("RogersTanimotoDissimilarity[{True, False, True}, {True, True, False}]", "4/5"),
    ("RogersTanimotoDissimilarity[{0, 0, 0, 0}, {1, 1, 1, 1}]", "1"),
    ("RussellRaoDissimilarity[{1, 0, 1, 1, 0}, {1, 1, 0, 1, 1}]", "3/5"),
    ("RussellRaoDissimilarity[{True, False, True}, {True, True, False}]", "2/3"),
    ("RussellRaoDissimilarity[{1, 1, 1, 1}, {1, 1, 1, 1}]", "0"),
    ("SokalSneathDissimilarity[{1, 0, 1, 1, 0}, {1, 1, 0, 1, 1}]", "3/4"),
    ("SokalSneathDissimilarity[{True, False, True}, {True, True, False}]", "4/5"),
    ("SokalSneathDissimilarity[{0, 0, 0, 0}, {1, 1, 1, 1}]", "1"),
    ("YuleDissimilarity[{1, 0, 1, 1, 0}, {1, 1, 0, 1, 1}]", "2"),
    ("YuleDissimilarity[{True, False, True}, {True, True, False}]", "2"),
    ("YuleDissimilarity[{0, 0, 0, 0}, {0, 0, 0, 0}]", "0"), ("YuleDissimilarity[{0, 1, 0, 1}, {1, 0, 1, 0}]", "2"),
    ("DistanceMatrix[{1, 2, 3, 4}]", "{{0, 1, 2, 3}, {1, 0, 1, 2}, {2, 1, 0, 1}, {3, 2, 1, 0}}"),
    ("DistanceMatrix[{1, 2, 3, 4}, {5, 6, 7}]", "{{4, 5, 6}, {3, 4, 5}, {2, 3, 4}, {1, 2, 3}}"),
    ("DistanceMatrix[{{1.5, 4.3}, {2.1, 5.6}, {6.2, 4.5}}, {{1.6, 4.2}, {2.3, 5.4}}]",
     near([[0.141421, 1.36015], [1.48661, 0.282843], [4.60977, 4.0025]], 1e-5)),
    ("DistanceMatrix[{\"abc\", \"abd\", \"x\"}]", "{{0, 1, 3}, {1, 0, 3}, {3, 3, 0}}"),
    ("DistanceMatrix[{{1, 2}, {4, 6}}, DistanceFunction -> ManhattanDistance]", "{{0, 7}, {7, 0}}"),
    ("WarpingDistance[{11, 12, 13, 14, 15}, {12, 13}]", "4."),
    ("WarpingDistance[{Pi, -E, E + Pi, -EulerGamma}, {GoldenRatio, -E * Pi, E ^ Pi, -Pi ^ E}]", near(46.5078, 1e-5)),
    ("WarpingDistance[PrimeQ /@ Range[1, 20], CoprimeQ[#, 6]& /@ Range[1, 20]]", "1."),
    ("WarpingDistance[{1, 2, 3}, {1, 2, 3}]", "0."),
    ("WarpingDistance[{1, 5, 1, 5}, {1, 1, 5, 5}, {\"Band\", 0}]", "8."),
    ("WarpingDistance[{1, 5, 1, 5}, {1, 1, 5, 5}]", "4."),
    ("WarpingCorrespondence[{1, 2, 3, 4, 5}, {1, 2, 3}]", "{{1, 2, 3, 4, 5}, {1, 2, 3, 3, 3}}"),
    ("WarpingCorrespondence[{1, 2, 3}, {1, 2, 2, 2, 3}]", "{{1, 2, 2, 2, 3}, {1, 2, 3, 4, 5}}"),
    ("WarpingCorrespondence[{1, 5, 1, 5}, {1, 1, 5, 5}, {\"Band\", 0}]", "{{1, 2, 3, 4}, {1, 2, 3, 4}}"),
    ("WarpingCorrespondence[{1, 2}, {}]", Raises()),
    ("FeatureDistance[{1, 2}, {4, 6}, Identity]", "5"),
    ("FeatureDistance[#^2 &][2, 3]", "5"),
    ("FeatureDistance[{1, 2}, {4, 6}, Identity, DistanceFunction -> ManhattanDistance]", "7"),
    ("FeatureDistance[1, 2, FeatureExtractorFunction[x]]", Raises()),
    ("Nearest[{{5, 9}, {2, 4}, {3, 8}, {0, 8}, {5, 2}}, {5, 5}, DistanceFunction -> ChessboardDistance]",
     "{{2, 4}, {3, 8}, {5, 2}}"),
    ("DistanceMatrix[{{1, 0}, {1, 1}}, DistanceFunction -> DiceDissimilarity]", "{{0, 1/3}, {1/3, 0}}"),
    ("Head[DistanceFunction]", "Symbol"),
]

CONSTRUCTORS = [
    # Band in SparseArray (reference examples)
    ("SparseArray[{Band[{1, 1}] -> x, Band[{2, 1}] -> y}, {5, 5}]",
     "{{x, 0, 0, 0, 0}, {y, x, 0, 0, 0}, {0, y, x, 0, 0}, {0, 0, y, x, 0}, {0, 0, 0, y, x}}"),
    ("SparseArray[{Band[{1, 1}] -> x, {3, 5} -> b}, {5, 5}]",
     "{{x, 0, 0, 0, 0}, {0, x, 0, 0, 0}, {0, 0, x, 0, b}, {0, 0, 0, x, 0}, {0, 0, 0, 0, x}}"),
    ("SparseArray[Band[{1, 1}] -> {x, y, z}, {5, 5}]",
     "{{x, 0, 0, 0, 0}, {0, y, 0, 0, 0}, {0, 0, z, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}"),
    ("SparseArray[Band[{1, 1}, {5, 5}] -> {x, y, z}, {5, 5}]",
     "{{x, 0, 0, 0, 0}, {0, y, 0, 0, 0}, {0, 0, z, 0, 0}, {0, 0, 0, x, 0}, {0, 0, 0, 0, y}}"),
    ("SparseArray[Band[{1, 2}, {3, 4}] -> x, {5, 5}]",
     "{{0, x, 0, 0, 0}, {0, 0, x, 0, 0}, {0, 0, 0, x, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}"),
    ("SparseArray[Band[{1, 1}, {5, 5}, 2] -> x, {5, 5}]",
     "{{x, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, x, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, x}}"),
    ("SparseArray[Band[{3, 3}, Automatic, {0, 1}] -> x, {5, 5}]",
     "{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, x, x, x}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}"),
    ("SparseArray[Band[{1, 5}, Automatic, {1, -1}] -> x, {5, 5}]",
     "{{0, 0, 0, 0, x}, {0, 0, 0, x, 0}, {0, 0, x, 0, 0}, {0, x, 0, 0, 0}, {x, 0, 0, 0, 0}}"),
    ("SparseArray[Band[{3, 3}] -> {{a, b}, {c, d}}, {5, 5}]",
     "{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, a, b, 0}, {0, 0, c, d, 0}, {0, 0, 0, 0, 0}}"),
    ("SparseArray[Band[{1, 1}, {5, 5}] -> {{{a, b}, {c, d}}}, {5, 5}]",
     "{{a, b, 0, 0, 0}, {c, d, 0, 0, 0}, {0, 0, a, b, 0}, {0, 0, c, d, 0}, {0, 0, 0, 0, a}}"),
    ("SparseArray[Band[{1, 1}, {5, 5}] -> {{{a, b}, {c, d}}, {{x}}}, {5, 5}]",
     "{{a, b, 0, 0, 0}, {c, d, 0, 0, 0}, {0, 0, x, 0, 0}, {0, 0, 0, a, b}, {0, 0, 0, c, d}}"),
    ("SparseArray[{Band[{2, 1}] -> a, Band[{1, 1}] -> b, Band[{1, 2}] -> c}, 5].Array[x, 5]",
     "{b x[1] + c x[2], a x[1] + b x[2] + c x[3], a x[2] + b x[3] + c x[4], a x[3] + b x[4] + c x[5], a x[4] + b x[5]}"),
    ("SparseArray[Band[{1, 1}] -> {a, b, c}]", "DiagonalMatrix[{a, b, c}]"),
    ("SparseArray[Band[{1, 1}, {-1, -1}] -> {{{1, 2}, {2, 1}}}, {4, 4}]",
     "{{1, 2, 0, 0}, {2, 1, 0, 0}, {0, 0, 1, 2}, {0, 0, 2, 1}}"),
    ("Total[Flatten[SparseArray[Band[{1, 1, 1}] -> x, {3, 3, 3}]]]", "3 x"),
    ("Head[Band[{1, 1}]]", "Band"),
    ("SparseArray[Band[{1, 1}] -> x]", Raises()),
    # CompanionMatrix
    ("CompanionMatrix[{5, 2, 3}]", "{{0, 0, -5}, {1, 0, -2}, {0, 1, -3}}"),
    ("CompanionMatrix[3 + x - 2x ^ 2 + 4x ^ 3, x]", "{{0, 0, -(3/4)}, {1, 0, -(1/4)}, {0, 1, (1/2)}}"),
    ("CompanionMatrix[{a, b, c}]", "{{0, 0, -a}, {1, 0, -b}, {0, 1, -c}}"),
    ("CompanionMatrix[{1, 3, 5, 7}, Bottom]", "{{0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {-1, -3, -5, -7}}"),
    ("CompanionMatrix[{1, 3, 5, 7}, Top]", "{{-7, -5, -3, -1}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}"),
    ("CompanionMatrix[{1, 3, 5, 7}, Left]", "{{-7, 1, 0, 0}, {-5, 0, 1, 0}, {-3, 0, 0, 1}, {-1, 0, 0, 0}}"),
    ("CompanionMatrix[{1.5, 3.8, -4.2, 5.1, -2.3}]",
     "{{0, 0, 0, 0, -1.5}, {1, 0, 0, 0, -3.8}, {0, 1, 0, 0, 4.2}, {0, 0, 1, 0, -5.1}, {0, 0, 0, 1, 2.3}}"),
    ("Inverse[CompanionMatrix[{c0, c1, c2, c3}]]",
     "{{-(c1/c0), 1, 0, 0}, {-(c2/c0), 0, 1, 0}, {-(c3/c0), 0, 0, 1}, {-(1/c0), 0, 0, 0}}"),
    ("Det[CompanionMatrix[{c0, c1, c2, c3}]]", "c0"),
    ("CompanionMatrix[6 + 92 x + 13 x^2 - x^3, x]", "{{0, 0, 6}, {1, 0, 92}, {0, 1, 13}}"),
    ("CompanionMatrix[2 + 3x + 4x ^ 2 + x ^ 3, x] == CompanionMatrix[{2, 3, 4}]", "True"),
    ("CompanionMatrix[5, x]", Raises()),
    # HadamardMatrix
    ("HadamardMatrix[4]",
     "{{(1/2), (1/2), (1/2), (1/2)}, {(1/2), (1/2), -(1/2), -(1/2)}, {(1/2), -(1/2), -(1/2), (1/2)}, "
     "{(1/2), -(1/2), (1/2), -(1/2)}}"),
    ("HadamardMatrix[4, Method -> \"BitComplement\"] == KroneckerProduct[{{1, 1}, {1, -1}}, {{1, 1}, {1, -1}}]/2",
     "True"),
    ("Map[Total[Abs[Differences[Sign[#]]]]/2 &, HadamardMatrix[8, Method -> \"BitComplement\"]]",
     "{0, 7, 3, 4, 1, 6, 2, 5}"),
    ("Map[Total[Abs[Differences[Sign[#]]]]/2 &, HadamardMatrix[8, Method -> \"GrayCode\"]]",
     "{0, 1, 3, 2, 7, 6, 4, 5}"),
    ("Map[Total[Abs[Differences[Sign[#]]]]/2 &, HadamardMatrix[8]]", "{0, 1, 2, 3, 4, 5, 6, 7}"),
    ("With[{h = HadamardMatrix[8]}, {SymmetricMatrixQ[h], h.h == IdentityMatrix[8]}]", "{True, True}"),
    ("HadamardMatrix[2, WorkingPrecision -> MachinePrecision]", near([[0.70710678, 0.70710678], [0.70710678, -0.70710678]])),
    ("HadamardMatrix[6]", Raises()),
    # JordanMatrix
    ("JordanMatrix[1 / 3, 2]", "{{(1/3), 1}, {0, (1/3)}}"),
    ("JordanMatrix[Sqrt[2], 3]", "{{Sqrt[2], 1, 0}, {0, Sqrt[2], 1}, {0, 0, Sqrt[2]}}"),
    ("JordanMatrix[a, 4, Right]", "{{a, 0, 0, 0}, {1, a, 0, 0}, {0, 1, a, 0}, {0, 0, 1, a}}"),
    ("JordanMatrix[a, 4] == Transpose[JordanMatrix[a, 4, Right]]", "True"),
    ("JordanMatrix[{3}, 2]", Raises()),
    # PermutationMatrix
    ("PermutationMatrix[{4, 1, 3, 2}]", "{{0, 0, 0, 1}, {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}}"),
    ("PermutationMatrix[Cycles[{{1, 4, 3}, {2, 5}}]]",
     "{{0, 0, 0, 1, 0}, {0, 0, 0, 0, 1}, {1, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 0, 0, 0}}"),
    ("PermutationMatrix[2 <-> 6]",
     "{{1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1}, {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0}, "
     "{0, 1, 0, 0, 0, 0}}"),
    ("Dimensions[PermutationMatrix[Cycles[{{1, 4, 3}, {2, 5}}], 7]]", "{7, 7}"),
    ("Det[PermutationMatrix[{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}]]", "1"),
    ("Det[PermutationMatrix[{4, 1, 3, 2}]] == Signature[{4, 1, 3, 2}]", "True"),
    ("PermutationMatrix[{1, 1, 2}]", Raises()),
    # BlockDiagonalMatrix and triangular / structured converters
    ("BlockDiagonalMatrix[{{{a11}}, {{b11, b12}, {b21, b22}}}]", "{{a11, 0, 0}, {0, b11, b12}, {0, b21, b22}}"),
    ("Det[BlockDiagonalMatrix[{{{a11}}, {{b11, b12}, {b21, b22}}}]]", "a11 (b11 b22 - b12 b21)"),
    ("BlockDiagonalMatrix[{{{3, 2}, {2, 3}}, {}, {{1}}}]", "{{3, 2, 0}, {2, 3, 0}, {0, 0, 1}}"),
    ("BlockDiagonalMatrix[HilbertMatrix[3]]", "HilbertMatrix[3]"),
    ("BlockDiagonalMatrix[{{{1, 2}}}]", Raises()),
    ("LowerTriangularMatrix[{{2, 0, 0}, {1, -5, 0}, {-3, 2, 4}}]", "{{2, 0, 0}, {1, -5, 0}, {-3, 2, 4}}"),
    ("Det[LowerTriangularMatrix[{{2, 0, 0}, {1, -5, 0}, {-3, 2, 4}}]]", "-40"),
    ("LowerTriangularMatrix[{{1, 2}, {3, 4}}]", Raises()),
    ("UpperTriangularMatrix[{{2, 3, 0}, {0, -5, 1}, {0, 0, 4}}]", "{{2, 3, 0}, {0, -5, 1}, {0, 0, 4}}"),
    ("Inverse[UpperTriangularMatrix[{{2, 3}, {0, 4}}]]", "{{1/2, -3/8}, {0, 1/4}}"),
    ("UpperTriangularMatrix[{{1, 2}, {3, 4}}]", Raises()),
    ("BlockLowerTriangularMatrix[{{1, 0, 0}, {2, 3, 4}, {5, 6, 7}}]", "{{1, 0, 0}, {2, 3, 4}, {5, 6, 7}}"),
    ("BlockLowerTriangularMatrix[HilbertMatrix[3]]", "HilbertMatrix[3]"),
    ("BlockLowerTriangularMatrix[{{1, 2}}]", Raises()),
    ("BlockUpperTriangularMatrix[{{1, 2, 3}, {0, 4, 5}, {0, 6, 7}}]", "{{1, 2, 3}, {0, 4, 5}, {0, 6, 7}}"),
    ("BlockUpperTriangularMatrix[HilbertMatrix[2]]", "HilbertMatrix[2]"),
    ("BlockUpperTriangularMatrix[{1, 2}]", Raises()),
    ("HermitianMatrix[{{1, 1 + I, 2I}, {1 - I, 5, -3}, {-2I, -3, 0}}]", "{{1, 1 + I, 2 I}, {1 - I, 5, -3}, {-2 I, -3, 0}}"),
    ("HermitianMatrix[{{1, 1 + I, 2I}, {5, -3}, {0}}]", "{{1, 1 + I, 2 I}, {1 - I, 5, -3}, {-2 I, -3, 0}}"),
    ("HermitianMatrix[{{1}, {1 - I, 5}, {-2I, -3, 0}}]", "{{1, 1 + I, 2 I}, {1 - I, 5, -3}, {-2 I, -3, 0}}"),
    ("HermitianMatrix[{{1, I}, {I, 1}}]", Raises()),
    ("SymmetricMatrix[{{8, 1, 7}, {1, 5, 2}, {7, 2, 3}}]", "{{8, 1, 7}, {1, 5, 2}, {7, 2, 3}}"),
    ("SymmetricMatrix[{{8, 1, 7}, {5, 2}, {3}}]", "{{8, 1, 7}, {1, 5, 2}, {7, 2, 3}}"),
    ("SymmetricMatrix[{{8}, {1, 5}, {7, 2, 3}}]", "{{8, 1, 7}, {1, 5, 2}, {7, 2, 3}}"),
    ("SymmetricMatrix[{{1, 2}, {3, 4}}]", Raises()),
    ("OrthogonalMatrix[{{1/2, 1/2, 1/2, 1/2}, {1/2, 1/2, -1/2, -1/2}, {1/2, -1/2, -1/2, 1/2}, {1/2, -1/2, 1/2, -1/2}}]",
     "HadamardMatrix[4]"),
    ("OrthogonalMatrix[RotationMatrix[t]]", "RotationMatrix[t]"),
    ("OrthogonalMatrix[{{1, 1}, {1, -1}}]", Raises()),
    ("UnitaryMatrix[{{1, I}, {I, 1}}/Sqrt[2]]", "{{1, I}, {I, 1}}/Sqrt[2]"),
    ("UnitaryMatrix[{{1/2, 1/2, 1/2, 1/2}, {1/2, I/2, -1/2, -I/2}, {1/2, -1/2, 1/2, -1/2}, {1/2, -I/2, -1/2, I/2}}] // Length",
     "4"),
    ("UnitaryMatrix[{{1, I}, {I, 1}}]", Raises()),
    # ArrayDot
    ("ArrayDot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, 1]", "{{19, 22}, {43, 50}}"),
    ("ArrayDot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, 2]", "70"),
    ("ArrayDot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}, {{1, 1}}]", "{{26, 30}, {38, 44}}"),
    ("ArrayDot[{1, 2}, {3, 4}, 0]", "{{3, 4}, {6, 8}}"),
    ("ArrayDot[Array[a, {2, 3, 4}], Array[b, {3, 4, 5}], 2] == "
     "Flatten[Array[a, {2, 3, 4}], {{1}, {2, 3}}] . Flatten[Array[b, {3, 4, 5}], {{1, 2}, {3}}]", "True"),
    ("ArrayDot[{1, 2}, {1, 2, 3}, 1]", Raises()),
]

_M4 = "{{7, 1, -3, -4}, {2, 1, -1, -2}, {-1, 0, 1, 1}, {8, 1, -4, -5}}"
_SYM3 = "{{1., 2., 3.}, {2., -1., 0.5}, {3., 0.5, 2.}}"
_A43 = "{{1.2, 3.4}, {5.6, 7.8}, {9.10, 11.12}, {13.14, 15.16}}"

DECOMPOSITIONS = [
    # LDLDecomposition (Wolfram reference outputs)
    ("LDLDecomposition[{{2, 6}, {6, 7}}]", "{{{1, 0}, {3, 1}}, {{2, 0}, {0, -11}}}"),
    ("LDLDecomposition[{{1, a}, {Conjugate[a], 5}}]",
     "{{{1, 0}, {Conjugate[a], 1}}, {{1, 0}, {0, 5 - a Conjugate[a]}}}"),
    ("LDLDecomposition[{{Pi, 5, 0}, {5, 0, 0}, {0, 0, 2}}]",
     "{{{1, 0, 0}, {5/Pi, 1, 0}, {0, 0, 1}}, {{Pi, 0, 0}, {0, -25/Pi, 0}, {0, 0, 2}}}"),
    ("LDLDecomposition[{{Pi, 0, 0}, {0, 5, 10}, {0, 10, 0}}]",
     "{{{1, 0, 0}, {0, 1, 0}, {0, 2, 1}}, {{Pi, 0, 0}, {0, 5, 0}, {0, 0, -20}}}"),
    ("LDLDecomposition[IdentityMatrix[3]]", "{IdentityMatrix[3], IdentityMatrix[3]}"),
    ("LDLDecomposition[{{2., 1.}, {1., 3.}}]", near([[1, 0], [0.5, 1], [2, 0], [0, 2.5]])),
    ("LDLDecomposition[{{0, 1}, {1, 0}}]", Raises(ValueError, "zero pivot")),
    ("LDLDecomposition[{{1, 2}, {3, 4}}]", Raises(ValueError, "Hermitian")),
    # BunchKaufmanDecomposition: P.m.P^T == L.B.L^H
    T("m = {{0, 1, 2}, {1, 0, 3}, {2, 3, 0}}; {l, b, p} = BunchKaufmanDecomposition[m]; "
      "p.m.Transpose[p] == l.b.ConjugateTranspose[l]"),
    ("BunchKaufmanDecomposition[{{0, 1, 2}, {1, 0, 3}, {2, 3, 0}}][[2]]", "{{0, 2, 0}, {2, 0, 0}, {0, 0, -3}}"),
    T("m = " + _SYM3 + "; {l, b, p} = BunchKaufmanDecomposition[m]; "
      "Max[Abs[p.m.Transpose[p] - l.b.Transpose[l]]] < 10^-12"),
    T("m = {{4, 1 + I}, {1 - I, 3}}; {l, b, p} = BunchKaufmanDecomposition[m]; "
      "Expand[p.m.Transpose[p] - l.b.ConjugateTranspose[l]] == {{0, 0}, {0, 0}}"),
    ("BunchKaufmanDecomposition[{{1, 2}, {3, 4}}]", Raises(ValueError, "not Hermitian or symmetric")),
    ("BunchKaufmanDecomposition[{{a, b}, {b, c}}]", Raises(NotImplementedError, "symbolic")),
    # RankDecomposition
    ("RankDecomposition[{{1, 3}, {2, 6}}]", "{{{1}, {2}}, {{1, 3}}}"),
    ("RankDecomposition[{{1, 2, 3, 4}, {2, 7, 3, 9}, {2, 7, 3, 9}}]",
     "{{{1, 2}, {2, 7}, {2, 7}}, {{1, 0, 5, 10/3}, {0, 1, -1, 1/3}}}"),
    ("RankDecomposition[{{3/2, 19/4, -16/5}, {-17/10, 67/10, -93/10}, {49/10, -865/100, 154/10}}]",
     "{{{3/2, 19/4}, {-17/10, 67/10}, {49/10, -173/20}}, {{1, 0, 4547/3625}, {0, 1, -3878/3625}}}"),
    ("RankDecomposition[{{a, b, c}, {c, b, a}, {0, 0, 0}}]",
     "{{{a, b}, {c, b}, {0, 0}}, {{1, 0, -1}, {0, 1, (a + c)/b}}}"),
    ("RankDecomposition[" + _A43 + "]",
     near([[0.127002, 0.832764], [0.359099, 0.422887], [0.538509, 0.0195669], [0.751616, -0.356775],
           [16.94, 20.6155], [-1.14248, 0.938791]], 1e-5)),
    ("RankDecomposition[{{0, 0}, {0, 0}}]", Raises(ValueError, "rank 0")),
    # EigenvalueDecomposition
    ("EigenvalueDecomposition[{{-3, 2}, {-15, 8}}]", "{{{1, 2}, {3, 5}}, {{3, 0}, {0, 2}}}"),
    ("EigenvalueDecomposition[{{1, 2}, {3, 4}}]",
     "{{{(-3 + Sqrt[33])/6, (-3 - Sqrt[33])/6}, {1, 1}}, {{(5 + Sqrt[33])/2, 0}, {0, (5 - Sqrt[33])/2}}}"),
    ("EigenvalueDecomposition[{{1.1, 2.2, 3.25}, {0.76, 4.6, 5}, {0.1, 0.1, 6.1}}][[2]]",
     near([[6.60674, 0, 0], [0, 4.52536, 0], [0, 0, 0.667901]], 1e-5)),
    T("m = {{1.1, 2.2, 3.25}, {0.76, 4.6, 5}, {0.1, 0.1, 6.1}}; {t, d} = EigenvalueDecomposition[m]; "
      "Max[Abs[m - t.d.Inverse[t]]] < 10^-12"),
    ("EigenvalueDecomposition[{{1, 1}, {0, 1}}]", Raises(ValueError, "not diagonalizable")),
    # SmithDecomposition / SmithReduce
    ("SmithDecomposition[{{1, 2, 3}, {5, 4, 3}, {8, 7, 9}}][[2]]", "{{1, 0, 0}, {0, 3, 0}, {0, 0, 6}}"),
    T("m = {{1, 2, 3}, {5, 4, 3}, {8, 7, 9}}; {u, r, v} = SmithDecomposition[m]; "
      "u.m.v == r && Abs[Det[u]] == 1 && Abs[Det[v]] == 1"),
    T("m = {{1/2, 5, 4}, {3, 1, 2}, {3, 5, 4}}; {u, r, v} = SmithDecomposition[m]; "
      "u.m.v == r && r == {{1/2, 0, 0}, {0, 1, 0}, {0, 0, 30}}"),
    T("m = {{2, 4, 4}, {-6, 6, 12}, {10, -4, -16}}; {u, r, v} = SmithDecomposition[m]; "
      "u.m.v == r && r == {{2, 0, 0}, {0, 6, 0}, {0, 0, 12}}"),
    ("SmithReduce[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]", "{{1, 0, 0}, {0, 3, 0}, {0, 0, 0}}"),
    ("SmithReduce[{{1/2, 5, 4}, {3, 1, 2}, {3, 5, 4}}]", "{{1/2, 0, 0}, {0, 1, 0}, {0, 0, 30}}"),
    ("SmithReduce[{{0, 0, 1}, {0, 2, 0}, {3, 0, 0}}]", "{{1, 0, 0}, {0, 1, 0}, {0, 0, 6}}"),
    ("SmithReduce[IdentityMatrix[{3, 4}]]", "IdentityMatrix[{3, 4}]"),
    ("SmithReduce[{{1., 2.}, {3., 4.}}]", Raises(ValueError, "exact")),
    ("SmithDecomposition[{{a, 1}, {2, 3}}]", Raises(NotImplementedError, "symbolic")),
    # FrobeniusDecomposition / FrobeniusReduce
    ("FrobeniusDecomposition[{{2, 3, 1}, {3, 1, 2}, {3, 2, 4}}][[2]]", "{{0, 0, -15}, {1, 0, 2}, {0, 1, 7}}"),
    T("m = {{1, 0, 0, 0}, {0, 1, 0, 0}, {-2, -2, 0, 1}, {-2, 0, -1, 2}}; {s, c} = FrobeniusDecomposition[m]; "
      "s.c.Inverse[s] == m && c == {{1, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, -3}, {0, 0, 1, 3}}"),
    T("m = {{1 + I, 0, 3 - I}, {-2 - 3 I, 1, 4 I}, {2, -5 + 2 I, 0}}; {s, c} = FrobeniusDecomposition[m]; "
      "Expand[s.c.Inverse[s] - m] == ConstantArray[0, {3, 3}]"),
    ("FrobeniusReduce[{{1 + I, 0, 3 - I}, {-2 - 3 I, 1, 4 I}, {2, -5 + 2 I, 0}}]",
     "{{0, 0, 41 + 47 I}, {1, 0, -3 - 23 I}, {0, 1, 2 + I}}"),
    ("Expand[FrobeniusReduce[{{1 + x, y}, {-2 z^2 + x, 4 y z}}]]",
     "{{0, x y - 4 y z - 4 x y z - 2 y z^2}, {1, 1 + x + 4 y z}}"),
    ("FrobeniusReduce[{{1, -3, 4}, {3, -7, 6}, {-2, -2, 19}}]", "{{0, 0, 6}, {1, 0, 92}, {0, 1, 13}}"),
    ("FrobeniusReduce[{{2, 0, 0}, {0, 2, 0}, {0, 0, 3}}]", "{{2, 0, 0}, {0, 0, -6}, {0, 1, 5}}"),
    ("FrobeniusReduce[{{1, 2}, {3, 4}}, Modulus -> 7]", Raises(NotImplementedError, "Modulus")),
    ("FrobeniusReduce[{{1., 2.}, {3., 4.}}]", Raises(NotImplementedError, "inexact")),
    # JordanReduce
    ("JordanReduce[{{-1.2, 2.7, 3.8}, {4.2, 4.4, 5.3}, {3.5, 7.6, 6.8}}]",
     near([[13.7715, 0, 0], [0, -2.12527, 0], [0, 0, -1.6462]], 1e-5)),
    ("JordanReduce[{{27, 48, 81}, {-6, 0, 0}, {1, 0, 3}}]", "{{6, 0, 0}, {0, 12, 1}, {0, 0, 12}}"),
    T("JordanReduce[{{27, 48, 81}, {-6, 0, 0}, {1, 0, 3}}] == BlockDiagonalMatrix[{{{6}}, JordanMatrix[12, 2]}]"),
    ("JordanReduce[{{1, 1}, {0, 1}}]", "{{1, 1}, {0, 1}}"),
    # CoreNilpotentDecomposition / DrazinInverse (Wolfram reference outputs)
    ("CoreNilpotentDecomposition[{{1, 2, 0}, {5, 1, 3}, {0, 0, 0}}]",
     "{{{1, 2, -2}, {5, 1, 1}, {0, 0, 3}}, {{1, 2}, {5, 1}}, {{0}}}"),
    ("CoreNilpotentDecomposition[{{14, -22, 2}, {7, -11, 1}, {0, 0, 0}}]",
     "{{{14, -1, 11}, {7, 0, 7}, {0, 7, 0}}, {{3}}, {{0, 0}, {0, 0}}}"),
    ("CoreNilpotentDecomposition[{{-4 + 6 I, 8 - 10 I, 2}, {-2 + 3 I, 4 - 5 I, 1}, {1, -1, 1}}]",
     "{{{-4 + 6 I, 8 - 10 I, -5}, {-2 + 3 I, 4 - 5 I, -3}, {1, -1, 2}}, "
     "{{-(3/2) + 6 I, (11/2) - 10 I}, {-(1/2) + 3 I, (5/2) - 5 I}}, {{0}}}"),
    ("CoreNilpotentDecomposition[{{1, 2}, {3, 4}}]", "{IdentityMatrix[2], {{1, 2}, {3, 4}}, {}}"),
    ("CoreNilpotentDecomposition[{{0, 1}, {0, 0}}]", "{IdentityMatrix[2], {}, {{0, 1}, {0, 0}}}"),
    T("m = " + _M4 + "; {t, c, n} = CoreNilpotentDecomposition[m]; "
      "m == t.BlockDiagonalMatrix[{c, n}].Inverse[t] && n.n == {{0, 0}, {0, 0}} && Det[c] != 0"),
    ("CoreNilpotentDecomposition[{{0.2, -0.32}, {0.1, -0.16}}][[2 ;; 3]]", near([[0.04], [0.]])),
    ("DrazinInverse[{{0.2, -0.32}, {0.1, -0.16}}]", near([[125., -200.], [62.5, -100.]], 1e-9)),
    ("DrazinInverse[{{-4 + 6 I, 8 - 10 I, 2}, {-2 + 3 I, 4 - 5 I, 1}, {1, -1, 1}}]",
     "{{-5 - 5 I, 9 + 9 I, 1 + I}, {-(5/2) - (5 I/2), (9/2) + (9 I/2), (1/2) + (I/2)}, "
     "{(1/2) + (5 I/2), -(1/2) - (9 I/2), (1/2) - (I/2)}}"),
    ("DrazinInverse[{{1, 2, -2}, {4, 5, -3}, {1, 5, -3}}]",
     "{{0, 1/3, -1/3}, {-3/4, 1/12, 5/12}, {-5/4, 1/4, 1/4}}"),
    ("DrazinInverse[{{a, a, 1}, {0, 0, 0}, {2, 3, -1}}]",
     "{{1/(2 + a), 3/(2 + a)^2, 1/(2 + a)}, {0, 0, 0}, {2/(2 + a), (6 + 2 a + a^2)/(2 + a)^2, -a/(2 + a)}}"),
    ("DrazinInverse[{{0, 1}, {0, 0}}]", "{{0, 0}, {0, 0}}"),
    T("m = " + _M4 + "; d = DrazinInverse[m]; d.m == m.d && d.m.d == d"),
    ("DrazinInverse[m]", "DrazinInverse[m]"),
    # PolarDecomposition
    T("m = {{1., 2.}, {3., 4.}}; {u, p} = PolarDecomposition[m]; Max[Abs[m - u.p]] < 10^-12 && "
      "Max[Abs[Transpose[u].u - IdentityMatrix[2]]] < 10^-12 && Max[Abs[p - Transpose[p]]] < 10^-12"),
    ("PolarDecomposition[{{1., 2.}, {3., 4.}}][[2]]", near([[2.05798, 2.40098], [2.40098, 3.77297]], 1e-5)),
    T("m = {{1. + I, 2.}, {0.5, 3. - 2. I}}; {u, p} = PolarDecomposition[m]; Max[Abs[m - u.p]] < 10^-12 && "
      "Max[Abs[ConjugateTranspose[u].u - IdentityMatrix[2]]] < 10^-12"),
    T("m = {{1., 2., 3.}, {4., 5., 6.}}; {u, p} = PolarDecomposition[m]; Dimensions[u] == {2, 3} && "
      "Dimensions[p] == {3, 3} && Max[Abs[m - u.p]] < 10^-12"),
    ("PolarDecomposition[{{1, 2}, {3, 4}}]", Raises(NotImplementedError, "machine")),
    # OrderedSchurDecomposition (needs SciPy)
    ("OrderedSchurDecomposition[{{2.7, 4.8, 8.1}, {-0.6, 0, 0}, {0.1, 0, 0.3}}][[2]]",
     near([[1.2, -2.90297, 8.24441], [0, 1.2, -4.0942], [0, 0, 0.6]], 1e-5), "scipy"),
    ("Diagonal[OrderedSchurDecomposition[{{1., 2.}, {3., 4.}}, \"IncreasingRe\"][[2]]]",
     near([-0.372281, 5.37228], 1e-5), "scipy"),
    ("Diagonal[OrderedSchurDecomposition[{{1., 0., 0.}, {0., -3., 0.}, {0., 0., 2.}}][[2]]]",
     near([-3, 2, 1]), "scipy"),
    ("Diagonal[OrderedSchurDecomposition[{{1., 0., 0.}, {0., -3., 0.}, {0., 0., 2.}}, \"IncreasingAbs\"][[2]]]",
     near([1, 2, -3]), "scipy"),
    (("m = {{1.81066, 0.31066, 1.5}, {-0.53033, 2.03033, 0.43934}, {-0.96967, -0.53033, 2.56066}}; "
      "{q, t} = OrderedSchurDecomposition[m]; {Max[Abs[m - q.t.Transpose[q]]] < 10^-12, t[[2, 1]] != 0, "
      "t[[3, 2]] == 0}"), "{True, True, True}", "scipy"),
    (("m = {{1., 2.}, {3., 4.}}; {q, t} = OrderedSchurDecomposition[m, RealBlockDiagonalForm -> False]; "
      "Max[Abs[m - q.t.ConjugateTranspose[q]]] < 10^-12"), True, "scipy"),
    (("{m, a} = {{{1., 2.}, {3., 4.}}, {{2., 0.}, {1., 1.}}}; {q, s, p, t} = OrderedSchurDecomposition[{m, a}]; "
      "{Max[Abs[m - q.s.Transpose[p]]] < 10^-12, Max[Abs[a - q.t.Transpose[p]]] < 10^-12, "
      "Abs[s[[1, 1]]/t[[1, 1]]] >= Abs[s[[2, 2]]/t[[2, 2]]]}"), "{True, True, True}", "scipy"),
    ("OrderedSchurDecomposition[{{1., 2.}, {3., 4.}}, \"Sideways\"]", Raises(ValueError, "unknown ordering")),
    # KarhunenLoeveDecomposition
    ("KarhunenLoeveDecomposition[{{1, 2, 4}, {2, 3, 10}}]",
     near([[2.21649, 3.48733, 10.7572], [0.295268, 0.915726, 0.530689], [0.32519, 0.945649],
           [0.945649, -0.32519]], 1e-5)),
    ("KarhunenLoeveDecomposition[{{{7, 3}, {0, 1}}, {{6, 8}, {10, 6}}, {{3, 8}, {3, 1}}}]",
     near([5.246, 2.65484, -2.27548, -0.734427, 3.81481, 9.83288, 6.67585, 3.1046, 7.20602, 5.76769, 7.69774,
           5.27466, 0.889127, -0.324397, 0.32283, -0.18142, 0.397775, 0.899367, 0.420166, 0.85822, -0.29482],
          1e-5)),
    ("KarhunenLoeveDecomposition[{{0, 0, 1}, {0, 1, 1}}, RotationMatrix[45 Degree]][[1]]",
     near([[0., 0.707107, 1.41421], [0., 0.707107, 0.]], 1e-5)),
    ("KarhunenLoeveDecomposition[{{0, 0, 1}}, RotationMatrix[45 Degree]][[1]]",
     near([[0., 0., 0.707107], [0., 0., -0.707107]], 1e-5)),
    ("Abs[First[KarhunenLoeveDecomposition[{{1, 2, 3, 4}, {0.8, 2.1, 3.2, 3.9}}, Standardized -> True]]]",
     near([[2.26549, 0.634514, 0.85168, 2.04832], [0.0870348, 0.085975, 0.121001, 0.119941]], 1e-5)),
    ("KarhunenLoeveDecomposition[{{1, 2}, {1, 2, 3}}]", Raises(ValueError, "same dimensions")),
    # PrincipalComponents (component signs are not unique)
    ("Abs[PrincipalComponents[N[{{1, 2}, {2, 3}, {4, 10}}]]]",
     near([[3.27053, 0.285293], [1.99969, 0.335165], [5.27023, 0.0498715]], 1e-5)),
    ("Abs[PrincipalComponents[{{1., 2., -1.}, {2., 3., 2.}, {4., 10., 9.}, {4., 5., 6.}}, "
     "Method -> \"Correlation\"]]",
     near([[1.81962, 0.226286, 0.0503109], [0.873896, 0.0527284, 0.0788127], [1.94708, 0.412467, 0.00116417],
           [0.746438, 0.586024, 0.0273376]], 1e-5)),
    T("pc = PrincipalComponents[{{1., 2., 0.5}, {2., 3., 2.}, {4., 10., 9.}, {4., 5., 6.}, {0., 1., 1.}}]; "
      "Max[Abs[Mean[pc]]] < 10^-12 && Max[Abs[Covariance[pc] - DiagonalMatrix[Diagonal[Covariance[pc]]]]] < "
      "10^-10 && OrderedQ[Reverse[Diagonal[Covariance[pc]]]]"),
    ("PrincipalComponents[{{a, 1}, {2, a - 1}}]", Raises(NotImplementedError, "symbolic")),
    # PfaffianDet
    ("PfaffianDet[{{0, 3, 2, 1}, {-3, 0, 5, 6}, {-2, -5, 0, 4}, {-1, -6, -4, 0}}]", "5"),
    ("PfaffianDet[{{0, I}, {-I, 0}}]", "I"),
    ("PfaffianDet[{{0, a}, {-a, 0}}]", "a"),
    ("PfaffianDet[{{0, a, b, c}, {-a, 0, d, e}, {-b, -d, 0, f}, {-c, -e, -f, 0}}]", "a f - b e + c d"),
    ("PfaffianDet[{{0, 1, 2}, {-1, 0, 3}, {-2, -3, 0}}]", "0"),
    ("PfaffianDet[{{0, 1.5, 2, 1}, {-1.5, 0, 5, 6}, {-2, -5, 0, 4}, {-1, -6, -4, 0}}]", near(-1.0)),
    T("m = {{0, 2, -1, 3, 1, 4}, {-2, 0, 5, -2, 3, 1}, {1, -5, 0, 2, -4, 6}, {-3, 2, -2, 0, 7, -1}, "
      "{-1, -3, 4, -7, 0, 2}, {-4, -1, -6, 1, -2, 0}}; PfaffianDet[m]^2 == Det[m]"),
    ("PfaffianDet[{{0, 3, 2, 1}, {-3, 0, 5, 6}, {-2, -5, 0, 4}, {-1, -6, -4, 0}}, Method -> \"Det\"]", "5"),
    ("PfaffianDet[{{1, 2}, {3, 4}}]", Raises(ValueError, "antisymmetric")),
    # LinearSolve[m] / LinearSolveFunction
    ("LinearSolve[{{1, 2}, {3, -2}}][{5, 6}]", "{11/4, 9/8}"),
    ("LinearSolve[{{1, 2}, {3, -2}}][{{5, 6}, {7, 8}}]", "{{3, 7/2}, {1, 5/4}}"),
    ("Head[LinearSolve[{{1, 2}, {3, 4}}]]", "LinearSolveFunction"),
    ("f = LinearSolve[HilbertMatrix[6]]; f[{1, 1, 1, 1, 1, 1}]", "{-6, 210, -1680, 5040, -6300, 2772}"),
    ("f = LinearSolve[N[HilbertMatrix[3]]]; f[{1, 1, 1}]", near([3, -24, 30], 1e-8)),
    ("LinearSolve[{{1, 2}, {2, 4}}]", Raises(ValueError, "singular")),
    # TransformationMatrix
    ("TransformationMatrix[TransformationFunction[{{1, 0, x0}, {0, 1, y0}, {0, 0, 1}}]]",
     "{{1, 0, x0}, {0, 1, y0}, {0, 0, 1}}"),
    ("TransformationMatrix[TransformationFunction[{{Cos[t], -Sin[t], 0}, {Sin[t], Cos[t], 0}, {0, 0, 1}}]]",
     "{{Cos[t], -Sin[t], 0}, {Sin[t], Cos[t], 0}, {0, 0, 1}}"),
    ("TransformationMatrix[TransformationFunction[{{2, 0}, {0, 1}}]].{x, 1}", "{2 x, 1}"),
    ("TransformationMatrix[f]", "TransformationMatrix[f]"),
]



def _flat_f(e):
    from _harness import to_complex
    return [to_complex(v).real for v in _flat(e)]

VECTOR_INEQUALITIES = [
    T("VectorLess[{{1, 2}, {3, 4}}]"),
    ("VectorLess[{{1, 5}, {3, 4}}]", False),
    ("VectorLess[{{1, 2}, {1, 4}}]", False),
    ("VectorLess[{x, y}]", "VectorLess[{x, y}]"),
    T("VectorLessEqual[{{1, 2}, {1, 4}}]"),
    T("VectorLessEqual[{{1, 2}, {3, 4}}, \"NonNegativeCone\"]"),
    ("VectorLessEqual[{{2, 2}, {1, 4}}]", False),
    T("VectorGreater[{{3, 5}, {1, 4}}]"),
    ("VectorGreater[{{1, 2}, {0, 3}}]", False),
    T("VectorGreater[{{3, 5}, 0}]"),
    T("VectorGreaterEqual[{{3, 4}, {3, 4}}]"),
    T("VectorGreaterEqual[{{1, 2}, 0}]"),
    ("VectorGreaterEqual[{{2, 1, 1}, 0}, \"NormCone\"]", False),
    T("VectorGreaterEqual[{{1, 1, 2}, 0}, \"NormCone\"]"),
    T("VectorGreaterEqual[{{{2, 1}, {1, 2}}, 0}, {\"SemidefiniteCone\", 2}]"),
    ("VectorGreaterEqual[{{{1, 2}, {2, 1}}, 0}, {\"SemidefiniteCone\", 2}]", False),
    ("Hold[x \\[VectorLess] y]", "Hold[VectorLess[{x, y}]]"),
    T("{1, 2} \\[VectorLessEqual] {1, 4} \\[VectorLessEqual] {2, 5}"),
    T("{3, 1} \\[VectorGreater] {1, 0}"),
    ("{3, 1} \\[VectorGreaterEqual] {3, 2}", False),
    ("Hold[a < b \\[VectorLess] c]", "Hold[a < b && VectorLess[{b, c}]]"),
]

LOCAL_OPTIMISATION = [
    ("FindArgMin[(x - 1)^2 + 2, x]", near([1.0])),
    ("FindArgMin[(x - 1)^2 + (y + 2)^2, {x, y}]", near([1.0, -2.0])),
    ("FindArgMin[{x^2, x >= 1}, x]", near([1.0])),
    ("FindArgMax[-(x - 3)^2, {x, 0}]", near([3.0])),
    ("FindArgMax[{x + y, x^2 + y^2 <= 1}, {x, y}]", near([0.7071067811865476, 0.7071067811865476], 1e-5), "scipy"),
    ("FindArgMax[Sin[x], {x, 1}]", near([1.5707963267948966], 1e-5), "scipy"),
    ("FindMinValue[(x - 1)^2 + 2, x]", near(2.0)),
    ("FindMinValue[{x^2, x >= 1}, x]", near(1.0)),
    ("FindMinValue[x^4 - 2 x^2, {x, 2}]", near(-1.0), "scipy"),
    ("FindMaxValue[Sin[x], {x, 1}]", near(1.0), "scipy"),
    ("FindMaxValue[-(x - 1)^2 - (y - 2)^2 + 5, {x, y}]", near(5.0)),
    ("FindMaxValue[{x, x^2 <= 4}, x]", near(2.0, 1e-5), "scipy"),
]

GLOBAL_NUMERIC_OPTIMISATION = [
    ("NArgMin[(x - 2)^2, x]", near(2.0)),
    ("NArgMin[{x + y, x >= 1, y >= 2}, {x, y}]", near([1.0, 2.0])),
    ("NArgMin[{x^2 + y^2, x + y >= 2}, {x, y}]", near([1.0, 1.0], 1e-5)),
    ("NArgMax[-(x - 1)^2, x]", near(1.0)),
    ("NArgMax[{x + 2 y, x >= 0, y >= 0, x + y <= 1}, {x, y}]", near([0.0, 1.0])),
    ("NArgMax[{x y, x + y == 2}, {x, y}]", Raises()),
    ("NMinValue[(x - 3)^2 + 1, x]", near(1.0)),
    ("NMinValue[{x + y, x^2 + y^2 <= 2}, {x, y}]", near(-2.0, 1e-6), "scipy"),
    ("NMinValue[{x, x >= 3}, x]", near(3.0)),
    ("NMaxValue[{x, x^2 <= 4}, x]", near(2.0, 1e-6), "scipy"),
    ("NMaxValue[-(x - 1)^2 + 4, x]", near(4.0)),
    ("NMaxValue[{x + y, x <= 1, y <= 2}, {x, y}]", near(3.0)),
]

LINEAR_OPTIMISATION = [
    ("LinearOptimization[x + y, {x + 2 y >= 3, x >= 0, y >= 0}, {x, y}]", "{x -> 0, y -> 3/2}"),
    ("LinearOptimization[-x - y, {x + 2 y <= 4, 3 x + y <= 6, x >= 0, y >= 0}, {x, y}]", "{x -> 8/5, y -> 6/5}"),
    ("LinearOptimization[x + y, {x + 2 y >= 3, x >= 0, y >= 0}, {x, y}, \"PrimalMinimumValue\"]", "3/2"),
    ("LinearOptimization[2 x + 3 y, {x + y >= 4, x - y <= 2, x >= 0, y >= 0}, {x, y}, \"PrimalMinimizerVector\"]",
     "{3, 1}"),
    ("LinearOptimization[x + y, {x + 2 y >= 3.5, x >= 0, y >= 0}, {Element[x, Integers], Element[y, Integers]}]",
     "{x -> 0, y -> 2}"),
    ("LinearOptimization[{1, 1}, {{{1, 2}, {1, 0}, {0, 1}}, {-3, 0, 0}}]", "{0, 3/2}"),
    ("LinearOptimization[x, {x >= 1, x <= 0}, {x}]", Raises()),
    ("LinearOptimization[-x, {x >= 0}, {x}]", Raises()),
]

QUADRATIC_OPTIMISATION = [
    ("{x, y} /. QuadraticOptimization[x^2 + y^2, {x + y >= 1}, {x, y}]", near([0.5, 0.5])),
    ("{x, y} /. QuadraticOptimization[(x - 1)^2 + (y - 2)^2, {x + y <= 1}, {x, y}]", near([0.0, 1.0])),
    ("QuadraticOptimization[x^2 + y^2, {x + y == 2}, {x, y}, \"PrimalMinimumValue\"]", near(2.0)),
    ("QuadraticOptimization[x^2 + 2 y^2 - x, {x >= 0, y >= 0}, {x, y}, {\"PrimalMinimizer\", \"PrimalMinimumValue\"}]",
     near([0.5, 0.0, -0.25])),
    ("QuadraticOptimization[-x^2, {x <= 1, x >= -1}, {x}]", Raises()),
]

LINEAR_FRACTIONAL_OPTIMISATION = [
    ("LinearFractionalOptimization[(x + 1)/(y + 1), {x >= 1, y <= 2, x + y <= 4, y >= 0}, {x, y}]", "{x -> 1, y -> 2}"),
    ("LinearFractionalOptimization[x/(x + y + 1), {x >= 1, y >= 1, x + y <= 3}, {x, y}]", "{x -> 1, y -> 2}"),
    ("LinearFractionalOptimization[(x + 2)/(y + 1), {x >= 0, y >= 0, x <= 2, y <= 3}, {x, y}, \"PrimalMinimumValue\"]",
     "1/2"),
]

CONVEX_OPTIMISATION = [
    ("{x, y} /. ConvexOptimization[x^2 + y^2, {x + y >= 2}, {x, y}]", near([1.0, 1.0], 1e-6), "scipy"),
    ("x /. ConvexOptimization[Exp[x] + x^2, {}, {x}]", near(-0.35173371124919584, 1e-6), "scipy"),
    ("ConvexOptimization[Abs[x - 1] + Abs[y + 1], {x + y >= 3}, {x, y}, \"PrimalMinimumValue\"]", near(3.0, 1e-6),
     "scipy"),
    ("ConvexOptimization[-x^2, {-1 <= x <= 1}, {x}]", Raises()),
    ("{x, y} /. ConicOptimization[x + y, {VectorGreaterEqual[{{x, y}, 0}], x + y >= 1}, {x, y}]",
     Pred(lambda g: abs(sum(_flat_f(g)) - 1) < 1e-6, "x + y == 1"), "scipy"),
    ("x /. ConicOptimization[x, {VectorGreaterEqual[{{x, 1, 1}, 0}, \"NormCone\"]}, {x}]", near(0.0, 1e-5), "scipy"),
    ("ConicOptimization[-x - y, {VectorGreaterEqual[{{1, x, y}, 0}, \"NormCone\"]}, {x, y}]", Raises(), "scipy"),
    ("{x, y} /. SecondOrderConeOptimization[-x - y, {Norm[{x, y}] <= 1}, {x, y}]",
     near([0.7071067811865476, 0.7071067811865476], 1e-6), "scipy"),
    ("{x, y} /. SecondOrderConeOptimization[x, {Norm[{x - 1, y}] <= 2}, {x, y}]", near([-1.0, 0.0], 1e-6), "scipy"),
    ("SecondOrderConeOptimization[x + y, {Norm[{x, y}] <= 1}, {x, y}, \"PrimalMinimumValue\"]",
     near(-1.4142135623730951, 1e-6), "scipy"),
    ("x /. SemidefiniteOptimization[x, {VectorGreaterEqual[{{{x, 1}, {1, x}}, 0}, {\"SemidefiniteCone\", 2}]}, {x}]",
     near(1.0, 1e-6), "scipy"),
    ("{x, y} /. SemidefiniteOptimization[x + y, {VectorGreaterEqual[{{{x, 1}, {1, y}}, 0}, {\"SemidefiniteCone\", 2}]}, "
     "{x, y}]", near([1.0, 1.0], 1e-5), "scipy"),
    ("SemidefiniteOptimization[x, {VectorGreaterEqual[{{{x, 0}, {0, x}}, {{1, 0}, {0, 3}}}, {\"SemidefiniteCone\", 2}]}, "
     "{x}, \"PrimalMinimumValue\"]", near(3.0, 1e-6), "scipy"),
    ("{x, y} /. GeometricOptimization[x + y, {x y >= 1}, {x, y}]", near([1.0, 1.0], 1e-6), "scipy"),
    ("x /. GeometricOptimization[1/x + x, {}, {x}]", near(1.0, 1e-6), "scipy"),
    ("GeometricOptimization[x y, {x >= 2, y >= 3}, {x, y}, \"PrimalMinimumValue\"]", near(6.0, 1e-6), "scipy"),
    ("GeometricOptimization[x - y, {x >= 1}, {x, y}]", Raises()),
]

PARAMETRIC_ROBUST = [
    ("x /. ParametricConvexOptimization[(x - a)^2, {x >= 0}, {x}, {a}][3]", near(3.0, 1e-6), "scipy"),
    ("x /. ParametricConvexOptimization[(x - a)^2, {x >= 0}, {x}, {a}][-2]", near(0.0, 1e-6), "scipy"),
    ("ParametricConvexOptimization[x^2 + y^2, {x + y >= a}, {x, y}, {a}, \"PrimalMinimumValue\"][2]", near(2.0, 1e-6),
     "scipy"),
    ("Head[ParametricConvexOptimization[(x - a)^2, {x >= 0}, {x}, {a}]]", "ParametricFunction"),
    ("x /. RobustConvexOptimization[x, {x >= a, -1 <= a <= 2}, {x}, {a}]", near(2.0, 1e-6), "scipy"),
    ("RobustConvexOptimization[x + y, {x + a y >= 1, x >= 0, y >= 0, 0 <= a <= 1}, {x, y}, {a}, \"PrimalMinimumValue\"]",
     near(1.0, 1e-6), "scipy"),
    ("x /. RobustConvexOptimization[-x, {x <= 3 + a, -1 <= a <= 1}, {x}, {a}]", near(2.0, 1e-6), "scipy"),
]

KNAPSACK_TOURS = [
    ("KnapsackSolve[{4, 2, 3}, 8]", "{0, 4, 0}"),
    ("KnapsackSolve[{{3, 4}, {2, 5}}, 10]", "{2, 0}"),
    ("KnapsackSolve[{{3, 2, 7}, {2, 1, 5}}, 10]", "{3, 4}"),
    ("KnapsackSolve[{{3, 2, 7}, {2, 1, 5}}, {15, 10, 5}]", "{5, 0}"),
    ("KnapsackSolve[<|\"apples\" -> {8, 3, 3}, \"oranges\" -> {9, 7, 5}|>, 25]", "<|\"apples\" -> 3, \"oranges\" -> 2|>"),
    ("KnapsackSolve[{4, 2, 3}, 11]", "{0, 5, 0}"),
    ("KnapsackSolve[{{3, 2}, {4, 3}, {5, 4}}, 6]", "{3, 0, 0}"),
    ("KnapsackSolve[{{3, 2, 7}, {2, 1, 5}}, <|\"MaxTotalCost\" -> 10, \"MaxTotalCount\" -> 2|>]", "{2, 0}"),
    ("FindShortestTour[{\"This\", \"finds\", \"the\", \"shortest\", \"tour\", \"through\", \"a\", \"list\", \"of\", "
     "\"strings\"}]", "{38, {1, 8, 2, 10, 4, 6, 5, 9, 7, 3, 1}}"),
    ("First[FindShortestTour[{{2, 2}, {5, 2}, {5, 3}, {4, 5}, {4, 4}, {3, 2}}]]", "5 + Sqrt[2] + Sqrt[13]"),
    ("Sort[Last[FindShortestTour[{{2, 2}, {5, 2}, {5, 3}, {4, 5}, {4, 4}, {3, 2}}]]]", "{1, 1, 2, 3, 4, 5, 6}"),
    ("First[FindShortestTour[{\"cat\", \"dog\", \"do\", \"cap\", \"dock\"}]]", "11"),
    ("FindShortestTour[{{0, 0}, {3, 0}, {3, 4}}, 1, 3]", "{7, {1, 2, 3}}"),
    ("First[FindShortestTour[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]]", "4"),
    ("FindShortestTour[Graph[{1 -> 2}]]", Raises()),
]

BAYESIAN = [
    ("Head[BayesianMinimization[(# - 2)^2 + 1 &, Interval[{0, 4}]]]", "BayesianMinimizationObject"),
    T("Abs[BayesianMinimization[(# - 2)^2 + 1 &, Interval[{0, 4}]][\"MinimumConfiguration\"] - 2] < 0.1"),
    T("BayesianMinimization[(# - 2)^2 + 1 &, Interval[{0, 4}]][\"MinimumValue\"] < 1.01"),
    ("BayesianMinimization[(#^2 - 3*# + 2) &, {5.7, 1.1, 3.4, 6.8, 2.3, 1.2}][\"MinimumConfiguration\"]", "1.2"),
    ("BayesianMinimization[(# - 2)^2 &, {0, 5}][\"Method\"]", "\"MaxExpectedImprovement\""),
    ("BayesianMinimization[Norm, Range[2, 5]][\"MinimumConfiguration\"]", "2"),
    T("BayesianMinimization[(#^2 - 5*# + 4) &, RandomReal[{0, 5}] &][\"MinimumValue\"] < -2.2"),
    ("Head[BayesianMaximization[-(# - 1)^2 &, Interval[{-3, 3}]]]", "BayesianMaximizationObject"),
    T("Abs[BayesianMaximization[-(# - 1)^2 &, Interval[{-3, 3}]][\"MaximumConfiguration\"] - 1] < 0.1"),
    T("BayesianMaximization[-(# - 1)^2 &, Interval[{-3, 3}]][\"MaximumValue\"] > -0.01"),
    ("BayesianMaximization[(#^2 - 3*# + 2) &, {5.7, 1.1, 3.4}][\"MaximumConfiguration\"]", "5.7"),
    T("MemberQ[BayesianMinimization[#^2 &, {1, 2}][\"Properties\"], \"MinimumConfiguration\"]"),
    T("MemberQ[BayesianMaximization[#^2 &, {1, 2}][\"Properties\"], \"MaximumValue\"]"),
]

MATRIX_GAMES = [
    ("MatrixGame[{{1, -1}, {-1, 1}}][\"PayoffArray\"]", "{{{1, -1}, {-1, 1}}, {{-1, 1}, {1, -1}}}"),
    ("MatrixGame[{{{1, 1}, {0, 0}}, {{0, 0}, {1, 1}}}][\"PayoffArray\"]", "{{{1, 1}, {0, 0}}, {{0, 0}, {1, 1}}}"),
    ("Head[MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}]]", "MatrixGame"),
    ("MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}, GameActionLabels -> {{\"C\", \"D\"}, {\"C\", \"D\"}}]"
     "[\"GameActionLabels\"]", "{{\"C\", \"D\"}, {\"C\", \"D\"}}"),
    ("MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}, GamePlayerLabels -> {\"Ann\", \"Bob\"}][\"GamePlayerLabels\"]",
     "{\"Ann\", \"Bob\"}"),
    ("MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}, GameActionLabels -> {{\"C\", \"D\"}, {\"L\", \"R\"}}, "
     "GamePlayerLabels -> {\"Ann\", \"Bob\"}][\"GameActionLabels\"]", "{{\"C\", \"D\"}, {\"L\", \"R\"}}"),
    ("MatrixGame[{{1, 2}, {3, 4}}, GamePlayerLabels -> {\"P\", \"Q\"}][\"GamePlayerLabels\"]", "{\"P\", \"Q\"}"),
    ("FindMatrixGameStrategies[MatrixGame[{{{1, 1}, {0, 0}}, {{0, 0}, {1, 1}}}]]",
     "{{{0, 1}, {0, 1}}, {{1, 0}, {1, 0}}, {{1/2, 1/2}, {1/2, 1/2}}}"),
    ("FindMatrixGameStrategies[MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}]]", "{{{0, 1}, {0, 1}}}"),
    ("FindMatrixGameStrategies[MatrixGame[{{1, -1}, {-1, 1}}]]", "{{{1/2, 1/2}, {1/2, 1/2}}}"),
    ("FindMatrixGameStrategies[MatrixGame[{{0, -1, 1}, {1, 0, -1}, {-1, 1, 0}}]]",
     "{{{1/3, 1/3, 1/3}, {1/3, 1/3, 1/3}}}"),
    T("VerifyMatrixGameStrategy[MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}], {{0, 1}, {0, 1}}]"),
    ("VerifyMatrixGameStrategy[MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}], {{1, 0}, {1, 0}}]", False),
    T("VerifyMatrixGameStrategy[MatrixGame[{{1, -1}, {-1, 1}}], {{1/2, 1/2}, {1/2, 1/2}}]"),
    ("MatrixGamePayoff[MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}], {{1, 0}, {0, 1}}]", "{0, 5}"),
    ("MatrixGamePayoff[MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}], {{1/2, 1/2}, {1/2, 1/2}}]", "{9/4, 9/4}"),
    ("MatrixGamePayoff[MatrixGame[{{1, -1}, {-1, 1}}], {{1, 0}, {1, 0}}]", "{1, -1}"),
    ("Head[MatrixGamePlot[MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}]]]", "Graphics"),
    ("Count[MatrixGamePlot[MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}]], _Rectangle, Infinity]", "12"),
    ("MatrixGamePlot[MatrixGame[{{{3, 3}, {0, 5}}, {{5, 0}, {1, 1}}}], PlotLayout -> \"Row\"]", Raises()),
]

_ULTIMATUM = ("TreeGame[{1, {2, {0, 0} -> \"No\", {2, 0} -> \"Yes\"} -> \"A\", {2, {0, 0} -> \"No\", {1, 1} -> \"Yes\"} "
              "-> \"B\", {2, {0, 0} -> \"No\", {0, 2} -> \"Yes\"} -> \"C\"}]")
_RPS_TREE = ("TreeGame[{\"A\", {\"B\", {0, 0} -> \"Rock\", {-1, 1} -> \"Paper\", {1, -1} -> \"Scissors\"} -> \"Rock\", "
             "{\"B\", {1, -1} -> \"Rock\", {0, 0} -> \"Paper\", {-1, 1} -> \"Scissors\"} -> \"Paper\", {\"B\", {-1, 1} "
             "-> \"Rock\", {1, -1} -> \"Paper\", {0, 0} -> \"Scissors\"} -> \"Scissors\"}]")
_INFO_TREE = "TreeGame[{1, {0, 5}, {2 -> 1, {5, 0}, {-1, -2}}, {2 -> 1, {5, 0}, {-3, -5}}}]"

TREE_GAMES = [
    (f"FindTreeGameStrategies[{_ULTIMATUM}]",
     "{<|1 -> <|{} -> {0, 1, 0}|>, 2 -> <|{1} -> {0, 1}, {2} -> {0, 1}, {3} -> {0, 1}|>|>, <|1 -> <|{} -> {0, 1, 0}|>, "
     "2 -> <|{1} -> {1, 0}, {2} -> {0, 1}, {3} -> {0, 1}|>|>, <|1 -> <|{} -> {1, 0, 0}|>, 2 -> <|{1} -> {0, 1}, {2} -> "
     "{0, 1}, {3} -> {0, 1}|>|>, <|1 -> <|{} -> {1, 0, 0}|>, 2 -> <|{1} -> {1, 0}, {2} -> {0, 1}, {3} -> {0, 1}|>|>}"),
    (f"FindTreeGameStrategies[{_RPS_TREE}]",
     "{<|\"A\" -> <|{} -> {0, 0, 1}|>, \"B\" -> <|{1} -> {0, 1, 0}, {2} -> {0, 0, 1}, {3} -> {1, 0, 0}|>|>, <|\"A\" -> "
     "<|{} -> {0, 1, 0}|>, \"B\" -> <|{1} -> {0, 1, 0}, {2} -> {0, 0, 1}, {3} -> {1, 0, 0}|>|>, <|\"A\" -> <|{} -> "
     "{1, 0, 0}|>, \"B\" -> <|{1} -> {0, 1, 0}, {2} -> {0, 0, 1}, {3} -> {1, 0, 0}|>|>}"),
    (f"FindTreeGameStrategies[{_INFO_TREE}]",
     "{<|1 -> <|{} -> {0, 0, 1}|>, 2 -> <|{2} -> {1, 0}, {3} -> {1, 0}|>|>, <|1 -> <|{} -> {0, 1, 0}|>, 2 -> "
     "<|{2} -> {1, 0}, {3} -> {1, 0}|>|>}"),
    ("Length[FindTreeGameStrategies[TreeGame[{\"A\", {\"B\", {\"C\", {1, 2, 3}, {3, 2, 1}, {3, 1, 2}}, {\"C\", {0, 2, 4}, "
     "{4, 2, 0}, {4, 0, 2}}}, {\"C\", {\"B\", {-1, 2, 5}, {5, 2, -1}, {5, -1, 2}}, {\"B\", {-2, 2, 6}, {6, 2, -2}, "
     "{6, -2, 2}}}}]]]", "32"),
    T(f"VerifyTreeGameStrategy[{_INFO_TREE}, <|1 -> <|{{}} -> {{0, 0, 1}}|>, 2 -> <|{{2}} -> {{1, 0}}, {{3}} -> {{1, 0}}|>|>]"),
    (f"VerifyTreeGameStrategy[{_INFO_TREE}, <|1 -> <|{{}} -> {{1, 0, 0}}|>, 2 -> <|{{2}} -> {{1, 0}}, {{3}} -> {{1, 0}}|>|>]",
     False),
    ("VerifyTreeGameStrategy[TreeGame[{1, {2, {1, -1}, {2, -2}}, {2, {-2, 2}, {-1, 4}}}], <|1 -> <|{} -> {1, 0}|>, "
     "2 -> <|{1} -> {0, 1}, {2} -> {1, 0}|>|>]", False),
    T("VerifyTreeGameStrategy[TreeGame[{\"A\", {\"B\", {1, 1}, {2, 1}, {3, 1}}}], <|\"A\" -> <|{} -> {1}|>, \"B\" -> "
      "<|{1} -> {0.3, 0.4, 0.3}|>|>]"),
    ("VerifyTreeGameStrategy[TreeGame[{\"A\", {\"B\", {1, 1}, {2, 1}, {3, 1}}}], <|\"A\" -> <|{} -> {1}|>|>]", Raises()),
]

TURING_RULEPLOT = [
    ("TuringMachine[2506, {1, {0, 0, 0, 0}}, 3]",
     "{{{1, 1, 0}, {0, 0, 0, 0}}, {{2, 2, 1}, {1, 0, 0, 0}}, {{1, 1, 0}, {1, 1, 0, 0}}, {{2, 4, -1}, {0, 1, 0, 0}}}"),
    ("TuringMachine[2506, {1, {{}, 0}}, 4]",
     "{{{1, 3, 0}, {0, 0, 0, 0}}, {{2, 4, 1}, {0, 0, 1, 0}}, {{1, 3, 0}, {0, 0, 1, 1}}, {{2, 2, -1}, {0, 0, 0, 1}}, "
     "{{1, 1, -2}, {0, 1, 0, 1}}}"),
    ("TuringMachine[2506, {1, {{}, 0}}, 6]",
     "{{{1, 3, 0}, {0, 0, 0, 0}}, {{2, 4, 1}, {0, 0, 1, 0}}, {{1, 3, 0}, {0, 0, 1, 1}}, {{2, 2, -1}, {0, 0, 0, 1}}, "
     "{{1, 1, -2}, {0, 1, 0, 1}}, {{2, 2, -1}, {1, 1, 0, 1}}, {{1, 3, 0}, {1, 0, 0, 1}}}"),
    ("TuringMachine[2506, {1, {0, 0, 0, 0}}]", "{{2, 2, 1}, {1, 0, 0, 0}}"),
    ("TuringMachine[2506][{1, {0, 0, 0, 0}}]", "{{2, 2, 1}, {1, 0, 0, 0}}"),
    ("TuringMachine[2506, {1, {0, 0, 0, 0}}, 1]", "{{{1, 1, 0}, {0, 0, 0, 0}}, {{2, 2, 1}, {1, 0, 0, 0}}}"),
    ("TuringMachine[{{state_, color_?EvenQ} :> {state, color/2, -1}, {state_, color_?OddQ} :> {state, 3 color + 1, 1}}, "
     "{1, {1, 2, 3, 4, 5}}, 5]",
     "{{{1, 1, 0}, {1, 2, 3, 4, 5}}, {{1, 2, 1}, {4, 2, 3, 4, 5}}, {{1, 1, 0}, {4, 1, 3, 4, 5}}, {{1, 5, -1}, "
     "{2, 1, 3, 4, 5}}, {{1, 1, 0}, {2, 1, 3, 4, 16}}, {{1, 5, -1}, {1, 1, 3, 4, 16}}}"),
    ("TuringMachine[5000, {1, {0}}, 1]", Raises()),
    ("Head[RulePlot[CellularAutomaton[30]]]", "Graphics"),
    ("Count[RulePlot[CellularAutomaton[30]], _Rectangle, Infinity]", "32"),
    ("Count[RulePlot[CellularAutomaton[{30, 2, 1}]], GrayLevel[0], Infinity]", "16"),
    ("Count[RulePlot[TuringMachine[2506]], _Arrow, Infinity]", "4"),
    ("RulePlot[Plus[1, 2]]", Raises()),
]

OPTION_NAMES = [
    ("Normalized", "Normalized"),
    T("OrthogonalMatrixQ[{{2, 0}, {0, 3}}, Normalized -> False]"),
    ("OrthogonalMatrixQ[{{2, 0}, {0, 3}}]", False),
    ("NormFunction", "NormFunction"),
    ("Head[NormFunction -> (Norm[#, 1] &)]", "Rule"),
    ("{NormFunction}", "{NormFunction}"),
    ("FitRegularization", "FitRegularization"),
    ("Head[FitRegularization -> {\"Tikhonov\", 1}]", "Rule"),
    ("{FitRegularization}", "{FitRegularization}"),
    ("Regularization", "Regularization"),
    ("Head[Regularization -> \"Abel\"]", "Rule"),
    ("{Regularization}", "{Regularization}"),
    ("LossFunction", "LossFunction"),
    ("Head[LossFunction -> \"MeanSquare\"]", "Rule"),
    ("{LossFunction}", "{LossFunction}"),
    ("UnconstrainedParameters", "UnconstrainedParameters"),
    ("Head[UnconstrainedParameters -> {a}]", "Rule"),
    ("{UnconstrainedParameters}", "{UnconstrainedParameters}"),
]


# ── GROUPS-END (new case groups are inserted above this line) ──

_SCIPY_ONLY = ("WarpingDistance[{Pi",)
GROUPS = [v for k, v in list(globals().items()) if k.isupper() and k not in ("GROUPS", "N50") and isinstance(v, list)
          and v and isinstance(v[0], tuple)]
CASES = [c for g in GROUPS for c in g
         if HAVE_SCIPY or not (c[0].startswith(_SCIPY_ONLY) or (len(c) > 2 and c[2] == "scipy"))]
CASES = [c[:2] if len(c) > 2 and c[2] == "scipy" else c for c in CASES]


def _scoped(src):
    """Run sources that assign variables inside Module so that no case leaks definitions into the next."""
    import re
    names = set()
    for grp in re.findall(r"\{([A-Za-z0-9, ]+)\}\s*=(?!=)", src):
        names.update(n.strip() for n in grp.split(","))
    names.update(re.findall(r"(?<![A-Za-z0-9\]\}])([a-z][A-Za-z0-9]*)\s*=(?![=>])", src))
    if not names:
        return src
    return "Module[{" + ", ".join(sorted(names)) + "}, " + src + "]"


CASES = [(_scoped(c[0]),) + tuple(c[1:]) for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w08(case):
    check(case)
