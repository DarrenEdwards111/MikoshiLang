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

DISTANCES = [
    ("EuclideanDistance[{a, b, c}, {x, y, z}]", "Sqrt[Abs[a - x]^2 + Abs[b - y]^2 + Abs[c - z]^2]"),
    ("EuclideanDistance[{1, 2, 3}, {2, 4, 6}]", "Sqrt[14]"),
    (f"EuclideanDistance[{N50}]", "21.494185260204677039244010575481712995963818467845`50"),
    ("EuclideanDistance[{a, b}, {a, c}]", "Abs[b - c]"),
    ("EuclideanDistance[3, 7.5]", "4.5"),
    ("EuclideanDistance[{1, 2}, {1, 2, 3}]", Raises()),
    ("SquaredEuclideanDistance[{a, b, c}, {x, y, z}]", "Abs[a - x]^2 + Abs[b - y]^2 + Abs[c - z]^2"),
    ("SquaredEuclideanDistance[{1, 2, 3}, {2, 4, 6}]", "14"),
    (f"SquaredEuclideanDistance[{N50}]", "462.0000000000000000000000000000000000000000000000`50"),
    ("ManhattanDistance[{a, b, c}, {x, y, z}]", "Abs[a - x] + Abs[b - y] + Abs[c - z]"),
    ("ManhattanDistance[{1, 2, 3}, {2, 4, 6}]", "6"),
    (f"ManhattanDistance[{N50}]", "38.00000000000000000000000000000000000000000000000`50"),
    ("ChessboardDistance[{a, b, c}, {x, y, z}]", "Max[Abs[a - x], Abs[b - y], Abs[c - z]]"),
    ("ChessboardDistance[{1, 2, 3}, {2, 4, 6}]", "3"),
    (f"ChessboardDistance[{N50}]", "18.000000000000000000000000000000000000000000000000`50"),
    ("BrayCurtisDistance[{1, 2, 3}, {2, 4, 6}]", "1/3"),
    ("BrayCurtisDistance[{a, b}, {x, y}]", "(Abs[a - x] + Abs[b - y])/(Abs[a + x] + Abs[b + y])"),
    (f"BrayCurtisDistance[{N50}]", "0.5428571428571428571428571428571428571428571428571`50"),
    ("CanberraDistance[{1, 2, 3}, {2, 4, 6}]", "1"),
    ("CanberraDistance[{a, b}, {a, c}]", "Abs[b - c]/(Abs[b] + Abs[c])"),
    (f"CanberraDistance[{N50}]", "2.5015151515151515151515151515151515151515151515152`50"),
    ("CorrelationDistance[{1, 2, 3}, {3, 5, 10}]", "1 - 7/(2 Sqrt[13])"),
    (f"CorrelationDistance[{N50}]", "1.2106635188398813533035274347761676650148587810028`50"),
    ("CorrelationDistance[{1, 1, 1}, {1, 2, 3}]", Raises()),
    ("CosineDistance[{1, 2, 3}, {3, 5, 7}]", "1 - 17 Sqrt[2/581]"),
    (f"CosineDistance[{N50}]", "0.4062940529572788872406931125582100673373718289257`50"),
    ("CosineDistance[{a, b}, {x, y}]",
     "1 - (a Conjugate[x] + b Conjugate[y])/(Sqrt[Abs[a]^2 + Abs[b]^2] Sqrt[Abs[x]^2 + Abs[y]^2])"),
    ("CosineDistance[{0, 0}, {1, 2}]", Raises()),
    ("NormalizedSquaredEuclideanDistance[{1, 2, 3}, {3, 5, 10}]", "1/4"),
    (f"NormalizedSquaredEuclideanDistance[{N50}]", "0.583333333333333333333333333333333333333333333333`50"),
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
    ("OrthogonalMatrix[RotationMatrix[t]] // Simplify", "RotationMatrix[t]"),
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
