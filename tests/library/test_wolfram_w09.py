"""Differential tests for Wolfram batch W09 (number theory, algebraic number fields, finite fields, domains, group
theory and fractals).

Reference values come from the Wolfram Language reference pages (reference.wolfram.com/language/ref/<name>.html;
the documented In/Out pairs), from mathematically verifiable identities (checked here symbolically or numerically)
and from standard tables (Conway polynomials, orders of the sporadic groups, OEIS sequences).  Numbers printed by
the documentation to six digits are compared with a relative tolerance of 1e-5.
"""
import pytest

from _harness import Pred, Raises, check, ids

# ─────────────────────────── finite fields ───────────────────────────
# Reference: reference.wolfram.com/language/ref/FiniteField.html and the pages of the other finite field functions.
# FiniteField[p, d] uses the Conway polynomial C(p, d): the documented field irreducible of FiniteField[29, 4] is
# 2 + 15 #1 + 2 #1^2 + #1^4 &; the other Conway polynomials agree with Luebeck's tables.

def FFE(p, d, cs):
    return f"FiniteFieldElement[FiniteField[{p}, {d}], {{{', '.join(map(str, cs))}}}]"


FINITE_FIELDS = [
    ("Information[FiniteField[29, 4], \"FieldIrreducible\"][x]", "2 + 15 x + 2 x^2 + x^4"),
    ("Information[FiniteField[5, 2], \"FieldIrreducible\"][x]", "2 + 4 x + x^2"),
    ("Information[FiniteField[7, 3, \"Exponential\"], \"FieldIrreducible\"][x]", "4 + 6 x^2 + x^3"),
    ("Information[FiniteField[2, 8], \"FieldIrreducible\"][x]", "1 + x^2 + x^3 + x^4 + x^8"),
    ("Information[FiniteField[1009, 3], {\"Characteristic\", \"ExtensionDegree\", \"FieldSize\"}]",
     "{1009, 3, 1027243729}"),
    ("FiniteField[1009, 3][\"ElementRepresentation\"]", "\"Polynomial\""),
    ("FiniteField[7, 3, \"Exponential\"][\"ElementRepresentation\"]", "\"Exponential\""),
    ("FiniteField[11] === FiniteField[11, 1]", "True"),
    ("FiniteField[49]", "FiniteField[7, 2]"),
    ("FiniteField[49, \"Exponential\"]", "FiniteField[7, 2, \"Exponential\"]"),
    ("FiniteField[7, 2, \"Polynomial\"]", "FiniteField[7, 2]"),
    ("FiniteField[6]", Raises()),
    ("FiniteField[5, #^2 + 1 &]", Raises()),                     # x^2 + 1 = (x + 2)(x + 3) mod 5
    ("Information[FiniteField[5, 2 #^2 + 4 # + 6 &], \"FieldIrreducible\"][x]", "1 + 4 x + 2 x^2"),
    ("FiniteField[7, #^3 + # + 1 &, \"Exponential\"]", Raises()),  # irreducible but not primitive
    ("FiniteField[5, 1][3] + FiniteField[5, 1][4]", FFE(5, 1, [2])),
    ("FiniteField[5, 1][2] FiniteField[5, 1][4]", FFE(5, 1, [3])),
    ("FiniteField[19, 1][12] FiniteField[19, 1][7] + FiniteField[19, 1][3]", FFE(19, 1, [(12 * 7 + 3) % 19])),
    ("FiniteField[17, 3][123]", FFE(17, 3, [4, 7])),
    ("FiniteField[17, 3][{1, 2, 3}][\"Index\"]", "1 + 2*17 + 3*17^2"),
    ("FiniteField[17, 3][123] - FiniteField[17, 3][123]", FFE(17, 3, [])),
    ("FiniteField[17, 3][345] / FiniteField[17, 3][345]", FFE(17, 3, [1])),
    ("With[{F = FiniteField[17, 3]}, (F[{1, 2, 3}]^2 + 2 F[123]) / (5 F[{1, 2, 3}] F[123] - 7) * "
     "(5 F[{1, 2, 3}] F[123] - 7) == F[{1, 2, 3}]^2 + 2 F[123]]", "True"),
    ("With[{F = FiniteField[3, 6]}, F[123]^321 == Fold[#1 F[123] &, F[1], Range[321]]]", "True"),
    ("With[{F = FiniteField[3, 6]}, (F[123]^(1/3))^3 == F[123]]", "True"),
    ("With[{F = FiniteField[3, 6]}, Sqrt[F[123]]^2 == F[123]]", "True"),
    ("With[{F = FiniteField[3, 6]}, MatchQ[Sqrt[F[432]], Sqrt[_FiniteFieldElement] | Power[_FiniteFieldElement, 1/2]]]", "True"),          # 432 is not a square: unevaluated
    ("With[{F = FiniteField[19, 3]}, {F[123] + 5, 7 F[123]}]", "{" + FFE(19, 3, [14, 6]) + ", " + FFE(19, 3, [6, 4]) + "}"),
    ("FiniteField[19, 3][123] + 7/8", FFE(19, 3, [(9 + 7 * pow(8, -1, 19)) % 19, 6])),
    ("FiniteField[19, 3][123] + 2/19", Raises()),
    ("FiniteField[19, 3][0] == 0", "True"),
    ("FiniteField[19, 3][1] == 1", "True"),
    ("FiniteField[19, 3][2] != 3", "True"),
    ("FiniteField[389, 2][123] - FiniteField[389, 4][123]", Raises()),
    ("With[{F = FiniteField[11, 3]}, Module[{a, b}, {a, b} = {F[123], F[456]}; (a + b)^11 == a^11 + b^11]]", "True"),
    ("With[{F = FiniteField[11, 3]}, F[{123, 234, 345}] == 123 + 234 F[{0, 1}] + 345 F[{0, 1}]^2]", "True"),
    ("With[{F = FiniteField[109, 5]}, Information[F, \"FieldIrreducible\"][F[{0, 1}]]]", FFE(109, 5, [])),
    ("With[{F = FiniteField[7, 3, \"Exponential\"]}, And @@ Table[F[i] == F[2]^(i - 1), {i, 342}]]", "True"),
    ("With[{F = FiniteField[7, 3, \"Exponential\"]}, F[21] - F[21] == 0 && F[32] / F[32] == 1]", "True"),
    ("FiniteField[3, 2][k]", "FiniteField[3, 2][k]"),
    # AES: the multiplicative inverse of 0x53 in GF(2^8) with the Rijndael polynomial is 0xCA
    ("(FiniteField[2, #^8 + #^4 + #^3 + # + 1 &][83]^(-1))[\"Index\"]", "202"),
    ("FiniteField[2, #^8 + #^4 + #^3 + # + 1 &][83][\"Coefficients\"]", "{1, 1, 0, 0, 1, 0, 1, 0}"),
    ("FiniteField[5, 2][7][\"Field\"]", "FiniteField[5, 2]"),
    ("FiniteField[5, 2][7][\"FieldSize\"]", "25"),
    ("Information[FiniteField[5, 2][7], \"Characteristic\"]", "5"),
    ("FiniteFieldElement[FiniteField[5, 2], 7]", FFE(5, 2, [2, 1])),
    ("FiniteFieldElement[FiniteField[5, 2], {7, 6, 1}]", "FiniteFieldElement[FiniteField[5, 2], {0, 2}]"),
    ("FiniteFieldElement[FiniteField[5, 2], 25]", Raises()),
    ("FiniteFieldElement[x, 3]", Raises()),
    # Diffie-Hellman with the documented 2048-bit prime
    ("With[{p = 32317006071311007300714876688669951960444102669715484032130345427524655138867890893197201411522913463"
     "688717960921898019494119559150490921095088152386448283120630877367300996091750197750389652106796057638384067"
     "568276792218642619756161838094338476170470581645852036305042887575891541065808607552399123930385521914333389"
     "668342420684974786564569494856176035326322058077805659331026192708460314150258592864177116725943603718461857"
     "357598351152301645904403697613233287231227125684710820209725157101726931323469678542580656697935045997268352"
     "998638215525166389437335543602135433229604645318478604952148193555853611059596248443}, "
     "With[{g = FiniteField[p][2]}, ((12345678901 (g^1234567)^987654321) / (g^987654321)^1234567)[\"Index\"]]]",
     "12345678901"),
    # Expand over a finite field: the product of x - a over GF(9) is x^9 - x
    ("FromFiniteField[Expand[Times @@ Table[x - FiniteField[3, 2][i], {i, 0, 8}]], FiniteField[3, 2]]", "x^9 + 2 x"),
    ("FromFiniteField[Expand[(FiniteField[5, 2][7] x + 1)^2], FiniteField[5, 2], t]", "2 x^2 + (4 + 2 t) x + 1"),
    ("FiniteField[5, 2][7] x + FiniteField[5, 2][3] x", FFE(5, 2, [0, 1]) + " x"),
    ("Factor[FiniteField[5, 2][3] x^2 + 1]", Raises()),
    ("Det[{{FiniteField[5, 2][3], 1}, {1, 1}}]", Raises()),
]

FROBENIUS = [
    ("With[{a = FiniteField[7, 5][123]}, FrobeniusAutomorphism[a] == a^7]", "True"),
    ("With[{a = FiniteField[17, 7][321]}, FrobeniusAutomorphism[a, 3] === ((a^17)^17)^17]", "True"),
    ("With[{a = FiniteField[23, 4][123]}, FrobeniusAutomorphism[a, 4] === a]", "True"),
    ("With[{a = FiniteField[23, 4][123]}, FrobeniusAutomorphism[a, -1] === FrobeniusAutomorphism[a, 3]]", "True"),
    ("With[{F = FiniteField[109, 5]}, Union[Information[F, \"FieldIrreducible\"] /@ "
     "Table[FrobeniusAutomorphism[F[{0, 1}], k], {k, 4}]]]", "{" + FFE(109, 5, []) + "}"),
    ("FrobeniusAutomorphism[3]", Raises()),
    # norms and traces: absolute values are integers; FiniteFieldElementNorm[a] = a^((q - 1)/(p - 1))
    ("FiniteFieldElementNorm[FiniteField[17, 6][123]]", "3"),
    ("With[{a = FiniteField[17, 6][123]}, a^((17^6 - 1)/16) == FiniteFieldElementNorm[a]]", "True"),
    ("FiniteFieldElementNorm[FiniteField[17, 6][123], 2]", FFE(17, 2, [2, 2])),
    ("With[{a = FiniteField[17, 6][123]}, FiniteFieldElementNorm[FiniteFieldElementNorm[a, 2]] == "
     "FiniteFieldElementNorm[a]]", "True"),
    ("FiniteFieldElementNorm[FiniteField[17, 6][123], 4]", Raises()),
    ("FiniteFieldElementTrace[FiniteField[17, 6][123]]", "7"),
    ("With[{a = FiniteField[17, 6][123]}, Total[Table[FrobeniusAutomorphism[a, k], {k, 0, 5}]] == "
     "FiniteFieldElementTrace[a]]", "True"),
    ("With[{a = FiniteField[17, 6][123], b = FiniteField[17, 6][4567]}, FiniteFieldElementTrace[2 a + 3 b] == "
     "Mod[2 FiniteFieldElementTrace[a] + 3 FiniteFieldElementTrace[b], 17]]", "True"),
    ("FiniteFieldElementTrace[FiniteField[17, 6][123], 3]", FFE(17, 3, [0, 8, 5])),
    ("With[{a = FiniteField[17, 6][123]}, FiniteFieldElementTrace[FiniteFieldElementTrace[a, 3]] == "
     "FiniteFieldElementTrace[a]]", "True"),
    ("FiniteFieldElementTrace[x]", Raises()),
    ("FiniteFieldElementPrimitiveQ[FiniteField[17, 3][123]]", "True"),
    ("FiniteFieldElementPrimitiveQ[FiniteField[7, 3, \"Exponential\"][2]]", "True"),
    ("FiniteFieldElementPrimitiveQ[FiniteField[7, 3, \"Exponential\"][3]]", "False"),   # theta^2: order 171
    ("Count[Table[FiniteFieldElementPrimitiveQ[FiniteField[3, 2][i]], {i, 0, 8}], True]", "EulerPhi[8]"),
    ("FiniteFieldElementPrimitiveQ[17]", "False"),
    ("FiniteFieldElementPrimitiveQ[x]", "False"),
]

INDICES = [
    ("FiniteFieldIndex[FiniteField[11, 3][{1, 2, 3}]]", "1 + 2*11 + 3*121"),
    ("FiniteFieldIndex[FiniteField[11, 3, \"Exponential\"][{1, 2, 3}]]", "80"),
    ("With[{F = FiniteField[11, 3, \"Exponential\"]}, F[2]^79 == F[{1, 2, 3}]]", "True"),
    ("FiniteFieldIndex[FromFiniteFieldIndex[{{11, 12}, {21, 22}}, FiniteField[5, 2]]]", "{{11, 12}, {21, 22}}"),
    ("With[{u = FiniteField[11, 3][1234]}, FiniteFieldIndex[u] === u[\"Index\"]]", "True"),
    ("FiniteFieldIndex[{FiniteField[7, 5][0], FiniteField[7, 5][1]}]", "{0, 1}"),
    ("FiniteFieldIndex[3]", Raises()),
    ("With[{ff = FiniteField[11, 3]}, FromFiniteFieldIndex[1234, ff] === ff[1234]]", "True"),
    ("FromFiniteFieldIndex[{0, 1, 5}, FiniteField[3, 2]]",
     "{" + FFE(3, 2, []) + ", " + FFE(3, 2, [1]) + ", " + FFE(3, 2, [2, 1]) + "}"),
    ("FromFiniteFieldIndex[123, FiniteField[3, 5, \"Exponential\"]] == "
     "FiniteField[3, 5, \"Exponential\"][2]^122", "True"),
    ("FromFiniteFieldIndex[7, x]", Raises()),
    ("ToFiniteField[21, FiniteField[11, 2]]", FFE(11, 2, [10])),
    ("ToFiniteField[123 + 456 x + 789 x^2 + x^3, FiniteField[7, 3]] /. x -> FiniteField[7, 3][{0, 1}]",
     "FiniteField[7, 3][{4, 1, 5}] + FiniteField[7, 3][{0, 1}]^3"),
    ("ToFiniteField[2 + 3 t + 4 t^2, FiniteField[19, 3], t]", FFE(19, 3, [2, 3, 4])),
    ("ToFiniteField[{123, 456/789}, FiniteField[7, 2]]", "{" + FFE(7, 2, [4]) + ", " + FFE(7, 2, [(456 * pow(789, -1, 7)) % 7]) + "}"),
    ("ToFiniteField[1/7, FiniteField[7, 2]]", Raises()),
    ("ToFiniteField[3, x]", Raises()),
    ("FromFiniteField[ToFiniteField[123 + 456 x + 789 x^2 + x^3, FiniteField[7, 3]], FiniteField[7, 3]]",
     "4 + x + 5 x^2 + x^3"),
    ("FromFiniteField[FiniteField[5, 2][3], FiniteField[5, 2]]", "3"),
    ("FromFiniteField[FiniteField[19, 4][123], FiniteField[19, 4]]", FFE(19, 4, [9, 6])),   # not in the prime field
    ("FromFiniteField[FiniteField[7, 3][123], FiniteField[7, 3], t]", "4 + 3 t + 2 t^2"),
    ("FromFiniteField[ToFiniteField[2 + 3 t + 4 t^2, FiniteField[19, 3], t], FiniteField[19, 3], t]", "2 + 3 t + 4 t^2"),
    ("FromFiniteField[1, x]", Raises()),
]

EMBEDDINGS = [
    ("FiniteFieldEmbedding[FiniteField[7, 3], FiniteField[7, 6]]",
     "FiniteFieldEmbedding[FiniteField[7, 3][{0, 1}] -> FiniteField[7, 6][{0, 1}]^((7^6 - 1)/(7^3 - 1))]"),
    # the image of the generator is the Conway-compatible root theta^((q2 - 1)/(q1 - 1)) of C(7, 3)
    ("Information[FiniteField[7, 3], \"FieldIrreducible\"][FiniteField[7, 6][{0, 1}]^((7^6 - 1)/(7^3 - 1))]",
     FFE(7, 6, [])),
    ("With[{K = FiniteField[7, 3]}, With[{e = FiniteFieldEmbedding[K, FiniteField[7, 6]]}, "
     "{e[K[45] + K[178]] == e[K[45]] + e[K[178]], e[K[45] K[178]] == e[K[45]] e[K[178]]}]]", "{True, True}"),
    ("With[{K = FiniteField[7, 3]}, With[{e = FiniteFieldEmbedding[K, FiniteField[7, 6]]}, "
     "e[\"Projection\"][e[K[123]]] == K[123]]]", "True"),
    ("With[{F = FiniteField[7, 6]}, With[{e = FiniteFieldEmbedding[FiniteField[7, 3], F]}, "
     "e[\"Projection\"][2 F[123] + 3 F[456]] == 2 e[\"Projection\"][F[123]] + 3 e[\"Projection\"][F[456]]]]", "True"),
    ("With[{K = FiniteField[17, 3]}, With[{e = FiniteFieldEmbedding[K, FiniteField[17, 6]]}, "
     "Information[K, \"FieldIrreducible\"][e[K[{0, 1}]]]]]", FFE(17, 6, [])),
    ("FiniteFieldEmbedding[FiniteField[17, 2], FiniteField[17, 3]]", Raises()),
    ("FiniteFieldEmbedding[FiniteField[5, 2], FiniteField[7, 2]]", Raises()),
    ("FiniteFieldEmbedding[FiniteField[2, 4][2] -> FiniteField[2, 8][7]]", Raises()),
    ("With[{F = FiniteField[2, 8]}, With[{e = FiniteFieldEmbedding[FiniteField[2, 4][2] -> F[2]^17]}, "
     "e[FiniteField[2, 4][5]] == 1 + F[2]^34]]", "True"),
    ("With[{e = FiniteFieldEmbedding[FiniteField[3, 2], FiniteField[3, 4]]}, e[2]]", FFE(3, 4, [2])),
]

# ─────────────────────────── paths and curves ───────────────────────────
# Reference: the ref pages of AnglePath, HilbertCurve, PeanoCurve, KochCurve, SierpinskiCurve (documented outputs,
# and the documented arc lengths 4^n - 1, (4/3)^n, 128 (2 + 3 Sqrt[2]), 128 (10 + 11 Sqrt[2]), ...).

def pts_ok(n_points, first, last):
    def p(g):
        pts = g.args[0].args
        return g.head == "Line" and len(pts) == n_points and tuple(pts[0].args) == first and tuple(pts[-1].args) == last
    return Pred(p, f"Line with {n_points} points from {first} to {last}")


CURVES = [
    ("AnglePath[{90 Degree, 90 Degree, 90 Degree}]", "{{0, 0}, {0, 1}, {-1, 1}, {-1, 0}}"),
    ("AnglePath[{90 Degree, Pi/3, -Pi/4}]",
     "{{0, 0}, {0, 1}, {-(Sqrt[3]/2), 3/2}, {-(Sqrt[3]/2) - (-1 + Sqrt[3])/(2 Sqrt[2]), 3/2 + (1 + Sqrt[3])/(2 Sqrt[2])}}"),
    ("AnglePath[{{0.7, 90 Degree}, {2.3, Pi/3}, {3.5, -Pi/4}}]",
     [[0, 0], [0, 0.7], [-1.99186, 1.85], [-2.89773, 5.23074]], 1e-5),
    ("AnglePath[{100, 200}, {90 Degree, Pi/3, -Pi/4}]",
     "{{100, 200}, {100, 201}, {100 - Sqrt[3]/2, 403/2}, {100 - Sqrt[3]/2 - (-1 + Sqrt[3])/(2 Sqrt[2]), 403/2 + (1 + Sqrt[3])/(2 Sqrt[2])}}"),
    ("AnglePath[Pi/6, {90 Degree, Pi/3, -Pi/4}]",
     "{{0, 0}, {-1/2, Sqrt[3]/2}, {-3/2, Sqrt[3]/2}, {-3/2 - 1/Sqrt[2], 1/Sqrt[2] + Sqrt[3]/2}}"),
    ("AnglePath[{{100, 200}, Pi/6}, {90 Degree, Pi/3, -Pi/4}]",
     "{{100, 200}, {199/2, 200 + Sqrt[3]/2}, {197/2, 200 + Sqrt[3]/2}, {197/2 - 1/Sqrt[2], 200 + 1/Sqrt[2] + Sqrt[3]/2}}"),
    ("AnglePath[{{100, 200}, Pi/6}, {{0.7, 90 Degree}, {2.3, Pi/3}, {3.5, -Pi/4}}]",
     [[100, 200], [99.65, 200.606], [97.35, 200.606], [94.8751, 203.081]], 1e-5),
    ("AnglePath[{{100, 200}, {Sqrt[3], 1}}, {{0.7, 90 Degree}, {2.3, Pi/3}, {3.5, -Pi/4}}]",
     [[100, 200], [99.65, 200.606], [97.35, 200.606], [94.8751, 203.081]], 1e-5),
    ("AnglePath[{x, y}, {{r1, t1}, {r2, t2}}]",
     "{{x, y}, {x + r1 Cos[t1], y + r1 Sin[t1]}, {x + r1 Cos[t1] + r2 Cos[t1 + t2], y + r1 Sin[t1] + r2 Sin[t1 + t2]}}"),
    ("AnglePath[{1, 1}, {Pi/3, Pi/2, -2 Pi/3}, \"RelativeMatrix\"]",
     "{{{1, 0}, {0, 1}}, {{1/2, -Sqrt[3]/2}, {Sqrt[3]/2, 1/2}}, {{0, -1}, {1, 0}}, {{-1/2, Sqrt[3]/2}, {-Sqrt[3]/2, -1/2}}}"),
    ("AnglePath[{1, 1}, {Pi/3, Pi/2, -2 Pi/3}, \"FrameAngle\"]", "{0, Pi/3, 5 Pi/6, Pi/6}"),
    ("AnglePath[{1, 1}, {Pi/3, Pi/2, -2 Pi/3}, \"RelativeAngle\"]", "{0, Pi/3, Pi/2, -2 Pi/3}"),
    ("AnglePath[{Pi/2}, \"FrameMatrix\"]", "{{{1, 0}, {0, 1}}, {{0, -1}, {1, 0}}}"),
    ("AnglePath[{Pi/2}, {\"Position\", \"FrameAngle\"}]", "{{{0, 0}, {0, 1}}, {0, Pi/2}}"),
    ("AnglePath[{Pi/2}, \"Rotation\"]", Raises()),
    ("AnglePath[x]", Raises()),
    ("HilbertCurve[1]", "Line[{{0, 0}, {0, 1}, {1, 1}, {1, 0}}]"),
    ("HilbertCurve[2]", "Line[{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 2}, {0, 3}, {1, 3}, {1, 2}, {2, 2}, {2, 3}, "
     "{3, 3}, {3, 2}, {3, 1}, {2, 1}, {2, 0}, {3, 0}}]"),
    ("HilbertCurve[3]", pts_ok(64, (0, 0), (7, 0))),
    ("Table[ArcLength[HilbertCurve[n]], {n, 5}]", "{3, 15, 63, 255, 1023}"),
    ("Sort[First[HilbertCurve[3]]] == Tuples[Range[0, 7], 2]", "True"),
    ("HilbertCurve[2, 2] === HilbertCurve[2]", "True"),
    ("HilbertCurve[1, DataRange -> {{0, 1}, {0, 1}}]", "Line[{{0., 0.}, {0., 1.}, {1., 1.}, {1., 0.}}]"),
    ("HilbertCurve[2, 3]", Raises()),
    ("HilbertCurve[-1]", Raises()),
    ("PeanoCurve[1]", "Line[{{0, 0}, {1, 0}, {2, 0}, {2, 1}, {1, 1}, {0, 1}, {0, 2}, {1, 2}, {2, 2}}]"),
    ("PeanoCurve[1, DataRange -> {{0, 1}, {0, 1}}]",
     "Line[{{0., 0.}, {0.5, 0.}, {1., 0.}, {1., 0.5}, {0.5, 0.5}, {0., 0.5}, {0., 1.}, {0.5, 1.}, {1., 1.}}]"),
    ("PeanoCurve[3]", pts_ok(729, (0, 0), (26, 26))),
    ("Sort[First[PeanoCurve[2]]] == Tuples[Range[0, 8], 2]", "True"),
    ("ArcLength[PeanoCurve[2]]", "80"),          # 81 points joined by unit steps
    ("PeanoCurve[x]", Raises()),
    ("First[KochCurve[1]]", [[0, 0], [1 / 3, 0], [0.5, 3 ** 0.5 / 6], [2 / 3, 0], [1, 0]], 1e-12),
    ("KochCurve[1]", Pred(lambda g: g.head == "Line", "a Line")),
    ("KochCurve[1, DataRange -> {{-1, 1}, {-1, 1}}]", Pred(lambda g: g.head == "Line" and
                                                         [round(c, 6) for c in g.args[0].args[1].args] == [-0.333333, -1.0],
                                                         "second point {-0.333333, -1.}")),
    ("Table[ArcLength[KochCurve[n]], {n, 5}]", [4 / 3, 16 / 9, 64 / 27, 256 / 81, 1024 / 243], 1e-9),
    ("Length[First[KochCurve[3]]]", "4^3 + 1"),
    ("KochCurve[2, {0, 90 Degree, -90 Degree}]", Raises()),
    ("SierpinskiCurve[1]", "Line[{{0, 0}, {32, -32}, {96, -32}, {128, 0}, {160, -32}, {128, -64}, {128, -128}, "
     "{160, -160}, {128, -192}, {96, -160}, {32, -160}, {0, -192}, {-32, -160}, {0, -128}, {0, -64}, {-32, -32}, {0, 0}}]"),
    ("Table[ArcLength[SierpinskiCurve[n]], {n, 5}]",
     "{128 (2 + 3 Sqrt[2]), 128 (10 + 11 Sqrt[2]), 128 (42 + 43 Sqrt[2]), 128 (170 + 171 Sqrt[2]), 128 (682 + 683 Sqrt[2])}"),
    ("First[SierpinskiCurve[1, DataRange -> {{0, 1}, {0, 1}}]]", [[1 / 6, 1], [1 / 3, 5 / 6], [2 / 3, 5 / 6], [5 / 6, 1], [1, 5 / 6],
                                                        [5 / 6, 2 / 3], [5 / 6, 1 / 3], [1, 1 / 6], [5 / 6, 0], [2 / 3, 1 / 6],
                                                        [1 / 3, 1 / 6], [1 / 6, 0], [0, 1 / 6], [1 / 6, 1 / 3], [1 / 6, 2 / 3],
                                                        [0, 5 / 6], [1 / 6, 1]], 1e-12),
    ("First[First[SierpinskiCurve[3]]] === Last[First[SierpinskiCurve[3]]]", "True"),
    ("SierpinskiCurve[0]", Raises()),
]

# ─────────────────────────── Mandelbrot and Julia sets ───────────────────────────
# Reference: the documented outputs on the ref pages of MandelbrotSetIterationCount, MandelbrotSetMemberQ,
# MandelbrotSetDistance, MandelbrotSetBoettcher and JuliaSetBoettcher.

MANDELBROT = [
    ("MandelbrotSetIterationCount[1/2 + I/2]", "4"),
    ("MandelbrotSetIterationCount[0]", "1001"),
    ("MandelbrotSetIterationCount[0.250001]", "1000"),
    ("MandelbrotSetIterationCount[0.250001, MaxIterations -> 4000]", "3139"),
    ("MandelbrotSetIterationCount[{1/4, 0.26, 1 + I, 1/Pi, N[1/Pi, 20]}]", "{1001, 29, 1, 9, 9}"),
    ("{MandelbrotSetIterationCount[-1.85 + I*10^-290], MandelbrotSetIterationCount[-1.85 + I*10^-290, "
     "MaxIterations -> 2000]}", "{1000, 1326}"),
    ("MandelbrotSetIterationCount[x]", Raises()),
    ("MandelbrotSetMemberQ[1 + I]", "False"),
    ("MandelbrotSetMemberQ[0]", "True"),
    ("{MandelbrotSetMemberQ[0.2501, MaxIterations -> 250], MandelbrotSetMemberQ[0.2501, MaxIterations -> 500]}",
     "{True, False}"),
    ("MandelbrotSetMemberQ[{0, 1, I/2, I + 1}]", "{True, False, True, False}"),
    ("MandelbrotSetMemberQ[{0, -2, -1.979, -2.9, 1/4, 0.26, 1 + I, -1/Pi, N[1/Pi, 200]}]",
     "{True, True, True, False, True, False, False, True, False}"),
    ("{MandelbrotSetMemberQ[0.250005, MaxIterations -> 1000], MandelbrotSetMemberQ[0.250005, MaxIterations -> 2000]}",
     "{True, False}"),
    ("MandelbrotSetMemberQ[x]", "False"),
    ("MandelbrotSetDistance[1 + I]", 1.98912, 1e-5),
    ("MandelbrotSetDistance[.73 - .9 I]", 1.1302, 1e-4),
    ("MandelbrotSetDistance[0]", "0"),
    ("{MandelbrotSetDistance[25745/100000, MaxIterations -> 30], MandelbrotSetDistance[25745/100000, "
     "MaxIterations -> 50000]}", [0.0, 0.00118428], 1e-7),
    ("MandelbrotSetDistance[9/7]", 2.09917, 1e-5),
    ("{MandelbrotSetDistance[100], MandelbrotSetDistance[101]}", [926.616, 937.847], 1e-5),
    ("MandelbrotSetDistance[0, \"Interior\"]", Raises()),
    ("MandelbrotSetBoettcher[1. + I]", 1.4717 + 0.986092j, 1e-5),
    ("MandelbrotSetBoettcher[.1]", "MandelbrotSetBoettcher[0.1]"),
    ("MandelbrotSetBoettcher[{1., I + 1.}]", [1.50284, 1.4717 + 0.986092j], 1e-5),
    ("MandelbrotSetBoettcher[-2.1979 + I]", -1.62506 + 1.11004j, 1e-5),
    ("MandelbrotSetBoettcher[1/3]", "MandelbrotSetBoettcher[1/3]"),
    ("Abs[N[MandelbrotSetBoettcher[1/3], 50] - 1.0034335637784507968641775431766194010317229326004`50] < 10^-45", "True"),
    ("N[MandelbrotSetBoettcher[1 + I]]", 1.4717 + 0.986092j, 1e-5),
    ("With[{c = 3. + 2. I}, MandelbrotSetBoettcher[c]^2 - JuliaSetBoettcher[c, c^2 + c]]", 0, 1e-9),
    ("JuliaSetBoettcher[1 + I, 2. - I]", 2.11555 - 0.675454j, 1e-5),
    ("JuliaSetBoettcher[1 + I, {2 I, I + 1, I + 2}]",
     "{JuliaSetBoettcher[1 + I, 2 I], JuliaSetBoettcher[1 + I, 1 + I], JuliaSetBoettcher[1 + I, 2 + I]}"),
    ("N[JuliaSetBoettcher[1 + I, {2 I, I + 1, I + 2}]]", [0.223497 + 1.7766j, 1.4717 + 0.986092j, 2.29902 + 1.08468j], 1e-5),
    ("N[JuliaSetBoettcher[1 + I, {-2, 3/4, 3 I, 1/Pi}]]",
     [-2.27436 - 0.236508j, 1.4361 + 0.369542j, 0.163683 + 2.84273j, 1.30384 + 0.385546j], 1e-5),
    ("JuliaSetBoettcher[1 + I, -2.9]", -3.08176 - 0.169848j, 1e-5),
    ("Abs[JuliaSetBoettcher[1 + I, N[1/Pi, 30]] - (1.3038392306688137909878382489`30 + 0.3855457428713203744363486474`30 I)] < 10^-26", "True"),
    ("Head[N[JuliaSetBoettcher[I/2, -0.1]]]", "JuliaSetBoettcher"),
]

# ─────────────────────────── numbers, domains and number parts ───────────────────────────
# Reference: the ref pages of Overflow, Underflow, RealValuedNumberQ, RealValuedNumericQ, RealSign, RealExponent
# (machine-number results), MixedFractionParts, NumeratorDenominator, NumberDecompose, NumberCompose, MixedRadix,
# Element with the domains, and the $MaxPrecision/$MinPrecision variables.

NUMBERS = [
    ("Overflow[] + 5", "Overflow[]"),
    ("2 Overflow[]", "Overflow[]"),
    ("Overflow[] === Overflow[]", "True"),
    ("Overflow[1]", Raises()),
    ("2 Underflow[]", "Underflow[]"),
    ("Underflow[] === Underflow[]", "True"),
    ("Underflow[1]", Raises()),
    ("RealValuedNumberQ[3/4]", "True"),
    ("RealValuedNumberQ[2.5]", "True"),
    ("RealValuedNumberQ[I]", "False"),
    ("RealValuedNumberQ[Pi]", "False"),           # Pi is numeric but not a number
    ("RealValuedNumberQ[Overflow[]]", "True"),
    ("RealValuedNumberQ[Underflow[]]", "True"),
    ("RealValuedNumericQ[Pi]", "True"),
    ("RealValuedNumericQ[Sqrt[2] + E]", "True"),
    ("RealValuedNumericQ[I]", "False"),
    ("RealValuedNumericQ[x]", "False"),
    ("RealSign[-3.5]", "-1"),
    ("RealSign[0]", "0"),
    ("RealSign[Pi]", "1"),
    ("RealSign[{-2, 0, 7}]", "{-1, 0, 1}"),
    ("RealSign[I]", "RealSign[I]"),
    ("RealExponent[1000]", 3.0),
    ("RealExponent[0.01]", -2.0),
    ("RealExponent[8, 2]", 3.0),
    ("RealExponent[-100]", 2.0),
    ("RealExponent[123.456]", 2.091512201627772),
    ("RealExponent[0]", "-Infinity"),
    ("MixedFractionParts[7/3]", "{2, 1/3}"),
    ("MixedFractionParts[-7/3]", "{-2, -1/3}"),
    ("MixedFractionParts[2.75]", "{2, 0.75}"),
    ("NumeratorDenominator[3/4]", "{3, 4}"),
    ("NumeratorDenominator[(x + 1)/(x - 1)]", "{1 + x, -1 + x}"),
    ("NumeratorDenominator[{2/3, 5}]", "{{2, 5}, {3, 1}}"),
    ("NumberDecompose[37, {10, 1}]", "{3, 7}"),
    ("NumberDecompose[100, {60, 7, 1}]", "{1, 5, 5}"),
    ("NumberDecompose[3661, {3600, 60, 1}]", "{1, 1, 1}"),
    ("NumberCompose[{3, 7}, {10, 1}]", "37"),
    ("NumberCompose[{1, 2, 3}, {100, 10, 1}]", "123"),
    ("NumberCompose[{2, 5}, {3, 1}]", "11"),
    ("NumberCompose[NumberDecompose[100, {60, 7, 1}], {60, 7, 1}]", "100"),
    ("IntegerDigits[100, MixedRadix[{24, 60, 60}]]", "{1, 40}"),
    ("IntegerDigits[3661, MixedRadix[{24, 60, 60}]]", "{1, 1, 1}"),
    ("FromDigits[{1, 2, 3}, MixedRadix[{24, 60, 60}]]", "3723"),
    ("FromDigits[IntegerDigits[86399, MixedRadix[{24, 60, 60}]], MixedRadix[{24, 60, 60}]]", "86399"),
    ("MixedRadix[{2, 3}]", "MixedRadix[{2, 3}]"),
    ("$MaxPrecision", "Infinity"),
    ("$MaxPrecision === Infinity", "True"),
    ("$MaxPrecision = 20", Raises()),
    ("$MinPrecision", "0"),
    ("$MinPrecision === 0", "True"),
    ("$MinPrecision = 5", Raises()),
]

DOMAINS = [
    ("Element[3, PositiveIntegers]", "True"),
    ("Element[-3, PositiveIntegers]", "False"),
    ("Element[0, PositiveIntegers]", "False"),
    ("Element[0, NonNegativeIntegers]", "True"),
    ("Element[7, NonNegativeIntegers]", "True"),
    ("Element[-1, NonNegativeIntegers]", "False"),
    ("Element[-5, NegativeIntegers]", "True"),
    ("Element[5, NegativeIntegers]", "False"),
    ("Element[0, NegativeIntegers]", "False"),
    ("Element[2, NonPositiveIntegers]", "False"),
    ("Element[0, NonPositiveIntegers]", "True"),
    ("Element[-4, NonPositiveIntegers]", "True"),
    ("Element[1/3, PositiveRationals]", "True"),
    ("Element[Sqrt[2], PositiveRationals]", "False"),
    ("Element[-1/3, PositiveRationals]", "False"),
    ("Element[-1/2, NegativeRationals]", "True"),
    ("Element[1/2, NegativeRationals]", "False"),
    ("Element[-Sqrt[2], NegativeRationals]", "False"),
    ("Element[0, NonNegativeRationals]", "True"),
    ("Element[5/2, NonNegativeRationals]", "True"),
    ("Element[-5/2, NonNegativeRationals]", "False"),
    ("Element[-2/3, NonPositiveRationals]", "True"),
    ("Element[-2, NonPositiveRationals]", "True"),
    ("Element[2/3, NonPositiveRationals]", "False"),
    ("Element[Pi, PositiveReals]", "True"),
    ("Element[-Pi, PositiveReals]", "False"),
    ("Element[I, PositiveReals]", "False"),
    ("Element[-Pi, NegativeReals]", "True"),
    ("Element[0, NegativeReals]", "False"),
    ("Element[E, NegativeReals]", "False"),
    ("Element[Pi, NonNegativeReals]", "True"),
    ("Element[0, NonNegativeReals]", "True"),
    ("Element[-1, NonNegativeReals]", "False"),
    ("Element[0, NonPositiveReals]", "True"),
    ("Element[-E, NonPositiveReals]", "True"),
    ("Element[1, NonPositiveReals]", "False"),
    ("Element[I, Complexes]", "True"),
    ("Element[1 + I, Complexes]", "True"),
    ("Element[Pi, Complexes]", "True"),
    ("Element[True, Booleans]", "True"),
    ("Element[False, Booleans]", "True"),
    ("Element[1, Booleans]", "False"),
    ("Element[x, Booleans]", "Element[x, Booleans]"),
    ("Element[7, Primes]", "True"),
    ("Element[9, Primes]", "False"),
    ("Element[1/2, Primes]", "False"),
    ("Element[Sqrt[2], Algebraics]", "True"),
    ("Element[Sqrt[2] + Sqrt[3], Algebraics]", "True"),
    ("Element[Pi, Algebraics]", "False"),
    ("Element[E, Algebraics]", "False"),
    ("Element[2/3, Rationals]", "True"),
    ("Element[Sqrt[2], Rationals]", "False"),
    ("Element[-7, Rationals]", "True"),
    ("Element[Pi, Reals]", "True"),
    ("Element[I, Reals]", "False"),
    ("Element[Sqrt[2], Reals]", "True"),
    ("Element[3, Integers]", "True"),
    ("Element[1/2, Integers]", "False"),
    ("Element[-12, Integers]", "True"),
]

# Reference: the ref pages of GaussianIntegers, Modulus and Extension (documented factorisations).
OPTIONS = [
    ("Factor[x^2 + 1, GaussianIntegers -> True]", "(x - I) (x + I)"),
    ("FactorInteger[5, GaussianIntegers -> True]", "{{-I, 1}, {1 + 2 I, 1}, {2 + I, 1}}"),
    ("PrimeQ[3, GaussianIntegers -> True]", "True"),
    ("PrimeQ[5, GaussianIntegers -> True]", "False"),
    ("PrimeQ[5, GaussianIntegers -> False]", "True"),
    ("Factor[x^4 + 1, Modulus -> 2]", "(1 + x)^4"),
    ("Det[{{1, 2}, {3, 4}}, Modulus -> 5]", "3"),
    ("Inverse[{{1, 2}, {3, 4}}, Modulus -> 7]", "{{5, 1}, {5, 3}}"),
    ("Mod[{{1, 2}, {3, 4}} . Inverse[{{1, 2}, {3, 4}}, Modulus -> 7], 7]", "{{1, 0}, {0, 1}}"),
    ("PolynomialGCD[x^2 + 1, x + 1, Modulus -> 2]", "1 + x"),
    ("Factor[x^2 - 2, Extension -> Sqrt[2]]", "(x - Sqrt[2]) (x + Sqrt[2])"),
    ("Factor[x^4 + 1, Extension -> I]", "(x^2 - I) (x^2 + I)"),
    ("Expand[Factor[x^4 - 5 x^2 + 6, Extension -> {Sqrt[2], Sqrt[3]}]]", "x^4 - 5 x^2 + 6"),
]

# ─────────────────────────── polynomials, sequences, counts ───────────────────────────
# Reference: the ref pages of Cyclotomic, PrimitivePolynomialQ, ContinuedFractionK, DeBruijnSequence,
# ShiftRegisterSequence, FiniteGroupCount / FiniteAbelianGroupCount (OEIS A000001 / A000688), Groupings
# (Catalan numbers count the binary groupings), RandomComplex.

def _all_in_rect(lo, hi):
    def p(r):
        vals = [complex(v) if not hasattr(v, "args") or v.args == () else None for v in r.args]
        return all(v is not None and lo.real <= v.real <= hi.real and lo.imag <= v.imag <= hi.imag for v in vals)
    return Pred(p, f"complex numbers in the rectangle {lo} .. {hi}")


SEQUENCES = [
    ("Cyclotomic[6, x]", "1 - x + x^2"),
    ("Cyclotomic[12, x]", "1 - x^2 + x^4"),
    ("Cyclotomic[1, x]", "-1 + x"),
    ("Cyclotomic[5, 2]", "31"),
    ("Expand[Cyclotomic[1, x] Cyclotomic[2, x] Cyclotomic[3, x] Cyclotomic[6, x]]", "x^6 - 1"),
    ("PrimitivePolynomialQ[x^4 + x + 1, 2]", "True"),
    ("PrimitivePolynomialQ[x^4 + x^3 + x^2 + x + 1, 2]", "False"),   # irreducible, but x has order 5
    ("PrimitivePolynomialQ[x^2 + x + 2, 3]", "True"),
    ("PrimitivePolynomialQ[x^2 + 1, 3]", "False"),
    ("ContinuedFractionK[1, {i, 1, 3}]", "2/3"),
    ("ContinuedFractionK[2, {i, 1, 2}]", "2/5"),
    ("ContinuedFractionK[i, i^2, {i, 1, 2}]", "2/3"),
    ("ContinuedFractionK[1, {i, 1, Infinity}]", "(Sqrt[5] - 1)/2"),
    ("DeBruijnSequence[{0, 1}, 3]", "{0, 0, 0, 1, 0, 1, 1, 1}"),
    ("DeBruijnSequence[{a, b}, 2]", "{a, a, b, b}"),
    ("Length[DeBruijnSequence[{0, 1, 2}, 3]]", "27"),
    ("With[{s = DeBruijnSequence[{0, 1}, 4]}, Length[Union[Partition[Join[s, Take[s, 3]], 4, 1]]]]", "16"),
    ("ShiftRegisterSequence[3]", "{0, 0, 1, 0, 1, 1, 1}"),
    ("Length[ShiftRegisterSequence[5]]", "31"),
    ("Total[ShiftRegisterSequence[5]]", "16"),          # an m-sequence of length 2^n - 1 has 2^(n-1) ones
    ("FiniteGroupCount[{1, 4, 8, 12, 16, 32}]", "{1, 2, 5, 5, 14, 51}"),
    ("FiniteGroupCount[64]", "267"),
    ("FiniteGroupCount[p]", "FiniteGroupCount[p]"),
    ("FiniteAbelianGroupCount[{8, 16, 72, 1}]", "{3, 5, 6, 1}"),
    ("FiniteAbelianGroupCount[32]", "7"),
    ("FiniteAbelianGroupCount[30]", "1"),
    ("Groupings[3, 2]", "{{{1, 2}, 3}, {1, {2, 3}}}"),
    ("Groupings[{a, b, c}, 2]", "{{{a, b}, c}, {a, {b, c}}}"),
    ("Groupings[{a, b, c}, f -> 2]", "{f[f[a, b], c], f[a, f[b, c]]}"),
    ("Length[Groupings[5, 2]]", "14"),
    ("Length[Groupings[4, 3]]", "0"),
    ("Length[Groupings[{a, b, c, d, e}, 3]]", "3"),
    ("SeedRandom[1]; Head[RandomComplex[]]", "Complex"),
    ("SeedRandom[1]; RandomComplex[{0, 1 + I}, 50]", _all_in_rect(0j, 1 + 1j)),
    ("SeedRandom[2]; RandomComplex[{-1 - I, 1 + 2 I}, 50]", _all_in_rect(-1 - 1j, 1 + 2j)),
    ("Dimensions[RandomComplex[2 + 2 I, {3, 4}]]", "{3, 4}"),
]

# ─────────────────────────── Dirichlet characters and series, integer relations ───────────────────────────
# Reference: ref pages of DirichletCharacter, DirichletL (L(2, chi_4) = Catalan), DirichletConvolve,
# DirichletTransform, FindIntegerNullVector and LatticeReduce (documented example).

DIRICHLET = [
    ("DirichletCharacter[4, 2, 3]", "-1"),
    ("DirichletCharacter[4, 2, 2]", "0"),
    ("DirichletCharacter[5, 1, 7]", "1"),
    ("Table[DirichletCharacter[3, 2, n], {n, 6}]", "{1, -1, 0, 1, -1, 0}"),
    ("DirichletL[1, 1, 2]", "Pi^2/6"),
    ("N[DirichletL[4, 2, 2]]", 0.915965594177219),
    ("DirichletL[3, 1, s]", "(1 - 3^-s) Zeta[s]"),
    ("DirichletConvolve[1, 1, n, 12]", "6"),
    ("DirichletConvolve[n, 1, n, 6]", "12"),
    ("DirichletConvolve[MoebiusMu[n], 1, n, 30]", "0"),
    ("DirichletConvolve[1, 1, n, m]", "DivisorSigma[0, m]"),
    ("DirichletTransform[1, n, s]", "Zeta[s]"),
    ("DirichletTransform[MoebiusMu[n], n, s]", "1/Zeta[s]"),
    ("DirichletTransform[DivisorSigma[0, n], n, s]", "Zeta[s]^2"),
    ("FindIntegerNullVector[{1, Sqrt[2], 1 + Sqrt[2]}]", "{1, 1, -1}"),
    ("FindIntegerNullVector[{N[GoldenRatio, 30]^2, N[GoldenRatio, 30], 1}]", "{1, -1, -1}"),
    ("With[{v = FindIntegerNullVector[{1, Sqrt[2], Sqrt[8]}]}, Simplify[v . {1, Sqrt[2], Sqrt[8]}] == 0 && v != {0, 0, 0}]", "True"),
    ("LatticeReduce[{{1, 0, 0, 1345}, {0, 1, 0, 35}, {0, 0, 1, 154}}]", "{{0, 9, -2, 7}, {1, 1, -9, -6}, {1, -3, -8, 8}}"),
    ("Abs[Det[LatticeReduce[{{1, 1}, {2, 3}}]]]", "1"),
    ("LatticeReduce[{{1, 2, 3}, {2, 4, 6}}]", "{{1, 2, 3}}"),
    ("LatticeReduce[{{2, 0}, {0, 3}, {1, 1}}]", Pred(lambda r: sorted(tuple(abs(int(c)) for c in v.args) for v in r.args) == [(0, 1), (1, 0)], "a unimodular basis of Z^2")),
    ("LatticeReduce[{{1.5, 2}}]", Raises()),
]

# ─────────────────────────── algebraic numbers and number fields ───────────────────────────
# Reference: the ref pages of AlgebraicNumber, ToNumberField, the AlgebraicNumber* functions, the NumberField*
# functions, RootOfUnityQ, RootReduce, RootSum and RootApproximant; class numbers, discriminants, units and
# regulators agree with the LMFDB tables for the quadratic and cubic fields used.

ALGEBRAIC = [
    ("AlgebraicNumber[Sqrt[2], {1, 2}]", "AlgebraicNumber[Sqrt[2], {1, 2}]"),
    ("AlgebraicNumber[Sqrt[2], {1, 2}] + AlgebraicNumber[Sqrt[2], {3, 1}]", "AlgebraicNumber[Sqrt[2], {4, 3}]"),
    ("AlgebraicNumber[Sqrt[2], {0, 1}]^2", "AlgebraicNumber[Sqrt[2], {2, 0}]"),
    ("AlgebraicNumber[Root[#^3 - 2 &, 1], {1, 1, 1}] AlgebraicNumber[Root[#^3 - 2 &, 1], {0, 1}]",
     "AlgebraicNumber[2^(1/3), {2, 1, 1}]"),
    ("ToNumberField[Sqrt[2] + 1, Sqrt[2]]", "AlgebraicNumber[Sqrt[2], {1, 1}]"),
    ("ToNumberField[Sqrt[8], Sqrt[2]]", "AlgebraicNumber[Sqrt[2], {0, 2}]"),
    ("ToNumberField[Sqrt[3], Sqrt[2]]", Raises()),
    ("With[{t = ToNumberField[{Sqrt[2], Sqrt[3]}]}, RootReduce[Normal[t] - {Sqrt[2], Sqrt[3]}]]", "{0, 0}"),
    ("AlgebraicIntegerQ[(1 + Sqrt[5])/2]", "True"),
    ("AlgebraicIntegerQ[Sqrt[2]/2]", "False"),
    ("AlgebraicIntegerQ[3]", "True"),
    ("AlgebraicIntegerQ[1/2]", "False"),
    ("AlgebraicNumberDenominator[Sqrt[2]/2]", "2"),
    ("AlgebraicNumberDenominator[(1 + Sqrt[5])/2]", "1"),
    ("AlgebraicNumberDenominator[1/6]", "6"),
    ("AlgebraicNumberNorm[1 + Sqrt[2]]", "-1"),
    ("AlgebraicNumberNorm[AlgebraicNumber[Sqrt[2], {1, 3}]]", "-17"),            # 1 - 9*2
    ("AlgebraicNumberNorm[2^(1/3)]", "2"),
    ("AlgebraicNumberTrace[1 + Sqrt[2]]", "2"),
    ("AlgebraicNumberTrace[2^(1/3)]", "0"),
    ("AlgebraicNumberTrace[AlgebraicNumber[Sqrt[2], {1, 3}]]", "2"),
    ("AlgebraicNumberPolynomial[AlgebraicNumber[Sqrt[2], {1, 3}], x]", "1 + 3 x"),
    ("AlgebraicNumberPolynomial[AlgebraicNumber[Root[#^3 - 2 &, 1], {1, 0, 5}], t]", "1 + 5 t^2"),
    ("AlgebraicNumberPolynomial[AlgebraicNumber[Sqrt[2], {1, 3}], x] /. x -> Sqrt[2]", "1 + 3 Sqrt[2]"),
    ("AlgebraicUnitQ[1 + Sqrt[2]]", "True"),
    ("AlgebraicUnitQ[2]", "False"),
    ("AlgebraicUnitQ[(1 + Sqrt[5])/2]", "True"),
    ("AlgebraicUnitQ[Sqrt[2]]", "False"),
    ("NumberFieldClassNumber[Sqrt[-5]]", "2"),
    ("NumberFieldClassNumber[Sqrt[-23]]", "3"),
    ("NumberFieldClassNumber[Sqrt[2]]", "1"),
    ("NumberFieldClassNumber[Sqrt[10]]", "2"),
    ("NumberFieldDiscriminant[Sqrt[2]]", "8"),
    ("NumberFieldDiscriminant[Sqrt[5]]", "5"),
    ("NumberFieldDiscriminant[2^(1/3)]", "-108"),
    ("NumberFieldDiscriminant[Sqrt[-3]]", "-3"),
    ("NumberFieldFundamentalUnits[Sqrt[2]]", "{AlgebraicNumber[Sqrt[2], {1, 1}]}"),
    ("NumberFieldFundamentalUnits[Sqrt[5]]", "{AlgebraicNumber[Sqrt[5], {1/2, 1/2}]}"),
    ("NumberFieldFundamentalUnits[Sqrt[3]]", "{AlgebraicNumber[Sqrt[3], {2, 1}]}"),
    ("NumberFieldFundamentalUnits[Sqrt[-1]]", "{}"),
    ("NumberFieldIntegralBasis[Sqrt[5]]", "{1, AlgebraicNumber[Sqrt[5], {1/2, 1/2}]}"),
    ("NumberFieldIntegralBasis[Sqrt[2]]", "{1, AlgebraicNumber[Sqrt[2], {0, 1}]}"),
    ("NumberFieldIntegralBasis[Sqrt[-3]]", "{1, AlgebraicNumber[I Sqrt[3], {1/2, 1/2}]}"),
    ("NumberFieldRegulator[Sqrt[2]]", "Log[1 + Sqrt[2]]"),
    ("NumberFieldRegulator[Sqrt[5]]", "Log[(1 + Sqrt[5])/2]"),
    ("NumberFieldRegulator[I]", "1"),
    ("NumberFieldRootsOfUnity[Sqrt[2]]", "{-1, 1}"),
    ("Length[NumberFieldRootsOfUnity[I]]", "4"),
    ("Length[NumberFieldRootsOfUnity[Sqrt[-3]]]", "6"),
    ("NumberFieldRootsOfUnity[I]", "{-1, 1, AlgebraicNumber[I, {0, -1}], AlgebraicNumber[I, {0, 1}]}"),
    ("NumberFieldSignature[Sqrt[2]]", "{2, 0}"),
    ("NumberFieldSignature[2^(1/3)]", "{1, 1}"),
    ("NumberFieldSignature[I]", "{0, 1}"),
    ("NumberFieldSignature[Root[#^4 - 2 &, 1]]", "{2, 1}"),
    ("AlgebraicNumberNorm /@ NumberFieldNormRepresentatives[Sqrt[2], 7]", "{7, 7}"),   # X01: implemented (quadratic)
    ("RootOfUnityQ[(-1)^(2/7)]", "True"),
    ("RootOfUnityQ[(1 + I)/Sqrt[2]]", "True"),
    ("RootOfUnityQ[-1]", "True"),
    ("RootOfUnityQ[2]", "False"),
    ("RootOfUnityQ[(3 + 4 I)/5]", "False"),                # |z| = 1 but not a root of unity
    ("RootReduce[Sqrt[2] + Sqrt[3]]", "Root[1 - 10 #^2 + #^4 &, 4]"),
    ("RootReduce[(Sqrt[2] + 1)^2]", "3 + 2 Sqrt[2]"),
    ("RootReduce[Root[#^2 - 2 &, 2]^2]", "2"),
    ("RootReduce[Sqrt[2] Sqrt[3] - Sqrt[6]]", "0"),
    ("RootSum[#^2 - 2 &, #^2 &]", "4"),
    ("RootSum[#^3 + # + 1 &, #^2 &]", "-2"),
    ("RootSum[#^3 - 2 &, # &]", "0"),
    ("RootApproximant[1.4142135623730951]", "Sqrt[2]"),
    ("RootApproximant[N[GoldenRatio, 20]]", "(1 + Sqrt[5])/2"),
    ("RootApproximant[1.2599210498948732]", "2^(1/3)"),
    ("RootApproximant[0.5]", "1/2"),
    ("Element[2^(1/3), Algebraics]", "True"),
]

# ─────────────────────────── group theory ───────────────────────────
# Reference: the ref pages of the sporadic group heads (orders from the ATLAS of Finite Groups), GroupStabilizer,
# GroupSetwiseStabilizer, GroupCentralizer, RightCosetRepresentative, GroupStabilizerChain, GroupActionBase,
# GroupElementFromWord / GroupElementToWord (PermutationProduct applies left to right) and CycleIndexPolynomial.

SPORADIC = [("MathieuGroupM11", 7920, True), ("MathieuGroupM12", 95040, True), ("MathieuGroupM22", 443520, True),
            ("MathieuGroupM23", 10200960, True), ("MathieuGroupM24", 244823040, True),
            ("JankoGroupJ1", 175560, None), ("JankoGroupJ2", 604800, None), ("JankoGroupJ3", 50232960, None),
            ("JankoGroupJ4", 86775571046077562880, None), ("HigmanSimsGroupHS", 44352000, None),
            ("McLaughlinGroupMcL", 898128000, None), ("HeldGroupHe", 4030387200, None),
            ("RudvalisGroupRu", 145926144000, None), ("SuzukiGroupSuz", 448345497600, None),
            ("ONanGroupON", 460815505920, None), ("ConwayGroupCo3", 495766656000, None),
            ("ConwayGroupCo2", 42305421312000, None), ("ConwayGroupCo1", 4157776806543360000, None),
            ("FischerGroupFi22", 64561751654400, None), ("FischerGroupFi23", 4089470473293004800, None),
            ("FischerGroupFi24Prime", 1255205709190661721292800, None),
            ("HaradaNortonGroupHN", 273030912000000, None), ("LyonsGroupLy", 51765179004000000, None),
            ("ThompsonGroupTh", 90745943887872000, None),
            ("BabyMonsterGroupB", 4154781481226426191177580544000000, None),
            ("MonsterGroupM", 808017424794512875886459904961710757005754368000000000, None),
            ("TitsGroupT", 17971200, None)]


def _sporadic_cases():
    out = []
    for name, order, perm in SPORADIC:
        out.append((f"GroupOrder[{name}[]]", str(order)))
        out.append((f"{name}[] === {name}[]", "True"))
        out.append((f"{name}[1]", Raises()))
        if perm:
            out.append((f"GroupOrder[PermutationGroup[GroupGenerators[{name}[]]]]", str(order)))
    return out


def _elements_ok(n):
    def p(r):
        return r.head == "List" and len(r.args) == n and len(set(map(str, r.args))) == n and str(r.args[0]) == "Cycles[{}]"
    return Pred(p, f"{n} distinct elements starting with the identity")


GROUPS = _sporadic_cases() + [
    ("GroupGenerators[MonsterGroupM[]]", Raises()),
    ("GroupOrder[GroupStabilizer[SymmetricGroup[4], {1}]]", "6"),
    ("GroupOrder[GroupStabilizer[SymmetricGroup[5], {1, 2}]]", "6"),
    ("GroupOrder[GroupStabilizer[MathieuGroupM11[], {1}]]", "720"),        # M11 is 4-transitive on 11 points
    ("GroupStabilizer[SymmetricGroup[4], {x}]", Raises()),
    ("GroupOrder[GroupSetwiseStabilizer[SymmetricGroup[5], {1, 2}]]", "12"),
    ("GroupOrder[GroupSetwiseStabilizer[CyclicGroup[6], {1, 4}]]", "2"),
    ("GroupOrder[GroupSetwiseStabilizer[SymmetricGroup[4], {1, 2, 3, 4}]]", "24"),
    ("GroupOrder[GroupCentralizer[SymmetricGroup[4], Cycles[{{1, 2}}]]]", "4"),
    ("GroupOrder[GroupCentralizer[SymmetricGroup[5], Cycles[{{1, 2, 3, 4, 5}}]]]", "5"),
    ("GroupOrder[GroupCentralizer[DihedralGroup[4], Cycles[{{1, 3}, {2, 4}}]]]", "8"),   # the centre of D4
    ("RightCosetRepresentative[SymmetricGroup[3], Cycles[{{1, 2}}]]", "Cycles[{}]"),
    ("RightCosetRepresentative[PermutationGroup[{Cycles[{{1, 2}}]}], Cycles[{{1, 2, 3}}]]", "Cycles[{{1, 2, 3}}]"),
    ("RightCosetRepresentative[PermutationGroup[{Cycles[{{1, 2}}]}], Cycles[{{1, 3}}]]", "Cycles[{{1, 2, 3}}]"),
    ("GroupStabilizerChain[SymmetricGroup[3]]",
     "{{} -> PermutationGroup[{Cycles[{{1, 2}}], Cycles[{{1, 2, 3}}]}], {1} -> PermutationGroup[{Cycles[{{2, 3}}]}], "
     "{1, 2} -> PermutationGroup[{}]}"),
    ("Length[GroupStabilizerChain[SymmetricGroup[5]]]", "5"),
    ("Map[GroupOrder, Last /@ GroupStabilizerChain[MathieuGroupM11[]]]", "{7920, 720, 72, 8, 1}"),
    ("First /@ GroupStabilizerChain[SymmetricGroup[4], GroupActionBase -> {4, 3}]", "{{}, {4}, {4, 3}, {4, 3, 1}}"),
    ("Map[GroupOrder, Last /@ GroupStabilizerChain[SymmetricGroup[4], GroupActionBase -> {4, 3}]]", "{24, 6, 2, 1}"),
    ("GroupElements[SymmetricGroup[3], GroupActionBase -> {3}]", _elements_ok(6)),
    ("With[{e = GroupElements[SymmetricGroup[4], GroupActionBase -> {4, 2}]}, "
     "Length[e] == 24 && And @@ (MemberQ[GroupElements[SymmetricGroup[4]], #] & /@ e)]", "True"),
    ("GroupElements[PermutationGroup[{Cycles[{{1, 2}, {3, 4}}], Cycles[{{1, 3}}]}], GroupActionBase -> {4, 2}]",
     _elements_ok(8)),
    ("GroupElementFromWord[SymmetricGroup[4], {1, 2}]", "Cycles[{{1, 3, 4}}]"),
    ("GroupElementFromWord[PermutationGroup[{Cycles[{{1, 2, 3}}]}], {1, 1}]", "Cycles[{{1, 3, 2}}]"),
    ("GroupElementFromWord[PermutationGroup[{Cycles[{{1, 2, 3}}]}], {-1}]", "Cycles[{{1, 3, 2}}]"),
    ("GroupElementFromWord[SymmetricGroup[4], {3}]", Raises()),
    ("GroupElementToWord[PermutationGroup[{Cycles[{{1, 2, 3}}]}], Cycles[{{1, 3, 2}}]]", "{-1}"),
    ("With[{G = SymmetricGroup[5], g = Cycles[{{1, 3, 5}, {2, 4}}]}, "
     "GroupElementFromWord[G, GroupElementToWord[G, g]] == g]", "True"),
    ("With[{G = MathieuGroupM11[], g = PermutationProduct @@ GroupGenerators[MathieuGroupM11[]]}, "
     "GroupElementFromWord[G, GroupElementToWord[G, g]] == g]", "True"),
    ("CycleIndexPolynomial[Cycles[{{1, 2}, {3, 4, 5}}], {cy1, cy2, cy3}]", "cy2 cy3"),
    ("CycleIndexPolynomial[Cycles[{{1, 2}}], {cy1, cy2}, 4]", "cy1^2 cy2"),
    ("CycleIndexPolynomial[SymmetricGroup[3], {cy1, cy2, cy3}]", "cy1^3/6 + cy1 cy2/2 + cy3/3"),
    ("CycleIndexPolynomial[CyclicGroup[4], {cy1, cy2, cy3, cy4}]", "cy1^4/4 + cy2^2/4 + cy4/2"),
    # Polya: the number of 2-colourings of the 4-cycle up to rotation is 6
    ("CycleIndexPolynomial[CyclicGroup[4], {cy1, cy2, cy3, cy4}] /. {cy1 -> 2, cy2 -> 2, cy3 -> 2, cy4 -> 2}", "6"),
]

# ─────────────────────────── AnglePath3D and JuliaSetPoints ───────────────────────────
# Reference: reference.wolfram.com/language/ref/AnglePath3D.html (steps rotate the local frame by
# EulerMatrix[{a, b, g}, {3, 2, 1}] and advance along the local x axis; the documented equivalences between the
# step forms, with AnglePath and with FoldList of the frame matrices) and JuliaSetPoints.html (points of the
# Julia set; JuliaSetPoints[c] === JuliaSetPoints[z^2 + c, z]).  The Julia set of z^2 is the unit circle and that
# of z^2 - 2 the segment [-2, 2].

ANGLEPATH3D = [
    ("AnglePath3D[{{Pi/2, 0, 0}, {Pi/2, 0, 0}, {Pi/2, 0, 0}}]", "{{0, 0, 0}, {0, 1, 0}, {-1, 1, 0}, {-1, 0, 0}}"),
    ("AnglePath3D[{{1, {Pi/2, Pi/2}}, {2, {Pi/2, Pi/2}}, {3, {Pi/2, Pi/2}}}]",
     "{{0, 0, 0}, {0, 0, -1}, {0, -2, -1}, {3, -2, -1}}"),
    ("AnglePath3D[{{2, 5, 7}, {{0, Pi/2, -5 Pi/6}, {3 Pi/2, 0, Pi/3}, {0, -Pi/3, 0}}}] === "
     "AnglePath3D[{{2, {0, Pi/2, -5 Pi/6}}, {5, {3 Pi/2, 0, Pi/3}}, {7, {0, -Pi/3, 0}}}]", "True"),
    ("AnglePath3D[{{2, 5, 7}, {{0, Pi/2, -5 Pi/6}, {3 Pi/2, 0, Pi/3}, {0, -Pi/3, 0}}}]",
     "{{0, 0, 0}, {0, 0, -2}, {5/2, 5 Sqrt[3]/2, -2}, {13/8, 41 Sqrt[3]/8, 13/4}}"),
    ("Last[AnglePath3D[ConstantArray[{0, Pi/3, 0}, 6]]]", "{0, 0, 0}"),             # closed hexagon
    ("AnglePath3D[ConstantArray[{0, Pi/3, 0}, 6]][[All, 2]]", "{0, 0, 0, 0, 0, 0, 0}"),   # in the x-z plane
    ("AnglePath3D[{{Pi/2, 0, 0}}, ConstantArray[{0, Pi/4, 0}, 8]][[All, 1]]", "{0, 0, 0, 0, 0, 0, 0, 0, 0}"),
    ("Last[AnglePath3D[{{Pi/2, 0, 0}}, ConstantArray[{0, Pi/4, 0}, 8]]]", "{0, 0, 0}"),  # closed octagon
    ("AnglePath3D[{{al1, be1}, {al2, be2}}] === AnglePath3D[{{al1, be1, 0}, {al2, be2, 0}}]", "True"),
    ("Simplify /@ Flatten[AnglePath3D[{{th1, 0, 0}, {th2, 0, 0}}][[All, {1, 2}]] - AnglePath[{th1, th2}]]",
     "{0, 0, 0, 0, 0, 0}"),
    ("AnglePath3D[{{th1, 0, 0}, {th2, 0, 0}}][[All, 3]]", "{0, 0, 0}"),
    ("AnglePath3D[{1, 3, -2}, {{Pi/2, 0, Pi/6}, {Pi/3, -2 Pi/3, 0}}] - "
     "AnglePath3D[{{Pi/2, 0, Pi/6}, {Pi/3, -2 Pi/3, 0}}]", "{{1, 3, -2}, {1, 3, -2}, {1, 3, -2}}"),
    ("AnglePath3D[{{{1/2, -Sqrt[3]/2, 0}, {Sqrt[3]/2, 1/2, 0}, {0, 0, 1}}, "
     "{{0, -1, 0}, {-1/2, 0, -Sqrt[3]/2}, {Sqrt[3]/2, 0, -1/2}}}]",
     "{{0, 0, 0}, {1/2, Sqrt[3]/2, 0}, {1/2 + Sqrt[3]/4, -1/4 + Sqrt[3]/2, Sqrt[3]/2}}"),
    ("AnglePath3D[{{Pi/3, 0}, {Pi/2, -2 Pi/3}}, \"FrameMatrix\"] == "
     "FoldList[Dot, AnglePath3D[{{Pi/3, 0}, {Pi/2, -2 Pi/3}}, \"EulerMatrix\"]]", "True"),
    ("First[AnglePath3D[{{Pi/3, 0}, {Pi/2, -2 Pi/3}}, \"FrameMatrix\"]]", "IdentityMatrix[3]"),
    ("With[{f = AnglePath3D[{{Pi/3, 0}, {Pi/2, -2 Pi/3}}, \"FrameMatrix\"]}, "
     "AnglePath3D[{{Pi/3, 0}, {Pi/2, -2 Pi/3}}] == FoldList[#1 + #2 . {1, 0, 0} &, {0, 0, 0}, Rest[f]]]", "True"),
    ("AnglePath3D[{{Pi/2, 0}, {0, -Pi/2}}, {\"EulerAngles\", \"Position\"}]",
     "{{{0, 0, 0}, {0, 0, 0}}, {{Pi/2, 0, 0}, {0, 1, 0}}, {{0, -Pi/2, 0}, {0, 1, 1}}}"),
    ("AnglePath3D[{{0.5, 0.2, 0.1}, {0.3, 0.1, 0.2}}]",
     [[0, 0, 0], [0.8600893382050473, 0.4698689469495153, -0.19866933079506122],
      [1.5204121720357495, 1.1753492178326606, -0.4561015403501878]], 1e-12),
    ("Last[AnglePath3D[Table[{Pi/3, 0, 0}, 7]]]", "{1/2, Sqrt[3]/2, 0}"),
    ("Length[AnglePath3D[Table[{Pi/3, 0, 0}, 300]]]", "301"),
    ("AnglePath3D[{{1, 2, 3, 4}}]", Raises()),
    ("AnglePath3D[{{Pi/2, 0}}, \"RotationTranslation\"]", Raises()),
    ("AnglePath3D[{{Pi/2, 0}}, \"NoSuchForm\"]", Raises()),
]

def _julia_ok(c, tol, n_min, on=None, steps=8):
    """Every point has a bounded forward orbit under z^2 + c and (optionally) satisfies on(z)."""
    def p(r):
        zs = [complex(float(q.args[0]), float(q.args[1])) for q in r.args]
        if len(zs) < n_min:
            return False
        for z in zs:
            w = z
            for _ in range(steps):
                w = w * w + c
            if abs(w) > 2 or (on is not None and not on(z)):
                return False
        # the points are at least the closeness tolerance apart (checked on a sample)
        sample = zs[:: max(1, len(zs) // 200)]
        return all(abs(a - b) >= tol * (1 - 1e-9) for i, a in enumerate(sample) for b in sample[i + 1:])
    return Pred(p, f"at least {n_min} points on the Julia set of z^2 + {c}")


def _rational_julia_ok(f, tol, n_min, bound=6):
    """f maps the computed points (numerically) onto the computed point set."""
    def p(r):
        zs = [complex(float(q.args[0]), float(q.args[1])) for q in r.args]
        if len(zs) < n_min:
            return False
        for z in zs[:: max(1, len(zs) // 300)]:
            w = f(z)
            if abs(w) < bound and min(abs(w - y) for y in zs) > 3 * tol:
                return False
        return True
    return Pred(p, f"at least {n_min} points of a forward-invariant set")


JULIA = [
    ("JuliaSetPoints[0]", _julia_ok(0, 0.004, 100, on=lambda z: abs(abs(z) - 1) < 1e-8)),          # unit circle
    ("JuliaSetPoints[-2]", _julia_ok(-2, 0.004, 100, on=lambda z: abs(z.imag) < 1e-6 and abs(z.real) <= 2 + 1e-6)),
    ("JuliaSetPoints[-0.77 + 0.22 I]", _julia_ok(-0.77 + 0.22j, 0.004, 1000)),
    ("JuliaSetPoints[-1, \"ClosenessTolerance\" -> 0.01]", _julia_ok(-1, 0.01, 100)),
    ("JuliaSetPoints[-0.6 - 0.5 I, \"ClosenessTolerance\" -> 0.02] === "
     "JuliaSetPoints[z^2 - 0.6 - 0.5 I, z, \"ClosenessTolerance\" -> 0.02]", "True"),
    ("Length[JuliaSetPoints[0.37 + 0.22 I, \"ClosenessTolerance\" -> 0.05]] < "
     "Length[JuliaSetPoints[0.37 + 0.22 I, \"ClosenessTolerance\" -> 0.02]]", "True"),
    ("JuliaSetPoints[(z^2 - 1)/(z^2 + 1), z, \"ClosenessTolerance\" -> 0.02]",
     _rational_julia_ok(lambda z: (z * z - 1) / (z * z + 1), 0.02, 50)),
    ("JuliaSetPoints[z^3 - z - 1, z, \"ClosenessTolerance\" -> 0.02]",
     _rational_julia_ok(lambda z: z ** 3 - z - 1, 0.02, 50, bound=3)),
    ("JuliaSetPoints[cJulia]", "JuliaSetPoints[cJulia]"),
    ("JuliaSetPoints[z + 1, z]", Raises()),
    ("JuliaSetPoints[0, \"ClosenessTolerance\" -> -1]", Raises()),
]

CASES = (FINITE_FIELDS + FROBENIUS + INDICES + EMBEDDINGS + CURVES + MANDELBROT + NUMBERS + DOMAINS + OPTIONS +
         SEQUENCES + DIRICHLET + ALGEBRAIC + GROUPS + ANGLEPATH3D + JULIA)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w09(case):
    check(case)
