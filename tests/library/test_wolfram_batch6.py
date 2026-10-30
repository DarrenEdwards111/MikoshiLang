"""Differential tests for coverage batch 6 (discrete maths, combinatorics, logic and sets).

Expected values come from the Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>.html:
examples and the identities stated in their Details sections), OEIS sequences, or independent references computed
here with itertools, SymPy (sympy.combinatorics, sympy.logic, sympy.ntheory) or plain Python.
"""
import itertools
import math
from fractions import Fraction

import pytest
import sympy as sp
from sympy.combinatorics import Permutation as SPerm

from _harness import Pred, Raises, check, ids


def plist_str(perms):
    return "{" + ", ".join("{" + ", ".join(map(str, p)) + "}" for p in perms) + "}"


def cycles_str(perm):
    """Canonical Cycles[...] source for a 1-based image tuple (independent implementation)."""
    n, seen, cyc = len(perm), set(), []
    for s in range(1, n + 1):
        if s in seen or perm[s - 1] == s:
            continue
        c, p = [], s
        while p not in seen:
            seen.add(p)
            c.append(p)
            p = perm[p - 1]
        cyc.append(c)
    return "Cycles[{" + ", ".join("{" + ", ".join(map(str, c)) + "}" for c in cyc) + "}]"


def S_n_lex(n):
    return [p for p in itertools.permutations(range(1, n + 1))]


def even(p):
    inv = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
    return inv % 2 == 0


S3 = "{" + ", ".join(cycles_str(p) for p in S_n_lex(3)) + "}"
A4 = "{" + ", ".join(cycles_str(p) for p in S_n_lex(4) if even(p)) + "}"


def table_ref(perms):
    """Multiplication table: position of (apply g_i then g_j) among perms (1-based image tuples)."""
    pos = {p: i + 1 for i, p in enumerate(perms)}
    return "{" + ", ".join("{" + ", ".join(str(pos[tuple(b[a[k] - 1] for k in range(len(a)))]) for b in perms) + "}"
                           for a in perms) + "}"


def dihedral(n):
    rot = tuple(list(range(2, n + 1)) + [1])
    ref = tuple([1] + list(range(n, 1, -1)))
    elems = {tuple(range(1, n + 1))}
    frontier = list(elems)
    while frontier:
        nxt = []
        for e in frontier:
            for g in (rot, ref):
                p = tuple(g[e[k] - 1] for k in range(n))
                if p not in elems:
                    elems.add(p)
                    nxt.append(p)
        frontier = nxt
    return sorted(elems)


def frobenius_brute(a):
    reach = [False] * 2000
    reach[0] = True
    for i in range(1, 2000):
        reach[i] = any(i >= x and reach[i - x] for x in a)
    return max(i for i in range(2000) if not reach[i])


def frobsolve_brute(a, b):
    return sorted(t for t in itertools.product(*[range(b // x + 1) for x in a]) if sum(x * y for x, y in zip(a, t)) == b)


def unrank_ref(n, r):
    subs = [c for k in range(n + 1) for c in itertools.combinations(range(1, n + 1), k)]
    return "{" + ", ".join(map(str, subs[r - 1])) + "}"


PERMUTATIONS = [
    # Cycles canonical form
    ("Cycles[{{3, 1, 2}, {5, 4}, {7}}]", "Cycles[{{1, 2, 3}, {4, 5}}]"),
    ("Cycles[{{4, 5}, {2, 1}}]", "Cycles[{{1, 2}, {4, 5}}]"), ("Cycles[{{1}}]", "Cycles[{}]"),
    ("Cycles[{{1, 1}}]", Raises()), ("Cycles[{{0, 1}}]", Raises()), ("Cycles[{{1, 2}, {2, 3}}]", Raises()),
    # PermutationCyclesQ / PermutationListQ
    ("PermutationCyclesQ[Cycles[{{1, 2}, {4, 5}}]]", True), ("PermutationCyclesQ[Cycles[{}]]", True),
    ("PermutationCyclesQ[{2, 1}]", False), ("PermutationCyclesQ[x]", False),
    ("PermutationListQ[{3, 1, 2}]", True), ("PermutationListQ[{}]", True), ("PermutationListQ[{1, 3}]", False),
    ("PermutationListQ[{1, 1}]", False), ("PermutationListQ[Cycles[{{1, 2}}]]", False),
    # PermutationCycles / PermutationList (reference: the list {p1, ..., pn} maps i -> p_i)
    ("PermutationCycles[{2, 3, 1, 5, 4}]", "Cycles[{{1, 2, 3}, {4, 5}}]"),
    ("PermutationCycles[{1, 2, 3}]", "Cycles[{}]"), ("PermutationCycles[{4, 1, 3, 2}]", "Cycles[{{1, 4, 2}}]"),
    ("PermutationList[Cycles[{{1, 3}}]]", "{3, 2, 1}"), ("PermutationList[Cycles[{{1, 3}}], 5]", "{3, 2, 1, 4, 5}"),
    ("PermutationList[Cycles[{}]]", "{}"), ("PermutationList[Cycles[{{2, 4}}], 3]", Raises()),
    ("PermutationList[{2, 1, 3}]", "{2, 1, 3}"),
    ("PermutationList[PermutationCycles[{5, 3, 2, 4, 6, 1}], 6]", "{5, 3, 2, 4, 6, 1}"),
    # PermutationProduct: left to right; PermutationProduct[l2, l1] == l1[[l2]] (reference page)
    ("PermutationProduct[Cycles[{{1, 2, 3}}], Cycles[{{2, 3}}]]", "Cycles[{{1, 3}}]"),
    ("PermutationProduct[Cycles[{{2, 3}}], Cycles[{{1, 2, 3}}]]", "Cycles[{{1, 2}}]"),
    ("PermutationProduct[{2, 1, 3, 5, 6, 4}, {5, 3, 2, 4, 6, 1}]", "{5, 3, 2, 4, 6, 1}[[{2, 1, 3, 5, 6, 4}]]"),
    ("PermutationProduct[Cycles[{{3, 5}}], Cycles[{{3, 7}}], Cycles[{{3, 4}}], Cycles[{{3, 9}}]]",
     "Cycles[{{3, 5, 7, 4, 9}}]"),
    ("PermutationProduct[]", "Cycles[{}]"), ("PermutationProduct[Cycles[{{1, 4, 3, 2}}]]", "Cycles[{{1, 4, 3, 2}}]"),
    ("PermutationProduct[Cycles[{{1, 4, 3, 2}}], Cycles[{}]]", "Cycles[{{1, 4, 3, 2}}]"),
    ("PermutationProduct[x]", "x"), ("PermutationProduct[I]", "I"),
    ("PermutationProduct[Cycles[{{1, 7, 2}, {3, 9}}], InversePermutation[Cycles[{{1, 7, 2}, {3, 9}}]]]", "Cycles[{}]"),
    # InversePermutation / PermutationPower / PermutationOrder
    ("InversePermutation[Cycles[{{1, 7, 2}, {3, 9}}]]", "Cycles[{{1, 2, 7}, {3, 9}}]"),
    ("InversePermutation[{2, 3, 1}]", "{3, 1, 2}"), ("InversePermutation[Cycles[{}]]", "Cycles[{}]"),
    ("PermutationPower[Cycles[{{1, 2, 3, 4}}], 2]", "Cycles[{{1, 3}, {2, 4}}]"),
    ("PermutationPower[Cycles[{{1, 2, 3, 4}}], -1]", "Cycles[{{1, 4, 3, 2}}]"),
    ("PermutationPower[Cycles[{{1, 2, 3}}], 3]", "Cycles[{}]"),
    ("PermutationPower[Cycles[{{2, 3, 5}, {1, 4, 6, 8, 10, 15, 11}}], 4] === "
     "PermutationProduct[#, #, #, #] &[Cycles[{{2, 3, 5}, {1, 4, 6, 8, 10, 15, 11}}]]", True),
    ("PermutationPower[{2, 3, 1}, 2]", "{3, 1, 2}"),
    ("PermutationOrder[Cycles[{{2, 3, 5}, {1, 4, 6, 8, 10, 15, 11}}]]", "21"),
    ("PermutationOrder[Cycles[{}]]", "1"), ("PermutationOrder[{2, 1, 4, 5, 3}]", "6"),
    # support, length, max, min (reference: Max of the empty support is -Infinity)
    ("PermutationSupport[Cycles[{{1, 6, 3}, {2, 5, 12, 9}}]]", "{1, 2, 3, 5, 6, 9, 12}"),
    ("PermutationSupport[{1, 4, 3, 2}]", "{2, 4}"), ("PermutationSupport[Cycles[{}]]", "{}"),
    ("PermutationLength[Cycles[{{1, 6, 3}, {2, 5, 12, 9}}]]", "7"), ("PermutationLength[{1, 2, 3}]", "0"),
    ("PermutationLength[{3, 2, 1}]", "2"),
    ("PermutationMax[Cycles[{{1, 6, 3}, {2, 5, 12, 9}}]]", "12"),
    ("PermutationMax[{1, 4, 3, 10, 2, 6, 5, 7, 9, 8, 11}]", "10"), ("PermutationMax[Cycles[{}]]", "-Infinity"),
    ("PermutationMax[{1, 2, 3, 4, 5}]", "-Infinity"),
    ("PermutationMax[PermutationGroup[{Cycles[{{1, 3, 5, 7}}], Cycles[{{1, 2}, {3, 4}}]}]]", "7"),
    ("PermutationMin[Cycles[{{4, 6}, {5, 9}}]]", "4"), ("PermutationMin[{1, 2, 4, 3}]", "3"),
    ("PermutationMin[Cycles[{}]]", "Infinity"),
    # PermutationReplace / Permute / FindPermutation
    ("PermutationReplace[{1, 2, 3, 4}, Cycles[{{1, 3, 2}}]]", "{3, 1, 2, 4}"),
    ("PermutationReplace[Range[5], {2, 3, 1}]", "{2, 3, 1, 4, 5}"),
    ("PermutationReplace[Cycles[{{1, 2}}], Cycles[{{2, 3}}]]", "Cycles[{{1, 3}}]"),
    ("PermutationReplace[{{1, 2}, x}, Cycles[{{1, 2}}]]", "{{2, 1}, x}"),
    ("PermutationReplace[2, SymmetricGroup[3]]", Pred(lambda g: sorted(g.args) == [1, 1, 2, 2, 3, 3], "orbit of 2")),
    ("Permute[{a, b, c, d, e}, Cycles[{{1, 3, 2}}]]", "{b, c, a, d, e}"),
    ("Permute[{a, b, c}, {2, 3, 1}]", "{c, a, b}"), ("Permute[f[a, b, c], Cycles[{{1, 2}}]]", "f[b, a, c]"),
    ("Permute[{a, b}, Cycles[{{1, 3}}]]", Raises()),
    ("Permute[F[a, b, c, d, e], AlternatingGroup[3]]", "{F[a, b, c, d, e], F[c, a, b, d, e], F[b, c, a, d, e]}"),
    ("FindPermutation[{a, b, c}, {c, a, b}]", "Cycles[{{1, 2, 3}}]"),
    ("Permute[{a, b, c, d, e}, FindPermutation[{a, b, c, d, e}, {d, e, a, c, b}]]", "{d, e, a, c, b}"),
    ("FindPermutation[{c, a, b}]", "Cycles[{{1, 2, 3}}]"), ("Permute[Sort[{c, a, b}], FindPermutation[{c, a, b}]]",
                                                              "{c, a, b}"),
    ("FindPermutation[{a, a, b}, {a, b, a}]", Raises()), ("FindPermutation[{a, b}, {a, c}]", Raises()),
    # Signature (sign of the sorting permutation; 0 for repeated elements)
    ("Signature[{a, b, c}]", "1"), ("Signature[{b, a, c}]", "-1"), ("Signature[{a, b, a}]", "0"),
    ("Signature[{3, 1, 2}]", "1"), ("Signature[{4, 3, 2, 1}]", "1"), ("Signature[{}]", "1"),
    ("Signature[{5, 2, 9, 1, 7}]", str(SPerm([sorted([5, 2, 9, 1, 7]).index(v) for v in [5, 2, 9, 1, 7]]).signature())),
    # RandomPermutation: seeded reproducibility and validity
    ("SeedRandom[7]; RandomPermutation[6] === (SeedRandom[7]; RandomPermutation[6])", True),
    ("PermutationCyclesQ[RandomPermutation[10]]", True), ("PermutationMax[RandomPermutation[5]] <= 5", True),
    ("Length[RandomPermutation[4, 3]]", "3"),
    ("GroupElementQ[AlternatingGroup[5], RandomPermutation[AlternatingGroup[5]]]", True),
]

GROUPS = [
    ("GroupOrder[SymmetricGroup[5]]", "120"), ("GroupOrder[SymmetricGroup[0]]", "1"),
    ("GroupOrder[AlternatingGroup[10]]", "1814400"), ("GroupOrder[AlternatingGroup[2]]", "1"),
    ("GroupOrder[AlternatingGroup[1]]", "1"),
    ("GroupOrder[CyclicGroup[7]]", "7"), ("GroupOrder[DihedralGroup[10]]", "20"), ("GroupOrder[DihedralGroup[1]]", "2"),
    ("GroupOrder[AbelianGroup[{2, 3, 4}]]", "24"), ("GroupOrder[AbelianGroup[{5}]]", "5"),
    ("AbelianGroup[{2, 0}]", Raises()),
    ("GroupOrder[PermutationGroup[{Cycles[{{1, 3}, {2, 4}}], Cycles[{{1, 2}}]}]]", "8"),
    ("GroupOrder[PermutationGroup[{Cycles[{{1, 2, 3, 4, 5}}], Cycles[{{1, 2}}]}]]", "120"),
    ("GroupOrder[PermutationGroup[{}]]", "1"), ("GroupOrder[SymmetricGroup[n]]", "GroupOrder[SymmetricGroup[n]]"),
    ("GroupOrder[CyclicGroup[0]]", Raises()), ("SymmetricGroup[-1]", Raises()),
    # GroupElements: sorted by image lists, identity first (reference "Possible Issues")
    ("GroupElements[SymmetricGroup[3]]", S3),
    ("GroupElements[AlternatingGroup[4]]", A4),
    ("GroupElements[AlternatingGroup[2]]", "{Cycles[{}]}"), ("GroupElements[SymmetricGroup[1]]", "{Cycles[{}]}"),
    ("GroupElements[DihedralGroup[5]]", "{" + ", ".join(cycles_str(p) for p in dihedral(5)) + "}"),
    ("Length[GroupElements[DihedralGroup[6]]]", "12"),
    ("GroupElements[CyclicGroup[4]]", "{Cycles[{}], Cycles[{{1, 2, 3, 4}}], Cycles[{{1, 3}, {2, 4}}], Cycles[{{1, 4, 3, 2}}]}"),
    ("GroupElements[PermutationGroup[{Cycles[{{1, 2, 3, 4, 5}}], Cycles[{{1, 2}}]}]] === GroupElements[SymmetricGroup[5]]", True),
    ("GroupElements[PermutationGroup[{Cycles[{{1, 2, 3}}], Cycles[{{2, 3, 4}}], Cycles[{{3, 4, 5}}], Cycles[{{4, 5, 6}}]}]] "
     "=== GroupElements[AlternatingGroup[6]]", True),
    ("GroupElements[PermutationGroup[{Cycles[{{1, 9, 6}, {3, 7}}]}], {1, 2, 3}]",
     "{Cycles[{}], Cycles[{{3, 7}}], Cycles[{{1, 6, 9}}]}"),
    ("GroupElements[PermutationGroup[{Cycles[{{1, 9, 6}, {3, 7}}]}], {-1}]", "{Cycles[{{1, 9, 6}, {3, 7}}]}"),
    ("GroupElements[SymmetricGroup[3], {0}]", Raises()), ("GroupElements[SymmetricGroup[3], {7}]", Raises()),
    ("GroupElements[DihedralGroup[2]]", Raises()), ("GroupElements[AbelianGroup[{2, 2}]]", Raises()),
    # the D4 subgroup sits at S4 positions {1, 2, 7, 8, 17, 18, 23, 24} (GroupMultiplicationTable reference page)
    ("GroupElementPosition[SymmetricGroup[4], GroupElements[PermutationGroup[{Cycles[{{1, 3}, {2, 4}}], Cycles[{{1, 2}}]}]]]",
     "{1, 2, 7, 8, 17, 18, 23, 24}"),
    ("GroupElementPosition[SymmetricGroup[3], Cycles[{{1, 3}}]]", "6"),
    ("GroupElementPosition[AlternatingGroup[3], Cycles[{{1, 2}}]]", Raises()),
    # GroupMultiplicationTable[[i, j]] = position of PermutationProduct[g_i, g_j]
    ("GroupMultiplicationTable[SymmetricGroup[3]]", table_ref(S_n_lex(3))),
    ("GroupMultiplicationTable[DihedralGroup[4]]", table_ref(dihedral(4))),
    ("GroupMultiplicationTable[PermutationGroup[{}]]", "{{1}}"),
    ("With[{t = GroupMultiplicationTable[SymmetricGroup[3]]}, t == Transpose[t]]", False),
    ("With[{t = GroupMultiplicationTable[CyclicGroup[5]]}, t == Transpose[t]]", True),
    # GroupGenerators
    ("GroupGenerators[SymmetricGroup[5]]", "{Cycles[{{1, 2}}], Cycles[{{1, 2, 3, 4, 5}}]}"),
    ("GroupGenerators[CyclicGroup[4]]", "{Cycles[{{1, 2, 3, 4}}]}"),
    ("GroupGenerators[PermutationGroup[{Cycles[{{1, 2}}], Cycles[{}], {2, 3, 1}}]]",
     "{Cycles[{{1, 2}}], Cycles[{{1, 2, 3}}]}"),
    ("GroupGenerators[AlternatingGroup[5]]", Raises()),
    # GroupElementQ / GroupOrbits
    ("GroupElementQ[SymmetricGroup[3], Cycles[{{1, 2}}]]", True),
    ("GroupElementQ[AlternatingGroup[3], Cycles[{{1, 2}}]]", False),
    ("GroupElementQ[AlternatingGroup[4], Cycles[{{1, 2}, {3, 4}}]]", True),
    ("GroupElementQ[SymmetricGroup[3], Cycles[{{1, 4}}]]", False),
    ("GroupOrbits[PermutationGroup[{Cycles[{{1, 2}}], Cycles[{{4, 5, 6}}]}]]", "{{1, 2}, {4, 5, 6}}"),
    ("GroupOrbits[PermutationGroup[{Cycles[{{1, 3}}], Cycles[{{3, 5}}]}], {5, 2}]", "{{1, 3, 5}, {2}}"),
    ("GroupOrbits[DihedralGroup[6]]", "{{1, 2, 3, 4, 5, 6}}"),
]

COUNTING = [
    # FrobeniusNumber / FrobeniusSolve (brute-force references)
    ("FrobeniusNumber[{6, 9, 20}]", "43"), ("FrobeniusNumber[{3, 5}]", "7"),
    ("FrobeniusNumber[{12, 16, 20, 27}]", str(frobenius_brute([12, 16, 20, 27]))),
    ("FrobeniusNumber[{5, 8, 11}]", str(frobenius_brute([5, 8, 11]))),
    ("FrobeniusNumber[{2, 4}]", "Infinity"), ("FrobeniusNumber[{1, 7}]", "-1"), ("FrobeniusNumber[{0, 3}]", Raises()),
    ("FrobeniusSolve[{2, 3}, 10]", "{{2, 2}, {5, 0}}"),
    ("FrobeniusSolve[{12, 16, 20, 27}, 123]", plist_str(frobsolve_brute([12, 16, 20, 27], 123))),
    ("FrobeniusSolve[{3, 5, 7}, 30]", plist_str(frobsolve_brute([3, 5, 7], 30))),
    ("FrobeniusSolve[{2, 4}, 7]", "{}"), ("FrobeniusSolve[{2, 3}, 0]", "{{0, 0}}"),
    ("FrobeniusSolve[{2, 3}, 10, 1]", Raises()),
    # Hyperfactorial (OEIS A002109), PolygonalNumber, AlternatingFactorial (A005165), Fibonorial (A003266)
    ("Hyperfactorial[Range[0, 6]]", "{1, 1, 4, 108, 27648, 86400000, 4031078400000}"),
    ("Hyperfactorial[2.5]", float(__import__("mpmath").hyperfac(2.5))),
    ("Hyperfactorial[x]", "Hyperfactorial[x]"),
    ("PolygonalNumber[Range[6]]", "{1, 3, 6, 10, 15, 21}"),
    ("PolygonalNumber[5, Range[5]]", "{1, 5, 12, 22, 35}"), ("PolygonalNumber[4, 7]", "49"),
    ("PolygonalNumber[6, 3]", "15"), ("PolygonalNumber[r, n]", "PolygonalNumber[r, n]"),
    ("AlternatingFactorial[Range[0, 8]]", "{0, 1, 1, 5, 19, 101, 619, 4421, 35899}"),
    ("AlternatingFactorial[20]", str(sum((-1) ** (20 - k) * math.factorial(k) for k in range(1, 21)))),
    ("AlternatingFactorial[1/2]", Raises()),
    ("Fibonorial[Range[0, 9]]", "{1, 1, 1, 2, 6, 30, 240, 3120, 65520, 2227680}"),
    ("Fibonorial[25]", str(math.prod(sp.fibonacci(k) for k in range(1, 26)))), ("Fibonorial[n]", "Fibonorial[n]"),
    # corrections to existing combinatorics
    ("Binomial[10, 3.]", 120.0, 0), ("Binomial[2.5, 1.5]", 2.5, 1e-15),
    ("Binomial[10., 3] == 120", True),
    ("IntegerPartitions[10, 3, All, 2]", "{{10}, {9, 1}}"),
    ("IntegerPartitions[10, All, {1, 3, 5}, 3]", "{{5, 5}, {5, 3, 1, 1}, {5, 1, 1, 1, 1, 1}}"),
    ("IntegerPartitions[6, All, All, 0]", "{}"),
    ("Subsets[{a, b, c, d, e}, {3}, 5]", "{{a, b, c}, {a, b, d}, {a, b, e}, {a, c, d}, {a, c, e}}"),
    ("Subsets[{a, b, c, d, e}, {0, 5, 2}]", "Join[Subsets[{a, b, c, d, e}, {0}], Subsets[{a, b, c, d, e}, {2}], "
                                            "Subsets[{a, b, c, d, e}, {4}]]"),
    ("Subsets[Range[20], All, {69381}]", unrank_ref(20, 69381)),
    ("Subsets[Range[12], All, {1000}]", unrank_ref(12, 1000)),
    ("Subsets[{a, b, c, d}, All, {15, 1, -2}]", "Take[Subsets[{a, b, c, d}], {15, 1, -2}]"),
    ("Subsets[Range[3], All, 10]", "Subsets[Range[3]]"), ("Subsets[Range[4], All, 8]", "Take[Subsets[Range[4]], 8]"),
    ("Subsets[f[a, b, c]]", "{f[], f[a], f[b], f[c], f[a, b], f[a, c], f[b, c], f[a, b, c]}"),
    ("Subsets[Range[4], All, {17}]", Raises()),
    ("Tuples[{a, b}, {2, 1}]", "{{{a}, {a}}, {{a}, {b}}, {{b}, {a}}, {{b}, {b}}}"),
    ("Tuples[f[a, b], 2]", "{f[a, a], f[a, b], f[b, a], f[b, b]}"),
    ("Length[Tuples[{0, 1}, {2, 2}]]", "16"), ("Dimensions[Tuples[{0, 1}, {2, 3}]]", "{64, 2, 3}"),
]


def tt(f, n):
    """Truth table of a Python predicate in Tuples[{True, False}, n] order, as Wolfram source."""
    vals = [f(*row) for row in itertools.product([True, False], repeat=n)]
    return "{" + ", ".join("True" if v else "False" for v in vals) + "}"


def bf_index(f, n):
    """Wolfram BooleanFunction index: FromDigits[Boole[BooleanTable[f]], 2]."""
    k = 0
    for row in itertools.product([True, False], repeat=n):
        k = 2 * k + (1 if f(*row) else 0)
    return k


def equiv(src, ref):
    return (f"TautologyQ[Equivalent[{src}, {ref}]]", True)


MAJ = lambda *a: sum(a) > len(a) // 2
XOR3 = lambda a, b, c: (a ^ b) ^ c

LOGIC = [
    # Boole / Majority
    ("Boole[True]", "1"), ("Boole[False]", "0"), ("Boole[{True, False, x}]", "{1, 0, Boole[x]}"),
    ("Boole[x]", "Boole[x]"), ("Total[Boole[BooleanTable[Xor[a, b, c], {a, b, c}]]]", "4"),
    ("Majority[True, False, True]", True), ("Majority[False, False, True]", False),
    ("Majority[True, True, False, False, True]", True), ("Majority[a, b, c]", "Majority[a, b, c]"),
    ("Majority[True, False]", Raises()),
    ("BooleanTable[Majority[a, b, c], {a, b, c}]", tt(MAJ, 3)),
    ("BooleanTable[Majority[a, b, c, d, e], {a, b, c, d, e}]", tt(MAJ, 5)),
    # BooleanVariables (canonical order)
    ("BooleanVariables[(a || b) && !c]", "{a, b, c}"), ("BooleanVariables[Xor[z, y] || x]", "{x, y, z}"),
    ("BooleanVariables[True]", "{}"), ("BooleanVariables[(p && q) || x > 0]", "{p, q, x > 0}"),
    # BooleanTable: rows in Tuples[{True, False}, n] order (reference page)
    ("BooleanTable[p || q, {p, q}]", tt(lambda p, q: p or q, 2)),
    ("BooleanTable[p || q, {p}, {q}]", "{{True, True}, {True, False}}"),
    ("BooleanTable[And[a, b, c], {a}, {b}, {c}]", "{{{True, False}, {False, False}}, {{False, False}, {False, False}}}"),
    ("BooleanTable[And[c, Or[a, b]]]", tt(lambda a, b, c: c and (a or b), 3)),
    ("BooleanTable[And[c, Or[a, b]]] === BooleanTable[And[c, Or[a, b]], BooleanVariables[And[c, Or[a, b]]]]", True),
    ("BooleanTable[Xor[a, b], {a, b}]", "Xor @@@ Tuples[{True, False}, 2]"),
    ("BooleanTable[Xnor[a, b, c], {a, b, c}]", "Xnor @@@ Tuples[{True, False}, 3]"),
    ("BooleanTable[f[a, b], {a, b}]", "{f[True, True], f[True, False], f[False, True], f[False, False]}"),
    ("BooleanTable[{a, b} -> And[a, b], {a, b}]",
     "{{True, True} -> True, {True, False} -> False, {False, True} -> False, {False, False} -> False}"),
    ("BooleanTable[Xor[a, b, c], {a, b}]", "{c, !c, !c, c}"),
    ("BooleanTable[Implies[a, b], {b, a}]", tt(lambda b, a: (not a) or b, 2)),
    ("BooleanTable[And[#1, #2, #3] &]", tt(lambda a, b, c: a and b and c, 3)),
    ("Length[BooleanTable[Xor[a, b, c, d], {a, b, c, d}]]", "16"),
    # BooleanFunction: truth table digits IntegerDigits[k, 2, 2^n]
    ("FromDigits[Boole[BooleanTable[BooleanFunction[30, 3]]], 2]", "30"),
    ("BooleanTable[BooleanFunction[30, 3]]", tt(lambda a, b, c: bool((30 >> (7 - (4 * (not a) + 2 * (not b) + (not c)))) & 1), 3)),
    ("BooleanFunction[8, 2][True, True]", True), ("BooleanFunction[8, 2][True, False]", False),
    ("BooleanFunction[%d, 3][True, False, False]" % bf_index(XOR3, 3), True),
    ("BooleanFunction[%d, 3][True, True, False]" % bf_index(XOR3, 3), False),
    ("BooleanFunction[{True, False, False, False}]", "BooleanFunction[8, 2]"),
    ("BooleanFunction[BooleanTable[Implies[a, b], {a, b}]]", "BooleanFunction[%d, 2]" % bf_index(lambda a, b: (not a) or b, 2)),
    equiv("BooleanFunction[30, {p, q, r}]", "Xor[p, q || r]"),
    equiv("BooleanFunction[BooleanTable[Implies[a, b], {a, b}], {a, b}]", "Implies[a, b]"),
    ("BooleanFunction[16, 2]", Raises()), ("BooleanFunction[k, 2]", "BooleanFunction[k, 2]"),
    ("BooleanFunction[{True, False, True}]", Raises()),
    # BooleanCountingFunction: kmax (at most), {k} (exactly), {k1, k2}, {{kmin, kmax}}
    ("BooleanCountingFunction[{2}, 3][True, True, False]", True), ("BooleanCountingFunction[1, 3][True, True, False]", False),
    ("BooleanTable[BooleanCountingFunction[2, 3]]", tt(lambda a, b, c: a + b + c <= 2, 3)),
    ("BooleanTable[BooleanCountingFunction[{{1, 2}}, 3]]", tt(lambda a, b, c: 1 <= a + b + c <= 2, 3)),
    equiv("BooleanCountingFunction[{2, 3}, {a, b, c, d}]", "BooleanCountingFunction[{2, 3}, 4][a, b, c, d]"),
    ("SatisfiabilityCount[BooleanCountingFunction[{2, 3}, 4][a, b, c, d]]", "10"),
    ("SatisfiabilityCount[BooleanCountingFunction[3, 20]]", str(sum(math.comb(20, k) for k in range(4)))),
    # SatisfiableQ / TautologyQ / SatisfiabilityCount
    ("SatisfiableQ[a && !a]", False), ("SatisfiableQ[a || b, {a, b}]", True), ("SatisfiableQ[Xor[a, b, c]]", True),
    ("SatisfiableQ[!Implies[Implies[a, b] && a, b], {a, b}]", False),
    ("SatisfiableQ[And @@ Table[x[i] || !x[i + 1], {i, 1, 30}] && !x[1] && x[31]]", False),
    ("TautologyQ[a || !a]", True), ("TautologyQ[Implies[Implies[a, b] && a, b]]", True), ("TautologyQ[a || b]", False),
    ("TautologyQ[Equivalent[!(a && b), !a || !b]]", True),
    ("TautologyQ[Implies[x && (y || z), Equivalent[Xor[x, y, z], BooleanConvert[Xor[x, y, z], \"DNF\", x && (y || z)]]]]",
     True),
    ("SatisfiableQ[x > 0 && x < 0]", Raises()), ("TautologyQ[a || b, {a}]", Raises()),
    ("SatisfiabilityCount[Xor[a, b, c]]", "4"), ("SatisfiabilityCount[Xor[a, b, c], {a, b, c}]", "4"),
    ("SatisfiabilityCount[a, {a, b}]", "2"), ("SatisfiabilityCount[a && !a]", "0"),
    ("SatisfiabilityCount[Majority[a, b, c, d, e]]", "16"),
    ("SatisfiabilityCount[BooleanFunction[%d, 4]]" % (2 ** 15 + 2 ** 3 + 1), "3"),
    ("Count[BooleanTable[BooleanFunction[1234567, 5]], True] == SatisfiabilityCount[BooleanFunction[1234567, 5]]", True),
    # SatisfiabilityInstances: valid instances, {} when unsatisfiable
    ("SatisfiabilityInstances[Equivalent[a, b], {a, b}]", "{{False, False}}"),
    ("SatisfiabilityInstances[Equivalent[a, b], {a, b}, 4]", "{{False, False}, {True, True}}"),
    ("Sort[SatisfiabilityInstances[a || b, {a, b}, All]]", "Sort[{{True, True}, {True, False}, {False, True}}]"),
    ("Sort[SatisfiabilityInstances[Xor[a, b, c], {a, b, c}, 2^3]]",
     "Sort[Select[Tuples[{True, False}, 3], Xor @@ # &]]"),
    ("SatisfiabilityInstances[!Implies[Implies[a, b] && a, b], {a, b}]", "{}"),
    ("And @@ (BooleanFunction[30, 3] @@@ SatisfiabilityInstances[BooleanFunction[30, 3], 3])", True),
    ("Length[SatisfiabilityInstances[BooleanFunction[30, 3], 3]]", "3"),
    ("SatisfiabilityInstances[a || b]", Raises()),
    # BooleanConvert: equivalent output in the requested shape
    equiv("BooleanConvert[Equivalent[x, y, z]]", "Equivalent[x, y, z]"),
    ("BooleanConvert[Equivalent[x, y, z]]", Pred(lambda g: g.head == "Or" and all(t.head == "And" for t in g.args), "DNF")),
    ("BooleanConvert[Xor[x, y], \"CNF\"]", Pred(lambda g: g.head == "And" and all(t.head == "Or" for t in g.args), "CNF")),
    equiv("BooleanConvert[Xor[x, y], \"CNF\"]", "Xor[x, y]"),
    equiv("BooleanConvert[Xor[x, y], \"NAND\"]", "Xor[x, y]"),
    ("FreeQ[BooleanConvert[Xor[x, y], \"NAND\"], And | Or | Xor]", True),
    equiv("BooleanConvert[Xor[x, y], \"NOR\"]", "Xor[x, y]"), ("FreeQ[BooleanConvert[Xor[x, y], \"NOR\"], And | Or]", True),
    equiv("BooleanConvert[Majority[a, b, c], \"AND\"]", "Majority[a, b, c]"),
    ("FreeQ[BooleanConvert[Majority[a, b, c], \"AND\"], Or]", True),
    equiv("BooleanConvert[Majority[a, b, c], \"OR\"]", "Majority[a, b, c]"),
    ("FreeQ[BooleanConvert[Majority[a, b, c], \"OR\"], And]", True),
    ("BooleanConvert[x || y, \"ANF\"]", "Xor[x, y, x && y]"), ("BooleanConvert[!x, \"ANF\"]", "Xor[True, x]"),
    ("BooleanConvert[Xor[a, b, c], \"ANF\"]", "Xor[a, b, c]"), ("BooleanConvert[a && !a, \"ANF\"]", False),
    equiv("BooleanConvert[Equivalent[a, b, c], \"ESOP\"]", "Equivalent[a, b, c]"),
    equiv("BooleanConvert[BooleanFunction[228, 3][a, b, c]]", "BooleanFunction[228, 3][a, b, c]"),
    ("BooleanConvert[Xor[x, y, z], \"DNF\", x && (y || z)]", Pred(lambda g: g.head in ("And", "Or"), "DNF")),
    ("BooleanConvert[a || !a]", True), ("BooleanConvert[x || y, \"ITE\"]", Raises()),
    # LogicalExpand
    ("LogicalExpand[(a || b) && c]", "(a && c) || (b && c)"), ("LogicalExpand[!(a && b)]", "!a || !b"),
    equiv("LogicalExpand[Xor[a, b]]", "Xor[a, b]"), equiv("LogicalExpand[Implies[a, b && c]]", "Implies[a, b && c]"),
    ("LogicalExpand[{a, b} == {c, d}]", "a == c && b == d"), ("LogicalExpand[{a, b} != {c, d}]", "a != c || b != d"),
    ("LogicalExpand[(x > 0 || y) && z]", "(x > 0 && z) || (y && z)"), ("LogicalExpand[a && !a]", False),
    # ConditionalExpression
    ("ConditionalExpression[x^2, True]", "x^2"), ("ConditionalExpression[x^2, False]", "Undefined"),
    ("ConditionalExpression[ConditionalExpression[x, a], b]", "ConditionalExpression[x, a && b]"),
    ("ConditionalExpression[x, x > 0]", "ConditionalExpression[x, x > 0]"),
    ("ConditionalExpression[1/n, n > 0] /. n -> 2", "1/2"),
    # ForAll / Exists / Resolve over Booleans
    ("ForAll[x, True]", True), ("ForAll[x, y]", "y"), ("Exists[x, False]", False), ("ForAll[x, x > 0]", "ForAll[x, x > 0]"),
    ("Exists[{x, y}, x && y]", "Exists[{x, y}, x && y]"), ("ForAll[1, x]", Raises()),
    ("Resolve[Exists[x, x || y]]", True), ("Resolve[ForAll[x, x || y]]", "y"), ("Resolve[Exists[x, x && !x]]", False),
    ("Resolve[ForAll[{x, y}, Implies[x && y, z]]]", "z"), equiv("Resolve[Exists[x, Xor[x, y] && z]]", "z"),
    ("Resolve[ForAll[x, x, y]]", "y"), ("Resolve[Exists[x, x, y]]", "y"),
    ("Resolve[ForAll[x, Exists[y, Xor[x, y]]]]", True), ("Resolve[Exists[y, ForAll[x, Xor[x, y]]]]", False),
    ("Resolve[ForAll[x, x^2 >= 0], Reals]", Raises(NotImplementedError)),
    ("Resolve[Exists[x, x > 0 && y]]", Raises(NotImplementedError)), ("Resolve[a && b]", "a && b"),
    # corrections: partial evaluation of connectives (Wolfram simplifies True/False arguments away)
    ("Xor[True, True, c]", "c"), ("Xor[True, c]", "!c"), ("Xor[False, c, d]", "Xor[c, d]"), ("Xor[True, False]", True),
    ("Not[Not[a]]", Pred(lambda g: str(g) == "a", "a")), ("!!(a && b)", "a && b"), ("Xnor[True, a]", "a"), ("Xnor[False, a]", "!a"), ("Xnor[True, True]", True),
    ("Implies[True, b]", "b"), ("Implies[False, b]", True), ("Implies[a, True]", True), ("Implies[a, False]", "!a"),
    ("Implies[a, b]", "Implies[a, b]"),
    ("Nand[True, x]", "!x"), ("Nand[False, x]", True), ("Nand[True, x, y]", "Nand[x, y]"),
    ("Nor[False, x]", "!x"), ("Nor[True, x]", False), ("Nor[False, x, y]", "Nor[x, y]"),
    ("Equivalent[True, a, b]", "a && b"), ("Equivalent[False, a]", "!a"), ("Equivalent[True, False, a]", False),
    ("Equivalent[False, a, b]", "!a && !b"), ("Equivalent[a, b]", "Equivalent[a, b]"),
    # Conjunction / Disjunction
    ("Conjunction[a || b, {a}]", "b"), ("Disjunction[a && b, {a}]", "b"), ("Conjunction[Xor[a, b], {a, b}]", False),
    ("Disjunction[Xor[a, b], {a, b}]", True), equiv("Conjunction[Majority[a, b, c], {c}]", "a && b"),
    equiv("Disjunction[Majority[a, b, c], {c}]", "a || b"),
]

# ─────────────────────────────── discrete mathematics (wl_discrete) ───────────────────────────────

def L_(xs):
    return "{" + ", ".join(L_(x) if isinstance(x, (list, tuple)) else fr_str(x) for x in xs) + "}"


def fr_str(x):
    if isinstance(x, Fraction):
        return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    if isinstance(x, bool):
        return "True" if x else "False"
    return str(x)


def farey_ref(n):
    return sorted({Fraction(a, b) for b in range(1, n + 1) for a in range(0, b + 1)})


def cf_ref(fr):
    out = []
    while True:
        a = math.floor(fr)
        out.append(a)
        fr -= a
        if fr == 0:
            return out
        fr = 1 / fr


def eca_ref(rule, cells, t):
    """Independent elementary CA on a zero background, width len(cells) + 2t (Wolfram's default region)."""
    w = len(cells) + 2 * t
    row = [0] * t + list(cells) + [0] * t
    rows = [row]
    for _ in range(t):
        prev = [0] + rows[-1] + [0]
        rows.append([(rule >> (4 * prev[i - 1] + 2 * prev[i] + prev[i + 1])) & 1 for i in range(1, w + 1)])
    return rows


def eca_cyclic(rule, cells, t):
    rows = [list(cells)]
    n = len(cells)
    for _ in range(t):
        r = rows[-1]
        rows.append([(rule >> (4 * r[i - 1] + 2 * r[i] + r[(i + 1) % n])) & 1 for i in range(n)])
    return rows


def squares_r_ref(d, n):
    m = math.isqrt(max(n, 0))
    return sum(1 for v in itertools.product(range(-m, m + 1), repeat=d) if sum(x * x for x in v) == n)


def powers_rep_ref(n, k, p):
    m = 0
    while (m + 1) ** p <= n:
        m += 1
    return [list(c) for c in itertools.combinations_with_replacement(range(m + 1), k) if sum(x ** p for x in c) == n]


def prim_roots_ref(n):
    phi = sp.totient(n)
    return [g for g in range(1, n) if math.gcd(g, n) == 1 and sp.n_order(g, n) == phi]


def ns_from(f, xs):
    return "{" + ", ".join(fr_str(f(x)) for x in xs) + "}"


THUE_MORSE = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1]  # A010060
RUDIN_SHAPIRO = [1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, 1,
                 -1, 1]  # A020985
PERFECT = [6, 28, 496, 8128, 33550336, 8589869056, 137438691328, 2305843008139952128]  # A000396
MERSENNE = [2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279]  # A000043

DISCRETE = [
    # Interval canonical form, union, intersection, membership
    ("Interval[{3, 1}, {2, 5}, {7, 8}]", "Interval[{1, 5}, {7, 8}]"), ("Interval[{1, 2}, {2, 3}]", "Interval[{1, 3}]"),
    ("Interval[{0, Infinity}, {-1, 1}]", "Interval[{-1, Infinity}]"), ("Interval[{1, 2}]", "Interval[{1, 2}]"),
    ("Interval[{1, 2, 3}]", Raises()), ("Interval[{a, 1}]", "Interval[{a, 1}]"),
    ("IntervalUnion[Interval[{1, 2}], Interval[{2, 4}], Interval[{6, Infinity}]]", "Interval[{1, 4}, {6, Infinity}]"),
    ("IntervalUnion[Interval[{1, 2}], Interval[{3, 4}]]", "Interval[{1, 2}, {3, 4}]"), ("IntervalUnion[]", "Interval[]"),
    ("IntervalUnion[Interval[{0, 1/2}], Interval[{0.25, 1}]]", "Interval[{0, 1}]"),
    ("IntervalIntersection[Interval[{1, 3}, {5, 8}], Interval[{2, 6}]]", "Interval[{2, 3}, {5, 6}]"),
    ("IntervalIntersection[Interval[{1, 2}], Interval[{3, 4}]]", "Interval[]"),
    ("IntervalIntersection[Interval[{1, 2}], Interval[{2, 4}]]", "Interval[{2, 2}]"),
    ("IntervalIntersection[Interval[{-Infinity, 0}], Interval[{-3, 5}], Interval[{-1, 1}]]", "Interval[{-1, 0}]"),
    ("IntervalMemberQ[Interval[{1, 3}], 2]", True), ("IntervalMemberQ[Interval[{1, 3}], Pi]", False),
    ("IntervalMemberQ[Interval[{1, 3}], E]", True), ("IntervalMemberQ[Interval[{1, 3}], Interval[{1.5, 2}]]", True),
    ("IntervalMemberQ[Interval[{1, 3}], Interval[{2, 4}]]", False), ("IntervalMemberQ[Interval[{1, 3}], x]", False),
    ("IntervalMemberQ[Interval[{1, 2}, {4, 5}], 3]", False),
    # discrete calculus (forward differences)
    ("DifferenceDelta[i^3, i]", "1 + 3 i + 3 i^2"), ("DifferenceDelta[i^3, {i, 2}]", "6 + 6 i"),
    ("DifferenceDelta[i^3, {i, 3}]", "6"), ("DifferenceDelta[i^3, {i, 4}]", "0"), ("DifferenceDelta[2^i, i]", "2^i"),
    ("DifferenceDelta[i!, i]", "i i!"), ("DifferenceDelta[Binomial[i, 2], i]", "i"),
    ("DifferenceDelta[f[i], {i, 1, h}]", "f[h + i] - f[i]"), ("DifferenceDelta[i j, i, j]", "1"),
    ("DifferenceDelta[i^2, {i, 1, 1/2}]", "1/4 + i"), ("DifferenceDelta[x, {1, 2}]", Raises()),
    ("DiscreteShift[f[i], i]", "f[1 + i]"), ("DiscreteShift[f[i], {i, 2}]", "f[2 + i]"),
    ("DiscreteShift[f[i], {i, 3, h}]", "f[i + 3 h]"), ("DiscreteShift[i^2, i]", "(1 + i)^2"),
    ("DiscreteRatio[i!, i]", "1 + i"), ("DiscreteRatio[2^i i, i]", "2 (1 + i)/i"), ("DiscreteRatio[3^i, {i, 2}]", "1"),
    ("DiscreteRatio[Binomial[n, i], i]", "(n - i)/(1 + i)"),
    # linear recurrences
    ("LinearRecurrence[{1, 1}, {1, 1}, 10]", ns_from(sp.fibonacci, range(1, 11))),
    ("LinearRecurrence[{1, 1}, {1, 1}, {10}]", "55"), ("LinearRecurrence[{1, 1}, {1, 1}, {5, 8}]", "{5, 8, 13, 21}"),
    ("LinearRecurrence[{1, 1}, {0, 1}, {10000}]", str(sp.fibonacci(9999))),
    ("LinearRecurrence[{1, 1}, {0, 1}, {9000, 9002}]", ns_from(lambda k: sp.fibonacci(k - 1), range(9000, 9003))),
    ("LinearRecurrence[{2, -1}, {1, 2}, 6]", "{1, 2, 3, 4, 5, 6}"),
    ("LinearRecurrence[{0, 1, 1}, {3, 0, 2}, 10]", "{3, 0, 2, 3, 2, 5, 5, 7, 10, 12}"),  # Perrin, A001608
    ("LinearRecurrence[{a, b}, {1, 1}, 4]", "{1, 1, a + b, b + a (a + b)}"),
    ("LinearRecurrence[{1/2, 1/3}, {1, 1}, 4]", "{1, 1, 5/6, 3/4}"), ("LinearRecurrence[{1, 1}, {1, 1}, 0]", "{}"),
    ("LinearRecurrence[{2, -1}, {1, 2, 3}, 5]", Raises()), ("LinearRecurrence[{1, 1}, {1, 1}, {0, 3}]", Raises()),
    ("FindLinearRecurrence[{1, 1, 2, 3, 5, 8, 13}]", "{1, 1}"),
    ("FindLinearRecurrence[" + ns_from(lambda k: k * k, range(1, 11)) + "]", "{3, -3, 1}"),
    ("FindLinearRecurrence[{1, 2, 4, 8, 16}]", "{2}"),
    ("FindLinearRecurrence[LinearRecurrence[{0, 1, 1}, {3, 0, 2}, 20]]", "{0, 1, 1}"),
    ("FindLinearRecurrence[{1, 2, 3}]", "FindLinearRecurrence[{1, 2, 3}]"),
    ("FindLinearRecurrence[" + ns_from(lambda k: k ** 3, range(1, 13)) + ", 3]",
     "FindLinearRecurrence[" + ns_from(lambda k: k ** 3, range(1, 13)) + ", 3]"),
    ("FindLinearRecurrence[{0, 0, 0}]", Raises()), ("FindLinearRecurrence[{1., 2., 4., 8.}]", Raises()),
    ("RecurrenceTable[{a[n + 1] == 3 a[n], a[1] == 7}, a, {n, 1, 5}]", "{7, 21, 63, 189, 567}"),
    ("RecurrenceTable[{a[n] == a[n - 1] + a[n - 2], a[1] == 1, a[2] == 1}, a[n], {n, 10}]",
     ns_from(sp.fibonacci, range(1, 11))),
    ("RecurrenceTable[{a[n + 1] == a[n] + n, a[0] == 0}, a[n]^2, {n, 0, 4}]", "{0, 0, 1, 9, 36}"),
    ("RecurrenceTable[{x[n + 1] == y[n], y[n + 1] == x[n] + y[n], x[1] == 0, y[1] == 1}, {x, y}, {n, 1, 5}]",
     "{{0, 1}, {1, 1}, {1, 2}, {2, 3}, {3, 5}}"),
    ("RecurrenceTable[{a[n + 1] == n a[n], a[1] == 1}, a, {n, 1, 7}]", ns_from(math.factorial, range(0, 7))),
    ("RecurrenceTable[{a[n + 1] == a[n]/2, a[1] == 1.}, a, {n, 1, 4}]", "{1., 0.5, 0.25, 0.125}"),
    ("RecurrenceTable[{a[n + 2] == a[n + 1] + a[n], a[1] == 1}, a, {n, 1, 5}]", Raises()),
    ("RecurrenceTable[{a[n]^2 == a[n - 1], a[1] == 4}, a, {n, 1, 3}]", Raises()),
    # continued fractions
    ("ContinuedFraction[415/93]", L_(cf_ref(Fraction(415, 93)))), ("ContinuedFraction[-7/3]", L_(cf_ref(Fraction(-7, 3)))),
    ("ContinuedFraction[3]", "{3}"), ("ContinuedFraction[Sqrt[13]]", "{3, {1, 1, 1, 1, 6}}"),
    ("ContinuedFraction[Sqrt[2]]", "{1, {2}}"), ("ContinuedFraction[GoldenRatio]", "{{1}}"),
    ("ContinuedFraction[(3 + Sqrt[7])/5]", "{1, 7, {1, 2, 1, 8, 13, 8}}"),
    ("ContinuedFraction[Pi, 10]", "{3, 7, 15, 1, 292, 1, 1, 1, 2, 1}"),  # A001203
    ("ContinuedFraction[E, 12]", "{2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8}"),  # A003417
    ("ContinuedFraction[2^(1/3), 8]", "{1, 3, 1, 5, 1, 1, 4, 1}"),  # A002945
    ("ContinuedFraction[Sqrt[7], 9]", "{2, 1, 1, 1, 4, 1, 1, 1, 4}"), ("ContinuedFraction[415/93, 2]", "{4, 2}"),
    ("ContinuedFraction[0.1]", "{0, 10}"), ("ContinuedFraction[3.25]", "{3, 4}"), ("ContinuedFraction[-2.5]", "{-3, 2}"),
    ("Take[ContinuedFraction[N[Pi]], 13]", "{3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14}"),
    ("Take[ContinuedFraction[N[Pi, 30]], 26]", "ContinuedFraction[Pi, 26]"),
    ("ContinuedFraction[Pi]", Raises()), ("ContinuedFraction[I]", Raises()), ("ContinuedFraction[x]", "ContinuedFraction[x]"),
    ("FromContinuedFraction[{3, 1, 1, 1, 1, 6}]", "119/33"), ("FromContinuedFraction[{3, {1, 1, 1, 1, 6}}]", "Sqrt[13]"),
    ("FromContinuedFraction[{1, {2}}]", "Sqrt[2]"), ("FromContinuedFraction[{a, b, c}]", "a + 1/(b + 1/c)"),
    ("FromContinuedFraction[ContinuedFraction[(3 + Sqrt[7])/5]]", "(3 + Sqrt[7])/5"),
    ("FromContinuedFraction[ContinuedFraction[415/93]]", "415/93"), ("FromContinuedFraction[{{1}}]", "GoldenRatio"),
    ("FromContinuedFraction[{}]", Raises()), ("FromContinuedFraction[{1, {}}]", Raises()),
    ("Convergents[{1, 2, 2, 2}]", "{1, 3/2, 7/5, 17/12}"), ("Convergents[Pi, 4]", "{3, 22/7, 333/106, 355/113}"),
    ("Convergents[{a, b, c}]", "{a, a + 1/b, a + 1/(b + 1/c)}"), ("Convergents[7/3]", "{2, 7/3}"),
    ("Convergents[Sqrt[2], 5]", "{1, 3/2, 7/5, 17/12, 41/29}"), ("Convergents[Sqrt[2]]", Raises()),
    ("QuadraticIrrationalQ[Sqrt[2]]", True), ("QuadraticIrrationalQ[(1 + Sqrt[5])/2]", True),
    ("QuadraticIrrationalQ[GoldenRatio]", True), ("QuadraticIrrationalQ[2^(1/3)]", False),
    ("QuadraticIrrationalQ[Pi]", False), ("QuadraticIrrationalQ[3/4]", False), ("QuadraticIrrationalQ[1.414]", False),
    ("QuadraticIrrationalQ[x]", False), ("QuadraticIrrationalQ[Sqrt[2] + Sqrt[3]]", False),
    ("QuadraticIrrationalQ[Sqrt[8] - 1/3]", True),
    ("FareySequence[5]", L_(farey_ref(5))), ("FareySequence[1]", "{0, 1}"), ("Length[FareySequence[40]]", str(len(farey_ref(40)))),
    ("FareySequence[7, 4]", fr_str(farey_ref(7)[3])), ("FareySequence[5, -1]", "1"), ("FareySequence[0]", Raises()),
    ("FareySequence[5, 20]", Raises()),
    # bitwise operations (Python's infinite two's complement as the reference)
    ("BitAnd[12, 10]", str(12 & 10)), ("BitAnd[]", "-1"), ("BitAnd[-1, 5]", "5"), ("BitAnd[-12, 7, 255]", str(-12 & 7 & 255)),
    ("BitAnd[{1, 2, 3}, 1]", "{1, 0, 1}"), ("BitAnd[x, 1]", "BitAnd[x, 1]"), ("BitAnd[1.5, 1]", Raises()),
    ("BitOr[12, 10]", str(12 | 10)), ("BitOr[]", "0"), ("BitOr[-8, 3]", str(-8 | 3)),
    ("BitXor[12, 10, 3]", str(12 ^ 10 ^ 3)), ("BitXor[]", "0"), ("BitXor[-5, 9]", str(-5 ^ 9)),
    ("BitNot[5]", "-6"), ("BitNot[-1]", "0"), ("BitNot[{0, 7}]", "{-1, -8}"),
    ("BitShiftLeft[3, 4]", "48"), ("BitShiftLeft[5]", "10"), ("BitShiftLeft[48, -4]", "3"), ("BitShiftLeft[-3, 2]", "-12"),
    ("BitShiftRight[-7, 1]", "IntegerPart[-7/2]"), ("BitShiftRight[7, -2]", "28"), ("BitShiftRight[1000, 3]", str(1000 >> 3)),
    ("BitShiftRight[-1, 5]", "0"),
    ("BitLength[255]", "8"), ("BitLength[256]", "9"), ("BitLength[-256]", "8"), ("BitLength[0]", "0"),
    ("BitLength[2^1000]", "1001"),
    ("BitGet[-1, 100]", "1"), ("BitGet[10, {0, 1, 2, 3}]", "{0, 1, 0, 1}"), ("BitGet[5, -1]", Raises()),
    ("BitSet[0, 5]", "32"), ("BitSet[-33, 5]", str(-33 | 32)), ("BitSet[5, {1, 3}]", "{7, 13}"),
    ("BitClear[-1, 0]", "-2"), ("BitClear[15, 2]", "11"), ("BitClear[8, 1]", "8"),
    ("BitFlip[5, 1]", "7"), ("BitFlip[7, 1]", "5"), ("BitFlip[-1, 3]", str(-1 ^ 8)),
    ("IntegerExponent[1000]", "3"), ("IntegerExponent[0, 2]", "Infinity"), ("IntegerExponent[-48, 2]", "4"),
    ("IntegerExponent[2^10 3^5, 6]", "5"), ("IntegerExponent[7, 1]", Raises()),
    # cellular automata and substitution systems
    ("CellularAutomaton[30, {{1}, 0}, 3]", L_(eca_ref(30, [1], 3))), ("CellularAutomaton[110, {{1}, 0}, 6]", L_(eca_ref(110, [1], 6))),
    ("CellularAutomaton[90, {{1}, 0}, 7]", L_(eca_ref(90, [1], 7))),
    ("CellularAutomaton[30, {0, 0, 1, 0, 0}, 2]", L_(eca_cyclic(30, [0, 0, 1, 0, 0], 2))),
    ("CellularAutomaton[90, {0, 0, 0, 1, 0, 0, 0}]", L_(eca_cyclic(90, [0, 0, 0, 1, 0, 0, 0], 1)[1])),
    ("CellularAutomaton[{30, 2, 1}, {{1}, 0}, 2]", L_(eca_ref(30, [1], 2))),
    ("CellularAutomaton[{{1, 1, 1} -> 0, {1, 1, 0} -> 0, {1, 0, 1} -> 0, {1, 0, 0} -> 1, {0, 1, 1} -> 1, {0, 1, 0} -> 1, "
     "{0, 0, 1} -> 1, {0, 0, 0} -> 0}, {{1}, 0}, 2]", L_(eca_ref(30, [1], 2))),
    ("CellularAutomaton[1, {{1}, 0}, 2]", "{{0, 0, 1, 0, 0}, {1, 0, 0, 0, 1}, {0, 0, 1, 0, 0}}"),
    ("CellularAutomaton[256, {0, 1, 0}, 1]", Raises()), ("CellularAutomaton[{1599, {3, 1}}, {0, 1, 0}, 1]", Raises()),
    ("SubstitutionSystem[{\"A\" -> \"AB\", \"B\" -> \"A\"}, \"A\", 4]", "{\"A\", \"AB\", \"ABA\", \"ABAAB\", \"ABAABABA\"}"),
    ("SubstitutionSystem[{0 -> {0, 1}, 1 -> {1, 0}}, {0}, 4]", "{{0}, {0, 1}, {0, 1, 1, 0}, {0, 1, 1, 0, 1, 0, 0, 1}, "
     + L_(THUE_MORSE[:16]) + "}"),
    ("SubstitutionSystem[{\"A\" -> \"AB\", \"B\" -> \"A\"}, \"A\"]", "\"AB\""),
    ("Map[StringLength, SubstitutionSystem[{\"A\" -> \"AB\", \"B\" -> \"A\"}, \"A\", 10]]",
     ns_from(sp.fibonacci, range(2, 13))),
    ("SubstitutionSystem[{\"AB\" -> \"B\"}, \"AB\", 1]", Raises()),
    ("ThueMorse[Range[0, 31]]", L_(THUE_MORSE)), ("ThueMorse[2^40]", "1"), ("ThueMorse[-1]", Raises()),
    ("RudinShapiro[Range[0, 31]]", L_(RUDIN_SHAPIRO)), ("RudinShapiro[0]", "1"), ("RudinShapiro[-3]", Raises()),
    # integer predicates and representations
    ("SquareFreeQ[30]", True), ("SquareFreeQ[12]", False), ("SquareFreeQ[0]", False), ("SquareFreeQ[1]", True),
    ("SquareFreeQ[-30]", True), ("Select[Range[30], SquareFreeQ]",
                                 ns_from(lambda k: k, [k for k in range(1, 31) if all(e == 1 for e in sp.factorint(k).values())])),
    ("SquareFreeQ[x^2 - 1]", True), ("SquareFreeQ[(x - 1)^2 (x + 2)]", False), ("SquareFreeQ[2 x + 2]", Raises()),
    ("SquareFreeQ[2.5]", False),
    ("Map[PerfectNumberQ, {6, 28, 496, 8128, 12, 0, -6, 7.}]", "{True, True, True, True, False, False, False, False}"), ("PerfectNumberQ[{6}]", False),
    ("Select[Range[10000], PerfectNumberQ]", "{6, 28, 496, 8128}"), ("PerfectNumberQ[" + str(PERFECT[-1]) + "]", True),
    ("PerfectNumberQ[2^60 (2^61 - 1) + 2]", False),
    ("PerfectNumber[Range[8]]", L_(PERFECT)), ("PerfectNumber[48] == 2^57885160 (2^57885161 - 1)", True),
    ("PerfectNumber[49]", Raises()), ("PerfectNumber[0]", Raises()),
    ("MersennePrimeExponent[Range[15]]", L_(MERSENNE)), ("MersennePrimeExponent[48]", "57885161"),
    ("MersennePrimeExponent[49]", Raises()),
    ("MersennePrimeExponentQ[{2, 11, 31, 57885161, 57885163}]", "{True, False, True, True, False}"),
    ("Select[Range[1300], MersennePrimeExponentQ]", L_(MERSENNE)), ("MersennePrimeExponentQ[57885167]", Raises()),
    ("MersennePrimeExponentQ[x]", False),
    ("CompositeQ[{4, 7, 1, 0, -9, 2.5}]", "{True, False, False, False, True, False}"),
    ("Select[Range[30], CompositeQ]", ns_from(lambda k: k, [k for k in range(2, 31) if not sp.isprime(k)])),
    ("CompositeQ[2^61 - 1]", False), ("CompositeQ[2^67 - 1]", True), ("CompositeQ[x]", False),
    ("PrimeNu[{12, 1, -30, 2^10}]", "{2, 0, 3, 1}"), ("PrimeNu[Range[20]]", ns_from(lambda k: len(sp.factorint(k)), range(1, 21))),
    ("PrimeNu[0]", Raises()), ("PrimeNu[x]", "PrimeNu[x]"),
    ("PrimeOmega[{12, 1, -30, 2^10}]", "{3, 0, 3, 10}"),
    ("PrimeOmega[Range[20]]", ns_from(lambda k: sum(sp.factorint(k).values()), range(1, 21))), ("PrimeOmega[0]", Raises()),
    ("Divisible[10, 5]", True), ("Divisible[10, 3]", False), ("Divisible[2 + 4 I, 2]", True), ("Divisible[3/2, 1/2]", True),
    ("Divisible[5 + 5 I, 1 + 2 I]", Pred(lambda g: g is ((5 + 5j) / (1 + 2j) == complex(round(((5 + 5j) / (1 + 2j)).real),
                                                                                         round(((5 + 5j) / (1 + 2j)).imag))),
                                        "Gaussian divisibility")),
    ("Divisible[{6, 7, 8}, 2]", "{True, False, True}"), ("Divisible[x, 2]", "Divisible[x, 2]"), ("Divisible[3, 0]", Raises()),
    ("Divisible[3., 1]", Raises()),
    ("CoprimeQ[3, 4]", True), ("CoprimeQ[6, 9]", False), ("CoprimeQ[2, 3, 4]", False), ("CoprimeQ[3, 5, 7]", True),
    ("CoprimeQ[{3, 6}, 5]", "{True, True}"), ("CoprimeQ[1, 0]", True), ("CoprimeQ[4]", Raises()),
    ("NumberExpand[1234]", "{1000, 200, 30, 4}"), ("NumberExpand[-12]", "{-10, -2}"),
    ("NumberExpand[255, 2]", "{128, 64, 32, 16, 8, 4, 2, 1}"), ("NumberExpand[12, 10, 4]", "{0, 0, 10, 2}"),
    ("NumberExpand[0]", "{0}"), ("Total[NumberExpand[987654321, 7]]", "987654321"), ("NumberExpand[1.5]", Raises()),
    ("NumberDigit[1234, 2]", "2"), ("NumberDigit[Pi, -5]", "9"), ("NumberDigit[1/7, -3]", "2"), ("NumberDigit[255, 4, 2]", "1"),
    ("NumberDigit[E, -10]", str(int(str(sp.N(sp.E, 20))[2 + 9]))), ("NumberDigit[-1234, 0]", "4"),
    ("NumberDigit[1234, 7]", "0"),
    ("SquaresR[2, 5]", str(squares_r_ref(2, 5))), ("SquaresR[4, 10]", str(squares_r_ref(4, 10))),
    ("SquaresR[3, Range[0, 12]]", ns_from(lambda k: squares_r_ref(3, k), range(13))), ("SquaresR[2, -1]", "0"),
    ("SquaresR[8, 3]", str(squares_r_ref(8, 3))), ("SquaresR[0, 4]", Raises()),
    ("PowersRepresentations[50, 2, 2]", L_(powers_rep_ref(50, 2, 2))),
    ("PowersRepresentations[1729, 2, 3]", L_(powers_rep_ref(1729, 2, 3))), ("PowersRepresentations[0, 2, 2]", "{{0, 0}}"),
    ("PowersRepresentations[100, 3, 2]", L_(powers_rep_ref(100, 3, 2))), ("PowersRepresentations[3, 1, 2]", "{}"),
    ("PrimitiveRootList[7]", L_(prim_roots_ref(7))), ("PrimitiveRootList[8]", "{}"), ("PrimitiveRootList[2]", "{1}"),
    ("PrimitiveRootList[18]", L_(prim_roots_ref(18))), ("PrimitiveRootList[41]", L_(prim_roots_ref(41))),
    ("PrimitiveRootList[1]", Raises()),
    ("PowerModList[2, 1/2, 7]", L_([x for x in range(7) if x * x % 7 == 2])), ("PowerModList[3, 2, 7]", "{2}"),
    ("PowerModList[2, -1, 4]", "{}"), ("PowerModList[4, 1/2, 15]", L_([x for x in range(15) if x * x % 15 == 4])),
    ("PowerModList[3, 2/3, 13]", L_([x for x in range(13) if pow(x, 3, 13) == 9])), ("PowerModList[3, -1, 7]", "{5}"),
    ("PowerModList[2, 1/2, 0]", Raises()),
    ("MantissaExponent[3.4]", Pred(lambda g: abs(g.args[0] - 0.34) < 1e-15 and g.args[1] == 1, "{0.34, 1}")),
    ("MantissaExponent[125]", "{1/8, 3}"), ("MantissaExponent[1/8, 2]", "{1/2, -2}"), ("MantissaExponent[Pi]", "{Pi/10, 1}"),
    ("MantissaExponent[0]", "{0, 0}"), ("MantissaExponent[-1000]", "{-1/10, 4}"),
    ("MantissaExponent[0.001]", Pred(lambda g: abs(g.args[0] - 0.1) < 1e-15 and g.args[1] == -2, "{0.1, -2}")),
    ("MantissaExponent[1000.]", Pred(lambda g: abs(g.args[0] - 0.1) < 1e-15 and g.args[1] == 4, "{0.1, 4}")),
    ("MantissaExponent[Sqrt[2] 10^20]", "{Sqrt[2]/10, 21}"), ("MantissaExponent[I]", Raises()),
    ("MantissaExponent[x]", "MantissaExponent[x]"), ("MantissaExponent[7, 1]", Raises()),
]


CASES = PERMUTATIONS + GROUPS + COUNTING + LOGIC + DISCRETE


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_batch6(case):
    check(case)
