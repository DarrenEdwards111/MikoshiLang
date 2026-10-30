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

CASES = PERMUTATIONS + GROUPS + COUNTING


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_batch6(case):
    check(case)
