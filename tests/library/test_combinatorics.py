import itertools

import mpmath as mp
import pytest
import sympy as sp

from _harness import Raises, Pred, check, ids

CASES = [
    ("Binomial[5, 2]", "10"), ("Binomial[10, 0]", "1"), ("Binomial[-3, 2]", "6"), ("Binomial[-3, -5]", "6"),
    ("Binomial[-3, -1]", "0"), ("Binomial[2, 5]", "0"), ("Binomial[4.5, 2]", mp.binomial(4.5, 2)), ("Binomial[n, 2]", "Binomial[n, 2]"),
    ("Factorial[5]", "120"), ("Factorial[0]", "1"), ("Factorial[1/2]", "Sqrt[Pi]/2"), ("Factorial[2.5]", mp.factorial(2.5)),
    ("Factorial[-1]", "ComplexInfinity"),
    ("DoubleFactorial[7]", "105"), ("DoubleFactorial[8]", "384"), ("DoubleFactorial[0]", "1"), ("DoubleFactorial[-1]", "1"),
    ("Factorial2[7]", "105"), ("Factorial2[10]", "3840"), ("Factorial2[-1]", "1"),
    ("HarmonicNumber[4]", "25/12"), ("HarmonicNumber[0]", "0"), ("HarmonicNumber[10, 2]", str(sp.harmonic(10, 2))),
    ("HarmonicNumber[2.5]", mp.harmonic(2.5)),
    ("Derangements[5]", "44"), ("Derangements[0]", "1"), ("Derangements[1]", "0"), ("Derangements[10]", str(sp.subfactorial(10))),
    ("Subfactorial[5]", "44"), ("Subfactorial[4]", "9"), ("Subfactorial[0]", "1"),
    ("PartitionCount[5]", "7"), ("PartitionCount[0]", "1"), ("PartitionCount[100]", "190569292"),
    ("PartitionsP[5]", "7"), ("PartitionsP[10]", "42"), ("PartitionsP[-1]", "0"),
    ("PartitionsQ[5]", "3"), ("PartitionsQ[10]", "10"), ("PartitionsQ[0]", "1"),
    ("IntegerPartitions[4]", "{{4}, {3, 1}, {2, 2}, {2, 1, 1}, {1, 1, 1, 1}}"),
    ("IntegerPartitions[5, 2]", "{{5}, {4, 1}, {3, 2}}"), ("IntegerPartitions[6, {2}]", "{{5, 1}, {4, 2}, {3, 3}}"),
    ("IntegerPartitions[0]", "{{}}"),
    ("PartitionsList[3]", "{{3}, {2, 1}, {1, 1, 1}}"), ("PartitionsList[1]", "{{1}}"),
    ("PartitionsList[6]", Pred(lambda r: len(r.args) == 11, "11 partitions")),
    ("Compositions[2, 2]", "{{0, 2}, {1, 1}, {2, 0}}"), ("Compositions[3, 1]", "{{3}}"),
    ("Compositions[4, 3]", Pred(lambda r: len(r.args) == 15, "C(6,2) = 15 compositions")),
    ("Multinomial[2, 2, 2]", "90"), ("Multinomial[1, 1]", "2"), ("Multinomial[3]", "1"), ("Multinomial[]", "1"),
    ("Permutations[{1, 2, 3}]", "{{1, 2, 3}, {1, 3, 2}, {2, 1, 3}, {2, 3, 1}, {3, 1, 2}, {3, 2, 1}}"),
    ("Permutations[{a, a, b}]", "{{a, a, b}, {a, b, a}, {b, a, a}}"),
    ("Permutations[{1, 2, 3}, {2}]", "{{1, 2}, {1, 3}, {2, 1}, {2, 3}, {3, 1}, {3, 2}}"),
    ("Permutations[5]", Raises(ValueError)),
    ("Subsets[{a, b}]", "{{}, {a}, {b}, {a, b}}"), ("Subsets[{a, b, c}, {2}]", "{{a, b}, {a, c}, {b, c}}"),
    ("Subsets[{1, 2, 3}, 1]", "{{}, {1}, {2}, {3}}"),
    ("PowerSet[{1, 2}]", "{{}, {1}, {2}, {1, 2}}"), ("PowerSet[{}]", "{{}}"),
    ("PowerSet[{1, 2, 3, 4}]", Pred(lambda r: len(r.args) == 16, "16 subsets")),
    ("CartesianProduct[{1, 2}, {a, b}]", "{{1, a}, {1, b}, {2, a}, {2, b}}"),
    ("CartesianProduct[{1}, {2}, {3, 4}]", "{{1, 2, 3}, {1, 2, 4}}"), ("CartesianProduct[{1, 2}, {}]", "{}"),
    ("Tuples[{0, 1}, 2]", "{{0, 0}, {0, 1}, {1, 0}, {1, 1}}"), ("Tuples[{a, b}, 0]", "{{}}"),
    ("Tuples[{{1, 2}, {x}}]", "{{1, x}, {2, x}}"),
    ("HookLengthFormula[{2, 1}]", "2"), ("HookLengthFormula[{3, 2}]", "5"), ("HookLengthFormula[{4, 2, 1}]", "35"),
    ("HookLengthFormula[{1, 2}]", Raises()),
    ("ArithmeticSequence[1, 2, 5]", "{1, 3, 5, 7, 9}"), ("ArithmeticSequence[1/2, 1/2, 3]", "{1/2, 1, 3/2}"),
    ("ArithmeticSequence[0, 1, 0]", "{}"),
    ("GeometricSequence[1, 2, 5]", "{1, 2, 4, 8, 16}"), ("GeometricSequence[3, 1/3, 3]", "{3, 1, 1/3}"),
    ("GeometricSequence[1, -1, 4]", "{1, -1, 1, -1}"),
    # first n Fibonacci numbers starting at F(0) (MikoshiLang convention)
    ("FibonacciSequence[7]", "{0, 1, 1, 2, 3, 5, 8}"), ("FibonacciSequence[1]", "{0}"), ("FibonacciSequence[0]", "{}"),
    ("YoungTableaux[{2, 1}]", "{{{1, 2}, {3}}, {{1, 3}, {2}}}"),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_combinatorics(case):
    check(case)
