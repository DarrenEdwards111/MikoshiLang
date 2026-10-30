"""Heads that the evaluator kernel implements before the bundled library is consulted. They are
recorded with status "kernel"; these cases confirm the kernel gives the Wolfram result."""
import pytest

from _harness import Raises, check, ids

CASES = [
    ("Apart[1/(x^2 - 1), x]", "1/(2 (x - 1)) - 1/(2 (x + 1))"), ("Apart[(x + 2)/(x (x + 1)), x]", "2/x - 1/(x + 1)"),
    ("Append[{1, 2}, 3]", "{1, 2, 3}"), ("Append[{}, a]", "{a}"),
    ("Cancel[(x^2 - 1)/(x - 1)]", "1 + x"), ("Cancel[(x^2 + 2 x + 1)/(x + 1)]", "x + 1"),
    ("D[x^3, x]", "3 x^2"), ("D[Sin[x] y, y]", "Sin[x]"), ("D[x^3, {x, 2}]", "6 x"),
    ("DSolve[y'[x] == y[x], y[x], x]", "{{y[x] -> C[1] E^x}}"),
    ("Det[{{1, 2}, {3, 4}}]", "-2"), ("Det[{{a, b}, {c, d}}]", "a d - b c"),
    ("Diff[x^2, x]", "2 x"), ("Diff[Exp[2 x], x]", "2 E^(2 x)"),
    ("Dot[{1, 2}, {3, 4}]", "11"), ("Dot[{{1, 0}, {0, 2}}, {3, 4}]", "{3, 8}"),
    ("DotProduct[{1, 2}, {3, 4}]", "11"), ("DotProduct[{a, b}, {c, d}]", "a c + b d"),
    ("Drop[{1, 2, 3}, 1]", "{2, 3}"), ("Drop[{1, 2, 3}, -1]", "{1, 2}"),
    ("EvenQ[4]", True), ("EvenQ[3]", False), ("EvenQ[x]", False),
    ("First[{a, b}]", "a"), ("First[{}]", Raises()),
    ("Flatten[{{1}, {2, {3}}}]", "{1, 2, 3}"), ("Flatten[{}]", "{}"),
    ("IntegerQ[3]", True), ("IntegerQ[3.0]", False), ("IntegerQ[x]", False),
    ("Inverse[{{1, 2}, {3, 4}}]", "{{-2, 1}, {3/2, -1/2}}"), ("Inverse[{{1, 1}, {1, 1}}]", Raises()),
    ("Join[{1}, {2}]", "{1, 2}"), ("Join[{1}, {}, {2, 3}]", "{1, 2, 3}"),
    ("Last[{a, b}]", "b"), ("Last[{}]", Raises()),
    ("Length[{a, b, c}]", "3"), ("Length[{}]", "0"),
    ("Limit[Sin[x]/x, x -> 0]", "1"), ("Limit[(1 + 1/n)^n, n -> Infinity]", "E"),
    ("MemberQ[{1, 2}, 2]", True), ("MemberQ[{1, 2}, 3]", False),
    ("Most[{1, 2, 3}]", "{1, 2}"), ("Most[{1}]", "{}"),
    ("N[1/4]", 0.25), ("N[Sqrt[2]]", 2 ** 0.5),
    ("Negative[-2]", True), ("Negative[0]", False),
    ("NonNegative[0]", True), ("NonNegative[-1/2]", False),
    ("NumberQ[2.5]", True), ("NumberQ[x]", False),
    ("OddQ[3]", True), ("OddQ[4]", False),
    ("Part[{a, b, c}, 2]", "b"), ("Part[{a, b, c}, -1]", "c"), ("Part[{a, b}, 5]", Raises()),
    ("PartialD[x^2 y, y]", "x^2"), ("PartialD[x^2 y, x]", "2 x y"),
    ("Plus[1, 2, x]", "3 + x"), ("Plus[]", "0"),
    ("Position[{a, b, a}, a]", "{{1}, {3}}"), ("Position[{a, b}, c]", "{}"),
    ("Positive[3]", True), ("Positive[0]", False),
    ("Power[2, 10]", "1024"), ("Power[4, 1/2]", "2"),
    ("Prepend[{2}, 1]", "{1, 2}"), ("Prepend[{}, 1]", "{1}"),
    ("PrimeQ[97]", True), ("PrimeQ[1]", False), ("PrimeQ[91]", False),
    ("Range[5]", "{1, 2, 3, 4, 5}"), ("Range[2, 6, 2]", "{2, 4, 6}"),
    ("Rest[{1, 2, 3}]", "{2, 3}"), ("Rest[{1}]", "{}"),
    ("Reverse[{1, 2, 3}]", "{3, 2, 1}"), ("Reverse[{}]", "{}"),
    ("Series[Exp[x], {x, 0, 3}]", "1 + x + x^2/2 + x^3/6 + O[x]^4"),
    ("Solve[x^2 == 4, x]", "{{x -> -2}, {x -> 2}}"), ("Solve[2 x + 1 == 0, x]", "{{x -> -1/2}}"),
    ("Sort[{3, 1, 2}]", "{1, 2, 3}"), ("Sort[{}]", "{}"),
    ("StringJoin[\"a\", \"b\"]", "\"ab\""), ("StringJoin[\"\", \"x\"]", "\"x\""),
    ("StringLength[\"abc\"]", "3"), ("StringLength[\"\"]", "0"),
    ("Take[{1, 2, 3}, 2]", "{1, 2}"), ("Take[{1, 2, 3}, -1]", "{3}"),
    ("Times[2, 3, x]", "6 x"), ("Times[]", "1"),
    ("Together[1/x + 1/y]", "(x + y)/(x y)"), ("Together[1/2 + x/2]", "(1 + x)/2"),
    ("Total[{1, 2, 3}]", "6"), ("Total[{}]", "0"),
    ("Transpose[{{1, 2}, {3, 4}}]", "{{1, 3}, {2, 4}}"), ("Transpose[{{1, 2, 3}}]", "{{1}, {2}, {3}}"),
    ("Table[i^2, {i, 3}]", "{1, 4, 9}"), ("Table[x, 2]", "{x, x}"),
    # kernel heads that were wrongly listed as quarantined
    ("AtomQ[x]", True), ("AtomQ[{1}]", False), ("AtomQ[\"s\"]", True),
    ("ListQ[{}]", True), ("ListQ[1]", False),
    ("StringQ[\"a\"]", True), ("StringQ[1]", False),
    ("ToString[1]", "\"1\""), ("ToString[x^2]", "\"x^2\""),
    ("ReplaceAll[x + 1, x -> 2]", "3"), ("ReplaceAll[{x, y}, {x -> 1, y -> 2}]", "{1, 2}"),
    ("Distance2D[{0, 0}, {3, 4}]", "5"), ("Distance2D[{0, 0}, {1, 1}]", "Sqrt[2]"), ("Distance2D[{1, 2}]", Raises()),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_kernel_heads(case):
    check(case)
