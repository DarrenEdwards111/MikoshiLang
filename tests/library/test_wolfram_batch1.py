"""Differential tests for coverage batch 1 (lists, strings, associations and the kernel extensions
that support them).

Expected values are Wolfram Language results: examples from the reference pages
(reference.wolfram.com/language/ref/<Name>.html) or results derived from their Details sections,
cross-checked where possible against independent Python references (``re``, ``difflib``,
``unicodedata``, ``itertools``, NumPy-style reshaping) computed in this file.
"""
import itertools
import re
import unicodedata

import pytest

from _harness import Raises, check, ids


def S(s):
    """Expected Wolfram string result."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t") + '"'


def SL(xs):
    return "{" + ", ".join(S(x) for x in xs) + "}"


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def roman(n):
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


# ---------------------------------------------------------------- kernel extensions
KERNEL = [
    # List drops Nothing and splices Splice
    ("{1, Nothing, 2}", "{1, 2}"), ("{a, Splice[{b, c}], d}", "{a, b, c, d}"), ("Length[{Nothing}]", "0"),
    ("Length[<|a -> 1, b -> 2|>]", "2"), ("Length[f[x, y, z]]", "3"), ("Length[x]", "0"),
    ("AtomQ[<|a -> 1|>]", True), ("AtomQ[1/2]", True), ("AtomQ[{1}]", False),
    # Part with Span, All, lists, negative indices, keys
    ("{a, b, c, d, e}[[2 ;; 4]]", "{b, c, d}"), ("{a, b, c, d, e}[[;; ;; 2]]", "{a, c, e}"),
    ("{a, b, c, d, e}[[-2 ;;]]", "{d, e}"), ("{{1, 2}, {3, 4}}[[All, 2]]", "{2, 4}"),
    ("{a, b, c}[[{1, 3}]]", "{a, c}"), ("<|a -> 1, b -> 2|>[[Key[b]]]", "2"), ("<|\"x\" -> 1|>[[\"x\"]]", "1"),
    ("<|a -> 1, b -> 2|>[[2]]", "2"), ("Part[{a, b}, 3]", Raises()),
    ("First[{}, x]", "x"), ("First[{a, b}]", "a"), ("First[<|a -> 1, b -> 2|>]", "1"),
    ("Last[{a, b}]", "b"), ("Last[{}, d]", "d"), ("Last[<|a -> 1, b -> 2|>]", "2"),
    ("Rest[{a, b, c}]", "{b, c}"), ("Rest[f[a, b]]", "f[b]"), ("Rest[{}]", Raises()),
    ("Most[{a, b, c}]", "{a, b}"), ("Most[f[a, b]]", "f[a]"), ("Most[{}]", Raises()),
    ("Append[{a, b}, c]", "{a, b, c}"), ("Append[<|a -> 1|>, b -> 2]", "<|a -> 1, b -> 2|>"),
    ("Append[<|a -> 1|>, a -> 3]", "<|a -> 3|>"),
    ("Prepend[{a, b}, c]", "{c, a, b}"), ("Prepend[<|a -> 1|>, b -> 2]", "<|b -> 2, a -> 1|>"), ("Prepend[f[a], x]", "f[x, a]"),
    ("Join[{a}, {b, c}]", "{a, b, c}"), ("Join[{{1}, {2}}, {{3}, {4}}, 2]", "{{1, 3}, {2, 4}}"),
    ("Join[<|a -> 1|>, <|b -> 2, a -> 3|>]", "<|a -> 3, b -> 2|>"),
    ("Reverse[{1, 2, 3}]", "{3, 2, 1}"), ("Reverse[{{1, 2}, {3, 4}}, 2]", "{{2, 1}, {4, 3}}"),
    ("Reverse[{{1, 2}, {3, 4}}, {1, 2}]", "{{4, 3}, {2, 1}}"),
    ("Take[{a, b, c, d}, UpTo[10]]", "{a, b, c, d}"), ("Take[{{1, 2, 3}, {4, 5, 6}}, 2, 2]", "{{1, 2}, {4, 5}}"),
    ("Take[{a, b, c, d, e}, {1, -1, 2}]", "{a, c, e}"), ("Take[{a, b}, 3]", Raises()),
    ("Drop[{a, b, c, d}, UpTo[10]]", "{}"), ("Drop[{{1, 2, 3}, {4, 5, 6}}, 1, 1]", "{{5, 6}}"),
    ("Drop[{a, b, c, d, e}, {2, -1, 2}]", "{a, c, e}"),
    ("Sort[<|a -> 3, b -> 1, c -> 2|>]", "<|b -> 1, c -> 2, a -> 3|>"), ("Sort[{3, 1, 2}, Greater]", "{3, 2, 1}"),
    ("Sort[{\"b\", \"a\", \"C\"}]", SL(["a", "b", "C"])),
    ("Total[{{1, 2}, {3, 4}}, 2]", "10"), ("Total[{{1, 2}, {3, 4}}, {2}]", "{3, 7}"), ("Total[<|a -> 1, b -> 2|>]", "3"),
    ("Total[{{1, 2}, {3, 4}}, Infinity]", "10"),
    ("Flatten[{a, {b, {c}}}, 1]", "{a, b, {c}}"), ("Flatten[f[a, f[b], g[c]], Infinity, f]", "f[a, b, g[c]]"),
    ("Flatten[{{a, b}, {c, d}}, {{2}, {1}}]", "{{a, c}, {b, d}}"),
    ("Transpose[{{1, 2, 3}, {4, 5, 6}}]", "{{1, 4}, {2, 5}, {3, 6}}"),
    ("Dimensions[Transpose[ConstantArray[0, {2, 3, 4}], {3, 1, 2}]]", "{3, 4, 2}"),
    ("Transpose[{{a, b}, {c, d}}, {1, 1}]", "{a, d}"),
    ("Cases[{1, {2, {3}}}, _Integer, Infinity]", "{1, 2, 3}"), ("Cases[{1, a, 2}, _Integer, 1, 1]", "{1}"),
    ("Cases[{a -> 1, b -> 2}, (x_ -> 2) :> x]", "{b}"),
    ("DeleteCases[{1, {2, a}}, _Integer, Infinity]", "{{a}}"), ("DeleteCases[{1, a, 2, b}, _Integer]", "{a, b}"),
    ("DeleteCases[{1, 2, 1, 2}, 1, 1, 1]", "{2, 1, 2}"),
    ("Count[{1, {1, 2}}, 1, Infinity]", "2"), ("Count[{a, b, a}, a]", "2"), ("Count[{1, {1}}, 1]", "1"),
    ("Position[{a, {b, a}}, a]", "{{1}, {2, 2}}"), ("Position[f[g[x]], _Symbol, Heads -> False]", "{{1, 1}}"), ("FreeQ[f[x], f, Heads -> False]", True),
    ("Position[{1, 2, 1}, 1, 1, 1]", "{{1}}"),
    ("MemberQ[{1, {2}}, 2]", False), ("MemberQ[{1, {2}}, 2, 2]", True), ("MemberQ[{a, b}, _Symbol]", True),
    ("FreeQ[{1, {2}}, 2]", False), ("FreeQ[f[x], f]", False), ("FreeQ[{1, {2}}, 2, 1]", True),
    ("Map[f, <|a -> 1, b -> 2|>]", "<|a -> f[1], b -> f[2]|>"), ("Map[f, {{a, b}}, {2}]", "{{f[a], f[b]}}"),
    ("Map[f][{1, 2}]", "{f[1], f[2]}"), ("Map[f, {1, {2}}, -1]", "{f[1], f[{f[2]}]}"), ("Map[f, {1, {2}}, {-1}]", "{f[1], {f[2]}}"),
    ("MapAll[f, {a, {b}}]", "f[{f[a], f[{f[b]}]}]"), ("MapAll[g, x]", "g[x]"), ("f //@ {a}", "f[{f[a]}]"),
    ("MapIndexed[f, {a, b}]", "{f[a, {1}], f[b, {2}]}"), ("MapIndexed[f, {{a}}, {2}]", "{{f[a, {1, 1}]}}"),
    ("MapIndexed[f, <|x -> 1|>]", "<|x -> f[1, {Key[x]}]|>"),
    ("Apply[f, {{1, 2}, {3}}, {1}]", "{f[1, 2], f[3]}"), ("f @@@ {{a, b}}", "{f[a, b]}"), ("Apply[f, <|a -> 1, b -> 2|>]", "f[1, 2]"),
    ("Select[<|a -> 1, b -> 2, c -> 3|>, OddQ]", "<|a -> 1, c -> 3|>"), ("Select[{1, 2, 3, 4}, EvenQ, 1]", "{2}"),
    ("Select[EvenQ][{1, 2}]", "{2}"),
    ("Thread[f[{a, b}, {c, d}]]", "{f[a, c], f[b, d]}"), ("Thread[{a, b} -> {1, 2}]", "{a -> 1, b -> 2}"),
    ("Thread[f[{a, b}, x]]", "{f[a, x], f[b, x]}"),
    ("Normal[<|a -> 1, b -> <|c -> 2|>|>]", "{a -> 1, b -> <|c -> 2|>}"), ("Normal[{<|a -> 1|>}]", "{{a -> 1}}"),
    ("Normal[<|a -> 1|>, SparseArray]", "<|a -> 1|>"),
    ("Range[1.5]", "{1.}"), ("Range[0, 1, 1/4]", "{0, 1/4, 1/2, 3/4, 1}"), ("Range[x, x + 2]", "{x, 1 + x, 2 + x}"),
    ("Range[5, 1, -2]", "{5, 3, 1}"),
    # comparisons with more than two arguments
    ("Equal[1, 1, 1]", True), ("Equal[1, 1, 2]", False), ("1 == 1.0 == 1", True),
    ("Unequal[1, 2, 3]", True), ("Unequal[1, 2, 1]", False), ("Unequal[a, a]", False),
    ("Less[1, 2, 3]", True), ("Less[1, 3, 2]", False), ("1 < 2 < 3", True),
    # operator forms and compound heads
    ("Composition[f, g][x]", "f[g[x]]"), ("(f @* g @* h)[x]", "f[g[h[x]]]"), ("Composition[][x]", "x"),
    ("RightComposition[f, g][x]", "g[f[x]]"), ("(f /* g)[x]", "g[f[x]]"), ("RightComposition[f][x]", "f[x]"),
    ("GreaterThan[2][3]", True), ("Select[{1, 5, 3}, GreaterThan[2]]", "{5, 3}"), ("GreaterThan[2][1]", False),
    ("LessThan[2][1]", True), ("Select[{1, 5, 3}, LessThan[4]]", "{1, 3}"), ("LessThan[2][2]", False),
    ("GreaterEqualThan[2][2]", True), ("GreaterEqualThan[2][1]", False), ("Select[{1, 2, 3}, GreaterEqualThan[2]]", "{2, 3}"),
    ("LessEqualThan[2][2]", True), ("LessEqualThan[2][3]", False), ("Select[{1, 2, 3}, LessEqualThan[2]]", "{1, 2}"),
    ("EqualTo[2][2.]", True), ("EqualTo[2][3]", False), ("Select[{1, 2, 1}, EqualTo[1]]", "{1, 1}"),
    ("UnequalTo[2][3]", True), ("UnequalTo[2][2]", False), ("Select[{1, 2, 1}, UnequalTo[1]]", "{2}"),
    ("SameAs[2][2]", True), ("SameAs[2][2.]", False), ("Select[{a, b, a}, SameAs[a]]", "{a, a}"),
    ("Key[a][<|a -> 3|>]", "3"), ("Key[b][<|a -> 3|>]", "Missing[\"KeyAbsent\", b]"), ("Map[Key[a], {<|a -> 1|>, <|a -> 2|>}]", "{1, 2}"),
    ("Missing[\"KeyAbsent\", x]", "Missing[\"KeyAbsent\", x]"), ("<|a -> 1|>[b]", "Missing[\"KeyAbsent\", b]"),
    ("Head[Missing[]]", "Missing"),
    ("Nothing", "Nothing"), ("Map[If[# > 1, #, Nothing] &, {1, 2, 3}]", "{2, 3}"), ("{Nothing, Nothing}", "{}"),
    ("Splice[{a}]", "Splice[{a}]"), ("{Splice[{}]}", "{}"), ("f[Splice[{a, b}]]", "f[Splice[{a, b}]]"),
    ("Span[1, 2]", "1 ;; 2"), ("{a, b, c}[[Span[2, All]]]", "{b, c}"), ("Span[1, 3] === (1 ;; 3)", True), ("Head[;; 3]", "Span"),
    ("Take[{1, 2}, UpTo[5]]", "{1, 2}"), ("UpTo[3]", "UpTo[3]"), ("Partition[{1, 2, 3}, UpTo[2]]", "{{1, 2}, {3}}"),
    # patterns: Except, Repeated, Optional
    ("Cases[{1, a, 2.5, \"s\"}, Except[_Integer]]", "{a, 2.5, \"s\"}"), ("Cases[{1, 2, 3, 4}, Except[2, _Integer]]", "{1, 3, 4}"),
    ("Count[{a, b, c}, Except[b]]", "2"), ("MatchQ[x, Except[x]]", False),
    ("MatchQ[{a, a, a}, {a ..}]", True), ("MatchQ[{a, b}, {a ..}]", False), ("MatchQ[{}, {a ..}]", False),
    ("MatchQ[{1, 2, 3}, {Repeated[_Integer, {2, 3}]}]", True), ("MatchQ[{1, 2, 3, 4}, {Repeated[_Integer, 3]}]", False),
    ("MatchQ[{x, x}, {y_ ..}]", True), ("MatchQ[{x, z}, {y_ ..}]", False),
    ("Cases[{{1, 2}, {1, 1, 1}, {}}, {1 ..}]", "{{1, 1, 1}}"),
    ("MatchQ[{}, {a ...}]", True), ("MatchQ[{a, a}, {RepeatedNull[a]}]", True), ("MatchQ[{b}, {a ...}]", False),
    ("Replace[{1, 2}, {x_, y_ : 0, z_ : 9} :> {x, y, z}]", "{1, 2, 9}"),
    ("ff[x_, y_ : 5] := x + y; {ff[1], ff[1, 2]}", "{6, 3}"),
    ("hh[x_. + y_] := {x, y}; {hh[a], hh[3 + a]}", "{{0, a}, {3, a}}"),
    ("kk[x_. * y_] := {x, y}; {kk[a], kk[2 a]}", "{{1, a}, {2, a}}"),
    ("pp[x_^n_.] := {x, n}; {pp[a], pp[a^3]}", "{{a, 1}, {a, 3}}"),
    ("Minus[1]", "-1"), ("Minus[{1, x}]", "{-1, -x}"), ("Minus[-x]", "x"),
    ("Optional[x_, 1]", "x_:1"), ("gg[x_, y_.] := {x, y}; gg[1]", "gg[1]"),
    # slots with names
    ("#a &[<|\"a\" -> 1, \"b\" -> 2|>]", "1"), ("(#x + #y &)[<|\"x\" -> 1, \"y\" -> 2|>]", "3"), ("Map[#n &, {<|\"n\" -> 5|>}]", "{5}"),
    ("#a &[<|a -> 1|>]", "Missing[\"KeyAbsent\", \"a\"]"),
    # assignment into a stored association
    ("asc = <|a -> 1|>; asc[b] = 2; asc", "<|a -> 1, b -> 2|>"), ("asc2 = <|a -> 1|>; asc2[a] = 5; asc2", "<|a -> 5|>"),
]

# ---------------------------------------------------------------- lists
LISTS = [
    ("Array[f, 3]", "{f[1], f[2], f[3]}"), ("Array[f, {2, 2}]", "{{f[1, 1], f[1, 2]}, {f[2, 1], f[2, 2]}}"),
    ("Array[f, 3, 0]", "{f[0], f[1], f[2]}"), ("Array[#^2 &, 4]", "{1, 4, 9, 16}"), ("Array[f, -1]", Raises()),
    ("ConstantArray[0, 3]", "{0, 0, 0}"), ("ConstantArray[x, {2, 2}]", "{{x, x}, {x, x}}"), ("ConstantArray[a, 0]", "{}"),
    ("Subdivide[4]", "{0, 1/4, 1/2, 3/4, 1}"), ("Subdivide[10, 2]", "{0, 5, 10}"), ("Subdivide[1, 2, 2]", "{1, 3/2, 2}"),
    ("PowerRange[1000]", "{1, 10, 100, 1000}"), ("PowerRange[1, 100, 3]", "{1, 3, 9, 27, 81}"), ("PowerRange[2, 20, 2]", "{2, 4, 8, 16}"),
    ("Subsequences[{a, b, c}]", "{{}, {a}, {b}, {c}, {a, b}, {b, c}, {a, b, c}}"), ("Subsequences[{a, b, c}, {2}]", "{{a, b}, {b, c}}"),
    ("Length[Subsequences[Range[5]]]", "16"),
    ("Reap[Sow[1]; Sow[2]; 3]", "{3, {{1, 2}}}"), ("Reap[Scan[Sow, {1, 2, 3}]]", "{Null, {{1, 2, 3}}}"), ("Reap[x]", "{x, {}}"),
    ("Reap[Sow[1, a]; Sow[2, b]; Sow[3, a], a]", "{3, {{1, 3}}}"), ("Sow[5]", "5"),
    ("Extract[{a, {b, c}}, {2, 1}]", "b"), ("Extract[{a, b, c}, {{1}, {3}}]", "{a, c}"), ("Extract[{a, b}, {5}]", Raises()),
    ("MapAt[f, {a, b, c}, 2]", "{a, f[b], c}"), ("MapAt[f, {a, b, c}, {{1}, {3}}]", "{f[a], b, f[c]}"),
    ("MapAt[f, {{a, b}, {c, d}}, {2, 1}]", "{{a, b}, {f[c], d}}"), ("MapAt[f, {a, b, c}, -1]", "{a, b, f[c]}"),
    ("ReplacePart[{a, b, c}, 2 -> x]", "{a, x, c}"), ("ReplacePart[{a, b, c}, {1 -> x, 3 -> y}]", "{x, b, y}"),
    ("ReplacePart[{{a, b}, {c, d}}, {2, 1} -> x]", "{{a, b}, {x, d}}"),
    ("Delete[{a, b, c}, 2]", "{a, c}"), ("Delete[{a, b, c}, {{1}, {3}}]", "{b}"), ("Delete[{{a, b}, c}, {1, 2}]", "{{a}, c}"),
    ("Insert[{a, b, c}, x, 2]", "{a, x, b, c}"), ("Insert[{a, b, c}, x, -1]", "{a, b, c, x}"), ("Insert[{{a}, b}, x, {1, 1}]", "{{x, a}, b}"),
    ("FirstPosition[{a, b, a}, a]", "{1}"), ("FirstPosition[{1, {2, 3}}, 3]", "{2, 2}"), ("FirstPosition[{a}, z]", "Missing[\"NotFound\"]"),
    ("SelectFirst[{1, 2, 3, 4}, EvenQ]", "2"), ("SelectFirst[{1, 3}, EvenQ]", "Missing[\"NotFound\"]"), ("SelectFirst[{1, 3}, EvenQ, 0]", "0"),
    ("FirstCase[{1, a, 2}, _Symbol]", "a"), ("FirstCase[{1, 2}, _Symbol]", "Missing[\"NotFound\"]"), ("FirstCase[{1, x -> 2}, (_ -> v_) :> v]", "2"),
    ("Pick[{a, b, c}, {True, False, True}]", "{a, c}"), ("Pick[{a, b, c}, {1, 0, 1}, 1]", "{a, c}"),
    ("Pick[{{a, b}, {c, d}}, {{1, 0}, {0, 1}}, 1]", "{{a}, {d}}"),
    ("TakeWhile[{1, 3, 4, 5}, OddQ]", "{1, 3}"), ("TakeWhile[{2, 3}, OddQ]", "{}"), ("TakeWhile[{1, 3}, OddQ]", "{1, 3}"),
    ("LengthWhile[{1, 3, 4, 5}, OddQ]", "2"), ("LengthWhile[{}, OddQ]", "0"), ("LengthWhile[{2}, OddQ]", "0"),
    ("TakeDrop[{a, b, c, d}, 2]", "{{a, b}, {c, d}}"), ("TakeDrop[{a, b, c}, -1]", "{{c}, {a, b}}"), ("TakeDrop[{a, b, c}, {2, 3}]", "{{b, c}, {a}}"),
    ("TakeList[{a, b, c, d, e}, {2, 3}]", "{{a, b}, {c, d, e}}"), ("TakeList[{a, b, c}, {1, 1}]", "{{a}, {b}}"), ("TakeList[{a}, {2}]", Raises()),
    ("DeleteAdjacentDuplicates[{1, 1, 2, 2, 1}]", "{1, 2, 1}"), ("DeleteAdjacentDuplicates[{}]", "{}"),
    ("DeleteAdjacentDuplicates[{1, 2, 4, 5}, Abs[#1 - #2] == 1 &]", "{1, 4}"),
    ("DeleteElements[{a, b, a, c}, {a}]", "{b, c}"), ("DeleteElements[{a, b, a, c}, 1 -> {a}]", "{b, a, c}"),
    ("DeleteElements[{1, 2, 3}, {4}]", "{1, 2, 3}"),
    ("MaximalBy[{-3, 1, 3}, Abs]", "{-3, 3}"), ("MaximalBy[{\"a\", \"bbb\", \"cc\"}, StringLength]", SL(["bbb"])),
    ("MaximalBy[{1, 2, 3, 4}, Mod[#, 3] &, 2]", "{2, 1}"),
    ("MinimalBy[{-3, 1, 3}, Abs]", "{1}"), ("MinimalBy[{{a, 2}, {b, 1}, {c, 1}}, Last]", "{{b, 1}, {c, 1}}"), ("MinimalBy[{}, f]", "{}"),
    ("TakeLargest[{1, 5, 3, 4}, 2]", "{5, 4}"), ("TakeLargest[{1, Missing[], 4}, 2]", "{4, 1}"), ("TakeLargest[{1, 2}, 3]", Raises()),
    ("TakeLargest[{3, 1}, UpTo[5]]", "{3, 1}"),
    ("TakeSmallest[{1, 5, 3, 4}, 2]", "{1, 3}"), ("TakeSmallest[<|a -> 3, b -> 1, c -> 2|>, 2]", "<|b -> 1, c -> 2|>"),
    ("TakeSmallest[{2, 1}, UpTo[3]]", "{1, 2}"),
    ("TakeLargestBy[{-5, 1, 3}, Abs, 2]", "{-5, 3}"), ("TakeLargestBy[{\"aa\", \"b\", \"ccc\"}, StringLength, 1]", SL(["ccc"])),
    ("TakeLargestBy[{1, 2}, Minus, 1]", "{1}"),
    ("TakeSmallestBy[{-5, 1, 3}, Abs, 2]", "{1, 3}"), ("TakeSmallestBy[{\"aa\", \"b\"}, StringLength, 1]", SL(["b"])),
    ("TakeSmallestBy[{1, 2}, Minus, 1]", "{2}"),
    ("PositionLargest[{1, 5, 3, 5}]", "{2, 4}"), ("PositionLargest[{1, 5, 3}, 2]", "{2, 3}"), ("PositionLargest[{7}]", "{1}"),
    ("PositionSmallest[{3, 1, 1}]", "{2, 3}"), ("PositionSmallest[{3, 1, 2}, 2]", "{2, 3}"), ("PositionSmallest[{7}]", "{1}"),
    ("SequenceCases[{a, b, a, b}, {a, b}]", "{{a, b}, {a, b}}"), ("SequenceCases[{1, 2, 3, 4}, {x_, y_} /; y == x + 1]", "{{1, 2}, {3, 4}}"),
    ("SequenceCases[{a, b, c}, {x_, b} :> x]", "{a}"), ("SequenceCases[{a, b, 1, 2, c, d, 3, 4}, {__Symbol}]", "{{a, b}, {c, d}}"),
    ("SequencePosition[{a, b, c, d}, {b, c}]", "{{2, 3}}"), ("SequencePosition[{a, b, a, b, a, b}, {a, b}]", "{{1, 2}, {3, 4}, {5, 6}}"),
    ("SequencePosition[{a, a, a}, {a, a}]", "{{1, 2}, {2, 3}}"),
    ("SequencePosition[{a, b, b, a, b, b, b}, {Repeated[b]}]", "{{2, 3}, {3, 3}, {5, 7}, {6, 7}, {7, 7}}"),
    ("SequenceCount[{a, b, a, b}, {a, b}]", "2"), ("SequenceCount[{a, a, a}, {a, a}]", "1"), ("SequenceCount[{a}, {b}]", "0"),
    ("SequenceReplace[{a, b, c, a, b}, {a, b} -> x]", "{x, c, x}"), ("SequenceReplace[{1, 2, 3}, {x_, y_} :> x + y]", "{3, 3}"),
    ("SequenceReplace[{a, b}, {c} -> d]", "{a, b}"),
    ("SequenceSplit[{a, x, b, x, c}, {x}]", "{{a}, {b}, {c}}"), ("SequenceSplit[{x, a, x}, {x}]", "{{a}}"),
    ("SequenceSplit[{a, x, b}, {x} -> y]", "{{a}, y, {b}}"),
    ("Level[{a, {b, c}}, {1}]", "{a, {b, c}}"), ("Level[{a, {b, c}}, {-1}]", "{a, b, c}"), ("Level[{a, {b, c}}, 2]", "{a, b, c, {b, c}}"),
    ("Level[f[g[x], y], {0, Infinity}]", "{x, g[x], y, f[g[x], y]}"), ("Level[{a, {b}}, {1}, f]", "f[a, {b}]"),
    ("Depth[{a, {b, {c}}}]", "4"), ("Depth[x]", "1"), ("Depth[{}]", "1"), ("Depth[<|a -> x|>]", "2"),
    ("Dimensions[{{1, 2, 3}, {4, 5, 6}}]", "{2, 3}"), ("Dimensions[{{1, 2}, {3}}]", "{2}"), ("Dimensions[x]", "{}"),
    ("Dimensions[{{{1}, {2}}}, 2]", "{1, 2}"), ("Dimensions[f[a, b]]", "{2}"), ("Dimensions[{f[a], f[b]}]", "{2}"),
    ("ArrayDepth[{{1, 2}, {3, 4}}]", "2"), ("ArrayDepth[{{1, 2}, {3}}]", "1"), ("ArrayDepth[x]", "0"),
    ("ArrayQ[{{1, 2}, {3, 4}}]", True), ("ArrayQ[{{1, 2}, {3}}]", False), ("ArrayQ[{{1, 2}, {3, 4}}, 2, IntegerQ]", True),
    ("ArrayQ[{1, {2}}]", False), ("ArrayQ[{{1, 2}, {3, 4}}, 1]", False), ("ArrayQ[x]", False),
    ("VectorQ[{1, 2}]", True), ("VectorQ[{{1}}]", False), ("VectorQ[{1, a}, NumberQ]", False), ("VectorQ[{1, 2}, NumberQ]", True),
    ("MatrixQ[{{1, 2}, {3, 4}}]", True), ("MatrixQ[{1, 2}]", False), ("MatrixQ[{{1, 2}, {3}}]", False), ("MatrixQ[{{1, x}}, NumberQ]", False),
    ("Split[{1, 1, 2, 2, 2, 1}]", "{{1, 1}, {2, 2, 2}, {1}}"), ("Split[{1, 2, 3, 5, 6, 8}, #2 - #1 == 1 &]", "{{1, 2, 3}, {5, 6}, {8}}"),
    ("Split[{}]", "{}"),
    ("SplitBy[{1, 3, 2, 4, 5}, OddQ]", "{{1, 3}, {2, 4}, {5}}"), ("SplitBy[{{a, 1}, {a, 2}, {b, 1}}, First]", "{{{a, 1}, {a, 2}}, {{b, 1}}}"),
    ("SplitBy[{}, f]", "{}"),
    ("Gather[{a, b, a, c, b}]", "{{a, a}, {b, b}, {c}}"), ("Gather[{1, 2, 3, 4, 5}, Mod[#1, 2] == Mod[#2, 2] &]", "{{1, 3, 5}, {2, 4}}"),
    ("Gather[{}]", "{}"),
    ("GatherBy[{1, 2, 3, 4, 5}, OddQ]", "{{1, 3, 5}, {2, 4}}"), ("GatherBy[{{a, 1}, {b, 2}, {a, 3}}, First]", "{{{a, 1}, {a, 3}}, {{b, 2}}}"),
    ("GatherBy[{}, f]", "{}"),
    ("FlattenAt[{a, {b, c}, {d}}, 2]", "{a, b, c, {d}}"), ("FlattenAt[{a, {b, c}, {d}}, {{2}, {3}}]", "{a, b, c, d}"),
    ("FlattenAt[{a, {b}}, -1]", "{a, b}"),
    ("ArrayReshape[{1, 2, 3, 4, 5, 6}, {2, 3}]", "{{1, 2, 3}, {4, 5, 6}}"), ("ArrayReshape[{1, 2, 3}, {2, 2}]", "{{1, 2}, {3, 0}}"),
    ("ArrayReshape[{{1, 2}, {3, 4}}, {4}]", "{1, 2, 3, 4}"),
    ("ArrayFlatten[{{{{1, 2}, {3, 4}}, 0}, {0, {{5, 6}, {7, 8}}}}]", "{{1, 2, 0, 0}, {3, 4, 0, 0}, {0, 0, 5, 6}, {0, 0, 7, 8}}"),
    ("ArrayFlatten[{{{{1}}, {{2}}}}]", "{{1, 2}}"), ("ArrayFlatten[{{{{1}}}, {{{2}}}}]", "{{1}, {2}}"),
    ("PadLeft[{1, 2, 3}, 5]", "{0, 0, 1, 2, 3}"), ("PadLeft[{1, 2, 3}, 2]", "{2, 3}"), ("PadLeft[{{1}, {2, 3}}]", "{{0, 1}, {2, 3}}"),
    ("PadLeft[{a}, 3, x]", "{x, x, a}"),
    ("PadRight[{1, 2, 3}, 5, x]", "{1, 2, 3, x, x}"), ("PadRight[{{1}, {2, 3}}]", "{{1, 0}, {2, 3}}"),
    ("PadRight[{{1, 2}, {3}}, {3, 3}]", "{{1, 2, 0}, {3, 0, 0}, {0, 0, 0}}"), ("PadRight[{a, b}, 1]", "{a}"),
    ("SortBy[{{a, 2}, {b, 1}}, Last]", "{{b, 1}, {a, 2}}"), ("SortBy[{-3, 1, 2, -2}, Abs]", "{1, -2, 2, -3}"),
    ("SortBy[{a, b, c}, 1 &]", "{a, b, c}"), ("SortBy[<|a -> 3, b -> 1|>, Identity]", "<|b -> 1, a -> 3|>"),
    ("ReverseSort[{3, 1, 2}]", "{3, 2, 1}"), ("ReverseSort[{\"a\", \"c\", \"b\"}]", SL(["c", "b", "a"])), ("ReverseSort[{}]", "{}"),
    ("ReverseSortBy[{-3, 1, 2}, Abs]", "{-3, 2, 1}"), ("ReverseSortBy[{{a, 1}, {b, 2}}, Last]", "{{b, 2}, {a, 1}}"), ("ReverseSortBy[{}, f]", "{}"),
    ("Ordering[{c, a, b}]", "{2, 3, 1}"), ("Ordering[{3, 1, 2}, 2]", "{2, 3}"), ("Ordering[{3, 1, 2}, -1]", "{1}"),
    ("Ordering[{1, 2, 3}, All, Greater]", "{3, 2, 1}"),
    ("OrderingBy[{-3, 1, 2}, Abs]", "{2, 3, 1}"), ("OrderingBy[{{a, 2}, {b, 1}}, Last]", "{2, 1}"), ("OrderingBy[{5, 4}, Minus, 1]", "{1}"),
    ("NumericalSort[{Pi, 3, E}]", "{E, 3, Pi}"), ("NumericalSort[{2, 1.5, 1}]", "{1, 1.5, 2}"), ("NumericalSort[{Sqrt[2], 1}]", "{1, Sqrt[2]}"),
    ("DeleteDuplicatesBy[{1, 2, 3, 4}, EvenQ]", "{1, 2}"), ("DeleteDuplicatesBy[{{a, 1}, {b, 1}}, Last]", "{{a, 1}}"),
    ("DeleteDuplicatesBy[{}, f]", "{}"),
    ("Catenate[{{1, 2}, {3}}]", "{1, 2, 3}"), ("Catenate[{<|a -> 1|>, <|b -> 2|>}]", "{1, 2}"), ("Catenate[{}]", "{}"),
    ("ContainsAll[{a, b, c}, {a, c}]", True), ("ContainsAll[{a, b}, {a, d}]", False), ("ContainsAll[{1, 2}, {1.}, SameTest -> Equal]", True),
    ("ContainsAny[{a, b}, {d, b}]", True), ("ContainsAny[{a}, {b}]", False), ("ContainsAny[{a}, {}]", False),
    ("ContainsNone[{a, b}, {c}]", True), ("ContainsNone[{a, b}, {b}]", False), ("ContainsNone[{}, {a}]", True),
    ("ContainsOnly[{a, a, b}, {a, b, c}]", True), ("ContainsOnly[{a, d}, {a, b}]", False), ("ContainsOnly[{}, {a}]", True),
    ("ContainsExactly[{a, b, a}, {b, a}]", True), ("ContainsExactly[{a, b}, {a}]", False), ("ContainsExactly[{}, {}]", True),
    ("DisjointQ[{1, 2}, {3}]", True), ("DisjointQ[{1, 2}, {2}]", False), ("DisjointQ[{}, {}]", True),
    ("IntersectingQ[{1, 2}, {2}]", True), ("IntersectingQ[{1}, {2}]", False), ("IntersectingQ[{}, {1}]", False),
    ("DuplicateFreeQ[{1, 2, 1}]", False), ("DuplicateFreeQ[{1, 2}]", True), ("DuplicateFreeQ[{1, 1.}]", True),
    ("Counts[{a, b, a, c, a}]", "<|a -> 3, b -> 1, c -> 1|>"), ("Counts[{}]", "<||>"), ("Counts[{1, 1.}]", "<|1 -> 1, 1. -> 1|>"),
    ("CountsBy[{1, 2, 3, 4, 5}, OddQ]", "<|True -> 3, False -> 2|>"), ("CountsBy[{\"a\", \"bb\", \"c\"}, StringLength]", "<|1 -> 2, 2 -> 1|>"),
    ("CountsBy[{}, f]", "<||>"),
    ("CountDistinct[{1, 2, 1, 3}]", "3"), ("CountDistinct[{}]", "0"), ("CountDistinct[{a, a}]", "1"),
    ("CountDistinctBy[{1, 2, 3, 4}, EvenQ]", "2"), ("CountDistinctBy[{}, f]", "0"), ("CountDistinctBy[{-1, 1, 2}, Abs]", "2"),
    ("PositionIndex[{a, b, a, c}]", "<|a -> {1, 3}, b -> {2}, c -> {4}|>"), ("PositionIndex[{}]", "<||>"),
    ("PositionIndex[<|x -> 1, y -> 1|>]", "<|1 -> {x, y}|>"),
    ("BinCounts[{1, 3, 2, 1, 4, 5, 6, 2}, {0, 10, 1}]", "{0, 2, 2, 1, 1, 1, 1, 0, 0, 0}"),
    ("BinCounts[{1, 3, 2, 1, 4, 5, 6, 2}, 2]", "{2, 3, 2, 1}"), ("BinCounts[{1.5, 2.5, 7}, {{0, 2, 5, 10}}]", "{1, 1, 1}"),
    ("BinCounts[{1, 2}]", Raises()), ("BinCounts[{1, 3, 2, 1, 4, 5, 6, 2}, {{-Infinity, 2, 5, 7, Infinity}}]", "{2, 4, 2, 0}"),
    ("BinLists[{1, 3, 2, 1, 4}, {0, 6, 2}]", "{{1, 1}, {3, 2}, {4}}"), ("BinLists[{0.5, 1.5}, {0, 2, 1}]", "{{0.5}, {1.5}}"),
    ("BinLists[{5}, {0, 4, 2}]", "{{}, {}}"),
    ("Ratios[{1, 2, 6, 24}]", "{2, 3, 4}"), ("Ratios[{1, 2, 6, 24}, 2]", "{3/2, 4/3}"), ("Ratios[{a, b}]", "{b/a}"),
    ("MinMax[{3, 1, 2}]", "{1, 3}"), ("MinMax[{3, 1, 2}, 1]", "{0, 4}"), ("MinMax[{{1, 5}, {-2}}]", "{-2, 5}"), ("MinMax[{1, 2, 3, 4}, Scaled[1/5]]", "{2/5, 23/5}"),
    ("MapApply[f, {{1, 2}, {3, 4}}]", "{f[1, 2], f[3, 4]}"), ("MapApply[f][{{a}}]", "{f[a]}"), ("MapApply[Plus, {{1, 2}}]", "{3}"),
    ("Scan[Print, {}]", "Null"), ("Reap[Scan[Sow, {a, b}]][[2]]", "{{a, b}}"), ("Reap[Scan[Sow, f[x, y]]][[2]]", "{{x, y}}"),
    ("Through[{f, g}[x]]", "{f[x], g[x]}"), ("Through[p[f, g][x, y]]", "p[f[x, y], g[x, y]]"), ("Through[{Sin, Cos}[0]]", "{0, 1}"),
    ("Operate[p, f[x, y]]", "p[f][x, y]"), ("Operate[p, f[x][y], 2]", "p[f][x][y]"), ("Operate[p, f[a][b][c], 2]", "p[f[a]][b][c]"), ("Operate[p, f[x], 0]", "p[f[x]]"),
    ("Construct[f, x, y]", "f[x, y]"), ("Construct[f]", "f[]"), ("Construct[Plus, 1, 2]", "3"),
    ("NestWhile[#/2 &, 123456, EvenQ]", "1929"), ("NestWhile[# + 1 &, 1, # < 10 &, 1, 5]", "6"),
    ("NestWhile[Mod[2 #, 7] &, 1, Unequal, All]", "1"),
    ("NestWhileList[#/2 &, 48, EvenQ]", "{48, 24, 12, 6, 3}"), ("NestWhileList[Mod[2 #, 7] &, 1, Unequal, All]", "{1, 2, 4, 1}"),
    ("NestWhileList[f, x, False &]", "{x}"),
    ("ComposeList[{f, g}, x]", "{x, f[x], g[f[x]]}"), ("ComposeList[{}, x]", "{x}"), ("ComposeList[{# + 1 &, 2 # &}, 1]", "{1, 2, 4}"),
    ("AllTrue[{2, 4, 6}, EvenQ]", True), ("AllTrue[{2, 3}, EvenQ]", False), ("AllTrue[{}, EvenQ]", True),
    ("AnyTrue[{1, 3, 4}, EvenQ]", True), ("AnyTrue[{}, EvenQ]", False), ("AnyTrue[{1}, EvenQ]", False),
    ("NoneTrue[{1, 3}, EvenQ]", True), ("NoneTrue[{2}, EvenQ]", False), ("NoneTrue[{}, EvenQ]", True),
    ("BlockMap[Total, {1, 2, 3, 4, 5, 6}, 2]", "{3, 7, 11}"), ("BlockMap[f, {1, 2, 3, 4}, 2, 1]", "{f[{1, 2}], f[{2, 3}], f[{3, 4}]}"),
    ("BlockMap[f, {1, 2, 3}, 2]", "{f[{1, 2}]}"),
    ("OrderedQ[{1, 2, 2, 3}]", True), ("OrderedQ[{2, 1}]", False), ("OrderedQ[{3, 2, 1}, Greater]", True), ("OrderedQ[{}]", True),
    ("Order[a, b]", "1"), ("Order[b, a]", "-1"), ("Order[1, 1]", "0"),
    ("Between[2, {1, 3}]", True), ("Between[4, {1, 3}]", False), ("Between[5, {{1, 2}, {4, 6}}]", True), ("Between[{1, 3}][2]", True),
    ("BooleanQ[True]", True), ("BooleanQ[1]", False), ("BooleanQ[x]", False),
    ("x1 = {1}; AppendTo[x1, 2]; x1", "{1, 2}"), ("x2 = {}; AppendTo[x2, a]", "{a}"), ("AppendTo[undefinedSymbolQ, 1]", Raises()),
    ("y1 = {1}; PrependTo[y1, 0]; y1", "{0, 1}"), ("y2 = {}; PrependTo[y2, a]", "{a}"), ("PrependTo[undefinedSymbolR, 1]", Raises()),
    ("Replace[x, x -> 1]", "1"), ("Replace[f[x], x -> 1]", "f[x]"), ("Replace[f[x], x -> 1, {1}]", "f[1]"),
    ("Replace[{1, {2, 3}}, n_Integer :> n^2, {2}]", "{1, {4, 9}}"), ("Replace[x, {{x -> 1}, {x -> 2}}]", "{1, 2}"),
    ("ReplaceList[{a, b, c}, {x__, y__} :> {{x}, {y}}]", "{{{a}, {b, c}}, {{a, b}, {c}}}"),
    ("ReplaceList[{a, b, c}, {___, x_, ___} :> x, 2]", "{a, b}"), ("ReplaceList[{a, a}, {___, x_, ___} :> x]", "{a, a}"),
    ("ContainsExactly[ReplaceList[a + b + c, x_ + y_ :> h[x, y]], "
     "{h[a, b + c], h[b, a + c], h[c, a + b], h[a + b, c], h[a + c, b], h[b + c, a]}]", True),
    ("Length[ReplaceList[a + b + c, x_ + y_ :> h[x, y]]]", "6"),
    # lists.py extensions
    ("Union[{1, 1.0, 2}, SameTest -> Equal]", "{1, 2}"), ("Intersection[{1, 2, 3}, {2., 3.}, SameTest -> Equal]", "{2, 3}"),
    ("Complement[{1, 2, 3}, {2.}, SameTest -> Equal]", "{1, 3}"),
    ("DeleteDuplicates[{1, 2, 3, 4}, Mod[#1, 2] == Mod[#2, 2] &]", "{1, 2}"),
    ("Tally[{1, 2, 3, 4, 5}, Mod[#1, 3] == Mod[#2, 3] &]", "{{1, 2}, {2, 2}, {3, 1}}"),
    ("Partition[{1, 2, 3, 4, 5}, UpTo[2]]", "{{1, 2}, {3, 4}, {5}}"),
    ("GroupBy[{{a, 1}, {b, 2}, {a, 3}}, First -> Last]", "<|a -> {1, 3}, b -> {2}|>"),
    ("GroupBy[{{a, 1}, {b, 2}, {a, 3}}, First -> Last, Total]", "<|a -> 4, b -> 2|>"),
    ("GroupBy[{1, 2, 3, 4, 5, 6}, {EvenQ, # > 3 &}]",
     "<|False -> <|False -> {1, 3}, True -> {5}|>, True -> <|False -> {2}, True -> {4, 6}|>|>"),
    ("GroupBy[<|x -> 1, y -> 2, z -> 3|>, OddQ]", "<|True -> <|x -> 1, z -> 3|>, False -> <|y -> 2|>|>"),
]

# ---------------------------------------------------------------- strings
TEXT = "Hello, world! It's a well-known fact."
STRINGS = [
    ("StringMatchQ[\"apple\", \"a*e\"]", True), ("StringMatchQ[\"apple\", \"a\" ~~ __ ~~ \"e\"]", True),
    ("StringMatchQ[\"Apple\", \"a*\", IgnoreCase -> True]", True), ("StringMatchQ[{\"ab\", \"cd\"}, \"a\" ~~ _]", "{True, False}"),
    ("StringMatchQ[\"aa\", x_ ~~ x_]", True), ("StringMatchQ[\"ab\", x_ ~~ x_]", False),
    ("StringMatchQ[\"abc\", \"a\" ~~ ___ ~~ \"c\"]", True), ("StringMatchQ[\"ab\", \"a\"]", False),
    ("StringMatchQ[\"Hello\", _?UpperCaseQ ~~ __?LowerCaseQ]", True), ("StringMatchQ[\"a\" ~~ _][\"ab\"]", True),
    ("StringContainsQ[\"abcd\", \"b\" ~~ _ ~~ \"d\"]", True), ("StringContainsQ[\"abc\", \"\"]", True),
    ("StringContainsQ[\"ABC\", \"b\", IgnoreCase -> True]", True), ("StringContainsQ[{\"a1\", \"b\"}, DigitCharacter]", "{True, False}"),
    ("StringFreeQ[\"abc\", \"z\"]", True), ("StringFreeQ[\"abc\", \"a\" | \"z\"]", False), ("StringFreeQ[\"a1\", DigitCharacter]", False),
    ("StringStartsQ[\"abc\", LetterCharacter]", True), ("StringStartsQ[\"1abc\", LetterCharacter]", False),
    ("Select[{\"apple\", \"kiwi\", \"avocado\"}, StringStartsQ[\"a\"]]", SL(["apple", "avocado"])),
    ("StringEndsQ[\"abc1\", DigitCharacter]", True), ("StringEndsQ[\"abc\", \"b\"]", False), ("StringEndsQ[\"ABC\", \"c\", IgnoreCase -> True]", True),
    ("StringCases[\"a1b22c333\", DigitCharacter ..]", SL(["1", "22", "333"])),
    ("StringCases[\"a1b22c333\", Shortest[DigitCharacter ..]]", SL(["1", "2", "2", "3", "3", "3"])),
    ("StringCases[\"abcadcacb\", \"a\" ~~ x_ ~~ \"c\" -> x]", SL(["b", "d"])),
    ("StringCases[\"the cat in the hat\", \"the\" | \"hat\"]", SL(["the", "the", "hat"])),
    ("StringCases[\"abc\", RegularExpression[\"[a-b]\"]]", SL(["a", "b"])),
    ("StringCases[\"aXbXc\", Except[\"X\"] ..]", SL(["a", "b", "c"])),
    ("StringCases[\"a1b22\", DigitCharacter .., 1]", SL(["1"])),
    ("StringCases[\"abcd\", RegularExpression[\"(.)(.)\"] -> \"$2$1\"]", SL(["ba", "dc"])),
    ("StringCases[\"aaaa\", \"aa\", Overlaps -> True]", SL(["aa", "aa", "aa"])),
    ("StringCases[\"x = 12, y = 345\", NumberString]", SL(["12", "345"])),
    ("StringCases[\"one two\", WordBoundary ~~ LetterCharacter]", SL(["o", "t"])),
    ("StringCases[\"ab\\ncd\", StartOfLine ~~ _]", SL(["a", "c"])),
    ("StringCases[\"ab\\ncd\", _ ~~ EndOfLine]", SL(["b", "d"])),
    ("StringCases[\"abc\", StartOfString ~~ _]", SL(["a"])), ("StringCases[\"abc\", _ ~~ EndOfString]", SL(["c"])),
    ("StringCases[\"a b\\tc\", WhitespaceCharacter]", SL([" ", "\t"])),
    ("StringCases[\"a,b.c\", PunctuationCharacter]", SL([",", "."])),
    ("StringCases[\"x1F zz\", HexadecimalCharacter ..]", SL(["1F"])),
    ("StringCases[\"a_b1\", WordCharacter ..]", SL(["a", "b1"])),
    ("StringCases[\"<a><b>\", Shortest[\"<\" ~~ __ ~~ \">\"]]", SL(["<a>", "<b>"])),
    ("StringCases[\"<a><b>\", Longest[\"<\" ~~ __ ~~ \">\"]]", SL(["<a><b>"])),
    ("StringCases[\"aaa\", Longest[\"a\" ..]]", SL(["aaa"])),
    ("StringCases[\"abcd\", {\"ab\", \"cd\"}]", SL(["ab", "cd"])),
    ("StringCases[\"a b\", Whitespace]", SL([" "])), ("StringCases[\"a  b\", Whitespace]", SL(["  "])),
    ("StringCases[\"a  b\", StringExpression[\"a\", Whitespace, \"b\"]]", SL(["a  b"])),
    ("StringCases[\"x\", Condition[a_, True]]", Raises()),
    ("StringPosition[\"abXYcdXY\", \"XY\"]", "{{3, 4}, {7, 8}}"), ("StringPosition[\"aaaa\", \"aa\"]", "{{1, 2}, {2, 3}, {3, 4}}"),
    ("StringPosition[\"aaaa\", \"aa\", Overlaps -> False]", "{{1, 2}, {3, 4}}"), ("StringPosition[\"abc\", \"z\"]", "{}"),
    ("StringCount[\"aaaa\", \"aa\"]", "2"), ("StringCount[\"a1b22c333\", DigitCharacter]", "6"),
    ("StringCount[\"abAB\", \"a\", IgnoreCase -> True]", "2"), ("StringCount[\"aaaa\", \"aa\", Overlaps -> True]", "3"),
    ("StringReplace[\"abc abcb abdc\", \"ab\" ~~ _ -> \"X\"]", S("X Xb Xc")), ("StringReplace[\"aaa\", \"a\" -> \"b\", 2]", S("bba")),
    ("StringReplace[\"hello world\", RegularExpression[\"(\\\\w+) (\\\\w+)\"] -> \"$2 $1\"]", S("world hello")),
    ("StringReplace[\"a1b2\", d:DigitCharacter :> ToString[2 ToExpression[d]]]", S("a2b4")),
    ("StringReplace[\"abc\", \"b\" -> x]", "\"a\" ~~ x ~~ \"c\""), ("StringReplace[{\"ab\", \"cb\"}, \"b\" -> \"X\"]", SL(["aX", "cX"])),
    ("StringReplace[\"The cat\", \"the\" -> \"a\", IgnoreCase -> True]", S("a cat")),
    ("StringReplace[\"aXbXc\", \"X\" ~~ x_ :> x <> x]", S("abbcc")),
    ("StringReplace[\"abc\", \"\" -> \"x\"]", Raises()),
    ("StringDelete[\"a1b2c3\", DigitCharacter]", S("abc")), ("StringDelete[\"abcABC\", \"a\", IgnoreCase -> True]", S("bcBC")),
    ("StringDelete[{\"a1\", \"2b\"}, DigitCharacter]", SL(["a", "b"])),
    ("StringTrim[\"  ab  \"]", S("ab")), ("StringTrim[\"xxabxx\", \"x\" ..]", S("ab")),
    ("StringTrim[\"++aa--\", (\"+\" | \"-\") ...]", S("aa")), ("StringTrim[{\" a\", \"b \"}]", SL(["a", "b"])),
    ("StringSplit[\"a b  c\"]", SL(["a", "b", "c"])), ("StringSplit[\" a b \"]", SL(["a", "b"])),
    ("StringSplit[\"a,b,,c\", \",\"]", SL(["a", "b", "", "c"])), ("StringSplit[\",a,b,\", \",\"]", SL(["a", "b"])),
    ("StringSplit[\",a,b,\", \",\", All]", SL(["", "a", "b", ""])), ("StringSplit[\"a1b22c\", DigitCharacter ..]", SL(["a", "b", "c"])),
    ("StringSplit[\"a-b+c\", {\"-\", \"+\"}]", SL(["a", "b", "c"])),
    ("StringSplit[\"a-b+c\", \"-\" | \"+\" -> \"op\"]", SL(["a", "op", "b", "op", "c"])),
    ("StringSplit[\"a b c d\", \" \", 2]", SL(["a", "b c d"])), ("StringSplit[{\"a b\", \"c d\"}]", "{{\"a\", \"b\"}, {\"c\", \"d\"}}"),
    ("StringSplit[\"aXbxc\", \"x\", IgnoreCase -> True]", SL(["a", "b", "c"])),
    ("StringSplit[\"a:b:c\", \":\" -> \":\"]", SL(["a", ":", "b", ":", "c"])),
    ("StringSplit[\"\"]", "{}"),
    ("StringRiffle[{\"a\", \"b\", \"c\"}]", S("a b c")), ("StringRiffle[{\"a\", \"b\", \"c\"}, \", \"]", S("a, b, c")),
    ("StringRiffle[{\"a\", \"b\"}, {\"(\", \", \", \")\"}]", S("(a, b)")), ("StringRiffle[{{\"a\", \"b\"}, {\"c\", \"d\"}}]", S("a b\nc d")),
    ("StringRiffle[{{\"a\", \"b\"}, {\"c\", \"d\"}}, \";\", \"-\"]", S("a-b;c-d")), ("StringRiffle[{1, 2, 3}, \"+\"]", S("1+2+3")),
    ("StringRiffle[{{{\"a\", \"b\"}, {\"c\"}}, {{\"d\"}}}]", S("a b\nc\n\nd")),
    ("StringPartition[\"abcdefg\", 3]", SL(["abc", "def"])), ("StringPartition[\"abcdefg\", 3, 1]", SL(["abc", "bcd", "cde", "def", "efg"])),
    ("StringPartition[\"abcdefg\", UpTo[3]]", SL(["abc", "def", "g"])),
    ("StringRotateLeft[\"abcdef\", 2]", S("cdefab")), ("StringRotateLeft[\"abc\"]", S("bca")), ("StringRotateLeft[\"abc\", -1]", S("cab")),
    ("StringRotateRight[\"abcdef\", 2]", S("efabcd")), ("StringRotateRight[\"abc\"]", S("cab")), ("StringRotateRight[\"\", 3]", S("")),
    ("StringTakeDrop[\"abcdef\", 2]", SL(["ab", "cdef"])), ("StringTakeDrop[\"abcdef\", -2]", SL(["ef", "abcd"])),
    ("StringTakeDrop[\"abcdef\", {2, 3}]", SL(["bc", "adef"])),
    ("StringReplacePart[\"abcdefg\", \"XYZ\", {2, 4}]", S("aXYZefg")),
    ("StringReplacePart[\"abcdefg\", {\"X\", \"Y\"}, {{1, 1}, {-2, -1}}]", S("XbcdeY")),
    ("StringReplacePart[\"abcdefg\", \"-\", {{1, 2}, {5, 6}}]", S("-cd-g")),
    ("ToUpperCase[\"abc Def\"]", S("ABC DEF")), ("ToUpperCase[{\"a\", \"b\"}]", SL(["A", "B"])), ("ToUpperCase[\"\u00e9\"]", S("\u00c9")),
    ("ToLowerCase[{\"AB\", \"Cd\"}]", SL(["ab", "cd"])), ("ToLowerCase[\"ABC\"]", S("abc")), ("ToLowerCase[\"a1!\"]", S("a1!")),
    ("Capitalize[\"hello world\"]", S("Hello world")), ("Capitalize[\"hello world\", \"AllWords\"]", S("Hello World")),
    ("Capitalize[\"this is a sentence\", \"LongWords\"]", S("This is a Sentence")), ("Capitalize[\"x\", \"TitleCase\"]", Raises()),
    ("Decapitalize[\"Hello\"]", S("hello")), ("Decapitalize[\"ABC\"]", S("aBC")), ("Decapitalize[\"\"]", S("")),
    ("UpperCaseQ[\"ABC\"]", True), ("UpperCaseQ[\"AbC\"]", False), ("UpperCaseQ[\"AB1\"]", False),
    ("LowerCaseQ[\"abc\"]", True), ("LowerCaseQ[\"aBc\"]", False), ("LowerCaseQ[\"a b\"]", False),
    ("LetterQ[\"abc\"]", True), ("LetterQ[\"ab1\"]", False), ("LetterQ[\"\u00e4\u00e9\"]", True), ("LetterQ[\"two words\"]", False),
    ("DigitQ[\"123\"]", True), ("DigitQ[\"12.3\"]", False), ("DigitQ[\"\"]", True),
    ("PrintableASCIIQ[\"a b~\"]", True), ("PrintableASCIIQ[\"a\\tb\"]", False), ("PrintableASCIIQ[\"\u00e9\"]", False),
    ("ToExpression[\"1 + 2\"]", "3"), ("ToExpression[\"{1, 2}^2\"]", "{1, 4}"), ("ToExpression[\"1 +\"]", "$Failed"),
    ("ToExpression[\"1 + 1\", InputForm, Hold]", "Hold[1 + 1]"),
    ("IntegerString[255, 16]", S("ff")), ("IntegerString[-42]", S("42")), ("IntegerString[5, 2, 8]", S("00000101")),
    ("IntegerString[12345, 10, 3]", S("345")), ("IntegerString[1999, \"Roman\"]", S(roman(1999))), ("IntegerString[5, 37]", Raises()),
    ("RomanNumeral[2023]", S(roman(2023))), ("RomanNumeral[0]", S("N")), ("RomanNumeral[-7]", S("VII")),
    ("RomanNumeral[{1, 4, 3999}]", SL(["I", "IV", roman(3999)])),
    ("FromRomanNumeral[\"MMXXIV\"]", "2024"), ("FromRomanNumeral[\"N\"]", "0"), ("FromRomanNumeral[\"IIII\"]", Raises()),
    ("TextWords[\"" + TEXT + "\"]", SL(["Hello", "world", "It's", "a", "well-known", "fact"])),
    ("TextWords[\"one, two\", 1]", SL(["one"])), ("TextWords[\"\"]", "{}"),
    ("Alphabet[]", SL("abcdefghijklmnopqrstuvwxyz")), ("Length[Alphabet[\"English\"]]", "26"), ("Alphabet[\"Greek\"]", Raises()),
    ("LetterNumber[\"c\"]", "3"), ("LetterNumber[\"aBz!\"]", "{1, 2, 26, 0}"), ("LetterNumber[{\"a\", \"Z\"}]", "{1, 26}"),
    ("FromLetterNumber[4]", S("d")), ("FromLetterNumber[{4, 10, 17}]", SL(["d", "j", "q"])), ("FromLetterNumber[-1]", S("z")),
    ("FromLetterNumber[0]", S(" ")),
    ("RemoveDiacritics[\"caf\u00e9 na\u00efve\"]", S("cafe naive")), ("RemoveDiacritics[\"\u00c5ngstr\u00f6m\"]", S("Angstrom")),
    ("RemoveDiacritics[\"abc\"]", S("abc")),
    ("CharacterRange[\"a\", \"e\"]", SL("abcde")), ("CharacterRange[48, 52]", SL("01234")), ("CharacterRange[\"e\", \"a\"]", "{}"),
    ("SymbolName[x]", S("x")), ("SymbolName[Pi]", S("Pi")), ("SymbolName[True]", S("True")),
    ("Symbol[\"Pi\"]", "Pi"), ("Symbol[\"x\"]", "x"), ("Symbol[\"1x\"]", Raises()),
    ("EditDistance[\"kitten\", \"sitting\"]", str(levenshtein("kitten", "sitting"))),
    ("EditDistance[\"abc\", \"ABC\", IgnoreCase -> True]", "0"), ("EditDistance[{1, 2, 3}, {1, 3}]", "1"),
    ("EditDistance[\"\", \"abc\"]", "3"),
    ("HammingDistance[\"karolin\", \"kathrin\"]", "3"), ("HammingDistance[{1, 0, 0, 1, 1}, {0, 0, 1, 0, 1}]", "3"),
    ("HammingDistance[\"ab\", \"AB\", IgnoreCase -> True]", "0"), ("HammingDistance[\"ab\", \"abc\"]", Raises()),
    ("LongestCommonSubsequence[\"AAABBBBCCCCC\", \"CCCBBBAAABABA\"]", S("AAAB")),
    ("LongestCommonSubsequence[\"abcdef\", \"zcdemf\"]", S("cde")), ("LongestCommonSubsequence[{1, 2, 3}, {2, 3, 1}]", "{2, 3}"),
    ("LongestCommonSequence[\"abcdef\", \"acf\"]", S("acf")), ("LongestCommonSequence[\"AGGTAB\", \"GXTXAYB\"]", S("GTAB")),
    ("LongestCommonSequence[{1, 2, 3, 4}, {1, 3, 4}]", "{1, 3, 4}"),
    ("PalindromeQ[\"racecar\"]", True), ("PalindromeQ[{1, 2, 1}]", True), ("PalindromeQ[12321]", True), ("PalindromeQ[123]", False),
    ("PalindromeQ[\"ab\"]", False),
]

# ---------------------------------------------------------------- associations
ASSOC = [
    ("<|a -> 1, a -> 2, b -> 3|>", "<|a -> 2, b -> 3|>"), ("Association[{a -> 1, b -> 2}]", "<|a -> 1, b -> 2|>"),
    ("Association[<|a -> 1|>, b -> 2]", "<|a -> 1, b -> 2|>"), ("Association[1]", Raises()),
    ("<|a :> 1 + 1|>[a]", "2"), ("<|a -> 1|>[z]", "Missing[\"KeyAbsent\", z]"), ("<|a -> <|b -> 5|>|>[a][b]", "5"),
    ("AssociationQ[<|a -> 1|>]", True), ("AssociationQ[{a -> 1}]", False), ("AssociationQ[<||>]", True),
    ("Keys[<|a -> 1, b -> 2|>]", "{a, b}"), ("Keys[{a -> 1, b -> 2}]", "{a, b}"), ("Keys[{<|a -> 1|>, <|b -> 2|>}]", "{{a}, {b}}"),
    ("Keys[<|a -> 1|>, f]", "{f[a]}"),
    ("Values[<|a -> 1, b -> 2|>]", "{1, 2}"), ("Values[<|a -> 1, b -> 2|>, f]", "{f[1], f[2]}"), ("Values[{a -> 1, b :> 1 + 1}]", "{1, 2}"),
    ("KeyTake[<|a -> 1, b -> 2, c -> 3|>, {c, a, z}]", "<|c -> 3, a -> 1|>"),
    ("KeyTake[{<|a -> 1, b -> 2|>, <|a -> 3|>}, {a}]", "{<|a -> 1|>, <|a -> 3|>}"), ("KeyTake[{a -> 1, b -> 2}, b]", "{b -> 2}"),
    ("KeyDrop[<|a -> 1, b -> 2, c -> 3|>, {a, c}]", "<|b -> 2|>"), ("KeyDrop[<|a -> 1, b -> 2|>, a]", "<|b -> 2|>"),
    ("KeyDrop[{a -> 1, b -> 2}, a]", "{b -> 2}"),
    ("KeySort[<|c -> 1, a -> 2, b -> 3|>]", "<|a -> 2, b -> 3, c -> 1|>"), ("KeySort[<|1 -> x, 3 -> y, 2 -> z|>, Greater]", "<|3 -> y, 2 -> z, 1 -> x|>"),
    ("KeySort[<||>]", "<||>"),
    ("KeySortBy[<|-3 -> a, 1 -> b, 2 -> c|>, Abs]", "<|1 -> b, 2 -> c, -3 -> a|>"), ("KeySortBy[<|b -> 1, a -> 2|>, Identity]", "<|a -> 2, b -> 1|>"),
    ("KeySortBy[<||>, f]", "<||>"),
    ("KeySelect[<|1 -> a, 2 -> b, 3 -> c|>, OddQ]", "<|1 -> a, 3 -> c|>"), ("KeySelect[<|a -> 1|>, IntegerQ]", "<||>"),
    ("KeySelect[<|1 -> a|>, IntegerQ]", "<|1 -> a|>"),
    ("KeyMap[f, <|a -> 1, b -> 2|>]", "<|f[a] -> 1, f[b] -> 2|>"), ("KeyMap[1 &, <|a -> 1, b -> 2|>]", "<|1 -> 2|>"),
    ("KeyMap[ToUpperCase, <|\"a\" -> 1|>]", "<|\"A\" -> 1|>"),
    ("KeyValueMap[f, <|a -> 1, b -> 2|>]", "{f[a, 1], f[b, 2]}"), ("KeyValueMap[#1 + #2 &, <|1 -> 10, 2 -> 20|>]", "{11, 22}"),
    ("KeyValueMap[f, <||>]", "{}"),
    ("Lookup[<|a -> 1, b -> 2|>, a]", "1"), ("Lookup[<|a -> 1|>, z]", "Missing[\"KeyAbsent\", z]"), ("Lookup[<|a -> 1|>, z, 0]", "0"),
    ("Lookup[<|a -> 1, b -> 2|>, {a, b, c}]", "{1, 2, Missing[\"KeyAbsent\", c]}"), ("Lookup[{<|a -> 1|>, <|a -> 2|>}, a]", "{1, 2}"),
    ("Lookup[{a -> 1}, a]", "1"), ("Lookup[a][<|a -> 5|>]", "5"),
    ("AssociationMap[f, {a, b}]", "<|a -> f[a], b -> f[b]|>"), ("AssociationMap[Reverse, <|a -> 1, b -> 2|>]", "<|1 -> a, 2 -> b|>"),
    ("AssociationMap[f, {}]", "<||>"), ("AssociationMap[f, <|a -> 1|>]", Raises()),
    ("AssociationThread[{a, b}, {1, 2}]", "<|a -> 1, b -> 2|>"), ("AssociationThread[{a, b} -> {1, 2}]", "<|a -> 1, b -> 2|>"),
    ("AssociationThread[{a, b}, {1}]", Raises()), ("AssociationThread[{a, a}, {1, 2}]", "<|a -> 2|>"),
    ("Merge[{<|a -> 1, b -> 2|>, <|a -> 3|>}, Total]", "<|a -> 4, b -> 2|>"),
    ("Merge[{<|a -> 1|>, <|b -> 2|>, <|a -> 5|>}, f]", "<|a -> f[{1, 5}], b -> f[{2}]|>"),
    ("Merge[{<|a -> 1|>, <|a -> 2|>}, Last]", "<|a -> 2|>"), ("Merge[{}, f]", "<||>"),
    ("KeyExistsQ[<|a -> 1|>, a]", True), ("KeyExistsQ[<|a -> 1|>, b]", False), ("KeyExistsQ[{a -> 1}, a]", True),
    ("KeyMemberQ[<|1 -> a, x -> b|>, _Integer]", True), ("KeyMemberQ[<|x -> b|>, _Integer]", False), ("KeyMemberQ[<|x -> 1|>, x]", True),
    ("KeyFreeQ[<|1 -> a|>, _String]", True), ("KeyFreeQ[<|1 -> a|>, _Integer]", False), ("KeyFreeQ[<||>, _]", True),
    ("KeyComplement[{<|a -> 1, b -> 2, c -> 3|>, <|b -> 5|>}]", "<|a -> 1, c -> 3|>"),
    ("KeyComplement[{<|a -> 1|>, <|a -> 2|>}]", "<||>"), ("KeyComplement[{<|a -> 1|>}]", "<|a -> 1|>"),
    ("KeyIntersection[{<|a -> 1, b -> 2|>, <|b -> 3, a -> 4, c -> 5|>}]", "{<|a -> 1, b -> 2|>, <|a -> 4, b -> 3|>}"),
    ("KeyIntersection[{<|a -> 1|>, <|b -> 2|>}]", "{<||>, <||>}"), ("KeyIntersection[{}]", "{}"),
    ("KeyUnion[{<|a -> 1|>, <|b -> 2|>}]", "{<|a -> 1, b -> Missing[\"KeyAbsent\", b]|>, <|a -> Missing[\"KeyAbsent\", a], b -> 2|>}"),
    ("KeyUnion[{<|a -> 1|>, <|b -> 2|>}, f]", "{<|a -> 1, b -> f[b]|>, <|a -> f[a], b -> 2|>}"), ("KeyUnion[{<|a -> 1|>}]", "{<|a -> 1|>}"),
    ("as1 = <|a -> 1|>; AssociateTo[as1, b -> 2]; as1", "<|a -> 1, b -> 2|>"), ("as2 = <|a -> 1|>; AssociateTo[as2, a -> 9]", "<|a -> 9|>"),
    ("as3 = {1}; AssociateTo[as3, a -> 1]", Raises()),
    ("as4 = <|a -> 1, b -> 2|>; KeyDropFrom[as4, a]; as4", "<|b -> 2|>"), ("as5 = <|a -> 1|>; KeyDropFrom[as5, {a, z}]", "<||>"),
    ("KeyDropFrom[undefinedSymbolK, a]", Raises()),
    ("MissingQ[Missing[\"KeyAbsent\", x]]", True), ("MissingQ[1]", False), ("MissingQ[Missing[]]", True),
    ("DeleteMissing[{1, Missing[], 2}]", "{1, 2}"), ("DeleteMissing[<|a -> 1, b -> Missing[]|>]", "<|a -> 1|>"),
    ("DeleteMissing[{1, {Missing[], 2}}, 2]", "{1, {2}}"),
]

EXTRA = [
    ("List[1, Nothing, 2]", "{1, 2}"), ("List[]", "{}"), ("Head[List[a]]", "List"),
    ("MapAll[f, {}]", "f[{}]"), ("Apply[f, {a, b}]", "f[a, b]"), ("Less[1, 1]", False),
    ("Composition[f][x]", "f[x]"), ("RightComposition[f, g, h][x]", "h[g[f[x]]]"),
    ("MatchQ[{}, {RepeatedNull[_Integer]}]", True), ("MatchQ[{1, a}, {RepeatedNull[_Integer]}]", False),
    ("MatchQ[f[a], f[_, Optional[_, 0]]]", True), ("MatchQ[f[a, b, c], f[_, Optional[_, 0]]]", False),
    ("StringMatchQ[\"ab\", StringExpression[\"a\", \"b\"]]", True), ("StringCases[\"aXb\", StringExpression[_, \"X\"]]", SL(["aX"])),
    ("StringReplace[\"aaa\", Shortest[\"a\" ..] -> \"b\"]", S("bbb")),
    ("StringReplace[\"aaa\", Longest[\"a\" ..] -> \"b\"]", S("b")),
    ("StringCount[\"a b-c\", WordCharacter]", "3"), ("StringMatchQ[\"x_1\", WordCharacter ..]", False),
    ("StringCount[\"a b\\nc\", WhitespaceCharacter]", "2"), ("StringSplit[\"a b\", WhitespaceCharacter]", SL(["a", "b"])),
    ("StringCount[\"a!b?\", PunctuationCharacter]", "2"), ("StringDelete[\"a,b\", PunctuationCharacter]", S("ab")),
    ("StringMatchQ[\"ff09\", HexadecimalCharacter ..]", True), ("StringMatchQ[\"fg\", HexadecimalCharacter ..]", False),
    ("StringReplace[\"aa\", StartOfString ~~ \"a\" -> \"b\"]", S("ba")), ("StringMatchQ[\"ab\", StartOfString ~~ \"ab\"]", True),
    ("StringReplace[\"aa\", \"a\" ~~ EndOfString -> \"b\"]", S("ab")), ("StringContainsQ[\"ab\", \"a\" ~~ EndOfString]", False),
    ("StringCases[\"a\\nb\\nc\", StartOfLine ~~ x_ :> x]", SL(["a", "b", "c"])), ("StringReplace[\"a\\nb\", StartOfLine ~~ x_ :> ToUpperCase[x]]", S("A\nB")),
    ("StringReplace[\"a\\nb\", x_ ~~ EndOfLine :> ToUpperCase[x]]", S("A\nB")), ("StringCases[\"ab\", _ ~~ EndOfLine]", SL(["b"])),
    ("StringCases[\"one two\", WordBoundary ~~ x_ :> x]", SL(["o", " ", "t"])), ("StringReplace[\"cat concat\", WordBoundary ~~ \"cat\" -> \"X\"]", S("X concat")),
    ("StringMatchQ[\"3.14\", NumberString]", True), ("StringMatchQ[\"3.1.4\", NumberString]", False),
]

# Part 2: outputs checked against the reference-page example outputs
PART2 = [
    ("AllMatch[{x^2, x^3, x^5, x^6}, x^_]", True), ("AllMatch[{x^2, y^3}, x^_]", False), ("AllMatch[_Integer][{1, 2}]", True),
    ("AnyMatch[{a, 1}, _Integer]", True), ("AnyMatch[{a, b}, _Integer]", False), ("AnyMatch[_Integer][{a, 1}]", True),
    ("NoneMatch[{a, b}, _Integer]", True), ("NoneMatch[{a, 1}, _Integer]", False), ("NoneMatch[{}, _]", True),
    ("Comap[{f, g, h}, x]", "{f[x], g[x], h[x]}"), ("Comap[{Sin, Cos, Tan}, Pi/3]", "{Sqrt[3]/2, 1/2, Sqrt[3]}"),
    ("Comap[<|a -> f, b -> g|>, x]", "<|a -> f[x], b -> g[x]|>"),
    ("ComapApply[{f, g, h}, {1, 2}]", "{f[1, 2], g[1, 2], h[1, 2]}"), ("ComapApply[<|a -> f|>, {x, y}]", "<|a -> f[x, y]|>"),
    ("ComapApply[{Plus, Times}, {2, 3}]", "{5, 6}"),
    ("FoldPairList[{p[#1, #2], q[#1, #2]} &, u, {1, 2, 3}]", "{p[u, 1], p[q[u, 1], 2], p[q[q[u, 1], 2], 3]}"),
    ("FoldPairList[{p[#1, #2], q[#1, #2]} &, u, {1, 2}, g]", "{g[{p[u, 1], q[u, 1]}], g[{p[q[u, 1], 2], q[q[u, 1], 2]}]}"),
    ("FoldPairList[{p[#1, #2], q[#1, #2]} &, {a, b, c}]", "{p[a, b], p[q[a, b], c]}"),
    ("FoldPairList[TakeDrop, {a, b, c, d, e, f}, {1, 3, 2}]", "{{a}, {b, c, d}, {e, f}}"),
    ("FoldPair[{p[#1, #2], q[#1, #2]} &, u, {1, 2, 3}]", "p[q[q[u, 1], 2], 3]"),
    ("FoldPair[{p[#1, #2], q[#1, #2]} &, u, {1, 2, 3}, Last]", "q[q[q[u, 1], 2], 3]"),
    ("FoldPair[TakeDrop, {a, b, c, d, e, f}, {1, 3, 2}]", "{e, f}"),
    ("ReverseApplied[f][a, b, c]", "f[c, b, a]"), ("ReverseApplied[List][1, 2]", "{2, 1}"), ("ReverseApplied[Subtract][1, 3]", "2"),
    ("LexicographicSort[{{c, d}, {a, b, c, d}}]", "{{a, b, c, d}, {c, d}}"),
    ("LexicographicSort[{{b, c}, {a, b, c}, {b, b, a}}]", "{{a, b, c}, {b, b, a}, {b, c}}"),
    ("LexicographicSort[{h[b, c], h[a, b, c], h[b, b, a]}]", "{h[a, b, c], h[b, b, a], h[b, c]}"),
    ("LexicographicSort[{{1, 2}, {1}}]", "{{1}, {1, 2}}"),
    ("LeafCount[1 + a + b^2]", "6"), ("LeafCount[{f[][x], g[y][x], h[g[y]][x]}]", "10"),
    ("LeafCount[{f[][x], g[y][x], h[g[y]][x]}, Heads -> False]", "3"), ("LeafCount[1/2]", "3"), ("LeafCount[x]", "1"),
    ("IntegerLength[12345]", "5"), ("IntegerLength[0]", "0"), ("IntegerLength[255, 16]", "2"), ("IntegerLength[-100]", "3"),
    ("IntegerReverse[123456]", "654321"), ("IntegerReverse[2015, 2]", "2015"), ("IntegerReverse[1234, 10, 6]", "432100"),
    ("IntegerReverse[1234]", "4321"),
    ("StringExtract[\"a bbb  cccc aa   d\", 2]", S("bbb")), ("StringExtract[\"a bbb  cccc aa   d\", 2 ;; 4]", SL(["bbb", "cccc", "aa"])),
    ("StringExtract[\"a--bbb--ccc--dddd\", \"--\" -> 3]", S("ccc")), ("StringExtract[\"a bbb  cccc aa   d\", -1]", S("d")),
    ("StringExtract[\"a bbb  cccc aa   d\", {1, 3}]", SL(["a", "cccc"])),
    ("StringReplaceList[\"cccc\", \"c\" -> \"XYX\"]", SL(["XYXccc", "cXYXcc", "ccXYXc", "cccXYX"])),
    ("StringReplaceList[\"abcdeabacde\", {\"abc\" -> \"X\", \"cde\" -> \"Y\"}]", SL(["Xdeabacde", "abYabacde", "abcdeabaY"])),
    ("StringReplaceList[\"abcd acbd\", RegularExpression[\"[ab]\"] -> \"XX\"]",
     SL(["XXbcd acbd", "aXXcd acbd", "abcd XXcbd", "abcd acXXd"])),
    ("BinCounts[{1, 3, 2, 1, 4, 5, 6, 2}, {{-Infinity, 2, 5, 7, Infinity}}]", "{2, 4, 2, 0}"),
]

# ---------------------------------------------------------------- independent cross-checks
def _xcheck_cases():
    out = []
    words = ["kitten", "sitting", "flaw", "lawn", "intention", "execution", "", "a"]
    for a, b in itertools.combinations(words, 2):
        out.append((f"EditDistance[{S(a)}, {S(b)}]", str(levenshtein(a, b))))
    text = "The year 1987 and 2024, then 7."
    out.append((f"StringCases[{S(text)}, DigitCharacter ..]", SL(re.findall(r"[0-9]+", text))))
    out.append((f"StringSplit[{S(text)}]", SL(text.split())))
    out.append((f"StringReplace[{S(text)}, DigitCharacter -> \"#\"]", S(re.sub(r"[0-9]", "#", text))))
    s = "\u00e9l\u00e8ve na\u00efve"
    plain = unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)))
    out.append((f"RemoveDiacritics[{S(s)}]", S(plain)))
    for n in (1, 4, 9, 14, 40, 90, 400, 1994, 2026, 3888):
        out.append((f"RomanNumeral[{n}]", S(roman(n))))
        out.append((f"FromRomanNumeral[{S(roman(n))}]", str(n)))
    for n, b in ((0, 2), (255, 2), (255, 36), (10 ** 20, 7)):
        digits, k = "", n
        while True:
            digits = "0123456789abcdefghijklmnopqrstuvwxyz"[k % b] + digits
            k //= b
            if k == 0:
                break
        out.append((f"IntegerString[{n}, {b}]", S(digits)))
    xs = [3, 1, 4, 1, 5, 9, 2, 6]
    out.append((f"Ordering[{{{', '.join(map(str, xs))}}}]", "{" + ", ".join(str(i + 1) for i in sorted(range(len(xs)), key=lambda i: xs[i])) + "}"))
    out.append(("Subsequences[Range[4], {2}]", "{" + ", ".join("{" + f"{i}, {i + 1}" + "}" for i in range(1, 4)) + "}"))
    return out


XCHECK = _xcheck_cases()

CASES = KERNEL + LISTS + STRINGS + ASSOC + EXTRA + PART2 + XCHECK


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_batch1(case):
    check(case)
