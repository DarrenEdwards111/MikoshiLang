"""Heads that the legacy library also defined but which are evaluated by the
kernel builtins before any library rule runs (legacy rules were dead code).
They are covered by the kernel test suite and spot-checked in
tests/library/test_kernelheads.py."""
from .core import record

KERNEL = """Apart Append Cancel D DSolve Det Diff Dot DotProduct Drop EvenQ First Flatten IntegerQ Inverse Join
Last Length Limit MemberQ Most N Negative NonNegative NumberQ OddQ Part PartialD Plus Position Positive Power
Prepend PrimeQ Range Rest Reverse Series Solve Sort StringJoin StringLength Take Times Together Total
Transpose""".split()

for _h in KERNEL:
    record(_h, "kernel", "kernel", "evaluated by the kernel builtin; legacy library rule was unreachable")


# Names that were on the quarantine list although the kernel already implements them correctly
# (the kernel runs before the quarantine check); reviewed, tested and removed from quarantined.json.
KERNEL_FROM_QUARANTINE = "AtomQ ListQ StringQ ToString ReplaceAll Distance2D".split()

for _h in KERNEL_FROM_QUARANTINE:
    record(_h, "kernel", "kernel", "kernel builtin (was listed as quarantined although the kernel evaluates it "
           "first); Wolfram results confirmed" if _h != "Distance2D" else
           "kernel builtin, Euclidean distance of two 2D points (not a Wolfram name; equals EuclideanDistance)")


# Kernel builtins extended in coverage batch 1 (kernel_lists.py): Wolfram level specifications, Span/UpTo
# sequence specifications, associations, operator forms and options.  Differential cases are in
# tests/library/test_wolfram_batch1.py.
BATCH1_FIXED = {
    "List": "drops Nothing and splices Splice[...]",
    "Length": "associations count their entries; atoms have length 0",
    "AtomQ": "associations and rationals are atoms",
    "Part": "Span (i;;j;;k), All, position lists, negative indices, Key[k] and string keys of associations",
    "First": "First[expr, default]; associations give their first value",
    "Last": "Last[expr, default]; associations give their last value",
    "Rest": "any head; empty expressions raise", "Most": "any head; empty expressions raise",
    "Append": "associations merge the new rule", "Prepend": "associations merge the new rule; any head",
    "Join": "Join[..., n] at level n; associations merge (later values win)",
    "Reverse": "Reverse[expr, n] and Reverse[expr, {n1, n2}]",
    "Take": "UpTo[n], {m, n, s} and one specification per level",
    "Drop": "UpTo[n], {m, n, s} and one specification per level",
    "Sort": "associations sort by value; ordering function p", "Total": "Total[list, n], {n}, {n1, n2}, Infinity; associations",
    "Flatten": "Flatten[expr, n], Flatten[expr, n, h] and dimension groups {{2}, {1}}",
    "Transpose": "permutation specifications (including repeated levels for diagonals)",
    "Cases": "level specifications and a maximum count", "DeleteCases": "level specifications and a maximum count",
    "Count": "level specifications", "Position": "level specifications, a maximum count, Heads -> False",
    "MemberQ": "level specifications", "FreeQ": "level specifications, Heads -> False",
    "Map": "level specifications (n, {n}, {m, n}, negative), associations, operator form Map[f]",
    "MapAll": "f //@ expr", "MapIndexed": "level specifications; associations index by Key[k]",
    "Apply": "level specifications (f @@@ expr); an association gives f[values]",
    "Select": "Select[list, crit, n]; associations; operator form",
    "Thread": "Thread[f[...], h] and Thread[expr, h, n]; rules thread",
    "Normal": "associations become rule lists at the first level only; Normal[expr, h]",
    "Range": "rational, real and symbolic steps",
    "Equal": "more than two arguments chain", "Unequal": "more than two arguments need all pairs distinct",
    "Less": "more than two arguments chain",
}
for _h, _n in BATCH1_FIXED.items():
    record(_h, "kernel", "fixed", "kernel builtin extended in batch 1: " + _n)
