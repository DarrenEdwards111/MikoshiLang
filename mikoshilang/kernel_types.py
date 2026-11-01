"""Hand-written type signatures of the kernel built-ins (types pillar; CONVENTIONS §2 vocabulary).

Kernel built-ins (``Plus``, ``Map``, ``Length``, ...) are implemented in ``mikoshilang.kernel`` without Python
annotations, so their signatures are written here. Each entry is a list of overloads
``(positional parameter types, type of further arguments or None, result type)``.

How to read them:

* A numeric type (``Number``, ``Real``, ``Integer``) still accepts symbolic arguments: ``x``, ``n + 1`` and
  ``f[y]`` are "not known yet", never mismatches. Only *concrete* values of another kind (strings, ``True``,
  lists, associations, rules) are reported. ``| Expr`` is therefore only written where concrete values of
  other kinds are valid too.
* Arithmetic threads over lists and associations, so those are part of its argument type.
* Results: predicates are ``Boolean`` (an unevaluated ``x > 1`` is symbolic, not a mismatch).
* Held built-ins (``If``, ``Table``, ``Set``, ``Module``, ...) receive unevaluated code and are not listed:
  ``TypeSignature`` reports them as ``{RepeatedNull[Held]} -> Expr``.
"""
from __future__ import annotations

ARITH = "Number | ListOf[Expr] | Assoc"
ORDER = "Real | ListOf[Expr]"
LEVEL = "Expr"                       # level specifications: n, {n}, {m, n}, Infinity, All

_E, _B, _I, _S, _F = "Expr", "Boolean", "Integer", "String", "Fn"

KERNEL_SIGNATURES = {
    # arithmetic
    "Plus": [([], ARITH, _E)], "Times": [([], ARITH, _E)],
    "Subtract": [([ARITH, ARITH], None, _E)], "Divide": [([ARITH, ARITH], None, _E)],
    "Power": [([ARITH, ARITH], None, _E)],
    # comparisons and predicates
    "Equal": [([], _E, _B)], "Unequal": [([], _E, _B)], "SameQ": [([], _E, _B)], "UnsameQ": [([], _E, _B)],
    "Greater": [([], ORDER, _B)], "GreaterEqual": [([], ORDER, _B)], "Less": [([], ORDER, _B)],
    "LessEqual": [([], ORDER, _B)],
    "Positive": [([_E], None, _B)], "Negative": [([_E], None, _B)], "NonNegative": [([_E], None, _B)],
    "NonPositive": [([_E], None, _B)],
    "AtomQ": [([_E], None, _B)], "EvenQ": [([_E], None, _B)], "OddQ": [([_E], None, _B)],
    "IntegerQ": [([_E], None, _B)], "ListQ": [([_E], None, _B)], "NumberQ": [([_E], None, _B)],
    "NumericQ": [([_E], None, _B)], "StringQ": [([_E], None, _B)], "TrueQ": [([_E], None, _B)],
    "PrimeQ": [([_E], _E, _B)], "Not": [([_B], None, _B)],
    "MatchQ": [([_E], _E, _B)], "MemberQ": [([_E], _E, _B)], "FreeQ": [([_E], _E, _B)],
    # structure and lists
    "List": [([], _E, "ListOf[Expr]")], "Sequence": [([], _E, _E)], "Identity": [([_E], None, _E)],
    "Evaluate": [([], _E, _E)], "ReleaseHold": [([_E], None, _E)], "Head": [([_E], _E, _E)],
    "Length": [([_E], None, _I)], "Part": [([_E], _E, _E)],
    "First": [([_E], _E, _E)], "Last": [([_E], _E, _E)], "Rest": [([_E], None, _E)], "Most": [([_E], None, _E)],
    "Append": [([_E, _E], None, _E)], "Prepend": [([_E, _E], None, _E)], "Join": [([], _E, _E)],
    "Reverse": [([_E], _E, _E)], "Take": [([_E], _E, _E)], "Drop": [([_E], _E, _E)],
    "Sort": [([_E], _E, _E)], "Total": [([_E], _E, _E)], "Flatten": [([_E], _E, _E)],
    "Transpose": [([_E], _E, _E)], "Normal": [([_E], _E, _E)],
    "Range": [(["Number"], "Number", "ListOf[Expr]")],
    "Cases": [([_E], _E, _E)], "DeleteCases": [([_E], _E, _E)], "Count": [([_E], _E, _I)],
    "Position": [([_E], _E, "ListOf[Expr]")], "Select": [([_E, _F], _E, _E)],
    "Thread": [([_E], _E, _E)], "Dot": [([], _E, _E)], "DotProduct": [([], _E, _E)],
    "Distance2D": [([], _E, _E)],
    # functional
    "Map": [([_F, _E], LEVEL, _E)], "MapAll": [([_F, _E], None, _E)], "MapIndexed": [([_F, _E], LEVEL, _E)],
    "MapThread": [([_F, _E], LEVEL, _E)], "Apply": [([_F, _E], LEVEL, _E)],
    "Nest": [([_F, _E, "NonNegInteger"], None, _E)], "NestList": [([_F, _E, "NonNegInteger"], None, "ListOf[Expr]")],
    "FixedPoint": [([_F, _E], _E, _E)], "FixedPointList": [([_F, _E], _E, "ListOf[Expr]")],
    "Fold": [([_F, _E], _E, _E)], "FoldList": [([_F, _E], _E, "ListOf[Expr]")],
    "Outer": [([_F], _E, _E)], "Inner": [([_F], _E, _E)],
    # strings and forms
    "StringJoin": [([], "String | ListOf[Expr]", _S)], "StringLength": [(["String | ListOf[Expr]"], None,
                                                                         "Integer | ListOf[Integer]")],
    "ToString": [([_E], _E, _S)], "InputForm": [([_E], _E, _E)], "FullForm": [([_E], _E, _E)],
    # algebra, calculus, linear algebra
    "N": [([_E], "Real | ListOf[Expr]", _E)],
    "D": [([_E], _E, _E)], "Diff": [([_E], _E, _E)], "PartialD": [([_E], _E, _E)],
    "Limit": [([_E], _E, _E)], "Series": [([_E], _E, _E)],
    "Solve": [([_E], _E, "ListOf[Expr]")], "DSolve": [([_E], _E, "ListOf[Expr]")],
    "Apart": [([_E], _E, _E)], "Together": [([_E], _E, _E)], "Cancel": [([_E], _E, _E)],
    "Det": [(["MatrixOf[Expr]"], _E, _E)], "Inverse": [(["MatrixOf[Expr]"], _E, "MatrixOf[Expr]")],
}

_PARSED = {}


def kernel_signatures(name):
    """Parsed signatures (types.Signature) of a kernel built-in, or None."""
    sigs = _PARSED.get(name)
    if sigs is not None:
        return sigs
    raw = KERNEL_SIGNATURES.get(name)
    if raw is None:
        return None
    from .types import Signature, parse_annotation
    sigs = [Signature([parse_annotation(p) for p in ps], None if rest is None else parse_annotation(rest),
                      parse_annotation(res)) for ps, rest, res in raw]
    _PARSED[name] = sigs
    return sigs
