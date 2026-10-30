"""Parser support for Wolfram operator syntax added with batch 1 (lists, strings, associations).

Expected FullForms follow the Wolfram Language operator table
(https://reference.wolfram.com/language/tutorial/OperatorInputForms.html).
"""
import pytest

from mikoshilang.formatting import full_form
from mikoshilang.parser import parse, ParseError

CASES = [
    ("<|a -> 1, \"b\" -> 2|>", 'Association[Rule[a, 1], Rule["b", 2]]'),
    ("<||>", "Association[]"),
    ("<|a -> <|b -> 1|>|>", "Association[Rule[a, Association[Rule[b, 1]]]]"),
    ("x[[2;;4]]", "Part[x, Span[2, 4]]"),
    ("x[[;;3]]", "Part[x, Span[1, 3]]"),
    ("x[[2;;]]", "Part[x, Span[2, All]]"),
    ("x[[;;]]", "Part[x, Span[1, All]]"),
    ("x[[1;;-1;;2]]", "Part[x, Span[1, -1, 2]]"),
    ("x[[;;;;2]]", "Part[x, Span[1, All, 2]]"),
    ("x[[1;;n-1]]", "Part[x, Span[1, Plus[n, -1]]]"),
    ("x[[All, 2;;3]]", "Part[x, All, Span[2, 3]]"),
    ("\"a\" ~~ x__ ~~ \"b\"", 'StringExpression["a", Pattern[x, BlankSequence[]], "b"]'),
    ("a | b | c", "Alternatives[a, b, c]"),
    ("x : (_Integer | _Real)", "Pattern[x, Alternatives[Blank[Integer], Blank[Real]]]"),
    ("f[x_, y_:0]", "f[Pattern[x, Blank[]], Optional[Pattern[y, Blank[]], 0]]"),
    ("x..", "Repeated[x]"),
    ("{_, _}...", "RepeatedNull[List[Blank[], Blank[]]]"),
    ("a | b..", "Alternatives[a, Repeated[b]]"),
    ("f @* g @* h", "Composition[f, g, h]"),
    ("f /* g", "RightComposition[f, g]"),
    ("(f @* g)[x]", "Composition[f, g][x]"),
    ("#name &", 'Function[Slot["name"]]'),
    ("#a + #b &", 'Function[Plus[Slot["a"], Slot["b"]]]'),
    ("f //@ x", "MapAll[f, x]"),
    ("f @@@ {{1, 2}}", "Apply[f, List[List[1, 2]], List[1]]"),
    ("\"x\" ~~ (\"a\" | \"b\") ~~ \"y\"", 'StringExpression["x", Alternatives["a", "b"], "y"]'),
    ("x_ /; x > 0 -> x", "Rule[Condition[Pattern[x, Blank[]], Greater[x, 0]], x]"),
    ("a || b | c", "Alternatives[Or[a, b], c]"),
    ("s : \"ab\" ~~ __", 'StringExpression[Pattern[s, "ab"], BlankSequence[]]'),
]


@pytest.mark.parametrize("src,want", CASES, ids=[c[0] for c in CASES])
def test_full_form(src, want):
    assert full_form(parse(src)) == want


@pytest.mark.parametrize("src", [c[0] for c in CASES])
def test_input_form_round_trip(src):
    e = parse(src)
    assert parse(repr(e)) == e


@pytest.mark.parametrize("src", ["<|a -> 1", "x[[1;;", "f[1 : 2]"])
def test_errors(src):
    with pytest.raises(ParseError):
        parse(src)
