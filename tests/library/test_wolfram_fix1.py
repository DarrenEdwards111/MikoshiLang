"""Tests for the notation operators added in the correctness fix-up pass (left out of the W31/W32 plan):
Precedes, Congruent, RightArrow, LeftRightArrow, UpArrow, PlusMinus, Del, Square and DoubleBracketingBar.

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): each says "<Name>[...] has no
  built-in meaning" and "x <op> y is by default interpreted as <Name>[x, y]" (``\\[Del] f`` is Del[f],
  ``\\[Square] x`` is Square[x], ``\\[LeftDoubleBracketingBar] x \\[RightDoubleBracketingBar]`` is
  DoubleBracketingBar[x], ``x ± y`` is PlusMinus[x, y] and ``±x`` is PlusMinus[x]); the PlusMinus page defines
  ``PlusMinus[a_, b_] := {a + b, a - b}`` in its examples.
* The operator table of the Wolfram Language Specification
  (https://wltools.github.io/LanguageSpec/Specification/Syntax/OperatorTableHTML/) for characters, precedences and
  associativity (the REF table below, written out independently of mikoshilang.notation_ops).
* Standard LaTeX symbols (Comprehensive LaTeX Symbol List) for TeXForm.

Placeholder symbols are prefixed ``fx`` and never assigned; definitions are made inside Block.
"""
import pytest

from _harness import Raises, check, ids
from _notation_cases import build, tex_cases, text_is
from mikoshilang.parser import ParseError

# name: (Wolfram character, precedence, affix, associativity) -- Wolfram Language Specification operator table.
REF = {
    "Precedes": ("\u227a", 290, "Infix", "None"),
    "Congruent": ("\u2261", 290, "Infix", "None"),
    "RightArrow": ("\u2192", 270, "Infix", "None"),
    "LeftRightArrow": ("\u2194", 270, "Infix", "None"),
    "UpArrow": ("\u2191", 580, "Infix", "None"),
    "PlusMinus": ("\u00b1", 310, "Infix", "Left"),
    "Del": ("\u2207", 550, "Prefix", "Right"),
}
# Square (\[Square], U+F520, prefix, precedence 540) prints as \[Square]x (private-use character): cases below.
SQUARE = ("\uf520", 540)

TEX = {"Precedes": "\\prec", "Congruent": "\\equiv", "RightArrow": "\\rightarrow",
       "LeftRightArrow": "\\leftrightarrow", "UpArrow": "\\uparrow", "PlusMinus": "\\pm"}

SPECIAL = [
    # Square: prefix, precedence 540 (just below Del, above SmallCircle 530)
    ("Hold[\\[Square]fxa]", "Hold[Square[fxa]]"),
    ("Hold[\uf520fxa] === Hold[Square[fxa]]", "True"),
    ("Hold[\\[Square]fxa^2]", "Hold[Square[fxa^2]]"),
    ("Hold[\\[Square]fxa \\[SmallCircle] fxb]", "Hold[SmallCircle[Square[fxa], fxb]]"),
    ("Hold[\\[Square]\\[Del]fxa]", "Hold[Square[Del[fxa]]]"),
    ("Hold[fxb \\[Square]fxa]", "Hold[Times[fxb, Square[fxa]]]"),
    ("Square[fxa, 1 + 1]", "Square[fxa, 2]"),
    ("Attributes[Square]", "{}"),
    ("ToString[Square[fxa]]", text_is("\\[Square]fxa")),
    ("ToString[Square[fxa, fxb]]", text_is("Square[fxa, fxb]")),
    ("ToExpression[ToString[Square[fxa + 1]]] === Square[fxa + 1]", "True"),
    ("Block[{Square}, Square[x_] := x^2; \\[Square]3]", "9"),
    # Del: prefix 550, binds tighter than Dot and **, looser than ^
    ("Hold[\\[Del]fxa . fxb]", "Hold[Dot[Del[fxa], fxb]]"),
    ("Block[{Del}, Del[f_] := D[f, x]; \\[Del](x^3)]", "3 x^2"),
    ("Del[fxa]", "Del[fxa]"),
    # PlusMinus: definitions as in the Wolfram documentation; prefix form like unary minus
    ("Block[{PlusMinus}, PlusMinus[a_, b_] := {a + b, a - b}; 3 \u00b1 1]", "{4, 2}"),
    ("Hold[\u00b1fxa]", "Hold[PlusMinus[fxa]]"),
    ("Hold[\u00b1fxa^2]", "Hold[PlusMinus[fxa^2]]"),
    ("Hold[fxa + fxb \u00b1 fxc]", "Hold[PlusMinus[Plus[fxa, fxb], fxc]]"),
    ("Hold[fxa \u00b1 fxb - fxc]", "Hold[Plus[PlusMinus[fxa, fxb], Times[-1, fxc]]]"),
    ("Hold[fxa \u00b1 fxb \u2213 fxc]", "Hold[MinusPlus[PlusMinus[fxa, fxb], fxc]]"),
    ("Hold[fxa^\u00b1fxb]", "Hold[Power[fxa, PlusMinus[fxb]]]"),
    ("ToString[PlusMinus[fxa]]", text_is("\u00b1fxa")),
    ("PlusMinus[1 + 1, 3]", "PlusMinus[2, 3]"),
    # RightArrow / LeftRightArrow (270) vs Rule (120): the arrow character is not Rule (that is \[Rule], U+F522)
    ("Hold[fxa \u2192 fxb -> fxc]", "Hold[Rule[RightArrow[fxa, fxb], fxc]]"),
    ("Hold[fxa \u2192 fxb == fxc]", "Hold[RightArrow[fxa, Equal[fxb, fxc]]]"),
    ("Hold[fxa \u2194 fxb && fxc]", "Hold[And[LeftRightArrow[fxa, fxb], fxc]]"),
    ("ToString[\"\u2192\", CharacterEncoding -> \"ASCII\"]", text_is("\\[RightArrow]")),
    ("{fxa, fxb} /. RightArrow -> List", "{fxa, fxb}"),
    ("RightArrow[1, 2] /. RightArrow -> Plus", "3"),
    # UpArrow (580): above ** and below ^
    ("Hold[fxa ** fxb \u2191 fxc]", "Hold[NonCommutativeMultiply[fxa, UpArrow[fxb, fxc]]]"),
    ("Hold[fxa \u2191 fxb ^ fxc]", "Hold[UpArrow[fxa, Power[fxb, fxc]]]"),
    ("Block[{UpArrow}, UpArrow[a_Integer, b_Integer] := a^b; 2 \u2191 10]", "1024"),
    # Precedes / Congruent (290): never chained with ==, < without parentheses
    ("Block[{Precedes}, Precedes[a_, b_] := a < b; {1 \u227a 2, 3 \u227a 2}]", "{True, False}"),
    ("Block[{Congruent}, Congruent[a_, b_] := Mod[a - b, 5] == 0; {7 \u2261 2, 7 \u2261 3}]", "{True, False}"),
    ("Precedes[fxa, fxb] === Precedes[fxa, fxb]", "True"),
    ("Congruent[fxa]", "Congruent[fxa]"),
    # DoubleBracketingBar: matchfix, inert (not Norm)
    ("Hold[\\[LeftDoubleBracketingBar]fxa\\[RightDoubleBracketingBar]]", "Hold[DoubleBracketingBar[fxa]]"),
    ("Hold[\\[LeftDoubleBracketingBar]fxa, fxb\\[RightDoubleBracketingBar]]",
     "Hold[DoubleBracketingBar[fxa, fxb]]"),
    ("Hold[\uf605fxa\uf606] === Hold[DoubleBracketingBar[fxa]]", "True"),
    ("Hold[2 \\[LeftDoubleBracketingBar]fxa\\[RightDoubleBracketingBar]^2]",
     "Hold[Times[2, Power[DoubleBracketingBar[fxa], 2]]]"),
    ("DoubleBracketingBar[{3, 4}]", "DoubleBracketingBar[{3, 4}]"),
    ("DoubleBracketingBar[]", "DoubleBracketingBar[]"),
    ("DoubleBracketingBar[1 + 1]", "DoubleBracketingBar[2]"),
    ("Attributes[DoubleBracketingBar]", "{}"),
    ("ToString[DoubleBracketingBar[fxa, fxb]]",
     text_is("\\[LeftDoubleBracketingBar]fxa, fxb\\[RightDoubleBracketingBar]")),
    ("ToExpression[ToString[DoubleBracketingBar[fxa + 1]]] === DoubleBracketingBar[fxa + 1]", "True"),
    ("Block[{DoubleBracketingBar}, DoubleBracketingBar[v_List] := Norm[v]; "
     "\\[LeftDoubleBracketingBar]{3, 4}\\[RightDoubleBracketingBar]]", "5"),
    ("CharacterName[\"\uf605\"]", "\"LeftDoubleBracketingBar\""),
]

# Different operators of one precedence level are not mixed without parentheses (ParseError); unclosed matchfix.
PARSE_ERRORS = ["fxa \u227a fxb == fxc", "fxa \u2261 fxb < fxc", "fxa \u2192 fxb \u21d2 fxc",
                "fxa \u2194 fxb \u2192 fxc", "fxa \u2191 fxb \u2193 fxc", "fxa \u227a fxb \u2261 fxc",
                "\\[LeftDoubleBracketingBar]fxa", "fxa\\[RightDoubleBracketingBar]"]

CASES = build("fx", REF) + tex_cases("fx", TEX) + SPECIAL + [(s, Raises(ParseError)) for s in PARSE_ERRORS]

NAMES = ["Precedes", "Congruent", "RightArrow", "LeftRightArrow", "UpArrow", "PlusMinus", "Del", "Square",
         "DoubleBracketingBar"]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_fix1_notation(case):
    check(case)


def test_fix1_every_symbol_has_three_cases():
    import re
    from mikoshilang import library
    from mikoshilang.library import wl_notation_rel
    names = sorted(n for n, (area, _s, _n) in library.STATUS.items() if area == wl_notation_rel.AREA_FIX1)
    assert names == sorted(NAMES)
    counts = {n: sum(1 for c in CASES if re.search(r"(?<![A-Za-z0-9])" + n + r"(?![A-Za-z0-9])", c[0]))
              for n in names}
    assert all(v >= 3 for v in counts.values()), {k: v for k, v in counts.items() if v < 3}


def test_fix1_parser_tables_match_reference():
    from mikoshilang import notation_ops as N
    ref = dict(REF, Square=(SQUARE[0], SQUARE[1], "Prefix", "Right"))
    for name, (ch, prec, affix, assoc) in ref.items():
        mch, mprec, kind, _tex, batch = N.OPS[name]
        assert batch == "FIX1" and mch == ch, name
        if affix == "Prefix":
            assert kind == "prefix" and mprec == prec, name
            continue
        assert mprec == prec, name
        assert kind == {"None": "nary", "Left": "left", "Right": "right"}[assoc], name
    assert (N.LEFT_DBAR, N.RIGHT_DBAR) == ("\uf605", "\uf606")
