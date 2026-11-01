"""Differential tests for Wolfram batch W32 (notation II: arrows, tees, vectors and equilibria; module
wl_notation_arrows).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): every head of the batch "has no
  built-in meaning" and "x <op> y is by default interpreted as <Name>[x, y]" (e.g. ``x ⊢ y // FullForm`` is
  ``RightTee[x, y]``, ``x ⇌ y`` is ``Equilibrium[x, y]``).
* The operator table of the Wolfram Language Specification
  (https://wltools.github.io/LanguageSpec/Specification/Syntax/OperatorTableHTML/) for characters, precedences
  (270 for horizontal arrows and vectors, 580 for vertical and long arrows, 290 for the equilibria, 190 for the
  tees) and associativity (RightTee / DoubleRightTee group to the right, the other tees to the left, the arrows are
  n-ary): the REF table below.
* Standard LaTeX symbols (Comprehensive LaTeX Symbol List) for TeXForm.

Placeholder symbols are prefixed ``w32`` and never assigned; definitions are made inside Block.
"""
import pytest

from _harness import Pred, Raises, check, ids
from _notation_cases import build, tex_cases, text_is
from mikoshilang.expr import Expr
from mikoshilang.parser import ParseError

# name: (Wolfram character, precedence, affix, associativity) -- Wolfram Language Specification operator table.
REF = {
    "DoubleDownArrow": ("\u21d3", 580, "Infix", "None"),
    "DoubleLeftArrow": ("\u21d0", 270, "Infix", "None"),
    "DoubleLeftRightArrow": ("\u21d4", 270, "Infix", "None"),
    "DoubleLeftTee": ("\u2ae4", 190, "Infix", "Left"),
    "DoubleLongLeftArrow": ("\u27f8", 580, "Infix", "None"),
    "DoubleLongLeftRightArrow": ("\u27fa", 580, "Infix", "None"),
    "DoubleLongRightArrow": ("\u27f9", 580, "Infix", "None"),
    "DoubleRightArrow": ("\u21d2", 270, "Infix", "None"),
    "DoubleRightTee": ("\u22a8", 190, "Infix", "Right"),
    "DoubleUpArrow": ("\u21d1", 580, "Infix", "None"),
    "DoubleUpDownArrow": ("\u21d5", 580, "Infix", "None"),
    "DownArrow": ("\u2193", 580, "Infix", "None"),
    "DownArrowBar": ("\u2913", 580, "Infix", "None"),
    "DownArrowUpArrow": ("\u21f5", 580, "Infix", "None"),
    "DownLeftRightVector": ("\u2950", 270, "Infix", "None"),
    "DownLeftTeeVector": ("\u295e", 270, "Infix", "None"),
    "DownLeftVector": ("\u21bd", 270, "Infix", "None"),
    "DownLeftVectorBar": ("\u2956", 270, "Infix", "None"),
    "DownRightTeeVector": ("\u295f", 270, "Infix", "None"),
    "DownRightVector": ("\u21c1", 270, "Infix", "None"),
    "DownRightVectorBar": ("\u2957", 270, "Infix", "None"),
    "DownTee": ("\u22a4", 190, "Infix", "Left"),
    "DownTeeArrow": ("\u21a7", 580, "Infix", "None"),
    "Equilibrium": ("\u21cc", 290, "Infix", "None"),
    "LeftArrow": ("\u2190", 270, "Infix", "None"),
    "LeftArrowBar": ("\u21e4", 270, "Infix", "None"),
    "LeftArrowRightArrow": ("\u21c6", 270, "Infix", "None"),
    "LeftDownTeeVector": ("\u2961", 580, "Infix", "None"),
    "LeftDownVector": ("\u21c3", 580, "Infix", "None"),
    "LeftDownVectorBar": ("\u2959", 580, "Infix", "None"),
    "LeftRightVector": ("\u294e", 270, "Infix", "None"),
    "LeftTee": ("\u22a3", 190, "Infix", "Left"),
    "LeftTeeArrow": ("\u21a4", 270, "Infix", "None"),
    "LeftTeeVector": ("\u295a", 270, "Infix", "None"),
    "LeftUpDownVector": ("\u2951", 580, "Infix", "None"),
    "LeftUpTeeVector": ("\u2960", 580, "Infix", "None"),
    "LeftUpVector": ("\u21bf", 580, "Infix", "None"),
    "LeftUpVectorBar": ("\u2958", 580, "Infix", "None"),
    "LeftVector": ("\u21bc", 270, "Infix", "None"),
    "LeftVectorBar": ("\u2952", 270, "Infix", "None"),
    "LongLeftArrow": ("\u27f5", 580, "Infix", "None"),
    "LongLeftRightArrow": ("\u27f7", 580, "Infix", "None"),
    "LongRightArrow": ("\u27f6", 580, "Infix", "None"),
    "LowerLeftArrow": ("\u2199", 270, "Infix", "None"),
    "LowerRightArrow": ("\u2198", 270, "Infix", "None"),
    "ReverseEquilibrium": ("\u21cb", 290, "Infix", "None"),
    "ReverseUpEquilibrium": ("\u296f", 580, "Infix", "None"),
    "RightArrowBar": ("\u21e5", 270, "Infix", "None"),
    "RightArrowLeftArrow": ("\u21c4", 270, "Infix", "None"),
    "RightDownTeeVector": ("\u295d", 580, "Infix", "None"),
    "RightDownVector": ("\u21c2", 580, "Infix", "None"),
    "RightDownVectorBar": ("\u2955", 580, "Infix", "None"),
    "RightTee": ("\u22a2", 190, "Infix", "Right"),
    "RightTeeArrow": ("\u21a6", 270, "Infix", "None"),
    "RightTeeVector": ("\u295b", 270, "Infix", "None"),
    "RightUpDownVector": ("\u294f", 580, "Infix", "None"),
    "RightUpTeeVector": ("\u295c", 580, "Infix", "None"),
    "RightUpVector": ("\u21be", 580, "Infix", "None"),
    "RightUpVectorBar": ("\u2954", 580, "Infix", "None"),
    "RightVector": ("\u21c0", 270, "Infix", "None"),
    "RightVectorBar": ("\u2953", 270, "Infix", "None"),
    "ShortDownArrow": ("\uf52b", 580, "Infix", "None"),
    "ShortLeftArrow": ("\uf526", 270, "Infix", "None"),
    "ShortRightArrow": ("\uf525", 270, "Infix", "None"),
    "ShortUpArrow": ("\uf52a", 580, "Infix", "None"),
    "UpArrowBar": ("\u2912", 580, "Infix", "None"),
    "UpArrowDownArrow": ("\u21c5", 580, "Infix", "None"),
    "UpDownArrow": ("\u2195", 580, "Infix", "None"),
    "UpEquilibrium": ("\u296e", 580, "Infix", "None"),
    "UpTee": ("\u22a5", 190, "Infix", "Left"),
    "UpTeeArrow": ("\u21a5", 580, "Infix", "None"),
    "UpperLeftArrow": ("\u2196", 270, "Infix", "None"),
    "UpperRightArrow": ("\u2197", 270, "Infix", "None"),
}

TEX = {
    "DoubleDownArrow": '\\Downarrow',
    "DoubleLeftArrow": '\\Leftarrow',
    "DoubleLeftRightArrow": '\\Leftrightarrow',
    "DoubleLongLeftArrow": '\\Longleftarrow',
    "DoubleLongLeftRightArrow": '\\Longleftrightarrow',
    "DoubleLongRightArrow": '\\Longrightarrow',
    "DoubleRightArrow": '\\Rightarrow',
    "DoubleRightTee": '\\vDash',
    "DoubleUpArrow": '\\Uparrow',
    "DoubleUpDownArrow": '\\Updownarrow',
    "DownArrow": '\\downarrow',
    "DownLeftVector": '\\leftharpoondown',
    "DownRightVector": '\\rightharpoondown',
    "DownTee": '\\top',
    "Equilibrium": '\\rightleftharpoons',
    "LeftArrow": '\\leftarrow',
    "LeftArrowRightArrow": '\\leftrightarrows',
    "LeftDownVector": '\\downharpoonleft',
    "LeftTee": '\\dashv',
    "LeftUpVector": '\\upharpoonleft',
    "LeftVector": '\\leftharpoonup',
    "LongLeftArrow": '\\longleftarrow',
    "LongLeftRightArrow": '\\longleftrightarrow',
    "LongRightArrow": '\\longrightarrow',
    "LowerLeftArrow": '\\swarrow',
    "LowerRightArrow": '\\searrow',
    "ReverseEquilibrium": '\\leftrightharpoons',
    "RightArrowLeftArrow": '\\rightleftarrows',
    "RightDownVector": '\\downharpoonright',
    "RightTee": '\\vdash',
    "RightTeeArrow": '\\mapsto',
    "RightUpVector": '\\upharpoonright',
    "RightVector": '\\rightharpoonup',
    "UpDownArrow": '\\updownarrow',
    "UpTee": '\\bot',
    "UpperLeftArrow": '\\nwarrow',
    "UpperRightArrow": '\\nearrow',
}

SPECIAL = [
    # arrows of different precedences nest
    ("Hold[w32a \\[LeftArrow] w32b \\[DownArrow] w32c]", "Hold[LeftArrow[w32a, DownArrow[w32b, w32c]]]"),
    ("Hold[w32a \\[LongRightArrow] w32b + w32c]", "Hold[Plus[LongRightArrow[w32a, w32b], w32c]]"),
    ("Hold[w32a \\[RightVector] w32b \\[Equilibrium] w32c]",
     "Hold[RightVector[w32a, Equilibrium[w32b, w32c]]]"),
    ("Hold[w32a \\[RightTee] w32b \\[LeftArrow] w32c]", "Hold[RightTee[w32a, LeftArrow[w32b, w32c]]]"),
    ("Hold[w32a \\[LeftTee] w32b \\[DownTee] w32c]", "Hold[DownTee[LeftTee[w32a, w32b], w32c]]"),
    ("Hold[w32a \\[RightTee] w32b \\[DoubleRightTee] w32c]", "Hold[RightTee[w32a, DoubleRightTee[w32b, w32c]]]"),
    ("ToString[RightTee[w32a, LeftTee[w32b, w32c]]]", text_is("w32a \u22a2 (w32b \u22a3 w32c)")),
    ("ToString[LeftTee[RightTee[w32a, w32b], w32c]]", text_is("(w32a \u22a2 w32b) \u22a3 w32c")),
    ("ToString[DownTee[LeftTee[w32a, w32b], w32c]]", text_is("w32a \u22a3 w32b \u22a4 w32c")),
    ("ToString[LeftArrow[Plus[w32a, w32b], w32c]]", text_is("w32a + w32b \u2190 w32c")),
    ("ToExpression[ToString[Times[DoubleUpArrow[w32a, w32b], w32c]]] === Times[DoubleUpArrow[w32a, w32b], w32c]",
     "True"),
    ("ToString[Power[DownArrow[w32a, w32b], 2]]", text_is("(w32a \u2193 w32b)^2")),
    ("ToExpression[ToString[Equal[Equilibrium[w32a, w32b], w32c]]] === Equal[Equilibrium[w32a, w32b], w32c]",
     "True"),
    # user definitions (no attributes, as in Wolfram)
    ("Block[{RightArrowLeftArrow}, RightArrowLeftArrow[a_, b_] := {a, b}; 1 \u21c4 2]", "{1, 2}"),
    ("Block[{LeftTee}, LeftTee[a_, b_] := a - b; (5 \u22a3 3) \u22a3 1]", "1"),
]

PARSE_ERRORS = ["w32a \u2190 w32b \u21d0 w32c", "w32a \u22a2 w32b \u22a3 w32c", "w32a \u22a3 w32b \u22a2 w32c",
                "w32a \u21cc w32b == w32c", "w32a < w32b \u21cc w32c", "w32a \u2193 w32b \u21d3 w32c",
                "w32a \u21cc w32b \u21cb w32c"]

CASES = build("w32", REF) + tex_cases("w32", TEX) + SPECIAL + [(s, Raises(ParseError)) for s in PARSE_ERRORS]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_w32(case):
    check(case)


def test_w32_every_symbol_has_three_cases():
    import re
    from mikoshilang import library
    from mikoshilang.library import wl_notation_arrows
    names = [n for n, (area, _s, _n) in library.STATUS.items() if area == wl_notation_arrows.AREA]
    assert len(names) == 73
    counts = {n: sum(1 for c in CASES if re.search(r"(?<![A-Za-z0-9])" + n + r"(?![A-Za-z0-9])", c[0]))
              for n in names}
    assert all(v >= 3 for v in counts.values()), {k: v for k, v in counts.items() if v < 3}


def test_w32_parser_tables_match_reference():
    """The parser's operator table agrees with the Wolfram specification table (characters, precedences,
    grouping)."""
    from mikoshilang import notation_ops as N
    for name, (ch, prec, affix, assoc) in REF.items():
        mch, mprec, kind, _tex, batch = N.OPS[name]
        assert batch == "W32" and mch == ch and mprec == prec, name
        assert kind == {"None": "nary", "Left": "left", "Right": "right"}[assoc], name
