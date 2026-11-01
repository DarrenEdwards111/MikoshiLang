"""Differential tests for Wolfram batch W31 (notation I: relational, set-like and bracket operators; module
wl_notation_rel).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): every operator head of the batch
  "has no built-in meaning" and "x <op> y is by default interpreted as <Name>[x, y]"; the documentation examples
  for NotElement (``NotElement[Sqrt[2], #] & /@ {Complexes, Algebraics, Reals, Rationals, Integers, Primes}`` is
  ``{False, False, False, True, True, True}``), Commutator / Anticommutator (``Commutator[x, y]`` is
  ``x ** y - y ** x``; with ``NonCommutativeAlgebra[<|"Multiplication" -> mult, "Addition" -> add|>]`` it is
  ``add[mult[x, y], -mult[y, x]]``; with ``{Dot, n}`` ``x . y - y . x``; with Composition ``x @* y - y @* x``),
  Ket / Bra / BraKet (the linearity, anti-linearity and orthogonality definitions of their Basic Examples),
  AngleBracket (``〈x, y, z〉 // FullForm`` is ``AngleBracket[x, y, z]``) and MinusPlus (``x ∓ y`` is
  ``MinusPlus[x, y]``, ``∓x`` is ``MinusPlus[x]``).
* The operator table of the Wolfram Language Specification
  (https://wltools.github.io/LanguageSpec/Specification/Syntax/OperatorTableHTML/) for characters, precedences and
  associativity (the REF table below), and the "Operator Input Forms" tutorial
  (reference.wolfram.com/language/tutorial/OperatorInputForms.html: ⩾ and ⩽ are read as GreaterEqual / LessEqual).
* Pauli matrices (independent identities): [σx, σy] = 2 I σz and {σx, σx} = 2 I₂.
* Standard LaTeX symbols (Comprehensive LaTeX Symbol List) for TeXForm.

Placeholder symbols are prefixed ``w31`` and never assigned; definitions are made inside Block.
"""
import pytest

from _harness import Pred, Raises, check, ids
from _notation_cases import build, tex_cases, text_is
from mikoshilang.expr import Expr
from mikoshilang.parser import ParseError

# name: (Wolfram character, precedence, affix, associativity) -- Wolfram Language Specification operator table.
REF = {
    "Backslash": ("\u2216", 460, "Infix", "None"),
    "Because": ("\u2235", 50, "Infix", "Left"),
    "Cap": ("\u2322", 350, "Infix", "None"),
    "CapitalDifferentialD": ("\uf74b", 560, "Prefix", "Right"),
    "CenterDot": ("\u00b7", 410, "Infix", "None"),
    "CircleDot": ("\u2299", 520, "Infix", "None"),
    "CircleMinus": ("\u2296", 330, "Infix", "Left"),
    "CirclePlus": ("\u2295", 330, "Infix", "None"),
    "CircleTimes": ("\u2297", 420, "Infix", "None"),
    "Colon": ("\u2236", 80, "Infix", "None"),
    "Coproduct": ("\u2210", 360, "Infix", "None"),
    "Cup": ("\u2323", 340, "Infix", "None"),
    "CupCap": ("\u224d", 290, "Infix", "None"),
    "Diamond": ("\u22c4", 450, "Infix", "None"),
    "DifferentialD": ("\uf74c", 560, "Prefix", "Right"),
    "DotEqual": ("\u2250", 290, "Infix", "None"),
    "DoubleVerticalBar": ("\u2225", 280, "Infix", "None"),
    "EqualTilde": ("\u2242", 290, "Infix", "None"),
    "GreaterEqualLess": ("\u22db", 290, "Infix", "None"),
    "GreaterFullEqual": ("\u2267", 290, "Infix", "None"),
    "GreaterGreater": ("\u226b", 290, "Infix", "None"),
    "GreaterLess": ("\u2277", 290, "Infix", "None"),
    "GreaterSlantEqual": ("\u2a7e", 290, "Infix", "None"),
    "GreaterTilde": ("\u2273", 290, "Infix", "None"),
    "HumpDownHump": ("\u224e", 290, "Infix", "None"),
    "HumpEqual": ("\u224f", 290, "Infix", "None"),
    "LeftTriangle": ("\u22b2", 290, "Infix", "None"),
    "LeftTriangleBar": ("\u29cf", 290, "Infix", "None"),
    "LeftTriangleEqual": ("\u22b4", 290, "Infix", "None"),
    "LessEqualGreater": ("\u22da", 290, "Infix", "None"),
    "LessFullEqual": ("\u2266", 290, "Infix", "None"),
    "LessGreater": ("\u2276", 290, "Infix", "None"),
    "LessLess": ("\u226a", 290, "Infix", "None"),
    "LessSlantEqual": ("\u2a7d", 290, "Infix", "None"),
    "LessTilde": ("\u2272", 290, "Infix", "None"),
    "MinusPlus": ("\u2213", 310, "Infix", "Left"),
    "NestedGreaterGreater": ("\u2aa2", 290, "Infix", "None"),
    "NestedLessLess": ("\u2aa1", 290, "Infix", "None"),
    "NotCongruent": ("\u2262", 290, "Infix", "None"),
    "NotCupCap": ("\u226d", 290, "Infix", "None"),
    "NotDoubleVerticalBar": ("\u2226", 280, "Infix", "None"),
    "NotEqualTilde": ("\uf400", 290, "Infix", "None"),
    "NotExists": ("\u2204", 240, "Prefix", "Right"),
    "NotGreater": ("\u226f", 290, "Infix", "None"),
    "NotGreaterEqual": ("\u2271", 290, "Infix", "None"),
    "NotGreaterFullEqual": ("\u2269", 290, "Infix", "None"),
    "NotGreaterGreater": ("\uf427", 290, "Infix", "None"),
    "NotGreaterLess": ("\u2279", 290, "Infix", "None"),
    "NotGreaterSlantEqual": ("\uf429", 290, "Infix", "None"),
    "NotGreaterTilde": ("\u2275", 290, "Infix", "None"),
    "NotHumpDownHump": ("\uf402", 290, "Infix", "None"),
    "NotHumpEqual": ("\uf401", 290, "Infix", "None"),
    "NotLeftTriangle": ("\u22ea", 290, "Infix", "None"),
    "NotLeftTriangleBar": ("\uf412", 290, "Infix", "None"),
    "NotLeftTriangleEqual": ("\u22ec", 290, "Infix", "None"),
    "NotLess": ("\u226e", 290, "Infix", "None"),
    "NotLessEqual": ("\u2270", 290, "Infix", "None"),
    "NotLessFullEqual": ("\u2268", 290, "Infix", "None"),
    "NotLessGreater": ("\u2278", 290, "Infix", "None"),
    "NotLessLess": ("\uf422", 290, "Infix", "None"),
    "NotLessSlantEqual": ("\uf424", 290, "Infix", "None"),
    "NotLessTilde": ("\u2274", 290, "Infix", "None"),
    "NotNestedGreaterGreater": ("\uf428", 290, "Infix", "None"),
    "NotNestedLessLess": ("\uf423", 290, "Infix", "None"),
    "NotPrecedes": ("\u2280", 290, "Infix", "None"),
    "NotPrecedesEqual": ("\uf42b", 290, "Infix", "None"),
    "NotPrecedesSlantEqual": ("\u22e0", 290, "Infix", "None"),
    "NotPrecedesTilde": ("\u22e8", 290, "Infix", "None"),
    "NotReverseElement": ("\u220c", 250, "Infix", "None"),
    "NotRightTriangle": ("\u22eb", 290, "Infix", "None"),
    "NotRightTriangleBar": ("\uf413", 290, "Infix", "None"),
    "NotRightTriangleEqual": ("\u22ed", 290, "Infix", "None"),
    "NotSquareSubset": ("\uf42e", 250, "Infix", "None"),
    "NotSquareSubsetEqual": ("\u22e2", 250, "Infix", "None"),
    "NotSquareSuperset": ("\uf42f", 250, "Infix", "None"),
    "NotSquareSupersetEqual": ("\u22e3", 250, "Infix", "None"),
    "NotSubset": ("\u2284", 250, "Infix", "None"),
    "NotSubsetEqual": ("\u2288", 250, "Infix", "None"),
    "NotSucceeds": ("\u2281", 290, "Infix", "None"),
    "NotSucceedsEqual": ("\uf42d", 290, "Infix", "None"),
    "NotSucceedsSlantEqual": ("\u22e1", 290, "Infix", "None"),
    "NotSucceedsTilde": ("\u22e9", 290, "Infix", "None"),
    "NotSuperset": ("\u2285", 250, "Infix", "None"),
    "NotSupersetEqual": ("\u2289", 250, "Infix", "None"),
    "NotTilde": ("\u2241", 290, "Infix", "None"),
    "NotTildeEqual": ("\u2244", 290, "Infix", "None"),
    "NotTildeFullEqual": ("\u2247", 290, "Infix", "None"),
    "NotTildeTilde": ("\u2249", 290, "Infix", "None"),
    "NotVerticalBar": ("\uf3d1", 280, "Infix", "None"),
    "PrecedesEqual": ("\u2aaf", 290, "Infix", "None"),
    "PrecedesSlantEqual": ("\u227c", 290, "Infix", "None"),
    "PrecedesTilde": ("\u227e", 290, "Infix", "None"),
    "Proportion": ("\u2237", 290, "Infix", "None"),
    "Proportional": ("\u221d", 290, "Infix", "None"),
    "ReverseElement": ("\u220b", 250, "Infix", "None"),
    "RightTriangle": ("\u22b3", 290, "Infix", "None"),
    "RightTriangleBar": ("\u29d0", 290, "Infix", "None"),
    "RightTriangleEqual": ("\u22b5", 290, "Infix", "None"),
    "SmallCircle": ("\u2218", 530, "Infix", "None"),
    "SquareIntersection": ("\u2293", 305, "Infix", "None"),
    "SquareSubset": ("\u228f", 250, "Infix", "None"),
    "SquareSubsetEqual": ("\u2291", 250, "Infix", "None"),
    "SquareSuperset": ("\u2290", 250, "Infix", "None"),
    "SquareSupersetEqual": ("\u2292", 250, "Infix", "None"),
    "SquareUnion": ("\u2294", 300, "Infix", "None"),
    "Star": ("\u22c6", 390, "Infix", "None"),
    "Subset": ("\u2282", 250, "Infix", "None"),
    "SubsetEqual": ("\u2286", 250, "Infix", "None"),
    "Succeeds": ("\u227b", 290, "Infix", "None"),
    "SucceedsEqual": ("\u2ab0", 290, "Infix", "None"),
    "SucceedsSlantEqual": ("\u227d", 290, "Infix", "None"),
    "SucceedsTilde": ("\u227f", 290, "Infix", "None"),
    "Superset": ("\u2283", 250, "Infix", "None"),
    "SupersetEqual": ("\u2287", 250, "Infix", "None"),
    "Therefore": ("\u2234", 50, "Infix", "Right"),
    "Tilde": ("\u223c", 290, "Infix", "None"),
    "TildeEqual": ("\u2243", 290, "Infix", "None"),
    "TildeFullEqual": ("\u2245", 290, "Infix", "None"),
    "TildeTilde": ("\u2248", 290, "Infix", "None"),
    "UnionPlus": ("\u228e", 300, "Infix", "None"),
    "Vee": ("\u22c1", 430, "Infix", "None"),
    "VerticalBar": ("\uf3d0", 280, "Infix", "None"),
    "VerticalSeparator": ("\uf432", 60, "Infix", "None"),
    "VerticalTilde": ("\u2240", 370, "Infix", "None"),
}

TEX = {
    "CirclePlus": '\\oplus',
    "CircleMinus": '\\ominus',
    "CircleTimes": '\\otimes',
    "CircleDot": '\\odot',
    "CenterDot": '\\cdot',
    "SmallCircle": '\\circ',
    "Star": '\\star',
    "Diamond": '\\diamond',
    "Vee": '\\vee',
    "Backslash": '\\backslash',
    "Cap": '\\frown',
    "Cup": '\\smile',
    "VerticalTilde": '\\wr',
    "Coproduct": '\\amalg',
    "SquareUnion": '\\sqcup',
    "SquareIntersection": '\\sqcap',
    "UnionPlus": '\\uplus',
    "MinusPlus": '\\mp',
    "SupersetEqual": '\\supseteq',
    "SquareSubset": '\\sqsubset',
    "SquareSupersetEqual": '\\sqsupseteq',
    "ReverseElement": '\\ni',
    "Succeeds": '\\succ',
    "SucceedsEqual": '\\succeq',
    "PrecedesEqual": '\\preceq',
    "Tilde": '\\sim',
    "NotTilde": '\\nsim',
    "TildeEqual": '\\simeq',
    "TildeFullEqual": '\\cong',
    "TildeTilde": '\\approx',
    "DotEqual": '\\doteq',
    "HumpEqual": '\\bumpeq',
    "CupCap": '\\asymp',
    "GreaterGreater": '\\gg',
    "LessLess": '\\ll',
    "GreaterTilde": '\\gtrsim',
    "LessTilde": '\\lesssim',
    "NotGreater": '\\ngtr',
    "NotLess": '\\nless',
    "LeftTriangle": '\\vartriangleleft',
    "RightTriangle": '\\vartriangleright',
    "DoubleVerticalBar": '\\parallel',
    "VerticalBar": '\\mid',
    "NotVerticalBar": '\\nmid',
    "Therefore": '\\therefore',
    "Because": '\\because',
}


def _scaled(factor, head, arg_text):
    """A product whose numeric factors multiply to ``factor`` times head[...] printed as arg_text (the
    evaluator does not always flatten nested Times after a user rule, so the factors are collected here)."""
    def ok(g):
        num, rest = 1, []

        def walk(e):
            nonlocal num
            if isinstance(e, Expr) and e.head == "Times":
                for a in e.args:
                    walk(a)
            elif isinstance(e, (int, complex)) or (isinstance(e, Expr) and e.head == "Complex"):
                num *= complex(*e.args) if isinstance(e, Expr) else e
            elif getattr(e, "name", None) == "I":
                num *= 1j
            else:
                rest.append(e)
        walk(g)
        return num == factor and len(rest) == 1 and isinstance(rest[0], Expr) and rest[0].head == head and \
            repr(rest[0]) == arg_text
    return Pred(ok, f"{factor} {arg_text}")


SPECIAL = [
    # NotElement (reference page examples; decided through Element)
    ("NotElement[Sqrt[2], #] & /@ {Complexes, Algebraics, Reals, Rationals, Integers, Primes}",
     "{False, False, False, True, True, True}"),
    ("NotElement[w31x, Reals]", Pred(lambda g: isinstance(g, Expr) and g.head == "NotElement", "unevaluated")),
    ("NotElement[2 | Sqrt[2], Integers]", "False"),
    ("NotElement[Sqrt[2] | Sqrt[3], Integers]", "True"),
    ("NotElement[Sqrt[2] | w31x, Integers]", Pred(lambda g: repr(g) == "w31x \u2209 Integers", "w31x ∉ Integers")),
    ("NotElement[Pi, Algebraics] === Not[Element[Pi, Algebraics]]", "True"),
    ("NotElement[3, Primes]", "False"),
    ("NotElement[1/2, Integers]", "True"),
    ("Hold[w31a \\[NotElement] w31b] === Hold[NotElement[w31a, w31b]]", "True"),
    ("Hold[w31a \u2209 Reals]", "Hold[NotElement[w31a, Reals]]"),
    ("Hold[w31a \\[NotElement] w31b == w31c]", "Hold[NotElement[w31a, Equal[w31b, w31c]]]"),
    ("Hold[w31a && w31b \\[NotElement] w31c]", "Hold[And[w31a, NotElement[w31b, w31c]]]"),
    ("ToString[NotElement[w31a, Reals]]", text_is("w31a \u2209 Reals")),
    ("ToString[TeXForm[NotElement[x, y]]]", text_is("x\\notin y")),
    ("CharacterName[\"\u2209\"]", "\"NotElement\""),
    ("Attributes[NotElement]", "{Protected}"),
    ("NotElement[w31x]", Raises(ValueError)),
    # Commutator / Anticommutator (reference page examples)
    ("Commutator[w31x, w31y]", "w31x ** w31y - w31y ** w31x"),
    ("Commutator[w31x, w31y, NonCommutativeAlgebra[<|\"Multiplication\" -> w31mult, \"Addition\" -> w31add|>]]",
     "w31add[w31mult[w31x, w31y], -w31mult[w31y, w31x]]"),
    ("Commutator[w31x, w31y, <|\"Multiplication\" -> w31mult|>]", "w31mult[w31x, w31y] - w31mult[w31y, w31x]"),
    ("Commutator[w31x, w31y, {Dot, w31n}]", "w31x . w31y - w31y . w31x"),
    ("Commutator[w31x, w31y, Composition]", "w31x @* w31y - w31y @* w31x"),
    ("Commutator[w31x, w31x]", "0"),
    ("Commutator[{{0, 1}, {1, 0}}, {{0, -I}, {I, 0}}, {Dot, 2}]", "2 I {{1, 0}, {0, -1}}"),
    ("Commutator[{{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}, Dot]", "{{0, 0}, {0, 0}}"),
    ("Commutator[w31x, w31y] + Anticommutator[w31x, w31y] === 2 w31x ** w31y", "True"),
    ("Commutator[w31x, w31y] === -Commutator[w31y, w31x]", "True"),
    ("Attributes[Commutator]", "{Protected}"),
    ("Commutator[w31x, w31y, \"alg\"]", Raises(TypeError)),
    ("Commutator[w31x, w31y, {Dot, 0}]", Raises(ValueError)),
    ("Commutator[w31x, w31y, {Dot}]", Raises(ValueError)),
    ("Commutator[{{0, 1}, {0, 0}}, {{0, 0}, {1, 0}}, {Dot, 3}]", Raises(ValueError)),
    ("Commutator[w31x, w31y, NonCommutativeAlgebra[<|\"Frobnicate\" -> 1|>]]", Raises(NotImplementedError)),
    ("Commutator[w31x]", Raises(ValueError)),
    ("Anticommutator[w31x, w31y]", "w31x ** w31y + w31y ** w31x"),
    ("Anticommutator[w31x, w31y, NonCommutativeAlgebra[<|\"Multiplication\" -> w31mult, \"Addition\" -> w31add|>]]",
     "w31add[w31mult[w31x, w31y], w31mult[w31y, w31x]]"),
    ("Anticommutator[w31x, w31y, {Dot, w31n}]", "w31x . w31y + w31y . w31x"),
    ("Anticommutator[w31x, w31y, Composition]", "w31x @* w31y + w31y @* w31x"),
    ("Anticommutator[{{0, 1}, {1, 0}}, {{0, 1}, {1, 0}}, {Dot, 2}]", "{{2, 0}, {0, 2}}"),
    ("Anticommutator[{{0, 1}, {1, 0}}, {{0, -I}, {I, 0}}, {Dot, 2}]", "{{0, 0}, {0, 0}}"),
    ("Anticommutator[w31x, w31y] === Anticommutator[w31y, w31x]", "True"),
    ("Attributes[Anticommutator]", "{Protected}"),
    ("Anticommutator[w31x, w31y, 1.5]", Raises(TypeError)),
    ("Anticommutator[w31x, w31y, {Dot, -2}]", Raises(ValueError)),
    ("Anticommutator[w31x, w31y, Dot, w31z]", Raises(ValueError)),
    # Ket / Bra / BraKet (inert; reference page Basic Examples with Block-local definitions)
    ("Ket[{w31k}]", "Ket[{w31k}]"),
    ("Ket[{1 + 1, w31k}]", "Ket[{2, w31k}]"),
    ("Block[{Ket}, Ket[{l___, x_ n_ ? NumberQ, r___}] := n Ket[{l, x, r}]; Ket[{2 w31j, 3 I w31k}]]",
     _scaled(6j, "Ket", "Ket[{w31j, w31k}]")),
    ("Attributes[Ket]", "{}"),
    ("ToString[Ket[{w31a, w31b}]]", text_is("Ket[{w31a, w31b}]")),
    ("Bra[{w31b}]", "Bra[{w31b}]"),
    ("Bra[{2^3}]", "Bra[{8}]"),
    ("Block[{Bra}, Bra[{l___, x_ n_ ? NumberQ, r___}] := Conjugate[n] Bra[{l, x, r}]; Bra[{2 w31j, 3 I w31k}]]",
     _scaled(-6j, "Bra", "Bra[{w31j, w31k}]")),
    ("Attributes[Bra]", "{}"),
    ("ToString[Bra[{w31a}]]", text_is("Bra[{w31a}]")),
    ("BraKet[{w31b}, {w31k}]", "BraKet[{w31b}, {w31k}]"),
    ("BraKet[{1 + 1}, {w31k}]", "BraKet[{2}, {w31k}]"),
    ("Block[{BraKet}, BraKet[{j1_, m1_}, {j2_, m2_}] := Subscript[w31d, j1, j2] Subscript[w31d, m1, m2]; "
     "BraKet[{w31j, w31m}, {w31j, w31n}]]", "Subscript[w31d, w31j, w31j] Subscript[w31d, w31m, w31n]"),
    ("Attributes[BraKet]", "{}"),
    # AngleBracket (matchfix)
    ("AngleBracket[w31x, 1 + 1]", "AngleBracket[w31x, 2]"),
    ("Hold[\u2329w31x, w31y, w31z\u232a] === Hold[AngleBracket[w31x, w31y, w31z]]", "True"),
    ("Hold[\\[LeftAngleBracket]w31x\\[RightAngleBracket]] === Hold[AngleBracket[w31x]]", "True"),
    ("Hold[2 \u2329w31x\u232a] === Hold[Times[2, AngleBracket[w31x]]]", "True"),
    ("Hold[\u2329\u232a] === Hold[AngleBracket[]]", "True"),
    ("ToString[AngleBracket[w31x, w31y]]", text_is("\u2329w31x, w31y\u232a")),
    ("ToExpression[ToString[AngleBracket[w31x, w31y + 1]]] === AngleBracket[w31x, w31y + 1]", "True"),
    ("ToString[AngleBracket[w31x], CharacterEncoding -> \"ASCII\"]",
     text_is("\\[LeftAngleBracket]w31x\\[RightAngleBracket]")),
    ("Attributes[AngleBracket]", "{}"),
    # NotExists (inert; its prefix input form is not parsed here)
    ("NotExists[w31x, 1 + 1]", "NotExists[w31x, 2]"),
    ("NotExists[w31x, w31x > 0]", Pred(lambda g: isinstance(g, Expr) and g.head == "NotExists", "unevaluated")),
    ("ToString[NotExists[w31x, w31y]]", text_is("NotExists[w31x, w31y]")),
    ("Attributes[NotExists]", "{}"),
    # MinusPlus: prefix form and grouping with + and - (same precedence, left)
    ("Hold[\u2213w31x]", "Hold[MinusPlus[w31x]]"),
    ("Hold[\\[MinusPlus]w31x^2]", "Hold[MinusPlus[w31x^2]]"),
    ("Hold[w31a + w31b \\[MinusPlus] w31c]", "Hold[MinusPlus[Plus[w31a, w31b], w31c]]"),
    ("Hold[w31a \\[MinusPlus] w31b + w31c]", "Hold[Plus[MinusPlus[w31a, w31b], w31c]]"),
    ("ToString[MinusPlus[w31x]]", text_is("\u2213w31x")),
    ("ToExpression[ToString[MinusPlus[w31x + 1]]] === MinusPlus[w31x + 1]", "True"),
    # user definitions (the reference pages define CirclePlus etc. this way)
    ("Block[{CirclePlus}, CirclePlus[a_, b_] := a + b; w31a \u2295 1 \u2295 2]", "w31a \u2295 1 \u2295 2"),
    ("Block[{CirclePlus}, CirclePlus[a_, b_] := a + b; 1 \u2295 2]", "3"),
    # band between Times and Divide: Times groups outside, "/" is not mixed without parentheses
    ("Hold[w31a \\[CenterDot] w31b w31c]", "Hold[Times[CenterDot[w31a, w31b], w31c]]"),
    ("Hold[(w31a \\[CenterDot] w31b)/w31c]", "Hold[Times[CenterDot[w31a, w31b], Power[w31c, -1]]]"),
    ("ToString[CircleTimes[w31a, w31b]/w31c]", text_is("(w31a \u2297 w31b)/w31c")),
    ("ToExpression[ToString[CircleTimes[w31a, w31b] w31c]] === CircleTimes[w31a, w31b] w31c", "True"),
    ("Hold[w31a \\[CircleDot] w31b \\[SmallCircle] w31c]", "Hold[CircleDot[w31a, SmallCircle[w31b, w31c]]]"),
    ("Hold[w31a \\[CirclePlus] w31b \\[CircleTimes] w31c]", "Hold[CirclePlus[w31a, CircleTimes[w31b, w31c]]]"),
    ("Hold[(w31a \\[CirclePlus] w31b) \\[CircleMinus] w31c]",
     "Hold[CircleMinus[CirclePlus[w31a, w31b], w31c]]"),
    ("Hold[(w31a ;; w31b) \\[SquareIntersection] w31c]", "Hold[SquareIntersection[Span[w31a, w31b], w31c]]"),
    # Therefore / Because with assignment and //
    ("Hold[w31x = w31a \\[Therefore] w31b]", "Hold[Set[w31x, Therefore[w31a, w31b]]]"),
    ("Hold[w31a \\[Because] w31b // w31f]", "Hold[Because[w31a, w31f[w31b]]]"),
    # \[GreaterSlantEqual] / \[LessSlantEqual] input is GreaterEqual / LessEqual (Wolfram)
    ("Hold[w31a \u2a7e w31b \u2a7e w31c] === Hold[w31a >= w31b >= w31c]", "True"),
    ("3 \u2a7d 4", "True"),
    # unchanged existing syntax
    ("Hold[w31a + w31b * w31c / w31d]", "Hold[Plus[w31a, Times[Times[w31b, w31c], Power[w31d, -1]]]]"),
    ("Hold[w31a < w31b == w31c]", "Hold[And[Less[w31a, w31b], Equal[w31b, w31c]]]"),
]

# Different operators of one precedence level are not mixed without parentheses (ParseError), so nothing is read
# differently from Wolfram.
PARSE_ERRORS = ["w31a \u2295 w31b \u2296 w31c", "w31a \u227b w31b == w31c", "w31a == w31b \u227b w31c",
                "w31a \u00b7 w31b / w31c", "w31a / w31b \u00b7 w31c", "w31a ;; w31b \u2293 w31c",
                "w31a \u2234 w31b \u2235 w31c", "w31a \u2295 w31b \u2295 w31c \u2296 w31d",
                "w31a \u227b w31b \u2aaf w31c", "\u2329w31a"]

CASES = build("w31", REF) + tex_cases("w31", TEX) + SPECIAL + [(s, Raises(ParseError)) for s in PARSE_ERRORS]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_w31(case):
    check(case)


def test_w31_every_symbol_has_three_cases():
    import re
    from mikoshilang import library
    from mikoshilang.library import wl_notation_rel
    names = [n for n, (area, _s, _n) in library.STATUS.items() if area == wl_notation_rel.AREA]
    assert len(names) == 131
    counts = {n: sum(1 for c in CASES if re.search(r"(?<![A-Za-z0-9])" + n + r"(?![A-Za-z0-9])", c[0]))
              for n in names}
    assert all(v >= 3 for v in counts.values()), {k: v for k, v in counts.items() if v < 3}


def test_w31_parser_tables_match_reference():
    """The parser's operator table agrees with the Wolfram specification table (characters, precedences,
    grouping)."""
    from mikoshilang import notation_ops as N
    for name, (ch, prec, affix, assoc) in REF.items():
        if name == "NotExists":
            assert name not in N.OPS          # prefix form not parsed (documented)
            continue
        mch, mprec, kind, _tex, batch = N.OPS[name]
        assert batch == "W31" and mch == ch, name
        if affix == "Prefix":
            assert kind == "prefix" and abs(mprec - prec) <= 10, name
            continue
        assert mprec == prec, name
        if name not in ("GreaterSlantEqual", "LessSlantEqual"):
            assert kind == {"None": "nary", "Left": "left", "Right": "right"}[assoc], name
