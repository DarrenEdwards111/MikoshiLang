"""Operator precedence audit: one test for every adjacent pair of precedence levels the parser supports.

Reference: the operator table of the Wolfram Language Specification
(https://wltools.github.io/LanguageSpec/Specification/Syntax/OperatorTableHTML/, column "Precedence[] Corrected"),
which agrees with the order of the "Operator Input Forms" tutorial
(https://reference.wolfram.com/language/tutorial/OperatorInputForms.html).  The LEVELS table below is written out
here, independently of the parser and of mikoshilang.notation_ops.

For two adjacent levels LO < HI (HI binds tighter):

* infix / infix:     ``a LO b HI c`` is LO[a, HI[b, c]]  and  ``a HI b LO c`` is LO[HI[a, b], c];
* infix / prefix:    ``HI a LO b`` is LO[HI[a], b];        prefix / infix:   ``LO a HI b`` is LO[HI[a, b]];
* infix / postfix:   ``a LO b HI`` is LO[a, HI[b]];        postfix / infix:  ``a HI b LO`` is LO[HI[a, b]].

Levels that need special operands (Put's file name, ``//``'s function, ``x : patt``) and the deliberate parse
errors (MikoshiLang does not mix the CenterDot..Backslash band with ``/``) are covered by explicit cases.
"""
import pytest

from mikoshilang.expr import Expr, Symbol
from mikoshilang.parser import ParseError, parse


def ff(x):
    if isinstance(x, Expr):
        h = x.head if isinstance(x.head, str) else ff(x.head)
        return f"{h}[{', '.join(ff(a) for a in x.args)}]"
    if isinstance(x, Symbol):
        return x.name
    return repr(x)


def B(name):
    return lambda x, y: f"{name}[{x}, {y}]"


def U(name):
    return lambda x: f"{name}[{x}]"


# (precedence, kind, operator text, FullForm builder)
LEVELS = [
    (10, "infix", ";", B("CompoundExpression")),
    (40, "infix", "=", B("Set")),
    (50, "infix", "\u2235", B("Because")),
    (60, "infix", "\\[VerticalSeparator]", B("VerticalSeparator")),
    (80, "infix", "\u2236", B("Colon")),
    (90, "postfix", "&", U("Function")),
    (100, "infix", "+=", B("AddTo")),
    (110, "infix", "/.", B("ReplaceAll")),
    (120, "infix", "->", B("Rule")),
    (125, "infix", "<->", B("TwoWayRule")),
    (130, "infix", "/;", B("Condition")),
    (135, "infix", "~~", B("StringExpression")),
    (160, "infix", "|", B("Alternatives")),
    (170, "postfix", "..", U("Repeated")),
    (190, "infix", "\u22a2", B("RightTee")),
    (210, "infix", "||", B("Or")),
    (220, "infix", "&&", B("And")),
    (230, "prefix", "!", U("Not")),
    (250, "infix", "\u2209", B("NotElement")),
    (260, "infix", "===", B("SameQ")),
    (270, "infix", "\u2192", B("RightArrow")),
    (280, "infix", "\u2225", B("DoubleVerticalBar")),
    (290, "infix", "==", B("Equal")),
    (295, "infix", "\\[DirectedEdge]", B("DirectedEdge")),
    (300, "infix", "\u2294", B("SquareUnion")),
    (305, "infix", "\u2293", B("SquareIntersection")),
    (310, "infix", "+", B("Plus")),
    (330, "infix", "\u2295", B("CirclePlus")),
    (340, "infix", "\u2323", B("Cup")),
    (350, "infix", "\u2322", B("Cap")),
    (360, "infix", "\u2210", B("Coproduct")),
    (370, "infix", "\u2240", B("VerticalTilde")),
    (390, "infix", "\u22c6", B("Star")),
    (400, "infix", "*", B("Times")),
    (410, "infix", "\u00b7", B("CenterDot")),
    (420, "infix", "\u2297", B("CircleTimes")),
    (430, "infix", "\u22c1", B("Vee")),
    (450, "infix", "\u22c4", B("Diamond")),
    (460, "infix", "\u2216", B("Backslash")),
    (470, "infix", "/", lambda x, y: f"Times[{x}, Power[{y}, -1]]"),
    (480, "prefix", "-", lambda x: f"Times[-1, {x}]"),
    (490, "infix", ".", B("Dot")),
    (510, "infix", "**", B("NonCommutativeMultiply")),
    (520, "infix", "\u2299", B("CircleDot")),
    (530, "infix", "\u2218", B("SmallCircle")),
    (540, "prefix", "\\[Square]", U("Square")),
    (550, "prefix", "\u2207", U("Del")),
    (580, "infix", "\u2191", B("UpArrow")),
    (590, "infix", "^", B("Power")),
    (600, "infix", "<>", B("StringJoin")),
    (610, "postfix", "!", U("Factorial")),
    (620, "infix", "/@", B("Map")),
    (640, "infix", "@", lambda x, y: f"{x}[{y}]"),
    (650, "infix", "/*", B("RightComposition")),
    (655, "infix", "@*", B("Composition")),
    (665, "postfix", "++", U("Increment")),
]

# Pairs that MikoshiLang deliberately rejects (a clear ParseError instead of a guess): the CenterDot..Backslash band
# (410-460) lies between Times (400) and Divide (470) in Wolfram, but the parser reads * and / in one loop.
REJECTED = {(460, 470)}


def _pair_cases():
    out = []
    for lo, hi in zip(LEVELS, LEVELS[1:]):
        (pl, kl, ol, fl), (ph, kh, oh, fh) = lo, hi
        tag = f"{pl}-{ph}"
        if (pl, ph) in REJECTED:
            out.append(pytest.param(f"a {ol} b {oh} c", ParseError, id=tag + "-lohi"))
            out.append(pytest.param(f"a {oh} b {ol} c", ParseError, id=tag + "-hilo"))
            continue
        if kl == "infix" and kh == "infix":
            out.append(pytest.param(f"a {ol} b {oh} c", fl("a", fh("b", "c")), id=tag + "-lohi"))
            out.append(pytest.param(f"a {oh} b {ol} c", fl(fh("a", "b"), "c"), id=tag + "-hilo"))
        elif kl == "infix" and kh == "prefix":
            out.append(pytest.param(f"{oh} a {ol} b", fl(fh("a"), "b"), id=tag))
        elif kl == "prefix" and kh == "infix":
            out.append(pytest.param(f"{ol} a {oh} b", fl(fh("a", "b")), id=tag))
        elif kl == "infix" and kh == "postfix":
            out.append(pytest.param(f"a {ol} b{oh}", fl("a", fh("b")), id=tag))
        elif kl == "postfix" and kh == "infix":
            out.append(pytest.param(f"a {oh} b{ol}", fl(fh("a", "b")), id=tag))
        elif kl == "prefix" and kh == "prefix":
            out.append(pytest.param(f"{ol} {oh} a", fl(fh("a")), id=tag))
        else:  # pragma: no cover - the table has no other combination
            raise AssertionError(tag)
    return out


def test_levels_are_strictly_increasing():
    precs = [p for p, *_ in LEVELS]
    assert precs == sorted(precs) and len(set(precs)) == len(precs)


@pytest.mark.parametrize("src, expected", _pair_cases())
def test_adjacent_levels(src, expected):
    if expected is ParseError:
        with pytest.raises(ParseError):
            parse(src)
    else:
        assert ff(parse(src)) == expected


# Levels with special operands, and the ends of the table.
SPECIAL = [
    ("a >> f; b", "CompoundExpression[Put[a, 'f'], b]"),                 # 10 / 30
    ("a = b >> f", "Put[Set[a, b], 'f']"),                               # 30 / 40
    ("a \\[VerticalSeparator] b // f", "VerticalSeparator[a, f[b]]"),    # 60 / 70
    ("a // f \\[VerticalSeparator] b", "VerticalSeparator[f[a], b]"),
    ("a \u2236 b // f", "f[Colon[a, b]]"),                                # 70 / 80
    ("a // f \u2236 b", "Colon[f, b][a]"),
    ("x : a | b", "Pattern[x, Alternatives[a, b]]"),                     # 150 / 160
    ("x : a ~~ b", "StringExpression[Pattern[x, a], b]"),                # 135 / 150
    ("f[x]++", "Increment[f[x]]"),                                       # 665 / 670
    ("a @* f[x]", "Composition[a, f[x]]"),                               # 655 / 670
]


@pytest.mark.parametrize("src, expected", SPECIAL)
def test_special_levels(src, expected):
    assert ff(parse(src)) == expected


# Grouping within one level.
ASSOCIATIVITY = [
    ("a ^ b ^ c", "Power[a, Power[b, c]]"),
    ("a = b = c", "Set[a, Set[b, c]]"),
    ("a -> b -> c", "Rule[a, Rule[b, c]]"),
    ("a <-> b <-> c", "TwoWayRule[a, TwoWayRule[b, c]]"),
    ("a += b += c", "AddTo[a, AddTo[b, c]]"),
    ("f /@ g /@ x", "Map[f, Map[g, x]]"),
    ("f @ g @ x", "f[g[x]]"),
    ("a === b === c", "SameQ[a, b, c]"),
    ("a =!= b =!= c", "UnsameQ[a, b, c]"),
    ("a <> b <> c", "StringJoin[a, b, c]"),
    ("f @* g @* h", "Composition[f, g, h]"),
    ("f /* g /* h", "RightComposition[f, g, h]"),
    ("a /. b /. c", "ReplaceAll[ReplaceAll[a, b], c]"),
    ("a /; b /; c", "Condition[Condition[a, b], c]"),
    ("a \u2235 b \u2235 c", "Because[Because[a, b], c]"),
    ("a \u2234 b \u2234 c", "Therefore[a, Therefore[b, c]]"),
    ("a ** b ** c", "NonCommutativeMultiply[a, b, c]"),
    ("a . b . c", "Dot[a, b, c]"),
]


@pytest.mark.parametrize("src, expected", ASSOCIATIVITY)
def test_associativity(src, expected):
    assert ff(parse(src)) == expected


# Regressions for the precedence bugs fixed in this audit (each was parsed differently before).
REGRESSIONS = [
    # Power (590) binds tighter than NonCommutativeMultiply (510), which binds tighter than Dot (490).
    ("a ** b ^ c . d", "Dot[NonCommutativeMultiply[a, Power[b, c]], d]"),
    ("b ^ c . d", "Dot[Power[b, c], d]"),
    ("a ^ b ** c", "NonCommutativeMultiply[Power[a, b], c]"),
    ("a ^ -b c", "Times[Power[a, Times[-1, b]], c]"),
    ("a ^ -b . c", "Dot[Power[a, Times[-1, b]], c]"),
    ("2 ^ -1", "Power[2, -1]"),
    # StringJoin (600) binds tighter than Plus and Power.
    ("a <> b + c", "Plus[StringJoin[a, b], c]"),
    ("a <> b ^ c", "Power[StringJoin[a, b], c]"),
    # Factorial (610) is below Map (620) and @ (640); n!! is Factorial2.
    ("f @ x!", "Factorial[f[x]]"),
    ("2 ^ 3!", "Power[2, Factorial[3]]"),
    ("5!!", "Factorial2[5]"),
    ("(5!)!", "Factorial[Factorial[5]]"),
    ("!!a", "Not[Not[a]]"),
    # Map (620) binds tighter than Power (590); @ (640) tighter than Map; Composition (655) tighter than @.
    ("f /@ a ^ b", "Power[Map[f, a], b]"),
    ("f @* g @ x", "Composition[f, g][x]"),
    ("f /* g @* h", "RightComposition[f, Composition[g, h]]"),
    ("f @ -x", "f[Times[-1, x]]"),
    # SameQ (260) is below Equal (290); += (100) is between & (90) and /. (110); <-> (125) above ->.
    ("a == b === c", "SameQ[Equal[a, b], c]"),
    ("x += 1 &", "Function[AddTo[x, 1]]"),
    ("x += y // f", "f[AddTo[x, y]]"),
    ("a <-> b -> c", "Rule[TwoWayRule[a, b], c]"),
    ("a \\[DirectedEdge] b -> c", "Rule[DirectedEdge[a, b], c]"),
    ("a \\[DirectedEdge] b == c", "Equal[DirectedEdge[a, b], c]"),
]


@pytest.mark.parametrize("src, expected", REGRESSIONS)
def test_precedence_regressions(src, expected):
    assert ff(parse(src)) == expected


@pytest.mark.parametrize("src", ["a === b =!= c", "a \\[Distributed] b \\[Distributed] c"])
def test_ambiguous_mixes_raise(src):
    with pytest.raises(ParseError):
        parse(src)


def test_evaluated_examples():
    from mikoshilang.evaluate import evaluate
    assert evaluate(parse("5!!")) == 15
    assert evaluate(parse("2^3!")) == 64
    assert evaluate(parse("\"a\" <> \"b\" === \"ab\"")) is True


@pytest.mark.parametrize("src", ["a <> b ^ c", "a <> (b ^ c)", "(a == b) === c", "a == (b === c)", "x += 1 &",
                                 "x += (1 &)", "(f @* g) /* h", "f @* (g /* h)"])
def test_input_form_round_trip(src):
    """InputForm text of the parsed expression parses back to the same expression."""
    from mikoshilang.formatting import input_form
    e = parse(src)
    assert ff(parse(input_form(e))) == ff(e)
