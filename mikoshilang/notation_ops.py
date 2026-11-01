"""Operator data for the Wolfram notation operators (batches W31 and W32): pure data, no imports.

Used by the parser (tokens and precedence levels) and by ``library.wl_notation_rel`` / ``library.wl_notation_arrows``
(InputForm/OutputForm/TeXForm printing).  Each entry is

    name: (character, precedence, kind, TeX macro or None, batch)

* ``character`` is Wolfram's character for ``\\[name]`` (some are in the Unicode private use area, as in Wolfram);
* ``precedence`` is Wolfram's operator precedence (reference: the operator table in the Wolfram Language
  Specification, https://wltools.github.io/LanguageSpec/Specification/Syntax/OperatorTableHTML/, which agrees with
  the "Operator Input Forms" tutorial, https://reference.wolfram.com/language/tutorial/OperatorInputForms.html);
* ``kind`` is how the infix form groups: ``nary`` (``x op y op z`` is ``name[x, y, z]``), ``left`` / ``right``
  (binary, ``(x op y) op z`` / ``x op (y op z)``), ``prefix`` (``op x``), or ``alias`` (the character is read as a
  different head: Wolfram reads \\[GreaterSlantEqual] and \\[LessSlantEqual] as GreaterEqual and LessEqual, so
  these heads have no input operator of their own).
"""

OPS = {
    "NotElement": ("\u2209", 250, "nary", None, "W31"),
    "Backslash": ("\u2216", 460, "nary", '\\backslash', "W31"),
    "Because": ("\u2235", 50, "left", '\\because', "W31"),
    "Cap": ("\u2322", 350, "nary", '\\frown', "W31"),
    "CapitalDifferentialD": ("\uf74b", 550, "prefix", None, "W31"),
    "CenterDot": ("\u00b7", 410, "nary", '\\cdot', "W31"),
    "CircleDot": ("\u2299", 520, "nary", '\\odot', "W31"),
    "CircleMinus": ("\u2296", 330, "left", '\\ominus', "W31"),
    "CirclePlus": ("\u2295", 330, "nary", '\\oplus', "W31"),
    "CircleTimes": ("\u2297", 420, "nary", '\\otimes', "W31"),
    "Colon": ("\u2236", 80, "nary", None, "W31"),
    "Coproduct": ("\u2210", 360, "nary", '\\amalg', "W31"),
    "Cup": ("\u2323", 340, "nary", '\\smile', "W31"),
    "CupCap": ("\u224d", 290, "nary", '\\asymp', "W31"),
    "Diamond": ("\u22c4", 450, "nary", '\\diamond', "W31"),
    "DifferentialD": ("\uf74c", 550, "prefix", None, "W31"),
    "DotEqual": ("\u2250", 290, "nary", '\\doteq', "W31"),
    "DoubleVerticalBar": ("\u2225", 280, "nary", '\\parallel', "W31"),
    "EqualTilde": ("\u2242", 290, "nary", '\\eqsim', "W31"),
    "GreaterEqualLess": ("\u22db", 290, "nary", '\\gtreqless', "W31"),
    "GreaterFullEqual": ("\u2267", 290, "nary", '\\geqq', "W31"),
    "GreaterGreater": ("\u226b", 290, "nary", '\\gg', "W31"),
    "GreaterLess": ("\u2277", 290, "nary", '\\gtrless', "W31"),
    "GreaterSlantEqual": ("\u2a7e", 290, "alias", '\\geqslant', "W31"),
    "GreaterTilde": ("\u2273", 290, "nary", '\\gtrsim', "W31"),
    "HumpDownHump": ("\u224e", 290, "nary", '\\Bumpeq', "W31"),
    "HumpEqual": ("\u224f", 290, "nary", '\\bumpeq', "W31"),
    "LeftTriangle": ("\u22b2", 290, "nary", '\\vartriangleleft', "W31"),
    "LeftTriangleBar": ("\u29cf", 290, "nary", None, "W31"),
    "LeftTriangleEqual": ("\u22b4", 290, "nary", '\\trianglelefteq', "W31"),
    "LessEqualGreater": ("\u22da", 290, "nary", '\\lesseqgtr', "W31"),
    "LessFullEqual": ("\u2266", 290, "nary", '\\leqq', "W31"),
    "LessGreater": ("\u2276", 290, "nary", '\\lessgtr', "W31"),
    "LessLess": ("\u226a", 290, "nary", '\\ll', "W31"),
    "LessSlantEqual": ("\u2a7d", 290, "alias", '\\leqslant', "W31"),
    "LessTilde": ("\u2272", 290, "nary", '\\lesssim', "W31"),
    "MinusPlus": ("\u2213", 310, "left", '\\mp', "W31"),
    "NestedGreaterGreater": ("\u2aa2", 290, "nary", None, "W31"),
    "NestedLessLess": ("\u2aa1", 290, "nary", None, "W31"),
    "NotCongruent": ("\u2262", 290, "nary", None, "W31"),
    "NotCupCap": ("\u226d", 290, "nary", None, "W31"),
    "NotDoubleVerticalBar": ("\u2226", 280, "nary", '\\nparallel', "W31"),
    "NotEqualTilde": ("\uf400", 290, "nary", None, "W31"),
    "NotGreater": ("\u226f", 290, "nary", '\\ngtr', "W31"),
    "NotGreaterEqual": ("\u2271", 290, "nary", '\\ngeq', "W31"),
    "NotGreaterFullEqual": ("\u2269", 290, "nary", '\\ngeqq', "W31"),
    "NotGreaterGreater": ("\uf427", 290, "nary", None, "W31"),
    "NotGreaterLess": ("\u2279", 290, "nary", None, "W31"),
    "NotGreaterSlantEqual": ("\uf429", 290, "nary", '\\ngeqslant', "W31"),
    "NotGreaterTilde": ("\u2275", 290, "nary", None, "W31"),
    "NotHumpDownHump": ("\uf402", 290, "nary", None, "W31"),
    "NotHumpEqual": ("\uf401", 290, "nary", None, "W31"),
    "NotLeftTriangle": ("\u22ea", 290, "nary", '\\ntriangleleft', "W31"),
    "NotLeftTriangleBar": ("\uf412", 290, "nary", None, "W31"),
    "NotLeftTriangleEqual": ("\u22ec", 290, "nary", '\\ntrianglelefteq', "W31"),
    "NotLess": ("\u226e", 290, "nary", '\\nless', "W31"),
    "NotLessEqual": ("\u2270", 290, "nary", '\\nleq', "W31"),
    "NotLessFullEqual": ("\u2268", 290, "nary", '\\nleqq', "W31"),
    "NotLessGreater": ("\u2278", 290, "nary", None, "W31"),
    "NotLessLess": ("\uf422", 290, "nary", None, "W31"),
    "NotLessSlantEqual": ("\uf424", 290, "nary", '\\nleqslant', "W31"),
    "NotLessTilde": ("\u2274", 290, "nary", None, "W31"),
    "NotNestedGreaterGreater": ("\uf428", 290, "nary", None, "W31"),
    "NotNestedLessLess": ("\uf423", 290, "nary", None, "W31"),
    "NotPrecedes": ("\u2280", 290, "nary", '\\nprec', "W31"),
    "NotPrecedesEqual": ("\uf42b", 290, "nary", None, "W31"),
    "NotPrecedesSlantEqual": ("\u22e0", 290, "nary", None, "W31"),
    "NotPrecedesTilde": ("\u22e8", 290, "nary", None, "W31"),
    "NotReverseElement": ("\u220c", 250, "nary", '\\not\\ni', "W31"),
    "NotRightTriangle": ("\u22eb", 290, "nary", '\\ntriangleright', "W31"),
    "NotRightTriangleBar": ("\uf413", 290, "nary", None, "W31"),
    "NotRightTriangleEqual": ("\u22ed", 290, "nary", '\\ntrianglerighteq', "W31"),
    "NotSquareSubset": ("\uf42e", 250, "nary", '\\not\\sqsubset', "W31"),
    "NotSquareSubsetEqual": ("\u22e2", 250, "nary", '\\not\\sqsubseteq', "W31"),
    "NotSquareSuperset": ("\uf42f", 250, "nary", '\\not\\sqsupset', "W31"),
    "NotSquareSupersetEqual": ("\u22e3", 250, "nary", '\\not\\sqsupseteq', "W31"),
    "NotSubset": ("\u2284", 250, "nary", '\\not\\subset', "W31"),
    "NotSubsetEqual": ("\u2288", 250, "nary", '\\nsubseteq', "W31"),
    "NotSucceeds": ("\u2281", 290, "nary", '\\nsucc', "W31"),
    "NotSucceedsEqual": ("\uf42d", 290, "nary", None, "W31"),
    "NotSucceedsSlantEqual": ("\u22e1", 290, "nary", None, "W31"),
    "NotSucceedsTilde": ("\u22e9", 290, "nary", None, "W31"),
    "NotSuperset": ("\u2285", 250, "nary", '\\not\\supset', "W31"),
    "NotSupersetEqual": ("\u2289", 250, "nary", '\\nsupseteq', "W31"),
    "NotTilde": ("\u2241", 290, "nary", '\\nsim', "W31"),
    "NotTildeEqual": ("\u2244", 290, "nary", None, "W31"),
    "NotTildeFullEqual": ("\u2247", 290, "nary", '\\ncong', "W31"),
    "NotTildeTilde": ("\u2249", 290, "nary", '\\not\\approx', "W31"),
    "NotVerticalBar": ("\uf3d1", 280, "nary", '\\nmid', "W31"),
    "PrecedesEqual": ("\u2aaf", 290, "nary", '\\preceq', "W31"),
    "PrecedesSlantEqual": ("\u227c", 290, "nary", '\\preccurlyeq', "W31"),
    "PrecedesTilde": ("\u227e", 290, "nary", '\\precsim', "W31"),
    "Proportion": ("\u2237", 290, "nary", None, "W31"),
    "Proportional": ("\u221d", 290, "nary", None, "W31"),
    "ReverseElement": ("\u220b", 250, "nary", '\\ni', "W31"),
    "RightTriangle": ("\u22b3", 290, "nary", '\\vartriangleright', "W31"),
    "RightTriangleBar": ("\u29d0", 290, "nary", None, "W31"),
    "RightTriangleEqual": ("\u22b5", 290, "nary", '\\trianglerighteq', "W31"),
    "SmallCircle": ("\u2218", 530, "nary", '\\circ', "W31"),
    "SquareIntersection": ("\u2293", 305, "nary", '\\sqcap', "W31"),
    "SquareSubset": ("\u228f", 250, "nary", '\\sqsubset', "W31"),
    "SquareSubsetEqual": ("\u2291", 250, "nary", '\\sqsubseteq', "W31"),
    "SquareSuperset": ("\u2290", 250, "nary", '\\sqsupset', "W31"),
    "SquareSupersetEqual": ("\u2292", 250, "nary", '\\sqsupseteq', "W31"),
    "SquareUnion": ("\u2294", 300, "nary", '\\sqcup', "W31"),
    "Star": ("\u22c6", 390, "nary", '\\star', "W31"),
    "Subset": ("\u2282", 250, "nary", None, "W31"),
    "SubsetEqual": ("\u2286", 250, "nary", None, "W31"),
    "Succeeds": ("\u227b", 290, "nary", '\\succ', "W31"),
    "SucceedsEqual": ("\u2ab0", 290, "nary", '\\succeq', "W31"),
    "SucceedsSlantEqual": ("\u227d", 290, "nary", '\\succcurlyeq', "W31"),
    "SucceedsTilde": ("\u227f", 290, "nary", '\\succsim', "W31"),
    "Superset": ("\u2283", 250, "nary", None, "W31"),
    "SupersetEqual": ("\u2287", 250, "nary", '\\supseteq', "W31"),
    "Therefore": ("\u2234", 50, "right", '\\therefore', "W31"),
    "Tilde": ("\u223c", 290, "nary", '\\sim', "W31"),
    "TildeEqual": ("\u2243", 290, "nary", '\\simeq', "W31"),
    "TildeFullEqual": ("\u2245", 290, "nary", '\\cong', "W31"),
    "TildeTilde": ("\u2248", 290, "nary", '\\approx', "W31"),
    "UnionPlus": ("\u228e", 300, "nary", '\\uplus', "W31"),
    "Vee": ("\u22c1", 430, "nary", '\\vee', "W31"),
    "VerticalBar": ("\uf3d0", 280, "nary", '\\mid', "W31"),
    "VerticalSeparator": ("\uf432", 60, "nary", None, "W31"),
    "VerticalTilde": ("\u2240", 370, "nary", '\\wr', "W31"),
    "DoubleDownArrow": ("\u21d3", 580, "nary", '\\Downarrow', "W32"),
    "DoubleLeftArrow": ("\u21d0", 270, "nary", '\\Leftarrow', "W32"),
    "DoubleLeftRightArrow": ("\u21d4", 270, "nary", '\\Leftrightarrow', "W32"),
    "DoubleLeftTee": ("\u2ae4", 190, "left", None, "W32"),
    "DoubleLongLeftArrow": ("\u27f8", 580, "nary", '\\Longleftarrow', "W32"),
    "DoubleLongLeftRightArrow": ("\u27fa", 580, "nary", '\\Longleftrightarrow', "W32"),
    "DoubleLongRightArrow": ("\u27f9", 580, "nary", '\\Longrightarrow', "W32"),
    "DoubleRightArrow": ("\u21d2", 270, "nary", '\\Rightarrow', "W32"),
    "DoubleRightTee": ("\u22a8", 190, "right", '\\vDash', "W32"),
    "DoubleUpArrow": ("\u21d1", 580, "nary", '\\Uparrow', "W32"),
    "DoubleUpDownArrow": ("\u21d5", 580, "nary", '\\Updownarrow', "W32"),
    "DownArrow": ("\u2193", 580, "nary", '\\downarrow', "W32"),
    "DownArrowBar": ("\u2913", 580, "nary", None, "W32"),
    "DownArrowUpArrow": ("\u21f5", 580, "nary", None, "W32"),
    "DownLeftRightVector": ("\u2950", 270, "nary", None, "W32"),
    "DownLeftTeeVector": ("\u295e", 270, "nary", None, "W32"),
    "DownLeftVector": ("\u21bd", 270, "nary", '\\leftharpoondown', "W32"),
    "DownLeftVectorBar": ("\u2956", 270, "nary", None, "W32"),
    "DownRightTeeVector": ("\u295f", 270, "nary", None, "W32"),
    "DownRightVector": ("\u21c1", 270, "nary", '\\rightharpoondown', "W32"),
    "DownRightVectorBar": ("\u2957", 270, "nary", None, "W32"),
    "DownTee": ("\u22a4", 190, "left", '\\top', "W32"),
    "DownTeeArrow": ("\u21a7", 580, "nary", None, "W32"),
    "Equilibrium": ("\u21cc", 290, "nary", '\\rightleftharpoons', "W32"),
    "LeftArrow": ("\u2190", 270, "nary", '\\leftarrow', "W32"),
    "LeftArrowBar": ("\u21e4", 270, "nary", None, "W32"),
    "LeftArrowRightArrow": ("\u21c6", 270, "nary", '\\leftrightarrows', "W32"),
    "LeftDownTeeVector": ("\u2961", 580, "nary", None, "W32"),
    "LeftDownVector": ("\u21c3", 580, "nary", '\\downharpoonleft', "W32"),
    "LeftDownVectorBar": ("\u2959", 580, "nary", None, "W32"),
    "LeftRightVector": ("\u294e", 270, "nary", None, "W32"),
    "LeftTee": ("\u22a3", 190, "left", '\\dashv', "W32"),
    "LeftTeeArrow": ("\u21a4", 270, "nary", None, "W32"),
    "LeftTeeVector": ("\u295a", 270, "nary", None, "W32"),
    "LeftUpDownVector": ("\u2951", 580, "nary", None, "W32"),
    "LeftUpTeeVector": ("\u2960", 580, "nary", None, "W32"),
    "LeftUpVector": ("\u21bf", 580, "nary", '\\upharpoonleft', "W32"),
    "LeftUpVectorBar": ("\u2958", 580, "nary", None, "W32"),
    "LeftVector": ("\u21bc", 270, "nary", '\\leftharpoonup', "W32"),
    "LeftVectorBar": ("\u2952", 270, "nary", None, "W32"),
    "LongLeftArrow": ("\u27f5", 580, "nary", '\\longleftarrow', "W32"),
    "LongLeftRightArrow": ("\u27f7", 580, "nary", '\\longleftrightarrow', "W32"),
    "LongRightArrow": ("\u27f6", 580, "nary", '\\longrightarrow', "W32"),
    "LowerLeftArrow": ("\u2199", 270, "nary", '\\swarrow', "W32"),
    "LowerRightArrow": ("\u2198", 270, "nary", '\\searrow', "W32"),
    "ReverseEquilibrium": ("\u21cb", 290, "nary", '\\leftrightharpoons', "W32"),
    "ReverseUpEquilibrium": ("\u296f", 580, "nary", None, "W32"),
    "RightArrowBar": ("\u21e5", 270, "nary", None, "W32"),
    "RightArrowLeftArrow": ("\u21c4", 270, "nary", '\\rightleftarrows', "W32"),
    "RightDownTeeVector": ("\u295d", 580, "nary", None, "W32"),
    "RightDownVector": ("\u21c2", 580, "nary", '\\downharpoonright', "W32"),
    "RightDownVectorBar": ("\u2955", 580, "nary", None, "W32"),
    "RightTee": ("\u22a2", 190, "right", '\\vdash', "W32"),
    "RightTeeArrow": ("\u21a6", 270, "nary", '\\mapsto', "W32"),
    "RightTeeVector": ("\u295b", 270, "nary", None, "W32"),
    "RightUpDownVector": ("\u294f", 580, "nary", None, "W32"),
    "RightUpTeeVector": ("\u295c", 580, "nary", None, "W32"),
    "RightUpVector": ("\u21be", 580, "nary", '\\upharpoonright', "W32"),
    "RightUpVectorBar": ("\u2954", 580, "nary", None, "W32"),
    "RightVector": ("\u21c0", 270, "nary", '\\rightharpoonup', "W32"),
    "RightVectorBar": ("\u2953", 270, "nary", None, "W32"),
    "ShortDownArrow": ("\uf52b", 580, "nary", None, "W32"),
    "ShortLeftArrow": ("\uf526", 270, "nary", None, "W32"),
    "ShortRightArrow": ("\uf525", 270, "nary", None, "W32"),
    "ShortUpArrow": ("\uf52a", 580, "nary", None, "W32"),
    "UpArrowBar": ("\u2912", 580, "nary", None, "W32"),
    "UpArrowDownArrow": ("\u21c5", 580, "nary", None, "W32"),
    "UpDownArrow": ("\u2195", 580, "nary", '\\updownarrow', "W32"),
    "UpEquilibrium": ("\u296e", 580, "nary", None, "W32"),
    "UpTee": ("\u22a5", 190, "left", '\\bot', "W32"),
    "UpTeeArrow": ("\u21a5", 580, "nary", None, "W32"),
    "UpperLeftArrow": ("\u2196", 270, "nary", '\\nwarrow', "W32"),
    "UpperRightArrow": ("\u2197", 270, "nary", '\\nearrow', "W32"),
    # Operators added in the correctness fix-up pass (FIX1): left out of the W31/W32 plan.
    "Precedes": ("\u227a", 290, "nary", '\\prec', "FIX1"),
    "Congruent": ("\u2261", 290, "nary", '\\equiv', "FIX1"),
    "RightArrow": ("\u2192", 270, "nary", '\\rightarrow', "FIX1"),
    "LeftRightArrow": ("\u2194", 270, "nary", '\\leftrightarrow', "FIX1"),
    "UpArrow": ("\u2191", 580, "nary", '\\uparrow', "FIX1"),
    "PlusMinus": ("\u00b1", 310, "left", '\\pm', "FIX1"),
    "Del": ("\u2207", 550, "prefix", None, "FIX1"),
    "Square": ("\uf520", 540, "prefix", None, "FIX1"),
}

# Extra input characters: standard Unicode forms accepted on input (Wolfram's documentation pages show them).
EXTRA_INPUT = {"\u2146": "DifferentialD", "\u2145": "CapitalDifferentialD"}

# Characters printed for prefix operators whose Wolfram character is in the private use area.
PRINT_CHAR = {"DifferentialD": "\u2146", "CapitalDifferentialD": "\u2145"}

# Matchfix AngleBracket[x, ...]: \\[LeftAngleBracket] x, ... \\[RightAngleBracket].
LEFT_ANGLE, RIGHT_ANGLE = "\u2329", "\u232a"

# Matchfix DoubleBracketingBar[x, ...]: \\[LeftDoubleBracketingBar] x, ... \\[RightDoubleBracketingBar] (Wolfram's
# private-use characters U+F605 / U+F606).
LEFT_DBAR, RIGHT_DBAR = "\uf605", "\uf606"


def is_private(ch):
    return 0xE000 <= ord(ch) <= 0xF8FF


def print_char(name):
    """The operator text printed for ``name``: its character, or ``\\[name]`` for private-use characters."""
    if name in PRINT_CHAR:
        return PRINT_CHAR[name]
    ch = OPS[name][0]
    return "\\[" + name + "]" if is_private(ch) else ch


def input_map():
    """Every input spelling (long name ``\\[name]`` and character) -> head name."""
    m = {}
    for name, (ch, _p, kind, _t, _b) in OPS.items():
        target = {"GreaterSlantEqual": "GreaterEqual", "LessSlantEqual": "LessEqual"}.get(name, name)
        m["\\[" + name + "]"] = target
        m[ch] = target
    for ch, name in EXTRA_INPUT.items():
        m[ch] = name
    return m
