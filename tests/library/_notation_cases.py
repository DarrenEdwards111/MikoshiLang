"""Case builder shared by the W31 / W32 notation-operator tests (test_wolfram_w31.py, test_wolfram_w32.py).

``REF`` tables in the test modules give, for each operator, Wolfram's character, its precedence, affix and
associativity as listed in the operator table of the Wolfram Language Specification
(https://wltools.github.io/LanguageSpec/Specification/Syntax/OperatorTableHTML/, "Precedence[] Corrected",
"Affix" and "Associativity" columns) with the characters from Wolfram's named-character list
(https://reference.wolfram.com/language/guide/ListingOfNamedCharacters.html; "wl-unicode" in the Mathics3
named-characters table).  They are written out in the tests, independently of mikoshilang.notation_ops.

Each Wolfram reference page says "<Name>[x, y, ...] has no built-in meaning" and "x <op> y is by default
interpreted as <Name>[x, y]"; the cases check that

* the head stays unevaluated with evaluated arguments (inert), with Attributes {} (the documentation attaches
  definitions with ``Name[...] := ...`` and removes them with ``Clear[Name]``, so the symbol is unprotected);
* ``\\[Name]`` and the character parse to the head, chains group as the table says (Associativity None: one flat
  call; Left / Right: nested), and the precedence places it between the nearest existing operators below and above;
* InputForm text (ToString) uses the operator (``\\[Name]`` for private-use characters), ToExpression reads it back,
  CharacterEncoding -> "ASCII" gives the long name, one-argument calls print in functional form.

Placeholder symbols are prefixed with the batch tag (w31 / w32) and never assigned.
"""
from _harness import Pred

# Existing operators just below / above each precedence (Wolfram precedences): (source, held FullForm) templates in
# terms of {A}, {B}, {C} and {OP} (the operator text) / {H} (the head).
_BELOW = {
    50: ("{A} = {B} {OP} {C}", "Set[{A}, {H}[{B}, {C}]]"),
    60: ("{A} = {B} {OP} {C}", "Set[{A}, {H}[{B}, {C}]]"),
    80: ("{A} // {B} {OP} {C}", None),
    190: ("{A} | {B} {OP} {C}", "Alternatives[{A}, {H}[{B}, {C}]]"),
    250: ("{A} && {B} {OP} {C}", "And[{A}, {H}[{B}, {C}]]"),
    270: ("{A} && {B} {OP} {C}", "And[{A}, {H}[{B}, {C}]]"),
    280: ("{A} && {B} {OP} {C}", "And[{A}, {H}[{B}, {C}]]"),
    290: ("{A} && {B} {OP} {C}", "And[{A}, {H}[{B}, {C}]]"),
    300: ("{A} == {B} {OP} {C}", "Equal[{A}, {H}[{B}, {C}]]"),
    305: ("{A} == {B} {OP} {C}", "Equal[{A}, {H}[{B}, {C}]]"),
    310: ("{A} == {B} {OP} {C}", "Equal[{A}, {H}[{B}, {C}]]"),
    330: ("{A} + {B} {OP} {C}", "Plus[{A}, {H}[{B}, {C}]]"),
    410: ("{A} * {B} {OP} {C}", "Times[{A}, {H}[{B}, {C}]]"),
    520: ("{A} ** {B} {OP} {C}", "NonCommutativeMultiply[{A}, {H}[{B}, {C}]]"),
    580: ("{A} ** {B} {OP} {C}", "NonCommutativeMultiply[{A}, {H}[{B}, {C}]]"),
}
_ABOVE = {
    50: ("{A} {OP} {B} // {C}", "{H}[{A}, {C}[{B}]]"),
    60: ("{A} {OP} {B} // {C}", "{H}[{A}, {C}[{B}]]"),
    80: ("{A} {OP} {B} -> {C}", "{H}[{A}, Rule[{B}, {C}]]"),
    190: ("{A} {OP} {B} || {C}", "{H}[{A}, Or[{B}, {C}]]"),
    250: ("{A} {OP} {B} == {C}", "{H}[{A}, Equal[{B}, {C}]]"),
    270: ("{A} {OP} {B} == {C}", "{H}[{A}, Equal[{B}, {C}]]"),
    280: ("{A} {OP} {B} == {C}", "{H}[{A}, Equal[{B}, {C}]]"),
    290: ("{A} {OP} {B} + {C}", "{H}[{A}, Plus[{B}, {C}]]"),
    300: ("{A} {OP} {B} + {C}", "{H}[{A}, Plus[{B}, {C}]]"),
    305: ("{A} {OP} {B} + {C}", "{H}[{A}, Plus[{B}, {C}]]"),
    310: ("{A} {OP} {B} * {C}", "{H}[{A}, Times[{B}, {C}]]"),
    330: ("{A} {OP} {B} * {C}", "{H}[{A}, Times[{B}, {C}]]"),
    410: ("{A} {OP} {B} . {C}", "{H}[{A}, Dot[{B}, {C}]]"),
    520: ("{A} {OP} {B} ^ {C}", "{H}[{A}, Power[{B}, {C}]]"),
    580: ("{A} {OP} {B} ^ {C}", "{H}[{A}, Power[{B}, {C}]]"),
}


def _band(prec):
    for p in (580, 520, 410, 330, 310, 305, 300, 290, 280, 270, 250, 190, 80, 60, 50):
        if prec >= p:
            return p
    raise AssertionError(prec)


def _fill(t, tag, op, name):
    return t.format(OP=op, A=tag + "a", B=tag + "b", C=tag + "c", H=name)


def _is_private(ch):
    return 0xE000 <= ord(ch) <= 0xF8FF


def text_is(want):
    return Pred(lambda g: g == want, repr(want))


def tex_cases(tag, tex):
    """TeXForm of name[x, y] is x<macro> y (standard LaTeX symbols; x and y are never assigned)."""
    return [(f"ToString[TeXForm[{n}[x, y]]]", text_is(f"x{m} y")) for n, m in tex.items()]


def build(tag, ref):
    """Cases for every infix / prefix operator in ``ref``: name -> (char, precedence, affix, associativity)."""
    a, b, c = tag + "a", tag + "b", tag + "c"
    cases = []
    for name, (ch, prec, affix, assoc) in ref.items():
        lname = "\\[" + name + "]"
        shown = lname if _is_private(ch) else ch
        cases.append((f"{name}[{a}, 1 + 1]", f"{name}[{a}, 2]"))
        cases.append((f"Attributes[{name}]", "{}"))
        if name in ("MinusPlus", "PlusMinus"):         # one argument prints as the prefix form
            cases.append((f"ToString[{name}[]]", text_is(f"{name}[]")))
        elif affix == "Prefix":
            cases.append((f"ToString[{name}[{a}, {b}]]", text_is(f"{name}[{a}, {b}]")))
        else:
            cases.append((f"ToString[{name}[{a}]]", text_is(f"{name}[{a}]")))
        if name in ("GreaterSlantEqual", "LessSlantEqual"):
            # Wolfram reads \[GreaterSlantEqual] / ⩾ as GreaterEqual (Operator Input Forms tutorial: "Exceptions
            # are ⩾, ⩽ ..."); the head itself has no operator form and prints functionally.
            target = {"GreaterSlantEqual": "GreaterEqual", "LessSlantEqual": "LessEqual"}[name]
            cases.append((f"Hold[{a} {lname} {b}] === Hold[{target}[{a}, {b}]]", "True"))
            cases.append((f"Hold[{a} {ch} {b}] === Hold[{target}[{a}, {b}]]", "True"))
            cases.append((f"ToString[{name}[{a}, {b}]]", text_is(f"{name}[{a}, {b}]")))
            cases.append((f"ToExpression[ToString[{name}[{a}, {b}]]] === {name}[{a}, {b}]", "True"))
            continue
        if name == "NotExists":
            continue                      # prefix operator not parsed here (tested separately)
        if affix == "Prefix":
            cases.append((f"Hold[{lname}{a}]", f"Hold[{name}[{a}]]"))
            cases.append((f"Hold[{ch}{a}] === Hold[{name}[{a}]]", "True"))
            cases.append((f"Hold[{lname}{a}^2]", f"Hold[{name}[{a}^2]]"))
            cases.append((f"Hold[{lname}{a} ** {b}]", f"Hold[NonCommutativeMultiply[{name}[{a}], {b}]]"))
            cases.append((f"Hold[{lname}{lname}{a}]", f"Hold[{name}[{name}[{a}]]]"))
            cases.append((f"Hold[{b} {lname}{a}]", f"Hold[Times[{b}, {name}[{a}]]]"))
            cases.append((f"ToString[{name}[{a}]]", Pred(lambda g, n=name: g.endswith(a) and len(g) == len(a) + 1,
                                                          f"<d>{a}")))
            cases.append((f"ToExpression[ToString[{name}[{a} + 1]]] === {name}[{a} + 1]", "True"))
            cases.append((f"ToString[{name}[{a}], CharacterEncoding -> \"ASCII\"]", text_is(f"{lname}{a}")))
            continue
        cases.append((f"Hold[{a} {lname} {b}]", f"Hold[{name}[{a}, {b}]]"))
        cases.append((f"Hold[{a} {ch} {b}] === Hold[{name}[{a}, {b}]]", "True"))
        if assoc == "None":
            cases.append((f"Hold[{a} {lname} {b} {lname} {c}]", f"Hold[{name}[{a}, {b}, {c}]]"))
            cases.append((f"ToString[{name}[{a}, {b}, {c}]]", text_is(f"{a} {shown} {b} {shown} {c}")))
            cases.append((f"ToExpression[ToString[{name}[{a}, {b} + 1, {c}]]] === {name}[{a}, {b} + 1, {c}]",
                          "True"))
        elif assoc == "Left":
            cases.append((f"Hold[{a} {lname} {b} {lname} {c}]", f"Hold[{name}[{name}[{a}, {b}], {c}]]"))
            cases.append((f"ToString[{name}[{name}[{a}, {b}], {c}]]", text_is(f"{a} {shown} {b} {shown} {c}")))
            cases.append((f"ToString[{name}[{a}, {name}[{b}, {c}]]]", text_is(f"{a} {shown} ({b} {shown} {c})")))
            cases.append((f"ToExpression[ToString[{name}[{a}, {name}[{b}, {c}]]]] === {name}[{a}, {name}[{b}, {c}]]",
                          "True"))
        else:
            cases.append((f"Hold[{a} {lname} {b} {lname} {c}]", f"Hold[{name}[{a}, {name}[{b}, {c}]]]"))
            cases.append((f"ToString[{name}[{a}, {name}[{b}, {c}]]]", text_is(f"{a} {shown} {b} {shown} {c}")))
            cases.append((f"ToString[{name}[{name}[{a}, {b}], {c}]]", text_is(f"({a} {shown} {b}) {shown} {c}")))
            cases.append((f"ToExpression[ToString[{name}[{name}[{a}, {b}], {c}]]] === {name}[{name}[{a}, {b}], {c}]",
                          "True"))
        cases.append((f"ToString[{name}[{a}, {b}]]", text_is(f"{a} {shown} {b}")))
        cases.append((f"ToString[{name}[{a}, {b}], CharacterEncoding -> \"ASCII\"]", text_is(f"{a} {lname} {b}")))
        band = _band(prec)
        below, above = _BELOW[band], _ABOVE[band]
        if band == 80:
            cases.append((f"Hold[{a} {lname} {b} // {c}]", f"Hold[{c}[{name}[{a}, {b}]]]"))
        else:
            cases.append((f"Hold[{_fill(below[0], tag, lname, name)}]", f"Hold[{_fill(below[1], tag, lname, name)}]"))
        cases.append((f"Hold[{_fill(above[0], tag, lname, name)}]", f"Hold[{_fill(above[1], tag, lname, name)}]"))
        cases.append((f"CharacterName[\"{ch}\"]", f"\"{name}\""))
    return cases
