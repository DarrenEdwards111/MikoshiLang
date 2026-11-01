"""Wolfram batch W31: Notation I -- relational, set-like and bracket operators (inert).

Almost every head here has no built-in meaning in Wolfram ("CirclePlus[x, y, ...] ... has no built-in meaning"):
the expression stays as entered after its arguments have been evaluated, it has no attributes (so users may give
it definitions, as the Wolfram documentation does with ``CirclePlus[a_, b_] := ...``), it never raises, and its
value lies in the notation: each head has an infix (or prefix / matchfix) input form with Wolfram's precedence and
grouping, and is printed back in that form.

* Parsing: ``\\[Name]`` long names and Wolfram's characters (for example ``a \\[CirclePlus] b`` or ``a ⊕ b`` is
  ``CirclePlus[a, b]``) are read by ``mikoshilang.parser`` from the table in ``mikoshilang.notation_ops``.
  Different operators of one precedence level are not mixed without parentheses (ParseError), so no input is
  read differently from Wolfram.  ``\\[GreaterSlantEqual]`` / ``⩾`` and ``\\[LessSlantEqual]`` / ``⩽`` are read
  as GreaterEqual and LessEqual, as in Wolfram.
* Printing: InputForm / OutputForm / ToString use the operator form (``x ⊕ y``; operators whose Wolfram character
  is in the private use area print as ``\\[Name]``), with parentheses from Wolfram's precedences; TeXForm uses
  the standard LaTeX symbol where there is one; ``ToString[..., CharacterEncoding -> "ASCII"]`` and
  ``CharacterName`` know the long names.  TraditionalForm prints these heads in functional form (deviation).
* Heads with semantics: ``NotElement`` (decided through ``Element``), ``Commutator`` and ``Anticommutator``
  (x ** y - y ** x and x ** y + y ** x, or the same in a given algebra).  ``Bra``, ``Ket`` and ``BraKet`` (inert
  in Wolfram too) are promoted from the quarantined list; they print in functional form.

Extension point: ``register_notation(batch, area)`` registers the inert heads of one batch from
``notation_ops.OPS`` (used by ``wl_notation_arrows`` for W32).
"""
from __future__ import annotations

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import formatting as F
from .. import notation_ops as N
from .core import fn, Unevaluated
from .wl_common import ev, apply
from . import wl_forms
from . import wl_text
from .wl_evaluation import _ASSIGNABLE

AREA = "notation: relational, set-like & bracket operators (W31)"


# ───────────────────────────── printing ─────────────────────────────

def _paren(text, prec, parent):
    return f"({text})" if prec < parent else text


def _child_parent(child, prec, kind):
    """Parent precedence for an operand on the grouping side of a left/right operator: an operand built with an
    operator of the same level and kind prints without parentheses (it re-parses the same way)."""
    if isinstance(child, Expr) and isinstance(child.head, str) and child.head in N.OPS:
        _c, p, k, _t, _b = N.OPS[child.head]
        if p == prec and k == kind:
            return prec
    return prec + 1


def _format_hook(name):
    _ch, prec, kind, _tex, _b = N.OPS[name]
    op = N.print_char(name)
    # The CenterDot..Backslash band sits between Times (400) and Divide (470) in Wolfram; the parser does not mix
    # it with "/", so these operators are parenthesised inside any product (Times prints its factors at 400).
    wrap_at = F.P_TIMES - 1 if 410 <= prec <= 460 else prec

    def hook(e, parent):
        a = e.args
        if kind == "prefix":
            if len(a) != 1:
                return None
            return _paren(op + F.input_form(a[0], prec), prec, parent)
        if name == "MinusPlus" and len(a) == 1:
            return _paren(op + F.input_form(a[0], F.P_UMINUS + 1), F.P_UMINUS, parent)
        if kind == "nary":
            if len(a) < 2:
                return None
            text = f" {op} ".join(F.input_form(x, prec + 1) for x in a)
        elif kind in ("left", "right"):
            if len(a) != 2:
                return None
            lp = _child_parent(a[0], prec, kind) if kind == "left" else prec + 1
            rp = _child_parent(a[1], prec, kind) if kind == "right" else prec + 1
            text = f"{F.input_form(a[0], lp)} {op} {F.input_form(a[1], rp)}"
        else:                                   # alias: functional form
            return None
        return _paren(text, wrap_at, parent)
    return hook


def _tex_hook(name):
    _ch, prec, kind, macro, _b = N.OPS[name]
    if prec <= 290:
        level, arg = wl_forms.P_REL, wl_forms.P_SUM
    elif prec < 400:
        level, arg = wl_forms.P_SUM, wl_forms.P_PROD
    else:
        level, arg = wl_forms.P_PROD, wl_forms.P_POW

    def hook(texer, e):
        a = e.args
        if (kind == "nary" and len(a) >= 2) or (kind in ("left", "right") and len(a) == 2):
            return f"{macro} ".join(texer.wrap(texer.tex(x), arg) for x in a), level
        return texer.call(texer.symbol(name), a)
    return hook


def _install_printing(name):
    ch, _p, kind, macro, _b = N.OPS[name]
    if kind != "alias":
        F.FORMAT_HOOKS[name] = _format_hook(name)
    if macro and kind not in ("alias", "prefix") and name not in wl_forms._TEX_REL:
        wl_forms.RENDER_TEX[name] = _tex_hook(name)
    if not N.is_private(ch):
        wl_forms._NAMED_CHARS.setdefault(ch, name)
    if name in N.PRINT_CHAR:
        wl_forms._NAMED_CHARS.setdefault(N.PRINT_CHAR[name], name)
    wl_text._NAMED.setdefault(ch, name)


# ───────────────────────────── inert heads ─────────────────────────────

def _forms(name):
    ch, prec, kind, macro, _b = N.OPS[name]
    lname = "\\[" + name + "]"
    shown = N.print_char(name)
    chars = lname if N.is_private(ch) else f"{lname} or {ch}"
    if name in N.EXTRA_INPUT.values():
        chars += " or " + "".join(c for c, n in N.EXTRA_INPUT.items() if n == name)
    if kind == "nary":
        form = (f"Input form x {shown} y {shown} ... (typed as {chars}; n-ary, precedence {prec}); printed in that "
                "form when there are at least two arguments, otherwise in functional form")
    elif kind in ("left", "right"):
        form = (f"Input form x {shown} y (typed as {chars}; binary, precedence {prec}, grouping to the {kind}); "
                "printed in that form for exactly two arguments, otherwise in functional form")
        if name == "MinusPlus":
            form += f"; also prefix {shown}x, like unary minus"
    elif kind == "prefix":
        form = (f"Input form {shown}x (typed as {chars}; prefix, precedence {prec}); printed {shown}x for one "
                "argument, otherwise in functional form")
    else:
        target = {"GreaterSlantEqual": "GreaterEqual", "LessSlantEqual": "LessEqual"}[name]
        form = (f"No input operator: as in Wolfram, {chars} is read as {target}; printed in functional form")
    if macro and kind not in ("alias", "prefix"):
        form += f"; TeXForm uses {macro}"
    return form


def _inert_notes(name, extra=""):
    return (f"{name}[x, ...] has no built-in meaning (as in Wolfram): it stays as entered with its arguments "
            "evaluated, accepts any number of arguments, has no attributes (users may attach definitions) and never "
            f"raises. {_forms(name)}.{extra} Check: exact (inert: the call is returned unchanged); reference (Wolfram "
            "documentation and operator table) for parsing and printing in tests.")


def _make_inert():
    def _inert(*args: Expr) -> Expr:
        raise Unevaluated()
    return _inert


def _register_inert(name, area, status="added", extra="", make=_make_inert):
    impl = fn(name, area=area, status=status, notes=_inert_notes(name, extra) if name in N.OPS else extra)(make())
    K.BUILTIN_ATTRIBUTES[name] = set()
    _ASSIGNABLE.add(name)
    if name in N.OPS:
        _install_printing(name)
    return impl


_SEMANTIC = ("NotElement",)


def register_notation(batch, area, make=_make_inert):
    """Register every inert notation head of ``batch`` ("W31" / "W32") from ``notation_ops.OPS``; ``make()``
    returns a fresh implementation function (``*args: Expr -> Expr`` raising Unevaluated)."""
    for name, entry in N.OPS.items():
        if entry[4] == batch and name not in _SEMANTIC:
            _register_inert(name, area, make=make)


register_notation("W31", AREA)


# AngleBracket: matchfix 〈x, y, ...〉
def _fmt_angle(e, parent):
    return N.LEFT_ANGLE + ", ".join(F.input_form(x) for x in e.args) + N.RIGHT_ANGLE


_register_inert("AngleBracket", AREA, extra=(
    "AngleBracket[x, y, ...] has no built-in meaning (as in Wolfram): it stays as entered with its arguments "
    "evaluated, accepts any number of arguments (including none), has no attributes and never raises. Input form "
    "\\[LeftAngleBracket] x, y, ... \\[RightAngleBracket] (or the characters 〈 〉, U+2329/U+232A), printed back in "
    "that form. Check: exact (inert: the call is returned unchanged); reference (Wolfram documentation) for parsing "
    "and printing in tests."))
F.FORMAT_HOOKS["AngleBracket"] = _fmt_angle
wl_text._NAMED.setdefault(N.LEFT_ANGLE, "LeftAngleBracket")
wl_text._NAMED.setdefault(N.RIGHT_ANGLE, "RightAngleBracket")
wl_forms._NAMED_CHARS.setdefault(N.LEFT_ANGLE, "LeftAngleBracket")
wl_forms._NAMED_CHARS.setdefault(N.RIGHT_ANGLE, "RightAngleBracket")


def _plain_inert(name, what, status="added"):
    _register_inert(name, AREA, status=status, extra=(
        f"{what} It has no built-in meaning (as in Wolfram): it stays as entered with its arguments evaluated, "
        "accepts any number of arguments, has no attributes (users may attach definitions, as in the Wolfram "
        "documentation examples) and never raises. Printed in functional form (Wolfram's 2-D form is not "
        "reproduced). Check: exact (inert: the call is returned unchanged); reference (Wolfram documentation "
        "examples) in tests."))


_plain_inert("NotExists", "NotExists[x, y] represents the statement that there exists no x satisfying y (Wolfram "
                          "prefix operator \\[NotExists]; not parsed here).")
_plain_inert("Ket", "Ket[x] is the quantum ket |x> (Dirac notation).", status="promoted")
_plain_inert("Bra", "Bra[x] is the quantum bra <x| (Dirac notation).", status="promoted")
_plain_inert("BraKet", "BraKet[a, b] is the quantum bracket <a|b> (Dirac notation).", status="promoted")


# ───────────────────────────── NotElement ─────────────────────────────

def _element(x, dom):
    try:
        return ev(Expr("Element", x, dom))
    except (TypeError, ValueError, NotImplementedError) as exc:
        msg = str(exc)
        raise type(exc)("NotElement: " + (msg[len("Element: "):] if msg.startswith("Element: ") else msg)) from exc


@fn("NotElement", area=AREA, status="added",
    notes="NotElement[x, dom] asserts that x is not an element of the domain dom (Reals, Integers, Rationals, "
          "Complexes, Algebraics, Primes, Booleans, ... as for Element); NotElement[x1 | x2 | ..., dom] asserts that "
          "none of the xi is.  Decided through Element: True when every xi is known not to be in dom, False when "
          "one is known to be in dom, otherwise the undecided items stay (NotElement[x, Reals] remains "
          "unevaluated). Input form x \\[NotElement] dom (or ∉; precedence 250), printed in that form; TeXForm "
          "\\notin. Attributes {Protected}. Errors raised by Element are re-raised with the NotElement: prefix. "
          "Check: delegated (Element); reference (Wolfram documentation examples) in tests.")
def _not_element(x: Expr, dom: Expr) -> Expr:
    items = list(x.args) if isinstance(x, Expr) and x.head == "Alternatives" else [x]
    left = []
    for item in items:
        r = _element(item, dom)
        if r is True:
            return False
        if r is False:
            continue
        left.append((item, r))
    if not left:
        return True
    if len(left) == 1 and len(items) == 1:
        item, r = left[0]
        if isinstance(r, Expr) and r.head == "Element" and len(r.args) == 2:
            if K.same(r.args[0], item) and K.same(r.args[1], dom):
                raise Unevaluated()
            return Expr("NotElement", *r.args)
        return ev(Expr("Not", r))
    if len(left) == len(items):
        raise Unevaluated()
    rest = left[0][0] if len(left) == 1 else Expr("Alternatives", *(i for i, _r in left))
    return Expr("NotElement", rest, dom)


F.FORMAT_HOOKS["NotElement"] = _format_hook("NotElement")
wl_text._NAMED.setdefault("\u2209", "NotElement")
wl_forms._NAMED_CHARS.setdefault("\u2209", "NotElement")


# ───────────────────────────── Commutator, Anticommutator ─────────────────────────────

def _algebra_ops(alg, fname):
    """(mult, add) of an algebra argument: NonCommutativeAlgebra[...] / association, a multiplication head, or
    {mult, n} (n x n matrices with that product)."""
    from .wl_polyalg import _nc_alg
    if alg is None:
        return Symbol("NonCommutativeMultiply"), Symbol("Plus"), None
    if isinstance(alg, Expr) and (alg.head in ("NonCommutativeAlgebra", "Association")):
        a = _nc_alg(alg, fname)
        return a.mult, a.add, None
    if K.is_list(alg):
        if len(alg.args) != 2:
            raise ValueError(f"{fname}: a matrix algebra is given as {{mult, n}}, not {F.input_form(alg)}")
        mult, n = alg.args
        if isinstance(n, (bool, str, float, complex)) or (isinstance(n, int) and n < 1) or \
                (isinstance(n, Expr) and n.head in ("List", "Rational", "Complex")):
            raise ValueError(f"{fname}: the matrix size in {{mult, n}} must be a positive integer, not "
                             f"{F.input_form(n)}")
        return mult, Symbol("Plus"), n
    if isinstance(alg, (str, bool, int, float, complex)) or (isinstance(alg, Expr) and alg.head in ("Rule",
                                                                                                    "RuleDelayed")):
        raise TypeError(f"{fname}: {F.input_form(alg)} is not an algebra specification (expected "
                        "NonCommutativeAlgebra[...], a multiplication function or {mult, n})")
    return alg, Symbol("Plus"), None


def _check_square(x, n, fname):
    if not (isinstance(n, int) and K.is_list(x)):
        return
    rows = x.args
    if len(rows) != n or not all(K.is_list(r) and len(r.args) == n for r in rows):
        raise ValueError(f"{fname}: {F.input_form(x)} is not a {n} x {n} matrix")


def _bracket(x, y, alg, fname, sign):
    mult, add, n = _algebra_ops(alg, fname)
    if n is not None:
        _check_square(x, n, fname)
        _check_square(y, n, fname)
    xy = apply(mult, x, y)
    yx = apply(mult, y, x)
    second = Expr("Times", -1, yx) if sign < 0 else yx
    return ev(apply(add, xy, second))


@fn("Commutator", area=AREA, status="added",
    notes="Commutator[x, y] gives x ** y - y ** x; Commutator[x, y, alg] gives (x ⊗ y) ⊕ (-(y ⊗ x)) for the "
          "multiplication ⊗ and addition ⊕ of alg: NonCommutativeAlgebra[<|\"Multiplication\" -> m, \"Addition\" -> "
          "a, ...|>] (or the association), a multiplication function such as Dot or Composition, or {mult, n} for "
          "n x n matrices (explicit matrices of another size raise ValueError). Other algebra arguments raise "
          "TypeError; unsupported NonCommutativeAlgebra properties NotImplementedError. Attributes {Protected}. "
          "Printed as the evaluated difference. Check: exact (the defining formula); reference (Wolfram "
          "documentation examples) in tests.")
def _commutator(x: Expr, y: Expr, alg: Expr = None) -> Expr:
    return _bracket(x, y, alg, "Commutator", -1)


@fn("Anticommutator", area=AREA, status="added",
    notes="Anticommutator[x, y] gives x ** y + y ** x; Anticommutator[x, y, alg] gives (x ⊗ y) ⊕ (y ⊗ x) for the "
          "multiplication ⊗ and addition ⊕ of alg: NonCommutativeAlgebra[<|...|>] (or the association), a "
          "multiplication function such as Dot or Composition, or {mult, n} for n x n matrices (explicit matrices "
          "of another size raise ValueError). Other algebra arguments raise TypeError; unsupported "
          "NonCommutativeAlgebra properties NotImplementedError. Attributes {Protected}. Check: exact (the "
          "defining formula); reference (Wolfram documentation examples) in tests.")
def _anticommutator(x: Expr, y: Expr, alg: Expr = None) -> Expr:
    return _bracket(x, y, alg, "Anticommutator", 1)
