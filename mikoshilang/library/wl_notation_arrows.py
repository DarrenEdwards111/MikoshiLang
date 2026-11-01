"""Wolfram batch W32: Notation II -- arrows, tees, vectors and equilibria (inert).

Every head here has no built-in meaning in Wolfram ("RightTee[x, y] ... has no built-in meaning"): the expression
stays as entered after its arguments have been evaluated, it has no attributes (users may attach definitions), it
never raises, and it has an infix input form with Wolfram's precedence and grouping, printed back in that form:

* horizontal and diagonal arrows, harpoons ("vectors") and LeftRightVector-style operators: n-ary, precedence 270
  (x \\[LeftArrow] y \\[LeftArrow] z is LeftArrow[x, y, z]);
* vertical arrows and vectors, the Long* and DoubleLong* arrows, UpEquilibrium / ReverseUpEquilibrium and the
  Short* arrows: n-ary, precedence 580 (between NonCommutativeMultiply and Power);
* Equilibrium / ReverseEquilibrium: n-ary, precedence 290 (with the comparison-like operators, but never chained
  with ==, <, ... without parentheses);
* tees: RightTee and DoubleRightTee group to the right, LeftTee, DoubleLeftTee, UpTee and DownTee to the left, all at
  precedence 190 (between Repeated and Or).

Parsing comes from ``mikoshilang.parser`` and the table in ``mikoshilang.notation_ops``; registration and printing
(InputForm / OutputForm / ToString, TeXForm where LaTeX has the symbol, CharacterEncoding -> "ASCII",
CharacterName) from ``wl_notation_rel.register_notation``.
"""
from __future__ import annotations

from ..expr import Expr
from .core import Unevaluated
from .wl_notation_rel import register_notation

AREA = "notation: arrows, tees, vectors & equilibria (W32)"


def _make_inert():
    def _inert(*args: Expr) -> Expr:
        raise Unevaluated()
    return _inert


register_notation("W32", AREA, make=_make_inert)
