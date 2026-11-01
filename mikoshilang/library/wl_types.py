"""Types pillar v1 (MikoshiLang extension, not Wolfram): signatures, declarations and the opt-in checker.

* ``TypeSignature[f]`` lists the signatures of a library head (read from its implementations' annotations,
  CONVENTIONS §2) or of a user function declared in this session; ``TypeSignature[f, {t1, ...} -> r]``
  declares one (several declarations are overloads).
* ``TypeMatchQ[value, type]`` is the strict membership test.
* ``TypeCheck[expr]`` evaluates ``expr`` with checking in report mode and returns the result with the
  mismatches found.

Types are the CONVENTIONS §2 names (``Integer``, ``PosInteger``, ``NonNegInteger``, ``Rational``, ``Real``,
``Number``, ``Numeric``, ``String``, ``Boolean``, ``Var``, ``Assoc``, ``Rule``, ``Option``, ``Fn``, ``Held``,
``Expr``), ``ListOf[t]`` (or ``{t}``), ``MatrixOf[t]`` and ``t1 | t2``.
"""
from __future__ import annotations

from ..expr import Expr, Symbol
from .core import fn, REGISTRY, Unevaluated
from .. import types as T

AREA = "types (T1)"
ADDED = dict(area=AREA, status="added")


def _name(f, fname: str) -> str:
    if isinstance(f, Symbol):
        return f.name
    if isinstance(f, str) and f:
        return f
    if isinstance(f, (bool, int, float)) or (isinstance(f, Expr) and f.head in ("List", "Rule", "Association")):
        raise TypeError(f"{fname}: argument 1 must be a symbol or a function name, not {f!r}")
    raise Unevaluated()


def _type(t, fname: str) -> tuple:
    try:
        return T.from_expr(t)
    except T.TypeSyntaxError as exc:
        raise ValueError(str(exc).replace("TypeSignature:", f"{fname}:", 1)) from None


@fn("TypeSignature", **ADDED,
    notes="TypeSignature[f] gives the signatures of f as {{t1, t2, ...} -> result, ...}: user declarations in this "
          "session first, otherwise the library signature read from the Python annotations of each "
          "implementation (RepeatedNull[t] stands for any number of further arguments); {} if none is known. "
          "TypeSignature[f, {t1, ...} -> r] declares a signature for a user function (repeat it for overloads) and "
          "returns it; RepeatedNull[t] is allowed last. The declaration is checked only when checking is on "
          "(Session(check_types=...) or TypeCheck). A non-symbol f raises TypeError; an unknown type raises "
          "ValueError. MikoshiLang extension. Check: exact; the signatures of library heads are compared with "
          "their annotations in tests.")
def _type_signature(f: Expr, sig: Expr = None) -> ListOf[Rule] | Rule:
    name = _name(f, "TypeSignature")
    if sig is None:
        return Expr("List", *(s.to_expr() for s in T.signatures_of(name)))
    try:
        return T.declare(name, sig).to_expr()
    except T.TypeSyntaxError as exc:
        raise ValueError(str(exc)) from None


@fn("TypeMatchQ", **ADDED,
    notes="TypeMatchQ[value, type] is True when value is known to have the type (Integer, Real, ListOf[t], "
          "t1 | t2, ...) and False otherwise, including for symbolic values whose type is not known yet (x is "
          "not an Integer). An unknown type raises ValueError. MikoshiLang extension. Check: exact; reference "
          "values for every type in tests.")
def _type_match_q(value: Expr, t: Expr) -> Boolean:
    return T.match(value, _type(t, "TypeMatchQ")) is True


def _assoc(pairs):
    return Expr("Association", *(Expr("Rule", k, v) for k, v in pairs))


@fn("TypeCheck", **ADDED, hold=True,
    notes="TypeCheck[expr] evaluates expr with type checking in report mode and gives "
          "<|\"Result\" -> value, \"Mismatches\" -> {<|\"Function\" -> f, \"Position\" -> i, \"Value\" -> v, "
          "\"Expected\" -> type|>, ...}|>. A mismatch is a concrete argument (or result) of the wrong kind; "
          "symbolic arguments never are. Position is the argument number, \"Result\" or \"Count\" (wrong number "
          "of arguments for a declared function). MikoshiLang extension. Check: exact; seeded mismatches in "
          "tests.")
def _type_check(expr: Held) -> Assoc:
    from .wl_common import ev
    with T.checking("report") as report:
        value = ev(expr)
    rows = []
    for m in report:
        pos = m["position"]
        pos = pos if isinstance(pos, int) else pos.capitalize()
        rows.append(_assoc([("Function", Symbol(m["function"])), ("Position", pos), ("Value", m["value"]),
                            ("Expected", m["expected"])]))
    return _assoc([("Result", value), ("Mismatches", Expr("List", *rows))])


def _hold_type_check():
    from ..evaluate import set_hold_all
    set_hold_all("TypeCheck")


_hold_type_check()
