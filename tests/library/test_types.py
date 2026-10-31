"""Types pillar T1: TypeSignature, TypeMatchQ, TypeCheck (MikoshiLang extensions, see docs/types.md)."""
import pytest
from _harness import Pred, Raises, check, ids


def _mismatches(g):
    """Rows of TypeCheck[...]["Mismatches"] as (function, position, expected) tuples."""
    rows = []
    for a in g.args:
        if a.args[0] == "Mismatches":
            for row in a.args[1].args:
                d = {r.args[0]: r.args[1] for r in row.args}
                rows.append((str(d["Function"]), d["Position"], d["Expected"]))
    return rows


def _result(g):
    return next(a.args[1] for a in g.args if a.args[0] == "Result")


CASES = [
    # TypeSignature: library signatures come from the implementations' annotations (CONVENTIONS §2)
    ("TypeSignature[BitLength]", "{{Integer} -> Expr}"),                         # wl_discrete: _bint(n) -> Integer
    ("TypeSignature[PolygonalNumber]", Pred(lambda g: len(g.args) >= 1 and all(r.head == "Rule" for r in g.args),
                                            "one rule per implementation")),
    ("TypeSignature[StringRiffle]", Pred(lambda g: g.head == "List", "a list of signature rules")),
    ("TypeSignature[NoSuchFunctionT1]", "{}"),                                   # nothing known
    ("TypeSignature[tsF1, {Integer, Integer} -> Integer]", "{Integer, Integer} -> Integer"),
    ("TypeSignature[tsF2, {String} -> Integer]; TypeSignature[tsF2, {ListOf[String]} -> {Integer}]; "
     "TypeSignature[tsF2]", "{{String} -> Integer, {ListOf[String]} -> ListOf[Integer]}"),   # overloads; {t} = ListOf[t]
    ("TypeSignature[tsF3, {Real, RepeatedNull[Option]} -> Real | Integer]",
     "{Real, RepeatedNull[Option]} -> Real | Integer"),
    ("TypeSignature[tsF4, {Banana} -> Integer]", Raises(ValueError)),             # unknown type
    ("TypeSignature[tsF4, Integer]", Raises(ValueError)),                         # not {..} -> r
    ("TypeSignature[3, {Integer} -> Integer]", Raises(TypeError)),                # not a symbol
    ("TypeSignature[tsF4, {RepeatedNull[Integer], Real} -> Integer]", Raises(ValueError)),

    # TypeMatchQ: strict membership
    ("TypeMatchQ[3, Integer]", "True"),
    ("TypeMatchQ[True, Integer]", "False"),
    ("TypeMatchQ[3, PosInteger]", "True"),
    ("TypeMatchQ[0, PosInteger]", "False"),
    ("TypeMatchQ[0, NonNegInteger]", "True"),
    ("TypeMatchQ[1/3, Rational]", "True"),
    ("TypeMatchQ[2.5, Rational]", "False"),
    ("TypeMatchQ[Pi, Real]", "True"),
    ("TypeMatchQ[1 + I, Real]", "False"),
    ("TypeMatchQ[1 + I, Number]", "True"),
    ("TypeMatchQ[Sqrt[2] + Pi, Numeric]", "True"),
    ("TypeMatchQ[\"a\", String]", "True"),
    ("TypeMatchQ[x, Integer]", "False"),                                         # symbolic: not known
    ("TypeMatchQ[x, Var]", "True"),
    ("TypeMatchQ[Pi, Var]", "False"),
    ("TypeMatchQ[{1, 2}, ListOf[Integer]]", "True"),
    ("TypeMatchQ[{1, 2.5}, {Integer}]", "False"),
    ("TypeMatchQ[{{1, 2}, {3, 4}}, MatrixOf[Integer]]", "True"),
    ("TypeMatchQ[{{1, 2}, {3}}, MatrixOf[Integer]]", "False"),                    # ragged
    ("TypeMatchQ[<|a -> 1|>, Assoc]", "True"),
    ("TypeMatchQ[a -> 1, Rule]", "True"),
    ("TypeMatchQ[Method -> \"x\", Option]", "True"),
    ("TypeMatchQ[False, Boolean]", "True"),
    ("TypeMatchQ[\"a\", Integer | String]", "True"),
    ("TypeMatchQ[f[x], Expr]", "True"),
    ("TypeMatchQ[3, Fn]", "False"),
    ("TypeMatchQ[3, Banana]", Raises(ValueError)),

    # TypeCheck: report mode, seeded mismatches (Check: tests)
    ("TypeCheck[BitLength[\"x\"]]",
     Pred(lambda g: _mismatches(g) == [("BitLength", 1, "Integer")], "one argument mismatch")),
    ("TypeCheck[BitLength[1024]]", Pred(lambda g: _mismatches(g) == [] and _result(g) == 11, "clean call")),
    ("TypeCheck[BitLength[n]]", Pred(lambda g: _mismatches(g) == [], "symbolic is not a mismatch")),
    ("TypeSignature[tcF1, {Integer, Integer} -> Integer]; tcF1[a_, b_] := a + b; TypeCheck[tcF1[1, \"a\"]]",
     Pred(lambda g: _mismatches(g) == [("tcF1", 2, "Integer")], "declared argument type")),
    ("TypeSignature[tcF2, {Integer} -> String]; tcF2[a_] := a; TypeCheck[tcF2[3]]",
     Pred(lambda g: _mismatches(g) == [("tcF2", "Result", "String")], "declared result type")),
    ("TypeSignature[tcF3, {Integer} -> Integer]; tcF3[a_] := a; TypeCheck[tcF3[1, 2]]",
     Pred(lambda g: [m[1] for m in _mismatches(g)] == ["Count"], "wrong argument count")),
    ("TypeSignature[tcF4, {Integer} -> Integer]; TypeSignature[tcF4, {String} -> String]; tcF4[a_] := a; "
     "TypeCheck[{tcF4[1], tcF4[\"s\"], tcF4[2.5]}]",
     Pred(lambda g: [m[:2] for m in _mismatches(g)] == [("tcF4", 1)], "overloads: only 2.5 fits none")),
    ("TypeCheck[Table[BitLength[k], {k, {1, 2, \"z\"}}]]",
     Pred(lambda g: len(_mismatches(g)) == 1, "nested call inside Table")),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_types(case):
    check(case)
