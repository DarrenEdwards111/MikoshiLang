"""Types pillar T1 at the Python boundary: Session(check_types=...), Entry.signature, the off-path gate."""
import pytest

from mikoshilang import Session, _typestate, types as T
from mikoshilang.library.core import REGISTRY


def test_entry_signature_from_annotations():
    sig = REGISTRY["BitLength"].signature
    assert len(sig) == 1 and sig[0].params == [("Integer",)]
    assert REGISTRY["TypeMatchQ"].signature[0].result == ("Boolean",)
    assert REGISTRY["TypeCheck"].signature[0].params == [("Held",)]          # hold=True heads


@pytest.mark.parametrize("ann, expected", [
    ("Integer", ("Integer",)),
    ("ListOf[Integer]", ("ListOf", ("Integer",))),
    ("MatrixOf[Real]", ("MatrixOf", ("Real",))),
    ("Integer | ListOf[Integer]", ("Or", (("Integer",), ("ListOf", ("Integer",))))),
    ("Integer | Expr", ("Expr",)),
    ("NotAType", ("Expr",)),                      # outside the vocabulary: unconstrained
    (int, ("Integer",)),                           # evaluated Python annotations map too
    (str, ("String",)),
])
def test_parse_annotation(ann, expected):
    assert T.parse_annotation(ann) == expected


def test_check_types_raise():
    s = Session(check_types=True)
    assert s.evaluate("BitLength[1024]") == 11
    assert str(s.evaluate("BitLength[n]")) == "BitLength[n]"            # symbolic: never a mismatch
    with pytest.raises(TypeError, match=r"^BitLength: argument 1 'a' is not of type Integer"):
        s.evaluate('BitLength["a"]')
    assert _typestate.ACTIVE == 0                                       # gate restored after the error


def test_check_types_report():
    s = Session(check_types="report")
    s.evaluate("TypeSignature[g, {Integer} -> Integer]; g[x_] := x")
    assert str(s.evaluate('g["s"]')) == "s"                             # evaluation carries on
    assert [(m["function"], m["position"], m["expected"]) for m in s.type_mismatches] == [
        ("g", 1, "Integer"), ("g", "result", "Integer")]


def test_check_types_off_by_default():
    s = Session()
    assert str(s.evaluate('BitLength["a"]')) == 'BitLength["a"]'       # the implementation leaves it alone
    s.evaluate("TypeSignature[h, {Integer} -> Integer]; h[x_] := x")
    assert s.evaluate('h["s"]') == "s" and s.type_mismatches == []


def test_check_types_invalid_value():
    with pytest.raises(ValueError, match="check_types"):
        Session(check_types="yes")


def test_declarations_are_per_session():
    a, b = Session(check_types=True), Session(check_types=True)
    a.evaluate("TypeSignature[k, {Integer} -> Integer]; k[x_] := x")
    b.evaluate("k[x_] := x")
    with pytest.raises(TypeError, match="^k: argument 1"):
        a.evaluate('k["s"]')
    assert b.evaluate('k["s"]') == "s"


def test_nested_checking_restores_gate():
    s = Session(check_types="report")
    out = s.evaluate('TypeCheck[BitLength["q"]]')                       # TypeCheck inside a checking session
    assert "Mismatches" in str(out) and _typestate.ACTIVE == 0
    # TypeCheck's result is final: it is not evaluated (and checked) again under the session's mode
    assert s.type_mismatches == []
    r = Session(check_types=True)
    assert str(r.evaluate("TypeCheck[BitLength[8]]")).startswith('<|"Result" -> 4')
    assert _typestate.ACTIVE == 0


def test_typecheck_reports_inside_raising_session():
    r = Session(check_types=True)
    out = r.evaluate('TypeCheck[BitLength["q"]]')
    assert 'BitLength["q"]' in str(out) and '"Expected" -> "Integer"' in str(out)
    with pytest.raises(TypeError, match="^BitLength: argument 1"):
        r.evaluate('BitLength["q"]')
    assert _typestate.ACTIVE == 0


def test_sign_constraints_are_enforced():
    from mikoshilang.library.core import REGISTRY
    assert any(("PosInteger",) in sig.params for sig in REGISTRY["FareySequence"].signature)
    r = Session(check_types=True)
    with pytest.raises(TypeError, match="FareySequence: argument 1 0 is not of type PosInteger"):
        r.evaluate("FareySequence[0]")
    with pytest.raises(ValueError):                    # unchecked: the function's own ValueError
        Session().evaluate("FareySequence[0]")
    s = Session(check_types="report")
    s.evaluate("TypeSignature[tsNN, {NonNegInteger} -> Integer]; tsNN[n_] := n; tsNN[-1]; tsNN[0]")
    assert [(m["function"], m["expected"]) for m in s.type_mismatches] == [("tsNN", "NonNegInteger")]
