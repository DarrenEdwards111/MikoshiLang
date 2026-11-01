"""Hand-narrowed types of the logic, text and association modules (logic, wl_logic, wl_text, wl_assoc)."""
import pytest

from mikoshilang import Session, types as T
from mikoshilang.library.core import REGISTRY


def _report(src):
    """Mismatches of one evaluation in report mode; the function's own error (if any) still surfaces after."""
    s = Session(check_types="report")
    try:
        s.evaluate(src)
    except (ValueError, TypeError, ArithmeticError, NotImplementedError):
        pass
    return [(m["function"], m["expected"]) for m in s.type_mismatches]


def _clean(src):
    s = Session(check_types="report")
    s.evaluate(src)
    return s.type_mismatches


def _types(name):
    """Parameter and result type texts of every implementation of a library head."""
    return [([T.type_text(p) for p in sig.params] + ([T.type_text(sig.rest) + "..."] if sig.rest else []),
             T.type_text(sig.result)) for sig in REGISTRY[name].signature]


@pytest.mark.parametrize("name, params, result", [
    ("Xor", ["Boolean..."], "Boolean"),
    ("Boole", ["Boolean"], "Integer"),
    ("SatisfiabilityCount", ["Expr", "Var | ListOf[Expr] | Integer"], "Integer"),
    ("Iff", ["Boolean..."], "Boolean"),
    ("BaseEncode", ["Expr", "String"], "String"),
    ("KeyComplement", ["ListOf[Expr]"], "Assoc"),
])
def test_narrowed_signatures(name, params, result):
    assert (params, result) in _types(name)


@pytest.mark.parametrize("src, head, expected", [
    ("Xor[1, 2]", "Xor", "Boolean"),
    ("Boole[2]", "Boole", "Boolean"),
    ('Implies["p", q]', "Implies", "Boolean"),
    ("ForAll[1, x > 0]", "ForAll", "Var | ListOf[Var]"),
    ('SatisfiabilityInstances[a || b, {a, b}, 0]', "SatisfiabilityInstances", "PosInteger"),
    ('Hash["abc", 5]', "Hash", "String | ListOf[Expr]"),
    ('BaseEncode[ByteArray[{1, 2}], 64]', "BaseEncode", "String"),
    ("Snippet[\"a\", 0]", "Snippet", "PosInteger"),
    ("KeyComplement[<|a -> 1|>]", "KeyComplement", "ListOf[Expr]"),
])
def test_narrowed_mismatches(src, head, expected):
    assert (head, expected) in _report(src)


@pytest.mark.parametrize("src", [
    "Xor[True, False]", "Xor[x, y]", "Implies[x > 0, y]", "Boole[{True, False, x}]", "Majority[True, False, True]",
    "SatisfiabilityCount[a || b, {a, b}]", "SatisfiableQ[a && !a]", "ForAll[{x, y}, True]", "Exists[x, x || y]",
    'Hash["abc", "MD5"]', 'Hash["abc", {"SHA", 256}, "HexString"]', "SatisfiabilityInstances[BooleanFunction[30, 3], 3]", 'BaseEncode[ByteArray[{1, 2, 3}]]', 'CharacterName["ξ"]',
    "KeyComplement[{<|a -> 1, b -> 2|>, <|b -> 5|>}]", "Merge[{<|a -> 1|>, <|a -> 3|>}, Total]",
    "Iff[True, True]", 'SyntaxLength["a[b]"]',
])
def test_symbolic_and_valid_calls_are_clean(src):
    assert _clean(src) == []


def test_values_unchanged_without_checking():
    s = Session()
    assert s.evaluate("Xor[True, False]") is True
    assert s.evaluate("SatisfiabilityCount[a || b, {a, b}]") == 3
    assert str(s.evaluate("Xor[1, 2]")) == "Xor[1, 2]"


def test_raise_mode_names_the_head():
    s = Session(check_types=True)
    with pytest.raises(TypeError, match="^Xor: argument 1 1 is not of type Boolean"):
        s.evaluate("Xor[1, 2]")
