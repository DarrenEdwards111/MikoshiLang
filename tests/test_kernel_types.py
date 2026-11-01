"""Hand-written signatures of the kernel built-ins (mikoshilang.kernel_types) and the narrowed library types."""
import pytest

from mikoshilang import Session, _typestate, types as T
from mikoshilang.kernel_types import KERNEL_SIGNATURES, kernel_signatures


def _report(src):
    """Mismatches of one evaluation in report mode; the function's own error (if any) still surfaces after."""
    s = Session(check_types="report")
    try:
        s.evaluate(src)
    except (ValueError, TypeError, ArithmeticError):
        pass
    return [(m["function"], m["expected"]) for m in s.type_mismatches]


# ---- kernel_signatures ----------------------------------------------------------------------------------------

def test_kernel_signatures_parse():
    sigs = kernel_signatures("Plus")
    assert sigs and sigs[0].rest == ("Or", (("Number",), ("ListOf", ("Expr",)), ("Assoc",)))


def test_kernel_signatures_unknown_head():
    assert kernel_signatures("NoSuchKernelHead") is None
    assert kernel_signatures("BitLength") is None          # library functions carry their own annotations


def test_kernel_signatures_cover_all_names_and_parse():
    for name in KERNEL_SIGNATURES:
        for sig in kernel_signatures(name):
            for p in sig.params:
                assert p[0] in T.ATOMS or p[0] in T.CONTAINERS or p[0] == "Or", (name, p)


# ---- check_kernel_call ----------------------------------------------------------------------------------------

def test_check_kernel_call_off_without_a_mode():
    assert T.check_kernel_call("Plus", ["a", 1]) is None


def test_check_kernel_call_reports_and_returns_signature():
    s = Session(check_types="report")
    s.evaluate('Plus["a", 1]')
    assert [(m["function"], m["position"], m["expected"]) for m in s.type_mismatches][:1] == [
        ("Plus", 1, "Number | ListOf[Expr] | Assoc")]


def test_check_kernel_call_raises_in_raise_mode():
    s = Session(check_types=True)
    with pytest.raises(TypeError, match="^StringLength: argument 1 3 is not of type"):
        s.evaluate("StringLength[3]")
    assert _typestate.ACTIVE == 0


# ---- signatures_of / TypeSignature on kernel heads -------------------------------------------------------------

def test_signatures_of_kernel_head():
    assert T.signatures_of("Not")[0].result == ("Boolean",)


def test_signatures_of_held_kernel_head():
    assert T.signatures_of("Block")[0].rest == ("Held",)


def test_typesignature_shows_kernel_signature():
    assert str(Session().evaluate("TypeSignature[Plus]")).startswith("{{(Number | ListOf[Expr] | Assoc)")


# ---- what the kernel signatures catch, and what they leave alone -----------------------------------------------

@pytest.mark.parametrize("src, head, expected", [
    ('Plus["a", 1]', "Plus", "Number | ListOf[Expr] | Assoc"),
    ("Nest[f, x, -1]", "Nest", "NonNegInteger"),
    ("StringLength[3]", "StringLength", "String | ListOf[Expr]"),
    ("Det[{1, 2}]", "Det", "MatrixOf[Expr]"),
    ('Not["a"]', "Not", "Boolean"),
])
def test_kernel_mismatches(src, head, expected):
    assert (head, expected) in _report(src)


@pytest.mark.parametrize("src", [
    "Plus[x, 1]", "Times[2, {1, 2}]", "Range[1, 10, 2]", "Nest[f, x, 3]", 'StringJoin["a", "b"]',
    "Det[{{1, 2}, {3, 4}}]", "Not[p]", "Greater[x, 1]", "Power[2, 1/2]",
])
def test_kernel_symbolic_and_valid_calls_are_clean(src):
    s = Session(check_types="report")
    s.evaluate(src)
    assert s.type_mismatches == []


# ---- narrowed library signatures --------------------------------------------------------------------------------

@pytest.mark.parametrize("src, head, expected", [
    ('Gamma["a"]', "Gamma", "Number"),
    ('Prime[0]', "Prime", "PosInteger"),
    ('StringReverse[5]', "StringReverse", "String"),
    ('Partition[{1, 2, 3}, 0]', "Partition", "PosInteger"),
    ('Cardinality["abc"]', "Cardinality", "ListOf[Expr]"),
])
def test_narrowed_library_mismatches(src, head, expected):
    assert (head, expected) in _report(src)


@pytest.mark.parametrize("src", ["Gamma[x]", "Gamma[5]", "Prime[n]", "Fibonacci[10]", "GCD[1/2, 1/3]",
                                 "StringReverse[s]", "Partition[{1, 2, 3, 4}, 2]", "BesselJ[0, 1.5]"])
def test_narrowed_library_symbolic_and_valid_calls_are_clean(src):
    s = Session(check_types="report")
    s.evaluate(src)
    assert s.type_mismatches == []
