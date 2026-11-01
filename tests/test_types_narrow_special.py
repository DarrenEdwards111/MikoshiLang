"""Numeric types of the listable special functions registered by the wl_special, wl_special_misc and
wl_special_elliptic factories (arguments are checked after Listable threading, so each one is a scalar)."""
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
    ("BesselJ", ["Numeric", "Numeric"], "Numeric"),
    ("EllipticK", ["Numeric"], "Numeric"),
    ("AngerJ", ["Numeric", "Numeric"], "Numeric"),
    ("QGamma", ["Numeric", "Numeric"], "Numeric"),
    ("HeavisidePi", ["Numeric"], "Numeric"),
    ("RamanujanTau", ["Numeric"], "Numeric"),
])
def test_narrowed_signatures(name, params, result):
    assert (params, result) in _types(name)


@pytest.mark.parametrize("name", ["CombinatorS", "MultipleZeta", "HypergeometricPFQ", "QHypergeometricPFQ"])
def test_non_listable_heads_stay_expr(name):
    assert all(all(p == "Expr" or p == "Expr..." for p in ps) for ps, _ in _types(name))


@pytest.mark.parametrize("src, head, expected", [
    ('BesselJ[0, "x"]', "BesselJ", "Numeric"),
    ('EllipticK["m"]', "EllipticK", "Numeric"),
    ("AiryAi[True]", "AiryAi", "Numeric"),
    ('QGamma[x, "q"]', "QGamma", "Numeric"),
    ('HeavisidePi["a"]', "HeavisidePi", "Numeric"),
    ("StruveL[0, a -> b]", "StruveL", "Numeric"),
])
def test_narrowed_mismatches(src, head, expected):
    assert (head, expected) in _report(src)


@pytest.mark.parametrize("src", [
    "BesselJ[0, 1.5]", "BesselJ[n, x]", "BesselJ[0, {1., 2.}]", "EllipticK[1/2]", "EllipticK[m]", "N[EllipticK[0.3]]",
    "AiryAi[0.5]", "AngerJ[1/2, 0.3]", "QGamma[3, 1/2]", "HeavisidePi[1/4]", "HeavisidePi[x, y]",
    "RamanujanTau[5]", "Zeta[3]", "PolyLog[2, 1/2]", "N[KelvinBer[0, 1]]",
])
def test_symbolic_and_valid_calls_are_clean(src):
    assert _clean(src) == []


def test_values_unchanged_without_checking():
    s = Session()
    assert s.evaluate("HeavisidePi[1/4]") == 1
    assert s.evaluate("RamanujanTau[2]") == -24
    assert abs(float(s.evaluate("N[BesselJ[0, 1]]")) - 0.7651976865579666) < 1e-12


def test_raise_mode_names_the_head():
    s = Session(check_types=True)
    with pytest.raises(TypeError, match="^EllipticK: argument 1 'm' is not of type Numeric"):
        s.evaluate('EllipticK["m"]')
