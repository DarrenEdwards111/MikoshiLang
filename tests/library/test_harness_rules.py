"""Tests for the case-grouping rules in tests/library/_harness.py: bare-symbol and $-variable cases, and
Via wrapper cases that count towards an inner head only when the expected value depends on its output."""
import pickle

import pytest

from _harness import PROBE, Pred, Raises, Via, check, head_of
from mikoshilang import verification


# --- bare symbols and $-variables ---------------------------------------------------------------------------

@pytest.mark.parametrize("src, head", [
    ("Pi", "Pi"), ("  E ", "E"), ("$MachinePrecision", "$MachinePrecision"), ("$Version", "$Version"),
])
def test_bare_symbol_cases_have_that_head(src, head):
    assert head_of((src, "x")) == head


@pytest.mark.parametrize("src", ["x = 3", "1 + 2", "$a = 1", "x -> 2", ""])
def test_setup_and_headless_lines_have_no_head(src):
    assert head_of((src, "x")) is None


def test_bare_symbol_value_check():
    check(("Pi", "Pi"))
    check(("$MachinePrecision", 15.954589770191003))
    with pytest.raises(AssertionError):
        check(("Pi", "E"))


def test_bare_symbol_type_check():
    check(("$MachinePrecision", Pred(lambda v: isinstance(v, float) and v > 15, "a machine real above 15")))
    with pytest.raises(AssertionError):
        check(("$MachinePrecision", Pred(lambda v: isinstance(v, int), "an integer")))


def test_load_cases_groups_bare_symbols(tmp_path):
    import shutil
    from pathlib import Path
    here = Path(__file__).parent
    shutil.copy(here / "_harness.py", tmp_path / "_harness.py")
    (tmp_path / "test_tmp_cases.py").write_text(
        'CASES = [("$MachinePrecision", 15.954589770191003), ("Pi", "Pi"), ("x = 3", "3"), ("Sin[0]", "0")]\n')
    got = verification.load_cases(tmp_path)
    assert {h: len(v) for h, v in got.items()} == {"$MachinePrecision": 1, "Pi": 1, "Sin": 1}


# --- wrapper cases (Via) -------------------------------------------------------------------------------------

def test_plain_wrapper_case_counts_towards_the_outer_head():
    assert head_of(("ToString[TeXForm[x^2]]", '"x^2"')) == "ToString"
    assert head_of(("N[Gamma[1/2]]", 1.772453850905516)) == "N"


def test_via_case_counts_towards_the_inner_head():
    assert head_of((Via("TeXForm", "ToString[TeXForm[x^2]]"), '"x^2"')) == "TeXForm"
    assert head_of((Via("Gamma", "N[Gamma[1/2]]"), 1.772453850905516)) == "Gamma"


def test_via_case_passes_when_the_expectation_depends_on_the_inner_output():
    check((Via("TeXForm", "ToString[TeXForm[x^2]]"), '"x^2"'))
    check((Via("Gamma", "N[Gamma[1/2]]"), 1.772453850905516))
    check((Via("Sin", "ToString[Sin[0]]"), '"0"'))


def test_via_case_fails_when_the_expectation_does_not_depend_on_the_inner_output():
    with pytest.raises(AssertionError, match="does not depend on Sin's own output"):
        check((Via("Sin", "Length[{Sin[1], 2}]"), "2"))
    with pytest.raises(AssertionError, match="does not depend on Gamma's own output"):
        check((Via("Gamma", "Head[{Gamma[1/2]}]"), "List"))
    with pytest.raises(AssertionError, match="does not depend"):
        check((Via("TeXForm", "StringQ[ToString[TeXForm[x^2]]]"), True))


def test_via_case_still_needs_the_right_value():
    with pytest.raises(AssertionError):
        check((Via("Gamma", "N[Gamma[1/2]]"), 2.0))


def test_via_requires_the_inner_head_inside_a_wrapper():
    with pytest.raises(ValueError, match="does not occur"):
        Via("Gamma", "N[Sin[1]]")
    with pytest.raises(ValueError, match="does not occur"):
        Via("Gamma", "N[LogGamma[1]]")           # LogGamma is not Gamma
    with pytest.raises(ValueError, match="already has head"):
        Via("Gamma", "Gamma[1/2]")


def test_via_probe_and_pickling():
    v = Via("Gamma", "N[Gamma[1/2] + LogGamma[2]]")
    assert v.probe() == f"N[{PROBE}[1/2] + LogGamma[2]]"
    w = pickle.loads(pickle.dumps(v))
    assert isinstance(w, Via) and w.inner == "Gamma" and str(w) == str(v)


def test_via_raises_cases_are_not_probed():
    check((Via("Gamma", "N[Gamma[1, 2, 3, 4, 5]]"), Raises()))
