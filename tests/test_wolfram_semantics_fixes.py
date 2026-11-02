"""Wolfram-semantics fixes: tagged Confirm outside Enclose, UniformDistribution[], PermutationGroup with a
non-list argument, NonCommutativeMultiply[a], SymmetrizedReplacePart on non-symmetric arrays."""
import pytest

from mikoshilang import Session


@pytest.fixture
def s():
    return Session()


# ---- Confirm outside Enclose returns a Failure -------------------------------------------------------------------

def test_tagged_confirm_without_enclose_returns_failure(s):
    out = s.evaluate('Confirm[$Failed, Null, "t"]')
    assert out.head == "Failure" and out.args[0] == "ConfirmationFailed"


def test_tagged_confirm_without_enclose_keeps_expression(s):
    out = s.evaluate('Confirm[Missing[], "info", "t"]')
    assert out.head == "Failure"
    assert "Missing[]" in str(out) and '"Information" -> "info"' in str(out)


def test_tagged_confirm_family_without_enclose(s):
    for src in ('ConfirmBy[$Failed, StringQ, Null, "t"]', 'ConfirmAssert[False, Null, "t"]',
                'ConfirmMatch[1, _String, Null, "t"]'):
        out = s.evaluate(src)
        assert out.head == "Failure" and out.args[0] == "ConfirmationFailed", src


def test_untagged_confirm_without_enclose_is_a_failure(s):
    assert s.evaluate("FailureQ[Confirm[$Failed]]") is True


def test_confirm_success_and_enclose_unchanged(s):
    assert s.evaluate('Confirm[2, Null, "t"]') == 2
    assert s.evaluate('Enclose[Confirm[$Failed, Null, "t"], 7 &, "t"]') == 7
    assert s.evaluate("Enclose[1 + Confirm[2]]") == 3


def test_other_uncaught_throws_still_raise(s):
    with pytest.raises(ValueError, match="Uncaught Throw"):
        s.evaluate("Throw[1]")


# ---- UniformDistribution[] ---------------------------------------------------------------------------------------

def test_uniform_default_normalises(s):
    assert str(s.evaluate("UniformDistribution[]")) == "UniformDistribution[{0, 1}]"


def test_uniform_default_equals_unit_interval(s):
    assert s.evaluate("UniformDistribution[] === UniformDistribution[{0, 1}]") is True


def test_uniform_default_properties(s):
    assert s.evaluate("Mean[UniformDistribution[]]") == s.evaluate("1/2")
    assert s.evaluate("Variance[UniformDistribution[]]") == s.evaluate("1/12")


def test_uniform_explicit_unchanged(s):
    assert str(s.evaluate("UniformDistribution[{2, 5}]")) == "UniformDistribution[{2, 5}]"


# ---- PermutationGroup ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("src", ["PermutationGroup[1]", 'PermutationGroup["a"]', "PermutationGroup[Cycles[{{1, 2}}]]"])
def test_permutationgroup_non_list_raises(s, src):
    with pytest.raises(ValueError, match="^PermutationGroup: "):
        s.evaluate(src)


def test_permutationgroup_list_stays_inert(s):
    assert str(s.evaluate("PermutationGroup[{Cycles[{{1, 2}}]}]")) == "PermutationGroup[{Cycles[{{1, 2}}]}]"


def test_permutationgroup_symbolic_stays(s):
    assert str(s.evaluate("PermutationGroup[g]")) == "PermutationGroup[g]"


def test_permutationgroup_order_unchanged(s):
    assert s.evaluate("GroupOrder[PermutationGroup[{Cycles[{{1, 2, 3}}]}]]") == 3


# ---- NonCommutativeMultiply[a] -------------------------------------------------------------------------------------

@pytest.mark.parametrize("src, want", [("NonCommutativeMultiply[a]", "a"), ("NonCommutativeMultiply[2]", "2"),
                                       ("NonCommutativeMultiply[f[x]]", "f[x]")])
def test_ncm_single_argument(s, src, want):
    assert str(s.evaluate(src)) == want


def test_ncm_several_arguments_unchanged(s):
    assert str(s.evaluate("a ** b ** c")) == "NonCommutativeMultiply[a, b, c]"


def test_ncm_flattening_unchanged(s):
    assert str(s.evaluate("NonCommutativeMultiply[a, NonCommutativeMultiply[b, c]]")) == \
        "NonCommutativeMultiply[a, b, c]"


# ---- SymmetrizedReplacePart ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("src", [
    "SymmetrizedReplacePart[{{a, b}, {c, d}}, {{1, 1} -> x}, Symmetric[{1, 2}]]",
    "SymmetrizedReplacePart[{{1, 2}, {3, 4}}, {{2, 1} -> x}, Antisymmetric[{1, 2}]]",
    "SymmetrizedReplacePart[{{1, 2}, {3, 1}}, {{1, 1} -> 0}, Symmetric[{1, 2}]]",
])
def test_symreplace_non_symmetric_raises(s, src):
    with pytest.raises(ValueError, match="^SymmetrizedReplacePart: the array does not have the given symmetry"):
        s.evaluate(src)


def test_symreplace_symmetric_array(s):
    assert s.evaluate("SymmetrizedReplacePart[{{a, b}, {b, d}}, {{2, 1} -> x}, Symmetric[{1, 2}]] == "
                      "{{a, x}, {x, d}}") is True


def test_symreplace_antisymmetric_array(s):
    assert s.evaluate("SymmetrizedReplacePart[{{0, b}, {-b, 0}}, {{2, 1} -> x}, Antisymmetric[{1, 2}]] == "
                      "{{0, -x}, {x, 0}}") is True


def test_symreplace_symmetrized_array_unchanged(s):
    assert s.evaluate("Normal[SymmetrizedReplacePart[SymmetrizedArray[{{a, b}, {b, c}}], {{2, 1} -> x}]] == "
                      "{{a, x}, {x, c}}") is True


def test_symreplace_doc_example_overwrites_whole_orbit(s):
    # Wolfram docs (Generalizations): the rule replaces both independent and dependent components, so the
    # non-symmetric pair b, c is overwritten rather than averaged.
    assert s.evaluate("SymmetrizedReplacePart[{{a, b}, {c, d}}, {{2, 1} -> x}, Symmetric[{1, 2}]] == "
                      "{{a, x}, {x, d}}") is True
