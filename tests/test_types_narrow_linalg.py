"""Hand-narrowed types of the linear algebra and numerics modules (linalg, numerics, wl_linalg, wl_linalg_optim,
wl_numerics, wl_numcalc): matrices, vectors, sizes and numeric parts."""
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
    ("Rank", ["MatrixOf[Expr]"], "Integer"),
    ("SingularValueDecomposition", ["MatrixOf[Expr]"], "ListOf[Expr]"),
    ("ToeplitzMatrix", ["PosInteger | ListOf[Expr]", "ListOf[Expr]"], "ListOf[Expr]"),
    ("RotationMatrix", ["Numeric | ListOf[Expr]", "ListOf[Expr]"], "ListOf[Expr]"),
    ("Re", ["Numeric"], "Numeric"),
    ("Precision", ["Expr"], "Real"),
])
def test_narrowed_signatures(name, params, result):
    assert (params, result) in _types(name)


@pytest.mark.parametrize("src, head, expected", [
    ("Rank[{1, 2}]", "Rank", "MatrixOf[Expr]"),
    ("ToeplitzMatrix[0]", "ToeplitzMatrix", "PosInteger | ListOf[Expr]"),
    ('HilbertMatrix["3"]', "HilbertMatrix", "PosInteger | ListOf[PosInteger]"),
    ("Minors[{1, 2}]", "Minors", "MatrixOf[Expr]"),
    ('Re["a"]', "Re", "Numeric"),
    ('IntegerPart["a"]', "IntegerPart", "Numeric"),
    ('SetPrecision[1/3, "20"]', "SetPrecision", "Numeric"),
    ('Chop[0.1, "x"]', "Chop", "Real"),
    ("NRoots[x^2 == 1, 3]", "NRoots", "Var"),
    ("HadamardMatrix[0]", "HadamardMatrix", "PosInteger"),
])
def test_narrowed_mismatches(src, head, expected):
    assert (head, expected) in _report(src)


@pytest.mark.parametrize("src", [
    "Rank[{{1, 2}, {2, 4}}]", "MatrixRank[{{a, b}, {c, d}}]", "ToeplitzMatrix[3]", "ToeplitzMatrix[{a, b, c}]",
    "HilbertMatrix[{2, 3}]", "RotationMatrix[t]", "RotationMatrix[Pi/2, {0, 0, 1}]", "ScalingMatrix[{2, 3}]",
    "ScalingMatrix[s, {1, 1}]", "Minors[{{1, 2}, {3, 4}}]", "Eigenvalues[{{2, 1}, {1, 2}}]", "Re[x]",
    "Re[{1 + I, 2}]", "IntegerPart[-5/4]", "Precision[1.5]", "SetPrecision[x, MachinePrecision]",
    "Orthogonalize[{{1, 0, 1}, {1, 1, 1}}]", "NRoots[x^2 - 3 x + 2 == 0, x]", "ZeroMatrix[2, 3]",
])
def test_symbolic_and_valid_calls_are_clean(src):
    assert _clean(src) == []


def test_lambda_aliases_now_annotated():
    for name in ("Rank", "SVD", "SingularValueDecomposition"):
        assert _types(name)[0][0] == ["MatrixOf[Expr]"]
    assert Session().evaluate("Rank[{{1, 2}, {2, 4}}]") == 1
    assert Session().evaluate("Length[SVD[N[{{1, 0}, {0, 2}}]]]") == 3


def test_raise_mode_names_the_head():
    s = Session(check_types=True)
    with pytest.raises(TypeError, match="^Re: argument 1 'a' is not of type Numeric"):
        s.evaluate('Re["a"]')
