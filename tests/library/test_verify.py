"""Verification pillar V1: Verify[expr] (MikoshiLang extension, see docs/verification.md)."""
import pytest
from _harness import Pred, Raises, check, ids


def tag(name, method=None):
    """Result is name[value, info] (and info's Method contains method)."""
    def ok(g):
        if not (hasattr(g, "head") and g.head == name):
            return False
        if method is None:
            return True
        info = {r.args[0]: r.args[1] for r in g.args[1].args}
        return method in str(info.get("Method", ""))
    return Pred(ok, f"{name}[...] ({method})" if method else f"{name}[...]")


CASES = [
    # calculus
    ("Verify[Integrate[Sin[x]^2, x]]", tag("Checked", "Differentiation")),
    ("Verify[Integrate[x^3 + 2 x, x]]", tag("Checked", "Differentiation")),
    ("Verify[Integrate[x Exp[x], {x, 0, 1}]]", tag("Checked", "quadrature")),
    ("Verify[Integrate[f[x], x]]", tag("Unchecked")),                       # not evaluated
    ("Verify[D[Sin[x] x^2, x]]", tag("Checked", "Numerical differentiation")),
    ("Verify[Limit[Sin[x]/x, x -> 0]]", tag("Checked", "approach")),
    ("Verify[Limit[(1 + 1/n)^n, n -> Infinity]]", tag("Checked", "approach")),
    ("Verify[Series[Exp[x], {x, 0, 5}]]", tag("Checked", "Remainder")),
    ("Verify[DSolve[y'[x] == y[x], y[x], x]]", tag("Checked", "differential equation")),
    ("Verify[DSolve[y''[x] + y[x] == 0, y[x], x]]", tag("Checked", "differential equation")),
    # equations
    ("Verify[Solve[x^2 == 2, x]]", tag("Certified", "Substitution")),
    ("Verify[Solve[x^2 + a x + 1 == 0, x]]", tag("Certified", "Substitution")),
    ("Verify[Solve[{x + y == 3, x - y == 1}, {x, y}]]", tag("Certified", "Substitution")),
    ("Verify[NSolve[x^3 - 2 == 0, x]]", tag("Checked", "Substitution")),
    ("Verify[Solve[x == x + 1, x]]", tag("Unchecked")),                     # no solutions: completeness not checked
    ("Verify[Solve[x^2 == -1 && x > 0, x]]", Raises(NotImplementedError)),   # Solve's own error propagates
    # number theory certificates (reference: OEIS A000043 Mersenne exponent 89; 561 is a Carmichael number)
    ("Verify[FactorInteger[2^64 + 1]]", tag("Certified", "primality proof")),   # Landry/Euler: 274177 * 67280421310721
    ("Verify[FactorInteger[10^30 + 57]]", tag("Certified", "primality proof")),
    ("Verify[FactorInteger[360]]", tag("Certified", "Product of the factors")),
    ("Verify[PrimeQ[2^89 - 1]]", tag("Certified", "Pratt")),
    ("Verify[PrimeQ[561]]", tag("Certified", "compositeness")),
    ("Verify[PrimeQ[97]]", tag("Certified", "Miller-Rabin")),
    ("Verify[GCD[240, 46]]", tag("Certified", "Bezout")),
    ("Verify[GCD[0, 5]]", tag("Certified", "Bezout")),
    ("Verify[GCD[2^100, 3^50]]", tag("Certified", "Bezout")),
    # algebra and linear algebra
    ("Verify[Factor[x^4 - 1]]", tag("Certified", "Expansion")),
    ("Verify[Simplify[Sin[x]^2 + Cos[x]^2]]", tag("Checked", "Equivalence")),
    ("Verify[Expand[(x + 1)^5]]", tag("Certified", "Equivalence")),
    ("Verify[Inverse[{{1, 2}, {3, 4}}]]", tag("Certified", "IdentityMatrix")),
    ("Verify[Inverse[{{1., 2.}, {3., 4.}}]]", tag("Checked", "IdentityMatrix")),
    ("Verify[LinearSolve[{{1, 2}, {3, 4}}, {5, 6}]]", tag("Certified", "m . x == b")),
    ("Verify[Det[{{1, 2}, {3, 4}}]]", tag("Certified", "Berkowitz")),
    ("Verify[Det[{{1., 2.}, {3., 4.}}]]", tag("Checked", "numpy")),
    ("Verify[Eigenvalues[{{2, 1}, {1, 2}}]]", tag("Checked", "characteristic polynomial")),
    # no checker
    ("Verify[Sort[{3, 1, 2}]]", Pred(lambda g: g.head == "Unchecked" and str(g.args[0]) == "{1, 2, 3}",
                                     "Unchecked[{1, 2, 3}, reason]")),
    ("Verify[1 + 1]", tag("Unchecked")),
    ("Verify[Verify[1]]", tag("Unchecked")),
    ("Verify[Integrate[Sin[x], x], 3]", Raises(ValueError)),                  # wrong argument count
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_verify(case):
    check(case)
