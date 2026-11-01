"""Regression tests for the correctness fix-up pass (PR "fixups-correctness"):

* Expand / ExpandAll / Together / Simplify / Factor / Apart / Cancel / Collect with a list inside a function
  (``Expand[2 f[{a}]]`` used to raise AttributeError from inside SymPy);
* re-canonicalisation of Flat / Orderless Plus and Times whose operands SymPy cannot represent (a rule that
  produced ``n gg[...]`` left ``Times[2, Times[I, Times[3, gg[...]]]]`` nested and unfolded);
* FindRoot honouring MaxIterations, and other options that used to be accepted but ignored.

Expected values follow the Wolfram Language semantics (reference.wolfram.com: Expand, Times "Flat, Orderless",
Plus "Flat, Orderless", FindRoot "MaxIterations"); structural checks use ``===`` so that the exact canonical form,
not only mathematical equality, is tested.  Placeholder symbols are prefixed ``fz`` and defined only inside Block.
"""
import pytest

from _harness import Raises, check, ids

LISTS = [
    ("Expand[2 fzf[{a}]] === 2 fzf[{a}]", True),
    ("Expand[(1 + x) fzf[{a, b}]] === x fzf[{a, b}] + fzf[{a, b}]", True),
    ("Expand[(1 + y) fzf[{{1 + x}}]] === y fzf[{{1 + x}}] + fzf[{{1 + x}}]", True),
    ("Expand[fzf[{a}] (a + b)^2] === a^2 fzf[{a}] + 2 a b fzf[{a}] + b^2 fzf[{a}]", True),
    ("ExpandAll[fzf[{(1 + x)^2}]] === fzf[{1 + 2 x + x^2}]", True),
    ("ExpandAll[(1 + x) fzf[{(1 + y)^2}]] === fzf[{1 + 2 y + y^2}] + x fzf[{1 + 2 y + y^2}]", True),
    ("Together[1/x + fzf[{a}]] === (1 + x fzf[{a}])/x", True),
    ("Simplify[fzf[{a}] + fzf[{a}]] === 2 fzf[{a}]", True),
    ("FullSimplify[fzf[{a}] - fzf[{a}]]", "0"),
    ("Apart[fzf[{a}]/(x (x + 1)), x] === fzf[{a}]/x - fzf[{a}]/(1 + x)", True),
    ("Collect[x fzf[{a}] + x b, x] === x (b + fzf[{a}])", True),
    ("Simplify[Cancel[(x^2 - 1) fzf[{a}]/(x - 1)] - (1 + x) fzf[{a}]]", "0"),
    ("Expand[(x + 1)^2 fzf[{a, {b, c}}]] - (x^2 + 2 x + 1) fzf[{a, {b, c}}] // Expand", "0"),
    ("Head[Expand[2 fzf[{a}]][[2, 1]]]", "List"),
    ("D[fzf[{y}], x]", "0"),
    ("D[fzf[{y}] + x^2, x]", "2 x"),
    # differentiating through a list argument has no meaning SymPy can carry: a clear error, never List'[x]
    ("D[fzf[{x}], x]", Raises(NotImplementedError, "D: differentiating through a list argument")),
    ("D[x fzf[{x, y}], x]", Raises(NotImplementedError, "D: differentiating through a list argument")),
]

CANONICAL = [
    # the original report: a user rule produces n gg[...]; the result must be flat, sorted and folded
    ("Block[{fzg}, fzg[{l___, x_ n_ ? NumberQ, r___}] := n fzg[{l, x, r}]; "
     "fzg[{2 fzj, 3 I fzk}] === 6 I fzg[{fzj, fzk}]]", True),
    ("Block[{fzg}, fzg[{l___, x_ n_ ? NumberQ, r___}] := n fzg[{l, x, r}]; "
     "FreeQ[List @@ fzg[{2 fzj, 3 I fzk}], _Times]]", True),
    ("Block[{fzg}, fzg[{l___, x_ n_ ? NumberQ, r___}] := n fzg[{l, x, r}]; "
     "fzg[{2 fzj, 3 I fzk, 5 \"s\"}] === 30 I fzg[{fzj, fzk, \"s\"}]]", True),
    ("Times[2, Times[x, fzg[\"s\"]]] === Times[2, x, fzg[\"s\"]]", True),
    ("Length[Times[2, Times[x, fzg[\"s\"]]]]", "3"),
    ("Times[2, Times[I, Times[3, fzg[fzh[x][y]]]]] === 6 I fzg[fzh[x][y]]", True),
    ("Plus[2, Plus[x, fzg[\"s\"]]] === Plus[2, x, fzg[\"s\"]]", True),
    ("Length[Plus[2, Plus[x, fzg[a -> b]]]]", "3"),
    ("fzg[\"s\"] + fzg[\"s\"] + 1 + 2 === 3 + 2 fzg[\"s\"]", True),
    ("2 \"s\" + 3 \"s\" === 5 \"s\"", True),
    ("fzh[a][b] + fzh[a][b] === 2 fzh[a][b]", True),
    ("fzh[a][b]^2 fzh[a][b]^3 === fzh[a][b]^5", True),
    ("\"a\" \"b\" === \"b\" \"a\"", True),
    ("x fzg[\"s\"] === fzg[\"s\"] x", True),
    ("Times[2, Times[x, fzg[Hold[1 + 1]]]] === Times[x, Times[2, fzg[Hold[1 + 1]]]]", True),
    ("Times[0, fzg[\"s\"]]", "0"),
    ("Times[0., fzg[\"s\"]]", "0."),
    ("fzg[\"s\"] - fzg[\"s\"]", "0"),
    ("Head[1.5 fzg[\"s\"] + 2]", "Plus"),
    # the opaque operand sorts like any other fzg[...]: x before fzg[...] (Wolfram canonical order)
    ("Times[2, fzg[\"s\"], x] === Times[2, x, fzg[\"s\"]] && Times[2, x, fzg[a]] === Times[fzg[a], x, 2]", True),
    ("Block[{fzr}, fzr[x_] := Times[x, Times[2, fzg[\"s\"]]]; Length[fzr[3]]]", "2"),
    ("Block[{fzr}, fzr[x_] := Times[x, Times[2, fzg[\"s\"]]]; fzr[3] === 6 fzg[\"s\"]]", True),
]

NI = Raises(NotImplementedError)
OPTIONS = [
    # FindRoot honours MaxIterations (mpmath maxsteps); too few iterations is an error, never an unconverged value
    ("FindRoot[Cos[x] - x, {x, 1}, MaxIterations -> 1]",
     Raises(ValueError, "FindRoot: the search did not converge within MaxIterations -> 1 iterations")),
    ("FindRoot[Cos[x] - x, {x, 1}, MaxIterations -> 3]", Raises(ValueError, "MaxIterations -> 3")),
    ("FindRoot[Cos[x] - x, {x, 1}, MaxIterations -> 10]", "{x -> 0.7390851332151607}"),
    ("FindRoot[Cos[x] - x, {x, 1}, MaxIterations -> 100]", "{x -> 0.7390851332151607}"),
    ("FindRoot[Cos[x] - x, {x, 1}, MaxIterations -> Infinity]", "{x -> 0.7390851332151607}"),
    ("FindRoot[Cos[x] - x, {x, 1}, MaxIterations -> Automatic]", "{x -> 0.7390851332151607}"),
    ("FindRoot[Cos[x] - x, {x, 0, 1}, MaxIterations -> 1]", Raises(ValueError, "MaxIterations -> 1")),
    ("FindRoot[{x^2 + y^2 == 4, x == y}, {{x, 1}, {y, 2}}, MaxIterations -> 1]",
     Raises(ValueError, "MaxIterations -> 1")),
    ("FindRoot[{x^2 + y^2 == 4, x == y}, {{x, 1}, {y, 2}}, MaxIterations -> 20]",
     "{x -> 1.4142135623730951, y -> 1.4142135623730951}"),
    ("FindRoot[x^2 + 1, {x, 1}, MaxIterations -> 5]", Raises(ValueError, "FindRoot: no root was found")),
    ("FindRoot[Cos[x] - x, {x, 1}, MaxIterations -> 0]", Raises(ValueError, "FindRoot: MaxIterations must be")),
    ("FindRoot[Cos[x] - x, {x, 1}, MaxIterations -> 2.5]", Raises(ValueError, "FindRoot: MaxIterations must be")),
    ("FindRoot[Cos[x] - x, {x, 1}, MaxIterations -> 30, WorkingPrecision -> 30] // Last // Last // Precision",
     "30."),
    # options that used to be accepted and silently ignored now raise unless they have their default value
    ("FindRoot[Cos[x] - x, {x, 1}, Method -> \"Newton\"]", Raises(NotImplementedError, "FindRoot: the option Method")),
    ("FindRoot[Cos[x] - x, {x, 1}, AccuracyGoal -> 3]", Raises(NotImplementedError, "FindRoot: the option Accuracy")),
    ("FindRoot[Cos[x] - x, {x, 1}, EvaluationMonitor :> Print[x]]", NI),
    ("FindRoot[Cos[x] - x, {x, 1}, Method -> Automatic, Compiled -> False]", "{x -> 0.7390851332151607}"),
    ("NIntegrate[x^2, {x, 0, 1}, MaxRecursion -> 3]", Raises(NotImplementedError, "NIntegrate: the option MaxRec")),
    ("NIntegrate[x^2, {x, 0, 1}, PrecisionGoal -> 12]", Raises(NotImplementedError, "PrecisionGoal -> 12")),
    ("NIntegrate[x^2, {x, 0, 1}, PrecisionGoal -> 6]", 1 / 3),
    ("NIntegrate[x^2, {x, 0, 1}, Method -> \"GaussKronrod\"]", 1 / 3),
    ("NSum[1/n^2, {n, 1, Infinity}, Method -> \"EulerMaclaurin\"]", NI),
    ("NSum[1/n^2, {n, 1, Infinity}, AccuracyGoal -> 5]", 1.6449340668482264),
    ("NProduct[1 + 1/n^2, {n, 1, Infinity}, MaxIterations -> 5]", NI),
    ("NSolve[x^2 == 2, x, Method -> \"Homotopy\"]", NI),
    ("NSolve[x^2 == 2, x, WorkingPrecision -> 20] // Length", "2"),
    ("NRoots[x^2 - 2 == 0, x, Method -> \"JenkinsTraub\"]", NI),
    ("NMinimize[x^2 - x, x, Integers]", Raises(NotImplementedError, "NMinimize: the domain Integers")),
    ("NMinimize[x^2 - x, x, Reals]", "{-0.25, {x -> 0.5}}"),
    ("NMaximize[x - x^2, x, Method -> \"NelderMead\"]", Raises(NotImplementedError, "NMaximize: the option Method")),
    ("NMinimize[x^2 - x, x, WorkingPrecision -> MachinePrecision]", "{-0.25, {x -> 0.5}}"),
    ("FindMinimum[x^2 - x, {x, 1}, MaxIterations -> 1]", Raises(ValueError, "FindMinimum")),
    ("LinearOptimization[x, {x >= 1}, x, Tolerance -> 0.1]", Raises(NotImplementedError, "Tolerance -> 0.1")),
    ("LinearOptimization[x, {x >= 1}, x, Method -> \"Simplex\"]", NI),
    ("LinearOptimization[x, {x >= 1}, x, Method -> Automatic]", "{x -> 1}"),
    ("NArgMin[x^2 - x, x, MaxIterations -> 3]", NI),
    ("ParametricConvexOptimization[(x - p)^2, {}, {x}, {p}, MaxIterations -> 3]", NI),
    ("FindShortestTour[{{0, 0}, {1, 0}, {1, 1}}, Method -> \"Greedy\"]", NI),
    ("FindShortestTour[{{0, 0}, {1, 0}, {1, 1}}, Method -> Automatic] // First", "2 + Sqrt[2]"),
    ("DistanceMatrix[{1, 2}, WorkingPrecision -> 30]", NI),
    ("FindIntegerNullVector[{1, Sqrt[2], 1 + Sqrt[2]}, ZeroTest -> (# == 0 &)]", NI),
    ("Interpolation[{1, 2, 3, 4}, PeriodicInterpolation -> True]", NI),
    ("Interpolation[{1, 2, 3, 4}, PeriodicInterpolation -> False][2]", 2.0),
    ("ListInterpolation[{1, 2, 3, 4}, Method -> \"Spline\"]", NI),
    ("NDSolve[{y'[x] == y[x], y[0] == 1}, y, {x, 0, 1}, Method -> \"ExplicitRungeKutta\"]", NI),
    ("NDSolve[{y'[x] == y[x], y[0] == 1}, y, {x, 0, 1}, PrecisionGoal -> 12]", NI),
]

CASES = LISTS + CANONICAL + OPTIONS


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_fixups(case):
    check(case)


def test_differentialroot_keeps_global_mpmath_precision():
    # DifferentialRoot[...][x] used to reset mpmath.mp.dps to 15 globally (clobbering a caller's precision)
    import mpmath
    from _harness import ev
    with mpmath.workdps(40):
        ev("DifferentialRoot[Function[{y, x}, {y'[x] == y[x], y[0] == 1}]][1.]")
        assert mpmath.mp.dps == 40
