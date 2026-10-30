"""Differential tests for coverage batch 2 (numerics and special functions).

Expected values are Wolfram Language results (examples on the reference pages
reference.wolfram.com/language/ref/<Name>.html, or values derived from their Details sections) or independent
references computed here with mpmath at higher precision.  ``HP`` checks an arbitrary-precision result against
mpmath to the requested number of digits and also checks the result's precision.
"""
import mpmath as mp
import pytest
import sympy as sp

from _harness import Pred, Raises, check, ids

mp.mp.dps = 50


def HP(ref, digits, what=None):
    """An arbitrary-precision result that agrees with mpmath reference ``ref`` (a callable run at high precision)
    to ``digits`` significant digits and carries at least that precision."""
    def ok(got):
        with mp.workdps(digits + 30):
            want = ref() if callable(ref) else ref
            if isinstance(got, sp.Float):
                if got._prec < digits * 3.32:
                    return False
                g = mp.mpf(got._mpf_)
            else:
                g = mp.mpmathify(sp.N(sp.sympify(got), digits + 10))
            return abs(g - want) <= abs(want) * mp.mpf(10) ** (1 - digits) + mp.mpf(10) ** (-digits - 20)
    return Pred(ok, what or f"{digits} digits")


def Near(want, tol=1e-12):
    return Pred(lambda g: abs(complex(sp.N(sp.sympify(g) if not isinstance(g, (int, float, complex)) else g))
                              - complex(want)) <= tol * max(1, abs(complex(want))), f"~{want}")


PI_RATIONALIZE_UNION = ("{3, 201/64, 267/85, 311/99, 333/106, 55358/17621, 72043/22932, 85533/27226, 94053/29938, "
                        "99023/31520, 101508/32311, 102928/32763, 103638/32989, 103993/33102, 208341/66317, "
                        "833719/265381, 4272943/1360120, 53047102/16885417, 63885804/20335483, 74724506/23785549, "
                        "80143857/25510582, 245850922/78256779, 5419351/1725033, 1146408/364913, 312689/99532, "
                        "104348/33215, 355/113, 22/7}")

NUMERICS = [
    # number marks (parser): x`p, x``a, x*^e
    ("Precision[1.2`20]", "20."), ("Precision[1.2`]", "MachinePrecision"), ("2*^3", "2000"), ("2*^-3", "1/500"),
    ("1.5*^-20", 1.5e-20), ("Precision[3.14159`50 x]", "50."),
    # Precision / Accuracy (reference page examples)
    ("Precision[1.2]", "MachinePrecision"), ("Precision[12/10]", "Infinity"), ("Precision[x + 2]", "Infinity"),
    ("Precision[N[Pi, 30]]", "30."), ("Precision[{1, N[Pi, 30], N[E, 40]}]", "30."),
    ("Precision[{1.5, N[Pi, 30]}]", "MachinePrecision"), ("Precision[0``20]", "0."),
    ("Accuracy[0``20]", "20."), ("Accuracy[1/3]", "Infinity"), ("Accuracy[0.]", Near(307.6526555685888)),
    ("Accuracy[1.5]", Near(15.954589770191003 - 0.17609125905568124)),
    ("Accuracy[N[Pi, 30]]", Near(30 - 0.49714987269413385)),
    # precision propagates as the minimum over the inputs
    ("Precision[N[Pi, 20] + N[E, 50]]", "20."), ("Precision[N[Pi, 20] N[E, 50]]", "20."),
    ("MachineNumberQ[N[Pi, 20] + 1.5]", True), ("Precision[Sin[N[Pi, 40]/6]]", "40."),
    ("Gamma[N[1/3, 25]]", HP(lambda: mp.gamma(mp.mpf(1) / 3), 25)),
    ("N[MachinePrecision]", Near(15.954589770191003)), ("$MachinePrecision", Near(15.954589770191003)),
    ("$MachineEpsilon", 2.0 ** -52), ("1. + $MachineEpsilon == 1.", False), ("$MaxMachineNumber", 1.7976931348623157e308),
    ("$MinMachineNumber", 2.2250738585072014e-308), ("$MinMachineNumber/2 > 0", True),
    ("Precision[$MinMachineNumber]", "MachinePrecision"), ("$MaxMachineNumber > 10^308", True),
    ("Precision[$MaxMachineNumber]", "MachinePrecision"), ("1. + $MachineEpsilon/4 == 1.", True),
    ("$MachineEpsilon == 2.^-52", True), ("$MachinePrecision == N[MachinePrecision]", True),
    ("Floor[$MachinePrecision]", "15"), ("N[Pi, MachinePrecision]", 3.141592653589793),
    # SetPrecision / SetAccuracy
    ("SetPrecision[a + 2 b + 3 c, 20]", Pred(lambda g: "2.0000000000000000000*b" in str(g) and str(g).startswith("a + "),
                                             "a + 2.000...b + 3.000...c")),
    ("SetPrecision[a + 2 b + 3 c, MachinePrecision]", "a + 2. b + 3. c"),
    ("SetPrecision[2.1, 20]", Pred(lambda g: str(g) == "2.1000000000000000888", "2.1000000000000000888")),
    ("SetPrecision[0.1, 30]", Pred(lambda g: str(g) == "0.100000000000000005551115123126", "binary value of 0.1")),
    ("SetPrecision[Pi + I, 32]", Pred(lambda g: str(g).startswith("3.1415926535897932384626433832795 + 1.0000"), "Pi + I")),
    ("SetPrecision[f[.1], Infinity]", "f[3602879701896397/36028797018963968]"),
    ("SetPrecision[1.5 x, Infinity]", "3 x/2"), ("Precision[SetPrecision[Sqrt[2], 40]]", "40."),
    ("SetPrecision[Sqrt[2], 20]", HP(lambda: mp.sqrt(2), 20)), ("SetPrecision[{1, 2.5}, MachinePrecision]", "{1., 2.5}"),
    ("SetAccuracy[1/3, 30]", HP(lambda: mp.mpf(1) / 3, 29)),
    ("Accuracy[SetAccuracy[1/3, 30]]", Pred(lambda g: abs(float(g) - 30) < 1, "about 30")),
    ("SetAccuracy[x, 20]", "x"),
    # Rationalize (reference page)
    ("Rationalize[6.75]", "27/4"), ("Rationalize[Pi, .01]", "22/7"), ("Rationalize[Pi, .001]", "201/64"),
    ("Rationalize[Exp[Sqrt[2]], 2^-12]", "218/53"), ("Rationalize[1.2 + 6.7 x]", "6/5 + 67 x/10"),
    ("Rationalize[N[Pi]]", 3.141592653589793), ("Rationalize[N[Pi], 0]", "245850922/78256779"),
    ("Table[Rationalize[Pi, 10^-n], {n, 1, 10}]",
     "{22/7, 22/7, 201/64, 333/106, 355/113, 355/113, 75948/24175, 100798/32085, 103993/33102, 312689/99532}"),
    ("Union[Table[Rationalize[N[Pi], 2^-k], {k, 53}]]", PI_RATIONALIZE_UNION),
    ("Rationalize[0.333333333333]", "1/3"), ("Rationalize[-0.75]", "-3/4"), ("Rationalize[2.5 + 1.25 I]", "5/2 + 5 I/4"),
    ("Rationalize[x]", "x"), ("Rationalize[Pi]", "Pi"),
    # Chop
    ("Chop[Exp[N[Range[4] Pi I]]]", "{-1., 1., -1., 1.}"), ("Chop[10.^-12 + 2. I]", "2. I"), ("Chop[2. + 10.^-12 I]", "2."),
    ("Chop[N[Pi] - Rationalize[N[Pi], 10^-12]] === 0", True), ("Chop[N[Pi] - Rationalize[N[Pi], 10^-12], 10^-14] === 0", False),
    ("Chop[1/10^12]", "1/1000000000000"), ("Chop[{1.5, 1.*^-11, a}]", "{1.5, 0, a}"), ("Chop[N[1/10^30, 40]]", "0"),
    ("Chop[0.001, 0.01]", "0"),
    # Round / Floor / Ceiling with steps; complex arguments (reference pages)
    ("Round[226, 10]", "230"), ("Round[-10.3, 3.5]", -10.5), ("Round[2 Pi - E, 5/4]", "15/4"),
    ("{Round[2.5], Round[3.5]}", "{2, 4}"), ("Round[5.37 - 1.3 I]", "5 - I"), ("Round[123.4`45]", "123"),
    ("Round[7.3, 2]", "8"), ("Round[Infinity]", "Infinity"), ("Round[{{1/2, -1}, {5/3, 1/2}}]", "{{0, -1}, {2, 0}}"),
    ("Floor[226, 10]", "220"), ("Floor[-10.3, 3.5]", -10.5), ("Floor[2 Pi - E, 5/4]", "5/2"),
    ("Floor[5.37 - 1.3 I]", "5 - 2 I"), ("Floor[2.5 + 3.7 I]", "2 + 3 I"), ("Floor[-3.7]", "-4"),
    ("Floor[Infinity]", "Infinity"), ("Floor[7545/4660]", "1"), ("Floor[x]", "Floor[x]"),
    ("Ceiling[7.3, 1/2]", "15/2"), ("Ceiling[226, 10]", "230"), ("Ceiling[-3.7]", "-3"), ("Ceiling[2.1 - 0.5 I]", "3"),
    ("Ceiling[-Infinity]", "-Infinity"),
    # Mod / Quotient / QuotientRemainder
    ("Mod[17, 5, 1]", "2"), ("Mod[5, 3, 1]", "2"), ("Mod[10, 5, 1]", "5"), ("Mod[5 + 3 I, 2]", "1 - I"),
    ("Mod[3.14, 2.]", 1.14), ("Mod[Sqrt[28], 3]", "-3 + 2 Sqrt[7]"), ("Mod[Pi, 2]", "Pi - 2"), ("Mod[-7, 3]", "2"),
    ("Mod[7, -3]", "-2"), ("Mod[5/2, 2]", "1/2"), ("Mod[10^10000, 10007]", "6333"), ("Mod[x, 2]", "Mod[x, 2]"),
    ("Mod[7, 0]", Raises()),
    ("Quotient[17, 5]", "3"), ("Quotient[19/3, 5/2]", "2"), ("Quotient[4.56, 2.5]", "1"), ("Quotient[-7, 3]", "-3"),
    ("Quotient[17, 5, 1]", "3"), ("Quotient[E^E^E, Pi]", "1214122"), ("Quotient[7, 0]", Raises()),
    ("QuotientRemainder[17, 5]", "{3, 2}"), ("QuotientRemainder[-17, 5]", "{-4, 3}"),
    ("QuotientRemainder[17, -5]", "{-4, -3}"), ("QuotientRemainder[7.5, 2]", "{3, 1.5}"),
    # IntegerPart / FractionalPart (reference pages)
    ("IntegerPart[2.4]", "2"), ("IntegerPart[-2.4]", "-2"), ("IntegerPart[-5/4]", "-1"), ("IntegerPart[Pi + E]", "5"),
    ("IntegerPart[78/47 - 4.2 I]", "1 - 4 I"), ("IntegerPart[Infinity]", "Infinity"),
    ("IntegerPart[{2.4, 2.5, 3.0}]", "{2, 2, 3}"), ("IntegerPart[x]", "IntegerPart[x]"),
    ("FractionalPart[2.4]", 0.4), ("FractionalPart[-2.4]", -0.4), ("FractionalPart[-5/4]", "-1/4"),
    ("N[FractionalPart[Pi + E]]", 0.8598744820488384), ("FractionalPart[235/47 + 5.3 I]", 0.3j),
    ("N[FractionalPart[4568/4456], 50]", HP(lambda: mp.mpf(4568) / 4456 - 1, 50)),
    ("FractionalPart[0]", "0"), ("FractionalPart[Pi]", "Pi - 3"), ("FractionalPart[{1/7, 5/4}]", "{1/7, 1/4}"),
    # RealDigits (reference page)
    ("RealDigits[123.55555]", "{{1, 2, 3, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0}, 3}"),
    ("RealDigits[0.000012355555]", "{{1, 2, 3, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0}, -4}"),
    ("RealDigits[Pi, 10, 25]", "{{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3}, 1}"),
    ("RealDigits[19/7, 10, 25]", "{{2, 7, 1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5, 7, 1, 4, 2, 8, 5}, 1}"),
    ("RealDigits[19/7]", "{{2, {7, 1, 4, 2, 8, 5}}, 1}"),
    ("RealDigits[1.234, 2, 15]", "{{1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1}, 1}"),
    ("RealDigits[1/7]", "{{{1, 4, 2, 8, 5, 7}}, 0}"), ("RealDigits[1/4]", "{{2, 5}, 0}"),
    ("RealDigits[1/12]", "{{8, {3}}, -1}"), ("RealDigits[123]", "{{1, 2, 3}, 3}"), ("RealDigits[-3/8]", "{{3, 7, 5}, 0}"),
    ("RealDigits[N[Pi]]", "{{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3}, 1}"),
    ("Last[First[RealDigits[N[Pi], 10, 20]]]", "Indeterminate"), ("Length[First[RealDigits[N[Pi], 2]]]", "53"),
    ("Length[First[RealDigits[N[Pi, 30]]]]", "30"), ("RealDigits[Pi, 10, 5, -3]", "{{1, 5, 9, 2, 6}, -2}"),
    ("RealDigits[E, 10, 10, 2]", "{{0, 0, 2, 7, 1, 8, 2, 8, 1, 8}, 3}"), ("RealDigits[255/256, 16]", "{{15, 15}, 0}"),
    ("RealDigits[Sqrt[2], 2, 10]", "{{1, 0, 1, 1, 0, 1, 0, 1, 0, 0}, 1}"), ("RealDigits[Pi]", Raises()),
    ("FromDigits[RealDigits[1/7]]", "1/7"), ("FromDigits[RealDigits[19/7]]", "19/7"), ("FromDigits[{{8, {3}}, -1}]", "1/12"),
    ("FromDigits[{{1, 2, 5}, 0}]", "1/8"), ("FromDigits[RealDigits[1/3, 2], 2]", "1/3"),
    # Rescale / Clip / Unitize / RealAbs
    ("Rescale[3, {1, 5}]", "1/2"), ("Rescale[x, {a, b}]", "(x - a)/(b - a)"), ("Rescale[{1, 2, 3, 4}]", "{0, 1/3, 2/3, 1}"),
    ("Rescale[2.5, {0, 10}, {-1, 1}]", -0.5), ("Rescale[{-1, 0, 3}]", "{0, 1/4, 1}"), ("Rescale[1, {2, 2}]", Raises()),
    ("Clip[0.5]", 0.5), ("Clip[2]", "1"), ("Clip[-3.5]", "-1"), ("Clip[{-2, 0, 2}]", "{-1, 0, 1}"), ("Clip[x]", "Clip[x]"),
    ("Clip[5, {0, 3}]", "3"), ("Clip[-1, {0, 3}, {a, b}]", "a"), ("Clip[7, {0, 3}, {a, b}]", "b"),
    ("Clip[1.5, {0, 3}, {a, b}]", 1.5),
    ("Unitize[{0, 1.5, 0., -2, 0}]", "{0, 1, 0, 1, 0}"), ("Unitize[x]", "Unitize[x]"), ("Unitize[0.001, 0.01]", "0"),
    ("Unitize[0.1, 0.01]", "1"),
    ("RealAbs[-3]", "3"), ("RealAbs[-2.5]", 2.5), ("RealAbs[x]", "RealAbs[x]"), ("RealAbs[3 + 4 I]", "RealAbs[3 + 4 I]"),
    ("RealAbs[-Pi]", "Pi"),
    # Sign (complex), Numerator / Denominator, number tests, Arg / ReIm / AbsArg / Re / Im
    ("Sign[3 + 4 I]", "3/5 + 4 I/5"), ("Sign[-2.5]", "-1"), ("Sign[1. + 1. I]", complex(0.7071067811865475, 0.7071067811865475)),
    ("Sign[0]", "0"), ("Sign[x]", "Sign[x]"),
    ("Numerator[3/4]", "3"), ("Numerator[x^2 y^-3]", "x^2"), ("Numerator[a/b + c/d]", "a/b + c/d"),
    ("Numerator[(1 + I)/2]", "1 + I"), ("Numerator[(x - 1)(x - 2)/(x - 3)^2]", "(x - 1)(x - 2)"),
    ("Denominator[3/4]", "4"), ("Denominator[x/y^2]", "y^2"), ("Denominator[5]", "1"), ("Denominator[1/2 + I/3]", "6"),
    ("Denominator[(x - 1)(x - 2)/(x - 3)^2]", "(x - 3)^2"), ("Denominator[2.5]", "1"),
    ("ExactNumberQ[1/2]", True), ("ExactNumberQ[1.5]", False), ("ExactNumberQ[Pi]", False), ("ExactNumberQ[1 + I]", True),
    ("ExactNumberQ[1. + I]", False),
    ("InexactNumberQ[1. + I]", True), ("InexactNumberQ[N[Pi, 30]]", True), ("InexactNumberQ[3]", False),
    ("MachineNumberQ[1.5]", True), ("MachineNumberQ[N[Pi, 30]]", False), ("MachineNumberQ[2]", False),
    ("MachineNumberQ[1.5 + 2. I]", True),
    ("Arg[1 + I]", "Pi/4"), ("Arg[-1]", "Pi"), ("Arg[0]", "0"), ("Arg[-2.5]", 3.141592653589793), ("Arg[x]", "Arg[x]"),
    ("Arg[-I]", "-Pi/2"),
    ("ReIm[3 + 4 I]", "{3, 4}"), ("ReIm[2.5]", "{2.5, 0}"), ("ReIm[{I, 2}]", "{{0, 1}, {2, 0}}"),
    ("AbsArg[-2]", "{2, Pi}"), ("AbsArg[1 + I]", "{Sqrt[2], Pi/4}"), ("AbsArg[2.]", "{2., 0}"),
    ("Re[3 + 4 I]", "3"), ("Re[Exp[I Pi/3]]", "1/2"), ("Re[x]", "Re[x]"),
    ("Im[2]", "0"), ("Im[2.5 - 1.5 I]", -1.5), ("Im[Exp[I Pi/3]]", "Sqrt[3]/2"),
]

CASES = NUMERICS


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_batch2(case):
    check(case)
