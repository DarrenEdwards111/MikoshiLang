"""Differential tests for coverage batch 2 (numerics and special functions).

Expected values are Wolfram Language results (examples on the reference pages
reference.wolfram.com/language/ref/<Name>.html, or values derived from their Details sections) or independent
references computed here with mpmath at higher precision.  ``HP`` checks an arbitrary-precision result against
mpmath to the requested number of digits and also checks the result's precision.
"""
import mpmath as mp
import pytest
import sympy as sp

from _harness import Pred, Raises, check, ids, to_complex

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
    return Pred(lambda g: abs(to_complex(g) - complex(want)) <= tol * max(1, abs(complex(want))), f"~{want}")


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

f = mp.mpf
JAC = {"SN": "sn", "CN": "cn", "DN": "dn", "SC": "sc", "SD": "sd", "CD": "cd", "CS": "cs", "DS": "ds", "DC": "dc",
       "NS": "ns", "NC": "nc", "ND": "nd"}
JAC_M0 = {"SN": "Sin[u]", "CN": "Cos[u]", "DN": "1", "SC": "Tan[u]", "SD": "Sin[u]", "CD": "Cos[u]", "CS": "Cot[u]",
          "DS": "Csc[u]", "DC": "Sec[u]", "NS": "Csc[u]", "NC": "Sec[u]", "ND": "1"}
JAC_U0 = {"SN": "0", "CN": "1", "DN": "1", "SC": "0", "SD": "0", "CD": "1", "DC": "1", "NC": "1", "ND": "1"}


def _jac_cases():
    out = []
    for k, e in JAC.items():
        out.append((f"N[Jacobi{k}[7/10, 3/10], 30]", HP(lambda e=e: mp.ellipfun(e, f(7) / 10, m=f(3) / 10), 30)))
        out.append((f"Jacobi{k}[u, 0]", JAC_M0[k]))
        if k in JAC_U0:
            out.append((f"Jacobi{k}[0, m]", JAC_U0[k]))
        else:
            out.append((f"Jacobi{k}[0.3, 1.5]", Near(complex(mp.ellipfun(e, 0.3, m=1.5)), 1e-10)))
    return out


def _invjac_cases():
    out = []
    for k, e in [("SN", "sn"), ("CN", "cn"), ("DN", "dn"), ("SC", "sc"), ("CS", "cs"), ("NS", "ns"), ("NC", "nc"),
                 ("SD", "sd"), ("CD", "cd")]:
        v = {"DN": f("0.9"), "NS": f(2), "NC": f(2), "CS": f(3)}.get(k, f("0.4"))
        vs = str(v)
        out.append((f"Jacobi{k}[InverseJacobi{k}[{vs}, 0.3], 0.3]", Near(complex(v), 1e-10)))
        out.append((f"N[Jacobi{k}[InverseJacobi{k}[{vs}`30, 3/10], 3/10], 25]", HP(v, 25)))
        out.append((f"InverseJacobi{k}[{vs}, 0.3] > 0", True))
    return out


SPECIAL = [
    # GammaRegularized
    ("GammaRegularized[2, 0]", "1"), ("GammaRegularized[a, Infinity]", "0"), ("GammaRegularized[a, 0]", "1"),
    ("GammaRegularized[3, 2]", "5 E^-2"),
    ("N[GammaRegularized[1/3, 2], 30]", HP(lambda: mp.gammainc(f(1) / 3, 2, regularized=True), 30)),
    ("GammaRegularized[2.5, 1., 3.]", Near(complex(mp.gammainc(2.5, 1, 3, regularized=True)))),
    # incomplete Beta / BetaRegularized
    ("Beta[1/2, 2, 3]", "11/192"), ("N[Beta[3/10, 5/2, 3/2], 30]", HP(lambda: mp.betainc(f(5) / 2, f(3) / 2, 0, f(3) / 10), 30)), ("MachineNumberQ[N[Beta[0.3`30, 2.5, 1.5], 30]]", True), ("Precision[N[Beta[0.3`20, 5/2, 3/2], 30]]", "20."),
    ("Beta[0.2, 0.7, 2, 3]", Near(complex(mp.betainc(2, 3, 0.2, 0.7)))), ("Beta[1, 2, 3]", "1/12"),
    ("BetaRegularized[1/2, 2, 3]", "11/16"), ("BetaRegularized[1, 2, 3]", "1"), ("BetaRegularized[0, 2, 3]", "0"),
    ("N[BetaRegularized[1/3, 1/2, 5/2], 30]", HP(lambda: mp.betainc(f(1) / 2, f(5) / 2, 0, f(1) / 3, regularized=True), 30)),
    ("BetaRegularized[0.1, 0.4, 2, 3]", Near(complex(mp.betainc(2, 3, 0.1, 0.4, regularized=True)))),
    # Erf[z0, z1], InverseErf, InverseErfc
    ("Erf[0, x]", "Erf[x]"), ("Erf[1., 2.]", Near(complex(mp.erf(2) - mp.erf(1)))),
    ("N[Erf[1/2, 3/2], 30]", HP(lambda: mp.erf(f(3) / 2) - mp.erf(f(1) / 2), 30)),
    ("InverseErf[0]", "0"), ("InverseErf[1]", "Infinity"), ("InverseErf[-1]", "-Infinity"),
    ("InverseErf[0.5]", Near(0.4769362762044699)), ("N[InverseErf[3/10], 30]", HP(lambda: mp.erfinv(f(3) / 10), 30)),
    ("Erf[InverseErf[0.999999]]", Near(0.999999, 1e-12)), ("InverseErf[2.]", Raises()),
    ("InverseErfc[1]", "0"), ("InverseErfc[0]", "Infinity"), ("InverseErfc[2]", "-Infinity"),
    ("InverseErfc[0.3]", Near(complex(mp.erfinv(0.7)))), ("Erfc[InverseErfc[1.*^-20]]", Near(1e-20, 1e-10)),
    ("N[InverseErfc[1/10], 30]", HP(lambda: mp.erfinv(1 - f(1) / 10), 30)),
    # hyperbolic and Fresnel integrals
    ("SinhIntegral[0]", "0"), ("SinhIntegral[1.]", Near(complex(mp.shi(1)))), ("SinhIntegral[-x]", "-SinhIntegral[x]"),
    ("N[SinhIntegral[2], 30]", HP(lambda: mp.shi(2), 30)),
    ("CoshIntegral[0]", "-Infinity"), ("CoshIntegral[1.]", Near(complex(mp.chi(1)))),
    ("N[CoshIntegral[3], 30]", HP(lambda: mp.chi(3), 30)), ("CoshIntegral[-1.]", Near(complex(mp.chi(-1)))),
    ("FresnelS[0]", "0"), ("FresnelS[Infinity]", "1/2"), ("FresnelS[1.]", Near(complex(mp.fresnels(1)))),
    ("N[FresnelS[3/2], 30]", HP(lambda: mp.fresnels(f(3) / 2), 30)),
    ("FresnelC[0]", "0"), ("FresnelC[Infinity]", "1/2"), ("FresnelC[1.]", Near(complex(mp.fresnelc(1)))),
    ("N[FresnelC[3/2], 30]", HP(lambda: mp.fresnelc(f(3) / 2), 30)),
    # spherical Bessel / Hankel, Airy derivatives
    ("SphericalBesselY[0, x]", "-Cos[x]/x"), ("SphericalBesselY[1, 2.]", Near(complex(-mp.cos(2) / 4 - mp.sin(2) / 2))),
    ("N[SphericalBesselY[2, 3], 30]", HP(lambda: mp.sqrt(mp.pi / 6) * mp.bessely(f(5) / 2, 3), 30)),
    ("SphericalHankelH1[0, x]", "-I E^(I x)/x"), ("SphericalHankelH1[1, 2.]",
     Near(complex(mp.sqrt(mp.pi / 4) * mp.hankel1(1.5, 2)))),
    ("N[SphericalHankelH1[2, 1/2], 20]", Near(complex(mp.sqrt(mp.pi) * mp.hankel1(2.5, 0.5)), 1e-15)),
    ("SphericalHankelH2[0, x]", "I E^(-I x)/x"), ("SphericalHankelH2[1, 2.]",
     Near(complex(mp.sqrt(mp.pi / 4) * mp.hankel2(1.5, 2)))),
    ("N[SphericalHankelH2[2, 1/2], 20]", Near(complex(mp.sqrt(mp.pi) * mp.hankel2(2.5, 0.5)), 1e-15)),
    ("AiryAiPrime[0]", "-1/(3^(1/3) Gamma[1/3])"), ("AiryAiPrime[1.]", Near(complex(mp.airyai(1, derivative=1)))),
    ("N[AiryAiPrime[-2], 30]", HP(lambda: mp.airyai(-2, derivative=1), 30)),
    ("AiryBiPrime[0]", "3^(1/6)/Gamma[1/3]"), ("AiryBiPrime[1.]", Near(complex(mp.airybi(1, derivative=1)))),
    ("N[AiryBiPrime[-2], 30]", HP(lambda: mp.airybi(-2, derivative=1), 30)),
    # generalised incomplete Gamma
    ("Gamma[2, 0, Infinity]", "1"), ("Gamma[1, 1, 2]", "E^-1 - E^-2"), ("Gamma[2.5, 1., 3.]", Near(complex(mp.gammainc(2.5, 1, 3)))),
    # PolyLog
    ("PolyLog[2, 1]", "Pi^2/6"), ("PolyLog[1, z]", "-Log[1 - z]"), ("PolyLog[0, z]", "z/(1 - z)"),
    ("PolyLog[-1, z]", "z/(1 - z)^2"), ("PolyLog[2, 1/2]", "Pi^2/12 - Log[2]^2/2"), ("PolyLog[2, -1]", "-Pi^2/12"),
    ("PolyLog[3, 1/2]", "(-2 Pi^2 Log[2] + 4 Log[2]^3 + 21 Zeta[3])/24"), ("PolyLog[2, .9]", Near(1.299714723004959)),
    ("PolyLog[0, 6.]", Near(-1.2)), ("PolyLog[2, 3.]", Near(complex(mp.polylog(2, 3)))),
    ("N[PolyLog[5/2, 1/3], 30]", HP(lambda: mp.polylog(f(5) / 2, f(1) / 3), 30)),
    # Hurwitz zeta, Zeta[s, a] (Wolfram convention), LerchPhi, StieltjesGamma
    ("Zeta[2, 1/2]", "Pi^2/2"), ("Zeta[2, 1]", "Pi^2/6"), ("N[Zeta[3, 1/3], 30]", HP(lambda: mp.zeta(3, f(1) / 3), 30)),
    ("Zeta[2, -1.5]", Near(complex(mp.zeta(2, 0.5) + 1 / 2.25 + 4))),
    ("HurwitzZeta[2, 1]", "Pi^2/6"), ("HurwitzZeta[3, 2.5]", Near(complex(mp.zeta(3, 2.5)))),
    ("N[HurwitzZeta[3/2, 1/4], 30]", HP(lambda: mp.zeta(f(3) / 2, f(1) / 4), 30)),
    ("LerchPhi[2, 3, -1.5]", Near(51.9819 - 2.1346j, 1e-5)), ("LerchPhi[7., 1, 2]",
     Near(-0.179424 - 0.0641141j, 1e-5)), ("LerchPhi[0.7, 1, 2]", Near(complex(mp.lerchphi(0.7, 1, 2)))),
    ("LerchPhi[49., 0, 2]", Near(-1 / 48)), ("LerchPhi[0.5, 2, -2.5]", Near(complex(mp.mpf(0.5) ** 3 / 0.25 * 0 + sum(0.5 ** k / ((k - 2.5) ** 2) for k in range(3)) + 0.5 ** 3 * mp.lerchphi(0.5, 2, 0.5)))),
    ("N[LerchPhi[1/2, 2, 1/3], 30]", HP(lambda: mp.lerchphi(f(1) / 2, 2, f(1) / 3), 30)),
    ("HurwitzLerchPhi[0.5, 2, 1.5]", Near(complex(mp.lerchphi(0.5, 2, 1.5)))), ("HurwitzLerchPhi[-0.5, 1, 2]", Near(complex(mp.lerchphi(-0.5, 1, 2)))),
    ("N[HurwitzLerchPhi[1/3, 3, 2], 30]", HP(lambda: mp.lerchphi(f(1) / 3, 3, 2), 30)),
    ("StieltjesGamma[0]", "EulerGamma"), ("StieltjesGamma[1.]", Near(complex(mp.stieltjes(1)))),
    ("N[StieltjesGamma[2], 30]", HP(lambda: mp.stieltjes(2), 30)), ("StieltjesGamma[0, 1/2]", "EulerGamma + 2 Log[2]"),
    ("StieltjesGamma[1, 0.5]", Near(complex(mp.stieltjes(1, 0.5)))),
    # Tricomi U, generalised and regularised hypergeometric functions, MeijerG
    ("HypergeometricU[3, 2, 1.]", Near(0.105479, 1e-5)), ("HypergeometricU[0, b, z]", "1"),
    ("N[HypergeometricU[1/3, 2/3, 2], 30]", HP(f("0.73215920249551738412474815958614"), 30)),
    ("HypergeometricU[3, 1.2, 3. I]", Near(complex(mp.hyperu(3, 1.2, 3j)))),
    ("HypergeometricPFQ[{}, {}, z]", "E^z"), ("HypergeometricPFQ[{1, 1}, {2}, -z]", "Log[1 + z]/z"),
    ("HypergeometricPFQ[{1, 2, 3}, {4, 5}, 0.5]", Near(complex(mp.hyp3f2(1, 2, 3, 4, 5, 0.5)))),
    ("N[HypergeometricPFQ[{1/2, 1/3}, {1/4}, 1/5], 30]", HP(lambda: mp.hyp2f1(f(1) / 2, f(1) / 3, f(1) / 4, f(1) / 5), 30)),
    ("HypergeometricPFQ[{-2, 1}, {3}, z]", "1 - 2 z/3 + z^2/6"),
    ("HypergeometricPFQRegularized[{1}, {-1}, 0.5]", Near(complex(0.25 * mp.exp(0.5)))),
    ("HypergeometricPFQRegularized[{1, 2}, {3, 4}, 0.3]", Near(complex(mp.hyper([1, 2], [3, 4], 0.3) / 12))),
    ("HypergeometricPFQRegularized[{1}, {1/2}, 0]", "1/Sqrt[Pi]"),
    ("Hypergeometric0F1Regularized[1/2, 0.4]", Near(complex(mp.hyp0f1(0.5, 0.4) / mp.gamma(0.5)))),
    ("Hypergeometric0F1Regularized[-1, 1.]", Near(complex(mp.besseli(2, 2)))),
    ("N[Hypergeometric0F1Regularized[3, 2], 30]", HP(lambda: mp.hyp0f1(3, 2) / 2, 30)),
    ("Hypergeometric1F1Regularized[1, 2, 0.5]", Near(complex(mp.hyp1f1(1, 2, 0.5)))),
    ("Hypergeometric1F1Regularized[2, -1, 0.5]", Near(complex(0.75 * mp.hyp1f1(4, 3, 0.5)))),
    ("N[Hypergeometric1F1Regularized[1/3, 3/2, 2], 30]", HP(lambda: mp.hyp1f1(f(1) / 3, f(3) / 2, 2) / mp.gamma(f(3) / 2), 30)),
    ("Hypergeometric2F1Regularized[1, 2, 3, 0.5]", Near(complex(mp.hyp2f1(1, 2, 3, 0.5) / 2))),
    ("Hypergeometric2F1Regularized[1, 2, -1, 0.25]",
     Near(complex(1 * 2 * 2 * 3 * 0.25 ** 2 / 2 * mp.hyp2f1(3, 4, 3, 0.25)))),
    ("N[Hypergeometric2F1Regularized[1/2, 1/3, 5/2, 1/4], 30]",
     HP(lambda: mp.hyp2f1(f(1) / 2, f(1) / 3, f(5) / 2, f(1) / 4) / mp.gamma(f(5) / 2), 30)),
    ("N[MeijerG[{{}, {2}}, {{1/2, 3/2}, {}}, 3], 30]", HP(f("0.041591159875396983670330390749778758626529715159039"), 30)),
    ("MeijerG[{{1}, {2}}, {{2.3, 1.5}, {4}}, 3]", Near(-1.82257421144959, 1e-10)),
    ("MeijerG[{{}, {}}, {{0}, {}}, 2.]", Near(complex(mp.exp(-2)))),
    # BarnesG, Pochhammer, FactorialPower
    ("BarnesG[10]", "5056584744960000"), ("BarnesG[3]", "1"), ("BarnesG[0]", "0"), ("BarnesG[-2]", "0"),
    ("BarnesG[0.5]", Near(0.603244281209446)), ("N[BarnesG[1/7], 20]", HP(f("0.15978438500356974041"), 19)),
    ("Pochhammer[a, 0]", "1"), ("Pochhammer[2, 3]", "24"), ("Pochhammer[x, 3]", "x (1 + x) (2 + x)"),
    ("Pochhammer[1/2, -1]", "-2"), ("Pochhammer[2.5, 1.5]", Near(complex(mp.rf(2.5, 1.5)))),
    ("N[Pochhammer[1/3, 1/2], 30]", HP(lambda: mp.rf(f(1) / 3, f(1) / 2), 30)),
    ("FactorialPower[5, 2]", "20"), ("FactorialPower[x, 3]", "FactorialPower[x, 3]"), ("FactorialPower[10, 3, 2]", "480"),
    ("FactorialPower[5.5, 2.5]", Near(complex(mp.gamma(6.5) / mp.gamma(4)))), ("FactorialPower[5, 0]", "1"),
    # Weierstrass elliptic functions
    ("WeierstrassPPrime[0.4, {1, 2}]^2 - (4 WeierstrassP[0.4, {1, 2}]^3 - WeierstrassP[0.4, {1, 2}] - 2) // Chop", "0"),
    ("Chop[WeierstrassPPrime[0.7, {2, -1}]^2 - (4 WeierstrassP[0.7, {2, -1}]^3 - 2 WeierstrassP[0.7, {2, -1}] + 1), 10^-9]", "0"),
    ("Abs[WeierstrassP[0.01, {1, 2}] - 10^4] < 0.1", True), ("WeierstrassP[0.3, {0, 0}]", Raises()),
    ("Abs[WeierstrassPPrime[0.01, {1, 2}] + 2*10^6] < 10", True),
    ("WeierstrassPPrime[0.3, {0, 0}]", Raises()),
] + _jac_cases() + _invjac_cases()

NUMBER_THEORY = [
    ("LucasL[3, x]", "x^3 + 3 x"), ("LucasL[0, x]", "2"), ("LucasL[5, 2]", "82"), ("LucasL[4, 1]", "7"),
    ("HarmonicNumber[1/2]", "2 - 2 Log[2]"), ("HarmonicNumber[3/2]", "8/3 - 2 Log[2]"), ("HarmonicNumber[5]", "137/60"),
    ("HarmonicNumber[3, 2]", "49/36"),
    ("CatalanNumber[1/2]", "8/(3 Pi)"), ("CatalanNumber[4]", "14"), ("CatalanNumber[n]", "CatalanNumber[n]"),
    ("PowerMod[3, 1/2, 11]", "5"), ("PowerMod[2, 1/2, 7]", "3"), ("PowerMod[3, 1/2, 7]", "PowerMod[3, 1/2, 7]"),
    ("PowerMod[2, 3/2, 7]", "6"), ("PowerMod[3, -1, 7]", "5"),
    ("MultiplicativeOrder[3, 7, {2, 6}]", "2"), ("MultiplicativeOrder[2, 7, {3}]", "MultiplicativeOrder[2, 7, {3}]"),
    ("MultiplicativeOrder[10, 41, {1}]", "5"), ("MultiplicativeOrder[{2, 3}, 7]", "{3, 6}"),
    ("PrimitiveRoot[7, 4]", "5"), ("PrimitiveRoot[23, 6]", "7"), ("PrimitiveRoot[8, 2]", "PrimitiveRoot[8, 2]"),
    ("MangoldtLambda[8]", "Log[2]"), ("MangoldtLambda[6]", "0"), ("MangoldtLambda[1]", "0"), ("MangoldtLambda[49]", "Log[7]"),
    ("MangoldtLambda[0]", Raises()),
    ("LiouvilleLambda[12]", "-1"), ("LiouvilleLambda[30]", "-1"), ("LiouvilleLambda[1]", "1"), ("LiouvilleLambda[36]", "1"),
    ("Table[LiouvilleLambda[n], {n, 10}]", "{1, -1, -1, 1, -1, 1, -1, -1, 1, 1}"),
    ("DivisorSum[20, #^2 &]", "546"), ("DivisorSum[20, # &, # < 10 &]", "12"), ("DivisorSum[1, f]", "f[1]"),
    ("DivisorSum[6, f]", "f[1] + f[2] + f[3] + f[6]"), ("DivisorSum[12, MoebiusMu]", "0"),
    ("PrimeQ[RandomPrime[100]]", True), ("RandomPrime[{24, 28}]", Raises()), ("Length[RandomPrime[{10, 20}, 5]]", "5"),
    ("Union[Table[RandomPrime[{2, 3}], {200}]]", "{2, 3}"), ("Dimensions[RandomPrime[50, {2, 3}]]", "{2, 3}"),
    ("RandomPrime[{14, 16}]", Raises()),
    # corrected special functions
    ("ExpIntegralE[2, 0]", "1"), ("ExpIntegralE[3, 0]", "1/2"), ("ExpIntegralE[1, 0]", "ComplexInfinity"),
    ("ExpIntegralE[1/2, 0]", "ComplexInfinity"), ("N[ExpIntegralE[1/2, 2], 30]", HP(lambda: mp.expint(f(1) / 2, 2), 30)),
    ("CosIntegral[0]", "-Infinity"), ("CosIntegral[-1.]", Near(complex(mp.ci(1)) + 3.141592653589793j)),
    ("N[CosIntegral[2], 30]", HP(lambda: mp.ci(2), 30)),
    ("LaguerreL[1/2, 0.3]", Near(complex(mp.laguerre(0.5, 0, 0.3)))), ("LaguerreL[1/2, 2, 0.3]", Near(complex(mp.laguerre(0.5, 2, 0.3)))),
    ("LaguerreL[3, x]", "1 - 3 x + 3 x^2/2 - x^3/6"), ("LaguerreL[2, a, x]", "1 + 3 a/2 + a^2/2 - 2 x + x^2/2 - a x"),
    ("N[LaguerreL[1/3, 1/2, 2], 30]", HP(lambda: mp.laguerre(f(1) / 3, f(1) / 2, 2), 30)),
    ("LegendreQ[0, 2.]", Near(0.5493061443340549 - 1.5707963267948966j)), ("LegendreQ[2, 0.5]", Near(-0.8186632680417568)),
    ("LegendreQ[1, x]", "x (Log[1 + x] - Log[1 - x])/2 - 1"), ("LegendreQ[2, 1, 0.5]", Near(complex(mp.legenq(2, 1, 0.5, type=2)))),
    ("LegendreQ[1, 1, 0.3]", Near(complex(mp.legenq(1, 1, 0.3, type=2)))), ("LegendreQ[1, 0, x]", "x (Log[1 + x] - Log[1 - x])/2 - 1"),
    ("LegendreQ[0, 0, 3, 2.]", Near(0.5493061443340549)), ("LegendreQ[1, 1, 3, 2.5]", Near(complex(mp.legenq(1, 1, 2.5, type=3)))),
    ("LegendreQ[1, 0, 5, x]", Raises()),
    ("LegendreP[2, 1, 3, 2.]", Near(10.392304845413264)), ("LegendreP[2, 1, 2, 0.5]", Near(-1.299038105676658)),
    ("LegendreP[2, 1, 7, x]", Raises()), ("LegendreP[1/2, 0.3]", Near(complex(mp.legenp(0.5, 0, 0.3)))),
    ("LegendreP[2, 1, x]", "-3 x Sqrt[1 - x^2]"), ("LegendreP[3, x]", "-3 x/2 + 5 x^3/2"),
]

NUMCALC = [
    # NIntegrate (Wolfram reference page values; mpmath references)
    ("NIntegrate[Sin[Sin[x]], {x, 0, 2}]", Near(1.24706, 1e-5)), ("NIntegrate[x^x, {x, 1, 5}]", Near(1241.03, 1e-5)),
    ("NIntegrate[Sin[x], {x, 0, Pi}]", Near(2.0)), ("NIntegrate[Exp[-x^2], {x, -Infinity, Infinity}]", Near(1.7724538509055159)),
    ("NIntegrate[1/Sqrt[x], {x, 0, 1}]", Near(2.0, 1e-10)), ("NIntegrate[x y, {x, 0, 1}, {y, 0, x}]", Near(0.125)),
    ("NIntegrate[1/Sqrt[Abs[x]], {x, -1, 0, 1}]", Near(4.0, 1e-10)),
    ("NIntegrate[Exp[I x], {x, 0, 1}]", Near(complex(mp.sin(1)) + 1j * complex(1 - mp.cos(1)))),
    ("NIntegrate[Exp[-x^2], {x, 0, 1}, WorkingPrecision -> 30]", HP(lambda: mp.sqrt(mp.pi) / 2 * mp.erf(1), 28)),
    ("NIntegrate[Exp[-x] Log[x], {x, 0, Infinity}]", Near(-0.5772156649015329, 1e-10)),
    ("NIntegrate[1/x, {x, 0, 1}]", Raises()), ("MachineNumberQ[NIntegrate[x^2, {x, 0, 1}]]", True),
    ("NIntegrate[x^2, {x, 0, 1}, Method -> \"GaussKronrod\"]", Near(1 / 3)),
    # NSum / NProduct
    ("NSum[1/n^2, {n, 1, Infinity}]", Near(1.6449340668482264)), ("NSum[(-1)^n/n, {n, 1, Infinity}]", Near(-0.6931471805599453)),
    ("NSum[1/n^3, {n, 1, Infinity}, WorkingPrecision -> 30]", HP(lambda: mp.zeta(3), 28)),
    ("NSum[1/n^2, {n, 1, Infinity, 2}]", Near(1.2337005501361697)), ("NSum[k, {k, 1, 100}]", Near(5050.0)),
    ("NSum[1/(i^2 j^2), {i, 1, Infinity}, {j, 1, 3}]", Near(float(mp.zeta(2) * (1 + mp.mpf(1) / 4 + mp.mpf(1) / 9)), 1e-9)),
    ("NSum[1/n, {n, 1, Infinity}]", Raises()),
    ("NProduct[1 - 1/n^2, {n, 2, Infinity}]", Near(0.5, 1e-10)), ("NProduct[k, {k, 1, 10}]", Near(3628800.0)),
    ("NProduct[1 + 1/n^2, {n, 1, Infinity}]", Near(float(mp.sinh(mp.pi) / mp.pi), 1e-9)),
    ("NProduct[1 + 1/2^k, {k, 1, Infinity}, WorkingPrecision -> 25]", HP(lambda: mp.nprod(lambda k: 1 + mp.mpf(2) ** -k, [1, mp.inf]), 20)),
    # NSolve / NRoots
    ("NSolve[x^5 - 2 x + 3 == 0, x]", Pred(lambda g: len(g.args) == 5 and abs(complex(g.args[0].args[0].args[1]) + 1.42361) < 1e-5, "5 roots, first -1.42361")),
    ("NSolve[x^5 - 2 x + 3 == 0, x, Reals]", Pred(lambda g: len(g.args) == 1 and abs(g.args[0].args[0].args[1] + 1.4236058485523317) < 1e-12, "{{x -> -1.42361}}")),
    ("NSolve[x^3 - 2 x + 1 == 0, x]", "{{x -> -1.618033988749895}, {x -> 0.6180339887498949}, {x -> 1.}}"),
    ("Length[NSolve[x^2 == 0, x]]", "2"), ("NSolve[x^4 == 1, x, Reals]", "{{x -> -1.}, {x -> 1.}}"),
    ("Length[NSolve[{x^2 + y^3 == 1, 2 x + 3 y == 4}, {x, y}]]", "3"),
    ("NSolve[{x^2 + y^3 == 1, 2 x + 3 y == 4}, {x, y}, Reals]", Pred(lambda g: len(g.args) == 1 and abs(g.args[0].args[0].args[1] - 7.93641) < 1e-5, "x -> 7.93641")),
    ("NSolve[x^2 - 2 == 0, x, WorkingPrecision -> 30]", Pred(lambda g: HP(mp.sqrt(2), 28).f(g.args[1].args[0].args[1]), "Sqrt[2] to 30 digits")),
    ("NSolve[Sin[x] == 0, x]", Raises()),
    ("NRoots[x^3 - 1 == 0, x]", Pred(lambda g: g.head == "Or" and len(g.args) == 3, "three roots")),
    ("NRoots[x^2 - 3 x + 2 == 0, x]", "x == 1. || x == 2."), ("NRoots[x - 5 == 0, x]", "x == 5."), ("NRoots[Sin[x] == 0, x]", Raises()),
    # FindRoot
    ("FindRoot[Sin[x] + Exp[x], {x, 0}]", "{x -> -0.5885327439818611}"),
    ("FindRoot[Cos[x] == x, {x, 1}, WorkingPrecision -> 30]", Pred(lambda g: HP(mp.findroot(lambda t: mp.cos(t) - t, 1), 28).f(g.args[0].args[1]), "30 digits")),
    ("FindRoot[{x^2 + y^2 == 1, y == x^3}, {{x, 1}, {y, 1}}]", Pred(lambda g: abs(g.args[0].args[1] - 0.826031357654187) < 1e-12, "x -> 0.826031")),
    ("FindRoot[x^2 + 1, {x, I}]", Pred(lambda g: abs(complex(to_complex(g.args[0].args[1])) - 1j) < 1e-12, "x -> I")),
    ("FindRoot[Cos[x] - x, {x, 0, 1}]", Pred(lambda g: abs(g.args[0].args[1] - 0.7390851332151607) < 1e-12, "secant")),
    ("FindRoot[x^2 + 1, {x, 1}]", Raises()),
    # FindMaximum / NMinimize / NMaximize
    ("FindMaximum[Sin[x], {x, 1}]", Pred(lambda g: abs(g.args[0] - 1) < 1e-12 and abs(g.args[1].args[0].args[1] - 1.5707963267948966) < 1e-7, "{1., {x -> 1.5708}}")),
    ("FindMaximum[-(x - 2)^2 + 3, {x, 0}]", Pred(lambda g: abs(g.args[0] - 3) < 1e-12, "3")),
    ("FindMaximum[x Exp[-x^2 - y^2], {{x, 1}, {y, 1}}]", Pred(lambda g: abs(g.args[0] - float(1 / mp.sqrt(2 * mp.e))) < 1e-10, "1/Sqrt[2 E]")),
    ("NMinimize[x^4 - 3 x^2 - x, x]", Pred(lambda g: abs(g.args[0] + 3.513905038934781) < 1e-9, "-3.51391")),
    ("NMinimize[{x^2 + y^2, x + y >= 2}, {x, y}]", Pred(lambda g: abs(g.args[0] - 2) < 1e-8, "2")),
    ("NMinimize[{(x - 3)^2, 0 <= x <= 2}, x]", Pred(lambda g: abs(g.args[0] - 1) < 1e-8, "1 at x = 2")),
    ("NMinimize[x, x]", Raises()), ("NMinimize[{x, Element[x, Integers]}, x]", Raises()),
    ("NMaximize[{x + y, x^2 + y^2 <= 1}, {x, y}]", Pred(lambda g: abs(g.args[0] - 2 ** 0.5) < 1e-8, "Sqrt[2]")),
    ("NMaximize[Sin[x] Exp[-x^2], x]", Pred(lambda g: abs(g.args[0] - 0.39665) < 1e-4, "0.39665")),
    ("NMaximize[{x y, x + y == 4}, {x, y}]", Pred(lambda g: abs(g.args[0] - 4) < 1e-8, "4")),
    # Fit / FindFit
    ("Coefficient[Fit[{{0, 1}, {1, 0}, {3, 2}, {5, 4}}, {1, x}, x], x]", Near(0.6949152542372882, 1e-10)),
    ("Fit[{{0, 1}, {1, 0}, {3, 2}, {5, 4}}, {1, x}, x] /. x -> 0", Near(0.18644067796610184, 1e-10)),
    ("Coefficient[Fit[{1, 4, 9, 16}, {1, x, x^2}, x], x, 2]", Near(1.0, 1e-10)), ("Abs[Fit[{1, 4, 9, 16}, {1, x, x^2}, x] /. x -> 0] < 10^-10", True),
    ("Chop[Fit[{{0, 0, 1}, {1, 0, 3}, {0, 1, 4}, {1, 1, 6}}, {1, x, y}, {x, y}] - (1. + 2. x + 3. y), 10^-10]", "0"),
    ("a /. FindFit[{{0, 1.}, {1, 2.7}, {2, 7.4}, {3, 20.1}}, a Exp[b x], {a, b}, x]", Near(0.9983168, 1e-6)),
    ("w /. FindFit[Table[{k/4, 3 Sin[k/2]}, {k, 0, 12}], a Sin[w x], {a, {w, 1.8}}, x]", Near(2.0, 1e-8)),
    ("FindFit[{{1, 2}, {2, 4}, {3, 6}}, {a x, a > 0}, {a}, x]", Raises()),
]

INTERP = [
    # Interpolation (reference page: Interpolation[{1, 2, 3, 5, 8, 5}][2.5] = 2.4375)
    ("Interpolation[{1, 2, 3, 5, 8, 5}][2.5]", Near(2.4375)), ("Interpolation[{1, 2, 3, 5, 8, 5}, 2.5]", Near(2.4375)),
    ("Interpolation[{1, 2, 3, 5, 8, 5}][3]", Near(3.0, 1e-14)),
    ("Interpolation[{{0, 0}, {1, 1}, {2, 4}, {3, 9}}][1.5]", Near(2.25)),
    ("Interpolation[{{0, 0}, {1, 1}, {2, 4}, {3, 9}}, InterpolationOrder -> 1][1.5]", Near(2.5)),
    ("Interpolation[{{0, 0}, {1, 1}, {2, 8}, {3, 27}, {4, 64}}][2.5]", Near(15.625)),
    ("Interpolation[{{0, 0}, {1, 1}, {2, 8}, {3, 27}, {4, 64}}]'[2.5]", Near(18.75)),
    ("Interpolation[{{0, 0}, {1, 1}, {2, 8}, {3, 27}, {4, 64}}][{0.5, 3.5}]", "{0.125, 42.875}"),
    ("Interpolation[{{1, 2}, {1, 3}}]", Raises()), ("Interpolation[{{0, {1, 2}}, {1, {2, 3}}}]", Raises()),
    # ListInterpolation
    ("ListInterpolation[{0, 1, 4, 9, 16}, {{0, 1}}][0.5]", Near(4.0)), ("ListInterpolation[{1, 4, 9, 16}][2.5]", Near(6.25)),
    ("ListInterpolation[{1, 2, 3}, {{0, 2}}][1.5]", Near(2.5)), ("ListInterpolation[{{1, 2}, {3, 4}}]", Raises()),
    # InterpolatingFunction objects
    ("Head[Interpolation[{1, 2, 4}]]", "InterpolatingFunction"),
    ("Head[Interpolation[{1, 2, 4}]] === InterpolatingFunction", True),
    ("MatchQ[ListInterpolation[{1, 2, 4}], InterpolatingFunction[{{1., 3.}}, _String]]", True),
    ("Head[NDSolveValue[{y'[x] == 1, y[0] == 0}, y, {x, 0, 1}]] === InterpolatingFunction", True),
    ("First[Interpolation[{{0, 1}, {2, 3}, {5, 1}}]]", "{{0., 5.}}"),
    ("Derivative[2][Interpolation[{{0, 0}, {1, 1}, {2, 4}, {3, 9}}]][1.2]", Near(2.0, 1e-10)),
    ("InterpolatingFunction[{{0, 1}}, \"nosuch\"][0.5]", Raises()),
    # NDSolve / NDSolveValue (reference page: {y[10.5], y'[12.5]} /. s = {{0.0469302, 0.113798}})
    ("{y[10.5], y'[12.5]} /. NDSolve[{y'[x] == y[x] Cos[x + y[x]], y[0] == 1}, y, {x, 0, 30}]",
     Pred(lambda g: abs(g.args[0].args[0] - 0.0469302) < 1e-6 and abs(g.args[0].args[1] - 0.113798) < 1e-6, "doc values")),
    ("(y /. First[NDSolve[{y'[x] == -y[x], y[0] == 1}, y, {x, 0, 2}]])[1.]", Near(float(mp.exp(-1)), 1e-8)),
    ("NDSolve[{y'[x] == y[x], y[1] == 1}, y[x], {x, 0, 2}] /. x -> 0", Pred(lambda g: abs(g.args[0].args[0].args[1] - float(mp.exp(-1))) < 1e-8, "{{y[0] -> E^-1}}")),
    ("Head[NDSolve[{y'[x] == -y[x], y[0] == 1}, y, {x, 0, 2}][[1, 1, 2]]]", "InterpolatingFunction"),
    ("NDSolve[{y''[x] == y[x], y[0] == 1, y[1] == 2}, y, {x, 0, 1}]", Raises()),
    ("NDSolve[{y'[x] == y[x]}, y, {x, 0, 1}]", Raises()),
    ("{x[1.], y[1.]} /. First[NDSolve[{x'[t] == -y[t] - x[t]^2, y'[t] == 2 x[t] - y[t]^3, x[0] == y[0] == 1}, {x, y}, {t, 20}]]",
     Pred(lambda g: abs(g.args[0] + 0.17032312) < 1e-7 and abs(g.args[1] - 0.70724363) < 1e-7, "SciPy reference")),
    ("NDSolveValue[{y''[x] + y[x] == 0, y[0] == 0, y'[0] == 1}, y, {x, 0, 10}][Pi/2]", Near(1.0, 1e-8)),
    ("NDSolveValue[{y''[x] + y[x] == 0, y[0] == 0, y'[0] == 1}, y, {x, 0, 10}]'[Pi]", Near(-1.0, 1e-8)),
    ("NDSolveValue[{y'[t] == y[t] (1 - y[t]), y[0] == 0.1}, y[t], {t, 0, 5}] /. t -> 3", Near(float(1 / (1 + 9 * mp.exp(-3))), 1e-8)),
    ("NDSolveValue[{y'[x] == x, y[1] == 0}, y, {x, 0, 2}][0]", Near(-0.5, 1e-9)),
    ("NDSolveValue[{y'[x] == -y[x], y[0] == 1}, y, {x, 0, 1}, WorkingPrecision -> 30]", Raises()),
]

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
NEEDS_SCIPY = ("FindMaximum", "NMinimize", "NMaximize", "FindFit", "NDSolve")

CASES = [c for c in NUMERICS + SPECIAL + NUMBER_THEORY + NUMCALC + INTERP
         if HAVE_SCIPY or not any(n in c[0] for n in NEEDS_SCIPY)]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_batch2(case):
    check(case)
