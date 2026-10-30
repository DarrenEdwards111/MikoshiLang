import math

import mpmath as mp
import pytest

from _harness import Raises, Pred, check, ids

mp.mp.dps = 30

CASES = [
    # trig (exact special values from Wolfram, numeric values from mpmath)
    ("Sin[Pi/6]", "1/2"), ("Sin[0.7]", mp.sin(0.7)), ("Sin[2 + I]", mp.sin(2 + 1j)), ("Sin[x]", "Sin[x]"),
    ("Cos[Pi/3]", "1/2"), ("Cos[1.3]", mp.cos(1.3)), ("Cos[Pi]", "-1"),
    ("Tan[Pi/4]", "1"), ("Tan[0.4]", mp.tan(0.4)), ("Tan[Pi/3]", "Sqrt[3]"),
    ("Cot[Pi/4]", "1"), ("Cot[0.4]", mp.cot(0.4)), ("Cot[Pi/6]", "Sqrt[3]"),
    ("Sec[0]", "1"), ("Sec[0.4]", mp.sec(0.4)), ("Sec[Pi/3]", "2"),
    ("Csc[Pi/2]", "1"), ("Csc[0.4]", mp.csc(0.4)), ("Csc[Pi/6]", "2"),
    ("Sinh[0]", "0"), ("Sinh[1.5]", mp.sinh(1.5)), ("Sinh[-0.3]", mp.sinh(-0.3)),
    ("Cosh[0]", "1"), ("Cosh[1.5]", mp.cosh(1.5)), ("Cosh[2 I]", mp.cos(2)),
    ("Tanh[0]", "0"), ("Tanh[0.5]", mp.tanh(0.5)), ("Tanh[-2.5]", mp.tanh(-2.5)),
    ("Coth[0.5]", mp.coth(0.5)), ("Coth[2.]", mp.coth(2)), ("Coth[-1.25]", mp.coth(-1.25)),
    ("Sech[0]", "1"), ("Sech[0.5]", mp.sech(0.5)), ("Sech[3.]", mp.sech(3)),
    ("Csch[0.5]", mp.csch(0.5)), ("Csch[2.]", mp.csch(2)), ("Csch[-1.]", mp.csch(-1)),
    ("ArcSin[1]", "Pi/2"), ("ArcSin[0.3]", mp.asin(0.3)), ("ArcSin[2.]", mp.asin(2)),
    ("ArcCos[0]", "Pi/2"), ("ArcCos[0.3]", mp.acos(0.3)), ("ArcCos[-1]", "Pi"),
    ("ArcTan[1]", "Pi/4"), ("ArcTan[0.3]", mp.atan(0.3)),
    ("ArcTan[-1, 0]", "Pi"), ("ArcTan[0, 1]", "Pi/2"), ("ArcTan[-1., -1.]", mp.atan2(-1, -1)),
    ("ArcCot[1]", "Pi/4"), ("ArcCot[0.3]", mp.acot(0.3)), ("ArcCot[-1]", "-Pi/4"),
    ("ArcSec[2]", "Pi/3"), ("ArcSec[3.]", mp.asec(3)), ("ArcSec[1]", "0"),
    ("ArcCsc[2]", "Pi/6"), ("ArcCsc[3.]", mp.acsc(3)), ("ArcCsc[1]", "Pi/2"),
    ("ArcSinh[0]", "0"), ("ArcSinh[0.7]", mp.asinh(0.7)), ("ArcSinh[-2.]", mp.asinh(-2)),
    ("ArcCosh[1]", "0"), ("ArcCosh[2.5]", mp.acosh(2.5)), ("ArcCosh[0.5]", mp.acosh(0.5)),
    ("ArcTanh[0]", "0"), ("ArcTanh[0.5]", mp.atanh(0.5)), ("ArcTanh[2.]", mp.atanh(2)),
    # exponential / roots
    ("Exp[0]", "1"), ("Exp[1.5]", mp.exp(1.5)), ("Exp[I Pi]", "-1"),
    ("Sqrt[16]", "4"), ("Sqrt[8]", "2 Sqrt[2]"), ("Sqrt[-4]", "2 I"), ("Sqrt[2.]", mp.sqrt(2)),
    ("Log[1]", "0"), ("Log[0]", "-Infinity"), ("Log[2.5]", mp.log(2.5)), ("Log[-1]", "I Pi"),
    ("Log[2, 8]", "3"), ("Log[10, 0.5]", mp.log(0.5, 10)), ("Log[E]", "1"),
    ("Log2[8]", "3"), ("Log2[10.]", mp.log(10, 2)), ("Log2[1/2]", "-1"),
    ("Log10[1000]", "3"), ("Log10[2.]", mp.log10(2)), ("Log10[1/100]", "-2"),
    ("CubeRoot[27]", "3"), ("CubeRoot[-8]", "-2"), ("CubeRoot[2.]", mp.cbrt(2)),
    ("Surd[-8, 3]", "-2"), ("Surd[16, 4]", "2"), ("Surd[-32., 5]", -2.0), ("Surd[-16, 4]", Raises()),
    ("Root[8, 3]", Raises(NotImplementedError, "Surd")),
    # absolute value & parts
    ("Abs[-3]", "3"), ("Abs[3 + 4 I]", "5"), ("Abs[-2.5]", 2.5),
    ("Sign[-2]", "-1"), ("Sign[0]", "0"), ("Sign[3.5]", "1"), ("Sign[3 + 4 I]", "3/5 + 4 I/5"),
    ("Floor[2.7]", "2"), ("Floor[-2.7]", "-3"), ("Floor[7/2]", "3"),
    ("Ceiling[2.1]", "3"), ("Ceiling[-2.7]", "-2"), ("Ceiling[7/2]", "4"),
    ("Conjugate[3 + 4 I]", "3 - 4 I"), ("Conjugate[2]", "2"), ("Conjugate[1.5 - 2. I]", 1.5 + 2j),
    ("RealPart[3 + 4 I]", "3"), ("RealPart[2.5]", 2.5), ("RealPart[-I]", "0"),
    ("ImaginaryPart[3 + 4 I]", "4"), ("ImaginaryPart[2]", "0"), ("ImaginaryPart[1. - 2.5 I]", -2.5),
    # Wolfram Round rounds half to even
    ("Round[2.5]", "2"), ("Round[3.5]", "4"), ("Round[-2.5]", "-2"), ("Round[7/2]", "4"), ("Round[2.6]", "3"),
    ("Round[17, 5]", "15"), ("Round[3.14159, 0.01]", 3.14),
    ("Mod[7, 3]", "1"), ("Mod[-7, 3]", "2"), ("Mod[7, -3]", "-2"), ("Mod[5.5, 2]", 1.5), ("Mod[7, 0]", Raises()),
    ("Quotient[7, 3]", "2"), ("Quotient[-7, 3]", "-3"), ("Quotient[7, -3]", "-3"), ("Quotient[1, 0]", Raises()),
    ("Max[1, 5, 3]", "5"), ("Max[{1, 7}, 3]", "7"), ("Max[]", "-Infinity"), ("Max[1/2, 0.4]", "1/2"),
    ("Min[1, 5, 3]", "1"), ("Min[{4, 2}, {9}]", "2"), ("Min[]", "Infinity"), ("Min[-1/3, -0.3]", "-1/3"),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_elementary(case):
    check(case)
