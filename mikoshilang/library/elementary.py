"""Elementary functions: trigonometric, hyperbolic, exponential, logarithmic,
rounding and complex-number parts.  Exact input gives exact output (SymPy);
machine-number input gives machine-number output."""
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, S, R, L, numeric_out, sym, number, is_list,
                   has_float)

AREA = "elementary"


def _unary(name, f, notes="verified against SymPy/mpmath", status="verified"):
    @fn(name, area=AREA, status=status, notes=notes, listable=True)
    def impl(z):
        v = f(sym(z, name))
        if name in _INT_VALUED and isinstance(v, sp.Integer):
            return R(v)          # Floor[2.7] = 2, Sign[-3.5] = -1 (integers, as in Wolfram)
        return numeric_out(v, z)
    return impl


_INT_VALUED = {"Sign", "Floor", "Ceiling"}


for _name, _f in [
    ("Sin", sp.sin), ("Cos", sp.cos), ("Tan", sp.tan), ("Cot", sp.cot), ("Sec", sp.sec), ("Csc", sp.csc),
    ("Sinh", sp.sinh), ("Cosh", sp.cosh), ("Tanh", sp.tanh), ("Coth", sp.coth), ("Sech", sp.sech),
    ("Csch", sp.csch), ("ArcSin", sp.asin), ("ArcCos", sp.acos), ("ArcCot", sp.acot), ("ArcSec", sp.asec),
    ("ArcCsc", sp.acsc), ("ArcSinh", sp.asinh), ("ArcCosh", sp.acosh), ("ArcTanh", sp.atanh),
    ("Exp", sp.exp), ("Sqrt", sp.sqrt), ("Abs", sp.Abs), ("Sign", sp.sign), ("Floor", sp.floor),
    ("Ceiling", sp.ceiling), ("Conjugate", sp.conjugate), ("RealPart", sp.re), ("ImaginaryPart", sp.im),
]:
    _unary(_name, _f)


@fn("ArcTan", area=AREA, status="fixed", listable=True,
    notes="ArcTan[x, y] was atan2(x, y); Wolfram's ArcTan[x, y] is the angle of the point (x, y), atan2(y, x)")
def _arctan1(z):
    return numeric_out(sp.atan(sym(z, "ArcTan")), z)


@fn("ArcTan", area=AREA, status="fixed")
def _arctan2(x, y):
    sx, sy = sym(x, "ArcTan"), sym(y, "ArcTan")
    if sx == 0 and sy == 0:
        raise ValueError("ArcTan[0, 0] is indeterminate")
    return numeric_out(sp.atan2(sy, sx), x, y)


def _log(z):
    if z == 0:
        return -sp.oo           # Wolfram: Log[0] = -Infinity (SymPy gives zoo)
    return sp.log(z)


@fn("Log", area=AREA, status="fixed", listable=True,
    notes="Log[0] returned ComplexInfinity; Wolfram gives -Infinity. Log[b, z] is log base b")
def _log1(z):
    return numeric_out(_log(sym(z, "Log")), z)


@fn("Log", area=AREA, status="fixed")
def _log2arg(b, z):
    sb, sz = sym(b, "Log"), sym(z, "Log")
    if sb == 1:
        raise ValueError("Log[1, z] is undefined (base 1)")
    if sz == 0:
        return numeric_out(-sp.oo if sb.is_number and sb > 1 else sp.log(sz, sb), b, z)
    return numeric_out(sp.log(sz, sb), b, z)


@fn("Log2", area=AREA, status="verified", listable=True, notes="log base 2; Log2[0] = -Infinity")
def _log2(z):
    s = sym(z, "Log2")
    return numeric_out(-sp.oo if s == 0 else sp.log(s, 2), z)


@fn("Log10", area=AREA, status="verified", listable=True, notes="log base 10; Log10[0] = -Infinity")
def _log10(z):
    s = sym(z, "Log10")
    return numeric_out(-sp.oo if s == 0 else sp.log(s, 10), z)


@fn("CubeRoot", area=AREA, status="fixed", listable=True,
    notes="returned the principal complex root ((-8)^(1/3) = 1 + I Sqrt[3]); Wolfram CubeRoot is the real root (-2)")
def _cuberoot(z):
    s = sym(z, "CubeRoot")
    if s.is_number and s.is_real is False:
        raise ValueError("CubeRoot is defined for real numbers only (use z^(1/3) for complex z)")
    return numeric_out(sp.real_root(s, 3) if s.is_number else sp.cbrt(s), z)


error("Root", AREA, "Wolfram's Root[f, k] denotes an algebraic number (the k-th root of a polynomial); "
      "the legacy Root[a, n] = a^(1/n) had different semantics. Use Surd[a, n] or Power[a, 1/n].")


@fn("Surd", area=AREA, status="promoted", listable=True, notes="real n-th root (Wolfram Surd)")
def _surd(x, n):
    s = sym(x, "Surd")
    from .core import int_arg
    k = int_arg(n, "Surd", "root index")
    if k == 0:
        raise ValueError("Surd: root index must be non-zero")
    if s.is_number and s.is_negative and k % 2 == 0:
        raise ValueError("Surd: even root of a negative number is not real")
    return numeric_out(sp.real_root(s, k) if s.is_number else s ** sp.Rational(1, k), x)


def _round_half_even(q: Fraction) -> int:
    fl = q.numerator // q.denominator
    rem = q - fl
    if rem > Fraction(1, 2):
        return fl + 1
    if rem < Fraction(1, 2):
        return fl
    return fl if fl % 2 == 0 else fl + 1


def _as_fraction(v):
    if v.is_Rational:
        return Fraction(int(v.p), int(v.q))
    return None


@fn("Round", area=AREA, status="fixed", listable=True,
    notes="Round[x] was not implemented and Round[x, a] returned floats (Round[7, 5] = 5.0); "
          "now Wolfram half-to-even rounding, exact for exact input")
def _round1(x):
    v = number(x, "Round", real=False)
    if v.is_real is False:
        re, im = sp.re(v), sp.im(v)
        return R(_round_real(re) + sp.I * _round_real(im))
    if v in (sp.oo, -sp.oo):
        return R(v)
    return R(_round_real(v))


def _round_real(v):
    q = _as_fraction(v)
    if q is None:
        if v.is_Float:
            q = Fraction(float(v)) if v._prec <= 53 else Fraction(str(v))
        else:
            q = Fraction(str(sp.N(v, 60)))
    return sp.Integer(_round_half_even(q))


@fn("Round", area=AREA, status="fixed")
def _round2(x, a):
    v = number(x, "Round")
    step = number(a, "Round", "rounding step")
    if step == 0:
        raise ValueError("Round: rounding step must be non-zero")
    k = _round_real(v / step)
    out = k * step
    return numeric_out(out, x, a)


@fn("Mod", area=AREA, status="verified", listable=True,
    notes="Mod[m, n] has the sign of n (Wolfram); Mod[m, 0] raises")
def _mod(m, n):
    sm, sn = sym(m, "Mod"), sym(n, "Mod")
    if sn == 0:
        raise ValueError("Mod: modulus must be non-zero")
    return numeric_out(sp.Mod(sm, sn), m, n)


@fn("Quotient", area=AREA, status="verified", listable=True, notes="Quotient[m, n] = Floor[m/n]")
def _quotient(m, n):
    sm, sn = sym(m, "Quotient"), sym(n, "Quotient")
    if sn == 0:
        raise ValueError("Quotient: divisor must be non-zero")
    return numeric_out(sp.floor(sm / sn), m, n)


def _flatten_args(args):
    out = []
    for a in args:
        if is_list(a):
            out.extend(_flatten_args(a.args))
        else:
            out.append(a)
    return out


@fn("Max", area=AREA, status="fixed",
    notes="returned floats for exact input (Max[1, 2] = 2.0) and only took 1-2 arguments; "
          "now exact, variadic, flattens lists, Max[] = -Infinity")
def _max(*args):
    items = _flatten_args(args)
    if not items:
        return R(-sp.oo)
    vals = [sym(v, "Max") for v in items]
    best = sp.Max(*vals)
    for item, v in zip(items, vals):
        if v == best:
            return item          # Wolfram returns the extremal element itself: Max[1/2, 0.4] = 1/2
    return R(best)


@fn("Min", area=AREA, status="fixed",
    notes="returned floats for exact input; now exact, variadic, flattens lists, Min[] = Infinity")
def _min(*args):
    items = _flatten_args(args)
    if not items:
        return R(sp.oo)
    vals = [sym(v, "Min") for v in items]
    best = sp.Min(*vals)
    for item, v in zip(items, vals):
        if v == best:
            return item          # Wolfram returns the extremal element itself: Min[1/2, 0.4] = 1/2
    return R(best)
