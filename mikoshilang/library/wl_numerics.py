"""Coverage batch 2: precision control and number representation (Wolfram semantics).

MikoshiLang numbers are machine reals (Python float/complex) or arbitrary-precision reals (SymPy Float with a
binary precision above 53 bits).  Arithmetic is fixed-precision (the result of an operation carries the lowest
input precision); Wolfram's significance arithmetic, which also lowers precision on cancellation, is not
modelled.  A requested precision at or below MachinePrecision gives a machine number.
"""

from __future__ import annotations
import math
from fractions import Fraction

import mpmath
import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, record, R, sym, number, is_list, has_float, int_arg, list_arg)

AREA = "numerics (batch 2)"
MACHINE_PRECISION = 53 * math.log10(2)            # 15.954589770191003
MACHINE_EPSILON = 2.0 ** -52
MIN_MACHINE = 2.2250738585072014e-308

INF = Symbol("Infinity")


def dps_bits(dps):
    """Binary precision for ``dps`` decimal digits (mpmath dps_to_prec, so N[x, n], x`n and SetPrecision agree)."""
    return max(1, int(round((float(dps) + 1) * 3.321928094887362)))


def bits_dps(bits):
    return float(mpmath.libmp.prec_to_dps(int(bits)))


# ───────────────────────────── precision of an expression ─────────────────────────────

def _num_prec(x):
    """None for exact numbers, "m" for machine numbers, else the binary precision of a bignum."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (float, complex)):
        return "m"
    if isinstance(x, sp.Float):
        return "m" if x._prec <= 53 else x._prec
    return None


def _leaves(x):
    if isinstance(x, Expr):
        for a in x.args:
            yield from _leaves(a)
        if isinstance(x.head, Expr):
            yield from _leaves(x.head)
    elif isinstance(x, sp.Basic) and not isinstance(x, sp.Float):
        for a in sp.preorder_traversal(x):
            if isinstance(a, sp.Float):
                yield a
    else:
        yield x


def _min_prec(x):
    best = None
    for leaf in _leaves(x):
        p = _num_prec(leaf)
        if p is None:
            continue
        if p == "m":
            return "m"
        best = p if best is None else min(best, p)
    return best


def _is_zero_bignum(x):
    return isinstance(x, sp.Float) and x._prec > 53 and x.is_zero


@fn("Precision", area=AREA, status="added",
    notes="Precision[x]: Infinity for exact numbers and symbolic expressions without approximate numbers, "
          "MachinePrecision for machine numbers, the decimal precision of a bignum; the minimum over an "
          "expression (a machine number makes the whole expression MachinePrecision). An arbitrary-precision zero "
          "has Precision 0."
          " Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _precision(x: Expr) -> Expr:
    zeros = [leaf for leaf in _leaves(x) if _is_zero_bignum(leaf)]
    p = _min_prec(x)
    if p is None:
        return INF
    if p == "m":
        return Symbol("MachinePrecision")
    if zeros:
        return 0.0
    return bits_dps(p)


def _abs_log10(v):
    a = abs(complex(v)) if isinstance(v, (float, complex)) else float(abs(sp.sympify(v)))
    return math.log10(a) if a else None


@fn("Accuracy", area=AREA, status="added",
    notes="Accuracy[x] = Precision[x] - Log10[Abs[x]] (digits after the decimal point); Infinity for exact "
          "numbers; machine zero has Accuracy -Log10[$MinMachineNumber] = 307.653; the minimum over an expression"
          ". Check: reference (differential cases in tests/library).")
def _accuracy(x: Expr) -> Expr:
    best = None
    for leaf in _leaves(x):
        p = _num_prec(leaf)
        if p is None:
            continue
        if p == "m":
            lg = _abs_log10(leaf)
            acc = -math.log10(MIN_MACHINE) if lg is None else MACHINE_PRECISION - lg
        else:
            lg = _abs_log10(leaf)
            acc = bits_dps(p) if lg is None else bits_dps(p) - lg
        best = acc if best is None else min(best, acc)
    return INF if best is None else best


_CONST = {"Pi": sp.pi, "E": sp.E, "EulerGamma": sp.EulerGamma, "GoldenRatio": sp.GoldenRatio,
          "Catalan": sp.Catalan, "Degree": sp.pi / 180}


def _exact(v):
    """Exact rational value of a number (binary value of floats)."""
    if isinstance(v, float):
        f = Fraction(v)
        return sp.Rational(f.numerator, f.denominator)
    if isinstance(v, sp.Float):
        return sp.Rational(v)
    return v


def _num_at(v, target):
    """Exact SymPy number ``v`` at the target precision."""
    if target[0] == "inf":
        return R(v)
    if v.is_real is False:
        re, im = v.as_real_imag()
        return R(sp.sympify(_num_at_real(re, target)) + sp.I * sp.sympify(_num_at_real(im, target)))
    return _num_at_real(v, target)


def _num_at_real(v, target):
    if target[0] == "m":
        return float(sp.N(v, 17))
    if target[0] == "acc":
        if v.is_zero:
            return sp.Float(0, precision=dps_bits(target[1]))
        prec = target[1] + math.log10(abs(float(sp.N(v, 17))))
    else:
        prec = target[1]
    if prec <= MACHINE_PRECISION:
        return float(sp.N(v, 17))
    bits = dps_bits(prec)
    if v.is_Rational:
        return sp.Float(v, precision=bits)
    return sp.Float(sp.N(v, int(prec) + 10), precision=bits)


def _exact_tree(x):
    """Replace approximate numbers by the exact rationals they represent."""
    if isinstance(x, (float, sp.Float)):
        return R(_exact(x))
    if isinstance(x, complex):
        return R(_exact(x.real) + sp.I * _exact(x.imag))
    if isinstance(x, Expr):
        return Expr(x.head, *(_exact_tree(a) for a in x.args))
    return x


def _set_prec(x, target, ev):
    """target: ("inf",) | ("m",) | ("bits", n) | ("acc", a)."""
    if isinstance(x, (bool, str)) or x is None:
        return x
    if isinstance(x, (Symbol, Expr)) and not (isinstance(x, Symbol) and x.name not in _CONST and x.name != "I"):
        try:
            v = sp.sympify(sym(_exact_tree(x), "SetPrecision"))
        except Exception:
            v = None
        if v is not None and v.is_number and not v.has(sp.oo, -sp.oo, sp.zoo, sp.nan):
            return _num_at(v, target)
    if isinstance(x, (int, float, complex, sp.Float, sp.Rational)):
        return _num_at(sp.sympify(_exact(x)) if not isinstance(x, complex)
                       else _exact(x.real) + sp.I * _exact(x.imag), target)
    if isinstance(x, Expr):
        head = x.head if not isinstance(x.head, Expr) else _set_prec(x.head, target, ev)
        return ev(Expr(head, *(_set_prec(a, target, ev) for a in x.args)))
    return x


def _target(p, name):
    if isinstance(p, Symbol) and p.name == "Infinity":
        return ("inf",)
    if isinstance(p, Symbol) and p.name == "MachinePrecision":
        return ("m",)
    v = number(p, name, "precision")
    if v == sp.oo:
        return ("inf",)
    return ("bits", float(v))


@fn("SetPrecision", area=AREA, status="added",
    notes="SetPrecision[expr, p] sets every number in expr (and the numeric constants Pi, E, ...) to precision p: "
          "machine numbers keep their exact binary value padded with binary zeros (SetPrecision[0.1, 20] = "
          "0.10000000000000000555), p = Infinity converts approximate numbers to the exact rationals they "
          "represent, p = MachinePrecision gives machine numbers; the result is re-evaluated"
          ". Check: reference (differential cases in tests/library).")
def _setprecision(x: Expr, p: Expr) -> Expr:
    from ..evaluate import evaluate
    return _set_prec(x, _target(p, "SetPrecision"), evaluate)


@fn("SetAccuracy", area=AREA, status="added",
    notes="SetAccuracy[expr, a]: each number gets precision a + Log10[Abs[x]] (an arbitrary-precision zero keeps "
          "accuracy a)")
def _setaccuracy(x: Expr, a: Expr) -> Expr:
    from ..evaluate import evaluate
    t = _target(a, "SetAccuracy")
    if t[0] == "bits":
        t = ("acc", t[1])
    return _set_prec(x, t, evaluate)


record("MachinePrecision", AREA, "added",
       "symbol for machine precision; N[MachinePrecision] = 15.9546 (53 Log10[2])")
record("$MachineEpsilon", AREA, "added", "2^-52 = 2.22045*10^-16")
record("$MachinePrecision", AREA, "added", "53 Log10[2] = 15.9546")
record("$MaxMachineNumber", AREA, "added", "1.79769*10^308")
record("$MinMachineNumber", AREA, "added", "2.22507*10^-308")

SYSTEM_CONSTANTS = {
    "$MachineEpsilon": MACHINE_EPSILON, "$MachinePrecision": MACHINE_PRECISION,
    "$MaxMachineNumber": 1.7976931348623157e308, "$MinMachineNumber": MIN_MACHINE,
}


# ───────────────────────────── Chop, Rationalize ─────────────────────────────

def _chop(x, d):
    if isinstance(x, bool):
        return x
    if isinstance(x, float):
        return 0 if abs(x) < d else x
    if isinstance(x, complex):
        re = 0 if abs(x.real) < d else x.real
        im = 0 if abs(x.imag) < d else x.imag
        if im == 0:
            return re
        return R(sp.sympify(re) + sp.I * sp.Float(im))
    if isinstance(x, sp.Float):
        return 0 if abs(x) < d else x
    if isinstance(x, Expr):
        from ..evaluate import evaluate
        if isinstance(x.head, str) and x.head in ("Plus", "Times") and has_float(x):
            try:
                v = sp.sympify(sym(x, "Chop"))
                if v.is_number:
                    re, im = v.as_real_imag()
                    re = sp.Integer(0) if abs(re) < d else re
                    im = sp.Integer(0) if abs(im) < d else im
                    return R(re + sp.I * im)
            except Exception:
                pass
        return evaluate(Expr(x.head, *(_chop(a, d) for a in x.args)))
    return x


@fn("Chop", area=AREA, status="added",
    notes="Chop[expr, delta] replaces approximate real numbers (and real or imaginary parts) with absolute value "
          "below delta (default 10^-10) by exact 0; exact numbers are untouched"
          ". Check: reference (differential cases in tests/library).")
def _chop_impl(x: Expr, delta: Expr = None) -> Expr:
    d = 1e-10 if delta is None else float(number(delta, "Chop", "tolerance"))
    return _chop(x, d)


def _cf(q: Fraction, limit=4000):
    """Continued fraction terms of a rational."""
    out = []
    p, r = q.numerator, q.denominator
    while r and len(out) < limit:
        a = p // r
        out.append(a)
        p, r = r, p - a * r
    return out


def _convergents(terms):
    p0, q0, p1, q1 = 1, 0, terms[0], 1
    yield Fraction(p1, q1), (p0, q0, p1, q1), terms[0]
    for a in terms[1:]:
        p0, q0, p1, q1 = p1, q1, a * p1 + p0, a * q1 + q0
        yield Fraction(p1, q1), (p0, q0, p1, q1), a


def _rationalize_dx(x: Fraction, dx: Fraction):
    """First convergent within dx; if it is not one of the first two, the smallest semiconvergent leading to
    it that is within dx (Wolfram: Rationalize[Pi, 1/10] = 22/7, Rationalize[Pi, 1/1000] = 201/64)."""
    terms = _cf(x)
    prev = None
    for i, (c, st, a) in enumerate(_convergents(terms)):
        if abs(c - x) < dx or c == x:
            if i >= 2 and prev is not None:
                pp0, qq0, pp1, qq1 = prev
                for k in range(1, a):
                    s = Fraction(k * pp1 + pp0, k * qq1 + qq0)
                    if abs(s - x) < dx:
                        return s
            return c
        prev = (st[0], st[1], st[2], st[3])
    return x


def _rationalize_auto(x: Fraction, rel: float):
    """Rationalize[x]: the first convergent p/q with |p/q - x| < 10^-4/q^2; x itself when none exists above
    the number's precision."""
    c0 = 1e-4
    for c, _st, _a in _convergents(_cf(x)):
        q = c.denominator
        if c0 / q ** 2 < rel:
            return None
        if abs(c - x) < Fraction(c0) / q ** 2:
            return c
    return None


def _rat_number(v, dx):
    if isinstance(v, bool) or isinstance(v, int):
        return v
    if isinstance(v, (float, sp.Float)) or (isinstance(v, sp.Basic) and v.is_number):
        if isinstance(v, sp.Basic) and v.is_number and v.is_real is False:
            re, im = v.as_real_imag()
            return R(sp.sympify(_rat_value(re, dx, v)) + sp.I * sp.sympify(_rat_value(im, dx, v)))
        return _rat_value(v, dx, v)
    return v


def _rat_value(v, dx, orig):
    if isinstance(v, sp.Basic) and v.is_Rational:
        return R(v)
    if isinstance(v, float) or (isinstance(v, sp.Float)):
        x = Fraction(v) if isinstance(v, float) else Fraction(sp.Rational(v).p, sp.Rational(v).q)
        rel = abs(float(v)) * (2.0 ** -53 if isinstance(v, float) or v._prec <= 53 else 2.0 ** -v._prec)
    else:
        if dx is None:
            return R(v)                                      # exact irrational numbers stay exact
        val = sp.N(v, 60)
        x = Fraction(sp.Rational(val).p, sp.Rational(val).q)
        rel = 0.0
    if dx is None:
        r = _rationalize_auto(x, rel)
        return R(v) if r is None else R(sp.Rational(r.numerator, r.denominator))
    if dx < 2 * Fraction(rel):          # tolerance below the number's precision: same as Rationalize[x, 0]
        dx = 0
    if dx == 0:
        if rel == 0:
            return R(sp.Rational(x.numerator, x.denominator))
        half = Fraction(rel) / 2 if rel else Fraction(0)
        r = _simplest_between(x - half, x + half)
        return R(sp.Rational(r.numerator, r.denominator))
    r = _rationalize_dx(x, dx)
    return R(sp.Rational(r.numerator, r.denominator))


def _simplest_between(lo: Fraction, hi: Fraction) -> Fraction:
    """Rational with the smallest denominator in the closed interval [lo, hi] (Stern-Brocot)."""
    fl = lo.numerator // lo.denominator
    if Fraction(fl) == lo:
        return Fraction(fl)
    if fl + 1 <= hi:
        return Fraction(fl + 1)
    r = _simplest_between(1 / (hi - fl), 1 / (lo - fl))
    return fl + 1 / r


def _rat_walk(x, dx):
    if isinstance(x, Expr):
        from ..evaluate import evaluate
        return evaluate(Expr(x.head, *(_rat_walk(a, dx) for a in x.args)))
    if isinstance(x, (float, complex, sp.Float)):
        if isinstance(x, complex):
            return R(sp.sympify(_rat_number(x.real, dx)) + sp.I * sp.sympify(_rat_number(x.imag, dx)))
        return _rat_number(x, dx)
    return x


@fn("Rationalize", area=AREA, status="added",
    notes="Rationalize[x] gives the first continued-fraction convergent p/q with |p/q - x| < 10^-4/q^2, or x "
          "unchanged when there is none (Rationalize[N[Pi]] = 3.14159); Rationalize[x, dx] gives the first "
          "convergent within dx, refined to the smallest semiconvergent within dx (Rationalize[Pi, 1/1000] = "
          "201/64); Rationalize[x, 0] gives the simplest rational within the precision of x; applies to every "
          "approximate number of an expression"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _rationalize(x: Expr, dx: Expr = None) -> Expr:
    if dx is None:
        return _rat_walk(x, None)
    d = number(dx, "Rationalize", "tolerance")
    if d < 0:
        raise ValueError("Rationalize: tolerance must be non-negative")
    dq = Fraction(0) if d.is_zero else Fraction(sp.Rational(d).p, sp.Rational(d).q) if d.is_Rational else \
        Fraction(sp.Rational(sp.N(d, 40)).p, sp.Rational(sp.N(d, 40)).q)
    if isinstance(x, (Expr, Symbol)) and not has_float(x):
        v = sym(x, "Rationalize")
        if v.is_number:
            return _rat_number(v, dq)
    return _rat_walk(x, dq)


# ───────────────────────────── IntegerPart, FractionalPart ─────────────────────────────

def _ipart_real(v):
    if v in (sp.oo, -sp.oo):
        return v
    if v.is_Rational:
        return sp.Integer(int(v))                          # truncates toward zero
    if v.is_Float:
        return sp.Integer(int(_exact(v)))
    f = sp.N(v, 50)
    if not f.is_real:
        raise Unevaluated()
    return sp.Integer(int(sp.Rational(f)))


@fn("IntegerPart", area=AREA, status="added", listable=True,
    notes="integer part (truncation toward zero) as an exact integer, separately for real and imaginary parts; "
          "IntegerPart[Infinity] = Infinity"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _integerpart(x: Expr) -> Expr:
    if isinstance(x, (Symbol, Expr)) and not isinstance(x, sp.Basic):
        v = sym(x, "IntegerPart")
        if not v.is_number:
            raise Unevaluated()
    v = number(x, "IntegerPart", real=False)
    if v in (sp.oo, -sp.oo):
        return R(v)
    if v.is_real is False:
        re, im = v.as_real_imag()
        return R(_ipart_real(re) + sp.I * _ipart_real(im))
    return R(_ipart_real(v))


@fn("FractionalPart", area=AREA, status="added", listable=True,
    notes="FractionalPart[x] = x - IntegerPart[x] (same sign as x: FractionalPart[-2.4] = -0.4), separately "
          "for real and imaginary parts; exact for exact input"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _fracpart(x: Expr) -> Expr:
    v = sym(x, "FractionalPart")
    if not v.is_number:
        raise Unevaluated()
    if v.has(sp.oo, -sp.oo, sp.zoo):
        raise Unevaluated()
    if v.is_real is False:
        re, im = v.as_real_imag()
        out = (re - _ipart_real(re)) + sp.I * (im - _ipart_real(im))
    else:
        out = v - _ipart_real(v)
    if has_float(x):
        return R(sp.sympify(out).evalf(max(15, _float_dps(x))) if _float_dps(x) > 15 else sp.N(out, 15))
    return R(sp.expand(out))


def _float_dps(x):
    p = _min_prec(x)
    return 15 if p in (None, "m") else int(bits_dps(p))


# ───────────────────────────── RealDigits ─────────────────────────────

def _int_base(b, name):
    k = int_arg(b, name, "base")
    if k < 2:
        raise ValueError(f"{name}: base must be an integer >= 2")
    return k


def _rational_digits(q: Fraction, b: int):
    """Exact digits of a positive rational: (digits, exponent) with the repeating block nested as a list."""
    if q == 0:
        return [0], 1
    e = 0
    while q >= Fraction(b) ** e:
        e += 1
    while e > -10 ** 6 and q < Fraction(b) ** (e - 1):
        e -= 1
    y = q / Fraction(b) ** e                     # in [1/b, 1)
    num, den = y.numerator, y.denominator
    digits, seen = [], {}
    r = num
    while r and r not in seen:
        seen[r] = len(digits)
        r *= b
        digits.append(r // den)
        r %= den
    if not r:
        return digits, e
    start = seen[r]
    return digits[:start] + [digits[start:]], e


def _digits_fixed(q: Fraction, b: int, length: int, top=None):
    """``length`` truncated digits of |q| (exact), starting at b^(e-1), or at b^top when ``top`` is given."""
    if top is None:
        if q == 0:
            return [0] * length, 0 if False else 1
        e = 0
        while q >= Fraction(b) ** e:
            e += 1
        while q < Fraction(b) ** (e - 1):
            e -= 1
    else:
        e = top + 1
    scaled = q * Fraction(b) ** (length - e)
    n = scaled.numerator // scaled.denominator
    out = []
    for _ in range(length):
        out.append(n % b)
        n //= b
    return out[::-1], e


@fn("RealDigits", area=AREA, status="added",
    notes="RealDigits[x, b, len, n]: {digits, exponent}. Machine numbers give their full precision "
          "(16 decimal digits, 53 binary), digits beyond the precision of an approximate number are "
          "Indeterminate; exact rationals give the exact digits with the recurring block as a nested list "
          "(RealDigits[19/7] = {{2, {7, 1, 4, 2, 8, 5}}, 1}); exact irrationals need len. Digits are of Abs[x]; "
          "an approximate number's digits are those of its exact binary value, rounded at its precision"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _realdigits(x: Expr, b: Integer = 10, length: Expr = None, start: Expr = None) -> ListOf[Expr]:
    name = "RealDigits"
    base = _int_base(b, name)
    v = sym(x, name)
    if not v.is_number:
        raise Unevaluated()
    if v.is_real is False:
        raise ValueError("RealDigits: argument must be a real number")
    v = abs(v)
    nlen = None if length is None else int_arg(length, name, "length")
    if nlen is not None and nlen < 1:
        raise ValueError("RealDigits: length must be positive")
    top = None if start is None else int_arg(start, name, "start position")
    if has_float(x):
        p = _min_prec(x)
        pdec = MACHINE_PRECISION if p == "m" else bits_dps(p)
        avail = int(math.ceil(pdec * math.log(10) / math.log(base) - 1e-9))
        q = Fraction(sp.Rational(sp.sympify(v)).p, sp.Rational(sp.sympify(v)).q) if not isinstance(v, float) \
            else Fraction(v)
        want = avail if nlen is None else nlen
        if q == 0:
            return Expr("List", Expr("List", *([0] * want)), 0 if top is None else top + 1)
        # digits correct to the precision: round at the last available digit
        digs, e = _digits_fixed(q, base, avail + 3)
        e0 = e
        exact_digits = digs
        n = 0
        for d in exact_digits[:avail]:
            n = n * base + d
        if exact_digits[avail] * 2 >= base:
            n += 1
        rounded = []
        for _ in range(avail):
            rounded.append(n % base)
            n //= base
        rounded = rounded[::-1]
        if n:                                         # carry into a new leading digit
            rounded = [n] + rounded[:-1]
            e0 += 1
        if top is not None:
            shift = (e0 - 1) - top                    # digits before the requested start
            if shift >= 0:
                rounded = rounded[shift:]
            else:
                rounded = [0] * (-shift) + rounded
            e0 = top + 1
        if len(rounded) >= want:
            out = rounded[:want]
        else:
            out = rounded + [Symbol("Indeterminate")] * (want - len(rounded))
        return Expr("List", Expr("List", *out), e0)
    if v.is_Rational:
        q = Fraction(int(v.p), int(v.q))
        if nlen is None and top is None:
            digs, e = _rational_digits(q, base)
            return Expr("List", R(digs), e)
        if nlen is None:
            raise ValueError("RealDigits: give a length with a start position")
        digs, e = _digits_fixed(q, base, nlen, top)
        return Expr("List", R(digs), e)
    if nlen is None:
        raise ValueError("RealDigits: an exact irrational number needs an explicit number of digits, "
                         "RealDigits[x, b, len]")
    # exact irrational: enough working precision for len digits
    guard = nlen * math.log2(base) + 64
    e_est = sp.N(sp.log(v, base), 20) if v != 0 else 0
    work = int(guard + abs(float(e_est)) * math.log2(base)) + 10
    val = sp.N(v, int(work / 3.32) + 10)
    q = Fraction(sp.Rational(val).p, sp.Rational(val).q)
    digs, e = _digits_fixed(q, base, nlen, top)
    return Expr("List", R(digs), e)


# ───────────────────────────── Rescale, Clip, Unitize, RealAbs ─────────────────────────────

def _numexpr(x, name):
    v = sym(x, name)
    return v


@fn("Rescale", area=AREA, status="added",
    notes="Rescale[x, {min, max}] = (x - min)/(max - min); Rescale[x, {min, max}, {ymin, ymax}]; Rescale[list] "
          "uses Min[list] and Max[list]; works symbolically"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _rescale(x: Expr, rng: Expr = None, out: Expr = None) -> Expr:
    if rng is None:
        if not is_list(x):
            raise ValueError("Rescale[list] needs a list")
        vals = [sym(a, "Rescale") for a in _flat(x)]
        lo, hi = sp.Min(*vals), sp.Max(*vals)
        rng = Expr("List", R(lo), R(hi))
    lo_hi = list_arg(rng, "Rescale", "range")
    if len(lo_hi) != 2:
        raise ValueError("Rescale: range must be {min, max}")
    lo, hi = (sym(v, "Rescale") for v in lo_hi)
    if out is None:
        y0, y1 = sp.Integer(0), sp.Integer(1)
    else:
        yy = list_arg(out, "Rescale", "output range")
        y0, y1 = (sym(v, "Rescale") for v in yy)
    if (hi - lo).is_zero:
        raise ValueError("Rescale: empty range")

    def one(e):
        if is_list(e):
            return Expr("List", *(one(a) for a in e.args))
        v = sym(e, "Rescale")
        r = y0 + (v - lo) * (y1 - y0) / (hi - lo)
        return _num_like(sp.simplify(r) if not has_float(r) else r, e, rng, out)
    return one(x)


def _flat(x):
    for a in x.args:
        if is_list(a):
            yield from _flat(a)
        else:
            yield a


def _num_like(v, *inputs):
    from .core import numeric_out
    return numeric_out(v, *[i for i in inputs if i is not None])


def _clip_one(x, lo, hi, vlo, vhi):
    v = sym(x, "Clip")
    if not v.is_number:
        raise Unevaluated()
    if v.is_real is False:
        raise ValueError("Clip: argument must be real")
    if v < lo:
        return vlo
    if v > hi:
        return vhi
    return x


@fn("Clip", area=AREA, status="added",
    notes="Clip[x] clips to [-1, 1]; Clip[x, {min, max}]; Clip[x, {min, max}, {vmin, vmax}] gives vmin below "
          "and vmax above; threads over lists; symbolic x stays unevaluated"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _clip(x: Expr, rng: Expr = None, vals: Expr = None) -> Expr:
    if rng is None:
        lo, hi = sp.Integer(-1), sp.Integer(1)
        rlo, rhi = -1, 1
    else:
        pr = list_arg(rng, "Clip", "range")
        lo, hi = (sym(v, "Clip") for v in pr)
        rlo, rhi = pr
    if vals is not None:
        vlo, vhi = list_arg(vals, "Clip", "clip values")
    else:
        vlo, vhi = rlo, rhi

    def one(e):
        if is_list(e):
            return Expr("List", *(one(a) for a in e.args))
        return _clip_one(e, lo, hi, vlo, vhi)
    if not is_list(x):
        v = sym(x, "Clip")
        if not v.is_number:
            raise Unevaluated()
    return one(x)


@fn("Unitize", area=AREA, status="added",
    notes="Unitize[x] is 0 for x == 0 and 1 otherwise (exact integers, also for 0.); Unitize[x, dx] is 0 when "
          "Abs[x] < dx; threads over lists"
          ". Check: reference (differential cases in tests/library).")
def _unitize(x: Expr, dx: Expr = None) -> Expr:
    if is_list(x):
        return Expr("List", *(_unitize(a, dx) for a in x.args))
    v = sym(x, "Unitize")
    if not v.is_number:
        raise Unevaluated()
    if dx is None:
        return 0 if v.is_zero else 1
    d = number(dx, "Unitize", "tolerance")
    return 0 if abs(v) < d else 1


@fn("RealAbs", area=AREA, status="added", listable=True,
    notes="absolute value for real numbers; stays unevaluated for symbolic and non-real arguments"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _realabs(x: Expr) -> Expr:
    v = sym(x, "RealAbs")
    if not v.is_number or v.is_real is False:
        raise Unevaluated()
    from .core import numeric_out
    return numeric_out(sp.Abs(v), x)


# ───────────────────────────── number parts and number tests ─────────────────────────────

def _numden(x, which):
    v = sym(x, which)
    if isinstance(v, sp.Float):
        return x if which == "Numerator" else 1
    if v.is_Rational:
        return R(v.p if which == "Numerator" else v.q)
    if v.is_number and v.is_real is False and all(t.is_Rational for t in v.as_real_imag()):
        # Gaussian rational: numerator/denominator of the complex number as a whole
        den = sp.ilcm(*(sp.Rational(t).q for t in v.as_real_imag()))
        return R(sp.expand(v * den)) if which == "Numerator" else R(den)
    n, d = sp.fraction(sp.together(v) if not v.is_Add else v)
    if v.is_Add:
        n, d = v, sp.Integer(1)                     # Wolfram does not combine a sum over a common denominator
    return R(n if which == "Numerator" else d)


@fn("Numerator", area=AREA, status="added", listable=True,
    notes="numerator of a rational number or of a product with negative exponents (Numerator[x^2 y^-3] = x^2); "
          "sums are not combined over a common denominator (Numerator[a/b + c/d] is the whole sum)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _numerator(x: Expr) -> Expr:
    return _numden(x, "Numerator")


@fn("Denominator", area=AREA, status="added", listable=True,
    notes="denominator of a rational number or of the factors with negative exponents (Denominator[x/y^2] = "
          "y^2); 1 for sums and approximate numbers"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _denominator(x: Expr) -> Expr:
    return _numden(x, "Denominator")


def _is_complex_num(x, pred):
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float, complex, sp.Float)):
        return pred(x)
    if isinstance(x, Expr):
        from ..formatting import rational_value
        if rational_value(x) is not None:
            return pred(sp.Rational(1, 2))
        if x.head in ("Plus", "Times") and len(x.args) == 2:
            # Complex numbers are Plus[a, Times[b, I]] or Times[b, I]
            try:
                v = sym(x, "NumberQ")
            except Exception:
                return False
            if v.is_number and v.is_real is False and all(t.is_Rational or t.is_Float for t in v.as_real_imag()):
                return pred(complex(v)) if has_float(x) else pred(sp.Rational(1, 2))
        return False
    if isinstance(x, Symbol):
        return x.name == "I" and pred(sp.Rational(1, 2))
    return False


@fn("ExactNumberQ", area=AREA, status="added",
    notes="True for integers, rationals and exact complex numbers; False for approximate numbers and for "
          "numeric expressions such as Pi or Sqrt[2] (which are not numbers)"
          ". Check: reference (differential cases in tests/library).")
def _exactq(x: Expr) -> Boolean:
    return _is_complex_num(x, lambda v: not isinstance(v, (float, complex, sp.Float)))


@fn("InexactNumberQ", area=AREA, status="added",
    notes="True for approximate (machine or arbitrary-precision) real and complex numbers"
          ". Check: reference (differential cases in tests/library).")
def _inexactq(x: Expr) -> Boolean:
    return _is_complex_num(x, lambda v: isinstance(v, (float, complex, sp.Float)))


@fn("MachineNumberQ", area=AREA, status="added",
    notes="True for machine-precision real and complex numbers only"
          ". Check: reference (differential cases in tests/library).")
def _machineq(x: Expr) -> Boolean:
    return _is_complex_num(x, lambda v: isinstance(v, (float, complex)) or
                           (isinstance(v, sp.Float) and v._prec <= 53))


def _arg(v):
    if v.is_zero:
        return sp.Integer(0)
    return sp.arg(v)


@fn("Arg", area=AREA, status="added", listable=True,
    notes="argument in (-Pi, Pi]; Arg[0] = 0 (Wolfram); Arg[-1] = Pi; exact for exact input"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _arg_impl(z: Expr) -> Expr:
    from .core import numeric_out
    v = sym(z, "Arg")
    if not v.is_number:
        raise Unevaluated()
    return numeric_out(_arg(v), z)


@fn("ReIm", area=AREA, status="added", listable=True, notes="{Re[z], Im[z]}"
                                                            ". Check: exact (exact input gives exact results), "
                                                            "reference (differential cases in tests/library).")
def _reim(z: Expr) -> ListOf[Expr]:
    from .core import numeric_out
    v = sym(z, "ReIm")
    if not v.is_number:
        return Expr("List", Expr("Re", z), Expr("Im", z))
    re, im = v.as_real_imag()
    return Expr("List", numeric_out(re, z), numeric_out(im, z))


@fn("AbsArg", area=AREA, status="added", listable=True, notes="{Abs[z], Arg[z]}"
                                                              ". Check: exact (exact input gives exact results), "
                                                              "reference (differential cases in tests/library).")
def _absarg(z: Expr) -> ListOf[Expr]:
    from .core import numeric_out
    v = sym(z, "AbsArg")
    if not v.is_number:
        raise Unevaluated()
    return Expr("List", numeric_out(sp.Abs(v), z), numeric_out(_arg(v), z))


def _part(name, f):
    @fn(name, area=AREA, status="added", listable=True,
        notes=f"{'real' if name == 'Re' else 'imaginary'} part (Wolfram name; RealPart/ImaginaryPart are the legacy "
              "MikoshiLang names); exact for exact input, 0 (exact) for the imaginary part of a real")
    def impl(z: Expr) -> Expr:
        from .core import numeric_out
        v = sym(z, name)
        if not v.is_number:
            raise Unevaluated()
        out = f(v)
        if name == "Im" and v.is_real:
            return 0
        return numeric_out(out, z)
    return impl


_part("Re", sp.re)
_part("Im", sp.im)
