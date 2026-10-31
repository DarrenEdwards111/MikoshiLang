"""Elementary functions: trigonometric, hyperbolic, exponential, logarithmic,
rounding and complex-number parts.  Exact input gives exact output (SymPy);
machine-number input gives machine-number output."""
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, S, R, L, numeric_out, sym, number, is_list,
                   has_float, is_numeric_value)

AREA = "elementary"


def _unary(name, f, notes="verified against SymPy/mpmath", status="verified"):
    @fn(name, area=AREA, status=status, notes=notes, listable=True)
    def impl(z):
        v = f(sym(z, name))
        if name in _INT_VALUED and isinstance(v, sp.Integer):
            return R(v)          # Floor[2.7] = 2, Sign[-3.5] = -1 (integers, as in Wolfram)
        return numeric_out(v, z, keep_zero=name not in _EXACT_ZERO)
    return impl


_INT_VALUED = {"Sign", "Floor", "Ceiling"}
_EXACT_ZERO = {"ImaginaryPart"}      # Im[2.5] = 0 exactly in Wolfram; Sin[0.] = 0. is inexact


for _name, _f in [
    ("Sin", sp.sin), ("Cos", sp.cos), ("Tan", sp.tan), ("Cot", sp.cot), ("Sec", sp.sec), ("Csc", sp.csc),
    ("Sinh", sp.sinh), ("Cosh", sp.cosh), ("Tanh", sp.tanh), ("Coth", sp.coth), ("Sech", sp.sech),
    ("Csch", sp.csch), ("ArcSin", sp.asin), ("ArcCos", sp.acos), ("ArcCot", sp.acot), ("ArcSec", sp.asec),
    ("ArcCsc", sp.acsc), ("ArcSinh", sp.asinh), ("ArcCosh", sp.acosh), ("ArcTanh", sp.atanh),
    ("Exp", sp.exp), ("Sqrt", sp.sqrt), ("Abs", sp.Abs), ("Conjugate", sp.conjugate), ("RealPart", sp.re), ("ImaginaryPart", sp.im),
]:
    _unary(_name, _f)


@fn("ArcTan", area=AREA, status="fixed", listable=True,
    notes="ArcTan[x, y] was atan2(x, y); Wolfram's ArcTan[x, y] is the angle of the point (x, y), atan2(y, x)")
def _arctan1(z):
    return numeric_out(sp.atan(sym(z, "ArcTan")), z, keep_zero=True)


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
    return numeric_out(_log(sym(z, "Log")), z, keep_zero=True)


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


@fn("Root", area=AREA, status="fixed",
    notes="Wolfram Root[f, k]: the k-th root of the polynomial f[x] (f a pure function), ordered as Wolfram: real "
          "roots ascending, then complex roots by real part, then |Im|, conjugate pairs adjacent (negative imaginary "
          "part first); SymPy's CRootOf index differs and is remapped. The polynomial is reduced to the irreducible factor that has "
          "the root; roots of linear, quadratic and binomial (x^n - a) factors become radicals, other roots stay "
          "Root[minpoly &, j] and evaluate numerically with N. Approximate coefficients give a numeric root. "
          "Legacy Root[a, n] meant a^(1/n) (now Surd/Power)")
def _root(f, k):
    if not (isinstance(f, Expr) and f.head == "Function" and len(f.args) == 1):
        raise ValueError("Root[f, k]: f must be a pure function such as #^3 - 2 &")
    from ..math_ops import root_to_sympy, _ROOT_VAR as x
    kk = sym(k, "Root")
    if not (kk.is_Integer and kk >= 1):
        raise ValueError("Root[f, k]: k must be a positive integer")
    from ..kernel import replace_slots
    body = sp.expand(sym(replace_slots(f.args[0], (Symbol(x.name),)), "Root"))
    if body.free_symbols - {x}:
        raise Unevaluated()
    if not body.is_polynomial(x) or sp.degree(body, x) < 1:
        raise ValueError("Root[f, k]: f[x] must be a non-constant polynomial in x")
    n = int(sp.degree(body, x))
    if kk > n:
        raise ValueError(f"Root[f, k]: k = {kk} exceeds the degree {n}")
    if body.has(sp.Float):
        roots = sp.Poly(body, x).nroots(n=30)
        roots = sorted(roots, key=lambda r: (0 if abs(sp.im(r)) < 1e-25 else 1, float(sp.re(r)), abs(float(sp.im(r))),
                                             float(sp.im(r))))
        r = roots[int(kk) - 1]
        return R(sp.Float(sp.re(r), 15) if abs(sp.im(r)) < 1e-25 else sp.Float(sp.re(r), 15) + sp.I * sp.Float(sp.im(r), 15))
    if any(not c.is_rational for c in sp.Poly(body, x).all_coeffs()):
        raise Unevaluated()
    from ..math_ops import wolfram_root_order, root_values
    order0, vals0 = root_values(body, x)
    tv = vals0[order0[int(kk) - 1]]
    # find the irreducible factor containing the root and the root's (Wolfram) index within it
    for fac, _mult in sp.factor_list(body, x)[1]:
        m = int(sp.degree(fac, x))
        order, vals = root_values(fac, x)
        for j in range(m):
            if abs(vals[order[j]] - tv) < 1e-12 * max(1.0, abs(tv)):
                lead = sp.Poly(fac, x).LC()
                fac = sp.expand(fac / lead) if lead < 0 else fac
                terms = len(sp.Poly(fac, x).terms())
                if m <= 2 or terms == 2:
                    for rv in sp.roots(sp.Poly(fac, x), multiple=True):
                        if abs(complex(sp.N(rv, 30)) - tv) < 1e-12 * max(1.0, abs(tv)):
                            return R(sp.nsimplify(rv) if rv.is_Rational else sp.radsimp(rv))
                from ..kernel import substitute
                body_out = substitute(R(fac), {x.name: Expr("Slot", 1)})
                return Expr("Root", Expr("Function", body_out), j + 1)
    raise ArithmeticError("Root: could not isolate the requested root")


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
    if v in (sp.oo, -sp.oo):
        return R(v)
    if v.is_real is False:
        re, im = sp.re(v), sp.im(v)
        return R(_round_real(re) + sp.I * _round_real(im))
    if v in (sp.oo, -sp.oo) or v is sp.zoo:
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
    if v in (sp.oo, -sp.oo):
        return R(v)
    k = _round_real(v / step)
    return R(k * step)            # a Round[x/a]: exact when a is exact (Round[7.3, 2] = 8), else inexact like a


def _cfloor(v, mode="floor"):
    """Floor/Ceiling/Round of a SymPy number, applied separately to real and imaginary parts (Wolfram)."""
    if v in (sp.oo, -sp.oo):
        return v
    if v.is_real is False:
        re, im = sp.re(v), sp.im(v)
        return _cfloor(re, mode) + sp.I * _cfloor(im, mode)
    if v in (sp.oo, -sp.oo):
        return v
    if mode == "round":
        return _round_real(v)
    q = _as_fraction(v)
    if q is None:
        if v.is_Float:
            q = Fraction(float(v)) if v._prec <= 53 else Fraction(sp.Rational(v).p, sp.Rational(v).q)
        else:
            f = sp.floor(v) if mode == "floor" else sp.ceiling(v)
            if f.is_Integer:
                return f
            raise Unevaluated()
    if mode == "floor":
        return sp.Integer(q.numerator // q.denominator)
    return sp.Integer(-((-q.numerator) // q.denominator))


def _rounding(name, mode):
    @fn(name, area=AREA, status="fixed", listable=True,
        notes=f"{name}[x] and {name}[x, a] (a {name}[x/a], exact when a is exact); complex numbers are rounded "
              "in their real and imaginary parts and give Gaussian integers (legacy gave floats: Floor[2.5 + 3.7 I] "
              "= 2. + 3. I, and had no step form)")
    def one(x):
        if isinstance(x, (Symbol, Expr)) and not is_numeric_value(x):
            raise Unevaluated()
        v = number(x, name, real=False)
        if v is sp.zoo or v is sp.nan:
            raise Unevaluated()
        return R(_cfloor(v, mode))

    @fn(name, area=AREA, status="fixed", listable=True)
    def two(x, a):
        v = number(x, name, real=False)
        step = number(a, name, "rounding step", real=False)
        if step == 0:
            raise ValueError(f"{name}: rounding step must be non-zero")
        if v in (sp.oo, -sp.oo):
            return R(v)
        return R(_cfloor(v / step, mode) * step)


_rounding("Floor", "floor")
_rounding("Ceiling", "ceiling")


@fn("Sign", area=AREA, status="fixed", listable=True,
    notes="Sign[z] = z/Abs[z] for complex z (legacy left Sign[3 + 4 I] unevaluated); integer -1, 0, 1 for reals")
def _sign(z):
    v = sym(z, "Sign")
    if v.is_number and v.is_real is False and v.is_finite:
        return numeric_out(sp.nsimplify(v / sp.Abs(v)) if not has_float(z) else v / sp.Abs(v), z)
    out = sp.sign(v)
    if isinstance(out, sp.Integer):
        return R(out)
    return numeric_out(out, z)


@fn("Mod", area=AREA, status="fixed", listable=True,
    notes="Mod[m, n] = m - n Floor[m/n] (sign of n); Mod[m, n, d] uses the offset d (result in [d, d + n)); "
          "complex m or n use Round (Gaussian remainder, Mod[5 + 3 I, 2] = 1 - I); Mod[m, 0] raises. The offset "
          "form and complex arguments were missing")
def _mod(m, n, d=0):
    sm, sn, sd = sym(m, "Mod"), sym(n, "Mod"), sym(d, "Mod")
    if sn == 0:
        raise ValueError("Mod: modulus must be non-zero")
    if all(v.is_number for v in (sm, sn, sd)):
        if sm.is_real is False or sn.is_real is False or sd.is_real is False:
            q = _cfloor(sp.expand((sm - sd) / sn), "round")
        else:
            q = _cfloor((sm - sd) / sn if not has_float(m) and not has_float(n) else sp.sympify((sm - sd) / sn),
                        "floor")
        return numeric_out(sp.expand(sm - sn * q), m, n, d)
    if d == 0:
        return numeric_out(sp.Mod(sm, sn), m, n)
    raise Unevaluated()


@fn("Quotient", area=AREA, status="fixed", listable=True,
    notes="Quotient[m, n] = Floor[m/n] as an integer (legacy gave Quotient[7.5, 2] = 3.); Quotient[m, n, d] = "
          "Floor[(m - d)/n]; complex arguments use Round")
def _quotient(m, n, d=0):
    sm, sn, sd = sym(m, "Quotient"), sym(n, "Quotient"), sym(d, "Quotient")
    if sn == 0:
        raise ValueError("Quotient: divisor must be non-zero")
    if not all(v.is_number for v in (sm, sn, sd)):
        raise Unevaluated()
    if sm.is_real is False or sn.is_real is False:
        return R(_cfloor(sp.expand((sm - sd) / sn), "round"))
    return R(_cfloor((sm - sd) / sn, "floor"))


@fn("QuotientRemainder", area=AREA, status="added", listable=True,
    notes="{Quotient[m, n], Mod[m, n]}")
def _quotrem(m, n):
    return Expr("List", _quotient(m, n), _mod(m, n))


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
