"""Wolfram batch W10: special functions II.

q-functions, multiple zeta values and polylogarithms, Appell and Fox H functions, angular momentum
coefficients, waveforms, generalized functions and mathematical constants.

Every function follows the Wolfram Language reference page (definition, argument order, branch cuts and
special values).  Approximate arguments are evaluated with mpmath at the lowest input precision (machine
numbers give machine numbers); exact arguments give exact values where Wolfram gives them and otherwise stay
unevaluated, and N[f[exact args], p] evaluates to p digits.  Where a case cannot be computed reliably the call
raises a clear error rather than returning a doubtful number.
"""

from __future__ import annotations
import inspect

import mpmath
import sympy as sp

from ..expr import Expr, Symbol
from ..precision import input_precision
from .core import Unevaluated, fn, R, sym, is_list
from .wl_special import MP_NUMERIC, mp_eval, to_mp, from_mp, _dispatcher

AREA = "special functions II (W10)"
STATUS = "added"


def _as_expr(a):
    """Python lists (as produced by N on list arguments) -> List expressions."""
    if isinstance(a, (list, tuple)):
        return Expr("List", *(_as_expr(x) for x in a))
    return a


def _register(name, nargs, impl, notes, listable=True, numeric=True, hold=False):
    lo, hi = (nargs, nargs) if isinstance(nargs, int) else nargs
    for n in range(lo, hi + 1):
        def make(n):
            def f(*args: Expr) -> Expr:
                return impl(*(_as_expr(a) for a in args))
            f.__signature__ = inspect.Signature(
                [inspect.Parameter(f"a{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation="Expr")
                 for i in range(n)], return_annotation="Expr")
            f.__name__ = f"_{name}_{n}"
            return f
        fn(name, area=AREA, status=STATUS, notes=notes, listable=listable, hold=hold)(make(n))
    if numeric:
        MP_NUMERIC[name] = _dispatcher(name)


def special(name, nargs, mpf, exact=None, notes="", symbolic=None, check=None, listable=True):
    """Numeric special function: ``mpf(*mpmath args)`` for approximate input, ``exact(*sympy args)`` (SymPy
    value / MikoshiLang expression, or None to stay unevaluated) for exact input, ``symbolic`` for arguments
    with free symbols."""
    def impl(*args):
        vals = [sym(a, name) for a in args]
        if check is not None:
            check(*vals)
        if not all(v.is_number for v in vals):
            if symbolic is not None:
                out = symbolic(*vals)
                if out is not None:
                    return out if isinstance(out, (Expr, Symbol)) else R(out)
            raise Unevaluated()
        prec = input_precision(args)
        if prec is None:
            if exact is not None:
                out = exact(*vals)
                if out is not None:
                    return out if isinstance(out, (Expr, Symbol)) else R(out)
            raise Unevaluated()
        return mp_eval(mpf, vals, prec)
    _register(name, nargs, impl, notes, listable=listable)
    return impl


def _clean(v, *funcs):
    """``v`` if it is free of the given SymPy function classes (and of unevaluated special functions)."""
    v = sp.sympify(v)
    if any(v.has(f) for f in funcs):
        return None
    return v


def _real(v):
    return v.is_number and v.is_real is True


def _num(v):
    """Exact or approximate SymPy value to an mpmath number at current precision."""
    return to_mp(v)


# ───────────────────────────── inverse reciprocal hyperbolic functions ─────────────────────────────

_NICE = (sp.log, sp.acoth, sp.asech, sp.acsch, sp.atanh, sp.acosh, sp.asinh, sp.atan, sp.acos, sp.asin)


def _arc_exact(sfun):
    def ex(z):
        try:
            v = sfun(z)
        except Exception:
            return None
        if isinstance(v, type(sfun(sp.Symbol("t")))) and v.func == sfun:
            return None
        return _clean(v, *_NICE)
    return ex


special("ArcCoth", 1, mpmath.acoth, _arc_exact(sp.acoth),
        notes="ArcCoth[z] = ArcTanh[1/z] (branch cut [-1, 1]); mpmath acoth; exact special values (ArcCoth[0] = "
              "I Pi/2, ArcCoth[Infinity] = 0, ArcCoth[1] = Infinity, ArcCoth[I] = -I Pi/4)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("ArcCsch", 1, mpmath.acsch, _arc_exact(sp.acsch),
        notes="ArcCsch[z] = ArcSinh[1/z]; mpmath acsch; exact special values (ArcCsch[I] = -I Pi/2, ArcCsch[0] = "
              "ComplexInfinity)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("ArcSech", 1, mpmath.asech, _arc_exact(sp.asech),
        notes="ArcSech[z] = ArcCosh[1/z]; mpmath asech; exact special values (ArcSech[1] = 0, ArcSech[0] = "
              "Infinity, ArcSech[2] = I Pi/3)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


# ───────────────────────────── Gudermannian, sigmoid, sinc ─────────────────────────────

def _gd(z):
    return 2 * mpmath.atan(mpmath.tanh(z / 2))


def _gd_exact(z):
    if z == 0:
        return sp.Integer(0)
    if z == sp.oo:
        return sp.pi / 2
    if z == -sp.oo:
        return -sp.pi / 2
    return None


special("Gudermannian", 1, _gd, _gd_exact,
        notes="Gudermannian[z] = 2 ArcTan[Tanh[z/2]] (reference definition; cuts from I Pi (2n + 1/2) to "
              "I Pi (2n + 3/2)); Gudermannian[0] = 0, Gudermannian[+-Infinity] = +-Pi/2"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _igd(z):
    return mpmath.log(mpmath.tan(z / 2 + mpmath.pi / 4))


def _igd_exact(z):
    if z == 0:
        return sp.Integer(0)
    if z == sp.pi / 2:
        return sp.oo
    if z == -sp.pi / 2:
        return -sp.oo
    return None


special("InverseGudermannian", 1, _igd, _igd_exact,
        notes="InverseGudermannian[z] = Log[Tan[z/2 + Pi/4]] (reference definition); "
              "InverseGudermannian[0] = 0, InverseGudermannian[+-Pi/2] = +-Infinity"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _sigmoid_exact(z):
    if z == 0:
        return sp.Rational(1, 2)
    if z == sp.oo:
        return sp.Integer(1)
    if z == -sp.oo:
        return sp.Integer(0)
    return None


special("LogisticSigmoid", 1, lambda z: 1 / (1 + mpmath.exp(-z)), _sigmoid_exact,
        notes="LogisticSigmoid[z] = 1/(1 + E^-z); LogisticSigmoid[0] = 1/2, LogisticSigmoid[Infinity] = 1, "
              "LogisticSigmoid[-Infinity] = 0"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _sinc_exact(z):
    if z == 0:
        return sp.Integer(1)
    if z in (sp.oo, -sp.oo):
        return sp.Integer(0)
    v = sp.sin(z) / z
    return _clean(v, sp.sin) if not v.has(sp.sin) else None


special("Sinc", 1, lambda z: mpmath.mpf(1) if z == 0 else mpmath.sin(z) / z, _sinc_exact,
        notes="Sinc[z] = Sin[z]/z, Sinc[0] = 1; exact where Sin has an exact value (Sinc[Pi] = 0, "
              "Sinc[Pi/2] = 2/Pi)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


# ───────────────────────────── generalized functions and waveforms ─────────────────────────────

def _real_value(v, name):
    """Real SymPy value of a numeric argument, or None if symbolic; complex numbers -> None (stay)."""
    if not v.is_number:
        return None
    if v.is_real is False:
        return None
    if v.is_real is None:
        try:
            if sp.im(sp.N(v, 30)) != 0:
                return None
        except Exception:
            return None
    return v


def _cmp(v, c):
    """Exact comparison v < c -> -1, == 0, > 1 (None if undecidable)."""
    d = sp.nsimplify(v - c) if isinstance(v, sp.Float) else sp.simplify(v - c)
    if d == 0:
        return 0
    if d.is_positive:
        return 1
    if d.is_negative:
        return -1
    try:
        x = sp.N(d, 50)
        if x > 0:
            return 1
        if x < 0:
            return -1
    except Exception:
        pass
    return None


def _ramp(x):
    v = sym(x, "Ramp")
    r = _real_value(v, "Ramp")
    if r is None:
        raise Unevaluated()
    c = _cmp(r, 0)
    if c is None:
        raise Unevaluated()
    return x if c >= 0 else (0.0 if input_precision([x]) == "m" else 0)


_register("Ramp", 1, _ramp, "Ramp[x] = x for real x >= 0 and 0 for real x < 0; complex and symbolic x stay "
          "unevaluated"
          ". Check: exact (exact input gives exact results), reference (differential cases in "
          "tests/library).", numeric=False)


def _abs_cmp(v, c):
    return _cmp(sp.Abs(v), c)


def _product_fn(name, value):
    """Multivariate product of a one-variable generalized function: value(v) -> 0, 1, SymPy expr or None."""
    def impl(*args):
        if not args:
            raise ValueError(f"{name}: at least one argument is expected")
        vals = []
        for a in args:
            v = _real_value(sym(a, name), name)
            vals.append(None if v is None else value(v))
        if any(isinstance(v, sp.Basic) and v == 0 for v in vals):
            return 0
        if any(v is None for v in vals):
            raise Unevaluated()
        out = sp.Integer(1)
        for v in vals:
            out *= v
        if out.is_Integer:
            return int(out)
        if input_precision(args) is not None:
            return R(sp.N(out)) if input_precision(args) == "m" else R(out)
        return R(out)
    return impl


def _pi_value(v):
    c = _abs_cmp(v, sp.Rational(1, 2))
    if c is None or c == 0:
        return None
    return sp.Integer(1) if c < 0 else sp.Integer(0)


def _lambda_value(v):
    c = _abs_cmp(v, 1)
    if c is None:
        return None
    if c >= 0:
        return sp.Integer(0)
    return 1 - sp.Abs(v)


def _comb_value(v):
    if isinstance(v, sp.Float):
        if float(v) == int(v):
            return None
        return sp.Integer(0)
    if v.is_integer:
        return None
    if v.is_integer is False:
        return sp.Integer(0)
    return None


_register("HeavisidePi", (1, 8), _product_fn("HeavisidePi", _pi_value),
          "HeavisidePi[x] = 1 for |x| < 1/2, 0 for |x| > 1/2, unevaluated at x = +-1/2 (reference); "
          "HeavisidePi[x1, x2, ...] is the product"
          ". Check: exact (exact input gives exact results), reference (differential cases in "
          "tests/library).", numeric=False)
_register("HeavisideLambda", (1, 8), _product_fn("HeavisideLambda", _lambda_value),
          "HeavisideLambda[x] = 1 - |x| for |x| < 1 and 0 otherwise (the convolution of two HeavisidePi); "
          "multivariate form is the product"
          ". Check: exact (exact input gives exact results), reference (differential cases in "
          "tests/library).", numeric=False)
_register("UnitTriangle", (1, 8), _product_fn("UnitTriangle", _lambda_value),
          "UnitTriangle[x] = Piecewise[{{x + 1, -1 <= x < 0}, {1 - x, 0 <= x <= 1}}] (0 otherwise); "
          "UnitTriangle[x1, ...] is the product"
          ". Check: exact (exact input gives exact results), reference (differential cases in "
          "tests/library).", numeric=False)
_register("DiracComb", (1, 8), _product_fn("DiracComb", _comb_value),
          "DiracComb[x] = Sum[DiracDelta[x - n], {n, -Infinity, Infinity}]: 0 for real non-integer x, "
          "unevaluated at integers and for symbolic x; multivariate form is the product"
          ". Check: exact (exact input gives exact results), reference (differential cases in "
          "tests/library).", numeric=False)


def _discrete_indicator(x, x1, us):
    from .. import kernel as K
    if not is_list(us):
        raise Unevaluated()
    items = list(us.args)
    if not any(K.same(x, u) for u in items) or not any(K.same(x1, u) for u in items):
        raise Unevaluated()
    return 1 if K.same(x, x1) else 0


_register("DiscreteIndicator", 3, _discrete_indicator,
          "DiscreteIndicator[x, x1, {u1, ...}] = 1 if x === x1, 0 if x is another ui; unevaluated if x or x1 is "
          "not one of the ui (reference)"
          ". Check: exact (exact input gives exact results), reference (differential "
          "cases in tests/library).", listable=False, numeric=False)


def _wave(name, shape, default):
    """Periodic waveform: shape(frac) on [0, 1) -> value in [-1, 1] or a selector for {y1, y2}."""
    def impl(*args):
        if len(args) == 1:
            yspec, x = default, args[0]
        else:
            yspec, x = args
            if not (is_list(yspec) and len(yspec.args) == 2):
                raise ValueError(f"{name}: the first argument must be a pair {{min, max}}")
        if is_list(x):
            return Expr("List", *(impl(*(args[:-1] + (e,))) for e in x.args))
        v = _real_value(sym(x, name), name)
        if v is None:
            raise Unevaluated()
        prec = input_precision([x])
        if prec is not None:
            vv = sp.Rational(v) if isinstance(v, sp.Float) else v
        else:
            vv = v
        f = vv - sp.floor(vv)
        y1, y2 = (sym(yspec.args[0], name), sym(yspec.args[1], name)) if yspec is not None else \
            ((0, 1) if name == "SawtoothWave" else (-1, 1))
        out = shape(f, sp.sympify(y1), sp.sympify(y2))
        if prec is not None and name != "SquareWave" and not input_precision([R(out)]):
            out = sp.Float(out, 15) if prec == "m" else sp.Float(out, precision=prec)
        return R(out)
    _register(name, (1, 2), impl, NOTES[name], listable=False, numeric=False)


def _square(f, y1, y2):
    return y2 if f < sp.Rational(1, 2) else y1


def _triangle(f, y1, y2):
    if f < sp.Rational(1, 4):
        t = 4 * f
    elif f < sp.Rational(3, 4):
        t = 2 - 4 * f
    else:
        t = 4 * f - 4
    return y1 + (y2 - y1) * (t + 1) / 2


def _sawtooth(f, y1, y2):
    return y1 + (y2 - y1) * f


NOTES = {
    "SquareWave": "SquareWave[x] = 1 for FractionalPart-type phase in [0, 1/2), -1 in [1/2, 1) (SquareWave[0] = 1, "
                  "SquareWave[1/2] = -1, reference PiecewiseExpand); SquareWave[{y1, y2}, x] gives y2 then y1"
                  ". Check: exact (exact input gives exact results), reference (differential cases in "
                  "tests/library).",
    "TriangleWave": "TriangleWave[x]: period 1, 0 at integers, 1 at 1/4, -1 at 3/4 (reference); "
                    "TriangleWave[{min, max}, x] rescales"
                    ". Check: exact (exact input gives exact results), reference (differential cases in "
                    "tests/library).",
    "SawtoothWave": "SawtoothWave[x] = x - Floor[x] (SawtoothWave[0] = 0); SawtoothWave[{min, max}, x] rescales"
                    ". Check: exact (exact input gives exact results), reference (differential cases in "
                    "tests/library).",
}
_wave("SquareWave", _square, None)
_wave("TriangleWave", _triangle, None)
_wave("SawtoothWave", _sawtooth, None)


# ───────────────────────────── mathematical constants ─────────────────────────────

class _WLConstant(sp.NumberSymbol):
    """A real Wolfram constant evaluated with mpmath (N[Glaisher, 50] etc.)."""
    is_real = True
    is_positive = True
    is_irrational = True
    is_number = True
    __slots__ = ()
    _wl_name = ""
    _mp = None
    _bounds = (0, 1)

    def _as_mpf_val(self, prec):
        with mpmath.workprec(prec + 20):
            return (+mpmath.mpf(type(self)._mp()))._mpf_

    def approximation_interval(self, number_cls):
        if issubclass(number_cls, sp.Integer):
            return (sp.Integer(self._bounds[0]), sp.Integer(self._bounds[1]))

    def _latex(self, printer):
        return self._wl_name

    def __str__(self):
        return self._wl_name

    __repr__ = __str__


def _constant(name, mpfun, bounds, notes):
    cls = type(name + "Constant", (_WLConstant,), {"__slots__": (), "_wl_name": name, "_mp": staticmethod(mpfun),
                                                   "_bounds": bounds})
    cls = sp.core.singleton.Singleton(cls.__name__, (cls,), {"__slots__": ()}) if False else cls
    inst = cls()
    from .. import math_ops
    math_ops._CONSTANTS[name] = inst
    from .core import record
    record(name, AREA, STATUS, notes)
    return inst


GLAISHER = _constant("Glaisher", lambda: mpmath.glaisher, (1, 2),
                     "Glaisher-Kinkelin constant A = 1.28242712910062...; symbolic constant, N[Glaisher, n] to any "
                     "precision (mpmath glaisher)")
KHINCHIN = _constant("Khinchin", lambda: mpmath.khinchin, (2, 3),
                     "Khinchin's constant K = 2.68545200106530...; symbolic constant, N[Khinchin, n] to any "
                     "precision (mpmath khinchin)")
GOLDEN_ANGLE = _constant("GoldenAngle", lambda: (3 - mpmath.sqrt(5)) * mpmath.pi, (2, 3),
                         "GoldenAngle = (3 - Sqrt[5]) Pi = 2.39996322972865...; symbolic constant, N to any precision")


def _champernowne(b):
    if float(b) != int(b) or b < 2:
        raise ValueError("ChampernowneNumber: the base must be an integer greater than 1")
    b = int(b)
    prec = mpmath.mp.prec
    x = mpmath.mpf(0)
    scale = mpmath.mpf(1)
    n = 1
    lim = mpmath.mpf(2) ** (-(prec + 10))
    while True:
        digits = len(mpmath.libmp.numeral(n, b)) if b <= 36 else len(_digits(n, b))
        scale = scale / mpmath.mpf(b) ** digits
        x += n * scale
        if scale * n < lim:
            break
        n += 1
    return x


def _digits(n, b):
    out = []
    while n:
        n, r = divmod(n, b)
        out.append(r)
    return out


def _champernowne_check(*vals):
    if len(vals) == 1 and vals[0].is_number:
        b = vals[0]
        if not (b.is_integer or (isinstance(b, sp.Float) and float(b) == int(b))) or b < 2:
            raise ValueError("ChampernowneNumber: the base must be an integer greater than 1")


def _champernowne_impl(*args):
    if not args:
        return Expr("ChampernowneNumber", 10)
    return _champernowne_numeric(*args)


_champernowne_numeric = special("ChampernowneNumber", 1, _champernowne, None, check=_champernowne_check,
                                notes="ChampernowneNumber[b] = 0.123456789101112... in base b (stays symbolic; N "
                                      "gives any precision); ChampernowneNumber[] is base 10")
_register("ChampernowneNumber", 0, _champernowne_impl, "ChampernowneNumber[] = ChampernowneNumber[10]")


# ───────────────────────────── harmonic-type numbers ─────────────────────────────

def _is_nonneg_int(v):
    return v.is_Integer and v >= 0


def _alt_harmonic(*args):
    name = "AlternatingHarmonicNumber"
    vals = [sym(a, name) for a in args]
    if not all(v.is_number for v in vals):
        raise Unevaluated()
    n = vals[0]
    r = vals[1] if len(vals) > 1 else sp.Integer(1)
    s = vals[2] if len(vals) > 2 else sp.Integer(1)
    prec = input_precision(args)
    if n.is_Integer:
        if n < 0:
            if prec is None:
                raise Unevaluated()
        else:
            if prec is None:
                return R(sp.Add(*[(-1) ** (i + 1) * s ** i / sp.Integer(i) ** r for i in range(1, int(n) + 1)]))
    if prec is None:
        raise Unevaluated()

    def f(n, r, s):
        sign = mpmath.power(-1, n)
        if len(vals) == 1:
            return mpmath.log(2) - sign * mpmath.lerchphi(-1, 1, 1 + n)
        if len(vals) == 2:
            return mpmath.power(2, -r) * ((mpmath.power(2, r) - 2) * mpmath.zeta(r) - sign * mpmath.zeta(r, (1 + n) / 2)
                                          + sign * mpmath.zeta(r, 1 + n / 2))
        return mpmath.power(-s, 1 + n) * mpmath.lerchphi(-s, r, 1 + n) - mpmath.polylog(r, -s)
    return mp_eval(f, [n, r, s], prec)


_register("AlternatingHarmonicNumber", (1, 3), _alt_harmonic,
          "AlternatingHarmonicNumber[n] = Sum[(-1)^(i+1)/i, {i, n}], [n, r] = Sum[(-1)^(i+1)/i^r], [n, r, s] = "
          "Sum[(-1)^(i+1) s^i/i^r]; exact for non-negative integer n, approximate arguments via the reference "
          "analytic continuations (LerchPhi / Hurwitz Zeta / PolyLog)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _harmonic_rs(k, r, s):
    return sp.Add(*[s ** i / sp.Integer(i) ** r for i in range(1, k + 1)])


def _hyper_harmonic(*args):
    name = "HyperHarmonicNumber"
    vals = [sym(a, name) for a in args]
    if not all(v.is_number for v in vals):
        if len(vals) == 2 and vals[0] == 1:
            return Expr("HarmonicNumber", args[1])
        raise Unevaluated()
    p, n = vals[0], vals[1]
    r = vals[2] if len(vals) > 2 else sp.Integer(1)
    s = vals[3] if len(vals) > 3 else sp.Integer(1)
    prec = input_precision(args)
    if _is_nonneg_int(p) and _is_nonneg_int(n) and int(n) <= 20000 and int(p) <= 200:
        p, n = int(p), int(n)
        if p == 0:
            if n == 0:
                return Symbol("ComplexInfinity") if len(vals) == 2 else Symbol("ComplexInfinity")
            out = s ** n / sp.Integer(n) ** r
        else:
            row = [sp.Integer(0)] + [s ** k / sp.Integer(k) ** r for k in range(1, n + 1)]   # level 0
            for _ in range(p):
                acc, new = sp.Integer(0), [sp.Integer(0)]
                for k in range(1, n + 1):
                    acc += row[k]
                    new.append(acc)
                row = new
            out = row[n]
        out = R(sp.nsimplify(out) if prec is None else out)
        return out
    if prec is None or len(vals) > 2:
        raise Unevaluated()
    if p.is_Integer and p <= 0:
        raise Unevaluated()

    def f(p, n):
        return mpmath.gamma(n + p) / (mpmath.gamma(n + 1) * mpmath.gamma(p)) * (mpmath.digamma(n + p) - mpmath.digamma(p))
    return mp_eval(f, [p, n], prec)


_register("HyperHarmonicNumber", (2, 4), _hyper_harmonic,
          "HyperHarmonicNumber[p, n]: p-fold cumulative sums of harmonic numbers (level 0 = 1/n); [p, n, r] and "
          "[p, n, r, s] with order r and decoration s (level 1 = HarmonicNumber[n, r, s] = Sum[s^k/k^r]); exact "
          "for non-negative integers p, n; HyperHarmonicNumber[1, n] = HarmonicNumber[n]; other approximate "
          "p, n via Gamma[n + p]/(n! Gamma[p]) (PolyGamma[n + p] - PolyGamma[p])"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _nested_sum(n, rs, ss, zero, one, power):
    """Sum over n >= n1 > n2 > ... > nk >= 1 of prod s_i^n_i / n_i^r_i (dynamic programming)."""
    k = len(rs)
    prev = [one] * (n + 1)          # depth beyond k: empty product
    for i in range(k - 1, -1, -1):
        cur = [zero] * (n + 1)
        acc = zero
        for m in range(1, n + 1):
            inner = prev[m - 1] if i < k - 1 else one
            acc = acc + power(ss[i], m) / power(m, rs[i]) * inner
            cur[m] = acc
        prev = cur
    return prev[n]


def _multiple_harmonic(*args):
    name = "MultipleHarmonicNumber"
    if len(args) == 1:
        return Expr("HarmonicNumber", args[0])
    n = args[0]
    if not is_list(args[1]) or (len(args) == 3 and not is_list(args[2])):
        raise ValueError(f"{name}: weights and decorations must be lists")
    rs = list(args[1].args)
    ss = list(args[2].args) if len(args) == 3 else [1] * len(rs)
    if len(ss) != len(rs):
        raise ValueError(f"{name}: the weight and decoration lists must have the same length")
    if not rs:
        return 1
    if isinstance(n, Symbol) and n.name == "Infinity":
        if len(args) == 2:
            return Expr("MultipleZeta", args[1])
        return Expr("MultiplePolyLog", args[1], args[2])
    nv = sym(n, name)
    if not nv.is_number:
        raise Unevaluated()
    if not _is_nonneg_int(nv):
        raise ValueError(f"{name}: the truncation index must be a non-negative integer")
    rv = [sym(x, name) for x in rs]
    sv = [sym(x, name) for x in ss]
    if not all(v.is_number for v in rv + sv):
        raise Unevaluated()
    N = int(nv)
    if N > 100000:
        raise ValueError(f"{name}: truncation index too large for direct summation")
    prec = input_precision(rs + ss)
    if prec is None:
        if N > 5000:
            raise ValueError(f"{name}: exact truncation index too large (at most 5000)")
        return R(sp.nsimplify(_nested_sum(N, rv, sv, sp.Integer(0), sp.Integer(1),
                                          lambda b, e: sp.Integer(b) ** e if isinstance(b, int) else b ** e)))
    return mp_eval(lambda *v: _nested_sum(N, v[:len(rv)], v[len(rv):], mpmath.mpf(0), mpmath.mpf(1), mpmath.power),
                   rv + sv, prec)


_register("MultipleHarmonicNumber", (1, 3), _multiple_harmonic,
          "MultipleHarmonicNumber[n, {r1, ..., rk}, {s1, ..., sk}] = Sum over n >= n1 > ... > nk >= 1 of "
          "s1^n1 ... sk^nk/(n1^r1 ... nk^rk) (exact for exact weights); MultipleHarmonicNumber[n] = "
          "HarmonicNumber[n]; n = Infinity gives MultipleZeta / MultiplePolyLog"
          ". Check: exact (exact input gives exact results), reference (differential "
          "cases in tests/library).", listable=False, numeric=False)


# ───────────────────────────── Bell and Nørlund polynomials, PascalBinomial ─────────────────────────────

def _bell_partial(n, k, xs):
    """Partial Bell polynomial B_{n,k}(x1, ..., x_{n-k+1}) by the standard recurrence."""
    memo = {}

    def B(n, k):
        if (n, k) in memo:
            return memo[(n, k)]
        if n == 0 and k == 0:
            v = sp.Integer(1)
        elif n == 0 or k == 0 or k > n:
            v = sp.Integer(0)
        else:
            v = sp.Add(*[sp.binomial(n - 1, i - 1) * xs[i - 1] * B(n - i, k - 1) for i in range(1, n - k + 2)])
        memo[(n, k)] = v
        return v
    return sp.expand(B(n, k))


def _bell_general(rows):
    """Generalized Bell polynomial of a matrix: n-th derivative of f1(f2(...fm(t))) where row i holds the
    derivatives f_i', ..., f_i^(n) (Faa di Bruno applied recursively)."""
    n = len(rows[0])
    if len(rows) == 1:
        return rows[0][n - 1]
    inner = [_bell_general([r[:j] for r in rows[1:]]) for j in range(1, n + 1)]
    return sp.expand(sp.Add(*[rows[0][k - 1] * _bell_partial(n, k, inner) for k in range(1, n + 1)]))


def _belly(*args):
    name = "BellY"
    if len(args) == 1:
        m = args[0]
        if not (is_list(m) and m.args and all(is_list(r) for r in m.args)):
            raise Unevaluated() if isinstance(m, Symbol) else ValueError(f"{name}: expected a matrix")
        rows = [[sym(x, name) for x in r.args] for r in m.args]
        if len({len(r) for r in rows}) != 1 or not rows[0]:
            raise ValueError(f"{name}: the matrix must be rectangular and non-empty")
        return R(_bell_general(rows))
    if len(args) != 3:
        raise ValueError(f"{name}: expected BellY[n, k, {{x1, ...}}] or BellY[matrix]")
    n, k, xs = args
    nv, kv = sym(n, name), sym(k, name)
    if not (nv.is_Integer and kv.is_Integer):
        if nv.is_number and kv.is_number:
            raise ValueError(f"{name}: n and k must be integers")
        raise Unevaluated()
    if not is_list(xs):
        raise Unevaluated()
    if xs.args and all(is_list(r) for r in xs.args):
        raise ValueError(f"{name}: the generalized partial form BellY[n, k, matrix] is not supported")
    n, k = int(nv), int(kv)
    if n < 0 or k < 0:
        raise ValueError(f"{name}: n and k must be non-negative")
    vals = [sym(x, name) for x in xs.args]
    need = n - k + 1
    if 1 <= k <= n and len(vals) < need:
        raise ValueError(f"{name}: BellY[{n}, {k}, list] needs at least {need} variables")
    return R(_bell_partial(n, k, vals + [sp.Integer(0)] * max(0, need - len(vals))))


_register("BellY", (1, 3), _belly,
          "BellY[n, k, {x1, ..., x(n-k+1)}] partial Bell polynomial (standard recurrence); BellY[m] generalized "
          "Bell polynomial of a matrix (n-th derivative of a composition, rows = derivatives of each function); "
          "BellY[n, k, matrix] raises (not supported)"
          ". Check: exact (exact input gives exact results), reference (differential "
          "cases in tests/library).", listable=False, numeric=False)


def _series_log1(c, N):
    """log of a power series with c[0] == 1, to order N."""
    d = [sp.Integer(0)] * (N + 1)
    for n in range(1, N + 1):
        d[n] = sp.expand(c[n] - sp.Add(*[sp.Rational(k, n) * d[k] * c[n - k] for k in range(1, n)]))
    return d


def _series_exp(d, N):
    """exp of a power series with d[0] == 0, to order N."""
    e = [sp.Integer(0)] * (N + 1)
    e[0] = sp.Integer(1)
    for n in range(1, N + 1):
        e[n] = sp.expand(sp.Add(*[sp.Rational(k, n) * d[k] * e[n - k] for k in range(1, n + 1)]))
    return e


def _norlund(n, a, x=None):
    g = [sp.bernoulli(k) / sp.factorial(k) for k in range(n + 1)]
    g[1] = sp.Rational(-1, 2)                   # t/(e^t - 1) convention
    L = _series_log1(g, n)
    d = [sp.expand(a * L[k]) for k in range(n + 1)]
    if x is not None and n >= 1:
        d[1] = sp.expand(d[1] + x)
    e = _series_exp(d, n)
    return sp.expand(sp.factorial(n) * e[n])


def _norlundb(*args):
    name = "NorlundB"
    vals = [sym(a, name) for a in args]
    n = vals[0]
    if not n.is_number:
        raise Unevaluated()
    if not (n.is_Integer and n >= 0):
        raise Unevaluated()
    if int(n) > 200:
        raise ValueError(f"{name}: degree too large")
    out = _norlund(int(n), vals[1], vals[2] if len(vals) > 2 else None)
    prec = input_precision(args[1:])
    if prec is not None and out.is_number:
        return R(sp.N(out, 15 if prec == "m" else mpmath.libmp.prec_to_dps(prec)))
    return R(out)


_register("NorlundB", (2, 3), _norlundb,
          "NorlundB[n, a] Norlund polynomials and NorlundB[n, a, x] generalized Bernoulli polynomials from "
          "(t/(E^t - 1))^a E^(x t) = Sum[NorlundB[n, a, x] t^n/n!] (exact power-series arithmetic) for "
          "non-negative integer n; NorlundB[n, 1] = BernoulliB[n]"
          ". Check: exact (exact input gives exact results), reference (differential cases in "
          "tests/library).", numeric=False)


def _pascal_binomial(n, m):
    name = "PascalBinomial"
    nv, mv = sym(n, name), sym(m, name)
    if nv.is_Integer and nv < 0:
        if not mv.is_number:
            raise Unevaluated()
        if mv.is_Integer:
            if mv < 0:
                return 0
            return R(sp.ff(nv, int(mv)) / sp.factorial(int(mv)))
        if input_precision([m]) is None and mv.is_integer is False:
            return Symbol("ComplexInfinity")
        raise Unevaluated()
    return Expr("Binomial", n, m)


_register("PascalBinomial", 2, _pascal_binomial,
          "PascalBinomial[n, m] agrees with Binomial except for negative integer n, where it is the iterated "
          "limit that preserves Pascal's identity: n(n-1)...(n-m+1)/m! for integer m >= 0, 0 for negative "
          "integer m, ComplexInfinity for non-integer m (reference definition)"
          ". Check: exact (exact input gives exact results), reference (differential cases in "
          "tests/library).", numeric=False)


# ───────────────────────────── Ramanujan tau and its L-function ─────────────────────────────

def _sigma_table(N):
    s = [0] * (N + 1)
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            s[m] += d
    return s


def _tau_prime(p, sig):
    """tau(n) = n^4 sigma(n) - 24 Sum_{k=1}^{n-1} (35 k^4 - 52 k^3 n + 18 k^2 n^2) sigma(k) sigma(n - k)."""
    n = p
    acc = 0
    for k in range(1, n):
        acc += (35 * k ** 4 - 52 * k ** 3 * n + 18 * k * k * n * n) * sig[k] * sig[n - k]
    return n ** 4 * sig[n] - 24 * acc


def ramanujan_tau(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    f = sp.factorint(n)
    if max(f) > 200000:
        raise ValueError("RamanujanTau: argument has a prime factor that is too large")
    sig = _sigma_table(max(f))
    out = 1
    for p, e in f.items():
        tp = _tau_prime(p, sig)
        a, b = 1, tp                     # tau(p^0), tau(p^1)
        for _ in range(e - 1):
            a, b = b, tp * b - p ** 11 * a
        out *= b
    return out


def _tau_list(N):
    """tau(1..N) from Delta = q prod (1 - q^n)^24 via eta^3 = Sum (-1)^m (2m+1) q^(m(m+1)/2)."""
    e3 = [0] * N
    m = 0
    while m * (m + 1) // 2 < N:
        e3[m * (m + 1) // 2] = (-1) ** m * (2 * m + 1)
        m += 1

    def mul(a, b):
        c = [0] * N
        for i, x in enumerate(a):
            if x:
                for j in range(N - i):
                    if b[j]:
                        c[i + j] += x * b[j]
        return c
    e6 = mul(e3, e3)
    e12 = mul(e6, e6)
    e24 = mul(e12, e12)
    return [0] + e24[:N]           # tau(n) = coefficient of q^(n-1) in eta^24 / q


def _tau(n):
    v = sym(n, "RamanujanTau")
    if not v.is_number:
        raise Unevaluated()
    if not v.is_Integer:
        raise ValueError("RamanujanTau: the argument must be an integer")
    return ramanujan_tau(int(v))


_register("RamanujanTau", 1, _tau,
          "RamanujanTau[n]: coefficients of Delta = q Prod[(1 - q^k)^24]; tau(p) from the divisor-sum identity "
          "n^4 sigma(n) - 24 Sum (35k^4 - 52k^3 n + 18k^2 n^2) sigma(k) sigma(n-k), multiplicativity and "
          "tau(p^(e+1)) = tau(p) tau(p^e) - p^11 tau(p^(e-1)); 0 for n <= 0"
          ". Check: exact (exact input gives exact results), reference (differential cases in "
          "tests/library).", numeric=False)


def _tau_L(s):
    """L(s) = Sum tau(n)/n^s via Lambda(s) = (2 Pi)^-s Gamma(s) L(s) = Lambda(12 - s) and incomplete gammas."""
    t = abs(mpmath.im(s))
    extra = int(t * 2.3) + 20
    with mpmath.workprec(mpmath.mp.prec + extra):
        twopi = 2 * mpmath.pi
        target = mpmath.mpf(2) ** (-(mpmath.mp.prec + 10))
        N = int((mpmath.mp.prec * 0.6931 + t) / 6.28) + 12
        taus = _tau_list(N + 1)
        lam = mpmath.mpf(0)
        for n in range(1, N + 1):
            x = twopi * n
            term = taus[n] * (mpmath.gammainc(s, x) * mpmath.power(x, -s)
                              + mpmath.gammainc(12 - s, x) * mpmath.power(x, s - 12))
            lam += term
            if n > 6 and abs(term) < target * (abs(lam) + mpmath.mpf(2) ** -40):
                break
        if s == 0 or (mpmath.im(s) == 0 and mpmath.re(s) <= 0 and mpmath.re(s) == int(mpmath.re(s))):
            return mpmath.mpf(0)            # trivial zeros at s = 0, -1, -2, ... (1/Gamma(s) = 0)
        return lam * mpmath.power(twopi, s) * mpmath.rgamma(s)


special("RamanujanTauL", 1, _tau_L,
        notes="RamanujanTauL[s] = Sum[RamanujanTau[n]/n^s] (Re s > 13/2), continued to all s through the "
              "functional equation Lambda(s) = (2 Pi)^-s Gamma(s) L(s) = Lambda(12 - s) (rapidly convergent "
              "incomplete-gamma series)")


def _tau_theta(t):
    return mpmath.im(mpmath.loggamma(6 + 1j * t)) - t * mpmath.log(2 * mpmath.pi) if mpmath.im(t) == 0 else \
        (-0.5j) * (mpmath.loggamma(6 + 1j * t) - mpmath.loggamma(6 - 1j * t)) - t * mpmath.log(2 * mpmath.pi)


special("RamanujanTauTheta", 1, _tau_theta, lambda t: sp.Integer(0) if t == 0 else None,
        notes="RamanujanTauTheta[t] = -I/2 (LogGamma[6 + I t] - LogGamma[6 - I t]) - t Log[2 Pi] (continuous "
              "branch; RamanujanTauTheta[9.22] = 1.40372 as in the reference); RamanujanTauTheta[0] = 0"
              ". Check: reference (differential cases in tests/library).")


def _tau_Z(t):
    return mpmath.exp(1j * _tau_theta(t)) * _tau_L(6 + 1j * t) if mpmath.im(t) != 0 else \
        mpmath.re(mpmath.exp(1j * _tau_theta(t)) * _tau_L(6 + 1j * t))


special("RamanujanTauZ", 1, _tau_Z,
        notes="RamanujanTauZ[t] = Exp[I RamanujanTauTheta[t]] RamanujanTauL[6 + I t] (real for real t)")


# ───────────────────────────── Zernike, Pauli, inert heads ─────────────────────────────

def _zernike(n, m, r):
    name = "ZernikeR"
    nv, mv, rv = sym(n, name), sym(m, name), sym(r, name)
    if nv.is_Integer and mv.is_Integer:
        N, M = int(nv), int(mv)
        if (N - M) % 2:
            return 0
        if 0 <= M <= N:
            out = sp.Add(*[(-1) ** k * sp.factorial(N - k) / (sp.factorial(k) * sp.factorial((N + M) // 2 - k)
                                                                 * sp.factorial((N - M) // 2 - k)) * rv ** (N - 2 * k)
                           for k in range((N - M) // 2 + 1)])
            out = sp.expand(out)
            prec = input_precision([r])
            if prec is not None and out.is_number:
                return R(sp.N(out, 15 if prec == "m" else mpmath.libmp.prec_to_dps(prec)))
            return R(out)
        raise Unevaluated()
    raise Unevaluated()


_register("ZernikeR", 3, _zernike,
          "ZernikeR[n, m, r]: explicit radial Zernike polynomial Sum[(-1)^k (n-k)!/(k! ((n+m)/2-k)! "
          "((n-m)/2-k)!) r^(n-2k)] for integers 0 <= m <= n, 0 when n - m is odd; non-integer orders stay "
          "unevaluated"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")

_PAULI = {
    0: [[1, 0], [0, 1]],
    1: [[0, 1], [1, 0]],
    2: [[0, -sp.I], [sp.I, 0]],
    3: [[1, 0], [0, -1]],
}


def _pauli(k: Expr, *opts: Option) -> Expr:
    from .wl_common import split_options
    _a, o = split_options(opts, {"TargetStructure", "WorkingPrecision"}, "PauliMatrix")
    if _a:
        raise ValueError("PauliMatrix: unexpected arguments")
    ts = o.get("TargetStructure")
    if ts is not None and not (isinstance(ts, str) and ts in ("Dense",)) and not (
            isinstance(ts, Symbol) and ts.name == "Automatic"):
        raise ValueError("PauliMatrix: only TargetStructure -> \"Dense\" is supported")
    if is_list(k):
        return Expr("List", *(_pauli(x, *opts) for x in k.args))
    v = sym(k, "PauliMatrix")
    if not v.is_number:
        raise Unevaluated()
    if not (v.is_Integer and 0 <= v <= 3):
        raise ValueError("PauliMatrix: the index must be 0, 1, 2 or 3")
    mat = sp.Matrix(_PAULI[int(v)])
    wp = o.get("WorkingPrecision")
    if wp is not None:
        if isinstance(wp, Symbol) and wp.name == "MachinePrecision":
            conv = lambda z: z if z == 0 else (complex(z) if sp.im(z) else float(z))
        else:
            d = sym(wp, "PauliMatrix")
            if not d.is_number or not d > 0:
                raise ValueError("PauliMatrix: invalid WorkingPrecision")
            conv = lambda z: z if z == 0 else R(sp.N(z, int(d)))
        return Expr("List", *(Expr("List", *(conv(z) for z in mat.row(i))) for i in range(2)))
    return R(mat)


fn("PauliMatrix", area=AREA, status=STATUS, listable=False,
   notes="PauliMatrix[k]: k = 1, 2, 3 Pauli spin matrices, PauliMatrix[0] = IdentityMatrix[2]; threads over "
         "lists; WorkingPrecision; TargetStructure -> \"Dense\" (other structures raise)"
         ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")(_pauli)


def _inert(name, notes, nargs=(0, 12)):
    def impl(*args):
        raise Unevaluated()
    _register(name, nargs, impl, notes, listable=False, numeric=False)


_inert("GeneralizedPower", "GeneralizedPower[f, x, k] represents Fold[f, Table[x, k]] (inert, as in the reference; "
       "typeset as a power)")
for _c, _rule in (("CombinatorS", "S x y z -> x z (y z)"), ("CombinatorK", "K x y -> x"), ("CombinatorI", "I x -> x"),
                  ("CombinatorB", "B x y z -> x (y z)"), ("CombinatorC", "C x y z -> x z y"),
                  ("CombinatorW", "W x y -> x y y"), ("CombinatorY", "Y f -> f (Y f)")):
    _inert(_c, f"{_c} represents the {_c[-1]} combinator ({_rule}); no transformations are applied automatically "
           f"(reference), so applications such as {_c}[x][y] stay unevaluated for rule-based reduction")


# ───────────────────────────── zeta relatives ─────────────────────────────

def _dbeta_exact(s):
    if s.is_Integer:
        n = int(s)
        if n == 0:
            return sp.Rational(1, 2)
        if n == 1:
            return sp.pi / 4
        if n == 2:
            return sp.Catalan
        if n < 0:
            return sp.euler(-n) / 2
        if n % 2 == 1:
            k = (n - 1) // 2
            return (-1) ** k * sp.euler(2 * k) * sp.pi ** n / (4 ** (k + 1) * sp.factorial(2 * k))
    return None


def _dbeta(s):
    if s == 1:
        return mpmath.pi / 4
    return mpmath.dirichlet(s, [0, 1, 0, -1])


special("DirichletBeta", 1, _dbeta, _dbeta_exact,
        notes="DirichletBeta[s] = Sum[(-1)^n/(2n + 1)^s, {n, 0, Infinity}] (analytically continued; mpmath dirichlet "
              "with the character mod 4); exact: beta(0) = 1/2, beta(1) = Pi/4, beta(2) = Catalan, odd s via Euler "
              "numbers, beta(-n) = EulerE[n]/2"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _dlambda_exact(s):
    if s == 1:
        return sp.zoo
    v = (1 - sp.Integer(2) ** (-s)) * sp.zeta(s)
    v = sp.simplify(v) if s.is_Integer else v
    return None if v.has(sp.zeta) else v


special("DirichletLambda", 1, lambda s: (1 - mpmath.power(2, -s)) * mpmath.zeta(s), _dlambda_exact,
        notes="DirichletLambda[s] = (1 - 2^-s) Zeta[s] = Sum[1/(2n + 1)^s, {n, 0, Infinity}]; exact where Zeta "
              "is (even positive and non-positive integers), DirichletLambda[1] = ComplexInfinity"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _primezeta(s):
    if mpmath.re(s) <= 0:
        raise ValueError("PrimeZetaP: the prime zeta function has a natural boundary at Re(s) = 0")
    return mpmath.primezeta(s)


special("PrimeZetaP", 1, _primezeta, lambda s: sp.oo if s == 1 else None,
        notes="PrimeZetaP[s] = Sum[1/Prime[k]^s] for Re(s) > 0 (mpmath primezeta; natural boundary Re(s) = 0 "
              "raises); PrimeZetaP[1] = Infinity"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("RiemannR", 1, mpmath.riemannr,
        notes="RiemannR[x] = Sum[MoebiusMu[n]/n LogIntegral[x^(1/n)]] (Gram series; mpmath riemannr)"
              ". Check: reference (differential cases in tests/library).")
special("RiemannSiegelTheta", 1, mpmath.siegeltheta, lambda t: sp.Integer(0) if t == 0 else None,
        notes="RiemannSiegelTheta[t] = Im[LogGamma[1/4 + I t/2]] - t Log[Pi]/2 (mpmath siegeltheta); value 0 at 0")
special("RiemannSiegelZ", 1, mpmath.siegelz,
        notes="RiemannSiegelZ[t] = Exp[I RiemannSiegelTheta[t]] Zeta[1/2 + I t] (mpmath siegelz)"
              ". Check: reference (differential cases in tests/library).")


def _xi(s):
    if s == 0 or s == 1:
        return mpmath.mpf(1) / 2
    if mpmath.re(s) < 0.5:
        s = 1 - s
    return s * (s - 1) / 2 * mpmath.power(mpmath.pi, -s / 2) * mpmath.gamma(s / 2) * mpmath.zeta(s)


def _xi_exact(s):
    if s == 0 or s == 1:
        return sp.Rational(1, 2)
    if s.is_Integer:
        n = int(s)
        if n <= 0:
            n = 1 - n
        if n % 2 == 0:
            m = sp.Integer(n)
            return sp.simplify(m * (m - 1) / 2 * sp.pi ** (-m / 2) * sp.gamma(m / 2) * sp.zeta(m))
    return None


special("RiemannXi", 1, _xi, _xi_exact,
        notes="RiemannXi[s] = s (s - 1) Pi^(-s/2) Gamma[s/2] Zeta[s]/2 with xi(s) = xi(1 - s); exact 1/2 at 0, 1 "
              "and closed forms at even integers (RiemannXi[2] = Pi/6)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _zetazero(k, t=None):
    if mpmath.im(k) != 0 or k != int(k) or k == 0:
        raise ValueError("ZetaZero: the index must be a non-zero integer")
    k = int(k)
    if t is not None:
        if mpmath.im(t) != 0 or t < 0 or k < 0:
            raise ValueError("ZetaZero: ZetaZero[k, t] is supported for positive k and real t >= 0")
        base = mpmath.nzeros(t)
        return mpmath.zetazero(base + k)
    z = mpmath.zetazero(abs(k))
    return z if k > 0 else mpmath.conj(z)


def _zetazero_check(*vals):
    k = vals[0]
    if k.is_number and input_precision([k]) is None and (not k.is_Integer or k == 0):
        raise ValueError("ZetaZero: the index must be a non-zero integer")


special("ZetaZero", (1, 2), _zetazero, None, check=_zetazero_check,
        notes="ZetaZero[k] represents the k-th zero 1/2 + I t of Zeta on the critical line (negative k: conjugate "
              "zeros), ZetaZero[k, t] the k-th zero above height t; stays symbolic for exact k, N gives any "
              "precision (mpmath zetazero / nzeros)")


def _log_barnes_g(z):
    """Analytic log G(z) on C minus (-Infinity, 0]: shift to large |z| with log G(z) = log G(z + N) -
    Sum LogGamma[z + j], then the Stirling-type asymptotic expansion of log G(w + 1)."""
    if mpmath.im(z) == 0 and mpmath.re(z) <= 0 and mpmath.re(z) == int(mpmath.re(z)):
        raise ValueError("LogBarnesG: logarithmic singularity at a non-positive integer")
    prec = mpmath.mp.prec
    R0 = max(10, (prec + 30) * 0.12)
    shift = 0
    while abs(z + shift) < R0 or mpmath.re(z + shift) < R0 / 2:
        shift += 1
    w = z + shift - 1                       # log G(w + 1) with w + 1 = z + shift
    with mpmath.workprec(prec + 30):
        lw = mpmath.log(w)
        val = w * w / 2 * lw - 3 * w * w / 4 + w / 2 * mpmath.log(2 * mpmath.pi) - lw / 12 + \
            mpmath.zeta(-1, derivative=1)
        k = 1
        eps = mpmath.mpf(2) ** (-(prec + 20))
        last = None
        while True:
            term = mpmath.bernoulli(2 * k + 2) / (4 * k * (k + 1) * mpmath.power(w, 2 * k))
            if last is not None and abs(term) > abs(last):
                break                       # asymptotic series: stop at the smallest term
            val += term
            last = term
            if abs(term) < eps * max(1, abs(val)):
                break
            k += 1
        for j in range(shift):
            val -= mpmath.loggamma(z + j)
    return val


def _lbg_exact(z):
    if z.is_Integer and z >= 1:
        g = sp.Integer(1)
        for k in range(1, int(z) - 1):
            g *= sp.factorial(k)
        return sp.log(g)
    return None


special("LogBarnesG", 1, _log_barnes_g, _lbg_exact,
        notes="LogBarnesG[z]: holomorphic logarithm of BarnesG with a branch cut along the negative real axis "
              "(recurrence with LogGamma plus the asymptotic expansion; log of Prod[k!, {k, n-2}] at positive "
              "integers); singular at non-positive integers"
              ". Check: reference (differential cases in tests/library).")


# ───────────────────────────── Dawson, Fresnel auxiliaries, Owen T, Marcum Q, Mittag-Leffler ─────────────────

def _dawson(z):
    return mpmath.sqrt(mpmath.pi) / 2 * mpmath.exp(-z * z) * mpmath.erfi(z)


special("DawsonF", 1, _dawson, lambda z: sp.Integer(0) if z in (0, sp.oo, -sp.oo) else None,
        notes="DawsonF[z] = E^-z^2 Integrate[E^t^2, {t, 0, z}] = Sqrt[Pi]/2 E^-z^2 Erfi[z]; 0 at 0 and +-Infinity"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _fresnel_aux(which):
    def f(z):
        extra = int(2 * mpmath.log(abs(z) + 2, 2) * 2) + 20
        with mpmath.workprec(mpmath.mp.prec + extra):
            S, C = mpmath.fresnels(z), mpmath.fresnelc(z)
            a = mpmath.pi * z * z / 2
            c, s = mpmath.cos(a), mpmath.sin(a)
            half = mpmath.mpf(1) / 2
            if which == "F":
                return (half - S) * c - (half - C) * s
            return (half - C) * c + (half - S) * s
    return f


special("FresnelF", 1, _fresnel_aux("F"),
        lambda z: sp.Rational(1, 2) if z == 0 else (sp.Integer(0) if z == sp.oo else None),
        notes="FresnelF[z] = (1/2 - FresnelS[z]) Cos[Pi z^2/2] - (1/2 - FresnelC[z]) Sin[Pi z^2/2] (extra working "
              "precision against cancellation); FresnelF[0] = 1/2, FresnelF[Infinity] = 0"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("FresnelG", 1, _fresnel_aux("G"),
        lambda z: sp.Rational(1, 2) if z == 0 else (sp.Integer(0) if z == sp.oo else None),
        notes="FresnelG[z] = (1/2 - FresnelC[z]) Cos[Pi z^2/2] + (1/2 - FresnelS[z]) Sin[Pi z^2/2]; FresnelG[0] = "
              "1/2, FresnelG[Infinity] = 0"
              ". Check: reference (differential cases in tests/library).")


def _owen(h, a):
    if mpmath.im(h) != 0 or mpmath.im(a) != 0:
        raise ValueError("OwenT: only real arguments are supported")
    f = lambda x: mpmath.exp(-h * h * (1 + x * x) / 2) / (1 + x * x)
    return mpmath.quad(f, [0, a]) / (2 * mpmath.pi)


def _owen_exact(h, a):
    if a == 0:
        return sp.Integer(0)
    if h == 0:
        return sp.atan(a) / (2 * sp.pi)
    return None


special("OwenT", 2, _owen, _owen_exact,
        notes="OwenT[h, a] = 1/(2 Pi) Integrate[E^(-h^2 (1 + x^2)/2)/(1 + x^2), {x, 0, a}] (mpmath quadrature, "
              "real arguments); OwenT[0, a] = ArcTan[a]/(2 Pi), OwenT[h, 0] = 0"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _marcum(m, a, b, b1=None):
    if b1 is not None:
        return _marcum(m, a, b) - _marcum(m, a, b1)
    for v in (m, a, b):
        if mpmath.im(v) != 0:
            raise ValueError("MarcumQ: only real arguments are supported")
    if m <= 0:
        raise ValueError("MarcumQ: the order m must be positive")
    x = a * a / 2
    y = b * b / 2
    with mpmath.workprec(mpmath.mp.prec + 20):
        total = mpmath.mpf(0)
        w = mpmath.exp(-x)
        eps = mpmath.mpf(2) ** (-(mpmath.mp.prec + 10))
        j = 0
        while True:
            term = w * mpmath.gammainc(m + j, y, regularized=True)
            total += term
            if j > x and w < eps:
                break
            j += 1
            w = w * x / j
            if j > 100000:
                raise ValueError("MarcumQ: series did not converge")
    return total


def _marcum_exact(m, a, b, b1=None):
    if b1 is not None:
        return None
    if b == 0:
        return sp.Integer(1)
    if a == 0:
        return Expr("GammaRegularized", R(m), R(b ** 2 / 2))
    return None


special("MarcumQ", (3, 4), _marcum, _marcum_exact,
        notes="MarcumQ[m, a, b] = Integrate[x (x/a)^(m-1) BesselI[m-1, a x] E^(-(a^2 + x^2)/2), {x, b, Infinity}] "
              "via the Poisson-weighted series Sum[E^(-a^2/2) (a^2/2)^j/j! GammaRegularized[m + j, b^2/2]] (real "
              "arguments, m > 0); MarcumQ[m, a, b0, b1] = MarcumQ[m, a, b0] - MarcumQ[m, a, b1]; exact "
              "MarcumQ[m, a, 0] = 1 and MarcumQ[m, 0, b] = GammaRegularized[m, b^2/2]"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _mittag(*v):
    if len(v) == 2:
        a, b, z = v[0], mpmath.mpf(1), v[1]
    else:
        a, b, z = v
    if mpmath.im(a) != 0 or a <= 0:
        raise ValueError("MittagLefflerE: only real alpha > 0 is supported")
    if z == 0:
        return mpmath.rgamma(b)
    size = abs(z) ** (1 / a)
    extra = int(size * 1.4427) + 30
    if extra > 40000:
        raise ValueError("MittagLefflerE: argument too large for the power series")
    with mpmath.workprec(mpmath.mp.prec + extra):
        total = mpmath.mpf(0)
        biggest = mpmath.mpf(0)
        k = 0
        zk = mpmath.mpf(1)
        eps = mpmath.mpf(2) ** (-(mpmath.mp.prec))
        while True:
            term = zk * mpmath.rgamma(a * k + b)
            total += term
            biggest = max(biggest, abs(term))
            if k > size + 5 and abs(term) < eps * biggest and abs(term) <= abs(total) * eps:
                break
            k += 1
            zk *= z
            if k > 200000:
                raise ValueError("MittagLefflerE: series did not converge")
    return total


def _mittag_sym(*v):
    if len(v) == 2:
        a, b, z = v[0], sp.Integer(1), v[1]
    else:
        a, b, z = v
    if z == 0 and b.is_number:
        return 1 / sp.gamma(b)
    if a == 1 and b == 1:
        return sp.exp(z)
    if a == 2 and b == 1:
        return sp.cosh(sp.sqrt(z))
    if a == 1 and b == 2:
        return (sp.exp(z) - 1) / z
    if a == sp.Rational(1, 2) and b == 1:
        return sp.exp(z ** 2) * sp.erfc(-z)
    return None


def _mittag_exact(*v):
    out = _mittag_sym(*v)
    return out


special("MittagLefflerE", (2, 3), _mittag, _mittag_exact, symbolic=_mittag_sym,
        notes="MittagLefflerE[a, b, z] = Sum[z^k/Gamma[a k + b], {k, 0, Infinity}] (power series with extra "
              "working precision against cancellation, real a > 0); MittagLefflerE[a, z] = MittagLefflerE[a, 1, z]; "
              "closed forms for (1, 1), (2, 1), (1, 2), (1/2, 1) and z = 0"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


# ───────────────────────────── inverse regularized gamma and beta ─────────────────────────────

def _bracket_root(f, lo, hi, increasing):
    """Bisection + secant on a monotone function on (lo, hi); f(lo) and f(hi) must differ in sign."""
    prec = mpmath.mp.prec
    with mpmath.workprec(prec + 30):
        flo, fhi = f(lo), f(hi)
        if flo == 0:
            return lo
        if fhi == 0:
            return hi
        if (flo > 0) == (fhi > 0):
            raise ValueError("no sign change")
        try:
            return mpmath.findroot(f, (lo, hi), solver="anderson", tol=mpmath.mpf(2) ** (-(prec + 10)))
        except (ValueError, ZeroDivisionError):
            for _ in range(prec + 40):
                mid = (lo + hi) / 2
                fm = f(mid)
                if (fm > 0) == (flo > 0):
                    lo, flo = mid, fm
                else:
                    hi = mid
            return (lo + hi) / 2


def _inv_gamma_reg(a, *rest):
    if len(rest) == 1:
        s = rest[0]
        target = s
    else:
        z0, s = rest
        target = mpmath.gammainc(a, z0, regularized=True) - s
    for v in (a, target):
        if mpmath.im(v) != 0:
            raise ValueError("InverseGammaRegularized: only real arguments are supported")
    if a <= 0:
        raise ValueError("InverseGammaRegularized: a must be positive")
    if target == 1:
        return mpmath.mpf(0)
    if target == 0:
        return mpmath.inf
    if not 0 < target < 1:
        raise ValueError("InverseGammaRegularized: the value must lie in (0, 1)")
    f = lambda z: mpmath.gammainc(a, z, regularized=True) - target
    hi = max(mpmath.mpf(1), a)
    while f(hi) > 0:
        hi *= 2
    lo = hi / 2
    while f(lo) < 0 and lo > mpmath.mpf(2) ** (-mpmath.mp.prec * 4):
        lo /= 2
    with mpmath.workprec(mpmath.mp.prec + 20):
        g = lambda t: mpmath.gammainc(a, mpmath.exp(t), regularized=True) - target
        t = _bracket_root(g, mpmath.log(lo), mpmath.log(hi), False)
    return mpmath.exp(t)


def _inv_gamma_exact(a, *rest):
    if len(rest) == 1:
        s = rest[0]
        if s == 1:
            return sp.Integer(0)
        if s == 0:
            return sp.oo
        if a == 1:
            return -sp.log(s)
    return None


special("InverseGammaRegularized", (2, 3), _inv_gamma_reg, _inv_gamma_exact,
        notes="InverseGammaRegularized[a, s] solves GammaRegularized[a, z] == s for z (upper regularized gamma; "
              "bracketing root finder at the input precision); InverseGammaRegularized[a, z0, s] inverts "
              "GammaRegularized[a, z0, z]; exact: s = 1 -> 0, s = 0 -> Infinity, a = 1 -> -Log[s]"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _inv_beta_reg(*v):
    if len(v) == 3:
        s, a, b = v
        target = s
    else:
        z0, s, a, b = v
        target = mpmath.betainc(a, b, 0, z0, regularized=True) + s
    for x in (a, b, target):
        if mpmath.im(x) != 0:
            raise ValueError("InverseBetaRegularized: only real arguments are supported")
    if a <= 0 or b <= 0:
        raise ValueError("InverseBetaRegularized: a and b must be positive")
    if target == 0:
        return mpmath.mpf(0)
    if target == 1:
        return mpmath.mpf(1)
    if not 0 < target < 1:
        raise ValueError("InverseBetaRegularized: the value must lie in (0, 1)")
    f = lambda z: mpmath.betainc(a, b, 0, z, regularized=True) - target
    return _bracket_root(f, mpmath.mpf(0), mpmath.mpf(1), True)


def _inv_beta_exact(*v):
    if len(v) == 3:
        s, a, b = v
        if s == 0:
            return sp.Integer(0)
        if s == 1:
            return sp.Integer(1)
        if b == 1:
            return s ** (1 / a)
        if a == 1:
            return 1 - (1 - s) ** (1 / b)
    return None


special("InverseBetaRegularized", (3, 4), _inv_beta_reg, _inv_beta_exact,
        notes="InverseBetaRegularized[s, a, b] solves BetaRegularized[z, a, b] == s for z in [0, 1] (bracketing "
              "root finder); InverseBetaRegularized[z0, s, a, b] inverts BetaRegularized[z0, z, a, b]; exact "
              "s = 0, 1 and b = 1 (s^(1/a)), a = 1 (1 - (1 - s)^(1/b))"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


# ───────────────────────────── Minkowski question mark and Cantor staircase ─────────────────────────────

def _qmark_cf(terms):
    """?(x) for a finite continued fraction [a0; a1, ..., an] (exact)."""
    a0, rest = terms[0], terms[1:]
    total = sp.Integer(a0)
    acc = 0
    for k, a in enumerate(rest, 1):
        acc += a
        total += 2 * sp.Integer(-1) ** (k + 1) / sp.Integer(2) ** acc
    return total


def _qmark_exact(x):
    if x.is_Rational:
        return _qmark_cf(list(sp.continued_fraction_iterator(x)))
    try:
        x = sp.expand_func(x).rewrite(sp.sqrt) if x.has(sp.GoldenRatio) else x
        t = sp.Symbol("t")
        if x.is_real and x.is_algebraic and sp.Poly(sp.minimal_polynomial(x, t), t).degree() == 2:
            cf = sp.continued_fraction(sp.radsimp(x))
        else:
            return None
    except Exception:
        return None
    if not cf or not isinstance(cf[-1], list):
        return None
    pre, period = list(cf[:-1]), list(cf[-1])
    if not pre:                             # purely periodic: a0 belongs to the period
        pre, period = [period[0]], period[1:] + period[:1]
    head = _qmark_cf(pre)
    k0 = len(pre) - 1                       # number of partial quotients after a0 in the prefix
    acc0 = sum(pre[1:])
    block = sp.Integer(0)
    acc = acc0
    for i, a in enumerate(period, 1):
        acc += a
        block += 2 * sp.Integer(-1) ** (k0 + i + 1) / sp.Integer(2) ** acc
    S, L = sum(period), len(period)
    ratio = sp.Integer(-1) ** L / sp.Integer(2) ** S
    return sp.nsimplify(head + block / (1 - ratio))


def _qmark_num(x):
    from fractions import Fraction
    if mpmath.im(x) != 0:
        raise ValueError("MinkowskiQuestionMark: the argument must be real")
    f = Fraction(*mpmath.mpf(x).man_exp[0:1] and (int(mpmath.mpf(x).man), 1)) if False else None
    m, e = mpmath.mpf(x).man_exp
    f = Fraction(int(m) * 2 ** e) if e >= 0 else Fraction(int(m), 2 ** (-e))
    a0 = f.numerator // f.denominator
    total = mpmath.mpf(a0)
    r = f - a0
    acc, k = 0, 0
    lim = mpmath.mp.prec + 10
    while r != 0:
        r = 1 / r
        a = r.numerator // r.denominator
        r -= a
        k += 1
        acc += a
        if acc > lim:
            break
        total += 2 * (-1) ** (k + 1) * mpmath.ldexp(1, -acc)
    return total


special("MinkowskiQuestionMark", 1, _qmark_num, _qmark_exact,
        notes="MinkowskiQuestionMark[x] = a0 + 2 Sum[(-1)^(k+1) 2^-(a1 + ... + ak)] for x = [a0; a1, a2, ...]: "
              "exact for rationals and quadratic irrationals (periodic continued fractions, e.g. ?(Sqrt[2]) = 7/5); "
              "approximate x uses its exact binary value"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _cantor_rational(p, q, maxdigits=None):
    """Cantor function of p/q in [0, 1]: exact via the ternary expansion (cycle detection)."""
    from fractions import Fraction
    r = Fraction(p, q)
    if r <= 0:
        return Fraction(0)
    if r >= 1:
        return Fraction(1)
    seen = {}
    bits = []
    while True:
        if maxdigits is None and r in seen:
            start = seen[r]
            pre, cyc = bits[:start], bits[start:]
            val = sum(Fraction(b, 2 ** (i + 1)) for i, b in enumerate(pre))
            cval = sum(Fraction(b, 2 ** (i + 1)) for i, b in enumerate(cyc))
            return val + cval / 2 ** len(pre) / (1 - Fraction(1, 2 ** len(cyc)))
        if maxdigits is not None and len(bits) >= maxdigits:
            return sum(Fraction(b, 2 ** (i + 1)) for i, b in enumerate(bits))
        seen[r] = len(bits)
        r *= 3
        d = r.numerator // r.denominator
        r -= d
        if d == 1:
            return sum(Fraction(b, 2 ** (i + 1)) for i, b in enumerate(bits)) + Fraction(1, 2 ** (len(bits) + 1))
        bits.append(d // 2)
        if r == 0:
            return sum(Fraction(b, 2 ** (i + 1)) for i, b in enumerate(bits))


def _cantor_exact(x):
    if x.is_Rational:
        if x.q > 10 ** 12:
            return None
        f = _cantor_rational(int(x.p), int(x.q))
        return sp.Rational(f.numerator, f.denominator)
    if x.is_real:
        if (x <= 0) is sp.true:
            return sp.Integer(0)
        if (x >= 1) is sp.true:
            return sp.Integer(1)
    return None


def _cantor_num(x):
    if mpmath.im(x) != 0:
        raise ValueError("CantorStaircase: the argument must be real")
    m, e = mpmath.mpf(x).man_exp
    from fractions import Fraction
    f = Fraction(int(m) * 2 ** e) if e >= 0 else Fraction(int(m), 2 ** (-e))
    v = _cantor_rational(f.numerator, f.denominator, maxdigits=mpmath.mp.prec + 10)
    return mpmath.mpf(v.numerator) / v.denominator


special("CantorStaircase", 1, _cantor_num, _cantor_exact,
        notes="CantorStaircase[x]: the Cantor function (0 for x <= 0, 1 for x >= 1); exact for rationals via the "
              "ternary expansion (periodic expansions summed exactly), approximate x via its exact binary value"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


# ───────────────────────────── q-analogues ─────────────────────────────

def _qpoch_finite_exact(a, q, n):
    if n >= 0:
        out = sp.Integer(1)
        for k in range(n):
            out *= 1 - a * q ** k
        return sp.nsimplify(sp.expand(out)) if out.is_number else out
    out = sp.Integer(1)
    for k in range(1, -n + 1):
        out *= 1 - a * q ** (-k)
    return 1 / out


def _qpochhammer(*args):
    name = "QPochhammer"
    vals = [sym(a, name) for a in args]
    if len(vals) == 1:
        return Expr("QPochhammer", args[0], args[0])
    a, q = vals[0], vals[1]
    n = vals[2] if len(vals) == 3 else None
    prec = input_precision(args)
    if n is not None and n == 0:
        return 1
    if n is not None and n.is_Integer and prec is None and a.is_number and q.is_number:
        v = _qpoch_finite_exact(a, q, int(n))
        return R(v)
    if n is None and a == 0:
        return 1
    if n is None and q == 0 and a.is_number:
        return R(1 - a)
    if not all(v.is_number for v in vals) or prec is None:
        raise Unevaluated()

    def f(a, q, n=None):
        if n is None:
            if abs(q) >= 1:
                raise ValueError("QPochhammer: the infinite product requires |q| < 1")
            return mpmath.qp(a, q)
        return mpmath.qp(a, q, n)
    return mp_eval(f, vals, prec)


_register("QPochhammer", (1, 3), _qpochhammer,
          "QPochhammer[a, q, n] = Prod[1 - a q^k, {k, 0, n-1}] (exact for integer n, negative n by the inverse "
          "product), QPochhammer[a, q] the infinite product (|q| < 1), QPochhammer[q] = QPochhammer[q, q]; "
          "mpmath qp for approximate arguments"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _qfactorial(n, q):
    name = "QFactorial"
    nv, qv = sym(n, name), sym(q, name)
    prec = input_precision([n, q])
    if nv.is_Integer and nv >= 0 and qv.is_number and prec is None:
        if qv == 1:
            return R(sp.factorial(nv))
        out = sp.Integer(1)
        for k in range(1, int(nv) + 1):
            out *= sp.cancel((1 - qv ** k) / (1 - qv))
        return R(sp.nsimplify(out))
    if not (nv.is_number and qv.is_number) or prec is None:
        raise Unevaluated()
    return mp_eval(lambda n, q: mpmath.qgamma(n + 1, q), [nv, qv], prec)


_register("QFactorial", 2, _qfactorial,
          "QFactorial[n, q] = QPochhammer[q, q, n]/(1 - q)^n = Prod[(1 - q^k)/(1 - q), {k, n}] (exact for "
          "non-negative integer n; general n via QGamma[n + 1, q])"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _qgamma(z, q):
    name = "QGamma"
    zv, qv = sym(z, name), sym(q, name)
    prec = input_precision([z, q])
    if zv.is_Integer and zv >= 1 and qv.is_number and prec is None:
        return _qfactorial(R(zv - 1), q)
    if not (zv.is_number and qv.is_number) or prec is None:
        raise Unevaluated()
    return mp_eval(mpmath.qgamma, [zv, qv], prec)


_register("QGamma", 2, _qgamma,
          "QGamma[z, q] = QPochhammer[q, q]/QPochhammer[q^z, q] (1 - q)^(1 - z) (mpmath qgamma); exact "
          "QFactorial[z - 1, q] at positive integers z"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _qbinomial(n, k, q):
    name = "QBinomial"
    nv, kv, qv = sym(n, name), sym(k, name), sym(q, name)
    prec = input_precision([n, k, q])
    if nv.is_Integer and kv.is_Integer and qv.is_number and prec is None:
        N, K = int(nv), int(kv)
        if N >= 0:
            if K < 0 or K > N:
                return 0
            out = sp.Integer(1)
            for i in range(1, K + 1):
                out *= (1 - qv ** (N - K + i)) / (1 - qv ** i)
            return R(sp.nsimplify(out))
        raise Unevaluated()
    if not all(v.is_number for v in (nv, kv, qv)) or prec is None:
        raise Unevaluated()

    def f(n, k, q):
        return mpmath.qgamma(n + 1, q) / (mpmath.qgamma(k + 1, q) * mpmath.qgamma(n - k + 1, q))
    return mp_eval(f, [nv, kv, qv], prec)


_register("QBinomial", 3, _qbinomial,
          "QBinomial[n, k, q] Gaussian binomial coefficient Prod[(1 - q^(n-k+i))/(1 - q^i), {i, k}] (exact for "
          "integers, 0 outside 0 <= k <= n); general arguments via QGamma (QBinomial[4, 2, 1/2] = 35/16)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _qpolygamma_core(n, z, q):
    if not 0 < abs(q) < 1:
        raise ValueError("QPolyGamma: only 0 < |q| < 1 is supported")
    lq = mpmath.log(q)
    shift_terms = []
    while abs(mpmath.power(q, z)) > 0.5:
        shift_terms.append(z)
        z = z + 1
    with mpmath.workprec(mpmath.mp.prec + 20):
        eps = mpmath.mpf(2) ** (-(mpmath.mp.prec + 10))
        total = mpmath.mpf(0)
        m = 1
        qz = mpmath.power(q, z)
        p = qz
        while True:
            term = mpmath.power(m, n) * p / (1 - mpmath.power(q, m))
            total += term
            if abs(term) < eps * max(abs(total), 1) and m > 3:
                break
            m += 1
            p *= qz
        val = mpmath.power(lq, n + 1) * total
        if n == 0:
            val += -mpmath.log(1 - q)
        for w in shift_terms:          # psi(w) = psi(w + 1) + d^n/dw^n [log q q^w/(1 - q^w)]
            g = lambda t: lq * mpmath.power(q, t) / (1 - mpmath.power(q, t))
            val += g(w) if n == 0 else mpmath.diff(g, w, n)
    return val


def _qpolygamma(*args):
    name = "QPolyGamma"
    vals = [sym(a, name) for a in args]
    if len(vals) == 2:
        n, z, q = sp.Integer(0), vals[0], vals[1]
    else:
        n, z, q = vals
    if not all(v.is_number for v in (n, z, q)):
        raise Unevaluated()
    if not (n.is_Integer and n >= 0):
        raise ValueError("QPolyGamma: the order n must be a non-negative integer")
    prec = input_precision(args)
    if prec is None:
        raise Unevaluated()
    N = int(n)
    return mp_eval(lambda z, q: _qpolygamma_core(N, z, q), [z, q], prec)


_register("QPolyGamma", (2, 3), _qpolygamma,
          "QPolyGamma[z, q] = D[Log[QGamma[z, q]], z] = -Log[1 - q] + Log[q] Sum[q^(m z)/(1 - q^m), {m, 1, "
          "Infinity}], QPolyGamma[n, z, q] its n-th derivative (Lambert-series form with the shift recurrence; "
          "0 < |q| < 1)"
          ". Check: reference (differential cases in tests/library).")


def _list_vals(x, name, what):
    if not is_list(x):
        raise Unevaluated() if isinstance(x, (Symbol, Expr)) else ValueError(f"{name}: {what} must be a list")
    return [sym(v, name) for v in x.args]


def _qhyper(a, b, q, z):
    name = "QHypergeometricPFQ"
    av, bv = _list_vals(a, name, "the first argument"), _list_vals(b, name, "the second argument")
    qv, zv = sym(q, name), sym(z, name)
    if zv == 0:
        return 1
    allv = av + bv + [qv, zv]
    if not all(v.is_number for v in allv):
        raise Unevaluated()
    prec = input_precision([a, b, q, z])
    if prec is None:
        raise Unevaluated()
    na = len(av)
    return mp_eval(lambda *v: mpmath.qhyper(list(v[:na]), list(v[na:-2]), v[-2], v[-1]), allv, prec)


_register("QHypergeometricPFQ", 4, _qhyper,
          "QHypergeometricPFQ[{a1, ..., ar}, {b1, ..., bs}, q, z] basic hypergeometric series Sum[(a1;q)_k ... "
          "/((b1;q)_k ... (q;q)_k) ((-1)^k q^(k(k-1)/2))^(1+s-r) z^k] (mpmath qhyper); value 1 at z = 0"
          ". Check: reference (differential cases in tests/library).",
          listable=False)


def _bihyper(a, b, z):
    name = "BilateralHypergeometricPFQ"
    av, bv = _list_vals(a, name, "the first argument"), _list_vals(b, name, "the second argument")
    zv = sym(z, name)
    allv = av + bv + [zv]
    if not all(v.is_number for v in allv):
        raise Unevaluated()
    prec = input_precision([a, b, z])
    if prec is None:
        raise Unevaluated()
    for x in av:
        if x.is_Integer and x > 0:
            raise ValueError(f"{name}: the a parameters cannot be positive integers")
    for x in bv:
        if x.is_Integer and x <= 0:
            raise ValueError(f"{name}: the b parameters cannot be zero or negative integers")
    na = len(av)
    if len(av) == len(bv):
        if zv == 0:
            raise ValueError(f"{name}: the series is singular at z = 0 when p = q")

        def f(*v):
            a_, b_, z_ = list(v[:na]), list(v[na:-1]), v[-1]
            # Wolfram's documented splitting: pFq(a, 1; b; z) + pFq(1, 1 - b; 1 - a; 1/z) - 1
            return (mpmath.hyper(a_ + [1], b_, z_) + mpmath.hyper([1] + [1 - x for x in b_], [1 - x for x in a_], 1 / z_)
                    - 1)
        return mp_eval(f, allv, prec)
    # p != q: one half diverges; mpmath's regularised (Borel-type) sum, as Wolfram documents
    return mp_eval(lambda *v: mpmath.bihyper(list(v[:na]), list(v[na:-1]), v[-1]), allv, prec)


_register("BilateralHypergeometricPFQ", 3, _bihyper,
          "BilateralHypergeometricPFQ[{a1, ...}, {b1, ...}, z] = Sum[Pochhammer[a1, n] ... /(Pochhammer[b1, n] "
          "...) z^n, {n, -Infinity, Infinity}] (mpmath bihyper)"
          ". Check: reference (differential cases in tests/library).", listable=False)


# ───────────────────────────── Appell functions ─────────────────────────────

def _appell(name, nparams, mpf, reduce_y0):
    def impl(*args):
        vals = [sym(a, name) for a in args]
        x, y = vals[-2], vals[-1]
        if x == 0 and y == 0:
            return 1
        if y == 0:
            return reduce_y0(args)
        if not all(v.is_number for v in vals):
            raise Unevaluated()
        prec = input_precision(args)
        if prec is None:
            raise Unevaluated()
        return mp_eval(mpf, vals, prec)
    _register(name, nparams + 2, impl, APPELL_NOTES[name])


APPELL_NOTES = {
    "AppellF1": "AppellF1[a, b1, b2, c, x, y] = Sum[(a)_(m+n) (b1)_m (b2)_n/((c)_(m+n) m! n!) x^m y^n] and its "
                "continuation (mpmath appellf1); y = 0 gives Hypergeometric2F1[a, b1, c, x]"
                ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).",
    "AppellF2": "AppellF2[a, b1, b2, c1, c2, x, y] = Sum[(a)_(m+n) (b1)_m (b2)_n/((c1)_m (c2)_n m! n!) x^m y^n] "
                "(mpmath appellf2); y = 0 gives Hypergeometric2F1[a, b1, c1, x]"
                ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).",
    "AppellF3": "AppellF3[a1, a2, b1, b2, c, x, y] = Sum[(a1)_m (a2)_n (b1)_m (b2)_n/((c)_(m+n) m! n!) x^m y^n] "
                "(mpmath appellf3); y = 0 gives Hypergeometric2F1[a1, b1, c, x]"
                ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).",
    "AppellF4": "AppellF4[a, b, c1, c2, x, y] = Sum[(a)_(m+n) (b)_(m+n)/((c1)_m (c2)_n m! n!) x^m y^n] (mpmath "
                "appellf4, |Sqrt[x]| + |Sqrt[y]| < 1); y = 0 gives Hypergeometric2F1[a, b, c1, x]"
                ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).",
}
_appell("AppellF1", 4, mpmath.appellf1, lambda a: Expr("Hypergeometric2F1", a[0], a[1], a[3], a[4]))
_appell("AppellF2", 5, mpmath.appellf2, lambda a: Expr("Hypergeometric2F1", a[0], a[1], a[3], a[5]))
_appell("AppellF3", 5, mpmath.appellf3, lambda a: Expr("Hypergeometric2F1", a[0], a[2], a[4], a[5]))
_appell("AppellF4", 4, mpmath.appellf4, lambda a: Expr("Hypergeometric2F1", a[0], a[1], a[2], a[4]))


# ───────────────────────────── angular momentum ─────────────────────────────

def _half_int(v, name, what):
    """Exact integer or half-integer (approximate numbers that are exact halves are accepted)."""
    if isinstance(v, sp.Float):
        r = sp.Rational(v)
        if (2 * r).is_Integer:
            return r
        raise ValueError(f"{name}: {what} {v} must be an integer or half-integer")
    if not v.is_number:
        raise Unevaluated()
    if v.is_Rational and (2 * v).is_Integer:
        return v
    raise ValueError(f"{name}: {what} {v} must be an integer or half-integer")


def _pairs(args, name, size):
    out = []
    for a in args:
        if not (is_list(a) and len(a.args) == size):
            raise Unevaluated() if isinstance(a, Symbol) else ValueError(f"{name}: malformed argument {a}")
        out.append([sym(x, name) for x in a.args])
    if not all(v.is_number for p in out for v in p):
        raise Unevaluated()
    return [[_half_int(v, name, "parameter") for v in p] for p in out]


def _jm_ok(j, m):
    return j >= 0 and (j - m).is_Integer and abs(m) <= j


def _approx_out(value, args):
    prec = input_precision(args)
    if prec is None:
        return R(value)
    return R(sp.N(value, 15 if prec == "m" else mpmath.libmp.prec_to_dps(prec)))


def _threej(a, b, c):
    from sympy.physics.wigner import wigner_3j
    (j1, m1), (j2, m2), (j3, m3) = _pairs((a, b, c), "ThreeJSymbol", 2)
    if not all(_jm_ok(j, m) for j, m in ((j1, m1), (j2, m2), (j3, m3))):
        if any(j < 0 or not (j - m).is_Integer for j, m in ((j1, m1), (j2, m2), (j3, m3))):
            raise ValueError("ThreeJSymbol: each j must be non-negative and j - m an integer")
        return 0
    return _approx_out(wigner_3j(j1, j2, j3, m1, m2, m3), (a, b, c))


def _clebsch(a, b, c):
    from sympy.physics.wigner import clebsch_gordan
    (j1, m1), (j2, m2), (j, m) = _pairs((a, b, c), "ClebschGordan", 2)
    if not all(_jm_ok(jj, mm) for jj, mm in ((j1, m1), (j2, m2), (j, m))):
        if any(jj < 0 or not (jj - mm).is_Integer for jj, mm in ((j1, m1), (j2, m2), (j, m))):
            raise ValueError("ClebschGordan: each j must be non-negative and j - m an integer")
        return 0
    return _approx_out(clebsch_gordan(j1, j2, j, m1, m2, m), (a, b, c))


def _sixj(a, b):
    from sympy.physics.wigner import wigner_6j
    (j1, j2, j3), (j4, j5, j6) = _pairs((a, b), "SixJSymbol", 3)
    if any(j < 0 for j in (j1, j2, j3, j4, j5, j6)):
        raise ValueError("SixJSymbol: the j must be non-negative")

    def tri(x, y, z):
        return (x + y + z).is_Integer and abs(x - y) <= z <= x + y
    if not (tri(j1, j2, j3) and tri(j1, j5, j6) and tri(j4, j2, j6) and tri(j4, j5, j3)):
        return 0
    return _approx_out(wigner_6j(j1, j2, j3, j4, j5, j6), (a, b))


_register("ThreeJSymbol", 3, _threej,
          "ThreeJSymbol[{j1, m1}, {j2, m2}, {j3, m3}] Wigner 3-j symbol (exact, sympy.physics.wigner); 0 unless "
          "m1 + m2 + m3 = 0, |mi| <= ji and the triangle condition holds; symbolic parameters stay unevaluated"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).",
          listable=False, numeric=False)
_register("ClebschGordan", 3, _clebsch,
          "ClebschGordan[{j1, m1}, {j2, m2}, {j, m}] (Condon-Shortley/Edmonds phase, exact; related to "
          "ThreeJSymbol by (-1)^(j1 - j2 + m) Sqrt[2 j + 1] ThreeJSymbol[{j1, m1}, {j2, m2}, {j, -m}])"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).",
          listable=False, numeric=False)
_register("SixJSymbol", 2, _sixj,
          "SixJSymbol[{j1, j2, j3}, {j4, j5, j6}] Wigner 6-j symbol (exact, Racah formula via "
          "sympy.physics.wigner); 0 unless the four triads satisfy the triangle conditions"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).",
          listable=False, numeric=False)


def _wigner_small_d(j, m1, m2, theta):
    """Wolfram's d^j_{m1 m2}(theta) = standard Wigner d^j_{m1 m2}(-theta) (WignerD[{1/2, 1/2, -1/2}, t] = Sin[t/2])."""
    c, s = sp.cos(theta / 2), sp.sin(theta / 2)
    out = sp.Integer(0)
    smin = max(0, int(m2 - m1))
    smax = min(int(j + m2), int(j - m1))
    pre = sp.sqrt(sp.factorial(j + m1) * sp.factorial(j - m1) * sp.factorial(j + m2) * sp.factorial(j - m2))
    for k in range(smin, smax + 1):
        out += (-1) ** k * pre / (sp.factorial(j + m2 - k) * sp.factorial(k) * sp.factorial(m1 - m2 + k)
                                  * sp.factorial(j - m1 - k)) * c ** (2 * j + m2 - m1 - 2 * k) * s ** (m1 - m2 + 2 * k)
    return out


def _wignerd(spec, *angles):
    name = "WignerD"
    if not (is_list(spec) and len(spec.args) == 3):
        raise Unevaluated() if isinstance(spec, Symbol) else ValueError(f"{name}: expected {{j, m1, m2}}")
    jv = [sym(x, name) for x in spec.args]
    if not all(v.is_number for v in jv):
        raise Unevaluated()
    j, m1, m2 = (_half_int(v, name, "parameter") for v in jv)
    if not (j >= 0 and (j - m1).is_Integer and (j - m2).is_Integer):
        raise Unevaluated()
    if abs(m1) > j or abs(m2) > j:
        return 0
    ang = [sym(a, name) for a in angles]
    if len(ang) == 1:
        psi, theta, phi = sp.Integer(0), ang[0], sp.Integer(0)
    elif len(ang) == 2:
        psi, theta, phi = sp.Integer(0), ang[0], ang[1]
    else:
        psi, theta, phi = ang
    val = sp.exp(sp.I * m1 * psi + sp.I * m2 * phi) * _wigner_small_d(j, m1, m2, theta)
    prec = input_precision(list(angles) + list(spec.args))
    if prec is not None and val.is_number:
        return R(sp.N(val, 15 if prec == "m" else mpmath.libmp.prec_to_dps(prec)))
    return R(val)


_register("WignerD", (2, 4), _wignerd,
          "WignerD[{j, m1, m2}, psi, theta, phi] = Exp[I m1 psi + I m2 phi] WignerD[{j, m1, m2}, 0, theta, 0] with "
          "Wolfram's phase convention (WignerD[{1/2, 1/2, -1/2}, t] = Sin[t/2]); WignerD[{j, m1, m2}, theta, phi] "
          "and WignerD[{j, m1, m2}, theta] set the leading angles to 0; physical (half-)integer parameters only"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).",
          listable=False, numeric=False)


# ───────────────────────────── multiple polylogarithms (Goncharov G) ─────────────────────────────

class _Divergent(Exception):
    """Upper-endpoint collision: the iterated integral diverges (Wolfram returns ComplexInfinity)."""


def _shuffle(u, v):
    if not u:
        return [tuple(v)]
    if not v:
        return [tuple(u)]
    return [(u[0],) + w for w in _shuffle(u[1:], v)] + [(v[0],) + w for w in _shuffle(u, v[1:])]


def _li_series(ms, xs, rate):
    """Li_{m1..mk}(x1..xk) = Sum_{n1 > ... > nk >= 1} prod x_i^n_i / n_i^m_i, truncated for geometric ``rate``."""
    k = len(ms)
    prec = mpmath.mp.prec
    N = int((prec * 0.6931 + 20 + 3 * k * mpmath.log(prec + 10)) / -mpmath.log(rate)) + 10 * k + 10
    if N > 400000:
        raise ValueError("series converges too slowly")
    # inner[i][n] = Sum over n > n_{i} > ... (levels i..k-1) ; iterate levels from the innermost outwards
    prev = None
    for i in range(k - 1, -1, -1):
        x, m = xs[i], ms[i]
        cur = [mpmath.mpf(0)] * (N + 1)        # cur[n] = Sum_{j < n} x^j/j^m * prev[j]
        acc = mpmath.mpf(0)
        xp = mpmath.mpf(1)
        for j in range(1, N + 1):
            xp = xp * x
            term = xp / mpmath.power(j, m) * (prev[j] if prev is not None else 1)
            acc += term
            if j < N:
                cur[j + 1] = acc
        cur_total = acc
        prev = cur
    return cur_total


def _g_series(word):
    """G(word; 1) for a word with non-zero last letter and every non-zero letter outside the unit disc."""
    ms, bs = [], []
    run = 0
    for a in word:
        if a == 0:
            run += 1
        else:
            ms.append(run + 1)
            bs.append(a)
            run = 0
    rate = max(1 / abs(b) for b in bs)
    if rate >= 1:
        raise ValueError("letter inside the unit disc")
    xs = [1 / bs[0]] + [bs[i - 1] / bs[i] for i in range(1, len(bs))]
    return (-1) ** len(bs) * _li_series(ms, xs, rate)


def _g_at(word, y, allow_holder):
    """G(word; y): regularised for trailing zeros (shuffle with G(0; y) = Log[y])."""
    word = tuple(word)
    if not word:
        return mpmath.mpf(1)
    if all(a == 0 for a in word):
        if y == 0:
            raise _Divergent()
        return mpmath.log(y) ** len(word) / mpmath.factorial(len(word))
    if y == 0:
        return mpmath.mpf(0)
    if word[0] == y:
        raise _Divergent()
    if word[-1] == 0:
        u = word[:-1]
        tz = 0
        for a in reversed(word):
            if a == 0:
                tz += 1
            else:
                break
        total = mpmath.log(y) * _g_at(u, y, allow_holder)
        for w in _shuffle((0,), u):
            if w != word:
                total -= _g_at(w, y, allow_holder)
        return total / tz
    scaled = tuple(a / y for a in word)
    return _g1(scaled, allow_holder)


def _g1(word, allow_holder=True):
    """G(word; 1), last letter non-zero, first letter != 1."""
    nz = [a for a in word if a != 0]
    direct = max(1 / abs(a) for a in nz)
    best, bestp = direct, None
    if allow_holder:
        for p in (2, 1.5, 3, 1.25, 4, 6, 10, 1.1):
            p = mpmath.mpf(p)
            r = max(max(1 / (p * abs(a)), (1 - 1 / p) / abs(1 - a) if a != 1 else 0) for a in nz)
            if r < best:
                best, bestp = r, p
    if best >= 0.92:
        raise ValueError("a letter lies on or too close to the integration path; analytic continuation around "
                         "letters on the path is not supported")
    if bestp is None:
        return _g_series(word)
    p = bestp
    w = len(word)
    total = mpmath.mpf(0)
    for j in range(w + 1):
        left = tuple(1 - a for a in reversed(word[:j]))
        right = word[j:]
        lv = _g_at(left, 1 - 1 / p, False) if left else mpmath.mpf(1)
        rv = _g_at(right, 1 / p, False) if right else mpmath.mpf(1)
        total += (-1) ** j * lv * rv
    return total


def goncharov(word, x):
    with mpmath.workprec(mpmath.mp.prec + 20 + 4 * len(word)):
        return _g_at(tuple(word), x, True)


def _complex_infinity():
    return Symbol("ComplexInfinity")


def _gpl_numeric(word_vals, x, prec):
    try:
        return mp_eval(lambda *v: goncharov(v[:-1], v[-1]), list(word_vals) + [x], prec)
    except _Divergent:
        return _complex_infinity()


def _gpl_impl(name, letters_of, sign_of):
    def impl(*args):
        if len(args) not in (2, 3):
            raise ValueError(f"{name}: expected a word, an endpoint and an optional base point")
        word = letters_of(args[0])
        x = sym(args[1], name)
        if len(args) == 3:
            if is_list(args[2]):
                ps = [sym(p, name) for p in args[2].args]
                if len(ps) != len(word):
                    raise ValueError(f"{name}: one base point per letter is expected")
                if any(sp.simplify(p - ps[0]) != 0 for p in ps):
                    raise ValueError(f"{name}: distinct base points per letter are not supported")
                base = ps[0]
            else:
                base = sym(args[2], name)
        else:
            base = sp.Integer(0)
        sgn = sign_of(word)
        if not word:
            return 1
        allv = word + [x, base]
        if not all(v.is_number for v in allv):
            raise Unevaluated()
        prec = input_precision(list(args))
        w2 = [a - base for a in word]
        x2 = x - base
        if prec is None:
            if all(a == 0 for a in w2) and x2 != 0:
                return R(sgn * sp.log(x2) ** len(w2) / sp.factorial(len(w2)))
            if len(w2) == 1 and x2 == 0:
                return 0
            raise Unevaluated()
        out = _gpl_numeric(w2, x2, prec)
        if isinstance(out, Symbol):
            return out
        return out if sgn == 1 else Expr("Times", -1, out)
    return impl


def _gpl_letters(w):
    if not is_list(w):
        raise Unevaluated() if isinstance(w, (Symbol, Expr)) else ValueError("GeneralizedPolyLog: the word must be a list")
    return [sym(a, "GeneralizedPolyLog") for a in w.args]


def _hpl_letters(w):
    if not is_list(w):
        raise Unevaluated() if isinstance(w, (Symbol, Expr)) else ValueError("HarmonicPolyLog: the word must be a list")
    out = []
    for a in w.args:
        v = sym(a, "HarmonicPolyLog")
        if not v.is_number:
            raise Unevaluated()
        if v.is_Float and float(v) == int(v):      # letters numericised by N[...]
            v = sp.Integer(int(v))
        if not v.is_Integer:
            raise ValueError("HarmonicPolyLog: letters must be 0, 1, -1 or abbreviated indices +-m")
        n = int(v)
        if abs(n) <= 1:
            out.append(sp.Integer(n))
        else:
            out.extend([sp.Integer(0)] * (abs(n) - 1) + [sp.Integer(1 if n > 0 else -1)])
    return out


_register("GeneralizedPolyLog", (2, 3), _gpl_impl("GeneralizedPolyLog", _gpl_letters, lambda w: 1),
          "GeneralizedPolyLog[{a1, ..., ak}, x] Goncharov polylogarithm Integrate[1/(t1 - a1) ... , {0 < tk < ... < "
          "t1 < x}]: trailing zeros regularised by shuffle (G(0; x) = Log[x]), upper-endpoint collision -> "
          "ComplexInfinity, numerics by the nested Li series with Hoelder convolution; base point p (a single p or a "
          "constant list) by translation; letters on the integration path raise (detour continuation not "
          "supported)"
          ". Check: exact (exact input gives exact results), reference (differential cases in "
          "tests/library).", listable=False)
_register("HarmonicPolyLog", (2, 3), _gpl_impl("HarmonicPolyLog", _hpl_letters,
                                              lambda w: (-1) ** sum(1 for a in w if a == 1)),
          "HarmonicPolyLog[{a1, ..., ak}, x] Remiddi-Vermaseren harmonic polylogarithm with kernels 1/t, 1/(1 - t), "
          "1/(1 + t) and the abbreviated +-m notation; computed as (-1)^(number of 1s) GeneralizedPolyLog of the "
          "same letters"
          ". Check: reference (differential cases in tests/library).", listable=False)


def _mzv_exact(s):
    k = len(s)
    if k == 1:
        return Expr("Zeta", R(s[0]))
    if all(v == 2 for v in s):
        return R(sp.pi ** (2 * k) / sp.factorial(2 * k + 1))
    if k == 2 and s[1] == 1 and s[0].is_Integer and s[0] >= 2:
        n = int(s[0])
        v = sp.Rational(n, 2) * sp.zeta(n + 1) - sp.Rational(1, 2) * sp.Add(
            *[sp.zeta(n - j) * sp.zeta(j + 1) for j in range(1, n - 1)])
        return R(sp.simplify(v))
    return None


def _mzv_numeric(svals):
    if not all(isinstance(v, (int,)) or (v == int(v)) for v in svals):
        raise ValueError("MultipleZeta: numerical evaluation needs positive integer indices")
    s = [int(v) for v in svals]
    if any(v < 1 for v in s):
        raise ValueError("MultipleZeta: numerical evaluation needs positive integer indices")
    if s[0] == 1:
        raise _Divergent()
    word = []
    for v in s:
        word += [0] * (v - 1) + [1]
    return (-1) ** len(s) * goncharov(word, mpmath.mpf(1))


def _multiple_zeta(*args):
    name = "MultipleZeta"
    if len(args) == 1 and is_list(args[0]):
        items = list(args[0].args)
    else:
        items = list(args)
    vals = [sym(a, name) for a in items]
    if not vals:
        return 1
    if not all(v.is_number for v in vals):
        if len(vals) == 1:
            return Expr("Zeta", items[0])
        raise Unevaluated()
    prec = input_precision(items)
    if prec is None:
        if vals[0] == 1:
            return _complex_infinity()
        if all(v.is_Integer and v >= 1 for v in vals):
            out = _mzv_exact(vals)
            if out is not None:
                return out
        if len(vals) == 1:
            return Expr("Zeta", items[0])
        raise Unevaluated()
    try:
        return mp_eval(lambda *v: _mzv_numeric(v), vals, prec)
    except _Divergent:
        return _complex_infinity()


_register("MultipleZeta", (1, 10), _multiple_zeta,
          "MultipleZeta[{s1, ..., sk}] = Sum over n1 > n2 > ... > nk >= 1 of 1/(n1^s1 ... nk^sk): MultipleZeta[{s}] "
          "= Zeta[s], exact ({2,...,2}) = Pi^(2k)/(2k+1)! and Euler's formula for {n, 1} (MultipleZeta[{2, 1}] = "
          "Zeta[3]); divergent s1 = 1 -> ComplexInfinity; numerics for positive integer indices via the iterated "
          "integral (Hoelder convolution); non-integer indices stay unevaluated / raise under N"
          ". Check: exact (exact input gives exact results), reference (differential cases in "
          "tests/library).", listable=False)


def _mpl(z, s):
    name = "MultiplePolyLog"
    if not (is_list(z) and is_list(s)):
        raise Unevaluated()
    zv = [sym(a, name) for a in z.args]
    sv = [sym(a, name) for a in s.args]
    if len(zv) != len(sv):
        raise ValueError(f"{name}: the index and argument lists must have the same length")
    if not zv:
        return 1
    if len(zv) == 1:
        return Expr("PolyLog", z.args[0], s.args[0])
    if all(v == 1 for v in sv) and input_precision(list(s.args)) is None:
        return Expr("MultipleZeta", z)
    if not all(v.is_number for v in zv + sv):
        raise Unevaluated()
    prec = input_precision([z, s])
    if prec is None:
        raise Unevaluated()
    k = len(zv)

    def f(*v):
        zs, ss = v[:k], v[k:]
        ps = []
        acc = mpmath.mpf(1)
        for x in ss:
            acc *= x
            ps.append(abs(acc))
        if max(ps) > 1:
            raise _Divergent()
        if max(ps) < 0.92:
            return _li_series(list(zs), list(ss), max(ps))
        if not all(mpmath.im(x) == 0 and x == int(x) and x >= 1 for x in zs):
            raise ValueError("MultiplePolyLog: on the boundary of convergence only positive integer indices are "
                             "supported")
        word = []
        acc = mpmath.mpf(1)
        for zi, si in zip(zs, ss):
            acc *= si
            word += [0] * (int(zi) - 1) + [1 / acc]
        return (-1) ** k * goncharov(word, mpmath.mpf(1))
    try:
        return mp_eval(f, zv + sv, prec)
    except _Divergent:
        return _complex_infinity()


_register("MultiplePolyLog", 2, _mpl,
          "MultiplePolyLog[{z1, ..., zk}, {s1, ..., sk}] = Sum over n1 > ... > nk >= 1 of s1^n1 ... sk^nk/(n1^z1 ... "
          "nk^zk): nested series inside the poly-disc |s1 ... sj| < 1, ComplexInfinity outside it, the boundary "
          "via GeneralizedPolyLog (positive integer indices); depth 1 = PolyLog, all s = 1 gives MultipleZeta"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).",
          listable=False)


# ───────────────────────────── Fox H, MeijerGReduce, FoxHReduce ─────────────────────────────

def _foxh_pairs(x, name, what):
    if not is_list(x):
        raise Unevaluated() if isinstance(x, Symbol) else ValueError(f"{name}: {what} must be a list of pairs")
    out = []
    for p in x.args:
        if not (is_list(p) and len(p.args) == 2):
            raise ValueError(f"{name}: {what} must contain pairs {{a, alpha}}")
        out.append((sym(p.args[0], name), sym(p.args[1], name)))
    return out


def _foxh_parse(A, B, name="FoxH"):
    if not (is_list(A) and len(A.args) == 2 and is_list(B) and len(B.args) == 2):
        raise ValueError(f"{name}: expected FoxH[{{{{{{a1, alpha1}}, ...}}, {{...}}}}, {{{{{{b1, beta1}}, ...}}, {{...}}}}, z]")
    a1 = _foxh_pairs(A.args[0], name, "the parameters")
    a2 = _foxh_pairs(A.args[1], name, "the parameters")
    b1 = _foxh_pairs(B.args[0], name, "the parameters")
    b2 = _foxh_pairs(B.args[1], name, "the parameters")
    return a1, a2, b1, b2


def _foxh_to_meijer(a1, a2, b1, b2):
    """Gauss multiplication: FoxH with rational alpha, beta = D C MeijerG[..., (z K^(-1/D))^D]; returns
    (D, C, K, A1, A2, B1, B2) with SymPy values."""
    scal = [al for _, al in a1 + a2 + b1 + b2]
    if any(not (v.is_Rational and v > 0) for v in scal):
        return None
    D = 1
    for v in scal:
        D = sp.ilcm(D, v.q)
    C = sp.Integer(1)
    K = sp.Integer(1)
    A1, A2, B1, B2 = [], [], [], []
    two_pi = 2 * sp.pi
    for a, al in a1:                              # numerator Gamma(1 - a - alpha s)
        k = int(al * D)
        A1 += [(a + j) / k for j in range(k)]
        C *= two_pi ** sp.Rational(1 - k, 2) * sp.Integer(k) ** (sp.Rational(1, 2) - a)
        K /= sp.Integer(k) ** k
    for a, al in a2:                              # denominator Gamma(a + alpha s)
        k = int(al * D)
        A2 += [(a + j) / k for j in range(k)]
        C /= two_pi ** sp.Rational(1 - k, 2) * sp.Integer(k) ** (a - sp.Rational(1, 2))
        K /= sp.Integer(k) ** k
    for b, be in b1:                              # numerator Gamma(b + beta s)
        k = int(be * D)
        B1 += [(b + j) / k for j in range(k)]
        C *= two_pi ** sp.Rational(1 - k, 2) * sp.Integer(k) ** (b - sp.Rational(1, 2))
        K *= sp.Integer(k) ** k
    for b, be in b2:                              # denominator Gamma(1 - b - beta s)
        k = int(be * D)
        B2 += [(b + j) / k for j in range(k)]
        C /= two_pi ** sp.Rational(1 - k, 2) * sp.Integer(k) ** (sp.Rational(1, 2) - b)
        K *= sp.Integer(k) ** k
    return D, C, K, A1, A2, B1, B2


def _lst(vals):
    return Expr("List", *(R(v) for v in vals))


def _foxh(A, B, z):
    name = "FoxH"
    a1, a2, b1, b2 = _foxh_parse(A, B)
    zv = sym(z, name)
    if all(al == 1 for _, al in a1 + a2 + b1 + b2):
        return Expr("MeijerG", Expr("List", _lst([a for a, _ in a1]), _lst([a for a, _ in a2])),
                    Expr("List", _lst([b for b, _ in b1]), _lst([b for b, _ in b2])), z)
    allv = [v for p in a1 + a2 + b1 + b2 for v in p] + [zv]
    if not all(v.is_number for v in allv):
        raise Unevaluated()
    prec = input_precision([A, B, z])
    if prec is None:
        raise Unevaluated()
    conv = _foxh_to_meijer(a1, a2, b1, b2)
    if conv is None:
        raise ValueError("FoxH: numerical evaluation is supported for positive rational alpha and beta")
    D, C, K, A1, A2, B1, B2 = conv
    if len(A1) + len(A2) + len(B1) + len(B2) > 400:
        raise ValueError("FoxH: the rational scale factors have too large a common denominator")
    if zv == 0:
        raise Unevaluated()

    def f(z):
        w = z * mpmath.power(to_mp(K), -mpmath.mpf(1) / D)
        g = mpmath.meijerg([[to_mp(x) for x in A1], [to_mp(x) for x in A2]],
                           [[to_mp(x) for x in B1], [to_mp(x) for x in B2]], w, r=mpmath.mpf(1) / D)
        return D * to_mp(C) * g
    return mp_eval(f, [zv], prec)


_register("FoxH", 3, _foxh,
          "FoxH[{{{a1, alpha1}, ...}, {...}}, {{{b1, beta1}, ...}, {...}}, z] Mellin-Barnes integral of "
          "Prod Gamma(b_j + beta_j s) Prod Gamma(1 - a_i - alpha_i s)/(...) z^-s (reference convention); all "
          "alpha = beta = 1 gives MeijerG; positive rational alpha, beta evaluated exactly as a MeijerG by the Gauss "
          "multiplication formula (mpmath meijerg with the z^(1/r) branch); other scale factors raise under N"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).",
          listable=False)


def _meijer_parts(f, x):
    """f = c x^p G(...) as (c, p, sympy meijerg) via SymPy's Meijer-G rewriting, or None."""
    from sympy.integrals.meijerint import _rewrite1
    try:
        r = _rewrite1(f, x)
    except Exception:
        return None
    if r is None:
        return None
    fac, power, gs, _cond = r            # f = fac * power * Sum[c x^s G]
    gs = [g for g in gs if g[0] != 0]
    if len(gs) != 1:
        return None
    c, s, g = gs[0]
    return sp.simplify(fac * power * c * x ** s), g


def _inactive(head, *args):
    return Expr(Expr("Inactive", Symbol(head)), *args)


def _reduce_common(name, expr, x, to_fox):
    if not isinstance(x, Symbol):
        raise ValueError(f"{name}: the second argument must be a variable")
    xs = sp.Symbol(x.name, positive=True)
    f = sym(expr, name).subs(sp.Symbol(x.name), xs)
    parts = _meijer_parts(f, xs)
    if parts is None:
        raise Unevaluated()
    pre, g = parts
    an, ap = [sp.sympify(v) for v in g.an], [sp.sympify(v) for v in g.aother]
    bm, bq = [sp.sympify(v) for v in g.bm], [sp.sympify(v) for v in g.bother]
    arg = g.argument.subs(xs, sp.Symbol(x.name))
    if to_fox:
        pair = lambda v: Expr("List", R(v), 1)
        core = _inactive("FoxH", Expr("List", Expr("List", *map(pair, an)), Expr("List", *map(pair, ap))),
                         Expr("List", Expr("List", *map(pair, bm)), Expr("List", *map(pair, bq))), R(arg))
    else:
        core = _inactive("MeijerG", Expr("List", _lst(an), _lst(ap)), Expr("List", _lst(bm), _lst(bq)), R(arg))
    pre = pre.subs(xs, sp.Symbol(x.name))
    if pre == 1:
        return core
    return Expr("Times", R(pre), core)


def _sympy_meijerg_back():
    """SymPy meijerg -> MeijerG when expressions holding MeijerG pass through SymPy (e.g. 2 MeijerG[...])."""
    from .. import math_ops
    math_ops._INV_FUNCS.setdefault(sp.meijerg, "MeijerG")


_sympy_meijerg_back()


def _meijerg_reduce(expr: Expr, x: Expr, *opts: Option) -> Expr:
    return _reduce_common("MeijerGReduce", expr, x, False)


def _foxh_reduce(expr: Expr, x: Expr, *opts: Option) -> Expr:
    return _reduce_common("FoxHReduce", expr, x, True)


fn("MeijerGReduce", area=AREA, status=STATUS,
   notes="MeijerGReduce[expr, x] -> c x^p Inactive[MeijerG][...] (SymPy's Meijer-G rewriting of elementary and "
         "special functions; Activate recovers expr, e.g. Sin[x] = Sqrt[Pi] MeijerG[{{}, {}}, {{1/2}, {0}}, x^2/4]); "
         "expressions with no single-G representation stay unevaluated; Assumptions are accepted and ignored")(
    _meijerg_reduce)
fn("FoxHReduce", area=AREA, status=STATUS,
   notes="FoxHReduce[expr, x] -> c x^p Inactive[FoxH][...] with unit scale factors (from the MeijerGReduce "
         "representation); Activate recovers expr; unsupported expressions stay unevaluated")(_foxh_reduce)
