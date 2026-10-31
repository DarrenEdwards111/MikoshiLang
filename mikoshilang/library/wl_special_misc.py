"""Wolfram batch W10: special functions II.

q-functions, multiple zeta values and polylogarithms, Appell and Fox H functions, angular momentum
coefficients, waveforms, generalized functions and mathematical constants.

Every function follows the Wolfram Language reference page (definition, argument order, branch cuts and
special values).  Approximate arguments are evaluated with mpmath at the lowest input precision (machine
numbers give machine numbers); exact arguments give exact values where Wolfram gives them and otherwise stay
unevaluated, and N[f[exact args], p] evaluates to p digits.  Where a case cannot be computed reliably the call
raises a clear error rather than returning a doubtful number.
"""
import inspect

import mpmath
import sympy as sp

from ..expr import Expr, Symbol
from ..precision import input_precision
from .core import Unevaluated, fn, R, sym, is_list
from .wl_special import MP_NUMERIC, mp_eval, to_mp, from_mp, _dispatcher

AREA = "special functions II (W10)"
STATUS = "added"


def _register(name, nargs, impl, notes, listable=True, numeric=True, hold=False):
    lo, hi = (nargs, nargs) if isinstance(nargs, int) else nargs
    for n in range(lo, hi + 1):
        def make(n):
            def f(*args):
                return impl(*args)
            f.__signature__ = inspect.Signature(
                [inspect.Parameter(f"a{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD) for i in range(n)])
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
              "I Pi/2, ArcCoth[Infinity] = 0, ArcCoth[1] = Infinity, ArcCoth[I] = -I Pi/4)")
special("ArcCsch", 1, mpmath.acsch, _arc_exact(sp.acsch),
        notes="ArcCsch[z] = ArcSinh[1/z]; mpmath acsch; exact special values (ArcCsch[I] = -I Pi/2, ArcCsch[0] = "
              "ComplexInfinity)")
special("ArcSech", 1, mpmath.asech, _arc_exact(sp.asech),
        notes="ArcSech[z] = ArcCosh[1/z]; mpmath asech; exact special values (ArcSech[1] = 0, ArcSech[0] = "
              "Infinity, ArcSech[2] = I Pi/3)")


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
              "I Pi (2n + 3/2)); Gudermannian[0] = 0, Gudermannian[+-Infinity] = +-Pi/2")


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
              "InverseGudermannian[0] = 0, InverseGudermannian[+-Pi/2] = +-Infinity")


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
              "LogisticSigmoid[-Infinity] = 0")


def _sinc_exact(z):
    if z == 0:
        return sp.Integer(1)
    if z in (sp.oo, -sp.oo):
        return sp.Integer(0)
    v = sp.sin(z) / z
    return _clean(v, sp.sin) if not v.has(sp.sin) else None


special("Sinc", 1, lambda z: mpmath.mpf(1) if z == 0 else mpmath.sin(z) / z, _sinc_exact,
        notes="Sinc[z] = Sin[z]/z, Sinc[0] = 1; exact where Sin has an exact value (Sinc[Pi] = 0, "
              "Sinc[Pi/2] = 2/Pi)")


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
          "unevaluated", numeric=False)


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
          "HeavisidePi[x1, x2, ...] is the product", numeric=False)
_register("HeavisideLambda", (1, 8), _product_fn("HeavisideLambda", _lambda_value),
          "HeavisideLambda[x] = 1 - |x| for |x| < 1 and 0 otherwise (the convolution of two HeavisidePi); "
          "multivariate form is the product", numeric=False)
_register("UnitTriangle", (1, 8), _product_fn("UnitTriangle", _lambda_value),
          "UnitTriangle[x] = Piecewise[{{x + 1, -1 <= x < 0}, {1 - x, 0 <= x <= 1}}] (0 otherwise); "
          "UnitTriangle[x1, ...] is the product", numeric=False)
_register("DiracComb", (1, 8), _product_fn("DiracComb", _comb_value),
          "DiracComb[x] = Sum[DiracDelta[x - n], {n, -Infinity, Infinity}]: 0 for real non-integer x, "
          "unevaluated at integers and for symbolic x; multivariate form is the product", numeric=False)


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
          "not one of the ui (reference)", listable=False, numeric=False)


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
                  "SquareWave[1/2] = -1, reference PiecewiseExpand); SquareWave[{y1, y2}, x] gives y2 then y1",
    "TriangleWave": "TriangleWave[x]: period 1, 0 at integers, 1 at 1/4, -1 at 3/4 (reference); "
                    "TriangleWave[{min, max}, x] rescales",
    "SawtoothWave": "SawtoothWave[x] = x - Floor[x] (SawtoothWave[0] = 0); SawtoothWave[{min, max}, x] rescales",
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
            v = type(self)._mp()
        return mpmath.mpf(v)._mpf_

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
          "analytic continuations (LerchPhi / Hurwitz Zeta / PolyLog)")


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
          "p, n via Gamma[n + p]/(n! Gamma[p]) (PolyGamma[n + p] - PolyGamma[p])")


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
          "HarmonicNumber[n]; n = Infinity gives MultipleZeta / MultiplePolyLog", listable=False, numeric=False)


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
    if k <= n and len(vals) < need and not (n == 0 and k == 0):
        raise ValueError(f"{name}: BellY[{n}, {k}, list] needs at least {need} variables")
    return R(_bell_partial(n, k, vals + [sp.Integer(0)] * max(0, need - len(vals))))


_register("BellY", (1, 3), _belly,
          "BellY[n, k, {x1, ..., x(n-k+1)}] partial Bell polynomial (standard recurrence); BellY[m] generalized "
          "Bell polynomial of a matrix (n-th derivative of a composition, rows = derivatives of each function); "
          "BellY[n, k, matrix] raises (not supported)", listable=False, numeric=False)


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
          "non-negative integer n; NorlundB[n, 1] = BernoulliB[n]", numeric=False)


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
          "integer m, ComplexInfinity for non-integer m (reference definition)", numeric=False)


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
          "tau(p^(e+1)) = tau(p) tau(p^e) - p^11 tau(p^(e-1)); 0 for n <= 0", numeric=False)


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
              "branch; RamanujanTauTheta[9.22] = 1.40372 as in the reference); RamanujanTauTheta[0] = 0")


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
          "unevaluated")

_PAULI = {
    0: [[1, 0], [0, 1]],
    1: [[0, 1], [1, 0]],
    2: [[0, -sp.I], [sp.I, 0]],
    3: [[1, 0], [0, -1]],
}


def _pauli(k, *opts):
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
         "lists; WorkingPrecision; TargetStructure -> \"Dense\" (other structures raise)")(_pauli)


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
