"""Coverage batch 2: special functions evaluated with mpmath at the precision of their inputs.

Each function follows Wolfram's definition, argument order and branch cuts (checked against the reference pages
and mpmath).  Exact arguments give exact special values where Wolfram gives them and otherwise stay unevaluated;
approximate arguments are evaluated with mpmath at the lowest input precision (machine numbers give machine
numbers); N[f[exact args], p] evaluates to p digits.
"""

from __future__ import annotations

import mpmath
import sympy as sp

from ..expr import Expr, Symbol
from ..precision import input_precision
from .core import Unevaluated, fn, R, sym, is_list, list_arg

AREA = "special functions (batch 2)"

# name -> implementation, used by N[] to evaluate calls that stayed symbolic for exact arguments
MP_NUMERIC = {}


def to_mp(v):
    """SymPy number -> mpmath number at the current working precision."""
    v = sp.sympify(v)
    if v is sp.oo:
        return mpmath.inf
    if v is -sp.oo or v == -sp.oo:
        return -mpmath.inf
    if isinstance(v, sp.Float):
        return mpmath.mpf(v._mpf_)
    if v.is_Integer:
        return mpmath.mpf(int(v))
    if v.is_Rational:
        return mpmath.mpf(int(v.p)) / int(v.q)
    dps = mpmath.mp.dps + 5
    if v.is_real is False or v.has(sp.I):
        re, im = v.as_real_imag()
        re, im = sp.N(re, dps), sp.N(im, dps)
        if im == 0:
            return mpmath.mpf(sp.Float(re, dps)._mpf_)
        return mpmath.mpc(sp.Float(re, dps)._mpf_, sp.Float(im, dps)._mpf_)
    f = sp.N(v, dps)
    if not f.is_Float:
        re, im = f.as_real_imag()
        return mpmath.mpc(sp.Float(re, dps)._mpf_, sp.Float(im, dps)._mpf_)
    return mpmath.mpf(f._mpf_)


def from_mp(r, prec):
    """mpmath result -> MikoshiLang number at ``prec`` ("m" or bits)."""
    if isinstance(r, (list, tuple)):
        return Expr("List", *(from_mp(x, prec) for x in r))
    if isinstance(r, bool):
        return r
    if isinstance(r, mpmath.mpc):
        if r.imag == 0:
            r = r.real
        else:
            if prec == "m":
                return R(complex(r))
            re = sp.Float(r.real, precision=prec) if r.real != 0 else sp.Integer(0)
            return R(re + sp.I * sp.Float(r.imag, precision=prec))
    if isinstance(r, mpmath.mpf):
        if mpmath.isinf(r):
            return Symbol("Infinity") if r > 0 else Expr("Times", -1, Symbol("Infinity"))
        if mpmath.isnan(r):
            return Symbol("Indeterminate")
        if prec == "m":
            return float(r)
        return sp.Float(r, precision=prec)
    if isinstance(r, (int, float, complex)):
        return R(r)
    return R(r)


def mp_eval(f, vals, prec):
    bits = 53 if prec == "m" else prec
    with mpmath.workprec(bits + 24):
        out = f(*[to_mp(v) for v in vals])
    return from_mp(out, prec)


def special(name, nargs, mpf, exact=None, notes="", status="added", listable=True, symbolic=None, check=None,
            replace=False):
    """Register ``name``: ``mpf(*mp_args)`` for approximate input, ``exact(*sympy_args)`` (a SymPy expression,
    or None to stay unevaluated) for exact numeric input; ``symbolic`` for arguments with free symbols."""
    lo, hi = (nargs, nargs) if isinstance(nargs, int) else nargs

    def impl(*args):
        vals = [sym(a, name) for a in args]
        numeric = all(v.is_number for v in vals)
        if check is not None:
            check(*vals)
        if not numeric:
            if symbolic is not None:
                out = symbolic(*vals)
                if out is not None:
                    return R(out)
            raise Unevaluated()
        prec = input_precision(args)
        if prec is None:
            if exact is not None:
                out = exact(*vals)
                if out is not None:
                    return R(out) if not isinstance(out, (Expr, Symbol)) else out
            raise Unevaluated()
        return mp_eval(mpf, vals, prec)

    impl.__name__ = f"_{name}"
    if replace:
        from .core import REGISTRY
        e = REGISTRY.get(name)
        if e is not None:
            e.impls = [(f, a, b) for (f, a, b) in e.impls if not (a <= hi and (b is None or b >= lo))]
    # register one implementation per arity
    for n in range(lo, hi + 1):
        def make(n):
            def f(*args: Expr) -> Expr:
                return impl(*args)
            import inspect
            params = [inspect.Parameter(f"a{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation="Expr")
                      for i in range(n)]
            f.__signature__ = inspect.Signature(params, return_annotation="Expr")
            return f
        fn(name, area=AREA, status=status, notes=notes, listable=listable)(make(n))
    MP_NUMERIC[name] = _dispatcher(name)
    return impl


def _dispatcher(name):
    def call(*args):
        from .core import dispatch, UNHANDLED
        out = dispatch(name, list(args))
        if out is UNHANDLED:
            raise Unevaluated()
        return out
    return call


def numeric_call(name, args, prec):
    """Evaluate ``name[args]`` with approximate arguments at ``prec`` (used by N)."""
    return mp_eval_by_name(name, args, prec)


def mp_eval_by_name(name, args, prec):
    impl = MP_NUMERIC[name]
    return impl(*args)


def _ok(v):
    """Simplified SymPy value, or None if it still contains unevaluated special functions."""
    return v


def _int(v):
    return v.is_Integer


def _negint(v):
    return v.is_Integer and v <= 0


# ───────────────────────────── gamma, beta, error functions ─────────────────────────────

def _exact_via(fsym, cleanup=None):
    """Exact value from a SymPy function when it simplifies to something without that function."""
    def ex(*vals):
        try:
            v = fsym(*vals)
        except Exception:
            return None
        v = sp.sympify(v)
        if cleanup is not None:
            v = cleanup(v)
        if v.has(sp.Function) and any(isinstance(t, (sp.uppergamma, sp.lowergamma, sp.betainc, sp.hyper,
                                                       sp.polylog, sp.lerchphi, sp.erfinv, sp.erfcinv,
                                                       sp.Shi, sp.Chi, sp.fresnels, sp.fresnelc))
                                      for t in sp.preorder_traversal(v)):
            return None
        return v
    return ex


def _gamma_reg_exact(a, z):
    if z == 0:
        return sp.Integer(1)
    if z == sp.oo:
        return sp.Integer(0)
    if a.is_Integer and a > 0:
        return sp.simplify(sp.uppergamma(a, z) / sp.gamma(a))
    return None


special("GammaRegularized", 2, lambda a, z: mpmath.gammainc(a, z, regularized=True), _gamma_reg_exact,
        symbolic=lambda a, z: sp.Integer(1) if z == 0 else sp.Integer(0) if z == sp.oo else None,
        notes="GammaRegularized[a, z] = Gamma[a, z]/Gamma[a] (regularized upper incomplete gamma Q); exact for "
              "positive integer a; mpmath gammainc(regularized=True)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("GammaRegularized", 3, lambda a, z0, z1: mpmath.gammainc(a, z0, z1, regularized=True),
        lambda a, z0, z1: None if _gamma_reg_exact(a, z0) is None or _gamma_reg_exact(a, z1) is None
        else sp.simplify(_gamma_reg_exact(a, z0) - _gamma_reg_exact(a, z1)),
        notes="GammaRegularized[a, z0, z1] = GammaRegularized[a, z0] - GammaRegularized[a, z1]")


def _poly_beta(z0, z1, a, b):
    """Exact incomplete beta for positive integer a, b."""
    if a.is_Integer and b.is_Integer and a > 0 and b > 0:
        t = sp.Symbol("_t")
        return sp.expand(sp.integrate(t ** (a - 1) * (1 - t) ** (b - 1), (t, z0, z1)))
    return None


def _beta3_exact(z, a, b):
    return _poly_beta(sp.Integer(0), z, a, b)


special("Beta", (3, 3), lambda z, a, b: mpmath.betainc(a, b, 0, z), _beta3_exact,
        notes="Beta[z, a, b] is the incomplete beta function Integral[t^(a-1) (1-t)^(b-1), {t, 0, z}]; exact "
              "for positive integer a and b; mpmath betainc", status="fixed")
_beta4 = special("Beta", (4, 4), lambda z0, z1, a, b: mpmath.betainc(a, b, z0, z1),
                 lambda z0, z1, a, b: _poly_beta(z0, z1, a, b),
                 notes="Beta[z0, z1, a, b] generalized incomplete beta", status="fixed")


def _betareg_exact(z, a, b):
    v = _poly_beta(sp.Integer(0), z, a, b)
    return None if v is None else sp.expand(v / sp.beta(a, b).rewrite(sp.gamma))


special("BetaRegularized", 3, lambda z, a, b: mpmath.betainc(a, b, 0, z, regularized=True), _betareg_exact,
        notes="BetaRegularized[z, a, b] = Beta[z, a, b]/Beta[a, b]; exact for positive integer a, b"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("BetaRegularized", 4, lambda z0, z1, a, b: mpmath.betainc(a, b, z0, z1, regularized=True),
        lambda z0, z1, a, b: None if _poly_beta(z0, z1, a, b) is None
        else sp.expand(_poly_beta(z0, z1, a, b) / sp.beta(a, b).rewrite(sp.gamma)),
        notes="BetaRegularized[z0, z1, a, b]")


def _erf2_exact(z0, z1):
    v = sp.erf(z1) - sp.erf(z0)
    return v


special("Erf", (2, 2), lambda z0, z1: mpmath.erf(z1) - mpmath.erf(z0), _erf2_exact,
        symbolic=lambda z0, z1: sp.erf(z1) if z0.is_zero else (-sp.erf(z0) if z1.is_zero else None),
        notes="Erf[z0, z1] = Erf[z1] - Erf[z0] (generalized error function)", status="verified")


def _inverf_exact(x):
    if x == 0:
        return sp.Integer(0)
    if x == 1:
        return sp.oo
    if x == -1:
        return -sp.oo
    return None


def _inverf_mp(x):
    if isinstance(x, mpmath.mpc) or abs(x) > 1:
        raise ValueError("InverseErf is only supported for real arguments in [-1, 1]")
    return mpmath.erfinv(x)


def _inverfc_mp(x):
    if isinstance(x, mpmath.mpc) or not (0 <= x <= 2):
        raise ValueError("InverseErfc is only supported for real arguments in [0, 2]")
    if x == 0:
        return mpmath.inf
    if x == 2:
        return -mpmath.inf
    if x > 1:
        return -_inverfc_mp(2 - x)
    if x > mpmath.mpf("0.5"):
        return mpmath.erfinv(1 - x)
    # small x: solve erfc(y) = x by Newton iteration (erfinv(1 - x) would lose the digits of x)
    y = mpmath.erfinv(1 - x) if x > mpmath.mpf(10) ** -300 else mpmath.sqrt(-mpmath.log(x * mpmath.sqrt(mpmath.pi)))
    if not mpmath.isfinite(y):
        y = mpmath.sqrt(-mpmath.log(x))
    return mpmath.findroot(lambda t: mpmath.erfc(t) - x, y, tol=mpmath.mpf(2) ** (-mpmath.mp.prec + 8))


special("InverseErf", 1, _inverf_mp, _inverf_exact,
        notes="inverse of Erf on [-1, 1]: InverseErf[0] = 0, InverseErf[1] = Infinity; complex and |x| > 1 "
              "arguments raise (not supported)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("InverseErfc", 1, _inverfc_mp,
        lambda x: sp.Integer(0) if x == 1 else sp.oo if x == 0 else -sp.oo if x == 2 else None,
        notes="inverse of Erfc on [0, 2] (Newton refinement keeps full relative precision for tiny arguments); "
              "complex arguments raise (not supported)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


# ───────────────────────────── exponential and trigonometric integrals ─────────────────────────────

special("SinhIntegral", 1, mpmath.shi, _exact_via(sp.Shi),
        notes="Shi(z) = Integral[Sinh[t]/t, {t, 0, z}]; mpmath shi"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("CoshIntegral", 1, mpmath.chi,
        lambda z: -sp.oo if z == 0 else _exact_via(sp.Chi)(z),
        notes="Chi(z) = EulerGamma + Log[z] + Integral[(Cosh[t] - 1)/t, {t, 0, z}], branch cut along the "
              "negative real axis (CoshIntegral[-1.] = 0.837867 + 3.14159 I); CoshIntegral[0] = -Infinity"
              ". Check: reference (differential cases in tests/library).")
special("FresnelS", 1, mpmath.fresnels, _exact_via(sp.fresnels),
        notes="FresnelS[z] = Integral[Sin[Pi t^2/2], {t, 0, z}]; FresnelS[Infinity] = 1/2"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("FresnelC", 1, mpmath.fresnelc, _exact_via(sp.fresnelc),
        notes="FresnelC[z] = Integral[Cos[Pi t^2/2], {t, 0, z}]; FresnelC[Infinity] = 1/2"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


# ───────────────────────────── Bessel-type functions ─────────────────────────────

def _sph(kind):
    def f(n, z):
        if kind == "j":
            return mpmath.sqrt(mpmath.pi / (2 * z)) * mpmath.besselj(n + mpmath.mpf(1) / 2, z) if z != 0 else \
                (mpmath.mpf(1) if n == 0 else mpmath.mpf(0))
        if kind == "y":
            return mpmath.sqrt(mpmath.pi / (2 * z)) * mpmath.bessely(n + mpmath.mpf(1) / 2, z)
        j = mpmath.sqrt(mpmath.pi / (2 * z)) * mpmath.besselj(n + mpmath.mpf(1) / 2, z)
        y = mpmath.sqrt(mpmath.pi / (2 * z)) * mpmath.bessely(n + mpmath.mpf(1) / 2, z)
        return j + 1j * y if kind == "h1" else j - 1j * y
    return f


def _sph_exact(kind):
    def ex(n, z):
        if z == 0:
            return sp.zoo
        if n.is_Integer and n >= 0:
            j = sp.expand_func(sp.jn(n, z))
            y = sp.expand_func(sp.yn(n, z))
            if kind == "j":
                return sp.simplify(j)
            if kind == "y":
                return sp.simplify(y)
            h = j + sp.I * y if kind == "h1" else j - sp.I * y
            return sp.factor(sp.simplify(h.rewrite(sp.exp)))
        return None
    return ex


special("SphericalBesselJ", 2, _sph("j"), lambda n, z: (sp.Integer(1) if n == 0 else sp.Integer(0)) if z == 0 and n.is_Integer and n >= 0 else _sph_exact("j")(n, z),
        symbolic=_sph_exact("j"), status="fixed", replace=True,
        notes="j_n(z) = Sqrt[Pi/(2 z)] BesselJ[n + 1/2, z], elementary for integer n >= 0 (SphericalBesselJ[1, x] = "
              "Sin[x]/x^2 - Cos[x]/x; legacy left it unevaluated); any order and precision via mpmath"
              ". Check: reference (differential cases in tests/library).")
special("SphericalBesselY", 2, _sph("y"), _sph_exact("y"), symbolic=_sph_exact("y"),
        notes="y_n(z) = Sqrt[Pi/(2 z)] BesselY[n + 1/2, z]; SphericalBesselY[n, 0] = ComplexInfinity")
special("SphericalHankelH1", 2, _sph("h1"), _sph_exact("h1"), symbolic=_sph_exact("h1"),
        notes="h1_n(z) = SphericalBesselJ[n, z] + I SphericalBesselY[n, z]")
special("SphericalHankelH2", 2, _sph("h2"), _sph_exact("h2"), symbolic=_sph_exact("h2"),
        notes="h2_n(z) = SphericalBesselJ[n, z] - I SphericalBesselY[n, z]")
special("AiryAiPrime", 1, lambda z: mpmath.airyai(z, derivative=1),
        lambda z: -1 / (sp.root(3, 3) * sp.gamma(sp.Rational(1, 3))) if z == 0 else None,
        notes="derivative of AiryAi; AiryAiPrime[0] = -1/(3^(1/3) Gamma[1/3])")
special("AiryBiPrime", 1, lambda z: mpmath.airybi(z, derivative=1),
        lambda z: sp.root(3, 6) / sp.gamma(sp.Rational(1, 3)) if z == 0 else None,
        notes="derivative of AiryBi; AiryBiPrime[0] = 3^(1/6)/Gamma[1/3]")


def _gamma3_exact(a, z0, z1):
    v = sp.uppergamma(a, z0) - sp.uppergamma(a, z1)
    return None if v.has(sp.uppergamma) else v


special("Gamma", (3, 3), lambda a, z0, z1: mpmath.gammainc(a, z0, z1), _gamma3_exact,
        notes="Gamma[a, z0, z1] = Gamma[a, z0] - Gamma[a, z1] (generalized incomplete gamma); Gamma[a, 0, z] is "
              "the lower incomplete gamma", status="verified")


# ───────────────────────────── zeta-type functions ─────────────────────────────

def _polylog_exact(n, z):
    if z == 0:
        return sp.Integer(0)
    if z == 1:
        if n.is_number and (n.is_real and n > 1):
            return sp.zeta(n)
        if n == 1:
            return sp.oo
        return None
    if n == 1:
        return -sp.log(1 - z)
    l2 = sp.log(2)
    if z == sp.Rational(1, 2) and n == 2:
        return sp.pi ** 2 / 12 - l2 ** 2 / 2
    if z == sp.Rational(1, 2) and n == 3:
        return (-2 * sp.pi ** 2 * l2 + 4 * l2 ** 3 + 21 * sp.zeta(3)) / 24
    if z == -1 and n.is_real and n > 1:
        return -(1 - 2 ** (1 - n)) * sp.zeta(n)
    if n.is_Integer and n <= 0:
        x = sp.Symbol("_x")
        f = x / (1 - x)
        for _ in range(int(-n)):
            f = sp.together(x * sp.diff(f, x))
        return sp.factor(f.subs(x, z))
    v = sp.polylog(n, z)
    if isinstance(v, sp.polylog):
        e = sp.expand_func(v)
        return None if e.has(sp.polylog) else e
    return None if v.has(sp.polylog) else v


def _polylog_sym(n, z):
    if z == 0:
        return sp.Integer(0)
    if n.is_Integer and n <= 1:
        return _polylog_exact(n, z)
    return None


special("PolyLog", 2, mpmath.polylog, _polylog_exact, symbolic=_polylog_sym,
        notes="PolyLog[n, z] = Sum[z^k/k^n, {k, 1, Infinity}] continued analytically, branch cut [1, Infinity) "
              "(PolyLog[2, 2.] = 2.4674 - 2.17759 I); PolyLog[1, z] = -Log[1 - z], PolyLog[n, z] for integer n <= 0 "
              "is rational, PolyLog[s, 1] = Zeta[s], PolyLog[2, 1/2] = Pi^2/12 - Log[2]^2/2. The Nielsen form "
              "PolyLog[n, p, z] is not supported"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _wolfram_zeta_terms(s, a, f_tail, z=1):
    """Sum over k of z^k ((k + a)^2)^(-s/2) for Re(a) <= 0 terms (k + a = 0 excluded), then the tail."""
    if isinstance(a, mpmath.mpc):
        re = a.real
    else:
        re = a
    if re > 0:
        return f_tail(a, 1)
    n = int(mpmath.floor(-re)) + 1
    total = 0
    for k in range(n):
        t = a + k
        if t == 0:
            continue
        total += z ** k * ((t ** 2) ** (-s / 2))
    return total + z ** n * f_tail(a + n, 1)


special("Zeta", (2, 2), lambda s, a: _wolfram_zeta_terms(s, a, lambda b, _z: mpmath.zeta(s, b)),
        lambda s, a: _zeta2_exact(s, a), status="fixed", replace=True,
        notes="Zeta[s, a] = Sum[((k + a)^2)^(-s/2), {k, 0, Infinity}] (terms with k + a = 0 excluded); this "
              "differs from HurwitzZeta[s, a] = Sum[(k + a)^-s] only for Re(a) < 0 (legacy returned the Hurwitz value "
              "there: Zeta[3, -1.5] = 0.118 instead of 16.71); exact values from SymPy (Zeta[2, 1/2] = Pi^2/2)")


def _zeta2_exact(s, a):
    if a.is_number and a.is_real and a <= 0:
        return None
    v = sp.zeta(s, a)
    if isinstance(v, sp.zeta):
        e = sp.expand_func(v)
        if not e.has(sp.zeta) or e != v:
            return e if not e.has(sp.zeta) or len(e.atoms(sp.zeta)) == 0 else _zeta_shift(s, a)
        return _zeta_shift(s, a)
    return v


def _zeta_shift(s, a):
    """Zeta[s, p/q] for an integer s >= 2 and a rational a: shift a into (0, 1] and use Zeta[s, 1/2] etc."""
    if s.is_Integer and s >= 2 and a.is_Rational and a > 0:
        frac = a - sp.floor(a) if a != sp.floor(a) else sp.Integer(1)
        if frac in (sp.Integer(1), sp.Rational(1, 2)):
            base = sp.zeta(s) if frac == 1 else (2 ** s - 1) * sp.zeta(s)
            shift = sum((frac + k) ** (-s) for k in range(int(a - frac)))
            return sp.nsimplify(base - shift) if not base.has(sp.zeta) else base - shift
    return None


special("HurwitzZeta", 2, mpmath.zeta, lambda s, a: _zeta2_exact(s, a) if not (a.is_real and a <= 0) else None,
        status="fixed", replace=True,
        notes="HurwitzZeta[s, a] = Sum[(k + a)^-s, {k, 0, Infinity}] (principal powers, also for Re(a) < 0)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _lerch_tail(z, s):
    return lambda b, _z: mpmath.lerchphi(z, s, b)


special("LerchPhi", 3, lambda z, s, a: _wolfram_zeta_terms(s, a, _lerch_tail(z, s), z),
        lambda z, s, a: sp.zeta(s, a) if z == 1 and a.is_positive and not isinstance(sp.zeta(s, a), sp.zeta)
        else (sp.Integer(1) / a ** s if z == 0 and a.is_positive else None),
        notes="LerchPhi[z, s, a] = Sum[z^k/((k + a)^2)^(s/2), {k, 0, Infinity}] (Wolfram's definition; terms with "
              "k + a = 0 excluded); mpmath lerchphi for Re(a) > 0"
              ". Check: reference (differential cases in tests/library).")
special("HurwitzLerchPhi", 3, mpmath.lerchphi,
        lambda z, s, a: sp.Integer(1) / a ** s if z == 0 else None,
        notes="HurwitzLerchPhi[z, s, a] = Sum[z^k/(k + a)^s, {k, 0, Infinity}] (principal powers)")


special("StieltjesGamma", 1, mpmath.stieltjes, lambda n: sp.EulerGamma if n == 0 else None,
        notes="Stieltjes constants gamma_n (Laurent coefficients of Zeta at s = 1); StieltjesGamma[0] = "
              "EulerGamma"
              ". Check: exact (exact input gives exact results), reference (differential "
              "cases in tests/library).", check=lambda n: None)
special("StieltjesGamma", 2, mpmath.stieltjes, lambda n, a: -sp.polygamma(0, a) if n == 0 else None,
        notes="generalized Stieltjes constants gamma_n(a) of Zeta[s, a]; StieltjesGamma[0, a] = -PolyGamma[a]")


# ───────────────────────────── hypergeometric functions ─────────────────────────────

def _hyperu_exact(a, b, z):
    if a == 0:
        return sp.Integer(1)
    return None


special("HypergeometricU", 3, mpmath.hyperu, _hyperu_exact,
        symbolic=lambda a, b, z: sp.Integer(1) if a == 0 else None,
        notes="Tricomi confluent hypergeometric function U(a, b, z) (branch cut along the negative real axis); "
              "HypergeometricU[0, b, z] = 1"
              ". Check: reference (differential cases in tests/library).")


def _params(x, name):
    if not is_list(x):
        raise ValueError(f"{name}: parameters must be lists")
    return list(x.args)


def _pfq_mp(regularized):
    def f(a, b, z):
        if regularized:
            return _hyper_reg(a, b, z)
        return mpmath.hyper(a, b, z)
    return f


def _hyper_reg(a, b, z):
    """Regularized pFq: pFq/Prod[Gamma[b]], finite when some b is a non-positive integer."""
    a, b = list(a), list(b)
    bad = [i for i, x in enumerate(b) if x == mpmath.floor(x) and x <= 0 and not isinstance(x, mpmath.mpc)]
    if not bad:
        g = 1
        for x in b:
            g *= mpmath.gamma(x)
        return mpmath.hyper(a, b, z) / g
    if len(bad) > 1:
        raise ValueError("regularized hypergeometric function with several non-positive integer lower "
                         "parameters is not supported")
    j = bad[0]
    m = int(-b[j])
    # F~ = z^(m+1) Prod (a)_(m+1) / (m+1)! / Prod_{k != j} Gamma(b_k + m + 1) * pFq(a + m + 1; b_k + m + 1, m + 2; z)
    num = mpmath.mpf(1)
    for x in a:
        num *= mpmath.rf(x, m + 1)
        if num == 0:
            return mpmath.mpf(0)
    den = mpmath.factorial(m + 1)
    nb = []
    for k, x in enumerate(b):
        if k == j:
            continue
        den *= mpmath.gamma(x + m + 1)
        nb.append(x + m + 1)
    return z ** (m + 1) * num / den * mpmath.hyper([x + m + 1 for x in a], nb + [m + 2], z)


def _pfq_impl(name, regularized):
    @fn(name, area=AREA, status="added", listable=False,
        notes=("regularized " if regularized else "") + "generalized hypergeometric function "
              "pFq(a1, ..., ap; b1, ..., bq; z)" + ("/Prod[Gamma[bj]] (finite for non-positive integer bj)" if regularized else
              "; exact closed forms via SymPy hyperexpand; p = q + 1 has the branch cut [1, Infinity)"))
    def impl(a: Expr, b: Expr, z: Expr) -> Expr:
        av = [sym(x, name) for x in _params(a, name)]
        bv = [sym(x, name) for x in _params(b, name)]
        zv = sym(z, name)
        vals = av + bv + [zv]
        if not all(v.is_number for v in vals):
            if regularized:
                raise Unevaluated()
            if (len(av), len(bv)) in ((0, 1), (1, 1), (2, 1)):
                from ..evaluate import evaluate
                return evaluate(_named_hyper(av, bv, zv, a, b, z))
            if all(v.is_number for v in av + bv):
                v = sp.hyperexpand(sp.hyper(av, bv, zv))
                if not v.has(sp.hyper):
                    return R(v)
            raise Unevaluated()
        prec = input_precision([a, b, z])
        if prec is None:
            if zv == 0 and not regularized:
                return 1
            if regularized:
                if any(x.is_Integer and x <= 0 for x in bv):
                    raise Unevaluated()
                g = sp.Mul(*[sp.gamma(x) for x in bv])
                inner = impl_plain(a, b, z)
                return R(sp.sympify(sym(inner, name)) / g) if not isinstance(inner, Expr) or \
                    inner.head != "HypergeometricPFQ" else Expr(name, a, b, z)
            if any(x.is_Integer and x <= 0 for x in bv):
                raise ValueError(f"{name}: lower parameter is a non-positive integer (the series is undefined; use "
                                 "HypergeometricPFQRegularized)")
            v = sp.hyperexpand(sp.hyper(av, bv, zv))
            if v.has(sp.hyper):
                return _named_hyper(av, bv, zv, a, b, z)
            return R(v)
        if not regularized and any(x.is_Integer and x <= 0 for x in bv):
            raise ValueError(f"{name}: lower parameter is a non-positive integer")
        f = _pfq_mp(regularized)
        bits = 53 if prec == "m" else prec
        with mpmath.workprec(bits + 24):
            out = f([to_mp(x) for x in av], [to_mp(x) for x in bv], to_mp(zv))
        return from_mp(out, prec)
    return impl


def _named_hyper(av, bv, zv, a, b, z):
    key = (len(av), len(bv))
    names = {(0, 1): "Hypergeometric0F1", (1, 1): "Hypergeometric1F1", (2, 1): "Hypergeometric2F1"}
    if key in names:
        return Expr(names[key], *a.args, *b.args, z)
    return Expr("HypergeometricPFQ", a, b, z)


impl_plain = _pfq_impl("HypergeometricPFQ", False)
_pfq_impl("HypergeometricPFQRegularized", True)
MP_NUMERIC["HypergeometricPFQ"] = _dispatcher("HypergeometricPFQ")
MP_NUMERIC["HypergeometricPFQRegularized"] = _dispatcher("HypergeometricPFQRegularized")


def _reg_named(name, p, q):
    def mpf(*args):
        return _hyper_reg(list(args[:p]), list(args[p:p + q]), args[-1])

    def ex(*vals):
        bv = vals[p:p + q]
        if any(x.is_Integer and x <= 0 for x in bv):
            return None
        g = sp.Mul(*[sp.gamma(x) for x in bv])
        v = sp.hyperexpand(sp.hyper(list(vals[:p]), list(bv), vals[-1]))
        if v.has(sp.hyper):
            return None
        return sp.simplify(v / g)
    special(name, p + q + 1, mpf, ex,
            notes=f"regularized {p}F{q}: Hypergeometric{p}F{q}/Gamma[b] (finite for non-positive integer b, e.g. "
                  f"via the limit formula z^(m+1) (a)_(m+1)/(m+1)! {p}F{q}(a + m + 1; m + 2; z) for b = -m)")


_reg_named("Hypergeometric0F1Regularized", 0, 1)
_reg_named("Hypergeometric1F1Regularized", 1, 1)
_reg_named("Hypergeometric2F1Regularized", 2, 1)


@fn("MeijerG", area=AREA, status="added",
    notes="MeijerG[{{a1, ..., an}, {an+1, ..., ap}}, {{b1, ..., bm}, {bm+1, ..., bq}}, z] (and the form with a "
          "final argument r) evaluated with mpmath meijerg for approximate arguments; exact arguments stay "
          "unevaluated (FunctionExpand is not implemented)")
def _meijerg(a: Expr, b: Expr, z: Expr, r: Expr = None) -> Expr:
    def two(x):
        if not (is_list(x) and len(x.args) == 2 and all(is_list(y) for y in x.args)):
            raise ValueError("MeijerG: parameters must be {{...}, {...}}")
        return [[sym(v, "MeijerG") for v in y.args] for y in x.args]
    av, bv, zv = two(a), two(b), sym(z, "MeijerG")
    rv = sym(r, "MeijerG") if r is not None else sp.Integer(1)
    vals = [v for grp in av + bv for v in grp] + [zv, rv]
    if not all(v.is_number for v in vals):
        raise Unevaluated()
    prec = input_precision([a, b, z] + ([r] if r is not None else []))
    if prec is None:
        raise Unevaluated()
    bits = 53 if prec == "m" else prec
    with mpmath.workprec(bits + 24):
        out = mpmath.meijerg([[to_mp(v) for v in g] for g in av], [[to_mp(v) for v in g] for g in bv], to_mp(zv),
                             to_mp(rv))
    return from_mp(out, prec)


MP_NUMERIC["MeijerG"] = _dispatcher("MeijerG")


# ───────────────────────────── elliptic functions ─────────────────────────────

_JACOBI = ["SN", "CN", "DN", "SC", "SD", "CD", "CS", "DS", "DC", "NS", "NC", "ND"]
_TRIG = {"SN": sp.sin, "CN": sp.cos, "DN": lambda u: sp.Integer(1), "SC": sp.tan, "SD": sp.sin, "CD": sp.cos,
         "CS": sp.cot, "DS": sp.csc, "DC": sp.sec, "NS": sp.csc, "NC": sp.sec, "ND": lambda u: sp.Integer(1)}
_HYP = {"SN": sp.tanh, "CN": sp.sech, "DN": sp.sech, "SC": sp.sinh, "SD": sp.sinh, "CD": lambda u: sp.Integer(1),
        "CS": sp.csch, "DS": sp.csch, "DC": lambda u: sp.Integer(1), "NS": sp.coth, "NC": sp.cosh, "ND": sp.cosh}
_AT0 = {"SN": 0, "CN": 1, "DN": 1, "SC": 0, "SD": 0, "CD": 1, "CS": sp.zoo, "DS": sp.zoo, "DC": 1, "NS": sp.zoo,
        "NC": 1, "ND": 1}


def _jacobi(kind):
    pq = kind.lower()

    def mpf(u, m):
        r = mpmath.ellipfun(pq, u, m=m)
        if isinstance(r, mpmath.mpc) and mpmath.im(u) == 0 and mpmath.im(m) == 0:
            r = mpmath.re(r)        # real for real u and m (also m > 1); drop round-off imaginary parts
        return r

    def ex(u, m):
        if u == 0:
            return sp.sympify(_AT0[kind])
        if m == 0:
            return _TRIG[kind](u)
        if m == 1:
            return _HYP[kind](u)
        return None
    special("Jacobi" + kind, 2, mpf, ex, symbolic=ex,
            notes=f"Jacobi elliptic function {pq}(u | m) with the parameter m = k^2 (Wolfram convention); "
                  f"reduces to trigonometric functions at m = 0 and hyperbolic functions at m = 1; mpmath ellipfun")


for _k in _JACOBI:
    _jacobi(_k)


def _arc(kind):
    """Inverse Jacobi functions as EllipticF[phi, m] (DLMF 22.15) on their principal real ranges."""
    def mpf(v, m):
        if isinstance(v, mpmath.mpc) or isinstance(m, mpmath.mpc):
            raise ValueError(f"InverseJacobi{kind}: complex arguments are not supported")
        F = mpmath.ellipf
        one = mpmath.mpf(1)
        if kind == "SN":
            if abs(v) > 1:
                raise ValueError("InverseJacobiSN: argument must lie in [-1, 1]")
            return F(mpmath.asin(v), m)
        if kind == "CN":
            if abs(v) > 1:
                raise ValueError("InverseJacobiCN: argument must lie in [-1, 1]")
            return F(mpmath.acos(v), m)
        if kind == "DN":
            if not (0 < m <= 1) or not (mpmath.sqrt(1 - m) <= v <= 1):
                raise ValueError("InverseJacobiDN: needs 0 < m <= 1 and Sqrt[1 - m] <= v <= 1")
            return F(mpmath.asin(mpmath.sqrt((1 - v ** 2) / m)), m)
        if kind == "SC":
            return F(mpmath.atan(v), m)
        if kind == "CS":
            if v == 0:
                return F(mpmath.pi / 2, m)
            return F(mpmath.atan(one / v), m) if v > 0 else -F(mpmath.atan(-one / v), m)
        if kind == "NS":
            if abs(v) < 1:
                raise ValueError("InverseJacobiNS: Abs[argument] must be >= 1")
            return F(mpmath.asin(one / v), m)
        if kind == "NC":
            if v < 1:
                raise ValueError("InverseJacobiNC: argument must be >= 1")
            return F(mpmath.acos(one / v), m)
        if kind == "SD":
            if m < 0 and abs(v) * mpmath.sqrt(-m) >= 1:
                raise ValueError("InverseJacobiSD: argument out of range")
            return F(mpmath.asin(v / mpmath.sqrt(1 + m * v ** 2)), m)
        if kind == "CD":
            if abs(v) > 1 or not (0 <= m < 1):
                raise ValueError("InverseJacobiCD: needs Abs[v] <= 1 and 0 <= m < 1")
            return F(mpmath.acos(v * mpmath.sqrt((1 - m) / (1 - m * v ** 2))), m) if False else \
                F(mpmath.asin(mpmath.sqrt((1 - v ** 2) / (1 - m * v ** 2))), m)
        raise ValueError(f"InverseJacobi{kind} is not supported")

    def ex(v, m):
        if kind in ("SN", "SC", "SD") and v == 0:
            return sp.Integer(0)
        if kind in ("CN", "CD") and v == 1:
            return sp.Integer(0)
        if kind == "DN" and v == 1:
            return sp.Integer(0)
        if kind == "NC" and v == 1:
            return sp.Integer(0)
        if kind == "SN" and m == 0:
            return sp.asin(v)
        if kind == "CN" and m == 0:
            return sp.acos(v)
        if kind == "SN" and m == 1:
            return sp.atanh(v)
        return None
    special("InverseJacobi" + kind, 2, mpf, ex, symbolic=lambda v, m: ex(v, m),
            notes=f"inverse of Jacobi{kind} in its parameter-m form, via EllipticF (DLMF 22.15) on the principal "
                  "real range; complex arguments and arguments outside that range raise (not supported)")


for _k in ["SN", "CN", "DN", "SC", "CS", "NS", "NC", "SD", "CD"]:
    _arc(_k)


def _weierstrass_roots(g2, g3):
    roots = mpmath.polyroots([4, 0, -g2, -g3], maxsteps=200, extraprec=60)
    if min(abs(roots[i] - roots[j]) for i in range(3) for j in range(i + 1, 3)) < mpmath.mpf(2) ** (-mpmath.mp.prec // 2):
        raise ValueError("WeierstrassP: degenerate invariants (g2^3 = 27 g3^2) are not supported")
    return roots


def _wp(u, g2, g3, deriv=False):
    e1, e2, e3 = _weierstrass_roots(g2, g3)
    w = mpmath.sqrt(e1 - e3)
    m = (e2 - e3) / (e1 - e3)
    sn, cn, dn = (mpmath.ellipfun(k, w * u, m=m) for k in ("sn", "cn", "dn"))
    if deriv:
        return -2 * (e1 - e3) * w * cn * dn / sn ** 3
    return e3 + (e1 - e3) / sn ** 2


def _wp_args(name):
    def check(u, g):
        return None
    return check


@fn("WeierstrassP", area=AREA, status="added",
    notes="WeierstrassP[u, {g2, g3}] via the Jacobi form e3 + (e1 - e3)/sn^2(Sqrt[e1 - e3] u | m), "
          "m = (e2 - e3)/(e1 - e3), with e_i the roots of 4 t^3 - g2 t - g3 (valid for any ordering of the roots); "
          "approximate arguments only; degenerate invariants raise"
          ". Check: reference (differential cases in tests/library).")
def _weierstrassp(u: Expr, g: Expr) -> Expr:
    return _weier(u, g, False, "WeierstrassP")


@fn("WeierstrassPPrime", area=AREA, status="added",
    notes="derivative of WeierstrassP[u, {g2, g3}] with respect to u"
          ". Check: reference (differential cases in tests/library).")
def _weierstrassprime(u: Expr, g: Expr) -> Expr:
    return _weier(u, g, True, "WeierstrassPPrime")


def _weier(u, g, deriv, name):
    gs = list_arg(g, name, "invariants")
    if len(gs) != 2:
        raise ValueError(f"{name}: invariants must be {{g2, g3}}")
    vals = [sym(u, name)] + [sym(x, name) for x in gs]
    if not all(v.is_number for v in vals):
        raise Unevaluated()
    prec = input_precision([u, g])
    if prec is None:
        if vals[0] == 0:
            return Symbol("ComplexInfinity")
        raise Unevaluated()
    bits = 53 if prec == "m" else prec
    with mpmath.workprec(bits + 40):
        out = _wp(*(to_mp(v) for v in vals), deriv=deriv)
        if all(v.is_real for v in vals) and isinstance(out, mpmath.mpc):
            out = out.real                   # real u and invariants give a real value
    return from_mp(out, prec)


MP_NUMERIC["WeierstrassP"] = _dispatcher("WeierstrassP")
MP_NUMERIC["WeierstrassPPrime"] = _dispatcher("WeierstrassPPrime")


# ───────────────────────────── gamma-type combinatorial functions ─────────────────────────────

def _barnes_exact(z):
    if z.is_Integer:
        if z <= 0:
            return sp.Integer(0)
        out = sp.Integer(1)
        for k in range(1, int(z) - 1):
            out *= sp.factorial(k)
        return out
    return None


special("BarnesG", 1, mpmath.barnesg, _barnes_exact,
        notes="Barnes G-function, G(z + 1) = Gamma(z) G(z); BarnesG[n] = Prod[k!, {k, 0, n - 2}] for positive "
              "integers, 0 at non-positive integers; mpmath barnesg"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _poch_exact(a, n):
    if n == 0:
        return sp.Integer(1)
    if n.is_Integer:
        return sp.expand(sp.rf(a, n)) if a.is_number else None
    v = sp.gamma(a + n) / sp.gamma(a)
    v = sp.simplify(v)
    return v


def _poch_mp(a, n):
    return mpmath.rf(a, n)


special("Pochhammer", 2, _poch_mp, _poch_exact,
        symbolic=lambda a, n: sp.Integer(1) if n == 0 else (sp.expand(sp.rf(a, n)) if n.is_Integer and n < 0 and False
                                                          else None),
        notes="rising factorial (a)_n = Gamma[a + n]/Gamma[a]; exact for integer n (including negative n: "
              "Pochhammer[a, -n] = 1/((a - 1)...(a - n))); stays unevaluated for symbolic a (Wolfram)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _fpower_exact(x, n, h=sp.Integer(1)):
    if n == 0:
        return sp.Integer(1)
    if n.is_Integer and n > 0:
        out = sp.Integer(1)
        for k in range(int(n)):
            out *= x - k * h
        return sp.expand(out)
    if n.is_Integer and n < 0:
        out = sp.Integer(1)
        for k in range(1, int(-n) + 1):
            out *= x + k * h
        return 1 / out
    return sp.simplify(h ** n * sp.gamma(x / h + 1) / sp.gamma(x / h - n + 1))


special("FactorialPower", 2, lambda x, n: mpmath.gamma(x + 1) / mpmath.gamma(x - n + 1), _fpower_exact,
        notes="falling factorial x (x - 1) ... (x - n + 1) = Gamma[x + 1]/Gamma[x - n + 1]; FactorialPower[x, n, h] "
              "= x (x - h) ... (x - (n - 1) h); stays unevaluated for symbolic x"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("FactorialPower", 3, lambda x, n, h: h ** n * mpmath.gamma(x / h + 1) / mpmath.gamma(x / h - n + 1),
        _fpower_exact, notes="FactorialPower[x, n, h]")


# ───────────────────────────── corrections to existing special functions ─────────────────────────────

def _expint_exact(n, z):
    if z.is_zero:
        if n.is_real and n > 1:
            return 1 / (n - 1)
        if n.is_real:
            return sp.zoo
        return None
    return sp.expint(n, z)


special("ExpIntegralE", 2, mpmath.expint, _expint_exact, symbolic=lambda n, z: sp.expint(n, z),
        status="fixed", replace=True,
        notes="ExpIntegralE[n, 0] = 1/(n - 1) for n > 1 and ComplexInfinity for n <= 1 (legacy left the integer "
              "cases unevaluated); numeric values at any precision via mpmath"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _ci_exact(z):
    if z.is_zero:
        return -sp.oo
    return sp.Ci(z)


special("CosIntegral", 1, mpmath.ci, _ci_exact, symbolic=sp.Ci, status="fixed", replace=True,
        notes="CosIntegral[0] = -Infinity (legacy gave ComplexInfinity); branch cut along the negative real axis "
              "(CosIntegral[-1.] = Ci(1) + I Pi)"
              ". Check: reference (differential cases in tests/library).")


def _lag_exact(n, x, a=None):
    if not n.is_Integer:
        return None
    if a is None:
        return sp.expand(sp.laguerre(n, x))
    if n < 0:
        return None
    return sp.expand(sp.assoc_laguerre(n, a, x))


special("LaguerreL", 2, lambda n, x: mpmath.laguerre(n, 0, x), lambda n, x: _lag_exact(n, x),
        symbolic=lambda n, x: _lag_exact(n, x) if n.is_Integer else None, status="fixed", replace=True,
        notes="LaguerreL[n, x]: polynomial for integer n, the confluent form 1F1(-n; 1; x) for non-integer n "
              "(legacy raised 'n should be an integer')"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("LaguerreL", 3, lambda n, a, x: mpmath.laguerre(n, a, x), lambda n, a, x: _lag_exact(n, x, a),
        symbolic=lambda n, a, x: _lag_exact(n, x, a) if n.is_Integer and n >= 0 else None, status="fixed",
        replace=True,
        notes="generalised LaguerreL[n, a, x] = Binomial[n + a, n] 1F1(-n; a + 1; x); legacy returned 1.0 for "
              "non-integer n")


def _legtype(a):
    t = int(a) if getattr(a, "is_Integer", False) else None
    if t not in (1, 2, 3):
        raise ValueError("Legendre functions: the type must be 1, 2 or 3")
    return 3 if t == 3 else 2


def _lp_exact(n, m, x, kind=2):
    if n.is_Integer and m.is_Integer and n >= 0 and kind == 2:
        return sp.expand(sp.assoc_legendre(n, m, x)) if abs(m) <= n else sp.Integer(0)
    return None


special("LegendreP", 2, lambda n, x: mpmath.legenp(n, 0, x, type=2),
        lambda n, x: sp.expand(sp.legendre(n, x)) if n.is_Integer and n >= 0 else None,
        symbolic=lambda n, x: sp.expand(sp.legendre(n, x)) if n.is_Integer and n >= 0 else None,
        status="fixed", replace=True,
        notes="LegendreP[n, x]: polynomial for integer n >= 0, the Legendre function for other n (mpmath)"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("LegendreP", 3, lambda n, m, x: mpmath.legenp(n, m, x, type=2), _lp_exact,
        symbolic=_lp_exact, status="fixed", replace=True,
        notes="LegendreP[n, m, x] with the Condon-Shortley phase (Wolfram type 1/2); non-integer degree/order "
              "via mpmath")
special("LegendreP", 4, lambda n, m, a, x: mpmath.legenp(n, m, x, type=_legtype(sp.Integer(int(a)))),
        lambda n, m, a, x: _lp_exact(n, m, x, _legtype(a)),
        symbolic=lambda n, m, a, x: _lp_exact(n, m, x, _legtype(a)), status="added",
        check=lambda n, m, a, x: _legtype(a),
        notes="LegendreP[n, m, a, z] of type a: types 1 and 2 are ((1+z)/(1-z))^(m/2) 2F1 forms (Condon-Shortley), "
              "type 3 is ((z+1)/(z-1))^(m/2) with the cut (-Infinity, 1]")


def _lq_sym(n, x):
    if not (n.is_Integer and n >= 0):
        return None
    q0 = (sp.log(1 + x) - sp.log(1 - x)) / 2
    a, b = q0, x * q0 - 1
    if n == 0:
        return q0
    for j in range(1, int(n)):
        a, b = b, ((2 * j + 1) * x * b - j * a) / (j + 1)
    return sp.expand(b)


special("LegendreQ", 2, lambda n, x: mpmath.legenq(n, 0, x, type=2), _lq_sym, symbolic=_lq_sym,
        status="fixed", replace=True,
        notes="LegendreQ[n, x] (Wolfram type 1): Q0 = (Log[1 + x] - Log[1 - x])/2 and the three-term recurrence, so "
              "|x| > 1 now continues analytically (LegendreQ[0, 2.] = 0.549306 - 1.5708 I) instead of raising"
              ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
special("LegendreQ", 3, lambda n, m, x: mpmath.legenq(n, m, x, type=2),
        lambda n, m, x: _lq_sym(n, x) if m.is_zero else None,
        symbolic=lambda n, m, x: _lq_sym(n, x) if m.is_zero else None, status="added",
        notes="associated LegendreQ[n, m, x] (Wolfram type 1, Condon-Shortley; mpmath legenq type 2)")
special("LegendreQ", 4, lambda n, m, a, x: mpmath.legenq(n, m, x, type=_legtype(sp.Integer(int(a)))),
        lambda n, m, a, x: _lq_sym(n, x) if m.is_zero and _legtype(a) == 2 else None,
        check=lambda n, m, a, x: _legtype(a), status="added",
        notes="LegendreQ[n, m, a, z] of type a (1/2: inside the unit interval form, 3: cut plane outside [-1, 1])")
