"""Wolfram batch W07: special functions I -- elliptic, theta, Weierstrass, Carlson and ODE-defined functions
(Heun, Mathieu, Lame, spheroidal, Coulomb, Kelvin, Lommel, Scorer, ...).

Conventions follow the Wolfram Language reference pages (argument order, parameter m = k^2 for elliptic
functions, nome q, half-periods {w1, w3} with w2 = -w1 - w3, branch cuts).  Numerical values come from mpmath
at the precision of the inputs (machine numbers give machine numbers, N[f[exact], p] gives p digits); functions
without an mpmath counterpart are computed here from their defining series / differential equations with
working-precision checks.  Exact arguments give exact special values where Wolfram does and otherwise stay
unevaluated.  Arguments outside the implemented domain raise a clear error instead of returning a guess.
"""

from __future__ import annotations
import mpmath
import sympy as sp
from sympy.core.function import AppliedUndef

from ..expr import Expr, Symbol
from ..precision import input_precision
from .core import Unevaluated, fn, R, sym, is_list, L, rule
from .wl_special import MP_NUMERIC, _dispatcher, from_mp, special, to_mp

AREA = "special functions: elliptic, theta & ODE-defined (W07)"


# ───────────────────────────── infrastructure ─────────────────────────────

def _flat(x, name):
    """MikoshiLang argument -> SymPy value, or nested Python list of SymPy values for List arguments."""
    if is_list(x):
        return [_flat(a, name) for a in x.args]
    try:
        return sym(x, name)
    except (AttributeError, TypeError):
        # a symbolic call with List arguments (e.g. WeierstrassHalfPeriodW1[{g2, g3}]) has no SymPy form: the
        # result stays unevaluated
        raise Unevaluated()


def _all_numbers(v):
    if isinstance(v, list):
        return all(_all_numbers(a) for a in v)
    return bool(v.is_number)


def _to_mp(v):
    if isinstance(v, list):
        return [_to_mp(a) for a in v]
    return to_mp(v)


def _realify(out, vals):
    """Drop round-off imaginary parts when every input is real (only for functions real on the real axis)."""
    return out


class _Pole(Exception):
    """The function has a pole at these arguments (Wolfram returns ComplexInfinity)."""


def mpfun(name, nargs, mpf, exact=None, notes="", listable=True, symbolic=None, extra=24, real=None,
          status="added"):
    """Register ``name`` with ``nargs`` (int or (lo, hi)) arguments, which may be numbers or Lists of numbers.

    ``mpf(*mp_args)`` evaluates approximate input (lists arrive as Python lists of mpmath numbers);
    ``exact(*sympy_args)`` returns an exact value (SymPy or MikoshiLang expression) or None to stay
    unevaluated; ``symbolic`` is tried when some argument is symbolic.  ``real(*sympy_args)`` -> True when the
    result is known to be real, so round-off imaginary parts are dropped.  Errors raised by mpf (ValueError,
    NotImplementedError, ZeroDivisionError) surface as clear errors."""
    lo, hi = (nargs, nargs) if isinstance(nargs, int) else nargs

    def impl(*args):
        vals = [_flat(a, name) for a in args]
        if not all(_all_numbers(v) for v in vals):
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
        bits = 53 if prec == "m" else prec
        with mpmath.workprec(bits + extra):
            try:
                out = mpf(*[_to_mp(v) for v in vals])
            except ZeroDivisionError:
                raise ValueError(f"{name}: the arguments are at a singularity")
            except _Pole:
                return Symbol("ComplexInfinity")
            if real is not None and real(*vals):
                out = _re(out)
            elif _allreal(*vals):
                out = _chop(out)
        return from_mp(out, prec)

    import inspect
    for n in range(lo, hi + 1):
        def make(n):
            def f(*args: Expr) -> Expr:
                return impl(*args)
            f.__signature__ = inspect.Signature(
                [inspect.Parameter(f"a{i}", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation="Expr")
                 for i in range(n)], return_annotation="Expr")
            f.__name__ = f"_{name}_{n}"
            return f
        fn(name, area=AREA, status=status, notes=notes, listable=listable)(make(n))
    MP_NUMERIC[name] = _dispatcher(name)
    return impl


def _chop(x):
    """Drop an imaginary part that is round-off (relative size below 2^(-3/4 prec)); used for real input."""
    if isinstance(x, (list, tuple)):
        return type(x)(_chop(a) for a in x)
    if isinstance(x, mpmath.mpc) and abs(x.imag) <= abs(x) * mpmath.mpf(2) ** (-(3 * mpmath.mp.prec) // 4):
        return x.real
    if isinstance(x, mpmath.mpc) and x.real != 0 and abs(x.real) <= abs(x) * mpmath.mpf(2) ** (-(3 * mpmath.mp.prec) // 4):
        return mpmath.mpc(0, x.imag)
    return x


def _re(x):
    if isinstance(x, (list, tuple)):
        return type(x)(_re(a) for a in x)
    if isinstance(x, mpmath.mpc):
        return x.real
    return x


def _allreal(*vals):
    def ok(v):
        if isinstance(v, list):
            return all(ok(a) for a in v)
        return v.is_real is True
    return all(ok(v) for v in vals)


def _isint(x):
    """mpmath number is an integer (exactly)."""
    if isinstance(x, mpmath.mpc):
        if x.imag != 0:
            return False
        x = x.real
    return mpmath.isint(x)


def _realval(x, name, what):
    if isinstance(x, mpmath.mpc):
        if x.imag != 0:
            raise ValueError(f"{name}: {what} must be real")
        return x.real
    return x


def _posint(x, name, what="k"):
    x = _realval(x, name, what)
    if not mpmath.isint(x) or x < 1:
        raise ValueError(f"{name}: {what} must be a positive integer")
    return int(x)


def _eps():
    return mpmath.mpf(2) ** (-mpmath.mp.prec)


# ───────────────────────────── zeros of Airy and Bessel functions ─────────────────────────────

def _airy_zero(which):
    zero = mpmath.airyaizero if which == "Ai" else mpmath.airybizero
    name = f"Airy{which}Zero"

    def one(k):
        return zero(_posint(k, name))

    def two(k, x0):
        k = _posint(k, name)
        x0 = _realval(x0, name, "x0")
        # number of zeros >= x0 (all zeros are negative); asymptotic estimate then correction
        if x0 >= 0:
            j0 = 0
        else:
            t = (-x0) ** mpmath.mpf(1.5)
            off = 1 if which == "Ai" else 3
            j0 = max(0, int(mpmath.floor((8 * t / (3 * mpmath.pi) + off) / 4)) - 2)
            while j0 > 0 and zero(j0) < x0:
                j0 -= 1
            while zero(j0 + 1) >= x0:
                j0 += 1
        return zero(j0 + k)

    mpfun(name, 1, one, notes=f"AiryAiZero-style: k-th (negative) zero of {which}; numerical (N) via mpmath "
                              f"airy{which.lower()}zero; k must be a positive integer", listable=True)
    mpfun(name, 2, two, notes=f"{name}[k, x0]: k-th zero of {which} less than x0", listable=True)


_airy_zero("Ai")
_airy_zero("Bi")


def _bessel_zero(which):
    zero = mpmath.besseljzero if which == "J" else mpmath.besselyzero
    name = f"Bessel{which}Zero"

    def order(n):
        n = _realval(n, name, "the order n")
        if n < 0:
            raise NotImplementedError(f"{name}: negative orders are not supported")
        return n

    def two(n, k):
        return zero(order(n), _posint(k, name))

    def three(n, k, x0):
        n = order(n)
        k = _posint(k, name)
        x0 = _realval(x0, name, "x0")
        if x0 < 0:
            x0 = mpmath.mpf(0)
        # McMahon: j_{n,s} ~ (s + n/2 - 1/4) pi (J), (s + n/2 - 3/4) pi (Y)
        shift = mpmath.mpf(1) / 4 if which == "J" else mpmath.mpf(3) / 4
        j0 = max(0, int(mpmath.floor(x0 / mpmath.pi - n / 2 + shift)) - 2)
        while j0 > 0 and zero(n, j0) > x0:
            j0 -= 1
        while zero(n, j0 + 1) <= x0:
            j0 += 1
        return zero(n, j0 + k)

    mpfun(name, 2, two, notes=f"{name}[n, k]: k-th positive zero of Bessel{which}[n, x] (real n >= 0); "
                              f"mpmath bessel{which.lower()}zero")
    mpfun(name, 3, three, notes=f"{name}[n, k, x0]: k-th zero of Bessel{which}[n, x] greater than x0")


_bessel_zero("J")
_bessel_zero("Y")


# ───────────────────────────── Anger, Weber, Struve L, Lommel, Kelvin, Scorer ─────────────────────────────

def _anger3(nu, mu, z, weber=False):
    trig = mpmath.sin if weber else mpmath.cos
    return mpmath.quad(lambda t: (2 * mpmath.sin(t)) ** mu * trig(nu * t - z * mpmath.sin(t)),
                       mpmath.linspace(0, mpmath.pi, 5)) / mpmath.pi


def _anger_exact(nu, z):
    if z == 0:
        return sp.sin(sp.pi * nu) / (sp.pi * nu) if nu != 0 else sp.Integer(1)
    if nu.is_Integer:
        return sp.besselj(nu, z) if z.is_number and z == 0 else None
    return None


mpfun("AngerJ", 2, mpmath.angerj,
      lambda nu, z: (sp.Integer(1) if nu == 0 else sp.sin(sp.pi * nu) / (sp.pi * nu)) if z == 0 else None,
      real=_allreal, notes="AngerJ[nu, z] = 1/Pi Integrate[Cos[nu t - z Sin[t]], {t, 0, Pi}]; mpmath angerj"
                           ". Check: reference (differential cases in tests/library).")
mpfun("AngerJ", 3, lambda nu, mu, z: _anger3(nu, mu, z),
      notes="associated Anger function AngerJ[nu, mu, z] = 1/Pi Integrate[(2 Sin[t])^mu Cos[nu t - z Sin[t]], "
            "{t, 0, Pi}] by quadrature (Re mu > -1)", real=_allreal)
mpfun("WeberE", 2, mpmath.webere,
      lambda nu, z: (sp.Integer(0) if nu == 0 else (1 - sp.cos(sp.pi * nu)) / (sp.pi * nu)) if z == 0 else None,
      real=_allreal, notes="WeberE[nu, z] = 1/Pi Integrate[Sin[nu t - z Sin[t]], {t, 0, Pi}]; mpmath webere"
                           ". Check: reference (differential cases in tests/library).")
mpfun("WeberE", 3, lambda nu, mu, z: _anger3(nu, mu, z, weber=True),
      notes="associated Weber function WeberE[nu, mu, z] = 1/Pi Integrate[(2 Sin[t])^mu Sin[nu t - z Sin[t]], "
            "{t, 0, Pi}] by quadrature (Re mu > -1)", real=_allreal)


def _struvel_exact(n, z):
    if z == 0 and (n.is_real and n > -1):
        return sp.Integer(0)
    if n == sp.Rational(1, 2):
        return sp.sqrt(2 / (sp.pi * z)) * (sp.cosh(z) - 1)
    if n == -sp.Rational(1, 2):
        return sp.sqrt(2 / (sp.pi * z)) * sp.sinh(z)
    return None


mpfun("StruveL", 2, mpmath.struvel, _struvel_exact,
      real=lambda n, z: _allreal(n, z) and z >= 0,
      notes="modified Struve function L_n(z); mpmath struvel; half-integer orders 1/2, -1/2 in closed form"
            ". Check: reference (differential cases in tests/library).")


def _lommel_exact(mu, nu, z):
    return sp.Integer(0) if z == 0 and mu.is_real and (mu + 1) > 0 else None


mpfun("LommelS1", 3, mpmath.lommels1, _lommel_exact, real=lambda m, n, z: _allreal(m, n, z) and z > 0,
      notes="Lommel function s_{mu,nu}(z) (solution of z^2 y'' + z y' + (z^2 - nu^2) y = z^(mu+1)); mpmath lommels1")
mpfun("LommelS2", 3, mpmath.lommels2, real=lambda m, n, z: _allreal(m, n, z) and z > 0,
      notes="Lommel function S_{mu,nu}(z); mpmath lommels2")


def _kelvin(name, f):
    def one(z):
        return f(0, z)

    def ex1(z):
        if z == 0:
            return {"KelvinBer": sp.Integer(1), "KelvinBei": sp.Integer(0), "KelvinKer": sp.oo,
                    "KelvinKei": -sp.pi / 4}[name]
        return None

    def ex2(n, z):
        return ex1(z) if n == 0 else (sp.Integer(0) if z == 0 and n.is_Integer and n > 0
                                      and name in ("KelvinBer", "KelvinBei") else None)
    realq = (lambda *v: _allreal(*v) and v[-1] >= 0)
    mpfun(name, 1, one, ex1, real=realq,
          notes=f"{name}[z] = {name}[0, z]; Kelvin function via mpmath ({name[6:].lower()})")
    mpfun(name, 2, f, ex2, real=realq,
          notes=f"{name}[n, z]: Kelvin function {name[6:].lower()}_n(z) (ber_n + i bei_n = J_n(z e^(3 Pi i/4))); "
                "mpmath")


_kelvin("KelvinBer", mpmath.ber)
_kelvin("KelvinBei", mpmath.bei)
_kelvin("KelvinKer", mpmath.ker)
_kelvin("KelvinKei", mpmath.kei)


_G23 = sp.gamma(sp.Rational(2, 3))
_G13 = sp.gamma(sp.Rational(1, 3))


def _scorer_prime(z, hi=False):
    # the formula cancels badly for large positive z (Bi' huge times a tiny tail integral): raise the precision
    return _rel_stable(lambda: _scorer_prime0(z, hi), "Scorer" + ("Hi" if hi else "Gi") + "Prime")


def _scorer_prime0(z, hi=False):
    # Gi'(z) = Bi'(z) Integral[Ai, {t, z, Infinity}] + Ai'(z) Integral[Bi, {t, 0, z}]  (DLMF 9.12.?, from Gi);
    # Hi = Bi - Gi (DLMF 9.12.15)
    ai_int = mpmath.airyai(z, derivative=-1)
    bi_int = mpmath.airybi(z, derivative=-1)
    gip = mpmath.airybi(z, 1) * (mpmath.mpf(1) / 3 - ai_int) + mpmath.airyai(z, 1) * bi_int
    return mpmath.airybi(z, 1) - gip if hi else gip


mpfun("ScorerGi", 1, mpmath.scorergi, lambda z: 1 / (3 ** sp.Rational(7, 6) * _G23) if z == 0 else None,
      real=_allreal, notes="Scorer function Gi(z) (Gi'' - z Gi = -1/Pi); mpmath scorergi; Gi(0) = 1/(3^(7/6) Gamma(2/3))"
                           ". Check: reference (differential cases in tests/library).")
mpfun("ScorerHi", 1, mpmath.scorerhi, lambda z: 2 / (3 ** sp.Rational(7, 6) * _G23) if z == 0 else None,
      real=_allreal, notes="Scorer function Hi(z) (Hi'' - z Hi = 1/Pi); mpmath scorerhi; Hi(0) = 2/(3^(7/6) Gamma(2/3))"
                           ". Check: reference (differential cases in tests/library).")
mpfun("ScorerGiPrime", 1, lambda z: _scorer_prime(z), lambda z: 1 / (3 ** sp.Rational(5, 6) * _G13) if z == 0 else None,
      real=_allreal, notes="Gi'(z) = Bi'(z) Integral[Ai, {z, Infinity}] + Ai'(z) Integral[Bi, {0, z}]; "
                           "Gi'(0) = 1/(3^(5/6) Gamma(1/3))"
                           ". Check: reference (differential cases in tests/library).")
mpfun("ScorerHiPrime", 1, lambda z: _scorer_prime(z, True),
      lambda z: 2 / (3 ** sp.Rational(5, 6) * _G13) if z == 0 else None,
      real=_allreal, notes="Hi'(z) = Bi'(z) - Gi'(z) (Gi + Hi = Bi); Hi'(0) = 2/(3^(5/6) Gamma(1/3))"
                           ". Check: reference (differential cases in tests/library).")


# ───────────────────────────── Whittaker, parabolic cylinder, Coulomb ─────────────────────────────

def _whitm_exact(k, m, z):
    return sp.Integer(0) if z == 0 and (m + sp.Rational(1, 2)).is_positive else None


mpfun("WhittakerM", 3, mpmath.whitm, _whitm_exact, real=lambda k, m, z: _allreal(k, m, z) and z > 0,
      notes="WhittakerM[k, m, z] = E^(-z/2) z^(m+1/2) Hypergeometric1F1[1/2 + m - k, 1 + 2 m, z]; mpmath whitm"
            ". Check: reference (differential cases in tests/library).")
mpfun("WhittakerW", 3, mpmath.whitw, real=lambda k, m, z: _allreal(k, m, z) and z > 0,
      notes="WhittakerW[k, m, z] (decays like E^(-z/2) z^k); mpmath whitw"
            ". Check: reference (differential cases in tests/library).")


def _pcfd_exact(nu, z):
    if nu == 0:
        return sp.exp(-z ** 2 / 4)
    if z == 0:
        g = sp.gamma((1 - nu) / 2)
        return sp.Integer(0) if g is sp.zoo else 2 ** (nu / 2) * sp.sqrt(sp.pi) / g
    return None


mpfun("ParabolicCylinderD", 2, mpmath.pcfd, _pcfd_exact, symbolic=lambda nu, z: _pcfd_exact(nu, z) if nu == 0 else None,
      real=_allreal, notes="parabolic cylinder function D_nu(z); mpmath pcfd; D_0(z) = E^(-z^2/4), "
                           "D_nu(0) = 2^(nu/2) Sqrt[Pi]/Gamma[(1-nu)/2]"
                           ". Check: reference (differential cases in tests/library).")


def _coulomb_real(l, eta, r):
    return _allreal(l, eta, r) and r > 0


mpfun("CoulombF", 3, mpmath.coulombf, lambda l, eta, r: sp.Integer(0) if r == 0 and l.is_real and l > -1 else None,
      real=_coulomb_real, notes="regular Coulomb wave function F_l(eta, r); mpmath coulombf"
                                ". Check: reference (differential cases in tests/library).")
mpfun("CoulombG", 3, mpmath.coulombg, real=_coulomb_real,
      notes="irregular Coulomb wave function G_l(eta, r); mpmath coulombg"
            ". Check: reference (differential cases in tests/library).")
mpfun("CoulombH1", 3, lambda l, eta, r: mpmath.coulombg(l, eta, r) + 1j * mpmath.coulombf(l, eta, r),
      notes="outgoing Coulomb wave function H+ = G + i F (~ exp(i (r - eta Log[2 r] ...)) for large r)"
            ". Check: reference (differential cases in tests/library).")
mpfun("CoulombH2", 3, lambda l, eta, r: mpmath.coulombg(l, eta, r) - 1j * mpmath.coulombf(l, eta, r),
      notes="incoming Coulomb wave function H- = G - i F"
            ". Check: reference (differential cases in tests/library).")


# ───────────────────────────── arithmetic-geometric mean, Carlson integrals ─────────────────────────────

def _agm_exact(a, b):
    if a == 0 or b == 0:
        return sp.Integer(0)
    if a == b:
        return a
    return None


def _agm(a, b):
    """Wolfram's branch: AGM[a, b] = Pi (a + b) / (4 EllipticK[((a - b)/(a + b))^2]) (equals the limit of the
    arithmetic-geometric iteration for a, b > 0; homogeneous, so AGM[-2, -5] = -AGM[2, 5])."""
    if a + b == 0 or a == 0 or b == 0:
        return mpmath.mpf(0)
    if a == b:
        return a
    m = ((a - b) / (a + b)) ** 2
    if m == 1:
        return mpmath.mpf(0)
    return mpmath.pi * (a + b) / (4 * mpmath.ellipk(m))


def _agm_exact2(a, b):
    if a + b == 0:
        return sp.Integer(0)
    return _agm_exact(a, b)


mpfun("ArithmeticGeometricMean", 2, _agm, _agm_exact2,
      real=lambda a, b: _allreal(a, b) and ((a >= 0 and b >= 0) or (a <= 0 and b <= 0)),
      notes="ArithmeticGeometricMean[a, b] = Pi (a + b)/(4 EllipticK[((a - b)/(a + b))^2]) (Wolfram's branch; the "
            "arithmetic-geometric iteration for positive a, b)"
            ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")


def _carlson(name, nargs, f, exact=None, notes="", pv=False):
    # pv: the last argument may be negative (Cauchy principal value, the real part of the limit)
    def real(*v):
        head = v[:-1] if pv else v
        return _allreal(*v) and all(x >= 0 for x in head)
    mpfun(name, nargs, f, exact, listable=True, real=real, notes=notes)


def _rf_exact(x, y, z):
    if x == y == z:
        return 1 / sp.sqrt(x) if x != 0 else sp.oo
    return None


_carlson("CarlsonRC", 2, mpmath.elliprc,
         lambda x, y: (1 / sp.sqrt(x) if x != 0 else None) if x == y else None,
         notes="Carlson R_C(x, y) = 1/2 Integrate[(t + x)^(-1/2) (t + y)^(-1), {t, 0, Infinity}] (Cauchy "
               "principal value for y < 0); mpmath elliprc"
               ". Check: reference (differential cases in tests/library).", pv=True)
_carlson("CarlsonRF", 3, mpmath.elliprf, _rf_exact,
         notes="Carlson R_F(x, y, z) = 1/2 Integrate[((t+x)(t+y)(t+z))^(-1/2), {t, 0, Infinity}]; mpmath elliprf"
               ". Check: reference (differential cases in tests/library).")
_carlson("CarlsonRD", 3, mpmath.elliprd,
         lambda x, y, z: (x ** sp.Rational(-3, 2) if x != 0 else None) if x == y == z else None,
         notes="Carlson R_D(x, y, z) = R_J(x, y, z, z); mpmath elliprd"
               ". Check: reference (differential cases in tests/library).")
_carlson("CarlsonRJ", 4, mpmath.elliprj,
         lambda x, y, z, p: (x ** sp.Rational(-3, 2) if x != 0 else None) if x == y == z == p else None,
         notes="Carlson R_J(x, y, z, p) = 3/2 Integrate[(t+p)^-1 ((t+x)(t+y)(t+z))^(-1/2), {t, 0, Infinity}] "
               "(principal value for p < 0); mpmath elliprj"
               ". Check: reference (differential cases in tests/library).", pv=True)
_carlson("CarlsonRG", 3, mpmath.elliprg,
         lambda x, y, z: (sp.sqrt(x) if x == y == z else None),
         notes="Carlson R_G(x, y, z); mpmath elliprg"
               ". Check: reference (differential cases in tests/library).")
_carlson("CarlsonRK", 2, lambda x, y: 2 / mpmath.pi * mpmath.elliprf(0, x, y),
         lambda x, y: (1 / sp.sqrt(x) if x != 0 else None) if x == y else None,
         notes="CarlsonRK[x, y] = 1/Pi Integrate[t^(-1/2) (t+x)^(-1/2) (t+y)^(-1/2), {t, 0, Infinity}] "
               "= 2/Pi R_F(0, x, y)")
_carlson("CarlsonRE", 2, lambda x, y: 4 / mpmath.pi * mpmath.elliprg(0, x, y),
         lambda x, y: (sp.sqrt(x) if x == y else None),
         notes="CarlsonRE[x, y] = 4/Pi R_G(0, x, y) (the Wolfram definition with the 1/Pi integral)")
_carlson("CarlsonRM", 3, lambda x, y, p: 4 / (3 * mpmath.pi) * mpmath.elliprj(0, x, y, p),
         lambda x, y, p: (x ** sp.Rational(-3, 2) if x != 0 else None) if x == y == p else None,
         notes="CarlsonRM[x, y, rho] = 2/Pi Integrate[t^(-1/2) (t+x)^(-1/2) (t+y)^(-1/2) (t+rho)^(-1), "
               "{t, 0, Infinity}] = 4/(3 Pi) R_J(0, x, y, rho) (principal value for rho < 0)", pv=True)


# ───────────────────────────── theta functions, nome, modular functions ─────────────────────────────

def _check_q(q, name):
    if abs(q) >= 1:
        raise ValueError(f"{name}: the nome q must satisfy Abs[q] < 1")


def _rel_stable(f, name):
    """Evaluate f() at increasing working precision until two successive values agree to the current precision
    in the *relative* sense (guards against cancellation in theta-type series, e.g. q close to 1)."""
    P = mpmath.mp.prec
    work, prev = P, None
    while True:
        with mpmath.workprec(work):
            v = f()
        if prev is not None:
            if v == 0 and prev == 0:
                return v
            if abs(v - prev) <= mpmath.mpf(2) ** (-(P - 6)) * abs(v):
                return v
        prev = v
        if work > max(8 * P, P + 4000):
            raise ArithmeticError(f"{name}: catastrophic cancellation; the value could not be computed to the "
                                  "requested precision")
        work = P + 64 if work == P else 2 * work


def _theta(a, u, q, d=0, name="EllipticTheta"):
    a = _realval(a, name, "a")
    if not mpmath.isint(a) or not 1 <= a <= 4:
        raise ValueError(f"{name}: the index a must be 1, 2, 3 or 4")
    _check_q(q, name)
    if q == 0:
        if d == 0:
            return mpmath.mpf(1) if int(a) in (3, 4) else mpmath.mpf(0)
        return mpmath.mpf(0)
    if u == 0 and (int(a) == 1) == (d % 2 == 0):
        return mpmath.mpf(0)      # odd function of u (theta_1 and odd derivatives of theta_2..4) vanishes at 0
    return _rel_stable(lambda: mpmath.jtheta(int(a), u, q, d), name)


def _theta_exact(a, u, q, d=0):
    if a.is_Integer and a in (1, 2, 3, 4):
        if q == 0:
            return sp.Integer(1) if (d == 0 and a in (3, 4)) else sp.Integer(0)
        if u == 0 and ((d == 0 and a == 1) or (d == 1 and a in (2, 3, 4))):
            return sp.Integer(0)
    return None


def _theta_real(a, u, q):
    return _allreal(a, u, q) and (q >= 0 or (a.is_Integer and a in (3, 4)))


mpfun("EllipticTheta", 3, lambda a, u, q: _theta(a, u, q), _theta_exact, real=_theta_real,
      notes="Jacobi theta function theta_a(u, q) (a = 1..4) in the nome convention of the Wolfram reference "
            "(theta_1 = 2 q^(1/4) Sum[(-1)^n q^(n(n+1)) Sin[(2n+1) u]] ...); mpmath jtheta; Abs[q] < 1"
            ". Check: reference (differential cases in tests/library).")
mpfun("EllipticTheta", 2, lambda a, q: _theta(a, 0, q), lambda a, q: _theta_exact(a, sp.Integer(0), q),
      real=lambda a, q: _theta_real(a, sp.Integer(0), q),
      notes="theta constant EllipticTheta[a, q] = EllipticTheta[a, 0, q]")
mpfun("EllipticThetaPrime", 3, lambda a, u, q: _theta(a, u, q, 1, "EllipticThetaPrime"),
      lambda a, u, q: _theta_exact(a, u, q, 1), real=_theta_real,
      notes="derivative of EllipticTheta[a, u, q] with respect to u; mpmath jtheta(..., 1)"
            ". Check: reference (differential cases in tests/library).")
mpfun("EllipticThetaPrime", 2, lambda a, q: _theta(a, 0, q, 1, "EllipticThetaPrime"),
      lambda a, q: _theta_exact(a, sp.Integer(0), q, 1), real=lambda a, q: _theta_real(a, sp.Integer(0), q),
      notes="EllipticThetaPrime[a, q] = EllipticThetaPrime[a, 0, q]")


def _nome_exact(m):
    if m == 0:
        return sp.Integer(0)
    if m == 1:
        return sp.Integer(1)
    if m == sp.Rational(1, 2):
        return sp.exp(-sp.pi)
    return None


def _nome(m):
    if m == 1:
        return mpmath.mpf(1)
    return mpmath.qfrom(m=m)


def _inv_nome(q):
    _check_q(q, "InverseEllipticNomeQ")
    return mpmath.mfrom(q=q)


mpfun("EllipticNomeQ", 1, _nome, _nome_exact, real=lambda m: _allreal(m) and m <= 1,
      notes="nome q(m) = Exp[-Pi EllipticK[1 - m]/EllipticK[m]]; mpmath qfrom"
            ". Check: reference (differential cases in tests/library).")
mpfun("InverseEllipticNomeQ", 1, _inv_nome,
      lambda q: sp.Integer(0) if q == 0 else (sp.Rational(1, 2) if sp.simplify(q - sp.exp(-sp.pi)) == 0 else None),
      real=lambda q: _allreal(q), notes="parameter m with EllipticNomeQ[m] == q (Abs[q] < 1); mpmath mfrom"
                                        ". Check: reference (differential cases in tests/library).")


def _upper(tau, name):
    if mpmath.im(tau) <= 0:
        raise ValueError(f"{name}: the argument must lie in the upper half-plane (Im[tau] > 0)")


def _eta(tau):
    _upper(tau, "DedekindEta")
    return mpmath.eta(tau)


def _klein(tau):
    _upper(tau, "KleinInvariantJ")
    return mpmath.kleinj(tau)


def _lambda(tau):
    _upper(tau, "ModularLambda")
    q = mpmath.exp(1j * mpmath.pi * tau)
    return (mpmath.jtheta(2, 0, q) / mpmath.jtheta(3, 0, q)) ** 4


_RHO = sp.Rational(1, 2) + sp.I * sp.sqrt(3) / 2


def _eq(a, b):
    try:
        return sp.simplify(sp.nsimplify(a) - b) == 0 if a.is_number else False
    except Exception:
        return False


def _im_tau(tau):
    return sp.im(tau).is_positive


mpfun("DedekindEta", 1, _eta,
      lambda t: sp.gamma(sp.Rational(1, 4)) / (2 * sp.pi ** sp.Rational(3, 4)) if t == sp.I else None,
      real=lambda t: _allreal(sp.re(t)) and sp.re(t) == 0,
      notes="Dedekind eta(tau) = q^(1/24) Product[1 - q^n] with q = E^(2 Pi I tau), Im tau > 0; mpmath eta; "
            "eta(I) = Gamma[1/4]/(2 Pi^(3/4))"
            ". Check: reference (differential cases in tests/library).")
mpfun("KleinInvariantJ", 1, _klein,
      lambda t: sp.Integer(1) if t == sp.I else (sp.Integer(0) if (_eq(t, _RHO) or _eq(t, _RHO - 1)) else None),
      real=lambda t: sp.re(t) == 0 or abs(sp.re(t)) == sp.Rational(1, 2),
      notes="Klein invariant J(tau) = g2^3/(g2^3 - 27 g3^2) (J(I) = 1, J(E^(2 Pi I/3)) = 0); mpmath kleinj"
            ". Check: reference (differential cases in tests/library).")
mpfun("ModularLambda", 1, _lambda, lambda t: sp.Rational(1, 2) if t == sp.I else None,
      real=lambda t: sp.re(t) == 0,
      notes="modular lambda(tau) = EllipticTheta[2, 0, q]^4/EllipticTheta[3, 0, q]^4, q = E^(I Pi tau)"
            ". Check: reference (differential cases in tests/library).")


# ───────────────────────────── Neville theta functions ─────────────────────────────

def _neville(kind):
    name = "NevilleTheta" + kind

    def mpf(z, m):
        if m == 0:
            return {"S": mpmath.sin(z), "C": mpmath.cos(z), "D": mpmath.mpf(1), "N": mpmath.mpf(1)}[kind]
        if m == 1:      # limits: theta_s -> Sinh, theta_n -> Cosh, theta_c, theta_d -> 1 (sn -> Tanh, cn = dn -> Sech)
            return {"S": mpmath.sinh(z), "C": mpmath.mpf(1), "D": mpmath.mpf(1), "N": mpmath.cosh(z)}[kind]
        return _rel_stable(lambda: _nev(z, m), name)

    def _nev(z, m):
        K = mpmath.ellipk(m)
        q = mpmath.qfrom(m=m)
        v = mpmath.pi * z / (2 * K)
        if kind == "S":
            return mpmath.sqrt(2 * mpmath.pi) / (2 * (1 - m) ** 0.25 * m ** 0.25 * mpmath.sqrt(K)) * \
                mpmath.jtheta(1, v, q)
        if kind == "C":
            return mpmath.sqrt(2 * mpmath.pi) / (2 * m ** 0.25 * mpmath.sqrt(K)) * mpmath.jtheta(2, v, q)
        if kind == "D":
            return mpmath.sqrt(mpmath.pi / (2 * K)) * mpmath.jtheta(3, v, q)
        return mpmath.sqrt(mpmath.pi / (2 * K)) / (1 - m) ** 0.25 * mpmath.jtheta(4, v, q)

    def ex(z, m):
        if m == 0:
            return {"S": sp.sin(z), "C": sp.cos(z), "D": sp.Integer(1), "N": sp.Integer(1)}[kind]
        if m == 1:
            return {"S": sp.sinh(z), "C": sp.Integer(1), "D": sp.Integer(1), "N": sp.cosh(z)}[kind]
        if z == 0:
            return {"S": sp.Integer(0), "C": sp.Integer(1), "D": sp.Integer(1), "N": sp.Integer(1)}[kind]
        return None
    mpfun(name, 2, mpf, ex, symbolic=ex, real=lambda z, m: _allreal(z, m) and 0 <= m < 1,
          notes=f"Neville theta function theta_{kind.lower()}(z | m) (normalised so that JacobiSN = theta_s/theta_n, "
                "JacobiCN = theta_c/theta_n, JacobiDN = theta_d/theta_n); from EllipticTheta with v = Pi z/(2 K(m))")


for _k in "SCDN":
    _neville(_k)


# ───────────────────────────── Jacobi amplitude, epsilon, zeta ─────────────────────────────

def _am(u, m):
    if isinstance(u, mpmath.mpc) or isinstance(m, mpmath.mpc):
        if mpmath.im(u) != 0 or mpmath.im(m) != 0:
            raise NotImplementedError("JacobiAmplitude: complex arguments are not supported")
        u, m = mpmath.re(u), mpmath.re(m)
    if m == 0:
        return u
    if m == 1:
        return 2 * mpmath.atan(mpmath.tanh(u / 2))
    if m > 1:
        # |sn| <= 1/Sqrt[m] < 1, so am = ArcSin[sn] is continuous and bounded
        return mpmath.asin(mpmath.re(mpmath.ellipfun("sn", u, m=m)))
    K = mpmath.ellipk(m)
    n = mpmath.nint(u / (2 * K))
    r = u - 2 * n * K
    sn, cn = mpmath.ellipfun("sn", r, m=m), mpmath.ellipfun("cn", r, m=m)
    return mpmath.atan2(mpmath.re(sn), mpmath.re(cn)) + n * mpmath.pi


def _am_exact(u, m):
    if u == 0:
        return sp.Integer(0)
    if m == 0:
        return u
    return None


mpfun("JacobiAmplitude", 2, _am, _am_exact, symbolic=_am_exact,
      notes="Jacobi amplitude am(u | m) (the continuous inverse of EllipticF[phi, m] in phi): reduced by "
            "am(u + 2K) = am(u) + Pi, ArcTan[cn, sn] on the principal interval; am(u|1) = Gudermannian; "
            "real arguments only (complex arguments raise)"
            ". Check: reference (differential cases in tests/library).")


def _zeta_theta(u, m):
    """Jacobi zeta Z(u | m) = d/du log theta_4(Pi u/(2K)) (DLMF 22.16.32)."""
    K = mpmath.ellipk(m)
    q = mpmath.qfrom(m=m)
    v = mpmath.pi * u / (2 * K)
    return mpmath.pi / (2 * K) * mpmath.jtheta(4, v, q, 1) / mpmath.jtheta(4, v, q)


def _check_m_lt1(m, name):
    if not isinstance(m, mpmath.mpc) or m.imag == 0:
        if mpmath.re(m) > 1:
            raise NotImplementedError(f"{name}: real m > 1 is not supported")


def _zn(u, m):
    if m == 0:
        return mpmath.mpf(0)
    if m == 1:
        return mpmath.tanh(u)
    _check_m_lt1(m, "JacobiZN")
    return _zeta_theta(u, m)


def _eps_fn(u, m):
    if m == 0:
        return u
    if m == 1:
        return mpmath.tanh(u)
    _check_m_lt1(m, "JacobiEpsilon")
    return _zeta_theta(u, m) + u * mpmath.ellipe(m) / mpmath.ellipk(m)


mpfun("JacobiZN", 2, _zn, lambda u, m: sp.Integer(0) if (u == 0 or m == 0) else (sp.tanh(u) if m == 1 else None),
      symbolic=lambda u, m: sp.Integer(0) if (u == 0 or m == 0) else (sp.tanh(u) if m == 1 else None),
      real=lambda u, m: _allreal(u, m) and m <= 1,
      notes="Jacobi zeta function Z(u | m) = Integrate[dn^2 - E(m)/K(m), {0, u}] via theta_4'/theta_4; "
            "real m > 1 raises"
            ". Check: reference (differential cases in tests/library).")
mpfun("JacobiEpsilon", 2, _eps_fn,
      lambda u, m: sp.Integer(0) if u == 0 else (u if m == 0 else (sp.tanh(u) if m == 1 else None)),
      symbolic=lambda u, m: sp.Integer(0) if u == 0 else (u if m == 0 else (sp.tanh(u) if m == 1 else None)),
      real=lambda u, m: _allreal(u, m) and m <= 1,
      notes="Jacobi epsilon function Integrate[dn(t|m)^2, {t, 0, u}] = Z(u|m) + u E(m)/K(m); real m > 1 raises"
            ". Check: reference (differential cases in tests/library).")


def _jzeta(phi, m):
    if m == 0:
        return mpmath.mpf(0)
    return mpmath.ellipe(phi, m) - mpmath.ellipe(m) / mpmath.ellipk(m) * mpmath.ellipf(phi, m)


mpfun("JacobiZeta", 2, _jzeta, lambda p, m: sp.Integer(0) if (p == 0 or m == 0) else None,
      symbolic=lambda p, m: sp.Integer(0) if (p == 0 or m == 0) else None,
      real=lambda p, m: _allreal(p, m) and m < 1,
      notes="Jacobi zeta Z(phi | m) = EllipticE[phi, m] - EllipticE[m] EllipticF[phi, m]/EllipticK[m] "
            "(amplitude argument, as in Wolfram); mpmath ellipe/ellipf"
            ". Check: reference (differential cases in tests/library).")


# inverse Jacobi functions missing from the earlier batch (DC, DS, ND), as EllipticF[phi, m] on principal ranges
def _inv_jacobi(kind):
    name = "InverseJacobi" + kind

    def mpf(v, m):
        if isinstance(v, mpmath.mpc) or isinstance(m, mpmath.mpc):
            raise NotImplementedError(f"{name}: complex arguments are not supported")
        F = mpmath.ellipf
        if kind == "ND":
            # nd = 1/dn on [0, K]: from 1 up to 1/Sqrt[1 - m]  (0 < m < 1)
            if not (0 < m < 1) or not (1 <= v <= 1 / mpmath.sqrt(1 - m)):
                raise ValueError(f"{name}: needs 0 < m < 1 and 1 <= v <= 1/Sqrt[1 - m]")
            return F(mpmath.asin(mpmath.sqrt((1 - 1 / v ** 2) / m)), m)
        if kind == "DC":
            # dc = dn/cn on [0, K): from 1 to Infinity  (0 <= m < 1)
            if not (0 <= m < 1) or v < 1:
                raise ValueError(f"{name}: needs 0 <= m < 1 and v >= 1")
            return F(mpmath.asin(mpmath.sqrt((v ** 2 - 1) / (v ** 2 - m))), m)
        # DS: ds = dn/sn, odd, decreasing from Infinity to Sqrt[1 - m] on (0, K]
        if not (0 <= m < 1) or abs(v) < mpmath.sqrt(1 - m):
            raise ValueError(f"{name}: needs 0 <= m < 1 and Abs[v] >= Sqrt[1 - m]")
        r = F(mpmath.asin(1 / mpmath.sqrt(v ** 2 + m)), m)
        return r if v > 0 else -r

    def ex(v, m):
        if kind in ("ND", "DC") and v == 1:
            return sp.Integer(0)
        if kind == "DC" and m == 0 and v.is_number:
            return sp.acos(1 / v)
        if kind == "DS" and m == 0 and v.is_number:
            return sp.acot(v)
        return None
    mpfun(name, 2, mpf, ex, symbolic=ex,
          notes=f"inverse of Jacobi{kind}(u | m) on its principal real range, via EllipticF (DLMF 22.15); "
                "complex arguments and arguments outside that range raise")


for _k in ("DC", "DS", "ND"):
    _inv_jacobi(_k)


# ───────────────────────────── Weierstrass lattice functions ─────────────────────────────
#
# Half-periods follow the Wolfram choice (checked against the reference examples): {w1, w3} is the lattice
# basis with tau = w3/w1 reduced to the fundamental domain -1/2 < Re tau <= 1/2, |tau| >= 1 (Re tau >= 0 on
# |tau| = 1), and Re w1 > 0 (Im w1 > 0 when w1 is imaginary); w2 = -w1 - w3.  e_i = P(w_i), eta_i = zeta(w_i).

def _tol():
    return mpmath.mpf(2) ** (-(mpmath.mp.prec * 2) // 3)


def _invariants_pair(g, name):
    if not isinstance(g, list) or len(g) != 2 or any(isinstance(x, list) for x in g):
        raise ValueError(f"{name}: invariants must be a list {{g2, g3}}")
    return g[0], g[1]


def _reduce_basis(w1, w3):
    """SL(2, Z) reduction of the half-period basis (w1, w3), Im(w3/w1) > 0, to the Wolfram fundamental domain."""
    if mpmath.im(w3 / w1) < 0:
        w3 = -w3
    tol = _tol()
    for _ in range(10000):
        tau = w3 / w1
        n = mpmath.nint(mpmath.re(tau))
        if n:
            w3 -= n * w1
            tau = w3 / w1
        if abs(tau) < 1 - tol:
            w1, w3 = w3, -w1
            continue
        break
    tau = w3 / w1
    if mpmath.re(tau) < -mpmath.mpf(1) / 2 + tol:
        w3 += w1
        tau = w3 / w1
    if abs(abs(tau) - 1) < tol and mpmath.re(tau) < -tol:
        w1, w3 = w3, -w1
    # among the bases with the same reduced tau (sign changes; rotations when |tau| = 1) take w1 with
    # Arg[w1] in (-Pi/2, Pi/2] of least absolute value, ties to the positive argument
    cands = [(w1, w3)]
    if abs(abs(tau) - 1) < tol:
        a, b = w1, w3
        for _ in range(6):
            a, b = b, -a
            if mpmath.re(b / a) < -mpmath.mpf(1) / 2 + tol:
                b += a
            if abs(b / a - tau) < tol * 10:
                cands.append((a, b))
    cands += [(-a, -b) for a, b in cands]

    def key(c):
        arg = mpmath.arg(c[0])
        if abs(arg + mpmath.pi / 2) < tol or abs(arg - mpmath.pi) < tol or arg < -mpmath.pi / 2 or arg > mpmath.pi / 2 + tol:
            return (1, 0, 0)
        return (0, abs(arg) - (0 if abs(arg) < tol else 0), -arg)
    best = min(cands, key=lambda c: (key(c)[0], round(float(abs(mpmath.arg(c[0]))), 12), -float(mpmath.arg(c[0]))))
    return best


def _theta_e(w1, w3):
    """e1, e2, e3 for the lattice with half-periods w1, w3 (DLMF 23.6.2-4)."""
    tau = w3 / w1
    q = mpmath.exp(1j * mpmath.pi * tau)
    t2, t4 = mpmath.jtheta(2, 0, q) ** 4, mpmath.jtheta(4, 0, q) ** 4
    c = mpmath.pi ** 2 / (12 * w1 ** 2)
    return c * (t2 + 2 * t4), c * (t2 - t4), -c * (2 * t2 + t4)


def _invariants(w1, w3):
    if w1 == 0 or mpmath.im(w3 / w1) == 0:
        raise ValueError("WeierstrassInvariants: the half-periods must have a non-real ratio")
    if mpmath.im(w3 / w1) < 0:
        w3 = -w3
    w1, w3 = _reduce_basis(w1, w3)
    e1, e2, e3 = _theta_e(w1, w3)
    return 2 * (e1 ** 2 + e2 ** 2 + e3 ** 2), 4 * e1 * e2 * e3


def _half_periods(g2, g3, name="WeierstrassHalfPeriods"):
    disc = g2 ** 3 - 27 * g3 ** 2
    scale = max(abs(g2) ** 3, 27 * abs(g3) ** 2, mpmath.mpf(1) * 0 + _eps())
    if abs(disc) <= scale * _tol():
        raise NotImplementedError(f"{name}: degenerate invariants (g2^3 == 27 g3^2) are not supported")
    roots = mpmath.polyroots([4, 0, -g2, -g3], maxsteps=400, extraprec=2 * mpmath.mp.prec)
    import itertools
    best = None
    for e1, e2, e3 in itertools.permutations(roots):
        m = (e2 - e3) / (e1 - e3)
        w = mpmath.sqrt(e1 - e3)
        wa, wb = mpmath.ellipk(m) / w, 1j * mpmath.ellipk(1 - m) / w
        if mpmath.im(wb / wa) == 0:
            continue
        try:
            h2, h3 = _invariants(wa, wb)
        except (ValueError, ZeroDivisionError):
            continue
        err = abs(h2 - g2) + abs(h3 - g3)
        if best is None or err < best[0]:
            best = (err, wa, wb)
        if err <= _tol() * (1 + abs(g2) + abs(g3)):
            break
    if best is None or best[0] > mpmath.mpf(2) ** (-mpmath.mp.prec // 3) * (1 + abs(g2) + abs(g3)):
        raise ArithmeticError(f"{name}: could not determine the period lattice")
    return _reduce_basis(best[1], best[2])


def _etas(w1, w3):
    tau = w3 / w1
    q = mpmath.exp(1j * mpmath.pi * tau)
    eta1 = -mpmath.pi ** 2 * mpmath.jtheta(1, 0, q, 3) / (12 * w1 * mpmath.jtheta(1, 0, q, 1))
    eta3 = (eta1 * w3 - 1j * mpmath.pi / 2) / w1          # Legendre relation eta1 w3 - eta3 w1 = I Pi/2
    return eta1, -eta1 - eta3, eta3


def _lattice(g):
    g2, g3 = g
    w1, w3 = _half_periods(g2, g3)
    return w1, w3


def _sigma_zeta(u, g, which):
    w1, w3 = _lattice(g)
    if which == "sigma" and u == 0:
        return mpmath.mpf(0)            # sigma is odd: exactly 0 (theta1(0) can carry round-off)
    eta1 = _etas(w1, w3)[0]
    q = mpmath.exp(1j * mpmath.pi * w3 / w1)
    v = mpmath.pi * u / (2 * w1)
    if which == "sigma":
        return 2 * w1 / mpmath.pi * mpmath.exp(eta1 * u ** 2 / (2 * w1)) * mpmath.jtheta(1, v, q) / \
            mpmath.jtheta(1, 0, q, 1)
    t1 = mpmath.jtheta(1, v, q)
    if t1 == 0:
        raise ValueError("WeierstrassZeta: the argument is a lattice point (pole)")
    if which == "zeta":
        return eta1 * u / w1 + mpmath.pi / (2 * w1) * mpmath.jtheta(1, v, q, 1) / t1
    c = mpmath.pi / (2 * w1)
    d1, d2, d3 = (mpmath.jtheta(1, v, q, k) / t1 for k in (1, 2, 3))
    if which == "p":            # P = -zeta'
        return -eta1 / w1 - c ** 2 * (d2 - d1 ** 2)
    return -c ** 3 * (d3 - 3 * d2 * d1 + 2 * d1 ** 3)       # P'


def _gcheck(g, name):
    return _invariants_pair(g, name)


def _real_g(*vals):
    return _allreal(*vals)


_G13c = sp.gamma(sp.Rational(1, 3)) ** 3 / (4 * sp.pi)
_G14c = sp.gamma(sp.Rational(1, 4)) ** 2 / (4 * sp.sqrt(sp.pi))
_EXACT_HP = {(0, 1): (_G13c, sp.exp(sp.I * sp.pi / 3) * _G13c), (1, 0): (_G14c, sp.I * _G14c)}
_EXACT_E1 = {(0, 1): 2 ** sp.Rational(-2, 3), (1, 0): sp.Rational(1, 2)}
_EXACT_ETA1 = {(0, 1): 2 * sp.pi ** 2 / (sp.sqrt(3) * sp.gamma(sp.Rational(1, 3)) ** 3),
               (1, 0): sp.pi ** sp.Rational(3, 2) / sp.gamma(sp.Rational(1, 4)) ** 2}


def _key(g):
    try:
        return (int(g[0]), int(g[1])) if all(x.is_Integer for x in g) else None
    except Exception:
        return None


def _wfun(name, f, exact_table=None, notes="", real_when_disc_pos=False):
    def mpf(g):
        g2, g3 = _gcheck(g, name)
        return f(g2, g3)

    def ex(g):
        if not isinstance(g, list) or len(g) != 2:
            raise ValueError(f"{name}: invariants must be a list {{g2, g3}}")
        if exact_table is not None:
            return exact_table(_key(g))
        return None

    def real(g):
        return real_when_disc_pos and _allreal(g) and (g[0] ** 3 - 27 * g[1] ** 2) > 0
    mpfun(name, 1, mpf, ex, listable=False, real=real, notes=notes)


def _hp_table(i):
    def t(k):
        if k not in _EXACT_HP:
            return None
        w1, w3 = _EXACT_HP[k]
        return {1: w1, 3: w3, 2: -w1 - w3, 0: [w1, w3]}[i]
    return t


_wfun("WeierstrassHalfPeriods", lambda g2, g3: list(_half_periods(g2, g3)), lambda k: _hp_table(0)(k),
      notes="half-periods {w1, w3} for invariants {g2, g3}: the lattice basis with w3/w1 reduced to the "
            "fundamental domain (as in Wolfram); periods from EllipticK of the roots, checked against the invariants"
            ". Check: reference (differential cases in tests/library).")
_wfun("WeierstrassHalfPeriodW1", lambda g2, g3: _half_periods(g2, g3)[0], _hp_table(1),
      notes="half-period w1 (first of WeierstrassHalfPeriods)")
_wfun("WeierstrassHalfPeriodW3", lambda g2, g3: _half_periods(g2, g3)[1], _hp_table(3),
      notes="half-period w3 (second of WeierstrassHalfPeriods)")
_wfun("WeierstrassHalfPeriodW2", lambda g2, g3: -sum(_half_periods(g2, g3)), _hp_table(2),
      notes="half-period w2 = -w1 - w3")


def _e_i(i):
    def f(g2, g3):
        return _theta_e(*_half_periods(g2, g3))[i - 1]
    return f


_wfun("WeierstrassE1", _e_i(1), lambda k: _EXACT_E1.get(k), real_when_disc_pos=True,
      notes="e1 = WeierstrassP[w1, {g2, g3}] (theta-constant formula DLMF 23.6.2 on the Wolfram half-periods)"
            ". Check: reference (differential cases in tests/library).")
_wfun("WeierstrassE2", _e_i(2), real_when_disc_pos=True, notes="e2 = WeierstrassP[w2, {g2, g3}], w2 = -w1 - w3")
_wfun("WeierstrassE3", _e_i(3), real_when_disc_pos=True, notes="e3 = WeierstrassP[w3, {g2, g3}]")


def _eta_i(i):
    def f(g2, g3):
        return _etas(*_half_periods(g2, g3))[i - 1]
    return f


_wfun("WeierstrassEta1", _eta_i(1), lambda k: _EXACT_ETA1.get(k),
      notes="eta1 = WeierstrassZeta[w1, {g2, g3}] = -Pi^2 theta1'''(0)/(12 w1 theta1'(0)) (DLMF 23.6.8)")
_wfun("WeierstrassEta2", _eta_i(2), notes="eta2 = WeierstrassZeta[w2, {g2, g3}] = -eta1 - eta3")
_wfun("WeierstrassEta3", _eta_i(3), notes="eta3 = WeierstrassZeta[w3, {g2, g3}], from the Legendre relation "
                                          "eta1 w3 - eta3 w1 = I Pi/2")


def _winv(name, idx):
    def mpf(w):
        if not isinstance(w, list) or len(w) != 2:
            raise ValueError(f"{name}: half-periods must be a list {{w1, w3}}")
        g2, g3 = _invariants(w[0], w[1])
        return [g2, g3] if idx is None else (g2, g3)[idx]

    def ex(w):
        if not isinstance(w, list) or len(w) != 2:
            raise ValueError(f"{name}: half-periods must be a list {{w1, w3}}")
        k = None
        for key, (a, b) in _EXACT_HP.items():
            if sp.simplify(w[0] - a) == 0 and sp.simplify(w[1] - b) == 0:
                k = key
        if w[1] == sp.I * w[0] and w[0] == 1:
            k = "i"
        if w[0] == 1 and _eq(w[1], sp.exp(sp.I * sp.pi / 3)):
            k = "rho"
        if k is None:
            return None
        g = {(0, 1): [sp.Integer(0), sp.Integer(1)], (1, 0): [sp.Integer(1), sp.Integer(0)],
             "i": [sp.gamma(sp.Rational(1, 4)) ** 8 / (256 * sp.pi ** 2), sp.Integer(0)],
             "rho": [sp.Integer(0), sp.gamma(sp.Rational(1, 3)) ** 18 / (4096 * sp.pi ** 6)]}[k]
        return Expr("List", *map(R, g)) if idx is None else g[idx]

    def real(w):
        # real lattices: w1 real and w3 imaginary, or w3 = w1/2 + imaginary
        return False
    mpfun(name, 1, mpf, ex, listable=False,
          notes=f"{'{g2, g3}' if idx is None else ('g2' if idx == 0 else 'g3')} for the lattice with half-periods "
                "{w1, w3}: g2 = 2 (e1^2 + e2^2 + e3^2), g3 = 4 e1 e2 e3 with e_i from theta constants")


_winv("WeierstrassInvariants", None)
_winv("WeierstrassInvariantG2", 0)
_winv("WeierstrassInvariantG3", 1)


def _wsz(name, which, exact0, notes):
    def mpf(u, g):
        g2, g3 = _gcheck(g, name)
        return _sigma_zeta(u, (g2, g3), which)

    def ex(u, g):
        return exact0 if u == 0 else None

    def real(u, g):
        return _allreal(u, g)
    mpfun(name, 2, mpf, ex, listable=False, real=real, notes=notes)


_wsz("WeierstrassSigma", "sigma", sp.Integer(0),
     "Weierstrass sigma(u; g2, g3) = (2 w1/Pi) Exp[eta1 u^2/(2 w1)] theta1(v)/theta1'(0), v = Pi u/(2 w1) "
     "(DLMF 23.6.9)")
_wsz("WeierstrassZeta", "zeta", sp.zoo,
     "Weierstrass zeta(u; g2, g3) = eta1 u/w1 + (Pi/(2 w1)) theta1'(v)/theta1(v) (logarithmic derivative of sigma)")


# EllipticExp / EllipticLog: the curve y^2 = x^3 + a x^2 + b x is x = P(u) - a/3, y = P'(u)/2 with
# g2 = 4 (a^2/3 - b), g3 = 4 a b/3 - 8 a^3/27 (so dx/du = 2 y and EllipticLog' = 1/(2 y)).

def _curve_g(a, b):
    return 4 * (a ** 2 / 3 - b), 4 * a * b / 3 - 8 * a ** 3 / 27


def _ellexp(u, ab, deriv=False):
    if not isinstance(ab, list) or len(ab) != 2:
        raise ValueError("EllipticExp: the curve parameters must be a list {a, b}")
    a, b = ab
    g2, g3 = _curve_g(a, b)
    if u == 0:
        raise ValueError("EllipticExp: u = 0 is a pole")
    if g2 == 0 and g3 == 0:
        p, dp, ddp = 1 / u ** 2, -2 / u ** 3, 6 / u ** 4
    else:
        p = _sigma_zeta(u, (g2, g3), "p")
        dp = _sigma_zeta(u, (g2, g3), "dp")
        ddp = 6 * p ** 2 - g2 / 2
    if deriv:
        return [dp, ddp / 2]
    return [p - a / 3, dp / 2]


def _ellexp_exact(u, ab, deriv=False):
    if not isinstance(ab, list) or len(ab) != 2:
        raise ValueError("EllipticExp: the curve parameters must be a list {a, b}")
    if ab[0] == 0 and ab[1] == 0:
        if u == 0:
            return Expr("List", Symbol("ComplexInfinity"), Symbol("ComplexInfinity"))
        return R([-2 / u ** 3, 3 / u ** 4]) if deriv else R([1 / u ** 2, -1 / u ** 3])
    if u == 0:
        return Expr("List", Symbol("ComplexInfinity"), Symbol("ComplexInfinity"))
    return None


mpfun("EllipticExp", 2, lambda u, ab: _ellexp(u, ab), _ellexp_exact,
      symbolic=lambda u, ab: _ellexp_exact(u, ab) if isinstance(ab, list) and ab == [0, 0] else None,
      listable=False, real=_allreal,
      notes="EllipticExp[u, {a, b}] = {x, y} on y^2 = x^3 + a x^2 + b x with u == EllipticLog[{x, y}, {a, b}]: "
            "x = WeierstrassP[u, {g2, g3}] - a/3, y = WeierstrassPPrime[u, {g2, g3}]/2, g2 = 4(a^2/3 - b), "
            "g3 = 4ab/3 - 8a^3/27"
            ". Check: reference (differential cases in tests/library).")
mpfun("EllipticExpPrime", 2, lambda u, ab: _ellexp(u, ab, True), lambda u, ab: _ellexp_exact(u, ab, True),
      symbolic=lambda u, ab: _ellexp_exact(u, ab, True) if isinstance(ab, list) and ab == [0, 0] else None,
      listable=False, real=_allreal, notes="derivative of EllipticExp[u, {a, b}] with respect to u: {2 y, 3 x^2 + 2 a x + b}"
                                           ". Check: reference (differential cases in tests/library).")


def _rf_branch(p, q, roots, name):
    """-sigma R_F(p - e1, p - e2, p - e3): the integral -Integrate[1/(2 Y), {t, p, Infinity}] with Y the branch
    of Sqrt[(t - e1)(t - e2)(t - e3)] equal to q at t = p (q = 0 picks the branch giving +R_F)."""
    s = mpmath.sqrt(p - roots[0]) * mpmath.sqrt(p - roots[1]) * mpmath.sqrt(p - roots[2])
    if s == 0:
        if q != 0:
            raise ValueError(f"{name}: the point is not on the curve")
        sigma = -1
    else:
        sigma = q / s
        if abs(abs(sigma) - 1) > mpmath.mpf(2) ** (-mpmath.mp.prec // 3):
            raise ValueError(f"{name}: the point is not on the curve")
        sigma = 1 if mpmath.re(sigma) > 0 else -1
    return -sigma * mpmath.elliprf(p - roots[0], p - roots[1], p - roots[2])


def _elllog(xy, ab):
    if not isinstance(xy, list) or len(xy) != 2 or not isinstance(ab, list) or len(ab) != 2:
        raise ValueError("EllipticLog: arguments must be {x, y} and {a, b}")
    (x, y), (a, b) = xy, ab
    r = mpmath.polyroots([1, a, b], extraprec=2 * mpmath.mp.prec) if b != 0 or a != 0 else [0, 0]
    u = _rf_branch(x, y, [mpmath.mpf(0)] + list(r), "EllipticLog")
    g2, g3 = _curve_g(a, b)
    if abs(g2 ** 3 - 27 * g3 ** 2) <= _tol() * (1 + abs(g2) ** 3 + abs(g3) ** 2):
        return u
    return _reduce_point(u, *_half_periods(g2, g3, "EllipticLog"))


mpfun("EllipticLog", 2, _elllog, listable=False,
      exact=lambda xy, ab: None,
      notes="EllipticLog[{x, y}, {a, b}] = Integrate[1/(2 Y(t)), {t, Infinity, x}] with Y^2 = t^3 + a t^2 + b t and "
            "Y(x) = y, evaluated as -sign R_F(x - e1, x - e2, x - e3) (Carlson; e_i the roots of the cubic)"
            ". Check: reference (differential cases in tests/library).")


def _inv_wp(pq, g, single):
    g2, g3 = _gcheck(g, "InverseWeierstrassP")
    if single:
        p = pq
        q = mpmath.sqrt(4 * p ** 3 - g2 * p - g3)
    else:
        if not isinstance(pq, list) or len(pq) != 2:
            raise ValueError("InverseWeierstrassP: the first argument must be p or {p, q}")
        p, q = pq
    roots = mpmath.polyroots([4, 0, -g2, -g3], extraprec=2 * mpmath.mp.prec)
    u = _rf_branch(p, q / 2, roots, "InverseWeierstrassP")
    w1, w3 = _half_periods(g2, g3, "InverseWeierstrassP")
    return _reduce_point(u, w1, w3)


def _reduce_point(u, w1, w3):
    """u modulo the lattice 2 w1 Z + 2 w3 Z, into u = 2 s w1 + 2 t w3 with -1/2 < s <= 1/2, -1/2 <= t < 1/2
    (the representative Wolfram returns for points on the boundary of the period parallelogram)."""
    M = mpmath.matrix([[mpmath.re(2 * w1), mpmath.re(2 * w3)], [mpmath.im(2 * w1), mpmath.im(2 * w3)]])
    st = mpmath.lu_solve(M, mpmath.matrix([mpmath.re(u), mpmath.im(u)]))
    tol = _tol()
    s0 = mpmath.ceil(st[0] - mpmath.mpf(1) / 2 - tol)
    t0 = mpmath.floor(st[1] + mpmath.mpf(1) / 2 + tol)
    return u - 2 * s0 * w1 - 2 * t0 * w3


mpfun("InverseWeierstrassP", 2, lambda pq, g: _inv_wp(pq, g, not isinstance(pq, list)), listable=False,
      exact=lambda pq, g: Symbol("ComplexInfinity") if (not isinstance(pq, list) and pq == sp.zoo) else None,
      notes="InverseWeierstrassP[p, {g2, g3}] / [{p, q}, {g2, g3}]: a u with WeierstrassP[u] == p (and "
            "WeierstrassPPrime[u] == q; q defaults to Sqrt[4 p^3 - g2 p - g3]) from Carlson R_F, reduced to the "
            "period parallelogram centred at 0; it can differ from Wolfram's representative by a period"
            ". Check: reference (differential cases in tests/library).")


# ───────────────────────────── modified Lommel functions ─────────────────────────────



def _lommel_t(mu, nu, z):
    b1, b2 = (mu - nu + 3) / 2, (mu + nu + 3) / 2
    d = (mu + 1) ** 2 - nu ** 2
    for b in (b1, b2):
        if _isint(b) and mpmath.re(b) <= 0:
            raise _Pole()
    if d == 0:
        raise _Pole()
    if z == 0:
        if mpmath.re(mu + 1) > 0:
            return mpmath.mpf(0)
        raise _Pole()
    return z ** (mu + 1) / d * mpmath.hyp1f2(1, b1, b2, z ** 2 / 4)


def _lommel_T2(mu, nu, z):
    g1, g2 = (mu + nu + 1) / 2, (mu - nu + 1) / 2
    if any(_isint(g) and mpmath.re(g) <= 0 for g in (g1, g2)):
        raise NotImplementedError("LommelT2: (mu +- nu + 1)/2 is a non-positive integer (limiting case not supported)")
    c = 2 ** (mu - 1) * mpmath.gamma(g1) * mpmath.gamma(g2)
    k = 2 / mpmath.pi * mpmath.cos((mu - nu) * mpmath.pi / 2) * mpmath.sin((mu + nu) * mpmath.pi / 2)
    return -_lommel_t(mu, nu, z) + c * (mpmath.besseli(nu, z) + k * mpmath.besselk(nu, z))


def _lommel_t_exact(mu, nu, z):
    if any(isinstance(v, list) for v in (mu, nu, z)):
        return None
    b1, b2 = (mu - nu + 3) / 2, (mu + nu + 3) / 2
    if ((mu + 1) ** 2 - nu ** 2) == 0 or any(b.is_integer and b <= 0 for b in (sp.sympify(b1), sp.sympify(b2))):
        return Symbol("ComplexInfinity")
    if z == 0 and sp.re(mu) > -1:
        return sp.Integer(0)
    return None


mpfun("LommelT1", 3, _lommel_t, _lommel_t_exact, real=lambda m, n, z: _allreal(m, n, z) and z > 0,
      notes="modified Lommel function t_{mu,nu}(z) = z^(mu+1)/((mu+1)^2 - nu^2) Hypergeometric1F2[1, (mu-nu+3)/2, "
            "(mu+nu+3)/2, z^2/4] (z^2 y'' + z y' - (z^2 + nu^2) y = z^(mu+1)); ComplexInfinity at the poles")
mpfun("LommelT2", 3, _lommel_T2, real=lambda m, n, z: _allreal(m, n, z) and z > 0,
      notes="modified Lommel function of the second kind: -t_{mu,nu}(z) + 2^(mu-1) Gamma[(mu+nu+1)/2] "
            "Gamma[(mu-nu+1)/2] (BesselI[nu, z] + 2/Pi Cos[(mu-nu) Pi/2] Sin[(mu+nu) Pi/2] BesselK[nu, z]) "
            "(the solution with algebraic growth; reduces to the documented (BesselI - StruveL) forms)"
            ". Check: reference (differential cases in tests/library).")


# ───────────────────────────── linear ODE solver (Taylor series continuation) ─────────────────────────────
#
# Solves P(z) y'' + Q(z) y' + R(z) y = 0 by power series: P, Q, R are given by coefficient functions of the
# expansion about the current centre.  The path is a straight segment; each step stays within half the distance
# to the nearest singular point.  The whole computation is repeated at a higher working precision and must agree.

class _ODE:
    def __init__(self, coeffs, singular):
        self.coeffs = coeffs          # coeffs(z0) -> (p, q, r): callables k -> k-th Taylor coefficient at z0
        self.singular = singular      # finite singular points

    def series(self, z0, y0, dy0, h, start_singular=False):
        p, q, r, (dp, dq, dr) = self.coeffs(z0)
        pk = [p(k) for k in range(dp + 1)] if dp is not None else None
        qk = [q(k) for k in range(dq + 1)] if dq is not None else None
        rk = [r(k) for k in range(dr + 1)] if dr is not None else None
        rcache = []

        def R_(k):
            if rk is not None:
                return rk[k] if k <= dr else 0
            while len(rcache) <= k:
                rcache.append(r(len(rcache)))
            return rcache[k]
        P_ = (lambda k: pk[k] if k <= dp else 0)
        Q_ = (lambda k: qk[k] if k <= dq else 0)
        c = [y0] if start_singular else [y0, dy0]
        eps = mpmath.mpf(2) ** (-mpmath.mp.prec - 8)
        total, dtotal = mpmath.mpf(0), mpmath.mpf(0)
        hp = mpmath.mpf(1)
        small = 0
        n = 0
        while True:
            while len(c) <= n:
                if start_singular:
                    N = len(c) - 1
                    acc = 0
                    for k in range(2, min(N + 1, dp) + 1):
                        acc += P_(k) * (N - k + 2) * (N - k + 1) * c[N - k + 2]
                    for k in range(1, min(N, dq) + 1):
                        acc += Q_(k) * (N - k + 1) * c[N - k + 1]
                    for k in range(0, (N if dr is None else min(N, dr)) + 1):
                        acc += R_(k) * c[N - k]
                    den = (N + 1) * (P_(1) * N + Q_(0))
                    if den == 0:
                        raise ValueError("logarithmic case: the regular power-series solution does not exist")
                    c.append(-acc / den)
                else:
                    N = len(c) - 2
                    acc = 0
                    for k in range(1, min(N, dp) + 1):
                        acc += P_(k) * (N - k + 2) * (N - k + 1) * c[N - k + 2]
                    for k in range(0, min(N + 1, dq) + 1):
                        acc += Q_(k) * (N - k + 1) * c[N - k + 1]
                    for k in range(0, (N if dr is None else min(N, dr)) + 1):
                        acc += R_(k) * c[N - k]
                    c.append(-acc / (P_(0) * (N + 2) * (N + 1)))
            term = c[n] * hp
            total += term
            if n >= 1:
                dtotal += n * c[n] * hp / h
            scale = abs(total) + abs(dtotal * h)
            if n > 4 and abs(term) * (n + 1) <= eps * scale:
                small += 1
                if small >= 3:
                    return total, dtotal
            else:
                small = 0
            n += 1
            hp *= h
            if n > 1500:
                raise ArithmeticError("series did not converge")

    def solve(self, z_start, y0, dy0, z_end, start_singular=False):
        z, y, dy = z_start, y0, dy0
        if z_end == z_start:
            return y, dy
        while True:
            dist = abs(z_end - z)
            others = [abs(s - z) for s in self.singular if abs(s - z) > 0]
            rad = min(others) if others else mpmath.inf
            step = min(dist, rad / 2, mpmath.mpf(1))
            if step <= 0:
                raise ValueError("the path runs into a singular point")
            h = (z_end - z) * (step / dist)
            for _ in range(60):
                try:
                    y1, dy1 = self.series(z, y, dy, h, start_singular)
                    break
                except ArithmeticError:
                    h /= 2
            else:
                raise ArithmeticError("ODE continuation failed")
            start_singular = False
            z, y, dy = z + h, y1, dy1
            if abs(z_end - z) <= abs(z_end) * mpmath.mpf(2) ** (-mpmath.mp.prec) + mpmath.mpf(2) ** (-mpmath.mp.prec):
                return y, dy


def _verified(f, name):
    """Evaluate f() at two working precisions (current + 20 and + 60 bits) and insist that they agree."""
    prec = mpmath.mp.prec
    with mpmath.workprec(prec + 20):
        a = f()
    for extra in (60, 160):
        with mpmath.workprec(prec + extra):
            b = f()
        tol = mpmath.mpf(2) ** (-(prec - 24)) * (1 + max(abs(x) for x in (a if isinstance(a, list) else [a])))
        diff = max(abs(x - y) for x, y in zip(a if isinstance(a, list) else [a], b if isinstance(b, list) else [b]))
        if diff <= tol:
            return b
        a = b
    raise ArithmeticError(f"{name}: numerical evaluation lost precision")


def _poly_coeffs(poly):
    """poly: list of coefficients (constant first) -> coeffs at z0 via Taylor shift."""
    def at(z0):
        n = len(poly)
        shifted = []
        for k in range(n):
            s = 0
            for j in range(k, n):
                s += mpmath.binomial(j, k) * poly[j] * z0 ** (j - k)
            shifted.append(s)
        return (lambda k: shifted[k] if k < n else 0), n - 1
    return at


def _poly_ode(P, Q, Rr, singular):
    pa, qa, ra = _poly_coeffs(P), _poly_coeffs(Q), _poly_coeffs(Rr)

    def co(z0):
        (p, dp), (q, dq), (r, dr) = pa(z0), qa(z0), ra(z0)
        return p, q, r, (dp, dq, dr)
    return _ODE(co, singular)


def _pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def _padd(*ps):
    n = max(len(p) for p in ps)
    return [sum(p[i] if i < len(p) else 0 for p in ps) for i in range(n)]


def _on_ray(z, s):
    """z lies on the ray {t s : t >= 1}."""
    if s == 0:
        return False
    t = z / s
    return abs(mpmath.im(t)) <= mpmath.mpf(2) ** (-mpmath.mp.prec // 2) * abs(t) and mpmath.re(t) >= 1


# ───────────────────────────── Heun functions ─────────────────────────────

def _heun_setup(kind, params):
    if kind == "G":
        a, q, al, be, ga, de = params
        if a == 0:
            raise ValueError("HeunG: the singular point a must be nonzero")
        ep = al + be + 1 - ga - de
        # P = z(z-1)(z-a), Q = ga (z-1)(z-a) + de z (z-a) + ep z (z-1), R = al be z - q
        P = _pmul(_pmul([0, 1], [-1, 1]), [-a, 1])
        Q = _padd([x * ga for x in _pmul([-1, 1], [-a, 1])], [x * de for x in _pmul([0, 1], [-a, 1])],
                  [x * ep for x in _pmul([0, 1], [-1, 1])])
        Rr = [-q, al * be]
        return _poly_ode(P, Q, Rr, [mpmath.mpf(1), a]), ga, (lambda z: _on_ray(z, 1) or _on_ray(z, a)), 0
    q, al, ga, de, ep = params
    if kind == "C":
        # z(z-1) y'' + (ga (z-1) + de z + ep z (z-1)) y' + (al z - q) y = 0
        P = [0, -1, 1]
        Q = _padd([-ga, ga], [0, de], [x * ep for x in [0, -1, 1]])
        return _poly_ode(P, Q, [-q, al], [mpmath.mpf(1)]), ga, (lambda z: _on_ray(z, 1)), 0
    if kind == "B":
        # z y'' + (ga + de z + ep z^2) y' + (al z - q) y = 0
        return _poly_ode([0, 1], [ga, de, ep], [-q, al], []), ga, (lambda z: False), 0
    if kind == "D":
        # z^2 y'' + (ga + de z + ep z^2) y' + (al z - q) y = 0, normalised y(1) = 1, y'(1) = 0
        return _poly_ode([0, 0, 1], [ga, de, ep], [-q, al], [mpmath.mpf(0)]), None, \
            (lambda z: mpmath.im(z) == 0 and mpmath.re(z) <= 0), 1
    # T: y'' + (ga + de z + ep z^2) y' + (al z - q) y = 0, y(0) = 1, y'(0) = 0
    return _poly_ode([1], [ga, de, ep], [-q, al], []), None, (lambda z: False), 0


def _heun(kind, deriv):
    name = "Heun" + kind + ("Prime" if deriv else "")

    def f(*args):
        params, z = args[:-1], args[-1]
        ode, ga, on_cut, z0 = _heun_setup(kind, params)
        if ga is not None and _isint(ga) and mpmath.re(ga) <= 0:
            raise ValueError(f"{name}: gamma is a non-positive integer (logarithmic case); the function is undefined")
        if on_cut(z):
            raise NotImplementedError(f"{name}: z lies on a branch cut or at a singular point")

        def run():
            p = [mpmath.mpf(1) * x if not isinstance(x, mpmath.mpc) else x for x in params]
            ode2, _, _, _ = _heun_setup(kind, p)
            zz = z
            if kind in ("G", "C", "B"):
                y, dy = ode2.solve(mpmath.mpf(0), mpmath.mpf(1), None, zz, start_singular=True)
            elif kind == "T":
                y, dy = ode2.solve(mpmath.mpf(0), mpmath.mpf(1), mpmath.mpf(0), zz)
            else:
                y, dy = ode2.solve(mpmath.mpf(1), mpmath.mpf(1), mpmath.mpf(0), zz)
            return dy if deriv else y
        if kind in ("G", "C", "B") and z == 0:
            if not deriv:
                return mpmath.mpf(1)
            P = params
            return {"G": lambda: P[1] / (P[0] * P[4]), "C": lambda: -P[0] / P[2], "B": lambda: P[0] / P[2]}[kind]()
        return _verified(run, name)

    nparams = 6 if kind == "G" else 5

    def ex(*args):
        z = args[-1]
        if kind in ("G", "C", "B", "T") and z == 0:
            if not deriv:
                return sp.Integer(1)
            P = args
            return {"G": lambda: P[1] / (P[0] * P[4]), "C": lambda: -P[0] / P[2], "B": lambda: P[0] / P[2],
                    "T": lambda: sp.Integer(0)}[kind]()
        if kind == "D" and z == 1:
            return sp.Integer(0) if deriv else sp.Integer(1)
        return None
    eq = {"G": "y'' + (gamma/z + delta/(z-1) + epsilon/(z-a)) y' + (alpha beta z - q)/(z(z-1)(z-a)) y = 0, "
               "epsilon = alpha + beta + 1 - gamma - delta",
          "C": "z(z-1) y'' + (gamma (z-1) + delta z + epsilon z (z-1)) y' + (alpha z - q) y = 0",
          "B": "z y'' + (gamma + delta z + epsilon z^2) y' + (alpha z - q) y = 0",
          "D": "z^2 y'' + (gamma + delta z + epsilon z^2) y' + (alpha z - q) y = 0",
          "T": "y'' + (gamma + delta z + epsilon z^2) y' + (alpha z - q) y = 0"}[kind]
    norm = {"G": "regular at 0 with y(0) = 1", "C": "regular at 0 with y(0) = 1", "B": "regular at 0 with y(0) = 1",
            "D": "y(1) = 1, y'(1) = 0", "T": "y(0) = 1, y'(0) = 0"}[kind]
    cuts = {"G": "branch cuts from 1 and from a radially to infinity (points on them raise)",
            "C": "branch cut [1, Infinity) (points on it raise)", "B": "entire in z", "T": "entire in z",
            "D": "z on (-Infinity, 0] raises"}[kind]
    mpfun(name, nparams + 1, f, ex, symbolic=lambda *a: ex(*a) if (a[-1] == 0 or (kind == "D" and a[-1] == 1)) else None,
          real=_allreal, extra=24,
          notes=f"{'derivative of the ' if deriv else ''}{name.replace('Prime', '')} solution of {eq}, {norm}; "
                f"power series continued along the straight path with verified precision; {cuts}; "
                "gamma a non-positive integer (logarithmic case) raises")


for _k in "GCBDT":
    _heun(_k, False)
    _heun(_k, True)


# ───────────────────────────── Mathieu functions ─────────────────────────────
#
# y'' + (a - 2 q Cos[2 z]) y = 0.  Characteristic values from the (truncated) Hill matrices; the characteristic
# exponent r from Cos[Pi r] = y_e(Pi) (y_e the even solution with y_e(0) = 1, computed by series continuation)
# with the branch fixed by the band structure (r in [n, n+1] in the n-th stability band, r = n + I mu in the
# n-th instability gap, as in Wolfram); MathieuC/MathieuS = (F(z) +- F(-z))/2(I) for the Floquet solution
# F(z) = Sum[c_k E^(I (r + 2k) z)] normalised by Sum[c_k^2] = 1 (dominant coefficient positive), which
# reproduces the Wolfram reference values (e.g. MathieuC[2, 1, 37/10] to 50 digits).

def _mathieu_ode(a, q):
    def coeffs(z0):
        c2, s2 = mpmath.cos(2 * z0), mpmath.sin(2 * z0)

        def r(k):
            if k % 2 == 0:
                t = c2 * (-1) ** (k // 2)
            else:
                t = -s2 * (-1) ** ((k - 1) // 2)
            v = -2 * q * mpmath.mpf(2) ** k / mpmath.factorial(k) * t
            return v + a if k == 0 else v
        return (lambda k: mpmath.mpf(1) if k == 0 else 0), (lambda k: 0), r, (0, 0, None)
    return _ODE(coeffs, [])


def _mathieu_ye_pi(a, q):
    y, _ = _mathieu_ode(a, q).solve(mpmath.mpf(0), mpmath.mpf(1), mpmath.mpf(0), +mpmath.pi)
    return y


def _hill_size(r, q):
    return int(abs(r) / 2) + 12 + int(3 * mpmath.sqrt(abs(q))) + mpmath.mp.dps // 3


def _tri_parts(M):
    n = M.rows
    return [M[i, i] for i in range(n)], [M[i, i + 1] for i in range(n - 1)]


def _sturm_count(d, e, x):
    """Number of eigenvalues < x of the symmetric tridiagonal matrix (d, e)."""
    cnt = 0
    tiny = mpmath.mpf(2) ** (-2 * mpmath.mp.prec)
    qv = d[0] - x
    for i in range(len(d)):
        if i:
            qv = d[i] - x - e[i - 1] ** 2 / qv
        if qv == 0:
            qv = tiny
        if qv < 0:
            cnt += 1
    return cnt


def _tri_kth(d, e, idx):
    """idx-th smallest eigenvalue (0-based) of a real symmetric tridiagonal matrix, by Sturm bisection."""
    n = len(d)
    rad = [(abs(e[i - 1]) if i else 0) + (abs(e[i]) if i < n - 1 else 0) for i in range(n)]
    lo = min(d[i] - rad[i] for i in range(n)) - 1
    hi = max(d[i] + rad[i] for i in range(n)) + 1
    eps = mpmath.mpf(2) ** (-mpmath.mp.prec + 2)
    while hi - lo > eps * (1 + abs(lo) + abs(hi)):
        mid = (lo + hi) / 2
        if _sturm_count(d, e, mid) > idx:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _tri_vec(d, e, lam):
    """Eigenvector for the eigenvalue lam (inverse iteration with a tridiagonal LU solve), unit 2-norm."""
    n = len(d)
    shift = lam + mpmath.mpf(2) ** (-mpmath.mp.prec + 4) * (1 + abs(lam))
    v = [mpmath.mpf(1)] * n
    tiny = mpmath.mpf(2) ** (-2 * mpmath.mp.prec)
    for _ in range(3):
        # Thomas algorithm for (T - shift) x = v
        cp, dp = [mpmath.mpf(0)] * n, [mpmath.mpf(0)] * n
        piv = d[0] - shift
        piv = piv if piv != 0 else tiny
        cp[0] = (e[0] / piv) if n > 1 else 0
        dp[0] = v[0] / piv
        for i in range(1, n):
            piv = d[i] - shift - e[i - 1] * cp[i - 1]
            piv = piv if piv != 0 else tiny
            cp[i] = e[i] / piv if i < n - 1 else 0
            dp[i] = (v[i] - e[i - 1] * dp[i - 1]) / piv
        x = [mpmath.mpf(0)] * n
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        nrm = mpmath.sqrt(mpmath.fsum(t * t for t in x))
        v = [t / nrm for t in x]
    return v


def _sym_eigs(M, real):
    if real:
        return sorted(mpmath.eigsy(M, eigvals_only=True))
    ev = mpmath.eig(M, left=False, right=False)
    return sorted(ev, key=lambda x: (mpmath.re(x), mpmath.im(x)))


def _mathieu_int_matrix(kind, par, q, N):
    """kind 'a' (even) / 'b' (odd), par 0 (period Pi) / 1 (period 2 Pi): truncated characteristic matrices
    (DLMF 28.4) for the coefficients of Cos/Sin[(2k + par) z]; symmetrised."""
    if kind == "a" and par == 0:
        M = mpmath.matrix(N, N)
        for k in range(N):
            M[k, k] = (2 * k) ** 2
            if k + 1 < N:
                M[k, k + 1] = M[k + 1, k] = q * (mpmath.sqrt(2) if k == 0 else 1)
        return M
    if kind == "b" and par == 0:
        M = mpmath.matrix(N, N)
        for k in range(N):
            M[k, k] = (2 * k + 2) ** 2
            if k + 1 < N:
                M[k, k + 1] = M[k + 1, k] = q
        return M
    M = mpmath.matrix(N, N)
    for k in range(N):
        M[k, k] = (2 * k + 1) ** 2
        if k + 1 < N:
            M[k, k + 1] = M[k + 1, k] = q
    M[0, 0] += q if kind == "a" else -q
    return M


def _floquet_matrix(r, q, N):
    M = mpmath.matrix(2 * N + 1, 2 * N + 1)
    for i, k in enumerate(range(-N, N + 1)):
        M[i, i] = (r + 2 * k) ** 2
        if i > 0:
            M[i, i - 1] = M[i - 1, i] = q
    return M


def _char_value(kind, r, q):
    """MathieuCharacteristicA (kind 'a') / B ('b') for real r and q."""
    real = not isinstance(q, mpmath.mpc) and not isinstance(r, mpmath.mpc)
    if not real:
        raise NotImplementedError("MathieuCharacteristicA/B: complex r or q are not supported")
    r = abs(r)
    N = _hill_size(r, q)
    if mpmath.isint(r):
        n = int(r)
        if kind == "b" and n == 0:
            raise ValueError("MathieuCharacteristicB: r must be a positive integer (there is no odd solution with r = 0)")
        par = n % 2
        idx = n // 2 if kind == "a" or par == 1 else n // 2 - 1
        return _tri_kth(*_tri_parts(_mathieu_int_matrix(kind, par, q, N + idx)), idx)
    return _tri_kth(*_tri_parts(_floquet_matrix(r, q, N + int(r))), int(mpmath.floor(r)))


def _char_values_below(a, q):
    """Number of Mathieu characteristic values (all a_n, b_n) below the real number a (real q)."""
    count = 0
    N = int(mpmath.sqrt(abs(a)) / 2) + 12 + int(3 * mpmath.sqrt(abs(q))) + mpmath.mp.dps // 3
    for kind, par in (("a", 0), ("a", 1), ("b", 0), ("b", 1)):
        d_, e_ = _tri_parts(_mathieu_int_matrix(kind, par, q, N))
        c = _sturm_count(d_, e_, a)
        if c > N - 4:
            raise ArithmeticError("Mathieu: characteristic-value matrix truncated too early")
        count += c
    return count


def _round_half_even(x):
    f = mpmath.floor(x)
    d = x - f
    if d > mpmath.mpf(1) / 2 or (d == mpmath.mpf(1) / 2 and int(f) % 2 == 1):
        return int(f) + 1
    return int(f)


def _mathieu_exponent(a, q):
    """Wolfram's branch: with n = Round[Re[Sqrt[a]]], r = n + ArcCos[(-1)^n y_e(Pi)]/Pi (principal ArcCos).
    This reproduces every documented value (e.g. r(Pi, E) = 2.30738, r(2, 1/2) = 1.36951, r(1/5647, 5) =
    1 + 1.31468 I, r(Pi/2, 2 + I) = 1.2109 - 0.665892 I)."""
    if q == 0:
        return mpmath.sqrt(a)
    y = _mathieu_ye_pi(a, q)
    n = _round_half_even(mpmath.re(mpmath.sqrt(a)))
    r = n + mpmath.acos((-1) ** n * y) / mpmath.pi
    if mpmath.im(a) == 0 and mpmath.im(q) == 0 and mpmath.im(r) < 0:
        r = mpmath.mpc(mpmath.re(r), -mpmath.im(r))   # Wolfram: 1 + 0.41039 I for (0.2, 1)
    return r


def _tri_solve(d, off, rhs):
    """Solve T x = rhs for the tridiagonal T with diagonal d and constant off-diagonal off (Thomas algorithm with
    partial pivoting replaced by a tiny-pivot guard; T is diagonally dominant away from the null direction)."""
    n = len(d)
    tiny = mpmath.mpf(2) ** (-2 * mpmath.mp.prec)
    cp, dp = [0] * n, [0] * n
    piv = d[0] if d[0] != 0 else tiny
    cp[0] = off / piv
    dp[0] = rhs[0] / piv
    for i in range(1, n):
        piv = d[i] - off * cp[i - 1]
        piv = piv if piv != 0 else tiny
        cp[i] = off / piv
        dp[i] = (rhs[i] - off * dp[i - 1]) / piv
    x = [0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _mathieu_coeffs(a, q, r):
    """Floquet coefficients c_k (k = -N..N) of the solution with exponent r, Sum c_k^2 = 1."""
    N = _hill_size(r, q) + int(mpmath.sqrt(abs(a)) / 2)
    # inverse iteration for the null vector of the tridiagonal matrix diag((r + 2k)^2 - a), off-diagonal q
    shift = mpmath.mpf(2) ** (-mpmath.mp.prec // 2) * (1 + abs(a) + abs(q))
    dd = [(r + 2 * k) ** 2 - a + shift for k in range(-N, N + 1)]
    v = [mpmath.mpf(1)] * (2 * N + 1)
    for _ in range(4):
        v = _tri_solve(dd, q, v)
        nrm = mpmath.sqrt(mpmath.fsum(abs(t) ** 2 for t in v))
        v = [t / nrm for t in v]
    c = v
    s2 = mpmath.fsum(x * x for x in c)
    if s2 == 0:
        raise ArithmeticError("Mathieu: degenerate Floquet coefficients")
    c = [x / mpmath.sqrt(s2) for x in c]
    big = max(range(len(c)), key=lambda i: abs(c[i]))
    if mpmath.re(c[big]) < 0:
        c = [-x for x in c]
    if abs(c[0]) + abs(c[-1]) > mpmath.mpf(2) ** (-mpmath.mp.prec + 10) * 0 + mpmath.mpf(2) ** (-mpmath.mp.prec // 2):
        raise ArithmeticError("Mathieu: Fourier series truncated too early")
    return c, N


def _mathieu_fun(kind, deriv):
    name = "Mathieu" + kind + ("Prime" if deriv else "")

    def f(a, q, z):
        if mpmath.im(a) != 0 or mpmath.im(q) != 0:
            raise NotImplementedError(f"{name}: complex a or q is not supported (Wolfram's normalisation for "
                                      "complex parameters could not be reproduced)")
        if q == 0:
            s = mpmath.sqrt(a)
            if kind == "C":
                return -s * mpmath.sin(s * z) if deriv else mpmath.cos(s * z)
            return s * mpmath.cos(s * z) if deriv else (mpmath.sin(s * z) if s != 0 else z)

        def run():
            r = _mathieu_exponent(a, q)
            if abs(r - mpmath.nint(mpmath.re(r))) < mpmath.mpf(2) ** (-mpmath.mp.prec // 3):
                r = mpmath.nint(mpmath.re(r))
            c, N = _mathieu_coeffs(a, q, r)
            ks = range(-N, N + 1)
            if mpmath.isint(r) if not isinstance(r, mpmath.mpc) else False:
                ev = mpmath.fsum(ck * mpmath.cos((r + 2 * k) * z) for ck, k in zip(c, ks))
                od = mpmath.fsum(ck * mpmath.sin((r + 2 * k) * z) for ck, k in zip(c, ks))
                # at a characteristic value only one parity is a Floquet solution
                tot = mpmath.fsum(abs(ck) for ck in c)
                sym_even = mpmath.fsum(abs(c[i] - c[2 * N - i - int(r)]) for i in range(len(c)) if 0 <= 2 * N - i - int(r) < len(c))
                sym_odd = mpmath.fsum(abs(c[i] + c[2 * N - i - int(r)]) for i in range(len(c)) if 0 <= 2 * N - i - int(r) < len(c))
                small = mpmath.mpf(2) ** (-mpmath.mp.prec // 2) * tot
                if (kind == "S" and sym_even < small) or (kind == "C" and sym_odd < small):
                    raise NotImplementedError(f"{name}: a is a characteristic value of the other parity "
                                              "(non-periodic case not supported)")
            if kind == "C":
                if deriv:
                    return -mpmath.fsum(ck * (r + 2 * k) * mpmath.sin((r + 2 * k) * z) for ck, k in zip(c, ks))
                return mpmath.fsum(ck * mpmath.cos((r + 2 * k) * z) for ck, k in zip(c, ks))
            if deriv:
                return mpmath.fsum(ck * (r + 2 * k) * mpmath.cos((r + 2 * k) * z) for ck, k in zip(c, ks))
            return mpmath.fsum(ck * mpmath.sin((r + 2 * k) * z) for ck, k in zip(c, ks))
        return _verified(run, name)

    def ex(a, q, z):
        if q == 0:
            s = sp.sqrt(a)
            if kind == "C":
                return -s * sp.sin(s * z) if deriv else sp.cos(s * z)
            return s * sp.cos(s * z) if deriv else sp.sin(s * z)
        if z == 0:
            if kind == "S" and not deriv:
                return sp.Integer(0)
            if kind == "C" and deriv:
                return sp.Integer(0)
        return None

    def real(a, q, z):
        return False
    mpfun(name, 3, f, ex, symbolic=lambda a, q, z: ex(a, q, z) if q == 0 else None,
          notes=f"{'derivative of ' if deriv else ''}{'even' if kind == 'C' else 'odd'} Mathieu function with "
                "characteristic value a (Wolfram normalisation: Floquet solution F(z) = Sum[c_k E^(I(r+2k)z)], "
                "Sum[c_k^2] = 1, r = MathieuCharacteristicExponent[a, q]); real a and q (complex z allowed); "
                "complex a or q raise")


for _k in "CS":
    _mathieu_fun(_k, False)
    _mathieu_fun(_k, True)


def _mca(kind):
    name = "MathieuCharacteristic" + kind.upper()

    def f(r, q):
        if q == 0:
            return r ** 2
        return _verified(lambda: _char_value(kind, r, q), name)

    def ex(r, q):
        if q == 0:
            if kind == "b" and r == 0:
                return None
            return r ** 2
        return None
    mpfun(name, 2, f, ex, symbolic=lambda r, q: r ** 2 if q == 0 else None, real=_allreal,
          notes=f"characteristic value {'a' if kind == 'a' else 'b'}_r(q) of the {'even' if kind == 'a' else 'odd'} "
                "Mathieu functions: integer r from the DLMF 28.4 tridiagonal matrices, non-integer r from the "
                "Floquet (Hill) matrix (the floor(|r|)-th eigenvalue); real r and q (complex raise)")


_mca("a")
_mca("b")
mpfun("MathieuCharacteristicExponent", 2, lambda a, q: _verified(lambda: _mathieu_exponent(a, q),
                                                                 "MathieuCharacteristicExponent"),
      lambda a, q: sp.sqrt(a) if q == 0 else None, symbolic=lambda a, q: sp.sqrt(a) if q == 0 else None,
      notes="characteristic exponent r with Cos[Pi r] = y_e(Pi) (even solution, y_e(0) = 1), r in [n, n+1] in the "
            "n-th stability band and r = n + I mu in the n-th instability gap (Wolfram's branch); real a, q "
            "(complex raise)"
            ". Check: reference (differential cases in tests/library).")


# ---------------------------------------------------------------------------------------------------------------
# Lame functions: y'' + (h - nu(nu+1) m sn(z|m)^2) y = 0, 4K-periodic solutions.  In w = Pi (K - z)/(2K) the
# equation is a Hill equation with m sn^2 = 1 - E/K + Sum d_n Cos[2 n w]; LameC = Sum A_k Cos[k w] (even about
# z = K), LameS = Sum B_k Sin[k w]; normalisation (1/Pi) Integrate[y^2, {w, 0, 2 Pi}] = 1, signs fixed by
# LameC[nu, j, K, m] > 0 and LameSPrime[nu, j, K, m] < 0 (the m = 0 limits Cos[j(Pi/2 - z)], Sin[j(Pi/2 - z)]).
# ---------------------------------------------------------------------------------------------------------------

def _lame_j(j, name, allow_zero):
    if isinstance(j, mpmath.mpc) and j.imag != 0 or not mpmath.isint(mpmath.re(j)) or mpmath.re(j) < (0 if allow_zero else 1):
        raise ValueError(f"{name}: j must be a {'non-negative' if allow_zero else 'positive'} integer")
    return int(mpmath.re(j))


def _lame_setup(nu, m):
    if mpmath.im(nu) != 0 or mpmath.im(m) != 0:
        raise NotImplementedError("Lame functions: complex order or parameter is not supported")
    nu, m = mpmath.re(nu), mpmath.re(m)
    if m >= 1:
        raise NotImplementedError("Lame functions: only parameters m < 1 are supported")
    K = mpmath.ellipk(m)
    E = mpmath.ellipe(m)
    q = mpmath.qfrom(m=m)
    if isinstance(q, mpmath.mpc):
        q = q.real          # m < 0: real negative nome
    L = nu * (nu + 1)
    return nu, m, K, E, q, L


def _lame_system(nu, j, m, kind):
    """Return (eigenvalue, list of (k, coefficient)) for LameC (kind 'C') or LameS (kind 'S')."""
    nu, m, K, E, q, L = _lame_setup(nu, m)
    p = j % 2
    if kind == "C":
        ks0 = [p]
    else:
        ks0 = [p if p else 2]
    idx = (j - ks0[0]) // 2
    tol = mpmath.mpf(2) ** (-mpmath.mp.prec - 10)
    # Fourier coefficients of m sn^2 in w
    d = []
    n = 1
    while True:
        dn = -(2 * mpmath.pi ** 2 / K ** 2) * (-1) ** n * n * q ** n / (1 - q ** (2 * n))
        if abs(dn) * (1 + abs(L)) < tol and n > 2:
            break
        d.append(dn)
        n += 1
        if n > 20000:
            raise ArithmeticError("Lame: nome too close to 1")
    c0 = 1 - E / K
    s = (mpmath.pi / (2 * K)) ** 2
    nb = idx + 12 + int(mpmath.mp.prec / max(1, -float(mpmath.log(max(abs(q), mpmath.mpf(2) ** -60), 2)))) + \
        int(mpmath.sqrt(abs(L)) / 2 + mpmath.sqrt(abs(L)) * mpmath.sqrt(abs(m)))
    ks = [ks0[0] + 2 * i for i in range(nb)]

    def tmat(coef):     # matrix of multiplication by Sum coef(t) Cos[2 t w] in the orthonormal basis
        T = mpmath.matrix(nb, nb)
        for a_, k in enumerate(ks):
            for b_, kp in enumerate(ks):
                v = 0
                t1 = abs(k - kp) // 2
                if t1 >= 1:
                    v += coef(t1) / 2
                t2 = (k + kp) // 2
                if t2 >= 1:
                    v += coef(t2) / 2 if kind == "C" else -coef(t2) / 2
                if kind == "C" and (k == 0) != (kp == 0):
                    v = v / mpmath.sqrt(2)
                T[a_, b_] = v
        return T

    T = tmat(lambda t: d[t - 1] if t <= len(d) else 0)
    M = L * T
    for a_, k in enumerate(ks):
        M[a_, a_] += s * k * k + L * c0
    ev, U = mpmath.eigsy(M)
    order = sorted(range(nb), key=lambda i: ev[i])
    i = order[idx]
    lam = ev[i]
    vec = [U[r_, i] for r_ in range(nb)]
    if abs(vec[-1]) + abs(vec[-2]) > mpmath.mpf(2) ** (-mpmath.mp.prec // 2) * 0 + mpmath.mpf(2) ** (-(mpmath.mp.prec * 3) // 4):
        raise ArithmeticError("Lame: Fourier series truncated too early")
    coeffs = []
    for k, v in zip(ks, vec):
        if kind == "C" and k == 0:
            coeffs.append((k, v / mpmath.sqrt(2)))
        else:
            coeffs.append((k, v))
    # Wolfram normalisation: (1/Pi) Integrate[y^2, {phi, 0, 2 Pi}] = 1 with phi = JacobiAmplitude[z, m];
    # d phi = dn dz and dn = Pi/(2K) (1 + 4 Sum (-1)^n q^n/(1 + q^(2n)) Cos[2 n w])
    G = tmat(lambda t: 4 * (-1) ** t * q ** t / (1 + q ** (2 * t)))
    for a_ in range(nb):
        G[a_, a_] += 1
    vv = mpmath.matrix(vec)
    nrm2 = (vv.T * G * vv)[0]
    coeffs = [(k, c / mpmath.sqrt(nrm2)) for k, c in coeffs]
    if kind == "C":
        sgn = mpmath.fsum(c for k, c in coeffs)
    else:
        sgn = mpmath.fsum(k * c for k, c in coeffs)
    if sgn < 0:
        coeffs = [(k, -c) for k, c in coeffs]
    return lam, coeffs, K


def _lame_eig(kind):
    name = "LameEigenvalue" + ("A" if kind == "C" else "B")

    def f(nu, j, m):
        jj = _lame_j(j, name, kind == "C")
        if m == 0:
            return mpmath.mpf(jj) ** 2
        return _verified(lambda: _lame_system(nu, jj, m, kind)[0], name)

    def ex(nu, j, m):
        if m == 0 and not isinstance(j, list) and j.is_integer and (j >= 0 if kind == "C" else j >= 1):
            return j ** 2
        return None
    mpfun(name, 3, f, ex, symbolic=lambda nu, j, m: ex(nu, j, m),
          notes=f"Lame eigenvalue h for Lame{kind} (j-th 4K-periodic eigenvalue of y'' + (h - nu(nu+1) m sn^2) y = 0 "
                f"of {'even' if kind == 'C' else 'odd'} symmetry about z = K); real nu and m < 1 (others raise); "
                f"{name}[nu, j, 0] = j^2")


def _lame_fun(kind, deriv):
    name = "Lame" + kind + ("Prime" if deriv else "")

    def f(nu, j, z, m):
        jj = _lame_j(j, name, kind == "C")

        def run():
            lam, coeffs, K = _lame_system(nu, jj, m, kind)
            w = mpmath.pi * (K - z) / (2 * K)
            qq = mpmath.qfrom(m=mpmath.re(m))
            if abs(mpmath.im(w)) > -mpmath.log(abs(qq)) / 4:
                raise NotImplementedError(f"{name}: |Im z| too large (outside the Fourier convergence strip)")
            dw = -mpmath.pi / (2 * K)
            if kind == "C":
                if deriv:
                    return dw * mpmath.fsum(-c * k * mpmath.sin(k * w) for k, c in coeffs)
                return mpmath.fsum(c * mpmath.cos(k * w) for k, c in coeffs)
            if deriv:
                return dw * mpmath.fsum(c * k * mpmath.cos(k * w) for k, c in coeffs)
            return mpmath.fsum(c * mpmath.sin(k * w) for k, c in coeffs)
        if m == 0:
            w = mpmath.pi / 2 - z
            if kind == "C":
                if jj == 0:
                    return mpmath.mpf(0) if deriv else 1 / mpmath.sqrt(2)
                return jj * mpmath.sin(jj * w) if deriv else mpmath.cos(jj * w)
            return -jj * mpmath.cos(jj * w) if deriv else mpmath.sin(jj * w)
        return _verified(run, name)

    def ex(nu, j, z, m):
        if isinstance(j, list) or isinstance(m, list) or isinstance(z, list) or m != 0 or not j.is_integer:
            return None
        w = sp.pi / 2 - z
        if kind == "C":
            if j == 0:
                return sp.Integer(0) if deriv else 1 / sp.sqrt(2)
            return j * sp.sin(j * w) if deriv else sp.cos(j * w)
        if j < 1:
            return None
        return -j * sp.cos(j * w) if deriv else sp.sin(j * w)
    mpfun(name, 4, f, ex, symbolic=ex,
          notes=f"{'z-derivative of the ' if deriv else ''}Lame function {'LameC (even about z = K)' if kind == 'C' else 'LameS (odd about z = K)'}; "
                "Fourier (Hill) expansion with (1/Pi) Integrate[y^2, {w, 0, 2 Pi}] = 1, w = Pi (K - z)/(2K); real nu, "
                "real m < 1 and z within half the Fourier strip (others raise)")


for _k in "CS":
    _lame_eig(_k)
    _lame_fun(_k, False)
    _lame_fun(_k, True)


# ---------------------------------------------------------------------------------------------------------------
# Spheroidal wave functions (DLMF chapter 30 conventions, which Wolfram follows):
# (1 - z^2) y'' - 2 z y' + (lambda + gamma^2 (1 - z^2) - m^2/(1 - z^2)) y = 0.
# Ps_n^m(x, gamma^2) = Sum_k (-1)^k a_{n,k}^m P_{n+2k}^m(x) (Ferrers/LegendreP type 1), Meixner-Schaefke norm
# Integrate[Ps^2, {x, -1, 1}] = 2 (n+m)!/((2n+1)(n-m)!), sign continuous from gamma = 0 (Ps -> LegendreP).
# Integer n >= |m| and real gamma^2 (gamma real or purely imaginary) only.
# ---------------------------------------------------------------------------------------------------------------

def _sph_nm(n, m, name):
    for v, w in ((n, "n"), (m, "m")):
        if isinstance(v, mpmath.mpc) and v.imag != 0 or not mpmath.isint(mpmath.re(v)):
            raise NotImplementedError(f"{name}: only integer n and m are supported")
    n, m = int(mpmath.re(n)), int(mpmath.re(m))
    if n < abs(m):
        raise NotImplementedError(f"{name}: only n >= |m| is supported")
    return n, m


def _sph_g2(g, name):
    g2 = g * g
    if isinstance(g2, mpmath.mpc):
        if abs(g2.imag) > abs(g2) * mpmath.mpf(2) ** (-mpmath.mp.prec + 8):
            raise NotImplementedError(f"{name}: gamma must be real or purely imaginary")
        g2 = g2.real
    return g2


def _sph_system(n, m, g2, nbig=0):
    """Return (lambda, {l: b_l}) with Ps_n^m = Sum_l b_l P_l^m (m >= 0), Meixner-Schaefke normalised.
    The eigenvalue comes from a basis of nb Legendre functions; the coefficient vector is computed on a basis of
    max(nb, nbig) functions (needed by the slowly convergent Neumann series of SpheroidalS2)."""
    m = abs(m)
    p = (n - m) % 2
    idx = (n - m) // 2
    nb0 = idx + 20 + int(mpmath.sqrt(abs(g2))) + mpmath.mp.prec // 6
    nb = max(nb0, nbig)
    ls = [m + p + 2 * i for i in range(nb)]

    def N(l):
        return mpmath.mpf(2) / (2 * l + 1) * mpmath.factorial(l + m) / mpmath.factorial(l - m)

    def beta(l):
        return mpmath.mpf((l - m + 1) * (l + m + 1)) / ((2 * l + 1) * (2 * l + 3)) + \
            mpmath.mpf((l + m) * (l - m)) / ((2 * l + 1) * (2 * l - 1))

    def alpha(l):   # x^2 P_l -> alpha(l) P_{l+2}
        return mpmath.mpf((l - m + 1) * (l - m + 2)) / ((2 * l + 1) * (2 * l + 3))
    M = mpmath.matrix(nb, nb)
    for i, l in enumerate(ls):
        M[i, i] = l * (l + 1) - g2 + g2 * beta(l)
        if i + 1 < nb:
            # symmetric element in the orthonormal basis P_l/sqrt(N_l):  <l+2| x^2 |l> = alpha(l) sqrt(N_{l+2}/N_l)
            e = g2 * alpha(l) * mpmath.sqrt(N(l + 2) / N(l))
            M[i, i + 1] = M[i + 1, i] = e
    d_, e_ = _tri_parts(M)
    lam = _tri_kth(d_[:nb0], e_[:nb0 - 1], idx)
    vec = _tri_vec(d_, e_, lam)
    # the tail of the eigenvector decays super-exponentially; recompute it from the anchor (largest component)
    # by the backward continued fraction for the minimal solution, so each coefficient has full relative accuracy
    anchor = max(range(nb), key=lambda i: abs(vec[i]))
    if anchor < nb - 2:
        rho = [mpmath.mpf(0)] * nb
        rho[nb - 2] = -e_[nb - 2] / (d_[nb - 1] - lam)
        for i in range(nb - 3, anchor - 1, -1):
            den = (d_[i + 1] - lam) + e_[i + 1] * rho[i + 1]
            rho[i] = -e_[i] / den if den != 0 else mpmath.mpf(0)
        for i in range(anchor, nb - 1):
            vec[i + 1] = rho[i] * vec[i]
    if anchor > 0:
        # and below the anchor (they decay towards l = m, like gamma^(n - l)) by the forward continued fraction
        sig = [mpmath.mpf(0)] * (anchor + 1)      # sig[i] = v[i-1] / v[i]
        sig[1] = -e_[0] / (d_[0] - lam)
        for i in range(1, anchor):
            den = (d_[i] - lam) + e_[i - 1] * sig[i]
            sig[i + 1] = -e_[i] / den if den != 0 else mpmath.mpf(0)
        for i in range(anchor, 0, -1):
            vec[i - 1] = sig[i] * vec[i]
    if abs(vec[-1]) > mpmath.mpf(2) ** (-mpmath.mp.prec):
        raise ArithmeticError("Spheroidal: Legendre expansion truncated too early")
    scale = mpmath.sqrt(N(n))
    b = {l: scale * v / mpmath.sqrt(N(l)) for l, v in zip(ls, vec)}
    # sign: continuous from gamma = 0 -> compare with LegendreP_n^m at 0 (n - m even) or its derivative (odd)
    if p == 0:
        P0 = _legP_table(max(b) + 1, m, mpmath.mpf(0))
        ref = mpmath.fsum(bl * P0[l] for l, bl in b.items())
        ref0 = P0[n]
    else:
        # d/dx P_l^m(0) = -(l - m + 1) P_{l+1}^m(0)  (DLMF 14.10.5 at x = 0)
        P0 = _legP_table(max(b) + 2, m, mpmath.mpf(0))
        ref = mpmath.fsum(-bl * (l - m + 1) * P0[l + 1] for l, bl in b.items())
        ref0 = -(n - m + 1) * P0[n + 1]
    if ref * ref0 < 0:
        b = {l: -v for l, v in b.items()}
    return lam, b


def _legP_table(L, m, x):
    """{l: P_l^m(x)} for l = m..L (Ferrers / LegendreP type 2, Condon-Shortley phase, analytically continued off
    [-1, 1] as mpmath legenp type 2), by the stable upward recurrence in l."""
    out = {m: (-1) ** m * mpmath.fac2(2 * m - 1) * mpmath.power(1 - x * x, mpmath.mpf(m) / 2)}
    if L > m:
        out[m + 1] = x * (2 * m + 1) * out[m]
    for l in range(m + 1, L):
        out[l + 1] = ((2 * l + 1) * x * out[l] - (l + m) * out[l - 1]) / (l - m + 1)
    return out


def _legP(l, m, x, deriv=False, table=None):
    """P_l^m(x) or its x-derivative, from a _legP_table (built here if not given)."""
    t = table if table is not None else _legP_table(l + 1, m, x)
    if not deriv:
        return t[l]
    # (1 - x^2) P' = (l + 1) x P_l - (l - m + 1) P_{l+1}   ... (DLMF 14.10.5 with mu = m)
    return ((l + 1) * x * t[l] - (l - m + 1) * t[l + 1]) / (1 - x * x)


def _sph_eig(n, m, g):
    n, m = _sph_nm(n, m, "SpheroidalEigenvalue")
    g2 = _sph_g2(g, "SpheroidalEigenvalue")
    if g2 == 0:
        return mpmath.mpf(n * (n + 1))
    return _verified(lambda: _sph_system(n, m, g2)[0], "SpheroidalEigenvalue")


mpfun("SpheroidalEigenvalue", 3, _sph_eig,
      lambda n, m, g: n * (n + 1) if (not isinstance(g, list) and g == 0 and not isinstance(n, list) and n.is_integer
                                      and not isinstance(m, list) and m.is_integer and n >= abs(m)) else None,
      notes="spheroidal eigenvalue lambda_n^m(gamma) of (1-z^2) y'' - 2 z y' + (lambda + gamma^2 (1-z^2) - "
            "m^2/(1-z^2)) y = 0 (Legendre-basis matrix); integer n >= |m|, real or imaginary gamma (others raise)"
            ". Check: reference (differential cases in tests/library).")


def _sph_ps(deriv):
    name = "SpheroidalPS" + ("Prime" if deriv else "")

    def f(n, m, g, x):
        n, m = _sph_nm(n, m, name)
        g2 = _sph_g2(g, name)
        if x in (1, -1) and m != 0 and not deriv:
            return mpmath.mpf(0) if m > 0 else mpmath.inf
        if deriv and x in (1, -1):
            raise ValueError(f"{name}: x = +-1 is a singular point")

        def run():
            if g2 == 0:
                b = {n: mpmath.mpf(1)}
            else:
                b = _sph_system(n, m, g2)[1]
            am = abs(m)
            tab = _legP_table(max(b) + 1, am, x)
            out = mpmath.fsum(bl * _legP(l, am, x, deriv, tab) for l, bl in b.items())
            if m < 0:
                out *= (-1) ** am * mpmath.factorial(n - am) / mpmath.factorial(n + am)
            return out
        return _verified(run, name)
    mpfun(name, 4, f,
          notes=f"{'x-derivative of the ' if deriv else ''}angular prolate spheroidal function Ps_n^m(gamma, x) "
                "(Meixner-Schaefke normalisation; SpheroidalPS[n, m, 0, x] = LegendreP[n, m, x]); Legendre series, "
                "integer n >= |m|, real or imaginary gamma (others raise)")


_sph_ps(False)
_sph_ps(True)


def _sph_radial_coeffs(n, m, g2, nbig=0):
    """a_{n,k}^{-m} up to a common factor, keyed by l = n + 2k, and A_n^{-m}."""
    m = abs(m)
    b = _sph_system(n, m, g2, nbig)[1] if g2 != 0 else {n: mpmath.mpf(1)}
    # Ps^m = Sum (-1)^k a^m_k P^m_l, b_l = (-1)^k a^m_k;  a^{-m}_k  ~  a^m_k (l+m)!/(l-m)!
    a = {}
    for l, bl in b.items():
        k = (l - n) // 2
        a[l] = (-1) ** k * bl * mpmath.factorial(l + m) / mpmath.factorial(l - m)
    A = mpmath.fsum((-1) ** ((l - n) // 2) * al for l, al in a.items())
    return a, A


def _sph_bessel(j, l, x, deriv):
    def psi(k):
        if j == 1:
            return mpmath.sqrt(mpmath.pi / (2 * x)) * mpmath.besselj(k + mpmath.mpf(1) / 2, x)
        return mpmath.sqrt(mpmath.pi / (2 * x)) * mpmath.bessely(k + mpmath.mpf(1) / 2, x)
    if not deriv:
        return psi(l)
    return l / x * psi(l) - psi(l + 1)


def _sph_radial(j, deriv):
    name = f"SpheroidalS{j}" + ("Prime" if deriv else "")

    def f(n, m, g, z):
        n, m = _sph_nm(n, m, name)
        g2 = _sph_g2(g, name)
        if g == 0:
            raise ValueError(f"{name}: gamma = 0 is not allowed (the Meixner-Schaefke radial functions vanish or blow up)")
        if z == 0:
            raise ValueError(f"{name}: z = 0 is a singular point of the series")
        if j == 2 and abs(z) <= 1:
            raise NotImplementedError(f"{name}: only |z| > 1 is supported (the Bessel series diverges for |z| <= 1)")

        nbig = 0
        if j == 2:
            # the Neumann series converges like |z|^(-2k): enough terms for the working precision
            nbig = (n - abs(m)) // 2 + 30 + int((mpmath.mp.prec + 200) * mpmath.log(2) / (2 * mpmath.log(abs(z))))
            if nbig > 4000:
                raise NotImplementedError(f"{name}: z is too close to the unit circle for the Neumann series")

        def run():
            a, A = _sph_radial_coeffs(n, m, g2, nbig)
            am = abs(m)
            fac = (1 - 1 / (z * z)) ** (mpmath.mpf(am) / 2)
            ser = mpmath.fsum(al * _sph_bessel(j, l, g * z, False) for l, al in a.items())
            if not deriv:
                return fac * ser / A
            dser = g * mpmath.fsum(al * _sph_bessel(j, l, g * z, True) for l, al in a.items())
            dfac = am / (z ** 3) * (1 - 1 / (z * z)) ** (mpmath.mpf(am) / 2 - 1) if am else 0
            return (dfac * ser + fac * dser) / A
        return _verified(run, name)
    mpfun(name, 4, f,
          notes=f"{'z-derivative of the ' if deriv else ''}radial spheroidal function S^({j})_n^m(gamma, z), DLMF 30.11.3 "
                "(Meixner-Schaefke): (1 - z^-2)^(m/2)/A Sum a^(-m)_{n,k} psi^({j})_(n+2k)(gamma z); integer n >= |m|, "
                f"real or imaginary gamma{', |z| > 1' if j == 2 else ''} (others raise)")


for _j in (1, 2):
    _sph_radial(_j, False)
    _sph_radial(_j, True)


# ---------------------------------------------------------------------------------------------------------------
# Spheroidal joining factor: SpheroidalS1[n, m, c, z] = SpheroidalJoiningFactor[n, m, c] SpheroidalPS[n, m, c, z]
# (documented for m = 0); for m != 0 the factor is real, the phase (-I)^|m| between the Ferrers function used by Ps
# and the real Legendre function on z > 1 being removed (this reproduces the documented values for m = 1, -1, 5).
# (SpheroidalRadialFactor is not implemented: its normalisation does not match DLMF 30.11.4 and could not be
# reproduced from the documented values.)
# ---------------------------------------------------------------------------------------------------------------

def _sph_joining(n, m, g):
    n, m = _sph_nm(n, m, "SpheroidalJoiningFactor")
    g2 = _sph_g2(g, "SpheroidalJoiningFactor")
    if g2 == 0:
        if n == 0:
            return mpmath.mpf(1)
        raise ValueError("SpheroidalJoiningFactor: c = 0 with n > 0 is a degenerate limit")

    am = abs(m)

    def run():
        a, A = _sph_radial_coeffs(n, am, g2)
        b = _sph_system(n, am, g2)[1]
        out = []
        for z in (mpmath.mpf(2), mpmath.mpf(3)):
            # S1 and Ps for real z > 1 with the Ferrers factor (1 - z^2)^(m/2) = I^m (z^2 - 1)^(m/2) removed
            s1 = (1 - 1 / (z * z)) ** (mpmath.mpf(am) / 2) * \
                mpmath.fsum(al * _sph_bessel(1, l, g * z, False) for l, al in a.items()) / A
            tab = _legP_table(max(b), am, z)
            ps = mpmath.fsum(bl * tab[l] for l, bl in b.items()) / mpmath.mpc(0, -1) ** am
            if m < 0:
                ps *= (-1) ** am * mpmath.factorial(n - am) / mpmath.factorial(n + am)
            out.append(s1 / ps)
        if abs(out[0] - out[1]) > mpmath.mpf(2) ** (-mpmath.mp.prec // 2) * (1 + abs(out[0])):
            raise ArithmeticError("SpheroidalJoiningFactor: inconsistent ratio (numerical failure)")
        return out[1]
    return _verified(run, "SpheroidalJoiningFactor")


mpfun("SpheroidalJoiningFactor", 3, _sph_joining,
      real=lambda n, m, g: _allreal(n, m, g),
      notes="joining factor SpheroidalS1[n, m, c, z] / SpheroidalPS[n, m, c, z] (documented for m = 0), for m != 0 "
            "with the Ferrers phase (-I)^|m| of SpheroidalPS on z > 1 removed (real for real c); integer n >= |m|, "
            "real or imaginary c"
            ". Check: reference (differential cases in tests/library).")


# ---------------------------------------------------------------------------------------------------------------
# Angular spheroidal functions of the second kind (DLMF 30.8.9, Meixner-Schaefke; Wolfram's SpheroidalQS):
# Qs_n^m(x) = Sum_{k <= -N-1} (-1)^k a'_k P^m_{n+2k}(x) + Sum_{k >= -N} (-1)^k a_k Q^m_{n+2k}(x),  N = Floor[(n+m)/2],
# a_k (k >= -R) the Ps coefficients, continued for -N <= k < -R by forward recursion of DLMF 30.8.4 from
# a_{-N-1} = 0, and a'_k the recessive solution of 30.8.4 as k -> -Infinity normalised by DLMF 30.8.10-11.
# Ferrers functions (type 2), analytically continued off [-1, 1] as LegendreQ[n, m, x] (mpmath type 2).
# Reproduces every documented value (SpheroidalQS[3, 2, 1, 0.5] = -0.263019, N[SpheroidalQS[2, 1, 1/3, -11], 20],
# SpheroidalQS[23, 5, 2 I, 2/3], ...).
# ---------------------------------------------------------------------------------------------------------------

def _sph_q_coeffs(n, m, g2):
    """({k: a_k} for k >= -N, {k: a'_k} for k <= -N-1) of DLMF 30.8.9 (m >= 0)."""
    lam, b = _sph_system(n, m, g2)
    a = {(l - n) // 2: (-1) ** ((l - n) // 2) * bl for l, bl in b.items()}
    R, N = (n - m) // 2, (n + m) // 2

    def A(k):
        return -g2 * mpmath.mpf((n - m + 2 * k - 1) * (n - m + 2 * k)) / ((2 * n + 4 * k - 3) * (2 * n + 4 * k - 1))

    def B(k):
        return (n + 2 * k) * (n + 2 * k + 1) - \
            2 * g2 * mpmath.mpf((n + 2 * k) * (n + 2 * k + 1) - 1 + m * m) / ((2 * n + 4 * k - 1) * (2 * n + 4 * k + 3))

    def C(k):
        return -g2 * mpmath.mpf((n + m + 2 * k + 1) * (n + m + 2 * k + 2)) / ((2 * n + 4 * k + 3) * (2 * n + 4 * k + 5))
    if N > R:
        f = {-N - 1: mpmath.mpf(0), -N: mpmath.mpf(1)}
        for k in range(-N, -R):
            f[k + 1] = -(A(k) * f[k - 1] + (B(k) - lam) * f[k]) / C(k)
        if f[-R] == 0:
            raise ValueError("SpheroidalQS: the coefficients a_{n,k} are undefined for these parameters (DLMF 30.8)")
        s = a[-R] / f[-R]
        for k in range(-N, -R):
            a[k] = f[k] * s
    Cp = g2 / (4 * m * m - 1) if (n - m) % 2 == 0 else -g2 / mpmath.mpf((2 * m - 1) * (2 * m - 3))
    K = -N - 1 - 40 - mpmath.mp.prec // 3
    rho, rhos = mpmath.mpf(0), {}
    for k in range(K + 1, -N):        # rho_k = a'_{k-1} / a'_k, by the recessive (downward-minimal) continued fraction
        rho = -C(k - 1) / (B(k - 1) - lam + A(k - 1) * rho)
        rhos[k] = rho
    ap = {-N - 1: -Cp * a[-N] / (A(-N - 1) * rhos[-N - 1] + B(-N - 1) - lam)}
    for k in range(-N - 1, K + 2, -1):
        ap[k - 1] = rhos[k] * ap[k]
    return a, ap


def _sph_qs(deriv):
    name = "SpheroidalQS" + ("Prime" if deriv else "")

    def f(n, m, g, x):
        n, m = _sph_nm(n, m, name)
        g2 = _sph_g2(g, name)
        if x in (1, -1):
            raise ValueError(f"{name}: x = +-1 is a singular point")
        am = abs(m)

        def run():
            if g2 == 0:
                a, ap = {0: mpmath.mpf(1)}, {}
            else:
                a, ap = _sph_q_coeffs(n, am, g2)
            degs = [n + 2 * k for k in a]
            lo, hi = min(degs), max(degs) + 1
            inside = not isinstance(x, mpmath.mpc) and abs(x) < 1
            if inside:
                # Ferrers Q on (-1, 1): the three-term recurrence in the degree is stable (P, Q both oscillatory)
                # (the recurrence cannot cross l = m - 1 -> m, whose coefficient l - m + 1 vanishes)
                top = max(lo + 1, am)
                Q = {l: mpmath.legenq(l, am, x, type=2) for l in range(lo, top + 1)}
                for l in range(top, hi):
                    Q[l + 1] = ((2 * l + 1) * x * Q[l] - (l + am) * Q[l - 1]) / (l - am + 1)
                getq = Q.__getitem__
            else:
                # off (-1, 1) the continued Ferrers Q mixes the dominant P and the minimal Q, so no recurrence
                # direction is stable: evaluate each degree directly, stopping once the terms are negligible
                Q = {}

                def getq(l):
                    if l not in Q:
                        Q[l] = mpmath.legenq(l, am, x, type=2)
                    return Q[l]

            def dq(l):
                if not deriv:
                    return getq(l)
                return ((l + 1) * x * getq(l) - (l - am + 1) * getq(l + 1)) / (1 - x * x)
            terms, small = [], 0
            tiny = mpmath.mpf(2) ** (-mpmath.mp.prec - 10)
            for k in sorted(a):
                t = (-1) ** k * a[k] * dq(n + 2 * k)
                terms.append(t)
                if not inside and k > -(n - am) // 2:
                    small = small + 1 if abs(t) <= tiny * max(abs(u) for u in terms) else 0
                    if small >= 3:
                        break
            out = mpmath.fsum(terms)
            if ap:
                Ls = {k: -(n + 2 * k) - 1 for k in ap}      # P^m_nu = P^m_{-nu-1}
                tab = _legP_table(max(Ls.values()) + 1, am, x)
                out += mpmath.fsum((-1) ** k * apk * _legP(Ls[k], am, x, deriv, tab) for k, apk in ap.items())
            if m < 0:
                out *= (-1) ** am * mpmath.factorial(n - am) / mpmath.factorial(n + am)
            return out
        return _verified(run, name)

    def exact(n, m, g, x):
        if any(isinstance(v, list) for v in (n, m, g, x)) or g != 0 or not (n.is_integer and m.is_integer):
            return None
        if deriv:
            return None
        return Expr("LegendreQ", n, m, x) if n >= abs(m) else None
    mpfun(name, 4, f, exact,
          notes=f"{'x-derivative of the ' if deriv else ''}angular spheroidal function of the second kind "
                "Qs_n^m(gamma, x) (DLMF 30.8.9 Meixner-Schaefke series in Ferrers functions; SpheroidalQS[n, m, 0, x] "
                "= LegendreQ[n, m, x]); integer n >= |m|, real or imaginary gamma (others raise); x = +-1 raises")


_sph_qs(False)
_sph_qs(True)


# ---------------------------------------------------------------------------------------------------------------
# SiegelTheta[Omega, s] = Sum_{n in Z^p} Exp[Pi I n.Omega.n + 2 Pi I n.s];
# SiegelTheta[{nu1, nu2}, Omega, s] = Sum Exp[Pi I (n+nu1).Omega.(n+nu1) + 2 Pi I (n+nu1).(s+nu2)].
# The lattice sum is truncated to the ellipsoid outside which every term is below 2^-(prec+20) of the largest.
# ---------------------------------------------------------------------------------------------------------------

_SIEGEL_MAX_TERMS = 400000


def _siegel_sum(nu, Om, s, bits):
    name = "SiegelTheta"
    if not (isinstance(Om, list) and Om and all(isinstance(r, list) and len(r) == len(Om) for r in Om)):
        raise ValueError(f"{name}: Omega must be a square matrix")
    p = len(Om)
    if not (isinstance(s, list) and len(s) == p and not any(isinstance(x, list) for x in s)):
        raise ValueError(f"{name}: s must be a vector of length {p}")
    if nu is None:
        nu1 = nu2 = [mpmath.mpf(0)] * p
    else:
        if not (isinstance(nu, list) and len(nu) == 2 and all(isinstance(v, list) and len(v) == p for v in nu)):
            raise ValueError(f"{name}: the characteristics must be {{nu1, nu2}} with vectors of length {p}")
        nu1, nu2 = nu
    Om = [[mpmath.mpmathify(x) for x in r] for r in Om]
    tol = mpmath.mpf(2) ** (-mpmath.mp.prec + 8)
    for i in range(p):
        for j in range(i + 1, p):
            if abs(Om[i][j] - Om[j][i]) > tol * (1 + abs(Om[i][j])):
                raise ValueError(f"{name}: Omega must be a symmetric matrix")
    Y = mpmath.matrix([[mpmath.im(Om[i][j]) for j in range(p)] for i in range(p)])
    try:
        ev = mpmath.eigsy(Y, eigvals_only=True)
    except Exception:
        raise ValueError(f"{name}: Im[Omega] must be positive definite")
    lmin = min(ev)
    if lmin <= 0:
        raise ValueError(f"{name}: Im[Omega] must be positive definite")
    # exponent real part: -Pi [ (n+nu1).Y.(n+nu1) + 2 (n+nu1).Im(s+nu2) ] = -Pi [ (v-c).Y.(v-c) - c.Y.c ], c = -Y^-1 Im
    w = mpmath.matrix([mpmath.im(mpmath.mpmathify(s[i]) + mpmath.mpmathify(nu2[i])) for i in range(p)])
    c = -(mpmath.inverse(Y) * w)
    R2 = bits * mpmath.log(2) / mpmath.pi + 2
    half = mpmath.sqrt(R2 / lmin) + 1
    centre = [c[i] - mpmath.re(mpmath.mpmathify(nu1[i])) for i in range(p)]
    ranges = [range(int(mpmath.floor(centre[i] - half)), int(mpmath.ceil(centre[i] + half)) + 1) for i in range(p)]
    total = 1
    for r in ranges:
        total *= len(r)
    if total > _SIEGEL_MAX_TERMS:
        raise NotImplementedError(f"{name}: Im[Omega] is too close to singular for the lattice sum "
                                  f"({total} terms would be needed)")
    import itertools
    ipi = mpmath.mpc(0, 1) * mpmath.pi
    s2 = [mpmath.mpmathify(s[i]) + mpmath.mpmathify(nu2[i]) for i in range(p)]
    n1 = [mpmath.mpmathify(x) for x in nu1]
    acc = []
    for nvec in itertools.product(*ranges):
        v = [nvec[i] + n1[i] for i in range(p)]
        q = mpmath.fsum(v[i] * Om[i][j] * v[j] for i in range(p) for j in range(p))
        lin = mpmath.fsum(v[i] * s2[i] for i in range(p))
        e = ipi * (q + 2 * lin)
        if mpmath.re(e) < -bits * mpmath.log(2) - mpmath.pi * mpmath.fsum(c[i] * w[i] for i in range(p)) - 10:
            continue
        acc.append(mpmath.exp(e))
    total = mpmath.fsum(acc)
    big = max((abs(t) for t in acc), default=mpmath.mpf(0))
    return total, big


def _siegel(nu, Om, s):
    """Lattice sum with the working precision raised until cancellation between terms is covered."""
    target = mpmath.mp.prec
    work, bits = target, target + 30
    while True:
        with mpmath.workprec(work):
            total, big = _siegel_sum(nu, Om, s, bits)
        if big == 0:
            return total
        loss = 0 if total == 0 else int(mpmath.log(big / abs(total), 2))
        if total != 0 and work - loss >= target + 20:
            return total
        if total == 0:
            loss = work
        if work > max(8 * target, target + 6000):
            break
        work = max(2 * work, target + loss + 40)
        bits = work + 30
    raise ArithmeticError("SiegelTheta: cancellation in the lattice sum could not be resolved")


def _siegel_dispatch(*args):
    if len(args) == 2:
        return _verified(lambda: _siegel(None, args[0], args[1]), "SiegelTheta")
    return _verified(lambda: _siegel(args[0], args[1], args[2]), "SiegelTheta")


mpfun("SiegelTheta", (2, 3), _siegel_dispatch, listable=False,
      notes="Siegel (Riemann) theta function Sum_n Exp[Pi I n.Omega.n + 2 Pi I n.s] over Z^p, with characteristics "
            "{nu1, nu2}; Omega symmetric with positive definite imaginary part (otherwise raises); numerical "
            "lattice sum (exact input stays unevaluated, N gives digits)"
            ". Check: reference (differential cases in tests/library).")


# ---------------------------------------------------------------------------------------------------------------
# CompleteIntegral[pde, u, {x1, ..., xn}]: a complete integral (n independent arbitrary constants) of a first-order
# PDE F(x, u, Du) = 0.  Methods (each candidate is substituted back into the PDE and its constants are checked to be
# independent -- rank n of the matrix (u_a, u_{x_i a}) -- before it is returned; otherwise the call stays
# unevaluated):
#   * Clairaut type u = x.Du + f(Du) (F free of x after u -> a.x + b, Du -> a): planes u = a.x + b(a);
#   * equations without u that separate additively, F = Sum_i G_i(x_i, p_i) (possibly after dividing by a factor
#     depending on one x_i), including F(Du) = 0: p_i = C[k] when G_i does not involve x_i, else G_i = C[k], and
#     the remaining p_j (the last p_j in which F is linear) from F = 0; u = Sum Integrate[p_i, x_i] + C[n].
# Conditions u[a1, ..., an] == v joined with && fix the constants (Solve).
# ---------------------------------------------------------------------------------------------------------------

def _ci_flatten(e):
    if isinstance(e, Expr) and e.head in ("And", "List"):
        return [y for a in e.args for y in _ci_flatten(a)]
    return [e]


def _ci_has_derivative(e):
    if isinstance(e, Expr):
        if isinstance(e.head, Expr):
            return True
        return any(_ci_has_derivative(a) for a in e.args)
    return False


def _complete_integral(pde, uspec, xs, *opts):
    from ..math_ops import to_sympy, from_sympy
    name = "CompleteIntegral"
    if opts:
        raise NotImplementedError(f"{name}: options are not supported")
    if not is_list(xs) or not xs.args or not all(isinstance(v, Symbol) for v in xs.args):
        raise ValueError(f"{name}: the independent variables must be a list of symbols")
    xs_ = list(xs.args)
    n = len(xs_)
    if isinstance(uspec, Symbol):
        uname, pure = uspec.name, True
    elif isinstance(uspec, Expr) and isinstance(uspec.head, str) and list(uspec.args) == xs_:
        uname, pure = uspec.head, False
    else:
        raise ValueError(f"{name}: the unknown must be u or u[x1, ..., xn] with the given variables")
    parts = _ci_flatten(pde)
    eqs = [e for e in parts if _ci_has_derivative(e)]
    conds = [e for e in parts if not _ci_has_derivative(e)]
    if len(eqs) != 1 or not (isinstance(eqs[0], Expr) and eqs[0].head == "Equal" and len(eqs[0].args) == 2):
        raise Unevaluated()
    for c in conds:
        if not (isinstance(c, Expr) and c.head == "Equal" and len(c.args) == 2 and isinstance(c.args[0], Expr)
                and c.args[0].head == uname and len(c.args[0].args) == n):
            raise ValueError(f"{name}: conditions must have the form {uname}[a1, ..., an] == value")
    X = [sp.Symbol(v.name) for v in xs_]
    F0 = to_sympy(eqs[0].args[0]) - to_sympy(eqs[0].args[1])
    ufun = sp.Function(uname)
    U = sp.Dummy("U")
    P = [sp.Dummy(f"p{i}") for i in range(n)]
    rep = {sp.Derivative(ufun(*X), X[i]): P[i] for i in range(n)}
    F = F0.xreplace(rep).xreplace({ufun(*X): U})
    if any(isinstance(t, (sp.Derivative, sp.Subs)) for t in sp.preorder_traversal(F)) or \
            any(isinstance(t, AppliedUndef) and t.func.__name__ == uname for t in sp.preorder_traversal(F)):
        raise Unevaluated()     # higher-order derivatives or u at other points: not a first-order PDE
    Cs = [sp.Dummy(f"C{k + 1}") for k in range(n)]

    cands = []
    if F.has(U):
        cands += _ci_clairaut(F, U, P, X, Cs)
    elif not (F.free_symbols & set(X)):
        cands += _ci_planes(F, P, X, Cs)
    else:
        cands += _ci_separable(F, P, X, Cs)
    for cand in cands:
        if _ci_verify(F0, ufun, X, cand, Cs):
            sols = [cand]
            if conds:
                sols = _ci_conditions(conds, cand, X, Cs, to_sympy)
                if sols is None:
                    continue
            Cfun = sp.Function("C")
            outl = []
            for s_ in sols:
                s_ = s_.xreplace({c: Cfun(k + 1) for k, c in enumerate(Cs)})
                val = from_sympy(s_)
                if pure:
                    outl.append(L(rule(Symbol(uname), Expr("Function", L(*xs_), val))))
                else:
                    outl.append(L(rule(uspec, val)))
            return L(*outl)
    raise Unevaluated()


def _ci_clairaut(F, U, P, X, Cs):
    D = sp.Dummy("D")
    G = sp.simplify(F.xreplace({U: sum(c * x for c, x in zip(Cs, X)) + D}).xreplace(dict(zip(P, Cs))))
    if G.free_symbols & set(X) or not G.has(D):
        return []
    try:
        roots = sp.solve(G, D, dict=False)
    except Exception:
        return []
    return [sum(c * x for c, x in zip(Cs, X)) + r for r in roots if not (r.free_symbols & set(X))]


def _ci_planes(F, P, X, Cs):
    """F(Du) = 0: planes u = Sum_{i != j} C[k] x_i + p_j(C) x_j + C[n], p_j from F = 0."""
    n = len(X)
    lin = [i for i in range(n) if F.is_polynomial(P[i]) and sp.Poly(F, P[i]).degree() == 1]
    order = ([lin[-1]] if lin else []) + [i for i in range(n) if not lin or i != lin[-1]]
    for j in order:
        others = [i for i in range(n) if i != j]
        sub = {P[i]: Cs[k] for k, i in enumerate(others)}
        try:
            r = sp.solve(F.xreplace(sub), P[j])
        except Exception:
            continue
        if r:
            return [sum(Cs[k] * X[i] for k, i in enumerate(others)) + r[0] * X[j] + Cs[n - 1]]
    return []


def _ci_separable(F, P, X, Cs):
    n = len(X)
    divisors = [sp.Integer(1)]
    for t in sp.Add.make_args(sp.expand(F)):
        for fac in sp.Mul.make_args(t):
            fs = fac.free_symbols
            if fs and fs <= set(X) and len(fs) == 1 and fac not in divisors:
                divisors.append(fac)
    for d in divisors:
        H = sp.expand(F / d)
        groups = [sp.Integer(0)] * n
        const = sp.Integer(0)
        ok = True
        for t in sp.Add.make_args(H):
            owners = [i for i in range(n) if t.has(X[i]) or t.has(P[i])]
            if len(owners) > 1:
                ok = False
                break
            if owners:
                groups[owners[0]] += t
            else:
                const += t
        if not ok or any(g == 0 for g in groups):
            continue
        lin = [i for i in range(n) if sp.Poly(groups[i], P[i]).degree() == 1] if all(
            groups[i].is_polynomial(P[i]) for i in range(n)) else \
            [i for i in range(n) if groups[i].is_polynomial(P[i]) and sp.Poly(groups[i], P[i]).degree() == 1]
        order = ([lin[-1]] if lin else []) + [i for i in range(n) if not lin or i != lin[-1]]
        for j in order:
            pvals, gvals, k = {}, {}, 0
            good = True
            for i in range(n):
                if i == j:
                    continue
                c = Cs[k]
                k += 1
                if not groups[i].has(X[i]):
                    pvals[i] = c
                    gvals[i] = groups[i].xreplace({P[i]: c})
                else:
                    try:
                        r = sp.solve(sp.Eq(groups[i], c), P[i])
                    except Exception:
                        r = []
                    if not r:
                        good = False
                        break
                    pvals[i] = r[0]
                    gvals[i] = c
            if not good:
                continue
            try:
                rj = sp.solve(groups[j] + sum(gvals.values()) + const, P[j])
            except Exception:
                continue
            if not rj:
                continue
            pvals[j] = rj[0]
            u = Cs[n - 1]
            for i in range(n):
                part = sp.integrate(pvals[i], X[i], conds="none")
                if part.has(sp.Integral) or part.has(sp.Piecewise):
                    good = False
                    break
                u += part
            if good:
                return [sp.expand(u) if not u.has(sp.Pow) else u]
    return []


def _ci_verify(F0, ufun, X, cand, Cs):
    import random
    lam = sp.Lambda(tuple(X), cand)
    try:
        r = F0.replace(ufun, lam).doit()
    except Exception:
        return False
    try:
        if sp.simplify(r) != 0:
            rnd = random.Random(7)
            syms = sorted(r.free_symbols, key=str)
            for _ in range(4):
                vals = {s_: sp.Rational(rnd.randint(3, 29), 10) for s_ in syms}
                z = complex(sp.N(r.subs(vals), 30))
                if abs(z) > 1e-15:
                    return False
    except Exception:
        return False
    # independence of the n constants: rank of [u_a, u_{x1 a}, ..., u_{xn a}]_a must be n (checked at random points)
    M = sp.Matrix([[sp.diff(cand, a)] + [sp.diff(cand, x, a) for x in X] for a in Cs])
    rnd = random.Random(3)
    for _ in range(3):
        vals = {s_: sp.Rational(rnd.randint(3, 29), 7) for s_ in M.free_symbols}
        try:
            Mn = M.subs(vals).evalf(30)
            if Mn.rank(iszerofunc=lambda e: abs(complex(e)) < 1e-20) == len(Cs):
                return True
        except Exception:
            continue
    return False


def _ci_conditions(conds, cand, X, Cs, to_sympy):
    eqs = []
    for c in conds:
        pts = [to_sympy(a) for a in c.args[0].args]
        eqs.append(sp.Eq(cand.xreplace(dict(zip(X, pts))), to_sympy(c.args[1])))
    try:
        sols = sp.solve(eqs, Cs, dict=True)
    except Exception:
        return None
    if not sols:
        return None
    return [sp.simplify(cand.xreplace(s_)) for s_ in sols]


@fn("CompleteIntegral", area=AREA, status="added",
    notes="complete integral of a first-order PDE (n independent constants C[k]): Clairaut equations and equations "
          "without u that separate additively (F(Du) = 0, Hamilton-Jacobi type, f(x, p) = g(y, q)); every result is "
          "substituted back and its constants checked independent; conditions u[...] == v fix the constants; other "
          "PDEs stay unevaluated. Constants may be numbered differently from Wolfram"
          ". Check: reference (differential cases in tests/library).")
def _complete_integral_3(pde: Expr, u: Expr, xs: Expr) -> Expr:
    return _complete_integral(pde, u, xs)
