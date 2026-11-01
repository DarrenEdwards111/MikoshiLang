"""Special functions.  All conventions follow the Wolfram Language (which agree
with SymPy/mpmath for the functions below unless stated)."""

from __future__ import annotations
import sympy as sp

from .core import (fn, error, S, R, numeric_out, sym, int_arg, nonneg_int, Unevaluated, has_float)

AREA = "special functions"


def _special(name, f, nargs, status, notes, listable=True):
    if nargs == 1:
        @fn(name, area=AREA, status=status, notes=notes, listable=listable)
        def impl(a: Number) -> Number:
            return numeric_out(f(sym(a, name)), a)
    elif nargs == 2:
        @fn(name, area=AREA, status=status, notes=notes, listable=listable)
        def impl(a: Number, b: Number) -> Number:
            return numeric_out(f(sym(a, name), sym(b, name)), a, b)
    elif nargs == 3:
        @fn(name, area=AREA, status=status, notes=notes, listable=listable)
        def impl(a: Number, b: Number, c: Number) -> Number:
            return numeric_out(f(sym(a, name), sym(b, name), sym(c, name)), a, b, c)
    else:
        @fn(name, area=AREA, status=status, notes=notes, listable=listable)
        def impl(a: Number, b: Number, c: Number, d: Number) -> Number:
            return numeric_out(f(*(sym(v, name) for v in (a, b, c, d))), a, b, c, d)
    return impl


EXACT = "now exact/symbolic via SymPy (legacy converted every argument to a float, so symbolic input failed and exact input gave floats)"

_special("Gamma", sp.gamma, 1, "verified", "Euler gamma (SymPy)")
_special("Gamma", sp.uppergamma, 2, "verified", "Gamma[a, z] is the upper incomplete gamma function")
_special("Beta", lambda a, b: sp.expand_func(sp.beta(a, b)) if a.is_number and b.is_number else sp.beta(a, b),
         2, "fixed", "Euler beta, exact closed forms for numeric arguments (Beta[1/2, 1/2] = Pi; legacy left it "
         "unevaluated)")
_special("LogGamma", sp.loggamma, 1, "fixed", EXACT)
_special("DiGamma", sp.digamma, 1, "verified", "MikoshiLang alias of PolyGamma[z]")
_special("TriGamma", lambda z: sp.polygamma(1, z), 1, "fixed", "TriGamma[z] = PolyGamma[1, z]; " + EXACT)
_special("PolyGamma", lambda z: sp.polygamma(0, z), 1, "verified", "PolyGamma[z] = digamma")
_special("PolyGamma", sp.polygamma, 2, "verified", "PolyGamma[n, z]")
_special("Zeta", sp.zeta, 1, "verified", "Riemann zeta")
_special("Zeta", sp.zeta, 2, "verified", "Zeta[s, a] is the Hurwitz zeta function")
_special("HurwitzZeta", sp.zeta, 2, "fixed", "HurwitzZeta[s, a]; " + EXACT)
_special("DirichletEta", sp.dirichlet_eta, 1, "fixed", EXACT)
_special("Erf", sp.erf, 1, "verified", "error function")
_special("Erfc", sp.erfc, 1, "verified", "complementary error function")
_special("Erfi", sp.erfi, 1, "fixed", EXACT)
_special("ExpIntegralEi", sp.Ei, 1, "fixed", EXACT)
_special("ExpIntegralE", sp.expint, 2, "promoted", "ExpIntegralE[n, z] (generalised exponential integral)")
_special("LogIntegral", sp.li, 1, "fixed", "LogIntegral[x] = li(x); " + EXACT)
_special("SinIntegral", sp.Si, 1, "fixed", EXACT)
_special("CosIntegral", sp.Ci, 1, "fixed", EXACT)
for _n, _f in [("BesselJ", sp.besselj), ("BesselY", sp.bessely), ("BesselI", sp.besseli), ("BesselK", sp.besselk)]:
    _special(_n, _f, 2, "verified", "Bessel function (same convention as Wolfram/mpmath)")
_special("HankelH1", sp.hankel1, 2, "fixed", "H1 = J + I Y; " + EXACT)
_special("HankelH2", sp.hankel2, 2, "fixed", "H2 = J - I Y; " + EXACT)
_special("SphericalBesselJ", sp.jn, 2, "promoted", "spherical Bessel j_n(z)")
_special("AiryAi", sp.airyai, 1, "verified", "Airy Ai")
_special("AiryBi", sp.airybi, 1, "verified", "Airy Bi")
_special("ChebyshevT", sp.chebyshevt, 2, "fixed", EXACT)
_special("ChebyshevU", sp.chebyshevu, 2, "fixed", EXACT)
_special("HermiteH", sp.hermite, 2, "fixed", "physicists' Hermite polynomials; " + EXACT)
_special("LaguerreL", sp.laguerre, 2, "fixed", EXACT)
_special("LaguerreL", sp.assoc_laguerre, 3, "fixed", "LaguerreL[n, a, x] (generalised Laguerre)")
_special("LegendreP", sp.legendre, 2, "fixed", EXACT)
_special("LegendreP", sp.assoc_legendre, 3, "fixed", "LegendreP[n, m, x] with the Condon-Shortley phase (Wolfram type 1)")
_special("GegenbauerC", sp.gegenbauer, 3, "fixed", EXACT)
_special("JacobiP", sp.jacobi, 4, "fixed", EXACT)
_special("EllipticK", sp.elliptic_k, 1, "fixed", "parameter m convention (Wolfram); " + EXACT)
_special("EllipticE", sp.elliptic_e, 1, "fixed", "complete, parameter m; " + EXACT)
_special("EllipticE", sp.elliptic_e, 2, "fixed", "EllipticE[phi, m] incomplete")
_special("EllipticEIncomplete", sp.elliptic_e, 2, "fixed", "alias of EllipticE[phi, m]; " + EXACT)
_special("EllipticF", sp.elliptic_f, 2, "fixed", "EllipticF[phi, m]; " + EXACT)
_special("EllipticPi", sp.elliptic_pi, 2, "fixed", "EllipticPi[n, m] complete; " + EXACT)
_special("EllipticPi", sp.elliptic_pi, 3, "fixed", "EllipticPi[n, phi, m] incomplete")
@fn("KroneckerDelta", area=AREA, status="promoted",
    notes="variadic: 1 if all arguments are equal, 0 if any differ, KroneckerDelta[n] = 1 iff n == 0; symbolic "
          "arguments stay unevaluated")
def _kdelta(*args: Number) -> Integer:
    if not args:
        return 1
    vals = [sym(a, "KroneckerDelta") for a in args]
    if len(vals) == 1:
        vals = [vals[0], sp.Integer(0)]
    out = sp.Integer(1)
    for v in vals[1:]:
        d = sp.KroneckerDelta(vals[0], v)
        if d == 0:
            return 0
        out *= d
    return R(out)


@fn("DiscreteDelta", area=AREA, status="promoted",
    notes="Wolfram DiscreteDelta[n1, n2, ...]: 1 if every argument is 0, 0 if any argument is a non-zero number; "
          "symbolic arguments stay unevaluated (previously always inert)")
def _ddelta(*args: Number) -> Integer:
    vals = [sym(a, "DiscreteDelta") for a in args]
    if any(v.is_number and v != 0 for v in vals):
        return 0
    if all(v == 0 for v in vals):
        return 1
    raise Unevaluated()


@fn("UnitStep", area=AREA, status="promoted",
    notes="Wolfram UnitStep[x1, ...]: 1 if every argument is >= 0 (UnitStep[0] = 1), 0 if any is < 0; real "
          "arguments only; symbolic arguments stay unevaluated (previously always inert)")
def _ustep(*args: Real) -> Integer:
    vals = [sym(a, "UnitStep") for a in args]
    for v in vals:
        if v.is_number and v.is_real is False:
            raise ValueError("UnitStep is defined for real arguments only")
    if any(v.is_number and v < 0 for v in vals):
        return 0
    if all(v.is_number and v >= 0 for v in vals):
        return 1
    raise Unevaluated()


@fn("SphericalHarmonicY", area=AREA, status="fixed",
    notes="Y_l^m(theta, phi) with the Condon-Shortley phase (Wolfram); legacy forced floats and returned Python complex")
def _ynm(l: Number, m: Number, theta: Number, phi: Number) -> Number:
    ll = nonneg_int(l, "SphericalHarmonicY", "degree l")
    mm = int_arg(m, "SphericalHarmonicY", "order m")
    if abs(mm) > ll:
        return R(0)
    th, ph = sym(theta, "SphericalHarmonicY"), sym(phi, "SphericalHarmonicY")
    val = sp.Ynm(ll, mm, th, ph).expand(func=True)
    return numeric_out(sp.simplify(val) if not has_float(theta) and not has_float(phi) else val, theta, phi)


def _hyper(p, q):
    def f(*a):
        return sp.hyperexpand(sp.hyper(list(a[:p]), list(a[p:p + q]), a[-1]))
    return f


_special("Hypergeometric0F1", _hyper(0, 1), 2, "fixed", "0F1(; b; z); " + EXACT)
_special("Hypergeometric1F1", _hyper(1, 1), 3, "fixed", "Kummer 1F1(a; b; z); " + EXACT)
_special("Hypergeometric2F1", _hyper(2, 1), 4, "fixed",
         "Gauss 2F1(a, b; c; z); the 2-argument legacy form returned an Error expression and now raises an arity error; " + EXACT)


@fn("LegendreQ", area=AREA, status="fixed", listable=True,
    notes="legacy used scipy lpmv (an associated Legendre P), so LegendreQ was LegendreP; now Legendre Q of "
          "the second kind (Wolfram type 2, |x| < 1) by the three-term recurrence from Q0 = ArcTanh[x]")
def _legendreq(n, x):
    k = nonneg_int(n, "LegendreQ", "degree")
    sx = sym(x, "LegendreQ")
    if sx.is_number and (sx.is_real is False or abs(sx) >= 1):
        raise ValueError("LegendreQ: only the type-2 function on -1 < x < 1 is supported")
    q0 = sp.atanh(sx)
    q1 = sx * q0 - 1
    if k == 0:
        return numeric_out(q0, x)
    a, b = q0, q1
    for j in range(1, k):
        a, b = b, ((2 * j + 1) * sx * b - j * a) / (j + 1)
    return numeric_out(sp.expand(b), x)


@fn("LambertW", area=AREA, status="fixed",
    notes="LambertW[z] and LambertW[z, k] (branch k; note Wolfram's ProductLog[k, z] argument order); "
          "legacy returned Python complex numbers for every input")
def _lambertw(z: Number, k: Integer = 0) -> Number:
    kk = int_arg(k, "LambertW", "branch")
    return numeric_out(sp.LambertW(sym(z, "LambertW"), kk), z)


@fn("ProductLog", area=AREA, status="promoted", notes="Wolfram ProductLog[z] and ProductLog[k, z]")
def _productlog(*a: Number) -> Number:
    if len(a) == 1:
        return numeric_out(sp.LambertW(sym(a[0], "ProductLog")), a[0])
    if len(a) == 2:
        kk = int_arg(a[0], "ProductLog", "branch")
        return numeric_out(sp.LambertW(sym(a[1], "ProductLog"), kk), a[1])
    raise ValueError("ProductLog expects 1 or 2 arguments")


@fn("StruveH", area=AREA, status="promoted", notes="Struve H_n(z), numeric via mpmath")
def _struveh(n: Number, z: Number) -> Number:
    import mpmath
    sn, sz = sym(n, "StruveH"), sym(z, "StruveH")
    if not (sn.is_number and sz.is_number):
        raise Unevaluated()
    if not (has_float(n) or has_float(z)):
        raise Unevaluated()        # no closed form in SymPy: stay exact/unevaluated like Wolfram
    return R(complex(mpmath.struveh(complex(sn), complex(sz))))
