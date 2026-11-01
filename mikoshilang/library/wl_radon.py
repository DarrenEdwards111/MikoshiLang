"""Wolfram batch X04: RadonTransform and InverseRadonTransform (symbolic, via SymPy, numerically cross-checked)."""
from __future__ import annotations

import math

import mpmath
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, R
from .wl_signal import _sym_expr

AREA = "image processing III (X04)"
_CHECK_POINTS = ((sp.Rational(3, 10), sp.Rational(7, 10)), (sp.Rational(4, 5), sp.Rational(21, 10)))
_BREAKS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)


def _vars(v, fname, what):
    if not (K.is_list(v) and len(v.args) == 2 and all(isinstance(a, Symbol) for a in v.args)):
        if isinstance(v, Symbol):
            raise TypeError(f"{fname}: the {what} must be a list of two symbols")
        raise TypeError(f"{fname}: the {what} must be a list of two symbols, not {v!r}")
    a, b = v.args
    if a.name == b.name:
        raise ValueError(f"{fname}: the {what} must be distinct")
    return a.name, b.name


def _options(opts, fname):
    for r in opts:
        if not (isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and isinstance(r.args[0], Symbol)):
            raise TypeError(f"{fname}: {r!r} is not an option")
        name, v = r.args[0].name, r.args[1]
        if name == "GenerateConditions":
            if v is True:
                raise NotImplementedError(f"{fname}: GenerateConditions -> True is not supported")
            if v is not False and not (isinstance(v, Symbol) and v.name == "Automatic"):
                raise ValueError(f"{fname}: GenerateConditions must be True or False")
        elif name == "Assumptions":
            if not (v is True or (isinstance(v, Symbol) and v.name in ("$Assumptions", "Automatic"))):
                raise NotImplementedError(f"{fname}: explicit Assumptions are not supported")
        elif name == "Method":
            if not (isinstance(v, Symbol) and v.name == "Automatic"):
                raise NotImplementedError(f"{fname}: Method {v!r} is not supported")
        else:
            raise ValueError(f"{fname}: unknown option {name}")


_REL_NAMES = {"Less": sp.Lt, "LessEqual": sp.Le, "Greater": sp.Gt, "GreaterEqual": sp.Ge}


def _rel(c):
    """A SymPy relational for a condition (also the undefined Less/Greater/... functions of converted Wolfram
    input); other conditions are returned unchanged."""
    name = getattr(c.func, "__name__", "")
    if name in _REL_NAMES and len(c.args) == 2:
        return _REL_NAMES[name](*c.args)
    return c


def _indicator_cond(fac):
    """The condition of an indicator factor (Boole, UnitStep, Heaviside, Piecewise((1, c), (0, True))) or None."""
    name = getattr(fac.func, "__name__", "")
    if name == "Boole" and len(fac.args) == 1:
        return _rel(fac.args[0])
    if name == "UnitStep" and len(fac.args) == 1:
        return sp.Ge(fac.args[0], 0)
    if isinstance(fac, sp.Heaviside) or name == "HeavisideTheta":
        return sp.Gt(fac.args[0], 0)
    if isinstance(fac, sp.Piecewise) and len(fac.args) == 2 and fac.args[0][0] == 1 and fac.args[1][0] == 0 \
            and fac.args[1][1] == sp.true:
        return fac.args[0][1]
    return None


def _split_indicators(g):
    inds, rest = [], []
    for fac in sp.Mul.make_args(g):
        c = _indicator_cond(fac)
        (inds if c is not None else rest).append(c if c is not None else fac)
    return inds, sp.Mul(*rest)


def _positive(a):
    """True when a (an expression in real symbols) is provably positive: SymPy's assumptions, or for a function of
    one real symbol, a positive global minimum (sympy.minimum)."""
    if a.is_positive:
        return True
    for form in (sp.expand(a.rewrite(sp.cos)), sp.trigsimp(a)):
        if form.is_positive:
            return True
    if len(a.free_symbols) == 1:
        v = next(iter(a.free_symbols))
        try:
            m = sp.minimum(a, v, sp.S.Reals)
        except Exception:
            return False
        return bool(m.is_positive)
    return False


def _interval_integral(h, cond, s):
    """∫ h ds over {s : cond}, for cond a strict or non-strict inequality q(s) < 0 with q quadratic in s and a
    positive leading coefficient (a bounded interval). None if cond has another form."""
    if not isinstance(cond, (sp.StrictLessThan, sp.LessThan, sp.StrictGreaterThan, sp.GreaterThan)):
        return None
    q = cond.lhs - cond.rhs if isinstance(cond, (sp.StrictLessThan, sp.LessThan)) else cond.rhs - cond.lhs
    q = sp.expand(sp.trigsimp(sp.expand(q)))
    poly = sp.Poly(q, s) if q.is_polynomial(s) else None
    if poly is None or poly.degree() != 2:
        return None
    a, b, c = [sp.trigsimp(v) for v in poly.all_coeffs()]
    if not _positive(a):
        return None
    D = sp.simplify(sp.trigsimp(b ** 2 - 4 * a * c))
    lo = (-b - sp.sqrt(D)) / (2 * a)
    hi = (-b + sp.sqrt(D)) / (2 * a)
    val = sp.integrate(h, (s, lo, hi), conds="none")
    if val.has(sp.Integral):
        raise Unevaluated()
    return sp.Piecewise((sp.simplify(val), D > 0), (0, True)), (lo, hi)


def _numeric(expr, subs):
    v = complex(sp.N(expr.subs(subs), 20))
    return v


def _quad(fun, extra=()):
    pts = sorted(set([-mpmath.inf] + [mpmath.mpf(b) for b in _BREAKS] + [mpmath.mpf(float(e)) for e in extra] +
                     [mpmath.inf]))
    with mpmath.workdps(20):
        val, err = mpmath.quad(fun, pts, error=True, maxdegree=8)
    return complex(val), float(err)


def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol * (1 + abs(a) + abs(b))


def _param_subs(expr, keep):
    return {v: sp.Rational(11 + 2 * k, 10) for k, v in enumerate(sorted(expr.free_symbols - set(keep), key=str))}


@fn("RadonTransform", area=AREA, status="added",
    notes="RadonTransform[f, {x, y}, {p, φ}] = ∫_-∞^∞ f(p Cos[φ] - s Sin[φ], p Sin[φ] + s Cos[φ]) ds (the integral "
          "of f along the line x Cos[φ] + y Sin[φ] = p), computed by SymPy's integrate (with the trigonometric "
          "simplification Cos^2 + Sin^2 = 1 first and principal-branch results, as GenerateConditions -> False). "
          "A single indicator factor (Boole[cond], UnitStep, HeavisideTheta or Piecewise[{{1, cond}}, 0]) whose "
          "condition is a quadratic inequality in s with a positive leading coefficient (discs, ellipses) is "
          "integrated over its chord, giving Piecewise[{{value, discriminant > 0}}, 0]. Integrals SymPy cannot do "
          "stay unevaluated. Options: GenerateConditions -> False, Assumptions -> $Assumptions and Method -> "
          "Automatic (other settings raise NotImplementedError). Check: numeric (the closed form is compared with "
          "mpmath quadrature of the line integral at (p, φ) = (0.3, 0.7) and (0.8, 2.1), other parameters set to "
          "1.1, 1.3, ...; a mismatch raises ArithmeticError).")
def _radontransform(f: Expr, xy: ListOf[Symbol], pphi: ListOf[Symbol], *opts: Option) -> Expr:
    fname = "RadonTransform"
    xn, yn = _vars(xy, fname, "integration variables")
    pn, phin = _vars(pphi, fname, "transform variables")
    _options(opts, fname)
    fe = _sym_expr(f, fname)
    X, Y = sp.Symbol(xn), sp.Symbol(yn)
    p, phi = sp.Symbol(pn, real=True), sp.Symbol(phin, real=True)
    s = sp.Dummy("s", real=True)
    fe = fe.subs({sp.Symbol(pn): p, sp.Symbol(phin): phi})
    line = fe.subs({X: p * sp.cos(phi) - s * sp.sin(phi), Y: p * sp.sin(phi) + s * sp.cos(phi)}, simultaneous=True)
    if not line.has(s):
        if sp.simplify(line) == 0:
            return 0
        raise ValueError(f"{fname}: the integral of a function that does not decay along lines diverges")
    inds, h = _split_indicators(line)
    h = sp.trigsimp(sp.expand(h)) if not h.has(sp.Piecewise) else h
    extra_pts = []
    if len(inds) == 1:
        got = _interval_integral(h, inds[0], s)
        if got is None:
            raise Unevaluated()
        res, (lo, hi) = got
        extra_pts = [lo, hi]
    elif inds:
        raise Unevaluated()
    else:
        try:
            res = sp.integrate(h, (s, -sp.oo, sp.oo), conds="none")
        except Exception:
            raise Unevaluated() from None
        if res.has(sp.Integral) or res.has(sp.nan) or res.has(sp.zoo):
            raise Unevaluated()
        res = sp.unpolarify(res)
    res = sp.simplify(sp.trigsimp(res))
    _check_radon(fname, line, res, s, p, phi, extra_pts)
    return R(res.subs({p: sp.Symbol(pn), phi: sp.Symbol(phin)}))


def _check_radon(fname, line, res, s, p, phi, extra_pts):
    for p0, f0 in _CHECK_POINTS:
        subs = {p: p0, phi: f0}
        subs.update(_param_subs(line, (s, p, phi)))
        try:
            exact = _numeric(res, subs)
            g = sp.lambdify(s, line.subs(subs).replace(
                lambda e: getattr(e.func, "__name__", "") in ("Boole", "UnitStep"),
                lambda e: sp.Piecewise((1, _indicator_cond(e)), (0, True))), modules="mpmath")
            pts = []
            for e in extra_pts:
                v = complex(sp.N(e.subs(subs)))
                if abs(v.imag) < 1e-12:
                    pts.append(v.real)
            direct, err = _quad(lambda t: g(t), pts)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError, AttributeError, NameError):
            continue
        if not (math.isfinite(abs(direct)) and math.isfinite(abs(exact))) or err > 1e-8 * (1 + abs(direct)):
            continue
        if not _close(direct, exact):
            raise ArithmeticError(f"{fname}: self-check failed (closed form {exact} vs quadrature {direct} at "
                                  f"p = {p0}, φ = {f0})")


@fn("InverseRadonTransform", area=AREA, status="added",
    notes="InverseRadonTransform[g, {p, φ}, {x, y}] for g independent of φ (the transform of a radial function; "
          "g must then be even in p): f(r) = -(1/Pi) ∫_r^∞ g'(p)/Sqrt[p^2 - r^2] dp (Abel inversion; computed as "
          "-(1/Pi) ∫_0^∞ g'(Sqrt[r^2 + t^2])/Sqrt[r^2 + t^2] dt with r > 0 by SymPy) with r = Sqrt[x^2 + y^2]. "
          "Transforms that depend on φ raise NotImplementedError (the general filtered back-projection / Cormack "
          "inversion is not implemented); integrals SymPy cannot do stay unevaluated. Options as RadonTransform. "
          "Check: numeric (the result's Radon transform, by mpmath quadrature, is compared with g at p = 0.3 and "
          "0.8; a mismatch raises ArithmeticError).")
def _inverseradontransform(g: Expr, pphi: ListOf[Symbol], xy: ListOf[Symbol], *opts: Option) -> Expr:
    fname = "InverseRadonTransform"
    pn, phin = _vars(pphi, fname, "transform variables")
    xn, yn = _vars(xy, fname, "result variables")
    _options(opts, fname)
    ge = _sym_expr(g, fname)
    P, PHI = sp.Symbol(pn), sp.Symbol(phin)
    if ge.has(PHI) and sp.simplify(sp.diff(ge, PHI)) != 0:
        raise NotImplementedError(f"{fname}: only transforms independent of {phin} (radial functions) are supported")
    ge = ge.subs(PHI, 0)
    if sp.simplify(ge.subs(P, -P) - ge) != 0:
        raise ValueError(f"{fname}: a Radon transform independent of {phin} must be even in {pn}")
    r = sp.Symbol("r", positive=True)
    t = sp.Dummy("t", positive=True)
    dg = sp.diff(ge, P)
    rho = sp.sqrt(r ** 2 + t ** 2)
    integrand = sp.simplify(dg.subs(P, rho) / rho)
    try:
        f = -sp.integrate(integrand, (t, 0, sp.oo), conds="none") / sp.pi
    except Exception:
        raise Unevaluated() from None
    if f.has(sp.Integral) or f.has(sp.nan) or f.has(sp.zoo):
        raise Unevaluated()
    f = sp.simplify(sp.unpolarify(f))
    for p0 in (sp.Rational(3, 10), sp.Rational(4, 5)):
        subs = _param_subs(f.subs(r, 1) + ge.subs(P, 1), ())
        try:
            fr = sp.lambdify(r, f.subs(subs), modules="mpmath")
            direct, err = _quad(lambda s_: fr(mpmath.sqrt(p0 ** 2 + s_ ** 2)) if s_ != 0 or p0 != 0 else fr(p0))
            want = _numeric(ge.subs(P, p0), subs)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError, AttributeError, NameError):
            continue
        if not (math.isfinite(abs(direct)) and math.isfinite(abs(want))) or err > 1e-8 * (1 + abs(direct)):
            continue
        if not _close(direct, want):
            raise ArithmeticError(f"{fname}: self-check failed (forward transform {direct} vs {want} at p = {p0})")
    X, Y = sp.Symbol(xn), sp.Symbol(yn)
    return R(sp.simplify(f.subs(r, sp.sqrt(X ** 2 + Y ** 2))))
