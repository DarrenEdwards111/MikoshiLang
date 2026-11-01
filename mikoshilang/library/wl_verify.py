"""Verification pillar v1 (MikoshiLang extension, not Wolfram): Verify[expr] checks a result independently.

``Verify[expr]`` evaluates ``expr`` and checks the result with a method that does not reuse the algorithm that
produced it:

* ``Certified[value, <|"Method" -> ..., "Certificate" -> ...|>]``: an exact certificate was checked with exact
  arithmetic (prime factorisations with primality proofs, Pratt certificates, Bezout identities, polynomial
  identities by expansion, exact matrix products);
* ``Checked[value, <|"Method" -> ..., ...|>]``: the check passed, but it rests on symbolic simplification or on
  numerical evidence at sample points, which is strong evidence and not a proof;
* ``Unchecked[value, reason]``: no checker applies (or the check was inconclusive);
* ``Failed[value, <|"Head" -> h, "Reason" -> ...|>]``: the check failed. This is Verify's documented result, so
  it is returned, not raised (CONVENTIONS §3). Errors raised by the computation itself still propagate.
"""
from __future__ import annotations

import random

import mpmath
import sympy as sp

from ..expr import Expr, Symbol
from .core import fn, S, R, is_list

AREA = "verification (V1)"
NUMERIC_TOL = mpmath.mpf(10) ** -12
SAMPLE_DPS = 40


def _assoc(**kw):
    return Expr("Association", *(Expr("Rule", k, v) for k, v in kw.items()))


def _certified(value, method, **info):
    return Expr("Certified", value, _assoc(Method=method, **info))


def _checked(value, method, **info):
    return Expr("Checked", value, _assoc(Method=method, **info))


def _unchecked(value, reason):
    return Expr("Unchecked", value, reason)


class VerificationFailure(ArithmeticError):
    """Raised by a checker when the result is wrong; Verify turns it into Failed[value, info]."""

    def __init__(self, head, detail):
        super().__init__(f"Verify: {head} result failed verification ({detail})")
        self.head, self.detail = head, detail


def _fail(head, detail):
    raise VerificationFailure(head, detail)


def _failed(value, exc: VerificationFailure):
    return Expr("Failed", value, _assoc(Head=Symbol(exc.head), Reason=exc.detail))


class _Inconclusive(Exception):
    pass


# ─────────────────────────── numeric sampling ───────────────────────────

def _free(e):
    """Free symbols, treating C[1]-style constants (undefined function applications) as symbols."""
    syms = set(e.free_symbols)
    for f in e.atoms(sp.core.function.AppliedUndef):
        if not f.free_symbols:
            syms.add(f)
    return syms


def _sample_points(syms, n=6, seed=1):
    rng = random.Random(seed)
    pts = []
    for _ in range(n):
        pts.append({s: sp.Rational(rng.randint(3, 23), 10) + sp.Rational(rng.randint(1, 97), 1000) for s in syms})
    return pts


def _num(e, subs):
    try:
        v = sp.N(e.subs(subs), SAMPLE_DPS)
    except Exception:
        return None
    if not v.is_number or v.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
        return None
    try:
        return mpmath.mpc(complex(v)) if not v.is_real else mpmath.mpf(str(v))
    except (TypeError, ValueError):
        return None


def _numerically_zero(d, scale_exprs=(), syms=None):
    """True if d vanishes at random sample points; False if clearly not; raises _Inconclusive otherwise."""
    syms = sorted(_free(d) if syms is None else syms, key=str)
    used = 0
    for subs in _sample_points(syms):
        v = _num(d, subs)
        if v is None:
            continue
        scale = mpmath.mpf(1)
        for s in scale_exprs:
            w = _num(s, subs)
            if w is not None:
                scale = max(scale, abs(w))
        if abs(v) > NUMERIC_TOL * scale:
            return False, subs, v
        used += 1
    if used < 3:
        raise _Inconclusive("too few sample points where the expression is defined")
    return True, None, None


def _zero(d, head, what, scale_exprs=()):
    """('symbolic'|'numeric') when d == 0; raises the verification error when it is not."""
    d = sp.sympify(d)
    if d == 0 or sp.expand(d) == 0:
        return "exact"
    try:
        if sp.simplify(d) == 0:
            return "symbolic"
    except Exception:
        pass
    ok, subs, v = _numerically_zero(d, scale_exprs)
    if not ok:
        at = ", ".join(f"{k} = {sp.N(val, 6)}" for k, val in subs.items())
        _fail(head, f"{what} is {mpmath.nstr(v, 6)}" + (f" at {at}" if at else ""))
    return "numeric"


def _level(kinds, value, method, exact_ok=True, **info):
    """Certified when every step was exact (and exact_ok), else Checked with the weakest evidence named."""
    if exact_ok and all(k == "exact" for k in kinds):
        return _certified(value, method, **info)
    evidence = "numerical sampling" if "numeric" in kinds else "symbolic simplification"
    return _checked(value, method, Evidence=evidence, **info)


# ─────────────────────────── checkers ───────────────────────────

def _var(x):
    if isinstance(x, Symbol):
        return sp.Symbol(x.name)
    raise _Inconclusive("the variable is not a symbol")


def _check_integrate(args, value):
    if len(args) != 2:
        raise _Inconclusive("only Integrate[f, x] and Integrate[f, {x, a, b}] are checked")
    f, spec = S(args[0]), args[1]
    F = S(value)
    if isinstance(F, sp.Basic) and F.has(sp.Integral):
        raise _Inconclusive("the integral was not evaluated")
    if isinstance(spec, Symbol):
        x = _var(spec)
        k = _zero(sp.diff(F, x) - f, "Integrate", "D[result, x] - integrand", (f,))
        return _level([k], value, "Differentiation", exact_ok=False, Check=Expr("HoldForm", Expr("Equal", Expr("D", value, spec), args[0])))
    if is_list(spec) and len(spec.args) == 3 and isinstance(spec.args[0], Symbol):
        x = _var(spec.args[0])
        a, b = S(spec.args[1]), S(spec.args[2])
        if _free(f) - {x} or _free(sp.sympify(a)) or _free(sp.sympify(b)):
            raise _Inconclusive("symbolic parameters in a definite integral")
        g = sp.lambdify(x, f, "mpmath")
        with mpmath.workdps(30):
            q = mpmath.quad(g, [mpmath.mpmathify(sp.N(a, 30)), mpmath.mpmathify(sp.N(b, 30))])
            v = mpmath.mpmathify(complex(sp.N(F, 30))) if not sp.sympify(F).is_real else mpmath.mpf(str(sp.N(F, 30)))
            if abs(q - v) > mpmath.mpf(10) ** -10 * max(1, abs(q)):
                _fail("Integrate", f"numerical quadrature gives {mpmath.nstr(q, 12)}, result is {mpmath.nstr(v, 12)}")
        return _checked(value, "Numerical quadrature (mpmath.quad)", Evidence="numerical")
    raise _Inconclusive("this integration form is not checked")


def _equations(e):
    if isinstance(e, Expr) and e.head in ("List", "And"):
        out = []
        for a in e.args:
            out.extend(_equations(a))
        return out
    if isinstance(e, Expr) and e.head == "Equal" and len(e.args) == 2:
        return [S(e.args[0]) - S(e.args[1])]
    raise _Inconclusive("only equations (==) are checked")


def _solutions(value):
    if not is_list(value):
        raise _Inconclusive("the result is not a list of solutions")
    sols = []
    for s in value.args:
        if not is_list(s) or not all(isinstance(r, Expr) and r.head == "Rule" for r in s.args):
            raise _Inconclusive("a solution is not a list of rules")
        sols.append({S(r.args[0]): S(r.args[1]) for r in s.args})
    return sols


def _check_solve(args, value, head="Solve"):
    if not args:
        raise _Inconclusive("no equations")
    eqs = _equations(args[0])
    sols = _solutions(value)
    if not sols:
        raise _Inconclusive("no solutions to substitute (completeness is not checked)")
    kinds = []
    for sol in sols:
        for d in eqs:
            kinds.append(_zero(d.subs(sol, simultaneous=True), head, f"residual for {sol}"))
    inexact = any(isinstance(a, sp.Float) for sol in sols for v in sol.values() for a in sp.preorder_traversal(v))
    method = "Substitution of every solution"
    if inexact:
        return _checked(value, method, Evidence="numerical residuals", Solutions=len(sols),
                        Completeness="not checked")
    return _level(kinds, value, method, Solutions=len(sols), Completeness="not checked")


def _check_dsolve(args, value):
    if len(args) != 3:
        raise _Inconclusive("only DSolve[eq, y, x] is checked")
    eq = args[0]
    if is_list(eq):
        eqs = [a for a in eq.args if not (isinstance(a, Expr) and a.head == "Equal" and _is_condition(a))]
        if len(eqs) != 1:
            raise _Inconclusive("systems and initial conditions are not checked")
        eq = eqs[0]
    if not (isinstance(eq, Expr) and eq.head == "Equal"):
        raise _Inconclusive("the equation is not an Equal")
    ode = S(eq.args[0]) - S(eq.args[1])
    sols = _solutions(value)
    x = _var(args[2])
    kinds = []
    for sol in sols:
        (lhs, rhs), = sol.items() if len(sol) == 1 else (None, None)
        if lhs is None:
            raise _Inconclusive("only single-function solutions are checked")
        if isinstance(lhs, sp.Symbol):                     # y -> Function[{x}, ...]
            raise _Inconclusive("pure-function solutions are not checked; use DSolve[eq, y[x], x]")
        fn_ = lhs.func
        res = ode.subs(lhs, rhs).doit()
        res = res.replace(lambda e: isinstance(e, sp.Derivative) and e.expr.func == fn_, lambda e: e)
        if res.has(fn_):
            res = ode.replace(fn_, sp.Lambda(x, rhs)).doit()
        kinds.append(_zero(res, "DSolve", "the residual of the equation"))
    return _level(kinds, value, "Substitution into the differential equation", exact_ok=False,
                  Solutions=len(sols))


def _is_condition(e):
    lhs = e.args[0]
    return isinstance(lhs, Expr) and not any(isinstance(a, Symbol) for a in lhs.args)


def _check_d(args, value):
    if len(args) != 2 or not isinstance(args[1], Symbol):
        raise _Inconclusive("only D[f, x] is checked")
    f, x, d = S(args[0]), _var(args[1]), S(value)
    syms = sorted(_free(f) | _free(d), key=str)
    used = 0
    for subs in _sample_points(syms):
        try:
            g = sp.lambdify(x, f.subs({k: v for k, v in subs.items() if k != x}), "mpmath")
            with mpmath.workdps(SAMPLE_DPS):
                num = mpmath.diff(g, mpmath.mpf(str(subs.get(x, sp.Rational(7, 10)))))
        except Exception:
            continue
        v = _num(d, subs)
        if v is None:
            continue
        if abs(num - v) > mpmath.mpf(10) ** -15 * max(1, abs(v)):
            _fail("D", f"numerical derivative {mpmath.nstr(num, 10)} differs from the result {mpmath.nstr(v, 10)}")
        used += 1
    if used < 3:
        raise _Inconclusive("too few sample points")
    return _checked(value, "Numerical differentiation (mpmath.diff)", Evidence="numerical sampling")


def _check_limit(args, value):
    if len(args) != 1 and not (len(args) == 2):
        raise _Inconclusive("only Limit[f, x -> a] is checked")
    r = args[1] if len(args) == 2 else None
    if not (isinstance(r, Expr) and r.head == "Rule" and isinstance(r.args[0], Symbol)):
        raise _Inconclusive("only Limit[f, x -> a] is checked")
    f, x, a, L = S(args[0]), _var(r.args[0]), S(r.args[1]), S(value)
    if _free(f) - {x} or not sp.sympify(a).is_number or not sp.sympify(L).is_number:
        raise _Inconclusive("symbolic parameters")
    if L.has(sp.oo, -sp.oo, sp.zoo) or a.has(sp.zoo):
        raise _Inconclusive("infinite limits are not checked")
    with mpmath.workdps(60):
        g = sp.lambdify(x, f, "mpmath")
        Lv = mpmath.mpmathify(complex(sp.N(L, 60)))
        errs = []
        for k in (10, 14, 18):
            h = mpmath.mpf(10) ** -k
            if a in (sp.oo, -sp.oo):
                pts = [mpmath.mpf(10) ** k * (1 if a == sp.oo else -1)]
            else:
                a_ = mpmath.mpmathify(complex(sp.N(a, 60)))
                pts = [a_ + h, a_ - h]
            try:
                errs.append(max(abs(g(p) - Lv) for p in pts))
            except (ZeroDivisionError, ValueError, TypeError):
                raise _Inconclusive("the function cannot be evaluated near the point")
        if errs[-1] > mpmath.mpf(10) ** -6 * max(1, abs(Lv)):
            _fail("Limit", f"f near the point differs from the limit by {mpmath.nstr(errs[-1], 6)}")
    return _checked(value, "Numerical approach to the point", Evidence="numerical")


def _check_series(args, value):
    if len(args) != 2 or not (is_list(args[1]) and len(args[1].args) == 3):
        raise _Inconclusive("only Series[f, {x, a, n}] is checked")
    x, a, n = _var(args[1].args[0]), S(args[1].args[1]), args[1].args[2]
    f = S(args[0])
    P = S(value)
    P = P.removeO() if hasattr(P, "removeO") else P
    if not isinstance(n, int) or _free(f) - {x} or not sp.sympify(a).is_number or a.has(sp.oo):
        raise _Inconclusive("symbolic parameters or expansion point")
    with mpmath.workdps(60):
        g, p = sp.lambdify(x, f, "mpmath"), sp.lambdify(x, P, "mpmath")
        a_ = mpmath.mpmathify(complex(sp.N(a, 60))) if not a.is_real else mpmath.mpf(str(sp.N(a, 60)))
        ratios = []
        for k in (3, 4, 5):
            h = mpmath.mpf(10) ** -k
            try:
                ratios.append(abs(g(a_ + h) - p(a_ + h)) / h ** n)
            except (ZeroDivisionError, ValueError, TypeError):
                raise _Inconclusive("the function cannot be evaluated near the point")
        # the remainder is O(h^(n+1)), so |f - P| / h^n must shrink roughly by 10 per step
        if not (ratios[-1] < mpmath.mpf(10) ** -3 * max(1, ratios[0] * 100) and ratios[-1] < 1e-3):
            _fail("Series", "the truncation error does not vanish to the stated order")
    return _checked(value, "Remainder estimate at points near the expansion point", Evidence="numerical")


# integer certificates -------------------------------------------------------

_DET_MR_BOUND = 3317044064679887385961981   # deterministic Miller-Rabin with the first 13 prime bases


def _mr_witness(n, a):
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    x = pow(a, d, n)
    if x in (1, n - 1):
        return False
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return False
    return True


def _det_mr(n):
    return all(not _mr_witness(n, a) for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41) if a % n)


def _pratt(n, depth=0):
    """A Pratt certificate (n, a, [(q, cert(q)), ...]) proving n prime, or None if n - 1 can't be factored fast."""
    if n < 2:
        return None
    if n == 2:
        return Expr("List", 2)
    fac = sp.factorint(n - 1, limit=10 ** 6)
    qs = sorted(fac)
    m = n - 1
    for q, e in fac.items():
        m //= q ** e
    if m != 1 and not sp.isprime(m):
        return None
    if m != 1:
        qs.append(m)
        qs = sorted(set(qs))
    for a in range(2, 200):
        if pow(a, n - 1, n) != 1:
            return None                                  # n is composite (Fermat witness)
        if all(pow(a, (n - 1) // q, n) != 1 for q in qs):
            subs = []
            for q in qs:
                c = Expr("List", q) if q < _DET_MR_BOUND and _det_mr(q) else _pratt(q, depth + 1)
                if c is None:
                    return None
                subs.append(c)
            return Expr("List", n, a, Expr("List", *subs))
    return None


def _prime_proof(p):
    """(method, certificate) proving p prime, or None."""
    if p < 2:
        return None
    if p < _DET_MR_BOUND:
        return ("deterministic Miller-Rabin (13 prime bases, n < 3.3*10^24)", p) if _det_mr(p) else None
    c = _pratt(p)
    return ("Pratt certificate", c) if c is not None else None


def _composite_witness(n):
    if n < 4:
        return None
    if n % 2 == 0:
        return Expr("Rule", "Factor", 2)
    for a in range(2, min(n - 1, 200)):
        if _mr_witness(n, a):
            return Expr("Rule", "MillerRabinWitness", a)
    return None


def _check_factorinteger(args, value):
    if len(args) != 1 or not isinstance(args[0], int) or isinstance(args[0], bool):
        raise _Inconclusive("only FactorInteger[n] for an integer n is checked")
    n = args[0]
    pairs = [(int(S(p.args[0])), int(p.args[1])) for p in value.args]
    prod = 1
    for p, e in pairs:
        prod *= p ** e
    if prod != n:
        _fail("FactorInteger", f"the product of the factors is {prod}, not {n}")
    proofs, probable = [], False
    for p, _e in pairs:
        if p in (-1, 0, 1):
            continue
        pr = _prime_proof(p)
        if pr is None:
            if sp.isprime(p):
                probable = True
                proofs.append(Expr("Rule", p, "BPSW probable prime"))
                continue
            _fail("FactorInteger", f"{p} is not prime")
        proofs.append(Expr("Rule", p, pr[0] if pr[0].startswith("deterministic") else pr[1]))
    info = dict(Certificate=Expr("List", *proofs))
    if probable:
        return _checked(value, "Product of the factors; primality of each factor", Evidence="probable primes", **info)
    return _certified(value, "Product of the factors; primality proof of each factor", **info)


def _check_primeq(args, value):
    if len(args) != 1 or not isinstance(args[0], int) or isinstance(args[0], bool) or value not in (True, False):
        raise _Inconclusive("only PrimeQ[n] for an integer n is checked")
    n = abs(args[0])
    if value is True:
        pr = _prime_proof(n)
        if pr is None:
            if sp.isprime(n):
                return _checked(value, "BPSW", Evidence="probable prime")
            _fail("PrimeQ", f"{n} has a Fermat or Miller-Rabin witness")
        return _certified(value, pr[0], Certificate=pr[1])
    if n < 2:
        return _certified(value, "definition (n < 2)")
    w = _composite_witness(n)
    if w is None:
        if sp.isprime(n):
            _fail("PrimeQ", f"{n} is prime")
        raise _Inconclusive("no small compositeness witness found")
    return _certified(value, "compositeness witness", Certificate=w)


def _check_gcd(args, value):
    if len(args) != 2 or not all(isinstance(a, int) and not isinstance(a, bool) for a in args):
        raise _Inconclusive("only GCD[a, b] for two integers is checked")
    a, b, g = args[0], args[1], int(value)
    s, t, _h = sp.gcdex(a, b) if (a or b) else (0, 0, 0)
    s, t = int(s), int(t)
    if _h < 0:
        s, t = -s, -t
    if g < 0 or (g and (a % g or b % g)) or (s * a + t * b) != g:
        _fail("GCD", f"{g} does not divide both arguments or is not a combination of them")
    return _certified(value, "Bezout identity", Certificate=Expr("List", s, t))


def _check_factor(args, value):
    if len(args) != 1:
        raise _Inconclusive("only Factor[p] is checked")
    d = sp.expand(S(value) - S(args[0]))
    if d != 0:
        k = _zero(d, "Factor", "Expand[result - input]")
        return _level([k], value, "Expansion of the product")
    return _certified(value, "Expansion of the product", Note="irreducibility of the factors is not checked")


def _check_equivalent(head):
    def check(args, value):
        if len(args) < 1:
            raise _Inconclusive("no input")
        k = _zero(S(value) - S(args[0]), head, "result - input", (S(args[0]),))
        return _level([k], value, "Equivalence with the input", exact_ok=True)
    return check


def _matrix(x):
    if not is_list(x) or not x.args or not all(is_list(r) for r in x.args):
        raise _Inconclusive("not a matrix")
    return sp.Matrix([[S(v) for v in r.args] for r in x.args])


def _has_float(M):
    return any(isinstance(a, sp.Float) for v in M for a in sp.preorder_traversal(v))


def _check_inverse(args, value):
    if len(args) != 1:
        raise _Inconclusive("only Inverse[m] is checked")
    M, V = _matrix(args[0]), _matrix(value)
    P = (M * V).applyfunc(sp.simplify)
    I_ = sp.eye(M.rows)
    if _has_float(M) or _has_float(V):
        err = max(abs(complex(sp.N(v))) for v in (P - I_))
        if err > 1e-9:
            _fail("Inverse", f"m . result differs from the identity by {err:.3g}")
        return _checked(value, "m . result == IdentityMatrix", Evidence="floating point")
    k = [_zero(v, "Inverse", "an entry of m . result - IdentityMatrix") for v in (M * V - I_)]
    return _level(k, value, "m . result == IdentityMatrix")


def _check_linearsolve(args, value):
    if len(args) != 2:
        raise _Inconclusive("only LinearSolve[m, b] is checked")
    M, b = _matrix(args[0]), args[1]
    if not is_list(b) or not is_list(value):
        raise _Inconclusive("only vector right-hand sides are checked")
    bv = sp.Matrix([S(v) for v in b.args])
    xv = sp.Matrix([S(v) for v in value.args])
    D = M * xv - bv
    if _has_float(M) or _has_float(xv):
        err = max(abs(complex(sp.N(v))) for v in D)
        if err > 1e-9:
            _fail("LinearSolve", f"m . x - b is {err:.3g}")
        return _checked(value, "m . x == b", Evidence="floating point")
    k = [_zero(v, "LinearSolve", "an entry of m . x - b") for v in D]
    return _level(k, value, "m . x == b")


def _check_det(args, value):
    if len(args) != 1:
        raise _Inconclusive("only Det[m] is checked")
    M = _matrix(args[0])
    if _has_float(M):
        import numpy as np
        ref = np.linalg.det(np.array(M.evalf(), dtype=complex))
        v = complex(sp.N(S(value)))
        if abs(ref - v) > 1e-9 * max(1, abs(ref)):
            _fail("Det", f"NumPy gives {ref}")
        return _checked(value, "numpy.linalg.det", Evidence="floating point")
    k = _zero(M.det(method="berkowitz") - S(value), "Det", "the Berkowitz determinant minus the result")
    return _level([k], value, "Second algorithm (Berkowitz)")


def _check_eigenvalues(args, value):
    if len(args) != 1 or not is_list(value):
        raise _Inconclusive("only Eigenvalues[m] is checked")
    M = _matrix(args[0])
    lam = sp.Symbol("lambda_")
    cp = (M - lam * sp.eye(M.rows)).det(method="berkowitz")
    kinds = []
    for v in value.args:
        kinds.append(_zero(cp.subs(lam, S(v)), "Eigenvalues", f"the characteristic polynomial at {v}",
                           (S(v) ** M.rows,)))
    if len(value.args) != M.rows:
        _fail("Eigenvalues", f"{len(value.args)} eigenvalues for a {M.rows} x {M.rows} matrix")
    return _level(kinds, value, "Each eigenvalue is a root of the characteristic polynomial", exact_ok=False,
                  Note="multiplicities are checked by count only")


CHECKERS = {
    "Integrate": _check_integrate, "Solve": _check_solve, "NSolve": lambda a, v: _check_solve(a, v, "NSolve"),
    "DSolve": _check_dsolve, "D": _check_d, "Limit": _check_limit, "Series": _check_series,
    "FactorInteger": _check_factorinteger, "PrimeQ": _check_primeq, "GCD": _check_gcd, "Factor": _check_factor,
    "Simplify": _check_equivalent("Simplify"), "FullSimplify": _check_equivalent("FullSimplify"),
    "Expand": _check_equivalent("Expand"), "Together": _check_equivalent("Together"),
    "Apart": _check_equivalent("Apart"), "TrigExpand": _check_equivalent("TrigExpand"),
    "Inverse": _check_inverse, "LinearSolve": _check_linearsolve, "Det": _check_det,
    "Eigenvalues": _check_eigenvalues,
}


@fn("Verify", area=AREA, status="added", hold=True,
    notes="Verify[expr] evaluates expr and checks the result independently: Integrate by differentiation (or "
          "mpmath quadrature for definite integrals), Solve/NSolve by substituting every solution (completeness "
          "is not checked), DSolve by substitution, D by numerical differentiation, Limit and Series numerically, "
          "FactorInteger and PrimeQ with primality proofs (deterministic Miller-Rabin below 3.3*10^24, Pratt "
          "certificates above) or compositeness witnesses, GCD by a Bezout identity, Factor by expansion, "
          "Simplify/FullSimplify/Expand/Together/Apart/TrigExpand by equivalence with the input, Inverse and "
          "LinearSolve by multiplying back, Det by a second algorithm and Eigenvalues through the characteristic "
          "polynomial. It gives Certified[value, info] when an exact certificate was checked, Checked[value, "
          "info] when the evidence is symbolic simplification or numerical sampling, and Unchecked[value, "
          "reason] when no checker applies; a failed check gives Failed[value, <|\"Head\" -> h, \"Reason\" -> "
          "text|>] (errors raised by the computation itself propagate). MikoshiLang extension. "
          "Check: self-check; seeded wrong results are caught in tests.")
def _verify(expr: Held) -> Expr:
    from .wl_common import ev
    if not isinstance(expr, Expr) or not isinstance(expr.head, str):
        return _unchecked(ev(expr), "no checker for this expression")
    head = expr.head
    checker = CHECKERS.get(head)
    if checker is None:
        return _unchecked(ev(expr), f"no checker for {head}")
    args = [ev(a) for a in expr.args]
    value = ev(Expr(head, *args))
    if isinstance(value, Expr) and value.head == head:
        return _unchecked(value, f"{head} did not evaluate")
    try:
        return checker(args, value)
    except _Inconclusive as exc:
        return _unchecked(value, f"{head}: {exc}")
    except VerificationFailure as exc:
        return _failed(value, exc)


def _hold_verify():
    from ..evaluate import set_hold_all
    set_hold_all("Verify")


_hold_verify()
