"""Wolfram batch W09 -- algebraic numbers and algebraic number fields.

AlgebraicNumber[theta, {c0, ..., c(n-1)}] represents c0 + c1 theta + ... in the field Q(theta).  As in the
Wolfram Language the generator is normalized to an algebraic integer (theta -> a theta for the leading
coefficient a of its minimal polynomial), kept as a radical for degree <= 2 and as a Root object otherwise, and the
coefficient list is reduced modulo the minimal polynomial and padded to the degree.  Number-field invariants use
SymPy's number-field machinery (Round Two integral bases); class numbers, units and regulators are computed for
quadratic fields only (other degrees raise).
"""
from __future__ import annotations

import math
from functools import lru_cache

import sympy as sp
from sympy.polys.numberfields import to_number_field, primitive_element
from sympy.polys.polyerrors import IsomorphismFailed, NotAlgebraic

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, record, error, Unevaluated, S, R, L, is_list, has_float, is_numeric_value
from .wl_common import ev, split_options
from . import wl_numbertheory2 as NT

AREA = "algebraic number theory (W09)"
ADDED = dict(area=AREA, status="added")
UNH = K.UNHANDLED
X = sp.Symbol("_w09x")
NT.SPECIAL_HEADS.add("AlgebraicNumber")


def _intq(x):
    return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, sp.Integer)


def is_an(x):
    return isinstance(x, Expr) and x.head == "AlgebraicNumber" and len(x.args) == 2 and is_list(x.args[1])


def _rational(x):
    """SymPy Rational for an exact rational Mikoshi value, else None."""
    if isinstance(x, bool):
        return None
    if _intq(x):
        return sp.Integer(int(x))
    if isinstance(x, Expr) and x.head in ("Rational", "Times", "Power") and not has_float(x):
        try:
            v = S(x)
        except Exception:
            return None
        return v if v.is_Rational else None
    if isinstance(x, sp.Rational):
        return x
    return None


def an_parts(a):
    """(theta (SymPy), [coeffs]) of an AlgebraicNumber expression."""
    th = alg_value(a.args[0])
    cs = []
    for c in a.args[1].args:
        r = _rational(c)
        if r is None:
            raise ValueError("AlgebraicNumber: the coefficients must be rational numbers")
        cs.append(r)
    return th, cs


def alg_value(x):
    """SymPy value of an exact algebraic Mikoshi number (AlgebraicNumber objects expanded), or None."""
    if is_an(x):
        th, cs = an_parts(x)
        if th is None:
            return None
        return sp.Add(*[c * th ** k for k, c in enumerate(cs)])
    if isinstance(x, bool) or isinstance(x, str):
        return None
    if isinstance(x, (float, complex)) or has_float(x):
        return None
    if isinstance(x, Expr) and _contains_an(x):
        # e.g. Plus[AlgebraicNumber[...], Sqrt[3]]
        try:
            parts = [alg_value(a) for a in x.args]
        except ValueError:
            return None
        if any(p is None for p in parts):
            return None
        try:
            return S(Expr(x.head, *[R(p) for p in parts]))
        except Exception:
            return None
    if not is_numeric_value(x) and not isinstance(x, (int, sp.Basic)):
        return None
    try:
        v = S(x)
    except Exception:
        return None
    if not isinstance(v, sp.Basic) or not v.is_number:
        return None
    if v.is_algebraic is not True:
        if v.is_algebraic is False:
            return None
        try:
            minpoly(v)
        except Exception:
            return None
    return v


def _contains_an(x):
    if is_an(x):
        return True
    return isinstance(x, Expr) and any(_contains_an(a) for a in x.args)


@lru_cache(maxsize=4096)
def _minpoly_cached(v):
    p = sp.minimal_polynomial(v, X, polys=True)
    p = sp.Poly(p, X, domain=sp.ZZ) if p.domain != sp.ZZ else p
    c, prim = p.primitive()
    if prim.LC() < 0:
        prim = -prim
    return prim


def minpoly(v):
    """Primitive integer minimal polynomial of the algebraic SymPy number v (positive leading coefficient)."""
    try:
        return _minpoly_cached(v)
    except (NotAlgebraic, NotImplementedError) as e:
        raise ValueError(f"{v} is not an algebraic number") from e


def _num(v, dps=60):
    return complex(sp.N(v, dps))


def root_index(P, v):
    """Index i with CRootOf(P, i) == v (P irreducible)."""
    roots = P.all_roots()
    if len(roots) == 1:
        return 0
    z = sp.N(v, 50)
    best, bi = None, None
    for i, r in enumerate(roots):
        d = abs(sp.N(r, 50) - z)
        if best is None or d < best:
            best, bi = d, i
    return bi


def _radical_of(P, v):
    """Radical form of the root v of the (degree <= 2) polynomial P."""
    if P.degree() == 1:
        a, b = P.all_coeffs()
        return sp.Rational(-b, a)
    a, b, c = P.all_coeffs()
    disc = b * b - 4 * a * c
    cands = [(-b + sp.sqrt(disc)) / (2 * a), (-b - sp.sqrt(disc)) / (2 * a)]
    z = sp.N(v, 50)
    r = min(cands, key=lambda t: abs(sp.N(t, 50) - z))
    return sp.expand(r)


def generator(v):
    """(theta', monic integer minpoly of theta', scale a) with theta' = a v an algebraic integer."""
    P0 = minpoly(v)
    a = int(P0.LC())
    n = P0.degree()
    cs = P0.all_coeffs()            # a_n ... a_0
    mon = [sp.Integer(1)] + [sp.Integer(int(cs[k]) * a ** (k - 1)) for k in range(1, n + 1)]
    P = sp.Poly(mon, X)
    if n <= 2:
        th = _radical_of(P, a * v)
    else:
        th = sp.CRootOf(P, root_index(P, a * v))
    return th, P, a


def field_rep(v):
    """Canonical generator data for the field Q(v)."""
    th, P, a = generator(v)
    return th, P


def _reduce(cs, P):
    """Coefficients (low to high) of sum cs[k] X^k reduced modulo P, padded to deg P."""
    n = P.degree()
    poly = sp.Poly(list(reversed(cs)) if cs else [0], X, domain=sp.QQ)
    r = poly.rem(sp.Poly(P.as_expr(), X, domain=sp.QQ))
    out = list(reversed(r.all_coeffs())) if not r.is_zero else [sp.Integer(0)]
    out = [sp.Rational(c) for c in out]
    return out + [sp.Integer(0)] * (n - len(out))


def make_an(th_expr, P, cs):
    """AlgebraicNumber[theta, cs] for a canonical generator (rational when deg P == 1)."""
    cs = _reduce(cs, P)
    if P.degree() == 1:
        return R(cs[0])
    th = th_expr if isinstance(th_expr, (Expr, Symbol, int)) and not isinstance(th_expr, sp.Basic) else R(th_expr)
    return Expr("AlgebraicNumber", th, L(*[R(c) for c in cs]))


def make_an_or_rational(th_expr, P, cs):
    cs = _reduce(cs, P)
    if all(c == 0 for c in cs[1:]):
        return R(cs[0])
    return make_an(th_expr, P, cs)


def canonical_an(th_v, cs):
    """AlgebraicNumber for theta value th_v (SymPy) and rational coefficients cs."""
    th, P, a = generator(th_v)
    scaled = [c / sp.Integer(a) ** k for k, c in enumerate(cs)]
    if P.degree() == 1:
        return R(sp.Add(*[c * th_v ** k for k, c in enumerate(cs)]))
    return make_an(th, P, scaled)


def _an_numeric(th_num, cs):
    tot = 0
    for k, c in enumerate(cs):
        tot = tot + c * th_num ** k
    return tot


@fn("AlgebraicNumber", **ADDED, notes="AlgebraicNumber[theta, {c0, c1, ...}] = c0 + c1 theta + ... in Q(theta): "
    "the generator becomes an algebraic integer (a theta for the leading minimal-polynomial coefficient a, "
    "coefficients rescaled), a radical for degree <= 2 or a Root object otherwise, with the coefficients reduced "
    "modulo the minimal polynomial and padded; a rational theta gives the number; an approximate theta gives the "
    "numeric value; arithmetic (+, *, integer powers, rationals) between numbers with the same generator stays "
    "in the field; N gives the numeric value")
def _algebraicnumber(th: Expr, cs: Expr) -> Expr:
    if not is_list(cs) or not cs.args:
        raise ValueError("AlgebraicNumber: the second argument must be a non-empty list of rational numbers")
    rats = [_rational(c) for c in cs.args]
    if any(r is None for r in rats):
        if all(is_numeric_value(c) for c in cs.args):
            raise ValueError("AlgebraicNumber: the coefficients must be rational numbers")
        raise Unevaluated()
    if has_float(th) and is_numeric_value(th):
        z = S(th)
        return R(sp.expand(_an_numeric(z, rats))) if not isinstance(th, (float, complex)) else \
            _an_numeric(th, [float(r) for r in rats])
    v = alg_value(th)
    if v is None:
        raise Unevaluated()
    out = canonical_an(v, rats)
    if is_an(out):
        th2 = ev(out.args[0])          # e.g. Root[-3 + #^5 &, 1] evaluates to 3^(1/5)
        if K.same(th2, th) and K.same(out.args[1], cs):
            raise Unevaluated()
        return Expr("AlgebraicNumber", th2, out.args[1])
    return out


def _mp_an(th, cs):
    vals = []
    if not is_list(cs):
        raise Unevaluated()
    for c in cs.args:
        vals.append(S(c))
    z = S(th)
    r = sp.expand(_an_numeric(z, vals))
    return r


def _register_numeric():
    from .wl_special import MP_NUMERIC
    MP_NUMERIC["AlgebraicNumber"] = _mp_an


_register_numeric()


# ── arithmetic in Q(theta) ──

def _an_key(a):
    return a.args[0]


def _an_data(a):
    th = alg_value(a.args[0])
    P = minpoly(th)
    cs = [_rational(c) for c in a.args[1].args]
    return th, P, cs


def _mul(c1, c2, P):
    p1 = sp.Poly(list(reversed(c1)), X, domain=sp.QQ)
    p2 = sp.Poly(list(reversed(c2)), X, domain=sp.QQ)
    return _reduce(list(reversed((p1 * p2).all_coeffs())), P)


def _inv(c, P):
    p = sp.Poly(list(reversed(c)), X, domain=sp.QQ)
    if p.is_zero:
        raise ZeroDivisionError("AlgebraicNumber: division by zero")
    s, t, h = sp.Poly(P.as_expr(), X, domain=sp.QQ).gcdex(p)
    # s P + t p = h (constant)
    inv = t.quo(h)
    return _reduce(list(reversed(inv.all_coeffs())), P)


def _an_arith(expr, ev_):
    h = expr.head
    if h not in ("Plus", "Times", "Power"):
        return UNH
    args = list(expr.args)
    if h == "Power":
        if len(args) != 2 or not is_an(args[0]) or not _intq(args[1]):
            return UNH
        th, P, cs = _an_data(args[0])
        e = int(args[1])
        base = cs if e >= 0 else _inv(cs, P)
        res = [sp.Integer(1)] + [sp.Integer(0)] * (P.degree() - 1)
        for _ in range(abs(e)):
            res = _mul(res, base, P)
        return make_an(args[0].args[0], P, res) if True else UNH
    ans = [a for a in args if is_an(a)]
    if not ans:
        return UNH
    key = _an_key(ans[0])
    same = [a for a in ans if K.same(_an_key(a), key)]
    rats = [a for a in args if not is_an(a) and _rational(a) is not None]
    items = len(same) + len(rats)
    if items < 2:
        return UNH
    th, P, _ = _an_data(same[0])
    n = P.degree()
    if h == "Plus":
        acc = [sp.Integer(0)] * n
        for a in same:
            cs = _reduce([_rational(c) for c in a.args[1].args], P)
            acc = [x + y for x, y in zip(acc, cs)]
        for r in rats:
            acc[0] += _rational(r)
    else:
        acc = [sp.Integer(1)] + [sp.Integer(0)] * (n - 1)
        for a in same:
            acc = _mul(acc, [_rational(c) for c in a.args[1].args], P)
        for r in rats:
            acc = [x * _rational(r) for x in acc]
    combined = make_an(key, P, acc)
    rest = [a for a in args if not (is_an(a) and K.same(_an_key(a), key)) and not (not is_an(a) and
                                                                               _rational(a) is not None)]
    if not rest:
        return combined
    return Expr(h, combined, *rest)


NT.ARITH_HANDLERS.append(_an_arith)


# ─────────────────────────── functions of algebraic numbers ───────────────────────────

def _field_of(a, fname):
    """(theta_canonical, P) of the field associated with a: Q(theta) for AlgebraicNumber, else Q(a)."""
    if is_an(a):
        th = alg_value(a.args[0])
        return field_rep(th)
    v = alg_value(a)
    if v is None:
        raise Unevaluated()
    return field_rep(v)


@fn("AlgebraicNumberPolynomial", **ADDED, listable=True, notes="AlgebraicNumberPolynomial[AlgebraicNumber[theta, "
    "{c0, ..., cn}], x] = c0 + c1 x + ... + cn x^n; a rational number gives itself")
def _anpoly(a: Expr, x: Expr) -> Expr:
    if is_an(a):
        return ev(Expr("Plus", *[Expr("Times", c, Expr("Power", x, k)) for k, c in enumerate(a.args[1].args)]))
    if _rational(a) is not None:
        return a
    raise Unevaluated()


def _norm_trace(a, fname, opts, which):
    ext = opts.get("Extension")
    v = alg_value(a)
    if v is None:
        if is_numeric_value(a) or isinstance(a, (float, complex)):
            raise ValueError(f"{fname}: {a} is not an exact algebraic number")
        raise Unevaluated()
    P = minpoly(v)
    d = P.degree()
    cs = P.all_coeffs()
    if which == "norm":
        base = sp.Rational((-1) ** d * cs[-1], cs[0])
    else:
        base = sp.Rational(-cs[1], cs[0]) if d >= 1 else sp.Integer(0)
    if is_an(a) and ext is None:
        n = minpoly(alg_value(a.args[0])).degree()
    elif ext is not None:
        ev_ext = ev(ext)
        exts = list(ev_ext.args) if is_list(ev_ext) else [ev_ext]
        vals = [alg_value(e) for e in exts]
        if any(t is None for t in vals):
            raise ValueError(f"{fname}: Extension must be a list of algebraic numbers")
        if len(vals) == 1:
            th = vals[0]
        else:
            mp, coeffs = primitive_element(vals, X)
            th = sp.Add(*[c * t for c, t in zip(coeffs, vals)])
        n = minpoly(th).degree()
        try:
            to_number_field(v, th)
        except IsomorphismFailed:
            raise ValueError(f"{fname}: {a} is not in the extension field")
    else:
        n = d
    k = n // d
    return R(base ** k if which == "norm" else base * k)


@fn("AlgebraicNumberNorm", **ADDED, notes="AlgebraicNumberNorm[a]: the norm of the algebraic number a over Q "
    "(product of its conjugates) in Q(a), or in Q(theta) for AlgebraicNumber[theta, ...]; Extension -> b gives "
    "the norm in Q(b) (the minimal-polynomial norm to the power [Q(b):Q(a)])")
def _annorm(a: Expr, *opts: Option) -> Expr:
    pos, o = split_options(opts, ("Extension",), "AlgebraicNumberNorm")
    if pos:
        raise ValueError("AlgebraicNumberNorm takes one argument and the Extension option")
    if is_list(a):
        return L(*[_annorm(x, *opts) for x in a.args])
    return _norm_trace(a, "AlgebraicNumberNorm", o, "norm")


@fn("AlgebraicNumberTrace", **ADDED, notes="AlgebraicNumberTrace[a]: the trace of a over Q (sum of its "
    "conjugates) in Q(a), or in Q(theta) for AlgebraicNumber[theta, ...]; Extension -> b gives the trace in Q(b)")
def _antrace(a: Expr, *opts: Option) -> Expr:
    pos, o = split_options(opts, ("Extension",), "AlgebraicNumberTrace")
    if pos:
        raise ValueError("AlgebraicNumberTrace takes one argument and the Extension option")
    if is_list(a):
        return L(*[_antrace(x, *opts) for x in a.args])
    return _norm_trace(a, "AlgebraicNumberTrace", o, "trace")


def _vp(n, p):
    if n == 0:
        return None
    k = 0
    while n % p == 0:
        n //= p
        k += 1
    return k


@fn("AlgebraicNumberDenominator", **ADDED, listable=True, notes="the smallest positive integer d such that d a "
    "is an algebraic integer")
def _anden(a: Expr) -> Expr:
    v = alg_value(a)
    if v is None:
        if is_numeric_value(a):
            raise ValueError(f"AlgebraicNumberDenominator: {a} is not an exact algebraic number")
        raise Unevaluated()
    P = minpoly(v)
    cs = [int(c) for c in P.all_coeffs()]
    an = cs[0]
    n = len(cs) - 1
    d = 1
    for p, v_ in sp.factorint(an).items():
        e = 0
        for k in range(1, n + 1):
            vk = _vp(cs[k], p)
            if vk is None:
                continue
            need = v_ - vk
            if need > 0:
                e = max(e, -(-need // k))
        d *= p ** e
    return d


def _monic_integer(v):
    P = minpoly(v)
    return P.LC() == 1, P


@fn("AlgebraicIntegerQ", **ADDED, listable=True, notes="True if a is an algebraic integer (its minimal polynomial "
    "is monic with integer coefficients); False for approximate numbers, transcendental numbers and symbols")
def _algintq(a: Expr) -> Boolean:
    v = alg_value(a)
    if v is None:
        return False
    try:
        return bool(_monic_integer(v)[0])
    except ValueError:
        return False


@fn("AlgebraicUnitQ", **ADDED, listable=True, notes="True if a is an algebraic unit: an algebraic integer whose "
    "inverse is also an algebraic integer (monic minimal polynomial with constant term +-1)")
def _algunitq(a: Expr) -> Boolean:
    v = alg_value(a)
    if v is None:
        return False
    try:
        ok, P = _monic_integer(v)
    except ValueError:
        return False
    return bool(ok and abs(P.all_coeffs()[-1]) == 1)


def _cyclotomic_order(P):
    """m when P is the m-th cyclotomic polynomial, else None."""
    n = P.degree()
    if P.LC() != 1:
        return None
    # phi(m) = n has finitely many solutions m <= 2 * n^2 (crude bound) -- search
    for m in range(1, max(3, 2 * n * n + 2) + 1):
        if sp.totient(m) == n and sp.Poly(sp.cyclotomic_poly(m, X), X) == P:
            return m
    return None


@fn("RootOfUnityQ", **ADDED, listable=True, notes="True if a is a root of unity (its minimal polynomial is "
    "cyclotomic); False for other numbers, approximate numbers and symbols")
def _rootofunityq(a: Expr) -> Boolean:
    v = alg_value(a)
    if v is None:
        return False
    try:
        P = minpoly(v)
    except ValueError:
        return False
    return _cyclotomic_order(P) is not None


@fn("Algebraics", area=AREA, status="added", notes="the domain of algebraic numbers (Element[Sqrt[2], Algebraics] "
    "-> True, Element[Pi, Algebraics] -> False)")
def _algebraics(*a: Expr) -> Expr:
    raise Unevaluated()


# ── ToNumberField ──

def _express(v, th):
    """Coefficients (low to high, length deg) of v in Q(th) (th SymPy), or raise ValueError."""
    try:
        an = to_number_field(v, th)
    except IsomorphismFailed:
        raise ValueError(f"ToNumberField: {R(v)} is not in the field generated by {R(th)}")
    return list(reversed([sp.Rational(c) for c in an.coeffs()]))


def _to_nf(v, th):
    cs = _express(v, th)
    return canonical_an(th, cs)


@fn("ToNumberField", **ADDED, notes="ToNumberField[a, theta] expresses the algebraic number a as an "
    "AlgebraicNumber in Q(theta) (raising when a is not in the field); ToNumberField[a] = AlgebraicNumber[a, {0, "
    "1}] in Q(a); ToNumberField[{a1, a2, ...}] expresses all ai in a common field generated by a primitive element "
    "(the primitive element may differ from the Wolfram Language's choice); ToNumberField[list, theta] uses Q("
    "theta); the All form is not supported")
def _tonumberfield(a: Expr, th: Expr = None) -> Expr:
    if isinstance(th, Symbol) and th.name == "All":
        raise NotImplementedError("ToNumberField[a, All] is not supported")
    if th is None:
        if is_list(a):
            vals = [alg_value(x) for x in a.args]
            if any(v is None for v in vals):
                raise Unevaluated()
            if not vals:
                return L()
            if len(vals) == 1:
                return L(_tonumberfield(a.args[0]))
            mp, coeffs = primitive_element(vals, X)
            theta = sp.Add(*[c * t for c, t in zip(coeffs, vals)])
            return L(*[_to_nf(v, theta) for v in vals])
        if is_an(a):
            return a
        v = alg_value(a)
        if v is None:
            raise Unevaluated()
        if minpoly(v).degree() == 1:
            return R(v)
        return canonical_an(v, [sp.Integer(0), sp.Integer(1)])
    tv = alg_value(th)
    if tv is None:
        raise Unevaluated()
    if is_list(a):
        return L(*[_tonumberfield(x, th) for x in a.args])
    v = alg_value(a)
    if v is None:
        raise Unevaluated()
    if minpoly(tv).degree() == 1:
        if minpoly(v).degree() != 1:
            raise ValueError(f"ToNumberField: {a} is not rational")
        return R(v)
    return _to_nf(v, tv)


# ─────────────────────────── number field invariants ───────────────────────────

def _field_listable(f):
    def g(a: Expr, *rest: Expr) -> Expr:
        if is_list(a):
            return L(*[g(x, *rest) for x in a.args])
        return f(a, *rest)
    g.__name__ = f.__name__
    g.__doc__ = f.__doc__
    return g


@lru_cache(maxsize=512)
def _round_two(P_expr):
    from sympy.polys.numberfields.basis import round_two
    P = sp.Poly(P_expr, X)
    ZK, dK = round_two(P)
    M = ZK.matrix.to_Matrix()
    den = ZK.denom
    basis = [[sp.Rational(M[i, j], den) for i in range(M.rows)] for j in range(M.cols)]
    return int(dK), basis


def _field_data(a, fname):
    th, P = _field_of(a, fname)
    return th, P


def _check_field_arg(a, fname):
    if isinstance(a, (float, complex)) or (has_float(a) and is_numeric_value(a)):
        raise ValueError(f"{fname}: {a} is not an exact algebraic number")


@fn("NumberFieldDiscriminant", **ADDED, notes="discriminant of the ring of integers of Q(a) (Q(theta) for "
    "AlgebraicNumber[theta, ...]), computed with the Round Two algorithm; threads over lists")
@_field_listable
def _nfdisc(a):
    _check_field_arg(a, "NumberFieldDiscriminant")
    th, P = _field_data(a, "NumberFieldDiscriminant")
    if P.degree() == 1:
        return 1
    return _round_two(P.as_expr())[0]


@fn("NumberFieldIntegralBasis", **ADDED, notes="an integral basis of the ring of integers of Q(a) (Round Two "
    "algorithm, Hermite-normal-form basis over the power basis of the canonical generator) as AlgebraicNumber "
    "objects (the first element 1); the basis may differ from the Wolfram Language's choice by a unimodular "
    "change; threads over lists")
@_field_listable
def _nfbasis(a):
    _check_field_arg(a, "NumberFieldIntegralBasis")
    th, P = _field_data(a, "NumberFieldIntegralBasis")
    if P.degree() == 1:
        return L(1)
    d, basis = _round_two(P.as_expr())
    return L(*[make_an_or_rational(th, P, cs) for cs in basis])


@fn("NumberFieldSignature", **ADDED, notes="{r1, r2}: the numbers of real and of pairs of complex embeddings of "
    "Q(a); threads over lists")
@_field_listable
def _nfsig(a):
    _check_field_arg(a, "NumberFieldSignature")
    th, P = _field_data(a, "NumberFieldSignature")
    r1 = len(sp.Poly(P.as_expr(), X).real_roots())
    return L(r1, (P.degree() - r1) // 2)


def _quadratic_d(P):
    """Square-free d with Q(theta) = Q(Sqrt[d]) for the quadratic minimal polynomial P."""
    a, b, c = [int(t) for t in P.all_coeffs()]
    disc = b * b - 4 * a * c
    sign = -1 if disc < 0 else 1
    f = sp.factorint(abs(disc))
    d = sign * math.prod(p for p, e in f.items() if e % 2)
    return d


def _fund_unit(d):
    """Fundamental unit > 1 of the ring of integers of Q(Sqrt[d]) (d > 1 square-free), as a SymPy number."""
    sq = sp.sqrt(d)
    # least solution of p^2 - d q^2 = +-1 from the continued fraction of Sqrt[d]
    a0 = math.isqrt(d)
    m, dd, a = 0, 1, a0
    p_prev, p = 1, a0
    q_prev, q = 0, 1
    while p * p - d * q * q not in (1, -1):
        m = dd * a - m
        dd = (d - m * m) // dd
        a = (a0 + m) // dd
        p_prev, p = p, a * p + p_prev
        q_prev, q = q, a * q + q_prev
    eps0 = p + q * sq
    if d % 8 != 5:
        return eps0
    # for d = 5 mod 8 the fundamental unit may be (t + u Sqrt[d])/2 with cube eps0
    digits = len(str(p)) + 30
    e = sp.N(eps0, digits) ** (sp.Rational(1, 3))
    for sgn in (1, -1):
        t = int(sp.Integer(sp.floor(sp.N(e + sgn / e, digits) + sp.Rational(1, 2))))
        u = int(sp.Integer(sp.floor(sp.N((e - sgn / e) / sq, digits) + sp.Rational(1, 2))))
        if t * t - d * u * u in (4, -4) and u > 0:
            cand = (t + u * sq) / 2
            if sp.expand(cand ** 3 - eps0) == 0:
                return sp.expand(cand)
    return eps0


def _real_quadratic(a, fname):
    th, P = _field_data(a, fname)
    n = P.degree()
    if n == 1:
        return th, P, None
    if n != 2:
        raise NotImplementedError(f"{fname}: only quadratic fields are supported (the field has degree {n})")
    return th, P, _quadratic_d(P)


@fn("NumberFieldFundamentalUnits", **ADDED, notes="fundamental units of the ring of integers of Q(a): for a real "
    "quadratic field the fundamental unit > 1 (as an AlgebraicNumber); {} for Q and imaginary quadratic fields; "
    "fields of degree > 2 raise; threads over lists")
@_field_listable
def _nfunits(a):
    _check_field_arg(a, "NumberFieldFundamentalUnits")
    th, P, d = _real_quadratic(a, "NumberFieldFundamentalUnits")
    if d is None or d < 0:
        return L()
    eps = _fund_unit(d)
    cs = _express(eps, th)
    return L(make_an(th, P, cs))


@fn("NumberFieldRegulator", **ADDED, notes="regulator of Q(a): Log of the fundamental unit for real quadratic "
    "fields (exact), 1 for Q and imaginary quadratic fields; fields of degree > 2 raise; threads over lists")
@_field_listable
def _nfreg(a):
    _check_field_arg(a, "NumberFieldRegulator")
    th, P, d = _real_quadratic(a, "NumberFieldRegulator")
    if d is None or d < 0:
        return 1
    return ev(Expr("Log", R(_fund_unit(d))))


def _class_number_imag(D):
    """h(D) for a negative fundamental discriminant D (reduced primitive forms)."""
    h = 0
    a = 1
    absD = -D
    while 3 * a * a <= absD:
        for b in range(-a + 1, a + 1):
            if (b * b - D) % (4 * a):
                continue
            c = (b * b - D) // (4 * a)
            if c < a:
                continue
            if b < 0 and (a == c):
                continue
            if math.gcd(math.gcd(a, abs(b)), c) != 1:
                continue
            h += 1
        a += 1
    return h


def _narrow_class_number_real(D):
    """h+(D) for a positive fundamental discriminant: the number of cycles of reduced primitive forms."""
    r = math.isqrt(D)
    if r * r == D:
        raise ValueError("not a field discriminant")
    sq = math.sqrt(D)
    forms = set()
    for b in range(1 if D % 2 else 2, r + 1, 2):
        if b * b >= D:
            break
        N = (D - b * b) // 4
        for a0 in sp.divisors(N):
            for a in (a0, -a0):
                c = -N // a
                if sq - b < 2 * abs(a) < sq + b and math.gcd(math.gcd(abs(a), b), abs(c)) == 1:
                    forms.add((a, b, c))
    seen = set()
    cycles = 0
    for f in forms:
        if f in seen:
            continue
        cycles += 1
        g = f
        while g not in seen:
            seen.add(g)
            a, b, c = g
            m = 2 * abs(c)
            # b' = -b mod 2|c| with sqrt(D) - 2|c| < b' < sqrt(D)
            t = (-b) % m
            lo = sq - m
            k = math.floor((sq - t) / m)
            bp = t + k * m
            if bp >= sq:
                bp -= m
            while bp <= lo:
                bp += m
            ap = (bp * bp - D) // (4 * c)
            g = (c, bp, ap)
            if g not in forms:
                raise ArithmeticError("NumberFieldClassNumber: form reduction failed")
    return cycles


def _class_number_real(d):
    """h for Q(Sqrt[d]), d > 1 square-free: narrow class number from cycles of reduced forms, halved when the
    fundamental unit has norm +1."""
    D = d if d % 4 == 1 else 4 * d
    if D > 10 ** 12:
        raise NotImplementedError("NumberFieldClassNumber: the discriminant is too large")
    hp = _narrow_class_number_real(D)
    eps = _fund_unit(d)
    nrm = sp.expand(eps * eps.subs(sp.sqrt(d), -sp.sqrt(d)))
    return hp if nrm == -1 else hp // 2


@fn("NumberFieldClassNumber", **ADDED, notes="class number of Q(a) for quadratic fields: reduced binary quadratic "
    "forms for imaginary fields and the analytic class number formula (with the exact fundamental unit) for real "
    "fields; Q gives 1; fields of degree > 2 raise; threads over lists")
@_field_listable
def _nfclass(a):
    _check_field_arg(a, "NumberFieldClassNumber")
    th, P, d = _real_quadratic(a, "NumberFieldClassNumber")
    if d is None:
        return 1
    D = d if d % 4 == 1 else 4 * d
    if d < 0:
        return _class_number_imag(D)
    return _class_number_real(d)


@fn("NumberFieldRootsOfUnity", **ADDED, notes="the roots of unity in Q(a): {-1, 1} followed by the others as "
    "AlgebraicNumber objects in canonical (coefficient-list) order; threads over lists")
@_field_listable
def _nfroots(a):
    _check_field_arg(a, "NumberFieldRootsOfUnity")
    th, P = _field_data(a, "NumberFieldRootsOfUnity")
    n = P.degree()
    if n == 1 or len(sp.Poly(P.as_expr(), X).real_roots()) > 0:
        return L(-1, 1)
    w = 2
    for m in range(3, 2 * n * n + 3):
        if m % 2 == 1 or n % int(sp.totient(m)):
            continue
        z = sp.exp(2 * sp.pi * sp.I / m)
        try:
            to_number_field(z, th)
            w = m
        except IsomorphismFailed:
            pass
    roots = []
    for k in range(w):
        if k in (0, w // 2):
            continue
        z = sp.exp(2 * sp.pi * sp.I * k / w)
        cs = _express(z, th)
        roots.append(cs)
    roots.sort(key=lambda cs: [float(c) for c in cs])
    return L(-1, 1, *[make_an(th, P, cs) for cs in roots])




# ─────────────────────────── Root objects ───────────────────────────

def root_reduce_value(v):
    """Canonical single algebraic number for the SymPy algebraic number v (rational, quadratic radical or
    Root object)."""
    P = minpoly(v)
    if P.degree() == 1:
        a, b = P.all_coeffs()
        return sp.Rational(-b, a)
    if P.degree() == 2:
        return _radical_of(P, v)
    return sp.CRootOf(P, root_index(P, v))


@fn("RootReduce", **ADDED, listable=True, notes="RootReduce[expr] reduces an exact algebraic number (radicals, "
    "Root objects, AlgebraicNumber objects and their combinations) to a single rational number, a quadratic "
    "radical or a Root object; RootReduce[a == b] decides the equation; non-algebraic or symbolic expressions are "
    "returned unchanged")
def _rootreduce(x: Expr, *opts: Option) -> Expr:
    if opts:
        split_options(opts, ("Method",), "RootReduce")
    if isinstance(x, Expr) and x.head in ("Equal", "Unequal") and len(x.args) == 2:
        a, b = alg_value(x.args[0]), alg_value(x.args[1])
        if a is None or b is None:
            return x
        eq = minpoly(a - b).degree() == 1 and root_reduce_value(a - b) == 0 if True else None
        return eq if x.head == "Equal" else (not eq)
    v = alg_value(x)
    if v is None:
        return x
    try:
        r = root_reduce_value(v)
    except ValueError:
        return x
    return ev(R(r))


def _function_poly(f, fname):
    """SymPy polynomial in X for the pure function f (f[X])."""
    body = ev(K.call(f, Symbol(X.name)))
    try:
        e = sp.expand(S(body))
    except Exception:
        raise ValueError(f"{fname}: {f} is not a polynomial function")
    if not e.is_polynomial(X):
        raise ValueError(f"{fname}: {f} is not a polynomial function")
    P = sp.Poly(e, X)
    if P.degree() < 1:
        raise ValueError(f"{fname}: the polynomial must have positive degree")
    return P


def _apply_form(form, r):
    return ev(K.call(form, R(r) if isinstance(r, sp.Basic) else r))


@fn("RootSum", **ADDED, notes="RootSum[f, form] = Sum[form[x], {x roots of f}] (with multiplicity): exact for "
    "rational functions form (via the partial-fraction RootSum algorithm), explicit when every irreducible "
    "factor of f has degree <= 2, numeric when f has approximate coefficients; otherwise unevaluated (N "
    "evaluates it numerically)")
def _rootsum(f: Fn, form: Fn) -> Expr:
    P = _function_poly(f, "RootSum")
    if any(isinstance(c, sp.Float) for c in P.all_coeffs()) or any(c.has(sp.Float) for c in P.all_coeffs()):
        v = _num_rootsum(Expr("RootSum", f, form), 15)
        if v is None:
            raise Unevaluated()
        return v
    # explicit roots when the polynomial splits into factors of degree <= 2
    try:
        facs = sp.factor_list(P.as_expr(), X)[1]
    except Exception:
        facs = []
    if facs and all(sp.degree(fc, X) <= 2 for fc, _ in facs):
        terms = []
        for fc, m in facs:
            for r in sp.roots(sp.Poly(fc, X), multiple=True):
                terms.extend([_apply_form(form, r)] * m)
        return ev(Expr("Plus", *terms))
    # rational-function forms
    t = sp.Symbol("_w09t")
    body = ev(K.call(form, Symbol(t.name)))
    try:
        g = S(body)
    except Exception:
        raise Unevaluated()
    if g.is_rational_function(t) and not P.free_symbols - {X}:
        res = sp.RootSum(sp.Poly(P.as_expr(), X), sp.Lambda(t, g), auto=True)
        if not isinstance(res, sp.RootSum):
            return ev(R(sp.simplify(res)))
    raise Unevaluated()


def _num_rootsum(expr, prec):
    f, form = expr.args
    P = _function_poly(f, "RootSum")
    if P.free_symbols - {X}:
        return None
    roots = P.nroots(n=max(15, prec) + 5)
    tot = sp.Integer(0)
    for r in roots:
        v = ev(Expr("N", _apply_form(form, R(r)), max(15, prec) + 5))
        if not is_numeric_value(v):
            return None
        tot = tot + S(v)
    tot = sp.N(tot, max(15, prec))
    if prec <= 15:
        c = complex(tot)
        return c.real if abs(c.imag) <= 1e-13 * max(1.0, abs(c)) else c
    re_, im_ = tot.as_real_imag()
    return R(re_ if abs(im_) <= sp.Float(10) ** (-prec) * max(1, abs(re_)) else tot)


NT.N_PRE["RootSum"] = _num_rootsum


@fn("RootApproximant", **ADDED, notes="RootApproximant[x] finds a simple algebraic number (rational, quadratic "
    "radical or Root object) approximating the number x to its precision, trying degrees 1, 2, ... (at most 8) "
    "with integer-relation (PSLQ) searches and accepting the first relation whose coefficients are small for the "
    "precision; RootApproximant[x, n] restricts to degree <= n; raises when none is found")
def _rootapprox(x: Expr, n: Expr = None) -> Expr:
    import mpmath
    if not is_numeric_value(x):
        if is_list(x):
            return L(*[_rootapprox(t, n) for t in x.args])
        raise Unevaluated()
    v = S(x)
    from ..precision import input_precision
    pin = input_precision([x])
    dps = 15 if pin in (None, "m") else int(mpmath.libmp.prec_to_dps(pin))
    if not has_float(x):
        dps = 30
        v = sp.N(v, 40)
    maxdeg = 8 if n is None else int(n)
    if maxdeg < 1:
        raise ValueError("RootApproximant: the degree must be a positive integer")
    z = complex(sp.N(v, dps + 5))
    is_real = abs(z.imag) <= 1e-12 * max(1.0, abs(z)) if dps <= 15 else sp.im(sp.N(v, dps)) == 0
    with mpmath.workdps(dps):
        xm = mpmath.mpf(str(sp.N(sp.re(v), dps + 5))) if is_real else \
            mpmath.mpc(str(sp.N(sp.re(v), dps + 5)), str(sp.N(sp.im(v), dps + 5)))
        for d in range(1, maxdeg + 1):
            bits = dps * 3.32
            maxcoeff = int(2 ** (bits / (d + 1) * 0.8))
            if is_real:
                rel = mpmath.findpoly(xm, d, maxcoeff=max(maxcoeff, 2), tol=mpmath.mpf(10) ** (-(dps - 2)))
            else:
                powers = [xm ** k for k in range(d + 1)]
                rel = _complex_relation(powers, maxcoeff, dps)
            if rel:
                coeffs = [int(c) for c in rel]             # highest degree first
                P = sp.Poly(coeffs if is_real else list(reversed(coeffs)), X)
                if P.degree() < 1:
                    continue
                facs = [fc for fc, _ in sp.factor_list(P.as_expr(), X)[1]]
                best, bd = None, None
                for fc in facs:
                    FP = sp.Poly(fc, X)
                    if FP.degree() < 1:
                        continue
                    for i, r in enumerate(FP.all_roots()):
                        dist = abs(complex(sp.N(r, 30)) - z)
                        if bd is None or dist < bd:
                            best, bd = r, dist
                if best is not None:
                    return ev(R(root_reduce_value(best)))
    raise ValueError("RootApproximant: no algebraic approximation found")


def _complex_relation(powers, maxcoeff, dps):
    import mpmath
    vec = [mpmath.re(p) for p in powers] + []
    # stack real and imaginary parts: find integer c with sum c_k Re(p_k) = sum c_k Im(p_k) = 0
    n = len(powers)
    w = mpmath.mpf(10) ** (dps - 3)
    M = sp.Matrix([[1 if i == j else 0 for j in range(n)] + [int(mpmath.nint(w * mpmath.re(powers[i]))),
                                                           int(mpmath.nint(w * mpmath.im(powers[i])))]
                   for i in range(n)])
    from sympy.polys.matrices import DomainMatrix
    red = DomainMatrix.from_Matrix(M).convert_to(sp.ZZ).lll().to_Matrix()
    for i in range(red.rows):
        c = [int(red[i, j]) for j in range(n)]
        if any(c) and max(abs(t) for t in c) <= maxcoeff and abs(red[i, n]) <= n and abs(red[i, n + 1]) <= n:
            return c
    return None
