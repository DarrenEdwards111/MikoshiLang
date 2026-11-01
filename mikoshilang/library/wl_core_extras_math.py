"""Wolfram batch X01 (part 6): NumberFieldNormRepresentatives (quadratic fields) and NBodySimulation."""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, L, is_list, Unevaluated
from .wl_common import ev, is_sym

AREA = "core extras (X01)"


def _show(x):
    try:
        from .. import formatting as F
        return F.input_form(x)
    except Exception:  # noqa: BLE001 - display only
        return repr(x)


# ═══════════════════════════════════════ NumberFieldNormRepresentatives ═══════════════════════════════════════

def _squarefree(n):
    s = 1 if n > 0 else -1
    n = abs(n)
    out = 1
    for p, e in sp.factorint(n).items():
        if e % 2:
            out *= p
    return s * out


def _quadratic_data(a, fname):
    """(D, p, q, r) with a = (p + q sqrt(D)) / r, D squarefree != 1, q != 0."""
    from .wl_numbertheory2_alg import alg_value, minpoly
    v = alg_value(a)
    if v is None:
        try:
            free = bool(K.to_sp(a).free_symbols)
        except Exception:  # noqa: BLE001 - not convertible: not a number
            free = isinstance(a, Symbol)
        if free:
            raise Unevaluated()
        raise TypeError(f"{fname}: {_show(a)} is not an exact algebraic number")
    P = minpoly(v)
    if P.degree() != 2:
        raise NotImplementedError(f"{fname}: only quadratic fields are supported; {_show(a)} has degree "
                                  f"{P.degree()} (Wolfram's representative choice for higher degrees is not "
                                  "documented)")
    c2, c1, c0 = (int(c) for c in P.all_coeffs())
    disc = c1 * c1 - 4 * c2 * c0               # a = (-c1 +- sqrt(disc)) / (2 c2)
    D = _squarefree(disc)
    s = sp.sqrt(sp.Integer(disc // D))
    s = int(s)
    plus = (sp.Integer(-c1) + s * sp.sqrt(D)) / (2 * c2)
    z = complex(sp.N(v, 50))
    zp = complex(sp.N(plus, 50))
    q = s if abs(zp - z) < 1e-30 * max(1.0, abs(z)) or abs(zp - z) < 1e-20 else -s
    return D, -c1, q, 2 * c2


def _fundamental_unit(D):
    """(X, Y) of the fundamental unit (X + Y sqrt(D)) / 2 > 1 of the ring of integers of Q(sqrt(D)), D > 1."""
    from sympy.solvers.diophantine.diophantine import diop_DN
    cands = []
    norms = (4, -4) if D % 4 == 1 else (1, -1)
    for N in norms:
        for x, y in diop_DN(D, N):
            x, y = abs(int(x)), abs(int(y))
            if y == 0:
                continue
            if D % 4 == 1:
                if (x - y) % 2:
                    continue
                cands.append((x, y))
            else:
                cands.append((2 * x, 2 * y))
    if D % 4 == 1:
        for x, y in diop_DN(D, 1) + diop_DN(D, -1):
            if int(y):
                cands.append((2 * abs(int(x)), 2 * abs(int(y))))
    return min(cands, key=lambda c: c[0] + c[1] * math.sqrt(D))


def _isqrt_exact(n):
    if n < 0:
        return None
    r = math.isqrt(n)
    return r if r * r == n else None


def _integral(X, Y, D):
    return (X - Y) % 2 == 0 if D % 4 == 1 else (X % 2 == 0 and Y % 2 == 0)


def _norm_reps(D, m):
    """Representatives (X, Y) of alpha = (X + Y sqrt(D)) / 2 with |N(alpha)| = m, one per class of associates."""
    sD = sp.sqrt(D)
    if D < 0:
        sols = []
        Y = 0
        while -D * Y * Y <= 4 * m:
            X = _isqrt_exact(4 * m + D * Y * Y)
            if X is not None:
                for xx in {X, -X}:
                    for yy in {Y, -Y}:
                        if _integral(xx, yy, D):
                            sols.append((xx, yy))
            Y += 1
        roots = [(2, 0), (-2, 0)]
        if D == -1:
            roots += [(0, 2), (0, -2)]
        if D == -3:
            roots += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

        def mul(a, b):          # ((a0 + a1 s)/2) ((b0 + b1 s)/2) = (c0 + c1 s)/2, s^2 = D
            return ((a[0] * b[0] + D * a[1] * b[1]) // 2, (a[0] * b[1] + a[1] * b[0]) // 2)
        seen, reps = set(), []
        for s0 in sorted(set(sols)):
            if s0 in seen:
                continue
            orbit = {mul(s0, u) for u in roots}
            seen |= orbit
            pos = [o for o in orbit if o[0] > 0] or sorted(orbit, reverse=True)
            reps.append(min(pos, key=lambda o: (o[0], o[1])))
        return sorted(reps)
    ex, ey = _fundamental_unit(D)
    eps = (sp.Integer(ex) + ey * sD) / 2
    lo = sp.sqrt(m)
    hi = lo * eps
    bound = float(sp.N(lo * (eps + 1), 30)) / math.sqrt(D) + 2
    reps = []
    for Y in range(0, int(bound) + 1):
        for sgn in (1, -1):
            X = _isqrt_exact(D * Y * Y + 4 * sgn * m)
            if X is None:
                continue
            for xx in {X, -X}:
                if not _integral(xx, Y, D):
                    continue
                al = (sp.Integer(xx) + Y * sD) / 2
                if (al - lo).is_nonnegative and (hi - al).is_positive:
                    reps.append((xx, Y))
    return sorted(set(reps))


@fn("NumberFieldNormRepresentatives", area=AREA, status="added", listable=True,
    notes="Partial (quadratic fields). NumberFieldNormRepresentatives[a, m] for a quadratic algebraic number a "
          "gives one algebraic integer of norm +m or -m from each class of associates (elements whose quotient "
          "is a unit) of the ring of integers of Q(a), as AlgebraicNumber[a, {c0, c1}] (= c0 + c1 a). The "
          "representatives are chosen by an explicit rule, which can differ from Wolfram's undocumented choice by a "
          "unit factor: for real fields the representative alpha > 0 with Sqrt[m] <= alpha < Sqrt[m] eps (eps > 1 "
          "the fundamental unit; every class has exactly one), for imaginary fields the associate with the "
          "smallest positive rational part. They are found by solving X^2 - D Y^2 = +-4m for alpha = (X + Y "
          "Sqrt[D])/2. Fields of degree other than 2 raise NotImplementedError; m must be a nonzero integer "
          "(ValueError); Listable. Check: reference (the documentation: the classes for Sqrt[2], 7 are those of "
          "-3 - Sqrt[2] and -3 + Sqrt[2]; for I, 5 the Gaussian integers 1 - 2 I, 1 + 2 I; for Sqrt[5], 5 one "
          "class) and exact norms in tests.")
def _nfnr(a: Expr, m: Integer) -> ListOf[Expr]:
    fname = "NumberFieldNormRepresentatives"
    if not (K.is_int(m) and not isinstance(m, bool)):
        if isinstance(m, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the norm must be an integer, not {_show(m)}")
    m = abs(int(m))
    if m == 0:
        raise ValueError(f"{fname}: the norm must be a nonzero integer")
    D, p, q, r = _quadratic_data(a, fname)
    out = []
    for X, Y in _norm_reps(D, m):
        # alpha = X/2 + (Y/2) sqrt(D),  sqrt(D) = (r a - p) / q
        c1 = sp.Rational(Y * r, 2 * q)
        c0 = sp.Rational(X, 2) - sp.Rational(Y * p, 2 * q)
        out.append(Expr("AlgebraicNumber", a, L(K.from_sp(c0), K.from_sp(c1))))
    return L(*out)
