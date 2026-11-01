"""Wolfram batch X01 (part 6): NumberFieldNormRepresentatives (quadratic fields) and NBodySimulation."""
from __future__ import annotations

import math
import re

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, L, is_list, Unevaluated
from .. import kernel_lists as KL
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
          "gives one algebraic integer from each class of associates (elements whose quotient is a unit) "
          "of norm +m or -m in the ring of integers of Q(a), as AlgebraicNumber[a, {c0, c1}] (= c0 + c1 a); as in "
          "the documented examples the representative has norm exactly m whenever its class contains such an "
          "element (real fields whose fundamental unit has norm -1), else norm -m (e.g. Sqrt[3], 2 gives an "
          "element of norm -2). The representatives are chosen by an explicit rule, which can differ from "
          "Wolfram's undocumented choice by a unit factor: for real fields the representative alpha > 0 with "
          "|m|^(1/2) <= alpha < |m|^(1/2) eps (eps > 1 the fundamental unit; every class has exactly one), "
          "multiplied by eps when that fixes the sign of the norm; for imaginary fields (norms are positive) the associate with the "
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
    msign = 1 if int(m) > 0 else -1
    m = abs(int(m))
    if m == 0:
        raise ValueError(f"{fname}: the norm must be a nonzero integer")
    D, p, q, r = _quadratic_data(a, fname)
    reps = _norm_reps(D, m)
    if D > 0:
        ex, ey = _fundamental_unit(D)
        unit_neg = (ex * ex - D * ey * ey) // 4 == -1
        fixed = []
        for X, Y in reps:
            if (X * X - D * Y * Y) // 4 * msign < 0 and unit_neg:
                # multiply by the fundamental unit of norm -1 to get norm exactly m
                X, Y = (X * ex + D * Y * ey) // 2, (X * ey + Y * ex) // 2
            fixed.append((X, Y))
        reps = fixed
    out = []
    for X, Y in reps:
        # alpha = X/2 + (Y/2) sqrt(D),  sqrt(D) = (r a - p) / q
        c1 = sp.Rational(Y * r, 2 * q)
        c0 = sp.Rational(X, 2) - sp.Rational(Y * p, 2 * q)
        out.append(Expr("AlgebraicNumber", a, L(K.from_sp(c0), K.from_sp(c1))))
    return L(*out)


# ═══════════════════════════════════════════════ NBodySimulation ═══════════════════════════════════════════════

import itertools as _it

import numpy as _np

_SIMS = {}
_SIM_IDS = _it.count(1)

# Dormand-Prince 5(4) coefficients (Hairer, Norsett & Wanner, Solving ODEs I, table 5.2)
_DP_C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
_DP_A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
         [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
         [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
         [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
_DP_B = _np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_DP_E = _np.array([71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40])


class _Law:
    """Accelerations and energies of a system: pairwise and external potentials or forces."""

    def __init__(self, n, dim, masses):
        self.n, self.dim, self.m = n, dim, _np.asarray(masses, float)
        self.pair_pot = []        # (i, j, grad(qi, qj) -> (fi, fj), energy(qi, qj))
        self.pair_force = []      # (i, j, force_on_i(qi, qj, vi, vj))
        self.ext_pot = []         # (i, grad(q) -> force, energy(q))
        self.ext_force = []       # (i, force(q, v))
        self.conservative = True

    def accel(self, q, v):
        F = _np.zeros_like(q)
        for i, j, force, _e in self.pair_pot:
            fi = force(q[i], q[j])
            F[i] += fi
            F[j] -= fi
        for i, j, f in self.pair_force:
            F[i] += f(q[i], q[j], v[i], v[j])
        for i, force, _e in self.ext_pot:
            F[i] += force(q[i])
        for i, f in self.ext_force:
            F[i] += f(q[i], v[i])
        return F / self.m[:, None]

    def energies(self, q, v):
        ke = 0.5 * float(_np.sum(self.m[:, None] * v * v))
        pe = sum(e(q[i], q[j]) for i, j, _f, e in self.pair_pot)
        xe = sum(e(q[i]) for i, _f, e in self.ext_pot)
        return ke, float(pe), float(xe)


def _inverse_square(mi, mj):
    def force(qi, qj):           # V = -mi mj / r: force on i = -mi mj (qi - qj) / r^3
        d = qi - qj
        r = math.sqrt(float(d @ d))
        return -mi * mj * d / r ** 3

    def energy(qi, qj):
        d = qi - qj
        return -mi * mj / math.sqrt(float(d @ d))
    return force, energy


def _harmonic(k, L0):
    def force(qi, qj):           # V = k/2 (r - L0)^2
        d = qi - qj
        r = math.sqrt(float(d @ d))
        if r == 0:
            return _np.zeros_like(d)
        return -k * (r - L0) * d / r

    def energy(qi, qj):
        d = qi - qj
        r = math.sqrt(float(d @ d))
        return 0.5 * k * (r - L0) ** 2
    return force, energy


def _num_value(x, fname, what):
    if isinstance(x, Expr) and x.head == "Quantity":
        raise NotImplementedError(f"{fname}: quantities with units are not supported; give unitless values (the "
                                  "\"InverseSquare\" and \"Harmonic\" laws are unitless)")
    v = ev(Expr("N", x))
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        if isinstance(v, Symbol) or (isinstance(v, Expr) and not is_list(v)):
            raise Unevaluated()
        raise TypeError(f"{fname}: {what} must be a real number, not {_show(x)}")
    return float(v)


def _vector(x, fname, what):
    if isinstance(x, Expr) and x.head == "Quantity":
        _num_value(x, fname, what)
    if not is_list(x) or not x.args:
        raise TypeError(f"{fname}: {what} must be a nonempty list of numbers, not {_show(x)}")
    return _np.array([_num_value(c, fname, what) for c in x.args], float)


def _harmonic_params(spec, fname):
    k, L0 = 1.0, 1.0
    items = []
    for a in spec:
        if isinstance(a, Expr) and a.head == "Association":
            items.extend(a.args)
        else:
            items.append(a)
    for r in items:
        if not (isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2):
            raise ValueError(f"{fname}: invalid \"Harmonic\" parameter {_show(r)}")
        key = r.args[0]
        if key == "SpringConstant":
            k = _num_value(r.args[1], fname, "the spring constant")
        elif key == "EquilibriumLength":
            L0 = _num_value(r.args[1], fname, "the equilibrium length")
        else:
            raise ValueError(f"{fname}: unknown \"Harmonic\" parameter {_show(key)} (SpringConstant, "
                             "EquilibriumLength)")
    return k, L0


def _named(spec, mi, mj, fname):
    """(force, energy) of a named pairwise law, or None when spec is not a named law."""
    if isinstance(spec, str):
        name, params = spec, []
    elif is_list(spec) and spec.args and isinstance(spec.args[0], str):
        name, params = spec.args[0], list(spec.args[1:])
    else:
        return None
    if name == "InverseSquare":
        if params:
            raise ValueError(f"{fname}: \"InverseSquare\" takes no parameters")
        return _inverse_square(mi, mj)
    if name == "Harmonic":
        return _harmonic(*_harmonic_params(params, fname))
    if name in ("Newtonian", "Coulomb"):
        raise NotImplementedError(f"{fname}: the {name!r} law uses physical units, which are not supported "
                                  "(use \"InverseSquare\" with unitless values)")
    raise ValueError(f"{fname}: unknown law {name!r} (InverseSquare, Harmonic, Newtonian, Coulomb)")


def _sym_state(st, idx, dim):
    """State association with symbolic position/velocity components (for potential/force functions)."""
    qs = [sp.Symbol(f"q{idx}_{k}", real=True) for k in range(dim)]
    vs = [sp.Symbol(f"v{idx}_{k}", real=True) for k in range(dim)]
    rules = []
    for r in st.args:
        key = r.args[0]
        if key == "Position":
            rules.append(Expr("Rule", key, L(*(K.from_sp(s) for s in qs))))
        elif key == "Velocity":
            rules.append(Expr("Rule", key, L(*(K.from_sp(s) for s in vs))))
        elif key == "Momentum":
            continue
        else:
            rules.append(r)
    m = next(r.args[1] for r in st.args if r.args[0] == "Mass")
    rules.append(Expr("Rule", "Momentum", L(*(K.from_sp(sp.Symbol(f"v{idx}_{k}", real=True)) for k in range(dim)))))
    rules[-1] = Expr("Rule", "Momentum", ev(Expr("Times", m, L(*(K.from_sp(s) for s in vs)))))
    return Expr("Association", *rules), qs, vs


def _to_sympy(x, fname, what):
    try:
        e = K.to_sp(x)
        # the state symbols come back without the real assumption: restore it by name
        return e.subs({s: sp.Symbol(s.name, real=True) for s in e.free_symbols
                       if isinstance(s, sp.Symbol) and _SYM_RE.match(s.name)})
    except Exception as e:  # noqa: BLE001 - reported as a type error
        raise TypeError(f"{fname}: the {what} did not evaluate to an expression of the positions: {_show(x)}") from e


def _function_pot(f, states, i, j, dim, fname):
    si, qi, vi = _sym_state(states[i], i, dim)
    sj, qj, vj = _sym_state(states[j], j, dim)
    V = _to_sympy(ev(Expr(f, si, sj)), fname, "pairwise potential")
    if V.free_symbols & set(vi + vj):
        raise NotImplementedError(f"{fname}: velocity-dependent potentials are not supported (use PairwiseForce)")
    gi = [sp.diff(V, s) for s in qi]
    args = qi + qj
    fV = sp.lambdify(args, V, "math")
    fg = sp.lambdify(args, gi, "math")

    def force(a, b):
        return -_np.array(fg(*a, *b), float)

    def energy(a, b):
        return float(fV(*a, *b))
    # the force on j is minus the force on i only for potentials of qi - qj; check that
    gj = [sp.diff(V, s) for s in qj]
    if any(sp.simplify(x + y) != 0 for x, y in zip(gi, gj)):
        raise NotImplementedError(f"{fname}: pairwise potentials must depend on the positions through their "
                                  "difference (translation invariant)")
    return force, energy


def _function_force(f, states, i, j, dim, fname):
    si, qi, vi = _sym_state(states[i], i, dim)
    sj, qj, vj = _sym_state(states[j], j, dim)
    Fx = ev(Expr(f, si, sj))
    if not is_list(Fx) or len(Fx.args) != dim:
        raise TypeError(f"{fname}: a pairwise force must give a vector of length {dim}, not {_show(Fx)}")
    comps = [_to_sympy(c, fname, "pairwise force") for c in Fx.args]
    fl = sp.lambdify(qi + qj + vi + vj, comps, modules=["math", {"Ramp": lambda x: max(x, 0.0),
                                                                  "UnitStep": lambda x: 1.0 if x >= 0 else 0.0}])

    def force(a, b, va, vb):
        return _np.array(fl(*a, *b, *va, *vb), float)
    return force


def _function_ext(f, states, i, dim, fname, kind):
    si, qi, vi = _sym_state(states[i], i, dim)
    x = ev(Expr(f, si))
    if kind == "potential":
        V = _to_sympy(x, fname, "external potential")
        g = sp.lambdify(qi, [sp.diff(V, s) for s in qi], "math")
        fV = sp.lambdify(qi, V, "math")
        return (lambda a: -_np.array(g(*a), float)), (lambda a: float(fV(*a)))
    if not is_list(x) or len(x.args) != dim:
        raise TypeError(f"{fname}: an external force must give a vector of length {dim}, not {_show(x)}")
    fl = sp.lambdify(qi + vi, [_to_sympy(c, fname, "external force") for c in x.args], "math")
    return lambda a, va: _np.array(fl(*a, *va), float)


def _build_law(law, states, masses, dim, fname):
    n = len(states)
    W = _Law(n, dim, masses)
    if isinstance(law, Expr) and law.head == "Association":
        spec = {r.args[0]: r.args[1] for r in law.args}
        for k in spec:
            if k not in ("PairwisePotential", "PairwiseForce", "ExternalPotential", "ExternalForce", "Region"):
                raise ValueError(f"{fname}: unknown law property {_show(k)}")
        if "Region" in spec:
            raise NotImplementedError(f"{fname}: confining \"Region\" specifications (reflection at the boundary) "
                                      "are not supported")
    else:
        spec = {"PairwisePotential": law}
    pp = spec.get("PairwisePotential")
    if pp is not None:
        if isinstance(pp, Expr) and pp.head == "Association":
            for r in pp.args:
                pair = r.args[0]
                if not (is_list(pair) and len(pair.args) == 2):
                    raise TypeError(f"{fname}: per-pair potentials are given as {{i, j}} -> law")
                i, j = (_body_index(b, states, fname) for b in pair.args)
                W.pair_pot.append((i, j, *_pair_law(r.args[1], states, masses, i, j, dim, fname)))
        else:
            for i, j in _it.combinations(range(n), 2):
                W.pair_pot.append((i, j, *_pair_law(pp, states, masses, i, j, dim, fname)))
    pf = spec.get("PairwiseForce")
    if pf is not None:
        W.conservative = False
        for i in range(n):
            for j in range(n):
                if i != j:
                    W.pair_force.append((i, j, _function_force(pf, states, i, j, dim, fname)))
    xp = spec.get("ExternalPotential")
    if xp is not None:
        for i in range(n):
            W.ext_pot.append((i, *_function_ext(xp, states, i, dim, fname, "potential")))
    xf = spec.get("ExternalForce")
    if xf is not None:
        W.conservative = False
        for i in range(n):
            W.ext_force.append((i, _function_ext(xf, states, i, dim, fname, "force")))
    return W


_SYM_RE = re.compile(r"^[qv]\d+_\d+$")


def _body_index(b, states, fname):
    names = _BODY_NAMES[-1]
    if b in names:
        return names.index(b)
    raise ValueError(f"{fname}: {_show(b)} is not a body of the system")


_BODY_NAMES = [[]]


def _pair_law(spec, states, masses, i, j, dim, fname):
    r = _named(spec, masses[i], masses[j], fname)
    if r is not None:
        return r
    return _function_pot(spec, states, i, j, dim, fname)


def _integrate(W, q0, v0, T, rtol=1e-11, atol=1e-12):
    """Adaptive Dormand-Prince 5(4) for q'' = a(q, v); returns (times, Q, V, A, stop time)."""
    n, dim = q0.shape
    y = _np.concatenate([q0.ravel(), v0.ravel()])

    def f(yy):
        q = yy[:n * dim].reshape(n, dim)
        v = yy[n * dim:].reshape(n, dim)
        return _np.concatenate([v.ravel(), W.accel(q, v).ravel()])
    t = 0.0
    k1 = f(y)
    scale = max(1.0, float(_np.max(_np.abs(y))))
    h = min(T, 1e-3 * max(T, 1e-12))
    ts, ys, ds = [0.0], [y.copy()], [k1.copy()]
    hmin = 1e-12 * max(T, 1.0)
    while t < T:
        h = min(h, T - t)
        ks = [k1]
        try:
            for s in range(1, 7):
                yi = y + h * sum(a * ks[k] for k, a in enumerate(_DP_A[s]) if a)
                ks.append(f(yi))
        except (ZeroDivisionError, FloatingPointError, OverflowError):
            h *= 0.25
            if h < hmin:
                break
            continue
        K7 = _np.array(ks)
        y5 = y + h * (_DP_B @ K7)
        err = h * (_DP_E @ K7)
        sc = atol + rtol * _np.maximum(_np.abs(y), _np.abs(y5))
        e = float(_np.sqrt(_np.mean((err / sc) ** 2)))
        if not _np.all(_np.isfinite(y5)):
            e = _np.inf
        if e <= 1.0:
            t += h
            y = y5
            k1 = ks[6]
            ts.append(t)
            ys.append(y.copy())
            ds.append(k1.copy())
        fac = 0.9 * (1.0 / e) ** 0.2 if e > 0 else 5.0
        h *= min(5.0, max(0.2, fac))
        if h < hmin:
            break                               # step size effectively zero: collision or singularity
    _ = scale
    Y = _np.array(ys)
    Dv = _np.array(ds)
    Q = Y[:, :n * dim].reshape(-1, n, dim)
    V = Y[:, n * dim:].reshape(-1, n, dim)
    A = Dv[:, n * dim:].reshape(-1, n, dim)
    return _np.array(ts), Q, V, A, t


class _Sim:
    def __init__(self, names, states, masses, extra, W, ts, Q, V, A, T):
        self.names, self.states, self.m, self.extra = names, states, masses, extra
        self.W, self.ts, self.Q, self.V, self.A, self.T = W, ts, Q, V, A, T

    def at(self, t):
        """(q, v) at time t by quintic Hermite interpolation of positions, cubic of velocities."""
        if not (0 <= t <= self.T * (1 + 1e-14)):
            raise ValueError(f"NBodySimulationData: the time {t} is outside the simulated interval [0, {self.T}]")
        ts = self.ts
        k = int(_np.searchsorted(ts, t, side="right")) - 1
        k = min(max(k, 0), len(ts) - 2) if len(ts) > 1 else 0
        if len(ts) == 1:
            return self.Q[0], self.V[0]
        t0, t1 = ts[k], ts[k + 1]
        h = t1 - t0
        s = (t - t0) / h
        q0, q1, v0, v1, a0, a1 = self.Q[k], self.Q[k + 1], self.V[k], self.V[k + 1], self.A[k], self.A[k + 1]
        h00 = 1 - 10 * s ** 3 + 15 * s ** 4 - 6 * s ** 5
        h10 = s - 6 * s ** 3 + 8 * s ** 4 - 3 * s ** 5
        h20 = 0.5 * s ** 2 - 1.5 * s ** 3 + 1.5 * s ** 4 - 0.5 * s ** 5
        h01 = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        h11 = -4 * s ** 3 + 7 * s ** 4 - 3 * s ** 5
        h21 = 0.5 * s ** 3 - s ** 4 + 0.5 * s ** 5
        q = h00 * q0 + h * h10 * v0 + h * h * h20 * a0 + h01 * q1 + h * h11 * v1 + h * h * h21 * a1
        c00, c10, c01, c11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, \
            s ** 3 - s ** 2
        v = c00 * v0 + h * c10 * a0 + c01 * v1 + h * c11 * a1
        return q, v


def _state_assoc(sim, i, q, v):
    vec = lambda x: L(*(float(c) for c in x))  # noqa: E731
    return Expr("Association", Expr("Rule", "Mass", sim.extra[i]["Mass"]), Expr("Rule", "Position", vec(q[i])),
                Expr("Rule", "Velocity", vec(v[i])), Expr("Rule", "Momentum", vec(sim.m[i] * v[i])))


_ENERGY_PROPS = ("TotalKineticEnergy", "TotalPairwiseInteractionEnergy", "TotalExternalEnergy", "TotalEnergy")
_STATE_PROPS = ("Mass", "Position", "Velocity", "Momentum")


def _sim_of(head, fname="NBodySimulationData"):
    a = head.args[0] if head.args else None
    sid = None
    if isinstance(a, Expr) and a.head == "Association":
        for r in a.args:
            if r.args[0] == "ID":
                sid = r.args[1]
    if sid not in _SIMS:
        raise ValueError(f"{fname}: this simulation is no longer available")
    return _SIMS[sid]


def _time(t):
    """Float time, None for a symbolic time (the property stays unevaluated); other values raise TypeError."""
    if isinstance(t, bool) or isinstance(t, str):
        raise TypeError(f"NBodySimulationData: the time must be a real number, not {_show(t)}")
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, Symbol):
        return None
    try:
        v = ev(Expr("N", t))
    except (ValueError, TypeError) as e:
        raise TypeError(f"NBodySimulationData: the time must be a real number, not {_show(t)}") from e
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _nbody_hook(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == "NBodySimulationData"):
        return K.UNHANDLED
    sim = _sim_of(head)
    args = list(args)
    if len(args) == 1 and isinstance(args[0], str):
        p = args[0]
        if p == "SimulationTime":
            return sim.T
        if p == "Bodies":
            return L(*sim.names)
        if p == "Properties":
            return L("Bodies", "SimulationTime", *_ENERGY_PROPS, *_STATE_PROPS)
        if p in ("Equations", "HamiltonEquations"):
            raise NotImplementedError(f"NBodySimulationData: the symbolic {p!r} are not given (the motion is "
                                      "integrated numerically)")
        if p in _ENERGY_PROPS:
            return K.UNHANDLED
    if len(args) == 2 and args[0] in _ENERGY_PROPS:
        t = _time(args[1])
        if t is None:
            return K.UNHANDLED
        q, v = sim.at(t)
        ke, pe, xe = sim.W.energies(q, v)
        return {"TotalKineticEnergy": ke, "TotalPairwiseInteractionEnergy": pe, "TotalExternalEnergy": xe,
                "TotalEnergy": ke + pe + xe}[args[0]]
    if len(args) in (2, 3):
        body = args[0]
        prop = args[1] if len(args) == 3 else None
        t = _time(args[-1])
        if prop is not None and prop not in _STATE_PROPS:
            raise ValueError(f"NBodySimulationData: unknown property {_show(prop)} ({', '.join(_STATE_PROPS)})")
        if is_sym(body, "All"):
            idx = list(range(len(sim.names)))
        elif body in sim.names:
            idx = [sim.names.index(body)]
        else:
            raise ValueError(f"NBodySimulationData: {_show(body)} is not a body of the simulation")
        if t is None:
            return K.UNHANDLED
        q, v = sim.at(t)
        out = []
        for i in idx:
            st = _state_assoc(sim, i, q, v)
            out.append(st if prop is None else next(r.args[1] for r in st.args if r.args[0] == prop))
        return L(*out) if is_sym(body, "All") else out[0]
    raise ValueError("NBodySimulationData: use data[body or All, t], data[body or All, prop, t], data[energy, t] or "
                     "data[\"SimulationTime\"]")


KL.COMPOUND_HEAD_HOOKS["NBodySimulationData"] = _nbody_hook


@fn("NBodySimulation", area=AREA, status="added",
    notes="Partial (unitless laws). NBodySimulation[law, {state1, ...}, t] and NBodySimulation[law, <|name -> state, "
          "...|>, t] integrate the classical equations of motion m_i q_i'' = F_i from states <|\"Mass\" -> m, "
          "\"Position\" -> {...}, \"Velocity\" -> {...}, ...|> (any dimension) over [0, t] with an adaptive "
          "Dormand-Prince 5(4) method (relative tolerance 1e-11) and give NBodySimulationData. Laws: "
          "\"InverseSquare\" (V = -m_i m_j / r, G = 1), \"Harmonic\" and {\"Harmonic\", \"SpringConstant\" -> k, "
          "\"EquilibriumLength\" -> l} or {\"Harmonic\", <|...|>} (V = k (r - l)^2 / 2, defaults 1 and 1), a "
          "pairwise potential function f[state_i, state_j] (translation invariant, differentiated symbolically), "
          "or <|\"PairwisePotential\" -> law or <|{i, j} -> law, ...|>, \"PairwiseForce\" -> f (force on i, "
          "applied for every ordered pair), \"ExternalPotential\" -> f[state], \"ExternalForce\" -> f[state]|>. "
          "When the step size becomes effectively zero (a collision) the simulation stops and "
          "\"SimulationTime\" is the time reached, as documented. data[All or body, t] gives <|Mass, Position, "
          "Velocity, Momentum|> (quintic Hermite interpolation between steps), data[All or body, prop, t] one "
          "property, data[\"TotalKineticEnergy\" | \"TotalPairwiseInteractionEnergy\" | \"TotalExternalEnergy\" | "
          "\"TotalEnergy\", t] the energies; symbolic t stays unevaluated (for Plot[Evaluate[...]]). For "
          "conservative laws the energy at the end is checked against the start (ArithmeticError if it drifted by "
          "more than 1e-6 relative). Quantities with units, the \"Newtonian\" and \"Coulomb\" laws, \"Region\" and "
          "the \"Equations\"/\"HamiltonEquations\" properties raise NotImplementedError. Check: reference (the "
          "documentation's three-body example data[All, 4] and the two-body collision time 1.32088 = (Pi/2) "
          "Sqrt[r0^3/(2 (m1 + m2))] for r0 = Sqrt[2]) in tests.")
def _nbodysimulation(law: Expr, states: Expr, t: Expr) -> Expr:
    fname = "NBodySimulation"
    T = _num_value(t, fname, "the simulation time")
    if not T > 0:
        raise ValueError(f"{fname}: the simulation time must be positive")
    if isinstance(states, Expr) and states.head == "Association":
        names = [r.args[0] for r in states.args]
        sts = [r.args[1] for r in states.args]
    elif is_list(states):
        names = list(range(1, len(states.args) + 1))
        sts = list(states.args)
    else:
        if isinstance(states, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the states must be a list or an association of state associations")
    if len(sts) < 1:
        raise ValueError(f"{fname}: at least one body is needed")
    masses, q0, v0, extra = [], [], [], []
    for s in sts:
        if not (isinstance(s, Expr) and s.head == "Association"):
            raise TypeError(f"{fname}: a body state must be an association, not {_show(s)}")
        d = {r.args[0]: r.args[1] for r in s.args}
        for k in ("Mass", "Position", "Velocity"):
            if k not in d:
                raise ValueError(f"{fname}: the state {_show(s)} has no {k!r}")
        m = _num_value(d["Mass"], fname, "a mass")
        if not m > 0:
            raise ValueError(f"{fname}: masses must be positive")
        masses.append(m)
        q0.append(_vector(d["Position"], fname, "a position"))
        v0.append(_vector(d["Velocity"], fname, "a velocity"))
        for k, v in d.items():
            if isinstance(v, Expr) and v.head == "Quantity":
                _num_value(v, fname, k)
        extra.append(d)
    dim = len(q0[0])
    if any(len(x) != dim for x in q0 + v0):
        raise ValueError(f"{fname}: all positions and velocities must have the same dimension")
    _BODY_NAMES.append(names)
    try:
        W = _build_law(law, sts, masses, dim, fname)
    finally:
        _BODY_NAMES.pop()
    Q0, V0 = _np.array(q0), _np.array(v0)
    with _np.errstate(divide="raise", invalid="raise", over="raise"):
        ts, Q, V, A, Tend = _integrate(W, Q0, V0, T)
    sim = _Sim(names, sts, _np.array(masses), extra, W, ts, Q, V, A, float(Tend))
    if W.conservative and Tend >= T * (1 - 1e-12):
        e0 = sum(W.energies(Q0, V0))
        e1 = sum(W.energies(Q[-1], V[-1]))
        mag = max(abs(x) for x in W.energies(Q0, V0) + W.energies(Q[-1], V[-1])) or 1.0
        if abs(e1 - e0) > 1e-6 * mag:
            raise ArithmeticError(f"{fname}: the total energy drifted from {e0} to {e1}; the integration is not "
                                  "accurate enough")
    sid = next(_SIM_IDS)
    _SIMS[sid] = sim
    return Expr("NBodySimulationData", Expr("Association", Expr("Rule", "Bodies", len(names)),
                                            Expr("Rule", "SimulationTime", float(Tend)), Expr("Rule", "ID", sid)))


# ═══════════════════════════════════════════════ SpheroidalRadialFactor ═══════════════════════════════════════════════

import mpmath as _mp

from . import wl_special_elliptic as _SE


def _srf(n, m, c):
    name = "SpheroidalRadialFactor"
    n, m = _SE._sph_nm(n, m, name)
    if m != 0:
        raise NotImplementedError(f"{name}: only order m = 0 is supported (the normalisation used for m != 0 is not "
                                  "documented and does not match DLMF 30.11.4)")
    g2 = _SE._sph_g2(c, name)
    if g2 == 0:
        return _mp.mpf(1)

    def run():
        return _SE._sph_radial_coeffs(n, 0, g2)[1]
    return _SE._verified(run, name)


_SE.mpfun("SpheroidalRadialFactor", 3, _srf,
          notes="Partial (order m = 0). SpheroidalRadialFactor[n, 0, c] is A_n^0(c^2) = Sum_k (-1)^k a^0_{n,k}(c^2) "
                "(DLMF 30.11.4) with the Meixner-Schaefke normalised expansion coefficients of Ps_n^0 used by "
                "SpheroidalS1 (so SpheroidalRadialFactor[n, 0, 0] = 1); integer n >= 0, real or purely imaginary c, "
                "arbitrary precision (evaluated at two working precisions which must agree). m != 0 (whose "
                "normalisation is undocumented) and non-integer n or m raise NotImplementedError. Check: reference "
                "(N[SpheroidalRadialFactor[7, 0, 1/4], 50] from the documentation, all 50 digits) in tests.")
