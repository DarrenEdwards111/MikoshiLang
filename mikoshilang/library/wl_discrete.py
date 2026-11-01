"""Coverage batch 6: discrete mathematics — intervals, discrete calculus, linear recurrences, continued fractions,
bitwise operations, cellular automata / substitution systems and elementary integer predicates.

Conventions follow the Wolfram Language reference pages:

* ``Interval[{a, b}, ...]`` is kept canonical: endpoints ordered, components sorted and overlapping/touching
  components merged (closed intervals);
* ``DifferenceDelta[f, {i, n, h}]`` is the n-th forward difference with step h, ``DiscreteShift`` the shift
  ``f(i + n h)`` and ``DiscreteRatio`` the (iterated) ratio ``f(i + h) / f(i)``;
* ``LinearRecurrence[ker, init, n]`` uses ``a(k) = ker[[1]] a(k-1) + ... + ker[[d]] a(k-d)``; the spec ``{n}``
  gives the n-th term only and ``{nmin, nmax}`` the terms nmin..nmax;
* ``ContinuedFraction`` of a quadratic irrational gives ``{a1, ..., {b1, ..., bk}}`` with the period as the last
  element; terminating fractions end with a term > 1 (``{..., k}`` rather than ``{..., k - 1, 1}``);
* bit operations use infinite two's complement for negative integers (``BitNot[n] == -1 - n``) and
  ``BitShiftRight[n, k] == IntegerPart[n / 2^k]``.
"""

from __future__ import annotations
import math
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import Unevaluated, fn, L, R, S, is_list, int_arg, nonneg_int, pos_int, list_arg, sym, \
    is_numeric_value, has_float
from .wl_common import ev

AREA = "discrete mathematics (batch 6)"


def _intq(x):
    return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, sp.Integer)


def _head(x, h, n=None):
    return isinstance(x, Expr) and x.head == h and (n is None or len(x.args) == n)


def _exact_rational(x):
    """Python Fraction for an exact rational (int or sympy Rational), else None."""
    if _intq(x):
        return Fraction(int(x))
    if isinstance(x, sp.Rational):
        return Fraction(int(x.p), int(x.q))
    if isinstance(x, Expr) and x.head == "Rational" and len(x.args) == 2 and all(_intq(a) for a in x.args):
        return Fraction(int(x.args[0]), int(x.args[1]))
    try:
        v = S(x)
    except Exception:
        return None
    if isinstance(v, sp.Rational):
        return Fraction(int(v.p), int(v.q))
    return None


def _out_frac(f):
    return int(f.numerator) if f.denominator == 1 else R(sp.Rational(f.numerator, f.denominator))


# ───────────────────────────────────────── intervals ─────────────────────────────────────────

def _endpoint(x, fname):
    """SymPy value of a real interval endpoint (exact, inexact or +-Infinity); symbolic -> Unevaluated."""
    try:
        v = S(x)
    except Exception:
        raise ValueError(f"{fname}: interval endpoint {x!r} is not a real number")
    if not isinstance(v, sp.Basic):
        raise ValueError(f"{fname}: interval endpoint {x!r} is not a real number")
    if v in (sp.oo, -sp.oo):
        return v
    if v.free_symbols or not v.is_number:
        raise Unevaluated()
    if v.is_real is False:
        raise ValueError(f"{fname}: interval endpoint {x!r} is not real")
    if v.is_real is None:
        raise Unevaluated()
    return v


def _lt(a, b):
    r = sp.Lt(a, b)
    if r not in (sp.true, sp.false):
        raise ValueError(f"cannot compare interval endpoints {a} and {b}")
    return bool(r)


def _components(iv, fname):
    """List of (lo, hi, lo_expr, hi_expr) for an Interval expression."""
    if not _head(iv, "Interval"):
        if isinstance(iv, (Symbol, Expr)):
            raise Unevaluated()
        raise ValueError(f"{fname}: {iv!r} is not an Interval")
    out = []
    for c in iv.args:
        if not (is_list(c) and len(c.args) == 2):
            raise ValueError(f"{fname}: Interval components must be pairs {{min, max}}, got {c!r}")
        a, b = c.args
        va, vb = _endpoint(a, fname), _endpoint(b, fname)
        if _lt(vb, va):
            va, vb, a, b = vb, va, b, a
        out.append((va, vb, a, b))
    return out


def _merge(comps):
    import functools

    def cmp(c1, c2):
        if c1[0] == c2[0]:
            return 0
        return -1 if _lt(c1[0], c2[0]) else 1
    comps = sorted(comps, key=functools.cmp_to_key(cmp))
    out = []
    for c in comps:
        if out and not _lt(out[-1][1], c[0]):          # overlapping or touching
            last = out[-1]
            if _lt(last[1], c[1]):
                out[-1] = (last[0], c[1], last[2], c[3])
        else:
            out.append(c)
    return out


def _interval(comps):
    return Expr("Interval", *(L(c[2], c[3]) for c in comps))


@fn("Interval", area=AREA, status="added",
    notes="canonical form only: endpoints ordered, components sorted and overlapping/touching components merged; "
          "interval arithmetic is not implemented"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _interval_fn(*args: Expr) -> Expr:
    if not args:
        raise Unevaluated()
    iv = Expr("Interval", *args)
    merged = _interval(_merge(_components(iv, "Interval")))
    if K.same(merged, iv):
        raise Unevaluated()
    return merged


@fn("IntervalUnion", area=AREA, status="added")
def _interval_union(*ivs: Expr) -> Expr:
    comps = []
    for iv in ivs:
        comps += _components(iv, "IntervalUnion")
    return _interval(_merge(comps))


@fn("IntervalIntersection", area=AREA, status="added")
def _interval_intersection(*ivs: Expr) -> Expr:
    if not ivs:
        raise ValueError("IntervalIntersection: needs at least one interval")
    cur = _merge(_components(ivs[0], "IntervalIntersection"))
    for iv in ivs[1:]:
        other = _merge(_components(iv, "IntervalIntersection"))
        nxt = []
        for a in cur:
            for b in other:
                lo = b if _lt(a[0], b[0]) else a
                hi = b if _lt(b[1], a[1]) else a
                if not _lt(hi[1], lo[0]):
                    nxt.append((lo[0], hi[1], lo[2], hi[3]))
        cur = _merge(nxt)
    return _interval(cur)


@fn("IntervalMemberQ", area=AREA, status="added",
    notes="False for symbolic x; an Interval x must lie within the union"
          ". Check: reference (differential cases in tests/library).")
def _interval_memberq(iv: Expr, x: Expr) -> Boolean:
    comps = _merge(_components(iv, "IntervalMemberQ"))
    if _head(x, "Interval"):
        try:
            xs = _components(x, "IntervalMemberQ")
        except Unevaluated:
            return False
        return all(any(not _lt(s[0], c[0]) and not _lt(c[1], s[1]) for c in comps) for s in xs)
    try:
        v = _endpoint(x, "IntervalMemberQ")
    except (Unevaluated, ValueError):
        return False
    return any(not _lt(v, c[0]) and not _lt(c[1], v) for c in comps)


# ───────────────────────────────────── discrete calculus ─────────────────────────────────────

def _step_specs(specs, fname):
    """[(sympy var, n, h)] from i | {i, n} | {i, n, h} specifications."""
    out = []
    for s in specs:
        if isinstance(s, Symbol):
            out.append((sp.Symbol(s.name), 1, sp.Integer(1)))
            continue
        if is_list(s) and 2 <= len(s.args) <= 3 and isinstance(s.args[0], Symbol):
            n = nonneg_int(s.args[1], fname, "order")
            h = S(s.args[2]) if len(s.args) == 3 else sp.Integer(1)
            out.append((sp.Symbol(s.args[0].name), n, h))
            continue
        raise ValueError(f"{fname}: invalid step specification {s!r}; use i, {{i, n}} or {{i, n, h}}")
    if not out:
        raise ValueError(f"{fname}: needs a variable specification")
    return out


def _tidy(e, var_list):
    if e.has(sp.factorial, sp.binomial):
        e = sp.combsimp(e).rewrite(sp.factorial) if e.has(sp.factorial) else sp.combsimp(e)
    else:
        e = sp.simplify(e)
    try:
        if all(e.is_polynomial(v) for v in var_list):
            e = sp.expand(e)
    except Exception:
        pass
    return e


@fn("DifferenceDelta", area=AREA, status="added",
    notes="n-th forward difference sum_k (-1)^(n-k) C(n,k) f(i + k h); result simplified (expanded when polynomial)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _difference_delta(f: Expr, *specs: Expr) -> Expr:
    if is_list(f):
        return L(*(_difference_delta(x, *specs) for x in f.args))
    e = sym(f, "DifferenceDelta")
    vs = _step_specs(specs, "DifferenceDelta")
    for (v, n, h) in vs:
        e = sp.Add(*[(-1) ** (n - k) * sp.binomial(n, k) * e.subs(v, v + k * h) for k in range(n + 1)])
    return R(_tidy(e, [v for v, _, _ in vs]))


@fn("DiscreteShift", area=AREA, status="added")
def _discrete_shift(f: Expr, *specs: Expr) -> Expr:
    if is_list(f):
        return L(*(_discrete_shift(x, *specs) for x in f.args))
    e = sym(f, "DiscreteShift")
    for (v, n, h) in _step_specs(specs, "DiscreteShift"):
        e = e.subs(v, v + n * h)
    return R(e)


@fn("DiscreteRatio", area=AREA, status="added",
    notes="iterated ratio r_{k+1}(i) = r_k(i + h) / r_k(i); simplified with combsimp/simplify"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _discrete_ratio(f: Expr, *specs: Expr) -> Expr:
    if is_list(f):
        return L(*(_discrete_ratio(x, *specs) for x in f.args))
    e = sym(f, "DiscreteRatio")
    for (v, n, h) in _step_specs(specs, "DiscreteRatio"):
        for _ in range(n):
            e = sp.simplify(sp.combsimp(e.subs(v, v + h) / e))
    return R(e)


# ───────────────────────────────────── linear recurrences ─────────────────────────────────────

_MAX_TERMS = 10 ** 6


def _companion(c):
    d = len(c)
    return [[c[j] if i == 0 else Fraction(1 if j == i - 1 else 0) for j in range(d)] for i in range(d)]


def _matmul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def _matpow(M, e):
    d = len(M)
    P = [[Fraction(1 if i == j else 0) for j in range(d)] for i in range(d)]
    while e:
        if e & 1:
            P = _matmul(P, M)
        M = _matmul(M, M)
        e >>= 1
    return P


@fn("LinearRecurrence", area=AREA, status="added",
    notes="ker and init must have the same length (Wolfram's 'only the last Length[ker] values are used' case is "
          "refused); {n} gives the n-th term, {nmin, nmax} needs 1 <= nmin <= nmax; matrix kernels are not supported"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _linear_recurrence(ker: ListOf[Expr], init: ListOf[Expr], spec: Expr) -> Expr:
    kl = list_arg(ker, "LinearRecurrence", "kernel", 1)
    il = list_arg(init, "LinearRecurrence", "initial values", 1)
    if any(is_list(x) for x in kl):
        raise NotImplementedError("LinearRecurrence: matrix-valued kernels are not supported")
    if len(il) != len(kl):
        raise ValueError(f"LinearRecurrence: the kernel has length {len(kl)} but {len(il)} initial values were "
                         "given; give exactly Length[ker] initial values")
    single = False
    if is_list(spec):
        if len(spec.args) == 1:
            nmin = nmax = pos_int(spec.args[0], "LinearRecurrence", "term index")
            single = True
        elif len(spec.args) == 2:
            nmin = pos_int(spec.args[0], "LinearRecurrence", "nmin")
            nmax = int_arg(spec.args[1], "LinearRecurrence", "nmax")
            if nmax < nmin:
                return L()
        else:
            raise ValueError("LinearRecurrence: term specification must be n, {n} or {nmin, nmax}")
    else:
        nmin, nmax = 1, nonneg_int(spec, "LinearRecurrence", "number of terms")
        if nmax == 0:
            return L()
    if nmax - nmin + 1 > _MAX_TERMS:
        raise ValueError(f"LinearRecurrence: at most {_MAX_TERMS} terms can be generated")
    fk = [_exact_rational(x) for x in kl]
    fi = [_exact_rational(x) for x in il]
    if all(x is not None for x in fk + fi):
        d = len(fk)
        if nmax > 5000 and nmax - nmin <= 64 and nmin > d:
            M = _companion(fk)
            v = _matpow(M, nmin - d)
            state = list(reversed(fi))                       # a(d), ..., a(1)
            v = [sum(row[j] * state[j] for j in range(d)) for row in v]
            out = []
            for _ in range(nmin, nmax + 1):
                out.append(_out_frac(v[0]))
                v = [sum(c * x for c, x in zip(fk, v))] + v[:-1]
        else:
            seq = list(fi)
            while len(seq) < nmax:
                seq.append(sum(c * seq[-1 - i] for i, c in enumerate(fk)))
            out = [_out_frac(x) for x in seq[nmin - 1:nmax]]
    else:
        if nmax > 20000:
            raise ValueError("LinearRecurrence: symbolic or inexact recurrences are limited to 20000 terms")
        seq = list(il)
        while len(seq) < nmax:
            seq.append(ev(Expr("Plus", *(ev(Expr("Times", c, seq[-1 - i])) for i, c in enumerate(kl)))))
        out = seq[nmin - 1:nmax]
    return out[0] if single else L(*out)


def _berlekamp_massey(s):
    """Minimal connection coefficients c (s_n = sum c_i s_{n-i}) over the rationals."""
    C, B = [Fraction(1)], [Fraction(1)]
    Ldeg, m, b = 0, 1, Fraction(1)
    for n in range(len(s)):
        d = s[n] + sum(C[i] * s[n - i] for i in range(1, Ldeg + 1))
        if d == 0:
            m += 1
            continue
        T = list(C)
        coef = d / b
        if len(C) < len(B) + m:
            C += [Fraction(0)] * (len(B) + m - len(C))
        for i, bi in enumerate(B):
            C[i + m] -= coef * bi
        if 2 * Ldeg <= n:
            Ldeg, B, b, m = n + 1 - Ldeg, T, d, 1
        else:
            m += 1
    C += [Fraction(0)] * (Ldeg + 1 - len(C))
    return Ldeg, [-C[i] for i in range(1, Ldeg + 1)]


@fn("FindLinearRecurrence", area=AREA, status="added",
    notes="exact rational sequences only (Berlekamp-Massey); the minimal recurrence is returned only when the list "
          "holds at least twice its order of terms; stays unevaluated otherwise"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _find_linear_recurrence(lst: ListOf[Expr], dmax: Expr = None) -> ListOf[Expr]:
    items = list_arg(lst, "FindLinearRecurrence", "sequence", 1)
    vals = [_exact_rational(x) for x in items]
    if any(v is None for v in vals):
        if any(has_float(x) for x in items):
            raise NotImplementedError("FindLinearRecurrence: only exact rational sequences are supported")
        raise Unevaluated()
    if dmax is not None:
        dmax = pos_int(dmax, "FindLinearRecurrence", "maximum order")
    if all(v == 0 for v in vals):
        raise ValueError("FindLinearRecurrence: the sequence is identically zero")
    order, c = _berlekamp_massey(vals)
    if order == 0 or 2 * order > len(vals) or (dmax is not None and order > dmax):
        raise Unevaluated()
    while c and c[-1] == 0:
        c.pop()
    if len(c) != order:
        raise Unevaluated()   # degenerate (leading zeros) — no proper order-d kernel
    return L(*(_out_frac(x) for x in c))


# ───────────────────────────────────── RecurrenceTable ─────────────────────────────────────

def _flat_eqs(e):
    if is_list(e) or _head(e, "And"):
        return [y for x in e.args for y in _flat_eqs(x)]
    if _head(e, "Equal", 2):
        return [e]
    raise ValueError(f"RecurrenceTable: {e!r} is not an equation")


@fn("RecurrenceTable", area=AREA, status="added",
    notes="explicit recurrences: each equation must contain exactly one term of maximal shift, which is solved "
          "for; initial conditions are f[k] == value; iterator {n, nmax} (from 1) or {n, nmin, nmax}"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _recurrence_table(eqns: Expr, expr: Expr, it: Expr) -> ListOf[Expr]:
    fname = "RecurrenceTable"
    if not (is_list(it) and len(it.args) in (2, 3) and isinstance(it.args[0], Symbol)):
        raise ValueError(f"{fname}: iterator must be {{n, nmax}} or {{n, nmin, nmax}}")
    nvar = it.args[0]
    nmin = int_arg(it.args[1], fname, "nmin") if len(it.args) == 3 else 1
    nmax = int_arg(it.args[-1], fname, "nmax")
    if nmax - nmin > 100000:
        raise ValueError(f"{fname}: at most 100000 steps are supported")
    n = sp.Symbol(nvar.name)
    eqs = _flat_eqs(eqns)
    known, rules, funcs = {}, [], set()
    for e in eqs:
        lhs, rhs = S(e.args[0]), S(e.args[1])
        expr_s = lhs - rhs
        apps = [a for a in expr_s.atoms(sp.core.function.AppliedUndef)]
        if not apps:
            raise ValueError(f"{fname}: equation {e!r} contains no unknown sequence")
        if n not in expr_s.free_symbols:
            # initial condition f[k] == value
            if len(apps) != 1 or not isinstance(lhs, sp.core.function.AppliedUndef) or len(lhs.args) != 1 \
                    or not lhs.args[0].is_Integer or rhs.atoms(sp.core.function.AppliedUndef):
                raise ValueError(f"{fname}: initial condition {e!r} must have the form f[k] == value")
            known[(lhs.func.__name__, int(lhs.args[0]))] = rhs
            funcs.add(lhs.func.__name__)
            continue
        shifts = []
        for a in apps:
            if len(a.args) != 1:
                raise NotImplementedError(f"{fname}: only one-index sequences are supported")
            k = sp.expand(a.args[0] - n)
            if not k.is_Integer:
                raise NotImplementedError(f"{fname}: term {a} is not of the form f[n + k]")
            shifts.append((int(k), a))
            funcs.add(a.func.__name__)
        top = max(k for k, _ in shifts)
        tops = {a for k, a in shifts if k == top}
        if len(tops) != 1:
            raise NotImplementedError(f"{fname}: equation {e!r} must contain exactly one term of highest shift")
        target = tops.pop()
        sol = sp.solve(expr_s, target, dict=False)
        if len(sol) != 1:
            raise NotImplementedError(f"{fname}: cannot solve {e!r} uniquely for {target}")
        # normalise so the rule defines f[n] from earlier terms
        rhs_n = sol[0].subs(n, n - top)
        rules.append((target.func.__name__, target.func, rhs_n))
    if not rules:
        raise ValueError(f"{fname}: no recurrence equation given")
    defined = {r[0] for r in rules}
    if len(defined) != len(rules):
        raise NotImplementedError(f"{fname}: each sequence needs exactly one defining recurrence")
    if not known:
        raise ValueError(f"{fname}: no initial conditions given")
    start = min(k for (_, k) in known)

    def value_of(app, at):
        idx = app.args[0].subs(n, at)
        key = (app.func.__name__, int(idx))
        if key not in known:
            raise KeyError(key)
        return known[key]

    pending_rules = rules
    for idx in range(start, nmax + 1):
        todo = [r for r in pending_rules if (r[0], idx) not in known]
        progress = True
        while todo and progress:
            progress = False
            for r in list(todo):
                name, f, rhs = r
                apps = rhs.atoms(sp.core.function.AppliedUndef)
                try:
                    sub = {a: value_of(a, idx) for a in apps}
                except KeyError:
                    continue
                known[(name, idx)] = rhs.subs(sub).subs(n, idx)
                todo.remove(r)
                progress = True
        if todo:
            if idx < nmin:
                continue
            raise ValueError(f"{fname}: insufficient initial conditions to compute {todo[0][0]}[{idx}]")

    def at(x, idx):
        if isinstance(x, Symbol) and x.name in funcs:
            key = (x.name, idx)
            if key not in known:
                raise ValueError(f"{fname}: {x.name}[{idx}] is not determined by the initial conditions")
            return R(known[key])
        if is_list(x):
            return L(*(at(y, idx) for y in x.args))
        e = sym(x, fname).subs(n, idx)
        sub = {}
        for a in e.atoms(sp.core.function.AppliedUndef):
            if a.func.__name__ in funcs:
                if len(a.args) != 1 or not a.args[0].is_Integer or (a.func.__name__, int(a.args[0])) not in known:
                    raise ValueError(f"{fname}: {a} is not determined by the recurrence")
                sub[a] = known[(a.func.__name__, int(a.args[0]))]
        return R(e.subs(sub))

    return L(*(at(expr, idx) for idx in range(nmin, nmax + 1)))


# ───────────────────────────────────── continued fractions ─────────────────────────────────────

def _cf_rational(fr, limit=None):
    out = []
    while limit is None or len(out) < limit:
        a = fr.numerator // fr.denominator
        out.append(a)
        fr -= a
        if fr == 0:
            break
        fr = 1 / fr
    return out


def _radical_form(v):
    return v.subs(sp.GoldenRatio, (1 + sp.sqrt(5)) / 2) if v.has(sp.GoldenRatio) else v


def _exact_real(v):
    return isinstance(v, sp.Basic) and v.is_number and not v.has(sp.Float) and v.is_real is True


def _algebraic_radical(v):
    """True when v is built only from rationals with +, *, and rational powers (so it is algebraic)."""
    if v.is_Rational:
        return True
    if v.is_Add or v.is_Mul:
        return all(_algebraic_radical(a) for a in v.args)
    if v.is_Pow:
        return _algebraic_radical(v.base) and v.exp.is_Rational
    if v == sp.I:
        return True
    return False


def _quadratic_irrational(v):
    if not _exact_real(v) or v.is_Rational:
        return False
    if not _algebraic_radical(v):
        if v.is_algebraic is False:
            return False
        if v.is_algebraic is None and (v.has(sp.pi) or v.has(sp.E)) and v.is_transcendental:
            return False
        raise NotImplementedError(f"cannot decide whether {v} is a quadratic irrational")
    return sp.degree(sp.minimal_polynomial(v, sp.Dummy("x"))) == 2


def _cf_numeric(v, n):
    """First n terms of the continued fraction of an exact irrational real v, certified by agreement at two
    precisions."""
    dps = max(30, 2 * n + 20)
    for _ in range(12):
        a = _cf_rational(Fraction(str(sp.N(v, dps))), n + 2)
        b = _cf_rational(Fraction(str(sp.N(v, 2 * dps))), n + 2)
        k = 0
        while k < min(len(a), len(b)) - 1 and a[k] == b[k]:
            k += 1
        if k >= n:
            return a[:n]
        dps *= 2
    raise ValueError(f"ContinuedFraction: cannot certify {n} terms of {v}")


def _cf_inexact(x):
    """Terms of the exact binary value of an inexact real, stopping at the first convergent that agrees with it to
    within its precision (so 0.1 gives {0, 10} and 3.25 gives {3, 4})."""
    from ..precision import input_precision
    v = S(x)
    p = input_precision([x])
    bits = 53 if p in ("m", None) else int(p)
    fr = Fraction(float(v)) if p in ("m", None) else Fraction(str(sp.Float(v, int(bits * 0.30103) + 3)))
    eps = abs(fr) * Fraction(1, 2 ** bits) if fr != 0 else Fraction(0)
    terms = _cf_rational(fr, 10 ** 4)
    out = []
    pm2, pm1, qm2, qm1 = 0, 1, 1, 0
    for a in terms:
        out.append(a)
        pn, qn = a * pm1 + pm2, a * qm1 + qm2
        if abs(fr - Fraction(pn, qn)) <= eps:
            break
        pm2, pm1, qm2, qm1 = pm1, pn, qm1, qn
    if len(out) > 1 and out[-1] == 1:
        out[-2:] = [out[-2] + 1]           # canonical form never ends in 1 (same value)
    return out


def _from_terms(terms):
    acc = Fraction(terms[-1])
    for t in reversed(terms[:-1]):
        acc = t + 1 / acc
    return acc


def _periodic_out(terms):
    """sympy continued_fraction output -> Wolfram list ({..., {period}})."""
    out = []
    for t in terms:
        out.append(L(*[int(z) for z in t]) if isinstance(t, list) else int(t))
    return L(*out)


@fn("ContinuedFraction", area=AREA, status="added",
    notes="rationals exactly; quadratic irrationals as {a1, ..., {period}}; other exact reals need the term count "
          "(certified at two precisions); inexact numbers give only the terms fixed by their precision"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _continued_fraction(x: Expr, n: Expr = None) -> Expr:
    fname = "ContinuedFraction"
    if n is not None:
        n = nonneg_int(n, fname, "number of terms")
    fr = _exact_rational(x)
    if fr is not None:
        return L(*_cf_rational(fr, n if n is not None else None)) if n != 0 else L()
    if has_float(x):
        v = S(x)
        if not (isinstance(v, sp.Basic) and v.is_number):
            raise Unevaluated()
        if v.is_real is False:
            raise ValueError(f"{fname}: {x!r} is not a real number")
        terms = _cf_inexact(x)
        return L(*(terms[:n] if n is not None else terms))
    v = S(x)
    if not isinstance(v, sp.Basic) or v.free_symbols or not v.is_number:
        raise Unevaluated()
    if v.is_real is not True:
        raise ValueError(f"{fname}: {x!r} is not a real number")
    v = _radical_form(v)
    if _algebraic_radical(v) and sp.degree(sp.minimal_polynomial(v, sp.Dummy("x"))) == 2:
        terms = list(sp.continued_fraction(v))
        if n is None:
            return _periodic_out(terms)
        flat = [int(t) for t in terms if not isinstance(t, list)]
        period = next((t for t in terms if isinstance(t, list)), [])
        while len(flat) < n:
            flat += [int(z) for z in period]
        return L(*flat[:n])
    if n is None:
        raise ValueError(f"{fname}: {x!r} has an infinite non-periodic continued fraction; give the number of terms "
                         "as ContinuedFraction[x, n]")
    return L(*_cf_numeric(v, n)) if n else L()


@fn("FromContinuedFraction", area=AREA, status="added",
    notes="a trailing list is a period (gives the quadratic irrational); symbolic terms allowed for finite lists"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _from_continued_fraction(lst: ListOf[Expr]) -> Expr:
    items = list_arg(lst, "FromContinuedFraction", "continued fraction", 1)
    if is_list(items[-1]):
        if any(is_list(x) for x in items[:-1]):
            raise ValueError("FromContinuedFraction: only the last element may be a (periodic) list")
        pre = [int_arg(x, "FromContinuedFraction", "term") for x in items[:-1]]
        per = [int_arg(x, "FromContinuedFraction", "term") for x in items[-1].args]
        if not per:
            raise ValueError("FromContinuedFraction: the period must be non-empty")
        if any(t <= 0 for t in per) or any(t <= 0 for t in pre[1:]):
            raise ValueError("FromContinuedFraction: periodic continued fractions need positive terms after the first")
        return R(sp.radsimp(sp.nsimplify(sp.continued_fraction_reduce(pre + [per]))))
    acc = None
    for t in reversed(items):
        acc = t if acc is None else ev(Expr("Plus", t, Expr("Power", acc, -1)))
    return acc


@fn("Convergents", area=AREA, status="added",
    notes="Convergents[list] for finite term lists (symbolic allowed); Convergents[x, n] via ContinuedFraction"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _convergents(x: Expr, n: Expr = None) -> ListOf[Expr]:
    if is_list(x):
        if n is not None:
            raise ValueError("Convergents: give either a term list or a number and a count")
        terms = list(x.args)
        if any(is_list(t) for t in terms):
            raise NotImplementedError("Convergents: periodic term lists are not supported; give ContinuedFraction[x, n]")
    else:
        cf = _continued_fraction(x, n) if n is not None else _continued_fraction(x)
        terms = list(cf.args)
        if any(is_list(t) for t in terms):
            raise ValueError("Convergents: the continued fraction is infinite; give the number of convergents")
    vals = [S(t) for t in terms]
    exact = all(isinstance(v, sp.Rational) for v in vals)
    out = []
    pm2, pm1, qm2, qm1 = sp.Integer(0), sp.Integer(1), sp.Integer(1), sp.Integer(0)
    for a in vals:
        p = a * pm1 + pm2
        q = a * qm1 + qm2
        out.append(p / q if exact else sp.factor(sp.together(p / q)))
        pm2, pm1, qm2, qm1 = pm1, p, qm1, q
    return L(*(R(v) for v in out))


@fn("QuadraticIrrationalQ", area=AREA, status="added",
    notes="True for irrational real roots of rational quadratics; False for rationals, inexact and symbolic input "
          "and non-real numbers; raises when algebraicity cannot be decided"
          ". Check: reference (differential cases in tests/library).")
def _quadratic_irrational_q(x: Expr) -> Boolean:
    if isinstance(x, (str, bool)) or has_float(x):
        return False
    try:
        v = S(x)
    except Exception:
        return False
    if not isinstance(v, sp.Basic) or not v.is_number or v.free_symbols:
        return False
    if v.is_real is not True:
        return False
    return _quadratic_irrational(_radical_form(v))


@fn("FareySequence", area=AREA, status="added")
def _farey(n: PosInteger, k: Expr = None) -> Expr:
    n = pos_int(n, "FareySequence", "order")
    if n > 3000:
        raise ValueError("FareySequence: order is limited to 3000")
    a, b, c, d = 0, 1, 1, n
    seq = [Fraction(0, 1)]
    while c <= n:
        kk = (n + b) // d
        a, b, c, d = c, d, kk * c - a, kk * d - b
        seq.append(Fraction(a, b))
    if k is not None:
        k = int_arg(k, "FareySequence", "index")
        if k == 0 or abs(k) > len(seq):
            raise ValueError(f"FareySequence: index {k} is out of range 1..{len(seq)}")
        return _out_frac(seq[k - 1 if k > 0 else k])
    return L(*(_out_frac(f) for f in seq))


# ───────────────────────────────────────── bit operations ─────────────────────────────────────────

def _bint(x, fname, what="argument"):
    return int_arg(x, fname, what)


@fn("BitAnd", area=AREA, status="added", listable=True)
def _bitand(*xs: Expr) -> Expr:
    r = -1
    for x in xs:
        r &= _bint(x, "BitAnd")
    return r


@fn("BitOr", area=AREA, status="added", listable=True)
def _bitor(*xs: Expr) -> Expr:
    r = 0
    for x in xs:
        r |= _bint(x, "BitOr")
    return r


@fn("BitXor", area=AREA, status="added", listable=True)
def _bitxor(*xs: Expr) -> Expr:
    r = 0
    for x in xs:
        r ^= _bint(x, "BitXor")
    return r


@fn("BitNot", area=AREA, status="added", listable=True)
def _bitnot(x: Integer) -> Expr:
    return -1 - _bint(x, "BitNot")


def _shift_right(n, k):
    if k < 0:
        return n << -k
    return n >> k if n >= 0 else -((-n) >> k)


@fn("BitShiftLeft", area=AREA, status="added", listable=True, notes="negative shifts shift right (IntegerPart)"
                                                                    ". Check: exact (exact input gives exact "
                                                                    "results), reference (differential cases in "
                                                                    "tests/library).")
def _bitshiftleft(n: Integer, k: Integer = 1) -> Expr:
    return _shift_right(_bint(n, "BitShiftLeft"), -_bint(k, "BitShiftLeft", "shift"))


@fn("BitShiftRight", area=AREA, status="added", listable=True, notes="IntegerPart[n / 2^k] (rounds toward zero)"
                                                                     ". Check: exact (exact input gives exact "
                                                                     "results), reference (differential cases in "
                                                                     "tests/library).")
def _bitshiftright(n: Integer, k: Integer = 1) -> Expr:
    return _shift_right(_bint(n, "BitShiftRight"), _bint(k, "BitShiftRight", "shift"))


@fn("BitLength", area=AREA, status="added", listable=True,
    notes="for negative n, the length of BitNot[n] (two's complement without the sign)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _bitlength(n: Integer) -> Expr:
    n = _bint(n, "BitLength")
    return n.bit_length() if n >= 0 else (-1 - n).bit_length()


def _bitpos(k, fname):
    k = _bint(k, fname, "bit position")
    if k < 0:
        raise ValueError(f"{fname}: bit position {k} must be non-negative")
    return k


@fn("BitGet", area=AREA, status="added", listable=True)
def _bitget(n: Integer, k: Integer) -> Expr:
    return (_bint(n, "BitGet") >> _bitpos(k, "BitGet")) & 1


@fn("BitSet", area=AREA, status="added", listable=True)
def _bitset(n: Integer, k: Integer) -> Expr:
    return _bint(n, "BitSet") | (1 << _bitpos(k, "BitSet"))


@fn("BitClear", area=AREA, status="added", listable=True)
def _bitclear(n: Integer, k: Integer) -> Expr:
    return _bint(n, "BitClear") & ~(1 << _bitpos(k, "BitClear"))


@fn("BitFlip", area=AREA, status="added", listable=True)
def _bitflip(n: Integer, k: Integer) -> Expr:
    return _bint(n, "BitFlip") ^ (1 << _bitpos(k, "BitFlip"))


@fn("IntegerExponent", area=AREA, status="added", listable=True,
    notes="default base 10; IntegerExponent[0, b] = Infinity; negative n uses |n|"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _integer_exponent(n: Integer, b: Integer = 10) -> Expr:
    n = int_arg(n, "IntegerExponent")
    b = int_arg(b, "IntegerExponent", "base")
    if b < 2:
        raise ValueError(f"IntegerExponent: base {b} must be an integer greater than 1")
    if n == 0:
        return Symbol("Infinity")
    n, k = abs(n), 0
    while n % b == 0:
        n //= b
        k += 1
    return k


# ─────────────────────────────── cellular automata & substitution systems ───────────────────────────────

def _ca_rule(rule, fname):
    """(k, r, table) with table[tuple(neighbourhood)] -> new value."""
    if _intq(rule):
        num, k, r = int(rule), 2, 1
    elif is_list(rule) and rule.args and all(_head(x, "Rule", 2) for x in rule.args):
        table = {}
        width = None
        for x in rule.args:
            lhs = x.args[0]
            if not (is_list(lhs) and all(_intq(v) for v in lhs.args)) or not _intq(x.args[1]):
                raise NotImplementedError(f"{fname}: only explicit rules {{l, c, r}} -> v with integer cells are "
                                          "supported")
            if width is None:
                width = len(lhs.args)
            if len(lhs.args) != width or width % 2 == 0:
                raise ValueError(f"{fname}: all rule neighbourhoods must have the same odd length")
            table[tuple(int(v) for v in lhs.args)] = int(x.args[1])
        k = max(max(max(t) for t in table), max(table.values())) + 1
        return max(k, 2), width // 2, table
    elif is_list(rule) and len(rule.args) in (2, 3) and all(_intq(v) for v in rule.args):
        num, k = int(rule.args[0]), int(rule.args[1])
        r = int(rule.args[2]) if len(rule.args) == 3 else 1
    else:
        raise NotImplementedError(f"{fname}: rule must be an integer, {{n, k}}, {{n, k, r}} or a list of explicit "
                                  "rules (totalistic and other rule forms are not supported)")
    if k < 2 or r < 0:
        raise ValueError(f"{fname}: need k >= 2 colours and range r >= 0")
    w = 2 * r + 1
    if num < 0 or num >= k ** (k ** w):
        raise ValueError(f"{fname}: rule number {num} is out of range for k = {k}, r = {r}")
    table = {}
    for idx in range(k ** w):
        neigh = []
        v = idx
        for _ in range(w):
            neigh.append(v % k)
            v //= k
        table[tuple(reversed(neigh))] = (num // k ** idx) % k
    return k, r, table


def _ca_step_cyclic(row, r, table, fname):
    n = len(row)
    out = []
    for i in range(n):
        key = tuple(row[(i + j) % n] for j in range(-r, r + 1))
        if key not in table:
            raise ValueError(f"{fname}: no rule given for neighbourhood {list(key)}")
        out.append(table[key])
    return out


@fn("CellularAutomaton", area=AREA, status="added",
    notes="1D rules (number, {n, k}, {n, k, r} or explicit neighbourhood rules); init a cyclic list or "
          "{{cells}, background}; CellularAutomaton[rule, init] is one step, [rule, init, t] gives t + 1 rows "
          "(the {{cells}, bg} form uses width m + 2 r t, Wolfram's default region)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _cellular_automaton(rule: Expr, init: Expr, t: Expr = None) -> ListOf[Expr]:
    fname = "CellularAutomaton"
    k, r, table = _ca_rule(rule, fname)
    steps = 1 if t is None else nonneg_int(t, fname, "number of steps")
    if steps > 5000:
        raise ValueError(f"{fname}: at most 5000 steps are supported")
    if is_list(init) and len(init.args) == 2 and is_list(init.args[0]) and _intq(init.args[1]):
        cells = [int_arg(v, fname, "cell") for v in init.args[0].args]
        bg = int(init.args[1])
        pad = r * steps
        row = [bg] * pad + cells + [bg] * pad
        rows = [row]
        cur_bg = bg
        # evolve on a window extended by r on each side per step so edge cells see the true background
        for _ in range(steps):
            ext = [cur_bg] * r + rows[-1] + [cur_bg] * r
            new = []
            for i in range(r, len(ext) - r):
                key = tuple(ext[i - r:i + r + 1])
                if key not in table:
                    raise ValueError(f"{fname}: no rule given for neighbourhood {list(key)}")
                new.append(table[key])
            nb = tuple([cur_bg] * (2 * r + 1))
            if nb not in table:
                raise ValueError(f"{fname}: no rule given for the background neighbourhood {list(nb)}")
            cur_bg = table[nb]
            rows.append(new)
    else:
        row = [int_arg(v, fname, "cell") for v in list_arg(init, fname, "initial condition", 1)]
        rows = [row]
        for _ in range(steps):
            rows.append(_ca_step_cyclic(rows[-1], r, table, fname))
    if any(v < 0 or v >= k for row in rows for v in row) and _intq(rule):
        raise ValueError(f"{fname}: cell values must be between 0 and {k - 1}")
    if t is None:
        return L(*rows[1])
    return L(*(L(*row) for row in rows))


@fn("SubstitutionSystem", area=AREA, status="added",
    notes="single-element rules on strings (character -> string) or lists (element -> list); "
          "SubstitutionSystem[rule, init, t] gives t + 1 states; neighbourhood rules are not supported"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _substitution_system(rules: Expr, init: Expr, t: Expr = None) -> Expr:
    fname = "SubstitutionSystem"
    rl = list(rules.args) if is_list(rules) else [rules]
    if not rl or not all(_head(x, "Rule", 2) for x in rl):
        raise NotImplementedError(f"{fname}: only lists of replacement rules are supported")
    steps = 1 if t is None else nonneg_int(t, fname, "number of steps")
    if isinstance(init, str):
        table = {}
        for x in rl:
            a, b = x.args
            if not (isinstance(a, str) and isinstance(b, str)) or len(a) != 1:
                raise NotImplementedError(f"{fname}: string rules must map single characters to strings")
            table[a] = b
        states = [init]
        for _ in range(steps):
            states.append("".join(table.get(c, c) for c in states[-1]))
            if len(states[-1]) > 10 ** 7:
                raise ValueError(f"{fname}: state grew beyond 10^7 elements")
    elif is_list(init):
        table = []
        for x in rl:
            a, b = x.args
            if not is_list(b):
                raise NotImplementedError(f"{fname}: list rules must map an element to a list")
            table.append((a, list(b.args)))
        states = [list(init.args)]
        for _ in range(steps):
            nxt = []
            for e in states[-1]:
                for a, b in table:
                    if K.same(a, e):
                        nxt += b
                        break
                else:
                    nxt.append(e)
            states.append(nxt)
            if len(nxt) > 10 ** 6:
                raise ValueError(f"{fname}: state grew beyond 10^6 elements")
        states = [L(*s) for s in states]
    else:
        raise ValueError(f"{fname}: initial condition must be a string or a list")
    if t is None:
        return states[1]
    return L(*states)


@fn("ThueMorse", area=AREA, status="added", listable=True)
def _thue_morse(n: NonNegInteger) -> Expr:
    n = nonneg_int(n, "ThueMorse")
    return bin(n).count("1") % 2


@fn("RudinShapiro", area=AREA, status="added", listable=True, notes="(-1)^(number of possibly overlapping 11 blocks)"
                                                                    ". Check: exact (exact input gives exact "
                                                                    "results), reference (differential cases in "
                                                                    "tests/library).")
def _rudin_shapiro(n: NonNegInteger) -> Expr:
    n = nonneg_int(n, "RudinShapiro")
    return -1 if bin(n & (n >> 1)).count("1") % 2 else 1


# ─────────────────────────────── integer predicates and representations ───────────────────────────────

_MERSENNE_EXPONENTS = [
    2, 3, 5, 7, 13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941, 11213,
    19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787, 1398269, 2976221, 3021377,
    6972593, 13466917, 20996011, 24036583, 25964951, 30402457, 32582657, 37156667, 42643801, 43112609, 57885161]
_MERSENNE_SET = set(_MERSENNE_EXPONENTS)
_FACTOR_DIGITS = 60


def _factorint(n, fname):
    if len(str(abs(n))) > _FACTOR_DIGITS and not sp.isprime(abs(n)):
        raise NotImplementedError(f"{fname}: integers with more than {_FACTOR_DIGITS} digits are not factored")
    return sp.factorint(abs(n))


@fn("SquareFreeQ", area=AREA, status="added",
    notes="integers (0 is not square-free) and polynomials with integer content 1 (via sqf_list)"
          ". Check: reference (differential cases in tests/library).")
def _square_free_q(x: Expr, *rest: Expr) -> Boolean:
    if rest:
        raise NotImplementedError("SquareFreeQ: the GaussianIntegers / Modulus forms are not supported")
    if _intq(x):
        n = int(x)
        if n == 0:
            return False
        return all(e == 1 for e in _factorint(n, "SquareFreeQ").values())
    if isinstance(x, (float, bool, str)) or has_float(x):
        return False
    v = S(x)
    if not isinstance(v, sp.Basic):
        return False
    if v.is_number:
        if v.is_Rational:
            raise NotImplementedError("SquareFreeQ: rational numbers are not supported")
        return False
    if not v.is_polynomial(*v.free_symbols):
        return False
    p = sp.Poly(v, *sorted(v.free_symbols, key=str))
    if not all(c.is_Integer for c in p.coeffs()):
        raise NotImplementedError("SquareFreeQ: only polynomials with integer coefficients are supported")
    content, factors = sp.sqf_list(p)
    if abs(content) != 1:
        raise NotImplementedError("SquareFreeQ: polynomials with a non-unit integer content are not supported")
    return all(m == 1 for _, m in factors)


@fn("PerfectNumberQ", area=AREA, status="added",
    notes="even perfect numbers via the 48 known Mersenne prime exponents; odd n beyond 10^1500 raise"
          ". Check: reference (differential cases in tests/library).")
def _perfect_number_q(x: Expr) -> Boolean:
    if not _intq(x):
        return False
    n = int(x)
    if n <= 0:
        return False
    if n % 2:
        if n.bit_length() > 4983:          # ~10^1500: no odd perfect number below (Ochem & Rao)
            raise NotImplementedError("PerfectNumberQ: odd perfect numbers above 10^1500 cannot be ruled out")
        return False
    v = (n & -n).bit_length() - 1
    p = v + 1
    if n != (1 << v) * ((1 << p) - 1):
        return False
    if p <= _MERSENNE_EXPONENTS[-1]:
        return p in _MERSENNE_SET
    raise NotImplementedError("PerfectNumberQ: beyond the verified range of Mersenne primes")


@fn("PerfectNumber", area=AREA, status="added", listable=True, notes="k <= 48 (verified Mersenne primes)"
                                                                     ". Check: reference (differential cases in "
                                                                     "tests/library).")
def _perfect_number(k: PosInteger) -> Expr:
    k = pos_int(k, "PerfectNumber", "index")
    if k > len(_MERSENNE_EXPONENTS):
        raise NotImplementedError(f"PerfectNumber: only the first {len(_MERSENNE_EXPONENTS)} perfect numbers are "
                                  "known to be consecutive")
    p = _MERSENNE_EXPONENTS[k - 1]
    return (1 << (p - 1)) * ((1 << p) - 1)


@fn("MersennePrimeExponent", area=AREA, status="added", listable=True, notes="k <= 48"
                                                                             ". Check: exact (exact input gives "
                                                                             "exact results), reference "
                                                                             "(differential cases in tests/library).")
def _mersenne_exponent(k: PosInteger) -> Expr:
    k = pos_int(k, "MersennePrimeExponent", "index")
    if k > len(_MERSENNE_EXPONENTS):
        raise NotImplementedError(f"MersennePrimeExponent: only the first {len(_MERSENNE_EXPONENTS)} exponents are "
                                  "known to be consecutive")
    return _MERSENNE_EXPONENTS[k - 1]


@fn("MersennePrimeExponentQ", area=AREA, status="added", listable=True,
    notes="decided up to 57885161 and for composite n; larger primes raise"
          ". Check: reference (differential cases in tests/library).")
def _mersenne_exponent_q(x: Expr) -> Boolean:
    if not _intq(x):
        return False
    n = int(x)
    if n <= _MERSENNE_EXPONENTS[-1]:
        return n in _MERSENNE_SET
    if not sp.isprime(n):
        return False
    raise NotImplementedError(f"MersennePrimeExponentQ: cannot decide primality of 2^{n} - 1")


@fn("CompositeQ", area=AREA, status="added", listable=True, notes="integers; CompositeQ[-n] == CompositeQ[n]"
                                                                  ". Check: reference (differential cases in "
                                                                  "tests/library).")
def _composite_q(x: Expr) -> Boolean:
    if not _intq(x):
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, bool):
            v = S(x) if not isinstance(x, Symbol) else None
            if v is not None and isinstance(v, sp.Basic) and v.is_number and not v.is_real and \
                    sp.re(v).is_Integer and sp.im(v).is_Integer:
                raise NotImplementedError("CompositeQ: Gaussian integers are not supported")
        return False
    n = abs(int(x))
    return n > 1 and not sp.isprime(n)


@fn("PrimeNu", area=AREA, status="added", listable=True, notes="number of distinct prime factors of |n|; 0 raises"
                                                               ". Check: exact (exact input gives exact results), "
                                                               "reference (differential cases in tests/library).")
def _prime_nu(x: Integer) -> Integer:
    n = int_arg(x, "PrimeNu")
    if n == 0:
        raise ValueError("PrimeNu: undefined for 0")
    return len(_factorint(n, "PrimeNu"))


@fn("PrimeOmega", area=AREA, status="added", listable=True,
    notes="prime factors of |n| counted with multiplicity; 0 raises"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _prime_omega(x: Integer) -> Expr:
    n = int_arg(x, "PrimeOmega")
    if n == 0:
        raise ValueError("PrimeOmega: undefined for 0")
    return sum(_factorint(n, "PrimeOmega").values())


@fn("Divisible", area=AREA, status="added", listable=True,
    notes="exact numbers: n/m an integer (Gaussian integer for complex input)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _divisible(n: Expr, m: Expr) -> Boolean:
    a, b = S(n), S(m)
    for v in (a, b):
        if not isinstance(v, sp.Basic) or v.free_symbols or not v.is_number:
            raise Unevaluated()
        if v.has(sp.Float):
            raise ValueError("Divisible: arguments must be exact numbers")
    if b == 0:
        raise ValueError("Divisible: division by zero")
    q = sp.nsimplify(sp.expand(a / b))
    re, im = sp.re(q), sp.im(q)
    if re.is_Rational is None or im.is_Rational is None or not (re.is_Rational and im.is_Rational):
        raise ValueError("Divisible: arguments must be exact rational or Gaussian rational numbers")
    return bool(re.is_Integer and im.is_Integer)


@fn("CoprimeQ", area=AREA, status="added", listable=True, notes="pairwise coprime integers"
                                                                ". Check: reference (differential cases in "
                                                                "tests/library).")
def _coprime_q(*xs: Expr) -> Boolean:
    if len(xs) < 2:
        raise ValueError("CoprimeQ: needs at least two integers")
    ns = [int_arg(x, "CoprimeQ") for x in xs]
    return all(math.gcd(ns[i], ns[j]) == 1 for i in range(len(ns)) for j in range(i + 1, len(ns)))


@fn("NumberExpand", area=AREA, status="added",
    notes="integers only: digit values d b^k, sign carried by every term; length pads with leading zeros"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _number_expand(x: Integer, b: Integer = 10, length: Expr = None) -> ListOf[Expr]:
    n = int_arg(x, "NumberExpand", "number")
    b = int_arg(b, "NumberExpand", "base")
    if b < 2:
        raise ValueError("NumberExpand: base must be at least 2")
    sign = -1 if n < 0 else 1
    m = abs(n)
    digits = []
    while m:
        digits.append(m % b)
        m //= b
    if not digits:
        digits = [0]
    terms = [sign * d * b ** k for k, d in enumerate(digits)][::-1]
    if length is not None:
        length = nonneg_int(length, "NumberExpand", "length")
        terms = ([0] * max(0, length - len(terms)) + terms)[-length:] if length else []
    return L(*terms)


@fn("NumberDigit", area=AREA, status="added", listable=True,
    notes="digit of |x| multiplying b^n; exact reals only (irrationals certified at two precisions)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _number_digit(x: Expr, n: Integer, b: Integer = 10) -> Expr:
    n = int_arg(n, "NumberDigit", "position")
    b = int_arg(b, "NumberDigit", "base")
    if b < 2:
        raise ValueError("NumberDigit: base must be at least 2")
    fr = _exact_rational(x)
    if fr is None:
        v = S(x)
        if not isinstance(v, sp.Basic) or v.free_symbols or not v.is_number:
            raise Unevaluated()
        if v.has(sp.Float):
            raise NotImplementedError("NumberDigit: inexact numbers are not supported")
        if v.is_real is not True:
            raise ValueError("NumberDigit: the number must be real")
        v = abs(v)
        scaled = v / sp.Integer(b) ** n
        dps = 30 + abs(n) * 2
        for _ in range(8):
            f1 = sp.floor(sp.N(scaled, dps))
            lo, hi = sp.N(scaled, dps) - sp.Rational(1, 10 ** (dps // 2)), sp.N(scaled, dps) + sp.Rational(1, 10 ** (dps // 2))
            if sp.floor(lo) == sp.floor(hi):
                return int(f1) % b
            dps *= 2
        raise ValueError("NumberDigit: cannot certify the digit")
    fr = abs(fr)
    q = fr / Fraction(b) ** n if n >= 0 else fr * b ** (-n)
    return int(q.numerator // q.denominator) % b


@fn("SquaresR", area=AREA, status="added", listable=True, notes="d <= 200, n <= 10^5 (exact counting)"
                                                                ". Check: exact (exact input gives exact results), "
                                                                "reference (differential cases in tests/library).")
def _squares_r(d: PosInteger, n: Integer) -> Expr:
    d = pos_int(d, "SquaresR", "dimension")
    n = int_arg(n, "SquaresR")
    if n < 0:
        return 0
    if n > 10 ** 5 or d > 200:
        raise NotImplementedError("SquaresR: limited to d <= 200 and n <= 10^5")
    one = [0] * (n + 1)
    k = 0
    while k * k <= n:
        one[k * k] += 1 if k == 0 else 2
        k += 1
    nz = [i for i, c in enumerate(one) if c]
    cur = one[:]
    for _ in range(d - 1):
        nxt = [0] * (n + 1)
        for s, c in enumerate(cur):
            if c:
                for i in nz:
                    if s + i > n:
                        break
                    nxt[s + i] += c * one[i]
        cur = nxt
    return cur[n]


@fn("PowersRepresentations", area=AREA, status="added",
    notes="non-decreasing k-tuples of non-negative integers whose p-th powers sum to n, in lexicographic order"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _powers_representations(n: NonNegInteger, k: PosInteger, p: PosInteger) -> ListOf[Expr]:
    n = nonneg_int(n, "PowersRepresentations")
    k = pos_int(k, "PowersRepresentations", "number of terms")
    p = pos_int(p, "PowersRepresentations", "power")
    if k > 12 or n > 10 ** 12:
        raise NotImplementedError("PowersRepresentations: limited to k <= 12 and n <= 10^12")
    out = []

    def rec(prefix, remaining, lo, left):
        if left == 0:
            if remaining == 0:
                out.append(L(*prefix))
            return
        x = lo
        while left * x ** p <= remaining:
            rec(prefix + [x], remaining - x ** p, x, left - 1)
            x += 1
            if len(out) > 10 ** 5:
                raise ValueError("PowersRepresentations: more than 10^5 representations")
    rec([], n, 0, k)
    return L(*out)


@fn("PrimitiveRootList", area=AREA, status="added", listable=True,
    notes="sorted primitive roots modulo n; {} when none exist; n <= 10^7"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _primitive_root_list(n: PosInteger) -> ListOf[Expr]:
    n = pos_int(n, "PrimitiveRootList", "modulus")
    if n == 1:
        raise ValueError("PrimitiveRootList: the modulus must be at least 2")
    if n > 10 ** 7:
        raise NotImplementedError("PrimitiveRootList: modulus limited to 10^7")
    g = sp.primitive_root(n)
    if g is None:
        return L()
    phi = int(sp.totient(n))
    roots = sorted(pow(int(g), k, n) for k in range(1, phi + 1) if math.gcd(k, phi) == 1)
    return L(*roots)


@fn("PowerModList", area=AREA, status="added",
    notes="all x in [0, m) with x^q == a^p (mod m) for s = p/q; brute force for m <= 10^6"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _power_mod_list(a: Integer, s: Expr, m: PosInteger) -> ListOf[Expr]:
    a = int_arg(a, "PowerModList")
    m = pos_int(m, "PowerModList", "modulus")
    fr = _exact_rational(s)
    if fr is None:
        if is_numeric_value(s):
            raise ValueError("PowerModList: the exponent must be rational")
        raise Unevaluated()
    p, q = fr.numerator, fr.denominator
    if p < 0:
        if math.gcd(a, m) != 1:
            return L()
        target = pow(pow(a, -1, m), -p, m)
    else:
        target = pow(a, p, m)
    if q == 1:
        return L(target)
    if m > 10 ** 6:
        raise NotImplementedError("PowerModList: rational exponents are limited to moduli up to 10^6")
    return L(*(x for x in range(m) if pow(x, q, m) == target))


@fn("MantissaExponent", area=AREA, status="added", listable=True,
    notes="{m, e} with 1/b <= |m| < 1 and x == m b^e; exact input stays exact; MantissaExponent[0] = {0, 0}"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _mantissa_exponent(x: Expr, b: Expr = 10) -> ListOf[Expr]:
    bb = int_arg(b, "MantissaExponent", "base") if not has_float(b) else None
    if bb is None or bb < 2:
        raise ValueError("MantissaExponent: base must be an integer greater than 1")
    v = S(x)
    if not isinstance(v, sp.Basic) or v.free_symbols or not v.is_number:
        raise Unevaluated()
    if v.is_real is not True:
        raise ValueError("MantissaExponent: the number must be real")
    if v == 0:
        return L(x if has_float(x) else 0, 0)
    av = abs(v)
    fr = _exact_rational(x)
    if fr is not None:
        f = abs(fr)
        e = 0
        while f >= 1:
            f /= bb
            e += 1
        while f < Fraction(1, bb):
            f *= bb
            e -= 1
        m = fr / Fraction(bb) ** e if e >= 0 else fr * bb ** (-e)
        return L(_out_frac(m), e)
    if isinstance(x, float):
        fx = Fraction(x)
        f, e = abs(fx), 0
        while f >= 1:
            f /= bb
            e += 1
        while f < Fraction(1, bb):
            f *= bb
            e -= 1
        return L(float(fx / Fraction(bb) ** e), e)       # correctly rounded mantissa
    if has_float(x):
        import mpmath
        from ..precision import input_precision
        p = input_precision([x])
        prec = 53 if p in ("m", None) else int(p)
        with mpmath.workprec(prec + 20):
            xv = mpmath.mpf(sp.Float(v, int(prec * 0.30103) + 5))
            e = int(mpmath.floor(mpmath.log(abs(xv), bb))) + 1
            m = xv / mpmath.mpf(bb) ** e
            if abs(m) >= 1:
                m /= bb
                e += 1
            elif abs(m) < mpmath.mpf(1) / bb:
                m *= bb
                e -= 1
        if p in ("m", None):
            return L(float(m), e)
        from ..precision import round_to
        return L(round_to(R(sp.Float(m, int(prec * 0.30103) + 5)), p), e)
    # exact irrational: certified floor(log_b |x|)
    e = None
    for dps in (30, 60, 120, 240):
        lg = sp.N(sp.log(av, bb), dps)
        lo, hi = sp.floor(lg - sp.Rational(1, 10 ** (dps // 2))), sp.floor(lg + sp.Rational(1, 10 ** (dps // 2)))
        if lo == hi:
            e = int(lo) + 1
            break
    if e is None:
        raise ValueError("MantissaExponent: cannot certify the exponent")
    return L(R(v / sp.Integer(bb) ** e), e)
