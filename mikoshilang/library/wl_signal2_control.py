"""Wolfram batch X03 (Signal II, part 2): linear systems, control plots, Kalman filtering, canonical time warping
and inverse Hilbert / bilateral Laplace / bilateral Z transforms.

* TransferFunctionModel / StateSpaceModel (minimal SISO/MIMO core: construction, conversion, evaluation, poles and
  zeros) and the frequency-response plots BodePlot, NyquistPlot, NicholsPlot, SingularValuePlot.
* KalmanFilter (exact best linear one-step predictors of scalar ARMA-type processes).
* CanonicalWarpingDistance / CanonicalWarpingCorrespondence (Zhou & De la Torre 2009).
* InverseHilbertTransform, InverseBilateralLaplaceTransform, InverseBilateralZTransform (closed forms with stated
  rules and numeric self-checks).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).  Defaults that Wolfram does not document are stated in the notes.
"""
from __future__ import annotations

import math

import mpmath
import numpy as np
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, Unevaluated, R, S, is_list
from .wl_common import L
from .wl_signal import _opts, _int_arg, _array, _to_np, _out

AREA = "signal processing II (X03)"


def ev(x):
    from ..evaluate import evaluate
    return evaluate(x)


def _sym(name):
    return Symbol(name)


def _is_sym(v, *names):
    return isinstance(v, Symbol) and v.name in names


def _rule(a, b):
    return Expr("Rule", a, b)


def _var(v, fname, what="variable"):
    if not isinstance(v, Symbol):
        raise TypeError(f"{fname}: the {what} must be a symbol, not {v!r}")
    return v


def _sp(x):
    return sp.sympify(S(x))


_REL = {"Less": sp.StrictLessThan, "Greater": sp.StrictGreaterThan, "LessEqual": sp.LessThan,
        "GreaterEqual": sp.GreaterThan, "Equal": sp.Eq, "Unequal": sp.Ne}


def _cond_sp(c):
    """A MikoshiLang condition (relations, And/Or/Not, Inequality) -> SymPy Boolean."""
    if c is True or _is_sym(c, "True"):
        return sp.true
    if c is False or _is_sym(c, "False"):
        return sp.false
    if isinstance(c, Expr):
        h = c.head
        if h in ("And", "Or"):
            parts = [_cond_sp(a) for a in c.args]
            return sp.And(*parts) if h == "And" else sp.Or(*parts)
        if h == "Not" and len(c.args) == 1:
            return sp.Not(_cond_sp(c.args[0]))
        if h in _REL:
            vals = [_sp(a) for a in c.args]
            return sp.And(*[_REL[h](vals[i], vals[i + 1]) for i in range(len(vals) - 1)])
        if h == "Inequality" and len(c.args) % 2 == 1:
            out = []
            for i in range(0, len(c.args) - 2, 2):
                op = c.args[i + 1]
                out.append(_REL[op.name](_sp(c.args[i]), _sp(c.args[i + 2])))
            return sp.And(*out)
    raise ValueError(f"unsupported condition {c!r}")


def _sp_pw(e):
    """S() that also converts Piecewise and relations (placeholders for Piecewise subtrees)."""
    reps = {}

    def walk(x):
        if isinstance(x, Expr):
            if x.head == "Piecewise" and x.args and is_list(x.args[0]):
                pairs = []
                for pr in x.args[0].args:
                    if not (is_list(pr) and len(pr.args) == 2):
                        raise ValueError("malformed Piecewise")
                    pairs.append((_sp_pw(pr.args[0]), _cond_sp(pr.args[1])))
                default = _sp_pw(x.args[1]) if len(x.args) > 1 else sp.Integer(0)
                d = sp.Dummy("pw")
                reps[d] = sp.Piecewise(*pairs, (default, True))
                return Symbol(d.name + "_" + str(id(d)))
            return Expr(x.head, *[walk(a) for a in x.args])
        return x
    w = walk(e)
    out = _sp(w)
    for d, pw in reps.items():
        out = out.subs(sp.Symbol(d.name + "_" + str(id(d))), pw)
    return out


def _is_num(x):
    return isinstance(x, sp.Basic) and x.is_number


# ═══════════════════════════════════════════════════════════════ TransferFunctionModel ══
_SYS_OPTS = {"SamplingPeriod", "SystemsModelLabels"}


def _sys_opts(opts):
    sp_ = opts.get("SamplingPeriod", _sym("None"))
    lab = opts.get("SystemsModelLabels", _sym("None"))
    return sp_, lab


def _period(v, fname):
    if v is None or _is_sym(v, "None"):
        return None
    try:
        T = _sp(v)
    except Exception:
        raise TypeError(f"{fname}: SamplingPeriod must be None or a positive number, not {v!r}")
    if T.is_number and not (T.is_real and T > 0):
        raise ValueError(f"{fname}: SamplingPeriod must be a positive number")
    return T


class _TF:
    """nums: list of lists of SymPy polynomials (outputs × inputs) over the common denominator den in variable x."""

    def __init__(self, nums, den, x, T=None, labels=None):
        self.nums, self.den, self.x, self.T, self.labels = nums, den, x, T, labels

    @property
    def shape(self):
        return len(self.nums), len(self.nums[0])

    def matrix(self):
        return sp.Matrix([[sp.cancel(nij / self.den) for nij in row] for row in self.nums])

    def at(self, v):
        return self.matrix().subs(self.x, v)

    def numeric(self, vals):
        """G(v) for an array of complex v -> array (len, p, q)."""
        f = sp.lambdify(self.x, self.matrix(), "numpy")
        out = np.empty((len(vals),) + self.shape, dtype=complex)
        for i, v in enumerate(vals):
            with np.errstate(all="ignore"):
                out[i] = np.array(f(complex(v)), dtype=complex).reshape(self.shape)
        return out


def _tf_from_matrix(M, x):
    """SymPy Matrix of rational functions -> (nums, den) over the least common denominator."""
    dens = [sp.fraction(sp.cancel(sp.together(e)))[1] for e in M]
    den = sp.Integer(1)
    for d in dens:
        den = sp.lcm(den, d)
    den = sp.factor_terms(sp.expand(den))
    nums = [[sp.expand(sp.cancel(M[i, j] * den)) for j in range(M.shape[1])] for i in range(M.shape[0])]
    for row in nums:
        for v in row:
            if not v.is_polynomial(x) or (sp.denom(sp.together(v)) != 1 and sp.denom(sp.together(v)).has(x)):
                raise ValueError("not rational")
    # remove a common factor shared by den and every numerator
    g = den
    for row in nums:
        for v in row:
            g = sp.gcd(g, v)
    if g.has(x):
        den = sp.cancel(den / g)
        nums = [[sp.cancel(v / g) for v in row] for row in nums]
    lc = sp.Poly(den, x).LC() if den.has(x) else den
    if lc != 1 and lc != 0:
        den = sp.expand(den / lc)
        nums = [[sp.expand(v / lc) for v in row] for row in nums]
    return nums, sp.expand(den)


def _rational_matrix(e, x, fname):
    if is_list(e):
        rows = e.args
        if rows and all(is_list(r) for r in rows):
            M = sp.Matrix([[_sp(c) for c in r.args] for r in rows])
        else:
            M = sp.Matrix([[_sp(c) for c in rows]])
    else:
        M = sp.Matrix([[_sp(e)]])
    for v in M:
        if not v.is_rational_function(x):
            raise NotImplementedError(f"{fname}: only rational transfer functions are supported (got {R(v)!r})")
    return M


def _parse_tfm(e, fname):
    if not (isinstance(e, Expr) and e.head == "TransferFunctionModel"):
        if isinstance(e, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a TransferFunctionModel, not {e!r}")
    args, opts = _opts(e.args, _SYS_OPTS, fname)
    if len(args) != 2 or not (is_list(args[0]) and len(args[0].args) == 2 and is_list(args[0].args[0])):
        raise ValueError(f"{fname}: the TransferFunctionModel is not in standard form")
    x = _sp(_var(args[1], fname))
    nums = [[_sp(c) for c in r.args] for r in args[0].args[0].args]
    den = _sp(args[0].args[1])
    if not nums or not nums[0] or any(len(r) != len(nums[0]) for r in nums):
        raise ValueError(f"{fname}: the numerator must be a nonempty matrix")
    return _TF(nums, den, x, _period(opts.get("SamplingPeriod"), fname), opts.get("SystemsModelLabels"))


def _tf_expr(tf, var):
    opts = [_rule(_sym("SamplingPeriod"), _sym("None") if tf.T is None else R(tf.T)),
            _rule(_sym("SystemsModelLabels"), tf.labels if tf.labels is not None else _sym("None"))]
    num = L(*[L(*[R(v) for v in row]) for row in tf.nums])
    return Expr("TransferFunctionModel", L(num, R(tf.den)), var, *opts)


def _ss_to_tf(ss, x):
    A, B, C, D = ss
    n = A.shape[0]
    if n == 0:
        return D
    M = C * (x * sp.eye(n) - A).inv() * B + D
    return M.applyfunc(sp.cancel)


def _tfm(sys: Expr, *rest: Expr) -> Expr:
    fname = "TransferFunctionModel"
    args, opts = _opts(rest, _SYS_OPTS, fname)
    if isinstance(sys, Expr) and sys.head == "StateSpaceModel":
        ss, T, labels = _parse_ssm(sys, fname)
        if len(args) > 1:
            raise ValueError(f"{fname}: TransferFunctionModel[ssm] or [ssm, var] expected")
        var = args[0] if args else _sym("z" if T is not None else "s")
        x = _sp(_var(var, fname))
        M = _ss_to_tf(ss, x)
        nums, den = _tf_from_matrix(M, x)
        return _tf_expr(_TF(nums, den, x, T, labels), var)
    if len(args) != 1:
        if isinstance(sys, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: TransferFunctionModel[expr, var] expected")
    var = _var(args[0], fname)
    x = _sp(var)
    T = _period(opts.get("SamplingPeriod"), fname)
    labels = opts.get("SystemsModelLabels")
    if isinstance(sys, Symbol):
        raise Unevaluated()
    if is_list(sys) and len(sys.args) == 2 and is_list(sys.args[0]) and sys.args[0].args and \
            all(is_list(r) for r in sys.args[0].args) and not is_list(sys.args[1]):
        num = _rational_matrix(sys.args[0], x, fname)
        den = _sp(sys.args[1])
        M = num / den
    elif is_list(sys) and len(sys.args) == 2 and is_list(sys.args[0]) and is_list(sys.args[1]) and \
            sys.args[0].args and all(is_list(r) for r in sys.args[0].args):
        num = _rational_matrix(sys.args[0], x, fname)
        den = _rational_matrix(sys.args[1], x, fname)
        if num.shape != den.shape:
            raise ValueError(f"{fname}: numerator and denominator matrices must have the same dimensions")
        M = sp.Matrix(num.shape[0], num.shape[1], lambda i, j: num[i, j] / den[i, j])
    else:
        M = _rational_matrix(sys, x, fname)
    if any(sp.cancel(sp.together(v)).as_numer_denom()[1] == 0 for v in M):
        raise ValueError(f"{fname}: a denominator is identically zero")
    try:
        nums, den = _tf_from_matrix(M, x)
    except ValueError:
        raise NotImplementedError(f"{fname}: only rational transfer functions are supported")
    tf = _TF(nums, den, x, T, labels)
    out = _tf_expr(tf, var)
    if isinstance(sys, Expr) and sys == out.args[0] and len(rest) == 3 and \
            all(isinstance(r, Expr) for r in rest[1:]) and out.args[2:] == tuple(rest[1:]):
        raise Unevaluated()
    return out


fn("TransferFunctionModel", area=AREA, status="added",
   notes="TransferFunctionModel[expr, s] (rational function or matrix of them), [{num, den}, s] (numerator matrix "
         "over a common denominator, or two matrices), [ssm] / [ssm, s] (= C (s I - A)^-1 B + D; variable z when "
         "SamplingPeriod is set) gives the standard form TransferFunctionModel[{nums, den}, s, SamplingPeriod -> "
         "T, SystemsModelLabels -> labels] with den the monic least common denominator and common factors "
         "cancelled. tfm[x] evaluates the transfer matrix at x; tfm[\"Poles\"] / tfm[\"Zeros\"] give the roots of "
         "den and (SISO) of the numerator (MikoshiLang accessors; Wolfram uses TransferFunctionPoles/Zeros), "
         "tfm[\"Dimensions\"] = {outputs, inputs}. Time delays and non-rational models raise NotImplementedError. "
         "Check: reference (hand-computed C (sI - A)^-1 B + D, Wolfram's documented standard forms; StateSpaceModel "
         "round trip) in tests.")(_tfm)


def _poles(tf):
    return sp.Poly(tf.den, tf.x).all_roots() if tf.den.has(tf.x) and _poly_numeric(tf.den, tf.x) else \
        list(sp.roots(sp.Poly(tf.den, tf.x), multiple=True)) if tf.den.has(tf.x) else []


def _poly_numeric(p, x):
    return all(c.is_number and c.is_real for c in sp.Poly(p, x).all_coeffs()) and \
        all(c.is_rational for c in sp.Poly(p, x).all_coeffs())


def _roots(p, x):
    if not p.has(x):
        return []
    P = sp.Poly(p, x)
    if _poly_numeric(p, x):
        return [sp.nsimplify(r) if r.is_Rational else r for r in P.all_roots()]
    rs = sp.roots(P, multiple=True)
    if len(rs) != P.degree():
        return [sp.Float(complex(r).real) + sp.I * sp.Float(complex(r).imag) for r in
                np.roots([complex(c) for c in P.all_coeffs()])]
    return rs


def _tfm_hook(head, args, ev_):
    fname = "TransferFunctionModel"
    try:
        tf = _parse_tfm(head, fname)
    except (ValueError, TypeError):
        return K.UNHANDLED
    if len(args) != 1:
        raise TypeError(f"{fname}: tfm[x] or tfm[\"prop\"] expected")
    a = args[0]
    if isinstance(a, str):
        if a == "Poles":
            return L(*[R(r) for r in _roots(tf.den, tf.x)])
        if a == "Zeros":
            if tf.shape != (1, 1):
                raise NotImplementedError(f"{fname}: transmission zeros of MIMO models are not supported")
            return L(*[R(r) for r in _roots(tf.nums[0][0], tf.x)])
        if a == "Dimensions":
            return L(*tf.shape)
        if a == "Properties":
            return L("Dimensions", "Poles", "Zeros")
        raise ValueError(f"{fname}: unknown property {a!r}")
    v = _sp(a)
    if v.is_number:
        dv = tf.den.subs(tf.x, v)
        if sp.simplify(dv) == 0:
            raise ValueError(f"{fname}: {a!r} is a pole of the transfer function")
    M = tf.at(v).applyfunc(sp.simplify)
    return L(*[L(*[R(M[i, j]) for j in range(M.shape[1])]) for i in range(M.shape[0])])


KL.COMPOUND_HEAD_HOOKS["TransferFunctionModel"] = _tfm_hook


# ═══════════════════════════════════════════════════════════════ StateSpaceModel ══
def _mat(e, fname, what):
    if not is_list(e):
        raise TypeError(f"{fname}: the {what} matrix must be a list of lists, not {e!r}")
    if not e.args:
        return sp.zeros(0, 0)
    if not all(is_list(r) for r in e.args):
        raise TypeError(f"{fname}: the {what} matrix must be a list of lists")
    n = len(e.args[0].args)
    if any(len(r.args) != n for r in e.args):
        raise ValueError(f"{fname}: the {what} matrix must be rectangular")
    return sp.Matrix([[_sp(c) for c in r.args] for r in e.args]) if n else sp.zeros(len(e.args), 0)


def _parse_ssm(e, fname):
    if not (isinstance(e, Expr) and e.head == "StateSpaceModel"):
        if isinstance(e, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a StateSpaceModel, not {e!r}")
    args, opts = _opts(e.args, _SYS_OPTS, fname)
    if len(args) != 1 or not is_list(args[0]) or len(args[0].args) not in (3, 4):
        raise ValueError(f"{fname}: StateSpaceModel[{{a, b, c, d}}] expected")
    ms = [_mat(m, fname, w) for m, w in zip(args[0].args, ("a", "b", "c", "d"))]
    A, B, C = ms[:3]
    n = A.shape[0]
    if A.shape[1] != n or B.shape[0] != n or C.shape[1] != n:
        raise ValueError(f"{fname}: inconsistent dimensions: a must be n×n, b n×q and c p×n")
    if B.shape[1] == 0 and n == 0 and len(ms) == 4:
        q = ms[3].shape[1]
    else:
        q = B.shape[1]
    p = C.shape[0] if C.shape[0] else (ms[3].shape[0] if len(ms) == 4 else 0)
    D = ms[3] if len(ms) == 4 else sp.zeros(p, q)
    if D.shape != (p, q):
        raise ValueError(f"{fname}: the d matrix must be {p}×{q}")
    return (A, B, C, D), _period(opts.get("SamplingPeriod"), fname), opts.get("SystemsModelLabels")


def _ss_expr(ss, T, labels):
    A, B, C, D = ss

    def m(M):
        return L(*[L(*[R(M[i, j]) for j in range(M.shape[1])]) for i in range(M.shape[0])])
    return Expr("StateSpaceModel", L(m(A), m(B), m(C), m(D)),
                _rule(_sym("SamplingPeriod"), _sym("None") if T is None else R(T)),
                _rule(_sym("SystemsModelLabels"), labels if labels is not None else _sym("None")))


def _realize(tf):
    """Block controllable-canonical realization: one companion block per input column."""
    x = tf.x
    p, q = tf.shape
    den = sp.Poly(tf.den, x)
    nd = den.degree()
    a = [c / den.LC() for c in den.all_coeffs()]
    blocks_A, blocks_B, cols_C = [], [], []
    D = sp.zeros(p, q)
    for j in range(q):
        cj = []
        for i in range(p):
            num = sp.Poly(tf.nums[i][j] / den.LC(), x) if nd else sp.Poly(tf.nums[i][j] / den.LC(), x)
            qt, rm = sp.div(num, sp.Poly(den.as_expr() / den.LC(), x)) if nd else (num, sp.Poly(0, x))
            if qt.degree() > 0:
                raise NotImplementedError("StateSpaceModel: improper transfer functions have no state-space "
                                          "realization")
            D[i, j] = qt.as_expr()
            coeffs = rm.all_coeffs()[::-1] + [0] * nd
            cj.append([sp.sympify(c) for c in coeffs[:nd]])
        if nd:
            Aj = sp.zeros(nd, nd)
            for k in range(nd - 1):
                Aj[k, k + 1] = 1
            for k in range(nd):
                Aj[nd - 1, k] = -a[nd - k]
            Bj = sp.zeros(nd, 1)
            Bj[nd - 1, 0] = 1
            blocks_A.append(Aj)
            blocks_B.append(Bj)
            cols_C.append(sp.Matrix(cj))
    if not nd:
        return sp.zeros(0, 0), sp.zeros(0, q), sp.zeros(p, 0), D
    A = sp.diag(*blocks_A)
    B = sp.zeros(nd * q, q)
    for j in range(q):
        B[j * nd:(j + 1) * nd, j] = blocks_B[j]
    C = sp.Matrix.hstack(*cols_C)
    return A, B, C, D


def _ssm(sys: Expr, *rest: Expr) -> Expr:
    fname = "StateSpaceModel"
    args, opts = _opts(rest, _SYS_OPTS, fname)
    if isinstance(sys, Symbol):
        raise Unevaluated()
    if isinstance(sys, Expr) and sys.head == "TransferFunctionModel":
        if args:
            raise ValueError(f"{fname}: StateSpaceModel[tfm] takes no further arguments")
        tf = _parse_tfm(sys, fname)
        ss = _realize(tf)
        back = _ss_to_tf(ss, tf.x)
        if any(sp.simplify(u - v) != 0 for u, v in zip(back, tf.matrix())):
            raise ArithmeticError(f"{fname}: result failed verification (realization does not reproduce the "
                                  f"transfer function)")
        return _ss_expr(ss, tf.T, tf.labels)
    if not is_list(sys):
        raise TypeError(f"{fname}: expected {{a, b, c, d}} or a TransferFunctionModel, not {sys!r}")
    if args:
        raise ValueError(f"{fname}: called with too many arguments")
    e = Expr("StateSpaceModel", sys, *[_rule(_sym(k), v) for k, v in opts.items()])
    ss, T, labels = _parse_ssm(e, fname)
    out = _ss_expr(ss, T, labels)
    if out == Expr("StateSpaceModel", sys, *rest):
        raise Unevaluated()
    return out


fn("StateSpaceModel", area=AREA, status="added",
   notes="StateSpaceModel[{a, b, c, d}] (d optional, default zero) validates the dimensions and gives the standard "
         "form StateSpaceModel[{a, b, c, d}, SamplingPeriod -> T, SystemsModelLabels -> labels]; "
         "StateSpaceModel[tfm] realizes a proper TransferFunctionModel in block controllable-canonical form (one "
         "companion block of the common denominator per input; improper models raise NotImplementedError). "
         "Check: self-check (C (sI - A)^-1 B + D of the realization must equal the transfer matrix, else "
         "ArithmeticError) and reference (hand-computed companion matrices) in tests.")(_ssm)


# ═══════════════════════════════════════════════════════════════ frequency-response plots ══
def _system(e, fname):
    if isinstance(e, Expr) and e.head == "TransferFunctionModel":
        return _parse_tfm(e, fname)
    if isinstance(e, Expr) and e.head == "StateSpaceModel":
        ss, T, labels = _parse_ssm(e, fname)
        x = sp.Symbol("z" if T is not None else "s")
        nums, den = _tf_from_matrix(_ss_to_tf(ss, x), x)
        return _TF(nums, den, x, T, labels)
    if isinstance(e, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: expected a TransferFunctionModel or StateSpaceModel, not {e!r}")


def _freq_range(tf, spec, fname):
    if spec is not None and not _is_sym(spec, "Automatic"):
        if not (is_list(spec) and len(spec.args) == 2):
            raise ValueError(f"{fname}: the frequency range must be {{ωmin, ωmax}}")
        try:
            lo, hi = float(_sp(spec.args[0])), float(_sp(spec.args[1]))
        except TypeError:
            raise Unevaluated()
        if not 0 < lo < hi:
            raise ValueError(f"{fname}: the frequency range must satisfy 0 < ωmin < ωmax")
        return lo, hi
    if any(not c.is_number for row in tf.nums for c in sp.Poly(row[0], tf.x).all_coeffs()) or \
            any(not c.is_number for c in sp.Poly(tf.den, tf.x).all_coeffs()):
        raise Unevaluated()
    pts = []
    for p in _roots(tf.den, tf.x) + [r for row in tf.nums for v in row for r in _roots(v, tf.x)]:
        c = complex(p)
        if tf.T is not None:
            if abs(c) > 0:
                c = complex(np.log(c)) / float(tf.T)
            else:
                continue
        if abs(c) > 1e-12:
            pts.append(abs(c))
    if tf.T is not None:
        hi = math.pi / float(tf.T)
        lo = min(pts) / 10 if pts else hi / 1000
        return min(lo, hi / 100), hi
    if not pts:
        return 0.01, 100.0
    return min(pts) / 10, max(pts) * 10


def _response(tf, w):
    if tf.T is None:
        return tf.numeric(1j * w)
    return tf.numeric(np.exp(1j * w * float(tf.T)))


def _numeric_system(tf):
    for row in tf.nums:
        for v in row:
            if v.free_symbols - {tf.x}:
                raise Unevaluated()
    if tf.den.free_symbols - {tf.x}:
        raise Unevaluated()
    if tf.T is not None and not tf.T.is_number:
        raise Unevaluated()


def _plot_args(sys, rest, fname):
    args = [a for a in rest if not (isinstance(a, Expr) and a.head in ("Rule", "RuleDelayed"))]
    other = [a for a in rest if isinstance(a, Expr) and a.head in ("Rule", "RuleDelayed")]
    if len(args) > 1:
        raise ValueError(f"{fname}: called with too many arguments")
    systems = list(sys.args) if is_list(sys) else [sys]
    tfs = [_system(s, fname) for s in systems]
    for tf in tfs:
        _numeric_system(tf)
    return tfs, (args[0] if args else None), other


def _curves(tfs, spec, fname, npts=400, full=False):
    out = []
    for tf in tfs:
        lo, hi = _freq_range(tf, spec, fname)
        w = np.logspace(math.log10(lo), math.log10(hi), npts)
        G = _response(tf, w)
        out.append((tf, w, G))
    return out


def _finite(pts):
    pts = np.asarray(pts, dtype=float)
    return pts[np.all(np.isfinite(pts), axis=1)]


def _bode(sys: Expr, *rest: Expr) -> Expr:
    fname = "BodePlot"
    tfs, spec, other = _plot_args(sys, rest, fname)
    mags, phs = [], []
    for tf, w, G in _curves(tfs, spec, fname):
        p, q = tf.shape
        for i in range(p):
            for j in range(q):
                g = G[:, i, j]
                with np.errstate(all="ignore"):
                    db = 20 * np.log10(np.abs(g))
                ph = np.degrees(np.unwrap(np.angle(np.where(np.isfinite(g), g, 1))))
                mags.append(_out(_finite(np.column_stack([w, db])), "m", real=True))
                phs.append(_out(_finite(np.column_stack([w, ph])), "m", real=True))
    m = ev(Expr("ListLogLinearPlot", L(*mags), _rule(_sym("Joined"), True), *other))
    ph = ev(Expr("ListLogLinearPlot", L(*phs), _rule(_sym("Joined"), True), *other))
    return ev(Expr("GraphicsColumn", L(m, ph)))


fn("BodePlot", area=AREA, status="added",
   notes="BodePlot[lsys] / [lsys, {ωmin, ωmax}] / [{lsys1, …}]: GraphicsColumn of the magnitude 20 Log10 |G(iω)| "
         "(dB) and the unwrapped phase Arg G(iω) (degrees) against log ω for every input/output pair (discrete "
         "systems use G(e^{iωT}) up to ω = π/T), 400 log-spaced points; the automatic range spans the pole/zero "
         "corner frequencies ±1 decade ({0.01, 100} without corners); infinite points (poles on the axis) are "
         "omitted; other options pass to ListLogLinearPlot; symbolic parameters stay unevaluated. Check: reference "
         "(the plotted points equal 20 Log10 Abs[tfm[I ω]] and the phase of a first-order lag) in "
         "tests.")(_bode)


def _nyquist(sys: Expr, *rest: Expr) -> Expr:
    fname = "NyquistPlot"
    tfs, spec, other = _plot_args(sys, rest, fname)
    curves = []
    for tf, w, G in _curves(tfs, spec, fname):
        p, q = tf.shape
        for i in range(p):
            for j in range(q):
                g = G[:, i, j]
                pos = np.column_stack([g.real, g.imag])
                neg = np.column_stack([g.real, -g.imag])[::-1]
                curves.append(_out(_finite(np.vstack([neg, pos])), "m", real=True))
    return ev(Expr("ListLinePlot", L(*curves), *other))


fn("NyquistPlot", area=AREA, status="added",
   notes="NyquistPlot[lsys] / [lsys, {ωmin, ωmax}]: ListLinePlot of {Re G(iω), Im G(iω)} for ω from -ωmax to "
         "-ωmin and ωmin to ωmax (the negative branch is the complex conjugate, real-coefficient systems) for "
         "every input/output pair; frequency range and sampling as in BodePlot; other options pass to "
         "ListLinePlot. Check: reference (points of a first-order lag lie on the circle |G - 1/2| = 1/2) in "
         "tests.")(_nyquist)


def _nichols(sys: Expr, *rest: Expr) -> Expr:
    fname = "NicholsPlot"
    tfs, spec, other = _plot_args(sys, rest, fname)
    curves = []
    for tf, w, G in _curves(tfs, spec, fname):
        p, q = tf.shape
        for i in range(p):
            for j in range(q):
                g = G[:, i, j]
                with np.errstate(all="ignore"):
                    db = 20 * np.log10(np.abs(g))
                ph = np.degrees(np.unwrap(np.angle(np.where(np.isfinite(g), g, 1))))
                curves.append(_out(_finite(np.column_stack([ph, db])), "m", real=True))
    return ev(Expr("ListLinePlot", L(*curves), *other))


fn("NicholsPlot", area=AREA, status="added",
   notes="NicholsPlot[lsys] / [lsys, {ωmin, ωmax}]: ListLinePlot of {phase (degrees, unwrapped), magnitude (dB)} of "
         "G(iω) over the BodePlot frequency grid for every input/output pair (no Nichols grid is drawn); other "
         "options pass to ListLinePlot. Check: reference (points equal the BodePlot phase and magnitude) in "
         "tests.")(_nichols)


def _svplot(sys: Expr, *rest: Expr) -> Expr:
    fname = "SingularValuePlot"
    tfs, spec, other = _plot_args(sys, rest, fname)
    curves = []
    for tf, w, G in _curves(tfs, spec, fname):
        good = np.all(np.isfinite(G.reshape(len(w), -1)), axis=1)
        sv = np.array([np.linalg.svd(G[k], compute_uv=False) if good[k] else
                       np.full(min(tf.shape), np.nan) for k in range(len(w))])
        with np.errstate(all="ignore"):
            db = 20 * np.log10(sv)
        for i in range(sv.shape[1]):
            curves.append(_out(_finite(np.column_stack([w, db[:, i]])), "m", real=True))
    return ev(Expr("ListLogLinearPlot", L(*curves), _rule(_sym("Joined"), True), *other))


fn("SingularValuePlot", area=AREA, status="added",
   notes="SingularValuePlot[lsys] / [lsys, {ωmin, ωmax}]: ListLogLinearPlot of the singular values σ_i(G(iω)) in dB "
         "(20 Log10 σ_i, decreasing order) against log ω, frequency grid as in BodePlot; for SISO systems this is "
         "the Bode magnitude; other options pass to ListLogLinearPlot. Check: reference (SingularValueList of the "
         "evaluated transfer matrix at sample frequencies) in tests.")(_svplot)


# ═══════════════════════════════════════════════════════════════ KalmanFilter ══
def _kalman(proc: Expr, data: Expr, *rest: Expr) -> ListOf[Expr]:
    fname = "KalmanFilter"
    if rest:
        raise ValueError(f"{fname}: called with too many arguments")
    if not isinstance(proc, Expr) or proc.head not in ("ARProcess", "MAProcess", "ARMAProcess"):
        if isinstance(proc, Symbol):
            raise Unevaluated()
        if isinstance(proc, Expr) and proc.head in ("ARIMAProcess", "SARIMAProcess", "SARMAProcess",
                                                    "FARIMAProcess", "TimeSeriesModel"):
            raise NotImplementedError(f"{fname}: {proc.head} is not supported (scalar AR/MA/ARMA processes)")
        raise TypeError(f"{fname}: expected a time series process, not {proc!r}")
    pa = list(proc.args)
    full = {"ARProcess": 3, "MAProcess": 3, "ARMAProcess": 4}[proc.head]
    if len(pa) == full:
        c = _sp(pa[0])
        if c != 0:
            raise NotImplementedError(f"{fname}: processes with a nonzero constant are not supported")
        pa = pa[1:]
    elif len(pa) != full - 1:
        if len(pa) == full + 1 and proc.head == "ARMAProcess":
            raise NotImplementedError(f"{fname}: ARMA processes with initial values are not supported")
        raise ValueError(f"{fname}: malformed {proc.head}")
    for a in pa:
        for v in (a.args if is_list(a) else [a]):
            if is_list(v):
                raise NotImplementedError(f"{fname}: vector processes are not supported")
            s = _sp(v)
            if not s.is_number:
                raise Unevaluated()
    if isinstance(data, Expr) and data.head in ("TemporalData", "TimeSeries"):
        raise NotImplementedError(f"{fname}: TemporalData input is not supported (give a list)")
    if not is_list(data):
        if isinstance(data, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list, not {data!r}")
    xs = []
    for v in data.args:
        if is_list(v):
            raise NotImplementedError(f"{fname}: vector data are not supported")
        if isinstance(v, (str, bool)):
            raise TypeError(f"{fname}: data elements must be numbers or symbols, not {v!r}")
        xs.append(_sp(v))
    n = len(xs)
    if n == 0:
        return L()
    numeric = any(isinstance(a, sp.Float) for v in pa for a in sp.preorder_traversal(_sp(v) if not is_list(v)
                                                                                    else sp.Integer(0))) or \
        any(isinstance(a, sp.Float) for v in pa if is_list(v) for c in v.args for a in sp.preorder_traversal(_sp(c)))
    r = []
    for h in range(n):
        rv = _sp(ev(Expr("CovarianceFunction", proc, h)))
        if not rv.is_number:
            raise NotImplementedError(f"{fname}: the autocovariance of {proc.head} could not be computed")
        r.append(sp.N(rv) if numeric else sp.nsimplify(rv))
    if r[0] == 0:
        raise ValueError(f"{fname}: the process has zero variance")
    out = [sp.Integer(0)]
    phi = []
    v = r[0]
    for t in range(1, n):
        # Durbin–Levinson step: φ_{t,t} = (r(t) - Σ φ_{t-1,j} r(t-j)) / v_{t-1}
        k = (r[t] - sum(phi[j] * r[t - 1 - j] for j in range(t - 1))) / v
        phi = [phi[j] - k * phi[t - 2 - j] for j in range(t - 1)] + [k]
        v = v * (1 - k * k)
        pred = sum(phi[j] * xs[t - 1 - j] for j in range(t))
        out.append(sp.expand(pred))
        if v == 0 or (v.is_number and abs(complex(v)) < 1e-300):
            out.extend([sp.nan] * 0)
    res = [R(sp.N(o) if numeric else o) for o in out]
    return L(*res)


fn("KalmanFilter", area=AREA, status="added",
   notes="KalmanFilter[tproc, data] for scalar ARProcess / MAProcess / ARMAProcess with numeric parameters and zero "
         "constant: the Kalman filter's one-step predictions x̂_t = E[x_t | x_1, …, x_{t-1}] started from the "
         "stationary distribution, computed exactly as the best linear predictors of the Durbin–Levinson "
         "recursion on CovarianceFunction[tproc, h] (Brockwell & Davis, §5.2), with the first element 0; data may be "
         "numeric or symbolic (the result is linear in the data); exact parameters give exact results. "
         "Nonzero constants, initial values, vector processes, (S)ARIMA/FARIMA and TemporalData raise "
         "NotImplementedError. Check: reference (Wolfram's documented AR(1) example {0, v/3, x/3, y/3} and the "
         "MA(1) innovations recursion) in tests.")(_kalman)


# ═══════════════════════════════════════════════════════════════ canonical time warping ══
def _seq(s, fname):
    if not is_list(s) or not s.args:
        if isinstance(s, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: a sequence must be a nonempty list of numbers or vectors, not {s!r}")
    if isinstance(s.args[0], Expr) and s.args[0].head == "Quantity":
        raise NotImplementedError(f"{fname}: quantity sequences are not supported")
    flat, shape = _array(s, fname, max_rank=2)
    a = _to_np(flat, shape)
    if not np.all(a.imag == 0):
        raise NotImplementedError(f"{fname}: complex sequences are not supported")
    a = a.real
    return a[:, None] if a.ndim == 1 else a


def _window_mask(n1, n2, win, fname):
    if win is None or _is_sym(win, "Automatic"):
        return None
    kind, r = "SlantedBand", None
    if is_list(win) and len(win.args) == 2 and isinstance(win.args[0], str):
        kind, r = win.args[0], win.args[1]
    else:
        r = win
    if kind == "Parallelogram":
        raise NotImplementedError(f"{fname}: the \"Parallelogram\" window is not supported")
    if kind not in ("SlantedBand", "Band"):
        raise ValueError(f"{fname}: unknown window {win!r}")
    rv = _sp(r)
    if not (rv.is_number and rv >= 0):
        raise ValueError(f"{fname}: the window radius must be a nonnegative number")
    rv = float(rv)
    i = np.arange(n1)[:, None]
    j = np.arange(n2)[None, :]
    if kind == "Band":
        return np.abs(i - j) <= rv
    slope = (n1 - 1) / (n2 - 1) if n2 > 1 else 0.0
    return np.abs(i - j * slope) <= rv + 1e-9


def _dtw_path(X, Y, mask):
    n1, n2 = len(X), len(Y)
    cost = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    if mask is not None:
        cost = np.where(mask, cost, np.inf)
    D = np.full((n1 + 1, n2 + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            c = cost[i - 1, j - 1]
            if c == np.inf:
                continue
            D[i, j] = c + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    if not np.isfinite(D[n1, n2]):
        raise ValueError("the search window admits no warping path")
    i, j = n1, n2
    path = [(i - 1, j - 1)]
    while (i, j) != (1, 1):
        cands = [(D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1)]
        cands = [c for c in cands if c[1] >= 1 and c[2] >= 1]
        _, i, j = min(cands, key=lambda c: c[0])
        path.append((i - 1, j - 1))
    return path[::-1]


def _inv_sqrt(C):
    w, V = np.linalg.eigh(C)
    if np.min(w) <= 0:
        raise ValueError("singular covariance")
    return V @ np.diag(w ** -0.5) @ V.T


def _ctw(s1, s2, rest, fname):
    args, opts = _opts(rest, {"DistanceFunction", "MaxIterations", "Method"}, fname)
    X, Y = _seq(s1, fname), _seq(s2, fname)
    n1, n2 = len(X), len(Y)
    if len(args) > 2:
        raise ValueError(f"{fname}: called with too many arguments")
    df = opts.get("DistanceFunction", _sym("Automatic"))
    if not (_is_sym(df, "Automatic", "SquaredEuclideanDistance")):
        raise NotImplementedError(f"{fname}: DistanceFunction -> {df!r} is not supported (squared Euclidean only)")
    mi = opts.get("MaxIterations", _sym("Automatic"))
    iters = 20 if _is_sym(mi, "Automatic") else _int_arg(mi, fname, "MaxIterations")
    lam, keep = None, None
    meth = opts.get("Method", _sym("Automatic"))
    if not _is_sym(meth, "Automatic"):
        rules = list(meth.args) if is_list(meth) else [meth]
        for rr in rules:
            if not (isinstance(rr, Expr) and rr.head == "Rule"):
                raise ValueError(f"{fname}: invalid Method setting {meth!r}")
            k, v = rr.args
            if k == "Lambdas":
                if not _is_sym(v, "Automatic"):
                    vals = [float(_sp(t)) for t in (v.args if is_list(v) else [v, v])]
                    if len(vals) != 2 or min(vals) < 0:
                        raise ValueError(f"{fname}: \"Lambdas\" must be {{λ1, λ2}} with nonnegative values")
                    lam = vals
            elif k == "DimensionsToKeep":
                if not _is_sym(v, "Automatic"):
                    keep = _int_arg(v, fname, "\"DimensionsToKeep\"")
            elif k in ("EnergyThreshold", "MatchingIntervals"):
                if not _is_sym(v, "Automatic"):
                    raise NotImplementedError(f"{fname}: the method option {k!r} is not supported")
            else:
                raise ValueError(f"{fname}: unknown method option {k!r}")
    mask = _window_mask(n1, n2, args[1] if len(args) > 1 else None, fname)
    if np.allclose(X - X.mean(axis=0), 0) or np.allclose(Y - Y.mean(axis=0), 0):
        raise ValueError(f"{fname}: canonical time warping needs non-constant sequences")
    d1, d2 = X.shape[1], Y.shape[1]
    b = min(d1, d2) if keep is None else keep
    if b > min(d1, d2):
        raise ValueError(f"{fname}: \"DimensionsToKeep\" cannot exceed min(dimensions) = {min(d1, d2)}")
    if len(args) >= 1 and not _is_sym(args[0], "Automatic"):
        init = args[0]
        if not (is_list(init) and len(init.args) == 2 and all(is_list(t) for t in init.args)
                and len(init.args[0].args) == len(init.args[1].args) and init.args[0].args):
            raise ValueError(f"{fname}: the initial correspondence must be {{{{n1, …}}, {{m1, …}}}}")
        path = []
        for a, c in zip(init.args[0].args, init.args[1].args):
            ia, ic = _int_arg(a, fname, "an index"), _int_arg(c, fname, "an index")
            if ia > n1 or ic > n2:
                raise ValueError(f"{fname}: initial correspondence index out of range")
            path.append((ia - 1, ic - 1))
    else:
        m = max(n1, n2)
        path = sorted({(int(round(k * (n1 - 1) / (m - 1))) if m > 1 else 0,
                        int(round(k * (n2 - 1) / (m - 1))) if m > 1 else 0) for k in range(m)})
    best = None
    seen = []
    for _ in range(iters + 1):
        ix = np.array([p[0] for p in path])
        iy = np.array([p[1] for p in path])
        Xc, Yc = X - X[ix].mean(axis=0), Y - Y[iy].mean(axis=0)
        Xp, Yp = Xc[ix], Yc[iy]
        Cxx, Cyy, Cxy = Xp.T @ Xp, Yp.T @ Yp, Xp.T @ Yp
        lx = lam[0] if lam else 1e-8 * max(np.trace(Cxx) / d1, 1e-300)
        ly = lam[1] if lam else 1e-8 * max(np.trace(Cyy) / d2, 1e-300)
        try:
            Kx, Ky = _inv_sqrt(Cxx + lx * np.eye(d1)), _inv_sqrt(Cyy + ly * np.eye(d2))
        except ValueError:
            raise ValueError(f"{fname}: the aligned sequences have singular covariance; give \"Lambdas\" > 0")
        U, sv, Vt = np.linalg.svd(Kx @ Cxy @ Ky)
        Vx, Vy = Kx @ U[:, :b], Ky @ Vt.T[:, :b]
        PX, PY = Xc @ Vx, Yc @ Vy
        J = float(((PX[ix] - PY[iy]) ** 2).sum())
        if best is None or J < best[0] - 1e-12 * max(1.0, abs(best[0])):
            best = (J, path)
        try:
            new = _dtw_path(PX, PY, mask)
        except ValueError as ex:
            raise ValueError(f"{fname}: {ex}")
        if new == path or new in seen:
            # evaluate the converged path once more for the reported distance
            ix2 = np.array([p[0] for p in new])
            iy2 = np.array([p[1] for p in new])
            J2 = float(((PX[ix2] - PY[iy2]) ** 2).sum())
            if J2 < best[0]:
                best = (J2, new)
            break
        seen.append(path)
        path = new
    return best


def _ctw_distance(s1: Expr, s2: Expr, *rest: Expr) -> Real:
    J, _path = _ctw(s1, s2, rest, "CanonicalWarpingDistance")
    return J


fn("CanonicalWarpingDistance", area=AREA, status="added",
   notes="CanonicalWarpingDistance[s1, s2, init, win]: canonical time warping (Zhou & De la Torre 2009): from the "
         "initial correspondence (default: the uniform alignment of length Max[n1, n2]) alternate regularized "
         "CCA on the aligned sequences, centred by their means along the current alignment (α^T C11 α = I, β^T C22 β = I, ridge λ_i = 10^-8 tr(C_ii)/d_i "
         "unless Method -> {\"Lambdas\" -> {λ1, λ2}}, \"DimensionsToKeep\" default Min[d1, d2]) and squared-"
         "Euclidean dynamic time warping of the projections (steps (1,0), (0,1), (1,1); window r / {\"SlantedBand\", "
         "r} / {\"Band\", r}), until the path repeats or MaxIterations (default 20); the result is the smallest "
         "Σ_k ||α^T (s1[[n_k]] - μ1) - β^T (s2[[m_k]] - μ2)||² met (μ_i = means along the alignment). Deviation: Wolfram's defaults are undocumented, "
         "so numbers differ from Wolfram's; constant sequences raise ValueError; \"Parallelogram\" windows, "
         "other DistanceFunctions, \"EnergyThreshold\"/\"MatchingIntervals\" and quantities raise "
         "NotImplementedError. Check: reference (sequences related by an affine map and a time warp have distance "
         "≈ 0 with the true correspondence; the distance of the returned correspondence recomputed directly) in "
         "tests.")(_ctw_distance)


def _ctw_corr(s1: Expr, s2: Expr, *rest: Expr) -> ListOf[ListOf[Integer]]:
    _J, path = _ctw(s1, s2, rest, "CanonicalWarpingCorrespondence")
    return L(L(*[p[0] + 1 for p in path]), L(*[p[1] + 1 for p in path]))


fn("CanonicalWarpingCorrespondence", area=AREA, status="added",
   notes="CanonicalWarpingCorrespondence[s1, s2, init, win] = {{n1, …}, {m1, …}}: the 1-based monotone "
         "correspondence of the alignment whose CTW distance CanonicalWarpingDistance reports (same algorithm, "
         "defaults and restrictions). Check: reference (recovery of a known warp of an affinely transformed "
         "sequence; endpoints {1, 1} and {n1, n2}) in tests.")(_ctw_corr)


# ═══════════════════════════════════════════════════════════════ InverseHilbertTransform ══
def _sign(a):
    return sp.sign(a)


def _iht_term(term, x, y, fname):
    """Inverse Hilbert transform of one additive term (convention IH = -H, H f(y) = (1/π) PV ∫ f(x)/(y - x) dx)."""
    if not term.has(x):
        return sp.Integer(0)
    # Dirac delta δ(x - a) (times a constant)
    c, rest = term.as_independent(x, as_Add=False)
    if isinstance(rest, sp.DiracDelta) and len(rest.args) == 1:
        arg = rest.args[0]
        if sp.Poly(arg, x).degree() == 1:
            k = sp.Poly(arg, x).LC()
            a = sp.solve(arg, x)[0]
            return -c / (sp.pi * sp.Abs(k) * (y - a))
    # exponential / trigonometric terms: rewrite as Σ c e^{i a x}
    if term.has(sp.sin, sp.cos, sp.exp) and not term.has(sp.DiracDelta):
        ex = sp.expand(sp.powsimp(sp.expand(term.rewrite(sp.exp))))
        out = sp.Integer(0)
        for t in sp.Add.make_args(ex):
            cc, ee = t.as_independent(x, as_Add=False)
            if ee == 1:
                continue
            if not isinstance(ee, sp.exp):
                ee2 = sp.powsimp(ee)
                if not isinstance(ee2, sp.exp):
                    raise NotImplementedError("unsupported")
                ee = ee2
            arg = sp.expand(ee.args[0])
            if not arg.is_polynomial(x) or sp.Poly(arg, x).degree() != 1:
                raise NotImplementedError("unsupported")
            k = sp.Poly(arg, x).LC()
            b0 = arg - k * x
            a = sp.simplify(k / sp.I)
            if a.has(sp.I) or a.is_real is False:
                raise NotImplementedError("unsupported")
            if b0.has(x):
                raise NotImplementedError("unsupported")
            out += cc * sp.I * _sign(a) * sp.exp(sp.I * a * y + b0)
        return out
    # rational terms via partial fractions over the poles
    if term.is_rational_function(x):
        num, den = sp.fraction(sp.cancel(sp.together(term)))
        if sp.degree(num, x) >= sp.degree(den, x):
            raise NotImplementedError("non-decaying rational part")
        out = sp.Integer(0)
        lcd = sp.Poly(den, x).LC()
        for p, mult in sp.roots(sp.Poly(den, x)).items():
            for kk in range(1, mult + 1):
                # coefficient of 1/(x-p)^kk
                g = sp.cancel(num / lcd * (x - p) ** mult / sp.cancel(den / lcd))
                j = mult - kk
                coef = sp.simplify(sp.diff(g, x, j).subs(x, p) / sp.factorial(j))
                if coef == 0:
                    continue
                im = sp.im(p)
                if im.is_number and im == 0 or im == 0:
                    # real pole: IH[1/(x-a)^k] = π (-1)^{k-1}/(k-1)! δ^{(k-1)}(y - a)
                    dd = sp.DiracDelta(y - p) if kk == 1 else sp.DiracDelta(y - p, kk - 1)
                    out += coef * sp.pi * (-1) ** (kk - 1) / sp.factorial(kk - 1) * dd
                elif im.is_number:
                    out += coef * (sp.I if im < 0 else -sp.I) / (y - p) ** kk
                else:
                    raise NotImplementedError("pole with unknown half-plane")
        if sum(sp.roots(sp.Poly(den, x)).values()) != sp.degree(den, x):
            raise NotImplementedError("unsolvable denominator")
        return out
    # compactly supported piecewise polynomials (UnitBox, Piecewise, Sign, HeavisideTheta products)
    return _iht_piecewise(term, x, y, fname)


def _iht_piecewise(term, x, y, fname):
    ub = [f for f in term.atoms(sp.Function) if type(f).__name__ == "UnitBox"]
    e = term
    for f in ub:
        e = e.subs(f, sp.Piecewise((1, sp.Abs(f.args[0]) < sp.Rational(1, 2)), (0, True)))
    hs = [f for f in e.atoms(sp.Function) if type(f).__name__ in ("HeavisideTheta", "Heaviside")]
    for f in hs:
        e = e.subs(f, sp.Piecewise((1, f.args[0] > 0), (0, True)))
    e = e.replace(sp.sign, lambda a: sp.Piecewise((1, a > 0), (-1, a < 0), (0, True)))
    e = sp.piecewise_fold(e)
    if not isinstance(e, sp.Piecewise):
        raise NotImplementedError("unsupported")
    out = sp.Integer(0)
    xr = sp.Symbol(x.name, real=True)
    e = e.subs(x, xr)
    for expr, st in e.as_expr_set_pairs(sp.Reals):
        expr = sp.simplify(expr)
        if expr == 0:
            continue
        if not expr.is_polynomial(xr):
            raise NotImplementedError("unsupported")
        ivs = st.args if isinstance(st, sp.Union) else [st]
        for iv in ivs:
            if not isinstance(iv, sp.Interval) or iv.inf.is_infinite or iv.sup.is_infinite:
                if isinstance(iv, sp.FiniteSet):
                    continue
                raise NotImplementedError("non-compact support")
            a, b = iv.inf, iv.sup
            pe = expr.subs(xr, x)
            q = sp.cancel((pe - pe.subs(x, y)) / (y - x))
            poly_part = sp.integrate(q, (x, a, b))
            out += -(pe.subs(x, y) * sp.log(sp.Abs((y - a) / (y - b))) + poly_part) / sp.pi
    return out


def _inverse_hilbert(f: Expr, t: Expr, s: Expr, *rest: Expr) -> Expr:
    fname = "InverseHilbertTransform"
    _args, opts = _opts(rest, {"Assumptions", "GenerateConditions", "Method", "AccuracyGoal", "PrecisionGoal",
                               "WorkingPrecision"}, fname)
    if _args:
        raise ValueError(f"{fname}: called with too many arguments")
    tv = _var(t, fname)
    x = sp.Symbol(tv.name, real=True)
    fe = _sp_pw(f).subs(sp.Symbol(tv.name), x)
    sv = s if isinstance(s, Symbol) else None
    y = sp.Symbol(sv.name, real=True) if sv is not None else sp.Symbol("_y_iht", real=True)
    out = sp.Integer(0)
    try:
        for term in sp.Add.make_args(sp.expand(fe)):
            out += _iht_term(term, x, y, fname)
    except NotImplementedError:
        if fe.free_symbols - {x} and not fe.has(sp.Function):
            raise Unevaluated()
        raise Unevaluated()
    out = sp.simplify(out.rewrite(sp.cos)) if out.has(sp.exp) else sp.simplify(out)
    if out.has(sp.I) and not fe.has(sp.I):
        out2 = sp.simplify(sp.expand_complex(out))
        if not out2.has(sp.I):
            out = out2
    if "Assumptions" in opts:
        res = ev(Expr("Refine", R(out.subs(y, sp.Symbol(y.name))), opts["Assumptions"]))
    else:
        res = R(out.subs(y, sp.Symbol(y.name)))
    if sv is None:
        val = _sp(s)
        if not val.is_number:
            raise TypeError(f"{fname}: the third argument must be a symbol or a number, not {s!r}")
        num = out.subs(y, val)
        return R(sp.N(num) if any(isinstance(a, sp.Float) for a in sp.preorder_traversal(val)) else
                 sp.simplify(num))
    return res


fn("InverseHilbertTransform", area=AREA, status="added",
   notes="InverseHilbertTransform[F, t, s] = -(1/π) PV ∫ F(t)/(s - t) dt (= minus HilbertTransform with the "
         "kernel 1/(π(s - t))), in closed form for sums of: constants (→ 0); e^{iat}, Cos[a t + b], Sin[a t + b] "
         "with real a (IH[e^{iat}] = i Sign[a] e^{ias}, so IH[Cos[t]] = -Sin[s], IH[Sin[t]] = Cos[s]); proper "
         "rational functions by partial fractions (IH[1/(t - p)^k] = i/(s - p)^k for Im p < 0, -i/(s - p)^k for "
         "Im p > 0, π (-1)^{k-1}/(k-1)! DiracDelta^{(k-1)}[s - p] for real p); DiracDelta[t - a] (→ -1/(π (s - "
         "a))); compactly supported piecewise polynomials (UnitBox, Piecewise, Sign/HeavisideTheta products: "
         "-(1/π)[p(s) Log|(s - a)/(s - b)| + ∫_a^b (p(t) - p(s))/(s - t) dt]); a numeric s evaluates the closed "
         "form; anything else stays unevaluated (no numeric principal-value integration). Check: reference "
         "(Wolfram documentation examples and mpmath principal-value quadrature at sample points) in "
         "tests.")(_inverse_hilbert)


# ═══════════════════════════════════════════════════════════════ InverseBilateralLaplaceTransform ══
def _strip(cond, sv, fname, var="Re"):
    """(lo, hi) bounds of Re[s] (or Abs[z]) from a condition."""
    lo, hi = -sp.oo if var == "Re" else sp.Integer(0), sp.oo
    parts = sp.And.make_args(cond) if isinstance(cond, sp.And) else [cond]
    target = sp.re(sv) if var == "Re" else sp.Abs(sv)
    for c in parts:
        if not isinstance(c, sp.core.relational.Relational):
            raise ValueError(f"{fname}: the region condition must be inequalities in {var}[{sv}]")
        lhs, rhs = c.lhs, c.rhs
        if isinstance(c, (sp.StrictGreaterThan, sp.GreaterThan)):
            lhs, rhs = rhs, lhs
        # now lhs < rhs
        if lhs == target and rhs.is_number:
            hi = min(hi, rhs)
        elif rhs == target and lhs.is_number:
            lo = max(lo, lhs)
        else:
            raise ValueError(f"{fname}: the region condition must be inequalities in {var}[{sv}]")
    return lo, hi


def _cond_input(F, var, fname, kind):
    if isinstance(F, Expr) and F.head == "ConditionalExpression" and len(F.args) == 2:
        sv = _sp(var)
        try:
            cond = _cond_sp(F.args[1])
        except (ValueError, KeyError, AttributeError):
            raise ValueError(f"{fname}: the region condition must be inequalities in {kind}[{var}]")
        return _sp(F.args[0]), _strip(cond, sv, fname, kind)
    return _sp(F), None


def _exp_groups(F, x):
    """F = Σ R_k(x) e^{-τ_k x} -> {τ_k: R_k}."""
    W = sp.Dummy("W", positive=True)
    F = F.replace(lambda u: isinstance(u, sp.Pow) and isinstance(u.base, sp.exp),
                  lambda u: sp.exp(u.base.args[0] * u.exp))

    def rep(arg):
        arg = sp.expand(arg)
        if not arg.has(x):
            return sp.exp(arg)
        if not (arg.is_polynomial(x) and sp.degree(arg, x) == 1):
            raise NotImplementedError("unsupported exponential")
        k = sp.Poly(arg, x).LC()
        return W ** k * sp.exp(arg - k * x)
    F2 = F.replace(lambda u: isinstance(u, sp.exp), lambda u: rep(u.args[0]))
    if F2.has(sp.exp) and any(a.has(x) for a in F2.atoms(sp.exp)):
        raise NotImplementedError("unsupported exponential")
    groups = {}
    for t in sp.Add.make_args(sp.expand(F2, power_exp=True)):
        coeff, k = t.as_coeff_exponent(W)
        if coeff.has(W):
            raise NotImplementedError("unsupported exponential")
        tau = -k
        groups[tau] = groups.get(tau, 0) + coeff
    return groups


def _ibl_rational(Rx, s, t, lo, hi, fname):
    num, den = sp.fraction(sp.cancel(sp.together(Rx)))
    if not (num.is_polynomial(s) and den.is_polynomial(s)):
        raise NotImplementedError("non-rational")
    q, r = sp.div(sp.Poly(num, s), sp.Poly(den, s))
    out = sp.Integer(0)
    for k, c in enumerate(q.all_coeffs()[::-1]):
        if c != 0:
            out += c * (sp.DiracDelta(t) if k == 0 else sp.DiracDelta(t, k))
    if r.is_zero:
        return out
    dpoly = sp.Poly(den, s)
    rts = sp.roots(dpoly)
    if sum(rts.values()) != dpoly.degree():
        raise NotImplementedError("unsolvable denominator")
    lcd = dpoly.LC()
    rem = r.as_expr()
    for p, mult in rts.items():
        g = sp.cancel(rem / lcd * (s - p) ** mult / sp.cancel(den / lcd))
        rp = sp.re(p)
        if not rp.is_number:
            raise Unevaluated()
        if (lo == -sp.oo or rp > lo) and (hi == sp.oo or rp < hi):
            raise ValueError(f"{fname}: the pole {R(p)!r} lies inside the region of convergence")
        causal = lo != -sp.oo and rp <= lo
        for kk in range(1, mult + 1):
            j = mult - kk
            coef = sp.simplify(sp.diff(g, s, j).subs(s, p) / sp.factorial(j))
            if coef == 0:
                continue
            base = coef * t ** (kk - 1) / sp.factorial(kk - 1) * sp.exp(p * t)
            out += base * sp.Heaviside(t) if causal else -base * sp.Heaviside(-t)
    return out


def _check_laplace(f, F, s, t, lo, hi, fname):
    """Numeric bilateral Laplace transform of f at a real point of the strip versus F."""
    if lo == -sp.oo and hi == sp.oo:
        s0 = 0.3
    elif lo == -sp.oo:
        s0 = float(hi) - 1
    elif hi == sp.oo:
        s0 = float(lo) + 1
    else:
        s0 = (float(lo) + float(hi)) / 2
    deltas = sp.Integer(0)
    smooth = sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(f)):
        if term.has(sp.DiracDelta):
            c, d = term.as_independent(t, as_Add=False)
            if isinstance(d, sp.DiracDelta):
                arg = d.args[0]
                k = d.args[1] if len(d.args) > 1 else 0
                tau = sp.solve(arg, t)[0]
                deltas += c * sp.Integer(s0) ** 0 * (sp.Float(s0) ** k) * sp.exp(-sp.Float(s0) * tau)
            else:
                return
        else:
            smooth += term
    if smooth.free_symbols - {t} or F.free_symbols - {s}:
        return
    g = sp.lambdify(t, smooth.rewrite(sp.Piecewise) * sp.exp(-s0 * t), "mpmath")
    brk = {0.0}
    for hv in smooth.atoms(sp.Heaviside):
        for r0 in sp.solve(hv.args[0], t):
            if r0.is_number:
                brk.add(float(r0))
    pts = [-mpmath.inf] + sorted(brk) + [mpmath.inf]
    try:
        val = mpmath.quad(lambda u: g(u), pts) + complex(sp.N(deltas))
    except Exception:
        return
    ref = complex(sp.N(F.subs(s, s0)))
    if abs(complex(val) - ref) > 1e-6 * max(1.0, abs(ref)):
        raise ArithmeticError(f"{fname}: result failed verification (numeric transform at s = {s0} differs)")


def _inverse_bilateral_laplace(F: Expr, s: Expr, t: Expr, *rest: Expr) -> Expr:
    fname = "InverseBilateralLaplaceTransform"
    _args, opts = _opts(rest, {"Assumptions", "GenerateConditions"}, fname)
    if _args:
        raise ValueError(f"{fname}: called with too many arguments")
    _var(s, fname)
    sv = _sp(s)
    F0, region = _cond_input(F, s, fname, "Re")
    if is_list(t):
        raise NotImplementedError(f"{fname}: multivariate transforms are not supported")
    tnum = None
    if isinstance(t, Symbol):
        tv = sp.Symbol(t.name, real=True)
    else:
        tnum = _sp(t)
        if not tnum.is_number:
            raise TypeError(f"{fname}: the result variable must be a symbol or a number")
        tv = sp.Symbol("_t_ibl", real=True)
    if F0.free_symbols - {sv}:
        raise Unevaluated()
    lo, hi = region if region is not None else (-sp.oo, sp.oo)
    if region is not None and lo >= hi:
        raise ValueError(f"{fname}: the strip {R(lo)} < Re[{s}] < {R(hi)} is empty")
    try:
        groups = _exp_groups(F0, sv)
        out = sp.Integer(0)
        for tau, Rx in groups.items():
            if not Rx.is_rational_function(sv):
                raise NotImplementedError("non-rational")
            num, den = sp.fraction(sp.cancel(sp.together(Rx)))
            if den.has(sv) and region is None:
                raise ValueError(f"{fname}: the region of convergence is ambiguous; give the input as "
                                 f"ConditionalExpression[F, a < Re[{s}] < b]")
            out += _ibl_rational(Rx, sv, tv, lo, hi, fname).subs(tv, tv - tau)
    except NotImplementedError:
        raise Unevaluated()
    out = sp.simplify(sp.expand_complex(out)) if out.has(sp.I) else out
    _check_laplace(out, F0, sv, tv, lo, hi, fname)
    if tnum is not None:
        val = out.subs(tv, tnum)
        return R(sp.N(val) if any(isinstance(a, sp.Float) for a in sp.preorder_traversal(tnum)) else
                 sp.simplify(val))
    return R(out.subs(tv, sp.Symbol(t.name)))


fn("InverseBilateralLaplaceTransform", area=AREA, status="added",
   notes="InverseBilateralLaplaceTransform[ConditionalExpression[F, a < Re[s] < b], s, t] for F = Σ R_k(s) e^{-τ_k s} "
         "with rational R_k: partial fractions over the poles; a pole p left of the strip (Re p ≤ a) gives the "
         "causal term c t^{k-1}/(k-1)! e^{pt} HeavisideTheta[t], one right of it the anticausal term -c t^{k-1}/"
         "(k-1)! e^{pt} HeavisideTheta[-t]; the polynomial part gives DiracDelta derivatives; e^{-τ s} shifts t → "
         "t - τ. A strip that is empty or contains a pole strictly inside raises ValueError; without a ConditionalExpression "
         "only entire (polynomial) F is accepted, otherwise the region is ambiguous and ValueError is raised; "
         "non-rational F stays unevaluated; a numeric t evaluates. Check: numeric cross-check (mpmath quadrature "
         "of the bilateral integral at a real point of the strip, ArithmeticError on mismatch) and reference "
         "(BilateralLaplaceTransform round trips) in tests.")(_inverse_bilateral_laplace)


# ═══════════════════════════════════════════════════════════════ InverseBilateralZTransform ══
def _ibz_rational(F, z, n, lo, hi, fname):
    G = sp.cancel(sp.together(F / z))
    num, den = sp.fraction(G)
    q, r = sp.div(sp.Poly(num, z), sp.Poly(den, z))
    US = sp.Function("UnitStep")
    DD = sp.Function("DiscreteDelta")
    out = sp.Integer(0)
    for k, c in enumerate(q.all_coeffs()[::-1]):
        if c != 0:
            out += c * DD(n + k + 1)          # z · z^k ↔ δ[n + k + 1]
    if r.is_zero:
        return out
    dpoly = sp.Poly(den, z)
    rts = sp.roots(dpoly)
    if sum(rts.values()) != dpoly.degree():
        raise NotImplementedError("unsolvable denominator")
    lcd = dpoly.LC()
    rem = r.as_expr()
    for p, mult in rts.items():
        g = sp.cancel(rem / lcd * (z - p) ** mult / sp.cancel(den / lcd))
        for kk in range(1, mult + 1):
            j = mult - kk
            coef = sp.simplify(sp.diff(g, z, j).subs(z, p) / sp.factorial(j))
            if coef == 0:
                continue
            if p == 0:
                out += coef * DD(n - kk + 1)     # z / z^k ↔ δ[n - (k - 1)]
                continue
            ap = sp.Abs(p)
            if not ap.is_number:
                raise Unevaluated()
            if lo < ap < hi:
                raise ValueError(f"{fname}: the pole {R(p)!r} lies inside the annulus of convergence")
            poly = sp.ff(n, kk - 1) / sp.factorial(kk - 1)
            pw = p ** (n - kk + 1) if p.is_real else \
                sp.Abs(p) ** (n - kk + 1) * sp.exp(sp.I * sp.arg(p) * (n - kk + 1))
            base = coef * poly * pw
            out += base * US(n) if ap <= lo else -base * US(-n - 1)
    return out


def _check_z(f, F, z, n, lo, hi, fname):
    if f.free_symbols - {n} or F.free_symbols - {z}:
        return
    lo_f = float(lo)
    hi_f = float(hi) if hi != sp.oo else max(2 * lo_f, lo_f + 2, 2.0)
    z0 = math.sqrt(lo_f * hi_f) if lo_f > 0 else hi_f / 2
    if hi == sp.oo:
        z0 = max(z0, lo_f + 1)
    fun = sp.lambdify(n, f.replace(sp.Function("UnitStep"), lambda a: sp.Piecewise((1, a >= 0), (0, True)))
                      .replace(sp.Function("DiscreteDelta"), lambda a: sp.Piecewise((1, sp.Eq(a, 0)), (0, True))),
                      "mpmath")
    rho = []
    if lo_f > 0:
        rho.append(lo_f / z0)
    if hi != sp.oo:
        rho.append(z0 / hi_f)
    r_ = max(rho) if rho else 0.5
    N = min(4000, int(60 / max(1e-3, -math.log(r_))) + 20) if r_ > 0 else 50
    try:
        with mpmath.workdps(30):
            tot = mpmath.fsum(fun(k) * mpmath.mpf(z0) ** (-k) for k in range(-N, N + 1))
    except Exception:
        return
    ref = complex(sp.N(F.subs(z, z0)))
    if abs(complex(tot) - ref) > 1e-6 * max(1.0, abs(ref)):
        raise ArithmeticError(f"{fname}: result failed verification (series sum at z = {z0} differs)")


def _inverse_bilateral_z(F: Expr, z: Expr, n: Expr, *rest: Expr) -> Expr:
    fname = "InverseBilateralZTransform"
    _args, opts = _opts(rest, {"Assumptions", "GenerateConditions"}, fname)
    if _args:
        raise ValueError(f"{fname}: called with too many arguments")
    _var(z, fname)
    zv = _sp(z)
    F0, region = _cond_input(F, z, fname, "Abs")
    if is_list(n):
        raise NotImplementedError(f"{fname}: multivariate transforms are not supported")
    nnum = None
    if isinstance(n, Symbol):
        nv = sp.Symbol(n.name, integer=True)
    else:
        nnum = _sp(n)
        if not (nnum.is_number and nnum.is_integer):
            raise TypeError(f"{fname}: the result variable must be a symbol or an integer")
        nv = sp.Symbol("_n_ibz", integer=True)
    if F0.free_symbols - {zv}:
        raise Unevaluated()
    if not F0.is_rational_function(zv):
        raise Unevaluated()
    num, den = sp.fraction(sp.cancel(sp.together(F0)))
    has_poles = any(p != 0 for p in sp.roots(sp.Poly(den, zv)).keys()) if den.has(zv) else False
    if region is None:
        if has_poles:
            raise ValueError(f"{fname}: the region of convergence is ambiguous; give the input as "
                             f"ConditionalExpression[F, r1 < Abs[{z}] < r2]")
        lo, hi = sp.Integer(0), sp.oo
    else:
        lo, hi = region
        if lo >= hi:
            raise ValueError(f"{fname}: the annulus {R(lo)} < Abs[{z}] < {R(hi)} is empty")
    try:
        out = _ibz_rational(F0, zv, nv, lo, hi, fname)
    except NotImplementedError:
        raise Unevaluated()
    if out.has(sp.I):
        nr = sp.Symbol("_nr_ibz", real=True)
        fs = sorted(out.atoms(sp.Function("UnitStep")) | out.atoms(sp.Function("DiscreteDelta")), key=str)
        dums = {f: sp.Dummy(real=True) for f in fs}
        o2 = out.subs(dums).subs(nv, nr)
        o2 = sp.simplify(sp.expand_complex(o2))
        if not o2.has(sp.I) or any(not c.is_real for c in (sp.Poly(num, zv).all_coeffs() +
                                                             sp.Poly(den, zv).all_coeffs())):
            out = o2.subs(nr, nv).subs({d: f for f, d in dums.items()})
    else:
        out = sp.expand(out)
    _check_z(out, F0, zv, nv, lo, hi, fname)
    if nnum is not None:
        val = out.subs(nv, nnum)
        val = val.replace(sp.Function("UnitStep"), lambda a: sp.Integer(1) if a >= 0 else sp.Integer(0))
        val = val.replace(sp.Function("DiscreteDelta"), lambda a: sp.Integer(1) if a == 0 else sp.Integer(0))
        return R(sp.simplify(val))
    return R(out.subs(nv, sp.Symbol(n.name)))


fn("InverseBilateralZTransform", area=AREA, status="added",
   notes="InverseBilateralZTransform[ConditionalExpression[F, r1 < Abs[z] < r2], z, n] for rational F: partial "
         "fractions of F/z; z/(z - p)^k gives Binomial-type terms FactorialPower[n, k-1]/(k-1)! p^{n-k+1} UnitStep[n] "
         "for |p| ≤ r1 (causal) and -FactorialPower[n, k-1]/(k-1)! p^{n-k+1} UnitStep[-n-1] for |p| ≥ r2 "
         "(anticausal); poles at 0 and the polynomial part give DiscreteDelta terms. An empty annulus or a pole "
         "strictly inside it raises ValueError; without a ConditionalExpression only F whose poles are all at 0 is "
         "accepted (else the region is ambiguous: ValueError); non-rational F stays unevaluated; an integer n "
         "evaluates. Check: numeric cross-check (the bilateral series Σ f(n) z0^-n summed at a point of the "
         "annulus, ArithmeticError on mismatch) and reference (BilateralZTransform round trips) in "
         "tests.")(_inverse_bilateral_z)
