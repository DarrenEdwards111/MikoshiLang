"""Wolfram batch W26: NDSolve extensions, numerical options and PDE modelling components.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).

* **PDE terms** (``LaplacianPDETerm``, ``DiffusionPDETerm``, ``ConvectionPDETerm``,
  ``ConservativeConvectionPDETerm``, ``ReactionPDETerm``, ``SourcePDETerm``, ``DerivativePDETerm``) build the
  documented inactive operator expressions, e.g. ``DiffusionPDETerm[{u[x], {x}}, 1]`` ->
  ``Inactive[Div][{{-1}} . Inactive[Grad][u[x], {x}], {x}]``; ``Activate`` turns them into derivatives.
* **PDE components** (heat transfer, mass transport, acoustics, electrostatics, electric currents,
  magnetostatics, magnetics, Poisson, Helmholtz, wave, Schrodinger, solid mechanics and fluid flow) combine those
  terms from the documented model parameters; the matching **boundary values / conditions** evaluate to
  ``NeumannValue[...]`` and ``DirichletCondition[...]`` expressions.
* **Solvers:** ``NDEigenvalues`` / ``NDEigensystem`` (Chebyshev spectral collocation on intervals and rectangles,
  with a resolution self-check), ``ParametricNDSolve`` / ``ParametricNDSolveValue`` / ``ParametricFunction``
  (re-using ``NDSolveValue`` from ``wl_interp``), and the 1-D boundary-value extension of ``NDSolve`` for
  ``DirichletCondition`` / ``NeumannValue`` / two-point conditions.  ``WhenEvent`` and the step options
  (``MaxStepSize``, ``StartingStepSize``, ``MaxSteps``, ``MaxStepFraction``, ``AccuracyGoal``, ``PrecisionGoal``)
  are implemented inside ``wl_interp``'s ``NDSolve``.

Representation note: operators are returned as inactive ``Div``/``Grad``/``Curl``/``Dot`` expressions that are
mathematically equal to Wolfram's after ``Activate``; where Wolfram rewrites an operator into its internal
coefficient form (magnetic curl-curl operators, solid and fluid stress divergences) the structure differs, which
the notes of those heads state.
"""
from __future__ import annotations

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, Unevaluated, L, is_list, is_numeric_value, S, R
from .wl_common import ev

AREA = "numerics: PDE modelling (W26)"

# ───────────────────────────── small expression helpers ─────────────────────────────


def E(h, *a):
    return Expr(h, *a)


def _inactive(h):
    return Expr("Inactive", Symbol(h))


def _grad(u, X):
    return Expr(_inactive("Grad"), u, X)


def _div(F, X):
    return Expr(_inactive("Div"), F, X)


def _curl(F, X):
    return Expr(_inactive("Curl"), F, X)


def _plus(*a):
    a = [x for x in a if not _is_zero(x)]
    if not a:
        return 0
    return a[0] if len(a) == 1 else Expr("Plus", *a)


def _times(*a):
    if any(_is_zero(x) for x in a):
        return 0
    a = [x for x in a if not (isinstance(x, int) and not isinstance(x, bool) and x == 1)]
    if not a:
        return 1
    return a[0] if len(a) == 1 else Expr("Times", *a)


def _neg(x):
    return _times(-1, x)


def _is_zero(x):
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)) and x == 0 and not isinstance(x, float):
        return True
    if is_list(x):
        return len(x.args) > 0 and all(_is_zero(a) for a in x.args)
    try:
        v = S(x)
    except Exception:
        return False
    return isinstance(v, sp.Basic) and v.is_number and v == 0 and not isinstance(v, sp.Float)


def _is_num(x):
    return not isinstance(x, (bool, str)) and not is_list(x) and is_numeric_value(x)


def _is_matrix(x, n=None):
    if not is_list(x) or not x.args or not all(is_list(r) for r in x.args):
        return False
    m = len(x.args)
    if any(len(r.args) != m for r in x.args):
        return False
    return n is None or m == n


def _identity(n, c=1):
    return L(*(L(*((c if i == j else 0) for j in range(n))) for i in range(n)))


def _head_is(x, h):
    return isinstance(x, Expr) and x.head == h


def _is_symbol_head(x, names):
    return isinstance(x, Expr) and isinstance(x.head, Expr) and x.head.head == "Inactive" and \
        isinstance(x.head.args[0], Symbol) and x.head.args[0].name in names


def _sym_of(x):
    return x.name if isinstance(x, Symbol) else None


# ───────────────────────────── model variables ─────────────────────────────

class Vars:
    """Parsed model variables {u[...], {x...}} / {u[t, ...], t, {x...}} / {p[...], omega, {x...}}."""

    def __init__(self, deps, X, t=None, w=None, system=False, src=None):
        self.deps, self.X, self.t, self.w, self.system, self.src = deps, X, t, w, system, src
        self.n = len(X)
        self.XL = L(*X)

    @property
    def u(self):
        return self.deps[0]


def parse_vars(v, fname, allow_freq=False, allow_system=True, require_time=False):
    if not is_list(v):
        if isinstance(v, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the model variables must be a list {{u[x, ...], {{x, ...}}}}, not {v}")
    if len(v.args) not in (2, 3):
        raise ValueError(f"{fname}: the model variables must be {{u[x, ...], {{x, ...}}}} or "
                         "{u[t, x, ...], t, {x, ...}}")
    dep, X = v.args[0], v.args[-1]
    if not is_list(X) or not X.args or not all(isinstance(x, Symbol) for x in X.args):
        raise ValueError(f"{fname}: the independent variables must be a non-empty list of symbols")
    Xs = list(X.args)
    if len({x.name for x in Xs}) != len(Xs):
        raise ValueError(f"{fname}: the independent variables must be distinct")
    system = is_list(dep)
    deps = list(dep.args) if system else [dep]
    if system and not allow_system:
        raise NotImplementedError(f"{fname}: systems of dependent variables are not supported")
    if not deps:
        raise ValueError(f"{fname}: no dependent variable given")
    for d in deps:
        if not (isinstance(d, Expr) and isinstance(d.head, str) and d.args and
                all(isinstance(a, Symbol) for a in d.args)):
            raise ValueError(f"{fname}: dependent variables must be of the form u[x, ...], not {d}")
    t = w = None
    mid = v.args[1] if len(v.args) == 3 else None
    if mid is not None and not isinstance(mid, Symbol):
        raise ValueError(f"{fname}: the time or frequency variable must be a symbol")
    names = [x.name for x in Xs]
    for d in deps:
        extra = [a for a in d.args if a.name not in names]
        missing = [x for x in names if x not in [a.name for a in d.args]]
        if missing and not (fname == "FluidFlowPDEComponent"):
            raise ValueError(f"{fname}: {d} does not depend on {', '.join(missing)}")
        if len(extra) > 1:
            raise ValueError(f"{fname}: {d} has more than one argument besides the independent variables")
        if extra:
            if mid is not None and extra[0].name != mid.name:
                raise ValueError(f"{fname}: {d} depends on {extra[0]}, not on the time variable {mid}")
            if t is not None and t.name != extra[0].name:
                raise ValueError(f"{fname}: inconsistent time variables")
            t = extra[0]
    if mid is not None and t is None:
        if mid.name in names:
            raise ValueError(f"{fname}: {mid} is both a time/frequency and an independent variable")
        w = mid
        if not allow_freq:
            w = None     # Wolfram ignores a middle symbol the dependent variable does not depend on
    if require_time and t is None:
        raise ValueError(f"{fname}: time-dependent model variables {{u[t, x, ...], t, {{x, ...}}}} are required")
    return Vars(deps, Xs, t, w, system, v)


# ───────────────────────────── parameters ─────────────────────────────

def parse_pars(p, fname, known=(), unsupported=()):
    """<|"Key" -> value, sym -> value|> -> (dict of string keys, list of symbol substitution rules)."""
    if p is None:
        return {}, []
    if not KL.is_assoc(p):
        if isinstance(p, Symbol):
            raise Unevaluated()
        if _head_is(p, "Entity"):
            raise NotImplementedError(f"{fname}: material entities (Entity[...]) are not supported; give the "
                                      "material constants as parameters")
        raise TypeError(f"{fname}: the parameters must be an association, not {p}")
    keys, subs = {}, []
    for r in p.args:
        k, val = r.args[0], (ev(r.args[1]) if r.head == "RuleDelayed" else r.args[1])
        if isinstance(k, str):
            if k in unsupported:
                raise NotImplementedError(f"{fname}: the parameter \"{k}\" is not supported")
            if known and k not in known:
                raise ValueError(f"{fname}: unknown parameter \"{k}\"")
            keys[k] = val
        else:
            subs.append(Expr("Rule", k, val))
    return keys, subs


def _apply_subs(e, subs):
    if not subs:
        return e
    return ev(Expr("ReplaceAll", e, L(*subs)))


def _region_symmetry(keys, fname):
    rs = keys.get("RegionSymmetry", Symbol("None"))
    if _sym_of(rs) == "None":
        return False
    if rs == "Axisymmetric":
        return True
    raise ValueError(f"{fname}: \"RegionSymmetry\" must be None or \"Axisymmetric\", not {rs}")


def _chart(keys, fname):
    c = keys.get("CoordinateChart", "Cartesian")
    if c != "Cartesian":
        raise NotImplementedError(f"{fname}: only the \"Cartesian\" coordinate chart is supported")


# ───────────────────────────── term builders (single dependent variable) ─────────────────────────────

def _coef_matrix(c, n, fname, sign=-1):
    """Diffusion coefficient -> (kind, value): ('matrix', sign*C) or ('scalar', sign*c)."""
    if c is None:
        return "matrix", _identity(n, sign)
    if _head_is(c, "MatrixSymbol") or _head_is(c, "VectorSymbol"):
        raise NotImplementedError(f"{fname}: MatrixSymbol/VectorSymbol coefficients are not supported")
    if isinstance(c, (bool, str)):
        raise TypeError(f"{fname}: invalid coefficient {c!r}")
    if is_list(c):
        if all(not is_list(a) for a in c.args):
            if len(c.args) != n:
                raise ValueError(f"{fname}: a vector coefficient must have length {n}")
            return "matrix", ev(L(*(L(*((_times(sign, c.args[i]) if i == j else 0) for j in range(n)))
                                    for i in range(n))))
        if _is_matrix(c, n):
            return "matrix", ev(Expr("Times", sign, c))
        raise ValueError(f"{fname}: a matrix coefficient must be {n} x {n}")
    if _is_num(c):
        return "matrix", ev(_identity(n, Expr("Times", sign, c)))
    return "scalar", ev(Expr("Times", sign, c))


def _diffusion(u, X, c, fname, axi=False, sign=-1):
    """Inactive[Div][M . Inactive[Grad][u, X], X] with M = sign * c (sign=-1: flux -c grad u)."""
    n = len(X.args)
    kind, M = _coef_matrix(c, n, fname, sign)
    if _is_zero(M):
        return 0
    g = _grad(u, X)
    flux = Expr("Dot", M, g) if kind == "matrix" else Expr("Times", M, g)
    out = [_div(flux, X)]
    if axi:
        r = X.args[0]
        if n == 3:
            raise NotImplementedError(f"{fname}: axisymmetric models take the reduced variables {{r}} or {{r, z}}")
        row = M.args[0] if kind == "matrix" else L(*([M] + [0] * (n - 1)))
        vec = ev(Expr("Times", Expr("Power", r, -1), row))
        if not _is_zero(vec):
            out.insert(0, Expr("Dot", vec, g))
    return _plus(*out)


def _vector(c, n, fname, what="coefficient"):
    if _head_is(c, "VectorSymbol") or _head_is(c, "MatrixSymbol"):
        raise NotImplementedError(f"{fname}: MatrixSymbol/VectorSymbol coefficients are not supported")
    if not is_list(c) or any(is_list(a) for a in c.args):
        if isinstance(c, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: the {what} must be a vector of length {n}")
    if len(c.args) != n:
        raise ValueError(f"{fname}: the {what} must be a vector of length {n}")
    return c


def _convection(u, X, beta, fname):
    beta = _vector(beta, len(X.args), fname, "convection coefficient")
    if _is_zero(beta):
        return 0
    return Expr("Dot", beta, _grad(u, X))


def _cconvection(u, X, alpha, fname, axi=False):
    alpha = _vector(alpha, len(X.args), fname, "conservative convection coefficient")
    if _is_zero(alpha):
        return 0
    F = ev(Expr("Times", -1, alpha, u))
    out = [_div(F, X)]
    if axi:
        out.insert(0, ev(Expr("Times", F.args[0], Expr("Power", X.args[0], -1))))
    return _plus(*out)


def _derivterm(X, gamma, fname, axi=False):
    gamma = _vector(gamma, len(X.args), fname, "derivative coefficient")
    if _is_zero(gamma):
        return 0
    out = [_div(gamma, X)]
    if axi:
        out.insert(0, ev(Expr("Times", gamma.args[0], Expr("Power", X.args[0], -1))))
    return _plus(*out)


def _time_d(u, t, k=1):
    if k == 1:
        return Expr("D", u, t)
    return Expr("D", u, L(t, k))


# ───────────────────────────── PDE terms ─────────────────────────────

_TERM_PARS = ("CoordinateChart", "RegionSymmetry")


def _term_setup(vars_, pars, fname, keys_ok=_TERM_PARS):
    V = parse_vars(vars_, fname)
    keys, subs = parse_pars(pars, fname, known=keys_ok)
    _chart(keys, fname)
    axi = _region_symmetry(keys, fname) if "RegionSymmetry" in keys_ok else False
    return V, axi, subs


def _block(c, i, j, m, fname, default):
    """Coefficient block c[i][j] of an m x m system specification (scalar numeric c -> c on the diagonal)."""
    if c is None:
        return default if i == j else 0
    if _is_num(c) or not is_list(c):
        return c if i == j else 0
    if len(c.args) != m or not all(is_list(r) and len(r.args) == m for r in c.args):
        raise ValueError(f"{fname}: a coefficient for {m} dependent variables must be an {m} x {m} array of blocks")
    return c.args[i].args[j]


def _system(V, build):
    """Apply build(i, j, u_j) over a system and return the list of equation operators."""
    m = len(V.deps)
    rows = []
    for i in range(m):
        rows.append(_plus(*(build(i, j, V.deps[j]) for j in range(m))))
    return L(*rows)


@fn("LaplacianPDETerm", area=AREA, status="added",
    notes="LaplacianPDETerm[vars] and LaplacianPDETerm[vars, pars] -> Inactive[Div][IdentityMatrix[n] . "
          "Inactive[Grad][u, {x, ...}], {x, ...}] (Activate gives Laplacian[u, {x, ...}]); stationary and "
          "time-dependent vars, systems {{u, v, ...}, {x, ...}} (one Laplacian per equation); "
          "\"RegionSymmetry\" -> \"Axisymmetric\" adds the {1/r, 0} . Grad u term; other coordinate charts raise "
          "NotImplementedError; malformed vars raise. Check: reference (Wolfram documentation forms) and Activate "
          "compared with Laplacian in tests.")
def _laplacian_term(vars_: Expr, *rest: Assoc) -> Expr:
    fname = "LaplacianPDETerm"
    if len(rest) > 1:
        raise ValueError(f"{fname}: called with too many arguments")
    V, axi, subs = _term_setup(vars_, rest[0] if rest else None, fname)
    X = V.XL
    if V.system:
        out = _system(V, lambda i, j, u: _diffusion(u, X, None, fname, axi, sign=1) if i == j else 0)
    else:
        out = _diffusion(V.u, X, None, fname, axi, sign=1)
    return _apply_subs(ev(out), subs)


@fn("DiffusionPDETerm", area=AREA, status="added",
    notes="DiffusionPDETerm[vars], DiffusionPDETerm[vars, c] and DiffusionPDETerm[vars, c, pars] -> "
          "Inactive[Div][-c . Inactive[Grad][u, X], X]: no c or numeric scalar c -> -c IdentityMatrix[n]; vector c "
          "-> -DiagonalMatrix[c]; n x n matrix c -> -c; a symbolic scalar c -> -c Inactive[Grad][u, X]; systems take "
          "an m x m array of such blocks (numeric scalar -> diagonal blocks); symbol keys of pars are substituted; "
          "\"RegionSymmetry\" -> \"Axisymmetric\" adds the -(1/r) (c . Grad u)_r term; MatrixSymbol coefficients and "
          "other coordinate charts raise NotImplementedError; wrongly shaped coefficients raise ValueError. "
          "Check: reference (Wolfram documentation forms) and Activate compared with -Div[c Grad[u]] in tests.")
def _diffusion_term(vars_: Expr, *rest: Expr) -> Expr:
    fname = "DiffusionPDETerm"
    if len(rest) > 2:
        raise ValueError(f"{fname}: called with too many arguments")
    c = rest[0] if rest else None
    V, axi, subs = _term_setup(vars_, rest[1] if len(rest) > 1 else None, fname)
    X = V.XL
    if V.system:
        m = len(V.deps)
        out = _system(V, lambda i, j, u: (lambda b: 0 if _is_zero(b) else _diffusion(u, X, b, fname, axi))(
            _block(c, i, j, m, fname, 1)))
    else:
        out = _diffusion(V.u, X, c, fname, axi)
    return _apply_subs(ev(out), subs)


@fn("ConvectionPDETerm", area=AREA, status="added",
    notes="ConvectionPDETerm[vars, beta] and ConvectionPDETerm[vars, beta, pars] -> beta . Inactive[Grad][u, X] "
          "for a vector beta of length n (systems: an m x m array of vectors); symbol keys of pars are "
          "substituted; non-vector coefficients raise ValueError, Vector/MatrixSymbol coefficients and other "
          "coordinate charts raise NotImplementedError. Check: reference (Wolfram documentation forms) and "
          "Activate compared with beta . Grad[u] in tests.")
def _convection_term(vars_: Expr, beta: ListOf[Expr], *rest: Assoc) -> Expr:
    fname = "ConvectionPDETerm"
    if len(rest) > 1:
        raise ValueError(f"{fname}: called with too many arguments")
    V, axi, subs = _term_setup(vars_, rest[0] if rest else None, fname, ("CoordinateChart",))
    X = V.XL
    if V.system:
        m = len(V.deps)
        out = _system(V, lambda i, j, u: (lambda b: 0 if _is_zero(b) else _convection(u, X, b, fname))(
            _block(beta, i, j, m, fname, None)))
    else:
        out = _convection(V.u, X, beta, fname)
    return _apply_subs(ev(out), subs)


@fn("ConservativeConvectionPDETerm", area=AREA, status="added",
    notes="ConservativeConvectionPDETerm[vars, alpha] and [vars, alpha, pars] -> Inactive[Div][-alpha u, X] for a "
          "vector alpha of length n (systems: an m x m array of vectors); \"RegionSymmetry\" -> \"Axisymmetric\" "
          "adds -alpha_r u / r; symbol keys of pars are substituted; non-vector coefficients raise ValueError. "
          "Check: reference (Wolfram documentation forms) and Activate compared with -Div[alpha u] in tests.")
def _cconvection_term(vars_: Expr, alpha: ListOf[Expr], *rest: Assoc) -> Expr:
    fname = "ConservativeConvectionPDETerm"
    if len(rest) > 1:
        raise ValueError(f"{fname}: called with too many arguments")
    V, axi, subs = _term_setup(vars_, rest[0] if rest else None, fname)
    X = V.XL
    if V.system:
        m = len(V.deps)
        out = _system(V, lambda i, j, u: (lambda b: 0 if _is_zero(b) else _cconvection(u, X, b, fname, axi))(
            _block(alpha, i, j, m, fname, None)))
    else:
        out = _cconvection(V.u, X, alpha, fname, axi)
    return _apply_subs(ev(out), subs)


@fn("DerivativePDETerm", area=AREA, status="added",
    notes="DerivativePDETerm[vars, gamma] and [vars, gamma, pars] -> Inactive[Div][gamma, X] for a vector gamma "
          "of length n (systems: one vector, or {vector}, per equation); \"RegionSymmetry\" -> \"Axisymmetric\" "
          "adds gamma_r / r; non-vector coefficients raise ValueError. Check: reference (Wolfram documentation "
          "forms) and Activate compared with Div[gamma] in tests.")
def _derivative_term(vars_: Expr, gamma: ListOf[Expr], *rest: Assoc) -> Expr:
    fname = "DerivativePDETerm"
    if len(rest) > 1:
        raise ValueError(f"{fname}: called with too many arguments")
    V, axi, subs = _term_setup(vars_, rest[0] if rest else None, fname)
    X = V.XL
    if V.system:
        m = len(V.deps)
        if not is_list(gamma) or len(gamma.args) != m:
            raise ValueError(f"{fname}: give one coefficient vector per equation")
        rows = []
        for g in gamma.args:
            if is_list(g) and len(g.args) == 1 and is_list(g.args[0]):
                g = g.args[0]
            rows.append(_derivterm(X, g, fname, axi))
        out = L(*rows)
    else:
        out = _derivterm(X, gamma, fname, axi)
    return _apply_subs(ev(out), subs)


@fn("ReactionPDETerm", area=AREA, status="added",
    notes="ReactionPDETerm[vars, a] and [vars, a, pars] -> a u (systems: {Sum[a_ij u_j]} for an m x m matrix a); "
          "symbol keys of pars are substituted (ReactionPDETerm[{u[x], {x}}, a, <|a -> 2|>] -> 2 u[x]); a list "
          "coefficient for a single variable raises ValueError. Check: reference (Wolfram documentation forms).")
def _reaction_term(vars_: Expr, a: Expr, *rest: Assoc) -> Expr:
    fname = "ReactionPDETerm"
    if len(rest) > 1:
        raise ValueError(f"{fname}: called with too many arguments")
    V, axi, subs = _term_setup(vars_, rest[0] if rest else None, fname)
    if V.system:
        m = len(V.deps)
        if not (is_list(a) and len(a.args) == m and all(is_list(r) and len(r.args) == m for r in a.args)):
            if is_list(a):
                raise ValueError(f"{fname}: the reaction coefficient must be an {m} x {m} matrix")
            out = L(*(_times(a, u) for u in V.deps))
        else:
            out = _system(V, lambda i, j, u: _times(a.args[i].args[j], u))
    else:
        if is_list(a):
            raise ValueError(f"{fname}: the reaction coefficient for one dependent variable must be a scalar")
        out = _times(a, V.u)
    return _apply_subs(ev(out), subs)


@fn("SourcePDETerm", area=AREA, status="added",
    notes="SourcePDETerm[vars, f] and [vars, f, pars] -> f (the right-hand side of operator == SourcePDETerm[...]); "
          "systems: {f1, f2, ...} from {{f1}, {f2}, ...} or {f1, f2, ...}; a list for a single variable raises "
          "ValueError. Check: reference (Wolfram documentation forms).")
def _source_term(vars_: Expr, f: Expr, *rest: Assoc) -> Expr:
    fname = "SourcePDETerm"
    if len(rest) > 1:
        raise ValueError(f"{fname}: called with too many arguments")
    V, axi, subs = _term_setup(vars_, rest[0] if rest else None, fname)
    if V.system:
        m = len(V.deps)
        if not (is_list(f) and len(f.args) == m):
            raise ValueError(f"{fname}: give one source per equation ({m})")
        out = L(*((g.args[0] if is_list(g) and len(g.args) == 1 else g) for g in f.args))
        if any(is_list(g) for g in out.args):
            raise ValueError(f"{fname}: each source must be a scalar")
    else:
        if is_list(f):
            raise ValueError(f"{fname}: the source for one dependent variable must be a scalar")
        out = f
    return _apply_subs(ev(out), subs)
