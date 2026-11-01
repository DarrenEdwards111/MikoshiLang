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
    flat = []
    for x in a:
        flat.extend(x.args if _head_is(x, "Plus") else [x])
    a = [x for x in flat if not _is_zero(x)]
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


# ───────────────────────────── PDE components ─────────────────────────────

MU0 = Expr("Times", 4, Symbol("Pi"), Expr("Power", 10, -7))                     # pi / 2500000 (exact)
EPS0 = Expr("Power", Expr("Times", MU0, Expr("Power", 299792458, 2)), -1)       # 625000 / (22468879468420441 pi)
H_PLANCK = Expr("Times", 662607015, Expr("Power", 10, -42))                     # exact SI value


def _p(keys, k, default):
    return keys[k] if k in keys else default


def _scale(keys, V, fname):
    """Thickness (2-D) / CrossSectionalArea (1-D) factor."""
    d = 1
    if "Thickness" in keys:
        if V.n != 2:
            raise ValueError(f"{fname}: \"Thickness\" applies to 2-D models only")
        d = keys["Thickness"]
    if "CrossSectionalArea" in keys:
        if V.n != 1:
            raise ValueError(f"{fname}: \"CrossSectionalArea\" applies to 1-D models only")
        d = keys["CrossSectionalArea"]
    return d


def _scaled(d, c):
    """d * c for a scalar/vector/matrix coefficient (None = identity)."""
    if c is None:
        c = 1
    if isinstance(d, int) and d == 1:
        return c
    return ev(Expr("Times", d, c))


def _zero_vec(n):
    return L(*([0] * n))


def _material(keys, fname):
    m = keys.get("Material", Symbol("Automatic"))
    if not (isinstance(m, Symbol) and m.name in ("Automatic", "None")):
        raise NotImplementedError(f"{fname}: \"Material\" data is not supported; give the material constants")


def _model_form(keys, fname, default="NonConservative"):
    f = keys.get("ModelForm", default)
    if f not in ("NonConservative", "Conservative"):
        raise ValueError(f"{fname}: \"ModelForm\" must be \"NonConservative\" or \"Conservative\", not {f}")
    return f == "Conservative"


def _component_args(vars_, pars, fname, known, unsupported=(), allow_freq=False, require_time=False,
                    allow_system=False):
    V = parse_vars(vars_, fname, allow_freq=allow_freq, allow_system=allow_system, require_time=require_time)
    keys, subs = parse_pars(pars, fname, known=tuple(known) + tuple(unsupported), unsupported=unsupported)
    _chart(keys, fname)
    axi = _region_symmetry(keys, fname) if "RegionSymmetry" in keys else False
    if "Material" in keys:
        _material(keys, fname)
    return V, keys, subs, axi


def _finish(e, subs):
    return _apply_subs(ev(e), subs)


_HEAT_KEYS = ("HeatConvectionVelocity", "HeatSource", "MassDensity", "Material", "ModelForm", "RegionSymmetry",
              "SpecificHeatCapacity", "ThermalConductivity", "Thickness", "CrossSectionalArea", "CoordinateChart",
              # boundary-condition keys that may share the same association
              "AmbientTemperature", "HeatTransferCoefficient", "Emissivity", "StefanBoltzmannConstant",
              "HeatFlux", "Temperature", "SurfaceArea")


@fn("HeatTransferPDEComponent", area=AREA, status="added",
    notes="HeatTransferPDEComponent[vars, pars] -> rho Cp D[T, t] + Inactive[Div][-k . Inactive[Grad][T, X], X] "
          "+ rho Cp v . Inactive[Grad][T, X] - Q (non-conservative, the default) or with Inactive[Div][rho Cp v T, X] "
          "for \"ModelForm\" -> \"Conservative\"; parameters \"MassDensity\", \"SpecificHeatCapacity\", "
          "\"ThermalConductivity\" (scalar, vector or matrix; default IdentityMatrix), \"HeatConvectionVelocity\", "
          "\"HeatSource\", \"Thickness\" (2-D) / \"CrossSectionalArea\" (1-D) multiply every term, "
          "\"RegionSymmetry\" -> \"Axisymmetric\"; zero terms are omitted; symbol keys are substituted. "
          "\"Material\" entities raise NotImplementedError; unknown keys raise ValueError. Check: reference "
          "(Wolfram documentation forms) and Activate compared with the written-out heat equation in tests.")
def _heat_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "HeatTransferPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _HEAT_KEYS)
    d = _scale(keys, V, fname)
    rho, cp = _p(keys, "MassDensity", 1), _p(keys, "SpecificHeatCapacity", 1)
    k = keys.get("ThermalConductivity")
    v = _p(keys, "HeatConvectionVelocity", _zero_vec(V.n))
    Q = _p(keys, "HeatSource", 0)
    rc = ev(Expr("Times", d, rho, cp))
    parts = []
    if V.t is not None:
        parts.append(_times(rc, _time_d(V.u, V.t)))
    parts.append(_diffusion(V.u, V.XL, _scaled(d, k), fname, axi))
    vv = _vector(v, V.n, fname, "\"HeatConvectionVelocity\"")
    if _model_form(keys, fname):
        parts.append(_cconvection(V.u, V.XL, ev(Expr("Times", -1, rc, vv)), fname, axi))
    else:
        parts.append(_convection(V.u, V.XL, ev(Expr("Times", rc, vv)), fname))
    parts.append(_neg(_times(d, Q)))
    return _finish(_plus(*parts), subs)


_MASS_KEYS = ("MassConvectionVelocity", "DiffusionCoefficient", "MassReactionRate", "MassSource", "ModelForm",
              "RegionSymmetry", "CoordinateChart", "Thickness", "CrossSectionalArea",
              "AmbientConcentration", "MassTransferCoefficient", "MassFlux", "Concentration")


@fn("MassTransportPDEComponent", area=AREA, status="added",
    notes="MassTransportPDEComponent[vars, pars] -> D[c, t] + Inactive[Div][-d . Inactive[Grad][c, X], X] + "
          "v . Inactive[Grad][c, X] + r c - R (non-conservative default) or Inactive[Div][v c, X] for "
          "\"ModelForm\" -> \"Conservative\"; parameters \"DiffusionCoefficient\" (scalar, vector or matrix; default "
          "IdentityMatrix), \"MassConvectionVelocity\", \"MassReactionRate\", \"MassSource\", \"RegionSymmetry\"; zero "
          "terms omitted; symbol keys substituted; unknown keys raise ValueError. Check: reference (Wolfram "
          "documentation forms) and Activate compared with the written-out transport equation in tests.")
def _mass_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "MassTransportPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _MASS_KEYS)
    d = _scale(keys, V, fname)
    D = keys.get("DiffusionCoefficient")
    v = _vector(_p(keys, "MassConvectionVelocity", _zero_vec(V.n)), V.n, fname, "\"MassConvectionVelocity\"")
    r, R_ = _p(keys, "MassReactionRate", 0), _p(keys, "MassSource", 0)
    parts = []
    if V.t is not None:
        parts.append(_times(d, _time_d(V.u, V.t)))
    parts.append(_diffusion(V.u, V.XL, _scaled(d, D), fname, axi))
    if _model_form(keys, fname):
        parts.append(_cconvection(V.u, V.XL, ev(Expr("Times", -1, d, v)), fname, axi))
    else:
        parts.append(_convection(V.u, V.XL, ev(Expr("Times", d, v)), fname))
    parts.append(_times(d, r, V.u))
    parts.append(_neg(_times(d, R_)))
    return _finish(_plus(*parts), subs)


_ACOUSTIC_KEYS = ("DipoleSource", "MassDensity", "MonopoleSource", "SoundSpeed", "RegionSymmetry",
                  "CoordinateChart", "Thickness", "CrossSectionalArea",
                  "AcousticPressure", "NormalVelocity", "NormalAcceleration", "Impedance", "SpecificImpedance",
                  "InwardPressure", "IncomingPressure")


def _recip(x):
    return ev(Expr("Power", x, -1))


@fn("AcousticPDEComponent", area=AREA, status="added",
    notes="AcousticPDEComponent[vars, pars]: time domain {p[t, x...], t, {x...}} -> 1/(rho c^2) D[p, {t, 2}] + "
          "Inactive[Div][-(1/rho) Inactive[Grad][p, X], X] + Inactive[Div][-F/rho, X] - Q; frequency domain "
          "{p[x...], omega, {x...}} -> Inactive[Div][...] + Inactive[Div][-F/rho, X] - omega^2/(rho c^2) p - Q; "
          "stationary vars give the spatial part; parameters \"MassDensity\" rho, \"SoundSpeed\" c, "
          "\"MonopoleSource\" Q, \"DipoleSource\" F, \"RegionSymmetry\"; unknown keys raise ValueError. Check: "
          "reference (Wolfram documentation forms, e.g. rho = 1.2, c = 343 -> 1/(1.2 343^2) = 7.08322*10^-6) in tests.")
def _acoustic_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "AcousticPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _ACOUSTIC_KEYS, allow_freq=True)
    rho, c = _p(keys, "MassDensity", 1), _p(keys, "SoundSpeed", 1)
    Q = _p(keys, "MonopoleSource", 0)
    F = _vector(_p(keys, "DipoleSource", _zero_vec(V.n)), V.n, fname, "\"DipoleSource\"")
    irho = _recip(rho)
    parts = []
    m = ev(Expr("Power", Expr("Times", rho, Expr("Power", c, 2)), -1))
    if V.t is not None:
        parts.append(_times(m, _time_d(V.u, V.t, 2)))
    parts.append(_diffusion(V.u, V.XL, irho, fname, axi))
    parts.append(_derivterm(V.XL, ev(Expr("Times", -1, irho, F)), fname, axi))
    if V.w is not None:
        parts.append(_times(-1, Expr("Power", V.w, 2), m, V.u))
    parts.append(_neg(Q))
    return _finish(_plus(*parts), subs)


_ES_KEYS = ("Polarization", "RelativePermittivity", "RemanentPolarization", "Thickness", "CrossSectionalArea",
            "VacuumPermittivity", "VolumeChargeDensity", "RegionSymmetry", "CoordinateChart", "Material",
            "ElectricPotential", "SurfaceChargeDensity", "ElectricFluxDensity")


@fn("ElectrostaticPDEComponent", area=AREA, status="added",
    notes="ElectrostaticPDEComponent[vars, pars] (stationary vars) -> Inactive[Div][-(d eps0 epsr) . "
          "Inactive[Grad][V, X], X] + Inactive[Div][d (P + Pr), X] - d rhov with \"VacuumPermittivity\" eps0 "
          "(default exact 1/(mu0 c^2) = 625000/(22468879468420441 Pi)), \"RelativePermittivity\" epsr (scalar, "
          "vector or matrix), \"Polarization\" P, \"RemanentPolarization\" Pr, \"VolumeChargeDensity\" rhov, "
          "\"Thickness\" (2-D) / \"CrossSectionalArea\" (1-D) d, \"RegionSymmetry\"; time-dependent vars raise "
          "NotImplementedError; unknown keys raise ValueError. Check: reference (Wolfram documentation forms) in "
          "tests.")
def _electrostatic_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "ElectrostaticPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _ES_KEYS)
    if V.t is not None:
        raise NotImplementedError(f"{fname}: time-dependent electrostatic models are not supported")
    d = _scale(keys, V, fname)
    eps = ev(Expr("Times", d, _p(keys, "VacuumPermittivity", EPS0), _p(keys, "RelativePermittivity", 1)))
    P = _vector(_p(keys, "Polarization", _zero_vec(V.n)), V.n, fname, "\"Polarization\"")
    Pr = _vector(_p(keys, "RemanentPolarization", _zero_vec(V.n)), V.n, fname, "\"RemanentPolarization\"")
    parts = [_diffusion(V.u, V.XL, eps, fname, axi),
             _derivterm(V.XL, ev(Expr("Times", d, Expr("Plus", P, Pr))), fname, axi),
             _neg(_times(d, _p(keys, "VolumeChargeDensity", 0)))]
    return _finish(_plus(*parts), subs)


_EC_KEYS = ("CrossSectionalArea", "CurrentSource", "ElectricalConductivity", "ExternalCurrentSource", "Material",
            "Thickness", "Polarization", "RelativePermittivity", "VacuumPermittivity", "RegionSymmetry",
            "CoordinateChart", "ElectricPotential", "CurrentDensity", "ElectricCurrentDensity")


@fn("ElectricCurrentPDEComponent", area=AREA, status="added",
    notes="ElectricCurrentPDEComponent[vars, pars]: stationary -> Inactive[Div][-(d sigma) . Inactive[Grad][V, X], X] "
          "+ Inactive[Div][d Je, X] - d Qv; time-dependent {V[t, ...], t, X} adds Inactive[Div][-(d eps0 epsr) . "
          "Inactive[Grad][D[V, t], X], X]; frequency {V[...], omega, X} adds Inactive[Div][-(I d omega eps0 epsr) . "
          "Inactive[Grad][V, X], X] + Inactive[Div][I omega d P, X]; parameters \"ElectricalConductivity\" sigma, "
          "\"ExternalCurrentSource\" Je, \"CurrentSource\" Qv, \"RelativePermittivity\", \"VacuumPermittivity\", "
          "\"Polarization\", \"Thickness\" / \"CrossSectionalArea\" d; \"Material\" entities raise "
          "NotImplementedError. Check: reference (Wolfram documentation forms) in tests.")
def _electric_current_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "ElectricCurrentPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _EC_KEYS, allow_freq=True)
    d = _scale(keys, V, fname)
    sigma = _p(keys, "ElectricalConductivity", 1)
    Je = _vector(_p(keys, "ExternalCurrentSource", _zero_vec(V.n)), V.n, fname, "\"ExternalCurrentSource\"")
    eps = ev(Expr("Times", _p(keys, "VacuumPermittivity", EPS0), _p(keys, "RelativePermittivity", 1)))
    parts = [_diffusion(V.u, V.XL, _scaled(d, sigma), fname, axi)]
    if V.t is not None:
        parts.append(_diffusion(_time_d(V.u, V.t), V.XL, ev(Expr("Times", d, eps)), fname, axi))
    if V.w is not None:
        parts.append(_diffusion(V.u, V.XL, ev(Expr("Times", Symbol("I"), d, V.w, eps)), fname, axi))
        P = _vector(_p(keys, "Polarization", _zero_vec(V.n)), V.n, fname, "\"Polarization\"")
        parts.append(_derivterm(V.XL, ev(Expr("Times", Symbol("I"), V.w, d, P)), fname, axi))
    parts.append(_derivterm(V.XL, ev(Expr("Times", d, Je)), fname, axi))
    parts.append(_neg(_times(d, _p(keys, "CurrentSource", 0))))
    return _finish(_plus(*parts), subs)


_MS_KEYS = ("Magnetization", "RelativePermeability", "RecoilPermeability", "RemanentMagneticFluxDensity",
            "Thickness", "CrossSectionalArea", "VacuumPermeability", "RegionSymmetry", "CoordinateChart", "Material",
            "MagneticFluxDensity", "MagneticPotential")


@fn("MagnetostaticPDEComponent", area=AREA, status="added",
    notes="MagnetostaticPDEComponent[vars, pars] (stationary scalar potential Vm) -> Inactive[Div][-(d mu0 mur) . "
          "Inactive[Grad][Vm, X], X] + Inactive[Div][d mu0 M, X]; with \"RemanentMagneticFluxDensity\" Br -> "
          "Inactive[Div][-(d mu0 murec) . Inactive[Grad][Vm, X], X] + Inactive[Div][d Br, X] "
          "(\"RecoilPermeability\" murec, default 1); \"VacuumPermeability\" mu0 defaults to the exact 4 Pi 10^-7 = "
          "Pi/2500000; giving both M and Br raises ValueError; time-dependent vars raise NotImplementedError. "
          "Check: reference (Wolfram documentation forms) in tests.")
def _magnetostatic_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "MagnetostaticPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _MS_KEYS)
    if V.t is not None:
        raise NotImplementedError(f"{fname}: time-dependent models are not supported (use MagneticPDEComponent)")
    d = _scale(keys, V, fname)
    mu0 = _p(keys, "VacuumPermeability", MU0)
    if "Magnetization" in keys and "RemanentMagneticFluxDensity" in keys:
        raise ValueError(f"{fname}: give either \"Magnetization\" or \"RemanentMagneticFluxDensity\", not both")
    if "RemanentMagneticFluxDensity" in keys:
        Br = _vector(keys["RemanentMagneticFluxDensity"], V.n, fname, "\"RemanentMagneticFluxDensity\"")
        c = ev(Expr("Times", d, mu0, _p(keys, "RecoilPermeability", 1)))
        src = ev(Expr("Times", d, Br))
    else:
        M = _vector(_p(keys, "Magnetization", _zero_vec(V.n)), V.n, fname, "\"Magnetization\"")
        c = ev(Expr("Times", d, mu0, _p(keys, "RelativePermeability", 1)))
        src = ev(Expr("Times", d, mu0, M))
    return _finish(_plus(_diffusion(V.u, V.XL, c, fname, axi), _derivterm(V.XL, src, fname, axi)), subs)


_MAG_KEYS = ("ExternalCurrentSource", "Magnetization", "RelativePermeability", "RecoilPermeability",
             "RemanentMagneticFluxDensity", "VacuumPermeability", "ElectricalConductivity", "CoordinateChart",
             "Material", "MagneticFluxDensity", "MagneticPotential", "SurfaceCurrentDensity")


def _curlcurl_rows(As, X, nu):
    """Rows of Curl[nu Curl[A]] as Sum_j Inactive[Div][C_ij . Inactive[Grad][A_j, X], X] (Wolfram's form)."""
    n = 3
    rows = []
    for i in range(n):
        terms = []
        for m in range(n):
            if m == i:
                C = [[(0 if (a == b and a == i) else (_neg(nu) if a == b else 0)) for b in range(n)] for a in range(n)]
            else:
                C = [[(nu if (a == m and b == i) else 0) for b in range(n)] for a in range(n)]
            terms.append(_div(Expr("Dot", L(*(L(*r) for r in C)), _grad(As[m], X)), X))
        rows.append(terms)
    return rows


@fn("MagneticPDEComponent", area=AREA, status="added",
    notes="MagneticPDEComponent[vars, pars] for the magnetic vector potential {{Ax, Ay, Az}, {x, y, z}} -> rows of "
          "Curl[(mu0 mur)^-1 Curl[A] - M] - Je written, as Wolfram does, as Sum_j Inactive[Div][C_ij . "
          "Inactive[Grad][A_j, X], X]; frequency vars {A, omega, X} add I omega sigma A, time vars add sigma D[A, t]; "
          "\"RemanentMagneticFluxDensity\" Br uses (mu0 murec)^-1 (Curl[A] - Br); \"VacuumPermeability\" defaults to "
          "the exact Pi/2500000; scalar permeabilities only (matrices raise NotImplementedError), 3-D only (2-D "
          "forms raise NotImplementedError). Check: Activate compared with the curl-curl identity Grad[Div[A]] - "
          "Laplacian[A] (numeric cross-check of the coefficient form) in tests.")
def _magnetic_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "MagneticPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _MAG_KEYS, allow_freq=True, allow_system=True)
    if not V.system or len(V.deps) != 3 or V.n != 3:
        raise NotImplementedError(f"{fname}: only the 3-D vector potential {{{{Ax, Ay, Az}}, {{x, y, z}}}} is "
                                  "supported")
    mu0 = _p(keys, "VacuumPermeability", MU0)
    if "Magnetization" in keys and "RemanentMagneticFluxDensity" in keys:
        raise ValueError(f"{fname}: give either \"Magnetization\" or \"RemanentMagneticFluxDensity\", not both")
    for k in ("RelativePermeability", "RecoilPermeability"):
        if k in keys and is_list(keys[k]):
            raise NotImplementedError(f"{fname}: only scalar \"{k}\" values are supported")
    if "RemanentMagneticFluxDensity" in keys:
        mu = ev(Expr("Times", mu0, _p(keys, "RecoilPermeability", 1)))
        M = ev(Expr("Times", Expr("Power", mu, -1),
                    _vector(keys["RemanentMagneticFluxDensity"], 3, fname, "\"RemanentMagneticFluxDensity\"")))
    else:
        mu = ev(Expr("Times", mu0, _p(keys, "RelativePermeability", 1)))
        M = _vector(_p(keys, "Magnetization", _zero_vec(3)), 3, fname, "\"Magnetization\"")
    nu = _recip(mu)
    Je = _vector(_p(keys, "ExternalCurrentSource", _zero_vec(3)), 3, fname, "\"ExternalCurrentSource\"")
    sigma = _p(keys, "ElectricalConductivity", 1 if (V.t is not None or V.w is not None) else 0)
    rows = _curlcurl_rows(V.deps, V.XL, nu)
    eps = [[[0, 0, 0], [0, 0, 1], [0, -1, 0]], [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
           [[0, 1, 0], [-1, 0, 0], [0, 0, 0]]]       # eps[i][j][k]
    out = []
    for i in range(3):
        parts = list(rows[i])
        if not _is_zero(M):
            # -(Curl M)_i = Inactive[Div][{-eps_ijk M_k}_j, X]
            vec = ev(L(*(Expr("Times", -1, Expr("Plus", *[_times(eps[i][j][k], M.args[k]) for k in range(3)] + [0]))
                         for j in range(3))))
            if not _is_zero(vec):
                parts.append(_div(vec, V.XL))
        if V.t is not None:
            parts.append(_times(sigma, _time_d(V.deps[i], V.t)))
        if V.w is not None:
            parts.append(_times(Symbol("I"), V.w, sigma, V.deps[i]))
        parts.append(_neg(Je.args[i]))
        out.append(_plus(*parts))
    return _finish(L(*out), subs)


@fn("PoissonPDEComponent", area=AREA, status="added",
    notes="PoissonPDEComponent[vars, pars] -> Inactive[Div][IdentityMatrix[n] . Inactive[Grad][u, X], X] - f with "
          "\"PoissonSourceTerm\" f (default 1) and \"RegionSymmetry\"; time-dependent vars add no time term (as "
          "documented). Check: reference (Wolfram documentation: PoissonPDEComponent[{u[x], {x}}, <||>] -> "
          "-1 + Inactive[Div][{{1}} . Inactive[Grad][u[x], {x}], {x}]) in tests.")
def _poisson_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "PoissonPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, ("PoissonSourceTerm", "RegionSymmetry",
                                                              "CoordinateChart"))
    return _finish(_plus(_diffusion(V.u, V.XL, None, fname, axi, sign=1),
                         _neg(_p(keys, "PoissonSourceTerm", 1))), subs)


@fn("HelmholtzPDEComponent", area=AREA, status="added",
    notes="HelmholtzPDEComponent[vars, pars] -> Inactive[Div][IdentityMatrix[n] . Inactive[Grad][u, X], X] + k^2 u "
          "with \"HelmholtzEigenvalue\" k (default 1) and \"RegionSymmetry\". Check: reference (Wolfram "
          "documentation forms) and the eigenvalue relation (NDEigenvalues) in tests.")
def _helmholtz_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "HelmholtzPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, ("HelmholtzEigenvalue", "RegionSymmetry",
                                                              "CoordinateChart"))
    k = _p(keys, "HelmholtzEigenvalue", 1)
    return _finish(_plus(_diffusion(V.u, V.XL, None, fname, axi, sign=1),
                         _times(ev(Expr("Power", k, 2)), V.u)), subs)


@fn("WavePDEComponent", area=AREA, status="added",
    notes="WavePDEComponent[{u[t, x...], t, {x...}}, pars] -> D[u, {t, 2}] + Inactive[Div][-c^2 . Inactive[Grad][u, "
          "X], X] with \"WaveCoefficient\" c (scalar, or an n x n matrix whose square c.c is used; default 1) and "
          "\"RegionSymmetry\"; stationary vars raise ValueError. Check: reference (Wolfram documentation forms) in "
          "tests.")
def _wave_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "WavePDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, ("WaveCoefficient", "RegionSymmetry",
                                                              "CoordinateChart"), require_time=True)
    c = _p(keys, "WaveCoefficient", 1)
    if _is_matrix(c):
        c2 = ev(Expr("Dot", c, c))
    elif is_list(c):
        raise ValueError(f"{fname}: \"WaveCoefficient\" must be a scalar or a square matrix")
    else:
        c2 = ev(Expr("Power", c, 2))
    return _finish(_plus(_time_d(V.u, V.t, 2), _diffusion(V.u, V.XL, c2, fname, axi)), subs)


_SCH_KEYS = ("AzimuthalQuantumNumber", "MagneticVectorPotential", "Mass", "PlanckConstant", "ReducedPlanckConstant",
             "SchrodingerPotential", "RegionSymmetry", "ParticleCharge", "CoordinateChart")


@fn("SchrodingerPDEComponent", area=AREA, status="added",
    notes="SchrodingerPDEComponent[vars, pars] -> -I hbar D[Psi, t] (time-dependent vars) + Inactive[Div][-(hbar^2/2) "
          "(1/m) . Inactive[Grad][Psi, X], X] + V Psi with \"Mass\" m (scalar, or a matrix used elementwise as "
          "1/m_ij), \"SchrodingerPotential\" V, \"ReducedPlanckConstant\" hbar or \"PlanckConstant\" h (hbar = "
          "h/(2 Pi)); the default hbar is exact (h = 662607015*10^-42); \"MagneticVectorPotential\" and "
          "\"AzimuthalQuantumNumber\" settings other than the defaults raise NotImplementedError. Check: reference "
          "(Wolfram documentation forms, hbar^2/2 = 662607015^2/(8 Pi^2 10^84)) in tests.")
def _schrodinger_component(vars_: Expr, pars: Assoc) -> Expr:
    fname = "SchrodingerPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _SCH_KEYS)
    if not _is_zero(keys.get("MagneticVectorPotential", 0)):
        raise NotImplementedError(f"{fname}: \"MagneticVectorPotential\" is not supported")
    if _sym_of(keys.get("AzimuthalQuantumNumber", Symbol("None"))) != "None":
        raise NotImplementedError(f"{fname}: \"AzimuthalQuantumNumber\" is not supported")
    if "ReducedPlanckConstant" in keys:
        hbar = keys["ReducedPlanckConstant"]
    else:
        hbar = ev(Expr("Times", _p(keys, "PlanckConstant", H_PLANCK), Expr("Power", Expr("Times", 2, Symbol("Pi")),
                                                                             -1)))
    m = _p(keys, "Mass", 1)
    if is_list(m):
        if not _is_matrix(m, V.n):
            raise ValueError(f"{fname}: a \"Mass\" matrix must be {V.n} x {V.n}")
        inv = L(*(L(*(Expr("Power", e, -1) for e in r.args)) for r in m.args))
    else:
        inv = Expr("Power", m, -1)
    c = ev(Expr("Times", Expr("Power", hbar, 2), Expr("Rational", 1, 2), inv))
    parts = []
    if V.t is not None:
        parts.append(_times(-1, Symbol("I"), hbar, _time_d(V.u, V.t)))
    parts.append(_diffusion(V.u, V.XL, c, fname, axi))
    parts.append(_times(_p(keys, "SchrodingerPotential", 0), V.u))
    return _finish(_plus(*parts), subs)


# ───────────────────────────── boundary values and conditions ─────────────────────────────

def _bc_args(pred, vars_, pars, rest, fname, known, allow_freq=False, allow_system=False):
    """pred, vars, pars[, lkey or <|...|>] -> (V, keys, subs); localized values override the shared ones."""
    if len(rest) > 1:
        raise ValueError(f"{fname}: called with too many arguments")
    if isinstance(pred, str):
        raise TypeError(f"{fname}: the boundary predicate must be a predicate, not a string")
    V = parse_vars(vars_, fname, allow_freq=allow_freq, allow_system=allow_system)
    if not KL.is_assoc(pars):
        if isinstance(pars, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the parameters must be an association")
    local = None
    if rest:
        lk = rest[0]
        if KL.is_assoc(lk):
            local = lk
        else:
            hit = [r for r in pars.args if K.same(r.args[0], lk)]
            if not hit:
                raise ValueError(f"{fname}: the localization key {lk} is not in the parameters")
            local = ev(hit[0].args[1])
            if not KL.is_assoc(local):
                raise ValueError(f"{fname}: the localized parameters pars[{lk}] must be an association")
    rules = [r for r in pars.args if not (rest and not KL.is_assoc(rest[0]) and K.same(r.args[0], rest[0]))]
    rules = [r for r in rules if not KL.is_assoc(r.args[1]) or not isinstance(r.args[0], str) or
             r.args[0] in known]
    merged = Expr("Association", *rules)
    keys, subs = parse_pars(merged, fname, known=known)
    if local is not None:
        k2, s2 = parse_pars(local, fname, known=known)
        keys.update(k2)
        subs = [r for r in subs if not any(K.same(r.args[0], q.args[0]) for q in s2)] + s2
    return V, keys, subs


def _normal(V, keys):
    n = keys.get("BoundaryUnitNormal", Symbol("Automatic"))
    if isinstance(n, Symbol) and n.name == "Automatic":
        return Expr("NeumannBoundaryUnitNormal", V.XL)
    return _vector(n, V.n, "BoundaryUnitNormal", "\"BoundaryUnitNormal\"")


def _ndot(V, keys, vec):
    if _is_zero(vec):
        return 0
    return Expr("Dot", _normal(V, keys), vec)


def _neumann(val, pred, subs):
    return _apply_subs(ev(Expr("NeumannValue", ev(val), pred)), subs)


def _dirichlet(lhs, rhs, pred, subs):
    if is_list(lhs):
        eqs = L(*(Expr("Equal", a, b) for a, b in zip(lhs.args, rhs.args)))
    else:
        eqs = Expr("Equal", lhs, rhs)
    return _apply_subs(ev(Expr("DirichletCondition", eqs, pred)), subs)


def _no_scale(keys, fname):
    for k in ("Thickness", "CrossSectionalArea"):
        if k in keys and not (is_numeric_value(keys[k]) and S(keys[k]) == 1):
            raise NotImplementedError(f"{fname}: \"{k}\" scaling of boundary values is not supported")


def _bc_doc(name, what, extra=""):
    return (f"{name}[pred, vars, pars] and {name}[pred, vars, pars, lkey] (lkey: a key of pars holding a localized "
            f"association, or an association) -> {what}; pars as for the PDE component; symbol keys are "
            f"substituted; unknown keys raise ValueError; non-association pars raise TypeError. {extra}"
            "Check: reference (Wolfram documentation forms) in tests.")


# heat transfer

def _heat_flux_core(V, keys, q, fname):
    rc = ev(Expr("Times", _p(keys, "MassDensity", 1), _p(keys, "SpecificHeatCapacity", 1)))
    v = _vector(_p(keys, "HeatConvectionVelocity", _zero_vec(V.n)), V.n, fname, "\"HeatConvectionVelocity\"")
    if _model_form(keys, fname):
        return q
    return _plus(q, _times(rc, V.u, _ndot(V, keys, v)))


_HEAT_BC_KEYS = _HEAT_KEYS + ("BoundaryUnitNormal", "SurfaceTemperature", "ReferenceTemperature",
                              "BoltzmannConstant")


@fn("HeatFluxValue", area=AREA, status="added",
    notes=_bc_doc("HeatFluxValue", "NeumannValue[q + rho Cp T NeumannBoundaryUnitNormal[X] . v, pred] "
                  "(non-conservative; \"HeatFlux\" q, default 0; the convective part only for a nonzero "
                  "\"HeatConvectionVelocity\") or NeumannValue[q, pred] (\"ModelForm\" -> \"Conservative\")",
                  "\"Thickness\"/\"CrossSectionalArea\" settings raise NotImplementedError. "))
def _heat_flux_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "HeatFluxValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _HEAT_BC_KEYS)
    _no_scale(keys, fname)
    return _neumann(_heat_flux_core(V, keys, _p(keys, "HeatFlux", 0), fname), pred, subs)


@fn("HeatInsulationValue", area=AREA, status="added",
    notes=_bc_doc("HeatInsulationValue", "HeatFluxValue with zero heat flux: NeumannValue[rho Cp T n . v, pred] "
                  "(non-conservative) or NeumannValue[0, pred]"))
def _heat_insulation_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "HeatInsulationValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _HEAT_BC_KEYS)
    _no_scale(keys, fname)
    return _neumann(_heat_flux_core(V, keys, 0, fname), pred, subs)


@fn("HeatOutflowValue", area=AREA, status="added",
    notes=_bc_doc("HeatOutflowValue", "NeumannValue[0, pred] (non-conservative: no conductive flux) or "
                  "NeumannValue[-rho Cp T n . v, pred] (\"ModelForm\" -> \"Conservative\")"))
def _heat_outflow_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "HeatOutflowValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _HEAT_BC_KEYS)
    _no_scale(keys, fname)
    if _model_form(keys, fname):
        rc = ev(Expr("Times", _p(keys, "MassDensity", 1), _p(keys, "SpecificHeatCapacity", 1)))
        v = _vector(_p(keys, "HeatConvectionVelocity", _zero_vec(V.n)), V.n, fname, "\"HeatConvectionVelocity\"")
        return _neumann(_neg(_times(rc, V.u, _ndot(V, keys, v))), pred, subs)
    return _neumann(0, pred, subs)


@fn("HeatSymmetryValue", area=AREA, status="added",
    notes=_bc_doc("HeatSymmetryValue", "NeumannValue[0, pred] (mirror symmetry: no flux across the boundary)"))
def _heat_symmetry_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _bc_args(pred, vars_, pars, rest, "HeatSymmetryValue", _HEAT_BC_KEYS)
    return _neumann(0, pred, subs)


# Stefan-Boltzmann constant from the exact SI k, h, c: 2 Pi^5 k^4 / (15 h^3 c^2)
_KB = Expr("Times", 1380649, Expr("Power", 10, -29))
SIGMA_SB = Expr("Times", 2, Expr("Power", Symbol("Pi"), 5), Expr("Power", _KB, 4),
                Expr("Power", Expr("Times", 15, Expr("Power", H_PLANCK, 3), Expr("Power", 299792458, 2)), -1))


@fn("HeatRadiationValue", area=AREA, status="added",
    notes=_bc_doc("HeatRadiationValue", "NeumannValue[eps sigma ((Tamb - Tref)^4 - (T - Tref)^4), pred] with "
                  "\"Emissivity\" eps (default 1), \"AmbientTemperature\" Tamb, \"ReferenceTemperature\" Tref (default "
                  "0) and \"BoltzmannConstant\" sigma (default the exact Stefan-Boltzmann constant 2 Pi^5 k^4/(15 h^3 "
                  "c^2) from the SI-defined k, h, c)"))
def _heat_radiation_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "HeatRadiationValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _HEAT_BC_KEYS)
    _no_scale(keys, fname)
    eps, sig = _p(keys, "Emissivity", 1), _p(keys, "BoltzmannConstant", SIGMA_SB)
    ta, tr = _p(keys, "AmbientTemperature", 0), _p(keys, "ReferenceTemperature", 0)
    val = Expr("Times", eps, sig, Expr("Plus", Expr("Power", Expr("Plus", ta, _neg(tr)), 4),
                                       _neg(Expr("Power", Expr("Plus", V.u, _neg(tr)), 4))))
    return _neumann(val, pred, subs)


@fn("HeatTransferValue", area=AREA, status="added",
    notes=_bc_doc("HeatTransferValue", "NeumannValue[h (Text - T), pred] with \"HeatTransferCoefficient\" h "
                  "(required) and \"AmbientTemperature\" Text (default 0)",
                  "A missing \"HeatTransferCoefficient\" raises ValueError. "))
def _heat_transfer_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "HeatTransferValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _HEAT_BC_KEYS)
    _no_scale(keys, fname)
    if "HeatTransferCoefficient" not in keys:
        raise ValueError(f"{fname}: the parameter \"HeatTransferCoefficient\" is required")
    h, ta = keys["HeatTransferCoefficient"], _p(keys, "AmbientTemperature", 0)
    return _neumann(Expr("Times", h, Expr("Plus", ta, _neg(V.u))), pred, subs)


@fn("HeatTemperatureCondition", area=AREA, status="added",
    notes=_bc_doc("HeatTemperatureCondition", "DirichletCondition[T == Ts, pred] with \"SurfaceTemperature\" Ts "
                  "(default 0)"))
def _heat_temperature_condition(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _bc_args(pred, vars_, pars, rest, "HeatTemperatureCondition", _HEAT_BC_KEYS)
    return _dirichlet(V.u, _p(keys, "SurfaceTemperature", 0), pred, subs)


# mass transport

_MASS_BC_KEYS = _MASS_KEYS + ("BoundaryUnitNormal", "MassConcentration")


def _mass_v(V, keys, fname):
    return _vector(_p(keys, "MassConvectionVelocity", _zero_vec(V.n)), V.n, fname, "\"MassConvectionVelocity\"")


@fn("MassFluxValue", area=AREA, status="added",
    notes=_bc_doc("MassFluxValue", "NeumannValue[q + c NeumannBoundaryUnitNormal[X] . v, pred] (non-conservative; "
                  "\"MassFlux\" q) or NeumannValue[q, pred] (\"ModelForm\" -> \"Conservative\")"))
def _mass_flux_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "MassFluxValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _MASS_BC_KEYS)
    q = _p(keys, "MassFlux", 0)
    if _model_form(keys, fname):
        return _neumann(q, pred, subs)
    return _neumann(_plus(_times(V.u, _ndot(V, keys, _mass_v(V, keys, fname))), q), pred, subs)


@fn("MassImpermeableBoundaryValue", area=AREA, status="added",
    notes=_bc_doc("MassImpermeableBoundaryValue", "MassFluxValue with zero flux: NeumannValue[c n . v, pred] "
                  "(non-conservative) or NeumannValue[0, pred] (conservative)"))
def _mass_impermeable_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "MassImpermeableBoundaryValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _MASS_BC_KEYS)
    if _model_form(keys, fname):
        return _neumann(0, pred, subs)
    return _neumann(_times(V.u, _ndot(V, keys, _mass_v(V, keys, fname))), pred, subs)


@fn("MassOutflowValue", area=AREA, status="added",
    notes=_bc_doc("MassOutflowValue", "NeumannValue[0, pred] (non-conservative: no diffusive flux) or "
                  "NeumannValue[-c n . v, pred] (\"ModelForm\" -> \"Conservative\")"))
def _mass_outflow_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "MassOutflowValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _MASS_BC_KEYS)
    if _model_form(keys, fname):
        return _neumann(_neg(_times(V.u, _ndot(V, keys, _mass_v(V, keys, fname)))), pred, subs)
    return _neumann(0, pred, subs)


@fn("MassSymmetryValue", area=AREA, status="added",
    notes=_bc_doc("MassSymmetryValue", "NeumannValue[0, pred] (mirror symmetry)"))
def _mass_symmetry_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _bc_args(pred, vars_, pars, rest, "MassSymmetryValue", _MASS_BC_KEYS)
    return _neumann(0, pred, subs)


@fn("MassTransferValue", area=AREA, status="added",
    notes=_bc_doc("MassTransferValue", "NeumannValue[k (cext - c), pred] with \"MassTransferCoefficient\" k "
                  "(default 1) and \"AmbientConcentration\" cext (default 0)"))
def _mass_transfer_value(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _bc_args(pred, vars_, pars, rest, "MassTransferValue", _MASS_BC_KEYS)
    k, ca = _p(keys, "MassTransferCoefficient", 1), _p(keys, "AmbientConcentration", 0)
    return _neumann(Expr("Times", k, Expr("Plus", ca, _neg(V.u))), pred, subs)


@fn("MassConcentrationCondition", area=AREA, status="added",
    notes=_bc_doc("MassConcentrationCondition", "DirichletCondition[c == cs, pred] with \"MassConcentration\" cs "
                  "(default 0)"))
def _mass_concentration_condition(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _bc_args(pred, vars_, pars, rest, "MassConcentrationCondition", _MASS_BC_KEYS)
    return _dirichlet(V.u, _p(keys, "MassConcentration", 0), pred, subs)


# acoustics

_AC_BC_KEYS = _ACOUSTIC_KEYS + ("BoundaryUnitNormal", "SpecificAcousticImpedance", "AcousticSoundParticleVelocity",
                                "AcousticSourceDistance", "SoundDirectionVector", "SoundIncidentPressure", "Material")


def _ac_args(pred, vars_, pars, rest, fname):
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _AC_BC_KEYS, allow_freq=True)
    if "Material" in keys:
        _material(keys, fname)
    return V, keys, subs


def _ac_rate(V, x, fname):
    """d/dt x in the time domain, I omega x in the frequency domain."""
    if V.t is not None:
        return Expr("D", x, V.t)
    if V.w is not None:
        return Expr("Times", Symbol("I"), V.w, x)
    raise ValueError(f"{fname}: time-dependent {{p[t, x...], t, {{x...}}}} or frequency {{p[x...], omega, "
                     "{x...}} variables are required")


@fn("AcousticSoundHardValue", area=AREA, status="added",
    notes=_bc_doc("AcousticSoundHardValue", "NeumannValue[0, pred] (zero normal acceleration)"))
def _ac_hard(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _ac_args(pred, vars_, pars, rest, "AcousticSoundHardValue")
    return _neumann(0, pred, subs)


@fn("AcousticSoundSoftCondition", area=AREA, status="added",
    notes=_bc_doc("AcousticSoundSoftCondition", "DirichletCondition[p == 0, pred]"))
def _ac_soft(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _ac_args(pred, vars_, pars, rest, "AcousticSoundSoftCondition")
    return _dirichlet(V.u, 0, pred, subs)


@fn("AcousticPressureCondition", area=AREA, status="added",
    notes=_bc_doc("AcousticPressureCondition", "DirichletCondition[p == p0, pred] with \"AcousticPressure\" p0 "
                  "(default 0)"))
def _ac_pressure(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _ac_args(pred, vars_, pars, rest, "AcousticPressureCondition")
    return _dirichlet(V.u, _p(keys, "AcousticPressure", 0), pred, subs)


@fn("AcousticNormalVelocityValue", area=AREA, status="added",
    notes=_bc_doc("AcousticNormalVelocityValue", "NeumannValer[D[vn, t], pred] in the time domain and "
                  "NeumannValue[I omega vn, pred] in the frequency domain (\"AcousticSoundParticleVelocity\" vn)",
                  "Stationary vars raise ValueError. ").replace("NeumannValer", "NeumannValue"))
def _ac_velocity(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "AcousticNormalVelocityValue"
    V, keys, subs = _ac_args(pred, vars_, pars, rest, fname)
    return _neumann(_ac_rate(V, _p(keys, "AcousticSoundParticleVelocity", 0), fname), pred, subs)


@fn("AcousticImpedanceValue", area=AREA, status="added",
    notes=_bc_doc("AcousticImpedanceValue", "NeumannValue[-(1/z) D[p, t], pred] (time domain) or "
                  "NeumannValue[-(I omega/z) p, pred] (frequency domain) with \"SpecificAcousticImpedance\" z "
                  "(default Infinity: NeumannValue[0, pred])", "Stationary vars raise ValueError. "))
def _ac_impedance(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "AcousticImpedanceValue"
    V, keys, subs = _ac_args(pred, vars_, pars, rest, fname)
    z = _p(keys, "SpecificAcousticImpedance", Symbol("Infinity"))
    if isinstance(z, Symbol) and z.name == "Infinity":
        return _neumann(0, pred, subs)
    return _neumann(Expr("Times", -1, Expr("Power", z, -1), _ac_rate(V, V.u, fname)), pred, subs)


@fn("AcousticAbsorbingValue", area=AREA, status="added",
    notes=_bc_doc("AcousticAbsorbingValue", "NeumannValue[-(1/(rho c)) D[p, t] - (d/rho) p, pred] (time domain) or "
                  "NeumannValue[-(I omega/(rho c)) p - (d/rho) p, pred] (frequency domain) with \"MassDensity\" rho, "
                  "\"SoundSpeed\" c and \"AcousticSourceDistance\" d (the inverse source distance term, default 0)",
                  "Stationary vars raise ValueError. "))
def _ac_absorbing(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "AcousticAbsorbingValue"
    V, keys, subs = _ac_args(pred, vars_, pars, rest, fname)
    rho, c, d = _p(keys, "MassDensity", 1), _p(keys, "SoundSpeed", 1), _p(keys, "AcousticSourceDistance", 0)
    val = _plus(Expr("Times", -1, Expr("Power", Expr("Times", rho, c), -1), _ac_rate(V, V.u, fname)),
                _times(-1, d, Expr("Power", rho, -1), V.u))
    return _neumann(val, pred, subs)


@fn("AcousticRadiationValue", area=AREA, status="added",
    notes=_bc_doc("AcousticRadiationValue", "NeumannValue[(1/(rho c)) ((1 + n . e/Sqrt[Total[e^2]]) D[pinc, t] - "
                  "D[p, t]), pred] (time domain; I omega x for D[x, t] in the frequency domain) with "
                  "\"SoundIncidentPressure\" pinc (default 0) and \"SoundDirectionVector\" e (required when pinc is "
                  "nonzero)", "Stationary vars raise ValueError. "))
def _ac_radiation(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "AcousticRadiationValue"
    V, keys, subs = _ac_args(pred, vars_, pars, rest, fname)
    rho, c = _p(keys, "MassDensity", 1), _p(keys, "SoundSpeed", 1)
    pinc = _p(keys, "SoundIncidentPressure", 0)
    inc = 0
    if not _is_zero(pinc):
        if "SoundDirectionVector" not in keys:
            raise ValueError(f"{fname}: \"SoundDirectionVector\" is required for a nonzero incident pressure")
        e = keys["SoundDirectionVector"]
        ehat = Expr("Times", e, Expr("Power", Expr("Sqrt", Expr("Total", Expr("Power", e, 2))), -1))
        inc = Expr("Times", _ac_rate(V, pinc, fname), Expr("Plus", 1, Expr("Dot", _normal(V, keys), ehat)))
    val = Expr("Times", Expr("Power", Expr("Times", rho, c), -1), _plus(inc, _neg(_ac_rate(V, V.u, fname))))
    return _neumann(val, pred, subs)


# electrostatics and electric currents

_E_BC_KEYS = tuple(set(_ES_KEYS + _EC_KEYS)) + ("BoundaryUnitNormal", "NormalCurrentDensity",
                                                "SurfaceChargeDensity", "ElectricFluxDensity", "Current")


@fn("ElectricPotentialCondition", area=AREA, status="added",
    notes=_bc_doc("ElectricPotentialCondition", "DirichletCondition[V == Vs, pred] with \"ElectricPotential\" Vs "
                  "(default 0)"))
def _e_potential(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _bc_args(pred, vars_, pars, rest, "ElectricPotentialCondition", _E_BC_KEYS, allow_freq=True)
    return _dirichlet(V.u, _p(keys, "ElectricPotential", 0), pred, subs)


@fn("ElectricSymmetryValue", area=AREA, status="added",
    notes=_bc_doc("ElectricSymmetryValue", "NeumannValue[0, pred] (mirror symmetry: zero normal flux)"))
def _e_symmetry(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _bc_args(pred, vars_, pars, rest, "ElectricSymmetryValue", _E_BC_KEYS, allow_freq=True)
    return _neumann(0, pred, subs)


@fn("ElectricFluxDensityValue", area=AREA, status="added",
    notes=_bc_doc("ElectricFluxDensityValue", "NeumannValue[-NeumannBoundaryUnitNormal[X] . D0, pred] for "
                  "\"ElectricFluxDensity\" D0, or NeumannValue[rhos, pred] for \"SurfaceChargeDensity\" rhos "
                  "(default NeumannValue[0, pred])", "Giving both raises ValueError. "))
def _e_flux(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "ElectricFluxDensityValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _E_BC_KEYS)
    if "ElectricFluxDensity" in keys and "SurfaceChargeDensity" in keys:
        raise ValueError(f"{fname}: give either \"ElectricFluxDensity\" or \"SurfaceChargeDensity\"")
    if "SurfaceChargeDensity" in keys:
        return _neumann(keys["SurfaceChargeDensity"], pred, subs)
    D0 = _vector(_p(keys, "ElectricFluxDensity", _zero_vec(V.n)), V.n, fname, "\"ElectricFluxDensity\"")
    return _neumann(_neg(_ndot(V, keys, D0)), pred, subs)


@fn("ElectricCurrentDensityValue", area=AREA, status="added",
    notes=_bc_doc("ElectricCurrentDensityValue", "NeumannValue[Jn, pred] for \"NormalCurrentDensity\" Jn or "
                  "NeumannValue[-NeumannBoundaryUnitNormal[X] . J0, pred] for \"CurrentDensity\" J0 (default 0)",
                  "\"Current\" (needs the boundary area) raises NotImplementedError; giving two of them raises "
                  "ValueError. "))
def _e_current(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "ElectricCurrentDensityValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _E_BC_KEYS, allow_freq=True)
    if "Current" in keys:
        raise NotImplementedError(f"{fname}: \"Current\" (total current over the boundary) is not supported; give "
                                  "\"NormalCurrentDensity\"")
    if "NormalCurrentDensity" in keys and "CurrentDensity" in keys:
        raise ValueError(f"{fname}: give either \"NormalCurrentDensity\" or \"CurrentDensity\"")
    if "NormalCurrentDensity" in keys:
        return _neumann(keys["NormalCurrentDensity"], pred, subs)
    J0 = _vector(_p(keys, "CurrentDensity", _zero_vec(V.n)), V.n, fname, "\"CurrentDensity\"")
    return _neumann(_neg(_ndot(V, keys, J0)), pred, subs)


# magnetics

_M_BC_KEYS = tuple(set(_MS_KEYS + _MAG_KEYS)) + ("BoundaryUnitNormal", "NormalMagneticFluxDensity")


@fn("MagneticFluxDensityValue", area=AREA, status="added",
    notes=_bc_doc("MagneticFluxDensityValue", "NeumannValue[-NeumannBoundaryUnitNormal[X] . B0, pred] for "
                  "\"MagneticFluxDensity\" B0 or NeumannValue[Bn, pred] for \"NormalMagneticFluxDensity\" Bn (scalar "
                  "potential models)", "Vector-potential vars raise NotImplementedError; giving both raises "
                  "ValueError. "))
def _m_flux(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "MagneticFluxDensityValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _M_BC_KEYS)
    if "MagneticFluxDensity" in keys and "NormalMagneticFluxDensity" in keys:
        raise ValueError(f"{fname}: give either \"MagneticFluxDensity\" or \"NormalMagneticFluxDensity\"")
    if "NormalMagneticFluxDensity" in keys:
        return _neumann(keys["NormalMagneticFluxDensity"], pred, subs)
    B0 = _vector(_p(keys, "MagneticFluxDensity", _zero_vec(V.n)), V.n, fname, "\"MagneticFluxDensity\"")
    return _neumann(_neg(_ndot(V, keys, B0)), pred, subs)


@fn("MagneticSymmetryValue", area=AREA, status="added",
    notes=_bc_doc("MagneticSymmetryValue", "NeumannValue[0, pred] (scalar potential models)",
                  "Vector-potential vars raise NotImplementedError. "))
def _m_symmetry(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _bc_args(pred, vars_, pars, rest, "MagneticSymmetryValue", _M_BC_KEYS)
    return _neumann(0, pred, subs)


@fn("MagneticPotentialCondition", area=AREA, status="added",
    notes=_bc_doc("MagneticPotentialCondition", "DirichletCondition[Vm == V0, pred] for a scalar potential or "
                  "DirichletCondition[{Ax == A0x, Ay == A0y, Az == A0z}, pred] for a vector potential "
                  "(\"MagneticPotential\", default 0 / {0, 0, 0})"))
def _m_potential(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "MagneticPotentialCondition"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _M_BC_KEYS, allow_freq=True, allow_system=True)
    if V.system:
        m = len(V.deps)
        val = _p(keys, "MagneticPotential", _zero_vec(m))
        if not is_list(val) or len(val.args) != m:
            raise ValueError(f"{fname}: \"MagneticPotential\" must be a vector of length {m}")
        return _dirichlet(L(*V.deps), val, pred, subs)
    return _dirichlet(V.u, _p(keys, "MagneticPotential", 0), pred, subs)
