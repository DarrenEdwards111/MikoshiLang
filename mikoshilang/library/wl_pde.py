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
from .wl_common import ev, split_options

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
              "AmbientTemperature", "HeatTransferCoefficient", "Emissivity",
              "HeatFlux")


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
              "AmbientConcentration", "MassTransferCoefficient", "MassFlux")


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
                  "AcousticPressure")


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
            "CoordinateChart", "ElectricPotential", "CurrentDensity")


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


_MS_KEYS = ("Magnetization", "RelativePermeability", "RemanentMagneticFluxDensity",
            "Thickness", "CrossSectionalArea", "VacuumPermeability", "RegionSymmetry", "CoordinateChart", "Material",
            "MagneticFluxDensity", "MagneticPotential")


@fn("MagnetostaticPDEComponent", area=AREA, status="added",
    notes="MagnetostaticPDEComponent[vars, pars] (stationary scalar potential Vm) -> Inactive[Div][-(d mu0 mur) . "
          "Inactive[Grad][Vm, X], X] + Inactive[Div][d mu0 M, X]; with \"RemanentMagneticFluxDensity\" Br -> "
          "Inactive[Div][-(d mu0 murec) . Inactive[Grad][Vm, X], X] + Inactive[Div][d Br, X] "
          "(\"RelativePermeability\" is then the recoil permeability murec, as documented); \"VacuumPermeability\" mu0 defaults to the exact 4 Pi 10^-7 = "
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
        c = ev(Expr("Times", d, mu0, _p(keys, "RelativePermeability", 1)))    # recoil permeability
        src = ev(Expr("Times", d, Br))
    else:
        M = _vector(_p(keys, "Magnetization", _zero_vec(V.n)), V.n, fname, "\"Magnetization\"")
        c = ev(Expr("Times", d, mu0, _p(keys, "RelativePermeability", 1)))
        src = ev(Expr("Times", d, mu0, M))
    return _finish(_plus(_diffusion(V.u, V.XL, c, fname, axi), _derivterm(V.XL, src, fname, axi)), subs)


_MAG_KEYS = ("ExternalCurrentSource", "Magnetization", "RelativePermeability",
             "RemanentMagneticFluxDensity", "VacuumPermeability", "ElectricalConductivity", "CoordinateChart",
             "Material", "MagneticFluxDensity", "MagneticPotential")


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
    if "RelativePermeability" in keys and is_list(keys["RelativePermeability"]):
        raise NotImplementedError(f"{fname}: only scalar \"RelativePermeability\" values are supported")
    if "RemanentMagneticFluxDensity" in keys:
        mu = ev(Expr("Times", mu0, _p(keys, "RelativePermeability", 1)))      # recoil permeability
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


# ───────────────────────────── solid mechanics ─────────────────────────────

_MODULI = ("YoungModulus", "PoissonRatio", "LameParameter", "ShearModulus", "BulkModulus", "PWaveModulus")
_SOLID_KEYS = _MODULI + ("MassDensity", "BodyLoad", "BodyLoadValue", "AnalysisType", "MaterialSymmetry",
                         "SolidMechanicsMaterialModel", "SolidMechanicsModelForm", "RegionSymmetry", "Material",
                         "CoordinateChart", "Thickness", "EngineeringStrain", "OutputStressMeasure",
                         "Displacement", "Pressure", "SpringConstant", "DampingConstant", "BoundaryUnitNormal",
                         "Force", "NormalForce", "NormalPressure", "InitialStress", "InitialStrain")


def _lame(keys, fname):
    """(lambda, mu) as SymPy expressions from any two isotropic moduli."""
    given = [k for k in _MODULI if k in keys]
    if len(given) != 2:
        raise ValueError(f"{fname}: exactly two of the elastic moduli {', '.join(_MODULI)} are required "
                         f"(got {len(given)})")
    lam, mu = sp.symbols("lam_ mu_")
    rel = {"YoungModulus": mu * (3 * lam + 2 * mu) / (lam + mu), "PoissonRatio": lam / (2 * (lam + mu)),
           "LameParameter": lam, "ShearModulus": mu, "BulkModulus": lam + 2 * mu / 3, "PWaveModulus": lam + 2 * mu}
    vals = {k: S(keys[k]) for k in given}
    if set(given) == {"YoungModulus", "PoissonRatio"}:
        E_, nu = vals["YoungModulus"], vals["PoissonRatio"]
        return E_ * nu / ((1 + nu) * (1 - 2 * nu)), E_ / (2 * (1 + nu))
    g1, g2 = sp.symbols("g1_ g2_")
    sols = sp.solve([rel[given[0]] - g1, rel[given[1]] - g2], [lam, mu], dict=True)
    if len(sols) != 1:
        raise NotImplementedError(f"{fname}: the moduli pair {given[0]}, {given[1]} does not determine the Lame "
                                  "parameters uniquely; give YoungModulus and PoissonRatio")
    s = sols[0]
    return (sp.simplify(s[lam].subs({g1: vals[given[0]], g2: vals[given[1]]})),
            sp.simplify(s[mu].subs({g1: vals[given[0]], g2: vals[given[1]]})))


def _solid_check(keys, V, fname):
    for k, ok in (("AnalysisType", "Automatic"), ("MaterialSymmetry", "Isotropic"),
                  ("SolidMechanicsMaterialModel", "LinearElastic"), ("SolidMechanicsModelForm", "Solid")):
        if k in keys:
            v = keys[k]
            if not (v == ok or (isinstance(v, Symbol) and v.name == ok)):
                raise NotImplementedError(f"{fname}: \"{k}\" -> {v} is not supported (only {ok})")
    for k in ("InitialStress", "InitialStrain", "Force", "NormalForce", "NormalPressure"):
        if k in keys:
            raise NotImplementedError(f"{fname}: the parameter \"{k}\" is not supported")
    if V.n != 3 or len(V.deps) != 3:
        raise NotImplementedError(f"{fname}: only 3-D solids {{{{u, v, w}}, {{x, y, z}}}} are supported (plane "
                                  "stress/strain model forms are not)")


@fn("SolidMechanicsPDEComponent", area=AREA, status="added",
    notes="SolidMechanicsPDEComponent[{{u, v, w}, {x, y, z}}, pars] for linear elastic isotropic 3-D solids: rows "
          "-Div[sigma(u)]_i - f_i (+ rho D[u_i, {t, 2}] for time vars, - rho omega^2 u_i for frequency vars) written, "
          "as Wolfram does, as Sum_j Inactive[Div][-C_ij . Inactive[Grad][u_j, X], X] with (C_ij)_kl = lambda "
          "d_ik d_jl + mu (d_ij d_kl + d_il d_jk); any two of \"YoungModulus\", \"PoissonRatio\", \"LameParameter\", "
          "\"ShearModulus\", \"BulkModulus\", \"PWaveModulus\" (pairs that do not fix the Lame parameters uniquely "
          "raise NotImplementedError), \"MassDensity\", \"BodyLoad\" f, \"BodyLoadValue\" g (adds -rho g); 2-D plane "
          "stress/strain forms, other material models/symmetries, \"Material\" entities and eigenmode analysis raise "
          "NotImplementedError. Check: Activate compared with the Navier-Cauchy equations (lambda + mu) Grad[Div[u]] "
          "+ mu Laplacian[u] in tests.")
def _solid_component(vars_: Expr, pars: Assoc) -> ListOf[Expr]:
    fname = "SolidMechanicsPDEComponent"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _SOLID_KEYS, allow_freq=True, allow_system=True)
    _solid_check(keys, V, fname)
    lam, mu = _lame(keys, fname)
    n = 3
    rho = keys.get("MassDensity")
    f = _vector(_p(keys, "BodyLoad", _zero_vec(n)), n, fname, "\"BodyLoad\"")
    g = keys.get("BodyLoadValue")
    if (V.t is not None or V.w is not None or g is not None) and rho is None:
        raise ValueError(f"{fname}: \"MassDensity\" is required for dynamic analyses and \"BodyLoadValue\"")
    rows = []
    for i in range(n):
        parts = []
        if V.t is not None:
            parts.append(_times(rho, _time_d(V.deps[i], V.t, 2)))
        for j in range(n):
            C = [[R(-(lam * (1 if (i == k and j == l) else 0) +
                      mu * ((1 if (i == j and k == l) else 0) + (1 if (i == l and j == k) else 0))))
                  for l in range(n)] for k in range(n)]
            parts.append(_div(Expr("Dot", L(*(L(*r) for r in C)), _grad(V.deps[j], V.XL)), V.XL))
        if V.w is not None:
            parts.append(_times(-1, rho, Expr("Power", V.w, 2), V.deps[i]))
        parts.append(_neg(f.args[i]))
        if g is not None:
            gv = _vector(g, n, fname, "\"BodyLoadValue\"")
            parts.append(_times(-1, rho, gv.args[i]))
        rows.append(_plus(*parts))
    return _finish(L(*rows), subs)


def _displ(d, V, fname):
    if not is_list(d) or len(d.args) != len(V.deps):
        raise ValueError(f"{fname}: the displacement must be a list of {len(V.deps)} expressions")
    return list(d.args)


def _sym_matrix_arg(m, fname, what="tensor"):
    if _head_is(m, "SymmetrizedArray"):
        m = ev(Expr("Normal", m))
    if not _is_matrix(m) or len(m.args) not in (2, 3):
        if isinstance(m, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: the {what} must be a 2 x 2 or 3 x 3 symmetric matrix (or SymmetrizedArray)")
    rows = [list(r.args) for r in m.args]
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if not K.same(rows[i][j], rows[j][i]) and sp.simplify(S(rows[i][j]) - S(rows[j][i])) != 0:
                raise ValueError(f"{fname}: the {what} must be symmetric")
    if n == 2:
        rows = [rows[0] + [0], rows[1] + [0], [0, 0, 0]]
    return rows


def _engineering(keys, fname):
    e = keys.get("EngineeringStrain", True)
    if e not in (True, False):
        raise ValueError(f"{fname}: \"EngineeringStrain\" must be True or False")
    return e


def _symarray(rows):
    return ev(Expr("SymmetrizedArray", L(*(L(*r) for r in rows))))


@fn("SolidMechanicsStrain", area=AREA, status="added",
    notes="SolidMechanicsStrain[vars, pars, displ] -> SymmetrizedArray of the infinitesimal strains of displacement "
          "displ = {du, dv, dw} in 3-D: normal strains D[d_i, x_i] and engineering shear strains D[d_i, x_j] + "
          "D[d_j, x_i] (\"EngineeringStrain\" -> False gives the tensor strains, half of those); 2-D plane forms "
          "and InterpolatingFunction displacements on meshes raise NotImplementedError; a displacement of the wrong "
          "length raises ValueError. Check: reference (Wolfram documentation example: torsion displacement {-a y "
          "z, a x z, w[x, y]}) and the identity strain = {{1,2,2},{2,1,2},{2,2,1}} * tensor strain in tests.")
def _solid_strain(vars_: Expr, pars: Assoc, displ: ListOf[Expr]) -> Expr:
    fname = "SolidMechanicsStrain"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _SOLID_KEYS, allow_system=True)
    _solid_check(keys, V, fname)
    d = _displ(displ, V, fname)
    X = V.X
    eng = _engineering(keys, fname)
    rows = []
    for i in range(3):
        row = []
        for j in range(3):
            if i == j:
                row.append(ev(Expr("D", d[i], X[i])))
            else:
                s_ = ev(Expr("Plus", Expr("D", d[i], X[j]), Expr("D", d[j], X[i])))
                row.append(s_ if eng else ev(Expr("Times", Expr("Rational", 1, 2), s_)))
        rows.append(row)
    return _apply_subs(_symarray(rows), subs)


@fn("SolidMechanicsStress", area=AREA, status="added",
    notes="SolidMechanicsStress[vars, pars, strain] -> SymmetrizedArray of the Cauchy stresses of a linear elastic "
          "isotropic 3-D solid: sigma_ii = lambda tr(eps) + 2 mu eps_ii, tau_ij = mu gamma_ij for engineering shear "
          "strains (2 mu eps_ij with \"EngineeringStrain\" -> False); strain is a 3 x 3 symmetric matrix or "
          "SymmetrizedArray as returned by SolidMechanicsStrain; moduli as for SolidMechanicsPDEComponent; other "
          "\"OutputStressMeasure\" values, initial strains/stresses and 2-D forms raise NotImplementedError. Check: "
          "reference (Hooke's law; Wolfram documentation example tau = Y gamma/(2 (1 + nu))) in tests.")
def _solid_stress(vars_: Expr, pars: Assoc, strain: Expr) -> Expr:
    fname = "SolidMechanicsStress"
    V, keys, subs, axi = _component_args(vars_, pars, fname, _SOLID_KEYS, allow_system=True)
    _solid_check(keys, V, fname)
    osm = keys.get("OutputStressMeasure", "Cauchy")
    if osm != "Cauchy":
        raise NotImplementedError(f"{fname}: only the \"Cauchy\" output stress measure is supported")
    lam, mu = _lame(keys, fname)
    eps = [[S(v) for v in r] for r in _sym_matrix_arg(strain, fname, "strain")]
    eng = _engineering(keys, fname)
    tr = eps[0][0] + eps[1][1] + eps[2][2]
    rows = [[R(sp.expand(lam * tr + 2 * mu * eps[i][i]) if i == j else
               (mu * eps[i][j] if eng else 2 * mu * eps[i][j])) for j in range(3)] for i in range(3)]
    return _apply_subs(_symarray(rows), subs)


def _mises(rows, fname):
    a = [[S(v) for v in r] for r in rows]
    return R(sp.sqrt(sp.Rational(1, 2) * ((a[0][0] - a[1][1]) ** 2 + (a[1][1] - a[2][2]) ** 2 +
                                          (a[2][2] - a[0][0]) ** 2 + 6 * (a[0][1] ** 2 + a[0][2] ** 2 +
                                                                          a[1][2] ** 2))))


@fn("VonMisesStress", area=AREA, status="added",
    notes="VonMisesStress[vars, pars, stress] -> Sqrt[((sxx - syy)^2 + (syy - szz)^2 + (szz - sxx)^2 + 6 (txy^2 + "
          "txz^2 + tyz^2))/2] for a symmetric 3 x 3 (or 2 x 2, padded with zeros) stress matrix or SymmetrizedArray; "
          "numeric stresses give a number, symbolic ones a formula; non-symmetric or wrongly shaped input raises "
          "ValueError. Check: reference (uniaxial stress s -> |s|, pure shear t -> Sqrt[3] |t|) in tests.")
def _von_mises(vars_: Expr, pars: Assoc, stress: Expr) -> Expr:
    fname = "VonMisesStress"
    parse_vars(vars_, fname, allow_system=True)
    parse_pars(pars, fname, known=_SOLID_KEYS)
    return ev(_mises(_sym_matrix_arg(stress, fname, "stress"), fname))


@fn("EquivalentStrain", area=AREA, status="added",
    notes="EquivalentStrain[vars, pars, strain] -> Sqrt[((exx - eyy)^2 + (eyy - ezz)^2 + (ezz - exx)^2 + 6 (gxy^2 + "
          "gxz^2 + gyz^2))/2] (the documented formula on the strain matrix entries) for a symmetric 3 x 3 (or 2 x 2, "
          "zero padded) matrix or SymmetrizedArray; non-symmetric or wrongly shaped input raises ValueError. "
          "Check: reference (Wolfram documentation formula) in tests.")
def _equivalent_strain(vars_: Expr, pars: Assoc, strain: Expr) -> Expr:
    fname = "EquivalentStrain"
    parse_vars(vars_, fname, allow_system=True)
    parse_pars(pars, fname, known=_SOLID_KEYS)
    return ev(_mises(_sym_matrix_arg(strain, fname, "strain"), fname))


@fn("SolidDisplacementCondition", area=AREA, status="added",
    notes=_bc_doc("SolidDisplacementCondition", "DirichletCondition[{u == d1, v == d2, w == d3}, pred] with "
                  "\"Displacement\" {d1, d2, d3} (default 0; None entries leave that component free)",
                  "A displacement of the wrong length raises ValueError. "))
def _solid_displacement(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    fname = "SolidDisplacementCondition"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _SOLID_KEYS, allow_system=True)
    m = len(V.deps)
    d = _p(keys, "Displacement", _zero_vec(m))
    if not is_list(d) or len(d.args) != m:
        raise ValueError(f"{fname}: \"Displacement\" must be a list of length {m}")
    pairs = [(u, v) for u, v in zip(V.deps, d.args) if not (isinstance(v, Symbol) and v.name == "None")]
    if not pairs:
        raise ValueError(f"{fname}: every displacement component is None")
    return _dirichlet(L(*(p[0] for p in pairs)), L(*(p[1] for p in pairs)), pred, subs)


@fn("SolidFixedCondition", area=AREA, status="added",
    notes=_bc_doc("SolidFixedCondition", "DirichletCondition[{u == 0, v == 0, ...}, pred] (all displacements "
                  "fixed)"))
def _solid_fixed(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> Expr:
    V, keys, subs = _bc_args(pred, vars_, pars, rest, "SolidFixedCondition", _SOLID_KEYS, allow_system=True)
    return _dirichlet(L(*V.deps), _zero_vec(len(V.deps)), pred, subs)


@fn("SolidBoundaryLoadValue", area=AREA, status="added",
    notes=_bc_doc("SolidBoundaryLoadValue", "{NeumannValue[p_i - k u_i - c D[u_i, t], pred], ...} (zero components "
                  "omitted as NeumannValue[0, pred]) for a \"Pressure\" (traction) vector p, \"SpringConstant\" k and "
                  "\"DampingConstant\" c (time vars)", "\"Force\", \"NormalForce\" (need the boundary area) and "
                  "\"NormalPressure\" raise NotImplementedError; \"DampingConstant\" without time vars raises "
                  "ValueError. "))
def _solid_load(pred: Expr, vars_: Expr, pars: Assoc, *rest: Expr) -> ListOf[Expr]:
    fname = "SolidBoundaryLoadValue"
    V, keys, subs = _bc_args(pred, vars_, pars, rest, fname, _SOLID_KEYS, allow_system=True)
    for k in ("Force", "NormalForce", "NormalPressure"):
        if k in keys:
            raise NotImplementedError(f"{fname}: \"{k}\" is not supported; give the traction as \"Pressure\"")
    m = len(V.deps)
    p = _vector(_p(keys, "Pressure", _zero_vec(m)), m, fname, "\"Pressure\"")
    k, c = _p(keys, "SpringConstant", 0), _p(keys, "DampingConstant", 0)
    if not _is_zero(c) and V.t is None:
        raise ValueError(f"{fname}: \"DampingConstant\" needs time-dependent variables")
    out = []
    for i in range(m):
        val = _plus(p.args[i], _times(-1, k, V.deps[i]),
                    (_times(-1, c, _time_d(V.deps[i], V.t)) if V.t is not None else 0))
        out.append(Expr("NeumannValue", val, pred))
    return _apply_subs(ev(L(*out)), subs)


# ───────────────────────────── fluid flow ─────────────────────────────

_FLUID_KEYS = ("CoordinateChart", "DynamicViscosity", "FluidLoad", "FluidDynamicsMaterialModel", "MassDensity",
               "Material", "ModelForm", "RegionSymmetry", "ReynoldsNumber")


def _fluid_setup(vars_, pars, fname, need_rho=True):
    V, keys, subs, axi = _component_args(vars_, pars, fname, _FLUID_KEYS, allow_system=True)
    if axi:
        raise NotImplementedError(f"{fname}: axisymmetric flow is not supported")
    mm = keys.get("FluidDynamicsMaterialModel", "Newtonian")
    if mm != "Newtonian":
        raise NotImplementedError(f"{fname}: only the \"Newtonian\" material model is supported")
    if not V.system or V.n not in (2, 3) or len(V.deps) != V.n + 1:
        raise ValueError(f"{fname}: the variables must be {{{{u, v, (w,) p}}, {{x, y, (z)}}}} (velocities and "
                         "pressure)")
    if "ReynoldsNumber" in keys:
        if "DynamicViscosity" in keys or "MassDensity" in keys:
            raise ValueError(f"{fname}: give either \"ReynoldsNumber\" or \"DynamicViscosity\" and \"MassDensity\"")
        rho, mu = 1, _recip(keys["ReynoldsNumber"])
    else:
        if "DynamicViscosity" not in keys or (need_rho and "MassDensity" not in keys):
            raise ValueError(f"{fname}: \"DynamicViscosity\"" + (" and \"MassDensity\"" if need_rho else "")
                             + " (or \"ReynoldsNumber\") required")
        rho, mu = keys.get("MassDensity", 1), keys["DynamicViscosity"]
    names = {x.name for x in V.X} | {d.head for d in V.deps} | ({V.t.name} if V.t is not None else set())
    if _free_names(rho) & names:
        raise NotImplementedError(f"{fname}: compressible flow (a variable \"MassDensity\") is not supported")
    return V, keys, subs, rho, mu


def _free_names(e):
    out = set()
    if isinstance(e, Symbol):
        out.add(e.name)
    elif isinstance(e, Expr):
        if isinstance(e.head, str):
            out.add(e.head)
        else:
            out |= _free_names(e.head)
        for a in e.args:
            out |= _free_names(a)
    return out


def _viscous_C(n, i, j, mu):
    return L(*(L(*(_neg(_times(mu, (1 if (i == j and k == l) else 0) + (1 if (i == l and j == k) else 0)))
                   if ((i == j and k == l) or (i == l and j == k)) else 0 for l in range(n))) for k in range(n)))


@fn("FluidFlowPDEComponent", area=AREA, status="added",
    notes="FluidFlowPDEComponent[{{u, v, (w,) p}, X}, pars] for incompressible Newtonian flow (constant "
          "\"MassDensity\" rho and \"DynamicViscosity\" mu, or \"ReynoldsNumber\" Re: rho = 1, mu = 1/Re): momentum "
          "rows rho D[v_i, t] + Inactive[Div][rho v_i {u, v, ...}, X] (\"ModelForm\" \"Conservative\", the default) "
          "or rho {u, v, ...} . Inactive[Grad][v_i, X] (\"NonConservative\") + e_i . Inactive[Grad][p, X] + "
          "Sum_j Inactive[Div][-C_ij . Inactive[Grad][v_j, X], X] (viscous stress mu (Grad v + Grad v^T)) - f_i "
          "(\"FluidLoad\" f), and the continuity row Sum_j e_j . Inactive[Grad][v_j, X]; compressible "
          "(variable density), non-Newtonian, axisymmetric and \"Material\" models raise NotImplementedError; "
          "missing material constants raise ValueError. Check: Activate compared with the written-out "
          "Navier-Stokes equations in tests.")
def _fluid_component(vars_: Expr, pars: Assoc) -> ListOf[Expr]:
    fname = "FluidFlowPDEComponent"
    V, keys, subs, rho, mu = _fluid_setup(vars_, pars, fname)
    n = V.n
    vel, p = V.deps[:n], V.deps[n]
    f = _vector(_p(keys, "FluidLoad", _zero_vec(n)), n, fname, "\"FluidLoad\"")
    cons = _model_form(keys, fname, default="Conservative")
    rows = []
    for i in range(n):
        parts = []
        if V.t is not None:
            parts.append(_times(rho, _time_d(vel[i], V.t)))
        if cons:
            parts.append(_div(ev(Expr("Times", rho, vel[i], L(*vel))), V.XL))
        else:
            parts.append(Expr("Dot", ev(Expr("Times", rho, L(*vel))), _grad(vel[i], V.XL)))
        parts.append(Expr("Dot", L(*((1 if k == i else 0) for k in range(n))), _grad(p, V.XL)))
        for j in range(n):
            parts.append(_div(Expr("Dot", ev(_viscous_C(n, i, j, mu)), _grad(vel[j], V.XL)), V.XL))
        parts.append(_neg(f.args[i]))
        rows.append(_plus(*parts))
    rows.append(_plus(*(Expr("Dot", L(*((1 if k == j else 0) for k in range(n))), _grad(vel[j], V.XL))
                        for j in range(n))))
    return _finish(L(*rows), subs)


def _velocity(vel, V, fname):
    if not is_list(vel) or len(vel.args) != V.n:
        raise ValueError(f"{fname}: the velocity must be a list of {V.n} expressions")
    return list(vel.args)


@fn("FluidViscosity", area=AREA, status="added",
    notes="FluidViscosity[vars, pars, velocity] -> the dynamic viscosity mu of a Newtonian fluid (1/Re for "
          "\"ReynoldsNumber\" Re); non-Newtonian models raise NotImplementedError; a velocity of the wrong length "
          "raises ValueError. Check: reference (Wolfram documentation: mu, 1/Re) in tests.")
def _fluid_viscosity(vars_: Expr, pars: Assoc, velocity: ListOf[Expr]) -> Expr:
    fname = "FluidViscosity"
    V, keys, subs, rho, mu = _fluid_setup(vars_, pars, fname, need_rho=False)
    _velocity(velocity, V, fname)
    return _apply_subs(ev(mu), subs)


@fn("FluidViscousStress", area=AREA, status="added",
    notes="FluidViscousStress[vars, pars, velocity] -> SymmetrizedArray mu (Grad[velocity] + Transpose[Grad[velocity]]) "
          "(the viscous stress 2 mu strain rate of an incompressible Newtonian fluid); non-Newtonian and "
          "compressible models raise NotImplementedError. Check: reference (Wolfram documentation: "
          "{{2 mu U_x, mu (U_y + V_x)}, {mu (U_y + V_x), 2 mu V_y}}) in tests.")
def _fluid_viscous_stress(vars_: Expr, pars: Assoc, velocity: ListOf[Expr]) -> Expr:
    fname = "FluidViscousStress"
    V, keys, subs, rho, mu = _fluid_setup(vars_, pars, fname, need_rho=False)
    vel = _velocity(velocity, V, fname)
    rows = [[ev(Expr("Times", mu, Expr("Plus", Expr("D", vel[i], V.X[j]), Expr("D", vel[j], V.X[i]))))
             for j in range(V.n)] for i in range(V.n)]
    return _apply_subs(_symarray(rows), subs)


# ───────────────────────────── magnetic field quantities ─────────────────────────────

def _potential_kind(V, pot, fname):
    if V.system:
        if len(V.deps) != 3 or V.n != 3:
            raise NotImplementedError(f"{fname}: vector potentials must be 3-D {{{{Ax, Ay, Az}}, {{x, y, z}}}}")
        if not is_list(pot) or len(pot.args) != 3:
            raise ValueError(f"{fname}: a vector potential must be a list of 3 expressions")
        return "vector"
    if is_list(pot):
        raise ValueError(f"{fname}: a scalar potential must be a scalar expression")
    return "scalar"


@fn("MagneticFieldIntensity", area=AREA, status="added",
    notes="MagneticFieldIntensity[vars, pars, Vm] -> -Grad[Vm, X] for a magnetic scalar potential; "
          "MagneticFieldIntensity[vars, pars, A] -> Curl[A, X]/mu0 - M for a 3-D vector potential (\"VacuumPermeability\" "
          "mu0, default Pi/2500000; \"Magnetization\" M); a \"RelativePermeability\" with a vector potential raises "
          "NotImplementedError. Check: reference (Wolfram documentation forms) in tests.")
def _m_field_intensity(vars_: Expr, pars: Assoc, pot: Expr) -> ListOf[Expr]:
    fname = "MagneticFieldIntensity"
    V = parse_vars(vars_, fname, allow_system=True)
    keys, subs = parse_pars(pars, fname, known=_M_BC_KEYS)
    if _potential_kind(V, pot, fname) == "scalar":
        return _apply_subs(ev(Expr("Times", -1, Expr("Grad", pot, V.XL))), subs)
    if "RelativePermeability" in keys:
        raise NotImplementedError(f"{fname}: \"RelativePermeability\" is not supported for vector potentials")
    mu0 = _p(keys, "VacuumPermeability", MU0)
    M = _vector(_p(keys, "Magnetization", _zero_vec(3)), 3, fname, "\"Magnetization\"")
    return _apply_subs(ev(Expr("Plus", Expr("Times", Expr("Power", mu0, -1), Expr("Curl", pot, V.XL)),
                               Expr("Times", -1, M))), subs)


@fn("MagneticFluxDensity", area=AREA, status="added",
    notes="MagneticFluxDensity[vars, pars, Vm] -> mur mu0 (-Grad[Vm, X] + M) for a magnetic scalar potential "
          "(\"RelativePermeability\" mur scalar or matrix, \"VacuumPermeability\" mu0 default Pi/2500000, "
          "\"Magnetization\" M); MagneticFluxDensity[vars, pars, A] -> Curl[A, X] - mu0 M for a 3-D vector "
          "potential. Check: reference (Wolfram documentation forms) in tests.")
def _m_flux_density(vars_: Expr, pars: Assoc, pot: Expr) -> ListOf[Expr]:
    fname = "MagneticFluxDensity"
    V = parse_vars(vars_, fname, allow_system=True)
    keys, subs = parse_pars(pars, fname, known=_M_BC_KEYS)
    mu0 = _p(keys, "VacuumPermeability", MU0)
    kind = _potential_kind(V, pot, fname)
    n = 3 if kind == "vector" else V.n
    M = _vector(_p(keys, "Magnetization", _zero_vec(n)), n, fname, "\"Magnetization\"")
    if kind == "vector":
        return _apply_subs(ev(Expr("Plus", Expr("Curl", pot, V.XL), Expr("Times", -1, mu0, M))), subs)
    mur = _p(keys, "RelativePermeability", 1)
    H = Expr("Plus", Expr("Times", -1, Expr("Grad", pot, V.XL)), M)
    if _is_matrix(mur, V.n):
        return _apply_subs(ev(Expr("Times", mu0, Expr("Dot", mur, H))), subs)
    return _apply_subs(ev(Expr("Times", mur, mu0, H)), subs)


# ───────────────────────────── spectral collocation (NDEigensystem, NDSolve BVPs) ─────────────────────────────
#
# Operators are discretised by Chebyshev-Lobatto collocation on intervals and (tensor-product) rectangles.
# Boundary rows follow Wolfram's finite-element conventions: no condition = natural flux n.(c Grad u + alpha u -
# gamma) = 0, NeumannValue[g - q u, pred] on the right-hand side sets that flux to g - q u (on the left-hand side
# the sign flips), DirichletCondition rows fix values, and periodic conditions identify values and fluxes.

import numpy as np  # noqa: E402

from ..expr import Expr as _Expr  # noqa: E402,F401


def _cheb(N, a, b):
    """Ascending Chebyshev-Lobatto nodes on [a, b] and the first-derivative matrix."""
    j = np.arange(N + 1)
    x = np.cos(np.pi * j / N)
    c = np.ones(N + 1)
    c[0] = c[-1] = 2.0
    c = c * (-1.0) ** j
    dX = x[:, None] - x[None, :]
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    x, D = x[::-1].copy(), D[::-1, ::-1].copy()
    return a + (x + 1.0) * (b - a) / 2.0, D * (2.0 / (b - a))


def _clencurt(N, a, b):
    """Clenshaw-Curtis quadrature weights for the ascending Chebyshev-Lobatto nodes on [a, b]."""
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    ii = np.arange(1, N)
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1.0 / (N ** 2 - 1)
        for k in range(1, N // 2):
            v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
        v = v - np.cos(N * theta[ii]) / (N ** 2 - 1)
    else:
        w[0] = w[N] = 1.0 / N ** 2
        for k in range(1, (N - 1) // 2 + 1):
            v = v - 2 * np.cos(2 * k * theta[ii]) / (4 * k * k - 1)
    w[ii] = 2 * v / N
    return w[::-1] * (b - a) / 2.0


def _bary_w(N):
    w = (-1.0) ** np.arange(N + 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def _bary(xs, w, vals, t):
    d = t - xs
    hit = np.nonzero(d == 0)[0]
    if hit.size:
        return vals[hit[0]]
    q = w / d
    return np.tensordot(q, vals, axes=(0, 0)) / q.sum()


class _ChebFun:
    """Chebyshev interpolant u(t) (and derivatives) through nodal values on [a, b]."""

    def __init__(self, xs, D, vals):
        self.xs, self.D, self.w = xs, D, _bary_w(len(xs) - 1)
        self.vals = [np.asarray(vals)]

    def _deriv(self, k):
        while len(self.vals) <= k:
            self.vals.append(self.D @ self.vals[-1])
        return self.vals[k]

    def __call__(self, t, k=0):
        return complex(_bary(self.xs, self.w, self._deriv(k), t))


class _ChebFun2:
    """Tensor-product Chebyshev interpolant u(x, y) through nodal values on a rectangle."""

    def __init__(self, xs, ys, V):
        self.xs, self.ys, self.V = xs, ys, np.asarray(V)
        self.wx, self.wy = _bary_w(len(xs) - 1), _bary_w(len(ys) - 1)

    def __call__(self, x, y):
        col = _bary(self.xs, self.wx, self.V, x)
        return complex(_bary(self.ys, self.wy, col, y))


def _scipy_linalg(fname):
    try:
        import scipy.linalg as sl
    except ImportError:
        raise NotImplementedError(f"{fname}: requires SciPy (pip install scipy)")
    return sl


class _Region:
    """Interval [a, b] or rectangle [a, b] x [c, d] with exact (Wolfram) and float bounds."""

    def __init__(self, xs, lo, hi):
        self.xs, self.lo, self.hi = xs, lo, hi             # Symbols, exact bounds (Wolfram exprs)
        self.flo = [_real(v) for v in lo]
        self.fhi = [_real(v) for v in hi]
        self.dim = len(xs)


def _real(v, fname="NDEigensystem"):
    try:
        z = complex(sp.N(S(v)))
    except Exception:
        raise ValueError(f"{fname}: the region bounds must be real numbers")
    if z.imag != 0:
        raise ValueError(f"{fname}: the region bounds must be real numbers")
    return z.real


def _region_of(specs, fname):
    """{x, a, b} [, {y, c, d}], x in Interval[{a, b}], {x} in Line[{{a}, {b}}], {x, y} in Rectangle[...]."""
    xs, lo, hi = [], [], []
    if len(specs) == 1 and _head_is(specs[0], "Element"):
        v, reg = specs[0].args
        vs = list(v.args) if is_list(v) else [v]
        if _head_is(reg, "Interval") and len(vs) == 1 and len(reg.args) == 1 and is_list(reg.args[0]):
            lo, hi = [reg.args[0].args[0]], [reg.args[0].args[1]]
        elif _head_is(reg, "Line") and len(vs) == 1 and len(reg.args) == 1 and len(reg.args[0].args) == 2:
            lo, hi = [reg.args[0].args[0].args[0]], [reg.args[0].args[1].args[0]]
        elif _head_is(reg, "Rectangle") and len(vs) == 2:
            p, q = (reg.args if len(reg.args) == 2 else (L(0, 0), L(1, 1)))
            lo, hi = list(p.args), list(q.args)
        else:
            raise NotImplementedError(f"{fname}: only intervals and rectangles are supported (not {reg})")
        xs = vs
    else:
        for s_ in specs:
            if not (is_list(s_) and len(s_.args) == 3):
                raise ValueError(f"{fname}: region specifications must be {{x, xmin, xmax}}")
            xs.append(s_.args[0])
            lo.append(s_.args[1])
            hi.append(s_.args[2])
    if not xs or len(xs) > 2:
        raise NotImplementedError(f"{fname}: only one- and two-dimensional regions are supported")
    for x in xs:
        if not isinstance(x, Symbol):
            raise ValueError(f"{fname}: {x} is not a valid variable")
    R_ = _Region(xs, lo, hi)
    for a, b in zip(R_.flo, R_.fhi):
        if not a < b:
            raise ValueError(f"{fname}: the region must have xmin < xmax")
    return R_


def _uname(u, fname):
    if isinstance(u, Symbol):
        return u.name
    if isinstance(u, Expr) and isinstance(u.head, str) and u.args:
        return u.head
    if is_list(u):
        raise NotImplementedError(f"{fname}: coupled systems are not supported")
    raise ValueError(f"{fname}: {u} is not an unknown function")


def _signed_terms(e, s=1):
    """[(sign, term)] of a sum, distributing numeric factors."""
    if _head_is(e, "Plus"):
        return [p for a in e.args for p in _signed_terms(a, s)]
    if _head_is(e, "Times") and len(e.args) == 2 and _is_num(e.args[0]) and isinstance(e.args[1], Expr) and (
            _head_is(e.args[1], "Plus") or _head_is(e.args[1], "NeumannValue") or _is_div(e.args[1])):
        return _signed_terms(e.args[1], ev(Expr("Times", s, e.args[0])))
    return [(s, e)]


def _is_div(e):
    return (isinstance(e, Expr) and isinstance(e.head, Expr) and e.head.head == "Inactive"
            and e.head.args and isinstance(e.head.args[0], Symbol) and e.head.args[0].name == "Div")


def _contains(e, name):
    if isinstance(e, Expr):
        if e.head == name or (isinstance(e.head, Expr) and _contains(e.head, name)):
            return True
        return any(_contains(a, name) for a in e.args)
    return isinstance(e, Symbol) and e.name == name


class _Op:
    """A linear operator split into flux (Div) terms, other terms and NeumannValue terms."""

    def __init__(self, e, fname):
        if _head_is(e, "Equal"):
            if len(e.args) != 2:
                raise ValueError(f"{fname}: equations must have two sides")
            terms = _signed_terms(e.args[0], 1) + _signed_terms(e.args[1], -1)
        else:
            terms = _signed_terms(e, 1)
        self.nv, self.div, self.rest = [], [], []
        for s_, t in terms:
            if _head_is(t, "NeumannValue"):
                if len(t.args) != 2:
                    raise ValueError(f"{fname}: NeumannValue[value, predicate] expected")
                self.nv.append((s_, t.args[0], t.args[1]))
            elif _contains(t, "NeumannValue"):
                raise NotImplementedError(f"{fname}: NeumannValue must appear as a separate term of the equation")
            elif _is_div(t):
                self.div.append((s_, t))
            else:
                self.rest.append((s_, t))


def _lin_form(e, uname, svars, fname, what="operator"):
    """e (SymPy) linear in u and its derivatives -> ({derivative orders: coefficient}, constant term)."""
    uf = sp.Function(uname)(*svars)
    reps, ys = {}, {}
    ders = sorted((d for d in e.atoms(sp.Derivative) if d.expr == uf), key=lambda d: -d.derivative_count)
    for d in ders:
        cnt = tuple(sum(c for v, c in d.variable_count if v == s_) for s_ in svars)
        y = ys.setdefault(cnt, sp.Symbol(f"__Y{'_'.join(map(str, cnt))}"))
        reps[d] = y
    e = e.xreplace(reps)
    ys.setdefault(tuple(0 for _ in svars), sp.Symbol("__Y" + "_".join("0" for _ in svars)))
    e = e.xreplace({uf: ys[tuple(0 for _ in svars)]})
    if any(isinstance(f, sp.core.function.AppliedUndef) and f.func.__name__ == uname for f in e.atoms(sp.Function)) \
            or any(isinstance(d, (sp.Derivative, sp.Subs)) and d.has(sp.Function(uname)) for d in
                   e.atoms(sp.Derivative, sp.Subs)):
        raise NotImplementedError(f"{fname}: the {what} must involve {uname} only at the variables "
                                  f"{tuple(map(str, svars))}")
    e = sp.expand(e)
    coeffs = {}
    syms = list(ys.values())
    for cnt, y in ys.items():
        c = e.coeff(y)
        if c.has(*syms):
            raise NotImplementedError(f"{fname}: the {what} must be linear in {uname}")
        if c != 0:
            coeffs[cnt] = c
    const = sp.expand(e - sum(c * ys[k] for k, c in coeffs.items()))
    if const.has(*syms):
        raise NotImplementedError(f"{fname}: the {what} must be linear in {uname}")
    return coeffs, const


def _sympy_of(e, fname):
    v = S(e)
    if isinstance(v, (list, tuple)):
        return [sp.sympify(a) for a in v]
    return sp.sympify(v)


class _Problem:
    """Discretisable linear problem: interior form, flux form, boundary items."""

    def __init__(self, fname, op_items, u, region, t=None, stationary_only=False):
        self.fname, self.region, self.t = fname, region, t
        self.uname = _uname(u, fname)
        self.svars = [sp.Symbol(x.name) for x in region.xs]
        if not op_items:
            raise ValueError(f"{fname}: no equation given")
        main, extra = None, []
        for it in op_items:
            if _head_is(it, "DirichletCondition") or _head_is(it, "PeriodicBoundaryCondition"):
                extra.append(it)
            elif main is None:
                main = it
            elif _head_is(it, "Equal") and not self._has_u_at_vars(it):
                extra.append(it)
            else:
                raise NotImplementedError(f"{fname}: coupled systems and several equations are not supported")
        if main is None:
            raise ValueError(f"{fname}: no equation given")
        op = _Op(main, fname)
        self.nv = op.nv
        tv = [sp.Symbol(t.name)] if t is not None else []
        allv = tv + self.svars
        full = Expr("Plus", 0, *(Expr("Times", s_, t_) for s_, t_ in op.rest),
                    *(Expr("Times", s_, Expr(_inactive("Div"), G[0], G[1])) for s_, G in
                      ((s_, self._div_parts(g)) for s_, g in op.div)))
        self.coeffs, self.const = _lin_form(_sympy_of(ev(Expr("Activate", full)), fname),
                                            self.uname, allv, fname)
        restc, _ = _lin_form(_sympy_of(ev(Expr("Activate", Expr("Plus", 0, *(Expr("Times", s_, t_)
                                                                                for s_, t_ in op.rest)))),
                                       fname), self.uname, allv, fname)
        n = len(self.svars)
        # time structure
        self.tcoef = {}
        if t is not None:
            for k, c in list(self.coeffs.items()):
                if k[0]:
                    if any(k[1:]):
                        raise NotImplementedError(f"{fname}: mixed time-space derivatives are not supported")
                    if k[0] > 2:
                        raise NotImplementedError(f"{fname}: time derivatives above second order are not supported")
                    self.tcoef[k[0]] = c
                    del self.coeffs[k]
            self.coeffs = {k[1:]: c for k, c in self.coeffs.items()}
            restc = {k[1:]: c for k, c in restc.items() if not k[0]}
            if not self.tcoef:
                raise ValueError(f"{fname}: no time derivative of {self.uname} appears in the equation")
            if any(c.has(allv[0]) for c in self.tcoef.values()) or any(
                    c.has(allv[0]) for c in self.coeffs.values()) or self.const.has(allv[0]):
                raise NotImplementedError(f"{fname}: the operator and boundary conditions need to be stationary "
                                          "(time-independent coefficients)")
        for k in self.coeffs:
            if sum(k) > 2:
                raise NotImplementedError(f"{fname}: only operators of order at most two in space are supported")
        if not any(sum(k) == 2 for k in self.coeffs):
            raise NotImplementedError(f"{fname}: a second-order spatial operator is required")
        # flux F = C Grad u + (alpha u - gamma) from the Div terms and the non-Div second-order part
        e_ = [tuple(1 if i == d else 0 for i in range(n)) for d in range(n)]
        self.flux = [dict() for _ in range(n)]          # per component: {order tuple: coef}, plus const
        self.flux_c = [sp.Integer(0)] * n
        for d in range(n):
            for d2 in range(n):
                k2 = tuple(e_[d][i] + e_[d2][i] for i in range(n))
                c = restc.get(k2, 0)
                if c != 0:
                    c = -c if d == d2 else -c / 2
                    self.flux[d][e_[d2]] = self.flux[d].get(e_[d2], 0) + c
        for s_, g in op.div:
            G, X = self._div_parts(g)
            Ga = ev(Expr("Activate", G))
            comps = list(Ga.args) if is_list(Ga) else [Ga]
            if len(comps) != n:
                raise ValueError(f"{fname}: a flux vector of length {n} is required")
            for d in range(n):
                cs, c0 = _lin_form(_sympy_of(comps[d], fname), self.uname, allv, fname, "flux")
                sgn = -S(s_)
                for k, c in cs.items():
                    kk = k[1:] if t is not None else k
                    if t is not None and k[0]:
                        raise NotImplementedError(f"{fname}: time derivatives inside fluxes are not supported")
                    if sum(kk) > 1:
                        raise NotImplementedError(f"{fname}: fluxes must be first order")
                    self.flux[d][kk] = self.flux[d].get(kk, 0) + sgn * c
                self.flux_c[d] = self.flux_c[d] + sgn * c0
        self.extra = extra

    def _has_u_at_vars(self, e):
        names = {x.name for x in self.region.xs}

        def walk(q):
            if isinstance(q, Expr):
                if q.head == self.uname and any(isinstance(a, Symbol) and a.name in names for a in q.args):
                    return True
                if isinstance(q.head, Expr) and _head_is(q.head, "Derivative") is False and walk(q.head):
                    return True
                if (isinstance(q.head, Expr) and isinstance(q.head.head, Expr) and q.head.head.head == "Derivative"
                        and any(isinstance(a, Symbol) and a.name in names for a in q.args)):
                    return True
                if _head_is(q, "D") or _head_is(q, "Laplacian") or _head_is(q, "Grad") or _is_div(q):
                    return True
                return any(walk(a) for a in q.args)
            return False
        return walk(e)

    def _div_parts(self, g):
        if len(g.args) != 2:
            raise ValueError(f"{self.fname}: Inactive[Div][flux, vars] expected")
        return g.args[0], g.args[1]


def _lam(expr, svars, fname="NDSolve"):
    extra = sp.sympify(expr).free_symbols - set(svars)
    if extra:
        raise ValueError(f"{fname}: the coefficient {expr} must be numerical (free symbols "
                         f"{', '.join(sorted(map(str, extra)))})")
    bad = [f for f in sp.sympify(expr).atoms(sp.Function) if isinstance(f, sp.core.function.AppliedUndef)]
    if bad:
        raise NotImplementedError(f"{fname}: the coefficient {expr} cannot be evaluated numerically "
                                  f"({bad[0].func})")
    f = sp.lambdify(svars, expr, "numpy")

    def call(*pts):
        try:
            with np.errstate(all="ignore"):
                v = np.asarray(f(*pts), dtype=complex)
        except (NameError, TypeError, ValueError) as exc:
            raise NotImplementedError(f"{fname}: the coefficient {expr} cannot be evaluated numerically ({exc})")
        return np.broadcast_to(v, np.broadcast(*pts).shape).astype(complex)
    return call


def _pred_true(pred, xs, pt, fname):
    if pred is True or (isinstance(pred, Symbol) and pred.name == "True"):
        return True
    rules = L(*(Expr("Rule", x, v) for x, v in zip(xs, pt)))
    r = ev(Expr("ReplaceAll", pred, rules))
    if r is True or (isinstance(r, Symbol) and r.name == "True"):
        return True
    if r is False or (isinstance(r, Symbol) and r.name == "False"):
        return False
    raise ValueError(f"{fname}: the predicate {pred} does not evaluate to True or False on the boundary")


class _Disc:
    """Collocation matrices for a _Problem at resolution N (per dimension)."""

    def __init__(self, P, N):
        self.P = P
        R_ = P.region
        self.dim = R_.dim
        self.grids = [_cheb(N, a, b) for a, b in zip(R_.flo, R_.fhi)]
        self.N = N
        if self.dim == 1:
            xs, D = self.grids[0]
            self.pts = [xs]
            self.Dk = {(0,): np.eye(N + 1), (1,): D, (2,): D @ D}
            self.size = N + 1
        else:
            (xs, Dx), (ys, Dy) = self.grids
            I = np.eye(N + 1)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            self.pts = [X.ravel(), Y.ravel()]
            Dx2, Dy2 = Dx @ Dx, Dy @ Dy
            self.Dk = {(0, 0): np.kron(I, I), (1, 0): np.kron(Dx, I), (0, 1): np.kron(I, Dy),
                       (2, 0): np.kron(Dx2, I), (0, 2): np.kron(I, Dy2), (1, 1): np.kron(Dx, Dy)}
            self.size = (N + 1) ** 2
        self.sv = P.svars

    def field(self, expr):
        return _lam(expr, self.sv, self.P.fname)(*self.pts)

    def operator(self, coeffs):
        A = np.zeros((self.size, self.size), dtype=complex)
        for k, c in coeffs.items():
            A += self.field(c)[:, None] * self.Dk[k]
        return A

    def boundary(self):
        """[(index, outward normal, exact point)] for boundary nodes; corners get the summed normal."""
        R_ = self.P.region
        N = self.N
        out = []
        if self.dim == 1:
            out.append((0, (-1.0,), [R_.lo[0]]))
            out.append((N, (1.0,), [R_.hi[0]]))
            return out
        xs, ys = self.grids[0][0], self.grids[1][0]
        for i in range(N + 1):
            for j in range(N + 1):
                if i not in (0, N) and j not in (0, N):
                    continue
                nx = -1.0 if i == 0 else (1.0 if i == N else 0.0)
                ny = -1.0 if j == 0 else (1.0 if j == N else 0.0)
                px = R_.lo[0] if i == 0 else (R_.hi[0] if i == N else float(xs[i]))
                py = R_.lo[1] if j == 0 else (R_.hi[1] if j == N else float(ys[j]))
                out.append((i * (N + 1) + j, (nx, ny), [px, py]))
        return out

    def flux_row(self, idx, normal):
        P = self.P
        row = np.zeros(self.size, dtype=complex)
        c0 = 0j
        pt = [p[idx] for p in self.pts]
        for d, nd in enumerate(normal):
            if nd == 0:
                continue
            for k, c in P.flux[d].items():
                row += nd * complex(_lam(c, self.sv, P.fname)(*pt)) * self.Dk[k][idx]
            c0 += nd * complex(_lam(P.flux_c[d], self.sv, P.fname)(*pt))
        return row, c0


def _point_value(expr_s, svars, pt_exact, fname="NDSolve"):
    v = sp.N(sp.sympify(expr_s).subs({s_: S(p) for s_, p in zip(svars, pt_exact)}))
    try:
        return complex(v)
    except TypeError:
        raise ValueError(f"{fname}: the boundary value {expr_s} must be numerical on the boundary")


def _assemble(P, Dz, eigen):
    """Rows (matrix A, B and right-hand side) with the boundary conditions applied."""
    fname = P.fname
    A = Dz.operator(P.coeffs)
    if eigen:
        if P.t is None:
            Bd = np.ones(Dz.size, dtype=complex)
            P.tord = 0
        else:
            if len(P.tcoef) != 1:
                raise NotImplementedError(f"{fname}: equations with both first and second time derivatives are "
                                          "not supported")
            (P.tord, c), = P.tcoef.items()
            Bd = -Dz.field(c)
        B = np.diag(Bd)
        if P.const != 0:
            raise ValueError(f"{fname}: the operator must be homogeneous (no source terms)")
    else:
        B = None
    rhs = -Dz.field(P.const) if P.const != 0 else np.zeros(Dz.size, dtype=complex)
    xs = P.region.xs
    bnodes = Dz.boundary()
    done = set()
    uname = P.uname
    # Dirichlet conditions
    for c in P.extra:
        if _head_is(c, "DirichletCondition"):
            if len(c.args) != 2 or not _head_is(c.args[0], "Equal"):
                raise ValueError(f"{fname}: DirichletCondition[u == value, predicate] expected")
            lhs, val = c.args[0].args
            if not (isinstance(lhs, Expr) and lhs.head == uname):
                raise NotImplementedError(f"{fname}: Dirichlet conditions must have the form {uname}[...] == value")
            vs = _sympy_of(val, fname)
            if P.t is not None and vs.has(sp.Symbol(P.t.name)):
                raise NotImplementedError(f"{fname}: time-dependent Dirichlet values are not supported")
            for idx, nrm, pt in bnodes:
                if idx in done or not _pred_true(c.args[1], xs, pt, fname):
                    continue
                v = _point_value(vs, P.svars, pt, fname)
                if eigen and abs(v) > 0:
                    raise ValueError(f"{fname}: eigensystems with inhomogeneous Dirichlet conditions cannot be "
                                     "solved")
                A[idx] = 0
                A[idx, idx] = 1
                rhs[idx] = v
                if eigen:
                    B[idx] = 0
                done.add(idx)
    # periodic conditions and two-point equations (1-D)
    # value conditions first, so a derivative two-point condition can be recognised as part of a periodic pair
    for c in sorted(P.extra, key=_has_derivative):
        if _head_is(c, "PeriodicBoundaryCondition") or _head_is(c, "Equal"):
            if P.region.dim != 1:
                raise NotImplementedError(f"{fname}: periodic and pointwise conditions are only supported in 1-D")
            _point_condition(P, Dz, c, A, B, rhs, done, eigen)
    # natural / Neumann rows
    for idx, nrm, pt in bnodes:
        if idx in done:
            continue
        row, c0 = Dz.flux_row(idx, nrm)
        for s_, val, pred in P.nv:
            if not _pred_true(pred, xs, pt, fname):
                continue
            q, g = _nv_parts(P, val)
            sv = complex(sp.N(S(s_)))
            row[idx] += sv * _point_value(q, P.svars, pt, fname)
            c0 += sv * _point_value(g, P.svars, pt, fname)
        if not np.any(np.abs(row) > 0):
            raise ValueError(f"{fname}: the flux vanishes on the boundary at {pt}; give a DirichletCondition there")
        if eigen and abs(c0) > 0:
            raise ValueError(f"{fname}: eigensystems with inhomogeneous Neumann values cannot be solved")
        A[idx] = row
        rhs[idx] = -c0
        if eigen:
            B[idx] = 0
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{fname}: the coefficients are singular on the region (at a collocation point)")
    return A, B, rhs


def _nv_parts(P, val):
    """NeumannValue value g - q u -> (q-like coefficient of u, constant g) with value = q u + g."""
    fname = P.fname
    vs = _sympy_of(val, fname)
    tv = [sp.Symbol(P.t.name)] if P.t is not None else []
    cs, c0 = _lin_form(vs, P.uname, tv + P.svars, fname, "NeumannValue")
    q = sp.Integer(0)
    for k, c in cs.items():
        if any(k):
            raise NotImplementedError(f"{fname}: NeumannValue may only involve {P.uname} itself, not its "
                                      "derivatives")
        q = c
    if P.t is not None and (q.has(tv[0]) or c0.has(tv[0])):
        raise NotImplementedError(f"{fname}: time-dependent Neumann values are not supported")
    return q, c0


def _has_derivative(e) -> bool:
    if isinstance(e, Expr):
        return e.head == "Derivative" or _has_derivative(e.head) or any(_has_derivative(a) for a in e.args)
    return False


def _point_condition(P, Dz, c, A, B, rhs, done, eigen):
    fname = P.fname
    a, b = P.region.lo[0], P.region.hi[0]
    fa, fb = P.region.flo[0], P.region.fhi[0]
    N = Dz.N
    idx_of = {}

    def which(pt):
        z = complex(sp.N(pt))
        if abs(z.imag) > 0:
            raise ValueError(f"{fname}: conditions must be at real points")
        for i_, f_ in ((0, fa), (N, fb)):
            if abs(z.real - f_) <= 1e-12 * max(1.0, abs(f_)):
                return i_
        raise NotImplementedError(f"{fname}: conditions are only supported at the ends of the interval")

    def flux_full(i_):
        row, c0 = Dz.flux_row(i_, (1.0,))
        return row, c0
    if _head_is(c, "PeriodicBoundaryCondition"):
        if len(c.args) != 3:
            raise ValueError(f"{fname}: PeriodicBoundaryCondition[k u[x], pred, f] expected")
        expr, pred, f = c.args
        uf = sp.Function(P.uname)(*P.svars)
        k = sp.simplify(_sympy_of(expr, fname) / uf)
        if k.has(P.uname) or k.free_symbols & set(P.svars) or (P.t is not None and k.has(sp.Symbol(P.t.name))):
            raise NotImplementedError(f"{fname}: PeriodicBoundaryCondition needs a constant multiple of "
                                      f"{P.uname}")
        kk = complex(sp.N(k))
        targets = [(i_, p) for i_, p in ((0, a), (N, b)) if _pred_true(pred, P.region.xs, [p], fname)]
        if len(targets) != 1:
            raise ValueError(f"{fname}: the PeriodicBoundaryCondition predicate must select one end point")
        ti, tp = targets[0]
        si = which(S(ev(Expr(f, tp))))
        if si == ti:
            raise ValueError(f"{fname}: the periodic map must send one end point to the other")
        rows = [(ti, np.eye(Dz.size, dtype=complex)[ti] - kk * np.eye(Dz.size, dtype=complex)[si])]
        ft, _ = flux_full(ti)
        fs, _ = flux_full(si)
        rows.append((si, ft - kk * fs))
        for i_, r in rows:
            if i_ in done:
                raise ValueError(f"{fname}: conflicting conditions at an end point")
            A[i_] = r
            rhs[i_] = 0
            if eigen:
                B[i_] = 0
            done.add(i_)
        return
    if len(c.args) != 2:
        raise ValueError(f"{fname}: boundary equations must have two sides")
    e = sp.expand(_sympy_of(c.args[0], fname) - _sympy_of(c.args[1], fname))
    uf = sp.Function(P.uname)
    terms = {}
    reps = {}
    for s_ in e.atoms(sp.Subs):
        if isinstance(s_.expr, sp.Derivative) and s_.expr.expr.func == uf:
            reps[s_] = (s_.expr.derivative_count, which(s_.point[0]))
    inner = set()
    for s_ in reps:
        inner |= s_.expr.atoms(sp.core.function.AppliedUndef) | s_.expr.atoms(sp.Derivative)
    for f_ in e.atoms(sp.core.function.AppliedUndef):
        if f_.func == uf and f_ not in inner:
            if len(f_.args) != 1 or f_.args[0].free_symbols:
                raise NotImplementedError(f"{fname}: conditions must evaluate {P.uname} at end points")
            reps[f_] = (0, which(f_.args[0]))
    for f_ in e.atoms(sp.Derivative):
        if f_.has(uf) and f_ not in inner:
            raise NotImplementedError(f"{fname}: conditions must evaluate {P.uname} at end points")
    syms = {atom: sp.Symbol(f"__B{i}") for i, atom in enumerate(reps)}
    ee = sp.expand(e.xreplace(syms))
    row = np.zeros(Dz.size, dtype=complex)
    pts_used = set()
    coef = {}
    for atom, sy in syms.items():
        cf = ee.coeff(sy)
        if cf.free_symbols & set(syms.values()) or cf.has(uf):
            raise NotImplementedError(f"{fname}: boundary conditions must be linear")
        k, i_ = reps[atom]
        pts_used.add(i_)
        coef[(k, i_)] = coef.get((k, i_), 0) + complex(sp.N(cf))
        Dk = Dz.Dk.get((k,))
        if Dk is None:
            Dk = np.linalg.matrix_power(Dz.Dk[(1,)], k)
        row += complex(sp.N(cf)) * Dk[i_]
    c0 = ee.subs({sy: 0 for sy in syms.values()})
    if c0.free_symbols:
        raise ValueError(f"{fname}: the condition {c} has free symbols")
    c0 = complex(sp.N(c0))
    if not pts_used:
        raise ValueError(f"{fname}: the condition {c} does not involve {P.uname}")
    if len(pts_used) == 2:
        ks = {k for k, _ in coef}
        if len(ks) != 1 or len(coef) != 2 or abs(c0) > 0:
            raise NotImplementedError(f"{fname}: two-point conditions must have the form u[a] == m u[b] or "
                                      f"Derivative[k][u][a] == m Derivative[k][u][b]")
        (k,) = ks
        m = -coef[(k, N)] / coef[(k, 0)]       # D^k u(a) == m D^k u(b)
        if k == 0:
            # u[a] == m u[b] (Wolfram: a periodic-type condition): values and fluxes are identified
            fa_, _ = flux_full(0)
            fb_, _ = flux_full(N)
            for i_, r in ((N, row / row[0]), (0, fa_ - m * fb_)):
                if i_ in done:
                    raise ValueError(f"{fname}: conflicting conditions at an end point")
                A[i_] = r
                rhs[i_] = 0
                if eigen:
                    B[i_] = 0
                done.add(i_)
            done.add(("periodic", round(m.real, 12), round(m.imag, 12)))
            return
        if k == 1 and ("periodic", round(m.real, 12), round(m.imag, 12)) in done:
            return                             # u'[a] == m u'[b] is already part of the periodic identification
        free = [i_ for i_ in (N, 0) if i_ not in done]
        if not free:
            raise ValueError(f"{fname}: too many boundary conditions")
        i_ = free[0]
        A[i_] = row
        rhs[i_] = 0
        if eigen:
            B[i_] = 0
        done.add(i_)
        return
    (i_,) = pts_used
    if i_ in done:
        raise ValueError(f"{fname}: too many conditions at x = {fa if i_ == 0 else fb}")
    if eigen and abs(c0) > 0:
        raise ValueError(f"{fname}: eigensystems with inhomogeneous boundary conditions cannot be solved")
    A[i_] = row
    rhs[i_] = -c0
    if eigen:
        B[i_] = 0
    done.add(i_)


def _weights(P, Dz):
    ws = [_clencurt(Dz.N, a, b) for a, b in zip(P.region.flo, P.region.fhi)]
    return ws[0] if len(ws) == 1 else np.kron(ws[0], ws[1])


def _sorted_eigs(lams):
    order = sorted(range(len(lams)), key=lambda i: abs(lams[i]))
    groups, out = [], []
    for i in order:
        if groups and abs(abs(lams[i]) - abs(lams[groups[-1][0]])) <= 1e-8 * (1 + abs(lams[i])):
            groups[-1].append(i)
        else:
            groups.append([i])
    for g in groups:
        out.extend(sorted(g, key=lambda i: (round(lams[i].imag, 8), round(lams[i].real, 8))))
    return out


_EIG_SEQ = {1: (32, 48, 64, 96, 128, 192, 256), 2: (14, 18, 22, 26, 30)}


def _eigen_solve(P, n, fname):
    """(eigenvalues, nodal eigenvectors, discretisation) with a two-resolution self-check."""
    sl = _scipy_linalg(fname)
    prev, last = None, None
    dim = P.region.dim
    seq = [N for N in _EIG_SEQ[dim] if (N + 1) ** dim >= 4 * n + 8]
    if not seq:
        raise ValueError(f"{fname}: too many eigenvalues requested ({n})")
    diff = np.inf
    for N in seq:
        Dz = _Disc(P, N)
        A, B, _ = _assemble(P, Dz, True)
        with np.errstate(all="ignore"):
            w, V = sl.eig(A, B)
        ok = np.isfinite(w) & (np.abs(w) < 1e12)
        w, V = w[ok], V[:, ok]
        if P.tord == 2:
            r = np.sqrt(w.astype(complex))
            w = np.concatenate([r, -r])
            V = np.concatenate([V, V], axis=1)
        idx = _sorted_eigs(list(w))[:n]
        if len(idx) < n:
            raise ValueError(f"{fname}: only {len(idx)} eigenvalues could be computed")
        lam = w[idx]
        cur = (lam, V[:, idx], Dz)
        if prev is not None:
            diff = float(np.max(np.abs(lam - prev[0]) / (1 + np.abs(lam))))
            if diff <= 1e-9:
                last = cur
                break
        prev = last = cur
    if diff > 1e-6:
        raise ArithmeticError(f"{fname}: self-check failed: the eigenvalues did not converge under grid "
                              f"refinement (relative change {diff:.2e}); the problem may be singular or "
                              "non-smooth")
    return last


def _normalise(P, Dz, v):
    W = _weights(P, Dz)
    nrm = np.sqrt(np.sum(W * np.abs(v) ** 2).real)
    if nrm == 0 or not np.isfinite(nrm):
        raise ArithmeticError(f"{P.fname}: self-check failed: a zero eigenvector was computed")
    v = v / nrm
    big = np.nonzero(np.abs(v) > 1e-3 * np.max(np.abs(v)))[0]
    k = int(big[1] if big.size > 1 and big[0] == 0 else big[0])    # first interior-side significant value
    v = v * (abs(v[k]) / v[k])
    if np.max(np.abs(v.imag)) <= 1e-9 * np.max(np.abs(v)):
        v = v.real
    return v


def _num(z):
    z = complex(z)
    if abs(z.imag) <= 1e-10 * (1 + abs(z)):
        return float(z.real)
    return R(z)


def _ifun_of(P, Dz, vals):
    from .wl_interp import _make, _make_nd
    if P.region.dim == 1:
        xs, D = Dz.grids[0]
        return _make(_ChebFun(xs, D, vals), P.region.flo[0], P.region.fhi[0])
    N = Dz.N
    return _make_nd(_ChebFun2(Dz.grids[0][0], Dz.grids[1][0], np.asarray(vals).reshape(N + 1, N + 1)),
                    list(zip(P.region.flo, P.region.fhi)))


def _eig_setup(op, u, rest, fname):
    pos, opts = split_options(rest, {"Method"}, fname)
    m = opts.get("Method")
    if m is not None and not (isinstance(m, Symbol) and m.name == "Automatic"):
        raise NotImplementedError(f"{fname}: only Method -> Automatic (Chebyshev collocation) is supported")
    if not pos:
        raise ValueError(f"{fname}: the region and the number of eigenvalues are required")
    n = pos[-1]
    if isinstance(n, Symbol) or (isinstance(n, Expr) and not _is_num(n)):
        raise Unevaluated()
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"{fname}: the number of eigenvalues must be a positive integer")
    specs = pos[:-1]
    t = None
    if specs and isinstance(specs[0], Symbol):
        t = specs.pop(0)
    if not specs:
        raise ValueError(f"{fname}: a region {{x, xmin, xmax}} is required")
    region = _region_of(specs, fname)
    if isinstance(u, Expr) and u.args and not is_list(u):
        want = ([t] if t is not None else []) + list(region.xs)
        if len(u.args) != len(want) or any(not K.same(a, b) for a, b in zip(u.args, want)):
            raise ValueError(f"{fname}: the unknown {u} must be a function of {L(*want)}")
    items = list(op.args) if is_list(op) else [op]
    P = _Problem(fname, items, u, region, t)
    return P, n


def _eig_funcs(P, u, lam, V, Dz):
    fs = []
    for i in range(len(lam)):
        f = _ifun_of(P, Dz, _normalise(P, Dz, V[:, i]))
        if isinstance(u, Expr) and u.args:
            f = Expr(f, *P.region.xs)
        fs.append(f)
    return fs


@fn("NDEigenvalues", area=AREA, status="added",
    notes="NDEigenvalues[op, u, {x, a, b}, n] (also u[x], x in Interval[{a, b}], {x, a, b}, {y, c, d} and {x, y} in "
          "Rectangle[...], and NDEigenvalues[eqns, u, t, region, n] for time-dependent equations, where D[u, t] -> "
          "lambda u and D[u, {t, 2}] -> lambda^2 u) -> the n eigenvalues of smallest magnitude of a linear operator "
          "of order at most two (constant or variable coefficients), sorted by increasing absolute value; op may "
          "contain Inactive[Div] flux terms and NeumannValue terms, and be given with DirichletCondition, "
          "PeriodicBoundaryCondition or periodic two-point equations u[a] == u[b]; no condition = natural "
          "(Neumann 0) boundary; Chebyshev spectral collocation (SciPy generalized eigensolver) instead of finite "
          "elements, so values agree with Wolfram's to within Wolfram's own discretisation error. Disks and "
          "general regions, coupled systems, 3-D and Method settings raise NotImplementedError; inhomogeneous "
          "conditions raise ValueError. Check: self-check — the eigenvalues are recomputed on a finer grid and an "
          "ArithmeticError is raised unless they agree to 1e-6 (relative).")
def _nd_eigenvalues(op: Expr, u: Expr, *rest: Expr) -> ListOf[Expr]:
    fname = "NDEigenvalues"
    P, n = _eig_setup(op, u, rest, fname)
    lam, V, Dz = _eigen_solve(P, n, fname)
    return L(*(_num(z) for z in lam))


@fn("NDEigensystem", area=AREA, status="added",
    notes="NDEigensystem[op, u, region, n] -> {{lambda1, ...}, {u1, ...}} with the eigenvalues as for NDEigenvalues "
          "and the eigenfunctions as InterpolatingFunction objects (ifun[x] when u is given as u[x]; "
          "two-dimensional InterpolatingFunction[{{a, b}, {c, d}}, ...] on rectangles) normalised to "
          "Integrate[Abs[u]^2] == 1 over the region (Clenshaw-Curtis quadrature) with the phase fixed so that the "
          "largest nodal value is positive (Wolfram leaves the sign arbitrary). Check: self-check — eigenvalues are "
          "compared across two grid resolutions (ArithmeticError on disagreement); tests also verify op[u_i] == "
          "lambda_i u_i at sample points and the analytic spectra of Laplacians.")
def _nd_eigensystem(op: Expr, u: Expr, *rest: Expr) -> ListOf[Expr]:
    fname = "NDEigensystem"
    P, n = _eig_setup(op, u, rest, fname)
    lam, V, Dz = _eigen_solve(P, n, fname)
    return L(L(*(_num(z) for z in lam)), L(*_eig_funcs(P, u, lam, V, Dz)))


# ── boundary-value problems for NDSolve / NDSolveValue ──

_BVP_SEQ = {1: (16, 24, 32, 48, 64, 96, 128, 192, 256), 2: (12, 16, 20, 24, 28, 32)}


def _bvp_solve(P, fname):
    prev, diff = None, np.inf
    out = None
    for N in _BVP_SEQ[P.region.dim]:
        Dz = _Disc(P, N)
        A, _, rhs = _assemble(P, Dz, False)
        try:
            cond = np.linalg.cond(A)
        except np.linalg.LinAlgError:
            cond = np.inf
        if not np.isfinite(cond) or cond > 1e15:
            raise ValueError(f"{fname}: the boundary-value problem has no unique solution (singular system; "
                             "add a Dirichlet condition)")
        sol = np.linalg.solve(A, rhs)
        if not np.all(np.isfinite(sol)):
            raise ValueError(f"{fname}: the boundary-value problem could not be solved")
        if prev is not None:
            pD, psol = prev
            if P.region.dim == 1:
                f = _ChebFun(Dz.grids[0][0], Dz.grids[0][1], sol)
                newv = np.array([f(t) for t in pD.grids[0][0]])
            else:
                f = _ChebFun2(Dz.grids[0][0], Dz.grids[1][0], sol.reshape(N + 1, N + 1))
                newv = np.array([f(x, y) for x, y in zip(*pD.pts)])
            diff = float(np.max(np.abs(newv - psol)) / (1 + np.max(np.abs(sol))))
            out = (Dz, sol)
            if diff <= 1e-11:
                break
        prev = out = (Dz, sol)
    if diff > 1e-7:
        raise ArithmeticError(f"{fname}: self-check failed: the boundary-value solution did not converge under "
                              f"grid refinement (relative change {diff:.2e}); non-smooth data is not supported")
    Dz, sol = out
    if np.max(np.abs(sol.imag)) <= 1e-12 * (1 + np.max(np.abs(sol))):
        sol = sol.real
    return Dz, sol


def _has_bc_heads(e):
    return any(_contains(e, h) for h in ("DirichletCondition", "NeumannValue", "PeriodicBoundaryCondition"))


def _two_point(items, uname, xname):
    """True when plain conditions evaluate the unknown at two or more distinct points."""
    pts = set()

    def walk(q):
        if isinstance(q, Expr):
            if (q.head == uname or (isinstance(q.head, Expr) and isinstance(q.head.head, Expr)
                                    and q.head.head.head == "Derivative" and q.head.args
                                    and isinstance(q.head.args[0], Symbol) and q.head.args[0].name == uname)) \
                    and len(q.args) == 1 and not _contains(q.args[0], xname):
                pts.add(str(q.args[0]))
            if isinstance(q.head, Expr):
                walk(q.head)
            for a in q.args:
                walk(a)
    for it in items:
        walk(it)
    return len(pts) >= 2


def ndsolve_bvp(eqs, unknowns, ranges, fname):
    """Hook used by wl_interp's NDSolve: (unknowns, {name: InterpolatingFunction}, var) for linear stationary
    boundary-value problems on intervals and rectangles, or None when the problem is not one."""
    items = list(eqs.args) if is_list(eqs) else [eqs]
    us = list(unknowns.args) if is_list(unknowns) else [unknowns]
    try:
        uname = _uname(us[0], fname)
    except (ValueError, NotImplementedError):
        return None
    if not ranges:
        return None
    xname = None
    r0 = ranges[0]
    if is_list(r0) and r0.args and isinstance(r0.args[0], Symbol):
        xname = r0.args[0].name
    elif _head_is(r0, "Element"):
        v = r0.args[0]
        xname = (v.args[0] if is_list(v) else v).name if isinstance(v.args[0] if is_list(v) else v, Symbol) else None
    is_bvp = any(_has_bc_heads(e) for e in items) or len(ranges) > 1 or _head_is(r0, "Element") or (
        xname is not None and _two_point(items, uname, xname))
    if not is_bvp:
        return None
    if len(us) != 1:
        raise NotImplementedError(f"{fname}: boundary-value problems for coupled systems are not supported")
    region = _region_of(list(ranges), fname)
    u = us[0]
    if isinstance(u, Expr) and u.args:
        if len(u.args) != region.dim or any(not K.same(a, b) for a, b in zip(u.args, region.xs)):
            raise NotImplementedError(f"{fname}: time-dependent PDEs (method of lines) are not supported; the "
                                      f"unknown must be a function of {L(*region.xs)}")
    P = _Problem(fname, items, u, region)
    Dz, sol = _bvp_solve(P, fname)
    return us, {uname: _ifun_of(P, Dz, sol)}, region.xs[0].name


# ───────────────────────────── ParametricNDSolve ─────────────────────────────

def _par_specs(pars, fname):
    """{p1, p2, ...} or {{p, pmin, pmax}, ...} -> [(Symbol, lo, hi)]."""
    items = list(pars.args) if is_list(pars) else [pars]
    out = []
    for it in items:
        if isinstance(it, Symbol):
            out.append((it, None, None))
        elif is_list(it) and len(it.args) == 3 and isinstance(it.args[0], Symbol):
            lo, hi = (_real(v, fname) for v in it.args[1:])
            if not lo <= hi:
                raise ValueError(f"{fname}: parameter ranges must have pmin <= pmax")
            out.append((it.args[0], lo, hi))
        else:
            raise ValueError(f"{fname}: parameters must be symbols or {{p, pmin, pmax}} (got {it})")
    if not out:
        raise ValueError(f"{fname}: at least one parameter is required")
    names = [p[0].name for p in out]
    if len(set(names)) != len(names):
        raise ValueError(f"{fname}: the parameters must be distinct")
    return out


def _pf(specs, call):
    plist = L(*(p if lo is None else L(p, lo, hi) for p, lo, hi in specs))
    return Expr("ParametricFunction", plist, Expr("Hold", call))


def _pnd_args(eqs, unknowns, rest, fname):
    rest = list(rest)
    k = len(rest)
    while k and isinstance(rest[k - 1], Expr) and rest[k - 1].head in ("Rule", "RuleDelayed"):
        k -= 1
    pos, opts = rest[:k], rest[k:]
    if len(pos) < 2:
        raise ValueError(f"{fname}: the range and the parameters are required")
    specs = _par_specs(pos[-1], fname)
    ranges = pos[:-1]
    names = {p.name for p, _, _ in specs}
    for r in ranges:
        if is_list(r) and r.args and isinstance(r.args[0], Symbol) and r.args[0].name in names:
            raise ValueError(f"{fname}: a parameter cannot also be an independent variable")
    return specs, ranges, opts


@fn("ParametricNDSolveValue", area=AREA, status="added",
    notes="ParametricNDSolveValue[eqns, u, {x, xmin, xmax}, {p1, p2, ...}] (parameters may be {p, pmin, pmax}) -> "
          "ParametricFunction[{p1, ...}, Hold[NDSolveValue[eqns, u, {x, xmin, xmax}, opts]]]; pf[v1, ...] substitutes "
          "the numerical parameter values and calls NDSolveValue (initial-value problems with WhenEvent and step "
          "options, boundary-value problems), so pf[1.][t] or pf[1.][t0] evaluate the solution; symbolic parameter "
          "values stay unevaluated, values outside {pmin, pmax} or the wrong number of values raise ValueError; "
          "parameter sensitivities (D with respect to a parameter) are not supported. Check: delegated (NDSolveValue "
          "with the parameters substituted; tests compare with closed-form solutions).")
def _pndsolvevalue(eqs: Expr, unknowns: Expr, *rest: Expr) -> Expr:
    fname = "ParametricNDSolveValue"
    specs, ranges, opts = _pnd_args(eqs, unknowns, rest, fname)
    return _pf(specs, Expr("NDSolveValue", eqs, unknowns, *ranges, *opts))


@fn("ParametricNDSolve", area=AREA, status="added",
    notes="ParametricNDSolve[eqns, u, {x, xmin, xmax}, pars] -> {u -> ParametricFunction[...]} (one rule per unknown "
          "for systems {u, v}); (u /. sol)[p][x] evaluates NDSolveValue for that unknown with the parameter values "
          "substituted; same forms and errors as ParametricNDSolveValue. Check: delegated (NDSolveValue; tests compare "
          "with closed-form solutions).")
def _pndsolve(eqs: Expr, unknowns: Expr, *rest: Expr) -> ListOf[Expr]:
    fname = "ParametricNDSolve"
    specs, ranges, opts = _pnd_args(eqs, unknowns, rest, fname)
    us = list(unknowns.args) if is_list(unknowns) else [unknowns]
    return L(*(Expr("Rule", u, _pf(specs, Expr("NDSolveValue", eqs, u, *ranges, *opts))) for u in us))


def _pf_parts(head):
    if len(head.args) != 2 or not _head_is(head.args[1], "Hold") or not is_list(head.args[0]):
        raise ValueError("ParametricFunction: ParametricFunction[{p, ...}, Hold[call]] expected")
    specs = []
    for p in head.args[0].args:
        if isinstance(p, Symbol):
            specs.append((p, None, None))
        else:
            specs.append((p.args[0], _real(p.args[1], "ParametricFunction"), _real(p.args[2], "ParametricFunction")))
    return specs, head.args[1].args[0]


def _pf_hook(head, args, ev_):
    specs, call = _pf_parts(head)
    if len(args) == 1 and args[0] == "Parameters":
        return L(*(p for p, _, _ in specs))
    if len(args) != len(specs):
        raise ValueError(f"ParametricFunction: {len(specs)} parameter value(s) expected, got {len(args)}")
    vals = []
    for a, (p, lo, hi) in zip(args, specs):
        try:
            z = complex(sp.N(S(a)))
        except Exception:
            return K.UNHANDLED
        if lo is not None and (z.imag != 0 or not lo - 1e-12 <= z.real <= hi + 1e-12):
            raise ValueError(f"ParametricFunction: the value {a} of {p} is outside the range [{lo}, {hi}]")
        vals.append(a)
    rules = L(*(Expr("Rule", p, v) for (p, _, _), v in zip(specs, vals)))
    held = ev_(Expr("ReplaceAll", Expr("Hold", call), rules))
    return ev_(held.args[0])


KL.COMPOUND_HEAD_HOOKS["ParametricFunction"] = _pf_hook

record("ParametricFunction", AREA, "added",
       "ParametricFunction[{p, ...}, Hold[NDSolveValue[...]]] from ParametricNDSolve / ParametricNDSolveValue: "
       "pf[v, ...] solves with the numerical values substituted (symbolic values stay unevaluated, wrong counts and "
       "out-of-range values raise ValueError), pf[\"Parameters\"] lists the parameters. Check: delegated "
       "(NDSolveValue).")


# ───────────────────────────── inert symbols and options ─────────────────────────────

def _hold_all(*names):
    from ..evaluate import set_hold_all
    for n in names:
        set_hold_all(n)


_hold_all("WhenEvent")

for _name, _note in (
        ("DirichletCondition",
         "DirichletCondition[u[x, ...] == value, pred] (inert): consumed by NDSolve / NDSolveValue boundary-value "
         "problems and NDEigensystem / NDEigenvalues on intervals and rectangles (value fixed at the boundary nodes "
         "where pred is True; True = whole boundary); produced by the *Condition PDE boundary heads. Check: tests "
         "solve problems with known closed forms."),
        ("NeumannValue",
         "NeumannValue[g - q u, pred] (inert): in NDSolve / NDEigensystem a term f + NeumannValue[...] on the "
         "right-hand side sets the outward flux n.(c Grad u + alpha u - gamma) to g - q u where pred holds (on the "
         "left-hand side the sign flips); no condition = NeumannValue[0, ...]; produced by the *Value PDE boundary "
         "heads. Check: reference (Wolfram NDEigensystem example -Laplacian + NeumannValue[u, x == 1] -> 0.740174) "
         "in tests."),
        ("PeriodicBoundaryCondition",
         "PeriodicBoundaryCondition[k u[x], pred, f] (inert): in 1-D NDEigensystem / NDSolve boundary-value "
         "problems identifies u at the end point where pred holds with k u at the other end f[x] (and the fluxes "
         "likewise); 2-D periodic conditions raise NotImplementedError. Check: reference (Bloch spectrum "
         "(n + alpha/(2 Pi))^2 for k = Exp[I alpha]) in tests."),
        ("NeumannBoundaryUnitNormal",
         "NeumannBoundaryUnitNormal[{x, ...}] (inert symbolic outward unit normal) appears in the values returned "
         "by the PDE boundary heads (e.g. MassFluxValue, HeatSymmetryValue) unless \"BoundaryUnitNormal\" is given; "
         "the solvers here require explicit normals or scalar fluxes. Check: reference (Wolfram documentation "
         "outputs of the *Value heads) in tests."),
        ("WhenEvent",
         "WhenEvent[event, action] (HoldAll, inert) inside NDSolve / NDSolveValue initial-value problems: events "
         "a == b (either crossing) or a > b / a < b (becoming True) located by SciPy root finding; actions "
         "\"StopIntegration\" (the solution domain ends there), \"RemoveEvent\", resets y[t] -> value / y'[t] -> "
         "value, lists of actions and any expression evaluated for side effects (Sow, counters) with the current "
         "state substituted; discrete variables, event options and integration in both directions from an "
         "interior initial point raise NotImplementedError. Check: reference (bouncing ball impact time Sqrt[2 h/g] "
         "and restitution) in tests."),
        ("AccuracyGoal",
         "AccuracyGoal -> a (option): honoured by NDSolve / NDSolveValue (absolute tolerance 10^-a, Infinity -> "
         "pure relative control); accepted by the existing NIntegrate / FindRoot / NMinimize implementations. "
         "Check: numeric cross-check in tests (looser goals take fewer NDSolve steps)."),
        ("PrecisionGoal",
         "PrecisionGoal -> p (option): honoured by NDSolve / NDSolveValue (relative tolerance max(10^-p, 2.3*10^-14)) "
         "; accepted by NIntegrate / FindRoot. Check: numeric cross-check in tests (looser goals take fewer "
         "NDSolve steps)."),
        ("WorkingPrecision",
         "WorkingPrecision -> MachinePrecision (option): NDSolve and the W26 solvers compute in machine precision "
         "(other values raise NotImplementedError); higher precision is honoured by NIntegrate / FindRoot / N where "
         "implemented. Check: reference in tests."),
        ("MaxIterations",
         "MaxIterations -> n (option): an iteration cap honoured by MandelbrotSetIterationCount, "
         "MandelbrotSetMemberQ, MandelbrotSetDistance and BayesianMinimization; NDSolve has no iterations (use "
         "MaxSteps) and rejects it. Check: reference in tests."),
        ("MaxStepFraction",
         "MaxStepFraction -> f (option, 0 < f <= 1, default 1/10 as in Wolfram; Infinity = no limit): NDSolve "
         "limits each step to f times the integration range. Check: numeric cross-check in tests (StepMonitor "
         "step spacing)."),
        ("MaxStepSize",
         "MaxStepSize -> h (option): NDSolve limits each step to h (SciPy max_step); Infinity / Automatic = no limit. "
         "Check: numeric cross-check in tests (StepMonitor step spacing)."),
        ("MaxSteps",
         "MaxSteps -> n (option): NDSolve raises ValueError when more than n steps would be needed (instead of "
         "Wolfram's truncated solution with NDSolve::mxst); Infinity / Automatic = no limit. Check: reference in "
         "tests."),
        ("StartingStepSize",
         "StartingStepSize -> h (option): NDSolve's first step size (SciPy first_step); larger than the range raises "
         "ValueError. Check: numeric cross-check in tests."),
        ("Tolerance",
         "Tolerance -> t (option): honoured by SingularValueList (singular values below t times the largest are "
         "dropped) and the other linear-algebra heads that document it. Check: reference in tests."),
        ("EvaluationMonitor",
         "EvaluationMonitor :> expr (option): evaluated at every function evaluation by NDSolve (with x and y[x] "
         "replaced by the current values); FindRoot / NIntegrate / NSolve raise NotImplementedError for it rather "
         "than ignore it. Check: reference in tests."),
        ("StepMonitor",
         "StepMonitor :> expr (option): evaluated after every accepted step of NDSolve (x, y[x], y'[x] replaced by "
         "the step values; the monitor runs once the integration finishes); FindRoot / NIntegrate / NSolve raise "
         "NotImplementedError for it rather than ignore it. Check: reference in tests."),
        ("VerifyConvergence",
         "VerifyConvergence -> True | False (option): accepted by NSum, which raises ValueError for a divergent "
         "sum such as Sum[1/n]. Check: reference in tests."),
):
    record(_name, AREA, "added", _note)
