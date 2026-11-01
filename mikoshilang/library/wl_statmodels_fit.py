"""Wolfram batch W18 (part 2): model fitting -- DesignMatrix, LinearModelFit, NonlinearModelFit,
GeneralizedLinearModelFit, LogitModelFit, ProbitModelFit and the FittedModel object; survival models
(SurvivalModelFit / SurvivalModel, CoxModelFit / CoxModel) and clustering (FindClusters, ClusteringComponents,
ClusteringTree, ClusteringMeasurements).

Fitted models are FittedModel[<|...|>] / SurvivalModel[<|...|>] / CoxModel[<|...|>] expressions that carry their
data, so every property is recomputed from the expression (no hidden state).  Numerical work uses NumPy and
mpmath only (no SciPy).  Formatted properties ("ParameterTable", "ANOVATable", ...) raise NotImplementedError.
"""
from __future__ import annotations

import math

import mpmath
import numpy as np
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import Unevaluated, fn, R, L, rule, S, is_list, has_float
from . import wl_statmodels as SM

AREA = "statistical models (W18)"
UNH = K.UNHANDLED
ev = SM.ev
opts = SM.opts


# ─────────────────────────── association helpers ───────────────────────────

def _assoc(**kw):
    return Expr("Association", *(rule(k, v) for k, v in kw.items()))


def _get(a, key):
    for r in a.args:
        if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and r.args[0] == key:
            return r.args[1]
    raise KeyError(key)


def _model_assoc(head, x):
    if isinstance(x, Expr) and x.head == head and len(x.args) == 1 and isinstance(x.args[0], Expr) and \
            x.args[0].head == "Association":
        return x.args[0]
    return None


def _np(m):
    """Mikoshi list / matrix -> float numpy array"""
    if is_list(m) and m.args and is_list(m.args[0]):
        return np.array([[SM.fl(S(v)) for v in r.args] for r in m.args], dtype=float)
    return np.array([SM.fl(S(v)) for v in m.args], dtype=float)


def _L(a):
    a = np.asarray(a)
    if a.ndim == 2:
        return L(*(L(*(float(v) for v in r)) for r in a))
    return L(*(float(v) for v in a))


def _syms(vars_, fname):
    vs = list(vars_.args) if is_list(vars_) else [vars_]
    if not all(isinstance(v, Symbol) for v in vs):
        raise TypeError(f"{fname}: the variables must be symbols")
    return vs


def _funs(f):
    return list(f.args) if is_list(f) else [f]


def _data_xy(data, nvars, fname):
    """(X rows of variable values, y) from {y1, ...} or {{x1, ..., y1}, ...}"""
    if isinstance(data, Symbol):
        raise Unevaluated()
    if not is_list(data) or not data.args:
        raise TypeError(f"{fname}: the data must be a non-empty list")
    if all(not is_list(r) for r in data.args):
        if nvars != 1:
            raise ValueError(f"{fname}: a data vector needs exactly one variable")
        ys = [SM._num(v, fname, "response") for v in data.args]
        return [[sp.Integer(i + 1)] for i in range(len(ys))], ys
    if not SM.is_matrix(data):
        raise TypeError(f"{fname}: the data must be a vector or a rectangular matrix")
    rows = SM.mat(data, fname)
    if len(rows[0]) != nvars + 1:
        raise ValueError(f"{fname}: each data point needs {nvars} variable value(s) and a response")
    return [r[:-1] for r in rows], [r[-1] for r in rows]


def _subst(f, vs, row):
    """numeric value (sympy) of the Mikoshi expression f at the variable values row"""
    v = ev(Expr("ReplaceAll", f, L(*(rule(s, R(x)) for s, x in zip(vs, row)))))
    sv = S(v)
    if not (isinstance(sv, sp.Basic) and sv.is_number):
        raise ValueError(f"basis function {f} is not numeric at {row}")
    return sv


def _design(data, funs, vars_, fname, const=True):
    vs = _syms(vars_, fname)
    X, y = _data_xy(data, len(vs), fname)
    fs = _funs(funs)
    rows = []
    for xr in X:
        try:
            r = [_subst(f, vs, xr) for f in fs]
        except ValueError as e:
            raise ValueError(f"{fname}: {e}")
        rows.append(([sp.Integer(1)] if const else []) + r)
    basis = ([1] if const else []) + fs
    return rows, y, basis, vs


def _const_opt(o, fname):
    v = o.get("IncludeConstantBasis", True)
    if v not in (True, False):
        raise ValueError(f"{fname}: IncludeConstantBasis must be True or False")
    return v


# ─────────────────────────── DesignMatrix ───────────────────────────

@fn("DesignMatrix", area=AREA, status="added",
    notes="DesignMatrix[{{x11, ..., y1}, ...}, {f1, f2, ...}, {x1, ...}] gives the rows {1, f1, f2, ...} at the data "
          "points (a data vector {y1, ...} uses x = 1, 2, ...); IncludeConstantBasis -> False drops the constant "
          "column; exact for exact data. Nominal variables and non-numeric basis values raise ValueError / "
          "NotImplementedError; symbolic data stays unevaluated. Check: the design matrix times the fitted "
          "parameters reproduces LinearModelFit's predicted response (tested).")
def _design_matrix(data: MatrixOf[Real] | ListOf[Real], funs: Expr | ListOf[Expr], vars_: Expr | ListOf[Expr],
                   *rest: Option) -> MatrixOf[Real]:
    name = "DesignMatrix"
    _, o = opts(rest, ("IncludeConstantBasis", "NominalVariables"), name)
    if "NominalVariables" in o and SM.str_or_sym(o["NominalVariables"]) not in ("None", "Automatic"):
        raise NotImplementedError(f"{name}: nominal variables are not supported")
    rows, _, _, _ = _design(data, funs, vars_, name, _const_opt(o, name))
    items = list(data.args)
    return L(*(L(*(SM.out(v, items) for v in r)) for r in rows))


# ─────────────────────────── linear models ───────────────────────────

FIT_OPTS = ("IncludeConstantBasis", "Weights", "ConfidenceLevel", "NominalVariables", "WorkingPrecision",
            "VarianceEstimatorFunction", "Method")


def _weights(o, n, fname):
    w = o.get("Weights")
    if w is None or SM.str_or_sym(w) == "Automatic":
        return None
    if not is_list(w) or len(w.args) != n:
        raise ValueError(f"{fname}: Weights must be a list with one weight per data point")
    ws = [SM._num(v, fname, "weight") for v in w.args]
    if any(v < 0 for v in ws):
        raise ValueError(f"{fname}: weights must be non-negative")
    return ws


def _bestfit(params, basis):
    return ev(Expr("Plus", *(Expr("Times", R(p) if isinstance(p, sp.Basic) else float(p), f)
                             for p, f in zip(params, basis))))


@fn("LinearModelFit", area=AREA, status="added",
    notes="LinearModelFit[data, {f1, ...}, {x1, ...}] (and [data, f, x], [{m, v}] with a design matrix m and "
          "response v) fits y = b0 + Sum[bi fi] by (weighted) least squares and returns a FittedModel; exact data "
          "give exact parameters (rational linear algebra), inexact data machine numbers. Options "
          "IncludeConstantBasis, Weights, ConfidenceLevel. Properties are listed under FittedModel. Rank-deficient "
          "designs and fewer points than parameters raise ValueError; NominalVariables and "
          "VarianceEstimatorFunction settings NotImplementedError. Check: reference values from statsmodels OLS / "
          "WLS (params, bse, rsquared, aic, bic) and the Wolfram documentation.")
def _lmf(data: MatrixOf[Real] | ListOf[Real] | ListOf[Expr], funs: Expr | ListOf[Expr] = None,
         vars_: Expr | ListOf[Expr] = None, *rest: Option) -> Expr:
    name = "LinearModelFit"
    rest = list(rest)
    if funs is not None and SM._is_rule(funs):
        rest.insert(0, funs)
        funs = None
    if vars_ is not None and SM._is_rule(vars_):
        rest.insert(0, vars_)
        vars_ = None
    _, o = opts(rest, FIT_OPTS, name)
    for k in ("NominalVariables", "VarianceEstimatorFunction"):
        if k in o and SM.str_or_sym(o[k]) not in ("None", "Automatic"):
            raise NotImplementedError(f"{name}: the {k} option is not supported")
    if isinstance(data, Symbol):
        raise Unevaluated()
    if funs is None:
        if not (is_list(data) and len(data.args) == 2 and SM.is_matrix(data.args[0]) and is_list(data.args[1])):
            raise TypeError(f"{name}: expected LinearModelFit[data, funs, vars] or LinearModelFit[{{m, v}}]")
        m = SM.mat(data.args[0], name)
        y = SM.vec(data.args[1], name, "response")[0]
        if len(m) != len(y):
            raise ValueError(f"{name}: the design matrix and response must have the same number of rows")
        rows = [list(r) for r in m]
        vs = []
        basis_expr = [Expr("Slot", i + 1) for i in range(len(rows[0]))]
    else:
        if vars_ is None:
            raise TypeError(f"{name}: the variables must be given")
        rows, y, basis_expr, vs = _design(data, funs, vars_, name, _const_opt(o, name))
    n, p = len(rows), len(rows[0])
    if n < p:
        raise ValueError(f"{name}: fewer data points ({n}) than parameters ({p})")
    w = _weights(o, n, name)
    inexact = has_float(data) or (w is not None and any(v.is_Float for v in w))
    if inexact:
        A = np.array([[SM.fl(v) for v in r] for r in rows])
        yy = np.array([SM.fl(v) for v in y])
        ww = np.ones(n) if w is None else np.array([SM.fl(v) for v in w])
        sw = np.sqrt(ww)
        if np.linalg.matrix_rank(A * sw[:, None]) < p:
            raise ValueError(f"{name}: the design matrix is rank deficient")
        beta = np.linalg.lstsq(A * sw[:, None], yy * sw, rcond=None)[0]
        params = [float(b) for b in beta]
    else:
        A = sp.Matrix(rows)
        Wm = sp.diag(*(w if w is not None else [1] * n))
        M = A.T * Wm * A
        if M.rank() < p:
            raise ValueError(f"{name}: the design matrix is rank deficient")
        beta = M.LUsolve(A.T * Wm * sp.Matrix(y))
        params = [sp.simplify(b) for b in beta]
    cl = o.get("ConfidenceLevel", sp.Rational(95, 100))
    clv = SM._num(cl, name, "ConfidenceLevel")
    if not 0 < clv < 1:
        raise ValueError(f"{name}: ConfidenceLevel must be between 0 and 1")
    a = _assoc(Type="Linear",
               BestFit=_bestfit(params, basis_expr),
               BestFitParameters=L(*(R(q) if isinstance(q, sp.Basic) else q for q in params)),
               BasisFunctions=L(*basis_expr), Variables=L(*vs),
               DesignMatrix=L(*(L(*(SM.out(v, [data]) for v in r)) for r in rows)),
               Response=L(*(SM.out(v, [data]) for v in y)),
               Weights=L(*([R(v) for v in w] if w is not None else [1] * n)),
               Data=data, ConfidenceLevel=cl)
    return Expr("FittedModel", a)


# ─────────────────────────── nonlinear models ───────────────────────────

def _lm_solve(f, jac, beta0, y, w, maxit=500):
    """Levenberg-Marquardt for min Sum[w (y - f(beta))^2]"""
    beta = np.array(beta0, dtype=float)
    sw = np.sqrt(w)

    def rss(b):
        r = (y - f(b)) * sw
        return float(r @ r)
    lam = 1e-3
    cur = rss(beta)
    for _ in range(maxit):
        J = jac(beta) * sw[:, None]
        r = (y - f(beta)) * sw
        g = J.T @ r
        H = J.T @ J
        improved = False
        for _ in range(60):
            try:
                step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            nb = beta + step
            new = rss(nb)
            if np.isfinite(new) and new <= cur:
                beta, lam, improved = nb, max(lam / 10, 1e-15), True
                done = abs(cur - new) <= 1e-15 * max(1.0, cur) and np.all(np.abs(step) <= 1e-12 * (1 + np.abs(nb)))
                cur = new
                break
            lam *= 10
        if not improved or done:
            break
    return beta, cur


@fn("NonlinearModelFit", area=AREA, status="added",
    notes="NonlinearModelFit[data, form, {b1, ...} | {{b1, b10}, ...}, {x1, ...}] fits the model form by "
          "(weighted) nonlinear least squares (Levenberg-Marquardt from the starting values, default 1; NumPy, no "
          "SciPy) and returns a FittedModel with the same properties as LinearModelFit (standard errors from the "
          "Gauss-Newton covariance EstimatedVariance Inverse[J'.W.J]); machine-precision results. Constraints "
          "({form, cons}) and Method settings raise NotImplementedError; a non-converging fit or non-numeric model "
          "values ArithmeticError / ValueError. Check: reference values from scipy.optimize.curve_fit / "
          "statsmodels on the same data (parameters and standard errors), and the residual normal equations "
          "J'.r = 0 at the optimum (self-check).")
def _nlmf(data: MatrixOf[Real] | ListOf[Real], form: Expr, pars: Expr | ListOf[Expr], vars_: Expr | ListOf[Expr],
          *rest: Option) -> Expr:
    name = "NonlinearModelFit"
    _, o = opts(rest, ("Weights", "ConfidenceLevel", "MaxIterations", "Method", "WorkingPrecision",
                       "AccuracyGoal", "PrecisionGoal", "VarianceEstimatorFunction"), name)
    if "Method" in o and SM.str_or_sym(o["Method"]) != "Automatic":
        raise NotImplementedError(f"{name}: Method settings are not supported")
    if is_list(form):
        raise NotImplementedError(f"{name}: constrained fits {{form, cons}} are not supported")
    if isinstance(data, Symbol):
        raise Unevaluated()
    vs = _syms(vars_, name)
    pl = list(pars.args) if is_list(pars) else [pars]
    names, starts = [], []
    for q in pl:
        if is_list(q) and len(q.args) == 2 and isinstance(q.args[0], Symbol):
            names.append(q.args[0])
            starts.append(SM.fl(SM._num(q.args[1], name, "starting value")))
        elif isinstance(q, Symbol):
            names.append(q)
            starts.append(1.0)
        else:
            raise TypeError(f"{name}: parameters must be symbols or {{symbol, start}}")
    X, y = _data_xy(data, len(vs), name)
    n, p = len(y), len(names)
    if n < p:
        raise ValueError(f"{name}: fewer data points than parameters")
    fx = S(form)
    if not isinstance(fx, sp.Basic):
        raise TypeError(f"{name}: the model must be an algebraic expression")
    bsyms = [sp.Symbol(s.name) for s in names]
    xsyms = [sp.Symbol(s.name) for s in vs]
    fx = fx.subs({sp.Symbol(str(k)): k for k in fx.free_symbols})
    extra = fx.free_symbols - set(bsyms) - set(xsyms)
    if extra:
        raise ValueError(f"{name}: the model has unknown symbols {sorted(map(str, extra))}")
    F = sp.lambdify(bsyms + xsyms, fx, "numpy")
    Js = [sp.lambdify(bsyms + xsyms, sp.diff(fx, b), "numpy") for b in bsyms]
    Xa = np.array([[SM.fl(v) for v in r] for r in X])
    ya = np.array([SM.fl(v) for v in y])
    w = _weights(o, n, name)
    wa = np.ones(n) if w is None else np.array([SM.fl(v) for v in w])

    def f(b):
        return np.broadcast_to(np.asarray(F(*b, *Xa.T), dtype=float), (n,)).astype(float)

    def jac(b):
        return np.column_stack([np.broadcast_to(np.asarray(J(*b, *Xa.T), dtype=float), (n,)) for J in Js])
    with np.errstate(all="ignore"):
        beta, rss = _lm_solve(f, jac, starts, ya, wa)
        if not np.all(np.isfinite(beta)) or not np.isfinite(rss):
            raise ArithmeticError(f"{name}: the fit did not converge")
        J = jac(beta)
        g = J.T @ ((ya - f(beta)) * wa)
        if np.linalg.norm(g) > 1e-6 * max(1.0, np.linalg.norm(J.T @ (ya * wa))):
            raise ArithmeticError(f"{name}: the fit did not converge (gradient {np.linalg.norm(g):.3g})")
    cl = o.get("ConfidenceLevel", sp.Rational(95, 100))
    sub = L(*(rule(s, float(b)) for s, b in zip(names, beta)))
    best = ev(Expr("ReplaceAll", form, sub))
    a = _assoc(Type="Nonlinear", BestFit=best, Model=form,
               BestFitParameters=L(*(rule(s, float(b)) for s, b in zip(names, beta))),
               Parameters=L(*names), Variables=L(*vs),
               DesignMatrix=_L(J), Response=_L(ya), Weights=_L(wa),
               Data=data, ConfidenceLevel=cl)
    return Expr("FittedModel", a)


# ─────────────────────────── generalized linear models ───────────────────────────

_FAMILIES = ("Gaussian", "Binomial", "Poisson", "Gamma", "InverseGaussian")
_CANON = {"Gaussian": "IdentityLink", "Binomial": "LogitLink", "Poisson": "LogLink", "Gamma": "InverseLink",
          "InverseGaussian": "InverseSquareLink"}


def _link(name):
    if name == "IdentityLink":
        return (lambda m: m), (lambda e: e), (lambda m: np.ones_like(m))
    if name == "LogLink":
        return np.log, np.exp, (lambda m: 1 / m)
    if name == "LogitLink":
        return (lambda m: np.log(m / (1 - m))), (lambda e: 1 / (1 + np.exp(-e))), (lambda m: 1 / (m * (1 - m)))
    if name == "ProbitLink":
        from math import erf, sqrt, exp, pi
        ncdf = np.vectorize(lambda e: 0.5 * (1 + erf(e / sqrt(2))))
        ninv = np.vectorize(lambda m: float(mpmath.sqrt(2) * mpmath.erfinv(2 * m - 1)))
        return ninv, ncdf, (lambda m: 1 / (np.exp(-ninv(m) ** 2 / 2) / sqrt(2 * pi)))
    if name == "ComplementaryLogLogLink":
        return (lambda m: np.log(-np.log(1 - m))), (lambda e: 1 - np.exp(-np.exp(e))), \
            (lambda m: 1 / ((m - 1) * np.log(1 - m)))
    if name == "InverseLink":
        return (lambda m: 1 / m), (lambda e: 1 / e), (lambda m: -1 / m ** 2)
    if name == "InverseSquareLink":
        return (lambda m: 1 / m ** 2), (lambda e: 1 / np.sqrt(e)), (lambda m: -2 / m ** 3)
    raise ValueError(f"unknown link function {name!r}")


def _var(fam, m):
    return {"Gaussian": np.ones_like(m), "Binomial": m * (1 - m), "Poisson": m, "Gamma": m ** 2,
            "InverseGaussian": m ** 3}[fam]


def _mean_expr(link, eta):
    """Mikoshi expression of the inverse link applied to the linear predictor eta"""
    if link == "IdentityLink":
        return eta
    if link == "LogLink":
        return Expr("Exp", eta)
    if link == "LogitLink":
        return Expr("Power", Expr("Plus", 1, Expr("Exp", Expr("Times", -1, eta))), -1)
    if link == "ProbitLink":
        return Expr("Times", 0.5,
                    Expr("Erfc", Expr("Times", -1, Expr("Power", 2, -0.5), eta)))
    if link == "ComplementaryLogLogLink":
        return Expr("Plus", 1, Expr("Times", -1, Expr("Exp", Expr("Times", -1, Expr("Exp", eta)))))
    if link == "InverseLink":
        return Expr("Power", eta, -1)
    return Expr("Power", eta, -0.5)


def _glm_fit(name, data, funs, vars_, rest, family=None, link=None):
    _, o = opts(rest, FIT_OPTS + ("ExponentialFamily", "LinkFunction", "LinearOffsetFunction",
                                  "DispersionEstimatorFunction", "CovarianceEstimatorFunction", "MaxIterations"),
                name)
    for k in ("NominalVariables", "DispersionEstimatorFunction", "VarianceEstimatorFunction"):
        if k in o and SM.str_or_sym(o[k]) not in ("None", "Automatic"):
            raise NotImplementedError(f"{name}: the {k} option is not supported")
    if "LinearOffsetFunction" in o and SM.str_or_sym(o["LinearOffsetFunction"]) != "None":
        raise NotImplementedError(f"{name}: LinearOffsetFunction is not supported")
    fam = family or SM.str_or_sym(o.get("ExponentialFamily", "Gaussian"))
    if fam not in _FAMILIES:
        if fam in ("QuasiLikelihood", "Multinomial") or (isinstance(o.get("ExponentialFamily"), Expr)):
            raise NotImplementedError(f"{name}: the {fam} family is not supported")
        raise ValueError(f"{name}: unknown ExponentialFamily {o.get('ExponentialFamily')!r}")
    lk = link or SM.str_or_sym(o.get("LinkFunction", "Automatic"))
    if lk == "Automatic":
        lk = _CANON[fam]
    if lk not in ("IdentityLink", "LogLink", "LogitLink", "ProbitLink", "ComplementaryLogLogLink", "InverseLink",
                  "InverseSquareLink"):
        if isinstance(o.get("LinkFunction"), Expr):
            raise NotImplementedError(f"{name}: custom link functions are not supported")
        raise ValueError(f"{name}: unknown LinkFunction {o.get('LinkFunction')!r}")
    if isinstance(data, Symbol):
        raise Unevaluated()
    rows, y, basis, vs = _design(data, funs, vars_, name, _const_opt(o, name))
    A = np.array([[SM.fl(v) for v in r] for r in rows])
    yy = np.array([SM.fl(v) for v in y])
    n, p = A.shape
    if n < p:
        raise ValueError(f"{name}: fewer data points than parameters")
    w = _weights(o, n, name)
    pw = np.ones(n) if w is None else np.array([SM.fl(v) for v in w])
    if fam == "Binomial" and np.any((yy < 0) | (yy > 1)):
        raise ValueError(f"{name}: Binomial responses must be proportions in [0, 1]")
    if fam == "Poisson" and np.any(yy < 0):
        raise ValueError(f"{name}: Poisson responses must be non-negative")
    if fam in ("Gamma", "InverseGaussian") and np.any(yy <= 0):
        raise ValueError(f"{name}: {fam} responses must be positive")
    g, ginv, gprime = _link(lk)
    mu = (yy + 0.5) / 2 if fam == "Binomial" else (yy + yy.mean()) / 2 if fam != "Gaussian" else yy.copy()
    if fam == "Poisson":
        mu = yy + 0.1
    eta = g(mu)
    beta = np.zeros(p)
    dev_old = np.inf
    with np.errstate(all="ignore"):
        for it in range(200):
            gp = gprime(mu)
            z = eta + (yy - mu) * gp
            W = pw / (gp ** 2 * _var(fam, mu))
            sw = np.sqrt(W)
            beta_new = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)[0]
            eta = A @ beta_new
            mu = ginv(eta)
            if fam == "Binomial":
                mu = np.clip(mu, 1e-15, 1 - 1e-15)
            dev = _deviance(fam, yy, mu, pw)
            if not np.isfinite(dev):
                raise ArithmeticError(f"{name}: the iteration diverged")
            beta = beta_new
            if abs(dev - dev_old) <= 1e-12 * (abs(dev) + 0.1):
                break
            dev_old = dev
        else:
            raise ArithmeticError(f"{name}: the iteration did not converge")
    if np.linalg.matrix_rank(A) < p:
        raise ValueError(f"{name}: the design matrix is rank deficient")
    cl = o.get("ConfidenceLevel", sp.Rational(95, 100))
    params = [float(b) for b in beta]
    eta_expr = _bestfit(params, basis)
    a = _assoc(Type="GLM", BestFit=ev(_mean_expr(lk, eta_expr)), LinearPredictor=eta_expr,
               BestFitParameters=L(*params), BasisFunctions=L(*basis), Variables=L(*vs),
               DesignMatrix=_L(A), Response=_L(yy), Weights=_L(pw), ExponentialFamily=fam, LinkFunction=lk,
               Data=data, ConfidenceLevel=cl)
    return Expr("FittedModel", a)


def _deviance(fam, y, mu, w):
    with np.errstate(all="ignore"):
        if fam == "Gaussian":
            d = (y - mu) ** 2
        elif fam == "Binomial":
            d = 2 * (np.where(y > 0, y * np.log(y / mu), 0) + np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0))
        elif fam == "Poisson":
            d = 2 * (np.where(y > 0, y * np.log(y / mu), 0) - (y - mu))
        elif fam == "Gamma":
            d = 2 * (-np.log(y / mu) + (y - mu) / mu)
        else:
            d = (y - mu) ** 2 / (mu ** 2 * y)
    return float(np.sum(w * d))


_GLM_NOTE = ("Properties are listed under FittedModel (\"BestFit\" is the mean function, \"LinearPredictor\", "
             "\"Deviance\", \"NullDeviance\", \"EstimatedDispersion\" = Pearson chi^2/(n - p) for Gaussian, Gamma and "
             "InverseGaussian, 1 for Binomial and Poisson, \"LogLikelihood\", \"AIC\"/\"BIC\" counting the "
             "dispersion parameter when it is estimated, \"ParameterErrors\", \"ParameterZStatistics\"/"
             "\"ParameterPValues\" (normal for Binomial/Poisson, Student t otherwise), ...). Binomial responses are "
             "proportions in [0, 1] (Weights = numbers of trials). Fitted by iteratively reweighted least squares "
             "(NumPy); NominalVariables, LinearOffsetFunction, custom links and quasi-likelihood families raise "
             "NotImplementedError; invalid responses ValueError; divergence ArithmeticError.")


@fn("GeneralizedLinearModelFit", area=AREA, status="added",
    notes="GeneralizedLinearModelFit[data, {f1, ...}, {x1, ...}] fits a generalized linear model with "
          "ExponentialFamily -> \"Gaussian\" (default), \"Binomial\", \"Poisson\", \"Gamma\" or \"InverseGaussian\" "
          "and LinkFunction -> Automatic (canonical), \"IdentityLink\", \"LogLink\", \"LogitLink\", \"ProbitLink\", "
          "\"ComplementaryLogLogLink\", \"InverseLink\" or \"InverseSquareLink\". " + _GLM_NOTE +
          " Check: reference values from statsmodels GLM (params, bse, deviance, scale, llf) on the same data.")
def _glmf(data: MatrixOf[Real] | ListOf[Real], funs: Expr | ListOf[Expr], vars_: Expr | ListOf[Expr],
          *rest: Option) -> Expr:
    return _glm_fit("GeneralizedLinearModelFit", data, funs, vars_, rest)


@fn("LogitModelFit", area=AREA, status="added",
    notes="LogitModelFit[data, {f1, ...}, {x1, ...}] = GeneralizedLinearModelFit with ExponentialFamily -> "
          "\"Binomial\" and LinkFunction -> \"LogitLink\". " + _GLM_NOTE +
          " Check: reference values from statsmodels Logit / GLM(Binomial) on the same data.")
def _logitf(data: MatrixOf[Real] | ListOf[Real], funs: Expr | ListOf[Expr], vars_: Expr | ListOf[Expr],
            *rest: Option) -> Expr:
    return _glm_fit("LogitModelFit", data, funs, vars_, rest, "Binomial", "LogitLink")


@fn("ProbitModelFit", area=AREA, status="added",
    notes="ProbitModelFit[data, {f1, ...}, {x1, ...}] = GeneralizedLinearModelFit with ExponentialFamily -> "
          "\"Binomial\" and LinkFunction -> \"ProbitLink\". " + _GLM_NOTE +
          " Check: reference values from statsmodels Probit / GLM(Binomial, probit) on the same data.")
def _probitf(data: MatrixOf[Real] | ListOf[Real], funs: Expr | ListOf[Expr], vars_: Expr | ListOf[Expr],
             *rest: Option) -> Expr:
    return _glm_fit("ProbitModelFit", data, funs, vars_, rest, "Binomial", "ProbitLink")


# ─────────────────────────── FittedModel properties ───────────────────────────

def _t_two(t, df):
    return 2 * SM.t_sf(abs(t), df)


def _tq(p, df):
    """Student t quantile by bisection on t_sf (mpmath)"""
    lo, hi = 0.0, 1e3
    target = (1 - p) / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if SM.t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _zq(p):
    return float(mpmath.sqrt(2) * mpmath.erfinv(p))


_FM_PROPS = ("BestFit", "BestFitParameters", "BasisFunctions", "Function", "DesignMatrix", "Response", "Weights",
             "FitResiduals", "PredictedResponse", "RSquared", "AdjustedRSquared", "EstimatedVariance",
             "ParameterErrors", "ParameterTStatistics", "ParameterZStatistics", "ParameterPValues",
             "ParameterConfidenceIntervals", "CovarianceMatrix", "CorrelationMatrix", "AIC", "AICc", "BIC",
             "LogLikelihood", "Deviance", "NullDeviance", "EstimatedDispersion", "LinearPredictor", "Properties",
             "ANOVATableDegreesOfFreedom", "ANOVATableSumsOfSquares", "Data")


def _fm_property(a, prop):
    typ = _get(a, "Type")
    if prop == "Properties":
        return L(*_FM_PROPS)
    if prop.endswith("Table") or prop in ("ParameterTableEntries", "ANOVATableEntries", "SinglePredictionBands",
                                          "MeanPredictionBands", "SinglePredictionConfidenceIntervalTable"):
        raise NotImplementedError(f'FittedModel: the property "{prop}" is not supported')
    if prop in ("BestFit", "BestFitParameters", "DesignMatrix", "Response", "Weights", "LinearPredictor") and \
            not (prop == "LinearPredictor" and typ != "GLM"):
        if prop == "BestFitParameters" and typ == "GLM":
            return _get(a, prop)
        return _get(a, prop)
    if prop == "BasisFunctions":
        return _get(a, "BasisFunctions") if typ != "Nonlinear" else _get(a, "Model")
    if prop == "Function":
        vs = _get(a, "Variables")
        return ev(Expr("Function", vs, _get(a, "BestFit")))
    if prop == "Data":
        return _get(a, "Data")
    exact = typ == "Linear" and not has_float(_get(a, "DesignMatrix")) and not has_float(_get(a, "Response")) and \
        not has_float(_get(a, "Weights"))
    X = _np(_get(a, "DesignMatrix"))
    y = _np(_get(a, "Response"))
    w = _np(_get(a, "Weights"))
    n, p = X.shape
    if typ == "Nonlinear":
        beta = np.array([SM.fl(S(r.args[1])) for r in _get(a, "BestFitParameters").args])
        vs = list(_get(a, "Variables").args)
        best = _get(a, "BestFit")
        data = _get(a, "Data")
        Xv, _ = _data_xy(data, len(vs), "FittedModel")
        pred = np.array([SM.fl(_subst(best, vs, r)) for r in Xv])
    elif typ == "GLM":
        beta = _np(_get(a, "BestFitParameters"))
        fam, lk = _get(a, "ExponentialFamily"), _get(a, "LinkFunction")
        _, ginv, gprime = _link(lk)
        eta = X @ beta
        pred = ginv(eta)
    else:
        beta = _np(_get(a, "BestFitParameters"))
        pred = X @ beta
    res = y - pred
    df = n - p
    cl = SM.fl(SM._num(_get(a, "ConfidenceLevel"), "FittedModel", "ConfidenceLevel"))
    if typ == "GLM":
        fam = _get(a, "ExponentialFamily")
        mu = pred
        gp = gprime(mu)
        Wt = w / (gp ** 2 * _var(fam, mu))
        if fam in ("Binomial", "Poisson"):
            disp = 1.0
        else:
            disp = float(np.sum(w * (y - mu) ** 2 / _var(fam, mu)) / df) if df > 0 else float("nan")
        cov = disp * np.linalg.inv(X.T @ (X * Wt[:, None]))
        dev = _deviance(fam, y, mu, w)
        if prop == "Deviance":
            return dev
        if prop == "NullDeviance":
            ybar = np.sum(w * y) / np.sum(w)
            return _deviance(fam, y, np.full(n, ybar), w)
        if prop == "EstimatedDispersion":
            return disp
        if prop in ("LogLikelihood", "AIC", "BIC", "AICc"):
            ll = _glm_loglik(fam, y, mu, w, dev, n)
            k = p + (0 if fam in ("Binomial", "Poisson") else 1)
            if prop == "LogLikelihood":
                return ll
            if prop == "AIC":
                return -2 * ll + 2 * k
            if prop == "AICc":
                return -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1)
            return -2 * ll + k * math.log(n)
        if prop == "FitResiduals":
            return _L(res)
        if prop == "PredictedResponse":
            return _L(mu)
        se = np.sqrt(np.diag(cov))
        if prop == "ParameterErrors":
            return _L(se)
        if prop == "CovarianceMatrix":
            return _L(cov)
        if prop == "CorrelationMatrix":
            return _L(cov / np.outer(se, se))
        z = beta / se
        normal = fam in ("Binomial", "Poisson")
        if prop in ("ParameterZStatistics", "ParameterTStatistics"):
            return _L(z)
        if prop == "ParameterPValues":
            return _L([2 * SM.norm_sf(abs(t)) if normal else _t_two(t, df) for t in z])
        if prop == "ParameterConfidenceIntervals":
            q = _zq(cl) if normal else _tq(cl, df)
            return L(*(L(float(b - q * s), float(b + q * s)) for b, s in zip(beta, se)))
        raise ValueError(f'FittedModel: unknown property "{prop}"')
    # least-squares models
    if exact:
        Xs = sp.Matrix([[S(v) for v in r.args] for r in _get(a, "DesignMatrix").args])
        ys = sp.Matrix([S(v) for v in _get(a, "Response").args])
        ws = [S(v) for v in _get(a, "Weights").args]
        bs = sp.Matrix([S(v) for v in _get(a, "BestFitParameters").args])
        preds = Xs * bs
        ress = ys - preds
        rss_e = sp.simplify(sum(wi * r ** 2 for wi, r in zip(ws, ress)))
        if prop == "FitResiduals":
            return L(*(R(sp.simplify(r)) for r in ress))
        if prop == "PredictedResponse":
            return L(*(R(sp.simplify(r)) for r in preds))
        if prop in ("RSquared", "AdjustedRSquared"):
            const = all(Xs[i, 0] == 1 for i in range(n))
            wsum = sum(ws)
            ybar = sum(wi * yi for wi, yi in zip(ws, ys)) / wsum if const else 0
            tss = sp.simplify(sum(wi * (yi - ybar) ** 2 for wi, yi in zip(ws, ys)))
            r2 = sp.simplify(1 - rss_e / tss)
            if prop == "RSquared":
                return R(r2)
            return R(sp.simplify(1 - (1 - r2) * (n - (1 if const else 0)) / df))
        if prop == "EstimatedVariance":
            return R(sp.simplify(rss_e / df))
        if prop == "CovarianceMatrix":
            W_ = sp.diag(*ws)
            return R(sp.simplify(rss_e / df * (Xs.T * W_ * Xs).inv()))
        if prop == "ParameterErrors":
            W_ = sp.diag(*ws)
            C = rss_e / df * (Xs.T * W_ * Xs).inv()
            return L(*(R(sp.sqrt(sp.simplify(C[i, i]))) for i in range(p)))
    rss = float(np.sum(w * res ** 2))
    if prop == "FitResiduals":
        return _L(res)
    if prop == "PredictedResponse":
        return _L(pred)
    const = typ == "Linear" and np.allclose(X[:, 0], 1.0)
    ybar = float(np.sum(w * y) / np.sum(w)) if const or typ == "Nonlinear" else 0.0
    tss = float(np.sum(w * (y - ybar) ** 2))
    if prop == "RSquared":
        return 1 - rss / tss
    if prop == "AdjustedRSquared":
        k0 = 1 if const or typ == "Nonlinear" else 0
        return 1 - (1 - (1 - rss / tss)) * (n - k0) / df
    if df <= 0:
        raise ValueError("FittedModel: no residual degrees of freedom")
    s2 = rss / df
    if prop == "EstimatedVariance":
        return s2
    cov = s2 * np.linalg.inv(X.T @ (X * w[:, None]))
    se = np.sqrt(np.diag(cov))
    if prop == "CovarianceMatrix":
        return _L(cov)
    if prop == "CorrelationMatrix":
        return _L(cov / np.outer(se, se))
    if prop == "ParameterErrors":
        return _L(se)
    t = beta / se
    if prop == "ParameterTStatistics":
        return _L(t)
    if prop == "ParameterPValues":
        return _L([_t_two(v, df) for v in t])
    if prop == "ParameterConfidenceIntervals":
        q = _tq(cl, df)
        return L(*(L(float(b - q * s), float(b + q * s)) for b, s in zip(beta, se)))
    if prop in ("LogLikelihood", "AIC", "AICc", "BIC"):
        # Gaussian log-likelihood with the ML variance (weights scale the variances)
        sig2 = rss / n
        ll = -n / 2 * (math.log(2 * math.pi * sig2) + 1) + 0.5 * float(np.sum(np.log(w[w > 0])))
        k = p + 1
        if prop == "LogLikelihood":
            return ll
        if prop == "AIC":
            return -2 * ll + 2 * k
        if prop == "AICc":
            return -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1)
        return -2 * ll + k * math.log(n)
    if prop == "ANOVATableDegreesOfFreedom":
        return L(*([1] * (p - (1 if const else 0))), df, n - (1 if const else 0))
    raise ValueError(f'FittedModel: unknown property "{prop}"')


def _glm_loglik(fam, y, mu, w, dev, n):
    with np.errstate(all="ignore"):
        if fam == "Binomial":
            return float(np.sum(w * (np.where(y > 0, y * np.log(mu), 0) + np.where(y < 1, (1 - y) * np.log(1 - mu), 0))))
        if fam == "Poisson":
            return float(np.sum(w * (y * np.log(mu) - mu - np.array([math.lgamma(v + 1) for v in y]))))
        if fam == "Gaussian":
            s2 = float(np.sum(w * (y - mu) ** 2)) / n
            return float(-n / 2 * (math.log(2 * math.pi * s2) + 1))
        if fam == "Gamma":
            # ML shape by Newton on the profile likelihood (as R's MASS::gamma.shape / statsmodels uses scale)
            disp = dev / n
            nu = 1 / disp if disp > 0 else 1e10
            return float(np.sum(w * (nu * np.log(nu * y / mu) - nu * y / mu - np.log(y) -
                                     np.array([math.lgamma(nu)] * len(y)))))
        disp = dev / n
        return float(np.sum(w * (-0.5 * (np.log(2 * math.pi * disp * y ** 3) + (y - mu) ** 2 / (disp * mu ** 2 * y)))))


@fn("FittedModel", area=AREA, status="added",
    notes="FittedModel[<|...|>] is the object returned by LinearModelFit, NonlinearModelFit, "
          "GeneralizedLinearModelFit, LogitModelFit and ProbitModelFit; fm[x1, ...] evaluates the best fit at a "
          "point and fm[\"prop\"] (or fm[{props}]) gives \"BestFit\", \"BestFitParameters\", \"BasisFunctions\", "
          "\"Function\", \"Data\", \"DesignMatrix\", \"Response\", \"Weights\", \"FitResiduals\", "
          "\"PredictedResponse\", \"RSquared\", \"AdjustedRSquared\", \"EstimatedVariance\", \"ParameterErrors\", "
          "\"ParameterTStatistics\", \"ParameterPValues\", \"ParameterConfidenceIntervals\" (ConfidenceLevel), "
          "\"CovarianceMatrix\", \"CorrelationMatrix\", \"LogLikelihood\", \"AIC\", \"AICc\", \"BIC\" (Gaussian "
          "likelihood with the ML variance, the variance counted as a parameter), \"ANOVATableDegreesOfFreedom\" "
          "and the GLM properties; exact linear fits give exact residuals, R^2, variance and errors. Formatted "
          "tables and prediction bands raise NotImplementedError; unknown properties ValueError. Check: reference "
          "values from statsmodels OLS / GLM results (tested).")
def _fitted_model(a: Expr) -> Expr:
    if not (isinstance(a, Expr) and a.head == "Association"):
        raise TypeError("FittedModel: expected FittedModel[<|...|>] (use LinearModelFit and related functions)")
    raise Unevaluated()


def _fm_apply(head, args, ev_):
    a = _model_assoc("FittedModel", head)
    if a is None:
        return UNH
    try:
        return _fm_dispatch(a, args, ev_)
    except KeyError as e:
        raise ValueError(f"FittedModel: malformed model object (missing {e})")


def _fm_dispatch(a, args, ev_):
    if len(args) == 1 and isinstance(args[0], str):
        return _fm_property(a, args[0])
    if len(args) == 1 and is_list(args[0]) and args[0].args and all(isinstance(q, str) for q in args[0].args):
        return L(*(_fm_property(a, q) for q in args[0].args))
    vs = list(_get(a, "Variables").args)
    best = _get(a, "BestFit")
    if not vs:
        return ev_(Expr(Expr("Function", best), *args))
    if len(args) != len(vs):
        raise ValueError(f"FittedModel: expected {len(vs)} argument(s)")
    return ev_(Expr("ReplaceAll", best, L(*(rule(s, v) for s, v in zip(vs, args)))))


KL.COMPOUND_HEAD_HOOKS["FittedModel"] = _fm_apply
