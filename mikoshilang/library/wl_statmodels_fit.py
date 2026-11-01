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
    except Unevaluated:
        return UNH
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


# ─────────────────────────── clustering ───────────────────────────

_LINKAGES = ("Single", "Complete", "Average", "WeightedAverage", "Centroid", "Median", "Ward")


def _dist_fn(spec, fname, elems):
    """Python distance for the DistanceFunction setting (Automatic -> SquaredEuclidean / Euclidean / EditDistance)"""
    nm = SM.str_or_sym(spec) if isinstance(spec, (str, Symbol)) else None
    strings = all(isinstance(e, str) for e in elems)
    if nm is None and spec is not None and not isinstance(spec, (str, Symbol)):
        def gen(a, b, f=spec):
            v = ev(Expr(f, a[1], b[1]))
            sv = S(v)
            if not (isinstance(sv, sp.Basic) and sv.is_number and sv.is_real):
                raise ValueError(f"{fname}: the distance function returned the non-real value {v}")
            return float(sv)
        return gen
    if strings:
        if nm in (None, "Automatic", "EditDistance"):
            return lambda a, b: float(_edit(a[1], b[1]))
        if nm == "HammingDistance":
            def ham(a, b):
                if len(a[1]) != len(b[1]):
                    raise ValueError(f"{fname}: HammingDistance needs strings of equal length")
                return float(sum(x != y for x, y in zip(a[1], b[1])))
            return ham
        raise ValueError(f"{fname}: the distance {nm} does not apply to strings")
    simple = {
        "EuclideanDistance": lambda u, v: float(np.sqrt(np.sum((u - v) ** 2))),
        "SquaredEuclideanDistance": lambda u, v: float(np.sum((u - v) ** 2)),
        "ManhattanDistance": lambda u, v: float(np.sum(np.abs(u - v))),
        "ChessboardDistance": lambda u, v: float(np.max(np.abs(u - v))),
        "CanberraDistance": lambda u, v: float(np.sum(np.where(np.abs(u) + np.abs(v) > 0,
                                                               np.abs(u - v) / np.where(np.abs(u) + np.abs(v) > 0,
                                                                                        np.abs(u) + np.abs(v), 1), 0))),
        "BrayCurtisDistance": lambda u, v: float(np.sum(np.abs(u - v)) / np.sum(np.abs(u + v))),
        "CosineDistance": lambda u, v: float(1 - u @ v / (np.linalg.norm(u) * np.linalg.norm(v))),
        "HammingDistance": lambda u, v: float(np.sum(u != v)),
    }
    if nm is None or nm == "Automatic":
        nm = "SquaredEuclideanDistance" if fname in ("FindClusters", "ClusteringComponents") else "EuclideanDistance"
    if nm not in simple:
        raise ValueError(f"{fname}: unknown DistanceFunction {nm}")
    g = simple[nm]
    return lambda a, b: g(a[0], b[0])


def _edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _elements(xs, fname):
    """[(numpy vector or None, python value)] for the clustering elements"""
    out_ = []
    for e in xs:
        if isinstance(e, str):
            out_.append((None, e))
        elif is_list(e) and e.args and all(not is_list(c) for c in e.args):
            try:
                out_.append((np.array([SM.fl(SM._num(c, fname)) for c in e.args]), e))
            except TypeError:
                raise NotImplementedError(f"{fname}: only numbers, numeric vectors and strings can be clustered")
        elif isinstance(e, Symbol) and e.name not in ("True", "False"):
            raise Unevaluated()
        elif isinstance(e, (bool,)) or (isinstance(e, Symbol)):
            raise NotImplementedError(f"{fname}: Boolean data is not supported")
        else:
            v = S(e)
            if not (isinstance(v, sp.Basic) and v.is_number):
                if isinstance(v, sp.Basic) and v.free_symbols:
                    raise Unevaluated()
                raise NotImplementedError(f"{fname}: only numbers, numeric vectors and strings can be clustered")
            if not v.is_real:
                raise NotImplementedError(f"{fname}: complex data is not supported")
            out_.append((np.array([float(v)]), e))
    if out_ and any(o[0] is not None for o in out_):
        if any(o[0] is None for o in out_) or len({o[0].shape for o in out_}) != 1:
            raise ValueError(f"{fname}: the elements must all be numbers, vectors of one length, or strings")
    return out_


def _dmatrix(elems, dist):
    n = len(elems)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dist(elems[i], elems[j])
    return D


def _agglomerate(D, linkage, fname):
    """Lance-Williams agglomeration (SciPy's convention: the update formulas act on D itself).
    Returns merges [(a, b, height, size)] with cluster ids as in scipy (leaves 0..n-1, new n, n+1, ...)."""
    n = D.shape[0]
    if linkage not in _LINKAGES:
        raise ValueError(f"{fname}: unknown ClusterDissimilarityFunction {linkage!r}")
    active = {i: [i] for i in range(n)}
    d = {(i, j): D[i, j] for i in range(n) for j in range(i + 1, n)}
    merges = []
    nxt = n
    while len(active) > 1:
        (a, b), h = min(d.items(), key=lambda kv: (kv[1], kv[0]))
        na, nb = len(active[a]), len(active[b])
        new = {}
        for c in active:
            if c in (a, b):
                continue
            dac = d[(min(a, c), max(a, c))]
            dbc = d[(min(b, c), max(b, c))]
            nc = len(active[c])
            if linkage == "Single":
                v = min(dac, dbc)
            elif linkage == "Complete":
                v = max(dac, dbc)
            elif linkage == "Average":
                v = (na * dac + nb * dbc) / (na + nb)
            elif linkage == "WeightedAverage":
                v = (dac + dbc) / 2
            elif linkage == "Centroid":
                v = math.sqrt(max(0.0, (na * dac ** 2 + nb * dbc ** 2) / (na + nb) - na * nb * h ** 2 / (na + nb) ** 2))
            elif linkage == "Median":
                v = math.sqrt(max(0.0, dac ** 2 / 2 + dbc ** 2 / 2 - h ** 2 / 4))
            else:
                t = na + nb + nc
                v = math.sqrt(max(0.0, ((na + nc) * dac ** 2 + (nb + nc) * dbc ** 2 - nc * h ** 2) / t))
            new[c] = v
        members = active.pop(a) + active.pop(b)
        d = {k: v for k, v in d.items() if a not in k and b not in k}
        for c, v in new.items():
            d[(min(c, nxt), max(c, nxt))] = v
        active[nxt] = sorted(members)
        merges.append((a, b, h, len(members)))
        nxt += 1
    return merges


def _cut(merges, n, k):
    """partition (lists of leaf indices) after n - k merges"""
    groups = {i: [i] for i in range(n)}
    nxt = n
    for a, b, _, _ in merges[:n - k]:
        groups[nxt] = sorted(groups.pop(a) + groups.pop(b))
        nxt += 1
    return sorted(groups.values(), key=lambda g: g[0])


def _method_linkage(o, fname, need_method=True):
    m = o.get("Method", "Automatic")
    link = None
    if is_list(m) and m.args and isinstance(m.args[0], str):
        for r in m.args[1:]:
            if SM._is_rule(r) and r.args[0] == "Linkage":
                link = SM.str_or_sym(r.args[1])
            else:
                raise ValueError(f"{fname}: unknown Method suboption {r}")
        m = m.args[0]
    m = SM.str_or_sym(m)
    cdf = o.get("ClusterDissimilarityFunction")
    if cdf is not None and SM.str_or_sym(cdf) != "Automatic":
        if not isinstance(cdf, (str, Symbol)):
            raise NotImplementedError(f"{fname}: custom ClusterDissimilarityFunction settings are not supported")
        link = SM.str_or_sym(cdf)
    if need_method:
        if m == "Automatic":
            raise NotImplementedError(f"{fname}: automatic method selection is not supported (its criterion is "
                                      f"undocumented); give Method -> \"Agglomerate\"")
        if m in ("KMeans", "KMedoids", "DBSCAN", "GaussianMixture", "JarvisPatrick", "MeanShift", "Spectral",
                 "SpanningTree", "NeighborhoodContraction", "Optimal", "Optimum", "KMedians"):
            raise NotImplementedError(f"{fname}: Method \"{m}\" is not supported (randomised or undocumented)")
        if m != "Agglomerate":
            raise ValueError(f"{fname}: unknown Method {m!r}")
    if link is not None and link not in _LINKAGES:
        raise ValueError(f"{fname}: unknown linkage {link!r}")
    return link or "Single"


def _cluster_input(x, fname):
    """(elements, labels) for list / rules / association / data -> labels"""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if SM._is_rule(x) and is_list(x.args[0]):
        data, labs = x.args
        if not is_list(labs) or len(labs.args) != len(data.args):
            raise ValueError(f"{fname}: data -> labels needs one label per element")
        return list(data.args), list(labs.args)
    if isinstance(x, Expr) and x.head == "Association":
        return [r.args[1] for r in x.args], [r.args[0] for r in x.args]
    if not is_list(x) or not x.args:
        raise TypeError(f"{fname}: the data must be a non-empty list")
    if all(SM._is_rule(r) for r in x.args):
        return [r.args[0] for r in x.args], [r.args[1] for r in x.args]
    return list(x.args), list(x.args)


def _n_arg(nn, n, fname):
    upto = False
    if isinstance(nn, Expr) and nn.head == "UpTo" and len(nn.args) == 1:
        nn, upto = nn.args[0], True
    if isinstance(nn, Symbol) or (isinstance(nn, Expr) and not isinstance(S(nn), sp.Integer)):
        if isinstance(nn, Symbol) and nn.name == "Automatic":
            raise NotImplementedError(f"{fname}: automatic choice of the number of clusters is not supported")
        if isinstance(nn, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the number of clusters must be a positive integer")
    if not isinstance(nn, int) or isinstance(nn, bool) or nn < 1:
        raise TypeError(f"{fname}: the number of clusters must be a positive integer")
    if nn > n:
        if upto:
            return n
        raise ValueError(f"{fname}: cannot find {nn} clusters in {n} elements")
    return nn


_CL_OPTS = ("Method", "DistanceFunction", "ClusterDissimilarityFunction", "CriterionFunction", "Weights",
            "PerformanceGoal", "RandomSeeding", "FeatureExtractor", "FeatureNames", "FeatureTypes")


def _cl_common(o, fname):
    for k in ("CriterionFunction", "Weights", "FeatureExtractor", "FeatureNames", "FeatureTypes"):
        if k in o and SM.str_or_sym(o[k]) not in ("Automatic", "Identity", "None"):
            raise NotImplementedError(f"{fname}: the {k} option is not supported")


def _partition(elems, nclust, o, fname):
    link = _method_linkage(o, fname)
    D = _dmatrix(elems, _dist_fn(o.get("DistanceFunction"), fname, [e[1] for e in elems]))
    merges = _agglomerate(D, link, fname)
    return _cut(merges, len(elems), nclust)


@fn("FindClusters", area=AREA, status="added",
    notes="FindClusters[{e1, ...}, n, Method -> \"Agglomerate\"] partitions numbers, numeric vectors or strings into "
          "n clusters by agglomerative clustering (WL's \"Agglomerate\" is single linkage; Method -> {\"Agglomerate\", "
          "\"Linkage\" -> l} or ClusterDissimilarityFunction -> l choose \"Single\", \"Complete\", \"Average\", "
          "\"WeightedAverage\", \"Centroid\", \"Median\" or \"Ward\"); DistanceFunction defaults to "
          "SquaredEuclideanDistance (numbers, vectors) or EditDistance (strings), named distances or any function "
          "f[a, b]. Forms {e1 -> v1, ...}, <|k1 -> e1, ...|> (keys returned) and {e1, ...} -> {v1, ...}; clusters "
          "are listed in order of their first element, elements in input order; ties merge the lowest-index pair "
          "first; UpTo[n] allowed. The automatic method and automatic number of clusters (undocumented criteria) "
          "and randomised methods (\"KMeans\", ...) raise NotImplementedError. Check: partitions and merge heights "
          "match scipy.cluster.hierarchy.linkage / fcluster(maxclust) (reference), and the WL doc example "
          "{1, 2, 10, 12, 3, 1, 13, 25} with 3 clusters.")
def _find_clusters(x: ListOf[Expr] | Rule | Expr, nn: PosInteger | Expr = None, *rest: Option) -> ListOf[ListOf[Expr]]:
    name = "FindClusters"
    rest = list(rest)
    if nn is not None and SM._is_rule(nn) and isinstance(nn.args[0], Symbol):
        rest.insert(0, nn)
        nn = None
    _, o = opts(rest, _CL_OPTS, name)
    _cl_common(o, name)
    data, labs = _cluster_input(x, name)
    elems = _elements(data, name)
    if nn is None:
        _method_linkage(o, name)
        raise NotImplementedError(f"{name}: automatic choice of the number of clusters is not supported; give n")
    k = _n_arg(nn, len(elems), name)
    parts = _partition(elems, k, o, name)
    return L(*(L(*(labs[i] for i in g)) for g in parts))


def _cc_struct(x, level, depth=1):
    """leaves to cluster: (paths, values); level None -> atomic entries"""
    if is_list(x) and (level is None or depth <= level):
        if level is not None and depth == level:
            return [((i,), c) for i, c in enumerate(x.args)]
        res = []
        for i, c in enumerate(x.args):
            if is_list(c):
                res += [((i,) + p, v) for p, v in _cc_struct(c, level, depth + 1)]
            else:
                res.append(((i,), c))
        return res
    return [((), x)]


def _cc_rebuild(x, labels, path=()):
    if path in labels:
        return labels[path]
    return L(*(_cc_rebuild(c, labels, path + (i,)) for i, c in enumerate(x.args)))


@fn("ClusteringComponents", area=AREA, status="added",
    notes="ClusteringComponents[array, n, level, Method -> \"Agglomerate\"] replaces each element (each atomic "
          "entry by default, the parts at the given level otherwise, e.g. the rows of a matrix with level 1) by the "
          "index of its cluster from FindClusters's agglomerative partition; clusters are numbered by first "
          "appearance, ragged arrays keep their shape; UpTo[n], DistanceFunction and linkage settings as in "
          "FindClusters. The automatic method, automatic number of clusters and images raise NotImplementedError. "
          "Check: the labels induce exactly the FindClusters partition (tested), and WL doc examples "
          "{{1, 2}, {3, 4}, {10, 11}, {12, 13}} with 2 clusters at levels 1 and 2.")
def _clustering_components(x: ListOf[Expr] | MatrixOf[Real], nn: PosInteger | Expr = None, level: PosInteger = None,
                           *rest: Option) -> ListOf[PosInteger] | MatrixOf[PosInteger]:
    name = "ClusteringComponents"
    rest = list(rest)
    for _ in range(2):
        if level is not None and SM._is_rule(level):
            rest.insert(0, level)
            level = None
        if nn is not None and SM._is_rule(nn) and level is None:
            rest.insert(0, nn)
            nn = None
    _, o = opts(rest, _CL_OPTS, name)
    _cl_common(o, name)
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not is_list(x) or not x.args:
        if isinstance(x, Expr) and x.head in ("Image", "Image3D"):
            raise NotImplementedError(f"{name}: images are not supported")
        raise TypeError(f"{name}: the data must be a non-empty list")
    if level is not None and (not isinstance(level, int) or isinstance(level, bool) or level < 1):
        raise TypeError(f"{name}: the level must be a positive integer")
    leaves = _cc_struct(x, level)
    elems = _elements([v for _, v in leaves], name)
    if nn is None:
        _method_linkage(o, name)
        raise NotImplementedError(f"{name}: automatic choice of the number of clusters is not supported; give n")
    k = _n_arg(nn, len(elems), name)
    parts = _partition(elems, k, o, name)
    lab = {}
    for c, g in enumerate(parts, 1):
        for i in g:
            lab[leaves[i][0]] = c
    if () in lab:
        return lab[()]
    return _cc_rebuild(x, lab)


@fn("ClusteringTree", area=AREA, status="added",
    notes="ClusteringTree[{e1, ...}, ClusterDissimilarityFunction -> l] gives the agglomerative hierarchy as a "
          "Graph tree: vertices 1, 2, ... in preorder (root 1; the child holding the lower-index element first), "
          "edges parent -> child, VertexWeight -> the merge dissimilarity (0 at leaves) and VertexLabels -> the "
          "leaf elements; ClusteringTree[data, d] first merges all clusters at dissimilarity < d, and those "
          "clusters (lists) become the leaves. Linkages and distances (default EuclideanDistance / EditDistance) as "
          "in FindClusters; data -> labels and <|...|> label the leaves. The Automatic linkage (undocumented), "
          "custom linkage functions and images raise NotImplementedError. Check: merge heights match "
          "scipy.cluster.hierarchy.linkage (reference), and the root weight equals the largest merge height "
          "(by construction).")
def _clustering_tree(x: ListOf[Expr] | Rule | Expr, d: Real = None, *rest: Option) -> Expr:
    name = "ClusteringTree"
    rest = list(rest)
    if d is not None and SM._is_rule(d) and isinstance(d.args[0], Symbol):
        rest.insert(0, d)
        d = None
    _, o = opts(rest, _CL_OPTS + ("GraphLayout", "VertexSize", "VertexShape", "VertexLabels", "EdgeStyle",
                                  "ImageSize", "VertexStyle", "PlotTheme"), name)
    _cl_common(o, name)
    cdf = o.get("ClusterDissimilarityFunction")
    if cdf is None or SM.str_or_sym(cdf) == "Automatic":
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise NotImplementedError(f"{name}: the Automatic linkage is undocumented; give ClusterDissimilarityFunction")
    link = _method_linkage(o, name, need_method=False)
    data, labs = _cluster_input(x, name)
    elems = _elements(data, name)
    thr = None
    if d is not None:
        thr = SM.fl(SM._num(d, name, "threshold"))
        if thr < 0:
            raise ValueError(f"{name}: the threshold must be non-negative")
    n = len(elems)
    D = _dmatrix(elems, _dist_fn(o.get("DistanceFunction"), name, [e[1] for e in elems]))
    merges = _agglomerate(D, link, name) if n > 1 else []
    children, height, members = {}, {}, {i: [i] for i in range(n)}
    for t, (a, b, h, _) in enumerate(merges):
        c = n + t
        children[c], height[c] = (a, b), h
        members[c] = sorted(members[a] + members[b])
    root = n + len(merges) - 1 if merges else 0

    def collapsed(c):
        return c in children and thr is not None and height[c] < thr

    verts, edges, weights, labels = [], [], [], []

    def walk(c, parent):
        vid = len(verts) + 1
        verts.append(vid)
        if parent is not None:
            edges.append(Expr("DirectedEdge", parent, vid))
        if c in children and not collapsed(c):
            weights.append(rule(vid, float(height[c])))
            for ch in sorted(children[c], key=lambda q: members[q][0]):
                walk(ch, vid)
        else:
            weights.append(rule(vid, 0))
            lab = L(*(labs[i] for i in members[c])) if thr is not None else labs[members[c][0]]
            labels.append(rule(vid, lab))
    walk(root, None)
    return Expr("Graph", L(*verts), L(*edges), rule(Symbol("VertexWeight"), L(*weights)),
                rule(Symbol("VertexLabels"), L(*labels)))


# ClusteringMeasurements

_CM_INTERNAL = ("CalinskiHarabasz", "Dunn", "RSquared", "Silhouette", "StandardDeviation", "SilhouetteScore",
                "SilhouetteScoreList")
_CM_EXTERNAL = ("Purity", "Rand")
_CM_UNSUPPORTED = ("DaviesBouldin", "DaviesBouldinScore", "RSquaredScore", "StandardDeviationScore", "PurityScore",
                   "RandScore", "Summary")


def _cm_clusters(x, fname):
    """list of clusters (lists of elements) and the cluster keys"""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if isinstance(x, Expr) and x.head == "Association":
        return [list(r.args[1].args) for r in x.args if is_list(r.args[1])], [r.args[0] for r in x.args]
    if SM._is_rule(x) and is_list(x.args[0]):
        data, labs = x.args
        if not is_list(labs):
            raise NotImplementedError(f"{fname}: classifier functions are not supported")
        if len(labs.args) != len(data.args):
            raise ValueError(f"{fname}: data -> labels needs one label per element")
        keys = []
        for lb in labs.args:
            if lb not in keys:
                keys.append(lb)
        return [[e for e, lb in zip(data.args, labs.args) if lb == k] for k in keys], keys
    if is_list(x) and x.args and all(SM._is_rule(r) for r in x.args):
        return _cm_clusters(Expr("Rule", L(*(r.args[0] for r in x.args)), L(*(r.args[1] for r in x.args))), fname)
    if is_list(x) and x.args and all(is_list(c) and c.args for c in x.args):
        return [list(c.args) for c in x.args], list(range(1, len(x.args) + 1))
    raise TypeError(f"{fname}: expected a list of clusters, an association of clusters or data -> assignments")


def _cm_prop(prop, X, lab, k, D, gt, fname):
    n = len(lab)
    idx = [np.where(lab == a)[0] for a in range(k)]
    if prop in _CM_EXTERNAL:
        if gt is None:
            raise ValueError(f"{fname}: \"{prop}\" needs ground-truth clusters")
        if prop == "Purity":
            return float(sum(max(np.sum(gt[i] == g) for g in set(gt[i])) for i in idx) / n)
        agree = sum((lab[i] == lab[j]) == (gt[i] == gt[j]) for i in range(n) for j in range(i + 1, n))
        return float(agree / (n * (n - 1) / 2))
    if prop in ("Silhouette", "SilhouetteScore", "SilhouetteScoreList"):
        if k < 2:
            raise ValueError(f"{fname}: the silhouette needs at least two clusters")
        s = np.zeros(n)
        for i in range(n):
            own = idx[lab[i]]
            if len(own) == 1:
                continue
            a = D[i, own].sum() / (len(own) - 1)
            b = min(D[i, idx[c]].mean() for c in range(k) if c != lab[i])
            s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        if prop == "Silhouette":
            return float(s.mean())
        if prop == "SilhouetteScore":
            return _L([s[i].mean() for i in idx])
        return L(*(L(*(float(s[j]) for j in i)) for i in idx))
    if prop == "Dunn":
        if k < 2:
            raise ValueError(f"{fname}: the Dunn index needs at least two clusters")
        sep = min(D[np.ix_(idx[a], idx[b])].min() for a in range(k) for b in range(a + 1, k))
        diam = max(D[np.ix_(i, i)].max() for i in idx)
        if diam == 0:
            raise ArithmeticError(f"{fname}: every cluster has zero diameter; the Dunn index is infinite")
        return float(sep / diam)
    if X is None:
        raise NotImplementedError(f"{fname}: \"{prop}\" needs numeric vectors (it uses cluster centroids)")
    cent = np.array([X[i].mean(axis=0) for i in idx])
    g = X.mean(axis=0)
    W = float(sum(((X[i] - cent[a]) ** 2).sum() for a, i in enumerate(idx)))
    if prop == "CalinskiHarabasz":
        if k < 2 or n <= k:
            raise ValueError(f"{fname}: CalinskiHarabasz needs 1 < k < n")
        B = float(sum(len(i) * ((cent[a] - g) ** 2).sum() for a, i in enumerate(idx)))
        if W == 0:
            raise ArithmeticError(f"{fname}: zero within-cluster dispersion")
        return B / (k - 1) / (W / (n - k))
    if prop == "StandardDeviation":
        if n <= k:
            raise ValueError(f"{fname}: StandardDeviation needs more examples than clusters")
        return math.sqrt(W / (n - k))
    # RSquared: 1 - (sum of distances to the cluster centroids) / (sum of distances to the dataset centroid)
    within = float(sum(np.linalg.norm(X[i] - cent[a], axis=1).sum() for a, i in enumerate(idx)))
    total = float(np.linalg.norm(X - g, axis=1).sum())
    if total == 0:
        raise ArithmeticError(f"{fname}: all examples coincide")
    return 1 - within / total


@fn("ClusteringMeasurements", area=AREA, status="added",
    notes="ClusteringMeasurements[clusters, \"prop\"] (clusters as {{e11, ...}, ...}, <|k1 -> {...}, ...|>, "
          "{e1 -> c1, ...} or {e1, ...} -> {c1, ...}) gives \"CalinskiHarabasz\", \"Dunn\", \"RSquared\" (1 - "
          "Sum[|e - centroid|] / Sum[|e - dataset centroid|], unsquared distances, as WL), \"Silhouette\", "
          "\"SilhouetteScore\" (per cluster), \"SilhouetteScoreList\" (per example; singletons score 0), "
          "\"StandardDeviation\" (Sqrt[within SS/(n - k)]); with ground truth ClusteringMeasurements[clusters, "
          "truth, \"prop\"]: \"Purity\" and \"Rand\"; {props} lists; \"Properties\". DistanceFunction defaults to "
          "EuclideanDistance (EditDistance for strings). \"DaviesBouldin\" (WL's dispersion definition could not "
          "be reproduced from the doc example), \"Summary\", the other per-cluster scores, classifier functions and "
          "FeatureExtractor raise NotImplementedError. Check: WL doc example values (CalinskiHarabasz 128.42, Dunn "
          "1.79653, RSquared 0.888558, Silhouette 0.775515, StandardDeviation 0.123269) and sklearn.metrics "
          "(calinski_harabasz_score, silhouette_score / silhouette_samples).")
def _clustering_measurements(x: ListOf[ListOf[Expr]] | Rule | Expr, b: Expr | String | ListOf[String] = None,
                             c: String | ListOf[String] = None, *rest: Option) -> Real | ListOf[Real] | Expr:
    name = "ClusteringMeasurements"
    rest = list(rest)
    for _ in range(2):
        if c is not None and SM._is_rule(c) and isinstance(c.args[0], Symbol):
            rest.insert(0, c)
            c = None
        if b is not None and SM._is_rule(b) and isinstance(b.args[0], Symbol) and c is None:
            rest.insert(0, b)
            b = None
    _, o = opts(rest, ("DistanceFunction", "FeatureExtractor"), name)
    if "FeatureExtractor" in o and SM.str_or_sym(o["FeatureExtractor"]) not in ("Identity", "Automatic"):
        raise NotImplementedError(f"{name}: FeatureExtractor is not supported")
    truth = None
    if c is not None:
        truth, prop = b, c
    else:
        prop = b
    if prop is None:
        raise TypeError(f"{name}: a property is required")
    if isinstance(prop, Symbol) or isinstance(x, Symbol) or isinstance(truth, Symbol):
        raise Unevaluated()
    props = list(prop.args) if is_list(prop) else [prop]
    if not all(isinstance(q, str) for q in props):
        raise TypeError(f"{name}: properties must be strings")
    for q in props:
        if q in _CM_UNSUPPORTED:
            raise NotImplementedError(f"{name}: the property \"{q}\" is not supported")
        if q not in _CM_INTERNAL + _CM_EXTERNAL + ("Properties",):
            raise ValueError(f"{name}: unknown property \"{q}\"")
    clusters, _ = _cm_clusters(x, name)
    flat = [e for cl in clusters for e in cl]
    lab = np.array([a for a, cl in enumerate(clusters) for _ in cl])
    k, n = len(clusters), len(flat)
    elems = _elements(flat, name)
    X = np.array([e[0] for e in elems]) if elems[0][0] is not None else None
    D = _dmatrix(elems, _dist_fn(o.get("DistanceFunction"), "ClusteringMeasurements", flat))
    gt = None
    if truth is not None:
        tcl, _ = _cm_clusters(truth, name)
        where = {}
        for a, cl in enumerate(tcl):
            for e in cl:
                where.setdefault(e, a)
        if sorted(map(str, flat)) != sorted(str(e) for cl in tcl for e in cl):
            raise ValueError(f"{name}: the ground truth must contain the same examples")
        gt = np.array([where[e] for e in flat])

    def one(q):
        if q == "Properties":
            return L(*(_CM_INTERNAL + _CM_EXTERNAL if gt is not None else _CM_INTERNAL))
        return _cm_prop(q, X, lab, k, D, gt, name)
    if is_list(prop):
        return L(*(one(q) for q in props))
    return one(prop)


# ─────────────────────────── survival models ───────────────────────────

def _is_inf(v):
    return (isinstance(v, Symbol) and v.name == "Infinity") or \
        (isinstance(v, Expr) and v.head == "DirectedInfinity" and v.args in ((), (1,))) or v == float("inf")


def _event_data(x, fname):
    """[(time, event?)] from {t1, ...} (with {t, Infinity} right-censored entries) or EventData[times, flags]
    / EventData[Automatic, {times, flags, None}] (flag 0 = event, 1 = right-censored)"""
    if isinstance(x, Symbol):
        raise Unevaluated()
    if isinstance(x, Expr) and x.head == "EventData":
        a = x.args
        if len(a) == 2 and isinstance(a[0], Symbol) and a[0].name == "Automatic" and is_list(a[1]) and \
                len(a[1].args) == 3 and SM.str_or_sym(a[1].args[2]) == "None":
            a = a[1].args[:2]
        if len(a) == 2 and is_list(a[0]) and is_list(a[1]):
            t, _ = SM.vec(a[0], fname, "event times")
            c = list(a[1].args)
            if len(c) != len(t):
                raise ValueError(f"{fname}: EventData needs one censoring flag per time")
            if any(q == -1 for q in c):
                raise NotImplementedError(f"{fname}: left censoring is not supported")
            if not all(q in (0, 1) for q in c):
                raise ValueError(f"{fname}: EventData censoring flags must be 0 (event) or 1 (right-censored)")
            res = [(SM.fl(v), q == 0) for v, q in zip(t, c)]
        elif len(a) == 1 and is_list(a[0]):
            res = _event_data(a[0], fname)
        else:
            raise NotImplementedError(f"{fname}: this EventData form (truncation, interval censoring, ...) is not "
                                      f"supported")
    elif is_list(x) and x.args:
        res = []
        for v in x.args:
            if is_list(v):
                if len(v.args) == 2 and _is_inf(v.args[1]):
                    res.append((SM.fl(SM._num(v.args[0], fname, "event time")), False))
                else:
                    raise NotImplementedError(f"{fname}: interval or left censoring is not supported")
            else:
                res.append((SM.fl(SM._num(v, fname, "event time")), True))
    else:
        raise TypeError(f"{fname}: the data must be a list of event times or EventData")
    if not res:
        raise ValueError(f"{fname}: no event times")
    if any(t < 0 for t, _ in res):
        raise ValueError(f"{fname}: event times must be non-negative")
    return res


def _tnum(t):
    """a time as an integer when it is integral, else a machine number"""
    return int(t) if float(t).is_integer() and abs(t) < 2 ** 53 else float(t)


def _tlist(ts):
    return L(*(_tnum(t) for t in ts))


def _ed_expr(ev_list):
    return Expr("EventData", _tlist([t for t, _ in ev_list]), L(*(0 if e else 1 for _, e in ev_list)))


def _km_table(evs):
    """rows (t, at risk, events, censored) at the distinct times"""
    times = sorted({t for t, _ in evs})
    rows = []
    for t in times:
        r = sum(1 for u, _ in evs if u >= t)
        d = sum(1 for u, e in evs if u == t and e)
        c = sum(1 for u, e in evs if u == t and not e)
        rows.append((t, r, d, c))
    return rows


def _z(cl):
    return float(mpmath.sqrt(2) * mpmath.erfinv(cl))


_TRANSFORMS = ("Linear", "LogLog", "Log", "ArcSinSqrt", "Logit")


def _ci(s, se, z, tr):
    """confidence interval for a survival probability s with standard error se under the transform tr"""
    if not (0 < s < 1) or not math.isfinite(se):
        if s == 1 and se == 0:
            return (1.0, 1.0)
        return None
    if tr == "Linear":
        lo, hi = s - z * se, s + z * se
    elif tr == "LogLog":
        w = z * se / (s * abs(math.log(s)))
        lo, hi = s ** math.exp(w), s ** math.exp(-w)
    elif tr == "Log":
        w = z * se / s
        lo, hi = s * math.exp(-w), s * math.exp(w)
    elif tr == "ArcSinSqrt":
        a = math.asin(math.sqrt(s))
        w = 0.5 * z * se / math.sqrt(s * (1 - s))
        lo, hi = math.sin(max(0.0, a - w)) ** 2, math.sin(min(math.pi / 2, a + w)) ** 2
    else:
        w = z * se / (s * (1 - s))
        lg = math.log(s / (1 - s))
        lo, hi = 1 / (1 + math.exp(-(lg - w))), 1 / (1 + math.exp(-(lg + w)))
    return (max(0.0, lo), min(1.0, hi))


def _ci_opts(o, fname):
    cl = SM.fl(SM._num(o.get("ConfidenceLevel", sp.Rational(95, 100)), fname, "ConfidenceLevel"))
    if not 0 < cl < 1:
        raise ValueError(f"{fname}: ConfidenceLevel must be between 0 and 1")
    tr = SM.str_or_sym(o.get("ConfidenceTransform", "LogLog"))
    if tr not in _TRANSFORMS:
        if not isinstance(o.get("ConfidenceTransform", "LogLog"), (str, Symbol)):
            raise NotImplementedError(f"{fname}: custom confidence transforms are not supported")
        raise ValueError(f"{fname}: unknown ConfidenceTransform {tr!r}")
    cr = o.get("ConfidenceRange")
    if cr is not None and SM.str_or_sym(cr) != "All":
        raise NotImplementedError(f"{fname}: ConfidenceRange (simultaneous bands) is not supported")
    return cl, tr


def _step(points, values, t, before):
    """right-continuous step function: value at the last point <= t, or before"""
    v = before
    for p, q in zip(points, values):
        if p <= t:
            v = q
        else:
            break
    return v


def _fnum(v):
    return Symbol("Indeterminate") if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


_SM_PROPS = ("SF", "CDF", "CHF", "EstimationPoints", "StandardErrors", "PointwiseIntervals", "MedianSurvival",
             "AtRisk", "EventCounts", "RightCensoringCounts", "ObservationCount", "EventData", "EventTable",
             "Properties")


def _sm_core(a):
    evs = [(SM.fl(S(t)), q == 0) for t, q in zip(_get(a, "EventData").args[0].args, _get(a, "EventData").args[1].args)]
    rows = _km_table(evs)
    s, surv, gw, ses = 1.0, [], 0.0, []
    for t, r, d, c in rows:
        s *= 1 - d / r
        surv.append(s)
        gw = gw + d / (r * (r - d)) if r > d else float("inf")
        ses.append(s * math.sqrt(gw) if math.isfinite(gw) else float("nan"))
    return evs, rows, surv, ses


def _sm_property(a, prop, rest):
    name = "SurvivalModel"
    _, o = opts(rest, ("ConfidenceLevel", "ConfidenceTransform", "ConfidenceRange"), name)
    base_o = {k: _get(a, k) for k in ("ConfidenceLevel", "ConfidenceTransform")}
    base_o.update(o)
    evs, rows, surv, ses = _sm_core(a)
    pts = [r[0] for r in rows]
    if prop == "Properties":
        return L(*_SM_PROPS)
    if prop == "EstimationPoints":
        return _tlist(pts)
    if prop == "AtRisk":
        return L(*(r[1] for r in rows))
    if prop == "EventCounts":
        return L(*(r[2] for r in rows))
    if prop == "RightCensoringCounts":
        return L(*(r[3] for r in rows))
    if prop == "ObservationCount":
        return len(evs)
    if prop == "EventData":
        return _get(a, "EventData")
    if prop == "EventTable":
        return L(*(L(_tnum(t), r, d, c, float(sv)) for (t, r, d, c), sv in zip(rows, surv)))
    if prop == "StandardErrors":
        return L(*(_fnum(v if sv > 0 else None) for v, sv in zip(ses, surv)))
    if prop == "PointwiseIntervals":
        cl, tr = _ci_opts(base_o, name)
        z = _z(cl)
        res = []
        for sv, se in zip(surv, ses):
            iv = _ci(sv, se, z, tr) if sv > 0 else None
            res.append(L(*(_fnum(v) for v in iv)) if iv else L(Symbol("Indeterminate"), Symbol("Indeterminate")))
        return L(*res)
    if prop == "MedianSurvival":
        for t, sv in zip(pts, surv):
            if sv <= 0.5 + 1e-12:
                return _tnum(t)
        return Symbol("Indeterminate")
    if prop in ("SF", "CDF", "CHF"):
        def f(t):
            sv = _step(pts, surv, t, 1.0)
            if prop == "SF":
                return float(sv)
            if prop == "CDF":
                return float(1 - sv)
            return float(-math.log(sv)) if sv > 0 else Symbol("Infinity")
        return ("fn", f)
    if prop.endswith("Table") or prop.endswith("Bands") or prop.endswith("Plot") or prop in (
            "EqualPrecisionIntervals", "HallWellnerIntervals", "MeanSurvival", "HF", "PDF", "SurvivalDistribution",
            "FullEventTable", "EventMatrix", "TruncationMatrix", "LeftCensoringCounts"):
        raise NotImplementedError(f'{name}: the property "{prop}" is not supported')
    raise ValueError(f'{name}: unknown property "{prop}"')


@fn("SurvivalModelFit", area=AREA, status="added",
    notes="SurvivalModelFit[{t1, ...}] or [EventData[times, flags]] (flag 1 = right-censored; {t, Infinity} entries "
          "are right-censored too) gives a SurvivalModel with the Kaplan-Meier estimate; options ConfidenceLevel "
          "(95/100) and ConfidenceTransform (\"LogLog\" default, \"Linear\", \"Log\", \"ArcSinSqrt\", \"Logit\") "
          "for \"PointwiseIntervals\" (Greenwood standard errors). Properties are listed under SurvivalModel. "
          "Left, interval censoring and truncation, simultaneous bands (ConfidenceRange) and Method settings raise "
          "NotImplementedError; negative times ValueError. Check: WL doc values (S(24) = 0.9, Greenwood errors "
          "{0.216506, 0.25, 0.216506, Indeterminate}, log-log pointwise intervals) and lifelines "
          "KaplanMeierFitter (survival and confidence interval).")
def _survival_model_fit(x: ListOf[Real] | Expr, *rest: Option) -> Expr:
    name = "SurvivalModelFit"
    _, o = opts(rest, ("ConfidenceLevel", "ConfidenceTransform", "ConfidenceRange", "Method", "WorkingPrecision"),
                name)
    if "Method" in o and SM.str_or_sym(o["Method"]) not in ("Automatic", "KaplanMeier"):
        raise NotImplementedError(f"{name}: Method settings other than Kaplan-Meier are not supported")
    cl, tr = _ci_opts(o, name)
    evs = _event_data(x, name)
    a = _assoc(EventData=_ed_expr(evs), ConfidenceLevel=o.get("ConfidenceLevel", sp.Rational(95, 100)),
               ConfidenceTransform=tr)
    return Expr("SurvivalModel", a)


@fn("SurvivalModel", area=AREA, status="added",
    notes="SurvivalModel[<|...|>] is the Kaplan-Meier model from SurvivalModelFit: sm[t] = S(t) (right-continuous "
          "step function, machine numbers); sm[\"SF\" | \"CDF\" | \"CHF\"][t] (CHF = -Log[S]), sm[\"prop\"] and "
          "sm[{props}] give \"EstimationPoints\" (distinct observed times), \"StandardErrors\" (Greenwood; "
          "Indeterminate where S = 0), \"PointwiseIntervals\" (optionally sm[\"PointwiseIntervals\", "
          "ConfidenceLevel -> p]), \"MedianSurvival\" (first time with S <= 1/2), \"AtRisk\", \"EventCounts\", "
          "\"RightCensoringCounts\", \"ObservationCount\", \"EventData\", \"EventTable\" (rows {t, at risk, events, "
          "censored, S}), \"Properties\". Formatted tables, simultaneous bands, \"MeanSurvival\", \"HF\", \"PDF\" "
          "and \"SurvivalDistribution\" raise NotImplementedError. Check: S(t) is the product of (1 - d/r) over the "
          "event table (tested) and matches lifelines KaplanMeierFitter.")
def _survival_model(a: Expr) -> Expr:
    if not (isinstance(a, Expr) and a.head == "Association"):
        raise TypeError("SurvivalModel: expected SurvivalModel[<|...|>] (use SurvivalModelFit)")
    raise Unevaluated()




def _sm_apply(head, args, ev_):
    try:
        return _sm_apply1(head, args, ev_)
    except Unevaluated:
        return UNH


def _sm_apply1(head, args, ev_):
    # sm["SF"][t]
    if isinstance(head, Expr) and _model_assoc("SurvivalModel", head.head) is not None and len(head.args) == 1 and \
            head.args[0] in ("SF", "CDF", "CHF"):
        a = _model_assoc("SurvivalModel", head.head)
        f = _sm_property(a, head.args[0], [])[1]
        if len(args) != 1:
            raise ValueError("SurvivalModel: expected one time argument")
        return _time_map(f, args[0], "SurvivalModel")
    a = _model_assoc("SurvivalModel", head)
    if a is None:
        return UNH
    try:
        if args and isinstance(args[0], str):
            if args[0] in ("SF", "CDF", "CHF") and len(args) == 1:
                return UNH
            return _sm_property(a, args[0], args[1:])
        if len(args) == 1 and is_list(args[0]) and args[0].args and all(isinstance(q, str) for q in args[0].args):
            if any(q in ("SF", "CDF", "CHF") for q in args[0].args):
                raise ValueError('SurvivalModel: "SF", "CDF" and "CHF" are functions; use sm["SF"][t]')
            return L(*(_sm_property(a, q, []) for q in args[0].args))
        if len(args) == 1:
            return _time_map(_sm_property(a, "SF", [])[1], args[0], "SurvivalModel")
    except KeyError as e:
        raise ValueError(f"SurvivalModel: malformed model object (missing {e})")
    raise ValueError("SurvivalModel: expected a time or a property")


def _time_map(f, t, fname):
    if is_list(t):
        return L(*(_time_map(f, u, fname) for u in t.args))
    if isinstance(t, Symbol) and not _is_inf(t):
        raise Unevaluated()
    if _is_inf(t):
        return f(float("inf"))
    v = S(t)
    if not (isinstance(v, sp.Basic) and v.is_number and v.is_real):
        if isinstance(v, sp.Basic) and v.free_symbols:
            raise Unevaluated()
        raise TypeError(f"{fname}: the time must be a real number")
    return f(float(v))


KL.COMPOUND_HEAD_HOOKS["SurvivalModel"] = _sm_apply
KL.DEEP_HEAD_HOOKS["SurvivalModel"] = _sm_apply


# ─────────────────────────── Cox proportional hazards ───────────────────────────

def _cox_loglik(beta, X, t, d, ties):
    """partial log-likelihood, score and information (Breslow or Efron ties)"""
    n, p = X.shape
    eta = X @ beta
    w = np.exp(eta)
    ll, U, I = 0.0, np.zeros(p), np.zeros((p, p))
    for tt in sorted(set(t[d == 1])):
        D = np.where((t == tt) & (d == 1))[0]
        Rs = np.where(t >= tt)[0]
        m = len(D)
        s0, s1 = w[Rs].sum(), (w[Rs, None] * X[Rs]).sum(0)
        s2 = (w[Rs, None, None] * X[Rs, :, None] * X[Rs, None, :]).sum(0)
        ll += eta[D].sum()
        U += X[D].sum(0)
        if ties == "Efron" and m > 1:
            e0, e1 = w[D].sum(), (w[D, None] * X[D]).sum(0)
            e2 = (w[D, None, None] * X[D, :, None] * X[D, None, :]).sum(0)
            for j in range(m):
                f = j / m
                a0, a1, a2 = s0 - f * e0, s1 - f * e1, s2 - f * e2
                ll -= math.log(a0)
                U -= a1 / a0
                I += a2 / a0 - np.outer(a1, a1) / a0 ** 2
        else:
            ll -= m * math.log(s0)
            U -= m * s1 / s0
            I += m * (s2 / s0 - np.outer(s1, s1) / s0 ** 2)
    return ll, U, I


def _cox_fit(X, t, d, ties, fname):
    p = X.shape[1]
    beta = np.zeros(p)
    ll, U, I = _cox_loglik(beta, X, t, d, ties)
    for _ in range(100):
        try:
            step = np.linalg.solve(I, U)
        except np.linalg.LinAlgError:
            raise ValueError(f"{fname}: the information matrix is singular (collinear covariates?)")
        h = 1.0
        while True:
            nb = beta + h * step
            nll, nU, nI = _cox_loglik(nb, X, t, d, ties)
            if nll >= ll - 1e-12 or h < 1e-8:
                break
            h /= 2
        conv = abs(nll - ll) < 1e-13 * max(1.0, abs(ll)) and np.max(np.abs(nb - beta)) < 1e-10
        beta, ll, U, I = nb, nll, nU, nI
        if conv:
            break
    else:
        raise ArithmeticError(f"{fname}: Newton-Raphson did not converge")
    spread = np.abs(beta) * (X.max(axis=0) - X.min(axis=0))
    if not np.all(np.isfinite(beta)) or np.any(spread > 30) or \
            np.max(np.abs(U)) > 1e-6 * max(1.0, np.max(np.abs(X))) * len(t):
        raise ArithmeticError(f"{fname}: the partial likelihood has no finite maximum (monotone likelihood)")
    return beta, ll, I


def _cox_baseline(X, t, d, beta):
    """Breslow cumulative baseline hazard at the distinct event times, and its variance (no-covariate form)"""
    w = np.exp(X @ beta) if X.shape[1] else np.ones(len(t))
    ets = sorted(set(t[d == 1]))
    H, V, h, v = [], [], 0.0, 0.0
    for tt in ets:
        m = np.sum((t == tt) & (d == 1))
        s0 = w[t >= tt].sum()
        h += m / s0
        v += m / s0 ** 2
        H.append(h)
        V.append(v)
    return ets, H, V


@fn("CoxModelFit", area=AREA, status="added",
    notes="CoxModelFit[edata] (no covariates) and CoxModelFit[{covariates, edata}, {f1, ...}, {x1, ...}] (a covariate "
          "vector or matrix with one row per subject; edata a list of times or EventData[times, flags] / "
          "EventData[Automatic, {times, flags, None}], flag 1 = right-censored) fit the proportional hazards model "
          "h(t) = h0(t) Exp[b.f(x)] by Newton-Raphson on the partial likelihood with Method -> \"Breslow\" "
          "(default) or \"Efron\" ties and give a CoxModel; the baseline is the Breslow estimate at x = 0. "
          "\"Exact\" ties, NominalVariables, StrataVariables and simultaneous bands raise NotImplementedError; a "
          "singular information matrix ValueError; a monotone likelihood (infinite estimate) ArithmeticError. "
          "Check: WL doc values (b = -0.291674, se 0.888806; b = {-0.0535922, 2.11712}, LR 7.54027, Wald 7.83201, "
          "Score 8.93641) and statsmodels PHReg (params, bse, llf) for both tie methods; the score vanishes at "
          "the estimate (self-check).")
def _cox_model_fit(x: ListOf[Expr] | Expr, funs: Expr | ListOf[Expr] = None, vars_: Expr | ListOf[Expr] = None,
                   *rest: Option) -> Expr:
    name = "CoxModelFit"
    rest = list(rest)
    for _ in range(2):
        if vars_ is not None and SM._is_rule(vars_):
            rest.insert(0, vars_)
            vars_ = None
        if funs is not None and SM._is_rule(funs) and vars_ is None:
            rest.insert(0, funs)
            funs = None
    _, o = opts(rest, ("ConfidenceLevel", "ConfidenceRange", "ConfidenceTransform", "Method", "NominalVariables",
                       "StrataVariables", "WorkingPrecision"), name)
    for k in ("NominalVariables", "StrataVariables"):
        if k in o and SM.str_or_sym(o[k]) not in ("None", "Automatic"):
            raise NotImplementedError(f"{name}: the {k} option is not supported")
    ties = SM.str_or_sym(o.get("Method", "Breslow"))
    if ties == "Automatic":
        ties = "Breslow"
    if ties == "Exact":
        raise NotImplementedError(f"{name}: the exact marginal likelihood is not supported")
    if ties not in ("Breslow", "Efron"):
        raise ValueError(f"{name}: unknown Method {ties!r}")
    cl, tr = _ci_opts(o, name)
    if isinstance(x, Symbol):
        raise Unevaluated()
    if funs is None:
        evs = _event_data(x, name)
        rows, basis, vs = [[] for _ in evs], [], []
    else:
        if vars_ is None:
            raise TypeError(f"{name}: the variables must be given")
        if not (is_list(x) and len(x.args) == 2):
            raise TypeError(f"{name}: expected {{covariates, event data}}")
        cov, ed = x.args
        evs = _event_data(ed, name)
        vs = _syms(vars_, name)
        if isinstance(cov, Symbol):
            raise Unevaluated()
        if not is_list(cov) or len(cov.args) != len(evs):
            raise ValueError(f"{name}: one covariate value (or row) per subject is required")
        cr = [list(c.args) if is_list(c) else [c] for c in cov.args]
        if any(len(c) != len(vs) for c in cr):
            raise ValueError(f"{name}: each covariate row needs {len(vs)} value(s)")
        cr = [[SM._num(v, name, "covariate") for v in c] for c in cr]
        basis = _funs(funs)
        rows = []
        for c in cr:
            try:
                rows.append([SM.fl(_subst(f, vs, c)) for f in basis])
            except ValueError as e:
                raise ValueError(f"{name}: {e}")
    t = np.array([u for u, _ in evs])
    d = np.array([1 if e else 0 for _, e in evs])
    if d.sum() == 0:
        raise ValueError(f"{name}: there are no events")
    X = np.array(rows, dtype=float).reshape(len(evs), len(basis))
    if basis:
        beta, ll, I = _cox_fit(X, t, d, ties, name)
        ll0 = _cox_loglik(np.zeros(len(basis)), X, t, d, ties)[0]
    else:
        beta, ll, I, ll0 = np.zeros(0), None, np.zeros((0, 0)), None
    a = _assoc(EventData=_ed_expr(evs), Covariates=_L(X) if basis else L(), BasisFunctions=L(*basis),
               Variables=L(*vs), BestFitParameters=_L(beta), Ties=ties,
               ConfidenceLevel=o.get("ConfidenceLevel", sp.Rational(95, 100)), ConfidenceTransform=tr)
    return Expr("CoxModel", a)


_COX_PROPS = ("BestFitParameters", "BasisFunctions", "ParameterStandardErrors", "CovarianceMatrix",
              "InformationMatrix", "ParameterConfidenceIntervals", "RelativeRisk", "RelativeRiskConfidenceIntervals",
              "LogLikelihood", "AIC", "LikelihoodRatioStatistic", "WaldStatistic", "ScoreStatistic",
              "MartingaleResiduals", "CoxSnellResiduals", "DevianceResiduals", "EventTimes", "ObservationTimes",
              "ObservationCount", "EventData", "SF", "CDF", "CHF", "StandardErrors", "MedianSurvival",
              "Properties")


def _cox_state(a):
    edx = _get(a, "EventData")
    t = np.array([SM.fl(S(v)) for v in edx.args[0].args])
    d = np.array([0 if q == 1 else 1 for q in edx.args[1].args])
    beta = _np(_get(a, "BestFitParameters")) if _get(a, "BestFitParameters").args else np.zeros(0)
    X = _np(_get(a, "Covariates")) if _get(a, "Covariates").args else np.zeros((len(t), 0))
    return t, d, beta, X.reshape(len(t), len(beta))


def _cox_x0(a, x0, fname):
    basis = list(_get(a, "BasisFunctions").args)
    vs = list(_get(a, "Variables").args)
    if x0 is None or (is_list(x0) and not x0.args):
        return np.zeros(len(basis))
    vals = list(x0.args) if is_list(x0) else [x0]
    if len(vals) != len(vs):
        raise ValueError(f"{fname}: expected {len(vs)} covariate value(s)")
    if any(isinstance(v, Symbol) for v in vals):
        raise Unevaluated()
    vals = [SM._num(v, fname, "covariate") for v in vals]
    return np.array([SM.fl(_subst(f, vs, vals)) for f in basis])


def _cox_fn(a, form, x0):
    t, d, beta, X = _cox_state(a)
    ets, H, _ = _cox_baseline(X, t, d, beta)
    r = math.exp(float(_cox_x0(a, x0, "CoxModel") @ beta)) if len(beta) else 1.0

    def f(u):
        h = _step(ets, H, u, 0.0) * r
        if form == "CHF":
            return float(h)
        if form == "CDF":
            return float(1 - math.exp(-h))
        return float(math.exp(-h))
    return f


def _cox_property(a, prop, rest):
    name = "CoxModel"
    _, o = opts(rest, ("ConfidenceLevel", "ConfidenceTransform", "ConfidenceRange"), name)
    t, d, beta, X = _cox_state(a)
    p = len(beta)
    ties = _get(a, "Ties")
    if prop == "Properties":
        return L(*_COX_PROPS)
    if prop == "BestFitParameters":
        return _get(a, "BestFitParameters")
    if prop == "BasisFunctions":
        return _get(a, "BasisFunctions")
    if prop == "EventData":
        return _get(a, "EventData")
    if prop == "EventTimes":
        return _tlist(sorted(set(t[d == 1])))
    if prop == "ObservationTimes":
        return _tlist(sorted(set(t)))
    if prop == "ObservationCount":
        return len(t)
    ets, H, V = _cox_baseline(X, t, d, beta)
    if prop in ("MartingaleResiduals", "CoxSnellResiduals", "DevianceResiduals"):
        cs = np.array([_step(ets, H, u, 0.0) for u in t]) * (np.exp(X @ beta) if p else 1.0)
        if prop == "CoxSnellResiduals":
            return _L(cs)
        m = d - cs
        if prop == "MartingaleResiduals":
            return _L(m)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = np.sign(m) * np.sqrt(-2 * (m + np.where(d > 0, d * np.log(np.where(d > 0, d - m, 1)), 0)))
        return _L(dev)
    if prop in ("StandardErrors", "MedianSurvival", "SF", "CDF", "CHF"):
        return ("x0", prop)
    if prop.endswith("Table") or prop.endswith("Bands") or prop.endswith("Plot") or prop in (
            "EqualPrecisionIntervals", "HallWellnerIntervals", "PointwiseIntervals", "MeanSurvival", "BIC",
            "RiskScores", "SchoenfeldResiduals", "ScoreResiduals", "BetaDifferences", "ScaledBetaDifferences",
            "RSquared", "MaxRSquared", "BaselineList", "Strata", "StrataModels", "StrataSummary", "TestTableEntries",
            "ParameterTableEntries", "SurvivalDistribution", "HF", "PDF", "FullEventTable"):
        raise NotImplementedError(f'{name}: the property "{prop}" is not supported')
    if prop not in _COX_PROPS:
        raise ValueError(f'{name}: unknown property "{prop}"')
    if p == 0:
        raise ValueError(f'{name}: "{prop}" needs a model with covariates')
    ll, U0, I0 = _cox_loglik(np.zeros(p), X, t, d, ties)
    llb, _, I = _cox_loglik(beta, X, t, d, ties)
    cov = np.linalg.inv(I)
    se = np.sqrt(np.diag(cov))
    base_o = {"ConfidenceLevel": _get(a, "ConfidenceLevel")}
    base_o.update(o)
    z = _z(SM.fl(SM._num(base_o["ConfidenceLevel"], name, "ConfidenceLevel")))
    if prop == "ParameterStandardErrors":
        return _L(se)
    if prop == "CovarianceMatrix":
        return _L(cov)
    if prop == "InformationMatrix":
        return _L(I)
    if prop == "ParameterConfidenceIntervals":
        return L(*(L(float(b - z * s), float(b + z * s)) for b, s in zip(beta, se)))
    if prop == "RelativeRisk":
        return _L(np.exp(beta))
    if prop == "RelativeRiskConfidenceIntervals":
        return L(*(L(float(math.exp(b - z * s)), float(math.exp(b + z * s))) for b, s in zip(beta, se)))
    if prop == "LogLikelihood":
        return float(llb)
    if prop == "AIC":
        return float(-2 * llb + 2 * p)
    if prop == "LikelihoodRatioStatistic":
        return float(2 * (llb - ll))
    if prop == "WaldStatistic":
        return float(beta @ I @ beta)
    return float(U0 @ np.linalg.solve(I0, U0))   # ScoreStatistic


@fn("CoxModel", area=AREA, status="added",
    notes="CoxModel[<|...|>] is the model from CoxModelFit: cm[t] is the baseline survival Exp[-H0(t)] (Breslow "
          "estimate); cm[\"SF\" | \"CDF\" | \"CHF\"][{x0...}][t] at covariate levels x0 ([] = baseline), "
          "cm[\"StandardErrors\"][x0] (S(t) Sqrt[Sum[d/r^2]] at the event times, no-covariate form), "
          "cm[\"MedianSurvival\"][x0]; cm[\"prop\"] / cm[{props}] give \"BestFitParameters\", "
          "\"ParameterStandardErrors\", \"CovarianceMatrix\", \"InformationMatrix\", "
          "\"ParameterConfidenceIntervals\", \"RelativeRisk\", \"RelativeRiskConfidenceIntervals\", "
          "\"LogLikelihood\" (partial), \"AIC\", \"LikelihoodRatioStatistic\", \"WaldStatistic\", "
          "\"ScoreStatistic\", \"MartingaleResiduals\", \"CoxSnellResiduals\", \"DevianceResiduals\", "
          "\"EventTimes\", \"ObservationTimes\", \"ObservationCount\", \"EventData\", \"Properties\". Formatted "
          "tables, bands, strata, BIC, R^2 and Schoenfeld/score residuals raise NotImplementedError. Check: WL doc "
          "values (cm /@ {1, 2, 3, 4} = {0.778801, 0.558035, 0.338465, 0.124514}, martingale and Cox-Snell "
          "residuals, StandardErrors {0.1947, 0.232515, 0.220292, 0.148565}, SF[{30, 0}][20] = 0.628764) and "
          "statsmodels PHReg.")
def _cox_model(a: Expr) -> Expr:
    if not (isinstance(a, Expr) and a.head == "Association"):
        raise TypeError("CoxModel: expected CoxModel[<|...|>] (use CoxModelFit)")
    raise Unevaluated()


def _cox_x0_prop(a, prop, x0):
    if prop in ("SF", "CDF", "CHF"):
        return ("fn", _cox_fn(a, prop, x0))
    t, d, beta, X = _cox_state(a)
    ets, H, V = _cox_baseline(X, t, d, beta)
    r = math.exp(float(_cox_x0(a, x0, "CoxModel") @ beta)) if len(beta) else 1.0
    if prop == "MedianSurvival":
        for tt, h in zip(ets, H):
            if math.exp(-h * r) <= 0.5 + 1e-12:
                return _tnum(tt)
        return Symbol("Indeterminate")
    if len(beta):
        raise NotImplementedError("CoxModel: StandardErrors for models with covariates are not supported")
    return _L([math.exp(-h) * math.sqrt(v) for h, v in zip(H, V)])


def _cox_apply(head, args, ev_):
    try:
        return _cox_apply1(head, args, ev_)
    except Unevaluated:
        return UNH


def _cox_apply1(head, args, ev_):
    # cm["SF"][x0] -> CoxModel`SF object; cm["SF"][x0][t]
    if isinstance(head, Expr) and isinstance(head.head, Expr) and \
            _model_assoc("CoxModel", head.head.head) is not None and len(head.head.args) == 1 and \
            head.head.args[0] in ("SF", "CDF", "CHF"):
        a = _model_assoc("CoxModel", head.head.head)
        x0 = head.args[0] if head.args else None
        f = _cox_fn(a, head.head.args[0], x0)
        if len(args) != 1:
            raise ValueError("CoxModel: expected one time argument")
        return _time_map(f, args[0], "CoxModel")
    if isinstance(head, Expr) and _model_assoc("CoxModel", head.head) is not None and len(head.args) == 1 and \
            head.args[0] in ("SF", "CDF", "CHF", "StandardErrors", "MedianSurvival"):
        a = _model_assoc("CoxModel", head.head)
        if head.args[0] in ("SF", "CDF", "CHF"):
            if len(args) > 1:
                raise ValueError("CoxModel: expected the covariate levels as one list")
            if args and isinstance(args[0], Symbol):
                raise Unevaluated()
            _cox_x0(a, args[0] if args else None, "CoxModel")
            return UNH
        return _cox_x0_prop(a, head.args[0], args[0] if args else None)
    a = _model_assoc("CoxModel", head)
    if a is None:
        return UNH
    try:
        if args and isinstance(args[0], str):
            if args[0] in ("SF", "CDF", "CHF", "StandardErrors", "MedianSurvival") and len(args) == 1:
                return UNH
            return _cox_property(a, args[0], args[1:])
        if len(args) == 1 and is_list(args[0]) and args[0].args and all(isinstance(q, str) for q in args[0].args):
            if any(q in ("SF", "CDF", "CHF", "StandardErrors", "MedianSurvival") for q in args[0].args):
                raise ValueError('CoxModel: "SF", "CDF", "CHF", "StandardErrors" and "MedianSurvival" take covariate '
                                 'levels; use cm["SF"][{x0}][t]')
            return L(*(_cox_property(a, q, []) for q in args[0].args))
        if len(args) == 1:
            return _time_map(_cox_fn(a, "SF", None), args[0], "CoxModel")
    except KeyError as e:
        raise ValueError(f"CoxModel: malformed model object (missing {e})")
    raise ValueError("CoxModel: expected a time or a property")


KL.COMPOUND_HEAD_HOOKS["CoxModel"] = _cox_apply
KL.DEEP_HEAD_HOOKS["CoxModel"] = _cox_apply
