"""Wolfram batch X02 (part 2): symbolic model specifications (LinearModel, PolynomialModel, ExponentialModel,
LogModel, PowerModel, PeriodicModel, NearestModel), model fitting and selection (ModelFit, ModelFitReport),
kernel and local regression (KernelModelFit, LocalModelFit) and fit plots (ListFitPlot, ListFitPlot3D and the
PlotFit option of ListPlot / ListLinePlot).

Model heads are inert specifications; ``model[x]`` expands them.  Default variables are x[1], x[2], ...; default
parameter names follow the Wolfram documentation (C[0], C[1], ... for polynomials, C[1], C[2], ... for the other
families; C[0], a[i], k[i], phi[i] for periodic models -- the documented Greek phase name is spelled ``phi`` because
the parser has no Greek letters).  Fitting is least squares: exact linear algebra for models linear in their
parameters and exact data, otherwise machine-precision Levenberg-Marquardt from a documented starting-value search,
with a gradient self-check.  Model selection is K-fold cross-validation with the deterministic fold assignment
"point i goes to fold Mod[i - 1, k] + 1" (Wolfram's random folds are not reproduced).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import Unevaluated, fn, L, R, rule, S, is_list, has_float
from . import wl_statmodels as SM
from . import wl_statmodels_fit as FIT

AREA = "statistical models II (X02)"
UNH = K.UNHANDLED
ev = SM.ev

MODEL_HEADS = ("LinearModel", "PolynomialModel", "ExponentialModel", "LogModel", "PowerModel", "PeriodicModel",
               "NearestModel")
LINEAR_KINDS = ("LinearModel", "PolynomialModel", "LogModel")
_SHORT = {"LinearModel": "Linear", "PolynomialModel": "Polynomial", "ExponentialModel": "Exponential",
          "LogModel": "Log", "PowerModel": "Power", "PeriodicModel": "Periodic", "NearestModel": "Nearest"}


def _sym(x, *names):
    return isinstance(x, Symbol) and (not names or x.name in names)


def _int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _C(i):
    return Expr("C", i)


def _V(i):
    return Symbol(f"x02var{i}")


def _P(i):
    return Symbol(f"x02par{i}")


def _is_assoc(x):
    return isinstance(x, Expr) and x.head == "Association"


def _assoc_get(a, key, default=None):
    for r in a.args:
        if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and r.args[0] == key:
            return r.args[1]
    return default


def _is_num(x):
    try:
        v = S(x)
    except Exception:
        return False
    return isinstance(v, sp.Basic) and v.is_number and v.is_real is not False


# ─────────────────────────── specifications ───────────────────────────

class Spec:
    """parsed model specification: head, hyperparameter (single value or family), parameter entries
    [(name, fixed, init)], variables (list or None) and, for LinearModel, the basis"""

    def __init__(self, head, hyper=None, pars=None, vars_=None, basis=None, weights="Uniform", expr=None):
        self.head, self.hyper, self.pars, self.vars, self.basis = head, hyper, pars, vars_, basis
        self.weights, self.expr = weights, expr

    def copy(self, **kw):
        s = Spec(self.head, self.hyper, self.pars, self.vars, self.basis, self.weights, self.expr)
        for k, v in kw.items():
            setattr(s, k, v)
        return s


def _vars(v, fname):
    """variable specification -> list of variables (n, symb or {symb, ...})"""
    if _int(v):
        if v < 1:
            raise ValueError(f"{fname}: the number of variables must be positive")
        return [Expr("x", i + 1) for i in range(v)]
    if is_list(v):
        if not v.args:
            raise ValueError(f"{fname}: the variable list is empty")
        for q in v.args:
            if not (isinstance(q, Symbol) or (isinstance(q, Expr) and not is_list(q) and not _is_num(q))):
                raise TypeError(f"{fname}: invalid variable {q}")
        return list(v.args)
    if isinstance(v, Symbol) or (isinstance(v, Expr) and v.head not in ("List", "Rule") and not _is_num(v)):
        return [v]
    raise TypeError(f"{fname}: invalid variable specification {v}")


def _pars(p, fname):
    """parameter specification {par, val, par -> val, {par, val0}, ...} -> [(name|None, fixed|None, init|None)]"""
    items = list(p.args) if is_list(p) else [p]
    out = []
    for q in items:
        if isinstance(q, Expr) and q.head == "Rule" and len(q.args) == 2:
            if not _is_num(q.args[1]):
                raise TypeError(f"{fname}: the parameter value in {q} must be a real number")
            out.append((q.args[0], S(q.args[1]), None))
        elif is_list(q) and len(q.args) == 2 and not _is_num(q.args[0]):
            if not _is_num(q.args[1]):
                raise TypeError(f"{fname}: the initial value in {q} must be a real number")
            out.append((q.args[0], None, float(S(q.args[1]))))
        elif _is_num(q):
            out.append((None, S(q), None))
        elif isinstance(q, Symbol) or (isinstance(q, Expr) and not is_list(q)):
            out.append((q, None, None))
        else:
            raise TypeError(f"{fname}: invalid parameter specification {q}")
    return out


def _hyper_value(h, fname, key):
    """hyperparameter: Automatic, n, UpTo[n], n ;; m (;; s), a | b | ..., <|key -> ...|>"""
    if _is_assoc(h):
        for r in h.args:
            if not (isinstance(r, Expr) and r.head == "Rule" and r.args[0] in (key,)):
                raise ValueError(f"{fname}: unknown hyperparameter {r.args[0] if isinstance(r, Expr) and r.args else r}")
        h = _assoc_get(h, key, Symbol("Automatic"))
    if _sym(h, "Automatic") or _int(h):
        return h
    if isinstance(h, Expr) and h.head == "UpTo" and len(h.args) == 1 and _int(h.args[0]):
        return h
    if isinstance(h, Expr) and h.head == "Span" and len(h.args) in (2, 3) and all(_int(a) for a in h.args):
        return h
    if isinstance(h, Expr) and h.head == "Alternatives" and all(_int(a) for a in h.args):
        return h
    raise TypeError(f"{fname}: invalid {key} specification {h}")


def _family(h, lo, auto_hi):
    """hyperparameter family -> list of integers (UpTo / Span / Alternatives / Automatic = lo .. auto_hi)"""
    if _int(h):
        return [h]
    if _sym(h, "Automatic"):
        return list(range(lo, max(lo, auto_hi) + 1))
    if h.head == "UpTo":
        return list(range(lo, h.args[0] + 1))
    if h.head == "Span":
        st = h.args[2] if len(h.args) == 3 else 1
        if st == 0:
            raise ValueError("ModelFit: the Span step must be nonzero")
        return list(range(h.args[0], h.args[1] + (1 if st > 0 else -1), st))
    return list(h.args)


def parse(m, fname):
    """Spec of a model expression (TypeError for something that is not a model specification)"""
    if not (isinstance(m, Expr) and m.head in MODEL_HEADS):
        raise TypeError(f"{fname}: {m} is not a model specification")
    h, a = m.head, list(m.args)
    if h == "PolynomialModel":
        if len(a) > 3:
            raise ValueError(f"{fname}: PolynomialModel takes at most 3 arguments")
        hyp = _hyper_value(a[0], fname, "Degree") if a else Symbol("Automatic")
        if _int(hyp) and hyp < 0:
            raise ValueError(f"{fname}: the polynomial degree must be non-negative")
        vs = _vars(a[-1], fname) if len(a) >= 2 else None
        ps = _pars(a[1], fname) if len(a) == 3 else None
        return Spec(h, hyp, ps, vs, expr=m)
    if h == "PeriodicModel":
        if len(a) > 3:
            raise ValueError(f"{fname}: PeriodicModel takes at most 3 arguments")
        hyp = Symbol("Automatic")
        if a:
            h0 = a[0]
            if _is_assoc(h0):
                for r in h0.args:
                    if not (isinstance(r, Expr) and r.head == "Rule" and r.args[0] in ("FrequencyCount",
                                                                                         "SamplePoints")):
                        raise ValueError(f"{fname}: unknown PeriodicModel hyperparameter {r}")
                sp_ = _assoc_get(h0, "SamplePoints", 10)
                if not (_int(sp_) and sp_ >= 1):
                    raise ValueError(f"{fname}: \"SamplePoints\" must be a positive integer")
                hyp = _hyper_value(_assoc_get(h0, "FrequencyCount", Symbol("Automatic")), fname, "FrequencyCount")
            else:
                hyp = _hyper_value(h0, fname, "FrequencyCount")
        if _int(hyp) and hyp < 1:
            raise ValueError(f"{fname}: the number of frequencies must be positive")
        vs = _vars(a[-1], fname) if len(a) >= 2 else None
        if vs is not None and len(vs) != 1:
            raise ValueError(f"{fname}: periodic models have a single variable")
        ps = _pars(a[1], fname) if len(a) == 3 else None
        return Spec(h, hyp, ps, vs, expr=m)
    if h == "NearestModel":
        if a and _is_assoc(a[0]) and _assoc_get(a[0], "TrainingData") is not None:
            return Spec(h, _assoc_get(a[0], "Neighbors"), None, _vars(a[1], fname) if len(a) > 1 else None,
                        weights=_assoc_get(a[0], "Weights", "Uniform"), expr=m)
        if len(a) > 2:
            raise ValueError(f"{fname}: NearestModel takes at most 2 arguments")
        nb, w = Symbol("Automatic"), "Uniform"
        if a:
            h0 = a[0]
            if _is_assoc(h0):
                for r in h0.args:
                    if not (isinstance(r, Expr) and r.head == "Rule" and r.args[0] in ("Neighbors", "Weights")):
                        raise ValueError(f"{fname}: unknown NearestModel hyperparameter {r}")
                nb = _assoc_get(h0, "Neighbors", nb)
                w = _assoc_get(h0, "Weights", w)
            else:
                nb = h0
        if w not in ("Uniform", "Distance"):
            raise ValueError(f'{fname}: the "Weights" hyperparameter must be "Uniform" or "Distance"')
        if not (_sym(nb, "Automatic") or (_int(nb) and nb >= 1) or
                (is_list(nb) and len(nb.args) == 2 and (_int(nb.args[0]) or _sym(nb.args[0], "All")) and
                 _is_num(nb.args[1])) or (isinstance(nb, Expr) and nb.head in ("UpTo", "Span", "Alternatives"))):
            raise TypeError(f"{fname}: invalid \"Neighbors\" specification {nb}")
        if isinstance(nb, Expr) and nb.head in ("UpTo", "Span", "Alternatives"):
            nb = _hyper_value(nb, fname, "Neighbors")
        vs = _vars(a[1], fname) if len(a) == 2 else None
        return Spec(h, nb, None, vs, weights=w, expr=m)
    if h == "LinearModel":
        if len(a) == 0:
            return Spec(h, None, None, None, basis=None, expr=m)
        if len(a) == 1:
            raise ValueError(f"{fname}: LinearModel[basis] needs the variables: LinearModel[basis, vars]")
        if len(a) > 3:
            raise ValueError(f"{fname}: LinearModel takes at most 3 arguments")
        b = a[0]
        if _is_assoc(b):
            b = _assoc_get(b, "Basis")
            if b is None:
                raise ValueError(f'{fname}: the association must give "Basis"')
        basis = list(b.args) if is_list(b) else [b]
        vs = _vars(a[-1], fname)
        ps = _pars(a[1], fname) if len(a) == 3 else None
        if ps is not None:
            names = [q[0] for q in ps if q[0] is not None]
            for f in basis:
                if any(ev(Expr("FreeQ", f, nm)) is not True for nm in names):
                    raise NotImplementedError(f"{fname}: parameters inside the basis functions are not supported")
        return Spec(h, None, ps, vs, basis=basis, expr=m)
    # ExponentialModel, LogModel, PowerModel
    if len(a) > 2:
        raise ValueError(f"{fname}: {h} takes at most 2 arguments")
    vs = _vars(a[-1], fname) if a else None
    ps = _pars(a[0], fname) if len(a) == 2 else None
    return Spec(h, None, ps, vs, expr=m)


def _monomials(d, nv):
    """exponent tuples of total degree <= d, sorted by total degree, max degree, then variable order"""
    out = [e for e in itertools.product(range(d + 1), repeat=nv) if sum(e) <= d]
    out.sort(key=lambda e: (sum(e), max(e) if e else 0, tuple(-x for x in e)))
    return out


def structure(spec, nv, fname):
    """(template Expr in placeholders x02var_i / x02par_j, default parameter names, linear basis or None) for a
    single-valued hyperparameter"""
    h = spec.head
    V = [_V(i) for i in range(nv)]
    if h == "PolynomialModel":
        mons = _monomials(spec.hyper, nv)
        basis = [Expr("Times", *(Expr("Power", V[i], e) for i, e in enumerate(m) if e)) if any(m) else 1
                 for m in mons]
        basis = [ev(b) if isinstance(b, Expr) else b for b in basis]
        names = [_C(i) for i in range(len(basis))]
    elif h == "LinearModel":
        if spec.basis is None:
            basis = list(V)
        else:
            sub = L(*(rule(v, V[i]) for i, v in enumerate(spec.vars)))
            basis = [ev(Expr("ReplaceAll", f, sub)) for f in spec.basis]
        names = [_C(i + 1) for i in range(len(basis))]
    elif h == "LogModel":
        basis = [1] + [Expr("Log", V[i]) for i in range(nv)]
        names = [_C(i + 1) for i in range(nv + 1)]
    elif h == "ExponentialModel":
        names = [_C(i + 1) for i in range(nv + 2)]
        t = Expr("Plus", _P(0), Expr("Times", _P(1), Expr("Exp", Expr("Plus", *(Expr("Times", _P(i + 2), V[i])
                                                                                 for i in range(nv))))))
        return t, names, None
    elif h == "PowerModel":
        names = [_C(i + 1) for i in range(nv + 2)]
        t = Expr("Plus", _P(0), Expr("Times", _P(1), *(Expr("Power", V[i], _P(i + 2)) for i in range(nv))))
        return t, names, None
    elif h == "PeriodicModel":
        if nv != 1:
            raise ValueError(f"{fname}: periodic models have a single variable")
        n = spec.hyper
        names = [_C(0)]
        terms = [_P(0)]
        for i in range(n):
            names += [Expr("a", i + 1), Expr("k", i + 1), Expr("phi", i + 1)]
            terms.append(Expr("Times", _P(3 * i + 1), Expr("Sin", Expr("Plus", Expr("Times", _P(3 * i + 2), V[0]),
                                                                          _P(3 * i + 3)))))
        return Expr("Plus", *terms), names, None
    else:
        raise TypeError(f"{fname}: {h} is not a parametric model")
    t = Expr("Plus", *(Expr("Times", _P(i), b) for i, b in enumerate(basis)))
    return t, names, basis


def _entries(spec, names, fname):
    """[(name, fixed, init)] for every parameter (defaults filled in)"""
    if spec.pars is None:
        return [(nm, None, None) for nm in names]
    if len(spec.pars) != len(names):
        raise ValueError(f"{fname}: {spec.head} needs {len(names)} parameter(s), {len(spec.pars)} given")
    return [(q[0] if q[0] is not None else nm, q[1], q[2]) for q, nm in zip(spec.pars, names)]


def _nv_of(spec):
    if spec.vars is not None:
        return len(spec.vars)
    if spec.head == "LinearModel" and spec.basis is not None:
        return None
    return None


def _single(spec):
    return spec.head in ("LinearModel", "ExponentialModel", "LogModel", "PowerModel") or _int(spec.hyper)


def expression(spec, args, fname, values=None):
    """the model expression at the variable values args (list), parameters by name or value"""
    nv = len(args)
    if spec.vars is not None and len(spec.vars) != nv:
        raise ValueError(f"{fname}: the model has {len(spec.vars)} variable(s), {nv} given")
    t, names, _ = structure(spec, nv, fname)
    ent = _entries(spec, names, fname)
    vals = values if values is not None else [R(f) if f is not None else nm for nm, f, _ in ent]
    sub = [rule(_P(i), v) for i, v in enumerate(vals)] + [rule(_V(i), a) for i, a in enumerate(args)]
    return ev(Expr("ReplaceAll", t, L(*sub)))


def _model_apply(head, args, ev_):
    """model[x], model[{x, y}], model[{{x1, y1}, ...}]"""
    if not (isinstance(head, Expr) and head.head in MODEL_HEADS):
        return UNH
    fname = head.head
    if len(args) != 1:
        return UNH
    try:
        spec = parse(head, fname)
    except (TypeError, ValueError):
        return UNH
    if spec.head == "NearestModel":
        return _nearest_apply(spec, args[0], ev_)
    if not _single(spec):
        return UNH
    x = args[0]
    nvk = len(spec.vars) if spec.vars is not None else None
    if is_list(x) and x.args and all(is_list(r) for r in x.args):
        return L(*(_model_apply(head, [r], ev_) for r in x.args))
    if is_list(x):
        if nvk == 1 or (nvk is None and spec.head == "PeriodicModel"):
            return L(*(expression(spec, [v], fname) for v in x.args))
        if not x.args:
            return UNH
        return expression(spec, list(x.args), fname)
    return expression(spec, [x], fname)


for _h in MODEL_HEADS:
    KL.COMPOUND_HEAD_HOOKS[_h] = _model_apply


def _model_notes(h, form, extra=""):
    return (f"{h} is an inert model specification for ModelFit and PlotFit; {h}[...][x] (or [{{x, y, ...}}], a "
            f"list of points) expands it: {form} Variables: n (x[1], ..., x[n]), a symbol or a list; parameters "
            "{par, val, par -> val, {par, val0}, ...} (fixed values are kept fixed in ModelFit, val0 starts the "
            f"fit). {extra}Information[...] properties are not supported. Check: expansions and fits compared "
            "with hand-computed expressions and numpy / statsmodels least squares (tests).")


@fn("LinearModel", area=AREA, status="added",
    notes=_model_notes("LinearModel", "Sum[C[i] f_i] over the basis {f1, ...} (an association <|\"Basis\" -> "
                       "...|> is accepted); LinearModel[] is the linear combination C[1] x[1] + ... of the input "
                       "features (no constant term, as documented).",
                       "Parameters that appear inside the basis functions raise NotImplementedError. "))
def _linear_model(*args: Expr) -> Expr:
    parse(Expr("LinearModel", *args), "LinearModel")
    raise Unevaluated()


@fn("PolynomialModel", area=AREA, status="added",
    notes=_model_notes("PolynomialModel", "Sum[C[i] m_i] over the monomials of total degree <= n sorted by total "
                       "degree, maximum degree and variable order, parameters C[0], C[1], ... (the documented "
                       "c_i x^i indexing).", "The degree may be n, UpTo[n], n ;; m (;; s), n | m | ... or "
                       "Automatic (a family that ModelFit selects from; Automatic = degrees 0 .. Min[5, m - 1] for "
                       "the smallest training set size m -- a stated choice). "))
def _polynomial_model(*args: Expr) -> Expr:
    parse(Expr("PolynomialModel", *args), "PolynomialModel")
    raise Unevaluated()


@fn("ExponentialModel", area=AREA, status="added",
    notes=_model_notes("ExponentialModel", "C[1] + C[2] Exp[C[3] x] (multivariate C[1] + C[2] Product[Exp[C[i] "
                       "x_(i-2)], {i, 3, n + 2}]).", "ModelFit starts from a grid search over C[3] (one variable) "
                       "or a log-linear regression (several), then Levenberg-Marquardt. "))
def _exponential_model(*args: Expr) -> Expr:
    parse(Expr("ExponentialModel", *args), "ExponentialModel")
    raise Unevaluated()


@fn("LogModel", area=AREA, status="added",
    notes=_model_notes("LogModel", "C[1] + C[2] Log[x] (multivariate C[1] + Sum[C[i + 1] Log[x_i]]).", ""))
def _log_model(*args: Expr) -> Expr:
    parse(Expr("LogModel", *args), "LogModel")
    raise Unevaluated()


@fn("PowerModel", area=AREA, status="added",
    notes=_model_notes("PowerModel", "C[1] + C[2] x^C[3] (multivariate C[1] + C[2] Product[x_i^C[i + 2]]).",
                       "ModelFit needs positive variable values and starts from a grid search over the exponent "
                       "(one variable) or a log-log regression (several), then Levenberg-Marquardt. "))
def _power_model(*args: Expr) -> Expr:
    parse(Expr("PowerModel", *args), "PowerModel")
    raise Unevaluated()


@fn("PeriodicModel", area=AREA, status="added",
    notes=_model_notes("PeriodicModel", "C[0] + Sum[a[i] Sin[k[i] x + phi[i]], {i, n}] with the parameter order "
                       "{C[0], a[1], k[1], phi[1], a[2], ...} (Wolfram's phase symbol is spelled phi).",
                       "PeriodicModel[n], PeriodicModel[<|\"FrequencyCount\" -> spec, \"SamplePoints\" -> s|>, "
                       "x]; Automatic means one frequency (documented). ModelFit finds starting values one "
                       "frequency at a time by a least-squares periodogram search on Max[10 s, 200] frequencies "
                       "between 2 Pi/(2 range) and the Nyquist frequency of the smallest spacing, then refines all "
                       "parameters by Levenberg-Marquardt. "))
def _periodic_model(*args: Expr) -> Expr:
    parse(Expr("PeriodicModel", *args), "PeriodicModel")
    raise Unevaluated()


@fn("NearestModel", area=AREA, status="added",
    notes="NearestModel[], NearestModel[k] and NearestModel[<|\"Neighbors\" -> spec, \"Weights\" -> w|>, vars] "
          "specify k-nearest-neighbour regression (Euclidean distance; the prediction is the mean of the "
          "neighbours' responses, or the inverse-distance weighted mean for \"Weights\" -> \"Distance\" -- an exact "
          "match returns the mean of the matching responses). \"Neighbors\": k, {k, d} (at most k within distance "
          "d), {All, d}, UpTo / Span / Alternatives families or Automatic (the family 1 .. Min[10, m], m the "
          "smallest training set, selected by ModelFit's cross-validation -- a stated choice). ModelFit returns "
          "NearestModel[<|\"Neighbors\" -> k, \"Weights\" -> w, \"TrainingData\" -> {{x.., y}, ...}|>, vars], "
          "which evaluates at numeric points; symbolic points leave it unevaluated. A point with no neighbour "
          "within d raises ValueError. Check: predictions compared with a hand-computed k-NN average (tests).")
def _nearest_model(*args: Expr) -> Expr:
    parse(Expr("NearestModel", *args), "NearestModel")
    raise Unevaluated()


# ─────────────────────────── data ───────────────────────────

def fit_data(data, fname):
    """(X rows (sympy numbers), y (sympy numbers)) from the documented data forms"""
    if isinstance(data, Symbol):
        raise Unevaluated()
    if isinstance(data, Expr) and data.head == "Tabular":
        raise NotImplementedError(f"{fname}: Tabular data is not supported")
    if isinstance(data, Expr) and data.head == "Rule" and len(data.args) == 2:
        lhs, rhs = data.args
        if isinstance(lhs, Expr) and lhs.head == "Tabular":
            raise NotImplementedError(f"{fname}: Tabular data is not supported")
        if _int(rhs) and SM.is_matrix(lhs):
            rows = SM.mat(lhs, fname)
            j = rhs - 1 if rhs > 0 else len(rows[0]) + rhs
            if not 0 <= j < len(rows[0]):
                raise ValueError(f"{fname}: column {rhs} does not exist")
            return [[v for i, v in enumerate(r) if i != j] for r in rows], [r[j] for r in rows]
        if is_list(lhs) and is_list(rhs):
            if len(lhs.args) != len(rhs.args):
                raise ValueError(f"{fname}: the inputs and responses must have the same length")
            y = SM.vec(rhs, fname, "response")[0]
            if SM.is_matrix(lhs):
                return [list(r) for r in SM.mat(lhs, fname)], y
            return [[v] for v in SM.vec(lhs, fname, "input")[0]], y
        raise TypeError(f"{fname}: invalid data {data}")
    if not is_list(data) or not data.args:
        raise TypeError(f"{fname}: the data must be a non-empty list")
    if all(isinstance(r, Expr) and r.head == "Rule" and len(r.args) == 2 for r in data.args):
        X, y = [], []
        for r in data.args:
            xv = r.args[0]
            X.append(list(SM.vec(xv, fname, "input")[0]) if is_list(xv) else [SM._num(xv, fname, "input")])
            y.append(SM._num(r.args[1], fname, "response"))
        if len({len(q) for q in X}) != 1:
            raise ValueError(f"{fname}: the inputs must all have the same length")
        return X, y
    if all(not is_list(v) for v in data.args):
        for v in data.args:
            if not _is_num(v):
                if isinstance(v, Symbol) or (isinstance(v, Expr) and v.head not in ("Rule",)):
                    raise Unevaluated()
        y = SM.vec(data, fname, "response")[0]
        return [[sp.Integer(i + 1)] for i in range(len(y))], y
    if not SM.is_matrix(data):
        raise TypeError(f"{fname}: the data must be a vector, a rectangular matrix or a list of rules")
    for r in data.args:
        for v in r.args:
            if not _is_num(v):
                raise Unevaluated() if isinstance(v, Symbol) else TypeError(f"{fname}: non-numeric data value {v}")
    rows = SM.mat(data, fname)
    if len(rows[0]) < 2:
        raise ValueError(f"{fname}: each data point needs at least one input and a response")
    return [list(r[:-1]) for r in rows], [r[-1] for r in rows]


def _canon(X, y):
    return L(*(L(*(R(v) for v in xr), R(yv)) for xr, yv in zip(X, y)))


# ─────────────────────────── fitting ───────────────────────────

class Fit:
    """a fitted parametric model: spec (single-valued), nv, parameter names, values (sympy or float), template"""

    def __init__(self, spec, nv, names, values, template, basis, free, J=None):
        self.spec, self.nv, self.names, self.values = spec, nv, names, values
        self.template, self.basis, self.free, self.J = template, basis, free, J
        self._f = None

    def predict(self, Xa):
        if self._f is None:
            sub = L(*(rule(_P(i), R(v) if isinstance(v, sp.Basic) else float(v)) for i, v in enumerate(self.values)))
            e = S(ev(Expr("ReplaceAll", self.template, sub)))
            xs = [sp.Symbol(f"x02var{i}") for i in range(self.nv)]
            e = e.subs({s: sp.Symbol(str(s)) for s in e.free_symbols}) if isinstance(e, sp.Basic) else sp.Float(e)
            self._f = sp.lambdify(xs, e, "numpy")
        Xa = np.asarray(Xa, dtype=float)
        with np.errstate(all="ignore"):
            out = self._f(*Xa.T)
        return np.broadcast_to(np.asarray(out, dtype=complex if np.iscomplexobj(out) else float),
                               (Xa.shape[0],)).real.astype(float)

    def nparams(self):
        return sum(self.free)

    def model_expr(self):
        """the fitted model as the same head with par -> value rules"""
        sp_ = self.spec
        vals = [R(v) if isinstance(v, sp.Basic) else float(v) for v in self.values]
        prs = L(*(rule(nm, v) for nm, v in zip(self.names, vals)))
        vs = L(*(sp_.vars if sp_.vars is not None else [Expr("x", i + 1) for i in range(self.nv)]))
        h = sp_.head
        if h == "PolynomialModel":
            return Expr(h, sp_.hyper, prs, vs)
        if h == "PeriodicModel":
            return Expr(h, sp_.hyper, prs, vs)
        if h == "LinearModel":
            if sp_.basis is None:
                return Expr(h, L(*vs.args), prs, vs)
            return Expr(h, L(*sp_.basis), prs, vs)
        return Expr(h, prs, vs)

    def variables(self):
        return list(self.spec.vars) if self.spec.vars is not None else [Expr("x", i + 1) for i in range(self.nv)]

    def expr(self):
        vals = [R(v) if isinstance(v, sp.Basic) else float(v) for v in self.values]
        return expression(self.spec, self.variables(), "ModelFit", vals)


def _sym_template(t, nv, npar):
    e = S(t)
    e = e.subs({s: sp.Symbol(str(s)) for s in e.free_symbols})
    return e, [sp.Symbol(f"x02var{i}") for i in range(nv)], [sp.Symbol(f"x02par{i}") for i in range(npar)]


def _lin_solve(A, y, exact):
    if exact:
        M = sp.Matrix(A)
        if M.rank() < M.cols:
            return None
        return list((M.T * M).LUsolve(M.T * sp.Matrix(y)))
    A = np.asarray(A, dtype=float)
    if A.shape[1] and np.linalg.matrix_rank(A) < A.shape[1]:
        return None
    return list(np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)[0]) if A.shape[1] else []


def _grid(lo, hi, n=161, exclude_zero=True):
    g = np.linspace(lo, hi, n)
    return g[np.abs(g) > 1e-12] if exclude_zero else g


def fit_one(spec, X, y, fname, exact=None, strict=True):
    """least-squares fit of a single-valued parametric spec to X (rows), y; returns Fit or None (failure when
    strict is False)"""
    nv = len(X[0])
    if spec.vars is not None and len(spec.vars) != nv:
        raise ValueError(f"{fname}: the model has {len(spec.vars)} variable(s) but the data {nv} input(s)")
    if spec.head == "LinearModel" and spec.basis is None and spec.vars is None:
        spec = spec.copy()
    t, names, basis = structure(spec, nv, fname)
    ent = _entries(spec, names, fname)
    names = [q[0] for q in ent]
    fixed = [q[1] for q in ent]
    inits = [q[2] for q in ent]
    free = [f is None for f in fixed]
    n, npar = len(y), len(names)
    if exact is None:
        exact = not any(isinstance(v, sp.Float) for r in X for v in r) and not any(isinstance(v, sp.Float) for v in y)
    exact = exact and not any(f is not None and isinstance(f, sp.Float) for f in fixed)
    if sum(free) > n:
        if strict:
            raise ValueError(f"{fname}: fewer data points ({n}) than free parameters ({sum(free)})")
        return None
    e, xs, ps = _sym_template(t, nv, npar)
    Xa = np.array([[float(v) for v in r] for r in X])
    ya = np.array([float(v) for v in y])
    if basis is not None:
        bsy = [sp.sympify(S(b)) if not isinstance(b, int) else sp.Integer(b) for b in basis]
        bsy = [b.subs({s: sp.Symbol(str(s)) for s in b.free_symbols}) for b in bsy]
        if exact:
            cols = [[b.subs(dict(zip(xs, r))) for r in X] for b in bsy]
            if any(not (v.is_number and v.is_real) for c in cols for v in c):
                if strict:
                    raise ValueError(f"{fname}: a basis function is not real at the data points")
                return None
        else:
            with np.errstate(all="ignore"):
                cols = [np.broadcast_to(np.asarray(sp.lambdify(xs, b, "numpy")(*Xa.T), dtype=complex), (n,))
                        for b in bsy]
            if any(np.any(~np.isfinite(c)) or np.any(np.abs(c.imag) > 1e-12 * (1 + np.abs(c.real))) for c in cols):
                if strict:
                    raise ValueError(f"{fname}: a basis function is not real at the data points (e.g. Log of a "
                                     "non-positive value)")
                return None
            cols = [c.real for c in cols]
        if exact:
            yy = [y[i] - sum(fixed[j] * cols[j][i] for j in range(npar) if fixed[j] is not None) for i in range(n)]
            A = [[cols[j][i] for j in range(npar) if free[j]] for i in range(n)]
        else:
            yy = ya - sum((float(fixed[j]) * cols[j] for j in range(npar) if fixed[j] is not None), np.zeros(n))
            A = np.column_stack([cols[j] for j in range(npar) if free[j]]) if any(free) else np.zeros((n, 0))
        sol = _lin_solve(A, yy, exact)
        if sol is None:
            if strict:
                raise ValueError(f"{fname}: the design matrix is rank deficient (too few distinct points for "
                                 f"{sum(free)} parameter(s))")
            return None
        it = iter(sol)
        values = [fixed[j] if fixed[j] is not None else (sp.nsimplify(next(it)) if exact else float(next(it)))
                  for j in range(npar)]
        if exact:
            values = [sp.simplify(v) for v in values]
        J = np.column_stack([np.asarray(cols[j], dtype=float) for j in range(npar) if free[j]]) if any(free) \
            else np.zeros((n, 0))
        return Fit(spec, nv, names, values, t, basis, free, J)
    # nonlinear: Levenberg-Marquardt from a starting-value search
    F = sp.lambdify(ps + xs, e, "numpy")
    Js = [sp.lambdify(ps + xs, sp.diff(e, p), "numpy") for p in ps]
    fidx = [j for j in range(npar) if free[j]]
    if spec.head == "PowerModel" and np.any(Xa <= 0):
        if strict:
            raise ValueError(f"{fname}: PowerModel needs positive variable values")
        return None

    def full(theta):
        out = [float(fixed[j]) if fixed[j] is not None else 0.0 for j in range(npar)]
        for k, j in enumerate(fidx):
            out[j] = theta[k]
        return out

    def f(theta):
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(F(*full(theta), *Xa.T), dtype=float), (n,)).astype(float)

    def jac(theta):
        with np.errstate(all="ignore"):
            return np.column_stack([np.broadcast_to(np.asarray(Js[j](*full(theta), *Xa.T), dtype=float), (n,))
                                    for j in fidx]) if fidx else np.zeros((n, 0))
    starts = _starts(spec, nv, Xa, ya, fixed, inits, fname)
    best = None
    for st in starts:
        theta0 = [st[j] for j in fidx]
        with np.errstate(all="ignore"):
            beta, rss = FIT._lm_solve(f, jac, theta0, ya, np.ones(n))
        if np.all(np.isfinite(beta)) and np.isfinite(rss) and (best is None or rss < best[1]):
            best = (beta, rss)
    if best is None:
        if strict:
            raise ArithmeticError(f"{fname}: the fit did not converge")
        return None
    beta = best[0]
    Jm = jac(beta)
    g = Jm.T @ (ya - f(beta))
    if not np.all(np.isfinite(Jm)) or np.linalg.norm(g) > 1e-6 * max(1.0, np.linalg.norm(Jm.T @ ya)):
        if strict:
            raise ArithmeticError(f"{fname}: the fit did not converge (gradient {np.linalg.norm(g):.3g})")
        return None
    values = [fixed[j] if fixed[j] is not None else float(v) for j, v in zip(range(npar), full(beta))]
    values = [fixed[j] if fixed[j] is not None else values[j] for j in range(npar)]
    return Fit(spec, nv, names, values, t, None, free, Jm)


def _ls2(cols, y):
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    return c, float(r @ r)


def _starts(spec, nv, Xa, ya, fixed, inits, fname):
    """starting parameter vectors for the nonlinear families (documented search; user initial values win)"""
    npar = len(fixed)
    base = [float(fixed[j]) if fixed[j] is not None else (inits[j] if inits[j] is not None else 1.0)
            for j in range(npar)]
    h = spec.head
    out = []
    if h in ("ExponentialModel", "PowerModel"):
        xin = Xa if h == "ExponentialModel" else np.log(Xa)
        if nv == 1 and fixed[2] is None and inits[2] is None:
            rng = float(np.ptp(xin[:, 0])) or 1.0
            cands = []
            for c in _grid(-10.0 / rng, 10.0 / rng, 201):
                z = np.exp(c * xin[:, 0])
                if not np.all(np.isfinite(z)) or np.ptp(z) == 0:
                    continue
                if fixed[0] is None and fixed[1] is None:
                    ab, rss = _ls2([np.ones(len(z)), z], ya)
                    cands.append((rss, [ab[0], ab[1], c]))
                elif fixed[0] is None:
                    ab, rss = _ls2([np.ones(len(z))], ya - float(fixed[1]) * z)
                    cands.append((rss, [ab[0], float(fixed[1]), c]))
                elif fixed[1] is None:
                    ab, rss = _ls2([z], ya - float(fixed[0]))
                    cands.append((rss, [float(fixed[0]), ab[0], c]))
            cands.sort(key=lambda q: q[0])
            out += [q[1] for q in cands[:3]]
        elif nv > 1 and np.all(ya > 0):
            A = np.column_stack([np.ones(len(ya))] + [xin[:, i] for i in range(nv)])
            c, *_ = np.linalg.lstsq(A, np.log(ya), rcond=None)
            st = [0.0, float(np.exp(c[0]))] + [float(v) for v in c[1:]]
            out.append([float(fixed[j]) if fixed[j] is not None else (inits[j] if inits[j] is not None else st[j])
                        for j in range(npar)])
        out.append(base)
    elif h == "PeriodicModel":
        s = spec.expr
        sp_pts = 10
        if s is not None and s.args and _is_assoc(s.args[0]):
            sp_pts = _assoc_get(s.args[0], "SamplePoints", 10)
        x = Xa[:, 0]
        xs_ = np.sort(np.unique(x))
        rng = float(np.ptp(x)) or 1.0
        dmin = float(np.min(np.diff(xs_))) if len(xs_) > 1 else 1.0
        freqs = np.linspace(2 * math.pi / (2 * rng), math.pi / dmin, max(10 * sp_pts, 200))
        st = list(base)
        resid = ya - (np.mean(ya) if fixed[0] is None else float(fixed[0]))
        st[0] = float(np.mean(ya)) if fixed[0] is None else float(fixed[0])
        for i in range(spec.hyper):
            ka = 3 * i + 2
            if fixed[ka] is not None or inits[ka] is not None:
                kk = [float(fixed[ka]) if fixed[ka] is not None else inits[ka]]
            else:
                kk = freqs
            bestq = None
            for k in kk:
                cols = [np.sin(k * x), np.cos(k * x)]
                try:
                    ab, rss = _ls2(cols, resid)
                except np.linalg.LinAlgError:
                    continue
                if bestq is None or rss < bestq[0]:
                    bestq = (rss, k, ab)
            if bestq is None:
                break
            _, k, (A_, B_) = bestq
            amp, ph = math.hypot(A_, B_), math.atan2(B_, A_)
            st[3 * i + 1] = amp if fixed[3 * i + 1] is None else st[3 * i + 1]
            st[ka] = k if fixed[ka] is None else st[ka]
            st[3 * i + 3] = ph if fixed[3 * i + 3] is None else st[3 * i + 3]
            resid = resid - (A_ * np.sin(k * x) + B_ * np.cos(k * x))
        for j in range(npar):
            if inits[j] is not None and fixed[j] is None and j % 3 != 2:
                st[j] = inits[j]
        out.append(st)
    else:
        out.append(base)
    return out


# ─────────────────────────── nearest neighbours ───────────────────────────

def knn_predict(Xtr, ytr, Xq, nb, weights, fname):
    Xtr = np.asarray(Xtr, dtype=float)
    ytr = np.asarray(ytr, dtype=float)
    out = []
    for q in np.asarray(Xq, dtype=float):
        d = np.sqrt(np.sum((Xtr - q) ** 2, axis=1))
        order = np.argsort(d, kind="stable")
        if isinstance(nb, tuple):
            k, rad = nb
            idx = [i for i in order if d[i] <= rad]
            if k is not None:
                idx = idx[:k]
        else:
            idx = list(order[:nb])
        if not idx:
            raise ValueError(f"{fname}: no training point lies within the neighbour distance of {list(q)}")
        dd, yy = d[idx], ytr[idx]
        if weights == "Distance":
            if np.any(dd == 0):
                out.append(float(np.mean(yy[dd == 0])))
            else:
                w = 1 / dd
                out.append(float(np.sum(w * yy) / np.sum(w)))
        else:
            out.append(float(np.mean(yy)))
    return np.array(out)


def _nb_spec(nb, fname):
    if _int(nb):
        return nb
    if is_list(nb):
        k = None if _sym(nb.args[0], "All") else nb.args[0]
        return (k, float(S(nb.args[1])))
    raise ValueError(f"{fname}: invalid \"Neighbors\" value {nb}")


def _nb_expr(nb):
    if _int(nb):
        return nb
    return L(Symbol("All") if nb[0] is None else nb[0], nb[1])


class KNNFit:
    def __init__(self, spec, nv, nb, weights, X, y):
        self.spec, self.nv, self.nb, self.weights, self.X, self.y = spec, nv, nb, weights, X, y
        self.free = []
        self.names, self.values = [], []

    def predict(self, Xa):
        return knn_predict(np.array([[float(v) for v in r] for r in self.X]), np.array([float(v) for v in self.y]),
                           Xa, self.nb, self.weights, "NearestModel")

    def nparams(self):
        return 0

    def variables(self):
        return list(self.spec.vars) if self.spec.vars is not None else [Expr("x", i + 1) for i in range(self.nv)]

    def model_expr(self):
        a = Expr("Association", rule("Neighbors", _nb_expr(self.nb)), rule("Weights", self.weights),
                 rule("TrainingData", _canon(self.X, self.y)))
        return Expr("NearestModel", a, L(*self.variables()))

    def expr(self):
        raise ValueError('ModelFit: "Expression" is not available for nearest-neighbour models')


def _nearest_apply(spec, x, ev_):
    a = spec.expr.args[0] if spec.expr.args else None
    if not (a is not None and _is_assoc(a) and _assoc_get(a, "TrainingData") is not None):
        return UNH
    X, y = fit_data(_assoc_get(a, "TrainingData"), "NearestModel")
    nv = len(X[0])
    pts = [x] if not is_list(x) or not x.args or not is_list(x.args[0]) else list(x.args)
    single = not (is_list(x) and x.args and is_list(x.args[0]))
    rows = []
    for p in pts:
        vals = list(p.args) if is_list(p) else [p]
        if len(vals) != nv:
            if nv == 1 and is_list(p) and single:
                return L(*(_nearest_apply(spec, v, ev_) for v in p.args))
            raise ValueError(f"NearestModel: expected {nv} input value(s)")
        if not all(_is_num(v) for v in vals):
            return UNH
        rows.append([float(S(v)) for v in vals])
    out = knn_predict([[float(v) for v in r] for r in X], [float(v) for v in y], rows,
                      _nb_spec(spec.hyper, "NearestModel"), spec.weights, "NearestModel")
    return float(out[0]) if single else L(*(float(v) for v in out))


def fit_knn(spec, X, y, nb, fname):
    return KNNFit(spec, len(X[0]), _nb_spec(nb, fname), spec.weights, X, y)


# ─────────────────────────── ModelFit ───────────────────────────

_NAMED = {"Linear": ("PolynomialModel", 1), "Quadratic": ("PolynomialModel", 2), "Cubic": ("PolynomialModel", 3),
          "Exponential": ("ExponentialModel", None), "Log": ("LogModel", None), "Power": ("PowerModel", None)}
_CRIT = ("RootMeanSquaredError", "MeanSquaredError", "MeanAbsoluteError", "MedianAbsoluteError")
MF_OPTS = ("CriterionFunction", "ValidationSet", "Weights", "Method", "PerformanceGoal", "RandomSeeding",
           "TimeConstraint")


def named_model(s, fname):
    if s not in _NAMED:
        raise NotImplementedError(f'{fname}: the named model "{s}" is not supported (use "Linear", "Quadratic", '
                                  '"Cubic", "Exponential", "Log", "Power" or a model specification)')
    h, d = _NAMED[s]
    return Expr(h, d) if d is not None else Expr(h)


def _loss(res, crit):
    a = np.abs(np.asarray(res, dtype=float))
    if crit == "RootMeanSquaredError":
        return float(np.sqrt(np.mean(a ** 2)))
    if crit == "MeanSquaredError":
        return float(np.mean(a ** 2))
    if crit == "MeanAbsoluteError":
        return float(np.mean(a))
    return float(np.median(a))


def _folds(n, vs, fname):
    """list of (train indices, test indices)"""
    if _sym(vs, "Automatic"):
        vs = Expr("Rule", "KFold", 5)
    if vs == "LeaveOneOut":
        return [([j for j in range(n) if j != i], [i]) for i in range(n)]
    if vs == "KFold":
        vs = Expr("Rule", "KFold", 5)
    if isinstance(vs, Expr) and vs.head == "Rule" and vs.args[0] == "KFold":
        k = vs.args[1]
        if not (_int(k) and k >= 2):
            raise ValueError(f"{fname}: \"KFold\" needs an integer >= 2")
        k = min(k, n)
        return [([j for j in range(n) if j % k != f], [j for j in range(n) if j % k == f]) for f in range(k)]
    if isinstance(vs, Expr) and vs.head == "Rule" and vs.args[0] == "LeavePOut":
        p = vs.args[1]
        if not (_int(p) and 1 <= p < n):
            raise ValueError(f"{fname}: \"LeavePOut\" needs an integer between 1 and n - 1")
        if math.comb(n, p) > 5000:
            raise NotImplementedError(f"{fname}: \"LeavePOut\" with more than 5000 splits is not supported")
        return [([j for j in range(n) if j not in c], list(c)) for c in itertools.combinations(range(n), p)]
    raise NotImplementedError(f"{fname}: ValidationSet -> {vs} is not supported (Automatic, \"KFold\" -> k, "
                              "\"LeaveOneOut\", \"LeavePOut\" -> p, None or explicit validation data)")


def candidates(m, fname, mmin, nv):
    """expand a model specification (or a list of them) into single-valued Specs"""
    if isinstance(m, str):
        m = named_model(m, fname)
    if is_list(m):
        out = []
        for q in m.args:
            out += candidates(q, fname, mmin, nv)
        return out
    if isinstance(m, Expr) and m.head in ("Alternatives",) and all(isinstance(q, Expr) for q in m.args):
        return candidates(L(*m.args), fname, mmin, nv)
    spec = parse(m, fname)
    if spec.head == "PolynomialModel" and not _int(spec.hyper):
        return [spec.copy(hyper=d) for d in _family(spec.hyper, 0, min(5, mmin - 1))]
    if spec.head == "PeriodicModel" and not _int(spec.hyper):
        return [spec.copy(hyper=d) for d in _family(spec.hyper, 1, 1)]
    if spec.head == "NearestModel":
        nb = spec.hyper
        if _sym(nb, "Automatic") or (isinstance(nb, Expr) and nb.head in ("UpTo", "Span", "Alternatives")):
            return [spec.copy(hyper=k) for k in _family(nb, 1, min(10, mmin))]
        return [spec]
    return [spec]


def _fit_any(spec, X, y, fname, strict=True, exact=None):
    if spec.head == "NearestModel":
        return fit_knn(spec, X, y, spec.hyper, fname)
    return fit_one(spec, X, y, fname, exact=exact, strict=strict)


def _label(spec):
    h = spec.head
    if h == "NearestModel":
        return Expr(h, _nb_expr(_nb_spec(spec.hyper, "ModelFit")) if not _sym(spec.hyper) else spec.hyper)
    if h in ("PolynomialModel", "PeriodicModel"):
        return Expr(h, spec.hyper)
    return spec.expr if spec.expr is not None else Expr(h)


def model_fit(data, mspec, fname, o):
    X, y = fit_data(data, fname)
    n = len(y)
    crit = o.get("CriterionFunction", Symbol("Automatic"))
    crit = "RootMeanSquaredError" if _sym(crit, "Automatic") else crit
    if crit not in _CRIT:
        raise NotImplementedError(f"{fname}: CriterionFunction -> {crit} is not supported ({', '.join(_CRIT)})")
    for k in ("Weights", "Method"):
        if k in o and not _sym(o[k], "Automatic", "None"):
            raise NotImplementedError(f"{fname}: the {k} option is not supported")
    vs = o.get("ValidationSet", Symbol("Automatic"))
    explicit_val = None
    if is_list(vs) or (isinstance(vs, Expr) and vs.head == "Rule" and vs.args[0] not in ("KFold", "LeavePOut")):
        explicit_val = fit_data(vs, fname)
        folds = None
    elif _sym(vs, "None"):
        folds = None
    else:
        folds = _folds(n, vs, fname)
    mmin = min(len(tr) for tr, _ in folds) if folds else n
    cands = candidates(mspec, fname, mmin, len(X[0]))
    if not cands:
        raise ValueError(f"{fname}: no candidate models")
    Xa = np.array([[float(v) for v in r] for r in X])
    ya = np.array([float(v) for v in y])
    single = len(cands) == 1
    cv = []
    for c in cands:
        if single:
            cv.append(None)
            continue
        if explicit_val is not None:
            ft = _fit_any(c, X, y, fname, strict=False)
            Xv = np.array([[float(v) for v in r] for r in explicit_val[0]])
            yv = np.array([float(v) for v in explicit_val[1]])
            losses = [_loss(yv - ft.predict(Xv), crit)] if ft is not None else [math.inf]
        elif folds is None:
            ft = _fit_any(c, X, y, fname, strict=False)
            losses = [_loss(ya - ft.predict(Xa), crit)] if ft is not None else [math.inf]
        else:
            losses = []
            for tr, te in folds:
                try:
                    ft = _fit_any(c, [X[i] for i in tr], [y[i] for i in tr], fname, strict=False, exact=False)
                    if c.head == "NearestModel" and isinstance(ft.nb, int) and ft.nb > len(tr):
                        ft = None
                    losses.append(_loss(ya[te] - ft.predict(Xa[te]), crit) if ft is not None else math.inf)
                except (ValueError, ArithmeticError, ZeroDivisionError):
                    losses.append(math.inf)
            losses = [v if np.isfinite(v) else math.inf for v in losses]
        cv.append(losses)
    if single:
        order = [0]
    else:
        means = [float(np.mean(ls)) for ls in cv]
        if all(not np.isfinite(m) for m in means):
            raise ValueError(f"{fname}: no candidate model could be fitted in cross-validation")
        order = sorted(range(len(cands)), key=lambda i: (means[i], i))
    best = _fit_any(cands[order[0]], X, y, fname, strict=True)
    return dict(X=X, y=y, data=_canon(X, y), cands=cands, cv=cv, order=order, best=best, crit=crit,
                vs=vs, folds=folds, fname=fname)


def _fm_object(r):
    """FittedModel[<|...|>] (W18 object) for a parametric best fit"""
    ft = r["best"]
    if isinstance(ft, KNNFit):
        raise ValueError('ModelFit: "FittedModel" is not available for nearest-neighbour models')
    vs = ft.variables()
    names = [nm for nm, fr in zip(ft.names, ft.free) if fr]
    vals = [v for v, fr in zip(ft.values, ft.free) if fr]
    best = ft.expr()
    conf = sp.Rational(95, 100)
    exact_lin = ft.basis is not None and all(ft.free) and all(isinstance(v, sp.Basic) and not v.is_Float
                                                            for v in ft.values)
    if exact_lin:
        sub_basis = [expression(ft.spec.copy(pars=[(None, 1 if i == j else 0, None) for i in range(len(ft.names))]),
                                vs, "ModelFit") for j in range(len(ft.names))]
        design = [[ev(Expr("ReplaceAll", b, L(*(rule(v, R(xv)) for v, xv in zip(vs, row)))))
                   for b in sub_basis] for row in r["X"]]
        a = FIT._assoc(Type="Linear", BestFit=best, BestFitParameters=L(*(R(v) for v in vals)),
                       BasisFunctions=L(*sub_basis), Variables=L(*vs),
                       DesignMatrix=L(*(L(*row) for row in design)), Response=L(*(R(v) for v in r["y"])),
                       Weights=L(*([1] * len(r["y"]))), Data=r["data"], ConfidenceLevel=R(conf))
        return Expr("FittedModel", a)
    a = FIT._assoc(Type="Nonlinear", BestFit=best,
                   Model=expression(ft.spec, vs, "ModelFit",
                                    [nm if fr else (R(v) if isinstance(v, sp.Basic) else float(v))
                                     for nm, v, fr in zip(ft.names, ft.values, ft.free)]),
                   BestFitParameters=L(*(rule(nm, float(v)) for nm, v in zip(names, vals))),
                   Parameters=L(*names), Variables=L(*vs), DesignMatrix=FIT._L(ft.J),
                   Response=FIT._L(np.array([float(v) for v in r["y"]])),
                   Weights=FIT._L(np.ones(len(r["y"]))), Data=r["data"], ConfidenceLevel=R(conf))
    return Expr("FittedModel", a)


REPORT_PROPS = ("BestModel", "Expression", "Function", "Name", "BaseType", "Hyperparameters", "Variables",
                "InputSize", "OutputSize", "Task", "Parameters", "ParameterNames", "ParameterValues",
                "ParameterAssociation", "ParameterRules", "ParameterCount", "EstimatedParameterCount",
                "ParameterInitialValues", "TrainingLoss", "CriterionFunction", "LossMetric", "FitResiduals",
                "PredictedResponse", "TotalSumOfSquares", "ResidualSumOfSquares", "MeanAbsoluteError",
                "MedianAbsoluteError", "RootMeanSquaredError", "CoefficientOfVariation", "RSquared", "Data",
                "Ordering", "ValidationSet", "CrossValidationData", "ModelList", "FittedModel", "Properties")
_NOT_SUPPORTED = ("ResidualHistogram", "ResidualPlot", "ResidualTrendPlot", "ProbabilityPlot", "QuantilePlot",
                  "ComparisonPlot", "CrossValidationChart", "TrainingTime", "MemoryUsed", "TotalTrainingTime",
                  "TotalMemoryUsed", "Schema", "ColumnNames", "ColumnVariableMap", "VariableColumnMap",
                  "TabularFunction", "InputType", "OutputType", "Constraints", "Weights",
                  "HyperparameterDomain", "HyperparameterDefaultDomain")


def report_assoc(r):
    """ModelFitReport association (all values computed here, so the report is a plain expression)"""
    ft = r["best"]
    Xa = np.array([[float(v) for v in q] for q in r["X"]])
    ya = np.array([float(v) for v in r["y"]])
    exact = isinstance(ft, Fit) and all(isinstance(v, sp.Basic) and not v.is_Float for v in ft.values) and \
        ft.basis is not None and not has_float(r["data"])
    if exact:
        pred_s = [ev(Expr("ReplaceAll", ft.expr(), L(*(rule(v, R(x)) for v, x in zip(ft.variables(), row)))))
                  for row in r["X"]]
        res = L(*(ev(Expr("Plus", R(yv), Expr("Times", -1, p))) for yv, p in zip(r["y"], pred_s)))
        pred = L(*pred_s)
    pa = ft.predict(Xa)
    resa = ya - pa
    if not exact:
        res, pred = FIT._L(resa), FIT._L(pa)
    rss = float(resa @ resa)
    tss = float(np.sum((ya - ya.mean()) ** 2))
    model = ft.model_expr()
    sp_ = ft.spec
    is_knn = isinstance(ft, KNNFit)
    hyp = Expr("Association")
    if sp_.head == "PolynomialModel":
        hyp = Expr("Association", rule("Degree", sp_.hyper))
    elif sp_.head == "PeriodicModel":
        hyp = Expr("Association", rule("FrequencyCount", sp_.hyper))
    elif is_knn:
        hyp = Expr("Association", rule("Neighbors", _nb_expr(ft.nb)), rule("Weights", ft.weights))
    vals = [R(v) if isinstance(v, sp.Basic) else float(v) for v in ft.values]
    cvd = []
    for i, c in enumerate(r["cands"]):
        if r["cv"][i] is None:
            continue
        ls = r["cv"][i]
        cvd.append(Expr("Association", rule("Model", _label(c)),
                        rule("Losses", L(*(Symbol("Infinity") if not np.isfinite(v) else float(v) for v in ls))),
                        rule("MeanLoss", Symbol("Infinity") if not np.all(np.isfinite(ls)) else float(np.mean(ls)))))
    mlist = []
    for i in r["order"]:
        c = r["cands"][i]
        try:
            mlist.append(_fit_any(c, r["X"], r["y"], r["fname"], strict=False).model_expr()
                         if i != r["order"][0] else model)
        except (ValueError, ArithmeticError, AttributeError):
            mlist.append(_label(c))
    vsx = r["vs"]
    vs_out = Symbol("None") if (_sym(vsx, "None") or len(r["cands"]) == 1) else \
        (Expr("Rule", "KFold", 5) if _sym(vsx, "Automatic") or vsx == "KFold" else vsx)
    ent = dict(
        BestModel=model,
        Name=sp_.head, BaseType="Nearest" if is_knn else ("Linear" if ft.basis is not None else "Formula"),
        Hyperparameters=hyp, Variables=L(*ft.variables()), InputSize=ft.nv, OutputSize=1, Task="Regression",
        ParameterNames=L(*ft.names), ParameterValues=L(*vals),
        ParameterCount=len(ft.names), EstimatedParameterCount=ft.nparams(),
        TrainingLoss=_loss(resa, r["crit"]), CriterionFunction=r["crit"], LossMetric=r["crit"],
        FitResiduals=res, PredictedResponse=pred, TotalSumOfSquares=tss, ResidualSumOfSquares=rss,
        MeanAbsoluteError=_loss(resa, "MeanAbsoluteError"), MedianAbsoluteError=_loss(resa, "MedianAbsoluteError"),
        RootMeanSquaredError=_loss(resa, "RootMeanSquaredError"),
        CoefficientOfVariation=_loss(resa, "RootMeanSquaredError") / abs(float(ya.mean())) if ya.mean() != 0
        else Symbol("Indeterminate"),
        RSquared=1 - rss / tss if tss > 0 else Symbol("Indeterminate"),
        Data=r["data"], Ordering=L(*(i + 1 for i in r["order"])), ValidationSet=vs_out,
        CrossValidationData=L(*cvd), ModelList=L(*mlist))
    if not is_knn:
        ent["Expression"] = ft.expr()
        ent["ParameterInitialValues"] = L(*_init_values(sp_, ft))
        try:
            ent["FittedModel"] = _fm_object(r)
        except (ValueError, TypeError):
            pass
    return Expr("Association", *(rule(k, v) for k, v in ent.items()))


def _init_values(spec, ft):
    out = []
    for (nm, fx, ini) in _entries(spec, ft.names, "ModelFit"):
        out.append(R(fx) if fx is not None else (ini if ini is not None else 1))
    return out


def report_property(a, prop, fname="ModelFitReport"):
    if prop == "Properties":
        return L(*REPORT_PROPS)
    if prop in _NOT_SUPPORTED:
        raise NotImplementedError(f'{fname}: the property "{prop}" is not supported')
    base = _assoc_get(a, "BaseType")
    if prop == "Function":
        if base == "Nearest":
            raise ValueError(f'{fname}: "Function" is not available for nearest-neighbour models')
        return ev(Expr("Function", _assoc_get(a, "Variables"), _assoc_get(a, "Expression")))
    if prop in ("Parameters", "ParameterAssociation", "ParameterRules"):
        names, vals = _assoc_get(a, "ParameterNames").args, _assoc_get(a, "ParameterValues").args
        if base == "Nearest":
            raise ValueError(f'{fname}: "{prop}" is not available for nearest-neighbour models')
        if prop == "Parameters":
            return L(*vals)
        rs = [rule(nm, v) for nm, v in zip(names, vals)]
        return Expr("Association", *rs) if prop == "ParameterAssociation" else L(*rs)
    if prop in ("Expression", "FittedModel", "ParameterInitialValues") and _assoc_get(a, prop) is None:
        raise ValueError(f'{fname}: "{prop}" is not available for this model')
    v = _assoc_get(a, prop)
    if v is None:
        raise ValueError(f'{fname}: unknown property "{prop}"')
    return v


@fn("ModelFit", area=AREA, status="added",
    notes="ModelFit[data, model] fits a model specification (LinearModel, PolynomialModel, ExponentialModel, "
          "LogModel, PowerModel, PeriodicModel, NearestModel), a named model (\"Linear\", \"Quadratic\", "
          "\"Cubic\", \"Exponential\", \"Log\", \"Power\") or a list / hyperparameter family of them to data "
          "{y1, ...} (x = 1, 2, ...), {{x.., y}, ...}, {x -> y, ...}, {xs} -> {ys} or matrix -> column, and returns "
          "the model with fitted parameter rules; ModelFit[data, model, prop] (or {props}) gives \"Model\" or any "
          "ModelFitReport property (\"Report\" gives the report). Parameters are least-squares estimates (exact "
          "for exact data and models linear in the parameters; otherwise Levenberg-Marquardt from the documented "
          "starting-value search of each family, with a gradient self-check). With several candidates the model "
          "with the smallest mean cross-validation loss wins: CriterionFunction \"RootMeanSquaredError\" "
          "(default), \"MeanSquaredError\", \"MeanAbsoluteError\", \"MedianAbsoluteError\"; ValidationSet "
          "Automatic (= \"KFold\" -> 5, as documented), \"KFold\" -> k, \"LeaveOneOut\", \"LeavePOut\" -> p, "
          "None (training loss) or explicit validation data. Folds are deterministic (point i in fold "
          "Mod[i - 1, k] + 1), so the selection is reproducible but need not equal Wolfram's random folds; a "
          "candidate that cannot be fitted on a fold gets loss Infinity, ties go to the earlier candidate. "
          "Tabular data, Weights, Method, \"Logistic\" and other models raise NotImplementedError. Check: "
          "parameters compared with numpy.linalg.lstsq / statsmodels OLS and scipy curve_fit on the same data, "
          "and the cross-validation losses with a hand computation (tests).")
def _model_fit(data: Expr, model: Expr | String | ListOf[Expr], prop: String | ListOf[String] = None,
               *rest: Option) -> Expr:
    name = "ModelFit"
    rest = list(rest)
    if prop is not None and SM._is_rule(prop):
        rest.insert(0, prop)
        prop = None
    _, o = SM.opts(rest, MF_OPTS, name)
    r = model_fit(data, model, name, o)
    if prop is None or prop == "Model":
        return r["best"].model_expr()

    def one(p):
        if p == "Model":
            return r["best"].model_expr()
        if p == "Report":
            return Expr("ModelFitReport", report_assoc(r))
        if not isinstance(p, str):
            raise TypeError(f"{name}: properties must be strings")
        if p == "FittedModel":
            return _fm_object(r)
        return report_property(report_assoc(r), p, name)
    if is_list(prop):
        return L(*(one(p) for p in prop.args))
    return one(prop)


@fn("ModelFitReport", area=AREA, status="added",
    notes="ModelFitReport[<|...|>] is the report returned by ModelFit[..., \"Report\"]; report[\"prop\"] (or "
          "[{props}]) gives \"BestModel\", \"Expression\", \"Function\", \"Name\", \"BaseType\", "
          "\"Hyperparameters\", \"Variables\", \"InputSize\", \"OutputSize\", \"Task\", \"Parameters\", "
          "\"ParameterNames\", \"ParameterValues\", \"ParameterAssociation\", \"ParameterRules\", "
          "\"ParameterCount\", \"EstimatedParameterCount\", \"ParameterInitialValues\", \"TrainingLoss\", "
          "\"CriterionFunction\", \"LossMetric\", \"FitResiduals\", \"PredictedResponse\", \"TotalSumOfSquares\", "
          "\"ResidualSumOfSquares\", \"MeanAbsoluteError\", \"MedianAbsoluteError\", \"RootMeanSquaredError\", "
          "\"CoefficientOfVariation\" (RMSE / |mean response|, a stated definition), \"RSquared\", \"Data\", "
          "\"Ordering\", \"ValidationSet\", \"CrossValidationData\" (per candidate <|\"Model\", \"Losses\", "
          "\"MeanLoss\"|>), \"ModelList\" (candidates refitted on all data, best first), \"FittedModel\" and "
          "\"Properties\". Charts, plots, timings, memory and Tabular schema properties raise "
          "NotImplementedError; unknown properties ValueError. Check: values recomputed from the residuals in "
          "tests (RSS, TSS, RMSE identities).")
def _model_fit_report(a: Expr) -> Expr:
    if not _is_assoc(a):
        raise TypeError("ModelFitReport: expected ModelFitReport[<|...|>] (use ModelFit[..., \"Report\"])")
    raise Unevaluated()


def _report_apply(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == "ModelFitReport" and len(head.args) == 1 and
            _is_assoc(head.args[0])):
        return UNH
    if len(args) != 1:
        return UNH
    p = args[0]
    if isinstance(p, str):
        return report_property(head.args[0], p)
    if is_list(p) and all(isinstance(q, str) for q in p.args):
        return L(*(report_property(head.args[0], q) for q in p.args))
    return UNH


KL.COMPOUND_HEAD_HOOKS["ModelFitReport"] = _report_apply
