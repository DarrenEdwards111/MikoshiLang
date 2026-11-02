"""Wolfram batch X08: geostatistics -- VariogramModel, BinnedVariogramList, EstimatedVariogramModel, SpatialEstimate
(kriging) and SpatialEstimatorFunction, with the options VariogramFunction, SpatialNoiseLevel and
SpatialTrendFunction.

Methods (Cressie, Statistics for Spatial Data, Wiley 1993, ch. 2-3; Chiles & Delfiner, Geostatistics, 2012):

* VariogramModel["model", {params}] with the formulas of the Wolfram documentation; the variogram is 0 at r = 0 and
  theta + model(r) for r > 0, theta the SpatialNoiseLevel (nugget; default 0).
* BinnedVariogramList: pairs farther apart than MaxDistanceRatio (1/3) times the largest distance are dropped, the
  distances binned (HistogramList-style spec; Automatic = Sturges on the pairs), bins with fewer than MinPairs (30)
  pairs dropped; per bin the Cressie-Hawkins estimate (mean |dz|^(1/2))^4 / (2 (0.457 + 0.494/N)) (default) or
  Matheron's mean(dz^2)/2, at the mean pair distance ("Centroid"); returns WeightedData[{{h, gamma}, ...}, {N, ...}].
* EstimatedVariogramModel: Cressie's (1985) weighted least squares Sum_k N_k (gamma_k/gamma(h_k) - 1)^2 with SciPy
  (L-BFGS-B over log / logistic transforms, several starts); Automatic picks the best of Spherical, Exponential and
  Gaussian by that criterion.
* SpatialEstimate: universal kriging in variogram form (constant trend = ordinary kriging) predicting the noise-free
  field S at x0 from Z_i = S(x_i) + eps_i: [Gamma F; F' 0][w; m] = [gamma0; f0] with Gamma_ij = gamma(|x_i - x_j|)
  (noise included off the diagonal) and gamma0_i = gamma_S(|x0 - x_i|) + theta/2; variance w.gamma0 + m.f0.  With
  theta = 0 it interpolates the data (self-checked).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel_lists as KL
from .core import Unevaluated, fn, L, R, S, is_list, rule
from . import wl_spatial2 as SP
from .wl_spatial2 import _np, _ev, _need_scipy, _opt_name, fnum, is_auto
from .wl_spatial2_stats import _inert_option

AREA = "spatial statistics II (X08)"

# name: (parameter names, family) ; family finite | asymptotic | oscillating | unbounded | noise
MODELS = {
    "Askey": ("abc", "finite"), "BoundedLinear": ("ab", "finite"), "Circular": ("ab", "finite"),
    "Cubic": ("ab", "finite"), "EuclidsHat": ("abc", "finite"), "Pentaspherical": ("ab", "finite"),
    "Spherical": ("ab", "finite"), "WhiteNoise": ("a", "noise"),
    "Dagum": ("abcd", "asymptotic"), "Exponential": ("ab", "asymptotic"), "Gaussian": ("ab", "asymptotic"),
    "GeneralizedCauchy": ("abcd", "asymptotic"), "Matern": ("abc", "asymptotic"),
    "PoweredExponential": ("abc", "asymptotic"),
    "GaussianLaguerre": ("abcd", "oscillating"), "Poisson": ("abc", "oscillating"),
    "SineHoleEffect": ("ab", "oscillating"), "Linear": ("a", "unbounded"), "Power": ("ab", "unbounded"),
}
NEEDS_SCIPY = ("EuclidsHat", "Matern", "GaussianLaguerre", "Poisson")


def _check_params(name, p, fname):
    """validate numeric parameters (symbolic ones are left alone)"""
    def num(k):
        v = p[k]
        return None if v.free_symbols else float(v)
    for k, nm in enumerate(MODELS[name][0]):
        v = num(k)
        if v is None:
            continue
        if not math.isfinite(v):
            raise ValueError(f"{fname}: the parameters must be finite")
        if name == "GaussianLaguerre" and nm == "d":
            if v < 0 or v != int(v):
                raise ValueError(f"{fname}: GaussianLaguerre needs a non-negative integer degree d")
            continue
        if not v > 0:
            raise ValueError(f"{fname}: the {name} parameter {nm} must be positive")
        if nm == "c" and name in ("PoweredExponential", "GeneralizedCauchy") and v > 2:
            raise ValueError(f"{fname}: {name} needs 0 < c <= 2")
        if nm == "b" and name == "Power" and not v < 2:
            raise ValueError(f"{fname}: Power needs an exponent 0 < b < 2")


def sym_formula(name, p, r):
    """SymPy expression of the model for r > 0 (without the noise level)"""
    a = p[0]
    b = p[1] if len(p) > 1 else None
    c = p[2] if len(p) > 2 else None
    d = p[3] if len(p) > 3 else None
    pi = sp.pi
    fin = {
        "Askey": lambda: a * (1 - (1 - r / b) ** c),
        "BoundedLinear": lambda: a * r / b,
        "Circular": lambda: 2 * a * (r * sp.sqrt(b ** 2 - r ** 2) / b ** 2 + sp.asin(r / b)) / pi,
        "Cubic": lambda: a * r ** 2 * (28 * b ** 5 - 35 * b ** 4 * r + 14 * b ** 2 * r ** 3 - 3 * r ** 5) / (4 * b ** 7),
        "EuclidsHat": lambda: 2 * a * r * sp.gamma((c + 2) / 2) * sp.hyper([sp.Rational(1, 2), (1 - c) / 2],
                                                                           [sp.Rational(3, 2)], r ** 2 / b ** 2)
        / (b * sp.sqrt(pi) * sp.gamma((c + 1) / 2)),
        "Pentaspherical": lambda: a * r / b * (sp.Rational(15, 8) + r ** 2 / b ** 2 * (-sp.Rational(5, 4) +
                                                                                     3 * r ** 2 / (8 * b ** 2))),
        "Spherical": lambda: -a * r * (r ** 2 - 3 * b ** 2) / (2 * b ** 3),
    }
    if name in fin:
        return sp.Piecewise((fin[name](), r < b), (a, True))
    other = {
        "WhiteNoise": lambda: a,
        "Dagum": lambda: a * (1 + (r / b) ** (-d)) ** (-c / d),
        "Exponential": lambda: a * (1 - sp.exp(-r / b)),
        "Gaussian": lambda: a * (1 - sp.exp(-r ** 2 / b ** 2)),
        "GeneralizedCauchy": lambda: a * (1 - (1 + (r / b) ** c) ** (-d / c)),
        "Matern": lambda: a * (1 - 2 ** (1 - c) * (r / b) ** c * sp.besselk(c, r / b) / sp.gamma(c)),
        "PoweredExponential": lambda: a * (1 - sp.exp(-(r / b) ** c)),
        "GaussianLaguerre": lambda: a * (1 - sp.exp(-r ** 2 / b ** 2) * sp.assoc_laguerre(d, c / 2, r ** 2 / b ** 2)),
        "Poisson": lambda: a * (1 - 2 ** c * (b / r) ** c * sp.besselj(c, r / b) * sp.gamma(c + 1)),
        "SineHoleEffect": lambda: a * (1 - sp.sin(pi * r / b) / (pi * r / b)),
        "Linear": lambda: a * r,
        "Power": lambda: (r / a) ** b,
    }
    return other[name]()


def num_model(name, p):
    """numpy function of r > 0 (array) for numeric parameters p (floats), without the noise level"""
    np = _np()
    a = p[0]
    b = p[1] if len(p) > 1 else None
    c = p[2] if len(p) > 2 else None
    d = p[3] if len(p) > 3 else None
    if name in NEEDS_SCIPY:
        _need_scipy("VariogramModel", f"the {name} model")
        from scipy import special as ss

    def fin(f):
        def g(r):
            r = np.asarray(r, dtype=float)
            with np.errstate(all="ignore"):
                return np.where(r < b, f(np.minimum(r, b)), a)
        return g
    if name == "Askey":
        return fin(lambda r: a * (1 - (1 - r / b) ** c))
    if name == "BoundedLinear":
        return fin(lambda r: a * r / b)
    if name == "Circular":
        return fin(lambda r: 2 * a * (r * np.sqrt(np.clip(b * b - r * r, 0, None)) / b ** 2 + np.arcsin(r / b)) / math.pi)
    if name == "Cubic":
        return fin(lambda r: a * r ** 2 * (28 * b ** 5 - 35 * b ** 4 * r + 14 * b ** 2 * r ** 3 - 3 * r ** 5) / (4 * b ** 7))
    if name == "EuclidsHat":
        k = 2 * a * math.gamma((c + 2) / 2) / (b * math.sqrt(math.pi) * math.gamma((c + 1) / 2))
        return fin(lambda r: k * r * ss.hyp2f1(0.5, (1 - c) / 2, 1.5, np.clip(r / b, 0, 1) ** 2))
    if name == "Pentaspherical":
        return fin(lambda r: a * r / b * (15 / 8 + r ** 2 / b ** 2 * (-5 / 4 + 3 * r ** 2 / (8 * b ** 2))))
    if name == "Spherical":
        return fin(lambda r: -a * r * (r ** 2 - 3 * b ** 2) / (2 * b ** 3))
    if name == "WhiteNoise":
        return lambda r: np.full(np.shape(r), a, dtype=float)
    if name == "Dagum":
        return lambda r: a * (1 + (np.asarray(r, float) / b) ** (-d)) ** (-c / d)
    if name == "Exponential":
        return lambda r: a * (1 - np.exp(-np.asarray(r, float) / b))
    if name == "Gaussian":
        return lambda r: a * (1 - np.exp(-np.asarray(r, float) ** 2 / b ** 2))
    if name == "GeneralizedCauchy":
        return lambda r: a * (1 - (1 + (np.asarray(r, float) / b) ** c) ** (-d / c))
    if name == "Matern":
        def mat(r):
            x = np.asarray(r, float) / b
            with np.errstate(all="ignore"):
                v = a * (1 - 2 ** (1 - c) * x ** c * ss.kv(c, x) / math.gamma(c))
            return np.where(x > 0, np.nan_to_num(v, nan=a), 0.0)
        return mat
    if name == "PoweredExponential":
        return lambda r: a * (1 - np.exp(-(np.asarray(r, float) / b) ** c))
    if name == "GaussianLaguerre":
        return lambda r: a * (1 - np.exp(-np.asarray(r, float) ** 2 / b ** 2) *
                              ss.eval_genlaguerre(int(d), c / 2, np.asarray(r, float) ** 2 / b ** 2))
    if name == "Poisson":
        def poi(r):
            x = np.asarray(r, float) / b
            with np.errstate(all="ignore"):
                v = a * (1 - 2 ** c * x ** (-c) * ss.jv(c, x) * math.gamma(c + 1))
            return np.where(x > 1e-8, v, 0.0)
        return poi
    if name == "SineHoleEffect":
        return lambda r: a * (1 - np.sinc(np.asarray(r, float) / b))
    if name == "Linear":
        return lambda r: a * np.asarray(r, float)
    if name == "Power":
        return lambda r: (np.asarray(r, float) / a) ** b
    raise ValueError(f"VariogramModel: unknown model {name}")


# ─────────────────────────── VariogramModel object ───────────────────────────

_EXTRA = ("BinnedVariogram", "WeightedSumOfResiduals")


class VM:
    def __init__(self, x, fname="VariogramModel"):
        if not (isinstance(x, Expr) and x.head == "VariogramModel"):
            raise TypeError(f"{fname}: expected VariogramModel[\"model\", {{params}}]")
        args = list(x.args)
        self.extra = {}
        self.noise = sp.Integer(0)
        while args and _opt_name(args[-1]) is not None:
            o = args.pop()
            k = _opt_name(o)
            if k == "SpatialNoiseLevel":
                v = o.args[1]
                if not is_auto(v):
                    nv = S(v)
                    if not isinstance(nv, sp.Basic):
                        raise TypeError(f"{fname}: SpatialNoiseLevel must be a non-negative number")
                    if not nv.free_symbols and not float(nv) >= 0:
                        raise ValueError(f"{fname}: SpatialNoiseLevel must be non-negative")
                    self.noise = nv
            elif k in _EXTRA:
                self.extra[k] = o.args[1]
            else:
                raise ValueError(f"{fname}: unknown option {k}")
        if len(args) != 2:
            raise TypeError(f"{fname}: expected VariogramModel[\"model\", {{params}}]")
        name, prm = args
        if isinstance(name, Symbol):
            raise Unevaluated()
        if not isinstance(name, str):
            raise TypeError(f"{fname}: the model must be a string such as \"Spherical\"")
        if name not in MODELS:
            raise ValueError(f"{fname}: unknown model \"{name}\" (use {', '.join(sorted(MODELS))})")
        if not is_list(prm):
            raise TypeError(f"{fname}: the parameters must be a list")
        k = len(MODELS[name][0])
        if len(prm.args) != k:
            raise ValueError(f"{fname}: the {name} model needs {k} parameters {{{', '.join(MODELS[name][0])}}}")
        ps = []
        for v in prm.args:
            sv = S(v)
            if isinstance(v, (str, bool)) or not isinstance(sv, sp.Basic) or (not sv.free_symbols and not sv.is_real):
                raise TypeError(f"{fname}: the parameters must be real numbers")
            ps.append(sv)
        _check_params(name, ps, fname)
        self.name, self.p, self.e = name, ps, x

    @property
    def numeric(self):
        return not any(v.free_symbols for v in self.p) and not self.noise.free_symbols

    def fp(self):
        return [float(v) for v in self.p]

    def gamma_num(self, r, with_noise=True):
        """numeric variogram at distances r (array): 0 at r = 0, noise + model otherwise"""
        np = _np()
        r = np.asarray(r, dtype=float)
        f = num_model(self.name, self.fp())
        v = np.where(r > 0, f(np.where(r > 0, r, 1.0)), 0.0)
        if with_noise:
            v = np.where(r > 0, v + float(self.noise), 0.0)
        return v

    def value(self, r):
        rs = sp.Symbol("r", positive=True)
        rv = S(r)
        if rv == 0:
            return sp.Integer(0)
        if rv.is_number and not rv.is_real:
            raise ValueError("VariogramModel: the distance must be real")
        if rv.is_number and rv < 0:
            raise ValueError("VariogramModel: the distance must be non-negative")
        v = self.noise + sym_formula(self.name, self.p, rs).subs(rs, rv)
        if isinstance(v, sp.Piecewise):
            v = sp.piecewise_fold(v)
        if rv.is_Float or any(isinstance(q, sp.Float) for q in self.p):
            v = sp.N(v)
        return sp.simplify(v) if not rv.is_Float else v

    def sill(self):
        if MODELS[self.name][1] == "unbounded":
            return sp.oo
        return self.p[0]

    def range_(self):
        fam = MODELS[self.name][1]
        if fam == "unbounded":
            return sp.oo
        if fam == "noise":
            return sp.Integer(0)
        if fam == "finite":
            return self.p[1]
        if not self.numeric:
            raise Unevaluated()
        b = float(self.p[1])
        target = 0.95 if fam == "asymptotic" else 1.0
        f = num_model(self.name, self.fp())
        a = float(self.p[0])
        np = _np()
        grid = b * np.logspace(-6, 4, 4001)
        v = f(grid) / a - target
        idx = np.nonzero(v >= 0)[0]
        if not len(idx):
            raise ArithmeticError(f"VariogramModel: no range found below 10^4 b for the {self.name} model")
        i = int(idx[0])
        lo, hi = (grid[i - 1] if i > 0 else 0.0), grid[i]
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(np.array([mid]))[0] / a - target >= 0:
                hi = mid
            else:
                lo = mid
        return sp.Float(hi)


def _vm_props(head, args, ev):
    from .. import kernel as K
    fname = "VariogramModel"
    if len(args) != 1:
        return K.UNHANDLED
    vm = VM(head, fname)
    a = args[0]
    if isinstance(a, str):
        props = ["ModelFormula", "Name", "SpatialNoiseLevel", "Parameters", "Properties", "Range", "Sill",
                 "Visualization"] + [k for k in _EXTRA if k in vm.extra]
        if a == "Properties":
            return L(*props)
        if a == "Name":
            return vm.name
        if a == "Parameters":
            return L(*(R(v) for v in vm.p))
        if a == "SpatialNoiseLevel":
            return R(vm.noise)
        if a == "Sill":
            return R(vm.sill())
        if a == "Range":
            return R(vm.range_())
        if a == "ModelFormula":
            rs = sp.Symbol("r", positive=True)
            f = sym_formula(vm.name, vm.p, rs)
            return Expr("Function", L(Symbol("r")), R(f.subs(rs, sp.Symbol("r"))))
        if a == "Visualization":
            rg = vm.range_()
            top = 2 * float(rg) if rg.is_finite and rg != 0 else (2 * float(vm.p[1]) if len(vm.p) > 1 else 1.0)
            if not vm.numeric:
                raise Unevaluated()
            return ev(Expr("Plot", Expr(head, Symbol("x08vmr")), L(Symbol("x08vmr"), R(0), R(top))))
        if a in vm.extra:
            return vm.extra[a]
        raise ValueError(f"{fname}: unknown property {a!r}")
    if isinstance(a, Symbol):
        return K.UNHANDLED
    if is_list(a):
        return L(*(R(vm.value(v)) for v in a.args))
    if not S(a).is_number:
        return K.UNHANDLED
    return R(vm.value(a))


KL.COMPOUND_HEAD_HOOKS["VariogramModel"] = _vm_props


@fn("VariogramModel", area=AREA, status="added",
    notes="VariogramModel[\"model\", {params}] with the Wolfram documentation's formulas: finite range Askey "
          "{a, b, c}, BoundedLinear, Circular, Cubic, EuclidsHat {a, b, c}, Pentaspherical, Spherical {a, b} "
          "(sill a reached at range b), WhiteNoise {a}; asymptotic Dagum {a, b, c, d}, Exponential, Gaussian, "
          "GeneralizedCauchy {a, b, c, d}, Matern {a, b, c}, PoweredExponential {a, b, c}; oscillating "
          "GaussianLaguerre {a, b, c, d}, Poisson {a, b, c}, SineHoleEffect {a, b}; unbounded Linear {a} and "
          "Power {a, b} ((r/a)^b). vm[r] is 0 at r = 0 and SpatialNoiseLevel + model(r) for r > 0 (exact for exact "
          "input; SpatialNoiseLevel -> theta, default 0). Properties \"ModelFormula\" (Function without noise), "
          "\"Name\", \"Parameters\", \"SpatialNoiseLevel\", \"Sill\" (a; Infinity if unbounded), \"Range\" (b for "
          "finite models; numerically where the model reaches 95% of the sill for asymptotic ones and first "
          "crosses it for oscillating ones), \"Visualization\" (Plot), \"Properties\", plus \"BinnedVariogram\" and "
          "\"WeightedSumOfResiduals\" from EstimatedVariogramModel. Parameters must be positive (0 < c <= 2 for "
          "PoweredExponential and GeneralizedCauchy, 0 < b < 2 for Power, integer d >= 0 for GaussianLaguerre; "
          "further dimension-dependent validity conditions are not checked); invalid values raise ValueError, "
          "unknown models ValueError, wrong kinds TypeError. Check: reference (closed forms at the range and "
          "Exponential 3b-type ranges by hand; numeric and exact evaluations agree) in tests.")
def _variogram_model(name: Expr, params: Expr, *opts: Option) -> Expr:
    VM(Expr("VariogramModel", name, params, *opts))
    raise Unevaluated()


# ─────────────────────────── data and trends ───────────────────────────

def parse_field(data, fname):
    """locations (n x d array), values (n,) from {loc -> val, ...} or {locs} -> {vals}"""
    np = _np()
    if isinstance(data, Expr) and data.head == "Rule" and is_list(data.args[0]) and is_list(data.args[1]):
        locs, vals = list(data.args[0].args), list(data.args[1].args)
        if len(locs) != len(vals):
            raise ValueError(f"{fname}: the numbers of locations and values differ")
    elif is_list(data) and data.args and all(isinstance(r_, Expr) and r_.head == "Rule" for r_ in data.args):
        locs, vals = [r_.args[0] for r_ in data.args], [r_.args[1] for r_ in data.args]
    elif isinstance(data, Symbol):
        raise Unevaluated()
    else:
        raise TypeError(f"{fname}: expected {{loc1 -> val1, ...}} or {{loc1, ...}} -> {{val1, ...}}")
    X = []
    for p in locs:
        if isinstance(p, Expr) and p.head in ("GeoPosition", "GeoPositionENU", "GeoPositionXYZ", "Entity"):
            raise NotImplementedError(f"{fname}: geographic locations are not supported")
        if is_list(p):
            X.append([fnum(c, fname, "a coordinate") for c in p.args])
        else:
            X.append([fnum(p, fname, "a location")])
    if len({len(x) for x in X}) != 1 or not X[0]:
        raise ValueError(f"{fname}: all locations must have the same (positive) dimension")
    z = []
    for v in vals:
        if isinstance(v, Expr) and v.head == "Quantity":
            raise NotImplementedError(f"{fname}: Quantity values are not supported")
        z.append(fnum(v, fname, "a value"))
    if len(z) < 2:
        raise ValueError(f"{fname}: at least 2 data points are needed")
    if len(z) > 4000:
        raise ValueError(f"{fname}: at most 4000 data points are supported")
    return np.array(X, dtype=float), np.array(z, dtype=float)


def _monomials(d, deg):
    from itertools import combinations_with_replacement
    out = [()]
    for k in range(1, deg + 1):
        out += list(combinations_with_replacement(range(d), k))
    return out


def trend_basis(spec, d, fname, default="constant"):
    """function X -> F (n x q) for a trend spec, or None (no trend)"""
    np = _np()
    if spec is None or spec == Symbol("None"):
        return None if default == "none" else (lambda X: np.ones((len(X), 1)))
    if is_auto(spec):
        return lambda X: np.ones((len(X), 1))
    if isinstance(spec, int) and not isinstance(spec, bool):
        if spec < 0:
            raise ValueError(f"{fname}: the trend degree must be a non-negative integer")
        mons = _monomials(d, spec)
        return lambda X: np.stack([np.prod(X[:, list(m)], axis=1) if m else np.ones(len(X)) for m in mons], axis=1)
    if is_list(spec) and len(spec.args) == 2 and is_list(spec.args[0]) and is_list(spec.args[1]):
        vs = spec.args[1].args
        if not all(isinstance(v, Symbol) for v in vs) or len(vs) != d:
            raise ValueError(f"{fname}: the trend variables must be {d} symbols")
        names = [v.name for v in vs]
        fs = []
        for f in spec.args[0].args:
            try:
                fs.append(SP.np_compile(_ev(f), names))
            except ValueError as ex:
                raise ValueError(f"{fname}: cannot compile the trend basis function {f} ({ex})")

        def F(X):
            cols = [np.ones(len(X))]
            for f in fs:
                cols.append(np.broadcast_to(np.asarray(f(*[X[:, k] for k in range(d)]), dtype=float), (len(X),)))
            return np.stack(cols, axis=1)
        return F
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: SpatialTrendFunction must be Automatic, None, a degree n or {{{{f1, ...}}, {{x, ...}}}}")


def detrend(X, z, F, fname):
    np = _np()
    if F is None:
        return z
    A = F(X)
    if np.linalg.matrix_rank(A) < A.shape[1]:
        raise ValueError(f"{fname}: the trend basis is degenerate at the data locations")
    beta = np.linalg.lstsq(A, z, rcond=None)[0]
    return z - A @ beta


# ─────────────────────────── BinnedVariogramList ───────────────────────────

_METHOD_OPTS = {"BinCenter": "Centroid", "MaxDistanceRatio": 1 / 3, "MinPairs": 30, "ScaleEstimator": "Cressie"}


def _method(m, fname):
    o = dict(_METHOD_OPTS)
    if is_auto(m) or m == "NonLattice":
        return o
    if is_list(m) and m.args and m.args[0] == "NonLattice":
        for r_ in m.args[1:]:
            k = _opt_name(r_)
            if k not in o:
                raise ValueError(f"{fname}: unknown method option {k}")
            o[k] = r_.args[1]
    elif isinstance(m, str) or is_list(m):
        raise NotImplementedError(f"{fname}: only Method -> \"NonLattice\" is supported")
    else:
        raise TypeError(f"{fname}: invalid Method setting")
    if o["BinCenter"] not in ("Centroid", "Midpoint"):
        raise ValueError(f"{fname}: \"BinCenter\" must be \"Centroid\" or \"Midpoint\"")
    if o["ScaleEstimator"] in ("Qn", "Sn"):
        raise NotImplementedError(f"{fname}: the \"{o['ScaleEstimator']}\" scale estimator is not supported "
                                  "(use \"Cressie\" or \"Matheron\")")
    if o["ScaleEstimator"] not in ("Cressie", "Matheron"):
        raise ValueError(f"{fname}: \"ScaleEstimator\" must be \"Cressie\" or \"Matheron\"")
    o["MaxDistanceRatio"] = fnum(o["MaxDistanceRatio"], fname, "\"MaxDistanceRatio\"")
    if not 0 < o["MaxDistanceRatio"] <= 1:
        raise ValueError(f"{fname}: \"MaxDistanceRatio\" must be in (0, 1]")
    mp = o["MinPairs"]
    if isinstance(mp, bool) or not isinstance(mp, int) or mp < 1:
        raise ValueError(f"{fname}: \"MinPairs\" must be a positive integer")
    return o


def _bins(spec, h, hmax, fname):
    np = _np()
    if is_auto(spec):
        return np.linspace(0, hmax, int(math.ceil(math.log2(len(h)))) + 2)
    if isinstance(spec, int) and not isinstance(spec, bool):
        if spec < 1:
            raise ValueError(f"{fname}: the number of bins must be positive")
        return np.linspace(0, hmax, spec + 1)
    if is_list(spec) and len(spec.args) == 1 and is_list(spec.args[0]):
        e = np.array([fnum(v, fname, "a bin edge") for v in spec.args[0].args])
        if len(e) < 2 or np.any(np.diff(e) <= 0):
            raise ValueError(f"{fname}: bin edges must be strictly increasing")
        return e
    if is_list(spec) and len(spec.args) == 1:
        w = fnum(spec.args[0], fname, "the bin width")
        if not w > 0:
            raise ValueError(f"{fname}: the bin width must be positive")
        return w * np.arange(int(math.ceil(hmax / w)) + 1)
    if is_list(spec) and len(spec.args) == 3:
        a, b, w = (fnum(v, fname, "a bin specification") for v in spec.args)
        if not (w > 0 and b > a):
            raise ValueError(f"{fname}: {{min, max, w}} needs min < max and w > 0")
        return a + w * np.arange(int(math.floor((b - a) / w + 1e-9)) + 1)
    if isinstance(spec, str):
        raise NotImplementedError(f"{fname}: named binning methods are not supported (give n, {{w}}, {{min, max, "
                                  "w}} or {{edges}})")
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: invalid bin specification")


def binned(X, z, spec, mo, fname):
    """(centres, gammas, counts) arrays"""
    np = _np()
    n = len(z)
    iu = np.triu_indices(n, 1)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))[iu]
    dz = (z[:, None] - z[None, :])[iu]
    if D.max() <= 0:
        raise ValueError(f"{fname}: the locations must not all coincide")
    keep = D <= mo["MaxDistanceRatio"] * D.max()
    D, dz = D[keep], dz[keep]
    if not len(D) or D.max() <= 0:
        raise ValueError(f"{fname}: no pairs of distinct locations within \"MaxDistanceRatio\" "
                         f"({mo['MaxDistanceRatio']:.4g}) of the largest distance")
    e = _bins(spec, D, float(D.max()), fname)
    idx = np.searchsorted(e, D, side="right") - 1
    idx[D == e[-1]] = len(e) - 2
    hs, gs, ns = [], [], []
    for k in range(len(e) - 1):
        m = idx == k
        N = int(m.sum())
        if N < mo["MinPairs"] or N == 0:
            continue
        if mo["ScaleEstimator"] == "Matheron":
            g = float((dz[m] ** 2).mean() / 2)
        else:
            g = float(np.mean(np.sqrt(np.abs(dz[m]))) ** 4 / (2 * (0.457 + 0.494 / N)))
        hs.append(float(D[m].mean()) if mo["BinCenter"] == "Centroid" else float((e[k] + e[k + 1]) / 2))
        gs.append(g)
        ns.append(N)
    if not hs:
        raise ValueError(f"{fname}: no distance bin has \"MinPairs\" ({mo['MinPairs']}) pairs; use fewer bins or a "
                         "smaller \"MinPairs\"")
    return np.array(hs), np.array(gs), np.array(ns)


def _wd(hs, gs, ns):
    return Expr("WeightedData", L(*(L(R(float(h)), R(float(g))) for h, g in zip(hs, gs))),
                L(*(int(v) for v in ns)))


def _split(args, names, fname):
    return SP.split_opts(args, names, fname)


@fn("BinnedVariogramList", area=AREA, status="added",
    notes="BinnedVariogramList[{loc -> val, ...}] and BinnedVariogramList[locs -> vals, spec]: empirical "
          "semivariogram. Pairs farther apart than \"MaxDistanceRatio\" (1/3) times the largest distance are "
          "dropped; distances are binned by spec (n bins, {w}, {min, max, w} or {{b1, b2, ...}}; Automatic = "
          "ceil(log2 pairs) + 1 equal bins); bins with fewer than \"MinPairs\" (30) pairs are dropped; per bin the "
          "Cressie-Hawkins estimate (Mean[Sqrt[|dz|]])^4/(2 (0.457 + 0.494/N)) (\"ScaleEstimator\" -> \"Cressie\", "
          "default) or Matheron's Mean[dz^2]/2 (\"Matheron\") at the mean pair distance (\"BinCenter\" -> "
          "\"Centroid\"; \"Midpoint\" the bin middle). Returns WeightedData[{{h, gamma}, ...}, {pairs, ...}]. "
          "Options Method -> {\"NonLattice\", opts}, SpatialTrendFunction (None default; n or {{f, ...}, {x, ...}} "
          "removes a least-squares trend first) and DistanceFunction (Automatic/EuclideanDistance). \"Qn\", \"Sn\", "
          "named binnings, other distance functions, geographic locations and quantities raise "
          "NotImplementedError. Check: exact (Matheron by hand on a small set) in tests.")
def _bvl(data: Expr, *rest: Expr) -> Expr:
    fname = "BinnedVariogramList"
    args, o = _split(rest, ("Method", "SpatialTrendFunction", "DistanceFunction"), fname)
    if len(args) > 1:
        raise TypeError(f"{fname}: expected BinnedVariogramList[data] or BinnedVariogramList[data, spec]")
    df = o.get("DistanceFunction")
    if not (is_auto(df) or df == Symbol("EuclideanDistance")):
        raise NotImplementedError(f"{fname}: only the Euclidean distance is supported")
    X, z = parse_field(data, fname)
    mo = _method(o.get("Method"), fname)
    F = trend_basis(o.get("SpatialTrendFunction", Symbol("None")), X.shape[1], fname, default="none")
    hs, gs, ns = binned(X, detrend(X, z, F, fname), args[0] if args else None, mo, fname)
    return _wd(hs, gs, ns)


# ─────────────────────────── EstimatedVariogramModel ───────────────────────────

def _transforms(name):
    """per parameter: 'pos' (exp), 'c2' (0, 2], 'b2' (0, 2), 'int' (fixed integer)"""
    out = []
    for nm in MODELS[name][0]:
        if name in ("PoweredExponential", "GeneralizedCauchy") and nm == "c":
            out.append("c2")
        elif name == "Power" and nm == "b":
            out.append("b2")
        elif name == "GaussianLaguerre" and nm == "d":
            out.append("int")
        else:
            out.append("pos")
    return out


def _starts(name, hs, gs):
    a0 = float(max(gs.max(), 1e-12))
    b0 = float(hs.max()) / 2
    base = {"a": a0, "b": b0, "c": 1.0, "d": 1.0}
    if name == "Askey" or name == "EuclidsHat":
        base["c"] = 2.0
    if name == "Matern":
        base["c"] = 0.5
    if name == "Linear":
        base["a"] = a0 / float(hs.max())
    if name == "Power":
        base["a"], base["b"] = float(hs.max()) / max(a0, 1e-12), 1.0
    return [base[nm] for nm in MODELS[name][0]]


def fit_model(name, given, hs, gs, ns, noise, fname):
    """(params, noise, wsr); given: list of sympy values or None (free); noise: float or None (free)"""
    np = _np()
    _need_scipy(fname, "variogram fitting")
    from scipy.optimize import minimize
    tr = _transforms(name)
    st = _starts(name, hs, gs)
    free = [i for i, g in enumerate(given) if g is None and tr[i] != "int"]
    for i, g in enumerate(given):
        if g is None and tr[i] == "int":
            given[i] = sp.Integer(1)
    fixed = [float(g) if g is not None else None for g in given]

    def unpack(t):
        p = list(fixed)
        for k, i in enumerate(free):
            if tr[i] == "pos":
                p[i] = math.exp(t[k])
            else:
                top = 2.0 if tr[i] == "c2" else 1.999
                p[i] = top / (1 + math.exp(-t[k]))
        th = math.exp(t[len(free)]) if noise is None else noise
        return p, th

    def obj(t):
        try:
            p, th = unpack(t)
            g = num_model(name, p)(hs) + th
        except (OverflowError, ValueError):
            return 1e300
        if not np.all(np.isfinite(g)) or np.any(g <= 0):
            return 1e300
        return float((ns * (gs / g - 1) ** 2).sum())

    def pack(vals, th):
        t = []
        for i in free:
            v = vals[i]
            if tr[i] == "pos":
                t.append(math.log(v))
            else:
                top = 2.0 if tr[i] == "c2" else 1.999
                q = min(max(v / top, 1e-6), 1 - 1e-6)
                t.append(math.log(q / (1 - q)))
        if noise is None:
            t.append(math.log(th))
        return np.array(t)

    best = None
    g0 = float(max(gs.max(), 1e-12))
    for fb in (0.25, 1.0, 3.0):
        for fth in (1e-6, 0.1):
            vals = [fixed[i] if fixed[i] is not None else st[i] for i in range(len(st))]
            for i in free:
                if MODELS[name][0][i] == "b" and tr[i] == "pos":
                    vals[i] = st[i] * fb
            if not free and noise is not None:
                v = obj(np.array([]))
                cand = (v, np.array([]))
            else:
                res = minimize(obj, pack(vals, fth * g0), method="Nelder-Mead",
                               options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-14})
                cand = (float(res.fun), res.x)
            if best is None or cand[0] < best[0]:
                best = cand
    if best[0] >= 1e299:
        raise ArithmeticError(f"{fname}: the {name} model could not be fitted to the binned variogram")
    p, th = unpack(best[1])
    if noise is None and th < 1e-9 * g0:
        th = 0.0
    return p, th, best[0]


AUTO_MODELS = ("Spherical", "Exponential", "Gaussian")


def _model_spec(m, fname):
    """(name, given list or None)"""
    if isinstance(m, str):
        if m not in MODELS:
            raise ValueError(f"{fname}: unknown variogram model \"{m}\"")
        return m, [None] * len(MODELS[m][0])
    if is_list(m) and len(m.args) == 2 and isinstance(m.args[0], str) and is_list(m.args[1]):
        name = m.args[0]
        if name not in MODELS:
            raise ValueError(f"{fname}: unknown variogram model \"{name}\"")
        if len(m.args[1].args) != len(MODELS[name][0]):
            raise ValueError(f"{fname}: the {name} model needs {len(MODELS[name][0])} parameters")
        given = []
        for v in m.args[1].args:
            sv = S(v)
            given.append(None if (isinstance(v, Symbol) or (isinstance(sv, sp.Basic) and sv.free_symbols)) else sv)
        _check_params(name, [g if g is not None else sp.Symbol("x08free") for g in given], fname)
        return name, given
    if isinstance(m, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: the model must be \"model\" or {{\"model\", params}}")


def _noise_opt(v, fname):
    if is_auto(v):
        return None
    x = fnum(v, fname, "SpatialNoiseLevel")
    if not x >= 0:
        raise ValueError(f"{fname}: SpatialNoiseLevel must be non-negative")
    return x


def estimate_vm(X, z, model, noise, trend_spec, fname):
    np = _np()
    F = trend_basis(trend_spec, X.shape[1], fname)
    r = detrend(X, z, F, fname)
    hs, gs, ns = binned(X, r, None, dict(_METHOD_OPTS), fname)
    if len(hs) < 2:
        raise ValueError(f"{fname}: too few distance bins ({len(hs)}) to fit a variogram")
    if is_auto(model):
        cands = [(nm, [None] * len(MODELS[nm][0])) for nm in AUTO_MODELS]
    else:
        cands = [_model_spec(model, fname)]
    best = None
    for nm, given in cands:
        p, th, w = fit_model(nm, list(given), hs, gs, ns, noise, fname)
        if best is None or w < best[3]:
            best = (nm, p, th, w)
    nm, p, th, w = best
    ex = Expr("VariogramModel", nm, L(*(R(float(v)) if not float(v).is_integer() or _transforms(nm)[i] != "int"
                                        else int(v) for i, v in enumerate(p))),
              rule(Symbol("SpatialNoiseLevel"), R(float(th))), rule("BinnedVariogram", _wd(hs, gs, ns)),
              rule("WeightedSumOfResiduals", R(float(w))))
    return ex


@fn("EstimatedVariogramModel", area=AREA, status="added",
    notes="EstimatedVariogramModel[{loc -> val, ...}] / [locs -> vals] / [..., \"model\"] / [..., {\"model\", "
          "params}]: fits the model (symbols in params are estimated, numbers kept) to the default "
          "BinnedVariogramList (Cressie-Hawkins, 1/3 maximum distance, MinPairs 30) of the data after removing the "
          "SpatialTrendFunction (Automatic = constant; n or {{f, ...}, {x, ...}}) by Cressie's (1985) weighted least "
          "squares Sum N_k (gamma_k/gamma(h_k) - 1)^2 (SciPy Nelder-Mead on log/logistic scales, several starts), "
          "with the nugget fitted (SpatialNoiseLevel -> Automatic) or fixed (-> theta). Automatic chooses the best "
          "of Spherical, Exponential and Gaussian by that criterion (Wolfram's choice may differ). Returns "
          "VariogramModel[name, params, SpatialNoiseLevel -> theta, \"BinnedVariogram\" -> ..., "
          "\"WeightedSumOfResiduals\" -> ...]; GaussianLaguerre's integer degree is not fitted (1 if symbolic). "
          "Needs SciPy (NotImplementedError otherwise); geographic locations and quantities raise "
          "NotImplementedError; too few bins raise ValueError. Check: statistical (recovers the range and sill of "
          "seeded simulated Gaussian fields and exact data from a model within tolerance) in tests.")
def _evm(data: Expr, *rest: Expr) -> Expr:
    fname = "EstimatedVariogramModel"
    args, o = _split(rest, ("SpatialNoiseLevel", "SpatialTrendFunction"), fname)
    if len(args) > 1:
        raise TypeError(f"{fname}: expected EstimatedVariogramModel[data] or EstimatedVariogramModel[data, model]")
    X, z = parse_field(data, fname)
    model = args[0] if args else Symbol("Automatic")
    if not is_auto(model):
        _model_spec(model, fname)
    return estimate_vm(X, z, model, _noise_opt(o.get("SpatialNoiseLevel"), fname),
                       o.get("SpatialTrendFunction", Symbol("Automatic")), fname)


# ─────────────────────────── SpatialEstimate ───────────────────────────

def _sef(X, z, vm_expr, trend_spec):
    return Expr("SpatialEstimatorFunction", Expr("Association",
                                                 rule("Locations", R([list(map(float, x)) for x in X])),
                                                 rule("Values", R([float(v) for v in z])),
                                                 rule("VariogramModel", vm_expr),
                                                 rule("SpatialTrendFunction", trend_spec)))


def _sef_parts(x, fname="SpatialEstimatorFunction"):
    if not (isinstance(x, Expr) and x.head == "SpatialEstimatorFunction" and len(x.args) == 1 and
            isinstance(x.args[0], Expr) and x.args[0].head == "Association"):
        raise TypeError(f"{fname}: expected SpatialEstimatorFunction[<|\"Locations\" -> ..., \"Values\" -> ..., "
                        "\"VariogramModel\" -> ..., \"SpatialTrendFunction\" -> ...|>] as produced by SpatialEstimate")
    d = {r_.args[0]: r_.args[1] for r_ in x.args[0].args}
    for k in ("Locations", "Values", "VariogramModel", "SpatialTrendFunction"):
        if k not in d:
            raise ValueError(f"{fname}: the key \"{k}\" is missing")
    return d


class Kriging:
    def __init__(self, X, z, vm, trend_spec, fname):
        np = _np()
        if not vm.numeric:
            raise Unevaluated()
        self.X, self.z, self.vm = X, z, vm
        self.th = float(vm.noise)
        self.F = trend_basis(trend_spec, X.shape[1], fname)
        n = len(z)
        Fx = self.F(X)
        q = Fx.shape[1]
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        G = vm.gamma_num(D)
        A = np.zeros((n + q, n + q))
        A[:n, :n] = G
        A[:n, n:] = Fx
        A[n:, :n] = Fx.T
        try:
            cond = np.linalg.cond(A)
        except np.linalg.LinAlgError:
            cond = np.inf
        if not np.isfinite(cond) or cond > 1e13:
            raise ValueError(f"{fname}: the kriging system is singular or ill-conditioned (repeated locations or a "
                             "very smooth variogram such as Gaussian without noise, or too few points for the trend); "
                             "set a positive SpatialNoiseLevel or another VariogramFunction")
        self.A = A
        self.n, self.q = n, q

    def predict(self, X0):
        np = _np()
        n = self.n
        D0 = np.sqrt(((X0[:, None, :] - self.X[None, :, :]) ** 2).sum(-1))
        g0 = self.vm.gamma_num(D0, with_noise=False) + self.th / 2
        rhs = np.hstack([g0, self.F(X0)]).T
        sol = np.linalg.solve(self.A, rhs)
        w, m = sol[:n], sol[n:]
        val = w.T @ self.z
        var = (w * g0.T).sum(0) + (m * rhs[n:]).sum(0)
        scale = max(1.0, float(np.abs(g0).max()))
        if np.any(var < -1e-8 * scale):
            raise ArithmeticError("SpatialEstimatorFunction: internal check failed (negative kriging variance)")
        return val, np.maximum(var, 0.0)


def _vm_from(vf, X, z, noise, trend_spec, fname):
    if isinstance(vf, Expr) and vf.head == "VariogramModel":
        vm = VM(vf, fname)
        if noise is not None:
            args = [a for a in vf.args if _opt_name(a) != "SpatialNoiseLevel"]
            vf = Expr("VariogramModel", *args, rule(Symbol("SpatialNoiseLevel"), R(noise)))
        return vf
    if is_list(vf) and len(vf.args) == 2 and isinstance(vf.args[0], str):
        name, given = _model_spec(vf, fname)
        if all(g is not None for g in given):
            ex = Expr("VariogramModel", name, vf.args[1], rule(Symbol("SpatialNoiseLevel"),
                                                                R(noise if noise is not None else 0)))
            VM(ex, fname)
            return ex
        return estimate_vm(X, z, vf, noise, trend_spec, fname)
    if is_auto(vf) or isinstance(vf, str):
        return estimate_vm(X, z, Symbol("Automatic") if is_auto(vf) else vf, noise, trend_spec, fname)
    raise TypeError(f"{fname}: VariogramFunction must be Automatic, \"model\", {{\"model\", params}} or "
                    "VariogramModel[...]")


@fn("SpatialEstimate", area=AREA, status="added",
    notes="SpatialEstimate[{loc -> val, ...}] / [locs -> vals]: kriging. SpatialTrendFunction: Automatic "
          "(constant mean, ordinary kriging), n (polynomial of total degree n) or {{f1, ...}, {x, ...}} (constant plus "
          "the basis functions; universal kriging). VariogramFunction: Automatic (EstimatedVariogramModel of the "
          "detrended data), \"model\" (fitted), {\"model\", params} (given, or fitted symbols) or VariogramModel[...]. "
          "SpatialNoiseLevel: Automatic (the variogram's nugget) or theta >= 0. The predictor targets the "
          "noise-free field (variogram-form universal kriging, gamma0 = gamma_S + theta/2), so with theta = 0 it "
          "interpolates the data and with theta > 0 it smooths. Returns a SpatialEstimatorFunction. At most 4000 "
          "points; repeated locations without noise raise ValueError; fitting needs SciPy (NotImplementedError); "
          "geographic locations and quantities raise NotImplementedError. Check: self-check (exact interpolation "
          "at the data for zero noise, ArithmeticError otherwise) and reference (simple ordinary-kriging weights "
          "by hand) in tests.")
def _spatial_estimate(data: Expr, *opts: Option) -> Expr:
    fname = "SpatialEstimate"
    args, o = _split(opts, ("SpatialTrendFunction", "VariogramFunction", "SpatialNoiseLevel"), fname)
    if args:
        raise TypeError(f"{fname}: expected SpatialEstimate[data, opts]")
    X, z = parse_field(data, fname)
    trend = o.get("SpatialTrendFunction", Symbol("Automatic"))
    trend_basis(trend, X.shape[1], fname)
    noise = _noise_opt(o.get("SpatialNoiseLevel"), fname)
    vme = _vm_from(o.get("VariogramFunction", Symbol("Automatic")), X, z, noise, trend, fname)
    kr = Kriging(X, z, VM(vme, fname), trend, fname)
    if kr.th == 0:
        v, _ = kr.predict(X)
        if not _np().allclose(v, z, rtol=1e-6, atol=1e-6 * max(1.0, float(_np().abs(z).max()))):
            raise ArithmeticError(f"{fname}: internal check failed (zero-noise kriging does not interpolate the data)")
    return _sef(X, z, vme, trend)


_KCACHE = {}


def _kriging_of(head, d, fname):
    key = id(head)
    hit = _KCACHE.get(key)
    if hit is not None and hit[0] is head:
        return hit[1]
    np = _np()
    X = np.array(SP_tolist(d["Locations"]), dtype=float)
    z = np.array(SP_tolist(d["Values"]), dtype=float)
    kr = Kriging(X, z, VM(d["VariogramModel"], fname), d["SpatialTrendFunction"], fname)
    if len(_KCACHE) > 16:
        _KCACHE.clear()
    _KCACHE[key] = (head, kr)
    return kr


def SP_tolist(e):
    if is_list(e):
        return [SP_tolist(a) for a in e.args]
    return float(sp.N(S(e)))


_SEF_PROPS = ("Value", "Around", "StandardDeviation", "Variance")


def _sef_apply(head, args, ev):
    from .. import kernel as K
    fname = "SpatialEstimatorFunction"
    if not 1 <= len(args) <= 2:
        return K.UNHANDLED
    d = _sef_parts(head)
    a = args[0]
    if len(args) == 1 and isinstance(a, str):
        if a == "Properties":
            return L(*_SEF_PROPS)
        if a in d:
            return d[a]
        raise ValueError(f"{fname}: unknown property {a!r}")
    props = args[1] if len(args) == 2 else "Value"
    plist = list(props.args) if is_list(props) else [props]
    for p in plist:
        if p not in _SEF_PROPS:
            raise ValueError(f"{fname}: unknown property {p!r} (use {', '.join(_SEF_PROPS)})")
    kr = _kriging_of(head, d, fname)
    dim = kr.X.shape[1]
    np = _np()
    if is_list(a) and a.args and all(is_list(p) for p in a.args):
        locs, lst = list(a.args), True
    elif is_list(a) and dim == 1 and a.args and not all(is_list(p) for p in a.args) and len(a.args) != 1:
        locs, lst = [L(v) for v in a.args], True
    elif is_list(a):
        locs, lst = [a], False
    else:
        locs, lst = [L(a)], False
    X0 = []
    for p in locs:
        if any(isinstance(c, Symbol) or (isinstance(S(c), sp.Basic) and S(c).free_symbols) for c in p.args):
            return K.UNHANDLED
        if len(p.args) != dim:
            raise ValueError(f"{fname}: the location must have {dim} coordinates")
        X0.append([fnum(c, fname, "a coordinate") for c in p.args])
    val, var = kr.predict(np.array(X0, dtype=float))

    def one(i, p):
        if p == "Value":
            return R(float(val[i]))
        if p == "Variance":
            return R(float(var[i]))
        if p == "StandardDeviation":
            return R(float(math.sqrt(var[i])))
        return Expr("Around", R(float(val[i])), R(float(math.sqrt(var[i]))))

    res = [L(*(one(i, p) for p in plist)) if is_list(props) else one(i, plist[0]) for i in range(len(X0))]
    return L(*res) if lst else res[0]


KL.COMPOUND_HEAD_HOOKS["SpatialEstimatorFunction"] = _sef_apply


@fn("SpatialEstimatorFunction", area=AREA, status="added",
    notes="SpatialEstimatorFunction[<|\"Locations\", \"Values\", \"VariogramModel\", \"SpatialTrendFunction\"|>] is "
          "produced by SpatialEstimate; sef[loc] and sef[{loc1, ...}] predict (machine numbers), sef[loc, prop] with "
          "\"Value\", \"Around\" (Around[value, sd]), \"StandardDeviation\", \"Variance\" (kriging variance of the "
          "noise-free field) or a list of them; sef[key] gives a part. Inert; malformed objects raise "
          "TypeError/ValueError, symbolic locations stay unevaluated. Check: exact (ordinary-kriging prediction and "
          "variance by hand for two points) in tests.")
def _sef_ctor(spec: Assoc) -> Expr:
    _sef_parts(Expr("SpatialEstimatorFunction", spec))
    raise Unevaluated()


_inert_option("VariogramFunction",
              "option of SpatialEstimate: Automatic (EstimatedVariogramModel of the detrended data), \"model\" "
              "(fitted), {\"model\", params} (given; symbols fitted) or VariogramModel[...].")
_inert_option("SpatialNoiseLevel",
              "option of SpatialEstimate, EstimatedVariogramModel and VariogramModel: the nugget (noise variance) "
              "theta >= 0, or Automatic (estimated by EstimatedVariogramModel; 0 for a VariogramModel given "
              "directly); negative values raise ValueError.")
_inert_option("SpatialTrendFunction",
              "option of SpatialEstimate, EstimatedVariogramModel and BinnedVariogramList: Automatic (constant "
              "mean), None (BinnedVariogramList default: no detrending), n (polynomial of total degree n) or "
              "{{f1, ...}, {x, ...}} (constant plus basis functions, as LinearModelFit).")
