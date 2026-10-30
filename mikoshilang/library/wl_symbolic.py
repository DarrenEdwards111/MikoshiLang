"""Shared infrastructure for coverage batch 4 (symbolic calculus and algebra).

* ``kernel_override`` replaces a kernel builtin (D, Limit, Series, Solve, Normal, ...) without editing kernel.py;
* ``replace`` replaces a reviewed-library entry (Integrate, Expand, Factor, ...) without editing the original module;
* ``out`` converts a SymPy result to MikoshiLang, first mapping SymPy-only forms to Wolfram ones
  (Heaviside -> HeavisideTheta, DiracDelta(x, k) -> Derivative[k][DiracDelta][x], Derivative(f(g(x)), g(x)) ->
  Derivative[1][f][g[x]]).
"""
import sympy as sp
from sympy.core.function import AppliedUndef

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import math_ops as M
from .core import fn, record, REGISTRY, R, L, Unevaluated

AREA_CALC = "calculus (batch 4)"
AREA_ALG = "algebra (batch 4)"


def kernel_override(name, area, notes, status="fixed"):
    """Install f(args, ev, expr) as the kernel builtin for ``name``; Unevaluated leaves the call as it is."""
    def deco(f):
        def wrapped(a, ev_, expr):
            try:
                return f(list(a), ev_, expr)
            except Unevaluated:
                return expr
        wrapped.__wrapped__ = f
        K._BUILTINS[name] = wrapped
        record(name, area, status, notes)
        return f
    return deco


def replace(name, area, notes, status="fixed", **kw):
    """Register f as the only library implementation of ``name`` (dropping earlier implementations)."""
    def deco(f):
        REGISTRY.pop(name, None)
        fn(name, area=area, status=status, notes=notes, **kw)(f)
        record(name, area, status, notes)
        return f
    return deco


# ───────────────────────── HeavisideTheta and DiracDelta in SymPy ─────────────────────────

class HeavisideTheta(sp.Heaviside):
    """Wolfram HeavisideTheta: 1 for x > 0, 0 for x < 0, unevaluated at 0 (SymPy's Heaviside gives 1/2 there).
    A Heaviside subclass, so SymPy's integrators, transforms and lambdify treat it as Heaviside."""

    def __new__(cls, arg, H0=sp.S.Half, **options):
        return sp.Function.__new__(cls, sp.sympify(arg), sp.sympify(H0), **options)

    @classmethod
    def eval(cls, arg, H0=sp.S.Half):
        if arg.is_extended_positive:
            return sp.Integer(1)
        if arg.is_extended_negative:
            return sp.Integer(0)
        if arg is sp.nan:
            return sp.nan
        return None

    def _as_heaviside(self, printer, *a, **k):
        return printer._print(sp.Heaviside(*self.args))

    _mpmathcode = _numpycode = _pythoncode = _lambdacode = _scipycode = _as_heaviside


def _dirac(*a):
    return sp.Mul(*[sp.DiracDelta(t) for t in a])


M._FUNCS["HeavisideTheta"] = HeavisideTheta     # was sp.Heaviside, which is 1/2 at 0
M._FUNCS["DiracDelta"] = _dirac                 # DiracDelta[x, y] is a product (SymPy's 2nd argument is an order)
M._INV_FUNCS[HeavisideTheta] = "HeavisideTheta"


def to_heaviside(v):
    """HeavisideTheta -> SymPy Heaviside (for SymPy's integrators and transforms)."""
    if isinstance(v, sp.Basic) and v.has(HeavisideTheta):
        return v.replace(lambda e: isinstance(e, HeavisideTheta), lambda e: sp.Heaviside(e.args[0]))
    return v


def wl_fix(v):
    """Map SymPy-only forms in a result to the forms MikoshiLang converts to Wolfram expressions."""
    if not isinstance(v, sp.Basic):
        return v
    if v.has(sp.Heaviside):
        v = v.replace(lambda e: isinstance(e, sp.Heaviside), lambda e: HeavisideTheta(e.args[0]))
    if v.has(sp.DiracDelta):
        def dd(e):
            if len(e.args) == 2 and e.args[1] != 0:
                d = sp.Dummy("d")
                return sp.Subs(sp.Derivative(sp.Function("DiracDelta")(d), (d, int(e.args[1]))), (d,), (e.args[0],))
            return sp.DiracDelta(e.args[0])
        v = v.replace(lambda e: isinstance(e, sp.DiracDelta) and len(e.args) == 2, dd)
    if v.has(sp.Derivative):
        def conv(d):
            f = d.expr
            if not isinstance(f, AppliedUndef):
                return d
            args = f.args
            if all(isinstance(a, sp.Symbol) for a in args) and len(set(args)) == len(args):
                return d
            counts = dict(d.variable_count)
            if len(set(args)) != len(args) or not all(any(var == a for a in args) for var in counts):
                return d
            dums = [sp.Dummy(f"t{i}") for i in range(len(args))]
            spec = [(dm, counts[a]) for a, dm in zip(args, dums) if counts.get(a)]
            return sp.Subs(sp.Derivative(f.func(*dums), *spec), tuple(dums), tuple(args))
        v = v.replace(lambda e: isinstance(e, sp.Derivative), conv)
    return v


def out(v):
    if isinstance(v, (list, tuple)):
        return L(*(out(x) for x in v))
    return R(wl_fix(v))


def S4(x, fname):
    """MikoshiLang -> SymPy for batch-4 functions (lists are rejected)."""
    from .core import sym
    return sym(x, fname)


def opt_value(opts, name, default=None):
    return opts.get(name, default)


def is_true(x):
    return x is True or (isinstance(x, Symbol) and x.name == "True")


def is_false(x):
    return x is False or (isinstance(x, Symbol) and x.name == "False")


def is_symbol(x, name=None):
    return isinstance(x, Symbol) and (name is None or x.name == name)


def is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def split_opts(args, names, fname, extra_known=()):
    """Trailing Name -> value options with Name in ``names``; a trailing rule naming a known but unsupported
    Wolfram option (``extra_known`` or a common option name) raises a clear error."""
    args = list(args)
    opts = {}
    known = set(extra_known) | _COMMON_OPTIONS
    while args and is_rule(args[-1]) and isinstance(args[-1].args[0], Symbol):
        name = args[-1].args[0].name
        if name in names:
            opts[name] = args.pop().args[1]
        elif name in known:
            raise NotImplementedError(f"{fname}: the option {name} is not supported")
        else:
            break
    return args, opts


_COMMON_OPTIONS = {"Method", "WorkingPrecision", "Assumptions", "GenerateConditions", "PrincipalValue", "Direction",
                   "Analytic", "Modulus", "Extension", "GaussianIntegers", "Trig", "MonomialOrder", "CoefficientDomain",
                   "Sort", "FourierParameters", "Constants", "NonConstants", "TimeConstraint", "ComplexityFunction",
                   "TransformationFunctions", "ExcludedForms", "Backsubstitution", "Cubics", "Quartics",
                   "InverseFunctions", "MaxExtraConditions", "VerifySolutions", "WorkingPrecision", "Heads",
                   "Exclusions", "PerformanceGoal", "Regularization", "VerifyConvergence", "SeriesTermGoal",
                   "MaxIterations", "AccuracyGoal", "PrecisionGoal"}
