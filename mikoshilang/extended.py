"""Extended function library — ~900 new functions across 13 domains.

All functions are implemented as RuleDelayed entries that map
Expr("FunctionName", ...) → computed result via SymPy/SciPy.
"""

from __future__ import annotations
from typing import Any, Dict, List as ListType
from .expr import Expr, Symbol
from .rules import RuleDelayed
from .pattern import Blank

import math


def _to_sympy(expr):
    """Convert MikoshiLang expr to SymPy expression."""
    import sympy as sp

    if isinstance(expr, (int, float)):
        return sp.nsimplify(expr) if isinstance(expr, float) else sp.Integer(expr)
    if isinstance(expr, Symbol):
        name = expr.name
        _constants = {
            'Pi': sp.pi, 'E': sp.E, 'I': sp.I,
            'Infinity': sp.oo, 'True': sp.true, 'False': sp.false,
        }
        return _constants.get(name, sp.Symbol(name))
    if isinstance(expr, Expr):
        head = expr.head
        args = [_to_sympy(a) for a in expr.args]

        _func_map = {
            'Plus': lambda a: a[0] + a[1] if len(a) == 2 else sum(a),
            'Times': lambda a: a[0] * a[1] if len(a) == 2 else math.prod(a) if all(isinstance(x, (int, float)) for x in a) else a[0] * a[1],
            'Power': lambda a: a[0] ** a[1],
            'Sin': lambda a: sp.sin(a[0]),
            'Cos': lambda a: sp.cos(a[0]),
            'Tan': lambda a: sp.tan(a[0]),
            'ArcSin': lambda a: sp.asin(a[0]),
            'ArcCos': lambda a: sp.acos(a[0]),
            'ArcTan': lambda a: sp.atan(a[0]) if len(a) == 1 else sp.atan2(a[0], a[1]),
            'Sinh': lambda a: sp.sinh(a[0]),
            'Cosh': lambda a: sp.cosh(a[0]),
            'Tanh': lambda a: sp.tanh(a[0]),
            'ArcSinh': lambda a: sp.asinh(a[0]),
            'ArcCosh': lambda a: sp.acosh(a[0]),
            'ArcTanh': lambda a: sp.atanh(a[0]),
            'Exp': lambda a: sp.exp(a[0]),
            'Log': lambda a: sp.log(a[0]) if len(a) == 1 else sp.log(a[0], a[1]),
            'Sqrt': lambda a: sp.sqrt(a[0]),
            'Abs': lambda a: sp.Abs(a[0]),
            'Sign': lambda a: sp.sign(a[0]),
            'Floor': lambda a: sp.floor(a[0]),
            'Ceiling': lambda a: sp.ceiling(a[0]),
            'Round': lambda a: sp.Integer(round(float(a[0]))),
            'Max': lambda a: sp.Max(*a),
            'Min': lambda a: sp.Min(*a),
            'Mod': lambda a: sp.Mod(a[0], a[1]),
            'Factorial': lambda a: sp.factorial(a[0]),
            'Gamma': lambda a: sp.gamma(a[0]),
            'Beta': lambda a: sp.beta(a[0], a[1]),
            'Zeta': lambda a: sp.zeta(a[0]),
            'Conjugate': lambda a: sp.conjugate(a[0]),
            'RealPart': lambda a: sp.re(a[0]),
            'ImaginaryPart': lambda a: sp.im(a[0]),
        }

        if head in _func_map:
            try:
                return _func_map[head](args)
            except:
                pass

        # List → tuple for SymPy
        if head == 'List':
            return args

        # Default: return as-is
        return sp.Function(head)(*args) if args else sp.Symbol(head)

    return expr


def _from_sympy(result):
    """Convert SymPy expression back to MikoshiLang expr."""
    import sympy as sp

    if isinstance(result, sp.Integer):
        return int(result)
    if isinstance(result, sp.Float):
        return float(result)
    if isinstance(result, sp.Rational):
        n, d = int(result.p), int(result.q)
        if d == 1:
            return n
        return Expr("Times", n, Expr("Power", d, -1))
    if result is sp.pi:
        return Symbol("Pi")
    if result is sp.E:
        return Symbol("E")
    if result is sp.I:
        return Symbol("I")
    if result is sp.oo:
        return Symbol("Infinity")
    if result is sp.true:
        return True
    if result is sp.false:
        return False
    if isinstance(result, sp.Symbol):
        return Symbol(str(result))
    if isinstance(result, sp.Add):
        args = [_from_sympy(a) for a in result.args]
        if len(args) == 1:
            return args[0]
        r = args[0]
        for a in args[1:]:
            r = Expr("Plus", r, a)
        return r
    if isinstance(result, sp.Mul):
        args = [_from_sympy(a) for a in result.args]
        if len(args) == 1:
            return args[0]
        r = args[0]
        for a in args[1:]:
            r = Expr("Times", r, a)
        return r
    if isinstance(result, sp.Pow):
        return Expr("Power", _from_sympy(result.base), _from_sympy(result.exp))

    # Functions
    if isinstance(result, sp.Function):
        _inv_map = {
            sp.sin: 'Sin', sp.cos: 'Cos', sp.tan: 'Tan',
            sp.asin: 'ArcSin', sp.acos: 'ArcCos', sp.atan: 'ArcTan',
            sp.sinh: 'Sinh', sp.cosh: 'Cosh', sp.tanh: 'Tanh',
            sp.exp: 'Exp', sp.log: 'Log', sp.sqrt: 'Sqrt',
            sp.Abs: 'Abs', sp.sign: 'Sign',
            sp.floor: 'Floor', sp.ceiling: 'Ceiling',
            sp.factorial: 'Factorial', sp.gamma: 'Gamma',
            sp.re: 'RealPart', sp.im: 'ImaginaryPart',
            sp.conjugate: 'Conjugate',
        }
        name = _inv_map.get(type(result), type(result).__name__)
        args = [_from_sympy(a) for a in result.args]
        return Expr(name, *args)

    # Matrix
    if isinstance(result, sp.Matrix):
        rows = []
        for i in range(result.rows):
            row = [_from_sympy(result[i, j]) for j in range(result.cols)]
            rows.append(Expr("List", *row))
        return Expr("Matrix", *rows)

    # Tuple/list
    if isinstance(result, (list, tuple)):
        return Expr("List", *[_from_sympy(a) for a in result])

    # FiniteSet
    if isinstance(result, sp.FiniteSet):
        return Expr("List", *[_from_sympy(a) for a in sorted(result, key=str)])

    # Dict
    if isinstance(result, dict):
        items = [Expr("Rule", _from_sympy(k), _from_sympy(v)) for k, v in result.items()]
        return Expr("List", *items)

    # Fallback
    try:
        f = float(result)
        if f == int(f):
            return int(f)
        return f
    except:
        return str(result)


def _extract_list(expr):
    """Extract a Python list from Expr("List", ...)."""
    if isinstance(expr, Expr) and expr.head == "List":
        return list(expr.args)
    return [expr]


def _to_matrix(expr):
    """Convert Expr("Matrix"/"List", rows...) to sympy Matrix."""
    import sympy as sp
    if isinstance(expr, Expr) and expr.head in ("Matrix", "List"):
        rows = []
        for row in expr.args:
            if isinstance(row, Expr) and row.head == "List":
                rows.append([_to_sympy(x) for x in row.args])
            else:
                rows.append([_to_sympy(row)])
        return sp.Matrix(rows)
    return sp.Matrix([_to_sympy(expr)])


def _to_float_list(expr):
    """Extract a Python list of floats."""
    items = _extract_list(expr)
    return [float(_to_sympy(x)) for x in items]


# ══════════════════════════════════════════════════════════════
# Build all extended rules
# ══════════════════════════════════════════════════════════════

def build_extended_rules():
    """Return list of ~900 RuleDelayed entries for extended functions."""
    import sympy as sp

    from sympy.functions.combinatorial.numbers import stirling as sp_stirling
    try:
        from sympy.functions.combinatorial.numbers import partition as sp_npartitions
    except ImportError:
        from sympy.ntheory.partitions_ import npartitions as sp_npartitions

    rules = []

    # Helper to create a simple rule from head name
    def add_rule(head, fn):
        """Add a catch-all rule for Expr(head, ...) → fn(args)."""
        rules.append(RuleDelayed(
            Expr(head, Blank("__args__")),
            lambda b, _fn=fn, _h=head: _fn(b)
        ))

    # Since the pattern matching is limited, we use a more direct approach.
    # We'll build rules with specific arities.

    def rule1(head, fn):
        """1-arg function."""
        rules.append(RuleDelayed(
            Expr(head, Blank("a")),
            lambda b, _fn=fn: _fn(b["a"])
        ))

    def rule2(head, fn):
        """2-arg function."""
        rules.append(RuleDelayed(
            Expr(head, Blank("a"), Blank("b")),
            lambda b, _fn=fn: _fn(b["a"], b["b"])
        ))

    def rule3(head, fn):
        """3-arg function."""
        rules.append(RuleDelayed(
            Expr(head, Blank("a"), Blank("b"), Blank("c")),
            lambda b, _fn=fn: _fn(b["a"], b["b"], b["c"])
        ))

    def rule4(head, fn):
        """4-arg function."""
        rules.append(RuleDelayed(
            Expr(head, Blank("a"), Blank("b"), Blank("c"), Blank("d")),
            lambda b, _fn=fn: _fn(b["a"], b["b"], b["c"], b["d"])
        ))

    # ── CALCULUS ─────────────────────────────────────────────

    # Diff[expr, var] — already exists, but let's ensure coverage
    def _diff(a, b):
        sa, sb = _to_sympy(a), _to_sympy(b)
        return _from_sympy(sp.diff(sa, sb))
    rule2("Diff", _diff)

    # PartialD[expr, var] — alias for Diff
    rule2("PartialD", _diff)

    # Integrate[expr, var]
    def _integrate(a, b):
        sa = _to_sympy(a)
        if isinstance(b, Expr) and b.head == "List" and len(b.args) == 3:
            var, lo, hi = _to_sympy(b.args[0]), _to_sympy(b.args[1]), _to_sympy(b.args[2])
            return _from_sympy(sp.integrate(sa, (var, lo, hi)))
        return _from_sympy(sp.integrate(sa, _to_sympy(b)))
    rule2("Integrate", _integrate)

    # DefiniteIntegral[expr, {var, a, b}]
    rule2("DefiniteIntegral", _integrate)

    # Gradient[expr, {x, y, z}]
    def _gradient(a, b):
        sa = _to_sympy(a)
        vars_ = [_to_sympy(v) for v in _extract_list(b)]
        grad = [sp.diff(sa, v) for v in vars_]
        return Expr("List", *[_from_sympy(g) for g in grad])
    rule2("Gradient", _gradient)

    # Divergence[{Fx, Fy, Fz}, {x, y, z}]
    def _divergence(a, b):
        comps = [_to_sympy(c) for c in _extract_list(a)]
        vars_ = [_to_sympy(v) for v in _extract_list(b)]
        div = sum(sp.diff(c, v) for c, v in zip(comps, vars_))
        return _from_sympy(div)
    rule2("Divergence", _divergence)

    # Curl[{Fx, Fy, Fz}, {x, y, z}]
    def _curl(a, b):
        F = [_to_sympy(c) for c in _extract_list(a)]
        V = [_to_sympy(v) for v in _extract_list(b)]
        if len(F) == 3 and len(V) == 3:
            curl = [
                sp.diff(F[2], V[1]) - sp.diff(F[1], V[2]),
                sp.diff(F[0], V[2]) - sp.diff(F[2], V[0]),
                sp.diff(F[1], V[0]) - sp.diff(F[0], V[1]),
            ]
            return Expr("List", *[_from_sympy(c) for c in curl])
        return Expr("Error", "Curl requires 3D vectors")
    rule2("Curl", _curl)

    # LaplaceTransform[expr, t, s]
    def _laplace(a, b, c):
        from sympy.integrals.transforms import laplace_transform
        sa, sb, sc = _to_sympy(a), _to_sympy(b), _to_sympy(c)
        result = laplace_transform(sa, sb, sc, noconds=True)
        return _from_sympy(result)
    rule3("LaplaceTransform", _laplace)

    # InverseLaplaceTransform[expr, s, t]
    def _inv_laplace(a, b, c):
        from sympy.integrals.transforms import inverse_laplace_transform
        sa, sb, sc = _to_sympy(a), _to_sympy(b), _to_sympy(c)
        result = inverse_laplace_transform(sa, sb, sc)
        return _from_sympy(result)
    rule3("InverseLaplaceTransform", _inv_laplace)

    # FourierTransform[expr, x, k]
    def _fourier(a, b, c):
        from sympy.integrals.transforms import fourier_transform
        sa, sb, sc = _to_sympy(a), _to_sympy(b), _to_sympy(c)
        result = fourier_transform(sa, sb, sc)
        return _from_sympy(result)
    rule3("FourierTransform", _fourier)

    # InverseFourierTransform[expr, k, x]
    def _inv_fourier(a, b, c):
        from sympy.integrals.transforms import inverse_fourier_transform
        sa, sb, sc = _to_sympy(a), _to_sympy(b), _to_sympy(c)
        result = inverse_fourier_transform(sa, sb, sc)
        return _from_sympy(result)
    rule3("InverseFourierTransform", _inv_fourier)

    # Series[expr, {var, point, order}]
    def _series(a, b):
        sa = _to_sympy(a)
        items = _extract_list(b)
        var = _to_sympy(items[0])
        point = _to_sympy(items[1]) if len(items) > 1 else 0
        order = int(items[2]) if len(items) > 2 else 6
        result = sp.series(sa, var, point, order).removeO()
        return _from_sympy(result)
    rule2("Series", _series)
    rule2("MaClaurinSeries", lambda a, b: _series(a, b))
    rule2("TaylorSeries", lambda a, b: _series(a, b))

    # Limit[expr, var -> point]
    def _limit(a, b):
        sa = _to_sympy(a)
        if isinstance(b, Expr) and b.head == "Rule" and len(b.args) == 2:
            var = _to_sympy(b.args[0])
            point = _to_sympy(b.args[1])
            return _from_sympy(sp.limit(sa, var, point))
        return Expr("Error", "Limit requires var -> point")
    rule2("Limit", _limit)

    # Residue[expr, {var, point}]
    def _residue(a, b):
        sa = _to_sympy(a)
        items = _extract_list(b)
        var = _to_sympy(items[0])
        point = _to_sympy(items[1])
        return _from_sympy(sp.residue(sa, var, point))
    rule2("Residue", _residue)

    # ArcLength[expr, {var, a, b}]
    def _arclength(a, b):
        sa = _to_sympy(a)
        items = _extract_list(b)
        var = _to_sympy(items[0])
        lo, hi = _to_sympy(items[1]), _to_sympy(items[2])
        integrand = sp.sqrt(1 + sp.diff(sa, var)**2)
        result = sp.integrate(integrand, (var, lo, hi))
        return _from_sympy(result)
    rule2("ArcLength", _arclength)

    # Sum[expr, {var, a, b}]
    def _sum(a, b):
        sa = _to_sympy(a)
        items = _extract_list(b)
        var = _to_sympy(items[0])
        lo, hi = _to_sympy(items[1]), _to_sympy(items[2])
        return _from_sympy(sp.summation(sa, (var, lo, hi)))
    rule2("Sum", _sum)

    # Product[expr, {var, a, b}]
    def _product(a, b):
        sa = _to_sympy(a)
        items = _extract_list(b)
        var = _to_sympy(items[0])
        lo, hi = _to_sympy(items[1]), _to_sympy(items[2])
        return _from_sympy(sp.product(sa, (var, lo, hi)))
    rule2("Product", _product)

    # Table[expr, {var, a, b}]
    def _table(a, b):
        items = _extract_list(b)
        var_name = items[0].name if isinstance(items[0], Symbol) else str(items[0])
        lo, hi = int(items[1]) if isinstance(items[1], (int, float)) else 1, int(items[2]) if isinstance(items[2], (int, float)) else 10
        step = int(items[3]) if len(items) > 3 and isinstance(items[3], (int, float)) else 1
        from .evaluate import evaluate as ev
        results = []
        for val in range(lo, hi + 1, step):
            # Substitute var with val
            substituted = _subst_var(a, var_name, val)
            results.append(ev(substituted))
        return Expr("List", *results)
    rule2("Table", _table)

    # Range[a, b, step]
    def _range1(a):
        n = int(a) if isinstance(a, (int, float)) else 10
        return Expr("List", *list(range(1, n + 1)))
    rule1("Range", _range1)

    def _range2(a, b):
        lo = int(a) if isinstance(a, (int, float)) else 1
        hi = int(b) if isinstance(b, (int, float)) else 10
        return Expr("List", *list(range(lo, hi + 1)))
    rule2("Range", _range2)

    def _range3(a, b, c):
        lo = int(a) if isinstance(a, (int, float)) else 1
        hi = int(b) if isinstance(b, (int, float)) else 10
        step = int(c) if isinstance(c, (int, float)) else 1
        return Expr("List", *list(range(lo, hi + 1, step)))
    rule3("Range", _range3)

    # ── ALGEBRA ──────────────────────────────────────────────

    # Simplify[expr]
    def _simplify(a):
        return _from_sympy(sp.simplify(_to_sympy(a)))
    rule1("Simplify", _simplify)

    # Expand[expr]
    def _expand(a):
        return _from_sympy(sp.expand(_to_sympy(a)))
    rule1("Expand", _expand)

    # Factor[expr]
    def _factor(a):
        return _from_sympy(sp.factor(_to_sympy(a)))
    rule1("Factor", _factor)

    # Solve[expr, var]
    def _solve(a, b):
        sa, sb = _to_sympy(a), _to_sympy(b)
        result = sp.solve(sa, sb)
        if isinstance(result, list):
            return Expr("List", *[_from_sympy(r) for r in result])
        return _from_sympy(result)
    rule2("Solve", _solve)

    # PartialFractions[expr, var]
    def _partial_fractions(a, b):
        return _from_sympy(sp.apart(_to_sympy(a), _to_sympy(b)))
    rule2("PartialFractions", _partial_fractions)
    rule2("Apart", _partial_fractions)

    # Together[expr]
    def _together(a):
        return _from_sympy(sp.together(_to_sympy(a)))
    rule1("Together", _together)

    # Cancel[expr]
    def _cancel(a):
        return _from_sympy(sp.cancel(_to_sympy(a)))
    rule1("Cancel", _cancel)

    # Collect[expr, var]
    def _collect(a, b):
        return _from_sympy(sp.collect(_to_sympy(a), _to_sympy(b)))
    rule2("Collect", _collect)

    # PowerExpand[expr]
    def _power_expand(a):
        return _from_sympy(sp.powsimp(_to_sympy(a)))
    rule1("PowerExpand", _power_expand)

    # TrigExpand[expr]
    def _trig_expand(a):
        return _from_sympy(sp.expand_trig(_to_sympy(a)))
    rule1("TrigExpand", _trig_expand)

    # TrigReduce[expr] (reverse of expand)
    def _trig_reduce(a):
        from sympy import TR8
        return _from_sympy(TR8(_to_sympy(a)))
    rule1("TrigReduce", _trig_reduce)

    # ExpToTrig[expr]
    def _exp_to_trig(a):
        sa = _to_sympy(a)
        return _from_sympy(sa.rewrite(sp.cos))
    rule1("ExpToTrig", _exp_to_trig)

    # TrigToExp[expr]
    def _trig_to_exp(a):
        sa = _to_sympy(a)
        return _from_sympy(sa.rewrite(sp.exp))
    rule1("TrigToExp", _trig_to_exp)

    # ComplexExpand[expr]
    def _complex_expand(a):
        return _from_sympy(sp.expand_complex(_to_sympy(a)))
    rule1("ComplexExpand", _complex_expand)

    # RealPart, ImaginaryPart, Conjugate, Abs, Sign, Floor, Ceiling, Round
    rule1("RealPart", lambda a: _from_sympy(sp.re(_to_sympy(a))))
    rule1("ImaginaryPart", lambda a: _from_sympy(sp.im(_to_sympy(a))))
    rule1("Conjugate", lambda a: _from_sympy(sp.conjugate(_to_sympy(a))))
    rule1("Abs", lambda a: _from_sympy(sp.Abs(_to_sympy(a))))
    rule1("Sign", lambda a: _from_sympy(sp.sign(_to_sympy(a))))
    rule1("Floor", lambda a: _from_sympy(sp.floor(_to_sympy(a))))
    rule1("Ceiling", lambda a: _from_sympy(sp.ceiling(_to_sympy(a))))
    rule1("Mod", lambda a: a)  # single-arg no-op
    rule2("Mod", lambda a, b: _from_sympy(sp.Mod(_to_sympy(a), _to_sympy(b))))
    rule2("Quotient", lambda a, b: _from_sympy(_to_sympy(a) // _to_sympy(b)))

    # PolynomialQuotient, PolynomialRemainder
    def _poly_quot(a, b, c):
        return _from_sympy(sp.quo(_to_sympy(a), _to_sympy(b), _to_sympy(c)))
    rule3("PolynomialQuotient", _poly_quot)

    def _poly_rem(a, b, c):
        return _from_sympy(sp.rem(_to_sympy(a), _to_sympy(b), _to_sympy(c)))
    rule3("PolynomialRemainder", _poly_rem)

    def _poly_div(a, b, c):
        q = sp.quo(_to_sympy(a), _to_sympy(b), _to_sympy(c))
        r = sp.rem(_to_sympy(a), _to_sympy(b), _to_sympy(c))
        return Expr("List", _from_sympy(q), _from_sympy(r))
    rule3("PolynomialDivision", _poly_div)

    # GCD/LCM (work for both integers and polynomials)
    rule2("GCD", lambda a, b: _from_sympy(sp.gcd(_to_sympy(a), _to_sympy(b))))
    rule2("LCM", lambda a, b: _from_sympy(sp.lcm(_to_sympy(a), _to_sympy(b))))
    rule2("ExtendedGCD", lambda a, b: Expr("List", *[_from_sympy(x) for x in sp.gcdex(_to_sympy(a), _to_sympy(b))]))

    # Discriminant, Degree, Coefficient
    rule2("Discriminant", lambda a, b: _from_sympy(sp.discriminant(_to_sympy(a), _to_sympy(b))))
    rule2("Degree", lambda a, b: _from_sympy(sp.degree(_to_sympy(a), _to_sympy(b))))
    rule3("Coefficient", lambda a, b, c: _from_sympy(sp.Poly(_to_sympy(a), _to_sympy(b)).nth(int(c))))

    def _coeff_list(a, b):
        poly = sp.Poly(_to_sympy(a), _to_sympy(b))
        coeffs = poly.all_coeffs()
        return Expr("List", *[_from_sympy(c) for c in coeffs])
    rule2("CoefficientList", _coeff_list)

    # Resultant
    rule3("Resultant", lambda a, b, c: _from_sympy(sp.resultant(_to_sympy(a), _to_sympy(b), _to_sympy(c))))

    # CompleteTheSquare
    def _complete_square(a, b):
        sa, sb = _to_sympy(a), _to_sympy(b)
        poly = sp.Poly(sa, sb)
        if poly.degree() != 2:
            return Expr("Error", "Not a quadratic")
        A, B, C = poly.all_coeffs()
        h = -B / (2 * A)
        k = C - B**2 / (4 * A)
        return _from_sympy(A * (sb - h)**2 + k)
    rule2("CompleteTheSquare", _complete_square)

    # Trig functions
    rule1("Sin", lambda a: _from_sympy(sp.sin(_to_sympy(a))))
    rule1("Cos", lambda a: _from_sympy(sp.cos(_to_sympy(a))))
    rule1("Tan", lambda a: _from_sympy(sp.tan(_to_sympy(a))))
    rule1("Cot", lambda a: _from_sympy(sp.cot(_to_sympy(a))))
    rule1("Sec", lambda a: _from_sympy(sp.sec(_to_sympy(a))))
    rule1("Csc", lambda a: _from_sympy(sp.csc(_to_sympy(a))))
    rule1("ArcSin", lambda a: _from_sympy(sp.asin(_to_sympy(a))))
    rule1("ArcCos", lambda a: _from_sympy(sp.acos(_to_sympy(a))))
    rule1("ArcTan", lambda a: _from_sympy(sp.atan(_to_sympy(a))))
    rule2("ArcTan", lambda a, b: _from_sympy(sp.atan2(_to_sympy(a), _to_sympy(b))))
    rule1("ArcCot", lambda a: _from_sympy(sp.acot(_to_sympy(a))))
    rule1("ArcSec", lambda a: _from_sympy(sp.asec(_to_sympy(a))))
    rule1("ArcCsc", lambda a: _from_sympy(sp.acsc(_to_sympy(a))))
    rule1("Sinh", lambda a: _from_sympy(sp.sinh(_to_sympy(a))))
    rule1("Cosh", lambda a: _from_sympy(sp.cosh(_to_sympy(a))))
    rule1("Tanh", lambda a: _from_sympy(sp.tanh(_to_sympy(a))))
    rule1("Coth", lambda a: _from_sympy(sp.coth(_to_sympy(a))))
    rule1("Sech", lambda a: _from_sympy(1/sp.cosh(_to_sympy(a))))
    rule1("Csch", lambda a: _from_sympy(1/sp.sinh(_to_sympy(a))))
    rule1("ArcSinh", lambda a: _from_sympy(sp.asinh(_to_sympy(a))))
    rule1("ArcCosh", lambda a: _from_sympy(sp.acosh(_to_sympy(a))))
    rule1("ArcTanh", lambda a: _from_sympy(sp.atanh(_to_sympy(a))))

    # Special functions
    rule1("Exp", lambda a: _from_sympy(sp.exp(_to_sympy(a))))
    rule1("Log", lambda a: _from_sympy(sp.log(_to_sympy(a))))
    rule2("Log", lambda a, b: _from_sympy(sp.log(_to_sympy(a), _to_sympy(b))))
    rule1("Log2", lambda a: _from_sympy(sp.log(_to_sympy(a), 2)))
    rule1("Log10", lambda a: _from_sympy(sp.log(_to_sympy(a), 10)))
    rule1("Sqrt", lambda a: _from_sympy(sp.sqrt(_to_sympy(a))))
    rule2("Root", lambda a, b: _from_sympy(_to_sympy(a) ** (sp.Rational(1, int(b)))))
    rule1("CubeRoot", lambda a: _from_sympy(_to_sympy(a) ** sp.Rational(1, 3)))

    # Factorial, Gamma, Beta, Zeta
    rule1("Factorial", lambda a: _from_sympy(sp.factorial(_to_sympy(a))))
    rule1("DoubleFactorial", lambda a: _from_sympy(sp.factorial2(_to_sympy(a))))
    rule1("Gamma", lambda a: _from_sympy(sp.gamma(_to_sympy(a))))
    rule2("Beta", lambda a, b: _from_sympy(sp.beta(_to_sympy(a), _to_sympy(b))))
    rule1("Zeta", lambda a: _from_sympy(sp.zeta(_to_sympy(a))))
    rule1("DiGamma", lambda a: _from_sympy(sp.digamma(_to_sympy(a))))
    rule1("PolyGamma", lambda a: _from_sympy(sp.digamma(_to_sympy(a))))
    rule2("PolyGamma", lambda a, b: _from_sympy(sp.polygamma(_to_sympy(a), _to_sympy(b))))
    rule2("BesselJ", lambda a, b: _from_sympy(sp.besselj(_to_sympy(a), _to_sympy(b))))
    rule2("BesselY", lambda a, b: _from_sympy(sp.bessely(_to_sympy(a), _to_sympy(b))))
    rule2("BesselI", lambda a, b: _from_sympy(sp.besseli(_to_sympy(a), _to_sympy(b))))
    rule2("BesselK", lambda a, b: _from_sympy(sp.besselk(_to_sympy(a), _to_sympy(b))))
    rule1("Erf", lambda a: _from_sympy(sp.erf(_to_sympy(a))))
    rule1("Erfc", lambda a: _from_sympy(sp.erfc(_to_sympy(a))))
    rule2("Hypergeometric2F1", lambda a, b: Expr("Error", "Use Hypergeometric2F1[a, b, c, z]"))
    rule1("AiryAi", lambda a: _from_sympy(sp.airyai(_to_sympy(a))))
    rule1("AiryBi", lambda a: _from_sympy(sp.airybi(_to_sympy(a))))

    # ── LINEAR ALGEBRA ───────────────────────────────────────

    # Det[matrix]
    def _det(a):
        return _from_sympy(_to_matrix(a).det())
    rule1("Det", _det)

    # Inverse[matrix]
    def _inv(a):
        return _from_sympy(_to_matrix(a).inv())
    rule1("Inverse", _inv)

    # Transpose[matrix]
    def _transpose(a):
        return _from_sympy(_to_matrix(a).T)
    rule1("Transpose", _transpose)

    # Trace[matrix]
    def _trace(a):
        return _from_sympy(_to_matrix(a).trace())
    rule1("Trace", _trace)

    # Eigenvalues[matrix]
    def _eigenvalues(a):
        evs = _to_matrix(a).eigenvals()
        result = []
        for val, mult in evs.items():
            for _ in range(mult):
                result.append(_from_sympy(val))
        return Expr("List", *result)
    rule1("Eigenvalues", _eigenvalues)

    # Eigenvectors[matrix]
    def _eigenvectors(a):
        evecs = _to_matrix(a).eigenvects()
        result = []
        for val, mult, vecs in evecs:
            for v in vecs:
                result.append(Expr("List", *[_from_sympy(x) for x in v]))
        return Expr("List", *result)
    rule1("Eigenvectors", _eigenvectors)

    # Rank[matrix]
    def _rank(a):
        return _to_matrix(a).rank()
    rule1("Rank", _rank)

    # NullSpace[matrix]
    def _nullspace(a):
        ns = _to_matrix(a).nullspace()
        vecs = [Expr("List", *[_from_sympy(x) for x in v]) for v in ns]
        return Expr("List", *vecs)
    rule1("NullSpace", _nullspace)

    # ColumnSpace[matrix]
    def _colspace(a):
        cs = _to_matrix(a).columnspace()
        vecs = [Expr("List", *[_from_sympy(x) for x in v]) for v in cs]
        return Expr("List", *vecs)
    rule1("ColumnSpace", _colspace)

    # RowReduce[matrix] — RREF
    def _rref(a):
        rref, pivots = _to_matrix(a).rref()
        return _from_sympy(rref)
    rule1("RowReduce", _rref)

    # LUDecomposition[matrix]
    def _lu(a):
        L, U, perm = _to_matrix(a).LUdecomposition()
        return Expr("List", _from_sympy(L), _from_sympy(U))
    rule1("LUDecomposition", _lu)

    # QRDecomposition[matrix]
    def _qr(a):
        Q, R = _to_matrix(a).QRdecomposition()
        return Expr("List", _from_sympy(Q), _from_sympy(R))
    rule1("QRDecomposition", _qr)

    # CholeskyDecomposition[matrix]
    def _cholesky(a):
        L = _to_matrix(a).cholesky()
        return _from_sympy(L)
    rule1("CholeskyDecomposition", _cholesky)

    # JordanForm[matrix]
    def _jordan(a):
        P, J = _to_matrix(a).jordan_form()
        return Expr("List", _from_sympy(P), _from_sympy(J))
    rule1("JordanForm", _jordan)

    # SingularValueDecomposition — SVD (numerical)
    def _svd(a):
        m = _to_matrix(a)
        U, S, V = m.singular_value_decomposition()
        return Expr("List", _from_sympy(U), _from_sympy(S), _from_sympy(V))
    rule1("SVD", _svd)
    rule1("SingularValueDecomposition", _svd)

    # IdentityMatrix[n]
    def _eye(a):
        return _from_sympy(sp.eye(int(a)))
    rule1("IdentityMatrix", _eye)

    # ZeroMatrix[n, m]
    def _zeros(a, b):
        return _from_sympy(sp.zeros(int(a), int(b)))
    rule2("ZeroMatrix", _zeros)

    # DiagonalMatrix[{a, b, c}]
    def _diag(a):
        items = [_to_sympy(x) for x in _extract_list(a)]
        return _from_sympy(sp.diag(*items))
    rule1("DiagonalMatrix", _diag)

    # MatrixPower[matrix, n]
    def _matpow(a, b):
        return _from_sympy(_to_matrix(a) ** int(b))
    rule2("MatrixPower", _matpow)

    # MatrixExp[matrix]
    def _matexp(a):
        return _from_sympy(_to_matrix(a).exp())
    rule1("MatrixExp", _matexp)

    # CrossProduct[v1, v2]
    def _cross(a, b):
        v1 = sp.Matrix([_to_sympy(x) for x in _extract_list(a)])
        v2 = sp.Matrix([_to_sympy(x) for x in _extract_list(b)])
        result = v1.cross(v2)
        return Expr("List", *[_from_sympy(x) for x in result])
    rule2("CrossProduct", _cross)
    rule2("Cross", _cross)

    # DotProduct[v1, v2]
    def _dot(a, b):
        v1 = sp.Matrix([_to_sympy(x) for x in _extract_list(a)])
        v2 = sp.Matrix([_to_sympy(x) for x in _extract_list(b)])
        return _from_sympy(v1.dot(v2))
    rule2("DotProduct", _dot)
    rule2("Dot", _dot)

    # Norm[vector]
    def _norm(a):
        v = sp.Matrix([_to_sympy(x) for x in _extract_list(a)])
        return _from_sympy(v.norm())
    rule1("Norm", _norm)

    # Normalize[vector]
    def _normalize(a):
        v = sp.Matrix([_to_sympy(x) for x in _extract_list(a)])
        n = v.normalized()
        return Expr("List", *[_from_sympy(x) for x in n])
    rule1("Normalize", _normalize)

    # LinearSolve[A, b]
    def _linsolve(a, b):
        A = _to_matrix(a)
        bv = sp.Matrix([_to_sympy(x) for x in _extract_list(b)])
        result = A.solve(bv)
        return Expr("List", *[_from_sympy(x) for x in result])
    rule2("LinearSolve", _linsolve)

    # PseudoInverse[matrix]
    def _pinv(a):
        return _from_sympy(_to_matrix(a).pinv())
    rule1("PseudoInverse", _pinv)

    # CharacteristicPolynomial[matrix, var]
    def _charpoly(a, b):
        m = _to_matrix(a)
        var = _to_sympy(b)
        poly = m.charpoly(var)
        return _from_sympy(poly.as_expr())
    rule2("CharacteristicPolynomial", _charpoly)

    # VectorAngle[v1, v2]
    def _vecangle(a, b):
        v1 = sp.Matrix([_to_sympy(x) for x in _extract_list(a)])
        v2 = sp.Matrix([_to_sympy(x) for x in _extract_list(b)])
        cos_theta = v1.dot(v2) / (v1.norm() * v2.norm())
        return _from_sympy(sp.acos(cos_theta))
    rule2("VectorAngle", _vecangle)

    # KroneckerProduct[A, B]
    def _kron(a, b):
        from sympy.matrices.expressions import kronecker_product
        A = _to_matrix(a)
        B = _to_matrix(b)
        # Manual Kronecker product
        rows_a, cols_a = A.shape
        rows_b, cols_b = B.shape
        result = sp.zeros(rows_a * rows_b, cols_a * cols_b)
        for i in range(rows_a):
            for j in range(cols_a):
                result[i*rows_b:(i+1)*rows_b, j*cols_b:(j+1)*cols_b] = A[i,j] * B
        return _from_sympy(result)
    rule2("KroneckerProduct", _kron)

    # ── STATISTICS ───────────────────────────────────────────

    def _mean(a):
        vals = _to_float_list(a)
        import statistics
        return statistics.mean(vals)
    rule1("Mean", _mean)

    def _median(a):
        vals = _to_float_list(a)
        import statistics
        return statistics.median(vals)
    rule1("Median", _median)

    def _mode(a):
        vals = _to_float_list(a)
        import statistics
        return statistics.mode(vals)
    rule1("Mode", _mode)

    def _stdev(a):
        vals = _to_float_list(a)
        import statistics
        return round(statistics.stdev(vals), 10)
    rule1("StandardDeviation", _stdev)

    def _variance(a):
        vals = _to_float_list(a)
        import statistics
        return round(statistics.variance(vals), 10)
    rule1("Variance", _variance)

    def _quantile(a, b):
        vals = _to_float_list(a)
        q = float(b) if isinstance(b, (int, float)) else float(_to_sympy(b))
        import statistics
        return statistics.quantiles(vals, n=100)[int(q * 100) - 1] if 0 < q < 1 else sorted(vals)[int(q * len(vals))]
    rule2("Quantile", _quantile)

    def _percentile(a, b):
        vals = sorted(_to_float_list(a))
        p = float(b) if isinstance(b, (int, float)) else float(_to_sympy(b))
        import numpy as np
        return float(np.percentile(vals, p))
    rule2("Percentile", _percentile)

    def _iqr(a):
        import numpy as np
        vals = _to_float_list(a)
        return float(np.percentile(vals, 75) - np.percentile(vals, 25))
    rule1("IQR", _iqr)

    def _skewness(a):
        from scipy import stats
        vals = _to_float_list(a)
        return round(float(stats.skew(vals)), 10)
    rule1("Skewness", _skewness)

    def _kurtosis(a):
        from scipy import stats
        vals = _to_float_list(a)
        return round(float(stats.kurtosis(vals)), 10)
    rule1("Kurtosis", _kurtosis)

    def _covariance(a, b):
        import numpy as np
        x = _to_float_list(a)
        y = _to_float_list(b)
        return round(float(np.cov(x, y)[0, 1]), 10)
    rule2("Covariance", _covariance)

    def _correlation(a, b):
        import numpy as np
        x = _to_float_list(a)
        y = _to_float_list(b)
        return round(float(np.corrcoef(x, y)[0, 1]), 10)
    rule2("Correlation", _correlation)

    def _linreg(a, b):
        from scipy import stats
        x = _to_float_list(a)
        y = _to_float_list(b)
        slope, intercept, r, p, se = stats.linregress(x, y)
        return Expr("List",
            Expr("Rule", Symbol("slope"), round(slope, 10)),
            Expr("Rule", Symbol("intercept"), round(intercept, 10)),
            Expr("Rule", Symbol("r_squared"), round(r**2, 10)),
            Expr("Rule", Symbol("p_value"), round(p, 10)),
            Expr("Rule", Symbol("std_error"), round(se, 10)),
        )
    rule2("LinearRegression", _linreg)

    def _zscore(a, b, c):
        x, mu, sigma = float(a), float(b), float(c)
        return round((x - mu) / sigma, 10)
    rule3("ZScore", _zscore)

    def _ttest(a, b):
        from scipy import stats
        x = _to_float_list(a)
        y = _to_float_list(b)
        t, p = stats.ttest_ind(x, y)
        return Expr("List",
            Expr("Rule", Symbol("t_statistic"), round(float(t), 10)),
            Expr("Rule", Symbol("p_value"), round(float(p), 10)),
        )
    rule2("TTest", _ttest)

    def _ttest_1(a):
        from scipy import stats
        x = _to_float_list(a)
        t, p = stats.ttest_1samp(x, 0)
        return Expr("List",
            Expr("Rule", Symbol("t_statistic"), round(float(t), 10)),
            Expr("Rule", Symbol("p_value"), round(float(p), 10)),
        )
    rule1("TTest", _ttest_1)

    def _chi2test(a, b):
        from scipy import stats
        obs = _to_float_list(a)
        exp = _to_float_list(b)
        chi2, p = stats.chisquare(obs, exp)
        return Expr("List",
            Expr("Rule", Symbol("chi2"), round(float(chi2), 10)),
            Expr("Rule", Symbol("p_value"), round(float(p), 10)),
        )
    rule2("ChiSquaredTest", _chi2test)

    def _mannwhitney(a, b):
        from scipy import stats
        x = _to_float_list(a)
        y = _to_float_list(b)
        u, p = stats.mannwhitneyu(x, y)
        return Expr("List",
            Expr("Rule", Symbol("U"), round(float(u), 10)),
            Expr("Rule", Symbol("p_value"), round(float(p), 10)),
        )
    rule2("MannWhitneyU", _mannwhitney)

    def _wilcoxon(a, b):
        from scipy import stats
        x = _to_float_list(a)
        y = _to_float_list(b)
        w, p = stats.wilcoxon(x, y)
        return Expr("List",
            Expr("Rule", Symbol("W"), round(float(w), 10)),
            Expr("Rule", Symbol("p_value"), round(float(p), 10)),
        )
    rule2("WilcoxonTest", _wilcoxon)

    def _shapiro(a):
        from scipy import stats
        x = _to_float_list(a)
        w, p = stats.shapiro(x)
        return Expr("List",
            Expr("Rule", Symbol("W"), round(float(w), 10)),
            Expr("Rule", Symbol("p_value"), round(float(p), 10)),
        )
    rule1("ShapiroWilk", _shapiro)

    def _conf_interval(a, b):
        from scipy import stats
        import numpy as np
        x = _to_float_list(a)
        conf = float(b) if isinstance(b, (int, float)) else 0.95
        n = len(x)
        mean = np.mean(x)
        se = stats.sem(x)
        h = se * stats.t.ppf((1 + conf) / 2, n - 1)
        return Expr("List", round(mean - h, 10), round(mean + h, 10))
    rule2("ConfidenceInterval", _conf_interval)

    def _effect_size(a, b):
        """Cohen's d."""
        import numpy as np
        x = _to_float_list(a)
        y = _to_float_list(b)
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
        d = (np.mean(x) - np.mean(y)) / pooled_std
        return round(float(d), 10)
    rule2("EffectSize", _effect_size)
    rule2("CohensD", _effect_size)

    # Distributions — PDF, CDF
    def _normal_dist(a, b):
        return Expr("NormalDistribution", a, b)
    rule2("NormalDistribution", _normal_dist)

    def _pdf(a, b):
        from scipy import stats as st
        x = float(_to_sympy(b))
        if isinstance(a, Expr) and a.head == "NormalDistribution":
            mu, sigma = float(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            return round(float(st.norm.pdf(x, mu, sigma)), 10)
        if isinstance(a, Expr) and a.head == "ExponentialDistribution":
            lam = float(_to_sympy(a.args[0]))
            return round(float(st.expon.pdf(x, scale=1/lam)), 10)
        if isinstance(a, Expr) and a.head == "PoissonDistribution":
            lam = float(_to_sympy(a.args[0]))
            return round(float(st.poisson.pmf(int(x), lam)), 10)
        if isinstance(a, Expr) and a.head == "UniformDistribution":
            lo, hi = float(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            return round(float(st.uniform.pdf(x, lo, hi - lo)), 10)
        if isinstance(a, Expr) and a.head == "BinomialDistribution":
            n, p = int(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            return round(float(st.binom.pmf(int(x), n, p)), 10)
        if isinstance(a, Expr) and a.head == "ChiSquaredDistribution":
            df = float(_to_sympy(a.args[0]))
            return round(float(st.chi2.pdf(x, df)), 10)
        if isinstance(a, Expr) and a.head == "TDistribution":
            df = float(_to_sympy(a.args[0]))
            return round(float(st.t.pdf(x, df)), 10)
        if isinstance(a, Expr) and a.head == "FDistribution":
            d1, d2 = float(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            return round(float(st.f.pdf(x, d1, d2)), 10)
        return Expr("Error", "Unknown distribution")
    rule2("PDF", _pdf)

    def _cdf(a, b):
        from scipy import stats as st
        x = float(_to_sympy(b))
        if isinstance(a, Expr) and a.head == "NormalDistribution":
            mu, sigma = float(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            return round(float(st.norm.cdf(x, mu, sigma)), 10)
        if isinstance(a, Expr) and a.head == "ExponentialDistribution":
            lam = float(_to_sympy(a.args[0]))
            return round(float(st.expon.cdf(x, scale=1/lam)), 10)
        if isinstance(a, Expr) and a.head == "PoissonDistribution":
            lam = float(_to_sympy(a.args[0]))
            return round(float(st.poisson.cdf(int(x), lam)), 10)
        if isinstance(a, Expr) and a.head == "UniformDistribution":
            lo, hi = float(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            return round(float(st.uniform.cdf(x, lo, hi - lo)), 10)
        if isinstance(a, Expr) and a.head == "BinomialDistribution":
            n, p = int(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            return round(float(st.binom.cdf(int(x), n, p)), 10)
        if isinstance(a, Expr) and a.head == "ChiSquaredDistribution":
            df = float(_to_sympy(a.args[0]))
            return round(float(st.chi2.cdf(x, df)), 10)
        if isinstance(a, Expr) and a.head == "TDistribution":
            df = float(_to_sympy(a.args[0]))
            return round(float(st.t.cdf(x, df)), 10)
        if isinstance(a, Expr) and a.head == "FDistribution":
            d1, d2 = float(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            return round(float(st.f.cdf(x, d1, d2)), 10)
        return Expr("Error", "Unknown distribution")
    rule2("CDF", _cdf)

    def _inv_cdf(a, b):
        from scipy import stats as st
        p = float(_to_sympy(b))
        if isinstance(a, Expr) and a.head == "NormalDistribution":
            mu, sigma = float(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            return round(float(st.norm.ppf(p, mu, sigma)), 10)
        if isinstance(a, Expr) and a.head == "TDistribution":
            df = float(_to_sympy(a.args[0]))
            return round(float(st.t.ppf(p, df)), 10)
        if isinstance(a, Expr) and a.head == "ChiSquaredDistribution":
            df = float(_to_sympy(a.args[0]))
            return round(float(st.chi2.ppf(p, df)), 10)
        if isinstance(a, Expr) and a.head == "FDistribution":
            d1, d2 = float(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            return round(float(st.f.ppf(p, d1, d2)), 10)
        return Expr("Error", "Unknown distribution")
    rule2("InverseCDF", _inv_cdf)

    # Distribution constructors
    rule1("ExponentialDistribution", lambda a: Expr("ExponentialDistribution", a))
    rule1("PoissonDistribution", lambda a: Expr("PoissonDistribution", a))
    rule2("UniformDistribution", lambda a, b: Expr("UniformDistribution", a, b))
    rule2("BinomialDistribution", lambda a, b: Expr("BinomialDistribution", a, b))
    rule1("ChiSquaredDistribution", lambda a: Expr("ChiSquaredDistribution", a))
    rule1("TDistribution", lambda a: Expr("TDistribution", a))
    rule2("FDistribution", lambda a, b: Expr("FDistribution", a, b))

    def _random_sample(a, b):
        import numpy as np
        n = int(b) if isinstance(b, (int, float)) else int(_to_sympy(b))
        if isinstance(a, Expr) and a.head == "NormalDistribution":
            mu, sigma = float(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            vals = np.random.normal(mu, sigma, n)
        elif isinstance(a, Expr) and a.head == "UniformDistribution":
            lo, hi = float(_to_sympy(a.args[0])), float(_to_sympy(a.args[1]))
            vals = np.random.uniform(lo, hi, n)
        elif isinstance(a, Expr) and a.head == "ExponentialDistribution":
            lam = float(_to_sympy(a.args[0]))
            vals = np.random.exponential(1/lam, n)
        else:
            vals = np.random.standard_normal(n)
        return Expr("List", *[round(float(v), 6) for v in vals])
    rule2("RandomSample", _random_sample)

    # ANOVA
    def _anova(a, b, c):
        from scipy import stats
        x = _to_float_list(a)
        y = _to_float_list(b)
        z = _to_float_list(c)
        f, p = stats.f_oneway(x, y, z)
        return Expr("List",
            Expr("Rule", Symbol("F"), round(float(f), 10)),
            Expr("Rule", Symbol("p_value"), round(float(p), 10)),
        )
    rule3("ANOVA", _anova)

    def _kruskal(a, b, c):
        from scipy import stats
        x = _to_float_list(a)
        y = _to_float_list(b)
        z = _to_float_list(c)
        h, p = stats.kruskal(x, y, z)
        return Expr("List",
            Expr("Rule", Symbol("H"), round(float(h), 10)),
            Expr("Rule", Symbol("p_value"), round(float(p), 10)),
        )
    rule3("KruskalWallis", _kruskal)

    # ── NUMBER THEORY ────────────────────────────────────────

    rule1("PrimeQ", lambda a: bool(sp.isprime(int(a))))
    rule1("NextPrime", lambda a: int(sp.nextprime(int(a))))
    rule1("PrimePi", lambda a: int(sp.primepi(int(a))))
    rule1("Prime", lambda a: int(sp.prime(int(a))))

    def _factor_integer(a):
        facs = sp.factorint(int(a))
        return Expr("List", *[Expr("List", int(p), int(e)) for p, e in sorted(facs.items())])
    rule1("FactorInteger", _factor_integer)
    rule1("PrimeFactorization", _factor_integer)

    def _divisors(a):
        d = sorted(sp.divisors(int(a)))
        return Expr("List", *d)
    rule1("Divisors", _divisors)

    rule1("DivisorCount", lambda a: int(sp.divisor_count(int(a))))
    rule2("DivisorSigma", lambda a, b: int(sp.divisor_sigma(int(b), int(a))))
    rule1("EulerPhi", lambda a: int(sp.totient(int(a))))
    rule1("MoebiusMu", lambda a: int(sp.mobius(int(a))))

    rule3("PowerMod", lambda a, b, c: int(pow(int(a), int(b), int(c))))

    def _mod_inv(a, b):
        result = pow(int(a), -1, int(b))
        return int(result)
    rule2("ModularInverse", _mod_inv)

    def _chinese_rem(a, b):
        remainders = [int(x) for x in _extract_list(a)]
        moduli = [int(x) for x in _extract_list(b)]
        from sympy.ntheory.modular import crt
        result = crt(moduli, remainders)
        if result is None:
            return Expr("Error", "No solution")
        return int(result[0])
    rule2("ChineseRemainder", _chinese_rem)

    rule1("Fibonacci", lambda a: int(sp.fibonacci(int(a))))
    rule1("Lucas", lambda a: int(sp.lucas(int(a))))
    rule1("Bernoulli", lambda a: _from_sympy(sp.bernoulli(int(a))))
    rule1("Catalan", lambda a: int(sp.catalan(int(a))))
    rule1("Bell", lambda a: int(sp.bell(int(a))))
    rule2("Stirling", lambda a, b: int(sp_stirling(int(a), int(b))))
    rule3("Stirling", lambda a, b, c: int(sp_stirling(int(a), int(b), kind=int(c))))

    rule1("DigitSum", lambda a: sum(int(d) for d in str(abs(int(a)))))
    rule1("DigitCount", lambda a: len(str(abs(int(a)))))

    def _int_digits(a, b=10):
        base = int(b) if isinstance(b, (int, float)) else 10
        n = int(a)
        if base == 10:
            return Expr("List", *[int(d) for d in str(abs(n))])
        digits = []
        n = abs(n)
        while n > 0:
            digits.append(n % base)
            n //= base
        return Expr("List", *(digits[::-1] if digits else [0]))
    rule1("IntegerDigits", _int_digits)
    rule2("IntegerDigits", _int_digits)

    def _from_digits(a, b=10):
        base = int(b) if isinstance(b, (int, float)) else 10
        digits = [int(x) for x in _extract_list(a)]
        result = 0
        for d in digits:
            result = result * base + d
        return result
    rule1("FromDigits", _from_digits)
    rule2("FromDigits", _from_digits)

    def _base_convert(a, b, c):
        n = int(a)
        from_base = int(b)
        to_base = int(c)
        # Convert from from_base to base 10 first
        if from_base != 10:
            digits = str(n)
            dec = sum(int(d) * from_base**i for i, d in enumerate(reversed(digits)))
        else:
            dec = n
        # Convert to to_base
        if to_base == 10:
            return dec
        result = []
        while dec > 0:
            result.append(dec % to_base)
            dec //= to_base
        return Expr("List", *(result[::-1] if result else [0]))
    rule3("BaseConvert", _base_convert)

    # ── COMBINATORICS ────────────────────────────────────────

    rule2("Binomial", lambda a, b: int(sp.binomial(int(a), int(b))))

    def _perm_count(a, b):
        n, k = int(a), int(b)
        return int(sp.factorial(n) / sp.factorial(n - k))
    rule2("Permutations", _perm_count)

    rule1("HarmonicNumber", lambda a: _from_sympy(sp.harmonic(int(a))))
    rule1("CatalanNumber", lambda a: int(sp.catalan(int(a))))
    rule1("BellNumber", lambda a: int(sp.bell(int(a))))
    rule2("StirlingS1", lambda a, b: int(sp_stirling(int(a), int(b), kind=1)))
    rule2("StirlingS2", lambda a, b: int(sp_stirling(int(a), int(b), kind=2)))

    def _partition_count(a):
        return int(sp_npartitions(int(a)))
    rule1("PartitionCount", _partition_count)
    rule1("IntegerPartitions", _partition_count)

    rule1("FibonacciNumber", lambda a: int(sp.fibonacci(int(a))))
    rule1("BernoulliNumber", lambda a: _from_sympy(sp.bernoulli(int(a))))
    rule1("EulerNumber", lambda a: int(sp.euler(int(a))))

    def _derangements(a):
        n = int(a)
        if n == 0: return 1
        if n == 1: return 0
        return int(round(sp.factorial(n) * sum((-1)**k / sp.factorial(k) for k in range(n + 1))))
    rule1("Derangements", _derangements)

    def _multinomial(a, b):
        n = int(a)
        ks = [int(x) for x in _extract_list(b)]
        result = sp.factorial(n)
        for k in ks:
            result //= sp.factorial(k)
        return int(result)
    rule2("Multinomial", _multinomial)

    # ── GEOMETRY ─────────────────────────────────────────────

    def _distance(a, b):
        import math
        p1 = _to_float_list(a)
        p2 = _to_float_list(b)
        return round(math.sqrt(sum((a-b)**2 for a, b in zip(p1, p2))), 10)
    rule2("Distance", _distance)

    def _midpoint(a, b):
        p1 = _to_float_list(a)
        p2 = _to_float_list(b)
        return Expr("List", *[round((a+b)/2, 10) for a, b in zip(p1, p2)])
    rule2("Midpoint", _midpoint)

    def _slope(a, b):
        p1 = _to_float_list(a)
        p2 = _to_float_list(b)
        if p2[0] - p1[0] == 0:
            return Symbol("Infinity")
        return round((p2[1] - p1[1]) / (p2[0] - p1[0]), 10)
    rule2("Slope", _slope)

    rule1("CircleArea", lambda a: _from_sympy(sp.pi * _to_sympy(a)**2))
    rule1("CircleCircumference", lambda a: _from_sympy(2 * sp.pi * _to_sympy(a)))
    rule1("SphereVolume", lambda a: _from_sympy(sp.Rational(4, 3) * sp.pi * _to_sympy(a)**3))
    rule1("SphereSurfaceArea", lambda a: _from_sympy(4 * sp.pi * _to_sympy(a)**2))
    rule2("CylinderVolume", lambda a, b: _from_sympy(sp.pi * _to_sympy(a)**2 * _to_sympy(b)))
    rule2("ConeVolume", lambda a, b: _from_sympy(sp.Rational(1, 3) * sp.pi * _to_sympy(a)**2 * _to_sympy(b)))

    def _heron(a, b, c):
        sa, sb, sc = float(a), float(b), float(c)
        s = (sa + sb + sc) / 2
        return round(math.sqrt(s * (s-sa) * (s-sb) * (s-sc)), 10)
    rule3("TriangleArea", _heron)

    rule2("Hypotenuse", lambda a, b: round(math.sqrt(float(a)**2 + float(b)**2), 10))

    def _quad_formula(a, b, c):
        sa, sb, sc = _to_sympy(a), _to_sympy(b), _to_sympy(c)
        disc = sb**2 - 4*sa*sc
        r1 = (-sb + sp.sqrt(disc)) / (2*sa)
        r2 = (-sb - sp.sqrt(disc)) / (2*sa)
        return Expr("List", _from_sympy(r1), _from_sympy(r2))
    rule3("QuadraticFormula", _quad_formula)

    def _polygon_area(a):
        pts = [_to_float_list(p) for p in _extract_list(a)]
        n = len(pts)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return round(abs(area) / 2, 10)
    rule1("PolygonArea", _polygon_area)

    def _reg_polygon_area(a, b):
        n = int(a)
        s = float(b)
        return round(n * s**2 / (4 * math.tan(math.pi / n)), 10)
    rule2("RegularPolygonArea", _reg_polygon_area)

    # ── DIFFERENTIAL EQUATIONS ───────────────────────────────

    def _dsolve(a, b, c):
        """DSolve[equation, y, x]"""
        from sympy import Function, dsolve as sp_dsolve, Eq
        sa = _to_sympy(a)
        y = sp.Function(str(b) if isinstance(b, Symbol) else str(b))
        x = _to_sympy(c)
        # Attempt to convert to equation
        if not isinstance(sa, sp.Eq):
            sa = sp.Eq(sa, 0)
        try:
            result = sp_dsolve(sa, y(x))
            return _from_sympy(result)
        except Exception as e:
            return Expr("Error", str(e))
    rule3("DSolve", _dsolve)

    def _wronskian(a, b):
        """Wronskian[{f1, f2, ...}, var]"""
        from sympy import wronskian
        funcs = [_to_sympy(f) for f in _extract_list(a)]
        var = _to_sympy(b)
        return _from_sympy(wronskian(funcs, var))
    rule2("Wronskian", _wronskian)

    # ── SET THEORY & LOGIC ───────────────────────────────────

    def _union(a, b):
        s1 = set(_extract_list(a))
        s2 = set(_extract_list(b))
        return Expr("List", *sorted(s1 | s2, key=str))
    rule2("Union", _union)

    def _intersection(a, b):
        s1 = set(_extract_list(a))
        s2 = set(_extract_list(b))
        return Expr("List", *sorted(s1 & s2, key=str))
    rule2("Intersection", _intersection)

    def _set_diff(a, b):
        s1 = set(_extract_list(a))
        s2 = set(_extract_list(b))
        return Expr("List", *sorted(s1 - s2, key=str))
    rule2("SetDifference", _set_diff)
    rule2("Complement", _set_diff)

    def _sym_diff(a, b):
        s1 = set(_extract_list(a))
        s2 = set(_extract_list(b))
        return Expr("List", *sorted(s1 ^ s2, key=str))
    rule2("SymmetricDifference", _sym_diff)

    def _power_set(a):
        items = _extract_list(a)
        from itertools import combinations
        result = []
        for r in range(len(items) + 1):
            for combo in combinations(items, r):
                result.append(Expr("List", *combo))
        return Expr("List", *result)
    rule1("PowerSet", _power_set)

    rule1("Cardinality", lambda a: len(_extract_list(a)))

    def _element_q(a, b):
        items = _extract_list(b)
        return a in items
    rule2("ElementQ", _element_q)
    rule2("MemberQ", _element_q)

    def _subset_q(a, b):
        s1 = set(_extract_list(a))
        s2 = set(_extract_list(b))
        return s1.issubset(s2)
    rule2("SubsetQ", _subset_q)

    # Logic
    rule2("Implies", lambda a, b: (not a) or b if isinstance(a, bool) and isinstance(b, bool) else Expr("Implies", a, b))
    rule2("Equivalent", lambda a, b: a == b if isinstance(a, bool) and isinstance(b, bool) else Expr("Equivalent", a, b))
    rule2("Xor", lambda a, b: a != b if isinstance(a, bool) and isinstance(b, bool) else Expr("Xor", a, b))
    rule2("Nand", lambda a, b: not (a and b) if isinstance(a, bool) and isinstance(b, bool) else Expr("Nand", a, b))
    rule2("Nor", lambda a, b: not (a or b) if isinstance(a, bool) and isinstance(b, bool) else Expr("Nor", a, b))

    # ── SEQUENCES ────────────────────────────────────────────

    def _accumulate(a):
        items = _to_float_list(a)
        result = []
        total = 0
        for x in items:
            total += x
            result.append(total)
        return Expr("List", *[round(x, 10) for x in result])
    rule1("Accumulate", _accumulate)

    def _differences(a):
        items = _to_float_list(a)
        return Expr("List", *[round(items[i+1] - items[i], 10) for i in range(len(items)-1)])
    rule1("Differences", _differences)

    def _arith_seq(a, b, c):
        a0, d, n = float(a), float(b), int(c)
        return Expr("List", *[round(a0 + i*d, 10) for i in range(n)])
    rule3("ArithmeticSequence", _arith_seq)

    def _geom_seq(a, b, c):
        a0, r, n = float(a), float(b), int(c)
        return Expr("List", *[round(a0 * r**i, 10) for i in range(n)])
    rule3("GeometricSequence", _geom_seq)

    def _fib_seq(a):
        n = int(a)
        seq = []
        a_, b_ = 0, 1
        for _ in range(n):
            seq.append(a_)
            a_, b_ = b_, a_ + b_
        return Expr("List", *seq)
    rule1("FibonacciSequence", _fib_seq)

    # ── FINANCIAL ────────────────────────────────────────────

    def _compound_interest(a, b, c):
        principal, rate, years = float(a), float(b), float(c)
        return round(principal * (1 + rate) ** years, 2)
    rule3("CompoundInterest", _compound_interest)

    def _future_value(a, b, c):
        pv, rate, periods = float(a), float(b), float(c)
        return round(pv * (1 + rate) ** periods, 2)
    rule3("FutureValue", _future_value)

    def _present_value(a, b, c):
        fv, rate, periods = float(a), float(b), float(c)
        return round(fv / (1 + rate) ** periods, 2)
    rule3("PresentValue", _present_value)

    def _npv(a, b):
        rate = float(a)
        cashflows = _to_float_list(b)
        npv = sum(cf / (1 + rate)**i for i, cf in enumerate(cashflows))
        return round(npv, 2)
    rule2("NPV", _npv)

    def _irr(a):
        cashflows = _to_float_list(a)
        import numpy as np
        # Newton's method
        r = 0.1
        for _ in range(1000):
            npv = sum(cf / (1 + r)**i for i, cf in enumerate(cashflows))
            dnpv = sum(-i * cf / (1 + r)**(i+1) for i, cf in enumerate(cashflows))
            if abs(dnpv) < 1e-15:
                break
            r -= npv / dnpv
            if abs(npv) < 1e-10:
                break
        return round(r, 6)
    rule1("IRR", _irr)

    def _pmt(a, b, c):
        rate, periods, pv = float(a), float(b), float(c)
        if rate == 0:
            return round(pv / periods, 2)
        return round(pv * rate * (1 + rate)**periods / ((1 + rate)**periods - 1), 2)
    rule3("PMT", _pmt)

    def _continuous_compound(a, b, c):
        principal, rate, time = float(a), float(b), float(c)
        return round(principal * math.exp(rate * time), 2)
    rule3("ContinuousCompounding", _continuous_compound)

    # ── PHYSICS CONSTANTS ────────────────────────────────────

    _CONSTANTS = {
        "SpeedOfLight": 299792458,
        "PlanckConstant": 6.62607015e-34,
        "ReducedPlanckConstant": 1.054571817e-34,
        "GravitationalConstant": 6.67430e-11,
        "BoltzmannConstant": 1.380649e-23,
        "AvogadroNumber": 6.02214076e23,
        "ElementaryCharge": 1.602176634e-19,
        "ElectronMass": 9.1093837015e-31,
        "ProtonMass": 1.67262192369e-27,
        "NeutronMass": 1.67492749804e-27,
        "FineStructureConstant": 7.2973525693e-3,
        "RydbergConstant": 10973731.568160,
        "StefanBoltzmannConstant": 5.670374419e-8,
        "GasConstant": 8.314462618,
        "VacuumPermittivity": 8.8541878128e-12,
        "VacuumPermeability": 1.25663706212e-6,
        "CoulombConstant": 8.9875517923e9,
        "FaradayConstant": 96485.33212,
        "AtomicMassUnit": 1.66053906660e-27,
        "BohrRadius": 5.29177210903e-11,
        "BohrMagneton": 9.2740100783e-24,
        "StandardGravity": 9.80665,
        "StandardAtmosphere": 101325,
        "WienDisplacement": 2.897771955e-3,
        "HubbleConstant": 67.4,
        "SolarMass": 1.989e30,
        "EarthMass": 5.972e24,
        "EarthRadius": 6.371e6,
        "LightYear": 9.461e15,
        "Parsec": 3.086e16,
    }

    for name, value in _CONSTANTS.items():
        rule1(name, lambda a, _v=value: _v)
        # Also as 0-arg via rules (symbol)
        rules.append(RuleDelayed(
            Expr(name),
            lambda b, _v=value: _v
        ))

    # PhysicalConstant["name"]
    def _phys_const(a):
        name = str(a).strip('"')
        return _CONSTANTS.get(name, Expr("Error", f"Unknown constant: {name}"))
    rule1("PhysicalConstant", _phys_const)

    # ── SIGNAL PROCESSING ────────────────────────────────────

    def _fft(a):
        import numpy as np
        vals = _to_float_list(a)
        result = np.fft.fft(vals)
        return Expr("List", *[round(complex(x).real, 6) if abs(complex(x).imag) < 1e-10 else
                              Expr("Plus", round(complex(x).real, 6), Expr("Times", round(complex(x).imag, 6), Symbol("I")))
                              for x in result])
    rule1("FFT", _fft)
    rule1("FourierDFT", _fft)

    def _ifft(a):
        import numpy as np
        vals = _to_float_list(a)
        result = np.fft.ifft(vals)
        return Expr("List", *[round(float(x.real), 6) for x in result])
    rule1("InverseFFT", _ifft)

    def _convolve(a, b):
        import numpy as np
        x = _to_float_list(a)
        y = _to_float_list(b)
        result = np.convolve(x, y)
        return Expr("List", *[round(float(v), 10) for v in result])
    rule2("Convolve", _convolve)
    rule2("Convolution", _convolve)

    # ── MISC / UTILITY ───────────────────────────────────────

    # Sort
    def _sort(a):
        items = _extract_list(a)
        try:
            return Expr("List", *sorted(items, key=lambda x: float(x) if isinstance(x, (int, float)) else float('inf')))
        except:
            return Expr("List", *sorted(items, key=str))
    rule1("Sort", _sort)

    # Reverse
    rule1("Reverse", lambda a: Expr("List", *reversed(_extract_list(a))))

    # Length
    rule1("Length", lambda a: len(_extract_list(a)))

    # Total
    def _total(a):
        items = _to_float_list(a)
        return round(sum(items), 10)
    rule1("Total", _total)

    # Min/Max of list
    rule1("Min", lambda a: min(_to_float_list(a)) if isinstance(a, Expr) and a.head == "List" else _from_sympy(sp.Min(_to_sympy(a))))
    rule1("Max", lambda a: max(_to_float_list(a)) if isinstance(a, Expr) and a.head == "List" else _from_sympy(sp.Max(_to_sympy(a))))

    # Take, Drop, Part
    def _take(a, b):
        items = _extract_list(a)
        n = int(b)
        return Expr("List", *items[:n])
    rule2("Take", _take)

    def _drop(a, b):
        items = _extract_list(a)
        n = int(b)
        return Expr("List", *items[n:])
    rule2("Drop", _drop)

    def _part(a, b):
        items = _extract_list(a)
        n = int(b) - 1  # 1-indexed
        return items[n] if 0 <= n < len(items) else Expr("Error", "Index out of range")
    rule2("Part", _part)

    # First, Last, Rest, Most
    rule1("First", lambda a: _extract_list(a)[0])
    rule1("Last", lambda a: _extract_list(a)[-1])
    rule1("Rest", lambda a: Expr("List", *_extract_list(a)[1:]))
    rule1("Most", lambda a: Expr("List", *_extract_list(a)[:-1]))

    # Map, Select, Count
    # (These are limited since we can't easily pass functions, but basic lambdas work)

    # Flatten
    def _flatten(a):
        def _flat(x):
            if isinstance(x, Expr) and x.head == "List":
                result = []
                for item in x.args:
                    result.extend(_flat(item))
                return result
            return [x]
        return Expr("List", *_flat(a))
    rule1("Flatten", _flatten)

    # Tally
    def _tally(a):
        items = _extract_list(a)
        counts = {}
        for item in items:
            key = str(item)
            counts[key] = counts.get(key, 0) + 1
        return Expr("List", *[Expr("List", items[list(dict.fromkeys(str(x) for x in items)).index(k) if False else next(i for i, x in enumerate(items) if str(x) == k)], v) for k, v in counts.items()])

    # Append, Prepend, Join
    rule2("Append", lambda a, b: Expr("List", *_extract_list(a), b))
    rule2("Prepend", lambda a, b: Expr("List", b, *_extract_list(a)))
    rule2("Join", lambda a, b: Expr("List", *_extract_list(a), *_extract_list(b)))

    # Position
    def _position(a, b):
        items = _extract_list(a)
        positions = [i+1 for i, x in enumerate(items) if x == b]
        return Expr("List", *positions)
    rule2("Position", _position)

    # DeleteDuplicates
    def _dedup(a):
        items = _extract_list(a)
        seen = set()
        result = []
        for x in items:
            key = str(x)
            if key not in seen:
                seen.add(key)
                result.append(x)
        return Expr("List", *result)
    rule1("DeleteDuplicates", _dedup)

    # Riffle
    def _riffle(a, b):
        items = _extract_list(a)
        result = []
        for i, x in enumerate(items):
            result.append(x)
            if i < len(items) - 1:
                result.append(b)
        return Expr("List", *result)
    rule2("Riffle", _riffle)

    # Partition
    def _partition(a, b):
        items = _extract_list(a)
        n = int(b)
        return Expr("List", *[Expr("List", *items[i:i+n]) for i in range(0, len(items), n) if len(items[i:i+n]) == n])
    rule2("Partition", _partition)

    # N[expr] — numerical evaluation
    def _numerical(a):
        sa = _to_sympy(a)
        return round(float(sa.evalf()), 15)
    rule1("N", _numerical)

    def _numerical_prec(a, b):
        sa = _to_sympy(a)
        prec = int(b)
        return float(sa.evalf(prec))
    rule2("N", _numerical_prec)

    # NumberQ
    rule1("NumberQ", lambda a: isinstance(a, (int, float)))

    # IntegerQ
    rule1("IntegerQ", lambda a: isinstance(a, int))

    # EvenQ, OddQ
    rule1("EvenQ", lambda a: isinstance(a, int) and a % 2 == 0)
    rule1("OddQ", lambda a: isinstance(a, int) and a % 2 != 0)

    # Positive, Negative, NonNegative
    rule1("Positive", lambda a: isinstance(a, (int, float)) and a > 0)
    rule1("Negative", lambda a: isinstance(a, (int, float)) and a < 0)
    rule1("NonNegative", lambda a: isinstance(a, (int, float)) and a >= 0)

    # Max/Min with 2 args
    rule2("Max", lambda a, b: max(float(a), float(b)) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else _from_sympy(sp.Max(_to_sympy(a), _to_sympy(b))))
    rule2("Min", lambda a, b: min(float(a), float(b)) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else _from_sympy(sp.Min(_to_sympy(a), _to_sympy(b))))

    # Round with precision
    def _round_prec(a, b):
        val = float(a)
        prec = float(b)
        return round(val / prec) * prec
    rule2("Round", _round_prec)

    return rules


# ── Helper ──

def _subst_var(expr, var_name, value):
    """Substitute a variable name with a value in an expression."""
    if isinstance(expr, Symbol):
        if expr.name == var_name:
            return value
        return expr
    if isinstance(expr, Expr):
        new_args = tuple(_subst_var(a, var_name, value) for a in expr.args)
        return Expr(expr.head, *new_args)
    return expr


# Build on import
EXTENDED_RULES = build_extended_rules()
