"""Symbolic math operations — wrapping SymPy through MikoshiLang's expression system."""

from __future__ import annotations
from typing import Any, Optional
import sympy as sp
from sympy.core.function import AppliedUndef
from .expr import Expr, Symbol


# ── Conversion: MikoshiLang <-> SymPy ──

_CONSTANTS = {"Pi": sp.pi, "E": sp.E, "I": sp.I, "Infinity": sp.oo,
              "ComplexInfinity": sp.zoo, "Indeterminate": sp.nan, "GoldenRatio": sp.GoldenRatio,
              "EulerGamma": sp.EulerGamma, "Degree": sp.pi / 180, "Catalan": sp.Catalan}

_FUNCS = {
    "Sin": sp.sin, "Cos": sp.cos, "Tan": sp.tan, "Cot": sp.cot, "Sec": sp.sec, "Csc": sp.csc,
    "ArcSin": sp.asin, "ArcCos": sp.acos, "ArcCot": sp.acot,
    "Sinh": sp.sinh, "Cosh": sp.cosh, "Tanh": sp.tanh, "Coth": sp.coth,
    "ArcSinh": sp.asinh, "ArcCosh": sp.acosh, "ArcTanh": sp.atanh,
    "Exp": sp.exp, "Sqrt": sp.sqrt, "Abs": sp.Abs, "Sign": sp.sign,
    "Floor": sp.floor, "Ceiling": sp.ceiling, "Factorial": sp.factorial, "Gamma": sp.gamma,
    "Beta": sp.beta, "Zeta": sp.zeta, "Conjugate": sp.conjugate, "Re": sp.re, "Im": sp.im,
    "RealPart": sp.re, "ImaginaryPart": sp.im, "Binomial": sp.binomial, "Erf": sp.erf,
    "Max": sp.Max, "Min": sp.Min, "Mod": sp.Mod, "BesselJ": sp.besselj, "BesselY": sp.bessely,
    "LogGamma": sp.loggamma, "PolyGamma": sp.polygamma, "Fibonacci": sp.fibonacci,
}

_INV_FUNCS = {
    sp.sin: "Sin", sp.cos: "Cos", sp.tan: "Tan", sp.cot: "Cot", sp.sec: "Sec", sp.csc: "Csc",
    sp.asin: "ArcSin", sp.acos: "ArcCos", sp.atan: "ArcTan", sp.acot: "ArcCot", sp.atan2: "ArcTan",
    sp.sinh: "Sinh", sp.cosh: "Cosh", sp.tanh: "Tanh", sp.coth: "Coth",
    sp.asinh: "ArcSinh", sp.acosh: "ArcCosh", sp.atanh: "ArcTanh",
    sp.exp: "Exp", sp.log: "Log", sp.Abs: "Abs", sp.sign: "Sign",
    sp.floor: "Floor", sp.ceiling: "Ceiling", sp.factorial: "Factorial", sp.gamma: "Gamma",
    sp.beta: "Beta", sp.zeta: "Zeta", sp.conjugate: "Conjugate", sp.re: "Re", sp.im: "Im",
    sp.binomial: "Binomial", sp.erf: "Erf", sp.Max: "Max", sp.Min: "Min", sp.Mod: "Mod",
    sp.besselj: "BesselJ", sp.bessely: "BesselY", sp.loggamma: "LogGamma",
    sp.polygamma: "PolyGamma", sp.fibonacci: "Fibonacci", sp.polylog: "PolyLog",
}


# Additional special functions (Wolfram head -> SymPy constructor).
def _gamma_w(*a):
    if len(a) == 1:
        return sp.gamma(a[0])
    if len(a) == 2:
        return sp.uppergamma(a[0], a[1])
    return sp.uppergamma(a[0], a[1]) - sp.uppergamma(a[0], a[2])


def _legendre_w(*a):
    return sp.legendre(*a) if len(a) == 2 else sp.assoc_legendre(*a)


def _laguerre_w(*a):
    return sp.laguerre(*a) if len(a) == 2 else sp.assoc_laguerre(*a)


def _polygamma_w(*a):
    return sp.polygamma(0, a[0]) if len(a) == 1 else sp.polygamma(*a)


def _hyper_w(p, q):
    def make(*a):
        return sp.hyper(list(a[:p]), list(a[p:p + q]), a[p + q])
    return make


def _elliptic_e_w(*a):
    return sp.elliptic_e(*a)


def _elliptic_pi_w(*a):
    return sp.elliptic_pi(*a)


_FUNCS.update({
    "Gamma": _gamma_w, "LegendreP": _legendre_w, "LaguerreL": _laguerre_w, "PolyGamma": _polygamma_w,
    "BetaRegularized": lambda *a: (sp.betainc_regularized(a[1], a[2], 0, a[0]) if len(a) == 3
                                   else sp.betainc_regularized(a[2], a[3], a[0], a[1])),
    "Erfc": sp.erfc, "Erfi": sp.erfi, "InverseErf": sp.erfinv, "InverseErfc": sp.erfcinv, "ExpIntegralEi": sp.Ei, "ExpIntegralE": sp.expint,
    "LogIntegral": sp.li, "SinIntegral": sp.Si, "CosIntegral": sp.Ci, "SinhIntegral": sp.Shi,
    "CoshIntegral": sp.Chi, "BesselI": sp.besseli, "BesselK": sp.besselk, "HankelH1": sp.hankel1,
    "HankelH2": sp.hankel2, "AiryAi": sp.airyai, "AiryBi": sp.airybi, "AiryAiPrime": sp.airyaiprime,
    "AiryBiPrime": sp.airybiprime, "SphericalBesselJ": sp.jn, "SphericalBesselY": sp.yn,
    "ChebyshevT": sp.chebyshevt, "ChebyshevU": sp.chebyshevu, "HermiteH": sp.hermite,
    "GegenbauerC": sp.gegenbauer, "JacobiP": sp.jacobi, "SphericalHarmonicY": sp.Ynm,
    "EllipticK": sp.elliptic_k, "EllipticF": sp.elliptic_f, "EllipticE": _elliptic_e_w,
    "EllipticPi": _elliptic_pi_w, "ProductLog": lambda *a: sp.LambertW(a[-1], a[0]) if len(a) == 2 else sp.LambertW(a[0]),
    "LambertW": sp.LambertW, "DirichletEta": sp.dirichlet_eta, "HarmonicNumber": sp.harmonic,
    "Hypergeometric0F1": _hyper_w(0, 1), "Hypergeometric1F1": _hyper_w(1, 1),
    "Hypergeometric2F1": _hyper_w(2, 1), "KroneckerDelta": sp.KroneckerDelta,
    "DiracDelta": sp.DiracDelta, "HeavisideTheta": sp.Heaviside, "Pochhammer": sp.RisingFactorial,
    "Factorial2": sp.factorial2, "Subfactorial": sp.subfactorial, "CatalanNumber": sp.catalan,
    "BellB": sp.bell, "BernoulliB": sp.bernoulli, "EulerE": sp.euler, "LucasL": sp.lucas,
    "EulerPhi": sp.totient, "PrimePi": sp.primepi, "StruveH": sp.Function("StruveH"),
    "Arg": sp.arg, "Csch": sp.csch, "Sech": sp.sech, "ArcSec": sp.asec, "ArcCsc": sp.acsc,
    "ArcSech": sp.asech, "ArcCsch": sp.acsch, "ArcCoth": sp.acoth,
})

# Heads whose contents must never be handed to SymPy (holding / structural / procedural forms).
NON_ALGEBRAIC_HEADS = frozenset({
    "Hold", "HoldForm", "HoldComplete", "HoldPattern", "Function", "Pattern", "Blank",
    "BlankSequence", "BlankNullSequence", "Condition", "PatternTest", "Alternatives", "Rule",
    "RuleDelayed", "Set", "SetDelayed", "CompoundExpression", "Module", "Block", "With", "If",
    "Which", "Switch", "Do", "While", "For", "Table", "Slot", "SlotSequence", "Sequence",
    "ReplaceAll", "ReplaceRepeated", "Part", "String", "Association", "Graphics", "Error",
    "InputForm", "FullForm", "Solve", "DSolve", "Series", "Take", "Drop", "Select", "Map", "Apply",
    "Matrix",
})


_ROOT_VAR = sp.Symbol("RootVar$")


def root_values(poly_expr, x):
    """(order, values): SymPy CRootOf indices in Wolfram Root order, and a complex approximation of the root
    with each SymPy index. Wolfram order: real roots ascending, then complex roots by increasing real part, then
    increasing |Im|, conjugates adjacent with the negative imaginary part first. (SymPy orders complex roots by
    its isolating intervals, which differs, e.g. for x^7 + 3x + 1.)"""
    key = (sp.srepr(sp.Poly(poly_expr, x).monic().as_expr()), str(x))
    hit = _ROOT_ORDER_CACHE.get(key)
    if hit is not None:
        return hit
    p = sp.Poly(poly_expr, x)
    sqf = sp.Poly(sp.quo(p, sp.gcd(p, p.diff(x))), x)       # distinct roots
    nums = [complex(r) for r in sqf.nroots(n=30, maxsteps=500)]
    nreal = len(p.real_roots())       # with multiplicity; CRootOf lists these first, ascending
    cplx = [r for r in nums if abs(r.imag) >= 1e-20]
    values = {}
    for i in range(p.degree()):
        c = sp.CRootOf(poly_expr, x, i)
        if not isinstance(c, sp.CRootOf):          # SymPy evaluated it (rational / repeated root)
            values[i] = complex(c)
        elif i < nreal:
            values[i] = complex(c.evalf(20))
        else:
            iv = c._get_interval()
            inside = [r for r in cplx if float(iv.ax) <= r.real <= float(iv.bx) and float(iv.ay) <= r.imag <= float(iv.by)]
            values[i] = inside[0] if inside else complex(c.evalf(20))
    comp = sorted(range(nreal, p.degree()),
                  key=lambda i: (round(values[i].real, 12), round(abs(values[i].imag), 12), values[i].imag))
    out = (list(range(nreal)) + comp, values)
    _ROOT_ORDER_CACHE[key] = out
    return out


def wolfram_root_order(poly_expr, x):
    return root_values(poly_expr, x)[0]


_ROOT_ORDER_CACHE = {}


def root_to_sympy(expr):
    """Root[f, k] (f a pure function of one slot) -> SymPy CRootOf(poly, k - 1)."""
    from .kernel import replace_slots
    if len(expr.args) != 2 or not (isinstance(expr.args[0], Expr) and expr.args[0].head == "Function"
                                   and len(expr.args[0].args) == 1):
        raise ValueError("Root[f, k] needs a pure function f such as #^3 - 2 &")
    body = to_sympy(replace_slots(expr.args[0].args[0], (Symbol(_ROOT_VAR.name),)))
    k = to_sympy(expr.args[1])
    if not (k.is_Integer and k >= 1):
        raise ValueError("Root[f, k]: k must be a positive integer")
    if not body.is_polynomial(_ROOT_VAR) or body.free_symbols - {_ROOT_VAR}:
        raise ValueError("Root[f, k]: f[x] must be a polynomial in x with numeric coefficients")
    n = int(sp.degree(body, _ROOT_VAR))
    if int(k) > n:
        raise ValueError(f"Root[f, k]: k = {k} exceeds the degree {n}")
    return sp.CRootOf(body, _ROOT_VAR, wolfram_root_order(body, _ROOT_VAR)[int(k) - 1])


def _derivative_to_sympy(head, args):
    """Derivative[n1, n2, ...][f][a1, a2, ...] -> SymPy Derivative / Subs."""
    orders = [int(n) for n in head.head.args]
    f = head.args[0] if head.args else None
    if not isinstance(f, Symbol) or len(orders) != len(args):
        raise ValueError("Unsupported Derivative form")
    sargs = [to_sympy(a) for a in args]
    func = sp.Function(f.name)
    if all(isinstance(a, sp.Symbol) for a in sargs) and len(set(sargs)) == len(sargs):
        spec = [(a, n) for a, n in zip(sargs, orders) if n]
        return sp.Derivative(func(*sargs), *spec) if spec else func(*sargs)
    dummies = [sp.Dummy(f"t{i}") for i in range(len(sargs))]
    spec = [(d, n) for d, n in zip(dummies, orders) if n]
    return sp.Subs(sp.Derivative(func(*dummies), *spec), tuple(dummies), tuple(sargs))


def _order_to_sympy(arg):
    syms = sorted(arg.free_symbols, key=str)
    if len(syms) != 1:
        raise ValueError("O[...] needs exactly one variable")
    s = syms[0]
    if arg == s:
        return sp.Order(s)
    point = sp.solve(arg, s)
    if len(point) != 1:
        raise ValueError("Unsupported O[...] argument")
    return sp.Order(arg, (s, point[0]))


def _list_as_function(x: object) -> object:
    """A List argument of a function head (a Python list after conversion) as the undefined SymPy function
    List(...): SymPy functions cannot hold Python lists (f([a]) fails inside SymPy), and from_sympy turns the
    undefined function List back into {...}, so f[{a}] round-trips unchanged."""
    if isinstance(x, list):
        return sp.Function("List")(*(_list_as_function(a) for a in x))
    return x


def _is_list_function(f: object) -> bool:
    return isinstance(f, AppliedUndef) and type(f).__name__ == "List"


_LIST_DIFF_MSG = "differentiating through a list argument of a function (as in f[{x}]) is not supported"


def to_sympy(expr) -> Any:
    """Convert a MikoshiLang expression to a SymPy expression."""
    if isinstance(expr, str):
        raise TypeError("Strings are not symbolic expressions; parse source explicitly")
    if isinstance(expr, bool):
        return sp.true if expr else sp.false
    if isinstance(expr, (int, float)):
        return sp.sympify(expr)
    if isinstance(expr, complex):
        return sp.sympify(expr)
    if isinstance(expr, Symbol):
        if expr.name in _CONSTANTS:
            return _CONSTANTS[expr.name]
        return sp.Symbol(expr.name)
    if isinstance(expr, Expr):
        head = expr.head
        if isinstance(head, Expr):
            if isinstance(head.head, Expr) and head.head.head == "Derivative":
                return _derivative_to_sympy(head, expr.args)
            raise ValueError("Cannot convert a compound head to SymPy")
        if head in NON_ALGEBRAIC_HEADS:
            raise ValueError(f"Cannot convert {head} to SymPy")
        if head == "Root":
            return root_to_sympy(expr)
        if head == "List":
            return [to_sympy(a) for a in expr.args]
        if head == "Piecewise" and expr.args and isinstance(expr.args[0], Expr) and expr.args[0].head == "List":
            pairs = []
            for pc in expr.args[0].args:
                if not (isinstance(pc, Expr) and pc.head == "List" and len(pc.args) == 2):
                    raise ValueError("Piecewise expects {{value, condition}, ...}")
                pairs.append((to_sympy(pc.args[0]), to_sympy(pc.args[1])))
            default = to_sympy(expr.args[1]) if len(expr.args) > 1 else sp.Integer(0)
            return sp.Piecewise(*pairs, (default, True))
        if head == "Rational" and len(expr.args) == 2:
            return sp.Rational(to_sympy(expr.args[0]), to_sympy(expr.args[1]))
        args = [to_sympy(a) for a in expr.args]
        if head == "O" and len(args) == 1:
            return _order_to_sympy(args[0])
        _map = {
            "Equal": lambda a: sp.Eq(*a, evaluate=False),
            "Unequal": lambda a: sp.Ne(*a, evaluate=False),
            "Plus": lambda a: sp.Add(*a),
            "Times": lambda a: sp.Mul(*a),
            "Power": lambda a: a[0] ** a[1],
            "Log": lambda a: sp.log(a[0]) if len(a) == 1 else sp.log(a[1], a[0]),
            "ArcTan": lambda a: sp.atan(a[0]) if len(a) == 1 else sp.atan2(a[1], a[0]),
        }
        if head in _map:
            return _map[head](args)
        if head in _FUNCS:
            return _FUNCS[head](*args)
        candidate = getattr(sp, head.lower(), None)
        if isinstance(candidate, type) and issubclass(candidate, sp.Function):
            return candidate(*(_list_as_function(a) for a in args))
        # Unknown heads are undefined functions: f[x] -> f(x)
        return sp.Function(head)(*(_list_as_function(a) for a in args))
    if isinstance(expr, (sp.Basic, sp.MatrixBase)):
        return sp.sympify(expr)
    raise TypeError(f"Unsupported symbolic value type: {type(expr).__name__}")


def _order_from_sympy(o):
    e = o.expr
    point = o.point[0] if o.point else 0
    var = o.variables[0] if o.variables else None
    if var is None:
        return Expr("O", from_sympy(e))
    base = var - point if point != 0 else var
    k = sp.degree(sp.expand(e.subs(var, sp.Symbol("_u") + point)), sp.Symbol("_u")) if e.is_polynomial(var) else None
    if k is not None and sp.expand(e.subs(var, sp.Symbol("_u") + point)) == sp.Symbol("_u") ** k:
        inner = Expr("O", from_sympy(base))
        return inner if k == 1 else Expr("Power", inner, int(k))
    return Expr("O", from_sympy(e))



_INV_FUNCS.update({
    sp.erfc: "Erfc", sp.erfi: "Erfi", sp.erfinv: "InverseErf", sp.erfcinv: "InverseErfc", sp.Ei: "ExpIntegralEi", sp.expint: "ExpIntegralE",
    sp.li: "LogIntegral", sp.Si: "SinIntegral", sp.Ci: "CosIntegral", sp.Shi: "SinhIntegral",
    sp.Chi: "CoshIntegral", sp.besseli: "BesselI", sp.besselk: "BesselK", sp.hankel1: "HankelH1",
    sp.hankel2: "HankelH2", sp.airyai: "AiryAi", sp.airybi: "AiryBi", sp.airyaiprime: "AiryAiPrime",
    sp.airybiprime: "AiryBiPrime", sp.jn: "SphericalBesselJ", sp.yn: "SphericalBesselY",
    sp.chebyshevt: "ChebyshevT", sp.chebyshevu: "ChebyshevU", sp.hermite: "HermiteH",
    sp.gegenbauer: "GegenbauerC", sp.jacobi: "JacobiP", sp.Ynm: "SphericalHarmonicY",
    sp.elliptic_k: "EllipticK", sp.elliptic_f: "EllipticF", sp.elliptic_e: "EllipticE",
    sp.elliptic_pi: "EllipticPi", sp.dirichlet_eta: "DirichletEta", sp.harmonic: "HarmonicNumber",
    sp.legendre: "LegendreP", sp.assoc_legendre: "LegendreP", sp.laguerre: "LaguerreL",
    sp.assoc_laguerre: "LaguerreL", sp.uppergamma: "Gamma", sp.KroneckerDelta: "KroneckerDelta",
    sp.DiracDelta: "DiracDelta", sp.Heaviside: "HeavisideTheta", sp.RisingFactorial: "Pochhammer",
    sp.factorial2: "Factorial2", sp.subfactorial: "Subfactorial", sp.catalan: "CatalanNumber",
    sp.bell: "BellB", sp.bernoulli: "BernoulliB", sp.euler: "EulerE", sp.lucas: "LucasL",
    sp.totient: "EulerPhi", sp.primepi: "PrimePi", sp.arg: "Arg", sp.csch: "Csch", sp.sech: "Sech",
    sp.asec: "ArcSec", sp.acsc: "ArcCsc", sp.asech: "ArcSech", sp.acsch: "ArcCsch", sp.acoth: "ArcCoth",
    sp.loggamma: "LogGamma",
})


def _special_from_sympy(sexpr):
    """Structured conversions that are not a plain head rename (None if not applicable)."""
    if isinstance(sexpr, sp.Heaviside):          # batch 4: Heaviside(x, H0) -> HeavisideTheta[x]
        return Expr("HeavisideTheta", from_sympy(sexpr.args[0]))
    if isinstance(sexpr, sp.DiracDelta) and len(sexpr.args) == 2 and sexpr.args[1] != 0:
        return Expr(Expr(Expr("Derivative", int(sexpr.args[1])), Symbol("DiracDelta")), from_sympy(sexpr.args[0]))
    if isinstance(sexpr, sp.LambertW):
        a = [from_sympy(x) for x in sexpr.args]
        return Expr("ProductLog", a[0]) if len(a) == 1 else Expr("ProductLog", a[1], a[0])
    if isinstance(sexpr, sp.lowergamma):
        a, z = (from_sympy(x) for x in sexpr.args)
        return Expr("Gamma", a, 0, z)
    if isinstance(sexpr, sp.hyper):
        ap = [from_sympy(x) for x in sexpr.ap]
        bq = [from_sympy(x) for x in sexpr.bq]
        z = from_sympy(sexpr.argument)
        name = {(0, 1): "Hypergeometric0F1", (1, 1): "Hypergeometric1F1", (2, 1): "Hypergeometric2F1"}.get(
            (len(ap), len(bq)))
        if name:
            return Expr(name, *ap, *bq, z)
        return Expr("HypergeometricPFQ", Expr("List", *ap), Expr("List", *bq), z)
    if isinstance(sexpr, sp.betainc_regularized):
        a, b, x1, x2 = (from_sympy(x) for x in sexpr.args)
        if sexpr.args[2] == 0:
            return Expr("BetaRegularized", x2, a, b)          # Wolfram BetaRegularized[z, a, b]
        return Expr("BetaRegularized", x1, x2, a, b)
    if isinstance(sexpr, sp.polygamma) and sexpr.args[0] == 0:
        return Expr("PolyGamma", from_sympy(sexpr.args[1]))
    if isinstance(sexpr, sp.Piecewise):
        pieces, default = [], None
        for e, c in sexpr.args:
            if c is sp.true:
                default = from_sympy(e)
            else:
                pieces.append(Expr("List", from_sympy(e), from_sympy(c)))
        return Expr("Piecewise", Expr("List", *pieces), *([default] if default is not None else []))
    if isinstance(sexpr, sp.Integral):
        f = from_sympy(sexpr.function)
        lims = []
        for lim in sexpr.limits:
            if len(lim) == 1:
                lims.append(from_sympy(lim[0]))
            else:
                lims.append(Expr("List", *(from_sympy(x) for x in lim)))
        return Expr("Integrate", f, *lims)
    if isinstance(sexpr, sp.Sum):
        return Expr("Sum", from_sympy(sexpr.function),
                    *(Expr("List", *(from_sympy(x) for x in lim)) for lim in sexpr.limits))
    if isinstance(sexpr, sp.Product):
        return Expr("Product", from_sympy(sexpr.function),
                    *(Expr("List", *(from_sympy(x) for x in lim)) for lim in sexpr.limits))
    return None


def from_sympy(sexpr) -> Any:
    """Convert a SymPy expression back to MikoshiLang."""
    if isinstance(sexpr, bool):
        return sexpr
    if sexpr is sp.true:
        return True
    if sexpr is sp.false:
        return False
    if sexpr is sp.pi:
        return Symbol("Pi")
    if sexpr is sp.E:
        return Symbol("E")
    if sexpr is sp.I:
        return Symbol("I")
    if sexpr is sp.oo:
        return Symbol("Infinity")
    if sexpr is sp.zoo:
        return Symbol("ComplexInfinity")
    if sexpr is sp.S.NegativeInfinity:
        return Expr("Times", -1, Symbol("Infinity"))
    if sexpr is sp.nan:
        return Symbol("Indeterminate")
    if sexpr is sp.GoldenRatio:
        return Symbol("GoldenRatio")
    if sexpr is sp.EulerGamma:
        return Symbol("EulerGamma")
    if sexpr is sp.Catalan:
        return Symbol("Catalan")
    if isinstance(sexpr, sp.Subs) and isinstance(sexpr.expr, sp.Derivative):
        d = sexpr.expr
        fexpr = d.expr
        if _is_list_function(fexpr):
            raise NotImplementedError(_LIST_DIFF_MSG)
        if isinstance(fexpr, sp.core.function.AppliedUndef) and tuple(fexpr.args) == tuple(sexpr.variables):
            counts = dict(d.variable_count)
            orders = [int(counts.get(v, 0)) for v in fexpr.args]
            pts = dict(zip(sexpr.variables, sexpr.point))
            return Expr(Expr(Expr("Derivative", *orders), Symbol(str(fexpr.func))),
                        *(from_sympy(pts[v]) for v in fexpr.args))
    if isinstance(sexpr, sp.CRootOf):
        x = sexpr.poly.gen
        body = from_sympy(sexpr.poly.as_expr().subs(x, _ROOT_VAR))
        from .kernel import substitute
        order = wolfram_root_order(sexpr.poly.as_expr(), x)
        return Expr("Root", Expr("Function", substitute(body, {_ROOT_VAR.name: Expr("Slot", 1)})),
                    order.index(int(sexpr.index)) + 1)
    if isinstance(sexpr, (list, tuple, sp.Tuple)):
        return Expr("List", *(from_sympy(a) for a in sexpr))
    if isinstance(sexpr, sp.MatrixBase):
        return Expr("List", *(Expr("List", *(from_sympy(sexpr[i, j]) for j in range(sexpr.cols)))
                              for i in range(sexpr.rows)))
    if isinstance(sexpr, sp.Integer):
        return int(sexpr)
    if isinstance(sexpr, sp.Rational):
        if sexpr.q == 1:
            return int(sexpr.p)
        return Expr("Times", int(sexpr.p), Expr("Power", int(sexpr.q), -1))
    if isinstance(sexpr, sp.Float):
        # machine precision -> Python float; arbitrary precision stays a SymPy Float
        return float(sexpr) if sexpr._prec <= 53 else sexpr
    if isinstance(sexpr, sp.Symbol):
        return Symbol(str(sexpr))
    if isinstance(sexpr, sp.Order):
        return _order_from_sympy(sexpr)
    if isinstance(sexpr, sp.Add):
        terms = [a for a in sexpr.args if not isinstance(a, sp.Order)]
        orders = [a for a in sexpr.args if isinstance(a, sp.Order)]
        if orders and terms and orders[0].variables:
            var = orders[0].variables[0]
            terms = sorted(terms, key=lambda t: (sp.degree(t, var) if t.is_polynomial(var) else 0,
                                                 sp.default_sort_key(t)))
        else:
            # Close to Wolfram's canonical order: 1 + 3*x + 3*x^2 + x^3
            terms = sorted(terms, key=sp.default_sort_key)
        args = [from_sympy(a) for a in terms + orders]
        return args[0] if len(args) == 1 else Expr("Plus", *args)
    if isinstance(sexpr, sp.Mul):
        args = [from_sympy(a) for a in sexpr.args]
        return args[0] if len(args) == 1 else Expr("Times", *args)
    if isinstance(sexpr, sp.Pow):
        return Expr("Power", from_sympy(sexpr.base), from_sympy(sexpr.exp))
    if isinstance(sexpr, sp.Derivative):
        f = sexpr.expr
        if _is_list_function(f):
            raise NotImplementedError(_LIST_DIFF_MSG)
        if isinstance(f, AppliedUndef) and all(isinstance(a, sp.Symbol) for a in f.args):
            counts = dict(sexpr.variable_count)
            orders = [int(counts.get(a, 0)) for a in f.args]
            if sum(orders) == sum(int(c) for c in counts.values()):
                head = Expr(Expr("Derivative", *orders), Symbol(type(f).__name__))
                return Expr(head, *(from_sympy(a) for a in f.args))
        spec = [Expr("List", from_sympy(v), int(c)) if c != 1 else from_sympy(v) for v, c in sexpr.variable_count]
        return Expr("D", from_sympy(f), *spec)
    if isinstance(sexpr, AppliedUndef):
        return Expr(type(sexpr).__name__, *(from_sympy(a) for a in sexpr.args))
    if isinstance(sexpr, sp.Equality):
        return Expr("Equal", from_sympy(sexpr.lhs), from_sympy(sexpr.rhs))
    _rel = {sp.Unequality: "Unequal", sp.StrictLessThan: "Less", sp.StrictGreaterThan: "Greater",
            sp.LessThan: "LessEqual", sp.GreaterThan: "GreaterEqual"}
    if type(sexpr) in _rel:
        return Expr(_rel[type(sexpr)], from_sympy(sexpr.lhs), from_sympy(sexpr.rhs))
    if isinstance(sexpr, sp.And):
        return Expr("And", *(from_sympy(a) for a in sexpr.args))
    if isinstance(sexpr, sp.Or):
        return Expr("Or", *(from_sympy(a) for a in sexpr.args))
    if isinstance(sexpr, sp.Not):
        return Expr("Not", from_sympy(sexpr.args[0]))
    if isinstance(sexpr, sp.exp):
        return Expr("Power", Symbol("E"), from_sympy(sexpr.args[0]))
    if isinstance(sexpr, sp.log):
        if len(sexpr.args) == 2:
            return Expr("Log", from_sympy(sexpr.args[1]), from_sympy(sexpr.args[0]))
        return Expr("Log", from_sympy(sexpr.args[0]))
    if isinstance(sexpr, sp.atan2):
        return Expr("ArcTan", from_sympy(sexpr.args[1]), from_sympy(sexpr.args[0]))
    special = _special_from_sympy(sexpr)
    if special is not None:
        return special
    if type(sexpr) in _INV_FUNCS:
        return Expr(_INV_FUNCS[type(sexpr)], *(from_sympy(a) for a in sexpr.args))
    if isinstance(sexpr, sp.Number):
        return float(sexpr)
    if isinstance(sexpr, sp.NumberSymbol) and getattr(sexpr, "_wl_name", ""):
        return Symbol(sexpr._wl_name)    # library constants (Glaisher, Khinchin, ...)
    # Fallback
    if hasattr(sexpr, 'args') and hasattr(sexpr, 'func'):
        fname = type(sexpr).__name__
        args = [from_sympy(a) for a in sexpr.args]
        return Expr(fname, *args)
    return str(sexpr)


# ── High-level operations ──

def simplify(expr) -> Any:
    """Algebraic simplification."""
    return from_sympy(sp.simplify(to_sympy(expr)))


def expand(expr) -> Any:
    """Expand products and powers."""
    return from_sympy(sp.expand(to_sympy(expr)))


def factor(expr) -> Any:
    """Factor an expression."""
    return from_sympy(sp.factor(to_sympy(expr)))


def diff(expr, var, n: int = 1) -> Any:
    """Differentiate expr with respect to var, n times."""
    sv = to_sympy(var) if isinstance(var, Symbol) else sp.Symbol(str(var))
    return from_sympy(sp.diff(to_sympy(expr), sv, n))


def integrate(expr, var, *limits) -> Any:
    """Integrate expr with respect to var. Optional limits for definite integrals."""
    sv = to_sympy(var) if isinstance(var, Symbol) else sp.Symbol(str(var))
    if limits:
        return from_sympy(sp.integrate(to_sympy(expr), (sv, *[to_sympy(l) for l in limits])))
    return from_sympy(sp.integrate(to_sympy(expr), sv))


def solve(expr, var) -> Any:
    """Solve expr == 0 for var."""
    sv = to_sympy(var) if isinstance(var, Symbol) else sp.Symbol(str(var))
    solutions = sp.solve(to_sympy(expr), sv)
    return Expr("List", *(from_sympy(s) for s in solutions))


def limit(expr, var, point) -> Any:
    """Compute limit of expr as var -> point."""
    sv = to_sympy(var) if isinstance(var, Symbol) else sp.Symbol(str(var))
    return from_sympy(sp.limit(to_sympy(expr), sv, to_sympy(point)))


def series(expr, var, point=0, order=6) -> Any:
    """Taylor polynomial through order ``order`` inclusive (Wolfram's Normal[Series[...]]).

    Unlike ``sympy.series`` (whose ``n`` is exclusive) the x^order term is included.
    Use the ``Series`` language function to keep the O[x]^(n+1) term.
    """
    sv = to_sympy(var) if isinstance(var, Symbol) else sp.Symbol(str(var))
    # Wolfram semantics: include terms up to and including order n, plus O[x]^(n+1)
    s = sp.series(to_sympy(expr), sv, to_sympy(point), order + 1)
    return from_sympy(s.removeO())
