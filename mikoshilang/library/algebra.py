"""Polynomial and symbolic algebra (SymPy-backed, Wolfram conventions)."""

from __future__ import annotations
import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, S, R, L, numeric_out, sym, number, is_list, int_arg,
                   nonneg_int, list_arg, variable)

AREA = "algebra"


def _vars(v, fname):
    if is_list(v):
        return [variable(x, fname) for x in v.args]
    return [variable(v, fname)]


@fn("Simplify", area=AREA, status="fixed",
    notes="SymPy simplify; Simplify[e, assum] and Simplify[e, Assumptions -> a] (default $Assumptions) refine "
          "with the assumptions before and after simplifying (Simplify[Sqrt[x^2], x > 0] -> x); legacy rejected "
          "assumptions")
def _simplify(e, *assum):
    from .assume import split_option, effective, to_sym, from_sym
    rest, opt = split_option(assum, "Simplify")
    if len(rest) > 1:
        raise ValueError("Simplify[expr, assum] takes one assumption argument")
    A = effective(rest, opt, "Simplify")
    v = A.refine(to_sym(e, "Simplify"))
    if isinstance(v, sp.Basic) and not isinstance(v, sp.logic.boolalg.BooleanAtom):
        v = A.refine(sp.simplify(v))
    return from_sym(A.unapply(v))


@fn("Expand", area=AREA, status="fixed",
    notes="now expands products and integer powers only, like Wolfram (legacy also rewrote Exp[a + b] -> "
          "E^a E^b and expanded inside function arguments); Expand[e, patt] raises")
def _expand(e, *patt):
    if patt:
        raise NotImplementedError("Expand: the pattern argument is not supported")
    v = sym(e, "Expand")
    return R(sp.expand(v, deep=False, mul=True, multinomial=True, power_exp=False, power_base=False,
                       log=False, basic=False))


@fn("Factor", area=AREA, status="verified", notes="factors over the rationals (SymPy factor)")
def _factor(e):
    return R(sp.factor(sym(e, "Factor")))


@fn("Collect", area=AREA, status="verified", notes="Collect[e, x] and Collect[e, {x, y}]"
                                                   ". Check: exact (exact input gives exact results), reference "
                                                   "(differential cases in tests/library).")
def _collect(e: Expr, v: Var | ListOf[Var]) -> Expr:
    return R(sp.collect(sp.expand(sym(e, "Collect")), _vars(v, "Collect")))


@fn("Coefficient", area=AREA, status="fixed",
    notes="legacy used Poly.nth, failing for non-polynomial expressions and negative powers; now "
          "Coefficient[e, x] and Coefficient[e, x, n] on the expanded expression (n may be 0 or negative)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _coefficient(e: Expr, x: Expr, n: Integer = 1) -> Expr:
    v = sp.expand(sym(e, "Coefficient"))
    form = sym(x, "Coefficient")
    k = int_arg(n, "Coefficient", "power")
    if k == 0:
        return R(v.as_independent(form, as_Add=True)[0] if form.is_Symbol else v.coeff(form, 0))
    return R(sp.expand(v).coeff(form, k))


def _coeff_list(p, gens):
    if not gens:
        return R(p)
    x = gens[0]
    if p == 0:
        return L()
    poly = sp.Poly(p, x)
    coeffs = list(reversed(poly.all_coeffs()))
    return L(*(_coeff_list(sp.expand(c), gens[1:]) for c in coeffs))


@fn("CoefficientList", area=AREA, status="fixed",
    notes="returned coefficients from the highest power down; Wolfram lists them from x^0 upwards; "
          "nested lists for {x, y}"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _coefflist(e: Expr, v: Var | ListOf[Var]) -> Expr:
    p = sp.expand(sym(e, "CoefficientList"))
    gens = _vars(v, "CoefficientList")
    if not p.is_polynomial(*gens):
        raise ValueError("CoefficientList: expression is not a polynomial in the given variables")
    if len(gens) > 1:
        degs = [sp.degree(p, g) if p != 0 else 0 for g in gens]
        def rec(q, i):
            if i == len(gens):
                return R(q)
            cs = sp.Poly(q, gens[i]).all_coeffs()[::-1] if q != 0 else [0]
            cs = list(cs) + [0] * (degs[i] + 1 - len(cs))
            return L(*(rec(sp.expand(c), i + 1) for c in cs))
        return L() if p == 0 else rec(p, 0)
    return _coeff_list(p, gens)


@fn("Degree", area=AREA, status="fixed",
    notes="Wolfram constant Degree = Pi/180 (30 Degree -> Pi/6, Sin[30 Degree] -> 1/2, N[Degree] -> 0.0174533); "
          "Degree[...] used as a function raises ValueError pointing to Exponent; legacy Degree[p, x] returned the "
          "polynomial degree")
def _degree_call(*args: Expr) -> Expr:
    raise ValueError("Degree is the constant Pi/180, not a function; use Exponent[p, x] for the degree of a "
                     "polynomial")


@fn("Exponent", area=AREA, status="promoted", notes="Exponent[p, x]: highest power of x (Exponent[0, x] = -Infinity)")
def _exponent(e, x):
    p = sp.expand(sym(e, "Exponent"))
    v = variable(x, "Exponent")
    if p == 0:
        return R(-sp.oo)
    if not p.is_polynomial(v):
        terms = sp.Add.make_args(p)
        pows = []
        for t in terms:
            c, k = t.as_coeff_exponent(v)
            if c.has(v):
                raise ValueError("Exponent: expression is not a sum of powers of the variable")
            pows.append(k)
        return R(sp.Max(*pows))
    return R(sp.degree(p, v))


@fn("Discriminant", area=AREA, status="verified", notes="SymPy discriminant (same normalisation as Wolfram)"
                                                        ". Check: exact (exact input gives exact results), reference "
                                                        "(differential cases in tests/library).")
def _disc(p: Expr, x: Var) -> Expr:
    return R(sp.discriminant(sym(p, "Discriminant"), variable(x, "Discriminant")))


@fn("Resultant", area=AREA, status="verified", notes="Sylvester resultant (SymPy)"
                                                     ". Check: exact (exact input gives exact results), reference "
                                                     "(differential cases in tests/library).")
def _resultant(p: Expr, q: Expr, x: Var) -> Expr:
    return R(sp.resultant(sym(p, "Resultant"), sym(q, "Resultant"), variable(x, "Resultant")))


def _polydiv(p, q, x, fname):
    sp_, sq = sym(p, fname), sym(q, fname)
    v = variable(x, fname)
    if sp.expand(sq) == 0:
        raise ValueError(f"{fname}: division by the zero polynomial")
    return sp.div(sp_, sq, v)


@fn("PolynomialQuotient", area=AREA, status="verified", notes="quotient over the rationals (SymPy div)"
                                                              ". Check: exact (exact input gives exact results), "
                                                              "reference (differential cases in tests/library).")
def _pquo(p: Expr, q: Expr, x: Var) -> Expr:
    return R(sp.expand(_polydiv(p, q, x, "PolynomialQuotient")[0]))


@fn("PolynomialRemainder", area=AREA, status="verified", notes="remainder over the rationals (SymPy div)"
                                                               ". Check: exact (exact input gives exact results), "
                                                               "reference (differential cases in tests/library).")
def _prem(p: Expr, q: Expr, x: Var) -> Expr:
    return R(sp.expand(_polydiv(p, q, x, "PolynomialRemainder")[1]))


@fn("PolynomialDivision", area=AREA, status="verified", notes="{quotient, remainder} (MikoshiLang)"
                                                              ". Check: exact (exact input gives exact results), "
                                                              "reference (differential cases in tests/library).")
def _pdiv(p: Expr, q: Expr, x: Var) -> ListOf[Expr]:
    qq, rr = _polydiv(p, q, x, "PolynomialDivision")
    return L(R(sp.expand(qq)), R(sp.expand(rr)))


@fn("PartialFractions", area=AREA, status="verified", notes="MikoshiLang alias of Apart[e, x]"
                                                            ". Check: exact (exact input gives exact results), "
                                                            "reference (differential cases in tests/library).")
def _partial(e: Expr, x: Var) -> Expr:
    return R(sp.apart(sym(e, "PartialFractions"), variable(x, "PartialFractions")))


@fn("ComplexExpand", area=AREA, status="fixed",
    notes="legacy treated variables as complex (ComplexExpand[Re[x + I y]] stayed symbolic); Wolfram assumes "
          "all variables are real"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _complexexpand(e: Expr) -> Expr:
    v = sym(e, "ComplexExpand")
    reals = {s: sp.Symbol(s.name, real=True) for s in v.free_symbols}
    out = sp.expand_complex(v.xreplace(reals))
    return R(out.xreplace({r: s for s, r in reals.items()}))


@fn("PowerExpand", area=AREA, status="fixed",
    notes="legacy used powsimp (which combines powers, the opposite operation); now Wolfram PowerExpand: "
          "(a b)^c -> a^c b^c, (a^b)^c -> a^(b c), Log[a b] -> Log[a] + Log[b], Sqrt[x^2] -> x"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _powerexpand(e: Expr) -> Expr:
    v = sym(e, "PowerExpand")
    v = sp.expand_log(v, force=True)
    v = sp.powdenest(v, force=True)
    v = sp.expand_power_base(v, force=True)
    return R(v)


@fn("TrigExpand", area=AREA, status="verified", notes="SymPy expand_trig followed by Expand"
                                                      ". Check: exact (exact input gives exact results), reference "
                                                      "(differential cases in tests/library).")
def _trigexpand(e: Expr) -> Expr:
    return R(sp.expand(sp.expand_trig(sym(e, "TrigExpand"))))


@fn("TrigReduce", area=AREA, status="verified", notes="products and powers of trig functions to multiple angles (SymPy fu TR8)"
                                                      ". Check: exact (exact input gives exact results), reference "
                                                      "(differential cases in tests/library).")
def _trigreduce(e: Expr) -> Expr:
    from sympy.simplify.fu import TR8
    return R(sp.expand(TR8(sym(e, "TrigReduce"))))


@fn("TrigToExp", area=AREA, status="verified", notes="rewrite trig/hyperbolic functions as exponentials"
                                                     ". Check: exact (exact input gives exact results), reference "
                                                     "(differential cases in tests/library).")
def _trigtoexp(e: Expr) -> Expr:
    return R(sp.expand(sym(e, "TrigToExp").rewrite(sp.exp)))


@fn("ExpToTrig", area=AREA, status="verified", notes="rewrite exponentials as trig/hyperbolic functions"
                                                     ". Check: exact (exact input gives exact results), reference "
                                                     "(differential cases in tests/library).")
def _exptotrig(e: Expr) -> Expr:
    return R(sp.expand(sym(e, "ExpToTrig").rewrite(sp.cos)))


@fn("CompleteTheSquare", area=AREA, status="fixed",
    notes="a (x - h)^2 + k form (MikoshiLang); non-quadratics now raise instead of returning an Error expression"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _cts(e: Expr, x: Var) -> Expr:
    v = variable(x, "CompleteTheSquare")
    p = sp.expand(sym(e, "CompleteTheSquare"))
    if not p.is_polynomial(v) or sp.degree(p, v) != 2:
        raise ValueError("CompleteTheSquare: expression must be a quadratic in the variable")
    A, B, C = sp.Poly(p, v).all_coeffs()
    h = -B / (2 * A)
    k = sp.simplify(C - B ** 2 / (4 * A))
    return R(A * (v - sp.simplify(h)) ** 2 + k)


@fn("QuadraticFormula", area=AREA, status="fixed",
    notes="roots {(-b + Sqrt[b^2 - 4ac])/(2a), (-b - Sqrt[...])/(2a)} (MikoshiLang); a = 0 now raises "
          "instead of dividing by zero"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _quadformula(a: Numeric, b: Numeric, c: Numeric) -> ListOf[Expr]:
    sa, sb, sc = sym(a, "QuadraticFormula"), sym(b, "QuadraticFormula"), sym(c, "QuadraticFormula")
    if sa == 0:
        raise ValueError("QuadraticFormula: leading coefficient a must be non-zero")
    d = sp.sqrt(sb ** 2 - 4 * sa * sc)
    return L(numeric_out((-sb + d) / (2 * sa), a, b, c), numeric_out((-sb - d) / (2 * sa), a, b, c))


@fn("RationalizeDenominator", area=AREA, status="fixed",
    notes="legacy multiplied by conj(d)/conj(d) (a no-op) and only simplified; now SymPy radsimp, "
          "e.g. 1/(1 + Sqrt[2]) -> -1 + Sqrt[2]"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _ratdenom(e: Expr) -> Expr:
    return R(sp.radsimp(sym(e, "RationalizeDenominator")))


# Refine, Assuming and the Assumptions option live in assume.py
